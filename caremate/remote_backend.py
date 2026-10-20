"""Client for the CareMate AWS perception service, with the same interface as
caremate.perception_v0_7.PerceptionBackend (perceive / warmup / close / method).

The agent keeps planning and robot control local and sends each fresh frame
(lossless PNG, so features are identical to the local run) to the endpoint.

  python -m caremate.agent_loop_v0_7 --target crunch_peanut --backend-url http://<host>:8000
"""
from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np
import requests

from caremate.perception_v0_7 import PerceptionResult

_FIELDS = set(PerceptionResult.__dataclass_fields__)


class RemotePerceptionBackend:
    def __init__(self, url: str, token: str | None = None, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.token = token or os.environ.get("CAREMATE_API_TOKEN", "")
        self.timeout = timeout
        self.session = requests.Session()
        info = self.session.get(f"{self.url}/v1/health", timeout=timeout).json()
        self.method = f"REMOTE[{self.url}] {info.get('method', '?')}"
        self.refs = {n: None for n in info.get("references", [])}
        self.last_timing: dict = {}

    def warmup(self) -> float:
        return 0.0      # the service warms up at startup

    def perceive(self, bgr: np.ndarray, target: str, camera_pose: dict | None = None,
                 roi_polygon: np.ndarray | None = None) -> PerceptionResult:
        from caremate.perception_v0_7 import image_id
        t0 = time.perf_counter()
        full_id = image_id(bgr)
        H, W = bgr.shape[:2]
        form = {"target": target}
        send = bgr
        if roi_polygon is not None:
            # Upload only the workspace region (same crop rule as the backend), lossless.
            # Results come back in full-frame coordinates, identical to a local call.
            poly = np.asarray(roi_polygon, np.float32)
            x0, y0 = np.floor(poly.min(0)).astype(int).clip(0)
            x1, y1 = np.ceil(poly.max(0)).astype(int)
            x1, y1 = min(W, x1 + 1), min(H, y1 + 1)
            send = bgr[y0:y1, x0:x1]
            form["roi_polygon"] = json.dumps((poly - [x0, y0]).tolist())
            form["origin"] = json.dumps([int(x0), int(y0)])
            form["frame_wh"] = json.dumps([W, H])
        ok, png = cv2.imencode(".png", send, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise RuntimeError("PNG encode failed")
        if camera_pose is not None:
            form["camera_pose"] = json.dumps(camera_pose, default=float)
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        t1 = time.perf_counter()
        try:
            r = self.session.post(f"{self.url}/v1/perceive", data=form, headers=headers,
                                  files={"image": ("frame.png", png.tobytes(), "image/png")},
                                  timeout=self.timeout)
            r.raise_for_status()
            d = r.json()
        except (requests.RequestException, ValueError) as exc:
            # Network failure is a perception error: the agent must not assume recognition.
            return PerceptionResult(target, "error", full_id, self.method, camera_pose,
                                    error=f"remote: {type(exc).__name__}: {exc}")
        t2 = time.perf_counter()
        server = d.pop("server", {})
        res = PerceptionResult(**{k: v for k, v in d.items() if k in _FIELDS})
        res.image_id = full_id          # stale-evidence guard works on the full frame
        self.last_timing = {"encode_ms": round(1000 * (t1 - t0), 1),
                            "round_trip_ms": round(1000 * (t2 - t1), 1),
                            "server_inference_ms": server.get("inference_ms"),
                            "network_and_queue_ms": round(1000 * (t2 - t1) - (server.get("inference_ms") or 0)
                                                          - (server.get("decode_ms") or 0), 1),
                            "upload_bytes": len(png)}
        res.latency_ms = {**(res.latency_ms or {}), **{f"remote_{k}": v for k, v in self.last_timing.items()}}
        return res

    def close(self):
        self.session.close()
