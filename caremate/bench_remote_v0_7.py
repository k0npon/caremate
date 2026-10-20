"""Local vs AWS perception: identical answers? how much latency does the cloud add?

Runs the same frames (with their saved camera poses -> workspace ROI) through
the local OpenCV 5 backend and through the remote service, and reports
agreement and timing (encode, network/queue, server inference).

  python -m caremate.bench_remote_v0_7 --backend-url http://HOST:8000 \\
      --frames "outputs/v07_*/sweep_*/rgb/*.png" --targets crunch_peanut reading_book
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import cv2
import numpy as np

from caremate.perception_v0_7 import PerceptionBackend
from caremate.remote_backend import RemotePerceptionBackend
from caremate.workspace_v0_7 import workspace_polygon

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend-url", required=True)
    p.add_argument("--frames", default="outputs/v07_*/sweep_*/rgb/*.png")
    p.add_argument("--targets", nargs="+", default=["crunch_peanut", "reading_book"])
    p.add_argument("--aliked", type=Path, default=ROOT / "models/vision/aliked-n16rot-top1k-640.onnx")
    p.add_argument("--refs", type=Path, default=ROOT / "references/v0_7")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/bench_remote_v0_7.json")
    a = p.parse_args()
    files = sorted(glob.glob(a.frames))
    if not files:
        p.error(f"no frames match {a.frames}")
    local = PerceptionBackend(a.aliked, a.refs)
    local.warmup()
    remote = RemotePerceptionBackend(a.backend_url)
    rows = []
    for f in files:
        img = cv2.imread(f)
        pf = Path(f).with_suffix(".pose.json")
        roi = workspace_polygon(json.loads(pf.read_text()), img.shape[1], img.shape[0]) if pf.is_file() else None
        for t in a.targets:
            t0 = time.perf_counter(); rl = local.perceive(img, t, roi_polygon=roi); tl = time.perf_counter() - t0
            t0 = time.perf_counter(); rr = remote.perceive(img, t, roi_polygon=roi); tr = time.perf_counter() - t0
            same = (rl.status == rr.status and rl.n_inliers == rr.n_inliers)
            rows.append({"frame": Path(f).name, "target": t, "local_status": rl.status, "remote_status": rr.status,
                         "identical": same, "local_ms": round(1000 * tl, 1), "remote_total_ms": round(1000 * tr, 1),
                         **remote.last_timing})
            print(f"{Path(f).name[:28]:28s} {t:15s} local={rl.status:9s} remote={rr.status:9s} "
                  f"same={same!s:5s} local={1000*tl:6.0f}ms remote={1000*tr:6.0f}ms "
                  f"(server {remote.last_timing.get('server_inference_ms')} ms, "
                  f"net {remote.last_timing.get('network_and_queue_ms')} ms)")
    med = lambda k: float(np.median([r[k] for r in rows if r.get(k) is not None]))
    summary = {"frames": len(files), "requests": len(rows),
               "identical_results": f"{sum(r['identical'] for r in rows)}/{len(rows)}",
               "median_local_ms": med("local_ms"), "median_remote_total_ms": med("remote_total_ms"),
               "median_server_inference_ms": med("server_inference_ms"),
               "median_network_and_queue_ms": med("network_and_queue_ms"),
               "median_upload_bytes": med("upload_bytes"), "endpoint": a.backend_url}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    remote.close(); local.close()


if __name__ == "__main__":
    main()
