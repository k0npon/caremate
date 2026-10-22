"""Render the demo head-camera frames that the web page offers as one-click samples.

Writes samples/v0_7/<id>.png (raw head-camera frame, lossless) and samples/v0_7/samples.json
(title, caption, target, workspace ROI polygon from camera forward kinematics, and the result the
LOCAL OpenCV 5 backend gives, so the deployed service can be checked against it).

Run from the repo root (needs the built scenes and the ALIKED model):
  MUJOCO_GL=egl python scripts/make_samples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a file from anywhere

from caremate.agent_loop_v0_7 import START, MuJoCoHeadCamera, ViewPose, decide
from caremate.perception_v0_7 import PerceptionBackend
from caremate.workspace_v0_7 import workspace_polygon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "samples" / "v0_7"
ALIKED = ROOT / "models/vision/aliked-n16rot-top1k-640.onnx"
REFS = ROOT / "references/v0_7"
SCENES = {"default": ROOT / "models/pal_tiago_dual/caremate_room_v0_7.xml",
          "no_book": ROOT / "models/pal_tiago_dual/caremate_room_v0_7_no_book.xml"}
FRONT_RIGHT = ViewPose(0.35, 0.0, -0.60, "front_right")   # the view the agent chose in the demo

SAMPLES = [
    dict(id="1_start_peanut", scene="default", view=START, target="crunch_peanut",
         title="1. Start view: asked for the peanut-butter Crunch",
         caption="The box in front is the caramel flavour: same brand, same layout. The geometric match is "
                 "strong, but the colour check says the region is the look-alike, so it is rejected "
                 "(orange) and the robot keeps searching."),
    dict(id="2_start_caramel", scene="default", view=START, target="crunch_caramel",
         title="2. Same view: asked for the caramel Crunch",
         caption="Same image, different request: now that box is the target and is accepted (green)."),
    dict(id="3_moved_peanut", scene="default", view=FRONT_RIGHT, target="crunch_peanut",
         title="3. After moving: asked for the peanut-butter Crunch again",
         caption="The next-best-view planner drove to the front-right station. The real target is now "
                 "visible and accepted; only then does the robot fetch it."),
    dict(id="4_absent_book", scene="no_book", view=START, target="reading_book",
         title="4. Object not there: asked for the reading book",
         caption="The book is not on the table. The service answers not_found instead of guessing; after its "
                 "view budget the robot says \"Cannot verify, please help\"."),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    be = PerceptionBackend(ALIKED, REFS)
    be.warmup()
    cams, meta = {}, []
    for s in SAMPLES:
        cam = cams.get(s["scene"]) or cams.setdefault(s["scene"], MuJoCoHeadCamera(SCENES[s["scene"]]))
        bgr, pose = cam.capture(s["view"])
        roi = workspace_polygon(pose, cam.W, cam.H)
        res = be.perceive(bgr, s["target"], roi_polygon=roi)
        decision, reason = decide(res, is_last=False)
        cv2.imwrite(str(OUT / f"{s['id']}.png"), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        meta.append({"id": s["id"], "title": s["title"], "caption": s["caption"], "target": s["target"],
                     "view": s["view"].name, "scene": s["scene"],
                     "roi_polygon": None if roi is None else [[round(float(x), 1), round(float(y), 1)] for x, y in roi],
                     "expected": {"status": res.status, "n_inliers": res.n_inliers, "decision": decision,
                                  "reason": reason}})
        print(f"{s['id']:16s} {res.status:9s} inl={res.n_inliers:3d} -> {decision}: {reason}")
    for c in cams.values():
        c.close()
    (OUT / "samples.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(meta)} samples to {OUT}")


if __name__ == "__main__":
    main()
