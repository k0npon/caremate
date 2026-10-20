"""CareMate v0.7 live demo: watch the agent search in the MuJoCo viewer.

Opens the interactive MuJoCo viewer and animates the robot between views
(base glides between stations, torso/head move smoothly), while an OpenCV
window shows each head-camera frame with the OpenCV 5 perception overlay and
the agent's decision.  Optionally records an mp4 for the submission video.

Motion here is purely visual interpolation between the kinematic poses the
agent chose; perception and decisions are identical to agent_loop_v0_7.

Examples (from ~/Desktop/caremate):
  # default v0.7 room, find the occluded look-alike
  python -m caremate.demo_v0_7 --target crunch_peanut
  # a randomised evaluation episode (same layout as eval_v0_7 --seed 31, episode 7)
  python -m caremate.demo_v0_7 --seed 31 --episode 7 --policy active
  # record a video (works headless too: add --no-viewer, MUJOCO_GL=egl)
  python -m caremate.demo_v0_7 --target crunch_peanut --record outputs/demo_peanut.mp4
"""
from __future__ import annotations

import argparse
import math
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import mujoco
import numpy as np

from caremate.agent_loop_v0_7 import (make_backend, POLICIES, ROOT, START, MuJoCoHeadCamera, SiftStandIn,
                                      ViewPose, run_agent)
from caremate.nbv_v0_7 import STATIONS
from caremate.perception_v0_7 import PerceptionBackend, draw

# TIAGo++ arms tucked beside the torso (display only; not in the head camera view).
TUCK_LEFT = (-1.10, 1.47, 2.71, 1.71, -1.57, 1.37, 0.0)
TUCK_RIGHT = (-1.10, 1.47, -2.71, 1.71, 1.57, 1.37, 0.0)
FPS = 30
WINDOW = "CareMate head camera (live) + OpenCV 5 perception"


def pump(ms=1):
    """Keep the OpenCV window responsive (GNOME flags it 'not responding' otherwise)."""
    return cv2.waitKey(ms) & 0xFF


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class DemoCamera(MuJoCoHeadCamera):
    """MuJoCoHeadCamera that animates to each requested view."""

    def __init__(self, scene, viewer=None, recorder=None, speed=1.0):
        super().__init__(scene)
        from caremate.run_sim import hold_target
        for i, v in enumerate(TUCK_LEFT):
            hold_target(self.model, self.data, f"arm_left_{i + 1}_joint", v)
        for i, v in enumerate(TUCK_RIGHT):
            hold_target(self.model, self.data, f"arm_right_{i + 1}_joint", v)
        self.viewer, self.recorder, self.speed = viewer, recorder, speed
        self.state = (*STATIONS[START.base], START.torso, START.pan, START.tilt)
        self.caption = ""
        self.show_window = False
        self.live = mujoco.Renderer(self.model, height=480, width=640)

    def live_frame(self):
        self.live.update_scene(self.data, camera="head_rgb")
        img = cv2.cvtColor(self.live.render(), cv2.COLOR_RGB2BGR)
        _caption(img, "LIVE  " + self.caption)
        return img

    def _frame(self):
        if self.viewer is not None:
            self.viewer.sync()
        if self.show_window:
            cv2.imshow(WINDOW, self.live_frame())
            pump(1)
        if self.recorder is not None:
            self.recorder.write_world(self.data, self.caption)

    def animate_to(self, v: ViewPose):
        x0, y0, yaw0, t0, p0, k0 = self.state
        x1, y1, yaw1 = STATIONS[v.base]
        dist = math.hypot(x1 - x0, y1 - y0)
        dyaw = _wrap(yaw1 - yaw0)
        secs = (0.8 + dist / 0.5 + abs(dyaw) / 1.0) / self.speed     # ~0.5 m/s, 1 rad/s
        n = max(2, int(secs * FPS))
        for i in range(1, n + 1):
            s = 0.5 - 0.5 * math.cos(math.pi * i / n)                 # ease in/out
            pose = (x0 + s * (x1 - x0), y0 + s * (y1 - y0), yaw0 + s * dyaw,
                    t0 + s * (v.torso - t0), p0 + s * (v.pan - p0), k0 + s * (v.tilt - k0))
            if self.viewer is not None:
                with self.viewer.lock():
                    self.set_robot(*pose)
            else:
                self.set_robot(*pose)
            self._frame()
            if self.viewer is not None and not self.show_window:
                time.sleep(1.0 / FPS)
        self.state = (x1, y1, yaw1, v.torso, v.pan, v.tilt)

    def close(self):
        self.live.close()
        super().close()

    def capture(self, v: ViewPose):
        self.caption = f"moving to {v.name}"
        self.animate_to(v)
        self.caption = f"perceiving at {v.name} (OpenCV 5 ALIKED + USAC)..."
        if self.show_window:
            cv2.imshow(WINDOW, self.live_frame())
            pump(1)
        if self.viewer is not None:
            with self.viewer.lock():
                out = super().capture(v)
        else:
            out = super().capture(v)
        return out


class Recorder:
    """Writes a 1280x720 video: third-person world view + head-camera inset."""

    def __init__(self, path: Path, model):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.w, self.h = 1280, 720
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (self.w, self.h))
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), self.w)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), self.h)
        self.r = mujoco.Renderer(model, self.h, self.w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = (0.85, -0.45, 0.65)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 3.6, 215.0, -32.0
        self.inset = None
        self.path = path

    def write_world(self, data, caption=""):
        for _ in range(2):   # see MuJoCoHeadCamera.capture (first frame after a move lags)
            self.r.update_scene(data, camera=self.cam)
            rgb = self.r.render()
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if self.inset is not None:
            ih, iw = self.inset.shape[:2]
            frame[12:12 + ih, self.w - iw - 12:self.w - 12] = self.inset
            cv2.rectangle(frame, (self.w - iw - 13, 11), (self.w - 12, 12 + ih), (255, 255, 255), 1)
        _caption(frame, caption)
        self.writer.write(frame)

    def hold(self, data, caption, seconds):
        for _ in range(int(seconds * FPS)):
            self.write_world(data, caption)

    def close(self):
        self.writer.release()
        self.r.close()


def _caption(img, text, y=None):
    y = y or img.shape[0] - 18
    cv2.rectangle(img, (0, y - 26), (img.shape[1], img.shape[0]), (20, 20, 20), -1)
    cv2.putText(img, text[:150], (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", type=Path, default=ROOT / "models/pal_tiago_dual/caremate_room_v0_7.xml")
    p.add_argument("--target", default=None, help="default scene: object to find")
    p.add_argument("--seed", type=int, default=None, help="use a randomised eval layout")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--policy", choices=list(POLICIES), default="active")
    p.add_argument("--budget", type=int, default=3)
    p.add_argument("--aliked", type=Path, default=ROOT / "models/vision/aliked-n16rot-top1k-640.onnx")
    p.add_argument("--refs", type=Path, default=ROOT / "references/v0_7")
    p.add_argument("--record", type=Path, default=None, help="write an mp4")
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--standin-sift", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--backend-url", default=None, help="use the AWS perception service")
    a = p.parse_args()

    backend = make_backend(a)
    print(f"{backend.method}; warm-up {backend.warmup():.1f}s")

    viewer = None
    if not a.no_viewer:
        import mujoco.viewer
    # The camera owns model/data; the viewer is attached after construction.
    cam = DemoCamera(a.scene, None, None, a.speed)
    target = a.target or "reading_book"
    if a.seed is not None:
        from caremate.eval_v0_7 import TARGETS, apply_layout, sample_layout
        rng = random.Random(a.seed)
        # Reproduce eval_v0_7's random stream up to the requested episode.
        for ep in range(a.episode + 1):
            t = rng.choice(TARGETS)
            present = rng.random() >= 0.25
            mode = rng.choices(["open", "occluded", "edge"], weights=[0.4, 0.4, 0.2])[0] if present else "absent"
            layout = sample_layout(rng, t, present, mode)
        target = a.target or t
        apply_layout(cam.model, cam.data, layout)
        print(f"Episode {a.episode} of seed {a.seed}: target={target} present={present} mode={mode}")
    cam._set(START)
    rec = Recorder(a.record, cam.model) if a.record else None
    cam.recorder = rec
    if not a.no_viewer:
        viewer = mujoco.viewer.launch_passive(cam.model, cam.data)
        viewer.cam.lookat[:] = (0.85, -0.45, 0.65)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.6, 215.0, -32.0
        cam.viewer = viewer
        cam.show_window = True
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(WINDOW, cam.live_frame())
        pump(1)
        viewer.sync()
    if rec:
        cam.caption = f"CareMate: find '{target}'  |  policy={a.policy}  |  OpenCV 5 ALIKED + USAC"
        rec.hold(cam.data, cam.caption, 1.5)

    def on_step(step, view, why_view, bgr, res, roi, decision, reason):
        vis = draw(bgr, res, None, roi)
        vis = cv2.resize(vis, (640, 480))
        _caption(vis, f"step {step}: {decision} - {reason}")
        if not a.no_viewer:
            # Hold the perception result on screen while keeping both windows alive.
            cv2.imshow(WINDOW, vis)
            t_end = time.time() + 2.5 / a.speed
            while time.time() < t_end:
                viewer.sync()
                pump(30)
        if rec:
            rec.inset = cv2.resize(vis, (448, 336))
            rec.hold(cam.data, f"step {step}: {res.status} ({res.n_inliers} inliers) -> {decision}: {reason}", 2.5)
        print(f"  why this view: {why_view}")

    out = ROOT / "outputs/demo_v0_7" / f"{datetime.now():%Y%m%d_%H%M%S}_{target}_{a.policy}"
    try:
        result = run_agent(target, backend, cam, POLICIES[a.policy](0), a.budget, out,
                           save_images=True, scene=a.scene.name, on_step=on_step)
        if rec:
            rec.hold(cam.data, f"RESULT: {result['outcome']} after {result['observations']} observation(s), "
                               f"base travel {result['base_travel_m']:.2f} m (grasp not authorised)", 3.0)
        if viewer is not None:
            print("Close the MuJoCo viewer window to exit.")
            while viewer.is_running():
                viewer.sync()
                if pump(30) == 27:      # Esc in the camera window also exits
                    break
    finally:
        if rec:
            rec.close()
            print(f"Video: {rec.path}")
        if viewer is not None:
            viewer.close()
        cv2.destroyAllWindows()
        cam.close()
        backend.close()


if __name__ == "__main__":
    main()
