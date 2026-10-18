"""CareMate v0.4 spectator viewer and head-camera preview.

Q/Esc quits; A/D rotate head, W/S tilt, C saves head RGB. This is a scene
preview, NOT grasp or autonomous navigation. MuJoCo renders camera frames.
"""
from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from caremate.build_room_v0_4 import SCENE_V04
from caremate.run_sim import initialise, hold_target, joint_address


def run(args: argparse.Namespace) -> None:
    if not SCENE_V04.is_file():
        raise FileNotFoundError("Run: python -m caremate.build_room_v0_4")
    model = mujoco.MjModel.from_xml_path(str(SCENE_V04))
    data = mujoco.MjData(model)
    initialise(model, data)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    viewer_ctx = (
        mujoco.viewer.launch_passive(model, data)
        if not (args.no_viewer or args.snapshot)
        else contextlib.nullcontext(None)
    )
    with viewer_ctx as viewer:
        if viewer is not None:
            # Third person camera looks from the open side. Mouse still works.
            viewer.cam.lookat[:] = [1.45, 0.15, 0.85]
            viewer.cam.distance = args.distance
            viewer.cam.azimuth = args.azimuth
            viewer.cam.elevation = args.elevation
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            win = "CareMate v0.4 | TIAGo++ RGB camera"
            if not args.snapshot:
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            frame_num = 0
            start = time.monotonic()
            steps = max(1, round(1 / (args.fps * model.opt.timestep)))
            paused = not args.physics
            try:
                while True:
                    if args.physics and not paused:
                        for _ in range(steps):
                            mujoco.mj_step(model, data)
                    else:
                        mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera="head_rgb")
                    raw_rgb = renderer.render()
                    bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
                    if args.snapshot:
                        path = args.save_dir / "caremate_v04_head_rgb.png"
                        if not cv2.imwrite(str(path), bgr):
                            raise IOError(f"Could not save {path}")
                        print(f"Saved screenshot: {path}")
                        break
                    cv2.putText(bgr, "CareMate v0.4 / Head RGB", (12, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250, 250, 250), 1, cv2.LINE_AA)
                    cv2.putText(bgr, "A/D pan  W/S tilt  C save  Q/Esc exit",
                                (12, 458), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (250, 250, 250), 1, cv2.LINE_AA)
                    cv2.imshow(win, bgr)
                    key = cv2.waitKey(max(1, int(1000 / args.fps))) & 255
                    if key in (ord("q"), 27):
                        break
                    if key in (ord("a"), ord("d"), ord("w"), ord("s")):
                        joint = ("head_1_joint" if key in (ord("a"), ord("d"))
                                 else "head_2_joint")
                        sign = 1 if key in (ord("a"), ord("w")) else -1
                        addr = joint_address(model, joint)
                        hold_target(model, data, joint, float(data.qpos[addr]) + sign * 0.075)
                    elif key == ord("c"):
                        path = args.save_dir / f"caremate_v04_{frame_num:05}.png"
                        if cv2.imwrite(str(path), cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)):
                            print(f"Saved: {path}")
                    elif key == ord(" ") and args.physics:
                        paused = not paused
                    if viewer is not None:
                        if not viewer.is_running():
                            break
                        viewer.sync()
                    frame_num += 1
                    if args.duration is not None and time.monotonic()-start >= args.duration:
                        break
            finally:
                if not args.snapshot:
                    cv2.destroyWindow(win)
                    cv2.waitKey(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", action="store_true", help="Save one frame, no windows")
    p.add_argument("--no-viewer", action="store_true", help="Only RGB window")
    p.add_argument("--physics", action="store_true", help="EXPERIMENTAL: step dynamics")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--save-dir", type=Path, default=Path("outputs"))
    p.add_argument("--azimuth", type=float, default=135.0,
                   help="Third-person orbit angle (degrees)")
    p.add_argument("--elevation", type=float, default=-23.0)
    p.add_argument("--distance", type=float, default=4.0)
    args = p.parse_args()
    if not 1 <= args.fps <= 60:
        p.error("--fps must be 1..60")
    if args.distance <= 0:
        p.error("--distance must be positive")
    try:
        run(args)
    except KeyboardInterrupt:
        print("Stopped. Prefer Q/Esc in the RGB window for graceful shutdown.")


if __name__ == "__main__":
    main()
