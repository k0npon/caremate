"""Interactive CareMate v0.1: world viewer + TIAGo++ head camera.

Controls in robot camera window: A/D pan; W/S tilt; SPACE pause; C save RGB;
Q or ESC quit. In `--no-viewer` mode only the camera preview is shown.

v0.1 intentionally does NOT include automatic fetching, navigation, grasping,
OpenCV 5 matching or AWS deployment. Those belong to later milestones.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from caremate.build_scene import SCENE_XML


def joint_address(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise KeyError(f"Cannot find {name} in robot model")
    return int(model.jnt_qposadr[joint_id])


def hold_target(model: mujoco.MjModel, data: mujoco.MjData,
                joint_name: str, new_position: float) -> None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    lo, hi = (float(x) for x in model.jnt_range[jid])
    desired = float(np.clip(new_position, lo, hi))
    data.qpos[int(model.jnt_qposadr[jid])] = desired
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hold_{joint_name}"
    )
    if actuator_id >= 0:
        data.ctrl[actuator_id] = desired


def initialise(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    # Position motors hold current initial posture rather than applying a 0
    # target blindly. This prepares later dynamic simulation experiments.
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name and name.startswith("hold_"):
            joint = name.removeprefix("hold_")
            data.ctrl[i] = data.qpos[joint_address(model, joint)]
    # The initial robot facing +X sees the storage table across the room.
    hold_target(model, data, "head_1_joint", 0.0)
    hold_target(model, data, "head_2_joint", -0.12)
    mujoco.mj_forward(model, data)


def run(show_viewer: bool, physics: bool, fps: int, save_dir: Path,
        snapshot: bool, duration: float | None) -> None:
    if not SCENE_XML.exists():
        raise FileNotFoundError("Scene missing. Run: python -m caremate.build_scene")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    initialise(model, data)
    save_dir.mkdir(parents=True, exist_ok=True)

    # We use separate MuJoCo and OpenCV windows. Head camera RGB is rendered
    # straight from MuJoCo; no fake image or external webcam is used.
    import contextlib
    if show_viewer:
        viewer_context = mujoco.viewer.launch_passive(model, data)
    else:
        viewer_context = contextlib.nullcontext(None)

    step_budget = max(1, round((1.0 / fps) / model.opt.timestep))
    with viewer_context as viewer:
        if viewer is not None:
            viewer.cam.lookat[:] = [1.3, 0.1, 0.85]
            viewer.cam.distance = 4.4
            viewer.cam.azimuth = 230
            viewer.cam.elevation = -21
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            if not snapshot:
                cv2.namedWindow("CareMate | TIAGo++ Head Camera", cv2.WINDOW_NORMAL)
            start = time.monotonic()
            frame_count = 0
            paused = not physics
            while True:
                if physics and not paused:
                    for _ in range(step_budget):
                        mujoco.mj_step(model, data)
                else:
                    mujoco.mj_forward(model, data)

                renderer.update_scene(data, camera="head_rgb")
                rgb = renderer.render()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(bgr, f"CareMate v0.1 | Head RGB | frame {frame_count}",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.54,
                            (230, 240, 245), 1, cv2.LINE_AA)
                cv2.putText(bgr, "A/D pan  W/S tilt  C save  Q exit",
                            (10, 457), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                            (235, 235, 235), 1, cv2.LINE_AA)
                if snapshot:
                    target = save_dir / "head_rgb.png"
                    cv2.imwrite(str(target), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    print(f"Saved screenshot: {target}")
                    break
                cv2.imshow("CareMate | TIAGo++ Head Camera", bgr)
                key = cv2.waitKey(max(1, round(1000.0 / fps))) & 0xFF
                if key in (ord("q"), 27):
                    break
                delta = 0.075
                if key in (ord("a"), ord("d"), ord("w"), ord("s")):
                    joint = "head_1_joint" if key in (ord("a"), ord("d")) else "head_2_joint"
                    address = joint_address(model, joint)
                    sign = 1 if key in (ord("a"), ord("w")) else -1
                    hold_target(model, data, joint, float(data.qpos[address]) + sign * delta)
                elif key == ord(" ") and physics:
                    paused = not paused
                elif key == ord("c"):
                    path = save_dir / f"head_rgb_{frame_count:05d}.png"
                    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    print(f"Saved: {path}")
                if viewer is not None:
                    if not viewer.is_running():
                        break
                    viewer.sync()
                frame_count += 1
                if duration is not None and time.monotonic() - start > duration:
                    break
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-viewer", action="store_true", help="Only OpenCV camera window")
    parser.add_argument("--physics", action="store_true", help="Experimental physics stepping")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--save-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--snapshot", action="store_true", help="Save RGB and exit; no OpenCV window")
    parser.add_argument("--duration", type=float, default=None, help="Quit after N wall-clock seconds")
    args = parser.parse_args()
    if not (1 <= args.fps <= 60):
        parser.error("--fps must be between 1 and 60")
    run(not args.no_viewer and not args.snapshot, args.physics, args.fps,
        args.save_dir, args.snapshot, args.duration)


if __name__ == "__main__":
    main()
