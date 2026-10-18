"""CareMate v0.5 step 0: kinematic viewpoint sweep of the TIAGo++ head camera.

Renders head_rgb over a grid of (torso_lift, head pan, head tilt) poses in the
v0.4 room and records, for every pose:

  * the clean RGB frame (input data for perception development)
  * the camera's world position / orientation (proves torso lift = translation)
  * GROUND-TRUTH object visibility from MuJoCo segmentation
  * separate joint-limit and frame-edge clipping diagnostics

IMPORTANT: the visibility numbers come from the simulator's segmentation buffer.
They are for offline design (choosing candidate viewpoints) and for evaluation
only. The perception / agent code must NEVER read them at decision time.
A visible segmentation mask touching the image border flags likely frame clipping;
it cannot establish complete object visibility or distinguish all occlusions.

Poses are applied kinematically (qpos + mj_forward via hold_target), not by
stepping the dynamics. Objects stay put because physics is not stepped.

Usage (from ~/Desktop/caremate):
    python -m caremate.viewpoint_sweep                     # default grid
    python -m caremate.viewpoint_sweep --torso 0 0.35 --pan -0.6 0 --tilt -0.3
    MUJOCO_GL=egl python -m caremate.viewpoint_sweep       # headless, if EGL works
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

from caremate.run_sim import hold_target, initialise

TARGETS = ("reading_book", "personal_care_box", "blue_mug", "yellow_mug")
JOINTS = ("torso_lift_joint", "head_1_joint", "head_2_joint")
CAMERA = "head_rgb"
# Distinct BGR colours for overlays on the contact sheet only.
COLOURS = {
    "reading_book": (60, 180, 255),
    "personal_care_box": (200, 90, 220),
    "blue_mug": (255, 140, 40),
    "yellow_mug": (40, 220, 240),
}


def default_scene() -> Path:
    from caremate.build_room_v0_4 import SCENE_V04
    return SCENE_V04


def frange(start: float, stop: float, step: float) -> list[float]:
    n = int(math.floor((stop - start) / step + 1e-9)) + 1
    return [round(start + i * step, 4) + 0.0 for i in range(n)]


def signed_label(value: float, digits: int = 2, show_plus: bool = True) -> str:
    """Format numbers without the confusing negative-zero representation."""
    rounded = round(float(value), digits)
    if rounded == 0.0:
        rounded = 0.0
    sign = "+" if show_plus else ""
    return f"{rounded:{sign}.{digits}f}"


def visible_mask_bbox(mask: np.ndarray) -> tuple[int, tuple[int, int, int, int], bool]:
    """Return visible pixel count, bounding box, and whether it touches frame edge.

    This uses *visible* segmentation pixels only; a fully off-screen or hidden
    object has 0 visible pixels and must NOT be treated as 'not clipped'.
    """
    ys, xs = np.nonzero(mask)
    n = int(xs.size)
    if not n:
        return 0, (-1, -1, -1, -1), False
    height, width = mask.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    edge_touch = x0 == 0 or y0 == 0 or x1 == width - 1 or y1 == height - 1
    return n, (x0, y0, x1, y1), edge_touch


def qpos_of(model: mujoco.MjModel, data: mujoco.MjData, joint: str) -> float:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    return float(data.qpos[model.jnt_qposadr[jid]])


def geom_owner_map(model: mujoco.MjModel, targets: tuple[str, ...]) -> np.ndarray:
    """geom id -> index into targets (or -1), walking up the body tree."""
    target_body = {}
    for k, name in enumerate(targets):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            target_body[bid] = k
    owner = np.full(model.ngeom, -1, dtype=np.int32)
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        while b > 0 and b not in target_body:
            b = int(model.body_parentid[b])
        if b in target_body:
            owner[g] = target_body[b]
    return owner


def project(point: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray,
            fovy_deg: float, width: int, height: int) -> tuple[float, float, bool]:
    """Pinhole projection. MuJoCo cameras look along -z, x right, y up."""
    p = cam_mat.reshape(3, 3).T @ (point - cam_pos)
    if p[2] >= -1e-6:
        return float("nan"), float("nan"), False
    f = 0.5 * height / math.tan(math.radians(fovy_deg) / 2)
    u = width / 2 + f * p[0] / -p[2]
    v = height / 2 - f * p[1] / -p[2]
    inside = 0 <= u < width and 0 <= v < height
    return float(u), float(v), inside


def label(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.45) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def run(args: argparse.Namespace) -> Path:
    scene = args.scene or default_scene()
    if not Path(scene).is_file():
        raise FileNotFoundError(f"{scene} missing. Run: python -m caremate.build_room_v0_4")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    initialise(model, data)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA)
    if cam_id < 0:
        raise KeyError(f"camera {CAMERA} not in model")
    fovy = float(model.cam_fovy[cam_id])
    for j in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jid < 0:
            raise KeyError(f"joint {j} not in model")
        lo, hi = model.jnt_range[jid]
        print(f"{j:18s} range [{lo:+.3f}, {hi:+.3f}]")

    present = [t for t in TARGETS
               if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, t) >= 0]
    if not present:
        raise RuntimeError("No target object bodies found in the scene")
    missing = sorted(set(TARGETS) - set(present))
    if missing:
        print(f"NOTE: objects not in scene (skipped): {missing}")
    owner = geom_owner_map(model, tuple(present))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = args.out / f"sweep_{stamp}"
    (out / "rgb").mkdir(parents=True, exist_ok=True)

    W, H = args.width, args.height
    grid = list(itertools.product(args.torso, args.pan, args.tilt))
    print(f"Rendering {len(grid)} viewpoints at {W}x{H} -> {out}")

    rows: list[dict] = []
    thumbs: dict[float, dict[tuple[float, float], np.ndarray]] = {t: {} for t in args.torso}
    # Resize the offscreen framebuffer before constructing either renderer.
    # Default MJCF dimensions can be smaller than --width/--height.
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), W)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), H)
    rgb_r = mujoco.Renderer(model, height=H, width=W)
    seg_r = mujoco.Renderer(model, height=H, width=W)
    seg_r.enable_segmentation_rendering()
    try:
        for i, (torso, pan, tilt) in enumerate(grid):
            hold_target(model, data, "torso_lift_joint", torso)
            hold_target(model, data, "head_1_joint", pan)
            hold_target(model, data, "head_2_joint", tilt)
            mujoco.mj_forward(model, data)
            actual = [qpos_of(model, data, j) for j in JOINTS]
            joint_clipped = any(abs(a - r) > 1e-6
                                for a, r in zip(actual, (torso, pan, tilt)))

            rgb_r.update_scene(data, camera=CAMERA)
            rgb = rgb_r.render()
            seg_r.update_scene(data, camera=CAMERA)
            seg = seg_r.render()
            geom_ids = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
            obj_idx = np.where(geom_ids >= 0, owner[np.clip(geom_ids, 0, None)], -1)

            vid = (f"v{i:03d}_t{signed_label(torso, 3, show_plus=False)}_"
                   f"p{signed_label(pan)}_k{signed_label(tilt)}")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out / "rgb" / f"{vid}.png"), bgr)

            cam_pos = data.cam_xpos[cam_id].copy()
            cam_mat = data.cam_xmat[cam_id].copy()
            # Optical axis (camera -z) in world frame.
            axis = -cam_mat.reshape(3, 3)[:, 2]
            row = {
                "view_id": vid, "torso": torso, "pan": pan, "tilt": tilt,
                "q_torso": actual[0], "q_pan": actual[1], "q_tilt": actual[2],
                "clipped_to_limits": int(joint_clipped),
                "cam_x": cam_pos[0], "cam_y": cam_pos[1], "cam_z": cam_pos[2],
                "axis_x": axis[0], "axis_y": axis[1], "axis_z": axis[2],
            }
            overlay = bgr.copy()
            frame_clipped_objects = []
            visible_objects = 0
            for k, name in enumerate(present):
                mask = obj_idx == k
                px, (x0, y0, x1, y1), frame_clipped = visible_mask_bbox(mask)
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                u, v, centre_in = project(data.xpos[bid], cam_pos, cam_mat, fovy, W, H)
                if px:
                    visible_objects += 1
                    if frame_clipped:
                        frame_clipped_objects.append(name)
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), COLOURS[name], 2)
                row.update({
                    f"{name}_px": px,
                    f"{name}_bbox": f"{x0} {y0} {x1} {y1}",
                    # Backwards-compatible CSV column.
                    f"{name}_truncated": int(frame_clipped),
                    f"{name}_frame_clipped": int(frame_clipped),
                    f"{name}_status": ("NOT_VISIBLE" if not px else
                                       "FRAME_CLIPPED" if frame_clipped else "VISIBLE"),
                    f"{name}_centre_in_frame": int(centre_in),
                    f"{name}_u": round(u, 1) if centre_in else "",
                    f"{name}_v": round(v, 1) if centre_in else "",
                })
            row["frame_clipped"] = int(bool(frame_clipped_objects))
            row["frame_clipped_objects"] = ",".join(frame_clipped_objects)
            row["n_visible_targets"] = visible_objects
            rows.append(row)

            thumb = cv2.resize(overlay, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
            # 'J' means a requested joint angle exceeded joint limits;
            # 'F' means at least one visible object reaches the frame boundary.
            flags = (" J!" if joint_clipped else "") + (" F!" if frame_clipped_objects else "")
            label(thumb, (f"t={signed_label(torso, show_plus=False)} p={signed_label(pan)} "
                          f"k={signed_label(tilt)}{flags}"), (6, 16), 0.42)
            y = 34
            for name in present:
                px = row[f"{name}_px"]
                suffix = " EDGE" if row[f"{name}_frame_clipped"] else "" if px else " MISS"
                label(thumb, f"{name[:10]}:{px}{suffix}", (6, y), 0.38)
                y += 14
            thumbs[torso][(pan, tilt)] = thumb
            if (i + 1) % 20 == 0 or i + 1 == len(grid):
                print(f"  {i + 1}/{len(grid)}")
    finally:
        rgb_r.close()
        seg_r.close()

    # CSV with every pose and visibility measure.
    with open(out / "viewpoints.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # One contact sheet per torso height: rows = tilt, columns = pan.
    for torso, cells in thumbs.items():
        tw, th = W // 2, H // 2
        sheet = np.full((th * len(args.tilt), tw * len(args.pan), 3), 30, np.uint8)
        for r, tilt in enumerate(args.tilt):
            for c, pan in enumerate(args.pan):
                sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = cells[(pan, tilt)]
        cv2.imwrite(str(out / f"sheet_torso_{signed_label(torso, 3, show_plus=False)}.png"), sheet)

    summary = summarise(rows, present, args)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(summary)
    print(f"\nDone. Open {out}/sheet_torso_*.png first, then viewpoints.csv")
    return out


def summarise(rows: list[dict], present: list[str], args: argparse.Namespace) -> dict:
    s: dict = {"n_views": len(rows), "objects": present,
               "note": "Visibility/frame-edge flags are simulator ground truth: design/evaluation only."
                       " A mask touching the border indicates likely cropping, not full visibility."}
    s["n_joint_clipped_views"] = sum(r["clipped_to_limits"] for r in rows)
    s["n_frame_clipped_views"] = sum(r["frame_clipped"] for r in rows)
    s["n_views_no_visible_targets"] = sum(r["n_visible_targets"] == 0 for r in rows)

    # Camera translation produced by torso lift at identical head angles.
    lo, hi = min(args.torso), max(args.torso)
    by_pose = {(r["torso"], r["pan"], r["tilt"]): r for r in rows}
    shifts = []
    for pan, tilt in itertools.product(args.pan, args.tilt):
        a, b = by_pose.get((lo, pan, tilt)), by_pose.get((hi, pan, tilt))
        if a and b:
            d = np.array([b["cam_x"] - a["cam_x"], b["cam_y"] - a["cam_y"], b["cam_z"] - a["cam_z"]])
            shifts.append(float(np.linalg.norm(d)))
    s["torso_camera_translation_m"] = {
        "torso_low": lo, "torso_high": hi,
        "mean": round(float(np.mean(shifts)), 4) if shifts else None,
    }

    s["best_views_per_object"] = {}
    for name in present:
        # Prefer valid, non-cropped views to a larger *partial* object image.
        ranked = sorted(rows, key=lambda r: (
            not r["clipped_to_limits"] and not r[f"{name}_frame_clipped"]
            and r[f"{name}_px"] > 0,
            r[f"{name}_px"]), reverse=True)[:5]
        s["best_views_per_object"][name] = [
            {"view_id": r["view_id"], "px": r[f"{name}_px"],
             "truncated": r[f"{name}_truncated"],
             "clipped_to_limits": r["clipped_to_limits"]}
            for r in ranked]

    # Views that see every object untruncated, ranked by the worst-seen object.
    full = [r for r in rows
            if not r["clipped_to_limits"] and all(
                r[f"{n}_px"] >= args.min_px and not r[f"{n}_frame_clipped"]
                for n in present)]
    full.sort(key=lambda r: min(r[f"{n}_px"] for n in present), reverse=True)
    s["views_seeing_all_objects"] = [
        {"view_id": r["view_id"], "min_px": min(r[f"{n}_px"] for n in present),
         **{n: r[f"{n}_px"] for n in present}} for r in full[:10]]
    s["count_views_seeing_all_objects"] = len(full)
    return s


def print_summary(s: dict) -> None:
    t = s["torso_camera_translation_m"]
    print(f"\nTorso {t['torso_low']} -> {t['torso_high']}: mean camera translation "
          f"{t['mean']} m at fixed head angles")
    print(f"Joint-limit clipped views: {s['n_joint_clipped_views']}")
    print(f"Frame-edge clipped views: {s['n_frame_clipped_views']}")
    print(f"Views with zero visible targets: {s['n_views_no_visible_targets']}")
    print(f"Views seeing all objects untruncated: {s['count_views_seeing_all_objects']}")
    for v in s["views_seeing_all_objects"][:5]:
        print(f"  {v['view_id']}  min_px={v['min_px']}")
    for name, views in s["best_views_per_object"].items():
        b = views[0]
        print(f"Best for {name:18s}: {b['view_id']}  px={b['px']}"
              f"{' (frame clipped)' if b['truncated'] else ''}"
              f"{' (joint limit)' if b['clipped_to_limits'] else ''}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", type=Path, default=None, help="MJCF (default: v0.4 room)")
    p.add_argument("--torso", type=float, nargs="+", default=[0.0, 0.175, 0.35])
    p.add_argument("--pan", type=float, nargs="+", default=frange(-0.9, 0.6, 0.3))
    p.add_argument("--tilt", type=float, nargs="+", default=[-0.6, -0.4, -0.2, 0.0, 0.2])
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--min-px", type=int, default=150,
                   help="Min visible pixels for an object to count as seen in the summary")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
