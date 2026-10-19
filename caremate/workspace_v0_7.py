"""CareMate v0.7: workspace ROI from a static furniture map + the robot's own camera pose.

Where should perception look?  NOT where the objects are (that would be
simulator ground truth).  A home robot legitimately knows two things:

  1. a static map of furniture (tables don't move; measured once at setup), and
  2. its own camera pose from joint encoders + forward kinematics.

Projecting "the volume above the tabletop" into the image gives a search
region that excludes floor, walls, the resident and the wheelchair without
knowing anything about the objects on the table.

Map values below are the v0.4/v0.7 open table (top surface z = 0.76 m),
recorded offline.  Runtime code must not query MuJoCo object bodies/geoms.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

# Static map: axis-aligned volume above each work surface (world frame, metres).
WORKSPACES = {
    "main_table": {"x": (0.80, 2.00), "y": (-0.825, -0.075), "z": (0.76, 1.10)},
}


def camera_pose_from_mujoco(model, data, camera: str = "head_rgb") -> dict:
    """Robot proprioception: the camera's world pose computed by forward kinematics."""
    import mujoco
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cid < 0:
        raise KeyError(camera)
    return {"pos": data.cam_xpos[cid].tolist(), "mat": data.cam_xmat[cid].tolist(),
            "fovy_deg": float(model.cam_fovy[cid])}


def project_points(pts_world: np.ndarray, pose: dict, width: int, height: int):
    """MuJoCo camera convention: looks along -z, x right, y up."""
    R = np.asarray(pose["mat"], float).reshape(3, 3)
    p = (np.asarray(pts_world, float) - np.asarray(pose["pos"], float)) @ R  # world -> cam
    f = 0.5 * height / math.tan(math.radians(pose["fovy_deg"]) / 2)
    z = -p[:, 2]
    front = z > 1e-3
    u = width / 2 + f * p[:, 0] / np.where(front, z, 1)
    v = height / 2 - f * p[:, 1] / np.where(front, z, 1)
    return np.stack([u, v], 1), front


def workspace_polygon(pose: dict, width: int, height: int, name: str = "main_table",
                      margin_px: int = 12) -> np.ndarray | None:
    """Convex image polygon of the workspace volume, clipped to the frame.

    Returns None if the workspace is not in view.  Corners behind the camera
    are handled by sampling the box edges densely and keeping front points.
    """
    w = WORKSPACES[name]
    xs, ys, zs = w["x"], w["y"], w["z"]
    t = np.linspace(0, 1, 25)
    pts = []
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    for i in range(8):
        for j in range(i + 1, 8):
            if np.count_nonzero(corners[i] != corners[j]) == 1:   # box edge
                pts.append(corners[i] + t[:, None] * (corners[j] - corners[i]))
    uv, front = project_points(np.concatenate(pts), pose, width, height)
    uv = uv[front]
    if len(uv) < 3:
        return None
    hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
    frame = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    area, clipped = cv2.intersectConvexConvex(hull, frame)
    if area < 1 or clipped is None:
        return None
    poly = clipped.reshape(-1, 2)
    if margin_px:
        c = poly.mean(0)
        d = poly - c
        n = np.linalg.norm(d, axis=1, keepdims=True)
        poly = c + d * (1 + margin_px / np.maximum(n, 1))
        poly[:, 0] = np.clip(poly[:, 0], 0, width - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, height - 1)
    return poly.astype(np.float32)


def polygon_mask(poly: np.ndarray, width: int, height: int) -> np.ndarray:
    m = np.zeros((height, width), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 255)
    return m
