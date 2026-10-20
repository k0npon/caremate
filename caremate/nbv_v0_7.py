"""CareMate v0.7: base stations + depth occupancy map for next-best-view selection.

Why: with the base fixed, the start view already covers ~90 % of the table
and an object hidden behind a taller one stays hidden from every head pose,
so no viewpoint policy can do better than any other.  The robot can move.

Runtime inputs (all legitimate robot sensing / prior knowledge):
  * head depth image (TIAGo++ has an RGB-D head camera) - used ONLY to decide
    where to look next, never for identity;
  * camera pose from forward kinematics;
  * static furniture map (workspace volume above the table) and a few
    pre-surveyed base stations around it.
Never used: segmentation, object IDs or object poses.

Base motion is KINEMATIC placement at a station (no navigation planner,
no wheel dynamics).  Travel distance is logged so policies pay for it.
"""
from __future__ import annotations

import math

import numpy as np

from caremate.workspace_v0_7 import WORKSPACES

TABLE_CENTRE = (1.40, -0.45)


def _facing(x, y):
    return math.atan2(TABLE_CENTRE[1] - y, TABLE_CENTRE[0] - x)


# Pre-surveyed base poses (x, y, yaw) around the main table; collision-checked
# offline (no robot-furniture contacts in the v0.7 room).
STATIONS = {
    "front": (0.0, 0.0, 0.0),
    "front_right": (0.35, -1.15, _facing(0.35, -1.15)),
    "front_left": (0.35, 0.25, _facing(0.35, 0.25)),
}


def station_distance(a: str, b: str) -> float:
    (xa, ya, _), (xb, yb, _) = STATIONS[a], STATIONS[b]
    return math.hypot(xa - xb, ya - yb)


UNKNOWN, FREE, OCCUPIED = 0, 1, 2


class OccupancyGrid:
    """Voxel map of the volume just above the tabletop, filled from depth.

    unknown  never observed (or always behind something)
    free     a depth ray passed through it
    occupied a depth ray ended in it (object surface)
    """

    def __init__(self, workspace="main_table", res=0.03, z_min_above=0.02, eps=0.025):
        w = WORKSPACES[workspace]
        self.res, self.eps = res, eps
        xs = np.arange(w["x"][0] + res / 2, w["x"][1], res)
        ys = np.arange(w["y"][0] + res / 2, w["y"][1], res)
        zs = np.arange(w["z"][0] + z_min_above + res / 2, w["z"][1], res)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        self.centres = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        self.state = np.zeros(len(self.centres), np.uint8)

    @staticmethod
    def _project(pts, pose, W, H):
        R = np.asarray(pose["mat"], float).reshape(3, 3)
        pc = (pts - np.asarray(pose["pos"], float)) @ R
        z = -pc[:, 2]
        f = 0.5 * H / math.tan(math.radians(pose["fovy_deg"]) / 2)
        zs = np.where(z > 1e-3, z, 1.0)
        u = W / 2 + f * pc[:, 0] / zs
        v = H / 2 - f * pc[:, 1] / zs
        return u, v, z

    def integrate(self, depth: np.ndarray, pose: dict) -> dict:
        H, W = depth.shape
        u, v, z = self._project(self.centres, pose, W, H)
        inside = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.nonzero(inside)[0]
        d = depth[v[idx].astype(int), u[idx].astype(int)]
        zi = z[idx]
        free = idx[zi < d - self.eps]
        occ = idx[np.abs(zi - d) <= self.eps]
        before = self.counts()
        self.state[free[self.state[free] != OCCUPIED]] = FREE
        self.state[occ] = OCCUPIED
        after = self.counts()
        return {"newly_observed": before["unknown"] - after["unknown"], **after}

    def counts(self) -> dict:
        return {"unknown": int((self.state == UNKNOWN).sum()),
                "free": int((self.state == FREE).sum()),
                "occupied": int((self.state == OCCUPIED).sum())}

    def expected_gain(self, pose: dict, W=160, H=120, margin=0.04) -> float:
        """Fraction of currently-unknown voxels a camera at `pose` would see.

        Occupied voxels are splatted into a coarse z-buffer; an unknown voxel
        counts if it projects inside the frame and in front of that buffer.
        """
        unk = self.state == UNKNOWN
        if not unk.any():
            return 0.0
        zbuf = np.full((H, W), np.inf, np.float32)
        occ = self.state == OCCUPIED
        if occ.any():
            u, v, z = self._project(self.centres[occ], pose, W, H)
            ok = (z > 0.05) & (u >= -2) & (u < W + 2) & (v >= -2) & (v < H + 2)
            f = 0.5 * H / math.tan(math.radians(pose["fovy_deg"]) / 2)
            for ui, vi, zi in zip(u[ok], v[ok], z[ok]):
                r = max(1, int(math.ceil(0.6 * f * self.res / zi)))
                x0, x1 = max(0, int(ui) - r), min(W, int(ui) + r + 1)
                y0, y1 = max(0, int(vi) - r), min(H, int(vi) + r + 1)
                if x0 < x1 and y0 < y1:
                    np.minimum(zbuf[y0:y1, x0:x1], zi, out=zbuf[y0:y1, x0:x1])
        u, v, z = self._project(self.centres[unk], pose, W, H)
        m = margin * min(W, H)
        inside = (z > 0.05) & (u >= m) & (u < W - m) & (v >= m) & (v < H - m)
        ui = np.clip(u.astype(int), 0, W - 1); vi = np.clip(v.astype(int), 0, H - 1)
        visible = inside & (z < zbuf[vi, ui] - self.res)
        return float(visible.sum()) / float(unk.sum())
