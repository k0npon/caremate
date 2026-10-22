"""CareMate v0.7: kinematic fetch-and-deliver phase (scripted demo, NOT a grasp planner).

After the agent stops with TARGET_LOCALISED_UNCALIBRATED, TIAGo++ reaches for
the localised object with its right arm, carries it to the delivery table next
to the resident and puts it down.

What this does and does not claim:
  * WHERE to reach comes only from perception + head depth: the support region
    of the final OpenCV 5 result is back-projected with the head depth image and
    the camera pose from forward kinematics.  The face plane fitted to those
    points gives the box's thickness direction, so the gripper closes across the
    thickness.  No object pose, ID or segmentation is read for this.
  * Planning uses the robot model, the STATIC furniture (as a map) and the head
    depth points of other objects (obstacles).  The held object is modelled as the
    box estimated from depth.  Collision checks are signed-distance queries
    (mj_geomDistance), independent of MuJoCo's contact filtering, and include arm
    self-collision.  The base follows a grid roadmap with reverse-out manoeuvres.
  * Execution is KINEMATIC: joint positions are set, not torque-controlled.
    Each finger closes until it touches something; the object both fingers touch
    is attached rigidly to the wrist (stand-in for a stable grasp).  Placing
    lowers until the held object touches the table (stand-in for force sensing).
    Nothing looks the target up by name: if perception picked the wrong box,
    the robot carries the wrong box.
  * Every executed frame is measured against the simulator's full geometry and
    the minimum clearances are logged (`execution_check`), so the trace shows
    whether the motion really was collision-free.
  * No grasp-quality metric, no physics, no force control.  grasp_authorised
    stays False in the agent result.
  * If the agent abstained, the arm does not move: "cannot verify, please help".
Ground truth (which object ended up on the delivery table) is computed only
after the fact, for offline scoring, under `offline_scoring_gt`.

Quick headless check (agent + fetch, no video), from ~/Desktop/caremate:
  MUJOCO_GL=egl python -m caremate.fetch_v0_7 --target crunch_peanut
Full demo with video: python -m caremate.demo_v0_7 --target crunch_peanut --record outputs/fetch.mp4
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from caremate.nbv_v0_7 import STATIONS
from caremate.run_sim import hold_target
from caremate.workspace_v0_7 import WORKSPACES, camera_pose_from_mujoco, polygon_mask

TABLE_Z = WORKSPACES["main_table"]["z"][0]
ARM = tuple(f"arm_right_{i}_joint" for i in range(1, 8))
FINGERS = ("gripper_right_right_finger_joint", "gripper_right_left_finger_joint")
FINGER_BODIES = ("gripper_right_right_finger_link", "gripper_right_left_finger_link")
# Right-arm home (tuck) pose, chosen by hand (2026-10-21) because the mirrored-left tuck
# (-1.0853, 1.4705, -3.533, 1.709, -1.5701, 1.3716, 0) sits on joint 1/3 limits and could only be unfolded
# by swinging the arm ~30-70 cm behind the robot.  --check-tuck: inside joint limits (2 and 3 at 1.569,
# model limit 1.5708; joint 3 = 90 deg is above the 45 deg of the newer dual-arm URDF), self clearance
# 2.1 cm, furniture 27 cm, in the head image in 47/54 search views but inside the perception ROI in only
# 4/54 views at <= 0.01 % of the ROI (a few pixels on the ROI margin).
TUCK_RIGHT = (0.0, 1.569, 1.569, 1.71, -1.57, 1.37, 0.0)
# Left arm: mirror image of TUCK_RIGHT, from  python scripts/mirror_tuck.py --to-left <TUCK_RIGHT>
# (link positions matched to the mirrored right arm, max error 2.5 mm).
TUCK_LEFT = (-0.0198, 1.5674, 1.5892, 1.7101, -1.5678, 1.3766, 0.0)
FINGER_OPEN = 0.045
MANIP_TORSO = 0.35
DELIVERY_SITE = "delivery_goal"
FALLBACK_DELIVERY = (2.0, 0.4, 0.77)
FPS = 30
ROI_OCCLUSION_MAX = 0.005   # arms may cover at most 0.5 % of the perception ROI in a search view
BEHIND_ALLOWANCE = None   # m the elbow / wrist / fingers may move further back than in the tuck pose.
# None = no such constraint (default): on the real TIAGo model even 12 cm left RRT without a path from the
# tuck pose; instead the offline precompute keeps the SHORTEST of several paths, which removes big swings.
ARM3_UPPER = None   # set by --strict-limits (45 deg, PAL dual-arm URDF) - the model allows 90 deg
FETCH_VERSION = "fetch_v0_7 r31 (2026-10-21 20:40)"

# IK (uncalibrated, chosen for the demo)
TOL_POS = 0.010
TOL_ANG = math.radians(8)
TOL_ROLL = 0.15                   # |z| of the closing axis: fingers close horizontally
W_ANG, W_ROLL = 0.15, 0.15
# Clearances (m)
CLEAR_BASE = 0.04                 # robot / held box vs furniture while the base moves
CLEAR_ARM = 0.015                 # arm vs furniture
CLEAR_GRASP = 0.005               # hand vs table at the grasp / place pose
CLEAR_SELF = 0.005                # arm vs rest of the robot
CLEAR_DEPTH = 0.02                # arm / held box vs depth points of other objects
# Grasp geometry
ASSUMED_HALF_THICKNESS = 0.0325   # depth sees one face; assume <= 6.5 cm thick (book 5, cereal 5.6, Crunch 6.4)
SIDE_CLEAR = 0.005                # free space wanted on each side of that box between the fingers
FINGER_BASE_GAP = 0.008           # finger base stops this far outside the box edge
PRE_BACK = 0.12
LIFT = 0.10
MAX_APPROACH_DEV = math.radians(55)
# Base pose search
RADII = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
N_ANG = 24
YAW_OFFSETS = (0.30, 0.15, 0.45)
MAX_REACH_TRIALS = 60
# Ready / carry posture: TCP in the base frame (x forward, y left, z up), approach forward.
READY_POINTS = tuple((x, y, z) for z in (1.00, 1.08, 0.94, 1.16) for y in (-0.30, -0.40, -0.22)
                     for x in (0.40, 0.32, 0.48))
GRID_RES = 0.20

_KEEPALIVE = None          # set by run_fetch: keeps viewer / OpenCV window responsive while planning
_tick = [0]


def _keepalive(every=40):
    _tick[0] += 1
    if _KEEPALIVE is not None and _tick[0] % every == 0:
        _KEEPALIVE()


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _r(x, n=3):
    """JSON-safe rounding."""
    if isinstance(x, dict):
        return {k: _r(v, n) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return _r(x.tolist(), n)
    if isinstance(x, (list, tuple, set)):
        return [_r(v, n) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), n)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _id(m, kind, name):
    i = mujoco.mj_name2id(m, kind, name)
    if i < 0:
        raise KeyError(f"{name} not found in model")
    return i


_NUDGE = (np.array([1e-5, 0, 0]), np.array([0, 1e-5, 0]), np.array([0, 0, 1e-5]))


def geom_distance(m, d, g1, g2, dmax):
    """Signed geom distance (mj_geomDistance) with a guard for a MuJoCo 3.6 glitch: for some
    degenerate alignments it returns exactly 0.0 for clearly separated convex pairs (~5-8 % of random
    box-box poses in our tests).  On an exact 0.0: re-query with either geom nudged by 10 um; if that
    still says 0.0, query a ladder of small distmax values (a correct answer is min(dist, distmax), the
    glitch only reports 0.0) and keep the largest.  A real penetration stays <= 0 in every query, so the
    guard never hides a contact (checked against analytic box-box distances)."""
    g1, g2 = int(g1), int(g2)
    r = mujoco.mj_geomDistance(m, d, g1, g2, dmax, None)
    if r != 0.0:
        return r
    for g in (g1, g2):
        x0 = d.geom_xpos[g].copy()
        try:
            for e in _NUDGE:
                d.geom_xpos[g] = x0 + e
                r = mujoco.mj_geomDistance(m, d, g1, g2, dmax, None)
                if r != 0.0:
                    return r
        finally:
            d.geom_xpos[g] = x0
    for dm in (0.005, 0.02, 0.06, 0.15):
        if dm < dmax:
            r = max(r, mujoco.mj_geomDistance(m, d, g1, g2, dm, None))
    return r



# ---------------------------------------------------------------- concave-aware self-collision
# MuJoCo collides meshes as convex hulls.  After URDF import the fixed torso column is merged into
# base_link, and its hull fills the recess where the arm tucks (the tucked forearm is ~14 cm "inside"
# it although it visibly clears the torso).  For arm-vs-body pairs whose body geom is a mesh we therefore
# use the body's real (possibly concave) surface: points sampled on its triangles are tested against the
# arm link's convex hull.  The result is exact penetration depth when points are inside and a lower
# bound on the distance otherwise, so it is never less safe than the true geometry.
SURFACE_SPACING = 0.015


def hull_planes(m, g):
    """Outward planes (n, b) of a geom's convex shape in its local frame: n.x + b <= 0 inside."""
    if m.geom_type[g] == int(mujoco.mjtGeom.mjGEOM_MESH) and m.geom_dataid[g] >= 0:
        mid = int(m.geom_dataid[g])
        V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
        a = int(m.mesh_graphadr[mid])
        if a >= 0:
            nv, nf = int(m.mesh_graph[a]), int(m.mesh_graph[a + 1])
            faces = m.mesh_graph[a + 2 + 3 * nv + 3 * nf: a + 2 + 3 * nv + 6 * nf].reshape(nf, 3)
        else:                                           # no hull graph: fall back to the AABB
            faces = None
        if faces is not None and len(faces):
            P = V[faces]
            n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
            L = np.linalg.norm(n, axis=1)
            keep = L > 1e-12
            n, P = n[keep] / L[keep, None], P[keep]
            c = V.mean(0)
            flip = np.einsum("ij,ij->i", n, P[:, 0] - c) < 0
            n[flip] *= -1
            return n, -np.einsum("ij,ij->i", n, P[:, 0]), float(np.linalg.norm(V, axis=1).max())
    c0, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]           # box / primitive: its AABB (conservative)
    n = np.vstack([np.eye(3), -np.eye(3)])
    return n, -(n @ c0) - np.concatenate([h, h]), float(np.linalg.norm(np.abs(c0) + h))


def surface_samples(m, g, spacing=SURFACE_SPACING):
    """Points on a mesh geom's real triangles (local frame), ~`spacing` apart."""
    mid = int(m.geom_dataid[g])
    V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
    Fc = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
    P = V[Fc]
    area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    k = np.maximum(1, np.ceil(area / (0.5 * spacing ** 2))).astype(int)
    rng = np.random.default_rng(g)
    idx = np.repeat(np.arange(len(P)), k)
    u, v = rng.random(len(idx)), rng.random(len(idx))
    f = u + v > 1
    u[f], v[f] = 1 - u[f], 1 - v[f]
    T = P[idx]
    pts = T[:, 0] + u[:, None] * (T[:, 1] - T[:, 0]) + v[:, None] * (T[:, 2] - T[:, 0])
    return np.vstack([V, pts])


def concave_distance(d, g_convex, planes, g_concave, samples, dmax):
    """min over surface samples of g_concave of max_j(n_j.x + b_j) in g_convex's frame (see above)."""
    Rc, pc = d.geom_xmat[g_concave].reshape(3, 3), d.geom_xpos[g_concave]
    Ra, pa = d.geom_xmat[g_convex].reshape(3, 3), d.geom_xpos[g_convex]
    loc = ((samples @ Rc.T + pc) - pa) @ Ra
    n, b, radius = planes
    reach = radius + dmax                                          # bounding sphere of the convex shape
    near = np.einsum("ij,ij->i", loc, loc) < reach ** 2
    if not near.any():
        return dmax
    sd = np.max(loc[near] @ n.T + b, axis=1)
    return float(min(dmax, sd.min()))


def _lerp(a, b, n):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return [a + (b - a) * (i / n) for i in range(1, n + 1)]


# ---------------------------------------------------------------- robot state
@dataclass
class RobotState:
    x: float
    y: float
    yaw: float
    torso: float
    pan: float
    tilt: float
    q: tuple
    f: float                    # right finger (and left, unless fl is set)
    fl: float | None = None     # left finger while closing finger-by-finger

    @property
    def base(self):
        return (self.x, self.y, self.yaw)


# ---------------------------------------------------------------- model index + geometry queries
class Kin:
    """Robot structure, geom classes and signed-distance checks."""

    def __init__(self, model):
        m = self.m = model
        J = mujoco.mjtObj.mjOBJ_JOINT
        self.arm_j = [_id(m, J, n) for n in ARM]
        self.arm_adr = np.array([m.jnt_qposadr[j] for j in self.arm_j])
        rng = m.jnt_range[self.arm_j]
        self.lo, self.hi = rng[:, 0] + 1e-3, rng[:, 1] - 1e-3
        if ARM3_UPPER is not None:       # --strict-limits: plan within the dual-arm URDF's arm_3 range
            self.hi[2] = min(self.hi[2], ARM3_UPPER - 1e-3)
        self.finger_j = [_id(m, J, n) for n in FINGERS]
        self.finger_adr = np.array([m.jnt_qposadr[j] for j in self.finger_j])
        tj = _id(m, J, "torso_lift_joint")
        self.torso_adr = int(m.jnt_qposadr[tj])
        hj = [_id(m, J, n) for n in ("head_1_joint", "head_2_joint")]
        self.head_adr = [int(m.jnt_qposadr[j]) for j in hj]
        self.head_rng = m.jnt_range[hj]
        base_j = _id(m, J, "reference")
        self.base_adr = int(m.jnt_qposadr[base_j])
        base_body = int(m.jnt_bodyid[base_j])
        self.wrist = int(m.jnt_bodyid[self.arm_j[-1]])
        fb = [_id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in FINGER_BODIES]
        self.finger_body = fb
        collidable = (m.geom_contype | m.geom_conaffinity) != 0
        bid = m.geom_bodyid
        self.finger_geoms = [np.nonzero(bid == b)[0] for b in fb]
        if any(len(g) == 0 for g in self.finger_geoms):
            raise RuntimeError("finger bodies have no geoms")
        self.finger_col = [g[collidable[g]] if collidable[g].any() else g for g in self.finger_geoms]
        root = m.body_rootid[bid]
        self.robot_geom = root == m.body_rootid[base_body]
        not_plane = m.geom_type != int(mujoco.mjtGeom.mjGEOM_PLANE)
        # Furniture = geoms welded to the world (world body or jointless child bodies such as a table).
        # (body_rootid is NOT enough: a jointless child of the world has its own root id.)
        self.static_geom = (m.body_weldid[bid] == 0) & not_plane
        free = int(mujoco.mjtJoint.mjJNT_FREE)
        self.objects = {int(m.jnt_bodyid[j]): int(m.jnt_qposadr[j]) for j in range(m.njnt)
                        if m.jnt_type[j] == free and m.jnt_bodyid[j] != base_body}
        self.object_geom = np.isin(bid, list(self.objects))
        self.obj_geoms = {}
        for b in self.objects:
            g = np.nonzero(bid == b)[0]
            self.obj_geoms[b] = g[collidable[g]] if collidable[g].any() else g
        self.all_obj_geoms = np.concatenate(list(self.obj_geoms.values())) if self.objects else np.zeros(0, int)
        # Robot collision geoms (fall back to all robot geoms if none are collidable).
        rg = np.nonzero(self.robot_geom)[0]
        self.robot_col = rg[collidable[rg]] if collidable[rg].any() else rg
        arm_root = int(m.jnt_bodyid[self.arm_j[0]])
        in_arm = np.isin(bid, [b for b in range(m.nbody) if self._below(b, arm_root)])
        in_hand = np.isin(bid, [b for b in range(m.nbody) if self._below(b, self.wrist)])
        self.arm_col = self.robot_col[in_arm[self.robot_col]]
        self.hand_col = self.robot_col[in_hand[self.robot_col]]
        self.body_col = self.robot_col[~in_arm[self.robot_col]]
        self.static_obs = np.nonzero(self.static_geom)[0]
        # Self-collision pairs: right arm vs rest of the robot, minus the shoulder's own mount.
        mount = int(m.body_parentid[arm_root])
        near = {arm_root} | {b for b in range(m.nbody) if m.body_parentid[b] == arm_root}
        pairs = [(a, b) for a in self.arm_col for b in self.body_col
                 if not (bid[a] in near and bid[b] == mount)]
        self.selfA = np.array([p[0] for p in pairs], int)
        self.selfB = np.array([p[1] for p in pairs], int)
        self.self_thr = np.full(len(self.selfA), CLEAR_SELF)
        self.self_overlapping_in_tuck = 0
        self.disabled_pairs = []
        self.home_pose_overlaps = []
        # Task constraint for arm planning: elbow, wrist and fingers never further behind (base frame x)
        # than in the tuck pose minus BEHIND_ALLOWANCE (12 cm; 3 cm was too tight: unfolding TIAGo's
        # tuck needs the wrist to move back a little).  Without it RRT swings the arm round the back.
        self.base_body = base_body
        self.track_bodies = [int(m.jnt_bodyid[j]) for j in self.arm_j[2:]] + list(self.finger_body)
        self.front_min = None
        names = list(ARM) + list(FINGERS) + ["torso_lift_joint", "head_1_joint", "head_2_joint"]
        self.audit_names = names
        self.audit_j = [_id(m, J, n) for n in names]
        self.audit_adr = np.array([m.jnt_qposadr[j] for j in self.audit_j])
        self.audit_lo, self.audit_hi = m.jnt_range[self.audit_j, 0], m.jnt_range[self.audit_j, 1]
        mesh_t = int(mujoco.mjtGeom.mjGEOM_MESH)
        self.planes = {int(g): hull_planes(m, g) for g in set(self.selfA.tolist())}
        self.samples = {int(g): surface_samples(m, g) for g in set(self.selfB.tolist())
                        if m.geom_type[g] == mesh_t and m.geom_dataid[g] >= 0}
        self.n_concave_body_geoms = len(self.samples)
        # Telescopic torso: a fixed-column box on the base that sits inside the lifting shell.  Only its
        # segment below the shell is exposed; the rest can only be reached through the shell, which has
        # its own collision geoms.  So for such a geom only the exposed part counts (shell bottom = lowest
        # point of the torso-lift link, updated with the torso height).
        self.torso_body = int(m.jnt_bodyid[tj])
        self.shell_geoms = [int(g) for g in np.nonzero(m.geom_bodyid == self.torso_body)[0]]
        self.column = {}
        box_t = int(mujoco.mjtGeom.mjGEOM_BOX)
        if self.shell_geoms:
            d0 = mujoco.MjData(m)
            mujoco.mj_kinematics(m, d0)
            sh = np.concatenate([self.geom_points(d0, g) for g in self.shell_geoms])
            lo_xy, hi_xy = sh[:, :2].min(0), sh[:, :2].max(0)
            for g in set(self.selfB.tolist()):
                if m.geom_type[g] != box_t or m.geom_bodyid[g] != base_body:
                    continue
                c = self.geom_points(d0, g)
                inside = (c[:, :2] >= lo_xy - 0.01).all() and (c[:, :2] <= hi_xy + 0.01).all()
                if inside and c[:, 2].max() > sh[:, 2].min():
                    h = m.geom_size[g]
                    pts = []
                    for ax in range(3):            # surface samples of the box faces
                        o = [i for i in range(3) if i != ax]
                        nu, nv = (max(2, int(2 * h[o[0]] / SURFACE_SPACING) + 1),
                                  max(2, int(2 * h[o[1]] / SURFACE_SPACING) + 1))
                        U, V = np.meshgrid(np.linspace(-h[o[0]], h[o[0]], nu), np.linspace(-h[o[1]], h[o[1]], nv))
                        for sgn in (-1, 1):
                            P = np.zeros((U.size, 3))
                            P[:, o[0]], P[:, o[1]], P[:, ax] = U.ravel(), V.ravel(), sgn * h[ax]
                            pts.append(P)
                    self.column[int(g)] = np.concatenate(pts)
        self._shell_cache = {}
        # Diagnostic: can MuJoCo's contact filter produce robot-furniture contacts at all?
        sc = self.static_obs[collidable[self.static_obs]]
        ct, ca = m.geom_contype, m.geom_conaffinity
        self.contact_filter_robot_static = bool(any(
            (ct[a] & ca[b]) or (ct[b] & ca[a]) for a in self.robot_col for b in sc))
        self.obstacles = np.zeros((0, 3))
        self.target_pts = np.zeros((0, 3))     # estimated target box: obstacle until it is grasped

    def _below(self, b, anc):
        while b > 0:
            if b == anc:
                return True
            b = int(self.m.body_parentid[b])
        return False

    def gname(self, g):
        n = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g)
        return n or f"{self.bname(int(self.m.geom_bodyid[g]))}#g{g}"

    def bname(self, b):
        return mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"

    def shell_bottom(self, d):
        """World z of the lowest point of the torso-lift shell (depends on the torso height only)."""
        key = round(float(d.qpos[self.torso_adr]), 4), round(float(d.qpos[self.base_adr + 2]), 4)
        if key not in self._shell_cache:
            self._shell_cache[key] = float(min(self.geom_points(d, g)[:, 2].min() for g in self.shell_geoms))
        return self._shell_cache[key]

    def self_distance(self, d, a, b, dmax):
        """Arm geom a vs body geom b: exposed segment for a telescopic column, concave-aware for mesh
        body geoms, exact for other primitives."""
        b = int(b)
        if b in self.column:
            loc = self.column[b]
            z = (loc @ d.geom_xmat[b].reshape(3, 3).T + d.geom_xpos[b])[:, 2]
            exposed = loc[z < self.shell_bottom(d) - 0.003]
            if not len(exposed):
                return dmax
            return concave_distance(d, int(a), self.planes[int(a)], b, exposed, dmax)
        if b in self.samples:
            return concave_distance(d, int(a), self.planes[int(a)], b, self.samples[b], dmax)
        return geom_distance(self.m, d, a, b, dmax)

    def set_front_limit(self, d):
        """Record how far back the arm is in the tuck pose (d must hold the tuck pose)."""
        self.front_min = None if BEHIND_ALLOWANCE is None else self._x_base(d) - BEHIND_ALLOWANCE

    def _x_base(self, d):
        R, p = d.xmat[self.base_body].reshape(3, 3), d.xpos[self.base_body]
        return ((d.xpos[self.track_bodies] - p) @ R)[:, 0]

    def backward_excursion(self, d, st, path):
        """Largest distance (m) that elbow / wrist / fingers go further back (base-frame x) than in the
        tuck pose, along a joint path.  0 = never behind where they start."""
        put(self, d, replace(st, q=TUCK_RIGHT))
        x0 = self._x_base(d)
        worst = 0.0
        for q in path:
            put(self, d, replace(st, q=tuple(q)))
            worst = max(worst, float(np.max(x0 - self._x_base(d))))
        return worst

    def behind(self, d):
        if self.front_min is None:
            return None
        x = self._x_base(d)
        i = int(np.argmin(x - self.front_min))
        return None if x[i] >= self.front_min[i] else self.bname(self.track_bodies[i])

    def set_self_baseline(self, ds):
        """Self-collision thresholds from the home (tuck) pose.  Nothing is disabled: a home pose that is
        itself in collision is a bug in the home pose (logged as home_pose_overlaps), not in the checker.
        Pairs that overlap a little at home may get at most 5 mm worse; all others need CLEAR_SELF."""
        base = np.full(len(self.selfA), np.inf)
        for d in ds:
            for i, (a, b) in enumerate(zip(self.selfA, self.selfB)):
                base[i] = min(base[i], self.self_distance(d, a, b, 0.3))
        disable = np.zeros(len(base), bool)     # nothing disabled: the old right tuck was REALLY in the column
        deep = base < -0.01
        self.disabled_pairs = []
        self.home_pose_overlaps = [f"{self.gname(a)}|{self.gname(b)} ({d:+.3f})"
                                   for a, b, d in zip(self.selfA[deep], self.selfB[deep], base[deep])]
        keep = ~disable
        self.selfA, self.selfB, base = self.selfA[keep], self.selfB[keep], base[keep]
        self.self_thr = np.minimum(CLEAR_SELF, base - 0.005)
        self.self_overlapping_in_tuck = int((base < CLEAR_SELF).sum())

    def self_margin(self, d):
        """(worst distance-minus-threshold, pair, distance); negative = self-collision."""
        if not len(self.selfA):
            return 1.0, None, 1.0
        A, B, thr = self.selfA, self.selfB, self.self_thr
        c = (np.linalg.norm(d.geom_xpos[A] - d.geom_xpos[B], axis=1)
             - self.m.geom_rbound[A] - self.m.geom_rbound[B])
        best = (1.0, None, 1.0)
        for i in np.nonzero(c < thr + 0.05)[0]:
            dist = self.self_distance(d, A[i], B[i], max(0.05, thr[i] + 0.05))
            if dist - thr[i] < best[0]:
                best = (dist - thr[i], (int(A[i]), int(B[i])), dist)
        return best

    # -- signed distances (geom poses must be current)
    def _pairs_min(self, d, A, B, dmax):
        if len(A) == 0:
            return dmax, None
        c = (np.linalg.norm(d.geom_xpos[A] - d.geom_xpos[B], axis=1)
             - self.m.geom_rbound[A] - self.m.geom_rbound[B])
        best = (dmax, None)
        for i in np.nonzero(c < dmax)[0]:
            dist = geom_distance(self.m, d, A[i], B[i], dmax)
            if dist < best[0]:
                best = (dist, (int(A[i]), int(B[i])))
        return best

    def _sets_min(self, d, A, B, dmax):
        if len(A) == 0 or len(B) == 0:
            return dmax, None
        AA, BB = np.meshgrid(A, B, indexing="ij")
        return self._pairs_min(d, AA.ravel(), BB.ravel(), dmax)

    def _pts_in_geoms(self, d, pts, geoms, margin):
        """First geom whose (local, expanded) AABB contains any point."""
        if len(pts) == 0:
            return None
        for g in geoms:
            rb = self.m.geom_rbound[g]
            near = pts[np.linalg.norm(pts - d.geom_xpos[g], axis=1) < rb + margin] if rb > 0 else pts
            if not len(near):
                continue
            R = d.geom_xmat[g].reshape(3, 3)
            loc = (near - d.geom_xpos[g]) @ R - self.m.geom_aabb[g, :3]
            if (np.abs(loc) < self.m.geom_aabb[g, 3:] + margin).all(1).any():
                return int(g)
        return None

    def violations(self, d, static_margin, self_check=True, depth=True, carried=None,
                   carried_static=True, robot_static=True, hand_static_margin=None, target=True):
        """Reasons the current pose is unsafe ([] = clear).  carried: world points of the held-box model."""
        out = []
        _keepalive()
        if robot_static:
            if hand_static_margin is None:
                dist, p = self._sets_min(d, self.robot_col, self.static_obs, static_margin + 0.05)
                if dist < static_margin:
                    out.append(f"robot~furniture {self.gname(p[0])}|{self.gname(p[1])} {dist:+.3f}")
            else:   # grasp/place pose: hand may come closer to the table than the rest of the arm
                rest = np.setdiff1d(self.robot_col, self.hand_col)
                for A, mg in ((rest, static_margin), (self.hand_col, hand_static_margin)):
                    dist, p = self._sets_min(d, A, self.static_obs, mg + 0.05)
                    if dist < mg:
                        out.append(f"robot~furniture {self.gname(p[0])}|{self.gname(p[1])} {dist:+.3f}")
        if self_check:
            marg, p, dist = self.self_margin(d)
            if marg < 0:
                out.append(f"self {self.gname(p[0])}|{self.gname(p[1])} {dist:+.3f}")
            b = self.behind(d)
            if b is not None:
                out.append(f"behind-the-robot {b}")
        if depth and len(self.obstacles):
            g = self._pts_in_geoms(d, self.obstacles, self.arm_col, CLEAR_DEPTH)
            if g is not None:
                out.append(f"depth-obstacle near {self.gname(g)}")
        if target and len(self.target_pts):
            g = self._pts_in_geoms(d, self.target_pts, self.arm_col, CLEAR_DEPTH)
            if g is not None:
                out.append(f"target-box (estimated) hit by {self.gname(g)}")
        if carried is not None:
            if carried_static:
                g = self._pts_in_geoms(d, carried, self.static_obs, static_margin)
                if g is not None:
                    out.append(f"held-box~furniture {self.gname(g)}")
            g = self._pts_in_geoms(d, carried, np.setdiff1d(self.body_col, self.hand_col), 0.0)
            if g is not None:
                out.append(f"held-box~robot {self.gname(g)}")
            if depth and len(self.obstacles):
                dd = np.linalg.norm(carried[:, None, :] - self.obstacles[None, ::3, :], axis=2)
                if dd.min() < CLEAR_DEPTH:
                    out.append("held-box~depth-obstacle")
        return out

    # -- tool frame
    def tool(self, d):
        """TCP = midpoint between the fingers; c = closing axis from the finger slide-joint axes;
        a = approach (wrist -> fingers, orthogonal to c)."""
        r = d.geom_xpos[self.finger_geoms[0]].mean(0)
        l = d.geom_xpos[self.finger_geoms[1]].mean(0)
        p = 0.5 * (r + l)
        c = d.xaxis[self.finger_j[1]] - d.xaxis[self.finger_j[0]]
        if np.linalg.norm(c) < 0.5:
            c = d.xaxis[self.finger_j[1]].copy()
        c = c / np.linalg.norm(c)
        a = p - d.xpos[self.wrist]
        a = a - (a @ c) * c
        a = a / max(np.linalg.norm(a), 1e-9)
        return p, a, c

    def geom_points(self, d, g):
        """World points describing a geom's extent: mesh vertices for meshes, box corners otherwise."""
        m = self.m
        R = d.geom_xmat[g].reshape(3, 3)
        if m.geom_type[g] == int(mujoco.mjtGeom.mjGEOM_MESH) and m.geom_dataid[g] >= 0:
            mid = int(m.geom_dataid[g])
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            return d.geom_xpos[g] + v @ R.T
        c0, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c0
        return d.geom_xpos[g] + corners @ R.T

    def finger_span_along(self, d):
        """(back, tip): extent of the finger collision geoms behind / beyond the TCP along the approach."""
        p, a, _ = self.tool(d)
        proj = np.concatenate([(self.geom_points(d, g) - p) @ a for g in np.concatenate(self.finger_col)])
        return float(-proj.min()), float(proj.max())

    def hand_geometry(self, d):
        """Measured once with the gripper open, relative to the TCP (s = along the approach,
        c = along the closing axis):
          palm_front: how far the non-finger hand geoms reach along the approach (< 0: behind the TCP);
          gap(s):     free half-width between the fingers at each s, from the 2-D convex hull of each
                      finger's collision vertices (MuJoCo collides meshes as convex hulls, which can be
                      fatter than the visible finger, e.g. an L-shaped finger near its base)."""
        import cv2
        p, a, c = self.tool(d)
        fingers = set(np.concatenate(self.finger_col).tolist())
        palm = [g for g in self.hand_col if int(g) not in fingers]
        palm_front = max(float(((self.geom_points(d, g) - p) @ a).max()) for g in palm) if palm else -1.0
        hulls = []
        for k in (0, 1):
            v = np.concatenate([self.geom_points(d, g) - p for g in self.finger_col[k]])
            hulls.append(cv2.convexHull(np.stack([v @ a, v @ c], 1).astype(np.float32)).reshape(-1, 2))
        s_grid = np.linspace(-0.15, 0.15, 121)
        gap = np.full(len(s_grid), np.nan)          # nan: not between both fingers
        for i, sv in enumerate(s_grid):
            inner = []
            for k, h in enumerate(hulls):
                cs = []
                for (s0, c0), (s1, c1) in zip(h, np.roll(h, -1, 0)):      # hull edges crossing s = sv
                    if (s0 - sv) * (s1 - sv) <= 0 and s0 != s1:
                        cs.append(c0 + (c1 - c0) * (sv - s0) / (s1 - s0))
                if cs:   # right finger (k=0) sits at -c: its inner face is max(c); left: min(c)
                    inner.append(-max(cs) if k == 0 else min(cs))
            if len(inner) == 2:
                gap[i] = min(inner)
        return palm_front, s_grid, gap

    def _eval(self, d, q, p_star, a_star):
        d.qpos[self.arm_adr] = q
        mujoco.mj_kinematics(self.m, d)
        p, a, c = self.tool(d)
        r = np.concatenate([p - p_star, W_ANG * (a - a_star), [W_ROLL * c[2]]])
        err = {"pos_m": float(np.linalg.norm(p - p_star)),
               "ang_rad": float(math.acos(np.clip(a @ a_star, -1, 1))),
               "roll_z": float(abs(c[2]))}
        err["ok"] = err["pos_m"] < TOL_POS and err["ang_rad"] < TOL_ANG and err["roll_z"] < TOL_ROLL
        err["score"] = err["pos_m"] + 0.1 * err["ang_rad"] + 0.05 * err["roll_z"]
        return r, err

    def ik(self, d, p_star, a_star, seeds, iters=200, lam=0.03):
        """Damped-least-squares IK (finite-difference Jacobian) on the right arm, joint limits enforced,
        null-space motion towards mid-range.  d must hold the base / torso / finger state."""
        p_star = np.asarray(p_star, float)
        a_star = np.asarray(a_star, float) / np.linalg.norm(a_star)
        mid, span = (self.lo + self.hi) / 2, (self.hi - self.lo)
        best = None
        for q0 in seeds:
            q = np.clip(np.asarray(q0, float), self.lo, self.hi)
            for it in range(iters):
                r, err = self._eval(d, q, p_star, a_star)
                if err["ok"]:
                    break
                if it % 25 == 0:
                    _keepalive(1)
                Jm = np.empty((len(r), 7))
                for k in range(7):
                    dq = q.copy()
                    dq[k] += 1e-4
                    Jm[:, k] = (self._eval(d, dq, p_star, a_star)[0] - r) / 1e-4
                Jp = Jm.T @ np.linalg.inv(Jm @ Jm.T + lam ** 2 * np.eye(len(r)))
                # Joint-limit avoidance in the null space, faded out near the goal (with damping the
                # projector is not exact and would otherwise hold the hand ~1 cm off target).
                k_null = 0.3 * min(1.0, (err["pos_m"] / 0.03) ** 2)
                step = -Jp @ r + (np.eye(7) - Jp @ Jm) @ (k_null * (mid - q) / span)
                n = np.linalg.norm(step)
                if n > 0.25:
                    step *= 0.25 / n
                q = np.clip(q + step, self.lo, self.hi)
            _, err = self._eval(d, q, p_star, a_star)
            if best is None or err["score"] < best[1]["score"]:
                best = (q.copy(), err)
            if err["ok"]:
                return q.copy(), err
        return None, best[1]

    def finger_touch(self, d, gap=0.001):
        """Movable objects within `gap` of each finger (tactile stand-in)."""
        touch = [set(), set()]
        for k in (0, 1):
            for b, og in self.obj_geoms.items():
                dist, _ = self._sets_min(d, self.finger_col[k], og, gap + 0.01)
                if dist <= gap:
                    touch[k].add(b)
        return touch


def read_state(kin: Kin, d) -> RobotState:
    a = kin.base_adr
    w, x, y, z = d.qpos[a + 3:a + 7]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return RobotState(float(d.qpos[a]), float(d.qpos[a + 1]), yaw, float(d.qpos[kin.torso_adr]),
                      float(d.qpos[kin.head_adr[0]]), float(d.qpos[kin.head_adr[1]]),
                      tuple(float(v) for v in d.qpos[kin.arm_adr]), float(d.qpos[kin.finger_adr[0]]))


class Attachment:
    """Kinematic stand-in for a stable grasp: object pose fixed relative to the wrist.

    Reads the object's pose ONCE, at the moment both fingers touch it - this is the
    physical consequence of closing the gripper, not perception.
    """

    def __init__(self, kin: Kin, d, body: int):
        if kin.m.body_parentid[body] != 0:
            raise RuntimeError(f"{kin.bname(body)} is not a top-level free body")
        self.kin, self.body, self.adr = kin, body, kin.objects[body]
        Rw, pw = d.xmat[kin.wrist].reshape(3, 3), d.xpos[kin.wrist]
        self.p_rel = Rw.T @ (d.xpos[body] - pw)
        self.R_rel = Rw.T @ d.xmat[body].reshape(3, 3)

    def apply(self, d):
        Rw, pw = d.xmat[self.kin.wrist].reshape(3, 3), d.xpos[self.kin.wrist]
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, (Rw @ self.R_rel).ravel())
        d.qpos[self.adr:self.adr + 3] = pw + Rw @ self.p_rel
        d.qpos[self.adr + 3:self.adr + 7] = quat


def box_points(centre, n_xy, half_w, half_h, half_t=ASSUMED_HALF_THICKNESS, k=5):
    """Surface samples of the box estimated from head depth (n = face normal, t along the face)."""
    n = np.array([n_xy[0], n_xy[1], 0.0])
    t = np.array([-n[1], n[0], 0.0])
    z = np.array([0.0, 0.0, 1.0])
    g = np.linspace(-1, 1, k)
    pts = []
    for i in g:
        for j in g:
            for s in (-1, 1):   # the six faces
                pts += [centre + s * half_t * n + i * half_w * t + j * half_h * z,
                        centre + i * half_t * n + s * half_w * t + j * half_h * z,
                        centre + i * half_t * n + j * half_w * t + s * half_h * z]
    return np.array(pts)


class HeldBox:
    """Perception model of the held object: the box estimated from head depth, fixed to the wrist."""

    def __init__(self, kin: Kin, d, centre, n_xy, half_w, half_h, half_t=ASSUMED_HALF_THICKNESS):
        pts = box_points(centre, n_xy, half_w, half_h, half_t)
        self.kin = kin
        Rw, pw = d.xmat[kin.wrist].reshape(3, 3), d.xpos[kin.wrist]
        self.local = (pts - pw) @ Rw
        self.half_h = half_h

    def world(self, d):
        Rw, pw = d.xmat[self.kin.wrist].reshape(3, 3), d.xpos[self.kin.wrist]
        return pw + self.local @ Rw.T


def put(kin: Kin, d, st: RobotState, attach: Attachment | None = None):
    """Planning-only state setter (kinematics only, no actuators)."""
    a = kin.base_adr
    d.qpos[a:a + 2] = (st.x, st.y)
    d.qpos[a + 3:a + 7] = (math.cos(st.yaw / 2), 0, 0, math.sin(st.yaw / 2))
    d.qpos[kin.torso_adr] = st.torso
    d.qpos[kin.head_adr[0]], d.qpos[kin.head_adr[1]] = st.pan, st.tilt
    d.qpos[kin.arm_adr] = st.q
    d.qpos[kin.finger_adr] = (st.f, st.f if st.fl is None else st.fl)
    mujoco.mj_kinematics(kin.m, d)
    if attach is not None:
        attach.apply(d)
        mujoco.mj_kinematics(kin.m, d)


def show(cam, kin: Kin, st: RobotState, attach: Attachment | None = None):
    """Display state setter: through hold_target / set_robot so actuator targets stay consistent."""
    d = cam.data
    for name, v in zip(ARM, st.q):
        hold_target(kin.m, d, name, v)
    for name, v in zip(FINGERS, (st.f, st.f if st.fl is None else st.fl)):
        hold_target(kin.m, d, name, v)
    cam.set_robot(st.x, st.y, st.yaw, st.torso, st.pan, st.tilt)
    if attach is not None:
        attach.apply(d)
        mujoco.mj_forward(kin.m, d)


def _plan_copy(cam):
    cam.plan_data.qpos[:] = cam.data.qpos
    mujoco.mj_kinematics(cam.model, cam.plan_data)
    return cam.plan_data


def look_at(kin: Kin, st: RobotState, cam_pos, p) -> RobotState:
    """Head pan/tilt towards p (TIAGo convention: pan + = left, tilt - = down)."""
    pan = _wrap(math.atan2(p[1] - st.y, p[0] - st.x) - st.yaw)
    tilt = -math.atan2(cam_pos[2] - p[2], math.hypot(p[0] - cam_pos[0], p[1] - cam_pos[1]))
    (plo, phi), (tlo, thi) = kin.head_rng
    return replace(st, pan=float(np.clip(pan, plo, phi)), tilt=float(np.clip(tilt, tlo, thi)))


# ---------------------------------------------------------------- perception -> 3D
def estimate_grasp_point(bbox, polygon, p_base, pose, depth, W, H):
    """Object centre, height and face plane from the LAST observation (head depth inside the
    OpenCV 5 support region).  Returns (centre, info, points)."""
    info = {"method": None, "n_depth_points": 0}
    if depth is None or bbox is None:
        info["reason"] = "no depth or no support region"
        return None, info, None
    Hd, Wd = depth.shape[:2]
    sx, sy = Wd / W, Hd / H
    if polygon is not None and len(polygon) >= 3:
        mask = polygon_mask(np.asarray(polygon, np.float32) * np.float32([sx, sy]), Wd, Hd) > 0
    else:
        mask = np.zeros((Hd, Wd), bool)
    if not mask.any():
        x0, y0, x1, y1 = bbox
        mask[max(0, int(y0 * sy)):int(math.ceil(y1 * sy)), max(0, int(x0 * sx)):int(math.ceil(x1 * sx))] = True
    pts_all = depth_points(depth, pose)
    pts, z = pts_all[mask.ravel()], depth.ravel()[mask.ravel()]
    ws = WORKSPACES["main_table"]
    keep = ((pts[:, 2] > TABLE_Z + 0.01) & (pts[:, 2] < TABLE_Z + 0.45) &
            (pts[:, 0] > ws["x"][0]) & (pts[:, 0] < ws["x"][1]) &
            (pts[:, 1] > ws["y"][0]) & (pts[:, 1] < ws["y"][1]) & (z > 0.05))
    if p_base is not None:
        keep &= np.hypot(pts[:, 0] - p_base[0], pts[:, 1] - p_base[1]) < 0.20
    pts = pts[keep]
    info["n_depth_points"] = int(len(pts))
    if len(pts) < 15:
        if p_base is None:
            info["reason"] = "too few depth points and no polygon base"
            return None, info, pts_all
        info["method"] = "polygon_base_fallback (+6 cm)"
        return np.array([p_base[0], p_base[1], TABLE_Z + 0.06]), info, pts_all
    xy = np.median(pts[:, :2], 0)
    top = float(np.percentile(pts[:, 2], 95))
    h = top - TABLE_Z
    info.update(method="head_depth_in_support_region", visible_centre_xy=_r(xy), height_m=round(h, 3))
    centre = np.array([xy[0], xy[1], TABLE_Z + float(np.clip(0.5 * h, 0.035, 0.12))])
    # Face plane of the matched reference face: normal = thickness direction, extent = face width.
    q = pts - pts.mean(0)
    _, evecs = np.linalg.eigh(q.T @ q / len(q))
    n = evecs[:, 0]
    if abs(n[2]) < 0.5:
        n = np.array([n[0], n[1], 0.0]) / np.hypot(n[0], n[1])
        if n @ (centre - np.asarray(pose["pos"], float)) < 0:
            n = -n                                    # into the object
        t = np.array([-n[1], n[0], 0.0])
        s = (pts[:, :2] - xy) @ t[:2]
        half_w = 0.5 * float(np.percentile(s, 97) - np.percentile(s, 3))
        centre[:2] += ASSUMED_HALF_THICKNESS * n[:2]
        info.update(face_normal_xy=_r(n[:2]), face_half_width_m=round(half_w, 3),
                    assumed_half_thickness_m=ASSUMED_HALF_THICKNESS)
    return centre, info, pts_all


def depth_points(depth, pose):
    """World points of every depth pixel (z-depth image, MuJoCo camera convention)."""
    Hd, Wd = depth.shape[:2]
    v, u = np.mgrid[0:Hd, 0:Wd]
    z = depth.astype(float)
    f = 0.5 * Hd / math.tan(math.radians(pose["fovy_deg"]) / 2)
    cam_pts = np.stack([(u + 0.5 - Wd / 2) / f * z, -(v + 0.5 - Hd / 2) / f * z, -z], -1).reshape(-1, 3)
    R = np.asarray(pose["mat"], float).reshape(3, 3)
    return np.asarray(pose["pos"], float) + cam_pts @ R.T


def obstacle_points(pts_all, centre, info, max_pts=3000):
    """Depth points of OTHER objects on the main table (target's estimated box removed)."""
    ws = WORKSPACES["main_table"]
    p = pts_all[(pts_all[:, 2] > TABLE_Z + 0.015) & (pts_all[:, 2] < TABLE_Z + 0.45) &
                (pts_all[:, 0] > ws["x"][0]) & (pts_all[:, 0] < ws["x"][1]) &
                (pts_all[:, 1] > ws["y"][0]) & (pts_all[:, 1] < ws["y"][1])]
    if "face_normal_xy" in info:
        n = np.array([*info["face_normal_xy"], 0.0])
        t = np.array([-n[1], n[0], 0.0])
        rel = p - centre
        target = (np.abs(rel @ t) < info["face_half_width_m"] + 0.03) & (np.abs(rel @ n) < 0.075)
    else:
        target = np.hypot(p[:, 0] - centre[0], p[:, 1] - centre[1]) < 0.12
    p = p[~target]
    if len(p) > max_pts:
        p = p[np.random.default_rng(0).choice(len(p), max_pts, replace=False)]
    return p


def surface_below(kin: Kin, d, xy, z0=1.6):
    """Height of the static furniture under xy (one-off static-map query) and that geom."""
    mujoco.mj_kinematics(kin.m, d)
    pnt = np.array([xy[0], xy[1], z0], float)
    vec = np.array([0.0, 0.0, -1.0])
    gid = np.array([-1], np.int32)
    for _ in range(10):
        dist = mujoco.mj_ray(kin.m, d, pnt, vec, None, 1, -1, gid)
        if dist < 0:
            return None, None
        hit = pnt + dist * vec
        if kin.static_geom[gid[0]]:
            return float(hit[2]), int(gid[0])
        pnt = hit + 1e-3 * vec
    return None, None


# ---------------------------------------------------------------- planning
def _reason_key(v):
    """'self a|b +0.01' -> 'self a|b'; keeps the colliding pair, drops the number."""
    parts = v.split()
    return " ".join(parts[:-1]) if parts and parts[-1][:1] in "+-" else v


def _note(stats, v):
    stats["last_reason"] = v[0]
    r = stats.setdefault("blocked_by", {})
    k = _reason_key(v[0])
    r[k] = r.get(k, 0) + 1


MAX_JOINT_STEP = 0.05     # rad between collision-checked configurations


def _densify(start, qs):
    """Insert configurations so consecutive ones differ by <= MAX_JOINT_STEP in every joint
    (a check only at the given samples can miss a link sweeping through the torso between them)."""
    out, prev = [], None if start is None else np.asarray(start, float)
    for q in qs:
        q = np.asarray(q, float)
        if prev is not None:
            n = int(math.ceil(float(np.max(np.abs(q - prev))) / MAX_JOINT_STEP))
            out += [prev + (q - prev) * (i / n) for i in range(1, n)]
        out.append(q)
        prev = q
    return out


def _arm_path_ok(kin, d, st, qs, attach=None, held=None, start=None, **kw):
    for q in _densify(start, qs):
        put(kin, d, replace(st, q=tuple(q)), attach)
        v = kin.violations(d, carried=None if held is None else held.world(d), **kw)
        if v:
            return v
    return []


def _cartesian(kin, d, st, q0, p0, p1, a, n):
    """IK along a straight TCP line, seeded sequentially; None if any point fails."""
    qs, q = [], q0
    for p in _lerp(p0, p1, n):
        put(kin, d, replace(st, q=tuple(q)))
        q, e = kin.ik(d, p, a, [q], iters=120)
        if q is None:
            return None
        qs.append(q)
    return qs


def _transfer(kin, d, st, q0, p1, a1, via_up=0.08, n=10, attach=None):
    """Hand moves on a straight line (via a point above the goal) while the approach heading turns
    smoothly; IK per point seeded with the previous one.  Returns joint waypoints or None."""
    put(kin, d, replace(st, q=tuple(q0)), attach)
    p0, a0, _ = kin.tool(d)
    h0, h1 = math.atan2(a0[1], a0[0]), math.atan2(a1[1], a1[0])
    dh = _wrap(h1 - h0)
    via = np.asarray(p1, float) + [0, 0, via_up]
    legs = [(p0, via, 0.0, 0.8), (via, np.asarray(p1, float), 0.8, 1.0)]
    qs, q = [], np.asarray(q0, float)
    for pa, pb, sa, sb in legs:
        m = n if sa == 0.0 else max(3, n // 3)
        for i in range(1, m + 1):
            u = i / m
            h = h0 + dh * (sa + (sb - sa) * u)
            a = np.array([math.cos(h), math.sin(h), 0.0])
            put(kin, d, replace(st, q=tuple(q)), attach)
            q, _ = kin.ik(d, pa + (pb - pa) * u, a, [q], iters=120)
            if q is None:
                return None
            qs.append(q)
    return qs


UNFOLD_ORDERS = (                       # joint groups moved one after another (0-based arm joints)
    ((0, 1, 2, 3, 4, 5, 6),),
    ((0, 1), (2, 3, 4, 5, 6)),
    ((3,), (0, 1), (2, 4, 5, 6)),
    ((1,), (0,), (3,), (2, 4, 5, 6)),
    ((0, 1, 2), (3, 4, 5, 6)),
    ((2, 4, 5, 6), (0, 1, 3)),
)


def _staged(q0, q1, order, n=8):
    qs, q = [], np.asarray(q0, float).copy()
    for group in order:
        tgt = q.copy()
        tgt[list(group)] = np.asarray(q1, float)[list(group)]
        qs += _lerp(q, tgt, n)
        q = tgt
    return qs


def _reach(kin, d, st, q0, p1, a1, via_up=0.08, attach=None, rng=None):
    """Hand path from q0 to (p1, a1): Cartesian transfer first; if its IK chain breaks, fall back to a
    multi-seed IK goal and a joint-space path (both are collision-checked by the caller)."""
    path = _transfer(kin, d, st, q0, p1, a1, via_up=via_up, attach=attach)
    if path is not None:
        return path
    rng = rng or np.random.default_rng(0)
    put(kin, d, replace(st, q=tuple(q0)), attach)
    q, _ = kin.ik(d, p1, a1, [q0, (kin.lo + kin.hi) / 2, *rng.uniform(kin.lo, kin.hi, (4, 7))])
    return None if q is None else _lerp(q0, q, 16)


def rrt_connect(ok, q_start, q_goal, lo, hi, rng, step=0.25, max_nodes=4000, time_budget=25.0, shortcuts=200):
    """RRT-Connect in joint space (Kuffner & LaValle 2000).  ok(q) -> True if q is collision-free.
    Every edge is checked at <= MAX_JOINT_STEP resolution.  Returns a shortcut-smoothed list of
    configurations from q_start to q_goal (inclusive), or None."""
    t_end = time.perf_counter() + time_budget

    def edge_ok(a, b):
        n = max(1, int(math.ceil(float(np.max(np.abs(b - a))) / MAX_JOINT_STEP)))
        return all(ok(a + (b - a) * (i / n)) for i in range(1, n + 1))

    def extend(T, P, q):
        """Grow tree T one step towards q.  Returns ('reached'|'advanced'|'trapped', index)."""
        i = int(np.argmin(np.max(np.abs(np.asarray(T) - q), 1)))
        a = T[i]
        dq = q - a
        dist = float(np.max(np.abs(dq)))
        new = q.copy() if dist <= step else a + dq * (step / dist)
        if not edge_ok(a, new):
            return "trapped", None
        T.append(new)
        P.append(i)
        return ("reached" if dist <= step else "advanced"), len(T) - 1

    def connect(T, P, q):
        while True:
            r, j = extend(T, P, q)
            if r != "advanced":
                return r, j

    q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
    if edge_ok(q_start, q_goal):
        return [q_start, q_goal]
    Ta, Pa, Tb, Pb = [q_start], [-1], [q_goal], [-1]
    swapped = False
    while len(Ta) + len(Tb) < max_nodes and time.perf_counter() < t_end:
        _keepalive(1)
        qr = rng.uniform(lo, hi) if rng.random() > 0.1 else Tb[0]
        r, ia = extend(Ta, Pa, qr)
        if r != "trapped":
            r2, ib = connect(Tb, Pb, Ta[ia])
            if r2 == "reached":
                def chain(T, P, i):
                    out = []
                    while i != -1:
                        out.append(T[i])
                        i = P[i]
                    return out
                a_part, b_part = chain(Ta, Pa, ia)[::-1], chain(Tb, Pb, ib)[1:]
                path = a_part + b_part
                if swapped:
                    path = path[::-1]
                # shortcut smoothing
                for _ in range(shortcuts):
                    if len(path) <= 2 or time.perf_counter() > t_end + 5:
                        break
                    i, j = sorted(rng.choice(len(path), 2, replace=False))
                    if j - i > 1 and edge_ok(path[i], path[j]):
                        path = path[:i + 1] + path[j:]
                return path
        Ta, Pa, Tb, Pb = Tb, Pb, Ta, Pa
        swapped = not swapped
    return None


def path_length(path):
    """Joint-space length: sum over segments of the largest joint change (what the motion 'costs')."""
    return float(sum(np.max(np.abs(np.asarray(b) - np.asarray(a))) for a, b in zip(path, path[1:])))


def plan_joint_path(ok, q_start, q_goal, lo, hi, seed=0, tries=2, budget=8.0):
    """Shortest of several RRT-Connect runs (a straight line is returned directly if it is free).
    RRT only guarantees *a* collision-free path; picking the shortest of a few and shortcutting
    removes most of the random detours."""
    best = None
    for k in range(tries):
        p = rrt_connect(ok, q_start, q_goal, lo, hi, np.random.default_rng(seed * 10 + k), time_budget=budget)
        if p is not None and (best is None or path_length(p) < path_length(best)):
            best = p
        if best is not None and len(best) == 2:          # straight line: cannot do better
            break
    return best


def find_ready(cam, kin: Kin, st: RobotState, rrt_cap=4, tries=2, budget=8.0, collect=1, deadline=None,
               verbose=False):
    """Ready / carry posture (TCP in front-right, approach forward, fingers closing horizontally) and a
    collision-free unfolding path from tuck.  With collect > 1, keeps searching and returns the
    shortest of the first `collect` solutions (used offline by --precompute-unfold).
    Returns (q_ready, unfold_path, report)."""
    found = []
    d = _plan_copy(cam)
    stm = replace(st, torso=MANIP_TORSO, q=TUCK_RIGHT, fl=None)        # fingers as they really are
    c, s = math.cos(st.yaw), math.sin(st.yaw)
    fwd = np.array([c, s, 0.0])
    rng = np.random.default_rng(0)
    report = []
    rrt_tries = 0
    t_start = time.perf_counter()

    def say(msg):
        if verbose:
            print(f"  [{time.perf_counter() - t_start:5.0f} s] {msg}", flush=True)

    for bx, by, bz in READY_POINTS:
        if deadline is not None and time.perf_counter() - t_start > deadline and found:
            say(f"time limit reached with {len(found)} path(s); keeping the shortest")
            break
        if deadline is not None and time.perf_counter() - t_start > 2 * deadline:
            say("time limit reached without a path")
            break
        p = np.array([st.x + c * bx - s * by, st.y + s * bx + c * by, bz])
        sols = []
        for seed in [TUCK_RIGHT, (kin.lo + kin.hi) / 2, *rng.uniform(kin.lo, kin.hi, (6, 7))]:
            put(kin, d, stm)
            q, _ = kin.ik(d, p, fwd, [seed])
            if q is not None and all(np.linalg.norm(q - o) > 0.3 for o in sols):
                sols.append(q)
            if len(sols) >= 3:
                break
        if not sols:
            report.append(((bx, by, bz), "ik_fail"))
            continue
        reason = None
        for q in sorted(sols, key=lambda q: float(np.abs(q - TUCK_RIGHT).sum())):
            put(kin, d, replace(stm, q=tuple(q)))
            v = kin.violations(d, CLEAR_ARM, depth=False, robot_static=False, target=False)
            if not v:            # the gripper opens at the ready pose
                put(kin, d, replace(stm, q=tuple(q), f=FINGER_OPEN))
                v = kin.violations(d, CLEAR_ARM, depth=False, robot_static=False, target=False)
            if v:
                reason = reason or f"ready pose: {v[0]}"
                continue
            for order in UNFOLD_ORDERS:
                path = _staged(TUCK_RIGHT, q, order)
                v = _arm_path_ok(kin, d, stm, path, start=TUCK_RIGHT, static_margin=CLEAR_ARM, depth=False,
                                 robot_static=False, target=False)
                if not v:
                    report.append(((bx, by, bz), f"ok (unfold order {order})"))
                    found.append((q, [np.asarray(TUCK_RIGHT, float)] + _densify(TUCK_RIGHT, path)))
                    break
                reason = reason or f"unfold: {v[0]}"
            else:
                v = True
            if not v:
                if len(found) >= collect:
                    break
                continue
            # No staged unfolding is collision-free: plan one (self-collision + furniture-free at this base).
            if rrt_tries >= rrt_cap:
                continue
            rrt_tries += 1
            say(f"ready {[bx, by, bz]}: RRT-Connect attempt {rrt_tries}/{rrt_cap} ({tries} runs x {budget:.0f} s max)")
            def ok(qq, _st=stm):
                put(kin, d, replace(_st, q=tuple(qq)))
                return not kin.violations(d, CLEAR_ARM, depth=False, target=False)
            t0 = time.perf_counter()
            path = plan_joint_path(ok, TUCK_RIGHT, q, kin.lo, kin.hi, seed=len(report), tries=tries,
                                   budget=budget)
            if path is not None:
                report.append(((bx, by, bz), f"ok (RRT-Connect, {len(path)} waypoints, "
                                             f"{path_length(path):.2f} rad, {time.perf_counter() - t0:.1f} s)"))
                found.append((q, _densify(None, path)))
                say(f"  found: {len(path)} waypoints, {path_length(path):.2f} rad")
                if len(found) >= collect:
                    break
                continue
            reason = f"{reason}; RRT-Connect found no path in {time.perf_counter() - t0:.0f} s"
            say("  no path")
        if len(found) >= collect:
            break
        report.append(((bx, by, bz), reason))
    if not found:
        return None, None, report
    # Choose the path whose arm goes least behind the robot (what looks wrong), then the shortest.
    stm_ = replace(st, torso=MANIP_TORSO, fl=None)
    scored = []
    for q, dense in found:
        back = kin.backward_excursion(d, stm_, dense)
        scored.append((round(back, 3), path_length(dense), q, dense))
        say(f"candidate: arm goes {100 * back:.1f} cm behind its tuck position, {path_length(dense):.2f} rad")
        report.append(("candidate", f"goes {100 * back:.1f} cm back, {path_length(dense):.2f} rad"))
    back, length, q, dense = min(scored, key=lambda t: (t[0], t[1]))
    report.append(("chosen", f"goes {100 * back:.1f} cm back, {length:.2f} rad"))
    return q, [np.asarray(x) for x in dense[1:]], report


def _base_candidates(st, gxy):
    cands = []
    for r in RADII:
        for k in range(N_ANG):
            th = 2 * math.pi * k / N_ANG
            b = gxy + r * np.array([math.cos(th), math.sin(th)])
            bearing = math.atan2(gxy[1] - b[1], gxy[0] - b[0])
            for off in YAW_OFFSETS:
                cands.append((math.hypot(b[0] - st.x, b[1] - st.y), float(b[0]), float(b[1]),
                              _wrap(bearing + off), bearing))
    return sorted(cands)


def plan_grasp(cam, kin, st, q_ready, unfold, centre, face, finger_back, finger_tip, stats,
               palm_front=-1.0, half_gap=0.04, s_pad=0.0):
    """Yield grasp plans (nearest base first): tuck -> ready (the precomputed / planned unfold path,
    gripper closed) -> open gripper -> pre-grasp (hand on a straight line) -> main (side approach) ->
    lift -> carry pose (= ready, with the held box checked)."""
    d = _plan_copy(cam)
    sides = None
    if face is not None:
        n = np.array([face["normal_xy"][0], face["normal_xy"][1], 0.0])
        t = np.array([-n[1], n[0], 0.0])
        sides = (t, -t)
    trials = 0
    for travel, bx, by, yaw, bearing in _base_candidates(st, centre[:2]):
        a = np.array([math.cos(bearing), math.sin(bearing), 0.0])
        if sides is not None:
            a = max(sides, key=lambda s: s @ a)
            if math.acos(np.clip(a @ [math.cos(bearing), math.sin(bearing), 0.0], -1, 1)) > MAX_APPROACH_DEV:
                stats["wrong_side"] += 1
                continue
        at_closed = replace(st, x=bx, y=by, yaw=yaw, torso=MANIP_TORSO, q=TUCK_RIGHT, fl=None)
        put(kin, d, at_closed)
        if kin.violations(d, CLEAR_BASE, self_check=False):
            stats["base_blocked"] += 1
            continue
        if trials >= MAX_REACH_TRIALS:
            break
        trials += 1
        # Unfolding at this base: self-collision was checked when the path was made (base-independent);
        # here furniture and other objects seen in depth.
        v = (_arm_path_ok(kin, d, at_closed, unfold, start=TUCK_RIGHT, static_margin=CLEAR_ARM, self_check=False)
             or _arm_path_ok(kin, d, replace(at_closed, f=FINGER_OPEN), [q_ready], static_margin=CLEAR_ARM))
        if v:
            stats["ready_blocked"] += 1
            _note(stats, v)
            continue
        at = replace(at_closed, f=FINGER_OPEN)        # gripper opens at the pre-grasp pose
        if sides is not None:
            # Across the thickness: centred on the assumed box behind the MEASURED face (the back face is
            # not seen); the fingers then have >= SIDE_CLEAR on each side wherever gap(s) >= need.
            gc = centre
            # Along the face: the box's near edge must stay where the fingers are fully open (the pads),
            # and ahead of the palm / wrist and the finger base.  e = edge position relative to the TCP.
            edge = gc - face["half_width"] * a
            e = max(s_pad + 0.005, palm_front + FINGER_BASE_GAP, -finger_back + FINGER_BASE_GAP)
            if finger_tip - e < 0.02:              # less than 2 cm of pad would hold the box
                stats["hand_too_short"] = stats.get("hand_too_short", 0) + 1
                continue
            depth_in = -e
            p_main = edge + depth_in * a
        else:
            p_main = centre + 0.02 * a
        # Pre-grasp: finger tips 3 cm outside the box edge (they reach finger_tip beyond the TCP).
        back_dist = max(PRE_BACK, (depth_in if sides is not None else 0.0) + finger_tip + 0.03)
        if sides is not None:
            stats["pad_overlap_m"] = round(finger_tip - e, 3)
        p_pre = p_main - back_dist * a
        # Ready -> pre-grasp: the hand moves on a straight line (via 3 cm above), heading turning smoothly.
        to_pre = _reach(kin, d, at, q_ready, p_pre, a, via_up=0.03, rng=np.random.default_rng(trials))
        if to_pre is None:
            stats["ik_fail"] += 1
            continue
        q_pre = to_pre[-1]
        v = _arm_path_ok(kin, d, at, to_pre, start=q_ready, static_margin=CLEAR_ARM)
        if v:
            stats["arm_blocked"] += 1
            _note(stats, v)
            continue
        approach = _cartesian(kin, d, at, q_pre, p_pre, p_main, a, max(6, int(back_dist / 0.02)))
        lift = _cartesian(kin, d, at, approach[-1], p_main, p_main + [0, 0, LIFT], a, 4) if approach else None
        if lift is None:
            stats["ik_fail"] += 1
            continue
        v = (_arm_path_ok(kin, d, at, approach, start=q_pre, static_margin=CLEAR_ARM, hand_static_margin=CLEAR_GRASP,
                          target=False) or
             _arm_path_ok(kin, d, at, lift, start=approach[-1], static_margin=CLEAR_ARM, hand_static_margin=CLEAR_GRASP,
                          depth=False, target=False))
        if not v:
            # From the lift on, the robot carries the box (modelled from depth, fixed to the wrist at the
            # grasp pose): it must not sweep into furniture or other objects on the way to the carry pose.
            put(kin, d, replace(at, q=tuple(approach[-1])))
            held_plan = HeldBox(kin, d, centre, face["normal_xy"] if face else (1.0, 0.0),
                                face["half_width"] if face else 0.04, max(0.02, centre[2] - TABLE_Z))
            # Hand moves in a straight line to the carry pose's TCP (box stays above the table), then a
            # short joint move onto q_ready itself (the fold path starts there).
            put(kin, d, replace(at, q=tuple(q_ready)))
            p_rdy, a_rdy, _ = kin.tool(d)
            carry = _reach(kin, d, at, lift[-1], p_rdy, a_rdy, via_up=0.0, rng=np.random.default_rng(trials))
            if carry is None:
                stats["ik_fail"] += 1
                continue
            carry = list(carry) + [np.asarray(q_ready, float)]
            v = _arm_path_ok(kin, d, at, carry, held=held_plan, start=lift[-1], static_margin=CLEAR_ARM,
                             target=False)
        if v:
            stats["arm_blocked"] += 1
            _note(stats, v)
            continue
        yield {"base": (bx, by, yaw), "approach": a, "q_pre": q_pre, "unfold_qs": unfold, "to_pre_qs": to_pre,
               "carry_qs": carry,
               "approach_qs": approach, "lift_qs": lift, "back_dist": back_dist,
               "p_main": p_main, "base_travel_m": travel}


def plan_place(cam, kin, st, q_ready, unfold, goal_xy, top_z, held, attach, stats, back_dist=PRE_BACK,
               f_closed=0.0):
    """Yield place plans: ready -> above goal -> descend (to table + 3 cm below the estimate) -> back off."""
    d = _plan_copy(cam)
    trials = 0
    for travel, bx, by, yaw, bearing in _base_candidates(st, goal_xy):
        a = np.array([math.cos(bearing), math.sin(bearing), 0.0])
        at = replace(st, x=bx, y=by, yaw=yaw, q=tuple(q_ready))
        put(kin, d, at, attach)
        if kin.violations(d, CLEAR_BASE, self_check=False, carried=held.world(d)):
            stats["base_blocked"] += 1
            continue
        if trials >= MAX_REACH_TRIALS:
            break
        trials += 1
        # TCP height so that the held box's bottom touches the table top (box model from depth).
        put(kin, d, at, attach)
        tcp, a_r, c_r = kin.tool(d)
        hw = held.world(d)
        bottom = hw[:, 2].min()
        # The box centre (not the TCP) goes on the goal: its offset in the hand frame (approach a, closing
        # axis c, both horizontal) is the same at the place pose, rotated with the approach direction.
        off = hw.mean(0) - tcp
        o_a, o_c = float(off @ a_r), float(off @ c_r)
        c_new = np.sign(np.cross(a_r, c_r)[2] or 1.0) * np.array([-a[1], a[0], 0.0])
        xy = np.asarray(goal_xy, float) - o_a * a[:2] - o_c * c_new[:2]
        p_touch = np.array([xy[0], xy[1], top_z + (tcp[2] - bottom)])
        p_pre = p_touch + [0, 0, 0.10]
        to_pre = _reach(kin, d, at, q_ready, p_pre, a, via_up=0.05, attach=attach,
                        rng=np.random.default_rng(trials))
        if to_pre is None:
            stats["ik_fail_pre"] = stats.get("ik_fail_pre", 0) + 1
            continue
        q_pre = to_pre[-1]
        v = _arm_path_ok(kin, d, at, to_pre, attach, held, start=q_ready, static_margin=CLEAR_ARM, target=False)
        if v:
            stats["arm_blocked"] += 1
            _note(stats, v)
            continue
        down = _cartesian(kin, d, at, q_pre, p_pre, p_touch - [0, 0, 0.03], a, 39)     # 3.3 mm steps
        back = _cartesian(kin, d, at, down[-10] if down else q_pre, p_touch, p_touch - back_dist * a + [0, 0, 0.04],
                          a, max(5, int(back_dist / 0.02))) if down else None
        if back is None:
            stats["ik_fail_" + ("down" if down is None else "back")] = stats.get(
                "ik_fail_" + ("down" if down is None else "back"), 0) + 1
            continue
        v = (_arm_path_ok(kin, d, at, down[:-10], attach, held, start=q_pre, static_margin=CLEAR_ARM,
                          hand_static_margin=CLEAR_GRASP, carried_static=False) or
             _arm_path_ok(kin, d, replace(at, f=FINGER_OPEN, fl=None), back, start=down[-10], static_margin=CLEAR_ARM,
                          hand_static_margin=CLEAR_GRASP, depth=False, target=False) or
             _arm_path_ok(kin, d, replace(at, f=FINGER_OPEN, fl=None), [q_ready], start=back[-1],
                          static_margin=CLEAR_ARM, depth=False, target=False) or
             _arm_path_ok(kin, d, replace(at, f=f_closed, fl=None), unfold[::-1] + [np.asarray(TUCK_RIGHT)],
                          start=q_ready, static_margin=CLEAR_ARM,
                          depth=False, target=False))
        if v:
            stats["arm_blocked"] += 1
            _note(stats, v)
            continue
        yield {"base": (bx, by, yaw), "approach": a, "q_pre": q_pre, "to_pre_qs": to_pre, "down_qs": down,
               "back_qs": back,
               "p_touch": p_touch, "base_travel_m": travel}


class Roadmap:
    """Grid roadmap for the kinematic base: turn on the spot, drive straight, reverse out of tight spots.

    A grid node is usable if the robot can turn on the spot there (8 headings clear).  Start / goal
    poses connect through a straight reverse / forward segment.  Obstacles: static furniture, the
    held box (if any), depth points of other objects.
    """

    def __init__(self, cam, kin, st, attach=None, held=None):
        self.kin, self.st, self.attach, self.held = kin, st, attach, held
        self.d = _plan_copy(cam)
        self.node_cache, self.checks = {}, 0
        pts = [np.array(s[:2]) for s in STATIONS.values()] + [np.array([st.x, st.y])]
        g = kin.static_obs
        pts += list(self.d.geom_xpos[g][:, :2])
        P = np.array(pts)
        self.lo = np.floor((P.min(0) - 1.2) / GRID_RES) * GRID_RES
        self.hi = P.max(0) + 1.2

    def pose_ok(self, x, y, yaw):
        self.checks += 1
        put(self.kin, self.d, replace(self.st, x=x, y=y, yaw=yaw), self.attach)
        held = None if self.held is None else self.held.world(self.d)
        return not self.kin.violations(self.d, CLEAR_BASE, self_check=False, carried=held)

    def turn_ok(self, x, y, y0, y1):
        dy = _wrap(y1 - y0)
        n = max(1, int(math.ceil(abs(dy) / 0.12)))
        return all(self.pose_ok(x, y, y0 + dy * i / n) for i in range(1, n + 1))

    def seg_ok(self, p0, p1, yaw):
        L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        n = max(1, int(math.ceil(L / 0.04)))
        return all(self.pose_ok(p0[0] + (p1[0] - p0[0]) * i / n, p0[1] + (p1[1] - p0[1]) * i / n, yaw)
                   for i in range(1, n + 1))

    def node_xy(self, ij):
        return (self.lo[0] + ij[0] * GRID_RES, self.lo[1] + ij[1] * GRID_RES)

    def node_ok(self, ij):
        if ij not in self.node_cache:
            x, y = self.node_xy(ij)
            ok = self.lo[0] <= x <= self.hi[0] and self.lo[1] <= y <= self.hi[1]
            self.node_cache[ij] = ok and all(self.pose_ok(x, y, k * math.pi / 4) for k in range(8))
        return self.node_cache[ij]

    def _ends(self, pose, forward_in):
        """Ways to leave (or enter) a pose: reverse straight out by 0 / 0.25 / 0.45 m."""
        x, y, yaw = pose
        out = []
        for r in (0.0, 0.25, 0.45):
            q = (x - r * math.cos(yaw), y - r * math.sin(yaw))
            if r == 0 or self.seg_ok((x, y), q, yaw):
                out.append((q, r))
        return out

    def _heading(self, p0, p1):
        return math.atan2(p1[1] - p0[1], p1[0] - p0[0])

    def _link(self, q, yaw_at_q, target):
        """Turn at q towards target, then drive straight there."""
        h = self._heading(q, target)
        return self.turn_ok(q[0], q[1], yaw_at_q, h) and self.seg_ok(q, target, h)

    def plan(self, start, goal):
        """Waypoints (x, y, yaw); consecutive waypoints differ either in position or in yaw."""
        if math.hypot(goal[0] - start[0], goal[1] - start[1]) < 1e-3 and abs(_wrap(goal[2] - start[2])) < 1e-3:
            return [start]
        s_ends, g_ends = self._ends(start, False), self._ends(goal, True)
        # Direct: start -> reverse -> turn -> drive -> turn -> forward into goal.
        for sq, _ in s_ends:
            for gq, _ in g_ends:
                if math.hypot(gq[0] - sq[0], gq[1] - sq[1]) < 1e-6:
                    continue
                h = self._heading(sq, gq)
                if (self.turn_ok(sq[0], sq[1], start[2], h) and self.seg_ok(sq, gq, h)
                        and self.turn_ok(gq[0], gq[1], h, goal[2])):
                    return self._waypoints(start, sq, [gq], goal)
        # Grid A*.
        def nearest(q):
            i0, j0 = int(round((q[0] - self.lo[0]) / GRID_RES)), int(round((q[1] - self.lo[1]) / GRID_RES))
            c = [(i0 + di, j0 + dj) for di in range(-3, 4) for dj in range(-3, 4)]
            return sorted(c, key=lambda ij: math.dist(self.node_xy(ij), q))
        starts = []
        for sq, _ in s_ends:
            for ij in nearest(sq)[:8]:
                if self.node_ok(ij) and self._link(sq, start[2], self.node_xy(ij)):
                    starts.append((sq, ij))
                    break
        goals = {}
        for gq, _ in g_ends:
            for ij in nearest(gq)[:8]:
                if self.node_ok(ij):
                    h = self._heading(self.node_xy(ij), gq)
                    if self.seg_ok(self.node_xy(ij), gq, h) and self.turn_ok(gq[0], gq[1], h, goal[2]):
                        goals.setdefault(ij, gq)
        if not starts or not goals:
            return None
        nbrs = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]
        gxy = self.node_xy(next(iter(goals)))
        for sq, sij in starts:
            openq = [(0.0, 0.0, sij, None)]
            came, cost = {}, {sij: 0.0}
            expansions = 0
            while openq and expansions < 1500:
                _, c, ij, parent = heapq.heappop(openq)
                if ij in came:
                    continue
                came[ij] = parent
                expansions += 1
                if ij in goals:
                    path = [ij]
                    while came[path[-1]] is not None:
                        path.append(came[path[-1]])
                    pts = [self.node_xy(p) for p in reversed(path)]
                    pts = self._shortcut(pts)
                    return self._waypoints(start, sq, pts + [goals[ij]], goal)
                for di, dj in nbrs:
                    nb = (ij[0] + di, ij[1] + dj)
                    if nb in came or not self.node_ok(nb):
                        continue
                    a, b = self.node_xy(ij), self.node_xy(nb)
                    nc = c + math.dist(a, b)
                    if nc < cost.get(nb, 1e9) and self.seg_ok(a, b, self._heading(a, b)):
                        cost[nb] = nc
                        heapq.heappush(openq, (nc + math.dist(b, gxy), nc, nb, ij))
        return None

    def _shortcut(self, pts):
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.seg_ok(pts[i], pts[j], self._heading(pts[i], pts[j])):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    def _waypoints(self, start, sq, mids, goal):
        wps = [tuple(start)]
        def add(x, y, yaw):
            if math.hypot(x - wps[-1][0], y - wps[-1][1]) > 1e-6 or abs(_wrap(yaw - wps[-1][2])) > 1e-6:
                wps.append((float(x), float(y), float(yaw)))
        add(sq[0], sq[1], start[2])                    # reverse out
        prev = sq
        for p in mids:
            h = self._heading(prev, p)
            add(prev[0], prev[1], h)                   # turn
            add(p[0], p[1], h)                         # drive
            prev = p
        add(prev[0], prev[1], goal[2])                 # turn
        add(goal[0], goal[1], goal[2])                 # forward in
        return wps


# ---------------------------------------------------------------- execution
class Animator:
    """Moves the displayed robot; every frame goes to the viewer / recorder and is clearance-checked."""

    def __init__(self, cam, kin: Kin, instant=False):
        self.cam, self.kin, self.instant = cam, kin, instant
        self.speed = getattr(cam, "speed", 1.0)
        self.st = read_state(kin, cam.data)
        self.attach: Attachment | None = None
        self.placing = False
        self.lifted = False
        self.obj_hits = {}
        self.frames = 0
        k = kin
        self.lim_margin = np.full(len(k.audit_j), np.inf)       # min distance to either limit
        self.max_speed = np.zeros(len(k.audit_j))              # displayed speed (video time)
        self._prev_q = None
        self.check = {"min_robot_furniture_m": 1.0, "min_self_margin_m": 1.0, "min_held_furniture_m": 1.0,
                      "min_robot_other_objects_m": 1.0, "frames_in_collision": 0,
                      "frames_below_self_clearance": 0, "min_self_distance_at_worst_margin_m": None,
                      "worst": None}

    def _audit_joints(self):
        k = self.kin
        q = self.cam.data.qpos[k.audit_adr].copy()
        self.lim_margin = np.minimum(self.lim_margin, np.minimum(q - k.audit_lo, k.audit_hi - q))
        if self._prev_q is not None and not self.instant:
            self.max_speed = np.maximum(self.max_speed, np.abs(q - self._prev_q) * FPS)
        self._prev_q = q

    def joint_report(self):
        k = self.kin
        rows = {}
        for i, n in enumerate(k.audit_names):
            rows[n] = {"range": [round(float(k.audit_lo[i]), 3), round(float(k.audit_hi[i]), 3)],
                       "min_margin_to_limit": round(float(self.lim_margin[i]), 4)}
            if not self.instant:
                rows[n]["max_speed_per_s_video"] = round(float(self.max_speed[i]), 3)
        bad = [n for n, r in rows.items() if r["min_margin_to_limit"] < -1e-6]
        return {"violations": bad, "joints": rows}

    def _measure(self):
        """Ground-truth geometry check of the displayed frame (logging only, not used for decisions)."""
        k, d = self.kin, self.cam.data
        rf = k._sets_min(d, k.robot_col, k.static_obs, 0.2)
        marg, sp, sdist = k.self_margin(d)
        sf = (marg, sp)
        others = np.array([g for b, gs in k.obj_geoms.items() if not (self.attach and b == self.attach.body)
                           for g in gs], int)
        oo = k._sets_min(d, k.robot_col, others, 0.2)
        # Per object: robot (and the carried object) vs every other movable object.  Which one is the
        # target is only known afterwards (offline scoring), so penetrations are kept per object and the
        # grasped one is excluded in the report.
        for b, gs in k.obj_geoms.items():
            if self.attach is not None and b == self.attach.body:
                continue
            dist, pair = k._sets_min(d, k.robot_col, gs, 0.05)
            if self.attach is not None and not self.placing:
                dh, ph = k._sets_min(d, k.obj_geoms[self.attach.body], gs, 0.05)
                if dh < dist:
                    dist, pair = dh, ph
            if dist < -0.002:
                rec = self.obj_hits.setdefault(b, {"frames": 0, "min_m": 0.0, "pair": None, "phase": None})
                rec["frames"] += 1
                if dist < rec["min_m"]:
                    rec["min_m"], rec["pair"], rec["phase"] = dist, f"{k.gname(pair[0])}|{k.gname(pair[1])}", \
                        self.cam.caption
        hf = ((1.0, None) if self.attach is None or self.placing or not self.lifted
              else k._sets_min(d, k.obj_geoms[self.attach.body], k.static_obs, 0.2))
        c = self.check
        for key, (dist, p) in (("min_robot_furniture_m", rf), ("min_self_margin_m", sf),
                               ("min_robot_other_objects_m", oo), ("min_held_furniture_m", hf)):
            if dist < c[key]:
                c[key] = dist
                c[key[:-2] + "_at"] = None if p is None else f"{k.gname(p[0])}|{k.gname(p[1])} ({self.cam.caption})"
                if dist < 0 and (c["worst"] is None or dist < c["worst"][1]):
                    c["worst"] = (f"{key}: {k.gname(p[0])}|{k.gname(p[1])}", round(float(dist), 3))
        # Penetration (> 2 mm into furniture, held box into furniture, or self deeper than the home-pose
        # baseline allows) vs merely closer than the 5 mm self-clearance target: logged separately.
        self_pen = sp is not None and marg < 0 and sdist < -0.002
        if min(rf[0], hf[0]) < -0.002 or self_pen:
            c["frames_in_collision"] += 1
        if marg < 0:
            c["frames_below_self_clearance"] += 1
            if c["min_self_distance_at_worst_margin_m"] is None or marg <= c["min_self_margin_m"]:
                c["min_self_distance_at_worst_margin_m"] = round(float(sdist), 4)

    def frame(self, caption):
        """Show one pose.  In instant mode (headless CLI) nothing is rendered, but the pose is still
        set and clearance-checked, so the CLI execution check covers every intermediate pose."""
        viewer = getattr(self.cam, "viewer", None)
        if viewer is not None:
            with viewer.lock():
                show(self.cam, self.kin, self.st, self.attach)
        else:
            show(self.cam, self.kin, self.st, self.attach)
        self.cam.caption = caption
        self._measure()
        self._audit_joints()
        fn = getattr(self.cam, "_frame", None)
        if fn is not None and not self.instant:
            fn()
            if viewer is not None and not getattr(self.cam, "show_window", False):
                time.sleep(1.0 / FPS)
        self.frames += 1

    def tween(self, caption, secs, base=None, torso=None, q=None, head=None):
        s0 = self.st
        n = max(2, int(secs * FPS / self.speed))
        if q is not None:   # never let one frame move a joint more than the planner's check step
            n = max(n, int(math.ceil(float(np.max(np.abs(np.asarray(q) - np.asarray(s0.q)))) / MAX_JOINT_STEP)))
        if base is not None:
            n = max(n, int(math.ceil(math.hypot(base[0] - s0.x, base[1] - s0.y) / 0.04)),
                    int(math.ceil(abs(_wrap(base[2] - s0.yaw)) / 0.1)))
        for i in range(1, n + 1):
            s = 0.5 - 0.5 * math.cos(math.pi * i / n)
            st = replace(s0)
            if base is not None:
                st.x = s0.x + s * (base[0] - s0.x)
                st.y = s0.y + s * (base[1] - s0.y)
                st.yaw = s0.yaw + s * _wrap(base[2] - s0.yaw)
            if torso is not None:
                st.torso = s0.torso + s * (torso - s0.torso)
            if q is not None:
                st.q = tuple(np.asarray(s0.q) + s * (np.asarray(q) - np.asarray(s0.q)))
            if head is not None:
                st.pan = s0.pan + s * (head[0] - s0.pan)
                st.tilt = s0.tilt + s * (head[1] - s0.tilt)
            self.st = st
            self.frame(caption)
        if base is not None:
            self.st.yaw = _wrap(self.st.yaw)

    def arm(self, caption, q, speed=0.8):
        dq = float(np.max(np.abs(np.asarray(q) - np.asarray(self.st.q))))
        self.tween(caption, 0.4 + dq / speed, q=q)

    def arm_path(self, caption, qs, secs_per_point=0.25):
        for q in qs:
            self.tween(caption, secs_per_point, q=q)

    def look(self, p, caption="", secs=0.6):
        cam_pos = camera_pose_from_mujoco(self.kin.m, self.cam.data)["pos"]
        t = look_at(self.kin, self.st, cam_pos, p)
        self.tween(caption, secs, head=(t.pan, t.tilt))

    def drive(self, caption, waypoints):
        for wp in waypoints[1:]:
            dist = math.hypot(wp[0] - self.st.x, wp[1] - self.st.y)
            secs = 0.3 + dist / 0.4 + abs(_wrap(wp[2] - self.st.yaw)) / 0.8
            self.tween(caption, secs, base=wp)

    def hold(self, caption, secs):
        for _ in range(1 if self.instant else int(secs * FPS / self.speed)):
            self.frame(caption)

    def close_until_contact(self, caption, step=0.0015):
        """Close each finger until it touches a movable object; success if both touch the same one."""
        fr = fl = self.st.f
        done = [False, False]
        t = [set(), set()]
        while not all(done):
            fr = fr if done[0] else max(0.0, fr - step)
            fl = fl if done[1] else max(0.0, fl - step)
            self.st = replace(self.st, f=fr, fl=fl)
            self.frame(caption)
            t = self.kin.finger_touch(self.cam.data)
            done = [done[0] or bool(t[0]) or fr <= 0, done[1] or bool(t[1]) or fl <= 0]
        common = t[0] & t[1]
        return (min(common) if common else None), t

    def fingers_to(self, caption, f, secs=0.6):
        s0 = self.st
        n = 1 if self.instant else max(2, int(secs * FPS / self.speed))
        l0 = s0.f if s0.fl is None else s0.fl
        for i in range(1, n + 1):
            s = 0.5 - 0.5 * math.cos(math.pi * i / n)
            self.st = replace(s0, f=s0.f + s * (f - s0.f), fl=l0 + s * (f - l0))
            self.frame(caption)
        self.st = replace(self.st, f=f, fl=None)

    def open_fingers(self, caption, secs=0.6):
        s0 = self.st
        n = 1 if self.instant else max(2, int(secs * FPS / self.speed))
        l0 = s0.f if s0.fl is None else s0.fl
        for i in range(1, n + 1):
            s = 0.5 - 0.5 * math.cos(math.pi * i / n)
            self.st = replace(s0, f=s0.f + s * (FINGER_OPEN - s0.f), fl=l0 + s * (FINGER_OPEN - l0))
            self.frame(caption)
        self.st = replace(self.st, f=FINGER_OPEN, fl=None)


def _update_trace(out: Path | None, log: dict):
    if out is None:
        return
    p = Path(out) / "trace.json"
    try:
        t = json.loads(p.read_text()) if p.is_file() else {}
    except (OSError, json.JSONDecodeError):
        t = {}
    t["fetch"] = log
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(t, indent=2))


def _make_keepalive(cam):
    viewer = getattr(cam, "viewer", None)
    window = getattr(cam, "show_window", False)
    if viewer is None and not window:
        return None

    def keep():
        if viewer is not None:
            viewer.sync()
        if window:
            import cv2
            cv2.waitKey(1)
    return keep


UNFOLD_FILE = Path(__file__).resolve().parent / "data" / "unfold_right_v0_7.json"


def prepare_kin(cam):
    """Kin with self-collision baseline and the behind-the-robot limit from the tuck pose."""
    kin = Kin(cam.model)
    st0 = read_state(kin, cam.data)
    ds = []
    for torso in (st0.torso, MANIP_TORSO):
        dd = mujoco.MjData(cam.model)
        dd.qpos[:] = cam.data.qpos
        put(kin, dd, replace(st0, torso=torso, q=TUCK_RIGHT))
        ds.append(dd)
    kin.set_self_baseline(ds)
    kin.set_front_limit(ds[1])
    return kin


def load_unfold(cam, kin, st):
    """Precomputed tuck -> ready path (scripts: --precompute-unfold).  Used only if it was made for the
    same tuck pose / torso height / joint-limit setting and is collision-free here and now."""
    if not UNFOLD_FILE.is_file():
        return None, None, f"no precomputed unfold ({UNFOLD_FILE.name}); planning online"
    try:
        u = json.loads(UNFOLD_FILE.read_text())
        if (not np.allclose(u["tuck_right"], TUCK_RIGHT, atol=1e-4) or abs(u["torso"] - MANIP_TORSO) > 1e-6
                or u.get("arm3_upper") != ARM3_UPPER or u.get("behind_allowance_m") != BEHIND_ALLOWANCE):
            return None, None, "precomputed unfold is for another tuck / torso / limit / behind setting; planning online"
        q_ready = np.asarray(u["q_ready"], float)
        path = [np.asarray(q, float) for q in u["path"]]
    except (OSError, KeyError, ValueError) as e:
        return None, None, f"precomputed unfold unreadable ({e}); planning online"
    if np.any(q_ready < kin.lo) or np.any(q_ready > kin.hi):
        return None, None, "precomputed unfold outside joint limits; planning online"
    d = _plan_copy(cam)
    stm = replace(st, torso=MANIP_TORSO, q=TUCK_RIGHT, fl=None)
    v = (_arm_path_ok(kin, d, stm, path, start=TUCK_RIGHT, static_margin=CLEAR_ARM, depth=False, target=False)
         or _arm_path_ok(kin, d, replace(stm, f=FINGER_OPEN), [q_ready], static_margin=CLEAR_ARM, depth=False,
                         target=False))
    if v:
        return None, None, f"precomputed unfold not collision-free here ({v[0]}); planning online"
    return q_ready, path, f"precomputed ({UNFOLD_FILE.name}, {u.get('created', '?')})"


def precompute_unfold(cam, collect=6, rrt_cap=8, tries=3, budget=15.0, deadline=360.0):
    kin = prepare_kin(cam)
    st = replace(read_state(kin, cam.data), torso=MANIP_TORSO, fl=None)
    t0 = time.perf_counter()
    print(f"precomputing unfold: up to {rrt_cap} RRT attempts, stops after {collect} paths or ~{deadline / 60:.0f} min",
          flush=True)
    q_ready, path, report = find_ready(cam, kin, st, rrt_cap=rrt_cap, tries=tries, budget=budget, collect=collect,
                                       deadline=deadline, verbose=True)
    for p_, why in report:
        print(f"  {p_ if isinstance(p_, str) else 'ready ' + str(np.round(p_, 2).tolist())}: {why}")
    if q_ready is None:
        print("no unfold path found")
        return None
    sparse = [np.asarray(TUCK_RIGHT, float)] + list(path)
    keep = [sparse[0]]                       # drop interpolated points, keep corners
    for i in range(1, len(sparse) - 1):
        d1, d2 = sparse[i] - keep[-1], sparse[i + 1] - sparse[i]
        u1, u2 = d1 / max(np.linalg.norm(d1), 1e-12), d2 / max(np.linalg.norm(d2), 1e-12)
        if np.linalg.norm(u1 - u2) > 1e-6:          # direction changes here: a real waypoint
            keep.append(sparse[i])
    keep.append(sparse[-1])
    UNFOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": FETCH_VERSION, "created": datetime.now().isoformat(timespec="seconds"),
            "tuck_right": list(TUCK_RIGHT), "torso": MANIP_TORSO, "arm3_upper": ARM3_UPPER,
            "behind_allowance_m": BEHIND_ALLOWANCE, "q_ready": [float(v) for v in q_ready],
            "path": [[float(v) for v in q] for q in _densify(TUCK_RIGHT, keep[1:])],
            "waypoints": len(keep), "joint_length_rad": round(path_length(keep), 3),
            "backward_excursion_m": round(kin.backward_excursion(_plan_copy(cam), st, sparse), 3),
            "seconds": round(time.perf_counter() - t0, 1)}
    UNFOLD_FILE.write_text(json.dumps(data, indent=1))
    print(f"saved {UNFOLD_FILE}: {len(keep)} waypoints, {data['joint_length_rad']} rad, "
          f"arm goes {100 * data['backward_excursion_m']:.1f} cm behind its tuck position, {data['seconds']} s")
    return data


def run_fetch(cam, agent_result: dict, last_res, out: Path | None = None, target: str | None = None,
              animate: bool = True) -> dict:
    """Fetch-and-deliver after the agent loop.  Returns a JSON-safe log (also written into trace.json)."""
    global _KEEPALIVE
    _KEEPALIVE = _make_keepalive(cam)
    t0 = time.perf_counter()
    target = target or agent_result.get("target")
    print(f"  {FETCH_VERSION}")
    log = {"version": FETCH_VERSION,
           "kind": "kinematic scripted fetch: joint positions set directly, rigid attach on finger contact; "
                   "not physics grasping, not a grasp planner",
           "target_requested": target, "outcome": None, "phases": []}
    kin = prepare_kin(cam)
    anim = Animator(cam, kin, instant=not animate)
    log["model"] = {"robot_collision_geoms": len(kin.robot_col), "furniture_geoms": len(kin.static_obs),
                    "self_pairs_checked": len(kin.selfA), "self_pairs_overlapping_in_tuck": kin.self_overlapping_in_tuck,
                    "body_meshes_checked_as_concave": kin.n_concave_body_geoms,
                    "home_pose_overlaps_over_1cm": kin.home_pose_overlaps,
                    "telescopic_column_geoms": [kin.gname(g) for g in kin.column],
                    "mujoco_contact_filter_allows_robot_furniture": kin.contact_filter_robot_static}

    unfold_box = [None, None]          # (unfold path, q_ready)
    f_box = [0.0]

    def phase(name, **kw):
        log["phases"].append({"phase": name, "t_s": round(time.perf_counter() - t0, 2), **_r(kw)})
        print(f"  fetch: {name} {_r(kw) if kw else ''}")

    def finish(outcome, caption, secs=3.0):
        log["outcome"] = outcome
        log["message"] = caption
        anim.hold(caption, secs)

    def back_to_tuck(plan):
        """Undo a grasp attempt the way it came: approach reversed, close the gripper, planned path back."""
        anim.arm_path("fetch: backing off", plan["approach_qs"][::-1] + [plan["q_pre"]])
        anim.arm_path("fetch: backing off", plan["to_pre_qs"][::-1] + [np.asarray(unfold_box[1])], 0.08)
        anim.fingers_to("fetch: backing off", f_box[0])
        anim.arm_path("fetch: backing off", plan["unfold_qs"][::-1] + [np.asarray(TUCK_RIGHT)], 0.05)

    def retreat_and_tuck(qs_back, q_ready, caption):
        if qs_back:
            anim.arm_path(caption, qs_back)
        if q_ready is not None:
            anim.arm(caption, q_ready)
            anim.fingers_to(caption, f_box[0])           # same gripper state as when unfolding
            anim.arm_path(caption, unfold_box[0][::-1] + [np.asarray(TUCK_RIGHT)], 0.05)   # fold back the same way
        else:
            anim.arm(caption, TUCK_RIGHT)

    try:
        if (agent_result.get("outcome") != "TARGET_LOCALISED_UNCALIBRATED" or last_res is None
                or getattr(last_res, "status", None) != "candidate"):
            finish("NO_FETCH_CANNOT_VERIFY",
                   f"Cannot verify '{target}' - please help. Robot does not pick anything up.")
            return log

        # 1. Where is it?  Perception + head depth only.
        from caremate.agent_loop_v0_7 import target_base_estimate
        pose = camera_pose_from_mujoco(cam.model, cam.data)
        p_base = target_base_estimate(last_res, pose, cam.W, cam.H)
        centre, est, pts_all = estimate_grasp_point(last_res.support_bbox_xyxy, last_res.projected_polygon,
                                                    p_base, pose, getattr(cam, "last_depth", None), cam.W, cam.H)
        log["target_estimate"] = _r({**est, "polygon_base_m": p_base, "object_centre_m": centre})
        if centre is None:
            finish("NO_FETCH_NO_3D_ESTIMATE", "Target seen but no 3D estimate - hand-off to staff.")
            return log
        kin.obstacles = obstacle_points(pts_all, centre, est)
        log["target_estimate"]["obstacle_depth_points"] = int(len(kin.obstacles))
        face = ({"normal_xy": est["face_normal_xy"], "half_width": est["face_half_width_m"]}
                if "face_normal_xy" in est else None)
        kin.target_pts = box_points(centre, face["normal_xy"] if face else (1.0, 0.0),
                                    face["half_width"] if face else 0.04, max(0.02, centre[2] - TABLE_Z))
        phase("estimate", centre_m=centre, side_grasp=face is not None)
        anim.hold(f"fetch: target at {np.round(centre, 2).tolist()} from OpenCV 5 + head depth; planning", 0.8)

        # 2. Ready posture, grasp plan, base route.
        f_closed = anim.st.f                     # gripper state while tucked / unfolding
        f_box[0] = f_closed
        st = replace(anim.st, torso=MANIP_TORSO, fl=None)
        q_ready, unfold, why = load_unfold(cam, kin, st)
        log["unfold_source"] = why
        if q_ready is None:
            q_ready, unfold, ready_report = find_ready(cam, kin, st)
            log["ready_search"] = _r([{"tcp_base_frame": p_, "result": w} for p_, w in ready_report][-6:])
        phase("unfold", source=why if q_ready is not None else "online search failed",
              joint_length_rad=path_length([np.asarray(TUCK_RIGHT)] + list(unfold)) if unfold else None)
        unfold_box[0], unfold_box[1] = unfold, q_ready
        if q_ready is None:
            finish("NO_FETCH_NO_READY_POSE", "No collision-free arm posture found - hand-off to manipulation module.")
            return log
        d = _plan_copy(cam)
        put(kin, d, replace(st, q=tuple(q_ready), f=FINGER_OPEN))
        finger_back, finger_tip = kin.finger_span_along(d)
        palm_front, s_grid, gap = kin.hand_geometry(d)
        present = ~np.isnan(gap)
        half_gap = float(np.nanmax(gap)) if present.any() else 0.0
        # s_pad: from here to the finger tips there is room for the assumed box (+ SIDE_CLEAR each side)
        need = ASSUMED_HALF_THICKNESS + SIDE_CLEAR
        ok = np.where(present, gap >= need, True)
        bad = np.nonzero(present & ~ok)[0]
        s_pad = float(s_grid[bad.max() + 1]) if len(bad) and bad.max() + 1 < len(s_grid) else (
            float(s_grid[present].min()) if not len(bad) else 1.0)
        log["model"]["finger_extent_m"] = _r({"behind_tcp": finger_back, "beyond_tcp": finger_tip,
                                             "palm_front_vs_tcp": palm_front, "pad_half_gap": half_gap,
                                             "pad_starts_at": s_pad,
                                             "half_gap_needed": need,
                                             "half_gap_profile": {f"{sv:+.2f}": g for sv, g in
                                                                  zip(s_grid[present][::6], gap[present][::6])}})
        stats = {"wrong_side": 0, "base_blocked": 0, "ready_blocked": 0, "ik_fail": 0, "arm_blocked": 0,
                 "no_route": 0}
        plan = path = None
        tucked = replace(st, q=TUCK_RIGHT)
        road = Roadmap(cam, kin, tucked)
        for cand in plan_grasp(cam, kin, st, q_ready, unfold, centre, face, finger_back, finger_tip, stats,
                               palm_front, half_gap, s_pad):
            path = road.plan(st.base, cand["base"])
            if path is not None:
                plan = cand
                break
            stats["no_route"] += 1
        log["grasp_planning"] = _r({"stats": stats, "base_checks": road.checks})
        if plan is None:
            finish("NO_FETCH_NO_REACH", "Target localised, but no collision-free reach found - "
                                        "hand-off to manipulation module.")
            return log
        log["grasp_planning"].update(_r({"base": plan["base"], "approach": plan["approach"],
                                         "base_route": path, "tcp_grasp_m": plan["p_main"]}))
        phase("grasp_plan", base=plan["base"], waypoints=len(path), approach=plan["approach"])

        # 3. Execute the grasp.
        anim.tween("fetch: raising torso", 1.0, torso=MANIP_TORSO)
        anim.drive("fetch: driving to the grasp pose (kinematic base, roadmap route)", path)
        anim.look(centre, "fetch: looking at the target")
        anim.arm_path("fetch: unfolding the arm (precomputed path)", plan["unfold_qs"], 0.05)
        anim.open_fingers("fetch: opening the gripper")
        anim.arm_path("fetch: hand to pre-grasp (straight line)", plan["to_pre_qs"], 0.08)
        anim.arm_path("fetch: approach", plan["approach_qs"])
        pre_touch = kin.finger_touch(cam.data, gap=-0.004)
        if pre_touch[0] or pre_touch[1]:
            phase("approach_collided", objects=[[kin.bname(b) for b in t] for t in pre_touch])
            back_to_tuck(plan)
            finish("FETCH_FAILED_APPROACH_COLLIDED", "Gripper hit the object while approaching - asking staff.")
            return log
        body, touch = anim.close_until_contact("fetch: closing each finger until it touches")
        if body is None:
            phase("grasp_missed", touched=[[kin.bname(b) for b in t] for t in touch])
            anim.open_fingers("fetch: grasp missed - opening")
            back_to_tuck(plan)
            finish("FETCH_FAILED_GRASP_MISSED", "Grasp missed (fingers closed on nothing) - asking staff.")
            return log
        _, _, c_axis = kin.tool(cam.data)
        anim.attach = Attachment(kin, cam.data, body)
        log["_grasped_body"] = body
        kin.target_pts = np.zeros((0, 3))       # now held: modelled by HeldBox instead
        held = HeldBox(kin, cam.data, centre, face["normal_xy"] if face else (1.0, 0.0),
                       face["half_width"] if face else 0.04, max(0.02, centre[2] - TABLE_Z))
        n_axis = np.array([*face["normal_xy"], 0.0]) if face else None
        phase("grasped", finger_qpos_m=[anim.st.f, anim.st.fl], closing_axis=c_axis,
              closing_axis_vs_face_normal=None if n_axis is None else abs(float(c_axis @ n_axis)))
        anim.arm_path("fetch: lift", plan["lift_qs"])
        anim.lifted = True
        anim.arm_path("fetch: carry posture", plan["carry_qs"], 0.08)

        # 4. Deliver.
        try:
            site = _id(cam.model, mujoco.mjtObj.mjOBJ_SITE, DELIVERY_SITE)
            goal_xy = cam.data.site_xpos[site][:2].copy()
            top, top_g = surface_below(kin, _plan_copy(cam), goal_xy)
        except KeyError:
            goal_xy, top, top_g = np.array(FALLBACK_DELIVERY[:2]), None, None
        if top is None:
            top, top_g = FALLBACK_DELIVERY[2], None
        st = anim.st
        stats2 = {"base_blocked": 0, "arm_blocked": 0, "no_route": 0}
        road2 = Roadmap(cam, kin, st, anim.attach, held)
        plan2 = path2 = None
        for cand in plan_place(cam, kin, st, q_ready, unfold, goal_xy, top, held, anim.attach, stats2,
                               back_dist=plan["back_dist"], f_closed=f_closed):
            path2 = road2.plan(st.base, cand["base"])
            if path2 is not None:
                plan2 = cand
                break
            stats2["no_route"] += 1
        log["place_planning"] = _r({"delivery_xy": goal_xy, "table_top_z": top,
                                    "table_geom": None if top_g is None else kin.gname(top_g),
                                    "stats": stats2, "base_checks": road2.checks})
        if plan2 is None:
            finish("FETCH_HOLDING_NO_PLACE_ROUTE", "Holding the object but cannot reach the delivery table - "
                                                   "calling staff.")
            return log
        log["place_planning"].update(_r({"base": plan2["base"], "base_route": path2}))
        phase("place_plan", base=plan2["base"], waypoints=len(path2))
        anim.drive("deliver: carrying to the table next to the resident", path2)
        anim.look((goal_xy[0], goal_xy[1], top), "deliver: looking at the delivery table")
        anim.arm_path("deliver: arm above the delivery table", plan2["to_pre_qs"], 0.12)
        anim.placing = True
        touched_at = None
        prev = plan2["q_pre"]
        for i, q in enumerate(plan2["down_qs"]):
            anim.tween("deliver: lowering until the object touches the table", 0.05, q=q)
            dist, _ = kin._sets_min(cam.data, kin.obj_geoms[body], kin.static_obs, 0.05)
            if dist <= 0.002:
                touched_at = i
                if dist < -0.001:          # stepped into the table: back up to the last free step
                    anim.tween("deliver: lowering until the object touches the table", 0.05, q=prev)
                break
            prev = q
        phase("lowered", touched=touched_at is not None)
        anim.open_fingers("deliver: releasing", 0.8)
        anim.attach = None
        # Released object settles under gravity (kinematic stand-in for the drop): lower it vertically
        # until it rests on the furniture below.  Display only; no decision uses the object's pose.
        adr = kin.objects[body]
        gap, _ = kin._sets_min(cam.data, kin.obj_geoms[body], kin.static_obs, 0.05)
        if 0.0005 < gap < 0.05:
            z0 = float(cam.data.qpos[adr + 2])
            n = 1 if anim.instant else max(2, int(0.25 * FPS / anim.speed))
            for i in range(1, n + 1):
                cam.data.qpos[adr + 2] = z0 - gap * (i / n) ** 2
                anim.frame("deliver: object settles on the table")
            phase("settled", drop_m=gap)
        retreat_and_tuck(plan2["back_qs"], q_ready, "deliver: retract and tuck")
        anim.placing = False
        phase("placed")

        # Offline scoring only (simulator ground truth, never used above).
        obj = kin.bname(body)
        dist_tab, _ = kin._sets_min(cam.data, kin.obj_geoms[body], kin.static_obs, 0.3)
        log["offline_scoring_gt"] = _r({
            "object_delivered": obj, "is_requested_target": obj == target,
            "dist_to_delivery_goal_m": float(np.linalg.norm(cam.data.xpos[body][:2] - goal_xy)),
            "object_gap_to_table_m": dist_tab})
        finish("DELIVERED_KINEMATIC" if touched_at is not None else "DELIVERED_KINEMATIC_NO_TABLE_CONTACT",
               f"Delivered '{obj}' next to the resident (kinematic demo; identity from OpenCV 5 evidence, uncalibrated)")
        return log
    finally:
        grasped_body = anim.attach.body if anim.attach is not None else log.get("_grasped_body")
        hits = {kin.bname(b): _r(h) for b, h in anim.obj_hits.items() if b != grasped_body}
        anim.check["other_object_collisions"] = hits
        anim.check["frames_in_collision"] += sum(h["frames"] for h in hits.values())
        log.pop("_grasped_body", None)
        log["execution_check"] = _r(anim.check)
        log["joint_limits_check"] = anim.joint_report()
        log["frames"] = anim.frames
        log["seconds"] = round(time.perf_counter() - t0, 2)
        st = anim.st
        if hasattr(cam, "state"):
            cam.state = (st.x, st.y, st.yaw, st.torso, st.pan, st.tilt)
        _update_trace(out, log)
        _KEEPALIVE = None
        print(f"FETCH: {log['outcome']} ({log['seconds']} s); execution check {log['execution_check']}")
        jl = log["joint_limits_check"]
        print(f"JOINT LIMITS: violations={jl['violations']}; closest to a limit: " + ", ".join(
            f"{n} {r['min_margin_to_limit']:+.3f}" for n, r in sorted(jl["joints"].items(),
                                                              key=lambda kv: kv[1]["min_margin_to_limit"])[:4]))


# ---------------------------------------------------------------- teach mode (human keyframes)
TEACH_HELP = """commands (joints 1-7 = arm_right_1..7, values in radians):
  <j> <value>     set joint j,  e.g.  2 0.8    or   3 -1.5
  <j> += <d>      move joint j, e.g.  4 += 0.3  or   4 -= 0.3
  show            print joints, hand position (base frame) and collision status
  key             record the current pose as a keyframe (must be collision-free, and so must the
                  straight joint move from the previous keyframe)
  undo            drop the last keyframe and go back to the one before
  back            return to the last keyframe
  done            connect the last keyframe to a ready (carry) pose, check everything, save
  quit            leave without saving"""


def _tcp_base(kin, d):
    p, a, _ = kin.tool(d)
    R, o = d.xmat[kin.base_body].reshape(3, 3), d.xpos[kin.base_body]
    return (p - o) @ R, a @ R


def teach_session(cam, commands, out=print, display=None):
    """Keyframe teaching: commands is an iterable of strings (stdin in the CLI, a list in tests).
    display(q) is called after every change (viewer update).  Returns the saved dict or None."""
    kin = prepare_kin(cam)
    st = replace(read_state(kin, cam.data), torso=MANIP_TORSO, fl=None)
    stm = replace(st, q=TUCK_RIGHT)
    d = _plan_copy(cam)
    q = np.asarray(TUCK_RIGHT, float).copy()
    keys = [q.copy()]

    def status(qq):
        put(kin, d, replace(stm, q=tuple(qq)))
        v = kin.violations(d, CLEAR_ARM, depth=False, target=False)
        p, a = _tcp_base(kin, d)
        put(kin, d, stm)
        x0 = kin._x_base(d)
        put(kin, d, replace(stm, q=tuple(qq)))
        back = float(np.max(x0 - kin._x_base(d)))
        return v, p, back

    def report(qq):
        v, p, back = status(qq)
        out("  joints: " + "  ".join(f"{i + 1}:{x:+.2f}" for i, x in enumerate(qq)))
        out(f"  hand (base frame): x {p[0]:+.2f} (forward)  y {p[1]:+.2f} (left)  z {p[2]:.2f}   "
            f"arm {100 * max(back, 0):.0f} cm behind tuck")
        out("  COLLISION: " + v[0] if v else "  collision-free")

    def seg_ok(qa, qb):
        return _arm_path_ok(kin, d, stm, [qb], start=qa, static_margin=CLEAR_ARM, depth=False, target=False)

    out(TEACH_HELP)
    out("  ranges: " + "  ".join(f"{i + 1}:[{lo:+.2f},{hi:+.2f}]" for i, (lo, hi) in enumerate(zip(kin.lo, kin.hi))))
    report(q)
    if display:
        display(q)
    for line in commands:
        parts = line.strip().split()
        if not parts:
            continue
        cmd = parts[0].lower()
        if cmd in ("quit", "exit", "q"):
            out("left without saving")
            return None
        if cmd == "show":
            report(q)
        elif cmd == "key":
            v = seg_ok(keys[-1], q)
            if v:
                out(f"  NOT recorded: move from the previous keyframe collides ({v[0]})")
            else:
                keys.append(q.copy())
                out(f"  keyframe {len(keys) - 1} recorded")
        elif cmd == "undo":
            if len(keys) > 1:
                keys.pop()
            q = keys[-1].copy()
            out(f"  back to keyframe {len(keys) - 1}")
            report(q)
        elif cmd == "back":
            q = keys[-1].copy()
            report(q)
        elif cmd == "done":
            res = _finish_teach(cam, kin, st, stm, d, keys, out)
            if res is not None:
                return res
        elif cmd.isdigit() and 1 <= int(cmd) <= 7 and len(parts) in (2, 3):
            j = int(cmd) - 1
            rel = len(parts) == 3 and parts[1] in ("+=", "-=")
            try:
                val = float(parts[-1])
            except ValueError:
                out("  usage: 2 -1.2   (set)   or   2 += 0.3 / 2 -= 0.3   (move)")
                continue
            if len(parts) == 3 and not rel:
                out("  usage: 2 -1.2   (set)   or   2 += 0.3 / 2 -= 0.3   (move)")
                continue
            target = q[j] + (val if parts[1] == "+=" else -val) if rel else val
            clipped = float(np.clip(target, kin.lo[j], kin.hi[j]))
            if abs(clipped - target) > 1e-9:
                out(f"  joint {j + 1} limited to its range [{kin.lo[j]:+.2f}, {kin.hi[j]:+.2f}]")
            q = q.copy()
            q[j] = clipped
            report(q)
        else:
            out("  ? (type 'help' for commands)" if cmd != "help" else TEACH_HELP)
        if display:
            display(q)
    return None


def _finish_teach(cam, kin, st, stm, d, keys, out):
    """Connect the last keyframe to a ready pose (IK seeded from it), verify, save."""
    c, s_ = math.cos(st.yaw), math.sin(st.yaw)
    fwd = np.array([c, s_, 0.0])
    best = None
    for bx, by, bz in READY_POINTS:
        p = np.array([st.x + c * bx - s_ * by, st.y + s_ * bx + c * by, bz])
        put(kin, d, replace(stm, q=tuple(keys[-1])))
        qr, _ = kin.ik(d, p, fwd, [keys[-1]])
        if qr is None:
            continue
        put(kin, d, replace(stm, q=tuple(qr), f=FINGER_OPEN))
        if kin.violations(d, CLEAR_ARM, depth=False, target=False):
            continue
        if _arm_path_ok(kin, d, stm, [qr], start=keys[-1], static_margin=CLEAR_ARM, depth=False, target=False):
            continue
        path = keys + [qr]
        back = kin.backward_excursion(d, stm, _densify(None, path))
        cand = (round(back, 3), path_length(path), qr)
        if best is None or cand[:2] < best[:2]:
            best = cand
    if best is None:
        out("  no ready pose reachable in a straight joint move from the last keyframe without collision;"
            " add another keyframe closer to the front (hand forward, ~1 m high) and try 'done' again")
        return None
    back, length, q_ready = best
    path = keys + [q_ready]
    UNFOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": FETCH_VERSION, "created": datetime.now().isoformat(timespec="seconds"),
            "source": "taught keyframes (--teach)", "tuck_right": list(TUCK_RIGHT), "torso": MANIP_TORSO,
            "arm3_upper": ARM3_UPPER, "behind_allowance_m": BEHIND_ALLOWANCE,
            "keyframes": [[float(v) for v in k] for k in keys[1:]],
            "q_ready": [float(v) for v in q_ready],
            "path": [[float(v) for v in x] for x in _densify(TUCK_RIGHT, path[1:])],
            "waypoints": len(path), "joint_length_rad": round(length, 3), "backward_excursion_m": back}
    UNFOLD_FILE.write_text(json.dumps(data, indent=1))
    out(f"saved {UNFOLD_FILE}: {len(keys) - 1} keyframe(s) + ready pose, {length:.2f} rad, "
        f"arm {100 * back:.1f} cm behind tuck at most; every step checked for collision and joint limits")
    return data


def teach_cli(scene):
    """Viewer shows the robot; you type commands in this terminal."""
    import mujoco.viewer
    from caremate.agent_loop_v0_7 import START
    from caremate.demo_v0_7 import DemoCamera
    cam = DemoCamera(scene)
    cam._set(START)
    kin = Kin(cam.model)
    st = replace(read_state(kin, cam.data), torso=MANIP_TORSO, fl=None)
    show(cam, kin, replace(st, q=TUCK_RIGHT))
    viewer = mujoco.viewer.launch_passive(cam.model, cam.data)
    viewer.cam.lookat[:] = (st.x, st.y, 0.9)
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.2, 135.0, -15.0

    def display(qq):
        with viewer.lock():
            show(cam, kin, replace(st, q=tuple(qq)))
        viewer.sync()

    def lines():
        while viewer.is_running():
            try:
                yield input("teach> ")
            except EOFError:
                return

    try:
        teach_session(cam, lines(), display=display)
    finally:
        viewer.close()
        cam.close()


# ---------------------------------------------------------------- tuck pose check
def check_tuck(cam, q_tuck, out=print, render=True):
    """Is q_tuck usable as the right-arm home pose?  (1) inside the joint limits, (2) no self-collision
    and no furniture contact at every base station and torso height, (3) not visible in the head camera
    for any search view (it would occlude / confuse perception).  Returns True if all pass."""
    m = cam.model
    kin = Kin(m)
    q_tuck = np.asarray(q_tuck, float)
    ok_all = True
    rng = m.jnt_range[kin.arm_j]
    bad = [f"joint {i + 1} {q_tuck[i]:+.4f} outside [{rng[i, 0]:+.4f}, {rng[i, 1]:+.4f}]"
           for i in range(7) if not (rng[i, 0] <= q_tuck[i] <= rng[i, 1])]
    out("1) joint limits: " + ("OK" if not bad else "; ".join(bad)))
    ok_all &= not bad
    if ARM3_UPPER is not None and q_tuck[2] > ARM3_UPPER:
        out(f"   note: joint 3 {np.degrees(q_tuck[2]):.0f} deg is above the {np.degrees(ARM3_UPPER):.0f} deg strict limit")
    # (2) collisions: CLEAR_SELF to every other robot part (no home-pose allowance: this IS the home pose)
    st0 = read_state(kin, cam.data)
    d = _plan_copy(cam)
    worst = []
    for name, (x, y, yaw) in STATIONS.items():
        for torso in (0.0, MANIP_TORSO):
            put(kin, d, replace(st0, x=x, y=y, yaw=yaw, torso=torso, q=tuple(q_tuck)))
            sd = min(((kin.self_distance(d, a, b, 0.1), (a, b)) for a, b in zip(kin.selfA, kin.selfB)),
                     key=lambda t: t[0])
            fd, fp = kin._sets_min(d, kin.robot_col, kin.static_obs, 0.3)
            worst.append((sd[0], fd, name, torso, sd[1], fp))
    wsd = min(worst, key=lambda w: w[0])
    wfd = min(worst, key=lambda w: w[1])
    out(f"2) self: closest {100 * wsd[0]:+.1f} cm ({kin.gname(wsd[4][0])}|{kin.gname(wsd[4][1])} at {wsd[2]}, "
        f"torso {wsd[3]:.2f})" + ("  OK" if wsd[0] >= CLEAR_SELF else "  TOO CLOSE / COLLIDING"))
    out(f"   furniture: closest {100 * wfd[1]:+.1f} cm at {wfd[2]}" + ("  OK" if wfd[1] >= CLEAR_BASE else "  TOO CLOSE"))
    ok_all &= wsd[0] >= CLEAR_SELF and wfd[1] >= CLEAR_BASE
    if not render:
        return ok_all
    # (3) head-camera visibility of the right arm, every search view
    from caremate.agent_loop_v0_7 import CANDIDATES
    arm_bodies = {b for b in range(m.nbody) if kin._below(b, int(m.jnt_bodyid[kin.arm_j[0]]))}
    lj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "arm_left_1_joint")
    if lj >= 0:      # the left arm (as currently posed) must not be in the search images either
        arm_bodies |= {b for b in range(m.nbody) if kin._below(b, int(m.jnt_bodyid[lj]))}
    W, H = 320, 240
    r = mujoco.Renderer(m, height=H, width=W)
    r.enable_segmentation_rendering()
    dv = mujoco.MjData(m)
    dv.qpos[:] = cam.data.qpos
    from caremate.workspace_v0_7 import workspace_polygon
    seen, in_roi = [], []
    for v in CANDIDATES:
        x, y, yaw = STATIONS[v.base]
        put(kin, dv, replace(st0, x=x, y=y, yaw=yaw, torso=v.torso, pan=v.pan, tilt=v.tilt, q=tuple(q_tuck)))
        mujoco.mj_forward(m, dv)
        r.update_scene(dv, camera="head_rgb")
        seg = r.render()
        ids, types = seg[..., 0], seg[..., 1]
        geom_mask = types == int(mujoco.mjtObj.mjOBJ_GEOM)
        body = np.where(geom_mask & (ids >= 0), m.geom_bodyid[np.clip(ids, 0, m.ngeom - 1)], -1)
        arm = np.isin(body, list(arm_bodies))
        if arm.any():
            seen.append((float(arm.mean()), v.name))
            # Perception only searches the workspace ROI (table volume projected with camera FK).
            poly = workspace_polygon(camera_pose_from_mujoco(m, dv), W, H)
            if poly is not None:
                roi = polygon_mask(poly, W, H) > 0
                frac_roi = float((arm & roi).sum()) / max(1, int(roi.sum()))
                if frac_roi > 0:
                    in_roi.append((frac_roi, v.name))
    r.close()
    if seen:
        seen.sort(reverse=True)
        out(f"3) head camera: arms in the image in {len(seen)}/{len(CANDIDATES)} search views "
            f"(max {100 * seen[0][0]:.1f}% of the image, {seen[0][1]})")
    else:
        out(f"3) head camera: arms not in the image in any of {len(CANDIDATES)} search views")
    if in_roi:
        in_roi.sort(reverse=True)
        worst = in_roi[0][0]
        verdict = ("OK - negligible (the robot can also mask its own links by forward kinematics)"
                   if worst <= ROI_OCCLUSION_MAX else "NOT OK - occludes a real part of the table area")
        out(f"   inside the perception ROI in {len(in_roi)} views, max {100 * worst:.2f}% of the ROI "
            f"({in_roi[0][1]})  {verdict}")
        ok_all &= worst <= ROI_OCCLUSION_MAX
    else:
        out("   never inside the perception ROI (table area)  OK - perception ignores those pixels")
    out("=> usable as home pose" if ok_all else "=> NOT usable as home pose")
    return ok_all


# ---------------------------------------------------------------- interactive home-pose tuning
TUNE_HELP = """--tune-tuck: drag the arm_right_* sliders in the viewer's Joint panel, or type here:
  <j> <value> | <j> += <d> | <j> -= <d>   right-arm joint j (1-7), radians
  view <i> | view next | view prev | view off   show search view i (0-53) in the head camera window
  torso <m>         torso height (0-0.35) when no view is selected
  show              print joints + checks          check   full --check-tuck on the current pose
  print             print the pose as a TUCK_RIGHT line        quit"""


def tune_cli(scene):
    """Viewer + live head-camera window (perception ROI in cyan, right-arm pixels in red) + terminal."""
    import queue
    import threading

    import cv2
    import mujoco.viewer
    from caremate.agent_loop_v0_7 import CANDIDATES, START
    from caremate.demo_v0_7 import DemoCamera
    from caremate.workspace_v0_7 import workspace_polygon

    cam = DemoCamera(scene)
    cam._set(START)
    m, data = cam.model, cam.data
    kin = Kin(m)
    st = replace(read_state(kin, data), torso=MANIP_TORSO, fl=None)
    arm_bodies = {b for b in range(m.nbody) if kin._below(b, int(m.jnt_bodyid[kin.arm_j[0]]))}
    W, H = 640, 480
    rgb_r = mujoco.Renderer(m, height=H, width=W)
    seg_r = mujoco.Renderer(m, height=H, width=W)
    seg_r.enable_segmentation_rendering()
    dchk = mujoco.MjData(m)
    state = {"q": np.asarray(TUCK_RIGHT, float).copy(), "view": None, "torso": MANIP_TORSO, "dirty": True}

    def robot_state():
        if state["view"] is None:
            return replace(st, torso=state["torso"], q=tuple(state["q"]))
        v = CANDIDATES[state["view"]]
        x, y, yaw = STATIONS[v.base]
        return replace(st, x=x, y=y, yaw=yaw, torso=v.torso, pan=v.pan, tilt=v.tilt, q=tuple(state["q"]))

    def checks(rs):
        dchk.qpos[:] = data.qpos
        put(kin, dchk, rs)
        sd = min(((kin.self_distance(dchk, a, b, 0.1), (a, b)) for a, b in zip(kin.selfA, kin.selfB)),
                 key=lambda t: t[0])
        fd, _ = kin._sets_min(dchk, kin.robot_col, kin.static_obs, 0.3)
        return sd, fd

    def status():
        rs = robot_state()
        sd, fd = checks(rs)
        q = state["q"]
        lim = [i + 1 for i in range(7) if not (m.jnt_range[kin.arm_j[i], 0] <= q[i] <= m.jnt_range[kin.arm_j[i], 1])]
        print("  joints: " + "  ".join(f"{i + 1}:{x:+.3f}" for i, x in enumerate(q)))
        print(f"  self {100 * sd[0]:+.1f} cm ({kin.gname(sd[1][0])}|{kin.gname(sd[1][1])})"
              f"{'  OK' if sd[0] >= CLEAR_SELF else '  TOO CLOSE'}   furniture {100 * fd:+.1f} cm"
              f"   limits {'OK' if not lim else 'OUT: ' + str(lim)}"
              f"   view {state['view'] if state['view'] is not None else 'off'}", flush=True)

    def head_image():
        rgb_r.update_scene(data, camera="head_rgb")
        img = cv2.cvtColor(rgb_r.render(), cv2.COLOR_RGB2BGR)
        seg_r.update_scene(data, camera="head_rgb")
        seg = seg_r.render()
        ids, types = seg[..., 0], seg[..., 1]
        body = np.where((types == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0),
                        m.geom_bodyid[np.clip(ids, 0, m.ngeom - 1)], -1)
        arm = np.isin(body, list(arm_bodies))
        img[arm] = (0.4 * img[arm] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        poly = workspace_polygon(camera_pose_from_mujoco(m, data), W, H)
        txt = f"arm in image {100 * arm.mean():.2f}%"
        if poly is not None:
            cv2.polylines(img, [np.round(poly).astype(np.int32)], True, (255, 255, 0), 1)
            roi = polygon_mask(poly, W, H) > 0
            inside = arm & roi
            names = sorted({kin.bname(int(b)) for b in np.unique(body[inside])})
            txt += f" | in ROI {inside.sum()} px" + (f" ({', '.join(names)})" if names else "")
        cv2.rectangle(img, (0, H - 26), (W, H), (20, 20, 20), -1)
        cv2.putText(img, txt[:110], (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    cmds = queue.Queue()

    def reader():
        while True:
            try:
                cmds.put(input())
            except EOFError:
                cmds.put("quit")
                return

    print(TUNE_HELP)
    threading.Thread(target=reader, daemon=True).start()
    viewer = mujoco.viewer.launch_passive(m, data)
    viewer.cam.lookat[:] = (st.x, st.y, 0.9)
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.2, 135.0, -15.0
    win = "head camera (cyan = perception ROI, red = right arm)"
    last_q = None
    try:
        while viewer.is_running():
            # slider edits in the viewer change data.qpos directly: adopt them
            qd = data.qpos[kin.arm_adr].copy()
            if last_q is not None and np.max(np.abs(qd - last_q)) > 1e-6:
                state["q"], state["dirty"] = qd, True
            while not cmds.empty():
                parts = cmds.get().strip().split()
                if not parts:
                    continue
                c = parts[0].lower()
                if c in ("quit", "exit", "q"):
                    viewer.close()
                    break
                if c == "show":
                    status()
                elif c == "print":
                    print("TUCK_RIGHT = (" + ", ".join(f"{x:.4f}" for x in state["q"]) + ")", flush=True)
                elif c == "check":
                    check_tuck(cam, state["q"])
                elif c == "view" and len(parts) == 2:
                    n = len(CANDIDATES)
                    v = state["view"]
                    state["view"] = (None if parts[1] == "off" else ((v or 0) + 1) % n if parts[1] == "next"
                                     else ((v or 0) - 1) % n if parts[1] == "prev" else int(parts[1]) % n)
                    if state["view"] is not None:
                        print(f"  view {state['view']}: {CANDIDATES[state['view']].name}", flush=True)
                    state["dirty"] = True
                elif c == "torso" and len(parts) == 2:
                    state["torso"], state["dirty"] = float(np.clip(float(parts[1]), 0.0, 0.35)), True
                elif c.isdigit() and 1 <= int(c) <= 7 and len(parts) in (2, 3):
                    j = int(c) - 1
                    try:
                        val = float(parts[-1])
                    except ValueError:
                        print("  usage: 2 -1.2  or  2 += 0.3", flush=True)
                        continue
                    t = state["q"][j] + (val if parts[1] == "+=" else -val) if len(parts) == 3 else val
                    lo, hi = m.jnt_range[kin.arm_j[j]]
                    state["q"] = state["q"].copy()
                    state["q"][j] = float(np.clip(t, lo, hi))
                    state["dirty"] = True
                else:
                    print(TUNE_HELP, flush=True)
            if not viewer.is_running():
                break
            if state["dirty"]:
                with viewer.lock():
                    show(cam, kin, robot_state())
                last_q = data.qpos[kin.arm_adr].copy()
                status()
                state["dirty"] = False
            else:
                with viewer.lock():
                    mujoco.mj_forward(m, data)
                last_q = data.qpos[kin.arm_adr].copy()
            viewer.sync()
            cv2.imshow(win, head_image())
            cv2.waitKey(30)
    finally:
        print("final pose: TUCK_RIGHT = (" + ", ".join(f"{x:.4f}" for x in state["q"]) + ")")
        if viewer.is_running():
            viewer.close()
        cv2.destroyAllWindows()
        rgb_r.close()
        seg_r.close()
        cam.close()


# ---------------------------------------------------------------- CLI (headless check)
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from caremate.agent_loop_v0_7 import POLICIES, ROOT, START, make_backend, run_agent
    p.add_argument("--scene", type=Path, default=ROOT / "models/pal_tiago_dual/caremate_room_v0_7.xml")
    p.add_argument("--target", default="crunch_peanut")
    p.add_argument("--policy", choices=list(POLICIES), default="active")
    p.add_argument("--budget", type=int, default=3)
    p.add_argument("--aliked", type=Path, default=ROOT / "models/vision/aliked-n16rot-top1k-640.onnx")
    p.add_argument("--refs", type=Path, default=ROOT / "references/v0_7")
    p.add_argument("--standin-sift", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--backend-url", default=None)
    p.add_argument("--strict-limits", action="store_true",
                   help="plan with arm_3 <= 45 deg (PAL dual-arm URDF) instead of the model's 90 deg")
    p.add_argument("--behind-allowance", type=float, default=None,
                   help="optional constraint: arm may go at most this far (m) further back than in the tuck pose")
    p.add_argument("--check-tuck", default=None, metavar="Q1,...,Q7",
                   help="check a candidate right-arm home pose (limits, collisions, head-camera visibility)")
    p.add_argument("--tune-tuck", action="store_true",
                   help="interactive: tune the right-arm home pose with the viewer + live head-camera window")
    p.add_argument("--teach", action="store_true",
                   help="interactive: pose the right arm by typed joint commands, record keyframes, save unfold")
    p.add_argument("--precompute-unfold", action="store_true",
                   help="offline: plan the tuck -> ready unfolding path thoroughly and save it to caremate/data/")
    p.add_argument("--check-stations", action="store_true",
                   help="only report robot-furniture clearance at the NBV base stations (arms tucked) and exit")
    a = p.parse_args()
    global ARM3_UPPER, BEHIND_ALLOWANCE
    BEHIND_ALLOWANCE = a.behind_allowance
    if a.strict_limits:
        ARM3_UPPER = math.radians(45)
    from caremate.demo_v0_7 import DemoCamera
    if a.check_tuck:
        cam = DemoCamera(a.scene)
        try:
            check_tuck(cam, [float(v) for v in a.check_tuck.split(",")])
        finally:
            cam.close()
        return
    if a.tune_tuck:
        tune_cli(a.scene)
        return
    if a.teach:
        teach_cli(a.scene)
        return
    if a.precompute_unfold:
        cam = DemoCamera(a.scene)
        cam._set(START)
        try:
            precompute_unfold(cam)
        finally:
            cam.close()
        return
    if a.check_stations:
        cam = DemoCamera(a.scene)
        kin = Kin(cam.model)
        print(f"MuJoCo contact filter allows robot-furniture contacts: {kin.contact_filter_robot_static}")
        st = read_state(kin, cam.data)
        d = _plan_copy(cam)
        for name, (x, y, yaw) in STATIONS.items():
            for torso in (0.0, 0.35):
                put(kin, d, replace(st, x=x, y=y, yaw=yaw, torso=torso))
                dist, p_ = kin._sets_min(d, kin.robot_col, kin.static_obs, 0.5)
                pair = "" if p_ is None else f"{kin.gname(p_[0])}|{kin.gname(p_[1])}"
                print(f"{name:12s} torso={torso:.2f}  min robot-furniture distance {dist:+.3f} m  {pair}")
        cam.close()
        return
    backend = make_backend(a)
    print(f"{backend.method}; warm-up {backend.warmup():.1f}s")
    cam = DemoCamera(a.scene)
    cam._set(START)
    out = ROOT / "outputs/fetch_v0_7" / f"{datetime.now():%Y%m%d_%H%M%S}_{a.target}_{a.policy}"
    last = {}
    try:
        result = run_agent(a.target, backend, cam, POLICIES[a.policy](0), a.budget, out, scene=a.scene.name,
                           on_step=lambda res, **kw: last.update(res=res))
        log = run_fetch(cam, result, last.get("res"), out, a.target, animate=False)
        print(json.dumps({k: v for k, v in log.items() if k != "phases"}, indent=2))
    finally:
        cam.close()
        backend.close()


if __name__ == "__main__":
    main()
