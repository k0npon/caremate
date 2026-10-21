"""Mirror an arm pose to the other arm (y -> -y in the base frame).

  python scripts/mirror_tuck.py                          left TUCK_LEFT  -> right (original use)
  python scripts/mirror_tuck.py --to-left Q1,...,Q7      right pose      -> left


Does not rely on joint-axis conventions: tries all 128 sign patterns of TUCK_LEFT as seeds, then
refines with Gauss-Newton so that every right-arm link / finger position equals the mirrored left one.
Also reports collision distances of both arms against the torso column and torso shell.

Run from ~/Desktop/caremate:  MUJOCO_GL=egl python scripts/mirror_tuck.py
"""
import argparse
import itertools
import math

import mujoco
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--to-left", default=None, help="right-arm pose to mirror onto the left arm")
args = ap.parse_args()
XML = "models/pal_tiago_dual/caremate_room_v0_7.xml"
TUCK_LEFT = np.array((-1.10, 1.47, 2.71, 1.71, -1.57, 1.37, 0.0))
TUCK_RIGHT_OLD = np.array((-1.10, 1.47, -2.71, 1.71, 1.57, 1.37, 0.0))
TORSO = 0.35

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
J, B = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_BODY


def jadr(name):
    return int(m.jnt_qposadr[mujoco.mj_name2id(m, J, name)])


SRC, DST = ("right", "left") if args.to_left else ("left", "right")
SRC_Q = np.array([float(v) for v in args.to_left.split(",")]) if args.to_left else TUCK_LEFT
L = [jadr(f"arm_{SRC}_{i}_joint") for i in range(1, 8)]
R = [jadr(f"arm_{DST}_{i}_joint") for i in range(1, 8)]
rj = [mujoco.mj_name2id(m, J, f"arm_{DST}_{i}_joint") for i in range(1, 8)]
lo, hi = m.jnt_range[rj, 0] + 1e-3, m.jnt_range[rj, 1] - 1e-3
bodies = [f"arm_{{}}_{i}_link" for i in range(2, 8)] + ["gripper_{}_right_finger_link", "gripper_{}_left_finger_link"]
Lb = [mujoco.mj_name2id(m, B, b.format(SRC)) for b in bodies]
Rb = [mujoco.mj_name2id(m, B, b.format(DST)) for b in bodies]
# the left gripper's "right" finger mirrors to the right gripper's "left" finger
Rb[-2], Rb[-1] = Rb[-1], Rb[-2]
base = mujoco.mj_name2id(m, B, "base_link")
d.qpos[jadr("torso_lift_joint")] = TORSO
d.qpos[L] = SRC_Q
mujoco.mj_kinematics(m, d)
Rb0 = d.xmat[base].reshape(3, 3)
p0 = d.xpos[base].copy()


def to_base(p):
    return (p - p0) @ Rb0


M = np.diag([1, -1, 1])
target = np.array([to_base(d.xpos[b]) @ M for b in Lb])
sh_l = to_base(d.xpos[mujoco.mj_name2id(m, B, f"arm_{SRC}_1_link")]) @ M
sh_r = to_base(d.xpos[mujoco.mj_name2id(m, B, f"arm_{DST}_1_link")])
print(f"model symmetry check: |mirrored left shoulder - right shoulder| = {np.linalg.norm(sh_l - sh_r) * 1000:.1f} mm")


def err(q):
    d.qpos[R] = q
    mujoco.mj_kinematics(m, d)
    return (np.array([to_base(d.xpos[b]) for b in Rb]) - target).ravel()


best = None
seeds = [np.clip(np.array(sg) * SRC_Q, lo, hi) for sg in itertools.product((1, -1), repeat=7)]
same = SRC_Q.copy()                                  # this model: mirror = same angles (joint 3 mod 2 pi)
same[2] = min((same[2] + k * 2 * math.pi for k in (-1, 0, 1)), key=lambda v: 0 if lo[2] <= v <= hi[2] else 1)
seeds.append(np.clip(same, lo, hi))
for q in seeds:
    e = float(np.linalg.norm(err(q)))
    if best is None or e < best[0]:
        best = (e, q)
q = best[1]
for _ in range(100):                                   # Gauss-Newton with joint limits
    r = err(q)
    Jm = np.empty((len(r), 7))
    for k in range(7):
        dq = q.copy()
        dq[k] += 1e-5
        Jm[:, k] = (err(dq) - r) / 1e-5
    step = -np.linalg.solve(Jm.T @ Jm + 1e-6 * np.eye(7), Jm.T @ r)
    q = np.clip(q + step, lo, hi)
    if np.linalg.norm(step) < 1e-7:
        break
e = err(q).reshape(-1, 3)
print(f"best sign seed error {best[0] * 1000:.1f} mm -> refined max link error {np.linalg.norm(e, axis=1).max() * 1000:.2f} mm")
print(f"TUCK_{DST.upper()} = (" + ", ".join(f"{v:.4f}" for v in q) + ")")
print(f"source ({SRC}) = (" + ", ".join(f"{v:.4f}" for v in SRC_Q) + ")")

# collision distances (MuJoCo convex shapes) vs the torso column / shell, both arms
torso_body = int(m.jnt_bodyid[mujoco.mj_name2id(m, J, "torso_lift_joint")])
body_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in (base, torso_body)]


def arm_geoms(side):
    root = mujoco.mj_name2id(m, B, f"arm_{side}_1_link")
    out = []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        while b > 0 and b != root:
            b = int(m.body_parentid[b])
        if b == root:
            out.append(g)
    return out


def report(side, qq):
    d.qpos[L] = TUCK_LEFT
    d.qpos[R] = qq
    mujoco.mj_kinematics(m, d)
    worst = (1.0, "")
    for a in arm_geoms(side):
        for b in body_geoms:
            dist = mujoco.mj_geomDistance(m, d, a, b, 0.3, None)
            if dist < worst[0]:
                worst = (dist, f"g{a}({mujoco.mj_id2name(m, B, m.geom_bodyid[a])})|g{b}")
    return worst


if not args.to_left:
    for label, side, qq in (("left  (reference)", "left", TUCK_RIGHT_OLD), ("right OLD", "right", TUCK_RIGHT_OLD),
                            ("right MIRRORED", "right", q)):
        dist, pair = report(side, qq)
        print(f"{label:18s} min distance to torso/base: {dist:+.3f} m  {pair}")
