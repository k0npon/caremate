"""Logic tests for caremate.fetch_v0_7 on a small stand-in robot (NOT TIAGo++).

The stand-in reuses TIAGo++ joint/body/site names so the real code paths run:
depth back-projection, base-pose search, DLS IK, static-map collision checks,
close-until-contact, kinematic attach, delivery and the abstain path.
Real behaviour must still be checked on the v0.7 room with the TIAGo++ model.
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import mujoco
import numpy as np
import pytest
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from caremate import fetch_v0_7 as F  # noqa: E402
from caremate.run_sim import hold_target  # noqa: E402
from caremate.workspace_v0_7 import camera_pose_from_mujoco, project_points  # noqa: E402

ACT = "".join(f'<position name="hold_{j}" joint="{j}" kp="50"/>' for j in
              ["torso_lift_joint", "head_1_joint", "head_2_joint", *F.ARM, *F.FINGERS])

XML = f"""
<mujoco>
  <option gravity="0 0 0"/>
  <compiler angle="radian" inertiafromgeom="true" boundmass="0.01" boundinertia="1e-5"/>
  <default><default class="robot"><geom contype="2" conaffinity="2"/></default>
    <default class="obj"><geom contype="3" conaffinity="3"/></default></default>
  <worldbody>
    <light pos="1 0 3"/>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <!-- main table: top z 0.76, x 0.80-2.00, y -0.825..-0.075 (as in the v0.7 room) -->
    <body name="caremate_open_table">   <!-- jointless child of the world, as in the v0.7 room -->
    <geom name="caremate_open_table_top" type="box" size=".6 .375 .025" pos="1.4 -.45 .735"/>
    <geom name="caremate_open_table_apron" type="box" size=".59 .26 .05" pos="1.4 -.45 .66"/>
    <geom name="t_leg1" type="box" size=".025 .025 .305" pos=".835 -.215 .305"/>
    <geom name="t_leg2" type="box" size=".025 .025 .305" pos=".835 -.685 .305"/>
    <geom name="t_leg3" type="box" size=".025 .025 .305" pos="1.965 -.215 .305"/>
    <geom name="t_leg4" type="box" size=".025 .025 .305" pos="1.965 -.685 .305"/>
    </body>
    <!-- delivery table + wheelchair -->
    <geom name="delivery_table_top" type="box" size=".3 .3 .02" pos="2.15 .46 .72"/>
    <geom name="delivery_leg" type="box" size=".03 .03 .35" pos="2.15 .46 .35"/>
    <geom name="wheelchair_seat" type="box" size=".3 .28 .055" pos="2.7 1.0 .4"/>
    <site name="delivery_goal" pos="2.0 .4 .805" size=".02"/>
    <body name="crunch_peanut" pos="1.0 -.3 .855"><freejoint name="crunch_peanut_free"/>
      <geom name="peanut_g" class="obj" type="box" size=".03 .075 .095" rgba=".9 .6 .1 1"/></body>
    <body name="cereal_box" pos="1.55 -.6 .9"><freejoint name="cereal_box_free"/>
      <geom name="cereal_g" class="obj" type="box" size=".03 .1 .14" rgba=".2 .4 .9 1"/></body>
    <body name="base_link" pos="0 0 0" childclass="robot"><freejoint name="reference"/>
      <geom name="base_g" type="cylinder" size=".27 .15" pos="0 0 .16"/>
      <geom name="column_g" type="box" size=".04 .05 .3" pos="-.05 0 .45"/>  <!-- fixed column inside the shell -->
      <body name="torso" pos="0 0 .32">
        <joint name="torso_lift_joint" type="slide" axis="0 0 1" range="0 .35"/>
        <geom name="torso_g" type="box" size=".1 .12 .3" pos="-.05 0 .3"/>
        <body name="head_1" pos="0 0 .72"><joint name="head_1_joint" axis="0 0 1" range="-1.3 1.3"/>
          <body name="head_2"><joint name="head_2_joint" axis="0 -1 0" range="-1 .8"/>
            <geom name="head_g" type="box" size=".05 .08 .05"/>
            <camera name="head_rgb" pos=".06 0 0" xyaxes="0 -1 0 0 0 1" fovy="58"/>
          </body></body>
        <body name="arm_right_1_link" pos="0 -.22 .5">
          <joint name="arm_right_1_joint" axis="0 0 1" range="-1.178 1.571"/>
          <body name="arm_right_2_link"><joint name="arm_right_2_joint" axis="0 1 0" range="-1.178 1.571"/>
            <body name="arm_right_3_link"><joint name="arm_right_3_joint" axis="1 0 0" range="-3.534 1.571"/>
              <geom name="upper_g" type="capsule" fromto="0 0 0 .3 0 0" size=".04"/>
              <body name="arm_right_4_link" pos=".3 0 0"><joint name="arm_right_4_joint" axis="0 1 0" range="-.393 2.356"/>
                <body name="arm_right_5_link"><joint name="arm_right_5_joint" axis="1 0 0" range="-2.094 2.094"/>
                  <geom name="fore_g" type="capsule" fromto="0 0 0 .3 0 0" size=".035"/>
                  <body name="arm_right_6_link" pos=".3 0 0"><joint name="arm_right_6_joint" axis="0 1 0" range="-1.414 1.414"/>
                    <body name="arm_right_7_link"><joint name="arm_right_7_joint" axis="1 0 0" range="-2.094 2.094"/>
                      <geom name="palm_g" type="box" size=".03 .05 .02" pos=".03 0 0"/>
                      <body name="gripper_right_right_finger_link" pos=".07 0 0">
                        <joint name="gripper_right_right_finger_joint" type="slide" axis="0 -1 0" range="0 .045"/>
                        <geom name="fr_g" type="box" size=".05 .004 .012" pos=".05 -.004 0"/>
                        <geom name="fr_base" type="box" size=".012 .012 .012" pos=".0 .012 0"/></body>
                      <body name="gripper_right_left_finger_link" pos=".07 0 0">
                        <joint name="gripper_right_left_finger_joint" type="slide" axis="0 1 0" range="0 .045"/>
                        <geom name="fl_g" type="box" size=".05 .004 .012" pos=".05 .004 0"/>
                        <geom name="fl_base" type="box" size=".012 .012 .012" pos=".0 -.012 0"/></body>
                    </body></body></body></body></body></body></body>
      </body>
    </body>
  </worldbody>
  <contact><exclude body1="torso" body2="base_link"/></contact>
  <actuator>{ACT}</actuator>
</mujoco>
"""


class FakeCam:
    """Minimal stand-in for MuJoCoHeadCamera (same set_robot/plan_data interface)."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.plan_data = mujoco.MjData(self.model)
        self.W, self.H = 1280, 960
        self._base_adr = int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                                                      "reference")])
        for name, v in zip(F.ARM, F.TUCK_RIGHT):
            hold_target(self.model, self.data, name, v)
        self.set_robot(0, 0, 0, 0, 0, -0.4)
        self.last_depth = None

    def set_robot(self, x, y, yaw, torso, pan, tilt, data=None):
        data = self.data if data is None else data
        a = self._base_adr
        data.qpos[a:a + 2] = (x, y)
        data.qpos[a + 3:a + 7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        for j, v in (("torso_lift_joint", torso), ("head_1_joint", pan), ("head_2_joint", tilt)):
            hold_target(self.model, data, j, v)
        mujoco.mj_forward(self.model, data)


def ray_depth(cam, Wd=160, Hd=120):
    """Z-depth image by ray casting (no OpenGL needed in CI)."""
    pose = camera_pose_from_mujoco(cam.model, cam.data)
    R = np.asarray(pose["mat"]).reshape(3, 3)
    o = np.asarray(pose["pos"])
    f = 0.5 * Hd / math.tan(math.radians(pose["fovy_deg"]) / 2)
    depth = np.full((Hd, Wd), 10.0, np.float32)
    gid = np.array([-1], np.int32)
    rb = mujoco.mj_name2id(cam.model, mujoco.mjtObj.mjOBJ_BODY, "head_2")
    for v in range(Hd):
        for u in range(Wd):
            dc = np.array([(u + .5 - Wd / 2) / f, -(v + .5 - Hd / 2) / f, -1.0])
            dist = mujoco.mj_ray(cam.model, cam.data, o, R @ dc, None, 1, rb, gid)
            if dist > 0:
                depth[v, u] = dist      # |dc_z| = 1, so ray parameter == z-depth
    return depth, pose


class FakeResult:
    def __init__(self, cam, body="crunch_peanut", status="candidate"):
        pose = camera_pose_from_mujoco(cam.model, cam.data)
        b = mujoco.mj_name2id(cam.model, mujoco.mjtObj.mjOBJ_BODY, body)
        g = int(np.nonzero(cam.model.geom_bodyid == b)[0][0])
        hs = cam.model.geom_size[g]
        Rb = cam.data.xmat[b].reshape(3, 3)
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * hs
        pts = cam.data.xpos[b] + corners @ Rb.T
        front = pts[pts[:, 0] < cam.data.xpos[b][0]]       # face towards the robot (-x)
        uv, _ = project_points(front, pose, cam.W, cam.H)
        self.status = status
        self.support_bbox_xyxy = [*uv.min(0).tolist(), *uv.max(0).tolist()]
        order = np.argsort(np.arctan2(uv[:, 1] - uv[:, 1].mean(), uv[:, 0] - uv[:, 0].mean()))
        self.projected_polygon = uv[order].tolist()


def _ensure_target_base_estimate():
    try:
        import caremate.agent_loop_v0_7  # noqa: F401  (real repo)
    except Exception:                    # sandbox without perception deps: minimal copy
        mod = types.ModuleType("caremate.agent_loop_v0_7")

        def target_base_estimate(res, pose, width, height):
            poly = np.asarray(res.projected_polygon, float)
            base = poly[np.argsort(poly[:, 1])[-2:]].mean(0)
            R = np.asarray(pose["mat"]).reshape(3, 3)
            fl = 0.5 * height / math.tan(math.radians(pose["fovy_deg"]) / 2)
            d = R @ np.array([(base[0] - width / 2) / fl, -(base[1] - height / 2) / fl, -1.0])
            o = np.asarray(pose["pos"])
            return o + (F.TABLE_Z - o[2]) / d[2] * d
        mod.target_base_estimate = target_base_estimate
        sys.modules["caremate.agent_loop_v0_7"] = mod


STANDIN_TUCK = (-1.10, 1.47, -2.71, 1.71, 1.57, 1.37, 0.0)   # tuck for the stand-in arm (TIAGo's is mirrored)


@pytest.fixture
def cam(monkeypatch):
    monkeypatch.setattr(F, "TUCK_RIGHT", STANDIN_TUCK)
    _ensure_target_base_estimate()
    c = FakeCam()
    c.last_depth, _ = ray_depth(c)
    return c


def test_depth_estimate_near_object(cam):
    res = FakeResult(cam)
    pose = camera_pose_from_mujoco(cam.model, cam.data)
    from caremate.agent_loop_v0_7 import target_base_estimate
    pb = target_base_estimate(res, pose, cam.W, cam.H)
    p, info, _ = F.estimate_grasp_point(res.support_bbox_xyxy, res.projected_polygon, pb, pose,
                                     cam.last_depth, cam.W, cam.H)
    assert info["method"] == "head_depth_in_support_region", info
    # visible face is the -x face at x = 0.97; y centre -0.3; height 0.19 -> grasp z = 0.76 + 0.095
    assert abs(p[0] - 0.995) < 0.03 and abs(p[1] + 0.3) < 0.03, p
    assert abs(info["height_m"] - 0.19) < 0.03 and abs(p[2] - 0.855) < 0.02, (info, p)
    # face plane: normal along +x (thickness), face half-width 0.075 along y
    assert abs(abs(info["face_normal_xy"][0]) - 1) < 0.05 and abs(info["face_half_width_m"] - 0.075) < 0.015, info


def test_attachment_roundtrip(cam):
    kin = F.Kin(cam.model)
    b = mujoco.mj_name2id(cam.model, mujoco.mjtObj.mjOBJ_BODY, "crunch_peanut")
    before = cam.data.xpos[b].copy()
    att = F.Attachment(kin, cam.data, b)
    att.apply(cam.data)
    mujoco.mj_forward(cam.model, cam.data)
    assert np.allclose(cam.data.xpos[b], before, atol=1e-9)


def test_surface_below_delivery(cam):
    kin = F.Kin(cam.model)
    z, g = F.surface_below(kin, cam.plan_data, (2.0, 0.4))
    assert kin.gname(g) == "delivery_table_top" and abs(z - 0.74) < 1e-6


def test_abstain_does_not_move_arm(cam):
    q0 = cam.data.qpos.copy()
    log = F.run_fetch(cam, {"outcome": "ABSTAIN_AMBIGUOUS"}, FakeResult(cam, status="ambiguous"),
                      None, "crunch_peanut", animate=False)
    assert log["outcome"] == "NO_FETCH_CANNOT_VERIFY"
    assert np.allclose(cam.data.qpos, q0)


def test_full_fetch_and_deliver(cam):
    kin = F.Kin(cam.model)
    # the table is a jointless child body: it must still count as furniture (bug in the first version)
    top = mujoco.mj_name2id(cam.model, mujoco.mjtObj.mjOBJ_GEOM, "caremate_open_table_top")
    assert kin.static_geom[top]
    res = FakeResult(cam)
    log = F.run_fetch(cam, {"outcome": "TARGET_LOCALISED_UNCALIBRATED"}, res, None, "crunch_peanut",
                      animate=True)              # animate: every frame is clearance-checked
    print({k: v for k, v in log.items() if k != "phases"})
    assert log["outcome"] == "DELIVERED_KINEMATIC", log
    gt = log["offline_scoring_gt"]
    assert gt["object_delivered"] == "crunch_peanut" and gt["is_requested_target"]
    assert gt["dist_to_delivery_goal_m"] < 0.05 and abs(gt["object_gap_to_table_m"]) < 0.008, gt
    chk = log["execution_check"]
    assert log["joint_limits_check"]["violations"] == [], log["joint_limits_check"]
    assert chk["frames_in_collision"] == 0 and chk["min_robot_furniture_m"] > 0 and chk["min_self_margin_m"] >= 0, chk
    grasp = next(ph for ph in log["phases"] if ph["phase"] == "grasped")
    assert grasp["closing_axis_vs_face_normal"] > 0.9, grasp     # fingers on front + back face
    assert abs(sum(grasp["finger_qpos_m"]) - 0.06) < 0.01, grasp     # fingers stopped on the 6 cm thickness
    b = mujoco.mj_name2id(cam.model, mujoco.mjtObj.mjOBJ_BODY, "crunch_peanut")
    assert cam.data.xmat[b].reshape(3, 3)[2, 2] > 0.99          # placed upright


def test_rrt_connect_goes_around_obstacle():
    """Straight joint path blocked by a slab in (q0, q1); RRT-Connect must find a collision-free detour."""
    lo, hi = np.full(7, -2.0), np.full(7, 2.0)

    def ok(q):
        return not (abs(q[0]) < 0.3 and q[1] < 1.2)      # wall across q0 = 0 except above q1 = 1.2

    a, b = np.r_[-1.0, 0.0, np.zeros(5)], np.r_[1.0, 0.0, np.zeros(5)]
    path = F.rrt_connect(ok, a, b, lo, hi, np.random.default_rng(0), time_budget=10)
    assert path is not None
    assert np.allclose(path[0], a) and np.allclose(path[-1], b)
    dense = F._densify(None, path)
    assert all(ok(q) for q in dense)
    assert all(np.max(np.abs(q1 - q0)) <= F.MAX_JOINT_STEP + 1e-9 for q0, q1 in zip(dense, dense[1:]))


def _box_mesh(c, h):
    v = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


def test_concave_distance_respects_recess():
    """A box in the notch of a U-shaped mesh: MuJoCo's hull says deep penetration, the real surface 6 cm."""
    parts = [_box_mesh([-0.15, 0, 0.1], [0.05, 0.1, 0.1]), _box_mesh([0.15, 0, 0.1], [0.05, 0.1, 0.1]),
             _box_mesh([0, 0, -0.025], [0.2, 0.1, 0.025])]
    V, Fc, off = [], [], 0
    for v, f in parts:
        V.append(v)
        Fc.append(f + off)
        off += len(v)
    V, Fc = np.concatenate(V), np.concatenate(Fc)
    xml = (f'<mujoco><asset><mesh name="U" vertex="{" ".join(map(str, V.ravel()))}" '
           f'face="{" ".join(map(str, Fc.ravel()))}"/>'
           f'<mesh name="B" vertex="{" ".join(map(str, _box_mesh([0, 0, 0], [0.04, 0.04, 0.04])[0].ravel()))}"/>'
           '</asset><worldbody><geom name="u" type="mesh" mesh="U"/>'
           '<body pos="0 0 0.1"><freejoint/><geom name="b" type="mesh" mesh="B"/></body></worldbody></mujoco>')
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert mujoco.mj_geomDistance(m, d, 0, 1, 0.5, None) < -0.1          # hull: "inside"
    planes, samples = F.hull_planes(m, 1), F.surface_samples(m, 0)
    dist = F.concave_distance(d, 1, planes, 0, samples, 0.5)
    assert 0.04 < dist <= 0.0605, dist                                    # true clearance 6 cm
    for z, expect in ((0.03, -0.01), (0.02, -0.02)):                     # sink into the bottom slab
        d.qpos[2] = z
        mujoco.mj_forward(m, d)
        dist = F.concave_distance(d, 1, planes, 0, samples, 0.5)
        assert abs(dist - expect) < 0.004, (z, dist)
    d.qpos[:3] = (0.09, 0, 0.1)                                          # push into the right post
    mujoco.mj_forward(m, d)
    assert F.concave_distance(d, 1, planes, 0, samples, 0.5) < -0.025


def test_telescopic_column_only_exposed_segment(cam):
    kin = F.Kin(cam.model)
    col = mujoco.mj_name2id(cam.model, mujoco.mjtObj.mjOBJ_GEOM, "column_g")
    assert col in kin.column
    d = cam.plan_data
    st = F.read_state(kin, cam.data)
    F.put(kin, d, replace(st, torso=0.0))
    low = kin.shell_bottom(d)
    F.put(kin, d, replace(st, torso=0.35))
    high = kin.shell_bottom(d)
    assert abs((high - low) - 0.35) < 1e-6
    # The shell (torso_g) encloses the column's top: only the exposed segment may count, so the
    # shell itself must not "collide" with the column although their full volumes overlap.
    shell = mujoco.mj_name2id(cam.model, mujoco.mjtObj.mjOBJ_GEOM, "torso_g")
    kin.planes[shell] = F.hull_planes(cam.model, shell)
    for torso in (0.0, 0.35):
        F.put(kin, d, replace(st, torso=torso))
        assert mujoco.mj_geomDistance(cam.model, d, shell, col, 0.3, None) < -0.05
        assert kin.self_distance(d, shell, col, 0.3) > -0.004


def test_precomputed_unfold_roundtrip(cam, tmp_path, monkeypatch):
    """--precompute-unfold writes a path that run-time loading accepts and re-validates."""
    monkeypatch.setattr(F, "UNFOLD_FILE", tmp_path / "unfold.json")
    data = F.precompute_unfold(cam, collect=1, rrt_cap=2, tries=2, budget=5)
    assert data is not None and F.UNFOLD_FILE.is_file()
    kin = F.prepare_kin(cam)
    st = replace(F.read_state(kin, cam.data), torso=F.MANIP_TORSO, fl=None)
    q_ready, path, why = F.load_unfold(cam, kin, st)
    assert q_ready is not None, why
    assert np.allclose(path[-1], q_ready)
    assert all(np.max(np.abs(b - a)) <= F.MAX_JOINT_STEP + 1e-9 for a, b in zip(path, path[1:]))
    monkeypatch.setattr(F, "TUCK_RIGHT", tuple(np.asarray(F.TUCK_RIGHT) + 0.1))   # other tuck: rejected
    assert F.load_unfold(cam, kin, st)[0] is None


def test_teach_session_records_and_saves(cam, tmp_path, monkeypatch):
    """Scripted teach session on the stand-in: limits enforced, colliding keyframes refused, save works."""
    monkeypatch.setattr(F, "UNFOLD_FILE", tmp_path / "unfold.json")
    out = []
    q0 = list(F.TUCK_RIGHT)
    cmds = ["4 += 99",               # clipped to the joint range
            "back", "key",           # tuck itself as keyframe: zero-length move, allowed
            "1 += 0.4", "key", "done"]
    data = F.teach_session(cam, cmds, out=out.append)
    text = "\n".join(out)
    assert "limited to its range" in text
    assert data is not None and F.UNFOLD_FILE.is_file(), text
    kin = F.prepare_kin(cam)
    st = replace(F.read_state(kin, cam.data), torso=F.MANIP_TORSO, fl=None)
    q_ready, path, why = F.load_unfold(cam, kin, st)          # saved path passes the run-time checks
    assert q_ready is not None, why
    assert abs(data["keyframes"][-1][0] - (q0[0] + 0.4)) < 1e-9 or data["keyframes"][-1][0] == kin.hi[0]


def test_check_tuck_limits_and_collisions(cam):
    """Collision / limit part of --check-tuck (head-camera visibility needs OpenGL, checked on the OMEN)."""
    out = []
    assert F.check_tuck(cam, STANDIN_TUCK, out=out.append, render=False), out
    out = []
    bad = list(STANDIN_TUCK)
    bad[3] = 5.0                                                   # outside joint 4's range
    assert not F.check_tuck(cam, bad, out=out.append, render=False)
    assert "outside" in out[0]
