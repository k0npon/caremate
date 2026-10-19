"""Pure-logic tests for agent_loop_v0_7 (no rendering, no ONNX)."""
import math

import numpy as np

from caremate import agent_loop_v0_7 as A
from caremate.perception_v0_7 import PerceptionResult


def res(status, border=False, competitors=None, inl=40):
    return PerceptionResult(target="t", status=status, image_id="x", method="test",
                            n_inliers=inl, touches_border=border, competitors=competitors or {},
                            support_bbox_xyxy=[100, 100, 200, 300] if status != "not_found" else None)


def test_stop_only_on_clean_candidate():
    assert A.decide(res("candidate"), False)[0] == "STOP_LOCALISED"
    assert A.decide(res("candidate", border=True), False)[0] == "REOBSERVE"
    assert A.decide(res("ambiguous"), False)[0] == "REOBSERVE"
    assert A.decide(res("not_found"), True)[0] == "ABSTAIN"
    unresolved = {"x": {"resolution": "unresolved"}}
    assert A.decide(res("candidate", competitors=unresolved), False)[0] == "REOBSERVE"


def test_backproject_roundtrip():
    # Camera 1 m above table looking straight down (MuJoCo: looks along -z).
    pose = {"pos": [0.0, 0.0, A.TABLE_Z + 1.0], "mat": np.eye(3).ravel().tolist(), "fovy_deg": 60.0}
    p = A.backproject_to_table((640, 480), pose, 1280, 960)
    assert np.allclose(p, [0, 0, A.TABLE_Z], atol=1e-6)
    f = 480 / math.tan(math.radians(30))
    p = A.backproject_to_table((640 + f * 0.2, 480), pose, 1280, 960)
    assert np.allclose(p[:2], [0.2, 0.0], atol=1e-6)


def test_fixed_policy_sequence():
    pol = A.FixedPolicy()
    hist = []
    for v in A.FIXED_SEQUENCE:
        nv, _ = pol.next_view(hist, None)
        assert nv == v
        hist.append({"view": nv})
    assert pol.next_view(hist, None)[0] is None
