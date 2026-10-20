"""Enrolment checks (SIFT stand-in; no ONNX weights needed)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from caremate import enrol_v0_7 as E
from caremate.perception_v0_7 import PerceptionBackend

REFS = Path(__file__).resolve().parents[1] / "references" / "v0_7"


@pytest.fixture(scope="module")
def backend():
    from caremate.agent_loop_v0_7 import SiftStandIn
    return PerceptionBackend(None, REFS, extractor=SiftStandIn())


def test_plain_object_rejected(backend):
    plain = np.full((300, 220, 3), (40, 90, 200), np.uint8)
    cv2.circle(plain, (110, 150), 60, (60, 110, 220), -1)
    r = E.check(backend, "plain_mug", "front", plain)
    assert not r["accepted"] and "texture" in r["reason"]


def test_textured_box_accepted_and_lookalike_reported(backend):
    img = cv2.imread(str(REFS / "crunch_caramel__az90.png"))
    r = E.check(backend, "new_snack", "front", E.normalise(img))
    assert r["accepted"] and r["self_match_inliers"] >= 12
    assert any(s["reference"] == "crunch_caramel" for s in r["similar_to"])
    assert "__enrol_candidate__" not in backend.refs


def test_names():
    assert E.NAME_RE.match("blister_pack_a") and not E.NAME_RE.match("Bad Name")
