"""Logic tests for perception_v0_7 using SIFT as a stand-in extractor.

No ONNX weights needed: checks letterboxing, tiling, coordinate mapping,
geometry gate and candidate/ambiguous/not_found semantics. ALIKED numbers
must still be measured separately.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from caremate import perception_v0_7 as P

REFS = Path(__file__).resolve().parents[1] / "references" / "v0_7"


class SiftStandIn(P.AlikedExtractor):
    def __init__(self, size=640):
        self.size, self.min_score, self.pad_margin, self.name = size, 0.0, 4, "SIFT-standin"
        self.net = cv2.SIFT_create(1000)


def paste(frame, ref, x, y, long_side=240):
    s = long_side / max(ref.shape[:2])
    r = cv2.resize(ref, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    frame[y:y + r.shape[0], x:x + r.shape[1]] = r


@pytest.fixture(scope="module")
def backend():
    if not (REFS / "crunch_caramel__az90.png").is_file():
        pytest.skip("references/v0_7 missing")
    return P.PerceptionBackend(None, REFS, extractor=SiftStandIn(), ref_sizes=(150, 220, 320))


def scene(backend, present):
    f = np.full((960, 1280, 3), 150, np.uint8)
    for i, name in enumerate(present):
        paste(f, backend.refs[name]['az90'], 120 + 560 * i, 300)
    return f


def test_letterbox_preserves_aspect():
    c, s, (nw, nh) = SiftStandIn(640).letterbox(np.zeros((960, 1280, 3), np.uint8))
    assert c.shape == (640, 640, 3) and s == 0.5 and (nw, nh) == (640, 480)


def test_target_found(backend):
    r = backend.perceive(scene(backend, ["reading_book"]), "reading_book")
    assert r.status == "candidate" and r.n_inliers >= 12 and not r.calibrated


def test_target_absent(backend):
    r = backend.perceive(scene(backend, ["cereal_box"]), "reading_book")
    assert r.status == "not_found"


def test_lookalike_is_not_a_candidate(backend):
    r = backend.perceive(scene(backend, ["crunch_caramel"]), "crunch_peanut")
    assert r.status in ("ambiguous", "not_found")


def test_image_id_changes_with_pixels(backend):
    a = scene(backend, ["reading_book"]); b = a.copy(); b[0, 0] = 0
    assert P.image_id(a) != P.image_id(b)
