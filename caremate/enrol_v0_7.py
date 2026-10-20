"""CareMate v0.7: checks before a new object is accepted as a reference.

Staff enrol an object by uploading one photo of a printed, flat-ish face
(box, carton, book cover).  Before it is used for verification we test:

  1. size       - enough pixels to be useful
  2. texture    - gradient energy; plain mugs/bottles have too little for
                  local-feature verification (they need colour/shape cues)
  3. matchable  - the reference must find ITSELF in a synthetic, smaller,
                  ~40 deg perspective view on a grey background, through the
                  exact same pipeline the robot uses
  4. look-alike - does any existing reference also match it?  Not a reason
                  to reject (the colour check separates look-alikes), but the
                  operator is told, and it is reported in the response.

Heuristic acceptance tests, not a guarantee of field performance (curved
surfaces, glare and real-camera noise are not modelled by check 3).
"""
from __future__ import annotations

import re

import cv2
import numpy as np

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
FACE_RE = re.compile(r"^[a-z0-9]{1,16}$")
MIN_SIDE = 120
MAX_SIDE = 480            # stored long side, same as make_reference_v0_7
MIN_TEXTURE = 150.0       # textured packaging scores ~380-800; plain mug ~40


def texture_score(bgr: np.ndarray) -> float:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def normalise(img: np.ndarray, bbox=None) -> np.ndarray:
    if bbox is not None:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        h, w = img.shape[:2]
        x0, x1 = max(0, min(x0, x1)), min(w, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(h, max(y0, y1))
        img = img[y0:y1, x0:x1]
    s = MAX_SIDE / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return img


def synthetic_view(ref: np.ndarray, frame_wh=(1280, 960), height_px=220, tilt=0.35):
    """Paste the reference into a grey frame with perspective (side-on ~40 deg) and shrink."""
    rh, rw = ref.shape[:2]
    s = height_px / rh
    w, h = rw * s, rh * s
    cx, cy = frame_wh[0] / 2, frame_wh[1] / 2
    dst = np.float32([[cx - w / 2, cy - h / 2], [cx + w / 2 * (1 - tilt), cy - h / 2 * (1 - tilt / 2)],
                      [cx + w / 2 * (1 - tilt), cy + h / 2 * (1 - tilt / 2)], [cx - w / 2, cy + h / 2]])
    src = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]])
    H = cv2.getPerspectiveTransform(src, dst)
    frame = np.full((frame_wh[1], frame_wh[0], 3), 128, np.uint8)
    warped = cv2.warpPerspective(ref, H, frame_wh, flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((rh, rw), 255, np.uint8), H, frame_wh) > 0
    frame[mask] = warped[mask]
    return frame


def check(backend, name: str, face: str, img: np.ndarray) -> dict:
    """Run all acceptance checks.  Caller holds the backend lock."""
    rep = {"name": name, "face": face, "size_wh": [img.shape[1], img.shape[0]]}
    if min(img.shape[:2]) < MIN_SIDE:
        return {**rep, "accepted": False, "reason": f"too small (short side < {MIN_SIDE}px)"}
    tex = texture_score(img)
    rep["texture_score"] = round(tex, 1)
    if tex < MIN_TEXTURE:
        return {**rep, "accepted": False,
                "reason": f"too little printed texture ({tex:.0f} < {MIN_TEXTURE:.0f}); local-feature "
                          "verification would be unreliable for this object"}
    tmp = "__enrol_candidate__"
    backend.add_reference(tmp, face, img)
    try:
        view = synthetic_view(img)
        fq, qs = backend.query_features(view)
        inst, best = backend.hypotheses(tmp, fq, (view.shape[1], view.shape[0]), qs)
        rep["self_match_inliers"] = int((inst[0] if inst else best)["n_inliers"])
        if not inst:
            return {**rep, "accepted": False,
                    "reason": "could not re-find itself in a smaller, tilted view; use a sharper, "
                              "front-on, well-lit photo of a printed face"}
        similar = []
        for other in list(backend.refs):
            if other in (tmp, name):
                continue
            oi, _ = backend.hypotheses(other, fq, (view.shape[1], view.shape[0]), qs)
            if oi:
                similar.append({"reference": other, "inliers": int(oi[0]["n_inliers"])})
        rep["similar_to"] = similar
    finally:
        backend.remove_reference(tmp)
    rep["accepted"] = True
    rep["note"] = ("looks like existing reference(s); the colour look-alike check will be used to tell "
                   "them apart" if rep["similar_to"] else "distinct from existing references")
    return rep
