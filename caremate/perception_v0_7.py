"""CareMate v0.7: single-pass, RGB-only reference verification backend.

Why v0.7 replaces the v0.5 tile/ROI pipeline as the *default* backend
(v0.5 scripts are kept unchanged as the historical baseline):

  * The ALIKED ONNX exports have a FIXED square input (640x640 or 1280x1280)
    and a FIXED top-k (1000 or 2000).  v0.5 fed arbitrary-size images, so
    OpenCV resized every frame/crop to 640x640: full 1280x960 frames were
    shrunk, a 184x236 reference was enlarged ~2.7x, and every call cost the
    same.  Fixed top-k also means ALIKED *always* returns k points, including
    low-score points on flat or padded regions.
  * Here every image is letterboxed by US to exactly the model size (no
    hidden resize, aspect preserved), keypoints inside the padding are
    dropped, and coordinates are mapped back to native pixels.
  * References are canonical (frontal) object images, extracted once at
    several apparent sizes and cached; the best scale is reported.
  * Default matcher: BFMatcher(L2) + Lowe ratio + mutual nearest neighbour.
    LightGlue is optional and runs in a crash-isolated worker process, because
    native SIGFPE in LightGlueMatcher.match() is not yet explained.
  * Geometry gate is v0.7-specific (evaluate_v07): v0.5's evaluate() measures
    inlier hull against the *tile* area, which wrongly rejects a correct small
    object in a full frame.  v0.7 measures how much of the REFERENCE the inliers
    cover (scale-free), plus reprojection, convexity and homography scale.

Decision semantics (NOT calibrated, never 'verified'):
  candidate  - target passes geometry gate and no distractor competes
  ambiguous  - another reference passes on overlapping support
  not_found  - target fails the gate (includes target-absent frames)
  error      - extraction/matching failed

Runtime never reads MuJoCo segmentation, object IDs or poses.

Example (from ~/Desktop/caremate):
  python -m caremate.perception_v0_7 \\
      --aliked models/vision/aliked-n16rot-top1k-640.onnx \\
      --refs references/v0_7 --target reading_book \\
      --query outputs/sweep_*/rgb/v001_*.png --debug-keypoints
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


Status = Literal["candidate", "ambiguous", "not_found", "error"]
PAD_VALUE = 127
# Apparent long-side sizes (native query pixels) at which each reference is
# pre-extracted.  Covers the book from ~0.8 m to ~2.5 m in a 1280x960 frame.
DEFAULT_REF_SIZES = (96, 144, 216, 324)


# ---------------------------------------------------------------- results
@dataclass
class PerceptionResult:
    target: str
    status: Status
    image_id: str                      # sha1 of the exact pixels judged
    method: str
    camera_pose: dict | None = None    # supplied by caller (e.g. agent), not inferred
    n_matches: int = 0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    median_reprojection_px: float | None = None
    support_bbox_xyxy: list[int] | None = None
    projected_polygon: list | None = None
    touches_border: bool = False       # support near frame edge -> likely clipped
    best_ref_size_px: int | None = None
    best_ref_face: str | None = None
    competitors: dict = field(default_factory=dict)  # other refs that passed
    rejected_instances: list = field(default_factory=list)  # target matches that were look-alikes
    notes: list = field(default_factory=list)
    per_reference: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)
    n_query_keypoints: int = 0
    error: str | None = None
    calibrated: bool = False

    def __post_init__(self):
        if self.calibrated:
            raise ValueError("No acceptance policy has been validated on held-out negatives")

    def to_json(self) -> dict:
        return asdict(self)


def image_id(bgr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(bgr).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------- ALIKED
def model_input_size(path: Path) -> int:
    """Read the square input size from the ONNX graph, else from the file name."""
    try:
        import onnx
        dims = onnx.load(str(path)).graph.input[0].type.tensor_type.shape.dim
        h, w = dims[2].dim_value, dims[3].dim_value
        if h and h == w:
            return int(h)
    except Exception:
        pass
    m = re.search(r"-(\d{3,4})\.onnx$", path.name)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot determine fixed input size of {path}; rename like *-1280.onnx")


@dataclass
class Features:
    pts: np.ndarray        # (N,2) float32, NATIVE image coordinates
    desc: np.ndarray       # (N,D) float32
    scores: np.ndarray     # (N,)  float32 (KeyPoint.response)
    model_pts: np.ndarray  # (N,2) coordinates in the S x S model canvas
    n_raw: int             # points ALIKED returned before filtering


class AlikedExtractor:
    """ALIKED with explicit letterboxing to the model's fixed square input."""

    def __init__(self, model: Path, min_score: float = 0.0, pad_margin: int = 4):
        self.size = model_input_size(model)
        params = cv2.ALIKED_Params()
        params.inputSize = (self.size, self.size)   # default is (640,640)!
        self.net = cv2.ALIKED_create(str(model), params)
        self.min_score = float(min_score)
        self.pad_margin = int(pad_margin)
        self.name = model.name

    def letterbox(self, bgr: np.ndarray, scale: float | None = None):
        h, w = bgr.shape[:2]
        if scale is None:
            scale = self.size / float(max(w, h))
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        if nw > self.size or nh > self.size:
            raise ValueError(f"Scaled image {nw}x{nh} exceeds model input {self.size}")
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        canvas = np.full((self.size, self.size, 3), PAD_VALUE, np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=interp) if scale != 1 else bgr
        return canvas, scale, (nw, nh)

    def extract(self, bgr: np.ndarray, scale: float | None = None) -> Features:
        canvas, s, (nw, nh) = self.letterbox(bgr, scale)
        kps, desc = self.net.detectAndCompute(canvas, None)
        kps = list(kps) if kps is not None else []
        n_raw = len(kps)
        if desc is None or not kps:
            e = np.zeros((0, 2), np.float32)
            return Features(e, np.zeros((0, 128), np.float32), np.zeros(0, np.float32), e, n_raw)
        mp_ = np.float32([k.pt for k in kps])
        sc = np.float32([k.response for k in kps])
        m = self.pad_margin
        keep = ((mp_[:, 0] < nw - m) & (mp_[:, 1] < nh - m) & (mp_[:, 0] >= m) & (mp_[:, 1] >= m)
                & (sc >= self.min_score) & np.isfinite(desc).all(axis=1))
        mp_, sc, desc = mp_[keep], sc[keep], np.asarray(desc, np.float32)[keep]
        return Features(mp_ / s, desc, sc, mp_, n_raw)

    def extract_tiled(self, bgr: np.ndarray, overlap: int = 96) -> Features:
        """Native-scale extraction over overlapping S x S tiles, merged globally.

        Fixed top-k means a single pass spends most of its budget on the
        checkered floor.  Tiling gives every region its own budget without
        resizing, so references only need native-scale variants.
        """
        h, w = bgr.shape[:2]
        S = self.size
        step = max(1, S - overlap)
        xs = list(range(0, max(1, w - S) + 1, step)) or [0]
        ys = list(range(0, max(1, h - S) + 1, step)) or [0]
        if xs[-1] + S < w: xs.append(w - S)
        if ys[-1] + S < h: ys.append(h - S)
        pts, desc, sc, raw = [], [], [], 0
        for y in ys:
            for x in xs:
                tile = bgr[y:y + S, x:x + S]
                f = self.extract(tile, scale=1.0)
                raw += f.n_raw
                if len(f.pts):
                    pts.append(f.pts + np.float32([x, y])); desc.append(f.desc); sc.append(f.scores)
        if not pts:
            e = np.zeros((0, 2), np.float32)
            return Features(e, np.zeros((0, 128), np.float32), np.zeros(0, np.float32), e, raw)
        pts, desc, sc = np.concatenate(pts), np.concatenate(desc), np.concatenate(sc)
        # Drop duplicates from tile overlaps (keep the higher score).
        order = np.argsort(-sc)
        grid = {}
        keep = []
        for i in order:
            key = (int(pts[i, 0] // 2), int(pts[i, 1] // 2))
            if key in grid:
                continue
            grid[key] = i
            keep.append(i)
        keep = np.array(sorted(keep))
        # model_pts are per-tile coordinates; LightGlue needs one canvas, so the
        # tiled mode is BF-only (enforced by PerceptionBackend).
        return Features(pts[keep], desc[keep], sc[keep], pts[keep], raw)


# ---------------------------------------------------------------- matching
def match_bf_mutual(dr: np.ndarray, dq: np.ndarray, ratio: float = 0.8) -> list[tuple[int, int]]:
    """Lowe ratio on ref->query plus mutual nearest-neighbour check."""
    if len(dr) < 2 or len(dq) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    fwd = bf.knnMatch(dr, dq, k=2)
    back = bf.match(dq, dr)
    best_back = {m.queryIdx: m.trainIdx for m in back}
    out = []
    for pair in fwd:
        if len(pair) < 2:
            continue
        a, b = pair
        if a.distance < ratio * b.distance and best_back.get(a.trainIdx) == a.queryIdx:
            out.append((a.queryIdx, a.trainIdx))
    return out


def _lightglue_worker(model_path: str, size: int, conn) -> None:
    import resource
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (OSError, ValueError):
        pass
    matcher = cv2.LightGlueMatcher_create(model_path)
    while True:
        msg = conn.recv()
        if msg is None:
            return
        pr, pq, dr, dq = msg
        matcher.setPairInfo(pr, pq, (size, size), (size, size))
        try:
            ms = matcher.match(dr, dq)
        finally:
            matcher.clearPairInfo()
        conn.send([(m.queryIdx, m.trainIdx) for m in ms])


class IsolatedLightGlue:
    """LightGlue in a child process; a native crash returns an error, not a dead agent."""

    def __init__(self, model: Path, size: int, timeout: float = 30.0):
        self.model, self.size, self.timeout = str(model), size, timeout
        self.ctx = mp.get_context("spawn")
        self.proc = None
        self.crashes = 0

    def _start(self):
        self.parent, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=_lightglue_worker,
                                     args=(self.model, self.size, child), daemon=True)
        self.proc.start()

    def match(self, fr: Features, fq: Features) -> list[tuple[int, int]]:
        # Guard degenerate inputs; zero-size inputs are a candidate SIGFPE cause.
        if len(fr.desc) < 8 or len(fq.desc) < 8:
            return []
        if self.proc is None or not self.proc.is_alive():
            self._start()
        self.parent.send((fr.model_pts, fq.model_pts, fr.desc, fq.desc))
        if not self.parent.poll(self.timeout):
            self.close()
            raise RuntimeError(f"lightglue_timeout_{self.timeout}s")
        try:
            pairs = self.parent.recv()
        except EOFError:
            self.proc.join(1)
            code = self.proc.exitcode
            self.proc, self.crashes = None, self.crashes + 1
            raise RuntimeError(f"lightglue_native_crash_exitcode_{code}")
        return [(int(a), int(b)) for a, b in pairs
                if 0 <= a < len(fr.desc) and 0 <= b < len(fq.desc)]

    def close(self):
        if self.proc is not None and self.proc.is_alive():
            try:
                self.parent.send(None)
                self.proc.join(2)
            except (BrokenPipeError, OSError):
                pass
            if self.proc.is_alive():
                self.proc.kill()
        self.proc = None


def _overlaps(a, b) -> bool:
    """Same image region?  IoU > 0.3, or one box mostly inside the other."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return _iou(a, b) > 0.3 or inter / max(1, small) > 0.5


# ---------------------------------------------------------------- geometry
def evaluate_v07(ref_pts, qry_pts, pairs, frame_wh, ref_wh, min_inliers=12,
                 thresh_px=3.5, frame_origin=(0, 0)) -> dict:
    """frame_origin: where the full camera frame's (0,0) lies in query coordinates
    negated - i.e. the query image is a crop starting at frame_origin in the frame."""
    """Homography verification. Heuristic gate, NOT a calibrated probability."""
    out = {"n_matches": len(pairs), "n_inliers": 0, "inlier_ratio": 0.0,
           "median_reprojection_px": None, "ref_coverage": 0.0,
           "visible_projected_fraction": None, "support_bbox_xyxy": None,
           "projected_polygon_global": None, "ranking_score": 0.0,
           "status": "too_few_matches"}
    if len(pairs) < max(6, min_inliers):
        return out
    idx_r = np.array([a for a, _ in pairs]); idx_q = np.array([b for _, b in pairs])
    src = ref_pts[idx_r].reshape(-1, 1, 2).astype(np.float32)
    dst = qry_pts[idx_q].reshape(-1, 1, 2).astype(np.float32)
    algo = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    try:
        H, mask = cv2.findHomography(src, dst, algo, thresh_px)
    except cv2.error:
        H, mask = None, None
    if H is None or mask is None or not np.isfinite(H).all():
        out["status"] = "homography_failed"
        return out
    inl = mask.ravel().astype(bool)
    n = int(inl.sum())
    out.update(n_inliers=n, inlier_ratio=round(n / len(pairs), 4))
    if n < 4:
        out["status"] = "too_few_inliers"
        return out
    proj = cv2.perspectiveTransform(src[inl], H).reshape(-1, 2)
    med = float(np.median(np.linalg.norm(proj - dst[inl].reshape(-1, 2), axis=1)))
    rw, rh = ref_wh
    ref_area = float(rw * rh)
    coverage = float(cv2.contourArea(cv2.convexHull(src[inl])) / max(ref_area, 1.0))
    corners = np.float32([[0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]]).reshape(-1, 1, 2)
    poly = cv2.perspectiveTransform(corners, H).reshape(4, 2)
    fw, fh = frame_wh
    convex = bool(np.isfinite(poly).all() and cv2.isContourConvex(poly.astype(np.float32)))
    area = abs(float(cv2.contourArea(poly.astype(np.float32)))) if convex else 0.0
    visible = 0.0
    if convex and area >= 1.0:
        ox, oy = frame_origin
        frame = np.float32([[-ox, -oy], [fw - ox, -oy], [fw - ox, fh - oy], [-ox, fh - oy]])
        try:
            clipped, _ = cv2.intersectConvexConvex(poly.astype(np.float32), frame)
            visible = float(np.clip(clipped / area, 0, 1))
        except cv2.error:
            visible = 0.0
    # Apparent scale of the reference in the frame; reject absurd warps.
    scale = float(np.sqrt(area / ref_area)) if area > 0 else 0.0
    q = dst[inl].reshape(-1, 2)
    x0, y0 = np.floor(q.min(0)).astype(int); x1, y1 = np.ceil(q.max(0)).astype(int)
    out.update(median_reprojection_px=round(med, 3), ref_coverage=round(coverage, 4),
               visible_projected_fraction=round(visible, 4), homography_scale=round(scale, 4),
               H=H.tolist(), projected_area_px=round(area, 1),
               support_bbox_xyxy=[int(x0), int(y0), int(x1), int(y1)],
               projected_polygon_global=np.round(poly, 1).tolist() if convex else None)
    ok = (n >= min_inliers and out["inlier_ratio"] >= 0.35 and med <= 2.5 and convex
          and coverage >= 0.10 and visible >= 0.30 and area >= 400 and 0.02 <= scale <= 20)
    out["status"] = "tentative" if ok else "weak_geometry"
    out["ranking_score"] = round(n * out["inlier_ratio"] * min(1.0, coverage / 0.3), 3)
    return out


# ---------------------------------------------------------------- look-alikes
def _gray(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) if x.ndim == 3 else x.astype(np.float32)


def _ncc(a, b, m):
    a, b = a[m], b[m]
    if a.size < 150:
        return None
    a = a - a.mean(); b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-6 else None


def _ab(x):
    """Lab chroma (a*, b*) as float: insensitive to brightness/shading changes."""
    lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[..., 1:3]


def discriminate(frame, ref_t, ref_c, H, H_ct=None, roi_mask=None, diff_thresh=18.0,
                 min_px=60):
    """Compare two look-alike references ONLY where their COLOURS differ.

    Packaging variants differ mainly in hue (e.g. a red vs purple flavour
    band), which grey-level comparisons miss.  Steps:
      1. register the competitor reference onto the target reference (H_ct);
      2. blur both and mark pixels whose Lab chroma differs by > diff_thresh;
      3. warp both references into the frame with the target homography H,
         blur the frame and the warps alike (fine text is not resolvable at
         this distance anyway);
      4. mean chroma distance frame<->target vs frame<->competitor on the mask.
    Returns (margin, d_target, d_competitor, n_pixels); margin in [-1, 1],
    positive = region looks like the target.  None if not measurable.
    Heuristic appearance check, not a probability.
    """
    if H_ct is None:
        return None, None, None, 0
    th, tw = ref_t.shape[:2]
    Hc = np.asarray(H_ct, np.float64)
    rc = cv2.warpPerspective(ref_c, Hc, (tw, th))
    valid = cv2.warpPerspective(np.full(ref_c.shape[:2], 255, np.uint8), Hc, (tw, th)) > 0
    k = max(3, (min(th, tw) // 40) | 1)
    dab = np.linalg.norm(_ab(cv2.GaussianBlur(ref_t, (k, k), 0)) - _ab(cv2.GaussianBlur(rc, (k, k), 0)), axis=2)
    dmask = ((dab > diff_thresh) & valid).astype(np.uint8) * 255
    dmask = cv2.morphologyEx(dmask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fh, fw = frame.shape[:2]
    Hm = np.asarray(H, np.float64)
    wt = cv2.warpPerspective(ref_t, Hm, (fw, fh), flags=cv2.INTER_AREA)
    wc = cv2.warpPerspective(rc, Hm, (fw, fh), flags=cv2.INTER_AREA)
    wm = cv2.warpPerspective(dmask, Hm, (fw, fh), flags=cv2.INTER_NEAREST) > 0
    wm = cv2.erode(wm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if roi_mask is not None:
        wm &= roi_mask > 0
    n = int(wm.sum())
    if n < min_px:
        return None, None, None, n
    blur = lambda x: cv2.GaussianBlur(x, (5, 5), 0)
    f, t, c = _ab(blur(frame))[wm], _ab(blur(wt))[wm], _ab(blur(wc))[wm]
    d_t = float(np.linalg.norm(f - t, axis=1).mean())
    d_c = float(np.linalg.norm(f - c, axis=1).mean())
    margin = (d_c - d_t) / max(d_c + d_t, 1e-6)
    return margin, d_t, d_c, n


# ---------------------------------------------------------------- backend
def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(1, ua)


class PerceptionBackend:
    def __init__(self, aliked: Path, refs: Path, backend: str = "bf",
                 lightglue: Path | None = None, ratio: float = 0.8,
                 min_inliers: int = 12, ref_sizes=DEFAULT_REF_SIZES,
                 min_score: float = 0.0, extractor=None, query_mode: str = "tiled"):
        self.extractor = extractor or AlikedExtractor(aliked, min_score=min_score)
        self.backend, self.ratio, self.min_inliers = backend, ratio, min_inliers
        self.colour_margin = 0.35   # look-alike decision margin; tune on validation views
        self.min_area_px = 4000     # smaller projected targets are too small to judge identity
        if query_mode not in ("tiled", "letterbox"):
            raise ValueError("query_mode must be tiled or letterbox")
        if query_mode == "tiled" and backend == "lightglue":
            raise ValueError("LightGlue needs one canvas: use --query-mode letterbox")
        self.query_mode = query_mode
        self.ref_sizes = tuple(int(s) for s in ref_sizes)
        # One object may have several reference faces: <name>__<face>.png.
        self.refs: dict[str, dict[str, np.ndarray]] = {}
        for p in sorted(Path(refs).glob("*.png")):
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                raise RuntimeError(f"Cannot read reference {p}")
            name, _, face = p.stem.partition("__")
            self.refs.setdefault(name, {})[face or "default"] = im
        if not self.refs:
            raise FileNotFoundError(f"No reference PNGs in {refs}")
        self.lg = None
        if backend == "lightglue":
            if lightglue is None:
                raise ValueError("--backend lightglue needs --lightglue")
            self.lg = IsolatedLightGlue(lightglue, self.extractor.size)
        self._cache: dict[tuple[str, int, float], Features] = {}
        self.method = (f"OpenCV {cv2.__version__} ALIKED[{getattr(self.extractor, 'name', 'custom')}] + "
                       + ("BF(L2,ratio,mutual)" if backend == "bf" else "LightGlue(isolated)")
                       + f" + USAC_MAGSAC [{query_mode}]")

    def _ref_features(self, name: str, size_px: int, query_scale: float, face: str = "default") -> Features:
        """Reference resized so its long side spans `size_px` native query pixels.

        Points are returned in ORIGINAL reference-image coordinates, so every
        scale variant yields a homography from the same canonical reference.
        """
        key = (name, face, size_px, round(query_scale, 4))
        if key not in self._cache:
            ref = self.refs[name][face]
            rh, rw = ref.shape[:2]
            s = size_px * query_scale / float(max(rw, rh))
            s = min(s, self.extractor.size / float(max(rw, rh)))
            self._cache[key] = self.extractor.extract(ref, scale=s)
        return self._cache[key]

    def _ref_alignment(self, src: str, src_face: str, dst: str, dst_face: str):
        """Homography registering reference src onto reference dst (cached, offline-able)."""
        key = ("align", src, src_face, dst, dst_face)
        if key not in self._cache:
            H = None
            a, b = self.refs[src][src_face], self.refs[dst][dst_face]
            fa = self.extractor.extract(a, scale=min(1.0, self.extractor.size / max(a.shape[:2])))
            fb = self.extractor.extract(b, scale=min(1.0, self.extractor.size / max(b.shape[:2])))
            pairs = match_bf_mutual(fa.desc, fb.desc, self.ratio)
            if len(pairs) >= 12:
                pa = fa.pts[[i for i, _ in pairs]]; pb = fb.pts[[j for _, j in pairs]]
                Hm, m = cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, 3.0)
                if Hm is not None and m is not None and int(m.sum()) >= 12:
                    H = Hm
            self._cache[key] = H
        return self._cache[key]

    def add_reference(self, name: str, face: str, img: np.ndarray) -> None:
        """Register (or replace) one reference face at runtime and drop stale caches."""
        self.refs.setdefault(name, {})[face] = img
        self._purge(name)

    def remove_reference(self, name: str) -> None:
        self.refs.pop(name, None)
        self._purge(name)

    def _purge(self, name: str) -> None:
        for k in [k for k in self._cache if name in k]:
            del self._cache[k]

    def warmup(self) -> float:
        """Pre-extract every reference face/size (tiled mode) so the first frame isn't slow."""
        t0 = time.perf_counter()
        if self.query_mode == "tiled":
            for name, faces in self.refs.items():
                for face in faces:
                    for size in self.ref_sizes:
                        self._ref_features(name, size, 1.0, face)
        return time.perf_counter() - t0

    def _match(self, fr: Features, fq: Features):
        if self.backend == "bf":
            return match_bf_mutual(fr.desc, fq.desc, self.ratio)
        return self.lg.match(fr, fq)

    def hypotheses(self, name: str, fq: Features, frame_wh, query_scale: float,
                   max_instances: int = 3, frame_origin=(0, 0)):
        """All plausible instances of reference `name` in the frame.

        Sequential RANSAC: after an instance passes the geometry gate, the
        query keypoints inside its support box are removed and the search
        repeats, so a reference can match several boxes (e.g. a look-alike
        AND the real target).  Returns (tentative_instances, best_report).
        """
        found, best = [], None
        for face, ref in self.refs[name].items():
            rh, rw = ref.shape[:2]
            for size in self.ref_sizes:
                fr = self._ref_features(name, size, query_scale, face)
                pairs = self._match(fr, fq)
                for _ in range(max_instances):
                    rep = evaluate_v07(fr.pts, fq.pts, pairs, frame_wh, (rw, rh), self.min_inliers,
                                       frame_origin=frame_origin)
                    rep.update(ref_size_px=size, ref_face=face, n_ref_keypoints=int(len(fr.pts)))
                    key = (rep["status"] == "tentative", rep.get("ranking_score", 0.0), rep["n_inliers"])
                    if best is None or key > best[0]:
                        best = (key, rep)
                    if rep["status"] != "tentative":
                        break
                    found.append(rep)
                    x0, y0, x1, y1 = rep["support_bbox_xyxy"]
                    q = fq.pts[[j for _, j in pairs]] if pairs else np.zeros((0, 2))
                    inside = ((q[:, 0] >= x0 - 10) & (q[:, 0] <= x1 + 10) &
                              (q[:, 1] >= y0 - 10) & (q[:, 1] <= y1 + 10))
                    pairs = [pr for pr, ins in zip(pairs, inside) if not ins]
        found.sort(key=lambda r: r["ranking_score"], reverse=True)
        unique = []
        for r in found:      # NMS across faces/sizes
            if not any(_overlaps(r["support_bbox_xyxy"], u["support_bbox_xyxy"]) for u in unique):
                unique.append(r)
        return unique, best[1]

    def verify_reference(self, name: str, fq: Features, frame_wh, query_scale: float) -> dict:
        """Best single instance (kept for tools that expect one report)."""
        inst, best = self.hypotheses(name, fq, frame_wh, query_scale)
        return inst[0] if inst else best

    def _resolve(self, bgr, target, t, n, r, roi_mask):
        """Look-alike check of target instance t against competitor instance r."""
        info = {"ranking_score": r["ranking_score"], "n_inliers": r["n_inliers"],
                "competitor_bbox": r["support_bbox_xyxy"]}
        face = t["ref_face"] if t["ref_face"] in self.refs[n] else r["ref_face"]
        H_ct = self._ref_alignment(n, face, target, t["ref_face"])
        m1, _, _, n1 = discriminate(bgr, self.refs[target][t["ref_face"]],
                                    self.refs[n][face], t["H"], H_ct, roi_mask)
        H_tc = self._ref_alignment(target, t["ref_face"], n, r["ref_face"])
        m2, _, _, n2 = discriminate(bgr, self.refs[n][r["ref_face"]],
                                    self.refs[target][t["ref_face"]], r["H"], H_tc, roi_mask)
        m2 = None if m2 is None else -m2      # flip: positive = target
        info.update(margin_targetH=None if m1 is None else round(m1, 3), px_targetH=n1,
                    margin_competitorH=None if m2 is None else round(m2, 3), px_competitorH=n2)
        ms = [m for m in (m1, m2) if m is not None]
        margin = float(np.mean(ms)) if ms else None
        info["colour_margin"] = None if margin is None else round(margin, 3)
        # Conservative: every measurable direction must agree and clear the margin.
        if not ms or any(abs(m) < self.colour_margin for m in ms) or (len(ms) == 2 and ms[0] * ms[1] < 0):
            info["resolution"] = "unresolved"
        elif margin < 0:
            info["resolution"] = "region_is_competitor"
        else:
            info["resolution"] = "target_wins"
        return info

    def query_features(self, bgr: np.ndarray, roi_polygon: np.ndarray | None = None):
        """Query features in native pixels; optionally restricted to an ROI polygon.

        Outside the polygon is filled with flat grey and the frame is cropped to
        the polygon's bounding box, so the fixed top-k budget is spent on the
        workspace and fewer tiles are processed.
        """
        h, w = bgr.shape[:2]
        if roi_polygon is None:
            if self.query_mode == "tiled":
                return self.extractor.extract_tiled(bgr), 1.0
            return self.extractor.extract(bgr), self.extractor.size / float(max(w, h))
        poly = np.asarray(roi_polygon, np.float32)
        x0, y0 = np.floor(poly.min(0)).astype(int).clip(0)
        x1, y1 = np.ceil(poly.max(0)).astype(int)
        x1, y1 = min(w, x1 + 1), min(h, y1 + 1)
        crop = bgr[y0:y1, x0:x1].copy()
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(poly - [x0, y0]).astype(np.int32)], 255)
        crop[mask == 0] = PAD_VALUE
        if self.query_mode == "tiled":
            f, qscale = self.extractor.extract_tiled(crop), 1.0
        else:
            f = self.extractor.extract(crop)
            qscale = self.extractor.size / float(max(crop.shape[:2]))
        # Drop points on the grey fill border (mask eroded by a few pixels).
        inner = cv2.erode(mask, np.ones((7, 7), np.uint8))
        xi = f.pts[:, 0].astype(int).clip(0, crop.shape[1] - 1)
        yi = f.pts[:, 1].astype(int).clip(0, crop.shape[0] - 1)
        keep = inner[yi, xi] > 0
        off = np.float32([x0, y0])
        return Features(f.pts[keep] + off, f.desc[keep], f.scores[keep],
                        f.model_pts[keep], f.n_raw), qscale

    def perceive(self, bgr: np.ndarray, target: str, camera_pose: dict | None = None,
                 roi_polygon: np.ndarray | None = None, origin=(0, 0),
                 frame_wh=None) -> PerceptionResult:
        """origin/frame_wh: `bgr` may be a crop of a larger camera frame (the remote
        client uploads only the workspace region).  `roi_polygon` is then in crop
        coordinates; all returned image coordinates are in FULL-frame pixels and
        frame-edge checks use the full frame, so results equal a full-frame call."""
        iid = image_id(bgr)
        if target not in self.refs:
            return PerceptionResult(target, "error", iid, self.method, camera_pose,
                                    error=f"no reference named {target}")
        lat = {}
        try:
            t0 = time.perf_counter()
            h, w = bgr.shape[:2]
            fq, qscale = self.query_features(bgr, roi_polygon)
            lat["extract_query"] = round(1000 * (time.perf_counter() - t0), 1)
            t1 = time.perf_counter()
            ox, oy = int(origin[0]), int(origin[1])
            FW, FH = frame_wh or (w, h)
            hyps = {n: self.hypotheses(n, fq, (FW, FH), qscale, frame_origin=(ox, oy)) for n in self.refs}
            lat["match_and_geometry"] = round(1000 * (time.perf_counter() - t1), 1)
            lat["total"] = round(1000 * (time.perf_counter() - t0), 1)
        except Exception as exc:  # includes isolated LightGlue crash
            return PerceptionResult(target, "error", iid, self.method, camera_pose,
                                    latency_ms=lat, error=f"{type(exc).__name__}: {exc}")
        roi_mask = None
        if roi_polygon is not None:
            roi_mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(roi_mask, [np.round(roi_polygon).astype(np.int32)], 255)
        # Judge every target instance against overlapping instances of other references.
        judged = []   # (verdict, instance, competitors)
        for t_i in hyps[target][0]:
            comps, verdict = {}, "candidate"
            for n, (insts, _) in hyps.items():
                if n == target:
                    continue
                for r in insts:
                    if not _overlaps(t_i["support_bbox_xyxy"], r["support_bbox_xyxy"]):
                        continue
                    info = self._resolve(bgr, target, t_i, n, r, roi_mask)
                    comps[n] = info
                    if info["resolution"] == "region_is_competitor":
                        verdict = "not_found"
                    elif info["resolution"] == "unresolved" and verdict == "candidate":
                        verdict = "ambiguous"
            judged.append((verdict, t_i, comps))
        rejected = [{"bbox": t_i["support_bbox_xyxy"], "n_inliers": t_i["n_inliers"],
                     "is_actually": [n for n, c in comps.items() if c["resolution"] == "region_is_competitor"]}
                    for v, t_i, comps in judged if v == "not_found"]
        pick = next((j for j in judged if j[0] == "candidate"), None) or \
            next((j for j in judged if j[0] == "ambiguous"), None)
        notes = []
        if pick:
            status, t, competitors = pick
            if status == "candidate" and t.get("projected_area_px", 0) < self.min_area_px:
                status = "ambiguous"
                notes.append(f"target too small to judge identity ({t.get('projected_area_px', 0):.0f} px^2 "
                             f"< {self.min_area_px}); view it closer")
        else:
            status, t, competitors = "not_found", hyps[target][1], {}
        def sh(b):          # crop -> full-frame coordinates
            return None if b is None else [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]
        bb = sh(t.get("support_bbox_xyxy"))
        border = bool(bb and (bb[0] <= 6 or bb[1] <= 6 or bb[2] >= FW - 7 or bb[3] >= FH - 7))
        poly = t.get("projected_polygon_global")
        if poly is not None and (ox or oy):
            poly = [[x + ox, y + oy] for x, y in poly]
        if ox or oy:
            rejected = [{**r, "bbox": sh(r["bbox"])} for r in rejected]
            competitors = {n: {**c, "competitor_bbox": sh(c.get("competitor_bbox"))} for n, c in competitors.items()}
        summary = {n: {**{k: (insts[0] if insts else best).get(k)
                          for k in ("status", "n_matches", "n_inliers", "inlier_ratio",
                                    "ranking_score", "ref_size_px", "ref_face")},
                       "support_bbox_xyxy": sh((insts[0] if insts else best).get("support_bbox_xyxy")),
                       "n_instances": len(insts)}
                   for n, (insts, best) in hyps.items()}
        return PerceptionResult(
            target=target, status=status, image_id=iid, method=self.method,
            camera_pose=camera_pose, n_matches=t["n_matches"], n_inliers=t["n_inliers"],
            inlier_ratio=t["inlier_ratio"], median_reprojection_px=t.get("median_reprojection_px"),
            support_bbox_xyxy=bb, projected_polygon=poly,
            touches_border=border, best_ref_size_px=t.get("ref_size_px"), best_ref_face=t.get("ref_face"),
            competitors=competitors, rejected_instances=rejected, notes=notes,
            per_reference=summary, latency_ms=lat,
            n_query_keypoints=int(len(fq.pts)))

    def close(self):
        if self.lg is not None:
            self.lg.close()


# ---------------------------------------------------------------- CLI
def draw(bgr, res: PerceptionResult, fq: Features | None = None, roi=None):
    vis = bgr.copy()
    if roi is not None:
        cv2.polylines(vis, [np.int32(roi)], True, (255, 255, 0), 1)
    if fq is not None:
        for x, y in fq.pts.astype(int):
            cv2.circle(vis, (int(x), int(y)), 2, (255, 120, 0), -1)
    col = {"candidate": (0, 220, 0), "ambiguous": (0, 200, 255),
           "not_found": (0, 0, 255), "error": (255, 0, 255)}[res.status]
    if res.projected_polygon and res.status != "not_found":
        cv2.polylines(vis, [np.int32(res.projected_polygon)], True, col, 1)
    if res.support_bbox_xyxy and res.status != "not_found":
        x0, y0, x1, y1 = res.support_bbox_xyxy
        cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
    # Matches of the target reference that turned out to be a look-alike.
    for rej in res.rejected_instances or []:
        x0, y0, x1, y1 = rej["bbox"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 140, 255), 2)
        lab = "look-alike: " + ", ".join(rej["is_actually"])
        cv2.putText(vis, lab, (x0, max(18, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, lab, (x0, max(18, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 140, 255), 2, cv2.LINE_AA)
    txt = f"{res.target}: {res.status.upper()} inl={res.n_inliers}/{res.n_matches}"
    if res.touches_border:
        txt += " [BORDER]"
    cv2.putText(vis, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, col, 2, cv2.LINE_AA)
    return vis


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aliked", type=Path, required=True)
    p.add_argument("--refs", type=Path, default=Path("references/v0_7"))
    p.add_argument("--target", default="reading_book")
    p.add_argument("--query", type=Path, nargs="+", required=True)
    p.add_argument("--backend", choices=("bf", "lightglue"), default="bf")
    p.add_argument("--query-mode", choices=("tiled", "letterbox"), default="tiled",
                   help="tiled: native-scale SxS tiles merged (BF only); letterbox: one resized pass")
    p.add_argument("--lightglue", type=Path, default=None)
    p.add_argument("--ratio", type=float, default=0.8)
    p.add_argument("--min-inliers", type=int, default=12)
    p.add_argument("--min-score", type=float, default=0.0,
                   help="Drop ALIKED keypoints with response below this (inspect scores first)")
    p.add_argument("--ref-sizes", type=int, nargs="+", default=list(DEFAULT_REF_SIZES))
    p.add_argument("--debug-keypoints", action="store_true", help="Draw all kept query keypoints")
    p.add_argument("--workspace", default=None,
                   help="Restrict search to this static-map workspace (e.g. main_table); "
                        "needs <query>.pose.json written by viewpoint_sweep")
    p.add_argument("--out", type=Path, default=Path("outputs/perception_v0_7"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    be = PerceptionBackend(a.aliked, a.refs, a.backend, a.lightglue, a.ratio,
                           a.min_inliers, a.ref_sizes, a.min_score, query_mode=a.query_mode)
    print(f"{be.method}; model input {be.extractor.size}px; refs={list(be.refs)}")
    print(f"Reference warm-up: {be.warmup():.1f}s (one-off at agent start)")
    rows = []
    try:
        for q in a.query:
            img = cv2.imread(str(q), cv2.IMREAD_COLOR)
            if img is None:
                print(f"SKIP unreadable {q}")
                continue
            pose, roi = None, None
            if a.workspace:
                from caremate.workspace_v0_7 import workspace_polygon
                pf = q.with_suffix(".pose.json")
                if not pf.is_file():
                    print(f"SKIP {q.name}: --workspace needs {pf.name}")
                    continue
                pose = json.loads(pf.read_text())
                roi = workspace_polygon(pose, img.shape[1], img.shape[0], a.workspace)
            res = be.perceive(img, a.target, camera_pose=pose, roi_polygon=roi)
            fq = None
            if a.debug_keypoints:
                fq, _ = be.query_features(img, roi)
                print(f"   keypoints raw={fq.n_raw} kept={len(fq.pts)} "
                      f"score p10/p50/p90={np.percentile(fq.scores, [10, 50, 90]).round(3).tolist() if len(fq.scores) else []}")
            cv2.imwrite(str(a.out / f"{q.stem}_v07.png"), draw(img, res, fq, roi))
            rows.append({"query": str(q), **res.to_json()})
            others = " ".join(f"{n}:{r['n_inliers']}" for n, r in res.per_reference.items() if n != a.target)
            print(f"{q.name:45s} {res.status:10s} inl={res.n_inliers:3d}/{res.n_matches:3d} "
                  f"face={res.best_ref_face} size={res.best_ref_size_px} border={res.touches_border} "
                  f"{res.latency_ms.get('total', '-')}ms | others {others} {res.error or ''}")
    finally:
        be.close()
    (a.out / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"Saved {a.out}/results.json and *_v07.png overlays. Uncalibrated; never 'verified'.")


if __name__ == "__main__":
    main()
