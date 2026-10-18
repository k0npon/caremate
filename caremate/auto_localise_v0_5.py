"""CareMate v0.5: RGB-only, automatic tiled object-search diagnostic.

Purpose: remove *manual cropping of query images*. Search an RGB camera frame
using overlapping image tiles, ALIKED, LightGlue, and USAC/MAGSAC. This is a
coarse-to-local feature-verification experiment, NOT a calibrated detector.

All inputs are RGB images; no MuJoCo segmentation, object positions, or depth.
The detected bounding box encloses MATCHED VISUAL SUPPORT, not necessarily the
complete 3D object silhouette. Never use it directly as a grasp pose.

From ~/Desktop/caremate:
  python -m caremate.auto_localise_v0_5 \\
      --query outputs/sweep_20261018_085321/rgb/v001_t0.000_p+0.00_k-0.60.png \\
      --reference references/v0_5/reading_book.png \\
      --aliked models/vision/aliked.onnx \\
      --lightglue models/vision/lightglue_aliked.onnx

Diagnostic shortcut: --query outputs/roi_debug/book_tilt.png

Tiling is deliberately simple (and slower than an object detector) so that the
first RGB-only localization result is easy to audit. Replace with a detector or
faster region proposal later if warranted by runtime measurements.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from caremate.perception_probe_v0_5 import FeaturePipeline, load_bgr


def starts(length: int, side: int, stride: int) -> list[int]:
    """Grid anchored to both image edges, even when side does not divide size."""
    if length <= side:
        return [0]
    loc = list(range(0, length - side + 1, stride))
    if loc[-1] != length - side:
        loc.append(length - side)
    return loc


def tile_grid(width: int, height: int, size: int, stride: int):
    for y in starts(height, size, stride):
        for x in starts(width, size, stride):
            yield (x, y, min(size, width - x), min(size, height - y))


def extract_matches(pipeline, ref_features, tile):
    kr, dr = ref_features
    kq, dq = pipeline.features(tile)
    if dr is None or dq is None or len(kr) < 4 or len(kq) < 4:
        return kr, kq, []
    pr = np.array([k.pt for k in kr], dtype=np.float32)
    pq = np.array([k.pt for k in kq], dtype=np.float32)
    # OpenCV 5 LightGlue's keypoint spatial context must use actual image sizes.
    pipeline.matcher.setPairInfo(pr, pq, (int(pipeline._ref_width), int(pipeline._ref_height)),
                                 (int(tile.shape[1]), int(tile.shape[0])))
    try:
        matches = list(pipeline.matcher.match(dr, dq))
    finally:
        pipeline.matcher.clearPairInfo()
    return kr, kq, [m for m in matches if 0 <= m.queryIdx < len(kr) and 0 <= m.trainIdx < len(kq)]


def evaluate(kr, kq, matches, tile_size, offset, ref_size, min_inliers):
    """Full geometry statistics; no 75%-of-crop constraint.

    We reject weak or degenerate evidence, but never claim a calibrated
    probability of correct identification.
    """
    x0, y0 = offset
    tw, th = tile_size
    rw, rh = ref_size
    out = {"n_matches": len(matches), "n_inliers": 0, "inlier_ratio": 0.0,
           "median_reprojection_px": None, "inlier_hull_fraction": 0.0,
           "visible_projected_fraction": None, "support_bbox_xyxy": None,
           "projected_polygon_global": None, "status": "too_few_matches"}
    if len(matches) < max(6, min_inliers):
        return out
    source = np.array([kr[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
    target = np.array([kq[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
    algo = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    try:
        H, mask = cv2.findHomography(source, target, algo, 3.5)
    except cv2.error:
        H, mask = None, None
    if H is None or mask is None or not np.isfinite(H).all():
        out["status"] = "homography_failed"
        return out
    flags = mask.ravel().astype(bool)
    count = int(np.count_nonzero(flags))
    ratio = float(count / len(matches))
    out.update(n_inliers=count, inlier_ratio=round(ratio, 4))
    if count < 4:
        out["status"] = "too_few_inliers"
        return out
    src_in = source[flags].reshape(-1, 1, 2)
    trg_in = target[flags].reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src_in, H).reshape(-1, 2)
    errors = np.linalg.norm(projected - trg_in.reshape(-1, 2), axis=1)
    med = float(np.median(errors))
    convex_hull = cv2.convexHull(trg_in)
    hull_frac = float(cv2.contourArea(convex_hull) / max(tw * th, 1)) if count >= 3 else 0.0
    rect = np.float32([[0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]]).reshape(-1, 1, 2)
    poly = cv2.perspectiveTransform(rect, H).reshape(4, 2)
    good_polygon = bool(np.isfinite(poly).all() and cv2.isContourConvex(poly.astype(np.float32)))
    visible = 0.0
    if good_polygon:
        area = abs(cv2.contourArea(poly))
        if area >= 1.0:
            bounds = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float32)
            try:
                clipped_area, _ = cv2.intersectConvexConvex(poly.astype(np.float32), bounds)
                visible = float(np.clip(clipped_area / area, 0.0, 1.0))
            except cv2.error:
                visible = 0.0
        poly_global = poly + np.array([x0, y0], dtype=np.float32)
        out["projected_polygon_global"] = np.round(poly_global, 1).tolist()
    global_pts = trg_in.reshape(-1, 2) + np.array([x0, y0], dtype=np.float32)
    bx0, by0 = np.floor(global_pts.min(axis=0)).astype(int)
    bx1, by1 = np.ceil(global_pts.max(axis=0)).astype(int)
    out.update(median_reprojection_px=round(med, 3),
               inlier_hull_fraction=round(hull_frac, 4),
               visible_projected_fraction=round(visible, 4),
               support_bbox_xyxy=[int(bx0), int(by0), int(bx1), int(by1)])
    # Inlier count alone is not enough: require spatial support, low reprojection
    # error, and some part of the projected reference to fall inside the tile.
    acceptable = (count >= min_inliers and ratio >= 0.35 and med <= 2.5
                  and hull_frac >= 0.015 and good_polygon and visible >= 0.15)
    out["status"] = "tentative" if acceptable else "weak_geometry"
    # Heuristic ranking only, *not* calibrated identity confidence.
    out["ranking_score"] = round(float(count) * ratio * min(1.0, hull_frac / 0.15), 3)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--query", required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--aliked", required=True, type=Path)
    p.add_argument("--lightglue", required=True, type=Path)
    p.add_argument("--tile", type=int, default=320, help="Square tile size, default 320")
    p.add_argument("--stride", type=int, default=160, help="Tile stride, default 160")
    p.add_argument("--min-inliers", type=int, default=12)
    p.add_argument("--max-tiles", type=int, default=0, help="0 = entire grid (default). Diagnostics only")
    p.add_argument("--out", type=Path, default=Path("outputs/perception_v0_5/auto_localise"))
    args = p.parse_args()
    if args.tile < 64 or args.stride < 1 or args.min_inliers < 4:
        p.error("--tile >= 64, --stride >= 1, --min-inliers >= 4 are required")
    frame, ref = load_bgr(args.query), load_bgr(args.reference)
    h, w = frame.shape[:2]
    pipeline = FeaturePipeline("aliked", args.aliked, args.lightglue)
    pipeline._ref_width = int(ref.shape[1])
    pipeline._ref_height = int(ref.shape[0])
    ref_features = pipeline.features(ref)
    if ref_features[1] is None:
        raise RuntimeError("ALIKED could not extract descriptors from the reference")
    grid = list(tile_grid(w, h, args.tile, args.stride))
    if args.max_tiles:
        grid = grid[:args.max_tiles]
    print(f"RGB-only scan: image={w}x{h}, ref={ref.shape[1]}x{ref.shape[0]}, {len(grid)} tiles")
    start = time.perf_counter()
    reports = []
    for index, (x, y, tw, th) in enumerate(grid):
        tile = frame[y:y+th, x:x+tw]
        kr, kq, matches = extract_matches(pipeline, ref_features, tile)
        r = evaluate(kr, kq, matches, (tw, th), (x, y),
                     (ref.shape[1], ref.shape[0]), args.min_inliers)
        r.update(tile_index=index, tile_xywh=[x, y, tw, th], n_ref_features=len(kr), n_tile_features=len(kq))
        reports.append(r)
        if (index + 1) % 5 == 0 or index + 1 == len(grid):
            best = max((z["n_inliers"] for z in reports), default=0)
            print(f"  {index+1}/{len(grid)} tiles | best inliers so far: {best}")
    elapsed = time.perf_counter() - start
    ranked = sorted(reports, key=lambda r: (r["status"] == "tentative", r.get("ranking_score", 0)), reverse=True)
    best = next((r for r in ranked if r["status"] == "tentative"), None)
    args.out.mkdir(parents=True, exist_ok=True)
    overlay = frame.copy()
    if best is not None:
        xyxy = best["support_bbox_xyxy"]
        cv2.rectangle(overlay, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 255, 0), 3)
        cv2.putText(overlay, f"TENTATIVE {best['n_inliers']} inliers", (max(5,xyxy[0]), max(24,xyxy[1]-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        tx, ty, tw, th = best["tile_xywh"]
        cv2.rectangle(overlay, (tx, ty), (tx + tw, ty + th), (0, 190, 255), 2)
    else:
        cv2.putText(overlay, "NO RELIABLE CANDIDATE: ABSTAIN", (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(args.out / "auto_candidates.png"), overlay)
    result = {
        "query": str(args.query), "reference": str(args.reference),
        "opencv_version": cv2.__version__, "image_wh": [w, h],
        "tile_size": args.tile, "stride": args.stride, "n_tiles": len(grid),
        "elapsed_seconds": round(elapsed, 2),
        "decision": "TENTATIVE_VISUAL_EVIDENCE_ONLY" if best else "ABSTAIN_NO_CANDIDATE",
        "best": best, "top5": ranked[:5], "all_tiles": reports,
        "limitations": [
            "RGB-only tiled search, not an independently validated object detector",
            "Matched-point box covers visible textured support, not complete object outline",
            "Thresholds not calibrated on held-out negatives: NEVER authorize grasping",
            "Scan runtime must be measured before real-time agent or AWS deployment"
        ]
    }
    (args.out / "auto_candidates.json").write_text(json.dumps(result, indent=2))
    if best:
        print(f"BEST: tile={best['tile_xywh']} matches={best['n_matches']} inliers={best['n_inliers']} "
              f"ratio={best['inlier_ratio']} bbox={best['support_bbox_xyxy']} (TENTATIVE)")
    else:
        print("No candidate passed conservative experimental checks. Don't force a positive!")
    print(f"Elapsed {elapsed:.1f}s; outputs: {args.out}/auto_candidates.png and auto_candidates.json")


if __name__ == "__main__":
    main()
