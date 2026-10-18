"""Diagnose CareMate v0.5 cross-view ALIKED/LightGlue homography rejection.

Read-only: does not change the reference gallery, MuJoCo scene, or existing probe.
Run from the CareMate project root, with the caremate environment active:

  python inspect_caremate_geometry_v0_5.py \
    --reference references/v0_5/reading_book.png \
    --query outputs/roi_debug/book_tilt.png \
    --aliked models/vision/aliked.onnx \
    --lightglue models/vision/lightglue_aliked.onnx

Needs caremate.perception_probe_v0_5, OpenCV 5, numpy, and existing ONNX weights.
The output is diagnostic evidence only, not verified object identity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from caremate.perception_probe_v0_5 import FeaturePipeline, load_bgr


def summarize_geometry(corners: np.ndarray, frame_h: int, frame_w: int) -> dict:
    """Same geometric gates as the existing probe; report every gate separately."""
    finite = bool(np.isfinite(corners).all())
    if not finite:
        return {"finite_polygon": False}
    p = corners.astype(np.float32).reshape(4, 2)
    area = float(abs(cv2.contourArea(p)))
    convex = bool(cv2.isContourConvex(p))
    fraction = area / (frame_w * frame_h)
    inside = bool(((p[:, 0] >= 0) & (p[:, 0] < frame_w) &
                   (p[:, 1] >= 0) & (p[:, 1] < frame_h)).all())
    return {
        "finite_polygon": True,
        "polygon_xy": np.round(p, 2).tolist(),
        "projected_area_px2": round(area, 2),
        "query_area_px2": frame_w * frame_h,
        "projected_area_fraction_of_query": round(fraction, 4),
        "area_at_least_80px2": area >= 80,
        "area_at_most_75pct_of_query": area <= .75 * frame_w * frame_h,
        "convex_polygon": convex,
        "all_projected_corners_inside_query": inside,  # informational only; not a v0.5 gate
    }


def feature_spread(pts: np.ndarray, width: int, height: int) -> dict:
    """How much of an image is supported by the inlier keypoints?"""
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return {"n": len(pts), "hull_area_fraction": 0.0}
    hull = cv2.convexHull(pts)
    return {
        "n": len(pts),
        "hull_area_fraction": round(float(cv2.contourArea(hull)) / (width * height), 4),
        "bbox_xyxy": np.array([pts[:, 0].min(), pts[:, 1].min(),
                               pts[:, 0].max(), pts[:, 1].max()]).round(1).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--query', type=Path, required=True)
    parser.add_argument('--aliked', type=Path, required=True)
    parser.add_argument('--lightglue', type=Path, required=True)
    parser.add_argument('--out', type=Path,
                        default=Path('outputs/perception_v0_5/geometry_diagnostic'))
    args = parser.parse_args()
    ref = load_bgr(args.reference)
    query = load_bgr(args.query)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    pipeline = FeaturePipeline('aliked', args.aliked, args.lightglue)
    kr, kq, matches, matcher_status = pipeline.correspondences(ref, query)
    report: dict = {
        "reference": str(args.reference), "query": str(args.query),
        "reference_wh": [ref.shape[1], ref.shape[0]],
        "query_wh": [query.shape[1], query.shape[0]],
        "n_reference_keypoints": len(kr), "n_query_keypoints": len(kq),
        "n_matches": len(matches), "matcher_status": matcher_status,
    }
    if len(matches) < 6:
        report['result'] = 'insufficient matches for verification'
    else:
        src = np.float32([kr[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kq[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        method = cv2.USAC_MAGSAC if hasattr(cv2, 'USAC_MAGSAC') else cv2.RANSAC
        H, mask = cv2.findHomography(src, dst, method, 3.5)
        if H is None or mask is None or not bool(np.isfinite(H).all()):
            report['result'] = 'homography failed'
        else:
            accepted = mask.ravel().astype(bool)
            ni = int(accepted.sum())
            ratio = ni / len(matches)
            h, w = ref.shape[:2]
            qh, qw = query.shape[:2]
            corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(corners, H).reshape(4, 2)
            geom = summarize_geometry(warped, qh, qw)
            proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
            errors = np.linalg.norm(proj - dst.reshape(-1, 2), axis=1)
            report.update({
                "n_inliers": ni, "inlier_ratio": round(ratio, 4),
                "reprojection_error_inliers_px": {
                    "median": round(float(np.median(errors[accepted])), 3) if ni else None,
                    "p90": round(float(np.percentile(errors[accepted], 90)), 3) if ni else None,
                },
                "reference_inlier_spread": feature_spread(src[accepted], w, h),
                "query_inlier_spread": feature_spread(dst[accepted], qw, qh),
                "geometry": geom,
            })
            gates = {
                "min_six_inliers": ni >= 6,
                "min_35pct_inliers": ratio >= .35,
                "min_area_80px2": geom.get('area_at_least_80px2', False),
                "max_area_75pct_of_query": geom.get('area_at_most_75pct_of_query', False),
                "convex_polygon": geom.get('convex_polygon', False),
                "finite_polygon": geom.get('finite_polygon', False),
            }
            report['original_probe_gates'] = gates
            report['failed_gates'] = [name for name, passed in gates.items() if not passed]
            # Draw original H, even if v0.5 would reject it: diagnostic only.
            img = query.copy()
            for xy in dst.reshape(-1, 2)[accepted]:
                x, y = map(int, xy)
                if 0 <= x < qw and 0 <= y < qh:
                    cv2.circle(img, (x, y), 2, (50, 230, 50), -1, cv2.LINE_AA)
            if bool(np.isfinite(warped).all()) and np.max(np.abs(warped)) < 100000:
                poly = np.clip(np.rint(warped), -100000, 100000).astype(np.int32)
                cv2.polylines(img, [poly], True, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(out / 'raw_polygon_and_inliers.png'), img)
            inlier_matches = [m for m, ok in zip(matches, accepted) if ok]
            # Limit lines for legibility, distribute sampling across correspondences.
            if len(inlier_matches) > 80:
                idxs = np.linspace(0, len(inlier_matches) - 1, 80, dtype=int)
                inlier_matches = [inlier_matches[i] for i in idxs]
            lines = cv2.drawMatches(ref, kr, query, kq, inlier_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(str(out / 'inlier_correspondences.png'), lines)
            report['result'] = 'original_gate_pass' if all(gates.values()) else 'original_gate_rejected'
    (out / 'geometry_debug.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('matches:', report['n_matches'], 'inliers:', report.get('n_inliers'))
    print('inlier ratio:', report.get('inlier_ratio'))
    print('polygon area / query area:', report.get('geometry', {}).get('projected_area_fraction_of_query'))
    print('polygon convex:', report.get('geometry', {}).get('convex_polygon'))
    print('inlier hull area fractions: ref=', report.get('reference_inlier_spread', {}).get('hull_area_fraction'),
          'query=', report.get('query_inlier_spread', {}).get('hull_area_fraction'))
    print('failed original gates:', report.get('failed_gates', 'unavailable'))
    print('Saved diagnostics in', out)
    print('No verification thresholds were changed; a high match count is NOT proof of identity.')


if __name__ == '__main__':
    main()
