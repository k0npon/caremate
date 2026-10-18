"""CareMate v0.5: OpenCV 5 ALIKED + BF(L2) diagnostic over existing RGB ROIs.

Why: LightGlueMatcher may crash natively (SIGFPE) on some ROIs. This diagnostic
uses the SAME OpenCV 5 ALIKED embeddings with a classical BFMatcher(L2) and
USAC_MAGSAC geometric verification, avoiding LightGlue entirely. It loads
ALIKED ONCE and extracts the reference ONCE. No simulator segmentation / labels.

This is a fallback/ablation, NOT a replacement for learned LightGlue, and
its scores are not calibrated identity probabilities. Do not authorize grasps.

Example, from ~/Desktop/caremate:
  python -m caremate.fast_match_bf_v0_5 \\
      --query outputs/sweep_20261018_085321/rgb/v001_t0.000_p+0.00_k-0.60.png \\
      --reference references/v0_5/reading_book.png \\
      --aliked models/vision/aliked.onnx \\
      --proposals outputs/perception_v0_5/fast_verified/proposals.json \\
      --out outputs/perception_v0_5/fast_bf_check
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from caremate.auto_localise_v0_5 import evaluate
from caremate.fast_localise_v0_5 import propose_regions


def load_image(p: Path):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f'Cannot read image: {p}')
    return im


def match_bf(ref_desc, qry_desc, threshold=0.8):
    """Same L2 Lowe-ratio diagnostic as diagnose_aliked_v0_5.py."""
    if ref_desc is None or qry_desc is None or len(ref_desc) < 2 or len(qry_desc) < 2:
        return []
    dr = np.asarray(ref_desc, dtype=np.float32)
    dq = np.asarray(qry_desc, dtype=np.float32)
    if not np.isfinite(dr).all() or not np.isfinite(dq).all():
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(dr, dq, k=2)
    return [a for pair in pairs if len(pair) == 2
            for a, b in [pair] if a.distance < threshold * b.distance]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--query', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--aliked', type=Path, required=True)
    p.add_argument('--proposals', type=Path, default=None,
                   help='Existing fast_verified/proposals.json to use IDENTICAL boxes')
    p.add_argument('--out', type=Path, default=Path('outputs/perception_v0_5/fast_bf_check'))
    p.add_argument('--max-rois', type=int, default=8)
    p.add_argument('--working-width', type=int, default=640)
    p.add_argument('--ratio', type=float, default=0.8)
    p.add_argument('--min-inliers', type=int, default=12)
    args = p.parse_args()
    if not (0.1 < args.ratio < 1.0) or args.min_inliers < 4:
        p.error('Use 0.1 < --ratio < 1.0 and --min-inliers >= 4')
    for f in (args.query, args.reference, args.aliked):
        if not f.is_file():
            p.error(f'File missing: {f}')
    frame = load_image(args.query)
    ref = load_image(args.reference)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.proposals:
        payload = json.loads(args.proposals.read_text())
        regions = payload['regions']
        w, h = frame.shape[1], frame.shape[0]
        if payload.get('image_wh') != [w, h]:
            p.error('Existing proposals.json image dimensions do not match query')
        origin = 'existing_json'
    else:
        regions, _ = propose_regions(frame, args.max_rois, args.working_width)
        origin = 'fresh_proposals'

    start = time.perf_counter()
    extractor = cv2.ALIKED_create(str(args.aliked))
    ref_k, ref_d = extractor.detectAndCompute(ref, None)
    ref_k = list(ref_k) if ref_k is not None else []
    results = []
    overlay = frame.copy()

    print(f'ALIKED + BF(L2) + USAC: {len(regions)} ROIs, reference features={len(ref_k)}', flush=True)
    for i, region in enumerate(regions):
        tic = time.perf_counter()
        x0, y0, x1, y1 = map(int, region['bbox_xyxy'])
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(frame.shape[1], x1), min(frame.shape[0], y1)
        report = {'roi_index': i, 'bbox_xyxy': [x0, y0, x1, y1]}
        if x1 <= x0 or y1 <= y0:
            report.update(status='empty_roi', n_matches=0, n_inliers=0)
        else:
            roi = frame[y0:y1, x0:x1]
            try:
                roi_k, roi_d = extractor.detectAndCompute(roi, None)
                roi_k = list(roi_k) if roi_k is not None else []
                matches = match_bf(ref_d, roi_d, args.ratio)
                details = evaluate(ref_k, roi_k, matches,
                                   (x1-x0, y1-y0), (x0, y0),
                                   (ref.shape[1], ref.shape[0]), args.min_inliers)
                report.update(details)
                report.update(n_ref_features=len(ref_k), n_roi_features=len(roi_k))
            except Exception as exc:
                report.update(status='python_error', error=f'{type(exc).__name__}: {exc}',
                              n_matches=0, n_inliers=0)
        report['elapsed_seconds'] = round(time.perf_counter()-tic, 4)
        results.append(report)
        col = (0, 255, 0) if report['status'] == 'tentative' else (0, 190, 255)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), col, 2)
        cv2.putText(overlay, f"{i}: {report['n_inliers']} inliers", (x0, max(25, y0-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, col, 2, cv2.LINE_AA)
        print(f" ROI {i:02d}: features={report.get('n_roi_features',0):4d} "
              f"matches={report['n_matches']:3d} inliers={report['n_inliers']:3d} "
              f"status={report['status']} time={report['elapsed_seconds']:.2f}s", flush=True)

    elapsed = time.perf_counter() - start
    ranked = sorted(results, key=lambda r: (r['status']=='tentative', r.get('ranking_score',0)), reverse=True)
    best = next((r for r in ranked if r['status'] == 'tentative'), None)
    cv2.imwrite(str(args.out/'bf_candidates.png'), overlay)
    summary = {
        'query': str(args.query), 'reference': str(args.reference),
        'method': 'OpenCV 5 ALIKED + BFMatcher(NORM_L2, ratio) + USAC_MAGSAC',
        'proposal_source': origin, 'proposals_file': str(args.proposals) if args.proposals else None,
        'opencv_version': cv2.__version__, 'reference_features': len(ref_k),
        'number_of_rois': len(regions), 'total_seconds': round(elapsed, 3),
        'best_tentative': best, 'results': results,
        'interpretation': 'Tentative matching evidence only; check image alignment and negatives; not calibrated or grasp-safe.'
    }
    (args.out/'bf_results.json').write_text(json.dumps(summary, indent=2))
    print(f'DONE {len(regions)} ROIs in {elapsed:.2f}s; tentative={best is not None}. '
          f'Results: {args.out}/bf_candidates.png and bf_results.json')

if __name__ == '__main__':
    main()
