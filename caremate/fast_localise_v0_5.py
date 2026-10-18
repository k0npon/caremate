"""CareMate: OpenCV-only fast region proposals and crash-isolated learned verification.

This is a scene-specific, fast *proposal* baseline, NOT object recognition.
It does not read MuJoCo ground truth. The optional verification stage uses ALIKED and LightGlue.
Verify ROI recall on held-out poses, target-absent/wrong-object images, and
measure latency. Never infer identity or grasp safety from proposal ranks alone.

Usage from ~/Desktop/caremate:
  python -m caremate.fast_localise_v0_5 \\
      --query outputs/sweep_20261018_085321/rgb/v001_t0.000_p+0.00_k-0.60.png \\
      --out outputs/perception_v0_5/fast_roi
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import math
import time
from pathlib import Path

import cv2
import numpy as np


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1, area_a + area_b - inter)


def propose_regions(bgr, max_rois=8, working_width=640):
    """Return tentative ROI list in ORIGINAL image coordinates and diagnostics.

    Primary candidates are external Canny contours near light-wood tabletop.
    The light-wood assumption is a scene-specific shortcut; it is NOT robust
    across arbitrary environments. If tabletop heuristic fails, report it.
    """
    orig_h, orig_w = bgr.shape[:2]
    scale = min(1., working_width / float(orig_w))
    w = max(1, round(orig_w * scale)); h = max(1, round(orig_h * scale))
    img = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA) if scale < 1 else bgr.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    wood = cv2.inRange(hsv, (7, 20, 85), (34, 150, 255))
    wood = cv2.morphologyEx(wood, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    wood_cs, _ = cv2.findContours(wood, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in wood_cs if cv2.contourArea(c) >= .018 * w * h]
    if not big:
        return [], {'reason': 'table_colour_not_detected', 'proposals_raw': 0,
                    'working_size_wh': [w, h]}
    table = max(big, key=cv2.contourArea)
    tx, ty, tw, th = cv2.boundingRect(table)
    # Extend upwards from tabletop for tall objects; limit search to table area.
    # This may miss objects in a different scene: explicitly record this prior.
    zx0 = max(0, int(tx - .025 * w))
    zx1 = min(w, int(tx + tw + .025 * w))
    zy0 = max(0, int(ty - .42 * h))
    zy1 = min(h, int(ty + .28 * h))
    roi_area = max(1, (zx1 - zx0) * (zy1 - zy0))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edge = cv2.Canny(gray, 38, 112)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # Avoid long table/floor edges by geometric constraints and table-colour test.
    contours, _ = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    raw = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if bw < max(9, round(.012 * w)) or bh < max(13, round(.022 * h)):
            continue
        if (area < max(190, .0004 * w * h) or area > .15 * w * h
                or bw > .32 * w or bh > .5 * h):
            continue
        cx = x + bw / 2; cy = y + bh / 2
        if not (zx0 <= cx <= zx1 and zy0 <= cy <= zy1):
            continue
        if x < zx0 - .025 * w:
            continue
        # Wholly wooden contours are probably table edges, not objects.
        wfrac = float(np.count_nonzero(wood[y:y+bh, x:x+bw])) / area
        if wfrac > .58:
            continue
        contour_area = abs(cv2.contourArea(c))
        # Reject extremely thin line fragments but allow partially occluded items.
        density = contour_area / max(1, area)
        if density < .025:
            continue
        # Prefer sizeable compact foreground regions, without hard category rules.
        compact = min(1., density / .45)
        objectness = math.sqrt(area) * (0.4 + 0.6 * compact) * (1.0 - .7 * wfrac)
        margin = max(6, round(.012 * min(w, h)))
        box = [max(0, x-margin), max(0, y-margin),
               min(w, x+bw+margin), min(h, y+bh+margin)]
        raw.append({'box': box, 'objectness': float(objectness),
                    'wood_fraction': round(wfrac, 3), 'original_bbox': [x,y,bw,bh]})
    raw.sort(key=lambda p: p['objectness'], reverse=True)
    chosen = []
    for candidate in raw:
        # Use high IoU suppression only: don't erase neighbouring small objects.
        if any(iou(candidate['box'], q['box']) > .68 for q in chosen):
            continue
        chosen.append(candidate)
        if len(chosen) >= max_rois:
            break
    for p in chosen:
        xa,ya,xb,yb = p.pop('box')
        p['bbox_xyxy'] = [max(0, int(xa/scale)), max(0,int(ya/scale)),
                          min(orig_w, int(math.ceil(xb/scale))),
                          min(orig_h, int(math.ceil(yb/scale)))]
        p['proposal_score'] = round(p.pop('objectness'), 2)
        p.pop('original_bbox')
    return chosen, {
        'working_size_wh': [w,h],
        'table_bbox_xywh_working': [tx,ty,tw,th],
        'candidate_zone_xyxy_working': [zx0,zy0,zx1,zy1],
        'table_wood_fraction': round(cv2.contourArea(table)/(w*h),4),
        'proposals_raw': len(raw),
        'warning': 'Heuristic depends on light-wood tabletop; validate recall across poses and lighting.'
    }



def stage(x):
    print('STAGE=' + x, flush=True)


def worker(args):
    # A native OpenCV SIGFPE must not kill the agent's main process.
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        pass
    from caremate.perception_probe_v0_5 import FeaturePipeline
    from caremate.auto_localise_v0_5 import evaluate

    stage('load_images')
    ref = cv2.imread(str(args.reference))
    crop = cv2.imread(str(args._crop))
    if ref is None or crop is None:
        raise FileNotFoundError('Worker reference or crop missing')

    stage('init_models')
    pipe = FeaturePipeline('aliked', args.aliked, args.lightglue)
    stage('extract_reference')
    kref, dref = pipe.features(ref)
    stage('extract_roi')
    kroi, droi = pipe.features(crop)

    matches = []
    if (dref is not None and droi is not None
            and len(kref) >= 4 and len(kroi) >= 4):
        a = np.asarray([k.pt for k in kref], dtype=np.float32)
        b = np.asarray([k.pt for k in kroi], dtype=np.float32)
        if (np.isfinite(a).all() and np.isfinite(b).all()
                and np.isfinite(dref).all() and np.isfinite(droi).all()):
            stage('pair_info')
            pipe.matcher.setPairInfo(a, b,
                                     (ref.shape[1], ref.shape[0]),
                                     (crop.shape[1], crop.shape[0]))
            try:
                stage('lightglue_match')
                raw = pipe.matcher.match(dref, droi)
                matches = [m for m in raw if 0 <= m.queryIdx < len(kref)
                           and 0 <= m.trainIdx < len(kroi)]
            finally:
                stage('clear_pair_info')
                pipe.matcher.clearPairInfo()
        else:
            stage('nonfinite_features')
    stage('geometry')
    x, y, bw, bh = args._roi_xywh
    report = evaluate(kref, kroi, matches,
                      (int(bw), int(bh)), (int(x), int(y)),
                      (ref.shape[1], ref.shape[0]), args.min_inliers)
    report.update(roi_index=args._worker_index, roi_xywh=[x, y, bw, bh],
                  n_reference_features=len(kref), n_roi_features=len(kroi))
    args._worker_result.write_text(json.dumps(report, indent=2))
    stage('complete')
    return 0


def status_code(returncode):
    if returncode < 0:
        try:
            return f'native_signal_{signal.Signals(-returncode).name}'
        except ValueError:
            return f'native_signal_{-returncode}'
    return f'worker_exit_{returncode}'


def parent(args):
    img = cv2.imread(str(args.query))
    if img is None:
        raise FileNotFoundError(f'Image not readable: {args.query}')
    args.out.mkdir(parents=True, exist_ok=True)
    crops = args.out / 'crops'
    crops.mkdir(exist_ok=True)
    logs = args.out / 'roi_logs'
    logs.mkdir(exist_ok=True)
    start = time.perf_counter()
    regions, diag = propose_regions(img, args.max_rois, args.working_width)
    proposal_seconds = time.perf_counter() - start
    overlay = img.copy()
    for i, r in enumerate(regions):
        x0,y0,x1,y1 = r['bbox_xyxy']
        cv2.imwrite(str(crops / f'roi_{i:02d}.png'), img[y0:y1, x0:x1])
        cv2.rectangle(overlay, (x0,y0), (x1,y1), (0,220,255), 2)
        cv2.putText(overlay, f'ROI {i}', (x0,max(18,y0-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0,220,255), 2,cv2.LINE_AA)
    cv2.imwrite(str(args.out / 'proposals.png'), overlay)
    proposal_report = {'query': str(args.query), 'opencv_version': cv2.__version__,
                       'proposal_seconds': round(proposal_seconds, 4),
                       'image_wh': [img.shape[1], img.shape[0]],
                       'regions': regions, 'diagnostics': diag,
                       'warning': 'RGB proposals only, NOT verified identities'}
    (args.out / 'proposals.json').write_text(json.dumps(proposal_report, indent=2))
    print(f'RGB proposals: {len(regions)} in {proposal_seconds:.3f}s', flush=True)
    if args.proposals_only:
        print(f'Proposals only: {args.out}/proposals.png', flush=True)
        return 0
    if not (args.reference and args.aliked and args.lightglue):
        raise ValueError('--reference --aliked --lightglue required unless --proposals-only')
    for path in (args.reference, args.aliked, args.lightglue):
        if not path.is_file():
            raise FileNotFoundError(path)

    results = []
    failures = []
    for i, r in enumerate(regions):
        x0,y0,x1,y1 = r['bbox_xyxy']
        result_file = logs / f'roi_{i:02d}.json'
        result_file.unlink(missing_ok=True)
        cmd = [sys.executable, '-u', '-m', 'caremate.fast_localise_v0_5',
               '--reference', str(args.reference), '--aliked', str(args.aliked),
               '--lightglue', str(args.lightglue), '--min-inliers', str(args.min_inliers),
               '--_worker-index', str(i), '--_worker-result', str(result_file),
               '--_crop', str(crops / f'roi_{i:02d}.png'),
               '--_roi-xywh', str(x0), str(y0), str(x1-x0), str(y1-y0)]
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', PYTHONFAULTHANDLER='1')
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, env=env)
            trace = p.stdout + '\n' + p.stderr
            problem = None if p.returncode == 0 and result_file.exists() else status_code(p.returncode)
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode(errors='replace') if isinstance(e.stdout, bytes) else (e.stdout or '')
            stderr = e.stderr.decode(errors='replace') if isinstance(e.stderr, bytes) else (e.stderr or '')
            trace = stdout + '\n' + stderr
            problem = f'timeout_{args.timeout}s'
        (logs / f'roi_{i:02d}.log').write_text(trace)
        if problem is not None:
            stages = [s.removeprefix('STAGE=') for s in trace.splitlines() if s.startswith('STAGE=')]
            info = {'roi_index': i, 'bbox_xyxy': r['bbox_xyxy'],
                    'error': problem, 'last_stage': stages[-1] if stages else None}
            failures.append(info)
            print(f" ROI {i:02d}: FAILED {problem} at {info['last_stage']}", flush=True)
        else:
            report = json.loads(result_file.read_text())
            results.append(report)
            print(f" ROI {i:02d}: matches={report['n_matches']:3d}, "
                  f"inliers={report['n_inliers']:3d}, status={report['status']}", flush=True)
        # Keep partial progress if a subprocess or the parent is interrupted.
        (args.out / 'progress.json').write_text(json.dumps({'results': results,
                                                           'failures': failures}, indent=2))
    ranked = sorted(results, key=lambda r:(r['status'] == 'tentative',
                                          r.get('ranking_score', 0)), reverse=True)
    best = next((r for r in ranked if r['status'] == 'tentative'), None)
    if best and best['support_bbox_xyxy']:
        x0,y0,x1,y1 = best['support_bbox_xyxy']
        cv2.rectangle(overlay,(x0,y0),(x1,y1),(0,255,0),3)
        cv2.putText(overlay, f"TENTATIVE {best['n_inliers']} inliers",
                    (max(x0,0),max(28,y0-8)), cv2.FONT_HERSHEY_SIMPLEX,
                    .65,(0,255,0),2,cv2.LINE_AA)
    if failures:
        cv2.putText(overlay,f'PARTIAL SCAN: {len(failures)} failed ROIs',(10,32),
                    cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,255),2,cv2.LINE_AA)
    cv2.imwrite(str(args.out / 'localisation.png'),overlay)
    result = {'query': str(args.query), 'reference': str(args.reference),
              'opencv_version': cv2.__version__, 'region_count': len(regions),
              'proposal_seconds': round(proposal_seconds,4),
              'total_seconds': round(time.perf_counter()-start,3),
              'completed_count': len(results), 'failed_count': len(failures),
              'decision': ('PARTIAL_SCAN_NOT_VERIFIED' if failures
                           else 'TENTATIVE_UNCALIBRATED' if best
                           else 'ABSTAIN_NO_TENTATIVE_CANDIDATE'),
              'best':best, 'results':results, 'failures':failures,
              'limitations': [
                  'Scene-specific tabletop/edge heuristic, not generic detection',
                  'Only candidates proposed were checked: missing ROI means false negative',
                  'Reference/inlier support bbox is NOT a safe grasp pose',
                  'No calibrated false-positive rate; require target-absent tests',
                  'Each ROI runs a crash-isolated worker; startup overhead is substantial',
                  'Native LightGlue crashes are only isolated, not fixed']}
    (args.out / 'localisation.json').write_text(json.dumps(result, indent=2))
    print(f"DONE {len(results)}/{len(regions)} ROIs, "
          f"elapsed {result['total_seconds']}s, status={result['decision']}")
    if best:
        print(f"Best tentative ROI {best['roi_index']} {best['n_inliers']} inliers; "
              f"see {args.out}/localisation.png")
    return 2 if failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--query', type=Path)
    ap.add_argument('--reference', type=Path)
    ap.add_argument('--aliked', type=Path)
    ap.add_argument('--lightglue', type=Path)
    ap.add_argument('--out', type=Path, default=Path('outputs/perception_v0_5/fast_localise'))
    ap.add_argument('--max-rois', type=int, default=8)
    ap.add_argument('--working-width', type=int, default=640)
    ap.add_argument('--timeout', type=int, default=30)
    ap.add_argument('--min-inliers', type=int, default=12)
    ap.add_argument('--proposals-only', action='store_true')
    ap.add_argument('--_worker-index', type=int, default=-1, help=argparse.SUPPRESS)
    ap.add_argument('--_worker-result', type=Path, help=argparse.SUPPRESS)
    ap.add_argument('--_crop', type=Path, help=argparse.SUPPRESS)
    ap.add_argument('--_roi-xywh', nargs=4, type=int, help=argparse.SUPPRESS)
    a=ap.parse_args()
    if a._worker_index >= 0:
        if a._worker_result is None or a._crop is None or a._roi_xywh is None:
            ap.error('Internal worker options missing')
        sys.exit(worker(a))
    if a.query is None:
        ap.error('--query required')
    if a.max_rois < 1 or a.working_width < 128 or a.timeout < 1 or a.min_inliers < 4:
        ap.error('--max-rois >=1 --working-width >=128 --timeout >=1 --min-inliers >=4')
    sys.exit(parent(a))


if __name__ == '__main__':
    main()
