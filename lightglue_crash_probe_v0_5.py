"""CareMate diagnostic: isolate OpenCV 5 LightGlue SIGFPE with fixed saved ROIs.

Only evaluates ALIKED extraction -> LightGlue matching, without USAC or region
proposals. Each experiment runs in its own subprocess because SIGFPE is native.

Usage from ~/Desktop/caremate:
    python lightglue_crash_probe_v0_5.py \
      --reference references/v0_5/reading_book.png \
      --good outputs/perception_v0_5/fast_verified/crops/roi_00.png \
      --bad outputs/perception_v0_5/fast_verified/crops/roi_01.png \
      --aliked models/vision/aliked.onnx \
      --lightglue models/vision/lightglue_aliked.onnx

Logs in outputs/perception_v0_5/lightglue_crash_probe/. This does not alter any
project files. keypoint limits are diagnostic interventions, not production fixes.
"""
from __future__ import annotations
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def worker(args):
    import cv2
    import numpy as np
    cv2.setUseOptimized(True)
    if args.threads == 1:
        cv2.setNumThreads(1)
    print('stage=read', flush=True)
    ref = cv2.imread(str(args.reference), cv2.IMREAD_COLOR)
    roi = cv2.imread(str(args.crop), cv2.IMREAD_COLOR)
    if ref is None or roi is None:
        raise FileNotFoundError('Missing reference or ROI image')
    print('stage=load_models', flush=True)
    extractor = cv2.ALIKED_create(str(args.aliked))
    matcher = cv2.LightGlueMatcher_create(str(args.lightglue))

    def extract(label, im):
        print('stage=extract_' + label, flush=True)
        kps, desc = extractor.detectAndCompute(im, None)
        kps = list(kps or [])
        if desc is None:
            raise RuntimeError(label + ': no descriptors')
        desc = np.asarray(desc, dtype=np.float32)
        # Preserve keypoint-descriptor alignment when testing an input subset.
        kps, desc = kps[:args.limit], np.ascontiguousarray(desc[:args.limit])
        pts = np.ascontiguousarray([kp.pt for kp in kps], dtype=np.float32)
        responses = np.array([kp.response for kp in kps], dtype=np.float32)
        norms = np.linalg.norm(desc, axis=1)
        stats = dict(image_wh=[im.shape[1], im.shape[0]],
                     kp_count=len(kps), desc_shape=list(desc.shape),
                     finite_pts=bool(np.isfinite(pts).all()),
                     finite_desc=bool(np.isfinite(desc).all()),
                     pts_unique_rounded_01=len(np.unique(np.round(pts, 1), axis=0)),
                     pts_min=pts.min(axis=0).tolist(), pts_max=pts.max(axis=0).tolist(),
                     desc_norm_min=float(norms.min()), desc_norm_max=float(norms.max()),
                     responses_finite=bool(np.isfinite(responses).all()),
                     response_min=float(responses.min()), response_max=float(responses.max()))
        print('STATS_' + label + '=' + json.dumps(stats, sort_keys=True), flush=True)
        if len(kps) < 4 or not stats['finite_pts'] or not stats['finite_desc']:
            raise ValueError('Invalid ' + label + ' features')
        return pts, desc

    ref_pts, ref_desc = extract('reference', ref)
    roi_pts, roi_desc = extract('roi', roi)
    print('stage=set_pair_info', flush=True)
    matcher.setPairInfo(ref_pts, roi_pts, (ref.shape[1], ref.shape[0]),
                        (roi.shape[1], roi.shape[0]))
    print('stage=lightglue_match', flush=True)
    matches = matcher.match(ref_desc, roi_desc)  # May cause SIGFPE, by design.
    print('stage=match_completed count=' + str(len(matches)), flush=True)
    matcher.clearPairInfo()
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--good', type=Path)
    p.add_argument('--bad', type=Path)
    p.add_argument('--crop', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--aliked', type=Path, required=True)
    p.add_argument('--lightglue', type=Path, required=True)
    p.add_argument('--limit', type=int, default=1000, help=argparse.SUPPRESS)
    p.add_argument('--threads', type=int, choices=(0, 1), default=0,
                   help='0 leave OpenCV default; 1 force OpenCV single-threaded')
    p.add_argument('--out', type=Path,
                   default=Path('outputs/perception_v0_5/lightglue_crash_probe'))
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    for f in (args.reference, args.aliked, args.lightglue):
        if not f.is_file():
            p.error('Missing: ' + str(f))
    if args.worker:
        if args.crop is None or not args.crop.is_file():
            p.error('--worker requires a valid --crop')
        return worker(args)
    for f in (args.good, args.bad):
        if f is None or not f.is_file():
            p.error('Provide --good and --bad existing crop images')
    args.out.mkdir(parents=True, exist_ok=True)
    out = []
    for label, crop in [('good', args.good), ('bad', args.bad)]:
        for limit in (1000, 512):
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()),
                   '--worker', '--reference', str(args.reference),
                   '--crop', str(crop), '--aliked', str(args.aliked),
                   '--lightglue', str(args.lightglue), '--limit', str(limit),
                   '--threads', str(args.threads)]
            env = dict(os.environ)
            env.update(PYTHONFAULTHANDLER='1', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            start = time.perf_counter()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      env=env, timeout=45)
                code = proc.returncode
                status = ('ok' if code == 0 else
                          signal.Signals(-code).name if code < 0 else f'exit_{code}')
                log = proc.stdout + '\n' + proc.stderr
            except subprocess.TimeoutExpired as ex:
                status, code = 'timeout', None
                log = str(ex)
            elapsed = time.perf_counter() - start
            logfile = args.out / f'{label}_top{limit}.log'
            logfile.write_text(log)
            last_stage = next((ln for ln in reversed(log.splitlines())
                               if ln.startswith('stage=')), 'stage=unknown')
            item = dict(roi=label, crop=str(crop), max_keypoints=limit,
                        status=status, returncode=code, last_stage=last_stage,
                        seconds=round(elapsed, 3), log=str(logfile))
            out.append(item)
            print(f'{label:4s} top{limit}: {status:9s} {last_stage} '
                  f'{elapsed:.2f}s | {logfile}', flush=True)
    dest = args.out / 'summary.json'
    dest.write_text(json.dumps(out, indent=2))
    print('Saved:', dest)


if __name__ == '__main__':
    raise SystemExit(main())
