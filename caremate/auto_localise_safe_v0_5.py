"""CareMate v0.5: crash-isolated RGB-only tiled ALIKED/LightGlue diagnostic.

Runs one image tile per SUBPROCESS because OpenCV native SIGFPE/SIGSEGV cannot
be caught by a Python try/except. Every tile gets its own ALIKED + LightGlue
instance. Failed tiles are logged; completed tiles are saved incrementally.

Requires existing caremate/auto_localise_v0_5.py and perception_probe_v0_5.py.
Does NOT use MuJoCo geometry, segmentation, labels or ground-truth positions.
Results are *uncalibrated evidence*, not verified identities or grasp poses.

Example (from ~/Desktop/caremate):
  python -m caremate.auto_localise_safe_v0_5 \\
    --query outputs/sweep_20261018_085321/rgb/v001_t0.000_p+0.00_k-0.60.png \\
    --reference references/v0_5/reading_book.png \\
    --aliked models/vision/aliked.onnx \\
    --lightglue models/vision/lightglue_aliked.onnx \\
    --out outputs/perception_v0_5/auto_safe_full

This is slower than in-process matching. It is intended to locate the crash,
not as a real-time active-perception backend.
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

import cv2
import numpy as np

from caremate.auto_localise_v0_5 import evaluate, tile_grid
from caremate.perception_probe_v0_5 import FeaturePipeline, load_bgr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--query', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--aliked', type=Path, required=True)
    p.add_argument('--lightglue', type=Path, required=True)
    p.add_argument('--tile', type=int, default=320)
    p.add_argument('--stride', type=int, default=160)
    p.add_argument('--min-inliers', type=int, default=12)
    p.add_argument('--timeout', type=float, default=90,
                   help='Maximum seconds per tile subprocess (default: 90)')
    p.add_argument('--max-tiles', type=int, default=0,
                   help='Optional diagnostic cap; 0 scans all 35 default tiles')
    p.add_argument('--out', type=Path, default=Path('outputs/perception_v0_5/auto_safe_full'))
    # Worker-only internal options, do not use manually.
    p.add_argument('--_worker-index', type=int, default=-1, help=argparse.SUPPRESS)
    p.add_argument('--_worker-result', type=Path, help=argparse.SUPPRESS)
    return p.parse_args()


def check_args(args: argparse.Namespace) -> None:
    for name in ('query', 'reference', 'aliked', 'lightglue'):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f'{name}: {path}')
    if args.tile < 64 or args.stride < 1 or args.min_inliers < 4 or args.timeout <= 0:
        raise ValueError('Require tile >=64, stride >=1, min-inliers >=4 and timeout >0')


def stage(name: str) -> None:
    print(f'STAGE={name}', flush=True)


def worker(args: argparse.Namespace) -> int:
    # On Linux avoid writing massive core dump files when native OpenCV crashes.
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        pass
    stage('read_images')
    frame, ref = load_bgr(args.query), load_bgr(args.reference)
    h, w = frame.shape[:2]
    grid = list(tile_grid(w, h, args.tile, args.stride))
    idx = args._worker_index
    if not 0 <= idx < len(grid):
        raise ValueError(f'Worker index {idx} outside 0..{len(grid)-1}')
    x, y, tw, th = grid[idx]
    tile = frame[y:y+th, x:x+tw]
    stage('create_models')
    pipe = FeaturePipeline('aliked', args.aliked, args.lightglue)
    stage('extract_reference')
    kr, dr = pipe.features(ref)
    stage('extract_tile')
    kq, dq = pipe.features(tile)
    matches = []
    if dr is not None and dq is not None and len(kr) >= 4 and len(kq) >= 4:
        if (not np.isfinite(dr).all() or not np.isfinite(dq).all()
                or not np.isfinite(np.asarray([k.pt for k in kr])).all()
                or not np.isfinite(np.asarray([k.pt for k in kq])).all()):
            stage('nonfinite_features')
        else:
            pr = np.asarray([k.pt for k in kr], dtype=np.float32)
            pq = np.asarray([k.pt for k in kq], dtype=np.float32)
            stage('set_pair_info')
            pipe.matcher.setPairInfo(pr, pq, (ref.shape[1], ref.shape[0]),
                                     (tile.shape[1], tile.shape[0]))
            try:
                stage('lightglue_match')
                raw = pipe.matcher.match(dr, dq)
                matches = [m for m in raw if 0 <= m.queryIdx < len(kr)
                           and 0 <= m.trainIdx < len(kq)]
            finally:
                stage('clear_pair_info')
                pipe.matcher.clearPairInfo()
    stage('evaluate_geometry')
    report = evaluate(kr, kq, matches, (tw, th), (x, y),
                      (ref.shape[1], ref.shape[0]), args.min_inliers)
    report.update(tile_index=idx, tile_xywh=[x, y, tw, th],
                  n_ref_features=len(kr), n_tile_features=len(kq))
    if args._worker_result is None:
        raise ValueError('Internal worker result path missing')
    args._worker_result.parent.mkdir(parents=True, exist_ok=True)
    args._worker_result.write_text(json.dumps(report, indent=2))
    stage('finished')
    return 0


def status_for_returncode(returncode: int) -> str:
    if returncode < 0:
        try:
            return f'native_signal_{signal.Signals(-returncode).name}'
        except ValueError:
            return f'native_signal_{-returncode}'
    return f'worker_exit_{returncode}'


def save_partial(args: argparse.Namespace, total: int, reports: list[dict], failures: list[dict]) -> None:
    partial = {
        'query': str(args.query), 'reference': str(args.reference),
        'total_grid_tiles': total,
        'processed_ok': len(reports), 'failed_tiles': failures,
        'successful_tiles': reports,
        'warning': 'PARTIAL UNCALIBRATED EVIDENCE; not suitable for autonomous grasp authorisation'
    }
    (args.out / 'progress.json').write_text(json.dumps(partial, indent=2))


def parent(args: argparse.Namespace) -> int:
    frame = load_bgr(args.query)
    h, w = frame.shape[:2]
    grid = list(tile_grid(w, h, args.tile, args.stride))
    run_grid = grid[:args.max_tiles] if args.max_tiles else grid
    args.out.mkdir(parents=True, exist_ok=True)
    logs = args.out / 'tile_logs'
    logs.mkdir(exist_ok=True)
    print(f'Crash-isolated RGB scan: {w}x{h}, {len(run_grid)}/{len(grid)} tiles', flush=True)
    print('Each tile uses a fresh OpenCV process; this is slower but preserves diagnostics.', flush=True)
    reports: list[dict] = []
    failures: list[dict] = []
    t0 = time.perf_counter()
    for idx in range(len(run_grid)):
        result_path = logs / f'tile_{idx:03d}.json'
        if result_path.exists():
            result_path.unlink()
        cmd = [sys.executable, '-u', '-m', 'caremate.auto_localise_safe_v0_5',
               '--query', str(args.query), '--reference', str(args.reference),
               '--aliked', str(args.aliked), '--lightglue', str(args.lightglue),
               '--tile', str(args.tile), '--stride', str(args.stride),
               '--min-inliers', str(args.min_inliers),
               '--_worker-index', str(idx), '--_worker-result', str(result_path)]
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENCV_FOR_THREADS_NUM='1', PYTHONFAULTHANDLER='1')
        try:
            run = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                 timeout=args.timeout, check=False)
            trace = run.stdout + '\n' + run.stderr
            rc = run.returncode
            reason = None if rc == 0 and result_path.exists() else status_for_returncode(rc)
            if rc == 0 and not result_path.exists():
                reason = 'missing_worker_json'
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode(errors='replace') if isinstance(exc.stdout, bytes) else (exc.stdout or '')
            stderr = exc.stderr.decode(errors='replace') if isinstance(exc.stderr, bytes) else (exc.stderr or '')
            trace = stdout + '\n' + stderr
            reason = f'timeout_{args.timeout:g}s'
        (logs / f'tile_{idx:03d}.log').write_text(trace)
        if reason is None:
            report = json.loads(result_path.read_text())
            reports.append(report)
            print(f"  {idx+1:2d}/{len(run_grid)} tile={idx:02d} matches={report['n_matches']:3d} "
                  f"inliers={report['n_inliers']:3d} status={report['status']}", flush=True)
        else:
            markers = [line for line in trace.splitlines() if line.startswith('STAGE=')]
            failed_at = markers[-1].removeprefix('STAGE=') if markers else 'unknown'
            fail = {'tile_index': idx, 'tile_xywh': list(grid[idx]),
                    'failure': reason, 'last_stage': failed_at,
                    'log': str(logs / f'tile_{idx:03d}.log')}
            failures.append(fail)
            print(f'  {idx+1:2d}/{len(run_grid)} tile={idx:02d} FAILED {reason} '
                  f'at {failed_at} (continuing)', flush=True)
        save_partial(args, len(grid), reports, failures)

    complete = len(reports) == len(grid) and not failures
    ranked = sorted(reports, key=lambda r: (r['status'] == 'tentative',
                                            r.get('ranking_score', 0)), reverse=True)
    best = next((r for r in ranked if r['status'] == 'tentative'), None)
    overlay = frame.copy()
    if best is not None:
        bb = best['support_bbox_xyxy']
        cv2.rectangle(overlay, tuple(bb[:2]), tuple(bb[2:]), (0, 255, 0), 3)
        bx, by, bw, bh = best['tile_xywh']
        cv2.rectangle(overlay, (bx, by), (bx+bw, by+bh), (0, 190, 255), 2)
        cv2.putText(overlay, f"TENTATIVE ({best['n_inliers']} inliers)",
                    (max(5,bb[0]), max(30,bb[1]-10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2, cv2.LINE_AA)
    if not complete:
        cv2.putText(overlay, f'PARTIAL SCAN: {len(failures)} failures', (16,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
    elif best is None:
        cv2.putText(overlay, 'NO TENTATIVE CANDIDATE', (16,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
    cv2.imwrite(str(args.out / 'auto_candidates.png'), overlay)
    decision = ('PARTIAL_SCAN_NOT_VERIFIED' if not complete
                else 'TENTATIVE_VISUAL_EVIDENCE_ONLY' if best
                else 'ABSTAIN_NO_TENTATIVE_CANDIDATE')
    summary = {
        'query': str(args.query), 'reference': str(args.reference),
        'opencv_version': cv2.__version__,
        'n_tiles_total': len(grid), 'n_tiles_attempted': len(run_grid),
        'n_tiles_succeeded': len(reports), 'n_tiles_failed': len(failures),
        'scan_complete': complete, 'failure_details': failures,
        'elapsed_seconds': round(time.perf_counter()-t0, 2),
        'decision': decision, 'best': best, 'top5': ranked[:5],
        'all_tiles': reports,
        'limitations': [
            'Native crashes are isolated, NOT fixed; inspect failed tile logs',
            'Unprocessed / failed tiles may contain a better candidate',
            'Support bbox is not a complete object silhouette or safe grasp pose',
            'Feature evidence not calibrated against wrong-object / target-absent negatives',
            'Subprocess-per-tile approach is diagnostic and inefficient for production'
        ]
    }
    (args.out / 'auto_candidates.json').write_text(json.dumps(summary, indent=2))
    print(f"Done: {len(reports)}/{len(grid)} tiles successful, {len(failures)} crashed/timed out", flush=True)
    if best:
        print(f"Best TENTATIVE: tile={best['tile_index']} "
              f"inliers={best['n_inliers']} bbox={best['support_bbox_xyxy']}", flush=True)
    print(f"Decision: {decision}; results in {args.out}", flush=True)
    return 0 if complete else 2


def main() -> None:
    args = parse_args()
    check_args(args)
    if args._worker_index >= 0:
        sys.exit(worker(args))
    sys.exit(parent(args))


if __name__ == '__main__':
    main()
