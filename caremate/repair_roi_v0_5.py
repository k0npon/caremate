"""CareMate v0.5: expand tiny contour proposals into context-preserving search windows.

Diagnostic / experimental preprocessing only; not an object detector or a fix for
LightGlue SIGFPE. Uses the SAME RGB-only proposals produced by fast_localise_v0_5
and makes larger windows that can contain an entire object rather than a label
fragment. No simulator ground truth. Use with fast_match_bf_v0_5 first.

Example (run from ~/Desktop/caremate):
  python -m caremate.repair_roi_v0_5 \\
    --query outputs/sweep_20261018_085321/rgb/v001_t0.000_p+0.00_k-0.60.png \\
    --proposals outputs/perception_v0_5/fast_verified/proposals.json \\
    --out outputs/perception_v0_5/roi_repaired
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import cv2


def overlap_iou(a, b):
    dx = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    dy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = dx * dy
    a_area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    b_area = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / max(1, a_area + b_area - intersection)


def centered_window(bbox, image_width, image_height, window_width, window_height):
    """Enlarge while retaining window dimensions near the frame boundaries."""
    window_width = min(window_width, image_width)
    window_height = min(window_height, image_height)
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = max(0, min(round(cx - window_width / 2), image_width - window_width))
    top = max(0, min(round(cy - window_height / 2), image_height - window_height))
    return [int(left), int(top), int(left + window_width), int(top + window_height)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--query', type=Path, required=True)
    ap.add_argument('--proposals', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('outputs/perception_v0_5/roi_repaired'))
    ap.add_argument('--window', type=int, default=320, help='Minimum context window width and height in original pixels')
    ap.add_argument('--max-rois', type=int, default=6)
    ap.add_argument('--nms-iou', type=float, default=0.65)
    args = ap.parse_args()
    if args.window < 64 or args.max_rois < 1 or not 0 < args.nms_iou <= 1:
        ap.error('Require --window >=64, --max-rois >=1, --nms-iou in (0,1]')
    frame = cv2.imread(str(args.query))
    if frame is None:
        ap.error(f'Cannot read query: {args.query}')
    h, w = frame.shape[:2]
    source = json.loads(args.proposals.read_text())
    if source.get('image_wh') != [w, h]:
        ap.error(f'Input proposals built for {source.get("image_wh")}; query is {[w, h]}')
    start = time.perf_counter()
    result = []
    for i, region in enumerate(source['regions']):
        old = region.get('bbox_xyxy')
        if not old or len(old) != 4:
            continue
        candidate = centered_window(old, w, h, args.window, args.window)
        if any(overlap_iou(candidate, r['bbox_xyxy']) >= args.nms_iou for r in result):
            continue
        result.append({'bbox_xyxy': candidate,
                       'proposal_score': region.get('proposal_score', 0),
                       'source_roi_index': i,
                       'source_bbox_xyxy': old})
        if len(result) >= args.max_rois:
            break
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'crops').mkdir(exist_ok=True)
    vis = frame.copy()
    for i, region in enumerate(result):
        x0, y0, x1, y1 = region['bbox_xyxy']
        cv2.rectangle(vis, (x0, y0), (x1-1, y1-1), (0, 220, 255), 2)
        cv2.putText(vis, f'ROI{i} / orig{region["source_roi_index"]}',
                    (x0 + 3, max(y0 + 23, 23)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, f'ROI{i} / orig{region["source_roi_index"]}',
                    (x0 + 3, max(y0 + 23, 23)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(args.out / 'crops' / f'roi_{i:02}.png'), frame[y0:y1,x0:x1])
    cv2.imwrite(str(args.out / 'proposals.png'), vis)
    payload = {'query': str(args.query), 'image_wh': [w, h], 'regions': result,
               'repair_seconds': round(time.perf_counter()-start, 4),
               'source_proposals': str(args.proposals), 'num_original': len(source['regions']),
               'num_repaired': len(result),
               'warning': 'Heuristic ROI enlargement only; visual confirmation and negative tests required. Does not fix LightGlue native crashes.'}
    (args.out / 'proposals.json').write_text(json.dumps(payload, indent=2))
    print(f"Repaired {payload['num_original']} tiny contour ROIs into {payload['num_repaired']} context windows.")
    for i, r in enumerate(result):
        print(f" ROI {i:02}: {r['bbox_xyxy']} (from {r['source_roi_index']})")
    print(f"See {args.out}/proposals.png")


if __name__ == '__main__':
    main()
