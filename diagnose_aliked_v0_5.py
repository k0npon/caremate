"""Non-invasive diagnostics for CareMate OpenCV 5 ALIKED + LightGlue.

Run from the project root (with the caremate mamba environment active):

python diagnose_aliked_v0_5.py \
  --reference references/v0_5/reading_book.png \
  --frame outputs/sweep_20261018_085321/rgb/v000_t0.000_p+0.00_k-0.40.png \
  --aliked models/vision/aliked.onnx \
  --lightglue models/vision/lightglue_aliked.onnx

Writes a JSON report and two keypoint visualisations. Does not modify
CareMate's reference gallery, scene, inference code, or input images.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def extract(detector, img):
    keypoints, desc = detector.detectAndCompute(img, None)
    keypoints = list(keypoints) if keypoints is not None else []
    return keypoints, desc


def desc_details(desc):
    if desc is None:
        return {"shape": None, "dtype": None, "finite": None}
    return {"shape": list(desc.shape), "dtype": str(desc.dtype),
            "finite": bool(np.isfinite(desc).all())}


def correspondences(matcher, x, y, kx, ky, dx, dy):
    if dx is None or dy is None or len(kx) < 4 or len(ky) < 4:
        return []
    pts0 = np.array([k.pt for k in kx], np.float32)
    pts1 = np.array([k.pt for k in ky], np.float32)
    matcher.setPairInfo(pts0, pts1, (x.shape[1], x.shape[0]),
                        (y.shape[1], y.shape[0]))
    try:
        return list(matcher.match(dx, dy))
    finally:
        matcher.clearPairInfo()


def bf_l2_matches(dx, dy):
    if dx is None or dy is None or len(dx) < 2 or len(dy) < 2:
        return 0
    # Diagnostic ONLY: matching ALIKED descriptors with classical L2 + ratio.
    # This is not a replacement for native OpenCV 5 LightGlue.
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = bf.knnMatch(np.asarray(dx, np.float32),
                        np.asarray(dy, np.float32), k=2)
    return sum(1 for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--frame', type=Path, required=True)
    p.add_argument('--aliked', type=Path, required=True)
    p.add_argument('--lightglue', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('outputs/perception_v0_5/aliked_diagnostics'))
    args = p.parse_args()
    for name in ('reference', 'frame', 'aliked', 'lightglue'):
        if not getattr(args, name).is_file():
            p.error(f'File missing: {getattr(args, name)}')

    ref = cv2.imread(str(args.reference))
    frame = cv2.imread(str(args.frame))
    if ref is None or frame is None:
        p.error('Failed to decode an input PNG')
    if not hasattr(cv2, 'ALIKED_create') or not hasattr(cv2, 'LightGlueMatcher_create'):
        p.error(f'OpenCV {cv2.__version__} lacks the OpenCV 5 learned-feature APIs')

    args.out.mkdir(parents=True, exist_ok=True)
    aliked = cv2.ALIKED_create(str(args.aliked))
    lightglue = cv2.LightGlueMatcher_create(str(args.lightglue))
    kr, dr = extract(aliked, ref)
    kf, df = extract(aliked, frame)

    images = {'ref': ref, 'frame': frame}
    kps = {'ref': kr, 'frame': kf}
    ds = {'ref': dr, 'frame': df}
    report = {'opencv_version': cv2.__version__,
              'reference': str(args.reference), 'frame': str(args.frame),
              'image_shapes_hwc': {k: list(im.shape) for k, im in images.items()},
              'features': {name: {'n_keypoints': len(kps[name]),
                                  'descriptors': desc_details(ds[name])}
                           for name in images},
              'tests': {}}

    for name, img in images.items():
        marked = cv2.drawKeypoints(img, kps[name], None,
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(str(args.out / f'{name}_keypoints.png'), marked)

    for label, a, b in [('reference_self', 'ref', 'ref'),
                        ('frame_self', 'frame', 'frame'),
                        ('reference_to_frame', 'ref', 'frame')]:
        try:
            matches = correspondences(lightglue, images[a], images[b],
                                      kps[a], kps[b], ds[a], ds[b])
            lg_count = len(matches)
            error = None
        except cv2.error as exc:
            lg_count = None
            matches = []
            error = str(exc)
        bf_count = bf_l2_matches(ds[a], ds[b])
        report['tests'][label] = {'lightglue_matches': lg_count,
                                  'aliked_bf_l2_ratio_matches': bf_count,
                                  'lightglue_error': error}
        print(f'{label:22s} ALIKED features {len(kps[a]):4d}/{len(kps[b]):4d}  '
              f'LightGlue matches={lg_count}  ALIKED+BF(L2) matches={bf_count}')

    dest = args.out / 'diagnostics.json'
    dest.write_text(json.dumps(report, indent=2))
    print(f'\nSaved {dest}')
    print('Also saved ref_keypoints.png and frame_keypoints.png.')
    print('Interpretation: self-match failure suggests feature/model/matcher issues;')
    print('good self-match but weak crop-to-frame suggests scale/context issues.')


if __name__ == '__main__':
    main()
