"""CareMate v0.5 / Step 1: RGB-only reference-gallery verification probe.

Goals:
  - Run ORB + RANSAC (USAC_MAGSAC when available) immediately.
  - Optionally test *OpenCV 5's* ALIKED + LightGlue with matching ONNX files.
  - Optionally run *YOLO26-Depth* (not OpenCV 5) on the same RGB query.

Does NOT read MuJoCo state, object IDs or segmentation. All decisions use RGB.
Results are preliminary evidence, NOT calibrated object-recognition confidence,
3D object localisation, safe grasp approval or verified agentic autonomy.

Make a manually cropped reference (requires graphical desktop):
  python -m caremate.perception_probe_v0_5 --add-ref --image /path/to/rgb.png --name reading_book

Match one frame against all saved references (no neural weights needed):
  python -m caremate.perception_probe_v0_5 --query /path/to/another_rgb.png --method orb

Try OpenCV 5 learned features after obtaining compatible standalone ONNX files:
  python -m caremate.perception_probe_v0_5 --query ... --method aliked \\
    --aliked-model models/vision/aliked.onnx --lightglue-model models/vision/lightglue_aliked.onnx

Optionally add --depth to either matching command. It requires Ultralytics,
loads yolo26n-depth.pt on first use, and reports *estimated* distance.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np


def load_bgr(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode image: {path}")
    return img


def add_reference(path: Path, name: str, refs: Path) -> None:
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", name):
        raise ValueError("--name must contain letters, digits, '_' or '-' only")
    img = load_bgr(path)
    print("Select a TIGHT box around ONE textured object, then Enter/Space to confirm.")
    print("Press C or Esc to cancel. Avoid the table/background in the crop.")
    rect = cv2.selectROI("CareMate / Select reference object", img, False, False)
    cv2.destroyAllWindows()
    x, y, w, h = (int(v) for v in rect)
    if w < 12 or h < 12:
        raise RuntimeError("No usable crop selected")
    refs.mkdir(parents=True, exist_ok=True)
    dest = refs / f"{name}.png"
    if not cv2.imwrite(str(dest), img[y:y+h, x:x+w]):
        raise OSError(f"Cannot write {dest}")
    print(f"Saved reference: {dest} ({w}x{h})")
    print("Use a different viewpoint as query. Same-frame matching alone proves very little.")


class FeaturePipeline:
    def __init__(self, method: str, aliked_model: Path | None, lightglue_model: Path | None):
        self.method = method
        if method == "orb":
            self.extractor = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            if not cv2.__version__.startswith("5."):
                raise RuntimeError(f"ALIKED requires OpenCV 5, found {cv2.__version__}")
            if not (hasattr(cv2, "ALIKED_create") and hasattr(cv2, "LightGlueMatcher_create")):
                raise RuntimeError("This OpenCV wheel has no ALIKED/LightGlue APIs")
            if aliked_model is None or lightglue_model is None:
                raise RuntimeError("--method aliked requires --aliked-model and --lightglue-model")
            if not (aliked_model.is_file() and lightglue_model.is_file()):
                raise FileNotFoundError("One or both ONNX model paths do not exist")
            # ALIKED and LightGlue ONNX weights must be compatible; don't pair
            # ALIKED with models trained for SIFT, DISK or SuperPoint.
            self.extractor = cv2.ALIKED_create(str(aliked_model))
            self.matcher = cv2.LightGlueMatcher_create(str(lightglue_model))

    def features(self, bgr: np.ndarray):
        # cv2.ALIKED accepts BGR and internally converts to RGB (official docs).
        keypoints, descriptors = self.extractor.detectAndCompute(bgr, None)
        return keypoints or [], descriptors

    def correspondences(self, ref_img: np.ndarray, query_img: np.ndarray):
        kr, dr = self.features(ref_img)
        kq, dq = self.features(query_img)
        if dr is None or dq is None or len(kr) < 4 or len(kq) < 4:
            return kr, kq, [], "too_few_features"
        if self.method == "orb":
            pairs = self.matcher.knnMatch(dr, dq, k=2)
            matches = [a for pair in pairs if len(pair) == 2
                       for a, b in [pair] if a.distance < 0.75 * b.distance]
        else:
            ref_pts = np.asarray([k.pt for k in kr], dtype=np.float32)
            query_pts = np.asarray([k.pt for k in kq], dtype=np.float32)
            self.matcher.setPairInfo(ref_pts, query_pts,
                                     (ref_img.shape[1], ref_img.shape[0]),
                                     (query_img.shape[1], query_img.shape[0]))
            try:
                matches = list(self.matcher.match(dr, dq))
            finally:
                self.matcher.clearPairInfo()
        # Matches are indexed into [reference keypoints, query keypoints].
        matches = [m for m in matches if m.queryIdx < len(kr) and m.trainIdx < len(kq)]
        return kr, kq, matches, "ok"


def verify(ref_img: np.ndarray, query_img: np.ndarray, pipeline: FeaturePipeline) -> dict:
    kr, kq, matches, status = pipeline.correspondences(ref_img, query_img)
    result = {"n_ref_features": len(kr), "n_query_features": len(kq),
              "n_matches": len(matches), "n_inliers": 0, "inlier_ratio": 0.0,
              "polygon": None, "status": status}
    if len(matches) < 6:
        result["status"] = "too_few_matches"
        return result
    pts_ref = np.float32([kr[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_qry = np.float32([kq[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    try:
        H, inliers = cv2.findHomography(pts_ref, pts_qry, method, 3.5)
    except cv2.error:
        H, inliers = None, None
    if H is None or inliers is None or not np.isfinite(H).all():
        result["status"] = "homography_failed"
        return result
    n_in = int(inliers.ravel().sum())
    ratio = n_in / len(matches)
    h, w = ref_img.shape[:2]
    corners = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]).reshape(-1, 1, 2)
    polygon = cv2.perspectiveTransform(corners, H).reshape(4, 2)
    qh, qw = query_img.shape[:2]
    area = float(abs(cv2.contourArea(polygon.astype(np.float32)))) if np.isfinite(polygon).all() else 0.0
    plausible = (n_in >= 6 and ratio >= 0.35 and area >= 80 and
                 area <= 0.75 * qw * qh and
                 bool(cv2.isContourConvex(polygon.astype(np.float32))) and
                 np.isfinite(polygon).all())
    result.update(n_inliers=n_in, inlier_ratio=round(ratio, 4),
                  polygon=polygon.round(1).tolist() if plausible else None,
                  status="candidate" if plausible else "unreliable_geometry")
    return result


def depth_probe(img_path: Path, out: Path, target_polygon: list | None, device: str) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("Optional dependency missing: python -m pip install ultralytics") from e
    model = YOLO("yolo26n-depth.pt")
    results = model.predict(source=str(img_path), device=device, verbose=False)
    if not results or getattr(results[0], "depth", None) is None:
        raise RuntimeError("Model returned no depth map; verify current Ultralytics version")
    depth = results[0].depth.data.cpu().numpy()
    depth = np.squeeze(depth).astype(np.float32)
    img = load_bgr(img_path)
    if depth.ndim != 2:
        raise RuntimeError(f"Expected HxW depth, got shape {depth.shape}")
    # Resize for presentation and per-pixel alignment if backend returns a
    # different map size. Distances remain estimates, not measured geometry.
    if depth.shape != img.shape[:2]:
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    np.save(str(out / "depth_estimated_m.npy"), depth)
    valid = np.isfinite(depth) & (depth > 0)
    lo, hi = np.percentile(depth[valid], [5, 95]) if np.any(valid) else (0.0, 1.0)
    scaled = np.clip((depth - lo) / max(hi-lo, 1e-6), 0, 1)
    colour = cv2.applyColorMap((255*scaled).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(out / "depth_estimated_visualisation.png"), colour)
    mask = valid.copy()
    if target_polygon is not None:
        mask &= False
        pts = np.array(target_polygon, np.int32).reshape(-1, 1, 2)
        roi = np.zeros(depth.shape, np.uint8)
        cv2.fillPoly(roi, [pts], 1)
        mask = valid & (roi > 0)
    return {"model": "yolo26n-depth.pt", "device": device,
            "source": "predicted monocular depth, NOT measured depth",
            "median_depth_m": float(np.median(depth[mask])) if np.any(mask) else None,
            "pixels_sampled": int(mask.sum()),
            "depth_map": "depth_estimated_m.npy"}


def run(args) -> None:
    refs = sorted([p for p in args.refs.glob("*.png") if p.is_file()])
    if not refs:
        raise FileNotFoundError(f"No reference PNGs found in {args.refs}. Use --add-ref first.")
    img = load_bgr(args.query)
    pipeline = FeaturePipeline(args.method, args.aliked_model, args.lightglue_model)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    overlay = img.copy()
    reports = {}
    for ref_path in refs:
        ref = load_bgr(ref_path)
        try:
            report = verify(ref, img, pipeline)
        except cv2.error as exc:
            report = {"status": "opencv_error", "error": str(exc)}
        reports[ref_path.stem] = report
        print(f"{ref_path.stem:22s} matches={report.get('n_matches', 0):3d} "
              f"inliers={report.get('n_inliers', 0):3d}  {report['status']}")
        if report.get("polygon") is not None:
            poly = np.rint(np.asarray(report["polygon"])).astype(np.int32)
            cv2.polylines(overlay, [poly], True, (40, 230, 40), 2)
            u, v = np.mean(poly, axis=0).astype(int)
            cv2.putText(overlay, ref_path.stem, (int(u), int(v)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 230, 40), 2, cv2.LINE_AA)
    cv2.imwrite(str(out / "candidate_matches.png"), overlay)
    # Do NOT call the highest-inlier reference 'verified' without tuning on
    # held-out positive AND negative images, incl. wrong and missing objects.
    candidates = [(name, r) for name, r in reports.items() if r.get('status') == 'candidate']
    candidates.sort(key=lambda x: x[1]['n_inliers'], reverse=True)
    result = {"opencv_version": cv2.__version__, "query": str(args.query),
              "method": args.method, "references": reports,
              "ranking_by_inliers_only": [name for name, _ in candidates],
              "decision": "NOT_CALIBRATED_DO_NOT_AUTHORISE_RETRIEVAL"}
    if args.depth:
        print("Running optional YOLO26-Depth (first run may download model weights)...")
        primary_poly = candidates[0][1].get('polygon') if candidates else None
        result["depth"] = depth_probe(args.query, out, primary_poly, args.device)
    (out / "evidence.json").write_text(json.dumps(result, indent=2))
    print(f"Saved {out / 'evidence.json'} and {out / 'candidate_matches.png'}")
    print("NOTE: tentative matches only; test across distinct views and negatives.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--add-ref", action="store_true", help="Crop and save reference from --image")
    p.add_argument("--image", type=Path, help="Source image for --add-ref")
    p.add_argument("--name", help="Reference object name for --add-ref")
    p.add_argument("--refs", type=Path, default=Path("references/v0_5"))
    p.add_argument("--query", type=Path, help="RGB frame to verify")
    p.add_argument("--method", choices=("orb", "aliked"), default="orb")
    p.add_argument("--aliked-model", type=Path)
    p.add_argument("--lightglue-model", type=Path)
    p.add_argument("--depth", action="store_true", help="Optionally predict monocular metric depth using YOLO26")
    p.add_argument("--device", default="cpu", help="Depth model inference device: cpu or 0")
    p.add_argument("--out", type=Path, default=Path("outputs/perception_v0_5"))
    args = p.parse_args()
    if args.add_ref:
        if args.image is None or args.name is None:
            p.error("--add-ref requires --image and --name")
        add_reference(args.image, args.name, args.refs)
    else:
        if args.query is None:
            p.error("Matching requires --query")
        run(args)


if __name__ == "__main__":
    main()
