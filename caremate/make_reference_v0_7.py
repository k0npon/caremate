"""CareMate v0.7: render canonical, background-free reference images of GSO objects.

v0.5 used a hand-cropped, oblique crop of a scene render (184x236, ~30%
background, perspective-distorted, biased to one viewpoint).  This tool loads
the object's own GSO model.xml ALONE, renders its six axis-aligned faces with a
narrow-FOV (near-orthographic) camera on a flat grey background, and saves a
tight crop.  Reference creation happens offline; it is product-catalogue data,
not runtime ground truth.

Usage (from ~/Desktop/caremate):
  # 1. render all six faces + a contact sheet, pick the most textured face
  python -m caremate.make_reference_v0_7 --name reading_book \\
      --folder Eat_to_Live_The_Amazing_NutrientRich_Program_for_Fast_and_Sustained_Weight_Loss_Revised_Edition_Book
  # 2. save the textured faces as references/v0_7/reading_book__<face>.png
  python -m caremate.make_reference_v0_7 --name reading_book --folder ... --pick az90 az270

  Headless: prefix with MUJOCO_GL=egl (or osmesa).
"""
from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GSO = ROOT / "third_party" / "mujoco_scanned_objects" / "models"
BG = 0.5  # grey background (0..1)
VIEWS = {  # name: (azimuth, elevation) of MuJoCo free camera
    "az0": (0, 0), "az90": (90, 0), "az180": (180, 0), "az270": (270, 0),
    "top": (90, -89.9), "bottom": (90, 89.9),
}


def obj_bounds(path: Path):
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                v = np.array([float(x) for x in line.split()[1:4]])
                lo, hi = np.minimum(lo, v), np.maximum(hi, v)
    if not np.isfinite(lo).all():
        raise ValueError(f"No vertices in {path}")
    return lo, hi


def isolated_model(source: Path, size: int) -> mujoco.MjModel:
    root = ET.parse(source / "model.xml").getroot()
    for c in root.findall("compiler"):
        root.remove(c)
    ET.SubElement(root, "compiler", assetdir=str(source))
    asset = root.find("asset")
    # Collision hulls are irrelevant for appearance and slow to compile.
    for a in list(asset):
        if a.tag == "mesh" and a.get("name", "").startswith("model_collision"):
            asset.remove(a)
    for body in root.iter("body"):
        for g in list(body.findall("geom")):
            if g.get("mesh", "").startswith("model_collision"):
                body.remove(g)
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="flat",
                  rgb1=f"{BG} {BG} {BG}", rgb2=f"{BG} {BG} {BG}", width="8", height="8")
    vis = root.find("visual")
    if vis is None:
        vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth=str(size), offheight=str(size), fovy="12")
    ET.SubElement(vis, "headlight", ambient="0.55 0.55 0.55", diffuse="0.45 0.45 0.45",
                  specular="0.05 0.05 0.05")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def tight_crop(rgb: np.ndarray, margin: int = 4):
    diff = np.abs(rgb.astype(np.int16) - int(255 * BG)).max(axis=2)
    ys, xs = np.nonzero(diff > 12)
    if xs.size == 0:
        return None
    x0, x1 = max(0, xs.min() - margin), min(rgb.shape[1], xs.max() + 1 + margin)
    y0, y1 = max(0, ys.min() - margin), min(rgb.shape[0], ys.max() + 1 + margin)
    return rgb[y0:y1, x0:x1]


def texture_score(bgr: np.ndarray) -> float:
    """Gradient-energy proxy for 'how matchable is this face' (heuristic)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def render_faces(source: Path, size: int = 768) -> dict[str, np.ndarray]:
    model = isolated_model(source, size)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lo, hi = obj_bounds(source / "model.obj")
    extent = float(np.max(hi - lo))
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (lo + hi) / 2
    cam.distance = 0.62 * extent / math.tan(math.radians(model.vis.global_.fovy / 2))
    faces = {}
    with mujoco.Renderer(model, size, size) as r:
        for name, (az, el) in VIEWS.items():
            cam.azimuth, cam.elevation = az, el
            r.update_scene(data, camera=cam)
            crop = tight_crop(r.render())
            if crop is not None:
                faces[name] = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    return faces


def contact_sheet(faces: dict[str, np.ndarray], scores: dict[str, float], h: int = 260):
    tiles = []
    for name, im in faces.items():
        s = h / im.shape[0]
        t = cv2.resize(im, (max(1, int(im.shape[1] * s)), h))
        t = cv2.copyMakeBorder(t, 34, 6, 6, 6, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        cv2.putText(t, f"{name} tex={scores[name]:.0f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    return cv2.hconcat(tiles) if tiles else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="Reference name used by the agent, e.g. reading_book")
    p.add_argument("--folder", required=True, help="GSO model folder under third_party/.../models")
    p.add_argument("--gso", type=Path, default=GSO)
    p.add_argument("--pick", choices=list(VIEWS), nargs="+",
                   help="Save these faces as references <name>__<face>.png (an object may show any face)")
    p.add_argument("--max-side", type=int, default=480, help="Long side of the saved reference")
    p.add_argument("--out", type=Path, default=ROOT / "references" / "v0_7")
    a = p.parse_args()
    source = a.gso / a.folder
    if not (source / "model.xml").is_file():
        p.error(f"Missing {source}/model.xml (git sparse-checkout add models/{a.folder})")
    faces = render_faces(source)
    scores = {k: texture_score(v) for k, v in faces.items()}
    work = ROOT / "outputs" / "references_v0_7" / a.name
    work.mkdir(parents=True, exist_ok=True)
    for k, v in faces.items():
        cv2.imwrite(str(work / f"{k}.png"), v)
    sheet = contact_sheet(faces, scores)
    cv2.imwrite(str(work / "faces.png"), sheet)
    ranked = sorted(scores, key=scores.get, reverse=True)
    print("Faces by texture score:", ", ".join(f"{k}={scores[k]:.0f}" for k in ranked))
    print(f"Contact sheet: {work / 'faces.png'}")
    if a.pick:
        a.out.mkdir(parents=True, exist_ok=True)
        for face in a.pick:
            im = faces[face]
            s = a.max_side / max(im.shape[:2])
            if s < 1:
                im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
            dest = a.out / f"{a.name}__{face}.png"
            cv2.imwrite(str(dest), im)
            meta = {"name": a.name, "gso_folder": a.folder, "face": face,
                    "texture_score": scores[face], "size_wh": [im.shape[1], im.shape[0]],
                    "note": "Canonical offline reference render; never used as runtime ground truth."}
            (a.out / f"{a.name}__{face}.json").write_text(json.dumps(meta, indent=2))
            print(f"Saved reference {dest} ({im.shape[1]}x{im.shape[0]})")
    else:
        print(f"Inspect the sheet, then rerun with --pick {ranked[0]} (or another face).")


if __name__ == "__main__":
    main()
