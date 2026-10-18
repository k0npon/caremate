"""Build a non-destructive, photo-textured CareMate scene using GSO objects.

Usage:
    python -m caremate.build_scanned_scene
    python -m caremate.build_scanned_scene --collision-mode full

Expects user-downloaded GSO objects under third_party/mujoco_scanned_objects/models/.
The base CareMate and furnished scenes are never modified.
"""
from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from caremate.build_scene import ROOT, ROBOT_DIR
from caremate.build_furnished_scene import FURNISHED_XML

GSO = ROOT / "third_party" / "mujoco_scanned_objects" / "models"
SCANNED_XML = ROBOT_DIR / "caremate_scanned_scene.xml"
MESH_DIR = ROBOT_DIR / "assets" / "caremate_scanned"
TEXTURE_DIR = ROBOT_DIR / "caremate_scanned_textures"

# Only objects that have already been downloaded/tested by the user.
OBJECTS = [
    {
        "name": "reading_book",
        "folder": "Eat_to_Live_The_Amazing_NutrientRich_Program_for_Fast_and_Sustained_Weight_Loss_Revised_Edition_Book",
        "xy": (1.17, -0.57),
        "yaw": 0.0,
        "mass": 0.35,  # assumed kg for simulation, NOT measured
    },
    {
        "name": "personal_care_box",
        "folder": "Marc_Anthony_Strictly_Curls_Curl_Envy_Perfect_Curl_Cream_6_fl_oz_bottle",
        "xy": (1.62, -0.55),
        "yaw": 0.0,
        "mass": 0.18,  # assumed kg for simulation, NOT measured
    },
]


def obj_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """AABB of the source OBJ vertices, in the source mesh's coordinates (metres)."""
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                values = line.split()
                if len(values) < 4:
                    continue
                xyz = np.array([float(values[1]), float(values[2]), float(values[3])])
                low = np.minimum(low, xyz)
                high = np.maximum(high, xyz)
                count += 1
    if count == 0 or not np.all(np.isfinite(low)):
        raise ValueError(f"No valid vertices found in {path}")
    return low, high


def _fmt(values) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


def _copy_asset(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Missing GSO asset: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _import_object(root: ET.Element, world: ET.Element, cfg: dict,
                   *, collision_mode: str, table_top: float) -> None:
    name, source = cfg["name"], GSO / cfg["folder"]
    model_path = source / "model.xml"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Missing {model_path}. Run git sparse-checkout add models/{cfg['folder']} "
            "inside third_party/mujoco_scanned_objects first."
        )
    original = ET.parse(model_path).getroot()
    source_assets = original.find("asset")
    source_body = original.find("worldbody/body")
    if source_assets is None or source_body is None:
        raise ValueError(f"Unexpected GSO model structure: {model_path}")
    target_assets = root.find("asset")
    if target_assets is None:
        target_assets = ET.SubElement(root, "asset")

    bounds_low, bounds_high = obj_bounds(source / "model.obj")
    dimensions = bounds_high - bounds_low
    if any(dim < 0.001 or dim > 0.65 for dim in dimensions):
        raise ValueError(f"Suspicious {name} dimensions {dimensions} m; inspect orientation and scale.")

    prefix = f"gso_{name}"
    texture_names, material_names, mesh_names = {}, {}, {}
    # Imported assets have unique identifiers, and are copied into CareMate
    # rather than linked to the third-party source repository.
    for original_asset in source_assets:
        asset = copy.deepcopy(original_asset)
        old_name = asset.get("name")
        if not old_name or asset.tag not in ("texture", "material", "mesh"):
            continue
        new_name = f"{prefix}_{old_name}"
        asset.set("name", new_name)
        if asset.tag == "texture":
            texture_names[old_name] = new_name
            filename = asset.get("file")
            if filename:
                _copy_asset(source / filename, TEXTURE_DIR / name / filename)
                # TIAGo++'s compiler defines meshdir but not texturedir.
                asset.set("file", f"caremate_scanned_textures/{name}/{filename}")
        elif asset.tag == "material":
            material_names[old_name] = new_name
        elif asset.tag == "mesh":
            # In proxy mode, skip loading all 32 detailed convex pieces.
            if collision_mode == "proxy" and old_name != "model":
                continue
            mesh_names[old_name] = new_name
            filename = asset.get("file")
            if filename:
                _copy_asset(source / filename, MESH_DIR / name / filename)
                # TIAGo++'s compiler meshdir points to ROBOT_DIR/assets.
                asset.set("file", f"caremate_scanned/{name}/{filename}")
        target_assets.append(asset)

    for asset in target_assets:
        if asset.tag == "material" and asset.get("name") == material_names.get("material_0"):
            old_tex = asset.get("texture")
            if old_tex:
                asset.set("texture", texture_names[old_tex])

    # Put bottom of OBJ exactly above the tabletop using measured OBJ bounds.
    # NOTE: GSO assets are authored with Z pointing upward.
    x, y = cfg["xy"]
    bottom_clearance = 0.004
    z_body = table_top + bottom_clearance - float(bounds_low[2])
    body = ET.SubElement(world, "body", {
        "name": name,
        "pos": _fmt((x, y, z_body)),
        "euler": _fmt((0, 0, cfg["yaw"])),
    })
    ET.SubElement(body, "freejoint", {"name": f"{name}_free"})

    # Simplified uniform-box inertia approximation. This makes bodies movable,
    # while avoiding guessed mass calculated from detailed scanned collision meshes.
    mass = float(cfg["mass"])
    dx, dy, dz = dimensions
    diagonal_inertia = mass * np.array([dy*dy + dz*dz,
                                         dx*dx + dz*dz,
                                         dx*dx + dy*dy]) / 12.0
    diagonal_inertia = np.maximum(diagonal_inertia, 1e-6)
    ET.SubElement(body, "inertial", {
        "pos": _fmt((bounds_low + bounds_high) / 2),
        "mass": f"{mass:.5g}",
        "diaginertia": _fmt(diagonal_inertia),
    })
    original_geoms = list(source_body.findall("geom"))
    visual_geoms = [g for g in original_geoms if g.get("mesh") == "model"]
    if len(visual_geoms) != 1:
        raise ValueError(f"Expected exactly one visual geom for {name}")
    visual = copy.deepcopy(visual_geoms[0])
    visual.set("name", f"{name}_visual")
    visual.set("mesh", mesh_names["model"])
    if visual.get("material"):
        visual.set("material", material_names[visual.get("material")])
    visual.set("group", "2")
    visual.set("contype", "0")
    visual.set("conaffinity", "0")
    visual.set("mass", "0")
    body.append(visual)

    if collision_mode == "proxy":
        # Exact visual mesh + inexpensive, approximate contact geometry.
        ET.SubElement(body, "geom", {
            "name": f"{name}_collision_proxy", "type": "box",
            "pos": _fmt((bounds_low + bounds_high) / 2),
            "size": _fmt(np.maximum(dimensions / 2, 0.004)),
            "group": "3", "rgba": "0 0 0 0",
            "contype": "1", "conaffinity": "1", "condim": "3",
            "friction": "0.9 0.01 0.001", "mass": "0",
        })
    else:
        collision_geoms = [g for g in original_geoms if g.get("mesh", "").startswith("model_collision_")]
        if not collision_geoms:
            raise ValueError(f"No GSO collision meshes found for {name}")
        for n, source_geom in enumerate(collision_geoms):
            collision = copy.deepcopy(source_geom)
            collision.set("name", f"{name}_collision_{n}")
            collision.set("mesh", mesh_names[source_geom.get("mesh")])
            collision.set("group", "3")
            collision.set("rgba", "0 0 0 0")
            collision.set("contype", "1")
            collision.set("conaffinity", "1")
            collision.set("condim", "3")
            collision.set("friction", "0.9 0.01 0.001")
            collision.set("mass", "0")
            body.append(collision)
    print(f"  {name}: size={_fmt(dimensions)} m, bottom={table_top+bottom_clearance:.3f} m, {collision_mode} collision")


def build(*, collision_mode: str = "proxy") -> Path:
    if collision_mode not in ("proxy", "full"):
        raise ValueError("collision_mode must be 'proxy' or 'full'")
    if not FURNISHED_XML.is_file():
        raise FileNotFoundError("Missing furnished scene. Run: python -m caremate.build_furnished_scene")
    root = ET.parse(FURNISHED_XML).getroot()
    root.set("model", "CareMate v0.3 - scanned everyday objects")
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("Scene has no worldbody")

    # Remove only previous synthetic tabletop objects; don't modify residents,
    # the TIAGo++ model, original project XMLs, or imported studyTable.
    for body in list(world):
        if body.tag == "body" and body.get("name") in {"tissue_box", "wipes_box", "book"}:
            world.remove(body)

    # furniture_sim studyTable top surface is 0.760 m in this scene.
    for cfg in OBJECTS:
        _import_object(root, world, cfg, collision_mode=collision_mode, table_top=0.760)

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(SCANNED_XML, encoding="unicode", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(SCANNED_XML))
    print(f"PASS: scanned scene compiled nq={model.nq}, nu={model.nu}, ngeom={model.ngeom}")
    print(f"Saved: {SCANNED_XML}")
    print("Existing v0.1 / furnished scenes and third_party sources unchanged.")
    return SCANNED_XML


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collision-mode", choices=("proxy", "full"), default="proxy")
    args = parser.parse_args()
    build(collision_mode=args.collision_mode)


if __name__ == "__main__":
    main()
