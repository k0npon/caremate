"""Non-destructive integration of furniture_sim's studyTable into CareMate.

Requires the existing working CareMate v0.1 scene and a local clone of
https://github.com/vikashplus/furniture_sim

Never edits the robot XML, existing CareMate scene or third_party repository.
"""
from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from caremate.build_scene import ROOT, ROBOT_DIR, SCENE_XML

FURNITURE = ROOT / "third_party" / "furniture_sim"
FURNISHED_XML = ROBOT_DIR / "caremate_furnished_scene.xml"
ASSET_DIR = ROBOT_DIR / "assets" / "caremate_furniture" / "studyTable"
TEXTURE_DIR = ROBOT_DIR / "caremate_furniture_textures" / "studyTable"

# The pre-existing CareMate box table occupies this spot.
REPLACED_GEOMS = {"storage_top", "storage_leg_left", "storage_leg_right"}

# furniture_sim/studyTable_body.xml uses visual-default classes. Expand the
# classes explicitly to avoid colliding with TIAGo++'s <default> hierarchy.
VISUAL_CLASSES = {
    "viz_stl_Wood_studyTable": "M_Wood_studyTable",
    "viz_stl_Metal_studyTable": "M_Metal_studyTable",
}

# Active drawer / slide joints, copied from furniture_sim's default definitions.
SLIDER_PARAMETERS = {"damping": "2", "frictionloss": "5", "armature": "0.1"}


def _copy_asset(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Missing furniture_sim asset: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Copy rather than symlink: independent of third_party checkout layout.
    shutil.copy2(source, destination)


def _import_assets(root: ET.Element, asset_xml: Path) -> None:
    source = ET.parse(asset_xml).getroot().find("asset")
    if source is None:
        raise RuntimeError(f"No <asset> in {asset_xml}")
    target = root.find("asset")
    if target is None:
        target = ET.SubElement(root, "asset")

    for asset in source:
        if asset.tag not in {"mesh", "texture", "material"}:
            continue
        file = asset.get("file")
        if asset.tag == "mesh" and file:
            # TIAGo++'s compiler has meshdir='./assets/'.
            # Thus this relative filename resolves beneath ROBOT_DIR/assets.
            source_path = FURNITURE / "studyTable" / "meshes" / Path(file).name
            target_path = ASSET_DIR / "meshes" / source_path.name
            _copy_asset(source_path, target_path)
            asset.set("file", f"caremate_furniture/studyTable/meshes/{source_path.name}")
        elif asset.tag == "texture" and file:
            # No texturedir in TIAGo++ compiler; textures resolve relative
            # to the generated scene XML, which lives in ROBOT_DIR.
            source_path = FURNITURE / "common" / "textures" / Path(file).name
            target_path = TEXTURE_DIR / source_path.name
            _copy_asset(source_path, target_path)
            asset.set("file", f"caremate_furniture_textures/studyTable/{source_path.name}")
        target.append(asset)


def _make_geometry_explicit(body: ET.Element) -> None:
    body.attrib.pop("childclass", None)
    for element in body.iter():
        if element.tag == "geom":
            geom_class = element.attrib.pop("class", None)
            if geom_class in VISUAL_CLASSES:
                element.set("type", "mesh")
                element.set("material", VISUAL_CLASSES[geom_class])
                element.set("group", "1")
                element.set("contype", "0")
                element.set("conaffinity", "0")
                # Imported STL is VISUAL ONLY, no inferred mesh mass.
                element.set("mass", "0")
            elif geom_class is None and element.get("type") != "mesh":
                # Keep original source collision proxies, but don't overlay
                # featureless pastel boxes on the textured wooden meshes.
                element.set("group", "3")
                element.set("contype", "1")
                element.set("conaffinity", "1")
                element.set("rgba", "0 0 0 0")
        elif element.tag == "joint":
            element.attrib.pop("class", None)
            for key, value in SLIDER_PARAMETERS.items():
                element.set(key, value)


def build() -> Path:
    if not SCENE_XML.is_file():
        raise FileNotFoundError("Missing base scene; first run: python -m caremate.build_scene")
    assets_source = FURNITURE / "studyTable" / "studyTable_asset.xml"
    bodies_source = FURNITURE / "studyTable" / "studyTable_body.xml"
    for f in (assets_source, bodies_source):
        if not f.is_file():
            raise FileNotFoundError(
                f"Missing {f}. First clone furniture_sim to third_party/furniture_sim"
            )

    root = ET.parse(SCENE_XML).getroot()
    root.set("model", "CareMate - studyTable furniture")
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("CareMate scene has no <worldbody>")

    # Remove ONLY the old box-table. Keep delivery table and resident intact.
    for geom in list(world):
        if geom.tag == "geom" and geom.get("name") in REPLACED_GEOMS:
            world.remove(geom)

    _import_assets(root, assets_source)

    # Rotate 180 degrees: drawers face the robot, rear hutch faces away.
    mount = ET.SubElement(world, "body", {
        "name": "caremate_storage_studytable",
        "pos": "1.4 -0.45 0",
        "euler": "0 0 3.141592653589793",
    })
    sources = ET.parse(bodies_source).getroot()
    bodies = list(sources.findall("body"))
    if not bodies:
        raise RuntimeError(f"No furniture <body> elements in {bodies_source}")
    for body in bodies:
        _make_geometry_explicit(body)
        mount.append(body)

    # Original storage tabletop top z = .735 + .035 = .770.
    # studyTable tabletop top z = .735 + .025 = .760.
    # Place all original graspable boxes on this surface.
    positions = {
        "tissue_box": "1.20 -0.57 0.815",  # half-height .055
        "wipes_box": "1.60 -0.50 0.805",   # half-height .045
        "book": "1.37 -0.33 0.782",        # half-height .022
    }
    for name, pos in positions.items():
        body = world.find(f"./body[@name='{name}']")
        if body is None:
            raise RuntimeError(f"Missing original CareMate object: {name}")
        body.set("pos", pos)

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(FURNISHED_XML, encoding="unicode", xml_declaration=True)
    # Validate on the user's local MuJoCo, with the real mesh/texture assets.
    model = mujoco.MjModel.from_xml_path(str(FURNISHED_XML))
    print(f"PASS: furnished scene compiled (nq={model.nq}, nu={model.nu}, ngeom={model.ngeom})")
    print(f"Saved: {FURNISHED_XML}")
    print("Original caremate_scene.xml and third_party repositories are unchanged.")
    return FURNISHED_XML


if __name__ == "__main__":
    build()
