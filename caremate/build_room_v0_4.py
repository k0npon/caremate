"""CareMate v0.4: readable care-room layout built from the existing v0.3 scene.

Does NOT alter prior scenes or third-party source directories.
Requires: build_scanned_scene.py has generated caremate_scanned_scene.xml,
furniture_sim's simpleTable model, and official MuJoCo humanoid.xml.

Changes: remove occluding walls & studyTable hutch; MuJoCo checker floor and
skybox; open wooden worktable from furniture_sim; actually seated humanoid pose;
4 plausible, textured household objects (2 mugs optional if sparse checkout
has not downloaded them). Purely static scene preview -- not safe navigation.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from caremate.build_scene import ROOT, ROBOT_DIR, HUMANOID_XML, fmt
from caremate.build_scanned_scene import (
    SCANNED_XML, GSO, obj_bounds, _import_object,
)

SCENE_V04 = ROBOT_DIR / "caremate_room_v0_4.xml"
TABLE_PART = ROOT / "third_party/furniture_sim/simpleTable/simpleWoodTable_body.xml"

# Object names are instance labels (not semantic class predictions).
OBJECTS = (
    {"name": "reading_book", "folder": "Eat_to_Live_The_Amazing_NutrientRich_Program_for_Fast_and_Sustained_Weight_Loss_Revised_Edition_Book", "xy": (1.06, -0.57), "yaw": 0.0, "mass": 0.35},
    {"name": "personal_care_box", "folder": "Marc_Anthony_Strictly_Curls_Curl_Envy_Perfect_Curl_Cream_6_fl_oz_bottle", "xy": (1.44, -0.54), "yaw": 0.0, "mass": 0.18},
    {"name": "blue_mug", "folder": "Cole_Hardware_Mug_Classic_Blue", "xy": (1.79, -0.62), "yaw": 0.0, "mass": 0.28},
    {"name": "yellow_mug", "folder": "Room_Essentials_Mug_White_Yellow", "xy": (1.78, -0.25), "yaw": 0.0, "mass": 0.28},
)


def _asset(root: ET.Element) -> ET.Element:
    section = root.find("asset")
    if section is None:
        section = ET.SubElement(root, "asset")
    return section


def fix_floor_and_background(root: ET.Element, world: ET.Element) -> None:
    """No walls. MuJoCo-style sky gradient and blue-grey checker plane."""
    for geom in list(world):
        if geom.tag == "geom" and geom.get("name") in {"north_wall", "west_wall"}:
            world.remove(geom)
    assets = _asset(root)
    if not any(t.tag == "texture" and t.get("type") == "skybox" for t in assets):
        ET.SubElement(assets, "texture", {
            "name": "caremate_sky_v04", "type": "skybox", "builtin": "gradient",
            "rgb1": "0.56 0.69 0.80", "rgb2": "0.92 0.95 0.97",
            "width": "256", "height": "512",
        })
    if not any(t.get("name") == "caremate_floorgrid_v04" for t in assets):
        ET.SubElement(assets, "texture", {
            "name": "caremate_floorgrid_v04", "type": "2d", "builtin": "checker",
            "rgb1": "0.27 0.37 0.47", "rgb2": "0.40 0.51 0.60",
            "width": "512", "height": "512",
        })
    if not any(m.get("name") == "caremate_floor_mat_v04" for m in assets):
        ET.SubElement(assets, "material", {
            "name": "caremate_floor_mat_v04", "texture": "caremate_floorgrid_v04",
            "texrepeat": "4 4", "texuniform": "true", "reflectance": "0.08",
        })
    floor = world.find("./geom[@name='caremate_floor']")
    if floor is None:
        floor = ET.SubElement(world, "geom", {"name": "caremate_floor"})
    floor.set("type", "plane")
    floor.set("size", "0 0 0.05")
    floor.set("pos", "0 0 0")
    floor.set("material", "caremate_floor_mat_v04")
    floor.set("friction", "1 0.01 0.001")
    floor.attrib.pop("rgba", None)

    # Viewer framing improves when all scene contents are in a known area.
    stat = root.find("statistic")
    if stat is None:
        stat = ET.SubElement(root, "statistic")
    stat.set("center", "1.45 0 0.75")
    stat.set("extent", "3.2")


def install_open_table(root: ET.Element, world: ET.Element) -> None:
    """Use source geometry from furniture_sim/simpleWoodTable, not a new box hack."""
    if not TABLE_PART.is_file():
        raise FileNotFoundError(f"Missing {TABLE_PART}: clone furniture_sim first")
    if not any(a.tag == "material" and a.get("name") == "M_Wood_studyTable" for a in _asset(root)):
        raise RuntimeError("Expected studyTable wood material from furnished scene")

    # Remove only the obstructing studyTable hutch, not imported asset definitions.
    old_study = world.find("./body[@name='caremate_storage_studytable']")
    if old_study is not None:
        world.remove(old_study)

    mount = ET.SubElement(world, "body", {
        "name": "caremate_accessible_storage_table", "pos": "1.4 -0.45 0",
    })
    doc = ET.parse(TABLE_PART).getroot()
    source = doc.find("body")
    if source is None:
        raise RuntimeError("simpleWoodTable_body.xml has no <body>")
    original = copy.deepcopy(source)
    original.set("name", "caremate_open_table")
    original.attrib.pop("childclass", None)
    for number, geom in enumerate(original.iter("geom")):
        was_visual = geom.attrib.pop("class", "").startswith("viz_")
        geom.set("name", f"caremate_open_table_{'visual' if was_visual else 'collision'}_{number}")
        geom.set("type", "box")
        geom.set("group", "1" if was_visual else "3")
        geom.set("contype", "0" if was_visual else "1")
        geom.set("conaffinity", "0" if was_visual else "1")
        geom.set("mass", "0")
        geom.attrib.pop("rgba", None)
        if was_visual:
            geom.set("material", "M_Wood_studyTable")
        else:
            geom.set("rgba", "0 0 0 0")
            geom.attrib.pop("material", None)
    mount.append(original)

    # Match delivery tabletop wood material; keep a collision surface.
    for gname in ("delivery_top", "delivery_leg"):
        g = world.find(f"./geom[@name='{gname}']")
        if g is not None:
            g.set("material", "M_Wood_studyTable")
            g.attrib.pop("rgba", None)
    # Position the bedside table near the seated resident's front-left side,
    # not one metre away from the wheelchair as in v0.1. This is still a
    # simulated placing surface, NOT direct person-to-robot handover.
    delivery_top = world.find("./geom[@name='delivery_top']")
    if delivery_top is not None:
        delivery_top.set("pos", "2.15 0.46 0.735")
    goal_site = world.find("./site[@name='delivery_goal']")
    if goal_site is not None:
        goal_site.set("pos", "2.00 0.40 0.805")
    # Replace the old single-leg handover plinth with a low four-leg bedside
    # table. Reuse imported wood material. All geoms are static and collidable.
    old = world.find("./geom[@name='delivery_leg']")
    if old is not None:
        world.remove(old)
    for ix, x in enumerate((1.87, 2.43)):
        for iy, y in enumerate((0.29, 0.63)):
            ET.SubElement(world, "geom", {
                "name": f"delivery_leg_{ix}_{iy}", "type": "box",
                "pos": f"{x} {y} 0.35", "size": "0.025 0.025 0.35",
                "material": "M_Wood_studyTable", "friction": "1 0.01 0.001",
            })


def install_seated_resident(world: ET.Element) -> None:
    """Bake a real seated pose of official articulated humanoid into fixed geoms.

    The torso and leg joint angles come from the original humanoid's joints.
    A static collision proxy is retained; no human-contact force claim is made.
    Robot perception never gets ground-truth human coordinates from this model.
    """
    old_resident = world.find("./body[@name='resident_static']")
    if old_resident is not None:
        world.remove(old_resident)
    hm = mujoco.MjModel.from_xml_path(str(HUMANOID_XML))
    hd = mujoco.MjData(hm)
    mujoco.mj_resetData(hm, hd)
    # Original torso root is 1.282 m in standing pose. Lower to chair height.
    hd.qpos[:3] = [0.0, 0.0, 0.965]
    hd.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    # Hip flexion brings upper legs forward; knee flexion lowers shins.
    for jn in ("hip_y_left", "hip_y_right", "knee_left", "knee_right"):
        jid = mujoco.mj_name2id(hm, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise RuntimeError(f"Humanoid joint {jn} missing")
        angle = -1.50 if jn.startswith("hip_") else -1.50
        hd.qpos[int(hm.jnt_qposadr[jid])] = angle
    mujoco.mj_forward(hm, hd)

    # Chair back is at world +X, therefore resident faces world -X.
    mount = ET.SubElement(world, "body", {
        "name": "resident_static", "pos": "2.70 1.00 0",
        "euler": f"0 0 {math.pi:.12g}",
    })
    geom_map = {
        int(mujoco.mjtGeom.mjGEOM_SPHERE): ("sphere", 1),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE): ("capsule", 2),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): ("cylinder", 2),
        int(mujoco.mjtGeom.mjGEOM_BOX): ("box", 3),
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): ("ellipsoid", 3),
    }
    num = 0
    for gid in range(hm.ngeom):
        if int(hm.geom_bodyid[gid]) == 0:
            continue
        ty = int(hm.geom_type[gid])
        if ty not in geom_map:
            continue
        shape, num_size = geom_map[ty]
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, hd.geom_xmat[gid].reshape(-1))
        source_name = mujoco.mj_id2name(hm, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        skin = "head" in source_name or "hand" in source_name
        ET.SubElement(mount, "geom", {
            "name": f"resident_seated_visual_{num}", "type": shape,
            "size": fmt(hm.geom_size[gid, :num_size]),
            "pos": fmt(hd.geom_xpos[gid]), "quat": fmt(q),
            "rgba": "0.69 0.49 0.37 1" if skin else "0.27 0.45 0.62 1",
            "group": "2", "contype": "0", "conaffinity": "0",
        })
        num += 1
    if num == 0:
        raise RuntimeError("Couldn't bake humanoid geometry")
    ET.SubElement(mount, "geom", {
        "name": "resident_seated_collision", "type": "capsule",
        "fromto": "0 0 0.59 0 0 1.05", "size": "0.20",
        "rgba": "0 0 0 0", "group": "3", "contype": "1", "conaffinity": "1",
    })
    # Match chair seat to pelvis height; hips sit above seat, not through it.
    adjustments = {
        "wheelchair_seat": ("2.70 1.00 0.40", "0.30 0.28 0.055"),
        "wheelchair_back": ("2.91 1.00 0.72", "0.06 0.28 0.36"),
        "wheelchair_arm_L": ("2.70 0.68 0.62", "0.27 0.035 0.04"),
        "wheelchair_arm_R": ("2.70 1.32 0.62", "0.27 0.035 0.04"),
    }
    for name, (pos, size) in adjustments.items():
        geom = world.find(f"./geom[@name='{name}']")
        if geom is None:
            raise RuntimeError(f"Missing wheelchair piece {name}")
        geom.set("pos", pos)
        geom.set("size", size)


def _proxy_if_full(body: ET.Element, cfg: dict) -> None:
    """Normalize preexisting scanned objects to visual + single proxy geom.

    User last generated v0.3 in 'full' mode (32 hulls per object). Keep that
    scene untouched but make v0.4 usable on an older T500 laptop.
    """
    name = cfg["name"]
    full = [g for g in body.findall("geom") if g.get("name", "").startswith(f"{name}_collision_")]
    if not full:
        return
    for g in full:
        body.remove(g)
    lo, hi = obj_bounds(GSO / cfg["folder"] / "model.obj")
    ET.SubElement(body, "geom", {
        "name": f"{name}_collision_proxy", "type": "box",
        "pos": fmt((lo + hi) / 2),
        "size": fmt(np.maximum((hi - lo) / 2, 0.004)),
        "group": "3", "rgba": "0 0 0 0", "contype": "1", "conaffinity": "1",
        "friction": "0.9 0.01 0.001", "mass": "0",
    })


def arrange_objects(root: ET.Element, world: ET.Element) -> int:
    installed = 0
    for cfg in OBJECTS:
        name = cfg["name"]
        old = world.find(f"./body[@name='{name}']")
        if old is not None:
            # Preserve existing scanned visual assets and the correct z offset.
            xyz = np.array([float(v) for v in old.get("pos", "0 0 0").split()])
            xyz[:2] = cfg["xy"]
            old.set("pos", fmt(xyz))
            _proxy_if_full(old, cfg)
            installed += 1
        else:
            if not (GSO / cfg["folder"] / "model.xml").exists():
                print(f"SKIP optional {name}: GSO model not downloaded")
                continue
            _import_object(root, world, cfg, collision_mode="proxy", table_top=0.760)
            installed += 1
    if installed < 2:
        raise RuntimeError("Need at least the reading_book and personal_care_box models")
    return installed


def build() -> Path:
    if not SCANNED_XML.is_file():
        raise FileNotFoundError("Run: python -m caremate.build_scanned_scene")
    root = ET.parse(SCANNED_XML).getroot()
    root.set("model", "CareMate v0.4 - open care room")
    world = root.find("worldbody")
    if world is None:
        raise ValueError("No worldbody")
    fix_floor_and_background(root, world)
    install_open_table(root, world)
    install_seated_resident(world)
    count = arrange_objects(root, world)
    # Prevent unused GSO collision meshes from being loaded when source v0.3
    # was last compiled with --collision-mode full. Source XML stays intact.
    assets = root.find("asset")
    if assets is not None:
        for a in list(assets):
            if a.tag == "mesh" and any(
                a.get("name", "").startswith(f"gso_{cfg['name']}_model_collision_")
                for cfg in OBJECTS
            ):
                assets.remove(a)
    # The green delivery-goal site is a controller target, not a decorative prop.
    goal = world.find("./site[@name='delivery_goal']")
    if goal is not None:
        goal.set("rgba", "0 0 0 0")
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(SCENE_V04, encoding="unicode", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(SCENE_V04))
    print(f"PASS: v0.4 compiles: nq={model.nq} nu={model.nu} ngeom={model.ngeom}")
    print(f"Installed realistic objects: {count}")
    print(f"Generated: {SCENE_V04}")
    print("Existing v0.1/v0.2/v0.3 scenes untouched.")
    return SCENE_V04


if __name__ == "__main__":
    build()
