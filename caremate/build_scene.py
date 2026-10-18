"""Build CareMate room by extending the original TIAGo++ MJCF.

Downloads are separate. The generated XML is stored beside the Menagerie robot
XML so its relative `assets/` mesh paths remain valid. This does not modify the
upstream robot description.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
ROBOT_DIR = MODELS / "pal_tiago_dual"
ROBOT_XML = ROBOT_DIR / "tiago_dual.xml"
HUMANOID_XML = MODELS / "humanoid.xml"
SCENE_XML = ROBOT_DIR / "caremate_scene.xml"


def fmt(values) -> str:
    return " ".join(f"{float(x):.8g}" for x in values)


def el(parent: ET.Element, tag: str, **kwargs) -> ET.Element:
    return ET.SubElement(parent, tag, {key: str(value) for key, value in kwargs.items()})


def generate_product_texture(path: Path, label: str, base: tuple[int, int, int]) -> None:
    """Generate high-contrast synthetic packaging to support later CV experiments."""
    im = Image.new("RGB", (512, 512), base)
    draw = ImageDraw.Draw(im)
    draw.rectangle((18, 18, 494, 494), outline=(25, 39, 51), width=11)
    draw.rectangle((20, 145, 490, 348), fill=(245, 244, 233))
    for i in range(5):
        x = 45 + i * 92
        draw.ellipse((x, 40, x + 42, 82), fill=(30, 70 + 20 * i, 95))
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), label, font=font)
    text_width = box[2] - box[0]
    x = max(32, (512 - text_width) // 2)
    draw.text((x, 240), label, fill=(23, 41, 53), font=font)
    for i in range(7):
        draw.line((35, 395 + i * 9, 475, 380 + i * 7), fill=(45, 65, 78), width=3)
    im.save(path)


def add_static_humanoid(world: ET.Element) -> None:
    """Bake MuJoCo's own humanoid `squat` keyframe into stationary visual geoms.

    This avoids a second dynamic free body falling down and keeps the visual
    human geometry from the actual official Humanoid model. It is an approximate
    seated resident, not a clinically realistic seated biomechanics model.
    """
    human_model = mujoco.MjModel.from_xml_path(str(HUMANOID_XML))
    human_data = mujoco.MjData(human_model)
    key = mujoco.mj_name2id(human_model, mujoco.mjtObj.mjOBJ_KEY, "squat")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(human_model, human_data, key)
    else:
        raise RuntimeError("Official humanoid model has no 'squat' keyframe.")
    mujoco.mj_forward(human_model, human_data)

    # The resident is a *fixed visual* derived from real humanoid joint geometry.
    resident = el(world, "body", name="resident_static", pos="2.70 1.00 0")
    types = {
        int(mujoco.mjtGeom.mjGEOM_SPHERE): ("sphere", 1),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE): ("capsule", 2),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): ("cylinder", 2),
        int(mujoco.mjtGeom.mjGEOM_BOX): ("box", 3),
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): ("ellipsoid", 3),
    }
    n_added = 0
    for i in range(human_model.ngeom):
        # Only body geoms, not the background floor.
        if int(human_model.geom_bodyid[i]) == 0:
            continue
        geom_type = int(human_model.geom_type[i])
        if geom_type not in types:
            continue
        tag, n_size = types[geom_type]
        quat = np.empty(4, dtype=float)
        mujoco.mju_mat2Quat(quat, human_data.geom_xmat[i].reshape(-1))
        source_name = mujoco.mj_id2name(human_model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        skin = "head" in source_name or "hand" in source_name
        rgba = "0.68 0.48 0.36 1" if skin else "0.38 0.53 0.66 1"
        el(resident, "geom", name=f"resident_visual_{n_added}", type=tag,
           size=fmt(human_model.geom_size[i, :n_size]),
           pos=fmt(human_data.geom_xpos[i]), quat=fmt(quat),
           rgba=rgba, contype="0", conaffinity="0", group="2")
        n_added += 1
    if not n_added:
        raise RuntimeError("No humanoid geometry was found to bake.")
    # Separate, conservative collision proxy around torso, away from the handover table.
    el(resident, "geom", name="resident_torso_collision", type="capsule",
       fromto="0 0 0.67 0 0 1.1", size="0.22", rgba="0 0 0 0",
       contype="1", conaffinity="1", group="3")


def add_room(world: ET.Element) -> None:
    el(world, "geom", name="caremate_floor", type="plane", size="6 6 0.1",
       rgba="0.89 0.91 0.88 1", friction="1 0.01 0.001", condim="3")
    el(world, "light", name="caremate_room_light", directional="true",
       pos="1 -1 3.5", dir="-0.2 0.3 -1", diffuse="0.85 0.85 0.85")
    el(world, "light", name="caremate_fill_light", pos="3 2 3", dir="-0.3 -0.2 -1",
       diffuse="0.45 0.45 0.45")
    for name, pos, half, rgba in [
        ("north_wall", "1.4 2.45 1.4", "4.0 0.06 1.4", "0.89 0.94 0.95 1"),
        ("west_wall", "-1.4 0 1.4", "0.06 2.5 1.4", "0.84 0.89 0.91 1"),
        ("storage_top", "1.4 -0.45 0.735", "0.48 0.36 0.035", "0.65 0.46 0.30 1"),
        ("storage_leg_left", "1.05 -0.70 0.355", "0.035 0.035 0.355", "0.48 0.34 0.27 1"),
        ("storage_leg_right", "1.75 -0.70 0.355", "0.035 0.035 0.355", "0.48 0.34 0.27 1"),
        ("delivery_top", "2.5 0.00 0.735", "0.35 0.23 0.035", "0.74 0.64 0.50 1"),
        ("delivery_leg", "2.50 0.00 0.355", "0.045 0.045 0.355", "0.62 0.53 0.45 1"),
        ("wheelchair_seat", "2.70 1.00 0.48", "0.30 0.28 0.06", "0.18 0.24 0.34 1"),
        ("wheelchair_back", "2.91 1.00 0.75", "0.06 0.28 0.36", "0.18 0.24 0.34 1"),
        ("wheelchair_arm_L", "2.7 0.68 0.69", "0.27 0.035 0.04", "0.16 0.20 0.27 1"),
        ("wheelchair_arm_R", "2.7 1.32 0.69", "0.27 0.035 0.04", "0.16 0.20 0.27 1"),
    ]:
        el(world, "geom", name=name, type="box", pos=pos, size=half, rgba=rgba)
    for suffix, y in [("left", 0.72), ("right", 1.28)]:
        for where, x, radius in [("rear", 2.92, 0.26), ("front", 2.47, 0.09)]:
            el(world, "geom", name=f"wheelchair_{suffix}_{where}_wheel",
               type="cylinder", pos=f"{x} {y} {radius}",
               euler="1.57079633 0 0", size=f"{radius} 0.025",
               rgba="0.13 0.14 0.18 1")
    el(world, "site", name="delivery_goal", pos="2.48 0 0.80",
       type="sphere", size="0.035", rgba="0.1 0.8 0.3 0.8")


def add_objects(world: ET.Element) -> None:
    objects = [
        # Top of storage table ~0.77 m; objects remain physically simulated.
        ("tissue_box", "1.20 -0.57 0.825", "0.115 0.080 0.055", "caremate_tissue"),
        ("wipes_box", "1.60 -0.50 0.815", "0.095 0.075 0.045", "caremate_wipes"),
        ("book", "1.37 -0.20 0.792", "0.105 0.082 0.022", "caremate_book"),
    ]
    for name, pos, half_size, material in objects:
        body = el(world, "body", name=name, pos=pos)
        el(body, "freejoint", name=f"{name}_free")
        el(body, "geom", name=f"{name}_geom", type="box", size=half_size,
           material=material, mass="0.16", friction="1.5 0.01 0.001", condim="4")


def add_cameras(root: ET.Element, world: ET.Element) -> None:
    # MuJoCo's camera points along its local -Z and local +Y is image up.
    # Aligning it with the head body may need a small orientation refinement in UI.
    head = world.find(".//body[@name='head_2_link']")
    if head is None:
        raise RuntimeError("Expected Menagerie TIAGo++ head_2_link not found.")
    el(head, "camera", name="head_rgb", pos="0.08 0.04 0.035",
       xyaxes="0 0 1 0 1 0", fovy="65")
    # Fixed, room-wide demo view (world viewer can freely orbit independently).
    el(world, "camera", name="room_overview", pos="1.7 -4.2 2.8",
       xyaxes="0.96 0.28 0 -0.11 0.38 0.92", fovy="65")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    el(visual, "global", offwidth="960", offheight="720")


def add_materials(root: ET.Element) -> None:
    assets = root.find("asset")
    if assets is None:
        assets = ET.SubElement(root, "asset")
    # Since the project XML resides beside the original robot XML, all PNGs do too.
    for n, colour in [("tissue", (92, 188, 220)),
                      ("wipes", (239, 162, 113)),
                      ("book", (153, 185, 115))]:
        filename = f"caremate_{n}.png"
        generate_product_texture(ROBOT_DIR / filename, n.upper(), colour)
        el(assets, "texture", name=f"caremate_{n}_texture",
           type="cube", file=filename)
        el(assets, "material", name=f"caremate_{n}",
           texture=f"caremate_{n}_texture", texuniform="false")


def add_holding_actuators(root: ET.Element, world: ET.Element) -> None:
    """Basic joint-space posture hold, *not* a grasp/manipulation controller."""
    actuators = root.find("actuator")
    if actuators is None:
        actuators = ET.SubElement(root, "actuator")
    for joint in world.findall(".//joint"):
        name = joint.get("name", "")
        if name.startswith("head_") and name.endswith("_joint"):
            el(actuators, "position", name=f"hold_{name}", joint=name,
               kp="30", kv="5")
        elif name.startswith("arm_") and name.endswith("_joint"):
            el(actuators, "position", name=f"hold_{name}", joint=name,
               kp="180", kv="10")
        elif name == "torso_lift_joint":
            el(actuators, "position", name=f"hold_{name}", joint=name,
               kp="300", kv="20")
        elif name.startswith("gripper_") and name.endswith("_joint"):
            el(actuators, "position", name=f"hold_{name}", joint=name,
               kp="100", kv="10")


def build() -> Path:
    if not ROBOT_XML.exists() or not HUMANOID_XML.exists():
        raise FileNotFoundError(
            "Missing models. First run: bash scripts/setup_assets.sh"
        )
    root = ET.parse(ROBOT_XML).getroot()
    root.set("model", "CareMate v0.1")
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("TIAGo++ model has no <worldbody>.")
    add_room(world)
    add_objects(world)
    add_static_humanoid(world)
    add_cameras(root, world)
    add_materials(root)
    add_holding_actuators(root, world)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(SCENE_XML, encoding="unicode", xml_declaration=True)
    # A genuine MuJoCo compile check, run on the user's machine with assets.
    try:
        compiled = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    except Exception as exc:
        raise RuntimeError(f"Scene written but MuJoCo compilation failed: {exc}") from exc
    print(f"Scene compiled: nq={compiled.nq}, nu={compiled.nu}, ngeom={compiled.ngeom}")
    print(f"Saved scene: {SCENE_XML}")
    return SCENE_XML


def main() -> None:
    _ = argparse.ArgumentParser(description=__doc__).parse_args()
    build()


if __name__ == "__main__":
    main()
