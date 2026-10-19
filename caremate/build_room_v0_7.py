"""CareMate v0.7 scene: v0.4 room + robot-side lighting + look-alike distractors.

Changes vs v0.4 (v0.4 XML and builder are left untouched):
  1. Lighting.  Both v0.4 lights shine from +x (behind the table as seen by
     TIAGo++), so every object face toward the robot was in shadow and its
     texture nearly black.  v0.7 adds a ceiling key light on the robot side
     and a weak fill, like an ordinary room ceiling lamp.  The
     original lights are kept (dimmed) so shadows remain.
  2. Distractors for identity evaluation: two Nestle Crunch boxes with
     near-identical packaging (look-alike pair) and a cereal box.  Without
     them, "found the only textured object" cannot be distinguished from
     "recognised the target".
  3. Book turned so its front cover faces the robot (v0.4 showed the back
     cover, mostly small text).  --layout builds target-absent / hard variants.

Runtime perception must still never read object positions or IDs from here.

  python -m caremate.build_room_v0_7                      # default layout
  python -m caremate.build_room_v0_7 --layout no_book     # target absent
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from caremate.build_scene import ROBOT_DIR
from caremate.build_room_v0_4 import SCENE_V04
from caremate.build_scanned_scene import GSO, _import_object

SCENE_V07 = ROBOT_DIR / "caremate_room_v0_7.xml"
TABLE_TOP = 0.760

DISTRACTORS = [
    {"name": "crunch_caramel", "folder": "Nestl_Crunch_Girl_Scouts_Cookie_Flavors_Caramel_Coconut_78_oz_box",
     "xy": (1.10, -0.33), "yaw": -1.5708, "mass": 0.22},
    {"name": "crunch_peanut", "folder": "Crunch_Girl_Scouts_Candy_Bars_Peanut_Butter_Creme_78_oz_box",
     "xy": (1.42, -0.74), "yaw": -1.5708, "mass": 0.22},
    {"name": "cereal_box", "folder": "Vans_Cereal_Honey_Nut_Crunch_11_oz_box",
     "xy": (1.62, -0.30), "yaw": -1.5708, "mass": 0.32},
]

# Book yaw: GSO front cover normal is -y at yaw 0 (back cover +y).  -2.06 rad turns the front
# cover toward the robot base at the origin; 1.08 shows the back cover
# (mostly small text: a hard case).  v0.4 used yaw 0 (back cover, ~60 deg oblique).
BOOK_FRONT, BOOK_BACK = -2.06, 1.08
LAYOUTS = {
    "default": {"remove": [], "yaw": {"reading_book": BOOK_FRONT}},
    "no_book": {"remove": ["reading_book"], "yaw": {}},                     # target absent
    "no_distractors": {"remove": [d["name"] for d in DISTRACTORS], "yaw": {"reading_book": BOOK_FRONT}},
    "book_back": {"remove": [], "yaw": {"reading_book": BOOK_BACK}},       # hard: back cover
    "book_v04": {"remove": [], "yaw": {}},                                 # v0.4 pose
}


def relight(world: ET.Element) -> None:
    for name, dim in (("caremate_room_light", "0.45 0.45 0.45"),
                      ("caremate_fill_light", "0.25 0.25 0.25")):
        light = world.find(f"./light[@name='{name}']")
        if light is not None:
            light.set("diffuse", dim)
    # Ceiling lamp between robot (origin) and table (x~1.4), aimed at table.
    ET.SubElement(world, "light", {
        "name": "v07_ceiling_key", "pos": "0.35 -0.45 2.6", "dir": "0.45 0 -1",
        "diffuse": "0.75 0.75 0.72", "specular": "0.1 0.1 0.1",
        "castshadow": "true", "cutoff": "70", "exponent": "2"})
    ET.SubElement(world, "light", {
        "name": "v07_robot_fill", "pos": "-0.6 0.6 1.6", "dir": "1 -0.6 -0.4",
        "diffuse": "0.25 0.25 0.25", "specular": "0 0 0", "castshadow": "false"})


def build(layout: str = "default", out: Path | None = None) -> Path:
    if not SCENE_V04.is_file():
        raise FileNotFoundError("Build v0.4 first: python -m caremate.build_room_v0_4")
    cfg = LAYOUTS[layout]
    root = ET.parse(SCENE_V04).getroot()
    root.set("model", f"CareMate v0.7 - {layout}")
    world = root.find("worldbody")
    relight(world)
    for name, yaw in cfg.get("yaw", {}).items():
        body = world.find(f"./body[@name='{name}']")
        if body is not None:
            body.attrib.pop("quat", None)
            body.set("euler", f"0 0 {yaw}")
    for name in cfg["remove"]:
        body = world.find(f"./body[@name='{name}']")
        if body is not None:
            world.remove(body)
    for d in DISTRACTORS:
        if d["name"] in cfg["remove"]:
            continue
        if not (GSO / d["folder"] / "model.xml").is_file():
            raise FileNotFoundError(
                f"Missing GSO {d['folder']}: cd third_party/mujoco_scanned_objects && "
                f"git sparse-checkout add models/{d['folder']}")
        _import_object(root, world, d, collision_mode="proxy", table_top=TABLE_TOP)
    out = out or (SCENE_V07 if layout == "default" else ROBOT_DIR / f"caremate_room_v0_7_{layout}.xml")
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(out, encoding="unicode", xml_declaration=True)
    m = mujoco.MjModel.from_xml_path(str(out))
    print(f"PASS: {out.name} compiles: nbody={m.nbody} ngeom={m.ngeom} nlight={m.nlight}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", choices=list(LAYOUTS), default="default")
    build(ap.parse_args().layout)
