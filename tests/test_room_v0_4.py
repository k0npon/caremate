"""MuJoCo compatibility/sanity checks on the user's installed robot assets."""
import xml.etree.ElementTree as ET
import mujoco
from caremate.build_room_v0_4 import SCENE_V04


def test_scene() -> None:
    assert SCENE_V04.is_file(), "First run: python -m caremate.build_room_v0_4"
    root = ET.parse(SCENE_V04).getroot()
    world = root.find("worldbody")
    assert world is not None
    assert all(world.find(f"./geom[@name='{n}']") is None for n in ("north_wall", "west_wall"))
    floor = world.find("./geom[@name='caremate_floor']")
    assert floor is not None and floor.get("material") == "caremate_floor_mat_v04"
    assert world.find("./body[@name='caremate_storage_studytable']") is None
    assert world.find("./body[@name='caremate_accessible_storage_table']") is not None
    resident = world.find("./body[@name='resident_static']")
    assert resident is not None
    assert len(resident.findall("geom")) > 8
    assert all(world.find(f"./body[@name='{n}']") is not None
               for n in ("reading_book", "personal_care_box"))
    goal = world.find("./site[@name='delivery_goal']")
    assert goal is not None and goal.get("rgba") == "0 0 0 0"
    model = mujoco.MjModel.from_xml_path(str(SCENE_V04))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_rgb") >= 0
    print(f"PASS: open room, seated humanoid, scanned objects; {model.ngeom} geoms")


if __name__ == "__main__":
    test_scene()
