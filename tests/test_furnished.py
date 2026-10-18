"""Smoketest furnished scene; run after building it."""
import mujoco

from caremate.build_furnished_scene import FURNISHED_XML


def main():
    if not FURNISHED_XML.is_file():
        raise FileNotFoundError("Run: python -m caremate.build_furnished_scene")
    model = mujoco.MjModel.from_xml_path(str(FURNISHED_XML))
    for body_name in ("caremate_storage_studytable", "studyTable_Main", "resident_static"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) >= 0, body_name
    for obj_name in ("tissue_box", "wipes_box", "book"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj_name) >= 0, obj_name
    print("PASS: furnished CareMate scene, robot resident, objects and studyTable")


if __name__ == "__main__":
    main()
