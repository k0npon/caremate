"""Smoke-test for the non-destructive GSO object integration."""
import mujoco
from caremate.build_scanned_scene import SCANNED_XML


def main():
    assert SCANNED_XML.is_file(), "Run: python -m caremate.build_scanned_scene"
    model = mujoco.MjModel.from_xml_path(str(SCANNED_XML))
    for name in ["caremate_storage_studytable", "resident_static", "reading_book", "personal_care_box"]:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0, name
    for old in ("tissue_box", "wipes_box", "book"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, old) < 0, old
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_rgb") >= 0
    print("PASS: CareMate scanned scene, TIAGo++ camera, furniture, resident and textured objects")


if __name__ == "__main__":
    main()
