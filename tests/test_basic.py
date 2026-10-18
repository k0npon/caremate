"""Quick import and asset smoke test on the target Ubuntu system."""
from pathlib import Path

import mujoco
import cv2
from caremate.build_scene import SCENE_XML


def test_scene_compiles():
    assert SCENE_XML.exists(), "Run: python -m caremate.build_scene"
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_rgb") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "resident_static") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tissue_box") >= 0
    assert cv2.__version__.startswith("5."), f"OpenCV 5 expected, got {cv2.__version__}"

if __name__ == "__main__":
    test_scene_compiles()
    print("PASS: scene compiles; head camera, resident, tissue box, OpenCV 5 found")
