"""Run the generated scanned-object CareMate scene with existing viewer.

    python -m caremate.run_scanned --snapshot
    python -m caremate.run_scanned
    python -m caremate.run_scanned --no-viewer
"""
from caremate import run_sim
from caremate.build_scanned_scene import SCANNED_XML

if __name__ == "__main__":
    if not SCANNED_XML.is_file():
        raise FileNotFoundError("Run: python -m caremate.build_scanned_scene")
    run_sim.SCENE_XML = SCANNED_XML
    run_sim.main()
