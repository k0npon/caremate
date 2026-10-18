"""Run the furnished CareMate scene, reusing the existing tested viewer logic.

Usage:
    python -m caremate.run_furnished --snapshot
    python -m caremate.run_furnished
"""
from __future__ import annotations

from caremate import run_sim
from caremate.build_furnished_scene import FURNISHED_XML

if __name__ == "__main__":
    if not FURNISHED_XML.is_file():
        raise FileNotFoundError(
            "Furnished scene missing. Run: python -m caremate.build_furnished_scene"
        )
    run_sim.SCENE_XML = FURNISHED_XML
    run_sim.main()
