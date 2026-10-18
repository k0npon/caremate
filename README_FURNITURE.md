# CareMate studyTable integration (optional, non-destructive)

This patch imports the `studyTable` visual meshes, associated wood/metal textures,
and collision proxies from `vikashplus/furniture_sim` into the **existing** CareMate
scene. Drawers retain sliding joints. The new scene is written beside the
original, not over it.

## Prerequisites

- `~/Desktop/caremate/caremate/build_scene.py` and `run_sim.py` work.
- `~/Desktop/caremate/third_party/furniture_sim` was cloned successfully.
- All operations run in `mamba activate caremate`.

## Commands

```bash
cd ~/Desktop/caremate
python -m caremate.build_scene                # original; still works
python -m caremate.build_furnished_scene     # new furnished scene
python -m tests.test_furnished
python -m caremate.run_furnished --snapshot
python -m caremate.run_furnished
```

To run your original v0.1 without furniture, use:

```bash
python -m caremate.run_sim
```

`furniture_sim` is authored by Vikash Kumar and contributors, Apache-2.0.
Keep upstream notices and license when distributing derived assets.
The patch does NOT install additional dependencies; it reuses your MuJoCo 3.6.0.

This scene has not been run against your ZBook here. If you see a MuJoCo
compilation error, share the full traceback. Physics grasping is NOT implemented.
