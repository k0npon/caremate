# CareMate v0.3 — Google Scanned Objects scene patch

Integrates TWO already-previewed phototextured scanned objects into the existing
`caremate_furnished_scene.xml`, without touching any existing scene or third-party checkout:

- `reading_book`: Eat to Live book (patient's reading material).
- `personal_care_box`: Marc Anthony personal-care product carton.

The three synthetic tabletop boxes are removed from the generated **new scene only**.
The elderly resident, TIAGo++, studyTable and delivery table remain unchanged.

## Installation

Download and extract the patch ZIP **from within** the existing project folder:

```bash
mamba activate caremate
cd ~/Desktop/caremate
unzip ~/Downloads/caremate_scanned_patch_v0_3.zip
```

Your earlier `git sparse-checkout add` already downloaded both objects; no further download is needed.
If the furnished scene is missing, first run `python -m caremate.build_furnished_scene`.

## Run

```bash
python -m caremate.build_scanned_scene
python -m tests.test_scanned
python -m caremate.run_scanned --snapshot
python -m caremate.run_scanned
```

Use **Q** or **Esc** (in the camera window) to close cleanly; avoid Ctrl+C while both windows run.

The generated file is:

`models/pal_tiago_dual/caremate_scanned_scene.xml`

Old runnable scenes remain available via `python -m caremate.run_sim` and
`python -m caremate.run_furnished`.

## Optional detailed collisions

For future manipulation experiments, full GSO convex collision meshes can be enabled:

```bash
python -m caremate.build_scanned_scene --collision-mode full
```

The default is **proxy**, which uses an accurate visual OBJ + one low-cost
box collider per object. Objects have a free joint, an approximate mass, and
uniform-box inertia. The proxy collision model is not suitable for evaluating
detailed grasp-contact accuracy. Mass, friction, camera calibration and stable
TIAGo++ actuation still require experimental validation.

## Notes

- This patch does NOT implement autonomous manipulation, ALIKED/LightGlue, or AWS.
- GSO textures are copied into the project with source files; nothing is symlinked.
- GSO 3D assets are CC BY 4.0 (attribution required); generated MJCF is MIT.
- The original source model and all original CareMate scenes are preserved.
- If the scene compiles but objects float/intersect the tabletop, share the
  MuJoCo screenshot plus build output (sizes printed). Source OBJ orientation
  and automatic MuJoCo mesh preprocessing may require fine-tuning in individual cases.
