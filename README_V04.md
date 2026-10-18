# CareMate v0.4 — Room layout repair

**Non-destructive** patch: Generates a separate `models/pal_tiago_dual/caremate_room_v0_4.xml` from your already-working `caremate_scanned_scene.xml`. Keeps TIAGo++, third-party sources, earlier XMLs, and original runtime modules intact.

## What's different

- Removes **both solid view-blocking walls** for an open-cutaway simulation space.
- Adds **blue-grey checker floor** and soft gradient skybox instead of black void.
- Replaces obstructive `studyTable` hutch with furniture_sim's accessible **simpleWoodTable**. Preserves wood texture from installed studyTable materials and closes unused cabinet scene geometry.
- Replaces squatting resident with a **static pose baked from the official MuJoCo humanoid's hip/knee joints**, facing forwards on the wheelchair. This is a demonstrative seated proxy; the final posture needs visual review, not biomechanics validation.
- Moves the bedside table closer to the wheelchair and improves it (four legs, wood material) and hides the bright green handover site. Handover site remains in the XML for later controllers.
- Places your textured **book** and **personal-care carton** on the accessible worktable. Adds the **two mugs** you downloaded if available, as background obstacles, not grasp targets (fragility). The medicines and cloth tasks remain excluded.
- If your v0.3 source uses the full 32-hull GSO collision meshes, normalises them back to lightweight *proxy* collisions **only in the new scene**, keeping photorealistic visual OBJ/texture intact.
- New viewer preview command with adjustable third-person azimuth and default frame. `--snapshot` saves to `outputs/caremate_v04_head_rgb.png` without overwriting old snapshots.

## Install

```bash
mamba activate caremate
cd ~/Desktop/caremate
unzip ~/Downloads/caremate_room_fix_v0_4.zip
python -m caremate.build_room_v0_4
python -m tests.test_room_v0_4
python -m caremate.run_room_v0_4 --snapshot
python -m caremate.run_room_v0_4
```

- Controls: **A/D** head pan, **W/S** head tilt, **C** screenshot, **Q/Esc** quit.
- If MuJoCo+Qt conflict, isolate the RGB window: `python -m caremate.run_room_v0_4 --no-viewer`.
- If the spectator view is from the wrong side, try `--azimuth 45` or `--azimuth 210`, or orbit the MuJoCo Viewer with the mouse. These options change only the *third-person* camera, not the robot's perception.
- Do **not** use `--physics` until robot controls and grasp contacts have been validated.

## Requirements

- Your working project at `~/Desktop/caremate` with `caremate/build_scene.py`, `caremate/build_furnished_scene.py`, `caremate/build_scanned_scene.py`, and `caremate/run_sim.py`.
- `models/pal_tiago_dual/caremate_scanned_scene.xml` already generated.
- `models/humanoid.xml` and `third_party/furniture_sim/simpleTable/simpleWoodTable_body.xml` present.
- Two previously downloaded Google Scanned Objects in `third_party/mujoco_scanned_objects/models/`.

## Verification

Code has been syntax-checked, and source XML elements were inspected. The patch has **not been compiled or rendered** in the assistant's environment because that runtime lacks MuJoCo and the ZBook's model assets. You must run the test and inspect the resulting snapshot/viewer image. The resident pose and tabletop locations may need refinement once the screenshot is available.

## Scope

This fixes only the **scene layout and presentation**. No autonomy, learned vision, grasping, or patient interaction is implemented in v0.4.
