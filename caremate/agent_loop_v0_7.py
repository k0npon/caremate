"""CareMate v0.7: perception-driven active-viewpoint agent loop.

Loop:  choose view -> move base station / torso / head (kinematic) -> render
       FRESH RGB (+ head depth) -> OpenCV 5 perception -> decision -> repeat.

Runtime inputs (all legitimate for a real robot):
  * head RGB image                      -> identity evidence (OpenCV 5 only)
  * head depth image (TIAGo++ RGB-D)    -> occupancy map for next-best-view only
  * camera pose from forward kinematics, static furniture map, base stations
  * the perception result of THIS image (image_id checked for staleness)
Never used: MuJoCo segmentation, object bodies/positions, object IDs.

Policies (same observation budget):
  active       evidence-driven next-best-view over base stations + head poses
               (look closer at tentative evidence; else maximise expected newly
               observed workspace from the depth occupancy map, minus travel)
  tour         fixed tour of the three base stations (ignores evidence)
  random       random reachable views
  fixed        v0.6 head-only preset (base never moves)
  active_head / random_head   ablations with the base fixed

Stop rules:
  candidate, not touching the frame border, no unresolved look-alike
      -> TARGET_LOCALISED_UNCALIBRATED   (never 'verified', never grasp-authorised)
  budget exhausted -> ABSTAIN_<reason>
Base motion is kinematic placement at pre-surveyed stations, not navigation.

Run from ~/Desktop/caremate:
  python -m caremate.agent_loop_v0_7 --target crunch_peanut --policy active
Headless: MUJOCO_GL=egl.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from caremate.nbv_v0_7 import STATIONS, OccupancyGrid, station_distance
from caremate.perception_v0_7 import PerceptionBackend, PerceptionResult
from caremate.workspace_v0_7 import WORKSPACES, project_points, workspace_polygon

ROOT = Path(__file__).resolve().parents[1]
TABLE_Z = WORKSPACES["main_table"]["z"][0]


@dataclass(frozen=True)
class ViewPose:
    torso: float
    pan: float
    tilt: float
    base: str = "front"          # base station (nbv_v0_7.STATIONS)

    @property
    def name(self) -> str:
        return f"{self.base}_t{self.torso:.3f}_p{self.pan:+.2f}_k{self.tilt:+.2f}"


HEAD_POSES = tuple(itertools.product((0.0, 0.175, 0.35), (-0.30, 0.0, 0.30), (-0.40, -0.60)))
# Head-only candidates (base fixed at the start station) and all candidates.
HEAD_CANDIDATES = tuple(ViewPose(t, p, k) for t, p, k in HEAD_POSES)
CANDIDATES = tuple(ViewPose(t, p, k, b) for b in STATIONS for t, p, k in HEAD_POSES)
START = ViewPose(0.0, 0.0, -0.40)
FIXED_SEQUENCE = (ViewPose(0.0, 0.0, -0.4), ViewPose(0.175, 0.0, -0.6), ViewPose(0.35, 0.0, -0.6))
TOUR_SEQUENCE = (START, ViewPose(0.175, 0.0, -0.4, "front_right"), ViewPose(0.175, 0.0, -0.4, "front_left"))


# ---------------------------------------------------------------- driver
class MuJoCoHeadCamera:
    """Moves torso/head kinematically and renders head_rgb. Reads NO object state."""

    def __init__(self, scene: Path, width=1280, height=960):
        import mujoco
        from caremate.run_sim import initialise
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), width)
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), height)
        self.data = mujoco.MjData(self.model)
        initialise(self.model, self.data)
        # Separate state for "what would the camera pose be if..." queries, so
        # planning never disturbs the displayed/rendered robot state.
        self.plan_data = mujoco.MjData(self.model)
        self.W, self.H = width, height
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        # Head RGB-D: depth at quarter resolution, used only for next-best-view.
        self.depth_renderer = mujoco.Renderer(self.model, height=height // 4, width=width // 4)
        self.depth_renderer.enable_depth_rendering()
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "reference")
        self._base_adr = int(self.model.jnt_qposadr[jid]) if jid >= 0 else None
        self.last_depth = None

    def set_robot(self, x, y, yaw, torso, pan, tilt, data=None):
        from caremate.run_sim import hold_target
        data = self.data if data is None else data
        if self._base_adr is not None:
            a = self._base_adr
            data.qpos[a:a + 2] = (x, y)
            data.qpos[a + 3:a + 7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        for joint, value in (("torso_lift_joint", torso), ("head_1_joint", pan),
                             ("head_2_joint", tilt)):
            hold_target(self.model, data, joint, value)
        self.mj.mj_forward(self.model, data)

    def _set(self, v: ViewPose, data=None):
        x, y, yaw = STATIONS[v.base]
        self.set_robot(x, y, yaw, v.torso, v.pan, v.tilt, data)

    def predict_pose(self, v: ViewPose) -> dict:
        """Camera pose for a hypothetical view: proprioception/kinematics only."""
        from caremate.workspace_v0_7 import camera_pose_from_mujoco
        self.plan_data.qpos[:] = self.data.qpos
        self._set(v, self.plan_data)
        return camera_pose_from_mujoco(self.model, self.plan_data)

    def capture(self, v: ViewPose) -> tuple[np.ndarray, dict]:
        from caremate.workspace_v0_7 import camera_pose_from_mujoco
        self._set(v)
        pose = camera_pose_from_mujoco(self.model, self.data)
        # Render twice: after a large base move MuJoCo's first frame still uses
        # stale shadow data (one-frame lag), so the image depends on where the
        # robot was before.  The second render is deterministic.
        for _ in range(2):
            self.renderer.update_scene(self.data, camera="head_rgb")
            rgb = self.renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for _ in range(2):
            self.depth_renderer.update_scene(self.data, camera="head_rgb")
            depth = self.depth_renderer.render()
        self.last_depth = depth.copy()
        return bgr, pose

    def close(self):
        self.renderer.close()
        self.depth_renderer.close()


# ---------------------------------------------------------------- geometry helpers
def backproject_to_table(uv, pose, width, height, z=TABLE_Z) -> np.ndarray | None:
    R = np.asarray(pose["mat"], float).reshape(3, 3)
    f = 0.5 * height / math.tan(math.radians(pose["fovy_deg"]) / 2)
    d_cam = np.array([(uv[0] - width / 2) / f, -(uv[1] - height / 2) / f, -1.0])
    d = R @ d_cam
    o = np.asarray(pose["pos"], float)
    if abs(d[2]) < 1e-6:
        return None
    s = (z - o[2]) / d[2]
    return o + s * d if s > 0 else None


def target_base_estimate(res: PerceptionResult, pose, width, height) -> np.ndarray | None:
    """3D point where the target meets the table, from THIS observation.

    Uses the projected reference polygon's two lowest corners (object base),
    falling back to the bottom-centre of the support box.
    """
    if res.status not in ("candidate", "ambiguous") or not res.support_bbox_xyxy:
        return None
    if res.projected_polygon:
        poly = np.asarray(res.projected_polygon, float)
        base = poly[np.argsort(poly[:, 1])[-2:]].mean(0)
    else:
        x0, y0, x1, y1 = res.support_bbox_xyxy
        base = np.array([(x0 + x1) / 2, y1], float)
    base[0] = np.clip(base[0], 0, width - 1)
    base[1] = np.clip(base[1], 0, height - 1)
    return backproject_to_table(base, pose, width, height)


# ---------------------------------------------------------------- policies
TRAVEL_COST = 0.10   # exploration-gain fraction a metre of base travel must buy
NEW_DIRECTION_BONUS = 0.20   # identity needs the label seen face-on: value an unvisited viewing direction


class Policy:
    name = "base"
    candidates = CANDIDATES

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def next_view(self, history, camera, grid=None) -> tuple[ViewPose | None, str]:
        raise NotImplementedError


class FixedPolicy(Policy):
    name = "fixed"

    def next_view(self, history, camera, grid=None):
        i = len(history)
        if i < len(FIXED_SEQUENCE):
            return FIXED_SEQUENCE[i], "preset head sequence, base fixed (v0.6)"
        return None, "preset sequence exhausted"


class TourPolicy(Policy):
    name = "tour"

    def next_view(self, history, camera, grid=None):
        i = len(history)
        if i < len(TOUR_SEQUENCE):
            return TOUR_SEQUENCE[i], "preset tour of base stations (ignores evidence)"
        return None, "tour exhausted"


class RandomPolicy(Policy):
    name = "random"

    def next_view(self, history, camera, grid=None):
        if not history:
            return START, "start view"
        seen = {h["view"] for h in history}
        options = [v for v in self.candidates if v not in seen]
        return (self.rng.choice(options), "random unvisited view") if options else (None, "no views left")


class ActivePolicy(Policy):
    """Evidence-driven next-best-view.

    1. Tentative target evidence (ambiguous / too small / clipped): go to the
       reachable view that sees the estimated target closest and centred.
    2. Otherwise: maximise expected newly-observed workspace volume from the
       depth occupancy map (sees behind occluders), minus travel cost.
    """
    name = "active"

    def next_view(self, history, camera, grid=None):
        if not history:
            return START, "start view"
        seen = {h["view"] for h in history}
        options = [v for v in self.candidates if v not in seen]
        if not options:
            return None, "no views left"
        W, H = camera.W, camera.H
        here = history[-1]["view"].base
        est = next((h["target_base"] for h in reversed(history) if h["target_base"] is not None), None)
        if est is not None:
            p = np.asarray(est) + np.array([0, 0, 0.08])   # aim at mid-height of the object
            best, best_s, why = None, -1e9, ""
            for v in options:
                pose = camera.predict_pose(v)
                uv, front = project_points(p[None], pose, W, H)
                if not front[0]:
                    continue
                u, vv = uv[0]
                off = math.hypot((u - W / 2) / (W / 2), (vv - H / 2) / (H / 2))
                margin = min(u, W - u, vv, H - vv) / min(W, H)
                if margin < 0.12:          # would clip again
                    continue
                dist = float(np.linalg.norm(p - np.asarray(pose["pos"])))
                travel = station_distance(here, v.base)
                # Re-observing tentative evidence: prefer re-aiming from the same
                # viewing direction (the label was readable from here) over moving.
                s = -dist - 0.35 * off - 1.0 * travel
                if s > best_s:
                    best, best_s = v, s
                    why = (f"target estimated at {np.round(est, 3).tolist()} from last tentative evidence; "
                           f"view keeps it in frame (offset {off:.2f}) at {dist:.2f} m, travel {travel:.2f} m")
            if best is not None:
                return best, why
        if grid is None:
            return self.rng.choice(options), "no occupancy map; random"
        visited = {h["view"].base for h in history}
        best, best_s, best_g, best_t = None, -1e9, 0.0, 0.0
        for v in options:
            g = grid.expected_gain(camera.predict_pose(v))
            t = station_distance(here, v.base)
            # Volume coverage alone is not enough: a box seen edge-on is "observed"
            # but its label is unreadable, so a new viewing direction has value.
            s = g - TRAVEL_COST * t + (NEW_DIRECTION_BONUS if v.base not in visited else 0.0)
            if s > best_s:
                best, best_s, best_g, best_t = v, s, g, t
        rejected = any(h["result"].rejected_instances for h in history)
        why = (f"next-best-view: expected to observe {best_g:.0%} of the still-unknown workspace "
               f"(depth occupancy map), travel {best_t:.2f} m"
               + (", new viewing direction" if best.base not in visited else ""))
        if rejected:
            why += "; earlier matches were a look-alike, search elsewhere"
        return best, why


class ActiveHeadOnlyPolicy(ActivePolicy):
    """Ablation: same NBV logic, base fixed at the start station."""
    name = "active_head"
    candidates = HEAD_CANDIDATES


class RandomHeadOnlyPolicy(RandomPolicy):
    name = "random_head"
    candidates = HEAD_CANDIDATES


POLICIES = {"active": ActivePolicy, "tour": TourPolicy, "random": RandomPolicy,
            "fixed": FixedPolicy, "active_head": ActiveHeadOnlyPolicy, "random_head": RandomHeadOnlyPolicy}


# ---------------------------------------------------------------- agent
def decide(res: PerceptionResult, is_last: bool) -> tuple[str, str]:
    lookalike = {n: c for n, c in res.competitors.items() if c.get("resolution") == "unresolved"}
    rejected = sorted({n for r in res.rejected_instances for n in r["is_actually"]} |
                      {n for n, c in res.competitors.items() if c.get("resolution") == "region_is_competitor"})
    if res.status == "candidate" and not res.touches_border and not lookalike:
        return "STOP_LOCALISED", (f"{res.n_inliers} geometric inliers, support fully inside frame, "
                                  "no unresolved look-alike")
    if res.status == "candidate" and res.touches_border:
        r = "target support touches frame border (likely clipped); re-centre"
    elif res.status == "ambiguous" and res.notes:
        r = "; ".join(res.notes)
    elif res.status == "ambiguous":
        r = (f"look-alike(s) {list(lookalike) or list(res.competitors)} explain the same region; "
             "need a closer/clearer view of the distinguishing area")
    elif res.status == "not_found" and rejected:
        r = f"matched region is actually {rejected}; keep searching elsewhere"
    elif res.status == "error":
        r = f"perception error: {res.error}; do not assume recognition"
    else:
        r = "no geometrically consistent evidence for target in this view"
    return ("ABSTAIN" if is_last else "REOBSERVE"), r


def run_agent(target, backend: PerceptionBackend, camera: MuJoCoHeadCamera, policy: Policy,
              budget: int, out: Path, save_images=True, scene: str | None = None,
              on_step=None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    history, trace = [], []
    seen_ids = set()
    grid = OccupancyGrid()
    travel_m = 0.0
    outcome, t_start = None, time.perf_counter()
    for step in range(budget):
        t_plan = time.perf_counter()
        view, why_view = policy.next_view(history, camera, grid)
        plan_ms = 1000 * (time.perf_counter() - t_plan)
        if view is None:
            outcome = "ABSTAIN_NO_VIEWS"
            break
        if history:
            travel_m += station_distance(history[-1]["view"].base, view.base)
        t0 = time.perf_counter()
        bgr, pose = camera.capture(view)
        t_render = time.perf_counter() - t0
        grid_info = grid.integrate(camera.last_depth, pose)
        roi = workspace_polygon(pose, camera.W, camera.H)
        res = backend.perceive(bgr, target, camera_pose=asdict(view) | {"fk": pose}, roi_polygon=roi)
        if res.image_id in seen_ids:     # stale evidence guard
            res.status, res.error = "error", "stale image: identical pixels to an earlier observation"
        seen_ids.add(res.image_id)
        base = target_base_estimate(res, pose, camera.W, camera.H)
        decision, reason = decide(res, step == budget - 1)
        img_path = None
        if save_images:
            from caremate.perception_v0_7 import draw
            img_path = out / f"step{step}_{view.name}.png"
            cv2.imwrite(str(img_path), draw(bgr, res, None, roi))
        if on_step is not None:
            on_step(step=step, view=view, why_view=why_view, bgr=bgr, res=res, roi=roi,
                    decision=decision, reason=reason)
        history.append({"view": view, "pose": pose, "result": res,
                        "target_base": None if base is None else base.tolist()})
        trace.append({
            "step": step, "view": asdict(view), "view_reason": why_view,
            "image": str(img_path) if img_path else None, "image_id": res.image_id,
            "render_ms": round(1000 * t_render, 1), "plan_ms": round(plan_ms, 1),
            "base_travel_m_total": round(travel_m, 3), "occupancy": grid_info,
            "perception": {k: v for k, v in res.to_json().items()
                           if k not in ("per_reference", "projected_polygon", "camera_pose")},
            "target_base_estimate_m": None if base is None else np.round(base, 3).tolist(),
            "decision": decision, "reason": reason,
        })
        print(f"step {step}: {view.name:22s} {res.status:9s} inl={res.n_inliers:3d} "
              f"border={res.touches_border!s:5s} -> {decision}: {reason}")
        if decision == "STOP_LOCALISED":
            outcome = "TARGET_LOCALISED_UNCALIBRATED"
            break
    if outcome is None:
        last = history[-1]["result"] if history else None
        outcome = "ABSTAIN_" + ({"ambiguous": "AMBIGUOUS", "not_found": "NOT_FOUND",
                                 "error": "PERCEPTION_ERROR"}.get(last.status, "BORDER")
                                if last else "NO_OBSERVATION")
    result = {
        "version": "v0.7", "scene": scene, "target": target, "policy": policy.name, "budget": budget,
        "outcome": outcome, "observations": len(trace), "base_travel_m": round(travel_m, 3),
        "grasp_authorised": False,
        "note": "Uncalibrated visual evidence; localisation is not identity verification.",
        "backend": backend.method,
        "total_seconds": round(time.perf_counter() - t_start, 2),
        "trace": trace,
    }
    (out / "trace.json").write_text(json.dumps(result, indent=2))
    print(f"FINAL: {outcome} after {len(trace)} observation(s); trace: {out / 'trace.json'}")
    return result


class SiftStandIn:
    """Test-only extractor so the loop can run without ONNX weights. NOT OpenCV 5 ALIKED."""

    def __new__(cls):
        from caremate.perception_v0_7 import AlikedExtractor

        class _S(AlikedExtractor):
            def __init__(s):
                s.size, s.min_score, s.pad_margin, s.name = 640, 0.0, 4, "SIFT-STANDIN-TEST-ONLY"
                s.net = cv2.SIFT_create(1000)
        return _S()


def make_backend(a):
    """Local OpenCV 5 backend, or the remote AWS service when --backend-url is given."""
    if getattr(a, "backend_url", None):
        from caremate.remote_backend import RemotePerceptionBackend
        return RemotePerceptionBackend(a.backend_url)
    ext = SiftStandIn() if getattr(a, "standin_sift", False) else None
    if ext is None and not Path(a.aliked).is_file():
        raise SystemExit(f"ALIKED model missing: {a.aliked}")
    return PerceptionBackend(a.aliked, a.refs, extractor=ext)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", type=Path, default=ROOT / "models/pal_tiago_dual/caremate_room_v0_7.xml")
    p.add_argument("--target", default="reading_book")
    p.add_argument("--policy", choices=list(POLICIES), default="active")
    p.add_argument("--budget", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--aliked", type=Path, default=ROOT / "models/vision/aliked-n16rot-top1k-640.onnx")
    p.add_argument("--refs", type=Path, default=ROOT / "references/v0_7")
    p.add_argument("--standin-sift", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--backend-url", default=None,
                   help="use the AWS perception service instead of local OpenCV (e.g. http://HOST:8000)")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    if not a.scene.is_file():
        p.error(f"Scene missing: {a.scene} (python -m caremate.build_room_v0_7)")
    backend = make_backend(a)
    print(f"{backend.method}; warm-up {backend.warmup():.1f}s")
    camera = MuJoCoHeadCamera(a.scene)
    layout = a.scene.stem.replace("caremate_room_v0_7", "").strip("_") or "default"
    out = a.out or ROOT / "outputs/agent_v0_7" / f"{datetime.now():%Y%m%d_%H%M%S}_{layout}_{a.target}_{a.policy}"
    try:
        run_agent(a.target, backend, camera, POLICIES[a.policy](a.seed), a.budget, out,
                  scene=a.scene.name)
    finally:
        camera.close()
        backend.close()


if __name__ == "__main__":
    main()
