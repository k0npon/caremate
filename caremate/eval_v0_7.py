"""CareMate v0.7 evaluation: active vs fixed vs random viewpoints on randomised tables.

Each episode:
  1. randomise the table: every object gets a new (x, y) on the tabletop and a
     yaw that roughly faces the robot (+-jitter, sometimes the back face);
     present-target episodes are open / occluded / edge (see sample_layout);
  2. pick a target; in ~25% of episodes the target is removed (target absent);
  3. run the SAME episode with each policy and the SAME observation budget.

Ground truth (MuJoCo segmentation, object poses) is used here ONLY to score
outcomes after the agent has finished.  The agent itself never sees it.

Scoring
  target present: SUCCESS       localised and the support box centre lies on the target's GT mask
                  WRONG_OBJECT  localised but the box is on something else  (false positive)
                  MISS          abstained
  target absent:  CORRECT_ABSTAIN / FALSE_POSITIVE
Episodes are also tagged by how hard the START view is for the target
(visible / partial / hidden), measured from GT, so the report can
show where active viewpoint selection matters.

Run (OMEN, real ALIKED):
  python -m caremate.eval_v0_7 --episodes 60 --seed 1
Outputs outputs/eval_v0_7/<stamp>/{episodes.jsonl, summary.json, summary.md}
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import mujoco
import numpy as np

from caremate.agent_loop_v0_7 import (make_backend, CANDIDATES, POLICIES, ROOT, START, MuJoCoHeadCamera,
                                      SiftStandIn, run_agent)
from caremate.perception_v0_7 import PerceptionBackend

TARGETS = ("reading_book", "crunch_caramel", "crunch_peanut", "cereal_box")
CLUTTER = ("personal_care_box", "blue_mug", "yellow_mug")
RADIUS = {"cereal_box": 0.12, "reading_book": 0.10, "crunch_caramel": 0.09,
          "crunch_peanut": 0.09, "personal_care_box": 0.05, "blue_mug": 0.06, "yellow_mug": 0.06}
X_RANGE, Y_RANGE = (0.92, 1.85), (-0.72, -0.18)
FAR_AWAY = (8.0, 8.0, 0.5)


# ---------------------------------------------------------------- scene randomisation
def facing_yaw(x, y):
    """Yaw that turns a GSO front face (normal -y at yaw 0) toward the robot base."""
    return math.atan2(-y, -x) + math.pi / 2


def _free(x, y, name, placed):
    return all(math.hypot(x - px, y - py) > RADIUS[name] + RADIUS[n] + 0.02
               for n, (px, py, _) in placed.items())


def _yaw(rng, name, x, y, back_prob, jitter):
    yaw = facing_yaw(x, y) + rng.uniform(-jitter, jitter)
    if name in TARGETS and rng.random() < back_prob:
        yaw += math.pi
    return yaw


def sample_layout(rng: random.Random, target: str, present: bool, mode: str,
                  back_prob=0.2, jitter=0.5):
    """Random tabletop layout.

    mode  open      everything uniformly random
          occluded  target far back (x 1.55-1.85) with a tall object 0.18-0.30 m in front
                    of it on the robot's line of sight (partial occlusion)
          edge      target on the near-right corner, outside/at the edge of the START view
    """
    placed = {}
    if present and mode == "occluded":
        for _ in range(500):
            tx, ty = rng.uniform(1.55, 1.85), rng.uniform(-0.65, -0.25)
            d = rng.uniform(0.18, 0.30)
            r = math.hypot(tx, ty)
            ox, oy = tx - d * tx / r, ty - d * ty / r + rng.uniform(-0.04, 0.04)
            occ = rng.choice([n for n in ("cereal_box", "reading_book", "crunch_caramel", "crunch_peanut")
                              if n != target])
            if X_RANGE[0] <= ox <= X_RANGE[1] and Y_RANGE[0] <= oy <= Y_RANGE[1]:
                placed[target] = (tx, ty, _yaw(rng, target, tx, ty, back_prob, jitter))
                placed[occ] = (ox, oy, _yaw(rng, occ, ox, oy, 0.0, jitter))
                break
    elif present and mode == "edge":
        tx, ty = rng.uniform(0.92, 1.12), rng.uniform(-0.74, -0.62)
        placed[target] = (tx, ty, _yaw(rng, target, tx, ty, back_prob, jitter))
    for name in list(TARGETS) + list(CLUTTER):
        if name in placed or (name == target and not present):
            continue
        for _ in range(500):
            x, y = rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE)
            if _free(x, y, name, placed):
                break
        else:
            raise RuntimeError("could not place objects; table too crowded")
        placed[name] = (x, y, _yaw(rng, name, x, y, back_prob, jitter))
    return placed


def apply_layout(model, data, layout: dict):
    for name in TARGETS + CLUTTER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
        if jid < 0:
            continue
        a = int(model.jnt_qposadr[jid])
        z = float(model.qpos0[a + 2])
        if name in layout:
            x, y, yaw = layout[name]
            data.qpos[a:a + 3] = (x, y, z)
            data.qpos[a + 3:a + 7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        else:
            data.qpos[a:a + 3] = FAR_AWAY
            data.qpos[a + 3:a + 7] = (1, 0, 0, 0)
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------- ground truth (scoring only)
class GroundTruth:
    def __init__(self, camera: MuJoCoHeadCamera):
        from caremate.viewpoint_sweep import geom_owner_map
        self.cam = camera
        self.owner = geom_owner_map(camera.model, TARGETS + CLUTTER)
        self.seg = mujoco.Renderer(camera.model, height=camera.H, width=camera.W)
        self.seg.enable_segmentation_rendering()

    def mask(self, view, name) -> np.ndarray:
        self.cam._set(view)
        for _ in range(2):   # see MuJoCoHeadCamera.capture: first frame after a move can lag
            self.seg.update_scene(self.cam.data, camera="head_rgb")
            s = self.seg.render()
        gid, typ = s[..., 0], s[..., 1]
        ok = (typ == int(mujoco.mjtObj.mjOBJ_GEOM)) & (gid >= 0)
        own = np.full(gid.shape, -1)
        own[ok] = self.owner[gid[ok]]
        k = (TARGETS + CLUTTER).index(name)
        return own == k

    def difficulty(self, view, name, layout) -> tuple[str, float]:
        """visible / partial / hidden from the visible fraction of the target.

        Visible fraction = target pixels in the real scene / target pixels
        when every other object is removed (occlusion), and frame clipping.
        """
        m = self.mask(view, name)
        apply_layout(self.cam.model, self.cam.data, {name: layout[name]})
        alone = self.mask(view, name)
        apply_layout(self.cam.model, self.cam.data, layout)
        n, n0 = int(m.sum()), int(alone.sum())
        frac = n / max(n0, 1)
        if n < 300 or frac < 0.25:
            return "hidden", frac
        ys, xs = np.nonzero(m)
        H, W = m.shape
        edge = xs.min() <= 2 or ys.min() <= 2 or xs.max() >= W - 3 or ys.max() >= H - 3
        return ("partial" if edge or frac < 0.7 else "visible"), frac

    def close(self):
        self.seg.close()


def score(result: dict, gt: GroundTruth, target: str, present: bool) -> str:
    localised = result["outcome"] == "TARGET_LOCALISED_UNCALIBRATED"
    if not present:
        return "FALSE_POSITIVE" if localised else "CORRECT_ABSTAIN"
    if not localised:
        return "MISS"
    last = result["trace"][-1]
    bb = last["perception"]["support_bbox_xyxy"]
    from caremate.agent_loop_v0_7 import ViewPose
    v = ViewPose(**last["view"])
    # Which object does the localised support box actually cover?  (A single
    # centre pixel is unreliable when the target is partly occluded: the box
    # centre can land on the occluder although the matches are on the target.)
    x0, y0, x1, y1 = [int(round(c)) for c in bb]
    H, W = gt.cam.H, gt.cam.W
    x0, x1 = max(0, x0), min(W, x1 + 1); y0, y1 = max(0, y0), min(H, y1 + 1)
    area = max(1, (x1 - x0) * (y1 - y0))
    counts = {n: int(gt.mask(v, n)[y0:y1, x0:x1].sum()) for n in TARGETS + CLUTTER}
    tgt = counts.pop(target)
    ok = tgt > 0 and (tgt >= max(counts.values()) or tgt >= 0.25 * area)
    return "SUCCESS" if ok else "WRONG_OBJECT"


# ---------------------------------------------------------------- summary
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_table(rows: list[dict]) -> list[str]:
    """Policy vs policy on the SAME target-present episodes."""
    succ = defaultdict(dict)
    for r in rows:
        if r["present"]:
            succ[r["policy"]][r["episode"]] = r["score"] == "SUCCESS"
    pols = list(succ)
    if "active" not in succ:
        return []
    out = ["", "Paired comparison vs active (target-present episodes; exact McNemar test):", "",
           "| comparison | active only succeeded | other only succeeded | p-value |", "|---|---|---|---|"]
    for o in pols:
        if o == "active":
            continue
        eps = succ["active"].keys() & succ[o].keys()
        b = sum(succ["active"][e] and not succ[o][e] for e in eps)
        c = sum(succ[o][e] and not succ["active"][e] for e in eps)
        out.append(f"| active vs {o} | {b} | {c} | {mcnemar_exact(b, c):.3f} |")
    return out


def summarise(rows: list[dict]) -> tuple[dict, str]:
    by = defaultdict(list)
    for r in rows:
        by[r["policy"]].append(r)
    summary = {}
    lines = ["| policy | present: success | wrong object | miss | absent: correct abstain | false positive | mean obs | mean base travel (m) | mean s/episode |",
             "|---|---|---|---|---|---|---|---|---|"]
    for pol, rs in by.items():
        pres = [r for r in rs if r["present"]]
        absn = [r for r in rs if not r["present"]]
        c = Counter(r["score"] for r in rs)
        pct = lambda k, d: f"{c[k]}/{len(d)} ({100 * c[k] / max(1, len(d)):.0f}%)"
        summary[pol] = {
            "n": len(rs), "scores": dict(c),
            "success_rate_present": c["SUCCESS"] / max(1, len(pres)),
            "false_positive_rate": (c["WRONG_OBJECT"] + c["FALSE_POSITIVE"]) / max(1, len(rs)),
            "mean_observations": float(np.mean([r["observations"] for r in rs])),
            "by_start_difficulty": {},
        }
        for d in ("visible", "partial", "hidden"):
            sub = [r for r in pres if r["start_difficulty"] == d]
            if sub:
                summary[pol]["by_start_difficulty"][d] = {
                    "n": len(sub), "success": sum(r["score"] == "SUCCESS" for r in sub) / len(sub)}
        lines.append(f"| {pol} | {pct('SUCCESS', pres)} | {pct('WRONG_OBJECT', pres)} | {pct('MISS', pres)} | "
                     f"{pct('CORRECT_ABSTAIN', absn)} | {pct('FALSE_POSITIVE', absn)} | "
                     f"{summary[pol]['mean_observations']:.2f} | "
                     f"{np.mean([r.get('base_travel_m', 0.0) for r in rs]):.2f} | "
                     f"{np.mean([r['seconds'] for r in rs]):.1f} |")
    lines += ["", "Success on target-present episodes by START-view difficulty:", "",
              "| policy | visible | partial (occluded/clipped) | hidden |", "|---|---|---|---|"]
    for pol in by:
        cells = []
        for d in ("visible", "partial", "hidden"):
            x = summary[pol]["by_start_difficulty"].get(d)
            cells.append(f"{100 * x['success']:.0f}% (n={x['n']})" if x else "-")
        lines.append(f"| {pol} | " + " | ".join(cells) + " |")
    lines += ["", "Success on target-present episodes by ORACLE visibility (best reachable view, GT):", "",
              "| policy | visible somewhere | only partial anywhere | hidden from every view |", "|---|---|---|---|"]
    for pol, rs in by.items():
        cells = []
        for d in ("visible", "partial", "hidden"):
            sub = [r for r in rs if r["present"] and r.get("oracle_best_view") == d]
            cells.append(f"{100 * sum(r['score'] == 'SUCCESS' for r in sub) / len(sub):.0f}% (n={len(sub)})" if sub else "-")
        lines.append(f"| {pol} | " + " | ".join(cells) + " |")
    lines += ["", "Success on target-present episodes by episode type:", "",
              "| policy | open | occluded | edge |", "|---|---|---|---|"]
    for pol, rs in by.items():
        cells = []
        for m in ("open", "occluded", "edge"):
            sub = [r for r in rs if r.get("mode") == m]
            cells.append(f"{100 * sum(r['score'] == 'SUCCESS' for r in sub) / len(sub):.0f}% (n={len(sub)})" if sub else "-")
        lines.append(f"| {pol} | " + " | ".join(cells) + " |")
    lines += paired_table(rows)
    return summary, "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", type=Path, default=ROOT / "models/pal_tiago_dual/caremate_room_v0_7.xml")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--budget", type=int, default=3)
    p.add_argument("--absent-prob", type=float, default=0.25)
    p.add_argument("--mode-weights", type=float, nargs=3, default=[0.4, 0.4, 0.2],
                   metavar=("OPEN", "OCCLUDED", "EDGE"), help="episode mix when the target is present")
    p.add_argument("--policies", nargs="+", default=["active", "tour", "random", "fixed"],
                   choices=list(POLICIES))
    p.add_argument("--aliked", type=Path, default=ROOT / "models/vision/aliked-n16rot-top1k-640.onnx")
    p.add_argument("--refs", type=Path, default=ROOT / "references/v0_7")
    p.add_argument("--save-images", type=int, default=5, help="save step images for the first N episodes")
    p.add_argument("--standin-sift", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--backend-url", default=None, help="use the AWS perception service")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    out = a.out or ROOT / "outputs/eval_v0_7" / f"{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    backend = make_backend(a)
    print(f"{backend.method}; warm-up {backend.warmup():.1f}s")
    camera = MuJoCoHeadCamera(a.scene)
    gt = GroundTruth(camera)
    rng = random.Random(a.seed)
    rows = []
    t_all = time.perf_counter()
    try:
        with (out / "episodes.jsonl").open("w") as fh:
            for ep in range(a.episodes):
                target = rng.choice(TARGETS)
                present = rng.random() >= a.absent_prob
                mode = rng.choices(["open", "occluded", "edge"], weights=a.mode_weights)[0] if present else "absent"
                layout = sample_layout(rng, target, present, mode)
                apply_layout(camera.model, camera.data, layout)
                diff, vis_frac = gt.difficulty(START, target, layout) if present else ("absent", 0.0)
                # Oracle: best GT visibility over every reachable view (upper bound for any policy).
                oracle = "absent"
                if present:
                    ds = [gt.difficulty(v, target, layout)[0] for v in CANDIDATES]
                    oracle = "visible" if "visible" in ds else "partial" if "partial" in ds else "hidden"
                start_ids = set()
                for pol in a.policies:
                    apply_layout(camera.model, camera.data, layout)
                    t0 = time.perf_counter()
                    res = run_agent(target, backend, camera, POLICIES[pol](a.seed * 1000 + ep),
                                    a.budget, out / f"ep{ep:03d}_{pol}", save_images=ep < a.save_images,
                                    scene=f"random_layout_ep{ep}")
                    sc = score(res, gt, target, present)
                    start_ids.add(res["trace"][0]["image_id"])
                    if len(start_ids) > 1:
                        print(f"WARNING ep {ep}: start image differs between policies "
                              f"(episodes are not paired!) {start_ids}")
                    row = {"episode": ep, "policy": pol, "target": target, "present": present, "mode": mode,
                           "start_difficulty": diff, "start_visible_fraction": round(vis_frac, 3), "oracle_best_view": oracle, "outcome": res["outcome"], "score": sc,
                           "observations": res["observations"], "base_travel_m": res.get("base_travel_m", 0.0),
                           "seconds": round(time.perf_counter() - t0, 2),
                           "views": [s["view"] for s in res["trace"]],
                           "statuses": [s["perception"]["status"] for s in res["trace"]],
                           "layout": {k: [round(v, 3) for v in xyz] for k, xyz in layout.items()}}
                    rows.append(row)
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    print(f"[ep {ep:03d}] {pol:6s} target={target:14s} present={present!s:5s} "
                          f"mode={mode:8s} start={diff:8s} oracle={oracle:8s} -> {res['outcome']:32s} {sc}")
    finally:
        gt.close(); camera.close(); backend.close()
    summary, table = summarise(rows)
    meta = {"episodes": a.episodes, "seed": a.seed, "budget": a.budget, "backend": backend.method,
            "absent_prob": a.absent_prob, "total_seconds": round(time.perf_counter() - t_all, 1),
            "note": "GT used only for scoring. Small-sample simulation result; thresholds uncalibrated."}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "policies": summary}, indent=2))
    (out / "summary.md").write_text(f"# CareMate v0.7 evaluation\n\n{json.dumps(meta, indent=2)}\n\n{table}\n")
    print("\n" + table)
    print(f"\nSaved {out}/summary.md, summary.json, episodes.jsonl")


if __name__ == "__main__":
    main()
