"""CareMate v0.6: agent-loop scaffold for PARALLEL development.

This is a bounded, deterministic, goal-directed skeleton, NOT a finished vision
system. In --mode sim-mock it really moves the TIAGo++ head and torso
KINEMATICALLY and renders camera images, but the perception results are MOCK.

Run from ~/Desktop/caremate:
  python -m caremate.agent_loop_v0_6 --mode mock
  python -m caremate.agent_loop_v0_6 --mode sim-mock
  python -m caremate.agent_loop_v0_6 --mode mock \\
      --evidence outputs/perception_v0_5/repaired_bf/bf_results.json

`--evidence` treats existing bf_results.json as ONE uncalibrated observation
and will NEVER report VERIFIED from a tentative score. This is a backend
contract rehearsal, not calibrated recognition. No ground-truth segmentation,
3D pose, grasp or autonomous base navigation is used.

Later, replace evidence_provider(...) with an OpenCV 5 / AWS inference adapter.
Do not change the agent's observation-result contract.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

Status = Literal['candidate', 'verified', 'not_found', 'error']


@dataclass(frozen=True)
class ViewPose:
    name: str
    torso: float
    pan: float
    tilt: float


# Views selected provisionally from the Oct-18 viewpoint sweep, not optimized.
VIEWS = (
    ViewPose('initial', 0.0, 0.0, -0.4),
    ViewPose('mid_elevated', 0.175, 0.0, -0.6),
    ViewPose('high_elevated', 0.35, 0.0, -0.6),
)


@dataclass(frozen=True)
class PerceptionResult:
    target: str
    status: Status
    method: str
    n_matches: int = 0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    bbox_xyxy: list[int] | None = None
    latency_ms: float | None = None
    error: str | None = None
    calibrated: bool = False

    def __post_init__(self) -> None:
        if self.status == 'verified' and not self.calibrated and self.method != 'MOCK':
            raise ValueError('Real backends cannot say verified without a validated acceptance policy')


def evidence_provider(view_index: int, target: str, evidence_file: Path | None) -> PerceptionResult:
    if evidence_file is None:
        # SYNTHETIC observations ONLY, to test agent state transitions.
        seq: tuple[Status, ...] = ('candidate', 'candidate', 'verified')
        return PerceptionResult(
            target=target, status=seq[min(view_index, len(seq)-1)], method='MOCK',
            n_matches=[2, 13, 28][min(view_index, 2)],
            n_inliers=[0, 9, 24][min(view_index, 2)],
            calibrated=False,
        )
    if view_index > 0:
        # One static evidence file is valid for one observation only.
        # Reusing it at other camera poses would be data leakage / stale evidence.
        return PerceptionResult(
            target=target, status='error', method='REPLAY_STALE_EVIDENCE',
            error='No fresh perception result is available for this viewpoint',
        )
    d = json.loads(evidence_file.read_text())
    best = d.get('best_tentative')
    if not best:
        return PerceptionResult(target=target, status='not_found', method=d.get('method', 'replay'))
    # An uncalibrated tentative result is never upgraded to VERIFIED.
    return PerceptionResult(
        target=target, status='candidate', method=d.get('method', 'replay'),
        n_matches=int(best.get('n_matches', 0)),
        n_inliers=int(best.get('n_inliers', 0)),
        inlier_ratio=float(best.get('inlier_ratio', 0.0)),
        bbox_xyxy=best.get('support_bbox_xyxy'),
        latency_ms=1000 * float(best.get('elapsed_seconds', 0)),
    )


def run_agent(target: str, max_observations: int, driver, evidence_file: Path | None,
              out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    trace: list[dict] = []
    outcome = 'ABSTAIN_OBSERVATION_BUDGET'
    for i, view in enumerate(VIEWS[:max_observations]):
        # Driver only moves camera / captures RGB. It never decides identity.
        image_path = driver.capture(view, out / f'observation_{i:02d}.png')
        evidence = evidence_provider(i, target, evidence_file)
        if evidence.status == 'verified':
            decision, outcome = 'STOP_VERIFIED', 'VERIFIED_MOCK_ONLY' if evidence.method == 'MOCK' else 'VERIFIED'
        elif evidence.status == 'error':
            decision = 'REOBSERVE' if i + 1 < max_observations else 'ABSTAIN_PERCEPTION_ERROR'
            if i + 1 == max_observations:
                outcome = 'ABSTAIN_PERCEPTION_ERROR'
        else:
            decision = 'REOBSERVE' if i + 1 < max_observations else 'ABSTAIN_OBSERVATION_BUDGET'
        trace.append({
            'step': i, 'goal': target, 'view': asdict(view),
            'image': str(image_path) if image_path else None,
            'evidence': asdict(evidence), 'decision': decision,
            'decision_reason': (
                'Explicitly verified by evidence adapter' if evidence.status == 'verified' else
                'Perception unavailable or failed; do not assume successful recognition' if evidence.status == 'error' else
                'Observation budget exhausted; unverified object must not be selected' if i+1 >= max_observations else
                'Insufficient evidence; select next prevalidated candidate view'
            ),
        })
        if evidence.status == 'verified':
            break
    result = {
        'version': 'v0.6 scaffold', 'mode': driver.description,
        'warning': 'MOCK demo is not actual visual identification; no grasp permission is issued.',
        'target': target, 'outcome': outcome, 'observation_budget': max_observations,
        'observations': len(trace), 'total_elapsed_seconds': round(time.perf_counter()-start, 3),
        'trace': trace,
    }
    (out / 'trace.json').write_text(json.dumps(result, indent=2))
    print(f'FINAL: {outcome}; observations={len(trace)}; trace={out / "trace.json"}')
    return result


class NoSimulationDriver:
    description = 'MOCK_LOGIC_ONLY'

    @staticmethod
    def capture(view: ViewPose, destination: Path) -> None:
        print(f'Camera pose (not executed): {view.name}: torso={view.torso}, pan={view.pan}, tilt={view.tilt}')
        return None


class MuJoCoKinematicDriver:
    description = 'KINEMATIC_MUJOCO_CAMERA_WITH_MOCK_EVIDENCE'

    def __init__(self):
        import mujoco
        from caremate.build_room_v0_4 import SCENE_V04
        from caremate.run_sim import initialise
        if not SCENE_V04.is_file():
            raise FileNotFoundError('Scene missing; build v0.4 first')
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_V04))
        self.data = mujoco.MjData(self.model)
        initialise(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

    def capture(self, view: ViewPose, destination: Path) -> Path:
        import cv2
        from caremate.run_sim import hold_target
        for joint, value in (
            ('torso_lift_joint', view.torso),
            ('head_1_joint', view.pan),
            ('head_2_joint', view.tilt),
        ):
            hold_target(self.model, self.data, joint, value)
        self.mujoco.mj_forward(self.model, self.data)  # Kinematic, NOT wheel-driven motion
        self.renderer.update_scene(self.data, camera='head_rgb')
        rgb = self.renderer.render()
        if not cv2.imwrite(str(destination), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise IOError(f'Could not save {destination}')
        print(f'Rendered {view.name}: {destination}')
        return destination

    def close(self) -> None:
        self.renderer.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['mock', 'sim-mock'], default='mock')
    p.add_argument('--target', default='reading_book')
    p.add_argument('--budget', type=int, default=3, help='Maximum observations (1..3)')
    p.add_argument('--evidence', type=Path, default=None,
                   help='Replay one uncalibrated bf_results.json as a candidate; never verifies identity')
    p.add_argument('--out', type=Path, default=None)
    args = p.parse_args()
    if not 1 <= args.budget <= len(VIEWS):
        p.error(f'--budget must be 1..{len(VIEWS)}')
    if args.evidence is not None and not args.evidence.is_file():
        p.error(f'Evidence file missing: {args.evidence}')
    out = args.out or Path('outputs/agent_v0_6') / datetime.now().strftime('%Y%m%d_%H%M%S')
    driver = NoSimulationDriver() if args.mode == 'mock' else MuJoCoKinematicDriver()
    try:
        for entry in run_agent(args.target, args.budget, driver, args.evidence, out)['trace']:
            evidence = entry['evidence']
            print(f"  step {entry['step']}: {entry['view']['name']} => {evidence['status']} "
                  f"[{evidence['method']}], action={entry['decision']}")
    finally:
        if hasattr(driver, 'close'):
            driver.close()


if __name__ == '__main__':
    main()
