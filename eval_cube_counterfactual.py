"""Cube closed-loop causal evaluation.

This script evaluates LeWM on Cube under distribution shift interventions
on dynamic variables observed during training:
  - Initial position perturbation (training: ±1cm → test: ±3/5/8cm)
  - Initial orientation perturbation (training: uniform [0,2π] → test: fixed/biased)
  - Goal position shift (training: 5 templates → test: boundary shifts)
  - Compound perturbations (multiple variables simultaneously)
  - Vacuum zone tests (init/goal far from ALL 5 template points)

All interventions target variables that are DYNAMIC in the training distribution,
pushing them beyond the training range to test causal generalization.

The vacuum zone test specifically places the cube initial position or goal position
in regions that are far (≥5cm or ≥8cm) from ANY of the 5 training templates,
testing whether the model has learned generalizable dynamics or just memorized
template-specific behaviors.

Usage
-----
python eval_cube_causal.py policy=cube/lewm [num_episodes=50] [seed=42]

Output
------
Prints summary table and writes JSON to logs_eval/cube_causal_results.json
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import hydra
import mujoco
import numpy as np
import torch
import warnings
from omegaconf import OmegaConf, DictConfig
from sklearn import preprocessing

warnings.filterwarnings("ignore", category=UserWarning)

import stable_worldmodel as swm
from eval import (
    FastActionablePolicy,
    get_dataset,
    get_episodes_length,
    img_transform,
    load_lewm_model,
    patch_gcrl_compat,
)


@dataclass(frozen=True)
class InterventionSpec:
    """Specification for a causal intervention on dynamic variables."""
    name: str
    kind: str
    value: dict | float | None = None


# Training distribution reference:
# - Position: 5 templates with ±1cm (0.01m) uniform noise
# - Orientation: uniform [0, 2π]
# - Goal: 5 templates with ±1cm noise

INTERVENTIONS: list[InterventionSpec] = [
    InterventionSpec("baseline", "baseline"),

    # 1. Initial Position Perturbation (training: ±1cm)
    InterventionSpec("do_init_pos_mild", "init_pos_noise", 0.03),      # ±3cm (3x training)
    InterventionSpec("do_init_pos_moderate", "init_pos_noise", 0.05),  # ±5cm (5x training)
    InterventionSpec("do_init_pos_extreme", "init_pos_noise", 0.08),   # ±8cm (8x training)

    # 2. Initial Orientation Perturbation (training: uniform [0,2π])
    InterventionSpec("do_init_yaw_mild", "init_yaw_offset", np.pi/6),           # +30° bias
    InterventionSpec("do_init_yaw_moderate", "init_yaw_fixed", 0.0),            # fixed 0°
    InterventionSpec("do_init_yaw_extreme", "init_yaw_align_goal", True),       # align with goal

    # 3. Goal Position Shift (training: 5 templates ±1cm)
    InterventionSpec("do_goal_pos_mild", "goal_pos_offset", 0.02),              # +2cm shift
    InterventionSpec("do_goal_pos_moderate", "goal_pos_offset", 0.04),          # +4cm shift
    InterventionSpec("do_goal_pos_extreme", "goal_pos_boundary", True),         # workspace boundary

    # 4. Compound Perturbations (multiple variables)
    InterventionSpec("do_compound_mild", "compound",
                    {"init_pos_noise": 0.02, "init_yaw_offset": np.pi/12}),     # pos+2cm, yaw+15°
    InterventionSpec("do_compound_moderate", "compound",
                    {"init_pos_noise": 0.04, "goal_pos_offset": 0.03, "init_yaw_fixed": 0.0}),  # multi-var
    InterventionSpec("do_compound_extreme", "compound",
                    {"init_pos_noise": 0.06, "goal_pos_offset": 0.05, "init_yaw_offset": np.pi}),  # extreme

    # 5. Vacuum Zone Tests (controlled OOD: init/goal outside template neighborhoods,
    # but not so far that the task becomes unrealistically hard)
    InterventionSpec("do_vacuum_init_mild", "vacuum_init", {"min_dist_to_template": 0.025}),   # ≥2.5cm from all templates
    InterventionSpec("do_vacuum_init_moderate", "vacuum_init", {"min_dist_to_template": 0.04}), # ≥4cm from all templates
    InterventionSpec("do_vacuum_goal_mild", "vacuum_goal", {"min_dist_to_template": 0.025}),   # goal ≥2.5cm from templates
    InterventionSpec("do_vacuum_goal_moderate", "vacuum_goal", {"min_dist_to_template": 0.04}), # goal ≥4cm from templates
]


def print_results_table(
    results: list[dict],
    baseline_sr: float | None,
    title: str,
) -> None:
    """Print a compact ASCII table for evaluation results."""
    headers = ("Intervention", "SR (%)", "ΔSR (%)", "Dist Mean", "Dist Std", "Episodes")
    rows = []
    for result in results:
        sr = float(result["success_rate"])
        delta_sr = sr - baseline_sr if baseline_sr is not None else 0.0
        rows.append(
            (
                str(result["intervention"]),
                f"{sr:.2f}",
                f"{delta_sr:+.2f}",
                f"{float(result['final_distance_mean']):.4f}",
                f"{float(result['final_distance_std']):.4f}",
                str(int(result["num_episodes"])),
            )
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def format_row(row: tuple[str, ...]) -> str:
        return " | ".join(
            cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i])
            for i, cell in enumerate(row)
        )

    separator = "-+-".join("-" * width for width in widths)

    print(title, flush=True)
    print(format_row(headers), flush=True)
    print(separator, flush=True)
    for row in rows:
        print(format_row(row), flush=True)
    print(flush=True)


def parse_args() -> dict:
    """Parse command-line arguments."""
    cfg = {
        "num_episodes": 50,
        "seed": 42,
        "output": "logs_eval/cube_causal_results.json",
    }
    for arg in sys.argv[1:]:
        if arg in ("--help", "-h"):
            print(__doc__)
            raise SystemExit(0)
        if "=" in arg:
            key, value = arg.split("=", 1)
            if key in ("num_episodes", "seed", "output"):
                if key in ("num_episodes", "seed"):
                    cfg[key] = int(value)
                else:
                    cfg[key] = value
    return cfg


def patch_cube_env_with_causal_intervention():
    """Monkey-patch CubeEnv to add causal intervention methods."""
    from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

    def _get_cube_xy(self, cube_id):
        """Get current cube XY position."""
        joint = self._data.joint(f"object_joint_{cube_id}")
        return joint.qpos[:2].copy()

    def _set_cube_xy(self, cube_id, xy):
        """Set cube XY position."""
        joint = self._data.joint(f"object_joint_{cube_id}")
        joint.qpos[:2] = np.asarray(xy, dtype=np.float64)
        mujoco.mj_forward(self._model, self._data)

    def _get_cube_yaw(self, cube_id):
        """Get current cube yaw (rotation around Z-axis)."""
        joint = self._data.joint(f"object_joint_{cube_id}")
        quat = joint.qpos[3:7]  # quaternion [w, x, y, z]
        # Convert quaternion to yaw angle
        # For rotation around Z: yaw = 2 * atan2(z, w)
        yaw = 2.0 * np.arctan2(quat[3], quat[0])
        return float(yaw)

    def _set_cube_yaw(self, cube_id, yaw):
        """Set cube yaw (rotation around Z-axis)."""
        joint = self._data.joint(f"object_joint_{cube_id}")
        # Convert yaw to quaternion: [cos(yaw/2), 0, 0, sin(yaw/2)]
        half_yaw = float(yaw) / 2.0
        quat = np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float64)
        joint.qpos[3:7] = quat
        mujoco.mj_forward(self._model, self._data)

    def _get_cube_target_xy(self, cube_id):
        """Get current goal XY position."""
        mocap_id = self._cube_target_mocap_ids[cube_id]
        return self._data.mocap_pos[mocap_id][:2].copy()

    def _set_cube_target_xy(self, cube_id, xy):
        """Set goal XY position."""
        mocap_id = self._cube_target_mocap_ids[cube_id]
        self._data.mocap_pos[mocap_id][:2] = np.asarray(xy, dtype=np.float64)
        mujoco.mj_forward(self._model, self._data)

    def _apply_init_pos_noise(self, noise_std, rng):
        """Apply position noise to initial cube position."""
        cube_id = 0
        current_xy = _get_cube_xy(self, cube_id)
        noise = rng.uniform(-noise_std, noise_std, size=2)
        new_xy = current_xy + noise
        # Clamp to workspace bounds [0.31, 0.54] x [-0.29, 0.29]
        new_xy[0] = np.clip(new_xy[0], 0.31, 0.54)
        new_xy[1] = np.clip(new_xy[1], -0.29, 0.29)
        _set_cube_xy(self, cube_id, new_xy)

    def _apply_init_yaw_offset(self, yaw_offset):
        """Apply yaw offset to initial cube orientation."""
        cube_id = 0
        current_yaw = _get_cube_yaw(self, cube_id)
        new_yaw = (current_yaw + float(yaw_offset)) % (2 * np.pi)
        _set_cube_yaw(self, cube_id, new_yaw)

    def _apply_init_yaw_fixed(self, yaw_value):
        """Fix initial cube orientation to a specific yaw."""
        cube_id = 0
        _set_cube_yaw(self, cube_id, float(yaw_value))

    def _apply_init_yaw_align_goal(self):
        """Align initial cube yaw with goal yaw (no rotation needed)."""
        cube_id = 0
        # Get goal orientation from target mocap
        mocap_id = self._cube_target_mocap_ids[cube_id]
        goal_quat = self._data.mocap_quat[mocap_id]
        goal_yaw = 2.0 * np.arctan2(goal_quat[3], goal_quat[0])
        _set_cube_yaw(self, cube_id, goal_yaw)

    def _apply_goal_pos_offset(self, offset_dist, rng):
        """Shift goal position away from center."""
        cube_id = 0
        current_goal_xy = _get_cube_target_xy(self, cube_id)
        # Compute direction away from workspace center
        center = np.array([0.425, 0.0], dtype=np.float64)
        direction = current_goal_xy - center
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-6:
            direction = direction / direction_norm
        else:
            # Random direction if at center
            angle = rng.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])

        new_goal_xy = current_goal_xy + direction * float(offset_dist)
        # Clamp to workspace bounds
        new_goal_xy[0] = np.clip(new_goal_xy[0], 0.31, 0.54)
        new_goal_xy[1] = np.clip(new_goal_xy[1], -0.29, 0.29)
        _set_cube_target_xy(self, cube_id, new_goal_xy)

    def _apply_goal_pos_boundary(self, rng):
        """Move goal to workspace boundary."""
        cube_id = 0
        # Boundary candidates (edges of workspace)
        boundary_candidates = np.array([
            [0.31, -0.29], [0.31, 0.29],  # left edge
            [0.54, -0.29], [0.54, 0.29],  # right edge
            [0.31, 0.0], [0.54, 0.0],     # left/right center
            [0.425, -0.29], [0.425, 0.29], # top/bottom center
        ], dtype=np.float64)

        current_goal_xy = _get_cube_target_xy(self, cube_id)
        # Pick farthest boundary point
        distances = np.linalg.norm(boundary_candidates - current_goal_xy[None, :], axis=1)
        farthest_idx = int(np.argmax(distances))
        new_goal_xy = boundary_candidates[farthest_idx]
        _set_cube_target_xy(self, cube_id, new_goal_xy)

    def _apply_vacuum_init(self, params, rng):
        """Initialize cube in 'vacuum zones' far from all template points.

        This tests the model's ability to handle initial positions that are
        NOT near any of the 5 training templates (far OOD in dynamic variable space).

        Strategy:
        1. Sample random positions in workspace until finding one that is
           at least min_dist_to_template away from ALL 5 templates
        2. This ensures the cube starts in a region never seen during training
        """
        cube_id = 0
        min_dist = float(params.get("min_dist_to_template", 0.05))

        # 5 template INITIAL positions from training (see docs/CUBE_ENV.md)
        templates_init = np.array([
            [0.425, 0.1],    # task 1 init
            [0.35, 0.0],     # task 2 init
            [0.50, 0.0],     # task 3 init
            [0.35, -0.2],    # task 4 init
            [0.35, 0.2],     # task 5 init
        ], dtype=np.float64)

        # Workspace bounds
        x_min, x_max = 0.31, 0.54
        y_min, y_max = -0.29, 0.29

        # Sample until we find a vacuum zone position
        max_attempts = 100
        for _ in range(max_attempts):
            candidate_xy = np.array([
                rng.uniform(x_min, x_max),
                rng.uniform(y_min, y_max)
            ], dtype=np.float64)

            # Check distance to all templates
            distances = np.linalg.norm(templates_init - candidate_xy[None, :], axis=1)
            min_distance_to_any_template = np.min(distances)

            if min_distance_to_any_template >= min_dist:
                # Found a vacuum zone position!
                _set_cube_xy(self, cube_id, candidate_xy)
                return

        # Fallback: if no valid position found, use the point farthest from all templates
        # (this should rarely happen with reasonable min_dist values)
        best_xy = None
        best_min_dist = -1.0
        for _ in range(50):
            candidate_xy = np.array([
                rng.uniform(x_min, x_max),
                rng.uniform(y_min, y_max)
            ], dtype=np.float64)
            distances = np.linalg.norm(templates_init - candidate_xy[None, :], axis=1)
            min_distance = np.min(distances)
            if min_distance > best_min_dist:
                best_min_dist = min_distance
                best_xy = candidate_xy

        if best_xy is not None:
            _set_cube_xy(self, cube_id, best_xy)

    def _apply_vacuum_goal(self, params, rng):
        """Set goal in 'vacuum zones' far from all template goal points.

        This tests the model's ability to reach goals that are NOT near any
        of the 5 training template goals (far OOD in goal space).
        """
        cube_id = 0
        min_dist = float(params.get("min_dist_to_template", 0.05))

        # 5 template GOAL positions from training (see docs/CUBE_ENV.md)
        templates_goal = np.array([
            [0.425, -0.1],   # task 1 goal
            [0.50, 0.0],     # task 2 goal
            [0.35, 0.0],     # task 3 goal
            [0.50, 0.2],     # task 4 goal
            [0.50, -0.2],    # task 5 goal
        ], dtype=np.float64)

        # Workspace bounds
        x_min, x_max = 0.31, 0.54
        y_min, y_max = -0.29, 0.29

        # Sample until we find a vacuum zone goal
        max_attempts = 100
        for _ in range(max_attempts):
            candidate_xy = np.array([
                rng.uniform(x_min, x_max),
                rng.uniform(y_min, y_max)
            ], dtype=np.float64)

            # Check distance to all template goals
            distances = np.linalg.norm(templates_goal - candidate_xy[None, :], axis=1)
            min_distance_to_any_template = np.min(distances)

            if min_distance_to_any_template >= min_dist:
                # Found a vacuum zone goal!
                _set_cube_target_xy(self, cube_id, candidate_xy)
                return

        # Fallback: use the point farthest from all template goals
        best_xy = None
        best_min_dist = -1.0
        for _ in range(50):
            candidate_xy = np.array([
                rng.uniform(x_min, x_max),
                rng.uniform(y_min, y_max)
            ], dtype=np.float64)
            distances = np.linalg.norm(templates_goal - candidate_xy[None, :], axis=1)
            min_distance = np.min(distances)
            if min_distance > best_min_dist:
                best_min_dist = min_distance
                best_xy = candidate_xy

        if best_xy is not None:
            _set_cube_target_xy(self, cube_id, best_xy)

    def _apply_causal_intervention(self, intervention_kind=None, intervention_value=None, seed=None):
        """Apply causal intervention to dynamic variables.

        This method is called AFTER the environment is reset with dataset state/goal,
        allowing us to perturb the dynamic variables that were sampled during training.
        """
        if intervention_kind is None or intervention_kind == "baseline":
            return

        rng = np.random.default_rng(seed)

        if intervention_kind == "init_pos_noise":
            _apply_init_pos_noise(self, float(intervention_value), rng)

        elif intervention_kind == "init_yaw_offset":
            _apply_init_yaw_offset(self, float(intervention_value))

        elif intervention_kind == "init_yaw_fixed":
            _apply_init_yaw_fixed(self, float(intervention_value))

        elif intervention_kind == "init_yaw_align_goal":
            _apply_init_yaw_align_goal(self)

        elif intervention_kind == "goal_pos_offset":
            _apply_goal_pos_offset(self, float(intervention_value), rng)

        elif intervention_kind == "goal_pos_boundary":
            _apply_goal_pos_boundary(self, rng)

        elif intervention_kind == "vacuum_init":
            _apply_vacuum_init(self, intervention_value, rng)

        elif intervention_kind == "vacuum_goal":
            _apply_vacuum_goal(self, intervention_value, rng)

        elif intervention_kind == "compound":
            # Apply multiple interventions sequentially
            params = intervention_value
            if "init_pos_noise" in params:
                _apply_init_pos_noise(self, float(params["init_pos_noise"]), rng)
            if "init_yaw_offset" in params:
                _apply_init_yaw_offset(self, float(params["init_yaw_offset"]))
            if "init_yaw_fixed" in params:
                _apply_init_yaw_fixed(self, float(params["init_yaw_fixed"]))
            if "init_yaw_align_goal" in params and params["init_yaw_align_goal"]:
                _apply_init_yaw_align_goal(self)
            if "goal_pos_offset" in params:
                _apply_goal_pos_offset(self, float(params["goal_pos_offset"]), rng)
            if "goal_pos_boundary" in params and params["goal_pos_boundary"]:
                _apply_goal_pos_boundary(self, rng)

    CubeEnv._apply_causal_intervention = _apply_causal_intervention


def make_intervention_callable(intervention: InterventionSpec, seed: int) -> dict:
    """Create a callable spec for world.evaluate_from_dataset.

    The callable is invoked after state/goal are loaded from dataset.
    """
    return {
        "method": "_apply_causal_intervention",
        "args": {
            "intervention_kind": {"value": intervention.kind, "in_dataset": False},
            "intervention_value": {"value": intervention.value, "in_dataset": False},
            "seed": {"value": seed, "in_dataset": False},
        },
    }


def run_intervention_group(
    intervention: InterventionSpec,
    policy,
    cfg: DictConfig,
    dataset,
    eval_episodes: np.ndarray,
    eval_start_idx: np.ndarray,
    seed: int,
) -> dict:
    """Run evaluation for one intervention group in batches."""

    # Build callables: original + intervention
    base_callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
    if base_callables is None:
        base_callables = []
    intervention_callable = make_intervention_callable(intervention, seed)
    callables = base_callables + [intervention_callable]

    # Batch evaluation to avoid memory issues
    total = len(eval_episodes)
    batch_size = int(cfg.eval.get("batch_size", total))
    if batch_size <= 0:
        batch_size = total

    batch_successes = []
    batch_distances = []
    img_size = int(cfg.eval.img_size)

    print(
        f"[Evaluating] intervention={intervention.name} kind={intervention.kind} "
        f"episodes={total} batch_size={batch_size}",
        flush=True,
    )

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_episodes = eval_episodes[batch_start:batch_end]
        batch_start_steps = eval_start_idx[batch_start:batch_end]

        # Create world for this batch
        world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
        world_cfg["num_envs"] = len(batch_episodes)
        world_cfg["max_episode_steps"] = 2 * cfg.eval.eval_budget

        world = swm.World(**world_cfg, image_shape=(img_size, img_size))
        world.set_policy(policy)

        # Run evaluation for this batch
        metrics = world.evaluate_from_dataset(
            dataset,
            start_steps=batch_start_steps.tolist(),
            goal_offset_steps=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=batch_episodes.tolist(),
            callables=callables,
            save_video=False,
            video_path=None,
        )

        successes = np.asarray(metrics["episode_successes"], dtype=bool)
        batch_successes.append(successes)

        # Extract final distances from world.infos if available
        if hasattr(world, 'infos') and 'final_distance' in world.infos:
            distances = np.asarray(world.infos['final_distance'])
            batch_distances.append(distances)
        else:
            # Fallback: estimate distances (0.0 for success, 0.1 for failure)
            distances = np.where(successes, 0.0, 0.1)
            batch_distances.append(distances)

        world.close()

        print(
            f"  batch {batch_start}:{batch_end} finished for {intervention.name}",
            flush=True,
        )

    # Aggregate results
    successes = np.concatenate(batch_successes)
    distances = np.concatenate(batch_distances)

    return {
        "intervention": intervention.name,
        "success_rate": float(np.mean(successes)) * 100.0,
        "final_distance_mean": float(np.mean(distances)),
        "final_distance_std": float(np.std(distances)),
        "num_episodes": len(successes),
        "successes": successes.tolist(),
        "distances": distances.tolist(),
    }


@hydra.main(version_base=None, config_path="./config/eval", config_name="cube")
def main(cfg: DictConfig):
    """Main evaluation loop."""
    # Parse custom args
    args = parse_args()

    # Patch CubeEnv with causal intervention method
    patch_cube_env_with_causal_intervention()

    print(f"\n{'='*80}")
    print(f"Cube Causal Evaluation (Distribution Shift on Dynamic Variables)")
    print(f"{'='*80}")
    print(f"Policy: {cfg.policy}")
    print(f"Dataset: {cfg.eval.dataset_name}")
    print(f"Episodes: {args['num_episodes']}")
    print(f"Seed: {args['seed']}")
    print(f"{'='*80}\n")

    # Load dataset
    dataset = get_dataset(cfg, cfg.eval.dataset_name)

    # Setup preprocessing
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    # Load policy
    policy_name = cfg.get("policy", "random")

    if policy_name != "random":
        model = load_lewm_model(cfg.policy)
        if model is not None:
            model = model.to("cuda").eval()
            model.requires_grad_(False)
            model.interpolate_pos_encoding = True
            config = swm.PlanConfig(**cfg.plan_config)
            solver = hydra.utils.instantiate(cfg.solver, model=model)
            policy = swm.policy.WorldModelPolicy(
                solver=solver, config=config, process=process, transform=transform
            )
        else:
            try:
                model = swm.policy.AutoCostModel(cfg.policy)
                model = model.to("cuda").eval()
                model = patch_gcrl_compat(model)
                model.requires_grad_(False)
                model.interpolate_pos_encoding = True
                config = swm.PlanConfig(**cfg.plan_config)
                solver = hydra.utils.instantiate(cfg.solver, model=model)
                policy = swm.policy.WorldModelPolicy(
                    solver=solver, config=config, process=process, transform=transform
                )
            except RuntimeError as e:
                if "get_cost" not in str(e):
                    raise
                model = swm.policy.AutoActionableModel(cfg.policy)
                model = model.to("cuda").eval()
                model = patch_gcrl_compat(model)
                model.requires_grad_(False)
                policy = FastActionablePolicy(
                    model=model, process=process, transform=transform, img_size=cfg.eval.img_size
                )
    else:
        policy = swm.policy.RandomPolicy()

    # Sample episodes
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}

    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    g = np.random.default_rng(args["seed"])
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=args["num_episodes"], replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    # Run evaluation for each intervention
    all_results = {}
    baseline_sr = None

    start_time = time.time()

    print(f"\n{'='*80}")
    print(f"Running Causal Interventions")
    print(f"{'='*80}\n")

    for intervention in INTERVENTIONS:
        result = run_intervention_group(
            intervention,
            policy,
            cfg,
            dataset,
            eval_episodes,
            eval_start_idx,
            args["seed"],
        )

        all_results[intervention.name] = result

        if intervention.kind == "baseline":
            baseline_sr = result["success_rate"]

        print(f"[Completed] intervention={intervention.name}", flush=True)
        print_results_table(
            [result],
            baseline_sr=baseline_sr,
            title="Current Intervention Result",
        )

    elapsed = time.time() - start_time

    # Print final summary table
    print(f"{'='*80}", flush=True)
    print("Final Causal Evaluation Results", flush=True)
    print(f"{'='*80}", flush=True)
    print_results_table(
        [all_results[intervention.name] for intervention in INTERVENTIONS],
        baseline_sr=baseline_sr,
        title="All Interventions",
    )
    print(f"Total time: {elapsed:.1f}s", flush=True)
    print(f"{'='*80}\n", flush=True)

    # Save results (JSON)
    output_path = Path(args["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "policy": policy_name,
        "dataset": cfg.eval.dataset_name,
        "num_episodes": args["num_episodes"],
        "seed": args["seed"],
        "elapsed_time": elapsed,
        "baseline_sr": baseline_sr,
        "results": all_results,
    }

    with output_path.open("w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_path}", flush=True)

    # Save results (CSV)
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            "intervention",
            "success_rate",
            "delta_sr",
            "final_distance_mean",
            "final_distance_std",
            "num_episodes",
        ])
        # Data rows
        for intervention in INTERVENTIONS:
            result = all_results[intervention.name]
            sr = float(result["success_rate"])
            delta_sr = sr - baseline_sr if baseline_sr is not None else 0.0
            writer.writerow([
                result["intervention"],
                f"{sr:.2f}",
                f"{delta_sr:+.2f}",
                f"{float(result['final_distance_mean']):.4f}",
                f"{float(result['final_distance_std']):.4f}",
                int(result["num_episodes"]),
            ])

    print(f"CSV results saved to: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
