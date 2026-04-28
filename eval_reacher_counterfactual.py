"""Reacher closed-loop causal evaluation with gradient distribution extrapolation.

This script evaluates LeWM on Reacher under distribution shift interventions
following the "Distribution Extrapolation" principle:

1. Variable Extrapolation: Push training-observed variables beyond their boundaries
2. Constant Bias: Modify training-fixed physical constants to detect overfitting
3. Intensity Levels: Level 1 (mild/near-OOD), Level 2 (moderate), Level 3 (extreme)

Three intervention dimensions:
  (1) Spatial Reachability: Target position beyond training workspace
  (2) Kinematic Limits: Joint range constraints tighter than training
  (3) Body Structure: Arm segment lengths different from training

Training Distribution Reference (from dm_control/suite/reacher.xml):
  - Arm length: 0.12m (shoulder to wrist)
  - Hand length: 0.1m (wrist to finger)
  - Wrist joint range: [-160°, 160°]
  - Target spawn: uniform within ~0.20m radius
  - Shoulder joint: unlimited rotation

Usage
-----
python eval_reacher_causal.py policy=reacher/lewm [num_episodes=50] [seed=42]

Output
------
Prints summary table and writes JSON to logs_eval/reacher_causal_results.json
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
    """Specification for a causal intervention on dynamic or structural variables."""
    name: str
    kind: str
    value: dict | float | tuple[float, float] | None = None


# Training distribution reference:
# - Arm: 0.12m, Hand: 0.1m, Total reach: ~0.22m
# - Wrist: [-160°, 160°] (nearly unlimited)
# - Target: uniform within ~0.20m radius (inside reachable workspace)

INTERVENTIONS: list[InterventionSpec] = [
    InterventionSpec("baseline", "baseline"),

    # ========== Dimension 1: Spatial Reachability (Variable Extrapolation) ==========
    # Training: target spawns uniformly within ~0.20m radius
    # Test: push target progressively toward/beyond workspace boundary (~0.22m)

    InterventionSpec(
        "do_target_radius_mild",
        "target_radius_shift",
        {"min_radius": 0.18, "max_radius": 0.21},  # Level 1: near boundary
    ),
    InterventionSpec(
        "do_target_radius_moderate",
        "target_radius_shift",
        {"min_radius": 0.20, "max_radius": 0.23},  # Level 2: at/beyond boundary
    ),
    InterventionSpec(
        "do_target_radius_extreme",
        "target_radius_shift",
        {"min_radius": 0.22, "max_radius": 0.25},  # Level 3: clearly unreachable
    ),

    # ========== Dimension 2: Kinematic Limits (Variable Extrapolation) ==========
    # Training: wrist joint [-160°, 160°] (nearly unlimited)
    # Test: progressively shrink wrist range to constrain motion diversity

    InterventionSpec(
        "do_wrist_limit_mild",
        "wrist_limit",
        (-135.0, 135.0),  # Level 1: 15% reduction (still wide)
    ),
    InterventionSpec(
        "do_wrist_limit_moderate",
        "wrist_limit",
        (-90.0, 90.0),  # Level 2: 44% reduction (half range)
    ),
    InterventionSpec(
        "do_wrist_limit_extreme",
        "wrist_limit",
        (-45.0, 45.0),  # Level 3: 72% reduction (severe constraint)
    ),

    # ========== Dimension 3: Body Structure (Constant Bias) ==========
    # Training: arm=0.12m, hand=0.1m (fixed in all training episodes)
    # Test: modify segment lengths to detect structural overfitting

    # 3a. Arm Length Variation (affects total reach and joint coupling)
    InterventionSpec(
        "do_arm_length_short_mild",
        "arm_length",
        0.108,  # Level 1: -10% (0.12 → 0.108m)
    ),
    InterventionSpec(
        "do_arm_length_short_moderate",
        "arm_length",
        0.096,  # Level 2: -20% (0.12 → 0.096m)
    ),
    InterventionSpec(
        "do_arm_length_short_extreme",
        "arm_length",
        0.084,  # Level 3: -30% (0.12 → 0.084m)
    ),
    InterventionSpec(
        "do_arm_length_long_mild",
        "arm_length",
        0.132,  # Level 1: +10% (0.12 → 0.132m)
    ),
    InterventionSpec(
        "do_arm_length_long_moderate",
        "arm_length",
        0.144,  # Level 2: +20% (0.12 → 0.144m)
    ),
    InterventionSpec(
        "do_arm_length_long_extreme",
        "arm_length",
        0.156,  # Level 3: +30% (0.12 → 0.156m)
    ),

    # 3b. Hand Length Variation (affects fingertip position and precision)
    InterventionSpec(
        "do_hand_length_short_mild",
        "hand_length",
        0.090,  # Level 1: -10% (0.1 → 0.09m)
    ),
    InterventionSpec(
        "do_hand_length_short_moderate",
        "hand_length",
        0.080,  # Level 2: -20% (0.1 → 0.08m)
    ),
    InterventionSpec(
        "do_hand_length_short_extreme",
        "hand_length",
        0.070,  # Level 3: -30% (0.1 → 0.07m)
    ),
    InterventionSpec(
        "do_hand_length_long_mild",
        "hand_length",
        0.110,  # Level 1: +10% (0.1 → 0.11m)
    ),
    InterventionSpec(
        "do_hand_length_long_moderate",
        "hand_length",
        0.120,  # Level 2: +20% (0.1 → 0.12m)
    ),
    InterventionSpec(
        "do_hand_length_long_extreme",
        "hand_length",
        0.130,  # Level 3: +30% (0.1 → 0.13m)
    ),

    # 3c. Compound Structure Shift (both segments simultaneously)
    InterventionSpec(
        "do_structure_compound_mild",
        "compound_structure",
        {"arm_length": 0.108, "hand_length": 0.110},  # Level 1: arm-10%, hand+10%
    ),
    InterventionSpec(
        "do_structure_compound_moderate",
        "compound_structure",
        {"arm_length": 0.096, "hand_length": 0.120},  # Level 2: arm-20%, hand+20%
    ),
    InterventionSpec(
        "do_structure_compound_extreme",
        "compound_structure",
        {"arm_length": 0.084, "hand_length": 0.130},  # Level 3: arm-30%, hand+30%
    ),
]


def print_results_table(
    results: list[dict],
    baseline_sr: float | None,
    title: str,
) -> None:
    """Print a compact ASCII table for evaluation results."""
    headers = ("Intervention", "SR (%)", "ΔSR (%)", "Episodes")
    rows = []
    for result in results:
        sr = float(result["success_rate"])
        delta_sr = sr - baseline_sr if baseline_sr is not None else 0.0
        rows.append(
            (
                str(result["intervention"]),
                f"{sr:.2f}",
                f"{delta_sr:+.2f}",
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
        "output": "logs_eval/reacher_causal_results.json",
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


def patch_reacher_env_with_causal_intervention():
    """Monkey-patch ReacherDMControlWrapper to add causal intervention methods."""
    from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper

    def _apply_arm_length(self, arm_length):
        """Modify arm segment length (shoulder to wrist)."""
        arm_geom = self._mjcf_model.find("geom", "arm")
        hand_body = self._mjcf_model.find("body", "hand")
        if arm_geom is not None:
            arm_geom.fromto = [0.0, 0.0, 0.0, float(arm_length), 0.0, 0.0]
        if hand_body is not None:
            hand_body.pos = [float(arm_length), 0.0, 0.0]
        self.mark_dirty()

    def _apply_hand_length(self, hand_length):
        """Modify hand segment length (wrist to finger)."""
        hand_geom = self._mjcf_model.find("geom", "hand")
        finger_body = self._mjcf_model.find("body", "finger")
        if hand_geom is not None:
            hand_geom.fromto = [0.0, 0.0, 0.0, float(hand_length), 0.0, 0.0]
        if finger_body is not None:
            finger_body.pos = [float(hand_length), 0.0, 0.0]
        self.mark_dirty()

    def _apply_wrist_limit(self, limit_range):
        """Modify wrist joint range limits."""
        wrist_joint = self._mjcf_model.find("joint", "wrist")
        if wrist_joint is not None:
            lower, upper = limit_range
            wrist_joint.range = [float(lower), float(upper)]
            wrist_joint.limited = True
        self.mark_dirty()

    def _apply_target_radius_shift(self, radius_params, rng):
        """Shift target to specified radius range (applied after reset)."""
        min_radius = radius_params["min_radius"]
        max_radius = radius_params["max_radius"]
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(min_radius, max_radius)
        self.env.physics.named.model.geom_pos["target", "x"] = radius * np.sin(angle)
        self.env.physics.named.model.geom_pos["target", "y"] = radius * np.cos(angle)
        self.env.physics.forward()

    def _apply_causal_intervention(
        self, intervention_kind=None, intervention_value=None, seed=None
    ):
        """Apply causal intervention to dynamic or structural variables.

        This method is called AFTER the environment is reset with dataset state/goal,
        allowing us to perturb variables beyond their training distribution.
        """
        if intervention_kind is None or intervention_kind == "baseline":
            return

        rng = np.random.default_rng(seed)

        if intervention_kind == "arm_length":
            _apply_arm_length(self, float(intervention_value))

        elif intervention_kind == "hand_length":
            _apply_hand_length(self, float(intervention_value))

        elif intervention_kind == "wrist_limit":
            _apply_wrist_limit(self, intervention_value)

        elif intervention_kind == "target_radius_shift":
            _apply_target_radius_shift(self, intervention_value, rng)

        elif intervention_kind == "compound_structure":
            params = intervention_value
            if "arm_length" in params:
                _apply_arm_length(self, float(params["arm_length"]))
            if "hand_length" in params:
                _apply_hand_length(self, float(params["hand_length"]))

    ReacherDMControlWrapper._apply_causal_intervention = _apply_causal_intervention


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

        world.close()

        print(
            f"  batch {batch_start}:{batch_end} finished for {intervention.name}",
            flush=True,
        )

    # Aggregate results
    successes = np.concatenate(batch_successes)

    return {
        "intervention": intervention.name,
        "success_rate": float(np.mean(successes)) * 100.0,
        "num_episodes": len(successes),
        "successes": successes.tolist(),
    }


@hydra.main(version_base=None, config_path="./config/eval", config_name="reacher")
def main(cfg: DictConfig):
    """Main evaluation loop."""
    # Parse custom args
    args = parse_args()

    # Patch ReacherDMControlWrapper with causal intervention method
    patch_reacher_env_with_causal_intervention()

    print(f"\n{'='*80}")
    print(f"Reacher Causal Evaluation (Distribution Extrapolation)")
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

    # Save results
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

    # Save CSV summary for easy analysis
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "intervention",
            "dimension",
            "level",
            "kind",
            "value",
            "success_rate",
            "delta_sr",
            "num_episodes",
            "num_success",
        ])

        for intervention in INTERVENTIONS:
            result = all_results[intervention.name]
            sr = result["success_rate"]
            delta_sr = sr - baseline_sr if baseline_sr is not None else 0.0
            num_success = int(sum(result["successes"]))

            # Parse dimension and level from intervention name
            dimension = "baseline"
            level = "N/A"
            if "target_radius" in intervention.name:
                dimension = "spatial_reachability"
            elif "wrist_limit" in intervention.name:
                dimension = "kinematic_limits"
            elif "arm_length" in intervention.name or "hand_length" in intervention.name or "structure_compound" in intervention.name:
                dimension = "body_structure"

            if "mild" in intervention.name:
                level = "1_mild"
            elif "moderate" in intervention.name:
                level = "2_moderate"
            elif "extreme" in intervention.name:
                level = "3_extreme"

            # Format value for CSV
            value_str = str(intervention.value) if intervention.value is not None else "N/A"

            writer.writerow([
                intervention.name,
                dimension,
                level,
                intervention.kind,
                value_str,
                f"{sr:.2f}",
                f"{delta_sr:.2f}",
                result["num_episodes"],
                num_success,
            ])

    print(f"CSV summary saved to: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
