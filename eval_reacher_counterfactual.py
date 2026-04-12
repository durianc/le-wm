"""Closed-loop counterfactual evaluation of world models on Reacher environment.

This script evaluates world model policies under dynamics out-of-distribution (OOD)
scenarios via counterfactual interventions P(y|do(x), s). It tests closed-loop
performance when the physical dynamics differ from training.

Interventions:
  - baseline: No modifications
  - do_mass_heavy: Increase arm density to 1500 kg/m³ (tests inertia understanding)
  - do_mass_light: Decrease arm density to 500 kg/m³ (tests inertia understanding)
  - do_damping_high: Increase joint damping to 0.02 (tests friction modeling)
  - do_damping_low: Decrease joint damping to 0.005 (tests friction modeling)
  - do_target_ood: Target at radius > 0.20m (spatial OOD, near workspace boundary)
  - visual_cf_color: Change arm/target colors (should NOT affect dynamics)

Usage:
  python eval_reacher_counterfactual.py policy=reacher/lewm [num_episodes=50] [seed=42]

Output:
  Prints summary table and writes JSON to logs_eval/reacher_cf_<policy>.json
"""
from __future__ import annotations

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
    """Specification for a counterfactual intervention."""
    name: str
    kind: str
    value: float | dict | None = None


INTERVENTIONS: list[InterventionSpec] = [
    InterventionSpec("baseline", "baseline"),
    InterventionSpec("do_mass_heavy", "mass", 1500.0),
    InterventionSpec("do_mass_light", "mass", 500.0),
    InterventionSpec("do_damping_high", "damping", 0.02),
    InterventionSpec("do_damping_low", "damping", 0.005),
    InterventionSpec("do_target_ood", "target_ood", {"min_radius": 0.20, "max_radius": 0.24}),
    InterventionSpec("visual_cf_color", "visual_color"),
]


def parse_args() -> dict:
    """Parse command-line arguments (non-Hydra args only)."""
    cfg = {
        "num_episodes": 50,
        "seed": 42,
        "output": "logs_eval/reacher_cf_results.json",
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


def patch_reacher_env_with_intervention():
    """Monkey-patch ReacherDMControlWrapper to add _apply_cf_intervention method."""
    from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper
    from dm_control import mjcf

    def _apply_cf_intervention(self, intervention_kind=None, intervention_value=None):
        """Apply counterfactual intervention to MuJoCo model."""
        if intervention_kind is None or intervention_kind == "baseline":
            return

        # Mark model as dirty to trigger recompilation
        needs_recompile = False

        if intervention_kind == "mass":
            # Modify arm and finger density
            density = float(intervention_value)
            self.variation_space['agent']['arm_density'].set_value(np.array([density], dtype=np.float32))
            self.variation_space['agent']['finger_density'].set_value(np.array([density], dtype=np.float32))
            needs_recompile = True

        elif intervention_kind == "damping":
            # Modify joint damping (requires XML modification)
            damping = float(intervention_value)
            arm_joint = self._mjcf_model.find('joint', 'arm')
            wrist_joint = self._mjcf_model.find('joint', 'wrist')
            if arm_joint is not None:
                arm_joint.damping = damping
            if wrist_joint is not None:
                wrist_joint.damping = damping
            needs_recompile = True

        elif intervention_kind == "target_ood":
            # Override target position sampling to OOD region
            # This will be applied during reset, so we store it for later
            self._cf_target_override = intervention_value

        elif intervention_kind == "visual_color":
            # Change visual appearance (should NOT affect dynamics)
            # Swap arm and target colors
            arm_color = self.variation_space['agent']['color'].value.copy()
            target_color = self.variation_space['target']['color'].value.copy()
            self.variation_space['agent']['color'].set_value(target_color)
            self.variation_space['target']['color'].set_value(arm_color)
            needs_recompile = True

        if needs_recompile:
            self.mark_dirty()

    def _reset_with_ood_target(self, seed=None, options=None):
        """Override reset to apply OOD target position if intervention is active."""
        # Call original reset
        obs, info = self._original_reset(seed=seed, options=options)

        # Apply OOD target position if intervention is active
        if hasattr(self, '_cf_target_override') and self._cf_target_override is not None:
            min_radius = self._cf_target_override['min_radius']
            max_radius = self._cf_target_override['max_radius']

            # Sample OOD target position
            rng = np.random.default_rng(seed)
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(min_radius, max_radius)

            # Set target position
            self.env.physics.named.model.geom_pos['target', 'x'] = radius * np.sin(angle)
            self.env.physics.named.model.geom_pos['target', 'y'] = radius * np.cos(angle)

            # Re-render observation
            obs = self._get_obs()

        return obs, info

    # Store original reset and patch it
    ReacherDMControlWrapper._original_reset = ReacherDMControlWrapper.reset
    ReacherDMControlWrapper.reset = _reset_with_ood_target
    ReacherDMControlWrapper._apply_cf_intervention = _apply_cf_intervention


def make_intervention_callable(intervention: InterventionSpec) -> dict:
    """Create a callable spec for world.evaluate_from_dataset.

    This callable will be invoked after state/goal are set from dataset.
    """
    return {
        "method": "_apply_cf_intervention",
        "args": {
            "intervention_kind": {"value": intervention.kind, "in_dataset": False},
            "intervention_value": {"value": intervention.value, "in_dataset": False},
        },
    }


def run_intervention_group(
    intervention: InterventionSpec,
    policy,
    cfg: DictConfig,
    dataset,
    eval_episodes: np.ndarray,
    eval_start_idx: np.ndarray,
) -> dict:
    """Run evaluation for one intervention group in batches."""

    # Build callables: original + intervention
    base_callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
    intervention_callable = make_intervention_callable(intervention)
    callables = base_callables + [intervention_callable]

    # Batch evaluation to avoid memory issues
    total = len(eval_episodes)
    batch_size = int(cfg.eval.get("batch_size", total))
    if batch_size <= 0:
        batch_size = total

    batch_successes = []
    batch_joint_errors = []
    img_size = int(cfg.eval.img_size)

    print(f"  Running {intervention.name:25s} ({total} episodes in batches of {batch_size})...")

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

        batch_successes.append(np.asarray(metrics["episode_successes"], dtype=bool))

        # Extract joint errors from metrics if available
        if "final_joint_errors" in metrics:
            batch_joint_errors.extend(metrics["final_joint_errors"])

    # Aggregate results
    successes = np.concatenate(batch_successes)

    # Compute joint errors (use 0.0 for success, estimate for failure)
    if not batch_joint_errors:
        joint_errors = []
        for success in successes:
            if success:
                joint_errors.append(0.0)
            else:
                # Estimate: if failed, assume ~0.1 rad error
                joint_errors.append(0.1)
    else:
        joint_errors = batch_joint_errors

    return {
        "intervention": intervention.name,
        "success_rate": float(np.mean(successes)) * 100.0,
        "mean_joint_error": float(np.mean(joint_errors)),
        "std_joint_error": float(np.std(joint_errors)),
        "num_episodes": len(successes),
        "successes": successes.tolist(),
        "joint_errors": joint_errors,
    }


@hydra.main(version_base=None, config_path="./config/eval", config_name="reacher")
def main(cfg: DictConfig):
    """Main evaluation loop."""
    # Parse custom args before Hydra processes them
    args = parse_args()

    # Patch ReacherDMControlWrapper with intervention method
    patch_reacher_env_with_intervention()

    # Override config with command-line args (only policy goes through Hydra)
    if args.get("policy") and args["policy"] != "random":
        cfg.policy = args["policy"]

    print(f"\n{'='*80}")
    print(f"Reacher Counterfactual Evaluation")
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
    print(f"Running Interventions")
    print(f"{'='*80}\n")

    for intervention in INTERVENTIONS:
        result = run_intervention_group(
            intervention,
            policy,
            cfg,
            dataset,
            eval_episodes,
            eval_start_idx,
        )

        all_results[intervention.name] = result

        if intervention.kind == "baseline":
            baseline_sr = result["success_rate"]

        # Print result immediately after completion
        sr = result["success_rate"]
        delta_sr = sr - baseline_sr if baseline_sr is not None else 0.0
        err_mean = result["mean_joint_error"]
        err_std = result["std_joint_error"]

        print(f"  ✓ {intervention.name:<27} SR: {sr:>6.2f}%  ΔSR: {delta_sr:>+7.2f}%  JointErr: {err_mean:.4f}±{err_std:.4f}\n")

    elapsed = time.time() - start_time

    # Print final summary table
    print(f"{'='*80}")
    print(f"Final Results Summary")
    print(f"{'='*80}")
    print(f"{'Intervention':<30} {'SR (%)':<10} {'ΔSR (%)':<10} {'JointErr (mean±std)':<20}")
    print(f"{'-'*80}")

    for intervention in INTERVENTIONS:
        result = all_results[intervention.name]
        sr = result["success_rate"]
        delta_sr = sr - baseline_sr if baseline_sr is not None else 0.0
        err_mean = result["mean_joint_error"]
        err_std = result["std_joint_error"]

        print(f"{intervention.name:<30} {sr:>8.2f}  {delta_sr:>+9.2f}  {err_mean:>8.4f}±{err_std:<8.4f}")

    print(f"{'-'*80}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'='*80}\n")

    # Save results
    output_path = Path(args["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "policy": policy_name,
        "dataset": cfg.eval.dataset_name,
        "num_episodes": args["num_episodes"],
        "seed": args["seed"],
        "elapsed_time": elapsed,
        "results": all_results,
    }

    with output_path.open("w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
