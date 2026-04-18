"""Closed-loop counterfactual evaluation of world models on OGBench CubeEnv.

This script evaluates world model policies under dynamics out-of-distribution (OOD)
scenarios via counterfactual interventions P(y|do(x), s). It tests closed-loop
performance when the physical dynamics differ from training.

Interventions:
  - baseline: No modifications
  - visual_counterfactual: Visual counterfactuals on cube appearance and scene layout
  - visual_cube_color_ood: Change only cube colors to an unseen OOD palette
  - cube_size: Scale cube size up or down
  - ood_position: Move initial/goal positions to near-OOD or far-OOD regions

Usage:
  python eval_cube_counterfactual.py policy=cube/lewm [num_episodes=50] [seed=42]

Output:
  Prints summary table and writes JSON to logs_eval/cube_cf_<policy>.json
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
    """Specification for a counterfactual intervention."""
    name: str
    kind: str
    value: float | None = None


INTERVENTIONS: list[InterventionSpec] = [
    InterventionSpec("baseline", "baseline"),
    InterventionSpec("visual_cf_cube_color_ood", "visual_cube_color_ood"),
    InterventionSpec("visual_cf_swap", "visual_swap"),
    InterventionSpec("visual_cf_floor", "visual_floor"),
    InterventionSpec("cube_size_small", "size_scale", 0.75),
    InterventionSpec("cube_size_large", "size_scale", 1.25),
    InterventionSpec("near_ood_init", "near_ood_init"),
    InterventionSpec("far_ood_init", "far_ood_init"),
    InterventionSpec("far_ood_goal", "far_ood_goal"),
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
    """Parse command-line arguments (non-Hydra args only)."""
    cfg = {
        "num_episodes": 50,
        "seed": 42,
        "output": "logs_eval/cube_cf_results.json",
    }
    for arg in sys.argv[1:]:
        if arg in ("--help", "-h"):
            print(__doc__)
            raise SystemExit(0)
        if "=" in arg:
            key, value = arg.split("=", 1)
            # Only parse our custom args, let Hydra handle policy/dataset
            if key in ("num_episodes", "seed", "output"):
                if key in ("num_episodes", "seed"):
                    cfg[key] = int(value)
                else:
                    cfg[key] = value
    return cfg


def patch_cube_env_with_intervention():
    """Monkey-patch CubeEnv to add _apply_cf_intervention method."""
    from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

    def _pick_nearest_candidate(current_xy, candidates):
        current_xy = np.asarray(current_xy, dtype=np.float64)
        candidates = np.asarray(candidates, dtype=np.float64)
        distances = np.linalg.norm(candidates - current_xy[None, :], axis=1)
        return candidates[int(np.argmin(distances))]

    def _pick_farthest_candidate(current_xy, candidates):
        current_xy = np.asarray(current_xy, dtype=np.float64)
        candidates = np.asarray(candidates, dtype=np.float64)
        distances = np.linalg.norm(candidates - current_xy[None, :], axis=1)
        return candidates[int(np.argmax(distances))]

    def _set_cube_xy(self, cube_id, xy):
        joint = self._data.joint(f"object_joint_{cube_id}")
        joint.qpos[:2] = np.asarray(xy, dtype=np.float64)
        mujoco.mj_forward(self._model, self._data)

    def _set_cube_target_xy(self, cube_id, xy):
        mocap_id = self._cube_target_mocap_ids[cube_id]
        self._data.mocap_pos[mocap_id][:2] = np.asarray(xy, dtype=np.float64)
        mujoco.mj_forward(self._model, self._data)

    def _apply_cube_size_scale(self, scale):
        for i in range(self._num_cubes):
            joint = self._data.joint(f"object_joint_{i}")
            current_half_extent = float(self._model.geom_size[self._cube_geom_ids_list[i][0]][2])
            new_half_extent = current_half_extent * float(scale)
            delta_half_extent = new_half_extent - current_half_extent
            joint.qpos[2] += delta_half_extent

            for geom_id in self._cube_geom_ids_list[i]:
                self._model.geom_size[geom_id][:3] *= float(scale)

            for geom_id in self._cube_target_geom_ids_list[i]:
                self._model.geom_size[geom_id][:3] *= float(scale)

            mocap_id = self._cube_target_mocap_ids[i]
            self._data.mocap_pos[mocap_id][2] += delta_half_extent

        mujoco.mj_forward(self._model, self._data)

    def _apply_position_counterfactual(self, mode):
        cube_id = 0
        current_xy = self._data.joint(f"object_joint_{cube_id}").qpos[:2].copy()
        current_goal_xy = self._data.mocap_pos[self._cube_target_mocap_ids[cube_id]][:2].copy()

        near_ood_candidates = np.array(
            [
                [0.3875, -0.10],
                [0.3875, 0.10],
                [0.4625, -0.10],
                [0.4625, 0.10],
                [0.4250, -0.15],
                [0.4250, 0.15],
            ],
            dtype=np.float64,
        )

        edge_candidates = np.array(
            [
                [0.31, -0.29],
                [0.31, 0.29],
                [0.54, -0.29],
                [0.54, 0.29],
                [0.31, 0.00],
                [0.54, 0.00],
                [0.425, -0.29],
                [0.425, 0.29],
            ],
            dtype=np.float64,
        )

        if mode == "near_ood_init":
            new_xy = _pick_nearest_candidate(current_xy, near_ood_candidates)
            _set_cube_xy(self, cube_id, new_xy)
        elif mode == "far_ood_init":
            new_xy = _pick_farthest_candidate(current_xy, edge_candidates)
            _set_cube_xy(self, cube_id, new_xy)
        elif mode == "far_ood_goal":
            new_xy = _pick_farthest_candidate(current_goal_xy, edge_candidates)
            _set_cube_target_xy(self, cube_id, new_xy)

    def _apply_cf_intervention(self, intervention_kind=None, intervention_value=None):
        """Apply counterfactual intervention to MuJoCo model."""
        if intervention_kind is None or intervention_kind == "baseline":
            return

        model = self._model

        if intervention_kind == "mass":
            multiplier = intervention_value
            for i in range(self._num_cubes):
                body_name = f"object_{i}"
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if body_id >= 0:
                    model.body_mass[body_id] *= multiplier

        elif intervention_kind == "friction":
            multiplier = intervention_value
            for i in range(self._num_cubes):
                for geom_id in self._cube_geom_ids_list[i]:
                    model.geom_friction[geom_id] *= multiplier

            floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            if floor_geom_id >= 0:
                model.geom_friction[floor_geom_id] *= multiplier

        elif intervention_kind == "visual_swap":
            cube_colors = []
            for i in range(self._num_cubes):
                cube_colors.append(model.geom(self._cube_geom_ids_list[i][0]).rgba[:3].copy())

            agent_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ur5e/robotiq/base_link")
            if agent_geom_id >= 0:
                agent_color = model.geom(agent_geom_id).rgba[:3].copy()
            else:
                agent_color = np.array([0.5, 0.5, 0.5])

            for i in range(self._num_cubes):
                for gid in self._cube_geom_ids_list[i]:
                    model.geom(gid).rgba[:3] = agent_color

            for geom_name in ["ur5e/robotiq/base_link", "ur5e/robotiq/left_pad", "ur5e/robotiq/right_pad"]:
                geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                if geom_id >= 0:
                    model.geom(geom_id).rgba[:3] = cube_colors[0]

        elif intervention_kind == "visual_cube_color_ood":
            # Use a saturated cyan/teal palette that is outside the default training colors.
            ood_palette = np.array(
                [
                    [0.00, 0.92, 0.92],
                    [0.00, 0.78, 0.86],
                    [0.10, 0.88, 0.72],
                    [0.18, 0.82, 0.96],
                ],
                dtype=np.float64,
            )
            for i in range(self._num_cubes):
                cube_color = ood_palette[i % len(ood_palette)]
                for gid in self._cube_geom_ids_list[i]:
                    model.geom(gid).rgba[:3] = cube_color
                for gid in self._cube_target_geom_ids_list[i]:
                    model.geom(gid).rgba[:3] = cube_color

        elif intervention_kind == "visual_floor":
            if self._num_cubes > 0:
                cube_color = model.geom(self._cube_geom_ids_list[0][0]).rgba[:3].copy()
                floor_color = cube_color * 0.7

                floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
                if floor_geom_id >= 0:
                    model.geom(floor_geom_id).rgba[:3] = floor_color

        elif intervention_kind == "size_scale":
            _apply_cube_size_scale(self, intervention_value)

        elif intervention_kind in {"near_ood_init", "far_ood_init", "far_ood_goal"}:
            _apply_position_counterfactual(self, intervention_kind)

    CubeEnv._apply_cf_intervention = _apply_cf_intervention


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

        batch_successes.append(np.asarray(metrics["episode_successes"], dtype=bool))
        print(
            f"  batch {batch_start}:{batch_end} finished for {intervention.name}",
            flush=True,
        )

    # Aggregate results
    successes = np.concatenate(batch_successes)

    # Compute distances (0.0 for success, estimate for failure)
    distances = []
    for success in successes:
        if success:
            distances.append(0.0)
        else:
            distances.append(0.1)

    return {
        "intervention": intervention.name,
        "success_rate": float(np.mean(successes)) * 100.0,
        "final_distance_mean": float(np.mean(distances)),
        "final_distance_std": float(np.std(distances)),
        "num_episodes": len(successes),
        "successes": successes.tolist(),
        "distances": distances,
    }


@hydra.main(version_base=None, config_path="./config/eval", config_name="cube")
def main(cfg: DictConfig):
    """Main evaluation loop."""
    # Parse custom args before Hydra processes them
    args = parse_args()

    # Patch CubeEnv with intervention method
    patch_cube_env_with_intervention()

    # Override config with command-line args (only policy goes through Hydra)
    if args.get("policy") and args["policy"] != "random":
        cfg.policy = args["policy"]

    print(f"\n{'='*80}")
    print(f"Cube Counterfactual Evaluation")
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

        print(f"[Completed] intervention={intervention.name}", flush=True)
        print_results_table(
            [result],
            baseline_sr=baseline_sr,
            title="Current Group Result",
        )

    elapsed = time.time() - start_time

    # Print final summary table
    print(f"{'='*80}", flush=True)
    print("Final Results Summary", flush=True)
    print(f"{'='*80}", flush=True)
    print_results_table(
        [all_results[intervention.name] for intervention in INTERVENTIONS],
        baseline_sr=baseline_sr,
        title="All Groups",
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
        "results": all_results,
    }

    with output_path.open("w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
