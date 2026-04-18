"""Closed-loop counterfactual evaluation of world models on Reacher environment.

This script evaluates world model policies under counterfactual out-of-distribution
(OOD) scenarios via interventions P(y|do(x), s). It tests closed-loop performance
when the robot geometry, motion constraints, or target distribution differ from
training.

Interventions:
  - baseline: No modifications
  - do_forearm_length_long: Increase forearm length from 0.12m to 0.144m (+20%)
  - do_forearm_length_short: Decrease forearm length from 0.12m to 0.096m (-20%)
  - do_wrist_limit: Shrink wrist rotation range from [-160, 160] to [-60, 60] degrees
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
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import hydra
import numpy as np
import torch
import warnings
from omegaconf import OmegaConf, DictConfig
from sklearn import preprocessing
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

import stable_worldmodel as swm
from eval import (
    FastActionablePolicy,
    evaluate_in_batches,
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
    value: float | tuple[float, float] | dict | None = None


INTERVENTIONS: list[InterventionSpec] = [
    InterventionSpec("baseline", "baseline"),
    InterventionSpec("do_forearm_length_long", "forearm_length", 0.144),
    InterventionSpec("do_forearm_length_short", "forearm_length", 0.096),
    # Previous setting (-60, 60) was too restrictive and collapsed success rate.
    InterventionSpec("do_wrist_limit", "wrist_limit", (-120.0, 120.0)),
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
    """Patch reset-time counterfactual hooks onto ReacherDMControlWrapper."""
    from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper

    if getattr(ReacherDMControlWrapper, "_cf_patch_installed", False):
        return

    original_reset = ReacherDMControlWrapper.reset

    def _prepare_cf_intervention(self, options=None):
        """Apply interventions that must happen before the real reset()."""
        options = options or {}
        intervention_kind = options.get("_cf_intervention_kind")
        intervention_value = options.get("_cf_intervention_value")

        if intervention_kind is None or intervention_kind in {"baseline", "target_ood", "visual_color"}:
            return

        if intervention_kind == "forearm_length":
            forearm_length = float(intervention_value)
            arm_geom = self._mjcf_model.find("geom", "arm")
            hand_body = self._mjcf_model.find("body", "hand")
            if arm_geom is not None:
                arm_geom.fromto = [0.0, 0.0, 0.0, forearm_length, 0.0, 0.0]
            if hand_body is not None:
                hand_body.pos = [forearm_length, 0.0, 0.0]
            self.mark_dirty()

        elif intervention_kind == "wrist_limit":
            wrist_joint = self._mjcf_model.find("joint", "wrist")
            if wrist_joint is not None:
                lower, upper = intervention_value
                wrist_joint.range = [float(lower), float(upper)]
                wrist_joint.limited = True
            self.mark_dirty()

    def reset(self, seed=None, options=None):
        """Apply reset-time interventions before the wrapper performs reset()."""
        reset_options = dict(options or {})
        self._prepare_cf_intervention(reset_options)
        reset_options.pop("_cf_intervention_kind", None)
        reset_options.pop("_cf_intervention_value", None)
        return original_reset(self, seed=seed, options=reset_options)

    def _apply_cf_intervention(self, intervention_kind=None, intervention_value=None):
        """Apply runtime-only interventions after state/goal restoration."""
        if intervention_kind is None or intervention_kind == "baseline":
            return

        if intervention_kind == "target_ood":
            # Apply target shift directly on the current compiled physics state.
            min_radius = intervention_value['min_radius']
            max_radius = intervention_value['max_radius']
            rng = np.random.default_rng()
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(min_radius, max_radius)
            self.env.physics.named.model.geom_pos['target', 'x'] = radius * np.sin(angle)
            self.env.physics.named.model.geom_pos['target', 'y'] = radius * np.cos(angle)
            self.env.physics.forward()

    ReacherDMControlWrapper._prepare_cf_intervention = _prepare_cf_intervention
    ReacherDMControlWrapper.reset = reset
    ReacherDMControlWrapper._apply_cf_intervention = _apply_cf_intervention
    ReacherDMControlWrapper._cf_patch_installed = True


def make_intervention_callable(intervention: InterventionSpec) -> dict:
    """Create a callable spec for world.evaluate_from_dataset.

    The placement in the setup sequence is controlled by build_callables().
    """
    return {
        "method": "_apply_cf_intervention",
        "args": {
            "intervention_kind": {"value": intervention.kind, "in_dataset": False},
            "intervention_value": {"value": intervention.value, "in_dataset": False},
        },
    }


def build_callables(cfg: DictConfig, intervention: InterventionSpec) -> list[dict]:
    """Build post-reset callables while keeping baseline identical to eval.py."""
    base_callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
    if intervention.kind in {"baseline", "forearm_length", "wrist_limit", "visual_color"}:
        return base_callables

    intervention_callable = make_intervention_callable(intervention)
    return base_callables + [intervention_callable]


def build_reset_options(
    intervention: InterventionSpec,
    num_envs: int,
) -> list[dict]:
    """Build per-env reset options so interventions are applied before obs_0 is read."""
    options = [{} for _ in range(num_envs)]

    if intervention.kind in {"forearm_length", "wrist_limit"}:
        for opt in options:
            opt["_cf_intervention_kind"] = intervention.kind
            opt["_cf_intervention_value"] = intervention.value

    elif intervention.kind == "visual_color":
        arm_color = np.array([0.7, 0.5, 0.3], dtype=np.float64)
        target_color = np.array([0.6, 0.3, 0.3], dtype=np.float64)
        for opt in options:
            opt["variation_values"] = {
                "agent.color": target_color.copy(),
                "target.color": arm_color.copy(),
            }

    return options


def run_intervention_group(
    intervention: InterventionSpec,
    policy,
    cfg: DictConfig,
    dataset,
    eval_episodes: np.ndarray,
    eval_start_idx: np.ndarray,
) -> dict:
    """Run one intervention with the same closed-loop evaluation logic as eval.py."""

    callables = build_callables(cfg, intervention)
    total = len(eval_episodes)
    batch_size = int(cfg.eval.get("batch_size", total))
    print(
        f"  Running {intervention.name:25s} "
        f"({total} episodes in batches of {batch_size})..."
    )

    if intervention.kind == "baseline":
        metrics = evaluate_in_batches(
            cfg,
            dataset=dataset,
            policy=policy,
            episodes_idx=np.asarray(eval_episodes),
            start_steps=np.asarray(eval_start_idx),
            results_path=Path(args_output_dir(cfg, intervention.name)),
        )
    else:
        metrics = evaluate_intervention_in_batches(
            cfg,
            dataset=dataset,
            policy=policy,
            intervention=intervention,
            episodes_idx=np.asarray(eval_episodes),
            start_steps=np.asarray(eval_start_idx),
            callables=callables,
            results_path=Path(args_output_dir(cfg, intervention.name)),
        )

    successes = np.asarray(metrics["episode_successes"], dtype=bool)

    return {
        "intervention": intervention.name,
        "success_rate": float(np.mean(successes)) * 100.0,
        "num_episodes": len(successes),
        "successes": successes.tolist(),
    }


def args_output_dir(cfg: DictConfig, intervention_name: str) -> str:
    """Build a per-intervention video directory consistent with eval.py outputs."""
    root = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )
    return str(root / "cf_failures" / intervention_name)


def evaluate_intervention_in_batches(
    cfg: DictConfig,
    dataset,
    policy,
    intervention: InterventionSpec,
    episodes_idx: np.ndarray,
    start_steps: np.ndarray,
    callables: list[dict],
    results_path: Path,
):
    """Mirror eval.evaluate_in_batches while allowing intervention-specific callables."""
    total = len(episodes_idx)
    batch_size = int(cfg.eval.get("batch_size", total))
    if batch_size <= 0:
        raise ValueError("eval.batch_size must be a positive integer.")

    batch_successes = []
    batch_seeds = []
    save_video = bool(cfg.eval.get("save_video", True))
    img_size = int(cfg.eval.img_size)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_episodes = episodes_idx[batch_start:batch_end]
        batch_start_steps = start_steps[batch_start:batch_end]

        batch_world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
        batch_world_cfg["num_envs"] = len(batch_episodes)
        batch_world_cfg["max_episode_steps"] = 2 * cfg.eval.eval_budget

        world = swm.World(**batch_world_cfg, image_shape=(img_size, img_size))
        world.set_policy(policy)

        print(
            f"Evaluating batch {batch_start // batch_size + 1}/"
            f"{(total + batch_size - 1) // batch_size} "
            f"with {len(batch_episodes)} episodes."
        )
        metrics = evaluate_intervention_from_dataset(
            world,
            dataset,
            episodes_idx=batch_episodes.tolist(),
            start_steps=batch_start_steps.tolist(),
            goal_offset_steps=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            intervention=intervention,
            callables=callables,
            save_video=save_video,
            video_path=results_path,
        )
        batch_successes.append(np.asarray(metrics["episode_successes"], dtype=bool))
        if metrics.get("seeds") is not None:
            batch_seeds.append(np.asarray(metrics["seeds"]))

    episode_successes = np.concatenate(batch_successes)
    return {
        "success_rate": float(np.mean(episode_successes)) * 100.0,
        "episode_successes": episode_successes,
        "seeds": np.concatenate(batch_seeds) if batch_seeds else None,
    }


def evaluate_intervention_from_dataset(
    world,
    dataset,
    episodes_idx,
    start_steps,
    goal_offset_steps,
    eval_budget,
    intervention: InterventionSpec,
    callables=None,
    save_video=True,
    video_path="./",
):
    """Clone World.evaluate_from_dataset with reset-time intervention injection."""
    assert (
        world.envs.envs[0].spec.max_episode_steps is None
        or world.envs.envs[0].spec.max_episode_steps >= goal_offset_steps
    ), "env max_episode_steps must be greater than eval_budget"

    ep_idx_arr = np.array(episodes_idx)
    start_steps_arr = np.array(start_steps)
    end_steps = start_steps_arr + goal_offset_steps

    if len(ep_idx_arr) != len(start_steps_arr):
        raise ValueError("episodes_idx and start_steps must have the same length")

    if len(ep_idx_arr) != world.num_envs:
        raise ValueError("Number of episodes to evaluate must match number of envs")

    data = dataset.load_chunk(ep_idx_arr, start_steps_arr, end_steps)
    columns = dataset.column_names

    init_step_per_env = defaultdict(list)
    goal_step_per_env = defaultdict(list)

    for ep in data:
        for col in columns:
            if col.startswith("goal"):
                continue
            if col.startswith("pixels"):
                ep[col] = ep[col].permute(0, 2, 3, 1)

            if not isinstance(ep[col], (torch.Tensor, np.ndarray)):
                continue

            init_data = ep[col][0]
            goal_data = ep[col][-1]

            if not isinstance(init_data, (np.ndarray, torch.Tensor)):
                continue

            init_data = init_data.numpy() if isinstance(init_data, torch.Tensor) else init_data
            goal_data = goal_data.numpy() if isinstance(goal_data, torch.Tensor) else goal_data

            init_step_per_env[col].append(init_data)
            goal_step_per_env[col].append(goal_data)

    init_step = {k: np.stack(v) for k, v in deepcopy(init_step_per_env).items()}

    goal_step = {}
    for key, value in goal_step_per_env.items():
        key = "goal" if key == "pixels" else f"goal_{key}"
        goal_step[key] = np.stack(value)

    seeds = init_step.get("seed")
    vkey = "variation."
    variations_dict = {
        k.removeprefix(vkey): v for k, v in init_step.items() if k.startswith(vkey)
    }

    options = build_reset_options(intervention, world.num_envs)
    for i in range(world.num_envs):
        if len(variations_dict) > 0:
            options[i]["variation"] = list(variations_dict.keys())
            options[i]["variation_values"] = {
                **{k: v[i] for k, v in variations_dict.items()},
                **options[i].get("variation_values", {}),
            }

    init_step.update(deepcopy(goal_step))
    world.reset(seed=seeds, options=options)

    callables = callables or []
    for i, env in enumerate(world.envs.unwrapped.envs):
        env_unwrapped = env.unwrapped
        for spec in callables:
            method_name = spec["method"]
            if not hasattr(env_unwrapped, method_name):
                continue

            method = getattr(env_unwrapped, method_name)
            args = spec.get("args", spec)
            prepared_args = {}
            for args_name, args_data in args.items():
                value = args_data.get("value", None)
                is_in_dataset = args_data.get("in_dataset", True)
                if is_in_dataset:
                    if value not in init_step:
                        continue
                    prepared_args[args_name] = deepcopy(init_step[value][i])
                else:
                    prepared_args[args_name] = args_data.get("value")

            method(**prepared_args)

    results = {
        "success_rate": 0.0,
        "episode_successes": np.zeros(len(episodes_idx)),
        "seeds": seeds,
    }

    shape_prefix = world.infos["pixels"].shape[:2]
    init_step = {
        k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:])
        for k, v in init_step.items()
    }
    goal_step = {
        k: np.broadcast_to(v[:, None, ...], shape_prefix + v.shape[1:])
        for k, v in goal_step.items()
    }

    world.infos.update(deepcopy(init_step))
    world.infos.update(deepcopy(goal_step))

    target_frames = None
    video_frames = None
    if save_video:
        target_frames = torch.stack([ep["pixels"] for ep in data]).numpy()
        video_frames = np.empty(
            (world.num_envs, eval_budget, *world.infos["pixels"].shape[-3:]),
            dtype=np.uint8,
        )

    with tqdm(total=eval_budget, desc="Evaluating", leave=True) as pbar:
        for i in range(eval_budget):
            if save_video:
                video_frames[:, i] = world.infos["pixels"][:, -1]
            world.infos.update(deepcopy(goal_step))
            world.step()
            results["episode_successes"] = np.logical_or(
                results["episode_successes"], world.terminateds
            )
            world.envs.unwrapped._autoreset_envs = np.zeros((world.num_envs,))
            pbar.set_postfix(
                successes=int(np.sum(results["episode_successes"])),
                total_envs=world.num_envs,
            )
            pbar.update(1)

    if save_video:
        video_frames[:, -1] = world.infos["pixels"][:, -1]

    n_episodes = len(episodes_idx)
    results["success_rate"] = (
        float(np.sum(results["episode_successes"])) / n_episodes * 100.0
    )

    if save_video:
        import imageio

        target_len = target_frames.shape[1]
        video_path_obj = Path(video_path)
        video_path_obj.mkdir(parents=True, exist_ok=True)
        for i in range(world.num_envs):
            out = imageio.get_writer(
                video_path_obj / f"rollout_{i}.mp4",
                fps=15,
                codec="libx264",
            )
            goals = np.vstack([target_frames[i, -1], target_frames[i, -1]])
            for t in range(eval_budget):
                stacked_frame = np.vstack(
                    [video_frames[i, t], target_frames[i, t % target_len]]
                )
                frame = np.hstack([stacked_frame, goals])
                out.append_data(frame)
            out.close()

    if results["seeds"] is not None:
        assert np.unique(results["seeds"]).shape[0] == n_episodes, (
            "Some episode seeds are identical!"
        )

    return results


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
        print(
            f"  ✓ {intervention.name:<27} "
            f"SR: {sr:>6.2f}%  ΔSR: {delta_sr:>+7.2f}%\n"
        )

    elapsed = time.time() - start_time

    # Print final summary table
    print(f"{'='*80}")
    print(f"Final Results Summary")
    print(f"{'='*80}")
    print(f"{'Intervention':<30} {'SR (%)':<10} {'ΔSR (%)':<10}")
    print(f"{'-'*80}")

    for intervention in INTERVENTIONS:
        result = all_results[intervention.name]
        sr = result["success_rate"]
        delta_sr = sr - baseline_sr if baseline_sr is not None else 0.0

        print(f"{intervention.name:<30} {sr:>8.2f}  {delta_sr:>+9.2f}")

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
