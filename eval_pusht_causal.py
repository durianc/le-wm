"""PushT closed-loop causal evaluation.

This script evaluates LeWM on PushT under static counterfactual interventions
and reports:
  - success rate (SR)
  - final block pose error
  - delta SR against the baseline distribution

The rollout is closed-loop: a policy is queried at every step until the episode
terminates or the evaluation budget is exhausted.

Interventions covered:
  A1. do(init_pos)          - move the T-block to boundary positions unseen in training
  A2. do(init_angle)        - rotate the T-block by 90 or 180 degrees at reset
  B.  do(target_pose)       - shift the goal pose away from the default center
  C.  do(visual_distractor) - change the T-block color
  D.  do(object_scale)      - resize the T-block (20 / 60)

Usage
-----
python eval_pusht_causal.py policy=pusht/lewm

By default the script reuses the PushT evaluation config from
config/eval/pusht.yaml.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing
from tqdm import tqdm

import stable_worldmodel as swm

from eval import get_dataset, get_episodes_length, img_transform, load_lewm_model

warnings.filterwarnings("ignore", category=UserWarning)


BOUNDARY_INIT_POSITIONS = [
    np.array([60.0, 60.0], dtype=np.float64),
    np.array([60.0, 450.0], dtype=np.float64),
    np.array([450.0, 60.0], dtype=np.float64),
    np.array([450.0, 450.0], dtype=np.float64),
]

TARGET_POSE_POSITIONS = [
    np.array([80.0, 80.0], dtype=np.float64),
    np.array([80.0, 430.0], dtype=np.float64),
    np.array([430.0, 80.0], dtype=np.float64),
    np.array([430.0, 430.0], dtype=np.float64),
    np.array([80.0, 256.0], dtype=np.float64),
    np.array([430.0, 256.0], dtype=np.float64),
]

TARGET_POSE_ANGLES = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]

VISUAL_DISTRACTOR_COLORS = [
    np.array([255, 96, 0], dtype=np.uint8),
    np.array([255, 0, 128], dtype=np.uint8),
    np.array([0, 180, 255], dtype=np.uint8),
]


@dataclass(frozen=True)
class InterventionSpec:
    name: str
    kind: str
    value: object | None = None


INTERVENTIONS: list[InterventionSpec] = [
    InterventionSpec("baseline", "baseline"),
    InterventionSpec("do_init_pos_boundary", "init_pos"),
    InterventionSpec("do_init_angle_90", "init_angle", np.pi / 2),
    InterventionSpec("do_init_angle_180", "init_angle", np.pi),
    InterventionSpec("do_target_pose", "target_pose"),
    InterventionSpec("do_visual_distractor", "visual_distractor"),
    InterventionSpec("do_object_scale_20", "object_scale", 20.0),
    InterventionSpec("do_object_scale_60", "object_scale", 60.0),
]


def parse_args() -> dict:
    cfg = {
        "policy": "pusht/lewm",
        "dataset": "pusht_expert_train",
        "num_episodes": 50,
        "seed": 42,
        "output": "logs_eval/pusht_causal_results.json",
    }
    for arg in sys.argv[1:]:
        if arg in ("--help", "-h"):
            print(__doc__)
            raise SystemExit(0)
        if "=" in arg:
            key, value = arg.split("=", 1)
            if key in ("num_episodes", "seed"):
                cfg[key] = int(value)
            else:
                cfg[key] = value
    return cfg


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _batch_value(value):
    if isinstance(value, str):
        return value
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value[None, None, ...]
    if np.isscalar(value):
        return np.asarray([[value]])
    return value


def _wrap_angle(angle: float) -> float:
    return float(angle % (2 * np.pi))


def _pick_from_pool(pool: list[np.ndarray], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, len(pool)))
    return np.array(pool[idx], dtype=np.float64)


def _build_dataset(cfg, dataset_ref: str):
    path = Path(dataset_ref)
    if path.exists() and path.suffix == ".h5":
        return swm.data.HDF5Dataset(
            path.stem,
            keys_to_cache=cfg.dataset.keys_to_cache,
            cache_dir=path.parent,
        )
    return get_dataset(cfg, dataset_ref)


def _prepare_env_state(env, state: np.ndarray) -> None:
    env = env.unwrapped
    state = np.asarray(state, dtype=np.float64)
    env.agent.velocity = tuple(state[-2:])
    env.agent.position = state[:2].tolist()
    env.block.angle = float(state[4])
    env.block.position = state[2:4].tolist()


def _episode_variation(
    spec: InterventionSpec,
    init_state: np.ndarray,
    goal_state: np.ndarray,
    episode_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict | None, np.ndarray]:
    """Return modified init/goal states, reset options, and goal pose.

    The target pose intervention changes the evaluation goal itself; the visual
    and scale interventions only alter rendering / geometry.
    """

    init_state_mod = np.array(init_state, dtype=np.float64, copy=True)
    goal_state_mod = np.array(goal_state, dtype=np.float64, copy=True)
    reset_options: dict | None = None

    if spec.kind == "baseline":
        pass

    elif spec.kind == "init_pos":
        init_state_mod[2:4] = _pick_from_pool(BOUNDARY_INIT_POSITIONS, episode_seed)

    elif spec.kind == "init_angle":
        init_state_mod[4] = _wrap_angle(goal_state_mod[4] + float(spec.value))

    elif spec.kind == "target_pose":
        # Keep the agent state from the current start and only move the target pose.
        goal_state_mod = np.array(init_state_mod, dtype=np.float64, copy=True)
        goal_state_mod[2:4] = _pick_from_pool(TARGET_POSE_POSITIONS, episode_seed)
        goal_state_mod[4] = _wrap_angle(
            TARGET_POSE_ANGLES[episode_seed % len(TARGET_POSE_ANGLES)]
        )

    elif spec.kind == "visual_distractor":
        color = VISUAL_DISTRACTOR_COLORS[episode_seed % len(VISUAL_DISTRACTOR_COLORS)]
        reset_options = {
            "variation": ["block.color"],
            "variation_values": {"block.color": color},
        }

    elif spec.kind == "object_scale":
        reset_options = {
            "variation": ["block.scale"],
            "variation_values": {"block.scale": float(spec.value)},
        }

    else:
        raise ValueError(f"Unknown intervention kind: {spec.kind}")

    goal_pose = np.array(
        [goal_state_mod[2], goal_state_mod[3], goal_state_mod[4]],
        dtype=np.float64,
    )
    return init_state_mod, goal_state_mod, reset_options, goal_pose


def _build_policy_inputs(env, current_pixels, goal_pixels, prev_action):
    # Keep the policy input minimal: WorldModelPolicy only requires pixels/goal.
    # LeWM-based cost models also expect an `action` key (history bootstrap).
    # Extra goal_* keys would be treated as image tensors by _prepare_info().
    return {
        "pixels": _batch_value(current_pixels),
        "goal": _batch_value(goal_pixels),
        "action": _batch_value(np.asarray(prev_action)),
    }


def _angle_error(cur_angle: float, goal_angle: float) -> float:
    diff = abs(cur_angle - goal_angle)
    return float(min(diff, 2 * np.pi - diff))


def _final_pose_metrics(cur_state: np.ndarray, goal_state: np.ndarray) -> dict:
    cur_state = np.asarray(cur_state, dtype=np.float64)
    goal_state = np.asarray(goal_state, dtype=np.float64)
    pos_error = float(np.linalg.norm(cur_state[2:4] - goal_state[2:4]))
    angle_error = _angle_error(cur_state[4], goal_state[4])
    combined = float(np.sqrt(pos_error**2 + angle_error**2))
    return {
        "pos_error_px": pos_error,
        "angle_error_rad": angle_error,
        "angle_error_deg": float(np.degrees(angle_error)),
        "combined_pose_error": combined,
    }


def _summarize_episode_result(metrics: list[dict], baseline_sr: float) -> dict:
    sr = float(np.mean([m["success"] for m in metrics])) if metrics else 0.0
    pos_errors = [m["final_pose"]["pos_error_px"] for m in metrics]
    angle_errors = [m["final_pose"]["angle_error_deg"] for m in metrics]
    combined_errors = [m["final_pose"]["combined_pose_error"] for m in metrics]
    return {
        "success_rate": sr,
        "delta_sr": sr - baseline_sr,
        "mean_final_pos_error_px": float(np.mean(pos_errors)) if pos_errors else float("nan"),
        "mean_final_angle_error_deg": float(np.mean(angle_errors)) if angle_errors else float("nan"),
        "mean_final_pose_error": float(np.mean(combined_errors)) if combined_errors else float("nan"),
        "n_episodes": len(metrics),
        "n_success": int(sum(m["success"] for m in metrics)),
        "episodes": metrics,
    }


def _make_policy(cfg, process, transform, device: torch.device):
    if cfg.policy == "random":
        return swm.policy.RandomPolicy(seed=cfg.seed)

    model = load_lewm_model(cfg.policy)
    if model is None:
        raise RuntimeError(f"Could not load model from policy={cfg.policy}")
    model = model.to(device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    plan_cfg = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_cfg,
        process=process,
        transform=transform,
    )


def _sample_eval_episodes(dataset, num_eval: int, seed: int, goal_offset_steps: int):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - goal_offset_steps - 1
    max_start_dict = {ep: max_start_idx[i] for i, ep in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start_dict[ep] for ep in dataset.get_col_data(col_name)])
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    rng = np.random.default_rng(seed)
    sampled = rng.choice(len(valid_indices) - 1, size=num_eval, replace=False)
    sampled = np.sort(valid_indices[sampled])
    eval_episodes = dataset.get_row_data(sampled)[col_name].tolist()
    eval_start_idx = dataset.get_row_data(sampled)["step_idx"].tolist()
    return eval_episodes, eval_start_idx


def _intervention_tag(name: str) -> str:
    if name.startswith("do_"):
        return name[3:]
    return name


def _save_failed_episode_videos(
    cfg,
    policy,
    dataset,
    spec: InterventionSpec,
    eval_episodes: list[int],
    eval_start_idx: list[int],
    episode_successes: np.ndarray,
    base_callables: list[dict] | None,
    video_root: Path,
) -> list[str]:
    """Save videos only for failed (terminated==False) episodes of one intervention."""
    failed_idx = np.flatnonzero(~np.asarray(episode_successes, dtype=bool))
    if failed_idx.size == 0:
        return []

    fail_episodes = [int(eval_episodes[i]) for i in failed_idx.tolist()]
    fail_start_idx = [int(eval_start_idx[i]) for i in failed_idx.tolist()]

    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    world_cfg["num_envs"] = int(failed_idx.size)

    img_size = int(cfg.eval.img_size)
    fail_world = swm.World(**world_cfg, image_shape=(img_size, img_size))
    try:
        fail_world.set_policy(policy)

        if hasattr(policy, "_action_buffer") and policy._action_buffer is not None:
            policy._action_buffer.clear()
        if hasattr(policy, "_next_init"):
            policy._next_init = None

        tmp_dir = video_root / "_tmp" / spec.name
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for old in tmp_dir.glob("rollout_*.mp4"):
            old.unlink()

        intervention_dataset = _IntervenedDataset(dataset, spec, int(cfg.seed))
        fail_world.evaluate_from_dataset(
            intervention_dataset,
            start_steps=fail_start_idx,
            goal_offset_steps=int(cfg.eval.goal_offset_steps),
            eval_budget=int(cfg.eval.eval_budget),
            episodes_idx=fail_episodes,
            callables=base_callables,
            save_video=True,
            video_path=tmp_dir,
        )

        saved_paths: list[str] = []
        tag = _intervention_tag(spec.name)
        video_root.mkdir(parents=True, exist_ok=True)
        for old_named in video_root.glob(f"fail_{tag}_ep*.mp4"):
            old_named.unlink()
        for k in range(int(failed_idx.size)):
            src = tmp_dir / f"rollout_{k}.mp4"
            if not src.exists():
                continue
            dst = video_root / f"fail_{tag}_ep{k + 1:02d}.mp4"
            src.replace(dst)
            saved_paths.append(str(dst))
        return saved_paths
    finally:
        fail_world.close()


class _IntervenedDataset:
    """Dataset wrapper that applies per-episode interventions inside load_chunk()."""

    def __init__(self, base_dataset, spec: InterventionSpec, base_seed: int):
        self.base = base_dataset
        self.spec = spec
        self.base_seed = int(base_seed)
        self._extra_columns = []
        if spec.kind == "visual_distractor":
            self._extra_columns = ["variation.block.color"]
        elif spec.kind == "object_scale":
            self._extra_columns = ["variation.block.scale"]

    @property
    def column_names(self):
        cols = list(self.base.column_names)
        for col in self._extra_columns:
            if col not in cols:
                cols.append(col)
        return cols

    def get_col_data(self, col):
        return self.base.get_col_data(col)

    def get_row_data(self, row_idx):
        return self.base.get_row_data(row_idx)

    def load_chunk(self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray):
        chunk = self.base.load_chunk(episodes_idx, start, end)

        for i, ep in enumerate(chunk):
            if "state" not in ep:
                continue
            state = ep["state"]
            if torch.is_tensor(state):
                state_np = state.detach().cpu().numpy().copy()
                state_dtype = state.dtype
            else:
                state_np = np.asarray(state).copy()
                state_dtype = None

            init_state = np.asarray(state_np[0], dtype=np.float64)
            goal_state = np.asarray(state_np[-1], dtype=np.float64)
            episode_seed = (
                self.base_seed
                + int(episodes_idx[i]) * 1009
                + int(start[i]) * 17
            )
            init_state_mod, goal_state_mod, reset_options, _ = _episode_variation(
                self.spec, init_state, goal_state, episode_seed
            )

            state_np[0] = init_state_mod
            state_np[-1] = goal_state_mod

            if state_dtype is not None:
                ep["state"] = torch.as_tensor(state_np, dtype=state_dtype)
            else:
                ep["state"] = state_np

            if reset_options:
                variation_values = reset_options.get("variation_values", {})
                for key, value in variation_values.items():
                    v = np.asarray(value)
                    tiled = np.repeat(v[None, ...], state_np.shape[0], axis=0)
                    ep_key = f"variation.{key}"
                    ep[ep_key] = torch.from_numpy(tiled)

        return chunk


def _extract_final_pose_metrics(world, episode_successes: np.ndarray) -> list[dict]:
    metrics = []
    if "state" not in world.infos or "goal_state" not in world.infos:
        return metrics

    final_states = np.asarray(world.infos["state"])[:, -1]
    goal_states = np.asarray(world.infos["goal_state"])[:, -1]

    for i in range(final_states.shape[0]):
        metrics.append(
            {
                "success": bool(episode_successes[i]),
                "terminated": bool(episode_successes[i]),
                "truncated": False,
                "final_pose": _final_pose_metrics(final_states[i], goal_states[i]),
                "final_info": {},
            }
        )
    return metrics


def _run_single_episode(
    env,
    policy,
    init_state: np.ndarray,
    goal_state: np.ndarray,
    episode_seed: int,
    eval_budget: int,
    intervention: InterventionSpec,
) -> dict:
    env = env.unwrapped
    init_state_mod, goal_state_mod, reset_options, goal_pose = _episode_variation(
        intervention, init_state, goal_state, episode_seed
    )

    env.reset(seed=episode_seed, options=reset_options)

    # Make the rendered target consistent with the evaluation goal.
    env.goal_state = np.array(goal_state_mod, dtype=np.float64, copy=True)
    env.goal_pose = np.array(goal_pose, dtype=np.float64, copy=True)

    # Start from the intervened initial state.
    _prepare_env_state(env, init_state_mod)
    current_pixels = env.render()

    # Render the goal image from the evaluation goal state.
    _prepare_env_state(env, goal_state_mod)
    goal_pixels = env.render()

    # Restore the actual rollout state.
    _prepare_env_state(env, init_state_mod)

    prev_action = np.full_like(env.action_space.sample(), np.nan)
    info = _build_policy_inputs(env, current_pixels, goal_pixels, prev_action)

    if hasattr(policy, "_action_buffer") and policy._action_buffer is not None:
        policy._action_buffer.clear()
    if hasattr(policy, "_next_init"):
        policy._next_init = None

    terminated = False
    truncated = False
    last_obs = env._get_obs()
    last_info = env._get_info()

    for _ in range(eval_budget):
        action = policy.get_action(info)
        action_env = np.asarray(action)
        if action_env.ndim > 1 and action_env.shape[0] == 1:
            action_env = action_env[0]
        obs, reward, terminated, truncated, step_info = env.step(action_env)
        last_obs = obs["state"]
        last_info = step_info
        prev_action = action_env

        if terminated or truncated:
            break

        current_pixels = env.render()
        info = _build_policy_inputs(env, current_pixels, goal_pixels, prev_action)

    final_pose = _final_pose_metrics(last_obs, goal_state_mod)
    return {
        "success": bool(terminated),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_pose": final_pose,
        "final_info": {
            k: v.tolist() if hasattr(v, "tolist") else v for k, v in last_info.items()
        },
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_path = Path(__file__).parent / "config" / "eval" / "pusht.yaml"
    with hydra.initialize_config_dir(config_dir=str(cfg_path.parent), version_base=None):
        cfg = hydra.compose(config_name="pusht")

    OmegaConf.update(cfg, "policy", args["policy"], merge=True)
    OmegaConf.update(cfg, "eval.num_eval", args["num_episodes"], merge=True)
    OmegaConf.update(cfg, "seed", args["seed"], merge=True)

    print("=" * 72)
    print("PushT Closed-loop Counterfactual Evaluation")
    print("=" * 72)
    print(f"  policy      : {cfg.policy}")
    print(f"  dataset     : {args['dataset']}")
    print(f"  num_episodes: {cfg.eval.num_eval} per intervention")
    print(f"  eval_budget : {cfg.eval.eval_budget} steps")
    print(f"  device      : {device}")
    print()

    dataset = _build_dataset(cfg, args["dataset"])

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = processor

    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    # Align with eval.py: vectorized World + evaluate_from_dataset.
    cfg.world.max_episode_steps = 2 * int(cfg.eval.eval_budget)
    img_size = int(cfg.eval.img_size)
    world = swm.World(**cfg.world, image_shape=(img_size, img_size))

    policy = _make_policy(cfg, process, transform, device)
    world.set_policy(policy)

    eval_episodes, eval_start_idx = _sample_eval_episodes(
        dataset,
        num_eval=int(cfg.eval.num_eval),
        seed=int(cfg.seed),
        goal_offset_steps=int(cfg.eval.goal_offset_steps),
    )

    all_results: dict[str, dict] = {}
    total_t0 = time.time()
    base_callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
    model_name = Path(cfg.policy).name if cfg.policy != "random" else "random"
    failure_video_root = Path("logs_eval") / f"pusht_causal_fail_videos_{model_name}"

    for spec in INTERVENTIONS:
        print(f"\n[{spec.name}]")
        t0 = time.time()

        world.set_policy(policy)
        if hasattr(policy, "_action_buffer") and policy._action_buffer is not None:
            policy._action_buffer.clear()
        if hasattr(policy, "_next_init"):
            policy._next_init = None

        intervention_dataset = _IntervenedDataset(dataset, spec, int(cfg.seed))
        metrics = world.evaluate_from_dataset(
            intervention_dataset,
            start_steps=eval_start_idx,
            goal_offset_steps=int(cfg.eval.goal_offset_steps),
            eval_budget=int(cfg.eval.eval_budget),
            episodes_idx=eval_episodes,
            callables=base_callables,
            save_video=False,
        )

        episode_successes = np.asarray(metrics["episode_successes"], dtype=bool)
        saved_fail_videos = _save_failed_episode_videos(
            cfg=cfg,
            policy=policy,
            dataset=dataset,
            spec=spec,
            eval_episodes=eval_episodes,
            eval_start_idx=eval_start_idx,
            episode_successes=episode_successes,
            base_callables=base_callables,
            video_root=failure_video_root,
        )

        elapsed = time.time() - t0
        episode_metrics = _extract_final_pose_metrics(
            world, episode_successes
        )
        baseline_sr = all_results.get("baseline", {}).get("success_rate", float("nan"))
        all_results[spec.name] = _summarize_episode_result(episode_metrics, baseline_sr=baseline_sr)
        all_results[spec.name]["elapsed_seconds"] = elapsed
        all_results[spec.name]["n_failed_not_terminated"] = int((~episode_successes).sum())
        all_results[spec.name]["failure_videos"] = saved_fail_videos

        sr = all_results[spec.name]["success_rate"]
        ns = all_results[spec.name]["n_success"]
        print(
            f"  SR={sr:.3f}  ({ns}/{cfg.eval.num_eval})  "
            f"final_pose={all_results[spec.name]['mean_final_pose_error']:.3f}  "
            f"[{elapsed:.0f}s]"
        )
        if saved_fail_videos:
            print(
                f"  saved failure videos: {len(saved_fail_videos)} -> {failure_video_root}"
            )

    baseline_sr = all_results.get("baseline", {}).get("success_rate", float("nan"))
    if not np.isnan(baseline_sr):
        for entry in all_results.values():
            entry["delta_sr"] = entry["success_rate"] - baseline_sr

    elapsed_total = time.time() - total_t0

    W = 84
    print("\n" + "=" * W)
    print("PUSHT CLOSED-LOOP COUNTERFACTUAL RESULTS")
    print(f"  policy={cfg.policy}  dataset={args['dataset']}  n={cfg.eval.num_eval}  budget={cfg.eval.eval_budget}")
    print("=" * W)
    print(f"  {'Intervention':<24}  {'SR':>6}  {'ΔSR':>7}  {'FinalPose':>12}  {'n_success':>9}")
    print("  " + "-" * (W - 2))
    for name, entry in all_results.items():
        delta = entry["delta_sr"]
        dstr = f"{delta:+.3f}" if not np.isnan(delta) else "   n/a"
        print(
            f"  {name:<24}  {entry['success_rate']:>6.3f}  {dstr:>7}  "
            f"{entry['mean_final_pose_error']:>12.3f}  {entry['n_success']:>5}/{entry['n_episodes']}"
        )
    print()
    print(f"  Total time: {elapsed_total:.0f}s")
    print("=" * W)

    out_path = Path(args["output"])
    stem = out_path.stem
    if not stem.endswith(f"_{model_name}"):
        out_path = out_path.with_name(f"{stem}_{model_name}{out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result_obj = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "cli_args": args,
        "baseline_sr": baseline_sr,
        "metrics": all_results,
        "elapsed_seconds": elapsed_total,
    }
    solver = getattr(policy, "solver", None)
    if solver is not None and hasattr(solver, "solve_calls"):
        calls = int(getattr(solver, "solve_calls", 0))
        total = float(getattr(solver, "total_solve_time", 0.0))
        result_obj["planner_timing"] = {
            "solver": solver.__class__.__name__,
            "solve_calls": calls,
            "total_solve_seconds": total,
            "mean_solve_seconds": (total / calls) if calls > 0 else float("nan"),
            "last_solve_seconds": float(getattr(solver, "last_solve_time", float("nan"))),
        }
    with out_path.open("w") as f:
        json.dump(result_obj, f, indent=2, default=_json_default)
    print(f"Results written to {out_path}")

    world.close()


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


if __name__ == "__main__":
    main()
