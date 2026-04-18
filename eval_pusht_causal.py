"""PushT closed-loop causal evaluation.

This script evaluates LeWM on PushT under static counterfactual interventions
and reports:
  - success rate (SR)
  - final block pose error
  - delta SR against the baseline distribution

The rollout is closed-loop: a policy is queried at every step until the episode
terminates or the evaluation budget is exhausted.

Interventions covered:
  A1. do(init_pos)          - move the T-block to boundary or near-boundary positions
  A2. do(init_angle)        - rotate the T-block by mild or large angles at reset
  B1. do(target_pose)       - shift only the T-block goal position
  B2. do(target_angle)      - rotate only the T-block goal angle
  C.  do(visual_distractor) - change the T-block color
  D.  do(object_scale)      - resize the T-block (20 / 60)

Usage
-----
python eval_pusht_new.py policy=pusht/lewm

By default the script reuses the PushT evaluation config from
config/eval/pusht.yaml.
"""
from __future__ import annotations

import json
import os
import sys
import time
import types
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
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

MILD_INIT_POSITIONS = [
    np.array([90.0, 90.0], dtype=np.float64),
    np.array([110.0, 110.0], dtype=np.float64),
]

TARGET_POSE_POSITIONS = [
    np.array([80.0, 80.0], dtype=np.float64),
    np.array([80.0, 430.0], dtype=np.float64),
    np.array([430.0, 80.0], dtype=np.float64),
    np.array([430.0, 430.0], dtype=np.float64),
    np.array([80.0, 256.0], dtype=np.float64),
    np.array([430.0, 256.0], dtype=np.float64),
]

MICRO_TARGET_POSE_POSITIONS = [
    np.array([261.0, 261.0], dtype=np.float64),
]

TINY_TARGET_POSE_POSITIONS = [
    np.array([268.0, 268.0], dtype=np.float64),
]

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
    InterventionSpec("do_init_pos_mild", "init_pos", MILD_INIT_POSITIONS),
    InterventionSpec("do_init_pos_boundary", "init_pos"),
    InterventionSpec("do_init_angle_10", "init_angle", np.pi / 18),
    InterventionSpec("do_init_angle_20", "init_angle", np.pi / 9),
    InterventionSpec("do_init_angle_30", "init_angle", np.pi / 6),
    InterventionSpec("do_init_angle_45", "init_angle", np.pi / 4),
    InterventionSpec(
        "do_target_pose_micro_shift",
        "target_pose",
        {"positions": MICRO_TARGET_POSE_POSITIONS, "keep_angle": True},
    ),
    InterventionSpec(
        "do_target_pose_tiny_shift",
        "target_pose",
        {"positions": TINY_TARGET_POSE_POSITIONS, "keep_angle": True},
    ),
    InterventionSpec("do_target_pose", "target_pose"),
    InterventionSpec("do_target_angle_10", "target_angle", np.pi / 18),
    InterventionSpec("do_target_angle_20", "target_angle", np.pi / 9),
    InterventionSpec("do_target_angle_30", "target_angle", np.pi / 6),
    InterventionSpec("do_target_angle_45", "target_angle", np.pi / 4),
    InterventionSpec("do_checkerboard", "background_checkerboard"),
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
    # CRITICAL FIX: Step physics to update rendering state
    env.space.step(env.dt)


def _episode_variation(
    spec: InterventionSpec,
    init_state: np.ndarray,
    goal_state: np.ndarray,
    episode_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict | None, np.ndarray]:
    """Return modified init/goal states, reset options, and goal pose.

    Target interventions change only the T-block goal components; the visual and
    scale interventions only alter rendering / geometry.
    """

    init_state_mod = np.array(init_state, dtype=np.float64, copy=True)
    goal_state_mod = np.array(goal_state, dtype=np.float64, copy=True)
    reset_options: dict | None = None

    if spec.kind == "baseline":
        pass

    elif spec.kind == "init_pos":
        pool = spec.value if spec.value is not None else BOUNDARY_INIT_POSITIONS
        init_state_mod[2:4] = _pick_from_pool(pool, episode_seed)

    elif spec.kind == "init_angle":
        init_state_mod[4] = _wrap_angle(goal_state_mod[4] + float(spec.value))

    elif spec.kind == "target_pose":
        # Only move the T-block goal position; keep agent goal and target angle.
        keep_angle = False
        if isinstance(spec.value, dict):
            pool = spec.value.get("positions", TARGET_POSE_POSITIONS)
            keep_angle = bool(spec.value.get("keep_angle", False))
        else:
            pool = spec.value if spec.value is not None else TARGET_POSE_POSITIONS
        goal_state_mod[2:4] = _pick_from_pool(pool, episode_seed)
        if keep_angle:
            goal_state_mod[4] = goal_state[4]

    elif spec.kind == "target_angle":
        # Only rotate the T-block goal angle; keep agent goal and target position.
        goal_state_mod[4] = _wrap_angle(goal_state_mod[4] + float(spec.value))

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


def _patch_lewm_batch_criterion(model: torch.nn.Module) -> torch.nn.Module:
    """Patch installed LeWM so CEM env batches >1 broadcast goal embeddings correctly."""

    def criterion(self, info_dict: dict):
        pred_emb = info_dict["predicted_emb"]  # (B, S, T, D)
        goal_emb = info_dict["goal_emb"]  # (B, T, D) in the installed package

        if goal_emb.ndim == 3:
            goal_emb = goal_emb.unsqueeze(1)  # (B, 1, T, D)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
        return F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))

    model.criterion = types.MethodType(criterion, model)
    return model


def _patch_lewm_cached_get_cost(model: torch.nn.Module) -> torch.nn.Module:
    """Cache goal embeddings within a single replan to avoid repeated ViT encodes."""

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal_cache = getattr(self, "_cached_goal_emb", None)
        if goal_cache is None:
            goal_cache = {}
            self._cached_goal_emb = goal_cache

        goal_tensor = info_dict["goal"]
        cache_key = (
            goal_tensor.data_ptr(),
            tuple(goal_tensor.shape),
            goal_tensor.device.type,
            goal_tensor.device.index,
        )
        goal_emb = goal_cache.get(cache_key)
        if goal_emb is None:
            goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
            goal["pixels"] = goal["goal"]

            goal_aliases = {}
            for k in list(goal.keys()):
                if k.startswith("goal_"):
                    goal_aliases[k[len("goal_") :]] = goal[k]
            goal.update(goal_aliases)
            goal.pop("action", None)

            goal = self.encode(goal)
            goal_emb = goal["emb"].detach()
            goal_cache[cache_key] = goal_emb

        info_dict["goal_emb"] = goal_emb
        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)

    model._cached_goal_emb = {}
    model.get_cost = types.MethodType(get_cost, model)
    return model


def _patch_policy_clear_goal_cache(policy: swm.policy.WorldModelPolicy) -> swm.policy.WorldModelPolicy:
    """Clear cached goal embeddings before each new CEM replan."""

    original_get_action = policy.get_action

    def get_action(self, info_dict: dict, **kwargs):
        if len(self._action_buffer) == 0 and hasattr(self.solver.model, "_cached_goal_emb"):
            self.solver.model._cached_goal_emb = {}
        return original_get_action(info_dict, **kwargs)

    policy.get_action = types.MethodType(get_action, policy)
    return policy


def _patch_fast_world_model_policy(
    policy: swm.policy.WorldModelPolicy, img_size: int
) -> swm.policy.WorldModelPolicy:
    """Replace per-image transforms with batched resize/normalize for planning."""

    policy._fast_img_size = int(img_size)
    policy._fast_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    policy._fast_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)

    def _prepare_image_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.ndim != 5:
            return x

        if x.shape[-1] == 3:
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.float()
        if x.numel() > 0 and x.max() > 1.0:
            x = x / 255.0

        h, w = x.shape[-2], x.shape[-1]
        if h != self._fast_img_size or w != self._fast_img_size:
            et = x.shape[:2]
            x = x.view(-1, *x.shape[-3:])
            x = F.interpolate(
                x,
                size=(self._fast_img_size, self._fast_img_size),
                mode="bilinear",
                align_corners=False,
            )
            x = x.view(*et, *x.shape[-3:])

        mean = self._fast_mean.to(device=x.device, dtype=x.dtype)
        std = self._fast_std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert "pixels" in info_dict, "'pixels' must be provided in info_dict"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"

        info_dict = dict(info_dict)

        for k, v in list(info_dict.items()):
            if not isinstance(v, (np.ndarray, np.generic)):
                continue
            if hasattr(self, "process") and k in self.process:
                shape = v.shape
                if len(shape) > 2:
                    v_flat = v.reshape(-1, *shape[2:])
                else:
                    v_flat = v
                v = self.process[k].transform(v_flat).reshape(shape)
            info_dict[k] = v

        if "goal" in info_dict:
            info_dict["goal_pixels"] = info_dict["goal"]

        for key in ("pixels", "goal", "goal_pixels"):
            if key in info_dict:
                info_dict[key] = self._prepare_image_tensor(info_dict[key])

        for k, v in list(info_dict.items()):
            if isinstance(v, (np.ndarray, np.generic)) and v.dtype.kind not in "USO":
                info_dict[k] = torch.from_numpy(v)

        device = next(self.solver.model.parameters()).device
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                info_dict[k] = v.to(device, non_blocking=True)

        if len(self._action_buffer) == 0:
            if hasattr(self.solver.model, "_cached_goal_emb"):
                self.solver.model._cached_goal_emb = {}
            outputs = self.solver(info_dict, init_action=self._next_init)

            actions = outputs["actions"]
            keep_horizon = self.cfg.receding_horizon
            plan = actions[:, :keep_horizon]
            rest = actions[:, keep_horizon:]
            self._next_init = rest if self.cfg.warm_start else None

            plan = plan.reshape(self.env.num_envs, self.flatten_receding_horizon, -1)
            self._action_buffer.extend(plan.transpose(0, 1))

        action = self._action_buffer.popleft()
        action = action.reshape(*self.env.action_space.shape)
        action = action.numpy()

        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)

        return action

    policy._prepare_image_tensor = types.MethodType(_prepare_image_tensor, policy)
    policy.get_action = types.MethodType(get_action, policy)
    return policy


def _patch_cem_solver_no_step_sync(solver) -> object:
    """Delay CPU transfers in CEM until the end of each env sub-batch."""

    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        with torch.inference_mode():
            start_time = time.time()
            outputs = {"costs": [], "mean": [], "var": []}

            mean, var = self.init_action_distrib(init_action)
            mean = mean.to(self.device)
            var = var.to(self.device)

            total_envs = self.n_envs
            for start_idx in range(0, total_envs, self.batch_size):
                end_idx = min(start_idx + self.batch_size, total_envs)
                current_bs = end_idx - start_idx

                batch_mean = mean[start_idx:end_idx]
                batch_var = var[start_idx:end_idx]

                expanded_infos = {}
                for k, v in info_dict.items():
                    v_batch = v[start_idx:end_idx]
                    if torch.is_tensor(v):
                        v_batch = v_batch.unsqueeze(1)
                        v_batch = v_batch.expand(current_bs, self.num_samples, *v_batch.shape[2:])
                    elif isinstance(v, np.ndarray):
                        v_batch = np.repeat(v_batch[:, None, ...], self.num_samples, axis=1)
                    expanded_infos[k] = v_batch

                final_batch_cost = None
                for _ in range(self.n_steps):
                    candidates = torch.randn(
                        current_bs,
                        self.num_samples,
                        self.horizon,
                        self.action_dim,
                        generator=self.torch_gen,
                        device=self.device,
                    )
                    candidates = candidates * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                    candidates[:, 0] = batch_mean

                    costs = self.model.get_cost(expanded_infos.copy(), candidates)
                    topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)

                    batch_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, self.topk)
                    topk_candidates = candidates[batch_indices, topk_inds]

                    batch_mean = topk_candidates.mean(dim=1)
                    batch_var = topk_candidates.std(dim=1)
                    final_batch_cost = topk_vals.mean(dim=1)

                mean[start_idx:end_idx] = batch_mean
                var[start_idx:end_idx] = batch_var
                outputs["costs"].extend(final_batch_cost.cpu().tolist())

            outputs["actions"] = mean.detach().cpu()
            outputs["mean"] = [mean.detach().cpu()]
            outputs["var"] = [var.detach().cpu()]
            print(f"CEM solve time: {time.time() - start_time:.4f} seconds")
            return outputs

    solver.solve = types.MethodType(solve, solver)
    return solver


def patch_pusht_visual_interventions() -> None:
    """Add background visual interventions to PushT via an env callable."""
    from stable_worldmodel.envs.pusht.env import PushTEnv

    if getattr(PushTEnv, "_lewm_visual_patch_installed", False):
        return

    original_render_frame = PushTEnv._render_frame

    def _render_frame(self, mode):
        img = original_render_frame(self, mode)

        if not getattr(self, "_use_checkerboard", False):
            return img

        bg = np.asarray(
            self.variation_space["background"]["color"].value, dtype=np.int16
        )
        img_int = img.astype(np.int16)
        bg_mask = np.all(np.abs(img_int - bg[None, None, :]) <= 5, axis=-1)

        h, w = img.shape[:2]
        tile = max(1, h // 8)
        y_idx = np.arange(h) // tile
        x_idx = np.arange(w) // tile
        checker = (y_idx[:, None] + x_idx[None, :]) % 2

        dark = np.array([80, 80, 80], dtype=np.uint8)
        light = np.array([200, 200, 200], dtype=np.uint8)
        pattern = np.where(checker[..., None] == 0, dark, light)

        out = img.copy()
        out[bg_mask] = pattern[bg_mask]
        return out

    def _set_visual_intervention(
        self,
        bg_color=None,
        checkerboard=False,
    ):
        self._use_checkerboard = bool(checkerboard)

        if bg_color is not None:
            self.variation_space["background"]["color"].set_value(
                np.asarray(bg_color, dtype=np.uint8)
            )

    PushTEnv._render_frame = _render_frame
    PushTEnv._set_visual_intervention = _set_visual_intervention
    PushTEnv._lewm_visual_patch_installed = True


def _make_visual_intervention_callable(
    bg_color: list[int] | None = None,
    checkerboard: bool = False,
) -> dict:
    return {
        "method": "_set_visual_intervention",
        "args": {
            "bg_color": {"value": bg_color, "in_dataset": False},
            "checkerboard": {"value": checkerboard, "in_dataset": False},
        },
    }


def _callables_for_intervention(
    base_callables: list[dict] | None,
    spec: InterventionSpec,
) -> list[dict] | None:
    callables = list(base_callables or [])

    if spec.kind == "background_checkerboard":
        callables.append(
            _make_visual_intervention_callable(
                bg_color=[255, 255, 255],
                checkerboard=True,
            )
        )

    return callables


def _make_policy(cfg, process, transform, device: torch.device):
    if cfg.policy == "random":
        return swm.policy.RandomPolicy(seed=cfg.seed)

    model = load_lewm_model(cfg.policy)
    if model is None:
        raise RuntimeError(f"Could not load model from policy={cfg.policy}")
    model = _patch_lewm_batch_criterion(model)
    model = _patch_lewm_cached_get_cost(model)
    model = model.to(device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    plan_cfg = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    solver = _patch_cem_solver_no_step_sync(solver)
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_cfg,
        process=process,
        transform=transform,
    )
    policy = _patch_policy_clear_goal_cache(policy)
    return _patch_fast_world_model_policy(policy, img_size=int(cfg.eval.img_size))


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
    if not bool(cfg.eval.get("save_video", False)):
        return []

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
        intervention_callables = _callables_for_intervention(base_callables, spec)
        fail_world.evaluate_from_dataset(
            intervention_dataset,
            start_steps=fail_start_idx,
            goal_offset_steps=int(cfg.eval.goal_offset_steps),
            eval_budget=int(cfg.eval.eval_budget),
            episodes_idx=fail_episodes,
            callables=intervention_callables,
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

    patch_pusht_visual_interventions()

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
        intervention_callables = _callables_for_intervention(base_callables, spec)
        metrics = world.evaluate_from_dataset(
            intervention_dataset,
            start_steps=eval_start_idx,
            goal_offset_steps=int(cfg.eval.goal_offset_steps),
            eval_budget=int(cfg.eval.eval_budget),
            episodes_idx=eval_episodes,
            callables=intervention_callables,
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
