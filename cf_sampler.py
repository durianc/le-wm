"""
Counterfactual sample collection for all four intervention tasks.

Each task draws a random *episode* from the dataset (via ep_offset/ep_len),
constructs factual and counterfactual environment parameters, runs oracle
rollouts, and checks effectful-sample criteria.  A while-loop accumulates
valid samples up to target_effectful_n or max_attempts, whichever comes first.

Supported tasks
---------------
  do_location      – offset agent's initial position
  do_door_y        – change door centre y-coordinate
  do_wall_x        – change wall x-coordinate
  do_door_y_wall_x – jointly change door_y and wall_x

Episode-level sampling
----------------------
The HDF5Dataset is step-indexed (len = total steps), so indexing it directly
yields a single-step row with a 1-step action — useless for oracle rollout.
Instead we read ep_offset / ep_len from the .h5 file directly and draw
random *episode* indices, then load the full action + proprio sequence.

Factual trajectory pre-filtering
---------------------------------
For structural interventions (do_door_y, do_wall_x, do_door_y_wall_x) we
only accept episodes where the agent actually passes near the wall during the
factual rollout (x ∈ [wall_x_fact - wall_margin, wall_x_fact + wall_margin]).
This ensures the intervention has a chance to flip through_door and lifts the
accept rate from <1 % to >10 %.
"""

from __future__ import annotations

import h5py
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from cf_oracle import oracle_rollout


# ── H5 episode loader ─────────────────────────────────────────────────────

def _load_h5_index(h5_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load episode index arrays and full proprio/action arrays from HDF5.

    Returns:
        ep_offsets : (N_ep,) int  — start step index for each episode
        ep_lens    : (N_ep,) int  — length of each episode
        proprio    : (total_steps, 2) float32  — agent positions
        actions    : (total_steps, 2) float32  — actions
    """
    with h5py.File(h5_path, "r") as f:
        ep_offsets = f["ep_offset"][:]
        ep_lens    = f["ep_len"][:]
        proprio    = f["proprio"][:]
        actions    = f["action"][:]
    # action stored as (total_steps, 1, 2) in some versions — squeeze if needed
    if actions.ndim == 3:
        actions = actions[:, 0, :]
    return ep_offsets, ep_lens, proprio.astype(np.float32), actions.astype(np.float32)


# ── Factual param builder (shared across tasks) ───────────────────────────

def _base_fact_params(task_cfg: DictConfig, agent_pos: np.ndarray) -> dict:
    """Build factual env params from initial agent position and task config."""
    params: dict = {"agent_pos": agent_pos.copy()}

    if hasattr(task_cfg, "wall_x_fact"):
        params["wall_x"] = int(task_cfg.wall_x_fact)
    if hasattr(task_cfg, "door_y_fact"):
        params["door_y"] = float(task_cfg.door_y_fact)

    return params


# ── Per-task CF param builders ────────────────────────────────────────────

def _build_cf_params_location(
    task_cfg: DictConfig, fact_params: dict, rng: np.random.Generator
) -> dict:
    offsets = task_cfg.loc_prime_offsets_xy  # list of [dx, dy]
    dx, dy = offsets[int(rng.integers(0, len(offsets)))]
    cf = dict(fact_params)
    new_pos = cf["agent_pos"].copy()
    new_pos[0] += float(dx)
    new_pos[1] += float(dy)
    cf["agent_pos"] = new_pos
    return cf


def _build_cf_params_door_y(
    task_cfg: DictConfig, fact_params: dict, rng: np.random.Generator
) -> dict:
    candidates = [
        float(v) for v in task_cfg.door_y_cf
        if abs(float(v) - float(task_cfg.door_y_fact)) > float(task_cfg.skip_same_value_eps)
    ]
    cf = dict(fact_params)
    cf["door_y"] = float(rng.choice(candidates))
    return cf


def _build_cf_params_wall_x(
    task_cfg: DictConfig, fact_params: dict, rng: np.random.Generator
) -> dict:
    candidates = [
        int(v) for v in task_cfg.wall_x_cf
        if abs(int(v) - int(task_cfg.wall_x_fact)) > float(task_cfg.skip_same_value_eps)
    ]
    cf = dict(fact_params)
    cf["wall_x"] = int(rng.choice(candidates))
    return cf


def _build_cf_params_door_y_wall_x(
    task_cfg: DictConfig, fact_params: dict, rng: np.random.Generator
) -> dict:
    door_candidates = [
        float(v) for v in task_cfg.door_y_cf
        if abs(float(v) - float(task_cfg.door_y_fact)) > float(task_cfg.skip_same_value_eps)
    ]
    wall_candidates = [
        int(v) for v in task_cfg.wall_x_cf
        if abs(int(v) - int(task_cfg.wall_x_fact)) > float(task_cfg.skip_same_value_eps)
    ]

    if task_cfg.pair_sampling == "random_k":
        k = int(task_cfg.k_pairs_per_sample)
        door_choices = rng.choice(door_candidates, size=k, replace=True)
        wall_choices = rng.choice(wall_candidates, size=k, replace=True)
        idx = rng.integers(0, k)
        door_y_cf = float(door_choices[idx])
        wall_x_cf = int(wall_choices[idx])
    else:
        door_y_cf = float(rng.choice(door_candidates))
        wall_x_cf = int(rng.choice(wall_candidates))

    cf = dict(fact_params)
    cf["door_y"] = door_y_cf
    cf["wall_x"] = wall_x_cf
    return cf


# ── Factual trajectory pre-filter ─────────────────────────────────────────

def _episode_near_wall(
    ep_proprio: np.ndarray,
    wall_x: float,
    margin: float,
) -> bool:
    """Return True if the episode trajectory passes within margin of wall_x."""
    xs = ep_proprio[:, 0]
    return bool(np.any((xs >= wall_x - margin) & (xs <= wall_x + margin)))


# ── Effectful-pair checker ────────────────────────────────────────────────

def _is_effectful(
    fact_oracle: dict,
    cf_oracle: dict,
    task_cfg: DictConfig,
) -> bool:
    """Return True if the pair passes all effectful-sample filters."""
    if task_cfg.require_oracle_pass_change:
        if fact_oracle["through_door"] == cf_oracle["through_door"]:
            return False

    min_mse = float(getattr(task_cfg, "min_oracle_effect_mse", 0.0))
    if min_mse > 0.0:
        fp = fact_oracle["positions"]
        cp = cf_oracle["positions"]
        min_len = min(len(fp), len(cp))
        effect_mse = float(np.mean((fp[:min_len] - cp[:min_len]) ** 2))
        if effect_mse < min_mse:
            return False

    return True


# ── Single-pair attempt ───────────────────────────────────────────────────

def _attempt_one(
    task_name: str,
    task_cfg: DictConfig,
    ep_offsets: np.ndarray,
    ep_lens: np.ndarray,
    proprio: np.ndarray,
    actions: np.ndarray,
    rng: np.random.Generator,
    wall_margin: float,
) -> dict | None:
    """
    Draw one episode, build fact/CF params, run oracle rollouts, filter.
    Returns a sample dict on success, None otherwise.
    """
    ep_idx  = int(rng.integers(0, len(ep_offsets)))
    off     = int(ep_offsets[ep_idx])
    l       = int(ep_lens[ep_idx])

    ep_proprio = proprio[off : off + l]   # (T, 2)
    ep_actions = actions[off : off + l]   # (T, 2)

    if l < 2:
        return None

    agent_pos = ep_proprio[0].copy()  # (2,) initial position
    fact_params = _base_fact_params(task_cfg, agent_pos)

    if task_name == "do_location":
        cf_params = _build_cf_params_location(task_cfg, fact_params, rng)

    elif task_name == "do_door_y":
        # Pre-filter: episode must pass near the wall for door change to matter.
        wall_x = float(getattr(task_cfg, "wall_x_fact", 112))
        if not _episode_near_wall(ep_proprio, wall_x, wall_margin):
            return None
        cf_params = _build_cf_params_door_y(task_cfg, fact_params, rng)

    elif task_name == "do_wall_x":
        wall_x = float(getattr(task_cfg, "wall_x_fact", 112))
        if not _episode_near_wall(ep_proprio, wall_x, wall_margin):
            return None
        cf_params = _build_cf_params_wall_x(task_cfg, fact_params, rng)

    elif task_name == "do_door_y_wall_x":
        wall_x = float(getattr(task_cfg, "wall_x_fact", 112))
        if not _episode_near_wall(ep_proprio, wall_x, wall_margin):
            return None
        cf_params = _build_cf_params_door_y_wall_x(task_cfg, fact_params, rng)

    else:
        raise ValueError(f"Unknown task: {task_name}")

    fact_oracle = oracle_rollout(fact_params, ep_actions)
    cf_oracle   = oracle_rollout(cf_params,   ep_actions)

    if not _is_effectful(fact_oracle, cf_oracle, task_cfg):
        return None

    fp = fact_oracle["positions"]
    cp = cf_oracle["positions"]
    min_len = min(len(fp), len(cp))
    effect_mse = float(np.mean((fp[:min_len] - cp[:min_len]) ** 2))

    return {
        "ep_idx":      ep_idx,
        "actions":     ep_actions,
        "fact_params": fact_params,
        "cf_params":   cf_params,
        "fact_oracle": fact_oracle,
        "cf_oracle":   cf_oracle,
        "pass_changed": fact_oracle["through_door"] != cf_oracle["through_door"],
        "effect_mse_oracle": effect_mse,
    }


# ── Public: collect effectful samples ────────────────────────────────────

def collect_effectful_samples(
    task_name: str,
    task_cfg: DictConfig,
    h5_path: str,
    rng: np.random.Generator,
) -> tuple[list[dict], int, float]:
    """
    Episode-level sampler: draw full episodes from the HDF5 file directly,
    apply factual pre-filtering, run oracle rollouts, and collect effectful pairs.

    Args:
        task_name : one of the four CF task names
        task_cfg  : task-specific Hydra config node
        h5_path   : path to the tworoom .h5 dataset file
        rng       : numpy random generator

    Returns:
        samples      : list of accepted sample dicts
        n_attempts   : total episode draws attempted
        accept_rate  : len(samples) / n_attempts
    """
    target_n  = int(task_cfg.target_effectful_n)
    max_tries = int(task_cfg.max_attempts)
    wall_margin = float(getattr(task_cfg, "wall_proximity_margin", 22.0))

    ep_offsets, ep_lens, proprio, actions = _load_h5_index(h5_path)

    samples: list[dict] = []
    n_attempts = 0

    pbar = tqdm(
        total=target_n,
        desc=task_name,
        unit="sample",
        postfix={"attempts": 0, "rate": "0.000"},
    )
    while len(samples) < target_n:
        if n_attempts >= max_tries:
            break
        n_attempts += 1
        pair = _attempt_one(
            task_name, task_cfg,
            ep_offsets, ep_lens, proprio, actions,
            rng, wall_margin,
        )
        if pair is not None:
            samples.append(pair)
            pbar.update(1)
            pbar.set_postfix(
                attempts=n_attempts,
                rate=f"{len(samples) / n_attempts:.3f}",
            )
        elif n_attempts % 50 == 0:
            pbar.set_postfix(
                attempts=n_attempts,
                rate=f"{len(samples) / n_attempts:.3f}",
            )
    pbar.close()

    accept_rate = len(samples) / max(n_attempts, 1)
    return samples, n_attempts, accept_rate
