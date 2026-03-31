"""
Counterfactual sample collection for all four intervention tasks.

Each task draws a random episode from the dataset, constructs factual and
counterfactual environment parameters, runs oracle rollouts, and checks
effectful-sample criteria.  A while-loop accumulates valid samples up to
target_effectful_n or max_attempts, whichever comes first.

Supported tasks
---------------
  do_location      – offset agent's initial position
  do_door_y        – change door centre y-coordinate
  do_wall_x        – change wall x-coordinate
  do_door_y_wall_x – jointly change door_y and wall_x
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from cf_oracle import oracle_rollout


# ── Factual param builder (shared across tasks) ───────────────────────────

def _base_fact_params(task_cfg: DictConfig, row: dict) -> dict:
    """Build factual env params from dataset row and task config."""
    params: dict = {}

    # Use dataset's initial agent position when available.
    if "state" in row:
        agent_pos = np.asarray(row["state"], dtype=np.float32)
        if agent_pos.ndim > 1:
            agent_pos = agent_pos[0]           # first frame
        params["agent_pos"] = agent_pos[:2]

    # Factual structural params (task-specific callers will override for CF).
    if hasattr(task_cfg, "wall_x_fact"):
        params["wall_x"] = int(task_cfg.wall_x_fact)
    if hasattr(task_cfg, "door_y_fact"):
        params["door_y"] = float(task_cfg.door_y_fact)

    return params


# ── Per-task CF param builders ────────────────────────────────────────────

def _build_cf_params_location(
    task_cfg: DictConfig, fact_params: dict, rng: np.random.Generator
) -> list[dict]:
    """
    Returns one CF-params dict per configured offset.
    do_location offsets the agent starting position.
    """
    offsets = task_cfg.loc_prime_offsets_xy  # list of [dx, dy]
    result = []
    for dx, dy in offsets:
        cf = dict(fact_params)
        if "agent_pos" in cf:
            new_pos = cf["agent_pos"].copy()
            new_pos[0] += float(dx)
            new_pos[1] += float(dy)
            cf["agent_pos"] = new_pos
        result.append(cf)
    return result


def _build_cf_params_door_y(
    task_cfg: DictConfig, fact_params: dict, rng: np.random.Generator
) -> dict:
    candidates = [
        float(v) for v in task_cfg.door_y_cf
        if abs(float(v) - float(task_cfg.door_y_fact)) > float(task_cfg.skip_same_value_eps)
    ]
    door_y_cf = float(rng.choice(candidates))
    cf = dict(fact_params)
    cf["door_y"] = door_y_cf
    return cf


def _build_cf_params_wall_x(
    task_cfg: DictConfig, fact_params: dict, rng: np.random.Generator
) -> dict:
    candidates = [
        int(v) for v in task_cfg.wall_x_cf
        if abs(int(v) - int(task_cfg.wall_x_fact)) > float(task_cfg.skip_same_value_eps)
    ]
    wall_x_cf = int(rng.choice(candidates))
    cf = dict(fact_params)
    cf["wall_x"] = wall_x_cf
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
    else:  # full_grid – pick one pair at random
        door_y_cf = float(rng.choice(door_candidates))
        wall_x_cf = int(rng.choice(wall_candidates))

    cf = dict(fact_params)
    cf["door_y"] = door_y_cf
    cf["wall_x"] = wall_x_cf
    return cf


# ── Effectful-pair checker ────────────────────────────────────────────────

def _is_effectful(
    fact_oracle: dict,
    cf_oracle: dict,
    task_cfg: DictConfig,
) -> bool:
    """Return True if the pair passes all effectful-sample filters."""
    # MSE between oracle position trajectories (truncate to shorter).
    fp = fact_oracle["positions"]
    cp = cf_oracle["positions"]
    min_len = min(len(fp), len(cp))
    effect_mse = float(np.mean((fp[:min_len] - cp[:min_len]) ** 2))

    if effect_mse < float(task_cfg.min_oracle_effect_mse):
        return False

    if task_cfg.require_oracle_pass_change:
        if fact_oracle["through_door"] == cf_oracle["through_door"]:
            return False

    return True


# ── Single-pair attempt ───────────────────────────────────────────────────

def _attempt_one(
    task_name: str,
    task_cfg: DictConfig,
    dataset,
    rng: np.random.Generator,
) -> dict | None:
    """
    Draw one episode, build fact/CF params, run oracle rollouts, filter.
    Returns a sample dict on success, None otherwise.
    """
    idx = int(rng.integers(0, len(dataset)))
    row = dataset[idx]

    # actions shape: (T, action_dim) – dataset rows are single steps,
    # so we load a sequence via the dataset's built-in windowing.
    actions = np.asarray(row.get("action", []), dtype=np.float32)
    if actions.ndim == 1:
        # Single action row – not useful for rollout; skip.
        return None

    fact_params = _base_fact_params(task_cfg, row)

    if task_name == "do_location":
        # do_location yields multiple CF offsets per episode.
        t_intervene_candidates = list(task_cfg.t_intervene)
        t = int(rng.choice(t_intervene_candidates))
        # Clamp t to available trajectory length.
        t = min(t, len(actions) - 1)
        cf_params_list = _build_cf_params_location(task_cfg, fact_params, rng)
        # Pick one offset at random for this attempt.
        cf_params = cf_params_list[int(rng.integers(0, len(cf_params_list)))]
        # For do_location, the "intervention time" means we run factual up to t
        # then switch agent position; here we encode it in agent_pos directly.

    elif task_name == "do_door_y":
        cf_params = _build_cf_params_door_y(task_cfg, fact_params, rng)

    elif task_name == "do_wall_x":
        cf_params = _build_cf_params_wall_x(task_cfg, fact_params, rng)

    elif task_name == "do_door_y_wall_x":
        cf_params = _build_cf_params_door_y_wall_x(task_cfg, fact_params, rng)

    else:
        raise ValueError(f"Unknown task: {task_name}")

    fact_oracle = oracle_rollout(fact_params, actions)
    cf_oracle   = oracle_rollout(cf_params,   actions)

    if not _is_effectful(fact_oracle, cf_oracle, task_cfg):
        return None

    fp = fact_oracle["positions"]
    cp = cf_oracle["positions"]
    min_len = min(len(fp), len(cp))
    effect_mse = float(np.mean((fp[:min_len] - cp[:min_len]) ** 2))

    return {
        "row":         row,
        "actions":     actions,
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
    dataset,
    rng: np.random.Generator,
) -> tuple[list[dict], int, float]:
    """
    While-loop sampler: keep trying until target_effectful_n valid pairs
    are found or max_attempts is reached.

    Returns:
        samples      : list of accepted sample dicts
        n_attempts   : total attempts made
        accept_rate  : len(samples) / n_attempts
    """
    target_n   = int(task_cfg.target_effectful_n)
    max_tries  = int(task_cfg.max_attempts)

    samples: list[dict] = []
    n_attempts = 0

    while len(samples) < target_n:
        if n_attempts >= max_tries:
            break
        n_attempts += 1
        pair = _attempt_one(task_name, task_cfg, dataset, rng)
        if pair is not None:
            samples.append(pair)

    accept_rate = len(samples) / max(n_attempts, 1)
    return samples, n_attempts, accept_rate
