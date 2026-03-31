"""
Oracle rollout for counterfactual evaluation.

Runs pure physics (no model) in TwoRoomCFEnv to obtain ground-truth
trajectories and door-crossing events.  Used for effectful-sample
filtering in cf_sampler.py.
"""

from __future__ import annotations

import numpy as np
import gymnasium

from cf_env import register_cf_env

register_cf_env()


def oracle_rollout(env_params: dict, actions: np.ndarray) -> dict:
    """Execute a physical rollout and detect door-crossing.

    Args:
        env_params: kwargs forwarded to TwoRoomCFEnv.reset(options=...).
                    Supported keys: wall_x, door_y, agent_pos.
        actions:    (T, 2) action sequence from the dataset.

    Returns:
        dict with:
          positions    : (T+1, 2) agent positions including the initial frame.
          through_door : bool — whether the agent crossed the central wall.
    """
    env = gymnasium.make("lewm/TwoRoomCF-v0")
    obs, info = env.reset(options=env_params)

    init_pos = info["state"].copy()          # (2,)
    wall_center = env.unwrapped.WALL_CENTER  # may have been overridden

    positions = [init_pos]
    init_side = (init_pos[0] < wall_center)
    crossed = False

    for a in actions:
        obs, _, terminated, truncated, info = env.step(a)
        pos = info["state"].copy()
        positions.append(pos)

        cur_side = (pos[0] < wall_center)
        if cur_side != init_side:
            crossed = True

        if terminated or truncated:
            break

    env.close()

    return {
        "positions": np.array(positions, dtype=np.float32),   # (T+1, 2)
        "through_door": crossed,
    }
