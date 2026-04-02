"""
Counterfactual TwoRoom environment.

Inherits TwoRoomEnv without modifying stable-worldmodel.
Supports structural interventions via reset(options=...):
  - options["wall_x"]   : override wall center x-coordinate (pixels)
  - options["door_y"]   : override door center y-coordinate (pixels, 1D along wall)
  - options["agent_pos"]: (2,) initial agent position [x, y]

Registered as "lewm/TwoRoomCF-v0" via register_cf_env().
"""

import numpy as np
import torch
import gymnasium

from stable_worldmodel.envs.two_room.env import TwoRoomEnv


class TwoRoomCFEnv(TwoRoomEnv):
    """TwoRoomEnv with explicit counterfactual intervention support."""

    # Store the original class-level constant so we can restore it per-instance.
    _BASE_WALL_CENTER = TwoRoomEnv.WALL_CENTER

    def reset(self, seed=None, options=None):
        options = options or {}

        # ── door_y intervention ──────────────────────────────────────────
        # Must happen before super().reset() calls _cache_params().
        if "door_y" in options:
            door_y = int(round(float(options["door_y"])))
            pos_val = np.asarray(self.variation_space["door"]["position"].value, dtype=int).copy()
            pos_val[0] = door_y
            self.variation_space["door"]["position"].set_value(pos_val)

        # ── wall_x intervention ──────────────────────────────────────────
        # TwoRoomEnv reads self.WALL_CENTER throughout rendering and collision.
        # Overriding on the instance shadows the class attribute cleanly.
        if "wall_x" in options:
            self.WALL_CENTER = int(options["wall_x"])
        else:
            # Restore default each reset so instances can be reused.
            self.WALL_CENTER = self._BASE_WALL_CENTER

        # ── agent_pos pass-through ───────────────────────────────────────
        # super().reset() reads options["state"] for agent position.
        if "agent_pos" in options:
            options = dict(options)  # shallow copy; don't mutate caller's dict
            options["state"] = np.asarray(options["agent_pos"], dtype=np.float32)

        obs, info = super().reset(seed=seed, options=options)

        # Keep wall_pos in sync (used by _get_obs for door coordinate reporting).
        self.wall_pos = float(self.WALL_CENTER)

        return obs, info

    def _render_frame(self, agent_pos: torch.Tensor):
        # _wall_and_door_masks and _apply_collisions already use self.WALL_CENTER,
        # so rendering is automatically consistent after the instance override.
        return super()._render_frame(agent_pos)


def register_cf_env():
    """Register lewm/TwoRoomCF-v0 once; safe to call multiple times."""
    env_id = "lewm/TwoRoomCF-v0"
    if env_id not in gymnasium.envs.registry:
        gymnasium.register(
            id=env_id,
            entry_point="cf_env:TwoRoomCFEnv",
        )
