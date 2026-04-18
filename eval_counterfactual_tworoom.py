"""
eval_counterfactual.py — Closed-loop counterfactual evaluation of LeWM in TwoRoom.

Directly reuses eval.py's full pipeline (World + WorldModelPolicy + CEMSolver +
evaluate_from_dataset) and adds per-intervention structural overrides via an
extra callable injected into evaluate_from_dataset's callables list.

The intervention callable calls _set_intervention(wall_x, door_y) on the
unwrapped TwoRoomCFEnv after the dataset state/goal have been set, which
overrides the wall position and door y-coordinate for the current episode.

Usage
-----
python eval_counterfactual_tworoom.py policy=tworoom/lewm [num_episodes=50] [seed=42]

Output
------
Prints a summary table and writes JSON to logs_eval/cf_closedloop_<model>.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import hydra
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from tqdm import tqdm

from omegaconf import OmegaConf, DictConfig
from sklearn import preprocessing

import stable_worldmodel as swm
from stable_worldmodel.data.utils import get_cache_dir

# ── reuse eval.py helpers verbatim ────────────────────────────────────────────
from eval import load_lewm_model, img_transform, get_dataset, get_episodes_length

# ── register CF env ───────────────────────────────────────────────────────────
from tworoom_eval.cf_env import register_cf_env
register_cf_env()


# ── Intervention definitions ───────────────────────────────────────────────────
# Each entry: (name, wall_x_override, door_y_override)
#   None = keep dataset/training default
INTERVENTIONS: list[tuple[str, int | None, float | None]] = [
    ("baseline",          None,  None),
    ("do_door_y",         None,  180.0),   # door shifted far from training y=49
    ("do_wall_x",         170,   None),    # wall shifted right
    ("do_door_y_wall_x",  170,   180.0),   # both shifted (strongest OOD)
    ("do_location",       None,  None),    # agent starts in right room — handled separately
    ("do_reverse_task",   None,  None),    # agent starts right, goal in left (full reversal)
    ("do_wall_color",     None,  None),    # bright blue walls (visual appearance shift)
    ("do_checkerboard",   None,  None),    # high-contrast checkerboard floor
]

# For do_location we override agent start pos instead of wall/door.
# These are the OOD right-room positions from counterfactual_eval.yaml.
DO_LOCATION_POSITIONS = [
    [140, 49], [160, 49], [185, 49],
    [140, 112], [160, 112], [185, 112],
    [140, 175], [160, 175], [185, 25], [185, 190],
]

# For do_reverse_task: sample positions randomly in valid ranges
# (computed per episode, not fixed list)


def _make_intervention_callable(
    wall_x: int | None,
    door_y: float | None,
    agent_pos: list[float] | None = None,
    goal_pos: list[float] | None = None,
    wall_color: list[int] | None = None,
    bg_color: list[int] | None = None,
    checkerboard: bool = False,
    sample_reverse_positions: bool = False,
    seed: int = 42,
) -> dict:
    """
    Build a callable spec understood by world.evaluate_from_dataset.

    We use _set_intervention on TwoRoomCFEnv, which overrides wall/door after
    _set_state/_set_goal_state have already been called by the standard callables.
    """
    return {
        "method": "_set_intervention",
        "args": {
            "wall_x":                  {"value": wall_x,                  "in_dataset": False},
            "door_y":                  {"value": door_y,                  "in_dataset": False},
            "agent_pos":               {"value": agent_pos,               "in_dataset": False},
            "goal_pos":                {"value": goal_pos,                "in_dataset": False},
            "wall_color":              {"value": wall_color,              "in_dataset": False},
            "bg_color":                {"value": bg_color,                "in_dataset": False},
            "checkerboard":            {"value": checkerboard,            "in_dataset": False},
            "sample_reverse_positions": {"value": sample_reverse_positions, "in_dataset": False},
            "seed":                    {"value": seed,                    "in_dataset": False},
        },
    }


def patch_cf_env(env_name: str) -> None:
    """
    Monkey-patch TwoRoomCFEnv with a _set_intervention method so it can be
    called as a callable by evaluate_from_dataset.

    This is called once before creating the World. The method is added to the
    class (not an instance), so all instances created afterwards have it.
    """
    from tworoom_eval.cf_env import TwoRoomCFEnv
    import torch as _torch
    import numpy as _np

    def _set_intervention(self, wall_x=None, door_y=None, agent_pos=None, goal_pos=None,
                          wall_color=None, bg_color=None, checkerboard=False,
                          sample_reverse_positions=False, seed=42):
        """Apply structural intervention after dataset state has been set."""

        # For do_reverse_task: sample random positions in right/left rooms
        if sample_reverse_positions:
            # Use a counter to ensure different positions per episode
            if not hasattr(self, '_reverse_counter'):
                self._reverse_counter = 0

            rng = _np.random.default_rng(seed + self._reverse_counter)
            self._reverse_counter += 1

            # Valid ranges (accounting for border + agent radius)
            BORDER_SIZE = 14
            AGENT_RADIUS = 7.0
            WALL_CENTER = 112
            WALL_HALF = 5  # wall_width // 2

            min_coord = BORDER_SIZE + AGENT_RADIUS  # 21
            max_coord = 224 - BORDER_SIZE - AGENT_RADIUS  # 203

            # Right room: x in [WALL_CENTER + WALL_HALF + AGENT_RADIUS, max_coord]
            right_x_min = WALL_CENTER + WALL_HALF + AGENT_RADIUS  # 124
            right_x_max = max_coord  # 203

            # Left room: x in [min_coord, WALL_CENTER - WALL_HALF - AGENT_RADIUS]
            left_x_min = min_coord  # 21
            left_x_max = WALL_CENTER - WALL_HALF - AGENT_RADIUS  # 100

            # Get door position to sample y near it
            door_y = float(self.door_positions[0].item())

            # Sample y within reasonable range of door (±60 pixels)
            # This ensures paths are feasible within eval_budget
            y_margin = 60.0
            y_min = max(min_coord, door_y - y_margin)  # at least 21
            y_max = min(max_coord, door_y + y_margin)  # at most 203

            # Sample agent position in right room
            agent_x = float(rng.uniform(right_x_min, right_x_max))
            agent_y = float(rng.uniform(y_min, y_max))
            agent_pos = [agent_x, agent_y]

            # Sample goal position in left room
            goal_x = float(rng.uniform(left_x_min, left_x_max))
            goal_y = float(rng.uniform(y_min, y_max))
            goal_pos = [goal_x, goal_y]

        # wall_x override
        if wall_x is not None:
            self.WALL_CENTER = int(wall_x)
            self.wall_pos = float(self.WALL_CENTER)
            self._cache_params()  # recompute wall/door geometry

        # door_y override — update the variation space value
        if door_y is not None:
            pos_val = _np.asarray(
                self.variation_space["door"]["position"].value, dtype=int
            ).copy()
            pos_val[0] = int(round(float(door_y)))
            self.variation_space["door"]["position"].set_value(pos_val)
            self._cache_params()

        # agent_pos override (do_location, do_reverse_task)
        if agent_pos is not None:
            self.agent_position = _torch.tensor(
                agent_pos, dtype=_torch.float32
            )

        # goal_pos override (do_reverse_task)
        if goal_pos is not None:
            self.target_position = _torch.tensor(
                goal_pos, dtype=_torch.float32
            )
            # Also update variation_space to keep it in sync
            self.variation_space["target"]["position"].set_value(
                _np.array(goal_pos, dtype=_np.float32)
            )

        # wall_color override (do_wall_color)
        if wall_color is not None:
            self.variation_space["wall"]["color"].set_value(
                _np.array(wall_color, dtype=_np.uint8)
            )

        # bg_color override (for checkerboard base)
        if bg_color is not None:
            self.variation_space["background"]["color"].set_value(
                _np.array(bg_color, dtype=_np.uint8)
            )

        # checkerboard override (do_checkerboard)
        if checkerboard:
            # Store checkerboard flag so _render_frame can use it
            self._use_checkerboard = True
            # Monkey-patch _render_frame to add checkerboard
            original_render = self._render_frame
            def _render_with_checkerboard(agent_pos):
                img = original_render(agent_pos)
                # Add checkerboard pattern to background
                H = W = self.IMG_SIZE
                tile_size = 28  # 28x28 tiles for 224x224 image (8x8 grid)
                y_idx = _torch.arange(H, device=img.device) // tile_size
                x_idx = _torch.arange(W, device=img.device) // tile_size
                y_grid, x_grid = _torch.meshgrid(y_idx, x_idx, indexing='ij')
                checker = ((y_grid + x_grid) % 2).to(_torch.uint8)

                # Create mask for non-wall, non-agent, non-target areas
                # We'll apply checkerboard only to "background" pixels
                # Simple heuristic: pixels that are close to bg_color
                bg = self.variation_space["background"]["color"].value
                bg_tensor = _torch.tensor([bg[0], bg[1], bg[2]], dtype=_torch.uint8, device=img.device).view(3, 1, 1)

                # Checkerboard colors: dark gray (80) and light gray (200)
                dark = 80
                light = 200

                # Apply checkerboard to all channels
                for c in range(3):
                    # Where checker is 0, use dark; where 1, use light
                    checker_channel = _torch.where(checker == 0, dark, light).to(_torch.uint8)
                    # Blend with existing image based on whether it's background
                    # Simple approach: if pixel is close to white (bg), replace with checker
                    is_bg = (img[c] > 240)  # white-ish pixels
                    img[c] = _torch.where(is_bg, checker_channel, img[c])

                return img
            self._render_frame = _render_with_checkerboard

        # Re-render the goal image with the (possibly updated) geometry
        self._target_img = self._render_frame(agent_pos=self.target_position)

    TwoRoomCFEnv._set_intervention = _set_intervention


def patch_world_progress(world) -> None:
    """Wrap world.evaluate_from_dataset to show a tqdm step-level progress bar.

    Intercepts world.step() calls that happen inside evaluate_from_dataset to
    tick a progress bar, without touching any evaluation logic.
    """
    import functools
    import stable_worldmodel as _swm

    original_eval  = _swm.World.evaluate_from_dataset
    original_step  = _swm.World.step

    # Shared state between the two wrappers (closure variable)
    _ctx: dict = {"pbar": None}

    @functools.wraps(original_step)
    def _step_with_tick(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        if _ctx["pbar"] is not None:
            _ctx["pbar"].update(1)
        return result

    @functools.wraps(original_eval)
    def _eval_with_progress(self, *args, desc="", **kwargs):
        eval_budget = kwargs.get("eval_budget") or (args[3] if len(args) > 3 else None)
        _ctx["pbar"] = tqdm(total=eval_budget, desc=desc, leave=False, unit="step")
        try:
            result = original_eval(self, *args, **kwargs)
        finally:
            _ctx["pbar"].close()
            _ctx["pbar"] = None
        return result

    world.step                  = _step_with_tick.__get__(world, type(world))
    world.evaluate_from_dataset = _eval_with_progress.__get__(world, type(world))


def parse_args() -> dict:
    cfg = {
        "policy":       "random",
        "num_episodes": 50,
        "seed":         42,
        "output":       "logs_eval/cf_closedloop.json",
    }
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            if k in ("num_episodes", "seed"):
                cfg[k] = int(v)
            else:
                cfg[k] = v
        elif arg in ("--help", "-h"):
            print(__doc__)
            raise SystemExit(0)
    return cfg


def run_one_intervention(
    intervention_name: str,
    wall_x: int | None,
    door_y: float | None,
    agent_pos_override: list[float] | None,
    goal_pos_override: list[float] | None,
    wall_color_override: list[int] | None,
    bg_color_override: list[int] | None,
    checkerboard_override: bool,
    sample_reverse_positions: bool,
    *,
    policy,
    world,
    dataset,
    eval_episodes: list[int],
    eval_start_idx: list[int],
    goal_offset_steps: int,
    eval_budget: int,
    base_callables: list[dict],
    seed: int,
) -> dict:
    """Run evaluate_from_dataset for one intervention type."""

    # Build callables: base (set_state + set_goal_state) + intervention
    callables = list(base_callables)
    if (wall_x is not None or door_y is not None or agent_pos_override is not None
        or goal_pos_override is not None or wall_color_override is not None
        or bg_color_override is not None or checkerboard_override or sample_reverse_positions):
        callables.append(
            _make_intervention_callable(
                wall_x, door_y, agent_pos_override, goal_pos_override,
                wall_color_override, bg_color_override, checkerboard_override,
                sample_reverse_positions, seed
            )
        )

    # Reset policy action buffer between interventions
    if hasattr(policy, "_action_buffer") and policy._action_buffer is not None:
        policy._action_buffer.clear()
    if hasattr(policy, "_next_init"):
        policy._next_init = None

    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx,
        goal_offset_steps=goal_offset_steps,
        eval_budget=eval_budget,
        episodes_idx=eval_episodes,
        callables=callables,
        save_video=False,
        desc=intervention_name,
    )
    return metrics


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load tworoom.yaml config (same as eval.py) ─────────────────────────
    # We load it manually (no Hydra) to avoid the @hydra.main decorator.
    cfg_path = Path(__file__).parent / "config" / "eval" / "tworoom.yaml"
    with hydra.initialize_config_dir(config_dir=str(cfg_path.parent), version_base=None):
        cfg = hydra.compose(config_name="tworoom")

    # Override policy from CLI
    OmegaConf.update(cfg, "policy", args["policy"], merge=True)
    OmegaConf.update(cfg, "eval.num_eval", args["num_episodes"], merge=True)
    OmegaConf.update(cfg, "seed", args["seed"], merge=True)

    print("=" * 70)
    print("LeWM Closed-loop Counterfactual Evaluation — TwoRoom")
    print("=" * 70)
    print(f"  policy       : {cfg.policy}")
    print(f"  num_episodes : {cfg.eval.num_eval} per intervention")
    print(f"  eval_budget  : {cfg.eval.eval_budget} steps")
    print(f"  horizon      : {cfg.plan_config.horizon}  action_block={cfg.plan_config.action_block}")
    print(f"  device       : {device}")
    print()

    # ── Patch TwoRoomCFEnv with _set_intervention ──────────────────────────
    patch_cf_env("lewm/TwoRoomCF-v0")

    # ── Build World using TwoRoomCF-v0 instead of TwoRoom-v1 ──────────────
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    img_size = int(cfg.eval.img_size)

    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    world_cfg["env_name"] = "lewm/TwoRoomCF-v0"   # swap to CF env
    world = swm.World(**world_cfg, image_shape=(img_size, img_size))

    # ── Inject tqdm progress bar into evaluate_from_dataset ───────────────
    patch_world_progress(world)

    transform = {
        "pixels": img_transform(cfg),
        "goal":   img_transform(cfg),
    }

    # ── Dataset + preprocessing (identical to eval.py) ─────────────────────
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

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

    # ── Policy (identical to eval.py) ─────────────────────────────────────
    if cfg.policy != "random":
        model = load_lewm_model(cfg.policy)
        if model is not None:
            model = model.to(device).eval().requires_grad_(False)
            model.interpolate_pos_encoding = True
            plan_cfg = swm.PlanConfig(**cfg.plan_config)
            solver = hydra.utils.instantiate(cfg.solver, model=model)
            policy = swm.policy.WorldModelPolicy(
                solver=solver, config=plan_cfg,
                process=process, transform=transform,
            )
        else:
            raise RuntimeError(f"Could not load model from policy={cfg.policy}")
    else:
        policy = swm.policy.RandomPolicy()

    world.set_policy(policy)

    # ── Sample episodes (identical to eval.py) ─────────────────────────────
    episode_len    = get_episodes_length(dataset, ep_indices)
    max_start_idx  = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_dict = {ep: max_start_idx[i] for i, ep in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_dict[ep] for ep in dataset.get_col_data(col_name)]
    )
    valid_mask    = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    g = np.random.default_rng(cfg.seed)
    sampled = g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)
    sampled = np.sort(valid_indices[sampled])

    eval_episodes  = dataset.get_row_data(sampled)[col_name].tolist()
    eval_start_idx = dataset.get_row_data(sampled)["step_idx"].tolist()

    # ── Base callables (set_state + set_goal_state from dataset) ──────────
    base_callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)

    # ── Run each intervention ──────────────────────────────────────────────
    all_results: dict[str, dict] = {}
    total_t0 = time.time()

    for iname, wall_x, door_y in INTERVENTIONS:
        print(f"\n[{iname}]  wall_x={wall_x}  door_y={door_y}")

        agent_pos_ov = None
        goal_pos_ov = None
        wall_color_ov = None
        bg_color_ov = None
        checkerboard_ov = False
        sample_reverse = False

        if iname == "do_location":
            # For do_location we randomly pick one OOD start position per run.
            rng_loc = np.random.default_rng(cfg.seed + 1)
            idx = int(rng_loc.integers(0, len(DO_LOCATION_POSITIONS)))
            agent_pos_ov = DO_LOCATION_POSITIONS[idx]
            print(f"  agent_pos_override={agent_pos_ov}")
        elif iname == "do_reverse_task":
            # For do_reverse_task: sample random positions in right/left rooms
            sample_reverse = True
            print(f"  sampling agent in RIGHT room, goal in LEFT room")
        elif iname == "do_wall_color":
            # Bright blue walls instead of black
            wall_color_ov = [30, 144, 255]  # dodger blue
            print(f"  wall_color_override={wall_color_ov}")
        elif iname == "do_checkerboard":
            # High-contrast checkerboard floor
            checkerboard_ov = True
            print(f"  checkerboard=True")

        t0 = time.time()
        metrics = run_one_intervention(
            iname, wall_x, door_y, agent_pos_ov, goal_pos_ov,
            wall_color_ov, bg_color_ov, checkerboard_ov, sample_reverse,
            policy=policy,
            world=world,
            dataset=dataset,
            eval_episodes=eval_episodes,
            eval_start_idx=eval_start_idx,
            goal_offset_steps=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            base_callables=base_callables,
            seed=cfg.seed,
        )
        elapsed_i = time.time() - t0

        sr = metrics["success_rate"] / 100.0   # evaluate_from_dataset returns %
        all_results[iname] = {
            "success_rate":      sr,
            "n_episodes":        cfg.eval.num_eval,
            "n_success":         int(round(sr * cfg.eval.num_eval)),
            "elapsed_seconds":   elapsed_i,
            "raw_metrics":       {k: v.tolist() if hasattr(v, "tolist") else v
                                  for k, v in metrics.items()},
        }
        print(f"  SR={sr:.3f}  ({all_results[iname]['n_success']}/{cfg.eval.num_eval})  [{elapsed_i:.0f}s]")

    elapsed = time.time() - total_t0

    # ── Summary table ──────────────────────────────────────────────────────
    baseline_sr = all_results.get("baseline", {}).get("success_rate", float("nan"))
    W = 72
    print("\n" + "=" * W)
    print("COUNTERFACTUAL EVALUATION RESULTS  (eval.py pipeline)")
    print(f"  policy={cfg.policy}  n={cfg.eval.num_eval}  budget={cfg.eval.eval_budget}steps")
    print("=" * W)
    print(f"  {'Intervention':<22}  {'SR':>6}  {'ΔSR':>7}  {'n_success':>9}")
    print("  " + "-" * (W - 2))
    for name, m in all_results.items():
        sr = m["success_rate"]
        delta = sr - baseline_sr
        dstr  = f"{delta:+.3f}" if not np.isnan(baseline_sr) else "   n/a"
        print(f"  {name:<22}  {sr:>6.3f}  {dstr:>7}  {m['n_success']:>5}/{m['n_episodes']}")
    print()
    print(f"  Total time: {elapsed:.0f}s")
    print("=" * W)

    # ── Save results ───────────────────────────────────────────────────────
    out_path = Path(args["output"])
    model_name = Path(cfg.policy).name if cfg.policy != "random" else "random"
    stem = out_path.stem
    if not stem.endswith(f"_{model_name}"):
        out_path = out_path.with_name(f"{stem}_{model_name}{out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result_obj = {
        "config":          OmegaConf.to_container(cfg, resolve=True),
        "cli_args":        args,
        "baseline_sr":     baseline_sr,
        "metrics":         all_results,
        "elapsed_seconds": elapsed,
    }
    with out_path.open("w") as f:
        json.dump(result_obj, f, indent=2, default=_json_default)
    print(f"Results written to {out_path}")


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
