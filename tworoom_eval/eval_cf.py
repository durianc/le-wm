"""
Counterfactual evaluation of LeWM in the TwoRoom environment.

For each of the four intervention tasks the script:
  1. Collects effectful (factual, CF) sample pairs via cf_sampler.
  2. Renders initial observations for both factual and CF environments.
  3. Encodes + rolls out the world model (model.encode / model.rollout).
  4. Applies a frozen linear probe to classify through_door from predicted embs.
  5. Computes effect_pass_acc_given_oracle_change and other task metrics.

Run
---
python eval_cf.py policy=<ckpt_name> probe_ckpt=<logs_eval/probes/probe.pt>
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import v2 as T
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import stable_worldmodel as swm
from stable_worldmodel.data.utils import get_cache_dir
from .cf_env import register_cf_env
from .cf_sampler import collect_effectful_samples
from .eval_factual import _load_jepa_from_weights, _resolve_weights_pt
from .probe import MLPProbe
from .utils import add_model_suffix, resolve_model_artifact_path


PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_DIR / "config" / "eval"

register_cf_env()

# ── Image preprocessing ────────────────────────────────────────────────────────

def _make_transform(img_size: int):
    return T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize(size=img_size),
        ]
    )


def _render_obs(env_params: dict, img_size: int, transform) -> torch.Tensor:
    """Reset TwoRoomCF with env_params, render one frame, return (1, 1, C, H, W)."""
    env = gymnasium.make("lewm/TwoRoomCF-v0", render_mode="rgb_array")
    env.reset(options=env_params)
    frame = env.render()  # (H, W, 3) uint8
    env.close()
    img = transform(frame)           # (C, H, W) float32
    return img.unsqueeze(0).unsqueeze(0)  # (1, 1, C, H, W)


# ── Model helpers ──────────────────────────────────────────────────────────────

def _load_jepa(cfg: DictConfig, device: torch.device):
    """Load a JEPA-compatible model that exposes encode() and rollout()."""
    if cfg.policy == "random":
        return None

    weights_pt = _resolve_weights_pt(cfg.policy)
    if weights_pt is not None:
        print(f"  Loading from state_dict: {weights_pt}")
        model = _load_jepa_from_weights(str(weights_pt), device)
    else:
        model = swm.policy.AutoCostModel(cfg.policy)
        # AutoActionableModel returns the module with get_action; we need encode().
        if not hasattr(model, "encode"):
            for child in model.children():
                if hasattr(child, "encode"):
                    model = child
                    break
    model = model.to(device).eval().requires_grad_(False)
    return model


def _load_probe(probe_ckpt: str, device: torch.device) -> tuple[MLPProbe, dict] | tuple[None, None]:
    path = Path(probe_ckpt)
    if not path.exists():
        return None, None
    ckpt = torch.load(path, map_location=device, weights_only=True)
    probe = MLPProbe(
        in_dim=ckpt["in_dim"],
        hidden_dim=ckpt.get("hidden_dim", 256),
        out_dim=ckpt.get("out_dim", 2),
    )
    probe.load_state_dict(ckpt["state_dict"])
    probe = probe.to(device).eval()
    stats = {
        "pos_mean": ckpt["pos_mean"].to(device),
        "pos_std":  ckpt["pos_std"].to(device),
    }
    return probe, stats


# ── Per-sample model evaluation ────────────────────────────────────────────────

@torch.no_grad()
def _model_predict_pass(
    model,
    probe: MLPProbe,
    probe_stats: dict,
    env_params: dict,
    actions: np.ndarray,
    img_size: int,
    transform,
    device: torch.device,
    history_size: int = 1,
) -> int | None:
    """
    Encode the initial observation, regress agent_pos with the linear probe,
    then run oracle_rollout with the predicted position to get through_door.

    The probe maps initial-frame embedding → agent_pos (x, y).  This measures
    whether the encoder has captured spatial structure, independent of the
    action sequence.

    Returns predicted class (0 or 1), or None if model/probe unavailable.
    """
    if model is None or probe is None:
        return None

    from .cf_oracle import oracle_rollout as _oracle_rollout

    # Render initial frame: (1, 1, C, H, W)
    obs = _render_obs(env_params, img_size, transform).to(device)

    # Encode the initial frame only — no rollout needed
    encode_info = {"pixels": obs}  # (1, 1, C, H, W)
    model.encode(encode_info)
    init_emb = encode_info["emb"][:, -1, :]  # (1, D)

    # Regress agent_pos from the initial embedding
    pos_norm = probe(init_emb)                                    # (1, 2)
    pos_pred = pos_norm * probe_stats["pos_std"] + probe_stats["pos_mean"]  # (1, 2)
    agent_pos_pred = pos_pred[0].cpu().numpy()                    # (2,)

    # Use oracle rollout with the predicted position to classify through_door.
    # Keep all other env params (wall_x, door_y) from the original env_params.
    pred_params = dict(env_params)
    pred_params["agent_pos"] = agent_pos_pred
    result = _oracle_rollout(pred_params, actions)
    return int(result["through_door"])


# ── Per-task metric computation ────────────────────────────────────────────────

def compute_task_metrics(
    task_name: str,
    task_cfg: DictConfig,
    h5_path: str,
    rng: np.random.Generator,
    model,
    probe: MLPProbe | None,
    probe_stats: dict | None,
    img_size: int,
    transform,
    device: torch.device,
    history_size: int,
) -> dict:
    """Run the full eval loop for one CF task, return metric dict."""
    print(f"\n[{task_name}] Collecting effectful samples...")
    t0 = time.time()
    samples, n_attempts, accept_rate = collect_effectful_samples(
        task_name, task_cfg, h5_path, rng
    )
    t_sample = time.time() - t0
    print(
        f"  {len(samples)}/{task_cfg.target_effectful_n} samples in "
        f"{n_attempts} attempts ({accept_rate:.3f} accept rate, {t_sample:.1f}s)"
    )

    if not samples:
        return {
            "n_samples": 0,
            "n_attempts": n_attempts,
            "accept_rate": accept_rate,
        }

    # Evaluate model predictions
    pass_change_correct = 0
    pass_change_total = 0
    total_correct = 0

    for s in tqdm(samples, desc=f"  {task_name} inference", unit="pair", leave=False):
        oracle_pass_changed = s["pass_changed"]  # bool

        fact_pred = _model_predict_pass(
            model, probe, probe_stats, s["fact_params"], s["actions"],
            img_size, transform, device, history_size,
        )
        cf_pred = _model_predict_pass(
            model, probe, probe_stats, s["cf_params"], s["actions"],
            img_size, transform, device, history_size,
        )

        if fact_pred is not None and cf_pred is not None:
            fact_oracle_pass = int(s["fact_oracle"]["through_door"])
            cf_oracle_pass   = int(s["cf_oracle"]["through_door"])

            fact_correct = int(fact_pred == fact_oracle_pass)
            cf_correct   = int(cf_pred   == cf_oracle_pass)
            total_correct += (fact_correct + cf_correct)

            if oracle_pass_changed:
                model_changed = int(fact_pred != cf_pred)
                pass_change_correct += model_changed
                pass_change_total   += 1

    n = len(samples)
    metrics: dict = {
        "n_samples": n,
        "n_attempts": n_attempts,
        "accept_rate": float(accept_rate),
        "pass_change_frac": float(
            sum(s["pass_changed"] for s in samples) / max(n, 1)
        ),
    }

    effect_mses = [
        s["effect_mse_oracle"]
        for s in samples
        if np.isfinite(s.get("effect_mse_oracle", np.nan))
    ]
    metrics["mean_effect_mse_oracle"] = (
        float(np.mean(effect_mses)) if effect_mses else None
    )

    if pass_change_total > 0:
        metrics["effect_pass_acc_given_oracle_change"] = (
            pass_change_correct / pass_change_total
        )
        metrics["n_oracle_change"] = pass_change_total

    if model is not None and probe is not None:
        metrics["mean_pair_accuracy"] = total_correct / (2 * n)

    return metrics


# ── Hydra entry point ──────────────────────────────────────────────────────────

TASK_NAMES = ["do_location", "do_door_y", "do_wall_x", "do_door_y_wall_x"]


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="counterfactual_eval")
def run(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(cfg.seed)

    img_size = int(cfg.env.img_size)
    transform = _make_transform(img_size)
    history_size = int(cfg.wm.history_size)

    print("Loading world model...")
    model = _load_jepa(cfg, device)

    print("Loading probe...")
    probe, probe_stats = _load_probe(cfg.probe_ckpt, device)
    if probe is None:
        print("  No probe found — pass/fail accuracy metrics will be skipped.")

    h5_path = str(Path(get_cache_dir()) / f"{cfg.dataset.name}.h5")
    print(f"Dataset: {h5_path}")

    all_metrics: dict[str, dict] = {}
    total_t0 = time.time()

    for task_name in TASK_NAMES:
        if not hasattr(cfg, task_name):
            print(f"Skipping {task_name} (no config section)")
            continue
        task_cfg = getattr(cfg, task_name)
        task_metrics = compute_task_metrics(
            task_name, task_cfg, h5_path, rng,
            model, probe, probe_stats, img_size, transform, device, history_size,
        )
        all_metrics[task_name] = task_metrics

    elapsed = time.time() - total_t0

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COUNTERFACTUAL EVALUATION RESULTS")
    print("=" * 60)
    key = "effect_pass_acc_given_oracle_change"
    for task_name, m in all_metrics.items():
        acc = m.get(key, float("nan"))
        n   = m.get("n_samples", 0)
        print(f"  {task_name:<24}  {key}={acc:.4f}  (n={n})")
    print(f"\nTotal time: {elapsed:.1f}s")

    # ── Persist ───────────────────────────────────────────────────────────────
    out_path = add_model_suffix(cfg.output.filename, cfg.policy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": all_metrics,
        "elapsed_seconds": elapsed,
    }
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    run()
