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
python eval_cf.py policy=<ckpt_name> probe_ckpt=<probe.pt>
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

import stable_worldmodel as swm
from cf_env import register_cf_env
from cf_sampler import collect_effectful_samples
from linear_probe import LinearProbe

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

    model = swm.policy.AutoActionableModel(cfg.policy)
    # AutoActionableModel returns the module with get_action; we need encode().
    if not hasattr(model, "encode"):
        for child in model.children():
            if hasattr(child, "encode"):
                model = child
                break
    model = model.to(device).eval().requires_grad_(False)
    return model


def _load_probe(probe_ckpt: str, device: torch.device) -> LinearProbe | None:
    path = Path(probe_ckpt)
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device, weights_only=True)
    probe = LinearProbe(in_dim=ckpt["in_dim"], n_classes=2)
    probe.load_state_dict(ckpt["state_dict"])
    probe = probe.to(device).eval()
    return probe


# ── Per-sample model evaluation ────────────────────────────────────────────────

@torch.no_grad()
def _model_predict_pass(
    model,
    probe: LinearProbe,
    env_params: dict,
    actions: np.ndarray,
    img_size: int,
    transform,
    device: torch.device,
    history_size: int = 1,
) -> int | None:
    """
    Encode the initial observation, roll out the model over `actions`,
    then classify through_door with the linear probe.

    Returns predicted class (0 or 1), or None if model/probe unavailable.
    """
    if model is None or probe is None:
        return None

    # Render initial frame: (1, 1, C, H, W)
    obs = _render_obs(env_params, img_size, transform).to(device)

    # Prepare action tensor: (1, 1, T, action_dim)  [B=1, S=1, T, A]
    act_t = torch.from_numpy(actions).float().to(device)  # (T, A)
    act_t = act_t.unsqueeze(0).unsqueeze(0)               # (1, 1, T, A)

    info = {"pixels": obs}
    # rollout expects pixels (B, S, T, C, H, W); obs is (1, 1, C, H, W)
    # Expand to include the S=1 sample dimension.
    info["pixels"] = obs.unsqueeze(1)  # (1, 1, 1, C, H, W) -> need (B,S,T,C,H,W)
    # Actually rollout() signature: pixels (B,S,T,...), actions (B,S,T,A).
    # We have B=1, S=1, T=history_size initial frames.
    # Use encode() + autoregressive predict manually for clarity.

    # Encode the initial frame (no sample dim for encode)
    encode_info = {"pixels": obs}  # (1, 1, C, H, W)  — B=1, T=1
    model.encode(encode_info)
    emb = encode_info["emb"]  # (1, T_hist, D)

    # Autoregressively predict for each action step
    act_seq = torch.from_numpy(actions).float().to(device)  # (T, A)
    HS = history_size
    for t in range(len(actions)):
        a = act_seq[t : t + 1].unsqueeze(0)  # (1, 1, A)
        act_emb = model.action_encoder(a)     # (1, 1, A_emb)
        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -1:]
        pred_emb = model.predict(emb_trunc, act_trunc)[:, -1:]  # (1, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

    # Final predicted embedding: last step, (1, D)
    final_emb = emb[:, -1, :]  # (1, D)
    pred_class = probe.predict_pass(final_emb)  # (1,)
    return int(pred_class.item())


# ── Per-task metric computation ────────────────────────────────────────────────

def compute_task_metrics(
    task_name: str,
    task_cfg: DictConfig,
    dataset,
    rng: np.random.Generator,
    model,
    probe: LinearProbe | None,
    img_size: int,
    transform,
    device: torch.device,
    history_size: int,
) -> dict:
    """Run the full eval loop for one CF task, return metric dict."""
    print(f"\n[{task_name}] Collecting effectful samples...")
    t0 = time.time()
    samples, n_attempts, accept_rate = collect_effectful_samples(
        task_name, task_cfg, dataset, rng
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

    for s in samples:
        oracle_pass_changed = s["pass_changed"]  # bool

        fact_pred = _model_predict_pass(
            model, probe, s["fact_params"], s["actions"],
            img_size, transform, device, history_size,
        )
        cf_pred = _model_predict_pass(
            model, probe, s["cf_params"], s["actions"],
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
        "mean_effect_mse_oracle": float(
            np.mean([s["effect_mse_oracle"] for s in samples])
        ),
        "pass_change_frac": float(
            sum(s["pass_changed"] for s in samples) / max(n, 1)
        ),
    }

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


@hydra.main(version_base=None, config_path="./config/eval", config_name="counterfactual_eval")
def run(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(cfg.seed)

    img_size = int(cfg.env.img_size)
    transform = _make_transform(img_size)
    history_size = int(cfg.wm.history_size)

    print("Loading world model...")
    model = _load_jepa(cfg, device)

    print("Loading linear probe...")
    probe = _load_probe(cfg.probe_ckpt, device)
    if probe is None:
        print("  No probe found — pass/fail accuracy metrics will be skipped.")

    print("Loading dataset...")
    dataset = swm.data.HDF5Dataset(
        cfg.dataset.name,
        keys_to_cache=OmegaConf.to_container(cfg.dataset.keys_to_cache),
    )

    all_metrics: dict[str, dict] = {}
    total_t0 = time.time()

    for task_name in TASK_NAMES:
        if not hasattr(cfg, task_name):
            print(f"Skipping {task_name} (no config section)")
            continue
        task_cfg = getattr(cfg, task_name)
        task_metrics = compute_task_metrics(
            task_name, task_cfg, dataset, rng,
            model, probe, img_size, transform, device, history_size,
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
    out_path = Path(cfg.output.filename)
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
