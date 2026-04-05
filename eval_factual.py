"""
eval_factual.py — H步自回归Unroll事实评测

Usage:
    python eval_factual.py policy=<ckpt_name>
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import gymnasium
import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import v2 as T

import stable_worldmodel as swm
from stable_worldmodel.data.utils import get_cache_dir
from cf_env import register_cf_env
from utils import add_model_suffix

register_cf_env()


def _make_transform(img_size: int):
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Resize(size=img_size),
    ])


def _render_frame_at_state(env, agent_pos: np.ndarray, transform, device) -> torch.Tensor:
    """Reset env to a specific agent pos, render one frame -> (1, 1, C, H, W)."""
    env.reset(options={"agent_pos": agent_pos})
    frame = env.render()  # (H, W, 3) uint8
    img = transform(frame)  # (C, H, W)
    return img.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, C, H, W)


def _load_jepa_from_weights(weights_path: str, device: torch.device):
    """Build JEPA directly from a weights.pt state_dict (no *_object.ckpt needed).

    Reads the checkpoint to infer action_encoder input_dim (frameskip × action_dim)
    and norm type (BatchNorm1d vs LayerNorm) from the saved keys, so no manual
    config is required.
    """
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP
    from stable_pretraining.backbone.utils import vit_hf

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    # ── infer architecture from weights ──────────────────────────────────────
    act_input_dim  = ckpt["action_encoder.patch_embed.weight"].shape[1]  # e.g. 10
    hidden_dim     = ckpt["projector.net.0.weight"].shape[0]             # MLP hidden, e.g. 2048
    embed_dim      = ckpt["projector.net.0.weight"].shape[1]             # D = 192
    predictor_frames = ckpt["predictor.pos_embedding"].shape[1]          # num_frames = 3
    predictor_depth  = sum(
        1 for k in ckpt if k.startswith("predictor.transformer.layers.")
        and k.endswith(".adaLN_modulation.1.weight")
    )

    # projector uses BatchNorm1d if running_mean present, else LayerNorm
    proj_norm = torch.nn.BatchNorm1d if "projector.net.1.running_mean" in ckpt else torch.nn.LayerNorm

    encoder = vit_hf(
        size="tiny", patch_size=14, image_size=224,
        pretrained=False, use_mask_token=False,
    )
    predictor = ARPredictor(
        num_frames=predictor_frames,
        input_dim=embed_dim, hidden_dim=embed_dim, output_dim=embed_dim,
        depth=predictor_depth, heads=16, mlp_dim=2048, dim_head=64,
        dropout=0.1, emb_dropout=0.0,
    )
    action_encoder = Embedder(input_dim=act_input_dim, emb_dim=embed_dim)
    projector  = MLP(input_dim=embed_dim, output_dim=embed_dim,
                     hidden_dim=hidden_dim, norm_fn=proj_norm)
    pred_proj  = MLP(input_dim=embed_dim, output_dim=embed_dim,
                     hidden_dim=hidden_dim, norm_fn=proj_norm)

    model = JEPA(encoder, predictor, action_encoder, projector, pred_proj)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading weights: {missing}")
    if unexpected:
        print(f"  [warn] Ignored unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    return model


def _resolve_weights_pt(policy: str) -> Path | None:
    """Return weights.pt path if policy resolves to a directory containing one.

    Checks (in order):
      1. policy itself is a path to a .pt file
      2. policy is a directory (absolute or relative) containing weights.pt
      3. $STABLEWM_HOME/policy/weights.pt  (e.g. policy="tworoom/lewm")
    """
    p = Path(policy)
    if p.suffix == ".pt" and p.exists():
        return p
    for candidate in (p, Path(get_cache_dir()) / policy):
        if candidate.is_dir():
            w = candidate / "weights.pt"
            if w.exists():
                return w
    return None


def _load_jepa(cfg: DictConfig, device: torch.device):
    if cfg.policy == "random":
        return None
    weights_pt = _resolve_weights_pt(cfg.policy)
    if weights_pt is not None:
        print(f"  Loading from state_dict: {weights_pt}")
        model = _load_jepa_from_weights(str(weights_pt), device)
    else:
        model = swm.policy.AutoCostModel(cfg.policy)
        if not hasattr(model, "encode"):
            for child in model.children():
                if hasattr(child, "encode"):
                    model = child
                    break
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def factual_unroll_episode(
    model,
    env,
    ep_proprio: np.ndarray,   # (T+1, 2): positions at each timestep
    ep_actions: np.ndarray,   # (T, 2): actions a_0..a_{T-1}
    transform,
    device: torch.device,
    horizon: int,
    history_size: int = 1,
) -> dict:
    """
    H步自回归Unroll，返回每步的MSE。

    流程：
      z_0^pred = Enc(o_0)
      for t in 0..H-1:
        act_emb = ActionEnc(a_t)
        z_{t+1}^pred = Pred(z_t^pred, act_emb)  # 用最近history_size步
        o_{t+1}  = render(proprio[t+1])           # 从H5位置重放渲染
        z_{t+1}^true = Enc(o_{t+1})
        mse[t] = ||z_{t+1}^pred - z_{t+1}^true||^2
    """
    H = min(horizon, len(ep_actions))
    mse_per_step = []

    # ── 1. 初始化 Context 缓冲 ─────────────────────────────────────────
    obs_0 = _render_frame_at_state(env, ep_proprio[0], transform, device)
    info = {"pixels": obs_0}
    model.encode(info)
    emb_context = info["emb"][:, -1:, :]  # (1, 1, D)

    # 推断训练时的 effective_act_dim (= frameskip * action_dim)
    # Embedder.patch_embed 是 Conv1d(input_dim, ...) → weight shape [out, in, 1]
    effective_act_dim = model.action_encoder.patch_embed.weight.shape[1]
    raw_act_dim = ep_actions.shape[-1]
    act_repeat = effective_act_dim // raw_act_dim  # = frameskip

    # 维护 action embedding 历史，用于 history_size 截断
    act_embs_history = []

    # ── 2. 自回归Unroll ────────────────────────────────────────────────
    for t in range(H):
        # a_t: (1, 1, effective_act_dim)  — tile action to match training dim
        a_t = torch.tensor(ep_actions[t], dtype=torch.float32, device=device)
        a_t = a_t.repeat(act_repeat)          # (effective_act_dim,)
        a_t = a_t.unsqueeze(0).unsqueeze(0)   # (1, 1, effective_act_dim)

        curr_act_emb = model.action_encoder(a_t)  # (1, 1, A_emb)
        act_embs_history.append(curr_act_emb)

        # 对齐 state/action context 长度
        recent_acts = act_embs_history[-history_size:]
        act_context = torch.cat(recent_acts, dim=1)  # (1, min(t+1,HS), A_emb)
        context_len = act_context.shape[1]
        emb_trunc = emb_context[:, -context_len:, :]  # (1, context_len, D)

        pred_emb_next = model.predict(emb_trunc, act_context)[:, -1:, :]  # (1, 1, D)

        # ── 3. 获取 z_{t+1}^true ────────────────────────────────────────
        obs_next = _render_frame_at_state(env, ep_proprio[t + 1], transform, device)
        info_true = {"pixels": obs_next}
        model.encode(info_true)
        true_emb_next = info_true["emb"][:, -1:, :]  # (1, 1, D)

        # ── 4. 计算隐空间误差 (Latent MSE) ──────────────────────────────
        mse = torch.mean((pred_emb_next - true_emb_next) ** 2).item()
        mse_per_step.append(mse)

        # ── 6. 用预测的 embedding 推进 context (Open-loop) ─────────────
        emb_context = torch.cat([emb_context, pred_emb_next], dim=1)

    return {
        "mse_per_step": np.array(mse_per_step),
    }


@torch.no_grad()
def run_factual_eval(
    model,
    h5_path: str,
    n_episodes: int,
    horizon: int,
    img_size: int,
    device: torch.device,
    rng: np.random.Generator,
    history_size: int = 1,
) -> dict:
    transform = _make_transform(img_size)

    with h5py.File(h5_path, "r") as f:
        ep_offsets = f["ep_offset"][:]
        ep_lens    = f["ep_len"][:]
        proprio    = f["proprio"][:]
        actions    = f["action"][:]
    if actions.ndim == 3:
        actions = actions[:, 0, :]

    env = gymnasium.make("lewm/TwoRoomCF-v0", render_mode="rgb_array")

    ep_indices = rng.choice(len(ep_offsets), size=min(n_episodes, len(ep_offsets)), replace=False)

    all_mse = []

    for ep_idx in ep_indices:
        off = int(ep_offsets[ep_idx])
        l   = int(ep_lens[ep_idx])

        if l < 2:
            continue

        # ep_proprio has T+1 entries (states), ep_actions has T entries
        ep_end = off + l
        ep_proprio = proprio[off : ep_end + 1] if ep_end + 1 <= len(proprio) else proprio[off : ep_end]
        ep_actions = actions[off : ep_end]

        H = min(horizon, len(ep_proprio) - 1, len(ep_actions))
        if H < 1:
            continue

        result = factual_unroll_episode(
            model, env,
            ep_proprio[:H + 1],
            ep_actions[:H],
            transform, device, H,
            history_size=history_size,
        )
        all_mse.append(result["mse_per_step"])

    env.close()

    # ── 统计（对齐到最短序列）────────────────────────────────────────────
    min_len = min(len(m) for m in all_mse)
    mse_matrix = np.stack([m[:min_len] for m in all_mse])  # (N_ep, H)

    metrics = {
        "mean_mse_per_step":  mse_matrix.mean(axis=0).tolist(),
        "std_mse_per_step":   mse_matrix.std(axis=0).tolist(),
        "mean_total_mse":     float(mse_matrix.mean()),
        "n_episodes":         len(all_mse),
        "drift_slope":        float(np.polyfit(np.arange(min_len), mse_matrix.mean(axis=0), 1)[0]),
    }

    return metrics


# ── Hydra entry point ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./config/eval", config_name="factual_eval")
def run(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(cfg.seed)

    print("Loading world model...")
    model = _load_jepa(cfg, device)
    if model is None:
        print("  policy=random: latent MSE will be meaningless (random encoder).")

    h5_path = str(Path(get_cache_dir()) / f"{cfg.dataset.name}.h5")
    print(f"Dataset: {h5_path}")
    print(f"Factual eval: {cfg.factual.n_episodes} episodes, horizon={cfg.factual.horizon}")

    t0 = time.time()
    metrics = run_factual_eval(
        model=model,
        h5_path=h5_path,
        n_episodes=int(cfg.factual.n_episodes),
        horizon=int(cfg.factual.horizon),
        img_size=int(cfg.env.img_size),
        device=device,
        rng=rng,
        history_size=int(cfg.wm.history_size),
    )
    elapsed = time.time() - t0

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FACTUAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"  n_episodes      : {metrics['n_episodes']}")
    print(f"  mean_total_mse  : {metrics['mean_total_mse']:.6f}")
    print(f"  drift_slope     : {metrics['drift_slope']:.6f}  (MSE/step)")
    mse_curve = metrics["mean_mse_per_step"]
    print(f"  MSE curve       : t=0→{mse_curve[0]:.4f}  t={len(mse_curve)-1}→{mse_curve[-1]:.4f}")
    print(f"\nTotal time: {elapsed:.1f}s")

    # ── Persist ────────────────────────────────────────────────────────────────
    out_path = add_model_suffix(cfg.output.filename, cfg.policy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics,
        "elapsed_seconds": elapsed,
    }
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    run()
