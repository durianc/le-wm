"""
eval_pusht_counterfactual.py — Push-T 动作干预反事实评测

从数据集中找到 Agent 与 Block 首次接触的关键帧 T_contact，在该帧前后
分叉出两条开环 Rollout：
  Factual     — 保持数据集原始动作序列
  Counterfactual — 将动作替换为零向量（Agent 静止）

两路均调用 LeWM.encode/predict 在隐空间展开，逐步计算：
  - 两路预测 embedding 之间的 MSE divergence（"分叉程度"）
  - 每路预测与真实 embedding 的 MSE（事实误差 / 反事实误差）

预期：若隐空间编码了物理信息，Factual 与 Counterfactual 的 divergence
应随时间增大（因为 Block 是否被推动会使两路隐状态越来越不同）。

数据集格式（stable-worldmodel HDF5，$STABLEWM_HOME/<name>.h5）：
  pixels    (N, 224, 224, 3) uint8   — HWC，blosc 压缩
  action    (N, 2)           float32
  state     (N, 7)           float64 — [ax, ay, bx, by, b_angle, vx, vy]
  ep_offset (E,)             int64   — 每条 episode 在全局行中的起始索引
  ep_len    (E,)             int32   — 每条 episode 的行数

Usage:
    python eval_pusht_counterfactual.py --dataset pusht_expert_train --policy pusht/lewm
    python eval_pusht_counterfactual.py --dataset pusht_expert_train --policy pusht/lewm --cf_mode random
    python eval_pusht_counterfactual.py --dataset pusht_expert_train --policy pusht/lewm --contact_thresh 80
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torchvision.transforms import v2 as T

import stable_worldmodel as swm


# ── 图像预处理（与 LeWM 训练完全一致）─────────────────────────────────────────

def _make_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        T.Resize(size=img_size),
    ])


def _to_model_input(frame_np: np.ndarray, transform: T.Compose,
                    device: torch.device) -> torch.Tensor:
    """np.ndarray (H,W,C) uint8 → Tensor (1,1,C,H,W) on device."""
    return transform(frame_np).unsqueeze(0).unsqueeze(0).to(device)


def _get_cache_dir() -> Path:
    return Path(swm.data.utils.get_cache_dir())


# ── 数据集加载 ─────────────────────────────────────────────────────────────────

class Episode:
    """单条 episode 的裸数据容器（numpy，未经预处理）。"""
    __slots__ = ("frames", "actions", "states")

    def __init__(self, frames: np.ndarray, actions: np.ndarray,
                 states: np.ndarray):
        # frames:  (T+1, H, W, C) uint8
        # actions: (T,   2)       float32
        # states:  (T+1, 7)       float64  — [ax, ay, bx, by, b_angle, vx, vy]
        assert len(frames) == len(actions) + 1
        assert len(states) == len(frames)
        self.frames  = frames
        self.actions = actions
        self.states  = states   # for contact detection

    def __len__(self) -> int:
        return len(self.actions)


def _iter_episodes_h5(
    dataset_path: str,
    n_episodes: int,
    rng: np.random.Generator,
    require_state: bool = True,
) -> Iterator[Episode]:
    """从 HDF5 数据集读取 episodes，附带 state 列用于关键帧判定。"""
    import hdf5plugin  # noqa: F401
    import h5py

    with h5py.File(dataset_path, "r") as f:
        has_state = "state" in f
        if require_state and not has_state:
            raise RuntimeError(
                "数据集中缺少 'state' 列（需要 [ax,ay,bx,by,...] 格式），"
                "无法判定 Agent-Block 接触帧。"
            )

        ep_offsets = f["ep_offset"][:]
        ep_lens    = f["ep_len"][:]
        n_ep_avail = len(ep_offsets)
        indices = rng.choice(n_ep_avail, size=min(n_episodes, n_ep_avail),
                             replace=False)

        for idx in indices:
            off = int(ep_offsets[idx])
            l   = int(ep_lens[idx])
            if l < 2:
                continue
            frames  = f["pixels"][off : off + l]       # (l, H, W, C)
            actions = f["action"][off : off + l - 1]   # (l-1, 2)
            if has_state:
                states = f["state"][off : off + l]     # (l, 7)
            else:
                # fallback: dummy zeros (contact detection will be skipped)
                states = np.zeros((l, 7), dtype=np.float64)
            yield Episode(frames=frames, actions=actions, states=states)


def load_episodes(
    dataset_name: str,
    n_episodes: int,
    rng: np.random.Generator,
) -> list[Episode]:
    """从 $STABLEWM_HOME/<dataset_name>.h5 加载最多 n_episodes 条 Episode。"""
    h5_path = _get_cache_dir() / f"{dataset_name}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"数据集文件不存在: {h5_path}\n"
            f"请确认 $STABLEWM_HOME 设置正确（当前: {_get_cache_dir()}）"
        )
    episodes = list(_iter_episodes_h5(str(h5_path), n_episodes, rng))
    print(f"  加载 {len(episodes)} 条 episodes（来自 {h5_path}）")
    return episodes


# ── 模型加载 ───────────────────────────────────────────────────────────────────

def load_model(policy: str, device: torch.device):
    """加载 LeWM JEPA 模型，与 eval.py 的 load_lewm_model() 完全一致。"""
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP
    from stable_pretraining.backbone.utils import vit_hf

    p = Path(policy)
    weights_pt: Path | None = None

    if p.suffix == ".pt" and p.exists():
        weights_pt = p
    else:
        cache_dir = _get_cache_dir()
        for candidate_dir in (p, cache_dir / policy):
            w = candidate_dir / "weights.pt"
            if w.exists():
                weights_pt = w
                break

        if weights_pt is None:
            direct = cache_dir / policy
            if direct.suffix == ".pt" and direct.exists():
                weights_pt = direct

    if weights_pt is None:
        raise RuntimeError(
            f"找不到模型权重 '{policy}'。\n"
            "请提供 .pt 路径、含 weights.pt 的目录，或 $STABLEWM_HOME 下的子目录名称。"
        )

    ckpt = torch.load(weights_pt, map_location="cpu", weights_only=False)

    act_input_dim    = ckpt["action_encoder.patch_embed.weight"].shape[1]
    hidden_dim       = ckpt["projector.net.0.weight"].shape[0]
    embed_dim        = ckpt["projector.net.0.weight"].shape[1]
    predictor_frames = ckpt["predictor.pos_embedding"].shape[1]
    predictor_depth  = sum(
        1 for k in ckpt
        if k.startswith("predictor.transformer.layers.")
        and k.endswith(".adaLN_modulation.1.weight")
    )
    proj_norm = (
        torch.nn.BatchNorm1d
        if "projector.net.1.running_mean" in ckpt
        else torch.nn.LayerNorm
    )

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
    projector  = MLP(input_dim=embed_dim, hidden_dim=hidden_dim,
                     output_dim=embed_dim, norm_fn=proj_norm)
    pred_proj  = MLP(input_dim=embed_dim, hidden_dim=hidden_dim,
                     output_dim=embed_dim, norm_fn=proj_norm)

    model = JEPA(encoder, predictor, action_encoder, projector, pred_proj)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        raise RuntimeError(f"权重缺失: {missing}")
    if unexpected:
        print(f"  [warn] 忽略多余的权重键: {unexpected[:5]}"
              f"{'...' if len(unexpected) > 5 else ''}")

    print(f"  模型架构: embed_dim={embed_dim}, hidden_dim={hidden_dim}, "
          f"predictor_depth={predictor_depth}, act_input_dim={act_input_dim}, "
          f"predictor_frames={predictor_frames}")
    return model.to(device).eval().requires_grad_(False)


# ── 关键帧检测 ─────────────────────────────────────────────────────────────────

def find_contact_frame(states: np.ndarray, contact_thresh: float) -> int | None:
    """在 native 帧序列中找到 Agent-Block 首次接近的帧索引。

    Args:
        states:         (T+1, ≥4) float — 至少含 [ax, ay, bx, by, ...]
        contact_thresh: 像素单位距离阈值（PushT 坐标系 512×512）

    Returns:
        首次 dist < contact_thresh 的帧索引，若从未接触则返回 None。
    """
    agent_pos = states[:, :2]   # (T+1, 2)
    block_pos = states[:, 2:4]  # (T+1, 2)
    dists = np.linalg.norm(agent_pos - block_pos, axis=1)  # (T+1,)
    candidates = np.where(dists < contact_thresh)[0]
    return int(candidates[0]) if len(candidates) > 0 else None


# ── 反事实动作构造 ─────────────────────────────────────────────────────────────

def make_counterfactual_actions(
    actions: np.ndarray,
    start: int,
    length: int,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """返回干预后的动作序列（仅替换 [start, start+length) 段）。

    Args:
        actions: (T, raw_act_dim) float32 — 原始动作序列
        start:   干预起始 native 帧索引（inclusive）
        length:  干预长度（native 步数 = horizon * frameskip）
        mode:    "zero" | "random"
        rng:     随机数生成器（仅 mode="random" 使用）

    Returns:
        修改后的 (T, raw_act_dim) 动作副本。
    """
    cf_actions = actions.copy()
    end = min(start + length, len(actions))
    if mode == "zero":
        cf_actions[start:end] = 0.0
    elif mode == "random":
        raw_act_dim = actions.shape[-1]
        # 在 PushT action_space [-1, 1]^2 内均匀随机
        cf_actions[start:end] = rng.uniform(-1.0, 1.0,
                                            size=(end - start, raw_act_dim))
    else:
        raise ValueError(f"未知的反事实模式 '{mode}'，请选择 'zero' 或 'random'")
    return cf_actions


# ── 双路开环 Rollout ────────────────────────────────────────────────────────────

@torch.no_grad()
def counterfactual_unroll_episode(
    model,
    episode: Episode,
    transform: T.Compose,
    device: torch.device,
    contact_thresh: float,
    horizon: int,
    history_size: int,
    cf_mode: str,
    rng: np.random.Generator,
) -> dict | None:
    """单 episode 的双路反事实 Unroll。

    流程：
      1. 用 state 列检测关键帧 T_contact（首次 Agent-Block 距离 < contact_thresh）
      2. 编码 T_contact 之前的历史帧（或首帧）作为 context
      3. Factual  路：用原始动作做 H 步开环预测
      4. CF 路：用干预动作做 H 步开环预测
      5. 同步编码真实帧，分别计算 fact_mse / cf_mse / divergence

    Returns:
        包含各项指标的 dict，或 None（若 episode 无有效接触帧或长度不足）。
    """
    # ── Step 0: 推断 frameskip ───────────────────────────────────────────────
    act_input_dim = model.action_encoder.patch_embed.weight.shape[1]
    raw_act_dim   = episode.actions.shape[-1]
    frameskip     = act_input_dim // raw_act_dim

    # ── Step 1: 关键帧检测 ───────────────────────────────────────────────────
    t_contact_native = find_contact_frame(episode.states, contact_thresh)
    if t_contact_native is None:
        return None  # 该 episode 从未接触，跳过

    # 对齐到模型步（model step = frameskip 个 native 步）
    # T_contact 在模型步粒度上的索引
    t_contact_model = t_contact_native // frameskip
    if t_contact_model < 1:
        t_contact_model = 1  # 至少保留 1 步历史

    # 剩余可用模型步数
    native_len  = len(episode)                  # = len(actions)
    model_steps = native_len // frameskip
    available   = model_steps - t_contact_model
    H = min(horizon, available)
    if H < 1:
        return None

    # ── Step 2: 构造反事实动作序列 ────────────────────────────────────────────
    cf_start_native = t_contact_model * frameskip
    cf_length_native = H * frameskip
    cf_actions = make_counterfactual_actions(
        episode.actions, cf_start_native, cf_length_native, cf_mode, rng
    )

    # ── Step 3: 编码 T_contact 处的起始帧作为 context ────────────────────────
    # 用接触帧本身（真实像素）作为分叉起点
    start_frame_idx = t_contact_model * frameskip
    obs_start = _to_model_input(episode.frames[start_frame_idx], transform, device)
    info = {"pixels": obs_start}
    model.encode(info)
    emb_start = info["emb"][:, -1:, :]   # (1, 1, D)

    # ── Step 4: 双路 Unroll ───────────────────────────────────────────────────
    fact_mse_list  = []
    cf_mse_list    = []
    diverg_list    = []

    emb_ctx_fact = emb_start.clone()
    emb_ctx_cf   = emb_start.clone()
    act_hist_fact: list[torch.Tensor] = []
    act_hist_cf:   list[torch.Tensor] = []

    for step in range(H):
        native_t = t_contact_model * frameskip + step * frameskip

        # ── 拼接动作块 ──────────────────────────────────────────────────────
        def _make_act_emb(actions_src, hist):
            a_block = actions_src[native_t : native_t + frameskip]
            a_t = torch.tensor(a_block.flatten(), dtype=torch.float32,
                               device=device)
            a_t = a_t.unsqueeze(0).unsqueeze(0)       # (1, 1, act_input_dim)
            curr_emb = model.action_encoder(a_t)       # (1, 1, A_emb)
            hist.append(curr_emb)
            recent = hist[-history_size:]
            return torch.cat(recent, dim=1)            # (1, min(step+1,HS), A_emb)

        act_ctx_fact = _make_act_emb(episode.actions, act_hist_fact)
        act_ctx_cf   = _make_act_emb(cf_actions,      act_hist_cf)

        # ── Factual 预测 ─────────────────────────────────────────────────────
        ctx_len = act_ctx_fact.shape[1]
        pred_fact = model.predict(
            emb_ctx_fact[:, -ctx_len:, :], act_ctx_fact
        )[:, -1:, :]  # (1, 1, D)

        # ── Counterfactual 预测 ──────────────────────────────────────────────
        ctx_len_cf = act_ctx_cf.shape[1]
        pred_cf = model.predict(
            emb_ctx_cf[:, -ctx_len_cf:, :], act_ctx_cf
        )[:, -1:, :]  # (1, 1, D)

        # ── 编码真实下一帧 ───────────────────────────────────────────────────
        next_frame_idx = start_frame_idx + (step + 1) * frameskip
        obs_next = _to_model_input(episode.frames[next_frame_idx],
                                   transform, device)
        info_true = {"pixels": obs_next}
        model.encode(info_true)
        true_emb = info_true["emb"][:, -1:, :]  # (1, 1, D)

        # ── 指标 ─────────────────────────────────────────────────────────────
        fact_mse  = torch.mean((pred_fact - true_emb) ** 2).item()
        cf_mse    = torch.mean((pred_cf   - true_emb) ** 2).item()
        divergence = torch.mean((pred_fact - pred_cf)  ** 2).item()

        fact_mse_list.append(fact_mse)
        cf_mse_list.append(cf_mse)
        diverg_list.append(divergence)

        # ── 推进 context（各路独立）──────────────────────────────────────────
        emb_ctx_fact = torch.cat([emb_ctx_fact, pred_fact], dim=1)
        emb_ctx_cf   = torch.cat([emb_ctx_cf,   pred_cf],   dim=1)

    return {
        "t_contact_native": t_contact_native,
        "t_contact_model":  t_contact_model,
        "H":                H,
        "fact_mse":         np.array(fact_mse_list),
        "cf_mse":           np.array(cf_mse_list),
        "divergence":       np.array(diverg_list),
    }


# ── 聚合评测 ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_counterfactual_eval(
    model,
    episodes: list[Episode],
    contact_thresh: float,
    horizon: int,
    img_size: int,
    device: torch.device,
    history_size: int,
    cf_mode: str,
    rng: np.random.Generator,
) -> dict:
    """遍历所有 episodes，聚合反事实评测指标。"""
    transform = _make_transform(img_size)
    results_per_ep: list[dict] = []
    skipped = 0

    for i, ep in enumerate(episodes):
        res = counterfactual_unroll_episode(
            model=model,
            episode=ep,
            transform=transform,
            device=device,
            contact_thresh=contact_thresh,
            horizon=horizon,
            history_size=history_size,
            cf_mode=cf_mode,
            rng=rng,
        )
        if res is None:
            skipped += 1
        else:
            results_per_ep.append(res)

        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(episodes)} episodes "
                  f"（有效 {len(results_per_ep)}，跳过 {skipped}）")

    if not results_per_ep:
        return {"error": "没有找到任何含有效接触帧的 episode，"
                         "请调大 --contact_thresh"}

    # ── 对齐到最短 horizon 后聚合 ─────────────────────────────────────────────
    min_H = min(r["H"] for r in results_per_ep)
    fact_mat  = np.stack([r["fact_mse"][:min_H]  for r in results_per_ep])
    cf_mat    = np.stack([r["cf_mse"][:min_H]    for r in results_per_ep])
    divg_mat  = np.stack([r["divergence"][:min_H] for r in results_per_ep])

    contact_steps = [r["t_contact_native"] for r in results_per_ep]

    return {
        "n_valid_episodes":      len(results_per_ep),
        "n_skipped_episodes":    skipped,
        "horizon":               min_H,
        "contact_thresh":        contact_thresh,
        "cf_mode":               cf_mode,

        # per-step curves (mean ± std across episodes)
        "mean_fact_mse":         fact_mat.mean(axis=0).tolist(),
        "std_fact_mse":          fact_mat.std(axis=0).tolist(),
        "mean_cf_mse":           cf_mat.mean(axis=0).tolist(),
        "std_cf_mse":            cf_mat.std(axis=0).tolist(),
        "mean_divergence":       divg_mat.mean(axis=0).tolist(),
        "std_divergence":        divg_mat.std(axis=0).tolist(),

        # scalar summaries
        "total_mean_fact_mse":   float(fact_mat.mean()),
        "total_mean_cf_mse":     float(cf_mat.mean()),
        "total_mean_divergence": float(divg_mat.mean()),
        "final_divergence":      float(divg_mat[:, -1].mean()),

        # divergence slope (linear fit)
        "divergence_slope": float(
            np.polyfit(np.arange(min_H), divg_mat.mean(axis=0), 1)[0]
        ),

        # contact frame stats
        "contact_step_mean": float(np.mean(contact_steps)),
        "contact_step_std":  float(np.std(contact_steps)),
    }


# ── 可视化 ─────────────────────────────────────────────────────────────────────

def plot_results(metrics: dict, out_path: str) -> None:
    """绘制三条曲线：Factual MSE、CF MSE、Divergence。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib 未安装，跳过绘图")
        return

    steps = np.arange(metrics["horizon"])

    mean_fact = np.array(metrics["mean_fact_mse"])
    std_fact  = np.array(metrics["std_fact_mse"])
    mean_cf   = np.array(metrics["mean_cf_mse"])
    std_cf    = np.array(metrics["std_cf_mse"])
    mean_divg = np.array(metrics["mean_divergence"])
    std_divg  = np.array(metrics["std_divergence"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # ── 左图：Factual vs CF vs True MSE ────────────────────────────────────
    ax = axes[0]
    ax.plot(steps, mean_fact, color="#1f77b4", lw=2, label="Factual MSE (pred vs true)")
    ax.fill_between(steps, mean_fact - std_fact, mean_fact + std_fact,
                    alpha=0.2, color="#1f77b4")
    ax.plot(steps, mean_cf, color="#ff7f0e", lw=2,
            label=f"CF MSE (pred vs true, mode={metrics['cf_mode']})")
    ax.fill_between(steps, mean_cf - std_cf, mean_cf + std_cf,
                    alpha=0.2, color="#ff7f0e")
    ax.set_xlabel("Steps after T_contact", fontsize=11)
    ax.set_ylabel("Latent MSE  ||z_pred - z_true||²", fontsize=11)
    ax.set_title("Factual vs Counterfactual Prediction Error", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 右图：Divergence（两路预测的距离）──────────────────────────────────
    ax = axes[1]
    ax.plot(steps, mean_divg, color="#2ca02c", lw=2,
            label="Divergence ||z_fact - z_cf||²")
    ax.fill_between(steps, mean_divg - std_divg, mean_divg + std_divg,
                    alpha=0.2, color="#2ca02c")

    slope = metrics["divergence_slope"]
    intercept = mean_divg[0]
    ax.plot(steps, slope * steps + intercept,
            "--", color="red", lw=1.2,
            label=f"Linear fit (slope={slope:.4e})")

    ax.set_xlabel("Steps after T_contact", fontsize=11)
    ax.set_ylabel("Latent Divergence  ||z_fact - z_cf||²", fontsize=11)
    ax.set_title("Factual vs Counterfactual Latent Divergence", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Push-T Counterfactual Eval  "
        f"(n={metrics['n_valid_episodes']}, "
        f"thresh={metrics['contact_thresh']}, "
        f"cf={metrics['cf_mode']})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存至 {out_path}")


# ── CLI 入口 ───────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Push-T Counterfactual Evaluation — Action Intervention"
    )
    p.add_argument(
        "--dataset", default="pusht_expert_train",
        help="数据集名称（$STABLEWM_HOME/<name>.h5），默认 pusht_expert_train"
    )
    p.add_argument(
        "--policy", default="pusht/lewm",
        help="模型引用：可传 pusht/lewm，.pt 路径，或含 weights.pt 的目录"
    )
    p.add_argument(
        "--contact_thresh", type=float, default=80.0,
        help="Agent-Block 接触距离阈值（PushT 坐标系像素，默认 80）"
    )
    p.add_argument(
        "--cf_mode", choices=["zero", "random"], default="zero",
        help="反事实动作模式：zero=静止，random=随机游走（默认 zero）"
    )
    p.add_argument(
        "--n_episodes", type=int, default=50,
        help="评测 episode 数量（默认 50）"
    )
    p.add_argument(
        "--horizon", type=int, default=15,
        help="接触后最大展开步数 H（模型步，默认 15）"
    )
    p.add_argument(
        "--history_size", type=int, default=1,
        help="预测器上下文窗口（对应训练时 wm.history_size，默认 1）"
    )
    p.add_argument(
        "--img_size", type=int, default=224,
        help="输入图像分辨率（默认 224）"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认 42）"
    )
    p.add_argument(
        "--output", default="logs_eval/pusht_counterfactual_results.json",
        help="JSON 结果输出路径"
    )
    p.add_argument(
        "--plot", default="logs_eval/pusht_counterfactual_curves.png",
        help="曲线图输出路径（传空字符串跳过）"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    print(f"设备: {device}")
    print(f"反事实模式: {args.cf_mode}，接触阈值: {args.contact_thresh}")

    print("加载数据集...")
    episodes = load_episodes(args.dataset, args.n_episodes, rng)
    if not episodes:
        print("错误：未加载到任何有效 episode")
        return

    print("加载模型...")
    model = load_model(args.policy, device)

    print(f"\n开始评测：{len(episodes)} episodes，"
          f"horizon={args.horizon}，history_size={args.history_size}")
    t0 = time.time()
    metrics = run_counterfactual_eval(
        model=model,
        episodes=episodes,
        contact_thresh=args.contact_thresh,
        horizon=args.horizon,
        img_size=args.img_size,
        device=device,
        history_size=args.history_size,
        cf_mode=args.cf_mode,
        rng=np.random.default_rng(args.seed + 1),  # 独立 rng 供 random 模式
    )
    elapsed = time.time() - t0

    # ── 打印摘要 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("PUSH-T COUNTERFACTUAL EVALUATION RESULTS")
    print("=" * 64)
    if "error" in metrics:
        print(f"  错误: {metrics['error']}")
        return

    print(f"  有效 episodes        : {metrics['n_valid_episodes']}"
          f"  （跳过 {metrics['n_skipped_episodes']}）")
    print(f"  horizon              : {metrics['horizon']} steps")
    print(f"  contact_thresh       : {metrics['contact_thresh']}")
    print(f"  cf_mode              : {metrics['cf_mode']}")
    print(f"  mean_fact_mse        : {metrics['total_mean_fact_mse']:.6f}")
    print(f"  mean_cf_mse          : {metrics['total_mean_cf_mse']:.6f}")
    print(f"  mean_divergence      : {metrics['total_mean_divergence']:.6f}")
    print(f"  final_divergence     : {metrics['final_divergence']:.6f}  "
          f"（末步 fact vs cf 距离）")
    print(f"  divergence_slope     : {metrics['divergence_slope']:.6e}  "
          f"（正值 = 两路预测随时间分叉）")
    print(f"  contact_step (mean)  : {metrics['contact_step_mean']:.1f} ± "
          f"{metrics['contact_step_std']:.1f} native steps")
    print(f"\n总耗时: {elapsed:.1f}s")

    # ── 可视化 ────────────────────────────────────────────────────────────────
    if args.plot:
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        plot_results(metrics, args.plot)

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "args":            vars(args),
        "metrics":         metrics,
        "elapsed_seconds": elapsed,
    }
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"结果已写入 {out_path}")


if __name__ == "__main__":
    main()
