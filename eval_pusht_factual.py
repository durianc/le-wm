"""
eval_pusht_factual.py — Push-T 事实评测：开环自回归 latent MSE

脚本从真实数据集中读取观测帧，按训练时的动作编码与预测器接口做开环 unroll，统计每一步预测 embedding 与真实 embedding 的均方误差。

数据与模型解析方式：
    - `--dataset pusht` 会自动解析为 `$STABLEWM_HOME/pusht_expert_train.h5`
    - `--policy pusht/lewm` 会自动解析为 `$STABLEWM_HOME/pusht/lewm/weights.pt`
    - 仍然兼容直接传入 `.h5` 数据集路径，或 `.pt` / 目录形式的模型路径

支持的数据集格式：
    - h5: 读取 `pixels`、`action`、`ep_offset`、`ep_len`

使用示例：
    python eval_pusht_factual.py --dataset pusht --policy pusht/lewm
    python eval_pusht_factual.py --dataset pusht --policy random
    python eval_pusht_factual.py --dataset /abs/path/to/pusht_expert_train.h5 --policy /abs/path/to/weights.pt
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
        T.ToImage(),                                         # HWC uint8 → CHW uint8
        T.ToDtype(torch.float32, scale=True),                # → [0,1] float32
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),              # ImageNet 归一化
        T.Resize(size=img_size),                             # → (img_size, img_size)
    ])


def _to_model_input(frame_np: np.ndarray, transform: T.Compose,
                    device: torch.device) -> torch.Tensor:
    """np.ndarray (H,W,C) uint8 → Tensor (1,1,C,H,W) on device."""
    return transform(frame_np).unsqueeze(0).unsqueeze(0).to(device)


def _get_cache_dir() -> Path:
    """返回 stable-worldmodel 缓存目录（等同于 $STABLEWM_HOME）。"""
    return Path(swm.data.utils.get_cache_dir())


def _resolve_dataset_path(dataset_ref: str) -> tuple[Path, str]:
    """解析数据集引用，优先支持短名，其次兼容直接路径。"""
    p = Path(dataset_ref)
    if p.exists():
        if p.suffix == ".h5":
            return p, "h5"
        raise ValueError(f"无法从数据集路径推断格式: {dataset_ref!r}，请传入 .h5 文件")

    cache_dir = _get_cache_dir()
    candidates = [
        cache_dir / dataset_ref,
        cache_dir / f"{dataset_ref}.h5",
    ]
    for candidate in candidates:
        if candidate.exists():
            if candidate.suffix == ".h5":
                return candidate, "h5"

    if dataset_ref == "pusht":
        candidate = cache_dir / "pusht_expert_train.h5"
        if candidate.exists():
            return candidate, "h5"

    raise FileNotFoundError(
        f"找不到数据集 '{dataset_ref}'。请传入绝对路径，或使用 $STABLEWM_HOME 下的短名。"
    )


# ── 数据集加载 ─────────────────────────────────────────────────────────────────

class Episode:
    """单条 episode 的裸数据容器（numpy，未经预处理）。"""
    __slots__ = ("frames", "actions")

    def __init__(self, frames: np.ndarray, actions: np.ndarray):
        # frames:  (T+1, H, W, C) uint8  — 对应 t=0..T 的观测帧
        # actions: (T,   2)       float  — 对应 a_0..a_{T-1}
        assert len(frames) == len(actions) + 1, (
            f"frames={len(frames)} 应比 actions={len(actions)} 多 1"
        )
        self.frames  = frames
        self.actions = actions

    def __len__(self) -> int:
        return len(self.actions)   # T


def _iter_episodes_h5(
    dataset_path: str,
    n_episodes: int,
    rng: np.random.Generator,
) -> Iterator[Episode]:
    """从 h5py 格式的 stable-worldmodel HDF5 数据集读取 episodes。

    HDF5 实际格式（pusht_expert_train.h5）：
        pixels    (N, 224, 224, 3) uint8   — HWC，blosc 压缩（需 hdf5plugin）
        action    (N, 2)           float32
        ep_offset (E,) int64       — episode 在全局行中的起始索引
        ep_len    (E,) int32       — episode 含多少行（每行一个时间步）

    对齐约定：
        frame[t] 是执行 action[t] 之前的观测，执行后到达 frame[t+1]。
        ep_len=l 的 episode 有 l 帧、l-1 对有效 (o_t, a_t, o_{t+1}) 转换。
    """
    import hdf5plugin  # noqa: F401 — 注册 blosc 解码器，必须先于 h5py 读取 pixels
    import h5py

    with h5py.File(dataset_path, "r") as f:
        ep_offsets = f["ep_offset"][:]
        ep_lens    = f["ep_len"][:]
        n_ep_avail = len(ep_offsets)
        indices = rng.choice(n_ep_avail, size=min(n_episodes, n_ep_avail), replace=False)

        for idx in indices:
            off = int(ep_offsets[idx])
            l   = int(ep_lens[idx])
            if l < 2:
                continue
            # pixels: (l, H, W, C) uint8 — HWC，直接传给 _to_model_input
            frames  = f["pixels"][off : off + l]      # (l, H, W, C)
            # actions: (l-1, act_dim) — 有效转换步数
            actions = f["action"][off : off + l - 1]  # (l-1, act_dim)
            yield Episode(frames=frames, actions=actions)


def load_episodes(
    dataset_path: str,
    n_episodes: int,
    rng: np.random.Generator,
) -> list[Episode]:
    """统一入口：返回最多 n_episodes 条 Episode 对象（numpy，无预处理）。"""
    episodes = list(_iter_episodes_h5(dataset_path, n_episodes, rng))
    print(f"  加载 {len(episodes)} 条 episodes（来自 {dataset_path}）")
    return episodes


# ── 模型加载 ───────────────────────────────────────────────────────────────────

def load_model(policy: str, device: torch.device):
    """加载 LeWM-style JEPA 模型。

    支持以下来源（优先级从高到低）：
      1. policy="random" → 返回 None（随机 encoder 基线，MSE 无参考意义）
      2. *.pt 文件路径   → 直接作为 weights.pt state_dict 加载
      3. 目录路径        → 读取目录内的 weights.pt
      4. checkpoint 名称 → 在 $STABLEWM_HOME 下按 eval.py 同样逻辑查找 weights.pt

    加载方式与 eval.py 的 load_lewm_model() 完全一致，通过权重形状自动推断架构。
    """
    if policy == "random":
        print("  policy=random：使用随机 encoder（MSE 无参考意义）")
        return None

    p = Path(policy)

    # ── 情况 2/3/4：统一走 _load_from_weights ────────────────────────────────
    # 与 eval.py load_lewm_model() 相同的查找顺序：
    #   a. 直接是 .pt 文件
    #   b. 目录下的 weights.pt
    #   c. $STABLEWM_HOME/<policy>/weights.pt
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
            "请提供以下之一：\n"
            "  • .pt 权重文件的绝对路径\n"
            "  • 含 weights.pt 的目录路径\n"
            "  • $STABLEWM_HOME 下含 weights.pt 的子目录名称\n"
            "  • 'random'（使用随机 encoder 基线）"
        )

    return _load_from_weights(weights_pt, device)


def _get_cache_dir() -> Path:
    """返回 stable-worldmodel 缓存目录（等同于 $STABLEWM_HOME）。"""
    import stable_worldmodel as swm
    return Path(swm.data.utils.get_cache_dir())


def _load_from_weights(weights_path: "str | Path", device: torch.device):
    """从 state_dict .pt 文件重建 JEPA，与 eval.py load_lewm_model() 完全一致。

    通过权重 tensor 形状自动推断架构，无需 config.json。
    """
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP
    from stable_pretraining.backbone.utils import vit_hf

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    # 推断架构参数（与 eval.py 一致）
    act_input_dim    = ckpt["action_encoder.patch_embed.weight"].shape[1]
    hidden_dim       = ckpt["projector.net.0.weight"].shape[0]   # Linear(embed_dim→hidden_dim)
    embed_dim        = ckpt["projector.net.0.weight"].shape[1]   # = input_dim of projector
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
    # The predictor operates fully in embed_dim space (hidden_dim = embed_dim = 192),
    # NOT in the projector's hidden_dim (2048). This matches eval.py exactly.
    predictor = ARPredictor(
        num_frames=predictor_frames,
        input_dim=embed_dim, hidden_dim=embed_dim, output_dim=embed_dim,
        depth=predictor_depth, heads=16, mlp_dim=2048, dim_head=64,
        dropout=0.1, emb_dropout=0.0,
    )
    action_encoder = Embedder(input_dim=act_input_dim, emb_dim=embed_dim)
    projector  = MLP(input_dim=embed_dim, hidden_dim=hidden_dim, output_dim=embed_dim, norm_fn=proj_norm)
    pred_proj  = MLP(input_dim=embed_dim, hidden_dim=hidden_dim, output_dim=embed_dim, norm_fn=proj_norm)

    model = JEPA(encoder, predictor, action_encoder, projector, pred_proj)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        raise RuntimeError(f"权重缺失: {missing}")
    if unexpected:
        print(f"  [warn] 忽略多余的权重键: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    print(f"  模型架构: embed_dim={embed_dim}, hidden_dim={hidden_dim}, "
          f"predictor_depth={predictor_depth}, act_input_dim={act_input_dim}, "
          f"predictor_frames={predictor_frames}")
    return model.to(device).eval().requires_grad_(False)


# ── 核心评测循环 ───────────────────────────────────────────────────────────────

@torch.no_grad()
def factual_unroll_episode(
    model,
    episode: Episode,
    transform: T.Compose,
    device: torch.device,
    horizon: int,
    history_size: int = 1,
) -> np.ndarray:
    """单 episode 的 H 步开环自回归 Unroll。

    流程：
      z_0^pred = Enc(img_0)
      for t in 0..H-1:
        z_{t+1}^pred = Pred(z_{t-HS+1:t+1}^pred, a_{t-HS+1:t+1})
        z_{t+1}^true = Enc(img_{t+1})
        mse[t] = ||z_{t+1}^pred - z_{t+1}^true||²

    Args:
        horizon:      最大展开步数 H（实际 = min(H, episode 长度)）
        history_size: 预测器的上下文窗口大小（对应训练时的 history_size）

    Returns:
        mse_per_step: np.ndarray shape (H,)
    """
    # 推断 frameskip：act_input_dim = frameskip × raw_act_dim
    # Conv1d weight: (out_channels, in_channels, kernel=1)，shape[1] = in_channels = act_input_dim
    act_input_dim = model.action_encoder.patch_embed.weight.shape[1]
    raw_act_dim   = episode.actions.shape[-1]
    frameskip     = act_input_dim // raw_act_dim   # 通常 = 5

    # 按 frameskip 对齐：每一"模型步"跨越 frameskip 个原始时间步
    # native step t → 原始帧 frame[t*frameskip]，动作 = concat(a[t*fs], ..., a[(t+1)*fs - 1])
    # 可用的模型步数受限于 episode 长度和 horizon
    native_len  = len(episode)              # = len(actions) = l - 1（Episode 约定）
    model_steps = native_len // frameskip   # 可完整拼出的动作块数
    H = min(horizon, model_steps)
    if H < 1:
        return np.array([])

    # ── Step 1: 编码初始帧 ────────────────────────────────────────────────────
    obs_0 = _to_model_input(episode.frames[0], transform, device)  # (1, 1, C, H, W)
    info  = {"pixels": obs_0}
    model.encode(info)
    emb_context = info["emb"][:, -1:, :]   # (1, 1, D)

    act_embs_history: list[torch.Tensor] = []
    mse_per_step: list[float] = []

    # ── Step 2: 自回归 Unroll ─────────────────────────────────────────────────
    for t in range(H):
        # 拼接 frameskip 个原始动作，还原训练时的 effective action 向量
        a_block = episode.actions[t * frameskip : (t + 1) * frameskip]   # (frameskip, raw_act_dim)
        a_t = torch.tensor(a_block.flatten(), dtype=torch.float32, device=device)
        a_t = a_t.unsqueeze(0).unsqueeze(0)                               # (1, 1, act_input_dim)

        curr_act_emb = model.action_encoder(a_t)                          # (1, 1, A_emb)
        act_embs_history.append(curr_act_emb)

        # 截取最近 history_size 步的上下文
        recent_acts = act_embs_history[-history_size:]
        act_context = torch.cat(recent_acts, dim=1)   # (1, min(t+1, HS), A_emb)
        context_len = act_context.shape[1]
        emb_trunc   = emb_context[:, -context_len:, :]

        # 预测下一步隐状态
        pred_emb_next = model.predict(emb_trunc, act_context)[:, -1:, :]  # (1, 1, D)

        # ── Step 3: 编码真实的下一帧（frameskip 步之后的帧）──────────────────
        next_frame_idx = (t + 1) * frameskip   # 在原始 episode 帧序列中的索引
        obs_next = _to_model_input(episode.frames[next_frame_idx], transform, device)
        info_true = {"pixels": obs_next}
        model.encode(info_true)
        true_emb_next = info_true["emb"][:, -1:, :]   # (1, 1, D)

        # ── Step 4: 隐空间 MSE ───────────────────────────────────────────────
        mse = torch.mean((pred_emb_next - true_emb_next) ** 2).item()
        mse_per_step.append(mse)

        # ── Step 5: 用预测 embedding 推进 context（Open-loop）──────────────
        emb_context = torch.cat([emb_context, pred_emb_next], dim=1)

    return np.array(mse_per_step)


@torch.no_grad()
def run_factual_eval(
    model,
    episodes: list[Episode],
    horizon: int,
    img_size: int,
    device: torch.device,
    history_size: int = 1,
) -> dict:
    """遍历所有 episodes，收集每步 MSE，汇总统计指标。"""
    transform = _make_transform(img_size)
    all_mse: list[np.ndarray] = []

    for i, ep in enumerate(episodes):
        mse = factual_unroll_episode(
            model, ep, transform, device, horizon, history_size
        )
        if len(mse) > 0:
            all_mse.append(mse)
        if (i + 1) % 10 == 0:
            print(f"  进度: {i + 1}/{len(episodes)} episodes")

    if not all_mse:
        return {"error": "无有效 episode"}

    # 对齐到最短序列后计算统计量
    min_len    = min(len(m) for m in all_mse)
    mse_matrix = np.stack([m[:min_len] for m in all_mse])  # (N_ep, H)
    mean_curve = mse_matrix.mean(axis=0)                    # (H,)
    std_curve  = mse_matrix.std(axis=0)                     # (H,)
    steps      = np.arange(min_len)

    # 线性回归拟合漂移斜率（MSE/step）
    drift_slope = float(np.polyfit(steps, mean_curve, 1)[0])

    return {
        "n_episodes":        len(all_mse),
        "horizon":           min_len,
        "mean_mse_per_step": mean_curve.tolist(),
        "std_mse_per_step":  std_curve.tolist(),
        "mean_total_mse":    float(mse_matrix.mean()),
        "drift_slope":       drift_slope,
    }


# ── 可视化 ─────────────────────────────────────────────────────────────────────

def plot_mse_curve(metrics: dict, out_path: str) -> None:
    """绘制 Timestep vs. Average MSE 曲线图并保存。"""
    try:
        import matplotlib
        matplotlib.use("Agg")   # 无 display 环境安全
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib 未安装，跳过绘图")
        return

    mean_curve = np.array(metrics["mean_mse_per_step"])
    std_curve  = np.array(metrics["std_mse_per_step"])
    steps      = np.arange(len(mean_curve))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, mean_curve, color="#1f77b4", linewidth=2, label="Mean MSE")
    ax.fill_between(
        steps,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.25, color="#1f77b4", label="±1 std",
    )

    # 线性拟合漂移线
    slope = metrics["drift_slope"]
    intercept = mean_curve[0] - slope * 0
    ax.plot(steps, slope * steps + intercept,
            "--", color="red", linewidth=1.2,
            label=f"Linear fit (slope={slope:.4e})")

    ax.set_xlabel("Timestep $t$", fontsize=12)
    ax.set_ylabel("Latent MSE $||z^{pred} - z^{true}||^2$", fontsize=12)
    ax.set_title("Push-T Factual Eval — Open-Loop Latent Drift", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  MSE 曲线已保存至 {out_path}")


# ── CLI 入口 ───────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Push-T Factual Evaluation — Open-Loop Latent MSE"
    )
    p.add_argument(
        "--dataset", required=True,
        help="数据集引用或路径：可传 pusht，也可传 zarr 目录或 h5 文件"
    )
    p.add_argument(
        "--dataset_fmt", choices=["h5"], default="h5",
        help="数据集格式；传短名 pusht 时会自动识别，传绝对路径时用于兜底"
    )
    p.add_argument(
        "--policy", default="random",
        help="模型引用或路径：可传 pusht/lewm，也可传 .pt 或含 weights.pt 的目录"
    )
    p.add_argument(
        "--n_episodes", type=int, default=50,
        help="评测 episode 数量（默认 50）"
    )
    p.add_argument(
        "--horizon", type=int, default=20,
        help="每条 episode 最大展开步数 H（默认 20）"
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
        help="随机种子（控制 episode 采样，默认 42）"
    )
    p.add_argument(
        "--output", default="logs_eval/pusht_factual_results.json",
        help="JSON 结果输出路径"
    )
    p.add_argument(
        "--plot", default="logs_eval/pusht_factual_mse_curve.png",
        help="MSE 曲线图输出路径（传空字符串跳过）"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    dataset_path, inferred_fmt = _resolve_dataset_path(args.dataset)
    args.dataset_fmt = inferred_fmt

    print(f"设备: {device}")
    print("加载数据集...")
    episodes = load_episodes(str(dataset_path), args.n_episodes, rng)
    if not episodes:
        print("错误：未加载到任何有效 episode，请检查数据集路径和格式")
        return

    print("加载模型...")
    model = load_model(args.policy, device)

    print(f"\n开始评测：{len(episodes)} episodes，horizon={args.horizon}，"
          f"history_size={args.history_size}")
    t0 = time.time()
    metrics = run_factual_eval(
        model=model,
        episodes=episodes,
        horizon=args.horizon,
        img_size=args.img_size,
        device=device,
        history_size=args.history_size,
    )
    elapsed = time.time() - t0

    # ── 打印摘要 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PUSH-T FACTUAL EVALUATION RESULTS")
    print("=" * 60)
    if "error" in metrics:
        print(f"  错误: {metrics['error']}")
        return

    print(f"  n_episodes      : {metrics['n_episodes']}")
    print(f"  horizon         : {metrics['horizon']} steps")
    print(f"  mean_total_mse  : {metrics['mean_total_mse']:.6f}")
    print(f"  drift_slope     : {metrics['drift_slope']:.6e}  (MSE/step)")
    curve = metrics["mean_mse_per_step"]
    print(f"  MSE curve       : t=0 → {curve[0]:.4f},  "
          f"t={len(curve)-1} → {curve[-1]:.4f}")
    print(f"\n总耗时: {elapsed:.1f}s")

    # ── 可视化 ────────────────────────────────────────────────────────────────
    if args.plot:
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        plot_mse_curve(metrics, args.plot)

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "args": vars(args),
        "metrics": metrics,
        "elapsed_seconds": elapsed,
    }
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"结果已写入 {out_path}")


if __name__ == "__main__":
    main()
