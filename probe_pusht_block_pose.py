"""
probe_pusht_block_pose.py — 训练 Push-T Block Pose Probe

从冻结的 LeWM encoder 中提取每帧的 CLS-token embedding，训练一个 MLP 回归头
将 embedding 映射到 Block 物理位置：

    embedding (D,) → [bx, by, sin(θ), cos(θ)]

目标变量说明（来自 HDF5 state 列，格式 [ax, ay, bx, by, b_angle, vx, vy]）：
  bx, by   — Block 中心坐标，PushT 内部坐标系（窗口 512×512）
  b_angle  — Block 旋转角（弧度，无界），用 sin/cos 双通道编码以消除角度跳变

坐标系：PushT 内部物理坐标，范围大致 [50, 450]×[50, 450]。
        不映射到像素坐标，直接在物理坐标空间做回归，与 env.eval_state() 一致。

Embedding 来源：直接读取 HDF5 的 pixels 列（预渲染帧），调用 model.encode()，
        无需启动物理环境（与 tworoom probe.py 的渲染式提取不同）。

Usage:
    python probe_pusht_block_pose.py --policy pusht/lewm --dataset pusht_expert_train
    python probe_pusht_block_pose.py --policy pusht/lewm --dataset pusht_expert_train \\
        --probe_ckpt logs_eval/probes/pusht_block_pose_probe.pt --epochs 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import v2 as T
from tqdm import tqdm

import stable_worldmodel as swm


# ── 坐标空间常量（PushT 内部物理坐标系）──────────────────────────────────────────

# PushT 窗口 512×512，Block 可活动范围约 [50, 450]
PUSHT_XY_MIN = 50.0
PUSHT_XY_MAX = 450.0

# state 列列索引（格式：[ax, ay, bx, by, b_angle, vx, vy]）
STATE_BX_IDX    = 2
STATE_BY_IDX    = 3
STATE_ANGLE_IDX = 4

# probe 输出维度：[bx, by, sin(θ), cos(θ)]
PROBE_OUT_DIM = 4


# ── Probe 模型 ────────────────────────────────────────────────────────────────

class BlockPoseProbe(nn.Module):
    """2 层 MLP 回归头：embedding → [bx, by, sin(θ), cos(θ)]。

    输出 4 维而非 3 维，将角度拆成 sin/cos 双通道，避免角度跳变（如
    从 2π-ε 到 0 的不连续性）破坏回归损失。推理时可用 atan2 还原角度。

    Args:
        in_dim:     输入 embedding 维度（通常 192，ViT-Tiny CLS token）
        hidden_dim: 隐层宽度（默认 256）
        out_dim:    输出维度，固定为 4（bx, by, sin_θ, cos_θ）
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256,
                 out_dim: int = PROBE_OUT_DIM) -> None:
        super().__init__()
        assert out_dim == PROBE_OUT_DIM, (
            f"out_dim 必须为 {PROBE_OUT_DIM}，收到 {out_dim}"
        )
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Args:
            emb: (N, D) 或 (N, T, D) — 若 3D 则取最后时间步
        Returns:
            (N, 4) — [bx_norm, by_norm, sin_θ, cos_θ]（bx/by 为归一化值）
        """
        if emb.ndim == 3:
            emb = emb[:, -1, :]
        return self.net(emb)

    @torch.no_grad()
    def decode_pose(
        self,
        pred: torch.Tensor,
        xy_mean: torch.Tensor,
        xy_std: torch.Tensor,
    ) -> torch.Tensor:
        """将网络输出还原为物理坐标 [bx, by, b_angle]。

        Args:
            pred:    (N, 4) 网络输出 [bx_norm, by_norm, sin_θ, cos_θ]
            xy_mean: (2,) bx/by 的训练集均值
            xy_std:  (2,) bx/by 的训练集标准差

        Returns:
            (N, 3) float32 — [bx, by, b_angle]，物理坐标系
        """
        xy_norm = pred[:, :2]                        # (N, 2)
        sin_t   = pred[:, 2:3]                       # (N, 1)
        cos_t   = pred[:, 3:4]                       # (N, 1)

        xy = xy_norm * xy_std.to(pred.device) + xy_mean.to(pred.device)  # (N, 2)
        angle = torch.atan2(sin_t, cos_t)            # (N, 1)，范围 (-π, π]

        return torch.cat([xy, angle], dim=1)         # (N, 3)


# ── LeWM 模型加载（与 eval_pusht_factual.py 一致）─────────────────────────────

def _get_cache_dir() -> Path:
    return Path(swm.data.utils.get_cache_dir())


def load_lewm_model(policy: str, device: torch.device) -> nn.Module:
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
            f"找不到模型权重 '{policy}'。请提供 .pt 路径、含 weights.pt 的目录，"
            "或 $STABLEWM_HOME 下的子目录名称。"
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

    print(f"  LeWM 架构: embed_dim={embed_dim}, predictor_depth={predictor_depth}")
    return model.to(device).eval().requires_grad_(False)


# ── Embedding 提取 ─────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    h5_path: str,
    img_size: int,
    device: torch.device,
    batch_size: int = 128,
    max_frames: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从 HDF5 逐帧读取 pixels，批量编码，收集 block pose 标签。

    直接读取数据集预渲染帧，无需启动 PushT 物理环境，速度快。

    每帧独立编码（history_size=1），与 eval 推理时分叉点的 context
    编码方式保持一致。

    Args:
        model:      JEPA 模型（已冻结，eval 模式）
        h5_path:    HDF5 数据集路径
        img_size:   图像分辨率（应与训练一致，通常 224）
        device:     torch 设备
        batch_size: 编码时的 batch 大小（受 GPU 显存约束）
        max_frames: 若指定，只取前 max_frames 帧（调试用）

    Returns:
        embeddings: (N, D) float32 — CLS-token embedding
        targets:    (N, 4) float32 — [bx, by, sin(θ), cos(θ)]，未归一化的 bx/by
    """
    import hdf5plugin  # noqa: F401

    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        T.Resize(size=img_size),
    ])

    with h5py.File(h5_path, "r") as f:
        if "state" not in f:
            raise RuntimeError(
                "HDF5 缺少 'state' 列，无法提取 Block Pose 标签。"
            )

        total = f["pixels"].shape[0]
        if max_frames is not None:
            total = min(total, max_frames)

        states = f["state"][:total]   # (N, 7) — 一次性加载，state 列体积小

        # 逐 batch 读取 pixels（blosc 压缩，避免全量解压）
        emb_list: list[torch.Tensor] = []

        for start in tqdm(range(0, total, batch_size), desc="  提取 embeddings"):
            end = min(start + batch_size, total)
            frames = f["pixels"][start:end]        # (B, H, W, C) uint8

            # 预处理：逐帧 transform → stack → (B, 1, C, H, W)
            imgs = torch.stack([
                transform(frames[i]) for i in range(len(frames))
            ]).unsqueeze(1).to(device)             # (B, 1, C, H, W)

            info = {"pixels": imgs}
            model.encode(info)
            emb = info["emb"][:, -1, :].cpu()     # (B, D)
            emb_list.append(emb)

    embeddings = torch.cat(emb_list, dim=0)        # (N, D)

    # ── 构造目标变量 [bx, by, sin(θ), cos(θ)] ──────────────────────────────
    bx    = torch.tensor(states[:, STATE_BX_IDX],    dtype=torch.float32)
    by    = torch.tensor(states[:, STATE_BY_IDX],    dtype=torch.float32)
    angle = torch.tensor(states[:, STATE_ANGLE_IDX], dtype=torch.float32)

    # 角度可能超出 (-π, π] 范围（数据集记录原始弧度），sin/cos 天然处理跳变
    sin_t = torch.sin(angle)
    cos_t = torch.cos(angle)

    targets = torch.stack([bx, by, sin_t, cos_t], dim=1)  # (N, 4)

    print(f"  提取完成：{len(embeddings)} 帧，embedding dim={embeddings.shape[1]}")
    print(f"  Block 坐标范围  bx=[{bx.min():.1f}, {bx.max():.1f}]"
          f"  by=[{by.min():.1f}, {by.max():.1f}]")
    print(f"  角度范围（原始弧度）: [{angle.min():.3f}, {angle.max():.3f}]")

    return embeddings, targets


# ── 归一化辅助 ─────────────────────────────────────────────────────────────────

def compute_xy_stats(targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 bx/by 的均值和标准差（仅对坐标分量，不对 sin/cos）。

    sin/cos 本身有界 [-1, 1]，不需要归一化。

    Returns:
        xy_mean: (2,) — [bx_mean, by_mean]
        xy_std:  (2,) — [bx_std, by_std]，clamp 防止除零
    """
    xy = targets[:, :2]                 # (N, 2)
    xy_mean = xy.mean(dim=0)            # (2,)
    xy_std  = xy.std(dim=0).clamp(min=1e-6)
    return xy_mean, xy_std


def normalize_targets(
    targets: torch.Tensor,
    xy_mean: torch.Tensor,
    xy_std: torch.Tensor,
) -> torch.Tensor:
    """将 targets [bx, by, sin_θ, cos_θ] 中的 bx/by 归一化。

    sin_θ / cos_θ 保持原值（已在 [-1, 1]）。

    Returns:
        (N, 4) — [bx_norm, by_norm, sin_θ, cos_θ]
    """
    t = targets.clone()
    t[:, :2] = (t[:, :2] - xy_mean) / xy_std
    return t


# ── 训练 ──────────────────────────────────────────────────────────────────────

def train_probe(
    probe: BlockPoseProbe,
    embeddings: torch.Tensor,
    targets_norm: torch.Tensor,
    xy_mean: torch.Tensor,
    xy_std: torch.Tensor,
    epochs: int = 150,
    lr: float = 1e-3,
    batch_size: int = 512,
    val_frac: float = 0.1,
    device: torch.device | None = None,
) -> dict:
    """用 MSE 损失训练 BlockPoseProbe，返回训练/验证指标。

    损失设计：
      - bx/by 分量：归一化空间中的 MSE（量纲一致）
      - sin/cos 分量：原始空间 MSE（自然有界，无需归一化）
      - 总损失 = mean(所有 4 分量的 MSE)，不加权重（各分量量级相近）

    Args:
        probe:        BlockPoseProbe 实例（将被 in-place 修改）
        embeddings:   (N, D) 输入 embedding
        targets_norm: (N, 4) 归一化后的目标 [bx_norm, by_norm, sin_θ, cos_θ]
        xy_mean:      (2,) bx/by 均值（用于打印像素空间误差）
        xy_std:       (2,) bx/by 标准差（用于打印像素空间误差）
        epochs:       训练轮数
        lr:           Adam 学习率
        batch_size:   mini-batch 大小
        val_frac:     验证集比例
        device:       训练设备

    Returns:
        dict 包含 final_train_loss, final_val_loss, final_xy_rmse_px,
        final_angle_mae_deg（角度误差，度）
    """
    device = device or torch.device("cpu")
    probe  = probe.to(device)
    embeddings   = embeddings.to(device)
    targets_norm = targets_norm.to(device)

    # ── 划分训练/验证集 ─────────────────────────────────────────────────────
    N = len(embeddings)
    n_val   = max(1, int(N * val_frac))
    n_train = N - n_val

    dataset  = TensorDataset(embeddings, targets_norm)
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    # 余弦退火到 lr/10，让后期精细拟合
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr / 10
    )

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(epochs):
        # ── 训练 ───────────────────────────────────────────────────────────
        probe.train()
        train_loss, n = 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = probe(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            n += len(y)
        train_loss /= n
        scheduler.step()

        # ── 验证 ───────────────────────────────────────────────────────────
        probe.eval()
        val_loss, n_v = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = probe(x)
                val_loss += criterion(pred, y).item() * len(y)
                n_v += len(y)
        val_loss /= n_v

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone()
                             for k, v in probe.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:>4}/{epochs}  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    # 恢复最优权重
    if best_state is not None:
        probe.load_state_dict(best_state)

    # ── 计算物理空间误差（更直观） ─────────────────────────────────────────
    probe.eval()
    all_preds, all_tgts_norm = [], []
    with torch.no_grad():
        for x, y in DataLoader(TensorDataset(embeddings, targets_norm),
                               batch_size=batch_size):
            all_preds.append(probe(x).cpu())
            all_tgts_norm.append(y.cpu())

    preds_norm = torch.cat(all_preds)      # (N, 4)
    tgts_norm  = torch.cat(all_tgts_norm)  # (N, 4)

    # bx/by 误差（物理坐标系，单位：px）
    xy_std_cpu  = xy_std.cpu()
    xy_mean_cpu = xy_mean.cpu()
    pred_xy_px  = preds_norm[:, :2] * xy_std_cpu + xy_mean_cpu
    true_xy_px  = tgts_norm[:, :2]  * xy_std_cpu + xy_mean_cpu
    xy_rmse_px  = float(((pred_xy_px - true_xy_px) ** 2).mean().sqrt())

    # 角度误差（度），通过 sin/cos 还原 atan2
    pred_angle = torch.atan2(preds_norm[:, 2], preds_norm[:, 3])  # (-π, π]
    true_angle = torch.atan2(tgts_norm[:, 2],  tgts_norm[:, 3])
    angle_diff = torch.abs(pred_angle - true_angle)
    # 处理角度跳变：取 min(diff, 2π-diff)
    angle_diff = torch.minimum(angle_diff, 2 * np.pi - angle_diff)
    angle_mae_deg = float(angle_diff.mean() * 180.0 / np.pi)

    print(f"\n  最优 val_loss = {best_val_loss:.5f}")
    print(f"  Block 位置 RMSE = {xy_rmse_px:.2f} px  "
          f"（坐标范围 ~[50, 450]，PushT 接触阈值 ~60 px）")
    print(f"  Block 角度 MAE  = {angle_mae_deg:.2f}°  "
          f"（PushT 成功判定 ≤ 20°）")

    return {
        "best_val_loss":    best_val_loss,
        "final_xy_rmse_px": xy_rmse_px,
        "final_angle_mae_deg": angle_mae_deg,
    }


# ── CLI 入口 ───────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Push-T Block Pose Probe (embedding → bx, by, angle)"
    )
    p.add_argument(
        "--policy", default="pusht/lewm",
        help="LeWM 模型引用：可传 pusht/lewm，.pt 路径，或含 weights.pt 的目录"
    )
    p.add_argument(
        "--dataset", default="pusht_expert_train",
        help="数据集名称（$STABLEWM_HOME/<name>.h5）"
    )
    p.add_argument(
        "--probe_ckpt",
        default="logs_eval/probes/pusht_block_pose_probe.pt",
        help="probe 权重保存路径"
    )
    p.add_argument(
        "--epochs", type=int, default=150,
        help="训练轮数（默认 150）"
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Adam 学习率（默认 1e-3）"
    )
    p.add_argument(
        "--hidden_dim", type=int, default=256,
        help="MLP 隐层宽度（默认 256）"
    )
    p.add_argument(
        "--batch_size", type=int, default=512,
        help="训练 batch size（默认 512）"
    )
    p.add_argument(
        "--encode_batch_size", type=int, default=128,
        help="embedding 提取时的 batch size（受 GPU 显存约束，默认 128）"
    )
    p.add_argument(
        "--val_frac", type=float, default=0.1,
        help="验证集比例（默认 0.1）"
    )
    p.add_argument(
        "--img_size", type=int, default=224,
        help="图像分辨率（与 LeWM 训练一致，默认 224）"
    )
    p.add_argument(
        "--max_frames", type=int, default=None,
        help="最多使用多少帧（调试用，默认全部）"
    )
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备（默认 cuda 若可用）"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    h5_path = _get_cache_dir() / f"{args.dataset}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"数据集文件不存在: {h5_path}\n"
            f"请确认 $STABLEWM_HOME 设置正确（当前: {_get_cache_dir()}）"
        )

    # ── Step 1: 加载冻结的 LeWM encoder ────────────────────────────────────
    print("加载 LeWM 模型...")
    model = load_lewm_model(args.policy, device)

    # ── Step 2: 提取全数据集 embeddings + block pose 标签 ──────────────────
    print(f"\n提取 embeddings（数据集: {h5_path}）...")
    embeddings, targets = extract_embeddings(
        model=model,
        h5_path=str(h5_path),
        img_size=args.img_size,
        device=device,
        batch_size=args.encode_batch_size,
        max_frames=args.max_frames,
    )

    # ── Step 3: 计算归一化统计量（仅 bx/by，sin/cos 不需要）────────────────
    xy_mean, xy_std = compute_xy_stats(targets)
    print(f"\n归一化统计量（物理坐标系）:")
    print(f"  bx: mean={xy_mean[0]:.2f}  std={xy_std[0]:.2f}")
    print(f"  by: mean={xy_mean[1]:.2f}  std={xy_std[1]:.2f}")

    targets_norm = normalize_targets(targets, xy_mean, xy_std)

    # ── Step 4: 训练 probe ────────────────────────────────────────────────
    in_dim = embeddings.shape[1]
    probe  = BlockPoseProbe(in_dim=in_dim, hidden_dim=args.hidden_dim)
    print(f"\n训练 probe (in_dim={in_dim}, hidden_dim={args.hidden_dim}, "
          f"out_dim={PROBE_OUT_DIM})...")
    print(f"  训练集={int(len(embeddings)*(1-args.val_frac))}  "
          f"验证集={int(len(embeddings)*args.val_frac)}")

    metrics = train_probe(
        probe=probe,
        embeddings=embeddings,
        targets_norm=targets_norm,
        xy_mean=xy_mean,
        xy_std=xy_std,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        device=device,
    )

    # ── Step 5: 保存 probe 权重 ───────────────────────────────────────────
    save_path = Path(args.probe_ckpt)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            # 模型权重
            "state_dict": probe.state_dict(),
            # 架构参数（重建时用）
            "in_dim":     in_dim,
            "hidden_dim": args.hidden_dim,
            "out_dim":    PROBE_OUT_DIM,
            # 归一化统计量（推理时反归一化 bx/by 用）
            "xy_mean":    xy_mean.cpu(),
            "xy_std":     xy_std.cpu(),
            # 训练配置（供审计）
            "policy":     args.policy,
            "dataset":    args.dataset,
            "img_size":   args.img_size,
            "epochs":     args.epochs,
            # 训练指标
            "metrics":    metrics,
        },
        save_path,
    )
    print(f"\nProbe 已保存至 {save_path}")
    print(f"  Block 位置 RMSE = {metrics['final_xy_rmse_px']:.2f} px")
    print(f"  Block 角度 MAE  = {metrics['final_angle_mae_deg']:.2f}°")
    print("\n说明：")
    print("  PushT eval_state() 成功判定标准：位置误差 < 20px，角度误差 < 20°")
    print("  若 probe 误差接近或超过该阈值，建议增大 --epochs 或 --hidden_dim")


if __name__ == "__main__":
    main()
