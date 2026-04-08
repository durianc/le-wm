"""
eval_pusht_counterfactual_imagined.py — Push-T 想象反事实评测

验证 LeWM 的因果推理能力：模型在动作干预下的「想象」是否符合物理规律？

完整流程：

  1. Replay（分叉前）
     从数据集读取 state，用 PushT._set_state() 在真实物理引擎中逐步回放动作，
     直到 Agent-Block 首次接触（T_contact）。确保分叉起点与数据集完全一致。

  2. Encode（分叉点）
     将 T_contact 处的真实渲染帧输入 LeWM.encode()，获得初始隐状态 z_0。

  3. Rollout × 2（分叉后，纯想象）
     Factual 路：输入数据集原始动作 → LeWM.rollout → 隐状态序列 ẑ_fact
     CF 路：输入全零动作 → LeWM.rollout → 隐状态序列 ẑ_cf

  4. Decode（隐状态 → 物理坐标）
     用 BlockPoseProbe 将末步 ẑ_{T+H} 解码为 [bx, by, b_angle]。
     注意：agent 位置无法从 embedding 解码，用 T_contact 时刻的真实
     agent 位置填充，与 eval_state() 比较时位置分量来自 goal_state[:2]（agent 目标）
     和 block 预测，两者独立判定。

  5. Imagined Prediction Accuracy
     将解码的 block pose 拼成 cur_state（7维），与
     target_state = episode.states[t_contact + H] 对比，
     判定模型是否准确预测了 H 步后的真实位姿（短程动力学精度）。

判定逻辑（短程动力学预测精度）：
  pos_diff  = ||target_state[2:4] - pred[2:4]||  < 20 px
              仅比较 block 位置 [bx, by]（probe 不预测 agent）
  angle_diff = |target_state[4] - pred[4]|         < π/9 (20°)

幻觉检验：若 CF（零动作）路的想象成功率 > 0，说明模型在「Agent 静止」条件下
仍预测 Block 能到达目标 → 轨迹记忆幻觉。

Usage:
    python eval_pusht_counterfactual_imagined.py \\
        --policy pusht/lewm \\
        --probe_ckpt logs_eval/probes/pusht_block_pose_probe.pt

    python eval_pusht_counterfactual_imagined.py \\
        --policy pusht/lewm \\
        --probe_ckpt logs_eval/probes/pusht_block_pose_probe.pt \\
        --horizon 10 --contact_thresh 80 --n_episodes 100
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2 as T
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import stable_worldmodel as swm


# ── 常量 ───────────────────────────────────────────────────────────────────────

# state 格式：[ax, ay, bx, by, b_angle, vx, vy]
_AX, _AY, _BX, _BY, _BA, _VX, _VY = 0, 1, 2, 3, 4, 5, 6

PROBE_OUT_DIM = 4   # [bx, by, sin_θ, cos_θ]


# ── 图像预处理 ─────────────────────────────────────────────────────────────────

def _make_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        T.Resize(size=img_size),
    ])


def _frame_to_tensor(frame: np.ndarray, transform: T.Compose,
                     device: torch.device) -> torch.Tensor:
    """(H,W,C) uint8 → (1,1,C,H,W) float32 on device."""
    return transform(frame).unsqueeze(0).unsqueeze(0).to(device)


def _get_cache_dir() -> Path:
    return Path(swm.data.utils.get_cache_dir())


# ── 数据集加载 ─────────────────────────────────────────────────────────────────

class Episode:
    __slots__ = ("frames", "actions", "states", "goal_state")

    def __init__(self, frames: np.ndarray, actions: np.ndarray,
                 states: np.ndarray, goal_state: np.ndarray):
        # frames:     (T+1, H, W, C) uint8
        # actions:    (T,   2)       float32
        # states:     (T+1, 7)       float64
        # goal_state: (7,)           float64  — episode 末帧 state
        assert len(frames) == len(actions) + 1
        assert len(states) == len(frames)
        self.frames     = frames
        self.actions    = actions
        self.states     = states
        self.goal_state = goal_state

    def __len__(self) -> int:
        return len(self.actions)


def _iter_episodes_h5(path: str, n: int,
                      rng: np.random.Generator) -> Iterator[Episode]:
    import hdf5plugin  # noqa: F401
    with h5py.File(path, "r") as f:
        if "state" not in f:
            raise RuntimeError("HDF5 缺少 'state' 列，无法进行反事实评测。")

        offsets = f["ep_offset"][:]
        lens    = f["ep_len"][:]
        idx     = rng.choice(len(offsets), size=min(n, len(offsets)), replace=False)

        for i in idx:
            off = int(offsets[i])
            l   = int(lens[i])
            if l < 3:
                continue
            frames  = f["pixels"][off : off + l]
            actions = f["action"][off : off + l - 1]
            states  = f["state"][off : off + l]
            yield Episode(frames=frames, actions=actions,
                          states=states, goal_state=states[-1].copy())


def load_episodes(dataset_name: str, n: int,
                  rng: np.random.Generator) -> list[Episode]:
    path = _get_cache_dir() / f"{dataset_name}.h5"
    if not path.exists():
        raise FileNotFoundError(f"数据集不存在: {path}")
    eps = list(_iter_episodes_h5(str(path), n, rng))
    print(f"  加载 {len(eps)} 条 episodes（{path}）")
    return eps


# ── 模型加载 ───────────────────────────────────────────────────────────────────

def load_lewm(policy: str, device: torch.device) -> nn.Module:
    """加载 LeWM JEPA，与 eval.py load_lewm_model() 完全一致。"""
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP
    from stable_pretraining.backbone.utils import vit_hf

    p = Path(policy)
    weights_pt: Path | None = None

    if p.suffix == ".pt" and p.exists():
        weights_pt = p
    else:
        cache_dir = _get_cache_dir()
        for cand in (p, cache_dir / policy):
            w = cand / "weights.pt"
            if w.exists():
                weights_pt = w
                break
        if weights_pt is None:
            direct = cache_dir / policy
            if direct.suffix == ".pt" and direct.exists():
                weights_pt = direct

    if weights_pt is None:
        raise RuntimeError(f"找不到模型权重 '{policy}'。")

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
    proj_norm = (nn.BatchNorm1d if "projector.net.1.running_mean" in ckpt
                 else nn.LayerNorm)

    encoder        = vit_hf(size="tiny", patch_size=14, image_size=224,
                            pretrained=False, use_mask_token=False)
    predictor      = ARPredictor(
        num_frames=predictor_frames,
        input_dim=embed_dim, hidden_dim=embed_dim, output_dim=embed_dim,
        depth=predictor_depth, heads=16, mlp_dim=2048, dim_head=64,
        dropout=0.1, emb_dropout=0.0,
    )
    action_encoder = Embedder(input_dim=act_input_dim, emb_dim=embed_dim)
    projector      = MLP(input_dim=embed_dim, hidden_dim=hidden_dim,
                         output_dim=embed_dim, norm_fn=proj_norm)
    pred_proj      = MLP(input_dim=embed_dim, hidden_dim=hidden_dim,
                         output_dim=embed_dim, norm_fn=proj_norm)

    model = JEPA(encoder, predictor, action_encoder, projector, pred_proj)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        raise RuntimeError(f"权重缺失: {missing}")
    if unexpected:
        print(f"  [warn] 忽略多余键: {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")

    print(f"  LeWM: embed_dim={embed_dim}, predictor_depth={predictor_depth}, "
          f"act_input_dim={act_input_dim}")
    return model.to(device).eval().requires_grad_(False)


# ── Probe 加载 ─────────────────────────────────────────────────────────────────

class _BlockPoseProbe(nn.Module):
    """推理专用的轻量 probe，只需 forward。"""
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, PROBE_OUT_DIM),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        if emb.ndim == 3:
            emb = emb[:, -1, :]
        return self.net(emb)


def load_probe(ckpt_path: str, device: torch.device):
    """加载 BlockPoseProbe 权重及归一化统计量。

    Returns:
        probe:   _BlockPoseProbe（eval 模式，已冻结）
        xy_mean: (2,) tensor — bx/by 训练集均值
        xy_std:  (2,) tensor — bx/by 训练集标准差
    """
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Probe 权重不存在: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=True)
    probe = _BlockPoseProbe(
        in_dim=ckpt["in_dim"],
        hidden_dim=ckpt.get("hidden_dim", 256),
    )
    probe.load_state_dict(ckpt["state_dict"])
    probe = probe.to(device).eval().requires_grad_(False)

    xy_mean = ckpt["xy_mean"].to(device)   # (2,)
    xy_std  = ckpt["xy_std"].to(device)    # (2,)

    print(f"  Probe: in_dim={ckpt['in_dim']}, hidden_dim={ckpt.get('hidden_dim',256)}")
    print(f"  训练时 RMSE={ckpt['metrics']['final_xy_rmse_px']:.2f}px  "
          f"角度 MAE={ckpt['metrics']['final_angle_mae_deg']:.2f}°")
    return probe, xy_mean, xy_std


# ── 关键帧检测 ─────────────────────────────────────────────────────────────────

def find_contact_frame(states: np.ndarray, thresh: float) -> int | None:
    """state[:,0:2]=agent_pos, state[:,2:4]=block_pos。返回首次距离<thresh的帧索引。"""
    dists = np.linalg.norm(states[:, :2] - states[:, 2:4], axis=1)
    hits  = np.where(dists < thresh)[0]
    return int(hits[0]) if len(hits) > 0 else None


# ── PushT 环境工具 ─────────────────────────────────────────────────────────────

def _make_env(img_size: int = 224):
    """裸 PushT 环境，不套 swm.World wrapper。"""
    return gym.make("swm/PushT-v1", render_mode="rgb_array",
                    resolution=img_size)


def _replay_to_contact(env, episode: Episode,
                       t_contact: int) -> tuple[np.ndarray, bool]:
    """在真实物理引擎中回放动作到 t_contact，返回 (渲染帧, 是否成功初始化)。

    使用 _set_state 精确还原每一步物理状态（位置+速度），避免物理积累误差
    导致分叉起点与数据集不一致。

    Returns:
        frame:   (H, W, C) uint8 — t_contact 处的渲染帧
        ok:      True 表示成功
    """
    raw = env.unwrapped

    # 用 episode 第一帧状态初始化
    env.reset()
    raw._set_state(episode.states[0])
    raw._set_goal_state(episode.goal_state)

    # 逐步回放，同时用真实 state 纠正物理偏差
    # 不直接 step(action)，而是每步结束后强制设回数据集 state，
    # 确保 t_contact 时刻的物理状态与数据集完全一致。
    for t in range(t_contact):
        if t < len(episode.actions):
            env.step(episode.actions[t])
        # 每步结束后强制对齐（消除 PD 控制器的积累误差）
        raw._set_state(episode.states[t + 1])

    frame = env.render()   # (H, W, C) uint8
    return frame, True


# ── 想象 Rollout（纯隐空间，不驱动物理引擎）────────────────────────────────────

@torch.no_grad()
def imagined_rollout(
    model: nn.Module,
    init_frame: np.ndarray,
    actions: np.ndarray,           # (H_native, 2) 原始动作序列（已对齐 frameskip）
    transform: T.Compose,
    device: torch.device,
    horizon: int,
    history_size: int,
) -> torch.Tensor:
    """从单帧出发，在隐空间做 horizon 步开环自回归 rollout。

    Args:
        init_frame: (H,W,C) uint8 — 分叉起点的真实帧
        actions:    (horizon*frameskip, 2) — 要喂给模型的动作序列
        horizon:    模型步数（每步跨 frameskip 个 native 步）

    Returns:
        emb_seq: (1, horizon+1, D) — 包含初始 z_0 及后续 horizon 步预测
    """
    act_input_dim = model.action_encoder.patch_embed.weight.shape[1]
    raw_act_dim   = actions.shape[-1]
    frameskip     = act_input_dim // raw_act_dim

    # 编码初始帧
    obs = _frame_to_tensor(init_frame, transform, device)
    info = {"pixels": obs}
    model.encode(info)
    emb_ctx = info["emb"][:, -1:, :]       # (1, 1, D)

    act_hist: list[torch.Tensor] = []

    for step in range(horizon):
        t_start = step * frameskip
        t_end   = t_start + frameskip

        if t_end <= len(actions):
            a_block = actions[t_start:t_end].flatten()
        else:
            # 动作序列不足时补零（CF 路零动作时不会触发）
            a_block = np.zeros(act_input_dim, dtype=np.float32)

        a_t = torch.tensor(a_block, dtype=torch.float32, device=device)
        a_t = a_t.unsqueeze(0).unsqueeze(0)                # (1, 1, act_input_dim)
        act_emb = model.action_encoder(a_t)                # (1, 1, A_emb)
        act_hist.append(act_emb)

        recent_acts = act_hist[-history_size:]
        act_ctx     = torch.cat(recent_acts, dim=1)        # (1, HS, A_emb)
        ctx_len     = act_ctx.shape[1]
        emb_trunc   = emb_ctx[:, -ctx_len:, :]

        pred_emb = model.predict(emb_trunc, act_ctx)[:, -1:, :]  # (1, 1, D)
        emb_ctx  = torch.cat([emb_ctx, pred_emb], dim=1)

    return emb_ctx   # (1, horizon+1, D)


def _collect_real_frames_for_reencode(
    env,
    start_state: np.ndarray,
    goal_state: np.ndarray,
    actions: np.ndarray,
    frameskip: int,
    horizon: int,
    aligned_states: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Collect real rendered frames at each model step boundary.

    Returns:
        frames: length horizon+1, where frames[t] is the real frame at model step t.
    """
    raw = env.unwrapped
    env.reset()
    raw._set_state(start_state)
    raw._set_goal_state(goal_state)

    frames: list[np.ndarray] = [env.render()]
    native_idx = 0

    for _ in range(horizon):
        for _ in range(frameskip):
            if native_idx < len(actions):
                env.step(actions[native_idx])
            if aligned_states is not None and native_idx + 1 < len(aligned_states):
                raw._set_state(aligned_states[native_idx + 1])
            native_idx += 1
        frames.append(env.render())

    return frames


@torch.no_grad()
def closed_loop_reencode_rollout(
    model: nn.Module,
    real_frames: list[np.ndarray],
    actions: np.ndarray,
    transform: T.Compose,
    device: torch.device,
    horizon: int,
    history_size: int,
) -> torch.Tensor:
    """Closed-loop rollout with per-step real-frame re-encoding.

    At each model step t:
      1) Encode real frame x_t -> z_t
      2) Predict z_{t+1} with action block a_t
    The final returned embedding is the predicted z_{t+horizon}.
    """
    act_input_dim = model.action_encoder.patch_embed.weight.shape[1]
    raw_act_dim = actions.shape[-1]
    frameskip = act_input_dim // raw_act_dim

    emb_hist: list[torch.Tensor] = []
    act_hist: list[torch.Tensor] = []
    last_pred: torch.Tensor | None = None

    if len(real_frames) < horizon + 1:
        raise ValueError("real_frames 长度不足，无法执行闭环重编码 rollout")

    for step in range(horizon):
        # 1) Re-encode current real frame.
        obs = _frame_to_tensor(real_frames[step], transform, device)
        info = {"pixels": obs}
        model.encode(info)
        emb_t = info["emb"][:, -1:, :]
        emb_hist.append(emb_t)

        # 2) Prepare action block for this model step.
        t_start = step * frameskip
        t_end = t_start + frameskip
        if t_end <= len(actions):
            a_block = actions[t_start:t_end].flatten()
        else:
            a_block = np.zeros(act_input_dim, dtype=np.float32)

        a_t = torch.tensor(a_block, dtype=torch.float32, device=device)
        a_t = a_t.unsqueeze(0).unsqueeze(0)
        act_emb = model.action_encoder(a_t)
        act_hist.append(act_emb)

        recent_acts = act_hist[-history_size:]
        ctx_len = len(recent_acts)
        act_ctx = torch.cat(recent_acts, dim=1)
        emb_ctx = torch.cat(emb_hist[-ctx_len:], dim=1)

        last_pred = model.predict(emb_ctx, act_ctx)[:, -1:, :]

    if last_pred is None:
        raise RuntimeError("closed_loop_reencode_rollout 未产生任何预测")
    return last_pred


# ── Probe 解码 → eval_state ───────────────────────────────────────────────────

def decode_and_judge(
    pred_emb: torch.Tensor,        # (1, 1, D) 末步预测 embedding
    probe: _BlockPoseProbe,
    xy_mean: torch.Tensor,
    xy_std: torch.Tensor,
    agent_state_at_contact: np.ndarray,   # (7,) 分叉点真实 state（取 agent 位置）
    target_state: np.ndarray,             # (7,) T+H 时刻的真实 state（局部目标）
) -> dict:
    """将末步 embedding 解码为 block pose，与 T+H 时刻真实位姿对比，判定预测精度。

    对比基准为 target_state = episode.states[t_contact_native + H * frameskip]，
    即分叉后 H 步的数据集真实状态，而非 episode 的最终任务目标。

    判定逻辑（短程动力学预测精度）：
      pos_diff   = ||target[2:4] - pred[2:4]||  < 20 px
                   仅比较 block 位置 [bx, by]（probe 不预测 agent）
      angle_diff = |target[4] - pred[4]|         < π/9 (20°)
      accurate   = pos_diff < 20.0 AND angle_diff < π/9

    Returns:
        dict with keys: accurate (bool), block_pose_pred (3,),
                        pos_diff (float), angle_diff_deg (float), state_dist (float)
    """
    emb_2d = pred_emb.squeeze(0)   # (1, D) 或 (D,)
    if emb_2d.ndim == 1:
        emb_2d = emb_2d.unsqueeze(0)

    with torch.no_grad():
        out = probe(emb_2d)        # (1, 4): [bx_norm, by_norm, sin_θ, cos_θ]

    bx_norm = out[0, 0]
    by_norm = out[0, 1]
    sin_t   = out[0, 2]
    cos_t   = out[0, 3]

    bx    = (bx_norm * xy_std[0] + xy_mean[0]).item()
    by    = (by_norm * xy_std[1] + xy_mean[1]).item()
    angle = float(torch.atan2(sin_t, cos_t).item())   # (-π, π]

    # 局部目标：T+H 时刻的真实 block 位姿
    target = target_state.astype(np.float64)
    pos_diff   = np.linalg.norm(target[2:4] - np.array([bx, by]))
    angle_diff = abs(target[_BA] - angle)
    angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
    accurate   = bool(pos_diff < 20.0 and angle_diff < np.pi / 9)

    # 完整 state 距离（agent 位置用分叉点真实值填充，速度设为 0）
    pred_state = np.array([
        agent_state_at_contact[_AX],
        agent_state_at_contact[_AY],
        bx, by, angle,
        0.0, 0.0,
    ], dtype=np.float64)
    state_dist = float(np.linalg.norm(target - pred_state))

    return {
        "accurate":         accurate,
        "block_pose_pred":  np.array([bx, by, angle], dtype=np.float32),
        "pos_diff":         float(pos_diff),
        "angle_diff_deg":   float(np.degrees(angle_diff)),
        "state_dist":       state_dist,
    }


# ── 单 episode 评测 ────────────────────────────────────────────────────────────

def evaluate_episode(
    episode: Episode,
    env,
    model: nn.Module,
    probe: _BlockPoseProbe,
    xy_mean: torch.Tensor,
    xy_std: torch.Tensor,
    transform: T.Compose,
    device: torch.device,
    contact_thresh: float,
    horizon: int,
    history_size: int,
    closed_loop_reencode: bool,
) -> dict | None:
    """对单条 episode 做完整的想象反事实评测。

    Returns:
        包含 factual/cf 各项结果的 dict，或 None（无有效接触帧 / 长度不足）。
    """
    # ── Step 1: 关键帧检测 ───────────────────────────────────────────────────
    t_contact = find_contact_frame(episode.states, contact_thresh)
    if t_contact is None:
        return None

    # 推断 frameskip，计算可用模型步数
    act_input_dim = model.action_encoder.patch_embed.weight.shape[1]
    raw_act_dim   = episode.actions.shape[-1]
    frameskip     = act_input_dim // raw_act_dim

    # 分叉点对齐到模型步边界
    t_contact_model = max(t_contact // frameskip, 1)
    t_contact_native = t_contact_model * frameskip

    available_model_steps = (len(episode) - t_contact_native) // frameskip
    H = min(horizon, available_model_steps)
    if H < 1:
        return None

    # ── Step 2: 真实物理引擎 replay 到 t_contact ────────────────────────────
    init_frame, ok = _replay_to_contact(env, episode, t_contact_native)
    if not ok:
        return None

    agent_state_at_contact = episode.states[t_contact_native]  # (7,)

    # ── Step 3: 构造 factual / CF 动作序列（native 步，长度 H*frameskip）────
    native_start = t_contact_native
    native_end   = native_start + H * frameskip

    fact_actions = episode.actions[native_start:native_end]            # (H*fs, 2)
    cf_actions   = np.zeros_like(fact_actions)                         # 全零

    # 局部目标：T+H 时刻的真实 state（短程动力学的 ground truth）
    target_state = episode.states[native_end]                          # (7,)

    # ── 维度 1：真实物理位移量（验证"CF 捡漏"假设）────────────────────────
    # Block 在这 H 步里实际移动了多少（从分叉点到 T+H 时刻）
    block_pos_start  = agent_state_at_contact[_BX : _BY + 1]          # (2,)
    block_pos_target = target_state[_BX : _BY + 1]                    # (2,)
    gt_block_disp_vec = block_pos_target - block_pos_start             # (2,) 真实位移向量
    gt_block_disp_px  = float(np.linalg.norm(gt_block_disp_vec))      # 真实位移幅度

    # ── Step 4: 双路想象 rollout ─────────────────────────────────────────────
    emb_fact = imagined_rollout(
        model, init_frame, fact_actions, transform, device, H, history_size
    )   # (1, H+1, D)
    emb_cf = imagined_rollout(
        model, init_frame, cf_actions,   transform, device, H, history_size
    )   # (1, H+1, D)

    # 取末步预测 embedding
    last_fact = emb_fact[:, -1:, :]   # (1, 1, D)
    last_cf   = emb_cf[:,   -1:, :]   # (1, 1, D)

    # ── Step 5: 解码 + 预测精度判定（基准为 T+H 真实位姿）──────────────────
    fact_result = decode_and_judge(
        last_fact, probe, xy_mean, xy_std,
        agent_state_at_contact, target_state,
    )
    cf_result = decode_and_judge(
        last_cf, probe, xy_mean, xy_std,
        agent_state_at_contact, target_state,
    )

    cl_fact_result = None
    cl_cf_result = None
    if closed_loop_reencode:
        factual_aligned_states = episode.states[native_start:native_end + 1]
        fact_real_frames = _collect_real_frames_for_reencode(
            env=env,
            start_state=agent_state_at_contact,
            goal_state=episode.goal_state,
            actions=fact_actions,
            frameskip=frameskip,
            horizon=H,
            aligned_states=factual_aligned_states,
        )
        cf_real_frames = _collect_real_frames_for_reencode(
            env=env,
            start_state=agent_state_at_contact,
            goal_state=episode.goal_state,
            actions=cf_actions,
            frameskip=frameskip,
            horizon=H,
            aligned_states=None,
        )

        cl_last_fact = closed_loop_reencode_rollout(
            model=model,
            real_frames=fact_real_frames,
            actions=fact_actions,
            transform=transform,
            device=device,
            horizon=H,
            history_size=history_size,
        )
        cl_last_cf = closed_loop_reencode_rollout(
            model=model,
            real_frames=cf_real_frames,
            actions=cf_actions,
            transform=transform,
            device=device,
            horizon=H,
            history_size=history_size,
        )

        cl_fact_result = decode_and_judge(
            cl_last_fact, probe, xy_mean, xy_std,
            agent_state_at_contact, target_state,
        )
        cl_cf_result = decode_and_judge(
            cl_last_cf, probe, xy_mean, xy_std,
            agent_state_at_contact, target_state,
        )

    # 隐空间分叉量（两路末步 embedding 的距离）
    divergence = float(
        torch.mean((last_fact - last_cf) ** 2).item()
    )

    # ── 维度 2：预测位移的幅度与方向误差 ────────────────────────────────────
    # 预测位移向量 = 预测 block 位置 - 分叉点 block 位置
    fact_pred_xy = fact_result["block_pose_pred"][:2]                  # (2,) [bx, by]
    cf_pred_xy   = cf_result["block_pose_pred"][:2]                    # (2,)
    fact_disp_vec = fact_pred_xy - block_pos_start                     # (2,)
    cf_disp_vec   = cf_pred_xy   - block_pos_start                     # (2,)

    fact_disp_px = float(np.linalg.norm(fact_disp_vec))
    cf_disp_px   = float(np.linalg.norm(cf_disp_vec))

    # 方向角误差（预测位移方向 vs 真实位移方向），GT 无位移时设为 nan
    def _direction_error_deg(pred_vec, gt_vec):
        gt_norm = np.linalg.norm(gt_vec)
        if gt_norm < 1e-3:
            return float("nan")
        pred_norm = np.linalg.norm(pred_vec)
        if pred_norm < 1e-3:
            return 180.0   # 预测无位移但 GT 有位移，方向完全错误
        cos_a = np.clip(np.dot(pred_vec, gt_vec) / (pred_norm * gt_norm), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_a)))

    fact_dir_err_deg = _direction_error_deg(fact_disp_vec, gt_block_disp_vec)
    cf_dir_err_deg   = _direction_error_deg(cf_disp_vec,   gt_block_disp_vec)

    # ── 新指标 1：相对误差 (Relative Error) ──────────────────────────────────
    # ||pred_pos - gt_target_pos|| / (||start_pos - gt_target_pos|| + ε)
    # 值为 1.0 表示模型等同于原地不动，<1 表示有效预测
    _eps = 1e-6
    _start_to_gt = float(np.linalg.norm(block_pos_start - block_pos_target))
    fact_relative_error = fact_result["pos_diff"] / (_start_to_gt + _eps)
    cf_relative_error   = cf_result["pos_diff"]   / (_start_to_gt + _eps)

    # ── 新指标 2：因果干预收益 (Causal Gain) ─────────────────────────────────
    # Factual 预测是否比 CF 预测更接近真值 → 证明模型提取到了动作信号
    fact_pos_diff = fact_result["pos_diff"]
    cf_pos_diff   = cf_result["pos_diff"]
    is_causal_success = bool(fact_pos_diff < cf_pos_diff)

    # ── 新指标 3：向量分解评估 (Vector Decomposition) ────────────────────────
    # 3a. 方向余弦相似度：预测位移向量 vs 真实位移向量
    def _cosine_similarity(pred_vec, gt_vec):
        gt_norm   = np.linalg.norm(gt_vec)
        pred_norm = np.linalg.norm(pred_vec)
        if gt_norm < 1e-3 or pred_norm < 1e-3:
            return float("nan")
        return float(np.dot(pred_vec, gt_vec) / (pred_norm * gt_norm))

    fact_dir_cosine = _cosine_similarity(fact_disp_vec, gt_block_disp_vec)
    cf_dir_cosine   = _cosine_similarity(cf_disp_vec,   gt_block_disp_vec)

    # 3b. 幅度缩放比：||pred_displacement|| / ||gt_displacement||
    def _magnitude_ratio(pred_vec, gt_vec):
        gt_norm = np.linalg.norm(gt_vec)
        if gt_norm < 1e-3:
            return float("nan")
        return float(np.linalg.norm(pred_vec) / gt_norm)

    fact_mag_ratio = _magnitude_ratio(fact_disp_vec, gt_block_disp_vec)
    cf_mag_ratio   = _magnitude_ratio(cf_disp_vec,   gt_block_disp_vec)

    out = {
        "t_contact_native":    t_contact_native,
        "t_contact_model":     t_contact_model,
        "H":                   H,
        "target_state":        target_state.tolist(),      # T+H 真实位姿
        "block_pos_start":     block_pos_start.tolist(),   # 分叉点 block 位置（可视化用）
        "init_frame":          init_frame,                 # 分叉点渲染帧（可视化用，非 JSON）

        # ── 维度 1：GT 位移 ───────────────────────────────────────────────────
        "gt_block_disp_px":    gt_block_disp_px,
        "gt_block_disp_vec":   gt_block_disp_vec.tolist(),

        # ── Factual ───────────────────────────────────────────────────────────
        "fact_accurate":       fact_result["accurate"],
        "fact_pos_diff":       fact_result["pos_diff"],
        "fact_angle_deg":      fact_result["angle_diff_deg"],
        "fact_block_pose":     fact_result["block_pose_pred"].tolist(),
        # 维度 2：Factual 预测位移
        "fact_disp_px":        fact_disp_px,
        "fact_disp_vec":       fact_disp_vec.tolist(),
        "fact_dir_err_deg":    fact_dir_err_deg,
        # 新指标 1：相对误差
        "fact_relative_error": fact_relative_error,
        # 新指标 2：因果干预收益
        "is_causal_success":   is_causal_success,
        # 新指标 3：向量分解
        "fact_dir_cosine":     fact_dir_cosine,
        "fact_mag_ratio":      fact_mag_ratio,

        # ── Counterfactual ────────────────────────────────────────────────────
        "cf_accurate":         cf_result["accurate"],
        "cf_pos_diff":         cf_result["pos_diff"],
        "cf_angle_deg":        cf_result["angle_diff_deg"],
        "cf_block_pose":       cf_result["block_pose_pred"].tolist(),
        # 维度 2：CF 预测位移
        "cf_disp_px":          cf_disp_px,
        "cf_disp_vec":         cf_disp_vec.tolist(),
        "cf_dir_err_deg":      cf_dir_err_deg,
        # 新指标 1：相对误差（CF）
        "cf_relative_error":   cf_relative_error,
        # 新指标 3：向量分解（CF）
        "cf_dir_cosine":       cf_dir_cosine,
        "cf_mag_ratio":        cf_mag_ratio,

        # ── Latent divergence ─────────────────────────────────────────────────
        "latent_divergence":   divergence,
    }

    if cl_fact_result is not None and cl_cf_result is not None:
        out.update(
            {
                # Closed-loop re-encode path
                "fact_cl_accurate":       cl_fact_result["accurate"],
                "fact_cl_pos_diff":       cl_fact_result["pos_diff"],
                "fact_cl_angle_deg":      cl_fact_result["angle_diff_deg"],
                "fact_cl_block_pose":     cl_fact_result["block_pose_pred"].tolist(),
                "cf_cl_accurate":         cl_cf_result["accurate"],
                "cf_cl_pos_diff":         cl_cf_result["pos_diff"],
                "cf_cl_angle_deg":        cl_cf_result["angle_diff_deg"],
                "cf_cl_block_pose":       cl_cf_result["block_pose_pred"].tolist(),
                "is_cl_causal_success":   bool(cl_fact_result["pos_diff"] < cl_cf_result["pos_diff"]),
            }
        )

    return out


# ── 聚合评测 ───────────────────────────────────────────────────────────────────

def run_imagined_eval(
    episodes: list[Episode],
    model: nn.Module,
    probe: _BlockPoseProbe,
    xy_mean: torch.Tensor,
    xy_std: torch.Tensor,
    contact_thresh: float,
    horizon: int,
    history_size: int,
    closed_loop_reencode: bool,
    img_size: int,
    device: torch.device,
) -> dict:
    transform = _make_transform(img_size)
    env = _make_env(img_size)

    results: list[dict] = []
    skipped = 0

    try:
        for i, ep in enumerate(tqdm(episodes, desc="评测 episodes")):
            res = evaluate_episode(
                ep, env, model, probe, xy_mean, xy_std,
                transform, device, contact_thresh, horizon, history_size,
                closed_loop_reencode,
            )
            if res is None:
                skipped += 1
            else:
                results.append(res)

            if (i + 1) % 20 == 0 and results:
                n = len(results)
                acc_f = sum(r["fact_accurate"] for r in results) / n * 100
                acc_c = sum(r["cf_accurate"]   for r in results) / n * 100
                msg = (f"  [{i+1}/{len(episodes)}] 有效={n} 跳过={skipped} "
                       f"OpenLoop: Fact={acc_f:.1f}% CF={acc_c:.1f}%")
                if closed_loop_reencode and "fact_cl_accurate" in results[0]:
                    acc_f_cl = sum(r["fact_cl_accurate"] for r in results) / n * 100
                    acc_c_cl = sum(r["cf_cl_accurate"] for r in results) / n * 100
                    msg += f" | ClosedLoop: Fact={acc_f_cl:.1f}% CF={acc_c_cl:.1f}%"
                print(msg)
    finally:
        env.close()

    if not results:
        return {"error": "无有效接触帧，请调大 --contact_thresh"}

    n = len(results)
    fact_acc = sum(r["fact_accurate"] for r in results) / n * 100
    cf_acc   = sum(r["cf_accurate"]   for r in results) / n * 100

    # 聚合各项物理误差指标
    contact_steps   = [r["t_contact_native"] for r in results]
    divergences     = [r["latent_divergence"] for r in results]
    fact_pos_diffs  = [r["fact_pos_diff"]     for r in results]
    cf_pos_diffs    = [r["cf_pos_diff"]       for r in results]
    fact_angle_degs = [r["fact_angle_deg"]    for r in results]
    cf_angle_degs   = [r["cf_angle_deg"]      for r in results]

    # 维度 1：GT 位移分布
    gt_disps       = [r["gt_block_disp_px"]  for r in results]

    # 维度 2：预测位移幅度与方向误差（过滤掉 GT 无位移导致的 nan）
    fact_disp_pxs  = [r["fact_disp_px"]      for r in results]
    cf_disp_pxs    = [r["cf_disp_px"]        for r in results]
    fact_dir_errs  = [r["fact_dir_err_deg"]  for r in results
                      if not np.isnan(r["fact_dir_err_deg"])]
    cf_dir_errs    = [r["cf_dir_err_deg"]    for r in results
                      if not np.isnan(r["cf_dir_err_deg"])]

    # 新指标 1：相对误差
    fact_rel_errs = [r["fact_relative_error"] for r in results]
    cf_rel_errs   = [r["cf_relative_error"]   for r in results]

    # 新指标 2：因果干预收益（Fact 比 CF 更接近真值的比例）
    causal_gain_rate = float(sum(r["is_causal_success"] for r in results) / n * 100)

    # 新指标 3：向量分解（过滤 nan）
    fact_dir_cosines = [r["fact_dir_cosine"] for r in results
                        if not np.isnan(r["fact_dir_cosine"])]
    cf_dir_cosines   = [r["cf_dir_cosine"]   for r in results
                        if not np.isnan(r["cf_dir_cosine"])]
    fact_mag_ratios  = [r["fact_mag_ratio"]  for r in results
                        if not np.isnan(r["fact_mag_ratio"])]
    cf_mag_ratios    = [r["cf_mag_ratio"]    for r in results
                        if not np.isnan(r["cf_mag_ratio"])]

    # 幻觉检验：CF 路（零动作）预测精度异常高 → 模型忽视了动作的作用
    cf_accurate_episodes = [r for r in results if r["cf_accurate"]]

    closed_loop_metrics: dict[str, float | int] = {}
    if closed_loop_reencode and "fact_cl_accurate" in results[0]:
        fact_cl_acc = sum(r["fact_cl_accurate"] for r in results) / n * 100
        cf_cl_acc = sum(r["cf_cl_accurate"] for r in results) / n * 100
        fact_cl_pos = [r["fact_cl_pos_diff"] for r in results]
        cf_cl_pos = [r["cf_cl_pos_diff"] for r in results]
        fact_cl_ang = [r["fact_cl_angle_deg"] for r in results]
        cf_cl_ang = [r["cf_cl_angle_deg"] for r in results]
        cl_gain = sum(r["is_cl_causal_success"] for r in results) / n * 100

        closed_loop_metrics = {
            "closed_loop_reencode_enabled": True,
            "factual_cl_prediction_accuracy": round(fact_cl_acc, 2),
            "cf_cl_prediction_accuracy": round(cf_cl_acc, 2),
            "prediction_accuracy_delta_cl": round(fact_cl_acc - cf_cl_acc, 2),
            "mean_fact_cl_pos_diff_px": float(np.mean(fact_cl_pos)),
            "mean_cf_cl_pos_diff_px": float(np.mean(cf_cl_pos)),
            "mean_fact_cl_angle_deg": float(np.mean(fact_cl_ang)),
            "mean_cf_cl_angle_deg": float(np.mean(cf_cl_ang)),
            "cl_causal_gain_rate_pct": round(float(cl_gain), 2),
            "n_cl_causal_success": int(sum(r["is_cl_causal_success"] for r in results)),
        }

    out = {
        "n_valid":   n,
        "n_skipped": skipped,
        "horizon":   horizon,
        "contact_thresh": contact_thresh,

        # ── Open-loop 核心指标（短程动力学预测精度，基准为 T+H 时刻真实位姿）──
        "factual_prediction_accuracy":    round(fact_acc, 2),
        "cf_prediction_accuracy":         round(cf_acc,   2),
        "prediction_accuracy_delta":      round(fact_acc - cf_acc, 2),

        # ── Open-loop 物理误差（相对 T+H 真实位姿）──────────────────────────
        "mean_fact_pos_diff_px":   float(np.mean(fact_pos_diffs)),
        "mean_cf_pos_diff_px":     float(np.mean(cf_pos_diffs)),
        "mean_fact_angle_deg":     float(np.mean(fact_angle_degs)),
        "mean_cf_angle_deg":       float(np.mean(cf_angle_degs)),

        # ── Open-loop 因果/幻觉诊断 ─────────────────────────────────────────
        "causal_gain_rate_pct":    round(causal_gain_rate, 2),
        "n_causal_success":        sum(r["is_causal_success"] for r in results),
        "cf_hallucination_accuracy": round(cf_acc, 2),
        "n_cf_accurate_episodes":    len(cf_accurate_episodes),

        # ── 诊断维度 1：GT 位移分布 ──────────────────────────────────────────
        "mean_gt_block_disp_px":   float(np.mean(gt_disps)),
        "std_gt_block_disp_px":    float(np.std(gt_disps)),
        "median_gt_block_disp_px": float(np.median(gt_disps)),

        # ── 诊断维度 2：预测位移幅度与方向误差 ───────────────────────────────
        "mean_fact_disp_px":       float(np.mean(fact_disp_pxs)),
        "mean_cf_disp_px":         float(np.mean(cf_disp_pxs)),
        # 方向角误差（仅统计 GT 有位移的 episode）
        "mean_fact_dir_err_deg":   float(np.mean(fact_dir_errs)) if fact_dir_errs else float("nan"),
        "mean_cf_dir_err_deg":     float(np.mean(cf_dir_errs))   if cf_dir_errs   else float("nan"),
        "n_dir_err_valid":         len(fact_dir_errs),

        # ── 诊断维度 3：相对误差 (Relative Error) ────────────────────────────
        # <1.0 说明比原地不动好，=1.0 等于没有预测能力
        "mean_fact_relative_error": float(np.mean(fact_rel_errs)),
        "mean_cf_relative_error":   float(np.mean(cf_rel_errs)),

        # ── 诊断维度 4：向量分解 (Vector Decomposition) ───────────────────────
        # 方向余弦相似度（-1~1，越大方向越对）
        "mean_fact_dir_cosine":    float(np.mean(fact_dir_cosines)) if fact_dir_cosines else float("nan"),
        "mean_cf_dir_cosine":      float(np.mean(cf_dir_cosines))   if cf_dir_cosines   else float("nan"),
        # 幅度缩放比（1.0 = 完美，>1 过预测，<1 欠预测）
        "mean_fact_mag_ratio":     float(np.mean(fact_mag_ratios))  if fact_mag_ratios  else float("nan"),
        "mean_cf_mag_ratio":       float(np.mean(cf_mag_ratios))    if cf_mag_ratios    else float("nan"),
        "n_vec_decomp_valid":      len(fact_dir_cosines),

        # ── 隐空间分叉量 ──────────────────────────────────────────────────────
        "mean_latent_divergence":  float(np.mean(divergences)),
        "std_latent_divergence":   float(np.std(divergences)),

        # ── 接触帧统计 ────────────────────────────────────────────────────────
        "contact_step_mean": float(np.mean(contact_steps)),
        "contact_step_std":  float(np.std(contact_steps)),

        # ── 逐 episode 明细 ───────────────────────────────────────────────────
        "per_episode": results,
    }
    out.update(closed_loop_metrics)
    return out


# ── 维度 3：典型失败案例可视化（"CF 准确但 Fact 失败"）─────────────────────────

def visualize_cf_failures(
    results: list[dict],
    out_dir: str,
    max_cases: int = 8,
    pusht_img_size: int = 224,
) -> None:
    """Visualize typical failure cases where CF is accurate but Fact fails.
    
    Overlaid scatter plot on the contact frame.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [skip] matplotlib is not installed, skipping failure case visualization")
        return

    # Filter episodes where CF is accurate but Fact fails
    cases = [r for r in results if r["cf_accurate"] and not r["fact_accurate"]]
    if not cases:
        print("  No episodes found where CF is accurate but Fact fails, skipping visualization")
        return

    cases = cases[:max_cases]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Physical coordinate (512x512) → rendered image (img_size x img_size) scale
    PHYS_SIZE = 512.0
    scale = pusht_img_size / PHYS_SIZE

    def _phys_to_px(xy):
        """Convert physical (bx, by) to image pixel (col, row)."""
        return float(xy[0] * scale), float(xy[1] * scale)

    for idx, r in enumerate(cases):
        fig, ax = plt.subplots(figsize=(5, 5))

        # Background: contact frame
        if "init_frame" in r and r["init_frame"] is not None:
            ax.imshow(r["init_frame"])
        else:
            ax.set_facecolor("#cccccc")

        ax.set_xlim(0, pusht_img_size)
        ax.set_ylim(pusht_img_size, 0)   # Image coordinate: y-axis downward

        # Convert coordinates
        start_xy  = r["block_pos_start"]
        target_xy = r["target_state"][_BX : _BY + 1]
        fact_xy   = r["fact_block_pose"][:2]
        cf_xy     = r["cf_block_pose"][:2]

        s_col, s_row = _phys_to_px(start_xy)
        t_col, t_row = _phys_to_px(target_xy)
        f_col, f_row = _phys_to_px(fact_xy)
        c_col, c_row = _phys_to_px(cf_xy)

        R = max(4.0, pusht_img_size * 0.018)

        # Start position (yellow hollow circle)
        ax.add_patch(plt.Circle((s_col, s_row), R, color="yellow",
                                fill=False, lw=2, zorder=3))
        # GT T+H position (green filled circle)
        ax.add_patch(plt.Circle((t_col, t_row), R, color="#00cc44",
                                fill=True, zorder=4))
        # Factual prediction (blue filled circle)
        ax.add_patch(plt.Circle((f_col, f_row), R, color="#2277ff",
                                fill=True, zorder=4))
        # CF prediction (red filled circle)
        ax.add_patch(plt.Circle((c_col, c_row), R, color="#ff3322",
                                fill=True, zorder=4))

        # Arrow: start → GT target
        ax.annotate("", xy=(t_col, t_row), xytext=(s_col, s_row),
                    arrowprops=dict(arrowstyle="->", color="#00cc44",
                                   lw=1.5), zorder=5)

        legend_patches = [
            mpatches.Patch(color="yellow",   label=f"Start (t={r['t_contact_native']})"),
            mpatches.Patch(color="#00cc44",  label=f"GT T+H (disp={r['gt_block_disp_px']:.1f}px)"),
            mpatches.Patch(color="#2277ff",  label=f"Fact pred (err={r['fact_pos_diff']:.1f}px)"),
            mpatches.Patch(color="#ff3322",  label=f"CF pred  (err={r['cf_pos_diff']:.1f}px)  ✓"),
        ]
        ax.legend(handles=legend_patches, fontsize=7, loc="lower right",
                  framealpha=0.85)

        ax.set_title(
            f"CF-accurate / Fact-fail  (ep idx={idx})\n"
            f"H={r['H']}  gt_disp={r['gt_block_disp_px']:.1f}px  "
            f"fact_dir_err={r['fact_dir_err_deg']:.0f}°  "
            f"cf_dir_err={r['cf_dir_err_deg']:.0f}°",
            fontsize=8,
        )
        ax.axis("off")
        fig.tight_layout()

        save_path = str(Path(out_dir) / f"cf_failure_{idx:02d}.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  [viz] Saved {save_path}")

    print(f"  Saved {len(cases)} failure case images to {out_dir}/")


# ── 可视化 ─────────────────────────────────────────────────────────────────────

def plot_results(metrics: dict, out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib is not installed")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()

    eps = metrics["per_episode"]

    # Plot 0: Prediction Accuracy Bar Chart
    ax = axes[0]
    labels = ["Factual\n(original actions)", "Counterfactual\n(zero actions)"]
    values = [metrics["factual_prediction_accuracy"],
              metrics["cf_prediction_accuracy"]]
    colors = ["#1f77b4", "#d62728"]
    bars   = ax.bar(labels, values, color=colors, width=0.4, edgecolor="gray")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(max(values) * 1.35 + 5, 10))
    ax.set_ylabel("Prediction Accuracy (%)", fontsize=11)
    ax.set_title(
        f"Prediction Accuracy (vs T+H ground truth)\n"
        f"Fact={values[0]:.1f}%  CF={values[1]:.1f}%  "
        f"Δ={metrics['prediction_accuracy_delta']:+.1f}%",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3)

    # Plot 1: Block Position Error Scatter (Fact vs CF)
    ax = axes[1]
    fact_dists = [r["fact_pos_diff"] for r in eps]
    cf_dists   = [r["cf_pos_diff"]   for r in eps]
    ax.scatter(fact_dists, cf_dists, alpha=0.5, s=20, color="#2ca02c")
    lim = max(max(fact_dists), max(cf_dists)) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="y=x")
    ax.axvline(20, color="gray", lw=0.8, linestyle=":", label="Accuracy threshold 20px")
    ax.axhline(20, color="gray", lw=0.8, linestyle=":")
    ax.set_xlabel("Fact Block → T+H truth distance (px)", fontsize=11)
    ax.set_ylabel("CF Block → T+H truth distance (px)",   fontsize=11)
    ax.set_title("Block Prediction Error: Factual vs CF\n(benchmark = T+H ground truth pose)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Plot 2: Latent Divergence Distribution
    ax = axes[2]
    divs = [r["latent_divergence"] for r in eps]
    ax.hist(divs, bins=25, color="#9467bd", edgecolor="gray", alpha=0.8)
    ax.axvline(np.mean(divs), color="red", lw=1.5, linestyle="--",
               label=f"mean={np.mean(divs):.4f}")
    ax.set_xlabel("Latent Divergence ||ẑ_fact - ẑ_cf||²", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Final-step latent divergence (H={metrics['horizon']})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Plot 3: Dimension 1 — GT Block Displacement Histogram
    ax = axes[3]
    gt_disps = [r["gt_block_disp_px"] for r in eps]
    ax.hist(gt_disps, bins=25, color="#8c564b", edgecolor="gray", alpha=0.8)
    ax.axvline(metrics["mean_gt_block_disp_px"], color="red", lw=1.5,
               linestyle="--",
               label=f"mean={metrics['mean_gt_block_disp_px']:.1f}px")
    ax.axvline(20, color="gray", lw=1.0, linestyle=":",
               label="Accuracy threshold 20px (leakage boundary)")
    ax.set_xlabel("GT Block Displacement (px)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        f"Dimension 1: Block real displacement distribution within H={metrics['horizon']} steps\n"
        f"mean={metrics['mean_gt_block_disp_px']:.1f}px  "
        f"median={metrics['median_gt_block_disp_px']:.1f}px",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Plot 4: Dimension 2 — GT vs Predicted Displacement Scatter
    ax = axes[4]
    fact_disps = [r["fact_disp_px"] for r in eps]
    cf_disps   = [r["cf_disp_px"]   for r in eps]
    ax.scatter(gt_disps, fact_disps, alpha=0.5, s=18, color="#1f77b4",
               label=f"Factual (mean={np.mean(fact_disps):.1f}px)")
    ax.scatter(gt_disps, cf_disps,   alpha=0.5, s=18, color="#d62728",
               marker="^", label=f"CF zero (mean={np.mean(cf_disps):.1f}px)")
    _lim = max(max(gt_disps), max(fact_disps), max(cf_disps)) * 1.05
    ax.plot([0, _lim], [0, _lim], "k--", lw=1, alpha=0.5, label="y=x (perfect)")
    ax.set_xlabel("GT Block Displacement (px)", fontsize=11)
    ax.set_ylabel("Predicted Displacement (px)", fontsize=11)
    ax.set_title(
        "Dimension 2: GT displacement vs predicted displacement magnitude\n"
        "(points on y=x = correct magnitude prediction)",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 5: Dimension 2 — Direction Error Histogram (Fact vs CF)
    ax = axes[5]
    fact_dir_errs = [r["fact_dir_err_deg"] for r in eps
                     if not np.isnan(r["fact_dir_err_deg"])]
    cf_dir_errs   = [r["cf_dir_err_deg"]   for r in eps
                     if not np.isnan(r["cf_dir_err_deg"])]
    bins = np.linspace(0, 180, 19)
    ax.hist(fact_dir_errs, bins=bins, color="#1f77b4", alpha=0.6, edgecolor="gray",
            label=f"Factual (mean={np.mean(fact_dir_errs):.0f}°)" if fact_dir_errs else "Factual")
    ax.hist(cf_dir_errs,   bins=bins, color="#d62728", alpha=0.6, edgecolor="gray",
            label=f"CF zero (mean={np.mean(cf_dir_errs):.0f}°)" if cf_dir_errs else "CF zero")
    ax.axvline(90, color="gray", lw=1.0, linestyle=":", label="Random direction baseline 90°")
    ax.set_xlabel("Direction Error (degrees)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        f"Dimension 2: Predicted displacement direction error (n={len(fact_dir_errs)} valid episodes)\n"
        "Error < 90° = valid direction, > 90° = opposite direction",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Push-T Imagined Counterfactual Evaluation  "
        f"(n={metrics['n_valid']}, thresh={metrics['contact_thresh']}, "
        f"H={metrics['horizon']})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plots saved to {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Push-T Imagined Counterfactual Eval — LeWM Causal Reasoning"
    )
    p.add_argument("--dataset",  default="pusht_expert_train",
                   help="数据集名称（默认 pusht_expert_train）")
    p.add_argument("--policy",   default="pusht/lewm",
                   help="LeWM 模型引用（默认 pusht/lewm）")
    p.add_argument("--probe_ckpt",
                   default="logs_eval/probes/pusht_block_pose_probe.pt",
                   help="BlockPoseProbe 权重路径")
    p.add_argument("--contact_thresh", type=float, default=80.0,
                   help="Agent-Block 接触距离阈值（物理坐标系 px，默认 80）")
    p.add_argument("--horizon",        type=int,   default=10,
                   help="分叉后 rollout 的模型步数（默认 10）")
    p.add_argument("--history_size",   type=int,   default=1,
                   help="预测器上下文窗口（默认 1）")
    p.add_argument("--closed_loop_reencode", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="是否启用闭环重编码版本（每步用真实帧重编码，默认启用）")
    p.add_argument("--n_episodes",     type=int,   default=100,
                   help="评测 episode 数量（默认 100）")
    p.add_argument("--img_size",       type=int,   default=224,
                   help="图像分辨率（默认 224）")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--output",
                   default="logs_eval/pusht_cf_imagined_results.json")
    p.add_argument("--plot",
                   default="logs_eval/pusht_cf_imagined_plot.png")
    p.add_argument("--failure_viz_dir",
                   default="logs_eval/cf_failures",
                   help="CF 准确但 Fact 失败的典型案例可视化输出目录")
    p.add_argument("--max_failure_viz", type=int, default=8,
                   help="最多输出几张失败案例图（默认 8）")
    return p.parse_args()


def main():
    args   = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng    = np.random.default_rng(args.seed)

    print(f"设备: {device}  horizon={args.horizon}  thresh={args.contact_thresh}")

    print("\n加载数据集...")
    episodes = load_episodes(args.dataset, args.n_episodes, rng)

    print("\n加载 LeWM 模型...")
    model = load_lewm(args.policy, device)

    print("\n加载 BlockPoseProbe...")
    probe, xy_mean, xy_std = load_probe(args.probe_ckpt, device)

    print(f"\n开始评测（{len(episodes)} episodes）...")
    t0 = time.time()
    metrics = run_imagined_eval(
        episodes=episodes,
        model=model,
        probe=probe,
        xy_mean=xy_mean,
        xy_std=xy_std,
        contact_thresh=args.contact_thresh,
        horizon=args.horizon,
        history_size=args.history_size,
        closed_loop_reencode=args.closed_loop_reencode,
        img_size=args.img_size,
        device=device,
    )
    elapsed = time.time() - t0

    # ── 打印摘要 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("PUSH-T IMAGINED COUNTERFACTUAL EVALUATION RESULTS")
    print("=" * 68)
    if "error" in metrics:
        print(f"  错误: {metrics['error']}")
        return

        print(f"  有效 episodes               : {metrics['n_valid']}"
          f"  （跳过 {metrics['n_skipped']}）")
        print(f"  horizon                     : {metrics['horizon']} model steps")
        print(f"  contact_thresh              : {metrics['contact_thresh']} px")
        print(f"  （判定基准：T+H 时刻真实位姿，非任务终点 goal_state）")

        print("\n[Open-loop 纯想象]")
        print()
        print(f"  Factual  预测精度           : {metrics['factual_prediction_accuracy']:.1f}%")
        print(f"  CF（零动作）预测精度        : {metrics['cf_prediction_accuracy']:.1f}%")
        print(f"  精度差值 (Fact - CF)        : {metrics['prediction_accuracy_delta']:+.1f}%")

        print(f"  Block 位置偏差 Fact         : {metrics['mean_fact_pos_diff_px']:.1f} px")
        print(f"  Block 位置偏差 CF           : {metrics['mean_cf_pos_diff_px']:.1f} px")
        print(f"  Block 角度偏差 Fact         : {metrics['mean_fact_angle_deg']:.1f}°")
        print(f"  Block 角度偏差 CF           : {metrics['mean_cf_angle_deg']:.1f}°")

        print(f"  因果干预收益（Open-loop）   : {metrics['causal_gain_rate_pct']:.1f}%"
            f"  ({metrics['n_causal_success']}/{metrics['n_valid']})")
        print(f"  CF 幻觉精度（Open-loop）    : {metrics['cf_hallucination_accuracy']:.1f}%"
            f"  （{metrics['n_cf_accurate_episodes']} 条 episodes）")

    if metrics.get("closed_loop_reencode_enabled", False):
        print()
        print("[Closed-loop 重编码] 每步真实帧 re-encode")
        print(f"  Factual-CL 预测精度         : {metrics['factual_cl_prediction_accuracy']:.1f}%")
        print(f"  CF-CL（零动作）预测精度     : {metrics['cf_cl_prediction_accuracy']:.1f}%")
        print(f"  精度差值 (Fact-CL - CF-CL)  : {metrics['prediction_accuracy_delta_cl']:+.1f}%")
        print(f"  Block 位置偏差 Fact-CL      : {metrics['mean_fact_cl_pos_diff_px']:.1f} px")
        print(f"  Block 位置偏差 CF-CL        : {metrics['mean_cf_cl_pos_diff_px']:.1f} px")
        print(f"  因果干预收益 CL             : {metrics['cl_causal_gain_rate_pct']:.1f}%"
              f"  ({metrics['n_cl_causal_success']}/{metrics['n_valid']})")

        print("\n[误差分解诊断（Open-loop）]")
        print()
        print(f"  [维度1] GT Block 位移       : "
            f"mean={metrics['mean_gt_block_disp_px']:.1f}px  "
            f"median={metrics['median_gt_block_disp_px']:.1f}px  "
            f"std={metrics['std_gt_block_disp_px']:.1f}px")
        print(f"  [维度2] 预测位移幅度 Fact   : {metrics['mean_fact_disp_px']:.1f} px")
        print(f"  [维度2] 预测位移幅度 CF     : {metrics['mean_cf_disp_px']:.1f} px")
        n_dir = metrics["n_dir_err_valid"]
        print(f"  [维度2] 方向角误差 Fact     : {metrics['mean_fact_dir_err_deg']:.1f}°  (n={n_dir})")
        print(f"  [维度2] 方向角误差 CF       : {metrics['mean_cf_dir_err_deg']:.1f}°  (n={n_dir})")

        print(f"  [维度3] 相对误差 Fact       : {metrics['mean_fact_relative_error']:.3f}  "
            f"（<1.0 = 比原地不动好）")
        print(f"  [维度3] 相对误差 CF         : {metrics['mean_cf_relative_error']:.3f}")

        n_vd = metrics["n_vec_decomp_valid"]
        print(f"  [维度4] 方向余弦 Fact       : {metrics['mean_fact_dir_cosine']:+.3f}  (n={n_vd})")
        print(f"  [维度4] 方向余弦 CF         : {metrics['mean_cf_dir_cosine']:+.3f}")
        print(f"  [维度4] 幅度缩放比 Fact     : {metrics['mean_fact_mag_ratio']:.3f}  （1.0=完美）")
        print(f"  [维度4] 幅度缩放比 CF       : {metrics['mean_cf_mag_ratio']:.3f}")

        print("\n[隐空间与结论]")
        print()
        cg = metrics["causal_gain_rate_pct"]
        print(f"  Open-loop 动作信号判定      : "
            f"{'✓ 模型成功提取动作信号' if cg > 50 else '⚠ 模型动作信号提取不足'}")
        print(f"  Open-loop 幻觉判定          : "
            f"{'⚠ 观察到轨迹记忆幻觉迹象' if metrics['cf_hallucination_accuracy'] > 0 else '✓ 无幻觉（零动作预测与真实轨迹不符）'}")
        print(f"  末步隐空间分叉量（mean±std）: "
          f"{metrics['mean_latent_divergence']:.4f} ± {metrics['std_latent_divergence']:.4f}")
        print(f"\n总耗时: {elapsed:.1f}s")

    # ── 可视化（统计图）──────────────────────────────────────────────────────
    if args.plot:
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        plot_results(metrics, args.plot)

    # ── 可视化（失败案例叠加图）──────────────────────────────────────────────
    if args.failure_viz_dir:
        visualize_cf_failures(
            results=metrics["per_episode"],
            out_dir=args.failure_viz_dir,
            max_cases=args.max_failure_viz,
            pusht_img_size=args.img_size,
        )

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # init_frame 是 numpy 数组，不可 JSON 序列化，序列化前剔除
    per_ep_serializable = [
        {k: v for k, v in r.items() if k != "init_frame"}
        for r in metrics["per_episode"]
    ]
    payload = {
        "args":            vars(args),
        "metrics":         {k: v for k, v in metrics.items() if k != "per_episode"},
        "per_episode":     per_ep_serializable,
        "elapsed_seconds": elapsed,
    }
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()
