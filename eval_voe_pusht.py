"""
eval_voe_pusht.py — PushT Violation-of-Expectation (VoE) 评测

目标：
  在 PushT 上对 LeWM (JEPA) 进行两类物理违约干预，并量化潜空间 Surprise：

    MSE(Pred(z_t, a_t), Enc(o_{t+1}^{VoE}))

支持两种干预：
  1) teleport：在干预时刻将 T 块位置瞬移 (x += dx, y += dy)
  2) color：将 T 块颜色强制改为亮红色

脚本特性：
  - 独立于现有 counterfactual/factual 评测脚本
  - 同一时刻 t_int 对比正常观测与干预观测（Dual-Track）
  - 输出每条轨迹的逐步 MSE 序列到 JSON
  - 绘制均值曲线，并在 t_int 处画垂直虚线

示例：
  python eval_voe_pusht.py --dataset pusht --policy pusht/lewm
  python eval_voe_pusht.py --dataset pusht_expert_train --policy pusht/lewm --n_episodes 80 --horizon 20
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
    return transform(frame_np).unsqueeze(0).unsqueeze(0).to(device)


def _get_cache_dir() -> Path:
    return Path(swm.data.utils.get_cache_dir())


def _resolve_dataset_path(dataset_ref: str) -> Path:
    p = Path(dataset_ref)
    if p.exists() and p.suffix == ".h5":
        return p

    cache_dir = _get_cache_dir()
    candidates = [
        cache_dir / dataset_ref,
        cache_dir / f"{dataset_ref}.h5",
    ]
    if dataset_ref == "pusht":
        candidates.append(cache_dir / "pusht_expert_train.h5")

    for c in candidates:
        if c.exists() and c.suffix == ".h5":
            return c

    raise FileNotFoundError(
        f"找不到数据集 '{dataset_ref}'，请传入 .h5 路径或 $STABLEWM_HOME 下短名。"
    )


class Episode:
    __slots__ = ("frames", "actions", "states", "goal_state")

    def __init__(self, frames: np.ndarray, actions: np.ndarray,
                 states: np.ndarray, goal_state: np.ndarray):
        assert len(frames) == len(actions) + 1
        assert len(states) == len(frames)
        self.frames = frames
        self.actions = actions
        self.states = states
        self.goal_state = goal_state

    def __len__(self) -> int:
        return len(self.actions)


def _iter_episodes_h5(dataset_path: str, n_episodes: int,
                      rng: np.random.Generator) -> Iterator[Episode]:
    import hdf5plugin  # noqa: F401
    import h5py

    with h5py.File(dataset_path, "r") as f:
        if "state" not in f:
            raise RuntimeError("数据集缺少 state 列，无法进行 teleport 干预。")

        offsets = f["ep_offset"][:]
        lens = f["ep_len"][:]
        idx = rng.choice(len(offsets), size=min(n_episodes, len(offsets)), replace=False)

        for i in idx:
            off = int(offsets[i])
            l = int(lens[i])
            if l < 4:
                continue
            frames = f["pixels"][off: off + l]
            actions = f["action"][off: off + l - 1]
            states = f["state"][off: off + l]
            yield Episode(frames=frames, actions=actions,
                          states=states, goal_state=states[-1].copy())


def load_episodes(dataset_path: str, n_episodes: int,
                  rng: np.random.Generator) -> list[Episode]:
    eps = list(_iter_episodes_h5(dataset_path, n_episodes, rng))
    print(f"  加载 {len(eps)} 条 episodes（来自 {dataset_path}）")
    return eps


def load_model(policy: str, device: torch.device):
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
            f"找不到模型权重 '{policy}'。请提供 .pt、含 weights.pt 的目录或短名。"
        )

    ckpt = torch.load(weights_pt, map_location="cpu", weights_only=False)

    act_input_dim = ckpt["action_encoder.patch_embed.weight"].shape[1]
    hidden_dim = ckpt["projector.net.0.weight"].shape[0]
    embed_dim = ckpt["projector.net.0.weight"].shape[1]
    predictor_frames = ckpt["predictor.pos_embedding"].shape[1]
    predictor_depth = sum(
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
    projector = MLP(input_dim=embed_dim, hidden_dim=hidden_dim,
                    output_dim=embed_dim, norm_fn=proj_norm)
    pred_proj = MLP(input_dim=embed_dim, hidden_dim=hidden_dim,
                    output_dim=embed_dim, norm_fn=proj_norm)

    model = JEPA(encoder, predictor, action_encoder, projector, pred_proj)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        raise RuntimeError(f"权重缺失: {missing}")
    if unexpected:
        print(f"  [warn] 忽略多余权重键: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    print(f"  模型: embed_dim={embed_dim}, predictor_depth={predictor_depth}, "
          f"act_input_dim={act_input_dim}, predictor_frames={predictor_frames}")
    return model.to(device).eval().requires_grad_(False)


class VoEEvaluator:
    """PushT VoE 评测器（Dual-Track: 正常观测 vs 干预观测）。"""

    def __init__(self, model, transform: T.Compose, device: torch.device,
                 img_size: int = 224, history_size: int = 1,
                 teleport_dx: float = 50.0, teleport_dy: float = 50.0,
                 color_rgb: tuple[int, int, int] = (255, 40, 40)):
        self.model = model
        self.transform = transform
        self.device = device
        self.img_size = img_size
        self.history_size = history_size
        self.teleport_dx = float(teleport_dx)
        self.teleport_dy = float(teleport_dy)
        self.color_rgb = np.array(color_rgb, dtype=np.uint8)

    def intervene_teleport(self, env, state: np.ndarray) -> np.ndarray:
        """将 T 块坐标瞬移，并调用 _set_state 同步物理引擎。"""
        raw = env.unwrapped
        new_state = np.array(state, dtype=np.float64).copy()
        new_state[2] = np.clip(new_state[2] + self.teleport_dx, 5.0, 506.0)
        new_state[3] = np.clip(new_state[3] + self.teleport_dy, 5.0, 506.0)
        raw._set_state(new_state)  # equivalent to env.set_state()
        return raw._get_obs().copy()

    def intervene_color(self, env) -> None:
        """强制修改 T 块颜色，确保后续 render 直接反映干预。"""
        raw = env.unwrapped
        raw.variation_space["block"]["color"].set_value(self.color_rgb)
        raw._set_body_color(raw.block, self.color_rgb.tolist())

    def _render_fresh(self, env) -> np.ndarray:
        """在个别后端下，第一次 render 可能读到旧缓冲；二次读取可避免取到干预前帧。"""
        _ = env.render()
        return env.render()

    @torch.no_grad()
    def _encode_frame(self, frame: np.ndarray) -> torch.Tensor:
        info = {"pixels": _to_model_input(frame, self.transform, self.device)}
        self.model.encode(info)
        return info["emb"][:, -1:, :]

    def _pack_action(self, actions: np.ndarray, step: int, frameskip: int,
                     act_input_dim: int) -> torch.Tensor:
        a_block = actions[step * frameskip: (step + 1) * frameskip]
        if len(a_block) != frameskip:
            raise RuntimeError("动作块长度不足，无法拼接 frameskip 输入。")
        a_t = torch.tensor(a_block.flatten(), dtype=torch.float32, device=self.device)
        if a_t.numel() != act_input_dim:
            raise RuntimeError(f"action dim mismatch: got {a_t.numel()}, expect {act_input_dim}")
        return a_t.unsqueeze(0).unsqueeze(0)

    def _build_voe_observations(self, episode: Episode, H: int,
                                frameskip: int, t_int: int,
                                intervention: str) -> tuple[list[np.ndarray], bool]:
        """构造模型步边界观测 o_k（k=0..H），其中 k=t_int+1 后进入干预轨迹。"""
        try:
            import gymnasium as gym
        except ImportError as e:
            raise RuntimeError(
                "未安装 gymnasium，无法创建 PushT 环境。"
                "请先安装 stable-worldmodel[env] 或 gymnasium。"
            ) from e

        factual_obs = [episode.frames[k * frameskip] for k in range(H + 1)]
        if t_int >= H:
            return factual_obs, False

        env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=self.img_size)
        try:
            env.reset()
            raw = env.unwrapped
            raw._set_goal_state(episode.goal_state)

            k0 = t_int + 1
            state_k0 = episode.states[k0 * frameskip].copy()
            raw._set_state(state_k0)

            if intervention == "teleport":
                self.intervene_teleport(env, state_k0)
            elif intervention == "color":
                self.intervene_color(env)
            else:
                raise ValueError(f"未知干预类型: {intervention}")

            voe_obs = factual_obs[:k0]
            voe_obs.append(self._render_fresh(env))

            for step in range(k0, H):
                a_block = episode.actions[step * frameskip: (step + 1) * frameskip]
                for a in a_block:
                    env.step(a)
                if intervention == "color":
                    self.intervene_color(env)
                voe_obs.append(self._render_fresh(env))

            if len(voe_obs) != H + 1:
                raise RuntimeError(f"VoE 观测长度错误: {len(voe_obs)} vs {H + 1}")
            return voe_obs, True
        finally:
            env.close()

    @torch.no_grad()
    def evaluate_episode(self, episode: Episode, horizon: int,
                         t_int: int, intervention: str) -> dict | None:
        act_input_dim = self.model.action_encoder.patch_embed.weight.shape[1]
        raw_act_dim = episode.actions.shape[-1]
        frameskip = act_input_dim // raw_act_dim
        if frameskip * raw_act_dim != act_input_dim:
            raise RuntimeError("frameskip 无法从模型和动作维度整除推断。")

        model_steps = len(episode) // frameskip
        H = min(horizon, model_steps)
        if H < 2:
            return None

        if t_int < 0:
            t_int = max(0, H // 2 - 1)
        t_int = int(np.clip(t_int, 0, H - 1))

        voe_obs, has_intervention = self._build_voe_observations(
            episode=episode,
            H=H,
            frameskip=frameskip,
            t_int=t_int,
            intervention=intervention,
        )
        if not has_intervention:
            return None

        # 预编码 factual / VoE 观测，并做双分支闭环：
        # normal 分支: Pred(Enc(o_t^N), a_t) -> Enc(o_{t+1}^N)
        # VoE    分支: Pred(Enc(o_t^V), a_t) -> Enc(o_{t+1}^V)
        # 这样在 t_int 之后，VoE 分支会用干预后的真实观测持续更新上下文，
        # 避免“旧潜变量”导致的长期高误差平台。
        factual_embs = [
            self._encode_frame(episode.frames[k * frameskip])
            for k in range(H + 1)
        ]
        voe_embs = [self._encode_frame(frame) for frame in voe_obs]

        emb_history_normal: list[torch.Tensor] = [factual_embs[0]]
        emb_history_voe: list[torch.Tensor] = [voe_embs[0]]
        act_history: list[torch.Tensor] = []

        mse_normal: list[float] = []
        mse_voe: list[float] = []
        surprise_delta: list[float] = []

        for t in range(H):
            a_t = self._pack_action(episode.actions, t, frameskip, act_input_dim)
            act_emb = self.model.action_encoder(a_t)
            act_history.append(act_emb)

            recent_acts = act_history[-self.history_size:]
            act_context = torch.cat(recent_acts, dim=1)

            emb_context_normal = torch.cat(
                emb_history_normal[-self.history_size:], dim=1
            )
            emb_context_voe = torch.cat(
                emb_history_voe[-self.history_size:], dim=1
            )

            pred_next_normal = self.model.predict(
                emb_context_normal, act_context
            )[:, -1:, :]
            pred_next_voe = self.model.predict(
                emb_context_voe, act_context
            )[:, -1:, :]

            true_next = factual_embs[t + 1]
            voe_next = voe_embs[t + 1]

            m_norm = torch.mean((pred_next_normal - true_next) ** 2).item()
            m_voe = torch.mean((pred_next_voe - voe_next) ** 2).item()
            mse_normal.append(m_norm)
            mse_voe.append(m_voe)
            surprise_delta.append(m_voe - m_norm)

            # 双分支闭环：各自用真实观测编码更新上下文，而不是回灌预测值。
            emb_history_normal.append(true_next)
            emb_history_voe.append(voe_next)

        return {
            "frameskip": frameskip,
            "horizon": H,
            "t_int": t_int,
            "intervention": intervention,
            "mse_normal": mse_normal,
            "mse_voe": mse_voe,
            "surprise_delta": surprise_delta,
        }

    @torch.no_grad()
    def evaluate(self, episodes: list[Episode], horizon: int, t_int: int,
                 intervention: str) -> dict:
        per_episode: list[dict] = []
        skipped = 0

        for i, ep in enumerate(episodes):
            out = self.evaluate_episode(ep, horizon=horizon, t_int=t_int,
                                        intervention=intervention)
            if out is None:
                skipped += 1
            else:
                per_episode.append(out)

            if (i + 1) % 10 == 0:
                print(f"  [{intervention}] 进度: {i + 1}/{len(episodes)}，有效={len(per_episode)}")

        if not per_episode:
            return {"error": f"{intervention}: 无有效 episode"}

        min_len = min(len(x["mse_voe"]) for x in per_episode)
        norm_mat = np.stack([np.array(x["mse_normal"][:min_len]) for x in per_episode])
        voe_mat = np.stack([np.array(x["mse_voe"][:min_len]) for x in per_episode])
        delta_mat = np.stack([np.array(x["surprise_delta"][:min_len]) for x in per_episode])

        t_int_vals = [int(x["t_int"]) for x in per_episode]
        t_int_used = int(round(float(np.mean(t_int_vals))))
        # 聚合后长度以 min_len 为准，干预索引也必须落在该范围内，
        # 否则在 spike 统计时会出现越界（例如 horizon 很大但有短轨迹）。
        t_int_used = int(np.clip(t_int_used, 0, min_len - 1))
        t_effect = min(t_int_used + 1, min_len - 1)

        return {
            "intervention": intervention,
            "n_valid_episodes": len(per_episode),
            "n_skipped_episodes": skipped,
            "horizon": min_len,
            "t_int": t_int_used,
            "t_effect": t_effect,
            "mean_mse_normal": norm_mat.mean(axis=0).tolist(),
            "std_mse_normal": norm_mat.std(axis=0).tolist(),
            "mean_mse_voe": voe_mat.mean(axis=0).tolist(),
            "std_mse_voe": voe_mat.std(axis=0).tolist(),
            "mean_surprise_delta": delta_mat.mean(axis=0).tolist(),
            "spike_mse_voe_at_t_int": float(voe_mat[:, t_int_used].mean()),
            "spike_delta_at_t_int": float(delta_mat[:, t_int_used].mean()),
            "spike_mse_voe_at_t_effect": float(voe_mat[:, t_effect].mean()),
            "spike_delta_at_t_effect": float(delta_mat[:, t_effect].mean()),
            "per_episode": per_episode,
        }


def plot_voe_curves(results: dict[str, dict], out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib 未安装，跳过绘图")
        return

    kinds = [k for k in ("teleport", "color") if k in results and "error" not in results[k]]
    if not kinds:
        return

    fig, axes = plt.subplots(1, len(kinds), figsize=(7 * len(kinds), 4), squeeze=False)
    axes = axes[0]

    for ax, kind in zip(axes, kinds):
        m = results[kind]
        mean_n = np.array(m["mean_mse_normal"])
        std_n = np.array(m["std_mse_normal"])
        mean_v = np.array(m["mean_mse_voe"])
        std_v = np.array(m["std_mse_voe"])
        steps = np.arange(len(mean_v))
        t_int = int(m["t_int"])
        t_effect = int(m.get("t_effect", min(t_int + 1, len(mean_v) - 1)))

        ax.plot(steps, mean_n, color="#1f77b4", linewidth=2, label="Normal target")
        ax.fill_between(steps, mean_n - std_n, mean_n + std_n,
                        color="#1f77b4", alpha=0.18)

        ax.plot(steps, mean_v, color="#d62728", linewidth=2, label="VoE target")
        ax.fill_between(steps, mean_v - std_v, mean_v + std_v,
                        color="#d62728", alpha=0.18)

        ax.axvline(t_int, linestyle="--", color="black", linewidth=1.3,
                   label=f"t_int={t_int}")
        if t_effect != t_int:
            ax.axvline(t_effect, linestyle=":", color="#ff7f0e", linewidth=1.3,
                       label=f"t_int+1={t_effect}")
        ax.set_title(f"PushT VoE: {kind}")
        ax.set_xlabel("Model step t")
        ax.set_ylabel("MSE(Pred(z_t,a_t), Enc(o_{t+1}))")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  曲线图已保存至 {out_path}")


def build_visual_tracks(evaluator: VoEEvaluator, episode: Episode,
                        horizon: int, t_int: int) -> tuple[dict[str, list[np.ndarray]], int, int] | None:
    """从同一条 episode 生成 3 组可同步播放的轨迹帧。"""
    act_input_dim = evaluator.model.action_encoder.patch_embed.weight.shape[1]
    raw_act_dim = episode.actions.shape[-1]
    frameskip = act_input_dim // raw_act_dim
    if frameskip * raw_act_dim != act_input_dim:
        return None

    model_steps = len(episode) // frameskip
    H = min(horizon, model_steps)
    if H < 2:
        return None

    if t_int < 0:
        t_int = max(0, H // 2 - 1)
    t_int = int(np.clip(t_int, 0, H - 1))

    normal_obs = [episode.frames[k * frameskip] for k in range(H + 1)]

    teleport_obs, ok_tp = evaluator._build_voe_observations(
        episode=episode,
        H=H,
        frameskip=frameskip,
        t_int=t_int,
        intervention="teleport",
    )
    color_obs, ok_cl = evaluator._build_voe_observations(
        episode=episode,
        H=H,
        frameskip=frameskip,
        t_int=t_int,
        intervention="color",
    )
    if not (ok_tp and ok_cl):
        return None

    tracks = {
        "unperturbed": [frame for frame in normal_obs],
        "color": [frame for frame in color_obs],
        "teleport": [frame for frame in teleport_obs],
    }
    return tracks, t_int, H


def plot_voe_combined_animation(results: dict[str, dict],
                                tracks: dict[str, list[np.ndarray]],
                                t_int: int, H: int,
                                out_path: str, fps: int = 6,
                                marker_mode: str = "t_int") -> None:
    """1x4 动画：左三图同步播放，右图显示完整的静态 Surprise 曲线，并严格根据超参数 t_int 标记虚线。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("  [skip] matplotlib/animation 不可用，跳过组合动图")
        return

    fig, axes = plt.subplots(
        1, 4,
        figsize=(20, 4.8),
        gridspec_kw={"width_ratios": [1, 1, 1, 1.8]},
        constrained_layout=True,
    )

    titles = [
        ("unperturbed", "Unperturbed"),
        ("color", "Block color change"),
        ("teleport", "Teleportation"),
    ]

    im_artists = []
    for i, (key, title) in enumerate(titles):
        im = axes[i].imshow(tracks[key][0])
        im_artists.append(im)
        axes[i].set_title(title)
        axes[i].axis("off")

    ax = axes[3]

    # 获取有效的数据长度
    available_lens: list[int] = []
    if "teleport" in results and "error" not in results["teleport"]:
        available_lens.append(len(results["teleport"]["mean_mse_voe"]))
    if "color" in results and "error" not in results["color"]:
        available_lens.append(len(results["color"]["mean_mse_voe"]))

    curve_H = min([H] + available_lens) if available_lens else H

    # 1. 严格根据超参数获取 t_int，定位虚线
    metrics_ref = results.get("teleport") or results.get("color")
    if metrics_ref and "error" not in metrics_ref:
        # 读取 evaluator 记录的实际 t_int
        t_mark = int(metrics_ref.get("t_int", t_int))
    else:
        t_mark = t_int

    t_mark = int(np.clip(t_mark, 0, max(0, curve_H - 1)))
    mark_label = f"Intervention t={t_mark}"

    # 提取完整数据
    base_normal = np.array(metrics_ref["mean_mse_normal"][:curve_H]) if metrics_ref else np.zeros(curve_H)
    data_teleport = np.array(results["teleport"]["mean_mse_voe"][:curve_H]) if "teleport" in results and "error" not in results["teleport"] else None
    data_color = np.array(results["color"]["mean_mse_voe"][:curve_H]) if "color" in results and "error" not in results["color"] else None

    # 2. 一次性绘制完整的静态折线图
    steps = np.arange(curve_H)
    if data_teleport is not None:
        ax.plot(steps, data_teleport, color="#d62728", linewidth=2, label="Teleportation")
    if data_color is not None:
        ax.plot(steps, data_color, color="#ff7f0e", linewidth=2, label="Block color change")
    if metrics_ref is not None:
        ax.plot(steps, base_normal, color="#1f77b4", linewidth=2, label="Unperturbed")

    # 绘制固定不变的黑色虚线和标签
    ax.axvline(t_mark, linestyle="--", color="black", linewidth=1.3, label=mark_label)

    # 修饰图表
    ax.set_title("Surprise Score (MSE)")
    ax.set_xlabel("Model step t")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    n_frames = H + 1

    # 3. 核心更新逻辑：现在只更新左侧的动图，不再改动右侧折线图
    def _update(frame_idx: int):
        for im, key in zip(im_artists, ("unperturbed", "color", "teleport")):
            im.set_data(tracks[key][frame_idx])
        return im_artists

    anim = FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=int(1000 / max(1, fps)),
        blit=False,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out = Path(out_path)
    suffix = out.suffix.lower()
    try:
        if suffix == ".gif":
            anim.save(out_path, writer="pillow", fps=fps)
        else:
            anim.save(out_path, writer="ffmpeg", fps=fps)
    except Exception as e:
        fallback = out.with_suffix(".gif")
        print(f"  [warn] 动画写入失败（{e}），回退到 GIF: {fallback}")
        anim.save(str(fallback), writer="pillow", fps=fps)

    plt.close(fig)
    print(f"  组合动图已保存至 {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PushT VoE Eval (teleport + color)")
    p.add_argument("--dataset", default="pusht",
                   help="数据集短名或 .h5 路径（默认 pusht）")
    p.add_argument("--policy", required=True,
                   help="LeWM 权重：短名、目录或 .pt")
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--horizon", type=int, default=20,
                   help="最大模型步数")
    p.add_argument("--history_size", type=int, default=1)
    p.add_argument("--t_int", type=int, default=8,
                   help="干预时刻（模型步）；-1 表示每条轨迹使用中点")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--teleport_dx", type=float, default=50.0)
    p.add_argument("--teleport_dy", type=float, default=50.0)
    p.add_argument("--color_r", type=int, default=255)
    p.add_argument("--color_g", type=int, default=40)
    p.add_argument("--color_b", type=int, default=40)

    p.add_argument("--output", default="logs_eval/pusht_voe_results.json")
    p.add_argument("--plot", default="logs_eval/pusht_voe_mse_curve.gif")
    p.add_argument("--fps", type=int, default=6,
                   help="组合动图帧率")
    p.add_argument("--marker_mode", choices=["t_int", "t_effect", "peak"],
                   default="t_effect",
                   help="第四图虚线标记位置：t_int(干预时刻) / t_effect(t_int+1) / peak(曲线峰值)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    dataset_path = _resolve_dataset_path(args.dataset)
    print(f"设备: {device}")
    print("加载数据集...")
    episodes = load_episodes(str(dataset_path), args.n_episodes, rng)
    if not episodes:
        print("错误：未加载到有效 episodes")
        return

    print("加载模型...")
    model = load_model(args.policy, device)

    evaluator = VoEEvaluator(
        model=model,
        transform=_make_transform(args.img_size),
        device=device,
        img_size=args.img_size,
        history_size=args.history_size,
        teleport_dx=args.teleport_dx,
        teleport_dy=args.teleport_dy,
        color_rgb=(args.color_r, args.color_g, args.color_b),
    )

    print("\n开始 VoE 评测（teleport + color）...")
    t0 = time.time()
    results: dict[str, dict] = {}
    for intervention in ("teleport", "color"):
        print(f"\n[{intervention}] 评测中...")
        results[intervention] = evaluator.evaluate(
            episodes=episodes,
            horizon=args.horizon,
            t_int=args.t_int,
            intervention=intervention,
        )
    elapsed = time.time() - t0

    print("\n" + "=" * 64)
    print("PUSHT VOE EVALUATION RESULTS")
    print("=" * 64)
    for intervention in ("teleport", "color"):
        m = results[intervention]
        if "error" in m:
            print(f"  {intervention:>9}: {m['error']}")
            continue
        t_effect = int(m.get("t_effect", m["t_int"]))
        print(f"  {intervention:>9}: n={m['n_valid_episodes']}  "
              f"t_int={m['t_int']}  "
              f"t_effect={t_effect}  "
              f"spike_voe(t_int+1)={m['spike_mse_voe_at_t_effect']:.6f}  "
              f"spike_delta(t_int+1)={m['spike_delta_at_t_effect']:.6f}  "
              f"[at_t_int: voe={m['spike_mse_voe_at_t_int']:.6f}, "
              f"delta={m['spike_delta_at_t_int']:.6f}]")
    print(f"\n总耗时: {elapsed:.1f}s")

    if args.plot:
        track_payload = None
        for ep in episodes:
            track_payload = build_visual_tracks(
                evaluator=evaluator,
                episode=ep,
                horizon=args.horizon,
                t_int=args.t_int,
            )
            if track_payload is not None:
                break

        if track_payload is None:
            print("  [skip] 未找到可用于动图可视化的有效 episode，改为仅输出曲线图")
            plot_voe_curves(results, args.plot)
        else:
            tracks, t_int_vis, H_vis = track_payload
            plot_voe_combined_animation(
                results=results,
                tracks=tracks,
                t_int=t_int_vis,
                H=H_vis,
                out_path=args.plot,
                fps=args.fps,
                marker_mode=args.marker_mode,
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "dataset_path": str(dataset_path),
        "elapsed_seconds": elapsed,
        "results": results,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"结果已写入 {out_path}")


if __name__ == "__main__":
    main()
