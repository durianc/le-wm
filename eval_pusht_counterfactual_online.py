"""
eval_pusht_counterfactual_online.py — Push-T 在线动作干预反事实评测

从数据集中回放真实动作直到 Agent-Block 首次接触（T_contact），
然后在真实 pymunk 物理环境中分叉两条轨迹：

  Factual     — 继续回放数据集原始动作 a_{T_contact : T_contact+H}
  Counterfactual — 替换为零动作（Agent 原地静止）或随机动作

两路均在真实 PushT 物理引擎中运行到 eval_budget 步，
记录 Block 是否到达目标位置，输出各自的成功率及差值。

数据集格式（stable-worldmodel HDF5，$STABLEWM_HOME/<name>.h5）：
  pixels    (N, 224, 224, 3) uint8
  action    (N, 2)           float32
  state     (N, 7)           float64 — [ax, ay, bx, by, b_angle, vx, vy]
  ep_offset (E,)             int64
  ep_len    (E,)             int32

Usage:
    python eval_pusht_counterfactual_online.py --dataset pusht_expert_train --policy pusht/lewm
    python eval_pusht_counterfactual_online.py --dataset pusht_expert_train --policy random
    python eval_pusht_counterfactual_online.py --dataset pusht_expert_train --policy pusht/lewm --cf_mode random
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

import stable_worldmodel as swm


# ── 缓存目录 ────────────────────────────────────────────────────────────────────

def _get_cache_dir() -> Path:
    return Path(swm.data.utils.get_cache_dir())


# ── 数据集加载 ─────────────────────────────────────────────────────────────────

class Episode:
    """单条 episode 的裸数据容器（numpy）。"""
    __slots__ = ("frames", "actions", "states", "goal_state")

    def __init__(self, frames: np.ndarray, actions: np.ndarray,
                 states: np.ndarray, goal_state: np.ndarray):
        # frames:     (T+1, H, W, C) uint8
        # actions:    (T,   2)       float32
        # states:     (T+1, 7)       float64 — [ax, ay, bx, by, b_angle, vx, vy]
        # goal_state: (7,)           float64 — 目标状态（同 _set_goal_state 格式）
        assert len(frames) == len(actions) + 1
        assert len(states) == len(frames)
        self.frames     = frames
        self.actions    = actions
        self.states     = states
        self.goal_state = goal_state

    def __len__(self) -> int:
        return len(self.actions)


def _iter_episodes_h5(
    dataset_path: str,
    n_episodes: int,
    rng: np.random.Generator,
) -> Iterator[Episode]:
    """从 HDF5 读取 episodes，加载 state 列用于接触检测和环境初始化。"""
    import hdf5plugin  # noqa: F401
    import h5py

    with h5py.File(dataset_path, "r") as f:
        if "state" not in f:
            raise RuntimeError(
                "数据集缺少 'state' 列（需要 [ax,ay,bx,by,b_angle,vx,vy]），"
                "无法进行在线反事实评测。"
            )

        ep_offsets = f["ep_offset"][:]
        ep_lens    = f["ep_len"][:]
        n_ep_avail = len(ep_offsets)
        indices = rng.choice(n_ep_avail, size=min(n_episodes, n_ep_avail),
                             replace=False)

        for idx in indices:
            off = int(ep_offsets[idx])
            l   = int(ep_lens[idx])
            if l < 3:
                continue

            frames  = f["pixels"][off : off + l]       # (l, H, W, C)
            actions = f["action"][off : off + l - 1]   # (l-1, 2)
            states  = f["state"][off : off + l]        # (l, 7)

            # goal_state：数据集最后一帧的 state（与 eval.py 的 goal_offset_steps 逻辑一致）
            goal_state = states[-1].copy()

            yield Episode(frames=frames, actions=actions,
                          states=states, goal_state=goal_state)


def load_episodes(
    dataset_name: str,
    n_episodes: int,
    rng: np.random.Generator,
) -> list[Episode]:
    """从 $STABLEWM_HOME/<dataset_name>.h5 加载 episodes。"""
    h5_path = _get_cache_dir() / f"{dataset_name}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"数据集文件不存在: {h5_path}\n"
            f"请确认 $STABLEWM_HOME 设置正确（当前: {_get_cache_dir()}）"
        )
    episodes = list(_iter_episodes_h5(str(h5_path), n_episodes, rng))
    print(f"  加载 {len(episodes)} 条 episodes（来自 {h5_path}）")
    return episodes


# ── 关键帧检测 ─────────────────────────────────────────────────────────────────

def find_contact_frame(states: np.ndarray, contact_thresh: float) -> int | None:
    """找到 Agent-Block 首次距离 < contact_thresh 的 native 帧索引。

    state 格式：[ax, ay, bx, by, b_angle, vx, vy]
    坐标系：PushT 内部坐标（512×512 窗口）。
    """
    agent_pos = states[:, :2]
    block_pos = states[:, 2:4]
    dists = np.linalg.norm(agent_pos - block_pos, axis=1)
    candidates = np.where(dists < contact_thresh)[0]
    return int(candidates[0]) if len(candidates) > 0 else None


# ── 反事实动作构造 ─────────────────────────────────────────────────────────────

def make_cf_actions(
    actions: np.ndarray,
    start: int,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """返回干预后的完整动作序列副本（[start:] 段被替换）。

    Args:
        actions: (T, 2) 原始动作序列
        start:   干预起始索引（含）
        mode:    "zero" | "random"
        rng:     随机数生成器（mode="random" 时使用）
    """
    cf = actions.copy()
    n_replace = len(actions) - start
    if n_replace <= 0:
        return cf
    if mode == "zero":
        cf[start:] = 0.0
    elif mode == "random":
        cf[start:] = rng.uniform(-1.0, 1.0, size=(n_replace, actions.shape[-1]))
    else:
        raise ValueError(f"未知的反事实模式 '{mode}'")
    return cf


# ── 单条 Episode 的在线反事实评测 ──────────────────────────────────────────────

def _make_pusht_env(resolution: int = 224):
    """创建裸 PushT 环境（不套 swm.World 的各层 wrapper）。

    直接实例化以便使用 _set_state / _set_goal_state 接口。
    """
    import gymnasium as gym
    env = gym.make(
        "swm/PushT-v1",
        render_mode="rgb_array",
        resolution=resolution,
    )
    # 取底层 PushT 实例
    return env


def _run_episode_in_env(
    env,
    init_state: np.ndarray,
    goal_state: np.ndarray,
    actions: np.ndarray,
    eval_budget: int,
) -> dict:
    """在真实 PushT 环境中运行一条轨迹，返回成功/失败及每步信息。

    流程：
      1. reset 环境
      2. 用 _set_state 设置初始物理状态
      3. 用 _set_goal_state 设置目标
      4. 依次执行 actions（不足 eval_budget 则补零）
      5. 返回是否在 eval_budget 内成功

    Args:
        env:         gym.Env（PushT）
        init_state:  (7,) float — 初始物理状态 [ax,ay,bx,by,b_angle,vx,vy]
        goal_state:  (7,) float — 目标物理状态（同 _set_goal_state 格式）
        actions:     (T, 2) float — 要回放的动作序列
        eval_budget: 最大步数上限

    Returns:
        dict with keys: success (bool), n_steps (int), final_dist (float)
    """
    env.reset()
    raw_env = env.unwrapped

    # 设置初始物理状态和目标
    raw_env._set_state(init_state)
    raw_env._set_goal_state(goal_state)

    success = False
    final_dist = float("inf")

    for step in range(eval_budget):
        if step < len(actions):
            action = actions[step]
        else:
            action = np.zeros(2, dtype=np.float32)   # 超出动作序列后静止

        _, _, terminated, _, info = env.step(action)

        if terminated:
            success = True
            final_dist = 0.0
            break

        # 记录末步距离（用于连续指标）
        if "block_pose" in info and raw_env.goal_state is not None:
            cur_state = raw_env._get_obs()
            _, dist = raw_env.eval_state(raw_env.goal_state, cur_state)
            final_dist = float(dist)

    return {
        "success":    success,
        "n_steps":    step + 1,
        "final_dist": final_dist,
    }


def evaluate_episode_online(
    episode: Episode,
    contact_thresh: float,
    eval_budget: int,
    cf_mode: str,
    rng: np.random.Generator,
    img_size: int = 224,
) -> dict | None:
    """对单条 episode 做在线双路反事实评测。

    Returns:
        包含 factual/cf 成功标志及中间信息的 dict，
        若 episode 无有效接触帧则返回 None。
    """
    # ── Step 1: 关键帧检测 ───────────────────────────────────────────────────
    t_contact = find_contact_frame(episode.states, contact_thresh)
    if t_contact is None:
        return None
    if t_contact >= len(episode.actions):
        return None   # 接触帧已是最后一步，无法分叉

    # ── Step 2: 构造两路动作序列 ─────────────────────────────────────────────
    fact_actions = episode.actions                               # 原始
    cf_actions   = make_cf_actions(episode.actions, t_contact,
                                   cf_mode, rng)                # 干预

    # 初始物理状态：episode 第 0 帧对应的物理状态
    init_state = episode.states[0]
    goal_state = episode.goal_state

    # ── Step 3: 双路在线运行 ─────────────────────────────────────────────────
    env = _make_pusht_env(resolution=img_size)
    try:
        fact_result = _run_episode_in_env(
            env, init_state, goal_state, fact_actions, eval_budget
        )
        cf_result = _run_episode_in_env(
            env, init_state, goal_state, cf_actions, eval_budget
        )
    finally:
        env.close()

    return {
        "t_contact":        t_contact,
        "fact_success":     fact_result["success"],
        "cf_success":       cf_result["success"],
        "fact_final_dist":  fact_result["final_dist"],
        "cf_final_dist":    cf_result["final_dist"],
        "fact_n_steps":     fact_result["n_steps"],
        "cf_n_steps":       cf_result["n_steps"],
    }


# ── 聚合评测 ───────────────────────────────────────────────────────────────────

def run_online_counterfactual_eval(
    episodes: list[Episode],
    contact_thresh: float,
    eval_budget: int,
    cf_mode: str,
    rng: np.random.Generator,
    img_size: int = 224,
) -> dict:
    """遍历所有 episodes，聚合在线反事实评测指标。"""
    results = []
    skipped = 0

    for i, ep in enumerate(episodes):
        res = evaluate_episode_online(
            episode=ep,
            contact_thresh=contact_thresh,
            eval_budget=eval_budget,
            cf_mode=cf_mode,
            rng=rng,
            img_size=img_size,
        )
        if res is None:
            skipped += 1
        else:
            results.append(res)

        if (i + 1) % 5 == 0:
            n = len(results)
            if n > 0:
                sr_f = sum(r["fact_success"] for r in results) / n * 100
                sr_c = sum(r["cf_success"]   for r in results) / n * 100
                print(f"  进度: {i+1}/{len(episodes)}  "
                      f"有效={n}  跳过={skipped}  "
                      f"Fact={sr_f:.1f}%  CF={sr_c:.1f}%")

    if not results:
        return {"error": "没有找到含有效接触帧的 episode，请调大 --contact_thresh"}

    n = len(results)
    fact_successes = [r["fact_success"] for r in results]
    cf_successes   = [r["cf_success"]   for r in results]
    contact_steps  = [r["t_contact"]    for r in results]

    fact_sr = float(sum(fact_successes)) / n * 100.0
    cf_sr   = float(sum(cf_successes))   / n * 100.0

    return {
        "n_valid_episodes":   n,
        "n_skipped_episodes": skipped,
        "contact_thresh":     contact_thresh,
        "cf_mode":            cf_mode,
        "eval_budget":        eval_budget,

        # ── 核心指标 ──────────────────────────────────────────────────────────
        "factual_success_rate":        fact_sr,
        "counterfactual_success_rate": cf_sr,
        "success_rate_delta":          fact_sr - cf_sr,   # 正值 = 干预确实降低了成功率

        # ── 每条 episode 的详细结果 ────────────────────────────────────────────
        "per_episode": results,

        # ── 距离指标（连续，不依赖阈值判定）──────────────────────────────────
        "mean_fact_final_dist": float(np.mean([r["fact_final_dist"] for r in results])),
        "mean_cf_final_dist":   float(np.mean([r["cf_final_dist"]   for r in results])),

        # ── 接触帧统计 ────────────────────────────────────────────────────────
        "contact_step_mean": float(np.mean(contact_steps)),
        "contact_step_std":  float(np.std(contact_steps)),
    }


# ── 可视化 ─────────────────────────────────────────────────────────────────────

def plot_results(metrics: dict, out_path: str) -> None:
    """绘制成功率对比条形图 + 接触帧分布直方图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib 未安装，跳过绘图")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── 左图：成功率对比 ────────────────────────────────────────────────────
    ax = axes[0]
    labels = ["Factual\n(原始动作)", f"Counterfactual\n(干预={metrics['cf_mode']})"]
    values = [metrics["factual_success_rate"],
              metrics["counterfactual_success_rate"]]
    colors = ["#1f77b4", "#ff7f0e"]
    bars = ax.bar(labels, values, color=colors, width=0.4, edgecolor="gray")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, max(values) * 1.25 + 5)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_title(
        f"Online Counterfactual Success Rate\n"
        f"Δ = {metrics['success_rate_delta']:+.1f}%  "
        f"(n={metrics['n_valid_episodes']}, thresh={metrics['contact_thresh']})",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3)

    # ── 右图：接触帧分布直方图 ──────────────────────────────────────────────
    ax = axes[1]
    contact_steps = [r["t_contact"] for r in metrics["per_episode"]]
    ax.hist(contact_steps, bins=20, color="#2ca02c", edgecolor="gray", alpha=0.8)
    ax.axvline(metrics["contact_step_mean"], color="red", lw=1.5,
               linestyle="--",
               label=f"mean={metrics['contact_step_mean']:.1f}")
    ax.set_xlabel("T_contact (native steps)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Contact Frame Distribution", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Push-T Online Counterfactual Eval  "
        f"(eval_budget={metrics['eval_budget']})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存至 {out_path}")


# ── CLI 入口 ───────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Push-T Online Counterfactual Evaluation — Action Intervention"
    )
    p.add_argument(
        "--dataset", default="pusht_expert_train",
        help="数据集名称（$STABLEWM_HOME/<name>.h5），默认 pusht_expert_train"
    )
    p.add_argument(
        "--policy", default="random",
        help="（保留参数，在线版本不使用 LeWM 模型，物理引擎即为 ground truth）"
    )
    p.add_argument(
        "--contact_thresh", type=float, default=80.0,
        help="Agent-Block 接触距离阈值（PushT 坐标系像素，默认 80）"
    )
    p.add_argument(
        "--cf_mode", choices=["zero", "random"], default="zero",
        help="反事实动作：zero=Agent 静止，random=随机游走（默认 zero）"
    )
    p.add_argument(
        "--n_episodes", type=int, default=50,
        help="评测 episode 数量（默认 50）"
    )
    p.add_argument(
        "--eval_budget", type=int, default=300,
        help="每条 episode 最大步数（默认 300）"
    )
    p.add_argument(
        "--img_size", type=int, default=224,
        help="渲染分辨率（默认 224）"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认 42）"
    )
    p.add_argument(
        "--output", default="logs_eval/pusht_cf_online_results.json",
        help="JSON 结果输出路径"
    )
    p.add_argument(
        "--plot", default="logs_eval/pusht_cf_online_plot.png",
        help="图表输出路径（传空字符串跳过）"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"反事实模式: {args.cf_mode}  接触阈值: {args.contact_thresh}  "
          f"eval_budget: {args.eval_budget}")

    print("加载数据集...")
    episodes = load_episodes(args.dataset, args.n_episodes, rng)
    if not episodes:
        print("错误：未加载到任何有效 episode")
        return

    print(f"\n开始在线评测：{len(episodes)} episodes")
    t0 = time.time()
    metrics = run_online_counterfactual_eval(
        episodes=episodes,
        contact_thresh=args.contact_thresh,
        eval_budget=args.eval_budget,
        cf_mode=args.cf_mode,
        rng=np.random.default_rng(args.seed + 1),
        img_size=args.img_size,
    )
    elapsed = time.time() - t0

    # ── 打印摘要 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("PUSH-T ONLINE COUNTERFACTUAL EVALUATION RESULTS")
    print("=" * 64)
    if "error" in metrics:
        print(f"  错误: {metrics['error']}")
        return

    print(f"  有效 episodes              : {metrics['n_valid_episodes']}"
          f"  （跳过 {metrics['n_skipped_episodes']}）")
    print(f"  eval_budget                : {metrics['eval_budget']} steps")
    print(f"  contact_thresh             : {metrics['contact_thresh']}")
    print(f"  cf_mode                    : {metrics['cf_mode']}")
    print()
    print(f"  Factual  成功率            : {metrics['factual_success_rate']:.1f}%")
    print(f"  Counterfactual 成功率      : {metrics['counterfactual_success_rate']:.1f}%")
    print(f"  成功率差值 (Fact - CF)     : {metrics['success_rate_delta']:+.1f}%")
    print()
    print(f"  Factual  末步距离均值      : {metrics['mean_fact_final_dist']:.2f}")
    print(f"  Counterfactual 末步距离均值: {metrics['mean_cf_final_dist']:.2f}")
    print()
    print(f"  T_contact (mean ± std)     : "
          f"{metrics['contact_step_mean']:.1f} ± {metrics['contact_step_std']:.1f} steps")
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
        "metrics":         {k: v for k, v in metrics.items()
                            if k != "per_episode"},
        "per_episode":     metrics["per_episode"],
        "elapsed_seconds": elapsed,
    }
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"结果已写入 {out_path}")


if __name__ == "__main__":
    main()
