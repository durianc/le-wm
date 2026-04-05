# Two-Room 评测逻辑说明

本文根据仓库中的 Two-Room 环境实现与评测脚本，对事实评测（Factual）与反事实评测（Counterfactual）的执行逻辑、因果结构与指标定义做形式化整理。文中所有结论以实际代码为准，尤其区分“配置中存在”与“脚本中真实生效”的参数。

## 1. 环境建模：Two-Room 的因果结构

Two-Room 是一个二维导航任务。环境中的关键结构变量可以抽象为：

$$
\mathcal{E} = (w_x, w_y, \theta_w)
$$

其中：

- $w_x$：墙的位置坐标。当前实现中固定为竖直墙的中心位置 `WALL_CENTER = 112`，但反事实环境允许通过 `reset(options={"wall_x": ...})` 进行覆盖。
- $w_y$：门的位置坐标。对于竖直墙，门沿 $y$ 轴分布，`door_y` 表示门中心在墙上的纵向坐标。
- $\theta_w$：墙和门的几何参数，包括墙厚、门半宽（door half-extent）、墙轴向等。

### 1.1 状态、动作与观测

环境的核心状态可以写成：

$$
s_t = (p_t, g, w_x, w_y)
$$

其中 $p_t=(x_t,y_t)$ 是 agent 位置，$g$ 是目标位置。代码中的观测向量是一个长度为 10 的状态向量，包含 agent、target 和最多 3 个门的中心坐标。

动作空间为二维连续动作 $a_t \in [-1,1]^2$。环境在 `step()` 中执行：

$$
\tilde{p}_{t+1} = p_t + \mathrm{clip}(a_t,-1,1) \cdot v
$$

其中 $v$ 是 agent speed。随后通过碰撞函数将 $\tilde{p}_{t+1}$ 投影回可行区域得到最终位置 $p_{t+1}$。

### 1.2 墙与门如何约束动力学

Two-Room 的因果约束不是“奖励驱动”，而是几何约束驱动：

1. 墙形成不可穿越带。墙中心在 $w_x=112$，墙厚由 `wall.thickness` 决定，代码在 `_apply_collisions()` 中将墙视为一段厚条带。
2. 门在墙上切出可通行窗口。门由中心坐标 $w_y$ 和半宽 $s$ 决定；只有当 agent 轨迹穿过门附近时，跨墙才被允许。
3. 碰撞判断考虑 agent 半径。即使 agent 中心尚未到达墙面，只要球形 agent 外沿触碰到墙带，也会被回推到墙边界。

对竖直墙而言，动力学可以写成带约束的状态转移：

$$
p_{t+1} = \Pi_{\Omega(w_x,w_y)}\bigl(p_t + v a_t\bigr)
$$

其中 $\Pi_{\Omega}$ 表示对由墙、门、边界共同定义的可行区域做投影。若 $p_{t+1}$ 穿过墙带但不满足门条件，则 x 分量被截断到墙边界；若满足门条件，则允许穿墙。

### 1.3 因果图

可以把 Two-Room 的生成过程抽象为：

$$
(w_x, w_y, p_0, \{a_t\}_{t=0}^{T-1}) \rightarrow \{p_t\}_{t=1}^{T}
$$

其中：

- $w_x$ 决定“墙在哪里”，即跨房间的约束边界。
- $w_y$ 决定“门开在哪里”，即边界上的可穿越窗口。
- $a_t$ 只在约束允许的区域内推动状态演化。

从因果角度看，`wall_x` 和 `door_y` 不是外观变量，而是直接改变动力学约束的结构变量。

## 2. 事实评测（Factual）：Latent Space 预测与 MSE

事实评测脚本是 `eval_factual.py`。它从 HDF5 轨迹中读取真实的 `proprio`（位置）和 `action` 序列，并在 TwoRoomCF 环境中按真实状态重渲染初始帧与后续帧，再在潜在空间中比较预测与真值。

### 2.1 评测流程

对每个 episode，代码执行的是开环（open-loop）latent rollout：

1. 从数据集读取初始位置 $p_0$、动作序列 $\{a_t\}_{t=0}^{H-1}$ 和真实位置轨迹 $\{p_t\}_{t=0}^{H}$。
2. 用 `reset(options={"agent_pos": p_0})` 将环境重置到真实初态，并渲染初始观测 $o_0$。
3. 编码得到初始潜变量：

$$
z_0^{\text{pred}} = \mathrm{Enc}(o_0)
$$

4. 对每个时间步 $t$：

$$
z_{t+1}^{\text{pred}} = \mathrm{Pred}(z_{t-h+1:t}^{\text{pred}}, \mathrm{Enc}_a(a_t))
$$

其中 $h$ 是 `wm.history_size`，在本仓库默认是 1。
5. 将预测的 $z_{t+1}^{\text{pred}}$ 作为下一步 context，继续无教师强制地展开。
6. 用真实位置 $p_{t+1}$ 重新渲染得到 $o_{t+1}$，并编码：

$$
z_{t+1}^{\text{true}} = \mathrm{Enc}(o_{t+1})
$$

7. 计算每一步的 latent MSE。

### 2.2 MSE 指标定义

脚本中的逐步误差是：

$$
\mathrm{MSE}_t = \frac{1}{D}\left\lVert z_{t+1}^{\text{pred}} - z_{t+1}^{\text{true}} \right\rVert_2^2
$$

其中 $D$ 是 latent 维度。

对一个 episode 的 horizon $H$，得到：

$$
\mathbf{m} = [\mathrm{MSE}_0, \mathrm{MSE}_1, \dots, \mathrm{MSE}_{H-1}]
$$

跨 episode 聚合时，代码对齐到最短序列长度，再计算：

$$
\overline{\mathrm{MSE}}_t = \mathbb{E}[\mathrm{MSE}_t], \qquad
\mathrm{MSE}_{\text{all}} = \mathbb{E}_{t,\text{ep}}[\mathrm{MSE}_t]
$$

同时还计算误差随时间的线性漂移斜率：

$$
\mathrm{drift\_slope} = \mathrm{coef}_1\left(\mathrm{polyfit}(t,\overline{\mathrm{MSE}}_t,1)\right)
$$

### 2.3 实现中的关键细节

- 评测不是 teacher-forcing；虽然 `config/eval/factual_eval.yaml` 里保留了 `open_loop: true` 注释，但脚本实际始终使用预测 embedding 继续展开。
- 动作输入会按训练时的有效维度重复展开：脚本从 `action_encoder.patch_embed.weight.shape[1]` 反推 frameskip，然后把原始 2 维动作复制到对应维数。
- `history_size` 取自 `cfg.wm.history_size`，在 Two-Room 的评测配置里为 1。

## 3. 反事实评测（Counterfactual）：重渲染、Oracle 与对比逻辑

反事实评测脚本是 `eval_cf.py`，样本采集逻辑在 `cf_sampler.py`，真值模拟器在 `cf_oracle.py`。

### 3.1 “重渲染观测”如何实现干预

反事实评测不修改 HDF5 轨迹中的动作序列，而是修改环境参数，然后重新渲染初始观测：

$$
o_0^{\text{cf}} = \mathrm{Render}\bigl(\mathrm{reset}(\text{options}=\xi^{\text{cf}})\bigr)
$$

其中 $\xi^{\text{cf}}$ 是反事实参数，例如：

- `agent_pos`：干预初始位置
- `wall_x`：干预墙位置
- `door_y`：干预门位置

这等价于在因果图中执行 `do()` 操作：把某个结构变量从 factual 值替换为 counterfactual 值，再在同一动作序列下比较结果。

### 3.2 Oracle 对比逻辑

Oracle 由 `oracle_rollout(env_params, actions)` 实现。它在 `lewm/TwoRoomCF-v0` 上做纯物理 rollout，返回：

- `positions`: 真值位置轨迹 $\{p_t\}$
- `through_door`: 是否发生穿门/跨墙事件

对每个采样对 $(\xi^{\text{fact}}, \xi^{\text{cf}})$，先分别跑 oracle：

$$
R^{\text{fact}} = \mathrm{Oracle}(\xi^{\text{fact}}, A), \qquad
R^{\text{cf}} = \mathrm{Oracle}(\xi^{\text{cf}}, A)
$$

如果它们的事件结果不同，即：

$$
R^{\text{fact}}_{\text{pass}} \neq R^{\text{cf}}_{\text{pass}}
$$

则该样本被视为“oracle 发生了有效因果变化”。

### 3.3 模型如何在反事实任务上被评估

`eval_cf.py` 中的模型预测并不是直接预测像素，而是：

1. 先编码初始重渲染帧，得到 latent 表示 $z_0$。
2. 通过一个冻结的线性 probe 从 $z_0$ 回归 agent 初始位置 $\hat{p}_0$。
3. 用预测位置和反事实环境参数重新调用 oracle rollout，得到模型对 `through_door` 的预测。

因此，反事实评测重点不是像素级重建，而是：

> 模型是否能从重渲染后的初始观测中恢复出结构变量所对应的空间语义，并在同一动作序列下正确推出结果变化。

## 4. 核心指标与公式

### 4.1 样本筛选指标

`cf_sampler.py` 会先构造 effectful sample pair，再筛掉无效样本。关键条件包括：

$$
\mathbb{I}\left(R^{\text{fact}}_{\text{pass}} \neq R^{\text{cf}}_{\text{pass}}\right)=1
$$

以及可选的最小轨迹差异约束：

$$
\mathrm{effect\_mse}_{\text{oracle}} = \frac{1}{T'}\sum_{t=0}^{T'-1} \left\lVert p_t^{\text{fact}} - p_t^{\text{cf}} \right\rVert_2^2 \ge \tau
$$

其中 $T' = \min(T_{\text{fact}}, T_{\text{cf}})$，$\tau$ 对应 `min_oracle_effect_mse`。

### 4.2 `mean_effect_mse_oracle`

对保留样本，代码统计：

$$
\mathrm{mean\_effect\_mse\_oracle} = \mathbb{E}[\mathrm{effect\_mse}_{\text{oracle}}]
$$

这度量 factual 和 counterfactual 轨迹在物理层面的平均偏离程度。

### 4.3 `pass_change_frac`

$$
\mathrm{pass\_change\_frac} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{I}\left(R^{\text{fact}}_{i,\text{pass}} \neq R^{\text{cf}}_{i,\text{pass}}\right)
$$

它表示筛出的样本里，有多少比例真的导致 pass-door 事件改变。

### 4.4 `effect_pass_acc_given_oracle_change`

这是反事实评测最关键指标。只在 oracle 确认“事件发生变化”的样本上，统计模型是否也预测出变化：

$$
\mathrm{effect\_pass\_acc\_given\_oracle\_change}
=
\frac{1}{|\mathcal{S}_{\Delta}|}
\sum_{i \in \mathcal{S}_{\Delta}}
\mathbb{I}\left(\hat{R}^{\text{fact}}_{i,\text{pass}} \neq \hat{R}^{\text{cf}}_{i,\text{pass}}\right)
$$

其中：

- $\mathcal{S}_{\Delta}$ 是满足 $R^{\text{fact}}_{\text{pass}} \neq R^{\text{cf}}_{\text{pass}}$ 的样本集合
- $\hat{R}$ 是模型侧经 probe + oracle rollout 得到的预测结果

### 4.5 `mean_pair_accuracy`

若 probe 可用，脚本还统计事实/反事实两侧各自是否与 oracle 一致：

$$
\mathrm{mean\_pair\_accuracy} = \frac{1}{2N}\sum_{i=1}^{N}
\left[
\mathbb{I}(\hat{R}^{\text{fact}}_i = R^{\text{fact}}_i) +
\mathbb{I}(\hat{R}^{\text{cf}}_i = R^{\text{cf}}_i)
\right]
$$

这个指标衡量的是“单边分类正确率”，不如 `effect_pass_acc_given_oracle_change` 更能反映干预推理能力。

## 5. 参数提取：来自 config/ 与脚本的关键评测超参数

### 5.1 事实评测参数

来自 `config/eval/factual_eval.yaml`：

- `seed = 42`
- `dataset.name = tworoom`
- `env.img_size = 224`
- `wm.history_size = 1`
- `factual.n_episodes = 200`
- `factual.horizon = 20`
- `factual.open_loop = true`（配置存在，但脚本中未作为条件分支使用）
- `output.filename = logs_eval/factual_results.json`

### 5.2 反事实评测参数

来自 `config/eval/counterfactual_eval.yaml`：

- `seed = 42`
- `policy = random`（默认占位，实际使用时需要替换为 checkpoint 路径）
- `probe_ckpt = /mnt/data/szeluresearch/stable-wm/tworoom/probe.pt`
- `wm.history_size = 1`
- `dataset.name = tworoom`
- `env.img_size = 224`
- `output.filename = logs_eval/cf_results.json`

四类任务的关键参数：

- `do_location.t_intervene = [3, 5, 8, 10, 15]`
- `do_location.loc_prime_offsets_xy`：位置干预偏移集合
- `do_door_y.wall_x_fact = 112`
- `do_door_y.door_y_fact = 54`
- `do_door_y.door_y_cf = [100, 120, 140, 160, 180]`
- `do_wall_x.wall_x_cf = [55, 70, 154, 169]`
- `do_door_y_wall_x.pair_sampling = random_k`
- `do_door_y_wall_x.k_pairs_per_sample = 4`
- `wall_proximity_margin = 22.0`
- `target_effectful_n = 100`
- `max_attempts = 5000`
- `require_oracle_pass_change = true`
- `min_oracle_effect_mse = 5.0`

### 5.3 需要特别说明的“配置字段但未直接参与脚本控制”的参数

以下字段在 config 中存在，但当前 `eval_cf.py` / `cf_sampler.py` 链路里没有直接消费它们作为控制分支：

- `t_intervene`

这不影响当前评测结果的计算，只表示配置层保留了更细粒度的干预时间信息。

## 6. 结论

Two-Room 评测的本质不是单纯的轨迹回归，而是围绕结构变量 $w_x$ 与 $w_y$ 的因果推理测试：

- 事实评测检验模型是否能在真实动作驱动下稳定展开 latent dynamics。
- 反事实评测检验模型是否能在改变墙/门/初始位置后，仍然正确判断“是否穿门”这一离散因果事件。
- 核心指标 `effect_pass_acc_given_oracle_change` 只在 oracle 确认干预真的改变事件时统计，因此最能反映模型是否学到了干预到结果的映射。

如果把 Two-Room 的因果问题压缩成一句话，就是：

> 墙决定能不能过，门决定从哪里过，模型要学的不是“像不像”，而是“干预以后会不会真的变”。