# Counterfactual Results Analysis（中文）

---

## 1. 评测设定（Evaluation Setting）

本实验在 Two-Rooms 环境中，对当前 **Baseline 模型**进行反事实评测，涵盖四类干预任务：

* `do(location)`：在时间步 ( t ) 对 agent 位置进行干预
* `do(door_y)`：改变门的纵向位置
* `do(wall_x)`：改变墙的横向位置
* `do(door_y, wall_x)`：同时改变门与墙（组合结构干预）

所有任务统一采用如下筛选条件：

* `require_effectful: true`
* `require_oracle_pass_change: true`

这意味着评测只保留满足以下条件的样本：

1. 干预在 oracle rollout 中产生**显著影响**
2. 干预导致 **是否穿过门（pass-door）事件发生变化**

因此，本评测：

> **不是一般的 OOD 测试，而是专门评估“干预 → 结果变化（Intervention → Effect）”的因果能力。**

---

## 2. 核心指标说明

| 指标                                        | 含义                           | 越好          |
| ----------------------------------------- | ---------------------------- | ----------- |
| `traj_mse_fact`                           | factual 分支轨迹误差               | ↓           |
| `traj_mse_cf`                             | counterfactual 分支轨迹误差        | ↓           |
| `effect_mse`                              | 干预效应误差（CF - factual）         | ↓           |
| `through_door_acc_cf`                     | CF 分支是否正确预测过门                | ↑           |
| `effect_pass_acc_given_oracle_change`     | 在 oracle 确实发生事件变化的条件下，是否预测正确 | ↑           |
| `oracle_pass_change_rate`                 | oracle 中事件变化比例               | ↑（表示干预更强）   |
| `pred_pass_rate_cf / oracle_pass_rate_cf` | 预测 vs oracle 过门比例            | 越接近越好       |
| `accept_rate_per_attempt`                 | 采样接受率                        | ↓（越难找到有效干预） |

### 最关键指标

> **`effect_pass_acc_given_oracle_change` 是最核心指标**

原因：

* 已经过滤掉“无效干预”
* 只在 **真实发生变化的样本上评估**
* 直接衡量模型是否学到“干预的后果”

---

## 3. 关键结果

### 3.1 核心指标

| Task             | traj_mse_fact | traj_mse_cf | effect_mse | through_door_acc_cf | effect_pass_acc_given_oracle_change | oracle_pass_change_rate |
| ---------------- | ------------: | ----------: | ---------: | ------------------: | ----------------------------------: | ----------------------: |
| do_location      |         1.888 |      19.618 |     20.482 |               0.860 |                               0.586 |                   0.870 |
| do_door_y        |         2.604 |       2.620 |      5.640 |               0.780 |                               0.697 |                   0.990 |
| do_wall_x        |         2.502 |      20.544 |     23.584 |               0.710 |                               0.409 |                   0.930 |
| do_door_y_wall_x |         1.957 |      23.632 |     26.578 |               0.840 |                           **0.233** |                   0.860 |

---

### 3.2 CF 分支事件分布

| Task             | pred_pass_rate_cf | oracle_pass_rate_cf |
| ---------------- | ----------------: | ------------------: |
| do_location      |              0.16 |                0.30 |
| do_door_y        |              0.26 |                0.46 |
| do_wall_x        |              0.06 |                0.35 |
| do_door_y_wall_x |              0.08 |                0.24 |

---

### 3.3 采样难度

| Task             | n_attempts | accept_rate |
| ---------------- | ---------: | ----------: |
| do_location      |        199 |        0.50 |
| do_door_y        |        311 |        0.32 |
| do_wall_x        |       1853 |       0.054 |
| do_door_y_wall_x |       2517 |       0.040 |

---

## 4. 分任务分析

---

### 4.1 `do(location)`

#### 观察

* CF 误差显著升高（1.89 → 19.62）
* `effect_pass_acc_given_oracle_change = 0.586`
* oracle 变化比例高（0.87）

#### 解释

模型在 factual 分支表现正常，但在“改变初始位置”后：

* 无法正确推断后续轨迹
* 无法稳定恢复事件变化

#### 结论

> Baseline 无法建模 **“状态干预 → 未来变化”**

---

### 4.2 `do(door_y)`

#### 观察

* 轨迹误差几乎不变
* effect-level ≈ 0.697
* CF 过门率明显被低估（0.26 vs 0.46）

#### 解释

模型在几何上保持稳定，但：

* 对“门位置变化 → 是否能通过”的逻辑理解不充分

#### 结论

> 模型捕捉了部分结构，但 **事件判定机制不稳定**

---

### 4.3 `do(wall_x)`

#### 观察

* CF 误差大幅上升（20.5）
* effect-level 仅 0.41
* 预测 pass 率极低（0.06 vs 0.35）

#### 解释

墙是核心结构变量：

* 改变约束条件（可通行路径）
* 模型明显无法适应

#### 结论

> 模型未学到 **环境结构 → 动力学约束**

---

### 4.4 `do(door_y, wall_x)`（最关键）

#### 观察

* CF 误差最高（23.6）
* `effect_pass_acc_given_oracle_change = 0.233`
* oracle 变化比例仍然很高（0.86）

#### 解释

这是**组合干预（compositional intervention）**：

* 同时改变两个结构变量
* 环境约束发生耦合变化

模型表现：

* factual 正常
* CF 严重失真
* 几乎无法预测事件变化

#### 结论

> 模型在组合因果推理上几乎完全失败

---

## 5. 关键发现

---

### 5.1 反事实退化是结构性的

统一模式：

* factual MSE 低
* CF MSE 高
* effect-level 下降

说明：

> 模型不是不会预测，而是不会“干预后预测”

---

### 5.2 评测确实在测因果能力

由于筛选条件：

* 所有样本都发生真实事件变化
* 因此结果直接反映：

```text
Intervention → Effect 是否被正确建模
```

---

### 5.3 结构变量显著更难

难度排序：

```text
door_y < location < wall_x < (door_y + wall_x)
```

说明：

* appearance-like 变量较容易
* 结构变量更难
* 组合结构最难

---

### 5.4 组合干预暴露核心问题

关键结果：

* `effect_pass_acc_given_oracle_change = 0.233`

含义：

> 即使 oracle 明确发生变化，模型仍无法预测

这说明：

> 模型没有学习到变量之间的**组合因果关系**

---

## 6. 哪些指标最可信

### 最重要

* `traj_mse_cf`
* `effect_mse`
* `effect_pass_acc_given_oracle_change`
* `oracle_pass_change_rate`

### 次要

* `through_door_acc_cf`
* `pass_rate`

### 不重要（本实验中）

* success 系列（几乎恒为 0）

---

## 7. 总结结论

> 在严格的 effectful counterfactual setting 下，Baseline 在所有干预任务中均表现出明显的反事实退化。模型在 factual 分支保持低误差，但在 counterfactual 分支误差显著放大，并且在 effect-level 指标上表现较差。
>
> 这种退化在结构变量（wall）和组合干预（door + wall）中尤为严重，其中组合干预的 effect-level 准确率仅为 23.3%，表明模型几乎无法正确建模多个变量共同作用下的因果结果变化。
>
> 因此，该评测不仅揭示了模型的 OOD 问题，更直接说明其在 **Intervention → Effect 层面缺乏稳定的因果建模能力**。

---

## 8. 简要总结

* `do_location`：明显退化（状态干预失败）
* `do_door_y`：轨迹稳定，但事件理解不完整
* `do_wall_x`：结构建模失败
* `do_door_y_wall_x`：**组合因果严重失败（23%）**
* 总体：Baseline 未学到可靠的反事实因果机制

