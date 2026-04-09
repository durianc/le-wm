# tworoom_eval 集成与使用说明

## 1. 包结构与文件职责

`tworoom_eval` 的代码位于仓库根目录下的 [tworoom_eval/](tworoom_eval) 目录中。它不是一组散落在仓库根目录的脚本，而是一个可通过 `python -m ...` 方式启动的包。

### 代码文件

- [tworoom_eval/__init__.py](tworoom_eval/__init__.py)
  - 包入口声明文件，使 `tworoom_eval` 成为可导入的 Python 包。
  - 统一导出 TwoRoom counterfactual 环境注册函数和环境类。

- [tworoom_eval/__main__.py](tworoom_eval/__main__.py)
  - 包级入口文件。
  - 支持 `python -m tworoom_eval` 这种启动方式时的默认行为。

- [tworoom_eval/cf_env.py](tworoom_eval/cf_env.py)
  - 定义 `TwoRoomCFEnv`，在原始 TwoRoom 环境基础上支持结构性干预。
  - 支持 `reset(options=...)` 传入以下干预参数：`wall_x`、`door_y`、`agent_pos`。
  - 提供 `register_cf_env()`，将环境注册为 `lewm/TwoRoomCF-v0`。

- [tworoom_eval/cf_oracle.py](tworoom_eval/cf_oracle.py)
  - 提供纯物理 oracle rollout。
  - 用于在 factual / counterfactual 参数下执行真实环境回放，返回轨迹与 `through_door` 事件。

- [tworoom_eval/cf_sampler.py](tworoom_eval/cf_sampler.py)
  - 负责采样 effectful 的 factual / counterfactual 样本对。
  - 通过读取 HDF5 数据集中的 episode 索引，构造干预参数并调用 oracle rollout。
  - 产出 `pass_changed`、`effect_mse_oracle` 等评估需要的中间结果。

- [tworoom_eval/probe.py](tworoom_eval/probe.py)
  - 提供 `MLPProbe`，用于从 JEPA encoder embedding 回归 `agent_pos`。
  - 同时包含 probe 训练、embedding 提取、模型保存等逻辑。
  - 该文件保存的 probe checkpoint 会被 `eval_cf.py` 加载。

- [tworoom_eval/eval_factual.py](tworoom_eval/eval_factual.py)
  - factual 评估入口。
  - 读取 TwoRoom HDF5 数据集，重放 episode，比较模型预测 embedding 与真实 embedding 的 MSE。
  - 输出逐步 MSE、总 MSE、漂移斜率等指标。

- [tworoom_eval/eval_cf.py](tworoom_eval/eval_cf.py)
  - counterfactual 评估入口。
  - 串联样本采样、环境重置、世界模型预测、probe 预测与 oracle 判定。
  - 输出 `effect_pass_acc_given_oracle_change`、`pass_change_frac`、`mean_pair_accuracy` 等指标。

- [tworoom_eval/utils.py](tworoom_eval/utils.py)
  - 提供通用工具函数。
  - 其中最关键的是模型名处理与结果文件路径处理：`add_model_suffix()`、`resolve_model_artifact_path()`。
  - 还包含训练时使用的 `ModelObjectCallBack`。

### 配置文件

- [tworoom_eval/config/eval/factual_eval.yaml](tworoom_eval/config/eval/factual_eval.yaml)
  - factual 评估配置。
  - 包含数据集名、模型 policy、评估 episode 数、horizon、图像尺寸等参数。

- [tworoom_eval/config/eval/counterfactual_eval.yaml](tworoom_eval/config/eval/counterfactual_eval.yaml)
  - counterfactual 评估配置。
  - 包含四类干预任务的参数定义：`do_location`、`do_door_y`、`do_wall_x`、`do_door_y_wall_x`。
  - 还包括 probe checkpoint 路径、效果性样本筛选阈值、每项任务目标样本数和最大尝试次数。

### 相关资源

- `tworoom_eval/probe.pt`
  - 训练好的 linear probe 权重文件。
  - 用于 counterfactual 评估中把 embedding 映射回 `agent_pos`。

- `$STABLEWM_HOME/tworoom/lewm/`
  - TwoRoom world model 的推荐放置位置。
  - 这里的 `lewm` 目录内容对应 Hugging Face 上 `quentinll/lewm-tworooms` 仓库中的模型文件，落盘时需要放在 `tworoom/lewm/` 子目录下。
  - 对于本文档中的默认示例为：`/mnt/data/szeluresearch/stable-wm/tworoom/lewm/weights.pt`。代码会自动解析 `weights.pt` 文件来加载模型。

## 2. 核心逻辑链路

`tworoom_eval` 的逻辑链路可以理解为“环境注册 - 数据读取 - 真实 oracle - 世界模型推理 - 指标汇总”这条链。

### factual 评估链路

1. `eval_factual.py` 启动后，先调用 `register_cf_env()`，确保 `lewm/TwoRoomCF-v0` 已在 Gymnasium 中注册。
2. 通过 `stable_worldmodel` 加载世界模型 `policy`，支持直接从 run name 或 `weights.pt` 反序列化加载。
3. 从 `$STABLEWM_HOME/tworoom.h5` 读取 episode 索引、`proprio` 和 `action` 数据。
4. 对每个 episode：
   - 用 `proprio[0]` 取初始位置。
   - 重置 `TwoRoomCFEnv` 并渲染初始帧。
   - 对真实帧做预处理后送入世界模型 encoder。
   - 逐步使用 `model.predict()` 做自回归 unroll，并与每步真实帧的 embedding 比较。
5. 汇总得到 factual 评估指标：`mean_mse_per_step`、`mean_total_mse`、`drift_slope` 等。
6. 最后将结果写入 `cfg.output.filename` 指定的 JSON 文件，并按模型名自动加后缀。

### counterfactual 评估链路

1. `eval_cf.py` 启动后，同样先执行 `register_cf_env()`，保证环境可创建。
2. 通过 `stable_worldmodel` 加载世界模型，并加载 `probe.pt` 作为冻结线性探针。
3. 对 `counterfactual_eval.yaml` 中定义的四个任务逐一执行：
   - `do_location`
   - `do_door_y`
   - `do_wall_x`
   - `do_door_y_wall_x`
4. 每个任务先调用 `collect_effectful_samples()`：
   - 从 HDF5 中随机抽 episode。
   - 构造 factual / counterfactual 参数。
   - 用 `oracle_rollout()` 跑出真实轨迹。
   - 只保留 effectful 样本对。
5. 对每个 effectful 样本对：
   - 渲染 factual / counterfactual 初始帧。
   - 用世界模型 `encode()` 得到 embedding。
   - 用 probe 把 embedding 回归成 `agent_pos`。
   - 再交给 `oracle_rollout()` 判断 `through_door`。
6. 汇总得到核心指标：
   - `effect_pass_acc_given_oracle_change`
   - `pass_change_frac`
   - `mean_effect_mse_oracle`
   - `mean_pair_accuracy`
7. 最后将结果输出到 JSON 文件。

### 依赖关系总结

- `cf_env.py` 是所有评估入口的基础。
- `cf_oracle.py` 和 `cf_sampler.py` 负责生成“ground truth 干预效果”。
- `probe.py` 负责把世界模型 embedding 映射回几何状态。
- `eval_factual.py` 关注 world model 的时序预测误差。
- `eval_cf.py` 关注“干预是否导致事件变化”的因果推理能力。

## 3. 使用指南

### 环境准备

确保以下依赖已经可用：

- Python 3.10
- `stable-worldmodel`
- `stable-pretraining`
- `hydra`、`gymnasium`、`torch`、`h5py`、`numpy`

建议先激活项目环境，并确保数据集文件存在于 `$STABLEWM_HOME` 下：

```bash
export STABLEWM_HOME=/path/to/your/storage
```
eg.
```bash
export STABLEWM_HOME=/mnt/data/szeluresearch/stable-wm
```

TwoRoom 数据集应位于：

```bash
$STABLEWM_HOME/tworoom.h5
```

probe 权重默认使用：

```bash
tworoom_eval/probe.pt
```

如果你把 probe 放到别的位置，可以通过 `probe_ckpt=` 覆盖。

### 3.1 运行 factual 评估

```bash
python -m tworoom_eval.eval_factual policy=tworoom/lejepa
```

这条命令会：

- 加载 world model `tworoom/lejepa`
- 读取 `tworoom.h5`
- 对若干 factual episodes 执行 autoregressive unroll
- 输出 embedding MSE 相关指标

常见输出内容包括：

- `mean_mse_per_step`
- `std_mse_per_step`
- `mean_total_mse`
- `drift_slope`

### 3.2 运行 counterfactual 评估

```bash
python -m tworoom_eval.eval_cf policy=tworoom/lejepa probe_ckpt=tworoom_eval/probe.pt
```

这条命令会：

- 加载 world model `tworoom/lejepa`
- 加载 probe 权重 `tworoom_eval/probe.pt`
- 针对四个干预任务采样 effectful 样本
- 评估模型对 `through_door` 变化的预测能力

常见输出内容包括：

- `effect_pass_acc_given_oracle_change`
- `pass_change_frac`
- `mean_effect_mse_oracle`
- `mean_pair_accuracy`

### 3.3 输出文件

评估结果会写入配置里的 `output.filename`。当前默认是：

- [tworoom_eval/config/eval/counterfactual_eval.yaml](tworoom_eval/config/eval/counterfactual_eval.yaml)
- [tworoom_eval/config/eval/factual_eval.yaml](tworoom_eval/config/eval/factual_eval.yaml)

如果需要按模型自动区分输出文件名，脚本会通过 [tworoom_eval/utils.py](tworoom_eval/utils.py) 中的 `add_model_suffix()` 自动追加模型标识。



## 运行示例

```bash
python -m tworoom_eval.eval_factual policy=tworoom/lewm
python -m tworoom_eval.eval_cf policy=tworoom/lewm probe_ckpt=tworoom_eval/probe.pt
```

如果需要指定不同的输出路径或 probe 路径，可以通过 Hydra override 继续传参。
