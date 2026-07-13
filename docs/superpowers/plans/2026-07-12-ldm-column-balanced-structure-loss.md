# LDM Column-Balanced Structure Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不依赖低阈值全局增密的前提下，保护 LiDAR target 中真实障碍物 `(X,Y)` 竖列，并抑制空背景竖列，使 LDM 输出保留可用于避障的基础主体、主干和竖向连续结构。

**Architecture:** 在现有一次 VAE decode 后，将 occupancy 沿 Z 聚合为可微的 column logit，对 target 正列和负列分别计算均值 BCE，形成独立的 positive/negative 组件。列存在性由新损失约束，列内部高度、顶部和连续性继续复用已有 structure losses；v10 runner 关闭旧 empty-column voxel density 项，避免重复施加负监督。

**Tech Stack:** Python 3.8、PyTorch、现有 VAE/LDM 多模态训练入口、unittest 风格测试、Bash mini runner。

---

## 1. 可行性与边界

### 当前失效模式

- epoch8 在阈值 `0.80` 时 near recall `0.9026`、trunk `0.7317`，但点数比为 `11.14`。
- epoch8 在质量阈值 `0.98` 时点数比降至 `2.93`，但 near recall `0.6934`、trunk `0.3167`。
- 说明模型不是完全不会生成主体，而是正负列的概率分布重叠，无法通过单一阈值同时获得召回和密度。
- 现有 `decoded_density_precision_loss()` 只在空 target 列内逐体素压制预测；它没有直接奖励真实障碍物列“至少存在”，容易与 top/IR-negative 项共同造成结构下压。

### 不采用的第一轮方案

1. **新增 BEV/column prediction head**：表达能力更强，但会改变网络结构和 checkpoint 协议，第一轮不采用。
2. **直接使用全图 BEV Dice/Focal**：类别比例敏感，难以独立观察正列召回与负列误报，不利于定位权重失衡。
3. **thermal edge 或预训练 IR backbone**：真实 IR 消融已经证明当前 IR 有效，不是最直接瓶颈。

### 推荐数学形式

设 occupancy voxel logit 为 `l_z`，Z 层数为 `Z`，温度为 `tau`：

```text
column_logit = tau * (logsumexp(l_z / tau, dim=Z) - log(Z))
target_column = any(target_occ >= target_threshold, dim=Z)
L_col_pos = mean(softplus(-column_logit[target_column]))
L_col_neg = mean(softplus( column_logit[~target_column]))
L_column = w_pos * L_col_pos + w_neg * L_col_neg
```

`logmeanexp` 比 `max` 能向多个 Z 层传播梯度，又比 noisy-OR 更不容易因为 64 层低概率累积而饱和。正负类分别取均值，避免数量极多的空背景列淹没真实障碍物列。

---

## 2. 文件与职责

- Modify: `diffusion_consistency_radar/scripts/unified_train.py`
  - 新增纯函数 `decoded_column_balanced_losses()`。
  - 接入 LDM 总损失、metrics、trainer config 和 checkpoint metadata。
- Modify: `test/unit/test_ldm_vertical_structure_loss.py`
  - 新增列级损失的数学、梯度、边界和单次 decode 测试。
- Modify: `test/mini-test/train_minimal.sh`
  - 传递三个 v10 配置：正列权重、负列权重、聚合温度。
- Modify: `test/unit/test_mini_train_script.py`
  - 验证环境变量、YAML、日志与旧默认兼容。
- Modify: `test/mini-test/run_ldm_vertical_experiment.sh`
  - 向通用实验 runner 透传列级参数。
- Create: `test/mini-test/run_ldm_z64_v10_column_experiment.sh`
  - 固定 Z64、500 帧、v9-A 其余权重和安全输出协议。
- Modify: `test/evaluation/ldm/select_ldm_checkpoint.py`
  - 不改排序数学，仅在报告中继续使用固定结构门槛。
- Modify: `TODO/findings.md`, `TODO/task_plan.md`, `TODO/progress.md`
  - 记录监督信号、实验结果和 CD gate。

---

### Task 1: Column Loss TDD Core

**Files:**
- Modify: `test/unit/test_ldm_vertical_structure_loss.py`
- Modify: `diffusion_consistency_radar/scripts/unified_train.py`

- [ ] **Step 1: 写 matched/missing/false-positive 的失败测试**

```python
def test_column_balanced_loss_prefers_matched_columns():
    target = torch.zeros(1, 1, 4, 2, 2)
    target[:, :, 1:3, 0, 0] = 1.0
    matched = torch.full_like(target, -8.0)
    matched[:, :, 1:3, 0, 0] = 8.0
    missing = torch.full_like(target, -8.0)
    false_positive = matched.clone()
    false_positive[:, :, 1:3, 1, 1] = 8.0

    matched_losses = decoded_column_balanced_losses(matched, target, "sigmoid")
    missing_losses = decoded_column_balanced_losses(missing, target, "sigmoid")
    fp_losses = decoded_column_balanced_losses(false_positive, target, "sigmoid")

    assert matched_losses["positive_loss"] < missing_losses["positive_loss"]
    assert matched_losses["negative_loss"] < fp_losses["negative_loss"]
```

- [ ] **Step 2: 运行 RED 测试**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_ldm_vertical_structure_loss.py -v
```

Expected: FAIL，提示 `decoded_column_balanced_losses` 不存在。

- [ ] **Step 3: 实现输入校验、logmeanexp 聚合和正负 BCE**

在 `unified_train.py` 中新增：

```python
def decoded_column_balanced_losses(
    decoded: torch.Tensor,
    target: torch.Tensor,
    occupancy_activation: str,
    temperature: float = 1.0,
    target_threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    _validate_decoded_occupancy_inputs(decoded, target, occupancy_activation)
    if not math.isfinite(float(temperature)) or temperature <= 0.0:
        raise ValueError("column temperature 必须是有限正数")
    if not 0.0 <= float(target_threshold) <= 1.0:
        raise ValueError("column target_threshold 必须位于 [0,1]")

    raw = decoded[:, 0:1].float()
    if occupancy_activation == "sigmoid":
        logits = raw
    else:
        probability = raw.clamp(1e-6, 1.0 - 1e-6)
        logits = torch.logit(probability)

    z_size = logits.shape[2]
    column_logits = temperature * (
        torch.logsumexp(logits / temperature, dim=2)
        - math.log(float(z_size))
    )
    target_columns = (
        target[:, 0:1].float().clamp(0.0, 1.0) >= target_threshold
    ).any(dim=2)
    graph_zero = raw.sum() * 0.0
    positive_loss = (
        torch.nn.functional.softplus(-column_logits)[target_columns].mean()
        if target_columns.any() else graph_zero
    )
    negative_columns = ~target_columns
    negative_loss = (
        torch.nn.functional.softplus(column_logits)[negative_columns].mean()
        if negative_columns.any() else graph_zero
    )
    return {
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
    }
```

- [ ] **Step 4: 增加数值与梯度边界测试**

覆盖：

```text
sigmoid 极端 logits 仍 finite 且有梯度
raw 0/1 概率经过 clamp 后 finite
全空 target 时 positive_loss 为 graph-connected zero
全正列 target 时 negative_loss 为 graph-connected zero
正负列数量改变不会改变各自 class mean 的尺度
soft target 低于 0.5 不被当作正列
temperature <= 0、NaN、Inf 被拒绝
B/Z/X/Y 或 activation 不合法沿用共享校验
```

- [ ] **Step 5: 运行 GREEN 测试**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_ldm_vertical_structure_loss.py -v
```

Expected: 所有新增和历史结构损失测试通过。

---

### Task 2: Integrate Components Without Extra VAE Decode

**Files:**
- Modify: `diffusion_consistency_radar/scripts/unified_train.py`
- Modify: `test/unit/test_ldm_vertical_structure_loss.py`

- [ ] **Step 1: 写总损失失败测试**

新增测试验证：

```python
total, components = compute_ldm_loss_components(
    denoised,
    z_target,
    target,
    vae,
    occupancy_activation="sigmoid",
    decoded_column_positive_weight=0.02,
    decoded_column_negative_weight=0.01,
    decoded_column_temperature=1.0,
    # 其余现有权重显式传入
)
assert vae.decode_calls == 1
assert "column_positive_loss" in components
assert "column_negative_loss" in components
```

并验证两个新权重均为 0 时总损失与当前实现逐值一致。

- [ ] **Step 2: 运行 RED 测试**

Expected: 新参数或新 components 不存在而失败。

- [ ] **Step 3: 扩展损失签名与日志组件**

修改：

```python
LDM_LOSS_COMPONENT_NAMES = (
    # existing names...
    "column_positive_loss",
    "column_negative_loss",
    "uncertainty_loss",
)
```

向 `compute_ldm_loss_components()` 增加：

```python
decoded_column_positive_weight: float = 0.0,
decoded_column_negative_weight: float = 0.0,
decoded_column_temperature: float = 1.0,
```

仅当任一列级权重大于 0 时调用一次 `decoded_column_balanced_losses()`，并把加权项加入已有 `loss`。复用 `decoded = vae.decode(denoised)`，不得新增第二次 decode。

- [ ] **Step 4: 扩展 trainer 与 checkpoint metadata**

在 `OptimizedLDMTrainer.__init__`、`_ldm_loss_config()` 和 `train_epoch()` 中接入：

```yaml
decoded_column_positive_weight: 0.0
decoded_column_negative_weight: 0.0
decoded_column_temperature: 1.0
```

默认权重必须为 0，保证旧 config 和旧训练行为不变。metrics 记录 raw component，不把权重乘入 CSV component 值，延续现有日志语义。

- [ ] **Step 5: 运行聚焦回归**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_ldm_vertical_structure_loss.py -v
conda run -n Radar-Diffusion python -m py_compile diffusion_consistency_radar/scripts/unified_train.py
```

---

### Task 3: Mini Config and Guarded v10 Runner

**Files:**
- Modify: `test/mini-test/train_minimal.sh`
- Modify: `test/mini-test/run_ldm_vertical_experiment.sh`
- Modify: `test/unit/test_mini_train_script.py`
- Create: `test/mini-test/run_ldm_z64_v10_column_experiment.sh`

- [ ] **Step 1: 写 shell contract 失败测试**

测试必须断言：

```text
MINI_LDM_COLUMN_POSITIVE_WEIGHT 默认 0.0
MINI_LDM_COLUMN_NEGATIVE_WEIGHT 默认 0.0
MINI_LDM_COLUMN_TEMPERATURE 默认 1.0
三个值写入 ldm YAML 对应字段
v10 runner 使用 Z64、500 samples、split seed 42
v10 runner 把旧 decoded_density_weight 设为 0.0
v10 runner 不启动 inference、ablation 或 CD
v10 runner 保留输出目录/锁/symlink 防护
```

- [ ] **Step 2: 运行 RED 测试**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_mini_train_script.py -v
```

- [ ] **Step 3: 扩展 mini YAML 生成**

在 `train_minimal.sh` 中读取并写入：

```bash
MINI_LDM_COLUMN_POSITIVE_WEIGHT="${MINI_LDM_COLUMN_POSITIVE_WEIGHT:-0.0}"
MINI_LDM_COLUMN_NEGATIVE_WEIGHT="${MINI_LDM_COLUMN_NEGATIVE_WEIGHT:-0.0}"
MINI_LDM_COLUMN_TEMPERATURE="${MINI_LDM_COLUMN_TEMPERATURE:-1.0}"
```

- [ ] **Step 4: 新增 v10 runner**

v10 第一轮只定义两个短筛选变体：

```text
v10-A: column positive=0.02, negative=0.01
v10-B: column positive=0.02, negative=0.02
```

共同配置：

```text
沿用 v9-A 的 decoded/height/top/top-overshoot/continuity/IR 权重
decoded_density_weight=0.0，避免与 column-negative 重复
column temperature=1.0
500 samples，3 epochs，Z64，seed 42
只训练 LDM；VAE 使用已通过上界检查的同一 checkpoint
```

- [ ] **Step 5: 运行 GREEN 与 shell 检查**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_mini_train_script.py -v
bash -n test/mini-test/run_ldm_z64_v10_column_experiment.sh
git diff --check
```

---

### Task 4: Finite-Gradient Smoke and Component Calibration

**Files:**
- Result only: `test/result/ldm/ablation/ldm_v10_column_smoke/`
- Modify after result: `TODO/progress.md`, `TODO/findings.md`

- [ ] **Step 1: 运行 1 sample、1 epoch smoke**

Run:

```bash
MINI_DATASET_DIR=/tmp/radar_v10_column_smoke/data \
MINI_CONFIG_PATH=/tmp/radar_v10_column_smoke/config.yaml \
MINI_RESULTS_DIR=/tmp/radar_v10_column_smoke \
SAMPLES_PER_SCENE=1 \
MINI_LDM_EPOCHS=1 \
MINI_NUM_WORKERS=0 \
MINI_LDM_COLUMN_POSITIVE_WEIGHT=0.02 \
MINI_LDM_COLUMN_NEGATIVE_WEIGHT=0.01 \
bash test/mini-test/train_minimal.sh ldm
```

Expected:

```text
loss、column_positive_loss、column_negative_loss 全部 finite
两个 component 均写入 metrics.csv
模型参数存在非零有限梯度
输出只位于 /tmp/radar_v10_column_smoke
```

- [ ] **Step 2: 检查 component 量级**

若加权列级项超过 latent loss 的 25%，不直接长训；保持数学形式不变，只把两个权重按相同比例缩小。不得依据单帧结果分别调两个权重。

---

### Task 5: Three-Epoch Isolated Screen

**Files:**
- Result: `test/result/ldm/ablation/ldm_near40_500_z64_v10a_column_screen/`
- Result: `test/result/ldm/ablation/ldm_near40_500_z64_v10b_column_screen/`

- [ ] **Step 1: 运行 v10-A 与 v10-B 各 3 epoch**

不得并行训练两个 GPU 实验，避免显存和吞吐互相影响。每个 runner 训练后只运行固定 32 帧 real-IR validation，不运行 500 帧正式推理。

- [ ] **Step 2: 选择短筛选胜者**

按以下顺序判断：

```text
1. 排除 NaN、空/全满输出或 count ratio > 6 的候选
2. near BEV recall 更高
3. trunk recall 更高
4. BEV IoU 更高
5. top 和 vertical connectivity 不得同时低于 v9-A screen
```

只允许一个胜者进入 10 epoch；若二者均被排除，停止并重新检查损失温度/权重，不启动第三个盲试实验。

---

### Task 6: Full v10 Winner and Fixed Evaluation Gate

**Files:**
- Result: `test/result/ldm/ablation/ldm_near40_500_z64_v10a_column_full/`
- Reuse: `test/mini-test/run_ldm_z64_checkpoint_selection.sh`
- Reuse: `diffusion_consistency_radar/scripts/sweep_occ_threshold.py`
- Reuse: `test/evaluation/ldm/evaluate_ldm_vertical_structure.py`
- Reuse: `test/visualization/generate_interactive_inference_compare.py`

- [ ] **Step 1: 训练唯一胜者 10 epoch**

保持 VAE、数据、Z64 网格、seed、训练帧和除列级损失外的配置固定。

- [ ] **Step 2: 按固定 32 帧结构指标选择 epoch checkpoint**

不得再按最低训练 loss 直接选教师。

- [ ] **Step 3: 仅对选中 checkpoint 做 500 帧 40-step Heun 推理**

- [ ] **Step 4: 同时报告 quality 和 safety 阈值**

```text
quality point: validation BEV-F1 最优
safety point: global BEV recall >= 0.80 且 0-20m recall >= 0.90
```

- [ ] **Step 5: 执行量化 gate**

进入 CD 前必须同时满足：

```text
safety point pred/target ratio <= 6.0
safety point 0-20m BEV recall >= 0.90
quality point full BEV IoU >= 0.2548
top-height recall >= 0.10
trunk recall >= 0.65
vertical connectivity >= 0.60
输出非空且非全满
```

- [ ] **Step 6: 执行基础结构可视化 gate**

从固定 validation IDs 生成 10 帧 raw LiDAR / Radar / prediction 交互 3D 图。禁止使用 target voxel 代替 raw LiDAR。至少 `8/10` 帧应满足：

```text
主要障碍物横向位置可辨认
低处主体或树干存在
Z 向不是单层散点，也没有明显贯穿天空的过冲柱
空背景没有形成与障碍物同密度的大面积伪结构
```

此 gate 只保证“基础障碍物结构”，不宣称恢复 LiDAR 级枝叶细节。

- [ ] **Step 7: CD 决策**

量化和可视化 gate 全部通过才训练 CD；任一失败则保持 HOLD，并从 positive/negative component 和错误列分布定位下一轮，不更换 IR backbone。

---

## 3. 影响说明

- **监督信号：** 新增 LiDAR target 的二维 column existence 正负监督；不修改 target 文件本身。
- **体素数量：** target 占用体素数量、Z64 网格和物理范围完全不变；预测点数预计通过负列项下降，同时正列项避免 trunk/主体随之消失。
- **训练开销：** 复用一次 decoded voxel，新增沿 Z 的 `logsumexp` 和二维 BCE，显存增量应远小于新增网络 head。
- **指标影响：** 目标是把 safety point 点数比从约 `11` 压到 `<=6`，同时保持 near recall `>=0.90` 和 trunk `>=0.65`；quality point 不能低于 v8 BEV IoU `0.2548`。
- **兼容性：** 新权重默认 0，旧 config/checkpoint 可继续加载；metrics 表头变化沿用现有 legacy archive 机制。
- **风险：** column-positive 只保证列存在，不能独立恢复列内形状，因此必须保留 top/continuity/height losses 和 raw-LiDAR 可视化 gate。

## 4. 完整验证命令

```bash
conda run -n Radar-Diffusion python test/unit/test_ldm_vertical_structure_loss.py -v
conda run -n Radar-Diffusion python test/unit/test_mini_train_script.py -v
conda run -n Radar-Diffusion python test/unit/test_ir_condition_ablation.py -v
conda run -n Radar-Diffusion python test/unit/test_ldm_checkpoint_selection.py -v
conda run -n Radar-Diffusion python -m py_compile \
  diffusion_consistency_radar/scripts/unified_train.py \
  test/evaluation/ldm/select_ldm_checkpoint.py
bash -n test/mini-test/train_minimal.sh
bash -n test/mini-test/run_ldm_vertical_experiment.sh
bash -n test/mini-test/run_ldm_z64_v10_column_experiment.sh
git diff --check
```
