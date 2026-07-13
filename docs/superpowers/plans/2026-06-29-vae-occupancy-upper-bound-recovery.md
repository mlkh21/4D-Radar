# VAE Occupancy Upper-Bound Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将近场 `0-40m` VAE 的 occupancy 重建 IoU 从当前 `0.3177` 提升到可支撑树木和障碍物结构生成的水平，并在达到验收门槛后再训练 LDM/CD。

**Architecture:** 保留现有四通道输入和近场体素协议，先把 occupancy 从普通回归改为显式的稀疏二分类监督，再比较 ultra-lightweight 与 lightweight VAE 的确定性重建上界。训练选择依据从总 MSE 改为验证集 occupancy IoU，checkpoint 保存完整模型配置，确保训练、诊断和推理使用同一结构与同一 occupancy 激活方式。

**Tech Stack:** Python 3.8、PyTorch、NumPy、YAML、现有 `VAE3D`/`NTU4DRadLM_VoxelDataset`/mini shell pipeline。

---

## 当前证据与决策

- 500 帧近场检查的最佳阈值为 `0.4`，IoU/Recall/Precision 为 `0.3177/0.4360/0.5393`。
- 单帧 IoU 中位数约 `0.3261`，不是少数异常帧拉低均值。
- 目标 occupancy 平均每帧约 `555` 个体素，占 `32*128*128` 网格的 `0.106%`。
- 当前 `occupied_weight=8` 无法对冲约 `1:943` 的正负样本不平衡。
- 训练损失从 epoch 1 的 `0.4050` 持续降至 epoch 10 的 `0.1320`，尚未收敛。
- `kl_weight=1e-6` 对总损失贡献约百万分之一，不是当前首要瓶颈。
- 当前 `ultra_lightweight` 仅 `base_channels=16`、`latent_dim=4`，适合通路 smoke，不适合作为树木细结构的最终 VAE。
- 因此修复顺序固定为：**指标一致性 -> occupancy 专用损失 -> 小样本过拟合 -> 容量对照 -> 500 帧验证 -> LDM/CD**。

## 文件结构

- 修改 `diffusion_consistency_radar/cm/vae_3d.py`：增加 occupancy logits/probability 约定及 BCE+Dice 稀疏占用损失。
- 修改 `diffusion_consistency_radar/scripts/unified_train.py`：记录验证 occupancy 指标，按验证 IoU 保存最佳模型，并把 VAE 配置写入 checkpoint。
- 修改 `diffusion_consistency_radar/scripts/diagnose_vae_reconstruction.py`：从 checkpoint 自动恢复 VAE 配置，统一对 occupancy logits 做 sigmoid，并输出分距离指标。
- 修改 `diffusion_consistency_radar/scripts/inference.py`：从 checkpoint 恢复 VAE 配置，避免固定创建 ultra-lightweight 模型。
- 修改 `test/mini-test/train_minimal.sh`：增加 VAE 架构、epoch、验证划分和损失参数的环境变量。
- 新建 `test/unit/test_vae_sparse_occupancy_loss.py`：验证 BCE+Dice、稀疏梯度和激活协议。
- 扩展 `test/unit/test_vae_reconstruction_diagnostic.py`：验证 checkpoint 配置恢复和分距离汇总。

### Task 1: 统一 occupancy 输出与诊断口径

**Files:**
- Modify: `diffusion_consistency_radar/cm/vae_3d.py`
- Modify: `diffusion_consistency_radar/scripts/diagnose_vae_reconstruction.py`
- Test: `test/unit/test_vae_sparse_occupancy_loss.py`
- Test: `test/unit/test_vae_reconstruction_diagnostic.py`

- [ ] **Step 1: 写失败测试，固定 occupancy 通道语义**

测试必须验证：

```python
logits = torch.tensor([[-2.0, 0.0, 2.0]])
prob = VAE3D.occupancy_probability(logits)
torch.testing.assert_close(prob, torch.sigmoid(logits))
```

同时验证非 occupancy 通道不经过 sigmoid，仍保持连续值回归语义。

- [ ] **Step 2: 运行测试并确认失败**

```bash
conda run -n Radar-Diffusion python test/unit/test_vae_sparse_occupancy_loss.py -v
```

预期：因 `occupancy_probability()` 尚不存在而失败。

- [ ] **Step 3: 增加统一概率转换接口**

在 `VAE3D` 中增加：

```python
@staticmethod
def occupancy_probability(logits: torch.Tensor) -> torch.Tensor:
    """将 occupancy logits 转为概率，供训练指标、诊断和推理共用。"""
    return torch.sigmoid(logits)
```

模型解码器继续输出 logits；只有计算 occupancy 指标、保存 occupancy 结果和地图阈值化时调用 sigmoid。

- [ ] **Step 4: 修改诊断脚本**

将：

```python
recon_occ_score = recon[:, 0].detach().cpu().numpy()
```

改为：

```python
recon_occ_score = model.occupancy_probability(recon[:, 0]).detach().cpu().numpy()
```

阈值扫描改为 `0.05,0.1,...,0.95`，避免继续用 raw logits 阈值制造不可比较结果。

- [ ] **Step 5: 运行测试**

```bash
conda run -n Radar-Diffusion python test/unit/test_vae_sparse_occupancy_loss.py -v
conda run -n Radar-Diffusion python test/unit/test_vae_reconstruction_diagnostic.py -v
```

预期：全部通过。

### Task 2: 用 BCE+Dice 替换 occupancy 普通 MSE

**Files:**
- Modify: `diffusion_consistency_radar/cm/vae_3d.py`
- Modify: `diffusion_consistency_radar/scripts/unified_train.py`
- Test: `test/unit/test_vae_sparse_occupancy_loss.py`

- [ ] **Step 1: 写失败测试**

覆盖以下行为：

```python
target = torch.zeros(1, 1, 4, 4, 4)
target[..., 1, 1, 1] = 1.0
good_logits = torch.full_like(target, -6.0)
good_logits[..., 1, 1, 1] = 6.0
empty_logits = torch.full_like(target, -6.0)
assert sparse_occupancy_loss(good_logits, target) < sparse_occupancy_loss(empty_logits, target)
```

再验证只有一个正体素时，正体素位置仍能获得非零且方向正确的梯度。

- [ ] **Step 2: 实现独立 occupancy 损失**

在 `VAE3D` 增加配置：

```python
occupancy_bce_weight: float = 1.0
occupancy_dice_weight: float = 1.0
occupancy_pos_weight_cap: float = 128.0
continuous_recon_weight: float = 1.0
```

每个 batch 根据正负体素数计算：

```python
pos_weight = (negative_count / positive_count.clamp_min(1.0)).clamp(
    max=self.occupancy_pos_weight_cap
)
bce = F.binary_cross_entropy_with_logits(
    x_recon[:, 0:1],
    (x[:, 0:1] > 0.5).float(),
    pos_weight=pos_weight,
)
prob = torch.sigmoid(x_recon[:, 0:1])
dice = 1.0 - (2.0 * (prob * target_occ).sum() + 1.0) / (
    prob.sum() + target_occ.sum() + 1.0
)
```

通道 1-3 只在目标 occupancy 或 Doppler-valid mask 内计算 Smooth L1，防止海量无效背景主导连续通道损失。旧 MSE 路径保留为 `occupancy_loss_type: legacy_mse`，仅用于回归对照。

- [ ] **Step 3: 扩展训练日志**

`metrics.csv` 增加：

```text
occ_bce_loss,occ_dice_loss,continuous_loss,val_iou,val_recall,val_precision
```

总损失继续包含 KL，但第一轮实验固定 `kl_weight=1e-6`。

- [ ] **Step 4: 运行单元测试和编译检查**

```bash
conda run -n Radar-Diffusion python test/unit/test_vae_sparse_occupancy_loss.py -v
conda run -n Radar-Diffusion python -m py_compile \
  diffusion_consistency_radar/cm/vae_3d.py \
  diffusion_consistency_radar/scripts/unified_train.py
```

预期：全部通过。

### Task 3: 增加验证集 IoU 与 checkpoint 自描述

**Files:**
- Modify: `diffusion_consistency_radar/scripts/unified_train.py`
- Modify: `diffusion_consistency_radar/scripts/diagnose_vae_reconstruction.py`
- Modify: `diffusion_consistency_radar/scripts/inference.py`
- Test: `test/unit/test_vae_reconstruction_diagnostic.py`

- [ ] **Step 1: 写失败测试**

构造带以下字段的临时 checkpoint：

```python
{
    "model_state_dict": model.state_dict(),
    "vae_config": {
        "config_type": "lightweight",
        "latent_dim": 8,
        "occupancy_loss_type": "bce_dice",
    },
    "occupancy_activation": "sigmoid",
}
```

验证诊断与推理加载器自动选择同一结构，不再要求人工传 `--config_type`。

- [ ] **Step 2: 划分训练/验证样本**

使用确定性索引划分，默认 `train_split=0.8`；训练集只用于反向传播，验证集每个 epoch 计算 threshold `0.5` 的 IoU/Recall/Precision。

- [ ] **Step 3: 修改最佳模型选择**

保存两个 checkpoint：

```text
vae_best_loss.pt
vae_best_iou.pt
```

LDM/CD 默认使用 `vae_best_iou.pt`。每个 checkpoint 写入 `vae_config`、`data_grid_config` 和 `occupancy_activation`。

- [ ] **Step 4: 修改诊断与推理加载**

加载顺序：

1. checkpoint 中存在 `vae_config` 时按其构建；
2. 历史 checkpoint 没有配置时使用 CLI `--config_type`；
3. key/shape 不匹配时输出明确错误，不静默回退到 ultra-lightweight。

- [ ] **Step 5: 运行测试**

```bash
conda run -n Radar-Diffusion python test/unit/test_vae_reconstruction_diagnostic.py -v
conda run -n Radar-Diffusion python test/unit/test_multimodal_inference_interface.py
```

预期：新旧 checkpoint 路径均通过。

### Task 4: 先做 32 帧可逆性过拟合

**Files:**
- Modify: `test/mini-test/train_minimal.sh`
- Output: `test/result/vae/overfit/vae_overfit_32/`

- [ ] **Step 1: 暴露 mini VAE 配置**

新增环境变量：

```bash
MINI_VAE_CONFIG_TYPE="${MINI_VAE_CONFIG_TYPE:-lightweight}"
MINI_VAE_LATENT_DIM="${MINI_VAE_LATENT_DIM:-8}"
MINI_VAE_OCC_LOSS="${MINI_VAE_OCC_LOSS:-bce_dice}"
MINI_VAE_PATIENCE="${MINI_VAE_PATIENCE:-10}"
```

默认训练网格保持 `(32,128,128)` 和 `0-40m`，本阶段不同时改变空间分辨率。

- [ ] **Step 2: 运行 32 帧过拟合**

```bash
SAMPLES_PER_SCENE=32 \
TRAIN_SCENES_OVERRIDE=loop3 \
MINI_VAE_EPOCHS=100 \
MINI_VAE_CONFIG_TYPE=lightweight \
MINI_VAE_LATENT_DIM=8 \
MINI_RESULTS_DIR=test/result/vae/overfit/vae_overfit_32 \
bash test/mini-test/train_minimal.sh vae
```

该命令由用户执行；代理不得自动运行长训练。

- [ ] **Step 3: 诊断同一批 32 帧**

```bash
conda run -n Radar-Diffusion python diffusion_consistency_radar/scripts/diagnose_vae_reconstruction.py \
  --vae_ckpt test/result/vae/overfit/vae_overfit_32/vae/vae_best_iou.pt \
  --target_voxel_dir Data/NTU4DRadLM_Pre_sensor_aware/loop3/target_voxel \
  --output_dir test/result/vae/overfit/vae_overfit_32_diagnostic \
  --max_files 32 \
  --target_size 32,128,128 \
  --source_pc_range 0,-20,-6,120,20,10 \
  --model_pc_range 0,-20,-6,40,20,10 \
  --device cuda
```

- [ ] **Step 4: 应用验收门槛**

```text
通过：train IoU >= 0.75 且 recall >= 0.80
不通过：train IoU < 0.75
```

若不通过，不训练 LDM/CD；进入 Task 5 的结构容量对照。

### Task 5: 做单变量 VAE 容量对照

**Files:**
- Modify: `test/mini-test/train_minimal.sh`
- Output: `test/result/vae/reconstruction/vae_ablation/`

- [ ] **Step 1: 固定数据、损失、epoch**

三个实验都使用相同 32 帧、BCE+Dice、100 epoch：

```text
A: ultra_lightweight, latent_dim=4
B: lightweight, latent_dim=8
C: lightweight, latent_dim=16
```

- [ ] **Step 2: 依次运行实验**

```bash
for spec in ultra_lightweight:4 lightweight:8 lightweight:16; do
  vae_type="${spec%%:*}"
  latent_dim="${spec##*:}"
  SAMPLES_PER_SCENE=32 \
  TRAIN_SCENES_OVERRIDE=loop3 \
  MINI_VAE_EPOCHS=100 \
  MINI_VAE_CONFIG_TYPE="$vae_type" \
  MINI_VAE_LATENT_DIM="$latent_dim" \
  MINI_RESULTS_DIR="test/result/vae/reconstruction/vae_ablation/${vae_type}_z${latent_dim}" \
  bash test/mini-test/train_minimal.sh vae
done
```

- [ ] **Step 3: 选择最小可用模型**

按以下顺序选择：

1. train IoU 达到 `0.75`；
2. 参数量和峰值显存最低；
3. recall 不低于 `0.80`；
4. 同条件下 precision 更高。

若 lightweight z8 已通过，不升级到 z16 作为正式默认。

### Task 6: 500 帧训练/验证并决定是否解锁 LDM/CD

**Files:**
- Output: `test/result/vae/reconstruction/vae_near40_500_v2/`
- Update: `TODO/findings.md`
- Update: `TODO/progress.md`
- Update: `TODO/task_plan.md`

- [ ] **Step 1: 用选定结构训练 500 帧**

示例命令以 lightweight z8 为候选：

```bash
SAMPLES_PER_SCENE=500 \
TRAIN_SCENES_OVERRIDE=loop3 \
MINI_VAE_EPOCHS=50 \
MINI_VAE_CONFIG_TYPE=lightweight \
MINI_VAE_LATENT_DIM=8 \
MINI_RESULTS_DIR=test/result/vae/reconstruction/vae_near40_500_v2 \
bash test/mini-test/train_minimal.sh vae
```

- [ ] **Step 2: 运行完整 VAE 上界诊断**

```bash
conda run -n Radar-Diffusion python diffusion_consistency_radar/scripts/diagnose_vae_reconstruction.py \
  --vae_ckpt test/result/vae/reconstruction/vae_near40_500_v2/vae/vae_best_iou.pt \
  --target_voxel_dir Data/NTU4DRadLM_Pre_sensor_aware/loop3/target_voxel \
  --output_dir test/result/vae/diagnostics/vae_near40_500_v2_diagnostic \
  --max_files 500 \
  --target_size 32,128,128 \
  --source_pc_range 0,-20,-6,120,20,10 \
  --model_pc_range 0,-20,-6,40,20,10 \
  --device cuda
```

- [ ] **Step 3: 按分层门槛决策**

```text
最低解锁门槛：validation IoU >= 0.50，recall >= 0.65
推荐解锁门槛：validation IoU >= 0.60，recall >= 0.75
结构检查：随机至少 10 帧，树干和主要冠层不能在 VAE 重建中消失
```

低于最低门槛时继续修 VAE，不启动 LDM/CD。达到最低门槛后，才用该 VAE checkpoint 重新训练 LDM；CD 必须最后训练。

- [ ] **Step 4: 回归测试**

```bash
conda run -n Radar-Diffusion python test/unit/test_vae_sparse_occupancy_loss.py -v
conda run -n Radar-Diffusion python test/unit/test_vae_reconstruction_diagnostic.py -v
conda run -n Radar-Diffusion python test/unit/test_dataset_protocol_metadata.py -v
conda run -n Radar-Diffusion python test/unit/test_multimodal_inference_interface.py
conda run -n Radar-Diffusion python -m py_compile \
  diffusion_consistency_radar/cm/vae_3d.py \
  diffusion_consistency_radar/scripts/unified_train.py \
  diffusion_consistency_radar/scripts/diagnose_vae_reconstruction.py \
  diffusion_consistency_radar/scripts/inference.py
bash -n test/mini-test/train_minimal.sh
```

预期：所有测试通过，`git diff --check` 无格式错误。

## 不建议现在做的事

- 不直接继续训练 LDM/CD：它们无法恢复 VAE 已丢失的结构。
- 不只增加 epoch 而保持旧 MSE：这会优化错误的目标。
- 不同时提高体素分辨率、模型容量、latent_dim 和损失权重：无法判断哪项真正有效。
- 不用训练集最优阈值汇报正式性能：阈值必须由验证集确定。
- 不把 mock IR 加入本轮 VAE 修复：VAE 当前只重建 target，先解决单模态上界。
