# 4D Radar Diffusion & Consistency Models

基于潜空间扩散和一致性蒸馏的 4D 雷达数据生成框架。

## 项目简介

本项目实现了完整的 4D 雷达生成模型训练流程：

1. **VAE (3D Variational Autoencoder)**: 将高维体素数据压缩到低维潜空间
2. **LDM (Latent Diffusion Model)**: 在潜空间中进行扩散训练
3. **CD (Consistency Distillation)**: 从扩散模型蒸馏一致性模型，实现快速生成

**? 推理详细教程**: 请参阅 [INFERENCE_GUIDE.md](diffusion_consistency_radar/INFERENCE_GUIDE.md)

## 项目结构

```
diffusion_consistency_radar/
├── cm/                          # 核心模块
│   ├── vae_3d.py               # 3D VAE 模型
│   ├── unet_optimized.py       # 优化的 UNet 模型
│   ├── karras_diffusion.py     # Karras 扩散框架
│   ├── attention_optimized.py  # 注意力机制
│   ├── sampling_optimized.py   # 采样模块
│   ├── dataset_loader.py       # 数据集加载器
│   ├── losses_3d.py            # 3D 损失函数
│   ├── memory_efficient.py     # 显存优化工具
│   ├── augmentation.py         # 数据增强
│   └── ...
├── scripts/                     # 训练脚本
│   ├── unified_train.py        # 统一训练脚本 (VAE/LDM)
│   ├── cd_train_optimized.py   # 一致性蒸馏脚本
│   ├── evaluate.py             # 评估脚本
│   └── image_sample_radar.py   # 采样与可视化
├── config/                      # 配置文件
│   └── default_config.yaml     # 默认配置
├── launch/                      # 启动脚本
│   ├── train_unified.sh        # 统一训练启动脚本
│   ├── inference_edm.sh        # EDM 推理脚本
│   └── inference_cd.sh         # CD 推理脚本
└── train_results/               # 训练结果保存目录
    ├── vae/
    ├── ldm/
    └── cd/
```

## 数据格式

- **输入体素**: `(B, 4, 32, 128, 128)` - 批次、通道(Occ/Int/Dop/Var)、深度、高度、宽度
- **潜空间**: `(B, 4, 8, 32, 32)` - 64 倍压缩

## 快速参考

| 任务 | 命令 | 说明 |
|------|------|------|
| **训练 VAE** | `bash launch/train_unified.sh vae` | 训练变分自编码器 |
| **训练 LDM** | `bash launch/train_unified.sh ldm` | 训练潜空间扩散模型 |
| **训练 CD** | `bash launch/train_unified.sh cd` | 蒸馏一致性模型 |
| **LDM 推理** | `bash launch/inference_ldm.sh` | 40步高质量生成 |
| **CD 推理** | `bash launch/inference_cd.sh` | 1步快速生成 |
| **完整推理示例** | `bash launch/run_inference_example.sh` | LDM + CD + 可视化 |
| **可视化** | `python scripts/visualize_results.py --input <file>` | 查看生成结果 |

## 三大核心模型详解

### 1. VAE (Variational Autoencoder) - 变分自编码器

#### 基础信息
- **作用**: 数据压缩器，将高维体素数据压缩到低维潜空间
- **输入**: 4D 雷达体素 `(4, 32, 128, 128)` - 4 通道，32×128×128 空间分辨率
- **输出**: 潜空间编码 `(4, 8, 32, 32)` - 压缩比 64:1 (4×4×4)
- **网络结构**: 编码器 + 解码器，带 KL 散度正则化

#### 工作原理
```
原始数据 → [Encoder] → 潜空间 (μ, σ) → [采样] → z → [Decoder] → 重建数据
   ↓                                                                    ↓
(4,32,128,128)                                                  (4,32,128,128)
   524,288 维                  (4,8,32,32)                          524,288 维
                                8,192 维
```

**损失函数**: 
- **重建损失**: MSE(原始数据, 重建数据) - 确保重建质量
- **KL 损失**: KL(潜空间分布 || 标准正态分布) - 确保潜空间平滑

#### 三种配置

| 配置类型 | 模型规模 | 参数量 | 显存占用 | 压缩质量 |
|---------|---------|--------|---------|---------|
| `ultra_lightweight` | 超轻量 | ~1M | 8GB | ??? |
| `lightweight` | 轻量 | ~5M | 12GB | ???? |
| `standard` | 标准 | ~20M | 18GB | ????? |

#### 训练流程
```bash
# 1. 训练 VAE
python scripts/unified_train.py --mode vae --config config/default_config.yaml

# 2. 验证压缩效果
python scripts/test_vae.py --ckpt train_results/vae/vae_best.pt

# 输出示例：
# Original: (4, 32, 128, 128) = 524,288 values
# Latent:   (4, 8, 32, 32)    = 8,192 values
# Compression ratio: 64:1
# Reconstruction MSE: 0.0023
```

#### 使用场景
- **训练阶段**: 为 LDM 提供压缩的潜空间表示
- **推理阶段**: 将 LDM 生成的潜空间解码回原始尺寸

---

### 2. LDM (Latent Diffusion Model) - 潜空间扩散模型

#### 基础信息
- **作用**: 在 VAE 压缩的潜空间中学习数据分布，生成新的雷达数据
- **输入**: 潜空间噪声 + 条件信息 `(8, 8, 32, 32)` 
- **输出**: 去噪后的潜空间 `(4, 8, 32, 32)`
- **网络结构**: 3D U-Net (时间步条件 + 空间条件)

#### 工作原理
```
训练阶段：
真实潜空间 z? → [加噪] → z_t → [U-Net 预测噪声] → ε? → [损失] MSE(ε, ε?)
                    ↑                    ↑
                时间步 t          条件潜空间 z_cond

推理阶段：
随机噪声 z_T → [去噪步骤 1] → z_{T-1} → ... → [去噪步骤 T] → z? → [VAE解码] → 雷达数据
          (纯噪声)                                        (干净潜空间)
```

**扩散过程** (Karras 调度):
- **前向扩散**: 逐步向真实数据添加噪声
  ```
  z? (真实) → z? → z? → ... → z_T (纯噪声)
  σ=0.002              σ=80.0
  ```

- **反向去噪**: 从噪声逐步恢复数据 (推理时使用)
  ```
  z_T → [Heun采样器] → z_{T-1} → ... → z?
  40步                                   生成结果
  ```

#### 关键参数

```yaml
ldm:
  model_channels: 32      # U-Net 基础通道数
  channel_mult: [1,2,3]   # 通道倍增策略
  sigma_min: 0.002        # 最小噪声水平
  sigma_max: 80.0         # 最大噪声水平
  epochs: 200             # 训练轮数
```

#### 训练流程
```bash
# 1. 确保 VAE 已训练完成
ls train_results/vae/vae_best.pt

# 2. 训练 LDM (在潜空间)
python scripts/unified_train.py --mode ldm \
    --config config/default_config.yaml \
    --vae_ckpt train_results/vae/vae_best.pt

# 训练日志示例：
# Epoch 1/200: loss=0.0123 | Time: 412s
# Epoch 50/200: loss=0.0045 | Time: 408s
```

#### 优势
- **显存效率**: 在 8×32×32 而非 32×128×128 上操作 → 64 倍显存节省
- **训练速度**: 更小的特征图 → 10+ 倍加速
- **生成质量**: 潜空间更平滑，更容易学习

---

### 3. CD (Consistency Distillation) - 一致性蒸馏

#### 基础信息
- **作用**: 将 LDM 的多步去噪过程蒸馏为单步/少步生成
- **教师模型**: LDM (40 步去噪)
- **学生模型**: CD (1-4 步生成)
- **加速比**: 10-40 倍推理加速

#### 工作原理
```
传统 LDM 推理：
z_T → step1 → step2 → ... → step40 → z?
      ↓       ↓              ↓
     0.5s    0.5s           0.5s
总时间：20秒

CD 推理：
z_T → [一致性模型] → z?
      ↓
     0.5s
总时间：0.5秒 (40倍加速)
```

**一致性蒸馏原理**:
1. **教师指导**: LDM 执行一步去噪 z_t → z_{t-Δt}
2. **学生模拟**: CD 直接预测 z_t → z?
3. **一致性损失**: 确保学生的多步输出保持一致
   ```
   f(z_t) ≈ f(z_{t-Δt})  (一致性约束)
   f(z_t) ≈ teacher(z_t) (蒸馏约束)
   ```

#### 蒸馏流程

```bash
# 1. 确保 LDM 和 VAE 已训练完成
ls train_results/ldm/ldm_best.pt
ls train_results/vae/vae_best.pt

# 2. 蒸馏 CD 模型
python scripts/cd_train_optimized.py \
    --ldm_ckpt train_results/ldm/ldm_best.pt \
    --vae_ckpt train_results/vae/vae_best.pt \
    --num_epochs 100

# 训练过程：
# Epoch 1: 学生模型从 LDM 初始化
# Epoch 50: 逐步学习一致性约束
# Epoch 100: 可以 1-shot 生成高质量结果
```

#### 采样对比

| 模型 | 采样步数 | 单样本耗时 | 生成质量 |
|------|---------|-----------|---------|
| LDM (Heun) | 40 | 2.5s | ????? |
| CD (1-shot) | 1 | 0.06s | ??? |
| CD (2-shot) | 2 | 0.12s | ???? |
| CD (4-shot) | 4 | 0.25s | ????? |

#### 使用场景
- **实时生成**: 需要快速生成雷达数据（如在线仿真）
- **批量生成**: 大规模数据增强
- **资源受限**: 边缘设备、移动平台

---

## 完整工作流程

### 端到端训练

```
阶段 1: VAE 训练 (20 小时)
├─ 输入: 原始雷达数据 (4, 32, 128, 128)
├─ 训练: 重建 + KL 正则化
└─ 输出: VAE 模型 (编码器 + 解码器)

阶段 2: LDM 训练 (80 小时)
├─ 加载: VAE 编码器 (冻结)
├─ 输入: VAE 潜空间 (4, 8, 32, 32)
├─ 训练: 扩散去噪
└─ 输出: LDM 模型 (U-Net)

阶段 3: CD 蒸馏 (40 小时)
├─ 加载: LDM (教师) + VAE
├─ 训练: 一致性蒸馏
└─ 输出: CD 模型 (快速生成器)
```

### 推理流程

#### 使用简化推理脚本（推荐）

**LDM 推理 - 高质量生成 (40 步 Heun 采样)**
```bash
# 使用启动脚本
bash diffusion_consistency_radar/launch/inference_ldm.sh

# 或直接运行
python diffusion_consistency_radar/scripts/inference.py \
    --vae_ckpt train_results/vae/best_model.pth \
    --model_ckpt train_results/ldm/best_model.pth \
    --model_type ldm \
    --steps 40 \
    --sampler heun \
    --num_samples 10 \
    --output_dir inference_results/ldm
```

**CD 推理 - 快速生成 (1 步生成)**
```bash
# 使用启动脚本
bash diffusion_consistency_radar/launch/inference_cd.sh

# 或直接运行
python diffusion_consistency_radar/scripts/inference.py \
    --vae_ckpt train_results/vae/best_model.pth \
    --model_ckpt train_results/cd/best_model.pth \
    --model_type cd \
    --steps 1 \
    --sampler euler \
    --num_samples 10 \
    --output_dir inference_results/cd
```

**参数说明**:
- `--steps`: 采样步数
  - LDM: 推荐 40 步 (高质量) 或 20 步 (平衡)
  - CD: 1-4 步 (1步最快, 4步质量更好)
- `--sampler`: 采样器类型
  - `heun`: 二阶方法，质量更好但稍慢
  - `euler`: 一阶方法，速度快
- `--num_samples`: 生成样本数量
- `--use_condition`: 是否使用数据集中的条件数据

**输出**:
- 生成数据保存为 `.npy` 文件
- 格式: `(num_samples, 4, 32, 128, 128)`
- 4 个通道: Occupancy, Intensity, Doppler, Variance

#### Python API 推理示例

**LDM 推理 (高质量)**
```python
from scripts.inference import RadarGenerator
import torch

# 1. 创建生成器
generator = RadarGenerator(
    vae_path='train_results/vae/best_model.pth',
    model_path='train_results/ldm/best_model.pth',
    model_type='ldm',
    device='cuda'
)

# 2. 生成样本 (40步Heun采样)
generated = generator.generate(
    condition=None,      # 无条件生成，或提供 (B,4,32,128,128) 条件数据
    num_samples=10,
    steps=40,
    sampler='heun'
)

# 3. 输出形状: (10, 4, 32, 128, 128)
print(f"Generated shape: {generated.shape}")
print(f"Value range: [{generated.min():.3f}, {generated.max():.3f}]")
```

**CD 推理 (快速)**
```python
# 1. 创建生成器
generator = RadarGenerator(
    vae_path='train_results/vae/best_model.pth',
    model_path='train_results/cd/best_model.pth',
    model_type='cd',
    device='cuda'
)

# 2. 一步生成（仅需0.06秒）
generated = generator.generate(
    condition=None,
    num_samples=10,
    steps=1,            # CD 一步生成！
    sampler='euler'
)

# 3. 多步生成以提升质量
generated_4step = generator.generate(
    condition=None,
    num_samples=10,
    steps=4,            # 4步质量接近LDM
    sampler='euler'
)
```

**条件生成示例**
```python
import numpy as np
import torch

# 1. 从数据集加载条件
from cm.dataset_loader import NTU4DRadLM_VoxelDataset
from torch.utils.data import DataLoader

dataset = NTU4DRadLM_VoxelDataset('./NTU4DRadLM_Pre', split='val')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
condition_data = next(iter(dataloader))[1]  # 获取条件数据

# 2. 基于条件生成
generated = generator.generate(
    condition=condition_data,
    num_samples=1,
    steps=40,
    sampler='heun'
)
```

#### 性能对比

| 模型 | 采样步数 | 单样本耗时 | 生成质量 | 适用场景 |
|------|---------|-----------|---------|---------|
| **LDM (Heun)** | 40 | ~2.5s | ????? | 离线数据增强、高质量生成 |
| **LDM (Heun)** | 20 | ~1.2s | ???? | 快速原型、批量生成 |
| **CD (1-step)** | 1 | ~0.06s | ??? | 实时仿真、边缘设备 |
| **CD (2-step)** | 2 | ~0.12s | ???? | 在线生成、低延迟需求 |
| **CD (4-step)** | 4 | ~0.25s | ????? | 平衡质量与速度 |

#### 旧版推理（legacy）

如需使用旧版推理脚本：
```bash
# EDM/LDM 推理
bash diffusion_consistency_radar/launch/inference_edm.sh

# 直接调用
python diffusion_consistency_radar/scripts/image_sample_radar.py \
    --model_path <checkpoint_path> \
    --training_mode edm \
    --steps 40
```

#### 可视化生成结果

使用可视化脚本查看生成的雷达数据：

```bash
# 可视化单个推理结果
python diffusion_consistency_radar/scripts/visualize_results.py \
    --input inference_results/ldm/ldm_samples_40steps.npy \
    --output_dir visualizations/ldm \
    --num_samples 5

# 对比 LDM 和 CD 生成质量
python diffusion_consistency_radar/scripts/visualize_results.py \
    --input inference_results/ldm/ldm_samples_40steps.npy \
    --compare inference_results/cd/cd_samples_1steps.npy \
    --output_dir visualizations/comparison
```

**输出说明**:
- `sample_XXX.png`: 单个样本的 4 通道可视化（顶视图 + 侧视图）
- `comparison.png`: LDM vs CD 对比图
- 终端输出包含详细统计信息（均值、标准差、非零比例等）


## 快速开始

### 环境配置

```bash
conda activate Radar  # 或您的环境名称
pip install -r requirements.txt
```

### 训练流程

使用统一启动脚本进行训练：

```bash
# 方式一：分步训练
sh diffusion_consistency_radar/launch/train_unified.sh vae   # Step 1: VAE
sh diffusion_consistency_radar/launch/train_unified.sh ldm   # Step 2: LDM
sh diffusion_consistency_radar/launch/train_unified.sh cd    # Step 3: CD

# 方式二：完整流程
sh diffusion_consistency_radar/launch/train_unified.sh all
```

或直接运行 Python 脚本：

```bash
# VAE 训练
python scripts/unified_train.py --mode vae --config config/default_config.yaml

# LDM 训练
python scripts/unified_train.py --mode ldm \
    --config config/default_config.yaml \
    --vae_ckpt train_results/vae/vae_best.pt

# CD 蒸馏
python scripts/cd_train_optimized.py \
    --ldm_ckpt train_results/ldm/ldm_best.pt \
    --vae_ckpt train_results/vae/vae_best.pt
```

### 断点续训

从最佳检查点恢复训练：

```bash
# VAE 断点续训
python scripts/unified_train.py --mode vae \
    --config config/default_config.yaml \
    --resume train_results/vae/best_model.pth

# LDM 断点续训
python scripts/unified_train.py --mode ldm \
    --config config/default_config.yaml \
    --vae_ckpt train_results/vae/best_model.pth \
    --resume train_results/ldm/best_model.pth
```

### 推理

请参考上方"推理流程"章节，使用简化推理脚本或 Python API 进行推理。

**快速开始**:
```bash
# 完整推理示例（LDM + CD + 可视化）
bash diffusion_consistency_radar/launch/run_inference_example.sh

# 或分别运行：
# LDM 高质量生成
bash diffusion_consistency_radar/launch/inference_ldm.sh

# CD 快速生成
bash diffusion_consistency_radar/launch/inference_cd.sh
```

## 配置说明

配置文件位于 `config/default_config.yaml`，主要参数：

```yaml
# 数据配置
data:
  batch_size: 2
  num_workers: 4

# VAE 配置
vae:
  config_type: ultra_lightweight  # ultra_lightweight / lightweight / standard
  epochs: 100
  lr: 1.0e-4
  kl_weight: 1.0e-6

# LDM 配置
ldm:
  model_channels: 32
  channel_mult: [1, 2, 3]
  num_res_blocks: 1
  epochs: 200
  lr: 1.0e-4

# CD 配置
cd:
  total_training_steps: 100000
  lr: 5.0e-5

# 显存优化
optimization:
  gradient_accumulation_steps: 8
  use_amp: true           # 混合精度
  use_checkpoint: true    # 梯度检查点
```

## 核心模块说明

### 训练日志

训练过程会自动生成以下日志文件：

```
train_results/
├── vae/
│   ├── training.log        # 完整训练日志（包含所有终端输出）
│   ├── metrics.csv         # 每个 epoch 的指标（loss, recon, kl, time）
│   ├── best_model.pth      # 最佳模型检查点
│   └── checkpoint_epoch0010.pth    # 定期保存的检查点
└── ldm/
    ├── training.log
    ├── metrics.csv         # 每个 epoch 的指标（loss, step, time）
    ├── best_model.pth
    └── checkpoint_step005000.pth
```

### VAE (vae_3d.py)

- 支持三种配置：`ultra_lightweight` / `lightweight` / `standard`
- 压缩比：4×4×4 = 64 倍
- 支持 VQ-VAE 变体

### UNet (unet_optimized.py)

- 多种注意力机制：Flash / Linear / Window / Sparse
- 非对称下采样策略
- 支持梯度检查点

### Karras Diffusion (karras_diffusion.py)

- Heun / Euler 采样器
- 可配置 sigma 调度

### Consistency Distillation (cd_train_optimized.py)

- 从 LDM 教师模型蒸馏
- EMA 目标模型管理
- 支持 1-shot 快速生成

## 显存要求

| 阶段 | 显存需求 | batch_size | 梯度累积 |
|------|---------|------------|---------|
| VAE  | ~18 GB  | 2          | 8       |
| LDM  | ~20 GB  | 1          | 8       |
| CD   | ~18 GB  | 2          | 8       |

## 参考文献

- [Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Consistency Models](https://arxiv.org/abs/2303.01469)
- [Consistency Distillation](https://arxiv.org/abs/2303.08941)
- [Karras et al. EDM](https://arxiv.org/abs/2206.00364)
