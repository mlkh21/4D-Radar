# 4D Radar Diffusion & Consistency Models

基于潜空间扩散和一致性蒸馏的 4D 雷达数据生成框架。

## 项目简介

本项目实现了完整的 4D 雷达生成模型训练流程：

1. **VAE (3D Variational Autoencoder)**: 将高维体素数据压缩到低维潜空间
2. **LDM (Latent Diffusion Model)**: 在潜空间中进行扩散训练
3. **CD (Consistency Distillation)**: 从扩散模型蒸馏一致性模型，实现快速生成

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
    --resume train_results/vae/vae_best.pt

# LDM 断点续训
python scripts/unified_train.py --mode ldm \
    --config config/default_config.yaml \
    --vae_ckpt train_results/vae/vae_best.pt \
    --resume train_results/ldm/ldm_best.pt
```

### 推理

```bash
# 使用扩散模型生成
python scripts/image_sample_radar.py --model_path <checkpoint_path>

# 使用一致性模型快速生成
sh diffusion_consistency_radar/launch/inference_cd.sh
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
│   ├── vae_best.pt         # 最佳模型检查点
│   └── vae_epoch0010.pt    # 定期保存的检查点
└── ldm/
    ├── training.log
    ├── metrics.csv         # 每个 epoch 的指标（loss, step, time）
    ├── ldm_best.pt
    └── ldm_step005000.pt
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
