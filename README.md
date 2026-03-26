# 4D Radar Diffusion & Consistency Models

基于潜空间扩散与一致性蒸馏的 4D 雷达体素生成框架，支持训练、推理、评估与可视化。

## 项目简介

当前代码已实现以下主流程：

1. **VAE (3D Variational Autoencoder)**：将体素数据压缩到潜空间。
2. **LDM (Latent Diffusion Model)**：在潜空间进行扩散建模与多步采样。
3. **CD (Consistency Distillation)**：由 LDM 蒸馏一致性模型，用于少步快速推理。

推理与评估细节请参考 [INFERENCE_GUIDE.md](diffusion_consistency_radar/INFERENCE_GUIDE.md)。

## 项目结构

```
diffusion_consistency_radar/
├── cm/                          # 核心模块
│   ├── vae_3d.py               # 3D VAE
│   ├── unet_optimized.py       # 优化 3D UNet
│   ├── karras_diffusion.py     # Karras 扩散与采样
│   ├── attention_optimized.py  # 注意力模块
│   ├── sampling_optimized.py   # 采样/上下采样模块
│   ├── dataset_loader.py       # 数据集加载
│   ├── losses_3d.py            # 3D 损失
│   ├── memory_efficient.py     # 显存优化工具
│   ├── probabilistic_mapping.py# 概率栅格与局部查询（新增）
│   └── ...
├── scripts/                     # 训练/推理/评估脚本
│   ├── unified_train.py        # 统一训练（VAE/LDM）
│   ├── cd_train_optimized.py   # CD 蒸馏训练
│   ├── inference.py            # 推理主脚本
│   ├── streaming_map_update.py # 在线概率地图更新（新增）
│   ├── evaluate.py             # 评估
│   └── visualize_results.py    # 可视化
├── config/                      # 配置文件
│   ├── default_config.yaml
│   └── data_loading_config.yml
├── launch/                      # 启动脚本
│   ├── train_unified.sh
│   ├── inference_ldm.sh
│   ├── inference_cd.sh
│   ├── inference_edm.sh
│   └── run_inference_example.sh
└── train_results/               # 训练结果目录
    ├── vae/
    ├── ldm/
    └── cd/
```

## 数据格式

- **体素输入（主流程）**：`(B, 4, 32, 128, 128)`，通道顺序为 `Occ/Int/Dop/Var`。
- **潜空间表示**：`(B, 4, 8, 32, 32)`。
- **推理脚本兼容输入**：稀疏 `.npz` 与 dense `.npy`。

## 快速参考

| 任务 | 命令 | 说明 |
|------|------|------|
| **训练 VAE** | `bash diffusion_consistency_radar/launch/train_unified.sh vae` | 训练 VAE |
| **训练 LDM** | `bash diffusion_consistency_radar/launch/train_unified.sh ldm` | 训练潜空间扩散模型 |
| **训练 CD** | `bash diffusion_consistency_radar/launch/train_unified.sh cd` | 蒸馏一致性模型 |
| **LDM 推理** | `bash diffusion_consistency_radar/launch/inference_ldm.sh` | 按 `data_loading_config.yml` 场景逐文件推理 |
| **CD 推理** | `bash diffusion_consistency_radar/launch/inference_cd.sh` | 按场景逐文件快速推理 |
| **完整推理示例** | `bash diffusion_consistency_radar/launch/run_inference_example.sh` | LDM + CD + CD-4step |
| **在线地图更新** | `python diffusion_consistency_radar/scripts/streaming_map_update.py --help` | 概率栅格/DEM更新 |
| **可视化** | `python diffusion_consistency_radar/scripts/visualize_results.py --input <file>` | 可视化生成结果 |

## 三大核心模型详解

### 1. VAE (Variational Autoencoder) - 变分自编码器

#### 基础信息
- **作用**：把高维体素压缩到潜空间，降低后续扩散开销。
- **输入**：`(B, 4, 32, 128, 128)`。
- **输出**：`(B, 4, 8, 32, 32)`。
- **配置**：`ultra_lightweight / lightweight / standard`（见 `cm/vae_3d.py` 与配置文件）。

#### 工作原理
```
体素输入 -> Encoder -> (mu, logvar) -> reparameterize -> z -> Decoder -> 重建体素
```

#### 三种配置

| 配置类型 | 用途 | 备注 |
|---------|------|------|
| `ultra_lightweight` | 资源受限场景 | 默认配置常用 |
| `lightweight` | 平衡模型容量与资源 | 中等显存压力 |
| `standard` | 更高容量实验 | 训练成本更高 |

#### 训练流程
```bash
python diffusion_consistency_radar/scripts/unified_train.py \
  --mode vae \
  --config diffusion_consistency_radar/config/default_config.yaml
```

#### 使用场景
- 为 LDM/CD 提供稳定潜空间表示。
- 将潜空间输出解码回原始体素空间。

---

### 2. LDM (Latent Diffusion Model) - 潜空间扩散模型

#### 基础信息
- **作用**：在潜空间中学习分布并生成样本。
- **输入**：噪声潜变量与条件潜变量拼接。
- **输出**：去噪后的潜变量，再经 VAE 解码。
- **采样器**：`heun` / `euler`（见 `scripts/inference.py`）。

#### 工作原理
```
训练: z0 -> add noise -> zt -> UNet denoise prediction -> loss
推理: zT -> iterative denoise (N steps) -> z0 -> VAE decode
```

#### 关键参数
```yaml
ldm:
  model_channels: 32
  channel_mult: [1, 2, 3]
  sigma_min: 0.002
  sigma_max: 80.0
  epochs: 200
```

#### 训练流程
```bash
python diffusion_consistency_radar/scripts/unified_train.py \
  --mode ldm \
  --config diffusion_consistency_radar/config/default_config.yaml \
  --vae_ckpt Result/train_results/vae/vae_best.pt
```

#### 优势
- 在潜空间训练，显存占用较原始体素扩散更低。
- 支持多步采样，质量与速度可权衡。

---

### 3. CD (Consistency Distillation) - 一致性蒸馏

#### 基础信息
- **作用**：将 LDM 多步去噪蒸馏为少步（1-4步）生成。
- **教师模型**：LDM。
- **学生模型**：CD（结构与教师兼容）。

#### 工作原理
```
Teacher(LDM) provides denoising targets -> Student(CD) learns consistent mapping
```

#### 蒸馏流程
```bash
python diffusion_consistency_radar/scripts/cd_train_optimized.py \
  --ldm_ckpt Result/train_results/ldm/ldm_best.pt \
  --vae_ckpt Result/train_results/vae/vae_best.pt \
  --dataset_dir ./Data/NTU4DRadLM_Pre
```

#### 采样对比

| 模型 | 采样步数 | 典型用途 |
|------|---------|---------|
| LDM | 40 | 高质量离线生成 |
| CD | 1 | 快速在线生成 |
| CD | 2-4 | 质量与速度折中 |

#### 使用场景
- 实时或准实时推理链路。
- 大批量样本生成。

---

## 完整工作流程

### 端到端训练

```
阶段 1: 训练 VAE
  输入: 4D 雷达体素
  输出: vae_best.pt

阶段 2: 训练 LDM
  输入: VAE 潜空间
  输出: ldm_best.pt

阶段 3: 蒸馏 CD
  输入: LDM + VAE
  输出: cd_best.pt
```

### 推理流程

#### 使用启动脚本（推荐）

```bash
# LDM 场景推理
bash diffusion_consistency_radar/launch/inference_ldm.sh

# CD 场景推理
bash diffusion_consistency_radar/launch/inference_cd.sh

# 完整示例（含 CD 4步）
bash diffusion_consistency_radar/launch/run_inference_example.sh
```

#### 直接调用推理脚本

```bash
python diffusion_consistency_radar/scripts/inference.py \
  --vae_ckpt Result/train_results/vae/vae_best.pt \
  --model_ckpt Result/train_results/ldm/ldm_best.pt \
  --model_type ldm \
  --steps 40 \
  --sampler heun \
  --radar_voxel_dir <radar_voxel_dir> \
  --save_pointcloud \
  --output_dir <output_dir>
```

#### 在线地图更新（新增）

```bash
python diffusion_consistency_radar/scripts/streaming_map_update.py \
  --radar_voxel_dir Result/inference_results/cd_4step \
  --output_dir Result/inference_results/streaming_map \
  --dt 0.05
```

#### 旧版推理（legacy）

```bash
bash diffusion_consistency_radar/launch/inference_edm.sh
```

#### 可视化生成结果

```bash
python diffusion_consistency_radar/scripts/visualize_results.py \
  --input Result/inference_results/ldm/ldm_samples_40steps.npy \
  --output_dir diffusion_consistency_radar/visualizations/ldm \
  --num_samples 5
```

## 快速开始

### 环境配置

本仓库当前没有 `requirements.txt`，建议使用以下方式安装核心依赖：

```bash
conda activate Radar
pip install -e diffusion_consistency_radar
```

如需完整环境，请根据 `diffusion_consistency_radar/setup.py` 和实际 GPU 环境补齐依赖版本。

### 训练流程

```bash
# 分步训练
sh diffusion_consistency_radar/launch/train_unified.sh vae
sh diffusion_consistency_radar/launch/train_unified.sh ldm
sh diffusion_consistency_radar/launch/train_unified.sh cd

# 完整流程
sh diffusion_consistency_radar/launch/train_unified.sh all
```

或直接运行 Python：

```bash
python diffusion_consistency_radar/scripts/unified_train.py \
  --mode vae \
  --config diffusion_consistency_radar/config/default_config.yaml

python diffusion_consistency_radar/scripts/unified_train.py \
  --mode ldm \
  --config diffusion_consistency_radar/config/default_config.yaml \
  --vae_ckpt Result/train_results/vae/vae_best.pt

python diffusion_consistency_radar/scripts/cd_train_optimized.py \
  --ldm_ckpt Result/train_results/ldm/ldm_best.pt \
  --vae_ckpt Result/train_results/vae/vae_best.pt \
  --dataset_dir ./Data/NTU4DRadLM_Pre
```

### 断点续训

```bash
python diffusion_consistency_radar/scripts/unified_train.py \
  --mode vae \
  --config diffusion_consistency_radar/config/default_config.yaml \
  --resume Result/train_results/vae/vae_best.pt

python diffusion_consistency_radar/scripts/unified_train.py \
  --mode ldm \
  --config diffusion_consistency_radar/config/default_config.yaml \
  --vae_ckpt Result/train_results/vae/vae_best.pt \
  --resume Result/train_results/ldm/ldm_best.pt
```

### 推理

优先使用 `launch` 下脚本进行场景级批量推理；单样本/调参建议直接调用 `scripts/inference.py`。

## 配置说明

主要配置文件：`diffusion_consistency_radar/config/default_config.yaml`

```yaml
data:
  dataset_dir: "./Data/NTU4DRadLM_Pre"
  batch_size: 2
  num_workers: 4

vae:
  config_type: "ultra_lightweight"
  epochs: 100
  lr: 1.0e-4
  save_dir: "./Result/train_results/vae"

ldm:
  model_channels: 32
  channel_mult: [1, 2, 3]
  epochs: 200
  lr: 1.0e-4
  save_dir: "./Result/train_results/ldm"

cd:
  epochs: 200
  lr: 5.0e-5
  save_dir: "./Result/train_results/cd"

optimization:
  use_amp: false
  gradient_accumulation_steps: 8
  use_checkpoint: true
```

## 核心模块说明

### 训练日志

训练结果通常保存在：

```
Result/train_results/
├── vae/
│   ├── training.log
│   ├── metrics.csv
│   └── vae_best.pt
├── ldm/
│   ├── training.log
│   ├── metrics.csv
│   └── ldm_best.pt
└── cd/
    ├── training.log
    ├── metrics.csv
    └── cd_best.pt
```

### VAE (vae_3d.py)

- 3D 编码器-解码器结构。
- 支持多档配置与潜空间压缩。

### UNet (unet_optimized.py)

- 用于 LDM/CD 的 3D UNet 主体。
- 包含轻量注意力与效率优化路径。

### Karras Diffusion (karras_diffusion.py)

- 噪声调度与采样支持。
- 与 `scripts/inference.py` 的 Heun/Euler 采样配套。

### Consistency Distillation (cd_train_optimized.py)

- 教师-学生蒸馏流程。
- 支持日志记录和断点恢复。

## 显存要求

下表为经验范围，实际取决于 batch size、模型配置与 CUDA 环境：

| 阶段 | 建议显存 | 备注 |
|------|---------|------|
| VAE 训练 | 12GB+ | 建议开启梯度累积 |
| LDM 训练 | 16GB+ | 通常显存压力最高 |
| CD 训练 | 12GB+ | 与配置和步数相关 |
| 推理 | 8GB+ | CD 通常低于 LDM |

## 参考文献

- [Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Consistency Models](https://arxiv.org/abs/2303.01469)
- [Consistency Distillation](https://arxiv.org/abs/2303.08941)
- [Karras et al. EDM](https://arxiv.org/abs/2206.00364)
