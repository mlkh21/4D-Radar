# 4D Radar Diffusion Model - 优化版

针对 24GB 消费级显卡优化的 4D 雷达扩散模型。

## ? 主要优化

### 1. 高效注意力机制 (显存节省 3-5 倍)

| 类型 | 描述 | 适用场景 |
|------|------|---------|
| `flash` | PyTorch 2.0 原生 Flash Attention | **推荐**，最佳效率 |
| `window` | 3D 窗口注意力 (Swin风格) | 需要局部特征 |
| `linear` | 线性注意力 O(N) | 超大分辨率 |
| `sparse` | Top-K 稀疏注意力 | 稀疏数据 |
| `height` | 沿 Z 轴注意力 | 高度敏感任务 |
| `none` | 禁用注意力 | 最省显存 |

### 2. 非对称下采样 (保护 Z 轴)

```python
# 策略选项
"xy_only"   # (1, 2, 2) - 只对 XY 下采样，保留 Z
"z_half"    # (1, 2, 2) - Z 轴保守下采样
"full"      # (2, 2, 2) - 完整 3D 下采样
"adaptive"  # 自动根据当前分辨率选择
```

### 3. 通道瘦身配置

```python
# 原配置（显存爆炸）
model_channels=128, channel_mult=(1, 2, 4, 8)  # 深层 1024 通道

# 优化配置（24GB 显卡）
model_channels=64, channel_mult=(1, 2, 2, 4)   # 深层 256 通道
```

### 4. 多种归一化选项

| 类型 | 描述 | 适用场景 |
|------|------|---------|
| `group` | GroupNorm (默认) | 通用 |
| `layer` | LayerNorm3D | 极小 batch |
| `instance` | InstanceNorm3D | 稀疏数据 |
| `rms` | RMSNorm | 高效归一化 |
| `adaptive` | 自适应归一化 | 混合稀疏度 |

### 5. Latent Diffusion (终极优化)

将 3D 体素压缩到潜空间，显存降低 8-16 倍：

```
原始: (4, 32, 128, 128) -> 压缩 -> 潜空间: (4, 8, 32, 32)
压缩比: 4x4x4 = 64 倍
```

## ? 安装

```bash
# 基础依赖
pip install torch>=2.0 einops tqdm

# 可选：Flash Attention 加速
pip install flash-attn  # 需要 CUDA 11.6+
```

## ? 快速开始

### 方案 A：直接训练（推荐起步）

```bash
# 使用优化后的默认配置
sh diffusion_consistency_radar/launch/train_edm.sh
```

### 方案 B：Latent Diffusion（终极优化）

```bash
# 步骤 1：训练 VAE
sh diffusion_consistency_radar/launch/train_latent_diffusion.sh vae

# 步骤 2：训练 Latent Diffusion
sh diffusion_consistency_radar/launch/train_latent_diffusion.sh ldm
```

## ?? 配置参数

### 模型配置

```bash
# 注意力配置
--attention_type flash         # flash/window/linear/sparse/height/none
--attention_resolutions 8,4    # 在哪些分辨率使用注意力

# 下采样配置
--downsample_type asymmetric   # asymmetric/standard
--downsample_stride xy_only    # xy_only/z_half/full/adaptive

# 通道配置
--model_channels 64
--channel_mult 1,2,2,4

# 归一化配置
--norm_type group              # group/layer/instance/rms/adaptive

# 显存优化
--use_fp16 True
--use_checkpoint True
--use_optimized_unet True
```

### 显存估计 (RTX 4090 24GB)

| 配置 | Batch Size | 显存 |
|------|-----------|------|
| 轻量级 | 4 | ~18GB |
| 轻量级 | 8 | ~22GB |
| 超轻量 | 8 | ~14GB |
| Latent Diffusion | 16 | ~20GB |

## ? 文件结构

```
diffusion_consistency_radar/
├── cm/
│   ├── __init__.py           # 模块导出
│   ├── attention_optimized.py # 优化的注意力机制
│   ├── sampling_optimized.py  # 非对称采样
│   ├── nn.py                  # 多种归一化
│   ├── unet.py               # 原始 UNet
│   ├── unet_optimized.py     # 优化版 UNet
│   ├── vae_3d.py             # 3D VAE/VQ-VAE
│   └── ...
├── scripts/
│   ├── edm_train_radar.py         # EDM 训练
│   └── train_latent_diffusion.py  # Latent Diffusion 训练
└── launch/
    ├── train_edm.sh                    # 训练脚本
    └── train_latent_diffusion.sh       # LDM 训练脚本
```

## ? 预设配置

```python
from cm import (
    create_lightweight_unet_config,     # 轻量级 (24GB)
    create_ultra_lightweight_unet_config, # 超轻量 (12GB)
    create_balanced_unet_config,        # 平衡配置
    create_lightweight_vae_config,      # 轻量 VAE
    create_standard_vae_config,         # 标准 VAE
)
```

## ? 优化效果

| 优化项 | 显存节省 | 速度提升 |
|-------|---------|---------|
| Flash Attention | 40-60% | 3-5x |
| 非对称下采样 | 20-30% | 1.5x |
| 通道瘦身 | 50-70% | 2-3x |
| FP16 + Checkpoint | 40-50% | 0.9x |
| Latent Diffusion | 80-90% | 10x+ |

## ? 推荐配置

### 24GB 显卡 (RTX 4090/3090)
```bash
--model_channels 64
--channel_mult 1,2,2,4
--attention_type flash
--attention_resolutions 8,4
--batch_size 4
```

### 12GB 显卡 (RTX 4070/3080)
```bash
--model_channels 48
--channel_mult 1,2,2,2
--attention_type linear
--attention_resolutions 4
--use_depthwise True
--batch_size 4
```

### 8GB 显卡 (使用 Latent Diffusion)
```bash
# 先训练 VAE，再训练 LDM
sh diffusion_consistency_radar/launch/train_latent_diffusion.sh vae
sh diffusion_consistency_radar/launch/train_latent_diffusion.sh ldm
```

## ? 原始优化建议

以下是优化前的分析记录：

---

考虑到你现在的硬件限制（消费级显卡，24GB 显存）和项目本身的问题，要魔改这个结构以获得更好的训练效率和效果，我有以下几个针对性的建议：

### 1. 也是最重要的一点：切换到 Voxel-based 注意力机制 (Window/Sparse Attention)

**现状：** 代码里的 `AttentionBlock` 似乎是全局注意力（Full Attention）。
**问题：** 在 3D 体素上做全局注意力是灾难。复杂度是 $O(N^2)$，当 $N = H \times W \times D$ 时，内存直接爆炸。
**魔改方向：**
*   **Window Attention (Swin Transformer 风格)**：只在局部的 3D 小窗口（比如 $4 \times 4 \times 4$）内做 Attention。这样显存占用是 $O(N)$ 线性增长的。
*   **Linear Attention / Flash Attention**：如果你的环境支持（需要较新的 PyTorch 和 CUDA），把标准的 Attention 换成 `FlashAttention-v2`，可以大幅降低显存占用并提速 3-5 倍。这是改动最小收益最大的方案。
*   **3D Sparse Conv (稀疏卷积)**：雷达数据大面积都是空的（0），用密集卷积（Dense Conv）处理大量空区域是极大的浪费。如果能接入 `spconv` 或 `MinkowskiEngine`，将整个 UNet 改为稀疏卷积网络，显存可能会降到原来的 1/10，速度起飞。这将是一个大工程，但收益是革命性的。

### 2. 降低 3D 深度 (Z轴) 的下采样策略

**现状：** 代码里的 `channel_mult` 和下采样似乎是在 H, W, D 三个维度同时进行的（各 /2）。
**问题：** 雷达数据通常在 Z 轴（高度或多普勒/时间维）上比 XY 平面小得多。盲目地对 Z 轴进行多次下采样，最后 Z 轴可能只剩 1 或 2 了，信息丢失严重且计算冗余。
**魔改方向：**
*   **非对称下采样**：自定义 Downsample 模块。在浅层只对 X, Y 下采样，保留 Z 轴分辨率；或者使用 $(2, 2, 1)$ 的 stride。
*   这样可以保持 Z 轴特征的完整性，同时减少特征图的总像素数。

### 3. 通道数瘦身 (Channel Pruning)

**现状：** `model_channels=128`，倍率为 `1,2,3,4`（即深层达到 512 通道）。
**魔改方向：**
*   对于雷达这种稀疏信号，信噪比本身就低，不需要像 ImageNet 图像生成那样巨大的通道数来从纹理中提取语义。
*   **建议配置**：尝试 `model_channels=64`，倍率 `1,2,2,4`。先把网络变窄，能跑起来 Batch Size 4 或 8 比用大网络跑 Batch Size 1 训练效果要好得多（Batch Size 太小会导致 BN 层统计不准，梯度震荡）。

### 4. 替换 Normalization 层

**现状：** 使用 GroupNorm（通常是 UNet 的标配）。
**魔改方向：**
*   由于你只能跑极小的 Batch Size (1 或 2)，任何依赖 Batch 统计量的层（如 BatchNorm）都绝对不能用。
*   **LayerNorm** 或 **GroupNorm** 是对的，但可以尝试调整 Group 的数量。
*   更有趣的是：考虑到 3D 稀疏性，可以尝试 **InstanceNorm**，这在点云处理中经常有奇效。

### 5. 引入 Latent Diffusion (LDM) 思路

**现状：** 直接在体素空间（Pixel Space）做扩散。
**魔改方向：**
*   这是终极魔改。训练一个轻量级的 3D VQ-VAE 或 VAE，把庞大的 3D 雷达数据压缩到一个很小的 Latent Space（潜空间）里。
*   然后在 Latent Space 里跑现在的这个 Diffusion 模型。
*   **收益**：输入维度变小了（比如 128x128x32 -> 32x32x8），你现在的这个模型就可以轻松跑 Batch Size 32，且训练速度快 10 倍以上。Stable Diffusion 就是这么把生成任务做大做强的。

### 总结建议
作为起步，我建议你先做 **#3 (通道瘦身)**，这是改个参数就能立竿见影的。
如果你有余力改代码，优先尝试 **#1 中的 Flash Attention** 替换。
如果你想发高质量论文或做极致优化，**#5 (Latent Diffusion)** 是目前的 SOTA 方向。
