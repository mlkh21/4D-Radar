# 推理使用指南

本文档提供 4D-Radar 模型推理的详细使用说明。

## 一、准备工作

### 1. 确认模型已训练

检查以下模型文件是否存在：

```bash
cd /home/ps/zxj_workspace/src/4D-Radar/diffusion_consistency_radar

# 检查 VAE 模型（必需）
ls train_results/vae/best_model.pth

# 检查 LDM 模型（可选）
ls train_results/ldm/best_model.pth

# 检查 CD 模型（可选）
ls train_results/cd/best_model.pth
```

如果模型不存在，请先运行训练：
```bash
bash launch/train_unified.sh vae   # 训练 VAE
bash launch/train_unified.sh ldm   # 训练 LDM
bash launch/train_unified.sh cd    # 训练 CD
```

### 2. 激活环境

```bash
conda activate Radar  # 或您的 Python 环境名称
```

## 二、推理方法

### 方法 1：使用启动脚本（最简单）

#### LDM 推理（高质量，40步采样）

```bash
cd /home/ps/zxj_workspace/src/4D-Radar/
bash diffusion_consistency_radar/launch/inference_ldm.sh
```

**输出**：
- 文件位置：`diffusion_consistency_radar/inference_results/ldm/ldm_samples_40steps.npy`
- 数据形状：`(10, 4, 32, 128, 128)` - 10个样本
- 生成时间：约 25 秒 (10 samples × 2.5s/sample)

#### CD 推理（快速，1步生成）

```bash
bash diffusion_consistency_radar/launch/inference_cd.sh
```

**输出**：
- 文件位置：`diffusion_consistency_radar/inference_results/cd/cd_samples_1steps.npy`
- 数据形状：`(10, 4, 32, 128, 128)`
- 生成时间：约 0.6 秒 (10 samples × 0.06s/sample)

#### 完整推理示例（推荐）

自动运行 LDM + CD + 可视化：

```bash
bash diffusion_consistency_radar/launch/run_inference_example.sh
```

这将：
1. 使用 LDM 生成 10 个样本（40 步）
2. 使用 CD 生成 10 个样本（1 步）
3. 使用 CD 生成 10 个样本（4 步，提升质量）
4. 自动生成可视化图片
5. 生成 LDM vs CD 对比图

### 方法 2：使用 Python 命令（自定义参数）

#### 基础命令

```bash
cd /home/ps/zxj_workspace/src/4D-Radar

# LDM 推理
python diffusion_consistency_radar/scripts/inference.py \
    --vae_ckpt diffusion_consistency_radar/train_results/vae/best_model.pth \
    --model_ckpt diffusion_consistency_radar/train_results/ldm/best_model.pth \
    --model_type ldm \
    --steps 40 \
    --sampler heun \
    --num_samples 10 \
    --output_dir diffusion_consistency_radar/inference_results/ldm \
    --device cuda

# CD 推理
python diffusion_consistency_radar/scripts/inference.py \
    --vae_ckpt diffusion_consistency_radar/train_results/vae/best_model.pth \
    --model_ckpt diffusion_consistency_radar/train_results/cd/best_model.pth \
    --model_type cd \
    --steps 1 \
    --sampler euler \
    --num_samples 10 \
    --output_dir diffusion_consistency_radar/inference_results/cd \
    --device cuda
```

#### 高级参数调优

**LDM 推理 - 调整质量与速度**

```bash
# 标准质量（推荐）
python diffusion_consistency_radar/scripts/inference.py \
    --vae_ckpt diffusion_consistency_radar/train_results/vae/best_model.pth \
    --model_ckpt diffusion_consistency_radar/train_results/ldm/best_model.pth \
    --model_type ldm \
    --steps 40 \
    --sampler heun \
    --num_samples 100

# 快速模式（质量稍降）
python diffusion_consistency_radar/scripts/inference.py \
    --vae_ckpt diffusion_consistency_radar/train_results/vae/best_model.pth \
    --model_ckpt diffusion_consistency_radar/train_results/ldm/best_model.pth \
    --model_type ldm \
    --steps 20 \
    --sampler euler \
    --num_samples 100

# 极致质量（慢）
python diffusion_consistency_radar/scripts/inference.py \
    --vae_ckpt diffusion_consistency_radar/train_results/vae/best_model.pth \
    --model_ckpt diffusion_consistency_radar/train_results/ldm/best_model.pth \
    --model_type ldm \
    --steps 100 \
    --sampler heun \
    --num_samples 10
```

**CD 推理 - 步数调优**

```bash
# 1步生成（最快，质量中等）
python diffusion_consistency_radar/scripts/inference.py ... --steps 1

# 2步生成（平衡）
python diffusion_consistency_radar/scripts/inference.py ... --steps 2

# 4步生成（接近 LDM 质量）
python diffusion_consistency_radar/scripts/inference.py ... --steps 4
```

### 方法 3：Python 代码调用

创建文件 `my_inference.py`:

```python
import sys
sys.path.insert(0, '/home/ps/zxj_workspace/src/4D-Radar/diffusion_consistency_radar')

from scripts.inference import RadarGenerator
import numpy as np

# 创建生成器
generator = RadarGenerator(
    vae_path='train_results/vae/best_model.pth',
    model_path='train_results/ldm/best_model.pth',
    model_type='ldm',  # 或 'cd'
    device='cuda'
)

# 生成数据
samples = generator.generate(
    condition=None,    # 无条件生成
    num_samples=10,
    steps=40,         # LDM: 40, CD: 1-4
    sampler='heun'    # 'heun' 或 'euler'
)

# 保存
np.save('my_samples.npy', samples.cpu().numpy())
print(f"Generated {samples.shape[0]} samples")
print(f"Shape: {samples.shape}")
```

运行：
```bash
python diffusion_consistency_radar/my_inference.py
```

## 三、可视化结果

### 基础可视化

```bash
# 可视化 LDM 生成结果
python diffusion_consistency_radar/scripts/visualize_results.py \
    --input diffusion_consistency_radar/inference_results/ldm/ldm_samples_40steps.npy \
    --output_dir diffusion_consistency_radar/visualizations/ldm \
    --num_samples 5

# 可视化 CD 生成结果
python diffusion_consistency_radar/scripts/visualize_results.py \
    --input diffusion_consistency_radar/inference_results/cd/cd_samples_1steps.npy \
    --output_dir diffusion_consistency_radar/visualizations/cd \
    --num_samples 5
```

**输出**：
- 每个样本生成一个 PNG 图片
- 包含 4 个通道的顶视图和侧视图
- 终端输出统计信息（均值、标准差、最大最小值等）

### 对比可视化

```bash
# LDM vs CD 对比
python diffusion_consistency_radar/scripts/visualize_results.py \
    --input diffusion_consistency_radar/inference_results/ldm/ldm_samples_40steps.npy \
    --compare diffusion_consistency_radar/inference_results/cd/cd_samples_1steps.npy \
    --output_dir diffusion_consistency_radar/visualizations/comparison
```

**输出**：
- `comparison.png`: 并排对比 LDM 和 CD 生成的第一个样本
- 分别显示 4 个通道的差异

## 四、常见问题

### 1. CUDA 显存不足

```bash
# 减少生成数量
python diffusion_consistency_radar/scripts/inference.py ... --num_samples 1

# 使用 CPU（慢）
python diffusion_consistency_radar/scripts/inference.py ... --device cpu
```

### 2. 模型文件找不到

确认路径正确：
```bash
ls -lh diffusion_consistency_radar/train_results/vae/best_model.pth
ls -lh diffusion_consistency_radar/train_results/ldm/best_model.pth
ls -lh diffusion_consistency_radar/train_results/cd/best_model.pth
```

### 3. 推理速度慢

- LDM: 减少步数 `--steps 20` 或使用 Euler 采样器
- CD: 已经是最快模式（1步仅需 0.06 秒）

### 4. 生成质量不佳

- LDM: 增加步数 `--steps 100` 或使用 Heun 采样器
- CD: 增加步数到 2-4 步

## 五、性能对比表

| 模型 | 步数 | 采样器 | 单样本耗时 | 生成质量 | 适用场景 |
|------|------|--------|-----------|---------|---------|
| LDM | 40 | Heun | 2.5s | ????? | 离线数据增强 |
| LDM | 20 | Euler | 1.0s | ???? | 批量生成 |
| CD | 1 | Euler | 0.06s | ??? | 实时仿真 |
| CD | 2 | Euler | 0.12s | ???? | 在线生成 |
| CD | 4 | Euler | 0.25s | ????? | 平衡质量速度 |

## 六、批量推理

生成大量样本用于数据增强：

```bash
# 生成 1000 个样本（LDM）
python diffusion_consistency_radar/scripts/inference.py \
    --vae_ckpt diffusion_consistency_radar/train_results/vae/best_model.pth \
    --model_ckpt diffusion_consistency_radar/train_results/ldm/best_model.pth \
    --model_type ldm \
    --steps 40 \
    --sampler heun \
    --num_samples 1000 \
    --output_dir diffusion_consistency_radar/inference_results/ldm_batch

# 预计耗时：1000 × 2.5s ≈ 42 分钟

# 生成 1000 个样本（CD）
python diffusion_consistency_radar/scripts/inference.py \
    --vae_ckpt diffusion_consistency_radar/train_results/vae/best_model.pth \
    --model_ckpt diffusion_consistency_radar/train_results/cd/best_model.pth \
    --model_type cd \
    --steps 1 \
    --sampler euler \
    --num_samples 1000 \
    --output_dir diffusion_consistency_radar/inference_results/cd_batch

# 预计耗时：1000 × 0.06s ≈ 1 分钟
```

## 七、输出数据格式

生成的 `.npy` 文件包含：

```python
import numpy as np

data = np.load('diffusion_consistency_radar/inference_results/ldm/ldm_samples_40steps.npy')
# Shape: (num_samples, 4, 32, 128, 128)
#         样本数      通道 深度  高   宽

# 通道说明：
# data[:, 0] - Occupancy (占用)
# data[:, 1] - Intensity (强度)
# data[:, 2] - Doppler (多普勒)
# data[:, 3] - Variance (方差)

# 空间分辨率：
# 深度: 32 bins
# 高度: 128 bins
# 宽度: 128 bins
```

## 八、下一步

- **模型评估**: 使用 `scripts/evaluate.py` 计算 FID、IS 等指标
- **数据增强**: 将生成的数据用于下游任务
- **模型微调**: 在特定场景数据上继续训练
