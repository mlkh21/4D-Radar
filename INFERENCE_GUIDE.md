# 推理使用指南

本文档基于当前仓库代码，给出可直接执行的推理与在线地图更新说明。

## 一、准备工作

### 1. 激活环境

```bash
conda activate Radar
```

### 2. 确认模型已训练

```bash
cd /home/ps/zxj_workspace/src/4D-Radar/diffusion_consistency_radar

# VAE（必需）
ls train_results/vae/vae_best.pt

# LDM（可选，运行 LDM 推理时必需）
ls train_results/ldm/ldm_best.pt

# CD（可选，运行 CD 推理时必需）
ls train_results/cd/cd_best.pt
```

如果模型不存在，请先训练：

```bash
bash launch/train_unified.sh vae
bash launch/train_unified.sh ldm
bash launch/train_unified.sh cd
```
## 二、推理方法

### 方法 1：使用启动脚本（推荐）

当前启动脚本是按场景逐文件推理模式，场景来源于：
- `diffusion_consistency_radar/config/data_loading_config.yml` 的 `data.test`

#### LDM 推理

```bash
cd /home/ps/zxj_workspace/src/4D-Radar
bash diffusion_consistency_radar/launch/inference_ldm.sh
```

输出目录示例：
- `Result/inference_results/<scene>_ldm_eval/`

主要输出文件：
- `*_pcl.npy`：逐文件点云输出
- `comparison_metrics.csv`：与 LiDAR 对比的 Chamfer 指标（启用对比时）

#### CD 推理

```bash
bash diffusion_consistency_radar/launch/inference_cd.sh
```

输出目录示例：
- `Result/inference_results/<scene>_cd_eval/`

#### 完整推理示例

```bash
bash diffusion_consistency_radar/launch/run_inference_example.sh
```

该脚本会按 test 场景执行：
1. LDM 推理（40步）
2. CD 推理（1步）
3. CD 推理（4步）

### 方法 2：使用 Python 命令（可调参数）

#### A. 场景逐文件推理（与 launch 脚本一致）

```bash
cd /home/ps/zxj_workspace/src/4D-Radar

python diffusion_consistency_radar/scripts/inference.py \
  --vae_ckpt Result/train_results/vae/vae_best.pt \
  --model_ckpt Result/train_results/ldm/ldm_best.pt \
  --model_type ldm \
  --steps 40 \
  --sampler heun \
  --radar_voxel_dir /path/to/radar_voxel \
  --save_pointcloud \
  --output_dir Result/inference_results/custom_ldm_eval \
  --device cuda
```

可选 LiDAR 对比参数：
- `--compare_with_lidar`
- `--raw_livox_dir /path/to/livox_lidar`
- `--lidar_index_file /path/to/lidar_index_sequence.txt`

#### B. 批量样本推理（输出一个聚合 .npy）

```bash
python diffusion_consistency_radar/scripts/inference.py \
  --vae_ckpt Result/train_results/vae/vae_best.pt \
  --model_ckpt Result/train_results/cd/cd_best.pt \
  --model_type cd \
  --steps 1 \
  --sampler euler \
  --num_samples 10 \
  --output_dir Result/inference_results/cd
```

该模式输出：
- `Result/inference_results/cd/cd_samples_1steps.npy`

#### 参数说明

- `--model_type`: `ldm` 或 `cd`
- `--steps`: 采样步数（LDM 常用 40；CD 常用 1/2/4）
- `--sampler`: `heun` 或 `euler`
- `--radar_voxel_dir`: 启用逐文件推理模式
- `--num_samples`: 批量样本推理模式的样本数
- `--save_pointcloud`: 逐样本保存点云
- `--save_voxel`: 逐样本保存体素

### 方法 3：Python 代码调用

```python
import sys
import numpy as np

sys.path.insert(0, '/home/ps/zxj_workspace/src/4D-Radar/diffusion_consistency_radar')
from scripts.inference import RadarGenerator

generator = RadarGenerator(
    vae_path='train_results/vae/vae_best.pt',
    model_path='train_results/ldm/ldm_best.pt',
    model_type='ldm',
    device='cuda'
)

samples = generator.generate(
    condition=None,
    num_samples=4,
    steps=40,
    sampler='heun'
)

np.save('my_samples.npy', samples.cpu().numpy())
print(samples.shape)
```

## 三、可视化结果

可视化脚本通常用于“批量样本推理模式”产出的聚合 `.npy` 文件。

### 基础可视化

```bash
python diffusion_consistency_radar/scripts/visualize_results.py \
  --input Result/inference_results/cd/cd_samples_1steps.npy \
  --output_dir diffusion_consistency_radar/visualizations/cd \
  --num_samples 5
```

### 对比可视化

```bash
python diffusion_consistency_radar/scripts/visualize_results.py \
  --input Result/inference_results/ldm/ldm_samples_40steps.npy \
  --compare Result/inference_results/cd/cd_samples_1steps.npy \
  --output_dir diffusion_consistency_radar/visualizations/comparison
```

## 四、在线概率地图更新

该模式用于在线建图验证，输入为雷达体素序列，输出概率栅格与 DEM：
- 占用概率图（D-S 融合）
- DEM 均值/方差
- 局部近障查询指标

```bash
cd /home/ps/zxj_workspace/src/4D-Radar

python diffusion_consistency_radar/scripts/streaming_map_update.py \
  --radar_voxel_dir Result/inference_results/cd_4step \
  --output_dir Result/inference_results/streaming_map \
  --dt 0.05 \
  --window_size 12 \
  --save_every 20
```

可选融合参数：

```bash
python diffusion_consistency_radar/scripts/streaming_map_update.py \
  --radar_voxel_dir Result/inference_results/cd_4step \
  --infrared_bev_dir /path/to/infrared_bev \
  --prior_dem /path/to/prior_dem.npy \
  --output_dir Result/inference_results/streaming_map_fused
```

输出文件：
- `streaming_metrics.csv`
- `map_snapshot_*.npz`
- `map_final.npz`

## 五、常见问题

### 1. CUDA 显存不足

```bash
# 降低样本数
python diffusion_consistency_radar/scripts/inference.py ... --num_samples 1

# 使用 CPU
python diffusion_consistency_radar/scripts/inference.py ... --device cpu
```

### 2. 模型文件找不到

请核对实际文件名：

```bash
ls -lh Result/train_results/vae/vae_best.pt
ls -lh Result/train_results/ldm/ldm_best.pt
ls -lh Result/train_results/cd/cd_best.pt
```

### 3. 推理速度慢

- LDM：减少 `--steps`，或切换 `--sampler euler`
- CD：优先 1 步或 2 步推理

### 4. 生成质量不佳

- LDM：增加 `--steps`
- CD：使用 2 步或 4 步
- 对比不同 `--occ_threshold` 观察点云稀疏程度

## 六、性能对比表

下表仅表示典型趋势，具体耗时取决于 GPU、输入规模与 I/O：

| 模型 | 步数 | 采样器 | 速度趋势 | 质量趋势 | 适用场景 |
|------|------|--------|----------|----------|----------|
| LDM | 40 | Heun | 较慢 | 较高 | 离线高质量生成 |
| LDM | 20 | Euler | 中等 | 中等 | 快速验证 |
| CD | 1 | Euler | 最快 | 中等 | 在线快速推理 |
| CD | 2 | Euler | 快 | 较高 | 实时折中 |
| CD | 4 | Euler | 中等 | 高 | 质量优先在线推理 |

## 七、批量推理

用于生成聚合样本文件（便于可视化或离线分析）：

```bash
python diffusion_consistency_radar/scripts/inference.py \
  --vae_ckpt Result/train_results/vae/vae_best.pt \
  --model_ckpt Result/train_results/ldm/ldm_best.pt \
  --model_type ldm \
  --steps 40 \
  --sampler heun \
  --num_samples 100 \
  --output_dir Result/inference_results/ldm_batch
```

```bash
python diffusion_consistency_radar/scripts/inference.py \
  --vae_ckpt Result/train_results/vae/vae_best.pt \
  --model_ckpt Result/train_results/cd/cd_best.pt \
  --model_type cd \
  --steps 1 \
  --sampler euler \
  --num_samples 100 \
  --output_dir Result/inference_results/cd_batch
```

## 八、输出数据格式

### 1. 批量样本模式输出

`<model_type>_samples_<steps>steps.npy`

```python
import numpy as np

data = np.load('Result/inference_results/cd/cd_samples_1steps.npy')
print(data.shape)  # (num_samples, 4, 32, 128, 128)
```

通道顺序：
- `0`: Occupancy
- `1`: Intensity
- `2`: Doppler
- `3`: Variance

### 2. 逐文件推理模式输出

- `*_pcl.npy`：点云 `(N, 4)`，列为 `x, y, z, intensity`
- `*_voxel.npy`：体素（启用 `--save_voxel`）
- `comparison_metrics.csv`：Chamfer 对比指标（启用对比时）

## 九、下一步

- 使用 `scripts/evaluate.py` 做系统化离线评估。
- 将推理输出接入 `scripts/streaming_map_update.py` 做在线地图更新验证。
- 基于实际场景调节 `steps/sampler/occ_threshold`，形成稳定参数配置。