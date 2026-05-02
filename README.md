# 4D Radar Diffusion

面向 4D Radar 点云稠密化的离线训练、推理、诊断与对比仓库。当前主流程围绕 NTU4DRadLM 数据集展开，支持：

- 预处理：原始 Radar / LiDAR 点云对齐、体素化、训练目标构建
- 训练：`VAE -> LDM -> CD`
- 推理：逐文件生成点云并输出指标
- 诊断：生成质量分析、阈值扫描、可视化对比
- 隔离测试：`test/mini-test/` 下的小规模快速验证流程

## 项目结构

```text
NTU4DRadLM_pre_processing/          # 原始数据预处理
diffusion_consistency_radar/
  cm/                               # 模型、损失、数据加载
  config/                           # YAML 配置
  launch/                           # 正式训练 / 推理 / 诊断入口
  scripts/                          # 训练、推理、评估、可视化脚本
test/
  mini-test/                        # 隔离的小规模 train / infer / diagnose 流程
Data/                               # 原始数据与预处理数据
Result/                             # 正式训练和推理输出
```

## 数据约定

- 原始数据目录：`Data/NTU4DRadLM_Raw/<scene>/`
- 预处理数据目录：`Data/NTU4DRadLM_Pre/<scene>/`
- 默认训练场景 / 测试场景：见 [data_loading_config.yml](./diffusion_consistency_radar/config/data_loading_config.yml)
- 默认点云范围：`[0, -20, -6, 120, 20, 10]`
- 原始体素分辨率：`0.2m x 0.2m x 0.2m`

当前训练输入输出的通道定义：

- `radar_voxel`: `Occ / Int / Dop / Var`
- `target_voxel`: `Occ / Int / Dop / Mask`

其中：

- `Occ` 和 `Mask` 在训练前会使用保结构的缩放逻辑
- `Dop` 监督不再要求 Radar 和 LiDAR 在同一细体素严格重叠，而是在 LiDAR 占据位置的局部 Radar 邻域内聚合

## 环境准备

仓库默认按已有 Conda 环境使用，常见环境名是 `Radar-Diffusion`。最少需要：

```bash
conda activate Radar-Diffusion
pip install -e diffusion_consistency_radar
```

如果你只做脚本检查，也可以直接使用系统 Python；但训练 / 推理 / 诊断通常需要项目环境中的 `torch`、`scipy`、`matplotlib`、`pypatchworkpp`。

## 正式流程

### 1. 预处理

从原始 Radar / LiDAR 数据生成 `radar_voxel`、`lidar_voxel`、`target_voxel`：

```bash
python NTU4DRadLM_pre_processing/NTU4DRadLM_pre_processing.py
```

关键逻辑在：

- [NTU4DRadLM_pre_processing.py](./NTU4DRadLM_pre_processing/NTU4DRadLM_pre_processing.py)
- 标定文件：[calib_radar_to_livox.txt](./Data/config/calib_radar_to_livox.txt)

注意：

- 预处理默认会做 Radar -> LiDAR 坐标变换
- LiDAR 会经过地面滤除
- 若标定文件缺失，脚本会直接报错；只有显式设置 `ALLOW_IDENTITY_CALIB=1` 才允许回退单位矩阵

### 2. 训练

正式训练入口是 [train_unified.sh](./diffusion_consistency_radar/launch/train_unified.sh)。

只训练 VAE：

```bash
bash diffusion_consistency_radar/launch/train_unified.sh vae
```

只训练 LDM：

```bash
bash diffusion_consistency_radar/launch/train_unified.sh ldm
```

蒸馏 CD：

```bash
bash diffusion_consistency_radar/launch/train_unified.sh cd
```

完整流程：

```bash
bash diffusion_consistency_radar/launch/train_unified.sh all
```

默认输出目录：

- `Result/train_results/vae/`
- `Result/train_results/ldm/`
- `Result/train_results/cd/`

训练配置文件：

- 主配置：[default_config.yaml](./diffusion_consistency_radar/config/default_config.yaml)
- 训练场景配置：[data_loading_config.yml](./diffusion_consistency_radar/config/data_loading_config.yml)

### 3. 推理

LDM 推理：

```bash
bash diffusion_consistency_radar/launch/inference_ldm.sh
```

CD 推理：

```bash
bash diffusion_consistency_radar/launch/inference_cd.sh
```

这两个脚本会按 `data_loading_config.yml` 里的 `data.test` 场景逐文件推理，并输出：

- 生成点云：`Result/inference_results/<scene>_ldm_eval/` 或 `..._cd_eval/`
- 指标文件：`inference_metrics.csv`
- 运行日志：`inference_runtime.log`

如果你想直接调用 Python 入口：

```bash
python diffusion_consistency_radar/scripts/inference.py \
  --vae_ckpt Result/train_results/vae/vae_best.pt \
  --model_ckpt Result/train_results/ldm/ldm_best.pt \
  --model_type ldm \
  --steps 40 \
  --sampler heun \
  --radar_voxel_dir Data/NTU4DRadLM_Pre/loop3/radar_voxel \
  --save_pointcloud \
  --compare_with_lidar \
  --raw_livox_dir Data/NTU4DRadLM_Raw/loop3/livox_lidar \
  --lidar_index_file Data/NTU4DRadLM_Raw/loop3/lidar_index_sequence.txt \
  --output_dir Result/inference_results/loop3_ldm_eval
```

### 4. 诊断与对比

生成质量诊断：

```bash
bash diffusion_consistency_radar/launch/diagnose.sh
```

或直接调用：

```bash
python diffusion_consistency_radar/scripts/diagnose_generation_quality.py \
  --radar_voxel_dir Data/NTU4DRadLM_Pre/loop3/radar_voxel \
  --target_voxel_dir Data/NTU4DRadLM_Pre/loop3/target_voxel \
  --pred_dir Result/inference_results/loop3_ldm_eval \
  --output_dir Result/diagnosis_results/loop3_ldm_eval \
  --max_files 20 \
  --pred_kind pcl \
  --occ_threshold 0.1
```

输出包括：

- `frames/*.png`
- `diagnosis_metrics.csv`
- `diagnosis_report.md`

Radar / LiDAR 结果图像对比：

```bash
bash diffusion_consistency_radar/launch/compare.sh
```

阈值扫描：

```bash
python diffusion_consistency_radar/scripts/sweep_occ_threshold.py --help
```

点云指标评估：

```bash
python diffusion_consistency_radar/scripts/evaluate.py \
  --pred_path <pred_dir> \
  --gt_path <gt_dir> \
  --output_path <output_json>
```

## Mini Test

`test/mini-test/` 是隔离的小规模验证区，用来快速做 train / infer / diagnose，不污染正式 `Result/` 目录。

文档见：

- [test/README.md](./test/README.md)
- [test/mini-test/README.md](./test/mini-test/README.md)

常用命令：

```bash
bash test/mini-test/train_minimal.sh all
bash test/mini-test/inference_minimal.sh ldm
bash test/mini-test/diagnose_minimal.sh
```

默认输出位置：

- `test/mini-test/train_results_mini/`
- `test/mini-test/inference_results_mini/`
- `test/mini-test/diagnostics/`

## 当前推荐使用的方法

如果你要做正式实验，推荐顺序是：

1. 先运行预处理，确保 `target_voxel` 基于当前逻辑重新生成
2. 训练 `vae`
3. 训练 `ldm`
4. 运行 `inference_ldm.sh`
5. 运行 `diagnose.sh`
6. 需要更快的推理时再考虑 `cd`

如果你只是想验证逻辑是否跑通，推荐直接走 `test/mini-test/`。

## 已知说明

- 当前主流程是离线训练和离线推理，不包含 ROS 实时闭环
- `launch/` 目录下只应视为正式入口；快速实验请放在 `test/`
- `.npy`、`.npz`、训练结果和推理结果默认不会纳入 Git 跟踪
- 某些历史脚本名或旧文档中提到的入口，已经不再是当前推荐路径
