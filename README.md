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
- 运动补偿默认是 `none`，不会静默使用固定 `50 m/s`；如确有可靠的机体系速度，可显式设置 `VELOCITY_MODE=fixed`，或使用 `VELOCITY_MODE=recorded` 加载 `timestamp,vx,vy,vz` 速度表
- 速度表必须与 Radar 文件名时间戳使用相同秒单位，默认最近邻时间差不得超过 `0.02s`；速度向量默认在 Radar 坐标系，预处理只用标定旋转转换到最终共享坐标系，不会把平移量加入速度
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

- `Result/train_results/formal_p1_04_full120_86p8_v1/vae/`
- `Result/train_results/formal_p1_04_full120_86p8_v1/ldm/`
- `Result/train_results/formal_p1_04_full120_86p8_v1/cd/`

正式入口固定使用：

- 数据：`Data/NTU4DRadLM_Pre_sensor_aware_p1_04_candidate/`
- Radar normalization：`radar_normalization_garden_32x128x128_full120_86p8_v1.json`
- Doppler 物理量程：`86.8 m/s`

已有结果目录默认拒绝隐式续训。确认恢复的是同一正式协议后，显式执行：

```bash
ALLOW_RESUME=1 bash diffusion_consistency_radar/launch/train_unified.sh <vae|ldm|cd|all>
```

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

这两个脚本会先校验正式 VAE/LDM/CD checkpoint 链与 candidate manifest，再按
`data_loading_config.yml` 里的 `data.test` 场景逐文件推理，并输出：

- LDM：`Result/inference_results/<scene>_formal_p1_04_full120_86p8_v1_ldm_deploy/`
- CD：`Result/inference_results/<scene>_formal_p1_04_full120_86p8_v1_cd_1step_deploy/`
- 运行协议：`inference_run.json`、`inference_runtime.csv`；正式 `inference_run.json`
  同时逐帧绑定 prediction voxel 与 observed mask 的文件名、SHA-256、CZXY shape
  和 dtype，严格地图入口不接受缺少任一内容收据的旧推理目录
- 预测产物：`*_voxel.npy`、`*_pcl.npy` 和可用的 `*_uncertainty.npy`

如果你想直接调用 Python 入口：

```bash
python diffusion_consistency_radar/scripts/inference.py \
  --vae_ckpt Result/train_results/formal_p1_04_full120_86p8_v1/vae/vae_best.pt \
  --model_ckpt Result/train_results/formal_p1_04_full120_86p8_v1/ldm/ldm_best.pt \
  --model_type ldm \
  --steps 40 \
  --sampler heun \
  --radar_voxel_dir Data/NTU4DRadLM_Pre_sensor_aware_p1_04_candidate/loop3/radar_voxel \
  --require_real_ir \
  --save_voxel \
  --save_pointcloud \
  --save_uncertainty \
  --output_dir Result/inference_results/loop3_formal_p1_04_full120_86p8_v1_ldm_deploy
```

正式指标评价与生成解耦，在预测保存完成后执行：

```bash
bash diffusion_consistency_radar/launch/evaluate_inference.sh ldm
```

### 4. 诊断与对比

生成质量诊断：

```bash
bash diffusion_consistency_radar/launch/diagnose.sh
```

或直接调用：

```bash
python diffusion_consistency_radar/scripts/diagnose_generation_quality.py \
  --radar_voxel_dir Data/NTU4DRadLM_Pre_sensor_aware_p1_04_candidate/loop3/radar_voxel \
  --target_voxel_dir Data/NTU4DRadLM_Pre_sensor_aware_p1_04_candidate/loop3/target_voxel \
  --pred_dir Result/inference_results/loop3_formal_p1_04_full120_86p8_v1_ldm_deploy \
  --output_dir Result/diagnosis_results/loop3_formal_p1_04_full120_86p8_v1_ldm_deploy \
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

## Formal v2 训练入口

当前正式训练合同为 0--80 m、Doppler scale 86.8 m/s、持久化 observed mask、
temporal purge split、train-only normalization 和 Radar point-count/Doppler-validity
统计。旧 full120 数据、artifact 与 mini checkpoint 仅用于 legacy/diagnostic，不能与
formal v2 结果混用。

正式 VAE 的准备与启动顺序如下：

```bash
# 1. fresh 全量重建；脚本会拒绝覆盖任何已有正式输出
bash NTU4DRadLM_pre_processing/preprocess-v2.sh

# 2. 记录新 normalization artifact 的固定身份
sha256sum \
  diffusion_consistency_radar/config/radar_normalization_garden_32x128x128_80m_train80_purge3s_86p8_v2.json

# 3. 在训练机器上只读预检；不会写训练配置，也不会启动 GPU 训练
EXPECTED_ARTIFACT_SHA256=<上一步的64位哈希> \
CUDA_DEVICES=0 \
FORMAL_EPOCHS=20 \
PREFLIGHT_ONLY=1 \
conda run -n Radar-Diffusion bash \
  diffusion_consistency_radar/launch/train_unified.sh vae

# 4. 只在具备长期训练条件的服务器上启动正式 VAE
EXPECTED_ARTIFACT_SHA256=<同一个64位哈希> \
CUDA_DEVICES=0 \
FORMAL_EPOCHS=20 \
conda run -n Radar-Diffusion bash \
  diffusion_consistency_radar/launch/train_unified.sh vae
```

`CUDA_DEVICES` 接受 `0` 或 `0,1` 形式。8 GB RTX 4070 Laptop 只建议执行预检和
短时 CPU/接口验证，不建议承担 formal v2 全量 VAE 长训练。已有同协议结果时 launcher
默认拒绝覆盖；只有确认 checkpoint 与协议一致后才能显式设置 `ALLOW_RESUME=1`。
正式服务器 launcher 固定 VAE/LDM/CD 各 20 epoch，并显式删除任何 mini 帧限制；garden
完整 temporal split 为 3210 train / 774 validation。正式阶段仍应按 `vae → ldm → cd`
顺序执行和验收，不能把笔记本的 400/100 checkpoint 接入正式链。

## Mini Test

`test/mini-test/` 是隔离的小规模验证区，用来快速做 train / infer / diagnose，不污染正式 `Result/` 目录。

文档见：

- [test/README.md](./test/README.md)
- [test/mini-test/README.md](./test/mini-test/README.md)

常用命令：

```bash
# 当前 formal v2：先只读预检，再由用户显式启动 8/4 帧 VAE mini
MINI_PREFLIGHT_ONLY=1 bash test/mini-test/run_formal_mini_8gb.sh vae
bash test/mini-test/run_formal_mini_8gb.sh vae

# 1 epoch smoke 验收后，可在独立目录预检/运行 fresh 3 epoch VAE
MINI_PREFLIGHT_ONLY=1 bash test/mini-test/run_formal_mini_8gb.sh vae short_train
bash test/mini-test/run_formal_mini_8gb.sh vae short_train

# RTX 4070 Laptop 中型质量筛查：固定 400 train + 100 validation、每阶段 20 epoch
MINI_PREFLIGHT_ONLY=1 bash test/mini-test/run_formal_mini_8gb.sh vae medium_train
bash test/mini-test/run_formal_mini_8gb.sh vae medium_train

# 历史 legacy mini
bash test/mini-test/train_minimal.sh all
bash test/mini-test/inference_minimal.sh ldm
bash test/mini-test/diagnose_minimal.sh
```

默认输出位置：

- `test/result/formal_mini_v2_80m_8gb_v1/`（8 GB formal v2 mini）
- `test/result/formal_mini_v2_80m_8gb_short_v1/`（fresh 3 epoch VAE short profile）
- `test/result/formal_medium_v2_80m_laptop_500f_20ep_v2/`（RTX 4070 Laptop 500 帧中型筛查；稳定 CUDA allocator）
- `test/result/formal_medium_v2_80m_laptop_500f_20ep_v1/`（失败现场，只用于 allocator 诊断，不续训）
- `test/mini-test/train_results_mini/`
- `test/mini-test/inference_results_mini/`
- `test/mini-test/diagnostics/`

## 当前推荐使用的方法

如果你要做正式实验，推荐顺序是：

1. 运行 `preprocess-v2.sh`，生成正式 training/deployment 数据与全部 artifact
2. 记录 normalization SHA-256，并用 `PREFLIGHT_ONLY=1` 完成无训练验收
3. 在服务器训练 `vae`
4. VAE 验收后训练 `ldm`
5. 使用 deployment v3 数据运行 `inference_ldm.sh`
6. 运行独立诊断与评价入口
7. 需要更快推理时再单独训练 `cd`

如果你只是想验证当前 formal v2 数据合同，可先使用预检和 `test/unit/` 聚焦测试；
需要验证 backward/checkpoint 时再分阶段运行 8 GB mini。它直接读取 0--80 m 正式数据，
按正式 split 取每场景 8 个 train 和 4 个 validation 帧，并写出
`formal_mini_chain_v2`。该 checkpoint 只用于接口 smoke，不能替代正式训练结果或进入
正式部署链。完整保护门禁见 [test/mini-test/README.md](./test/mini-test/README.md)。
formal mini 的 LDM/CD 无训练预检会在创建配置和输出前验证父 checkpoint 的
stage/protocol/data identity；short VAE 的后续阶段必须显式复用同一 short 结果根。
`medium_train` 是独立的 400/100、20 epoch 链，三阶段均需使用该 profile 和同一结果根。
VAE/LDM 会逐 epoch 使用 100 帧 validation；当前 CD 训练器只消费 400 帧 train，100 帧
留出集用于 CD 完成后的独立推理/评价。该中型结果可用于初步判断训练趋势和指标是否接近
需求，但不能替代服务器 full split 的正式训练与评价。

## 已知说明

- 当前主流程是离线训练和离线推理，不包含 ROS 实时闭环
- `launch/` 目录下只应视为正式入口；快速实验请放在 `test/`
- `.npy`、`.npz`、训练结果和推理结果默认不会纳入 Git 跟踪
- 某些历史脚本名或旧文档中提到的入口，已经不再是当前推荐路径
