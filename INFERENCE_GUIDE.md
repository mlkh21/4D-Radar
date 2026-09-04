# 推理使用指南

本文档说明当前 `withir` 分支的 Radar–IR 多模态推理、prediction 输出、离线评价和地图回放边界。网络与训练监督的完整代码审计见 [docs/current_architecture.md](./docs/current_architecture.md)。

## 1. 正式推理的定义

当前正式 prediction forward 的传感器输入是：

- 四通道 `radar_voxel`；
- 同一 frame ID 的真实 thermal IR；
- LiDAR-frame→thermal 的外参 `R/T` 和 thermal 内参 `K`；
- 与数据协议匹配的 VAE、LDM 或 CD checkpoint；
- checkpoint 内嵌的 Radar normalization；
- 与生成 checkpoint SHA-256 绑定的 occupancy threshold artifact。

**正式生成不需要 LiDAR、target voxel 或训练 observed mask。** 部署视图只包含 `radar_voxel` 与 `ir_image`。LiDAR/target 只在 prediction 已保存后用于离线评价。

正式部署会从 Radar endpoint 和 Radar→LiDAR 标定原点构造新的 deployment observed mask。这份 mask 与 prediction 一起落盘，供地图/评价合同使用；它不是生成模型的输入，也不是 LiDAR observed mask 的运行时替代输入。

## 2. 当前默认链的已知阻断点

默认 train/inference launcher 使用：

```text
PROTOCOL_TAG=formal_v2_80m_86p8_v1
training data=Data/NTU4DRadLM_Pre_formal_v2_80m_86p8_v1
deployment data=Data/NTU4DRadLM_Deploy_formal_v2_80m_86p8_v1
```

但默认 normalization artifact 内部仍是旧的 `radar_normalization_v1` 和 `occupied_voxel_equal_weight_total_variance`；当前 preprocessor 输出 `radar_point_count_field_validity_v2`，loader 要求 `field_valid_count_weighted_total_variance_v2`。这组默认输入会被 fail-closed 拒绝。

`preprocess-v2.sh` 实际生成 formal-v2.1，`preprocess-v3.sh` 生成 formal-v3；两者都没有自动改写默认训练/推理 tag。正式运行前必须确认以下身份来自同一条数据链：

- `PROTOCOL_TAG`；
- training/deployment 数据根；
- temporal split artifact；
- formal data protocol；
- Radar normalization 及 SHA-256；
- VAE → LDM → CD checkpoint 父链；
- validation threshold artifact。

不要因为文件名含 `v2` 就跳过 JSON 内部 protocol/aggregation 校验。

当前还有第二个独立阻断：三个 inference launcher 都把 checkpoint-chain CLI 指向 `diffusion_consistency_radar/scripts/diagnose_checkpoint_chain.py`，但当前 checkout 没有这个文件。核心校验逻辑位于 `diffusion_consistency_radar/checkpoint_chain.py`，测试目录也有诊断脚本，但两者都不是 launcher 当前声明的正式 CLI 路径。本文不把测试脚本当作生产替代品。

## 3. 环境与目录

使用项目环境：

```bash
conda activate Radar-Diffusion
```

以下命令均从仓库根目录执行，不依赖旧服务器的 `/home/ps/...` 路径。

默认 formal-v2 checkpoint 位置：

```text
Result/train_results/formal_v2_80m_86p8_v1/vae/vae_best.pt
Result/train_results/formal_v2_80m_86p8_v1/ldm/ldm_best.pt
Result/train_results/formal_v2_80m_86p8_v1/cd/cd_best.pt
Result/train_results/formal_v2_80m_86p8_v1/ldm/occupancy_threshold.json
Result/train_results/formal_v2_80m_86p8_v1/cd/occupancy_threshold.json
```

场景列表来自 `diffusion_consistency_radar/config/data_loading_config.yml` 的 `data.test`；当前配置为 `loop3`。

## 4. 运行前只读检查

先确认输入存在，不要直接开始生成：

```bash
TAG=formal_v2_80m_86p8_v1
TRAIN_ROOT="Result/train_results/${TAG}"
DEPLOY_ROOT="Data/NTU4DRadLM_Deploy_formal_v2_80m_86p8_v1"

test -f "${TRAIN_ROOT}/vae/vae_best.pt"
test -f "${TRAIN_ROOT}/ldm/ldm_best.pt"
test -f "${TRAIN_ROOT}/ldm/occupancy_threshold.json"
test -d "${DEPLOY_ROOT}/loop3/radar_voxel"
test -d "${DEPLOY_ROOT}/loop3/ir_image"
```

确认 launcher 要求的 checkpoint-chain CLI 是否存在：

```bash
test -f diffusion_consistency_radar/scripts/diagnose_checkpoint_chain.py
```

当前该检查会失败，这表示正式 launcher 尚不能继续。即使 checkpoint 文件都存在，也不能跳过父链校验后声称完成了正式部署验证。

确认 CD 文件是否齐全：

```bash
test -f "${TRAIN_ROOT}/cd/cd_best.pt"
test -f "${TRAIN_ROOT}/cd/occupancy_threshold.json"
```

校验 deployment dataset 收据和场景内容：

```bash
conda run -n Radar-Diffusion python \
  diffusion_consistency_radar/scripts/build_deployment_view.py validate \
  --dataset_dir "${DEPLOY_ROOT}" \
  --scene loop3
```

任一检查失败时，应修正路径或协议绑定，不要使用 legacy 开关绕过正式门禁。

## 5. 正式 launcher 的设计合同与当前阻断

### 5.1 LDM：40-step Heun

```bash
bash diffusion_consistency_radar/launch/inference_ldm.sh
```

当前会先因缺少 `scripts/diagnose_checkpoint_chain.py` 停止。恢复该文件后，设计流程才是：

该脚本会：

1. 校验 VAE/LDM checkpoint 父链和 threshold artifact；
2. 校验完整 deployment dataset；
3. 对 `data.test` 中每个场景逐文件加载 Radar、IR 和真实标定；
4. 从 `sigma_max` 高斯 latent 开始做 40-step Heun；
5. 保存 voxel、点云、不确定性和运行收据。

默认输出：

```text
Result/inference_results/<scene>_formal_v2_80m_86p8_v1_ldm_deploy/
```

### 5.2 CD：1-step

```bash
bash diffusion_consistency_radar/launch/inference_cd.sh
```

当前同样受缺失 checkpoint-chain CLI 阻断。

该脚本校验 VAE/LDM/CD 完整父链，并从 `sigma_max` 噪声执行一次 CD online model forward，再应用训练一致的 boundary parameterization。

默认输出：

```text
Result/inference_results/<scene>_formal_v2_80m_86p8_v1_cd_1step_deploy/
```

### 5.3 统一入口

```bash
bash diffusion_consistency_radar/launch/inference_uniified.sh
```

当前同样受缺失 checkpoint-chain CLI 阻断。

源码中的实际文件名是 `inference_uniified.sh`。它依次运行：

1. LDM 40-step Heun；
2. CD 1-step；
3. CD 权重的 4-step Euler 实验路径。

统一入口要求 LDM/CD 两个 threshold artifact 和完整 CD checkpoint chain 都存在；正式 CLI 恢复后，只想运行一个模型时应使用前两个独立入口。

CD 4-step 的真实行为需要特别说明：`RadarGenerator.generate()` 只有在 `model_type=cd AND steps=1` 时进入 `_cd_sample()`。当 `steps=4` 时实际走 `_ldm_sample()`，把 CD 权重输出当作 Euler denoiser，不会在每一步套用 CD boundary。当前代码没有证据支持“4-step 必然比 1-step 质量更高”的结论。

## 6. 成套覆盖非默认数据链

如果 checkpoint 和数据来自 formal-v2.1 或 formal-v3，至少要同时覆盖 tag 和 deployment root：

```bash
PROTOCOL_TAG=<与checkpoint一致的tag> \
PREPROCESSED_ROOT=<对应的deployment数据根> \
CALIBRATION_DIR="Data/config" \
bash diffusion_consistency_radar/launch/inference_ldm.sh
```

CD 使用同样的三个变量。checkpoint 目录会由 `PROTOCOL_TAG` 推导为 `Result/train_results/<tag>/`；如 threshold artifact 不在默认阶段目录，还需分别设置：

```text
LDM_THRESHOLD_ARTIFACT
CD_THRESHOLD_ARTIFACT
```

仅覆盖 `PREPROCESSED_ROOT` 而沿用旧 checkpoint/normalization/threshold，不是合法迁移。

注意：`launch/evaluate_inference.sh` 当前把 formal-v2 tag、training root 和 raw root写成固定值，不消费上述 `PROTOCOL_TAG/PREPROCESSED_ROOT` 覆盖。非默认数据链应使用第 10 节的直接 evaluator 命令。

## 7. 直接调用 Python 入口

直接 Python 调用可以检查模型本体的数据路径，但会绕过 launcher 当前缺失的父链预检，因此只能视为手工诊断/恢复验证，不能替代完整正式入口。应尽量保留 inference 本身的严格输入门禁。LDM 单场景示例：

```bash
SCENE=loop3
TAG=formal_v2_80m_86p8_v1
TRAIN_ROOT="Result/train_results/${TAG}"
DEPLOY_ROOT="Data/NTU4DRadLM_Deploy_formal_v2_80m_86p8_v1"
OUTPUT_ROOT="Result/inference_results/${SCENE}_${TAG}_ldm_deploy_manual"

conda run -n Radar-Diffusion python \
  diffusion_consistency_radar/scripts/inference.py \
  --vae_ckpt "${TRAIN_ROOT}/vae/vae_best.pt" \
  --model_ckpt "${TRAIN_ROOT}/ldm/ldm_best.pt" \
  --model_type ldm \
  --steps 40 \
  --sampler heun \
  --radar_voxel_dir "${DEPLOY_ROOT}/${SCENE}/radar_voxel" \
  --deployment_scene_dir "${DEPLOY_ROOT}/${SCENE}" \
  --calibration_dir "Data/config" \
  --threshold_artifact "${TRAIN_ROOT}/ldm/occupancy_threshold.json" \
  --seed 42 \
  --require_real_ir \
  --save_voxel \
  --save_pointcloud \
  --save_uncertainty \
  --output_dir "${OUTPUT_ROOT}" \
  --device cuda
```

CD 1-step 只需将 model 和采样参数替换为：

```text
--model_ckpt "${TRAIN_ROOT}/cd/cd_best.pt"
--model_type cd
--steps 1
--sampler euler
--threshold_artifact "${TRAIN_ROOT}/cd/occupancy_threshold.json"
```

正式模式的 seed 必须是非负整数。输出目录应使用新的独立路径，避免覆盖已有运行收据和 prediction。

## 8. 模型内部推理路径

逐帧执行顺序为：

```text
Radar NPZ
  → crop / count-aware resize / checkpoint normalization
  → radar tensor (1,4,32,128,128)

IR NPY + R/T/K
  → IR 2D backbone
  → voxel-center projection + frustum mask

Radar16 + projected IR32 + confidence1
  → IR gate
  → fusion Conv 得到 16ch condition
  → 插值到 latent (16,32,32)
  → 将 4ch noisy latent 加到前四通道
  → 3D UNet 输出 4ch denoised latent
  → VAE decoder
  → 4ch prediction
```

只有 prediction ch0 按 checkpoint occupancy protocol 做 sigmoid。ch1–3 保持 VAE decoder 原值。

## 9. 输出文件与通道

正式逐文件输出目录包含：

| 文件 | 内容 |
|---|---|
| `<frame>_voxel.npy` | `(4,Z,X,Y)` prediction |
| `<radar_filename>_pcl.npy` | `(N,4)` 点云 |
| `<radar_filename>_uncertainty.npy` | 可用时保存独立不确定性 |
| `<frame>_observed_mask.npy` | Radar ray 构造的部署可观测域 |
| `inference_runtime.csv` | 逐帧耗时和运行汇总 |
| `inference_runtime.log` | 可读日志 |
| `inference_run.json` | checkpoint、数据、threshold、prediction 与 mask 内容收据 |

prediction voxel 通道：

| 通道 | 语义 |
|---:|---|
| 0 | occupancy probability |
| 1 | LiDAR return-strength 的生成重建值 |
| 2 | Radar-neighborhood Doppler 的生成重建值 |
| 3 | Doppler-valid target 的生成重建值 |

ch3 **不是 uncertainty/variance**。不确定性由模型的 physical + learned variance head 单独输出。

点云文件的列为：

```text
x, y, z, prediction_ch1
```

坐标是 occupancy 超过 threshold 的 voxel center；第四列取 prediction ch1。

## 10. 生成后离线评价

正式生成和正式评价是两个阶段。默认 formal-v2 可运行：

```bash
bash diffusion_consistency_radar/launch/evaluate_inference.sh ldm
bash diffusion_consistency_radar/launch/evaluate_inference.sh cd
bash diffusion_consistency_radar/launch/evaluate_inference.sh cd4
```

该入口不重新加载或运行生成模型，而是验证已保存 prediction、`inference_run.json`、observed mask、training manifest 和帧配对，然后输出：

```text
evaluation_frames.csv
evaluation_summary.json
```

`evaluation_summary.json` 的正式协议是 `formal_saved_prediction_observed_domain_evaluation_v1`。Raw LiDAR Chamfer 是辅助诊断，不属于正式 observed-domain 指标。

对于 formal-v2.1/v3 或其他成套 override，直接指定路径：

```bash
conda run -n Radar-Diffusion python \
  diffusion_consistency_radar/scripts/evaluate_saved_predictions.py \
  --pred_voxel_dir <已保存prediction目录> \
  --radar_voxel_dir <同协议training场景/radar_voxel> \
  --target_voxel_dir <同协议training场景/target_voxel> \
  --output_dir <新的评价输出目录> \
  --run_metadata_path <prediction目录/inference_run.json> \
  --target_threshold 0.5
```

若只需要额外 raw LiDAR Chamfer，可再提供：

```text
--raw_livox_dir <raw场景/livox_lidar>
--lidar_index_file <raw场景/lidar_index_sequence.txt>
```

不要在正式生成阶段添加 `--compare_with_lidar`；先保存 prediction，再由 evaluator 读取 LiDAR/target。

## 11. Threshold 规则

正式推理不接受手工 `--occ_threshold`。threshold 必须来自 validation artifact，并与生成 checkpoint SHA-256 绑定：

```text
--threshold_artifact <occupancy_threshold.json>
```

`--empty_fallback_topk` 默认是 0。启用 top-k 会改变空预测的点云表现，只适合明确标注的诊断，不应静默用于正式指标。

历史脚本 `sweep_occ_threshold.py` 和手工 `--occ_threshold` 仅可用于 diagnostic/legacy 分析，不能替代正式 threshold receipt。

## 12. 非正式聚合样本模式

当不提供 `--radar_voxel_dir` 时，`inference.py` 会保存：

```text
<model_type>_samples_<steps>steps.npy
```

但对当前多模态 checkpoint，`condition=None` 会构造零 Radar voxel，并由 metadata fallback 提供 mock IR/标定。这不是实际 Radar+IR 条件预测，也不能用于正式部署或正式评价。

`--use_condition` 可从 Dataset 取一个条件样本，但只有同时正确提供 matching dataset 和 multimodal metadata 时才有诊断意义。推荐的真实场景推理仍是第 5 节的逐文件正式入口。

## 13. 概率地图回放

`streaming_map_update.py` 消费已保存 prediction 序列，输出 D-S occupancy、DEM 和风险查询结果。它是离线文件回放原型。

最小 legacy 回放示例：

```bash
conda run -n Radar-Diffusion python \
  diffusion_consistency_radar/scripts/streaming_map_update.py \
  --radar_voxel_dir <prediction目录> \
  --radar_voxel_layout czxy \
  --output_dir <新的map输出目录> \
  --dt 0.05 \
  --window_size 12 \
  --save_every 20
```

该命令没有真实 body→local pose，只能视为 legacy/诊断回放。

严格 `--formal_mapping` 额外要求：

```text
--observed_mask_dir
--inference_run
--pose_file
--lidar_to_body_calib
```

严格模式禁止读取离线 target，也拒绝未绑定 frame receipt 的 uncertainty、IR-BEV、dynamic evidence 和 prior DEM。即使 formal mapping 校验通过，也只证明离线输入合同成立，不证明 ROS1/PX4 实时闭环。

## 14. 常见失败

### Radar normalization aggregation 不一致

症状通常包含：

```text
Radar normalization resize aggregation 与输入统计协议不一致
```

原因是 normalization artifact 与 Radar NPZ 的 statistics protocol 不属于同一数据链。应重新选择匹配 artifact 并同步 checkpoint/data protocol，不要启用 `--allow_legacy_radar_units` 绕过正式检查。

### 缺少真实 IR 或热相机标定

`--require_real_ir` 要求每帧 `ir_image/<frame>_ir.npy`、`calib_livox_to_thermal.txt` 和 thermal intrinsics 均有效。正式入口在创建输出目录前 preflight 全部帧，任一帧失败都会终止。

### checkpoint chain 或 threshold 不匹配

正式 LDM/CD checkpoint 必须携带 data protocol、Radar normalization、multimodal model config 和父 checkpoint 身份；CD 还必须有 consistency receipt。threshold artifact 必须绑定当前生成 checkpoint。

当前 launcher 所需的 `scripts/diagnose_checkpoint_chain.py` 还缺失；这是代码入口缺口，不是通过改路径或 legacy 参数可解决的数据错误。

### 输出目录已存在

使用新的输出目录。不要删除或覆盖既有 prediction、日志、checkpoint 或评价结果；需要重跑时采用新的实验 tag/目录。

### CUDA 显存不足

正式推理固定为逐文件 batch 1。可以在诊断时使用 CPU，但不要把 CPU smoke 的耗时当作目标 GPU 性能。没有验证依据时不要通过减少正式 LDM steps 或改 sampler 来宣称等价质量。

## 15. 能力边界

- 当前网络是单帧 Radar–IR 条件生成，不做时序融合。
- 正式数据默认 `velocity_mode=none`，没有 Radar egomotion Doppler 补偿。
- LiDAR-free prediction 接口已经存在；这不等于无 LiDAR 训练监督。
- CD 1-step 是当前明确的 consistency 部署路径；CD 4-step 是实验性 Euler 路径。
- 地图更新仍是离线回放；ROS1 service/action、PX4 HIL 与控制器闭环未由当前代码路径证明。
- 实际 checkpoint 是否存在、能否加载以及性能/精度必须在目标服务器上现场验证。
