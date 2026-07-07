# Findings

## Recovered Context
- Prior work focused on NTU4DRadLM radar/LiDAR preprocessing, calibration, and inference diagnostics.
- Earlier checks suggested roll changes can improve dz but may worsen or leave dy unresolved, so global centroid offsets are likely misleading.
- The next proposed step was "shared visible region evaluation": compare only common/nearby observable structure using radar-to-LiDAR nearest-neighbor distances, distance and height bands, non-ground filtering, and BEV grid IoU.
- Existing generated/mentioned files include `test/check_radar_axis_conventions.py`, `test/alignment_sanity_check.py`, `test/compare_voxel_triplets.py`, and `test/generate_interactive_raw_compare.py`.

## Working Hypothesis
dy may be dominated by radar/LiDAR point distribution, FOV, effective detections, and ground filtering differences rather than a simple extrinsic translation/rotation error.

## Implemented Diagnostic
- Added `test/shared_visibility_eval.py` to evaluate voxelized radar/lidar/target overlap using nearest-neighbor distances, match ratios, range bins, z-min filters, and BEV occupancy IoU.
- Added `test/test_shared_visibility_eval.py` with synthetic checks for nearest-neighbor match ratios and BEV IoU.
- Runtime validation now passes in the Ubuntu/Radar-Diffusion environment.

## Loop3 Shared Visibility Results

All runs used 120 shared frames and default bands: `x0_20`, `x20_40`, `x40_80`, `x80_120`, with `z_min=-6,-1,0`.

Near-range `radar_vs_lidar`, `x0_20`, `z_min=-1`:

| Dataset | nn_mean | match_ratio_2 | BEV IoU |
| --- | ---: | ---: | ---: |
| `Data/NTU4DRadLM_Pre/loop3` | 1.2539 m | 0.8149 | 0.3785 |
| `Data/NTU4DRadLM_Pre_alignfix/loop3` | 1.1580 m | 0.7973 | 0.2637 |
| `Data/NTU4DRadLM_Pre_radarframe/loop3` | 1.2682 m | 0.8235 | 0.3931 |

Key observations:

- Removing low points improves the original near-range radar/LiDAR comparison: for original `x0_20`, `z_min=-6` has `nn_mean=1.4344`, while `z_min=-1` improves to `nn_mean=1.2539` and `match_ratio_2=0.8149`.
- Long range is much worse for all variants. Original `x80_120`, `z_min=-1` has `nn_mean=3.6329`, `match_ratio_2=0.2646`, and `BEV IoU=0.0162`.
- `alignfix` does not provide a clear improvement. It lowers near-range `nn_mean` slightly, but worsens BEV IoU and long/mid-range match ratios.
- `radarframe` is close to original and slightly better on near-range BEV IoU/match ratio, but it does not remove the global distribution mismatch.

## Conclusion

The evidence does not support a simple "calibration is inverted" or "single axis convention is wrong" explanation. The large global dy/dz is more likely dominated by radar/LiDAR effective point distribution, FOV/range sparsity, low-z/ground handling, and sensor modality differences. For training/evaluation, prefer shared-visible-region metrics and range/height-banded diagnosis over global centroid offsets alone.

## Recommended Next Direction

The project goal is airborne obstacle map construction and scene map update, so the next step should shift from pure point-cloud reconstruction to task-oriented occupancy supervision:

1. Build a "sensor-aware target" protocol:
   - Keep the original LiDAR target as a reference only.
   - Add a filtered target for training/evaluation that removes low-z/ground-heavy regions and optionally limits supervision to radar-reachable/shared-visible regions.
   - Report metrics by range and height band instead of only global Chamfer.

2. Update preprocessing/evaluation before expensive retraining:
   - Add configurable `z_min`, range bins, and optional shared-visible mask generation.
   - Save metadata describing the target policy used for each preprocessed dataset.
   - Make inference reports include near-range obstacle metrics, far-range metrics, BEV IoU, occupancy precision/recall, and old global Chamfer for comparison.

3. Retrain only after the protocol is fixed:
   - First train VAE on the new target policy and check reconstruction.
   - Then train LDM.
   - Finally distill CD.
   - Compare old-vs-new on both point metrics and downstream map update metrics.

4. Connect to the final map-update task:
   - Use generated occupancy as a probabilistic observation, not as perfect LiDAR.
   - Add confidence decay by range/sparsity and stronger uncertainty for far-range predictions.
   - Evaluate map snapshots with obstacle recall, false-positive occupancy, update latency, and memory/runtime.

## Implemented Sensor-Aware Target Utility

Added `NTU4DRadLM_pre_processing/sensor_aware_target.py`:

- `SensorAwareTargetPolicy`: stores the target policy.
- `build_sensor_aware_target`: filters LiDAR occupancy by height/range and optional radar-visible neighborhood.
- `build_scene_targets`: creates a training-ready scene directory with `radar_voxel`, generated `target_voxel`, and `target_policy.json`.
- `build_dataset_targets`: applies the same policy to selected scenes under a dataset root.

The utility keeps the project channel convention:

- Target channel 0: filtered LiDAR occupancy.
- Target channel 1: filtered LiDAR intensity.
- Target channel 2: local radar Doppler aggregated around kept LiDAR cells.
- Target channel 3: Doppler-valid mask.

Smoke dataset generated:

```bash
conda run -n Radar-Diffusion python -m NTU4DRadLM_pre_processing.sensor_aware_target \
  --input_root Data/NTU4DRadLM_Pre \
  --output_root Data/NTU4DRadLM_Pre_sensor_aware \
  --scenes loop3 \
  --z_min -1.0 \
  --x_max 80.0 \
  --require_radar_visibility \
  --radar_visibility_radius 2 \
  --max_files 120
```

Result:

- `Data/NTU4DRadLM_Pre_sensor_aware/loop3` contains 120 frames.
- Training loader successfully reads 120 samples and returns `(4, 32, 128, 128)` target/radar tensors.
- Frame `000000`: original target occupancy `4709`, sensor-aware target occupancy `659`, Doppler mask `228`, radar occupancy `519`.

This confirms the new target policy strongly suppresses low-z / far / non-radar-visible LiDAR supervision before retraining.

## Airborne Multimodal Refactor Findings

- The requested refactor is feasible, but the full physical grid `(600, 200, 80)` is too large for routine unit tests and likely too expensive for default LDM training. The implementation keeps `DualModalityProjectionLayer` defaulting to that physical shape, while `OptimizedLDMTrainer` uses configurable `ldm.fusion_voxel_shape` with a throughput-oriented default `(32, 128, 128)` that matches the preprocessed training tensor.
- The current server environment does not provide a usable `torchvision` import, so `IR2DFeatureExtractor` uses ResNet-18 when available and falls back to a small CNN otherwise. This keeps tests and training code runnable without silently removing the IR pathway.
- VAE latent size can be smaller than the radar voxel size. The fusion network therefore downsamples fused radar+IR features to the passed `noised_latent` spatial size during LDM training, avoiding shape mismatch in the denoiser loss.
- LDM checkpoints saved before this refactor contain the old bare UNet state dict. They should be treated as architecture-incompatible unless a dedicated migration script strips or remaps wrapper keys.

## Offline Loop Closure Implementation Findings

- `inference.py` now supports both legacy 8-channel UNet checkpoints and new `CompleteDualModalityPerceptionNet` checkpoints by inspecting checkpoint state-dict keys. New multimodal inference can use sidecar `ir_image/*_ir.npy` and calibration metadata, with mock fallback when missing.
- Formal task metrics were promoted into `diffusion_consistency_radar/cm/evaluation_metrics.py` and connected to inference/diagnosis reporting. The first production summary focuses on near obstacle metrics (`x=0-20m`, `z>=-1m`) because this is the most relevant band for airborne local obstacle-map updates.
- `dataset_loader.py` now marks `is_mock_ir` and `is_mock_calib`, and scene-level `preprocess_policy.json` is loaded into `meta_dict`. This makes mock/fallback data explicit instead of hidden.
- `probabilistic_mapping.py` now converts Doppler variance and range into a per-cell observation reliability map. High variance and far-range cells produce lower belief and higher map/DEM uncertainty.
- `streaming_map_update.py` now ignores `*_pcl.npy` files when looking for voxel inputs; this fixed the smoke-test failure where point clouds were incorrectly parsed as 4D voxels.
- Mini shell helpers should not use `conda run -n Radar-Diffusion python - <<'PY'` for heredoc snippets. In this environment the stdin script can be swallowed, causing config generation to silently leave old YAML values in place. Use system `python3` for lightweight YAML helpers and reserve `conda run` for actual PyTorch training/inference commands.
- A bad mini CD smoke run exposed this issue by ignoring `MINI_CD_EPOCHS` and writing to `Result/train_results/cd`; after switching config helpers to `python3`, mini CD correctly uses `test/mini-test/train_results_mini/cd` and `cd.epochs=1`.

## Mini 500-Sample Inference Diagnosis

- Mini training used 500 `garden` samples for VAE/LDM/CD, 10 epochs each, then inferred 500 `loop3` frames. This is a loop-closure smoke test, not a formal quality result, because train and inference scenes differ and the sample count is tiny.
- Final mini losses:
  - VAE epoch 10 loss: `0.143191`
  - LDM best loss: `0.058654`
  - CD best loss: `0.000551`
- Inference summary:
  - LDM: `mean_pred_target_chamfer=7.591485`, `avg_infer_seconds=1.295140`, `avg_pred_point_count=5708`
  - CD: `mean_pred_target_chamfer=8.399870`, `avg_infer_seconds=0.024213`, `avg_pred_point_count=9889`
  - Radar baseline target Chamfer in the same CSV: `5.572554`
- Root-cause evidence from typical frames (`000068`, `000150`, `000253`, `000386`, `000478`, `000488`):
  - LDM predicts about `1.56x` target point count; CD predicts about `2.70x`, so CD is over-dense after 1-step distillation.
  - Predicted point clouds are biased toward smaller x. Example frame `000488`: target centroid x `30.7`, LDM x `27.3`, CD x `18.8`.
  - Predicted point clouds are biased lower in z. Across inspected frames, target z90 often reaches `6.7-8.1m`, while LDM/CD z90 is mostly `3.8-4.8m`.
  - y distribution also differs: target often has negative y centroid in late loop3 frames, while predictions stay positive.
- Working hypothesis: current poor mini inference is primarily caused by undertrained/cross-scene mini training and distribution bias learned from limited data, not by a new runtime loading failure. CD trades quality for speed and currently amplifies over-density.
- Generated interactive 3D visualizations under `Result/visualization/mini_inference_compare/`.

## Ubuntu Commands To Run
```bash
conda run -n Radar-Diffusion python test/test_shared_visibility_eval.py
conda run -n Radar-Diffusion python test/shared_visibility_eval.py \
  --pre_dir Data/NTU4DRadLM_Pre/loop3 \
  --output_dir Result/alignment_check/loop3/shared_visibility_original \
  --max_files 120
conda run -n Radar-Diffusion python test/shared_visibility_eval.py \
  --pre_dir Data/NTU4DRadLM_Pre_alignfix/loop3 \
  --output_dir Result/alignment_check/loop3/shared_visibility_alignfix \
  --max_files 120
conda run -n Radar-Diffusion python test/shared_visibility_eval.py \
  --pre_dir Data/NTU4DRadLM_Pre_radarframe/loop3 \
  --output_dir Result/alignment_check/loop3/shared_visibility_radarframe \
  --max_files 120
```

## 2026-06-22 Sensor-Aware Mini Fusion Results

- New training used 500 `garden` sensor-aware samples for 10 epochs each and evaluated 500 `loop3` frames.
- LDM: mean Chamfer `4.086251`, mean latency `1.901586s`, near recall `0.824063`, near precision `0.305613`, BEV IoU `0.281782`.
- CD 1-step: mean Chamfer `3.794822`, mean latency `0.037270s`, near recall `0.871762`, near precision `0.296941`, BEV IoU `0.280092`.
- CD is about 51x faster than LDM and beats LDM Chamfer on 71.4% of frames; both models beat the radar baseline on 360/500 frames.
- Remaining density problem is severe at threshold 0.2: mean pred/target count ratio is `4.12` for LDM and `4.81` for CD. High recall plus low precision indicates over-prediction.
- Saved uncertainty arrays are nearly zero and identical for LDM/CD because the current uncertainty head is a deterministic transform of radar Doppler variance, not a learned model-error estimate.
- Before increasing training penalties, calibrate a global occupancy threshold on validation outputs to separate score calibration error from geometry/model error.
- Broad threshold sweep confirmed score calibration is the dominant density issue:
  - LDM pred/target count ratio falls from `3.33` at threshold `0.2` to `1.11` at `0.6`.
  - CD pred/target count ratio falls from `3.94` at threshold `0.2` to `1.04` at `0.7`.
- A fixed `0.2` threshold is therefore not comparable across legacy and sensor-aware checkpoints. A validation-calibrated global threshold should be saved with each model/evaluation protocol.
- Task-region exact-voxel F1 calibration selected threshold `0.5` for both models:
  - LDM: precision `0.173`, recall `0.237`, F1 `0.200`, count ratio `1.37`.
  - CD: precision `0.182`, recall `0.320`, F1 `0.232`, count ratio `1.76`.
- Thresholding corrects much of the density calibration but exact voxel overlap remains low, so geometry learning still needs formal retraining/validation.
- The uncertainty architecture now separates physical variance from learned model-error variance. The learned branch is initialized conservatively and is optimized with detached-residual Gaussian NLL so it cannot reduce denoising loss by merely inflating variance.
- A one-sample isolated LDM smoke completed in `0.6s` with about `1.5GB` peak reserved GPU memory. The uncertainty head final-layer weight moved from exactly zero to an absolute sum of about `8e-4`, confirming NLL gradients update it.
- Formal inference now reports uncertainty ECE, Brier score, Bernoulli NLL, and uncertainty-error correlation. These interpret `variance/(1+variance)` as predicted occupancy-error probability.

## 2026-06-23 Tree-Structure Failure Root Cause

- The visually poor tree reconstruction is real and is hidden by metrics computed against the reduced sensor-aware target.
- Representative loop3 frame `000008` contains 4737 occupied cells in the original LiDAR voxel but only 616 after hard radar-visible masking, so about 87% of LiDAR structure is removed before training.
- The loader resizes `(X,Y,Z)=(600,200,80)` into tensor `(Z,X,Y)=(32,128,128)`. Over the full physical range this gives about `0.50m x 0.94m x 0.31m` resolution in `(Z,X,Y)`, too coarse along X for trunks and branches.
- A direct VAE upper-bound check with the saved checkpoint produced loop3 occupancy recall `0.250`, precision `0.256`, and IoU `0.145`; the generation stages cannot recover geometry already lost by the VAE.
- IR geometry has two confirmed defects: the calibration file defines `p_camera=R*p_radar+T` but projection applies the inverse transform, and the projection layer interprets tensor `(Z,X,Y)` as physical `(X,Y,Z)`.
- Mini training uses 500 garden frames and evaluates loop3, so it mixes architecture quality with cross-scene generalization.
- Corrective order: preserve the LiDAR obstacle target, fix IR projection, introduce a near-field high-resolution grid, establish a same-scene VAE upper bound, then resume LDM/CD training.
- The repository now uses the integrated preprocessing script as the surviving target-generation implementation; the old standalone module is deleted. Tests must import the integrated vectorized function directly so stale `__pycache__` files cannot hide missing source.

## 2026-06-26 Phase 7 Near-Field / VAE Upper-Bound Findings

- The data loader, unified trainer, CD trainer, inference script, and mini scripts now share a common grid protocol: `source_pc_range` is the original preprocessed voxel range, while `model_pc_range` is the cropped physical range actually learned and reported by the model.
- The mini default is now a near-field `0-40m` model range with unchanged tensor size `(Z,X,Y)=(32,128,128)`. This improves physical X resolution from roughly `0.94m` over `0-120m` to roughly `0.31m` over `0-40m` without increasing tensor memory.
- Inference point cloud conversion now uses `--pc_range` as the model/output range and `--source_pc_range` only for loading/cropping input voxels. This prevents near-field outputs from being stretched back to `0-120m`.
- A 1-frame VAE reconstruction smoke with the existing mini calibrated checkpoint and near-field crop produced best IoU `0.2402` at threshold `0.3`, recall `0.4460`, precision `0.3423`. This is not a formal metric, but it confirms the diagnostic path and supports the hypothesis that VAE reconstruction quality is currently a bottleneck for fine tree structure.

## 2026-06-29 VAE Upper-Bound Diagnosis

- The completed 500-frame near-field VAE check reached best IoU/recall/precision `0.3177/0.4360/0.5393` at a raw decoder threshold of `0.4`.
- Per-frame IoU mean/median are approximately `0.3315/0.3261`; the low result is systematic rather than caused by a few outliers.
- The cropped target contains only about `555` occupied voxels per frame, an occupancy rate of `0.106%`. This is roughly one occupied voxel for every 943 empty voxels, while the current `occupied_weight` is only `8`.
- VAE training loss continued falling from `0.4050` to `0.1320` through epoch 10, so the mini run was not converged. However, adding epochs alone would continue optimizing an occupancy MSE objective that is poorly matched to extreme sparsity.
- The effective KL contribution is negligible because `kl_weight=1e-6`; KL collapse/over-regularization is not the first issue to address.
- The current model is `ultra_lightweight` with `base_channels=16` and `latent_dim=4`. It is appropriate for smoke tests but likely too weak for thin trunks and canopy boundaries after XY downsampling.
- Corrective priority: establish occupancy-logit semantics, use BCE+Dice with valid-region regression for channels 1-3, checkpoint by validation IoU, pass a 32-frame overfit gate, then compare VAE capacity before any new LDM/CD run.
- Detailed plan: `docs/superpowers/plans/2026-06-29-vae-occupancy-upper-bound-recovery.md`.

## 2026-06-29 Task 3 Specification Review

- CD 独立入口已复用共享 VAE checkpoint 协议；checkpoint 元数据优先，历史权重仅接受显式 fallback。
- diagnostic 与 inference 的 fallback 默认改为 `None`，不再静默解释历史权重。
- 网格解析优先级统一为显式 CLI、checkpoint `data_grid_config`、历史默认值。
- inference 后续模型构建、输入/目标加载、坐标转换和指标均使用解析后的有效网格。
- lightweight preset 恢复历史 `latent_dim=4`、`base_channels=24`；z8 由完整 override 配置描述。
- mini 脚本本轮未修改，其工作区 diff 属于 Phase 7 既有改动。

## 2026-06-29 Task 3 Final Review

- 同一 epoch 的 loss/IoU 改善状态现在先统一更新，再构造唯一 payload；best-loss、
  best-IoU、兼容别名和 epoch checkpoint 均记录更新后的两个全局 best。
- 有条件推理继续以 `vae.get_latent(condition)` 的实际结果确定潜空间 shape。
- 无条件推理改为使用已加载 VAE 的 `latent_dim`，并按 encoder 的真实下采样卷积参数
  推导空间 shape，不再硬编码 z4 或默认网格，也不分配巨型 dummy 输入。

## 2026-06-29 Task 3 Quality Review

- legacy 生成 UNet 统一采用 `2 * latent_dim -> latent_dim`；multimodal backbone 使用
  `max(16, 2 * latent_dim) -> latent_dim`，保留现有 z4/16-channel checkpoint 兼容性。
- unified LDM、独立/统一 CD 和 inference 均从 VAE 或生成 checkpoint metadata 获取
  latent_dim；旧生成权重可从输入/输出卷积 shape 推导。
- LDM/CD checkpoint 新增 `latent_dim` 与完整 `model_config`。
- VAE checkpoint 新增 scheduler 状态；旧 checkpoint 恢复时按已完成 epoch 设置
  scheduler 进度并保留 optimizer 中的连续 LR。
- VAE checkpoint 使用同目录临时文件加 `os.replace`；best-IoU alias 通过独立临时副本
  原子替换，避免半写文件并确保内容一致。

## 2026-06-30 32-Frame VAE Overfit Result

- The lightweight VAE with `latent_dim=8`, BCE+Dice occupancy supervision, and the
  `0-40m` near-field grid completed 100 epochs on 32 loop3 frames.
- The deterministic split used 31 training frames and 1 validation frame. Final epoch
  validation IoU/recall/precision at threshold 0.5 were approximately
  `0.6626/0.8410/0.7575`.
- The `vae_best_iou.pt` checkpoint was evaluated across all 32 selected frames using
  the formal diagnostic. The best probability threshold was `0.55`.
- Aggregate 32-frame IoU/recall/precision were `0.8417/0.9727/0.8621`, with predicted
  occupancy count ratio `1.1284`.
- This passes the planned overfit gate (`IoU >= 0.75`, `recall >= 0.80`) and confirms
  that the VAE architecture can retain near-field obstacle structure after the loss and
  capacity changes.
- The result is an overfit/reconstructability test, not a generalization result. The next
  required experiment is the 500-frame train/validation run before restarting LDM/CD.
- The historical lightweight preset exposed a GroupNorm defect: 24/72 channels were
  incompatible with a fixed 32-group normalization. Shared normalization now selects the
  largest valid divisor without changing channel widths or checkpoint tensor shapes.
- Final review also aligned decoded LDM occupancy auxiliary losses with raw/sigmoid VAE
  semantics and made metadata-free checkpoint fallback explicitly preserve
  `legacy_mse + raw`.

## 2026-06-30 Final Review Fixes

- LDM decoded occupancy auxiliaries previously compared channel-0 logits directly with
  binary targets. For sigmoid VAE checkpoints this made zero logits score better than
  confident negative logits for empty targets.
- Decoded occupancy MSE/FP/mass now apply sigmoid only to channel 0 when the loaded VAE
  declares `occupancy_activation=sigmoid`; raw historical checkpoints retain the legacy
  path and the three configured weights remain independent.
- Metadata-free preset fallback now explicitly restores `legacy_mse` plus `raw`.
  A complete explicit fallback config remains authoritative.
- Multimodal inference previously used compatible-subset loading even with `strict=True`.
  Strict entrypoints now require every key and shape; `strict=False` retains lightweight
  test construction.
- Supervision targets, continuous-channel auxiliaries, and voxel counts are unchanged.
  Sigmoid-checkpoint decoded occupancy loss values and invalid-load failure behavior change.
- Follow-up review found that `strict=True` still skipped loading for an empty dictionary
  because loading was guarded by state-dict truthiness. Strict inference now rejects empty
  states before model construction and explicitly requires the first input and final output
  convolution weights for both legacy and multimodal prefixes.
- `strict=False` intentionally continues to allow empty/partial states for lightweight test
  construction; production entrypoints use `strict=True`.

## 2026-07-02 Final Reviewer Follow-up

- 最终 reviewer follow-up 修补点已同步：
  - `max_files` 改为在 split 之后应用；
  - threshold 参数要求为有限值且必须落在 `[0,1]`；
  - target loader 已覆盖真实 `.npz` 稀疏 target；
  - JSON 显式保留 `deprecated_x_max` 兼容字段。

## 2026-07-01 LDM Occupancy Threshold Scan Review

- The 500-frame VAE upper-bound experiment passed strongly: at threshold `0.7`, aggregate
  IoU/recall/precision are `0.7888/0.9533/0.8205`.
- The trained LDM 500-frame inference at threshold `0.5` reports Chamfer `1.3749` versus
  radar baseline `2.0553`, and near recall/precision/BEV IoU
  `0.5742/0.6646/0.4478`.
- The saved-output threshold sweep recommended `0.1`, but its best exact voxel F1 is only
  `0.007126`; this recommendation is not accepted.
- Root cause: the sweep directly resizes the complete `0-120m` target to `(32,128,128)`,
  while the prediction represents the cropped `0-40m` model range. Prediction and target
  therefore use different physical X scales.
- Threshold selection also uses all 500 frames and exact voxel F1. It must use the
  deterministic validation partition and prioritize `0-20m/20-40m` BEV task metrics.
- Do not retrain LDM or start CD from the current recommendation. First repair the sweep,
  then reuse the saved 500 prediction voxels for a corrected scan.
- Detailed plan:
  `docs/superpowers/plans/2026-07-01-ldm-threshold-evaluation-and-cd-gate.md`.

## 2026-07-01 Corrected Threshold Evaluation

- Corrected validation scan selected threshold `0.1` on 100 deterministic validation
  frames. Overall BEV precision/recall/F1/IoU are
  `0.5985/0.5711/0.5845/0.4129`, match@2m is `0.9622`, and prediction/target point ratio
  is `1.1435`.
- At `0-20m`, precision/recall/F1/IoU are `0.6568/0.6271/0.6416/0.4723`; at `20-40m`
  they are `0.5079/0.4842/0.4958/0.3296`.
- Threshold `0.2` has BEV F1 `0.5801`, only `0.0044` below threshold `0.1`; the local
  threshold curve is smooth rather than an isolated numerical spike.
- A new 500-frame inference at threshold `0.1` improved Chamfer from `1.3749` to `1.3101`,
  near recall from `0.5742` to `0.6102`, and near BEV IoU from `0.4478` to `0.4604`.
  Near precision decreased modestly from `0.6646` to `0.6498`.
- The old threshold `0.5` and new threshold `0.1` results came from two independent
  stochastic LDM inference runs. The observed metric improvement cannot be attributed
  entirely to the threshold, and the runtime difference cannot be interpreted as a
  threshold cost.
- This is an internal train/validation experiment over 500 frames from the same `loop3`
  scene, not an independent-scene generalization result. A formal paper result requires
  evaluation on an independent scene or a persisted split manifest.
- Threshold `0.1` increased the average predicted point count by `17.39%`. Recall and
  BEV IoU improved, while near NN mean worsened from `0.4731` to `0.5116`, indicating a
  coverage gain traded against local geometric precision.
- All ten visual checks (`000003`, `000010`, `000011`, `000018`, `000019`, `000021`,
  `000023`, `000025`, `000028`, `000037`) are complete. In all `10/10` frames, the LDM
  covers the approximate regions of the main raw-LiDAR obstacles, but its point cloud
  remains sparse, and neither trunk continuity nor fine canopy structure is recovered
  consistently. Therefore, the visual gate for tree-structure reconstruction has not
  passed and must not be reported as achieved.
- A second fixed-seed visual review used the first ten validation frames in their original
  randomized order: `000280`, `000195`, `000103`, `000303`, `000229`, `000311`, `000037`,
  `000454`, `000431`, and `000493`. The HTML files are stored in
  `test/result/ldm_near40_500_v2_threshold_validated/raw_lidar_visuals_random_validation/`.
  All `10/10` frames approximately cover the main obstacle regions, but frames such as
  `000280` and `000311` are visibly sparse. Trunk continuity and fine canopy structure
  remain unstable, so the tree-structure visual gate does not pass.

## 2026-07-02 CD Gate Decision

| 准入项 | 统计集合 | 实际结果 | 门槛 | 判定 |
| --- | --- | ---: | ---: | --- |
| Near BEV IoU | 500 帧内部全量复评（含训练帧） | `0.460417` | `>= 0.40` | PASS |
| Near recall | 500 帧内部全量复评（含训练帧） | `0.610190` | `>= 0.55` | PASS |
| Near precision | 500 帧内部全量复评（含训练帧） | `0.649795` | `>= 0.60` | PASS |
| Validation pred/target ratio | 100 帧 validation | `1.143535` | `[0.8, 1.3]` | PASS |
| Pred-target Chamfer | 500 帧内部全量复评（含训练帧） | `1.310079` | `< radar 2.055266` | PASS |
| 主要障碍物区域覆盖 | 按 `split_seed=42` 确定的 validation 帧集合 | `10/10` 大致覆盖 | `>= 8/10` | PASS |
| 树木结构恢复 | 按 `split_seed=42` 确定的 validation 帧集合 | `000280`、`000311` 明显稀疏，树干连续性和树冠细结构均未稳定恢复 | 当前核心目标要求结构稳定 | **FAIL** |

- **总判定：HOLD / FAIL，不启动 CD 蒸馏。**
- Threshold `0.1` 的选择和点数比例来自 100 帧 validation；near
  IoU/recall/precision 与 Chamfer 来自包含训练帧的 500 帧内部全量复评。当前证据
  统计集合不统一，正式准入必须在统一的独立 validation/test 集上重新计算；这一
  限制不会改变当前由树木结构失败触发的 HOLD 结论。
- 视觉帧由 `split_seed=42` 的确定性数据划分选出，只保证 validation 帧集合可复现，
  不代表 LDM 生成采样使用了固定随机 seed。
- 当前 LDM 已达到一般障碍物概率地图的数值准入水平，但没有达到本阶段“恢复树木
  结构”的目标。宽泛障碍物覆盖不能写成树木结构恢复成功。
- CD 只会蒸馏当前 LDM 教师的生成能力，不能补回教师尚未恢复的树干连续性和树冠
  细节。现在启动 CD 会把教师的结构缺陷固化到快速模型中，因此应先改善 LDM 的
  结构生成能力，再重新执行同一套 gate。
- 当前树木结构判断仍是定性结果。后续需要实现最高点高度召回、垂直连通率和树干
  区域 recall 等指标，再依据实验分布制定门槛；本阶段不预先捏造阈值。
- 后续诊断顺序：
  1. 先实现高度覆盖率、垂直连通率和树干召回率。
  2. 使用这些指标检查 VAE 重建上界。
  3. 若 VAE 通过而 LDM 失败，再加入垂直结构或高度分布损失并重训 LDM。
  4. 若 VAE 也失败，先提高 Z/X 方向物理分辨率或调整监督目标，而不是启动 CD。

## 2026-07-06 Phase 10 VAE Vertical-Structure Gate

- 新增四类树木结构指标：占用高度覆盖、最高点高度召回、竖向连续段召回和主干区域
  recall。所有指标同时保存 numerator/denominator，跨帧汇总采用 micro average。
- `test/result/vae_overfit_32_vertical_diagnostic/` 的 32 帧过拟合 checkpoint 在最佳
  IoU 阈值 `0.55` 下得到：
  - IoU/Recall/Precision：`0.8417/0.9727/0.8621`
  - height coverage：`0.9727`
  - top-height recall：`0.9184`
  - vertical connectivity：`0.9735`
  - trunk-region recall：`0.9845`
- `test/result/vae_near40_500_v2_vertical_diagnostic_32/` 的 500 帧训练 checkpoint
  在最佳 IoU 阈值 `0.70` 下得到：
  - IoU/Recall/Precision：`0.8477/0.9637/0.8756`
  - height coverage：`0.9637`
  - top-height recall：`0.9230`
  - vertical connectivity：`0.9663`
  - trunk-region recall：`0.9720`
- 两个 VAE checkpoint 都能较好保留垂直高度、连续段和下部主干结构。结合 Phase 9
  中 LDM 可视化仍缺少稳定树干/树冠的证据，当前主要损失发生在条件扩散去噪阶段，
  不是 VAE 编解码上界。
- 下一步不扩大 VAE，也不立即启动 CD。应先为 LDM 增加可微的高度分布/竖向连续性
  辅助约束，并在固定 prediction seed、统一独立验证集上重新评估结构指标。
- 本轮没有改变监督信号、target 占用体素数量、模型体素尺寸或 checkpoint；新增指标
  会让报告对“点云数量接近但树木结构断裂”的情况更敏感。

## 2026-07-06 LDM Structure-Loss Design

- `OptimizedLDMTrainer.train_epoch()` 已在启用 decoded occupancy auxiliary 时调用
  `vae.decode(denoised)`，因此结构损失应复用该解码结果，避免重复解码和额外显存峰值。
- v1 不直接优化不可微的“最长连续段”或 hard top-height 指标，而使用两个稳定代理：
  1. 按 `(X,Y)` 列归一化的 Z 轴软占用分布差异，用于约束高度位置；
  2. 相邻 Z 层软占用差分差异，用于约束竖向断裂和连续性。
- 两项损失必须支持 `raw` 与 `sigmoid` occupancy 语义、空 target、有限梯度，并通过
  独立配置权重启用；默认权重为 `0.0`，保持旧训练配置完全兼容。
- 该改动会改变启用后的 LDM 监督信号和 loss 数值，但不会改变 target 体素数量、
  模型网格、模型参数形状或旧 checkpoint 加载协议。
- Task 1 首轮质量审查发现两个 AMP 风险：半精度 sigmoid 在转 float32 前执行会让
  极端负 logits 的梯度出现 NaN；目标非空而预测质量为零时直接除 `eps=1e-6` 会把
  梯度放大到约 `6.7e5`。正式接入 trainer 前必须先修复并增加回归测试。

## 2026-07-07 LDM Vertical Structure Loss Stability

- The differentiable height-distribution and vertical-continuity losses now keep the same
  model/checkpoint tensor shapes and do not change target voxel count.
- The supervision signal changes only when future LDM configs enable nonzero structure-loss
  weights: decoded occupancy columns are compared along the physical Z tensor axis, while
  empty target columns remain unsupervised.
- The helper now performs occupancy activation in float32 and avoids `1/eps` normalization
  spikes for all-empty predictions, reducing AMP/half precision overflow risk during LDM
  training.
- Existing inference metrics are unchanged by this helper alone. Future LDM training with
  nonzero height/continuity weights will change total-loss scale, so loss components must
  be logged before comparing training curves.

## 2026-07-07 LDM Structure Loss Integration Findings

- Enabling the default config now changes LDM supervision by adding two decoded-occupancy
  structure terms with weights `0.02/0.02`. This does not alter target voxel count,
  model grid size, latent dimension, model parameter tensor shapes, or VAE checkpoint
  compatibility.
- The decoded auxiliary path is now shared: occupancy, height distribution, and vertical
  continuity reuse one VAE decode per batch. This keeps the new supervision cheaper than
  decoding separately for each term.
- New LDM CSV component columns are required for interpreting training curves because the
  total loss now combines latent MSE, decoded occupancy, height distribution, continuity,
  and optional uncertainty terms. Old CSV files are archived on resume to avoid mixed row
  widths.
- Tail-gradient handling in LDM now matches the VAE trainer principle: an incomplete
  gradient accumulation window is rescaled to the actual batch mean before clipping/step.
  Under AMP, rescaling occurs after `GradScaler.unscale_()` to avoid false overflow
  detection on scaled gradients.
- Mini experiments can disable or tune the new structure terms with
  `MINI_LDM_HEIGHT_WEIGHT` and `MINI_LDM_CONTINUITY_WEIGHT`, which is useful for ablation:
  `0/0` gives the previous decoded-occupancy-only behavior, while `0.02/0.02` is the new
  default smoke setting.

## 2026-07-07 LDM Structure Smoke Findings

- The short smoke confirms the new LDM structure-loss path is executable, finite, logged,
  and checkpointed. It is not a quality result because it used only two samples and one
  LDM epoch.
- The project split logic requires at least two samples even for LDM smoke runs because
  the shared train/validation split helper rejects one-sample datasets.
- The smoke output demonstrates all expected component columns are populated in
  `metrics.csv`; the `uncertainty_loss` column records the raw component while the total
  loss applies `uncertainty_loss_weight`, matching the existing component-logging style.
- The next meaningful experiment is to retrain LDM on the already selected 500-frame
  near-field protocol with fixed validation split and then re-run the Phase 10 vertical
  structure metrics and raw-LiDAR visual gate before any CD distillation.
