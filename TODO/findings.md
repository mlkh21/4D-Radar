# Findings

## 2026-07-13 v10 Task3 列级损失实验入口
- `train_minimal.sh` 的 YAML argv 契约可在末尾追加三项而不改变既有 31 项顺序；列正/负权重默认均为 `0.0`，temperature 默认 `1.0`。
- v10 A/B 仅改变列负样本权重（A=`0.01`、B=`0.02`），列正样本权重均为 `0.02`；旧 `decoded_density_weight` 在 v10 固定为 `0.0`。
- v10 必须在自身完成空目录审计并持有独立 `.v10.lock` 后，向 generic runner 内部传递覆盖许可，以复用刚复制的 VAE；该许可不提供用户结果删除能力。
- 结果整理续审：`test/result/ldm/` 根层的 checkpoint、日志和配置属于 2026-07-13 一次未完成的 10 epoch LDM 运行（日志只到 epoch 7），不能并入已有 v10 实验；应整体保留到 `archive` 叶目录。
- `test/result/vae/vae_best.pt` 与 `test/result/vae/reconstruction/vae_near40_500_v2/vae/vae_best.pt` 内容完全一致，但前者是根级运行使用的独立副本，不能删除或覆盖，应随该未命名运行归档。
- `test/result/cd/` 当前为空；根级 `.tmp_mini_train_dataset` 与 `.tmp_ldm_config.yaml` 与上述 09:11 根级运行时间一致，应随该运行归档，保留符号链接及临时配置。
- `test/result/ldm/ablation/ldm_near40_500_z64_v10d_neg0005_screen.lock` 和 `.v10.lock` 实际属于 v10-D 运行且位于其旁边；移动它们会破坏 runner 的 `${EXP_DIR}.lock` / `${EXP_DIR}.v10.lock` 约定，因此保持原位。
- 已命名结果的续整理映射：`vae_near40_500_v2` → `vae/reconstruction`；seeded v10-A → `ldm/ablation`；raw LiDAR/LDM/CD HTML → `comparison/alignment_check`；数据协议审计 → `comparison`。
- 真实 shell smoke 验证 v10 -> generic vertical -> train_minimal -> unified_train 的参数链，仅替换最终训练 Python 命令，未运行长训练。
- 规格复审修复：generic 的输出 allowlist 必须独立于 `ALLOW_OVERWRITE`，仅接受项目 `test/result` 或 `/tmp` 的严格子目录；两者根目录均拒绝。
- v10 在持锁后对 EXP 实体、VAE/LDM、scratch/config 的 symlink 与 canonical parent 重复审计，并在 `train_minimal` 可能删除 scratch 前后复查；VAE 与最终 LDM checkpoint 均要求 `-s`。
- v10 的 500 samples、3 epochs、Z64、source/model range、seed42、garden 均为不可由宿主环境覆盖的常量。
- Important 复审修复：`MINI_REQUIRE_FRESH_SCRATCH=1` 会保留原始 scratch 输入以识别有效及 dangling symlink，拒绝任何既有实体，并要求规范化路径严格位于 `MINI_RESULTS_DIR` 子目录。
- fresh 模式不执行 `rm -rf`，只以 `mkdir --` 创建一次并复核 canonical path；v10 固定启用该模式，上层 audit 对不存在 scratch 仅审计而不创建。
- Config 竞态修复：`MINI_REQUIRE_FRESH_CONFIG=1` 保留原始 config 输入以拒绝现有文件、有效及 dangling symlink，并要求 canonical parent 位于 `MINI_RESULTS_DIR` 内。
- YAML 生成器通过追加 argv 选择 `open(..., mode='x')`；即使入口检查后路径被文件或 symlink 抢占，也由原子独占创建失败而不会跟随或截断。默认模式仍使用 `w`。
- Generic runner 现在将相对 scratch/config 锚定到 EXP，并要求 canonical 路径为 EXP 严格子路径、basename 以 `.tmp_` 开头，同时拒绝 symlink 及 dataset/config 类型冲突；外部 `/tmp` 路径不再因全局 allowlist 而被接受。
- v10 在获得自身锁后、首次实验目录写入前再次检查 EXP 必须不存在或为空；持锁瞬间注入文件的接口测试会在创建 `vae/` 前失败。

## Recovered Context
- Prior work focused on NTU4DRadLM radar/LiDAR preprocessing, calibration, and inference diagnostics.
- Earlier checks suggested roll changes can improve dz but may worsen or leave dy unresolved, so global centroid offsets are likely misleading.
- The next proposed step was "shared visible region evaluation": compare only common/nearby observable structure using radar-to-LiDAR nearest-neighbor distances, distance and height bands, non-ground filtering, and BEV grid IoU.
- Existing generated/mentioned files include `test/diagnostics/radar/check_radar_axis_conventions.py`, `test/diagnostics/alignment/alignment_sanity_check.py`, `test/evaluation/comparison/compare_voxel_triplets.py`, and `test/visualization/generate_interactive_raw_compare.py`.

## Working Hypothesis
dy may be dominated by radar/LiDAR point distribution, FOV, effective detections, and ground filtering differences rather than a simple extrinsic translation/rotation error.

## Implemented Diagnostic
- Added `test/diagnostics/alignment/shared_visibility_eval.py` to evaluate voxelized radar/lidar/target overlap using nearest-neighbor distances, match ratios, range bins, z-min filters, and BEV occupancy IoU.
- Added `test/unit/test_shared_visibility_eval.py` with synthetic checks for nearest-neighbor match ratios and BEV IoU.
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
- Generated interactive 3D visualizations under `test/result/ldm/visualization/mini_inference_compare/`.

## Ubuntu Commands To Run
```bash
conda run -n Radar-Diffusion python test/unit/test_shared_visibility_eval.py
conda run -n Radar-Diffusion python test/diagnostics/alignment/shared_visibility_eval.py \
  --pre_dir Data/NTU4DRadLM_Pre/loop3 \
  --output_dir test/result/comparison/alignment_check/loop3/shared_visibility_original \
  --max_files 120
conda run -n Radar-Diffusion python test/diagnostics/alignment/shared_visibility_eval.py \
  --pre_dir Data/NTU4DRadLM_Pre_alignfix/loop3 \
  --output_dir test/result/comparison/alignment_check/loop3/shared_visibility_alignfix \
  --max_files 120
conda run -n Radar-Diffusion python test/diagnostics/alignment/shared_visibility_eval.py \
  --pre_dir Data/NTU4DRadLM_Pre_radarframe/loop3 \
  --output_dir test/result/comparison/alignment_check/loop3/shared_visibility_radarframe \
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
  `test/result/ldm/evaluation/ldm_near40_500_v2_threshold_validated/raw_lidar_visuals_random_validation/`.
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
- `test/result/vae/overfit/vae_overfit_32_vertical_diagnostic/` 的 32 帧过拟合 checkpoint 在最佳
  IoU 阈值 `0.55` 下得到：
  - IoU/Recall/Precision：`0.8417/0.9727/0.8621`
  - height coverage：`0.9727`
  - top-height recall：`0.9184`
  - vertical connectivity：`0.9735`
  - trunk-region recall：`0.9845`
- `test/result/vae/diagnostics/vae_near40_500_v2_vertical_diagnostic_32/` 的 500 帧训练 checkpoint
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

## 2026-07-08 LDM Vertical Evaluation Sidecar Filter

- `iter_prediction_files()` previously allowed generic `.npy` files except `_pcl.npy`,
  so LDM sidecar outputs such as `000000_uncertainty.npy` could be interpreted as dense
  prediction voxels and produce a wrong frame id.
- The evaluation should only consume saved dense prediction files matching
  `*_voxel.npy`; `_uncertainty.npy` and `_pcl.npy` are sidecar outputs.
- This fix changes only saved-output file discovery. It does not change supervision
  signals, target voxel counts, model voxel dimensions, or metric formulas.

## 2026-07-08 LDM Prediction Layout Ambiguity

- `load_prediction_occupancy()` had an ambiguous 4D layout case when prediction shape was
  `[Z,X,Y,4]` and `Z == 4`; the old order could treat axis 0 as channels and return
  `arr[0]`.
- Channel-last prediction voxels are now preferred when the last axis is a clear channel
  axis and the two spatial middle axes are not channel-sized.
- This changes only prediction array layout parsing for evaluation. It does not change
  target construction, target voxel count, model grid size, or metric formulas.

## 2026-07-08 LDM Vertical Structure Evaluation on 500-Frame Run

- The saved-output vertical evaluator must crop original target voxels from the source
  `0-120m` range to the learned `0-40m` model range, then resize to `(Z,X,Y)=(32,128,128)`.
  Without this step the script compares prediction shape `(32,128,128)` against raw target
  shape `(80,600,200)`, which is an invalid metric rather than a model failure.
- Running the fixed evaluator on
  `test/result/ldm/vertical_structure/ldm_near40_500_vertical_v1/loop3_ldm_eval` with threshold `0.05` produced:
  height coverage `0.3853`, top-height `0.1788`, vertical connectivity `0.3956`,
  trunk recall `0.4190`.
- This matches the raw-LiDAR HTML observation: the model can recover part of the lower
  trunk/near obstacle structure, but it severely under-recovers upper height and canopy
  extent. The main next target is therefore top-height / vertical extent recovery, not CD
  distillation.
- Prediction occupancy total is still higher than target (`701438` vs `277657` over 500
  frames at threshold `0.05`), so simply lowering the threshold further would improve
  recall at the cost of more false positives. The next LDM experiment should adjust
  vertical supervision/loss balance and inspect precision/BEV metrics together.

## 2026-07-08 One-Click LDM Vertical Experiment Runner

- Added a shell runner for the next v2 experiment so training, inference, vertical
  structure evaluation, and raw-LiDAR HTML generation share one explicit protocol.
- The runner changes experiment hyperparameters by default, not model code: height
  distribution weight `0.05`, vertical continuity weight `0.02`, 500 samples, 10 LDM
  epochs, near-field `0-40m` crop, and occupancy threshold `0.05`.
- It reuses an existing VAE checkpoint by copying it into the new experiment directory
  before LDM training. This keeps the VAE upper-bound protocol fixed while isolating the
  effect of LDM vertical supervision.
- The expected metric impact is on LDM prediction structure only. Target voxel count,
  preprocessing policy, model grid size, and VAE checkpoint semantics remain unchanged.

## 2026-07-08 LDM Vertical v2 Result Analysis

- The v2 experiment (`height_weight=0.05`, `continuity_weight=0.02`) improved the intended
  vertical recall metrics compared with v1:
  - height coverage: `0.3853 -> 0.4256` (`+10.45%`)
  - top-height: `0.1788 -> 0.2138` (`+19.55%`)
  - vertical connectivity: `0.3956 -> 0.4392` (`+11.02%`)
  - trunk recall: `0.4190 -> 0.4097` (`-2.22%`)
- The improvement is not just a few frames: height coverage improved on `364/500` frames,
  top-height on `341/500`, and vertical connectivity on `365/500`.
- However, prediction density increased from `701438` to `835832` occupied voxels
  (`+19.16%`) while target voxel count stayed fixed. Formal inference metrics show the
  same trade-off: near recall improved `0.7440 -> 0.8173`, but precision dropped
  `0.3990 -> 0.3363`, BEV IoU dropped `0.3365 -> 0.3058`, and NN mean worsened
  `0.9512 -> 1.0479`.
- Conclusion: increasing height supervision is directionally useful for tree vertical
  extent, but the current v2 setting over-expands predictions. The next experiment should
  not simply raise height weight again; it should keep or slightly reduce height weight
  while adding stronger density/precision control or threshold calibration.

## 2026-07-08 LDM Vertical v3 Result Analysis

- The v3 experiment (`height_weight=0.04`, `continuity_weight=0.04`) did not produce a
  better trade-off than v2.
- Compared with v1, v3 only modestly improves height coverage (`+0.0155`) and vertical
  connectivity (`+0.0192`), while top-height is almost unchanged (`+0.0026`) and trunk
  recall is effectively flat (`-0.0008`).
- Compared with v2, v3 loses most of the height/top recovery:
  height coverage `0.4256 -> 0.4008`, top-height `0.2138 -> 0.1814`, vertical
  connectivity `0.4392 -> 0.4148`. It recovers trunk recall slightly
  (`0.4097 -> 0.4182`), but that is not enough to justify the loss in upper structure.
- Formal inference also degrades versus v2: Chamfer `2.1402 -> 2.2566`,
  near recall `0.8173 -> 0.7688`, BEV IoU `0.3058 -> 0.3125` remains below v1, and
  task NN mean worsens `1.0479 -> 1.1315`.
- Conclusion: rebalancing height/continuity alone is not enough. The current best for
  vertical extent is still v2, but v2 over-densifies. The next code change should add a
  density/precision regularizer or calibrate threshold per validation split before more
  LDM retraining; CD remains held.

## 2026-07-08 LDM Density / Precision Regularizer

- Added a decoded density/precision auxiliary loss for LDM to address the v2 failure mode:
  better vertical height recovery but too many predicted occupied voxels.
- The loss has two terms: a per-sample excess occupancy mass penalty and a target-empty
  false-positive probability penalty. This targets over-dense predictions without changing
  target construction, target voxel count, model voxel dimensions, or checkpoint tensor
  shapes.
- The new config key is `ldm.decoded_density_weight`; the code and mini scripts default it
  to `0.0`, so old experiments remain behavior-compatible until the weight is explicitly
  enabled.
- When enabled, the loss reuses the same decoded tensor as decoded occupancy/height/
  continuity auxiliaries, so height + continuity + density still require only one
  `vae.decode()` per batch.
- The next v4 experiment should use v2-like height recovery plus a small density weight,
  then judge success by retaining top-height while recovering near precision and BEV IoU.

## 2026-07-08 LDM Vertical v4 Result Analysis

- v4 (`height_weight=0.05`, `continuity_weight=0.02`, `density_weight=0.05`) reduced
  density compared with v2 but did not recover the desired precision/BEV trade-off.
- At threshold `0.05`, v4 has fewer predicted occupied voxels than v2
  (`835832 -> 761552`) and fewer average points (`1671.66 -> 1523.10`), so the density
  regularizer is active in the expected direction.
- v4 achieves the best top-height recall so far (`0.2219` vs v2 `0.2138`), but loses
  height coverage (`0.3829`), vertical connectivity (`0.3961`), and trunk recall
  (`0.3751`). This suggests the regularizer suppresses lower/mid-column occupancy while
  still allowing some high-Z activations.
- Formal metrics remain below the target trade-off: near recall `0.7826`, precision
  `0.3467`, BEV IoU `0.3067`, Chamfer `2.2724`. Compared with v2, precision improves
  slightly but BEV IoU and Chamfer remain worse; compared with v1, precision/BEV are still
  lower.
- Threshold sweep selected `0.4` by validation BEV F1. At that threshold, validation
  pred/target ratio falls to about `2.31`, BEV F1 is slightly best (`0.4218`), and top
  height remains high in vertical evaluation (`0.2152`), but height coverage/trunk/vertical
  continuity drop further.
- Conclusion: the density loss is useful but too blunt at the current form/weight. The
  next step should not be CD. Prefer either a smaller density weight (`0.02`) or a
  target-aware density loss that protects occupied columns/trunk regions while penalizing
  empty-space spread.

## 2026-07-08 Raw LiDAR + Voxel Column Visualization

- 曾尝试用 predicted voxel columns 叠加原始 raw LiDAR 点云，以排查近场
  `z` 方向体素间距约 `0.5m` 是否造成树结构“分层”的显示假象。
- 视觉复查后确认：即使以 voxel-column 方式显示，预测结构与 raw LiDAR 仍有明显
  差距；结构质量差不是 HTML 显示层或点中心渲染造成的。
- 该可视化功能已撤销，当前诊断恢复为原始 raw LiDAR 点云对比，不再把
  `ldm_voxel_columns` 作为当前脚本能力或推荐诊断路径。
- 历史生成目录
  `test/result/ldm/vertical_structure/ldm_near40_500_vertical_v4/raw_lidar_voxel_column_visuals`
  仅保留为旧实验输出记录，不代表当前脚本默认功能。

## 2026-07-08 Z=64 VAE Upper-Bound Runner

- 新增 `test/mini-test/run_vae_z64_upper_bound.sh`，用于手动启动近场
  `Z=64, X=128, Y=128` 的 VAE 上界实验。
- 默认实验目录为 `test/result/vae/reconstruction/vae_near40_500_z64_upper_bound`，默认训练
  `garden` 场景 500 帧、10 epoch，VAE 使用 `lightweight`、`latent_dim=8` 和
  `bce_dice` occupancy loss。
- 脚本复用 `train_minimal.sh vae`，训练完成后检查
  `${MINI_RESULTS_DIR}/vae/vae_best.pt` 和 target voxel 目录，再调用
  `diagnose_vae_reconstruction.py` 输出到 `${EXP_DIR}/vae_upper_bound`。
- 这次只做脚本语法和静态契约测试，没有启动 VAE 长训练；因此不会产生新的
  checkpoint、训练日志或诊断结果。

## 2026-07-09 Z=64 LDM Inheritance Check and Optimization

- Z=64 VAE reconstruction upper bound passes the structure gate:
  - best threshold `0.95`;
  - IoU/recall/precision `0.6027 / 0.8344 / 0.6847`;
  - structure recall: height coverage `0.8344`, top height `0.4639`,
    vertical connectivity `0.8516`, trunk region `0.8628`.
- The current Z=64 LDM did not inherit that VAE upper bound:
  - threshold sweep selects `0.99`, indicating output probabilities are still too dense;
  - voxel IoU at the selected threshold is only `0.0916`;
  - voxel recall/precision are `0.2978 / 0.1168`;
  - pred/target count ratio remains `2.55x` even at threshold `0.99`;
  - task BEV IoU is `0.2483`, which is useful for coarse occupancy but far below the
    VAE reconstruction ceiling for fine structure.
- LDM training loss confirms that the issue is in the conditional generation stage rather
  than VAE capacity:
  - latent loss decreases to about `0.0222`;
  - decoded occupancy loss stays very small, around `5e-4`;
  - uncertainty NLL is negative and numerically dominates the total loss scale.
- Optimization implemented:
  - changed decoded density precision loss from per-empty-voxel/global-mass suppression
    to pure empty-column suppression, so target-occupied `(X,Y)` columns can still learn
    vertical tree/trunk completion without receiving this density penalty;
  - changed sigmoid-activation density loss to use empty-class `softplus(logit)` rather
    than squared sigmoid probability, preserving useful gradients for high-confidence
    background false positives;
  - exposed mini-script controls for `MINI_LDM_DECODED_WEIGHT`,
    `MINI_LDM_DECODED_FP_WEIGHT`, `MINI_LDM_DECODED_MASS_WEIGHT`, and
    `MINI_LDM_UNCERTAINTY_WEIGHT`;
  - propagated those controls through `run_ldm_vertical_experiment.sh`.
  - added `test/mini-test/run_ldm_z64_v5_experiment.sh` as a short Z=64 v5 runner that
    reuses the existing Z=64 VAE checkpoint, disables uncertainty NLL, and enables a
    small empty-column density weight by default.
- Quality review follow-up:
  - generic LDM vertical runner now normalizes relative `EXP_DIR` and `BASE_VAE_CKPT` to
    repository-root paths and calls helper scripts by absolute path;
  - Z=64 v5 runner refuses to overwrite an existing default experiment unless
    `ALLOW_OVERWRITE=1`;
  - `train_minimal.sh` now respects an externally supplied `CUDA_VISIBLE_DEVICES` when
    `CUDA_DEVICES` is not explicitly set.
- Expected effect on supervision and metrics:
  - target voxel count and grid shape are unchanged;
  - density control should reduce background spread without destroying existing obstacle
    vertical columns;
  - disabling uncertainty loss for the next geometry run should make checkpoint selection
    less sensitive to negative NLL and more reflective of decoded occupancy structure.

## 2026-07-09 Z=64 LDM v5 Threshold Diagnosis and v6 Top-Height Loss

- v5 threshold sweep still selected `0.99`, so BEV/task metric calibration continues to
  prefer a very high threshold.
- Vertical structure at thresholds `0.5 / 0.7 / 0.85 / 0.95` shows:
  - height coverage improves at lower thresholds (`0.4698` at `0.5` vs `0.3869` at
    `0.95`);
  - trunk recall also improves at lower thresholds (`0.5164` at `0.5` vs `0.4172` at
    `0.95`);
  - top-height recall remains nearly flat and very low (`0.0991 -> 0.0949`).
- Conclusion: v5's missing upper canopy / top structure is not an occupancy-threshold
  calibration artifact. The LDM is not assigning useful probability to target top voxels
  in the first place.
- Implemented v6 top-height auxiliary supervision:
  - `decoded_vertical_structure_losses()` now returns `top_height_loss`;
  - for each non-empty `(X,Y)` target column, the loss directly supervises the highest
    target occupied Z cell;
  - sigmoid/logit models use positive-class `softplus(-logit)` to preserve gradients for
    missed high-Z targets;
  - raw/probability models use `1 - p_top`.
- Added `MINI_LDM_TOP_WEIGHT` and `ldm.decoded_top_height_weight` so top supervision can
  be tuned independently from height-distribution and vertical-continuity losses.
- Added `test/mini-test/run_ldm_z64_v6_top_experiment.sh`; default settings reuse the
  existing Z=64 VAE, set `MINI_LDM_TOP_WEIGHT=0.08`, reduce density to `0.015`, and keep
  uncertainty loss disabled.

## 2026-07-09 IR-Supervision Feasibility and v7 Preflight

- Feasibility judgment: the proposed critique is valid. The current pipeline uses LiDAR
  target -> VAE latent as the main supervision and uses radar voxel + IR image only as an
  LDM/CD condition. Without direct IR geometry/visibility/structure constraints, the model
  can learn coarse occupancy while mostly ignoring IR for tree trunks, upper canopy, top
  height, and vertical continuity.
- Calibration protocol risk confirmed and fixed:
  - only `calib_radar_to_thermal.txt` is now treated as real IR projection calibration;
  - `calib_radar_to_livox.txt` is recorded as available but no longer counted as real
    thermal calibration;
  - dataset meta now includes `calib_source`, `calib_is_thermal`,
    `has_thermal_calib`, `has_livox_calib`, `calib_fallback_reason`, `velocity_m_s`,
    `dt_sync_us`, and `sync_displacement_x_m`.
- Dataset audit now reports dataset-loader IR coverage separately from compatible IR file
  coverage, plus `mock_ir_ratio`, `mock_calib_ratio`, calibration source, fallback reason,
  and an estimated IR frustum voxel ratio. This prevents future reports from accidentally
  claiming real IR fusion when training actually used mock IR or mock calibration.
- Fusion network change:
  - `ir_gate` now receives `[radar_cond, ir_feat_3d, confidence]` instead of only
    `[radar_cond, confidence]`;
  - the projection layer can return the frustum mask and the multimodal network caches
    `last_ir_frustum_mask` for diagnostics and optional loss supervision.
- LDM supervision change:
  - added optional `decoded_ir_frustum_occupancy_weight` and
    `decoded_ir_frustum_top_weight`;
  - these losses supervise only target-positive voxels/columns inside the IR frustum,
    so they should encourage visible-structure recall without becoming another global
    over-density term;
  - defaults remain `0.0`, so old experiments are not silently changed.
- LDM training metrics now include `mock_ir_ratio`, `mock_calib_ratio`, and
  `ir_frustum_voxel_ratio`; warning logs are emitted if mock IR/calib dominates an epoch.
- Added `test/ablation/diagnose_ir_condition_ablation.py`: same frame, same random seed, compare
  real IR vs zero IR vs mock IR. If output differences are near zero, v7 is not actually
  using IR; if differences are large but metrics degrade, projection/calibration/features
  are likely wrong.
- Added `test/mini-test/run_ldm_z64_v7_ir_experiment.sh` as the next guarded experiment
  runner. It reuses the Z=64 VAE, starts from v6-style structure weights, and adds small
  IR-frustum occupancy/top weights.
- Review fixes before v7:
  - added `migrate_ir_gate_state_dict()` so old multimodal checkpoints with the previous
    `[radar_cond, confidence]` gate can still be loaded after the new
    `[radar_cond, ir_feat_3d, confidence]` gate change;
  - the migration preserves old radar/confidence weights and initializes the new IR gate
    slice to zero, so old checkpoints keep near-legacy behavior until retrained;
  - IR ablation now preserves bool/int/float `is_mock_ir` and `is_mock_calib` metadata
    instead of silently treating missing tensor flags as real IR/calibration;
  - audit mock-calibration frustum ratio now matches the dataset loader's actual mock
    projection path, including the default 0.01 m sync compensation;
  - `run_ldm_vertical_experiment.sh` now prints and forwards the two IR-frustum weights.

## 2026-07-10 Z64 LDM v7 IR Experiment Review

- The guarded v7 experiment completed 10 epochs and 500-frame loop3 inference with real IR
  and real radar-to-thermal calibration. Training logs report `mock_ir_ratio=0`,
  `mock_calib_ratio=0`, and mean `ir_frustum_voxel_ratio=0.607824`.
- At threshold 0.85, v7 changed the vertical metrics relative to v6:
  - height coverage recall: `0.6066 -> 0.7475`;
  - vertical connectivity recall: `0.6091 -> 0.7506`;
  - trunk recall: `0.7490 -> 0.8311`;
  - top-height recall: `0.0659 -> 0.0449`.
- A paired 500-frame analysis found the first three improvements statistically significant,
  while top-height significantly regressed. Near-field recall, precision, BEV IoU, and Chamfer
  also improved, but the aggregate prediction/target count ratio worsened and remains severely
  over-dense. The fixed CD gate therefore remains `HOLD`.
- Both v6 and v7 validation threshold sweeps select 0.99 by task BEV F1. At this same threshold:
  - v7: height `0.6364`, top `0.0628`, connectivity `0.6410`, trunk `0.7169`;
  - v6: height `0.2167`, top `0.0659`, connectivity `0.2226`, trunk `0.3061`.
  This confirms that v7 materially improves body/column structure beyond a threshold artifact,
  while top height remains essentially unchanged.
- Post-v7 IR ablation on train indices 0, 100, and 300 shows that the model responds strongly
  to real IR. Replacing real IR with zero IR lowers mean occupancy from `0.0143-0.0163` to
  `0.0081-0.0096`; mock IR lowers it to `0.0095-0.0116`.
- The current ablation reports only output difference and occupancy mass. It cannot yet prove
  that the extra IR-driven occupancy is closer to the LiDAR target rather than merely denser.
  The next gate is target-aware real/zero/mock IR comparison before designing v8.

## 2026-07-10 Target-Aware IR Ablation Runner

- Extended `test/ablation/diagnose_ir_condition_ablation.py` without removing its legacy single-frame
  CSV/JSON outputs. It can now load the model once, sample multiple frames deterministically,
  and write per-frame plus micro-aggregated target metrics for real/zero/mock IR.
- Added target-aware metrics: voxel precision/recall/F1/IoU, BEV precision/recall/F1/IoU,
  prediction/target count ratio, height coverage, top height, vertical connectivity, and trunk
  recall. This does not change supervision, checkpoints, target voxels, or formal inference.
- Added `test/mini-test/run_ldm_z64_v7_target_ablation.sh`. Defaults are validation split,
  32 evenly spaced frames, the v7 Z64 checkpoint, 20 Euler steps, near-field `0-40m`, and the
  calibrated occupancy threshold 0.99.
- The 1-frame/1-step smoke demonstrates why the aggregate test is needed: real IR improved
  height coverage (`0.6929`) over zero (`0.2901`) and mock (`0.4165`), but also raised the
  count ratio to `9.65`. The 32-frame result must decide whether the structural gain is stable
  enough to justify v8 while explicitly accounting for over-density.

## 2026-07-10 32-Frame Target-Aware IR Ablation Decision

- The full 32-frame validation ablation completed at occupancy threshold 0.99.
- Micro-aggregated real/zero/mock results were:
  - voxel precision: `0.0704 / 0.0535 / 0.0590`;
  - voxel recall: `0.6644 / 0.3322 / 0.4574`;
  - voxel IoU: `0.0680 / 0.0483 / 0.0552`;
  - BEV IoU: `0.1677 / 0.1125 / 0.1454`;
  - count ratio: `9.44 / 6.21 / 7.75`;
  - height coverage: `0.6644 / 0.3322 / 0.4574`;
  - top height: `0.0737 / 0.0526 / 0.0537`;
  - vertical connectivity: `0.6679 / 0.3384 / 0.4646`;
  - trunk recall: `0.6874 / 0.3676 / 0.5036`.
- Paired-frame evidence shows real IR provides stable target-aligned structure rather than only
  changing the output:
  - real beats zero on voxel IoU in 81.2% of frames and on BEV IoU in 90.6%;
  - real beats mock on voxel IoU in 78.1% and on BEV IoU in 71.9%;
  - real beats zero on voxel recall, height coverage, and connectivity in all 32 frames;
  - Wilcoxon tests for these coverage/IoU gains are significant.
- The gain is still accompanied by over-density. Real IR predicts about 52% more occupied
  voxels than zero IR and 22% more than mock IR. BEV precision versus mock is not significantly
  better, and top-height improvement versus zero is not significant.
- The immediate root cause is loss/metric mismatch, not lack of IR influence:
  - current top-height loss only makes the target top voxel positive;
  - the strict evaluation rejects columns whose predicted top extends above the target top;
  - current density loss only suppresses completely empty target columns, so above-target
    occupancy inside a valid obstacle column is not penalized;
  - current IR-frustum occupancy auxiliary is positive-only and therefore favors recall/density.
- Decision: v8 should first add above-target overshoot suppression and balanced IR-frustum
  negative supervision. Thermal semantic/edge loss and IR-backbone pretraining remain valid
  later options, but they are not the smallest or best-isolated next experiment. Do not change
  default config until the v8 protocol is validated, and keep CD on hold.

## 2026-07-11 LDM v8 Loss Alignment Implementation

- Added an above-target top-overshoot loss. It supervises only non-empty target columns and
  only voxels above each column's LiDAR target top, so it directly matches the strict top-height
  evaluation failure observed in v7. The sigmoid path uses stable logits BCE; empty selections
  return a graph-connected zero.
- Added a separate IR-frustum negative occupancy loss while retaining the existing positive
  recall loss. Negative supervision applies only where the IR frustum is valid and the LiDAR
  target is exactly zero; soft target values are excluded from the negative class.
- Both new weights default to zero and are enabled only by the v8 experiment runner. The global
  default config and old experiment behavior therefore remain unchanged.
- The two losses reuse the existing once-per-batch VAE decode. Their values are recorded in LDM
  metrics and the effective loss configuration is persisted in checkpoints.
- Supervision impact: v8 adds negative evidence above target tops and in visible IR background.
  It does not modify LiDAR targets, source radar voxels, Z64 grid dimensions, or target voxel
  counts. Expected metric effect is lower prediction/target count ratio and higher precision,
  top-height recall, and BEV IoU, with a possible recall reduction if negative weights are too high.
- Added a guarded Z64/500-frame/10-epoch v8 runner. Experiment-local dataset/config scratch paths,
  destructive-path validation, canonical output paths, symlink rejection, non-empty-output checks,
  and an atomic experiment lock prevent accidental deletion, overwrite, or concurrent corruption.
- Code readiness does not prove model improvement. The remaining evidence gate is to run v8, then
  compare it with v7 using the same threshold protocol, 32-frame target-aware IR ablation, 500-frame
  task/vertical metrics, and raw-LiDAR 3D overlays. CD remains on hold.

## 2026-07-11 Z64 LDM v8 Evaluation Decision

- v8 completed 10 epochs and produced all 500 loop3 predictions. Training was numerically stable;
  total loss fell from `0.318846` to `0.063394`. Top-overshoot and IR-negative components also fell,
  but epoch 10 was still improving and no validation loss was recorded.
- Fine threshold scanning selected `0.99995` by unconstrained validation BEV F1. This is a quality
  optimum, not a safe airborne operating point: the 500-frame voxel recall at this threshold is only
  `0.2855` even though BEV F1/IoU improve to `0.4421/0.2838`.
- At the same threshold `0.99`, v8 versus v7 reduces prediction/target ratio from `8.6593` to
  `5.0675`, raises voxel IoU from `0.0705` to `0.0977`, and raises BEV IoU from `0.2163` to
  `0.2548`. Top-height recall rises from `0.0628` to `0.1166`.
- The tradeoff is too strong for the final mapping task: height coverage falls from `0.6364` to
  `0.5403`, connectivity from `0.6410` to `0.5546`, and trunk recall from `0.7169` to `0.5658`.
  Therefore v8 solves over-density/top alignment but introduces structure under-prediction.
- The 32-frame v8 ablation proves real IR remains useful. Real IR beats zero/mock in voxel and BEV
  overlap, height, top, connectivity, and trunk metrics. Real and mock have almost identical count
  ratio (`2.1274` vs `2.1160`) while real has better precision/IoU, so real-IR gains are not merely
  extra density.
- Root decision: do not add thermal edge supervision or replace the IR backbone yet. Keep the IR
  path, hold CD, and tune the two new suppression weights one variable at a time. Also replace pure
  BEV-F1 threshold selection with a recall-constrained safety operating point.
- Full report: `test/result/ldm/ablation/ldm_near40_500_z64_v8_balanced/v7_v8_evaluation_report.md`.

## 2026-07-12 Z64 LDM v9-A Evaluation Decision

- Recall-constrained threshold selection is now operational and preserves the old unconstrained
  selector when constraints are disabled. The v9-A validation quality optimum is `0.98`; the
  threshold satisfying global BEV recall `>=0.80` and 0-20m recall `>=0.90` is `0.70`.
- The safety threshold is not deployable: validation pred/target ratio is `12.48`, and the full
  500-frame ratio is `11.98`. It recovers trunk recall `0.7500`, but BEV IoU is only `0.1639`
  and top-height recall is `0.0827`.
- At the quality optimum `0.98`, the full-500 BEV IoU is `0.2272`, top-height recall is `0.0869`,
  and trunk recall is `0.2922`; all are below the required teacher gate and below the relevant
  v8 structure/quality results.
- The fixed-threshold comparison also rejects v9-A: at `0.99`, v9-A BEV IoU is `0.2132` versus
  v8 `0.2548`, top recall is `0.0605` versus `0.1166`, and trunk recall is `0.1760` versus
  `0.5658`.
- The 32-frame ablation still confirms useful real-IR conditioning. Real IR reaches BEV recall/
  IoU `0.4372/0.1699`, compared with zero IR `0.1813/0.1001` and mock IR `0.2356/0.1385`.
  The failure is therefore a suppression/calibration tradeoff, not an inactive IR branch.
- Decision: v9-A does not replace v8 and CD remains HOLD. Before another loss experiment, select
  among saved epoch checkpoints using fixed validation task/structure metrics instead of minimum
  training loss. Full report: `test/result/ldm/ablation/ldm_near40_500_z64_v9a_top_full/v8_v9_evaluation_report.md`.

## 2026-07-12 LDM Validation Checkpoint Selection

- Added a real-IR-only diagnostic mode and a checkpoint selector that binds every result to the
  fixed validation protocol, exact checkpoint/VAE hashes, dataset manifest, and 32 deterministic
  sample indices. The runner only accepts a fresh output directory and uses a run lock; it never
  copies or overwrites model weights.
- Training-loss checkpoint selection was confirmed to be suboptimal. The structure-aware selector
  chose epoch8 instead of epoch10. On the 32-frame validation subset epoch8 reaches BEV IoU/recall
  `0.3451/0.8371`, top recall `0.1572`, trunk recall `0.5159`, and ratio `1.9266`.
- No saved epoch passes every teacher gate. Epoch8 fails trunk recall (`0.5159 < 0.65`), while
  epoch9 has higher BEV IoU `0.3733` but lower worst-gate satisfaction due to trunk `0.4728`.
- Full loop3 evaluation confirms a real but insufficient improvement. Epoch8 at quality threshold
  `0.98` reaches BEV IoU `0.2497`, near recall `0.6934`, top `0.0921`, and trunk `0.3167`, improving
  epoch10 but remaining below the v8 quality baseline and teacher gates.
- Epoch8 at safety threshold `0.80` reaches near recall `0.9026`, top `0.1049`, and trunk `0.7317`,
  but count ratio rises to `11.14` and BEV IoU falls to `0.1813`. No threshold solves the joint
  recall/density/structure problem.
- This stage changes checkpoint selection and reporting only. It does not alter supervision,
  target occupancy count, voxel dimensions, or existing checkpoint files. CD remains HOLD.

## 2026-07-12 Column-Balanced Structure Loss Design

- The epoch8 Pareto curve proves that threshold calibration alone cannot meet the joint gate:
  safety recall/trunk requires a count ratio near `11`, while a reasonable count ratio loses the
  obstacle body. The next loss must separate column existence recall from empty-column precision.
- The recommended v10 objective uses a temperature-controlled Z-axis logmeanexp column logit and
  separately averaged positive/negative BCE terms. Separate class means prevent the many empty
  columns from overwhelming sparse true obstacle columns.
- The new objective does not replace height/top/connectivity supervision. Column-positive BCE can
  preserve whether a structure exists at `(X,Y)`, while existing vertical losses remain responsible
  for its Z extent and continuity.
- v10 will set the old decoded empty-column density weight to zero in its runner to avoid duplicate
  negative supervision. Global defaults remain unchanged and both new weights default to zero.
- Success is defined by a shifted Pareto curve, not training loss: safety near recall `>=0.90` with
  ratio `<=6`, quality BEV IoU `>=0.2548`, top `>=0.10`, trunk `>=0.65`, connectivity `>=0.60`,
  plus at least 8/10 raw-LiDAR overlays showing recognizable basic obstacle structure.
- Detailed implementation plan:
  `docs/superpowers/plans/2026-07-12-ldm-column-balanced-structure-loss.md`.

## 2026-07-13 Column-Balanced Loss Task 1

- Added the pure `decoded_column_balanced_losses()` helper without connecting it to LDM training.
- The helper converts sigmoid logits or clamped raw probabilities into float32 voxel logits, then
  applies temperature-controlled Z-axis logmeanexp aggregation.
- Positive and negative target columns are averaged independently, preventing the numerous empty
  columns from overwhelming sparse obstacle columns. Missing classes return graph-connected zero.
- Temperature is restricted to `[1e-3, 100]`; empty spatial dimensions, device mismatch, invalid
  threshold, layout, and activation values are rejected before aggregation.
- This task does not modify supervision files, target occupancy counts, grid dimensions, model
  architecture, or checkpoint loading. Training behavior is still unchanged because Task 2 wiring
  has not yet been implemented.

## 2026-07-13 Column-Balanced Loss Task 2

- Integrated separate `column_positive_loss` and `column_negative_loss` components into the shared
  LDM decoded-loss path. Both reuse the existing single VAE decode.
- New trainer fields default to positive/negative weights `0/0` and temperature `1.0`; therefore
  historical configs preserve their previous loss and decode behavior.
- The raw loss components are logged separately, while total loss applies the configured weights.
  Effective values are persisted through the existing `ldm_loss_config` checkpoint metadata.
- Formal trainer construction rejects negative or non-finite weights and temperatures outside the
  Task 1 stability range. Mini/config wiring remains intentionally deferred to Task 3.
- This wiring changes supervision only when explicitly enabled. It does not change target voxels,
  target counts, Z64 dimensions, model channels, or checkpoint tensor shapes.

## 2026-07-13 Column-Balanced Loss Tasks 3-4

- Added mini-config controls and a guarded v10 A/B runner. Both variants preserve the v9-A data,
  VAE, Z64 grid, random seed, and structure weights; only the column-negative weight changes from
  `0.01` (A) to `0.02` (B). The column-positive weight is fixed at `0.02`, temperature at `1.0`,
  and the old empty-column density weight is disabled to avoid duplicate negative supervision.
- The v10 runner is training-only and requires a fresh experiment directory, fresh scratch/config
  paths, a non-empty VAE checkpoint, and a non-empty final LDM checkpoint. It does not start
  inference, ablation, visualization, or CD.
- A real 2-frame/1-epoch finite-gradient smoke completed successfully. The first batch reported
  total loss `2.0402`, raw column-positive loss `7.4342`, and raw column-negative loss `0.0020`.
  Both components were finite, backward completed, and a checkpoint was saved.
- The raw scale shows the intended positive-column protection is active immediately, while the
  empty-background term starts much smaller. This is not enough to choose A or B: the 3-epoch
  screens must compare the resulting validation Pareto curve, count ratio, BEV IoU, and vertical
  structure instead of comparing raw training-loss magnitudes alone.
- These changes affect only the enabled LDM supervision signal. They do not modify LiDAR targets,
  radar inputs, target occupied-voxel counts, the `64x128x128` grid, or existing checkpoints.

## 2026-07-13 Z64 LDM v10 A/B Screen Decision

- Both 500-frame, 3-epoch screens completed and were evaluated on the same deterministic 32-frame
  validation subset with real IR, 20 Euler steps, seed 42, and occupancy threshold `0.99`.
- v10-A epoch3 passes all five checkpoint-selection gates: BEV IoU/recall `0.4127/0.9109`, top
  recall `0.1178`, trunk recall `0.8084`, connectivity `0.7790`, and count ratio `3.3617`.
- v10-B epoch3 also passes all gates and lowers the count ratio to `2.9682`, but the stronger
  negative-column weight suppresses useful structure: versus A3, voxel recall is 30.5% lower,
  connectivity 29.4% lower, trunk recall 22.4% lower, and BEV IoU 5.9% lower. Its only meaningful
  structural advantage is a small top-recall increase (`0.1211` versus `0.1178`).
- v10-A is therefore the screen winner. Compared with the previous v9 epoch8 validation candidate,
  A3 improves BEV IoU `0.3451 -> 0.4127`, trunk `0.5159 -> 0.8084`, and connectivity
  `0.5191 -> 0.7790`. The tradeoff is a higher count ratio (`1.9266 -> 3.3617`) and lower top
  recall (`0.1572 -> 0.1178`).
- This is evidence that separate column-positive/negative supervision shifts the validation Pareto
  curve toward recognizable obstacle bodies. It is not yet independent-test evidence: no loop3
  500-frame inference, threshold sweep, raw-LiDAR overlay review, or CD training has been run for
  v10.

## 2026-07-13 Z64 LDM v10-A Full Reproducibility Finding

- The isolated v10-A 10-epoch run completed normally and saved all ten checkpoints. Training loss
  decreased monotonically from `0.3642` to `0.1445`; no NaN, CUDA, or checkpoint failure occurred.
- Fixed 32-frame checkpoint selection did not reproduce the screen result. No full-run epoch passed
  all gates. Epoch2 was selected by worst-gate satisfaction with BEV IoU/recall `0.3415/0.8019`,
  top `0.1040`, trunk `0.6112`, connectivity `0.5856`, and count ratio `2.6070`.
- Epoch8 was the other near-Pareto candidate (`0.3738` BEV IoU, `0.8217` recall, `0.1338` top), but
  trunk/connectivity remained `0.5678/0.5676`. Epoch10 became too conservative: count ratio fell to
  `0.9786`, while BEV recall/trunk/connectivity fell to `0.6524/0.3384/0.3590`.
- Root cause inspection found that `split_seed=42` only fixed train/validation indices. The training
  entry did not seed Python, NumPy, PyTorch, CUDA, or DataLoader shuffle, so the successful screen
  and failed full run used different model initialization and batch order despite identical loss
  settings.
- Added a unified training seed protocol. `data.training_seed` now controls Python/NumPy/Torch/CUDA
  RNGs and an independent DataLoader generator; it falls back to the existing `split_seed` for old
  configs. This changes stochastic initialization/order only, not targets, voxel counts, model
  shapes, or checkpoint compatibility.
- Because the unseeded full run fails the gate, loop3 500-frame inference and CD remain HOLD. The
  next experiment must first prove same-seed short-run reproducibility before spending another
  10-epoch run.
- Two independent 2-frame/1-epoch runs after the seed fix produced identical CSV metrics. Only the
  final output convolution weight/bias differed at CUDA floating-point roundoff scale; the maximum
  absolute tensor difference was `7.04e-08`. This is accepted as numerically reproducible and is
far too small to explain the previous screen/full metric divergence.

## 2026-07-13 Test Directory Organization

- 按 `test/AGENTS.md` 将 14 个功能脚本分类到 `evaluation/`、`diagnostics/`、`ablation/`、`visualization/` 和 `utils/legacy/`。
- 将 22 个回归、接口、协议和损失测试移动到新增的 `test/unit/`；该目录是对既有规则树的最小补充。
- 保持 `test/mini-test/`、不确定结果目录、锁目录、临时目录和顶层未登记结果原位。
- 将用途已确认的 VAE/LDM 结果只调整上级分类，保留所有实验叶目录、checkpoint、日志、CSV、JSON、HTML、图片和帧级数组。
- 修复了移动后的 Python import、测试路径、mini-test Shell 调用和默认结果路径；历史实验报告内的原始路径记录未改写。
- 这次整理不改变监督信号、target voxel 数量、体素尺寸、模型结构或训练参数。
- 最小 `--help` 验证发现 IR 消融脚本移动后项目根路径多上移一级，已修正为 `test/ablation/` 对应的两级父目录。
- 复查默认输出时补齐了 `inference_minimal.sh` 的正式结果回退路径和 legacy 损失图的输入路径，均不再使用旧的 `Result/` 根目录。

## 2026-07-13 Seeded v10-A Three-Epoch Validation Gate

- The fresh seeded v10-A run completed three epochs without runtime or numerical failure. Training
  loss decreased from `0.416545` to `0.180452`, but lower training loss did not translate into
  better validation structure.
- The fixed 32-frame selector chose epoch1, and no epoch passed all gates. Epoch1 achieved BEV
  IoU/recall `0.3846/0.5638`, top recall `0.0794`, trunk recall `0.4164`, connectivity `0.3655`,
  and count ratio `1.0311`. Epoch2 and epoch3 retained even less obstacle structure.
- The count ratio near one shows that gross over-density has been controlled. The failed gates are
  now caused by false-negative columns and missing vertical obstacle bodies, not excessive output
  density. The current positive/negative column balance is therefore too conservative for this
  fixed seed and training trajectory.
- Do not authorize another 10-epoch run, 500-frame test, raw-LiDAR acceptance, or CD from this
  candidate. The next experiment must be a bounded seeded weight screen that reduces empty-column
  suppression and/or strengthens occupied-column protection while keeping the same data split,
  VAE, seed, threshold, and 32-frame validation protocol.
- This evaluation changed no supervision targets, voxel counts, model shapes, checkpoints, or
  result files; it only establishes that the current v10-A loss weights are not robust enough.

## 2026-07-13 Seeded Column-Loss Calibration Runner

- Extended the existing guarded v10 training-only runner with two isolated calibration variants:
  C uses column positive/negative weights `0.03/0.01`, while D uses `0.02/0.005`.
- C changes only occupied-column protection relative to A; D changes only empty-column suppression.
  All other Z64, VAE, data split, training seed, epoch count, and structure-loss settings remain
  fixed, so validation differences can be attributed to the selected column weight.
- Unknown variants are rejected before the experiment directory, VAE copy, or training command is
  reached. Existing output, locking, scratch-path, and final-checkpoint guards remain active.
- The change affects only explicitly selected future LDM supervision weights. It does not alter
  LiDAR targets, occupied-voxel counts, input channels, model shapes, or existing checkpoints.

## 2026-07-13 Seeded v10 C/D Calibration Result

- Both isolated 500-frame garden screens completed three seeded epochs and saved every checkpoint.
  C (`positive=0.03, negative=0.01`) finished at training loss `0.1792`; D
  (`positive=0.02, negative=0.005`) finished at `0.1772`. Training loss was not used for selection.
- Every checkpoint was evaluated with the unchanged 32-frame validation protocol: real IR, 20
  Euler steps, seed 42, occupancy threshold `0.99`, Z64, and the same validation indices.
- C selected epoch3 but passed only 2/5 gates: BEV IoU/recall `0.3227/0.5747`, top `0.0996`,
  trunk `0.2493`, connectivity `0.3031`, and count ratio `1.0168`. Stronger occupied-column loss
  did not restore the obstacle body within three epochs.
- D selected epoch1 and also passed only 2/5 gates: BEV IoU/recall `0.3817/0.5888`, top `0.0867`,
  trunk `0.3806`, connectivity `0.3663`, and count ratio `1.0894`. Weakening empty-column
  suppression slightly improved recall versus seeded A, but remained far below the structure gates.
- Both selected candidates have near-target point-count ratios, so excessive density is no longer
  the limiting failure. The remaining error is false-negative obstacle columns and missing vertical
  bodies. Continuing to tune only the positive/negative column weights is not justified by these
  results.
- No C/D checkpoint is authorized for 500-frame loop3 evaluation, raw-LiDAR acceptance, 10-epoch
  continuation, or CD. The next design step should revisit optimization dynamics or use a
  curriculum/recall constraint rather than another one-dimensional column-weight sweep.

## 2026-07-13 v11 Column Curriculum Design

- The approved v11 direction uses an epoch-wise linear curriculum instead of another fixed-weight
  screen. For three epochs, positive/negative weights are exactly `0.03/0.00`, `0.025/0.005`, and
  `0.02/0.01`.
- The curriculum will be opt-in and disabled by default. Historical configs retain their fixed
  column weights and checkpoint behavior.
- The design requires effective per-epoch weights in metrics and checkpoint metadata, so resumed
  training and result audits can reconstruct the actual supervision schedule.
- This changes only LDM supervision timing. It does not change LiDAR targets, target voxel counts,
  Z64 geometry, VAE/model parameters, inference inputs, or evaluation thresholds.
- The written specification is stored at
  `docs/superpowers/specs/2026-07-13-ldm-column-curriculum-design.md`. A selective documentation
  commit was not created because the existing Git index already contains many unrelated staged
  test-file moves; committing would have exceeded the intended scope.

## 2026-07-13 Result Directory Organization Completion Notes

- All affected historical result-record path fields were updated from moved absolute experiment
  locations to their current relative category paths. Metrics, checkpoints, logs, images, point
  clouds, NPZ, HTML, and configuration payloads were not changed.
- The two remaining `*.lock` files intentionally stay beside the V10-D experiment because its
  runner checks `${EXP_DIR}.lock` and `${EXP_DIR}.v10.lock`.
- Safe validation is the only remaining task; training, full inference, and overwrite-prone
  commands remain intentionally unexecuted.
- Final filesystem verification no longer finds the two V10-D lock directories recorded during the
  audit. They were not recreated because an empty lock directory could incorrectly block a future
  run; this external state discrepancy needs user confirmation if those lock markers are required.
