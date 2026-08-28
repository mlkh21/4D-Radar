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
- The implementation plan keeps the curriculum calculation as a pure function, computes effective
  weights once per epoch, and records effective weights separately from raw positive/negative loss
  components. This avoids changing the existing loss math or decoding frequency.
- `test/AGENTS.md` prefers extending an existing script for parameter variants, so v11 will extend
  the guarded column experiment runner rather than copy it into another near-duplicate script.

## 2026-07-13 v11 Curriculum Task 1

- Added a pure `column_curriculum_weights()` helper with no trainer side effects. It implements the
  approved epoch-linear interpolation and returns the historical final weights when disabled.
- The helper strictly validates integer epoch bounds, a real boolean enable flag, and finite
  non-negative start/final weights. Boolean weight values are explicitly rejected before Python can
  coerce them to `1.0/0.0`.
- The exact three-epoch curve is `0.03/0.00`, `0.025/0.005`, `0.02/0.01`; a one-epoch enabled run
  uses the start weights.
- This task changes no effective training supervision yet because the trainer is not connected. It
  does not alter target voxels, occupied-voxel counts, Z64 tensor dimensions, model parameters,
  checkpoint format, or inference behavior.

## 2026-07-13 v11 Curriculum Task 2

- `OptimizedLDMTrainer` now computes the positive/negative column-loss weights once per epoch and
  forwards those effective values to the existing loss function; the underlying column loss math
  and decode frequency are unchanged.
- The curriculum remains opt-in. When disabled, the effective values exactly equal the historical
  fixed final weights, preserving old configurations and training behavior.
- Epoch metrics now record both effective weights, and best/periodic checkpoints persist the full
  schedule plus the effective epoch values. This makes resumed or audited runs self-describing.
- This changes only the timing and auditability of LDM supervision. Target voxels, occupied-voxel
  counts, Z64 geometry, VAE/network shapes, inference inputs, and evaluation thresholds are not
  modified.

## 2026-07-13 v11 Curriculum Task 3

- The existing guarded v10 column experiment runner now also provides a `V11` variant; no duplicate
  experiment script was introduced. V11 fixes the three-epoch schedule to positive `0.03 -> 0.02`
  and negative `0.00 -> 0.01`.
- Historical A-D variants explicitly disable the curriculum and use start weights equal to their
  final weights, so their supervision remains fixed and reproducible.
- The mini config generator strictly parses the curriculum enable flag and emits all three new YAML
  keys. Ambiguous boolean text now fails before training rather than being silently coerced.
- Runner protocol tests show hostile environment values cannot override V11's sample count, epochs,
  Z64 grid, ranges, split, scene, or curriculum weights. The runner remains training-only.
- These script changes do not modify targets, target occupied-voxel counts, VAE/model shapes,
  inference data, or evaluation metrics; they only expose the approved supervision schedule.

## 2026-07-13 v11 Curriculum Final Verification

- Resume metadata now includes `curriculum_total_epochs`. New-format checkpoints must match the
  curriculum enable flag, total epochs, and all four start/final weights before model or optimizer
  state is loaded; legacy checkpoints with incomplete curriculum metadata remain compatible.
- The final two-frame, one-epoch smoke produced finite loss `0.535496`. CSV, best checkpoint, and
  epoch checkpoint all recorded effective weights `0.03/0.0`; both checkpoints recorded
  `curriculum_total_epochs=1`.
- V11 changes only the epoch timing of weighted positive/negative column supervision. Targets,
  occupied-target voxel counts, Z64 tensor size, VAE/model architecture, inputs, and inference
  threshold protocol are unchanged. Prediction voxel density may change by design and remains
  controlled by the fixed 32-frame count-ratio and structure gates.
- The smoke proves the training/log/checkpoint path, not obstacle-structure improvement. Formal
  three-epoch results must pass the unchanged fixed 32-frame gate before 500-frame evaluation,
  visualization acceptance, or CD.
- Residual audit-only issue: the extended runner retains `.v10.lock` and a `v10 training complete`
  message. This does not affect V11 locking or training semantics but can be renamed in a later
  maintenance-only change.

## 2026-07-15 v11 Fixed 32-Frame Evaluation

- The formal seeded V11 run completed all three epochs with the intended effective column weights:
  `0.03/0.00`, `0.025/0.005`, and `0.02/0.01`. Training loss decreased from `0.418417` to
  `0.178341`, but training loss was not used for checkpoint selection.
- The unchanged fixed validation protocol evaluated every epoch on the same 32 garden frames with
  real IR, 20 Euler steps, seed 42, occupancy threshold `0.99`, and the established Z64 crop.
- No epoch passed all five gates. The selector chose epoch2 by maximum worst normalized gate
  satisfaction, but it passed only 2/5: BEV IoU/recall `0.3245/0.4778`, top `0.0644`, trunk
  `0.2474`, connectivity `0.2347`, and prediction/target count ratio `0.6872`.
- Relative to the seeded v10-A epoch1 reference, V11 epoch2 is lower by 15.6% BEV IoU, 15.3% BEV
  recall, 18.9% top recall, 40.6% trunk recall, and 35.8% connectivity. Voxel recall also falls
  36.9%; precision changes only -5.3%.
- The curriculum did not solve missing obstacle bodies. Its best candidate is under-dense rather
  than over-dense, so 500-frame loop3 evaluation, raw-LiDAR 3D acceptance, and CD remain blocked.
- This evaluation did not alter supervision targets, target voxel counts, model/checkpoint weights,
  or the threshold protocol. It only generated new fixed-protocol diagnostic files.

## 2026-07-15 v11 Threshold-Calibration Diagnostic

- A bounded threshold diagnostic on the selected epoch2 checkpoint shows a strong calibration
  shift. At thresholds `0.99/0.95/0.94/0.93/0.925`, the candidate passes `2/5`, `3/5`, `4/5`,
  `4/5`, and `5/5` gates respectively on the same fixed 32 validation frames.
- At `0.925`, metrics are BEV IoU/recall `0.2919/0.8278`, top `0.1193`, trunk `0.6504`, vertical
  connectivity `0.6147`, and prediction/target ratio `3.1422`. All five configured gates pass,
  although voxel precision remains low at `0.1881`.
- This does not retroactively pass the fixed `0.99` comparison protocol. It establishes a separate
  calibrated operating-point candidate and shows that V11 retained more obstacle structure than
  the `0.99` binary output suggests.
- Retraining or CD should not start yet. The next evidence should be independent-scene validation
  at both `0.99` and the preselected `0.925` threshold, followed by raw-LiDAR 3D inspection because
  the calibrated prediction is roughly 3.14 times the target occupancy count.

## 2026-07-15 v11 Independent loop3 Gate

- Evaluated the preselected epoch2 checkpoint on 32 evenly spaced loop3 frames, using real IR,
  20 Euler steps, seed 42, and the same Z64 geometry. The dataset loader explicitly selected only
  `loop3` from the two-scene dataset root.
- At `0.99`, loop3 passes only 1/5 gates: BEV IoU/recall `0.1669/0.2374`, top `0.0347`, trunk
  `0.0730`, connectivity `0.1268`, and count ratio `0.6181`.
- At the garden-calibrated `0.925`, loop3 still passes only 1/5: BEV IoU/recall `0.1869/0.5478`,
  top `0.0667`, trunk `0.3151`, connectivity `0.3654`, and count ratio `3.1010`.
- Frame-level instability is substantial at `0.925`: median trunk recall is `0.2784`, trunk P10 is
  zero, and per-frame prediction/target ratio P90 is `8.31`. A single lower threshold therefore
  trades missed structure for uncontrolled false positives on some loop3 frames.
- The garden `0.925` operating point does not generalize. The main remaining issue is cross-scene
  representation/calibration shift, not another scalar threshold or another small column-weight
  adjustment. Raw-LiDAR 3D acceptance, 500-frame inference, and CD remain blocked.

## 2026-07-15 garden/loop3 Distribution and Doppler Audit

- Added a sparse-space audit that deterministically samples 500 paired frames per scene and measures
  Radar/target occupancy, range and height bands, Doppler, Doppler variance, IR intensity, calibration,
  and frustum coverage inside the exact near40 physical crop.
- loop3 has 38.9% fewer Radar voxels and 75.7% fewer target voxels than garden. Its mean Radar/target
  ratio is 0.973 versus 0.312, and its target distribution shifts from garden's near 0-10 m band toward
  10-40 m. This confirms a large supervision-density/domain shift behind the independent-scene failure.
- IR is present for every sampled frame, both scenes use the same real thermal calibration, and the Z64
  near40 frustum coverage is 0.608. IR availability is not the dominant cross-scene difference.
- The raw first-frame Doppler is zero in garden and approximately -0.002 m/s in loop3, while the matched
  preprocessed frames are approximately -47.9 and -46.8 m/s. The current preprocessor's default fixed
  `vx=50 m/s` compensation is therefore physically incompatible with this NTU ground-platform data.
- In 70-76% of sampled frames, the per-frame P90 of voxel Doppler variance is still zero. Most occupied
  voxels contain one Radar point, so within-voxel variance cannot carry the intended uncertainty signal.
- Full garden retraining is blocked. The data protocol must first use explicit none/fixed/recorded
  egomotion modes and validate corrected 32-frame outputs; recorded GNSS velocity also requires timestamp
  matching and world-to-sensor rotation before use.

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

## 2026-07-15 TODO/26 Priority Audit

- `TODO/26-7-15.md` is GB18030/GBK text rather than UTF-8. Read-only conversion recovers all 684
  lines; the source file was not rewritten.
- The review lists 6 P0, 10 P1, 5 P2, and 2 P3 issues. Its explicit first-stage order differs from
  the handoff's Doppler-next order, so the TODO takes priority per the current user request.
- Actual `unified_train.py` confirms P0-01: both base datasets use `split='train'`, then
  `deterministic_split_indices()` partitions frames from the same scene. With a one-scene training
  root, adjacent correlated frames can cross the train/validation boundary.
- Actual `inference_ldm.sh` confirms P0-04: it defaults to `Data/NTU4DRadLM_Pre`, sets
  `USE_MULTIMODAL_META=0`, and requires target/raw-LiDAR inputs for the launcher path.
- Actual `inference.py` confirms conditional P0-06: `--adaptive_occ_from_target` loads each frame's
  target occupancy count and derives the prediction threshold from that test target. The flag is
  opt-in, but enabling it is oracle evaluation leakage.
- `deterministic_split_indices()` is a seeded `torch.randperm()` over individual sample indices.
  The formal launcher links only `garden`, and both base datasets use `split='train'`, so the
  randomness is reproducible but does not prevent temporally adjacent leakage.
- A read-only adjacency-ratio diagnostic was stopped after three command-level failures. The first
  omitted a shell export; the next two counted zero files because the ignored data path was not
  enumerated as expected. No dataset file was read or changed, and the static root-cause evidence
  is sufficient for the design decision.
- The worktree already contains 9 modified tracked files and 5 untracked files; these historical
  V11/audit changes must be preserved. No production or test code has been modified in this audit.

## 2026-07-15 P0-01 Temporal Block Split Design

- The approved minimal scope keeps `loop3` as the independent test scene and changes only the
  `garden` train/validation member assignment from a seeded random permutation to ordered prefix/tail
  blocks.
- The design intentionally does not change Dataset scene discovery, target generation, augmentation,
  model structure, checkpoint schema, mini runners, or launch scripts.
- Per-sample supervision and voxel counts remain unchanged. Validation metrics may decrease because
  the new protocol measures temporal extrapolation instead of randomly interleaved neighboring frames.
- The written specification is
  `docs/superpowers/specs/2026-07-15-temporal-block-validation-split-design.md`.
- The self-review found no placeholders, contradictions, ambiguous split semantics, or scope drift.
  The specification was committed alone as `d363650`; no historical V11 or TODO change entered that
  commit.

## 2026-07-15 P0-01 Implementation Planning

- The implementation plan uses one RED/GREEN cycle around a pure
  `temporal_block_split_indices(dataset_size, train_split)` API, then wires the existing `Subset`
  construction to it.
- The plan keeps `training_seed` behavior unchanged and deliberately adds no split seed, embargo gap,
  scene grouping, Dataset change, launcher change, or mini-config plumbing.
- Because `unified_train.py` already contains historical V11 edits, the implementation commit must use
  selective hunk staging and prove the cached diff excludes all curriculum changes.
- The plan is stored at
  `docs/superpowers/plans/2026-07-15-temporal-block-validation-split-implementation.md`.
- Plan self-review found full spec coverage, no placeholders, and consistent helper/test signatures.
  The 296-line plan was committed alone as `96df62c`.

## 2026-07-15 `26-7-15.md` 分阶段修复续作审计

- `TODO/26-7-15.md` 共 684 行，内容为 GB18030/GBK 编码；本轮通过 `iconv` 管道只读审阅，未改写源文件。
- 当前工作区已有未提交的训练脚本、测试、三份 TODO 和新增诊断文件；这些均视为用户或既有任务改动，后续不得覆盖或混入无关重构。
- 近期提交 `d363650` 与 `96df62c` 已分别固化 P0-01 连续时间块切分的设计和实施计划；当前工作区还显示对应实现/测试改动，需先验证而不是重复实现。
- 审计文件按四阶段列出问题。第一阶段要求依次处理：时间块切分、禁用 target 自适应阈值、数据 manifest、正式真实 IR 推理与部署/评价解耦、正式 checkpoint 链。
- 下一项修复必须保持单一根因、先 RED 后 GREEN；涉及监督或 target 的变更需单独说明监督信号、体素数量及评价指标变化。
- P0-01 的测试文件差异仍保留 RED 阶段使用的模块级 `getattr()` 包装；是否已完成 GREEN 与最终重构不能只靠截断 diff 判断，需聚焦检查实现符号、主入口调用和测试结果。
- P0-01 RED 已实测：21 项中仅 3 个时间块契约测试失败，统一原因为 `temporal_block_split_indices` 不存在；这证明测试确实覆盖缺失行为，而不是语法、导入或环境故障。
- P0-01 GREEN 已实测：新 helper 生成有序前缀/后缀、划分完整且互斥，并且训练随机种子变化不再改变成员归属；主入口仍保留 `training_seed` 控制模型与 DataLoader 随机性。
- 此修改不改变 target 内容、每帧 occupied 体素数、数据集总样本数、80/20 数量比例、网格尺寸、模型或 checkpoint；只改变 train/validation 的成员协议，历史随机交错验证指标不再可直接比较。
- 审计文件第一阶段按风险依赖排序，而不是按问题编号排序；P0-01 后的下一项是 P0-06（禁止正式结果使用 `adaptive_occ_from_target`），随后才是 manifest、正式真实 IR 推理和 checkpoint 链。

## 2026-07-15 P0-06 根因审计（只读）

- `inference.py` 将 `--adaptive_occ_from_target` 作为公开布尔 CLI；开启后要求 `target_voxel_dir`，逐帧加载 LiDAR 派生 target，并用 target 占据体素数量反推该帧预测阈值。
- `find_adaptive_occ_threshold()` 通过预测 occupancy 的第 k 大值使输出点数尽量匹配 target 数量；这不是普通评价读取，而是让测试真值直接改变部署输出。
- 正式 `inference_ldm.sh` 当前没有开启该 flag，所以默认正式入口未触发泄漏；但通用推理 CLI 和 `test/mini-test/inference_minimal.sh` 仍允许无显式 oracle 标记地开启。
- `compare_with_target` 只读取 target 计算指标，可以保留；P0-06 应只隔离“target 改变预测阈值”的路径，不能误删正常离线评价。
- 现有 unit tests 没有覆盖 adaptive flag 的准入策略或 oracle 标记；下一步设计应先增加 CLI/策略级失败测试，再修改实现。
- 可复用现有 `test/unit/test_multimodal_inference_interface.py` 测试通用推理协议，并在 `test/unit/test_mini_scripts_protocol.py` 静态约束 mini launcher；无需新建重复测试文件。
- 最小可逆边界是“双重显式 opt-in”：保留 oracle 诊断能力，但单独增加 `allow_oracle` 准入，缺少第二确认时在模型加载和输出目录创建前失败；正常固定阈值与 `compare_with_target` 均不受影响。
- 用户已选择更强的“移到独立诊断脚本”边界，上一条双重 opt-in 不再作为实施方案。
- 现有 `sweep_occ_threshold.py` 选择验证集统一的全局阈值；旧 adaptive 功能为每帧匹配 target 数量，二者统计协议不同，不应放在同一运行模式中。
- 推荐新增只消费已保存 `*_voxel.npy` 与 target 的 oracle 诊断入口；正式推理先保存不受 target 影响的预测体素，诊断脚本随后离线计算每帧 oracle 阈值、计数与明确标记的报告。
- 需要用户确认的兼容策略只有一项：旧 `--adaptive_occ_from_target` 是保留为 fail-fast 迁移提示，还是直接成为 argparse 未知参数。
- 用户确认采用“从 argparse 删除，但在解析前识别旧 flag 并输出迁移提示”，并要求独立诊断同时保存 CSV/JSON 与 oracle 点云。
- 正式 `inference_metrics.csv` 中 `target_occ_count` 仅由旧 adaptive 路径填写，可随迁移删除；`effective_occ_threshold` 在固定阈值模式仍有审计价值，应保留并恒等于 CLI 固定阈值。
- `find_matching_voxel_file()` 与 `voxel_to_pointcloud()` 仍服务正常 target 对比和固定阈值点云，不能随 adaptive 逻辑误删；`load_target_occ_resized()` 在 inference 中仅服务 adaptive，可移出正式入口。
- 额外发现：`sweep_occ_threshold.py` 及 `test_occ_threshold_grid_protocol.py` 仍复制旧随机 train/validation 切分协议，与刚完成的训练时间块切分不一致。这不是 oracle 迁移本身，但会影响全局阈值正式验证，需作为 P0-01 后续单独修复，不能混入 P0-06 实现。
- P0-06 设计已写入 `docs/superpowers/specs/2026-07-15-oracle-target-adaptation-diagnostic-design.md`；设计固定正式/诊断单向边界、旧参数迁移错误、CSV/JSON/点云输出协议、零 target 兼容语义和非空目录保护。
- P0-06 RED/GREEN 实施计划已写入 `docs/superpowers/plans/2026-07-15-oracle-target-adaptation-diagnostic-implementation.md`；计划分为正式入口 RED/GREEN、独立诊断 RED/GREEN和最终聚焦验证，明确不提交。
- 计划自审将新测试模块加载从可能与标准库 `test` 包冲突的名称导入，改为按绝对文件路径加载；规格/计划函数名与输出文件名一致。
- P0-06 Task 1 RED 已证实两个缺失行为：正式 inference 没有旧 oracle 参数迁移检查；mini launcher 仍会把旧环境变量传入原 adaptive 路径。
- P0-06 Task 2 GREEN 已将 oracle 计算、parser 参数、运行分支、日志和正式 CSV target count 移出 inference；固定阈值、正常 target/LiDAR 对比与点云转换保留。
- mini launcher 现在在 checkpoint 检查前拒绝两个旧环境变量，避免它们被静默忽略或进入生成路径。
- P0-06 Task 3 RED 的 6 项测试全部因独立诊断模块不存在而明确失败，覆盖算法、零 target 兼容、点云/报告、缺 target、错误 shape 与非空目录保护。
- 独立诊断首次 GREEN 揭示旧 `float(np.nextafter(float32_value, -inf))` 实际产生 float64 前驱；NumPy 与 float32 数组比较时标量被舍入回原值，严格 `>` 只保留 `k-1` 点。改为同 dtype `nextafter` 后才真正实现 top-k 意图。
- 独立脚本现可生成 `(N,4)` oracle 点云、逐帧 CSV 和 `deployable=false` JSON，并对缺 target、错误 prediction shape 和非空输出目录 fail-fast。
- P0-06 最终聚焦回归共 25 项通过（inference 接口 16、mini 协议 3、oracle 诊断 6）；相关 Python 编译、shell 语法与 `git diff --check` 均通过。
- 真实旧参数调用会在检查必填 checkpoint 前以 exit 2 退出，并指向 `test/diagnostics/occupancy/diagnose_oracle_target_adaptation.py`；正式帮助仅保留固定阈值接口。
- 本项不改变训练监督、target 内容、网格尺寸、每帧预测体素值、模型或 checkpoint。固定阈值正式输出的占据点数不变；oracle 点数只存在于独立、明确不可部署的诊断结果中。
- 正式 `inference_metrics.csv` 删除旧 adaptive 专用的 `target_occ_count` 列，因此旧表与新表的 schema 不完全一致；固定 `effective_occ_threshold` 及正常 target/LiDAR 评价指标继续保留。
- 未运行训练、预处理、完整推理或全量评价，也未暂存或提交本轮修改。阈值扫描脚本仍复制旧随机 validation 切分，应作为下一项独立修复处理。

## 2026-07-15 阈值扫描 validation 协议续修审计

- `unified_train.py` 已使用 `temporal_block_split_indices()`，正式训练的 validation 是排序样本的连续后缀，且不再由 `split_seed` 决定成员归属。
- `sweep_occ_threshold.py` 的 `select_evaluation_files()` 仍自行创建 `torch.Generator` 并执行 `torch.randperm()`；因此同一 `train_split` 下，阈值校准样本与正式 validation 已经不一致。
- `test_occ_threshold_grid_protocol.py` 仍把“匹配训练 randperm”写成正确契约，说明问题不是调用参数偶发错误，而是测试与脚本共同固化了已废弃协议。
- 当前工作区对 `sweep_occ_threshold.py` 和对应阈值测试没有现存差异，可以在不覆盖用户修改的前提下小步处理；`unified_train.py` 含 P0-01 与其他历史改动，不应为本项再次修改。
- 历史阈值命令把包含 `000000` 至 `000499` 的完整预测目录传给脚本，再由脚本选出 100 帧；因此改为连续 validation 后缀可以在现有输入协议内完成，不需要先新增文件清单格式。
- 历史示例实际对 `loop3` 预测目录执行 validation 子切分，但 P0-01 已把 `loop3` 定义为独立 test 场景；仅修正索引算法不能证明输入场景是训练场景的 validation。场景身份校验需要后续 dataset manifest 提供依据，本项至少应在帮助与输出元数据中明确调用方责任。
- `split_seed` 现在只服务已废弃的随机成员选择；是否从阈值 CLI/JSON 删除属于兼容性选择，需在设计阶段明确，不能留下“参数存在但实际无效”的静默行为。
- 用户确认彻底删除 `split_seed`，并批准旧参数在 argparse 解析前给出迁移错误；新 JSON 用显式 `split_protocol` 代替随机种子元数据。
- 已写入 `docs/superpowers/specs/2026-07-15-threshold-sweep-temporal-validation-design.md`。占位符、内部一致性、范围和歧义自审通过，暂存区保持为空。
- 已写入 `docs/superpowers/plans/2026-07-15-threshold-sweep-temporal-validation-implementation.md`；计划将成员切分与 CLI/JSON 迁移拆成两个独立 RED/GREEN，再做集中验证。
- 计划自审发现成员选择示例曾包含省略号占位，已替换为完整校验实现；最终占位符、接口签名、规格覆盖和空白检查通过。
- `sweep_occ_threshold.py` 现按输入顺序返回连续 train 前缀或 validation 后缀；既有纯数字帧、连续帧、非空划分和划分后 `max_files` 保护均保留。
- `split_seed` 已从选择函数、准备函数、argparse、主调用链和新 JSON schema 删除。源代码中的字符串只用于精确识别旧参数并给出迁移错误，不再影响成员选择。
- 新推荐 JSON 使用 `split_protocol=temporal_block_prefix_train_suffix_validation`；旧 JSON 的 `split_seed` 字段不再生成，因此历史与新 schema 需分开解释。
- 最终聚焦测试 30/30 通过，相关 Python 编译、真实 CLI 迁移、帮助文本、`git diff --check` 和空暂存区检查均符合设计。
- 修改不改变监督信号、target、网格尺寸、每帧预测体素内容、模型或 checkpoint；相同 `train_split` 的样本数量不变，但成员和聚合指标可能变化，历史推荐阈值不可直接比较。
- 本轮没有运行训练、完整推理或全量阈值扫描，没有重写历史结果。历史 `loop3` 校准仍不自动视为合法 validation，后续 manifest 必须提供场景身份和预处理版本的机器校验。

## 2026-07-15 Dataset Manifest 根因审计

- 正式训练 launcher 固定读取 `Data/NTU4DRadLM_Pre_sensor_aware`，正式 LDM/CD 推理 launcher 仍固定读取 `Data/NTU4DRadLM_Pre`；同一 checkpoint 链的训练和推理预处理根目录天然不一致。
- 预处理脚本已经为每个场景写 `preprocess_policy.json`，Dataset 也会把它作为可选字典塞进 meta，但缺失时静默使用空字典，不校验场景名、协议版本、源索引或产物完整性。
- 现有 `audit_dataset_protocol.py` 只报告 policy 是否存在以及 IR/标定覆盖，不产生不可变文件清单，也不让训练/推理入口 fail-fast。
- Dataset 的场景发现只要求 `radar_voxel` 与 `target_voxel` 目录存在；随后按目录文件名组样本。目录内混入不同预处理批次时，当前加载链无法识别。
- 项目已有实验级 `dataset_manifest_sha256()`，但其输入为相对路径、大小和 mtime，适合确认候选实验使用同一目录快照，不足以作为可跨复制验证的预处理产物内容协议。
- 真实 `Data/NTU4DRadLM_Pre_sensor_aware` 没有任何 `preprocess_policy.json` 或 `target_policy.json`。因此现有 Dataset 对 garden/loop3 都静默返回空 policy，无法知道实际生成参数。
- sensor-aware garden 的 4014 个 radar 文件全部是指向旧 `NTU4DRadLM_Pre/garden/radar_voxel` 的绝对符号链接；loop3 的 `000000` 至 `000119` 共 120 帧同样链接旧根，剩余 6330 帧为普通文件。
- sensor-aware garden 的 target/lidar/IR 各有 4014 个普通文件；loop3 的 target/lidar/IR 各有 6450 个普通文件。Radar 单模态已明确包含不同来源形态，且 symlink 目标可被外部目录修改，不能视为不可变数据集。
- 现有 metadata 单测允许 policy 缺失并验证 mock fallback；若 formal 路径改为 strict manifest，必须把“通用 Dataset/临时单测兼容”与“正式 launcher fail-closed”区分，避免破坏诊断和小测试。
- `data_loading_config.yml` 明确 garden=train、loop3=test。三个正式推理 launcher 均仍指向旧 Pre；这属于下一阶段真实 sensor-aware/IR 入口修复，manifest 本项不应通过猜路径替代它。
- 用户选择严格方案 1：per-scene、per-frame 内容 SHA-256 manifest；正式入口缺失或不匹配时直接拒绝，不提供 warning-only 或 legacy adoption。
- 设计已写入 `docs/superpowers/specs/2026-07-15-strict-dataset-manifest-design.md`，并补充不可覆盖的原子发布及预处理非空输出目录 preflight，防止“先覆盖旧数据、后生成 manifest”。
- 规格的占位符、内部一致性、范围和歧义自审通过；暂存区保持为空，未创建提交。
- 计划前复核发现 `cm/__init__.py` 会导入整套 Torch/模型模块；为保持 manifest 核心纯标准库、避免预处理新增重依赖，核心文件位置调整为包根 `diffusion_consistency_radar/dataset_manifest.py`，CLI 路径不变。
- 严格 manifest 实施计划已写入 `docs/superpowers/plans/2026-07-15-strict-dataset-manifest-implementation.md`，拆为核心内容协议、CLI/预处理集成、四个正式 launcher gate 和最终验证四项。
- 计划自审补齐了目录级 symlink、非连续帧、policy/manifest 篡改和 provenance symlink 覆盖；最终无实现占位或旧 `cm/` 路径引用。

## 2026-07-15 Dataset Manifest 方案 1 实施结论

- 新增 per-scene manifest v1，以逐帧文件内容 SHA-256 固化 radar、lidar、target、IR 四模态；记录路径均为场景内相对路径，不含绝对路径、mtime 或生成时间，复制到其他根目录后仍可验证。
- 核心生成与验证会拒绝缺失/错场景 policy、模态错帧、非连续帧、未知文件、目录或文件 symlink、内容/大小/hash 篡改、缺失 provenance 和已有 manifest 覆盖。
- 预处理器只接受不存在或为空的场景输出目录，所有 worker 和 `preprocess_policy.json` 完成后才原子发布 manifest；任一场景失败会令主进程非零退出，未签署的失败输出保留用于诊断但不会被视为正式数据。
- 四个正式 launcher 没有兼容或跳过开关：训练在删除 `.tmp_train_dataset` 前验证全部 train scene，推理在调用 `inference.py` 前验证全部 test scene。
- 当前真实 sensor-aware 数据没有 policy/manifest，且 radar 中存在旧根 symlink，因此严格入口会按设计阻断。只读实测 `loop3` 返回 exit 2 且未创建 manifest；没有为 legacy 数据自动补签。
- 23 项聚焦测试全部通过，相关 Python 编译、shell 语法与差异格式检查通过。未运行长训练、完整预处理、正式推理或全量内容 hash。
- 监督信号、target、网格和每帧体素内容均未修改；现有体素文件数量也未变化。指标计算公式不变，但旧数据生成的指标缺少新协议来源证明，不可作为新正式实验结果直接比较或汇总。

## 2026-07-15 正式真实 IR 与部署/评价解耦初步证据

- 正式训练和正式推理的数据根不同：训练为 sensor-aware，LDM/CD/unified 推理仍为旧 Pre；真实 IR 因而没有进入默认正式推理数据流。
- `--use_multimodal_meta` 当前只是可选开关；LDM launcher 默认值为 0，CD launcher同样默认关闭，unified launcher没有传入该参数。
- IR 文件缺失不会在正式生成前失败，而会进入 mock thermal 路径；`is_mock_ir` 目前只参与元数据/不确定性处理，不能证明融合分支已关闭。
- target/LiDAR 只用于后处理评价，但正式 launcher 把它们与生成参数绑定，造成部署入口在结构上依赖离线真值。
- 模型 checkpoint 的 state dict 已可区分 `CompleteDualModalityPerceptionNet` 和 legacy 单模态 UNet；严格真实 IR 要求可以绑定到多模态 checkpoint，而不是无条件套到所有历史模型。
- `CalibrationProvider.load_with_metadata()` 已能拒绝把 radar-to-Livox 当作 thermal 外参，并报告 `is_mock_calib`；不过 thermal K/D 的真实解析属于审计第二阶段，不纳入当前第一阶段最小修复。
- `CompleteDualModalityPerceptionNet.forward()` 无论 `is_mock_ir` 为何都会编码、投影并门控 `ir_img`；该标志只增加 uncertainty，不能作为 Radar-only 降级开关。
- 现有离线评价实现分散：生成质量诊断覆盖 prediction/target/radar，垂直结构评估覆盖已保存 prediction/target，raw LiDAR Chamfer 仍嵌在 `inference.py` 的生成循环里。
- 将“评价 launcher”继续指向 `inference.py` 只会重新生成一次随机预测，并没有实现部署/评价数据流分离；正式评价应消费部署阶段已经保存的同一批 `*_voxel.npy`。
- `NTU4DRadLM_VoxelDataset._get_mock_calibration()` 对真实/回退标定都会增加 `0.01m` 同步位移，`load_multimodal_meta_for_radar()` 当前只在 mock 标定时增加；这会让正式真实 IR 投影偏离训练协议，需在第一阶段做一致性修复。
- 用户已复核并同意书面规格；实施拆为严格 IR preflight、纯 runtime 产物、已保存预测离线评价、正式 shell 边界和最终验证五个 TDD 单元。
- 实施计划固定所有公开接口和验证命令，并明确保留现有脏工作区、不暂存、不提交；当前尚未修改本项生产/测试实现。

## 2026-07-20 正式真实 IR 与部署/评价解耦实施结论

- `inference.py` 新增 `--require_real_ir`：正式模式先验证 checkpoint 多模态属性、全部待推理 frame 的普通 IR 文件、有限值/可接受维度和真实 thermal 外参，再创建输出目录；缺失项不回退 mock。
- 真实 thermal 外参 inference 现在与 Dataset 共同使用现有 `+0.01m` legacy x 同步补偿；该值已明确标注为固定历史协议，不等价于真实逐帧动力学同步。
- 无 target/raw LiDAR 评价参数的部署生成写 `inference_runtime.csv` 与 `inference_run.json`；显式兼容评价仍写历史 `inference_metrics.csv`。运行 metadata 固化实际 target/source/model grid、voxel size、阈值和模态信息。
- 新 `evaluate_saved_predictions.py` 只消费已保存 `*_voxel.npy`，严格配对 Radar/target frame，使用 metadata voxel size 转坐标，支持 fixed-threshold target/radar/near/raw-LiDAR/uncertainty 指标，并在输出前拒绝缺失、错配、非法数组、索引越界和非空目录。
- 三个正式生成 launcher 已统一 sensor-aware 根、manifest gate、`--require_real_ir` 和 `_deploy` 输出；新 `evaluate_inference.sh` 只调用保存预测 evaluator，target/raw LiDAR 参数不再进入部署生成。
- 监督信号、target 生成、模型结构、checkpoint、网格大小和数据文件数量未改变；正式预测值/占据点数可能因真实 IR 替换 mock/disabled IR 及训练一致的固定同步位移而改变，旧结果不可直接合并。
- 最终修正前聚焦回归为 50/50；协议细节修正后 formal protocol 已扩展为 10 项，下一步重跑总计 54 项聚焦回归和静态验证。

## 2026-07-20 P1-06 正式 checkpoint 链续修

- 只读审计确认 `Result/train_results/vae/vae_best.pt` 与 `Result/train_results/ldm/ldm_best.pt` 缺失；现有 `Result/train_results/cd/cd_best.pt` 只有旧的 216 个 legacy UNet state key，缺少 `model_config`、融合范围、Radar encoder 和 uncertainty head，不能与新 sensor-aware VAE/LDM 组成正式链。
- `test/result/archive/ldm_sensor_aware_partial_20260713/vae/vae_best.pt` 和 `ldm/ldm_best.pt` 虽然具备部分自描述字段，但不是当前正式路径，且 archive LDM 缺少当前 fusion 网格字段；没有将它们复制、重命名或伪装成正式权重。
- 新增 `diffusion_consistency_radar/checkpoint_chain.py`，协议名为 `formal_chain_v1`：逐阶段安全读取普通文件，验证 `data_grid_config`、LDM/CD `model_config`、VAE latent_dim、父 checkpoint SHA-256 和四类实际持久化多模态 state 前缀；投影几何由 fusion 配置校验，错误聚合后 fail-closed。
- 新增独立 `scripts/diagnose_checkpoint_chain.py`。默认只读 CPU metadata validate；`--construct` 才按 checkpoint 配置构建并严格加载三阶段模型，不读取数据、不执行 forward，报告只在成功且目标目录为空时原子发布。
- VAE/LDM/CD 新保存 payload 增加协议版本、stage、实际训练网格、fusion shape/range 和父权重 hash；LDM/CD 仍保留旧构造函数兼容性，但缺父 hash 或 legacy 模型的产物会被正式门禁拒绝。
- CD 训练器对 legacy 教师仍可兼容运行，但其新保存 payload 标记为 `legacy_cd_v0`，只有多模态学生才写 `formal_chain_v1`，避免把兼容产物误称为正式链。
- LDM/CD 训练入口只计算父 checkpoint SHA-256 并传入 trainer，没有启动训练；target、监督通道、体素数量、模型前向和指标算法均未改变。
- 三份正式生成 launcher 现在在 manifest/第一帧生成前调用 checkpoint-chain 门禁；unified 不再对缺失 LDM/CD 打印 warning 后跳过，而是整链直接失败。离线评价入口未改为加载 checkpoint。
- 新增协议测试覆盖有效链、网格/父 hash/legacy/symlink 拒绝、报告目录保护和 CPU strict construct 调用；checkpoint-chain 6/6、formal inference protocol 11/11 通过，VAE payload 22/22 通过，静态编译和 shell 语法通过。
- 该项没有生成或改写任何正式权重；下一步只需完成聚焦总回归和对当前 `Result/train_results` 的只读失败诊断，然后将正式 VAE/LDM/CD 重训作为显式长任务安排。

## 2026-07-20 P0-06 独立诊断依赖边界加固

- 复查发现 P0-06 诊断脚本虽然不执行模型 forward，但仍直接导入 `inference.py` 和 `sweep_occ_threshold.py`；这会把正式入口的模型/MPI 依赖带入离线 oracle 诊断，违背“只读取已保存预测与 target”的隔离目标。
- 新增 `diffusion_consistency_radar/diagnostics/occupancy_helpers.py`，仅依赖 NumPy/PyTorch，内置 target 稀疏加载、物理范围裁剪、通道感知重采样和体素转点云协议；没有模型、checkpoint、正式评价入口或数据集扫描依赖。
- `diagnose_oracle_target_adaptation.py` 改为只调用上述轻量辅助模块；保留原有 top-k 阈值、target 配对、CSV/JSON 和非空输出保护，避免复制正式脚本的导入副作用。
- 新增回归测试静态检查两类正式入口路径不出现在独立诊断脚本中；oracle 聚焦测试 7/7 通过。
- 本轮只改变诊断工具的依赖边界和代码位置，不改变监督信号、target 内容、网格尺寸、预测体素、正式固定阈值输出或任何指标公式；oracle 结果仍明确 `deployable=false`。
- 未运行训练、预处理、模型采样、全量阈值扫描，未修改或删除数据、checkpoint、日志和历史结果；未暂存或提交。

## 2026-07-20 P0-03 多普勒运动补偿协议修复与代码审查

- 根因确认：预处理 shell 的 `--vx=50` 虽可被参数覆盖，但旧调用链没有运动模式，默认会把固定速度传入每一帧 Radar/LiDAR 体素化；在当前 NTU 数据上会把接近零的原始 Doppler 推成约 `-47 m/s`，且没有速度来源或时间匹配证据。
- 新增 `NTU4DRadLM_pre_processing/motion_protocol.py`，把速度解析固定为 `none/fixed/recorded` 三种显式模式；`recorded` 只接受严格递增的 `timestamp,vx,vy,vz` 表，并要求最近邻时间差不超过 `velocity_max_delta`。
- 速度接口明确规定来源坐标系为 Radar 或 LiDAR；worker 在 Radar/LiDAR 对齐后只使用标定旋转转换速度，不把外参平移量加入速度。`align_to=radar` 现在同步变换 LiDAR 点云，避免两模态落在不同坐标系后再生成 target。
- 默认模式改为 `none`，`50 m/s` 仅保留为显式 fixed 参数；recorded 源文件 basename、行数、SHA-256、坐标系和时间容差写入 `preprocess_policy.json`/`target_policy.json`，便于审计实际补偿来源。
- 代码审查发现并修复直接执行接口冲突：脚本文件名会遮蔽同名包，导致 `python NTU4DRadLM_pre_processing/NTU4DRadLM_pre_processing.py --help` 无法导入新模块；现在包导入和文件路径执行均有明确回退路径。
- 同时增加旧 Namespace 的安全默认值、有限数校验、空帧零进程 fail-fast；不会把旧隐式固定补偿悄悄恢复。
- 监督信号、target 定义、体素网格尺寸和指标公式未改变；未来重新预处理时默认 Doppler/空间同步补偿会改变，这是预期的物理协议修正，现有数据、checkpoint、日志和结果未被改写。
- 未运行训练、完整预处理、推理或全量评价；未暂存或提交。

## 2026-07-20 P0-05 LiDAR 未观测空间与 free evidence 修复

- 根因复核确认：`SlidingProbabilisticGridMap.update_from_voxel()` 先把每个 BEV 单元的 `max_z(occupancy)` 作为观测，再把整张 reliability map 传给 D-S 融合；因此空白但未被 LiDAR/Radar 射线观测的单元会由 `1-p` 产生 free mass。原有 snapshot 也没有暴露 unknown mass，调用方无法区分“未知”和“低占据”。
- 采用保守兼容边界：没有显式 `observed_mask` 时只把当前帧确实含 occupied voxel 的 BEV 单元视为已观测，其余单元 reliability 置零并保留 D-S unknown/ignorance；显式 mask 支持 `(X,Y)` 或 `(X,Y,Z)`，且 occupied 单元即使漏标也不会被屏蔽。
- `streaming_map_update.py` 新增按帧 `<frame>_observed_mask.npy/.npz` 输入、有限值/shape 校验、CSV 的 mask 使用量与 `unknown_fraction`，快照/最终结果新增 `unknown_mass`。同目录 mask 会从 voxel 文件列表排除，mask 目录在输出目录创建前校验并拒绝 symlink，避免接口误配或输出副作用。
- `update_from_voxel()` 新参数追加在既有参数末尾，保持旧位置参数调用兼容；体素和 mask 在时间衰减及地图写入前校验，错误不会部分更新地图。稀疏 `.npz` mask 复用现有 `coords/features/shape` 协议。
- RED 阶段复现了旧行为：空白单元 `occ_prob` 被推向 free（约 0.3619），并且旧接口不接受 `observed_mask`；GREEN 后 P0-05 聚焦测试 12/12 通过。
- 本轮只修复地图更新的安全边界和 mask 输入契约，没有重生成数据、改变 target/监督通道、体素网格数量、模型结构或指标公式；现有无 mask 历史数据会更保守地保持 unknown。离线射线投射 mask 的生产与 VAE “只监督可见 free”训练链仍需以独立数据协议继续设计，不能由空白体素反推 free。
- 最终静态检查：相关 Python `py_compile`、streaming CLI `--help`、`git diff --check` 和空暂存区检查通过；未运行训练、预处理、推理或全量地图更新，未暂存/提交。

## 2026-07-20 P0-05 训练监督链续修

- 训练侧调用链确认：`NTU4DRadLM_VoxelDataset.__getitem__ → meta_dict → unified_train.train_epoch → VAE3D.compute_loss` 原先没有任何可见性 mask，`bce_dice` 会对所有 target 空白体素计算负类损失。
- Dataset 现在优先读取配对 `lidar_voxel`，从传感器原点向 occupied 端点投射 `(X,Y,Z)` observed mask；同一离散方向只保留最近端点，避免重复计算遮挡后的共线射线。缺少独立 LiDAR 文件时安全退化为 occupied-only mask，不把 target 空白推断为 free。
- mask 与 target 一起经过物理范围 crop、目标尺寸 resize，并作为 `occupancy_observed_mask` 放入 metadata；现有 Voxel/Cutout/Composed 几何增强在收到 mask 时同步 flip/rotate，避免监督错位。
- VAE 的 BCE、Dice、连续通道损失和 legacy MSE 可选接收 `observed_mask`；unknown 空白不参与负类监督，occupied target 始终强制保留。无 mask 时仍沿用旧全网格行为；trainer 仅在 batch 提供 mask 时传新关键字，兼容旧模型替身/旧 batch。
- RED 发现一个真实接口问题：旧 trainer 单测替身不接受新关键字；已改为无 mask 时走三参数旧接口，随后训练损失 20 项、Dataset metadata 11 项、概率地图 12 项、多模态投影 9 项和 sensor-aware target 4 项全部通过。
- 本续修改变新训练样本的监督有效区域：target 内容、occupied 体素数量、网格尺寸和模型输出通道不变，但未观测空白不再贡献 free 负类梯度；历史 checkpoint 不被修改，缺 LiDAR 的旧样本保持 occupied-only 保守语义。

## 2026-07-20 P1-01 多传感器时间戳对齐与容差修复

- 根因确认：`unpack_rosbag.py` 只使用 bag receipt time；Radar/LiDAR 索引和预处理 IR 匹配均按文件名最近邻且没有最大时间差，丢帧时会静默配入错误模态帧。
- RED 阶段新增 `test/unit/test_timestamp_alignment_protocol.py`，5 项均因缺少统一 helper/`generate_scene_indices` 而失败；没有读取真实 bag 或执行长预处理。
- 新增标准库模块 `NTU4DRadLM_pre_processing/timestamp_alignment.py`：优先 `msg.header.stamp`，支持 ROS `to_sec()` 与 `secs/nsecs`，无效时显式回退 receipt；最近邻 helper 校验有限、严格递增序列，返回实际绝对 delta，并在超出独立容差时抛出 `ValueError`。
- `NTU4DRadLM_timestamp_index.py` 改为按数值时间戳排序，先在内存完成全部 Radar/LiDAR 匹配，再原子写入两份索引和 `radar_lidar_sync.csv`；CSV 固化 Radar/LiDAR 时间戳、绝对 delta 和带符号 delta。默认 Radar-LiDAR 容差为 30ms，可通过 CLI/环境变量覆盖。
- `unpack_rosbag.py` 的点云/图像文件名和非点云 CSV 均改用 header 优先时间戳；直接文件执行使用相对模块导入，避免同名 `NTU4DRadLM_pre_processing.py` 遮蔽轻量 helper 并引入 ROS/OpenCV 隐式依赖。
- 预处理器新增独立 `--radar_ir_max_delta`（默认 20ms）和 `--radar_lidar_max_delta`；IR 匹配在主进程预计算并超限即失败，worker 使用已验证索引；输出写入 `radar_ir_sync.csv`，policy 记录两类容差和同步记录文件名。直接绕过 Step 1 时，缺失/错配/超限的 `radar_lidar_sync.csv` 也会 fail-fast。
- 代码审查将输出目录创建延后到所有时间和索引检查之后，避免失败场景留下不能重跑的半成品；保留 `dt_sync` 作为显式 legacy 固定补偿，真实逐帧 signed delta 已持久化，后续动力学补偿可按明确的时间方向约定消费，未擅自改变现有点云位移符号。
- 聚焦验证：时间戳协议 5/5、预处理运动协议 8/8、manifest 10/10、airborne 多模态 9/9、sensor-aware target 4/4、Dataset metadata 11/11 通过；相关 Python 编译、直接入口 `--help`、Shell 语法和 `git diff --check` 通过。
- 本项不改模型结构、checkpoint、target 通道或体素网格；未来重新解包/预处理时，header 时间戳、超限拒绝和帧配对变化可能改变有效帧数量、target 对应的 Radar/LiDAR/IR 样本和指标，旧结果不可与新同步协议直接混合。未重生成数据、未运行训练/完整预处理/推理/全量评价，未暂存或提交。

## 2026-07-20 P1-02 Thermal 标定与 IR 投影几何统一

- 根因确认：`CalibrationProvider` 只解析 Radar→thermal 的 R/T，K 硬编码为旧值；`Data/config/calib_cam_thermal.txt` 中的原始尺寸 `640×512`、K 和 D 未被使用，模型输入却固定为 `640×480`。训练 Dataset 与 inference 还分别复制了 `t_vec[0] += 0.01`。
- RED：新增 `test/unit/test_thermal_calibration_protocol.py` 3 项，分别锁定 K/D/S 解析缩放、去畸变效果和训练/推理同步函数身份；初始均因旧接口/硬编码而失败。
- `CalibrationProvider` 现在统一读取 `calib_cam_thermal.txt` 的 `S_00/K_00/D_00`，按输出尺寸缩放 K（fx/cx 按宽度，fy/cy 按高度），并在 metadata 中保存原始/输出尺寸、D、来源和是否具备真实内参。
- `_prepare_ir_array()`/`_resize_or_pad_ir_tensor()` 共用同一图像协议：先调整到 `640×480`，再用缩放后的 K 和 D 执行 OpenCV 去畸变；训练 Dataset 与逐帧 inference 均通过该函数，投影层接收与图像尺寸一致的 K。
- `apply_legacy_sync_compensation()` 成为训练和推理的唯一固定同步补偿函数；真实逐帧动力学同步仍未伪装成该 legacy 位移，现有 `0.01m` 协议数值保持不变。
- 严格 `--require_real_ir` 现在同时要求真实 Radar→thermal 外参和完整 `calib_cam_thermal.txt (S/K/D)`；非严格兼容路径可使用默认 K，但 metadata 明确标记 `thermal_intrinsics_source=default`。独立 `audit_dataset_protocol.py` 移除重复硬编码 K，改用 Provider 的实际结果。
- 回归通过：thermal 协议 3/3、multimodal inference 26/26、Dataset metadata 11/11、airborne 多模态 9/9、sensor-aware target 4/4；当前真实 `Data/config` 只读解析得到 K≈`[[471.964,0,339.031],[0,442.956,260.382],[0,0,1]]`、D 五项和 `640×512→640×480` 缩放。
- 监督 target、体素网格、模型结构和 checkpoint 未改变；未来重新训练/推理时 IR 像素采样会因真实 K/D 去畸变和缩放而变化，真实 IR 预测与指标不可和旧“硬编码 K/未去畸变”结果直接混合。未运行长训练、完整预处理或推理，未修改数据/结果，未暂存或提交。

## 2026-07-20 P1-03 PointCloud2 字段 schema 固定化

- 根因确认：`unpack_rosbag.save_pointcloud()` 用 `None` 占位后过滤字段，再把 `read_points()` 返回的短 tuple 直接保存；缺 intensity 时 Doppler 左移到 col3，后续体素化固定读取 `pcl[:,3]`/`pcl[:,4]` 后产生强度与速度错位。
- RED：新增 `test/unit/test_pointcloud_schema_protocol.py`，缺 intensity 与缺 Doppler 两个用例初始均得到 `(N,4)` 而非固定 `(N,5)`，证明测试能捕获原问题。
- GREEN：PointCloud2 按字段名和大小写不敏感别名读取（intensity/reflectivity/power/rcs/snr、velocity/doppler/v_r/radial_velocity），显式构造 `[x,y,z,intensity,doppler]`；缺失特征填零，缺少坐标则拒绝该帧。
- 每个 Radar PointCloud2 输出目录增加原子 `pointcloud_schema.json`，记录固定列顺序、源字段、实际映射、缺失字段、shape 和 dtype；`.json` 不会被时间戳索引当作点云帧。
- 下游 `voxelize_pcl_airborne_optimized()` 的五列接口、四通道体素监督和网格尺寸保持不变；修复只影响重新解包时的字段解释，历史错误列数据不会被自动改写或伪装修复，旧结果与新解包结果需按 schema 分开比较。
- 回归通过：PointCloud2 2/2、时间戳 5/5、运动协议 8/8、Airborne 多模态 9/9、sensor-aware target 4/4；Python 编译、`git diff --check` 和空暂存区检查通过。未读取真实 bag、未执行完整预处理/训练/推理、未修改数据和实验结果。

## 2026-07-22 P1-04 初始审计

- 审计目标来自 `26-7-15.md`：原始 9.6M 体素下采样到模型网格时，强度/Doppler 没有单位协议，现有 variance resize 只是普通插值，不满足合并后的二阶矩公式。
- 下游体素化固定把 col3/col4 聚合为强度均值、Doppler 均值，并以 `E[v²]-E[v]²` 生成局部方差；P1-04 需要先明确 Dataset 下采样究竟消费局部统计量还是重新聚合原始点，避免仅在网络入口做表面缩放。
- 当前工作区包含用户此前多项未提交修改；本项继续原地小步修改，不覆盖、暂存或提交既有变更。
- `resize_voxel_channels()` 当前对通道 1～3 统一执行 `interpolate(channel*occ)/interpolate(occ)`；这对均值通道近似成立，但对 variance 只是在插值局部方差，完全丢失不同细体素均值之间的离散项。
- 预处理体素已经保存局部 Doppler mean/variance，但没有保存每个细体素的原始点数。依据审计文档给出的 `E[Var_local + Mean_local²]-E[Mean_local]²`，现有四通道能实现“按 occupied 细体素等权”的二阶矩合并；若要求按原始 Radar 点数精确加权，则必须扩展原始体素协议，属于更大兼容性变更。
- Dataset 和逐文件 inference 都直接调用同一个 `resize_voxel_channels()`，因此方差修复必须位于共享 helper；否则训练/推理会发生隐式接口分叉。
- `NTU4DRadLM_VoxelDataset` 当前没有归一化统计参数；训练、CD 和 inference 都只传 target size/物理范围。若新增统计协议，必须在三个入口共享同一解析器，并把统计文件身份写入 checkpoint/运行 metadata，否则推理可能静默使用另一套量纲。
- 现有 `data_loading_config.yml` 只声明 `garden` 为 train、`loop3` 为 test；已有分布审计按稀疏体素读取 Doppler/variance，但没有统计 intensity 分位数，也没有生成可直接消费的训练集 normalization artifact。
- 动态按样本或按场景归一化会泄漏验证/测试分布并破坏物理幅值；硬编码常量则无法证明来源。更可审计的边界是：只从训练场景生成冻结 JSON，Dataset/inference 显式加载并记录 hash，正式路径缺失或不匹配时 fail-closed。
- 现有 v11 只读审计确认 garden/loop3 的 Doppler 均值约为 `-48.13/-53.54 m/s`，且局部 variance 均值仅约 `5.56e-4/4.03e-4`；这与旧预处理的固定自运动补偿协议一致，也证明 normalization artifact 必须绑定预处理 policy，不能跨新 `velocity_mode=none` 数据复用。
- 20 帧轻量抽样中，garden intensity 中位数约 `11.65`、p1～p99 约 `5.28～21.44`；loop3 中位数约 `10.44`、p1～p99 约 `5.02～21.41`。强度适合由训练场景冻结 `log1p + median/IQR`，而不是逐场景归一化。
- Doppler 建议用显式物理量程做对称缩放并裁剪，保留正负号；variance 保持 `(m/s)^2` 供现有不确定性头消费，并在 resize 中通过 `E[var+mean²]-E[mean]²` 重算。若同时把 variance 除以量程平方，现有 `UncertaintyHead` 的物理语义会被破坏。
- 正式 checkpoint 链目前只验证网格、父权重 hash 和多模态 state 前缀，没有 Radar normalization 协议。LDM/CD 的 Radar encoder 依赖输入量纲，因此两阶段必须携带并校验相同 normalization metadata/hash；VAE 不消费 Radar，可不绑定该统计。
- `inference_run.json` 当前记录网格、阈值和模态信息，但没有输入 normalization 身份。正式推理必须从实际加载的 LDM/CD checkpoint 取得协议并写入运行 metadata，不能仅从本机数据目录猜测。
- 仓库内未找到 Radar 硬件的无模糊 Doppler 量程或可作为权威来源的传感器型号配置；只有飞行速度 `35～70m/s` 的任务约束。因而不能把训练分位数或任意 `80/100m/s` 常量伪装成传感器物理量程。
- 当前 `Data/NTU4DRadLM_Pre_sensor_aware/{garden,loop3}` 都缺少 `preprocess_policy.json`，再次证明现有历史体素不能自动获得可信的运动/单位 provenance。新 normalization builder 应要求调用方显式给出正有限 `doppler_scale_mps`，并记录训练场景和输入 policy/manifest 身份；不在代码中猜默认值。
- 增强顺序存在隐形单位依赖：`VoxelAugmentation` 会对 target/condition 同时施加物理 Doppler shift，并对 condition 全通道加噪。Radar normalization 必须放在 resize 和增强之后；若先归一化，`0.1` 对 target 表示 `0.1m/s`、对 condition 却表示 `0.1*doppler_scale_mps`，两者不再一致。
- `default_config.yaml` 已有 `data.augmentation`，但 `unified_train.py` 没有把它传给 Dataset，当前实际使用 Dataset 内部默认 jitter。P1-04 不应顺便全面修复 P2-01，但必须让 normalization 的顺序对现有默认增强安全，并在设计中标出该剩余问题。
- 用户批准方案 1：冻结训练场景 artifact、Dataset/inference 共享入口归一化、LDM/CD checkpoint/hash 绑定；正式入口对缺失协议的旧 checkpoint fail-closed，Doppler 量程由配置显式给出且不猜默认值。
- 设计规格已写入 `docs/superpowers/specs/2026-07-22-radar-normalization-variance-resampling-design.md`；自审补齐 `formal` 标记、逐场景 manifest SHA-256 和目标文件原子发布边界，未发现 TBD、相互矛盾或范围外重构。
- 用户已复核并确认书面规格；进入详细 RED/GREEN 实施计划编写阶段，尚未修改生产代码。
- `RadarGenerator._load_model()` 当前丢弃生成 checkpoint 的非模型 metadata；实施时必须保存完整 checkpoint metadata/normalization spec 到 generator，再让 `load_radar_voxel_as_tensor()` 消费，不能从 state dict 反推。
- formal chain 的测试 fixture 可直接增加共享 `radar_normalization` 与 hash，并添加 LDM/CD 缺失和不一致用例；VAE fixture保持无 normalization，符合已批准边界。
- 计划前复核发现 Radar tensor 同时可能进入 VAE condition encoder 和多模态 `radar_encoder`；必须继续确认训练/推理中 `z_cond` 的实际使用，避免 normalization 只适配一条分支而破坏另一条隐形输入。
- 实际数据流已确认：`unified_train.py` 的多模态 LDM 分支会计算 `z_cond=vae.get_latent(cond)`，但模型调用不消费它；`cd_train_optimized.py` 的多模态 denoiser 同样忽略 `z_cond`；`inference.py` 只借该 latent 推导 shape，采样模型仍直接消费 Radar voxel。正式多模态链应删除这条无效 VAE 条件编码，训练从 `z_target`、推理从 VAE shape API 取得潜空间尺寸；显式 legacy 单模态诊断才保留 `z_cond`。
- 该收紧不会改变多模态模型实际前向结果，因为原 `z_cond` 未进入正式多模态 denoiser；它会消除一次无效 VAE 编码，并防止后续把规范化 Radar 数值误解释为 target 的 occupancy/intensity/Doppler/variance 语义。
- `resize_voxel_channels()` 还是 target、LiDAR observed mask 等通用通道的既有接口，不能把 Radar 方差二阶矩规则无条件套到所有四通道体素。实施应新增语义明确的 `resize_radar_voxel_channels()`，仅替换 Dataset condition、normalization builder 与逐文件 Radar inference 的调用；通用 target/mask resize 保持不变。
- `dataset_manifest.validate_scene_manifest(scene_dir, expected_scene)` 会重扫逐帧文件并返回已验证 manifest，其中 `content_sha256` 可作为 normalization artifact 的逐场景 provenance；builder 无需复制一套 manifest 校验实现。
- 实施计划自审移除了两个范围外分支：P1-04 不接通尚未生效的 augmentation YAML，也不让 shell 复制 Python 的 JSON/schema 校验。当前默认/直接传入增强仍必须满足“物理增强后归一化”，正式 fail-fast 以 Python 入口为唯一权威。
- 当前 condition 高斯噪声会连 occupancy 一起扰动，并可能把约 `1e-4` 量级 variance 变为负数；若不收紧，后置 normalization 无法维持 occupancy/variance 物理语义。P1-04 只保留 occupancy、将增强后 variance 限制为非负，其他 augmentation 配置接线仍留给 P2-01。
- Task 1 基线确认现有 Dataset/推理接口测试通过；Airborne 并行命令输出为空，不能仅凭并行调用完成就声称通过，后续独立复跑。第一批新测试只锁定可由现有四通道精确表达的 occupied-voxel 等权二阶矩，不引入原始点数这一不可用权重。
- Task 1 已建立严格四通道边界：专用 resize 用 `E[var+mean²]-E[mean]²`，normalization loader 同时约束 JSON schema、网格、显式 Doppler scale、formal 标记、manifest provenance 和真实文件 hash；target/mask 通用 resize 未改。
- Airborne 外部回归不是失败断言：前 5 项通过，第 6 项长时间停在融合前向且没有新输出，最终未取得完整退出码。P1-04 的纯函数与 Dataset metadata 回归均已独立通过，因此继续 Task 2，但最终审查仍需避免把 Airborne 记为全通过。
- Task 2 builder 复用 `validate_scene_manifest()` 的全内容重算结果并要求 Radar 帧数等于 manifest；统计严格按 `crop → resize_radar_voxel_channels → occupied log1p`，目标 JSON 只在全部统计和 spec 自校验完成后原子发布。`max_frames>0` 的小样本即使覆盖了全部现有帧也固定为 `formal=false`。
- 本轮仅在 `TemporaryDirectory` 生成小体素并调用 mock manifest，没有读取 `Data/` 真实场景，也没有猜测或写入实际 Doppler scale；正式 artifact 仍需用户后续明确硬件量程后单独生成。
- Task 3 首轮 GREEN 发现旧 `audit_dataset_protocol.py` 通过顶层 `cm.dataset_loader` 导入；此时 `from ..radar_normalization` 抛 `ValueError: attempted relative import beyond top-level package`，而不是 `ImportError`。Dataset loader 的兼容分支已同时捕获两类导入失败，继续支持包内正式导入和旧顶层诊断导入。

## 2026-07-22 P1-04 实施结论

- 新增严格 `radar_normalization_v1`：强度使用 occupied `log1p` 的冻结 median/IQR，Doppler 使用显式 `doppler_scale_mps` 对称缩放并裁剪；没有默认量程、运行时重估或 validation/test 泄漏。
- 新增 Radar 专用四通道 resize。occupancy 继续 max-pool，intensity/Doppler 按 occupied 权重合并，Doppler variance 按 `E[var + mean^2] - E[mean]^2` 计算总方差；通用 target/observed-mask resize 未被改写。
- artifact builder 只接受显式训练场景，先验证 scene manifest，再按真实 crop/resize 顺序统计；`max_frames>0` 的抽样产物固定 `formal=false`，已有路径和 symlink 均拒绝覆盖。本轮未生成真实正式 artifact。
- Dataset 默认缺 spec 即失败，物理增强完成后才归一化 Radar；occupancy 不再加入高斯噪声，occupied variance 保持非负，空体素仍为零。legacy 只能显式启用且进入 sample metadata。
- LDM/CD checkpoint 保存完整 spec 与 artifact 文件 SHA-256；CD 在输出目录前比较配置 artifact 与教师 LDM，resume 在加载 model/EMA/optimizer 前比较协议。VAE 不消费 Radar，因此 checkpoint 不绑定该字段。
- formal checkpoint chain 现在拒绝 LDM/CD 缺字段、缺 hash、非正式 spec、内容/hash 不一致和 VAE 错带 normalization；报告记录统一 protocol/hash。
- 推理只从实际 LDM/CD checkpoint 读取 spec，不接受 CLI 统计覆盖；逐文件和 Dataset 两条入口复用同一 Radar resize/normalize。`inference_run.json` 记录完整 spec/hash 与 `formal_protocol`，旧 checkpoint 仅能用显式 legacy 开关并标记非正式。
- 正式多模态训练/推理不再把 Radar condition 送入 target VAE；VAE 只编码 target 或提供公开 latent shape。legacy 单模态仍显式保留 `z_cond`，避免 `None` 拼接和 shape 接口错误。
- 代码审查额外发现并修复两处隐形依赖：LDM 原子保存后旧测试仍 mock 底层 `torch.save`；IR 消融 Dataset 固定 legacy、generator 默认 formal 且先创建输出目录。测试现改 mock 公共原子保存接口；消融先校验 generator，再继承其 grid/spec/hash 构造 Dataset，最后创建输出。
- mini train/inference 和两个历史 v7 诊断 runner 显式声明 legacy；正式 train 和三个 formal inference launcher 均不包含 legacy 开关。
- 监督与体素影响：target、observed mask、occupied 坐标、网格尺寸、每帧体素总数、四通道模型结构和损失/指标公式不变；变化只发生在 Radar condition 的 intensity/Doppler 数值尺度及下采样 variance。variance 新增组间均值差贡献，因此不确定性输入、预测及最终指标可能变化。
- 可比性边界：缺协议的旧 LDM/CD checkpoint 与新正式链不兼容；旧结果不能与新 normalization/variance 协议结果直接混合。默认 YAML 的空 artifact 路径和 null scale 是故意的 fail-closed 未配置状态。
- 最终聚焦验证通过 212 项 unittest 和 2 份直接接口测试；相关 Python 编译、9 份 shell 语法、`git diff --check` 与空暂存区检查均为 exit 0。未运行训练、完整预处理、正式 artifact 全量统计、模型采样或全量评价。

## 2026-07-22 P1-01 真实数据时间容差续修

- 用户首次执行 `preprocess-v2.sh` 时，严格索引在 garden 首个 LiDAR 时间戳处失败：目标 `1652439548.528784990`、最近 Radar `1652439548.579992056`，偏差 `51.207066ms` 超过当前 `30ms`。
- 该失败发生在索引生成、候选体素目录创建和全量预处理之前；当前证据尚不能判断是单个启动边界帧，还是 Radar/LiDAR 异步帧率使 30ms 对整个序列过严，禁止直接把阈值改成 60ms 后重跑。
- 全量文件名时间戳只读统计排除了“仅首个边界帧”解释：garden 的 4014 个 LiDAR 主轴帧有 1251 个最近 Radar 偏差超过 30ms，重叠区最大 `63.858ms`；loop3 的 6450 帧有 1858 个超过 30ms，重叠区最大 `63.322ms`。两场景各只有 2 个主轴帧位于另一传感器时间边界之外。
- Radar-LiDAR 最近邻偏差中位数为 garden `22.442ms`、loop3 `21.150ms`；P99 分别为 `57.958ms`、`45.643ms`。直接改为 60ms 仍会遗漏少量重叠区帧，而且会把来源不明的接收抖动合法化。
- Radar-IR 最近邻偏差也并非严格落在 20ms 内：garden 4816 个 Radar 帧中 52 个超过 20ms、最大 `22.401ms`；loop3 7738 帧中 96 个超过 20ms、最大 `22.927ms`。需先核验当前 Raw 文件名是否仍来自旧 receipt-time 解包。
- 当前 Raw provenance 已由 bag 首帧直接确认：garden `/radar_pcl` header=`1652439548.553762913`、receipt=`1652439548.579992294`，现有文件名为 `1652439548.579992.npy`；`/livox/lidar` header=`1652439548.559700966`、receipt=`1652439548.528785467`，现有文件名为 `1652439548.528785.npy`。现有 Radar/LiDAR/IR 文件名都匹配 receipt，而非 P1-01 新实现要求的 header 优先协议。
- 因此 30ms 大量失败的根因是当前 `Data/NTU4DRadLM_Raw` 属于旧解包产物，包含各 ROS 通道不同的传输/写包时延；直接放宽阈值会掩盖旧时间源。应从原始 bag 用新 `unpack_rosbag.py` 解包到独立 Raw 候选目录，并在切换前验证 header-based 30ms/20ms 分布。
- 对 4 个原始 bag 的全部相关消息直接读取 header 后，30ms 仍不成立：garden Radar-LiDAR 重叠区中位/P99/max 为 `26.472/39.440/43.569ms`，1527 对超过 30ms；loop3 为 `20.631/43.391/81.049ms`，1790 对超过 30ms。原因是约 12Hz Radar 与 10Hz LiDAR 的异步节拍，正常最近邻上限接近半个 Radar 周期 `41.7ms`，不是代码故障。
- loop3 的 `81.049ms` 最大值明显超过正常半周期窗口，符合 Radar 掉帧/间隙异常；不应为了保留该帧把全局阈值提高到 85ms。更合理的协议是以约 45ms 接受正常异步相位，并显式拒绝、记录少量掉帧型候选，而不是任意一帧超限就使整个场景无产出。
- header-based Radar-IR 的 P99/max 为 garden `20.518/22.461ms`、loop3 `19.969/22.658ms`；20ms 分别拒绝 70 和 72 帧。Thermal 约 25Hz，理论半周期 20ms 加少量抖动，25ms 是比当前 20ms 更符合采样周期的候选上限，仍需以测试锁定。
- receipt-header 偏差呈明显通道差异：garden Radar 中位 `+10.922ms`、LiDAR `-28.874ms`、IR `+5.886ms`；loop3 Radar `+8.146ms`、LiDAR `-48.970ms`、IR `+5.733ms`。这进一步确认必须重解包为 header 时间源，不能继续基于旧 Raw 调参。
- 按新解包脚本实际的 6 位文件名精度只读重算，45ms 门禁仅拒绝 garden `1/4014`（`0.0249%`）和 loop3 `18/6450`（`0.2791%`）个 LiDAR 主轴候选，均明显低于 1% 门禁；最大偏差约 `64.248/81.049ms`，会进入 `radar_lidar_rejected.csv` 而不会被强行配对。
- 最终协议采用 Radar-LiDAR `45ms + skip_unmatched + 1% reject gate`、Radar-IR `25ms fail-closed`。前者适配 12Hz/10Hz 正常异步相位且保留掉帧审计，后者覆盖 25Hz Thermal 半周期与已测抖动。
- 代码审查发现正式 conda 环境缺少 pandas，而解包器在解析 `--help` 前就导入 pandas；同时 open3d 仅服务于已注释的预览代码。已改用标准库 `csv.DictWriter` 并保留动态字段并集，移除两项无效依赖，正式解包入口现可直接启动。
- 解包器遇到任一损坏 bag 由“继续并最终报告成功”改为立即抛错；v2 脚本在索引前同时检查 Radar、LiDAR、Thermal 三类场景目录，避免不完整数据延迟到体素 worker 才失败。
- 最终聚焦回归共 37 项通过：时间戳 8、PointCloud/解包 4、运动协议 8、manifest 10、Thermal 3、sensor-aware target 4；相关 Python 编译、两个 shell 语法、索引/解包直接 `--help`、`git diff --check` 和候选目录不存在保护均为 exit 0。
- 监督与数量影响：按当前 bag 只读统计，新索引预计保留 garden `4013`、loop3 `6432` 对，分别审计拒绝 1 和 18 对；target 定义、每帧网格尺寸和通道数不变，但帧成员及 Radar/LiDAR/IR 配对会改变。后续新 normalization、checkpoint 和指标不得与旧 receipt-time 数据链混用。
- 本轮没有执行完整解包、预处理、normalization 全量统计、训练、推理或评价，没有创建候选 Raw/体素目录，也没有删除或覆盖旧数据、checkpoint、日志和结果。

## 2026-08-20 P1-05 移动平台局部地图更新启动

- 当前 `SlidingProbabilisticGridMap` 的状态全部是 `(X,Y)`；`update_from_voxel()` 对输入 `(X,Y,Z,C)` 直接沿 Z 取 max，并要求输入 XY 与地图完全同形，没有接收传感器到局部地图的位姿。
- `streaming_map_update.py` 使用 `timestamp=i*dt`，只把 `odom_cov_trace` 作为观测可靠度折扣；没有逐帧真实时间戳或 `T_local_body` 输入，所以移动平台上的静态障碍会按机体系网格索引直接叠加。
- P0-05 已在同一调用链加入 observed/free/unknown 语义和 sidecar 读取；P1-05 必须保留这些未提交改动，位姿 warp 后 observed mask、occupancy、uncertainty 与 DEM 需要使用同一坐标映射。
- 最小兼容边界：保留现有 2D `occ_prob/belief/plausibility/unknown_mass` 作为分层状态的 BEV 聚合输出，同时新增 `(X,Y,Z)` 分层证据；旧调用方不传位姿时继续采用单位变换，正式 pose-aware 模式则必须逐帧提供有效位姿和时间戳。
- 仓库内 `SlidingProbabilisticGridMap` 只有 streaming 入口和单元测试调用，现有快照消费者只读取既有 2D 键；因此可通过“保留旧键、追加 layers/pose metadata”扩展而不破坏现有离线消费者。
- 既有 roadmap 明确 ROS/PX4/HIL 需等离线地图在 35/50/70m/s 档稳定后再设计，本轮不越界新增 ROS service/action；先完成可由离线测试验证的局部坐标、时间和高度协议。
- 位姿输入拟采用每帧 CSV：`frame,timestamp,tx,ty,tz,qx,qy,qz,qw`，四元数表示从当前 body 到固定 local map 的旋转；帧覆盖、时间严格递增、有限数、单位四元数和刚体矩阵均在创建输出目录前校验。
- 第二批 GREEN 暴露旧布局猜测的隐形依赖：推理保存的是 `(C,Z,X,Y)`，预处理体素是 `(X,Y,Z,C)`；旧 `to_xyzc()` 只按首/末维是否小于 8 猜测，小尺寸或低分辨率输入会歧义并静默交换 XY。正式入口需支持显式 layout，auto 遇到两种解释同时成立时拒绝。
- P1-05 最终实现采用固定 `local` 地图系和严格 body→local CSV；帧覆盖、时间递增、四元数归一化、刚体方向、体素布局、prior DEM shape 和 target 帧覆盖均在输出目录创建前校验。`map_run.json` 记录 pose hash、方向、layout、网格、阈值和指标坐标系。
- 权威地图状态新增 `(X,Y,Z)` 的 occupancy/belief/plausibility/unknown 四层，旧 `(X,Y)` 键继续保留。`128×128×32` 网格新增持久分层状态约 `4×524288×4 = 8 MiB`；稀疏 warp 只为 observed/occupied 单元构造坐标，避免每帧为全网格建立多组 float64 meshgrid。
- 代码审查修复了三态质量不一致：旧实现会把未观测 `p=0.5` 重新解释成 occupied/free 各半的高可靠先验，使 unknown 从 1 降到 0.1。现在直接融合已有 D-S 质量，使用 `m_occ + 0.5*m_unknown` 输出 pignistic occupancy，并让时间衰减把 occupied/free 质量转回 unknown。
- 最近障碍查询在提供 `z_m` 时使用三维层和当前 body 原点；逐帧 target 点先用同一 `T_local_body` 变换到 local，再与分层地图评价。现有 obstacle precision/recall 仍由 `occupancy_prf` 在 BEV XY 单元上计算，因此数值会因位姿对齐和阈值概率语义修正而变化，不能与旧 body/`z=0` 日志直接混合。
- `streaming_map_update.py` 的直接入口不再导入完整 `cm/__init__.py` 训练栈，`--help` 不触发 Torch/OpenMPI。单帧输入若带 batch 维必须 `batch=1`，不再静默丢弃其余样本。
- 本项不改变模型、训练监督、target 生成、单帧输入体素数或 checkpoint。地图持久单元由旧 `X×Y` 扩展为权威 `X×Y×Z`，并保留旧 BEV；输出快照与指标协议已变化。
- 动态障碍仍没有可信 evidence 来源。本轮没有从 Doppler 猜测动态阈值，因为 P0-03/P1-04 后通道可能处于不同物理/归一化协议；应单独定义显式动态 mask 或跟踪器输出、来源 metadata 和更快衰减，再接入独立动态层。
- 动态层续修全仓搜索没有发现现成动态障碍 mask、跟踪器或可消费动态层；测试中的 `dynamic_*channels` 指模型通道配置，不是运动目标 evidence。
- 预处理 channel 2 标注为 egomotion-compensated mean Doppler，但正式 Dataset/inference 会按 normalization artifact 的 `scale_mps` 归一化；streaming 输入还可能是生成体素，不能仅凭数组通道位置判断当前值是 m/s、归一化值还是生成特征。
- 因此动态层入口必须消费显式二值/概率 sidecar 及 provenance，而不是在 `SlidingProbabilisticGridMap` 内新增隐式 Doppler 阈值。未提供 evidence 时动态状态应保持未启用，避免固定增加约 8 MiB 三态层内存。
- inference 逐文件模式保存生成体素并在完成后发布 `inference_run.json`；预处理场景则用 `preprocess_policy.json` 记录 `velocity_mode`、速度文件 hash 和 channel 语义。这些 run/policy 文件可作为外部动态分类器的输入 provenance，但现有地图入口未校验它们。
- 动态 evidence 帧文件设计为 `<frame>_dynamic_evidence.npz`，同时包含 `(X,Y,Z)` 的 `probability` 与显式布尔 `observed`。这样 `probability=0` 只在 `observed=1` 时表示静态证据，未观察位置继续是 unknown，不复发 P0-05 的空白即 free 错误。
- 目录级 `dynamic_evidence.json` 应固定 `body_voxel` 坐标、概率/observed 语义、网格/pc_range、帧数、来源类型和 64 位来源 artifact hash；streaming 用同一 `T_local_body` warp evidence，并在最终 map metadata 记录实际消费文件聚合 hash。

## 2026-08-20 Codex VS Code 历史会话读取修复

- VS Code 扩展为 `openai.chatgpt@26.818.21641`，捆绑 Codex CLI `0.148.0-alpha.21`；终端默认 CLI 为 `0.142.4`，本次验证固定使用扩展版本。
- 会话数据没有丢失：`state_5.sqlite` quick check 为 `ok`，共 219 条线程（216 活跃、3 归档），全部 rollout 路径存在；`session_index.jsonl` 169 行均为合法 JSON。
- 根因是用户配置把提供方设为自定义 `OpenAI`（大写），旧会话记录为内置 `openai`（小写）。app-server 的 `thread/list.modelProviders` 是精确过滤；实际复现中 `OpenAI` 只返回 2 条新会话，`openai` 能返回旧历史。
- 旧会话内容可读：对 2026-04-29 会话执行 `thread/read(includeTurns=true)` 成功返回完整 turns，排除 JSONL/SQLite 损坏。
- `disable_response_storage`、`network_access`、`windows_wsl_setup_acknowledged` 均为当前扩展 CLI 的未知旧字段；其中有效历史策略仍是 `history.persistence=save-all`。
- 修复将 `model_provider` 统一为内置 `openai`，删除重复的 `[model_providers.OpenAI]` 和三个未知旧字段。原配置备份为 `/home/zxj/.codex/config.toml.bak-20260820-codex-history-fix`。
- 候选配置已通过 `app-server --strict-config`、按 `openai` 枚举旧会话和旧会话全文读取；未改动 SQLite、session index、sessions 或 archived sessions。
- 本修改只涉及 Codex 用户配置，不改变项目监督信号、体素数量、模型结构、checkpoint 或指标结果。

## 2026-08-20 P1-05 动态层代码审查

- 首轮 GREEN 的兼容快照分别用 `max/max/min` 合并静态与动态 belief、plausibility、unknown，结果可能不再满足 `unknown = plausibility - belief`，也无法由同一组 D-S 质量解释。兼容的静态并动态状态必须先合并 belief/plausibility，再统一推导 unknown 与 pignistic probability。
- 动态 sidecar 是外部分类器/跟踪器的显式输出，不能隐式复用 Radar 距离衰减、Doppler 方差或生成模型 uncertainty。动态 probability 表达分类结果，`observed` 表达有效域；融合可靠度只额外折扣 body→local 位姿的 odometry 置信度。
- `GridMapConfig` 的直接 Python API 需要保证两项衰减率均为有限非负数；“动态严格快于静态”的关系只在实际提供动态 evidence 时、且必须在任何状态修改前验证。速度缩放同时作用于两者，因此不会改变已验证的次序。
- 严格 JSON 协议当前会把数字字符串和布尔值转换成 `pc_range` 浮点数，形成“JSON schema 看似严格、实际自动降级”的接口不匹配；需显式要求六个 JSON number（拒绝 bool/string），并严格要求 `shape_xyz` 为三个正整数。
- `map_run.json` 只记录速度缩放后的 `dynamic_decay_rate`，无法区分用户输入与 35/50/70m/s 速度修正；审计输出应同时记录静态/动态 base 与 effective 衰减率。
- 聚焦回归后的兼容性复审发现：若在 `GridMapConfig` 构造期无条件要求动态衰减快于静态衰减，会使从未启用动态 sidecar 的旧静态调用也受新参数约束。正确边界是两项衰减率始终要求有限非负，但“动态严格更快”仅在实际提供动态 evidence 时、且在任何地图状态修改前校验；CLI 同样只在指定 `--dynamic_evidence_dir` 时启用该关系门禁。
- streaming 协议 loader 首轮只预检 metadata、文件名和普通文件属性，NPZ 的 key/shape/数值却到逐帧循环才验证；若后续帧损坏，输出目录和早期快照已经产生。正式协议应在创建输出前逐文件解析校验并记录 SHA-256，运行时重新读取后对照预检 hash，避免损坏输入留下可误用半成品并检测预检后的文件替换。
- 动态快照首轮先复制既有静态状态，再为 `static_*` 键重复复制一遍同一数组。`static_*` 可以安全引用快照内已经与内部地图隔离的第一份副本；后续替换 legacy 组合键不会改变该引用，可减少常用 `128×128×32` 快照约 8 MiB 的重复三维副本。
- D-S envelope 复审发现更深一层语义问题：动态层 `unknown=1` 表示外部分类器未覆盖，不是独立传感器报告“可能有任意动态障碍”。若与静态 plausibility 直接取最大，启用一个稀疏 sidecar 就会把其他位置已经确认的 static free 全部抬回 unknown。兼容总 occupancy 应采用动态占用覆盖：仅当动态 pignistic probability `>0.5` 且高于静态 probability 时，选择该动态单元完整的 probability/belief/plausibility/unknown 状态，否则完整保留静态状态；这样既保持 D-S 一致，也不让未覆盖域污染总图。
- 文件名协议允许同时出现 `<frame>.npy` 与 `<frame>_voxel.npy`，二者会映射到同一 frame 键；identity 模式此前不会像 pose loader 一样拒绝，动态 sidecar 还可能被重复消费。主入口现在对所有模式线性预检 frame 键唯一性。
- `source_artifact_sha256` 是外部 producer 在 metadata 中声明的模型/规则 artifact hash，当前目录协议没有提供可本地解析的 source artifact 路径，地图端不能把它表述为已验证。run metadata 需显式记录 `declared_by_metadata_unresolved`，同时继续用实际读取的 metadata/逐帧 NPZ hash 审计本次输入。
- 最终动态协议不改变模型监督、LiDAR target、单帧输入体素数量或 checkpoint。未启用 sidecar 时不分配动态三维状态；启用后常用 `128×128×32` 网格新增四个持久 float32 动态状态约 8 MiB，快照还需序列化分离与组合状态，但已移除约 8 MiB 的重复静态副本。
- 提供动态 evidence 后，旧 `occ_prob*` 键变为 static 与明确 dynamic-occupied 的一致覆盖，障碍查询、precision/recall、unknown 统计可能变化，不能与无动态协议旧日志直接混合。仓库仍不包含可信动态分类器/跟踪器；本项实现的是严格消费、位姿对齐、独立衰减和审计层，不从未校准 Doppler 生成伪标签。
- 最终验证为 36/36 聚焦测试通过，相关三份 Python 编译、streaming 直接 `--help`、`git diff --check` 与 `git diff --cached --quiet` 均为 exit 0；没有运行真实数据重放、预处理、训练、推理或全量评价，也没有暂存、提交或推送。

## 2026-08-20 P1-07 LDM 验证与 CD 训练语义审计

- `VAETrainer` 已有独立 `validate()`、`best_iou` 与同 epoch checkpoint 一致性；`OptimizedLDMTrainer` 仍只有 `best_loss`，训练循环直接用 epoch train loss 保存 `ldm_best.pt`，没有 LDM validation loader 或验证选择协议。
- LDM checkpoint 已经由 P1-04/P1-06 补齐 `model_config`、fusion voxel/pc range、VAE SHA-256 和 Radar normalization，P1-07 不重复改造这些字段，只扩展验证选择与恢复状态。
- standalone CD 会加载冻结 LDM checkpoint 并用其初始化 CD/EMA 参数，但逐步 consistency target 实际来自 `cd_model_ema`，并非每步调用冻结 LDM；因此本轮按已批准边界准确声明“LDM 初始化后的 EMA consistency training”，不把它伪装成持续 LDM teacher distillation，也不在同一项重写算法。
- 最小实施边界：正式 LDM 必须接收独立 validation loader，按验证期的任务相关指标选择 best，并在 checkpoint 中记录 selector、验证指标和 split 语义；legacy/旧测试需显式兼容，不能把缺字段 checkpoint 自动补签为新正式协议。
- `main()` 已构造无增强、无 shuffle 的连续时间后缀 `val_loader`，VAE 会消费它，但 LDM 当前只调用 `trainer.train(train_loader)`；因此无需重新设计数据切分，只需把已有独立 loader 接入 LDM，并锁定接口不允许隐式回退到训练集。
- 现有 `test/evaluation/ldm/select_ldm_checkpoint.py` 使用固定 32 帧、20 步完整生成、IR real-only 和结构门槛做最终候选选择，适合正式训练后的离线 gate，不适合每 epoch 内嵌。训练期应使用固定 seed/sigma/noise 的单步 denoising validation proxy，至少记录 val loss 与解码 occupancy IoU；最终生成质量仍由既有固定协议选择器裁决，两者名称必须区分。
- LDM 训练期 best 建议采用 `max val_denoising_occupancy_iou`、同 IoU 时 `min val_loss`，并记录固定 validation seed/sigma/occupancy threshold。这样 best 不再依赖 train loss，同时不会把单步代理指标冒充 20 步生成指标。
- `OptimizedLDMTrainer.train()` 现强制接收独立 `val_loader`，并拒绝空 loader、同一个 DataLoader 或同一个 Dataset；统一入口已把既有连续时间后缀验证集接入 LDM，不再存在训练集隐式回退。
- 每轮验证固定使用 seed `42`、sigma `0.5` 和 occupancy threshold `0.5`。同一固定噪声序列计算单步 denoising latent MSE，并将输出经冻结 VAE 解码后累计 micro occupancy IoU；这些名称均带 `denoising`，避免与既有 20 步完整生成门禁混淆。
- `ldm_best.pt` 现按最大验证 denoising occupancy IoU 选择，同 IoU 才比较更小的验证 latent loss；训练 loss 只作为 `best_train_loss`/兼容 `best_loss` 审计字段保留。CSV、epoch checkpoint 和 best checkpoint 都记录当前/历史最优验证状态及 selector。
- 新协议 checkpoint 恢复时严格比较 protocol、split、selector、seed、sigma 和 threshold，并验证 best 不劣于 current；所有比较在加载 model/optimizer state 前完成。无 `ldm_validation` 的旧 checkpoint 仍可恢复，但不会被自动补签，必须经过下一轮独立验证后才能保存新正式 checkpoint。
- CD 算法没有被改写：LDM checkpoint 仍只初始化 `cd_model`/`cd_model_ema`，逐步目标仍来自持续更新的 `cd_model_ema`。checkpoint 新增 `training_semantics=ldm_initialized_ema_consistency_v1`、`ldm_role=initialization_checkpoint` 和 `consistency_target_source=cd_model_ema`，旧 `teacher_model_path` 仅作为兼容配置名保留。
- 本项不改变 target、每帧 occupied target 体素数、训练样本成员、网格尺寸、LDM/CD 网络结构或训练损失公式。每 epoch 新增一次完整验证集的 LDM 前向和 VAE decode；`ldm_best.pt` 与历史按 train loss 选择的结果不可直接等同，最终生成质量仍须通过固定 32 帧离线 gate。
- 聚焦回归通过：LDM validation 协议 5 项、LDM 结构/训练器 81 项、VAE/checkpoint 23 项、多模态推理 31 项、离线 LDM selector 10 项、机载多模态 9 项，以及两个 CD 脚本式接口测试；相关 Python 编译和 `git diff --check` 通过，未运行长时间训练、全量预处理或真实数据推理。

## 2026-08-20 Radar normalization 零 IQR 初步诊断

- 候选数据预处理和两个场景 manifest 已完成，失败只发生在步骤 6 的 garden normalization artifact 统计。
- `build_radar_normalization.py` 对 occupied Radar 体素的 `log1p(intensity)` 使用 Q25/Q75 IQR；当前报错证明合并后的中间 50% intensity 完全相同，但尚不能据此判断是源数据常量、体素聚合量化或通道读取错误。
- 现有失败保护是正确的 fail-closed 行为，不能直接把 IQR 硬编码为任意常数；需先比较 manifest、原始 `.npy/.npz` 四通道和 crop/resize 后分布。
- 已生成的候选 Raw/体素目录不得删除或覆盖；修复后应只重跑步骤 6，不重复执行全量解包和预处理。
- 候选 garden manifest、`preprocess_policy.json` 和 `target_policy.json` 均存在，记录 4013 帧、四通道 `[occupancy, mean_intensity, mean_doppler, doppler_variance]`、`velocity_mode=none` 和 45ms Radar-LiDAR 容差；normalization artifact 不存在，说明原子发布前失败，没有半成品。
- 体素化实现确实从点云第 4 列计算 occupied voxel 的 mean intensity；Dataset/normalization loader 也按通道 1 读取，静态调用链暂未发现通道索引错位。
- 新 `unpack_rosbag.py` 将 PointCloud2 固定导出为 `[x,y,z,intensity,doppler]`，但 intensity 支持多个 alias；下一步需要核对实际 schema 的 selected/missing fields，并直接统计候选 Raw 与体素 intensity 的唯一值和分位数。
- 全量 garden Raw 共有 4816 帧、3432896 个点，intensity Q25/median/Q75 为 `9.70/11.78/14.02`；候选 4013 个体素帧在 resize 前共有 2399425 个 occupied voxel，intensity Q25/median/Q75 为 `9.69/11.68/13.83`。源数据与体素化均非退化，排除 intensity channel 缺失或常数化。
- 真实均匀抽取 16 帧验证了重采样根因：现算法先用 adaptive max 标记 coarse occupancy，却用 trilinear 中心采样计算 intensity，导致 14829 个 coarse occupied voxel 中 `76.49%` 的 intensity 被错误置零，log Q25/Q50/Q75 全为零。
- 使用与 occupancy 相同 adaptive 分箱的 occupied-weighted average 后，相同 14829 个 coarse occupied voxel 的 intensity 零比例为 `0%`，Q25/median/Q75 恢复为 `9.28/11.27/13.48`，log IQR 约 `0.343`。修复应统一分箱，而不是放宽 IQR fail-closed 条件。
- 同一错位也会把 coarse occupied voxel 的 Doppler 均值/总方差错误清零，因此这是训练输入监督条件的重采样 bug，不只是 artifact builder 的统计问题。
- 调用端复核确认训练 Dataset、正式 inference Radar loader 和 normalization builder 都复用 `resize_radar_voxel_channels()`；本次单点修复能保持三条链一致。target/observed mask 和离线 prediction 评价继续使用各自的通用重采样，不受影响。
- 修复后用真实 garden 前 32 帧执行不落盘 builder 烟测：`log_median=2.486572`、`log_iqr=0.366153`，IQR 已恢复为正有限数；烟测显式为 `formal=false` 且 writer 被 mock，没有生成可误用 artifact。
- 最终聚焦回归共 73 项通过：normalization 12、builder 4、Dataset metadata 13、sensor-aware target 4、多模态 inference 31、机载多模态 9；相关 Python 编译和 `git diff --check` 通过。
- 本修复不改变磁盘上的 Radar/target/IR/LiDAR 文件、manifest、target 内容、coarse occupancy 体素数量或网格尺寸；会恢复此前错误清零的 Radar intensity、Doppler 和 variance 条件值。旧错误重采样下的模型输入与指标不可同新正式协议直接比较，当前尚未开始正式训练，因此应先生成 artifact 再训练。
- 本阶段结束时步骤 6 的目标 artifact 尚不存在，只能重跑 normalization CLI，不能重复 `preprocess-v2.sh`；该状态已由下一节记录的用户重跑结果解除。

## 2026-08-20 正式 Radar normalization 与训练入口切换

- 用户已成功生成正式 artifact：`radar_normalization_garden_32x128x128_full120_86p8_v1.json`，SHA-256 为 `2c9c92650b98ec686d621b53eccb5e7f376cb6b8ea1047d4fb594349af90c4d5`；其训练来源为 garden 全 4013 帧，网格 `[32,128,128]`，source/model pc range 均为 `[0,-20,-6,120,20,10]`，Doppler scale 为 `86.8 m/s`。
- 用户的正式输入验收通过：garden manifest 为 4013 帧、loop3 为 6432 帧；首个 target/Radar 均为 `(4,32,128,128)`，Radar occupied 为 1028，真实 IR 与真实 calibration 均可用。
- `default_config.yaml` 已从故意未配置状态切换为该正式协议，并显式冻结 target size、source/model range、artifact 和 scale。VAE/LDM/CD 保存目录统一隔离到 `Result/train_results/formal_p1_04_full120_86p8_v1/`，避免覆盖或误续训旧结果。
- `train_unified.sh` 在重建临时训练链接前依次校验训练场景 manifest、artifact schema/grid/scale、固定 SHA-256、training scene 和 frame count；生成的 override 使用绝对数据/artifact/结果路径，消除启动工作目录造成的相对路径漂移。
- 已移除 launcher 的隐式 checkpoint 探测恢复。任一阶段结果目录非空时默认 fail-closed；只有用户显式设置 `ALLOW_RESUME=1` 且对应 best checkpoint 存在时才传 `--resume`，随后 Python 层继续校验 LDM/CD normalization spec/hash 和 checkpoint 协议。
- 三个正式生成 launcher 使用同一 candidate loop3 输入及新 checkpoint 根，输出目录携带 `formal_p1_04_full120_86p8_v1`；独立评价入口同步使用 candidate preprocessed 根、header-time Raw 根和相同部署目录映射，避免新预测与旧 Raw/index 跨协议配对。
- 代码审查发现 mini 训练复制正式默认 YAML 后仍传 `--allow_legacy_radar_units`，会触发正式/legacy 互斥。现已仅在 mini 派生配置中显式清空 artifact/scale；正式 launcher 继续禁止 legacy 降级。
- 本轮不改变磁盘数据、target 定义、单帧通道或体素网格。garden 4013 帧按连续时间块产生 3210 个训练样本和 803 个验证样本，loop3 6432 帧保持独立测试；每样本仍为四通道 `32×128×128`。旧 receipt-time/旧 normalization checkpoint 与后续新指标不可直接混合，输出协议名已物理隔离。
- 最终验证通过 202 项具名 unittest 与两份 CD 直接接口测试；7 份相关 shell 通过 `bash -n`，默认 config/artifact/result 对照、candidate 路径存在性和 `git diff --check` 均通过。没有启动训练、模型采样、推理或全量评价，正式结果根仍不存在且可安全开始新训练。

## 2026-08-20 正式训练入口导入路径续修

- 用户首次执行正式 VAE launcher 时在参数解析前失败：`unified_train.py` 和 `cd_train_optimized.py` 只把 `diffusion_consistency_radar/` 加入 `sys.path`，可解析旧式 `cm.*`，却无法解析新增的 `diffusion_consistency_radar.checkpoint_chain` 包路径。
- 原 fallback 将 `checkpoint_chain.py` 当顶层模块导入，但该模块内部仍依赖 `diffusion_consistency_radar.radar_normalization`，因此直接脚本接口与包内接口不匹配；同时捕获整个 `ModuleNotFoundError` 可能掩盖依赖内部真正缺包。
- 两个训练入口现在同时显式引导仓库根与包目录，并统一使用 `diffusion_consistency_radar.*` 唯一包名；删除失效 fallback，避免同一 `cd_train_optimized.py` 以 `scripts.*` 和 `diffusion_consistency_radar.scripts.*` 双重身份加载。
- 新测试从临时工作目录清除 `PYTHONPATH`，分别执行两个入口的 `--help`；RED 复现与用户相同异常，GREEN 后均成功。checkpoint 链 8 项、VAE checkpoint 23 项、mini launcher 6 项及两份 CD 接口测试通过，Python 编译、Bash 语法和 `git diff --check` 通过。
- 失败发生在 Trainer、DataLoader 和输出目录创建前，正式结果根仍不存在，没有 checkpoint 或训练日志需要恢复。本修复不改变监督信号、target、样本数、体素数量、模型结构、normalization 或指标定义。

## 2026-08-20 正式 VAE batch metadata 续修

- 第二次启动已成功进入 VAE epoch 1，但首个 batch 在 worker 的 `default_collate` 失败；异常发生在 batch 交付前，没有执行前向、反向或优化器更新。
- 真实 garden 样本的顶层 metadata 全部可拼接；唯一根因是 `preprocess_policy` 内 `velocity_mode=none` 对应的 `v_drone`、`velocity_file`、`velocity_file_sha256`、`velocity_record_count` 为合法 JSON null。将其伪造为 0/空字符串会破坏 provenance 语义。
- 新共享 `collate_voxel_samples()` 继续用 PyTorch `default_collate` 严格拼接 target、Radar、observed mask 和多模态字段，仅将审计用 `preprocess_policy` 保留为逐样本字典列表；其他非法 metadata 仍 fail-closed。
- 调用链审查发现统一训练的 train/val、standalone CD、条件推理共四个 Dataset DataLoader，现已全部显式使用同一 collator，避免只修 VAE 后在 LDM/CD/推理复发接口不匹配。
- 真实 garden 前两个样本、多 worker 烟测通过：target/Radar 为 `(2,4,32,128,128)`，observed mask 为 `(2,1,32,128,128)`，两个 policy 的 `v_drone` 均原样为 `None`。
- 零 epoch 失败目录只有 header-only `metrics.csv` 和启动日志，无 checkpoint，已无损归档到 `Result/train_results/formal_p1_04_full120_86p8_v1/failed_starts/vae_20260820_212426_collate_failure/`；active `vae/` 已释放，可 fresh 重跑且不需要 `ALLOW_RESUME=1`。
- 本修复不改变 target/监督、4013 个基础样本、3210/803 时间块划分、每样本体素数量、网络、loss、normalization 或指标定义。Dataset metadata 进入 batch 后，`preprocess_policy` 的接口由递归字典张量改为逐样本原始字典列表。

## 2026-08-21 8 GB 单卡 formal mini 训练审计

- 历史 `train_minimal.sh` 默认为 legacy Radar 单位、旧数据根和已有 `test/mini-test/train_results_mini/`，不能代表已验收的 `formal_p1_04_full120_86p8_v1`；现保留该默认兼容，同时增加显式 formal 分支。
- formal mini 固定 candidate garden、artifact SHA-256 `2c9c92650b98ec686d621b53eccb5e7f376cb6b8ea1047d4fb594349af90c4d5`、`86.8 m/s` 和 source/model full120。每帧仍为四通道 `32×128×128`，即 524288 个空间体素、2097152 个通道值；只把默认抽样降为 16 帧和每阶段 1 epoch，不改变单帧监督定义或体素数量。
- mini VAE/LDM/CD checkpoint 统一写入 `formal_mini_chain_v1`；正式 `formal_chain_v1` 校验器已用行为测试确认拒绝整条 mini 链，避免短训练权重被误用于正式部署或指标报告。
- 新入口 `run_formal_mini_8gb.sh` 只允许 `vae|ldm|cd` 单阶段，固定 batch 1、worker 0、梯度累积 1、ultra-lightweight VAE、checkpointing 开启、AMP/FP16 关闭；最多 32 帧、20 分钟，启动温度不高于 65°C、运行达到 80°C 中止，总/空闲显存门槛为 7500/6000 MiB。
- 温度读取失败、过热或超时会对独立训练进程组按 `INT → TERM → KILL` 逐级中止。每阶段 scratch/config 独立且要求 fresh，非空阶段输出拒绝覆盖；历史数据、checkpoint、日志和结果均未删除或覆盖。
- 代码审查发现 `conda run -n Radar-Diffusion python -` 在当前环境会返回 0 却不转发 heredoc stdin，导致 artifact Python 校验被静默跳过。现改为 `conda run --no-capture-output ...`，并增加错误 SHA 必须在任何写入前失败的回归测试。
- 真实只读预检确认设备为 RTX 4070 Laptop GPU、8188 MiB；最终观测空闲 7186 MiB、37°C，artifact hash/full120/scene 校验通过。预检路径 `/tmp/formal_mini_8gb_preflight_codex` 未创建，没有启动 CUDA 训练。
- mini 脚本协议 11 项、配置/路径安全 103 项、checkpoint 链 10 项、VAE payload 23 项和 CD 入口测试均通过；相关 shell `bash -n`、Python 编译和差异检查纳入最终验收。formal mini 指标只用于调用链烟测，不与 garden 全量训练或 loop3 正式测试结果比较。

## 2026-08-21 外部审查第一批修复与 80--120 m 监督审计

- 外部意见中的核心问题成立：旧正式链未完整绑定监督/split/标定身份，训练场景软链接会隐藏 calibration，`align_to=lidar` 却可能使用 Radar→Thermal，固定 0.01 m 时间补偿没有逐帧依据，且当前 80--120 m target 全部被裁成 0。
- 正式 checkpoint 已提升为 `formal_chain_v2`，结构化 `formal_data_v2` 绑定 manifest、split、target policy、observed mask、标定和 Radar--IR sync。VAE/LDM/CD 的 parent hash、stage、网格、normalization 与 data identity 在模型/优化器恢复前 fail-closed；v1 只允许显式诊断。
- Dataset 和 launcher 改为显式 `scene_names`、`calibration_dir`、real IR/calibration 和 `voxel_coordinate_frame=lidar`；训练不再创建 `.tmp_train_dataset`。training/deployment manifest v2 分离模态要求，但都绑定 preprocessing provenance。
- 推理端不再只做两次互不关联的校验：VAE 与 LDM/CD data protocol 必须相同，deployment scene/Radar 目录/manifest/当前标定必须交叉一致，部署身份写入 `inference_run.json`。
- Radar--LiDAR 时间补偿统一采用 `p_ref=p_sensor+v*(t_sensor-t_ref)` 的 signed delta，只移动非参考传感器；Radar Doppler 补偿在 Radar 原点坐标内完成，Dataset/inference 删除固定 0.01 m 位移。`velocity_mode=none` 明确记录 `raw_mean_doppler`。
- 三外参闭环审计仍显示旋转元素最大差 `0.0508489`、平移 L2 `0.203829 m`，因此当前以直接 LiDAR→Thermal 为投影权威，同时保留闭环残差，不能把组合链无依据视为真值。
- garden 4013 帧 v2 审计显示：XYZ 正式体素盒内 80--120 m raw LiDAR 点 `372780`，远距 LiDAR occupied `365069`，但 target occupied 为 `0`；3934 帧有远距 occupied，target 保留率仍为 0。
- 远距证据显著弱于近距：raw 点数为近距的 `0.404%`、occupied voxel 为 `1.488%`；32 帧 ray coverage 均值约为远距 `0.194%`、近距 `3.845%`；128 帧 Radar→LiDAR 最近邻在 1 m/2 m 内匹配率均值仅 `5.23%/23.24%`。
- 同网格相邻帧远距 Jaccard 中位数 `0.00649`，但没有位姿补偿，不能表述为 world-frame 稳定性指标。IR 64 帧抽样确认原始为 `uint8 [512,640,3]` 且通道有差异，现处理后通道完全相同；是否保留彩色编码需独立消融，不能声称恢复 16-bit。
- 基于现有证据，推荐 formal v2 模型和评价先收敛到 `0--80 m`，`80--120 m` 在地图中保持 unknown。若保持 `32×128×128`，空间体素总数仍为 524288，但 x 分辨率由 full120 的 0.9375 m 改为 0.625 m；旧 full120 数据/checkpoint/指标均不可直接比较。
- 若选择 0--120 m，不能只把 `x_max` 改成 120 或仅给辅助 loss 加 mask；必须先持久化 observed/unknown，并让 target encoder/latent 监督、全部 occupancy loss 和指标统一感知 unknown，随后从头重训。
- 第一批没有重建数据、运行 mini/full training 或 GPU 推理，也未删除/覆盖旧 candidate、artifact、checkpoint、日志和结果。正式 v2 入口在范围、observed/split 和新 artifact 完成前保持 `range_pending` fail-closed。

## 2026-08-21 formal v2 0--80 m 数据协议实施

- 用户确认采用 0--80 m 模型/评价范围；预处理 source 网格从 `600x200x80` 变为 `400x200x80`，X 向源体素数减少 33.3%。模型仍输出四通道 `32x128x128`，每帧空间体素仍为 524288，但 X 分辨率由 0.9375 m 变为 0.625 m。
- 预处理逐帧持久化严格稀疏 observed mask；training manifest 现在精确要求 Radar/LiDAR/target/observed/IR 五模态。formal Dataset 缺 mask、协议/range/shape 不匹配时直接失败，只有 legacy 诊断路径可运行时重建。
- temporal split artifact 由 manifest 绑定的 Radar 时间构建连续 train/purge/validation，加载时按当前数据重建全文比对。target 相关性审计据此选择 3.0 s purge；该值是防近邻泄漏门禁，不是无 pose warp 的世界坐标独立性证明。
- normalization 正式发布必须绑定 split 文件 SHA-256 且扫描全部 train IDs；formal data protocol 从 manifest、split、target policy、observed record、Radar--IR sync 和标定自动派生，VAE/LDM/CD 使用同一个身份。
- 统一训练入口分别构造精确 train/validation Dataset，不再为 formal v2 运行时按下标切分；独立 CD 同步强制 scene、真实 IR、真实标定、LiDAR frame 与持久 observed，并仅允许显式 `--resume`。
- 模型 evidence range 与地图 range 已分离：80 m 模型张量只写入 0--80 m 区域，0--120 m 地图中的 80--120 m 保持 `probability=0.5`、`unknown=1`，不再因范围拉伸被解释成 free。
- 4 帧 garden fresh smoke 的五模态 manifest 和正式 Dataset 加载通过，首帧 persisted observed voxel 为 24922；没有运行全量数据重建、mini/full training、GPU 推理或正式评价。
- 指标影响：后续 formal v2 occupancy 指标只统计 0--80 m observed domain，并报告 coverage；它与旧 full120/无 purge/全场景 normalization 结果不可直接比较。旧数据、artifact、checkpoint、日志与结果全部保留为 legacy/diagnostic。
- 当前剩余隐形依赖是 deployment 数据生产：正式推理 launcher 已指向新的 deployment root，但本阶段只实现训练数据生产链；缺 deployment manifest 时会正确 fail-closed，后续需新增或确认严格 deployment-profile 视图生成步骤。

## 2026-08-21 严格 deployment-profile v3 生成链

- training manifest 保持 schema v2；新 deployment view 使用 schema v3，避免无版本地改变既有数据合同。v3 场景只允许 `radar_voxel`、`ir_image`、`radar_ir_sync.csv`、preprocess policy、源 training manifest 快照、scene receipt 和最终 manifest，额外 LiDAR/target/observed 或未知文件均被拒绝。
- 新生产器在创建输出根前验证 training manifest、当前预处理脚本、三组外参、thermal intrinsics 和 scene 内 Radar--IR sync 的 SHA-256；任何 provenance 漂移都不会留下正式输出。
- 默认 hardlink 复用 Radar/IR 普通文件以节省本地磁盘，明确禁止 symlink；另支持显式 copy。receipt 记录 `materialization_mode_at_creation`，运行时只依赖内容 hash，因此传到服务器后即使 hardlink 被展开为独立文件，视图仍可验证。
- 根级 `deployment_dataset.json` 固定精确场景集合和每个 scene manifest hash；场景级 `deployment_view.json` 绑定父 training manifest、policy、Radar/IR 记录、同步和物化方式。父 training manifest 快照随视图携带，部署无需访问 LiDAR/target 文件也能审计派生关系。
- 三个正式 inference launcher 在模型运行前一次性验证根收据、精确场景集合和全部 scene；Python 推理入口随后再次校验当前 scene，并把 deployment manifest、父 training manifest、receipt、当前标定与 checkpoint `formal_data_v2` 交叉绑定，身份写入 `inference_run.json`。
- `preprocess-v2.sh` 增加第 8 步：全量 loop3 training v2 完成后生成 `Data/NTU4DRadLM_Deploy_formal_v2_80m_86p8_v1`。generic manifest CLI 不允许手工创建 deployment v3，必须使用专用生产器，防止绕过父身份和 receipt。
- garden 4 帧最终 smoke 位于 `test/result/comparison/formal_v2_80m_deployment_smoke_v2`：dataset hash `e6044ee8...a96a5`、scene manifest hash `3bf82b72...b4c9e`、父 training manifest hash `a0a70ea5...01be34`，frame_count=4，二次只读验证通过。
- 本项不改变训练监督、target/observed、训练样本集合、模型张量 `32x128x128`、checkpoint 或指标定义；deployment 视图有意不包含监督文件，因此不能用于离线评价。旧数据、checkpoint、日志和正式结果均未删除或覆盖。

## 2026-08-26 Deployment observed/frame/risk 运行时安全链

- 正式推理现在从输入 Radar occupied endpoint 在 LiDAR 体素网格中生成 `radar_endpoint_ray_visibility_v1` observed mask，不从模型 sigmoid 或 IR 投影猜测 free-space。所有 endpoint 都保留为 observed，每个离散方向只向最近 endpoint 铸造射线，近端障碍之后的间隔继续保持 unknown。
- `inference_run.json` 逐帧绑定 mask 文件名、SHA-256、endpoint/observed 数量、Radar 原点和 Radar→LiDAR 标定 SHA，并与 deployment identity 交叉校验；正式输出 frame 明确为 LiDAR。
- 正式地图入口要求 inference run、全部逐帧 mask、body→local pose CSV 和显式 LiDAR→body 外参，融合使用 `T_local_voxel=T_local_body@T_body_voxel`。mask 篡改、帧集不一致、标定身份错配或缺参数均在输出目录创建前失败。
- 风险查询已改为 `clear/obstacle/unknown` 三态，保留兼容 `is_risky`键。空地图、高 unknown mass、搜索半径或地图范围小于 `v*t_reaction + v^2/(2*a_brake) + margin` 均返回 `unknown` 且 risky；高不确定性不再降低风险。默认参数下 35/50/70 m/s 的安全距离为 99.0625/186.25/346.25 m。
- uncertainty、IR BEV、dynamic evidence、prior DEM 和 target 尚未拥有与 inference receipt 一致的 frame/provenance 合同，因此 formal 地图本阶段拒绝它们；legacy 诊断能力保留。这是显式边界，不是已完成的多源在线融合。
- 推理和地图正式输出均拒绝符号链接/非空目录，metadata 原子发布。若中途中断，缺 run metadata 的部分输出无法进入 formal map。
- 本项不改训练监督、target、模型体素数、网络结构、checkpoint 或历史指标。`uint8` 稠密 mask 每帧约 0.5 MiB，loop3 6432 帧约新增 3.14 GiB；point-count/Doppler-validity sidecar 与 `UncertaintyHead` 修改继续留在下一子阶段。
- 当前 `Data/config` 不含明确的 LiDAR/Livox→body 外参；只有 Radar→IMU 等标定不能在未声明 `body==IMU` 时被自动推导为真值。因此真实 formal map 运行仍需用户提供经验收的 LiDAR→body 标定和 body→local 位姿记录。

## 2026-08-26 Mapping pose candidate 诊断设计

- loop3 存在 6445 条 `gt_odom.txt` pose 与 6432 个 Radar timestamp，但 GT 首时刻比 Radar 首帧晚 `0.398283 s`，前 4 帧不在可插值范围内。
- 最近邻配对会造成 653 帧复用同一 GT pose，因此候选链必须使用平移线性插值和四元数最短弧 SLERP，并对超出时间范围/过大 GT gap 的帧禁止外推。
- 现有 `calib_radar_to_imu.txt` 只是无方向注释的 4×4 矩阵；诊断只能在显式未验证假设下组合 `T_imu_lidar=T_imu_radar@inv(T_lidar_radar)`，并同时保留 GT-as-IMU / GT-as-LiDAR 两种 pose 假设。

## 2026-08-26 Mapping pose candidate 诊断结果

- 新增独立脚本 `test/diagnostics/alignment/build_mapping_pose_candidates.py`，组合候选 LiDAR→IMU-body 外参，并按 Radar timestamp 对平移线性插值、对四元数执行最短弧 SLERP；输出目录必须 fresh，禁止符号链接和覆盖。
- 候选矩阵平移为 `[-0.462580471, -0.136749595, 0.194256529] m`；该值仍依赖 `body=IMU` 与旧 4×4 文件方向为 `T_imu_radar` 两项未验证假设，不能写入正式标定目录。
- loop3 6432 个 Radar 时间戳中 6162 帧满足严格 0.2 s GT gap 门限。270 帧未覆盖：前 4 帧早于 GT，另 266 帧位于 `0.200822--0.261580 s` 的 GT 间断内。保守诊断未为提高覆盖率放宽门限或执行外推。
- 两套 pose CSV 均为 6162 行、timestamp 严格递增、四元数最大 norm 误差小于 `7.2e-13`，内容 SHA-256 与 `audit.json` 一致；所有行均携带 `diagnostic_formal=false`。
- 正式外参与 pose loader 已增加内容级 fail-closed 门禁：带 `formal=false` 注释的外参、带 `diagnostic_formal` 列的 pose CSV 均被拒绝，避免候选仅靠文件名隔离。
- 本项不改变训练监督、target、每帧体素数、模型结构、checkpoint 或指标；只新增小型文本/JSON 诊断结果，未运行训练、模型前向或 GPU 任务。

## 2026-08-26 Mapping frame 来源初查

- 仓库内未找到原始 bag、TF/static TF 转储、GT 导出器、R2LIVE 源码副本或 Radar→IMU 标定生成脚本；当前数值文件已丢失足以直接判定方向的 frame metadata。
- Radar→IMU 与 Radar→Livox 文件具有相同本地复制时间，但只有 Radar→Livox 文件声明方向，不能用 mtime 或相邻文件命名推断 Radar→IMU 语义。
- `gt_odom.txt` 只含 timestamp/translation/quaternion，无法从文本区分 IMU/body 与 LiDAR pose；必须继续查外部原始包/官方说明，或仅做非正式多窗口反证。
- `/home/zxj/下载/数据集/NTU4DRadLM` 和工作区均保留 loop3 三段原始 bag；下载副本与工作区的 Radar→IMU、GT 文件哈希分别完全一致，可直接转入 ROS topic/TF 只读审计。
- loop3 bag 无 Odometry/`/tf_static`；`/tf` 只有恒等 `map→base_link`。topic header 为 VectorNav `imu_frame`、Livox/内置 IMU `livox_frame`、Radar `base_link`，却没有三者安装 TF，因此无法从 bag 直接确认 Radar→IMU 数值方向。
- bag 内同时存在 VectorNav 与 Livox IMU，当前标定文件没有声明目标 IMU；必须把 IMU 身份与变换方向一起确认。
- 原始 IMU intrinsic 指向 VectorNav；NTU4DRadLM 论文明确 `extrinsic_xx_to_xx` 为从前一传感器到后一传感器并遵循 KITTI，因此文件方向现在有官方强证据为 Radar→VectorNav IMU，不应取逆。
- 仍存在独立的 body 轴约定问题：第三方 loader 会翻转 IMU Y/Z 后使用该外参；原始矩阵未经轴转换不能自动等同于项目的 airborne body frame。
- 论文称 `gt_odom.txt` 来自 `gt_odom.bag`，但本地下载没有后者；GT bag/exporter frame 仍是当前最关键缺口。
- 官方 4DRadarSLAM 明确会把 Radar 点云变换到 Livox frame 后处理，因此 bag 的 `frame_id=base_link` 不能当作点已在 body frame 的证明。
- LiDAR 多帧重合存在不可辨识边界：GT-as-LiDAR 时 LiDAR→body 外参在投回 LiDAR 的链中消去，无法借此确认 Radar→IMU 方向；经验指标只能作为 GT frame 反证，不能替代原始 frame metadata。

## 2026-08-26 Mapping frame 多窗口诊断结论

- 预处理保存链证明稀疏 `coords` 为 XYZ 索引，物理中心为 `pc_min + (coords + 0.5) * voxel_size`；不存在 ZXY/XYZ 猜测。
- v1 首轮暴露时间基准接口不匹配：candidate pose 按 Radar time 插值，而 `align_to=lidar` 的体素处于 LiDAR reference time。v1 结果保留为历史对照，不能作为最终诊断。
- v2 从 `Data/NTU4DRadLM_Raw_p1_01_candidate/loop3/radar_lidar_sync.csv` 逐帧验证 Radar time 后采用 LiDAR time，并把 SHA-256=`3ce134bd...ab79` 的 sync snapshot 封存在候选目录；overlap 在相信逐帧收据前先验证 manifest canonical `content_sha256`。
- 1.0 s、2--50 m、rotation≥3° 的 48 对中，GT-as-LiDAR 的 pair-median NN 中位数为 `0.4123 m`、GT-as-IMU 为 `2.3012 m`，paired 差值中位数 `1.8102 m` 且 48/48 同向。0.5 s 敏感性 32/32 同向；2.0 s 因共同视野减少为 30/32，但汇总中位仍同向。
- 代码已证明该数据/候选合同下 GT-as-LiDAR 更自洽；配置和官方命名支持 Radar→VectorNav IMU。尚未证明的是 `gt_odom` exporter 的权威 child frame，以及 VectorNav IMU 到 airborne body 的轴定义。因此 formal map 继续 fail-closed，不发布正式 LiDAR→body/body→local receipt。

## 2026-08-27 经验 LiDAR pose 离线地图合同

- 权威 `gt_odom` exporter frame 与 VectorNav IMU→airborne body 轴定义仍不可得，因此没有把经验结果包装成正式 LiDAR→body/body→local 链；机载、PX4 和避障声明继续 fail-closed。
- 新合同只采用 overlap 一致支持的 GT-as-LiDAR 分支，直接发布 `T_local_lidar`。它绑定 candidate/overlap audit、candidate pose、诊断外参、Radar--LiDAR sync snapshot 及所有成员 SHA-256，并在运行时重新组合和逐项校验。
- loop3 共声明 6432 个可用推理帧，经验 pose 覆盖 6165 帧，另 267 帧保持 uncovered；loader 按 receipt 顺序选择交集，不允许 `frame_limit`、首尾 pose 复制、外推或人工删文件。
- 地图核心新增显式 `T_local_voxel` 直通接口，并与 `T_local_body/T_body_voxel` 互斥。经验链的查询原点和 CSV 字段明确为 LiDAR，不再把 LiDAR pose 伪装为 body pose。
- 代码审查发现旧 `inference_run.json` 只绑定 observed mask，没有绑定真正消费的 prediction voxel 内容。现统一 `generated_voxel_artifact_v1` 协议，逐帧记录文件名、SHA-256、CZXY shape 与 float32 dtype；strict map 在创建输出目录前重算并校验。
- 兼容性边界：缺少 `prediction_voxel` 收据的旧推理目录仍保留为 legacy，但不能进入 formal 或 offline empirical strict map，必须重新推理生成可信 metadata，禁止手工补 JSON。
- 对监督和指标的影响：本项不改变 target、训练样本、模型张量、每帧 524288 个空间体素、loss、checkpoint 或推理数值。离线地图只处理 6165 个 pose-covered 帧，覆盖数量与 6432 帧全量运行不同；新地图协议和 LiDAR 原点风险查询不能与旧 identity/body/无收据地图指标直接比较。

## 2026-08-27 Radar point-count / Doppler-validity 正式数据合同

- Radar 稀疏 NPZ 现在在原 `coords/features/shape` 外保存与 occupied coords 一一对齐的 `uint32 point_count`、`uint32 doppler_valid_count` 和 `radar_point_count_doppler_validity_v1`。加载器拒绝重复/越界坐标、非有限 feature、零 point count、valid count 超界和不完整字段。
- 统计和原四通道 Radar 共存于同一 NPZ，不增加独立模态目录，也不改变 occupancy/intensity/height/Doppler 的聚合值。真实 4 帧 smoke 中总点数分别为 770、770、762、761，Doppler 有效数相同；多点体素分别为 129、130、142、138。
- formal Dataset 在构造时验证所有选中 Radar 文件；metadata 明确摘要引用 `pre_augmentation_persisted_radar_voxel` 且 `model_consumed=false`。模型输入仍为 `[4,32,128,128]`，未修改 `UncertaintyHead`、loss 或 checkpoint schema。
- 正式训练 launcher 不再固定 VAE/LDM 双卡 `0,1`；`CUDA_DEVICES` 默认 `0` 并接受逗号分隔 GPU 列表。`PREFLIGHT_ONLY=1` 会验证 training manifest、normalization SHA、全部 Radar statistics 和可重建 formal data protocol，然后在写训练配置、访问 GPU 前退出。
- 当前工作区仍缺全量 `NTU4DRadLM_Pre_formal_v2_80m_86p8_v1` 与对应 v2 normalization artifact，因此没有运行正式预检或训练。旧 full120 mini 与 `formal_mini_chain_v1` 只保留为 legacy，不能作为当前 formal v2 结果。
- 监督/体素/指标影响：target、observed mask、样本切分、每帧 524288 个模型空间体素和现有指标公式均不变；NPZ 仅增加 occupied voxel 对齐计数。未来若把统计接入 uncertainty，必须另行升级模型、loss、checkpoint 和评价协议。

## 2026-08-27 Formal v2 8 GB mini 训练链

- 旧保护入口实际绑定 p1_04/full120、旧 normalization 和 `formal_mini_chain_v1`，不能验证当前 0--80 m `formal_data_v2`；现改为只读复用完整 v2 training root，不再构造缺 manifest/provenance 的临时软链接数据视图。
- mini 从正式 temporal split 的有序 ID 中确定性取每场景前 8 个 train 和 4 个 validation 帧，并把策略/数量写入 `data_protocol.mini_selection`。VAE/LDM/CD 的父链与 resume 会拒绝不同子集身份，全量 `formal_chain_v2` 则拒绝任何隐藏 mini limit。
- 统一训练入口和独立 CD 入口均把 `formal_mini_chain_v2` 纳入真实 IR、真实标定、LiDAR frame、persisted observed、Radar statistics、split 和 normalization 门禁；修复了独立 CD 原先只识别全量协议的接口分叉。
- mini inference 只有显式 `--allow_formal_mini_checkpoint` 才接受 mini-v2，且必须同时使用 strict real-IR deployment view。输出身份为 `formal_mini_smoke`、`formal_protocol=false`；正式 checkpoint validator/launcher 仍拒绝 mini 权重。
- 代码审查同时发现单模型 LDM/CD 正式 inference launcher 引用了未定义的 `SCENE_DIR`，已改为绑定 `${PREPROCESSED_ROOT}/${SCENE}`。
- 监督和资源边界：target、loss、四通道输入、模型结构、单帧 524288 个空间体素及 X 分辨率 0.625 m 均不变；样本数减少只缩短迭代次数，不降低单样本显存，也不提供收敛或指标证据。
- 真实只读 preflight 已在 RTX 4070 Laptop（8188 MiB）通过：空闲 6923 MiB、41°C、artifact SHA-256 `11f59d84...e97c`、train/validation=8/4；没有创建训练输出或启动 CUDA 训练。

## 2026-08-27 Formal v2 VAE smoke 验收与 short profile

- 用户实际完成 8/4 帧、batch 1、1 epoch VAE smoke：8 个训练 batch、约 5.08 s，loss `5.051657`、validation IoU `0.017946`、recall `0.750919`、precision `0.018053`；日志未发现 OOM、NaN、过热或异常退出。该结果只证明训练/checkpoint 链闭合，低 IoU/precision 不能用于模型质量结论。
- `vae_best.pt` 已只读核对为 `formal_mini_chain_v2`、`formal_data_v2`、ordered-prefix train/validation `8/4`、0--80 m `32×128×128` 网格和 latent dim 4；原 smoke 目录保持不变。
- 新增第二位置参数 `smoke|short_train`。默认 smoke 行为不变；`short_train` 仅允许 fresh VAE，固定 3 epoch、8/4 帧和 batch 1，默认写入独立 `formal_mini_v2_80m_8gb_short_v1`，并把启动/运行温度上限收紧到 60/75°C。
- short profile 不改变 target、observed mask、loss、模型结构、每帧 524288 个空间体素或单样本显存；训练 batch 从 8 增为 24，只适合观察极短 loss 趋势和小样本过拟合，不提供正式泛化指标。
- 用户完成 short VAE 3/3 epoch：loss `5.051601→4.792839→4.690389`，validation IoU `0.017942→0.022273→0.024969`，precision `0.018050→0.022474→0.025194`；约 14.05 s 的 24 个训练 batch 中未发现 OOM、NaN 或异常退出。趋势为正，但绝对 IoU 仍很低。
- `vae_best.pt` 为 epoch 3，模型 state 与 `vae_epoch0003.pt` 逐张量一致；内部为 `formal_mini_chain_v2` / `formal_data_v2`、ordered-prefix 8/4、0--80 m `32×128×128`、latent dim 4，SHA-256=`a55c0bb0...03510`。原 1 epoch smoke checkpoint 哈希仍为 `1ae08bf8...c61f50`。
- 训练后 GPU 已回落到 41°C、855 MiB used、约 13.73 W。该 VAE 仅足以作为后续 1 epoch LDM 工程 smoke 的父 checkpoint，不能据此宣称 VAE 质量合格或开始正式 CD/评价。
- 调用链复核发现旧无训练 preflight 只检查父 checkpoint 文件存在，真正的 `assert_checkpoint_training_identity` 要到 `unified_train.py` 启动后才执行。现将同一身份断言前移到 formal preflight，并额外拒绝符号链接/空 state；CD 预检还核对 LDM 中记录的 `vae_checkpoint_sha256`，避免到训练启动后才暴露父链混用。
- 用户提出所有阶段使用 500 帧、20 epoch。formal split 有 3210 train/774 validation；“500 总帧（400/100）”与“train/validation 各 500”会形成不同 checkpoint 数据身份，需先确认。short VAE 实测外推仅 VAE 就约 78--98 分钟，不能直接沿用 20 分钟笔记本门禁。

## 2026-08-28 RTX 4070 Laptop 500 帧中型筛查与服务器 20 epoch 合同

- 用户确认 laptop 中型链固定选择 400 train + 100 validation，共 500 个唯一 formal split 帧；VAE/LDM/CD 各 20 epoch。新增 `medium_train`；初始结果根为 `v1`，allocator 失败修复后默认使用 fresh `test/result/formal_medium_v2_80m_laptop_500f_20ep_v2`，不会覆盖失败现场或既有 smoke/short。
- profile 固定 batch 1、worker 0、梯度累积 1、启动/运行温度 55/72°C、启动空闲显存至少 6500 MiB、单阶段最多 180 分钟，并强制 `nvidia-smi` 设备名为 `NVIDIA GeForce RTX 4070 Laptop GPU`。帧数、epoch 和保护门槛只能收紧，不能通过环境变量放宽。
- 400/100 选择仍写入 `formal_mini_chain_v2` 的 `mini_selection`，父 checkpoint preflight 会校验同一身份。VAE/LDM 每个 epoch 使用 100 帧 validation；当前 CD 训练器只接收 train loader，因此 CD 的 100 帧留出集需在训练后通过独立推理/评价消费，不能声称 CD 已逐 epoch 验证。
- 正式 `train_unified.sh` 不再继承默认配置的 VAE/LDM/CD `100/200/200` epoch，而是 fail-closed 固定 `20/20/20`；生成正式配置时显式移除两个 mini frame 字段，使用 garden 的完整 3210 train / 774 validation split 和 `formal_chain_v2`。
- 监督/体素/指标影响：本项不改 target、observed mask、loss、模型结构或指标公式；每帧仍有 `32×128×128=524288` 个空间体素，单帧显存不变。中型 VAE/LDM 各有 8000 个 batch-1 训练样本访问和 2000 个验证样本访问；样本覆盖更强但仍不能替代服务器 full-split 泛化评价。
- 最终真实零训练 preflight 在 RTX 4070 Laptop 上通过：8188 MiB、空闲 6619 MiB、42°C，artifact SHA-256=`11f59d84...e97c`、选择 400/100、epochs 20/20/20；新结果根未创建，原 smoke/short checkpoint SHA-256 仍为 `1ae08bf8...c61f50` / `a55c0bb0...03510`。

## 2026-08-28 medium VAE CUDA allocator 断言诊断

- 用户实际启动 `medium_train` 后，VAE 在 epoch 1 第 50 个 batch 的 `scaled_loss.backward()` 触发 `CUDACachingAllocator.cpp:2586` 内部断言；失败时 loss 为有限的 `4.4529`，日志没有 OOM、NaN 或数据读取异常。
- 真实传播链为 `run_formal_mini_8gb.sh → train_minimal.sh → unified_train.py`：笔记本入口在导入 PyTorch 前导出 `expandable_segments:True,max_split_size_mb:128`。本机为 PyTorch `2.4.1+cu121`；仓库中的 `apply_memory_optimizations()` 没有调用者，且其运行期环境写入不是本次触发源。
- 修复后 laptop 与正式 launcher 都固定 `max_split_size_mb:128`；runner 覆盖外部 hostile allocator、打印实际值，生成 YAML 在 `hardware.cuda_allocator_conf` 中记录它。运行期 helper 不再修改 allocator，消除隐藏环境依赖。
- 失败 `v1` 仅有 `mini_vae_config.yaml`、日志和空 metrics 表头，没有 checkpoint，已登记并原样保留；新运行使用 fresh `v2` 结果根，禁止把失败现场伪装成 resume。
- 本项不改变 target、observed mask、模型结构、loss、样本选择、每帧 `524288` 个空间体素或指标公式，只改变进程级 CUDA 内存分配策略与审计元数据。
- CPU 回归通过配置/安全 103/103、脚本协议 20/20、shell/Python 静态检查和 `git diff --check`。真实无训练预检因当前桌面占用导致空闲显存 6375 MiB，低于固定 6500 MiB 门槛而正确拒绝；没有放宽保护，也没有启动 GPU backward。
