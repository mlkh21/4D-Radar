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
