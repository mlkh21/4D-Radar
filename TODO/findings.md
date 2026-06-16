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
