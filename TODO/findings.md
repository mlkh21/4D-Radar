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
- Existing Windows Python lacks `numpy`; default `py_compile` succeeded before the user asked to defer further checks, but runtime validation should be done on Ubuntu with the project environment.

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
