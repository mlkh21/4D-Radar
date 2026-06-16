# Progress

## 2026-06-15
- Read the requested `planning-with-files-zh` skill instructions.
- Confirmed no existing root `task_plan.md` or `.planning` directory was present.
- Read the provided JSONL rollout enough to recover the unfinished task: implement/run common-visible-region evaluation for radar/LiDAR alignment on loop3.
- Created planning files in the project root.
- Inspected existing scripts: `alignment_sanity_check.py`, `check_radar_axis_conventions.py`, `compare_voxel_triplets.py`, and `generate_interactive_raw_compare.py`.
- Added `test/shared_visibility_eval.py` and `test/test_shared_visibility_eval.py`.
- Confirmed default Windows `python` can byte-compile the new files, but it lacks `numpy` for runtime tests.
- Tried `conda run -n Radar-Diffusion`, but Windows has no usable corresponding environment in this sandbox; user asked to defer syntax/runtime checks to Ubuntu.

## 2026-06-16
- Re-read `test/shared_visibility_eval.py` and confirmed it writes `frame_metrics.csv`, `summary_metrics.csv`, and `shared_visibility_report.md`.
- Ran `python -m py_compile test/shared_visibility_eval.py test/test_shared_visibility_eval.py`; passed.
- Ran `conda run -n Radar-Diffusion python test/test_shared_visibility_eval.py`; passed with 2 tests.
- Ran shared visibility evaluation for:
  - `Data/NTU4DRadLM_Pre/loop3`
  - `Data/NTU4DRadLM_Pre_alignfix/loop3`
  - `Data/NTU4DRadLM_Pre_radarframe/loop3`
- Wrote outputs under `Result/alignment_check/loop3/shared_visibility_*`.
- Updated `TODO/task_plan.md` after fixing invalid UTF-8 content.
- Updated `TODO/findings.md` with loop3 metrics and conclusion.
- Reviewed `README.md`, `INFERENCE_GUIDE.md`, `default_config.yaml`, and `data_loading_config.yml` to align the next step with the repository's formal flow: preprocessing -> VAE -> LDM -> CD -> inference/diagnosis -> streaming map update.
- Added Phase 2-5 to `TODO/task_plan.md`: sensor-aware protocol, filtered/shared-visible targets, retraining comparison, and map-update integration.
- Added the recommended next direction to `TODO/findings.md`.
- Used `planning-with-files-zh` and `test-driven-development` for the sensor-aware target implementation.
- TDD RED/GREEN cycles completed:
  - Added `test/test_sensor_aware_target.py`; first failure confirmed missing module.
  - Implemented `NTU4DRadLM_pre_processing/sensor_aware_target.py`.
  - Added tests for height/range filtering, radar-visible neighborhood filtering, scene generation, dataset-root generation, and `max_files`.
- Verification:
  - `conda run -n Radar-Diffusion python test/test_sensor_aware_target.py` passed with 5 tests.
  - `python -m py_compile NTU4DRadLM_pre_processing/sensor_aware_target.py test/test_sensor_aware_target.py` passed.
  - `conda run -n Radar-Diffusion python test/test_shared_visibility_eval.py` passed with 2 tests.
- Generated a 120-frame loop3 smoke dataset under `Data/NTU4DRadLM_Pre_sensor_aware`.
- Verified `NTU4DRadLM_VoxelDataset` can load the generated dataset: 120 samples, target/radar tensors both `(4, 32, 128, 128)`.
- Checked frame `000000`: original target occupancy 4709, sensor-aware target occupancy 659, Doppler mask 228, radar occupancy 519.
