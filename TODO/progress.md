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

## 2026-06-17
- Continued the explicit `planning-with-files-zh` + `test-driven-development` workflow for the airborne multimodal refactor.
- Added and ran `test/test_airborne_multimodal_refactor.py` for:
  - airborne voxelization sync-offset correction, egomotion Doppler compensation, and clipped Doppler variance;
  - dataset IR tensor and compensated calibration metadata return path;
  - IR-to-3D projection/fusion, including latent-shape downsampling;
  - unified training batch unpacking and 16-channel LDM multimodal entry construction.
- Updated `diffusion_consistency_radar/cm/multimodal_fusion.py`:
  - optional ResNet-18 IR extractor with fallback CNN when `torchvision` is unavailable;
  - registered 3D voxel centers and frustum-masked IR projection;
  - `CompleteDualModalityPerceptionNet` that fuses 4 radar channels + 32 IR channels into 16 channels and injects `noised_latent`.
- Updated `diffusion_consistency_radar/scripts/unified_train.py`:
  - LDM now builds a 16-channel `OptimizedUNetModel` backbone wrapped by `CompleteDualModalityPerceptionNet`;
  - train loop unpacks `(target, radar, meta)` batches, moves meta tensors to device, and sends `radar_vox/ir_img/r_mat/t_vec/k_mat/sigmas/noised_latent` through the multimodal path;
  - legacy `(target, radar)` batches still use the internal UNet with zero-padded 16-channel input.
- Verification:
  - `conda run -n Radar-Diffusion python test/test_airborne_multimodal_refactor.py` passed with 5 tests.
  - `conda run -n Radar-Diffusion python -m py_compile NTU4DRadLM_pre_processing/NTU4DRadLM_pre_processing.py diffusion_consistency_radar/cm/dataset_loader.py diffusion_consistency_radar/cm/multimodal_fusion.py diffusion_consistency_radar/scripts/unified_train.py test/test_airborne_multimodal_refactor.py` passed.
  - `conda run -n Radar-Diffusion python test/test_sensor_aware_target.py` passed with 5 tests.
  - `conda run -n Radar-Diffusion python test/test_shared_visibility_eval.py` passed with 2 tests.

## 2026-06-17 Continued
- Used `planning-with-files-zh` and `executing-plans` to implement the offline loop-closure plan.
- Added tests:
  - `test/test_multimodal_inference_interface.py`
  - `test/test_formal_task_metrics.py`
  - `test/test_dataset_protocol_metadata.py`
  - `test/test_probabilistic_mapping_uncertainty.py`
- Implemented multimodal inference compatibility:
  - `inference.py` detects multimodal checkpoint keys and builds `CompleteDualModalityPerceptionNet`.
  - `RadarGenerator.generate()` accepts `meta_dict`; CD/LDM sampling pass `radar_vox`, `ir_img`, `r_mat`, `t_vec`, `k_mat`, and `noised_latent` for multimodal checkpoints.
  - `--use_multimodal_meta` reads sidecar `ir_image/{frame}_ir.npy` and calibration metadata when available.
- Promoted task metrics into formal code:
  - Added `diffusion_consistency_radar/cm/evaluation_metrics.py`.
  - `inference.py` can append task-oriented summary fields with `--report_task_metrics`.
  - `diagnose_generation_quality.py` now writes near-obstacle precision/recall/BEV IoU into metrics/report output.
- Solidified dataset/preprocessing protocol:
  - Preprocessing writes `preprocess_policy.json`.
  - Dataset meta now includes `is_mock_ir`, `is_mock_calib`, and `preprocess_policy`.
- Implemented uncertainty-aware mapping:
  - `probabilistic_mapping.py` uses Doppler variance and range to lower observation reliability.
  - DEM variance now includes Doppler variance contribution.
  - `streaming_map_update.py` adds obstacle precision/recall/false-positive/mean-uncertainty metrics when target voxels are provided.
  - Fixed streaming input discovery so `*_pcl.npy` point clouds are not treated as voxel files.
- Verification:
  - `conda run -n Radar-Diffusion python test/test_multimodal_inference_interface.py` passed with 2 tests.
  - `conda run -n Radar-Diffusion python test/test_formal_task_metrics.py` passed with 2 tests.
  - `conda run -n Radar-Diffusion python test/test_dataset_protocol_metadata.py` passed with 2 tests.
  - `conda run -n Radar-Diffusion python test/test_probabilistic_mapping_uncertainty.py` passed with 2 tests.
  - `conda run -n Radar-Diffusion python test/test_airborne_multimodal_refactor.py` passed with 5 tests.
  - `conda run -n Radar-Diffusion python test/test_sensor_aware_target.py` passed with 5 tests.
  - `conda run -n Radar-Diffusion python test/test_shared_visibility_eval.py` passed with 2 tests.
  - `conda run -n Radar-Diffusion python -m py_compile ...` passed for inference, diagnosis, streaming map, evaluation metrics, probabilistic mapping, dataset loader, and preprocessing.
  - `conda run -n Radar-Diffusion python diffusion_consistency_radar/scripts/streaming_map_update.py --radar_voxel_dir Result/inference_results/loop3_ldm_eval --output_dir Result/inference_results/streaming_map_smoke --frame_limit 20` passed after fixing voxel-file filtering.
  - `git diff --check` initially failed on trailing whitespace in changed preprocessing/loader files; after mechanical whitespace cleanup it passed.

## 2026-06-17 CD / Mini Loop Closure
- Continued the implementation plan in Default mode after the planning pass.
- Finished CD multimodal compatibility:
  - `cd_train_optimized.py` now detects multimodal checkpoints, builds matching legacy/multimodal CD models, and routes student/EMA/teacher denoising through `call_cd_denoiser()`.
  - CD `train_epoch()` now accepts `(target, radar)`, `(target, radar, meta)`, and `(target, radar, meta, path)` batches.
  - Residual gradients are stepped at epoch end when the number of batches is not divisible by `grad_accum_steps`.
- Finished unified CD entry:
  - `unified_train.py` now supports `--mode cd`, `--ldm_ckpt`, and `cd.teacher_model_path` fallback.
  - `launch/train_unified.sh` now routes CD training through `unified_train.py` instead of a divergent standalone path.
- Extended mini loop:
  - `test/mini-test/train_minimal.sh` now supports `cd` and `all_with_cd`.
  - `test/mini-test/inference_minimal.sh` now defaults CD to `test/mini-test/train_results_mini/cd/cd_best.pt`, emits task metrics, and auto-detects multimodal checkpoint metadata.
  - `test/mini-test/run_minimal_experiment.sh` now runs VAE/LDM/CD and both LDM/CD inference.
- Added dataset protocol audit:
  - `diffusion_consistency_radar/scripts/audit_dataset_protocol.py` writes CSV/Markdown with IR coverage, policy presence, alignment mode, and calibration fallback status.
- Extended uncertainty-aware mapping:
  - `GridMapConfig.speed_m_s` adjusts window size, decay rate, and far-range reliability.
  - `streaming_map_update.py` accepts `--speed_m_s` and `--odom_cov_trace` and logs both in `streaming_metrics.csv`.
- Tests and smokes completed:
  - CD interface and entrypoint tests passed.
  - Dataset protocol audit and probabilistic mapping uncertainty tests passed.
  - Full listed regression tests passed once before final script fixes; targeted rechecks passed after script fixes.
  - `streaming_map_update.py` speed-50 smoke passed and wrote `Result/inference_results/streaming_map_speed50`.
  - Dataset audit smoke passed and wrote `Result/dataset_protocol_audit_smoke`.
  - Mini CD 1-epoch smoke passed after fixing heredoc/config generation.
  - Mini LDM and CD 1-frame inference smokes passed.
- Important note:
  - Before the mini config bug was fixed, one mini CD run used default CD settings and wrote outputs under `Result/train_results/cd`. This was left in place rather than deleted automatically.

## 2026-06-18 Mini Inference Diagnosis / Visualization

- Used `planning-with-files-zh` and `systematic-debugging` to analyze the 500-frame mini inference results before proposing any fix.
- Confirmed VAE/LDM/CD mini training completed with 500 samples and 10 epochs each.
- Confirmed 500-frame LDM and CD mini inference completed:
  - LDM metrics: `test/mini-test/inference_results_mini/loop3_ldm_eval/inference_metrics.csv`
  - CD metrics: `test/mini-test/inference_results_mini/loop3_cd_eval/inference_metrics.csv`
- Evidence gathered:
  - LDM summary `mean_pred_target_chamfer=7.591485`, `avg_infer_seconds=1.295140`.
  - CD summary `mean_pred_target_chamfer=8.399870`, `avg_infer_seconds=0.024213`.
  - Radar baseline target Chamfer from the same reports is `5.572554`.
  - Typical-frame point statistics show predictions are too near in x, too low in z, and CD is too dense.
- Added `test/generate_interactive_inference_compare.py` to create self-contained interactive HTML overlays for radar/target/LDM/CD point clouds.
- Generated visualizations:
  - `Result/visualization/mini_inference_compare/inference_compare_000068.html`
  - `Result/visualization/mini_inference_compare/inference_compare_000150.html`
  - `Result/visualization/mini_inference_compare/inference_compare_000253.html`
  - `Result/visualization/mini_inference_compare/inference_compare_000386.html`
  - `Result/visualization/mini_inference_compare/inference_compare_000478.html`
  - `Result/visualization/mini_inference_compare/inference_compare_000488.html`
- Verification:
  - `conda run -n Radar-Diffusion python -m py_compile test/generate_interactive_inference_compare.py` passed.
  - `conda run -n Radar-Diffusion python test/generate_interactive_inference_compare.py --frames 000068,000150,000253,000386,000478,000488 --output_dir Result/visualization/mini_inference_compare` passed.

## 2026-06-22 Sensor-Aware Mini Quality Correction

- Started Phase 6 after completing the new 500-frame sensor-aware mini train/inference run.
- Confirmed new checkpoints were used and all 500 LDM/CD frames produced voxel and uncertainty outputs.
- Diagnosed three linked issues: fixed threshold over-density, low near-range precision, and non-informative deterministic uncertainty.
- Selected implementation order: task-aware threshold calibration, per-frame metric reporting fix, then learnable uncertainty with Gaussian NLL.
- Completed broad saved-output count sweeps for LDM/CD across thresholds `0.1-0.9`.
- Evidence supports threshold calibration before retraining: density is near target at LDM `0.6` and CD `0.7`.
- Added task-region voxel Precision/Recall/F1/IoU to `sweep_occ_threshold.py` and JSON threshold recommendations.
- Fixed inference so task metric values are written into each frame row as well as the summary row.
- `test/test_formal_task_metrics.py` now has 3 passing tests; modified scripts compile successfully.
- Added trainable `model_uncertainty_head` and combined it with Doppler/metadata physical variance.
- Added heteroscedastic Gaussian NLL to LDM training (`uncertainty_loss_weight=0.05`) and changed inference sidecars to save variance.
- Fusion, inference, CD interface tests and compilation all pass after the uncertainty refactor.
- Added ECE, Brier, Bernoulli NLL, and uncertainty-error correlation to formal inference rows and summaries.
- Set the sensor-aware mini inference default occupancy threshold to the calibrated `0.5` while retaining environment override support.
- Completed an isolated 1-sample/1-epoch LDM uncertainty training smoke under `/tmp/radar_uncertainty_smoke`; checkpoint save and uncertainty-head parameter updates were verified.
- Completed Phase 6 regression verification: airborne fusion, formal metrics, inference compatibility, probabilistic mapping, dataset protocol, sensor-aware targets, shared visibility, CD interfaces/entrypoints, and mini script tests all pass.
- Final `py_compile`, both mini Bash syntax checks, and `git diff --check` passed.
- Remaining experiment action: retrain VAE/LDM/CD into a new result directory, then rerun inference at calibrated threshold `0.5` to obtain learned uncertainty calibration numbers.
