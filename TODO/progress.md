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

## 2026-06-23 Tree-Structure Recovery

- Used planning-with-files and executing-plans to start Phase 7 after visual inspection showed that tree structure was not reconstructed.
- Confirmed the current checkout is the non-main `withir` feature branch. It is not a linked worktree, but existing uncommitted multimodal changes are prerequisites, so work continues in place without copying or reverting unrelated changes.
- Quantified the data bottleneck on loop3 frame `000008`: original LiDAR voxel 4737 occupied cells, hard sensor-aware target 616 occupied cells.
- Measured the saved VAE upper bound: garden frame IoU `0.3595`; loop3 frame IoU `0.1448` at threshold `0.5`.
- Confirmed radar-to-thermal transform direction and voxel-axis interpretation are wrong in the current IR projection implementation.
- Added Phase 7 tasks to `TODO/task_plan.md`; implementation will follow TDD.
- The standalone `sensor_aware_target.py` was found deleted on continuation while its vectorized target logic remained in `NTU4DRadLM_pre_processing.py`. Earlier imports could resolve stale bytecode, so that test result is not accepted as final evidence. Phase 7 will migrate protocol tests to the integrated script and keep the deletion intact.
- Migrated `test_sensor_aware_target.py` to the integrated vectorized preprocessing function, removing reliance on deleted source/stale bytecode.
- Completed RED/GREEN for `visibility_mode`: `preserve` retains cropped LiDAR structure, `hard` keeps legacy radar-neighborhood masking, and the one-click preprocessing script now defaults to `VISIBILITY_MODE=preserve`.
- Verification: 4 target-protocol tests pass and `bash -n NTU4DRadLM_pre_processing/preprocess.sh` passes.
- Completed IR geometry RED/GREEN: projection grids now map tensor `(Z,X,Y)` to physical `(x,y,z)`, and calibration is applied as the dataset defines it, `p_camera=R*p_radar+T`.
- Two focused IR geometry tests pass independently; projection buffers remain non-persistent, so checkpoint state dictionaries remain compatible.

## 2026-06-26 Phase 7 Continuation

- Continued Phase 7 using `planning-with-files-zh` and `executing-plans`.
- Confirmed current checkout is normal repo branch `withir`, not a linked worktree. Because existing uncommitted multimodal changes are prerequisites, continued in place and did not revert/delete unrelated files.
- Verified the dataset near-field crop test by directly running `test/test_dataset_protocol_metadata.py -v`; the earlier `python -m unittest test...` form failed only because `test/` is not a Python package.
- Propagated configurable grid protocol through:
  - `diffusion_consistency_radar/scripts/unified_train.py`
  - `diffusion_consistency_radar/scripts/cd_train_optimized.py`
  - `diffusion_consistency_radar/scripts/inference.py`
  - `test/mini-test/train_minimal.sh`
  - `test/mini-test/inference_minimal.sh`
- Added inference-side voxel crop coverage to `test/test_dataset_protocol_metadata.py`.
- Added `diffusion_consistency_radar/scripts/diagnose_vae_reconstruction.py` for VAE reconstruction upper-bound diagnostics and `test/test_vae_reconstruction_diagnostic.py` for metric unit tests.
- Smoke-ran the VAE diagnostic for 1 loop3 frame using `test/mini-test/train_results_mini_calibrated/vae/vae_best.pt`; outputs were saved under `test/result/vae_reconstruction_smoke`.
- Verification completed:
  - `conda run -n Radar-Diffusion python test/test_dataset_protocol_metadata.py -v` passed with 6 tests.
  - `conda run -n Radar-Diffusion python test/test_vae_reconstruction_diagnostic.py -v` passed with 3 tests.
  - `conda run -n Radar-Diffusion python test/test_sensor_aware_target.py -v` passed with 4 tests.
  - `conda run -n Radar-Diffusion python test/test_airborne_multimodal_refactor.py -v` passed with 9 tests.
  - `conda run -n Radar-Diffusion python test/test_multimodal_inference_interface.py` passed with 2 tests.
  - `py_compile` passed for inference, unified train, CD train, VAE diagnostic, dataset loader, and multimodal fusion.
  - `bash -n` passed for both mini train and mini inference scripts.
- The 1-sample/1-epoch mini VAE smoke initially failed inside the sandbox with OpenMPI socket/interface initialization errors. The same command passed outside the sandbox, confirming this was an execution-environment issue rather than a code issue.
- Small VAE overfit smoke result: 1 garden sample, 1 epoch, near-field `model_pc_range=0,-20,-6,40,20,10`, loss `1.7175`, checkpoint saved under `/tmp/radar_phase7_vae_smoke/results/vae`.
- Phase 7 is now implementation-complete. Next experimental step is to regenerate/retrain mini near-field VAE/LDM/CD with enough samples, then compare tree/obstacle structure against raw LiDAR visualizations before formal full retraining.

## 2026-06-29 VAE Upper-Bound Recovery Planning

- Reviewed the completed 500-frame near-field VAE reconstruction report and training history.
- Confirmed best aggregate IoU is `0.3177`, with recall `0.4360` and precision `0.5393`.
- Quantified target sparsity at `0.106%` occupied voxels and identified the current MSE-style occupancy objective plus ultra-lightweight bottleneck as higher-priority issues than KL regularization.
- Added Phase 8 to `TODO/task_plan.md`.
- Saved the TDD implementation and experiment plan to `docs/superpowers/plans/2026-06-29-vae-occupancy-upper-bound-recovery.md`.
- No long training command was executed during this planning pass.

## 2026-06-29 VAE Upper-Bound Recovery Execution

- Started Tasks 1-3 and the 32-frame overfit experiment using subagent-driven development.
- Confirmed the current checkout is the `withir` feature branch in a normal repository checkout.
- Continued in place because the uncommitted Phase 7 multimodal/grid changes are prerequisites for this work; no existing changes were reverted or copied into a separate worktree.
- Baseline verification passed before Task 1:
  - `test/test_vae_reconstruction_diagnostic.py`: 3 tests passed.
  - `test/test_multimodal_inference_interface.py`: 2 tests passed.
- Task 1 completed with TDD and two-stage review:
  - Added explicit occupancy-logit probability conversion without changing decoder output channels.
  - Diagnostics preserve raw semantics for legacy checkpoints and use sigmoid only when checkpoint metadata requests it.
  - Sparse occupancy tests pass (2/2); diagnostic tests pass (5/5).
  - Spec-compliance and code-quality reviews both approved the final Task 1 implementation.
- Task 2 completed with TDD and two-stage review:
  - Added FP32 BCE+Dice occupancy supervision with capped dynamic positive weighting.
  - Continuous channels now use masked Smooth L1 only in occupancy/Doppler-valid regions.
  - Preserved the legacy weighted-MSE path for historical experiments.
  - Made VAE metric logging robust to non-finite batches, partial gradient accumulation, empty loaders, and old CSV schemas.
  - Sparse-loss/trainer tests pass (16/16); diagnostic tests pass (5/5).
  - Spec-compliance and code-quality reviews approved the final Task 2 implementation.

## 2026-06-29 Task 3 Review Fixes

- RED：共享网格 resolver、diagnostic/inference 网格适配器和 CD checkpoint helper 缺失。
- GREEN：checkpoint 10/10、occupancy loss 16/16、diagnostic 8/8、inference 6/6。
- CD 多模态接口与 CD 入口测试通过；相关文件 `py_compile` 和 `git diff --check` 通过。
- sandbox 内 OpenMPI socket 初始化失败，已在 sandbox 外用 Radar-Diffusion 环境完成短测试。
- 未运行训练，未修改 mini 脚本。

## 2026-06-29 Task 3 Final Review

- RED：trainer 缺少统一 best 状态更新步骤；无条件 inference 未调用 VAE shape 推导。
- GREEN：checkpoint 11/11、inference 7/7、occupancy loss 16/16、diagnostic 8/8，
  CD 接口与入口测试通过。
- 相关生产/测试文件 `py_compile` 与 `git diff --check` 通过；未运行训练或修改 mini。

## 2026-06-29 Task 3 Quality Review

- RED：z8 inference build API 缺失、CD legacy UNet 仍为 8->4、checkpoint 无 scheduler/
  atomic helper。
- GREEN：checkpoint 15/15、inference/unified z8 10/10、loss 16/16、diagnostic 8/8，
  CD 接口与入口测试通过。
- `py_compile` 与 `git diff --check` 通过；未运行训练、未修改 mini、未提交。
- Task 3 passed final spec-compliance and code-quality reviews after the z8 latent,
  scheduler-resume, and atomic-checkpoint fixes.

## 2026-06-30 32-Frame Overfit Retry

- The first 32-frame run stopped before epoch 1 because the historical lightweight
  `base_channels=24` preset was incompatible with a fixed 32-group GroupNorm.
- Root cause was reproduced at VAE construction: 24 and 72 channels are not divisible
  by 32.
- Updated shared normalization to choose the largest valid group divisor while preserving
  all channel widths and checkpoint parameter shapes.
- Added 10 focused tests, including a real lightweight latent-8 VAE forward pass.
- GroupNorm fix passed both specification and code-quality review.
- Completed 100 VAE epochs in about 0.40 hours:
  - final training loss: `0.6912`
  - final one-frame validation IoU/recall/precision: `0.6626/0.8410/0.7575`
- Ran the 32-frame reconstruction diagnostic with `vae_best_iou.pt`:
  - best threshold: `0.55`
  - IoU/recall/precision: `0.8417/0.9727/0.8621`
  - predicted/target occupancy count ratio: `1.1284`
- The 32-frame overfit gate passed.
- Final review fixes completed:
  - decoded LDM occupancy losses now respect VAE raw/sigmoid semantics;
  - metadata-free fallback restores `legacy_mse + raw`;
  - strict inference loading rejects empty or structurally incomplete checkpoints.
- Final regression passed:
  - inference interface: 13 tests
  - checkpoint protocol: 18 tests
  - sparse occupancy/trainer: 17 tests
  - VAE diagnostic: 8 tests
  - CD multimodal interface and CD entrypoint tests
  - `py_compile`, mini Bash syntax, and `git diff --check`

## 2026-06-30 Final Review TDD

- RED: checkpoint test import failed because the activation-aware decoded helper was absent.
- RED: strict multimodal inference accepted a checkpoint with one compatible tensor.
- RED: component-weight coverage exposed that the initial helper API did not preserve
  independent decoded MSE/FP/mass weights.
- RED: loaded VAE lacked the resolved `occupancy_activation` runtime attribute.
- GREEN: checkpoint protocol tests passed 18/18 and inference interface tests passed 11/11.
- GREEN: full regression passed with loss 17/17, checkpoint 18/18, inference 11/11,
  diagnostic 8/8, plus both CD interface/entrypoint scripts.
- Target production/tests passed `py_compile`; `git diff --check` passed. No training,
  commit, rollback, dataset, checkpoint, log, or experiment-result operation was run.
- Follow-up RED: `strict=True` returned a random model for `{}`, and a state missing
  `out.2.weight` lacked a clear critical-weight protocol error.
- Follow-up GREEN: inference interface passed 13/13 after adding empty-state and critical
  first/last-layer validation; checkpoint/CD regression and static checks are next.
- Follow-up static checks passed: inference/test `py_compile` and `git diff --check`.
- The combined checkpoint/CD rerun was blocked before test execution because the current
  sandbox uses `--unshare-net` and OpenMPI aborts while creating its initialization socket.
  An outside-sandbox request was rejected by policy, so no checkpoint/CD pass is claimed
  for this follow-up run. The earlier final-review run passed checkpoint 18/18 and both CD
  scripts before this inference-only change.

## 2026-07-01 Threshold Scan Review And Planning

- Inspected the LDM threshold recommendation, sweep CSV, formal inference summary, and
  threshold-sweep implementation.
- Rejected the current `0.1` recommendation because its exact voxel F1 is `0.007126` and
  the sweep compares a `0-40m` prediction grid with a target resized from `0-120m`.
- Confirmed meaningful LDM task-level improvement at threshold `0.5`: near BEV IoU
  `0.4478`, recall `0.5742`, precision `0.6646`, and Chamfer `1.3749` versus radar
  baseline `2.0553`.
- Added Phase 9 and saved the implementation/experiment plan to
  `docs/superpowers/plans/2026-07-01-ldm-threshold-evaluation-and-cd-gate.md`.
- No production code, prediction voxel, checkpoint, dataset, or training result was
  modified; no inference or long training command was run.

## 2026-07-01 Threshold Protocol Task 1

- Added a real `.npy` regression test for source `0-120m` to model `0-40m` cropping.
- Updated the threshold sweep target loader to crop `(C,Z,X,Y)` by physical range before
  sparse-aware resize.
- Added explicit source/model ranges and target size to the CLI and recommendation JSON.
- RED failed on the old two-argument loader; GREEN passed the new crop test and
  `py_compile`.
- Independent specification review passed. Independent quality review found no
  Critical/Important issues; only dynamic docstring, target-size validation, and optional
  `.npz` edge coverage were noted as minor follow-ups.
- The existing dirty worktree was preserved and no commit was created.

## 2026-07-01 Threshold Protocol Task 2

- Added deterministic train/validation selection matching the trainer's seeded
  `torch.randperm` protocol.
- Added `0-20m` and `20-40m` task metrics with BEV micro precision/recall/F1/IoU.
- Changed NN match aggregation to matched prediction count divided by query count; empty
  target plus non-empty prediction now contributes zero rather than being skipped.
- Added fail-fast checks for missing prediction frames and missing selected validation
  targets, plus finite/non-overlapping/model-bounded range-bin validation.
- Preserved strict voxel metrics as diagnostics and made task BEV F1 the default selector.
- External verification passed: grid protocol `10/10`, formal metrics `4/4`,
  `py_compile`, and `git diff --check`.
- Independent specification and code-quality reviews passed after two repair cycles.

## 2026-07-01 Threshold Protocol Task 3

- Re-scanned the saved LDM outputs using the core protocol parameters:
  `source_pc_range=[0,-20,-6,120,20,10]`,
  `model_pc_range=[0,-20,-6,40,20,10]`, target size `(32,128,128)`,
  deterministic `validation` split with `train_split=0.8` and `split_seed=42`,
  range bands `0-20m/20-40m`, and `task_bev_f1` selection.
- The 100-frame validation scan recommended occupancy threshold `0.1`, with overall BEV
  F1 `0.5845`, BEV IoU `0.4129`, match@2m `0.9622`, and point-count ratio `1.1435`.
- Produced a separate 500-frame fixed-threshold `0.1` reevaluation output without
  overwriting the old threshold `0.5` results.
- Reviewed raw-LiDAR comparison HTML files in
  `test/result/ldm_near40_500_v2_threshold_validated/raw_lidar_visuals/` and the
  fixed-seed validation-order set in
  `test/result/ldm_near40_500_v2_threshold_validated/raw_lidar_visuals_random_validation/`.
- Review conclusion: threshold `0.1` improves occupancy coverage and BEV recall/IoU but
  increases predicted points by `17.39%` and worsens near NN mean from `0.4731` to
  `0.5116`. Both 10-frame visual reviews show broad obstacle coverage but unstable trunk
  continuity and canopy detail, so the tree-structure visual gate does not pass.
- The old/new comparisons are separate stochastic LDM runs over one `loop3` internal
  split. Their quality and latency differences are not pure threshold effects and are not
  independent-scene generalization evidence. CD remains gated.

## 2026-07-02 Threshold Protocol Task 4

- 完成 CD 准入判定与记录；按项目规则未运行长时间训练、推理或 CD 蒸馏。
- 内部数值门槛全部通过，但统计集合不同：near BEV IoU/recall/precision
  `0.460417/0.610190/0.649795` 和 Chamfer `1.310079` 来自包含训练帧的 500 帧
  全量复评；threshold selection 和点数比例 `1.143535` 来自 100 帧 validation。
  正式准入需统一到独立 validation/test 集，当前 HOLD 结论不受影响。
- 按 `split_seed=42` 确定性划分得到的 validation 可视化帧
  `280,195,103,303,229,311,37,454,431,493` 中，`10/10` 大致覆盖主要障碍物
  区域，但 `280/311` 明显稀疏，树干连续性和树冠细结构均未稳定恢复。这里固定的
  是数据划分，不代表 LDM 生成采样 seed 固定。
- Gate 总判定为 **HOLD / FAIL**：当前不启动 CD，避免把 LDM 教师的树木结构缺陷
  蒸馏并固化到学生模型。
- 后续建议：
  - 先实现高度覆盖率、垂直连通率、树干区域 recall 等结构指标；最高点高度召回、
    垂直连通率和树干区域 recall 的正式门槛需依据实验分布确定，本阶段不设虚构阈值。
  - 使用新增结构指标检查 VAE 重建上界。
  - 若 VAE 通过而 LDM 失败，再加入垂直结构或高度分布损失并重训 LDM。
  - 若 VAE 也失败，先提高 Z/X 物理分辨率或调整监督目标。
  - 后续阈值对比固定 LDM 推理随机 seed，或复用同一批保存的 prediction voxel，
    排除随机采样干扰。

## 2026-07-02 Phase 9 Final Regression

- Final short regression passed outside the socket-restricted sandbox:
  - occupancy threshold grid protocol: `14/14`
  - formal task metrics: `4/4`
  - multimodal inference interface: `13/13`
  - raw-LiDAR interactive visualization: `3/3`
- 其中 occupancy threshold grid protocol 的 `14/14` 为最终 reviewer 修补后复跑结果。
- `py_compile` passed for the threshold sweep, inference, and interactive visualization
  scripts; `git diff --check` passed.
- Phase 9 is complete. CD training and inference were intentionally not run because the
  tree-structure visual gate remains failed. Structure recovery continues as Phase 10.

## 2026-07-06 Phase 10 Vertical-Structure Metrics And VAE Gate

- Added `vertical_structure_metrics()` with height coverage, top-height recall, vertical
  connectivity recall, and trunk-region recall plus micro-aggregation counts.
- Added `test/test_vertical_structure_metrics.py`; focused metric regression passed
  `10/10`.
- Integrated the four metrics into `diagnose_vae_reconstruction.py` per-frame CSV,
  threshold summary CSV, and best-IoU report.
- Added strict structure-row completeness checks while preserving legacy binary-only
  summary rows. The VAE diagnostic regression passed `13/13`.
- Completed two-stage subagent review for both metric implementation and diagnostic
  integration; the final specification and code-quality reviews approved the changes.
- Ran a 32-frame VAE structure diagnostic with the overfit checkpoint:
  IoU `0.8417`, height coverage `0.9727`, top-height recall `0.9184`,
  vertical connectivity `0.9735`, and trunk-region recall `0.9845`.
- Ran the same 32-frame diagnostic with the 500-frame VAE checkpoint:
  IoU `0.8477`, height coverage `0.9637`, top-height recall `0.9230`,
  vertical connectivity `0.9663`, and trunk-region recall `0.9720`.
- No training was run. Supervision targets, voxel counts, model grid, and checkpoints
  were unchanged; only the evaluation/reporting surface and test outputs changed.
- Decision: the current VAE can preserve vertical structure. The next implementation
  target is LDM vertical-structure/height-distribution supervision; CD remains on hold.
- The first combined final-regression attempt was blocked before the first test by the
  sandbox OpenMPI socket restriction; an outside-sandbox rerun is required before making
  a final pass claim.
- Outside-sandbox final verification passed: vertical metric tests `10/10`, VAE
  diagnostic tests `13/13`, focused `py_compile`, and `git diff --check`.
- A fresh final reviewer approved the complete Phase 10 increment, including metric
  semantics, diagnostic integration, result interpretation, and TODO documentation.

## 2026-07-06 Phase 10 LDM Structure-Supervision Start

- Recovered the Phase 10 plan and confirmed the VAE structure gate remains passed.
- Inspected the LDM training path: `OptimizedLDMTrainer` already decodes predicted
  latents for optional occupancy auxiliary losses, so structure supervision can be added
  without changing the network architecture or checkpoint tensor shapes.
- Selected a backward-compatible design: differentiable Z-height distribution loss plus
  adjacent-Z continuity loss, both disabled by default with zero weights.
- Work continues in the existing dirty `withir` workspace because the feature depends on
  the current uncommitted VAE/LDM protocol changes; no separate worktree was created.
- Saved the approved implementation plan to
  `docs/superpowers/plans/2026-07-06-ldm-vertical-structure-supervision.md`.
- Task 1 implementer completed the loss helper with RED import-failure evidence and
  reported `8/8` GREEN tests. The controller's first independent rerun was blocked before
  test import by the known sandbox OpenMPI socket restriction.
- Outside-sandbox Task 1 verification passed `8/8`; specification review passed.
- Code-quality review requested numerical fixes for float16 sigmoid gradients and
  all-zero predicted-column normalization before trainer integration.

## 2026-07-07 LDM Vertical Structure Supervision Continuation

- Continued Phase 10 using `planning-with-files-zh` and `subagent-driven-development`.
- Recovered the active plan, current Phase 10 state, and the previous Task 1 review blocker.
- Dispatched a fresh worker to fix the `decoded_vertical_structure_losses()` numerical risks without touching unrelated dirty files.
- Task 1 fix completed:
  - occupancy channel is converted to `float32` before sigmoid/raw clamp;
  - `eps` is validated as a finite positive value;
  - all-empty prediction columns use a bounded denominator instead of `1/eps`;
  - raw occupancy clamp keeps the historical raw/probability compatibility path explicit.
- Focused verification reported by the worker:
  - `conda run -n Radar-Diffusion python test/test_ldm_vertical_structure_loss.py -v` passed with 11 tests.
  - `conda run -n Radar-Diffusion python -m py_compile diffusion_consistency_radar/scripts/unified_train.py test/test_ldm_vertical_structure_loss.py` passed.
  - `git diff --check -- diffusion_consistency_radar/scripts/unified_train.py test/test_ldm_vertical_structure_loss.py` passed.
- Two-stage review passed:
  - specification review: PASS.
  - code-quality review: PASS, with only a non-blocking suggestion to add a future multi-batch/mixed-column mask test.

## 2026-07-07 LDM Structure Loss Trainer Integration

- Completed Task 2 with subagent-driven implementation and two-stage review.
- `OptimizedLDMTrainer` now reads `decoded_height_distribution_weight` and `decoded_vertical_continuity_weight` with code fallback `0.0`; default project configs set both to `0.02`.
- LDM loss computation now keeps latent MSE as the primary term and independently adds decoded occupancy, height distribution, vertical continuity, and uncertainty losses.
- Decoded auxiliary losses share a single `vae.decode(denoised)` call per batch when enabled, and skip decoding entirely when all decoded auxiliary weights are zero.
- LDM CSV logging now includes component columns for latent, decoded occupancy, height, continuity, and uncertainty losses; old LDM CSV headers are archived on resume before writing the new schema.
- LDM checkpoints now include `ldm_loss_config` while old checkpoints remain loadable because resume does not require the new field.
- Mini training script now supports `MINI_LDM_HEIGHT_WEIGHT` and `MINI_LDM_CONTINUITY_WEIGHT`, and static script tests verify the values reach the generated YAML.
- Code-quality review required and verified three stability fixes:
  - tail gradient accumulation is rescaled by actual accumulated batch count;
  - AMP tail rescale now happens after `GradScaler.unscale_()` and before clipping;
  - uncertainty loss raises a clear `ValueError` if `variance` is missing.
- Verification completed: structure-loss tests `18/18`, mini train script tests `10/10`, focused `py_compile`, and targeted `git diff --check` all passed.

## 2026-07-07 LDM Structure Smoke Verification

- Ran the planned short LDM smoke into `test/result/ldm_vertical_structure_smoke/`.
- The first sandbox run failed before training with the known OpenMPI socket restriction.
- The outside-sandbox 1-sample run passed OpenMPI but hit the project split guard: `deterministic_split_indices` requires at least 2 samples.
- Re-ran the same smoke with 2 samples, 1 epoch, 0 workers, `MINI_LDM_HEIGHT_WEIGHT=0.02`, and `MINI_LDM_CONTINUITY_WEIGHT=0.02`; it completed successfully in about 0.9s of training time.
- Smoke metrics row: total loss `0.475098`, latent `0.357750`, decoded occupancy `0.001069`, height distribution `0.323127`, vertical continuity `0.069169`, uncertainty `2.168658`.
- Confirmed `ldm_best.pt` stores `ldm_loss_config` with height/continuity weights `0.02/0.02`.
- Additional verification passed:
  - `test/test_ldm_vertical_structure_loss.py -v`: 18 tests.
  - `test/test_mini_train_script.py -v`: 10 tests.
  - `test/test_multimodal_inference_interface.py -v`: 13 tests.
  - focused `py_compile` and full `git diff --check`.
- No formal multi-epoch LDM retraining was run; that remains an explicit experiment step after this code change.

## 2026-07-07 Final Review

- Final reviewer approved the complete Phase 10 LDM vertical-structure supervision increment.
- Reviewer confirmed the LDM train path, one-decode auxiliary losses, CSV migration, checkpoint `ldm_loss_config`, and smoke output are consistent.
- Reviewer also confirmed the existing 500-frame VAE checkpoint carries the expected sigmoid occupancy semantics, latent_dim=8, BCE+Dice protocol, and near-field grid metadata, so the next step can be 500-frame LDM retraining followed by vertical metrics evaluation.
