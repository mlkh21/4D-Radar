# Progress

## 2026-07-13：完成列级损失计划 Task3 runner
- RED：完整 `test/unit/test_mini_train_script.py` 首次运行 78 项中 18 failed、1 error，确认缺失 env/YAML/runner；新增真实接口 smoke 后再次因 generic 非空目录交接得到 1 个预期失败。
- GREEN：新增三项列损失 env/YAML/打印/透传；新增训练-only Z64 v10 A/B runner、路径与 symlink 防护、非空拒绝、原子锁、统一 CUDA、VAE/最终 LDM checkpoint 检查。
- GREEN：`conda run -n Radar-Diffusion python test/unit/test_mini_train_script.py` 通过 79 项；未启动 inference、ablation、CD 或长时间训练。
- 规格复审 RED：新增 allowlist、运行时路径重审、固定协议抗覆盖、空 checkpoint 与 train-only 真实接口断言后，确认 generic/v10 的对应缺口。
- 规格复审 GREEN：完整脚本测试扩展至 85 项，覆盖 `ALLOW_OVERWRITE` 不绕过 allowlist、恶意 env 不覆盖协议、空 VAE/final LDM 拒绝，以及真实接口仅调用一次 LDM 训练。
- Important 复审 RED：fresh-scratch 测试证明历史分支会删除现有目录和 symlink 目标 marker，并缺少 dangling symlink 拒绝。
- Important 复审 GREEN：新增非破坏 fresh-scratch 分支，完整脚本测试扩展至 89 项；现有目录、有效/dangling symlink 均拒绝且内容保留，全新实验内路径正常创建，v10 真实接口继续通过。
- Config 竞态 RED：新增入口路径与独占写测试后，确认现有文件/symlink 未被拒绝、生成器仍会覆盖竞态文件、v10 未启用 fresh config。
- Config 竞态 GREEN：新增 fresh-config 入口审计和 YAML `x` 模式，完整脚本测试扩展至 91 项；外部目标不变、新 config 创建成功、检查后替换模拟被原子拒绝，v10 smoke 通过。
- 质量复审 RED：generic 接受外部 `/tmp` scratch/config，v10 持锁后注入内容未被拒绝；测试隔离不足还使一次 generic 负例短暂进入真实训练并因 GPU OOM 退出，未形成长训练结果。
- 质量复审 GREEN：generic 增加 EXP 子路径 canonical 契约，v10 增加持锁后空目录复检；完整脚本测试扩展至 94 项，所有负例均在训练前停止。

## 2026-06-15
- Read the requested `planning-with-files-zh` skill instructions.
- Confirmed no existing root `task_plan.md` or `.planning` directory was present.
- Read the provided JSONL rollout enough to recover the unfinished task: implement/run common-visible-region evaluation for radar/LiDAR alignment on loop3.
- Created planning files in the project root.
- Inspected existing scripts: `alignment_sanity_check.py`, `check_radar_axis_conventions.py`, `compare_voxel_triplets.py`, and `generate_interactive_raw_compare.py`.
- Added `test/diagnostics/alignment/shared_visibility_eval.py` and `test/unit/test_shared_visibility_eval.py`.
- Confirmed default Windows `python` can byte-compile the new files, but it lacks `numpy` for runtime tests.
- Tried `conda run -n Radar-Diffusion`, but Windows has no usable corresponding environment in this sandbox; user asked to defer syntax/runtime checks to Ubuntu.

## 2026-06-16
- Re-read `test/diagnostics/alignment/shared_visibility_eval.py` and confirmed it writes `frame_metrics.csv`, `summary_metrics.csv`, and `shared_visibility_report.md`.
- Ran `python -m py_compile test/diagnostics/alignment/shared_visibility_eval.py test/unit/test_shared_visibility_eval.py`; passed.
- Ran `conda run -n Radar-Diffusion python test/unit/test_shared_visibility_eval.py`; passed with 2 tests.
- Ran shared visibility evaluation for:
  - `Data/NTU4DRadLM_Pre/loop3`
  - `Data/NTU4DRadLM_Pre_alignfix/loop3`
  - `Data/NTU4DRadLM_Pre_radarframe/loop3`
- Wrote outputs under `test/result/comparison/alignment_check/loop3/shared_visibility_*`.
- Updated `TODO/task_plan.md` after fixing invalid UTF-8 content.
- Updated `TODO/findings.md` with loop3 metrics and conclusion.
- Reviewed `README.md`, `INFERENCE_GUIDE.md`, `default_config.yaml`, and `data_loading_config.yml` to align the next step with the repository's formal flow: preprocessing -> VAE -> LDM -> CD -> inference/diagnosis -> streaming map update.
- Added Phase 2-5 to `TODO/task_plan.md`: sensor-aware protocol, filtered/shared-visible targets, retraining comparison, and map-update integration.
- Added the recommended next direction to `TODO/findings.md`.
- Used `planning-with-files-zh` and `test-driven-development` for the sensor-aware target implementation.
- TDD RED/GREEN cycles completed:
  - Added `test/unit/test_sensor_aware_target.py`; first failure confirmed missing module.
  - Implemented `NTU4DRadLM_pre_processing/sensor_aware_target.py`.
  - Added tests for height/range filtering, radar-visible neighborhood filtering, scene generation, dataset-root generation, and `max_files`.
- Verification:
  - `conda run -n Radar-Diffusion python test/unit/test_sensor_aware_target.py` passed with 5 tests.
  - `python -m py_compile NTU4DRadLM_pre_processing/sensor_aware_target.py test/unit/test_sensor_aware_target.py` passed.
  - `conda run -n Radar-Diffusion python test/unit/test_shared_visibility_eval.py` passed with 2 tests.
- Generated a 120-frame loop3 smoke dataset under `Data/NTU4DRadLM_Pre_sensor_aware`.
- Verified `NTU4DRadLM_VoxelDataset` can load the generated dataset: 120 samples, target/radar tensors both `(4, 32, 128, 128)`.
- Checked frame `000000`: original target occupancy 4709, sensor-aware target occupancy 659, Doppler mask 228, radar occupancy 519.

## 2026-06-17
- Continued the explicit `planning-with-files-zh` + `test-driven-development` workflow for the airborne multimodal refactor.
- Added and ran `test/unit/test_airborne_multimodal_refactor.py` for:
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
  - `conda run -n Radar-Diffusion python test/unit/test_airborne_multimodal_refactor.py` passed with 5 tests.
  - `conda run -n Radar-Diffusion python -m py_compile NTU4DRadLM_pre_processing/NTU4DRadLM_pre_processing.py diffusion_consistency_radar/cm/dataset_loader.py diffusion_consistency_radar/cm/multimodal_fusion.py diffusion_consistency_radar/scripts/unified_train.py test/unit/test_airborne_multimodal_refactor.py` passed.
  - `conda run -n Radar-Diffusion python test/unit/test_sensor_aware_target.py` passed with 5 tests.
  - `conda run -n Radar-Diffusion python test/unit/test_shared_visibility_eval.py` passed with 2 tests.

## 2026-06-17 Continued
- Used `planning-with-files-zh` and `executing-plans` to implement the offline loop-closure plan.
- Added tests:
  - `test/unit/test_multimodal_inference_interface.py`
  - `test/unit/test_formal_task_metrics.py`
  - `test/unit/test_dataset_protocol_metadata.py`
  - `test/unit/test_probabilistic_mapping_uncertainty.py`
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
  - `conda run -n Radar-Diffusion python test/unit/test_multimodal_inference_interface.py` passed with 2 tests.
  - `conda run -n Radar-Diffusion python test/unit/test_formal_task_metrics.py` passed with 2 tests.
  - `conda run -n Radar-Diffusion python test/unit/test_dataset_protocol_metadata.py` passed with 2 tests.
  - `conda run -n Radar-Diffusion python test/unit/test_probabilistic_mapping_uncertainty.py` passed with 2 tests.
  - `conda run -n Radar-Diffusion python test/unit/test_airborne_multimodal_refactor.py` passed with 5 tests.
  - `conda run -n Radar-Diffusion python test/unit/test_sensor_aware_target.py` passed with 5 tests.
  - `conda run -n Radar-Diffusion python test/unit/test_shared_visibility_eval.py` passed with 2 tests.
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
  - Dataset audit smoke passed and wrote `test/result/comparison/dataset_protocol_audit_smoke`.
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
- Added `test/visualization/generate_interactive_inference_compare.py` to create self-contained interactive HTML overlays for radar/target/LDM/CD point clouds.
- Generated visualizations:
  - `test/result/ldm/visualization/mini_inference_compare/inference_compare_000068.html`
  - `test/result/ldm/visualization/mini_inference_compare/inference_compare_000150.html`
  - `test/result/ldm/visualization/mini_inference_compare/inference_compare_000253.html`
  - `test/result/ldm/visualization/mini_inference_compare/inference_compare_000386.html`
  - `test/result/ldm/visualization/mini_inference_compare/inference_compare_000478.html`
  - `test/result/ldm/visualization/mini_inference_compare/inference_compare_000488.html`
- Verification:
  - `conda run -n Radar-Diffusion python -m py_compile test/visualization/generate_interactive_inference_compare.py` passed.
  - `conda run -n Radar-Diffusion python test/visualization/generate_interactive_inference_compare.py --frames 000068,000150,000253,000386,000478,000488 --output_dir test/result/ldm/visualization/mini_inference_compare` passed.

## 2026-06-22 Sensor-Aware Mini Quality Correction

- Started Phase 6 after completing the new 500-frame sensor-aware mini train/inference run.
- Confirmed new checkpoints were used and all 500 LDM/CD frames produced voxel and uncertainty outputs.
- Diagnosed three linked issues: fixed threshold over-density, low near-range precision, and non-informative deterministic uncertainty.
- Selected implementation order: task-aware threshold calibration, per-frame metric reporting fix, then learnable uncertainty with Gaussian NLL.
- Completed broad saved-output count sweeps for LDM/CD across thresholds `0.1-0.9`.
- Evidence supports threshold calibration before retraining: density is near target at LDM `0.6` and CD `0.7`.
- Added task-region voxel Precision/Recall/F1/IoU to `sweep_occ_threshold.py` and JSON threshold recommendations.
- Fixed inference so task metric values are written into each frame row as well as the summary row.
- `test/unit/test_formal_task_metrics.py` now has 3 passing tests; modified scripts compile successfully.
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
- Verified the dataset near-field crop test by directly running `test/unit/test_dataset_protocol_metadata.py -v`; the earlier `python -m unittest test...` form failed only because `test/` is not a Python package.
- Propagated configurable grid protocol through:
  - `diffusion_consistency_radar/scripts/unified_train.py`
  - `diffusion_consistency_radar/scripts/cd_train_optimized.py`
  - `diffusion_consistency_radar/scripts/inference.py`
  - `test/mini-test/train_minimal.sh`
  - `test/mini-test/inference_minimal.sh`
- Added inference-side voxel crop coverage to `test/unit/test_dataset_protocol_metadata.py`.
- Added `diffusion_consistency_radar/scripts/diagnose_vae_reconstruction.py` for VAE reconstruction upper-bound diagnostics and `test/unit/test_vae_reconstruction_diagnostic.py` for metric unit tests.
- Smoke-ran the VAE diagnostic for 1 loop3 frame using `test/mini-test/train_results_mini_calibrated/vae/vae_best.pt`; outputs were saved under `test/result/vae/reconstruction/vae_reconstruction_smoke`.
- Verification completed:
  - `conda run -n Radar-Diffusion python test/unit/test_dataset_protocol_metadata.py -v` passed with 6 tests.
  - `conda run -n Radar-Diffusion python test/unit/test_vae_reconstruction_diagnostic.py -v` passed with 3 tests.
  - `conda run -n Radar-Diffusion python test/unit/test_sensor_aware_target.py -v` passed with 4 tests.
  - `conda run -n Radar-Diffusion python test/unit/test_airborne_multimodal_refactor.py -v` passed with 9 tests.
  - `conda run -n Radar-Diffusion python test/unit/test_multimodal_inference_interface.py` passed with 2 tests.
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
  - `test/unit/test_vae_reconstruction_diagnostic.py`: 3 tests passed.
  - `test/unit/test_multimodal_inference_interface.py`: 2 tests passed.
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
  `test/result/ldm/evaluation/ldm_near40_500_v2_threshold_validated/raw_lidar_visuals/` and the
  fixed-seed validation-order set in
  `test/result/ldm/evaluation/ldm_near40_500_v2_threshold_validated/raw_lidar_visuals_random_validation/`.
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
- Added `test/unit/test_vertical_structure_metrics.py`; focused metric regression passed
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
  - `conda run -n Radar-Diffusion python test/unit/test_ldm_vertical_structure_loss.py -v` passed with 11 tests.
  - `conda run -n Radar-Diffusion python -m py_compile diffusion_consistency_radar/scripts/unified_train.py test/unit/test_ldm_vertical_structure_loss.py` passed.
  - `git diff --check -- diffusion_consistency_radar/scripts/unified_train.py test/unit/test_ldm_vertical_structure_loss.py` passed.
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

- Ran the planned short LDM smoke into `test/result/ldm/vertical_structure/ldm_vertical_structure_smoke/`.
- The first sandbox run failed before training with the known OpenMPI socket restriction.
- The outside-sandbox 1-sample run passed OpenMPI but hit the project split guard: `deterministic_split_indices` requires at least 2 samples.
- Re-ran the same smoke with 2 samples, 1 epoch, 0 workers, `MINI_LDM_HEIGHT_WEIGHT=0.02`, and `MINI_LDM_CONTINUITY_WEIGHT=0.02`; it completed successfully in about 0.9s of training time.
- Smoke metrics row: total loss `0.475098`, latent `0.357750`, decoded occupancy `0.001069`, height distribution `0.323127`, vertical continuity `0.069169`, uncertainty `2.168658`.
- Confirmed `ldm_best.pt` stores `ldm_loss_config` with height/continuity weights `0.02/0.02`.
- Additional verification passed:
  - `test/unit/test_ldm_vertical_structure_loss.py -v`: 18 tests.
  - `test/unit/test_mini_train_script.py -v`: 10 tests.
  - `test/unit/test_multimodal_inference_interface.py -v`: 13 tests.
  - focused `py_compile` and full `git diff --check`.
- No formal multi-epoch LDM retraining was run; that remains an explicit experiment step after this code change.

## 2026-07-07 Final Review

- Final reviewer approved the complete Phase 10 LDM vertical-structure supervision increment.
- Reviewer confirmed the LDM train path, one-decode auxiliary losses, CSV migration, checkpoint `ldm_loss_config`, and smoke output are consistent.
- Reviewer also confirmed the existing 500-frame VAE checkpoint carries the expected sigmoid occupancy semantics, latent_dim=8, BCE+Dice protocol, and near-field grid metadata, so the next step can be 500-frame LDM retraining followed by vertical metrics evaluation.

## 2026-07-08 LDM Vertical Evaluation Bugfix

- Added a focused regression test showing that a prediction directory containing
  `000000_voxel.npy`, `000000_uncertainty.npy`, and `000000_pcl.npy` must evaluate only
  `000000_voxel.npy`.
- Confirmed the new test fails before the implementation change because
  `000000_uncertainty.npy` is included.
- Updated `iter_prediction_files()` to return only basenames ending with `_voxel.npy`.
- Verification will run the focused evaluation tests, py_compile, and targeted
  `git diff --check`.

## 2026-07-08 LDM Prediction Layout Bugfix

- Added a focused regression test for channel-last prediction shape `(Z,X,Y,C)` with
  `Z == 4`, which previously failed by returning shape `(2,2,4)`.
- Updated `load_prediction_occupancy()` to prefer channel-last parsing when the last axis
  is channel-sized and middle spatial axes are not channel-sized.
- Verification will run the focused evaluation tests, py_compile, and targeted
  `git diff --check`.

## 2026-07-08 LDM Vertical Evaluation Completion

- Extended `test/evaluation/ldm/evaluate_ldm_vertical_structure.py` so target loading follows the same
  source-range crop, model-range crop, and target-size resize protocol used by training
  and inference.
- Added a regression test proving a larger source target is cropped to the model range
  before vertical metrics are computed.
- Verification passed:
  - `conda run -n Radar-Diffusion python test/unit/test_ldm_vertical_structure_eval.py -v`
    passed with 8 tests.
  - `conda run -n Radar-Diffusion python -m py_compile
    test/evaluation/ldm/evaluate_ldm_vertical_structure.py test/unit/test_ldm_vertical_structure_eval.py` passed.
  - `git diff --check -- test/evaluation/ldm/evaluate_ldm_vertical_structure.py
    test/unit/test_ldm_vertical_structure_eval.py` passed.
- Ran the 500-frame vertical report for
  `test/result/ldm/vertical_structure/ldm_near40_500_vertical_v1/loop3_ldm_eval` into
  `test/result/ldm/vertical_structure/ldm_near40_500_vertical_v1/vertical_structure_eval`.
- Summary metrics: height coverage `0.3853`, top-height `0.1788`, vertical connectivity
  `0.3956`, trunk recall `0.4190`. This supports holding CD and continuing LDM vertical
  recovery.

## 2026-07-08 LDM Vertical Experiment Runner

- Added `test/mini-test/run_ldm_vertical_experiment.sh` as a directly runnable v2
  experiment wrapper.
- Default flow:
  - reuse/copy an existing `vae_best.pt` into `test/result/ldm/vertical_structure/ldm_near40_500_vertical_v2`;
  - train LDM with 500 samples, 10 epochs, height/continuity weights `0.05/0.02`;
  - run 500-frame LDM inference on `loop3` at threshold `0.05`;
  - run saved-output vertical structure evaluation;
  - generate raw-LiDAR interactive HTML overlays.
- Verification completed:
  - `bash -n test/mini-test/run_ldm_vertical_experiment.sh` passed.
  - `git diff --check -- test/mini-test/run_ldm_vertical_experiment.sh` passed.
- Long training was not executed automatically; the script is ready for the user to run
  when GPU time is available.

## 2026-07-08 LDM Vertical v2 Analysis

- Read the completed v2 outputs:
  - `test/result/ldm/vertical_structure/ldm_near40_500_vertical_v2/vertical_structure_eval/vertical_structure_report.md`
  - `test/result/ldm/vertical_structure/ldm_near40_500_vertical_v2/vertical_structure_eval/vertical_structure_summary.csv`
  - `test/result/ldm/vertical_structure/ldm_near40_500_vertical_v2/loop3_ldm_eval/inference_metrics.csv`
  - `test/result/ldm/vertical_structure/ldm_near40_500_vertical_v2/ldm/metrics.csv`
- Confirmed 10 raw-LiDAR HTML comparison files were generated under
  `test/result/ldm/vertical_structure/ldm_near40_500_vertical_v2/raw_lidar_visuals`.
- Compared v2 against v1:
  - height coverage improved by `+0.0403`;
  - top-height improved by `+0.0350`;
  - vertical connectivity improved by `+0.0436`;
  - trunk recall dropped by `-0.0093`.
- Formal task metrics show the cost of that improvement:
  - near recall improved from `0.7440` to `0.8173`;
  - near precision dropped from `0.3990` to `0.3363`;
  - near BEV IoU dropped from `0.3365` to `0.3058`;
  - average predicted point count increased from `1316.44` to `1671.66`.
- Current gate decision: do not start CD yet. The next LDM experiment should improve
  vertical height without further over-densifying the scene.

## 2026-07-08 LDM Vertical v3 Analysis

- Read the completed v3 outputs:
  - `test/result/ldm/vertical_structure/ldm_near40_500_vertical_v3/vertical_structure_eval/vertical_structure_report.md`
  - `test/result/ldm/vertical_structure/ldm_near40_500_vertical_v3/vertical_structure_eval/vertical_structure_summary.csv`
  - `test/result/ldm/vertical_structure/ldm_near40_500_vertical_v3/loop3_ldm_eval/inference_metrics.csv`
  - `test/result/ldm/vertical_structure/ldm_near40_500_vertical_v3/ldm/metrics.csv`
- Confirmed 10 raw-LiDAR HTML comparison files were generated.
- Compared v3 against v1/v2:
  - v3 improves height coverage and vertical connectivity slightly over v1, but top-height
    is nearly unchanged and trunk recall is flat.
  - v3 is worse than v2 for height coverage, top-height, and vertical connectivity.
  - v3 has fewer predicted points than v2, but still more than v1, and its Chamfer/NN
    metrics are worse.
- Current gate decision remains: do not start CD. Next work should add density/precision
  control instead of only tuning height and continuity weights.

## 2026-07-08 LDM Density Regularizer Implementation

- Used `planning-with-files-zh` and `subagent-driven-development` to implement the next
  Phase 10 step.
- Added `decoded_density_precision_loss()` in `unified_train.py`.
- Connected `decoded_density_weight` through:
  - LDM component names and CSV header;
  - `compute_ldm_loss_components()`;
  - `OptimizedLDMTrainer` config parsing;
  - checkpoint `ldm_loss_config`;
  - tqdm component display;
  - `test/mini-test/train_minimal.sh`;
  - `test/mini-test/run_ldm_vertical_experiment.sh`.
- Subagent specification review: PASS.
- Subagent code-quality review: PASS.
- Added tests for over-dense prediction penalty, empty target finite gradients, weight-zero
  no-op behavior, weighted total-loss contribution, one-decode sharing across all decoded
  auxiliaries, and mini YAML propagation.
- Verification passed:
  - `conda run -n Radar-Diffusion python test/unit/test_ldm_vertical_structure_loss.py -v`
    passed with 24 tests.
  - `conda run -n Radar-Diffusion python test/unit/test_mini_train_script.py -v`
    passed with 10 tests.
  - `conda run -n Radar-Diffusion python -m py_compile
    diffusion_consistency_radar/scripts/unified_train.py
    test/unit/test_ldm_vertical_structure_loss.py test/unit/test_mini_train_script.py` passed.
  - `bash -n test/mini-test/train_minimal.sh &&
    bash -n test/mini-test/run_ldm_vertical_experiment.sh` passed.
  - targeted `git diff --check` passed.
- Long training was not executed automatically. Next action is a v4 500-frame LDM
  experiment with a small nonzero `MINI_LDM_DENSITY_WEIGHT`.

## 2026-07-08 LDM Vertical v4 Analysis

- Read v4 outputs from `test/result/ldm/vertical_structure/ldm_near40_500_vertical_v4`.
- Confirmed 10 raw-LiDAR HTML files exist under `raw_lidar_visuals`.
- Compared v4 against v1/v2/v3:
  - v4 has the best top-height recall (`0.2219`);
  - v4 reduces predicted occupancy versus v2 (`835832 -> 761552`);
  - v4 still has worse precision/BEV IoU than v1 and worse Chamfer/BEV than v2;
  - v4 loses trunk recall (`0.3751`), which is the clearest new regression.
- Ran threshold sweep on v4 saved outputs. Recommendation:
  - threshold `0.4` by validation task BEV F1;
  - pred/target ratio about `2.31`;
  - task BEV F1 about `0.4218`.
- Ran vertical structure evaluation at threshold `0.4` into
  `test/result/ldm/vertical_structure/ldm_near40_500_vertical_v4/vertical_structure_eval_thr040`.
  Result: height coverage `0.3290`, top-height `0.2152`, vertical connectivity `0.3416`,
  trunk recall `0.3110`.
- Current gate decision remains: do not start CD. The density regularizer works in the
  expected direction, but the current weight/form suppresses trunk/lower-column occupancy
  too much.

## 2026-07-08 Voxel-Column Raw-LiDAR Visualization

- Used `planning-with-files-zh` and `subagent-driven-development` to continue Phase 10.
- Tried optional predicted-voxel column rendering to test whether the apparent tree
  layering was only a visualization artifact.
- Visual review confirmed the structural mismatch remains against raw LiDAR, so the
  feature was reverted and the current script is back to the original raw-LiDAR point
  cloud comparison.
- Current verification after reverting the visualization path:
  - `conda run -n Radar-Diffusion python test/unit/test_interactive_inference_compare.py -v`
    passed with 3 tests.
  - `conda run -n Radar-Diffusion python -m py_compile
    test/visualization/generate_interactive_inference_compare.py test/unit/test_interactive_inference_compare.py`
    passed.
  - `git diff --check -- test/visualization/generate_interactive_inference_compare.py
    test/unit/test_interactive_inference_compare.py` passed.
- The 10 HTML files under
  `test/result/ldm/vertical_structure/ldm_near40_500_vertical_v4/raw_lidar_voxel_column_visuals` are historical
  outputs from the reverted experiment only; they do not represent the current script's
  default capability.

## 2026-07-08 Z=64 VAE Upper-Bound Runner

- Added `test/mini-test/run_vae_z64_upper_bound.sh` with a Chinese file header and
  executable bit.
- The runner defaults to:
  - `EXP_DIR=test/result/vae/reconstruction/vae_near40_500_z64_upper_bound`
  - `MINI_TARGET_SIZE=64,128,128`
  - `MINI_MODEL_PC_RANGE=0,-20,-6,40,20,10`
  - `MINI_SOURCE_PC_RANGE=0,-20,-6,120,20,10`
  - `SAMPLES_PER_SCENE=500`, `MINI_VAE_EPOCHS=10`, `MINI_VAE_CONFIG_TYPE=lightweight`,
    `MINI_VAE_LATENT_DIM=8`, `MINI_VAE_OCC_LOSS=bce_dice`, `MINI_NUM_WORKERS=2`
- Added static coverage in `test/unit/test_mini_train_script.py` for the new runner defaults,
  env exports, train-before-diagnose order, checkpoint guard, and diagnostic grid args.
- Verified only syntax/static checks; no long VAE training was launched.

## 2026-07-09 Z=64 LDM Inheritance Analysis and Loss Update

- Used `planning-with-files-zh` and `subagent-driven-development`.
- Read Z=64 VAE upper-bound report, Z=64 LDM threshold recommendation, LDM inference
  summary, LDM training metrics, and the relevant LDM loss/script code paths.
- Conclusion: VAE upper bound is good, but current LDM does not inherit it. VAE IoU is
  `0.6027`, while LDM selected-threshold voxel IoU is only `0.0916`, and the selected
  threshold is `0.99` with pred/target ratio still `2.55x`.
- Updated `diffusion_consistency_radar/scripts/unified_train.py`:
  - `decoded_density_precision_loss()` now penalizes predictions in target-empty
    `(X,Y)` columns rather than every target-empty voxel;
  - this is intended to suppress background occupancy while protecting tree/trunk columns.
  - sigmoid density loss now uses empty-class `softplus(logit)` so high-confidence
    background false positives keep strong gradients.
- Updated `test/mini-test/train_minimal.sh`:
  - added `MINI_LDM_DECODED_WEIGHT`, `MINI_LDM_DECODED_FP_WEIGHT`,
    `MINI_LDM_DECODED_MASS_WEIGHT`, and `MINI_LDM_UNCERTAINTY_WEIGHT`.
- Updated `test/mini-test/run_ldm_vertical_experiment.sh` to pass those same controls.
- Added `test/mini-test/run_ldm_z64_v5_experiment.sh`, a Z=64 v5 wrapper that uses the
  existing Z=64 VAE checkpoint, sets `MINI_TARGET_SIZE=64,128,128`, disables
  uncertainty NLL, and enables a small empty-column density weight.
- Quality-review fixes:
  - generic vertical runner now resolves relative output/checkpoint paths against the
    repository root and calls diagnostic helpers by absolute path;
  - Z=64 v5 wrapper refuses to overwrite an existing default experiment unless
    `ALLOW_OVERWRITE=1`;
  - `train_minimal.sh` now preserves externally supplied `CUDA_VISIBLE_DEVICES`.
- Updated tests:
  - added empty-column density behavior coverage;
  - added coverage that extra Z occupancy inside a target-occupied column does not
    trigger the density penalty;
  - added coverage for high-logit sigmoid false-positive gradients, LDM runner path
    normalization, overwrite protection, and CUDA device propagation;
  - extended mini-script static contract coverage for decoded and uncertainty weights.
- Verification:
  - sandboxed test attempt failed before code execution due to the known OpenMPI socket
    restriction;
  - reran outside the sandbox successfully:
    `conda run -n Radar-Diffusion python test/unit/test_ldm_vertical_structure_loss.py -v`
    passed with 27 tests;
    `conda run -n Radar-Diffusion python test/unit/test_mini_train_script.py -v`
    passed with 25 tests;
    `py_compile`, `bash -n`, and targeted `git diff --check` passed.
- Long training was not started automatically. Next action is a Z=64 LDM v5 experiment
  using the existing Z=64 VAE checkpoint, uncertainty loss disabled, and a small
  empty-column density weight.

## 2026-07-09 Z=64 LDM v5 Threshold Diagnosis and v6 Implementation

- Read the completed v5 threshold sweep and vertical evaluations:
  - `occ_threshold_recommendation.json` recommends threshold `0.99`;
  - vertical evaluations at `0.5/0.7/0.85/0.95` show top-height recall remains stuck
    around `0.095-0.099`.
- Conclusion: lowering threshold helps height coverage and trunk recall but does not
  recover top-height, so the top-structure failure is a generation/supervision issue.
- Updated `diffusion_consistency_radar/scripts/unified_train.py`:
  - added `top_height_loss` inside `decoded_vertical_structure_losses()`;
  - connected `decoded_top_height_weight` through LDM component logging, total loss,
    checkpoint metadata, and tqdm display.
- Updated mini scripts:
  - `train_minimal.sh` now accepts `MINI_LDM_TOP_WEIGHT`;
  - `run_ldm_vertical_experiment.sh` forwards and prints top-height weight;
  - added `test/mini-test/run_ldm_z64_v6_top_experiment.sh` with Z=64 near40 defaults,
    top-height loss enabled, lower density weight, and overwrite protection.
- Updated tests:
  - added direct top-height loss behavior coverage;
  - expanded mini-script contract tests for `MINI_LDM_TOP_WEIGHT`;
  - added static coverage for the v6 runner defaults.
- Verification passed outside the sandbox:
  - `conda run -n Radar-Diffusion python test/unit/test_ldm_vertical_structure_loss.py -v`
    passed with 28 tests;
  - `conda run -n Radar-Diffusion python test/unit/test_mini_train_script.py -v` passed
    with 29 tests;
  - `py_compile`, `bash -n`, and targeted `git diff --check` passed.
- Long training was not started automatically. Next action is to run the v6 runner and
  compare top-height recall against v5.

## 2026-07-09 16:32 CST IR Supervision v7 Preflight Implementation

- Used `planning-with-files-zh` and `subagent-driven-development`.
- Ran two read-only subagent checks:
  - dataset/audit branch confirmed that `CalibrationProvider` previously accepted
    `calib_radar_to_livox.txt` as non-mock IR calibration and that audit could overstate
    training IR coverage;
  - multimodal/training branch confirmed that `ir_gate` only used radar condition plus
    confidence and that `DualModalityProjectionLayer` already computed a mask internally.
- Updated `diffusion_consistency_radar/cm/dataset_loader.py`:
  - added `CalibrationProvider.load_with_metadata()`;
  - only real `calib_radar_to_thermal.txt` is now non-mock for IR projection;
  - LiDAR calibration availability is recorded but not treated as thermal calibration;
  - dataset meta carries calibration source/fallback and sync-compensation metadata.
- Updated `diffusion_consistency_radar/scripts/audit_dataset_protocol.py`:
  - added dataset-loader IR coverage, compatible IR coverage, mock IR/calib ratios,
    calibration source/fallback fields, and estimated IR frustum voxel ratio.
- Updated `diffusion_consistency_radar/cm/multimodal_fusion.py`:
  - projection can return `frustum_mask`;
  - `CompleteDualModalityPerceptionNet` caches `last_ir_frustum_mask`;
  - `ir_gate` now uses IR features in addition to radar features and confidence.
- Updated `diffusion_consistency_radar/scripts/unified_train.py`:
  - added optional IR-frustum occupancy/top losses;
  - added LDM CSV/log fields for `mock_ir_ratio`, `mock_calib_ratio`, and
    `ir_frustum_voxel_ratio`;
  - added warnings when mock IR/calib dominate training.
- Updated mini scripts:
  - `train_minimal.sh` now accepts `MINI_LDM_IR_FRUSTUM_OCC_WEIGHT` and
    `MINI_LDM_IR_FRUSTUM_TOP_WEIGHT`;
  - added guarded runner `test/mini-test/run_ldm_z64_v7_ir_experiment.sh`.
- Added diagnostic utility `test/ablation/diagnose_ir_condition_ablation.py` and unit test
  `test/unit/test_ir_condition_ablation.py`.
- Verification:
  - `conda run -n Radar-Diffusion python test/unit/test_dataset_protocol_metadata.py -v`
    passed with 9 tests;
  - `conda run -n Radar-Diffusion python test/unit/test_airborne_multimodal_refactor.py -v`
    passed with 9 tests;
  - `conda run -n Radar-Diffusion python test/unit/test_ldm_vertical_structure_loss.py -v`
    passed with 30 tests;
  - `conda run -n Radar-Diffusion python test/unit/test_ir_condition_ablation.py -v`
    passed with 3 tests;
  - `conda run -n Radar-Diffusion python test/unit/test_mini_train_script.py -v`
    passed with 32 tests;
  - `py_compile`, `bash -n`, and `git diff --check` passed.
- Long v7 training was not started automatically. Next action is to run dataset audit,
  run one IR ablation diagnostic on the current v6/v7 candidate checkpoint, then launch
  the guarded v7 runner if IR/calib coverage is credible.

## 2026-07-10 IR Supervision v7 Review Fixes

- Continued the v7 preflight task after review.
- Fixed review blockers:
  - `inference.py` import indentation was corrected after the new migration import;
  - `migrate_ir_gate_state_dict()` was added and connected to strict inference loading
    and LDM resume loading, so old multimodal checkpoints can migrate the previous
    `ir_gate.0.weight` shape;
  - `diagnose_ir_condition_ablation.py` now preserves bool/int/float mock flags as
    batch tensors;
  - dataset audit now computes mock-calibration frustum ratio using the same fallback
    geometry and sync compensation as the dataset loader;
  - `run_ldm_vertical_experiment.sh` now prints and forwards IR-frustum weights.
- Added/updated tests:
  - strict multimodal inference load migrates legacy IR gate weights;
  - IR ablation preserves Python bool mock flags;
  - mini vertical runner logs and forwards IR-frustum weights.
- Verification passed:
  - `conda run -n Radar-Diffusion python test/unit/test_multimodal_inference_interface.py -v`
    passed with 14 tests;
  - `conda run -n Radar-Diffusion python test/unit/test_ir_condition_ablation.py -v`
    passed with 4 tests;
  - `conda run -n Radar-Diffusion python test/unit/test_dataset_protocol_metadata.py -v`
    passed with 9 tests;
  - `conda run -n Radar-Diffusion python test/unit/test_mini_train_script.py -v`
    passed with 33 tests;
  - `conda run -n Radar-Diffusion python test/unit/test_airborne_multimodal_refactor.py -v`
    passed with 9 tests;
  - `conda run -n Radar-Diffusion python test/unit/test_ldm_vertical_structure_loss.py -v`
    passed with 30 tests;
  - targeted `py_compile`, `bash -n`, and `git diff --check` passed.
- No long training was launched.

## 2026-07-10 Z64 LDM v7 Result Analysis

- Confirmed `test/result/ldm/ablation/ldm_near40_500_z64_v7_ir` contains the 10-epoch LDM checkpoint,
  500 loop3 predictions, inference metrics, vertical report, and raw-LiDAR comparisons.
- Compared v7 with v6 at threshold 0.85. v7 improved height coverage, connectivity, trunk
  recall, near-field task metrics, and Chamfer, but reduced top-height recall and increased
  predicted density.
- Ran formal threshold sweeps on the saved v6 and v7 outputs with the same Z64 near-field
  crop/resize protocol. Both recommend threshold 0.99 by validation task BEV F1.
- Re-ran vertical evaluation at threshold 0.99:
  - v7: `height=0.6364`, `top=0.0628`, `connectivity=0.6410`, `trunk=0.7169`;
  - v6: `height=0.2167`, `top=0.0659`, `connectivity=0.2226`, `trunk=0.3061`.
- Ran post-v7 IR ablation on train samples 0, 100, and 300 with the same seed and 20-step
  Euler sampling. Real IR consistently produces substantially more occupancy than zero/mock IR.
- A first planning-file patch missed the exact findings heading context and was rejected by
  `apply_patch`; the corrected patch used the actual file tail and preserved all existing text.
- No model code or long training was started. CD remains held until target-aware IR ablation
  proves that the extra occupancy improves LiDAR-aligned structure.

## 2026-07-10 Target-Aware IR Ablation Script

- Added TDD coverage for perfect overlap, top-height overshoot, micro aggregation,
  deterministic multi-frame sampling, and the v7 one-click runner contract.
- Verified the RED phase: five tests failed because the new metric helpers and runner did not
  exist.
- Extended `test/ablation/diagnose_ir_condition_ablation.py` with multi-frame target-aware evaluation
  while preserving the original single-frame comparison files.
- Added executable `test/mini-test/run_ldm_z64_v7_target_ablation.sh` with the approved v7
  Z64/near-field/validation defaults.
- Focused unit test passed with 9 tests.
- The first 1-frame/1-step smoke failed inside the sandbox due to the known OpenMPI socket
  restriction. The same command completed outside the restricted sandbox and wrote legacy,
  per-frame, summary, JSON, and Markdown reports under `/tmp/radar_v7_target_ablation_smoke`.
- The full 32-frame diagnostic was intentionally not auto-started; it is the next user-run
  experiment and does not perform training.
- Final verification: 9 focused tests passed; Python compilation, shell syntax, executable
  mode, and `git diff --check` passed. A 1-frame/1-step end-to-end run also completed.
- Read-only review found no metric or micro-aggregation correctness issue. It noted that the
  Python utility retains legacy `train`/single-frame defaults; this is intentional backward
  compatibility, while the new one-click wrapper owns the validation/32-frame defaults.

## 2026-07-10 32-Frame IR Ablation Analysis

- Read the completed report and both per-frame/summary CSV files under
  `test/result/ldm/ablation/ldm_near40_500_z64_v7_ir/ir_target_ablation_32`.
- Confirmed real IR significantly improves voxel/BEV overlap, recall, height coverage,
  vertical connectivity, and trunk recall over zero/mock IR across the 32 validation frames.
- Confirmed the remaining failure mode: count ratio remains `9.44`, voxel precision is only
  `0.0704`, and top-height recall is only `0.0737`.
- Two independent read-only reviews were requested. The first pair did not return before the
  analysis deadline and was stopped; a focused replacement pair completed the statistical and
  architecture checks.
- Code inspection found a direct loss/metric mismatch: target-top positive supervision does
  not penalize predictions above the target top, while empty-column density control cannot
  suppress this error inside occupied columns.
- Updated the Phase 10 plan with three v8 tasks: top-overshoot loss, balanced IR-frustum
  positive/negative supervision, and a guarded v7/v8 comparison runner.
- No model code was changed and no v8/CD training was started.

## 2026-07-11 LDM v8 Balanced-Supervision Implementation

- Implemented `decoded_top_overshoot_weight` and the corresponding above-target loss in
  `diffusion_consistency_radar/scripts/unified_train.py`.
- Implemented `decoded_ir_frustum_negative_weight` and balanced the existing IR positive-only
  supervision with visible target-negative supervision.
- Added strict shared validation for decoded/target dimensions, channels, B/Z/X/Y grids,
  occupancy activation, and IR frustum mask batch/channel semantics.
- Propagated both weights through mini config generation, the generic vertical experiment runner,
  LDM metric CSV fields, trainer state, and checkpoint loss metadata.
- Added `test/mini-test/run_ldm_z64_v8_balanced_experiment.sh` with Z64, 500-frame, 10-epoch
  defaults. It does not automatically run ablation or CD.
- Hardened experiment scripts after review: validate mode before destructive setup, restrict
  scratch deletion paths, normalize paths, reject symlink/non-empty outputs, keep CUDA variables
  consistent, use experiment-local scratch/config files, and serialize identical runs with an
  atomic `${EXP_DIR}.lock` owned only by the generic runner.
- Completed per-task specification review, iterative code-quality review, and final integrated
  review. All Critical/Important findings were closed.
- Final short verification passed outside the socket-restricted sandbox:
  - `test/unit/test_ldm_vertical_structure_loss.py`: 43 tests;
  - `test/unit/test_mini_train_script.py`: 59 tests;
  - `test/unit/test_ir_condition_ablation.py`: 9 tests;
  - Python compilation, shell syntax checks, and `git diff --check` passed.
- No long v8 training or CD training was started. Next action is the explicit user-run v8
  experiment, followed by threshold calibration, 32-frame target-aware IR ablation, 500-frame
  vertical/task evaluation, and raw-LiDAR visualization comparison against v7.

## 2026-07-11 Z64 LDM v8 Full Evaluation

- Verified the completed v8 experiment: 10 epoch checkpoints, `ldm_best.pt`, 500 voxel predictions,
  inference metrics, and vertical-structure outputs are present.
- Ran a coarse and fine validation threshold sweep. The final BEV-F1 recommendation is `0.99995`;
  custom full-500 task reports were saved for v8@0.99995, v8@0.99, and v7@0.99.
- Re-ran the v8 500-frame vertical evaluation at `0.99995` and retained the existing same-threshold
  v8@0.99/v7@0.99 reports for model-effect analysis.
- Ran the full 32-frame, 20-step, target-aware real/zero/mock IR ablation for v8 at `0.99995`.
  Reports were written under `ir_target_ablation_32_thr099995/`.
- Computed paired 500-frame and 32-frame win rates/Wilcoxon tests from the saved CSV files using
  Python CSV + NumPy/SciPy. No training or checkpoint modification was performed.
- Wrote `v7_v8_evaluation_report.md` with the complete comparison, CD HOLD decision, and v9 plan.
- Next work is a small v9 experiment matrix: add recall-constrained threshold selection, screen
  top-overshoot and IR-negative weights independently, then run one full 10-epoch winner only.

## 2026-07-12 Z64 LDM v9-A Full Evaluation

- Added and verified recall-constrained threshold selection; focused threshold tests (28) and
  mini-runner tests (73), compilation, shell syntax, and `git diff --check` passed before training.
- Completed the isolated 3-epoch v9-A/v9-B screens. v9-A won on every 32-frame structure metric,
  so only v9-A was trained for 10 epochs; no CD training was started.
- Completed v9-A 10-epoch training. Total loss decreased from `0.373599` to `0.136242`.
- Completed the automatic 32-frame real/zero/mock IR ablation at threshold `0.99`; real IR remained
  clearly target-aligned and outperformed both controls.
- Completed 500-frame, 40-step Heun inference under
  `test/result/ldm/ablation/ldm_near40_500_z64_v9a_top_full/loop3_ldm_eval`.
- Completed validation threshold calibration. Unconstrained BEV-F1 selected `0.98`; recall
  constraints selected `0.70`, but its validation count ratio is `12.48`.
- Completed full-500 task and vertical evaluation at `0.70`, `0.98`, and `0.99` and wrote
  `v8_v9_evaluation_report.md`.
- Final decision: v9-A fails the joint density/BEV/top/trunk teacher gate, does not replace v8,
  and CD remains HOLD. The next implementation task is validation task/structure checkpoint
  selection over saved epochs.
- The first attempt to print script help inside the restricted sandbox hit the known OpenMPI
  socket denial. Source inspection supplied the CLI contract, and metric commands ran outside
  the restricted sandbox without changing experiment outputs.

## 2026-07-12 LDM Validation Checkpoint Selection

- Implemented `--variants real` and strict sample-count support in the target-aware IR diagnostic
  while preserving the legacy real/zero/mock default.
- Added `test/evaluation/ldm/select_ldm_checkpoint.py` and the guarded
  `test/mini-test/run_ldm_z64_checkpoint_selection.sh` runner.
- Added fixed-protocol, checkpoint/VAE hash, dataset manifest, source-stability, fresh-output,
  locking, path-boundary, finite-gate, and strict-JSON safeguards after two-stage review.
- TDD verification passed: 11 IR diagnostic tests and 10 checkpoint-selection tests, plus Python
  compilation, shell syntax, and `git diff --check`. Specification and quality reviews approved.
- Evaluated all 10 v9-A epoch checkpoints on the same 32 validation frames. No checkpoint passed
  every gate; epoch8 was selected as the most balanced candidate.
- Completed a new 500-frame, 40-step Heun loop3 inference for epoch8 and evaluated thresholds
  `0.80`, `0.98`, and `0.99`. Epoch8 improves over epoch10 but still fails the joint teacher gate.
- No model file was copied or replaced, no supervision/target/grid protocol changed, and CD was
  not started. Next task is the isolated column-balanced loss design and test phase.

## 2026-07-12 Column-Balanced Structure Loss Planning

- Inspected the decoded occupancy, vertical structure, top overshoot, empty-column density,
  IR-frustum positive/negative, trainer logging, mini config, and checkpoint-selection call chain.
- Compared three approaches: balanced column BCE, global BEV Dice/Focal, and an auxiliary BEV head.
  Selected balanced column BCE as the smallest checkpoint-compatible change with independently
  observable recall and precision components.
- Defined the logmeanexp column aggregation, positive/negative class-balanced losses, TDD cases,
  config/logging protocol, guarded v10 A/B screen, full evaluation gate, and raw-LiDAR visual gate.
- No training or model code was changed in this planning phase. CD remains HOLD.

## 2026-07-13 Column-Balanced Loss Task 1

- Implemented the standalone positive/negative column-existence loss using TDD.
- RED exposed the missing API and later contract gaps; GREEN completed with 59 structure-loss tests.
- Added exact logmeanexp formula checks, positive and negative gradient direction tests, raw and
  sigmoid numerical boundaries, class-balanced averaging, soft-target threshold, graph-zero,
  shape, empty-dimension, device, and temperature validation.
- Specification review and iterative code-quality review both approved the Task 1 implementation.
- Python compilation and `git diff --check` passed. No training or CD run was started.

## 2026-07-13 Column-Balanced Loss Task 2

- Connected the Task 1 helper to `compute_ldm_loss_components`, metrics, trainer config, progress
  output, train-epoch forwarding, and checkpoint loss metadata.
- Added TDD coverage for zero-weight compatibility, exact weighted totals, single decode, component
  headers, explicit/default config values, and invalid config rejection.
- Full focused structure-loss suite passed with 65 tests; compilation and diff checks passed.
- Specification and quality reviews approved the integration. No mini script or experiment was run.

## 2026-07-13 Column-Balanced Loss Tasks 3-4

- Wired `MINI_LDM_COLUMN_POSITIVE_WEIGHT`, `MINI_LDM_COLUMN_NEGATIVE_WEIGHT`, and
  `MINI_LDM_COLUMN_TEMPERATURE` through mini config generation and added the guarded
  `run_ldm_z64_v10_column_experiment.sh` A/B training-only runner.
- Completed 94 serial mini-runner tests. An earlier parallel review run had read shell scripts while
  they were still being edited and accidentally launched two real LDM jobs; both process trees were
  identified from their parent test processes and terminated without touching user experiments.
  A clean serial rerun left no `unified_train.py` process behind.
- The first smoke attempt inside the restricted sandbox failed at OpenMPI local-interface setup.
  The next attempt exposed a relative `PREPROCESSED_ROOT` path resolving from the subproject and
  loading zero samples. Re-running outside the socket-restricted sandbox with the absolute dataset
  root fixed both environmental issues.
- The final 2-frame/1-epoch Z64 smoke completed in about one second of training. It logged finite
  values (`loss=2.0402`, `column_positive_loss=7.4342`, `column_negative_loss=0.0020`), completed
  backward, and saved `ldm_best.pt` plus the epoch checkpoint under
  `/tmp/radar_v10_column_smoke6_20260713`.
- No 3-epoch A/B screen, 10-epoch winner, inference, or CD training was auto-started. The next
  experiment is v10-A followed by v10-B, then fixed validation checkpoint/threshold selection.

## 2026-07-13 Z64 LDM v10 A/B Screen Evaluation

- Verified complete A/B training artifacts: each experiment contains three epoch checkpoints,
  `ldm_best.pt`, metrics CSV, training log, VAE checkpoint, and generated config.
- Ran the guarded checkpoint-selection protocol over all six epoch checkpoints. Both groups select
  epoch3 and pass every validation gate; reports are stored in each experiment's
  `checkpoint_selection_32_thr099` directory.
- Selected v10-A epoch3 as the screen winner. A3 provides the strongest combined BEV overlap,
  obstacle-body recall, trunk retention, and vertical connectivity. B3's lower point-count ratio
  does not compensate for its substantial loss of useful structure.
- No checkpoint was copied or overwritten. No 500-frame loop3 inference, raw-LiDAR visualization,
  10-epoch winner training, or CD distillation was started during this evaluation.
- Next action: train only the v10-A weights for 10 epochs, retain every epoch checkpoint, run the
  same 32-frame checkpoint selector, and evaluate only the selected epoch on loop3 500 frames.

## 2026-07-13 Z64 LDM v10-A Full Training and Seed Fix

- Trained the v10-A winner for 10 epochs on 500 garden frames in a fresh output directory. All ten
  epoch checkpoints and `ldm_best.pt` were saved; total runtime was about 0.77 hours.
- Evaluated all ten checkpoints on the fixed 32-frame validation protocol. No checkpoint passed all
  gates, so loop3 500-frame inference, visualization, and CD were intentionally not started.
- Traced the screen/full mismatch to an incomplete seed protocol: only the split indices were
  deterministic, while model initialization, augmentation RNG, CUDA RNG, and train shuffle were not.
- Added `seed_training_run()` and connected its generator to the train DataLoader. The helper uses
  `data.training_seed`, falling back to `data.split_seed=42` for current mini configs.
- Verification passed: 20 VAE/checkpoint protocol tests, including Python/NumPy/Torch/shuffle
  reproduction and invalid-seed rejection; 65 LDM structure-loss tests; Python compilation and
  `git diff --check`.
- Next action is a cheap same-seed duplicate smoke. Do not start another 10-epoch run until its
  metrics and checkpoint behavior are reproducible.
- Completed two independent 2-frame/1-epoch runs with the corrected seed protocol. Every logged
  metric except wall-clock time matched exactly. Two final-output tensors differed only by CUDA
  roundoff (`max_abs=7.04e-08`); all other state tensors matched exactly.
- The short reproducibility gate passes. The next bounded experiment is one seeded 3-epoch v10-A
  run plus the fixed 32-frame selector. Another 10-epoch run remains deferred until that gate passes.

## 2026-07-13 Test Directory Organization

- 已完成源码分类移动：功能脚本进入评估、诊断、消融、可视化和 legacy 目录，测试脚本进入 `test/unit/`。
- 已完成用途明确结果目录的上级分类移动，未删除或重命名历史实验叶目录。
- 已同步 `.gitignore` 的源码目录例外、Python import、Shell 调用、结果默认路径、`test/README.md` 和 `test/result/INDEX.md`。
- mini-test、未知用途结果、锁目录、临时数据和历史报告中的原始路径暂不改写。
- 最小验证已完成：`compileall`、主要脚本 `--help`、公共模块导入、Shell 路径存在性和旧路径残留检查；未运行训练或完整推理。
- 后续验证确认 LDM evaluator 和 IR 消融脚本的 `--help` 均可在沙箱外通过；沙箱内 torch 导入仍受 OpenMPI socket 限制。
- 最后路径检查确认活动源码、Shell 和配置中不再残留旧 `Result/` 输出根目录或项目绝对路径。

## 2026-07-13 Seeded v10-A Three-Epoch Recheck

- Completed the fresh seeded three-epoch v10-A run and retained all epoch checkpoints and metrics
  under `test/result/ldm/ablation/ldm_near40_500_z64_v10a_seeded_recheck`.
- Ran the fixed 32-frame real-IR validation selector for epochs 1-3 with 20 Euler steps, seed 42,
  occupancy threshold `0.99`, and the unchanged Z64 crop/resize protocol.
- No checkpoint passed all five gates. Epoch1 was selected by maximum worst normalized gate
  satisfaction, but passed only 2/5 gates; epochs 2-3 were more conservative and lost additional
  recall, trunk, and connectivity.
- Another 10-epoch run, loop3 500-frame inference, visualization acceptance, and CD remain on hold.
  Next work is a small seeded loss-weight calibration screen followed by the identical 32-frame
  selector.

## 2026-07-13 Seeded v10 C/D Calibration Preparation

- Added RED tests for isolated C/D column-weight variants, then extended the guarded v10 runner.
- C is `positive=0.03, negative=0.01`; D is `positive=0.02, negative=0.005`. Both retain the same
  VAE, 500-frame garden dataset, seed 42, Z64 grid, three epochs, and every non-column loss weight.
- Corrected one stale training-only assertion that confused the reorganized result directory name
  `ablation` with an executed ablation command; the runner itself did not execute ablation.
- Verification passed: all 95 mini-runner tests and `bash -n` for the v10 runner.
- The first attempted focused unittest import failed because `test/` is not a Python package; the
  established direct-file test command was used instead and passed.

## 2026-07-13 Seeded v10 C/D Training and Gate Evaluation

- Closed the experiment-protocol review gap by explicitly fixing dataset/calibration roots, batch
  size, workers, gradient accumulation, augmentation, train split, VAE protocol, Z64 ranges, seed,
  fresh scratch/config, and training-only behavior in the shared runner.
- Re-ran the complete mini-runner suite after the protocol fix: 96 tests passed; `bash -n` passed.
- Completed C and D as sequential GPU jobs. Both produced non-empty best checkpoints and all three
  epoch checkpoints without NaN, CUDA, or checkpoint failures.
- Ran the fixed 32-frame selector over all six checkpoints. Neither experiment passed all gates;
  C selected epoch3 and D selected epoch1, each passing only 2/5 gates.
- Held the 10-epoch continuation, loop3 500-frame inference, raw-LiDAR visualization acceptance,
  and CD. The result shows that density is controlled but basic obstacle-column recall and vertical
  structure remain insufficient.

## 2026-07-13 v11 Column Curriculum Design

- Confirmed the three-epoch curriculum route with the user and wrote the design specification at
  `docs/superpowers/specs/2026-07-13-ldm-column-curriculum-design.md`.
- Compared epoch-wise, step-wise, and metric-feedback schedules; selected epoch-wise linear
  interpolation because it is deterministic, resume-safe, and exact for the three-epoch screen.
- Completed placeholder and consistency review; the specification contains explicit config,
  logging, checkpoint, testing, and fixed 32-frame acceptance behavior.
- Attempted to create a documentation-only commit, but the safety review detected many unrelated
  staged test-file moves already in the index. The commit was not performed and no existing staged
  changes were altered.
- Implementation remains blocked on written-spec review, as required by the design workflow. No
  training code or experiment was changed or started in this phase.

## 2026-07-13 Result Directory Organization Continuation

- 重新审计了用户指定的根级结果、临时目录和锁目录；未删除或覆盖任何 checkpoint、日志、CSV、JSON、HTML 或符号链接。
- 已确认根级 `ldm/` 输出只完成到 epoch 7，配置为无列结构损失的传感器感知 LDM 运行；与已命名 v10 结果不同，需作为部分完成历史运行归档。
- 已确认 `vae_near40_500_v2` 为独立的 VAE 重建/后续 LDM 结果叶目录，seeded v10-A 为独立的 3 epoch checkpoint 选择实验。
- 本阶段计划：移动已确认叶目录、归档未命名根级运行、同步配置/报告/脚本默认路径和三份 test 文档，然后执行静态验证。

## 2026-07-13 Result Directory Organization Execution

- 已将确认的 VAE、seeded v10-A、交互可视化和 dataset protocol audit 结果移动到现有分类目录。
- 已将未命名的根级 LDM/VAE/CD/临时产物整体归档到
  `test/result/archive/ldm_sensor_aware_partial_20260713/`，未删除任何结果文件。
- 已保持 V10-D 的两个锁目录原位，并同步配置、报告、历史结果记录、README、AGENTS、INDEX
  和 legacy loss plotting helper 的路径。
- 待执行静态验证：`compileall`、主要脚本 `--help`、公共模块导入、Shell 引用存在性、旧路径残留和
  Git 差异检查；不运行训练、完整推理或全量评估。

## 2026-07-13 Result Directory Organization Verification

- `conda run -n Radar-Diffusion python -m compileall test` 通过。
- 主要评估、诊断、消融、可视化和协议审计入口的 `--help` 全部通过；公共模块导入和全部
  `test/**/*.sh` 的语法检查通过，Shell 引用的 Python 文件均存在。
- 结果目录内无旧根级叶目录引用或项目绝对 result 路径；README/INDEX 中登记的脚本路径均存在。
- 未运行训练、完整推理、全量评估或覆盖已有结果的命令。
- 审计阶段记录的两个 V10-D 空锁目录在最终工作区中未找到，未擅自重建，待确认是否由外部运行清理。
