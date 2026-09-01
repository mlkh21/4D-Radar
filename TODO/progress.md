# Progress

## 2026-08-29：完成审查修复阶段 1

- 更新 `diffusion_consistency_radar/config/default_config.yaml`：删除静态 `num_gpus`，保留 `cuda_devices` 作为默认设备唯一来源；运行时 GPU 数仍由 launcher 严格派生。
- 从 formal YAML 删除五个仅作用于 legacy MSE 的 VAE 参数，显式加入当前生效的 BCE+Dice 参数。
- 更新 `test/unit/test_formal_training_yaml_defaults.py`，覆盖两卡默认、无静态 GPU 数、active loss 字段和 legacy-only 字段缺席合同。
- 运行 65 项短回归全部通过：formal YAML 5、VAE sparse loss 20、VAE checkpoint 26、distributed protocol 14；未启动正式训练。
- 进入阶段 2：LDM persisted observed-mask 监督与验证合同。

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

## 2026-07-13 v11 Column Curriculum Implementation Plan

- The user approved the written specification.
- Created and self-reviewed
  `docs/superpowers/plans/2026-07-13-ldm-column-curriculum-implementation.md`.
- Split implementation into four TDD tasks: pure schedule math, trainer/log/checkpoint wiring,
  mini-config plus guarded V11 runner wiring, and final verification/project records.
- The plan explicitly preserves old fixed-weight behavior, target voxels, Z64 dimensions, VAE and
  network shapes, and the fixed 32-frame acceptance protocol.
- Formal 500-frame/3-epoch V11 training remains manual. The implementation phase may run only unit
  tests, syntax checks, and one fresh two-frame/one-epoch finite-gradient smoke.

## 2026-07-13 v11 Curriculum Task 1

- Implemented the pure epoch curriculum helper and mathematical-contract tests using RED/GREEN TDD.
- Initial RED failed because the helper did not exist; the first GREEN run passed 71 tests.
- Code-quality review found that boolean weights could be silently coerced to floats. Added eight
  RED cases across four weights, then rejected boolean weights explicitly; 72 tests passed.
- Specification review passed, final code-quality review approved, `py_compile` and
  `git diff --check` passed.
- No trainer wiring, training, inference, checkpoint migration, or result generation occurred.

## 2026-07-13 v11 Curriculum Task 2

- Wired the opt-in curriculum into `OptimizedLDMTrainer`, computing one effective positive/negative
  pair per epoch and passing it into every batch loss call.
- Added effective-weight columns to LDM metrics and stored static schedule plus effective epoch
  values in both best and periodic checkpoint payloads.
- Added RED/GREEN coverage for defaults, invalid configuration, epoch interpolation, one-batch loss
  forwarding, CSV alignment, and both checkpoint paths.
- Verification passed: 76 LDM structure-loss tests, 20 VAE checkpoint-protocol tests,
  `py_compile`, and `git diff --check`. Specification and code-quality reviews both approved.
- No mini runner changes, training, inference, or experiment outputs were produced in this task.

## 2026-07-13 v11 Curriculum Task 3

- Added mini YAML plumbing for the curriculum enable flag and positive/negative start weights with
  strict boolean parsing.
- Extended the existing guarded column runner with V11 while preserving A-D fixed-weight behavior,
  overwrite checks, path audits, locks, reproducibility settings, and training-only scope.
- TDD RED initially reported the missing fields/V11 branch; the final suite passed 101 tests after
  closing review gaps around complete V11-vs-A comparison and V11 hostile-environment protection.
- `bash -n` passed for both scripts and `git diff --check` passed. Specification review and
  code-quality review approved the final Task 3 implementation.
- No real training, inference, evaluation, visualization, CD, or experiment output was started.

## 2026-07-13 v11 Curriculum Task 4

- Closed final-review resume gaps by storing `curriculum_total_epochs`, validating the complete
  six-field curriculum protocol before state loading, and adding real resume/mismatch/legacy tests.
- Final regression passed: 81 LDM curriculum/structure tests, 20 VAE checkpoint-protocol tests, and
  101 mini-runner tests. `compileall`, `py_compile`, both shell syntax checks, and
  `git diff --check` passed during the verification sequence.
- The first bounded smoke attempt stopped before training because OpenMPI could not create a local
  socket inside the sandbox. Re-running the identical two-frame/one-epoch command outside the
  sandbox succeeded; no sample count or epoch limit was increased.
- Repeated the smoke after the resume fix in fresh
  `/tmp/radar_v11_curriculum_smoke_final_20260713`: finite loss `0.535496`, effective CSV weights
  `0.03/0.0`, and matching best/epoch checkpoint metadata with `curriculum_total_epochs=1`.
- Task 2 follow-up specification review, code-quality review, and the final whole-implementation
  review all passed. No formal 500-frame V11 training, inference, evaluation, visualization, or CD
  was run automatically.

## 2026-07-15 v11 Training and Validation Gate

- Confirmed the user-completed V11 result contains three non-empty epoch checkpoints, a non-empty
  best checkpoint, VAE checkpoint, generated config, metrics, and training log.
- Verified the logged schedule is exactly `0.03/0.00 -> 0.025/0.005 -> 0.02/0.01`; mock IR and mock
  calibration ratios are both zero, and all three training losses are finite.
- Ran `test/mini-test/run_ldm_z64_checkpoint_selection.sh` against V11 in a fresh protected output
  directory. All 96 inference samples (3 checkpoints x 32 frames) completed without CUDA, NaN,
  source-hash, sample-count, or protocol errors.
- The selector chose `ldm_epoch0002.pt`, but `all gates passed` is false. Epochs 1/2/3 passed
  1/5, 2/5, and 2/5 gates respectively; epoch2 remains well below recall, top, and trunk gates.
- Stopped at the planned gate. Did not run loop3 500-frame inference, threshold retuning, raw-LiDAR
  visualization acceptance, or CD distillation.

## 2026-07-15 v11 Threshold Sensitivity

- After recording the fixed `0.99` failure, ran a separate selected-checkpoint diagnostic at
  `0.95`, `0.94`, `0.93`, and `0.925`, always using the identical 32 samples, real IR, 20 Euler
  steps, seed 42, Z64 geometry, and epoch2 checkpoint.
- Threshold `0.925` passes all five numeric gates on the validation subset; `0.93` misses only trunk
  by about 0.014. This isolates output probability calibration as a major part of the V11 failure.
- Kept the official checkpoint-selection report unchanged at threshold `0.99`. The new threshold
  folders are explicitly diagnostic and were not fed back into `select_ldm_checkpoint.py`.
- No 500-frame inference, raw-LiDAR visualization, model retraining, checkpoint modification, or CD
  was performed during the threshold diagnostic.

## 2026-07-15 v11 Independent loop3 Evaluation

- Confirmed from the dataset-loader call chain that `split=validation` on
  `Data/NTU4DRadLM_Pre_sensor_aware` selects only `loop3`, then evaluated 32 evenly spaced indices
  from 0 through 6449.
- Completed two independent-scene runs for epoch2 at thresholds `0.99` and `0.925`; both wrote
  complete CSV, JSON, per-frame metrics, summary, and Markdown reports without runtime errors.
- Neither threshold generalizes from garden: both pass only 1/5 gates. The calibrated threshold
  raises recall but also produces strongly frame-dependent density and low voxel precision.
- Stopped before raw-LiDAR 3D generation because the independent numeric gate failed. Also did not
  run 500-frame inference, retraining, or CD.

## 2026-07-15 Cross-Scene Distribution Audit

- Added `test/diagnostics/radar/audit_scene_distribution_shift.py` and four focused unit tests for
  deterministic sampling, common-frame pairing, physical sparse-voxel cropping, channel statistics,
  and IR NPY loading. The focused tests and Python compile check pass.
- Ran a two-frame real-data smoke, then a read-only 500-frame-per-scene audit into
  `test/result/comparison/scene_distribution_audit_v11/`; no checkpoint, target, or dataset file changed.
- Confirmed substantial garden/loop3 occupancy and range-distribution shift, while IR coverage and real
  thermal calibration are consistent across scenes.
- Traced the approximately -50 m/s preprocessed Doppler to the default fixed 50 m/s compensation. Raw
  first frames are near zero Doppler, so another long training run on the current dataset is blocked.
- Did not start full garden training, data regeneration, 500-frame model inference, or CD. The next task
  is a TDD preprocessing-protocol fix followed by a corrected 32-frame data audit.
- Final combined verification initially hit the known OpenMPI sandbox socket failure during module
  import. Moved PyTorch calibration imports inside the real audit path so pure sparse-statistics tests
  stay lightweight; all four tests, compile checks, 1000-row result protocol checks, and
  `git diff --check` then passed.

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

## 2026-07-15 TODO/26 Priority Audit

- 完整读取根目录和 `test/` 的 `AGENTS.md`、`CODEX_HANDOFF.md`，未尝试恢复旧聊天记录。
- 检查 `git status`、diff 统计和现有差异；确认暂存区为空，保留全部用户/历史修改。
- 发现 `TODO/26-7-15.md` 为 GB18030/GBK 编码，采用管道只读转码完整审阅，未改写源文件。
- 已静态核验第一阶段中的 P0-01、P0-04 和 P0-06 调用链，三项描述均与当前代码一致。
- 尝试用实际 garden 帧数统计随机切分的相邻泄漏比例；三次均因只读诊断命令的帧数传递/枚举问题失败，已按协议停止重试。静态代码证据不受影响。
- 当前只完成审计和范围整理；尚未修改生产/测试代码，未运行训练、预处理、推理或评估。
- 下一步先确认本轮修复边界，再提出小步设计；获得批准前不进入实现。

## 2026-07-15 P0-01 Temporal Block Split Design

- 用户批准本轮只处理 P0-01，并批准采用 `garden` 前缀 train、后缀 validation、`loop3`
  保持独立 test 的最小方案。
- 新增并自审书面规格
  `docs/superpowers/specs/2026-07-15-temporal-block-validation-split-design.md`。
- 规格限定生产修改为 `unified_train.py`，测试修改为既有
  `test/unit/test_vae_checkpoint_protocol.py`；不修改 Dataset、launcher、mini runner 或配置。
- 尚未编写实现代码或 RED 测试，未运行训练、推理、预处理或评估。
- 规格自审和暂存差异检查通过；仅该 106 行规格文件提交为 `d363650`，其他工作区修改仍未暂存。
- 当前等待用户复核书面规格；复核通过后才进入 implementation plan 和 RED/GREEN TDD。

## 2026-07-15 P0-01 Implementation Planning

- 用户确认书面规格后，按 `writing-plans` 工作流生成了逐步 RED/GREEN 实现计划。
- 计划包含精确测试代码、预期 RED 原因、最小 GREEN 实现、聚焦验证命令、TODO 记录要求和
  `unified_train.py` 脏工作树的选择性暂存保护。
- 当前仍未修改生产/测试代码，未运行训练、预处理、推理或评估；下一步等待选择执行方式。
- 实现计划自审及 cached diff 检查通过，仅该计划文件提交为 `96df62c`；TODO 与既有 V11
  修改未进入提交。

## 2026-07-15 `26-7-15.md` 分阶段修复续作启动

- 已启用文件化规划、系统化调试和 TDD 工作流，完成问题清单、仓库状态、近期提交与测试目录规则的只读审计。
- 已确认审计文件需要 GB18030 转码读取，并确认工作区存在既有未提交改动；没有覆盖、删除或暂存任何既有文件。
- 已确认 P0-01 设计和实施计划已提交，当前应先确定是验证现有 P0-01 实现，还是直接进入 P0-02。
- 本阶段没有修改生产/测试代码，没有运行训练、预处理、推理或评估。
- 完整编号扫描的首个转码管道失败，已记录并停止重复；下一步采用不同的容错读取方式定位 P0 清单。
- P0-01 RED 命令：`conda run -n Radar-Diffusion python test/unit/test_vae_checkpoint_protocol.py -v`。
- RED 结果：运行 21 项，18 项通过；3 个 `test_temporal_block_split_*` 测试按预期因缺少 `temporal_block_split_indices` 失败。
- 测试出现既有 `torch.load(weights_only=False)` 与 scheduler 调用顺序 warning；它们未导致测试失败，本项不顺带修改。
- 首次 GREEN 尚未运行断言即因测试仍直接导入已移除的旧 helper 而 ImportError；实施计划的步骤顺序存在缺口，已按同一计划提前完成最终测试导入重构，不改变契约断言。
- 最终 GREEN 命令同 RED；结果为 21/21 通过，三个时间块契约测试全部通过。
- 生产修改仅涉及 `unified_train.py` 的纯切分 helper 和主入口调用；测试修改仅涉及既有 `test_vae_checkpoint_protocol.py` 的三个契约测试与导入。
- 完成前复验首次在沙箱内因 OpenMPI 无法创建本地 socket 退出；在沙箱外重跑完全相同的测试、`py_compile` 和 `git diff --check` 后 exit 0，测试为 21/21 通过。
- P0-01 已完成但未提交：训练/验证保持相同 80/20 数量，成员从随机交错改为时间有序前缀/后缀；`loop3` 独立测试、监督 target、每帧体素数、模型和 checkpoint 均未改变。
- 没有运行训练、预处理、推理或评估；没有修改暂存区。第一阶段下一项应按审计优先级进入 P0-06。

## 2026-07-15 P0-06 根因审计启动

- 只读追踪了 `adaptive_occ_from_target` 的参数定义、target 目录校验、逐帧 target 加载、阈值反推、运行日志、正式 launcher 和 mini launcher。
- 已确认根因是通用推理入口允许测试 target 参与输出阈值选择；正式 launcher 默认并未开启该 flag。
- 尚未修改 P0-06 生产或测试代码，尚未运行推理；下一步先形成最小准入设计并取得确认。
- 已比较现有测试和 launcher 模式：推荐扩展两个既有测试文件，不新增测试脚本；设计保持 oracle 上限诊断可复现，但要求第二个明确许可标志并在日志中标记。
- 用户改选独立诊断脚本后重新审计了现有阈值扫描入口；确认全局 validation threshold 与逐帧 oracle count matching 职责不同，新建单职责诊断脚本具有必要性。
- 当前仍处于设计阶段，没有修改 P0-06 生产/测试实现。
- 已确认迁移后的正式 CSV 删除仅 adaptive 使用的 `target_occ_count`，保留固定阈值审计列；正常 target/LiDAR 对比和点云转换继续存在。
- 审计时发现阈值扫描仍使用旧随机 validation 协议，已记录为 P0-01 后续问题；本轮 P0-06 设计不顺带修改该算法。
- 用户批准独立诊断脚本设计并要求写入后直接实施；已新增 P0-06 设计规格，尚未修改生产/测试实现。
- 规格占位符/范围/空白自审通过。隔离文档提交未获批准，因此没有创建 commit；已撤销该文件的暂存，保持原暂存区为空。
- 已新增并自审 406 行 P0-06 实施计划；首次补丁格式错误未产生部分文件，修正后写入成功。
- 用户已授权写入后直接实施，下一步进入正式入口 RED；暂存区和提交历史保持不变。
- P0-06 执行基线：当前仍为普通 `withir` 共享工作区；现有 inference 接口 14/14、mini 协议 2/2 通过，未运行模型推理。
- Task 1 RED：inference 接口 16 项中新增 2 项按预期失败；mini 协议 3 项中新增 1 项按预期失败。
- mini RED 进入旧脚本后在 VAE checkpoint 协议加载处终止，未执行生成、未写推理结果。
- Task 2 GREEN：inference 接口 16/16、mini 协议 3/3 通过，`bash -n test/mini-test/inference_minimal.sh` 通过。
- Task 3 RED：`test_oracle_target_adaptation.py` 运行 6 项，均按预期因诊断脚本尚未创建而失败。
- Task 4 首次 GREEN 有 3 项失败，均因旧算法的 float64/float32 前驱精度问题导致 `k-1` 点；最小 dtype-aware 修复后 6/6 通过。
- P0-06 最终复验：`test_multimodal_inference_interface.py` 16/16、`test_mini_scripts_protocol.py` 3/3、`test_oracle_target_adaptation.py` 6/6，共 25/25 通过。
- `py_compile` 覆盖正式 inference、新诊断脚本及三个测试文件，`bash -n` 与 `git diff --check` 均 exit 0。
- 真实执行旧 `--adaptive_occ_from_target` 得到预期 exit 2 和新诊断路径迁移提示，且发生在必填 checkpoint 校验之前。
- 已完成正式入口与 oracle 诊断的单向隔离：正式推理只用固定阈值，诊断只消费已保存预测体素，不加载模型、不回写正式输出。
- 本项未改变监督信号、target、体素网格、模型、checkpoint 或固定阈值生成结果；正式 CSV schema 删除 adaptive 专用列，历史表需按 schema 区分。
- 没有运行训练、预处理、完整推理或全量评价，没有暂存或提交；P0-06 完成，下一步应单独修复阈值扫描的旧随机 validation 协议。

## 2026-07-15 阈值扫描 validation 协议续修启动

- 已读取适用技能与 `test/AGENTS.md`，并检查阈值扫描、正式训练及两组协议测试的引用和现有差异。
- 已确认正式训练为连续后缀 validation，而阈值扫描及其测试仍使用 seeded `torch.randperm`；本轮尚未修改生产或测试实现。
- 预计只需修改 `diffusion_consistency_radar/scripts/sweep_occ_threshold.py` 与 `test/unit/test_occ_threshold_grid_protocol.py`，三份 TODO 仅记录过程；不会重新改动含历史修改的训练脚本。
- 已核验历史真实命令和结果：输入目录含连续 500 帧，脚本按旧随机协议选择 100 帧；可直接改为尾部连续 100 帧。
- 同时发现历史校准目录属于 `loop3` 独立 test 场景。当前脚本没有可信场景元数据，暂时只能修正成员算法并明确调用约束；待 manifest 修复时再做机器可验证的场景身份保护。
- 用户选择删除 `split_seed` 并批准最小本地修复设计。设计规格已写入并完成占位符、范围、歧义和空白检查；未暂存、未提交。
- 用户已明确要求规格写入后直接修改，因此不再为书面规格单独暂停，下一步转入实施计划和 RED/GREEN。
- 已生成并自审阈值扫描时间块协议实施计划；首次自审发现并消除一处代码省略号，最终计划无实现占位，接口签名与规格一致。
- 执行前检测确认当前为普通 `withir` 工作目录；根据用户“直接修改”和保留同目录未提交 P0 上下文的要求，未创建隔离 worktree。
- 阈值扫描协议基线测试 28/28 通过，证明后续 RED 可与既有失败区分；未运行模型或全量扫描。
- Task 1 RED：更新为时间块契约后共运行 29 项，24 项通过；5 项仅因 `select_evaluation_files`/`prepare_evaluation_files` 仍要求旧 `split_seed` 而 ERROR，失败原因与缺失行为一致。
- Task 1 首次 GREEN 为 28/29：时间块选择测试均通过，唯一失败是 `main()` 仍传入已删除的 `split_seed` 关键字。已将该调用链清理从计划 Task 2 前移到 Task 1，未改变设计范围。
- Task 1 最终 GREEN：29/29 通过，连续 train 前缀、validation 后缀及划分后 `max_files` 均满足契约。
- Task 2 RED：30 项中 27 项通过；JSON 测试因缺少 `split_protocol` ERROR，两种旧 `split_seed` 写法均因未给出时间块迁移说明而 FAIL，失败原因与缺失行为一致。
- Task 2 GREEN：30/30 通过；CLI 已删除 `split_seed` 并提供迁移错误，推荐 JSON 改为记录 `temporal_block_prefix_train_suffix_validation`。
- 最终复验：阈值协议测试 30/30 通过；两个 Python 文件 `py_compile`、`git diff --check` 和空暂存区检查均 exit 0。
- 真实 `--split_seed=42` 调用在必填目录检查前以 exit 2 返回时间块迁移提示；`--help` 不再列出旧参数，并明确 validation 仅用于训练场景阈值标定。
- 本项不改变监督、target、网格、模型、checkpoint 或每帧体素，只改变校准成员与 JSON schema；未运行训练、完整推理或全量扫描，未覆盖历史结果。
- 阈值扫描时间块续修完成且未提交。下一项回到审计第一阶段，设计 dataset manifest 来验证场景身份与预处理版本。
- 分支收尾复验再次运行同一聚焦套件，30/30 通过；当前是普通 `withir` checkout，按既定选择保留现场，不合并、不推送、不清理。

## 2026-07-15 Dataset Manifest 审计启动

- 已只读追踪正式训练、LDM 推理、CD 推理、Dataset loader、预处理脚本和现有数据协议审计。
- 确认训练使用 sensor-aware 根目录而正式推理使用旧 Pre 根目录；现有 per-scene policy 为可选元数据，不能阻止缺失或混合预处理批次。
- 尚未修改 manifest 生产或测试代码，尚未读取体素内容、运行预处理、训练或推理。
- 已只读统计真实目录：sensor-aware garden radar 为 4014 个旧根 symlink；loop3 radar 为 120 个旧根 symlink 加 6330 个普通文件，其他三类模态帧数完整。
- 两个预处理根均未找到 per-scene policy。没有写入、删除、重链或加载任何体素文件；混用证据来自目录项类型和 symlink 目标。
- 用户批准 per-frame SHA-256 严格 manifest，正式入口 fail-closed。已写入并自审设计规格，未暂存、未提交。
- 已生成并自审严格 manifest RED/GREEN 实施计划；一次自审正则错误已记录并改用普通模式完成检查。
- 执行前基线：`test_dataset_protocol_metadata.py` 9/9、`test_sensor_aware_target.py` 4/4 通过；当前仍为普通 `withir` 工作区，继续原地实施。
- Manifest Task 1 首次 RED 为 7 项失败，但错误指向顶层 namespace package 不可见；已补测试项目根路径，尚未写生产模块。
- Manifest Task 1 精确 RED：修正测试加载路径后 7/7 均以 `dataset manifest 模块尚未实现` FAIL，确认覆盖的是缺失核心而非测试环境。
- Manifest Task 1 GREEN：内容级核心 7/7 通过，覆盖跨绝对路径复制、逐文件 SHA-256、四模态连续对齐、symlink/未知文件拒绝、policy/provenance 校验、篡改检测和不可覆盖发布。
- Manifest Task 2 RED：9 项中核心 7 项通过；CLI 测试仅因入口文件不存在失败，预处理安全测试仅因尚无 fresh-output/manifest/failure 汇总集成失败，符合计划缺失行为。
- Manifest Task 2 GREEN：manifest 9/9、既有 sensor-aware target 4/4 通过；CLI create/validate 可用，预处理只接受不存在或空场景目录，policy 后发布 manifest，任一场景失败最终返回非零。
- Manifest Task 3 RED：新增正式 launcher 门禁契约后 10 项中其余 9 项通过，四个子测试均只因对应脚本缺少 `MANIFEST_SCRIPT` 失败。
- Manifest Task 3 GREEN：manifest 10/10 通过，四份 launcher `bash -n` 均 exit 0；训练在清理临时链接目录前验证全部 train scene，推理在第一帧输出前验证全部 test scene，无跳过开关。
- Manifest 最终聚焦复验：manifest 10/10、dataset metadata 9/9、sensor-aware target 4/4，共 23/23 通过；相关 Python 编译、四份 shell 语法和 `git diff --check` 均 exit 0。
- 组合验证在沙箱内曾被 OpenMPI 本地 socket 权限中断；沙箱外重跑完全相同的短命令后通过，未扩大到训练、预处理或推理。
- 对真实 `Data/NTU4DRadLM_Pre_sensor_aware/loop3` 的只读 `validate` 以 exit 2 明确报告缺少 `dataset_manifest.json`；验证前后目录均无 manifest，没有补签、改写或删除任何数据。
- 本项不改变监督信号、target、网格尺寸、已有帧数、每帧体素值、模型或 checkpoint。指标算法不变，但旧场景现被正式入口拒绝，旧指标只能保留为 legacy 结果，不能与新严格协议结果直接合并。
- 严格 manifest 方案 1 已完成且保留在普通 `withir` 工作区，未暂存、未提交、未推送。下一项单独处理正式推理 sensor-aware/真实 IR 与部署/离线评价入口解耦。

## 2026-07-15 正式真实 IR 与部署/评价解耦审计启动

- 已读取适用技能并进入只读根因调查；尚未修改推理、launcher、模型或测试代码。
- 正式训练读取 `NTU4DRadLM_Pre_sensor_aware`，三个正式推理 launcher 仍读取旧 `NTU4DRadLM_Pre`，导致训练/推理预处理根和 IR 可用性不一致。
- LDM/CD launcher 当前强制 target、raw LiDAR 和 LiDAR index，并在同一模型运行中生成预测与评价；底层模型前向实际只需要 Radar 和可选 IR meta。
- `load_multimodal_meta_for_radar()` 在 IR 缺失时只标记 `is_mock_ir=1`，随后 `prepare_multimodal_meta()` 保留合成 thermal；现有 flag 不会把 IR 融合 gate 归零，因此 mock IR 仍可能改变输出。
- checkpoint state dict 已能判定 `model.is_multimodal`；因此正式入口可只对多模态 checkpoint 强制真实 IR，单模态历史 checkpoint 不应伪装成 Radar+IR 正式结果。
- 当前 `CalibrationProvider` 能找到 `Data/config/calib_radar_to_thermal.txt` 并区分 mock/真实外参，但内参仍为硬编码；真实 K/D 解析属于审计第二阶段，本项不应混入。
- 审计原文第一阶段第 4 项的最小边界正是：sensor-aware+真实 IR、移除部署 launch 的 target/LiDAR 强制依赖、将评价参数放入独立 eval 脚本。
- 已搜索现有离线工具：`diagnose_generation_quality.py` 可读取已保存预测并对 target/radar 做诊断图，`evaluate_ldm_vertical_structure.py` 可离线评估 target 垂直结构，但没有一个正式、轻量、同时复现当前 target/raw-LiDAR 指标的统一入口。
- 因此不能只把旧参数复制到另一个 inference launcher；若要让评价真正离线，应复用已保存 `*_voxel.npy`，不重新加载 checkpoint 或运行生成模型。
- 设计前基线：`test_multimodal_inference_interface.py` 16/16、`test_dataset_protocol_metadata.py` 9/9、`test_dataset_manifest_protocol.py` 10/10，共 35/35 通过；未加载正式 checkpoint 或真实帧。
- 规格化复核发现 Dataset 对真实 thermal 外参也施加固定 `50m/s * 200us = 0.01m` 的 x 向同步补偿，逐文件 inference 却只对 mock 标定补偿；本项需统一为现有训练协议，但不把固定值误称为真实飞行动力学。
- 用户批准推荐的两阶段设计：严格真实 IR 正式生成，离线评价复用同一批已保存 voxel；正式缺 IR/thermal 外参或单模态 checkpoint 时 fail-closed。
- 设计规格已写入 `docs/superpowers/specs/2026-07-15-formal-real-ir-deployment-evaluation-design.md`。自审首次发现 evaluator 不能靠默认 40m 网格猜测 checkpoint 输出，已补 `inference_run.json` 作为实际网格/阈值协议；最终无占位、矛盾或未定义范围。
- 按用户此前选择，规格未暂存、未提交；当前等待书面规格复核，尚未进入实施计划或修改生产/测试代码。

## 2026-07-20 正式真实 IR 与部署/评价解耦实施规划

- 用户确认继续此前任务；检查确认实施计划文件尚不存在，没有半写计划需要恢复。
- 首次尝试同时新增计划和更新三份 TODO 时，因 `findings.md` 尾部上下文不精确而整体未应用；已改为分开写入，未留下部分补丁。
- 已把获批规格拆成五个带独立 RED/GREEN 检查点的实施单元，并固定接口、文件、测试命令和失败前置校验。
- 实施计划的规格关键词覆盖、占位符、接口名、空白格式和空暂存区自检通过；未发现遗漏的正式生成或离线评价边界。
- 继续在普通 `withir` 脏工作区原地实施；保留全部既有修改，禁止暂存、提交、推送或运行长任务。
- 当前只写入实施计划和持久化记录，尚未修改本项生产/测试实现，下一步先执行严格真实 IR 的 RED。
- 执行前确认当前为普通 `withir` checkout，并沿用用户已选的原地实施；基线多模态接口 16/16、数据协议 9/9、manifest 10/10，共 35/35 通过。
- Task 1 RED：新增严格 IR 契约后运行 23 项，原 16 项及兼容 missing-IR fallback 均通过；7 项仅因 `require_real_ir`/模型门禁尚不存在而 ERROR，失败原因与预期一致。
- Task 1 GREEN：23/23 通过；严格真实 IR 拒绝缺帧、symlink、非法 shape/数值、mock thermal 外参与单模态模型，并为真实外参补齐训练协议已有的 `+0.01m` legacy 同步位移。
- Task 2 RED：25 项中 23 项通过，仅新增的实际网格 metadata builder 与 runtime/legacy CSV 名称分流 helper 因尚不存在而 ERROR。
- Task 2 GREEN：25/25 通过；无真值参数时写 13 列 `inference_runtime.csv`，显式兼容评价仍写历史 `inference_metrics.csv`，逐文件完成后原子发布实际网格/阈值 `inference_run.json`。
- Task 3 RED：新离线评价测试运行 6 项（7 个子用例），全部仅因 `evaluate_saved_predictions.py` 尚不存在而 ERROR；未加载 checkpoint 或模型。
- Task 3 GREEN：6/6 通过；两帧固定阈值评价正确传播实际网格和 raw LiDAR index，输出逐帧 CSV/汇总 JSON，prediction 哈希不变；全部错误用例均在输出目录创建前失败。
- Task 4 RED：8 项中 evaluator 既有 6 项通过；3 个生成 launcher 子用例因旧 Pre 根/真值参数失败，评价 launcher 因文件缺失 ERROR，边界失败与预期一致。
- Task 4 首次 GREEN 为 7/8，四份 shell 语法均通过；唯一失败由静态测试把 manifest 前的 evaluator 文件存在性检查误判为执行调用，已基于行号证据把断言收窄到实际 `conda run`。
- Task 4 GREEN：formal protocol 8/8 通过，四份 launcher `bash -n` 通过；正式生成只传 Radar/IR 与保存参数，独立评价只调用已保存预测脚本并保留 target/raw LiDAR 参数。
- Task 5 开始：准备运行 50 个聚焦回归（多模态 25、formal protocol 8、dataset metadata 9、manifest 10），不加载正式 checkpoint、不执行模型采样。
- 聚焦回归完成：多模态 25/25、formal protocol 8/8、dataset metadata 9/9、manifest 10/10，共 50/50 通过。
- 静态验证完成：两份生产脚本与两份测试 `py_compile`、四份 launcher `bash -n`、`git diff --check` 和空暂存区检查均通过。
- 真实 manifest 检查首次误用旧 CLI 参数并以 exit 2 返回用法错误；已根据 CLI 帮助改为 `--scene_dir/--expected_scene`，待重跑确认当前 loop3 历史数据仍缺 manifest 且前后无写入。
- 完成前协议复核补充两项 RED：metadata 缺少 `occ_threshold` 必须是明确 ValueError；点坐标必须使用记录的 `voxel_size` 而不是再次猜测。
- 协议复核 RED 确认两处真实问题：缺阈值产生 TypeError，点转换 helper 不接受记录的 voxel size；已分别补充显式 ValueError 和 metadata voxel-size 坐标步长实现。
- 协议复核 GREEN：formal protocol 扩展为 10/10 通过；离线评价按 `inference_run.json` 的实际 voxel size 转坐标，并对不完整阈值 metadata fail-closed。
- 修正后的真实 manifest 验证使用实际 `--scene_dir/--expected_scene` 接口，exit 2 明确报告 `loop3` 缺少 `dataset_manifest.json`，验证前后 manifest 路径均为空。
- 最终验证：多模态接口 25/25、formal protocol 10/10、dataset metadata 9/9、dataset manifest 10/10，共 54/54 通过；未加载正式 checkpoint、未运行模型采样。
- 最终静态验证：`py_compile` 两份生产脚本/两份测试、四份 launcher `bash -n`、`git diff --check`、空暂存区均 exit 0。
- 计划全部完成；保留普通 `withir` checkout 的既有脏工作区，不暂存、不提交、不推送，不删除数据、checkpoint、日志或结果。

## 2026-07-20 P1-06 正式 checkpoint 链实施进展

- 已读取 TODO P1-06、正式 launcher、VAE/LDM/CD 保存代码和现有权重元数据；确认正式 VAE/LDM 缺失、正式 CD 为旧 legacy 协议。
- 已写入 `docs/superpowers/specs/2026-07-20-formal-checkpoint-chain-design.md` 与对应实施计划，范围限定为严格校验、保存元数据和入口门禁，不训练、不覆盖旧结果。
- Task 1 RED/GREEN：新增协议测试后实现 `checkpoint_chain.py`，覆盖普通文件、协议/stage、网格、latent、fusion 范围、父 SHA-256、multimodal state 前缀和 symlink 拒绝。
- Task 2 RED/GREEN：新增 `diagnose_checkpoint_chain.py`；validate 不写报告，成功报告采用空目录保护和原子发布；`--construct --device cpu` 只构建/严格加载，不执行 forward。
- Task 3 RED/GREEN：VAE、LDM、CD 保存 payload 固化 `formal_chain_v1`、stage、grid/fusion config 和父 hash；训练入口只计算 hash，未启动训练。
- Task 4 RED/GREEN：LDM/CD/unified 正式入口在 manifest/生成前执行整链门禁；unified 缺任一阶段直接失败，不再静默跳过。
- 当前已验证：checkpoint-chain 6/6、formal inference protocol 11/11、VAE checkpoint protocol 22/22、multimodal interface 25/25、dataset metadata 9/9、manifest 10/10；CD 新 payload 通过直接函数测试。相关 Python `py_compile`、四份 launcher `bash -n`、`git diff --check` 和空暂存区均通过。
- 监督信号、target、体素数量、模型前向和指标算法不变；正式链门禁只会使缺元数据/旧 legacy 权重的旧入口更早失败。未创建或修改正式 checkpoint、未运行训练/完整推理/预处理。
- 复核实际 `CompleteDualModalityPerceptionNet` state dict 后移除 `projection_layer.` 必需前缀：该层的几何 buffer 为 `persistent=False`，正式协议改为检查实际四类持久化权重并校验 `fusion_*` 网格配置，避免误拒绝新多模态权重。
- 当前正式路径只读诊断以非零状态报告缺失 VAE/LDM 与 legacy CD；未指定报告目录，仓库中没有新增 `checkpoint_chain.json`，现有 `Result/train_results/cd/cd_best.pt` 未被写入。
- CD 保存协议已补充 legacy 分支标记：legacy 教师产物写 `legacy_cd_v0`，多模态学生才写 `formal_chain_v1`；对应多模态 payload 直测仍通过。

## 2026-07-20 P0-06 独立诊断依赖边界加固

- 已按 P0-06 后续审计复查诊断调用链，确认旧脚本对正式 inference 与阈值扫描模块存在直接 import 耦合。
- 已新增轻量 `diffusion_consistency_radar/diagnostics/occupancy_helpers.py`，并将 oracle 诊断切换到该模块；该模块只处理已保存体素，不构造模型、不读取 checkpoint、不执行正式推理。
- 已先写独立性 RED 测试，再完成 GREEN；`test_oracle_target_adaptation.py` 共 7/7 通过，包含原有阈值/输出协议与新依赖边界断言。
- 监督信号、target、网格尺寸、体素数量、正式输出和指标协议均不变；变化仅限离线诊断实现位置与导入边界。
- 后续保持 P0-06 已完成状态，继续按第一阶段顺序处理其他条目；本轮没有长任务、全量评价或任何数据/结果写入。

## 2026-07-20 P0-03 多普勒运动补偿协议修复与代码审查

- 已先新增 RED 测试，覆盖 none/fixed/recorded 速度模式、时间容差、非法速度、Radar↔LiDAR 旋转转换和 shell 默认值；初始因 `motion_protocol` 不存在而失败。
- 已完成 GREEN：新增 `motion_protocol.py`，预处理默认 `velocity_mode=none`；fixed 只接受显式速度，recorded 按帧时间戳解析并在超容差时 fail-fast。
- 已修改并统一 shell、Python parser、`process_scene_task`、multiprocessing worker 和 policy metadata 接口；recorded 源 hash 与速度协议写入 policy，但没有把速度源加入不兼容的 manifest provenance 字段。
- 代码审查发现直接文件执行的同名包遮蔽问题，增加本地模块回退并新增真实 `--help` 子进程回归；该接口问题已验证修复。
- 最终验证：运动协议 8/8、airborne multimodal 9/9、sensor-aware target 4/4；Python 编译、shell 语法、`git diff --check` 和空暂存区均通过。
- 未运行长任务、预处理数据重生成、训练、推理或全量评价；既有数据/结果未改变。监督信号、网格尺寸和指标协议保持不变，下一次重建数据时才会体现默认 none 的 Doppler 分布变化。

## 2026-07-20 P0-05 LiDAR 未观测空间与 free evidence 修复

- 已追踪 `probabilistic_mapping.py` 的 `prob_to_mass → update_from_voxel → streaming_map_update` 调用链，确认空白 voxel 被整图 reliability 当作 free evidence 的根因。
- 已先新增 RED 契约：无 mask 的空白单元必须保持 `occ_prob=0.5`/unknown，有 mask 的显式 free 单元才可下降，mask shape 错误必须在地图改变前失败；同时覆盖 streaming mask 文件发现、同目录排除和稀疏 `.npz` 读取。
- 已完成 GREEN：地图新增 `unknown_mass`，无 mask 默认 occupied-only observed，显式 `(X,Y)`/`(X,Y,Z)` mask 按 BEV 聚合；streaming CLI 新增 `--observed_mask_dir`，CSV/快照记录 mask 与 unknown 统计。
- 代码审查修复了 mask 目录 symlink/错误路径在输出目录创建后的副作用，并防止 `*_observed_mask.npy/.npz` 被 `list_voxel_files()` 当成输入体素；旧 `update_from_voxel` 位置参数保持兼容。
- 聚焦回归：`test_probabilistic_mapping_uncertainty.py` 12/12 通过；相关 Python 编译、CLI 帮助、差异格式检查和空暂存区检查通过。
- 监督信号、target 内容、体素网格/数量、模型和指标公式未改变；没有显式 mask 的历史数据只会从“错误 free”转为保守 unknown。离线 LiDAR 射线投射 mask 与 VAE 可见 free 损失作为后续独立任务，当前没有生成数据或运行长任务。
- 未修改/删除数据、checkpoint、日志或实验结果；未运行预处理、训练、推理、全量评价；未暂存或提交。

## 2026-07-20 P0-05 训练监督链续修

- 已新增 RED 测试：射线 mask 只标记可见路径；VAE 在显式 mask 下忽略未观测空白；无可见空白时损失保持有限；几何增强同步变换 mask。
- 已完成 GREEN：Dataset 从 `lidar_voxel` 生成并 resize `occupancy_observed_mask`，缺文件时退化 occupied-only；`unified_train` 将 mask 传给 VAE loss，旧三参数 `compute_loss` 仍可用。
- VAE BCE/Dice 按可见体素重新统计正负比例，连续通道仅在可见 occupied/有效通道监督；legacy MSE 同样支持可选 mask。occupied target 无论外部 mask 如何均保留监督。
- 代码审查优化射线方向去重，并修复 trainer 新关键字与旧模型替身不匹配；增强接口保持无 mask 时原有二元返回格式。
- 聚焦回归：VAE 稀疏损失 20/20、Dataset metadata 11/11、概率地图 12/12、多模态接口 9/9、sensor-aware target 4/4 通过。
- 监督信号有效区域发生预期变化，但 target 文件、体素数量、模型结构、checkpoint 和指标公式未被改写；未运行训练/预处理/推理或全量评价，未修改数据和实验结果。

## 2026-07-20 P1-01 多传感器时间戳对齐与容差修复

- 已完成调用链审计：bag 解包 receipt time → 文件名、Radar/LiDAR 无阈值索引、预处理器 IR `argmin`，确认三处时间协议不一致。
- RED：`test_timestamp_alignment_protocol.py` 5/5 按预期因 helper/新索引接口缺失失败；未读取真实数据。
- GREEN：新增 header 优先/receipt 回退和最近邻容差 helper；索引按数值时间排序，超限在任何输出前失败，并原子写入 `radar_lidar_sync.csv`（含绝对/带符号 delta）。
- GREEN：解包使用 header 时间戳；预处理器增加 Radar-LiDAR 记录校验、独立 Radar-IR 容差、主进程预计算 IR 配对和 `radar_ir_sync.csv`；失败前不创建输出目录。
- 代码审查修复直接文件执行时同名包遮蔽的隐式重依赖；新增 CLI/环境参数默认 Radar-LiDAR 30ms、Radar-IR 20ms，均可显式覆盖。
- 回归通过：时间戳 5/5、运动协议 8/8、manifest 10/10、airborne 9/9、sensor-aware 4/4、Dataset metadata 11/11；静态编译、直接入口帮助、Shell 语法和差异检查通过。
- 监督/target/网格/模型未改；未来重建数据时帧成员和跨模态对应关系可能变化，实际 delta 已记录但 `dt_sync` 仍是显式 legacy 参数，未在没有符号约定的情况下强行改变运动补偿方向。未运行长任务、预处理、训练、推理或全量评价，未暂存/提交。

## 2026-07-20 P1-02 Thermal 标定与 IR 投影几何统一

- 已审计 `CalibrationProvider`、Dataset IR 加载、inference 元数据加载、投影层和 `audit_dataset_protocol.py`，确认 K/D/S 未接入且同步补偿重复实现。
- RED：新增 thermal 协议 3 项测试，旧实现无法解析 K/D/S、无法接收标定 metadata，也无法证明训练/推理共享补偿函数。
- GREEN：Provider 解析 `calib_cam_thermal.txt`，将原始 K 按 `640×512→640×480` 缩放并记录 D/S/source；IR 图像通过共享 resize+undistort 函数处理。
- GREEN：Dataset 与 inference 均调用 Provider 和共享 `apply_legacy_sync_compensation`；严格真实 IR 缺少 thermal 相机 S/K/D 时 fail-closed；审计脚本移除重复 K。
- 回归通过：thermal 3/3、multimodal inference 26/26、Dataset metadata 11/11、airborne 9/9、sensor-aware 4/4；Python 编译与既有入口协议保持兼容。
- 不改变监督信号、target、体素数量、模型/权重；旧结果不与去畸变新协议直接混合。未运行训练、完整预处理、推理或全量评价，未修改数据和实验结果。

## 2026-07-20 P1-03 PointCloud2 字段 schema 固定化

- 已完成调用链审计：bag 解包的 PointCloud2 字段选择直接决定 `radar_pcl/*.npy` 列，预处理体素化固定将 col3/col4 解释为 intensity/Doppler。
- RED 测试先验证缺 intensity 或缺 Doppler 时原实现会输出四列并错位；没有读取真实 bag 或写入数据集。
- 已完成 GREEN：按字段名和别名构造固定五列，缺失特征补零，缺失坐标抛出明确错误；输出 schema 元数据采用临时文件加 `os.replace` 原子发布。
- 代码审查确认旧 PointCloud v1 与 Livox 分支仍保留，时间戳索引只扫描 `.npy/.npz`，不会把 `pointcloud_schema.json` 当帧；下游五列接口无需改变。
- 聚焦验证：PointCloud2 2/2、时间戳 5/5、运动协议 8/8、Airborne 多模态 9/9、sensor-aware target 4/4；静态编译、差异检查和空暂存区通过。
- 监督/体素/指标影响：新解包数据的强度与 Doppler 通道位置确定且缺失值为零，点数、体素网格、target 生成和模型结构不变；历史错误列文件未改写，相关指标不可与新 schema 数据无条件混合。未解包、训练、推理或修改任何数据/结果。

## 2026-07-22 P1-04 启动

- 已恢复 P1-01 至 P1-03 的持久化上下文，并把 P1-04 拆为调用链审计、方案批准、规格/计划、TDD 实施和回归五阶段。
- 当前只进行只读审计和设计，不修改生产实现；不会自动运行完整数据统计、预处理、训练或推理。
- 已定位第一处根因：方差通道被当作普通均值插值，缺少 `mean²` 的组间方差项；同时确认训练 Dataset 与逐帧 inference 共用该 helper，可在单一边界修复。
- 已确认第二处接口缺口：训练/CD/inference 没有统一 normalization artifact 参数；现有审计结果只报告 Doppler/variance，尚不能作为强度稳健缩放配置直接消费。
- 已完成 20 帧/场景轻量只读抽样：确认旧数据 Doppler 存在约 -50m/s 整体偏置、intensity 主要位于 5～22、细体素 variance 大多为零；没有写入测试结果或数据文件。
- 已审计 checkpoint/运行 metadata：Radar normalization 尚未进入 LDM/CD 链或 inference run，设计中需要增加同一协议的训练、蒸馏、推理绑定。
- 已搜索仓库传感器量程和历史 policy：没有可信 Doppler 硬件上限，且当前 sensor-aware 场景缺 preprocess policy；设计将采用显式必填量程，拒绝自动猜值。
- 已完成增强调用链审计：归一化将安排在物理量 resize/augmentation 之后，避免 target 与 Radar 的 Doppler jitter 单位分叉；YAML augmentation 未接线仍留给 P2-01。
- 用户已批准方案 1、正式 fail-closed 和显式 `doppler_scale_mps`；设计规格已完成并自审，当前等待书面规格复核，尚未写实施计划或修改生产代码。
- 用户已确认书面规格；开始编写实施计划，计划完成前不进入生产实现。
- 实施计划接口审计已覆盖 inference checkpoint loader 与 formal chain fixture；发现 Radar condition 的 VAE latent/专用 encoder 双路径，下一步先确认实际数据流再锁定任务边界。
- 已确认正式多模态 denoiser 实际不消费 `z_cond`：训练中的 Radar→VAE 是无效计算，推理只借其取得 shape。设计规格已补充为多模态 Radar 仅走专用 encoder，legacy 单模态才保留 VAE condition latent。
- 实施接口进一步收口：保留通用 target/mask resize，新建 Radar 专用二阶矩 resize；正式统计 builder 复用现有场景 manifest 重算校验，不另造 provenance 校验分支。
- 完整 RED/GREEN 实施计划已写入并自审；删除 P2-01 YAML 接线和 shell 重复 schema 校验两处范围外内容。沿用用户批准的 `withir` 原地实施，开始 Task 1 聚焦基线与 RED。
- Task 1 基线：Dataset metadata 11/11、multimodal inference 26/26 通过；并行 Airborne 命令未返回可判定输出，暂不计为通过并安排单独重跑。已新增方差总公式、局部+组间方差、空体素/输入不变性三项 RED。
- Task 1 resize RED 已确认 3/3 因专用接口缺失失败；已实现独立 `resize_radar_voxel_channels()`，通用 target/mask resize 未改，准备运行最小 GREEN。
- Task 1 resize GREEN：3/3 通过。第二批 RED 已加入 normalization 数学、artifact 文件 hash、非法 schema/nonformal/网格/量程/symlink 拒绝；同时把规格中的输入接口从含糊的“至少四通道”收紧为模型实际消费的严格四通道。
- Task 1 normalization RED：原有 resize 3/3 继续通过，新 3 项均因协议模块不存在失败。已新增纯协议模块，准备验证 schema、文件 hash 和运行时数学 GREEN。
- Task 1 normalization 初步 GREEN：6/6 通过；代码审查继续补充非法 Radar 输入和 spec/hash 双重绑定测试，再执行该任务完整回归。
- Task 1 完成：normalization/resize 8/8、Dataset metadata 11/11、相关编译与 `git diff --check` 通过。Airborne 外部回归通过前 5 项后在第 6 项底层算子长时间无输出，已终止等待并记为未完成而非通过。
- Task 2 RED 已写入：显式场景/全帧统计、crop/resize/log1p 顺序、manifest provenance、抽样非正式标志、原子不覆盖和无 occupied 失败边界；测试仅使用临时小体素。
- Task 2 RED 确认 4/4 因 builder 模块不存在失败；已实现 manifest 验证、真实 crop/Radar resize、occupied log 分位数、formal 标记与末端原子发布，准备运行 GREEN。
- Task 2 完成：builder 4/4、共享 normalization 8/8、CLI `--help`、相关编译与 `git diff --check` 全部通过；没有读取真实训练场景或生成正式 artifact。开始 Task 3 Dataset/增强/配置预检 RED。
- Task 3 第一批 RED 已写入：Dataset 默认严格/显式 legacy、物理 Doppler 增强后归一化与 hash metadata，以及 condition 噪声保持 occupancy/非负 variance。
- Task 3 RED 已确认：Dataset 新接口 1 error + 严格默认 1 failure，condition 噪声 1 failure；其余既有用例保持通过。已实现 Dataset 协议字段、Radar 专用 resize/后置 normalization 和增强物理边界，准备迁移既有测试/诊断的显式 legacy 调用。
- Task 3 初步 GREEN：normalization/增强 9/9，通过的新 Dataset strict/legacy/顺序用例也已转绿；Dataset suite 仅余旧顶层 `cm` 导入触发的 2 个 ValueError，已局部修复兼容捕获并准备复跑。
- Task 3 Dataset GREEN：13/13 通过。已新增训练 preflight/default YAML RED，锁定正式空配置拒绝、有效 artifact/hash、显式 legacy 互斥与默认空值语义。
- Task 3 preflight RED 确认 2/2 因解析函数不存在失败；已在统一训练 Dataset 创建前接入 artifact/scale 校验，新增显式诊断 legacy CLI，并把默认 YAML 保持为空路径/null。
- Task 3 完成：preflight/normalization 11/11、Dataset 13/13、thermal 3/3、统一训练 `--help`、编译和差异检查通过。Task 4 RED 已加入 LDM/CD payload、教师继承比较和多模态 target-only VAE 编码边界。
- Task 4 RED：LDM payload 缺 normalization，LDM/CD latent helper 均不存在；同时修正 CD 测试直接入口未执行 payload/preflight 的假绿。共享协议模块已增加 checkpoint embedded spec/hash 提取器。
- Task 4 实现进行中：LDM/CD payload 与 resume 已接入 normalization；正式多模态 latent helper 只编码 target。standalone CD 在 Dataset/save_dir 前加载配置 artifact 并比较教师 embedded spec/hash，legacy 必须显式且互斥。

## 2026-07-22 P1-04 完成

- Task 4 完成：LDM/CD payload、教师继承和 resume 均绑定 normalization spec/hash；正式多模态训练只编码 target。修复原子保存测试替身接口后，LDM 结构回归 81/81、VAE checkpoint 23/23、两份 CD 接口测试全部通过。
- Task 5 完成：formal checkpoint chain 扩展 normalization 校验，8/8 通过；inference 默认拒绝旧 checkpoint，逐文件/Dataset 共用 embedded spec，多模态采样不编码 Radar，接口回归 31/31 通过。
- Task 6 完成：mini/历史 runner 显式 legacy，正式 launcher 保持无 legacy；审查并修复 IR 消融的量纲、网格和输出副作用顺序，mini 5/5、formal inference 11/11、IR 消融 12/12 通过。
- 最终回归：normalization 11/11、builder 4/4、Dataset 13/13、VAE checkpoint 23/23、checkpoint chain 8/8、multimodal inference 31/31、formal inference 11/11、manifest 10/10、mini protocol 5/5、IR 消融 12/12、thermal 3/3、LDM 结构 81/81，另两份 CD 直接接口测试通过。
- 静态验证：9 个相关 Python 文件 `py_compile`、正式/mini/历史诊断共 9 份 shell `bash -n`、`git diff --check`、`git diff --cached --quiet` 全部 exit 0。
- 本轮没有生成正式 artifact，没有重写数据、checkpoint、日志或结果，没有训练、完整预处理、模型采样、全量推理/评价，也没有暂存、提交或推送。
- P1-04 代码实施已完成；后续长任务必须先为明确的正式训练场景生成全帧 artifact，写入 `radar_normalization_path` 和匹配的 `doppler_scale_mps`，然后从头训练新的正式 LDM/CD。

## 2026-07-22 P1-01 真实数据时间容差续修启动

- 用户执行全量候选数据脚本，在 Step 1 garden 严格 Radar-LiDAR 索引处因首帧最近邻偏差 `51.207066ms > 30ms` 失败。
- 失败早于候选体素生成；未启动预处理、normalization artifact、训练或推理。当前进入只读时间差分布审计，不重复原命令，也不先验放宽阈值。
- 已完成 garden/loop3 全量文件名时间戳只读统计：30ms 分别拒绝 1251/4014 和 1858/6450 对，重叠区最大约 64ms；20ms Radar-IR 也分别拒绝 52/4816 和 96/7738 帧。下一步审计 Raw 解包 provenance 与 bag header/receipt 差异。
- 已只读对照 garden bag 首帧：现有 Radar/LiDAR/IR 文件名均精确对应 receipt time，而非各消息 header time，确认当前 Raw 是旧解包协议产物。下一步直接扫描 bag header 时间差，验证新解包后既定 30ms/20ms 阈值是否成立。
- 已完成 4 个 bag 的 header 全量只读扫描。30ms Radar-LiDAR 在 garden/loop3 仍分别拒绝 1527/1790 个重叠帧，证明阈值与 12Hz/10Hz 异步节拍不兼容；正常尾部约 43.6ms，loop3 另有 81.0ms 掉帧异常。Radar-IR 最大约 22.7ms，20ms 也略严。
- 当前拟定最小续修：45ms Radar-LiDAR 正常窗口、25ms Radar-IR 窗口；索引对超限候选显式记录并跳过，而非把异常强行匹配或整场失败。实施前将补足精确超限数量、RED 契约和现有调用链审查。
- 已新增 3 项聚焦 RED：少量超限候选必须写 rejected CSV 并跳过、拒绝比例过高必须在任何索引发布前失败、v2 正式脚本必须先重解包 header-time Raw。执行结果为新增 3 项 ERROR、既有 5 项通过，失败原因分别是新 API/sidecar/解包步骤尚未实现，测试有效。
- 第一轮 GREEN 完成：v2 先从 bag 重建独立 header-time Raw；索引使用 45ms 正常窗口，把少量异常写入 rejected sidecar，并以 1% 比例 fail-closed；Radar-IR 更新为独立 25ms 门禁。
- 真实 bag 精确只读复核得到 garden/loop3 的 `>45ms` 候选分别为 `1/4014` 和 `18/6450`，证明 1% 门禁不会误拒正常场景，又能阻止继续使用旧 receipt-time Raw。
- 最终审查补齐损坏 bag 立即失败、Thermal 目录完整性检查，并移除正式环境不存在的 pandas 和未使用 open3d 依赖；标准库 CSV 动态字段测试与解包器直接 `--help` 已通过。
- P1-01 真实容差续修完成：37 项聚焦回归全部通过，Python/Shell/CLI/diff 静态检查通过，两个候选目录仍不存在；下一步由用户显式运行更新后的 `preprocess-v2.sh` 执行长时间数据重建。

## 2026-08-20 P1-05 移动平台局部地图更新启动

- 已读取根目录与测试规则、现有脏工作区、概率地图核心、streaming 入口和聚焦测试；确认在 P0-05 unknown/free 未提交修改上做局部增量。
- 根因已定位为“协方差参与 reliability，但位姿和真实时间戳未进入空间更新”；当前正在设计保持旧 2D consumer 兼容的 pose-aware 分层地图协议。
- 已完成 consumer/roadmap 审计；实施边界确定为离线 pose-aware 3D layers + 旧 BEV 兼容，不在本项同时接入 ROS/PX4。
- P1-05 RED 已确认：既有 12 项全部通过；新增位移、旋转/Z、非法位姿/时间、3D mask、pose CSV 五项分别因缺 `T_local_body`、layer snapshot 和 loader 接口而 ERROR，测试准确命中旧实现缺口。
- 第一轮 GREEN 17/17：核心已支持严格 `T_local_body`、三维前向 warp、分层 D-S 状态和无副作用时间门禁；pose CSV helper 已验证帧覆盖、严格时间和四元数方向。下一步接入 streaming 主循环与输出审计。
- 第二批 RED 已确认：核心 17 项继续通过；新增三项分别因主入口缺 `--pose_file`、mask loader 缺高度保留参数而 ERROR，证明 helper 尚未被正式数据流消费。
- 第二批 GREEN 首轮为 19/20；唯一失败不是 pose 对齐，而是旧体素 layout 猜测把 `(4,2,2,4)` 误作通道优先。已转入显式 layout 接口修复，避免用放大测试尺寸掩盖真实问题。
- 显式 `xyzc/czxy` layout 与歧义拒绝完成；pose-aware 主入口、分层快照和 `map_run.json` 微型端到端测试转绿。
- 代码审查完成：三维地图点不再写死 `z=0`，查询使用 body local 三维位置，target 同位姿变换后再评价；prior DEM shape、target 帧覆盖和多样本 batch 均 fail-closed。
- 修复 D-S 隐形语义错误：未观测体素保持 `belief=0/unknown=1`，时间衰减同步作用于 occupied/free/unknown 质量；旧 BEV 键保留兼容。
- 修复直接 CLI 对完整 `cm` 训练包的重依赖；首次 `--help` 的 OpenMPI 失败不计为通过，轻量导入后单独复验成功。
- 最终聚焦测试 27/27 通过；相关 Python 静态编译、直接 CLI `--help` 与 `git diff --check` 通过。仅运行临时小数组/两帧微型入口，未执行真实数据重放、预处理、训练、推理或全量评价。
- 监督与资源影响：模型监督、target 和单帧体素数不变；正式常用 `128×128×32` 地图新增四个 float32 分层状态约 8 MiB，输出指标因 local 位姿对齐、三维来源和 pignistic occupancy 语义改变而不可与旧日志直接混合。
- P1-05 的 pose/真实时间/分高度层建议已完成；动态层因缺可信 evidence 协议保留为独立后续，不使用未校准 Doppler 阈值制造假动态标签。
- 开始 P1-05 动态层续修：全仓审计确认没有可直接复用的动态 mask/跟踪器；Doppler 在物理预处理、normalization 和生成输出之间单位不统一，当前转向显式 sidecar evidence + provenance 方案。
- 已审计 inference 输出与 preprocess provenance；确定动态 evidence 使用每帧 probability+observed `.npz` 和目录级严格 JSON，地图仅融合外部证据并做更快衰减，不负责 Doppler 分类。
- 动态续修基线 27/27 通过；开始 RED，范围仅为地图核心、streaming 协议 loader 和现有聚焦测试。
- 动态层 RED 确认：原 27 项保持通过；新增 4 项因缺 `dynamic_decay_rate`、核心 evidence 参数/快照、streaming CLI 协议而 ERROR，测试命中预期缺口。
- 动态核心首轮 GREEN 3/3：显式 probability+observed 经同一 pose warp；静态层扣除动态份额，旧 occupancy 键输出静态∪动态组合；动态四态数组按需创建并使用更高时间衰减率。
- 动态层审查修复完成：兼容快照由统一 belief/plausibility 推导 probability/unknown；外部动态 evidence 与 Radar 专属可靠度解耦，仅保留 odometry 折扣；直接 API 与 JSON 数值类型均 fail-closed，run metadata 同时记录 base/effective 衰减率。
- 动态层聚焦回归最终扩展至 36/36 通过：补齐稀疏 sidecar 不污染 static-free、NPZ 全量预检、重复 frame 键、来源 hash 状态和可选参数兼容边界。
- P1-05 动态续修完成：逐帧 probability+observed 严格协议、body→local 对齐、static/dynamic 分离、更快时间衰减、D-S 一致兼容输出、实际输入 hash 与 base/effective 参数均已接线。
- 最终静态验证：三份相关 Python `py_compile`、streaming 直接 `--help`、`git diff --check`、`git diff --cached --quiet` 全部 exit 0。仅运行临时小数组和单/双帧临时目录；未运行真实数据、预处理、训练、推理或全量评价，未暂存/提交/推送。
- 监督与资源影响：模型监督、target、输入体素数量和 checkpoint 不变；动态 sidecar 启用时为常用 `128×128×32` 网格增加约 8 MiB 持久状态，旧 occupancy/查询/地图指标变为 static 与明确 dynamic-occupied 的覆盖结果，不能与旧日志直接混合。

## 2026-08-20 P1-07 启动

- 已复核原审计、现有计划和训练代码：P1-07 的 checkpoint 网格/hash 子问题已解决，但 LDM 独立验证与 CD 教师表述仍未解决。
- 实施边界确定为 LDM validation/best 协议和 CD EMA consistency provenance；先写轻量接口 RED，不运行正式训练、完整推理或全量评价。
- 已确认主入口现成 `val_loader` 仅漏传给 LDM；既有完整生成 checkpoint selector 保留为训练后 gate。本轮训练期将增加确定性 denoising validation proxy，不调用 20 步采样器。

## 2026-08-20 Codex VS Code 历史会话读取修复

- 已完成扩展日志、配置层、会话索引、SQLite 状态库和 app-server 协议调用链审计。
- 已确认历史文件完整且旧会话全文可读，并复现 `OpenAI`/`openai` 提供方精确过滤造成的旧历史缺失。
- 已在 `/tmp` 快照验证候选修复：严格配置解析通过，旧会话列表与 `thread/read(includeTurns=true)` 通过。
- 已备份 `/home/zxj/.codex/config.toml` 并安装修复配置；未修改项目代码、数据集、checkpoint、训练日志、实验结果或任何历史会话文件。
- 最终剩余操作是用户在当前回复交付后执行 `Developer: Reload Window`，使正在运行的扩展进程重新加载用户配置。

## 2026-08-20 P1-07 LDM 独立验证与 CD 语义

- 已完成 RED：新增 LDM validation/best/checkpoint 协议测试与 CD checkpoint 语义测试。
- RED 结果符合预期：LDM 测试因验证协议符号尚不存在而导入失败；CD 测试因 checkpoint 缺少 `training_semantics` 失败。
- 本轮只运行相关单元测试，不启动长时间 LDM/CD 训练或全量数据预处理。
- 已完成 GREEN：LDM 每 epoch 消费独立连续时间后缀验证集，固定噪声计算 denoising latent loss 与解码 occupancy IoU，并按验证 IoU/latent loss 保存 `ldm_best.pt`。
- 已完成 checkpoint/恢复协议：保存 selector、split、固定验证参数和 current/best 指标；新协议 mismatch 在权重加载前失败，旧 checkpoint 保持显式兼容并由下一轮验证升级。
- 已准确标记 CD 当前实现为 `ldm_initialized_ema_consistency_v1`：LDM 是初始化来源，consistency target 是 `cd_model_ema`；没有改写 CD 优化公式。
- 已完成接口审查：统一入口传入 `val_loader`，所有仓库内 LDM trainer 调用已同步；正式/mini checkpoint 路径和离线固定 32 帧 selector 无需改接口。
- 最终聚焦回归全部通过：159 项具名 unittest、两个 CD 脚本式接口测试、相关 Python 编译和 `git diff --check`；没有启动长训练、全量预处理或真实数据推理。

## 2026-08-20 Radar normalization 步骤 6 失败续修

- 用户完成 preprocess-v2 步骤 1-5；步骤 6 因 occupied intensity 的 `log1p` IQR 为零而 fail-closed，artifact 未发布。
- 已启动只读数据分布与统计调用链诊断；不删除候选数据、不重跑全量预处理、不启动训练。
- 已确认步骤 1-5 产物完整到 manifest 层，artifact 原子发布前失败且路径不存在；初步排除体素/loader 通道索引错位。
- 已用全量 Raw/resize 前统计与 16 帧 resize 对照定位根因：max-pooled occupancy 与 trilinear attribute 采样不对齐，76.49% coarse occupied intensity 被伪置零。
- 决定保持 normalization 的零 IQR 拒绝逻辑，转入 Radar 重采样 aligned-bin RED/GREEN；候选数据无需重建，只需修复后重跑步骤 6。
- aligned-bin RED 精确失败：12 项中原 11 项通过，新增稀疏边缘点用例仅因两个 coarse occupied 输出的 intensity 被现有 trilinear 路径清零而 FAIL，证明测试命中真实缺口。
- aligned-bin GREEN：Radar 属性分子/occupied 分母改用相同 adaptive average pooling 分箱，保留既有 max occupancy 与总方差公式；normalization 协议 12/12、builder 4/4 通过。
- 调用端审查确认训练、正式推理和 artifact builder 共用修复函数；真实 garden 32 帧不落盘烟测得到正有限 `log_iqr=0.366153`。
- 最终聚焦回归 73/73 通过；修复范围为 `dataset_loader.py` 与既有 normalization 协议测试，没有修改候选数据或 manifest。
- 本阶段结束时等待用户只重跑 normalization builder 的全 garden 命令；后续成功结果见下一节，未重复全量 preprocess-v2。

## 2026-08-20 正式训练协议切换完成

- 用户已生成 artifact 并完成 `test_normalization_artifact.sh` 验收：garden 4013、loop3 6432、首样本四通道 32×128×128、真实 IR/标定和 artifact SHA-256 全部通过。
- 完成训练/推理/评价调用链审查，确认旧数据根、默认空 normalization、固定旧结果根和隐式 resume 会造成接口不匹配与历史结果污染。
- RED 精确命中：normalization 11/12、launcher 5/6；正式推理评价既有 7 项通过、新协议断言 4 处失败。
- GREEN 将 default YAML、训练 launcher、三个正式生成 launcher 和独立评价 launcher 切换到 `formal_p1_04_full120_86p8_v1`，加入固定 artifact hash、独立结果根与显式 `ALLOW_RESUME=1` 门禁。
- 代码审查补获 mini legacy 派生配置冲突；RED 5/6 后最小清空 mini artifact/scale，扩展回归 101 项 mini launcher 测试通过。
- 最终回归通过：normalization 12、mini/formal launcher 6、formal inference/evaluation 11、mini train 101、manifest 10、checkpoint chain 8、VAE checkpoint 23、多模态 inference 31，另两份 CD 接口测试通过，共 202 项具名 unittest。
- 7 份 shell 语法、candidate preprocessed/Raw/artifact 路径、默认配置与 artifact 数值对照、`git diff --check` 全部通过。正式结果根仍为 fresh；没有启动训练、推理、评价，没有删除或覆盖数据、checkpoint、日志或历史结果。
- 下一步由用户显式运行 `bash diffusion_consistency_radar/launch/train_unified.sh vae`，这是长时间训练命令，本轮未自动执行。

## 2026-08-20 正式 VAE 启动导入错误续修

- 用户首次启动在 Python 导入阶段遇到 `No module named 'diffusion_consistency_radar'`；检查确认正式结果根未创建，没有产生可恢复 checkpoint。
- 已增加脱离仓库工作目录且清除 `PYTHONPATH` 的入口测试；RED 精确复现 `unified_train.py → cd_train_optimized.py → checkpoint_chain.py` 异常。
- 已为 `unified_train.py`、`cd_train_optimized.py` 同时引导仓库根/包目录，并统一正式包导入，删除会掩盖内部缺包及造成模块双重身份的 fallback。
- 聚焦回归全部通过：两个直接入口、CD 多模态接口、checkpoint 链 8 项、VAE checkpoint 23 项、mini launcher 6 项；相关 Python 编译、训练 launcher `bash -n` 和 `git diff --check` 通过。
- 未自动重启长训练；监督信号、样本成员、体素数量、模型/checkpoint 协议和指标均未改变。用户可重新执行同一 VAE launcher，不需要 `ALLOW_RESUME=1`。

## 2026-08-20 正式 VAE 首 batch 拼接错误续修

- 第二次正式启动已进入 epoch 1，随后在 DataLoader worker 首 batch 因 `preprocess_policy` 中合法 JSON null 无法被默认 collate 而退出；无前向、反向、优化器更新或 checkpoint。
- RED 新增 nullable policy 纯内存用例：既有 13 项通过，新用例因共享 collator 不存在而精确 ERROR。
- GREEN 增加共享 `collate_voxel_samples()`：模型字段保持默认严格拼接，审计 policy 保留逐样本原始字典；统一 train/val、standalone CD 和条件推理四个 DataLoader 已接入。
- Dataset 14 项、机载多模态 9 项、多模态推理 31 项及两份 CD 脚本式接口测试通过；真实 garden 两样本、多 worker batch 烟测和相关 Python 编译、调用端静态审查、`git diff --check` 通过。
- 已将失败的 header-only CSV/日志无损归档到协议结果根 `failed_starts/vae_20260820_212426_collate_failure/`；active `vae/` 不存在，下一次应 fresh 启动，不设置 `ALLOW_RESUME=1`。
- 未自动重启长训练；监督、样本划分、体素数量、模型、loss、normalization、checkpoint 与指标定义均未改变。

## 2026-08-21 8 GB 单卡正式协议 mini 训练入口

- 完成 legacy/formal mini 分支、`formal_mini_chain_v1` checkpoint 传播和正式链隔离；保留历史 legacy 入口与结果不变。
- 新增 `test/mini-test/run_formal_mini_8gb.sh`：单阶段、低样本、固定 batch/worker、显存/温度/时长门禁、进程组逐级停止、fresh 阶段 scratch/config 和非空输出保护。
- 新增 `MINI_PREFLIGHT_ONLY=1`；它在任何 scratch/config/output 创建和训练启动前验证 GPU、artifact、full120 网格、garden 数据及输出路径。
- 负向预检捕获并修复 Conda heredoc stdin 静默丢失问题；错误 artifact SHA 现在 fail-closed，正确 SHA 会明确打印校验成功行。
- 完成文档更新，给出预检、VAE/LDM/CD 分阶段运行、阶段间冷却、1 帧推理烟测和失败现场保留方法。
- 最终真实预检通过：RTX 4070 Laptop GPU 8188 MiB、空闲 7186 MiB、37°C，未启动训练且未创建预检输出目录。
- 聚焦验证通过：mini 协议 11、mini 配置/安全 103、checkpoint chain 10、VAE checkpoint 23，CD 入口测试通过；shell 语法和 Python 编译通过。没有删除或覆盖数据、checkpoint、训练日志和实验结果。

## 2026-08-21 外部审查第一批修复完成

- 实施 0--2 已完成：formal checkpoint/data v2、manifest profile、显式场景/标定、真实 IR 门禁、LiDAR 投影 frame 和 signed 时间补偿均已接线。
- 代码审查补齐两项接口缺口：训练 preflight 交叉检查 manifest 内 target policy/Radar--IR sync/calibration provenance；推理进程交叉绑定 VAE/LDM/CD data identity 与当前 deployment manifest/calibration。
- 正式训练不再创建 `.tmp_train_dataset`，正式预处理/训练/推理默认路径统一改为 `formal_v2_range_pending`；缺少范围决定、observed/split/data artifact 时明确失败，不会误用 v1 candidate 或结果根。
- 已输出 `test/result/comparison/far_range_supervision_audit_v2/far_range_audit.json` 和逐帧 CSV。v1 宽口径结果保留不覆盖，但后续决策只使用限制完整 XYZ 体素盒的 v2。
- v2 关键结果：garden 4013 帧中 3934 帧存在 80--120 m LiDAR occupied，总计 365069；当前 target 远距为 0，保留率 0；远距 raw/near raw 为 0.404%，远距/近距 occupied 为 1.488%。
- 当前推荐 0--80 m formal v2，保持张量 `32×128×128`，把 80--120 m 留给地图 unknown；等待用户确认后才实现 observed/split/normalization 和 smoke。
- 最终逐文件轻量回归通过：checkpoint 13、VAE 24、Dataset 15、manifest 12、thermal 6、motion 9、multimodal inference 33、formal inference 11、normalization 12、far audit 4，CD 两份直接接口通过；相关 Python 编译、6 份 shell 语法和 `git diff --check` 通过。
- 完整 airborne 多模态测试中的既有完整 3D CPU forward 两次超过 90 秒无输出后仅终止本轮测试进程；直接相关几何 5 项通过，因此不把完整文件报告为通过。
- 未运行数据重建、mini/full training、GPU 推理或正式评价；未删除、覆盖、移动数据、checkpoint、日志和实验结果。

## 2026-08-21 formal v2 0--80 m 分支完成

- 完成持久 observed mask、五模态 training manifest、基于真实 Radar 时间的 split/purge、train-only normalization 和可重建 formal data protocol。
- 完成正式配置/launcher 的 0--80 m fresh v2 路径切换；旧 full120 路径未删除或覆盖，正式训练仍因全量新 artifact 尚不存在而 fail-closed。
- 完成 evidence 0--80 m / map 0--120 m 双范围接口，新增远距地图单元保持 unknown 的反例测试，mapping 37/37 通过。
- 完成 garden 4 帧 fresh smoke：五模态 manifest `frame_count=4`，正式 Dataset 加载 4 个样本，真实 IR/标定和 persisted observed 均通过。
- 聚焦回归通过：observed 4、temporal/data protocol 4、manifest 12、Dataset 15、normalization builder 4、normalization protocol 12、mapping 37、formal inference 11、mini launcher 11、VAE checkpoint 24、checkpoint chain 13，以及 CD 两组入口测试；相关 Python/Shell 静态检查通过。
- 代码审查补齐独立 CD 与 unified 的 formal 数据门禁不对称；未发现正式 launcher 残留 `range_pending`、旧 full120 数据根或 `.tmp_train_dataset`。
- 尚未执行全量 v2 重建、8 GB mini、正式训练、GPU 推理或正式评价；deployment-profile 数据视图生产是下一项明确工作。

## 2026-08-21 严格 deployment-profile 生成链完成

- 新增 `deployment_view.py` 和 `build_deployment_view.py`，实现 fresh 输出、training v2 → deployment v3、根/场景双收据、hardlink/copy 和自包含父身份验证。
- manifest 正式 deployment 入口只接受 schema v3；旧 v2 deployment 仅可作为非正式历史诊断，不能通过正式推理门禁。
- 三个正式推理 launcher 已改为完整 deployment dataset validator；推理 Python 身份门禁同步接入父 manifest/receipt/current calibration/checkpoint 交叉校验。
- `preprocess-v2.sh` 已扩展为 8 步，最后从 loop3 training root 生成与 launcher 路径一致的 deployment root。
- 第一次 4 帧 smoke 因错误假设同步 CSV 位于 Raw 根而在输出创建前失败；实查后移除 Raw 隐形依赖，改为读取并复制 training scene 的 `radar_ir_sync.csv`，最终 v2 smoke 通过。
- 最终回归 82 项通过：deployment 6、manifest 12、formal inference 11、多模态 inference 33、mini launcher 11、motion 9；相关 Python 编译、四份 shell 语法和 `git diff --check` 全部成功。
- 未运行全量预处理、训练或 GPU 推理；正式 deployment root 仍需等待全量 0--80 m training 数据生成后由第 8 步发布。

## 2026-08-26 Deployment observed/frame/risk 运行时安全链完成

- 新增轻量 `geometry_protocol.py`，统一严格 R/T 外参解析，使地图 CLI `--help` 无需导入 Torch/OpenMPI。
- 正式 inference 自动发布逐帧 Radar endpoint-ray observed mask 及内容收据，增加遮挡保护、LiDAR frame 声明、fresh 目录和原子文件写入。
- formal streaming map 已接入 run/mask/pose/LiDAR→body 四项门禁、`T_local_body@T_body_voxel` 坐标组合、三态风险和动态安全距离；未绑定 frame/provenance 的可选 evidence 在 formal 模式失败关闭。
- 真实 garden 4 帧只读 smoke 通过：endpoint `983/964/952/994`，observed `11727/11263/11208/11384`，所有 endpoint 都被 mask 覆盖；Radar→LiDAR SHA-256 为 `e50426da...be08d`。
- 最终聚焦回归 103 项通过：multimodal inference 36、probabilistic mapping 45、formal inference 11、mini launcher 11。相关 Python 编译、streaming CLI `--help`、三份 inference shell 语法和 `git diff --check` 全部通过。
- 未运行训练、模型前向、GPU 推理、全量数据生成或正式地图回放；未删除或覆盖数据、checkpoint、日志和实验结果。
- 已知外部依赖：仓库暂无可信 LiDAR→body 标定和真实 body→local pose 文件，因此 formal map 实数据运行保持 fail-closed。

## 2026-08-26 Mapping pose candidate 诊断启动

- 用户授权新增独立候选生成/审计脚本，不改 formal 标定目录、deployment receipt 或地图入口。
- 新增 3 项合成 RED：外参组合+双 pose-frame 假设 SLERP、超界/gap 无外推、非空/符号链接输出拒绝。首次运行因 `build_mapping_pose_candidates.py` 尚不存在而在收集阶段失败，符合预期且无输出副作用。

## 2026-08-26 Mapping pose candidate 诊断完成

- 已实现独立 alignment 诊断脚本和 4 项合成测试；新增第 4 项锁定正式外参/pose loader 必须拒绝诊断候选，RED 复现旧入口接受候选，GREEN 后 4/4 通过。
- 已生成 fresh 结果 `test/result/comparison/alignment_check/mapping_pose_contract_loop3_candidate_v1`：6432 帧中覆盖 6162 帧，4 帧早于 GT，266 帧因相邻 GT gap 超过 0.2 s 保守拒绝。
- 两套候选 CSV 的行数、时间递增、四元数归一化、`formal=false` 标记和输出哈希均通过只读自检；未修改 Data/config、deployment receipt、数据集、checkpoint、日志或旧结果。
- 最终验证通过候选协议 4/4、概率地图 45/45，相关 4 个 Python 文件编译、诊断 CLI `--help` 与 `git diff --check` 均成功；未执行模型前向、训练或 GPU 任务。

## 2026-08-26 Mapping frame 语义确认启动

- 用户授权继续确认 Radar→IMU 方向和 GT pose frame；本阶段优先只读查找原始定义，缺失时才新增独立多窗口 LiDAR 反证诊断。
- formal 硬门禁保持不变：指标胜出不能替代 TF、导出代码、标定报告或 CAD/实测轴定义，所有新结果继续标记 `formal=false`。

## 2026-08-26 Mapping frame 语义确认与多窗口反证完成

- 原始 bag 只读审计未找到 GT Odometry/static TF；官方 NTU4DRadLM 命名约定支持 `calib_radar_to_imu.txt` 为 Radar→VectorNav IMU，不应取逆，但 VectorNav IMU→airborne body 轴约定仍缺权威证据。
- 新增 `evaluate_mapping_pose_overlap.py` 与 4 项合成测试，验证 candidate/sync/manifest/所选 voxel hash、`T_local_body@T_body_lidar` 组合、fresh 输出及 `formal=false` 不可辨识边界。
- 代码审查发现并修复 v1 pose 使用 Radar timestamp、LiDAR-aligned voxel 使用 LiDAR timestamp 的接口不匹配；candidate v2 显式读取并封存 6432 行 Radar--LiDAR sync snapshot。
- loop3 LiDAR-time v2 覆盖 6165/6432，1.0 s 高转角窗口从 1244 eligible pair 均匀取 48 对；GT-as-LiDAR 48/48 残差更低，pair-median NN 中位数为 `0.4123 m`，GT-as-IMU 为 `2.3012 m`。
- 临时敏感性检查中 0.5 s 为 32/32、2.0 s 为 30/32 支持同一汇总结论。所有结果仍为 diagnostic；没有修改 formal 标定、数据、checkpoint、日志或模型，也未运行训练/GPU。
- 最终回归通过 candidate 6、overlap 5、概率地图 45，共 56 项；相关 Python 编译、两份 CLI `--help` 与 `git diff --check` 全部成功。地图回归首次被沙箱 OpenMPI socket 限制中止，随后按既有批准在沙箱外原样 45/45 通过。

## 2026-08-27 经验 LiDAR pose 离线地图合同完成

- 新增 `empirical_pose_contract.py` 与独立 builder，生成自包含直接 LiDAR→local 合同；真实结果位于 `test/result/comparison/alignment_check/empirical_lidar_pose_loop3_v1`，6432 available、6165 selected、267 uncovered。
- receipt SHA-256 为 `13977fe498acca71c64a7d2deec44467b84d327d0ae3e5b5bd659325628cb432`，直接 pose CSV SHA-256 为 `46719ffc6f94a4b55f56c0b5069cce18c3bda328850f5dde4f91e13d88917d46`；运行时只读复核通过。
- `streaming_map_update.py` 增加离线经验模式，和 airborne formal 模式互斥；地图核心支持直接 `T_local_voxel`，metadata 明确 `airborne_formal=false`、`avoidance_formal=false`、LiDAR 查询原点与 pose coverage。
- 代码审查补齐 prediction voxel provenance 隐形依赖：inference 发布逐帧内容收据，strict map 对实际消费文件执行 hash/shape/dtype/frame 预检；旧无收据推理结果保留但失败关闭。
- 共享协议重构后静态编译通过；推理接口 37/37、概率地图 46/46、经验姿态 6/6 全部通过。没有运行训练、模型 forward、GPU 推理、全量预处理或正式地图回放，也未删除或覆盖历史数据与结果。

## 2026-08-27 Radar point-count / Doppler-validity 数据合同完成

- 新增共享 Radar statistics 严格存储/加载协议，预处理对每个 occupied voxel 统计原始点数与有限 Doppler 样本数，和四通道 Radar 一起写入稀疏 NPZ。
- formal Dataset/统一训练/CD 入口要求该协议；审计摘要保留在 metadata，但明确对应增强前持久化体素且不进入模型。
- 正式 launcher 支持 `CUDA_DEVICES` 和 `PREFLIGHT_ONLY=1`，预检遍历 manifest 中全部 Radar NPZ 并重建 formal data protocol；修复了默认 `0,1` 的单卡接口不匹配。
- 4 帧 garden fresh CPU smoke 已通过 manifest、严格 Dataset、真实 IR/标定、persisted observed 与统计加载；Radar tensor 仍为 `[4,32,128,128]`。
- RED 分别复现 launcher 缺单卡/预检接口和 metadata 缺引用边界；GREEN 后 launcher 协议 11/11、Radar statistics 5/5 通过。后者首次在沙箱内受 Open MPI socket 禁止中断，沙箱外同一 Conda Python 重跑通过。
- 最终聚焦回归另通过 Dataset metadata 15/15、manifest 12/12、temporal/formal data 4/4、VAE checkpoint 24/24，以及 CD 两份直接接口脚本；相关 Python 编译、两份 shell 语法和 `git diff --check` 通过。
- 未运行全量预处理、模型 forward、GPU 或训练；未写正式训练配置，未删除或覆盖任何数据、checkpoint、日志和结果。下一步是全量 `preprocess-v2.sh`、记录 artifact SHA、服务器只读预检和正式 VAE。

## 2026-08-27 Formal v2 8 GB mini 训练入口完成

- 将 `run_formal_mini_8gb.sh` 和 formal `train_minimal.sh` 切换到当前 0--80 m v2 数据、split、data protocol、normalization 与 `formal_mini_chain_v2`，默认 8/4 帧、单阶段 1 epoch。
- checkpoint/data protocol 新增严格 mini selection 收据；统一训练和独立 CD 均按正式 split 限制实际 Dataset frame IDs，full/mini、不同样本数之间不能隐式续训或串链。
- 补齐 formal mini deployment smoke：只读取 deployment Radar/IR 与当前标定，不读取 target/LiDAR 真值；mini 权重需显式授权，运行 metadata 不冒充正式部署协议。
- 修复正式 LDM/CD launcher 的未定义 scene 目录变量，并修正 formal mini 日志中误导性的 legacy sample 数显示。
- 真实 GPU/data preflight 通过且未创建输出；最终回归通过 temporal 5/5、checkpoint 14/14、VAE 26/26、multimodal inference 38/38、mini config/safety 103/103、mini shell/protocol 13/13 及两份 CD 直接接口套件，相关 shell/Python 静态检查和 `git diff --check` 通过。
- 未启动实际 VAE/LDM/CD 训练或模型推理，未删除、覆盖或移动数据、checkpoint、日志及历史结果。

## 2026-08-27 VAE smoke 验收与 short profile

- 用户已完成默认 1 epoch VAE smoke；只读验收通过 checkpoint 协议、8/4 数据选择、0--80 m 网格、有限指标和错误日志检查，原结果继续保留在 `test/result/formal_mini_v2_80m_8gb_v1/`。
- 已在 `test/result/INDEX.md` 登记为“已验证、非推荐”的工程 smoke；没有为尚未运行的 short profile 创建结果记录。
- RED/GREEN 新增 `short_train`：只允许 VAE，固定 3 epoch、fresh 独立结果根和更严格的 60/75°C 温度门禁；未知 profile、多余参数、非 VAE short 和放宽温度均在训练启动前拒绝。
- 完整轻量回归通过：mini shell/protocol 16/16、mini config/safety 103/103，以及 shell 语法、Python 编译和 `git diff --check`。
- 真实 short 无训练 preflight 在 RTX 4070 Laptop 上通过：8188 MiB、空闲 6967 MiB、42°C；artifact 与 train/validation=8/4 协议校验通过，未创建 short 结果根，原 smoke checkpoint SHA-256 保持 `1ae08bf8...c61f50`。未自动启动 short training。
- 用户已完成 fresh 3 epoch short VAE；验收确认 3 个 epoch/checkpoint 完整、loss 与 validation IoU 持续改善、协议和模型 state 匹配，错误扫描无 OOM/NaN/异常，训练后 GPU 41°C。
- 已将 short VAE 登记为“已验证、非推荐”的工程结果；下一步仅允许在同一 short 结果根先执行 1 epoch LDM 的无训练 preflight，不把 mini 指标当作质量门槛。
- 首次 LDM preflight 虽通过，但审查确认其父 checkpoint 仅做存在性检查；已完成聚焦 RED/GREEN，把 stage/protocol/data identity、非空 state 和 CD 父哈希校验前移到零输出 preflight，等待完整回归和真实复验。
- 最终回归通过 mini shell/protocol 17/17、mini config/safety 103/103、checkpoint chain 14/14，以及两份 shell 语法、Python 编译和 `git diff --check`。
- 修复后的真实 LDM preflight 通过并打印 VAE SHA-256 `a55c0bb0...03510`；GPU 42°C、空闲 7004 MiB，`ldm/` 仍为空且 `mini_ldm_config.yaml` 不存在，未启动训练。
- 已只读评估 500 帧 × 20 epoch 请求：split 容量充足，但数据口径与长时运行设备尚需确认；未修改现有 smoke/short 参数，也未启动训练。

## 2026-08-28 500 帧 medium_train 与服务器 full 20 epoch 完成

- RED/GREEN 新增 `medium_train`：固定 400 train / 100 validation、VAE/LDM/CD 各 20 epoch、独立结果根，并拒绝帧数、epoch、温度、时长和空闲显存门槛漂移；原 `smoke`/`short_train` 行为不变。
- 收尾审查补齐 RTX 4070 Laptop 设备名门禁；伪 RTX 4090 在训练启动前被拒绝，避免把 laptop mini profile 误当服务器正式入口。
- 正式 launcher 固定 full-split `20/20/20`，生成配置时删除 mini frame 字段；`FORMAL_EPOCHS=19` 负向测试在数据读取前拒绝，20 epoch 只读预检验证 garden 4013 帧 manifest/Radar statistics 后退出，未生成配置或启动训练。
- 回归通过 mini shell/protocol 19/19、mini config/safety 103/103、checkpoint chain 14/14、两份 shell 语法和 `git diff --check`。
- 最终 laptop 真实预检通过：RTX 4070 Laptop、8188/6619 MiB、42°C、400/100、20/20/20、72°C 和 180 分钟门禁均正确；没有创建 medium 结果根，既有 smoke/short checkpoint 哈希不变。
- README、test README、mini README 和持久计划已更新。没有启动模型训练、forward/backward 或推理，也没有删除、覆盖、移动数据、checkpoint、日志和历史结果。

## 2026-08-28 medium VAE allocator 失败修复

- 用户实际 medium VAE 在 epoch 1 batch 50 的 backward 触发 CUDA expandable-segment 内部断言并退出；失败 `v1` 没有 checkpoint，目录和日志完整保留。
- 已沿 runner、训练脚本、Python 入口和 memory helper 核对真实依赖；移除 `expandable_segments`，固定 `max_split_size_mb:128`，并把 allocator 打印且记录到生成 YAML。默认 medium 结果根升级为 fresh `v2`。
- 新增 hostile 环境覆盖、fresh 结果根和正式 launcher 配置审计测试；修复过程中暴露并修正两个 heredoc import 边界问题。最终配置/安全 103/103、脚本协议 20/20、shell/Python 静态检查与 `git diff --check` 通过。
- 真实无训练 preflight 在 RTX 4070 Laptop 上因空闲显存 6375 MiB 小于 6500 MiB 安全门槛而拒绝；GPU 43°C、无计算进程，未放宽门禁、未运行 backward、未创建 `v2` 结果。

## 2026-08-28 正式训练单机 2--4 GPU DDP 改造

- 新增共享分布式训练模块，正式 VAE/LDM/CD 已接入单机 NCCL DDP、分布式 train/eval sampler、跨 rank 指标归并、rank-0 I/O 和无 `module.` 前缀 checkpoint。
- launcher 现支持 1--4 个不重复 GPU，单卡仍使用普通 Python，多卡按 stage 使用 `torch.distributed.run`；`all` 不嵌套进程组，继续串行执行三阶段。
- 补齐恢复合同、训练 sampler padding 元数据、LDM 样本身份验证噪声、CD EMA/尾部累积，并消除 Karras 模块的旧 MPI 隐形依赖与 legacy DDP forward 旁路。
- 收尾回归命中并修复 Karras 感知损失硬编码 `.cuda()` 的设备接口问题，正式训练与推理现在显式使用当前 rank-local device。
- LDM formal 验证噪声改用不依赖绝对数据根的 `scene/frame_id` 身份，并把 noise identity 纳入 checkpoint 恢复校验，支持项目迁移到服务器后保持同帧验证输入不变。
- 回归通过分布式协议、mini launcher、VAE checkpoint/sparse loss、LDM validation/vertical loss，以及 CD entrypoint/多模态接口测试；shell 语法、Python 编译和 diff 空白检查通过。完整 airborne CPU 套件因超过 60 秒主动停止，未宣称通过。
- 本项没有启动 GPU/NCCL 训练，没有删除或覆盖数据、checkpoint、日志和结果。真实 2--4 GPU 行为仍需在服务器先以短时 smoke 验证。

## 2026-08-29 正式训练 YAML 预设改造

- 已把正式 20/20/20 epoch、每阶段每场景 3210/774 帧、4 卡默认列表和 normalization 固定 SHA 写入 YAML；launcher 解析后再接受阶段或通用环境变量临时覆盖。
- 已实现 `formal_stage_selection_v1`，记录每阶段 ordered-prefix 的实际数量和有序 frame ID SHA-256；三个阶段 checkpoint 写入该字段，同阶段 resume 在加载模型/优化器前拒绝身份漂移。
- launcher 运行时 override 同步记录实际 `cuda_devices/num_gpus/world_size`，DDP batch 仍由共享安全计划解析，不由 YAML 帧数旁路。
- 新增 5 项 YAML/覆盖/选择测试；通过 launcher/DDP 37 项、VAE checkpoint 26 项、LDM validation 5 项及 CD 入口回归。shell 语法、Python 3.8 模块导入和 diff 空白检查纳入收尾。
- 真实 `PREFLIGHT_ONLY=1` 在 `Radar-Diffusion` Conda 环境中通过全部 4013 帧只读数据门禁，未创建 override、checkpoint 或训练结果，也未启动 GPU forward/backward。
## 2026-08-29：阶段 2 完成

- 完成 LDM observed-mask 数学层、trainer 传参、formal fail-closed、验证指标和 checkpoint 协议改造。
- `test_ldm_observed_supervision.py`：6/6 PASS。
- `test_ldm_vertical_structure_loss.py`：83/83 PASS。
- `py_compile` 通过；未运行长训练、完整预处理或全量推理。

## 2026-08-29：阶段 3 完成

- 完成 DDP SyncBatchNorm、normalization checkpoint 协议、CD EMA parameters/buffers 和 formal resume 门禁。
- DDP、CD、多模态 CD、VAE/LDM checkpoint、LDM validation、formal YAML 短测试均通过。
- 未运行双卡长训练；当前验证边界是结构/状态合同测试，服务器首次正式训练仍需以 preflight 加短 smoke 确认 NCCL/SyncBatchNorm 运行环境。

## 2026-08-29：阶段 4C 完成

- 已完成 CD validation 数据链、online/EMA 确定性 observed-domain 选优、checkpoint/resume 协议和推理部署权重解析。
- CD entrypoint、多模态 CD、推理接口 40 项、默认 YAML、Python 静态编译及 `git diff --check` 均通过。
- 未运行训练、GPU forward、完整推理或覆盖历史 checkpoint；下一步处理绑定 checkpoint 的 validation-only threshold artifact。

## 2026-08-29：阶段 4 完成

- 完成固定 seed/CUDA 同步计时、observed-mask evaluator、CD online/EMA 部署选优和 validation-only threshold artifact 全链。
- 聚焦回归通过：artifact 3 项、formal inference 13 项、runtime 4 项、multimodal inference 40 项、LDM validation 5 项、VAE/LDM checkpoint 26 项及 CD 两组接口；shell/Python 静态检查和 diff 空白检查通过。
- 未启动正式训练或推理；旧 checkpoint 没有 threshold sweep，不能直接生成新 formal artifact，需要按新代码重新训练对应 LDM/CD 阶段。

## 2026-08-29：阶段 5 完成

- 已完成 prediction artifact v2、formal authoritative observed、prediction-only 地图消费和 DEM 单位合同改造。
- 概率地图 50/50、经验位姿合同 6/6、推理接口 40/40 PASS；覆盖 mask 外 0.9 概率保持 unknown、合法收据中的超域概率 fail closed，以及辅助通道不改变 belief/DEM。
- 未运行训练、GPU 推理或全场景地图回放；新协议行为目前由小数组、临时目录和接口测试证明。

## 2026-08-30：阶段 6 完成

- 完成 body/LiDAR 锚定的整体素 rolling window，拆分固定 source evidence range 和动态 destination local bounds，并收据化最终范围/移动次数。
- 完成逐帧 local trajectory artifact、制动距离内三态走廊查询、首个风险点 CSV 字段与 artifact/records SHA-256 收据。
- 正式/经验协议分别升为 v5/v3，`map_run.json` 明确仅离线回放，不再将严格数据合同误声称为机载避障 formal。
- 回归通过：概率地图 54/54、经验位姿 6/6，相关 Python 编译与 `git diff --check` 通过。未运行训练、GPU 推理、全量地图回放或 ROS/PX4。

## 2026-08-30：阶段 7A 完成

- 完成 Radar 分字段 finite 聚合、float64 防溢出、statistics v2 稀疏存储及 dataset/launcher payload-policy 一致性检查。
- 保留 v1 读取和现有 formal-v2 预检兼容；新生成帧固定写入 v2，未改写任何既有预处理数据或 artifact。
- 回归通过：Radar statistics 7/7、VAE/checkpoint 26/26，相关 Python 编译、launcher shell 语法和 `git diff --check` 均通过；未运行全量预处理或训练。

## 2026-08-30：阶段 7B 完成

- 完成 strict field semantics artifact、权威 evidence 内容校验、解包 layout/semantics 交叉绑定、PointCloud v1 sidecar、NaN 缺失值和 Doppler 正方向补偿。
- 预处理 CLI/policy/manifest 已绑定 schema 与 layout SHA；现有 velocity-none v2 保持兼容，fixed/recorded 未验证语义时 fail-closed。
- 回归通过：字段 schema 8/8、运动协议 9/9、Radar statistics 7/7、airborne Doppler 定点 2/2，以及相关 Python 编译和 `git diff --check`。未读取真实 bag、未运行全量预处理或训练。

## 2026-08-30：阶段 7C 完成

- 完成逐场景 extraction receipt、逐帧/逐 bag 失败传播、关键模态即时与最终门禁，以及跨帧 PointCloud layout 漂移拒绝。
- 回归通过：pointcloud/receipt 11/11、timestamp alignment 8/8，相关 Python 编译与 `git diff --check`；全部输出仅在临时目录，未读取真实 bag 或改写数据集。

## 2026-08-30：阶段 7D 与阶段 7完成

- 发布独立 `formal_data_v3` validator/builder/loader，绑定 statistics-v2、field schema/layout、extraction receipt 和 Radar 物理合同；训练 Python 与 launcher 在 v3 时强制逐字段 statistics-v2，v2 保持 v1/v2 兼容。
- 预处理 policy 接入 complete receipt 和 verified schema 身份；新增 fresh `preprocess-v3.sh`，在创建输出前校验 schema evidence，并拒绝覆盖 v2 或任意已有输出。
- 回归通过：temporal/formal-data 7/7、checkpoint chain 14/14、VAE/checkpoint 27/27、field/schema/receipt 12/12、motion 9/9，以及 shell/Python 静态检查和 `git diff --check`。仅运行缺 schema 的前置失败和临时目录短测，未执行全量解包、预处理或训练。

## 2026-08-30：阶段 8A--8C 完成

- 已完成 Dataset 单帧合同、未实现参数 fail-fast、Karras history 隔离和多模态 forward 的签名能力分派；不再用运行时 `TypeError` 猜测接口。
- 已把 CD 对外语义收敛为“LDM initialization + CD EMA consistency”，保留空的 `teacher_model_path` legacy 别名用于迁移冲突检查，删除默认/mini 配置中的无效蒸馏旋钮。
- 新增共享 CD training/sampling receipt；checkpoint 保存和恢复、正式 chain preflight、CD inference 都消费同一份 sigma/rho/EMA/scale 身份。
- 短回归通过：Dataset 16/16、多模态推理 42/42、checkpoint chain 14/14、CD entrypoint，以及相关 Python/shell 语法检查。未启动训练、GPU forward、完整推理或数据预处理。

## 2026-08-30：阶段 8D 核心完成

- 已修复 saved evaluator 的 observed mask“只验证不消费”，并发布正式 evaluation protocol、正式指标集合、指标域和 aggregation 收据。
- 已把 legacy 点云评价/图像对比/质量诊断与唯一正式 saved-prediction evaluator 明确分流，`compare.sh` 同时移除本机绝对路径隐形依赖。
- 回归通过：formal inference evaluator 15/15、task metrics 4/4、formal YAML 5/5、mini launcher 21/21、checkpoint 15/15、多模态 inference 44/44 与 CD entrypoint；Python/shell 静态检查和 `git diff --check` 通过。
- 未运行训练、GPU 推理或全量评价；下一步只做阶段 8 跨模块最终短回归与工作树审查。

## 2026-08-30：阶段 8 与顺序修复计划完成

- 阶段 8 最终跨模块回归通过：Dataset 16/16、checkpoint chain 15/15、多模态 inference 44/44、formal evaluator 15/15、task metrics 4/4、formal YAML 5/5、mini launcher 21/21，以及 CD training 两组接口测试。
- 本阶段涉及的 Python 模块均通过 `py_compile`，8 个 shell 入口通过 `bash -n`，`git diff --check` 通过。
- 八阶段代码修改计划至此全部完成。没有启动长训练、GPU 推理、全量预处理或全量评价，也没有删除/覆盖数据集、checkpoint、训练日志和实验结果。
