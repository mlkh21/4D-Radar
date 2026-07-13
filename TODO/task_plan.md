# Task Plan

## Goal

- 任务：基于机载传感器融合的障碍物地图构建与场景地图更新
  考虑机载传感器噪声大，点云稀疏等问题，开展4D毫米波雷达点云融合滤波、红外与毫米波雷达点云稠密点云生成等方法研究；同时，考虑机载传感器误差和里程计误差，优化概率栅格障碍物地图表征方式；考虑机载飞行平台内存约束，设计实时感知障碍物地图与数字高程地图的融合更新方法。
  解释--参考论文P. R. Florence, J. Carter, J. Ware and R. Tedrake, "NanoMap: Fast, Uncertainty-Aware Proximity Queries with Lazy Search Over Local 3D Data," 2018 IEEE International Conference on Robotics and Automation (ICRA), Brisbane, QLD, Australia, 2018, pp. 7631-7638, doi: 10.1109/ICRA.2018.8463195.
  部分内容参考浙大高飞前期成果https://github.com/ZJU-FAST-Lab/Radar-Diffusion
  思路可以借鉴当前3D占用网络（ Occupancy Network）相关内容，如：MetaOcc: Surround-View 4D Radar and Camera Fusion Framework for 3D Occupancy Prediction with Dual Training Strategies，RadarOcc，LiCROcc等。
- 约束条件
  飞行速度35m/s-70m/s；
  飞行器动力学模拟基于JSBsim；
  整套仿真系统基于ros框架，以服务方式发布航迹点信息，以action方式定义所设计的控制器，与PX4实现硬在环仿真集成。
- 思路：4D毫米波点云稠密化生成后与红外摄像机融合形成栅格地图，构建基于概率栅格图滑动更新方法与不确定性传播的地图构建与更新框架，在通过基于D-S证据理论的感知地图与先验DEM动态融合机制与数字高程图进行融合更新
- 问题：

## Current Phase

- [x] Recover prior context from chat log and project files
- [x] Inspect existing alignment utilities and outputs
- [x] Implement missing shared visibility / nearest-neighbor / BEV IoU evaluation
- [x] Run loop3 metrics on Ubuntu/Radar-Diffusion environment
- [x] Summarize whether dy is caused by calibration, ground filtering, FOV, or distribution mismatch
- [x] Phase 2: Define sensor-aware training/evaluation protocol for airborne obstacle mapping
- [x] Phase 3: Implement filtered/shared-visible target generation and metrics in the main pipeline
  - [x] Add sensor-aware target generation utility
  - [x] Validate generated target dataset with the training dataset loader
  - [x] Add airborne egomotion-aware voxelization and Doppler variance channel tests
  - [x] Add offline IR metadata loading path and multimodal fusion entry for LDM training
  - [x] Add the new banded/shared-visible metrics into formal inference reports
- [ ] Phase 4: Retrain VAE/LDM/CD with the corrected protocol and compare against the old baseline
  - [x] Close CD training/inference interface for legacy and multimodal checkpoints
  - [x] Add `unified_train.py --mode cd` and route formal CD launcher through it
  - [x] Add mini CD training/inference smoke path
  - [ ] Run full formal VAE/LDM/CD retraining on the complete sensor-aware dataset
  - [ ] Produce baseline comparison table for old LDM, new LDM, CD 1-step, and CD 4-step
- [ ] Phase 5: Connect densified radar outputs to uncertainty-aware sliding occupancy/DEM map update
  - [x] Add Doppler/range-aware reliability to offline probabilistic map updates
  - [x] Add speed-band and odometry-covariance controls to offline map updates
  - [x] Add streaming map smoke metrics for obstacle precision/recall/uncertainty
  - [ ] Add ROS1 service/action bridge after offline retraining metrics are stable
- [x] Phase 6: Correct mini-model over-density and calibrate uncertainty before formal retraining
  - [x] Add task-aware occupancy-threshold sweep and select a deployable global threshold
  - [x] Fix per-frame task metric columns in `inference_metrics.csv`
  - [x] Add a learnable model-error uncertainty head while retaining Doppler/range physical confidence
  - [x] Train uncertainty with latent Gaussian NLL and report calibration metrics
  - [x] Re-run targeted unit tests and a saved-output threshold calibration smoke
- [x] Phase 7: Restore tree/obstacle structure before formal retraining
  - [x] Replace hard radar-visible target deletion with structure-preserving LiDAR supervision
  - [x] Correct radar-to-thermal extrinsic direction and `(Z,X,Y)` voxel-axis projection
  - [x] Add a configurable near-field high-resolution model grid for `0-40m`
  - [x] Add a VAE reconstruction upper-bound diagnostic for train/test scenes
  - [x] Run unit tests and a small VAE overfit smoke before any LDM/CD retraining
- [ ] Phase 8: Recover the sparse occupancy VAE upper bound (in progress)
  - [x] Unify occupancy logits/probability semantics across training, diagnosis, and inference
  - [x] Replace occupancy MSE with BCE+Dice while masking continuous-channel supervision
  - [x] Select VAE checkpoints by validation occupancy IoU and persist the exact VAE config
  - [x] Pass the 32-frame reconstruction overfit gate before any longer experiment
  - [x] Add a manual Z=64 near-field VAE upper-bound runner without auto-starting training
  - [ ] Compare ultra-lightweight z4, lightweight z8, and lightweight z16 one variable at a time
  - [x] Reach validation IoU >= 0.50 and recall >= 0.65 on the 500-frame near-field experiment
  - [x] Resume LDM training only after the VAE upper-bound gate passes
  - [ ] Resume CD training only after the corrected LDM threshold/evaluation gate passes
  - [x] Task 3 review: unify standalone CD VAE loading and checkpoint grid metadata
  - [x] Task 3 review: require explicit fallback for metadata-free diagnostic/inference checkpoints
  - [x] Task 3 final review: synchronize same-epoch best-loss/best-IoU checkpoint payloads
  - [x] Task 3 final review: derive conditional/unconditional latent shapes from the loaded VAE
  - [x] Task 3 quality review: propagate non-4 latent dimensions through LDM/CD/inference
  - [x] Task 3 quality review: resume VAE scheduler and atomically save checkpoints
  - [x] Final review: align LDM decoded occupancy auxiliaries with VAE activation semantics
  - [x] Final review: restore metadata-free fallback checkpoints with legacy raw/MSE semantics
  - [x] Final review: reject sparse compatible-subset multimodal loads in strict inference
  - [x] Final review blocker: reject empty or critical-key-incomplete strict inference states
- [x] Phase 9: Correct the LDM occupancy-threshold evaluation protocol (CD held by visual gate)
  - [x] Crop source target voxels from `0-120m` to the model `0-40m` range before resize
  - [x] Select the threshold on the deterministic validation split using near-field BEV task metrics
  - [x] Re-scan the existing 500 saved LDM outputs without rerunning inference
  - [x] Re-evaluate 500 frames once with the fixed threshold and inspect 10 raw-LiDAR overlays
  - [x] Evaluate CD gate: HOLD because tree-structure visual criterion failed
- [ ] Phase 10: Quantify and recover vertical tree structure
  - [x] Implement height coverage, top-height, vertical-connectivity, and trunk-region recall metrics
  - [x] Evaluate the VAE reconstruction upper bound with the new structure metrics
  - [x] Select the recovery branch: VAE passes, so keep the current VAE/grid protocol
  - [ ] Add vertical-structure/height-distribution supervision to LDM and retrain it
    - [x] Implement differentiable height-distribution and vertical-continuity losses
    - [x] Integrate independent weights, once-per-batch decoding, and component logging
    - [x] Add mini-script controls and run one short finite-gradient smoke
    - [x] Add structure-preserving empty-column density control and decoded/uncertainty weight controls
    - [x] Add a guarded Z=64 LDM v5 runner with existing-result overwrite protection
    - [x] Add top-height target-voxel supervision and a guarded Z=64 LDM v6 runner
    - [x] Add strict IR calibration metadata, audit coverage, and mock IR/calib training logs
    - [x] Add IR-feature-aware fusion gate and expose IR frustum masks
    - [x] Add optional IR-frustum occupancy/top-height supervision for LDM v7
    - [x] Add IR condition ablation diagnostic before any v7 long run
    - [x] Leave formal multi-epoch LDM retraining for an explicit experiment run
  - [ ] Re-evaluate all gate metrics on one independent validation/test set
    - [x] Add saved-output LDM vertical-structure evaluation against LiDAR target voxels
    - [x] Restrict saved-output evaluation to `*_voxel.npy` and ignore LDM sidecars
    - [x] Preserve `[Z,X,Y,C]` prediction layout when `Z == 4`
    - [x] Align raw target voxels to the model crop/resize protocol before vertical evaluation
    - [x] Run the vertical-structure report on `ldm_near40_500_vertical_v1`
    - [x] Compare the report with raw-LiDAR HTML visual observations before changing loss weights
    - [x] Add a one-click v2 LDM vertical experiment runner
    - [x] Add decoded density/precision regularization to control over-dense LDM predictions
    - [x] Run a v4 LDM experiment with height recovery plus density control
    - [x] Revert voxel-column visualization and restore original raw-LiDAR point-cloud comparison
    - [x] Try a less blunt structure-preserving density control after visual review
    - [x] Run Z=64 LDM v5 with uncertainty NLL disabled and empty-column density enabled
    - [x] Run Z=64 LDM v6 with top-height supervision enabled
    - [x] Run dataset audit and IR ablation on the current candidate checkpoint
    - [x] Run guarded Z=64 LDM v7 with IR-frustum supervision after audit/ablation justified it
    - [x] Re-scan v6/v7 thresholds and compare vertical metrics at the same 0.99 threshold
    - [x] Run post-v7 real/zero/mock IR ablation on three training frames
    - [x] Add target-aware IR ablation metrics to distinguish useful structure recovery from simple densification
    - [x] Run the 32-frame validation target-aware IR ablation and compare real/zero/mock summaries
    - [x] Decide the minimal v8 top-height correction after target-aware IR ablation
    - [x] v8 Task 1: add an above-target top-overshoot loss aligned with the strict top-height metric
    - [x] v8 Task 2: add balanced IR-frustum positive/negative occupancy supervision to control IR-driven over-density
    - [x] v8 Task 3a: add a guarded Z64 runner with safe scratch paths and atomic experiment locking
    - [x] v8 Task 3b: run v8 and compare v7/v8 on 32-frame ablation plus 500-frame task/vertical metrics
    - [x] v9 Task 1: add recall-constrained threshold selection and report both quality/safety operating points
    - [x] v9 Task 2: run one-variable short screens for top-overshoot `0.05 -> 0.02` and IR-negative `0.02 -> 0.01`
    - [x] v9 Task 3: train only the winning screen for 10 epochs and re-run the fixed 32/500-frame protocol
    - [x] v9 Task 4: add validation task/structure checkpoint selection before any CD distillation
    - [x] v10 Task 1: evaluate saved v9 epoch checkpoints on a fixed validation subset and select by task/structure score rather than training loss
    - [x] v10 Task 2: if no epoch passes, add a column-balanced objective that protects occupied-column recall while suppressing empty-space false positives
    - [ ] v10 Task 3: repeat the fixed 32/500-frame gate only for the selected checkpoint or one isolated loss candidate
      - [x] Add separately logged positive/negative column-existence losses using stable Z-axis logmeanexp aggregation
      - [x] Keep new weights disabled by default and replace the old empty-column density term only in the v10 runner（已完成规格复审：allowlist、EXP 子路径审计、持锁后空目录复检、固定协议、非空 checkpoint、fresh scratch/config 无破坏链路）
      - [x] Run finite-gradient smoke, then isolated 3-epoch A/B screens（A/B 均完成固定 32 帧验证，v10-A epoch3 胜出）
      - [x] Train only one winning candidate for 10 epochs and select checkpoint by fixed validation structure metrics（已完成，但未复现 screen 门槛；定位到训练随机种子未固定）
      - [x] Verify two short runs with the same `training_seed` reproduce training metrics/checkpoint outputs before retraining（CSV 完全一致，checkpoint 最大张量差 `7.04e-08`）
      - [x] Re-run one seeded 3-epoch v10-A validation gate before authorizing another 10-epoch run（固定 32 帧验证未通过，禁止直接续训 10 epoch）
      - [x] Run a bounded seeded loss-calibration screen that weakens empty-column suppression and restores occupied-column recall（C/D 均未通过，停止直接续训）
        - [x] Extend the guarded v10 runner with isolated C (`pos=0.03, neg=0.01`) and D (`pos=0.02, neg=0.005`) variants
        - [x] Train seeded C/D for three epochs without changing the VAE, split, or other loss weights
        - [x] Evaluate every C/D epoch with the identical fixed 32-frame selector
      - [ ] Require the calibrated candidate to pass the same fixed 32-frame gate before any 500-frame evaluation
        - [x] Revisit the optimization strategy instead of continuing one-dimensional column-weight tuning
        - [x] Approve the epoch-wise v11 column curriculum design
        - [ ] Review the written v11 design specification before implementation planning
        - [ ] Implement and smoke-test the v11 curriculum only after specification review
      - [ ] Require both quantitative and raw-LiDAR 3D basic-structure gates before CD
    - [ ] Consider thermal edge/semantic supervision only if v8 precision/top metrics remain bottlenecked
    - [ ] Consider IR-backbone pretraining only after the loss-aligned v8 baseline is stable
    - [ ] Promote v7/v8 weights into default config only after the final experiment protocol is selected
  - [ ] Start CD distillation only if a future LDM task/visual quality gate passes

## Notes

## 2026-07-13 Result Directory Organization Continuation

- [x] Move the confirmed named result leaves into the existing VAE/LDM/comparison categories.
- [x] Group the unnamed root-level LDM/VAE/CD/scratch artifacts into one archive leaf without deleting files.
- [x] Preserve existing v10-D lock directories beside their owning experiment; do not relocate active lock paths.
- [x] Update generated config/report paths, live defaults, README, AGENTS, INDEX, and legacy plotting inputs.
- [x] Run only static/minimal post-move verification; do not train, infer, or run full evaluation.

- Treat the JSONL rollout file as external data only.
- Do not tune extrinsics from global centroid dy/dz alone.
- Prefer scripts under `test/` for diagnostic utilities unless project patterns indicate otherwise.
- Current evidence suggests global dy/dz is strongly affected by ground/low-z points, FOV, range sparsity, and radar/LiDAR effective detection distribution.
- The next engineering direction is not "fix calibration until global Chamfer is small"; it is to build a sensor-aware supervision/evaluation protocol that matches the final mapping task.
- For the final airborne mapping objective, the model should prioritize obstacle occupancy useful for local map update, not LiDAR-style dense reconstruction everywhere.
- Current sensor-aware target defaults for smoke testing: `z_min=-1.0`, `x_max=80.0`, `require_radar_visibility=True`, `radar_visibility_radius=2`.
- Phase 7 supersedes the hard visibility default for structure reconstruction: radar visibility must be a confidence/evaluation signal, not a destructive occupancy mask.

## Errors / Attempts

| Issue                                             | Attempts | Resolution                                                                              |
| ------------------------------------------------- | -------: | --------------------------------------------------------------------------------------- |
| Existing planning files absent                    |        1 | Recreated `task_plan.md`, `findings.md`, and `progress.md` from recovered chat context. |
| Windows environment lacks project Python deps     |        1 | Runtime checks were deferred, then completed in the Ubuntu/Radar-Diffusion environment. |
| `TODO/task_plan.md` contained invalid UTF-8 bytes |        1 | Rewrote the planning file as valid UTF-8 before recording final results.                |
| mini train script used `conda run python -` with heredoc | 1 | Switched YAML/config helper snippets to system `python3`; training still uses Radar-Diffusion env. |
| mini CD smoke initially wrote to formal `Result/train_results/cd` | 1 | Fixed mini config generation and reran CD 1-epoch smoke into `test/mini-test/train_results_mini/cd`. |
| mini dataset-level `config` directory was treated as a scene | 1 | Scene discovery now requires both `radar_voxel` and `target_voxel`; verified 500 garden samples load. |
| Tree structure absent despite improved task metrics | 1 | Root cause isolated to destructive target masking, coarse model grid, IR projection geometry errors, VAE underfit, and cross-scene mini evaluation; Phase 7 addresses these before formal retraining. |
| Standalone `sensor_aware_target.py` disappeared during Phase 7 | 1 | Confirmed its vectorized functionality now lives in the integrated preprocessing script; continue against the integrated entry and remove stale test/pyc dependence rather than restoring the deleted file. |
| `python -m unittest test...` could not import `test/` modules | 1 | Directly executed test files because `test/` is not a package in this repository. |
| 1-sample VAE smoke failed in sandbox with OpenMPI socket errors | 1 | Reran the same short smoke outside the sandbox; it passed in 0.7s training time. |
| 32-frame lightweight VAE failed before epoch 1 because GroupNorm used 32 groups for 24/72 channels | 1 | Made normalization select the largest valid divisor without changing channel/checkpoint shapes; 10 focused tests and two-stage review passed. |
| GroupNorm quality re-review agent hit its usage limit | 1 | Dispatched a fresh independent reviewer and completed the review without changing implementation scope. |
| Final regression referenced missing `test_unified_cd_entrypoint.py` | 1 | Located and ran the real `test/unit/test_cd_training_entrypoints.py`; it passed. |
| Phase 10 combined regression hit OpenMPI socket denial in sandbox | 1 | Re-run the same short verification outside the socket-restricted sandbox. |
| LDM structure-loss test hit the same OpenMPI sandbox socket denial | 1 | Run the focused test outside the sandbox before Task 1 review. |
| Visualization code-quality reviewer hit usage limit | 1 | Dispatched a fresh independent reviewer; the replacement review passed. |
| Target-aware IR smoke hit OpenMPI socket denial in sandbox | 1 | Re-ran the same 1-frame/1-step diagnostic outside the socket-restricted sandbox; it completed and wrote all reports. |
| v8 Task 3 safety-fix subagent hit its usage limit after writing RED tests | 1 | A fresh implementation agent resumed from the existing tests; later reviews closed the symlink, direct-run, and concurrency-lock gaps. |
| v8 final regression hit OpenMPI socket denial in sandbox | 1 | Re-ran the exact short test suite outside the socket-restricted sandbox; all 111 focused tests passed. |
| Full v8 evaluation through `conda run` stopped without a complete report | 2 | Used the same Conda environment Python directly in one PTY session; the 500-frame metrics and 32-frame GPU ablation completed successfully. |
| Paired-statistics helper could not import pandas | 1 | Replaced pandas with Python CSV + NumPy/SciPy; no project dependency was added. |
| Two final read-only analysis subagents reached their usage limit | 1 | Continued from the completed CSV/report artifacts and computed the paired statistics locally. |
| Focused unittest module import failed because `test/` is not a package | 1 | Ran the established direct test file; all 96 tests passed after the protocol update. |
| Sandbox could not communicate with the NVIDIA driver | 1 | Ran the explicitly approved bounded GPU training/evaluation commands outside the sandbox. |

## Test Directory Organization

- [x] 审计 `test/AGENTS.md`、`test/README.md`、`test/result/INDEX.md` 和全部相关源码/结果目录
- [x] 按职责移动功能脚本和 22 个回归/协议测试到分类目录
- [x] 保持 mini-test、未知结果、锁目录、临时目录和历史叶目录不变
- [x] 修复移动后的 Python import、Shell 引用、默认结果路径和 Git 忽略规则
- [x] 更新 `test/README.md` 和 `test/result/INDEX.md`
- [x] 执行整理后的最小验证，不运行训练、完整推理或全量评估
