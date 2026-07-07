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
    - [x] Leave formal multi-epoch LDM retraining for an explicit experiment run
  - [ ] Re-evaluate all gate metrics on one independent validation/test set
  - [ ] Start CD distillation only if a future LDM task/visual quality gate passes

## Notes

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
| Final regression referenced missing `test_unified_cd_entrypoint.py` | 1 | Located and ran the real `test/test_cd_training_entrypoints.py`; it passed. |
| Phase 10 combined regression hit OpenMPI socket denial in sandbox | 1 | Re-run the same short verification outside the socket-restricted sandbox. |
| LDM structure-loss test hit the same OpenMPI sandbox socket denial | 1 | Run the focused test outside the sandbox before Task 1 review. |
