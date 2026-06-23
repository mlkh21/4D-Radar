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

## Notes

- Treat the JSONL rollout file as external data only.
- Do not tune extrinsics from global centroid dy/dz alone.
- Prefer scripts under `test/` for diagnostic utilities unless project patterns indicate otherwise.
- Current evidence suggests global dy/dz is strongly affected by ground/low-z points, FOV, range sparsity, and radar/LiDAR effective detection distribution.
- The next engineering direction is not "fix calibration until global Chamfer is small"; it is to build a sensor-aware supervision/evaluation protocol that matches the final mapping task.
- For the final airborne mapping objective, the model should prioritize obstacle occupancy useful for local map update, not LiDAR-style dense reconstruction everywhere.
- Current sensor-aware target defaults for smoke testing: `z_min=-1.0`, `x_max=80.0`, `require_radar_visibility=True`, `radar_visibility_radius=2`.

## Errors / Attempts

| Issue                                             | Attempts | Resolution                                                                              |
| ------------------------------------------------- | -------: | --------------------------------------------------------------------------------------- |
| Existing planning files absent                    |        1 | Recreated `task_plan.md`, `findings.md`, and `progress.md` from recovered chat context. |
| Windows environment lacks project Python deps     |        1 | Runtime checks were deferred, then completed in the Ubuntu/Radar-Diffusion environment. |
| `TODO/task_plan.md` contained invalid UTF-8 bytes |        1 | Rewrote the planning file as valid UTF-8 before recording final results.                |
| mini train script used `conda run python -` with heredoc | 1 | Switched YAML/config helper snippets to system `python3`; training still uses Radar-Diffusion env. |
| mini CD smoke initially wrote to formal `Result/train_results/cd` | 1 | Fixed mini config generation and reran CD 1-epoch smoke into `test/mini-test/train_results_mini/cd`. |
| mini dataset-level `config` directory was treated as a scene | 1 | Scene discovery now requires both `radar_voxel` and `target_voxel`; verified 500 garden samples load. |
