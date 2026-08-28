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
        - [x] Review the written v11 design specification before implementation planning
        - [x] Write and self-review the v11 TDD implementation plan
        - [x] Implement and smoke-test the v11 curriculum task by task
          - [x] Task 1: add and review the pure epoch curriculum function
          - [x] Task 2: wire trainer, metrics, and checkpoint metadata
          - [x] Task 3: wire mini config and guarded V11 runner
          - [x] Task 4: run focused regression and bounded smoke
        - [x] Train V11 for three epochs and run the unchanged fixed 32-frame selector
        - [ ] Pass the fixed 32-frame gate before authorizing 500-frame evaluation (V11 failed 2/5)
        - [x] Diagnose V11 epoch2 threshold calibration on the same fixed 32 frames
        - [x] Evaluate fixed `0.99` and calibrated `0.925` on 32 independent loop3 frames
        - [ ] Pass the independent-scene structure/calibration gate before 500-frame inference or CD
        - [x] Audit garden/loop3 Radar、target、IR、range/height 与 Doppler 分布（各 500 帧）
        - [x] Confirm fixed 50 m/s preprocessing corrupts NTU raw Doppler and block full-data retraining
        - [ ] Replace implicit fixed-speed compensation with explicit `none/fixed/recorded` protocol
        - [ ] Rebuild and audit a 32-frame corrected preprocessing subset before any full regeneration
        - [ ] Add neighborhood or temporal Doppler variance because per-voxel variance is mostly zero
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
| Scene-audit unit test imported the PyTorch/OpenMPI calibration chain | 2 | Moved calibration/frustum imports into the real audit function; pure unit tests then passed without MPI initialization. |
| P0-01 adjacency diagnostic did not pass the shell frame count into `conda run` | 1 | Changed from an unexported environment lookup to an explicit command-line argument. |
| Ignored dataset files produced a zero frame count in the P0-01 diagnostic | 2 | Tried `rg --files -uu`, then stopped after the third total diagnostic failure; the static call chain already proves frame-level random splitting. |

## Test Directory Organization

- [x] 审计 `test/AGENTS.md`、`test/README.md`、`test/result/INDEX.md` 和全部相关源码/结果目录
- [x] 按职责移动功能脚本和 22 个回归/协议测试到分类目录
- [x] 保持 mini-test、未知结果、锁目录、临时目录和历史叶目录不变
- [x] 修复移动后的 Python import、Shell 引用、默认结果路径和 Git 忽略规则
- [x] 更新 `test/README.md` 和 `test/result/INDEX.md`
- [x] 执行整理后的最小验证，不运行训练、完整推理或全量评估

## 2026-07-15 TODO/26 优先问题修复

- [x] 按顺序读取适用 `AGENTS.md`、`CODEX_HANDOFF.md` 并检查 Git 现场
- [x] 以 GB18030 只读转码方式完整审阅 `TODO/26-7-15.md`
- [x] 对照实际代码核验第一阶段 P0 问题及其调用链
- [x] 与用户确认本轮单一修复边界，避免同时打包多个研究协议变更
- [x] 提出 P0-01 单场景连续时间块设计并取得口头批准
- [x] 写入、自审并独立提交 P0-01 书面规格
- [x] 用户复核书面规格
- [x] 编写并自审 P0-01 RED/GREEN 实现计划
- [x] 选择计划执行方式后开始 RED
- [x] 按 RED/GREEN TDD 实施获批修复并执行小范围验证
- [x] 说明监督信号、体素数量和指标协议影响，禁止长时间训练或完整推理

## 2026-07-15 `26-7-15.md` 分阶段修复续作

- [x] 用 GB18030 只读转码审阅问题清单，不改写原始审计文件
- [x] 检查工作区、近期提交、现有 TODO 记录和 `test/AGENTS.md`
- [x] 确认并完成此前只写完 RED/计划的 P0-01
- [x] 为第一阶段下一项 P0-06 完成根因、调用链、工作样例与最小假设审计
- [x] 提出 P0-06 隔离策略并取得独立诊断脚本设计批准
- [x] 写入并复核 P0-06 设计规格与 RED/GREEN 实施计划
- [x] 按 TDD 实施单一修复并运行聚焦回归
- [ ] 逐项重复上述流程；不自动运行长训练、完整预处理或全量推理
- [x] 完成 P0-01/P0-06 的监督信号、体素数量与指标协议说明，并同步三份 TODO 记录

### P0-06 完成状态

- [x] RED：正式 inference 与 mini launcher 的旧 oracle 入口测试按预期失败
- [x] GREEN：移除正式 adaptive 参数/分支并加入迁移提示，固定阈值与正常离线评价保持可用
- [x] RED/GREEN：新增独立 oracle 诊断脚本，输出点云、CSV 与 `deployable=false` JSON
- [x] 安全保护：缺 target、错误 shape、非空输出目录均在写结果前失败
- [x] 聚焦回归：25/25 通过；Python 编译、shell 语法和差异格式检查通过
- [x] 未运行训练、预处理、完整推理或全量评价；未暂存或提交
- [x] 下一项单独统一 `sweep_occ_threshold.py` 的 validation 切分协议，不能与 P0-06 混改

### P0-06 独立诊断依赖边界加固（2026-07-20）

- [x] 复查独立 oracle 诊断的 import 调用链，确认其不应加载正式 inference/阈值扫描入口
- [x] RED：新增测试锁定正式入口路径不得出现在诊断脚本中
- [x] GREEN：提取轻量 occupancy 诊断辅助模块并切断正式入口耦合
- [x] 回归 oracle 阈值、点云、CSV/JSON 和非空输出保护协议
- [x] 记录监督信号、体素数量和指标协议均不变；未运行长任务且未暂存/提交

### P0-03 多普勒运动补偿协议修复与代码审查（2026-07-20）

- [x] 追踪 shell、parser、场景控制器、worker 和体素化函数的速度调用链
- [x] RED：新增 none/fixed/recorded、时间匹配和坐标变换契约测试
- [x] GREEN：默认 none，显式 fixed，严格 recorded 速度表及逐帧解析
- [x] 修复 Radar/LiDAR 对齐后的速度坐标接口，记录速度源 hash 和协议元数据
- [x] 代码审查并修复直接文件执行的同名包导入冲突、非法值和空帧接口问题
- [x] 聚焦回归与静态验证通过；不重生成数据、不训练、不推理、不暂存/提交

### P0-05 LiDAR 未观测空间与 free evidence 修复（2026-07-20）

- [x] 追踪 D-S 融合、地图更新和 streaming 入口，确认空白 voxel 被误当作 free evidence 的调用链
- [x] RED：覆盖无 mask unknown、显式 free mask、shape fail-fast、同目录 mask 排除和稀疏 `.npz` 读取
- [x] GREEN：无 mask 采用 occupied-only observed，显式 mask 才产生 free evidence，并暴露 `unknown_mass`
- [x] 接入 streaming `--observed_mask_dir`、CSV/快照 unknown 统计和输入路径校验
- [x] 代码审查：拒绝 symlink mask 目录，避免输出目录副作用；保持旧 `update_from_voxel` 位置参数兼容
- [x] 聚焦回归 12/12、Python 编译、CLI 帮助、`git diff --check` 和空暂存区检查通过
- [x] 接入 LiDAR 射线 observed mask、VAE 可见区域损失和增强同步接口
- [x] 修复旧 trainer/model 三参数接口兼容，并完成方向去重性能审查
- [x] 记录监督有效区域变化、体素数量/指标协议影响；未重生成数据、训练、推理、全量地图更新或暂存/提交

### 阈值扫描 validation 协议续修

- [x] 检查正式训练、阈值扫描和既有测试的切分调用链
- [x] 确认校准数据范围与成功标准
- [x] 比较可选实现边界并取得用户批准
- [x] 写入、自审并复核设计规格
- [x] 按 RED/GREEN TDD 修改脚本与既有测试
- [x] 执行聚焦回归并记录监督、体素数量和指标可比性影响
- [ ] 下一项按第一阶段顺序设计 dataset manifest，固化场景与预处理版本

### Dataset Manifest 根因审计

- [x] 追踪预处理、Dataset、训练与推理入口的数据根目录
- [x] 核验现有 `preprocess_policy.json`、协议审计和实验 hash 的能力边界
- [x] 核验真实场景目录的混用证据与现有 policy 内容
- [x] 确认 manifest 的生成/校验入口和旧数据兼容策略
- [x] 比较方案并取得用户批准
- [x] 写入规格和实施计划，完成 manifest 核心 RED/GREEN（7/7）
- [x] 按 TDD 接入 CLI、预处理器和正式 launcher
- [x] 执行聚焦回归并记录真实旧数据 fail-closed 结果
- [ ] 下一项单独修复正式推理的 sensor-aware/真实 IR 路径，并拆分部署与离线评价入口

### 正式真实 IR 与部署/评价解耦

- [x] 追踪 launcher、checkpoint 模态、IR/标定加载和评价调用链
- [x] 明确正式入口缺失 IR/标定时采用 fail-closed，不使用 mock/Radar-only 降级
- [x] 比较部署/评价拆分方案并取得用户批准
- [x] 写入、自审并复核设计规格
- [x] 写入并自审 RED/GREEN 实施计划
- [x] 按 RED/GREEN TDD 小步实施，不运行完整推理
- [x] 执行聚焦回归并记录监督、体素数量和指标协议影响

### Errors / Attempts

| Issue | Attempts | Resolution |
| --- | ---: | --- |
| 首次三文件追加补丁因 `progress.md` 尾部上下文不精确而整体未应用 | 1 | 改用各文件精确尾部上下文重新应用，不重复原补丁 |
| `iconv -f gb18030 ... | rg` 在扫描完整审计文件时报告第 3 行不可转换 | 1 | 不重复相同管道；后续改用容错只读转码并用已知行段定位编号 |
| 首次 GREEN 在收集测试时因旧 helper 直接导入而 ImportError | 1 | 提前执行既定测试导入重构，改为直接导入新 helper 后重跑 |
| 完成前复验在沙箱内触发 OpenMPI 本地 socket 限制 | 1 | 在沙箱外重跑完全相同的短测试/编译/差异检查，全部 exit 0 |
| 设计文档提交请求未获批准，文件曾单独暂存 | 1 | 未创建提交；已将该文件撤出暂存并确认暂存区恢复为空，继续未提交实施 |
| 首次新增 P0-06 实施计划的补丁正文缺少一行 `+` 标记 | 1 | 补丁整体未应用；修正新增标记后成功写入并自审 |
| 诊断首次 GREEN 的 top-k 三项测试只得到 `k-1` 点 | 1 | 复现实验证明 float64 前驱被比较规则舍入回 float32 原值；改用 prediction dtype 的 `nextafter` 后 6/6 通过 |
| 阈值时间块 Task 1 首次 GREEN 剩余主入口 TypeError | 1 | 计划错误地把删除旧关键字调用放在 Task 2；将该必要调用链修改前移后再复验 |
| 完成检查组合命令因 `rg` 无匹配返回 1 | 1 | 无未完成 Step 正是预期；拆分差异、暂存区和未完成项检查，避免把预期无匹配误报为整体失败 |
| 严格 manifest 计划自审 `rg` 使用不支持的 look-around | 1 | 文件未改动；拆为普通 `rg` 模式重新检查，不重复该正则 |
| Manifest Task 1 首次 RED 找不到顶层 namespace package | 1 | 测试直接执行未加入项目根；先修测试 `sys.path`，再重跑确认因目标模块缺失失败 |
| Manifest 最终组合验证触发 OpenMPI 本地 socket 限制 | 1 | 未改变测试范围；在沙箱外重跑完全相同的短测试/编译/shell/diff 命令，23/23 聚焦测试通过 |
| Manifest 完成记录的首次多文件补丁上下文不匹配 | 1 | 补丁整体未应用；按实际文件尾部分开更新三份 TODO 和实施计划 |
| 用 GB18030 转码实际 UTF-8 的 `26-7-15.md` 得到乱码 | 1 | `file` 确认文件为 UTF-8；后续直接按 UTF-8 只读，不重复错误转码 |
| 正式 IR 实施计划首次多文件补丁上下文不匹配 | 1 | 补丁整体未应用；改为先单独新增计划，再按实际尾部分别更新三份 TODO |
| Task 4 首次 GREEN 的调用顺序断言误命中 evaluator 存在性检查 | 1 | 根因定位到测试搜索首个变量引用；收窄为实际 `conda run` 调用，不修改正确的 manifest-first 生产流程 |
| Task 5 首次真实 manifest 验证使用旧 CLI 参数 | 1 | CLI 错误明确要求 `--scene_dir/--expected_scene`；按实际接口重跑，不涉及数据写入 |
| P1-01 首次 RED 时间戳协议测试因 helper/新索引接口缺失而全部 ERROR | 1 | 新增标准库时间戳 helper 与 `generate_scene_indices`，随后 5/5 GREEN |
| 直接执行时间戳索引脚本误解析同名预处理模块并触发 cv2 缺失 | 1 | 改为按 `__package__` 选择相对/同目录导入，直接 `--help` 不再加载重依赖 |
| 初版索引容差校验使用不存在的 `os.path.isfinite` | 1 | 改用 `math.isfinite`，重新通过协议测试和静态编译 |
| v2 全量脚本首次运行在旧 receipt-time Raw 上被 30ms 门禁拒绝 | 1 | 不重复原命令；bag/header 全量审计后改为独立 Raw 重解包与物理采样窗口协议 |
| P1-01 首次 GREEN 大补丁因目标注释文本与实际文件不一致而未应用 | 1 | 补丁未产生修改；按实际行号拆成小补丁后成功应用 |
| 最终静态组合命令未使用 fail-fast，解包器 `--help` 缺 pandas 虽失败但组合退出码仍为 0 | 1 | 不把该次组合命令记为通过；移除非必要依赖后单独复验入口，并在最终检查中分别收集退出码 |
| P1-05 第一轮核心补丁同时匹配新 helper 与旧函数上下文失败 | 1 | 该补丁整体未应用；读取当前函数顺序后拆为 helper、fusion、update 三个小补丁 |
| P1-05 streaming GREEN 小体素输出 XY 维互换 | 1 | 定位为旧 `to_xyzc` 用维度大小猜布局；改为可显式声明 `xyzc/czxy`，auto 只接受无歧义形状 |
| P1-05 显式 layout 补丁重复指定同一目标文件 | 1 | 补丁整体未应用；合并为同一文件的一组 update 后成功应用 |
| P1-05 首次静态组合检查的 CLI `--help` 触发 OpenMPI socket 失败 | 1 | 不把组合命令记为全通过；切断 `cm/__init__.py` 重依赖后单独复验 CLI 成功 |
| P1-05 最终记录补丁两次因同文件重复操作或 hunk 逆序失败 | 2 | 两次均整体未应用；按实际顺序和单文件操作重新写入 |
| P1-02 RED 暴露硬编码 K、IR helper 不接收标定和同步函数重复 | 1 | 新增 thermal 标定/去畸变/共享补偿协议，再完成 GREEN |
| 代码审查发现 audit_dataset_protocol.py 仍复制旧 K | 1 | 改为复用 `CalibrationProvider` 的实际 K，避免第三套投影参数 |

### P0-01 完成状态

- [x] RED：三个时间块契约测试因缺少新 API 按预期失败
- [x] GREEN：实现连续训练前缀/验证后缀并接入正式训练入口
- [x] 聚焦回归：21/21 通过
- [x] 静态验证：无旧 helper 引用，`py_compile` 与 `git diff --check` 通过
- [x] 记录监督信号、体素数量、样本数量与指标可比性影响
- [x] 保持现有脏工作区和暂存区，不自动提交

### P1-06 正式 VAE/LDM/CD checkpoint 链

- [x] 审计正式权重目录：当前只有旧 CD，正式 VAE/LDM 路径缺失；历史 archive 权重与正式 CD 不是同一协议链
- [x] 写入协议设计与实施计划，明确不伪造/覆盖旧权重
- [x] 新增 `formal_chain_v1` 核心校验：普通文件、阶段、网格、latent、父 SHA-256、多模态关键权重
- [x] 新增独立 `diagnose_checkpoint_chain.py`，支持只读 validate 和 CPU strict construct，不执行 forward
- [x] VAE/LDM/CD 新保存 payload 写入协议、网格、融合 config 和父 checkpoint hash
- [x] 三个正式生成入口增加全链门禁，unified 不再缺阶段静默跳过
- [x] 完成 checkpoint-chain、VAE/CD payload、正式入口聚焦回归与当前旧链只读诊断；正式 VAE/LDM/CD 重训仍作为后续显式长任务

### P1-01 多传感器时间戳对齐与容差

- [x] 阅读 bag 解包、Radar/LiDAR 索引、IR 匹配和 preprocess shell 的完整调用链
- [x] RED：锁定 header 优先、最近邻超限拒绝、索引 delta 落盘和不产生半成品输出的测试契约
- [x] GREEN：新增标准库时间戳 helper，统一 header/receipt 回退和最近邻容差
- [x] GREEN：按数值时间排序并原子写入 Radar-LiDAR 索引、绝对/带符号 delta；预处理严格校验同步记录
- [x] GREEN：IR 使用独立容差，写入逐帧 `radar_ir_sync.csv`；解包和脚本参数完成接线
- [x] 代码审查：修复直接文件执行的同名模块遮蔽、旧索引绕过、输出目录副作用和非法时间戳接口
- [x] 聚焦回归与静态验证通过；未重生成数据、训练、推理、全量评价或暂存/提交
- [x] 记录同步协议对帧成员、监督配对和指标可比性的影响；`dt_sync` 保持显式 legacy 语义，后续按物理符号约定消费 signed delta

### P1-02 Thermal 标定与 IR 投影几何统一

- [x] 追踪 CalibrationProvider、Dataset、inference、投影层和审计脚本的 K/D/S 与同步补偿调用链
- [x] RED：锁定原始尺寸/K/D 解析、resize 后 K 缩放、去畸变和共享补偿函数契约
- [x] GREEN：Provider 统一解析 `calib_cam_thermal.txt`，输出缩放 K、D、S 和来源 metadata
- [x] GREEN：训练/推理共用 IR resize+undistort 与 legacy sync compensation；严格真实 IR 缺失 S/K/D 时拒绝
- [x] 代码审查：移除 audit 脚本的重复硬编码 K，并保留 mock/旧测试兼容边界
- [x] 聚焦回归和真实配置只读解析通过；未重生成数据、训练、推理或暂存/提交
- [x] 记录对 IR 像素采样、监督/体素/指标可比性的影响

### P1-03 PointCloud2 字段 schema 固定化

- [x] 追踪 `unpack_rosbag.save_pointcloud → radar_pcl/*.npy → voxelize_pcl_airborne_optimized` 的固定列依赖
- [x] RED：覆盖缺 intensity、缺 Doppler 时的列位置保持和 schema 元数据输出
- [x] GREEN：按字段名/别名读取 PointCloud2，固定输出 `[x,y,z,intensity,doppler]`，缺失特征显式补零
- [x] GREEN：在点云目录原子写入 `pointcloud_schema.json`，记录来源字段、映射、缺失列、shape 和 dtype
- [x] 代码审查：保留旧 PointCloud v1、Livox 分支与下游五列接口，不改变已有数据文件
- [x] 聚焦回归、静态编译和差异检查通过；未解包真实 bag、重建体素、训练或推理
- [x] 记录字段协议对强度/Doppler 监督通道、体素数量和指标可比性的影响

### P1-04 Radar 物理通道规范化与统计方差重采样

- [x] 审计预处理四通道生成、Dataset crop/resize、训练/推理配置和现有场景统计
- [x] 比较稳健统计量的生成/消费边界、旧数据兼容策略和方差合并方案并取得设计批准
- [x] 写入并自审设计规格与 RED/GREEN 实施计划
- [x] RED：锁定强度/Doppler 规范化协议、二阶矩方差合并和 metadata 传播
- [x] GREEN：按批准方案小步实现，未运行完整预处理或训练
- [x] 代码审查：消除训练/推理隐式默认、shape/单位接口不匹配和错误 legacy 降级
- [x] 聚焦回归、静态检查与轻量只读统计通过
- [x] 记录对监督信号、体素数量、checkpoint/指标可比性的影响
- [x] 为训练场景 garden 生成并验收 4013 帧正式 artifact，写入匹配的 `86.8 m/s` 与固定网格配置
- [x] 将训练/生成/评价 launcher 切换到 candidate 数据和独立 `formal_p1_04_full120_86p8_v1` 结果协议，默认拒绝隐式续训
- [ ] 显式长任务：在独立结果根从头训练正式 VAE/LDM/CD；不得把旧 checkpoint 自动补签为新协议

### P1-01 真实数据时间容差续修（2026-07-22）

- [x] 只读统计 garden/loop3 的 Radar、LiDAR 帧率与最近邻时间差分布
- [x] 区分启动/结束边界无重叠帧与正常重叠区间的异步采样偏差
- [x] 先写聚焦 RED 测试，再最小修改索引协议或执行脚本参数（RED 3 项已确认）
- [x] 运行时间戳、manifest、运动协议聚焦回归和真实数据只读索引验证
- [x] 记录帧成员、监督配对、体素数量和指标可比性影响

### Codex VS Code 历史会话读取修复（2026-08-20）

- [x] 核对官方 OpenAI Codex 文档入口、扩展版本和专用日志
- [x] 只读验证会话 JSONL、索引和 `state_5.sqlite` 完整性
- [x] 使用扩展捆绑 app-server 复现提供方过滤导致的旧历史缺失
- [x] 在 `/tmp` 候选配置上通过严格解析、旧会话列表和全文读取回归
- [x] 备份并修复用户级 `~/.codex/config.toml`，未修改任何历史会话文件
- [x] 记录修改、验证和回滚路径；由用户在本轮结束后重载 VS Code 窗口

### P1-05 移动平台局部地图更新（2026-08-20）

- [x] 审计 `prediction voxel → streaming_map_update → SlidingProbabilisticGridMap` 的坐标、时间和输出调用链
- [x] 设计保持现有 2D 调用兼容的位姿变换与分高度层最小接口
- [x] RED：锁定已知机体位移/旋转后的静态障碍对齐、缺失位姿拒绝和高度层保持
- [x] GREEN：接入逐帧时间戳与 `T_local_body`，输出可审计的分层 occupancy/unknown 状态
- [x] 代码审查：处理坐标系方向、边界、接口不匹配和旧 2D consumer 兼容
- [x] 运行聚焦回归与静态检查，记录地图单元数量、监督/指标和实时内存影响
- [x] 动态层审计：确认现有 Doppler 单位、推理输出和可复用的 mask/provenance 接口
- [x] RED：锁定显式动态 evidence、pose warp、静态/动态分离、快速衰减和缺失 sidecar 行为
- [x] GREEN：增加可选动态三态层及严格 sidecar metadata，未启用时不分配额外三维状态
- [x] 代码审查与回归：消除单位猜测、帧覆盖、shape、坐标系和旧调用接口不匹配

### P1-07 LDM 验证与 CD 训练语义（2026-08-20）

- [x] 审计 LDM/CD 的训练、验证、best checkpoint、教师目标和既有协议字段调用链
- [x] RED：锁定独立 validation、best 选择依据、checkpoint 恢复及 CD 语义声明
- [x] GREEN：为 LDM 接入独立验证与可审计 best 选择，并准确标记 EMA consistency 训练
- [x] 代码审查与回归：消除 train/val 接口、旧 checkpoint、resume 和正式入口不匹配

### Radar normalization 零 IQR 续修（2026-08-20）

- [x] 只读核验候选 manifest、Radar 四通道统计和 resize 前后 intensity 分布
- [x] 定位零 IQR 根因，明确统计退化策略及其物理/监督影响
- [x] RED/GREEN：补充退化分布测试并实施最小修复
- [x] 代码审查与聚焦回归，提供只重跑步骤 6 的安全命令

### 正式训练协议切换（2026-08-20）

- [x] 验收 garden/loop3 manifest、正式 normalization artifact 和一个真实 IR/标定训练样本
- [x] RED/GREEN：默认 YAML 精确绑定 candidate 数据、32×128×128 网格、full120 pc range、86.8 m/s 和 artifact SHA-256
- [x] 训练输出隔离到 `formal_p1_04_full120_86p8_v1`，已有非空阶段目录默认拒绝；仅 `ALLOW_RESUME=1` 允许同协议恢复
- [x] 三个正式生成入口及独立评价入口统一 candidate preprocessed/Raw、checkpoint 根和带协议标识的输出目录
- [x] 代码审查修复 mini legacy 配置继承正式 artifact 的互斥冲突，并同步 README
- [x] 完成聚焦回归、shell 语法、路径存在性、配置/artifact 对照和差异检查
- [x] 修复直接训练入口缺失仓库包根、失效 fallback 与模块双重身份问题，并增加脱离工作目录的 `--help` 回归
- [x] 修复真实 preprocess policy 中 JSON null 导致的 batch 拼接失败，统一 train/val、standalone CD 与条件推理 collator
- [x] 无损归档零 epoch 失败日志，保持非空结果门禁且恢复 fresh VAE 输出路径
- [ ] 由用户显式启动正式 VAE 长训练，完成后依次训练 LDM 与 CD

### 8 GB 单卡正式协议 mini 训练（2026-08-21）

- [x] 审计现有 mini 脚本的旧数据根、legacy Radar 单位、输出复用与硬件负载风险
- [x] RED：覆盖正式 candidate/artifact、独立结果根、短时 epoch/sample 配置和温度预检
- [x] GREEN：最小扩展现有 mini 入口，不复制新训练实现、不覆盖历史结果
- [x] 对 RTX 4070 Laptop 给出单阶段执行与停止条件，不自动启动训练
- [x] 运行聚焦单元测试、Bash 语法和无训练配置预检
- [x] 记录监督信号、样本/体素数量、指标可比性与笔记本热负载影响

### 外部审查第一批修复与范围决策（2026-08-21）

- [x] 冻结旧 v1，建立 checkpoint/data protocol v2 和显式 legacy 诊断边界
- [x] 消除 `.tmp_train_dataset`、场景猜测、标定 fallback 和训练/推理 provenance 隐形依赖
- [x] 修复 LiDAR→Thermal 投影方向、严格 R/T/K 校验和逐帧 signed 时间补偿职责
- [x] 建立 training/deployment manifest profile，并交叉绑定部署输入、标定与 checkpoint 身份
- [x] 对 garden 4013 帧执行 0--80/80--120 m 只读监督、ray、叠加、时序与 IR 审计
- [x] 用户确认 formal v2 使用 0--80 m，80--120 m 在地图保持 unknown
- [x] 实现逐帧持久 observed mask，formal Dataset 禁止 occupied-only fallback
- [x] 实现唯一 temporal split/purge artifact，normalization 只读取 train frame IDs
- [x] 实现 `formal_data_protocol.json` 与 train-only normalization 的生成/重建校验接口，所有正式输出使用 fresh v2 路径
- [x] 完成 4 帧 fresh 预处理 smoke；未启动 8 GB mini 或全量训练
- [ ] 在具备磁盘与运行条件后执行全量 0--80 m v2 重建，生成 split/data protocol/normalization 实物并填写 launcher 的固定 artifact SHA-256
- [x] 实现严格 deployment-profile v3 数据视图生产/校验链；正式推理禁止把 training root 直接冒充 deployment root
- [ ] 全量 0--80 m training 数据生成后执行第 8 步，发布 loop3 正式 deployment root；4 帧 smoke 已完成但不能替代全量数据
- [ ] full training、服务器传输和正式评价继续保持独立确认，不在范围选择时自动执行

### Deployment observed/frame/risk 运行时安全链（2026-08-26）

- [x] 正式 inference 从 Radar endpoint 生成逐帧 observed mask，绑定 mask/frame/标定 SHA 并保护近端遮挡后 unknown
- [x] 正式 map 强制 inference run、mask、body→local pose 和 LiDAR→body 外参，在产生输出前完成全帧预检
- [x] 地图坐标链改为 `T_local_body@T_body_voxel`，快照和 run metadata 同时保存三段变换
- [x] 风险查询实现 `clear/obstacle/unknown` 三态、动态制动安全距离和 unknown fail-closed
- [x] 正式 inference/map 拒绝非空或符号链接输出目录，metadata 原子发布
- [x] 完成 103 项聚焦回归、4 帧真实 Radar smoke 与静态检查
- [ ] 获取并验收真实 LiDAR→body 外参和 body→local pose，后续在 fresh 目录运行 formal map 回放
- [ ] 下一子阶段持久化 Radar point-count/Doppler-validity sidecar，再设计 `UncertaintyHead` 升级与 checkpoint 迁移；此项需训练条件后独立确认

### Mapping pose candidate 诊断（2026-08-26）

- [x] 审计 loop3 GT/Radar 时间范围、现有外参格式和 formal map 输入合同
- [x] 建立外参组合、双 pose-frame 假设、SLERP、无外推和 fresh 输出 RED
- [x] 实现 `test/diagnostics/alignment/build_mapping_pose_candidates.py`，所有候选强制 `formal=false`，正式 loader 按内容拒绝
- [x] 生成 loop3 独立诊断结果：6162 帧覆盖，4 帧早于 GT，另有 266 帧因 GT gap 超过 0.2 s 拒绝插值
- [x] 代码审查修复 Radar-time pose 与 LiDAR-time voxel 的接口不匹配，v2 封存 Radar--LiDAR sync 并覆盖 6165 帧
- [ ] 确认 GT/export frame 和 VectorNav IMU→airborne body 轴约定后，才能另行生成 formal 标定/位姿合同

### Mapping frame 语义确认与反证（2026-08-26）

- [x] 只读定位原始 bag/TF、GT 导出代码、标定来源和 CAD/安装轴线索
- [x] 官方命名约定确认矩阵方向为 Radar→VectorNav IMU；冻结 GT-as-IMU/body 与 GT-as-LiDAR 两种可辨识假设及外参消去边界
- [x] 新增多窗口静态 LiDAR 一致性 RED/GREEN 诊断，结果强制 `formal=false`
- [x] 在 loop3 小规模高转角窗口比较残差与体素重合度，不运行训练/GPU；LiDAR-time v2 为 48/48 支持 GT-as-LiDAR
- [ ] 只有权威 frame 定义与独立反证一致后，才另行规划 formal receipt

### 经验 LiDAR pose 离线地图合同（2026-08-27）

- [x] 沿 `inference_run → streaming_map_update → SlidingProbabilisticGridMap` 审计姿态、预测、mask 与 metadata 调用链
- [x] 发布自包含 `empirical_lidar_pose_contract_v1`，绑定 LiDAR-time candidate、overlap、sync、外参与逐帧 pose，禁止外推
- [x] 增加与正式 body 链互斥的 `T_local_voxel` 直通接口，经验链不伪造 body pose
- [x] 离线经验模式固定 `airborne_formal=false`、`avoidance_formal=false`，保留 unknown/risk fail-closed
- [x] inference 为实际 prediction voxel 发布逐帧内容收据，strict map 在创建输出前验证 hash/shape/dtype/frame 顺序
- [x] 完成静态编译、37 项推理接口、46 项地图和 6 项经验位姿回归；未运行训练、模型 forward 或 GPU 任务
- [ ] 取得权威 GT/export frame 与 IMU→airborne body 定义后，另行构建正式机载 pose/extrinsic receipt；经验合同不得直接升级
- [ ] 具备正式 checkpoint 与 deployment v3 数据后重新运行 inference，再在 fresh 目录执行离线经验地图回放

### Radar point-count / Doppler-validity 正式数据合同（2026-08-27）

- [x] 沿预处理 Radar 点云聚合、稀疏 NPZ、manifest、Dataset 与正式训练入口审计真实调用链
- [x] RED/GREEN：持久化与 coords 对齐的 `point_count`、`doppler_valid_count` 和严格协议字段，保持原四通道 Radar 数值不变
- [x] formal Dataset 全帧 fail-closed 预检，并把增强前审计摘要保留在 metadata，明确 `model_consumed=false`
- [x] 修复正式 launcher 的隐形双卡依赖，增加可配置 `CUDA_DEVICES` 和无训练 `PREFLIGHT_ONLY=1`
- [x] 完成 4 帧真实 CPU smoke、聚焦单元测试、launcher shell/静态协议测试和代码审查
- [ ] 在具备磁盘条件后运行 `preprocess-v2.sh`，生成全量 v2 数据、split、normalization、formal data protocol 与 deployment v3
- [ ] 记录新 normalization SHA-256，在服务器通过只读预检后显式启动正式 VAE；8 GB 笔记本不运行全量长训练

### Formal v2 8 GB 单卡 mini 训练（2026-08-27）

- [x] 审计旧 full120/v1 mini、统一训练、独立 CD、推理和 checkpoint 身份调用链
- [x] RED/GREEN：从正式 split 确定性选择 8/4 帧并持久化 `mini_selection`，拒绝 full/mini 或不同子集混用
- [x] 将保护 runner 接线 0--80 m v2 training root、artifact、formal data protocol 和 fresh 结果根
- [x] 补齐 strict deployment mini smoke，显式区分 `formal_mini_smoke` 与正式部署权重
- [x] 修复独立 CD 门禁分叉、正式 LDM/CD launcher 未定义 scene 路径和误导性样本日志
- [x] 完成真实无训练 preflight、聚焦回归、shell/Python 静态检查和文档更新
- [x] 由用户在机器通风且 GPU 冷却后显式运行并验收 1 epoch VAE smoke
- [x] 增加独立 `short_train` VAE profile：3 epoch、fresh 结果根、60/75°C 温度门禁
- [x] 完成 short profile 全量轻量回归和真实无训练 preflight；未创建输出或启动训练
- [x] 由用户显式完成 fresh 3 epoch VAE short training
- [x] 验收 short VAE 并确认设备冷却；只授权同一结果根上的 1 epoch LDM 工程 smoke
- [x] 修复并完成 LDM 无训练 preflight：父 checkpoint/data identity 通过，未创建输出
- [ ] 由用户决定是否显式启动同一 short 结果根的 1 epoch LDM；不用 mini 指标替代正式结果
- [x] 确认“500 帧”为 400 train/100 validation 总计 500，并新增 RTX 4070 Laptop 独立 20 epoch `medium_train` profile
- [x] 中型 profile 不覆盖 smoke/short；完成真实零训练 preflight，并固定设备名、55/72°C、6500 MiB 空闲显存和 180 分钟门禁
- [x] 正式服务器 launcher 使用 3210/774 full split，移除 mini 帧限制并固定 VAE/LDM/CD 各 20 epoch
- [ ] 由用户选择合适时间显式启动 laptop `medium_train` VAE；验收 checkpoint/验证指标和设备状态后再决定 LDM

### medium VAE CUDA allocator 断言修复（2026-08-28）

- [x] 只读保留并核对失败 `v1` 日志、配置、PyTorch/CUDA 版本和 allocator 传播链
- [x] RED/GREEN 移除 `expandable_segments`，统一 laptop/server allocator，并把实际配置写入 YAML
- [x] 将失败 `v1` 登记为无 checkpoint 的诊断现场，默认结果根升级为 fresh `v2`
- [x] 完成 103 项配置/安全回归、20 项脚本协议回归与静态检查
- [ ] 关闭部分图形程序、空闲显存恢复到至少 6500 MiB 后，先跑无训练 preflight，再跑受限的短 backward 诊断
- [ ] 短诊断超过原失败点且温度正常后，由用户显式启动 fresh `v2` 的 20 epoch VAE
