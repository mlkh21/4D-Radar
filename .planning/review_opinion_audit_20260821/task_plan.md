<!-- 文件功能：记录外部审查意见的核验状态与待用户确认的分阶段实施计划。 -->
# 外部审查意见核验与实施计划

## 目标和边界

- 目标：修正会破坏标定、监督、部署输入、unknown/free 和正式实验可复现性的缺陷，再恢复 mini 验证；全量训练单独确认。
- 本轮：用户已确认 0--80 m formal v2 分支，并完成实施 3 的数据协议补齐与 4 帧 fresh smoke；不重建全量数据、不启动训练。
- 保留：现有 candidate、normalization artifact、epoch 16 VAE checkpoint、日志和结果全部保留，仅标为 legacy/diagnostic。
- 实施原则：每一阶段先写 RED 小测试，再做最小修改；任何正式输出使用新协议 tag 和新目录，拒绝覆盖旧结果。

## 核验阶段

### 阶段 1：完整提取审查意见

- [x] 读取附件全部 488 行并建立 P0/P1/P2/P3 清单
- [x] 区分事实、风险推断、建议和易变化训练状态
- **状态：** 完成

### 阶段 2：当前工作树证据核验

- [x] 核验 P0 调用链、配置、数据 metadata 和训练状态
- [x] 核验 P1/P2 及当前 HEAD 已实现的部分能力
- [x] 标注存在、部分存在、不成立和证据不足项目
- **状态：** 完成

### 阶段 3：方向和依赖审查

- [x] 判断建议是否最小且与正式协议兼容
- [x] 识别监督信号、体素、checkpoint 和指标可比性影响
- [x] 加入审查遗漏的 `.tmp_train_dataset` 标定丢失和 provenance 缺口
- **状态：** 完成

## 已确认实施计划

### 实施 0：冻结旧协议并建立 v2 身份

**状态：完成。**

目的：保证后续修复不会误续训或覆盖 `formal_p1_04_full120_86p8_v1`。

准备修改：

- `diffusion_consistency_radar/config/default_config.yaml`
- `diffusion_consistency_radar/launch/train_unified.sh`
- `diffusion_consistency_radar/checkpoint_chain.py`
- `diffusion_consistency_radar/scripts/diagnose_checkpoint_chain.py`
- `test/unit/test_checkpoint_chain_protocol.py`
- `test/unit/test_vae_checkpoint_protocol.py`

步骤：

1. 新建 formal v2 协议 tag；旧 v1 只允许诊断和只读加载，不允许成为 v2 的 resume 起点。
2. checkpoint 新增 `dataset_manifest/split/target_policy/observed_mask/calibration` 等协议 hash；VAE绑定监督和 split，LDM再绑定 IR/normalization/VAE，CD绑定 LDM。
3. checkpoint validator 支持 `--target-stage vae|ldm|cd`，只验证目标阶段及父链；修复 LDM 被 CD 阻塞。
4. 正式 checkpoint 只允许 `weights_only=True`；历史可信文件需显式 legacy 诊断开关。

验收：父 hash、stage、网格或协议任一不一致均在模型构建前失败；原 v1 文件内容和目录不变。

### 实施 1：消除场景和标定隐形依赖

**状态：完成。**

目的：先阻止正式 LDM 在 mock 外参上训练，并解除 `.tmp_train_dataset` 对 cwd/目录猜测的依赖。

准备修改：

- `diffusion_consistency_radar/cm/dataset_loader.py`
- `diffusion_consistency_radar/dataset_manifest.py`
- `diffusion_consistency_radar/scripts/dataset_manifest.py`
- `diffusion_consistency_radar/launch/train_unified.sh`
- `diffusion_consistency_radar/launch/inference_ldm.sh`
- `diffusion_consistency_radar/launch/inference_cd.sh`
- `diffusion_consistency_radar/launch/inference_uniified.sh`
- `diffusion_consistency_radar/config/data_loading_config.yml`
- `test/unit/test_dataset_protocol_metadata.py`
- `test/unit/test_dataset_manifest_protocol.py`
- `test/unit/test_formal_inference_protocol.py`

步骤：

1. Dataset 显式接收 `scene_names` 和 `calibration_dir`；正式路径禁止通过 cwd 猜测、禁止 mock fallback。
2. launcher 直接使用正式 dataset root + scene allowlist，不再创建/删除临时场景软链接。
3. manifest schema 增加明确 `profile=training|deployment`：training 保持 Radar/LiDAR/target/IR 完整；deployment 只要求 Radar/IR，但必须绑定 policy、同步记录、外参和内参 hash。
4. 正式启动前加载一帧做 real IR/calibration preflight，任何 mock 比例都直接失败，而非 epoch 结束后 warning。

验收：只有 Radar+IR 的 deployment fixture 通过；缺 target 的 training fixture 失败；正式训练 fixture 的 `is_mock_ir/is_mock_calib` 必须全为 false。

### 实施 2：修复 IR 坐标链和时间补偿职责

**状态：完成。**

目的：确保模型体素 frame、投影外参 source frame 和时间参考一致。

准备修改：

- `NTU4DRadLM_pre_processing/NTU4DRadLM_pre_processing.py`
- `NTU4DRadLM_pre_processing/NTU4DRadLM_timestamp_index.py`
- `NTU4DRadLM_pre_processing/motion_protocol.py`
- `NTU4DRadLM_pre_processing/preprocess-v2.sh`
- `diffusion_consistency_radar/cm/dataset_loader.py`
- `diffusion_consistency_radar/cm/multimodal_fusion.py`
- `test/unit/test_preprocessing_motion_protocol.py`
- `test/unit/test_thermal_calibration_protocol.py`
- `test/unit/test_airborne_multimodal_refactor.py`

步骤：

1. policy/manifest 明确 `voxel_coordinate_frame`；`align_to=lidar` 时只允许 LiDAR→Thermal 外参进入投影。
2. 对 R/T/K 做 shape、finite、旋转正交、det 和方向语义校验；记录直接外参与组合外参残差。
3. 当前三外参不闭环，因此先把直接 LiDAR→Thermal 作为候选权威并用重投影可视化人工验收；闭环残差记录为 calibration audit，不能无依据把组合外参当真值。
4. 删除 Dataset/inference 的固定 `0.01 m` camera-x 修改；所有时差补偿只在预处理完成。
5. Radar--LiDAR 和 Radar--IR 均保存 signed delta；只把非参考传感器移动到明确参考时刻，禁止同时同向移动两份点云。
6. 当前 garden 无可信平移速度时保持 `raw_mean_doppler`，禁止 compensated 声明；未来 recorded velocity+attitude 输入齐全时才启用补偿。
7. IR 原始 PNG 已是 8-bit 3 通道，不实施虚假的“恢复 16-bit”；另做 raw 3-channel 与 grayscale 的只读统计/消融，再决定是否重建 IR 表示。

验收：合成点经 LiDAR→Thermal 投影与解析外参一致；正负 delta 方向测试通过；`velocity_mode=none` metadata 必须为 raw；源码不再存在 legacy 0.01 m 正式调用。

### 实施 3：冻结 80--120 m 监督、observed 和 split 协议

**状态：本轮授权范围内已完成；全量 artifact 生成与训练仍待具备条件后执行。**

目的：在重建数据前决定“120 m 有监督”还是“80 m 外 unknown”，避免只改 loss 掩盖 latent/标签矛盾。

准备修改：

- `NTU4DRadLM_pre_processing/NTU4DRadLM_pre_processing.py`
- `NTU4DRadLM_pre_processing/preprocess-v2.sh`
- `diffusion_consistency_radar/cm/dataset_loader.py`
- `diffusion_consistency_radar/scripts/build_radar_normalization.py`
- `diffusion_consistency_radar/scripts/unified_train.py`
- `diffusion_consistency_radar/cm/vae_3d.py`
- `diffusion_consistency_radar/config/default_config.yaml`
- `test/unit/test_vae_sparse_occupancy_loss.py`
- `test/unit/test_ldm_vertical_structure_loss.py`
- `test/unit/test_ldm_validation_protocol.py`
- 新增 `test/unit/test_observed_supervision_protocol.py`

步骤：

1. 先输出全场景 0--80/80--120 m 的 LiDAR点密度、ray覆盖、跨帧稳定性和外参叠加审计，不修改数据。
2. 设置硬决策门：
   - 若 80--120 m LiDAR 通过可信度验收：target范围改为120 m并重建；
   - 若不通过：正式模型/评价范围收敛到可信范围，120 m 外只作为地图 unknown；若仍要模型输出120 m，则需新增显式 observed/unknown 表示并重训架构，不能把 0 当 free。
3. 预处理持久化逐帧 observed mask，并把监督可信范围应用到 mask；formal Dataset 禁止运行时 occupied-only fallback。
4. VAE loss、LDM全部 decoded occupancy/FP/mass/density/column/IR-negative loss及验证指标统一按 observed 统计；positive target 始终保留。
5. 明确记录 latent MSE 的局限：若选择 120 m+unknown 架构分支，需要 observed-aware target/encoder，而非只给辅助损失传 mask。
6. 生成唯一 split artifact，记录按时间排序的 train/val/purge frame IDs、时间范围和 hash；purge 时长先根据相关性审计确定。
7. normalization builder 只接受 split artifact 中的 train IDs；DataLoader 和阈值评价读取同一 split，checkpoint 绑定该 hash。

验收：任何 unknown 体素的预测变化不改变负类/FP/mass/验证指标；normalization 不读取 val/purge；train、purge、val frame ID 两两不相交。

此阶段完成审计后，在全量数据重建前再次向用户报告并确认范围分支。

本轮实际完成：逐帧持久 observed mask、五模态 training manifest、唯一时间 split、3 s purge、train-only normalization、formal data protocol、统一训练入口精确 train/validation frame IDs，以及模型 0--80 m evidence / 地图 0--120 m range 分离。4 帧 fresh smoke 已验收，但未生成全量正式 artifact。

### 实施 4：修复部署 observed、Radar 置信度、frame 和风险契约

**当前子阶段：deployment observed/frame/risk 运行时安全契约已完成。**

本子阶段边界：从已验收的 training-profile 预处理根生成只含 Radar/IR 的不可变 deployment 视图和严格 manifest；复用并校验原 training manifest 的 preprocessing/sync/calibration provenance，不复制或修改 LiDAR/target/observed，不启动推理或训练。

本子阶段准备修改：

- `diffusion_consistency_radar/dataset_manifest.py`：保留 training v2，新增严格 deployment v3 合同。
- 新增 `diffusion_consistency_radar/deployment_view.py` 与 `scripts/build_deployment_view.py`：生产、receipt 和自包含校验。
- `diffusion_consistency_radar/scripts/inference.py` 与三个正式推理 launcher：正式入口要求完整 deployment view，而非仅验证 profile 字符串。
- `NTU4DRadLM_pre_processing/preprocess-v2.sh`：全量 training artifact 后生成 loop3 deployment view。
- `test/unit/test_deployment_view_protocol.py`、manifest/formal inference 测试：RED/GREEN、篡改、额外模态、符号链接和身份漂移反例。

本子阶段完成结果：training manifest 保持 schema v2；deployment view 使用 schema v3 和根/场景两级 receipt，携带父 training manifest 快照与 Radar--IR sync，默认 hardlink 且支持 copy。正式 launcher 在推理前校验精确场景集合，Python 入口再把视图、当前标定与 checkpoint data protocol 交叉绑定。garden 4 帧 smoke 与服务器展开 hardlink 的复制回归均通过。

本轮授权边界：先完成不依赖重训的运行时安全部分，即 Radar 可见域 observed mask、正式地图 mask/frame 门禁、LiDAR→body→local 显式坐标链、三态风险查询与输出非覆盖/原子 metadata；Radar point-count/Doppler-validity 与 `UncertaintyHead` 结构调整留作下一子阶段，避免在无训练条件下改变 checkpoint 兼容性。

完成结果：formal inference 发布遮挡安全的 Radar endpoint-ray mask 和内容收据；formal map 实施 run/mask/pose/LiDAR→body 门禁、`T_local_body@T_body_voxel` 坐标链、三态风险与 fresh/原子输出。103 项聚焦回归和 garden 4 帧只读 smoke 通过。point-count/Doppler-validity 与 `UncertaintyHead` 改造按授权边界保留到下一子阶段。

**当前后续：LiDAR→body / body→local 候选契约诊断已完成，等待权威 frame 语义。**

- 新增独立诊断脚本，显式假设现有 4×4 文件为 `T_imu_radar`、R/T 文件为 `T_lidar_radar`，组合候选 `T_imu_lidar`。
- 对 `gt_odom` 分别生成“GT 是 IMU pose”和“GT 是 LiDAR pose”两套候选，使用 Radar timestamp、平移线性插值和四元数 SLERP。
- 超出 GT 时间范围或插值 gap 超限的帧只记录为 uncovered，禁止外推；所有输出必须标记 `formal=false`。
- loop3 实际严格覆盖为 6162/6432；4 帧早于 GT，266 帧位于超过 0.2 s 的 GT gap。正式 loader 已按内容拒绝这些候选。

**当前执行：frame 语义权威证据与多窗口反证。**

- [x] 只读定位原始 bag、TF/static TF、GT 导出代码、标定生成记录或 CAD 线索；原始 frame 定义是 formal 硬门禁。
- [x] 用官方命名约定把 `calib_radar_to_imu.txt` 收敛为 Radar→VectorNav IMU；body 轴约定与 GT 导出 frame 仍未确认。
- [x] 冻结 GT-as-IMU/body 与 GT-as-LiDAR 两种仍可辨识假设，新增独立多窗口静态 LiDAR 一致性诊断；Radar→IMU 取逆不再作为等权候选。
- 诊断只用于排除明显错误方向，输出继续 `formal=false`；无权威元数据时不得因为某项指标胜出就发布正式合同。

**本阶段结论：** LiDAR-time v2 在 48 个 1 s 高转角 pair 中全部支持 GT-as-LiDAR，0.5/2.0 s 敏感性汇总结论一致；代码、收据和文档已完成。formal 发布继续等待 `gt_odom` exporter frame 与 VectorNav IMU→airborne body 轴约定。

**当前执行：经验 LiDAR pose 离线地图合同。**

- [x] 沿 formal streaming map 的 pose/extrinsic/preflight/metadata 调用链确认最小接口边界。
- [x] 新增 `empirical_lidar_pose_contract_v1` receipt 生产/校验，绑定 GT、LiDAR-time sync、candidate/overlap 证据与逐帧 pose，不允许外推。
- [x] formal map 增加仅限离线的 `T_local_voxel` 直通模式；与现有 airborne `T_local_body@T_body_voxel` 模式互斥。
- [x] offline empirical 模式写入 `airborne_formal=false`，禁止与飞行/PX4/避障模式混用，并保持 unknown/risk fail-closed。
- [x] 完成 RED/GREEN、少量帧 smoke、兼容回归和文档更新；不运行训练/GPU。
  - 共享 prediction artifact 协议重构后的静态编译、37 项推理接口、46 项地图与 6 项经验姿态测试已通过。
  - 两个 CLI `--help` 已通过；真实 receipt 的最终 loader 复核改用其公开的 voxel 文件名列表接口。
  - 真实 receipt 运行时复核为 6432 available、6165 selected、267 uncovered，首尾选中帧为 000005/006431，`airborne_formal=false`。
  - 最终调用链审查确认 prediction/observed 帧顺序、内容 hash、shape/dtype 与经验子集均在输出创建前验证；共享 prediction 协议无重复实现，`git diff --check` 通过。

目的：让模型输出进入地图时不再把 unknown 写成 free，也不再高不确定性 fail-open。

准备修改：

- `diffusion_consistency_radar/cm/multimodal_fusion.py`
- `diffusion_consistency_radar/scripts/inference.py`
- `diffusion_consistency_radar/cm/probabilistic_mapping.py`
- `diffusion_consistency_radar/scripts/streaming_map_update.py`
- `test/unit/test_airborne_multimodal_refactor.py`
- `test/unit/test_multimodal_inference_interface.py`
- `test/unit/test_probabilistic_mapping_uncertainty.py`

步骤：

1. 体素化持久化 point-count/Doppler-validity sidecar；`UncertaintyHead` 分离 `radar_validity` 与条件方差 confidence，空体素不能得到 confidence=1。
2. 推理输出 `*_observed_mask` 及其 metadata。第一版由输入 Radar 端点射线可见域生成，IR frustum只表示投影视锥，不能单独冒充 free-space 观测。
3. formal map 必须收到 observed mask；兼容模式缺 mask 时保持只有明确 occupied 可融合，并在 metadata 标记 degraded，不能声称 free-space。
4. inference metadata 明确输出为 LiDAR frame；地图正式入口新增显式 LiDAR→body 外参，之后再使用逐帧 body→local pose；正式模式禁止 identity pose。
5. 风险查询改为 `clear/obstacle/unknown` 三态和可审计 reason。安全距离使用 `v*t_reaction + v^2/(2*a_brake) + margin`；地图/搜索半径不足以覆盖安全距离时返回 unknown/risky。
6. inference 和 streaming map 在任何输出前拒绝非空目录和符号链接，采用原子 metadata 写入。

验收：全低概率 sigmoid + 无 observed 不降低 unknown mass；空地图/高不确定/视野不足均不返回 clear；35/50/70 m/s 的安全距离单调增加；静态障碍经 LiDAR→body→local 后保持同一 local 位置。

### 实施 5：原始数据、CD 和配置 fail-closed

目的：关闭剩余会静默制造错误数据或隐式续训的接口。

准备修改：

- `NTU4DRadLM_pre_processing/unpack_rosbag.py`
- `diffusion_consistency_radar/scripts/cd_train_optimized.py`
- `diffusion_consistency_radar/scripts/unified_train.py`
- `diffusion_consistency_radar/config/default_config.yaml`
- `test/unit/test_pointcloud_schema_protocol.py`
- `test/unit/test_cd_training_entrypoints.py`

步骤：

1. PointCloud v1/v2 共用精确字段映射和逐场景 schema；formal Radar 缺 intensity或Doppler直接失败，不能补 0 伪装有效测量。
2. 点云/图像保存失败向上抛错；文件名使用纳秒级时间或时间+序号，并在写前拒绝碰撞/覆盖。
3. 正式 `sequence_length` 暂时只允许1；移除“加载多帧但只用最后一帧”的假时序行为。真正时序融合另立模型任务。
4. CD 显式接线并保存 EMA rate/schedule/scales 语义；无法支持的 YAML 字段启动时拒绝，不静默忽略。
5. 独立 CD 入口取消自动发现 `cd_best.pt`；仅显式 `--resume` 可续训。

验收：字段缺失、坏 schema、保存碰撞、未支持时序/CD配置均在产生输出前失败；无 `--resume` 时永不读取旧 CD。

### 实施 6：正式评价和端到端性能协议

目的：把散落诊断能力合并成可复现的正式评价，不把 I/O 外的 kernel 时间冒充系统延迟。

准备修改：

- `diffusion_consistency_radar/cm/evaluation_metrics.py`
- `diffusion_consistency_radar/scripts/evaluate_saved_predictions.py`
- `diffusion_consistency_radar/scripts/inference.py`
- `diffusion_consistency_radar/scripts/streaming_map_update.py`
- `diffusion_consistency_radar/launch/evaluate_inference.sh`
- 对应 `test/unit/test_*evaluation*` 和 mapping 测试

步骤：

1. 指标全部按 observed domain 统计 3D occupied/free IoU、precision/recall/F1，并单独报告 unknown coverage。
2. 增加 0--20、20--40、40--80、80--120 m 与高度 bins；复用现有垂直结构指标，增加 ground FP、细障碍指标前先定义可复现标签规则。
3. 增加跨帧稳定性、地图更新/查询 P50/P95/P99。
4. 分解 Radar/IR I/O、预处理、GPU推理、后处理、保存、地图、总时延，并记录峰值显存/RSS；不在本轮承诺 ROS 时延。
5. 增加 deterministic 模式和完整随机/库版本 metadata；若算子不支持确定性则明确失败或标记 nondeterministic。

验收：评价读取同一 split/observed/protocol hash；旧结果缺协议时只允许 legacy 报告，不与 v2 直接横向比较。

### 实施 7：数据重建与 mini 门禁（不做全量训练）

1. 所有单元测试和 `py_compile` 通过后，用 2--4 帧新目录运行预处理 smoke，检查 frame、signed delta、标定、mask、schema 和 manifest。
2. 经用户第二次确认后，生成新的全量 v2 candidate、split artifact 和仅训练帧 normalization；绝不覆盖 v1 candidate/artifact。
3. 运行现有 8 GB mini preflight 和 1 epoch/极少 batch VAE、LDM、CD smoke；设置保守显存、worker、温度和输出新目录，不启动长期训练。
4. mini 通过后只给出服务器全量训练命令和验收门禁；是否上传/训练由用户另行确认。

## 影响说明

- **监督信号：** unknown 不再作为 free；80--120 m 分支会改变正负样本和所有 occupancy 指标。LDM 的负类、质量和结构损失都会收缩到 observed domain。
- **体素数量：** 若 120 m 标签可信并保留现有网格，模型体素仍为 `32×128×128=524288`/帧；若收敛到80 m，必须同时冻结新的物理范围和分辨率，不能只改 `x_max`。
- **checkpoint：** 因监督域、split、normalization、IR几何和 provenance 变化，v1 VAE/LDM/CD 均不能作为 v2 正式 resume 链；旧 checkpoint 保留作诊断。
- **指标：** v2 指标只在 observed domain 且新 split/purge 上计算，数值与历史结果不可直接比较；报告中必须并列记录 coverage。
- **数据/存储：** observed mask、point-count/validity 和新 manifest 会增加数据文件；优先使用稀疏/压缩 sidecar，实际增量先在 4 帧 smoke 中测量。

## 停止条件和确认点

- 当前执行点：实施 0--3 和实施 4 的无重训运行时安全子阶段已完成；实施 4 仅剩会改变数据/权重协议的 point-count/Doppler-validity 和 `UncertaintyHead`。
- 在选择 80/120 m 分支、全量重建、mini 训练和正式训练前分别再次报告并等待确认。
- 若缺少可信速度、LiDAR→body 外参或制动参数，代码支持可以完成，但正式 airborne/avoidance 门禁保持失败，不填入猜测常量。
- **计划状态：用户已确认 formal v2 0--80 m；只执行代码、单测和 2--4 帧 smoke，不执行全量重建或训练。**

## 错误记录

### 当前执行：Radar point-count / Doppler-validity sidecar（2026-08-27）

- [x] 沿预处理入口、voxel 保存、manifest、Dataset、formal data protocol 与训练 launcher 追踪真实调用链。
- [x] 用 RED 测试冻结 point count、有效 Doppler 数、空/单点体素、sidecar hash 与 formal fail-closed 语义。
- [x] 最小实现稀疏/压缩 sidecar 和 manifest/Dataset 接口；本阶段不修改 `UncertaintyHead`、网络输入、loss 或 checkpoint。
- [x] 用 2--4 帧 CPU smoke、聚焦回归和静态检查验证；不运行全量预处理、模型 forward 或训练。
- [x] 同步根 TODO/README；旧 full120 mini 明确标为 legacy，给出 formal v2 全量数据、只读预检和服务器 VAE 脚本及前置门禁。

边界：sidecar 先作为审计 metadata 进入 batch，不改变四通道 Radar 张量；未来只有在独立 checkpoint 协议与重训计划确认后，才允许模型消费 validity/count。

| 错误 | 尝试 | 处理 |
|---|---:|---|
| 附件首次读取因输出上限只返回部分内容 | 1 | 改为按行号分段读取，未据截断内容提前下结论 |
| 首次搜索概率地图用了错误路径 | 1 | 通过 `rg --files` 定位到 `cm/probabilistic_mapping.py` 后按函数精确读取 |
| checkpoint 单测的 construct case 导入模型链时 OpenMPI 无法在 sandbox 建立本地通信 socket，进程退出 1 | 1 | 协议测试本身已开始通过；先分离不触发 MPI 的测试，再在需要时以允许本地 socket 的测试环境复核 construct case，不重复原命令 |
| 用 `from test.unit...` 分离单测失败，因为 `test/unit` 不是 Python package | 1 | 改用 unittest 的文件路径/name 机制或 `importlib.util.spec_from_file_location` 加载，不重复 package import |
| 遮挡语义聚焦回归再次误用 `python -m unittest test.unit...`，在收集阶段报同类 `ModuleNotFoundError` | 2 | 立即改为直接运行 `test/unit/test_multimodal_inference_interface.py -v`，36/36 通过；后续禁止使用 `test.unit` package 名 |
| VAE resume 补丁因函数 docstring 与预期上下文不一致而未应用 | 1 | 先按精确行号读取三个 resume/main 片段，再使用更小上下文补丁；没有任何部分写入 |
| inference 显式 calibration_dir 补丁因 preflight 调用参数上下文不一致而未应用 | 1 | 精确读取 preflight 和逐帧调用片段后分成多个小补丁；没有任何部分写入 |
| Dataset/IR 回归中 4 项失败：3 项测试用 `__new__` 构造 generator 缺新属性，1 项 legacy LDM 测试缺 v2 data protocol | 1 | 正式 IR/Dataset 35 项已通过；generator 用安全 `getattr(..., False)` 保持单元构造兼容，legacy Radar 单位测试允许显式 legacy data identity，正式路径继续严格 |
| Manifest CLI 补丁因 argparse 参数采用多行格式而上下文未匹配 | 1 | 读取完整 172 行 CLI 后按函数块替换，不重复原补丁；核心 manifest 模块改动不受影响 |
| Manifest v2 在目标 Conda Python 3.8 导入时报 `str | None` 不支持 | 1 | 保留类型注解并增加 `from __future__ import annotations`，下一次必须用目标 Conda 环境复测，不能只依赖系统 Python py_compile |
| 用 `find | sort | head` 限制大目录输出时，上游 `sort` 收到下游提前关闭造成 broken pipe | 1 | 后续改用 Python 限量遍历或 `find -print -quit`；该只读失败没有修改文件 |
| 为检查 manifest 结构直接打印完整 `modalities`，导致数千帧记录输出被截断 | 1 | 后续只读取键、计数和首条记录，不再整块打印大 manifest |
| 新增 overlap 单测首次导入时目标诊断脚本不存在，报 `FileNotFoundError` | 1 | 这是预期 RED，随后实现最小脚本；未创建结果目录 |
| LiDAR-time/sync snapshot 与 manifest self-hash 两项新测试分别因旧接口不接收参数、旧校验未抛错而失败 | 1 | 两项均为预期 RED；补 v2 显式时间合同与 canonical hash 门禁后 9/9 通过 |
| 最终概率地图回归在首项导入 MPI 时因沙箱禁止本地通信 socket 退出，断言未执行 | 1 | 新增 candidate/overlap 11 项已通过；地图测试按既有批准在沙箱外原命令复核，不重复沙箱内运行 |
| Radar statistics 新增 5 项 RED 全部 ERROR | 1 | 目标模块不存在、voxelizer 不接受 `return_statistics`，与预期缺口一致；进入最小 GREEN，不重复旧实现命令 |
