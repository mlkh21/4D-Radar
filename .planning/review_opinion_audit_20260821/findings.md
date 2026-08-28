<!-- 文件功能：记录外部审查意见与当前工作树的证据对照。 -->
# 审查意见核验发现

## 范围与约束

- 审查对象是当前脏工作树，必须重新核验附件中的 HEAD、文件行号、checkpoint 状态和统计值。
- 本轮仅做只读诊断和计划；不修改源码、数据、checkpoint、日志或实验结果，不运行训练。
- 已知附件至少提出 IR 外参、Doppler 语义、部署 manifest、80--120 m 监督、地图 observed/risk、正式 checkpoint 等 P0 问题；完整清单仍在分段提取。

## 附件问题清单

- P0-01：LiDAR-frame 体素误用 Radar→Thermal 外参，并叠加固定 0.01 m 补偿。
- P0-02：正式 Doppler 未自运动补偿，metadata 却宣称 compensated。
- P0-03：正式推理 manifest 硬依赖 LiDAR/target，部署不能只带 Radar+IR。
- P0-04：模型输出 120 m，但 target 只监督至 80 m，远距 unknown/free 语义不清。
- P0-05：地图缺 observed mask 时用 `probability > 0`，导致 unknown 坍缩。
- P0-06：风险查询把高不确定性判为不危险，且固定 5 m 不适配 35--70 m/s。
- P0-07：正式 VAE/LDM/CD 状态不完整，应先处理协议问题再继续耗时训练。
- P1-01 至 P1-10：真实同步 delta、PointCloud schema、normalization split 泄漏、purge gap、LDM loss mask、空 Radar confidence、按阶段 checkpoint 校验、输出覆盖、地图 frame/pose、正式评价覆盖。
- P2/P3：标定/保存 fail-open、文件名精度、sequence_length 无效、CD 配置未接线/隐式续训、不可信 checkpoint 回退、端到端计时、确定性、thermal 专用输入、动态 evidence、rolling map 和独立 DEM。
- 附件的事实断言、严重级别和建议需要分别核验；尤其训练目录状态、行号、已有未提交修复和数据统计是易变化快照。

## P0 初步证据

- 附件的 Git 状态已经过期：当前 HEAD 为 `02c4fd3`，而非 `96df62c`；当前可见工作树状态也不再是附件记录的约 46 个跟踪修改。所有“当前状态”必须以本轮实查为准。
- 正式结果目录仍只有 VAE 链和失败归档，没有 LDM/CD；VAE 日志最后记录 epoch 16/100，存在 `vae_best.pt` 等约 8.5 MB checkpoint。是否仍在运行尚待进程核验。
- P0-01 的关键前提成立：candidate `align_to=lidar`，而当前 CalibrationProvider 读取 `calib_radar_to_thermal.txt`；Dataset 仍存在固定 `LEGACY_SYNC_DISPLACEMENT_X_M=0.01` 并无条件调用 legacy compensation。需继续确认投影矩阵方向和调用端是否有额外变换。
- P0-02 的 metadata 矛盾成立：正式脚本和 policy 都是 `velocity_mode=none`，但 policy channel 2 固定写成 `egomotion_compensated_mean_doppler`。当前体素函数只有在 `v_drone` 非空时才做 Doppler 补偿，正式候选数据并不满足该条件。
- P0-03 的结构事实成立：manifest v1 的 `MODALITY_PATTERNS` 固定包含 radar/lidar/target/IR，严格要求四模态 frame ID；正式推理 launcher 在核心推理前调用同一 manifest validator。是否应“拆成两种文件格式”仍待接口设计评估，也可能用同一 schema 的 profile/required-modalities 更小步实现。
- 进程复核未发现真实训练进程；此前 `pgrep` 命中的是诊断命令自身。当前正式训练可判定为停止在 VAE epoch 16，而非“仍在训练”。停止原因仅凭日志不足以确定。
- P0-01 的坐标链已经确认：预处理在 `align_to=lidar` 时把 Radar 点变换到 LiDAR 坐标，IR 投影层却把输入体素坐标直接按 Radar→Thermal 外参投影。因此该问题不是命名争议，而是实际坐标系不匹配。合理修复应改为 LiDAR→Thermal，或经严格验证的 Radar→LiDAR→Thermal 组合；不能继续依赖 Radar→Thermal 直接投影。
- 固定 0.01 m 位移也确实会对所有样本应用，但它只影响 IR 投影链，不影响仅以 target 为输入的 VAE。建议删除前必须把同步补偿职责移到带真实逐帧时间差与运动状态的预处理阶段，避免用另一个固定值替代。
- P0-02 对“正式补偿”的指控成立，但建议需要拆成两步：立即把 `velocity_mode=none` 下的通道语义改为 raw Doppler 并禁止误称 compensated；只有取得可信的逐帧速度、姿态和传感器时间差后，才实施真正自运动补偿。直接代入任务描述中的 35--70 m/s 会制造伪监督。
- P0-03 的修复方向合理，但建议用同一 manifest schema 的明确 `profile`/`role` 区分 `training` 与 `deployment`，保留训练 manifest 的四模态完整性；不宜简单放宽现有 validator，也不宜让命令行临时决定任意必需模态。
- P0-04 的事实前提成立：target 构建在 80 m 外把 occupancy 初始化为 0，而模型范围是 120 m。是否已经由 observed mask 正确屏蔽，以及 VAE/LDM 各损失是否消费该 mask，仍需沿训练调用链核验。若没有屏蔽，80--120 m 会被错误训练成 free；即便有屏蔽，导出和指标也必须保留 unknown 语义。
- “IR 预处理丢失 16-bit 原始热成像”的结论目前只有代码侧证据：当前使用 `IMREAD_GRAYSCALE`、除以 255 并复制为 3 通道。源 rosbag 的压缩图像是否本来就只有 8-bit 尚未证明，因此不能直接把“保留 16-bit”列为确定修复；应先审计源编码。缺少饱和度、低对比度等 IR 质量标记则可作为独立增强项。

## 逐项结论

| 编号 | 当前结论 | 方向判断 | 关键证据/修正 |
|---|---|---|---|
| P0-01 | 存在 | 合理，但需先冻结标定权威来源 | `align_to=lidar` 后模型体素是 LiDAR 系，投影却直接使用 Radar→Thermal。直接 LiDAR→Thermal 与 Radar→LiDAR→Thermal 的实测最大旋转元素差 `0.0508489`、最大平移分量差 `0.151702 m`，不能把闭环检查当成会自然通过的门禁。建议以直接 LiDAR→Thermal 为当前候选并做可视化验收，同时记录闭环残差，后续重标定。 |
| P0-02 | 存在 | 方向合理，但不能伪造速度 | 正式数据 `velocity_mode=none`，Doppler 未补偿，policy 却称 compensated。当前 raw garden 没有平移速度文件，只有 IMU；35--70 m/s 任务参数不能作为数据速度。应先改成 raw 语义，再为未来有可信速度/姿态的数据接通补偿。 |
| P0-03 | 存在 | 合理，建议同 schema 分 profile | manifest v1 固定要求 Radar/LiDAR/target/IR，所有正式推理 launcher 先调用它。应保留严格 training profile，新增明确 deployment profile，只允许 Radar+IR 及其标定/同步 provenance。 |
| P0-04 | 存在且已实测 | 合理，必须设决策门 | 000000 原始网格在 80--120 m 有 48 个 LiDAR occupied，但 target 为 0；ray mask 把该区 5714 个体素标为 observed。缩放到训练网格后仍有 1936 个远距 observed、0 个 target occupied，VAE 会把它们作为 free 监督。 |
| P0-05 | 存在 | 合理 | 地图虽已支持显式 observed/unknown，但缺 mask 时仍用 `occupancy>0`；sigmoid 输出几乎全正。正式推理又不保存 observed mask，因此集成路径仍会令 unknown 坍缩。 |
| P0-06 | 存在 | 合理，但需定义安全契约 | 空地图、超搜索半径和高不确定性都可能返回 `is_risky=0`；近障条件仍是固定 `distance<5m AND uncertainty<0.7`。应输出 clear/obstacle/unknown 三态，并用反应时间、制动能力和余量计算安全距离。 |
| P0-07 | 状态存在，不是代码缺陷 | 建议需收窄 | 正式训练停止在 VAE epoch 16/100，无真实训练进程，无 LDM/CD。VAE不受 IR/Doppler/manifest 直接影响，但受 P0-04 错误监督影响，修复协议后不应继续作为正式 v2 checkpoint；旧产物应保留作诊断，不能删除。 |
| P1-01 | 存在 | 合理 | Radar--LiDAR signed delta 已写入 CSV，却只被校验、不传 worker；worker 对 Radar 与 LiDAR 同时应用固定 `dt_sync=0.002`，不能消除相对时差。IR CSV只存绝对 delta，Dataset另加固定 0.01 m。 |
| P1-02 | 部分存在 | 合理 | PointCloud2 已固定五列并写 schema；PointCloud v1 仍用子串 alias、缺字段补 0 且不写 schema。当前 raw garden 没有 schema，无法证明列 3 是 RCS 或缺 Doppler 的来源。 |
| P1-03 | 存在 | 合理 | normalization builder扫描 garden 全 4013 帧，正式优化集仅前 80%；artifact 也固定声明 4013。必须改成精确训练 frame ID 列表并绑定 split hash。 |
| P1-04 | 存在 | 合理，但 gap 值需数据审计 | `temporal_block_split_indices()` 是相邻前后缀，无 embargo/purge。应按真实时间戳生成一次性 split artifact，再据相关性审计确定 gap，不能随意硬编码帧数。 |
| P1-05 | 存在且范围更广 | 合理 | LDM decoded occupancy/FP/mass、density、column、IR-negative 和验证 IoU 均未接 observed mask；latent MSE 也无法直接做空间 mask。若继续 120 m+unknown，需要调整目标表示/编码协议，而不只是给三个辅助损失多传一个参数。 |
| P1-06 | 存在 | 合理 | `UncertaintyHead` 对空体素 variance=0 得到 confidence=1，并参与 IR gate/fusion。地图在没有 observed 时会乘零，影响被部分缓解，但模型融合语义仍反转。仅靠 occupancy 可修空体素，单点方差可信度还需要 point-count/validity sidecar。 |
| P1-07 | 存在 | 合理 | `inference_ldm.sh` 明明只需要 VAE+LDM，却调用要求 VAE/LDM/CD 全齐的 validator。应按目标 stage 验证父链。 |
| P1-08 | 存在 | 合理 | inference 和 streaming map 对固定文件名直接覆盖；离线 evaluator 已有非空保护，说明同一模式可复用。 |
| P1-09 | 存在但已有部分基础 | 合理 | 地图已实现三维 pose-aware warp 和严格 pose CSV，但输入声明为 body voxel；模型输出实际为 LiDAR frame，缺 LiDAR→body；无 pose 仍允许 identity legacy。正式入口应强制完整 frame chain 和 pose，兼容入口可保留。 |
| P1-10 | 存在 | 合理 | 正式 saved-prediction evaluator 仍只聚合近距 BEV/NN/Chamfer/不确定度；垂直结构指标散落在诊断脚本，未形成 unknown-aware 3D、range/height、free-space 和连续稳定性正式协议。 |

## P2/P3 复核

- 标定解析对坏行 `continue`、缺项回落默认值，正式训练没有 fail-closed；意见成立。
- 点云/图像保存异常只打印后继续、文件名保留 6 位小数且没有覆盖检测；意见成立。
- `sequence_length` 即使收集多帧也只返回最后一帧，而且统一训练入口根本没有把配置传入 Dataset；意见成立。当前最小安全做法是正式模式明确只允许 1，时序融合另立设计任务。
- CD 的 EMA 实际固定 `0.999`，多个 YAML 字段未接线；独立 CD main 会自动选择 `cd_best.pt` 续训；意见成立。统一 launcher 的显式 `ALLOW_RESUME` 已经部分缓解。
- 三个训练/推理入口都可能从 `weights_only=True` 回退到普通 `torch.load`；对可信本地产物是兼容性风险而非当前数据错误，但正式模式应禁止，legacy 诊断需显式 opt-in。
- 推理 `inference_seconds` 只包围 `generator.generate()`，不含 I/O、IR准备、后处理、保存和地图；确定性也只设置随机种子，未冻结算法；意见成立。
- IR 单通道 stem、质量门控、在线动态 evidence、空间 rolling map、独立地面/DEM 和 ROS 节点属于后续能力建设，不应混入首批协议修复。
- “源 16-bit 热像被丢失”不成立于当前 garden 已解包数据：抽查原始 PNG 为 `uint8`、`512x640x3`、范围 0--255。当前真实问题是 3 通道压缩图被灰度化后复制，是否损失有用伪彩/传感器信息需要另做审计，不能声称恢复不存在的 16-bit 辐射值。

## 审查之外新增的阻塞问题

- 正式 launcher 通过 `.tmp_train_dataset/garden` 场景软链接隔离训练场景，但 `CalibrationProvider` 根据 dataset root/cwd 猜标定目录。该临时根下没有 `Data/config`；实际加载一帧得到 `is_mock_calib=True`、`calib_source=mock_default`、thermal intrinsics 也是 default。VAE 不消费 IR，因此旧训练未受影响；LDM 若现在启动会在 mock 外参上训练，而且 trainer 只在 mock ratio 大于 0.5 时 warning，不会阻止正式结果。这是新增 P0。
- 当前 manifest 只绑定 Radar→LiDAR 标定 hash，没有绑定 LiDAR→Thermal、Thermal 内参和逐帧 observed mask；即使数据内容不变，IR 几何也无法由 manifest 完整复现。
- VAE checkpoint 记录模型/网格协议，但不记录 dataset manifest、split、target policy 或 observed-mask 协议 hash。现有 epoch 16 checkpoint 无法证明所用监督域，不能进入修复后的正式链。

## 实施期复核（2026-08-21）

- 用户确认后重新核验：当前 HEAD 为 `02c4fd3`，源码工作树干净；只有本任务 `.planning` 记录未跟踪，因此可以按小步补丁实施且不会覆盖用户源码改动。
- `checkpoint_chain.py` 当前只有 `formal_chain_v1/formal_mini_chain_v1`，正式 validator 无目标阶段参数并固定加载 VAE、LDM、CD 三个文件；这直接造成 LDM 推理被尚不存在的 CD 阻塞。
- `safe_torch_load()` 当前会在 weights-only 失败时无条件回退普通 `torch.load`；v2 正式链需要默认 fail-closed，legacy 兼容必须由显式开关触发。
- 当前 checkpoint payload 只覆盖模型、网格、父 checkpoint 与 Radar normalization；没有统一的训练数据协议 hash 容器。实施时应新增结构化 `data_protocol`，而非在各 stage 零散增加顶层字段。
- `dataset_manifest.py` 目前将 v1 schema、四模态目录和四项 provenance 写死在模块常量中；新增 deployment 不能复用 `_collect_modalities()` 后再跳过缺失目录，必须让 profile 决定精确 required modalities，同时保持 training 的严格全集合校验。
- `NTU4DRadLM_VoxelDataset` 当前从根目录自动枚举场景并按场景数再次切分，既忽略 `data_loading_config.yml` 的显式 allowlist，也会让单场景 train/val 都加载同一场景；实施 1 必须显式传入 `scene_names`，且不在 Dataset 内做场景级猜测。
- Dataset 在 `__getitem__` 中调用 `_get_mock_calibration()`，并对真实或 mock 外参都叠加 `velocity_m_s=50, dt=200us` 得到的固定 `0.01m`；这应从正式加载链完全移除。
- `train_unified.sh` 在 preflight 后仍把配置的 `dataset_dir` 指向 `.tmp_train_dataset`，随后删除/重建该目录并创建场景 symlink；这是 mock calibration 根因。最小修复是直接把正式根写入 override，并把 `TRAIN_SCENES` 写成显式 `scene_names`，完全移除临时根操作。
- 当前 `CalibrationProvider` 只把 `calib_radar_to_thermal.txt` 视为真实 IR 外参。因为体素已经对齐到 LiDAR frame，实施 2 必须把正式选择改成 `calib_livox_to_thermal.txt`（数据命名中的 livox 即监督 LiDAR），同时保留 Radar→Thermal 和 Radar→Livox 仅用于闭环审计。
- unified trainer 的三个 stage 都从 `data.checkpoint_protocol` 获取协议，但没有 `data_protocol` 对象；新增后必须在 VAE/LDM/CD 构造器和 resume preflight 中共用，而不是只在最终保存时补字段。
- Radar--LiDAR CSV 已含 `signed_delta_seconds`，但 `_load_radar_lidar_sync()` 只返回整行且调用方丢弃返回值；worker 仍把一个固定 `dt_sync` 同时传给 Radar 和 LiDAR voxelizer，导致两份点云同向移动，无法消除相对时间差。
- Radar--IR 记录当前只写绝对 `delta_seconds`，最近邻函数也只返回绝对值；可由已选文件时间戳计算 `ir_timestamp-radar_timestamp` 并持久化 signed delta，无需改变配对规则。
- `velocity_mode=none` 时仍把 channel 2 写成 `egomotion_compensated_mean_doppler`；这可直接通过 policy 构造函数按模式选择 `raw_mean_doppler` 修复，无需等待速度输入。
- 现有 thermal 测试反而把共享固定 `0.01m` 当作正确协议；实施 2 必须把这些断言改成“Dataset/inference 不再导入或调用 legacy compensation”。
- 不安全 `torch.load` 回退并不只在诊断模块：unified train、独立 CD 和 inference 各自复制了一份。已决定统一委托给 `checkpoint_chain.safe_torch_load()`，正式默认 weights-only，CLI 历史兼容另加显式开关。
- 独立 CD main 仍自动探测 `${save_dir}/cd_best.pt` 并续训；虽然属于后续实施 5，但接入 v2 身份时若保留会先触发 data protocol 检查。应在本轮至少确保它无法跨协议隐式续训，完整改成显式 `--resume` 留在实施 5。
- inference 的实时 IR meta 同样调用 legacy 0.01 m，并在 run metadata 宣称该补偿；删除 Dataset 调用时必须同步删除这两处，避免训练/推理接口再次分叉。
- 当前正式 Conda 环境是 Python 3.8；源码若使用 PEP 604 `X | None` 必须启用 `from __future__ import annotations`，系统 Python 的 `py_compile` 通过不能替代目标 Conda 环境导入测试。
- signed delta 的空间平移符号应由静态世界点关系统一定义为 `t_sensor-t_reference`：LiDAR 参考时 Radar delta 为 `t_r-t_l`、LiDAR 为 0；Radar 参考时 LiDAR delta 为 `t_l-t_r`、Radar 为 0。不能直接把 CSV 的 `lidar-radar` 符号不加转换地应用到两种参考分支。
- 三外参闭环当前只适合作为审计量。代码继续以 LiDAR-frame 的直接 LiDAR→Thermal 外参作为投影权威，并记录组合残差；没有设置会因现有不闭环标定而永远失败的伪门禁。

## 总体判断

- 审查的主结论是可信的：当前可以称为“Radar--IR 条件模型 + 离线概率地图原型”，还不能称为已闭环的 120 m 高速无人机部署系统。
- 7 项 P0 中，P0-01 至 P0-06 都有当前证据；P0-07 是状态门禁。10 项 P1 中 9 项成立、P1-02 为部分成立。
- 修复顺序需要调整：必须把“显式场景/标定依赖”和“监督域决策”放到首位；不能先继续 VAE/LDM，也不能只在 loss 层补 mask 后宣称 120 m unknown 已解决。

## 第一批实施与范围审计结论

- 实施 0--2 已完成并通过聚焦回归；formal v2 训练/推理在 range、observed、split 和新 artifact 未冻结前保持 fail-closed，旧 v1 仅保留诊断用途。
- 推理复审补获“checkpoint 链与 deployment manifest 分别通过但未交叉绑定”的缺口；现已在模型构建/输出前验证 VAE/生成模型 data protocol、当前标定和 deployment provenance，并记录 deployment identity。
- garden 4013 帧 v2 审计采用完整 XYZ 体素盒口径：80--120 m raw 点 372780、occupied voxel 365069、target occupied 0；远距/近距 raw 比 0.404%，occupied 比 1.488%。
- 远距 ray coverage 均值仅 0.194%，近距为 3.845%；抽样 Radar--LiDAR 1 m/2 m 匹配均值 5.23%/23.24%。这证明“存在远距 LiDAR 点”，但尚不足以证明 120 m 监督可靠。
- 推荐选择 0--80 m formal v2，保持 `32×128×128` 张量并把 80--120 m 作为地图 unknown；若用户坚持 120 m，则必须进入 observed-aware target/latent 架构分支，不能做参数级扩距。

## 0--80 m 正式分支实施前调用链复核

- 预处理主链当前同时生成 Radar、LiDAR 和 target 体素，但只落盘 `radar_voxel`、`lidar_voxel`、`target_voxel` 与 IR；观测区域掩码尚未成为持久化数据契约。
- Dataset 当前在 `__getitem__` 中根据 LiDAR 临时重建 ray-observed mask，LiDAR 缺失时退化为 occupied-only；正式协议需要强制读取预处理阶段的掩码，兼容路径可以保留但不得进入正式训练。
- VAE 训练链已经把 `occupancy_observed_mask` 传入损失，因而本轮不需要改 VAE 张量接口；重点是把掩码来源改为可校验的持久输入。
- `unified_train.py` 当前只按下标切相邻前缀/后缀，没有唯一 split artifact 或 purge gap；正式 v2 需要按 scene/frame_id 消费唯一切分文件，禁止运行时重算。
- normalization 构建器当前遍历全 scene 或按 `max_frames` 截断；正式 v2 必须绑定 split artifact，并且只统计 train frame。
- manifest v2 尚未把 observed mask 列为训练模态；formal data v2 虽声明 `observed_mask_sha256`，仓库仍缺少从 manifest、split 和持久掩码生成正式 artifact 的生产脚本。
- 0--80 m 分支保持模型张量 `32x128x128` 不变，X 轴物理分辨率由 `120/128 m` 改为 `80/128 m`；80--120 m 不进入训练和正式评价，地图端必须标为 unknown，不能解释为 free。
- observed mask 稀疏文件采用精确字段 `protocol/coords/shape/pc_range` 且禁用 pickle；正式 Dataset 对缺文件、协议、shape 和物理范围均 fail-closed。旧数据只有未开启正式门禁时才可动态重建，避免破坏历史诊断。
- 时间切分必须以每帧 Radar timestamp 而不是文件数量作为 purge 判据；artifact 同时绑定 manifest content hash 与 `radar_ir_sync` 文件 hash，加载时用当前数据重建全文比对，避免只校验 JSON 自身 hash 而遗漏数据漂移。
- normalization 的 `formal` 不能再由 `max_frames==0` 单独决定；现在只有“正式 split + 统计全部 train IDs”才为 true，旧全场景 artifact 因包含验证帧只能作为 legacy/diagnostic。
- 直接把 streaming map 的单一 `pc_range` 改成 80 m 会丢失 80--120 m unknown 地图域；继续传 120 m 又会把模型 128 格错误解释为 120 m。必须分离 evidence range 与 map range，并允许小 evidence 张量投影到更大的同分辨率地图。
- 3 s purge 是基于同网格 target 重合率的防邻帧泄漏下界，不是世界坐标统计独立性的证明；审计没有 pose warp，因此结论按保守工程门禁使用。

## 0--80 m 分支实施结果

- fresh smoke 仅处理 garden 4 帧，输出到独立测试目录；五模态 training manifest 校验通过，正式 Dataset 从持久 mask、真实 IR 与真实标定加载 4 个样本，首帧 observed voxel 为 24922。
- source 物理网格由 120 m 的 `600x200x80` 收敛为 80 m 的 `400x200x80`，X 向源体素数减少 33.3%；模型张量仍为 `32x128x128`，因此 checkpoint 结构尺寸不变，但物理 X 分辨率改为 0.625 m，旧 checkpoint 仍不能跨协议 resume。
- 新监督协议没有把 80--120 m 改成 free：该区间不进入 formal target/loss/metric；地图证据范围与地图范围已经分离，80--120 m 初值保持 probability=0.5、unknown mass=1。
- 独立 CD 入口已经取消自动探测 `cd_best.pt`，并接入 split/data protocol/normalization 身份；最终复审仍需确认 formal 主函数与 unified 入口使用同一组真实数据门禁。
- 最终窄审查确认独立 CD 的正式入口缺少 unified 已有的 `scene_names/real_ir/real_calibration/lidar_frame/persisted_observed` 启动门禁；Dataset 虽会接收这些布尔值，但配置漏写时会静默变成 false，属于接口不对称，需在构建模型和 DataLoader 前直接拒绝。
- inference launcher 现在指向新的 80 m deployment root，但本阶段没有生成 deployment profile 数据视图；因此正式推理仍会在缺 deployment manifest 时 fail-closed，这是下一阶段的明确依赖，不应把训练数据根直接冒充部署根。

## 严格 deployment-profile 生成链复核

- `dataset_manifest.py` 已定义 deployment profile，只要求 `radar_voxel/ir_image` 并绑定 preprocessing script、三组外参与 thermal intrinsics、Radar--IR sync；正式推理 launcher 也已经要求 `expected_profile=deployment`。
- 当前缺口是生产器而不是 validator：仓库没有从 training-profile 根派生独立 deployment root 的命令，也没有父 training manifest/data protocol 的派生收据，launcher 指向的新 deployment 路径因此必然缺失。
- 现有 manifest 构建器只扫描 profile 要求的目录，不检查场景根是否额外带 LiDAR/target/observed；因此不能把 training 场景复制后直接写 deployment manifest，否则“只含部署模态”的边界无法由 manifest 证明。
- `preprocess-v2.sh` 已掌握 training root、Raw root、preprocess script 与 calibration 路径，适合在全量 training artifacts 完成后追加 deployment 视图步骤；测试场景配置当前由推理 launcher 从 `data_loading_config.yml` 读取。
- 协议决定：training manifest 保持 schema v2；新严格 deployment view 使用 schema v3，避免对既有 v2 deployment 合同做无版本变更。v3 场景根只允许 Radar、IR、policy、源 training manifest 快照、deployment receipt 和最终 manifest。
- 默认物化方式采用普通文件硬链接以避免重复占用大体素数据；明确禁止 symlink，不修改源文件权限。manifest/receipt 绑定每个文件内容，因此源或视图任一侧被原地修改都会在正式推理前失败。另保留显式 copy 模式用于跨文件系统或独立归档。
- 源 training manifest 快照随视图携带，deployment validator 将逐项比较 Radar/IR 记录、policy hash、frame/frame-count 和共享 provenance；这比只在 receipt 中声明一个无法解析的父 hash 更可审计。
- 推理 Python 身份函数原先只返回 deployment manifest、Radar--IR sync、两项标定与 frame；现应复用完整 view validator，并把 `source_training_manifest_content_sha256`、receipt SHA 和物化模式一起写入 `inference_run.json` 的 deployment identity。
- 三个正式 launcher 原先逐场景调用通用 manifest validator，不能验证根级精确场景集合和 dataset receipt；已决定改为一次调用 deployment dataset validator，逐帧推理内部仍对当前 scene 再做自包含验证，形成启动前全根门禁和模型构建后单场景门禁两层检查。
- 4 帧 smoke 纠正了一个接口假设：`radar_ir_sync.csv` 由预处理写入 training scene，而不是来自 Raw 根。deployment 生产器应直接从已验证 training scene 读取并复制该同步记录，避免额外 Raw 路径依赖，也让部署视图对逐帧 Radar--IR 配对身份自包含。
- 真实 4 帧 deployment smoke 已生成 schema v3：dataset receipt hash `bf9fc3ea...a6373`、scene manifest hash `257f05b3...ba811`、父 training manifest content hash `a0a70ea5...01be34`，均可重建验证。
- smoke 的 Radar/IR 源文件与视图文件 inode 相同且 link count=2，确认 hardlink 模式不复制大体素数据；视图根只包含 Radar、IR、同步 CSV、policy、父 manifest 快照和两级 receipt/manifest，没有 LiDAR/target/observed。
- 最终安全复审发现：同步 CSV 虽已复制并由 manifest 声明 hash，但通用 manifest validator不会重新打开任意 provenance 文件；deployment view validator 必须显式重算视图内同步 CSV hash，并在解析 receipt/父快照前先比较两者文件 hash，避免篡改导致非协议异常。
- 服务器传输会把 hardlink 展开为独立文件但不改变内容；receipt 因此只能记录 `materialization_mode_at_creation`，不能把 inode 关系作为运行时协议。正式校验继续只依赖普通文件、精确目录项和 SHA-256，保证复制到服务器后仍有效。

## Deployment observed/frame/risk 运行时安全审计

- 当前 `inference.py` 只保存 `*_voxel.npy`、可选 uncertainty 和 point cloud，没有生成与预测同帧键的 `*_observed_mask.npy`，因此 deployment Radar/IR 推理结果无法直接满足地图端 observed/free/unknown 合同。
- `streaming_map_update.py` 的 `--observed_mask_dir` 仍为 optional，缺失 mask 时继续运行；没有 pose CSV 时设置 `pose_mode=identity_legacy` 并使用单位阵。该行为可以保留为显式 legacy 诊断，但不能用于 formal map。
- `LazyLocalMapQuery.query_proximity()` 仍用 `min_dist < 5.0 and uncertainty < 0.7` 计算 `is_risky`；空地图、搜索窗外和高不确定性分支均返回不危险，属于已复现的 fail-open。
- 本轮先修复不需重训的 runtime 契约；point-count/Doppler-validity 与 `UncertaintyHead` 输入调整会影响数据/权重协议，留到下一子阶段。
- formal inference launcher 已经统一带 `--save_voxel`，因此可在 formal `require_real_ir` 模式自动按同一 frame key 保存 observed mask，无需新增会被遗漏的可选 launcher 开关。mask 必须由输入 Radar endpoint ray visibility 产生，不能从预测 sigmoid 产生。
- 推理输出体素声明为 LiDAR frame，而 mapping 核心当前把 `T_local_body` 直接作用到输入 voxel。最小兼容修复是在地图更新接口增加显式 `T_body_voxel`，使用 `T_local_voxel=T_local_body@T_body_voxel` 投影 evidence，并同时审计保存三者；旧调用默认 voxel=body。
- formal map 应显式要求 inference run metadata、逐帧 observed mask、body→local pose CSV 和严格 4x4 LiDAR→body 外参。若 IR BEV 尚无独立 frame 合同，formal 模式先拒绝该可选输入，避免再次引入隐式 frame。
- 风险接口保留历史 `distance/uncertainty/is_risky` 键，同时增加 `state=clear|obstacle|unknown`、`reason` 与动态 `safety_distance_m`，从而不立即破坏现有消费者。
- GREEN 后静态审查发现 formal map 仍可能消费未被 inference receipt 绑定的 uncertainty、IR BEV、dynamic evidence 或 prior DEM；这些输入的 frame/provenance 尚未统一，formal 本阶段必须 fail-closed 拒绝，legacy 保留既有能力。
- observed receipt 记录的 Radar→LiDAR hash 还需与 `deployment_identity.calibration_sha256.radar_to_lidar` 交叉一致；否则伪造 JSON 可形成内部自洽但数据来源错误的 mask 合同。
- 最终遮挡语义与 training observed ray 对齐：同一离散方向只向最近 Radar endpoint 铸造 free-space ray，但所有实测 endpoint 仍保留 observed，避免把近端障碍后的空间标为 free。
- garden 4 帧复测中 endpoint 数 983/964/952/994 不变，observed 数由 11744/11280/11222/11399 收缩到 11727/11263/11208/11384，所有 endpoint 仍被覆盖；这是遮挡保护带来的预期 unknown 增加。
- 默认 reaction=0.5 s、brake=8 m/s²、margin=5 m 时，35/50/70 m/s 对应安全距离 99.0625/186.25/346.25 m。当局部地图或查询半径不足以覆盖该距离时，formal 查询必须返回 unknown/risky，不允许回退固定 5 m clear。
- 稠密 `uint8` mask 是当前无重训最小合同，每帧 `32×128×128` 约 0.5 MiB，loop3 6432 帧约 3.14 GiB。现有 training 稀疏 mask 格式的坐标/来源语义不能直接冒充 deployment Radar visibility，因此本轮不扩展为新稀疏协议。
- 仓库 `Data/config` 中没有显式 LiDAR/Livox→body 外参。Radar→IMU 与 Radar→Livox 只能在机体 frame 语义被外部确认后组合，代码不会猜测 `body==IMU`；真实 formal map smoke 因此正确保持 fail-closed。
- loop3 已有 `gt_odom.txt`，共 6445 条 `timestamp tx ty tz qx qy qz qw`；候选 deployment sync 有 6432 个 Radar timestamp。GT 首时刻比 Radar 首帧晚 0.398283 s，因此前 4 帧不得外推或复制首 pose。
- 最近邻时间配对会让 6432 帧中 653 帧复用同一 GT pose；正确的候选生成应对平移做线性插值、对四元数做最短弧 SLERP。
- `calib_radar_to_imu.txt` 是无语义注释的 4×4 矩阵，`calib_radar_to_livox.txt` 明确声明 `p_l=R*p_r+T`。可计算 `T_imu_lidar=T_imu_radar@inv(T_lidar_radar)` 候选，但未确认前不能写入 formal `Data/config`。

## Mapping pose candidate 诊断结果

- 独立脚本输出 LiDAR→IMU-body、GT-as-IMU pose 与 GT-as-LiDAR pose 三组候选，全部携带 `formal=false`，并记录输入/输出 SHA-256、假设和 uncovered 明细。
- loop3 严格 0.2 s 门限下覆盖 6162/6432；未覆盖由 4 帧早于 GT 和 266 帧过大 GT gap 构成，拒绝 gap 范围为 `0.200822--0.261580 s`。
- 正式外参/pose loader 已加入内容级拒绝，诊断文件不能靠改路径直接进入 formal map。仍需权威确认 Radar→IMU 方向与 GT export frame 后，才能发布新的正式合同。

## Mapping frame 来源审计

- 仓库内文件名与全文搜索未发现 ROS bag、TF/static TF 转储、`gt_odom.txt` 导出器、R2LIVE 源码副本或 `calib_radar_to_imu.txt` 生成脚本；命中项只有本轮诊断/测试和记录文件。
- `calib_radar_to_imu.txt` 与 `calib_radar_to_livox.txt` 的本地 mtime 相同，但只有后者携带 `p_l=R*p_r+T` 方向注释；相同复制时间不能证明前者方向。
- `gt_odom.txt` 仅保留数值列和通用列名，ROS `header.frame_id`、`child_frame_id` 及 exporter 变量均已丢失，不能从该文本本身恢复 pose frame。
- 扩大到 `/home/zxj` 后找到原始下载目录与 loop3 三段 bag；工作区也保留三段 bag（约 3.22 GB、3.22 GB、90.5 MB），因此可以直接审计 topic/TF，而不必只做几何猜测。
- 下载目录与工作区的 Radar→IMU 文件 SHA-256 都是 `e71bd907...c6799`，GT 文件都是 `eb0de1bc...ce1`，说明当前文件未相对原始下载副本发生漂移，但原始副本本身仍缺方向注释。
- ROS Noetic 的 `rosbag` 可执行文件存在；直接 `/usr/bin/python3 import rosbag` 因未加载 ROS `PYTHONPATH` 失败，后续改为显式 source `/opt/ros/noetic/setup.bash`，不重复裸 Python 导入。
- loop3 bag 不含 Odometry 或 `/tf_static`。唯一 `/tf` 对是 `map→base_link`，第一段中 49898 条均为恒等变换；它不能提供 Radar↔IMU/LiDAR 安装外参，也不能还原 `gt_odom.txt` 的 exporter frame。
- 第一段 bag 的 header 声明：VectorNav IMU=`imu_frame`，Livox 点云和 Livox IMU=`livox_frame`，Radar 点云=`base_link`。但 bag 缺少 `base_link↔imu_frame/livox_frame` TF，且 Radar 点是否已被驱动实际变换到 base_link 尚未由代码/文档证明，不能只依赖 frame_id 取消现有 Radar→LiDAR 外参。
- bag 同时包含 VectorNav IMU 与 Livox 内置 IMU；无注释的 `calib_radar_to_imu.txt` 连“IMU 指哪一个”也未在文件内声明，这是方向之外的新增正式 blocker。
- 原始 `calib_intrinsic_imu.yaml` 明确把 intrinsic IMU topic 指向 `/vectornav/IMU`（bag 实际大小写为 `/vectornav/imu`），因此 Radar→IMU 文件的目标传感器可强关联为 VectorNav，而不是 Livox 内置 IMU。
- NTU4DRadLM 论文说明 `extrinsic_xx_to_xx` 表示从前一传感器到后一传感器的外参并遵循 KITTI 格式；结合原始文件名 `calib_radar_to_imu.txt`，Radar→IMU 的方向已有官方强证据，应优先解释为 Radar 坐标到 VectorNav IMU 坐标，而不是取逆。
- 论文还说明正式 ground truth 原本同时有 `gt_odom.bag` 和由其生成的 `gt_odom.txt`，但当前下载/工作区只找到 txt；缺失的 GT bag 本应携带 Odometry frame，是确认 GT pose frame 的最直接证据。
- 一个第三方 NTU4DRadLM loader 在使用同一 Radar→IMU 数值前对 IMU Y/Z 做符号翻转，形成 `diag(1,-1,-1) @ T_file`。这说明“传感器 IMU frame”与下游算法 body convention 可能不同；即使方向确认，也不能把原文件未经轴约定转换直接等同于 airborne `body`。
- 官方 4DRadarSLAM 说明其预处理会把输入 Radar 点云显式变换到 Livox LiDAR frame 后再做里程计，因此 bag 中 Radar 消息的 `frame_id=base_link` 不能证明点坐标已处于 vehicle body；现有项目继续使用有方向注释的 Radar→Livox 外参是合理的。
- 公开仓库/issue 搜索尚未找到 NTU4DRadLM `gt_odom.bag` 的 frame 字段或 txt exporter。R2LIVE/FAST-LIO 类实现通常发布内部 body/IMU state，但数据集还做了 pose-graph drift correction，不能把上游默认发布语义无条件传递到最终 txt。
- 经验诊断的可辨识性需显式限制：GT-as-LiDAR 分支在将 LiDAR 点投到 local 时外参代数消去，因此 LiDAR 自重合指标不能判断 Radar→IMU 方向；它只能比较“GT 是 body 且使用某外参”与“GT 已是 LiDAR pose”的轨迹解释。
- 候选预处理 `loop3` 的 `preprocess_policy.json` 固定 `align_to=lidar`、`pc_range=[0,-20,-6,120,20,10]`、`voxel_size=[0.2,0.2,0.2]`、`invert_calib=false`；`lidar_voxel/*.npz` 为稀疏 `coords/features/shape`，共有 6432 帧。后续必须沿保存代码确认 `coords` 轴序后再恢复物理点，不能依据样本范围猜测。
- 多窗口诊断只比较两个 GT frame 解释在共同 covered 帧上的静态 LiDAR 重合度，报告稳健最近邻距离和阈值内比例；因外参在 GT-as-LiDAR 分支代数消去，报告必须声明它不能确认 Radar→IMU direction，也不能替代 CAD/原始 `gt_odom.bag`。
- 预处理调用链已经证明稀疏坐标轴序：`voxelize_pcl_airborne_optimized()` 按 `(x_idx,y_idx,z_idx)` 构造 `[X,Y,Z,4]` 网格，`save_voxel()` 用 `np.where(occupied)` 原样保存三列。因此物理中心严格为 `pc_min + (coords + 0.5) * voxel_size`，无需再做 ZXY/XYZ 猜测。
- 现有 `shared_visibility_eval.py` 只比较同帧 Radar/LiDAR 可见域，`build_mapping_pose_candidates.py` 只生成位姿候选；二者都没有跨帧 pose-conditioned LiDAR 重合。为保持单一职责，新建一个 alignment 诊断脚本比把时间重合逻辑塞进候选生成器更合理。
- 两个候选 CSV 都声明输出方向为 IMU-body→local。诊断必须统一计算 `T_local_lidar=T_local_body@T_body_lidar`：GT-as-IMU 分支实际使用外参；GT-as-LiDAR 分支严格还原原始 GT pose，外参代数消去。这一等式会在合成测试中固定。
- loop3 候选轨迹的 1.0 s 配对约有 5322 对，旋转角 q90=4.15°、q95=5.60°、最大 21.41°，平移 q90=8.45 m；选择 1.0±0.15 s、旋转≥3°、平移≤12 m 的跨全序列均匀子集，可在保持视野重合的同时获得外参旋转可辨识性。2--5 s 虽转角更大，但平移中位数达 14--34 m，新增视野会更强地污染静态重合指标。
- 诊断默认只使用 2--50 m LiDAR 体素中心，并在 local frame 计算双向最近邻的 median/p90 与 0.5/1.0 m 命中率；两个假设必须使用完全相同的 frame pairs 和点集，结果仅给 empirical ranking，不设 formal winner。
- loop3 首轮 48 对结果全部由 GT-as-LiDAR 获得更低 pair median：其 pair-median NN 总结中位数为 0.4138 m，GT-as-IMU 为 2.2446 m；差值中位数 1.7397 m，最小仍为 0.6744 m。0.5 m 命中率中位数分别为 56.95% 与 4.04%，经验反证很强。
- 绝对残差不能解释成纯标定误差：候选 pose 对齐 Radar timestamp，而 `align_to=lidar` 的体素来自 LiDAR reference time，二者仍有至多严格同步门限量级的时差；场景动态、遮挡与 1 s 平移也会贡献 NN 尾部。应主要使用同 pair 的假设间相对差，而不是把 0.414 m 当作传感器精度。
- 代码审查确认这是实际接口不匹配：`build_mapping_pose_candidates.py` 从 `radar_ir_sync.csv` 取 Radar timestamp，但预处理在 `align_to=lidar` 时明确以 LiDAR timestamp 作为体素 reference；当前 processed manifest 又没有携带 Radar--LiDAR sync 收据。首轮排序可保留为 diagnostic v1，但在修复时间基准并重生新目录前不能视为最终经验结果。
- overlap 脚本目前验证所选 voxel 文件 hash，却只记录 manifest 文件 hash，未先验证 manifest 自身 `content_sha256`；被重写的 manifest 可以连同伪造 record hash 一起被信任。应增加 canonical manifest content hash 门禁，再运行 v2 结果。
- 精确来源已定位为 `Data/NTU4DRadLM_Raw_p1_01_candidate/loop3/radar_lidar_sync.csv`，正是 `preprocess-v2.sh` 的 `RAW_ROOT`。它有 6432 行，Radar timestamp 与 processed `radar_ir_sync.csv` 逐帧完全一致，LiDAR timestamp 为独立列，SHA-256=`3ce134bd...ab79`；因此可显式修复而无需从文件名猜时间。
- processed policy 虽声明 `radar_lidar_sync_filename` 和 LiDAR reference，却没有把 sync CSV 复制到 processed scene，legacy manifest v1 也没绑定其 hash。新 candidate 应携带 sync snapshot/hash，使后续 overlap 结果不依赖易漂移的外部绝对路径。
- 修正后的 candidate v2 以 LiDAR timestamp 插值，coverage 为 6165/6432（比 Radar-time v1 多 3 帧），并封存 SHA-256=`3ce134bd...ab79` 的 sync snapshot。overlap v2 验证 manifest self-hash 后仍为 48/48 支持 GT-as-LiDAR。
- v2 的 pair-median NN 中位数为 GT-as-LiDAR 0.4123 m、GT-as-IMU 2.3012 m；paired 差值中位数 1.8102 m、最小 0.9297 m。修正参考时间没有推翻排序，且最弱 pair 的分离反而增大；这提高了反证稳健性，但 formal blocker 仍不变。
- 参数敏感性：0.5 s/≥2° 的 32 对全部支持 GT-as-LiDAR，pair-median 差值中位数 1.0291 m、最小 0.4115 m；2.0 s/≥5° 的 32 对有 30 对支持，汇总差值中位数 2.7938 m，但最差 pair 反向 -3.0211 m。长窗口受共同视野减少/动态影响，验证正式摘要应采用 0.5--1.0 s paired aggregate，不要求逐 pair 全胜。
- 缺权威 body frame 时可避免未经验证的 LiDAR→body：离线地图直接消费 `T_local_voxel=T_local_lidar`。该模式只能发布项目内经验收据，不能改变 airborne formal 门禁；需要在 CLI 和 run metadata 中与 body 链模式显式互斥。
- 现有 `streaming_map_update.py` 的 `--formal_mapping` 把 inference run、observed mask、body→local pose 和 LiDAR→body 外参作为四项硬依赖，随后统一计算 `T_local_body@T_body_voxel`；metadata 固定声明 `pose_direction=body_to_local`。直接复用 `pose_file` 填 LiDAR pose 会造成接口语义造假，必须新增独立模式/receipt，而不能只传单位外参。
- formal 共用的安全门禁（完整 inference/mask、禁止 frame_limit、拒绝未绑定 uncertainty/IR/dynamic/DEM/target）应抽成 strict mapping 条件，让 airborne formal 与 offline empirical 都保留；两种模式只在 pose 来源与坐标组合处不同。
- 现有候选审计协议为 `mapping_pose_candidate_diagnostic_v2`，明确 `formal=false`、`candidate_only=true`、`no_extrapolation=true`，覆盖 6165/6432 帧；它可作为经验合同来源，但不能直接改名冒充机载正式位姿。
- 现有重合诊断协议为 `mapping_pose_overlap_diagnostic_v1`，明确 `formal=false`、`diagnostic_only=true`；其经验排序首选 `gt_as_lidar`，且 `gt_as_lidar_external_cancels=true`，因此只足以支持直接 `T_local_lidar` 离线链，不能推出 LiDAR 到 airborne body 外参。
- 经验合同构建器需要核验候选 audit、重合 audit、LiDAR 时间同步快照和候选位姿文件的哈希，并生成干净的 `lidar_to_local.csv`；运行时必须再次校验 receipt 与成员文件，禁止仅信任文件名或外部路径。
- loop3 经验 pose 只覆盖 6165/6432 帧，而正式 inference receipt 可覆盖完整 6432 帧；离线入口不能靠 `frame_limit` 或目录手工删帧。经验 receipt 必须成为唯一的帧子集选择器：验证其 pose 帧是 inference 帧的有序子集，并只融合这 6165 帧，同时在 map receipt 中记录 available/selected/uncovered 数量。
- inference receipt 的完整 records digest 仍应验证；receipt-bound 子集模式只需重算被消费帧的实体 mask/hash，未消费帧的声明继续由 inference metadata hash 和 records digest 绑定。这不会放宽原 airborne formal 的“逐帧实体全验证”行为。
- 经验模式代码审查暴露了一个既有 provenance 缺口：`inference_run.json` 只绑定 observed-mask 文件哈希，不绑定实际被地图消费的 `*_voxel.npy` 预测内容。当前 formal/empirical map 即使通过 inference metadata，也无法发现 prediction voxel 在推理后被替换；应在 inference 发布时增加逐帧 prediction receipt，并在地图预检中对被消费帧重算哈希。
- prediction receipt 的记录格式和摘要算法若在 inference/map 两端复制会形成新的隐形依赖；已集中到轻量 `prediction_artifact_protocol.py`，训练/模型模块不参与导入。正式记录固定 NPY、LiDAR frame、CZXY、float32、逐帧 shape/hash；地图仍负责实体文件与有限值预检。
- 最终审查未发现 empirical 子集绕过：loader 要求完整 6432 帧集合等于 selected 与 uncovered 的并集，strict inference contract 先验证完整 records digest 和帧序，再对 6165 个实际消费 prediction/mask 重算实体 hash；`frame_limit` 和松散 pose/extrinsic 参数均被禁止。
- `test/result/` 受项目 `.gitignore` 的 `test/*` 规则管理，因此真实 receipt 与 `INDEX.md` 更新存在于工作区但不会出现在普通 `git status`；它们仍按测试目录规则保留，未通过删除或强制 add 改变历史结果管理策略。
- sidecar 调用链入口已定位：`NTU4DRadLM_pre_processing.py::_parallel_frame_worker()` 在时间对齐后调用 `voxelize_pcl_airborne_optimized()`，当前只返回四通道稠密 voxel，随后 `save_voxel()` 仅为 occupied 体素持久化 `coords/features/shape`；point count 与 Doppler 有效性在聚合后丢失。
- 场景输出当前固定创建 radar/lidar/target/observed/IR 五类目录，`preprocess_policy.json` 只声明四通道语义；training manifest v2 的精确模态集合和 deployment view 的精确两模态集合意味着不能直接新增第六个 modality 而不显式升级/区分协议。
- 最小兼容方向应优先把统计数组作为每帧 radar NPZ 内的额外键，而不是新增目录：旧 `load_sparse_voxel()` 可继续只读 `coords/features/shape`，manifest 的既有 radar 文件 SHA 会自然绑定统计内容，deployment hardlink/copy 也可原样携带；formal Dataset 再显式验证/加载额外键。
- `voxelize_pcl_airborne_optimized()` 已计算 `unique_counts`，但当前 channel 2/3 无法区分“无 Doppler 列或无有效值”与真实零均值/零方差；每体素 `point_count` 与 `doppler_valid_count` 可在同一次 scatter 中低成本获得，且不必改变四通道数值。
- Dataset 当前用通用 `load_sparse_voxel()` 恢复四通道，并在 meta 中返回 observed/calibration/policy；可新增 Radar 专用严格 loader，验证统计协议、稀疏坐标对齐、整数范围和 count 关系，只把每帧摘要作为审计 metadata 交给 collator，避免把未使用的全分辨率统计稠密化占用内存。
- formal data protocol 已绑定完整 training manifest hash，而 manifest radar records 又绑定每个 NPZ 文件 hash，policy hash也在 manifest 内，因此新增 NPZ 键和 policy 协议会自然改变正式 data identity；本阶段无需给 checkpoint schema 增加新字段或改网络结构。
- deployment view 直接复用 source manifest 的 radar records 并 hardlink/copy 原 NPZ，所以统计键会随文件无损携带。现阶段模型不消费这些键，推理通用 loader保持兼容；待未来 `UncertaintyHead` 迁移时再提升为部署运行时必需输入。
- formal unified 和 standalone CD 都从同一 `NTU4DRadLM_VoxelDataset` 构造数据；新增 `require_radar_statistics` 参数需在两处由 formal 模式强制为 true，legacy/简单 inference condition 保持 false，避免旧诊断路径无意中升级。
- `collate_voxel_samples()` 目前只对 `preprocess_policy` 保留逐样本原始对象；统计摘要包含序列级列表/整数，必须加入同类 audit-only 列表白名单，否则 default collate 会递归改变结构或在可变序列长度时失败。
- 现有 `run_formal_mini_8gb.sh` 仍硬编码旧 `full120/p1_04` 数据、artifact 和 `formal_mini_chain_v1`，与当前 0--80 m formal v2/sidecar 数据链不兼容，不能作为本轮“下一步训练脚本”原样推荐。需要新增或参数化一个 v2 sidecar 专用 guarded 入口，并在 preflight 阶段先验证数据/normalization/data protocol 实物。
- 默认配置与 formal unified/CD 已有 `require_persisted_observed_mask=true` 对称门禁，可按同一模式增加 `require_radar_statistics=true`；`assert_formal_dataset_preflight()` 还需确认首样本摘要协议，避免只靠 YAML 布尔值。
- `train_minimal.sh` 从默认配置派生，若直接把默认统计门禁设为 true，会让历史 full120 formal-mini 无意失败。应增加显式 `MINI_REQUIRE_RADAR_STATISTICS`，旧 wrapper 固定 0，新 0--80 m sidecar wrapper 固定 1；统一训练代码只按配置加载，formal_chain_v2 另外强制 true。
- 现有 VAE formal preflight 行为测试的 mock metadata 需增加 statistics 摘要，否则会正确命中新门禁；独立 CD 配置测试也需把 `require_radar_statistics` 纳入逐项 false 拒绝集合。
- `train_unified.sh` 已显式覆盖 observed/IR/calibration 等正式布尔字段；即使 default YAML 已设 true，也应在 launcher 派生配置中显式写 `require_radar_statistics=true`，防止用户基础 YAML 漂移绕过。
## 2026-08-27 Radar 统计契约训练入口复核

- `diffusion_consistency_radar/launch/train_unified.sh` 当前把 VAE/LDM 的 `CUDA_VISIBLE_DEVICES` 固定为 `0,1`，这对单卡 8 GB 笔记本和单卡服务器形成了隐形双卡依赖。
- 正式训练入口在启动训练前应遍历 manifest 中实际选用场景的 Radar NPZ，并通过共享的严格加载器验证 `point_count`、`doppler_valid_count` 与协议版本；这样可以在 GPU 训练开始前暴露旧数据或损坏 sidecar。
- 训练脚本需要提供 `PREFLIGHT_ONLY=1`，使用户能够在不创建训练结果、不占用 GPU 的情况下完成数据、manifest、normalization artifact、formal data protocol 与 Radar 统计契约验收。
- Radar 统计摘要对应磁盘中持久化、尚未数据增强的体素，且当前不进入模型。Dataset 元数据应显式声明该引用边界，避免下游把统计摘要误认为增强后张量或模型输入。
- launcher 的 `all` 模式通过重新调用自身执行三个阶段，因此 GPU 选择和预检变量必须通过环境变量继承；不能只修改某个 Python 入口。
- 现有 launcher 在 normalization 校验完成后立即写入 `.default_config.train_override.yaml`。只预检模式应在写运行配置前退出，确保预检不创建或覆盖训练配置和结果。
- Radar 统计摘要的类型注解当前写成 `Dict[str, int]`，但摘要含字符串 `protocol`，属于静态接口不匹配；应改为 `Dict[str, object]`。
- Dataset 目前把统计摘要附在 metadata 中，但没有声明它对应增强前持久化 Radar，且未声明 `model_consumed=False`；需要通过统一辅助函数补齐，避免构造阶段缓存和 `__getitem__` 返回值漂移。
- manifest 的 Radar 记录提供场景内相对 `path`，可在 launcher 预检中直接遍历，而不依赖文件名猜测或重新扫描目录。
- 当前工作区尚不存在 `Data/NTU4DRadLM_Pre_formal_v2_80m_86p8_v1` 和对应 v2 normalization artifact，因此现在不能诚实地执行正式预检或训练；下一步必须先运行 v2 全量预处理链。
- 根 README 仍展示旧 `run_formal_mini_8gb.sh` 入口。该 mini 脚本绑定旧 full120 数据合同，不能作为本轮 formal v2 Radar statistics 数据的下一步训练命令；文档必须明确区分 legacy mini 与 formal v2。
- `preprocess-v2.sh` 已固定 0--80 m、Doppler 86.8 m/s、garden/loop3 全量、train/purge/validation、train-only normalization、formal data protocol 和 deployment v3 共 8 步；它会拒绝任何已有输出，不能用于续跑或覆盖。
- 因此正式训练前的准确顺序是：完整运行 8 步 → 读取新 normalization SHA-256 → `PREFLIGHT_ONLY=1` 验收 → 仅在服务器启动 VAE。当前 8 GB 笔记本不应承担全量 VAE 长训练。
- 最终接口复核确认：预处理 policy 确实写入 `radar_statistics_protocol` 与 `radar_statistics_model_consumed=false`；默认配置、unified train 和 standalone CD 均强制 `require_radar_statistics=true`；训练场景来自 `data_loading_config.yml` 的精确 `garden`，不存在场景名猜测。
- 当前工作树包含此前多个阶段的大量未提交修改和旧 planning 目录删除，本轮未回滚、移动或清理这些用户/既有变更；最终报告只归纳本子阶段实际修改范围。
