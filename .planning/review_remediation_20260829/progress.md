<!-- 文件功能：记录顺序修复每一步修改、测试结果和未完成事项。 -->
# 顺序修复进展

- 2026-08-29：按用户确认建立八阶段实施计划。
- 2026-08-29：读取 planning-with-files 和 `test/AGENTS.md`，确认只运行短时回归。
- 2026-08-29：确认并保护用户现有服务器 `preprocess.sh` 修改。
- 2026-08-29：阶段 1 开始，准备先补配置合同测试。
- 2026-08-29：完成阶段 1 fresh/resume VAE 配置及 launcher 派生调用链审计；决定默认 YAML 只保留 active BCE+Dice 参数，运行时 `num_gpus` 继续由设备列表派生。
- 2026-08-29：阶段 1 RED 测试运行 5 项，1 项按预期失败，证明默认 YAML 仍存在静态 `num_gpus: 4`；其余 epoch/覆盖合同通过。
- 2026-08-29：修改 `default_config.yaml` 和 YAML 合同测试；删除静态 `num_gpus` 与 formal 无效 legacy-MSE 参数，显式声明 active BCE+Dice 参数。
- 2026-08-29：阶段 1 共 65 项短回归全部通过；未启动 GPU 训练或写入实验结果。
- 2026-08-29：阶段 1 完成，进入阶段 2 LDM observed-mask 调用链与 RED 测试设计。
- 2026-08-29：确认 LDM 训练和验证虽然已搬运 metadata，但 observed mask 尚未进入主 latent loss、decoded 组件或 checkpoint-selection 指标。
- 2026-08-29：冻结阶段 2 voxel/column/latent mask 语义和 legacy 兼容边界，准备新增未知区扰动不影响损失/指标的 RED 测试。
- 2026-08-29：新增 6 项 LDM observed-domain RED 测试，全部因现有函数不接受 `observed_mask` 而失败，确认测试确实覆盖接口缺口。
- 2026-08-29：实现 observed-mask 数学层并复跑 6 项反例全部通过；开始接入 trainer、validation 和 checkpoint 身份。
## 2026-08-29 阶段 2 进行中

- 数学层 6 个 observed-mask 测试由 RED 转为 PASS。
- 原有 LDM 垂直结构/训练工具 81 个短测试全部通过，证明无 mask 的 legacy 行为暂未回归。
- 下一步：补训练传参、formal 缺失 mask、验证域和 checkpoint 协议的接口级测试。

## 2026-08-29 阶段 2 完成

- 新增 6 个数学层测试，并将原 LDM 工具测试扩充到 83 个；共 89 个测试通过。
- 训练调用链现会传递 persisted observed mask，formal 缺失输入会 fail closed。
- latent MSE、uncertainty NLL、decoded 辅助损失和验证 IoU 统一限制在 observed domain。
- checkpoint 写入 observation-supervision protocol，formal resume 对缺失或不匹配协议执行严格拒绝。
- 保留 legacy 无 mask 的全域行为，且恢复了 `decoded_vertical_structure_losses` 原有位置参数顺序兼容。
- 一次只读搜索引用了不存在的 `diffusion_consistency_radar/data/dataset.py`，实际 loader 位于 `cm/dataset_loader.py`；该错误不影响代码或测试。

## 2026-08-29 阶段 3 启动

- 已沿 `IR2DFeatureExtractor -> CompleteDualModalityPerceptionNet -> DDP -> CD EMA` 调用链确认问题存在。
- 预定最小修复：集中式多卡 SyncBatchNorm 转换，加上具名且类型安全的 EMA parameter/buffer 更新协议。
- RED 已确认：新增测试因缺少 `DDP_NORMALIZATION_PROTOCOL`、`prepare_model_for_distributed` 和 `CD_EMA_UPDATE_PROTOCOL` 导入失败，符合修复前预期。
- GREEN：DDP helper 16/16 PASS；CD entrypoint 回归 PASS。多卡模型会在 wrapper/optimizer 前转换 SyncBatchNorm，单卡保持普通 BatchNorm。
- CD EMA 已按名称更新 parameters 与 floating buffers，并直接复制整数 buffers；checkpoint 记录 `named_parameter_and_buffer_ema_v1`。
- 第二轮 RED：旧多卡 checkpoint 尚未因缺少 normalization protocol 被拒绝；CD 尚无独立 resume EMA-protocol 校验函数。两项均按预期失败。
- 第二轮 GREEN：DDP 协议 17/17 PASS，CD entrypoint 与 multimodal CD interface 均 PASS。
- 多卡 checkpoint 现记录 `sync_batchnorm_v1`；旧多卡局部 BatchNorm 状态不再允许正式恢复。formal CD resume 同样要求具名 parameter/buffer EMA 协议一致。
- 扩展 checkpoint 回归发现 1 个接口兼容问题：`OptimizedLDMTrainer.__new__` 最小测试对象没有 `observation_supervision_protocol`，checkpoint payload 直接访问属性导致 AttributeError；需改为协议常量 fallback 后重测。
- 更新首个最小 checkpoint fixture 后，LDM validation 回归又暴露同类隐式依赖：最小对象缺少 `require_persisted_observed_mask`。实现层应对这两个新增字段提供 legacy-safe fallback，而非要求所有诊断 fixture 重复 trainer 初始化细节。

## 2026-08-29 阶段 3 完成

- 多卡 normalization 固定为 `sync_batchnorm_v1`，转换发生在 DDP 和 optimizer 之前；单卡保留本地 BatchNorm。
- CD online/EMA 模型保持相同 normalization 类型；EMA 通过具名状态同步 parameters、浮点 buffers 与整数 counters。
- 新多卡/EMA 协议进入 checkpoint；旧多卡局部 BN 状态与旧 formal parameters-only EMA 轨迹均 fail closed。
- 回归：DDP 17/17、VAE/LDM checkpoint 26/26、LDM validation 5/5、YAML 5/5、CD 两组接口测试全部通过，`git diff --check` 通过。

## 2026-08-29 阶段 4A RED

- 新增正式推理 runtime 协议测试 3 项，修复前均按预期失败：缺少 CUDA 同步 helper、缺少 formal seed 校验、三个正式 launcher 均未显式传 seed。
- 首次合并补丁因同一文件包含两个 update block 被 `apply_patch` 拒绝，未产生部分写入；随后拆分为单文件 update 成功实施。

## 2026-08-29 阶段 4A 完成

- CLI 默认 seed 改为 42，formal mode 拒绝 `seed=-1`；三个正式 launcher 均显式传递 `INFERENCE_SEED`。
- CUDA 每帧计时在 `perf_counter()` 两侧同步目标设备；run metadata 记录 seed、sampling protocol 和 timing protocol。
- runtime 3/3、multimodal inference 38/38、formal inference 11/11 PASS。

## 2026-08-29 阶段 4B RED

- formal evaluator fixture 加入真实 `*_observed_mask.npy` 与 run metadata 合同后，原成功路径立即因“unknown file”失败，复现 P1-07；其余前置错误也被该未知文件提前遮蔽。
- GREEN：formal evaluator 12/12 PASS；现会校验 observed frame 集、shape、0/1、逐文件 SHA-256、voxel count 和 records digest，并在 CSV/summary 中记录该合同。
- 代码审查待办：evaluator 暂从 inference script 导入私有 digest helper；需抽成共享 artifact-protocol 模块，消除脚本间隐藏依赖后再关闭 4B。

## 2026-08-29 阶段 4B 完成

- observed protocol/digest 已抽到无 Torch 依赖的 `observed_artifact_protocol.py`，inference、saved evaluator、streaming map 共用同一实现。
- formal evaluator 验证逐帧 mask 内容收据并输出 observed 字段；formal inference 12/12、empirical pose/map 6/6 PASS，`git diff --check` 通过。

## 2026-08-29 阶段 4C 启动

- 首次同时修改 CD 与 inference 测试的补丁因 inference 测试上下文不匹配被整体拒绝，未产生部分修改；改为按文件精确定位后分别实施。
- 4C RED 已确认：CD 测试因缺少 validation protocol/selector 导入失败；推理新增的 formal EMA 选择与 legacy online fallback 两项测试因缺少 `resolve_inference_state_dict` 失败。现有其余 38 项推理接口测试仍通过。
- 4C 首轮实现已加入 checkpoint 选优合同和推理权重解析；短测发现 `math` 漏导入。另一次以模块名运行非 package 化的 `test/unit` 失败，属于测试调用方式错误，后续改回直接脚本运行。
- 4C 纯合同转绿：CD checkpoint/selector 测试通过，推理接口 40/40 PASS。
- 新增 trainer 级确定性 validation 反例：online/EMA 复用 sample-id 固定噪声，按 observed IoU 选中 EMA；formal 缺失 persisted mask 会拒绝。接口测试与 py_compile 通过。
- 已接通 unified 与 standalone 两个 CD validation 数据入口；standalone 也按正式 temporal split/阶段帧上限生成 train/validation 选择，并在 DDP 下使用无补齐 validation sampler。
- 接口/YAML 短回归通过：CD entrypoint、multimodal CD、默认 YAML 共 3 组；相关 py_compile 与 `git diff --check` 通过。
- 4C 收尾：formal resume 会核对 validation protocol/seed/sigma/threshold/metrics/selected source，推理端也核对共享协议与部署权重来源；legacy CD 保留 online fallback。
- 4C 回归通过：CD entrypoint、multimodal CD、multimodal inference 40/40、py_compile 与 `git diff --check`；未启动训练或 GPU 推理。进入 4D threshold artifact。
- 4D 共享合同已实现：observed-domain threshold sweep、IoU/recall/lower-threshold 选择、checkpoint SHA/stage/weight-source 绑定及原子 JSON 发布。
- 新增 threshold artifact 3 项短测试，覆盖 unknown 排除、并列规则与 checkpoint 内容漂移拒绝；全部 PASS，py_compile/diff check 通过。
- LDM/CD validation 已在同一次 decoded 输出上累计候选阈值 TP/FP/FN，并将所选部署权重对应的 sweep 写入 formal checkpoint；不增加训练监督或额外采样。
- 首轮 LDM RED 命中最小 fixture 缺候选集/扫描状态，补齐 formal fixture 后 LDM validation 5/5、VAE/LDM checkpoint 26/26、CD 两组接口回归通过。
- 新增 artifact builder；正式 LDM/CD launcher 改为传 `--threshold_artifact`，删除 `OCC_THRESHOLD` 自由入口。inference 校验 checkpoint hash/stage/部署权重源并把完整合同写入 run metadata。
- saved evaluator 在 formal run 中拒绝阈值 CLI override，并核对 metadata 阈值来自 validation artifact。formal inference 13/13、runtime 4/4、multimodal inference 40/40 与 artifact 3 项回归通过。
- 4D 收尾：formal unified/standalone 训练在 LDM/CD best checkpoint 落盘后自动构建 `occupancy_threshold.json`；evaluator summary 记录 validation artifact SHA 与阈值来源。
- 阶段 4 完成；未运行训练或推理。进入阶段 5 概率地图 observed/prediction/DEM 合同审计。

## 2026-08-29 阶段 5 启动

- 已恢复持久计划并复核先前审查证据；当前先处理 prediction 通道身份、显式 observed 权威边界和 DEM 单位。
- 首轮宽搜索输出过长发生截断，已记录并切换为 `rg` 定位符号加窄行段读取；该错误未修改代码。
- 计划先写小数组/metadata RED 测试，不运行训练、GPU 推理、全量预处理或离线全场景建图。
- 已完成第一轮函数级窄读：确认 prediction artifact 缺通道 schema、observed 显式 mask 被预测扩张、DEM 混入第 4 通道三项问题均存在。
- 已确认 formal streaming 消费的是 inference prediction 而非原始 Radar voxel；下一步对生成通道语义和 formal contract 分流写 RED。
- 已定位现有概率地图测试文件；其中“高第 4 通道同时降低 belief 并提高 DEM variance”的旧测试本身编码了错误合同，需要拆成“prediction 辅助通道不影响地图/DEM”和“显式模型高度不确定性可增加 DEM variance”。
- 阶段 5 RED：概率地图 48 项中新增 3 项因缺少 `observed_mask_authoritative`/`evidence_semantics` 接口报错；推理接口 40 项中 artifact v2 断言失败。其余既有用例通过，说明反例精确命中尚未实现的合同。
- 已确定最小兼容边界：formal/经验地图严格走 prediction-only v2；legacy 默认行为不删除。开始修改共享 artifact、地图核心和 streaming formal 调用点。
- 首轮 GREEN 回归仍有 3 个测试错误：测试使用了早期草案字符串 `generated_occupancy_prediction_v1`，实现与 artifact 最终固定为 `generated_occupancy_probability_v1`。这是测试常量漂移，不是地图行为失败；改为共享常量后重测。
- GREEN：概率地图 50/50、推理接口 40/40、经验位姿合同 6/6 PASS。新增 formal 集成反例证明：合法收据中的 ch0 超域会在输出目录创建前拒绝；mask 外 0.9 预测仍保持 unknown。
- 开始阶段 5 收尾代码审查：检查共享常量、formal/legacy 边界、地图 metadata 单位和所有 prediction artifact 消费者。
- 已消除 streaming 中 legacy semantics 字面量，统一引用共享常量；正式/经验地图协议已因语义变化显式升版，并补 DEM 单位 metadata。
- 阶段 5 文档与持久计划已同步；完成后进入阶段 6，只读审计 body-centered rolling、轨迹走廊查询和 ROS 接口边界。
- 阶段 5 收尾通过：formal inference 13/13、相关 `py_compile` 与 `git diff --check` 均通过；未执行长任务。
- 阶段 6 初始调用链审计完成：确认空间 rolling、trajectory corridor 和 ROS 可调用接口三项均不存在；开始拆分 6A/6B/6C 最小协议。
- 已冻结 6A/6B/6C 设计边界；先为 rolling source/destination 分离、新暴露 unknown 和状态收据编写 RED。
- 首次插入 6A 测试的补丁因旧测试末尾上下文与预期不一致被整体拒绝，未产生部分修改；改为精确定位相邻方法后重试。
- 6A RED：概率地图 52 项中新增 2 项均因 `GridMapConfig` 缺少 `rolling_enabled` 报错，其余 50 项保持通过；反例覆盖一格滚动和超窗口清空。
- 6A GREEN：概率地图 52/52 PASS。rolling window 已按整数体素移动，source evidence range 与 destination local bounds 分离，超窗口移动不回卷旧状态。
- 经验模式回归 5/6 PASS；唯一失败仍按旧固定 local map 索引 3 断言。rolling 后同一障碍应位于 body-relative 索引 1，需更新测试并同时核对 local bounds 收据。
- 经验模式断言已改为 rolling 后的 body-relative 索引 1，并增加窗口 local bounds 断言。一次回归误用了不存在的 `test/test_empirical_streaming_mapping.py`，已定位正确入口为 `test/unit/test_empirical_lidar_pose_contract.py`；该路径错误未修改代码。
- 正确经验模式回归仍为 5/6：新断言中索引 1 占用概率为 0.3078，证明不能仅机械更改旧索引。已进入 direct LiDAR pose x=3m、source 体素中心、rolling bounds 与 destination index 的调试。
- 调试确认 rolling 实现正确：direct LiDAR pose x=3m 使窗口从 `[3,0,0]` 移到 `[11,1,1]`，源占用体素中心 0.5m 落在新索引 0；索引 1 是显式 free 证据。已修正测试中的错误位姿/窗口预期，不改生产算法。
- 6A 回归转绿：经验位姿合同 6/6、概率地图 52/52 PASS。确认 source evidence 物理范围不随窗口移动，旧 local 证据无回卷，窗口外状态恢复 unknown。进入 6B 轨迹走廊查询。
- 6B RED 已确认：新增的轨迹走廊用例因 `LazyLocalMapQuery` 缺少 `query_trajectory_corridor` 而失败。反例同时锁定 clear、走廊内 obstacle、unknown fail-closed 和轨迹制动视界不足四种结果。
- 6B 核心 GREEN：`query_trajectory_corridor` 已实现制动距离内 local-frame 折线采样，强制有效采样间距不大于走廊半径，对 obstacle/unknown 立即 fail-closed，轨迹长度不足制动距离时也返回 unknown；针对性用例 PASS。下一步给 streaming 入口增加显式轨迹 artifact 合同，禁止默认伪造直线轨迹。
- streaming 调用链已确认：当前每帧只对 body/LiDAR 原点调用 `query_proximity`，CSV 与 `map_run.json` 也无轨迹身份字段。将在输出目录创建前加载逐帧 local 轨迹并精确校验消费帧集。
- streaming artifact RED 已确认：CLI 尚不识别 `--trajectory_file`/走廊参数。集成反例要求错帧 artifact 在输出目录前拒绝，合法 artifact 实际产生 corridor obstacle 结果和身份收据。
- streaming artifact GREEN：新增 `local_trajectory_frames_v1` 共享校验模块，严格要求 local 坐标、逐帧顺序/覆盖、有限非零长折线，并记录 artifact/records SHA-256。集成用例证明错帧在输出前拒绝，合法轨迹会真正替代原点查询并输出 corridor obstacle；用例 PASS。进入 6C 离线/ROS 边界收据。
- 6C RED 已确认：formal streaming 仍发布 v4，且 metadata 尚未表达“仅离线文件回放”。新断言要求 v5、`airborne_formal=false`、`avoidance_formal=false`、ROS1 service/action/PX4 未实现以及 rolling 最终坐标范围收据。
- 6C GREEN 针对回归通过：formal 升为 v5、经验升为 v3；`map_run.json` 明确 `execution_mode=offline_file_replay`、两个 formal 部署声明均为 false，并写出 ROS1 node/publisher/service/action/PX4/online-latency 全部未实现的可机读边界。rolling 的 body-relative 窗口、最终 local bounds、重定位次数和最后 shift 也已收据化。formal 2 项和经验 6/6 PASS。
- 阶段 6 首轮完整回归：概率地图 54/54 PASS，新旧模块 `py_compile` 和 `git diff --check` 通过。未运行训练、GPU 推理或全量地图回放。收尾审查发现采样球半径等于走廊半径时，理论上仍可在相邻采样点中点留下窄缝；需改为基于采样间距的保守膨胀查询半径。
- 轨迹走廊几何缝隙已修复：实际查询半径为 `corridor_radius + effective_spacing/2`，并写入每帧查询结果；实现是保守过近似，不会将该窄带误判为 clear。核心/集成 2 项短测 PASS。
- 阶段 6 最终回归完成：概率地图 54/54、经验位姿 6/6 PASS，新增/修改模块 `py_compile` 和 `git diff --check` 通过。README、test README 与三份 TODO 已同步，当前转入阶段 7。
- 阶段 7 初始检索确认审查意见的两个主切入点：Radar 聚合在 `NTU4DRadLM_pre_processing.py` 约 450--515 行，当前虽统计 finite Doppler 计数，但需核对非有限值是否仍进入累加；manifest 已有 training v2/deployment v3 schema，但 formal data protocol 仍是 v1 命名与门禁。后续将窄读真实调用链后先写 RED。
- 调用链窄读确认 P1-01 实际存在：`features` 和 `doppler` 的 NaN/Inf 仍直接进入 `np.add.at`，Doppler 均值/二阶矩错用总 `unique_counts` 作分母；之后 `save_sparse_radar_voxel` 才因非有限 voxel 拒绝，不能保留可审计的输入丢弃计数。P0-03/P1-02 仍缺 Doppler 符号与反射字段单位的权威身份，不应在 v2 上默默启用补偿。
- 阶段 7 拆分为 7A--7D：7A 修复分字段 finite 聚合并升级 statistics；7B 把 raw pointcloud 字段/单位/符号绑定为显式 schema，无权威值时 fail-closed；7C 为 rosbag 解包单帧失败生成收据和关键模态门禁；7D 发布独立 formal v3 链与训练门禁。保留旧 v1/v2 读取兼容，不就地伪造升级旧数据。
- 7A RED 已确认：同体素混合有限样本与 intensity NaN / Doppler Inf 时，当前 voxel 含非有限值并触发 variance RuntimeWarning。反例锁定预期为 point/intensity/Doppler 计数 3/2/2，ch1 均值 3、ch2 均值 4、ch3 方差 1。
- 7A 首轮实现已将统计合同升级为逐字段 finite-count v2，并保留 v1 NPZ 读取兼容；数据加载器会核对 scene policy 与每帧 payload 的协议身份。
- 测试夹具审查发现 strict-missing 与 v2-roundtrip 两处 policy 常量被错位替换，已按真实语义纠正：缺统计兼容测试保持 v1，新增统计写入测试声明 v2。
- 7A 单元回归 6/6 PASS，finite intensity/Doppler 不再互相污染，统计 NPZ v2 roundtrip 与篡改拒绝均通过。
- 扩展 checkpoint 回归暴露兼容边界：`assert_formal_dataset_preflight()` 仍把“当前新统计常量”当作所有 formal-v2 数据唯一协议，因而会拒绝服务器既有 v1 数据。7A 应接受受支持的 v1/v2；只有后续独立 formal-v3 门禁才强制 v2，不能提前破坏现有训练链。
- 兼容修复后 checkpoint 26/26、Radar statistics 6/6、shell/Python 静态检查与 `git diff --check` 通过；launcher 逐帧核对 payload/policy，并在 v2 时汇总 intensity finite count。
- 7A 收尾代码审查继续追踪“有限输入是否保证有限输出”，准备增加极大有限 float32 的溢出反例，不把首轮 GREEN 当成最终结论。
- 溢出反例先 RED：两个有限 float32 max 在 intensity/Doppler 求和和 Doppler 平方处产生 RuntimeWarning/Inf；改用 float64 中间累加后转绿。
- 7A 最终回归：Radar statistics 7/7、checkpoint 26/26、Python/shell 静态检查和 diff check 通过；三份 TODO 已记录监督/体素/指标影响。进入 7B 字段语义 schema。
- 7B 已完成首轮只读调用链：PointCloud2/PointCloud v1 的别名压平会丢失字段物理身份，预处理随后按固定列位置消费；准备把“存储列布局”和“经证据确认的物理语义”拆成严格 schema。
- 已复核历史 P1-03 设计与现有 raw 目录：旧 sidecar 只证明列位置，不证明物理量。7B 将保留该布局收据但显式标记 unverified，并以独立 field semantics artifact 承担正式门禁。
- 7B 首轮 RED 测试已写入既有 pointcloud schema 测试入口；首次运行使用 `PYTHONNOUSERSITE=1` 导致 ROS Python 路径不可见、在 setUpClass 提前失败，尚未到达新增反例。该调用方式不适用于依赖 ROS 的解包测试，需改用项目约定的 Conda 命令复现真实 RED。
- 当前 Conda 环境同样缺 ROS Python 包；为保证单元测试隔离，测试入口仅在 ROS 不可导入时注入最小模块替身。真实 RED 已命中：缺字段仍写 0、schema 无 physical status、补偿函数不接收正方向、严格 field-schema 模块不存在；其余既有 CSV/bag-open 用例通过。
- 7B 第一轮 GREEN：解包缺字段改为 NaN，layout sidecar 明确 `unverified_layout_only`；新增严格 field semantics 模块，Doppler 补偿按 toward/away 选择减/加 ego radial。字段 schema 7/7 PASS。
- 误启动的 airborne 整套回归在完成前 6 项后超过短测预算，未留下运行进程；不宣称整套通过，后续只跑两项 Doppler 相关用例。
- field semantics loader 已增加权威 evidence 文件内容校验；下一步接入预处理入口，在输出目录创建前完成 schema/速度模式门禁，并把 schema 身份传入 worker/policy/manifest。
- 预处理已接入 schema resolver、worker Doppler 正方向、policy/manifest provenance 和 CLI；字段协议 8/8 PASS。运动协议唯一失败来自用户保留的 `preprocess.sh` 已改为显式 `--velocity_mode none`，旧测试却只接受环境默认语法；不修改用户脚本，调整测试接受两种等价 fail-safe 表达。
- 运动协议调整后 9/9 PASS；随后两项 airborne 定点回归误用 `test.unit...` 模块路径，但 `test/` 不是 Python package，导入失败且未执行用例。改用直接脚本加 testcase 名称重跑。
- 正确的 airborne 两项与 Radar statistics 7/7 已通过。7B 收尾增加解包 layout/物理 semantics 交叉绑定和 PointCloud v1 sidecar，准备重跑字段/运动/统计短测后记录完成。
- 7B 最终回归通过：field schema 8/8、motion 9/9、Radar statistics 7/7、airborne Doppler 2/2、Python 编译和 diff check。三份 TODO 已记录无权威材料时只能继续 unverified/velocity-none 的边界；进入 7C 解包失败收据。
- 7C 核心已实现：共享 receipt 模块、点云/图像保存异常向上传播、bag-open/逐消息失败原子收据、关键模态即时门禁、成功计数与场景最终覆盖门禁。下一步补伪 bag 回归并做异常路径代码审查。
- 7C 首轮回归 10/10 PASS，覆盖 bag-open、Radar 单帧写盘失败和 complete receipt。收尾审查继续补消息循环外 bag 异常、close 异常和跨帧 layout 漂移，之后再扩大到时间戳/字段短回归。
- 收尾后字段/receipt 11/11 PASS；时间戳扩展 7/8，唯一错误是旧测试仍要求 `preprocess-v2.sh` 自己重新解包，而当前既定入口已分为“`preprocess.sh` 完整重建”和“`preprocess-v2.sh` 复用已解包 Raw”。仅更新测试指向真实完整重建入口，不修改用户脚本。
- 时间戳测试改指真实完整重建入口后 8/8 PASS，pointcloud/receipt 11/11 再次通过，静态检查通过。7C 已写入三份 TODO；进入 7D 独立 formal-v3 协议、入口与训练门禁。
- 7D 已完成旧链只读审计并冻结兼容边界：不升级/覆盖当前服务器 formal-v2；先扩展 data protocol validator/builder，再接 complete receipt 到 preprocess policy/manifest，最后新增 fresh v3 shell 与训练 preflight 门禁。
- 7D builder RED 已确认：新增 v3 成功/legacy-policy 拒绝两项均因现有 API 不接受 `protocol_version` 而失败，准确命中尚未实现的独立协议分支。
- v3 checkpoint-data validator 与 protocol builder 首轮 GREEN：两项针对测试均通过。v3 精确绑定 statistics、field schema/layout、extraction receipt 和共享物理合同；v2 常量/默认保持兼容。
- 代码审查确认 training manifest v2 provenance 不允许新增键；将撤回预处理器此前附加的 schema/layout provenance，只通过已被 manifest content hash 绑定的 preprocess policy 携带这些身份。
- 已撤回额外 manifest provenance，complete extraction receipt 现由 preprocess policy/content hash 绑定；builder CLI 增加显式 v2/v3 选择。训练 Python preflight 与 shell 全帧预检在 data=v3 时强制 statistics v2，data=v2 继续接受旧 v1。
- 继续任务后的首轮复核确认：formal-data builder 默认仍为 v2，v3 仅由显式 `protocol_version=v3` 启用；preprocess policy 携带 schema/receipt 身份，但 training manifest 的既有 provenance 精确集合未被扩展，服务器 formal-v2 兼容边界保持不变。
- 7D 扩展短回归通过：temporal/formal-data 7/7、checkpoint chain 14/14、VAE/checkpoint preflight 26/26、preprocessing motion 9/9。现有 v2 fixture 和旧 statistics-v1 训练预检继续通过，未发现兼容性回退。
- 已新增隔离的 `preprocess-v3.sh`：动态项目根目录、全新 Raw/Pre/Deploy/normalization 名称、输出不可覆盖，并在解包前验证带 evidence 的 verified Radar field schema；预处理强制 complete extraction receipt，protocol builder 显式选择 v3。新增训练 preflight 反例锁定 v3 只接受逐字段 statistics-v2。
- 7D 新增回归通过：formal-v3 shell 缺 schema 在任何解包前 fail-closed，字段/schema/receipt 12/12、VAE/checkpoint 27/27、shell/Python 静态检查和 `git diff --check` 全部通过；没有创建正式数据或启动训练。
- 7D 已完成文档/三份 TODO 收据与兼容文案收尾，阶段 7 完成；进入阶段 8，先回到审查原文逐项重建真实调用链，再做最小工程清理。
- 8A 已写入单帧 Dataset 合同：非 1 sequence、非空 transform、非默认 alignment 参数明确拒绝；内部滑窗列表改为同名单帧 Radar 路径，删除未消费状态。默认 sequence=1 的 frame selection、target、observed mask 和模型输入 shape 保持不变，准备运行 Dataset 聚焦回归。
- 8A Dataset 聚焦回归 16/16 PASS，模块编译和 diff check 通过。8B 调用链确认当前 formal forward 支持完整审计 kwargs，legacy fixture 只支持 `noised_latent`；将用签名能力缓存选择 kwargs，让 forward 内部 TypeError 原样传播，同时修复 Karras history 的可变默认值。
- 8B 多模态推理 42/42 PASS：Karras 默认 history 不再共享，多模态 forward 按签名缓存选择可选 kwargs，内部 TypeError 只调用一次并原样传播。8C 进一步确认 default YAML 的旧 `training_mode/start_scales/distill_steps/loss_norm` 未被当前 CD trainer 消费，而实际 num_scales=40、EMA=0.999、sigma 范围均硬编码；准备把真实生效值显式配置并写入 checkpoint。
- 8C 续审发现训练/推理契约仍未闭合：训练 checkpoint 已开始写入真实 consistency 配置，但 `RadarGenerator` 仍固定使用 sigma 80/0.002、rho 7，正式 checkpoint chain 也未校验这份收据；下一步抽取共享协议并同步接入训练、resume、chain preflight 与 CD inference。
- `test/mini-test/train_minimal.sh` 仍生成旧 `teacher_model_path/training_mode/start_scales/distill_steps/loss_norm` 假配置；这些键不进入当前 `ConsistencyDistillationTrainer`，将改为 initialization checkpoint 与真实 `num_scales/ema_rate/sigma/rho`。
- 8C 核心转绿：新增共享 consistency receipt，训练、resume、formal checkpoint chain 和 CD inference 使用同一 sigma/rho 配置；mini 配置移除无效蒸馏键。CD entrypoint、checkpoint 14/14、multimodal inference 42/42 及静态语法检查通过。
- 8D 调用链确认正式 saved evaluator 虽消费 observed artifact，但尚未把 mask 应用于指标数组；将先补 domain 反例，再统一正式 summary 协议和 diagnostic-only 入口标签。
- 8D 首轮实现已将 pred/Radar/target occupancy 与 uncertainty calibration 限定到逐帧 authoritative observed mask，并在 summary 分离 `formal_metrics` 与未裁剪 raw LiDAR 辅助诊断。既有 formal inference 13/13、task metrics 4/4 PASS；仍需增加 mask 外反例锁定行为。
- 8D 核心转绿：mask 外正预测反例、正式 summary 协议和 diagnostic-only 入口标签已完成；formal evaluator 15/15、task metrics 4/4、YAML 5/5、mini launcher 21/21、checkpoint 15/15、multimodal inference 44/44 PASS。进入阶段 8 最终跨模块短回归。
- 阶段 8 最终跨模块短回归全部通过：Dataset 16/16、CD 两组接口、checkpoint 15/15、multimodal inference 44/44、formal evaluator 15/15、task metrics 4/4、YAML 5/5、mini launcher 21/21，相关 Python/shell 静态检查与 diff check 通过。八阶段顺序修复计划完成。
