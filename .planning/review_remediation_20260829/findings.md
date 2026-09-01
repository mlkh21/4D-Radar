<!-- 文件功能：记录顺序修复过程中的调用链证据、设计决定和兼容性发现。 -->
# 顺序修复发现

## 初始状态

- 当前分支 `withir`，HEAD `3e21c3a`。
- 用户已有 `NTU4DRadLM_pre_processing/preprocess.sh` 修改，内容为服务器专用全量预处理流程，本任务不触碰。
- 审查核验计划确认：VAE 可继续；正式 LDM/CD 应在阶段 2、3 完成后启动。
- 当前 YAML 默认 `cuda_devices: "0,1"`，兼容字段仍为 `num_gpus: 4`；正式 launcher 会按设备列表覆盖实际 GPU 数，但直接消费者和测试存在歧义。
- formal VAE 使用 `bce_dice`；`occupied_weight`、`empty_weight`、`channel_weights`、`false_positive_weight`、`occupancy_mass_weight` 仅被 `legacy_mse` 路径消费。

## 设计原则

- 阶段 1 将 GPU 数量收敛为由 `cuda_devices` 唯一派生，保留读取兼容时也不得出现矛盾默认值。
- legacy-only VAE 参数不能继续伪装成当前 BCE+Dice 可调参数；优先以嵌套 `legacy_mse` 配置或显式命名隔离，并保持旧 checkpoint 加载兼容。

## 阶段 1 调用链

- fresh VAE 由 `create_vae_config(config_type)` 建立 preset，再由 `apply_vae_config_overrides()` 消费 YAML；resume 则严格使用 checkpoint 内嵌的完整 `vae_config`。
- 当前 preset 为新 checkpoint 保留 legacy MSE 构造参数，但 `occupancy_loss_type` 固定为 `bce_dice`；因此从 formal YAML 删除五个 legacy-only 覆盖项不会改变当前训练数值或 checkpoint 结构。
- 为让 formal YAML 显示真正生效的损失旋钮，应显式写入 `occupancy_loss_type`、BCE、Dice、positive cap 和 continuous reconstruction 权重。
- `hardware.num_gpus` 只在 YAML 中作为矛盾兼容字段存在；正式 launcher 已根据 `cuda_devices` 长度写入运行时 override。阶段 1 决定从默认 YAML 删除该字段，运行时生成配置仍保留派生后的 `num_gpus` 供下游审计。

## 阶段 1 结果

- 默认 YAML 现在只声明 `cuda_devices: "0,1"`；launcher 仍在运行时写入派生后的 `num_gpus/world_size/effective_global_batch_size`，服务器临时 `CUDA_DEVICES` 覆盖优先级不变。
- formal VAE YAML 已显式声明 BCE+Dice、positive-weight cap 和连续通道重建权重；五个 legacy-MSE-only 字段从 formal YAML 删除，但 VAE preset/checkpoint 构造参数未删除，旧 checkpoint 兼容性保持。
- 该修改不改变 target、observed mask、体素数量或当前 BCE+Dice 数值；只消除无效调参入口和 GPU 默认冲突。
- 65 项短回归全部通过：YAML/覆盖 5、VAE loss 20、VAE checkpoint 26、DDP 协议 14。

## 阶段 2 初始调用链

- `train_epoch()` 已把 batch metadata 移到 device，但当前只把 IR/标定字段传给模型，没有提取 `occupancy_observed_mask` 传入 `compute_ldm_loss_components()`。
- `compute_ldm_loss_components()` 的主损失直接对全部 latent 元素做 MSE，decoded occupancy、density、column、IR-frustum 和 uncertainty 组件也没有统一 observed-domain 参数。
- `validate()` 同样对全部 latent 元素累计平方误差，并让 `micro_occupancy_metrics()` 在整个体素网格统计 IoU；因此 checkpoint 选择会把 unknown 体素当作 free/negative。
- 垂直结构损失已有 `column_mask` 入口但调用方未传；它只监督 target 非空列，因此需要将 observed column 与 target-positive column 联合约束，而不能仅把 mask 粗暴乘在最终标量上。

## 阶段 2 设计决定

- 新增单一 observed-mask 解析器，接受 `[B,1,Z,X,Y]`（兼容 `[B,Z,X,Y]`），验证 batch/shape/finite/0-1，并把 target positive 合入可观测域以防数据漏标正样本。
- voxel decoded loss 只在 observed voxel 求均值；mass 使用 observed-domain 归一化，false-positive 只把“observed 且 target-free”作为负样本。
- 垂直结构不能只筛列：预测/target 的高度分布、overshoot 和 continuity 都必须排除列内未观测 Z；列级 existence 则对每列 observed logits 做按有效体素数归一化的 log-sum-exp，并完全排除未观测列。
- latent 主 MSE 和 uncertainty NLL 使用 `adaptive_max_pool3d` 将 voxel observed mask 映射为“对应块内任一体素可观测”的 latent mask；协议需写入 LDM checkpoint，避免旧目标函数被静默恢复。
- legacy batch 没有 mask 时保留全域旧行为；formal dataset 已要求 persisted mask，显式传入全零且无 target positive 的 batch 应 fail-closed，而不是生成零损失 checkpoint。

## 阶段 2 loss 层结果

- 统一 mask 解析、体素 masked mean、latent adaptive-any mask 已接入 `micro_occupancy_metrics()`、decoded occupancy、density、column、vertical、IR-frustum 和 uncertainty NLL。
- 垂直 continuity 现在只在相邻两个高度体素都 observed 时比较 transition；overshoot 也只惩罚 target top 以上的 observed 体素。
- 6 项未知区扰动反例已转绿：改变未观测体素/列/高度/latent 块不再改变对应 loss 或 IoU；显式空 observed domain 会拒绝计算。
- 下一步仍需把 dataset metadata 真正传入 train/validation，并把监督协议写入 checkpoint/resume 身份；当前仅函数层正确。
## 2026-08-29 阶段 2 接口复核

- `OptimizedLDMTrainer.train_epoch()` 已从 batch metadata 读取 `occupancy_observed_mask` 并传给统一损失入口；formal 配置缺失时 fail closed。
- `OptimizedLDMTrainer.validate()` 已在 latent 误差和 decoded occupancy IoU 中使用同一 observed domain，并在显式要求 persisted mask 时拒绝缺失输入。
- LDM checkpoint 的 `ldm_loss_config` 与顶层字段均记录 observation-supervision protocol；formal resume 会拒绝缺失或不匹配协议。
- 仍需补 trainer 级回归，防止未来调用处漏传 mask；同时保持 `decoded_vertical_structure_losses` 原有位置参数兼容。

## 2026-08-29 阶段 3 初始审计

- 审查意见 P1-04 属实：IR extractor 的 torchvision ResNet18 与 fallback 都包含 `BatchNorm2d`。
- 正式 DDP 包装当前设置 `broadcast_buffers=False`；普通 BatchNorm 的 running mean/variance 会按各 rank 本地 IR batch 更新，rank0 checkpoint 不代表全局统计。
- 仅把 `broadcast_buffers` 改为 true 仍不足：它只在 forward 前广播，forward 后各 rank 会再次用本地 batch 更新。多卡正式训练应在 DDP 包装前把 BatchNorm 转为 `SyncBatchNorm`。
- CD `_update_ema()` 只 zip parameters，确实不更新 floating running statistics 和 integer counters；EMA checkpoint 因而混合了已更新参数与初始化 buffer。
- EMA buffer 需要按名字严格匹配：floating buffer 使用同一 EMA 率，非 floating buffer（如 `num_batches_tracked`）直接复制；不能依赖无名字的 zip 静默截断。
- LDM 模型在 optimizer 创建前调用共享 `wrap_model_for_ddp()`；将 SyncBatchNorm 转换集中到该 helper 可覆盖正式多卡 LDM，并保持单卡 checkpoint 结构不变。
- CD 当前先创建 EMA 副本、再包装 online 模型；若仅在 wrapper 内转换，online/EMA 模块类型会不一致。CD 必须在 EMA 深拷贝之前准备 normalization，之后再包装而不重复转换。

## 2026-08-29 阶段 4 初始审计

- P1-05 属实：`RadarGenerator._load_model()` 无条件优先 `model_state_dict`，没有读取 CD 的 EMA 权重或权重选择声明。
- CD 正式入口目前只构造 train partition 和 train loader，没有独立 validation loader，因此现阶段不能诚实声称 online/EMA 已由验证集选择。
- P1-06 属实：CLI 默认 threshold=0.1，formal launcher 另传环境阈值；阈值未作为绑定 checkpoint hash 的独立 validation artifact。
- P1-07 部分已修复：inference 已生成逐帧 observed-mask SHA-256 合同并写入 `inference_run.json`；但 saved evaluator 的严格目录发现器仍不接受 `*_observed_mask.npy`，闭环仍断开。
- P1-09 属实：formal launcher 未传 seed，CLI `seed=-1` 会关闭固定随机性；逐帧 `perf_counter()` 前后没有 CUDA synchronize，异步 kernel 会让报告偏低。
- observed-mask protocol/digest 逻辑当前在 inference 与 streaming-map 两处重复；evaluator 若再从 inference 私有函数导入会形成第三种隐式耦合。应抽取无 Torch 依赖的共享 artifact-protocol 模块，并由三条链共同消费。
- unified CD 入口已经构造独立 `val_loader`，但没有传给 CD trainer；standalone CD 只构造 train partition。4C 必须同时修两个入口，否则服务器使用不同入口会得到不同 checkpoint 合同。
- CD 可复用固定噪声的单步 denoising proxy 比较 online/EMA：在同一 validation sample/seed/sigma 上计算 observed-latent MSE 与 decoded observed-IoU，按 IoU 优先、latent loss 次优选择 deployment state；不能继续按 train consistency loss 选部署权重。
- 4D 审计确认 formal launcher 仍默认注入 `OCC_THRESHOLD=0.05`，CLI 又默认 0.1；二者均未绑定模型 checkpoint 或 validation split，saved evaluator 还允许 CLI 覆盖正式运行阈值。
- threshold artifact 应在 checkpoint 保存后由训练期 validation threshold sweep 记录构建，从而同时绑定 checkpoint 文件 SHA、validation 数据身份和 CD 实际部署权重源；正式 inference/evaluator 不再接受自由阈值覆盖。
- 阈值选择固定为 observed-domain micro IoU 最大，IoU 并列时 recall 最大，再并列时取较低阈值，以避免通过 deployment/test target 调参。

## 2026-08-29 阶段 5 初始审计

- 已沿 `inference.py -> prediction artifact -> streaming_map_update.py -> SlidingProbabilisticGridMap.update_from_voxel()` 收窄地图调用链；宽范围搜索输出发生截断，后续改用函数级窄读，避免遗漏真实接口。
- 已有审查反例确认 `_observed_layer_mask()` 将显式 observed mask 与 `prediction_occ > 0` 取并集；正式推理 sigmoid 占用概率通常大于 0，因此预测本身会扩张 observed 域，破坏 unknown/free 权威边界。
- 已有审查证据确认 target 第 4 通道是 Doppler-valid mask，而地图曾把输入第 4 通道当 Doppler variance 并加入 DEM 高度方差；这既是通道合同错配，也是速度方差/无量纲量与高度方差 m^2 相加的量纲错误。
- 阶段 5 必须把“生成预测通道”“原始 Radar 统计通道”“observed 权威 mask”“DEM 高度证据”拆成显式接口，不能继续依赖 `voxel[..., 0/3]` 的位置猜测。
- 当前 prediction metadata 仅绑定 `coordinate_frame=lidar`、`layout=czxy`、shape、dtype 和文件哈希，没有声明通道名称、数值域或单位；地图无法从合同证明 `[...,0]` 是 occupancy probability，更不能证明 `[...,3]` 是 Radar Doppler variance。
- `_observed_layer_mask()` 与旧 BEV helper 都会把显式 mask 与 occupancy 正值取并集；formal 路径需要新增显式 authoritative 语义，legacy 无 mask fallback 可保留。
- `_update_dem_from_voxel()` 已可仅通过 occupancy 沿 Z 的概率矩计算高度均值与几何高度方差，随后额外叠加 `_doppler_variance_bev(voxel[...,3])`；删除该跨量纲项不会减少 occupied/observed 体素，只会让 DEM variance 回到 m^2 加模型高度不确定性的合同。
- formal streaming 的 `radar_voxel_dir` 实际指向 inference 生成的 prediction voxel；它以 `sensor="radar"` 更新地图，但输入并非原始 Radar 统计体素。因此 `observation_reliability_map()` 从 prediction 第 4 通道读取 Radar 方差属于跨 artifact 隐式耦合。
- `update_from_voxel()` 使用 observed mask 生成 warp/mapping mask 与 reliability mask，但传给 DEM 的 `warped_voxel` 仍保留所有预测概率；DEM 更新也应显式接收 warped observed mask，否则 unknown 区预测可能继续进入高度统计。
- formal streaming 已在 `load_formal_inference_contract()` 前置加载 run metadata，适合在创建输出目录前校验 prediction 通道 schema、occupancy probability 数值域和 authoritative observed 语义；legacy `auto` layout 路径可保持兼容但不得标为 formal。
- 生成链已证明 `RadarGenerator._apply_vae_occupancy_activation()` 只对 channel 0 做 sigmoid；其余三个 decoder 输出保持连续原值。target 真实语义为 ch0 LiDAR occupancy、ch1 LiDAR intensity、ch2 邻域 Radar Doppler、ch3 Doppler-valid mask，因此 prediction ch3 不是 Doppler variance。
- inference 保存的 prediction ndarray 已是 ch0 概率、ch1--3 连续重建；artifact v2 应声明地图唯一可消费字段为 ch0 occupancy probability，并将 ch1--3 标为 auxiliary/non-mapping，而不是给它们伪造可靠物理单位。
- formal contract 当前只检查 prediction 数组 finite/shape/dtype，不检查 ch0 是否在 [0,1]；应在输出目录创建前拒绝超域概率。

## 2026-08-29 阶段 5 设计决定

- prediction artifact 升级为 v2，固定 4 通道身份，并内嵌 `generated_occupancy_mapping_input_v1`：地图只消费 ch0 `[0,1]` occupancy probability，ch1--3 明确为非地图辅助输出，observed 域来自外部权威 mask。
- 地图 API 新增显式 `evidence_semantics` 与 `observed_mask_authoritative`。formal/经验 inference mapping 必须传 generated-prediction 语义与 authoritative mask；legacy 缺 mask 路径保留原 occupied-only fallback。
- generated-prediction 语义禁止从 ch3 推导 Radar variance；原始/legacy voxel 的 ch3 可靠度降权暂保留兼容，但必须由显式 legacy semantics 才能触发。
- DEM 只从 authoritative observed 层内的 occupancy Z 分布计算 `mean_m` 与 `variance_m2`；generic model uncertainty 仅用于可靠度，不再直接加到 m^2。未来若有校准后的高度方差，需通过独立 `height_variance_m2` 接口和 artifact 单位合同接入。
- 收尾审查发现旧 `map_run.protocol` 无法区分新旧 observed/DEM 语义；正式地图升级为 `pose_aware_layered_map_v4`，经验地图升级为 `...offline_empirical_v2`，legacy 非严格入口暂保留 v3。
- `map_run.json` 现显式记录 prediction mapping contract、authoritative observed 与 DEM mean/variance 单位；避免下游仅凭数组名猜测 `dem_var` 的物理量。

## 2026-08-29 阶段 6 初始审计

- `SlidingProbabilisticGridMap` 没有 recenter/shift 方法；`history` 只在二维融合后保存若干完整快照，既不参与融合也不移动地图，因此类名中的 sliding 目前只是时间窗口命名，不是空间 rolling map。
- local map 的 `cfg.x_min/...` 同时被当作 source evidence 体素坐标起点和 destination map 边界；一旦移动地图原点，source body/LiDAR voxel 坐标会一起错误移动。6A 必须先拆分 evidence range 与 rolling map bounds。
- streaming 每帧只对当前 body/LiDAR 原点调用一次 `query_proximity()`；没有航迹点输入、走廊半径、分段采样、首次风险点或覆盖收据，因此不能代表轨迹走廊安全查询。
- 默认 50 m/s、0.5 s 反应、8 m/s^2 制动、5 m 余量对应 186.25 m 安全距离，而默认 query radius 30 m、evidence 80 m；当前查询会正确返回 `search_radius_below_safety_distance`，但这也证明现配置不能给出 clear 结论。
- ROS 代码只有文件末尾 TODO；没有 `rospy`、publisher/service/action 类型、节点入口或消息 schema。现阶段必须继续标为 offline-only，不能把新增纯 Python 接口描述成 ROS 已实现。

## 2026-08-29 阶段 6 设计拆分

- 6A 采用 `body_anchored_integer_voxel_roll_v1`：严格地图把 `map_pc_range` 解释为相对当前 body/LiDAR 原点的窗口偏移，按整数体素移动 local bounds；旧单元搬移到新索引，新暴露区域重置为 occupied=0.5/belief=0/plausibility=1/unknown=1/DEM=NaN。
- source voxel 中心必须始终由独立 `evidence_pc_range` 与 source shape 计算；destination 索引才使用当前 rolling local bounds，避免 body 位移被重复加到 evidence 坐标。
- 默认 legacy 地图不启用 rolling，保持既有直接 API 行为；formal 与 offline empirical streaming 显式启用并在 snapshot/map_run 中记录当前 local bounds、累计 recenter 次数与最后 shift。
- 6B 将轨迹走廊的“沿程停止距离”和“横向/竖向 corridor radius”拆开：轨迹弧长必须覆盖停止距离，沿路径按固定间距采样，每个样点在 corridor 半径内做三态查询并返回首个风险点。
- 6C 不实现虚假的 ROS shim；先把 `avoidance_formal` 固定为 false，并发布 transport-neutral 的输入/输出 schema 与 `ros_node/service/action_implemented=false` 收据，待真实 ROS1 包和消息类型存在后再升协议。

## 2026-08-30 阶段 6 结论

- rolling 、trajectory corridor 和离线/ROS 边界均已完成可执行代码与机读收据；formal/empirical 地图协议分别升为 v5/v3。
- 轨迹查询使用保守膨胀采样球，unknown 和制动视界不足均 fail-closed。地图仍仅是离线文件回放，无 ROS1/PX4 执行链。
- 阶段 7 应继续保持不覆盖用户现有 `preprocess.sh`，将新协议改动放在 formal v3 独立链与小型测试中。

## 2026-08-30 阶段 7A 统计协议决定

- Radar occupied point 仅由有限 XYZ 决定；intensity 与 Doppler 各自使用独立 finite mask、计数与分母，避免某字段缺失污染另一字段。
- `radar_point_count_field_validity_v2` 新增 `intensity_valid_count`，并继续保存 `point_count`/`doppler_valid_count`；v1 数据仍可严格读取，但新体素化结果只发布 v2。
- scene `preprocess_policy.json` 声明的统计协议必须与每帧 NPZ payload 完全一致，禁止 policy 声明 v2、实际混入 v1 的隐式数据集。
- 首轮代码审查发现仅过滤 NaN/Inf 仍不足：多个极大但有限的 float32 Doppler 在平方或累加时仍可能溢出为 Inf。累加与二阶矩应使用 float64，最终才写回 float32/受限方差。

## 2026-08-30 阶段 7B 原始字段调用链

- `unpack_rosbag._read_pointcloud2_fixed_columns()` 会把 `intensity/reflectivity/power/rcs/snr` 任一别名统一写到第 4 列，把 `velocity/doppler/v_r/radial_velocity` 任一别名写到第 5 列；schema v1 只记录选中了哪个字段，不记录物理语义、单位、坐标系或 Doppler 正方向。
- PointCloud v1 分支使用子串匹配且不写 schema；缺字段时用 0 填充，预处理会把这些 0 错记为 finite 有效测量。Livox 与 Radar 共用保存函数，也没有 sensor role 参数约束列语义。
- voxelizer 固定把第 4/5 列解释为 intensity/Doppler；`compensate_radar_doppler()` 固定执行 `raw - ego_radial`。在不知道源 Doppler 正方向时，这个符号没有代码证据支持，正式新链必须在补偿前验证权威 schema，不能凭别名猜测。
- 本机现有 `garden/loop3` raw Radar 目录均没有 `pointcloud_schema.json`，符合其来自旧 PointCloud v1 解包链的事实；不能从列值或文件名反向伪造 source field/单位/正方向。
- 已有 2026-07-20 schema v1 修复只解决“缺列后错位”，并明确用 0 填充缺字段。7A 的 finite-count 协议使该 0 看起来仍是有效测量，因此 7B 还需把缺字段编码改为 NaN，才能让 finite count 正确表达缺失。
- 设计边界：解包自动生成的 sidecar 只能声明布局与 `unverified physical semantics`；另建严格 Radar field semantics artifact，只有带可校验权威来源的 `verified` artifact 才允许 formal-v3 或 ego-motion Doppler 补偿。现有 formal-v2/velocity none 保持可运行。
- 代码审查进一步收紧“verified”含义：schema 不只填写一个声明性哈希，还必须引用同目录内的安全相对 evidence 文件；loader 会拒绝符号链接/越界路径并重算 SHA-256。没有权威材料时只能保存 unverified schema，不能靠手写 `status=verified` 绕过。
- 仅验证 semantics artifact 仍可能与实际解包列脱节，因此新增 layout/semantics 交叉核对：实际 `field_mapping/selected_fields/source_fields` 必须与 verified artifact 的两个 source field 完全一致，缺字段或 0 缺失值编码均拒绝。
- PointCloud v1 分支现在也写同一 layout sidecar，并改用精确别名选择；这样 formal-v3 不会因消息类型分支而失去字段来源证据。

## 2026-08-30 阶段 7C 解包失败收据设计

- 旧 `save_pointcloud()`/`save_compressed_image()` 在内部捕获所有异常后 `pass`，上层仍继续并最终打印成功；未知点云类型、空点云、图像解码/写盘失败都会造成静默缺帧。
- 新 `rosbag_extraction_receipt_v1` 按场景记录 expected/processed bag、有落盘依据的逐 topic 成功数、逐条失败的 bag/topic/timestamp/source/error/critical，以及 Radar/LiDAR/IR 派生状态。
- critical 定义固定为 `radar_pcl`、`livox/lidar`、thermal compressed image；任一消息保存失败立即原子写 failed receipt 并终止。全部 bag 遍历后仍要求三项至少各成功一帧，避免“没有触发异常但 topic 缺失”的假成功。
- 首轮代码审查发现消息循环外的 `get_type_and_topic_info()`/`read_messages()` 异常与 `bag.close()` 异常仍可能绕过逐消息 handler；这些也应作为 `__bag__` critical failure 入账。
- 同一 scene 多帧会反复写 layout sidecar；若消息字段在分卷/帧之间漂移，不能让最后一帧静默覆盖前一合同。sidecar 写入需比较除 shape 外的稳定字段，漂移时交给关键失败收据终止。

## 2026-08-30 阶段 7D formal-v3 调用链

- 当前 `formal_data_v2` 只绑定 manifest/split/target/observed/calibration/Radar-IR sync；checkpoint validator 也只接受这一协议，完全不知道 Radar statistics v2、field semantics、layout 或 extraction receipt。
- 现有服务器训练依赖 `formal_chain_v2 + formal_data_v2`。为了不让已生成服务器数据突然失效，checkpoint 链协议保持 v2，但 data protocol validator 增加独立 `formal_data_v3` 分支；checkpoint 中完整 data identity 会自然阻止 v2/v3 resume 混用。
- formal-v3 应在既有字段上新增：statistics v2、每场景 field schema/layout/extraction receipt SHA，以及共享的 return/Doppler 物理合同。v3 顶层字段必须精确，避免未验证字段被静默携带。
- 预处理 v3 在输出前加载 complete extraction receipt 和 verified field schema；policy 保留经验证对象/状态，manifest provenance 绑定三个源 artifact。protocol builder 从 policy+manifest 重建，训练无需继续挂载 raw bag 目录。
- 新建独立 `preprocess-v3.sh`，使用 fresh raw/preprocessed/normalization 名称和显式必填 schema，不改写用户现有 `preprocess.sh` 或 v2 数据；缺权威材料时脚本在创建输出前失败，v2 训练仍可继续。

## 2026-08-30 阶段 8 工程清理调用链

- Dataset 的 `sequence_length>1` 确实只扩大滑窗 I/O，`__getitem__()` 最终固定取 `radar_seq_tensors[-1]`；全仓调用都使用默认或显式 1，没有真实时序融合消费者。保留参数兼容但对非 1 fail-fast，并把内部样本合同收敛为单 Radar 路径，能消除“时序建模”误导且不改变正式训练样本。
- Dataset 的 `transform`、`alignment_size`、`self.ir_dir`、`default_K` 没有下游读取。对两个外部参数静默接受会制造假配置；应对非默认/非空值明确拒绝，并删除无用实例状态，但不移除函数参数以保留清晰迁移错误。
- `karras_sample()` 的 `last_sample_result_list=[]` 是真实跨调用共享状态，应改为 `None` 后每次建立空列表；仅显式传入列表时才允许 prior history 跨调用。
- inference 多模态 forward 用覆盖整个调用的 `except TypeError` 回退；模型内部 TypeError 会被误当成旧接口。当前 formal 模型签名可由 `inspect.signature(forward)` 在调用前解析是否支持 `return_uncertainty`/审计 kwargs，不能捕获模型执行异常来猜接口。
- formal saved evaluator 已成为绑定 inference metadata、prediction、observed mask 和 threshold artifact 的严格入口；`scripts/evaluate.py` 仍只是按同名 `.npy` 点云配对的 legacy 工具，应保留文件但在文件头/CLI 明确 diagnostic-only。统一 launcher 同时跑 LDM/CD，并非 legacy；分阶段 launcher也是正式入口，不宜错误降级。
- CD 生产 checkpoint 已明确记录 `ldm_initialized_ema_consistency_v1`、LDM 只用于初始化、target 来自 CD EMA；仍残留 `teacher_model_path`、`consistency_distillation` 和部分变量/提示命名。应保留 YAML key 兼容，但把用户可见说明和内部解析函数改为 initialization checkpoint，禁止宣传为持续冻结教师蒸馏。
- `KarrasDenoiser` 在 LDM/CD 真实训练和推理中仍被调用，不能仅重命名类文件而破坏 checkpoint/API；阶段 8 应新增明确的运行语义 receipt/注释，区分当前直接 x0/EMA consistency 路径与未证明的完整 EDM 预条件训练声明。
- 当前 CD 推理的 `KarrasDenoiser` 在模型 checkpoint 加载完成后仍以固定 `sigma_max=80.0`、`sigma_min=0.002` 构造，`_get_sigmas()` 又固定 `rho=7.0`；因此一旦 YAML 中真实生效值被修改，训练与部署采样将产生隐形接口不匹配。
- `validate_formal_checkpoint_chain()` 目前只检查 CD 的父链 hash、网格、normalization 和多模态权重，不检查 `training_semantics/denoising_parameterization/consistency_training_config`；正式预检尚不能阻止旧命名或采样配置漂移。
- 8D 收尾审查发现 `evaluate_saved_predictions.py` 只校验/记录了 authoritative observed mask，却仍用完整 `pred/target/radar` occupancy 生成点云并计算 Chamfer/BEV/PRF，uncertainty calibration 也统计全体素；旧 summary 因而不能证明 observed-domain 指标。
- `scripts/evaluate.py` 仅按同名 `.npy` 配对任意 XYZ 点云，不验证 formal run metadata、checkpoint、阈值 artifact、observed mask 或数据协议；它应保留为 legacy diagnostic，不能与正式 saved-prediction evaluator 并列描述。
