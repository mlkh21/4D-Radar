<!-- 文件功能：记录新审查意见核验中的证据与判断。 -->
# 审查意见核验发现

- 当前实际 HEAD 为 `3e21c3a`，附件审查对象为 `7b22149`。
- 当前工作树保留用户对 `NTU4DRadLM_pre_processing/preprocess.sh` 的修改，以及既有预处理诊断计划；本次不改动。
- 附件列出 4 个 P0、10 个 P1、6 个 P2 及若干 P3 建议；其中预处理/训练直接相关的核心声称为 decoded loss 未使用 observed mask、formal Doppler 未补偿、Radar 非有限值聚合、CD EMA/BN/DDP buffer、YAML GPU 数量冲突。
- 地图/部署相关声称包括 observed mask 被 `prediction>0` 扩张、prediction ch3 合同错配、固定非滚动地图、阈值不一致、正式推理输出与 saved evaluator 不闭环、计时/seed 和 DEM 量纲问题。
- 报告自述只运行静态/轻量测试，未训练、未做 GPU/NCCL/ROS；其“服务器训练影响”必须按训练入口与当前 formal 配置重新分级，不能由报告的 P0/P1 标签直接决定。
- `probabilistic_mapping.py` 在收到显式 `observed_mask` 后仍与 `prediction_occ > 0` 做并集；对 sigmoid 生成结果会扩大可观测域，破坏 unknown/free 边界。该问题不进入 VAE/LDM/CD 训练损失，但阻断正式概率建图与部署评估。
- target 第 4 通道由预处理器写入 Doppler 有效性掩码，建图却把生成结果第 4 通道解释成 Doppler 方差，并进一步加到高度方差。通道语义和量纲均不匹配；不阻断训练执行，但阻断正式建图。
- VAE 的 BCE+Dice 路径会读取 `occupancy_observed_mask`；LDM 的 decoded occupancy、false-positive、mass、vertical、density、column 及验证指标当前没有完整使用该掩码。默认正式配置中若干相关权重非零，因此这是开始正式 LDM 训练前应解决的问题，而不是 VAE 训练阻断项。
- 正式 v2 预处理固定使用 `velocity_mode=none`。训练可以运行，但只能作为未做自运动补偿的 Doppler 基线；在没有权威坐标系、符号和速度来源时不应猜测补偿。若改变，需要建立新数据协议并重做预处理、normalization 与后续 checkpoint 链。
- Radar 聚合前没有用有限性掩码排除 NaN/Inf；现有冻结数据已通过预检，故通常不影响本轮服务器训练，但未来重新预处理前应修复。
- IR 投影只有视锥过滤，没有遮挡或 z-buffer；这会影响 IR 融合质量与论文主张，但不是训练启动阻断项。
- 多模态 IR 编码器包含 BatchNorm，DDP 使用 `broadcast_buffers=False`，CD EMA 又只更新参数不更新 buffer。VAE 不依赖该 IR 编码器；正式多 GPU LDM/CD 前应修复并做双卡一致性测试。
- 推理默认加载在线 `model_state_dict` 而不是 CD EMA；推理阈值默认 0.05，而 LDM 验证阈值为 0.5；正式推理和评估前必须统一，但不阻断当前训练。
- 推理会写 `*_observed_mask.npy`，严格评估器的辅助文件白名单却不接受该文件，正式评估链当前不闭合。
- YAML 中 `cuda_devices: "0,1"` 与兼容字段 `num_gpus: 4` 表面矛盾，但正式 launcher 会根据设备列表重写实际 GPU 数；经 launcher 启动不是训练阻断项，直接调用 Python 时仍有配置歧义。
- `sequence_length>1` 当前只消费最后一帧；正式默认值为 1，因此本轮训练无影响。
- Karras 采样器可变默认参数、推理宽泛捕获 `TypeError`、rosbag 解包宽泛吞保存错误均属健壮性问题；现有冻结数据训练不受影响，未来预处理/推理前应修复。
- 当前 formal 数据目录没有可追溯到原始 bag 字段的 `pointcloud_schema.json`，因此只能把 ch1 称为通用 intensity，不能据此声称为 RCS；这是科学解释风险，不妨碍张量训练。
- 当前 formal VAE 使用 `bce_dice`，而 YAML 中 `occupied_weight/channel_weights/false_positive_weight/occupancy_mass_weight` 只被 legacy MSE 路径消费，报告 P2-01 成立但不改变本轮训练数值。
- `KarrasDenoiser` 在当前 LDM trainer 中只提供 sigma 范围，正式训练没有调用其 `training_losses`/EDM 预条件；应修正文档命名或另立实验实现，不能在不做消融的情况下改动当前正式训练算法。

## 当前服务器训练门禁

- VAE：可以继续；observed mask 已进入 BCE+Dice 重建损失。YAML GPU 默认值测试失败不影响 launcher 实际派生 GPU 数。
- LDM：正式多卡训练前应先修复 observation-aware loss/validation，并解决 IR BatchNorm 在 DDP 下的 buffer 一致性；否则 checkpoint 选择和各 rank 状态不可信。
- CD：在 LDM 修复基础上，还需同步 EMA buffers，并加入 online/EMA 独立验证和部署权重选择。
- 推理/地图：训练可先不等，但在正式评价、部署或避障结论前必须修复阈值 artifact、评估目录协议、observed 语义和 ch3/DEM 量纲合同。

## 后续实施顺序

1. 训练配置小修：统一 YAML 的设备唯一来源和单元测试，移除或明确 legacy-only VAE 字段。
2. LDM 监督修复：把 persisted observed mask 显式传入训练与验证；decoded loss/结构损失/指标只在观测域统计，并评估主 latent MSE 是否需要下采样 mask 加权。
3. 多卡状态修复：为 IR 编码器确定 SyncBatchNorm、冻结 BN 或无 batch 统计归一化方案；CD EMA 同步参数和 buffers；补两 rank 一致性测试。
4. 推理评估闭环：checkpoint 记录并选择 online/EMA；验证集生成阈值 artifact；saved evaluator 接受并核验 observed-mask；固定 seed 并用 CUDA event 或同步计时。
5. 地图合同修复：formal 显式 observed mask 为权威；prediction 只提供 occupancy；Radar Doppler variance 以独立 schema 输入且只影响经标定的可靠度，不直接加到高度方差。
6. 局部地图接口：实现 body-centered rolling map、轨迹走廊安全查询和 ROS 接口；在 35/50/70 m/s 制动参数下做 fail-closed 验证。
7. 新预处理协议：先修 finite 聚合、字段/单位 schema 和保存失败收据；取得 Doppler 符号/坐标系/速度权威后另建 formal v3，重做数据、normalization 和 checkpoint 链，不能静默覆盖 v2。
8. 清理与研究声明：固定 sequence_length=1 或真正实现时序；修可变默认参数/宽泛异常；收敛 formal 入口；澄清 Karras/EDM 和 EMA consistency 命名；最后统一正式指标与消融。

## 轻量测试结果

- 当前版本共直接运行 90 项针对性单元测试：89 通过，1 失败。
- 唯一失败是 `test_formal_training_yaml_defaults` 仍期待 4 卡，而 YAML 已明确改为默认 2 卡；正式 launcher 会从 `cuda_devices` 派生 `num_gpus`，因此属于测试/兼容字段失配，不是训练入口失效。
- observed-supervision、probabilistic-mapping、distributed-training、checkpoint-selection、formal-inference 现有测试均通过；这些测试没有覆盖本报告发现的反例，所以“通过”不能否定相应缺陷。
