<!-- 文件功能：记录 formal v2 多 GPU DDP 调用链发现和设计结论。 -->
# Formal v2 多 GPU DDP 发现

## 已确认事实

- 正式配置为 `batch_size=2`、`gradient_accumulation_steps=8`，单卡有效 batch size 为 16。
- `hardware.num_gpus=2` 当前只是声明；`MemoryOptimizer` 使用 `torch.device('cuda')`，VAE/LDM/CD 都把模型直接移到单一设备。
- VAE checkpoint 代码有 `isinstance(model, nn.DataParallel)` 兼容分支，但仓库没有创建 DataParallel 的调用点。
- `cm/train_util_cond.py` 有另一套 DDP helper，但正式 `unified_train.py → OptimizedVAETrainer/OptimizedLDMTrainer → ConsistencyDistillationTrainer` 不消费它，不能按文件名推断已支持 DDP。
- launcher 接受逗号分隔 GPU，但目前只设置 `CUDA_VISIBLE_DEVICES`；直接 `torchrun` 会让各进程重复读取全量数据并竞争同一日志/checkpoint。
- 正式 `all` 依赖 VAE checkpoint 再启动 LDM、依赖 LDM checkpoint 再启动 CD，因此应在 shell 父进程按阶段顺序执行独立 `torchrun`。

## 待确认边界

- VAE/LDM epoch 指标的累计方式与最佳 checkpoint 判定需要跨 rank 精确求和，而不是平均 rank 均值。
- DistributedSampler 的 padding 会影响验证集精确计数，需要无重复验证 sampler 或带权去重聚合。
- CD 在线模型、EMA 目标和初始化教师的参数更新/保存边界需要沿真实方法继续审计。
- 当前测试环境只有 RTX 4070 Laptop 单 GPU，真实 NCCL 2--4 GPU 验证需服务器短 smoke。

## 训练循环审计

- VAE/LDM 在 trainer 构造时先 `model.to(cuda)` 再创建 optimizer；DDP 必须在 optimizer 之前包装在线模型，checkpoint 必须保存 unwrap 后的 state dict。
- VAE validation 使用 occupancy 交并计数，适合按 int64 跨 rank 求和；训练 loss/组件必须聚合“总和 + 有效 batch 数”，不能平均各 rank 均值。
- LDM validation 同时包含 latent loss 和 occupancy 计数；需要把 loss sum/count 与 occupancy counts 分开 all-reduce，再由所有 rank 得到相同 best-selector 结果。
- CD 有冻结 LDM 初始化模型、冻结 VAE、在线 `cd_model` 和 `cd_model_ema` 四个模型角色。只有在线 `cd_model` 应包装 DDP；EMA 每个 rank 由同步后的在线参数做相同更新，checkpoint 保存 unwrap 在线模型与本地一致 EMA。
- 三套训练循环当前都直接创建 tqdm、logging、CSV 和 checkpoint；DDP 下非主 rank 必须使用空 handler/禁用 tqdm，并禁止任何文件写入。
- 现有 `all` 递归调用同一 shell 入口，改造后父 shell可继续顺序调用子阶段，但每个子阶段必须单独 `torchrun`，不能让 `all` 自身也被 torchrun 包裹。

## Batch 与 sampler 决策

- 标准 `DistributedSampler` 为保持各 rank 相同步数会最多补齐 `world_size-1` 个训练样本；训练 checkpoint 需记录 sampler padding 数，避免隐藏重复样本。验证不允许 padding 重复，使用按 rank 步进切片的无重复 eval sampler，并通过 all-reduce 精确聚合。
- 2 GPU 可用 per-rank batch 1 × accumulation 8 保持全局有效 batch 16；4 GPU 用 accumulation 4。
- 3 GPU 无法用固定整数 accumulation 精确得到 16。若必须支持 3 GPU，最小透明合同是 accumulation 6、有效 batch 18，并在生成配置、日志和 checkpoint 中显式记录；不得打印成 16。后续 RED 应锁定这项非静默差异。

## 不相关工作树

- `.planning/project_full_review_20260722/` 三个文件在本任务开始前已处于删除状态，属于用户既有改动，本任务不恢复、不暂存、不提交。
## 2026-08-28 DDP 前向边界补充

- `CompleteDualModalityPerceptionNet.forward(...)` 已覆盖正式 Radar/IR/标定输入以及 `noised_latent`，正式 LDM/CD 可安全通过 DDP 包装器的 `forward` 进入模型。
- legacy LDM/CD 当前会直接调用 `model.unet_3d(...)`；若模型外层直接包装 DDP，这条旁路不会经过 DDP `forward`，属于需要消除的隐形依赖。
- `call_cd_denoiser(...)` 依赖 `is_multimodal` 与 `unet_3d` 属性；DDP 包装器不保证透传这些业务属性，因此 CD 需要一个显式前向适配模块，并在保存/EMA/恢复时解包到真实生成模型。
- LDM checkpoint 当前直接保存 `self.model.state_dict()`，CD 也直接保存 `self.cd_model.state_dict()`；多卡实现必须统一解包，避免写入 `module.` 前缀并保持现有推理接口兼容。
- VAE、LDM、CD 的进度条、CSV、文本日志和 checkpoint 当前均没有 rank-0 门禁；多进程直接运行会造成并发覆盖，必须集中为主进程写入。

## 2026-08-28 阶段 1 冻结合同

- 单卡保持 `batch_size=2`、`gradient_accumulation_steps=8`，有效 batch 为 16；2 卡采用每 rank 1、累积 8，有效 batch 为 16；4 卡采用每 rank 1、累积 4，有效 batch 为 16。
- 3 卡无法用固定整数累积精确得到 16，采用每 rank 1、累积 6，有效 batch 为 18，并写入配置、日志与 checkpoint 元数据，禁止显示成 16。
- 训练使用等长 `DistributedSampler`；其补齐样本数必须写入运行元数据。验证使用按 rank 步进的无补齐 sampler，保证全局每个验证样本只计算一次。
- 每个 stage 由 shell 父进程单独启动一次 `torch.distributed.run`；`all` 仍由 shell 顺序编排 VAE、LDM、CD，禁止把整个 `all` 递归脚本包进一个 torchrun 作业。
- 只允许 rank 0 写日志、CSV、checkpoint 和终端进度；所有 rank 都参与训练统计与验证计数的 `all_reduce`。

## 2026-08-28 单卡非有限 batch 兼容边界

- 既有 VAE 单卡合同允许跳过个别非有限 batch，并按实际有效 batch 修正尾部梯度。
- DDP 下不同 rank 独立跳过会破坏反向调用次数一致性，因此所有 rank 先归并有限性；任一 rank 非有限时全部一致失败。
- 单卡继续保持原有跳过行为，多卡采用 fail-closed，不改变既有诊断测试语义。

## 2026-08-28 LDM 验证噪声身份

- 原固定 seed 仅固定“遍历顺序上的随机数流”，分布式无补齐分片会改变样本与噪声的对应关系。
- formal 验证改为 `sample_path_sha256_v1`：每个样本由固定 seed 与样本路径 SHA-256 共同确定噪声，跨 1/2/4 卡保持同一样本同一噪声。
- target、observed mask、体素网格和占用计数公式不变；可能变化的是旧顺序随机流下的 LDM 验证数值，因此新 checkpoint 会显式记录噪声身份。

## 2026-08-28 CD 梯度与显存边界

- CD 的 LDM 模型只在构造阶段复制一次权重，之后不参与训练或目标计算；初始化后释放该 GPU 副本，不改变 teacher/EMA 语义并降低每 rank 常驻显存。
- CD 原尾部累计直接提交了仍按完整 `grad_accum_steps` 缩小的梯度；当 batch 数不能整除累积步数时会低估尾部更新。现按实际余数恢复尺度，与 VAE/LDM 合同一致。
- DDP 只包装在线 CD 学生；EMA 保持每 rank 本地、无梯度，并在同步 optimizer step 后用相同参数独立更新。
- 多卡正式 CD 必须走含真实 IR/标定的多模态 `forward`；缺少 metadata 的 legacy 多模态旁路在 DDP 下显式拒绝，避免绕过 DDP reducer。

## 2026-08-28 旧 MPI 隐形依赖

- `KarrasDenoiser` 所在模块原本在 import 时加载 `dist_util` 和 `random_util`，两者会导入 `mpi4py` 并执行 OpenMPI 初始化。
- 这条旧 MPI 拓扑不属于 formal unified 调用链，会与 torchrun/NCCL 形成双重分布式运行时，并在受限环境中仅导入就失败。
- Karras 默认随机生成器改为直接使用 `torch` 随机接口，三个迭代图像工具的 device 改由输入 `x.device` 决定；不再为正式训练导入旧 MPI 栈。

## 2026-08-28 收尾接口审查

- LDM legacy 训练原先会解包 DDP 后直接调用 `unet_3d`，反向传播不会经过 DDP reducer；多卡现显式拒绝缺少 IR/标定的旁路 batch，单卡 legacy 行为保持兼容。
- 验证阶段使用解包模型是有意设计：验证不反向传播，无补齐 sampler 允许各 rank batch 数不同，最终只归并精确计数。
- 监督 target、observed mask、损失公式、模型空间网格和单帧 `32×128×128=524288` 个体素不变。训练 sampler 最多补齐 `world_size-1` 个样本并写入 checkpoint；验证不补齐、不重复。
- LDM formal 验证噪声从顺序随机流改为样本路径身份，保证同一样本跨 world size 使用同一噪声；这可能改变旧版验证数值，但不改变评价公式。
- 收尾 CPU 回归暴露 `KarrasDenoiser` 无条件 `.cuda()` 的隐藏设备依赖；正式 trainer 现显式传入 rank-local device，未指定时才按 CUDA 可用性回退 CPU，推理入口同步使用同一接口。
- launcher 写入的 distributed protocol/world size/effective batch 原先只有 world size 被 Python 消费；现共享入口同时核对三项与真实 batch/累积/进程数，避免配置声明与优化语义漂移。
- 多进程不再接受缺失 DDP protocol 的直接入口；声明 `single_node_ddp_v1` 后还必须精确匹配对应卡数的 per-rank batch 与梯度累积，不能只让乘积碰巧相同。
- Dataset 的 `return_path` 是绝对路径，若直接作为验证噪声身份，数据迁移到服务器后同一帧会改变噪声。formal 验证现使用 `scene/frame_id`，并把 `scene_frame_sha256_v1` 纳入 checkpoint 恢复校验。
