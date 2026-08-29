<!-- 文件功能：记录 formal v2 多 GPU DDP 改造进展和测试结果。 -->
# Formal v2 多 GPU DDP 进展

## 2026-08-28

- 用户授权按 DDP 方向修改，使正式完整训练真实使用 2--4 GPU。
- 已读取 planning-with-files、根 AGENTS 和 test/AGENTS 规则；不会自动运行长训练或删除结果。
- 初步调用链确认 launcher 仅暴露多卡，正式三阶段训练器实际仍为单设备；直接给 `torchrun` 命令会产生重复训练和 checkpoint 冲突。
- 已建立独立计划，阶段 1 开始。
- 已审计 VAE/LDM/CD 训练循环：三者均需 rank-0 I/O、在线模型 DDP 包装和跨 rank 指标聚合；CD 只包装在线模型，EMA 在各 rank 同步更新。
- 已冻结 sampler 合同：训练使用标准 padding sampler并记录最多 `world_size-1` 个重复，验证使用无重复分片；2/4 GPU 保持有效 batch 16，3 GPU 显式使用 18。

## 测试结果

| 测试 | 结果 | 备注 |
|---|---|---|
| 尚未运行 | — | 先完成调用链和 RED 设计 |

## 错误记录

| 错误 | 尝试 | 处理 |
|---|---:|---|
| 暂无 | 0 | — |
## 2026-08-28 调用链审查进展

- 已核对 VAE、LDM、CD 的模型构造、优化器、训练/验证入口、EMA 与 checkpoint 路径。
- 已确认正式多模态前向可以纳入 DDP；legacy `unet_3d` 旁路需要显式适配。
- 尚未运行训练；下一步补齐精确的指标聚合字段与现有测试合同，然后进入 RED 测试。

## 2026-08-28 阶段 1 完成

- 已冻结 DDP 进程拓扑、batch/累积、训练补齐、验证无补齐、rank-0 写入以及 checkpoint 解包合同。
- 已确认 `all` 的正确边界是父 shell 顺序启动三个独立分布式 stage。
- 进入阶段 2：先增加 helper 与 launcher 协议测试并确认现状失败。

## 2026-08-28 RED 测试进展

- 已新增 `test/unit/test_distributed_training_protocol.py`，覆盖 batch 计划、无补齐验证分片、聚合/解包、checkpoint 元数据和 launcher 编排。
- 首次执行命中测试入口未加入仓库根的自有问题，已修正 `sys.path`；需重新执行确认真实 RED 缺口。

## 2026-08-28 RED 命中与 helper 实现

- 修正测试入口后，RED 准确命中缺失的 `diffusion_consistency_radar.distributed_training`。
- 已实现共享 DDP helper；8 项聚焦测试中 5 项 helper 测试通过，3 项 launcher 测试按预期失败。
- 未初始化 NCCL、未访问数据集、未启动训练。下一步将上下文接入 VAE/LDM/CD，再修改 launcher。

## 2026-08-28 VAE 接入进展

- 已接入 DDP 包装、全局训练/验证指标归并、无 `module.` checkpoint 保存、rank-0 日志/CSV/checkpoint 门禁和全局有效 batch 展示。
- 已保留单卡跳过非有限 batch 的既有合同；多卡改为所有 rank 一致失败，避免反向次数分叉。
- 尚未完成 LDM/CD 与 main DataLoader 接入，暂未运行回归测试。

## 2026-08-28 LDM 接入进展

- 已接入 LDM DDP 包装、可选未使用参数发现、训练损失组件全局聚合、验证计数全局聚合和解包保存/恢复。
- 已将正式 LDM 验证噪声绑定到样本路径，消除不同 world size 改变指标输入的隐形依赖。
- 尚需完成 LDM `train()` rank-0 门禁、main sampler、CD 和 launcher。

## 2026-08-28 三阶段 Python 接入进展

- LDM `train()` 已完成 rank-0 写入与全局有效 batch 展示。
- unified main 已初始化/销毁进程组，训练使用 `DistributedSampler`，验证使用无补齐 sampler，并向 VAE/LDM/CD 传递同一上下文。
- CD 已完成在线模型 DDP、EMA 解包、全局 loss、rank-0 原子保存和尾部梯度修正。
- 尚未完成 standalone CD main 和 shell launcher；尚未执行语法/回归测试。

## 2026-08-28 launcher 与首轮回归

- standalone CD main 和 formal shell launcher 已接入 1--4 GPU 进程拓扑；`all` 保持三阶段独立作业。
- shell 语法和四个 Python 文件编译检查通过。
- 首轮测试发现旧 trainer mock 缺少 distributed 字段，已兼容；随后发现 Karras eager import 触发旧 MPI，已移除该正式调用链隐形依赖。
- VAE sparse/robustness 20 项与 LDM validation 5 项已通过；尚需运行完整聚焦回归和 launcher 协议测试。

## 2026-08-28 完整聚焦回归与收尾审查

- 分布式协议初轮 11/11、mini launcher 协议 20/20、VAE checkpoint 26/26、VAE sparse loss 20/20、LDM validation 5/5、LDM vertical loss 81/81 全部通过；收尾新增 legacy DDP 旁路断言后分布式协议为 12/12。
- CD entrypoint 与多模态 CD 接口脚本测试通过；launcher shell 语法、相关 Python 编译和 `git diff --check` 通过。
- 完整 airborne 多模态 CPU 测试因执行超过 60 秒被主动停止，不将其记为通过；未运行 GPU/NCCL 或任何长训练。
- 收尾审查补齐 LDM legacy DDP 旁路拒绝，并更新 README 与项目 TODO；下一步仅需重跑受影响的短时协议测试和静态检查。
- 收尾重跑首次把 LDM vertical 测试误写为旧路径 `test/test_ldm_vertical_structure_loss.py`，命令在读取文件前失败；实际路径为 `test/unit/test_ldm_vertical_structure_loss.py`，不属于代码失败。
- 修正路径后 81 项中 58 项通过、23 项因 `KarrasDenoiser` 硬编码 `.cuda()` 在当前无可见 GPU 环境失败；该依赖同时可能影响 rank-local 设备语义，已改为 trainer 显式传入 device，等待重跑。
- Karras device 修复后 LDM vertical 81/81、推理接口 38/38、CD 多模态接口与 entrypoint 测试通过；shell、Python 编译和 `git diff --check` 同步通过。
- 稳定 `scene/frame_id` 噪声身份回归首轮分布式协议 14/14；LDM validation 4/5，剩余 1 项为最小 `__new__` trainer 缺少 config，已加兼容并让测试显式携带 noise identity，等待重跑。
- 最终重跑通过：LDM validation 5/5、LDM vertical 81/81、分布式协议 14/14；相关 Python 编译、launcher shell 语法和 `git diff --check` 通过。
- 阶段 6 完成。本地没有可执行的 2--4 GPU NCCL 环境，因此只把服务器 preflight/短时 smoke 作为后续运行门槛，不宣称真实多卡训练已经验证。
