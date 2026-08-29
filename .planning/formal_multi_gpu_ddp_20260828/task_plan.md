<!-- 文件功能：记录 formal v2 VAE/LDM/CD 服务器 2--4 GPU DDP 改造计划。 -->
# Formal v2 多 GPU DDP 实施计划

## 目标

为正式 `train_unified.sh all` 增加真实的单机 2--4 GPU DDP 支持，同时保持单卡兼容、formal 数据/checkpoint 身份、有效 batch size、指标含义和 fresh/resume 安全合同。

## 当前阶段

已完成；真实 2--4 GPU NCCL smoke 留待服务器执行。

## 阶段

### 阶段 1：调用链和不变量

- [x] 审计 VAE/LDM/CD 模型、优化器、DataLoader、指标、EMA、日志和 checkpoint 全调用链。
- [x] 冻结 1/2/4 GPU 的 per-rank batch、梯度累积、sampler 与结果写入合同。
- [x] 记录单卡兼容和 `all` 编排边界。
- **状态：已完成。**

### 阶段 2：RED 测试

- [x] 增加 launcher 的 1/2/4 GPU、非法卡数、`torchrun` 和 `all` 编排协议测试。
- [x] 增加分布式 helper、sampler、指标聚合、rank-0 写入及 checkpoint unwrap 测试。
- [x] 运行聚焦 RED，确认命中真实缺口。
- **状态：已完成。**

### 阶段 3：共享 DDP 基础设施

- [x] 初始化/销毁单机 NCCL 进程组并绑定 `LOCAL_RANK`。
- [x] 实现 rank/world-size、barrier、all-reduce 和模型 unwrap 公共接口。
- [x] 单卡路径不初始化进程组，保持旧 checkpoint 可读。
- **状态：已完成。**

### 阶段 4：VAE/LDM/CD 接线

- [x] DataLoader 使用 DistributedSampler，训练 epoch 调用 `set_epoch()`。
- [x] VAE/LDM/CD 仅 rank 0 写日志、CSV、checkpoint，所有 rank 使用全局指标更新最佳状态。
- [x] CD 在线模型使用 DDP，EMA 按同步后的在线参数在各 rank 一致更新。
- [x] 保持 1/2/4 GPU 有效 batch size 为 16，3 GPU 显式记录为 18。
- **状态：已完成。**

### 阶段 5：launcher 与恢复链

- [x] `CUDA_DEVICES` 只接受 1--4 个不重复编号，并以可见卡数启动 `torchrun`。
- [x] `all` 在 shell 层顺序启动 VAE→LDM→CD，每阶段独立创建/回收进程组。
- [x] fresh、resume、父 checkpoint 和失败退出继续 fail-closed。
- **状态：已完成。**

### 阶段 6：验证、审查与文档

- [x] 运行 CPU/静态/协议回归，不运行长时间训练。
- [x] 如本机仅单 GPU，明确 2--4 GPU 只完成合成/进程级验证，服务器需先做短 smoke。
- [x] 审查隐藏依赖、接口不匹配、监督/体素/指标影响。
- [x] 更新 README、TODO/findings.md、TODO/task_plan.md、TODO/progress.md。
- **状态：已完成。**

## 关键决策

| 决策 | 理由 |
|---|---|
| 使用 DDP 而非 DataParallel | 4 GPU 时避免 batch=2 导致空闲卡和 GPU0 汇总瓶颈 |
| 每阶段独立 `torchrun` | VAE→LDM→CD 有父 checkpoint 依赖，避免在 Python 内嵌套启动器 |
| 2/4 GPU 有效 batch size 固定 16；3 GPU 显式为 18 | 单卡历史合同为 16；3 无法用固定整数 accumulation 精确整除，禁止静默伪装 |
| 不自动运行服务器长训练 | 本地没有 2--4 GPU 正式训练条件，遵守项目安全规则 |

## 错误记录

| 错误 | 尝试 | 处理 |
|---|---:|---|
| 测试通过 `__new__` 构造 trainer，缺少新增 distributed 字段 | 1 | 统一回退为显式单进程上下文，保持旧内部接口 |
| 导入 KarrasDenoiser 触发旧 `mpi4py`/OpenMPI 初始化 | 1 | 移除 Karras 核心的 eager MPI 依赖，设备改由输入张量决定 |
| 收尾命令使用了 LDM vertical 测试旧路径 | 1 | 用 `rg --files` 定位为 `test/unit/` 后重跑，未执行训练 |
| LDM CPU 回归命中 Karras 感知损失硬编码 `.cuda()` | 1 | 增加显式 device 接口，formal trainer/inference 使用当前本地设备 |
| 最小 LDM resume 测试对象缺少正式入口才有的 config | 1 | 仅最小测试对象跳过 batch 身份检查，正式入口继续 fail-closed |
