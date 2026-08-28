<!-- 文件功能：记录 formal v2 0--80 m 在 8 GB 单卡上的受保护 mini 训练实施计划。 -->
# Formal v2 8 GB mini 训练实施计划

## 目标和边界

- 将现有旧 full120/v1 mini 入口升级为当前 `formal_data_v2`、`formal_mini_chain_v2` 和 0--80 m 网格。
- 从正式 temporal split 中确定性选择少量 train/validation 帧，不伪造或改写正式数据身份。
- 输出到全新的 `test/result/formal_mini_v2_80m_8gb_v1/`，拒绝覆盖或隐式续训。
- 保留温度、显存、单阶段和时长保护；默认 smoke 为 1 epoch，独立 short VAE 为 3 epoch；不自动启动 GPU 训练。
- 不修改或删除正式数据、normalization、checkpoint、日志和历史结果。

## 阶段

### 阶段 1：调用链和协议设计

- [x] 核对 Dataset、formal split/data protocol、normalization 和 checkpoint payload 的实际消费链。
- [x] 确定 mini train/validation 选择及 checkpoint 可审计字段。
- **状态：完成。**

### 阶段 2：RED 测试

- [x] 增加 formal mini v2 协议、确定性帧限制、旧 v1 拒绝和保护入口路径测试。
- [x] 运行聚焦测试，确认失败命中旧入口缺口。
- **状态：完成。**

### 阶段 3：最小实现

- [x] 训练入口只在 `formal_mini_chain_v2` 下接受显式小样本限制。
- [x] mini 脚本接线 formal v2 数据/split/protocol/artifact，并隔离新结果根。
- [x] checkpoint 保存实际 mini 选择身份，不改变正式 full-chain 行为。
- **状态：完成。**

### 阶段 4：审查与验证

- [x] 运行单元测试、shell 语法、Python 编译和 `git diff --check`。
- [x] 运行无训练 preflight；不运行实际训练。
- [x] 审查隐形依赖、接口不匹配、覆盖风险和指标边界。
- **状态：完成。**

### 阶段 5：文档与交付

- [x] 更新 mini 文档、根 README、TODO/findings.md、TODO/progress.md、TODO/task_plan.md。
- [x] 给出用户可直接执行的 preflight 与 VAE mini 命令，以及停止条件。
- **状态：完成。**

### 阶段 6：1 epoch smoke 验收与 short profile 设计

- [x] 只读验收用户生成的 VAE checkpoint、指标、协议、GPU 冷却状态和错误日志。
- [x] 确定 short profile 使用 fresh 结果根、VAE 3 epoch 和更严格温度上限，不覆盖 smoke 结果。
- **状态：完成。**

### 阶段 7：short_train RED/GREEN

- [x] 先补 profile 参数、隔离结果根、固定 epoch/温度和错误组合测试。
- [x] 最小扩展保护 runner；不修改训练核心，不自动启动训练。
- **状态：完成。**

### 阶段 8：回归、文档与交付

- [x] 运行 fake-GPU 行为测试、shell 语法、静态检查和无训练 short preflight。
- [x] 更新 mini README 与三份 TODO，给出用户下一条命令。
- **状态：完成。**

### 阶段 9：LDM preflight 父 checkpoint 身份闭环

- [x] 验收 short VAE，并沿 `runner → train_minimal → unified_train` 核对 LDM 父 checkpoint 调用链。
- [x] RED 证明 preflight 只检查父 checkpoint 文件存在、未验证 stage/protocol/data identity。
- [x] 在不创建 config/output 的前提下复用正式训练的 checkpoint identity 校验。
- [x] 运行负向/正向回归及真实无训练 LDM preflight，更新三份 TODO。
- **状态：完成。**

### 阶段 10：500 帧 × 20 epoch 中型训练 profile

- [x] 核对 formal split 容量、当前门禁和 short VAE 实测吞吐。
- [x] 确认每阶段总计 500 帧（400 train/100 validation）、20 epoch，并在 RTX 4070 Laptop 分阶段运行；服务器使用 full split 20 epoch。
- [x] 新增独立 profile、结果根和 RED/GREEN；保留 smoke/short 行为。
- [x] 完成 laptop 与服务器零训练 preflight、文档和 TODO 更新；不自动启动长训练。
- **状态：完成。**

### 阶段 11：medium VAE CUDA allocator 断言修复

- [x] 只读核对失败现场、PyTorch/CUDA 版本和 allocator 环境变量真实传播链。
- [x] 为不兼容的 `expandable_segments` 与 fresh v2 结果根补 RED/GREEN，保留失败的 v1 现场。
- [ ] 运行无训练回归和最小 GPU backward 诊断，不自动重跑 20 epoch。
- [x] 更新 README、结果索引和三份 TODO，交付新的显式运行命令。
- **状态：进行中；GPU 验证等待空闲显存恢复到 6500 MiB。**

## 错误记录

| 错误 | 尝试 | 处理 |
|---|---:|---|
| `limit_frame_ids_by_scene` ImportError | 1 | RED 预期缺口，进入实现 |
| `build_formal_mini_selection` ImportError | 1 | RED 预期缺口，进入实现 |
| 完整 checkpoint 测试被沙箱 OpenMPI socket 限制中止 | 1 | 新增用例改为具名直跑，稳定得到目标 ImportError；完整回归后续按既有环境方式运行 |
| bad-artifact 负向预检超过测试 15 s | 1 | 根因是先重建全量 split/manifest 再比较 1 KiB artifact hash；调整为先做无副作用 SHA 快速拒绝，再执行完整协议验证 |
| 沙箱内 `nvidia-smi -i 0` 无法读取 GPU 状态 | 1 | 硬件预检必须真实读取设备；改为申请沙箱外只读预检，不启动训练 |
| 新增具名 RED 使用 `test.unit...` 导入失败 | 1 | `test/unit` 不是 package；改用测试文件原生 unittest 参数和 importlib 具名加载，不计为功能测试结果 |
| `CUDACachingAllocator.cpp:2586` expandable segment 内部断言 | 1 | medium VAE 第 1 epoch 第 50 个 batch 的真实失败；保留 v1 现场，进入 allocator 调用链诊断，不原命令重试 |
| 配置 heredoc `NameError: os is not defined` | 2 | 两次按重复 `import sys` 上下文移动 import 都命中了错误 heredoc；改为在唯一 allocator 读取点局部导入，并先跑单项测试再恢复全套回归 |
| 正式数据预检 heredoc `IndentationError` | 2 | 第一次只修了 artifact heredoc，定向测试显示实际先失败于 scene 配置解析 heredoc；检查全部 heredoc 后去除该片段三个顶层 import 的缩进 |
