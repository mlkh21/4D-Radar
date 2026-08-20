<!-- 文件功能：记录候选数据切换至正式训练/推理协议的实施计划。 -->
# 正式训练协议切换计划

## 目标

将已验收的 header-time sensor-aware 数据与 Radar normalization artifact 接入正式训练/推理入口，并阻止旧 checkpoint 被隐式续训。

## 阶段

### 阶段 1：调用链与风险审计

- [x] 核验候选 garden/loop3 manifest、artifact 和单样本加载结果
- [x] 审计训练/推理启动器、配置覆盖、checkpoint 恢复与既有测试
- **状态：** 完成

### 阶段 2：RED/GREEN 小步修改

- [x] 先增加配置、数据根、独立结果根和显式续训协议测试
- [x] 修改默认配置及正式 launcher
- [x] 保留旧数据、checkpoint、日志和结果，不自动启动训练
- **状态：** 完成

### 阶段 3：代码审查与验证

- [x] 运行聚焦单元测试、shell 语法、配置预检和差异检查
- [x] 审查路径、artifact/hash、checkpoint 和训练/推理接口一致性
- [x] 更新 `TODO/findings.md`、`TODO/task_plan.md`、`TODO/progress.md`
- **状态：** 完成

### 阶段 4：训练入口包路径续修

- [x] 复核 `unified_train.py`、`cd_train_optimized.py`、`checkpoint_chain.py` 导入链
- [x] 增加脱离仓库工作目录的直接脚本入口 RED 测试
- [x] 让训练入口同时显式引导包根和既有模块根
- [x] 运行聚焦回归并更新三份项目 TODO
- **状态：** 完成

### 阶段 5：真实 batch metadata 拼接续修

- [x] 复核真实首样本、Dataset metadata、三个 DataLoader 调用端和失败输出
- [x] RED：锁定含合法 JSON null 的 preprocess policy 可安全组成 batch
- [x] GREEN：增加共享 collator 并接入统一训练、独立 CD 和条件推理
- [x] 无损归档零 epoch 失败日志并验证可 fresh 重跑
- [x] 运行聚焦回归并同步三份项目 TODO
- **状态：** 完成

### 阶段 6：正式训练完成后的全链路计划

- [x] 核验 VAE 完成条件、checkpoint 与重建诊断入口
- [x] 核验 LDM 训练、训练期验证和离线 checkpoint 选择门禁
- [x] 核验 CD 训练语义、checkpoint 链诊断和生成入口
- [x] 核验 loop3 正式推理、独立评价、阈值校准与地图更新入口
- [x] 输出带停止条件、失败处理和精确命令的分阶段执行计划
- **状态：** 完成

### 阶段 7：8 GB 单卡正式协议 mini 训练入口

- [x] 审计 legacy mini 与正式 candidate/artifact/checkpoint 协议差异
- [x] RED：锁定正式 normalization、独立输出、单卡低负载和温度门禁
- [x] GREEN：增加显式 formal mini 模式，保留 legacy 用例兼容
- [x] 更新 mini 使用说明和项目三份 TODO 记录
- [x] 运行静态、单元和无训练预检；不自动启动任何训练
- **状态：** 完成

## 错误记录

| 错误 | 尝试 | 处理 |
|---|---:|---|
| `python -m unittest test.unit...` 因仓库 `test/` 不是 Python package 而无法导入 | 1 | 不重复模块路径方式；改为直接执行三个既有测试文件 |
| 首次同时追加 planning/TODO 计划时使用了不存在的 TODO 尾部标题 | 1 | 已确认补丁未部分应用，改用两个文件各自的真实锚点分别追加 |
| RED 组合命令后半段触发 OpenMPI 无可用网络接口 | 1 | mini RED 已获得精确失败证据；后续把 payload 测试拆开并设置禁用 MPI 相关环境，避免重复组合命令 |
| 新的源目录预检使旧 fresh-scratch 测试提前失败 | 1 | 保留更安全的先验顺序，给测试补最小有效源目录后继续验证 scratch 创建语义 |
| `nvidia-smi` 在设备沙箱内无法连接驱动 | 1 | 只读命令在沙箱外复核成功；未修改功耗、频率或驱动状态 |
| `conda run ... python -` 返回 0 但未执行 heredoc artifact 校验 | 1 | 改用 `--no-capture-output` 转发 stdin，并用错误 SHA 行为测试锁定 fail-closed |
