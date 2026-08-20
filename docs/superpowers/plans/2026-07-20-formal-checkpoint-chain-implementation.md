# Formal Checkpoint Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 固化带完整网格、融合结构和父权重哈希的 VAE/LDM/CD checkpoint 协议，并让正式部署入口在生成前 fail-closed。

**Architecture:** 新增轻量 `checkpoint_chain.py` 负责安全读取、哈希和协议校验；独立诊断脚本提供 CLI 和可选 CPU 构建加载。训练保存路径只增加协议元数据，三个正式 launcher 先调用诊断门禁，不修改历史权重或运行训练。

**Tech Stack:** Python 3、PyTorch `weights_only` 兼容加载、SHA-256、Bash、unittest/pytest。

## Global Constraints

- 不运行长时间训练、完整推理或数据预处理。
- 不覆盖、删除、移动、重链任何已有数据、checkpoint、日志和实验结果。
- 新文件使用中文文件头注释；代码注释默认中文。
- 修改后更新 `TODO/findings.md`、`TODO/task_plan.md`、`TODO/progress.md`，不暂存、不提交。

### Task 1: 协议校验核心

**Files:**
- Create: `diffusion_consistency_radar/checkpoint_chain.py`
- Test: `test/unit/test_checkpoint_chain_protocol.py`

**Interfaces:**
- `validate_formal_checkpoint_chain(vae_path, ldm_path, cd_path, require_multimodal=True) -> dict`
- `CheckpointChainError` 聚合所有协议错误。

- [x] 先写覆盖有效链、缺字段、网格不一致、父 hash 不匹配、legacy CD 和 symlink 拒绝的失败测试。
- [x] 运行单测确认核心模块尚不存在导致预期失败。
- [x] 实现安全加载、SHA-256、阶段/网格/latent/多模态前缀和父 hash 校验。
- [x] 运行单测，确认所有协议测试通过。

### Task 2: 独立诊断脚本

**Files:**
- Create: `diffusion_consistency_radar/scripts/diagnose_checkpoint_chain.py`
- Modify: `test/unit/test_checkpoint_chain_protocol.py`

**Interfaces:**
- CLI `validate --vae_ckpt PATH --ldm_ckpt PATH --cd_ckpt PATH`。
- CLI `--construct` 在 CPU 上构建并严格加载三阶段，失败不写报告。
- `--report_dir` 只接受不存在或空目录，成功原子写 `checkpoint_chain.json`。

- [x] 先写 CLI 成功/失败和空目录保护测试。
- [x] 运行 RED。
- [x] 实现独立 CLI；默认不导入推理采样，不读取数据。
- [x] 运行 GREEN 与 `py_compile`。

### Task 3: 新 checkpoint 保存元数据

**Files:**
- Modify: `diffusion_consistency_radar/scripts/unified_train.py`
- Modify: `diffusion_consistency_radar/scripts/cd_train_optimized.py`
- Modify: `test/unit/test_vae_checkpoint_protocol.py`
- Modify: `test/unit/test_cd_training_entrypoints.py`

**Interfaces:**
- VAE/LDM/CD 新保存 payload 写 `checkpoint_protocol`、`stage`、`data_grid_config`。
- LDM/CD 写 `model_config` 融合字段及父 checkpoint SHA-256；保留现有构造函数兼容性。

- [x] 先增加 payload 元数据断言并运行 RED。
- [x] 在训练入口计算父权重 hash，传入 trainer；只改保存字典，不启动训练。
- [x] 运行相关单测和静态编译。

### Task 4: 正式 launcher 门禁

**Files:**
- Modify: `diffusion_consistency_radar/launch/inference_ldm.sh`
- Modify: `diffusion_consistency_radar/launch/inference_cd.sh`
- Modify: `diffusion_consistency_radar/launch/inference_uniified.sh`
- Modify: `test/unit/test_formal_inference_protocol.py`

- [x] 先增加静态契约：三份正式入口调用 checkpoint-chain 诊断且不能静默跳过缺失阶段。
- [x] 运行 RED。
- [x] 增加统一门禁调用，门禁在 manifest/第一帧生成前执行；统一入口缺任一阶段直接失败。
- [x] 运行 GREEN、四份 `bash -n` 和聚焦回归。

### Task 5: 记录与最终验证

- [x] 更新三份 TODO 持久化记录，明确当前正式权重审计结果和后续重训计划。
- [x] 运行 checkpoint 协议、VAE/CD/正式入口相关单测、`py_compile`、`bash -n`、`git diff --check`。
- [x] 用当前 `Result/train_results` 做一次只读诊断，确认旧链以非零退出且没有创建报告/改写权重。
