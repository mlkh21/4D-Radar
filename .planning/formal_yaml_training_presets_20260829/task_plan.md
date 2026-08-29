<!-- 文件功能：记录正式训练 YAML 默认值与环境变量覆盖改造计划。 -->
# 正式训练 YAML 预设与极简启动计划

## 目标

让 `default_config.yaml` 成为 VAE/LDM/CD epoch、每轮训练/验证帧数和默认 GPU 列表的唯一默认来源；同名环境变量仅作为单次运行的高优先级覆盖，同时保持 DDP、formal split、checkpoint 身份和极简启动命令。

## 当前阶段

已完成：YAML 默认值、临时覆盖、阶段帧身份、真实预检和短回归均已闭环。

## 阶段

### 阶段 1：配置与数据语义审计

- [x] 确认 YAML、launcher 环境变量和生成 override 的当前优先级。
- [x] 确认“每 epoch 帧数”在 formal split、DDP sampler 和 checkpoint 中的安全语义。
- [x] 冻结极简 fresh/preflight/resume 命令。
- **状态：已完成。**

### 阶段 2：RED 协议测试

- [x] 增加 YAML 默认值、环境变量覆盖、阶段独立 epoch/帧数和极简入口测试。
- [x] 证明旧 launcher 的固定 20 epoch/删除帧限制行为不满足新合同。
- **状态：已完成。**

### 阶段 3：实现

- [x] 扩展默认 YAML 配置。
- [x] 改造 launcher 的 YAML 读取、严格校验和环境变量覆盖。
- [x] 接入 formal stage selection checkpoint/resume 身份。
- **状态：已完成。**

### 阶段 4：验证与文档

- [x] 运行短时协议/单元/静态回归，不启动训练。
- [x] 审查监督、体素、指标、resume 与 DDP 接口影响。
- [x] 更新 README 与 TODO 三份项目记录。
- **状态：已完成。**

## 关键约束

- 不手工编辑每次生成的 `.default_config.train_override.yaml`。
- 不删除或覆盖数据、checkpoint、日志和实验结果。
- 环境变量未设置时必须完全使用 YAML；设置后只覆盖本次运行。
- 正式子集选择必须确定、可审计，不能让不同 rank 各自随机截断。

## 错误记录

| 错误 | 尝试 | 处理 |
|---|---:|---|
| 新测试首次导入缺少 stage selection builder | 1 | 完成 RED 后实现共享 builder/validator |
| Python 3.8 导入拒绝 `tuple[int, int]` | 1 | 改用已导入的 `Tuple[int, int]` 并补真实模块导入回归 |
| 最小 VAE trainer 缺少新属性 | 1 | resume/payload 兼容路径使用 `getattr(..., None)` |
| 未激活 Conda 时系统 Python 缺少 torch | 1 | 按项目合同使用 `conda run --no-capture-output` 完成真实预检 |
