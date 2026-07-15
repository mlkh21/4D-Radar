# 单场景连续时间块训练/验证切分设计

## 文档用途

本文定义 P0-01 的最小修复协议：消除同一场景内按单帧随机切分造成的时序泄漏。
本设计只覆盖正式训练入口的 train/validation 成员划分，不扩展到数据预处理、模型结构、
推理协议或独立测试场景选择。

## 背景与根因

正式训练 launcher 当前只将 `garden` 链接到训练数据根。`unified_train.py` 随后创建两个
`split='train'` 的 Dataset，并对同一份按文件名排序的样本执行 seeded `torch.randperm()`。
随机种子使划分可复现，却会让高度相关的相邻帧大量交错进入 train 和 validation，导致验证指标、
checkpoint 选择及消融结论偏乐观。

`loop3` 已承担独立场景测试职责，不能改作 validation，否则当前两场景协议将失去独立 test。

## 目标与非目标

目标：

- `garden` 按现有样本顺序形成两个不交错的连续块；
- 前 `train_split` 比例作为 train，剩余后缀作为 validation；
- validation 保持无数据增强；
- `loop3` 继续作为独立 test；
- 非法或空划分应在训练开始前明确失败。

非目标：

- 本轮不实现多场景逐场景切分、bag 级 manifest 或时间戳间隔阈值；
- 不修改 Dataset 的场景发现、滑窗、增强或体素加载协议；
- 不修改 target、模型、损失、checkpoint 或推理阈值；
- 不运行训练、完整推理或全量评估。

## 方案比较

1. **当前单场景连续块切分（采用）**：在正式 `garden` 训练根内使用前缀 train、后缀 validation。
   改动最小，直接消除随机交错，同时保留 `loop3` 独立测试。
2. **每场景分别连续切分**：让 Dataset 加载全部训练场景，再逐场景划分前后缀。
   更通用，但会改变多场景样本纳入规则，超出当前单场景 P0-01 的最小范围。
3. **按场景完全隔离 validation**：使用 `garden` 训练、`loop3` 验证。
   隔离最强，但会占用唯一独立测试场景，因此不采用。

## 接口与数据流

新增纯函数 `temporal_block_split_indices(dataset_size, train_split)`：

```text
train_size = int(dataset_size * train_split)
train_indices = [0, ..., train_size - 1]
val_indices = [train_size, ..., dataset_size - 1]
```

正式数据流变为：

```text
garden 按现有文件名排序加载
  -> temporal_block_split_indices()
  -> 前缀 Subset（train，启用既有增强）
  -> 后缀 Subset（validation，禁用增强）
loop3
  -> 保持现有独立推理/评价路径
```

`training_seed` 继续控制 Python、NumPy、Torch、模型初始化和 train DataLoader shuffle。
train/validation 成员不再由 `split_seed` 随机决定；历史配置中的 `split_seed` 暂时保留，避免扩大配置
兼容修改范围。

连续块之间仍存在一个时间边界，但不再有随机散布的大量相邻帧跨集合。若后续获得 bag 边界、
可靠时间戳或更多场景，再单独设计 embargo gap 和多场景分组协议。

## 错误处理

- `dataset_size < 2`：训练前报错，避免空 train 或 validation；
- `train_split` 不在 `(0, 1)`：训练前报错；
- 取整后任一集合为空：报告 dataset size、比例和计算出的 train size；
- train/validation 两个 Dataset 样本数量不一致：保留现有安全检查。

## TDD 与验证

先在 `test/unit/test_vae_checkpoint_protocol.py` 添加 RED 测试，覆盖：

1. 十个有序样本按 `0.8` 精确得到 `0..7` 和 `8..9`；
2. 两集合互斥、并集覆盖全部样本、各自保持升序连续；
3. 相同输入重复调用结果一致且不依赖随机状态；
4. 单样本、非法比例及取整空划分明确失败。

确认 RED 因当前缺少连续块接口或仍执行随机排列而失败后，只修改最小生产代码使其 GREEN。
完成后运行该单元测试文件、`py_compile` 和 `git diff --check`。不自动运行训练或模型推理。

## 文件边界

- `diffusion_consistency_radar/scripts/unified_train.py`：新增纯切分函数并替换 main 中的随机成员划分；
- `test/unit/test_vae_checkpoint_protocol.py`：增加连续块切分契约测试；
- `TODO/task_plan.md`、`TODO/findings.md`、`TODO/progress.md`：记录设计、RED/GREEN 和验证结果。

不修改 `dataset_loader.py`、launch 脚本、mini runner 或配置生成脚本。

## 影响说明

- **监督信号**：每个样本的 LiDAR target、Radar/IR 输入和损失定义完全不变；
- **体素数量**：单帧张量尺寸和占用体素数不变，当前单场景 80/20 的 train/validation 样本总数不变；
- **模型与 checkpoint**：网络参数量、checkpoint schema 和恢复协议不变；
- **指标**：validation 时间分布更独立，指标可能下降并更接近时序外推表现；历史随机切分结果不能与新结果
  直接视为同一协议；
- **独立测试**：`loop3` 的角色和现有评价结果不变。
