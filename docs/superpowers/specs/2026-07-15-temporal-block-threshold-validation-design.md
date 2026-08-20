# 阈值扫描时间块 Validation 协议设计

## 背景

正式训练入口已经把单场景样本划分改为连续时间块：排序后的前缀用于训练，后缀用于验证。`sweep_occ_threshold.py` 仍复制旧版 seeded `torch.randperm`，导致阈值校准使用的所谓 validation 与训练期间的 validation 成员不一致。

本项是 P0-01 的协议收尾，只修复阈值扫描的成员选择，不修改模型、训练、推理或指标公式。

## 目标

1. `train` 和 `validation` 选择与正式训练的连续时间块规则一致。
2. 删除已经失去语义的 `split_seed` Python API、CLI 参数和 JSON 字段。
3. 旧命令使用 `--split_seed` 时明确失败并给出删除提示，不能静默忽略。
4. 输出 JSON 明确记录当前切分协议，便于区分历史随机校准结果。
5. 保持现有帧连续性、非空划分、缺 target 和 `max_files` 安全检查。

## 非目标

- 不修改 `unified_train.py` 或 Dataset 排序逻辑。
- 不新增 manifest 或场景身份字段；该能力属于下一项数据 manifest 修复。
- 不自动重算或覆盖历史阈值 CSV/JSON。
- 不修改 target 重采样、阈值候选、指标累计或推荐阈值选择逻辑。
- 不运行训练、完整推理或全量阈值扫描。

## 方案选择

### 采用：阈值脚本内最小有序切片

`select_evaluation_files()` 保留现有输入校验，在得到 `train_size` 后直接返回：

- `train`：`ordered[:train_size]`
- `validation`：`ordered[train_size:]`
- `all`：原样返回完整输入

主入口已经对 `*_voxel.npy` 文件名排序；成员选择函数继续要求调用方传入有序、数字且严格连续的完整预测清单。

### 不采用：从训练入口导入 helper

从 `unified_train.py` 导入纯切分函数会把阈值诊断脚本耦合到训练模块及其依赖，增加启动和维护成本。

### 不采用：本轮抽取公共 split 模块

公共模块在长期上可以消除两处简单逻辑重复，但会迫使本轮再次修改已有大量未提交变更的训练入口，超过单根因修复范围。若出现第三个正式消费者，再单独抽取。

## API 与 CLI

### Python API

修改为：

```python
select_evaluation_files(files, evaluation_split, train_split)
prepare_evaluation_files(files, evaluation_split, train_split, max_files)
```

两个函数都不再接受 `split_seed`。

### CLI

从 argparse 中删除 `--split_seed`。在 `parse_args()` 之前扫描原始 argv，若出现以下任一形式则通过 `parser.error()` 退出：

- `--split_seed 42`
- `--split_seed=42`

错误信息说明时间块切分不使用随机种子，并要求从命令中删除该参数。该检查应发生在必填路径校验或输出创建之前。

`--evaluation_split` 的帮助文本改为“按排序帧的连续时间块选择子集”。

## 输出协议

推荐阈值 JSON 删除：

```json
"split_seed": 42
```

并新增：

```json
"split_protocol": "temporal_block_prefix_train_suffix_validation"
```

`evaluation_split`、`train_split`、`selected_frame_count` 和 `evaluated_frame_count` 保持不变。

历史 JSON 不会被改写；消费者可通过 `split_protocol` 是否存在区分新旧协议。

## 错误处理

- 非 `all` 模式仍要求帧 ID 为纯数字且严格连续。
- 样本数小于 2、非法 `train_split` 或空划分继续 fail-fast。
- `max_files` 继续在完成完整清单切分之后应用，避免改变 validation 边界。
- 旧 `--split_seed` 在任何评估或结果写入前失败。
- 场景是否真的是训练场景 validation 暂时无法仅凭目录可靠判断；CLI 帮助应明确输入必须是单一训练场景的完整连续预测清单。后续 manifest 将提供机器可验证的场景保护。

## 测试设计

在既有 `test/unit/test_occ_threshold_grid_protocol.py` 中先写 RED 测试：

1. validation 精确等于排序清单的连续后缀。
2. train 精确等于连续前缀，二者完整且互斥。
3. `max_files` 在 validation 切分后截取尾部子集的前若干帧。
4. Python API 不再接受 `split_seed`。
5. 真实 CLI 的 `--split_seed` 与 `--split_seed=42` 都返回迁移错误。
6. JSON 包含 `split_protocol` 且不包含 `split_seed`。

GREEN 后运行该测试文件、相关 P0-01 切分测试、Python 编译、CLI 帮助和 `git diff --check`。不运行数据集训练或全量阈值扫描。

## 影响

- 监督信号、target 内容、体素网格、每帧预测体素值、模型和 checkpoint 均不改变。
- 相同总帧数和 `train_split` 下，train/validation 数量保持不变；成员从随机分散帧改为连续前缀/后缀。
- 阈值扫描累计的占据体素数量可能因成员变化而改变，推荐阈值及 Precision/Recall/F1/IoU 也可能变化。
- 旧随机 validation 的指标与新时间块 validation 指标不能直接横向比较。
- 历史 `loop3` 校准结果仍属于独立 test 场景结果；本项不会把它们重新解释为正式 validation，也不会覆盖它们。

## 修改文件

- `diffusion_consistency_radar/scripts/sweep_occ_threshold.py`
- `test/unit/test_occ_threshold_grid_protocol.py`
- `TODO/findings.md`
- `TODO/task_plan.md`
- `TODO/progress.md`

## 完成标准

- 新 RED 测试确实因旧随机协议或仍存在的 `split_seed` 而失败。
- 最小实现后全部聚焦测试通过。
- argparse 帮助中没有 `--split_seed`，旧命令得到明确迁移错误。
- 新 JSON 记录时间块协议且不再记录随机种子。
- 没有训练、完整推理、全量扫描、结果覆盖、暂存或提交。
