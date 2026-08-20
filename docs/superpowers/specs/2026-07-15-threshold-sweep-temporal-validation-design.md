# 阈值扫描时间块 Validation 协议设计

日期：2026-07-15

## 目标

让 `sweep_occ_threshold.py` 的 train/validation 成员选择与正式训练当前采用的时间块协议一致：排序后的连续前缀作为 train，连续后缀作为 validation。彻底删除只服务旧随机划分的 `split_seed`，避免阈值校准继续复现已废弃的逐帧随机交错协议。

## 根因

正式训练已通过 `temporal_block_split_indices()` 使用连续前缀/后缀划分，但阈值扫描脚本仍在本地用 seeded `torch.randperm()` 选择成员，对应单元测试也把随机划分固化成正确契约。因此相同 `train_split` 下，阈值校准样本不再等于正式 validation 样本。

历史阈值命令输入的是包含完整连续帧的预测目录，脚本再自行选择 validation，因此无需引入新的清单文件即可修复成员算法。

## 修改范围

生产代码只修改：

- `diffusion_consistency_radar/scripts/sweep_occ_threshold.py`

测试代码只修改：

- `test/unit/test_occ_threshold_grid_protocol.py`

同时更新本设计、实施计划和 `TODO/findings.md`、`TODO/task_plan.md`、`TODO/progress.md`。不修改 `unified_train.py`、Dataset、推理脚本、checkpoint、现有数据或历史结果。

## 协议设计

### 成员选择

`select_evaluation_files(files, evaluation_split, train_split)` 保持输入文件顺序。正式入口传入按文件名排序的完整预测清单。

在 `evaluation_split` 为 `train` 或 `validation` 时，脚本继续要求：

- 文件名能够解析出纯数字 frame ID；
- 输入 frame ID 严格连续；
- 样本数至少为 2；
- `train_split` 严格位于 `(0, 1)`；
- train 与 validation 均非空。

设 `train_size = int(len(files) * train_split)`：

- `train` 返回 `files[:train_size]`；
- `validation` 返回 `files[train_size:]`；
- `all` 保持原行为，返回完整输入清单。

`max_files` 继续在完成成员划分后应用，不改变其现有含义。

### 删除 `split_seed`

从以下公开接口删除 `split_seed`：

- `select_evaluation_files()`；
- `prepare_evaluation_files()`；
- argparse 参数；
- 调用链；
- 新生成的推荐 JSON。

脚本在 `parse_args()` 前检查原始 argv。检测到 `--split_seed` 或 `--split_seed=<value>` 时，用 argparse 风格错误退出，并提示时间块划分不再使用随机种子、调用者应删除该参数。旧参数不会继续出现在 `--help` 中，也不会被静默忽略。

### 输出元数据

推荐 JSON 删除 `split_seed`，新增：

```json
"split_protocol": "temporal_block_prefix_train_suffix_validation"
```

保留 `evaluation_split`、`train_split`、选中帧数和实际评价帧数，便于审计新旧结果。

### 场景边界

阈值只能在训练场景的独立 validation 上标定，不能在 `loop3` 独立 test 场景上重新选阈值。当前脚本仅接收预测目录，缺少可信场景 manifest，无法可靠验证目录身份；本项在 CLI 帮助中明确这一调用约束，但不根据路径名猜测场景。

后续 dataset manifest 修复应提供可机器验证的场景和预处理版本元数据，再增加场景身份保护。本项不提前实现 manifest。

## 测试设计

更新既有协议测试，先观察 RED，再实现最小 GREEN：

- validation 必须是连续后缀；
- train 必须是连续前缀；
- 成员选择不接受或依赖 seed；
- `max_files` 仍在划分后应用；
- 缺帧、非法比例和空划分继续失败；
- 真实 CLI 旧 `--split_seed` 在其他必填参数检查前给出迁移提示；
- 推荐 JSON 包含新 `split_protocol` 且不含 `split_seed`。

最终只运行对应单元测试、Python 编译、CLI 帮助/迁移检查和 `git diff --check`，不运行训练、完整推理或全量阈值扫描。

## 研究协议影响

- 监督信号、target 内容、模型结构、checkpoint、网格大小和每帧体素值均不改变。
- 相同 `train_split` 下 train/validation 数量不变，但成员从随机交错帧变为连续时间块。
- 用新协议重新扫描时，聚合指标与推荐阈值可能变化；历史随机 validation 结果不得与新结果直接比较。
- 本项不重新生成或覆盖任何历史 CSV、JSON、点云或模型结果。

## 非目标

- 不抽取新的公共切分模块；
- 不让阈值脚本导入体量较大的训练入口；
- 不修改训练随机性或 DataLoader seed；
- 不实现 dataset manifest；
- 不校准或发布新的正式阈值；
- 不修复其他 P0/P1 项目。
