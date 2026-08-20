# P0-06 Oracle Target 自适应诊断隔离设计

## 目标

从正式推理链中彻底移除“根据当前帧 LiDAR target 占据数量反推预测阈值”的能力，并将其迁移为只消费已保存预测体素的离线 oracle 诊断。正式推理输出不得再由当前测试真值改变；历史 oracle 上限实验仍可通过明确标记的独立脚本复现。

## 背景与根因

当前 `diffusion_consistency_radar/scripts/inference.py` 暴露
`--adaptive_occ_from_target` 和 `--adaptive_target_threshold`。开启后，推理逐帧读取
`target_voxel`，计算 target 占据体素数量，并用预测 occupancy 的第 k 大值反推该帧阈值。
因此测试真值直接改变了预测点云，属于 oracle 评价泄漏。

正式 `inference_ldm.sh` 当前默认没有开启该参数，但通用推理 CLI 和 mini launcher 仍允许启用，容易让诊断结果被误当作可部署结果。

## 架构边界

### 正式推理入口

`inference.py` 只保留固定 `--occ_threshold` 点云生成：

- 从 argparse 和帮助信息删除 `--adaptive_occ_from_target`、
  `--adaptive_target_threshold`。
- 删除 `find_adaptive_occ_threshold()`、逐帧 target 数量反推阈值、相关日志和只服务该路径的 target occupancy loader。
- 保留 `--target_voxel_dir` 与 `--compare_with_target`，它们只用于离线指标计算，不能改变预测体素或点云。
- 保留 `effective_occ_threshold` CSV 列；每帧值恒等于固定的 `--occ_threshold`，用于协议审计。
- 删除正式 CSV 中只由 adaptive 路径填写的 `target_occ_count` 列。
- 保留 `find_matching_voxel_file()` 和 `voxel_to_pointcloud()`，因为正常 target 对比、Radar 基线与固定阈值点云仍使用它们。

### 旧参数迁移错误

旧参数不再注册到 argparse，但在 `parse_args()` 前检查原始参数列表：

- 同时识别 `--adaptive_occ_from_target`、`--adaptive_target_threshold VALUE` 与
  `--adaptive_target_threshold=VALUE`。
- 命中后通过 parser error 立即退出，错误信息说明参数已从正式推理移除，并指向
  `test/diagnostics/occupancy/diagnose_oracle_target_adaptation.py`。
- 检查发生在 checkpoint 加载、输出目录创建和结果写入之前。
- 其他未知参数仍由 argparse 使用标准错误处理。

### 独立诊断入口

新增：

`test/diagnostics/occupancy/diagnose_oracle_target_adaptation.py`

该脚本只处理已保存预测，不执行模型推理。必需输入为：

- `--pred_voxel_dir`：包含 `<frame>_voxel.npy` 的固定阈值推理输出目录；
- `--target_voxel_dir`：包含 `<frame>.npy` 或 `<frame>.npz` 的 target 目录；
- `--output_dir`：全新的 oracle 诊断输出目录。

可配置输入为：

- `--target_threshold`，默认 `0.1`；
- `--source_pc_range`，默认 `[0,-20,-6,120,20,10]`；
- `--model_pc_range`，默认 `[0,-20,-6,40,20,10]`；
- `--target_size`，默认 `[32,128,128]`，顺序为 `Z,X,Y`；
- `--voxel_size`，默认由 `model_pc_range` 和预测网格自动推导；
- `--max_files`，默认 `0`，表示处理全部排序后的匹配帧。

## 数据流

```text
固定阈值 inference.py
  └─ <frame>_voxel.npy（不读取 target 改变输出）
       ↓
diagnose_oracle_target_adaptation.py
  ├─ 严格匹配 <frame>.npy/.npz target
  ├─ 按训练协议裁剪和重采样 target occupancy
  ├─ 统计 target occupied count
  ├─ 使用原 top-k + nextafter 算法计算逐帧 oracle threshold
  ├─ 生成 <frame>_oracle_pcl.npy（XYZ+intensity）
  ├─ 写入 oracle_target_adaptation_frames.csv
  └─ 写入 oracle_target_adaptation_report.json
```

oracle 阈值算法保持原 top-k 意图，并使用与 prediction occupancy 相同 dtype 的
`nextafter` 前驱，确保严格 `>` 真正保留第 k 大体素。历史实现先转 Python float，
float64 前驱与 float32 prediction 比较时会舍入回原值，实际只输出 `k-1` 点；新报告需明确采用修正后的 dtype-aware 协议。target 数量为零时仍以最少一个预测体素作为有效 k；CSV 必须同时记录原始 `target_occ_count` 和实际 `effective_match_count`，避免该兼容行为被隐藏。

## 输出协议

### 每帧点云

文件名：`<frame>_oracle_pcl.npy`

数组形状为 `(N,4)`，列为 `X,Y,Z,intensity`，与原 inference 点云格式一致。诊断点云只用于 oracle 上限分析，不得作为部署输出。

### CSV

`oracle_target_adaptation_frames.csv` 至少包含：

- `index`
- `frame_id`
- `pred_voxel_file`
- `target_voxel_file`
- `target_occ_count`
- `effective_match_count`
- `oracle_occ_threshold`
- `oracle_pred_point_count`
- `pred_to_target_count_ratio`
- `oracle_pointcloud_file`

### JSON

`oracle_target_adaptation_report.json` 必须包含：

- `protocol: "oracle_target_count_matching"`
- `deployable: false`
- 输入目录与输出目录；
- target/grid/physical range 配置；
- 实际处理帧数；
- threshold、target count、prediction count 与 count ratio 的汇总统计；
- 明确警告：该结果使用测试 target 改变逐帧输出，不得报告为正式推理性能。

## 错误处理与结果保护

- prediction 目录不存在、没有 `*_voxel.npy`、target 目录不存在时立即失败。
- 每个选中 prediction 必须有同 frame ID 的 target；缺失时立即失败，不静默跳过。
- prediction 必须为 `(C,Z,X,Y)` 且至少两个通道；网格必须与 `target_size` 一致。
- `target_threshold` 必须是 `[0,1]` 内有限数；`max_files` 不能为负数。
- `output_dir` 已存在且非空时拒绝运行；不提供覆盖开关，不删除任何已有诊断结果。
- 所有参数验证和帧匹配检查在写第一个输出文件前完成，避免部分结果。

## Mini launcher 兼容

`test/mini-test/inference_minimal.sh` 删除 adaptive 默认变量与参数拼装。若调用环境仍设置
`ADAPTIVE_OCC_FROM_TARGET` 或 `ADAPTIVE_TARGET_THRESHOLD`，脚本在 checkpoint 检查和推理前失败，并提示使用独立诊断脚本，避免旧环境变量被静默忽略。

## 测试设计

### RED 1：正式推理迁移边界

扩展 `test/unit/test_multimodal_inference_interface.py`：

- 旧 adaptive flag 和 `--flag=value` 均返回包含新脚本路径的迁移错误；
- 正常固定阈值参数不触发迁移错误；
- 正式 inference 模块不再暴露 `find_adaptive_occ_threshold`。

### RED 2：独立诊断算法与输出

新增 `test/unit/test_oracle_target_adaptation.py`：

- top-k/`nextafter` 阈值产生预期点数；
- target count 为零时显式记录兼容的 effective match count；
- 临时 prediction/target 输入生成 `(N,4)` 点云、CSV 和 JSON；
- JSON 明确包含 `deployable=false` 和 oracle 协议名；
- target 缺失、shape 错误、非空输出目录均在写入前失败。

### RED 3：mini launcher 迁移边界

扩展 `test/unit/test_mini_scripts_protocol.py`：

- 脚本不再拼装 `--adaptive_occ_from_target`；
- 旧环境变量存在时有独立诊断脚本迁移提示并退出。

## 影响分析

- 训练监督信号不变。
- target 文件及每帧 occupied 体素数量不变。
- 模型输入、输出体素、网格、参数量与 checkpoint 不变。
- 固定阈值推理的预测体素和点云不变。
- 正式 inference CSV 删除一个过去默认为空的 `target_occ_count` 列，属于明确的协议清理。
- 历史 adaptive 结果只能与新 oracle 诊断结果比较，不能与固定阈值部署结果混为同一协议。

## 非目标

- 不修改全局 validation 阈值选择策略。
- 不在本项修复 `sweep_occ_threshold.py` 仍使用旧随机 train/validation 划分的问题；该问题作为 P0-01 后续单独处理。
- 不修改 target 生成、Radar 可见性策略、free/unknown 监督、IR 标定或 checkpoint 链。
- 不运行训练、完整推理或全量评估。

## 完成标准

- 正式推理无法使用 target 数量改变逐帧阈值。
- 旧参数和旧 mini 环境变量得到明确迁移错误，而非静默忽略。
- 独立诊断可从已保存预测生成逐帧 oracle 阈值、CSV、JSON 和 XYZ+intensity 点云。
- 聚焦单元测试、Python 编译、Shell 语法与 `git diff --check` 全部通过。
