# test 目录工作规则

## 1. 适用范围

本文件适用于 `test/` 目录及其所有子目录。

任何涉及 `test/` 的新增、修改、移动、重命名、删除、评估、诊断、消融、可视化或实验结果管理任务，都必须遵守本文件。

## 2. 基本原则

- 默认使用中文说明修改内容。
- 修改代码前先阅读相关调用链、入口脚本和输出路径。
- 不根据文件名直接判断用途，应检查实际代码、调用关系和生成内容。
- 不进行与当前任务无关的大范围重构。
- 不删除数据集、checkpoint、训练日志、评估结果、图片、点云、HTML 或历史实验目录。
- 不自动运行长时间训练、完整数据预处理、完整模型推理或全量评估。
- 运行测试前说明测试范围。
- 优先采用静态检查、导入检查和小规模样本验证。
- 新增脚本前先搜索已有实现，避免重复创建功能相近的文件。
- 不确定文件用途或影响范围时，先停止并说明风险，不得强行移动或删除。

## 3. 推荐目录结构

新增文件必须根据实际用途放入对应目录：

```text
test/
├── AGENTS.md
├── README.md
├── mini-test/
├── evaluation/
│   ├── vae/
│   ├── ldm/
│   └── comparison/
├── diagnostics/
│   ├── alignment/
│   ├── radar/
│   ├── infrared/
│   └── vertical_structure/
├── ablation/
├── visualization/
├── utils/
│   └── legacy/
├── configs/
└── result/
    ├── vae/
    │   ├── evaluation/
    │   ├── reconstruction/
    │   ├── diagnostics/
    │   └── overfit/
    ├── ldm/
    │   ├── evaluation/
    │   ├── vertical_structure/
    │   ├── ablation/
    │   └── visualization/
    ├── comparison/
    │   ├── alignment_check/
    │   └── dataset_protocol_audit_v7/
    ├── archive/
    └── INDEX.md
```

禁止因为暂时无法判断分类而直接把新文件放在 `test/` 根目录。

## 4. 目录用途

### `evaluation/vae/`

用于：

- VAE 重建质量评估；
- VAE IoU、Precision、Recall、F1 等指标计算；
- VAE 重建上限分析；
- VAE checkpoint 对比。

### `evaluation/ldm/`

用于：

- LDM、CD1Step、CD4Step 等生成模型评估；
- 占用预测定量指标计算；
- 单模型或单 checkpoint 的正式评估。

### `evaluation/comparison/`

用于：

- 不同模型之间的统一比较；
- 不同 checkpoint、不同推理步数或不同输出的比较；
- VAE、LDM、CD、原始 Radar、LiDAR target 的联合比较。

### `diagnostics/alignment/`

用于：

- 坐标系检查；
- 标定方向检查；
- 雷达与 LiDAR 对齐；
- 质心偏移检查；
- 体素索引和物理坐标映射检查。

### `diagnostics/radar/`

用于：

- Radar 轴定义；
- Radar 输入通道检查；
- Radar voxel 分布和范围检查；
- Doppler、强度和占用输入诊断。

### `diagnostics/infrared/`

用于：

- 红外条件输入检查；
- 红外投影有效性检查；
- 红外特征是否生效的诊断；
- 红外条件消融前的输入验证。

### `diagnostics/vertical_structure/`

用于：

- 高度分布分析；
- 地面类误生成分析；
- 垂直结构恢复能力诊断；
- 分层占用率和高度区间指标计算。

### `ablation/`

用于：

- 条件移除实验；
- 红外条件消融；
- 后处理参数消融；
- 体素高度、范围、阈值和筛选策略消融；
- 不同模块或输入通道的对比实验。

### `visualization/`

用于：

- 点云可视化；
- 体素可视化；
- 推理结果对比图；
- 交互式 HTML；
- CloudCompare 导出；
- 论文用静态图生成。

### `utils/`

用于多个测试脚本共用的辅助模块，不应作为主要运行入口。

### `utils/legacy/`

用于仍需保留但不再作为正式入口的一次性修复脚本、历史脚本和兼容脚本。

### `configs/`

用于测试和评估专用的 YAML、JSON、TOML 等配置文件。

### `mini-test/`

用于小规模训练、推理和端到端流程验证。

### `result/`

用于保存测试、评估、诊断、消融和可视化输出。

- CD 输出作为所属实验的 `cd/` 子目录保存，不创建无实验归属的 `test/result/cd/` 根目录。
- `*.lock` 必须紧邻其所属实验目录，保持 runner 的锁路径约定。
- `.tmp_*` 只能位于所属实验或明确的 archive 叶目录内；根级临时输出必须在确认归属后归档。
- 未完成、用途不明但必须保留的结果放入 `test/result/archive/`，并在 `INDEX.md` 标注状态和依据。

## 5. 新增文件前的决策流程

创建新文件前必须依次完成：

1. 使用 `rg` 或其他搜索工具查找是否已有相同或相近功能。
2. 检查现有脚本是否可以通过新增命令行参数完成需求。
3. 确认新文件只有一个主要职责。
4. 根据主要职责选择目标目录。
5. 检查输入数据、checkpoint、配置和输出路径。
6. 确定结果目录分类和实验名称。
7. 检查是否需要更新 `test/README.md`。
8. 正式实验完成后检查是否需要更新 `test/result/INDEX.md`。

仅当现有脚本无法合理扩展时，才允许创建新脚本。

禁止为了单次参数变化复制整个脚本并创建以下形式的文件：

```text
test2.py
new_test.py
final_test.py
fix.py
fix_test.py
temp.py
xxx_v2.py
xxx_v3.py
```

参数变化应优先通过以下方式实现：

- 命令行参数；
- YAML 或 JSON 配置；
- 小型公共函数；
- Git 历史。

## 6. 文件命名规则

Python 文件统一使用小写下划线命名，并尽量采用“动作 + 对象”的形式。

推荐：

```text
evaluate_vae_reconstruction.py
evaluate_ldm_vertical_structure.py
compare_voxel_triplets.py
diagnose_radar_alignment.py
diagnose_ir_condition.py
visualize_inference_pointcloud.py
export_pointcloud_for_cloudcompare.py
```

不推荐：

```text
test.py
test2.py
new.py
final.py
fix.py
temp.py
try.py
```

一次性修复脚本确实需要保留时，应放入：

```text
test/utils/legacy/
```

并在文件开头注明：

- 脚本用途；
- 创建原因；
- 是否仍被调用；
- 是否可以在未来删除。

## 7. 脚本职责

每个脚本应尽量只有一个主要职责。

- 评估脚本负责读取结果、计算指标和保存评估数据。
- 诊断脚本负责定位数据、坐标、输入条件或模型输出问题。
- 可视化脚本负责生成图片、点云或 HTML。
- 消融脚本负责比较明确的模块、输入或参数变化。
- 公共函数应放入 `utils/`，避免在多个脚本中复制。
- 端到端训练、推理、评估和可视化只允许出现在明确的 `mini-test` 流程中。

除非用户明确要求，不要在单个新脚本中同时执行：

- 数据预处理；
- 完整训练；
- 模型推理；
- 指标计算；
- 结果可视化。

## 8. 路径处理规则

- 不写死 `/home/zxj/` 等绝对用户路径。
- 优先使用 `pathlib.Path`。
- 项目根目录应根据脚本位置、命令行参数或统一配置推导。
- 数据集目录、checkpoint、结果目录应支持命令行参数。
- 默认测试输出应位于 `test/result/` 的对应分类中。
- 不允许在没有兼容处理的情况下修改已有 `argparse` 参数名称和含义。
- 移动文件后必须同步检查 Python import、Shell 调用、配置文件和文档命令。
- 不应默认覆盖已有实验结果。
- 可能覆盖结果时，应增加明确的 `--overwrite` 参数，且默认关闭。
- 新建输出目录时使用 `mkdir(parents=True, exist_ok=True)`。

推荐写法：

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "test" / "result"
```

实际 `parents` 层级必须根据脚本所在目录确认，不得机械复制。

## 9. result 目录规则

新增实验结果必须根据模型和用途分类。

```text
test/result/
├── vae/
│   ├── evaluation/
│   ├── reconstruction/
│   ├── diagnostics/
│   └── overfit/
├── ldm/
│   ├── evaluation/
│   ├── vertical_structure/
│   ├── ablation/
│   └── visualization/
├── comparison/
└── archive/
```

历史实验目录的叶子名称应尽量保留，例如：

```text
ldm_near40_500_vertical_v4
ldm_near40_500_z64_v9a_top_full
vae_overfit_32_vertical_diagnostic
```

允许调整其上级分类，但不要轻易修改原始叶子目录名称，以免丢失实验历史和破坏已有引用。

对无法确认价值的历史结果：

- 不删除；
- 不覆盖；
- 不重命名；
- 必要时移动到 `archive/`；
- 在索引中标记为历史结果。

## 10. 实验命名规则

正式实验目录建议使用：

```text
<model>_<data-scope>_<purpose>_<variant>
```

例如：

```text
ldm_near40_500_vertical_structure_z64
ldm_loop3_ground_filter_ablation
vae_loop3_reconstruction_evaluation
cd1_loop3_inference_comparison
```

实验名称应尽量体现：

- 模型；
- 数据范围或场景；
- 实验目的；
- 关键变体。

不再仅使用 `v2`、`v3`、`final`、`new` 作为主要含义。

确实需要版本号时，应同时保留有意义的描述，并在索引中解释差异。

## 11. 实验结果记录

新的正式实验结果目录应尽可能包含：

```text
config.yaml
metrics.json
command.txt
README.md
```

至少记录：

- 模型名称；
- checkpoint 路径或标识；
- 数据集和场景；
- 样本数量；
- 主要参数；
- 执行命令；
- 生成脚本；
- 实验目的；
- 结果状态；
- 是否为当前推荐结果；
- 与前一版本的主要差异。

不得复制大型 checkpoint 到 `test/result/`，除非用户明确要求。

## 12. 文档同步规则

出现以下情况时必须更新 `test/README.md`：

- 新增主要目录；
- 新增正式评估入口；
- 新增正式诊断入口；
- 修改常用运行命令；
- 修改默认输出路径；
- 替换推荐脚本；
- 移动已有脚本。

正式实验结果完成后，应更新：

```text
test/result/INDEX.md
```

索引建议包含：

| 实验目录 | 模型 | 场景 | 实验目的 | 主要参数 | checkpoint | 生成脚本 | 状态 | 推荐结果 | 备注 |
| -------- | ---- | ---- | -------- | -------- | ---------- | -------- | ---- | -------- | ---- |

临时调试输出不必立即加入索引，但不得与正式实验混放。

## 13. 修改现有文件前的检查

移动、重命名或删除现有文件前，必须检查所有引用。

建议执行：

```bash
rg "旧文件名|旧相对路径" .
rg "test/result|result/" test
rg "python .*\.py" test --glob "*.sh"
rg "subprocess|os\.system|Popen" test --glob "*.py"
```

必须检查：

- Python import；
- Shell 脚本调用；
- YAML、JSON 和 TOML 路径；
- README 中的命令；
- checkpoint 路径；
- 默认输出路径；
- `subprocess` 调用；
- `os.system` 调用；
- 其他脚本是否按字符串引用目标文件。

无法确认调用关系时，不得移动。

## 14. 现有根目录文件的处理原则

对当前已经存在于 `test/` 根目录的脚本，整理时应先读取代码并按实际用途分类。

典型分类原则：

- `alignment_sanity_check.py`：优先考虑 `diagnostics/alignment/`。
- `check_IoU_vae.py`：优先考虑 `evaluation/vae/`。
- `check_radar_axis_conventions.py`：优先考虑 `diagnostics/radar/`。
- `compare_voxel_triplets.py`：根据主要用途放入 `evaluation/comparison/` 或 `visualization/`。
- `diagnose_ir_condition_ablation.py`：根据主要用途放入 `diagnostics/infrared/` 或 `ablation/`。
- `evaluate_ldm_vertical_structure.py`：优先考虑 `evaluation/ldm/`；若主要用于问题定位，可放入 `diagnostics/vertical_structure/`。
- `fix_test.py`：必须检查实际内容。一次性脚本优先放入 `utils/legacy/`，不得直接删除。

以上只是候选位置，最终分类必须以代码实际职责为准。

## 15. 最小验证要求

一般修改完成后优先执行：

```bash
conda run -n Radar-Diffusion python -m compileall test
```

对于修改过的命令行脚本，可以执行：

```bash
conda run -n Radar-Diffusion python <脚本路径> --help
```

对于公共模块，可以进行最小导入检查：

```bash
conda run -n Radar-Diffusion python -c "import <模块路径>"
```

对于路径调整，应检查：

```bash
git status --short
git diff --stat
```

除非用户明确要求，不运行：

- 长时间训练；
- 全量数据预处理；
- 完整 `loop3` 推理；
- 大规模评估；
- 会覆盖已有结果的命令。

脚本不支持 `--help` 时，不得直接运行耗时流程，只进行静态检查或安全导入检查。

## 16. 整理任务的执行顺序

目录整理必须按以下顺序进行：

1. 审计现有目录和脚本用途。
2. 搜索调用关系和硬编码路径。
3. 输出建议目录树。
4. 输出“当前路径 → 建议路径”迁移表。
5. 说明迁移风险。
6. 等待用户确认。
7. 使用 `git mv` 移动受 Git 管理的文件。
8. 更新路径引用。
9. 更新 README 和结果索引。
10. 执行最小验证。
11. 输出 `git status` 和 `git diff --stat`。

未经用户确认，不得直接执行大规模迁移。

## 17. 禁止事项

禁止执行以下操作：

- 删除 checkpoint；
- 删除训练日志；
- 删除数据集；
- 删除历史评估结果；
- 删除图片、点云或 HTML；
- 擅自覆盖已有实验结果；
- 自动启动长时间训练；
- 自动执行完整推理；
- 大范围重构模型代码；
- 大范围修改数据加载逻辑；
- 未检查调用链就移动脚本；
- 仅根据文件名判断文件用途；
- 在 `test/` 根目录随意创建新脚本；
- 创建含义不清的 `final`、`new`、`fix`、`temp` 文件；
- 为参数变化复制整份脚本；
- 修改与当前任务无关的文件。

## 18. 新增文件后的报告要求

在 `test/` 中新增文件后，最终报告必须说明：

- 为什么需要新增该文件；
- 是否检查过已有相同或相近功能；
- 为什么不能直接扩展现有脚本；
- 文件为什么放在当前目录；
- 输入是什么；
- 输出保存到哪里；
- 是否更新 `test/README.md`；
- 是否更新 `test/result/INDEX.md`；
- 执行了哪些小范围验证；
- 是否存在未解决的路径兼容问题。

## 19. 修改完成后的输出格式

完成任务后应至少输出：

```text
1. 修改目的
2. 新增或修改的文件
3. 文件分类依据
4. 路径引用修改
5. 验证范围
6. 验证结果
7. 未解决问题
8. git status
9. git diff --stat
```

如果没有执行训练或完整推理，应明确写明“未运行长时间训练或完整推理”。

## 20. 与项目根规则的关系

本文件是 `test/` 目录的专用规则。

若项目根目录的 `AGENTS.md` 与本文件同时存在：

- 项目通用规则继续生效；
- 涉及 `test/` 的任务优先遵守本文件中的更具体规则；
- 若规则冲突，应优先采用更安全、影响范围更小的方案；
- 无法判断时应停止并向用户说明。
