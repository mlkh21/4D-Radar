# Threshold Sweep Temporal Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让占用阈值扫描使用连续前缀 train / 连续后缀 validation，并彻底删除旧 `split_seed` 接口。

**Architecture:** 在阈值扫描脚本本地把 seeded `torch.randperm()` 替换为有序切片，避免导入体量较大的训练入口或扩大为公共模块重构。argparse 解析前单独识别已删除参数，JSON 用显式切分协议字段记录新语义。

**Tech Stack:** Python 3、argparse、NumPy、PyTorch、unittest。

## Global Constraints

- 只修改 `diffusion_consistency_radar/scripts/sweep_occ_threshold.py`、`test/unit/test_occ_threshold_grid_protocol.py`、本计划和三份 TODO 记录。
- 不修改 `unified_train.py`、Dataset、推理入口、checkpoint、数据或历史结果。
- 新代码功能注释使用中文。
- 先观察每组新契约测试 RED，再写最小 GREEN。
- 不运行训练、完整推理或全量阈值扫描。
- 当前工作区包含用户和历史未提交改动；不暂存、不提交，不覆盖无关差异。

---

### Task 1: 时间块成员选择

**Files:**
- Modify: `test/unit/test_occ_threshold_grid_protocol.py:156-201`
- Modify: `diffusion_consistency_radar/scripts/sweep_occ_threshold.py:98-157`

**Interfaces:**
- Produces: `select_evaluation_files(files: Sequence[str], evaluation_split: str, train_split: float) -> List[str]`
- Produces: `prepare_evaluation_files(files: Sequence[str], evaluation_split: str, train_split: float, max_files: int) -> List[str]`
- Preserves: `evaluation_split="all"` 返回原输入顺序；`max_files` 在划分后应用。

- [x] **Step 1: 把既有随机契约测试改为时间块 RED**

将随机 validation 测试改成：

```python
def test_validation_split_returns_temporal_suffix(self):
    files = [f"{index:06d}_voxel.npy" for index in range(10)]

    selected = select_evaluation_files(files, "validation", 0.8)

    self.assertEqual(selected, files[8:])

def test_train_split_returns_temporal_prefix(self):
    files = [f"{index:06d}_voxel.npy" for index in range(10)]

    selected = select_evaluation_files(files, "train", 0.8)

    self.assertEqual(selected, files[:8])
```

把同一区域内其他调用的第四个 seed 参数删除：

```python
select_evaluation_files(files, "validation", 0.8)

prepare_evaluation_files(
    files,
    evaluation_split="validation",
    train_split=0.8,
    max_files=1,
)
```

- [x] **Step 2: 运行测试并确认按预期失败**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_occ_threshold_grid_protocol.py -v
```

Expected: 新时间块测试因旧函数仍要求 `split_seed` 而 ERROR；其余既有测试不应出现新的无关失败。

- [x] **Step 3: 实现最小有序切片**

将两个函数签名和选择逻辑改为：

```python
def select_evaluation_files(
    files: Sequence[str],
    evaluation_split: str,
    train_split: float,
) -> List[str]:
    """按正式训练的连续时间块协议选择评估文件。"""
    ordered = list(files)
    if evaluation_split == "all":
        return ordered
    frame_ids = []
    suffix = "_voxel.npy"
    for filename in ordered:
        frame_id = filename[:-len(suffix)] if filename.endswith(suffix) else ""
        if not frame_id.isdigit():
            raise ValueError(
                "train/validation 划分要求预测 frame ID 为纯数字；"
                "非数字命名请使用 --evaluation_split all"
            )
        frame_ids.append(int(frame_id))
    if any(
        current != previous + 1
        for previous, current in zip(frame_ids, frame_ids[1:])
    ):
        raise ValueError(
            "train/validation 划分要求排序后的预测 frame ID 严格连续；"
            "检测到缺帧，请补齐预测或使用 --evaluation_split all"
        )
    if len(ordered) < 2:
        raise ValueError("训练/验证划分至少需要 2 个样本")
    if not 0.0 < float(train_split) < 1.0:
        raise ValueError("train_split 必须严格位于 (0, 1)")
    train_size = int(len(ordered) * float(train_split))
    if train_size <= 0 or train_size >= len(ordered):
        raise ValueError(
            f"train_split={train_split} 导致空划分："
            f"dataset_size={len(ordered)}, train_size={train_size}"
        )
    selected = (
        ordered[:train_size]
        if evaluation_split == "train"
        else ordered[train_size:]
    )
    return selected


def prepare_evaluation_files(
    files: Sequence[str],
    evaluation_split: str,
    train_split: float,
    max_files: int,
) -> List[str]:
    """先在完整预测清单上复现时间块划分，再限制实际评估帧数。"""
    selected = select_evaluation_files(
        files,
        evaluation_split=evaluation_split,
        train_split=train_split,
    )
    if int(max_files) > 0:
        return selected[: int(max_files)]
    return selected
```

删除该区域创建 `torch.Generator`、调用 `torch.randperm()` 和按随机 indices 取文件的代码。`torch` import 仍由 target resize/device 使用，不删除。

同时从 `main()` 调用 `prepare_evaluation_files()` 的关键字参数中删除：

```python
split_seed=args.split_seed,
```

argparse 参数暂留到 Task 2，使 Task 1 只完成内部成员协议迁移。

- [x] **Step 4: 运行测试并确认 GREEN**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_occ_threshold_grid_protocol.py -v
```

Expected: 全部测试 PASS；validation 为 `files[8:]`，train 为 `files[:8]`。

---

### Task 2: 删除 CLI seed 并写入新协议元数据

**Files:**
- Modify: `test/unit/test_occ_threshold_grid_protocol.py:1-25, 470-535`
- Modify: `diffusion_consistency_radar/scripts/sweep_occ_threshold.py:560-615, 858-882`

**Interfaces:**
- Produces: `reject_removed_split_seed_arguments(argv: Sequence[str]) -> None`
- Produces: JSON `split_protocol="temporal_block_prefix_train_suffix_validation"`
- Removes: argparse `--split_seed` 和 JSON `split_seed`。

- [x] **Step 1: 写旧参数迁移与 JSON schema RED 测试**

在测试文件导入 `io`。迁移契约通过真实 `main()` 入口验证，避免 RED 阶段因尚不存在的 helper 产生模块导入错误。新增：

```python
def test_removed_split_seed_argument_reports_temporal_migration(self):
    for argument in ("--split_seed", "--split_seed=42"):
        with self.subTest(argument=argument):
            with mock.patch("sys.argv", ["sweep_occ_threshold.py", argument]), mock.patch(
                "sys.stderr", new_callable=io.StringIO
            ) as stderr, self.assertRaises(SystemExit) as raised:
                main()

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("时间块", stderr.getvalue())
            self.assertIn("删除 --split_seed", stderr.getvalue())
```

在现有 `test_main_writes_constraint_json_and_constraint_changes_threshold` 的 JSON 断言中加入：

```python
self.assertEqual(
    constrained["split_protocol"],
    "temporal_block_prefix_train_suffix_validation",
)
self.assertNotIn("split_seed", constrained)
```

- [x] **Step 2: 运行测试并确认按预期失败**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_occ_threshold_grid_protocol.py -v
```

Expected: 旧参数错误不含迁移说明，JSON 缺少 `split_protocol` 且仍包含 `split_seed`。

- [x] **Step 3: 实现 CLI 删除、迁移错误和 JSON schema**

在参数解析逻辑之前新增：

```python
REMOVED_SPLIT_ARGUMENT = "--split_seed"
SPLIT_PROTOCOL = "temporal_block_prefix_train_suffix_validation"


def reject_removed_split_seed_arguments(argv: Sequence[str]) -> None:
    """拒绝旧随机切分参数，并提示迁移到固定时间块协议。"""
    for argument in argv:
        if argument == REMOVED_SPLIT_ARGUMENT or argument.startswith(
            f"{REMOVED_SPLIT_ARGUMENT}="
        ):
            raise ValueError(
                "阈值扫描已改为固定时间块划分；请删除 --split_seed，"
                "并确保输入是训练场景的完整连续预测清单"
            )
```

创建 parser 后、调用 `parse_args()` 前执行：

```python
try:
    reject_removed_split_seed_arguments(sys.argv[1:])
except ValueError as exc:
    parser.error(str(exc))
args = parser.parse_args()
```

同时：

- 删除 `parser.add_argument("--split_seed", type=int, default=42)`；
- 将 `--evaluation_split` help 改为“按连续时间块选择训练场景的评估子集；validation 只能用于阈值标定”；
- JSON 删除 `"split_seed": int(args.split_seed)`；
- JSON 增加 `"split_protocol": SPLIT_PROTOCOL`。

- [x] **Step 4: 运行测试并确认 GREEN**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_occ_threshold_grid_protocol.py -v
```

Expected: 全部测试 PASS；两种旧参数写法均 exit 2 并包含迁移说明，新 JSON 只有协议字段。

---

### Task 3: 最终聚焦验证与记录

**Files:**
- Modify: `TODO/findings.md`
- Modify: `TODO/task_plan.md`
- Modify: `TODO/progress.md`

**Interfaces:**
- Verifies: 新切分 API、CLI 帮助、旧参数迁移、JSON schema 和补丁格式。

- [x] **Step 1: 运行聚焦回归和语法检查**

先向用户说明范围，然后运行：

```bash
conda run -n Radar-Diffusion python test/unit/test_occ_threshold_grid_protocol.py -v
conda run -n Radar-Diffusion python -m py_compile \
  diffusion_consistency_radar/scripts/sweep_occ_threshold.py \
  test/unit/test_occ_threshold_grid_protocol.py
git diff --check
```

Expected: 单元测试全部 PASS，编译和差异检查 exit 0。

- [x] **Step 2: 验证真实 CLI 表面**

Run:

```bash
conda run -n Radar-Diffusion python diffusion_consistency_radar/scripts/sweep_occ_threshold.py --split_seed=42
```

Expected: exit 2；提示删除 `--split_seed`，且先于 `--pred_voxel_dir` 等必填参数错误。

Run:

```bash
conda run -n Radar-Diffusion python diffusion_consistency_radar/scripts/sweep_occ_threshold.py --help
```

Expected: exit 0；帮助中不包含 `--split_seed`，并说明 validation 只用于训练场景的阈值标定。

- [x] **Step 3: 更新三份 TODO 记录**

记录：

- RED/GREEN 测试数量和预期失败原因；
- 新旧 CLI/JSON schema 差异；
- validation 样本数量不变、成员改为连续后缀；
- 监督、target、网格和每帧体素内容不变；
- 历史指标和推荐阈值不可直接比较；
- `loop3` 场景身份需要后续 manifest 机器校验；
- 未运行训练、完整推理、全量扫描，未覆盖历史结果；
- 暂存区仍为空且未提交。

- [x] **Step 4: 检查计划完成状态**

确认 Task 1 至 Task 3 全部完成，将 `TODO/task_plan.md` 对应续修状态标为完成，并保留下一项 dataset manifest 为待处理。
