# P0-06 Oracle Target 自适应诊断隔离 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从正式推理移除逐帧 target 数量自适应阈值，并提供独立离线诊断脚本生成 oracle CSV、JSON 和 XYZ+intensity 点云。

**Architecture:** 正式 `inference.py` 仅使用固定阈值，并在 argparse 注册前识别旧参数以给出迁移错误。独立诊断脚本只读取已保存预测体素和 target，复用现有 target 重采样与点云转换协议，不执行模型推理。

**Tech Stack:** Python 3.8、NumPy、PyTorch、argparse、csv/json、unittest、Bash、Conda `Radar-Diffusion`。

## Global Constraints

- 默认中文功能注释；新增 Python 文件必须包含中文文件头注释。
- 测试放在 `test/unit/`，诊断脚本放在 `test/diagnostics/occupancy/`。
- 不运行训练、完整推理、全量评估或数据预处理。
- 不删除、覆盖数据集、checkpoint、日志或实验结果。
- 新诊断输出目录非空时拒绝执行，不提供覆盖开关。
- 不修改 `sweep_occ_threshold.py` 的旧随机 validation 协议；该问题单独处理。
- 不改变监督 target、体素数量、模型结构、checkpoint 或固定阈值输出。
- 本轮不暂存、不提交；保留全部既有脏工作区修改。

---

## File Structure and Responsibilities

- Modify: `diffusion_consistency_radar/scripts/inference.py`
  - 删除 oracle 能力，保留固定阈值与正常 target/LiDAR 对比。
  - 提供 `reject_removed_oracle_arguments(argv) -> None` 迁移检查。
- Modify: `test/mini-test/inference_minimal.sh`
  - 删除 adaptive 参数拼装，旧环境变量改为 fail-fast 迁移提示。
- Create: `test/diagnostics/occupancy/diagnose_oracle_target_adaptation.py`
  - 独立读取已保存体素、匹配 target、计算 oracle 阈值、保存点云和报告。
- Modify: `test/unit/test_multimodal_inference_interface.py`
  - 验证正式推理旧参数迁移边界。
- Modify: `test/unit/test_mini_scripts_protocol.py`
  - 验证 mini launcher 旧环境变量 fail-fast。
- Create: `test/unit/test_oracle_target_adaptation.py`
  - 验证算法、输出协议和错误保护。
- Modify: `TODO/task_plan.md`, `TODO/findings.md`, `TODO/progress.md`
  - 记录 RED/GREEN、协议影响和验证结果。

---

### Task 1: RED — 锁定正式推理与 mini launcher 的迁移边界

**Interfaces:**

- Consumes: 当前 `inference.py` 旧 adaptive CLI 与 mini 环境变量。
- Produces: 对 `reject_removed_oracle_arguments(argv)` 和 shell fail-fast 行为的失败测试。

- [ ] **Step 1: 在 inference 接口测试中写 RED**

向 `MultimodalInferenceInterfaceTest` 添加：

```python
    def test_removed_oracle_arguments_report_diagnostic_migration(self):
        from diffusion_consistency_radar.scripts import inference

        reject = getattr(inference, "reject_removed_oracle_arguments", None)
        self.assertIsNotNone(reject, "尚未实现旧 oracle 参数迁移检查")
        for argv in (
            ["--adaptive_occ_from_target"],
            ["--adaptive_target_threshold", "0.1"],
            ["--adaptive_target_threshold=0.1"],
        ):
            with self.subTest(argv=argv):
                with self.assertRaisesRegex(
                    ValueError,
                    "diagnose_oracle_target_adaptation.py",
                ):
                    reject(argv)

    def test_fixed_threshold_arguments_do_not_trigger_oracle_migration(self):
        from diffusion_consistency_radar.scripts import inference

        reject = getattr(inference, "reject_removed_oracle_arguments", None)
        self.assertIsNotNone(reject, "尚未实现旧 oracle 参数迁移检查")
        reject(["--occ_threshold", "0.5", "--compare_with_target"])
```

- [ ] **Step 2: 在 mini 协议测试中写 RED**

为 `test/unit/test_mini_scripts_protocol.py` 增加 `subprocess` 导入和：

```python
    def test_inference_script_rejects_removed_oracle_environment(self):
        script_path = os.path.join(ROOT, "test/mini-test/inference_minimal.sh")
        env = os.environ.copy()
        env["ADAPTIVE_OCC_FROM_TARGET"] = "1"
        result = subprocess.run(
            ["bash", script_path, "ldm"],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("diagnose_oracle_target_adaptation.py", result.stdout)
```

- [ ] **Step 3: 运行 RED**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_multimodal_inference_interface.py -v
conda run -n Radar-Diffusion python test/unit/test_mini_scripts_protocol.py -v
```

Expected: inference 测试因迁移检查函数不存在而失败；mini 测试因没有迁移提示而失败；其余历史测试保持通过。

---

### Task 2: GREEN — 从正式推理和 mini launcher 移除 oracle 能力

**Interfaces:**

- Consumes: Task 1 的迁移边界测试。
- Produces: `reject_removed_oracle_arguments(argv) -> None`，固定阈值正式推理。

- [ ] **Step 1: 实现旧参数迁移检查**

在 `inference.py` 参数解析辅助区域添加：

```python
REMOVED_ORACLE_ARGUMENTS = (
    "--adaptive_occ_from_target",
    "--adaptive_target_threshold",
)


def reject_removed_oracle_arguments(argv) -> None:
    """拒绝已从正式推理移除的 target 自适应参数。"""
    for token in argv:
        text = str(token)
        if any(text == name or text.startswith(f"{name}=") for name in REMOVED_ORACLE_ARGUMENTS):
            raise ValueError(
                f"{text.split('=', 1)[0]} 已从正式推理移除；"
                "请先用固定 --occ_threshold 保存预测体素，再运行 "
                "test/diagnostics/occupancy/diagnose_oracle_target_adaptation.py"
            )
```

在 `args = parser.parse_args()` 前执行：

```python
    try:
        reject_removed_oracle_arguments(sys.argv[1:])
    except ValueError as exc:
        parser.error(str(exc))
```

- [ ] **Step 2: 删除正式 adaptive 数据流**

从 `inference.py` 删除：

```text
--adaptive_occ_from_target
--adaptive_target_threshold
find_adaptive_occ_threshold()
load_target_occ_resized()
adaptive 参数目录校验分支
adaptive runtime log 字段
逐帧 target_occ_count 初始化与阈值反推分支
inference_metrics.csv 的 target_occ_count 表头与行值
```

将 target 目录需求改为：

```python
        needs_target_voxel = args.compare_with_target
```

保持：

```python
            effective_occ_threshold = float(args.occ_threshold)
```

- [ ] **Step 3: 修改 mini launcher**

在 checkpoint 检查之前加入：

```bash
if [[ -n "${ADAPTIVE_OCC_FROM_TARGET+x}" || -n "${ADAPTIVE_TARGET_THRESHOLD+x}" ]]; then
  echo "Error: adaptive target threshold 已从推理入口移除；请运行 test/diagnostics/occupancy/diagnose_oracle_target_adaptation.py"
  exit 2
fi
```

删除 adaptive 默认变量、setup echo、`EXTRA_ADAPTIVE_ARGS` 和命令行展开。

- [ ] **Step 4: 运行 GREEN**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_multimodal_inference_interface.py -v
conda run -n Radar-Diffusion python test/unit/test_mini_scripts_protocol.py -v
bash -n test/mini-test/inference_minimal.sh
```

Expected: 两个测试文件全通过，shell 语法通过。

---

### Task 3: RED — 锁定独立 oracle 诊断协议

**Interfaces:**

- Consumes: `<frame>_voxel.npy`、`<frame>.npy/.npz` target。
- Produces: `find_oracle_occ_threshold()`、`run_diagnostic()` 与 CLI 输出契约失败测试。

- [ ] **Step 1: 新增测试文件和延迟模块断言**

创建 `test/unit/test_oracle_target_adaptation.py`，文件头说明“验证 oracle target 数量匹配诊断的阈值、点云与报告协议”。使用：

```python
MODULE_PATH = os.path.join(
    PROJECT_ROOT,
    "test/diagnostics/occupancy/diagnose_oracle_target_adaptation.py",
)


def load_oracle_module(test_case):
    if not os.path.isfile(MODULE_PATH):
        test_case.fail("尚未创建独立 oracle target 诊断脚本")
    spec = importlib.util.spec_from_file_location(
        "diagnose_oracle_target_adaptation",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

测试阈值：

```python
    def test_oracle_threshold_matches_requested_topk(self):
        module = load_oracle_module(self)
        pred = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
        threshold, effective_count = module.find_oracle_occ_threshold(pred, 2)
        self.assertEqual(effective_count, 2)
        self.assertEqual(int(np.count_nonzero(pred > threshold)), 2)

    def test_zero_target_count_preserves_legacy_minimum_one(self):
        module = load_oracle_module(self)
        pred = np.asarray([0.9, 0.1], dtype=np.float32)
        threshold, effective_count = module.find_oracle_occ_threshold(pred, 0)
        self.assertEqual(effective_count, 1)
        self.assertEqual(int(np.count_nonzero(pred > threshold)), 1)
```

临时目录端到端测试构造 `(C,Z,X,Y)=(2,1,4,1)` prediction 和
`(X,Y,Z,C)=(4,1,1,4)` target，调用：

```python
module.run_diagnostic(
    pred_voxel_dir=str(pred_dir),
    target_voxel_dir=str(target_dir),
    output_dir=str(output_dir),
    target_threshold=0.5,
    source_pc_range=(0, 0, 0, 4, 1, 1),
    model_pc_range=(0, 0, 0, 4, 1, 1),
    target_size=(1, 4, 1),
    voxel_size=None,
    max_files=0,
)
```

断言 `<frame>_oracle_pcl.npy` 形状为 `(2,4)`，CSV 包含一行，JSON 包含：

```python
{"protocol": "oracle_target_count_matching", "deployable": False}
```

另加 target 缺失、prediction shape 错误和非空输出目录测试，均断言未新增输出文件。

- [ ] **Step 2: 运行 RED**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_oracle_target_adaptation.py -v
```

Expected: 测试因诊断模块不存在而以明确 assertion failure 失败。

---

### Task 4: GREEN — 实现独立 oracle 诊断脚本

**Interfaces:**

- Produces:
  - `find_oracle_occ_threshold(pred_occ: np.ndarray, target_count: int) -> Tuple[float, int]`
  - `run_diagnostic(...)->Dict[str, object]`
  - `main(argv=None) -> None`

- [ ] **Step 1: 创建目录和带中文文件头的脚本**

实现 `find_oracle_occ_threshold()`，沿用原 top-k 意图，使用 prediction dtype 的
`np.nextafter` 前驱，并返回 `(threshold, effective_match_count)`；不得先转成 Python float
再求 float64 前驱，否则与 float32 prediction 比较会退化为 `k-1` 点。

- [ ] **Step 2: 实现严格预检**

`run_diagnostic()` 在创建输出目录前完成：

```text
输入目录存在
max_files >= 0
target_threshold 为 [0,1] 有限数
output_dir 不存在或为空
预测清单非空
所有选中 prediction 都存在 target
所有 prediction shape 为 C,Z,X,Y，C>=2 且空间尺寸等于 target_size
```

- [ ] **Step 3: 实现逐帧输出**

复用：

```python
from diffusion_consistency_radar.scripts.inference import voxel_to_pointcloud
from diffusion_consistency_radar.scripts.sweep_occ_threshold import load_target_occ_resized
```

每帧计算 target count、oracle threshold、点云并保存
`<frame>_oracle_pcl.npy`。写入设计规格定义的 CSV 列。

- [ ] **Step 4: 实现 JSON 汇总与 CLI**

JSON 固定包含：

```python
{
    "protocol": "oracle_target_count_matching",
    "deployable": False,
    "warning": "该结果使用测试 target 改变逐帧输出，不得作为正式推理性能。",
}
```

CLI 使用设计规格的参数名和默认值；不提供 overwrite 参数。

- [ ] **Step 5: 运行 GREEN**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_oracle_target_adaptation.py -v
```

Expected: 全部算法、输出和错误保护测试通过。

---

### Task 5: Final Verification and Project Records

- [ ] **Step 1: 运行聚焦回归**

```bash
conda run -n Radar-Diffusion python test/unit/test_multimodal_inference_interface.py -v
conda run -n Radar-Diffusion python test/unit/test_mini_scripts_protocol.py -v
conda run -n Radar-Diffusion python test/unit/test_oracle_target_adaptation.py -v
```

- [ ] **Step 2: 运行静态验证**

```bash
conda run -n Radar-Diffusion python -m py_compile \
  diffusion_consistency_radar/scripts/inference.py \
  test/diagnostics/occupancy/diagnose_oracle_target_adaptation.py \
  test/unit/test_multimodal_inference_interface.py \
  test/unit/test_mini_scripts_protocol.py \
  test/unit/test_oracle_target_adaptation.py
bash -n test/mini-test/inference_minimal.sh
git diff --check
```

- [ ] **Step 3: 审计旧路径残留**

```bash
rg -n "adaptive_occ_from_target|adaptive_target_threshold|find_adaptive_occ_threshold|target_occ_count" \
  diffusion_consistency_radar/scripts/inference.py \
  test/mini-test/inference_minimal.sh
```

Expected: 只允许迁移检测常量/错误提示命中旧参数名；不允许 parser 注册、运行分支、日志或 CSV adaptive 字段残留。

- [ ] **Step 4: 更新三份 TODO**

记录 RED/GREEN 命令与结果，并明确：

```text
监督 target 不变
每帧 occupied 体素数不变
固定阈值预测体素/点云不变
正式 CSV 删除历史空 target_occ_count 列
oracle 点云与报告明确不可部署
未运行训练、预处理、完整推理或全量评估
P0-01 阈值扫描随机切分仍待单独修复
```

## Completion Criteria

- 正式 inference 不能使用 target 数量改变输出阈值。
- 旧 CLI 和 mini 环境变量返回明确的新脚本迁移提示。
- 独立脚本保存 oracle 点云、CSV 和 JSON，并拒绝缺 target、错误 shape 与非空输出目录。
- 三个聚焦测试文件、Python 编译、Shell 语法、旧路径审计和 `git diff --check` 通过。
- 暂存区和提交历史不变。
