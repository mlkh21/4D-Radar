# Strict Dataset Manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为每个预处理场景生成 per-frame SHA-256 manifest，并让正式训练和推理 launcher 对缺失、混用、symlink 或内容不一致的数据 fail-closed。

**Architecture:** 纯标准库核心位于包根，负责扫描、内容寻址、原子发布和完整验证；轻量 CLI 供预处理与 shell launcher 复用。通用 Dataset 和 mini/诊断入口保持兼容，严格性仅接入正式入口。

**Tech Stack:** Python 3 标准库（argparse、hashlib、json、os、re、tempfile）、Bash、unittest。

## Global Constraints

- 不删除、覆盖、重链或补签现有 Data、checkpoint、日志和结果。
- 不修改通用 Dataset 默认行为，不切换正式推理数据根，不启用真实 IR。
- manifest v1 必须拒绝目录级/文件级 symlink、缺 policy、模态错帧和内容 hash 不一致。
- manifest 不记录绝对路径、mtime 或时间戳，正式文件已存在时拒绝覆盖。
- 新 Python 文件必须有中文文件头和功能注释。
- 测试只使用临时小文件；不运行预处理、训练、推理或全量真实数据 hashing。
- 当前工作区包含未提交历史改动；不暂存、不提交，不覆盖无关文件。

---

### Task 1: 内容级 manifest 核心

**Files:**
- Create: `diffusion_consistency_radar/dataset_manifest.py`
- Create: `test/unit/test_dataset_manifest_protocol.py`

**Interfaces:**
- Produces: `DatasetManifestError(ValueError)`
- Produces: `build_scene_manifest(scene_dir: str, scene: str, expected_frame_count: int, provenance_paths: Mapping[str, str]) -> dict`
- Produces: `write_scene_manifest_atomic(scene_dir: str, scene: str, expected_frame_count: int, provenance_paths: Mapping[str, str]) -> str`
- Produces: `validate_scene_manifest(scene_dir: str, expected_scene: str) -> dict`

- [x] **Step 1: 写核心协议 RED 测试**

新测试文件使用下列固定 helper 创建两帧场景：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证严格 dataset manifest 的生成、内容寻址和拒绝策略。"""

import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_manifest_module():
    try:
        return importlib.import_module("diffusion_consistency_radar.dataset_manifest")
    except ModuleNotFoundError as exc:
        raise AssertionError("dataset manifest 模块尚未实现") from exc


def write_bytes(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(payload)


def create_scene(root, scene="garden", frame_count=2):
    scene_dir = os.path.join(root, scene)
    os.makedirs(scene_dir)
    with open(os.path.join(scene_dir, "preprocess_policy.json"), "w", encoding="utf-8") as handle:
        json.dump({"source_scene": scene, "frames_written": frame_count}, handle)
    for index in range(frame_count):
        frame_id = f"{index:06d}"
        write_bytes(os.path.join(scene_dir, "radar_voxel", f"{frame_id}.npz"), b"radar" + bytes([index]))
        write_bytes(os.path.join(scene_dir, "lidar_voxel", f"{frame_id}.npz"), b"lidar" + bytes([index]))
        write_bytes(os.path.join(scene_dir, "target_voxel", f"{frame_id}.npz"), b"target" + bytes([index]))
        write_bytes(os.path.join(scene_dir, "ir_image", f"{frame_id}_ir.npy"), b"ir" + bytes([index]))
    provenance = {}
    for key in ("preprocess_script", "calibration", "radar_index", "lidar_index"):
        path = os.path.join(root, f"{key}.txt")
        write_bytes(path, key.encode("utf-8"))
        provenance[key] = path
    return scene_dir, provenance
```

新增以下独立测试：

```python
class DatasetManifestProtocolTest(unittest.TestCase):
    def test_valid_manifest_round_trip_is_portable_and_content_addressed(self):
        module = load_manifest_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_dir, provenance = create_scene(temp_dir)
            path = module.write_scene_manifest_atomic(scene_dir, "garden", 2, provenance)
            manifest = module.validate_scene_manifest(scene_dir, "garden")
            copied = os.path.join(temp_dir, "copied", "garden")
            shutil.copytree(scene_dir, copied)
            copied_manifest = module.validate_scene_manifest(copied, "garden")
            serialized = json.dumps(manifest, sort_keys=True)
        self.assertEqual(os.path.basename(path), "dataset_manifest.json")
        self.assertEqual(manifest["content_sha256"], copied_manifest["content_sha256"])
        self.assertNotIn(temp_dir, serialized)
        self.assertNotIn("mtime", serialized)

    def test_missing_policy_and_scene_mismatch_are_rejected(self):
        module = load_manifest_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_dir, provenance = create_scene(temp_dir)
            os.remove(os.path.join(scene_dir, "preprocess_policy.json"))
            with self.assertRaisesRegex(module.DatasetManifestError, "preprocess_policy"):
                module.build_scene_manifest(scene_dir, "garden", 2, provenance)
            with open(os.path.join(scene_dir, "preprocess_policy.json"), "w", encoding="utf-8") as handle:
                json.dump({"source_scene": "loop3", "frames_written": 2}, handle)
            with self.assertRaisesRegex(module.DatasetManifestError, "source_scene"):
                module.build_scene_manifest(scene_dir, "garden", 2, provenance)

    def test_modality_mismatch_noncontinuous_and_unknown_files_are_rejected(self):
        module = load_manifest_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_dir, provenance = create_scene(os.path.join(temp_dir, "mismatch"))
            os.remove(os.path.join(scene_dir, "target_voxel", "000001.npz"))
            with self.assertRaisesRegex(module.DatasetManifestError, "frame ID"):
                module.build_scene_manifest(scene_dir, "garden", 2, provenance)
            scene_dir, provenance = create_scene(os.path.join(temp_dir, "gap"))
            for modality in ("radar_voxel", "lidar_voxel", "target_voxel"):
                os.rename(
                    os.path.join(scene_dir, modality, "000001.npz"),
                    os.path.join(scene_dir, modality, "000002.npz"),
                )
            os.rename(
                os.path.join(scene_dir, "ir_image", "000001_ir.npy"),
                os.path.join(scene_dir, "ir_image", "000002_ir.npy"),
            )
            with self.assertRaisesRegex(module.DatasetManifestError, "连续"):
                module.build_scene_manifest(scene_dir, "garden", 2, provenance)
            scene_dir, provenance = create_scene(os.path.join(temp_dir, "unknown"))
            write_bytes(os.path.join(scene_dir, "radar_voxel", "README.txt"), b"unexpected")
            with self.assertRaisesRegex(module.DatasetManifestError, "未知文件"):
                module.build_scene_manifest(scene_dir, "garden", 2, provenance)

    def test_file_and_directory_symlinks_are_rejected(self):
        module = load_manifest_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_dir, provenance = create_scene(temp_dir)
            radar_path = os.path.join(scene_dir, "radar_voxel", "000000.npz")
            target_path = os.path.join(temp_dir, "external.npz")
            write_bytes(target_path, b"external")
            os.remove(radar_path)
            os.symlink(target_path, radar_path)
            with self.assertRaisesRegex(module.DatasetManifestError, "符号链接"):
                module.build_scene_manifest(scene_dir, "garden", 2, provenance)
            os.remove(radar_path)
            write_bytes(radar_path, b"radar0")
            ir_dir = os.path.join(scene_dir, "ir_image")
            external_ir = os.path.join(temp_dir, "external_ir")
            os.rename(ir_dir, external_ir)
            os.symlink(external_ir, ir_dir)
            with self.assertRaisesRegex(module.DatasetManifestError, "符号链接"):
                module.build_scene_manifest(scene_dir, "garden", 2, provenance)

    def test_mutated_artifact_policy_and_manifest_are_rejected(self):
        module = load_manifest_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_dir, provenance = create_scene(os.path.join(temp_dir, "artifact"))
            module.write_scene_manifest_atomic(scene_dir, "garden", 2, provenance)
            write_bytes(os.path.join(scene_dir, "radar_voxel", "000000.npz"), b"mutated")
            with self.assertRaisesRegex(module.DatasetManifestError, "不一致"):
                module.validate_scene_manifest(scene_dir, "garden")
            scene_dir, provenance = create_scene(os.path.join(temp_dir, "policy"))
            module.write_scene_manifest_atomic(scene_dir, "garden", 2, provenance)
            with open(os.path.join(scene_dir, "preprocess_policy.json"), "w", encoding="utf-8") as handle:
                json.dump({"source_scene": "garden", "frames_written": 999}, handle)
            with self.assertRaisesRegex(module.DatasetManifestError, "policy"):
                module.validate_scene_manifest(scene_dir, "garden")
            scene_dir, provenance = create_scene(os.path.join(temp_dir, "manifest"))
            manifest_path = module.write_scene_manifest_atomic(scene_dir, "garden", 2, provenance)
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["frame_count"] = 999
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            with self.assertRaisesRegex(module.DatasetManifestError, "content_sha256"):
                module.validate_scene_manifest(scene_dir, "garden")

    def test_provenance_is_complete_regular_and_not_symlinked(self):
        module = load_manifest_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_dir, provenance = create_scene(temp_dir)
            provenance.pop("calibration")
            with self.assertRaisesRegex(module.DatasetManifestError, "provenance"):
                module.build_scene_manifest(scene_dir, "garden", 2, provenance)
            target = os.path.join(temp_dir, "real_calibration.txt")
            write_bytes(target, b"calibration")
            link = os.path.join(temp_dir, "calibration_link.txt")
            os.symlink(target, link)
            provenance["calibration"] = link
            with self.assertRaisesRegex(module.DatasetManifestError, "符号链接"):
                module.build_scene_manifest(scene_dir, "garden", 2, provenance)

    def test_existing_manifest_is_not_overwritten_and_no_temp_file_remains(self):
        module = load_manifest_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_dir, provenance = create_scene(temp_dir)
            path = module.write_scene_manifest_atomic(scene_dir, "garden", 2, provenance)
            with open(path, "rb") as handle:
                before = handle.read()
            with self.assertRaisesRegex(module.DatasetManifestError, "已存在"):
                module.write_scene_manifest_atomic(scene_dir, "garden", 2, provenance)
            with open(path, "rb") as handle:
                after = handle.read()
            leftovers = [name for name in os.listdir(scene_dir) if name.startswith(".dataset_manifest.")]
        self.assertEqual(before, after)
        self.assertEqual(leftovers, [])
```

- [x] **Step 2: 运行 RED 并确认仅因模块缺失失败**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_dataset_manifest_protocol.py -v
```

Expected: 7 项均 FAIL，信息为 `dataset manifest 模块尚未实现`，不是临时目录或测试语法错误。

- [x] **Step 3: 实现纯标准库核心**

核心必须定义以下常量和严格规则：

```python
MANIFEST_FILENAME = "dataset_manifest.json"
SCHEMA_VERSION = 1
REQUIRED_PROVENANCE = (
    "preprocess_script",
    "calibration",
    "radar_index",
    "lidar_index",
)
MODALITY_PATTERNS = {
    "radar_voxel": re.compile(r"^(\d{6})\.(?:npy|npz)$"),
    "lidar_voxel": re.compile(r"^(\d{6})\.(?:npy|npz)$"),
    "target_voxel": re.compile(r"^(\d{6})\.(?:npy|npz)$"),
    "ir_image": re.compile(r"^(\d{6})_ir\.npy$"),
}
```

实现必须：

- 用 `os.path.lexists` 与 `os.path.islink` 区分缺失和 symlink；
- `os.scandir()` 中拒绝未知文件、子目录、symlink 和重复 frame ID；
- 比较四个模态 frame ID 集合，并要求等于 `000000..expected_frame_count-1`；
- 用 1 MiB 分块计算 SHA-256；
- 用 `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)` 生成规范字节；
- `content_sha256` 只对不含自身的 payload 计算；
- validator 重新扫描实际文件并逐项比较 path、size、hash 和 policy hash；
- provenance 只保存 basename 与 SHA-256，validator 检查字段完整和 hash 格式，不依赖原绝对路径；
- 原子写入以 `tempfile.mkstemp(dir=scene_dir, prefix=".dataset_manifest.", suffix=".tmp")` 创建临时文件，flush/fsync 后用 `os.link(temp_path, final_path)` 发布，最终清理临时文件；
- final path 已 `lexists` 时在构建和临时文件创建前失败。

- [x] **Step 4: 运行核心测试并确认 GREEN**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_dataset_manifest_protocol.py -v
```

Expected: 7/7 PASS。

---

### Task 2: CLI 与预处理集成

**Files:**
- Create: `diffusion_consistency_radar/scripts/dataset_manifest.py`
- Modify: `NTU4DRadLM_pre_processing/NTU4DRadLM_pre_processing.py`
- Modify: `test/unit/test_dataset_manifest_protocol.py`

**Interfaces:**
- Consumes: Task 1 三个核心函数。
- Produces: `create` / `validate` CLI。
- Produces: `ensure_fresh_scene_output(scene_out_path: str) -> None`。

- [x] **Step 1: 写 CLI 与预处理安全 RED 测试**

在测试文件新增 `subprocess` 和 `sys` import，并加入：

```python
def test_cli_create_and_validate_round_trip(self):
    with tempfile.TemporaryDirectory() as temp_dir:
        scene_dir, provenance = create_scene(temp_dir)
        script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "diffusion_consistency_radar", "scripts", "dataset_manifest.py",
        )
        create_result = subprocess.run(
            [sys.executable, script, "create", "--scene_dir", scene_dir, "--scene", "garden",
             "--expected_frame_count", "2", "--preprocess_script", provenance["preprocess_script"],
             "--calibration", provenance["calibration"], "--radar_index", provenance["radar_index"],
             "--lidar_index", provenance["lidar_index"]],
            text=True, capture_output=True, check=False,
        )
        validate_result = subprocess.run(
            [sys.executable, script, "validate", "--scene_dir", scene_dir, "--expected_scene", "garden"],
            text=True, capture_output=True, check=False,
        )
    self.assertEqual(create_result.returncode, 0, create_result.stderr)
    self.assertEqual(validate_result.returncode, 0, validate_result.stderr)
    self.assertIn("content_sha256", validate_result.stdout)

def test_preprocess_requires_fresh_output_and_writes_manifest_after_policy(self):
    source_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "NTU4DRadLM_pre_processing", "NTU4DRadLM_pre_processing.py",
    )
    with open(source_path, encoding="utf-8") as handle:
        source = handle.read()
    self.assertIn("def ensure_fresh_scene_output", source)
    self.assertLess(source.index("ensure_fresh_scene_output(scene_out_path)"), source.index("ensure_dir(os.path.join(scene_out_path"))
    self.assertLess(source.index('"preprocess_policy.json"'), source.index("write_scene_manifest_atomic("))
    self.assertIn("if failures:", source)
    self.assertIn("raise SystemExit(1)", source)
```

- [x] **Step 2: 运行 RED**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_dataset_manifest_protocol.py -v
```

Expected: 核心 7 项通过；CLI 文件缺失和预处理集成断言失败。

- [x] **Step 3: 实现 CLI 和预处理 fail-closed**

CLI 使用子命令 parser，捕获 `DatasetManifestError` 后调用 `parser.error(str(exc))`。`create` 将四个 provenance 参数映射为固定 key，`validate` 不写文件。

预处理脚本：

- 在项目根加入 `sys.path` 后导入 `write_scene_manifest_atomic`；
- 新增 `ensure_fresh_scene_output()`，对 symlink、非目录和非空目录抛 `RuntimeError`，不存在时仅创建场景根；
- 在 `radar_voxel`、`lidar_voxel`、`target_voxel` 和 `ir_image` 四个 `ensure_dir` 调用前执行 preflight；
- 缺 index 时抛异常，不再 print 后 return；
- policy 写完后调用 manifest writer，传 `written` 和四个 provenance 路径；
- 主入口收集 `(scene, exception)`，打印 traceback，最后存在失败时 `raise SystemExit(1)`。

- [x] **Step 4: 运行 Task 2 GREEN**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_dataset_manifest_protocol.py -v
conda run -n Radar-Diffusion python test/unit/test_sensor_aware_target.py -v
```

Expected: manifest 测试 9 项全部 PASS，既有 sensor-aware target 测试全部 PASS。

---

### Task 3: 正式 launcher 强制验证

**Files:**
- Modify: `diffusion_consistency_radar/launch/train_unified.sh`
- Modify: `diffusion_consistency_radar/launch/inference_ldm.sh`
- Modify: `diffusion_consistency_radar/launch/inference_cd.sh`
- Modify: `diffusion_consistency_radar/launch/inference_uniified.sh`
- Modify: `test/unit/test_dataset_manifest_protocol.py`

**Interfaces:**
- Consumes: `python dataset_manifest.py validate --scene_dir DIR --expected_scene NAME`。
- Produces: 四个正式 launcher 的不可跳过 fail-closed gate。

- [x] **Step 1: 写 launcher 顺序 RED 测试**

```python
def test_formal_launchers_validate_manifest_without_skip_switch(self):
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    relative_paths = (
        "diffusion_consistency_radar/launch/train_unified.sh",
        "diffusion_consistency_radar/launch/inference_ldm.sh",
        "diffusion_consistency_radar/launch/inference_cd.sh",
        "diffusion_consistency_radar/launch/inference_uniified.sh",
    )
    for relative_path in relative_paths:
        with self.subTest(path=relative_path):
            with open(os.path.join(root, relative_path), encoding="utf-8") as handle:
                script = handle.read()
            self.assertIn('MANIFEST_SCRIPT="${PROJECT_DIR}/scripts/dataset_manifest.py"', script)
            self.assertIn('"${MANIFEST_SCRIPT}" validate', script)
            self.assertNotIn("SKIP_MANIFEST", script)
            validation_index = script.index('"${MANIFEST_SCRIPT}" validate')
            if relative_path.endswith("train_unified.sh"):
                self.assertLess(validation_index, script.index('rm -rf "${TRAIN_DATASET_DIR}"'))
            else:
                self.assertLess(validation_index, script.index('python "${INFER_SCRIPT}"'))
```

- [x] **Step 2: 运行 RED**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_dataset_manifest_protocol.py -v
```

Expected: 只有 launcher gate 测试失败，四个脚本缺少 `MANIFEST_SCRIPT`。

- [x] **Step 3: 接入四个正式 launcher**

所有脚本在变量区加入：

```bash
MANIFEST_SCRIPT="${PROJECT_DIR}/scripts/dataset_manifest.py"
```

训练脚本先用独立循环验证所有 `TRAIN_SCENES`，全部通过后才执行现有 `rm -rf` 和 symlink 创建：

```bash
for SCENE in "${TRAIN_SCENES[@]}"; do
    SRC_SCENE_DIR="${PREPROCESSED_ROOT}/${SCENE}"
    if [ ! -d "${SRC_SCENE_DIR}" ]; then
        echo "Error: train scene directory not found: ${SRC_SCENE_DIR}"
        exit 1
    fi
    python "${MANIFEST_SCRIPT}" validate \
        --scene_dir "${SRC_SCENE_DIR}" \
        --expected_scene "${SCENE}"
done
```

三个推理脚本在 test scene 解析完成后、第一次调用 inference 前逐场景执行同一 validate 命令。`inference_uniified.sh` 只验证一次全部 `TEST_SCENES`，后续 LDM/CD/1-step/4-step 复用结果。

- [x] **Step 4: 运行 Task 3 GREEN 与 shell 语法检查**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_dataset_manifest_protocol.py -v
bash -n diffusion_consistency_radar/launch/train_unified.sh
bash -n diffusion_consistency_radar/launch/inference_ldm.sh
bash -n diffusion_consistency_radar/launch/inference_cd.sh
bash -n diffusion_consistency_radar/launch/inference_uniified.sh
```

Expected: 测试全部 PASS，四个 shell exit 0。

---

### Task 4: 最终验证与记录

**Files:**
- Modify: `TODO/findings.md`
- Modify: `TODO/task_plan.md`
- Modify: `TODO/progress.md`

- [x] **Step 1: 运行全部聚焦验证**

```bash
conda run -n Radar-Diffusion python test/unit/test_dataset_manifest_protocol.py -v
conda run -n Radar-Diffusion python test/unit/test_dataset_protocol_metadata.py -v
conda run -n Radar-Diffusion python test/unit/test_sensor_aware_target.py -v
conda run -n Radar-Diffusion python -m py_compile \
  diffusion_consistency_radar/dataset_manifest.py \
  diffusion_consistency_radar/scripts/dataset_manifest.py \
  NTU4DRadLM_pre_processing/NTU4DRadLM_pre_processing.py \
  test/unit/test_dataset_manifest_protocol.py
bash -n diffusion_consistency_radar/launch/train_unified.sh
bash -n diffusion_consistency_radar/launch/inference_ldm.sh
bash -n diffusion_consistency_radar/launch/inference_cd.sh
bash -n diffusion_consistency_radar/launch/inference_uniified.sh
git diff --check
```

Expected: 全部单元测试 PASS，编译/shell/diff exit 0。

- [x] **Step 2: 只读验证当前真实场景被拒绝**

```bash
conda run -n Radar-Diffusion python diffusion_consistency_radar/scripts/dataset_manifest.py validate \
  --scene_dir Data/NTU4DRadLM_Pre_sensor_aware/loop3 \
  --expected_scene loop3
```

Expected: exit 2，明确报告缺少 `dataset_manifest.json`；不生成或修改文件。

- [x] **Step 3: 更新三份 TODO**

记录 RED/GREEN 数量、真实数据拒绝证据、manifest schema、launcher gate 顺序、未运行长任务，以及对监督/体素/指标的影响。

- [x] **Step 4: 完成状态检查**

确认本计划所有 Step 完成、暂存区为空且未提交。下一项仍为正式推理 sensor-aware/真实 IR 与部署/评价入口解耦。
