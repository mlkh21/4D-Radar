# 正式真实 IR 与部署/离线评价解耦实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让正式 LDM/CD 生成只使用 sensor-aware Radar、真实 IR 与真实 thermal 外参，并让 target/raw LiDAR 仅由消费已保存预测的独立离线评价入口读取。

**Architecture:** `inference.py` 增加 fail-closed 的严格真实 IR preflight，同时保留现有非正式兼容模式；无评价参数时只写部署运行时 CSV 和可复现网格元数据。新的 `evaluate_saved_predictions.py` 不加载 checkpoint、不导入生成入口，严格配对同一批已保存 voxel 后计算离线指标。正式生成 shell 与离线评价 shell 分开，各自保留 dataset manifest 门禁。

**Tech Stack:** Python 3、NumPy、PyTorch、SciPy `cKDTree`、`unittest`、Bash、现有 `dataset_manifest.py` 与 `cm.evaluation_metrics`。

## Global Constraints

- 使用当前普通 `withir` checkout 原地实施，保留所有既有未提交修改；不创建 worktree，不暂存、不提交、不推送。
- 先写 RED 测试，再做最小 GREEN；不运行训练、预处理、正式推理或全量评价。
- 正式数据根固定为 `Data/NTU4DRadLM_Pre_sensor_aware`，继续 fail-closed 验证四模态 `dataset_manifest.json`。
- 不修改监督 target、模型结构、checkpoint、体素网格配置、thermal K/D 或真实逐帧速度/时间同步协议。
- 严格模式拒绝单模态 checkpoint、缺失/符号链接/非法 IR 以及 mock/non-thermal 外参；兼容模式保留 mock fallback。
- 正式生成不接收 target/raw LiDAR；离线评价不得重新运行模型，且不得修改已保存 prediction。
- 所有新文件带中文文件头说明，新增功能注释默认中文。
- 每项完成后同步 `TODO/findings.md`、`TODO/task_plan.md`、`TODO/progress.md`；记录监督、体素数量和指标协议影响。
- 本计划不含 commit 步骤，因为用户已明确拒绝自动提交；任何实现结束后暂存区必须保持为空。

---

## 文件职责映射

- Modify `diffusion_consistency_radar/scripts/inference.py`：严格真实 IR preflight、运行时 CSV、`inference_run.json`。
- Create `diffusion_consistency_radar/scripts/evaluate_saved_predictions.py`：纯离线已保存预测评价。
- Modify `diffusion_consistency_radar/launch/inference_ldm.sh`：LDM 正式部署生成入口。
- Modify `diffusion_consistency_radar/launch/inference_cd.sh`：CD 1/4 步正式部署生成入口。
- Modify `diffusion_consistency_radar/launch/inference_uniified.sh`：统一正式部署生成入口。
- Create `diffusion_consistency_radar/launch/evaluate_inference.sh`：正式离线评价入口。
- Modify `test/unit/test_multimodal_inference_interface.py`：严格 IR 与运行元数据契约。
- Create `test/unit/test_formal_inference_protocol.py`：shell 边界和离线 evaluator 功能契约。
- Modify `TODO/findings.md`, `TODO/task_plan.md`, `TODO/progress.md`：决策、进展、验证证据。

---

### Task 1: 严格真实 IR preflight

**Files:**

- Modify: `test/unit/test_multimodal_inference_interface.py`
- Modify: `diffusion_consistency_radar/scripts/inference.py:278-345,804-1010`

**Interfaces:**

- Produces: `validate_real_ir_model(model) -> None`
- Produces: `load_multimodal_meta_for_radar(radar_path: str, device, require_real_ir: bool = False) -> dict`
- Produces: `preflight_real_ir_inputs(model, radar_paths: Sequence[str]) -> None`
- Consumes: `CalibrationProvider.load_with_metadata()` and `_resize_or_pad_ir_tensor()`.

- [x] **Step 1: Write strict-mode RED tests**

Add `numpy`, `tempfile` and `unittest.mock` imports. Add a fixture which creates `root/scene/{radar_voxel,ir_image}`, a radar file, an IR file, and `root/config/calib_radar_to_thermal.txt` with identity R and `T: 1 2 3`. Test:

```python
meta = inference.load_multimodal_meta_for_radar(
    radar_path, torch.device("cpu"), require_real_ir=True
)
self.assertEqual(float(meta["is_mock_ir"].item()), 0.0)
self.assertEqual(float(meta["is_mock_calib"].item()), 0.0)
self.assertAlmostEqual(float(meta["t_vec"][0].item()), 1.01, places=6)
self.assertAlmostEqual(float(meta["legacy_sync_displacement_x_m"]), 0.01, places=6)
```

Also assert `RuntimeError` for missing IR, an IR symlink, `ndim=1`, NaN, missing thermal calibration and a model with `is_multimodal=False`; assert non-strict missing IR still returns `is_mock_ir=1`.

- [x] **Step 2: Run RED**

Run `conda run -n Radar-Diffusion python test/unit/test_multimodal_inference_interface.py -v`.

Expected: existing 16 tests pass; new tests fail because `require_real_ir`, `validate_real_ir_model` and the legacy-sync field do not exist.

- [x] **Step 3: Implement strict loader and model gate**

Add:

```python
LEGACY_SYNC_DISPLACEMENT_X_M = 0.01


def validate_real_ir_model(model):
    """正式真实 IR 模式只接受实际构建出的多模态生成模型。"""
    if not bool(getattr(model, "is_multimodal", False)):
        raise RuntimeError("--require_real_ir requires a multimodal Radar+IR checkpoint")


def _load_ir_array(ir_path: str, require_real_ir: bool) -> np.ndarray:
    if require_real_ir and os.path.islink(ir_path):
        raise RuntimeError(f"严格真实 IR 模式拒绝符号链接: {ir_path}")
    if require_real_ir and not os.path.isfile(ir_path):
        raise RuntimeError(f"严格真实 IR 模式缺少普通 IR 文件: {ir_path}")
    array = np.load(ir_path).astype(np.float32)
    if array.ndim not in (2, 3):
        raise RuntimeError(f"IR 数组维度非法 {array.shape}: {ir_path}")
    if not np.isfinite(array).all():
        raise RuntimeError(f"IR 数组含非有限值: {ir_path}")
    return array
```

Change `load_multimodal_meta_for_radar` to accept `require_real_ir=False`, use `os.path.lexists`, reject missing strict IR, call `CalibrationProvider.load_with_metadata()`, reject mock/non-thermal calibration in strict mode, and always apply `t_vec[0] += 0.01` so real inference matches the existing training projection protocol. Return `legacy_sync_displacement_x_m` and `calib_source` alongside the current tensor fields.

Add:

```python
def preflight_real_ir_inputs(model, radar_paths):
    """在创建输出目录前验证全部正式帧，禁止留下部分正式结果。"""
    validate_real_ir_model(model)
    for radar_path in radar_paths:
        meta = load_multimodal_meta_for_radar(
            radar_path, torch.device("cpu"), require_real_ir=True
        )
        if float(meta["is_mock_ir"].item()) != 0.0 or float(meta["is_mock_calib"].item()) != 0.0:
            raise RuntimeError(f"严格真实 IR preflight 得到 mock meta: {radar_path}")
```

Add `--require_real_ir`; in per-file mode collect paths and run the preflight before `os.makedirs(args.output_dir, exist_ok=True)`. Load meta per frame when `args.use_multimodal_meta or args.require_real_ir` and pass `require_real_ir=args.require_real_ir`.

- [x] **Step 4: Run GREEN**

Run the same test file. Expected: all tests pass and compatibility mock behavior remains covered.

---

### Task 2: Deployment runtime artifact protocol

**Files:**

- Modify: `test/unit/test_multimodal_inference_interface.py`
- Modify: `diffusion_consistency_radar/scripts/inference.py:840-1270`

**Interfaces:**

- Produces: `resolve_effective_voxel_size(voxel_size, pc_range, target_size) -> List[float]`
- Produces: `build_inference_run_metadata(args, generator, frame_count: int) -> dict`
- Produces files: `inference_runtime.csv`, `inference_run.json`.

- [x] **Step 1: Write runtime metadata RED tests**

Use `types.SimpleNamespace` and a dummy generator. Assert `target_size=[2,4,5]`, `[0,0,0,40,20,16]` resolves to `voxel_size=[10,4,8]`, strict/multimodal flags are true, and `frame_count=2`. Assert `RUNTIME_FIELDS` equals:

```python
[
    "index", "radar_file", "radar_point_count", "effective_occ_threshold",
    "inference_seconds", "pred_point_count", "is_empty_frame",
    "used_topk_fallback", "train_duration_seconds", "total_infer_seconds",
    "avg_infer_seconds", "avg_pred_point_count", "empty_frame_rate",
]
```

- [x] **Step 2: Run RED**

Run the multimodal inference interface test. Expected: only new helper/field tests fail.

- [x] **Step 3: Implement metadata and runtime-only CSV branch**

Implement the exact `RUNTIME_FIELDS` above and:

```python
def resolve_effective_voxel_size(voxel_size, pc_range, target_size):
    if voxel_size is not None:
        return [float(value) for value in voxel_size]
    z_size, x_size, y_size = [int(value) for value in target_size]
    return [
        (float(pc_range[3]) - float(pc_range[0])) / x_size,
        (float(pc_range[4]) - float(pc_range[1])) / y_size,
        (float(pc_range[5]) - float(pc_range[2])) / z_size,
    ]
```

`build_inference_run_metadata` returns `stage="deployment_generation"`, resolved target/source/model grids, resolved voxel size, fixed threshold, model type, steps, sampler, `model_is_multimodal`, `require_real_ir`, frame count and strict-mode legacy sync displacement.

Set `legacy_evaluation = args.compare_with_target or args.compare_with_lidar or args.report_task_metrics`. Keep the current `inference_metrics.csv` and schema only for that branch. Otherwise write `inference_runtime.csv` with `RUNTIME_FIELDS`, one runtime row per generated frame and one `__summary__` row. After successful generation, atomically write `inference_run.json` through a sibling temporary file and `os.replace`.

- [x] **Step 4: Run GREEN**

Run the same test file. Expected: new runtime helpers and existing legacy interfaces all pass.

---

### Task 3: Saved-prediction offline evaluator

**Files:**

- Create: `test/unit/test_formal_inference_protocol.py`
- Create: `diffusion_consistency_radar/scripts/evaluate_saved_predictions.py`

**Interfaces:**

- Produces the approved `evaluate_saved_predictions(...) -> dict` signature.
- Consumes: `cm.dataset_loader.load_sparse_voxel`, `crop_voxel_channels_to_pc_range`, `resize_voxel_channels` and metrics from `cm.evaluation_metrics`.
- Produces files: `evaluation_frames.csv`, `evaluation_summary.json`.

- [x] **Step 1: Write evaluator RED tests**

Create two `*_voxel.npy` predictions, matching radar/target arrays, matching uncertainty arrays and complete `inference_run.json` with `target_size=[2,2,2]` and a two-metre cube range. Create two raw LiDAR files and index lines `1`/`0`. Call the public function and assert:

```python
self.assertEqual(summary["stage"], "offline_evaluation")
self.assertTrue(summary["prediction_unchanged"])
self.assertEqual(summary["frame_count"], 2)
self.assertEqual(summary["model_pc_range"], [0.0, 0.0, 0.0, 2.0, 2.0, 2.0])
self.assertTrue(os.path.isfile(os.path.join(output_dir, "evaluation_frames.csv")))
self.assertTrue(os.path.isfile(os.path.join(output_dir, "evaluation_summary.json")))
```

Hash predictions before/after and assert equality. Add failures for non-empty output, missing metadata, frame-set mismatch, invalid/NaN arrays, only one raw LiDAR argument and out-of-range index.

- [x] **Step 2: Run RED**

Run `conda run -n Radar-Diffusion python test/unit/test_formal_inference_protocol.py -v`.

Expected: import failure because the evaluator file does not exist.

- [x] **Step 3: Implement evaluator**

Start the new file with a Chinese functional docstring. Implement exactly:

```python
def evaluate_saved_predictions(
    pred_voxel_dir: str,
    radar_voxel_dir: str,
    target_voxel_dir: str,
    output_dir: str,
    run_metadata_path: str = "",
    raw_livox_dir: str = "",
    lidar_index_file: str = "",
    occ_threshold: Optional[float] = None,
    target_threshold: float = 0.5,
    source_pc_range: Optional[Sequence[float]] = None,
    model_pc_range: Optional[Sequence[float]] = None,
    target_size: Optional[Sequence[int]] = None,
    max_files: int = 0,
) -> dict:
```

Preflight all inputs before output creation: require prediction/radar/target directories; reject non-empty output; require complete default run metadata unless a value has an explicit diagnostic override; accept only `^(\d+)_voxel\.npy$` prediction names; require identical prediction/radar/target frame sets; validate all shapes/finite values; require raw LiDAR and index together; map raw LiDAR with `indices[int(frame_id)]` and reject bounds errors.

Convert source XY-Z-C radar/target to C-Z-X-Y using existing crop/resize helpers. Keep saved prediction in C-Z-X-Y. Use fixed prediction/radar threshold and independent target threshold. Compute point counts, symmetric Chamfer, count ratio, centroid deltas, near-field precision/recall/BEV IoU/NN mean/2m match ratio, optional raw LiDAR Chamfer and optional uncertainty ECE/Brier/NLL/correlation. Do not import `inference.py`.

Only after successful preflight create the output, write `evaluation_frames.csv`, and atomically publish a summary containing:

```python
{
    "stage": "offline_evaluation",
    "prediction_unchanged": True,
    "frame_count": len(frame_ids),
    "occ_threshold": float(resolved_occ_threshold),
    "target_threshold": float(target_threshold),
    "target_size": list(resolved_target_size),
    "source_pc_range": list(resolved_source_range),
    "model_pc_range": list(resolved_model_range),
}
```

- [x] **Step 4: Run GREEN**

Run the formal protocol test. Expected: functional and fail-closed evaluator cases pass without checkpoint loading or model sampling.

---

### Task 4: Formal deployment and evaluation shell boundary

**Files:**

- Modify: `diffusion_consistency_radar/launch/inference_ldm.sh`
- Modify: `diffusion_consistency_radar/launch/inference_cd.sh`
- Modify: `diffusion_consistency_radar/launch/inference_uniified.sh`
- Create: `diffusion_consistency_radar/launch/evaluate_inference.sh`
- Modify: `test/unit/test_formal_inference_protocol.py`

**Interfaces:**

- Deployment commands consume checkpoint + sensor-aware radar/IR only.
- Evaluation command consumes `_deploy` saved voxels + sensor-aware radar/target + raw LiDAR mapping only.

- [x] **Step 1: Add static launcher RED tests**

For each generation launcher assert sensor-aware root, `--require_real_ir`, three save flags, `_deploy`, and manifest validation before `inference.py`; assert no target/LiDAR evaluation arguments. For evaluation launcher assert `evaluate_saved_predictions.py`, sensor-aware root, manifest validation, target/raw LiDAR/index and `_evaluation`; assert no VAE/model checkpoint and no `scripts/inference.py`.

- [x] **Step 2: Run RED**

Run the formal protocol test. Expected: evaluator cases remain green; launcher cases fail against old mixed scripts and the missing evaluator launcher.

- [x] **Step 3: Modify generation launchers**

Set `PREPROCESSED_ROOT="${PROJECT_ROOT}/Data/NTU4DRadLM_Pre_sensor_aware"`. Remove target/raw-LiDAR variables, checks and arguments. Preserve manifest validation before Python inference. Add `--require_real_ir --save_voxel --save_pointcloud --save_uncertainty` and rename leaves to `_ldm_deploy`, `_cd_1step_deploy`, `_cd_4step_deploy` as applicable.

- [x] **Step 4: Create evaluation launcher**

Create a Chinese-header Bash script with `set -euo pipefail`. Accept exactly `ldm|cd|cd4`, resolve the matching deploy/evaluation suffix, read test scenes using the existing YAML helper pattern, validate each sensor-aware scene manifest, require deploy directory and `inference_run.json`, then call only `evaluate_saved_predictions.py` with saved prediction, sensor-aware radar/target, raw Livox and LiDAR index paths.

- [x] **Step 5: Run GREEN and shell syntax checks**

Run the formal protocol test and `bash -n` on the four generation/evaluation launchers. Expected: all pass.

---

### Task 5: Regression, records and completion evidence

**Files:**

- Modify: `TODO/findings.md`
- Modify: `TODO/task_plan.md`
- Modify: `TODO/progress.md`
- Modify: this implementation plan checkbox status.

**Interfaces:** none; this task records and validates the completed protocol.

- [x] **Step 1: Run focused regression**

Run these four direct unittest files:

```bash
conda run -n Radar-Diffusion python test/unit/test_multimodal_inference_interface.py -v
conda run -n Radar-Diffusion python test/unit/test_formal_inference_protocol.py -v
conda run -n Radar-Diffusion python test/unit/test_dataset_protocol_metadata.py -v
conda run -n Radar-Diffusion python test/unit/test_dataset_manifest_protocol.py -v
```

Expected: every test passes; no checkpoint is loaded and no model sampling occurs.

- [x] **Step 2: Run static verification**

Run `py_compile` for the two production Python scripts and two tests; run `bash -n` for all four launchers; run `git diff --check` and `git diff --cached --quiet`. Expected: all exit 0 and staging remains empty.

- [x] **Step 3: Verify current real data remains safely blocked**

Run only strict manifest validation for sensor-aware `loop3`, not inference:

```bash
conda run -n Radar-Diffusion python diffusion_consistency_radar/scripts/dataset_manifest.py validate \
  --scene_dir Data/NTU4DRadLM_Pre_sensor_aware/loop3 \
  --expected_scene loop3
```

Expected: exit 2 because historical data lacks `dataset_manifest.json`; no manifest or result is written.

- [x] **Step 4: Update persistent records**

Record exact test counts, errors/resolutions and research impact: supervision/target/model/checkpoint/grid/frame membership unchanged; stored voxel counts unchanged; formal predictions may change because real IR replaces mock/disabled IR and real extrinsics receive the same legacy `+0.01m` training compensation; runtime/evaluation schemas are intentionally separated; historical data remains blocked until clean preprocessing creates a strict manifest.

- [x] **Step 5: Final status review without commit**

Run `git status --short` and `git diff --stat`; verify no data/checkpoint/log/result deletion and no staged files. Mark completed checkboxes. Do not invoke staging, commit, push, training, preprocessing, formal inference or full evaluation.
