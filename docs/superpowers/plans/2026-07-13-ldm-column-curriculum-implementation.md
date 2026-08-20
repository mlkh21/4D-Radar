# LDM Column Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 LDM 列级正负平衡损失增加默认关闭、可恢复、可审计的 epoch 线性课程，并提供受保护的 v11 短实验入口。

**Architecture:** 在 `unified_train.py` 中用一个无状态纯函数根据 `epoch/total_epochs` 计算有效正负列权重，trainer 每轮只计算一次并传入现有损失函数。课程配置、当轮有效权重同时写入 metrics 和 checkpoint；旧配置默认保持固定权重。测试入口继续扩展现有 v10 runner，不复制整份脚本。

**Tech Stack:** Python 3.8, PyTorch, YAML, Bash, `unittest`, 现有 Conda `Radar-Diffusion` 环境。

---

## File Map

- Modify: `diffusion_consistency_radar/scripts/unified_train.py`  
  负责课程计算、配置验证、trainer 接线、CSV 和 checkpoint 自描述。
- Modify: `test/unit/test_ldm_vertical_structure_loss.py`  
  负责课程数学边界、兼容、trainer 接线和 checkpoint 元数据测试。
- Modify: `test/mini-test/train_minimal.sh`  
  负责将 mini 环境变量写入生成的 YAML。
- Modify: `test/mini-test/run_ldm_z64_v10_column_experiment.sh`  
  扩展现有安全 runner，增加 `V11` 课程变体，保留非空目录、锁、路径和 training-only 保护。
- Modify: `test/unit/test_mini_train_script.py`  
  负责 mini YAML 参数传递、V11 变体单变量和 hostile-environment 协议测试。
- Modify: `TODO/findings.md`, `TODO/task_plan.md`, `TODO/progress.md`  
  记录监督信号、体素数量影响、测试结果和下一步门槛。

### Task 1: Add the pure epoch curriculum and validation

**Files:**
- Modify: `diffusion_consistency_radar/scripts/unified_train.py:590-640`
- Test: `test/unit/test_ldm_vertical_structure_loss.py:1405-1490`

- [ ] **Step 1: Write failing mathematical-contract tests**

Add imports and a new test class covering exact interpolation and compatibility:

```python
from diffusion_consistency_radar.scripts.unified_train import column_curriculum_weights


class ColumnCurriculumWeightsTest(unittest.TestCase):
    def test_three_epoch_curve_matches_v11_contract(self):
        actual = [
            column_curriculum_weights(
                epoch,
                3,
                enabled=True,
                positive_start=0.03,
                positive_final=0.02,
                negative_start=0.0,
                negative_final=0.01,
            )
            for epoch in (1, 2, 3)
        ]
        self.assertEqual(actual, [(0.03, 0.0), (0.025, 0.005), (0.02, 0.01)])

    def test_disabled_curriculum_returns_fixed_final_weights(self):
        self.assertEqual(
            column_curriculum_weights(
                1, 3, enabled=False,
                positive_start=99.0, positive_final=0.2,
                negative_start=99.0, negative_final=0.4,
            ),
            (0.2, 0.4),
        )

    def test_single_epoch_uses_start_weights_when_enabled(self):
        self.assertEqual(
            column_curriculum_weights(
                1, 1, enabled=True,
                positive_start=0.03, positive_final=0.02,
                negative_start=0.0, negative_final=0.01,
            ),
            (0.03, 0.0),
        )
```

Add subtests rejecting `epoch=0`, `epoch>total_epochs`, `total_epochs=0`, non-boolean `enabled`, and every negative/NaN/Inf start/final weight.

- [ ] **Step 2: Run RED test**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_ldm_vertical_structure_loss.py
```

Expected: FAIL because `column_curriculum_weights` does not exist.

- [ ] **Step 3: Implement the minimal pure function**

Add before `LDM_LOSS_COMPONENT_NAMES`:

```python
def column_curriculum_weights(
    epoch: int,
    total_epochs: int,
    *,
    enabled: bool,
    positive_start: float,
    positive_final: float,
    negative_start: float,
    negative_final: float,
) -> tuple:
    """计算当前 epoch 的列级正负损失权重。"""
    if not isinstance(enabled, bool):
        raise TypeError("enabled 必须是 bool")
    if not isinstance(epoch, int) or not isinstance(total_epochs, int):
        raise TypeError("epoch 和 total_epochs 必须是整数")
    if total_epochs < 1 or not 1 <= epoch <= total_epochs:
        raise ValueError("epoch 必须在 [1,total_epochs] 内")

    weights = {
        "positive_start": float(positive_start),
        "positive_final": float(positive_final),
        "negative_start": float(negative_start),
        "negative_final": float(negative_final),
    }
    for name, value in weights.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} 必须是有限非负数，实际为 {value!r}")
    if not enabled:
        return weights["positive_final"], weights["negative_final"]

    progress = (epoch - 1) / max(total_epochs - 1, 1)
    positive = weights["positive_start"] + progress * (
        weights["positive_final"] - weights["positive_start"]
    )
    negative = weights["negative_start"] + progress * (
        weights["negative_final"] - weights["negative_start"]
    )
    return positive, negative
```

- [ ] **Step 4: Run GREEN test**

Run the same direct test file. Expected: all existing and new tests PASS.

- [ ] **Step 5: Check syntax and diff**

```bash
conda run -n Radar-Diffusion python -m py_compile diffusion_consistency_radar/scripts/unified_train.py
git diff --check
```

Expected: both commands succeed. Do not commit while unrelated staged moves remain in the Git index; record the completed task in `TODO/progress.md` instead.

### Task 2: Wire curriculum into trainer, metrics, resume-safe checkpoint metadata

**Files:**
- Modify: `diffusion_consistency_radar/scripts/unified_train.py:610-635,1450-1870`
- Test: `test/unit/test_ldm_vertical_structure_loss.py:1405-1505`

- [ ] **Step 1: Write failing trainer/config tests**

Extend `LDMTrainerUtilityTest` to assert:

```python
def test_curriculum_defaults_are_backward_compatible(self):
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = self._make_trainer(temp_dir)
        self.assertFalse(trainer.decoded_column_curriculum_enabled)
        self.assertEqual(trainer._column_weights_for_epoch(1), (0.0, 0.0))

def test_v11_curriculum_is_epoch_deterministic_and_self_describing(self):
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = self._make_trainer(temp_dir, {
            "epochs": 3,
            "decoded_column_curriculum_enabled": True,
            "decoded_column_positive_start_weight": 0.03,
            "decoded_column_positive_weight": 0.02,
            "decoded_column_negative_start_weight": 0.0,
            "decoded_column_negative_weight": 0.01,
        })
        self.assertEqual(trainer._column_weights_for_epoch(2), (0.025, 0.005))
        metadata = trainer._ldm_loss_config(epoch=2)
        self.assertTrue(metadata["decoded_column_curriculum_enabled"])
        self.assertEqual(metadata["effective_column_positive_weight"], 0.025)
        self.assertEqual(metadata["effective_column_negative_weight"], 0.005)
```

Patch `compute_ldm_loss_components` in one `train_epoch` unit test and assert epoch 1 receives
`decoded_column_positive_weight=0.03` and `decoded_column_negative_weight=0.0`.
Add invalid-config cases for a string enable flag and invalid start weights.

- [ ] **Step 2: Write failing CSV/checkpoint protocol tests**

Assert `LDM_METRICS_HEADER` contains `effective_column_positive_weight` and
`effective_column_negative_weight` before `lr`, and `_log_metrics()` writes the active values.
Assert `_ldm_loss_config(epoch=3)` stores the enable flag, start/final values, and exact effective
weights. Expected RED: attributes, header fields, and epoch argument are absent.

- [ ] **Step 3: Implement trainer configuration and epoch lookup**

In `OptimizedLDMTrainer.__init__` read and validate:

```python
self.decoded_column_curriculum_enabled = ldm_config.get(
    "decoded_column_curriculum_enabled", False
)
self.decoded_column_positive_start_weight = float(ldm_config.get(
    "decoded_column_positive_start_weight", self.decoded_column_positive_weight
))
self.decoded_column_negative_start_weight = float(ldm_config.get(
    "decoded_column_negative_start_weight", self.decoded_column_negative_weight
))
```

Reject non-boolean enable flags and pass all four weights through the pure helper's validation.
Add:

```python
def _column_weights_for_epoch(self, epoch: int) -> tuple:
    return column_curriculum_weights(
        epoch,
        int(self.ldm_config.get("epochs", 200)),
        enabled=self.decoded_column_curriculum_enabled,
        positive_start=self.decoded_column_positive_start_weight,
        positive_final=self.decoded_column_positive_weight,
        negative_start=self.decoded_column_negative_start_weight,
        negative_final=self.decoded_column_negative_weight,
    )
```

- [ ] **Step 4: Use effective weights exactly once per epoch**

At the start of `train_epoch()` compute:

```python
effective_positive_weight, effective_negative_weight = self._column_weights_for_epoch(epoch)
self.last_effective_column_weights = (
    effective_positive_weight,
    effective_negative_weight,
)
```

Pass these local values, rather than the static final weights, to
`compute_ldm_loss_components()`. This leaves all decoded tensors, target occupancy, and raw loss
components unchanged; only the weighted contribution changes.

- [ ] **Step 5: Add audit fields to CSV and checkpoint metadata**

Insert into `LDM_METRICS_HEADER` before `lr`:

```python
"effective_column_positive_weight",
"effective_column_negative_weight",
```

Write `self.last_effective_column_weights` in `_log_metrics()`. Extend
`_ldm_loss_config(epoch=None)` with static schedule fields and, only when an epoch is supplied,
effective fields from `_column_weights_for_epoch(epoch)`. Change both checkpoint payloads in
`train()` to call `_ldm_loss_config(epoch=epoch)`.

- [ ] **Step 6: Run focused tests and compile check**

```bash
conda run -n Radar-Diffusion python test/unit/test_ldm_vertical_structure_loss.py
conda run -n Radar-Diffusion python test/unit/test_vae_checkpoint_protocol.py
conda run -n Radar-Diffusion python -m py_compile diffusion_consistency_radar/scripts/unified_train.py
```

Expected: all tests pass; no real training starts.

### Task 3: Wire mini config and extend the guarded runner with V11

**Files:**
- Modify: `test/mini-test/train_minimal.sh:35-65,225-255,314-410`
- Modify: `test/mini-test/run_ldm_z64_v10_column_experiment.sh:1-215`
- Modify: `test/unit/test_mini_train_script.py:70-100,280-360,1868-2140`

- [ ] **Step 1: Write failing mini-config tests**

Extend the existing shell argument/parser test with:

```python
("MINI_LDM_COLUMN_CURRICULUM_ENABLED", "ldm_column_curriculum_enabled"),
("MINI_LDM_COLUMN_POSITIVE_START_WEIGHT", "ldm_column_positive_start_weight"),
("MINI_LDM_COLUMN_NEGATIVE_START_WEIGHT", "ldm_column_negative_start_weight"),
```

Generate a temporary config using values `true`, `0.03`, `0.0`, then assert:

```python
self.assertIs(generated["ldm"]["decoded_column_curriculum_enabled"], True)
self.assertEqual(generated["ldm"]["decoded_column_positive_start_weight"], 0.03)
self.assertEqual(generated["ldm"]["decoded_column_negative_start_weight"], 0.0)
```

Expected RED: these environment variables are not parsed or written.

- [ ] **Step 2: Implement mini YAML wiring**

Add defaults near existing column variables:

```bash
MINI_LDM_COLUMN_CURRICULUM_ENABLED="${MINI_LDM_COLUMN_CURRICULUM_ENABLED:-false}"
MINI_LDM_COLUMN_POSITIVE_START_WEIGHT="${MINI_LDM_COLUMN_POSITIVE_START_WEIGHT:-0.0}"
MINI_LDM_COLUMN_NEGATIVE_START_WEIGHT="${MINI_LDM_COLUMN_NEGATIVE_START_WEIGHT:-0.0}"
```

Pass them to the embedded Python config generator, parse the enable flag with the same accepted
boolean strings as `MINI_USE_AUG`, and write the three `decoded_column_*` YAML keys. Reject any
other boolean text with `SystemExit` rather than silently treating it as false.

- [ ] **Step 3: Write failing V11 runner tests**

Extend `LDMZ64V10ColumnExperimentTest` so the fake training call records the three curriculum
variables. Assert `V11` produces:

```text
enabled=true
positive_start=0.03
positive_final=0.02
negative_start=0.0
negative_final=0.01
epochs=3
```

Compare every recorded non-curriculum field with A. Also assert the default V11 output is
`test/result/ldm/ablation/ldm_near40_500_z64_column_curriculum_v11_screen`, unknown variants still
fail before writes, and the script contains no inference, evaluation, visualization, or CD command.

- [ ] **Step 4: Add V11 to the existing guarded runner**

Add one case without creating a duplicate runner:

```bash
V11)
  DEFAULT_EXP_DIR="${ROOT_DIR}/test/result/ldm/ablation/ldm_near40_500_z64_column_curriculum_v11_screen"
  COLUMN_CURRICULUM_ENABLED="true"
  COLUMN_POSITIVE_START_WEIGHT="0.03"
  COLUMN_POSITIVE_WEIGHT="0.02"
  COLUMN_NEGATIVE_START_WEIGHT="0.0"
  COLUMN_NEGATIVE_WEIGHT="0.01"
  ;;
```

For A-D, set curriculum disabled and start weights equal to final weights. Export the three new
mini variables before calling `run_ldm_vertical_experiment.sh`. Preserve all existing safety checks
and explicitly fixed reproducibility protocol.

- [ ] **Step 5: Run shell protocol tests**

```bash
conda run -n Radar-Diffusion python test/unit/test_mini_train_script.py
bash -n test/mini-test/train_minimal.sh
bash -n test/mini-test/run_ldm_z64_v10_column_experiment.sh
```

Expected: all tests and syntax checks pass; no real training starts.

- [ ] **Step 6: Run one bounded finite-gradient smoke**

Use a fresh `/tmp` experiment directory, two frames, one epoch, and the audited Z64 VAE checkpoint.
Override the runner only through the underlying mini entry because the formal V11 runner
intentionally fixes 500 frames/3 epochs. First verify that `SMOKE_DIR` does not exist, then prepare
only the VAE file required by LDM mode:

```bash
SMOKE_DIR=/tmp/radar_v11_curriculum_smoke_20260713
test ! -e "${SMOKE_DIR}"
mkdir -p "${SMOKE_DIR}/vae"
cp -a test/result/vae/reconstruction/vae_near40_500_z64_upper_bound/vae/vae_best.pt \
  "${SMOKE_DIR}/vae/vae_best.pt"

MINI_DATASET_DIR="${SMOKE_DIR}/data" \
MINI_CONFIG_PATH="${SMOKE_DIR}/config.yaml" \
MINI_RESULTS_DIR="${SMOKE_DIR}" \
SAMPLES_PER_SCENE=2 MINI_LDM_EPOCHS=1 MINI_NUM_WORKERS=0 \
MINI_TARGET_SIZE=64,128,128 \
MINI_SOURCE_PC_RANGE=0,-20,-6,120,20,10 \
MINI_MODEL_PC_RANGE=0,-20,-6,40,20,10 \
MINI_LDM_COLUMN_CURRICULUM_ENABLED=true \
MINI_LDM_COLUMN_POSITIVE_START_WEIGHT=0.03 \
MINI_LDM_COLUMN_POSITIVE_WEIGHT=0.02 \
MINI_LDM_COLUMN_NEGATIVE_START_WEIGHT=0.0 \
MINI_LDM_COLUMN_NEGATIVE_WEIGHT=0.01 \
bash test/mini-test/train_minimal.sh ldm
```

Expected: one finite epoch, effective weights `0.03/0.0` in CSV/checkpoint, no inference/CD. The
source VAE remains unchanged and the copied checkpoint lives only in the fresh `/tmp` smoke tree.

### Task 4: Final verification and project records

**Files:**
- Modify: `TODO/findings.md`
- Modify: `TODO/task_plan.md`
- Modify: `TODO/progress.md`
- Optionally Modify: `test/README.md` only if the existing runner command is documented there

- [ ] **Step 1: Run the focused regression suite**

```bash
conda run -n Radar-Diffusion python test/unit/test_ldm_vertical_structure_loss.py
conda run -n Radar-Diffusion python test/unit/test_vae_checkpoint_protocol.py
conda run -n Radar-Diffusion python test/unit/test_mini_train_script.py
conda run -n Radar-Diffusion python -m compileall test
conda run -n Radar-Diffusion python -m py_compile diffusion_consistency_radar/scripts/unified_train.py
bash -n test/mini-test/train_minimal.sh
bash -n test/mini-test/run_ldm_z64_v10_column_experiment.sh
git diff --check
```

Expected: all tests, compilation, Shell syntax, and whitespace checks pass.

- [ ] **Step 2: Record supervision and voxel-count impact**

Append to project TODO files:

```text
v11 changes only the epoch timing of weighted positive/negative column supervision.
Targets, occupied-target voxel counts, Z64 tensor size, VAE/model architecture, and inference
threshold protocol are unchanged. Early prediction density may rise by design; the fixed 32-frame
count-ratio gate remains the acceptance control.
```

- [ ] **Step 3: Report Git scope without committing unrelated staged changes**

Run:

```bash
git status --short
git diff --stat
```

Report all modified/new files. Do not stage or commit while unrelated test-directory moves remain
in the index.

- [ ] **Step 4: Stop before formal training**

Do not automatically execute the 500-frame/3-epoch V11 runner. Provide the exact command only after
the implementation and smoke checks pass:

```bash
V10_VARIANT=V11 \
PYTHON_BIN=/home/zxj/anaconda3/envs/Radar-Diffusion/bin/python \
bash test/mini-test/run_ldm_z64_v10_column_experiment.sh
```

The user must explicitly start this formal experiment. After it finishes, run the unchanged fixed
32-frame selector before any 500-frame test or CD.
