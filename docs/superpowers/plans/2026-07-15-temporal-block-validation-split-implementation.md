# Temporal Block Validation Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace frame-level random train/validation membership with ordered `garden` prefix/tail blocks while preserving `loop3` as the independent test scene.

**Architecture:** Add one pure temporal split helper in the existing unified training module and route the existing `Subset` construction through it. Keep the two Dataset instances, augmentation behavior, sample counts, targets, model, checkpoint schema, launchers, and mini protocol unchanged.

**Tech Stack:** Python 3.8, PyTorch `Dataset`/`Subset`, `unittest`, Conda environment `Radar-Diffusion`, Git selective staging.

---

## File Structure and Safety Boundary

- Modify: `diffusion_consistency_radar/scripts/unified_train.py:166-184,2105-2109`
  - Owns the pure index split and the formal training-entry wiring.
  - Preserve every pre-existing V11 curriculum hunk in this already-dirty file.
- Modify: `test/unit/test_vae_checkpoint_protocol.py:27-36,213-228`
  - Owns the split contract tests; do not create another test file.
- Modify: `TODO/task_plan.md`, `TODO/findings.md`, `TODO/progress.md`
  - Record RED/GREEN evidence, protocol impact, and final verification.
- Do not modify: `diffusion_consistency_radar/cm/dataset_loader.py`, launch scripts, mini runners,
  YAML config, target generation, model code, checkpoint code, or result directories.

### Task 1: RED — Define the Temporal Block Contract

**Files:**
- Modify: `test/unit/test_vae_checkpoint_protocol.py:27-36`
- Modify: `test/unit/test_vae_checkpoint_protocol.py:213-228`

- [ ] **Step 1: Add a module-level lookup for the not-yet-implemented API**

Add this import beside the existing unified-train imports without removing the current
`deterministic_split_indices` import yet:

```python
from diffusion_consistency_radar.scripts import unified_train as unified_train_module
```

Replace the two existing split tests with the following tests. `getattr()` plus `assertIsNotNone()`
ensures RED is an assertion failure rather than an import error:

```python
    def _temporal_block_split(self, dataset_size, train_split=0.8):
        split_fn = getattr(
            unified_train_module,
            "temporal_block_split_indices",
            None,
        )
        self.assertIsNotNone(
            split_fn,
            "unified_train.temporal_block_split_indices 尚未实现",
        )
        return split_fn(dataset_size, train_split=train_split)

    def test_temporal_block_split_returns_ordered_prefix_and_suffix(self):
        train_indices, val_indices = self._temporal_block_split(10, train_split=0.8)

        self.assertEqual(train_indices, list(range(8)))
        self.assertEqual(val_indices, [8, 9])

    def test_temporal_block_split_is_disjoint_complete_and_rng_independent(self):
        seed_training_run(7)
        first = self._temporal_block_split(10, train_split=0.6)
        seed_training_run(99)
        second = self._temporal_block_split(10, train_split=0.6)

        self.assertEqual(first, second)
        train_indices, val_indices = first
        self.assertFalse(set(train_indices) & set(val_indices))
        self.assertEqual(set(train_indices) | set(val_indices), set(range(10)))
        self.assertEqual(train_indices, list(range(6)))
        self.assertEqual(val_indices, list(range(6, 10)))

    def test_temporal_block_split_rejects_invalid_or_empty_partitions(self):
        with self.assertRaisesRegex(ValueError, "至少需要 2"):
            self._temporal_block_split(1, train_split=0.8)
        with self.assertRaisesRegex(ValueError, "train_split"):
            self._temporal_block_split(10, train_split=1.0)
        with self.assertRaisesRegex(ValueError, "空划分"):
            self._temporal_block_split(2, train_split=0.1)
```

- [ ] **Step 2: Run the focused test file and verify RED**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_vae_checkpoint_protocol.py -v
```

Expected: the three new temporal-split tests fail with
`unified_train.temporal_block_split_indices 尚未实现`; existing unrelated tests remain passing.
If the Conda import hits the known OpenMPI socket denial, rerun the same test with the environment
Python outside the socket-restricted sandbox; do not change test logic or skip RED.

- [ ] **Step 3: Record RED evidence**

Append the exact test command, failing test names, and expected missing-API reason to
`TODO/progress.md`. Add the confirmed behavioral contract to `TODO/findings.md`; do not mark the
implementation checkbox complete.

### Task 2: GREEN — Implement and Wire the Ordered Split

**Files:**
- Modify: `diffusion_consistency_radar/scripts/unified_train.py:166-184`
- Modify: `diffusion_consistency_radar/scripts/unified_train.py:2105-2109`
- Modify: `test/unit/test_vae_checkpoint_protocol.py:27-36,213-250`

- [ ] **Step 1: Replace the random helper with the minimal pure implementation**

Replace `deterministic_split_indices()` with:

```python
def temporal_block_split_indices(
    dataset_size: int,
    train_split: float = 0.8,
) -> Tuple[List[int], List[int]]:
    """按样本时间顺序划分连续的训练前缀和验证后缀。"""
    if dataset_size < 2:
        raise ValueError("训练/验证划分至少需要 2 个样本")
    if not 0.0 < train_split < 1.0:
        raise ValueError("data.train_split 必须严格位于 (0, 1)")
    train_size = int(dataset_size * train_split)
    if train_size <= 0 or train_size >= dataset_size:
        raise ValueError(
            f"train_split={train_split} 导致空划分："
            f"dataset_size={dataset_size}, train_size={train_size}"
        )
    return list(range(train_size)), list(range(train_size, dataset_size))
```

Do not add a seed parameter, random generator, shuffled indices, embargo gap, scene grouping, or
new configuration key.

- [ ] **Step 2: Route formal Subset construction through the new helper**

Replace the main-entry call with:

```python
    train_indices, val_indices = temporal_block_split_indices(
        len(train_dataset_base),
        train_split=float(data_config.get("train_split", 0.8)),
    )
```

Keep `training_seed = int(data_config.get("training_seed", data_config.get("split_seed", 42)))`
unchanged so historical configs still control model initialization and DataLoader shuffle.

- [ ] **Step 3: Run the focused test file and verify GREEN**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_vae_checkpoint_protocol.py -v
```

Expected: all tests pass, including the three temporal-block tests. There must be no warning or
failure indicating an empty partition, random membership, model/checkpoint regression, or import
error.

- [ ] **Step 4: Refactor the test import without changing behavior**

In the existing grouped import from `unified_train`, replace:

```python
    deterministic_split_indices,
```

with:

```python
    temporal_block_split_indices,
```

Remove the module alias added for RED, remove `_temporal_block_split()`, and call
`temporal_block_split_indices()` directly in the three tests. Do not alter their assertions.

- [ ] **Step 5: Re-run GREEN after the test-only refactor**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_vae_checkpoint_protocol.py -v
```

Expected: the same full file passes with no changed test count other than the replacement of the
two historical random-split tests by three temporal-block tests.

### Task 3: Verification, Records, and Selective Commit

**Files:**
- Verify: `diffusion_consistency_radar/scripts/unified_train.py`
- Verify: `test/unit/test_vae_checkpoint_protocol.py`
- Modify: `TODO/task_plan.md`
- Modify: `TODO/findings.md`
- Modify: `TODO/progress.md`

- [ ] **Step 1: Run syntax and whitespace verification**

Run:

```bash
conda run -n Radar-Diffusion python -m py_compile \
  diffusion_consistency_radar/scripts/unified_train.py \
  test/unit/test_vae_checkpoint_protocol.py
git diff --check
```

Expected: both commands exit 0 with no syntax or whitespace errors.

- [ ] **Step 2: Inspect the exact scope and preserve dirty V11 work**

Run:

```bash
git diff -- diffusion_consistency_radar/scripts/unified_train.py \
  test/unit/test_vae_checkpoint_protocol.py
git status --short
```

Confirm the new P0-01 hunks are limited to the split helper, main call, imports, and three tests.
Pre-existing curriculum changes in `unified_train.py` must remain byte-for-byte present and unstaged
unless they were already staged before this task.

- [ ] **Step 3: Update the three project records**

Record all of the following:

```text
- RED command and expected missing temporal API failure
- GREEN command and passing count
- production/test files changed
- supervision target and per-frame occupied voxel counts unchanged
- current single-scene 80/20 sample counts unchanged
- validation membership changed from random interleaving to ordered prefix/tail blocks
- historical random-split metrics are not directly comparable to the new protocol
- loop3 remains independent test
- no training, preprocessing, full inference, or full evaluation was run
```

Mark the P0-01 implementation and focused verification checkboxes complete only after all commands
pass.

- [ ] **Step 4: Selectively stage only P0-01 code and tests**

Because `unified_train.py` was dirty before this task, use interactive hunk staging:

```bash
git add -p diffusion_consistency_radar/scripts/unified_train.py
git add test/unit/test_vae_checkpoint_protocol.py
git diff --cached --check
git diff --cached --stat
git diff --cached
```

Accept only the temporal split helper and main-call hunks from `unified_train.py`; reject every V11
curriculum hunk. The cached diff must contain only P0-01 code/tests and no TODO history, V11 runner,
curriculum, audit, result, checkpoint, or dataset file.

- [ ] **Step 5: Commit the isolated implementation**

Run:

```bash
git commit -m "fix: use temporal validation blocks"
```

Expected: one commit containing only `unified_train.py` P0-01 hunks and
`test/unit/test_vae_checkpoint_protocol.py`. If safe selective staging cannot be proven, do not
commit; leave changes unstaged and report the blocker instead of including unrelated work.

- [ ] **Step 6: Report final repository state**

Run:

```bash
git status --short --branch
git diff --stat
git diff --cached --stat
```

Expected: no P0-01 code/test changes remain uncommitted after a successful selective commit; all
pre-existing V11, audit, handoff, TODO, result, checkpoint, and dataset state remains preserved.

## Completion Criteria

- Three temporal-block contract tests were observed failing for the intended missing behavior before
  production modification, then passing after the minimal implementation.
- Formal train/validation membership is an ordered prefix/tail split with the existing 80/20 size.
- Training/DataLoader randomness remains seeded; membership no longer depends on `split_seed`.
- Focused unit tests, `py_compile`, and `git diff --check` pass.
- No long training, preprocessing, full inference, full evaluation, data overwrite, or result overwrite
  occurred.
- TODO records explicitly describe supervision, voxel-count, metric-comparability, and independent-test
  impact.
