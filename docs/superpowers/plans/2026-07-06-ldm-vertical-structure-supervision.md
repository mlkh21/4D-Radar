# LDM Vertical Structure Supervision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add differentiable height-distribution and vertical-continuity supervision to LDM training so tree trunks and canopy height are less likely to disappear during latent denoising.

**Architecture:** Reuse the existing decoded-occupancy auxiliary path in `OptimizedLDMTrainer`. Convert decoded channel 0 to a stable soft occupancy tensor, compare target and prediction along the physical Z tensor axis, and add independently weighted losses without changing model parameters or checkpoint tensor shapes.

**Tech Stack:** Python, PyTorch, unittest, YAML, Bash.

---

### Task 1: Differentiable vertical structure losses

**Files:**
- Modify: `diffusion_consistency_radar/scripts/unified_train.py`
- Create: `test/test_ldm_vertical_structure_loss.py`

- [ ] **Step 1: Write failing tests**

Add synthetic `[B,C,Z,X,Y]` tests that require:

```python
losses = decoded_vertical_structure_losses(
    decoded,
    target,
    occupancy_activation="sigmoid",
)
```

The tests must prove that aligned columns score lower than shifted-height columns, a broken vertical column increases continuity loss, empty targets return finite differentiable zeros, gradients reach decoded occupancy logits, and invalid activation/shape inputs fail clearly.

- [ ] **Step 2: Verify RED**

Run:

```bash
conda run -n Radar-Diffusion python test/test_ldm_vertical_structure_loss.py -v
```

Expected: failure because `decoded_vertical_structure_losses` does not exist.

- [ ] **Step 3: Implement the minimal loss helper**

Implement in `unified_train.py`:

```python
def decoded_vertical_structure_losses(decoded, target, occupancy_activation, eps=1e-6):
    ...
    return {
        "height_distribution_loss": height_loss,
        "vertical_continuity_loss": continuity_loss,
    }
```

Requirements:

- Tensor layout is `[B,C,Z,X,Y]`; Z is dimension 2.
- Sigmoid checkpoints use `torch.sigmoid`; raw checkpoints use a bounded `[0,1]` soft occupancy conversion.
- Height distribution compares per-column normalized Z cumulative distributions only for target-nonempty `(X,Y)` columns.
- Continuity compares adjacent-Z absolute occupancy differences on the same valid columns.
- Empty target returns graph-connected scalar zeros.
- Reductions are computed in float32 for AMP stability.

- [ ] **Step 4: Verify GREEN**

Run the Task 1 test and `py_compile`; both must pass.

### Task 2: Trainer, logging, and configuration integration

**Files:**
- Modify: `diffusion_consistency_radar/scripts/unified_train.py`
- Modify: `diffusion_consistency_radar/config/default_config.yaml`
- Modify: `diffusion_consistency_radar/config/.default_config.train_override.yaml`
- Modify: `test/mini-test/train_minimal.sh`
- Modify: `test/test_ldm_vertical_structure_loss.py`

- [ ] **Step 1: Write failing trainer/config tests**

Cover:

- zero weights preserve the old latent-only/decoded-loss behavior;
- either structure weight triggers exactly one VAE decode;
- the weighted height and continuity terms are added independently;
- epoch component logs expose latent, decoded occupancy, height, continuity, and uncertainty terms;
- mini environment variables reach the generated YAML.

- [ ] **Step 2: Integrate once-per-batch decoding**

Add:

```yaml
ldm:
  decoded_height_distribution_weight: 0.02
  decoded_vertical_continuity_weight: 0.02
```

Code fallbacks remain `0.0` for old configs. Decode `denoised` once whenever any decoded auxiliary is enabled, reuse it for occupancy and structure losses, and keep the current latent MSE as the primary objective.

- [ ] **Step 3: Extend metrics and checkpoints**

Add loss component columns to the LDM CSV and persist the effective LDM loss weights in checkpoint metadata. Resume loading must remain compatible with older checkpoint payloads.

- [ ] **Step 4: Add mini controls**

Support:

```bash
MINI_LDM_HEIGHT_WEIGHT=0.02
MINI_LDM_CONTINUITY_WEIGHT=0.02
```

in `test/mini-test/train_minimal.sh`.

- [ ] **Step 5: Run focused verification**

Run:

```bash
conda run -n Radar-Diffusion python test/test_ldm_vertical_structure_loss.py -v
conda run -n Radar-Diffusion python test/test_multimodal_inference_interface.py -v
conda run -n Radar-Diffusion python -m py_compile \
  diffusion_consistency_radar/scripts/unified_train.py \
  test/test_ldm_vertical_structure_loss.py
git diff --check
```

No long training is part of this task.

### Task 3: One-batch smoke and project records

**Files:**
- Modify: `TODO/task_plan.md`
- Modify: `TODO/findings.md`
- Modify: `TODO/progress.md`
- Output: `test/result/ldm_vertical_structure_smoke/`

- [ ] **Step 1: Run one short LDM smoke**

Use one sample, one epoch, zero workers, and the existing VAE checkpoint through the mini script or a focused trainer fixture. Confirm all losses are finite and one optimizer step completes.

- [ ] **Step 2: Record impact**

Document that enabled runs receive new structure supervision and changed total-loss scale, while target voxel count, model grid, model parameter shapes, and legacy checkpoint loading remain unchanged.

- [ ] **Step 3: Complete two-stage reviews**

Each implementation task requires specification review followed by code-quality review. A final reviewer checks the complete increment before handoff.
