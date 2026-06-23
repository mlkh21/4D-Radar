# Airborne Perception Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the offline airborne radar densification loop before moving to ROS/PX4/HIL integration.

**Architecture:** Keep legacy radar-only checkpoints compatible while routing new LDM/CD training and inference through the multimodal radar+IR metadata interface. Promote task-oriented obstacle-map metrics and uncertainty-aware mapping to first-class offline validation.

**Tech Stack:** PyTorch, NumPy, YAML configs, shell launchers, NTU4DRadLM preprocessed voxels.

---

### Task 1: Stabilize CD Training

**Files:**
- Modify: `diffusion_consistency_radar/scripts/cd_train_optimized.py`
- Modify: `diffusion_consistency_radar/scripts/unified_train.py`
- Test: `test/test_multimodal_cd_training_interface.py`
- Test: `test/test_cd_training_entrypoints.py`

- [x] Route student, EMA, and teacher CD denoising through one `call_cd_denoiser()` helper.
- [x] Preserve legacy 8-channel latent concat checkpoints.
- [x] Support multimodal CD checkpoints with `radar_voxel`, IR image, calibration matrices, and `noised_latent`.
- [x] Add `unified_train.py --mode cd` with `--ldm_ckpt` and config fallback.
- [x] Verify CD interface and entrypoint tests pass.

### Task 2: Complete Mini Offline Loop

**Files:**
- Modify: `test/mini-test/train_minimal.sh`
- Modify: `test/mini-test/inference_minimal.sh`
- Modify: `test/mini-test/run_minimal_experiment.sh`

- [x] Add `cd` and `all_with_cd` mini training modes.
- [x] Store mini CD checkpoints under `test/mini-test/train_results_mini/cd`.
- [x] Add task metrics to mini inference.
- [x] Auto-enable multimodal metadata for multimodal checkpoints.
- [x] Run 2-sample VAE/LDM/CD training smoke and 1-frame LDM/CD inference smoke.

### Task 3: Audit Dataset Protocol

**Files:**
- Create: `diffusion_consistency_radar/scripts/audit_dataset_protocol.py`
- Modify: `test/test_dataset_protocol_metadata.py`

- [x] Report IR coverage, target/radar frame counts, preprocess policy presence, alignment policy, and calibration fallback status.
- [x] Add unit coverage for audit rows.

### Task 4: Add Speed-Aware Mapping Parameters

**Files:**
- Modify: `diffusion_consistency_radar/cm/probabilistic_mapping.py`
- Modify: `diffusion_consistency_radar/scripts/streaming_map_update.py`
- Modify: `test/test_probabilistic_mapping_uncertainty.py`

- [x] Add `speed_m_s` to map config and use it to adjust sliding window, decay, and far-range reliability.
- [x] Add `--speed_m_s` and `--odom_cov_trace` to streaming map update.
- [x] Verify higher speed and odometry uncertainty reduce reliability/belief.

### Task 5: Formal Retraining Follow-Up

**Files:**
- Use: `diffusion_consistency_radar/launch/train_unified.sh`
- Use: `diffusion_consistency_radar/launch/inference_ldm.sh`
- Use: `diffusion_consistency_radar/launch/inference_cd.sh`

- [ ] Regenerate the full sensor-aware dataset.
- [ ] Train formal VAE, LDM, and CD checkpoints.
- [ ] Compare old LDM, new LDM, CD 1-step, and CD 4-step with task metrics.
- [ ] Only after stable offline metrics, design ROS1 service/action and PX4-HIL bridge.
