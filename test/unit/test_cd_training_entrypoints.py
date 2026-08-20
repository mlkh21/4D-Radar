#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import tempfile

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion_consistency_radar.scripts.unified_train import (
    ConfigManager,
    ConsistencyDistillationTrainer,
    resolve_cd_teacher_checkpoint,
)


def _config(path, teacher):
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "cd:\n"
            f"  teacher_model_path: \"{teacher}\"\n"
        )


def _radar_normalization_spec():
    return {
        "protocol": "radar_normalization_v1",
        "formal": True,
        "training_scenes": ["garden"],
        "frame_count": 2,
        "target_size": [4, 8, 8],
        "source_pc_range": [0, -4, -2, 16, 4, 2],
        "model_pc_range": [0, -2, -1, 8, 2, 1],
        "intensity": {
            "transform": "log1p_robust_zscore",
            "log_median": 1.0,
            "log_iqr": 2.0,
            "clip": [-5.0, 5.0],
        },
        "doppler": {
            "transform": "symmetric_physical_scale",
            "scale_mps": 4.0,
            "clip": [-1.0, 1.0],
        },
        "variance": {
            "transform": "identity",
            "unit": "m2_s2",
            "aggregation": "occupied_voxel_equal_weight_total_variance",
        },
        "input_provenance": {
            "dataset_manifest_sha256": {"garden": "f" * 64},
        },
    }


def test_cd_teacher_checkpoint_prefers_cli_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "config.yaml")
        _config(cfg_path, "/from/config.pt")
        config = ConfigManager(cfg_path)
        assert resolve_cd_teacher_checkpoint("/from/cli.pt", config) == "/from/cli.pt"


def test_cd_teacher_checkpoint_falls_back_to_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "config.yaml")
        _config(cfg_path, "/from/config.pt")
        config = ConfigManager(cfg_path)
        assert resolve_cd_teacher_checkpoint("", config) == "/from/config.pt"


def test_cd_checkpoint_payload_contains_chain_metadata():
    trainer = ConsistencyDistillationTrainer.__new__(ConsistencyDistillationTrainer)
    trainer.cd_model = torch.nn.Linear(2, 2)
    trainer.cd_model_ema = torch.nn.Linear(2, 2)
    trainer.optimizer = torch.optim.AdamW(trainer.cd_model.parameters())
    trainer.model_config = {
        "latent_dim": 4,
        "in_channels": 16,
        "out_channels": 4,
        "fusion_voxel_shape": [4, 8, 8],
        "fusion_latent_shape": [2, 4, 4],
        "fusion_pc_range": [0, -2, -1, 8, 2, 1],
    }
    trainer.data_grid_config = {
        "target_size": [4, 8, 8],
        "source_pc_range": [0, -4, -2, 16, 4, 2],
        "model_pc_range": [0, -2, -1, 8, 2, 1],
    }
    trainer.use_multimodal = True
    trainer.vae_checkpoint_sha256 = "b" * 64
    trainer.ldm_checkpoint_sha256 = "c" * 64
    trainer.radar_normalization = _radar_normalization_spec()
    trainer.radar_normalization_sha256 = "d" * 64

    payload = trainer._checkpoint_payload(epoch=1, loss=0.2, best_loss=0.2)

    assert payload["checkpoint_protocol"] == "formal_chain_v1"
    assert payload["stage"] == "cd"
    assert payload["vae_checkpoint_sha256"] == "b" * 64
    assert payload["ldm_checkpoint_sha256"] == "c" * 64
    assert payload["model_family"] == "multimodal"
    assert payload["radar_normalization"] == _radar_normalization_spec()
    assert payload["radar_normalization_sha256"] == "d" * 64
    assert payload["training_semantics"] == "ldm_initialized_ema_consistency_v1"
    assert payload["ldm_role"] == "initialization_checkpoint"
    assert payload["consistency_target_source"] == "cd_model_ema"

    trainer.checkpoint_protocol = "formal_mini_chain_v1"
    mini_payload = trainer._checkpoint_payload(epoch=1, loss=0.2, best_loss=0.2)
    assert mini_payload["checkpoint_protocol"] == "formal_mini_chain_v1"


def test_cd_preflight_requires_teacher_and_config_normalization_match():
    from diffusion_consistency_radar.radar_normalization import (
        RadarNormalizationError,
    )
    from diffusion_consistency_radar.scripts.cd_train_optimized import (
        resolve_cd_radar_normalization,
    )

    spec = _radar_normalization_spec()
    checkpoint = {
        "radar_normalization": spec,
        "radar_normalization_sha256": "d" * 64,
    }
    resolved, digest = resolve_cd_radar_normalization(
        checkpoint,
        spec,
        "d" * 64,
        data_grid_config={
            "target_size": [4, 8, 8],
            "source_pc_range": [0, -4, -2, 16, 4, 2],
            "model_pc_range": [0, -2, -1, 8, 2, 1],
        },
    )
    assert resolved == spec
    assert digest == "d" * 64

    changed = dict(spec)
    changed["frame_count"] = 3
    try:
        resolve_cd_radar_normalization(
            checkpoint,
            changed,
            "d" * 64,
            data_grid_config={
                "target_size": [4, 8, 8],
                "source_pc_range": [0, -4, -2, 16, 4, 2],
                "model_pc_range": [0, -2, -1, 8, 2, 1],
            },
        )
    except RadarNormalizationError as exc:
        assert "spec" in str(exc) or "内容" in str(exc)
    else:
        raise AssertionError("CD must reject teacher/config normalization mismatch")


def test_training_scripts_help_without_repository_working_directory():
    """直接执行训练脚本时不应隐式依赖仓库工作目录或 PYTHONPATH。"""
    scripts_dir = os.path.join(PROJECT_ROOT, "diffusion_consistency_radar", "scripts")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"

    with tempfile.TemporaryDirectory() as tmpdir:
        for script_name in ("unified_train.py", "cd_train_optimized.py"):
            result = subprocess.run(
                [sys.executable, os.path.join(scripts_dir, script_name), "--help"],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            assert result.returncode == 0, (
                f"{script_name} 直接入口失败:\n{result.stdout}\n{result.stderr}"
            )


if __name__ == "__main__":
    test_cd_teacher_checkpoint_prefers_cli_path()
    test_cd_teacher_checkpoint_falls_back_to_config()
    test_cd_checkpoint_payload_contains_chain_metadata()
    test_cd_preflight_requires_teacher_and_config_normalization_match()
    test_training_scripts_help_without_repository_working_directory()
    print("test_cd_training_entrypoints passed")
