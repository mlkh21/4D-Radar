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
    resolve_cd_initialization_checkpoint,
    resolve_cd_teacher_checkpoint,
)
from diffusion_consistency_radar.scripts.cd_train_optimized import (
    CD_EMA_UPDATE_PROTOCOL,
    CD_VALIDATION_PROTOCOL,
    assert_cd_ema_update_protocol,
    assert_cd_validation_checkpoint_protocol,
    resolve_cd_consistency_config,
    resolve_cd_validation_config,
    select_cd_deployment_weight_source,
)


def _config(path, teacher):
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "cd:\n"
            f"  initialization_model_path: \"{teacher}\"\n"
            "  teacher_model_path: \"\"\n"
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


def _threshold_sweep():
    return [
        {
            "threshold": threshold,
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "iou": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
        for threshold in resolve_cd_validation_config({})[
            "threshold_candidates"
        ]
    ]


def _data_protocol():
    return {
        "protocol": "formal_data_v2",
        "dataset_manifest_sha256": {"garden": "a" * 64},
        "split_artifact_sha256": "b" * 64,
        "target_policy_sha256": {"garden": "c" * 64},
        "observed_mask_sha256": {"garden": "d" * 64},
        "observed_mask_protocol": "lidar_ray_v2",
        "calibration_sha256": {
            "lidar_to_thermal": "e" * 64,
            "thermal_intrinsics": "f" * 64,
        },
        "radar_ir_sync_sha256": {"garden": "1" * 64},
    }


def test_standalone_cd_treats_formal_mini_as_strict_and_binds_selection():
    """独立 CD 入口不能让 mini-v2 绕过正式门禁或丢失子集身份。"""
    from diffusion_consistency_radar.scripts.cd_train_optimized import (
        is_formal_cd_training,
        prepare_cd_data_protocol,
    )

    data_config = {
        "mini_train_frames_per_scene": 8,
        "mini_validation_frames_per_scene": 4,
    }
    resolved = prepare_cd_data_protocol(
        _data_protocol(),
        data_config,
        checkpoint_protocol="formal_mini_chain_v2",
    )

    assert is_formal_cd_training("formal_mini_chain_v2", False) is True
    assert resolved["mini_selection"] == {
        "protocol": "formal_mini_selection_v1",
        "strategy": "ordered_prefix_per_scene",
        "train_frames_per_scene": 8,
        "validation_frames_per_scene": 4,
    }


def test_standalone_full_cd_rejects_hidden_mini_limits():
    """全量正式 CD 配置不能悄悄携带 mini 截断参数。"""
    from diffusion_consistency_radar.scripts.cd_train_optimized import (
        prepare_cd_data_protocol,
    )

    try:
        prepare_cd_data_protocol(
            _data_protocol(),
            {"mini_train_frames_per_scene": 8},
            checkpoint_protocol="formal_chain_v2",
        )
    except ValueError as exc:
        assert "禁止隐式截断" in str(exc)
    else:
        raise AssertionError("正式全量 CD 未拒绝 mini 截断参数")


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


def test_cd_initialization_path_rejects_conflicting_legacy_alias():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "config.yaml")
        with open(cfg_path, "w", encoding="utf-8") as handle:
            handle.write(
                "cd:\n"
                "  initialization_model_path: /new.pt\n"
                "  teacher_model_path: /old.pt\n"
            )
        config = ConfigManager(cfg_path)
        try:
            resolve_cd_initialization_checkpoint("", config)
        except ValueError as exc:
            assert "冲突" in str(exc)
        else:
            raise AssertionError("CD 初始化路径冲突未被拒绝")


def test_cd_consistency_config_exposes_only_active_hyperparameters():
    resolved = resolve_cd_consistency_config(
        {
            "training_semantics": "ldm_initialized_ema_consistency_v1",
            "num_scales": 32,
            "ema_rate": 0.995,
            "sigma_min": 0.01,
            "sigma_max": 20.0,
            "rho": 5.0,
        }
    )
    assert resolved["protocol"] == "ema_consistency_training_config_v1"
    assert resolved["denoising_parameterization"] == "direct_x0_sigma_conditioned_v1"
    assert resolved["consistency_target_source"] == "cd_model_ema"
    assert resolved["num_scales"] == 32
    assert resolved["ema_rate"] == 0.995
    try:
        resolve_cd_consistency_config({"num_scales": 4.5})
    except ValueError as exc:
        assert "num_scales" in str(exc)
    else:
        raise AssertionError("fractional num_scales must be rejected")


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
    trainer.data_protocol = _data_protocol()
    trainer.last_validation_metrics = {
        "model_state_dict": {
            "denoising_latent_loss": 0.3,
            "denoising_occupancy_iou": 0.6,
        },
        "ema_model_state_dict": {
            "denoising_latent_loss": 0.2,
            "denoising_occupancy_iou": 0.7,
        },
    }
    trainer.deployment_weight_source = "ema_model_state_dict"
    trainer.validation_config = resolve_cd_validation_config({})
    trainer.best_val_loss = 0.2
    trainer.best_val_iou = 0.7
    trainer.last_validation_threshold_sweeps = {
        "model_state_dict": _threshold_sweep(),
        "ema_model_state_dict": _threshold_sweep(),
    }

    payload = trainer._checkpoint_payload(epoch=1, loss=0.2, best_loss=0.2)

    assert payload["checkpoint_protocol"] == "formal_chain_v2"
    assert payload["data_protocol"] == _data_protocol()
    assert payload["stage"] == "cd"
    assert payload["vae_checkpoint_sha256"] == "b" * 64
    assert payload["ldm_checkpoint_sha256"] == "c" * 64
    assert payload["model_family"] == "multimodal"
    assert payload["radar_normalization"] == _radar_normalization_spec()
    assert payload["radar_normalization_sha256"] == "d" * 64
    assert payload["training_semantics"] == "ldm_initialized_ema_consistency_v1"
    assert payload["ldm_role"] == "initialization_checkpoint"
    assert payload["consistency_target_source"] == "cd_model_ema"
    assert payload["denoising_parameterization"] == "direct_x0_sigma_conditioned_v1"
    assert payload["consistency_training_config"] == resolve_cd_consistency_config({})
    assert payload["ema_update_protocol"] == CD_EMA_UPDATE_PROTOCOL
    assert payload["deployment_weight_source"] == "ema_model_state_dict"
    assert payload["cd_validation"]["protocol"] == CD_VALIDATION_PROTOCOL

    trainer.checkpoint_protocol = "formal_mini_chain_v2"
    mini_payload = trainer._checkpoint_payload(epoch=1, loss=0.2, best_loss=0.2)
    assert mini_payload["checkpoint_protocol"] == "formal_mini_chain_v2"


def test_cd_ema_updates_parameters_and_batchnorm_buffers_by_name():
    """EMA 必须覆盖 BN 浮点统计和整数计数，不能只更新 parameters。"""
    trainer = ConsistencyDistillationTrainer.__new__(ConsistencyDistillationTrainer)
    trainer.cd_model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 1, 1, bias=False),
        torch.nn.BatchNorm2d(1),
    )
    trainer.cd_model_ema = torch.nn.Sequential(
        torch.nn.Conv2d(1, 1, 1, bias=False),
        torch.nn.BatchNorm2d(1),
    )
    with torch.no_grad():
        for parameter in trainer.cd_model.parameters():
            parameter.fill_(2.0)
        for parameter in trainer.cd_model_ema.parameters():
            parameter.zero_()
        trainer.cd_model[1].running_mean.fill_(4.0)
        trainer.cd_model[1].running_var.fill_(6.0)
        trainer.cd_model[1].num_batches_tracked.fill_(7)
        trainer.cd_model_ema[1].running_mean.zero_()
        trainer.cd_model_ema[1].running_var.zero_()
        trainer.cd_model_ema[1].num_batches_tracked.fill_(1)

    trainer._update_ema(ema_rate=0.5)

    for parameter in trainer.cd_model_ema.parameters():
        assert torch.equal(parameter, torch.ones_like(parameter))
    assert torch.equal(
        trainer.cd_model_ema[1].running_mean,
        torch.full_like(trainer.cd_model_ema[1].running_mean, 2.0),
    )
    assert torch.equal(
        trainer.cd_model_ema[1].running_var,
        torch.full_like(trainer.cd_model_ema[1].running_var, 3.0),
    )
    assert trainer.cd_model_ema[1].num_batches_tracked.item() == 7


def test_cd_ema_rejects_state_name_mismatch():
    trainer = ConsistencyDistillationTrainer.__new__(ConsistencyDistillationTrainer)
    trainer.cd_model = torch.nn.Sequential(torch.nn.Linear(1, 1))
    trainer.cd_model_ema = torch.nn.Sequential(
        torch.nn.Linear(1, 1),
        torch.nn.BatchNorm1d(1),
    )

    try:
        trainer._update_ema(ema_rate=0.9)
    except RuntimeError as exc:
        assert "EMA" in str(exc) and "不一致" in str(exc)
    else:
        raise AssertionError("EMA 更新必须拒绝 online/target 状态名称不一致")


def test_formal_cd_resume_requires_matching_ema_update_protocol():
    assert_cd_ema_update_protocol(
        {"ema_update_protocol": CD_EMA_UPDATE_PROTOCOL},
        require_formal=True,
    )
    assert_cd_ema_update_protocol({}, require_formal=False)

    for checkpoint in (
        {},
        {"ema_update_protocol": "parameters_only_v0"},
    ):
        try:
            assert_cd_ema_update_protocol(checkpoint, require_formal=True)
        except ValueError as exc:
            assert "EMA update protocol" in str(exc)
        else:
            raise AssertionError("formal CD resume 必须拒绝旧 EMA 轨迹")


def test_cd_deployment_selector_uses_iou_then_latent_loss_and_prefers_ema_tie():
    online = {
        "denoising_latent_loss": 0.1,
        "denoising_occupancy_iou": 0.6,
    }
    ema = {
        "denoising_latent_loss": 0.2,
        "denoising_occupancy_iou": 0.7,
    }
    assert select_cd_deployment_weight_source(online, ema) == "ema_model_state_dict"

    ema["denoising_occupancy_iou"] = 0.6
    assert select_cd_deployment_weight_source(online, ema) == "model_state_dict"

    ema["denoising_latent_loss"] = 0.1
    assert select_cd_deployment_weight_source(online, ema) == "ema_model_state_dict"


def test_formal_cd_resume_requires_matching_validation_selection_protocol():
    trainer = ConsistencyDistillationTrainer.__new__(ConsistencyDistillationTrainer)
    trainer.cd_model = torch.nn.Linear(2, 2)
    trainer.cd_model_ema = torch.nn.Linear(2, 2)
    trainer.optimizer = torch.optim.AdamW(trainer.cd_model.parameters())
    trainer.model_config = {"latent_dim": 1}
    trainer.data_grid_config = {}
    trainer.use_multimodal = True
    trainer.vae_checkpoint_sha256 = "b" * 64
    trainer.ldm_checkpoint_sha256 = "c" * 64
    trainer.radar_normalization = _radar_normalization_spec()
    trainer.radar_normalization_sha256 = "d" * 64
    trainer.data_protocol = _data_protocol()
    trainer.validation_config = resolve_cd_validation_config({})
    trainer.last_validation_metrics = {
        "model_state_dict": {
            "denoising_latent_loss": 0.3,
            "denoising_occupancy_iou": 0.6,
        },
        "ema_model_state_dict": {
            "denoising_latent_loss": 0.2,
            "denoising_occupancy_iou": 0.7,
        },
    }
    trainer.deployment_weight_source = "ema_model_state_dict"
    trainer.best_val_loss = 0.2
    trainer.best_val_iou = 0.7
    trainer.last_validation_threshold_sweeps = {
        "model_state_dict": _threshold_sweep(),
        "ema_model_state_dict": _threshold_sweep(),
    }
    payload = trainer._checkpoint_payload(1, 0.1, 0.1)

    restored = assert_cd_validation_checkpoint_protocol(
        payload,
        current_config=trainer.validation_config,
        require_formal=True,
    )
    assert restored["selected_source"] == "ema_model_state_dict"

    payload["cd_validation"]["seed"] = 43
    try:
        assert_cd_validation_checkpoint_protocol(
            payload,
            current_config=trainer.validation_config,
            require_formal=True,
        )
    except ValueError as exc:
        assert "seed" in str(exc)
    else:
        raise AssertionError("formal CD resume 必须拒绝 validation seed 漂移")

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


def test_standalone_formal_cd_requires_real_persisted_inputs():
    """独立 CD 不能因配置漏项退化到 mock 标定或运行时 observed。"""
    from diffusion_consistency_radar.scripts.cd_train_optimized import (
        assert_formal_cd_data_config,
    )

    formal_data = {
        "scene_names": ["garden"],
        "require_real_ir": True,
        "require_real_calibration": True,
        "require_persisted_observed_mask": True,
        "require_radar_statistics": True,
        "voxel_coordinate_frame": "lidar",
    }
    assert_formal_cd_data_config(formal_data)

    for key in (
        "require_real_ir",
        "require_real_calibration",
        "require_persisted_observed_mask",
        "require_radar_statistics",
    ):
        invalid = dict(formal_data)
        invalid[key] = False
        try:
            assert_formal_cd_data_config(invalid)
        except ValueError as exc:
            assert key in str(exc)
        else:
            raise AssertionError(f"正式 CD 必须拒绝 {key}=false")

    missing_scene = dict(formal_data)
    missing_scene["scene_names"] = []
    try:
        assert_formal_cd_data_config(missing_scene)
    except ValueError as exc:
        assert "scene_names" in str(exc)
    else:
        raise AssertionError("正式 CD 必须拒绝空 scene_names")


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
    test_cd_initialization_path_rejects_conflicting_legacy_alias()
    test_cd_consistency_config_exposes_only_active_hyperparameters()
    test_cd_checkpoint_payload_contains_chain_metadata()
    test_formal_cd_resume_requires_matching_ema_update_protocol()
    test_cd_deployment_selector_uses_iou_then_latent_loss_and_prefers_ema_tie()
    test_formal_cd_resume_requires_matching_validation_selection_protocol()
    test_cd_preflight_requires_teacher_and_config_normalization_match()
    test_standalone_formal_cd_requires_real_persisted_inputs()
    test_training_scripts_help_without_repository_working_directory()
    print("test_cd_training_entrypoints passed")
