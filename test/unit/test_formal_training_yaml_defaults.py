#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证正式训练 YAML 默认值、阶段帧选择及恢复身份合同。"""

import os
import subprocess
import sys
import unittest

import yaml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffusion_consistency_radar.checkpoint_chain import (
    CheckpointChainError,
    assert_checkpoint_training_identity,
    build_formal_stage_training_selection,
)


class FormalTrainingYamlDefaultsTest(unittest.TestCase):
    """默认配置应足以启动正式训练，环境变量只负责临时覆盖。"""

    @staticmethod
    def _data_protocol():
        return {
            "protocol": "formal_data_v2",
            "dataset_manifest_sha256": {"garden": "a" * 64},
            "split_artifact_sha256": "b" * 64,
            "target_policy_sha256": {"garden": "c" * 64},
            "observed_mask_sha256": {"garden": "d" * 64},
            "observed_mask_protocol": "lidar_ray_observed_v1",
            "calibration_sha256": {
                "lidar_to_thermal": "e" * 64,
                "thermal_intrinsics": "f" * 64,
            },
            "radar_ir_sync_sha256": {"garden": "1" * 64},
        }

    def test_default_yaml_declares_each_stage_and_default_devices(self):
        path = os.path.join(
            ROOT, "diffusion_consistency_radar/config/default_config.yaml"
        )
        with open(path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        self.assertEqual(config["hardware"]["cuda_devices"], "0,1")
        self.assertNotIn("num_gpus", config["hardware"])
        self.assertEqual(
            config["data"]["radar_normalization_sha256"],
            "11f59d84cc186c39256c112154faf458ec9ead5fec9b08b997abd5058b68e97c",
        )
        self.assertEqual(config["vae"]["occupancy_loss_type"], "bce_dice")
        self.assertEqual(config["vae"]["occupancy_bce_weight"], 1.0)
        self.assertEqual(config["vae"]["occupancy_dice_weight"], 1.0)
        self.assertEqual(config["vae"]["occupancy_pos_weight_cap"], 128.0)
        self.assertEqual(config["vae"]["continuous_recon_weight"], 1.0)
        for legacy_only_key in (
            "occupied_weight",
            "empty_weight",
            "channel_weights",
            "false_positive_weight",
            "occupancy_mass_weight",
        ):
            with self.subTest(legacy_only_key=legacy_only_key):
                self.assertNotIn(legacy_only_key, config["vae"])
        for stage in ("vae", "ldm", "cd"):
            with self.subTest(stage=stage):
                self.assertEqual(config[stage]["epochs"], 20)
                self.assertEqual(config[stage]["train_frames_per_epoch"], 3210)
                self.assertEqual(
                    config[stage]["validation_frames_per_epoch"], 774
                )
        self.assertEqual(config["cd"]["initialization_model_path"], "")
        self.assertEqual(config["cd"]["teacher_model_path"], "")
        self.assertEqual(
            config["cd"]["training_semantics"],
            "ldm_initialized_ema_consistency_v1",
        )
        for key, expected in (
            ("num_scales", 40),
            ("ema_rate", 0.999),
            ("sigma_min", 0.002),
            ("sigma_max", 80.0),
            ("rho", 7.0),
        ):
            self.assertEqual(config["cd"][key], expected)
        for unused_key in (
            "training_mode",
            "target_ema_mode",
            "start_ema",
            "scale_mode",
            "start_scales",
            "end_scales",
            "distill_steps_per_iter",
            "loss_norm",
        ):
            self.assertNotIn(unused_key, config["cd"])

    def test_stage_selection_hashes_actual_ordered_frame_ids(self):
        selection = build_formal_stage_training_selection(
            stage="vae",
            train_frame_ids_by_scene={"garden": ["000000", "000002"]},
            validation_frame_ids_by_scene={"garden": ["000010"]},
            configured_train_frames_per_scene=2,
            configured_validation_frames_per_scene=1,
        )

        self.assertEqual(selection["protocol"], "formal_stage_selection_v1")
        self.assertEqual(selection["strategy"], "ordered_prefix_per_scene")
        self.assertEqual(selection["train_frame_count_by_scene"], {"garden": 2})
        self.assertEqual(
            selection["validation_frame_count_by_scene"], {"garden": 1}
        )
        self.assertEqual(len(selection["train_frame_ids_sha256"]["garden"]), 64)

    def test_resume_rejects_changed_stage_selection(self):
        expected = build_formal_stage_training_selection(
            stage="ldm",
            train_frame_ids_by_scene={"garden": ["000000", "000001"]},
            validation_frame_ids_by_scene={"garden": ["000010"]},
            configured_train_frames_per_scene=2,
            configured_validation_frames_per_scene=1,
        )
        changed = build_formal_stage_training_selection(
            stage="ldm",
            train_frame_ids_by_scene={"garden": ["000000"]},
            validation_frame_ids_by_scene={"garden": ["000010"]},
            configured_train_frames_per_scene=1,
            configured_validation_frames_per_scene=1,
        )
        checkpoint = {
            "stage": "ldm",
            "checkpoint_protocol": "formal_chain_v2",
            "data_protocol": self._data_protocol(),
            "stage_training_selection": changed,
        }

        with self.assertRaisesRegex(CheckpointChainError, "stage_training_selection"):
            assert_checkpoint_training_identity(
                checkpoint,
                expected_stage="ldm",
                checkpoint_protocol="formal_chain_v2",
                data_protocol=self._data_protocol(),
                stage_training_selection=expected,
            )

    def test_stage_environment_override_precedes_common_and_yaml(self):
        """阶段专用值应盖过 FORMAL 通用值，通用值应盖过 YAML。"""
        launcher = os.path.join(
            ROOT, "diffusion_consistency_radar/launch/train_unified.sh"
        )
        env = os.environ.copy()
        for name in (
            "VAE_EPOCHS",
            "LDM_EPOCHS",
            "CD_EPOCHS",
            "FORMAL_EPOCHS",
        ):
            env.pop(name, None)
        env.update(
            {
                "VAE_EPOCHS": "7",
                "FORMAL_EPOCHS": "0",
                "PREFLIGHT_ONLY": "1",
            }
        )
        result = subprocess.run(
            ["bash", launcher, "vae"],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("LDM_EPOCHS 必须是正整数，实际为 0", result.stdout)
        self.assertNotIn("VAE_EPOCHS 必须是正整数", result.stdout)

    def test_stage_frame_override_precedes_common_and_yaml(self):
        launcher = os.path.join(
            ROOT, "diffusion_consistency_radar/launch/train_unified.sh"
        )
        env = os.environ.copy()
        for name in (
            "VAE_TRAIN_FRAMES_PER_EPOCH",
            "LDM_TRAIN_FRAMES_PER_EPOCH",
            "CD_TRAIN_FRAMES_PER_EPOCH",
            "FORMAL_TRAIN_FRAMES_PER_EPOCH",
        ):
            env.pop(name, None)
        env.update(
            {
                "VAE_TRAIN_FRAMES_PER_EPOCH": "5",
                "FORMAL_TRAIN_FRAMES_PER_EPOCH": "invalid",
                "PREFLIGHT_ONLY": "1",
            }
        )
        result = subprocess.run(
            ["bash", launcher, "vae"],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "LDM_TRAIN_FRAMES_PER_EPOCH 必须是非负整数", result.stdout
        )
        self.assertNotIn(
            "VAE_TRAIN_FRAMES_PER_EPOCH 必须是非负整数", result.stdout
        )


if __name__ == "__main__":
    unittest.main()
