#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证正式 VAE/LDM/CD checkpoint 链的自描述协议与 fail-closed 行为。"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class CheckpointChainProtocolTest(unittest.TestCase):
    def _sha256(self, path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _grid(self):
        return {
            "target_size": [4, 8, 8],
            "source_pc_range": [0.0, -4.0, -2.0, 16.0, 4.0, 2.0],
            "model_pc_range": [0.0, -2.0, -1.0, 8.0, 2.0, 1.0],
        }

    def _model_config(self):
        return {
            "latent_dim": 4,
            "in_channels": 16,
            "out_channels": 4,
            "model_channels": 4,
            "num_res_blocks": 1,
            "attention_resolutions": [],
            "channel_mult": [1],
            "fusion_voxel_shape": [4, 8, 8],
            "fusion_latent_shape": [2, 4, 4],
            "fusion_pc_range": [0.0, -2.0, -1.0, 8.0, 2.0, 1.0],
        }

    def _radar_normalization(self):
        grid = self._grid()
        return {
            "protocol": "radar_normalization_v1",
            "formal": True,
            "training_scenes": ["unit_scene"],
            "frame_count": 2,
            "target_size": list(grid["target_size"]),
            "source_pc_range": list(grid["source_pc_range"]),
            "model_pc_range": list(grid["model_pc_range"]),
            "intensity": {
                "transform": "log1p_robust_zscore",
                "log_median": 1.0,
                "log_iqr": 2.0,
                "clip": [-4.0, 4.0],
            },
            "doppler": {
                "transform": "symmetric_physical_scale",
                "scale_mps": 12.0,
                "clip": [-1.0, 1.0],
            },
            "variance": {
                "transform": "identity",
                "unit": "m2_s2",
                "aggregation": "occupied_voxel_equal_weight_total_variance",
            },
            "input_provenance": {
                "dataset_manifest_sha256": {"unit_scene": "a" * 64}
            },
        }

    def _data_protocol(self):
        """构造绑定监督、划分、observed、同步和标定身份的最小 v2 协议。"""
        return {
            "protocol": "formal_data_v2",
            "dataset_manifest_sha256": {"unit_scene": "1" * 64},
            "split_artifact_sha256": "2" * 64,
            "target_policy_sha256": {"unit_scene": "3" * 64},
            "observed_mask_protocol": "persisted_lidar_ray_v1",
            "observed_mask_sha256": {"unit_scene": "4" * 64},
            "calibration_sha256": {
                "lidar_to_thermal": "5" * 64,
                "thermal_intrinsics": "6" * 64,
            },
            "radar_ir_sync_sha256": {"unit_scene": "7" * 64},
        }

    def _write_chain(
        self,
        root,
        *,
        cd_legacy=False,
        ldm_grid=None,
        bad_parent=False,
        missing_normalization_stage=None,
        missing_normalization_hash_stage=None,
        mismatched_cd_normalization=False,
        mismatched_cd_normalization_hash=False,
        nonformal_normalization_stage=None,
        vae_has_normalization=False,
        checkpoint_protocol="formal_chain_v2",
        missing_data_protocol_stage=None,
        mismatched_data_protocol_stage=None,
    ):
        from diffusion_consistency_radar.checkpoint_chain import sha256_file

        os.makedirs(root, exist_ok=True)
        grid = self._grid()
        vae_path = os.path.join(root, "vae.pt")
        ldm_path = os.path.join(root, "ldm.pt")
        cd_path = os.path.join(root, "cd.pt")
        vae_payload = {
                "checkpoint_protocol": checkpoint_protocol,
                "stage": "vae",
                "model_state_dict": {"encoder.conv_in.weight": torch.zeros(1)},
                "vae_config": {
                    "in_channels": 4,
                    "out_channels": 4,
                    "latent_dim": 4,
                    "occupancy_loss_type": "bce_dice",
                },
                "vae_config_type": "ultra_lightweight",
                "data_grid_config": grid,
                "occupancy_activation": "sigmoid",
        }
        if missing_data_protocol_stage != "vae":
            vae_payload["data_protocol"] = self._data_protocol()
        if vae_has_normalization:
            vae_payload["radar_normalization"] = self._radar_normalization()
            vae_payload["radar_normalization_sha256"] = "b" * 64
        torch.save(vae_payload, vae_path)
        ldm_grid = dict(ldm_grid or grid)
        ldm_model_config = self._model_config()
        ldm_model_config["fusion_voxel_shape"] = list(ldm_grid["target_size"])
        ldm_model_config["fusion_pc_range"] = list(ldm_grid["model_pc_range"])
        ldm_state = {
            "unet_3d.input_blocks.0.0.weight": torch.zeros(1),
            "radar_encoder.conv.weight": torch.zeros(1),
            "model_uncertainty_head.0.weight": torch.zeros(1),
            "ir_extractor.backbone.0.weight": torch.zeros(1),
            "projection_layer.dummy": torch.zeros(1),
            "fusion_conv.0.weight": torch.zeros(1),
        }
        ldm_payload = {
                "checkpoint_protocol": checkpoint_protocol,
                "stage": "ldm",
                "model_state_dict": ldm_state,
                "latent_dim": 4,
                "model_config": ldm_model_config,
                "data_grid_config": ldm_grid,
                "vae_checkpoint_sha256": sha256_file(vae_path),
        }
        if missing_data_protocol_stage != "ldm":
            ldm_payload["data_protocol"] = self._data_protocol()
            if mismatched_data_protocol_stage == "ldm":
                ldm_payload["data_protocol"]["split_artifact_sha256"] = "8" * 64
        if missing_normalization_stage != "ldm":
            ldm_normalization = self._radar_normalization()
            if nonformal_normalization_stage == "ldm":
                ldm_normalization["formal"] = False
            ldm_payload["radar_normalization"] = ldm_normalization
            if missing_normalization_hash_stage != "ldm":
                ldm_payload["radar_normalization_sha256"] = "b" * 64
        torch.save(ldm_payload, ldm_path)
        cd_state = (
            {
                "input_blocks.0.0.weight": torch.zeros(1),
                "out.2.weight": torch.zeros(1),
            }
            if cd_legacy
            else dict(ldm_state)
        )
        cd_vae_hash = "bad" if bad_parent else sha256_file(vae_path)
        cd_payload = {
                "checkpoint_protocol": checkpoint_protocol,
                "stage": "cd",
                "model_state_dict": cd_state,
                "latent_dim": 4,
                "model_config": self._model_config(),
                "data_grid_config": grid,
                "vae_checkpoint_sha256": cd_vae_hash,
                "ldm_checkpoint_sha256": sha256_file(ldm_path),
                "training_semantics": "ldm_initialized_ema_consistency_v1",
                "ldm_role": "initialization_checkpoint",
                "consistency_target_source": "cd_model_ema",
                "denoising_parameterization": "direct_x0_sigma_conditioned_v1",
                "consistency_training_config": {
                    "protocol": "ema_consistency_training_config_v1",
                    "training_semantics": "ldm_initialized_ema_consistency_v1",
                    "denoising_parameterization": "direct_x0_sigma_conditioned_v1",
                    "consistency_target_source": "cd_model_ema",
                    "loss": "mse",
                    "num_scales": 40,
                    "ema_rate": 0.999,
                    "sigma_min": 0.002,
                    "sigma_max": 80.0,
                    "rho": 7.0,
                },
        }
        if missing_data_protocol_stage != "cd":
            cd_payload["data_protocol"] = self._data_protocol()
            if mismatched_data_protocol_stage == "cd":
                cd_payload["data_protocol"]["split_artifact_sha256"] = "8" * 64
        if missing_normalization_stage != "cd":
            cd_normalization = self._radar_normalization()
            if mismatched_cd_normalization:
                cd_normalization["intensity"]["log_median"] = 3.0
            if nonformal_normalization_stage == "cd":
                cd_normalization["formal"] = False
            cd_payload["radar_normalization"] = cd_normalization
            if missing_normalization_hash_stage != "cd":
                cd_payload["radar_normalization_sha256"] = (
                    "c" * 64 if mismatched_cd_normalization_hash else "b" * 64
                )
        torch.save(cd_payload, cd_path)
        return vae_path, ldm_path, cd_path

    def test_valid_chain_is_accepted_and_reports_hashes(self):
        from diffusion_consistency_radar.checkpoint_chain import (
            validate_formal_checkpoint_chain,
        )

        with tempfile.TemporaryDirectory() as root:
            paths = self._write_chain(root)
            report = validate_formal_checkpoint_chain(*paths)

        self.assertTrue(report["chain_valid"])
        self.assertEqual(report["protocol"], "formal_chain_v2")
        self.assertEqual(set(report["stages"]), {"vae", "ldm", "cd"})
        self.assertEqual(len(report["sha256"]), 3)
        self.assertEqual(report["radar_normalization_protocol"], "radar_normalization_v1")
        self.assertEqual(report["radar_normalization_sha256"], "b" * 64)
        self.assertEqual(
            report["cd_consistency_training_config"]["protocol"],
            "ema_consistency_training_config_v1",
        )

    def test_formal_cd_requires_consistency_training_receipt(self):
        from diffusion_consistency_radar.checkpoint_chain import (
            CheckpointChainError,
            validate_formal_checkpoint_chain,
        )

        with tempfile.TemporaryDirectory() as root:
            paths = self._write_chain(root)
            cd_payload = torch.load(
                paths[2],
                map_location="cpu",
                weights_only=True,
            )
            del cd_payload["consistency_training_config"]
            torch.save(cd_payload, paths[2])
            with self.assertRaisesRegex(
                CheckpointChainError,
                "consistency_training_config",
            ):
                validate_formal_checkpoint_chain(*paths)

    def test_training_protocol_resolver_accepts_only_formal_or_isolated_mini(self):
        from diffusion_consistency_radar.checkpoint_chain import (
            FORMAL_CHECKPOINT_PROTOCOL,
            FORMAL_MINI_CHECKPOINT_PROTOCOL,
            resolve_training_checkpoint_protocol,
        )

        self.assertEqual(resolve_training_checkpoint_protocol(None), FORMAL_CHECKPOINT_PROTOCOL)
        self.assertEqual(resolve_training_checkpoint_protocol(""), FORMAL_CHECKPOINT_PROTOCOL)
        self.assertEqual(
            resolve_training_checkpoint_protocol(FORMAL_MINI_CHECKPOINT_PROTOCOL),
            FORMAL_MINI_CHECKPOINT_PROTOCOL,
        )
        with self.assertRaisesRegex(ValueError, "checkpoint_protocol"):
            resolve_training_checkpoint_protocol("legacy_or_typo")

    def test_formal_mini_selection_is_strict_and_part_of_resume_identity(self):
        from diffusion_consistency_radar.checkpoint_chain import (
            CheckpointChainError,
            assert_checkpoint_training_identity,
            build_formal_mini_selection,
        )

        selection = build_formal_mini_selection(8, 4)
        self.assertEqual(
            selection,
            {
                "protocol": "formal_mini_selection_v1",
                "strategy": "ordered_prefix_per_scene",
                "train_frames_per_scene": 8,
                "validation_frames_per_scene": 4,
            },
        )
        for invalid in (0, -1, True, "8"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "frames_per_scene"
            ):
                build_formal_mini_selection(invalid, 4)

        current_protocol = self._data_protocol()
        current_protocol["mini_selection"] = selection
        checkpoint_protocol = self._data_protocol()
        checkpoint_protocol["mini_selection"] = build_formal_mini_selection(7, 4)
        checkpoint = {
            "stage": "vae",
            "checkpoint_protocol": "formal_mini_chain_v2",
            "data_protocol": checkpoint_protocol,
        }
        with self.assertRaisesRegex(CheckpointChainError, "data_protocol"):
            assert_checkpoint_training_identity(
                checkpoint,
                expected_stage="vae",
                checkpoint_protocol="formal_mini_chain_v2",
                data_protocol=current_protocol,
            )

        del checkpoint["data_protocol"]["mini_selection"]
        with self.assertRaisesRegex(CheckpointChainError, "mini_selection"):
            assert_checkpoint_training_identity(
                checkpoint,
                expected_stage="vae",
                checkpoint_protocol="formal_mini_chain_v2",
                data_protocol=current_protocol,
            )

    def test_formal_validator_rejects_isolated_mini_chain(self):
        from diffusion_consistency_radar.checkpoint_chain import (
            CheckpointChainError,
            validate_formal_checkpoint_chain,
        )

        with tempfile.TemporaryDirectory() as root:
            paths = self._write_chain(
                root,
                checkpoint_protocol="formal_mini_chain_v2",
            )
            with self.assertRaisesRegex(CheckpointChainError, "formal_chain_v2"):
                validate_formal_checkpoint_chain(*paths)

    def test_target_ldm_validates_only_vae_and_ldm(self):
        from diffusion_consistency_radar.checkpoint_chain import (
            validate_formal_checkpoint_chain,
        )

        with tempfile.TemporaryDirectory() as root:
            vae_path, ldm_path, _cd_path = self._write_chain(root)
            report = validate_formal_checkpoint_chain(
                vae_path,
                ldm_path,
                target_stage="ldm",
            )

        self.assertEqual(report["target_stage"], "ldm")
        self.assertEqual(report["stages"], ["vae", "ldm"])

    def test_v2_data_protocol_is_required_and_must_match_parent(self):
        from diffusion_consistency_radar.checkpoint_chain import (
            CheckpointChainError,
            validate_formal_checkpoint_chain,
        )

        for kwargs in (
            {"missing_data_protocol_stage": "vae"},
            {"missing_data_protocol_stage": "ldm"},
            {"mismatched_data_protocol_stage": "ldm"},
        ):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as root:
                vae_path, ldm_path, _cd_path = self._write_chain(root, **kwargs)
                with self.assertRaisesRegex(CheckpointChainError, "data_protocol"):
                    validate_formal_checkpoint_chain(
                        vae_path,
                        ldm_path,
                        target_stage="ldm",
                    )

    def test_v1_chain_is_diagnostic_only_with_explicit_legacy_switch(self):
        from diffusion_consistency_radar.checkpoint_chain import (
            CheckpointChainError,
            validate_formal_checkpoint_chain,
        )

        with tempfile.TemporaryDirectory() as root:
            paths = self._write_chain(root, checkpoint_protocol="formal_chain_v1")
            with self.assertRaisesRegex(CheckpointChainError, "formal_chain_v2"):
                validate_formal_checkpoint_chain(*paths)
            report = validate_formal_checkpoint_chain(
                *paths,
                allow_legacy_protocol=True,
            )

        self.assertTrue(report["legacy_diagnostic"])
        self.assertEqual(report["protocol"], "formal_chain_v1")

    def test_missing_or_mismatched_radar_normalization_is_rejected(self):
        from diffusion_consistency_radar.checkpoint_chain import (
            CheckpointChainError,
            validate_formal_checkpoint_chain,
        )

        for kwargs in (
            {"missing_normalization_stage": "ldm"},
            {"missing_normalization_stage": "cd"},
            {"missing_normalization_hash_stage": "ldm"},
            {"missing_normalization_hash_stage": "cd"},
            {"mismatched_cd_normalization": True},
            {"mismatched_cd_normalization_hash": True},
            {"nonformal_normalization_stage": "ldm"},
            {"nonformal_normalization_stage": "cd"},
        ):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as root:
                paths = self._write_chain(root, **kwargs)
                with self.assertRaisesRegex(
                    CheckpointChainError,
                    "radar_normalization|normalization",
                ):
                    validate_formal_checkpoint_chain(*paths)

    def test_vae_must_not_embed_radar_normalization(self):
        from diffusion_consistency_radar.checkpoint_chain import (
            CheckpointChainError,
            validate_formal_checkpoint_chain,
        )

        with tempfile.TemporaryDirectory() as root:
            paths = self._write_chain(root, vae_has_normalization=True)
            with self.assertRaisesRegex(CheckpointChainError, "VAE|vae|normalization"):
                validate_formal_checkpoint_chain(*paths)

    def test_grid_mismatch_is_rejected_before_model_loading(self):
        from diffusion_consistency_radar.checkpoint_chain import (
            CheckpointChainError,
            validate_formal_checkpoint_chain,
        )

        with tempfile.TemporaryDirectory() as root:
            paths = self._write_chain(
                root,
                ldm_grid={
                    **self._grid(),
                    "model_pc_range": [0.0, -3.0, -1.0, 8.0, 3.0, 1.0],
                },
            )
            with self.assertRaisesRegex(CheckpointChainError, "网格|grid"):
                validate_formal_checkpoint_chain(*paths)

    def test_parent_hash_and_legacy_cd_are_rejected(self):
        from diffusion_consistency_radar.checkpoint_chain import (
            CheckpointChainError,
            validate_formal_checkpoint_chain,
        )

        for kwargs, message in (
            ({"bad_parent": True}, "sha256|hash"),
            ({"cd_legacy": True}, "多模态|multimodal|legacy"),
        ):
            with self.subTest(kwargs=kwargs):
                with tempfile.TemporaryDirectory() as root:
                    paths = self._write_chain(root, **kwargs)
                    with self.assertRaisesRegex(CheckpointChainError, message):
                        validate_formal_checkpoint_chain(*paths)

    def test_symlink_checkpoint_is_rejected(self):
        from diffusion_consistency_radar.checkpoint_chain import (
            CheckpointChainError,
            validate_formal_checkpoint_chain,
        )

        with tempfile.TemporaryDirectory() as root:
            paths = self._write_chain(root)
            link_path = os.path.join(root, "vae_link.pt")
            os.symlink(paths[0], link_path)
            with self.assertRaisesRegex(CheckpointChainError, "符号链接|symlink"):
                validate_formal_checkpoint_chain(link_path, paths[1], paths[2])

    def test_cli_writes_report_only_for_valid_chain(self):
        script = os.path.join(
            ROOT, "diffusion_consistency_radar", "scripts", "diagnose_checkpoint_chain.py"
        )
        with tempfile.TemporaryDirectory() as root:
            paths = self._write_chain(root)
            report_dir = os.path.join(root, "report")
            result = subprocess.run(
                [
                    sys.executable,
                    script,
                    "validate",
                    "--vae_ckpt",
                    paths[0],
                    "--ldm_ckpt",
                    paths[1],
                    "--cd_ckpt",
                    paths[2],
                    "--report_dir",
                    report_dir,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(os.path.join(report_dir, "checkpoint_chain.json"), encoding="utf-8") as handle:
                self.assertTrue(json.load(handle)["chain_valid"])

            invalid_report = os.path.join(root, "invalid_report")
            bad_paths = self._write_chain(os.path.join(root, "bad"), bad_parent=True)
            result = subprocess.run(
                [
                    sys.executable,
                    script,
                    "validate",
                    "--vae_ckpt",
                    bad_paths[0],
                    "--ldm_ckpt",
                    bad_paths[1],
                    "--cd_ckpt",
                    bad_paths[2],
                    "--report_dir",
                    invalid_report,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(os.path.exists(invalid_report))

    def test_construct_mode_loads_all_stages_strictly_on_cpu_without_forward(self):
        from diffusion_consistency_radar.scripts import diagnose_checkpoint_chain

        class FakeVAE(torch.nn.Module):
            latent_dim = 4

        class FakeModel(torch.nn.Module):
            is_multimodal = True

        with tempfile.TemporaryDirectory() as root:
            paths = self._write_chain(root)
            calls = []

            def fake_builder(state, device, strict, **kwargs):
                calls.append((device.type, strict, tuple(kwargs["fusion_voxel_shape"])))
                return FakeModel()

            with mock.patch(
                "diffusion_consistency_radar.cm.vae_3d.build_vae_from_checkpoint",
                return_value=(FakeVAE(), {}),
            ), mock.patch(
                "diffusion_consistency_radar.scripts.inference.build_inference_model",
                side_effect=fake_builder,
            ), mock.patch(
                "diffusion_consistency_radar.scripts.inference.resolve_generation_model_config",
                return_value=self._model_config(),
            ):
                report = diagnose_checkpoint_chain.diagnose_checkpoint_chain(
                    *paths,
                    construct=True,
                    device="cpu",
                )

        self.assertEqual([call[0] for call in calls], ["cpu", "cpu"])
        self.assertEqual([call[1] for call in calls], [True, True])
        self.assertEqual(report["construct"]["device"], "cpu")
        self.assertEqual(set(report["construct"]["stages_loaded"]), {"vae", "ldm", "cd"})


if __name__ == "__main__":
    unittest.main()
