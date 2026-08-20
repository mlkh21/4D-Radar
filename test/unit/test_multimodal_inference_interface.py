import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class MultimodalInferenceInterfaceTest(unittest.TestCase):
    def _radar_normalization(self, target_size=(2, 4, 5)):
        return {
            "protocol": "radar_normalization_v1",
            "formal": True,
            "training_scenes": ["unit_scene"],
            "frame_count": 2,
            "target_size": list(target_size),
            "source_pc_range": [0.0, 0.0, 0.0, 40.0, 20.0, 16.0],
            "model_pc_range": [0.0, 0.0, 0.0, 40.0, 20.0, 16.0],
            "intensity": {
                "transform": "log1p_robust_zscore",
                "log_median": 1.0,
                "log_iqr": 2.0,
                "clip": [-4.0, 4.0],
            },
            "doppler": {
                "transform": "symmetric_physical_scale",
                "scale_mps": 10.0,
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

    def _write_real_ir_fixture(self, root, ir_array=None, write_calibration=True):
        """构造逐文件推理所需的最小真实 IR/thermal 外参目录。"""
        radar_dir = os.path.join(root, "scene", "radar_voxel")
        ir_dir = os.path.join(root, "scene", "ir_image")
        config_dir = os.path.join(root, "config")
        os.makedirs(radar_dir)
        os.makedirs(ir_dir)
        os.makedirs(config_dir)

        radar_path = os.path.join(radar_dir, "000000.npy")
        ir_path = os.path.join(ir_dir, "000000_ir.npy")
        np.save(radar_path, np.zeros((2, 2, 2, 4), dtype=np.float32))
        np.save(
            ir_path,
            np.ones((8, 10), dtype=np.float32)
            if ir_array is None
            else ir_array,
        )
        if write_calibration:
            with open(
                os.path.join(config_dir, "calib_radar_to_thermal.txt"),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write("R: 1 0 0 0 1 0 0 0 1\nT: 1 2 3\n")
            with open(
                os.path.join(config_dir, "calib_cam_thermal.txt"),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    "S_00: 640 512\n"
                    "K_00: 457.2 0 323.1 0 457.9 242.5 0 0 1\n"
                    "D_00: 0 0 0 0 0\n"
                )
        return radar_path, ir_path

    def test_removed_oracle_arguments_report_diagnostic_migration(self):
        """旧 oracle 参数必须明确迁移到独立诊断脚本。"""
        from diffusion_consistency_radar.scripts import inference

        reject = getattr(inference, "reject_removed_oracle_arguments", None)
        self.assertIsNotNone(reject, "尚未实现旧 oracle 参数迁移检查")
        for argv in (
            ["--adaptive_occ_from_target"],
            ["--adaptive_target_threshold", "0.1"],
            ["--adaptive_target_threshold=0.1"],
        ):
            with self.subTest(argv=argv):
                with self.assertRaisesRegex(
                    ValueError,
                    "diagnose_oracle_target_adaptation.py",
                ):
                    reject(argv)

    def test_fixed_threshold_arguments_do_not_trigger_oracle_migration(self):
        """固定阈值和正常 target 对比不应触发迁移错误。"""
        from diffusion_consistency_radar.scripts import inference

        reject = getattr(inference, "reject_removed_oracle_arguments", None)
        self.assertIsNotNone(reject, "尚未实现旧 oracle 参数迁移检查")
        reject(["--occ_threshold", "0.5", "--compare_with_target"])

    def test_build_legacy_z8_model_uses_dynamic_latent_channels(self):
        from diffusion_consistency_radar.scripts import inference

        model = inference.build_inference_model(
            {},
            "cpu",
            strict=False,
            latent_dim=8,
            model_config={"model_channels": 8, "channel_mult": [1]},
        )
        output = model(
            torch.randn(1, 16, 2, 4, 4),
            torch.ones(1),
        )

        self.assertEqual(model.in_channels, 16)
        self.assertEqual(model.out_channels, 8)
        self.assertEqual(tuple(output.shape), (1, 8, 2, 4, 4))

    def test_unified_ldm_trainer_builds_z8_multimodal_backbone(self):
        from diffusion_consistency_radar.cm.vae_3d import (
            VAE3D,
            create_ultra_lightweight_vae_config,
        )
        from diffusion_consistency_radar.scripts.unified_train import (
            OptimizedLDMTrainer,
        )

        class Config:
            @staticmethod
            def get(key, default=None):
                if key == "ldm":
                    return {
                        "save_dir": "/tmp/task3-z8-ldm-test",
                        "model_channels": 8,
                        "channel_mult": [1],
                        "fusion_voxel_shape": [4, 8, 8],
                        "fusion_latent_shape": [2, 2, 2],
                    }
                return default

        class Memory:
            device = torch.device("cpu")

        vae_config = create_ultra_lightweight_vae_config()
        vae_config["latent_dim"] = 8
        vae = VAE3D(**vae_config)
        trainer = OptimizedLDMTrainer(
            vae,
            Config(),
            Memory(),
            allow_legacy_radar_units=True,
        )

        self.assertEqual(trainer.model.unet_3d.in_channels, 16)
        self.assertEqual(trainer.model.unet_3d.out_channels, 8)

    def test_multimodal_ldm_encodes_only_target_but_legacy_encodes_condition(self):
        from diffusion_consistency_radar.scripts.unified_train import (
            encode_ldm_training_latents,
        )

        class CountingVAE:
            def __init__(self):
                self.inputs = []

            def get_latent(self, value):
                self.inputs.append(value)
                return value[:, :1]

        target = torch.randn(1, 4, 2, 2, 2)
        condition = torch.randn_like(target)
        meta = {
            "ir_img": torch.zeros(1, 3, 2, 2),
            "r_mat": torch.eye(3).unsqueeze(0),
            "t_vec": torch.zeros(1, 3),
            "k_mat": torch.eye(3).unsqueeze(0),
        }
        vae = CountingVAE()
        _z_target, z_cond = encode_ldm_training_latents(
            vae, target, condition, meta
        )
        self.assertIsNone(z_cond)
        self.assertEqual(vae.inputs, [target])

        vae = CountingVAE()
        _z_target, z_cond = encode_ldm_training_latents(
            vae, target, condition, {}
        )
        self.assertIsNotNone(z_cond)
        self.assertEqual(vae.inputs, [target, condition])

    def test_generation_checkpoint_shape_infers_legacy_latent_dim(self):
        from diffusion_consistency_radar.scripts.inference import (
            resolve_generation_model_config,
        )

        state = {
            "input_blocks.0.0.weight": torch.zeros(8, 16, 3, 3, 3),
            "out.2.weight": torch.zeros(8, 8, 3, 3, 3),
        }

        resolved = resolve_generation_model_config(
            {"model_state_dict": state},
            fallback_latent_dim=None,
        )

        self.assertEqual(resolved["latent_dim"], 8)
        self.assertEqual(resolved["in_channels"], 16)

    def test_unconditional_generate_derives_z8_nondefault_latent_shape(self):
        from diffusion_consistency_radar.cm.karras_diffusion import KarrasDenoiser
        from diffusion_consistency_radar.scripts import inference

        class DummyVAE(torch.nn.Module):
            latent_dim = 8

            def latent_spatial_shape(self, input_size):
                self.last_input_size = tuple(input_size)
                return (4, 12, 10)

            def decode(self, z):
                self.last_decode_shape = tuple(z.shape)
                return torch.zeros(z.shape[0], 4, 8, 48, 40)

        class DummyCDModel(torch.nn.Module):
            is_multimodal = False

            def forward(self, model_input, sigma):
                self.last_input_shape = tuple(model_input.shape)
                return model_input[:, :8]

        generator = inference.RadarGenerator.__new__(inference.RadarGenerator)
        generator.device = torch.device("cpu")
        generator.model_type = "cd"
        generator.target_size = (8, 48, 40)
        generator.vae = DummyVAE()
        generator.model = DummyCDModel()
        generator.vae_checkpoint_metadata = {"occupancy_activation": "raw"}
        generator.denoiser = KarrasDenoiser(
            sigma_data=0.5, sigma_max=1.0, sigma_min=0.1
        )

        generated = generator.generate(None, num_samples=2, steps=1)

        self.assertEqual(generator.vae.last_input_size, (8, 48, 40))
        self.assertEqual(generator.vae.last_decode_shape, (2, 8, 4, 12, 10))
        self.assertEqual(generator.model.last_input_shape, (2, 16, 4, 12, 10))
        self.assertEqual(tuple(generated.shape), (2, 4, 8, 48, 40))
    def test_generated_occupancy_activation_only_converts_channel_zero(self):
        from diffusion_consistency_radar.scripts import inference

        generator = inference.RadarGenerator.__new__(inference.RadarGenerator)
        generator.vae_checkpoint_metadata = {"occupancy_activation": "sigmoid"}
        decoded = torch.tensor([[[[[-2.0]]], [[[3.0]]], [[[4.0]]], [[[5.0]]]]])

        activated = generator._apply_vae_occupancy_activation(decoded)

        torch.testing.assert_close(activated[:, 0], torch.sigmoid(decoded[:, 0]))
        torch.testing.assert_close(activated[:, 1:], decoded[:, 1:])

    def test_vae_loader_prefers_checkpoint_metadata_over_legacy_fallback(self):
        from diffusion_consistency_radar.cm.vae_3d import (
            VAE3D,
            create_lightweight_vae_config,
        )
        from diffusion_consistency_radar.scripts import inference

        config = create_lightweight_vae_config()
        config["latent_dim"] = 8
        config["base_channels"] = 32
        checkpoint = {
            "model_state_dict": VAE3D(**config).state_dict(),
            "vae_config": config,
            "vae_config_type": "lightweight",
            "occupancy_activation": "sigmoid",
        }
        generator = inference.RadarGenerator.__new__(inference.RadarGenerator)
        generator.device = torch.device("cpu")
        generator.vae_fallback_config_type = "ultra_lightweight"

        with unittest.mock.patch.object(inference, "safe_torch_load", return_value=checkpoint):
            model = generator._load_vae("unused.pt")

        self.assertEqual(model.latent_dim, 8)

    def test_vae_loader_rejects_legacy_checkpoint_without_explicit_fallback(self):
        from diffusion_consistency_radar.cm.vae_3d import (
            VAE3D,
            create_ultra_lightweight_vae_config,
        )
        from diffusion_consistency_radar.scripts import inference

        config = create_ultra_lightweight_vae_config()
        checkpoint = {"model_state_dict": VAE3D(**config).state_dict()}
        generator = inference.RadarGenerator.__new__(inference.RadarGenerator)
        generator.device = torch.device("cpu")
        generator.vae_fallback_config_type = None

        with unittest.mock.patch.object(inference, "safe_torch_load", return_value=checkpoint):
            with self.assertRaisesRegex(ValueError, "fallback"):
                generator._load_vae("unused.pt")

    def test_inference_grid_uses_checkpoint_metadata_when_not_explicit(self):
        from diffusion_consistency_radar.scripts.inference import (
            resolve_inference_grid_config,
        )

        metadata = {
            "data_grid_config": {
                "target_size": [16, 64, 80],
                "source_pc_range": [0, -30, -5, 100, 30, 15],
                "model_pc_range": [0, -10, -3, 40, 10, 9],
            }
        }

        target_size, source_range, model_range = resolve_inference_grid_config(
            metadata, None, None, None
        )

        self.assertEqual(target_size, (16, 64, 80))
        self.assertEqual(source_range[1], -30.0)
        self.assertEqual(model_range[4], 10.0)

    def test_inference_normalization_fails_closed_and_legacy_is_explicit(self):
        from diffusion_consistency_radar.radar_normalization import (
            RadarNormalizationError,
        )
        from diffusion_consistency_radar.scripts.inference import (
            resolve_inference_radar_normalization,
        )

        geometry = {
            "target_size": (2, 4, 5),
            "source_pc_range": (0, 0, 0, 40, 20, 16),
            "model_pc_range": (0, 0, 0, 40, 20, 16),
        }
        with self.assertRaisesRegex(RadarNormalizationError, "radar_normalization"):
            resolve_inference_radar_normalization({}, **geometry)

        spec, digest = resolve_inference_radar_normalization(
            {},
            allow_legacy_radar_units=True,
            **geometry,
        )
        self.assertIsNone(spec)
        self.assertEqual(digest, "")

        checkpoint = {
            "radar_normalization": self._radar_normalization(),
            "radar_normalization_sha256": "b" * 64,
        }
        resolved, digest = resolve_inference_radar_normalization(
            checkpoint,
            **geometry,
        )
        self.assertEqual(resolved, self._radar_normalization())
        self.assertEqual(digest, "b" * 64)
        with self.assertRaisesRegex(RadarNormalizationError, "legacy|正式"):
            resolve_inference_radar_normalization(
                checkpoint,
                allow_legacy_radar_units=True,
                **geometry,
            )

    def test_model_loader_retains_checkpoint_normalization_before_model_build(self):
        from diffusion_consistency_radar.scripts import inference

        checkpoint = {
            "model_state_dict": {"weight": torch.zeros(1)},
            "radar_normalization": self._radar_normalization(),
            "radar_normalization_sha256": "b" * 64,
        }
        generator = inference.RadarGenerator.__new__(inference.RadarGenerator)
        generator.device = torch.device("cpu")
        generator.target_size = (2, 4, 5)
        generator.source_pc_range = (0, 0, 0, 40, 20, 16)
        generator.pc_range = (0, 0, 0, 40, 20, 16)
        generator.allow_legacy_radar_units = False
        generator.vae = SimpleNamespace(latent_dim=4)
        built_model = SimpleNamespace()

        with unittest.mock.patch.object(
            inference,
            "safe_torch_load",
            return_value=checkpoint,
        ), unittest.mock.patch.object(
            inference,
            "resolve_generation_model_config",
            return_value={"latent_dim": 4},
        ), unittest.mock.patch.object(
            inference,
            "build_inference_model",
            return_value=built_model,
        ) as build_model:
            model = generator._load_model("unused.pt")

        self.assertIs(model, built_model)
        self.assertIs(generator.model_checkpoint_metadata, checkpoint)
        self.assertEqual(generator.radar_normalization, self._radar_normalization())
        self.assertEqual(generator.radar_normalization_sha256, "b" * 64)
        self.assertEqual(build_model.call_count, 1)

    def test_inference_radar_loader_reuses_physical_resize_and_normalization(self):
        from diffusion_consistency_radar.scripts import inference

        spec = self._radar_normalization(target_size=(1, 1, 1))
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "radar.npy")
            raw = np.array([[[[1.0, np.expm1(3.0), 5.0, 2.0]]]], dtype=np.float32)
            np.save(path, raw)
            tensor = inference.load_radar_voxel_as_tensor(
                path,
                torch.device("cpu"),
                target_size=(1, 1, 1),
                source_pc_range=(0, 0, 0, 40, 20, 16),
                model_pc_range=(0, 0, 0, 40, 20, 16),
                radar_normalization=spec,
                radar_normalization_sha256="b" * 64,
            )

        torch.testing.assert_close(
            tensor[:, 0, 0, 0],
            torch.tensor([1.0, 1.0, 0.5, 2.0]),
        )

    def test_checkpoint_key_detection_selects_multimodal_or_legacy_model(self):
        from diffusion_consistency_radar.scripts import inference
        from diffusion_consistency_radar.cm.multimodal_fusion import CompleteDualModalityPerceptionNet
        from diffusion_consistency_radar.cm.unet_optimized import OptimizedUNetModel

        multimodal_state = {
            "unet_3d.input_blocks.0.0.weight": torch.zeros(1),
            "ir_extractor.backbone.0.weight": torch.zeros(1),
        }
        legacy_state = {
            "input_blocks.0.0.weight": torch.zeros(1),
            "out.2.weight": torch.zeros(1),
        }

        self.assertTrue(inference.is_multimodal_state_dict(multimodal_state))
        self.assertFalse(inference.is_multimodal_state_dict(legacy_state))
        self.assertIsInstance(
            inference.build_inference_model(multimodal_state, "cpu", strict=False),
            CompleteDualModalityPerceptionNet,
        )
        self.assertIsInstance(
            inference.build_inference_model(legacy_state, "cpu", strict=False),
            OptimizedUNetModel,
        )

    def test_strict_multimodal_load_rejects_sparse_compatible_subset(self):
        from diffusion_consistency_radar.scripts import inference

        sparse_state = {
            "unet_3d.input_blocks.0.0.weight": torch.zeros(
                8, 16, 3, 3, 3
            ),
        }

        with self.assertRaisesRegex(RuntimeError, "多模态.*checkpoint"):
            inference.build_inference_model(
                sparse_state,
                "cpu",
                strict=True,
                fusion_voxel_shape=(4, 8, 8),
                model_config={"model_channels": 8, "channel_mult": [1]},
            )

    def test_strict_multimodal_load_migrates_legacy_ir_gate_weight(self):
        from diffusion_consistency_radar.scripts import inference

        model = inference.build_inference_model(
            {
                "unet_3d.input_blocks.0.0.weight": torch.zeros(1),
                "ir_extractor.backbone.0.weight": torch.zeros(1),
            },
            "cpu",
            strict=False,
            fusion_voxel_shape=(4, 8, 8),
            model_config={"model_channels": 8, "channel_mult": [1], "in_channels": 16, "latent_dim": 4},
        )
        state = model.state_dict()
        old_gate = state["ir_gate.0.weight"][:, :17].clone()
        state["ir_gate.0.weight"] = old_gate

        loaded = inference.build_inference_model(
            state,
            "cpu",
            strict=True,
            fusion_voxel_shape=(4, 8, 8),
            model_config={"model_channels": 8, "channel_mult": [1], "in_channels": 16, "latent_dim": 4},
        )

        self.assertEqual(tuple(loaded.state_dict()["ir_gate.0.weight"].shape[1:2]), (49,))

    def test_strict_load_rejects_empty_state_dict(self):
        from diffusion_consistency_radar.scripts import inference

        with self.assertRaisesRegex(ValueError, "strict=True.*空"):
            inference.build_inference_model(
                {},
                "cpu",
                strict=True,
                model_config={"model_channels": 8, "channel_mult": [1]},
            )

    def test_strict_load_reports_missing_critical_output_weight(self):
        from diffusion_consistency_radar.scripts import inference

        model = inference.build_inference_model(
            {},
            "cpu",
            strict=False,
            model_config={"model_channels": 8, "channel_mult": [1]},
        )
        incomplete_state = model.state_dict()
        del incomplete_state["out.2.weight"]

        with self.assertRaisesRegex(RuntimeError, "关键权重.*out\\.2\\.weight"):
            inference.build_inference_model(
                incomplete_state,
                "cpu",
                strict=True,
                model_config={"model_channels": 8, "channel_mult": [1]},
            )

    def test_strict_real_ir_loads_real_meta_and_applies_training_sync_offset(self):
        """严格模式必须使用真实 IR/thermal 外参并匹配训练同步位移。"""
        from diffusion_consistency_radar.scripts import inference

        with tempfile.TemporaryDirectory() as root:
            radar_path, _ = self._write_real_ir_fixture(root)
            meta = inference.load_multimodal_meta_for_radar(
                radar_path,
                torch.device("cpu"),
                require_real_ir=True,
            )

        self.assertEqual(float(meta["is_mock_ir"].item()), 0.0)
        self.assertEqual(float(meta["is_mock_calib"].item()), 0.0)
        self.assertAlmostEqual(float(meta["t_vec"][0].item()), 1.01, places=6)
        self.assertAlmostEqual(
            float(meta["legacy_sync_displacement_x_m"]), 0.01, places=6
        )

    def test_strict_real_ir_rejects_missing_frame(self):
        """严格模式不得在 IR 缺失时静默使用 mock thermal。"""
        from diffusion_consistency_radar.scripts import inference

        with tempfile.TemporaryDirectory() as root:
            radar_path, ir_path = self._write_real_ir_fixture(root)
            os.remove(ir_path)
            with self.assertRaisesRegex(RuntimeError, "缺少.*IR"):
                inference.load_multimodal_meta_for_radar(
                    radar_path,
                    torch.device("cpu"),
                    require_real_ir=True,
                )

    def test_strict_real_ir_rejects_symlink(self):
        """正式 IR 文件必须是当前数据根内的普通文件。"""
        from diffusion_consistency_radar.scripts import inference

        with tempfile.TemporaryDirectory() as root:
            radar_path, ir_path = self._write_real_ir_fixture(root)
            linked_source = os.path.join(root, "linked_ir.npy")
            np.save(linked_source, np.ones((8, 10), dtype=np.float32))
            os.remove(ir_path)
            os.symlink(linked_source, ir_path)
            with self.assertRaisesRegex(RuntimeError, "符号链接"):
                inference.load_multimodal_meta_for_radar(
                    radar_path,
                    torch.device("cpu"),
                    require_real_ir=True,
                )

    def test_strict_real_ir_rejects_invalid_shape_and_nonfinite_values(self):
        """严格 preflight 必须在采样前拦截不可投影的 IR 数组。"""
        from diffusion_consistency_radar.scripts import inference

        invalid_arrays = (
            np.ones((8,), dtype=np.float32),
            np.full((8, 10), np.nan, dtype=np.float32),
        )
        for array in invalid_arrays:
            with self.subTest(shape=array.shape, finite=np.isfinite(array).all()):
                with tempfile.TemporaryDirectory() as root:
                    radar_path, _ = self._write_real_ir_fixture(root, ir_array=array)
                    with self.assertRaisesRegex(RuntimeError, "维度非法|非有限"):
                        inference.load_multimodal_meta_for_radar(
                            radar_path,
                            torch.device("cpu"),
                            require_real_ir=True,
                        )

    def test_strict_real_ir_rejects_missing_thermal_calibration(self):
        """Livox/mock fallback 不能冒充正式 thermal 外参。"""
        from diffusion_consistency_radar.scripts import inference

        with tempfile.TemporaryDirectory() as root:
            radar_path, _ = self._write_real_ir_fixture(
                root,
                write_calibration=False,
            )
            with self.assertRaisesRegex(RuntimeError, "thermal|calib_radar_to_thermal"):
                inference.load_multimodal_meta_for_radar(
                    radar_path,
                    torch.device("cpu"),
                    require_real_ir=True,
                )

    def test_strict_real_ir_rejects_missing_thermal_intrinsics(self):
        """正式真实 IR 还必须有 calib_cam_thermal.txt 的 S/K/D。"""
        from diffusion_consistency_radar.scripts import inference

        with tempfile.TemporaryDirectory() as root:
            radar_path, _ = self._write_real_ir_fixture(root)
            os.remove(os.path.join(root, "config", "calib_cam_thermal.txt"))
            with self.assertRaisesRegex(RuntimeError, "calib_cam_thermal"):
                inference.load_multimodal_meta_for_radar(
                    radar_path,
                    torch.device("cpu"),
                    require_real_ir=True,
                )

    def test_strict_real_ir_rejects_single_modality_model(self):
        """单模态 checkpoint 不能标记为正式 Radar+IR 结果。"""
        from diffusion_consistency_radar.scripts import inference

        class SingleModalityModel:
            is_multimodal = False

        with self.assertRaisesRegex(RuntimeError, "multimodal|Radar\+IR"):
            inference.validate_real_ir_model(SingleModalityModel())

    def test_compatibility_meta_keeps_missing_ir_mock_fallback(self):
        """非正式直接入口继续保留既有 mock IR 诊断能力。"""
        from diffusion_consistency_radar.scripts import inference

        with tempfile.TemporaryDirectory() as root:
            radar_path, ir_path = self._write_real_ir_fixture(root)
            os.remove(ir_path)
            meta = inference.load_multimodal_meta_for_radar(
                radar_path,
                torch.device("cpu"),
            )

        self.assertEqual(float(meta["is_mock_ir"].item()), 1.0)

    def test_runtime_fields_and_run_metadata_use_resolved_grid(self):
        """部署运行记录必须携带生成实际使用的网格和固定阈值。"""
        from diffusion_consistency_radar.scripts import inference

        args = SimpleNamespace(
            target_size=(2, 4, 5),
            source_pc_range=(0, 0, 0, 40, 20, 16),
            pc_range=(0, 0, 0, 40, 20, 16),
            voxel_size=None,
            occ_threshold=0.35,
            model_type="ldm",
            steps=40,
            sampler="heun",
            require_real_ir=True,
        )
        generator = SimpleNamespace(model=SimpleNamespace(is_multimodal=True))
        generator.radar_normalization = self._radar_normalization()
        generator.radar_normalization_sha256 = "b" * 64
        generator.allow_legacy_radar_units = False

        metadata = inference.build_inference_run_metadata(
            args,
            generator,
            frame_count=2,
        )

        self.assertEqual(
            inference.RUNTIME_FIELDS,
            [
                "index",
                "radar_file",
                "radar_point_count",
                "effective_occ_threshold",
                "inference_seconds",
                "pred_point_count",
                "is_empty_frame",
                "used_topk_fallback",
                "train_duration_seconds",
                "total_infer_seconds",
                "avg_infer_seconds",
                "avg_pred_point_count",
                "empty_frame_rate",
            ],
        )
        self.assertEqual(metadata["target_size"], [2, 4, 5])
        self.assertEqual(metadata["voxel_size"], [10.0, 4.0, 8.0])
        self.assertEqual(metadata["frame_count"], 2)
        self.assertTrue(metadata["model_is_multimodal"])
        self.assertTrue(metadata["require_real_ir"])
        self.assertEqual(metadata["radar_normalization"], self._radar_normalization())
        self.assertEqual(metadata["radar_normalization_sha256"], "b" * 64)
        self.assertEqual(metadata["radar_normalization_protocol"], "radar_normalization_v1")
        self.assertTrue(metadata["formal_protocol"])

    def test_legacy_inference_metadata_cannot_look_formal(self):
        from diffusion_consistency_radar.scripts import inference

        args = SimpleNamespace(
            target_size=(2, 4, 5),
            source_pc_range=(0, 0, 0, 40, 20, 16),
            pc_range=(0, 0, 0, 40, 20, 16),
            voxel_size=None,
            occ_threshold=0.35,
            model_type="ldm",
            steps=40,
            sampler="heun",
            require_real_ir=False,
        )
        generator = SimpleNamespace(
            model=SimpleNamespace(is_multimodal=False),
            radar_normalization=None,
            radar_normalization_sha256="",
            allow_legacy_radar_units=True,
        )

        metadata = inference.build_inference_run_metadata(args, generator, 1)

        self.assertEqual(metadata["radar_normalization_protocol"], "legacy_identity")
        self.assertFalse(metadata["formal_protocol"])
        self.assertTrue(metadata["allow_legacy_radar_units"])

    def test_csv_name_separates_runtime_from_legacy_evaluation(self):
        """无真值参数时不得把纯部署运行记录命名成评价指标。"""
        from diffusion_consistency_radar.scripts import inference

        runtime_args = SimpleNamespace(
            compare_with_target=False,
            compare_with_lidar=False,
            report_task_metrics=False,
        )
        evaluation_args = SimpleNamespace(
            compare_with_target=True,
            compare_with_lidar=False,
            report_task_metrics=False,
        )

        self.assertEqual(
            inference.inference_csv_name(runtime_args),
            "inference_runtime.csv",
        )
        self.assertEqual(
            inference.inference_csv_name(evaluation_args),
            "inference_metrics.csv",
        )

    def test_missing_multimodal_meta_uses_mock_ir_and_generate_keeps_output_shape(self):
        from diffusion_consistency_radar.cm.karras_diffusion import KarrasDenoiser
        from diffusion_consistency_radar.scripts import inference

        class DummyVAE(torch.nn.Module):
            latent_dim = 4

            def get_latent(self, x):
                raise AssertionError("多模态推理不得把 Radar condition 编码为 VAE latent")

            def latent_spatial_shape(self, input_size):
                return (2, 4, 4)

            def decode(self, z):
                return torch.zeros(z.shape[0], 4, 32, 128, 128, device=z.device)

        class DummyMultimodalModel(torch.nn.Module):
            is_multimodal = True

            def forward(self, radar_voxel, ir_img, r_mat, t_vec, k_mat, timesteps, noised_latent=None):
                self.last_ir_shape = tuple(ir_img.shape)
                self.last_t_shape = tuple(t_vec.shape)
                return torch.zeros_like(noised_latent)

        generator = inference.RadarGenerator.__new__(inference.RadarGenerator)
        generator.device = torch.device("cpu")
        generator.model_type = "cd"
        generator.target_size = (32, 128, 128)
        generator.vae = DummyVAE()
        generator.model = DummyMultimodalModel()
        generator.denoiser = KarrasDenoiser(sigma_data=0.5, sigma_max=1.0, sigma_min=0.1)

        condition = torch.zeros(1, 4, 32, 128, 128)
        generated = generator.generate(condition, steps=1, meta_dict=None)

        self.assertEqual(tuple(generated.shape), (1, 4, 32, 128, 128))
        self.assertEqual(generator.model.last_ir_shape, (1, 3, 480, 640))
        self.assertEqual(generator.model.last_t_shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
