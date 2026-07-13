import os
import sys
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class MultimodalInferenceInterfaceTest(unittest.TestCase):
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
        trainer = OptimizedLDMTrainer(vae, Config(), Memory())

        self.assertEqual(trainer.model.unet_3d.in_channels, 16)
        self.assertEqual(trainer.model.unet_3d.out_channels, 8)

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

    def test_missing_multimodal_meta_uses_mock_ir_and_generate_keeps_output_shape(self):
        from diffusion_consistency_radar.cm.karras_diffusion import KarrasDenoiser
        from diffusion_consistency_radar.scripts import inference

        class DummyVAE(torch.nn.Module):
            def get_latent(self, x):
                return torch.zeros(x.shape[0], 4, 2, 4, 4, device=x.device)

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
