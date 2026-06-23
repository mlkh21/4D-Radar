import os
import sys
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class MultimodalInferenceInterfaceTest(unittest.TestCase):
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
