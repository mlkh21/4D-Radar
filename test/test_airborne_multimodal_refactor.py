import os
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class AirborneVoxelizationTest(unittest.TestCase):
    def test_compensates_sync_offset_and_egomotion_doppler_variance(self):
        from NTU4DRadLM_pre_processing.NTU4DRadLM_pre_processing import (
            voxelize_pcl_airborne_optimized,
        )

        pcl = np.array(
            [
                [0.40, 0.0, 0.0, 2.0, 12.0],
                [0.60, 0.0, 0.0, 4.0, 14.0],
                [1.60, 0.0, 0.0, 1.0, 100.0],
                [1.70, 0.0, 0.0, 1.0, -100.0],
            ],
            dtype=np.float32,
        )

        voxel = voxelize_pcl_airborne_optimized(
            pcl,
            voxel_size=[1.0, 2.0, 2.0],
            pc_range=[0.0, -1.0, -1.0, 3.0, 1.0, 1.0],
            v_drone=np.array([10.0, 0.0, 0.0], dtype=np.float32),
            dt_sync=0.1,
        )

        self.assertEqual(voxel[1, 0, 0, 0], 1.0)
        self.assertAlmostEqual(float(voxel[1, 0, 0, 1]), 3.0, places=5)
        self.assertAlmostEqual(float(voxel[1, 0, 0, 2]), 3.0, places=5)
        self.assertAlmostEqual(float(voxel[1, 0, 0, 3]), 1.0, places=5)
        self.assertEqual(voxel[2, 0, 0, 3], 50.0)


class DatasetIRMetaTest(unittest.TestCase):
    def _write_sparse(self, path, voxel):
        occupied = voxel[..., 0] > 0
        np.savez_compressed(
            path,
            coords=np.column_stack(np.where(occupied)),
            features=voxel[occupied],
            shape=voxel.shape,
        )

    def test_dataset_returns_ir_tensor_and_compensated_calibration_meta(self):
        from diffusion_consistency_radar.cm.dataset_loader import NTU4DRadLM_VoxelDataset

        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "garden")
            radar_dir = os.path.join(scene_dir, "radar_voxel")
            target_dir = os.path.join(scene_dir, "target_voxel")
            os.makedirs(radar_dir)
            os.makedirs(target_dir)

            voxel = np.zeros((4, 4, 4, 4), dtype=np.float32)
            voxel[1, 1, 1, 0] = 1.0
            voxel[1, 1, 1, 1] = 0.5
            self._write_sparse(os.path.join(radar_dir, "000000.npz"), voxel)
            self._write_sparse(os.path.join(target_dir, "000000.npz"), voxel)

            ds = NTU4DRadLM_VoxelDataset(
                tmp,
                split="train",
                use_augmentation=False,
                return_path=True,
                sequence_length=1,
            )
            target, radar, meta, path = ds[0]

            self.assertEqual(tuple(target.shape), (4, 32, 128, 128))
            self.assertEqual(tuple(radar.shape), (4, 32, 128, 128))
            self.assertEqual(tuple(meta["ir_img"].shape), (3, 480, 640))
            self.assertGreater(float(meta["ir_img"].std()), 0.0)
            self.assertEqual(tuple(meta["r_mat"].shape), (3, 3))
            self.assertEqual(tuple(meta["t_vec"].shape), (3,))
            self.assertAlmostEqual(float(meta["t_vec"][0]), 0.01, places=6)
            self.assertEqual(tuple(meta["k_mat"].shape), (3, 3))
            self.assertTrue(path.endswith("000000.npz"))


class MultimodalFusionTest(unittest.TestCase):
    def test_heteroscedastic_nll_prefers_variance_matching_squared_error(self):
        from diffusion_consistency_radar.cm.multimodal_fusion import heteroscedastic_gaussian_nll

        prediction = torch.full((1, 4, 1, 1, 1), 0.5)
        target = torch.zeros_like(prediction)
        matched_variance = torch.full((1, 1, 1, 1, 1), 0.25)
        underestimated_variance = torch.full((1, 1, 1, 1, 1), 0.01)

        matched = heteroscedastic_gaussian_nll(prediction, target, matched_variance)
        underestimated = heteroscedastic_gaussian_nll(prediction, target, underestimated_variance)

        self.assertLess(float(matched), float(underestimated))

    def test_radar_encoder_and_uncertainty_head_expose_physical_condition_features(self):
        from diffusion_consistency_radar.cm.multimodal_fusion import RadarStructureEncoder, UncertaintyHead

        radar = torch.zeros(1, 4, 4, 4, 4)
        radar[:, 0] = 1.0
        radar[:, 3, :, :, :] = 0.0
        high_var = radar.clone()
        high_var[:, 3, :, :, :] = 50.0

        encoder = RadarStructureEncoder(in_channels=4, out_channels=16)
        encoded = encoder(radar)
        self.assertEqual(tuple(encoded.shape), (1, 16, 4, 4, 4))

        head = UncertaintyHead()
        low_unc = head(radar)
        high_unc = head(high_var)

        self.assertIn("confidence", low_unc)
        self.assertIn("log_var", low_unc)
        self.assertIn("variance", low_unc)
        self.assertEqual(tuple(low_unc["confidence"].shape), (1, 1, 4, 4, 4))
        self.assertGreater(
            float(low_unc["confidence"].mean()),
            float(high_unc["confidence"].mean()),
        )
        self.assertLess(
            float(low_unc["log_var"].mean()),
            float(high_unc["log_var"].mean()),
        )

    def test_projection_masks_invalid_frustum_and_fusion_outputs_sixteen_channels(self):
        from diffusion_consistency_radar.cm.multimodal_fusion import (
            CompleteDualModalityPerceptionNet,
            DualModalityProjectionLayer,
        )

        projection = DualModalityProjectionLayer(
            voxel_shape=(2, 2, 2),
            pc_range=[1.0, -1.0, -1.0, 3.0, 1.0, 1.0],
        )
        ir_features = torch.ones(1, 32, 8, 8)
        r_mat = torch.eye(3).unsqueeze(0)
        t_vec = torch.zeros(1, 3)
        k_mat = torch.tensor([[[4.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]]])

        projected = projection(ir_features, r_mat, t_vec, k_mat, (8, 8))

        self.assertEqual(tuple(projected.shape), (1, 32, 2, 2, 2))
        self.assertTrue(torch.isfinite(projected).all())

        class TinyBackbone(torch.nn.Module):
            def forward(self, x, timesteps):
                return x

        net = CompleteDualModalityPerceptionNet(
            TinyBackbone(),
            voxel_shape=(2, 2, 2),
            pc_range=[1.0, -1.0, -1.0, 3.0, 1.0, 1.0],
            downsample_to_latent=False,
        )
        radar = torch.ones(1, 4, 2, 2, 2)
        noised_latent = torch.full((1, 4, 2, 2, 2), 2.0)
        out = net(
            radar,
            torch.ones(1, 3, 480, 640),
            r_mat,
            t_vec,
            k_mat,
            torch.ones(1),
            noised_latent=noised_latent,
        )

        self.assertEqual(tuple(out.shape), (1, 16, 2, 2, 2))
        self.assertTrue(torch.all(out[:, :4] >= 2.0))

        out_with_unc, unc = net(
            radar,
            torch.ones(1, 3, 480, 640),
            r_mat,
            t_vec,
            k_mat,
            torch.ones(1),
            noised_latent=noised_latent,
            return_uncertainty=True,
        )
        self.assertEqual(tuple(out_with_unc.shape), (1, 16, 2, 2, 2))
        self.assertEqual(tuple(unc["confidence"].shape), (1, 1, 2, 2, 2))
        self.assertEqual(tuple(unc["variance"].shape), (1, 1, 2, 2, 2))
        self.assertGreater(float(unc["variance"].mean()), 0.0)

        unc_loss = unc["variance"].mean()
        unc_loss.backward()
        uncertainty_grads = [
            parameter.grad
            for parameter in net.model_uncertainty_head.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(grad is not None and torch.isfinite(grad).all() for grad in uncertainty_grads))

        latent_net = CompleteDualModalityPerceptionNet(
            TinyBackbone(),
            voxel_shape=(2, 2, 2),
            pc_range=[1.0, -1.0, -1.0, 3.0, 1.0, 1.0],
            downsample_to_latent=True,
            latent_shape=(2, 2, 2),
        )
        latent_out = latent_net(
            radar,
            torch.ones(1, 3, 480, 640),
            r_mat,
            t_vec,
            k_mat,
            torch.ones(1),
            noised_latent=torch.full((1, 4, 1, 1, 1), 2.0),
        )
        self.assertEqual(tuple(latent_out.shape), (1, 16, 1, 1, 1))


class UnifiedTrainBatchTest(unittest.TestCase):
    def test_unpack_training_batch_supports_meta_dict_and_legacy_batches(self):
        from diffusion_consistency_radar.scripts.unified_train import unpack_training_batch

        target = torch.zeros(1, 4, 2, 2, 2)
        radar = torch.ones(1, 4, 2, 2, 2)
        meta = {"ir_img": torch.zeros(1, 3, 480, 640)}

        self.assertEqual(unpack_training_batch((target, radar, meta))[2], meta)
        self.assertEqual(unpack_training_batch((target, radar))[2], {})

    def test_ldm_trainer_builds_sixteen_channel_unet_entry(self):
        from diffusion_consistency_radar.scripts.unified_train import ConfigManager, MemoryOptimizer, OptimizedLDMTrainer

        class DummyConfig(ConfigManager):
            def __init__(self):
                self.config = {
                    "hardware": {"device": "cpu"},
                    "optimization": {"use_amp": False, "use_checkpoint": False, "gradient_accumulation_steps": 1},
                    "ldm": {
                        "model_channels": 4,
                        "channel_mult": [1],
                        "num_res_blocks": 1,
                        "attention_resolutions": [],
                        "save_dir": "/tmp/ldm_test",
                    },
                }

        class DummyVAE(torch.nn.Module):
            pass

        cfg = DummyConfig()
        trainer = OptimizedLDMTrainer(DummyVAE(), cfg, MemoryOptimizer(cfg))

        self.assertEqual(trainer.model.in_channels, 16)
        self.assertEqual(trainer.model.__class__.__name__, "CompleteDualModalityPerceptionNet")


if __name__ == "__main__":
    unittest.main()
