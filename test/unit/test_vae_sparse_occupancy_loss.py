#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 VAE 对极稀疏 occupancy 与有效连续体素的重建损失。
"""

import os
import sys
import csv
import tempfile
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffusion_consistency_radar.cm.vae_3d import (
    VAE3D,
    create_lightweight_vae_config,
    create_standard_vae_config,
    create_ultra_lightweight_vae_config,
)
from diffusion_consistency_radar.scripts.diagnose_vae_reconstruction import (
    extract_reconstruction_occupancy,
)
from diffusion_consistency_radar.scripts.unified_train import (
    OptimizedVAETrainer,
    apply_vae_config_overrides,
)


class _TrainerMemory:
    """为 trainer 单测提供纯 CPU 的最小显存配置。"""

    device = torch.device("cpu")
    use_amp = False
    scaler = None
    grad_accum_steps = 2

    @staticmethod
    def clear_cache():
        return None


class _TrainerModel(torch.nn.Module):
    """用 target 首元素控制有限或无限损失的最小模型。"""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.loss_components = {
            "occ_bce_loss": torch.tensor(1.0),
            "occ_dice_loss": torch.tensor(2.0),
            "continuous_loss": torch.tensor(3.0),
        }

    def forward(self, target):
        posterior = torch.zeros(target.shape[0], 1, 1, 1, 1)
        return target * self.weight, (posterior, posterior)

    def compute_loss(self, target, reconstruction, posterior):
        marker = target.flatten()[0]
        loss = self.weight * marker
        if marker.item() == 2.0:
            loss = loss * torch.tensor(float("inf"))
        return loss, loss, self.weight * 0.0


class _CountingSGD(torch.optim.SGD):
    """记录参数更新次数，验证有效 batch 的累计边界。"""

    def __init__(self, params):
        super().__init__(params, lr=0.1)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


class _Scheduler:
    def __init__(self):
        self.step_count = 0

    def step(self):
        self.step_count += 1


class VAESparseOccupancyLossTest(unittest.TestCase):
    @staticmethod
    def _make_model(**overrides):
        config = {
            "base_channels": 8,
            "encoder_channel_mult": (1,),
            "decoder_channel_mult": (1,),
            "num_res_blocks": 1,
            "use_attention": False,
            "use_checkpoint": False,
            "kl_weight": 0.0,
        }
        config.update(overrides)
        return VAE3D(**config)

    @staticmethod
    def _zero_posterior(batch_size=1):
        mean = torch.zeros(batch_size, 4, 1, 1, 1)
        return mean, mean.clone()

    def test_occupancy_probability_matches_sigmoid(self):
        logits = torch.tensor([-4.0, 0.0, 2.0], dtype=torch.float32)

        probabilities = VAE3D.occupancy_probability(logits)

        torch.testing.assert_close(probabilities, torch.sigmoid(logits))

    def test_reconstruction_extraction_only_uses_occupancy_channel(self):
        raw = torch.tensor(
            [[[[[-2.0]]], [[[1.25]]], [[[-0.75]]], [[[3.5]]]]],
            dtype=torch.float32,
        )
        changed_continuous_channels = raw.clone()
        changed_continuous_channels[:, 1:] = 1000.0

        occupancy = extract_reconstruction_occupancy(raw, "sigmoid")
        occupancy_after_change = extract_reconstruction_occupancy(
            changed_continuous_channels,
            "sigmoid",
        )

        torch.testing.assert_close(occupancy, torch.sigmoid(raw[:, 0]))
        torch.testing.assert_close(occupancy_after_change, occupancy)

    def test_sparse_correct_logits_have_lower_loss_than_all_empty_logits(self):
        model = self._make_model()
        target = torch.zeros(1, 4, 1, 1, 64)
        target[:, 0, 0, 0, 7] = 1.0
        correct_logits = torch.full_like(target, -8.0)
        correct_logits[:, 0, 0, 0, 7] = 8.0
        all_empty_logits = torch.full_like(target, -8.0)

        correct_loss = model.compute_loss(
            target, correct_logits, self._zero_posterior()
        )[1]
        all_empty_loss = model.compute_loss(
            target, all_empty_logits, self._zero_posterior()
        )[1]

        self.assertLess(correct_loss.item(), all_empty_loss.item())

    def test_single_positive_voxel_has_nonzero_gradient_toward_positive_logit(self):
        model = self._make_model()
        target = torch.zeros(1, 4, 1, 1, 32)
        target[:, 0, 0, 0, 3] = 1.0
        logits = torch.zeros_like(target, requires_grad=True)

        loss = model.compute_loss(target, logits, self._zero_posterior())[0]
        loss.backward()

        positive_gradient = logits.grad[0, 0, 0, 0, 3]
        self.assertLess(positive_gradient.item(), 0.0)
        self.assertGreater(abs(positive_gradient.item()), 0.0)

    def test_continuous_background_error_does_not_dominate_valid_voxels(self):
        model = self._make_model()
        target = torch.zeros(1, 4, 1, 1, 16)
        target[:, 0, 0, 0, 2] = 1.0
        target[:, 1, 0, 0, 2] = 2.0
        target[:, 3, 0, 0, 5] = 1.0
        target[:, 2, 0, 0, 5] = -3.0

        valid_error = target.clone()
        valid_error[:, 1, 0, 0, 2] += 1.0
        valid_error[:, 2, 0, 0, 5] += 1.0
        background_error = target.clone()
        background_error[:, 1:4, 0, 0, 8:] = 1000.0

        model.compute_loss(target, valid_error, self._zero_posterior())
        valid_loss = model.loss_components["continuous_loss"]
        model.compute_loss(target, background_error, self._zero_posterior())
        background_loss = model.loss_components["continuous_loss"]

        self.assertGreater(valid_loss.item(), 0.0)
        torch.testing.assert_close(background_loss, torch.zeros_like(background_loss))

    def test_continuous_loss_is_finite_when_no_voxel_is_valid(self):
        model = self._make_model()
        target = torch.zeros(1, 4, 1, 1, 8)
        reconstruction = target.clone()
        reconstruction[:, 1:] = 1000.0

        total_loss = model.compute_loss(
            target, reconstruction, self._zero_posterior()
        )[0]

        self.assertTrue(torch.isfinite(total_loss))
        torch.testing.assert_close(
            model.loss_components["continuous_loss"],
            torch.zeros_like(model.loss_components["continuous_loss"]),
        )

    def test_empty_batch_dice_uses_global_plus_one_smoothing(self):
        model = self._make_model(
            occupancy_bce_weight=0.0,
            continuous_recon_weight=0.0,
        )
        target = torch.zeros(2, 4, 1, 1, 2)
        logits = torch.zeros_like(target)

        model.compute_loss(target, logits, self._zero_posterior(batch_size=2))

        expected = 1.0 - 1.0 / (4.0 * 0.5 + 1.0)
        self.assertAlmostEqual(
            model.loss_components["occ_dice_loss"].item(), expected, places=6
        )

    def test_mixed_batch_dice_aggregates_batch_and_space_together(self):
        model = self._make_model(
            occupancy_bce_weight=0.0,
            continuous_recon_weight=0.0,
        )
        target = torch.zeros(2, 4, 1, 1, 2)
        target[0, 0, 0, 0, 0] = 1.0
        logits = torch.zeros_like(target)

        model.compute_loss(target, logits, self._zero_posterior(batch_size=2))

        expected = 1.0 - (2.0 * 0.5 + 1.0) / (2.0 + 1.0 + 1.0)
        self.assertAlmostEqual(
            model.loss_components["occ_dice_loss"].item(), expected, places=6
        )

    def test_half_large_grid_uses_finite_fp32_occupancy_components(self):
        model = self._make_model(continuous_recon_weight=0.0)
        target = torch.zeros(1, 1, 1, 1, 70000, dtype=torch.float16)
        target[:, 0, 0, 0, 0] = 1.0
        logits = torch.zeros_like(target, requires_grad=True)

        loss = model.compute_loss(target, logits, self._zero_posterior())[0]
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(model.loss_components["occ_bce_loss"].dtype, torch.float32)
        self.assertEqual(model.loss_components["occ_dice_loss"].dtype, torch.float32)

    def test_legacy_mse_preserves_default_channel_weights(self):
        model = self._make_model(
            occupancy_loss_type="legacy_mse",
            occupied_weight=3.0,
            empty_weight=1.0,
            false_positive_weight=0.0,
            occupancy_mass_weight=0.0,
        )
        target = torch.zeros(1, 4, 1, 1, 2)
        target[:, 0, 0, 0, 0] = 1.0
        reconstruction = torch.ones_like(target)
        spatial_weight = torch.tensor([3.0, 1.0]).view(1, 1, 1, 1, 2)
        channel_weight = torch.tensor([4.0, 0.2, 0.5, 0.2]).view(1, 4, 1, 1, 1)
        expected = (
            ((reconstruction - target) ** 2 * spatial_weight * channel_weight).mean()
            / (spatial_weight.mean() * channel_weight.mean())
        )

        reconstruction_loss = model.compute_loss(
            target, reconstruction, self._zero_posterior()
        )[1]

        torch.testing.assert_close(reconstruction_loss, expected)

    def test_legacy_mse_preserves_false_positive_and_mass_penalties(self):
        model = self._make_model(
            occupancy_loss_type="legacy_mse",
            occupied_weight=1.0,
            empty_weight=1.0,
            channel_weights=None,
            false_positive_weight=2.0,
            occupancy_mass_weight=3.0,
        )
        target = torch.zeros(1, 4, 1, 1, 2)
        reconstruction = torch.zeros_like(target)
        reconstruction[:, 0, 0, 0, 0] = 2.0
        mse = ((reconstruction - target) ** 2).mean()
        false_positive = (torch.relu(reconstruction[:, 0:1]) ** 2).mean()
        mass = torch.relu(reconstruction[:, 0:1]).mean()
        expected = mse + 2.0 * false_positive + 3.0 * mass

        reconstruction_loss = model.compute_loss(
            target, reconstruction, self._zero_posterior()
        )[1]

        torch.testing.assert_close(reconstruction_loss, expected)

    def test_presets_and_yaml_override_expose_new_loss_configuration(self):
        required = {
            "occupancy_loss_type",
            "occupancy_bce_weight",
            "occupancy_dice_weight",
            "occupancy_pos_weight_cap",
            "continuous_recon_weight",
        }
        for preset in (
            create_ultra_lightweight_vae_config(),
            create_lightweight_vae_config(),
            create_standard_vae_config(),
        ):
            self.assertTrue(required.issubset(preset))
            self.assertEqual(preset["occupancy_loss_type"], "bce_dice")

        class FakeConfig:
            @staticmethod
            def get(key, default=None):
                if key == "vae":
                    return {
                        "occupancy_loss_type": "legacy_mse",
                        "occupancy_bce_weight": 2.5,
                        "continuous_recon_weight": 0.25,
                    }
                return default

        merged = apply_vae_config_overrides(
            create_ultra_lightweight_vae_config(), FakeConfig()
        )
        self.assertEqual(merged["occupancy_loss_type"], "legacy_mse")
        self.assertEqual(merged["occupancy_bce_weight"], 2.5)
        self.assertEqual(merged["continuous_recon_weight"], 0.25)

    def test_yaml_latent_dim_override_builds_matching_vae_channels(self):
        class FakeConfig:
            @staticmethod
            def get(key, default=None):
                if key == "vae":
                    return {
                        "latent_dim": 8,
                        "unknown_architecture_key": 99,
                    }
                return default

        merged = apply_vae_config_overrides(
            create_ultra_lightweight_vae_config(), FakeConfig()
        )
        self.assertEqual(merged["latent_dim"], 8)
        self.assertNotIn("unknown_architecture_key", merged)

        model = VAE3D(**merged)
        self.assertEqual(model.latent_dim, 8)
        self.assertEqual(model.encoder.conv_out.out_channels, 16)
        self.assertEqual(model.decoder.conv_in.in_channels, 8)


class VAETrainerRobustnessTest(unittest.TestCase):
    @staticmethod
    def _make_trainer():
        trainer = OptimizedVAETrainer.__new__(OptimizedVAETrainer)
        trainer.model = _TrainerModel()
        trainer.device = torch.device("cpu")
        trainer.memory_opt = _TrainerMemory()
        trainer.optimizer = _CountingSGD(trainer.model.parameters())
        trainer.scheduler = _Scheduler()
        trainer.epochs = 1
        return trainer

    @staticmethod
    def _batch(marker):
        target = torch.full((1, 1, 1, 1, 1), marker, dtype=torch.float32)
        return target, torch.zeros_like(target)

    def test_train_epoch_skips_nonfinite_loss_and_accumulates_valid_batches(self):
        trainer = self._make_trainer()
        train_loader = [
            self._batch(1.0),
            self._batch(2.0),
            self._batch(3.0),
        ]

        loss, recon_loss, kl_loss = trainer.train_epoch(1, train_loader)

        self.assertEqual(trainer.optimizer.step_count, 1)
        self.assertAlmostEqual(loss, 2.0)
        self.assertAlmostEqual(recon_loss, 2.0)
        self.assertAlmostEqual(kl_loss, 0.0)
        self.assertEqual(trainer.last_epoch_valid_batch_count, 2)
        self.assertTrue(torch.isfinite(trainer.model.weight))

    def test_train_epoch_raises_when_no_batch_is_finite(self):
        trainer = self._make_trainer()
        train_loader = [
            self._batch(float("nan")),
            self._batch(float("inf")),
        ]

        with self.assertRaisesRegex(RuntimeError, "有效 batch"):
            trainer.train_epoch(1, train_loader)

        self.assertEqual(trainer.optimizer.step_count, 0)

    def test_tail_accumulation_rescales_single_valid_batch_gradient(self):
        trainer = self._make_trainer()
        train_loader = [
            self._batch(2.0),
            self._batch(0.2),
        ]

        trainer.train_epoch(1, train_loader)

        self.assertEqual(trainer.optimizer.step_count, 1)
        self.assertAlmostEqual(trainer.model.weight.item(), 0.98, places=6)

    def test_resume_migrates_legacy_metrics_csv_without_data_loss(self):
        trainer = OptimizedVAETrainer.__new__(OptimizedVAETrainer)
        trainer.is_resumed = True
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_dir = tmpdir
            trainer.log_file = os.path.join(tmpdir, "training.log")
            trainer.csv_file = os.path.join(tmpdir, "metrics.csv")
            legacy_rows = [
                ["epoch", "loss", "recon_loss", "kl_loss", "lr", "time_seconds"],
                ["1", "1.0", "0.9", "0.1", "0.001", "2.0"],
            ]
            with open(trainer.csv_file, "w", newline="") as f:
                csv.writer(f).writerows(legacy_rows)

            trainer._setup_logging()

            legacy_path = os.path.join(tmpdir, "metrics_legacy.csv")
            with open(legacy_path, newline="") as f:
                self.assertEqual(list(csv.reader(f)), legacy_rows)
            with open(trainer.csv_file, newline="") as f:
                header = next(csv.reader(f))
            self.assertEqual(
                header,
                [
                    "epoch", "loss", "recon_loss", "kl_loss",
                    "occ_bce_loss", "occ_dice_loss", "continuous_loss",
                    "lr", "time_seconds",
                ],
            )


if __name__ == "__main__":
    unittest.main()
