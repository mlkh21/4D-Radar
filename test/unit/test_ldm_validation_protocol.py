#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 LDM 独立验证、best 选择和 checkpoint 审计协议。"""

import os
import sys
import tempfile
import unittest
from unittest import mock

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffusion_consistency_radar.scripts.unified_train import (
    LDM_VALIDATION_PROTOCOL,
    LDM_VALIDATION_SELECTOR,
    OptimizedLDMTrainer,
    ldm_validation_is_improved,
    resolve_ldm_validation_config,
)


def _perfect_threshold_sweep(validation_config):
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
        for threshold in validation_config["threshold_candidates"]
    ]


class LDMValidationProtocolTest(unittest.TestCase):
    def test_selector_prioritizes_validation_iou_then_validation_loss(self):
        best = {
            "denoising_latent_loss": 0.4,
            "denoising_occupancy_iou": 0.6,
        }

        self.assertFalse(
            ldm_validation_is_improved(
                {
                    "denoising_latent_loss": 0.2,
                    "denoising_occupancy_iou": 0.5,
                },
                best,
            )
        )
        self.assertTrue(
            ldm_validation_is_improved(
                {
                    "denoising_latent_loss": 0.3,
                    "denoising_occupancy_iou": 0.7,
                },
                best,
            )
        )
        self.assertTrue(
            ldm_validation_is_improved(
                {
                    "denoising_latent_loss": 0.3,
                    "denoising_occupancy_iou": 0.6,
                },
                best,
            )
        )

    def test_formal_ldm_payload_records_validation_selector_and_metrics(self):
        trainer = OptimizedLDMTrainer.__new__(OptimizedLDMTrainer)
        trainer.model = torch.nn.Linear(2, 2)
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters())
        trainer.model_config = {
            "latent_dim": 4,
            "fusion_voxel_shape": [4, 8, 8],
            "fusion_latent_shape": [2, 4, 4],
            "fusion_pc_range": [0, -2, -1, 8, 2, 1],
        }
        trainer.data_grid_config = {
            "target_size": [4, 8, 8],
            "source_pc_range": [0, -4, -2, 16, 4, 2],
            "model_pc_range": [0, -2, -1, 8, 2, 1],
        }
        trainer.vae_checkpoint_sha256 = "a" * 64
        trainer.latent_dim = 4
        trainer.global_step = 2
        trainer._ldm_loss_config = lambda epoch: {"epoch": epoch}
        trainer.radar_normalization = {"protocol": "radar_normalization_v1"}
        trainer.radar_normalization_sha256 = "b" * 64
        trainer.validation_config = resolve_ldm_validation_config({})
        trainer.validation_selector = LDM_VALIDATION_SELECTOR
        trainer.best_val_iou = 0.6
        trainer.best_val_loss = 0.3
        trainer.last_validation_metrics = {
            "denoising_latent_loss": 0.35,
            "denoising_occupancy_iou": 0.55,
        }
        trainer.last_validation_threshold_sweep = _perfect_threshold_sweep(
            trainer.validation_config
        )

        payload = trainer._checkpoint_payload(epoch=2, loss=0.1, best_loss=0.1)

        validation = payload["ldm_validation"]
        self.assertEqual(validation["protocol"], LDM_VALIDATION_PROTOCOL)
        self.assertEqual(validation["selector"], LDM_VALIDATION_SELECTOR)
        self.assertEqual(validation["split"], "temporal_block_validation_suffix")
        self.assertEqual(
            validation["current"]["denoising_latent_loss"], 0.35
        )
        self.assertEqual(validation["best"]["denoising_occupancy_iou"], 0.6)
        self.assertNotEqual(validation["selector"], "min_train_loss")

    def test_validate_uses_fixed_noise_and_restores_training_mode(self):
        class IdentityVAE(torch.nn.Module):
            def get_latent(self, value):
                return value

            def decode(self, value):
                return value

        class FirstChannelDenoiser(torch.nn.Module):
            def forward(self, value, _sigma):
                return value[:, 0:1]

        class LegacyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.unet_3d = FirstChannelDenoiser()

        trainer = OptimizedLDMTrainer.__new__(OptimizedLDMTrainer)
        trainer.vae = IdentityVAE()
        trainer.model = LegacyModel()
        trainer.model.train()
        trainer.device = torch.device("cpu")
        trainer.memory_opt = mock.Mock(use_amp=False)
        trainer.occupancy_activation = "raw"
        trainer.validation_config = resolve_ldm_validation_config({})
        val_loader = [(
            torch.ones(1, 1, 1, 2, 2),
            torch.zeros(1, 1, 1, 2, 2),
        )]

        first = trainer.validate(val_loader)
        second = trainer.validate(val_loader)

        self.assertEqual(first, second)
        self.assertGreaterEqual(first["denoising_latent_loss"], 0.0)
        self.assertTrue(0.0 <= first["denoising_occupancy_iou"] <= 1.0)
        self.assertTrue(trainer.model.training)

    def test_resume_rejects_validation_mismatch_before_loading_state(self):
        trainer = OptimizedLDMTrainer.__new__(OptimizedLDMTrainer)
        trainer.device = torch.device("cpu")
        trainer.data_grid_config = {
            "target_size": [4, 8, 8],
            "source_pc_range": [0, -4, -2, 16, 4, 2],
            "model_pc_range": [0, -2, -1, 8, 2, 1],
        }
        trainer.radar_normalization = None
        trainer.radar_normalization_sha256 = ""
        trainer.allow_legacy_radar_units = True
        trainer.validation_config = resolve_ldm_validation_config({})
        trainer.validation_selector = LDM_VALIDATION_SELECTOR
        trainer.last_validation_metrics = None
        trainer.best_val_iou = float("-inf")
        trainer.best_val_loss = float("inf")
        trainer.model = mock.Mock()
        trainer.optimizer = mock.Mock()
        trainer._preloaded_resume_checkpoint = {
            "model_state_dict": {},
            "ldm_validation": {
                **trainer.validation_config,
                "selector": "min_train_loss",
                "current": {
                    "denoising_latent_loss": 0.4,
                    "denoising_occupancy_iou": 0.6,
                },
                "best": {
                    "denoising_latent_loss": 0.4,
                    "denoising_occupancy_iou": 0.6,
                },
            },
        }

        with mock.patch(
            "diffusion_consistency_radar.scripts.unified_train."
            "assert_checkpoint_radar_normalization"
        ), self.assertRaisesRegex(ValueError, "selector"):
            trainer._resume_from_checkpoint("unused.pt")

        trainer.model.load_state_dict.assert_not_called()
        trainer.optimizer.load_state_dict.assert_not_called()

    def test_train_consumes_independent_validation_and_ignores_better_train_loss(self):
        trainer = OptimizedLDMTrainer.__new__(OptimizedLDMTrainer)
        trainer.ldm_config = {"epochs": 2, "save_every": 99}
        trainer.start_epoch = 1
        trainer.global_step = 0
        trainer.best_loss = float("inf")
        trainer.best_val_iou = float("-inf")
        trainer.best_val_loss = float("inf")
        trainer.last_validation_metrics = None
        trainer.validation_config = {
            "protocol": LDM_VALIDATION_PROTOCOL,
            "split": "temporal_block_validation_suffix",
            "seed": 42,
            "sigma": 0.5,
            "occupancy_threshold": 0.5,
        }
        trainer.validation_selector = LDM_VALIDATION_SELECTOR
        trainer.save_dir = "/tmp/unused_ldm_validation_test"
        trainer.log_file = "/tmp/unused_ldm_validation_test.log"
        trainer.csv_file = "/tmp/unused_ldm_validation_test.csv"
        trainer.logger = mock.Mock()
        trainer.memory_opt = mock.Mock(grad_accum_steps=1)
        trainer.train_epoch = mock.Mock(side_effect=[0.4, 0.1])
        trainer.validate = mock.Mock(side_effect=[
            {
                "denoising_latent_loss": 0.4,
                "denoising_occupancy_iou": 0.7,
            },
            {
                "denoising_latent_loss": 0.2,
                "denoising_occupancy_iou": 0.6,
            },
        ])
        trainer._log_metrics = mock.Mock()
        trainer._checkpoint_payload = mock.Mock(
            side_effect=lambda epoch, loss, best_loss: {
                "epoch": epoch,
                "loss": loss,
                "best_loss": best_loss,
                "validation": dict(trainer.last_validation_metrics),
            }
        )

        train_loader = mock.Mock(batch_size=1)
        train_loader.__len__ = mock.Mock(return_value=1)
        val_loader = mock.Mock(batch_size=1)
        val_loader.__len__ = mock.Mock(return_value=1)

        with mock.patch(
            "diffusion_consistency_radar.scripts.unified_train.atomic_torch_save"
        ) as save_checkpoint:
            trainer.train(train_loader, val_loader)

        self.assertEqual(trainer.validate.call_args_list, [mock.call(val_loader), mock.call(val_loader)])
        self.assertEqual(trainer.best_val_iou, 0.7)
        self.assertEqual(trainer.best_val_loss, 0.4)
        self.assertEqual(save_checkpoint.call_count, 1)
        self.assertEqual(save_checkpoint.call_args.args[0]["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
