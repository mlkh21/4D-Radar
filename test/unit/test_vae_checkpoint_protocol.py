#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 VAE checkpoint 自描述协议、确定性数据划分和验证指标。
"""

import os
import random
import sys
import tempfile
import unittest

import torch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffusion_consistency_radar.cm.vae_3d import (
    VAE3D,
    build_vae_from_checkpoint,
    create_lightweight_vae_config,
    create_ultra_lightweight_vae_config,
    resolve_checkpoint_grid_config,
)
from diffusion_consistency_radar.scripts.unified_train import (
    OptimizedVAETrainer,
    OptimizedLDMTrainer,
    atomic_copy_file,
    atomic_torch_save,
    decoded_occupancy_auxiliary_loss,
    deterministic_split_indices,
    seed_training_run,
    micro_occupancy_metrics,
)


class VAECheckpointProtocolTest(unittest.TestCase):
    def test_training_seed_reproduces_python_numpy_torch_and_loader_generator(self):
        first_generator = seed_training_run(123)
        first = (
            random.random(),
            float(np.random.random()),
            torch.rand(3),
            torch.randperm(8, generator=first_generator),
        )

        second_generator = seed_training_run(123)
        second = (
            random.random(),
            float(np.random.random()),
            torch.rand(3),
            torch.randperm(8, generator=second_generator),
        )

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertTrue(torch.equal(first[3], second[3]))

    def test_training_seed_rejects_negative_values(self):
        with self.assertRaisesRegex(ValueError, "training_seed"):
            seed_training_run(-1)

    def test_lightweight_preset_keeps_historical_architecture(self):
        config = create_lightweight_vae_config()

        self.assertEqual(config["latent_dim"], 4)
        self.assertEqual(config["base_channels"], 24)

    def test_metadata_builds_lightweight_latent_dim_eight(self):
        config = create_lightweight_vae_config()
        config["latent_dim"] = 8
        config["base_channels"] = 32
        source = VAE3D(**config)
        checkpoint = {
            "model_state_dict": source.state_dict(),
            "vae_config": config,
            "vae_config_type": "lightweight",
            "occupancy_activation": "sigmoid",
        }

        loaded, metadata = build_vae_from_checkpoint(checkpoint)

        self.assertEqual(loaded.latent_dim, 8)
        self.assertEqual(loaded.encoder.conv_in.out_channels, config["base_channels"])
        self.assertEqual(metadata["vae_config_type"], "lightweight")
        self.assertEqual(metadata["occupancy_activation"], "sigmoid")
        self.assertEqual(loaded.occupancy_activation, "sigmoid")

    def test_checkpoint_grid_metadata_is_used_unless_explicitly_overridden(self):
        metadata = {
            "data_grid_config": {
                "target_size": [16, 64, 80],
                "source_pc_range": [0, -30, -5, 100, 30, 15],
                "model_pc_range": [0, -10, -3, 40, 10, 9],
            }
        }

        resolved = resolve_checkpoint_grid_config(metadata)
        overridden = resolve_checkpoint_grid_config(
            metadata,
            target_size=[8, 32, 32],
            model_pc_range=[0, -5, -2, 20, 5, 6],
        )

        self.assertEqual(resolved[0], (16, 64, 80))
        self.assertEqual(resolved[1], (0.0, -30.0, -5.0, 100.0, 30.0, 15.0))
        self.assertEqual(resolved[2], (0.0, -10.0, -3.0, 40.0, 10.0, 9.0))
        self.assertEqual(overridden[0], (8, 32, 32))
        self.assertEqual(overridden[1], resolved[1])
        self.assertEqual(overridden[2], (0.0, -5.0, -2.0, 20.0, 5.0, 6.0))

    def test_legacy_checkpoint_requires_and_accepts_explicit_fallback(self):
        config = create_ultra_lightweight_vae_config()
        checkpoint = {"model_state_dict": VAE3D(**config).state_dict()}

        with self.assertRaisesRegex(ValueError, "fallback"):
            build_vae_from_checkpoint(checkpoint)

        loaded, metadata = build_vae_from_checkpoint(
            checkpoint,
            fallback_config_type="ultra_lightweight",
        )
        self.assertEqual(loaded.latent_dim, 4)
        self.assertEqual(metadata["vae_config_type"], "ultra_lightweight")
        self.assertEqual(metadata["occupancy_activation"], "raw")
        self.assertEqual(loaded.occupancy_loss_type, "legacy_mse")
        self.assertEqual(metadata["vae_config"]["occupancy_loss_type"], "legacy_mse")

    def test_explicit_complete_fallback_override_is_respected(self):
        config = create_ultra_lightweight_vae_config()
        config["occupancy_loss_type"] = "bce_dice"
        source = VAE3D(**config)
        checkpoint = {"model_state_dict": source.state_dict()}

        loaded, metadata = build_vae_from_checkpoint(
            checkpoint,
            fallback_config_type="ultra_lightweight",
            fallback_vae_config=config,
        )

        self.assertEqual(loaded.occupancy_loss_type, "bce_dice")
        self.assertEqual(metadata["vae_config"]["occupancy_loss_type"], "bce_dice")
        self.assertEqual(metadata["occupancy_activation"], "sigmoid")

    def test_sigmoid_decoded_auxiliary_loss_prefers_good_negative_empty_logits(self):
        target = torch.zeros(1, 4, 1, 1, 1)
        zero_logits = torch.zeros_like(target)
        good_negative_logits = zero_logits.clone()
        good_negative_logits[:, 0:1] = -8.0

        zero_loss = decoded_occupancy_auxiliary_loss(
            zero_logits,
            target,
            occupancy_activation="sigmoid",
            false_positive_weight=1.0,
            mass_weight=1.0,
        )
        negative_loss = decoded_occupancy_auxiliary_loss(
            good_negative_logits,
            target,
            occupancy_activation="sigmoid",
            false_positive_weight=1.0,
            mass_weight=1.0,
        )

        self.assertLess(negative_loss.item(), zero_loss.item())

    def test_decoded_auxiliary_component_weights_remain_independent(self):
        target = torch.zeros(1, 4, 1, 1, 1)
        decoded = torch.zeros_like(target)
        decoded[:, 0:1] = 1.0

        fp_only = decoded_occupancy_auxiliary_loss(
            decoded,
            target,
            occupancy_activation="raw",
            reconstruction_weight=0.0,
            false_positive_weight=1.0,
            mass_weight=0.0,
        )

        self.assertEqual(fp_only.item(), 1.0)

    def test_unknown_constructor_key_is_reported_clearly(self):
        config = create_lightweight_vae_config()
        config["config_type"] = "lightweight"
        checkpoint = {"model_state_dict": {}, "vae_config": config}

        with self.assertRaisesRegex(ValueError, "config_type"):
            build_vae_from_checkpoint(checkpoint)

    def test_shape_mismatch_does_not_fallback_to_ultra(self):
        config = create_lightweight_vae_config()
        config["base_channels"] = 32
        source = VAE3D(**config)
        state = source.state_dict()
        state["encoder.conv_in.weight"] = state["encoder.conv_in.weight"][:1]
        checkpoint = {
            "model_state_dict": state,
            "vae_config": config,
            "vae_config_type": "lightweight",
        }

        with self.assertRaisesRegex(RuntimeError, "结构不匹配"):
            build_vae_from_checkpoint(
                checkpoint,
                fallback_config_type="ultra_lightweight",
            )

    def test_deterministic_split_has_non_empty_disjoint_partitions(self):
        first = deterministic_split_indices(10, train_split=0.8, split_seed=42)
        second = deterministic_split_indices(10, train_split=0.8, split_seed=42)

        self.assertEqual(first, second)
        train_indices, val_indices = first
        self.assertEqual(len(train_indices), 8)
        self.assertEqual(len(val_indices), 2)
        self.assertFalse(set(train_indices) & set(val_indices))
        self.assertEqual(set(train_indices) | set(val_indices), set(range(10)))

    def test_split_rejects_zero_length_partition(self):
        with self.assertRaisesRegex(ValueError, "至少需要 2"):
            deterministic_split_indices(1, train_split=0.8, split_seed=42)
        with self.assertRaisesRegex(ValueError, "train_split"):
            deterministic_split_indices(10, train_split=1.0, split_seed=42)

    def test_micro_metrics_accumulate_counts_before_division(self):
        target = torch.tensor([[[[[1.0, 1.0, 0.0, 0.0]]]]])
        probability = torch.tensor([[[[[0.9, 0.1, 0.8, 0.1]]]]])

        metrics = micro_occupancy_metrics(probability, target, threshold=0.5)

        self.assertEqual(metrics["intersection"], 1)
        self.assertEqual(metrics["union"], 3)
        self.assertEqual(metrics["target_positive"], 2)
        self.assertEqual(metrics["predicted_positive"], 2)

    def test_checkpoint_payload_contains_complete_protocol_fields(self):
        config = create_ultra_lightweight_vae_config()
        trainer = OptimizedVAETrainer.__new__(OptimizedVAETrainer)
        trainer.model = VAE3D(**config)
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters())
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=10
        )
        trainer.vae_model_config = config
        trainer.vae_config_type = "ultra_lightweight"
        trainer.data_grid_config = {
            "target_size": [32, 128, 128],
            "source_pc_range": [0, -20, -6, 120, 20, 10],
            "model_pc_range": [0, -20, -6, 40, 20, 10],
        }
        trainer.occupancy_activation = "sigmoid"

        payload = trainer._checkpoint_payload(
            epoch=3,
            loss=0.4,
            best_loss=0.3,
            best_iou=0.6,
        )

        required = {
            "model_state_dict",
            "optimizer_state_dict",
            "vae_config",
            "vae_config_type",
            "data_grid_config",
            "occupancy_activation",
            "best_loss",
            "best_iou",
        }
        self.assertTrue(required.issubset(payload))
        self.assertEqual(payload["vae_config"], config)
        self.assertEqual(payload["best_iou"], 0.6)

    def test_epoch_best_state_is_updated_before_payload_is_built(self):
        config = create_ultra_lightweight_vae_config()
        trainer = OptimizedVAETrainer.__new__(OptimizedVAETrainer)
        trainer.model = VAE3D(**config)
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters())
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=10
        )
        trainer.vae_model_config = config
        trainer.vae_config_type = "ultra_lightweight"
        trainer.data_grid_config = {}
        trainer.occupancy_activation = "sigmoid"
        trainer.best_loss = 1.0
        trainer.best_iou = 0.2

        improved_loss, improved_iou = trainer._update_best_metrics(
            loss=0.8,
            val_iou=0.4,
        )
        payload = trainer._checkpoint_payload(
            epoch=2,
            loss=0.8,
            best_loss=trainer.best_loss,
            best_iou=trainer.best_iou,
        )

        self.assertTrue(improved_loss)
        self.assertTrue(improved_iou)
        self.assertEqual(payload["best_loss"], 0.8)
        self.assertEqual(payload["best_iou"], 0.4)

    def test_checkpoint_payload_contains_scheduler_state(self):
        config = create_ultra_lightweight_vae_config()
        trainer = OptimizedVAETrainer.__new__(OptimizedVAETrainer)
        trainer.model = VAE3D(**config)
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.1)
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=10
        )
        trainer.vae_model_config = config
        trainer.vae_config_type = "ultra_lightweight"
        trainer.data_grid_config = {}
        trainer.occupancy_activation = "sigmoid"

        payload = trainer._checkpoint_payload(2, 0.8, 0.8, 0.4)

        self.assertIn("scheduler_state_dict", payload)

    def test_resume_restores_scheduler_and_best_metrics(self):
        config = create_ultra_lightweight_vae_config()
        source_model = VAE3D(**config)
        source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.1)
        source_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            source_optimizer, T_max=10
        )
        source_scheduler.step()
        source_scheduler.step()
        checkpoint = {
            "epoch": 2,
            "model_state_dict": source_model.state_dict(),
            "optimizer_state_dict": source_optimizer.state_dict(),
            "scheduler_state_dict": source_scheduler.state_dict(),
            "best_loss": 0.7,
            "best_iou": 0.5,
        }

        trainer = OptimizedVAETrainer.__new__(OptimizedVAETrainer)
        trainer.model = VAE3D(**config)
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.1)
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=10
        )
        trainer.device = torch.device("cpu")
        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            torch.save(checkpoint, handle.name)
            trainer._resume_from_checkpoint(handle.name)

        self.assertEqual(trainer.start_epoch, 3)
        self.assertEqual(trainer.best_loss, 0.7)
        self.assertEqual(trainer.best_iou, 0.5)
        self.assertEqual(trainer.scheduler.last_epoch, source_scheduler.last_epoch)
        self.assertEqual(
            trainer.optimizer.param_groups[0]["lr"],
            source_optimizer.param_groups[0]["lr"],
        )

    def test_legacy_resume_advances_scheduler_without_resetting_lr(self):
        config = create_ultra_lightweight_vae_config()
        source_model = VAE3D(**config)
        source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=0.03)
        checkpoint = {
            "epoch": 3,
            "model_state_dict": source_model.state_dict(),
            "optimizer_state_dict": source_optimizer.state_dict(),
            "best_loss": 0.6,
            "best_iou": 0.4,
        }
        trainer = OptimizedVAETrainer.__new__(OptimizedVAETrainer)
        trainer.model = VAE3D(**config)
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.1)
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=10
        )
        trainer.device = torch.device("cpu")

        with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
            torch.save(checkpoint, handle.name)
            trainer._resume_from_checkpoint(handle.name)

        self.assertEqual(trainer.scheduler.last_epoch, 3)
        self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 0.03)

    def test_atomic_save_and_alias_leave_no_temp_and_same_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = os.path.join(tmpdir, "vae_best_iou.pt")
            alias = os.path.join(tmpdir, "vae_best.pt")
            atomic_torch_save({"best_iou": 0.6}, source)
            atomic_copy_file(source, alias)

            self.assertEqual(torch.load(source), torch.load(alias))
            self.assertFalse(
                any(".tmp-" in name for name in os.listdir(tmpdir))
            )


if __name__ == "__main__":
    unittest.main()
