#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 LDM 解码结果的可微垂直结构损失。"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffusion_consistency_radar.scripts.unified_train import (
    LDM_METRICS_HEADER,
    LDM_LOSS_COMPONENT_NAMES,
    compute_ldm_loss_components,
    decoded_occupancy_auxiliary_loss,
    decoded_vertical_structure_losses,
    prepare_ldm_metrics_csv,
    rescale_accumulated_gradients,
)


def make_volume(z_size=6, channels=1, dtype=torch.float32):
    """构造布局为 [B,C,Z,X,Y] 的小型测试张量。"""
    return torch.zeros(1, channels, z_size, 2, 2, dtype=dtype)


class CountingVAE(torch.nn.Module):
    """记录 decode 调用次数的轻量 VAE 替身。"""

    def __init__(self, decoded):
        super().__init__()
        self.decoded = decoded
        self.decode_calls = 0

    def decode(self, latent):
        self.decode_calls += 1
        return self.decoded + latent.sum() * 0.0


class DecodedVerticalStructureLossTest(unittest.TestCase):
    def test_aligned_column_has_lower_losses_than_vertical_shifts(self):
        target = make_volume()
        target[:, 0, 2:4, 0, 0] = 1.0
        aligned = target.clone()
        shifted_down = make_volume()
        shifted_down[:, 0, 1:3, 0, 0] = 1.0
        shifted_up = make_volume()
        shifted_up[:, 0, 3:5, 0, 0] = 1.0

        aligned_losses = decoded_vertical_structure_losses(
            aligned, target, occupancy_activation="raw"
        )
        down_losses = decoded_vertical_structure_losses(
            shifted_down, target, occupancy_activation="raw"
        )
        up_losses = decoded_vertical_structure_losses(
            shifted_up, target, occupancy_activation="raw"
        )

        self.assertEqual(
            set(aligned_losses),
            {"height_distribution_loss", "vertical_continuity_loss"},
        )
        for name in aligned_losses:
            self.assertLess(aligned_losses[name].item(), down_losses[name].item())
            self.assertLess(aligned_losses[name].item(), up_losses[name].item())

    def test_solid_column_has_lower_continuity_loss_than_middle_gap(self):
        target = make_volume()
        target[:, 0, 1:5, 0, 0] = 1.0
        solid = target.clone()
        middle_gap = target.clone()
        middle_gap[:, 0, 3, 0, 0] = 0.0

        solid_loss = decoded_vertical_structure_losses(
            solid, target, occupancy_activation="raw"
        )["vertical_continuity_loss"]
        gap_loss = decoded_vertical_structure_losses(
            middle_gap, target, occupancy_activation="raw"
        )["vertical_continuity_loss"]

        self.assertLess(solid_loss.item(), gap_loss.item())

    def test_empty_target_returns_graph_connected_finite_zeros(self):
        decoded = torch.randn(2, 2, 4, 2, 3, requires_grad=True)
        target = torch.zeros(2, 1, 4, 2, 3)

        losses = decoded_vertical_structure_losses(
            decoded, target, occupancy_activation="raw"
        )
        total = sum(losses.values())
        total.backward()

        for loss in losses.values():
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(loss.item(), 0.0)
        self.assertIsNotNone(decoded.grad)
        self.assertTrue(torch.isfinite(decoded.grad).all())

    def test_sigmoid_logits_receive_nonzero_finite_gradient(self):
        logits = make_volume().requires_grad_()
        target = make_volume()
        target[:, 0, 4, 0, 0] = 1.0
        with torch.no_grad():
            logits[:, 0, 1, 0, 0] = 2.0
            logits[:, 0, 4, 0, 0] = -1.0

        losses = decoded_vertical_structure_losses(
            logits, target, occupancy_activation="sigmoid"
        )
        sum(losses.values()).backward()

        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_float16_extreme_sigmoid_logits_are_activated_in_float32(self):
        logits = torch.full(
            (1, 1, 1, 1, 1),
            -20.0,
            dtype=torch.float16,
            requires_grad=True,
        )
        target = torch.ones_like(logits)

        losses = decoded_vertical_structure_losses(
            logits,
            target,
            occupancy_activation="sigmoid",
            eps=1e-12,
        )
        sum(losses.values()).backward()

        for loss in losses.values():
            self.assertTrue(torch.isfinite(loss))
        self.assertLess(losses["height_distribution_loss"].item(), 1e-4)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_zero_raw_prediction_has_bounded_finite_gradient(self):
        decoded = make_volume().requires_grad_()
        target = make_volume()
        target[:, 0, 4, 0, 0] = 1.0

        losses = decoded_vertical_structure_losses(
            decoded, target, occupancy_activation="raw"
        )
        sum(losses.values()).backward()

        for loss in losses.values():
            self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(decoded.grad)
        self.assertTrue(torch.isfinite(decoded.grad).all())
        self.assertLessEqual(decoded.grad.abs().max().item(), 10.0)

    def test_raw_path_clamps_soft_occupancy_and_returns_float32_losses(self):
        decoded = torch.tensor(
            [[[[[2.0]], [[-1.0]], [[0.5]]]]], dtype=torch.float64
        )
        target = torch.tensor(
            [[[[[1.5]], [[-0.5]], [[0.25]]]]], dtype=torch.float64
        )

        losses = decoded_vertical_structure_losses(
            decoded, target, occupancy_activation="raw"
        )

        for loss in losses.values():
            self.assertEqual(loss.dtype, torch.float32)
            self.assertTrue(torch.isfinite(loss))

    def test_invalid_activation_is_rejected(self):
        volume = make_volume()

        with self.assertRaisesRegex(ValueError, "occupancy_activation"):
            decoded_vertical_structure_losses(
                volume, volume, occupancy_activation="relu"
            )

    def test_eps_must_be_finite_and_positive(self):
        volume = make_volume()

        for eps in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(eps=eps):
                with self.assertRaisesRegex(ValueError, "eps"):
                    decoded_vertical_structure_losses(
                        volume, volume, occupancy_activation="raw", eps=eps
                    )

    def test_invalid_dimensions_and_spatial_mismatch_are_rejected(self):
        volume = make_volume()

        with self.assertRaisesRegex(ValueError, "5"):
            decoded_vertical_structure_losses(
                volume.squeeze(0), volume, occupancy_activation="raw"
            )
        with self.assertRaisesRegex(ValueError, "至少 1 个通道"):
            decoded_vertical_structure_losses(
                volume[:, :0], volume, occupancy_activation="raw"
            )
        with self.assertRaisesRegex(ValueError, "B/Z/X/Y"):
            decoded_vertical_structure_losses(
                volume, make_volume(z_size=5), occupancy_activation="raw"
            )

    def test_single_z_layer_has_zero_graph_connected_continuity_loss(self):
        decoded = make_volume(z_size=1).requires_grad_()
        target = make_volume(z_size=1)
        target[:, 0, 0, 0, 0] = 1.0

        continuity = decoded_vertical_structure_losses(
            decoded, target, occupancy_activation="sigmoid"
        )["vertical_continuity_loss"]
        continuity.backward()

        self.assertEqual(continuity.item(), 0.0)
        self.assertIsNotNone(decoded.grad)
        self.assertTrue(torch.isfinite(decoded.grad).all())


class LDMDecodedStructureIntegrationTest(unittest.TestCase):
    def test_zero_decoded_weights_do_not_decode_and_keep_latent_only_loss(self):
        denoised = torch.zeros(1, 1, 1, 1, 1)
        z_target = torch.ones_like(denoised)
        target = make_volume(z_size=1)
        vae = CountingVAE(decoded=target.clone())

        loss, components = compute_ldm_loss_components(
            denoised,
            z_target,
            target,
            vae=vae,
            occupancy_activation="raw",
            decoded_loss_weight=0.0,
            decoded_false_positive_weight=0.0,
            decoded_mass_weight=0.0,
            decoded_height_distribution_weight=0.0,
            decoded_vertical_continuity_weight=0.0,
            uncertainty_loss_weight=0.0,
            uncertainty=None,
        )

        self.assertEqual(vae.decode_calls, 0)
        self.assertAlmostEqual(loss.item(), 1.0)
        self.assertAlmostEqual(components["latent_loss"].item(), 1.0)
        for name in (
            "decoded_occupancy_loss",
            "height_distribution_loss",
            "vertical_continuity_loss",
            "uncertainty_loss",
        ):
            self.assertEqual(components[name].item(), 0.0)

    def test_structure_weights_decode_once_and_add_independent_contributions(self):
        denoised = torch.zeros(1, 1, 1, 1, 1)
        z_target = torch.zeros_like(denoised)
        target = make_volume(z_size=3)
        target[:, 0, 2, 0, 0] = 1.0
        decoded = make_volume(z_size=3)
        decoded[:, 0, 0, 0, 0] = 1.0
        vae = CountingVAE(decoded=decoded)

        height_weight = 0.25
        continuity_weight = 0.75
        loss, components = compute_ldm_loss_components(
            denoised,
            z_target,
            target,
            vae=vae,
            occupancy_activation="raw",
            decoded_loss_weight=0.0,
            decoded_false_positive_weight=0.0,
            decoded_mass_weight=0.0,
            decoded_height_distribution_weight=height_weight,
            decoded_vertical_continuity_weight=continuity_weight,
            uncertainty_loss_weight=0.0,
            uncertainty=None,
        )

        self.assertEqual(vae.decode_calls, 1)
        self.assertGreater(components["height_distribution_loss"].item(), 0.0)
        self.assertGreater(components["vertical_continuity_loss"].item(), 0.0)
        expected = (
            height_weight * components["height_distribution_loss"]
            + continuity_weight * components["vertical_continuity_loss"]
        )
        self.assertTrue(torch.allclose(loss, expected))

    def test_zero_structure_weights_keep_occupancy_only_loss_behavior(self):
        denoised = torch.zeros(1, 1, 1, 1, 1)
        z_target = torch.zeros_like(denoised)
        target = make_volume(z_size=2)
        target[:, 0, 1, 0, 0] = 1.0
        decoded = make_volume(z_size=2)
        decoded[:, 0, 0, 0, 0] = 1.0
        vae = CountingVAE(decoded=decoded)

        loss, components = compute_ldm_loss_components(
            denoised,
            z_target,
            target,
            vae=vae,
            occupancy_activation="raw",
            decoded_loss_weight=0.5,
            decoded_false_positive_weight=0.25,
            decoded_mass_weight=0.0,
            decoded_height_distribution_weight=0.0,
            decoded_vertical_continuity_weight=0.0,
            uncertainty_loss_weight=0.0,
            uncertainty=None,
        )
        expected = decoded_occupancy_auxiliary_loss(
            decoded,
            target,
            occupancy_activation="raw",
            reconstruction_weight=0.5,
            false_positive_weight=0.25,
            mass_weight=0.0,
        )

        self.assertEqual(vae.decode_calls, 1)
        self.assertTrue(torch.allclose(loss, expected))
        self.assertTrue(torch.allclose(components["decoded_occupancy_loss"], expected))
        self.assertEqual(components["height_distribution_loss"].item(), 0.0)
        self.assertEqual(components["vertical_continuity_loss"].item(), 0.0)

    def test_component_contract_contains_all_ldm_loss_terms(self):
        expected = {
            "latent_loss",
            "decoded_occupancy_loss",
            "height_distribution_loss",
            "vertical_continuity_loss",
            "uncertainty_loss",
        }

        self.assertTrue(expected.issubset(set(LDM_LOSS_COMPONENT_NAMES)))
        self.assertTrue(expected.issubset(set(LDM_METRICS_HEADER)))

    def test_uncertainty_loss_requires_variance_tensor(self):
        denoised = torch.zeros(1, 1, 1, 1, 1)
        z_target = torch.zeros_like(denoised)
        target = make_volume(z_size=1)
        vae = CountingVAE(decoded=target.clone())

        with self.assertRaisesRegex(ValueError, "variance"):
            compute_ldm_loss_components(
                denoised,
                z_target,
                target,
                vae=vae,
                occupancy_activation="raw",
                decoded_loss_weight=0.0,
                decoded_false_positive_weight=0.0,
                decoded_mass_weight=0.0,
                decoded_height_distribution_weight=0.0,
                decoded_vertical_continuity_weight=0.0,
                uncertainty_loss_weight=0.1,
                uncertainty={},
            )


class LDMTrainerUtilityTest(unittest.TestCase):
    def test_prepare_metrics_csv_archives_legacy_resume_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "metrics.csv"
            csv_path.write_text(
                "epoch,step,loss,lr,time_seconds\n1,1,2.0,0.001,3.0\n",
                encoding="utf-8",
            )

            prepare_ldm_metrics_csv(str(csv_path), is_resumed=True)

            self.assertEqual(
                csv_path.read_text(encoding="utf-8").strip(),
                ",".join(LDM_METRICS_HEADER),
            )
            legacy_path = Path(temp_dir) / "metrics_legacy.csv"
            self.assertTrue(legacy_path.exists())
            self.assertIn(
                "epoch,step,loss,lr,time_seconds",
                legacy_path.read_text(encoding="utf-8"),
            )

    def test_rescale_accumulated_gradients_restores_tail_batch_average(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        grad_accum_steps = 4
        accumulation_count = 2

        for _ in range(accumulation_count):
            loss = model(torch.ones(1, 1)).sum()
            (loss / grad_accum_steps).backward()

        self.assertAlmostEqual(model.weight.grad.item(), 0.5)
        rescale_accumulated_gradients(
            model.parameters(),
            grad_accum_steps=grad_accum_steps,
            accumulation_count=accumulation_count,
        )

        self.assertAlmostEqual(model.weight.grad.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
