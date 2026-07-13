#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 LDM 解码结果的可微垂直结构损失。"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffusion_consistency_radar.scripts.unified_train import (
    LDM_METRICS_HEADER,
    LDM_LOSS_COMPONENT_NAMES,
    OptimizedLDMTrainer,
    compute_ldm_loss_components,
    decoded_column_balanced_losses,
    decoded_density_precision_loss,
    decoded_ir_frustum_negative_occupancy_loss,
    decoded_ir_frustum_occupancy_loss,
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


class TrainerVAE(torch.nn.Module):
    """提供 trainer 初始化所需协议的轻量 VAE。"""

    latent_dim = 1
    occupancy_activation = "raw"

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))


class DecodedColumnBalancedLossTest(unittest.TestCase):
    """验证 Z 列存在性的正负类独立监督。"""

    def test_matched_columns_beat_missing_positive_and_false_positive(self):
        target = make_volume(z_size=3)
        target[:, 0, 1, 0, 0] = 1.0
        matched = torch.full_like(target, -8.0)
        matched[:, 0, 1, 0, 0] = 8.0
        missing = matched.clone()
        missing[:, 0, :, 0, 0] = -8.0
        false_positive = matched.clone()
        false_positive[:, 0, :, 1, 1] = 8.0

        matched_losses = decoded_column_balanced_losses(
            matched, target, occupancy_activation="sigmoid"
        )
        missing_losses = decoded_column_balanced_losses(
            missing, target, occupancy_activation="sigmoid"
        )
        false_positive_losses = decoded_column_balanced_losses(
            false_positive, target, occupancy_activation="sigmoid"
        )

        self.assertEqual(
            set(matched_losses), {"positive_loss", "negative_loss"}
        )
        self.assertLess(
            matched_losses["positive_loss"],
            missing_losses["positive_loss"],
        )
        self.assertLess(
            matched_losses["negative_loss"],
            false_positive_losses["negative_loss"],
        )

    def test_positive_and_negative_means_are_independent_of_class_counts(self):
        target_small = torch.zeros(1, 1, 2, 1, 2)
        target_small[:, :, 0, 0, 0] = 1.0
        decoded_small = torch.tensor([[[[[2.0, -3.0]], [[2.0, -3.0]]]]])
        target_large = torch.zeros(1, 1, 2, 1, 5)
        target_large[:, :, 0, 0, 0] = 1.0
        decoded_large = torch.empty_like(target_large)
        decoded_large[:, :, :, :, 0] = 2.0
        decoded_large[:, :, :, :, 1:] = -3.0

        small = decoded_column_balanced_losses(
            decoded_small, target_small, occupancy_activation="sigmoid"
        )
        large = decoded_column_balanced_losses(
            decoded_large, target_large, occupancy_activation="sigmoid"
        )

        self.assertTrue(torch.allclose(small["positive_loss"], large["positive_loss"]))
        self.assertTrue(torch.allclose(small["negative_loss"], large["negative_loss"]))

    def test_soft_target_uses_inclusive_threshold_and_any_z(self):
        target = torch.zeros(1, 1, 3, 1, 2)
        target[:, :, 2, 0, 0] = 0.5
        target[:, :, 0, 0, 1] = 0.49
        decoded = torch.tensor([[[[[-4.0, 4.0]], [[-4.0, 4.0]], [[4.0, 4.0]]]]])

        losses = decoded_column_balanced_losses(
            decoded,
            target,
            occupancy_activation="sigmoid",
            target_threshold=0.5,
        )

        self.assertLess(losses["positive_loss"].item(), 0.1)
        self.assertGreater(losses["negative_loss"].item(), 3.0)

    def test_temperature_changes_logmeanexp_column_aggregation(self):
        decoded = torch.tensor([[[[[0.0]], [[2.0]]]]])
        target = torch.ones_like(decoded)

        cold = decoded_column_balanced_losses(
            decoded, target, "sigmoid", temperature=0.25
        )["positive_loss"]
        warm = decoded_column_balanced_losses(
            decoded, target, "sigmoid", temperature=2.0
        )["positive_loss"]

        self.assertLess(cold, warm)

    def test_extreme_sigmoid_and_raw_inputs_are_finite(self):
        for activation, values in (
            ("sigmoid", [-1000.0, 1000.0]),
            ("raw", [0.0, 1.0]),
        ):
            with self.subTest(activation=activation):
                decoded = torch.tensor(values, dtype=torch.float32).view(1, 1, 2, 1, 1)
                decoded.requires_grad_()
                target = torch.ones_like(decoded)
                losses = decoded_column_balanced_losses(decoded, target, activation)
                total = sum(losses.values())
                total.backward()

                self.assertTrue(torch.isfinite(total))
                self.assertIsNotNone(decoded.grad)
                self.assertTrue(torch.isfinite(decoded.grad).all())

    def test_unsaturated_sigmoid_and_raw_inputs_have_nonzero_gradients(self):
        for activation, values in (
            ("sigmoid", [-4.0, 4.0]),
            ("raw", [0.1, 0.9]),
        ):
            with self.subTest(activation=activation):
                decoded = torch.tensor(values).view(1, 1, 2, 1, 1)
                decoded.requires_grad_()
                target = torch.ones_like(decoded)

                sum(decoded_column_balanced_losses(decoded, target, activation).values()).backward()

                self.assertTrue(torch.isfinite(decoded.grad).all())
                self.assertGreater(decoded.grad.abs().max().item(), 0.0)

    def test_negative_loss_pushes_empty_column_logits_down(self):
        target = torch.zeros(1, 1, 2, 1, 1)
        decoded = torch.tensor([0.5, -0.5]).view_as(target).requires_grad_()

        loss = decoded_column_balanced_losses(decoded, target, "sigmoid")[
            "negative_loss"
        ]
        loss.backward()
        lower_logits_loss = decoded_column_balanced_losses(
            decoded.detach() - 1.0, target, "sigmoid"
        )["negative_loss"]

        self.assertTrue(torch.all(decoded.grad > 0.0))
        self.assertLess(lower_logits_loss.item(), loss.item())

    def test_raw_probabilities_use_fixed_clamp_before_logit(self):
        decoded = torch.tensor([0.0, 1.0]).view(1, 1, 1, 1, 2)
        target = torch.tensor([1.0, 0.0]).view_as(decoded)

        losses = decoded_column_balanced_losses(decoded, target, "raw")
        expected_lower_logit = torch.logit(torch.tensor(1e-6))
        expected_upper_logit = torch.logit(torch.tensor(1.0 - 1e-6))

        self.assertTrue(
            torch.allclose(
                losses["positive_loss"],
                torch.nn.functional.softplus(-expected_lower_logit),
            )
        )
        self.assertTrue(
            torch.allclose(
                losses["negative_loss"],
                torch.nn.functional.softplus(expected_upper_logit),
            )
        )

    def test_all_empty_and_all_positive_return_graph_connected_class_zero(self):
        for target_value, zero_name in (
            (0.0, "positive_loss"),
            (1.0, "negative_loss"),
        ):
            with self.subTest(target_value=target_value):
                decoded = torch.randn(1, 1, 2, 1, 2, requires_grad=True)
                target = torch.full_like(decoded, target_value)
                losses = decoded_column_balanced_losses(decoded, target, "sigmoid")
                losses[zero_name].backward()

                self.assertEqual(losses[zero_name].item(), 0.0)
                self.assertIsNotNone(decoded.grad)
                self.assertTrue(torch.isfinite(decoded.grad).all())

    def test_temperature_bounds_are_finite_and_invalid_values_are_rejected(self):
        volume = make_volume(z_size=2)
        for temperature in (1e-3, 100.0):
            with self.subTest(temperature=temperature):
                losses = decoded_column_balanced_losses(
                    volume, volume, "raw", temperature
                )
                self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))
        for temperature in (9e-4, 100.1, float("nan"), float("inf")):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(ValueError, r"temperature.*\[1e-3,100\.0\]"):
                    decoded_column_balanced_losses(volume, volume, "raw", temperature)

    def test_uses_shared_layout_and_activation_validation(self):
        volume = make_volume(z_size=2)
        with self.assertRaisesRegex(ValueError, "5 维"):
            decoded_column_balanced_losses(volume.squeeze(0), volume, "raw")
        with self.assertRaisesRegex(ValueError, "B/Z/X/Y"):
            decoded_column_balanced_losses(volume, make_volume(z_size=3), "raw")
        with self.assertRaisesRegex(ValueError, "occupancy_activation"):
            decoded_column_balanced_losses(volume, volume, "softmax")

    def test_rejects_empty_spatial_dimensions(self):
        for dimension in (2, 3, 4):
            with self.subTest(dimension=dimension):
                shape = [1, 1, 2, 2, 2]
                shape[dimension] = 0
                volume = torch.empty(shape)
                with self.assertRaisesRegex(ValueError, "Z/X/Y.*大于 0"):
                    decoded_column_balanced_losses(volume, volume, "raw")

    def test_rejects_decoded_target_device_mismatch(self):
        decoded = make_volume(z_size=2)
        target = torch.empty(decoded.shape, device="meta")

        with self.assertRaisesRegex(ValueError, "device"):
            decoded_column_balanced_losses(decoded, target, "raw")

    def test_uses_exact_temperature_scaled_logmeanexp_column_logits(self):
        temperature = 0.5
        positive_logits = torch.tensor([-1.0, 0.5, 2.0])
        negative_logits = torch.tensor([-2.0, -0.5, 1.0])
        decoded = torch.stack((positive_logits, negative_logits), dim=-1).view(
            1, 1, 3, 1, 2
        )
        target = torch.zeros_like(decoded)
        target[:, :, 0, 0, 0] = 1.0

        losses = decoded_column_balanced_losses(
            decoded, target, "sigmoid", temperature=temperature
        )
        expected_column_logits = temperature * (
            torch.logsumexp(decoded[:, 0] / temperature, dim=1)
            - torch.log(torch.tensor(3.0))
        )

        self.assertTrue(
            torch.allclose(
                losses["positive_loss"],
                torch.nn.functional.softplus(-expected_column_logits[0, 0, 0]),
            )
        )
        self.assertTrue(
            torch.allclose(
                losses["negative_loss"],
                torch.nn.functional.softplus(expected_column_logits[0, 0, 1]),
            )
        )

    def test_shared_validation_rejects_invalid_target_layout_and_batch(self):
        volume = make_volume(z_size=2)
        with self.assertRaisesRegex(ValueError, "5 维"):
            decoded_column_balanced_losses(volume, volume.squeeze(0), "raw")
        with self.assertRaisesRegex(ValueError, "至少 1 个通道"):
            decoded_column_balanced_losses(volume, volume[:, :0], "raw")
        with self.assertRaisesRegex(ValueError, "B/Z/X/Y"):
            decoded_column_balanced_losses(
                volume.expand(2, -1, -1, -1, -1), volume, "raw"
            )

    def test_rejects_target_threshold_outside_unit_interval(self):
        volume = make_volume(z_size=2)
        for threshold in (-1e-6, 1.0 + 1e-6, float("nan"), float("inf")):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, "target_threshold"):
                    decoded_column_balanced_losses(
                        volume,
                        volume,
                        "raw",
                        target_threshold=threshold,
                    )


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
            {
                "height_distribution_loss",
                "top_height_loss",
                "top_overshoot_loss",
                "vertical_continuity_loss",
            },
        )
        for name in ("height_distribution_loss", "vertical_continuity_loss"):
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

    def test_predicted_target_top_voxel_has_lower_top_height_loss(self):
        target = make_volume(z_size=5)
        target[:, 0, 1:4, 0, 0] = 1.0
        misses_top = make_volume(z_size=5)
        misses_top[:, 0, 1:3, 0, 0] = 1.0
        hits_top = misses_top.clone()
        hits_top[:, 0, 3, 0, 0] = 1.0

        miss_loss = decoded_vertical_structure_losses(
            misses_top, target, occupancy_activation="raw"
        )["top_height_loss"]
        hit_loss = decoded_vertical_structure_losses(
            hits_top, target, occupancy_activation="raw"
        )["top_height_loss"]

        self.assertLess(hit_loss.item(), miss_loss.item())

    def test_top_overshoot_loss_penalizes_only_voxels_above_target_top(self):
        target = make_volume(z_size=5)
        target[:, 0, 1:3, 0, 0] = 1.0
        aligned = target.clone()
        below_and_top = make_volume(z_size=5)
        below_and_top[:, 0, :3, 0, 0] = 1.0
        overshoot = aligned.clone()
        overshoot[:, 0, 3, 0, 0] = 1.0

        aligned_loss = decoded_vertical_structure_losses(
            aligned, target, occupancy_activation="raw"
        )["top_overshoot_loss"]
        below_and_top_loss = decoded_vertical_structure_losses(
            below_and_top, target, occupancy_activation="raw"
        )["top_overshoot_loss"]
        overshoot_loss = decoded_vertical_structure_losses(
            overshoot, target, occupancy_activation="raw"
        )["top_overshoot_loss"]

        self.assertLess(aligned_loss.item(), 1e-6)
        self.assertLess(below_and_top_loss.item(), 1e-6)
        self.assertGreater(overshoot_loss.item(), aligned_loss.item())

    def test_top_overshoot_sigmoid_extreme_logits_are_finite_with_gradient(self):
        logits = torch.full(
            (1, 1, 5, 1, 1),
            -20.0,
            dtype=torch.float16,
            requires_grad=True,
        )
        target = torch.zeros_like(logits)
        target[:, 0, 2, 0, 0] = 1.0
        with torch.no_grad():
            logits[:, 0, 3:, 0, 0] = 20.0

        loss = decoded_vertical_structure_losses(
            logits, target, occupancy_activation="sigmoid"
        )["top_overshoot_loss"]
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[:, :, 3:].abs().sum().item(), 0.0)
        self.assertEqual(logits.grad[:, :, :3].abs().sum().item(), 0.0)

    def test_column_mask_limits_top_overshoot_supervision(self):
        target = make_volume(z_size=4)
        target[:, 0, 1, 0, 0] = 1.0
        target[:, 0, 1, 1, 1] = 1.0
        decoded = target.clone()
        decoded[:, 0, 2, 1, 1] = 1.0
        mask = torch.zeros(1, 1, 4, 2, 2, dtype=torch.bool)
        mask[:, :, :, 0, 0] = True

        masked_loss = decoded_vertical_structure_losses(
            decoded,
            target,
            occupancy_activation="raw",
            column_mask=mask,
        )["top_overshoot_loss"]
        unmasked_loss = decoded_vertical_structure_losses(
            decoded, target, occupancy_activation="raw"
        )["top_overshoot_loss"]

        self.assertEqual(masked_loss.item(), 0.0)
        self.assertGreater(unmasked_loss.item(), masked_loss.item())

    def test_target_at_highest_z_has_graph_connected_zero_overshoot_loss(self):
        decoded = torch.randn(1, 1, 4, 1, 1, requires_grad=True)
        target = torch.zeros_like(decoded)
        target[:, 0, -1, 0, 0] = 1.0

        loss = decoded_vertical_structure_losses(
            decoded, target, occupancy_activation="sigmoid"
        )["top_overshoot_loss"]
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.0)
        self.assertIsNotNone(decoded.grad)
        self.assertTrue(torch.isfinite(decoded.grad).all())

    def test_ir_frustum_column_mask_limits_top_height_supervision(self):
        target = make_volume(z_size=4)
        decoded = make_volume(z_size=4)
        target[:, 0, 3, 0, 0] = 1.0
        target[:, 0, 3, 1, 1] = 1.0
        decoded[:, 0, 3, 0, 0] = 1.0
        mask = torch.zeros(1, 1, 4, 2, 2, dtype=torch.bool)
        mask[:, :, :, 0, 0] = True

        masked_losses = decoded_vertical_structure_losses(
            decoded,
            target,
            occupancy_activation="raw",
            column_mask=mask,
        )
        unmasked_losses = decoded_vertical_structure_losses(
            decoded,
            target,
            occupancy_activation="raw",
        )

        self.assertLess(
            masked_losses["top_height_loss"].item(),
            unmasked_losses["top_height_loss"].item(),
        )

    def test_ir_frustum_occupancy_loss_only_uses_visible_positive_voxels(self):
        target = make_volume(z_size=4)
        decoded = make_volume(z_size=4)
        target[:, 0, 2, 0, 0] = 1.0
        target[:, 0, 2, 1, 1] = 1.0
        decoded[:, 0, 2, 0, 0] = 1.0
        mask = torch.zeros(1, 1, 4, 2, 2, dtype=torch.bool)
        mask[:, :, :, 0, 0] = True

        loss = decoded_ir_frustum_occupancy_loss(
            decoded,
            target,
            occupancy_activation="raw",
            frustum_mask=mask,
        )

        self.assertLess(loss.item(), 1e-6)

    def test_ir_frustum_losses_reject_mask_batch_or_channel_mismatch(self):
        decoded = make_volume(z_size=2)
        target = make_volume(z_size=2)
        loss_functions = (
            decoded_ir_frustum_occupancy_loss,
            decoded_ir_frustum_negative_occupancy_loss,
        )
        invalid_masks = (
            (torch.ones(2, 1, 2, 2, 2, dtype=torch.bool), "batch"),
            (torch.ones(1, 2, 2, 2, 2, dtype=torch.bool), "channel"),
        )

        for loss_function in loss_functions:
            for frustum_mask, expected_message in invalid_masks:
                with self.subTest(
                    loss_function=loss_function.__name__,
                    mismatch=expected_message,
                ):
                    with self.assertRaisesRegex(ValueError, expected_message):
                        loss_function(
                            decoded,
                            target,
                            occupancy_activation="raw",
                            frustum_mask=frustum_mask,
                        )

    def test_ir_frustum_losses_reject_invalid_decoded_or_target_shape(self):
        volume = make_volume(z_size=2)
        loss_functions = (
            decoded_ir_frustum_occupancy_loss,
            decoded_ir_frustum_negative_occupancy_loss,
        )
        invalid_inputs = (
            (volume.squeeze(0), volume, "5"),
            (volume, volume.squeeze(0), "5"),
            (volume[:, :0], volume, "至少 1 个通道"),
            (volume, volume[:, :0], "至少 1 个通道"),
            (volume.expand(2, -1, -1, -1, -1), volume, "B/Z/X/Y"),
            (volume, make_volume(z_size=3), "B/Z/X/Y"),
        )

        for loss_function in loss_functions:
            for decoded, target, expected_message in invalid_inputs:
                with self.subTest(
                    loss_function=loss_function.__name__,
                    expected_message=expected_message,
                    decoded_shape=tuple(decoded.shape),
                    target_shape=tuple(target.shape),
                ):
                    with self.assertRaisesRegex(ValueError, expected_message):
                        loss_function(
                            decoded,
                            target,
                            occupancy_activation="raw",
                            frustum_mask=None,
                        )

    def test_ir_frustum_losses_reject_invalid_activation_without_mask(self):
        volume = make_volume(z_size=2)

        for loss_function in (
            decoded_ir_frustum_occupancy_loss,
            decoded_ir_frustum_negative_occupancy_loss,
        ):
            with self.subTest(loss_function=loss_function.__name__):
                with self.assertRaisesRegex(ValueError, "occupancy_activation"):
                    loss_function(
                        volume,
                        volume,
                        occupancy_activation="relu",
                        frustum_mask=None,
                    )

    def test_ir_frustum_negative_loss_only_uses_visible_negative_voxels(self):
        target = make_volume(z_size=3)
        decoded = make_volume(z_size=3)
        mask = torch.zeros(1, 1, 3, 2, 2, dtype=torch.bool)
        mask[:, :, 0, 0, 0] = True
        mask[:, :, 1, 0, 0] = True
        target[:, 0, 1, 0, 0] = 1.0

        decoded[:, 0, 0, 0, 0] = 0.5
        decoded[:, 0, 1, 0, 0] = 1.0
        decoded[:, 0, 2, 1, 1] = 1.0
        loss = decoded_ir_frustum_negative_occupancy_loss(
            decoded,
            target,
            occupancy_activation="raw",
            frustum_mask=mask,
        )

        self.assertAlmostEqual(loss.item(), 0.25)

    def test_ir_frustum_negative_sigmoid_extreme_logits_are_finite_with_gradient(self):
        logits = torch.full(
            (1, 1, 2, 1, 1),
            -20.0,
            dtype=torch.float16,
            requires_grad=True,
        )
        target = torch.zeros_like(logits)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[:, :, 1] = True
        with torch.no_grad():
            logits[:, :, 1] = 20.0

        loss = decoded_ir_frustum_negative_occupancy_loss(
            logits,
            target,
            occupancy_activation="sigmoid",
            frustum_mask=mask,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[:, :, 1].abs().mean().item(), 0.9)
        self.assertEqual(logits.grad[:, :, 0].abs().sum().item(), 0.0)

    def test_ir_frustum_negative_loss_without_selected_negatives_is_graph_zero(self):
        decoded = torch.randn(1, 1, 2, 1, 1, requires_grad=True)
        target = torch.ones_like(decoded)
        occupied_mask = torch.ones_like(decoded, dtype=torch.bool)

        no_mask_loss = decoded_ir_frustum_negative_occupancy_loss(
            decoded,
            target,
            occupancy_activation="raw",
            frustum_mask=None,
        )
        no_negative_loss = decoded_ir_frustum_negative_occupancy_loss(
            decoded,
            target,
            occupancy_activation="raw",
            frustum_mask=occupied_mask,
        )
        (no_mask_loss + no_negative_loss).backward()

        self.assertTrue(torch.isfinite(no_mask_loss))
        self.assertTrue(torch.isfinite(no_negative_loss))
        self.assertEqual(no_mask_loss.item(), 0.0)
        self.assertEqual(no_negative_loss.item(), 0.0)
        self.assertIsNotNone(decoded.grad)
        self.assertTrue(torch.isfinite(decoded.grad).all())

    def test_ir_frustum_negative_loss_excludes_soft_targets_from_negative_class(self):
        decoded = torch.randn(1, 1, 2, 1, 1, requires_grad=True)
        target = torch.full_like(decoded, 0.5)
        soft_target_mask = torch.ones_like(decoded, dtype=torch.bool)

        loss = decoded_ir_frustum_negative_occupancy_loss(
            decoded,
            target,
            occupancy_activation="raw",
            frustum_mask=soft_target_mask,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.0)
        self.assertIsNotNone(decoded.grad)
        self.assertTrue(torch.isfinite(decoded.grad).all())
        self.assertEqual(decoded.grad.abs().sum().item(), 0.0)

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
    def test_density_loss_penalizes_overdense_prediction_more_than_matched_prediction(self):
        target = make_volume(z_size=2)
        target[:, 0, 0, 0, 0] = 1.0
        matched = target.clone()
        overdense = torch.ones_like(target)

        matched_loss = decoded_density_precision_loss(
            matched,
            target,
            occupancy_activation="raw",
        )
        overdense_loss = decoded_density_precision_loss(
            overdense,
            target,
            occupancy_activation="raw",
        )

        self.assertTrue(torch.isfinite(matched_loss))
        self.assertTrue(torch.isfinite(overdense_loss))
        self.assertGreater(overdense_loss.item(), matched_loss.item())

    def test_density_loss_penalizes_empty_columns_more_than_occupied_columns(self):
        target = make_volume(z_size=3)
        target[:, 0, 1, 0, 0] = 1.0
        same_column_extra = target.clone()
        same_column_extra[:, 0, 2, 0, 0] = 1.0
        empty_column_extra = target.clone()
        empty_column_extra[:, 0, 2, 1, 1] = 1.0

        same_column_loss = decoded_density_precision_loss(
            same_column_extra,
            target,
            occupancy_activation="raw",
        )
        empty_column_loss = decoded_density_precision_loss(
            empty_column_extra,
            target,
            occupancy_activation="raw",
        )

        self.assertGreater(empty_column_loss.item(), same_column_loss.item())

    def test_density_loss_does_not_penalize_extra_z_inside_occupied_column(self):
        target = make_volume(z_size=3)
        target[:, 0, 1, 0, 0] = 1.0
        same_column_extra = target.clone()
        same_column_extra[:, 0, 0, 0, 0] = 1.0
        same_column_extra[:, 0, 2, 0, 0] = 1.0

        matched_loss = decoded_density_precision_loss(
            target,
            target,
            occupancy_activation="raw",
        )
        same_column_loss = decoded_density_precision_loss(
            same_column_extra,
            target,
            occupancy_activation="raw",
        )

        self.assertEqual(matched_loss.item(), 0.0)
        self.assertEqual(same_column_loss.item(), 0.0)

    def test_density_loss_penalizes_empty_target_prediction_and_stays_finite(self):
        logits = torch.full(
            (1, 1, 2, 2, 2),
            4.0,
            dtype=torch.float16,
            requires_grad=True,
        )
        target = torch.zeros_like(logits)

        loss = decoded_density_precision_loss(
            logits,
            target,
            occupancy_activation="sigmoid",
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_density_loss_sigmoid_high_false_positive_keeps_strong_gradient(self):
        logits = torch.full(
            (1, 1, 2, 1, 1),
            20.0,
            dtype=torch.float32,
            requires_grad=True,
        )
        target = torch.zeros_like(logits)

        loss = decoded_density_precision_loss(
            logits,
            target,
            occupancy_activation="sigmoid",
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(logits.grad.abs().mean().item(), 0.1)

    def test_density_loss_raw_empty_prediction_and_target_has_finite_gradient(self):
        decoded = torch.zeros(1, 1, 2, 2, 2, requires_grad=True)
        target = torch.zeros_like(decoded)

        loss = decoded_density_precision_loss(
            decoded,
            target,
            occupancy_activation="raw",
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.0)
        self.assertIsNotNone(decoded.grad)
        self.assertTrue(torch.isfinite(decoded.grad).all())

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
            decoded_top_height_weight=0.0,
            decoded_vertical_continuity_weight=0.0,
            decoded_density_weight=0.0,
            decoded_top_overshoot_weight=0.0,
            uncertainty_loss_weight=0.0,
            uncertainty=None,
        )

        self.assertEqual(vae.decode_calls, 0)
        self.assertAlmostEqual(loss.item(), 1.0)
        self.assertAlmostEqual(components["latent_loss"].item(), 1.0)
        for name in (
            "decoded_occupancy_loss",
            "height_distribution_loss",
            "top_height_loss",
            "top_overshoot_loss",
            "vertical_continuity_loss",
            "uncertainty_loss",
        ):
            self.assertEqual(components[name].item(), 0.0)
        self.assertEqual(components["ir_frustum_negative_loss"].item(), 0.0)

    def test_ir_frustum_negative_weight_adds_exact_component_after_single_decode(self):
        denoised = torch.zeros(1, 1, 1, 1, 1)
        z_target = torch.zeros_like(denoised)
        target = make_volume(z_size=2)
        decoded = make_volume(z_size=2)
        decoded[:, 0, 0, 0, 0] = 0.5
        mask = torch.zeros_like(target, dtype=torch.bool)
        mask[:, :, 0, 0, 0] = True
        vae = CountingVAE(decoded=decoded)
        negative_weight = 0.4

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
            decoded_top_height_weight=0.0,
            decoded_vertical_continuity_weight=0.0,
            decoded_density_weight=0.0,
            decoded_ir_frustum_negative_weight=negative_weight,
            ir_frustum_mask=mask,
        )

        self.assertEqual(vae.decode_calls, 1)
        self.assertAlmostEqual(components["ir_frustum_negative_loss"].item(), 0.25)
        self.assertTrue(
            torch.allclose(
                loss,
                negative_weight * components["ir_frustum_negative_loss"],
            )
        )

    def test_structure_weights_decode_once_and_add_independent_contributions(self):
        denoised = torch.zeros(1, 1, 1, 1, 1)
        z_target = torch.zeros_like(denoised)
        target = make_volume(z_size=3)
        target[:, 0, 2, 0, 0] = 1.0
        decoded = make_volume(z_size=3)
        decoded[:, 0, 0, 0, 0] = 1.0
        vae = CountingVAE(decoded=decoded)

        height_weight = 0.25
        top_weight = 0.5
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
            decoded_top_height_weight=top_weight,
            decoded_vertical_continuity_weight=continuity_weight,
            decoded_density_weight=0.0,
            uncertainty_loss_weight=0.0,
            uncertainty=None,
        )

        self.assertEqual(vae.decode_calls, 1)
        self.assertGreater(components["height_distribution_loss"].item(), 0.0)
        self.assertGreater(components["top_height_loss"].item(), 0.0)
        self.assertGreater(components["vertical_continuity_loss"].item(), 0.0)
        expected = (
            height_weight * components["height_distribution_loss"]
            + top_weight * components["top_height_loss"]
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
            decoded_top_height_weight=0.0,
            decoded_vertical_continuity_weight=0.0,
            decoded_density_weight=0.0,
            decoded_top_overshoot_weight=0.0,
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
        self.assertEqual(components["top_height_loss"].item(), 0.0)
        self.assertEqual(components["top_overshoot_loss"].item(), 0.0)
        self.assertEqual(components["vertical_continuity_loss"].item(), 0.0)

    def test_top_overshoot_weight_zero_and_nonzero_change_total_exactly(self):
        denoised = torch.zeros(1, 1, 1, 1, 1)
        z_target = torch.zeros_like(denoised)
        target = make_volume(z_size=3)
        target[:, 0, 1, 0, 0] = 1.0
        decoded = target.clone()
        decoded[:, 0, 2, 0, 0] = 1.0
        zero_weight_vae = CountingVAE(decoded=decoded)
        weighted_vae = CountingVAE(decoded=decoded)
        overshoot_weight = 0.4

        zero_weight_loss, zero_weight_components = compute_ldm_loss_components(
            denoised,
            z_target,
            target,
            vae=zero_weight_vae,
            occupancy_activation="raw",
            decoded_loss_weight=0.0,
            decoded_false_positive_weight=0.0,
            decoded_mass_weight=0.0,
            decoded_height_distribution_weight=0.0,
            decoded_top_height_weight=0.0,
            decoded_vertical_continuity_weight=0.0,
            decoded_density_weight=0.0,
            decoded_top_overshoot_weight=0.0,
            uncertainty_loss_weight=0.0,
            uncertainty=None,
        )
        weighted_loss, weighted_components = compute_ldm_loss_components(
            denoised,
            z_target,
            target,
            vae=weighted_vae,
            occupancy_activation="raw",
            decoded_loss_weight=0.0,
            decoded_false_positive_weight=0.0,
            decoded_mass_weight=0.0,
            decoded_height_distribution_weight=0.0,
            decoded_top_height_weight=0.0,
            decoded_vertical_continuity_weight=0.0,
            decoded_density_weight=0.0,
            decoded_top_overshoot_weight=overshoot_weight,
            uncertainty_loss_weight=0.0,
            uncertainty=None,
        )

        self.assertEqual(zero_weight_vae.decode_calls, 0)
        self.assertEqual(weighted_vae.decode_calls, 1)
        self.assertEqual(zero_weight_loss.item(), 0.0)
        self.assertEqual(zero_weight_components["top_overshoot_loss"].item(), 0.0)
        self.assertGreater(weighted_components["top_overshoot_loss"].item(), 0.0)
        self.assertTrue(
            torch.allclose(
                weighted_loss - zero_weight_loss,
                overshoot_weight * weighted_components["top_overshoot_loss"],
            )
        )

    def test_zero_density_weight_does_not_change_total_loss(self):
        denoised = torch.zeros(1, 1, 1, 1, 1)
        z_target = torch.ones_like(denoised)
        target = make_volume(z_size=2)
        decoded = torch.ones_like(target)
        vae = CountingVAE(decoded=decoded)

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
            decoded_top_height_weight=0.0,
            decoded_vertical_continuity_weight=0.0,
            decoded_density_weight=0.0,
            uncertainty_loss_weight=0.0,
            uncertainty=None,
        )

        self.assertEqual(vae.decode_calls, 0)
        self.assertAlmostEqual(loss.item(), 1.0)
        self.assertEqual(components["decoded_density_loss"].item(), 0.0)

    def test_density_weight_adds_weighted_component_after_single_decode(self):
        denoised = torch.zeros(1, 1, 1, 1, 1)
        z_target = torch.zeros_like(denoised)
        target = make_volume(z_size=2)
        target[:, 0, 0, 0, 0] = 1.0
        decoded = torch.ones_like(target)
        vae = CountingVAE(decoded=decoded)
        density_weight = 0.4

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
            decoded_top_height_weight=0.0,
            decoded_vertical_continuity_weight=0.0,
            decoded_density_weight=density_weight,
            uncertainty_loss_weight=0.0,
            uncertainty=None,
        )

        self.assertEqual(vae.decode_calls, 1)
        self.assertGreater(components["decoded_density_loss"].item(), 0.0)
        self.assertTrue(
            torch.allclose(
                loss,
                density_weight * components["decoded_density_loss"],
            )
        )

    def test_all_decoded_auxiliaries_share_one_decode_call(self):
        denoised = torch.zeros(1, 1, 1, 1, 1)
        z_target = torch.zeros_like(denoised)
        target = make_volume(z_size=3)
        target[:, 0, 2, 0, 0] = 1.0
        decoded = torch.ones_like(target)
        vae = CountingVAE(decoded=decoded)

        loss, components = compute_ldm_loss_components(
            denoised,
            z_target,
            target,
            vae=vae,
            occupancy_activation="raw",
            decoded_loss_weight=0.5,
            decoded_false_positive_weight=0.25,
            decoded_mass_weight=0.1,
            decoded_height_distribution_weight=0.2,
            decoded_top_height_weight=0.25,
            decoded_vertical_continuity_weight=0.3,
            decoded_density_weight=0.4,
            uncertainty_loss_weight=0.0,
            uncertainty=None,
        )

        self.assertEqual(vae.decode_calls, 1)
        self.assertTrue(torch.isfinite(loss))
        for name in (
            "decoded_occupancy_loss",
            "height_distribution_loss",
            "top_height_loss",
            "top_overshoot_loss",
            "vertical_continuity_loss",
            "decoded_density_loss",
        ):
            self.assertGreaterEqual(components[name].item(), 0.0)

    def test_component_contract_contains_all_ldm_loss_terms(self):
        expected = {
            "latent_loss",
            "decoded_occupancy_loss",
            "height_distribution_loss",
            "top_height_loss",
            "vertical_continuity_loss",
            "decoded_density_loss",
            "ir_frustum_occupancy_loss",
            "ir_frustum_negative_loss",
            "ir_frustum_top_height_loss",
            "uncertainty_loss",
        }

        self.assertTrue(expected.issubset(set(LDM_LOSS_COMPONENT_NAMES)))
        self.assertTrue(expected.issubset(set(LDM_METRICS_HEADER)))
        self.assertIn("top_overshoot_loss", LDM_LOSS_COMPONENT_NAMES)
        self.assertIn("top_overshoot_loss", LDM_METRICS_HEADER)
        self.assertIn("ir_frustum_negative_loss", LDM_METRICS_HEADER)
        self.assertIn("column_positive_loss", LDM_LOSS_COMPONENT_NAMES)
        self.assertIn("column_negative_loss", LDM_LOSS_COMPONENT_NAMES)
        self.assertIn("column_positive_loss", LDM_METRICS_HEADER)
        self.assertIn("column_negative_loss", LDM_METRICS_HEADER)
        self.assertTrue(
            {"mock_ir_ratio", "mock_calib_ratio", "ir_frustum_voxel_ratio"}.issubset(
                set(LDM_METRICS_HEADER)
            )
        )

    def test_zero_column_weights_preserve_latent_loss_without_decode(self):
        denoised = torch.tensor([[[[[2.0]]]]])
        z_target = torch.tensor([[[[[1.0]]]]])
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
            decoded_top_height_weight=0.0,
            decoded_vertical_continuity_weight=0.0,
            decoded_density_weight=0.0,
            decoded_column_positive_weight=0.0,
            decoded_column_negative_weight=0.0,
            decoded_column_temperature=1.0,
        )

        self.assertEqual(vae.decode_calls, 0)
        self.assertEqual(loss.item(), 1.0)
        self.assertEqual(components["column_positive_loss"].item(), 0.0)
        self.assertEqual(components["column_negative_loss"].item(), 0.0)

    def test_column_weights_add_raw_components_to_total_exactly(self):
        denoised = torch.zeros(1, 1, 1, 1, 1)
        z_target = torch.zeros_like(denoised)
        target = make_volume(z_size=2)
        target[:, 0, 0, 0, 0] = 1.0
        decoded = torch.full_like(target, -1.0)
        decoded[:, 0, 0, 0, 0] = 2.0
        vae = CountingVAE(decoded=decoded)
        positive_weight = 0.3
        negative_weight = 0.7
        temperature = 0.5

        loss, components = compute_ldm_loss_components(
            denoised,
            z_target,
            target,
            vae=vae,
            occupancy_activation="sigmoid",
            decoded_loss_weight=0.0,
            decoded_false_positive_weight=0.0,
            decoded_mass_weight=0.0,
            decoded_height_distribution_weight=0.0,
            decoded_top_height_weight=0.0,
            decoded_vertical_continuity_weight=0.0,
            decoded_density_weight=0.0,
            decoded_column_positive_weight=positive_weight,
            decoded_column_negative_weight=negative_weight,
            decoded_column_temperature=temperature,
        )
        expected_components = decoded_column_balanced_losses(
            decoded, target, "sigmoid", temperature=temperature
        )

        self.assertEqual(vae.decode_calls, 1)
        self.assertTrue(torch.equal(components["column_positive_loss"], expected_components["positive_loss"]))
        self.assertTrue(torch.equal(components["column_negative_loss"], expected_components["negative_loss"]))
        self.assertTrue(
            torch.allclose(
                loss,
                positive_weight * expected_components["positive_loss"]
                + negative_weight * expected_components["negative_loss"],
            )
        )

    def test_column_and_existing_decoded_losses_share_one_decode(self):
        denoised = torch.zeros(1, 1, 1, 1, 1)
        target = make_volume(z_size=2)
        target[:, 0, 0, 0, 0] = 1.0
        vae = CountingVAE(decoded=torch.ones_like(target))

        compute_ldm_loss_components(
            denoised,
            torch.zeros_like(denoised),
            target,
            vae=vae,
            occupancy_activation="raw",
            decoded_loss_weight=0.2,
            decoded_false_positive_weight=0.0,
            decoded_mass_weight=0.0,
            decoded_height_distribution_weight=0.0,
            decoded_top_height_weight=0.0,
            decoded_vertical_continuity_weight=0.0,
            decoded_density_weight=0.0,
            decoded_column_positive_weight=0.1,
            decoded_column_negative_weight=0.1,
        )

        self.assertEqual(vae.decode_calls, 1)

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
                decoded_top_height_weight=0.0,
                decoded_vertical_continuity_weight=0.0,
                decoded_density_weight=0.0,
                uncertainty_loss_weight=0.1,
                uncertainty={},
            )


class LDMTrainerUtilityTest(unittest.TestCase):
    def _make_trainer(self, temp_dir, ldm_overrides=None):
        ldm_config = {
            "save_dir": temp_dir,
            "fusion_voxel_shape": [1, 1, 1],
            "fusion_latent_shape": [1, 1, 1],
            "channel_mult": [1],
        }
        ldm_config.update(ldm_overrides or {})
        config = mock.Mock()
        config.get.side_effect = lambda key, default=None: ldm_config if key == "ldm" else default
        memory_opt = mock.Mock(device=torch.device("cpu"), use_amp=False)

        class TinyModel(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

        with mock.patch(
            "diffusion_consistency_radar.scripts.unified_train.OptimizedUNetModel",
            TinyModel,
        ), mock.patch(
            "diffusion_consistency_radar.scripts.unified_train.CompleteDualModalityPerceptionNet",
            TinyModel,
        ):
            return OptimizedLDMTrainer(TrainerVAE(), config, memory_opt)

    def test_column_config_defaults_and_checkpoint_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._make_trainer(temp_dir)
            self.assertEqual(trainer.decoded_column_positive_weight, 0.0)
            self.assertEqual(trainer.decoded_column_negative_weight, 0.0)
            self.assertEqual(trainer.decoded_column_temperature, 1.0)

            metadata = trainer._ldm_loss_config()
            self.assertEqual(metadata["decoded_column_positive_weight"], 0.0)
            self.assertEqual(metadata["decoded_column_negative_weight"], 0.0)
            self.assertEqual(metadata["decoded_column_temperature"], 1.0)

    def test_column_config_reads_explicit_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = self._make_trainer(
                temp_dir,
                {
                    "decoded_column_positive_weight": 0.2,
                    "decoded_column_negative_weight": 0.4,
                    "decoded_column_temperature": 2.5,
                },
            )

            self.assertEqual(trainer.decoded_column_positive_weight, 0.2)
            self.assertEqual(trainer.decoded_column_negative_weight, 0.4)
            self.assertEqual(trainer.decoded_column_temperature, 2.5)
            self.assertEqual(
                trainer._ldm_loss_config()["decoded_column_temperature"], 2.5
            )

    def test_column_config_rejects_invalid_weights_and_temperature(self):
        invalid_configs = (
            ({"decoded_column_positive_weight": -0.1}, "positive"),
            ({"decoded_column_negative_weight": float("nan")}, "negative"),
            ({"decoded_column_temperature": 0.0}, "temperature"),
            ({"decoded_column_temperature": float("nan")}, "temperature"),
        )
        for overrides, message in invalid_configs:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as temp_dir:
                with self.assertRaisesRegex(ValueError, message):
                    self._make_trainer(temp_dir, overrides)

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
            written_header = csv_path.read_text(encoding="utf-8").strip().split(",")
            self.assertIn("top_overshoot_loss", written_header)
            self.assertIn("ir_frustum_negative_loss", written_header)
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
