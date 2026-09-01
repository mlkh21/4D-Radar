#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 LDM 仅在 persisted observed domain 内计算训练损失和验证指标。"""

import os
import sys
import unittest

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffusion_consistency_radar.scripts.unified_train import (
    compute_ldm_loss_components,
    decoded_column_balanced_losses,
    decoded_density_precision_loss,
    decoded_occupancy_auxiliary_loss,
    decoded_vertical_structure_losses,
    micro_occupancy_metrics,
)


class _NoDecodeVAE:
    """零 decoded 权重测试中禁止意外触发 VAE decode。"""

    def decode(self, _latent):
        raise AssertionError("零 decoded 权重不应调用 VAE.decode")


class LDMObservedSupervisionTest(unittest.TestCase):
    """unknown 体素不得被当作 free/negative 监督。"""

    def test_micro_metrics_ignore_predictions_outside_observed_domain(self):
        target = torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]])
        probability = torch.tensor([[[[[0.9, 0.1, 0.9, 0.9]]]]])
        observed = torch.tensor([[[[[1.0, 1.0, 0.0, 0.0]]]]])

        metrics = micro_occupancy_metrics(
            probability,
            target,
            threshold=0.5,
            observed_mask=observed,
        )

        self.assertEqual(metrics["intersection"], 1)
        self.assertEqual(metrics["union"], 1)
        self.assertEqual(metrics["target_positive"], 1)
        self.assertEqual(metrics["predicted_positive"], 1)

    def test_decoded_auxiliary_loss_is_invariant_to_unknown_prediction(self):
        target = torch.zeros(1, 4, 1, 1, 2)
        observed = torch.tensor([[[[[1.0, 0.0]]]]])
        baseline = torch.zeros_like(target)
        changed_unknown = baseline.clone()
        changed_unknown[:, 0, 0, 0, 1] = 100.0

        baseline_loss = decoded_occupancy_auxiliary_loss(
            baseline,
            target,
            occupancy_activation="raw",
            reconstruction_weight=1.0,
            false_positive_weight=1.0,
            mass_weight=1.0,
            observed_mask=observed,
        )
        changed_loss = decoded_occupancy_auxiliary_loss(
            changed_unknown,
            target,
            occupancy_activation="raw",
            reconstruction_weight=1.0,
            false_positive_weight=1.0,
            mass_weight=1.0,
            observed_mask=observed,
        )

        self.assertTrue(torch.equal(baseline_loss, changed_loss))

    def test_density_and_column_negative_ignore_unobserved_column(self):
        target = torch.zeros(1, 4, 2, 1, 2)
        observed = torch.zeros(1, 1, 2, 1, 2)
        observed[:, :, :, :, 0] = 1.0
        baseline = torch.full_like(target, -8.0)
        changed_unknown = baseline.clone()
        changed_unknown[:, 0, :, :, 1] = 8.0

        baseline_density = decoded_density_precision_loss(
            baseline,
            target,
            "sigmoid",
            observed_mask=observed,
        )
        changed_density = decoded_density_precision_loss(
            changed_unknown,
            target,
            "sigmoid",
            observed_mask=observed,
        )
        baseline_columns = decoded_column_balanced_losses(
            baseline,
            target,
            "sigmoid",
            observed_mask=observed,
        )
        changed_columns = decoded_column_balanced_losses(
            changed_unknown,
            target,
            "sigmoid",
            observed_mask=observed,
        )

        self.assertTrue(torch.equal(baseline_density, changed_density))
        self.assertTrue(
            torch.equal(
                baseline_columns["negative_loss"],
                changed_columns["negative_loss"],
            )
        )

    def test_vertical_losses_ignore_unobserved_height_cells(self):
        target = torch.zeros(1, 4, 4, 1, 1)
        target[:, 0, 0, 0, 0] = 1.0
        observed = torch.zeros(1, 1, 4, 1, 1)
        observed[:, :, :2] = 1.0
        baseline = target.clone()
        changed_unknown = baseline.clone()
        changed_unknown[:, 0, 2:, 0, 0] = 1.0

        baseline_losses = decoded_vertical_structure_losses(
            baseline,
            target,
            "raw",
            observed_mask=observed,
        )
        changed_losses = decoded_vertical_structure_losses(
            changed_unknown,
            target,
            "raw",
            observed_mask=observed,
        )

        for name in baseline_losses:
            with self.subTest(name=name):
                self.assertTrue(
                    torch.equal(baseline_losses[name], changed_losses[name])
                )

    def test_latent_loss_ignores_latent_block_without_observation(self):
        target = torch.zeros(1, 4, 1, 1, 4)
        observed = torch.tensor([[[[[1.0, 1.0, 0.0, 0.0]]]]])
        z_target = torch.zeros(1, 1, 1, 1, 2)
        denoised = z_target.clone()
        denoised[..., 1] = 10.0

        loss, components = compute_ldm_loss_components(
            denoised,
            z_target,
            target,
            vae=_NoDecodeVAE(),
            occupancy_activation="raw",
            decoded_loss_weight=0.0,
            decoded_false_positive_weight=0.0,
            decoded_mass_weight=0.0,
            decoded_height_distribution_weight=0.0,
            decoded_top_height_weight=0.0,
            decoded_vertical_continuity_weight=0.0,
            decoded_density_weight=0.0,
            observed_mask=observed,
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(components["latent_loss"].item(), 0.0)

    def test_explicit_empty_observed_domain_fails_closed(self):
        target = torch.zeros(1, 4, 1, 1, 2)
        z_target = torch.zeros(1, 1, 1, 1, 1)

        with self.assertRaisesRegex(ValueError, "observed"):
            compute_ldm_loss_components(
                z_target.clone(),
                z_target,
                target,
                vae=_NoDecodeVAE(),
                occupancy_activation="raw",
                decoded_loss_weight=0.0,
                decoded_false_positive_weight=0.0,
                decoded_mass_weight=0.0,
                decoded_height_distribution_weight=0.0,
                decoded_top_height_weight=0.0,
                decoded_vertical_continuity_weight=0.0,
                decoded_density_weight=0.0,
                observed_mask=torch.zeros(1, 1, 1, 1, 2),
            )


if __name__ == "__main__":
    unittest.main()
