#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试归一化工厂为轻量级 VAE 通道数选择合法的 GroupNorm 组数。
"""

import os
import sys
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffusion_consistency_radar.cm.nn import AdaptiveNorm3D, GroupNorm32, normalization
from diffusion_consistency_radar.cm.unet import ResBlock
from diffusion_consistency_radar.cm.vae_3d import VAE3D, create_lightweight_vae_config


class NormalizationGroupSelectionTest(unittest.TestCase):
    def test_lightweight_channels_construct_and_forward(self):
        for channels, expected_groups in ((24, 24), (72, 24)):
            with self.subTest(channels=channels):
                norm = normalization(channels)
                output = norm(torch.randn(2, channels, 2, 2, 2))

                self.assertEqual(norm.num_groups, expected_groups)
                self.assertEqual(tuple(output.shape), (2, channels, 2, 2, 2))

    def test_divisible_channels_keep_requested_32_groups(self):
        norm = normalization(96)

        self.assertEqual(norm.num_groups, 32)

    def test_group_default_fallback_and_adaptive_share_legal_group_count(self):
        group = normalization(72, norm_type="group")
        fallback = normalization(72, norm_type="unknown")
        adaptive = normalization(72, norm_type="adaptive")

        self.assertIsInstance(group, GroupNorm32)
        self.assertEqual(group.num_groups, 24)
        self.assertIsInstance(fallback, GroupNorm32)
        self.assertEqual(fallback.num_groups, 24)
        self.assertIsInstance(adaptive, AdaptiveNorm3D)
        self.assertEqual(adaptive.group_norm.num_groups, 24)

    def test_explicit_group_limit_selects_largest_divisor(self):
        norm = normalization(24, num_groups=10)

        self.assertEqual(norm.num_groups, 8)

    def test_none_uses_default_and_explicit_one_is_preserved(self):
        default_norm = normalization(72, num_groups=None)
        one_group_norm = normalization(72, num_groups=1)

        self.assertEqual(default_norm.num_groups, 24)
        self.assertEqual(one_group_norm.num_groups, 1)

    def test_non_positive_num_groups_raise_value_error(self):
        for num_groups in (0, -1):
            with self.subTest(num_groups=num_groups):
                with self.assertRaisesRegex(ValueError, "num_groups.*正整数"):
                    normalization(24, num_groups=num_groups)

    def test_wrong_type_num_groups_raise_type_error(self):
        for num_groups in (True, False, 1.5, "8"):
            with self.subTest(num_groups=num_groups):
                with self.assertRaisesRegex(TypeError, "num_groups.*正整数"):
                    normalization(24, num_groups=num_groups)

    def test_non_positive_channels_raise_clear_error(self):
        for channels in (0, -1):
            with self.subTest(channels=channels):
                with self.assertRaisesRegex(ValueError, "channels.*正整数"):
                    normalization(channels)

    def test_shared_unet_resblock_forwards_with_lightweight_channels(self):
        block = ResBlock(
            channels=24,
            emb_channels=16,
            dropout=0.0,
            out_channels=72,
            dims=3,
            use_checkpoint=False,
        )

        output = block(torch.randn(2, 24, 2, 2, 2), torch.randn(2, 16))

        self.assertEqual(tuple(output.shape), (2, 72, 2, 2, 2))

    def test_lightweight_vae_forwards_with_latent_dim_eight_override(self):
        config = create_lightweight_vae_config()
        config["latent_dim"] = 8
        model = VAE3D(**config)
        model.eval()
        input_tensor = torch.randn(1, config["in_channels"], 2, 4, 4)

        with torch.no_grad():
            reconstruction, (mean, logvar) = model(
                input_tensor,
                sample_posterior=False,
            )

        self.assertEqual(model.latent_dim, 8)
        self.assertEqual(model.encoder.conv_in.out_channels, 24)
        self.assertEqual(tuple(reconstruction.shape), tuple(input_tensor.shape))
        self.assertEqual(mean.shape[1], 8)
        self.assertEqual(logvar.shape[1], 8)
        self.assertTrue(torch.isfinite(reconstruction).all().item())
        self.assertTrue(torch.isfinite(mean).all().item())
        self.assertTrue(torch.isfinite(logvar).all().item())


if __name__ == "__main__":
    unittest.main()
