#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 VAE 重建上界诊断脚本中的 occupancy 指标计算。
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffusion_consistency_radar.scripts.diagnose_vae_reconstruction import (
    build_best_threshold_report_lines,
    build_vae_from_diagnostic_checkpoint,
    compute_binary_occ_metrics,
    evaluate_reconstruction_threshold,
    parse_csv_floats,
    resolve_occupancy_activation,
    resolve_diagnostic_grid_config,
    summarize_threshold_rows,
)


class VAEReconstructionDiagnosticTest(unittest.TestCase):
    def test_binary_occ_metrics_match_expected_overlap(self):
        gt = np.array([[[True, True, False]]])
        pred = np.array([[[True, False, True]]])

        metrics = compute_binary_occ_metrics(gt, pred)

        self.assertEqual(metrics["intersection"], 1)
        self.assertEqual(metrics["union"], 3)
        self.assertEqual(metrics["gt_occ"], 2)
        self.assertEqual(metrics["recon_occ"], 2)
        self.assertAlmostEqual(metrics["iou"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)

    def test_summary_accumulates_micro_metrics(self):
        rows = [
            {"intersection": 1, "union": 3, "gt_occ": 2, "recon_occ": 2},
            {"intersection": 2, "union": 4, "gt_occ": 4, "recon_occ": 2},
        ]

        summary = summarize_threshold_rows(rows)

        self.assertEqual(summary["frames"], 2)
        self.assertEqual(summary["intersection"], 3)
        self.assertEqual(summary["union"], 7)
        self.assertAlmostEqual(summary["iou"], 3.0 / 7.0)
        self.assertAlmostEqual(summary["recall"], 3.0 / 6.0)
        self.assertAlmostEqual(summary["precision"], 3.0 / 4.0)

    def test_summary_micro_aggregates_vertical_structure_metrics(self):
        rows = [
            {
                "intersection": 1,
                "union": 3,
                "gt_occ": 2,
                "recon_occ": 2,
                "height_coverage_numerator": 3.0,
                "height_coverage_denominator": 4.0,
                "top_height_numerator": 1.0,
                "top_height_denominator": 2.0,
                "vertical_connectivity_numerator": 2.0,
                "vertical_connectivity_denominator": 5.0,
                "trunk_region_numerator": 4.0,
                "trunk_region_denominator": 8.0,
            },
            {
                "intersection": 2,
                "union": 4,
                "gt_occ": 4,
                "recon_occ": 2,
                "height_coverage_numerator": 5.0,
                "height_coverage_denominator": 6.0,
                "top_height_numerator": 2.0,
                "top_height_denominator": 3.0,
                "vertical_connectivity_numerator": 4.0,
                "vertical_connectivity_denominator": 7.0,
                "trunk_region_numerator": 3.0,
                "trunk_region_denominator": 4.0,
            },
        ]

        summary = summarize_threshold_rows(rows)

        self.assertEqual(summary["height_coverage_numerator"], 8.0)
        self.assertEqual(summary["height_coverage_denominator"], 10.0)
        self.assertAlmostEqual(summary["height_coverage_recall"], 0.8)
        self.assertEqual(summary["top_height_numerator"], 3.0)
        self.assertEqual(summary["top_height_denominator"], 5.0)
        self.assertAlmostEqual(summary["top_height_recall"], 0.6)
        self.assertEqual(summary["vertical_connectivity_numerator"], 6.0)
        self.assertEqual(summary["vertical_connectivity_denominator"], 12.0)
        self.assertAlmostEqual(summary["vertical_connectivity_recall"], 0.5)
        self.assertEqual(summary["trunk_region_numerator"], 7.0)
        self.assertEqual(summary["trunk_region_denominator"], 12.0)
        self.assertAlmostEqual(summary["trunk_region_recall"], 7.0 / 12.0)

    def test_summary_keeps_legacy_binary_rows_compatible(self):
        rows = [
            {"intersection": 1, "union": 3, "gt_occ": 2, "recon_occ": 2},
            {"intersection": 0, "union": 1, "gt_occ": 1, "recon_occ": 0},
        ]

        summary = summarize_threshold_rows(rows)

        self.assertEqual(summary["height_coverage_numerator"], 0.0)
        self.assertEqual(summary["height_coverage_denominator"], 0.0)
        self.assertEqual(summary["height_coverage_recall"], 0.0)
        self.assertEqual(summary["trunk_region_numerator"], 0.0)
        self.assertEqual(summary["trunk_region_denominator"], 0.0)
        self.assertEqual(summary["trunk_region_recall"], 0.0)

    def test_summary_rejects_incomplete_structure_rows(self):
        complete_structure_counts = {
            "height_coverage_numerator": 1.0,
            "height_coverage_denominator": 2.0,
            "top_height_numerator": 1.0,
            "top_height_denominator": 1.0,
            "vertical_connectivity_numerator": 1.0,
            "vertical_connectivity_denominator": 2.0,
            "trunk_region_numerator": 1.0,
            "trunk_region_denominator": 2.0,
        }
        rows = [
            {
                "intersection": 1,
                "union": 2,
                "gt_occ": 2,
                "recon_occ": 1,
                **complete_structure_counts,
            },
            {
                "intersection": 1,
                "union": 2,
                "gt_occ": 2,
                "recon_occ": 1,
                **{
                    key: value
                    for key, value in complete_structure_counts.items()
                    if key != "trunk_region_denominator"
                },
            },
        ]

        with self.assertRaisesRegex(ValueError, "trunk_region_denominator"):
            summarize_threshold_rows(rows)

    def test_threshold_evaluation_recomputes_vertical_structure(self):
        target = np.zeros((4, 2, 2), dtype=np.float32)
        target[:, 0, 0] = 1.0
        reconstruction = np.zeros_like(target)
        reconstruction[:, 0, 0] = np.array([0.9, 0.7, 0.7, 0.7], dtype=np.float32)
        kwargs = {
            "pc_range": (0.0, 0.0, 0.0, 1.0, 1.0, 4.0),
            "top_height_tolerance_m": 0.0,
            "trunk_base_max_z": 1.0,
            "trunk_min_height_m": 2.0,
            "trunk_height_cap_m": 3.0,
        }

        metrics_at_05 = evaluate_reconstruction_threshold(
            reconstruction, target, threshold=0.5, **kwargs
        )
        metrics_at_08 = evaluate_reconstruction_threshold(
            reconstruction, target, threshold=0.8, **kwargs
        )

        for metric_name in (
            "height_coverage_recall",
            "top_height_recall",
            "vertical_connectivity_recall",
            "trunk_region_recall",
        ):
            self.assertGreater(metrics_at_05[metric_name], metrics_at_08[metric_name])

    def test_report_lines_include_best_structure_metrics(self):
        best = {
            "threshold": 0.35,
            "iou": 0.42,
            "recall": 0.56,
            "precision": 0.70,
            "height_coverage_recall": 0.81,
            "top_height_recall": 0.62,
            "vertical_connectivity_recall": 0.58,
            "trunk_region_recall": 0.77,
        }

        lines = build_best_threshold_report_lines(best)
        report_text = "\n".join(lines)

        self.assertIn("best threshold by IoU: 0.350", report_text)
        self.assertIn("best IoU / Recall / Precision: 0.4200 / 0.5600 / 0.7000", report_text)
        self.assertIn("best structure recall", report_text)
        self.assertIn("height_coverage=0.8100", report_text)
        self.assertIn("top_height=0.6200", report_text)
        self.assertIn("vertical_connectivity=0.5800", report_text)
        self.assertIn("trunk_region=0.7700", report_text)

    def test_parse_csv_floats_validates_fixed_length(self):
        self.assertEqual(parse_csv_floats("0,-20,-6,40,20,10", 6), [0.0, -20.0, -6.0, 40.0, 20.0, 10.0])
        with self.assertRaises(ValueError):
            parse_csv_floats("0,1", 3)

    def test_checkpoint_metadata_selects_sigmoid_occupancy(self):
        checkpoint = {"occupancy_activation": "sigmoid", "model_state_dict": {}}

        self.assertEqual(resolve_occupancy_activation(checkpoint), "sigmoid")

    def test_legacy_checkpoint_defaults_to_raw_occupancy(self):
        checkpoint = {"model_state_dict": {}}

        self.assertEqual(resolve_occupancy_activation(checkpoint), "raw")

    def test_diagnostic_uses_checkpoint_vae_config_before_fallback(self):
        from diffusion_consistency_radar.cm.vae_3d import (
            VAE3D,
            create_lightweight_vae_config,
        )

        config = create_lightweight_vae_config()
        config["latent_dim"] = 8
        config["base_channels"] = 32
        checkpoint = {
            "model_state_dict": VAE3D(**config).state_dict(),
            "vae_config": config,
            "vae_config_type": "lightweight",
            "occupancy_activation": "sigmoid",
        }

        model, metadata = build_vae_from_diagnostic_checkpoint(
            checkpoint,
            fallback_config_type="ultra_lightweight",
        )

        self.assertEqual(model.latent_dim, 8)
        self.assertEqual(metadata["occupancy_activation"], "sigmoid")

    def test_diagnostic_legacy_checkpoint_requires_explicit_fallback(self):
        from diffusion_consistency_radar.cm.vae_3d import (
            VAE3D,
            create_ultra_lightweight_vae_config,
        )

        config = create_ultra_lightweight_vae_config()
        checkpoint = {"model_state_dict": VAE3D(**config).state_dict()}

        with self.assertRaisesRegex(ValueError, "fallback"):
            build_vae_from_diagnostic_checkpoint(checkpoint, fallback_config_type=None)

    def test_diagnostic_grid_uses_checkpoint_metadata_without_cli_override(self):
        metadata = {
            "data_grid_config": {
                "target_size": [16, 64, 80],
                "source_pc_range": [0, -30, -5, 100, 30, 15],
                "model_pc_range": [0, -10, -3, 40, 10, 9],
            }
        }

        target_size, source_range, model_range = resolve_diagnostic_grid_config(
            metadata, None, None, None
        )

        self.assertEqual(target_size, (16, 64, 80))
        self.assertEqual(source_range[-1], 15.0)
        self.assertEqual(model_range[3], 40.0)


if __name__ == "__main__":
    unittest.main()
