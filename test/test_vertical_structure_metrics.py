# -*- coding: utf-8 -*-
"""测试垂直结构评估指标。"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def make_column(shape, x, y, z_indices):
    """构造单个竖向占用列。"""
    volume = np.zeros(shape, dtype=np.float32)
    for z in z_indices:
        volume[z, x, y] = 1.0
    return volume


class VerticalStructureMetricsTest(unittest.TestCase):
    def test_perfect_overlap_returns_full_recall(self):
        from diffusion_consistency_radar.cm.evaluation_metrics import vertical_structure_metrics

        target = make_column((6, 2, 2), 0, 0, [0, 1, 2, 3])
        pred = target.copy()

        metrics = vertical_structure_metrics(
            pred,
            target,
            pc_range=(0.0, 0.0, 0.0, 2.0, 2.0, 6.0),
            top_height_tolerance_m=0.0,
            trunk_base_max_z=0.5,
            trunk_min_height_m=3.0,
            trunk_height_cap_m=2.0,
        )

        self.assertEqual(metrics["height_coverage_recall"], 1.0)
        self.assertEqual(metrics["top_height_recall"], 1.0)
        self.assertEqual(metrics["vertical_connectivity_recall"], 1.0)
        self.assertEqual(metrics["trunk_region_recall"], 1.0)

    def test_missing_top_voxel_reduces_top_and_height_recall(self):
        from diffusion_consistency_radar.cm.evaluation_metrics import vertical_structure_metrics

        target = make_column((6, 2, 2), 0, 0, [0, 1, 2, 3])
        pred = make_column((6, 2, 2), 0, 0, [0, 1, 2])

        metrics = vertical_structure_metrics(
            pred,
            target,
            pc_range=(0.0, 0.0, 0.0, 2.0, 2.0, 6.0),
            top_height_tolerance_m=0.0,
        )

        self.assertEqual(metrics["height_coverage_numerator"], 3.0)
        self.assertEqual(metrics["height_coverage_denominator"], 4.0)
        self.assertEqual(metrics["height_coverage_recall"], 0.75)
        self.assertEqual(metrics["top_height_recall"], 0.0)

    def test_vertical_break_reduces_connectivity_recall(self):
        from diffusion_consistency_radar.cm.evaluation_metrics import vertical_structure_metrics

        target = make_column((6, 2, 2), 0, 0, [0, 1, 2, 3])
        pred = make_column((6, 2, 2), 0, 0, [0, 2, 3])

        metrics = vertical_structure_metrics(
            pred,
            target,
            pc_range=(0.0, 0.0, 0.0, 2.0, 2.0, 6.0),
        )

        self.assertEqual(metrics["height_coverage_numerator"], 3.0)
        self.assertEqual(metrics["height_coverage_denominator"], 4.0)
        self.assertEqual(metrics["height_coverage_recall"], 0.75)
        self.assertEqual(metrics["vertical_connectivity_numerator"], 2.0)
        self.assertEqual(metrics["vertical_connectivity_denominator"], 4.0)
        self.assertEqual(metrics["vertical_connectivity_recall"], 0.5)

    def test_shifted_equal_length_column_does_not_get_false_high_overlap_scores(self):
        from diffusion_consistency_radar.cm.evaluation_metrics import vertical_structure_metrics

        target = make_column((6, 2, 2), 0, 0, [0, 1, 2, 3])
        pred = make_column((6, 2, 2), 0, 0, [2, 3, 4, 5])

        metrics = vertical_structure_metrics(
            pred,
            target,
            pc_range=(0.0, 0.0, 0.0, 2.0, 2.0, 6.0),
        )

        self.assertEqual(metrics["height_coverage_numerator"], 2.0)
        self.assertEqual(metrics["height_coverage_denominator"], 4.0)
        self.assertEqual(metrics["height_coverage_recall"], 0.5)
        self.assertEqual(metrics["vertical_connectivity_numerator"], 2.0)
        self.assertEqual(metrics["vertical_connectivity_denominator"], 4.0)
        self.assertEqual(metrics["vertical_connectivity_recall"], 0.5)

    def test_top_height_recall_rejects_prediction_one_voxel_above_target_even_with_one_voxel_tolerance(self):
        from diffusion_consistency_radar.cm.evaluation_metrics import vertical_structure_metrics

        target = make_column((6, 2, 2), 0, 0, [0, 1, 2, 3])
        pred = make_column((6, 2, 2), 0, 0, [0, 1, 2, 4])

        metrics = vertical_structure_metrics(
            pred,
            target,
            pc_range=(0.0, 0.0, 0.0, 2.0, 2.0, 6.0),
            top_height_tolerance_m=1.0,
        )

        self.assertEqual(metrics["top_height_numerator"], 0.0)
        self.assertEqual(metrics["top_height_denominator"], 1.0)
        self.assertEqual(metrics["top_height_recall"], 0.0)

    def test_top_height_recall_requires_prediction_not_to_overshoot_target(self):
        from diffusion_consistency_radar.cm.evaluation_metrics import vertical_structure_metrics

        target = make_column((6, 2, 2), 0, 0, [0, 1, 2, 3])
        pred = make_column((6, 2, 2), 0, 0, [0, 1, 2, 3, 4, 5])

        metrics = vertical_structure_metrics(
            pred,
            target,
            pc_range=(0.0, 0.0, 0.0, 2.0, 2.0, 6.0),
            top_height_tolerance_m=1.0,
        )

        self.assertEqual(metrics["top_height_numerator"], 0.0)
        self.assertEqual(metrics["top_height_denominator"], 1.0)
        self.assertEqual(metrics["top_height_recall"], 0.0)

    def test_subvoxel_top_tolerance_does_not_expand_to_one_voxel(self):
        from diffusion_consistency_radar.cm.evaluation_metrics import vertical_structure_metrics

        target = make_column((6, 2, 2), 0, 0, [0, 1, 2, 3])
        pred = make_column((6, 2, 2), 0, 0, [0, 1, 2])

        metrics = vertical_structure_metrics(
            pred,
            target,
            pc_range=(0.0, 0.0, 0.0, 2.0, 2.0, 6.0),
            top_height_tolerance_m=0.99,
        )

        self.assertEqual(metrics["top_height_numerator"], 0.0)
        self.assertEqual(metrics["top_height_denominator"], 1.0)
        self.assertEqual(metrics["top_height_recall"], 0.0)

    def test_one_voxel_top_tolerance_allows_prediction_one_voxel_below_target(self):
        from diffusion_consistency_radar.cm.evaluation_metrics import vertical_structure_metrics

        target = make_column((6, 2, 2), 0, 0, [0, 1, 2, 3])
        pred = make_column((6, 2, 2), 0, 0, [0, 1, 2])

        metrics = vertical_structure_metrics(
            pred,
            target,
            pc_range=(0.0, 0.0, 0.0, 2.0, 2.0, 6.0),
            top_height_tolerance_m=1.0,
        )

        self.assertEqual(metrics["top_height_numerator"], 1.0)
        self.assertEqual(metrics["top_height_denominator"], 1.0)
        self.assertEqual(metrics["top_height_recall"], 1.0)

    def test_trunk_region_ignores_high_canopy_false_positive(self):
        from diffusion_consistency_radar.cm.evaluation_metrics import vertical_structure_metrics

        target = make_column((6, 2, 2), 0, 0, [0, 1, 2, 3])
        pred = make_column((6, 2, 2), 0, 0, [0, 1, 4, 5])

        metrics = vertical_structure_metrics(
            pred,
            target,
            pc_range=(0.0, 0.0, 0.0, 2.0, 2.0, 6.0),
            trunk_base_max_z=0.5,
            trunk_min_height_m=3.0,
            trunk_height_cap_m=2.0,
        )

        self.assertEqual(metrics["trunk_region_numerator"], 2.0)
        self.assertEqual(metrics["trunk_region_denominator"], 2.0)
        self.assertEqual(metrics["trunk_region_recall"], 1.0)

    def test_empty_target_returns_zero_without_nan(self):
        from diffusion_consistency_radar.cm.evaluation_metrics import vertical_structure_metrics

        pred = np.zeros((6, 2, 2), dtype=np.float32)
        target = np.zeros_like(pred)

        metrics = vertical_structure_metrics(pred, target, pc_range=(0.0, 0.0, 0.0, 2.0, 2.0, 6.0))

        for key, value in metrics.items():
            self.assertFalse(np.isnan(value), msg=key)
            self.assertEqual(value, 0.0, msg=key)


if __name__ == "__main__":
    unittest.main()
