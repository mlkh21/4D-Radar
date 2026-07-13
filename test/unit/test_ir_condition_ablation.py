#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试 IR 条件消融诊断脚本中的轻量工具函数。"""

import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
MODULE_DIR = os.path.join(ROOT, "test", "ablation")
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

import diagnose_ir_condition_ablation as ablation
from diagnose_ir_condition_ablation import compare_outputs, make_meta_variant


class IRConditionAblationUtilityTest(unittest.TestCase):
    def test_parse_variants_accepts_real_only_and_preserves_order(self):
        parser = getattr(ablation, "parse_variants", None)
        self.assertIsNotNone(parser, "需要支持仅运行 real IR 的 checkpoint 验证")

        self.assertEqual(parser("real"), ("real",))
        self.assertEqual(parser("mock,real"), ("mock", "real"))

    def test_parse_variants_rejects_unknown_or_duplicate_values(self):
        parser = getattr(ablation, "parse_variants", None)
        self.assertIsNotNone(parser, "需要实现 IR variant 参数校验")

        with self.assertRaisesRegex(ValueError, "未知"):
            parser("real,thermal")
        with self.assertRaisesRegex(ValueError, "重复"):
            parser("real,real")
        with self.assertRaisesRegex(ValueError, "real"):
            parser("zero,mock")

    def test_zero_variant_preserves_geometry_and_zeros_ir_image(self):
        meta = {
            "ir_img": torch.ones(3, 8, 8),
            "r_mat": torch.eye(3),
            "t_vec": torch.zeros(3),
            "k_mat": torch.eye(3),
            "is_mock_ir": torch.tensor(False),
            "is_mock_calib": torch.tensor(True),
        }

        zero_meta = make_meta_variant(meta, "zero", torch.device("cpu"))

        self.assertEqual(tuple(zero_meta["ir_img"].shape), (1, 3, 8, 8))
        self.assertEqual(float(zero_meta["ir_img"].sum()), 0.0)
        self.assertEqual(tuple(zero_meta["r_mat"].shape), (1, 3, 3))
        self.assertEqual(tuple(zero_meta["t_vec"].shape), (1, 3))

    def test_bool_mock_flags_are_preserved_as_batch_tensors(self):
        meta = {
            "ir_img": torch.ones(3, 8, 8),
            "r_mat": torch.eye(3),
            "t_vec": torch.zeros(3),
            "k_mat": torch.eye(3),
            "is_mock_ir": True,
            "is_mock_calib": True,
        }

        copied = make_meta_variant(meta, "real", torch.device("cpu"))

        self.assertEqual(float(copied["is_mock_ir"].item()), 1.0)
        self.assertEqual(float(copied["is_mock_calib"].item()), 1.0)

    def test_mock_variant_marks_mock_ir_and_keeps_nonzero_thermal_pattern(self):
        meta = {
            "ir_img": torch.ones(1, 3, 8, 8),
            "r_mat": torch.eye(3).unsqueeze(0),
            "t_vec": torch.zeros(1, 3),
            "k_mat": torch.eye(3).unsqueeze(0),
        }

        mock_meta = make_meta_variant(meta, "mock", torch.device("cpu"))

        self.assertEqual(tuple(mock_meta["ir_img"].shape), (1, 3, 8, 8))
        self.assertGreater(float(mock_meta["ir_img"].std()), 0.0)
        self.assertEqual(float(mock_meta["is_mock_ir"].item()), 1.0)

    def test_compare_outputs_reports_occupancy_difference_only(self):
        reference = torch.zeros(1, 4, 1, 1, 2)
        other = reference.clone()
        other[:, 0, 0, 0, 1] = 1.0
        other[:, 1, 0, 0, 0] = 100.0

        metrics = compare_outputs(reference, other)

        self.assertAlmostEqual(metrics["mean_abs_diff"], 0.5)
        self.assertAlmostEqual(metrics["max_abs_diff"], 1.0)

    def test_target_metrics_report_perfect_overlap(self):
        compute_metrics = getattr(ablation, "compute_target_metrics", None)
        self.assertIsNotNone(compute_metrics, "需要实现 target-aware IR 消融指标")
        target = np.zeros((1, 4, 2, 2), dtype=np.float32)
        target[0, 1:3, 0, 0] = 1.0

        metrics = compute_metrics(
            target,
            target,
            occ_threshold=0.5,
            target_threshold=0.5,
            pc_range=(0, -1, -1, 2, 1, 3),
        )

        self.assertAlmostEqual(metrics["voxel_precision"], 1.0)
        self.assertAlmostEqual(metrics["voxel_recall"], 1.0)
        self.assertAlmostEqual(metrics["voxel_iou"], 1.0)
        self.assertAlmostEqual(metrics["bev_iou"], 1.0)
        self.assertAlmostEqual(metrics["top_height_recall"], 1.0)

    def test_target_metrics_expose_density_and_top_overshoot(self):
        compute_metrics = getattr(ablation, "compute_target_metrics", None)
        self.assertIsNotNone(compute_metrics, "需要实现 target-aware IR 消融指标")
        target = np.zeros((1, 4, 1, 1), dtype=np.float32)
        target[0, 1, 0, 0] = 1.0
        prediction = target.copy()
        prediction[0, 2, 0, 0] = 1.0

        metrics = compute_metrics(
            prediction,
            target,
            occ_threshold=0.5,
            target_threshold=0.5,
            pc_range=(0, -1, -1, 1, 1, 3),
        )

        self.assertAlmostEqual(metrics["pred_to_target_ratio"], 2.0)
        self.assertAlmostEqual(metrics["voxel_precision"], 0.5)
        self.assertAlmostEqual(metrics["voxel_recall"], 1.0)
        self.assertAlmostEqual(metrics["top_height_recall"], 0.0)

    def test_target_metric_summary_uses_micro_aggregation(self):
        summarize = getattr(ablation, "summarize_target_rows", None)
        self.assertIsNotNone(summarize, "需要实现 target-aware IR 消融汇总")
        rows = [
            {
                "variant": "real",
                "pred_occ_count": 2.0,
                "target_occ_count": 1.0,
                "voxel_tp": 1.0,
                "voxel_fp": 1.0,
                "voxel_fn": 0.0,
                "bev_tp": 1.0,
                "bev_fp": 0.0,
                "bev_fn": 0.0,
                "height_coverage_numerator": 1.0,
                "height_coverage_denominator": 1.0,
                "top_height_numerator": 0.0,
                "top_height_denominator": 1.0,
                "vertical_connectivity_numerator": 1.0,
                "vertical_connectivity_denominator": 1.0,
                "trunk_region_numerator": 0.0,
                "trunk_region_denominator": 0.0,
            },
            {
                "variant": "real",
                "pred_occ_count": 1.0,
                "target_occ_count": 3.0,
                "voxel_tp": 1.0,
                "voxel_fp": 0.0,
                "voxel_fn": 2.0,
                "bev_tp": 1.0,
                "bev_fp": 0.0,
                "bev_fn": 2.0,
                "height_coverage_numerator": 1.0,
                "height_coverage_denominator": 3.0,
                "top_height_numerator": 1.0,
                "top_height_denominator": 3.0,
                "vertical_connectivity_numerator": 1.0,
                "vertical_connectivity_denominator": 3.0,
                "trunk_region_numerator": 1.0,
                "trunk_region_denominator": 2.0,
            },
        ]

        summary = summarize(rows)

        self.assertEqual(summary["frames"], 2)
        self.assertAlmostEqual(summary["voxel_precision"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["voxel_recall"], 0.5)
        self.assertAlmostEqual(summary["pred_to_target_ratio"], 0.75)
        self.assertAlmostEqual(summary["height_coverage_recall"], 0.5)

    def test_sample_selection_is_deterministic_and_spans_dataset(self):
        select_indices = getattr(ablation, "select_sample_indices", None)
        self.assertIsNotNone(select_indices, "需要实现多帧消融抽样")

        indices = select_indices(dataset_size=101, sample_index=7, max_samples=5)

        self.assertEqual(indices, [0, 25, 50, 75, 100])
        self.assertEqual(select_indices(101, sample_index=7, max_samples=1), [7])

    def test_v7_target_ablation_runner_uses_32_validation_frames(self):
        runner = os.path.join(ROOT, "test", "mini-test", "run_ldm_z64_v7_target_ablation.sh")
        self.assertTrue(os.path.exists(runner), "需要提供 v7 target-aware 消融一键脚本")
        with open(runner, "r", encoding="utf-8") as f:
            text = f.read()

        self.assertIn('ABLATION_MAX_SAMPLES="${ABLATION_MAX_SAMPLES:-32}"', text)
        self.assertIn("--split validation", text)
        self.assertIn("--max_samples \"${ABLATION_MAX_SAMPLES}\"", text)
        self.assertIn("--occ_threshold \"${OCC_THRESHOLD}\"", text)


if __name__ == "__main__":
    unittest.main()
