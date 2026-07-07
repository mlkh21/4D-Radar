#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证占用阈值扫描时 target 的物理裁剪与目标网格协议。"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion_consistency_radar.scripts.sweep_occ_threshold import (
    evaluate_task_thresholds,
    load_target_occ_resized,
    parse_thresholds,
    prepare_evaluation_files,
    resolve_target_path,
    select_evaluation_files,
    select_recommended_threshold,
    validate_range_bins,
)


class OccThresholdGridProtocolTest(unittest.TestCase):
    def test_target_is_cropped_to_model_range_before_resize(self):
        source_pc_range = (0.0, -20.0, -6.0, 120.0, 20.0, 10.0)
        model_pc_range = (0.0, -20.0, -6.0, 40.0, 20.0, 10.0)

        # 输入布局为 (X,Y,Z,C)。X 中心分别为 5,15,...,115m。
        voxel = np.zeros((12, 4, 4, 4), dtype=np.float32)
        voxel[1, 2, 1, 0] = 1.0  # 15m，位于模型范围内。
        voxel[8, 2, 1, 0] = 1.0  # 85m，位于模型范围外。

        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = os.path.join(temp_dir, "target.npy")
            np.save(target_path, voxel)

            occupancy = load_target_occ_resized(
                target_path,
                torch.device("cpu"),
                source_pc_range,
                model_pc_range,
                (4, 4, 4),
            )

        self.assertEqual(occupancy.shape, (4, 4, 4))
        self.assertEqual(int(np.count_nonzero(occupancy)), 1)
        self.assertEqual(float(occupancy[1, 1, 2]), 1.0)

    def test_sparse_npz_target_is_cropped_to_model_range_before_resize(self):
        source_pc_range = (0.0, -20.0, -6.0, 120.0, 20.0, 10.0)
        model_pc_range = (0.0, -20.0, -6.0, 40.0, 20.0, 10.0)

        # 稀疏协议与 load_sparse_voxel 一致：coords 为 XYZ，features 为通道向量。
        coords = np.asarray([[1, 2, 1], [8, 2, 1]], dtype=np.int64)
        features = np.zeros((2, 4), dtype=np.float32)
        features[:, 0] = 1.0

        with tempfile.TemporaryDirectory() as temp_dir:
            target_path = os.path.join(temp_dir, "target.npz")
            np.savez(
                target_path,
                shape=np.asarray([12, 4, 4, 4], dtype=np.int64),
                coords=coords,
                features=features,
            )
            occupancy = load_target_occ_resized(
                target_path,
                torch.device("cpu"),
                source_pc_range,
                model_pc_range,
                (4, 4, 4),
            )

        self.assertEqual(occupancy.shape, (4, 4, 4))
        self.assertEqual(int(np.count_nonzero(occupancy)), 1)
        self.assertEqual(float(occupancy[1, 1, 2]), 1.0)

    def test_validation_split_matches_training_randperm_protocol(self):
        files = [f"{index:06d}_voxel.npy" for index in range(10)]
        expected_indices = torch.randperm(
            10,
            generator=torch.Generator().manual_seed(42),
        ).tolist()[8:]

        first = select_evaluation_files(files, "validation", 0.8, 42)
        second = select_evaluation_files(files, "validation", 0.8, 42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(first, [files[index] for index in expected_indices])

    def test_validation_split_rejects_missing_prediction_frame(self):
        files = ["000000_voxel.npy", "000002_voxel.npy", "000003_voxel.npy"]

        with self.assertRaisesRegex(ValueError, "连续"):
            select_evaluation_files(files, "validation", 0.8, 42)

    def test_max_files_is_applied_after_validation_split(self):
        files = [f"{index:06d}_voxel.npy" for index in range(10)]
        full_validation = select_evaluation_files(files, "validation", 0.8, 42)

        selected = prepare_evaluation_files(
            files,
            evaluation_split="validation",
            train_split=0.8,
            split_seed=42,
            max_files=1,
        )

        self.assertEqual(selected, full_validation[:1])

    def test_max_files_limits_all_split_without_reordering(self):
        files = [f"{index:06d}_voxel.npy" for index in range(10)]

        selected = prepare_evaluation_files(
            files,
            evaluation_split="all",
            train_split=0.8,
            split_seed=42,
            max_files=3,
        )

        self.assertEqual(selected, files[:3])

    def test_thresholds_reject_nonfinite_and_out_of_range_values(self):
        for raw in ("nan", "inf", "-inf", "-0.01", "1.01", "0.2,nan"):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, r"\[0, 1\]|有限"):
                    parse_thresholds(raw)

    def test_task_metric_threshold_selection_uses_documented_tiebreakers(self):
        metrics = {
            0.3: {
                "task_bev_f1": 0.7,
                "task_bev_iou": 0.6,
                "pred_to_target_ratio": 1.2,
                "voxel_f1": 0.2,
            },
            0.4: {
                "task_bev_f1": 0.7,
                "task_bev_iou": 0.6,
                "pred_to_target_ratio": 1.0,
                "voxel_f1": 0.1,
            },
            0.6: {
                "task_bev_f1": 0.7,
                "task_bev_iou": 0.6,
                "pred_to_target_ratio": 1.0,
                "voxel_f1": 0.9,
            },
        }

        selected = select_recommended_threshold(metrics, "task_bev_f1")
        selected_by_default = select_recommended_threshold(metrics)

        # 0.4 与 0.6 前三项相同，最后选择更接近 0.5 的较小阈值。
        self.assertEqual(selected, 0.4)
        self.assertEqual(selected_by_default, selected)
        self.assertEqual(select_recommended_threshold(metrics, "voxel_f1"), 0.6)

    def test_task_metrics_cover_range_bands_and_do_not_follow_voxel_f1_only(self):
        pc_range = (0.0, -6.0, -2.0, 40.0, 6.0, 2.0)
        pred = np.zeros((4, 40, 12), dtype=np.float32)
        target = np.zeros_like(pred)

        # 两个高度不同但投影到同一 BEV 单元的点，严格 voxel 不重合。
        target[2, 10, 6] = 1.0
        pred[3, 10, 6] = 0.8
        # 20-40m 内再放置一个完全重合点。
        target[2, 30, 5] = 1.0
        pred[2, 30, 5] = 0.8

        result = evaluate_task_thresholds(
            pred,
            target,
            thresholds=(0.5, 0.9),
            target_threshold=0.1,
            pc_range=pc_range,
            z_min=-1.0,
            range_bins=(("x0_20", 0.0, 20.0), ("x20_40", 20.0, 40.0)),
            bev_cell_size=1.0,
        )

        good = result[0.5]
        self.assertEqual(good["task_bev_precision"], 1.0)
        self.assertEqual(good["task_bev_recall"], 1.0)
        self.assertEqual(good["task_bev_f1"], 1.0)
        self.assertEqual(good["task_bev_iou"], 1.0)
        self.assertEqual(good["task_match_ratio_2"], 1.0)
        self.assertEqual(good["bands"]["x0_20"]["task_bev_f1"], 1.0)
        self.assertEqual(good["bands"]["x20_40"]["task_bev_f1"], 1.0)
        self.assertLess(good["voxel_f1"], good["task_bev_f1"])
        self.assertEqual(
            select_recommended_threshold(result, "task_bev_f1"),
            0.5,
        )

    def test_empty_target_counts_nonempty_prediction_as_zero_nn_match(self):
        pc_range = (0.0, -6.0, -2.0, 40.0, 6.0, 2.0)
        pred = np.zeros((4, 40, 12), dtype=np.float32)
        target = np.zeros_like(pred)
        pred[2, 5, 6] = 1.0

        result = evaluate_task_thresholds(
            pred,
            target,
            thresholds=(0.5,),
            target_threshold=0.1,
            pc_range=pc_range,
            z_min=-1.0,
            range_bins=(("x0_20", 0.0, 20.0), ("x20_40", 20.0, 40.0)),
            bev_cell_size=1.0,
        )[0.5]

        self.assertEqual(result["task_match_ratio_2"], 0.0)
        self.assertEqual(result["matched_pred_count"], 0.0)
        self.assertEqual(result["match_query_count"], 1)
        self.assertEqual(result["bands"]["x0_20"]["task_match_ratio_2"], 0.0)

    def test_nn_match_ratio_is_weighted_by_prediction_point_count(self):
        pc_range = (0.0, -6.0, -2.0, 40.0, 6.0, 2.0)
        pred = np.zeros((4, 40, 12), dtype=np.float32)
        target = np.zeros_like(pred)

        # 近距离 band 为 1/1 匹配。
        pred[2, 5, 6] = 1.0
        target[2, 5, 6] = 1.0
        # 中距离 band 为 1/3 匹配，总体应为 (1+1)/(1+3)=0.5。
        pred[2, 25, 6] = 1.0
        pred[2, 30, 0] = 1.0
        pred[2, 35, 11] = 1.0
        target[2, 25, 6] = 1.0

        result = evaluate_task_thresholds(
            pred,
            target,
            thresholds=(0.5,),
            target_threshold=0.1,
            pc_range=pc_range,
            z_min=-1.0,
            range_bins=(("x0_20", 0.0, 20.0), ("x20_40", 20.0, 40.0)),
            bev_cell_size=1.0,
        )[0.5]

        self.assertAlmostEqual(result["task_match_ratio_2"], 0.5)
        self.assertAlmostEqual(result["matched_pred_count"], 2.0)
        self.assertEqual(result["match_query_count"], 4)

    def test_range_bins_must_be_ordered_nonoverlapping_and_inside_model_range(self):
        with self.assertRaisesRegex(ValueError, "有序且不重叠"):
            validate_range_bins(
                (("x0_20", 0.0, 20.0), ("x10_30", 10.0, 30.0)),
                (0.0, -20.0, -6.0, 40.0, 20.0, 10.0),
            )

    def test_range_bins_reject_nonfinite_bounds(self):
        with self.assertRaisesRegex(ValueError, "有限数"):
            validate_range_bins(
                (("x0_nan", 0.0, float("nan")),),
                (0.0, -20.0, -6.0, 40.0, 20.0, 10.0),
            )

    def test_validation_split_fails_fast_when_selected_target_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(RuntimeError, "target"):
                resolve_target_path(temp_dir, "000007", "validation")

            self.assertIsNone(resolve_target_path(temp_dir, "000007", "all"))


if __name__ == "__main__":
    unittest.main()
