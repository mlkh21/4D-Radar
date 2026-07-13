#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证占用阈值扫描时 target 的物理裁剪与目标网格协议。"""

import os
import json
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion_consistency_radar.scripts.sweep_occ_threshold import (
    evaluate_task_thresholds,
    load_target_occ_resized,
    main,
    parse_thresholds,
    prepare_evaluation_files,
    resolve_target_path,
    select_evaluation_files,
    select_recommended_threshold,
    select_threshold_with_constraints,
    validate_range_bins,
)


class OccThresholdGridProtocolTest(unittest.TestCase):
    @staticmethod
    def _constraint_metrics():
        return {
            0.2: {
                "task_bev_f1": 0.90,
                "task_bev_iou": 0.80,
                "task_bev_precision": 0.95,
                "task_bev_recall": 0.60,
                "pred_to_target_ratio": 1.0,
                "bands": {"x0_20": {"task_bev_recall": 0.55}},
            },
            0.3: {
                "task_bev_f1": 0.85,
                "task_bev_iou": 0.75,
                "task_bev_precision": 0.90,
                "task_bev_recall": 0.82,
                "pred_to_target_ratio": 1.0,
                "bands": {"x0_20": {"task_bev_recall": 0.72}},
            },
            0.4: {
                "task_bev_f1": 0.80,
                "task_bev_iou": 0.70,
                "task_bev_precision": 0.88,
                "task_bev_recall": 0.90,
                "pred_to_target_ratio": 1.0,
                "bands": {"x0_20": {"task_bev_recall": 0.65}},
            },
        }

    @staticmethod
    def _frame_metrics(tp, fp, fn):
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        return {
            "voxel_precision": precision,
            "voxel_recall": recall,
            "voxel_f1": f1,
            "voxel_iou": tp / max(tp + fp + fn, 1),
            "voxel_tp": tp,
            "voxel_fp": fp,
            "voxel_fn": fn,
            "task_bev_precision": precision,
            "task_bev_recall": recall,
            "task_bev_f1": f1,
            "task_bev_iou": tp / max(tp + fp + fn, 1),
            "task_match_ratio_2": None,
            "matched_pred_count": 0.0,
            "match_query_count": 0,
            "pred_count": tp + fp,
            "target_count": tp + fn,
            "bev_tp": tp,
            "bev_fp": fp,
            "bev_fn": fn,
            "bands": {
                "x0_40": {
                    "pred_count": tp + fp,
                    "target_count": tp + fn,
                    "bev_tp": tp,
                    "bev_fp": fp,
                    "bev_fn": fn,
                    "matched_pred_count": 0.0,
                    "match_query_count": 0,
                }
            },
        }

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

    def test_constraint_helper_preserves_unconstrained_selection(self):
        result = select_threshold_with_constraints(
            self._constraint_metrics(),
            range_bin_labels=("x0_20",),
        )

        self.assertEqual(result["unconstrained_recommended_threshold"], 0.2)
        self.assertEqual(result["recommended_threshold"], 0.2)
        self.assertTrue(result["constraint_satisfied"])
        self.assertEqual(result["effective_constraints"], {})
        self.assertIsNone(result["near_band_label"])

    def test_global_recall_constraint_filters_candidates_before_metric_sort(self):
        result = select_threshold_with_constraints(
            self._constraint_metrics(),
            min_task_bev_recall=0.8,
            range_bin_labels=("x0_20",),
        )

        self.assertEqual(result["recommended_threshold"], 0.3)
        self.assertTrue(result["constraint_satisfied"])

    def test_global_and_near_recall_constraints_must_both_hold(self):
        result = select_threshold_with_constraints(
            self._constraint_metrics(),
            min_task_bev_recall=0.8,
            min_near_bev_recall=0.68,
            near_band_label="x0_20",
            range_bin_labels=("x0_20",),
        )

        self.assertEqual(result["recommended_threshold"], 0.3)
        self.assertEqual(
            result["effective_constraints"],
            {
                "min_task_bev_recall": 0.8,
                "near_band_label": "x0_20",
                "min_near_bev_recall": 0.68,
            },
        )

    def test_infeasible_constraints_choose_best_minimum_normalized_satisfaction(self):
        metrics = self._constraint_metrics()
        metrics[0.2]["task_bev_f1"] = 0.99
        result = select_threshold_with_constraints(
            metrics,
            min_task_bev_recall=0.95,
            min_near_bev_recall=0.9,
            near_band_label="x0_20",
            range_bin_labels=("x0_20",),
        )

        # 0.3 的 min(0.82/0.95, 0.72/0.9)=0.8，高于其他候选。
        self.assertEqual(result["recommended_threshold"], 0.3)
        self.assertFalse(result["constraint_satisfied"])
        self.assertIn("无阈值满足", result["constraint_reason"])

    def test_constraint_validation_rejects_invalid_range_and_band(self):
        for value in (-0.01, 1.01, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, r"\[0, 1\]|有限"):
                    select_threshold_with_constraints(
                        self._constraint_metrics(),
                        min_task_bev_recall=value,
                        range_bin_labels=("x0_20",),
                    )
        with self.assertRaisesRegex(ValueError, "range_bins"):
            select_threshold_with_constraints(
                self._constraint_metrics(),
                min_near_bev_recall=0.5,
                near_band_label="x20_40",
                range_bin_labels=("x0_20",),
            )

    def test_enabled_global_constraint_rejects_invalid_metric_values_with_path(self):
        for value in (None, float("nan"), float("inf"), "invalid"):
            with self.subTest(value=value):
                metrics = self._constraint_metrics()
                metrics[0.3]["task_bev_recall"] = value
                with self.assertRaisesRegex(
                    ValueError, r"threshold=0\.3.*task_bev_recall"
                ):
                    select_threshold_with_constraints(
                        metrics,
                        min_task_bev_recall=0.8,
                        range_bin_labels=("x0_20",),
                    )

    def test_enabled_near_constraint_rejects_missing_metric_path(self):
        mutations = (
            lambda item: item.pop("bands"),
            lambda item: item["bands"].pop("x0_20"),
            lambda item: item["bands"]["x0_20"].pop("task_bev_recall"),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                metrics = self._constraint_metrics()
                mutate(metrics[0.3])
                with self.assertRaisesRegex(
                    ValueError, r"threshold=0\.3.*bands\.x0_20\.task_bev_recall"
                ):
                    select_threshold_with_constraints(
                        metrics,
                        min_near_bev_recall=0.6,
                        near_band_label="x0_20",
                        range_bin_labels=("x0_20",),
                    )

    def test_enabled_near_constraint_rejects_invalid_metric_values(self):
        for value in (None, float("nan"), float("inf"), "invalid"):
            with self.subTest(value=value):
                metrics = self._constraint_metrics()
                metrics[0.3]["bands"]["x0_20"]["task_bev_recall"] = value
                with self.assertRaisesRegex(
                    ValueError, r"threshold=0\.3.*bands\.x0_20\.task_bev_recall"
                ):
                    select_threshold_with_constraints(
                        metrics,
                        min_near_bev_recall=0.6,
                        near_band_label="x0_20",
                        range_bin_labels=("x0_20",),
                    )

    def test_constraint_result_is_json_serializable_contract(self):
        result = select_threshold_with_constraints(
            self._constraint_metrics(),
            min_near_bev_recall=0.7,
            near_band_label="",
            range_bin_labels=("x0_20",),
        )

        encoded = json.loads(json.dumps(result))
        self.assertEqual(encoded["near_band_label"], "x0_20")
        self.assertIn("unconstrained_recommended_threshold", encoded)
        self.assertIn("recommended_threshold", encoded)
        self.assertIn("constraint_satisfied", encoded)
        self.assertIn("effective_constraints", encoded)

    def test_feasible_ties_reuse_original_task_metric_sorting(self):
        metrics = {
            0.3: {
                "task_bev_f1": 0.8, "task_bev_iou": 0.7,
                "task_bev_precision": 0.99, "task_bev_recall": 0.9,
                "pred_to_target_ratio": 1.0,
            },
            0.4: {
                "task_bev_f1": 0.8, "task_bev_iou": 0.8,
                "task_bev_precision": 0.8, "task_bev_recall": 0.9,
                "pred_to_target_ratio": 1.0,
            },
            0.6: {
                "task_bev_f1": 0.8, "task_bev_iou": 0.8,
                "task_bev_precision": 0.8, "task_bev_recall": 0.9,
                "pred_to_target_ratio": 1.0,
            },
        }

        result = select_threshold_with_constraints(
            metrics,
            min_task_bev_recall=0.8,
        )

        # precision 不参与旧任务排序；IoU 排除 0.3，阈值规则选择 0.4。
        self.assertEqual(result["recommended_threshold"], 0.4)

    def test_fallback_satisfaction_tie_reuses_original_task_sorting(self):
        metrics = {
            0.3: {
                "task_bev_f1": 0.8, "task_bev_iou": 0.7,
                "task_bev_precision": 0.99, "task_bev_recall": 0.7,
                "pred_to_target_ratio": 1.0,
            },
            0.4: {
                "task_bev_f1": 0.8, "task_bev_iou": 0.8,
                "task_bev_precision": 0.8, "task_bev_recall": 0.7,
                "pred_to_target_ratio": 1.0,
            },
            0.6: {
                "task_bev_f1": 0.8, "task_bev_iou": 0.8,
                "task_bev_precision": 0.8, "task_bev_recall": 0.7,
                "pred_to_target_ratio": 1.0,
            },
        }

        result = select_threshold_with_constraints(
            metrics,
            min_task_bev_recall=0.8,
        )

        self.assertFalse(result["constraint_satisfied"])
        self.assertEqual(result["recommended_threshold"], 0.4)

    def test_fallback_treats_nearly_equal_satisfaction_as_tied(self):
        metrics = {
            0.3: {
                "task_bev_f1": 0.9, "task_bev_iou": 0.8,
                "task_bev_recall": 0.7, "pred_to_target_ratio": 1.0,
            },
            0.4: {
                "task_bev_f1": 0.8, "task_bev_iou": 0.7,
                "task_bev_recall": 0.7000000000000001,
                "pred_to_target_ratio": 1.0,
            },
        }

        result = select_threshold_with_constraints(
            metrics,
            min_task_bev_recall=0.8,
        )

        self.assertFalse(result["constraint_satisfied"])
        self.assertEqual(result["recommended_threshold"], 0.3)

    def test_unconstrained_conflict_exactly_matches_legacy_selector(self):
        metrics = {
            0.3: {
                "task_bev_f1": 0.8, "task_bev_iou": 0.7,
                "task_bev_precision": 0.99, "pred_to_target_ratio": 1.0,
            },
            0.4: {
                "task_bev_f1": 0.8, "task_bev_iou": 0.8,
                "task_bev_precision": 0.5, "pred_to_target_ratio": 1.0,
            },
        }

        legacy = select_recommended_threshold(metrics)
        result = select_threshold_with_constraints(metrics)

        self.assertEqual(legacy, 0.4)
        self.assertEqual(result["unconstrained_recommended_threshold"], legacy)
        self.assertEqual(result["recommended_threshold"], legacy)

    def test_main_writes_constraint_json_and_constraint_changes_threshold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pred_dir = os.path.join(temp_dir, "pred")
            target_dir = os.path.join(temp_dir, "target")
            os.makedirs(pred_dir)
            os.makedirs(target_dir)
            np.save(os.path.join(pred_dir, "000000_voxel.npy"), np.zeros((1, 4, 4, 4)))
            np.save(os.path.join(target_dir, "000000.npy"), np.zeros((4, 4, 4, 4)))

            controlled = {
                0.2: self._frame_metrics(tp=6, fp=0, fn=4),
                0.8: self._frame_metrics(tp=8, fp=8, fn=2),
            }

            def run_cli(output_name, *constraint_args):
                output_json = os.path.join(temp_dir, output_name)
                argv = [
                    "sweep_occ_threshold.py",
                    "--pred_voxel_dir", pred_dir,
                    "--target_voxel_dir", target_dir,
                    "--thresholds", "0.2,0.8",
                    "--evaluation_split", "all",
                    "--range_bins", "0-40",
                    "--source_pc_range", "0", "-2", "-2", "40", "2", "2",
                    "--model_pc_range", "0", "-2", "-2", "40", "2", "2",
                    "--target_size", "4", "4", "4",
                    "--output_json", output_json,
                    *constraint_args,
                ]
                with mock.patch("sys.argv", argv), mock.patch(
                    "diffusion_consistency_radar.scripts.sweep_occ_threshold.evaluate_task_thresholds",
                    return_value=controlled,
                ):
                    main()
                with open(output_json, "r", encoding="utf-8") as handle:
                    return json.load(handle)

            unconstrained = run_cli("unconstrained.json")
            constrained = run_cli(
                "constrained.json", "--min_task_bev_recall", "0.75"
            )
            fallback = run_cli(
                "fallback.json", "--min_task_bev_recall", "0.95"
            )

        required_keys = {
            "unconstrained_recommended_threshold",
            "recommended_threshold",
            "constraints",
            "constraint_satisfied",
            "constraint_reason",
        }
        self.assertTrue(required_keys.issubset(constrained))
        self.assertEqual(unconstrained["recommended_threshold"], 0.2)
        self.assertEqual(constrained["unconstrained_recommended_threshold"], 0.2)
        self.assertEqual(constrained["recommended_threshold"], 0.8)
        self.assertEqual(constrained["constraints"], {"min_task_bev_recall": 0.75})
        self.assertTrue(constrained["constraint_satisfied"])
        self.assertIsNone(constrained["constraint_reason"])
        self.assertTrue(required_keys.issubset(fallback))
        self.assertFalse(fallback["constraint_satisfied"])
        self.assertEqual(fallback["constraints"], {"min_task_bev_recall": 0.95})
        self.assertIsInstance(fallback["constraint_reason"], str)
        self.assertEqual(fallback["unconstrained_recommended_threshold"], 0.2)

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
