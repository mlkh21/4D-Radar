#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证 oracle target 数量匹配诊断的阈值、点云与报告协议。"""

import csv
import importlib.util
import json
import os
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_PATH = os.path.join(
    PROJECT_ROOT,
    "test/diagnostics/occupancy/diagnose_oracle_target_adaptation.py",
)


def load_oracle_module(test_case):
    """按文件路径加载待实现脚本，缺失时产生明确 RED 断言。"""
    if not os.path.isfile(MODULE_PATH):
        test_case.fail("尚未创建独立 oracle target 诊断脚本")
    spec = importlib.util.spec_from_file_location(
        "diagnose_oracle_target_adaptation",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OracleTargetAdaptationTest(unittest.TestCase):
    def test_diagnostic_is_decoupled_from_formal_inference_and_threshold_sweep(self):
        """oracle 诊断只能读取体素，不应反向加载正式推理或阈值扫描入口。"""
        with open(MODULE_PATH, "r", encoding="utf-8") as source_file:
            source = source_file.read()

        self.assertNotIn(
            "diffusion_consistency_radar.scripts.inference",
            source,
        )
        self.assertNotIn(
            "diffusion_consistency_radar.scripts.sweep_occ_threshold",
            source,
        )

    @staticmethod
    def _write_valid_pair(root_dir):
        pred_dir = os.path.join(root_dir, "pred")
        target_dir = os.path.join(root_dir, "target")
        os.makedirs(pred_dir)
        os.makedirs(target_dir)

        prediction = np.zeros((2, 1, 4, 1), dtype=np.float32)
        prediction[0, 0, :, 0] = [0.9, 0.8, 0.2, 0.1]
        prediction[1, 0, :, 0] = [10.0, 20.0, 30.0, 40.0]
        np.save(os.path.join(pred_dir, "000000_voxel.npy"), prediction)

        target = np.zeros((4, 1, 1, 4), dtype=np.float32)
        target[:2, 0, 0, 0] = 1.0
        target[:2, 0, 0, 3] = 1.0
        np.save(os.path.join(target_dir, "000000.npy"), target)
        return pred_dir, target_dir

    @staticmethod
    def _run(module, pred_dir, target_dir, output_dir):
        return module.run_diagnostic(
            pred_voxel_dir=pred_dir,
            target_voxel_dir=target_dir,
            output_dir=output_dir,
            target_threshold=0.5,
            source_pc_range=(0, 0, 0, 4, 1, 1),
            model_pc_range=(0, 0, 0, 4, 1, 1),
            target_size=(1, 4, 1),
            voxel_size=None,
            max_files=0,
        )

    def test_oracle_threshold_matches_requested_topk(self):
        module = load_oracle_module(self)
        prediction = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float32)

        threshold, effective_count = module.find_oracle_occ_threshold(prediction, 2)

        self.assertEqual(effective_count, 2)
        self.assertEqual(int(np.count_nonzero(prediction > threshold)), 2)

    def test_zero_target_count_preserves_legacy_minimum_one(self):
        module = load_oracle_module(self)
        prediction = np.asarray([0.9, 0.1], dtype=np.float32)

        threshold, effective_count = module.find_oracle_occ_threshold(prediction, 0)

        self.assertEqual(effective_count, 1)
        self.assertEqual(int(np.count_nonzero(prediction > threshold)), 1)

    def test_diagnostic_writes_pointcloud_csv_and_non_deployable_json(self):
        module = load_oracle_module(self)
        with tempfile.TemporaryDirectory() as temp_dir:
            pred_dir, target_dir = self._write_valid_pair(temp_dir)
            output_dir = os.path.join(temp_dir, "oracle")

            report = self._run(module, pred_dir, target_dir, output_dir)

            pointcloud_path = os.path.join(output_dir, "000000_oracle_pcl.npy")
            pointcloud = np.load(pointcloud_path)
            self.assertEqual(pointcloud.shape, (2, 4))

            csv_path = os.path.join(output_dir, "oracle_target_adaptation_frames.csv")
            with open(csv_path, "r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["target_occ_count"], "2")
            self.assertEqual(rows[0]["effective_match_count"], "2")
            self.assertEqual(rows[0]["oracle_pred_point_count"], "2")
            self.assertEqual(rows[0]["oracle_pointcloud_file"], "000000_oracle_pcl.npy")

            json_path = os.path.join(output_dir, "oracle_target_adaptation_report.json")
            with open(json_path, "r", encoding="utf-8") as json_file:
                saved_report = json.load(json_file)
            self.assertEqual(saved_report["protocol"], "oracle_target_count_matching")
            self.assertIs(saved_report["deployable"], False)
            self.assertIn("不得作为正式推理性能", saved_report["warning"])
            self.assertEqual(saved_report, report)

    def test_missing_target_fails_before_output_creation(self):
        module = load_oracle_module(self)
        with tempfile.TemporaryDirectory() as temp_dir:
            pred_dir, target_dir = self._write_valid_pair(temp_dir)
            os.unlink(os.path.join(target_dir, "000000.npy"))
            output_dir = os.path.join(temp_dir, "oracle")

            with self.assertRaisesRegex(RuntimeError, "target"):
                self._run(module, pred_dir, target_dir, output_dir)

            self.assertFalse(os.path.exists(output_dir))

    def test_invalid_prediction_shape_fails_before_output_creation(self):
        module = load_oracle_module(self)
        with tempfile.TemporaryDirectory() as temp_dir:
            pred_dir, target_dir = self._write_valid_pair(temp_dir)
            np.save(
                os.path.join(pred_dir, "000000_voxel.npy"),
                np.zeros((1, 1, 4, 1), dtype=np.float32),
            )
            output_dir = os.path.join(temp_dir, "oracle")

            with self.assertRaisesRegex(ValueError, "C,Z,X,Y"):
                self._run(module, pred_dir, target_dir, output_dir)

            self.assertFalse(os.path.exists(output_dir))

    def test_nonempty_output_directory_is_never_overwritten(self):
        module = load_oracle_module(self)
        with tempfile.TemporaryDirectory() as temp_dir:
            pred_dir, target_dir = self._write_valid_pair(temp_dir)
            output_dir = os.path.join(temp_dir, "oracle")
            os.makedirs(output_dir)
            marker_path = os.path.join(output_dir, "marker.txt")
            with open(marker_path, "w", encoding="utf-8") as marker_file:
                marker_file.write("preserve")

            with self.assertRaisesRegex(ValueError, "非空"):
                self._run(module, pred_dir, target_dir, output_dir)

            with open(marker_path, "r", encoding="utf-8") as marker_file:
                self.assertEqual(marker_file.read(), "preserve")
            self.assertEqual(os.listdir(output_dir), ["marker.txt"])


if __name__ == "__main__":
    unittest.main()
