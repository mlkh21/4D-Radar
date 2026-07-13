#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试已保存 LDM voxel 与 LiDAR target 的垂直结构评估脚本。
"""

import csv
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
MODULE_DIR = os.path.join(ROOT, "test", "evaluation", "ldm")
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from evaluate_ldm_vertical_structure import (
    evaluate_directories,
    iter_prediction_files,
    load_prediction_occupancy,
    load_target_occupancy,
)


def write_sparse_target(path, occ_zyx):
    """按预处理协议写入 sparse target，输入为 ZXY 占用。"""
    occ_xyz = np.asarray(occ_zyx, dtype=np.float32).transpose(1, 2, 0)
    dense = np.zeros(occ_xyz.shape + (4,), dtype=np.float32)
    dense[..., 0] = occ_xyz
    occupied = dense[..., 0] > 0.0
    coords = np.column_stack(np.where(occupied))
    features = dense[occupied]
    np.savez(path, coords=coords, features=features, shape=dense.shape)


def make_column(shape=(6, 2, 2), x=0, y=0, z_indices=(0, 1, 2, 3), value=1.0):
    """构造一个 ZXY 竖向占用列。"""
    volume = np.zeros(shape, dtype=np.float32)
    for z in z_indices:
        volume[z, x, y] = value
    return volume


class LDMVerticalStructureEvalTest(unittest.TestCase):
    def test_perfect_case_writes_summary_recalls_as_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred_dir = os.path.join(tmp, "pred")
            target_dir = os.path.join(tmp, "target")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(pred_dir)
            os.makedirs(target_dir)

            occ = make_column()
            pred = np.zeros((4,) + occ.shape, dtype=np.float32)
            pred[0] = occ
            np.save(os.path.join(pred_dir, "000000_voxel.npy"), pred)
            write_sparse_target(os.path.join(target_dir, "000000.npz"), occ)

            summary = evaluate_directories(
                pred_voxel_dir=pred_dir,
                target_voxel_dir=target_dir,
                output_dir=out_dir,
                occ_threshold=0.05,
                target_threshold=0.5,
                pc_range=(0.0, 0.0, 0.0, 2.0, 2.0, 6.0),
                source_pc_range=(0.0, 0.0, 0.0, 2.0, 2.0, 6.0),
                target_size=occ.shape,
                max_files=0,
            )

            self.assertEqual(summary["frame_id"], "__summary__")
            self.assertEqual(summary["frames"], 1)
            for key in (
                "height_coverage_recall",
                "top_height_recall",
                "vertical_connectivity_recall",
                "trunk_region_recall",
            ):
                self.assertAlmostEqual(summary[key], 1.0, msg=key)
            self.assertTrue(os.path.exists(os.path.join(out_dir, "vertical_structure_metrics.csv")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "vertical_structure_summary.csv")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "vertical_structure_report.md")))

    def test_missing_top_and_broken_column_reduce_corresponding_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred_dir = os.path.join(tmp, "pred")
            target_dir = os.path.join(tmp, "target")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(pred_dir)
            os.makedirs(target_dir)

            target = make_column()
            missing_top = make_column(z_indices=(0, 1, 2))
            broken = make_column(x=1, y=1, z_indices=(0, 2, 3))
            target_two_cols = target + make_column(x=1, y=1)
            pred_two_cols = missing_top + broken
            np.save(os.path.join(pred_dir, "000000_voxel.npy"), pred_two_cols)
            write_sparse_target(os.path.join(target_dir, "000000.npz"), target_two_cols)

            summary = evaluate_directories(
                pred_voxel_dir=pred_dir,
                target_voxel_dir=target_dir,
                output_dir=out_dir,
                pc_range=(0.0, 0.0, 0.0, 2.0, 2.0, 6.0),
                source_pc_range=(0.0, 0.0, 0.0, 2.0, 2.0, 6.0),
                target_size=target.shape,
                max_files=0,
            )

            self.assertLess(summary["height_coverage_recall"], 1.0)
            self.assertLess(summary["top_height_recall"], 1.0)
            self.assertLess(summary["vertical_connectivity_recall"], 1.0)

    def test_sparse_target_npz_axis_conversion_returns_zyx_occupancy(self):
        with tempfile.TemporaryDirectory() as tmp:
            target_path = os.path.join(tmp, "000000.npz")
            target = np.zeros((3, 4, 5), dtype=np.float32)
            target[2, 1, 3] = 1.0
            write_sparse_target(target_path, target)

            loaded = load_target_occupancy(
                target_path,
                target_threshold=0.5,
                source_pc_range=(0.0, 0.0, 0.0, 4.0, 5.0, 3.0),
                model_pc_range=(0.0, 0.0, 0.0, 4.0, 5.0, 3.0),
                target_size=target.shape,
            )

            self.assertEqual(loaded.shape, target.shape)
            self.assertEqual(float(loaded[2, 1, 3]), 1.0)
            self.assertEqual(int(loaded.sum()), 1)

    def test_target_loader_crops_source_range_to_model_range_before_resize(self):
        with tempfile.TemporaryDirectory() as tmp:
            target_path = os.path.join(tmp, "000000.npz")
            target = np.zeros((4, 6, 2), dtype=np.float32)
            target[1, 1, 0] = 1.0
            target[1, 5, 0] = 1.0
            write_sparse_target(target_path, target)

            loaded = load_target_occupancy(
                target_path,
                target_threshold=0.5,
                source_pc_range=(0.0, 0.0, 0.0, 6.0, 2.0, 4.0),
                model_pc_range=(0.0, 0.0, 0.0, 3.0, 2.0, 4.0),
                target_size=(4, 3, 2),
            )

            self.assertEqual(loaded.shape, (4, 3, 2))
            self.assertEqual(int(loaded.sum()), 1)
            self.assertEqual(float(loaded[1, 1, 0]), 1.0)

    def test_prediction_layouts_extract_occupancy_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            occ = make_column(shape=(3, 2, 2), z_indices=(0, 1, 2))

            czyx_path = os.path.join(tmp, "czyx.npy")
            czyx = np.zeros((4,) + occ.shape, dtype=np.float32)
            czyx[0] = occ
            np.save(czyx_path, czyx)
            np.testing.assert_array_equal(load_prediction_occupancy(czyx_path), occ)

            zxyc_path = os.path.join(tmp, "zxyc.npy")
            zxyc = np.zeros(occ.shape + (4,), dtype=np.float32)
            zxyc[..., 0] = occ
            np.save(zxyc_path, zxyc)
            np.testing.assert_array_equal(load_prediction_occupancy(zxyc_path), occ)

            zxy_path = os.path.join(tmp, "zxy.npy")
            np.save(zxy_path, occ)
            np.testing.assert_array_equal(load_prediction_occupancy(zxy_path), occ)

    def test_prediction_channel_last_layout_keeps_z_axis_when_z_equals_channel_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            occ = make_column(shape=(4, 2, 2), z_indices=(0, 1, 2, 3))
            zxyc_path = os.path.join(tmp, "zxyc_z4.npy")
            zxyc = np.zeros(occ.shape + (4,), dtype=np.float32)
            zxyc[..., 0] = occ
            np.save(zxyc_path, zxyc)

            loaded = load_prediction_occupancy(zxyc_path)

            self.assertEqual(loaded.shape, occ.shape)
            np.testing.assert_array_equal(loaded, occ)

    def test_max_files_limits_evaluated_frame_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred_dir = os.path.join(tmp, "pred")
            target_dir = os.path.join(tmp, "target")
            out_dir = os.path.join(tmp, "out")
            os.makedirs(pred_dir)
            os.makedirs(target_dir)
            occ = make_column()

            for frame_id in ("000000", "000001", "000002"):
                np.save(os.path.join(pred_dir, f"{frame_id}_voxel.npy"), occ)
                write_sparse_target(os.path.join(target_dir, f"{frame_id}.npz"), occ)

            summary = evaluate_directories(
                pred_voxel_dir=pred_dir,
                target_voxel_dir=target_dir,
                output_dir=out_dir,
                pc_range=(0.0, 0.0, 0.0, 2.0, 2.0, 6.0),
                source_pc_range=(0.0, 0.0, 0.0, 2.0, 2.0, 6.0),
                target_size=occ.shape,
                max_files=2,
            )

            self.assertEqual(summary["frames"], 2)
            with open(os.path.join(out_dir, "vertical_structure_metrics.csv"), newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([row["frame_id"] for row in rows], ["000000", "000001"])

    def test_prediction_iterator_ignores_ldm_sidecar_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred_dir = os.path.join(tmp, "pred")
            os.makedirs(pred_dir)
            occ = make_column()

            np.save(os.path.join(pred_dir, "000000_voxel.npy"), occ)
            np.save(os.path.join(pred_dir, "000000_uncertainty.npy"), occ)
            np.save(os.path.join(pred_dir, "000000_pcl.npy"), occ)

            files = iter_prediction_files(pred_dir)

            self.assertEqual([os.path.basename(path) for path in files], ["000000_voxel.npy"])


if __name__ == "__main__":
    unittest.main()
