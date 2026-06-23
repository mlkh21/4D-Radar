#!/usr/bin/env python
"""Tests for raw-LiDAR interactive inference visualization."""

import argparse
import os
import sys
import tempfile
import unittest

import numpy as np


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from generate_interactive_inference_compare import (
    build_frame,
    load_raw_lidar_points,
    resolve_raw_lidar_path,
)


class RawLidarVisualizationTest(unittest.TestCase):
    def test_raw_lidar_loader_removes_only_invalid_and_zero_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lidar.npy")
            np.save(
                path,
                np.array(
                    [[0, 0, 0, 0], [1, 2, 3, 4], [np.nan, 1, 2, 3]],
                    dtype=np.float32,
                ),
            )

            points = load_raw_lidar_points(path)

            self.assertEqual(points.shape, (1, 4))
            np.testing.assert_array_equal(points[0], np.array([1, 2, 3, 4], dtype=np.float32))

    def test_resolves_raw_lidar_using_sequence_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            lidar_dir = os.path.join(tmp, "livox_lidar")
            os.makedirs(lidar_dir)
            for name in ("100.0.npy", "101.0.npy", "102.0.npy"):
                np.save(os.path.join(lidar_dir, name), np.zeros((1, 4), dtype=np.float32))
            index_path = os.path.join(tmp, "lidar_index_sequence.txt")
            with open(index_path, "w", encoding="utf-8") as handle:
                handle.write("2\n0\n1\n")

            path = resolve_raw_lidar_path("000001", lidar_dir, index_path)

            self.assertEqual(os.path.basename(path), "100.0.npy")

    def test_generated_html_replaces_target_with_raw_lidar(self):
        with tempfile.TemporaryDirectory() as tmp:
            pre_dir = os.path.join(tmp, "pre")
            radar_dir = os.path.join(pre_dir, "radar_voxel")
            lidar_dir = os.path.join(tmp, "livox_lidar")
            ldm_dir = os.path.join(tmp, "ldm")
            cd_dir = os.path.join(tmp, "cd")
            output_dir = os.path.join(tmp, "out")
            for path in (radar_dir, lidar_dir, ldm_dir, cd_dir, output_dir):
                os.makedirs(path)

            np.savez_compressed(
                os.path.join(radar_dir, "000000.npz"),
                coords=np.array([[1, 1, 1]], dtype=np.int32),
                features=np.ones((1, 4), dtype=np.float32),
                shape=np.array([4, 4, 4, 4], dtype=np.int32),
            )
            np.save(os.path.join(lidar_dir, "100.0.npy"), np.array([[10, 0, 1, 5]], dtype=np.float32))
            np.save(os.path.join(ldm_dir, "000000_pcl.npy"), np.array([[11, 0, 1]], dtype=np.float32))
            np.save(os.path.join(cd_dir, "000000_pcl.npy"), np.array([[12, 0, 1]], dtype=np.float32))
            index_path = os.path.join(tmp, "lidar_index_sequence.txt")
            with open(index_path, "w", encoding="utf-8") as handle:
                handle.write("0\n")

            args = argparse.Namespace(
                pre_dir=pre_dir,
                raw_lidar_dir=lidar_dir,
                lidar_index_file=index_path,
                ldm_dir=ldm_dir,
                cd_dir=cd_dir,
                output_dir=output_dir,
                pc_range=[0, -20, -6, 120, 20, 10],
                z_min=-1.0,
                x_max=80.0,
                max_radar_points=100,
                max_lidar_points=100,
                max_pred_points=100,
            )

            output = build_frame(args, "000000", {})
            with open(output, "r", encoding="utf-8") as handle:
                html = handle.read()

            self.assertIn('"name":"raw_lidar"', html)
            self.assertNotIn('"name":"target"', html)
            self.assertIn("100.0.npy", html)


if __name__ == "__main__":
    unittest.main()
