#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared_visibility_eval import bev_iou, nearest_neighbor_metrics


class SharedVisibilityEvalTest(unittest.TestCase):
    def test_nearest_neighbor_metrics_reports_visible_fraction(self):
        radar = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 5.0, 0.0],
        ]
        lidar = [
            [0.1, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [9.0, 9.0, 0.0],
        ]

        metrics = nearest_neighbor_metrics(radar, lidar, thresholds=[0.25, 1.0])

        self.assertEqual(metrics["radar_count"], 3)
        self.assertEqual(metrics["lidar_count"], 3)
        self.assertAlmostEqual(metrics["nn_mean"], (0.1 + 0.2 + math.sqrt(32.0)) / 3.0)
        self.assertAlmostEqual(metrics["match_ratio_0.25"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["match_ratio_1"], 2.0 / 3.0)

    def test_bev_iou_uses_occupied_xy_cells(self):
        pc_range = [0.0, 0.0, -1.0, 4.0, 4.0, 1.0]
        radar = [
            [0.2, 0.2, 0.0],
            [1.2, 0.2, 0.0],
            [3.2, 3.2, 0.0],
        ]
        lidar = [
            [0.8, 0.8, 0.0],
            [1.8, 0.8, 0.0],
            [2.2, 2.2, 0.0],
        ]

        metrics = bev_iou(radar, lidar, pc_range=pc_range, cell_size=1.0)

        self.assertEqual(metrics["bev_intersection"], 2)
        self.assertEqual(metrics["bev_union"], 4)
        self.assertAlmostEqual(metrics["bev_iou"], 0.5)


if __name__ == "__main__":
    unittest.main()
