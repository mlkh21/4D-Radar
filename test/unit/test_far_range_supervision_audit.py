#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证远距离监督审计的距离分带、signed 时序稳定性和射线统计纯函数。"""

import importlib.util
import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_PATH = os.path.join(
    ROOT,
    "test",
    "diagnostics",
    "alignment",
    "audit_far_range_supervision.py",
)
SPEC = importlib.util.spec_from_file_location("far_range_audit", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FarRangeSupervisionAuditTest(unittest.TestCase):
    def test_physical_bands_keep_80m_boundary_unambiguous(self):
        coords = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.int64)
        centers = MODULE.physical_centers(
            coords,
            shape=(3, 1, 1),
            pc_range=(0, -1, -1, 120, 1, 1),
        )
        counts = MODULE.count_bands_x(centers[:, 0])
        self.assertEqual(counts, {"near_0_80": 2, "far_80_120": 1})

    def test_temporal_jaccard_uses_union_and_handles_empty(self):
        self.assertAlmostEqual(
            MODULE.jaccard_sorted(
                np.asarray([1, 2, 3]),
                np.asarray([2, 3, 4]),
            ),
            0.5,
        )

    def test_raw_point_bands_exclude_points_outside_yz_grid(self):
        points = np.asarray(
            [
                [40.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [100.0, 21.0, 0.0],
                [40.0, 0.0, 11.0],
            ]
        )
        self.assertEqual(
            MODULE.count_points_in_grid_bands(points),
            {"near_0_80": 1, "far_80_120": 1},
        )
        self.assertTrue(
            np.isnan(MODULE.jaccard_sorted(np.asarray([]), np.asarray([])))
        )

    def test_sparse_ray_counts_do_not_mark_behind_endpoint(self):
        coords = np.asarray([[3, 1, 1]], dtype=np.int64)
        counts = MODULE.sparse_ray_band_counts(
            coords,
            shape=(6, 3, 3),
            pc_range=(0, -1, -1, 120, 2, 2),
        )
        self.assertGreater(counts["near_0_80"], 0)
        self.assertEqual(counts["far_80_120"], 0)


if __name__ == "__main__":
    unittest.main()
