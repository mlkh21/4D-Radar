"""测试 garden/loop3 场景分布审计中的采样、裁剪和统计逻辑。"""

import os
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
DIAGNOSTIC_DIR = os.path.join(ROOT, "test", "diagnostics", "radar")
if DIAGNOSTIC_DIR not in sys.path:
    sys.path.insert(0, DIAGNOSTIC_DIR)


def write_sparse(path, coords, features, shape):
    np.savez_compressed(
        path,
        coords=np.asarray(coords, dtype=np.int32),
        features=np.asarray(features, dtype=np.float32),
        shape=np.asarray(shape, dtype=np.int32),
    )


class SceneDistributionAuditTest(unittest.TestCase):
    def test_evenly_spaced_indices_include_both_ends(self):
        from audit_scene_distribution_shift import evenly_spaced_indices

        self.assertEqual(evenly_spaced_indices(10, 4), [0, 3, 6, 9])
        self.assertEqual(evenly_spaced_indices(3, 0), [0, 1, 2])
        self.assertEqual(evenly_spaced_indices(0, 5), [])

    def test_sparse_stats_use_physical_centers_and_model_crop(self):
        from audit_scene_distribution_shift import sparse_voxel_stats

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "000000.npz")
            write_sparse(
                path,
                coords=[
                    [0, 0, 0],  # z=-1.5，位于模型裁剪范围外
                    [0, 0, 1],  # x=0.5, z=-0.5
                    [1, 1, 2],  # x=1.5, z=0.5
                    [2, 1, 2],  # x=2.5，位于模型裁剪范围外
                ],
                features=[
                    [1.0, 1.0, 99.0, 50.0],
                    [1.0, 2.0, 2.0, 4.0],
                    [1.0, 3.0, -2.0, 8.0],
                    [1.0, 4.0, 99.0, 50.0],
                ],
                shape=[4, 2, 4, 4],
            )

            stats = sparse_voxel_stats(
                path,
                source_pc_range=(0.0, -1.0, -2.0, 4.0, 1.0, 2.0),
                model_pc_range=(0.0, -1.0, -1.0, 2.0, 1.0, 2.0),
                x_edges=(0.0, 1.0, 2.0),
                z_edges=(-1.0, 0.0, 2.0),
                include_radar_channels=True,
            )

            self.assertEqual(stats["occupied_count"], 2)
            self.assertEqual(stats["x_band_counts"], [1, 1])
            self.assertEqual(stats["z_band_counts"], [1, 1])
            self.assertAlmostEqual(stats["doppler_mean"], 0.0)
            self.assertAlmostEqual(stats["doppler_abs_mean"], 2.0)
            self.assertAlmostEqual(stats["doppler_variance_mean"], 6.0)
            self.assertAlmostEqual(stats["doppler_variance_p90"], 7.6)

    def test_scene_pairing_uses_only_common_frame_stems(self):
        from audit_scene_distribution_shift import paired_frame_paths

        with tempfile.TemporaryDirectory() as tmp:
            radar = os.path.join(tmp, "radar_voxel")
            target = os.path.join(tmp, "target_voxel")
            os.makedirs(radar)
            os.makedirs(target)
            for stem in ("000000", "000001", "000002"):
                open(os.path.join(radar, f"{stem}.npz"), "wb").close()
            for stem in ("000001", "000002", "000003"):
                open(os.path.join(target, f"{stem}.npz"), "wb").close()

            pairs = paired_frame_paths(radar, target)

            self.assertEqual([item[0] for item in pairs], ["000001", "000002"])

    def test_ir_npy_statistics_are_read_without_loading_full_dataset(self):
        from audit_scene_distribution_shift import _ir_stats

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "000000_ir.npy")
            np.save(path, np.arange(64, dtype=np.float32).reshape(8, 8))

            stats = _ir_stats(path)

            self.assertEqual(stats["ir_available"], 1.0)
            self.assertGreater(stats["ir_mean"], 0.0)
            self.assertGreater(stats["ir_p90"], stats["ir_p10"])


if __name__ == "__main__":
    unittest.main()
