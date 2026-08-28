# -*- coding: utf-8 -*-
"""文件功能：验证 Radar point-count/Doppler-validity 稀疏统计合同。"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _save_sparse_voxel(path, voxel):
    occupied = voxel[..., 0] > 0
    np.savez_compressed(
        path,
        coords=np.column_stack(np.where(occupied)),
        features=voxel[occupied],
        shape=np.asarray(voxel.shape, dtype=np.int64),
    )


def _example_points(with_doppler=True):
    rows = [
        [0.10, 0.10, 0.10, 2.0, 3.0],
        [0.20, 0.10, 0.10, 4.0, 5.0],
        [1.10, 0.10, 0.10, 1.0, -2.0],
    ]
    points = np.asarray(rows, dtype=np.float32)
    return points if with_doppler else points[:, :4]


class RadarStatisticsProtocolTest(unittest.TestCase):
    def test_voxelizer_preserves_four_channels_and_reports_counts(self):
        from NTU4DRadLM_pre_processing.NTU4DRadLM_pre_processing import (
            voxelize_pcl_airborne_optimized,
        )

        kwargs = {
            "pcl": _example_points(),
            "voxel_size": (1.0, 1.0, 1.0),
            "pc_range": (0.0, 0.0, 0.0, 2.0, 1.0, 1.0),
        }
        legacy_voxel = voxelize_pcl_airborne_optimized(**kwargs)
        voxel, statistics = voxelize_pcl_airborne_optimized(
            **kwargs,
            return_statistics=True,
        )

        np.testing.assert_allclose(voxel, legacy_voxel, atol=0.0, rtol=0.0)
        np.testing.assert_array_equal(
            statistics["coords"],
            np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            statistics["point_count"],
            np.asarray([2, 1], dtype=np.uint32),
        )
        np.testing.assert_array_equal(
            statistics["doppler_valid_count"],
            np.asarray([2, 1], dtype=np.uint32),
        )

    def test_missing_doppler_column_is_not_misreported_as_valid_zero(self):
        from NTU4DRadLM_pre_processing.NTU4DRadLM_pre_processing import (
            voxelize_pcl_airborne_optimized,
        )

        voxel, statistics = voxelize_pcl_airborne_optimized(
            _example_points(with_doppler=False),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 2.0, 1.0, 1.0),
            return_statistics=True,
        )

        self.assertEqual(float(voxel[0, 0, 0, 2]), 0.0)
        self.assertEqual(float(voxel[0, 0, 0, 3]), 0.0)
        np.testing.assert_array_equal(
            statistics["doppler_valid_count"],
            np.zeros(2, dtype=np.uint32),
        )

    def test_sparse_radar_statistics_roundtrip_and_tamper_rejection(self):
        from NTU4DRadLM_pre_processing.NTU4DRadLM_pre_processing import (
            voxelize_pcl_airborne_optimized,
        )
        from diffusion_consistency_radar.radar_statistics import (
            load_sparse_radar_voxel,
            save_sparse_radar_voxel,
        )

        voxel, statistics = voxelize_pcl_airborne_optimized(
            _example_points(),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 2.0, 1.0, 1.0),
            return_statistics=True,
        )
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "000000.npz")
            save_sparse_radar_voxel(path, voxel, statistics)
            loaded, summary = load_sparse_radar_voxel(
                path,
                require_statistics=True,
            )

            np.testing.assert_allclose(loaded, voxel, atol=0.0, rtol=0.0)
            self.assertEqual(summary["occupied_voxels"], 2)
            self.assertEqual(summary["total_point_count"], 3)
            self.assertEqual(summary["total_doppler_valid_count"], 3)
            self.assertEqual(summary["multi_point_voxels"], 1)
            self.assertEqual(summary["doppler_multi_sample_voxels"], 1)

            with np.load(path, allow_pickle=False) as payload:
                values = {key: payload[key] for key in payload.files}
            values["doppler_valid_count"] = np.asarray([3, 1], dtype=np.uint32)
            np.savez_compressed(path, **values)
            with self.assertRaisesRegex(ValueError, "Doppler.*point count"):
                load_sparse_radar_voxel(path, require_statistics=True)

    def test_strict_loader_and_dataset_reject_missing_statistics(self):
        from diffusion_consistency_radar.cm.dataset_loader import (
            NTU4DRadLM_VoxelDataset,
        )
        from diffusion_consistency_radar.radar_statistics import (
            load_sparse_radar_voxel,
        )

        with tempfile.TemporaryDirectory() as root:
            scene = os.path.join(root, "garden")
            for name in ("radar_voxel", "target_voxel", "lidar_voxel", "ir_image"):
                os.makedirs(os.path.join(scene, name), exist_ok=True)
            voxel = np.zeros((2, 1, 1, 4), dtype=np.float32)
            voxel[0, 0, 0, 0] = 1.0
            radar_path = os.path.join(scene, "radar_voxel", "000000.npz")
            for name in ("radar_voxel", "target_voxel", "lidar_voxel"):
                _save_sparse_voxel(os.path.join(scene, name, "000000.npz"), voxel)
            with open(os.path.join(scene, "preprocess_policy.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "source_scene": "garden",
                        "frames_written": 1,
                        "voxel_coordinate_frame": "lidar",
                        "radar_statistics_protocol": "radar_point_count_doppler_validity_v1",
                    },
                    handle,
                )

            with self.assertRaisesRegex(ValueError, "statistics"):
                load_sparse_radar_voxel(radar_path, require_statistics=True)
            with self.assertRaisesRegex(ValueError, "statistics"):
                NTU4DRadLM_VoxelDataset(
                    root,
                    scene_names=["garden"],
                    use_augmentation=False,
                    target_size=(1, 2, 1),
                    source_pc_range=(0.0, 0.0, 0.0, 2.0, 1.0, 1.0),
                    allow_legacy_radar_units=True,
                    require_radar_statistics=True,
                )

    def test_dataset_exposes_audit_summary_without_changing_model_tensor(self):
        from NTU4DRadLM_pre_processing.NTU4DRadLM_pre_processing import (
            voxelize_pcl_airborne_optimized,
        )
        from diffusion_consistency_radar.cm.dataset_loader import (
            NTU4DRadLM_VoxelDataset,
            collate_voxel_samples,
        )
        from diffusion_consistency_radar.radar_statistics import (
            save_sparse_radar_voxel,
        )

        voxel, statistics = voxelize_pcl_airborne_optimized(
            _example_points(),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 2.0, 1.0, 1.0),
            return_statistics=True,
        )
        with tempfile.TemporaryDirectory() as root:
            scene = os.path.join(root, "garden")
            for name in ("radar_voxel", "target_voxel", "lidar_voxel", "ir_image"):
                os.makedirs(os.path.join(scene, name), exist_ok=True)
            radar_path = os.path.join(scene, "radar_voxel", "000000.npz")
            save_sparse_radar_voxel(radar_path, voxel, statistics)
            for name in ("target_voxel", "lidar_voxel"):
                _save_sparse_voxel(os.path.join(scene, name, "000000.npz"), voxel)
            with open(os.path.join(scene, "preprocess_policy.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "source_scene": "garden",
                        "frames_written": 1,
                        "voxel_coordinate_frame": "lidar",
                        "radar_statistics_protocol": "radar_point_count_doppler_validity_v1",
                    },
                    handle,
                )

            dataset = NTU4DRadLM_VoxelDataset(
                root,
                scene_names=["garden"],
                use_augmentation=False,
                target_size=(1, 2, 1),
                source_pc_range=(0.0, 0.0, 0.0, 2.0, 1.0, 1.0),
                allow_legacy_radar_units=True,
                require_radar_statistics=True,
            )
            sample = dataset[0]
            _, radar_tensor, metadata = sample
            self.assertEqual(tuple(radar_tensor.shape), (4, 1, 2, 1))
            self.assertEqual(
                metadata["radar_statistics"][0]["total_point_count"],
                3,
            )
            self.assertEqual(
                metadata["radar_statistics"][0]["reference"],
                "pre_augmentation_persisted_radar_voxel",
            )
            self.assertIs(
                metadata["radar_statistics"][0]["model_consumed"],
                False,
            )
            batch = collate_voxel_samples([sample, sample])
            self.assertEqual(
                batch[2]["radar_statistics"][1][0]["total_doppler_valid_count"],
                3,
            )


if __name__ == "__main__":
    unittest.main()
