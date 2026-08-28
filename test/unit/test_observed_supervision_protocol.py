# -*- coding: utf-8 -*-
"""文件功能：验证 LiDAR 射线 observed mask 的持久化与正式 Dataset 门禁。"""

import os
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion_consistency_radar.cm.dataset_loader import NTU4DRadLM_VoxelDataset
from diffusion_consistency_radar.observed_mask import (
    build_lidar_observed_mask,
    load_observed_mask,
    save_observed_mask,
)


def _save_sparse_voxel(path, voxel):
    occupied = voxel[..., 0] > 0
    np.savez(
        path,
        coords=np.column_stack(np.where(occupied)),
        features=voxel[occupied],
        shape=voxel.shape,
    )


class ObservedMaskProtocolTest(unittest.TestCase):
    def test_nearest_collinear_endpoint_blocks_cells_behind_obstacle(self):
        lidar = np.zeros((6, 3, 2, 4), dtype=np.float32)
        lidar[1, 1, 1, 0] = 1.0
        lidar[4, 1, 1, 0] = 1.0

        mask = build_lidar_observed_mask(
            lidar,
            (0.0, -1.5, -1.0, 6.0, 1.5, 1.0),
        )

        self.assertTrue(mask[1, 1, 1])
        self.assertTrue(mask[4, 1, 1])
        self.assertFalse(mask[5, 1, 1])

    def test_sparse_roundtrip_binds_shape_range_and_protocol(self):
        mask = np.zeros((4, 3, 2), dtype=bool)
        mask[0, 1, 1] = True
        mask[3, 2, 0] = True
        pc_range = (0.0, -1.5, -1.0, 4.0, 1.5, 1.0)

        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "000000.npz")
            save_observed_mask(path, mask, pc_range)
            loaded = load_observed_mask(
                path,
                expected_shape=mask.shape,
                expected_pc_range=pc_range,
            )
            np.testing.assert_array_equal(loaded, mask)
            with self.assertRaisesRegex(ValueError, "pc_range"):
                load_observed_mask(
                    path,
                    expected_shape=mask.shape,
                    expected_pc_range=(0.0, -1.5, -1.0, 5.0, 1.5, 1.0),
                )

    def test_formal_dataset_rejects_missing_persisted_mask(self):
        with tempfile.TemporaryDirectory() as root:
            scene_root = os.path.join(root, "garden")
            for name in ("radar_voxel", "lidar_voxel", "target_voxel", "ir_image"):
                os.makedirs(os.path.join(scene_root, name), exist_ok=True)
            voxel = np.zeros((4, 3, 2, 4), dtype=np.float32)
            voxel[1, 1, 1, 0] = 1.0
            for name in ("radar_voxel", "lidar_voxel", "target_voxel"):
                _save_sparse_voxel(os.path.join(scene_root, name, "000000.npz"), voxel)

            with self.assertRaisesRegex(FileNotFoundError, "observed"):
                NTU4DRadLM_VoxelDataset(
                    root,
                    scene_names=["garden"],
                    use_augmentation=False,
                    target_size=(2, 4, 3),
                    source_pc_range=(0.0, -1.5, -1.0, 4.0, 1.5, 1.0),
                    allow_legacy_radar_units=True,
                    require_persisted_observed_mask=True,
                )

    def test_formal_dataset_loads_persisted_mask(self):
        with tempfile.TemporaryDirectory() as root:
            scene_root = os.path.join(root, "garden")
            for name in (
                "radar_voxel",
                "lidar_voxel",
                "target_voxel",
                "observed_mask",
                "ir_image",
            ):
                os.makedirs(os.path.join(scene_root, name), exist_ok=True)
            voxel = np.zeros((4, 3, 2, 4), dtype=np.float32)
            voxel[1, 1, 1, 0] = 1.0
            for name in ("radar_voxel", "lidar_voxel", "target_voxel"):
                _save_sparse_voxel(os.path.join(scene_root, name, "000000.npz"), voxel)
            pc_range = (0.0, -1.5, -1.0, 4.0, 1.5, 1.0)
            mask = build_lidar_observed_mask(voxel, pc_range)
            save_observed_mask(
                os.path.join(scene_root, "observed_mask", "000000.npz"),
                mask,
                pc_range,
            )

            dataset = NTU4DRadLM_VoxelDataset(
                root,
                scene_names=["garden"],
                use_augmentation=False,
                target_size=(2, 4, 3),
                source_pc_range=pc_range,
                allow_legacy_radar_units=True,
                require_persisted_observed_mask=True,
            )
            _, _, metadata = dataset[0]
            self.assertEqual(
                metadata["occupancy_observed_mask_source"],
                "persisted_lidar_ray_v1",
            )


if __name__ == "__main__":
    unittest.main()
