import unittest
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from NTU4DRadLM_pre_processing.sensor_aware_target import (
    SensorAwareTargetPolicy,
    build_dataset_targets,
    build_scene_targets,
    build_sensor_aware_target,
)


class SensorAwareTargetTest(unittest.TestCase):
    def test_filters_lidar_target_by_height_and_range(self):
        lidar = np.zeros((6, 4, 5, 4), dtype=np.float32)
        radar = np.zeros_like(lidar)
        lidar[1, 1, 0, 0] = 1.0
        lidar[1, 1, 0, 1] = 0.5
        lidar[2, 1, 2, 0] = 1.0
        lidar[2, 1, 2, 1] = 0.8
        lidar[5, 1, 2, 0] = 1.0

        policy = SensorAwareTargetPolicy(
            pc_range=(0.0, -2.0, -2.0, 6.0, 2.0, 3.0),
            z_min=0.0,
            x_max=4.0,
        )

        target = build_sensor_aware_target(lidar, radar, policy)

        self.assertEqual(target[1, 1, 0, 0], 0.0)
        self.assertEqual(target[5, 1, 2, 0], 0.0)
        self.assertEqual(target[2, 1, 2, 0], 1.0)
        self.assertAlmostEqual(float(target[2, 1, 2, 1]), 0.8, places=6)

    def test_limits_lidar_target_to_radar_visible_neighborhood(self):
        lidar = np.zeros((5, 5, 3, 4), dtype=np.float32)
        radar = np.zeros_like(lidar)
        lidar[2, 2, 1, 0] = 1.0
        lidar[4, 4, 1, 0] = 1.0
        radar[1, 2, 1, 0] = 1.0
        radar[1, 2, 1, 2] = 3.0

        policy = SensorAwareTargetPolicy(
            pc_range=(0.0, -2.5, -1.5, 5.0, 2.5, 1.5),
            require_radar_visibility=True,
            radar_visibility_radius=1,
        )

        target = build_sensor_aware_target(lidar, radar, policy)

        self.assertEqual(target[2, 2, 1, 0], 1.0)
        self.assertEqual(target[4, 4, 1, 0], 0.0)
        self.assertEqual(target[2, 2, 1, 2], 3.0)
        self.assertEqual(target[2, 2, 1, 3], 1.0)

    def test_builds_scene_target_directory_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "loop3")
            radar_dir = os.path.join(scene_dir, "radar_voxel")
            lidar_dir = os.path.join(scene_dir, "lidar_voxel")
            out_dir = os.path.join(tmp, "loop3_sensor")
            os.makedirs(radar_dir)
            os.makedirs(lidar_dir)

            lidar = np.zeros((4, 3, 3, 4), dtype=np.float32)
            radar = np.zeros_like(lidar)
            lidar[1, 1, 1, 0] = 1.0
            radar[1, 1, 1, 0] = 1.0
            radar[1, 1, 1, 2] = 2.0
            np.save(os.path.join(lidar_dir, "000000.npy"), lidar)
            np.save(os.path.join(radar_dir, "000000.npy"), radar)

            policy = SensorAwareTargetPolicy(
                pc_range=(0.0, -1.5, -1.5, 4.0, 1.5, 1.5),
                require_radar_visibility=True,
            )

            written = build_scene_targets(scene_dir, out_dir, policy)

            target = np.load(os.path.join(out_dir, "target_voxel", "000000.npy"))
            self.assertEqual(written, 1)
            self.assertEqual(target[1, 1, 1, 0], 1.0)
            self.assertTrue(os.path.exists(os.path.join(out_dir, "radar_voxel", "000000.npy")))
            self.assertTrue(os.path.exists(os.path.join(out_dir, "target_policy.json")))

    def test_builds_dataset_root_for_selected_scenes(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_root = os.path.join(tmp, "pre")
            output_root = os.path.join(tmp, "sensor")
            scene_dir = os.path.join(input_root, "garden")
            radar_dir = os.path.join(scene_dir, "radar_voxel")
            lidar_dir = os.path.join(scene_dir, "lidar_voxel")
            os.makedirs(radar_dir)
            os.makedirs(lidar_dir)

            lidar = np.zeros((3, 3, 3, 4), dtype=np.float32)
            radar = np.zeros_like(lidar)
            lidar[1, 1, 1, 0] = 1.0
            radar[1, 1, 1, 0] = 1.0
            np.save(os.path.join(lidar_dir, "000000.npy"), lidar)
            np.save(os.path.join(radar_dir, "000000.npy"), radar)

            policy = SensorAwareTargetPolicy(pc_range=(0.0, -1.5, -1.5, 3.0, 1.5, 1.5))

            counts = build_dataset_targets(input_root, output_root, policy, scenes=["garden"])

            self.assertEqual(counts, {"garden": 1})
            self.assertTrue(os.path.exists(os.path.join(output_root, "garden", "target_voxel", "000000.npy")))

    def test_limits_scene_generation_by_max_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "loop3")
            radar_dir = os.path.join(scene_dir, "radar_voxel")
            lidar_dir = os.path.join(scene_dir, "lidar_voxel")
            out_dir = os.path.join(tmp, "loop3_sensor")
            os.makedirs(radar_dir)
            os.makedirs(lidar_dir)

            lidar = np.zeros((3, 3, 3, 4), dtype=np.float32)
            radar = np.zeros_like(lidar)
            for frame_id in ("000000", "000001"):
                np.save(os.path.join(lidar_dir, f"{frame_id}.npy"), lidar)
                np.save(os.path.join(radar_dir, f"{frame_id}.npy"), radar)

            policy = SensorAwareTargetPolicy(pc_range=(0.0, -1.5, -1.5, 3.0, 1.5, 1.5))

            written = build_scene_targets(scene_dir, out_dir, policy, max_files=1)

            self.assertEqual(written, 1)
            self.assertTrue(os.path.exists(os.path.join(out_dir, "target_voxel", "000000.npy")))
            self.assertFalse(os.path.exists(os.path.join(out_dir, "target_voxel", "000001.npy")))


if __name__ == "__main__":
    unittest.main()
