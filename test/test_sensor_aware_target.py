import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from NTU4DRadLM_pre_processing.NTU4DRadLM_pre_processing import (
    build_sensor_aware_target_vectorized,
)


PC_RANGE = (0.0, -2.5, -1.5, 5.0, 2.5, 1.5)


def build_target(lidar, radar, visibility_mode="preserve", z_min=None, x_max=None):
    return build_sensor_aware_target_vectorized(
        lidar_voxel=lidar,
        radar_voxel=radar,
        pc_range=PC_RANGE,
        z_min=z_min,
        x_max=x_max,
        require_radar_visibility=False,
        radar_visibility_radius=1,
        doppler_radius=1,
        visibility_mode=visibility_mode,
    )


class SensorAwareTargetTest(unittest.TestCase):
    def test_oneclick_preprocess_defaults_to_structure_preserving_visibility(self):
        script_path = os.path.join(ROOT, "NTU4DRadLM_pre_processing", "preprocess.sh")
        with open(script_path, "r", encoding="utf-8") as handle:
            script = handle.read()

        self.assertIn('VISIBILITY_MODE="${VISIBILITY_MODE:-preserve}"', script)
        self.assertIn('--visibility_mode "$VISIBILITY_MODE"', script)
        self.assertNotIn('--require_radar_visibility\n', script)

    def test_preserve_mode_keeps_structure_outside_radar_neighborhood(self):
        lidar = np.zeros((5, 5, 3, 4), dtype=np.float32)
        radar = np.zeros_like(lidar)
        lidar[2, 2, 1, 0] = 1.0
        lidar[4, 4, 1, 0] = 1.0
        radar[1, 2, 1, 0] = 1.0
        radar[1, 2, 1, 2] = 3.0

        target = build_target(lidar, radar, visibility_mode="preserve")

        self.assertEqual(target[2, 2, 1, 0], 1.0)
        self.assertEqual(target[4, 4, 1, 0], 1.0)
        self.assertEqual(target[2, 2, 1, 2], 3.0)
        self.assertEqual(target[2, 2, 1, 3], 1.0)
        self.assertEqual(target[4, 4, 1, 3], 0.0)

    def test_hard_mode_limits_target_to_radar_neighborhood(self):
        lidar = np.zeros((5, 5, 3, 4), dtype=np.float32)
        radar = np.zeros_like(lidar)
        lidar[2, 2, 1, 0] = 1.0
        lidar[4, 4, 1, 0] = 1.0
        radar[1, 2, 1, 0] = 1.0

        target = build_target(lidar, radar, visibility_mode="hard")

        self.assertEqual(target[2, 2, 1, 0], 1.0)
        self.assertEqual(target[4, 4, 1, 0], 0.0)

    def test_height_and_range_filters_still_apply_in_preserve_mode(self):
        lidar = np.zeros((5, 5, 3, 4), dtype=np.float32)
        radar = np.zeros_like(lidar)
        lidar[1, 1, 0, 0] = 1.0
        lidar[2, 1, 2, 0] = 1.0
        lidar[4, 1, 2, 0] = 1.0

        target = build_target(
            lidar,
            radar,
            visibility_mode="preserve",
            z_min=0.0,
            x_max=3.0,
        )

        self.assertEqual(target[1, 1, 0, 0], 0.0)
        self.assertEqual(target[2, 1, 2, 0], 1.0)
        self.assertEqual(target[4, 1, 2, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
