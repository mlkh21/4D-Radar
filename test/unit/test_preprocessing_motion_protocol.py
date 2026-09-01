#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证预处理多普勒运动补偿的显式模式、时间匹配和坐标变换协议。"""

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from NTU4DRadLM_pre_processing.motion_protocol import (
    load_recorded_velocity_table,
    resolve_frame_velocity,
    sensor_to_reference_motion_delta,
    transform_velocity,
)


class PreprocessingMotionProtocolTest(unittest.TestCase):
    def test_only_non_reference_sensor_receives_signed_motion_delta(self):
        radar_timestamp = 10.00
        lidar_timestamp = 10.04
        self.assertAlmostEqual(
            sensor_to_reference_motion_delta(radar_timestamp, lidar_timestamp),
            -0.04,
        )
        self.assertEqual(
            sensor_to_reference_motion_delta(lidar_timestamp, lidar_timestamp),
            0.0,
        )
        self.assertEqual(
            sensor_to_reference_motion_delta(radar_timestamp, radar_timestamp),
            0.0,
        )
        self.assertAlmostEqual(
            sensor_to_reference_motion_delta(lidar_timestamp, radar_timestamp),
            0.04,
        )

    def test_none_mode_never_invents_a_velocity(self):
        self.assertIsNone(
            resolve_frame_velocity(
                mode="none",
                fixed_velocity=(50.0, 0.0, 0.0),
                frame_timestamp=10.0,
                recorded_table=None,
                max_delta=0.02,
            )
        )

    def test_fixed_mode_returns_explicit_finite_vector(self):
        velocity = resolve_frame_velocity(
            mode="fixed",
            fixed_velocity=(3.0, -2.0, 1.0),
            frame_timestamp=10.0,
            recorded_table=None,
            max_delta=0.02,
        )
        np.testing.assert_allclose(velocity, (3.0, -2.0, 1.0))

    def test_fixed_mode_rejects_nonfinite_vector(self):
        with self.assertRaisesRegex(ValueError, "三个有限数"):
            resolve_frame_velocity(
                mode="fixed",
                fixed_velocity=(np.nan, 0.0, 0.0),
                frame_timestamp=10.0,
                recorded_table=None,
                max_delta=0.02,
            )

    def test_recorded_mode_matches_nearest_timestamp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "velocity.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("timestamp,vx,vy,vz\n")
                handle.write("10.000,1.0,2.0,3.0\n")
                handle.write("10.100,4.0,5.0,6.0\n")

            table = load_recorded_velocity_table(path)
            velocity = resolve_frame_velocity(
                mode="recorded",
                fixed_velocity=None,
                frame_timestamp=10.006,
                recorded_table=table,
                max_delta=0.01,
            )
            np.testing.assert_allclose(velocity, (1.0, 2.0, 3.0))

    def test_recorded_mode_rejects_timestamp_gap(self):
        table = np.asarray([[10.0, 1.0, 2.0, 3.0]], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "时间差"):
            resolve_frame_velocity(
                mode="recorded",
                fixed_velocity=None,
                frame_timestamp=10.1,
                recorded_table=table,
                max_delta=0.01,
            )

    def test_velocity_is_transformed_between_radar_and_lidar_frames(self):
        rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        velocity = transform_velocity(
            (1.0, 0.0, 0.0),
            source_frame="radar",
            target_frame="lidar",
            radar_to_lidar_rotation=rotation,
        )
        np.testing.assert_allclose(velocity, (0.0, 1.0, 0.0))

    def test_preprocess_shell_defaults_to_no_implicit_compensation(self):
        script_path = os.path.join(ROOT, "NTU4DRadLM_pre_processing", "preprocess.sh")
        with open(script_path, "r", encoding="utf-8") as handle:
            script = handle.read()
        uses_safe_environment_default = (
            'VELOCITY_MODE="${VELOCITY_MODE:-none}"' in script
            and '--velocity_mode "$VELOCITY_MODE"' in script
        )
        uses_explicit_safe_value = "--velocity_mode none" in script
        self.assertTrue(
            uses_safe_environment_default or uses_explicit_safe_value,
            "预处理脚本必须显式使用 velocity_mode=none 作为安全默认值",
        )

    def test_preprocessing_script_help_works_when_executed_by_file_path(self):
        script_path = os.path.join(
            ROOT,
            "NTU4DRadLM_pre_processing",
            "NTU4DRadLM_pre_processing.py",
        )
        completed = subprocess.run(
            [sys.executable, script_path, "--help"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertIn("--velocity_mode", completed.stdout)
        self.assertIn("--radar_field_schema", completed.stdout)
        self.assertIn("--require_verified_radar_field_schema", completed.stdout)
        self.assertIn("--require_complete_extraction_receipt", completed.stdout)


if __name__ == "__main__":
    unittest.main()
