# -*- coding: utf-8 -*-
"""验证 PointCloud2 固定五列输出与字段 schema 元数据协议。"""

import importlib
import csv
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class _Field:
    def __init__(self, name):
        self.name = name


class _PointCloud2:
    width = 2

    def __init__(self, fields):
        self.fields = [_Field(name) for name in fields]


class PointCloudSchemaProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 解包正式入口不得依赖未参与计算的 pandas/open3d；直接按真实环境导入。
        cls.unpack_rosbag = importlib.import_module(
            "NTU4DRadLM_pre_processing.unpack_rosbag"
        )

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("NTU4DRadLM_pre_processing.unpack_rosbag", None)

    def test_missing_intensity_keeps_doppler_in_column_four_and_writes_schema(self):
        msg = _PointCloud2(["x", "y", "z", "doppler"])

        def read_points(_msg, field_names, skip_nans):
            self.assertEqual(field_names, ["x", "y", "z", "doppler"])
            self.assertTrue(skip_nans)
            return iter([(1.0, 2.0, 3.0, 7.5), (4.0, 5.0, 6.0, -2.0)])

        with tempfile.TemporaryDirectory() as output_dir:
            with mock.patch.object(self.unpack_rosbag.pc2, "read_points", read_points):
                self.unpack_rosbag.save_pointcloud(msg, output_dir, 1.25)

            points = np.load(os.path.join(output_dir, "1.250000.npy"))
            self.assertEqual(points.shape, (2, 5))
            np.testing.assert_allclose(points[:, 3], 0.0)
            np.testing.assert_allclose(points[:, 4], [7.5, -2.0])

            with open(
                os.path.join(output_dir, "pointcloud_schema.json"),
                encoding="utf-8",
            ) as handle:
                schema = json.load(handle)
            self.assertEqual(
                schema["columns"], ["x", "y", "z", "intensity", "doppler"]
            )
            self.assertEqual(schema["field_mapping"]["doppler"], "doppler")
            self.assertIsNone(schema["field_mapping"]["intensity"])
            self.assertEqual(schema["missing_fields"], ["intensity"])
            self.assertEqual(schema["shape"], [2, 5])

    def test_missing_doppler_keeps_intensity_in_column_three(self):
        msg = _PointCloud2(["x", "y", "z", "intensity"])

        def read_points(_msg, field_names, skip_nans):
            self.assertEqual(field_names, ["x", "y", "z", "intensity"])
            return iter([(1.0, 2.0, 3.0, 11.0), (4.0, 5.0, 6.0, 12.0)])

        with tempfile.TemporaryDirectory() as output_dir:
            with mock.patch.object(self.unpack_rosbag.pc2, "read_points", read_points):
                self.unpack_rosbag.save_pointcloud(msg, output_dir, 2.5)

            points = np.load(os.path.join(output_dir, "2.500000.npy"))
            self.assertEqual(points.shape, (2, 5))
            np.testing.assert_allclose(points[:, 3], [11.0, 12.0])
            np.testing.assert_allclose(points[:, 4], 0.0)

    def test_bag_open_failure_is_not_reported_as_success(self):
        """任一分卷无法打开时必须终止，不能继续产出不完整场景。"""
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
            bag_path = os.path.join(input_dir, "garden_2022-05-13_0.bag")
            open(bag_path, "wb").close()

            with mock.patch.object(
                self.unpack_rosbag.rosbag,
                "Bag",
                side_effect=OSError("broken bag"),
            ):
                with self.assertRaisesRegex(RuntimeError, "无法打开 rosbag 分卷"):
                    self.unpack_rosbag.process_ntu_dataset(input_dir, output_dir)

    def test_non_pointcloud_csv_uses_standard_library_and_keeps_late_fields(self):
        """替代 pandas 后仍需保留后续消息才出现的字段。"""
        rows = [
            {"timestamp": 1.0, "timestamp_source": "header", "seq": 1},
            {
                "timestamp": 2.0,
                "timestamp_source": "header",
                "seq": 2,
                "velocity_x": 3.5,
            },
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            csv_path = os.path.join(output_dir, "records.csv")
            self.unpack_rosbag._write_csv_records(csv_path, rows)
            with open(csv_path, encoding="utf-8", newline="") as handle:
                loaded = list(csv.DictReader(handle))

        self.assertEqual(list(loaded[0]), ["timestamp", "timestamp_source", "seq", "velocity_x"])
        self.assertEqual(loaded[0]["velocity_x"], "")
        self.assertEqual(loaded[1]["velocity_x"], "3.5")


if __name__ == "__main__":
    unittest.main()
