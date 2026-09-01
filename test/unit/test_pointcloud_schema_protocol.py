# -*- coding: utf-8 -*-
"""验证 PointCloud 固定列布局与 Radar 物理字段 schema 协议。"""

import importlib
import csv
import hashlib
import json
import os
import sys
import tempfile
import types
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
        # 单元测试只验证纯 Python 解包逻辑；无 ROS 环境时提供最小模块替身，
        # 不把 rosbag 安装状态变成字段协议测试的隐式前置条件。
        cls._stubbed_ros_modules = []
        if "rosbag" not in sys.modules:
            try:
                importlib.import_module("rosbag")
            except ImportError:
                rosbag_module = types.ModuleType("rosbag")
                rosbag_module.Bag = None
                sys.modules["rosbag"] = rosbag_module
                cls._stubbed_ros_modules.append("rosbag")
        try:
            importlib.import_module("sensor_msgs.point_cloud2")
        except ImportError:
            sensor_msgs_module = types.ModuleType("sensor_msgs")
            point_cloud2_module = types.ModuleType("sensor_msgs.point_cloud2")
            point_cloud2_module.read_points = None
            sensor_msgs_module.point_cloud2 = point_cloud2_module
            sys.modules["sensor_msgs"] = sensor_msgs_module
            sys.modules["sensor_msgs.point_cloud2"] = point_cloud2_module
            cls._stubbed_ros_modules.extend(
                ["sensor_msgs.point_cloud2", "sensor_msgs"]
            )
        cls.unpack_rosbag = importlib.import_module(
            "NTU4DRadLM_pre_processing.unpack_rosbag"
        )

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("NTU4DRadLM_pre_processing.unpack_rosbag", None)
        for module_name in cls._stubbed_ros_modules:
            sys.modules.pop(module_name, None)

    def test_missing_intensity_keeps_doppler_in_column_four_and_writes_schema(self):
        msg = _PointCloud2(["x", "y", "z", "doppler"])

        def read_points(_msg, field_names, skip_nans):
            self.assertEqual(field_names, ["x", "y", "z", "doppler"])
            self.assertFalse(skip_nans)
            return iter([(1.0, 2.0, 3.0, 7.5), (4.0, 5.0, 6.0, -2.0)])

        with tempfile.TemporaryDirectory() as output_dir:
            with mock.patch.object(self.unpack_rosbag.pc2, "read_points", read_points):
                self.unpack_rosbag.save_pointcloud(msg, output_dir, 1.25)

            points = np.load(os.path.join(output_dir, "1.250000.npy"))
            self.assertEqual(points.shape, (2, 5))
            self.assertTrue(np.isnan(points[:, 3]).all())
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
            self.assertEqual(
                schema["physical_semantics_status"],
                "unverified_layout_only",
            )
            self.assertEqual(schema["missing_value_encoding"], "nan")
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
            self.assertTrue(np.isnan(points[:, 4]).all())

    def test_pointcloud_layout_drift_is_rejected_before_frame_publish(self):
        messages = [
            _PointCloud2(["x", "y", "z", "intensity", "velocity"]),
            _PointCloud2(["x", "y", "z", "intensity", "doppler"]),
        ]

        def read_points(_msg, field_names, skip_nans):
            return iter([(1.0, 2.0, 3.0, 4.0, 5.0)])

        with tempfile.TemporaryDirectory() as output_dir:
            with mock.patch.object(self.unpack_rosbag.pc2, "read_points", read_points):
                self.unpack_rosbag.save_pointcloud(messages[0], output_dir, 1.0)
                with self.assertRaisesRegex(ValueError, "layout 发生漂移"):
                    self.unpack_rosbag.save_pointcloud(messages[1], output_dir, 2.0)

            self.assertTrue(os.path.isfile(os.path.join(output_dir, "1.000000.npy")))
            self.assertFalse(os.path.exists(os.path.join(output_dir, "2.000000.npy")))

    def test_verified_radar_schema_binds_units_direction_and_authority(self):
        from diffusion_consistency_radar.radar_field_schema import (
            load_radar_field_schema_artifact,
            validate_radar_layout_schema,
        )

        schema = {
            "protocol": "radar_raw_field_semantics_v1",
            "storage": {
                "format": "npy",
                "dtype": "float32",
                "column_count": 5,
            },
            "fields": {
                "xyz": {
                    "column_indices": [0, 1, 2],
                    "unit": "m",
                    "coordinate_frame": "radar",
                },
                "return_strength": {
                    "column_index": 3,
                    "source_field": "intensity",
                    "quantity": "intensity",
                    "unit": "sensor_native_linear_nonnegative",
                    "missing_value": "nan",
                },
                "doppler": {
                    "column_index": 4,
                    "source_field": "velocity",
                    "quantity": "radial_velocity",
                    "unit": "m/s",
                    "reference": "sensor_relative",
                    "positive_direction": "toward_sensor",
                    "missing_value": "nan",
                },
            },
            "verification": {
                "status": "verified",
                "authority_type": "official_message_definition",
                "reference": "RadarPointCloud.msg revision 1",
                "evidence_file": "RadarPointCloud.msg",
                "sha256": None,
            },
        }
        with tempfile.TemporaryDirectory() as root:
            evidence = b"float32 intensity\nfloat32 velocity\n"
            evidence_path = os.path.join(root, "RadarPointCloud.msg")
            with open(evidence_path, "wb") as handle:
                handle.write(evidence)
            schema["verification"]["sha256"] = hashlib.sha256(evidence).hexdigest()
            path = os.path.join(root, "radar_field_schema.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(schema, handle)
            loaded, digest = load_radar_field_schema_artifact(
                path,
                require_verified=True,
            )
            layout = {
                "schema_version": 1,
                "storage_format": "npy",
                "columns": ["x", "y", "z", "intensity", "doppler"],
                "column_indices": {
                    "x": 0,
                    "y": 1,
                    "z": 2,
                    "intensity": 3,
                    "doppler": 4,
                },
                "source_fields": ["x", "y", "z", "intensity", "velocity"],
                "selected_fields": ["x", "y", "z", "intensity", "velocity"],
                "field_mapping": {
                    "x": "x",
                    "y": "y",
                    "z": "z",
                    "intensity": "intensity",
                    "doppler": "velocity",
                },
                "missing_fields": [],
                "physical_semantics_status": "unverified_layout_only",
                "missing_value_encoding": "nan",
                "shape": [2, 5],
                "dtype": "float32",
            }
            validate_radar_layout_schema(layout, loaded)
            layout["field_mapping"]["doppler"] = "doppler"
            with self.assertRaisesRegex(ValueError, "source field"):
                validate_radar_layout_schema(layout, loaded)

            with open(evidence_path, "ab") as handle:
                handle.write(b"# tampered\n")
            with self.assertRaisesRegex(ValueError, "权威证据 SHA-256"):
                load_radar_field_schema_artifact(
                    path,
                    require_verified=True,
                )

        self.assertEqual(loaded, schema)
        self.assertEqual(len(digest), 64)

    def test_unverified_radar_schema_cannot_enable_formal_semantics(self):
        from diffusion_consistency_radar.radar_field_schema import (
            validate_radar_field_schema,
        )

        schema = {
            "protocol": "radar_raw_field_semantics_v1",
            "storage": {
                "format": "npy",
                "dtype": "float32",
                "column_count": 5,
            },
            "fields": {
                "xyz": {
                    "column_indices": [0, 1, 2],
                    "unit": "m",
                    "coordinate_frame": "radar",
                },
                "return_strength": {
                    "column_index": 3,
                    "source_field": "intensity",
                    "quantity": "intensity",
                    "unit": "unknown",
                    "missing_value": "nan",
                },
                "doppler": {
                    "column_index": 4,
                    "source_field": "velocity",
                    "quantity": "radial_velocity",
                    "unit": "unknown",
                    "reference": "sensor_relative",
                    "positive_direction": "unknown",
                    "missing_value": "nan",
                },
            },
            "verification": {
                "status": "unverified",
                "authority_type": None,
                "reference": None,
                "evidence_file": None,
                "sha256": None,
            },
        }
        with self.assertRaisesRegex(ValueError, "未通过权威验证"):
            validate_radar_field_schema(schema, require_verified=True)

    def test_doppler_positive_direction_controls_ego_motion_sign(self):
        from NTU4DRadLM_pre_processing.NTU4DRadLM_pre_processing import (
            compensate_radar_doppler,
        )

        points = np.asarray([[10.0, 0.0, 0.0, 1.0, 3.0]], dtype=np.float32)
        velocity = np.asarray([2.0, 0.0, 0.0], dtype=np.float32)

        toward = compensate_radar_doppler(
            points,
            velocity,
            positive_direction="toward_sensor",
        )
        away = compensate_radar_doppler(
            points,
            velocity,
            positive_direction="away_from_sensor",
        )

        self.assertAlmostEqual(float(toward[0, 4]), 1.0)
        self.assertAlmostEqual(float(away[0, 4]), 5.0)

    def test_motion_compensation_requires_verified_field_schema(self):
        from NTU4DRadLM_pre_processing.NTU4DRadLM_pre_processing import (
            resolve_radar_field_schema,
        )

        self.assertEqual(
            resolve_radar_field_schema(
                "",
                require_verified=False,
                velocity_mode="none",
            ),
            (None, None),
        )
        with self.assertRaisesRegex(ValueError, "运动补偿.*verified"):
            resolve_radar_field_schema(
                "",
                require_verified=False,
                velocity_mode="fixed",
            )

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

            receipt_path = os.path.join(
                output_dir,
                "garden",
                "extraction_receipt.json",
            )
            with open(receipt_path, encoding="utf-8") as handle:
                receipt = json.load(handle)
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["failures"][0]["topic"], "__bag__")
            self.assertIs(receipt["failures"][0]["critical"], True)

    def test_critical_frame_save_failure_is_receipted_and_stops_scene(self):
        """Radar/LiDAR/IR 任一帧保存失败都不得被吞掉。"""
        class _BagTime:
            @staticmethod
            def to_sec():
                return 1.25

        class _Bag:
            def get_type_and_topic_info(self):
                return types.SimpleNamespace(
                    topics={
                        "/radar_pcl": types.SimpleNamespace(
                            msg_type="sensor_msgs/PointCloud2"
                        )
                    }
                )

            def read_messages(self):
                return iter(
                    [
                        (
                            "/radar_pcl",
                            _PointCloud2(["x", "y", "z", "intensity", "velocity"]),
                            _BagTime(),
                        )
                    ]
                )

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
            bag_path = os.path.join(input_dir, "garden_2022-05-13_0.bag")
            open(bag_path, "wb").close()
            with mock.patch.object(self.unpack_rosbag.rosbag, "Bag", return_value=_Bag()), mock.patch.object(
                self.unpack_rosbag,
                "save_pointcloud",
                side_effect=OSError("disk write failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "关键模态帧解包失败"):
                    self.unpack_rosbag.process_ntu_dataset(input_dir, output_dir)

            with open(
                os.path.join(output_dir, "garden", "extraction_receipt.json"),
                encoding="utf-8",
            ) as handle:
                receipt = json.load(handle)
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["failures"][0]["topic"], "radar_pcl")
            self.assertEqual(receipt["failures"][0]["timestamp"], 1.25)
            self.assertEqual(receipt["failures"][0]["error_type"], "OSError")
            self.assertEqual(
                receipt["critical_modalities"]["radar_pcl"]["status"],
                "failed",
            )

    def test_complete_extraction_receipt_requires_all_critical_modalities(self):
        from diffusion_consistency_radar.extraction_receipt import (
            CRITICAL_EXTRACTION_TOPICS,
            finalize_extraction_receipt,
            load_extraction_receipt_artifact,
            mark_bag_processed,
            new_extraction_receipt,
            record_extraction_success,
            write_extraction_receipt_atomic,
        )

        receipt = new_extraction_receipt("garden", ["garden_0.bag"])
        for topic in CRITICAL_EXTRACTION_TOPICS:
            record_extraction_success(receipt, topic)
        mark_bag_processed(receipt, "garden_0.bag")
        self.assertTrue(finalize_extraction_receipt(receipt))

        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "extraction_receipt.json")
            write_extraction_receipt_atomic(path, receipt)
            loaded, digest = load_extraction_receipt_artifact(
                path,
                require_complete=True,
            )

        self.assertEqual(loaded["status"], "complete")
        self.assertEqual(len(digest), 64)

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

    def test_formal_v3_shell_is_fresh_and_fail_closed(self):
        """新入口必须隔离 v2 输出，并显式启用 schema/receipt/v3 门禁。"""
        script_path = os.path.join(
            ROOT,
            "NTU4DRadLM_pre_processing",
            "preprocess-v3.sh",
        )
        with open(script_path, encoding="utf-8") as handle:
            script = handle.read()

        self.assertIn("NTU4DRadLM_Raw_formal_v3", script)
        self.assertIn("NTU4DRadLM_Pre_formal_v3", script)
        self.assertNotIn("NTU4DRadLM_Pre_formal_v2_80m_86p8_v1", script)
        schema_gate = script.index("require_verified=True")
        unpack_step = script.index("unpack_rosbag.py")
        self.assertLess(schema_gate, unpack_step)
        self.assertIn("--require_verified_radar_field_schema", script)
        self.assertIn("--require_complete_extraction_receipt", script)
        self.assertIn("--protocol_version v3", script)
        self.assertIn('if [[ -e "$output" ]]', script)


if __name__ == "__main__":
    unittest.main()
