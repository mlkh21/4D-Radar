# -*- coding: utf-8 -*-
"""测试 P1-01 的 header 时间戳优先、最近邻阈值和对齐记录协议。"""

import csv
import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class TimestampAlignmentProtocolTest(unittest.TestCase):
    class _Stamp:
        def __init__(self, value):
            self.value = float(value)

        def to_sec(self):
            return self.value

    class _Header:
        def __init__(self, stamp):
            self.stamp = stamp

    class _Message:
        def __init__(self, stamp):
            self.header = TimestampAlignmentProtocolTest._Header(stamp)

    def test_header_timestamp_is_preferred_over_bag_receipt_time(self):
        from NTU4DRadLM_pre_processing.timestamp_alignment import (
            preferred_message_timestamp,
        )

        timestamp, source = preferred_message_timestamp(
            self._Message(self._Stamp(10.125)),
            receipt_timestamp=10.250,
        )

        self.assertAlmostEqual(timestamp, 10.125, places=6)
        self.assertEqual(source, "header")

    def test_invalid_header_timestamp_falls_back_to_receipt_time(self):
        from NTU4DRadLM_pre_processing.timestamp_alignment import (
            preferred_message_timestamp,
        )

        timestamp, source = preferred_message_timestamp(
            self._Message(self._Stamp(0.0)),
            receipt_timestamp=10.250,
        )

        self.assertAlmostEqual(timestamp, 10.250, places=6)
        self.assertEqual(source, "receipt")

    def test_nearest_match_reports_delta_and_rejects_out_of_threshold(self):
        from NTU4DRadLM_pre_processing.timestamp_alignment import (
            nearest_timestamp_match,
        )

        index, delta = nearest_timestamp_match(
            [1.00, 1.04, 1.10],
            target=1.035,
            max_delta=0.01,
        )
        self.assertEqual(index, 1)
        self.assertAlmostEqual(delta, 0.005, places=6)

        with self.assertRaisesRegex(ValueError, "超过时间容差"):
            nearest_timestamp_match(
                [1.00, 1.04, 1.10],
                target=1.075,
                max_delta=0.01,
            )

    def test_timestamp_index_writes_pair_delta_without_partial_mismatch(self):
        from NTU4DRadLM_pre_processing.NTU4DRadLM_timestamp_index import (
            generate_scene_indices,
        )

        with tempfile.TemporaryDirectory() as tmp:
            scene = os.path.join(tmp, "garden")
            radar_dir = os.path.join(scene, "radar_pcl")
            lidar_dir = os.path.join(scene, "livox_lidar")
            os.makedirs(radar_dir)
            os.makedirs(lidar_dir)
            for name in ("1.000000.npy", "1.100000.npy"):
                open(os.path.join(radar_dir, name), "wb").close()
            for name in ("1.010000.npy", "1.090000.npy"):
                open(os.path.join(lidar_dir, name), "wb").close()

            records = generate_scene_indices(scene, radar_lidar_max_delta=0.02)

            self.assertEqual(len(records), 2)
            with open(os.path.join(scene, "radar_index_sequence.txt"), encoding="utf-8") as handle:
                self.assertEqual(handle.read().splitlines(), ["0", "1"])
            with open(os.path.join(scene, "lidar_index_sequence.txt"), encoding="utf-8") as handle:
                self.assertEqual(handle.read().splitlines(), ["0", "1"])
            with open(os.path.join(scene, "radar_lidar_sync.csv"), newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(float(rows[0]["delta_seconds"]), 0.01, places=6)

    def test_timestamp_index_fails_before_writing_when_delta_exceeds_threshold(self):
        from NTU4DRadLM_pre_processing.NTU4DRadLM_timestamp_index import (
            generate_scene_indices,
        )

        with tempfile.TemporaryDirectory() as tmp:
            scene = os.path.join(tmp, "garden")
            radar_dir = os.path.join(scene, "radar_pcl")
            lidar_dir = os.path.join(scene, "livox_lidar")
            os.makedirs(radar_dir)
            os.makedirs(lidar_dir)
            open(os.path.join(radar_dir, "1.000000.npy"), "wb").close()
            open(os.path.join(lidar_dir, "1.100000.npy"), "wb").close()

            with self.assertRaisesRegex(ValueError, "超过时间容差"):
                generate_scene_indices(scene, radar_lidar_max_delta=0.02)

            self.assertFalse(os.path.exists(os.path.join(scene, "radar_index_sequence.txt")))
            self.assertFalse(os.path.exists(os.path.join(scene, "radar_lidar_sync.csv")))

    def test_timestamp_index_records_and_skips_sparse_unmatched_candidates(self):
        """异步采样中的少量掉帧候选必须显式记录，不能强行配对。"""
        from NTU4DRadLM_pre_processing.NTU4DRadLM_timestamp_index import (
            generate_scene_indices,
        )

        with tempfile.TemporaryDirectory() as tmp:
            scene = os.path.join(tmp, "garden")
            radar_dir = os.path.join(scene, "radar_pcl")
            lidar_dir = os.path.join(scene, "livox_lidar")
            os.makedirs(radar_dir)
            os.makedirs(lidar_dir)
            for name in ("1.000000.npy", "1.040000.npy", "1.080000.npy", "1.120000.npy"):
                open(os.path.join(radar_dir, name), "wb").close()
            for name in ("1.010000.npy", "1.090000.npy", "1.300000.npy"):
                open(os.path.join(lidar_dir, name), "wb").close()

            records = generate_scene_indices(
                scene,
                radar_lidar_max_delta=0.02,
                skip_unmatched=True,
                max_rejected_fraction=0.5,
            )

            self.assertEqual(len(records), 2)
            with open(os.path.join(scene, "radar_index_sequence.txt"), encoding="utf-8") as handle:
                self.assertEqual(handle.read().splitlines(), ["0", "2"])
            with open(os.path.join(scene, "lidar_index_sequence.txt"), encoding="utf-8") as handle:
                self.assertEqual(handle.read().splitlines(), ["0", "1"])
            with open(
                os.path.join(scene, "radar_lidar_rejected.csv"),
                newline="",
                encoding="utf-8",
            ) as handle:
                rejected = list(csv.DictReader(handle))
            self.assertEqual(len(rejected), 1)
            self.assertEqual(rejected[0]["candidate_index"], "2")
            self.assertEqual(rejected[0]["reason"], "exceeds_max_delta")
            self.assertAlmostEqual(float(rejected[0]["delta_seconds"]), 0.18, places=6)

    def test_skip_unmatched_rejects_excessive_fraction_before_writing(self):
        """阈值错误导致大量候选被丢弃时，仍须在发布任何索引前失败。"""
        from NTU4DRadLM_pre_processing.NTU4DRadLM_timestamp_index import (
            generate_scene_indices,
        )

        with tempfile.TemporaryDirectory() as tmp:
            scene = os.path.join(tmp, "garden")
            radar_dir = os.path.join(scene, "radar_pcl")
            lidar_dir = os.path.join(scene, "livox_lidar")
            os.makedirs(radar_dir)
            os.makedirs(lidar_dir)
            for name in ("1.000000.npy", "1.040000.npy", "1.080000.npy", "1.120000.npy"):
                open(os.path.join(radar_dir, name), "wb").close()
            for name in ("1.010000.npy", "1.090000.npy", "1.300000.npy"):
                open(os.path.join(lidar_dir, name), "wb").close()

            with self.assertRaisesRegex(ValueError, "拒绝比例"):
                generate_scene_indices(
                    scene,
                    radar_lidar_max_delta=0.02,
                    skip_unmatched=True,
                    max_rejected_fraction=0.2,
                )

            for filename in (
                "radar_index_sequence.txt",
                "lidar_index_sequence.txt",
                "radar_lidar_sync.csv",
                "radar_lidar_rejected.csv",
            ):
                self.assertFalse(os.path.exists(os.path.join(scene, filename)))

    def test_full_rebuild_script_reextracts_header_timestamps_before_indexing(self):
        """正式重建脚本必须从 bag 生成独立 Raw 候选，不能复用 receipt-time 旧目录。"""
        script_path = os.path.join(
            ROOT,
            "NTU4DRadLM_pre_processing",
            "preprocess.sh",
        )
        with open(script_path, encoding="utf-8") as handle:
            script = handle.read()

        unpack_call = script.index("unpack_rosbag.py")
        index_call = script.index("NTU4DRadLM_timestamp_index.py")
        self.assertLess(unpack_call, index_call)
        self.assertIn("NTU4DRadLM_Raw_p1_01_candidate", script)
        self.assertIn("--skip_unmatched", script)
        self.assertIn("--max_rejected_fraction 0.01", script)
        self.assertIn("--radar_lidar_max_delta 0.045", script)
        self.assertIn("--radar_ir_max_delta 0.025", script)
        self.assertIn('--raw_data_path "$RAW_ROOT"', script)


if __name__ == "__main__":
    unittest.main()
