# -*- coding: utf-8 -*-
"""文件功能：验证 LiDAR→body 外参组合与 body→local 双假设轨迹候选诊断。"""

import csv
import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "test"
    / "diagnostics"
    / "alignment"
    / "build_mapping_pose_candidates.py"
)
SPEC = importlib.util.spec_from_file_location("build_mapping_pose_candidates", SCRIPT_PATH)
mapping_pose_candidates = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mapping_pose_candidates)


def _write_inputs(root: str, *, ground_truth_times=(1.0, 3.0), radar_times=(0.0, 2.0, 4.0)):
    """写入可手算的外参、GT 和 Radar 时间合成契约。"""
    radar_to_imu = os.path.join(root, "radar_to_imu.txt")
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = 1.0
    np.savetxt(radar_to_imu, matrix, fmt="%.9f")

    radar_to_lidar = os.path.join(root, "radar_to_lidar.txt")
    with open(radar_to_lidar, "w", encoding="utf-8") as handle:
        handle.write("R: 1 0 0 0 1 0 0 0 1\n")
        handle.write("T: 0.25 0 0\n")

    gt_odom = os.path.join(root, "gt_odom.txt")
    with open(gt_odom, "w", encoding="utf-8") as handle:
        handle.write("# timestamp tx ty tz qx qy qz qw\n")
        handle.write(
            f"{ground_truth_times[0]:.9f} 0 0 0 0 0 0 1\n"
        )
        handle.write(
            f"{ground_truth_times[1]:.9f} 2 0 0 0 0 1 0\n"
        )

    sync_csv = os.path.join(root, "radar_ir_sync.csv")
    with open(sync_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("frame_index", "radar_timestamp", "ir_timestamp"),
        )
        writer.writeheader()
        for index, timestamp in enumerate(radar_times):
            writer.writerow(
                {
                    "frame_index": index,
                    "radar_timestamp": f"{timestamp:.9f}",
                    "ir_timestamp": f"{timestamp:.9f}",
                }
            )
    return radar_to_imu, radar_to_lidar, gt_odom, sync_csv


def _read_pose_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_radar_lidar_sync(root, radar_times, lidar_times):
    """写入显式 Radar/LiDAR 时间收据，供 LiDAR-reference 候选使用。"""
    path = os.path.join(root, "radar_lidar_sync.csv")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "pair_index",
                "radar_index",
                "lidar_index",
                "radar_timestamp",
                "lidar_timestamp",
                "delta_seconds",
                "signed_delta_seconds",
            ),
        )
        writer.writeheader()
        for index, (radar_time, lidar_time) in enumerate(zip(radar_times, lidar_times)):
            signed_delta = float(lidar_time) - float(radar_time)
            writer.writerow(
                {
                    "pair_index": index,
                    "radar_index": index,
                    "lidar_index": index,
                    "radar_timestamp": f"{radar_time:.9f}",
                    "lidar_timestamp": f"{lidar_time:.9f}",
                    "delta_seconds": f"{abs(signed_delta):.9f}",
                    "signed_delta_seconds": f"{signed_delta:.9f}",
                }
            )
    return path


class MappingPoseCandidatesTest(unittest.TestCase):
    def test_lidar_reference_rejects_mismatched_radar_timestamp_before_output(self):
        """独立 sync 必须与 frame 收据的 Radar 时间一致，不能按行号盲拼。"""
        with tempfile.TemporaryDirectory() as root:
            inputs = _write_inputs(root, radar_times=(0.0, 2.0, 4.0))
            radar_lidar_sync = _write_radar_lidar_sync(
                root,
                radar_times=(0.1, 2.0, 4.0),
                lidar_times=(0.0, 2.1, 4.1),
            )
            output_dir = os.path.join(root, "output")
            with self.assertRaisesRegex(ValueError, "Radar.*时间不一致"):
                mapping_pose_candidates.build_mapping_pose_candidates(
                    radar_to_imu_matrix_path=inputs[0],
                    radar_to_lidar_path=inputs[1],
                    ground_truth_path=inputs[2],
                    radar_sync_csv_path=inputs[3],
                    radar_lidar_sync_csv_path=radar_lidar_sync,
                    pose_reference_sensor="lidar",
                    output_dir=output_dir,
                    max_interpolation_gap_s=3.0,
                )
            self.assertFalse(os.path.exists(output_dir))

    def test_lidar_reference_uses_lidar_timestamp_and_snapshots_sync(self):
        """LiDAR 对齐体素的 pose 必须按 LiDAR 时间插值并绑定同步收据。"""
        with tempfile.TemporaryDirectory() as root:
            inputs = _write_inputs(
                root,
                ground_truth_times=(1.0, 3.0),
                radar_times=(0.0, 2.0, 4.0),
            )
            radar_lidar_sync = _write_radar_lidar_sync(
                root,
                radar_times=(0.0, 2.0, 4.0),
                lidar_times=(-0.1, 2.5, 4.1),
            )
            output_dir = os.path.join(root, "output")
            report = mapping_pose_candidates.build_mapping_pose_candidates(
                radar_to_imu_matrix_path=inputs[0],
                radar_to_lidar_path=inputs[1],
                ground_truth_path=inputs[2],
                radar_sync_csv_path=inputs[3],
                radar_lidar_sync_csv_path=radar_lidar_sync,
                pose_reference_sensor="lidar",
                output_dir=output_dir,
                max_interpolation_gap_s=3.0,
            )

            self.assertEqual(report["protocol"], "mapping_pose_candidate_diagnostic_v2")
            self.assertEqual(report["timing"]["pose_reference_sensor"], "lidar")
            snapshot = report["inputs"]["radar_lidar_sync_snapshot"]
            snapshot_path = os.path.join(output_dir, snapshot["file"])
            self.assertTrue(os.path.isfile(snapshot_path))
            self.assertEqual(snapshot["sha256"], mapping_pose_candidates._sha256_file(snapshot_path))
            rows = _read_pose_rows(
                os.path.join(
                    output_dir,
                    report["pose_candidates"]["gt_as_imu"]["file"],
                )
            )
            self.assertEqual([row["frame"] for row in rows], ["000001"])
            self.assertAlmostEqual(float(rows[0]["timestamp"]), 2.5, places=8)
            self.assertAlmostEqual(float(rows[0]["tx"]), 1.5, places=8)

    def test_builds_composed_extrinsic_and_two_pose_hypotheses_with_slerp(self):
        """候选必须组合外参，且对同一 GT 输出两种 frame 假设。"""
        with tempfile.TemporaryDirectory() as root:
            radar_to_imu, radar_to_lidar, gt_odom, sync_csv = _write_inputs(root)
            output_dir = os.path.join(root, "output")

            report = mapping_pose_candidates.build_mapping_pose_candidates(
                radar_to_imu_matrix_path=radar_to_imu,
                radar_to_lidar_path=radar_to_lidar,
                ground_truth_path=gt_odom,
                radar_sync_csv_path=sync_csv,
                output_dir=output_dir,
                max_interpolation_gap_s=3.0,
            )

            self.assertEqual(report["protocol"], "mapping_pose_candidate_diagnostic_v1")
            self.assertFalse(report["formal"])
            self.assertTrue(report["candidate_only"])
            self.assertEqual(report["coverage"]["covered_frame_count"], 1)
            self.assertEqual(report["coverage"]["uncovered_frame_ids"], ["000000", "000002"])
            np.testing.assert_allclose(
                np.asarray(report["candidate_lidar_to_body"]["matrix_4x4"]),
                np.asarray(
                    [
                        [1, 0, 0, 0.75],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ],
                    dtype=np.float64,
                ),
                atol=1e-8,
            )

            imu_rows = _read_pose_rows(
                os.path.join(output_dir, report["pose_candidates"]["gt_as_imu"]["file"])
            )
            lidar_rows = _read_pose_rows(
                os.path.join(output_dir, report["pose_candidates"]["gt_as_lidar"]["file"])
            )
            self.assertEqual([row["frame"] for row in imu_rows], ["000001"])
            self.assertEqual(imu_rows[0]["diagnostic_formal"], "false")
            self.assertAlmostEqual(float(imu_rows[0]["tx"]), 1.0, places=8)
            self.assertAlmostEqual(float(imu_rows[0]["qz"]), math.sqrt(0.5), places=7)
            self.assertAlmostEqual(float(imu_rows[0]["qw"]), math.sqrt(0.5), places=7)
            self.assertAlmostEqual(float(lidar_rows[0]["tx"]), 1.0, places=7)
            self.assertAlmostEqual(float(lidar_rows[0]["ty"]), -0.75, places=7)

            with open(os.path.join(output_dir, "audit.json"), encoding="utf-8") as handle:
                persisted = json.load(handle)
            self.assertEqual(persisted, report)

    def test_out_of_range_and_large_gap_frames_are_reported_without_extrapolation(self):
        """无 GT 包围或插值间隔超限时，只报 uncovered 而不伪造 pose。"""
        with tempfile.TemporaryDirectory() as root:
            inputs = _write_inputs(
                root,
                ground_truth_times=(1.0, 11.0),
                radar_times=(0.0, 2.0, 12.0),
            )
            output_dir = os.path.join(root, "output")
            report = mapping_pose_candidates.build_mapping_pose_candidates(
                radar_to_imu_matrix_path=inputs[0],
                radar_to_lidar_path=inputs[1],
                ground_truth_path=inputs[2],
                radar_sync_csv_path=inputs[3],
                output_dir=output_dir,
                max_interpolation_gap_s=2.0,
            )

            self.assertEqual(report["coverage"]["covered_frame_count"], 0)
            self.assertEqual(
                [record["reason"] for record in report["coverage"]["uncovered_records"]],
                ["before_ground_truth", "interpolation_gap_exceeded", "after_ground_truth"],
            )
            for candidate in report["pose_candidates"].values():
                self.assertEqual(
                    _read_pose_rows(os.path.join(output_dir, candidate["file"])),
                    [],
                )

    def test_nonempty_or_symlink_output_is_rejected_without_overwrite(self):
        """诊断也不得覆盖历史输出或写入符号链接。"""
        with tempfile.TemporaryDirectory() as root:
            inputs = _write_inputs(root)
            nonempty = os.path.join(root, "nonempty")
            os.makedirs(nonempty)
            with open(os.path.join(nonempty, "old.txt"), "w", encoding="utf-8") as handle:
                handle.write("old")
            kwargs = {
                "radar_to_imu_matrix_path": inputs[0],
                "radar_to_lidar_path": inputs[1],
                "ground_truth_path": inputs[2],
                "radar_sync_csv_path": inputs[3],
                "max_interpolation_gap_s": 3.0,
            }
            with self.assertRaisesRegex(ValueError, "非空"):
                mapping_pose_candidates.build_mapping_pose_candidates(
                    output_dir=nonempty,
                    **kwargs,
                )
            self.assertEqual(os.listdir(nonempty), ["old.txt"])

            real_dir = os.path.join(root, "real")
            linked_dir = os.path.join(root, "linked")
            os.makedirs(real_dir)
            os.symlink(real_dir, linked_dir)
            with self.assertRaisesRegex(ValueError, "符号链接"):
                mapping_pose_candidates.build_mapping_pose_candidates(
                    output_dir=linked_dir,
                    **kwargs,
                )

    def test_formal_mapping_loaders_reject_diagnostic_candidates(self):
        """formal 地图入口必须按内容拒绝候选，不能只依赖文件名约定。"""
        from diffusion_consistency_radar.geometry_protocol import (
            load_extrinsic_transform,
        )
        from diffusion_consistency_radar.scripts.streaming_map_update import (
            load_pose_table,
        )

        with tempfile.TemporaryDirectory() as root:
            inputs = _write_inputs(root)
            output_dir = os.path.join(root, "output")
            report = mapping_pose_candidates.build_mapping_pose_candidates(
                radar_to_imu_matrix_path=inputs[0],
                radar_to_lidar_path=inputs[1],
                ground_truth_path=inputs[2],
                radar_sync_csv_path=inputs[3],
                output_dir=output_dir,
                max_interpolation_gap_s=3.0,
            )

            extrinsic_path = os.path.join(
                output_dir,
                report["candidate_lidar_to_body"]["file"],
            )
            pose_path = os.path.join(
                output_dir,
                report["pose_candidates"]["gt_as_imu"]["file"],
            )
            with self.assertRaisesRegex(ValueError, "诊断候选"):
                load_extrinsic_transform(extrinsic_path)
            with self.assertRaisesRegex(ValueError, "诊断候选"):
                load_pose_table(pose_path, ["000001_voxel.npy"])


if __name__ == "__main__":
    unittest.main()
