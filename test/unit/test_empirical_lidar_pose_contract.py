#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""文件功能：验证经验 LiDAR 位姿收据、direct pose 地图接口与离线严格入口。"""

import csv
import hashlib
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

from diffusion_consistency_radar.prediction_artifact_protocol import (
    build_prediction_voxel_metadata,
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


class EmpiricalLidarPoseContractTest(unittest.TestCase):
    @staticmethod
    def _write_source_diagnostics(root):
        """构造 2 帧 inference、1 帧无外推 pose 的最小诊断来源。"""
        candidate_dir = os.path.join(root, "candidate")
        overlap_dir = os.path.join(root, "overlap")
        os.makedirs(candidate_dir)
        os.makedirs(overlap_dir)

        pose_file = "candidate_body_to_local_gt_as_lidar.diagnostic.csv"
        pose_path = os.path.join(candidate_dir, pose_file)
        with open(pose_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "frame",
                    "timestamp",
                    "tx",
                    "ty",
                    "tz",
                    "qx",
                    "qy",
                    "qz",
                    "qw",
                    "diagnostic_formal",
                    "gt_pose_hypothesis",
                ]
            )
            # T_local_body.x=2，候选 T_body_lidar.x=1，直接 LiDAR pose 应为 x=3。
            writer.writerow(
                ["000001", "10.0", "2", "0", "0", "0", "0", "0", "1", "false", "gt_pose_is_lidar_then_convert_to_imu_body"]
            )

        extrinsic_file = "candidate_lidar_to_imu_body.diagnostic.txt"
        extrinsic_path = os.path.join(candidate_dir, extrinsic_file)
        with open(extrinsic_path, "w", encoding="utf-8") as handle:
            handle.write("R: 1 0 0 0 1 0 0 0 1\n")
            handle.write("T: 1 0 0\n")

        sync_file = "radar_lidar_sync.snapshot.csv"
        sync_path = os.path.join(candidate_dir, sync_file)
        with open(sync_path, "w", encoding="utf-8", newline="") as handle:
            handle.write("frame,radar_timestamp,lidar_timestamp\n")
            handle.write("000000,9.0,9.1\n")
            handle.write("000001,9.9,10.0\n")

        candidate_audit = {
            "protocol": "mapping_pose_candidate_diagnostic_v2",
            "formal": False,
            "candidate_only": True,
            "assumptions_resolved": False,
            "formal_blockers": [
                "ground_truth_pose_frame_not_authoritatively_verified",
                "airborne_body_axes_not_authoritatively_verified",
            ],
            "timing": {"pose_reference_sensor": "lidar"},
            "coverage": {
                "radar_frame_count": 2,
                "covered_frame_count": 1,
                "uncovered_frame_count": 1,
                "uncovered_frame_ids": ["000000"],
                "no_extrapolation": True,
            },
            "inputs": {
                "radar_lidar_sync_snapshot": {
                    "file": sync_file,
                    "sha256": _sha256_file(sync_path),
                    "source_sha256": _sha256_file(sync_path),
                }
            },
            "candidate_lidar_to_body": {
                "body_frame_candidate": "imu",
                "direction": "lidar_to_imu_body",
                "file": extrinsic_file,
                "sha256": _sha256_file(extrinsic_path),
                "matrix_4x4": [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            "pose_candidates": {
                "gt_as_lidar": {
                    "direction": "imu_body_to_local",
                    "file": pose_file,
                    "frame_count": 1,
                    "gt_pose_hypothesis": "lidar",
                    "sha256": _sha256_file(pose_path),
                }
            },
        }
        candidate_audit_path = os.path.join(candidate_dir, "audit.json")
        _write_json(candidate_audit_path, candidate_audit)

        overlap_audit = {
            "protocol": "mapping_pose_overlap_diagnostic_v1",
            "formal": False,
            "diagnostic_only": True,
            "inputs": {
                "candidate_audit_sha256": _sha256_file(candidate_audit_path),
                "radar_lidar_sync_snapshot_sha256": _sha256_file(sync_path),
            },
            "coordinate_contract": {
                "voxel_coordinate_frame": "lidar",
                "pose_composition": "T_local_lidar = T_local_body @ T_body_lidar",
                "gt_as_lidar_external_cancels": True,
            },
            "empirical_ranking": {
                "lower_median_residual_first": ["gt_as_lidar", "gt_as_imu"],
                "preferred_hypothesis_diagnostic_only": "gt_as_lidar",
                "metric": "median_of_pair_symmetric_nn_median_m_lower_is_better",
            },
            "pair_selection": {"selected_pair_count": 2},
            "identifiability": {
                "can_publish_formal_pose": False,
                "can_confirm_radar_to_imu_direction": False,
            },
            "hypothesis_summary": {
                "gt_as_lidar": {"pair_median_nn_m": {"median": 0.4}},
                "gt_as_imu": {"pair_median_nn_m": {"median": 2.3}},
            },
        }
        _write_json(os.path.join(overlap_dir, "audit.json"), overlap_audit)
        return candidate_dir, overlap_dir

    @classmethod
    def _build_contract(cls, root):
        from diffusion_consistency_radar.empirical_pose_contract import (
            build_empirical_lidar_pose_contract,
        )

        candidate_dir, overlap_dir = cls._write_source_diagnostics(root)
        output_dir = os.path.join(root, "empirical_contract")
        receipt = build_empirical_lidar_pose_contract(
            candidate_dir=candidate_dir,
            overlap_dir=overlap_dir,
            output_dir=output_dir,
            command_line="synthetic unit test",
        )
        return output_dir, receipt

    def test_builder_publishes_self_contained_direct_lidar_pose(self):
        from diffusion_consistency_radar.empirical_pose_contract import (
            load_empirical_lidar_pose_contract,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir, receipt = self._build_contract(tmp)
            receipt_path = os.path.join(output_dir, "empirical_pose_receipt.json")
            loaded = load_empirical_lidar_pose_contract(
                receipt_path,
                ["000000_voxel.npy", "000001_voxel.npy"],
            )

            self.assertEqual(receipt["protocol"], "empirical_lidar_pose_contract_v1")
            self.assertTrue(receipt["offline_empirical_mapping"])
            self.assertFalse(receipt["airborne_formal"])
            self.assertFalse(receipt["avoidance_formal"])
            self.assertEqual(receipt["pose_direction"], "lidar_to_local")
            self.assertEqual(loaded["selected_voxel_file_names"], ["000001_voxel.npy"])
            self.assertEqual(loaded["available_frame_count"], 2)
            self.assertEqual(loaded["selected_frame_count"], 1)
            np.testing.assert_allclose(
                loaded["pose_table"]["000001"]["T_local_voxel"][:3, 3],
                [3.0, 0.0, 0.0],
                atol=1e-6,
            )
            expected_members = {
                "lidar_to_local_pose",
                "source_candidate_audit",
                "source_overlap_audit",
                "source_candidate_pose",
                "source_candidate_lidar_to_body",
                "radar_lidar_sync_snapshot",
            }
            self.assertEqual(set(receipt["members"]), expected_members)

    def test_builder_rejects_overlap_not_bound_to_candidate_before_output(self):
        from diffusion_consistency_radar.empirical_pose_contract import (
            build_empirical_lidar_pose_contract,
        )

        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir, overlap_dir = self._write_source_diagnostics(tmp)
            overlap_path = os.path.join(overlap_dir, "audit.json")
            with open(overlap_path, "r", encoding="utf-8") as handle:
                overlap = json.load(handle)
            overlap["inputs"]["candidate_audit_sha256"] = "0" * 64
            _write_json(overlap_path, overlap)
            output_dir = os.path.join(tmp, "must_not_exist")

            with self.assertRaisesRegex(ValueError, "candidate audit SHA-256"):
                build_empirical_lidar_pose_contract(
                    candidate_dir=candidate_dir,
                    overlap_dir=overlap_dir,
                    output_dir=output_dir,
                )
            self.assertFalse(os.path.exists(output_dir))

    def test_runtime_rejects_tampered_direct_pose(self):
        from diffusion_consistency_radar.empirical_pose_contract import (
            load_empirical_lidar_pose_contract,
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_dir, receipt = self._build_contract(tmp)
            pose_path = os.path.join(
                output_dir,
                receipt["members"]["lidar_to_local_pose"]["file"],
            )
            with open(pose_path, "a", encoding="utf-8") as handle:
                handle.write("\n")

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_empirical_lidar_pose_contract(
                    os.path.join(output_dir, "empirical_pose_receipt.json"),
                    ["000001_voxel.npy"],
                )

    def test_direct_local_voxel_pose_is_exclusive_and_audited(self):
        from diffusion_consistency_radar.cm.probabilistic_mapping import (
            GridMapConfig,
            SlidingProbabilisticGridMap,
        )

        cfg = GridMapConfig(
            x_min=0,
            x_max=8,
            y_min=0,
            y_max=1,
            x_resolution=1,
            y_resolution=1,
            z_min=0,
            z_max=1,
            z_resolution=1,
            evidence_pc_range=(0, 0, 0, 2, 1, 1),
        )
        grid = SlidingProbabilisticGridMap(cfg)
        voxel = np.zeros((2, 1, 1, 4), dtype=np.float32)
        voxel[0, 0, 0, 0] = 1.0
        direct_pose = np.eye(4, dtype=np.float32)
        direct_pose[0, 3] = 3.0

        grid.update_from_voxel(
            voxel,
            timestamp=1.0,
            observed_mask=np.ones((2, 1, 1), dtype=np.uint8),
            T_local_voxel=direct_pose,
        )
        snapshot = grid.snapshot()

        self.assertGreater(float(snapshot["occ_prob_layers"][3, 0, 0]), 0.5)
        np.testing.assert_allclose(snapshot["last_T_local_voxel"], direct_pose)
        self.assertEqual(str(snapshot["last_pose_contract"]), "direct_local_voxel")
        self.assertEqual(int(snapshot["last_body_pose_available"]), 0)
        with self.assertRaisesRegex(ValueError, "互斥"):
            grid.update_from_voxel(
                voxel,
                timestamp=2.0,
                T_local_voxel=direct_pose,
                T_local_body=np.eye(4, dtype=np.float32),
            )

    @staticmethod
    def _write_inference_fixture(root):
        from diffusion_consistency_radar.scripts.streaming_map_update import (
            _observed_mask_records_digest,
        )

        inference_dir = os.path.join(root, "inference")
        os.makedirs(inference_dir)
        records = []
        prediction_records = []
        for frame in ("000000", "000001"):
            voxel = np.zeros((4, 1, 2, 1), dtype=np.float32)
            voxel[0, 0, 0, 0] = 1.0
            voxel_file = f"{frame}_voxel.npy"
            voxel_path = os.path.join(inference_dir, voxel_file)
            np.save(voxel_path, voxel)
            prediction_records.append(
                {
                    "frame_id": frame,
                    "file": voxel_file,
                    "sha256": _sha256_file(voxel_path),
                    "shape_czxy": [4, 1, 2, 1],
                    "dtype": "float32",
                }
            )
            mask_file = f"{frame}_observed_mask.npy"
            mask_path = os.path.join(inference_dir, mask_file)
            np.save(mask_path, np.ones((2, 1, 1), dtype=np.uint8))
            records.append(
                {
                    "frame_id": frame,
                    "file": mask_file,
                    "sha256": _sha256_file(mask_path),
                    "observed_voxels": 2,
                }
            )
        run = {
            "stage": "deployment_generation",
            "formal_protocol": True,
            "require_real_ir": True,
            "model_is_multimodal": True,
            "voxel_coordinate_frame": "lidar",
            "frame_count": 2,
            "deployment_identity": {
                "calibration_sha256": {"radar_to_lidar": "a" * 64}
            },
            "observed_mask": {
                "protocol": "radar_endpoint_ray_visibility_v1",
                "coordinate_frame": "lidar",
                "source": "radar_endpoint_rays",
                "ir_frustum_marks_free_space": False,
                "frame_count": 2,
                "observed_voxels": 4,
                "radar_origin_lidar_m": [0.0, 0.0, 0.0],
                "radar_to_lidar_sha256": "a" * 64,
                "files_sha256": _observed_mask_records_digest(records),
                "records": records,
            },
            "prediction_voxel": build_prediction_voxel_metadata(
                prediction_records
            ),
        }
        run_path = os.path.join(root, "inference_run.json")
        _write_json(run_path, run)
        return inference_dir, run_path

    def test_streaming_offline_empirical_selects_receipt_frames_and_never_claims_airborne(self):
        from diffusion_consistency_radar.scripts.streaming_map_update import main

        with tempfile.TemporaryDirectory() as tmp:
            contract_dir, _receipt = self._build_contract(tmp)
            inference_dir, inference_run = self._write_inference_fixture(tmp)
            output_dir = os.path.join(tmp, "map")
            argv = [
                "streaming_map_update.py",
                "--radar_voxel_dir",
                inference_dir,
                "--radar_voxel_layout",
                "czxy",
                "--offline_empirical_mapping",
                "--empirical_pose_receipt",
                os.path.join(contract_dir, "empirical_pose_receipt.json"),
                "--inference_run",
                inference_run,
                "--observed_mask_dir",
                inference_dir,
                "--output_dir",
                output_dir,
                "--pc_range",
                "0",
                "0",
                "0",
                "2",
                "1",
                "1",
                "--map_pc_range",
                "0",
                "0",
                "0",
                "8",
                "1",
                "1",
                "--save_every",
                "1",
            ]

            with mock.patch.object(sys, "argv", argv):
                main()

            with np.load(os.path.join(output_dir, "map_final.npz")) as snapshot:
                # 直接 LiDAR pose 为 x=3m，rolling 窗口因此从 3m 开始；
                # 源占用体素中心 x=0.5m 应落在新窗口索引 0。
                self.assertGreater(float(snapshot["occ_prob_layers"][0, 0, 0]), 0.5)
                self.assertEqual(str(snapshot["last_pose_contract"]), "direct_local_voxel")
                self.assertEqual(int(snapshot["last_body_pose_available"]), 0)
                self.assertEqual(int(snapshot["rolling_enabled"]), 1)
                np.testing.assert_allclose(
                    snapshot["map_pc_range_local"],
                    [3.0, 0.0, 0.0, 11.0, 1.0, 1.0],
                )
            with open(os.path.join(output_dir, "map_run.json"), "r", encoding="utf-8") as handle:
                run = json.load(handle)
            self.assertFalse(run["formal_mapping"])
            self.assertTrue(run["offline_empirical_mapping"])
            self.assertFalse(run["airborne_formal"])
            self.assertFalse(run["avoidance_formal"])
            self.assertEqual(
                run["protocol"],
                "pose_aware_layered_map_offline_empirical_v3",
            )
            self.assertEqual(run["runtime_contract_status"], "offline_empirical_fail_closed")
            self.assertEqual(run["pose_mode"], "empirical_lidar_to_local")
            self.assertEqual(run["pose_direction"], "lidar_to_local")
            self.assertEqual(run["proximity_query"], "lidar_origin_local_3d_three_state_v1")
            self.assertEqual(run["available_inference_frame_count"], 2)
            self.assertEqual(run["frame_count"], 1)

    def test_streaming_mapping_modes_are_mutually_exclusive_before_output(self):
        from diffusion_consistency_radar.scripts.streaming_map_update import main

        with tempfile.TemporaryDirectory() as tmp:
            voxel_dir = os.path.join(tmp, "voxels")
            output_dir = os.path.join(tmp, "map")
            os.makedirs(voxel_dir)
            np.save(
                os.path.join(voxel_dir, "000001_voxel.npy"),
                np.zeros((2, 1, 1, 4), dtype=np.float32),
            )
            argv = [
                "streaming_map_update.py",
                "--radar_voxel_dir",
                voxel_dir,
                "--formal_mapping",
                "--offline_empirical_mapping",
                "--output_dir",
                output_dir,
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "互斥"):
                    main()
            self.assertFalse(os.path.exists(output_dir))


if __name__ == "__main__":
    unittest.main()
