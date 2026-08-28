# -*- coding: utf-8 -*-
"""文件功能：验证双 GT-frame 假设的跨帧 LiDAR 重合诊断合同。"""

import csv
import hashlib
import importlib.util
import json
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
    / "evaluate_mapping_pose_overlap.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_mapping_pose_overlap",
    SCRIPT_PATH,
)
mapping_pose_overlap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mapping_pose_overlap)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rotation_z(degrees):
    radians = np.deg2rad(degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rotation_x(degrees):
    radians = np.deg2rad(degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def _matrix_to_quaternion(matrix):
    """把测试刚体旋转转换成 CSV 使用的 xyzw 四元数。"""
    rotation = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                ]
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
    return quaternion / np.linalg.norm(quaternion)


def _write_pose_csv(path, transforms, hypothesis):
    fields = (
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
    )
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, transform in enumerate(transforms):
            quaternion = _matrix_to_quaternion(transform[:3, :3])
            writer.writerow(
                {
                    "frame": f"{index:06d}",
                    "timestamp": f"{float(index):.9f}",
                    "tx": f"{transform[0, 3]:.12f}",
                    "ty": f"{transform[1, 3]:.12f}",
                    "tz": f"{transform[2, 3]:.12f}",
                    "qx": f"{quaternion[0]:.12f}",
                    "qy": f"{quaternion[1]:.12f}",
                    "qz": f"{quaternion[2]:.12f}",
                    "qw": f"{quaternion[3]:.12f}",
                    "diagnostic_formal": "false",
                    "gt_pose_hypothesis": hypothesis,
                }
            )


def _write_sparse_voxel(path, points, pc_range, voxel_size):
    minimum = np.asarray(pc_range[:3], dtype=np.float64)
    size = np.asarray(voxel_size, dtype=np.float64)
    shape_xyz = np.rint(
        (np.asarray(pc_range[3:], dtype=np.float64) - minimum) / size
    ).astype(np.int64)
    coords = np.floor((np.asarray(points) - minimum) / size).astype(np.int64)
    centers = minimum + (coords.astype(np.float64) + 0.5) * size
    np.testing.assert_allclose(centers, points, atol=1e-8)
    features = np.zeros((coords.shape[0], 4), dtype=np.float32)
    features[:, 0] = 1.0
    np.savez(
        path,
        coords=coords,
        features=features,
        shape=np.asarray([*shape_xyz.tolist(), 4], dtype=np.int64),
    )


def _write_fixture(root):
    scene_dir = os.path.join(root, "scene")
    candidate_dir = os.path.join(root, "candidates")
    voxel_dir = os.path.join(scene_dir, "lidar_voxel")
    os.makedirs(voxel_dir)
    os.makedirs(candidate_dir)

    pc_range = [-5.0, -5.0, -3.0, 5.0, 5.0, 3.0]
    voxel_size = [1.0, 1.0, 1.0]
    world_points = np.asarray(
        [
            [0.5, 0.5, 0.5],
            [1.5, 0.5, 0.5],
            [2.5, -0.5, 0.5],
            [-1.5, 2.5, -0.5],
            [-2.5, -1.5, 1.5],
            [0.5, -2.5, -1.5],
        ],
        dtype=np.float64,
    )
    true_lidar_poses = []
    for index, degrees in enumerate((0.0, 90.0)):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = _rotation_z(degrees)
        true_lidar_poses.append(transform)
        local_points = (transform[:3, :3].T @ world_points.T).T
        _write_sparse_voxel(
            os.path.join(voxel_dir, f"{index:06d}.npz"),
            local_points,
            pc_range,
            voxel_size,
        )

    policy_path = os.path.join(scene_dir, "preprocess_policy.json")
    with open(policy_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "align_to": "lidar",
                "pc_range": pc_range,
                "voxel_size": voxel_size,
                "frames_written": 2,
            },
            handle,
        )

    records = []
    for index in range(2):
        path = os.path.join(voxel_dir, f"{index:06d}.npz")
        records.append(
            {
                "frame_id": f"{index:06d}",
                "path": f"lidar_voxel/{index:06d}.npz",
                "sha256": _sha256(path),
                "size": os.path.getsize(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "scene": "synthetic",
        "frame_count": 2,
        "modalities": {"lidar_voxel": records},
    }
    manifest["content_sha256"] = _canonical_sha256(manifest)
    with open(os.path.join(scene_dir, "dataset_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    sync_path = os.path.join(scene_dir, "radar_ir_sync.csv")
    with open(sync_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("frame_index", "radar_timestamp", "ir_timestamp"),
        )
        writer.writeheader()
        for index in range(2):
            writer.writerow(
                {
                    "frame_index": index,
                    "radar_timestamp": f"{float(index):.9f}",
                    "ir_timestamp": f"{float(index):.9f}",
                }
            )

    body_from_lidar = np.eye(4, dtype=np.float64)
    body_from_lidar[:3, :3] = _rotation_x(90.0)
    extrinsic_path = os.path.join(
        candidate_dir,
        "candidate_lidar_to_imu_body.diagnostic.txt",
    )
    with open(extrinsic_path, "w", encoding="utf-8") as handle:
        handle.write("# DIAGNOSTIC CANDIDATE ONLY; formal=false\n")
        handle.write(
            "R: "
            + " ".join(f"{value:.12f}" for value in body_from_lidar[:3, :3].reshape(-1))
            + "\n"
        )
        handle.write("T: 0 0 0\n")

    imu_pose_path = os.path.join(
        candidate_dir,
        "candidate_body_to_local_gt_as_imu.diagnostic.csv",
    )
    lidar_pose_path = os.path.join(
        candidate_dir,
        "candidate_body_to_local_gt_as_lidar.diagnostic.csv",
    )
    _write_pose_csv(imu_pose_path, true_lidar_poses, "gt_pose_is_imu")
    lidar_body_poses = [
        transform @ np.linalg.inv(body_from_lidar) for transform in true_lidar_poses
    ]
    _write_pose_csv(
        lidar_pose_path,
        lidar_body_poses,
        "gt_pose_is_lidar_then_convert_to_imu_body",
    )

    audit = {
        "protocol": "mapping_pose_candidate_diagnostic_v1",
        "formal": False,
        "candidate_lidar_to_body": {
            "file": os.path.basename(extrinsic_path),
            "sha256": _sha256(extrinsic_path),
            "matrix_4x4": body_from_lidar.tolist(),
        },
        "pose_candidates": {
            "gt_as_imu": {
                "file": os.path.basename(imu_pose_path),
                "sha256": _sha256(imu_pose_path),
            },
            "gt_as_lidar": {
                "file": os.path.basename(lidar_pose_path),
                "sha256": _sha256(lidar_pose_path),
            },
        },
        "inputs": {
            "radar_sync_csv": {
                "path": sync_path,
                "sha256": _sha256(sync_path),
            }
        },
    }
    with open(os.path.join(candidate_dir, "audit.json"), "w", encoding="utf-8") as handle:
        json.dump(audit, handle)
    return scene_dir, candidate_dir


def _upgrade_candidate_fixture_to_v2(candidate_dir):
    """给合成候选增加自包含 Radar--LiDAR snapshot 收据。"""
    snapshot_path = os.path.join(candidate_dir, "radar_lidar_sync.snapshot.csv")
    Path(snapshot_path).write_text(
        "pair_index,radar_timestamp,lidar_timestamp\n0,0.0,0.0\n",
        encoding="utf-8",
    )
    audit_path = os.path.join(candidate_dir, "audit.json")
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    audit["protocol"] = "mapping_pose_candidate_diagnostic_v2"
    audit["inputs"]["radar_lidar_sync_snapshot"] = {
        "source_path": "/diagnostic/source/radar_lidar_sync.csv",
        "source_sha256": _sha256(snapshot_path),
        "file": os.path.basename(snapshot_path),
        "sha256": _sha256(snapshot_path),
    }
    Path(audit_path).write_text(json.dumps(audit), encoding="utf-8")
    return snapshot_path


class MappingPoseOverlapDiagnosticTest(unittest.TestCase):
    def test_candidate_v2_sync_snapshot_tamper_is_rejected(self):
        """LiDAR-time candidate 的封存 sync 漂移必须在读取体素前失败。"""
        with tempfile.TemporaryDirectory() as root:
            scene_dir, candidate_dir = _write_fixture(root)
            snapshot_path = _upgrade_candidate_fixture_to_v2(candidate_dir)
            with open(snapshot_path, "a", encoding="utf-8") as handle:
                handle.write("1,1.0,1.0\n")
            with self.assertRaisesRegex(ValueError, "Radar--LiDAR.*SHA-256"):
                mapping_pose_overlap.evaluate_mapping_pose_overlap(
                    processed_scene_dir=scene_dir,
                    candidate_dir=candidate_dir,
                    output_dir=os.path.join(root, "result"),
                    pair_delta_s=1.0,
                    pair_delta_tolerance_s=0.01,
                    min_rotation_deg=45.0,
                    max_translation_m=1.0,
                    min_sensor_range_m=0.0,
                    max_sensor_range_m=10.0,
                    max_pairs=4,
                )

    def test_manifest_self_hash_is_verified_before_voxel_receipts(self):
        """被整体改写的 manifest 不能仅靠同步改写逐帧 hash 获得信任。"""
        with tempfile.TemporaryDirectory() as root:
            scene_dir, candidate_dir = _write_fixture(root)
            manifest_path = os.path.join(scene_dir, "dataset_manifest.json")
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            manifest["scene"] = "tampered"
            Path(manifest_path).write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content_sha256"):
                mapping_pose_overlap.evaluate_mapping_pose_overlap(
                    processed_scene_dir=scene_dir,
                    candidate_dir=candidate_dir,
                    output_dir=os.path.join(root, "result"),
                    pair_delta_s=1.0,
                    pair_delta_tolerance_s=0.01,
                    min_rotation_deg=45.0,
                    max_translation_m=1.0,
                    min_sensor_range_m=0.0,
                    max_sensor_range_m=10.0,
                    max_pairs=4,
                )

    def test_correct_gt_frame_has_lower_overlap_residual_but_stays_diagnostic(self):
        """正确 LiDAR pose 假设应胜出，但结论不得冒充正式 frame 证明。"""
        with tempfile.TemporaryDirectory() as root:
            scene_dir, candidate_dir = _write_fixture(root)
            output_dir = os.path.join(root, "result")
            report = mapping_pose_overlap.evaluate_mapping_pose_overlap(
                processed_scene_dir=scene_dir,
                candidate_dir=candidate_dir,
                output_dir=output_dir,
                pair_delta_s=1.0,
                pair_delta_tolerance_s=0.01,
                min_rotation_deg=45.0,
                max_translation_m=1.0,
                min_sensor_range_m=0.0,
                max_sensor_range_m=10.0,
                max_pairs=4,
            )

            self.assertEqual(report["protocol"], "mapping_pose_overlap_diagnostic_v1")
            self.assertFalse(report["formal"])
            self.assertEqual(report["pair_selection"]["selected_pair_count"], 1)
            self.assertEqual(
                report["empirical_ranking"]["lower_median_residual_first"][0],
                "gt_as_lidar",
            )
            lidar_score = report["hypothesis_summary"]["gt_as_lidar"][
                "pair_median_nn_m"
            ]["median"]
            imu_score = report["hypothesis_summary"]["gt_as_imu"][
                "pair_median_nn_m"
            ]["median"]
            self.assertLess(lidar_score, imu_score)
            self.assertFalse(
                report["identifiability"]["can_confirm_radar_to_imu_direction"]
            )
            self.assertFalse(report["identifiability"]["can_publish_formal_pose"])
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "pair_metrics.csv")))

    def test_tampered_voxel_is_rejected_by_manifest_hash(self):
        """输入体素内容漂移必须在计算指标前失败。"""
        with tempfile.TemporaryDirectory() as root:
            scene_dir, candidate_dir = _write_fixture(root)
            voxel_path = os.path.join(scene_dir, "lidar_voxel", "000001.npz")
            with open(voxel_path, "ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                mapping_pose_overlap.evaluate_mapping_pose_overlap(
                    processed_scene_dir=scene_dir,
                    candidate_dir=candidate_dir,
                    output_dir=os.path.join(root, "result"),
                    pair_delta_s=1.0,
                    pair_delta_tolerance_s=0.01,
                    min_rotation_deg=45.0,
                    max_translation_m=1.0,
                    min_sensor_range_m=0.0,
                    max_sensor_range_m=10.0,
                    max_pairs=4,
                )

    def test_nonempty_output_is_rejected_without_overwrite(self):
        """诊断结果必须发布到 fresh 目录，不能覆盖历史输出。"""
        with tempfile.TemporaryDirectory() as root:
            scene_dir, candidate_dir = _write_fixture(root)
            output_dir = os.path.join(root, "result")
            os.makedirs(output_dir)
            old_path = os.path.join(output_dir, "old.txt")
            with open(old_path, "w", encoding="utf-8") as handle:
                handle.write("keep")
            with self.assertRaisesRegex(ValueError, "非空"):
                mapping_pose_overlap.evaluate_mapping_pose_overlap(
                    processed_scene_dir=scene_dir,
                    candidate_dir=candidate_dir,
                    output_dir=output_dir,
                    pair_delta_s=1.0,
                    pair_delta_tolerance_s=0.01,
                    min_rotation_deg=45.0,
                    max_translation_m=1.0,
                    min_sensor_range_m=0.0,
                    max_sensor_range_m=10.0,
                    max_pairs=4,
                )
            self.assertEqual(Path(old_path).read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
