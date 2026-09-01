# -*- coding: utf-8 -*-
"""文件功能：验证唯一 temporal split artifact、purge gap 与内容绑定。"""

import csv
import json
import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion_consistency_radar.dataset_manifest import write_scene_manifest_atomic
from diffusion_consistency_radar.formal_data_protocol import (
    FormalDataProtocolError,
    build_and_write_formal_data_protocol,
    load_formal_data_protocol_artifact,
)
from diffusion_consistency_radar.extraction_receipt import (
    CRITICAL_EXTRACTION_TOPICS,
    finalize_extraction_receipt,
    mark_bag_processed,
    new_extraction_receipt,
    record_extraction_success,
)
from diffusion_consistency_radar.temporal_split import (
    TemporalSplitError,
    build_and_write_temporal_split,
    limit_frame_ids_by_scene,
    load_temporal_split_artifact,
    split_frame_ids_by_scene,
)


def _write_bytes(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(payload)


def _create_scene(root, frame_count=6, *, formal_v3=False):
    scene = "garden"
    scene_dir = os.path.join(root, scene)
    os.makedirs(scene_dir)
    policy = {
        "source_scene": scene,
        "frames_written": frame_count,
        "voxel_coordinate_frame": "lidar",
        "observed_mask_protocol": "lidar_ray_observed_v1",
    }
    if formal_v3:
        receipt = new_extraction_receipt(scene, ["garden_0.bag"])
        for topic in CRITICAL_EXTRACTION_TOPICS:
            record_extraction_success(receipt, topic)
        mark_bag_processed(receipt, "garden_0.bag")
        finalize_extraction_receipt(receipt)
        field_schema = {
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
                "sha256": "9" * 64,
            },
        }
        policy.update(
            {
                "radar_statistics_protocol": "radar_point_count_field_validity_v2",
                "radar_aggregation_semantics": (
                    "per_field_finite_count_mean_and_doppler_variance_v2"
                ),
                "radar_field_schema": field_schema,
                "radar_field_schema_sha256": "a" * 64,
                "radar_field_schema_status": "verified",
                "radar_pointcloud_layout_sha256": "b" * 64,
                "radar_doppler_positive_direction": "toward_sensor",
                "extraction_receipt": receipt,
                "extraction_receipt_sha256": "c" * 64,
                "extraction_receipt_status": "complete",
            }
        )
    with open(
        os.path.join(scene_dir, "preprocess_policy.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(policy, handle)
    for index in range(frame_count):
        frame_id = f"{index:06d}"
        for modality in ("radar_voxel", "lidar_voxel", "target_voxel"):
            _write_bytes(
                os.path.join(scene_dir, modality, f"{frame_id}.npz"),
                f"{modality}-{frame_id}".encode("utf-8"),
            )
        _write_bytes(
            os.path.join(scene_dir, "observed_mask", f"{frame_id}.npz"),
            f"observed-{frame_id}".encode("utf-8"),
        )
        _write_bytes(
            os.path.join(scene_dir, "ir_image", f"{frame_id}_ir.npy"),
            f"ir-{frame_id}".encode("utf-8"),
        )

    sync_path = os.path.join(scene_dir, "radar_ir_sync.csv")
    with open(sync_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "frame_index",
                "radar_timestamp",
                "ir_timestamp",
                "delta_seconds",
                "signed_delta_seconds",
            ),
        )
        writer.writeheader()
        for index in range(frame_count):
            writer.writerow(
                {
                    "frame_index": index,
                    "radar_timestamp": f"{float(index):.9f}",
                    "ir_timestamp": f"{float(index):.9f}",
                    "delta_seconds": "0.000000000",
                    "signed_delta_seconds": "0.000000000",
                }
            )

    provenance = {}
    for key in (
        "preprocess_script",
        "radar_to_lidar",
        "radar_to_thermal",
        "lidar_to_thermal",
        "thermal_intrinsics",
        "radar_lidar_sync",
        "target_policy",
    ):
        path = os.path.join(scene_dir, f"{key}.txt")
        _write_bytes(path, key.encode("utf-8"))
        provenance[key] = path
    provenance["radar_ir_sync"] = sync_path
    write_scene_manifest_atomic(
        scene_dir,
        scene,
        frame_count,
        provenance,
        profile="training",
    )
    return scene_dir


class TemporalSplitProtocolTest(unittest.TestCase):
    def test_formal_mini_frame_limit_uses_ordered_prefix_and_rejects_short_split(self):
        frame_ids = {
            "garden": ["000002", "000004", "000007"],
            "loop3": ["000010", "000011", "000015"],
        }

        self.assertEqual(
            limit_frame_ids_by_scene(frame_ids, 2, partition="train"),
            {
                "garden": ["000002", "000004"],
                "loop3": ["000010", "000011"],
            },
        )
        with self.assertRaisesRegex(TemporalSplitError, "validation.*3"):
            limit_frame_ids_by_scene(
                {"garden": ["000020", "000021"]},
                3,
                partition="validation",
            )

    def test_data_protocol_is_derived_from_manifest_split_and_observed_records(self):
        with tempfile.TemporaryDirectory() as root:
            _create_scene(root)
            split_path = os.path.join(root, "split.json")
            build_and_write_temporal_split(
                dataset_dir=root,
                scenes=["garden"],
                output_path=split_path,
                train_fraction=0.5,
                purge_seconds=1.0,
                formal=True,
            )
            output = os.path.join(root, "formal_data.json")
            build_and_write_formal_data_protocol(
                dataset_dir=root,
                scenes=["garden"],
                split_artifact_path=split_path,
                output_path=output,
            )
            protocol, digest = load_formal_data_protocol_artifact(
                output,
                dataset_dir=root,
                scenes=["garden"],
                split_artifact_path=split_path,
                stage="ldm",
            )
            self.assertEqual(len(digest), 64)
            self.assertEqual(protocol["protocol"], "formal_data_v2")
            self.assertEqual(
                protocol["observed_mask_protocol"],
                "lidar_ray_observed_v1",
            )
            self.assertEqual(set(protocol["observed_mask_sha256"]), {"garden"})

    def test_formal_v3_binds_verified_radar_and_complete_extraction_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            _create_scene(root, formal_v3=True)
            split_path = os.path.join(root, "split.json")
            build_and_write_temporal_split(
                dataset_dir=root,
                scenes=["garden"],
                output_path=split_path,
                train_fraction=0.5,
                purge_seconds=1.0,
                formal=True,
            )
            output = os.path.join(root, "formal_data_v3.json")
            build_and_write_formal_data_protocol(
                dataset_dir=root,
                scenes=["garden"],
                split_artifact_path=split_path,
                output_path=output,
                protocol_version="v3",
            )
            protocol, _digest = load_formal_data_protocol_artifact(
                output,
                dataset_dir=root,
                scenes=["garden"],
                split_artifact_path=split_path,
                stage="ldm",
            )

            self.assertEqual(protocol["protocol"], "formal_data_v3")
            self.assertEqual(
                protocol["radar_statistics_protocol"],
                "radar_point_count_field_validity_v2",
            )
            self.assertEqual(
                protocol["radar_input_contract"]["doppler"]["positive_direction"],
                "toward_sensor",
            )
            self.assertEqual(
                protocol["extraction_receipt_sha256"],
                {"garden": "c" * 64},
            )

    def test_formal_v3_rejects_legacy_policy_without_verified_sources(self):
        with tempfile.TemporaryDirectory() as root:
            _create_scene(root, formal_v3=False)
            split_path = os.path.join(root, "split.json")
            build_and_write_temporal_split(
                dataset_dir=root,
                scenes=["garden"],
                output_path=split_path,
                train_fraction=0.5,
                purge_seconds=1.0,
                formal=True,
            )
            with self.assertRaisesRegex(FormalDataProtocolError, "Radar statistics"):
                build_and_write_formal_data_protocol(
                    dataset_dir=root,
                    scenes=["garden"],
                    split_artifact_path=split_path,
                    output_path=os.path.join(root, "formal_data_v3.json"),
                    protocol_version="v3",
                )

    def test_builds_disjoint_train_purge_validation_blocks(self):
        with tempfile.TemporaryDirectory() as root:
            _create_scene(root)
            output = os.path.join(root, "split.json")
            build_and_write_temporal_split(
                dataset_dir=root,
                scenes=["garden"],
                output_path=output,
                train_fraction=0.5,
                purge_seconds=1.5,
                formal=True,
            )
            artifact, digest = load_temporal_split_artifact(
                output,
                dataset_dir=root,
                expected_scenes=["garden"],
                require_formal=True,
            )
            self.assertEqual(len(digest), 64)
            scene = artifact["scenes"]["garden"]
            self.assertEqual(scene["train_frame_ids"], ["000000", "000001", "000002"])
            self.assertEqual(scene["purged_frame_ids"], ["000003"])
            self.assertEqual(scene["validation_frame_ids"], ["000004", "000005"])
            self.assertEqual(
                split_frame_ids_by_scene(artifact, "train"),
                {"garden": ["000000", "000001", "000002"]},
            )

    def test_formal_split_rejects_zero_purge(self):
        with tempfile.TemporaryDirectory() as root:
            _create_scene(root)
            with self.assertRaisesRegex(TemporalSplitError, "purge_seconds"):
                build_and_write_temporal_split(
                    dataset_dir=root,
                    scenes=["garden"],
                    output_path=os.path.join(root, "split.json"),
                    train_fraction=0.5,
                    purge_seconds=0.0,
                    formal=True,
                )

    def test_output_is_immutable(self):
        with tempfile.TemporaryDirectory() as root:
            _create_scene(root)
            output = os.path.join(root, "split.json")
            kwargs = dict(
                dataset_dir=root,
                scenes=["garden"],
                output_path=output,
                train_fraction=0.5,
                purge_seconds=1.0,
                formal=True,
            )
            build_and_write_temporal_split(**kwargs)
            with self.assertRaisesRegex(TemporalSplitError, "已存在"):
                build_and_write_temporal_split(**kwargs)


if __name__ == "__main__":
    unittest.main()
