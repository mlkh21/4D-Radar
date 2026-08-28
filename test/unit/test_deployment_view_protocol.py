#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证严格 deployment-profile 数据视图的生产、身份绑定与拒绝策略。"""

import json
import os
import shutil
import sys
import tempfile
import unittest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion_consistency_radar.dataset_manifest import (  # noqa: E402
    validate_scene_manifest,
    write_scene_manifest_atomic,
)


def _write_bytes(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(payload)


def _build_training_fixture(root, scene="loop3", frame_count=2):
    """创建带 training v2 manifest 的两帧自包含小数据。"""
    training_root = os.path.join(root, "training")
    calibration_dir = os.path.join(root, "config")
    scene_dir = os.path.join(training_root, scene)
    os.makedirs(scene_dir)

    policy_path = os.path.join(scene_dir, "preprocess_policy.json")
    with open(policy_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_scene": scene,
                "frames_written": frame_count,
                "voxel_coordinate_frame": "lidar",
            },
            handle,
        )
    target_policy_path = os.path.join(scene_dir, "target_policy.json")
    _write_bytes(target_policy_path, b"target-policy")

    for index in range(frame_count):
        frame_id = f"{index:06d}"
        _write_bytes(
            os.path.join(scene_dir, "radar_voxel", f"{frame_id}.npz"),
            b"radar" + bytes([index]),
        )
        _write_bytes(
            os.path.join(scene_dir, "lidar_voxel", f"{frame_id}.npz"),
            b"lidar" + bytes([index]),
        )
        _write_bytes(
            os.path.join(scene_dir, "target_voxel", f"{frame_id}.npz"),
            b"target" + bytes([index]),
        )
        _write_bytes(
            os.path.join(scene_dir, "observed_mask", f"{frame_id}.npz"),
            b"observed" + bytes([index]),
        )
        _write_bytes(
            os.path.join(scene_dir, "ir_image", f"{frame_id}_ir.npy"),
            b"ir" + bytes([index]),
        )

    preprocess_script = os.path.join(root, "preprocess.py")
    _write_bytes(preprocess_script, b"preprocess-script")
    calibration_paths = {
        "radar_to_lidar": os.path.join(
            calibration_dir, "calib_radar_to_livox.txt"
        ),
        "radar_to_thermal": os.path.join(
            calibration_dir, "calib_radar_to_thermal.txt"
        ),
        "lidar_to_thermal": os.path.join(
            calibration_dir, "calib_livox_to_thermal.txt"
        ),
        "thermal_intrinsics": os.path.join(
            calibration_dir, "calib_cam_thermal.txt"
        ),
    }
    for key, path in calibration_paths.items():
        _write_bytes(path, key.encode("utf-8"))

    radar_lidar_sync = os.path.join(root, "radar_lidar_sync.csv")
    radar_ir_sync = os.path.join(scene_dir, "radar_ir_sync.csv")
    _write_bytes(radar_lidar_sync, b"radar-lidar-sync")
    _write_bytes(radar_ir_sync, b"radar-ir-sync")
    provenance = {
        "preprocess_script": preprocess_script,
        **calibration_paths,
        "radar_lidar_sync": radar_lidar_sync,
        "radar_ir_sync": radar_ir_sync,
        "target_policy": target_policy_path,
    }
    write_scene_manifest_atomic(
        scene_dir,
        scene,
        frame_count,
        provenance,
        profile="training",
    )
    return {
        "training_root": training_root,
        "calibration_dir": calibration_dir,
        "preprocess_script": preprocess_script,
        "scene": scene,
    }


class DeploymentViewProtocolTest(unittest.TestCase):
    def _build(self, fixture, output_root, link_mode="hardlink"):
        from diffusion_consistency_radar.deployment_view import (
            build_deployment_dataset,
        )

        return build_deployment_dataset(
            training_dataset_dir=fixture["training_root"],
            output_dataset_dir=output_root,
            scenes=[fixture["scene"]],
            calibration_dir=fixture["calibration_dir"],
            preprocess_script=fixture["preprocess_script"],
            link_mode=link_mode,
        )

    def test_hardlink_view_is_schema_v3_and_contains_no_supervision(self):
        from diffusion_consistency_radar.deployment_view import (
            validate_deployment_dataset,
        )

        with tempfile.TemporaryDirectory() as root:
            fixture = _build_training_fixture(root)
            output_root = os.path.join(root, "deployment")
            result = self._build(fixture, output_root)
            validated = validate_deployment_dataset(
                output_root,
                scenes=[fixture["scene"]],
            )
            scene_dir = os.path.join(output_root, fixture["scene"])
            manifest = validate_scene_manifest(
                scene_dir,
                fixture["scene"],
                expected_profile="deployment",
            )

            self.assertEqual(manifest["schema_version"], 3)
            self.assertEqual(result["protocol"], "deployment_dataset_v1")
            self.assertEqual(validated["scenes"], [fixture["scene"]])
            self.assertEqual(
                set(os.listdir(scene_dir)),
                {
                    "radar_voxel",
                    "ir_image",
                    "preprocess_policy.json",
                    "radar_ir_sync.csv",
                    "source_training_manifest.json",
                    "deployment_view.json",
                    "dataset_manifest.json",
                },
            )
            source_radar = os.path.join(
                fixture["training_root"], fixture["scene"], "radar_voxel", "000000.npz"
            )
            view_radar = os.path.join(scene_dir, "radar_voxel", "000000.npz")
            self.assertEqual(os.stat(source_radar).st_ino, os.stat(view_radar).st_ino)

    def test_provenance_drift_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = _build_training_fixture(root)
            with open(
                os.path.join(fixture["calibration_dir"], "calib_cam_thermal.txt"),
                "ab",
            ) as handle:
                handle.write(b"changed")
            output_root = os.path.join(root, "deployment")

            with self.assertRaisesRegex(Exception, "provenance|SHA-256|标定"):
                self._build(fixture, output_root)
            self.assertFalse(os.path.exists(output_root))

    def test_extra_supervision_or_content_tamper_is_rejected(self):
        from diffusion_consistency_radar.deployment_view import (
            validate_deployment_dataset,
        )

        for mutation in (
            "extra_target",
            "radar_tamper",
            "sync_tamper",
            "radar_symlink",
        ):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as root:
                    fixture = _build_training_fixture(root)
                    output_root = os.path.join(root, "deployment")
                    self._build(fixture, output_root, link_mode="copy")
                    scene_dir = os.path.join(output_root, fixture["scene"])
                    if mutation == "extra_target":
                        os.makedirs(os.path.join(scene_dir, "target_voxel"))
                    elif mutation == "radar_tamper":
                        with open(
                            os.path.join(scene_dir, "radar_voxel", "000000.npz"),
                            "ab",
                        ) as handle:
                            handle.write(b"tampered")
                    elif mutation == "sync_tamper":
                        with open(
                            os.path.join(scene_dir, "radar_ir_sync.csv"),
                            "ab",
                        ) as handle:
                            handle.write(b"tampered")
                    else:
                        destination = os.path.join(
                            scene_dir, "radar_voxel", "000000.npz"
                        )
                        os.remove(destination)
                        os.symlink(
                            os.path.join(
                                fixture["training_root"],
                                fixture["scene"],
                                "radar_voxel",
                                "000000.npz",
                            ),
                            destination,
                        )

                    with self.assertRaisesRegex(
                        Exception,
                        "未知|内容|manifest|SHA-256|符号链接",
                    ):
                        validate_deployment_dataset(
                            output_root,
                            scenes=[fixture["scene"]],
                        )

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = _build_training_fixture(root)
            output_root = os.path.join(root, "deployment")
            os.makedirs(output_root)
            sentinel = os.path.join(output_root, "keep.txt")
            _write_bytes(sentinel, b"keep")

            with self.assertRaisesRegex(Exception, "存在|覆盖|fresh"):
                self._build(fixture, output_root)
            with open(sentinel, "rb") as handle:
                self.assertEqual(handle.read(), b"keep")

    def test_scene_traversal_and_output_inside_training_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = _build_training_fixture(root)
            from diffusion_consistency_radar.deployment_view import (
                build_deployment_dataset,
            )

            with self.assertRaisesRegex(Exception, "scene.*非法"):
                build_deployment_dataset(
                    training_dataset_dir=fixture["training_root"],
                    output_dataset_dir=os.path.join(root, "deployment"),
                    scenes=["../loop3"],
                    calibration_dir=fixture["calibration_dir"],
                    preprocess_script=fixture["preprocess_script"],
                )
            nested_output = os.path.join(fixture["training_root"], "deployment")
            with self.assertRaisesRegex(Exception, "training dataset 内部"):
                self._build(fixture, nested_output)
            self.assertFalse(os.path.exists(nested_output))

    def test_hardlink_view_remains_valid_after_transfer_expands_links(self):
        """服务器传输可展开 hardlink；协议只绑定内容，不绑定 inode。"""
        from diffusion_consistency_radar.deployment_view import (
            validate_deployment_dataset,
        )

        with tempfile.TemporaryDirectory() as root:
            fixture = _build_training_fixture(root)
            local_view = os.path.join(root, "deployment")
            transferred_view = os.path.join(root, "transferred")
            self._build(fixture, local_view, link_mode="hardlink")
            shutil.copytree(local_view, transferred_view, copy_function=shutil.copy2)

            local_radar = os.path.join(
                local_view, fixture["scene"], "radar_voxel", "000000.npz"
            )
            transferred_radar = os.path.join(
                transferred_view,
                fixture["scene"],
                "radar_voxel",
                "000000.npz",
            )
            self.assertNotEqual(
                os.stat(local_radar).st_ino,
                os.stat(transferred_radar).st_ino,
            )
            identity = validate_deployment_dataset(
                transferred_view,
                scenes=[fixture["scene"]],
            )
            self.assertEqual(
                identity["scene_results"][fixture["scene"]][
                    "materialization_mode_at_creation"
                ],
                "hardlink",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
