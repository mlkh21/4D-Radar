#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证严格 dataset manifest 的生成、内容寻址和拒绝策略。"""

import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_manifest_module():
    """延迟加载待实现模块，使 RED 以明确断言失败。"""
    try:
        return importlib.import_module("diffusion_consistency_radar.dataset_manifest")
    except ModuleNotFoundError as exc:
        raise AssertionError("dataset manifest 模块尚未实现") from exc


def write_bytes(path, payload):
    """写入小型测试文件，避免读取真实数据。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(payload)


def create_scene(root, scene="garden", frame_count=2):
    """创建含正式 observed mask 的临时场景及 legacy provenance。"""
    scene_dir = os.path.join(root, scene)
    os.makedirs(scene_dir)
    with open(
        os.path.join(scene_dir, "preprocess_policy.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "source_scene": scene,
                "frames_written": frame_count,
                "voxel_coordinate_frame": "lidar",
            },
            handle,
        )

    for index in range(frame_count):
        frame_id = f"{index:06d}"
        write_bytes(
            os.path.join(scene_dir, "radar_voxel", f"{frame_id}.npz"),
            b"radar" + bytes([index]),
        )
        write_bytes(
            os.path.join(scene_dir, "lidar_voxel", f"{frame_id}.npz"),
            b"lidar" + bytes([index]),
        )
        write_bytes(
            os.path.join(scene_dir, "target_voxel", f"{frame_id}.npz"),
            b"target" + bytes([index]),
        )
        write_bytes(
            os.path.join(scene_dir, "observed_mask", f"{frame_id}.npz"),
            b"observed" + bytes([index]),
        )
        write_bytes(
            os.path.join(scene_dir, "ir_image", f"{frame_id}_ir.npy"),
            b"ir" + bytes([index]),
        )

    provenance = {}
    for key in (
        "preprocess_script",
        "calibration",
        "radar_index",
        "lidar_index",
    ):
        path = os.path.join(root, f"{key}.txt")
        write_bytes(path, key.encode("utf-8"))
        provenance[key] = path
    return scene_dir, provenance


def create_v2_provenance(root, profile):
    """创建与 training v2 / deployment v3 精确匹配的 provenance。"""
    keys = [
        "preprocess_script",
        "radar_to_lidar",
        "radar_to_thermal",
        "lidar_to_thermal",
        "thermal_intrinsics",
        "radar_ir_sync",
    ]
    if profile == "training":
        keys.extend(("radar_lidar_sync", "target_policy"))
    provenance = {}
    for key in keys:
        path = os.path.join(root, f"v2_{key}.txt")
        write_bytes(path, key.encode("utf-8"))
        provenance[key] = path
    if profile == "deployment":
        scene_sync_path = os.path.join(root, "garden", "radar_ir_sync.csv")
        write_bytes(scene_sync_path, b"radar_ir_sync")
        provenance["radar_ir_sync"] = scene_sync_path
        special_paths = {
            "source_training_manifest": os.path.join(
                root, "garden", "source_training_manifest.json"
            ),
            "deployment_view_receipt": os.path.join(
                root, "garden", "deployment_view.json"
            ),
        }
        for key, path in special_paths.items():
            write_bytes(path, key.encode("utf-8"))
            provenance[key] = path
    return provenance


class DatasetManifestProtocolTest(unittest.TestCase):
    def test_training_v2_and_deployment_v3_enforce_exact_modalities(self):
        module = load_manifest_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            training_dir, _legacy = create_scene(
                os.path.join(temp_dir, "training")
            )
            training_provenance = create_v2_provenance(
                os.path.join(temp_dir, "training"),
                "training",
            )
            module.write_scene_manifest_atomic(
                training_dir,
                "garden",
                2,
                training_provenance,
                profile="training",
            )
            training = module.validate_scene_manifest(
                training_dir,
                "garden",
                expected_profile="training",
            )
            self.assertEqual(training["schema_version"], 2)
            self.assertEqual(training["voxel_coordinate_frame"], "lidar")
            self.assertIn("observed_mask", training["modalities"])

            deployment_dir, _legacy = create_scene(
                os.path.join(temp_dir, "deployment")
            )
            shutil.rmtree(os.path.join(deployment_dir, "lidar_voxel"))
            shutil.rmtree(os.path.join(deployment_dir, "target_voxel"))
            shutil.rmtree(os.path.join(deployment_dir, "observed_mask"))
            deployment_provenance = create_v2_provenance(
                os.path.join(temp_dir, "deployment"),
                "deployment",
            )
            module.write_scene_manifest_atomic(
                deployment_dir,
                "garden",
                2,
                deployment_provenance,
                profile="deployment",
            )
            deployment = module.validate_scene_manifest(
                deployment_dir,
                "garden",
                expected_profile="deployment",
            )
            self.assertEqual(deployment["schema_version"], 3)
            self.assertEqual(set(deployment["modalities"]), {"radar_voxel", "ir_image"})
            with self.assertRaisesRegex(module.DatasetManifestError, "profile"):
                module.validate_scene_manifest(
                    deployment_dir,
                    "garden",
                    expected_profile="training",
                )

    def test_formal_profile_rejects_legacy_v1_manifest(self):
        module = load_manifest_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_dir, provenance = create_scene(temp_dir)
            module.write_scene_manifest_atomic(
                scene_dir,
                "garden",
                2,
                provenance,
            )
            with self.assertRaisesRegex(module.DatasetManifestError, "legacy v1"):
                module.validate_scene_manifest(
                    scene_dir,
                    "garden",
                    expected_profile="training",
                )

    def test_valid_manifest_round_trip_is_portable_and_content_addressed(self):
        module = load_manifest_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_dir, provenance = create_scene(temp_dir)
            path = module.write_scene_manifest_atomic(
                scene_dir,
                "garden",
                2,
                provenance,
            )
            manifest = module.validate_scene_manifest(scene_dir, "garden")
            copied = os.path.join(temp_dir, "copied", "garden")
            shutil.copytree(scene_dir, copied)
            copied_manifest = module.validate_scene_manifest(copied, "garden")
            serialized = json.dumps(manifest, sort_keys=True)

            self.assertEqual(os.path.basename(path), "dataset_manifest.json")
            self.assertEqual(
                manifest["content_sha256"],
                copied_manifest["content_sha256"],
            )
            self.assertNotIn(temp_dir, serialized)
            self.assertNotIn("mtime", serialized)

    def test_missing_policy_and_scene_mismatch_are_rejected(self):
        module = load_manifest_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_dir, missing_provenance = create_scene(
                os.path.join(temp_dir, "missing")
            )
            os.remove(os.path.join(missing_dir, "preprocess_policy.json"))
            with self.assertRaisesRegex(
                module.DatasetManifestError,
                "preprocess_policy",
            ):
                module.build_scene_manifest(
                    missing_dir,
                    "garden",
                    2,
                    missing_provenance,
                )

            mismatch_dir, mismatch_provenance = create_scene(
                os.path.join(temp_dir, "mismatch")
            )
            with open(
                os.path.join(mismatch_dir, "preprocess_policy.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {"source_scene": "loop3", "frames_written": 2},
                    handle,
                )
            with self.assertRaisesRegex(
                module.DatasetManifestError,
                "source_scene",
            ):
                module.build_scene_manifest(
                    mismatch_dir,
                    "garden",
                    2,
                    mismatch_provenance,
                )

    def test_modality_mismatch_noncontinuous_and_unknown_files_are_rejected(self):
        module = load_manifest_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            mismatch_dir, mismatch_provenance = create_scene(
                os.path.join(temp_dir, "mismatch")
            )
            os.remove(
                os.path.join(mismatch_dir, "target_voxel", "000001.npz")
            )
            with self.assertRaisesRegex(module.DatasetManifestError, "frame ID"):
                module.build_scene_manifest(
                    mismatch_dir,
                    "garden",
                    2,
                    mismatch_provenance,
                )

            gap_dir, gap_provenance = create_scene(os.path.join(temp_dir, "gap"))
            for modality in ("radar_voxel", "lidar_voxel", "target_voxel"):
                os.rename(
                    os.path.join(gap_dir, modality, "000001.npz"),
                    os.path.join(gap_dir, modality, "000002.npz"),
                )
            os.rename(
                os.path.join(gap_dir, "ir_image", "000001_ir.npy"),
                os.path.join(gap_dir, "ir_image", "000002_ir.npy"),
            )
            with self.assertRaisesRegex(module.DatasetManifestError, "连续"):
                module.build_scene_manifest(
                    gap_dir,
                    "garden",
                    2,
                    gap_provenance,
                )

            unknown_dir, unknown_provenance = create_scene(
                os.path.join(temp_dir, "unknown")
            )
            write_bytes(
                os.path.join(unknown_dir, "radar_voxel", "README.txt"),
                b"unexpected",
            )
            with self.assertRaisesRegex(module.DatasetManifestError, "未知文件"):
                module.build_scene_manifest(
                    unknown_dir,
                    "garden",
                    2,
                    unknown_provenance,
                )

    def test_file_and_directory_symlinks_are_rejected(self):
        module = load_manifest_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            file_dir, file_provenance = create_scene(
                os.path.join(temp_dir, "file")
            )
            radar_path = os.path.join(
                file_dir,
                "radar_voxel",
                "000000.npz",
            )
            external = os.path.join(temp_dir, "external.npz")
            write_bytes(external, b"external")
            os.remove(radar_path)
            os.symlink(external, radar_path)
            with self.assertRaisesRegex(module.DatasetManifestError, "符号链接"):
                module.build_scene_manifest(
                    file_dir,
                    "garden",
                    2,
                    file_provenance,
                )

            directory_dir, directory_provenance = create_scene(
                os.path.join(temp_dir, "directory")
            )
            ir_dir = os.path.join(directory_dir, "ir_image")
            external_ir = os.path.join(temp_dir, "external_ir")
            os.rename(ir_dir, external_ir)
            os.symlink(external_ir, ir_dir)
            with self.assertRaisesRegex(module.DatasetManifestError, "符号链接"):
                module.build_scene_manifest(
                    directory_dir,
                    "garden",
                    2,
                    directory_provenance,
                )

    def test_mutated_artifact_policy_and_manifest_are_rejected(self):
        module = load_manifest_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir, artifact_provenance = create_scene(
                os.path.join(temp_dir, "artifact")
            )
            module.write_scene_manifest_atomic(
                artifact_dir,
                "garden",
                2,
                artifact_provenance,
            )
            write_bytes(
                os.path.join(
                    artifact_dir,
                    "radar_voxel",
                    "000000.npz",
                ),
                b"mutated",
            )
            with self.assertRaisesRegex(module.DatasetManifestError, "不一致"):
                module.validate_scene_manifest(artifact_dir, "garden")

            policy_dir, policy_provenance = create_scene(
                os.path.join(temp_dir, "policy")
            )
            module.write_scene_manifest_atomic(
                policy_dir,
                "garden",
                2,
                policy_provenance,
            )
            with open(
                os.path.join(policy_dir, "preprocess_policy.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {"source_scene": "garden", "frames_written": 999},
                    handle,
                )
            with self.assertRaisesRegex(module.DatasetManifestError, "policy"):
                module.validate_scene_manifest(policy_dir, "garden")

            manifest_dir, manifest_provenance = create_scene(
                os.path.join(temp_dir, "manifest")
            )
            manifest_path = module.write_scene_manifest_atomic(
                manifest_dir,
                "garden",
                2,
                manifest_provenance,
            )
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["frame_count"] = 999
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            with self.assertRaisesRegex(
                module.DatasetManifestError,
                "content_sha256",
            ):
                module.validate_scene_manifest(manifest_dir, "garden")

    def test_provenance_is_complete_regular_and_not_symlinked(self):
        module = load_manifest_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_dir, missing_provenance = create_scene(
                os.path.join(temp_dir, "missing")
            )
            missing_provenance.pop("calibration")
            with self.assertRaisesRegex(module.DatasetManifestError, "provenance"):
                module.build_scene_manifest(
                    missing_dir,
                    "garden",
                    2,
                    missing_provenance,
                )

            link_dir, link_provenance = create_scene(
                os.path.join(temp_dir, "link")
            )
            real_path = link_provenance["calibration"]
            link_path = os.path.join(temp_dir, "calibration_link.txt")
            os.symlink(real_path, link_path)
            link_provenance["calibration"] = link_path
            with self.assertRaisesRegex(module.DatasetManifestError, "符号链接"):
                module.build_scene_manifest(
                    link_dir,
                    "garden",
                    2,
                    link_provenance,
                )

    def test_existing_manifest_is_not_overwritten_and_no_temp_file_remains(self):
        module = load_manifest_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_dir, provenance = create_scene(temp_dir)
            path = module.write_scene_manifest_atomic(
                scene_dir,
                "garden",
                2,
                provenance,
            )
            with open(path, "rb") as handle:
                before = handle.read()
            with self.assertRaisesRegex(module.DatasetManifestError, "已存在"):
                module.write_scene_manifest_atomic(
                    scene_dir,
                    "garden",
                    2,
                    provenance,
                )
            with open(path, "rb") as handle:
                after = handle.read()
            leftovers = [
                name
                for name in os.listdir(scene_dir)
                if name.startswith(".dataset_manifest.")
            ]

            self.assertEqual(before, after)
            self.assertEqual(leftovers, [])

    def test_cli_create_and_validate_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_dir, _legacy = create_scene(temp_dir)
            provenance = create_v2_provenance(temp_dir, "training")
            script = os.path.join(
                PROJECT_ROOT,
                "diffusion_consistency_radar",
                "scripts",
                "dataset_manifest.py",
            )
            create_result = subprocess.run(
                [
                    sys.executable,
                    script,
                    "create",
                    "--scene_dir",
                    scene_dir,
                    "--scene",
                    "garden",
                    "--expected_frame_count",
                    "2",
                    "--profile",
                    "training",
                    "--preprocess_script",
                    provenance["preprocess_script"],
                    "--radar_to_lidar",
                    provenance["radar_to_lidar"],
                    "--radar_to_thermal",
                    provenance["radar_to_thermal"],
                    "--lidar_to_thermal",
                    provenance["lidar_to_thermal"],
                    "--thermal_intrinsics",
                    provenance["thermal_intrinsics"],
                    "--radar_ir_sync",
                    provenance["radar_ir_sync"],
                    "--radar_lidar_sync",
                    provenance["radar_lidar_sync"],
                    "--target_policy",
                    provenance["target_policy"],
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            validate_result = subprocess.run(
                [
                    sys.executable,
                    script,
                    "validate",
                    "--scene_dir",
                    scene_dir,
                    "--expected_scene",
                    "garden",
                    "--expected_profile",
                    "training",
                ],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(create_result.returncode, 0, create_result.stderr)
            self.assertEqual(
                validate_result.returncode,
                0,
                validate_result.stderr,
            )
            self.assertIn("content_sha256", validate_result.stdout)

    def test_preprocess_requires_fresh_output_and_writes_manifest_after_policy(self):
        source_path = os.path.join(
            PROJECT_ROOT,
            "NTU4DRadLM_pre_processing",
            "NTU4DRadLM_pre_processing.py",
        )
        with open(source_path, encoding="utf-8") as handle:
            source = handle.read()

        self.assertIn("def ensure_fresh_scene_output", source)
        self.assertLess(
            source.index("ensure_fresh_scene_output(scene_out_path)"),
            source.index("ensure_dir(os.path.join(scene_out_path"),
        )
        self.assertLess(
            source.index('"preprocess_policy.json"'),
            source.index("write_scene_manifest_atomic("),
        )
        self.assertIn("if failures:", source)
        self.assertIn("raise SystemExit(1)", source)

    def test_formal_launchers_validate_manifest_without_skip_switch(self):
        relative_paths = (
            "diffusion_consistency_radar/launch/train_unified.sh",
            "diffusion_consistency_radar/launch/inference_ldm.sh",
            "diffusion_consistency_radar/launch/inference_cd.sh",
            "diffusion_consistency_radar/launch/inference_uniified.sh",
        )
        for relative_path in relative_paths:
            with self.subTest(path=relative_path):
                with open(
                    os.path.join(PROJECT_ROOT, relative_path),
                    encoding="utf-8",
                ) as handle:
                    script = handle.read()
                self.assertNotIn("SKIP_MANIFEST", script)
                if relative_path.endswith("train_unified.sh"):
                    self.assertIn(
                        'MANIFEST_SCRIPT="${PROJECT_DIR}/scripts/dataset_manifest.py"',
                        script,
                    )
                    self.assertIn('"${MANIFEST_SCRIPT}" validate', script)
                    self.assertIn("--expected_profile training", script)
                    self.assertNotIn(".tmp_train_dataset", script)
                    self.assertNotIn('rm -rf "${TRAIN_DATASET_DIR}"', script)
                else:
                    self.assertIn(
                        'DEPLOYMENT_VIEW_SCRIPT="${PROJECT_DIR}/scripts/build_deployment_view.py"',
                        script,
                    )
                    self.assertIn('"${DEPLOYMENT_VIEW_SCRIPT}"', script)
                    self.assertIn("validate --dataset_dir", script)
                    validation_index = script.index('"${DEPLOYMENT_VIEW_SCRIPT}"')
                    self.assertLess(
                        validation_index,
                        script.index('python "${INFER_SCRIPT}"'),
                    )


if __name__ == "__main__":
    unittest.main()
