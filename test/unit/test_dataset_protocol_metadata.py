import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def write_sparse(path, voxel):
    occupied = voxel[..., 0] > 0
    np.savez_compressed(
        path,
        coords=np.column_stack(np.where(occupied)),
        features=voxel[occupied],
        shape=voxel.shape,
    )


class DatasetProtocolMetadataTest(unittest.TestCase):
    @staticmethod
    def _radar_normalization_spec(target_size, source_pc_range, model_pc_range):
        return {
            "protocol": "radar_normalization_v1",
            "formal": True,
            "training_scenes": ["garden"],
            "frame_count": 1,
            "target_size": list(target_size),
            "source_pc_range": list(source_pc_range),
            "model_pc_range": list(model_pc_range),
            "intensity": {
                "transform": "log1p_robust_zscore",
                "log_median": 0.0,
                "log_iqr": 1.0,
                "clip": [-5.0, 5.0],
            },
            "doppler": {
                "transform": "symmetric_physical_scale",
                "scale_mps": 4.0,
                "clip": [-1.0, 1.0],
            },
            "variance": {
                "transform": "identity",
                "unit": "m2_s2",
                "aggregation": "occupied_voxel_equal_weight_total_variance",
            },
            "input_provenance": {
                "dataset_manifest_sha256": {"garden": "a" * 64},
            },
        }

    @staticmethod
    def _write_dense_scene(root):
        scene = os.path.join(root, "garden")
        radar_dir = os.path.join(scene, "radar_voxel")
        target_dir = os.path.join(scene, "target_voxel")
        os.makedirs(radar_dir)
        os.makedirs(target_dir)
        voxel = np.zeros((2, 1, 1, 4), dtype=np.float32)
        voxel[..., 0] = 1.0
        np.save(os.path.join(radar_dir, "000000.npy"), voxel)
        np.save(os.path.join(target_dir, "000000.npy"), voxel)

    def test_dataset_requires_normalization_unless_legacy_is_explicit(self):
        """Dataset 默认 fail-closed，只有显式诊断开关保留原始量纲。"""
        from diffusion_consistency_radar.cm.dataset_loader import (
            NTU4DRadLM_VoxelDataset,
        )
        from diffusion_consistency_radar.radar_normalization import (
            RadarNormalizationError,
        )

        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RadarNormalizationError, "normalization|归一化"):
                NTU4DRadLM_VoxelDataset(
                    root,
                    split="train",
                    use_augmentation=False,
                )

            self._write_dense_scene(root)
            dataset = NTU4DRadLM_VoxelDataset(
                root,
                split="train",
                use_augmentation=False,
                target_size=(1, 2, 1),
                source_pc_range=(0, -1, -1, 2, 1, 1),
                allow_legacy_radar_units=True,
            )
            _target, radar, meta = dataset[0]

        self.assertEqual(tuple(radar.shape), (4, 1, 2, 1))
        self.assertTrue(bool(meta["legacy_radar_units"]))
        self.assertEqual(meta["radar_normalization_protocol"], "legacy_identity")
        self.assertEqual(meta["radar_normalization_sha256"], "")

    def test_dataset_rejects_ineffective_temporal_and_transform_arguments(self):
        """未实现的参数不得继续被静默接受为已生效功能。"""
        from diffusion_consistency_radar.cm.dataset_loader import (
            NTU4DRadLM_VoxelDataset,
        )

        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "sequence_length.*1"):
                NTU4DRadLM_VoxelDataset(
                    root,
                    sequence_length=2,
                    allow_legacy_radar_units=True,
                )
            with self.assertRaisesRegex(ValueError, "transform.*未实现"):
                NTU4DRadLM_VoxelDataset(
                    root,
                    transform=lambda value: value,
                    allow_legacy_radar_units=True,
                )
            with self.assertRaisesRegex(ValueError, "alignment_size.*未实现"):
                NTU4DRadLM_VoxelDataset(
                    root,
                    alignment_size=16,
                    allow_legacy_radar_units=True,
                )

    def test_dataset_normalizes_after_physical_augmentation_and_records_hash(self):
        """已知 m/s shift 应先施加，再由冻结 scale 转成网络量纲。"""
        from diffusion_consistency_radar.cm.dataset_loader import (
            NTU4DRadLM_VoxelDataset,
        )

        target_size = (1, 2, 1)
        source_range = (0, -1, -1, 2, 1, 1)
        spec = self._radar_normalization_spec(
            target_size,
            source_range,
            source_range,
        )

        class PhysicalDopplerShift:
            def __call__(self, target, condition, observed_mask):
                shifted = condition.clone()
                shifted[2:3] += 4.0 * (shifted[0:1] > 0)
                return target, shifted, observed_mask

        with tempfile.TemporaryDirectory() as root:
            self._write_dense_scene(root)
            dataset = NTU4DRadLM_VoxelDataset(
                root,
                split="train",
                use_augmentation=False,
                target_size=target_size,
                source_pc_range=source_range,
                radar_normalization=spec,
                radar_normalization_sha256="b" * 64,
            )
            dataset.augmentation = PhysicalDopplerShift()
            _target, radar, meta = dataset[0]

        torch.testing.assert_close(radar[2], torch.ones_like(radar[2]))
        self.assertFalse(bool(meta["legacy_radar_units"]))
        self.assertEqual(meta["radar_normalization_protocol"], "radar_normalization_v1")
        self.assertEqual(meta["radar_normalization_sha256"], "b" * 64)

    def test_lidar_observed_mask_raycast_marks_visible_path_only(self):
        from diffusion_consistency_radar.cm.dataset_loader import (
            build_lidar_observed_mask,
        )

        lidar = np.zeros((4, 3, 3, 4), dtype=np.float32)
        lidar[3, 1, 1, 0] = 1.0
        mask = build_lidar_observed_mask(
            lidar,
            pc_range=(0.0, -1.0, -1.0, 4.0, 2.0, 2.0),
        )

        self.assertTrue(bool(mask[3, 1, 1]))
        self.assertTrue(bool(mask[0, 1, 1]))
        self.assertFalse(bool(mask[3, 0, 1]))
        self.assertFalse(bool(mask[2, 2, 2]))

    def test_dataset_exposes_lidar_observed_mask_and_fallback_is_occupied_only(self):
        from diffusion_consistency_radar.cm.dataset_loader import NTU4DRadLM_VoxelDataset

        with tempfile.TemporaryDirectory() as tmp:
            scene = os.path.join(tmp, "garden")
            radar_dir = os.path.join(scene, "radar_voxel")
            target_dir = os.path.join(scene, "target_voxel")
            lidar_dir = os.path.join(scene, "lidar_voxel")
            os.makedirs(radar_dir)
            os.makedirs(target_dir)
            os.makedirs(lidar_dir)

            voxel = np.zeros((4, 3, 3, 4), dtype=np.float32)
            voxel[3, 1, 1, 0] = 1.0
            write_sparse(os.path.join(radar_dir, "000000.npz"), voxel)
            write_sparse(os.path.join(target_dir, "000000.npz"), voxel)
            write_sparse(os.path.join(lidar_dir, "000000.npz"), voxel)

            ds = NTU4DRadLM_VoxelDataset(
                tmp,
                split="train",
                use_augmentation=False,
                target_size=(3, 4, 3),
                source_pc_range=(0.0, -1.0, -1.0, 4.0, 2.0, 2.0),
                allow_legacy_radar_units=True,
            )
            _target, _radar, meta = ds[0]

            self.assertEqual(meta["occupancy_observed_mask_source"], "lidar_ray")
            observed = meta["occupancy_observed_mask"]
            self.assertEqual(tuple(observed.shape), (1, 3, 4, 3))
            self.assertGreater(float(observed.sum()), 1.0)

    def test_dataset_crops_physical_near_field_before_high_resolution_resize(self):
        from diffusion_consistency_radar.cm.dataset_loader import NTU4DRadLM_VoxelDataset

        with tempfile.TemporaryDirectory() as tmp:
            scene = os.path.join(tmp, "garden")
            radar_dir = os.path.join(scene, "radar_voxel")
            target_dir = os.path.join(scene, "target_voxel")
            os.makedirs(radar_dir)
            os.makedirs(target_dir)

            voxel = np.zeros((6, 4, 4, 4), dtype=np.float32)
            voxel[1, 1, 1, 0] = 1.0
            voxel[4, 1, 1, 0] = 1.0
            write_sparse(os.path.join(radar_dir, "000000.npz"), voxel)
            write_sparse(os.path.join(target_dir, "000000.npz"), voxel)

            ds = NTU4DRadLM_VoxelDataset(
                tmp,
                split="train",
                use_augmentation=False,
                target_size=(4, 3, 4),
                source_pc_range=(0.0, -2.0, -2.0, 6.0, 2.0, 2.0),
                model_pc_range=(0.0, -2.0, -2.0, 3.0, 2.0, 2.0),
                allow_legacy_radar_units=True,
            )

            target, radar, meta = ds[0]

            self.assertEqual(tuple(target.shape), (4, 4, 3, 4))
            self.assertEqual(tuple(radar.shape), (4, 4, 3, 4))
            self.assertEqual(int((target[0] > 0).sum()), 1)
            self.assertEqual(meta["model_pc_range"], [0.0, -2.0, -2.0, 3.0, 2.0, 2.0])

    def test_inference_voxel_loader_uses_same_near_field_crop(self):
        from diffusion_consistency_radar.scripts.inference import load_voxel_as_czxy

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "000000.npz")
            voxel = np.zeros((6, 4, 4, 4), dtype=np.float32)
            voxel[1, 1, 1, 0] = 1.0
            voxel[4, 1, 1, 0] = 1.0
            write_sparse(path, voxel)

            loaded = load_voxel_as_czxy(
                path,
                device="cpu",
                target_size=(4, 3, 4),
                source_pc_range=(0.0, -2.0, -2.0, 6.0, 2.0, 2.0),
                model_pc_range=(0.0, -2.0, -2.0, 3.0, 2.0, 2.0),
            )

            self.assertEqual(tuple(loaded.shape), (4, 4, 3, 4))
            self.assertEqual(int((loaded[0] > 0).sum()), 1)

    def test_non_scene_config_directory_is_excluded_from_scene_split(self):
        from diffusion_consistency_radar.cm.dataset_loader import NTU4DRadLM_VoxelDataset

        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "config"))
            scene = os.path.join(tmp, "garden")
            radar = os.path.join(scene, "radar_voxel")
            target = os.path.join(scene, "target_voxel")
            os.makedirs(radar)
            os.makedirs(target)

            voxel = np.zeros((2, 2, 2, 4), dtype=np.float32)
            voxel[0, 0, 0, 0] = 1.0
            write_sparse(os.path.join(radar, "000000.npz"), voxel)
            write_sparse(os.path.join(target, "000000.npz"), voxel)

            ds = NTU4DRadLM_VoxelDataset(
                tmp,
                split="train",
                use_augmentation=False,
                allow_legacy_radar_units=True,
            )

            self.assertEqual(len(ds), 1)
            self.assertTrue(ds.samples[0][1].endswith("garden/target_voxel/000000.npz"))

    def test_dataset_reports_policy_mock_flags_and_real_ir(self):
        from diffusion_consistency_radar.cm.dataset_loader import NTU4DRadLM_VoxelDataset

        with tempfile.TemporaryDirectory() as tmp:
            scene = os.path.join(tmp, "loop3")
            radar = os.path.join(scene, "radar_voxel")
            target = os.path.join(scene, "target_voxel")
            ir = os.path.join(scene, "ir_image")
            os.makedirs(radar)
            os.makedirs(target)
            os.makedirs(ir)

            with open(os.path.join(scene, "preprocess_policy.json"), "w", encoding="utf-8") as f:
                json.dump({"pc_range": [0, -20, -6, 120, 20, 10], "z_min": -1.0}, f)

            voxel = np.zeros((4, 4, 4, 4), dtype=np.float32)
            voxel[1, 1, 1, 0] = 1.0
            write_sparse(os.path.join(radar, "000000.npz"), voxel)
            write_sparse(os.path.join(target, "000000.npz"), voxel)
            np.save(os.path.join(ir, "000000_ir.npy"), np.ones((8, 8), dtype=np.float32))

            ds = NTU4DRadLM_VoxelDataset(
                tmp,
                split="train",
                use_augmentation=False,
                allow_legacy_radar_units=True,
            )
            _target, _radar, meta = ds[0]

            self.assertFalse(bool(meta["is_mock_ir"]))
            self.assertTrue(bool(meta["is_mock_calib"]))
            self.assertEqual(meta["calib_source"], "mock_default")
            self.assertFalse(bool(meta["calib_is_thermal"]))
            self.assertEqual(meta["preprocess_policy"]["z_min"], -1.0)
            self.assertEqual(tuple(meta["ir_img"].shape), (3, 480, 640))

    def test_voxel_collator_preserves_nullable_preprocess_policy(self):
        """审计 policy 的 JSON null 不应阻断模型 batch 拼接。"""
        from diffusion_consistency_radar.cm.dataset_loader import (
            collate_voxel_samples,
        )

        target = torch.zeros((4, 2, 2, 2), dtype=torch.float32)
        radar = torch.ones_like(target)
        samples = []
        for index in range(2):
            samples.append(
                (
                    target + index,
                    radar + index,
                    {
                        "occupancy_observed_mask": torch.ones((1, 2, 2, 2)),
                        "is_mock_ir": False,
                        "preprocess_policy": {
                            "velocity_mode": "none",
                            "v_drone": None,
                        },
                    },
                )
            )

        batch_target, batch_radar, batch_meta = collate_voxel_samples(samples)

        self.assertEqual(tuple(batch_target.shape), (2, 4, 2, 2, 2))
        self.assertEqual(tuple(batch_radar.shape), (2, 4, 2, 2, 2))
        self.assertEqual(
            tuple(batch_meta["occupancy_observed_mask"].shape),
            (2, 1, 2, 2, 2),
        )
        self.assertEqual(len(batch_meta["preprocess_policy"]), 2)
        self.assertIsNone(batch_meta["preprocess_policy"][0]["v_drone"])

    def test_missing_ir_uses_mock_without_inventing_sync_displacement(self):
        from diffusion_consistency_radar.cm.dataset_loader import NTU4DRadLM_VoxelDataset

        with tempfile.TemporaryDirectory() as tmp:
            scene = os.path.join(tmp, "loop3")
            radar = os.path.join(scene, "radar_voxel")
            target = os.path.join(scene, "target_voxel")
            os.makedirs(radar)
            os.makedirs(target)

            voxel = np.zeros((4, 4, 4, 4), dtype=np.float32)
            voxel[1, 1, 1, 0] = 1.0
            write_sparse(os.path.join(radar, "000000.npz"), voxel)
            write_sparse(os.path.join(target, "000000.npz"), voxel)

            ds = NTU4DRadLM_VoxelDataset(
                tmp,
                split="train",
                use_augmentation=False,
                allow_legacy_radar_units=True,
            )
            _target, _radar, meta = ds[0]

            self.assertTrue(bool(meta["is_mock_ir"]))
            self.assertTrue(bool(meta["is_mock_calib"]))
            self.assertAlmostEqual(float(meta["t_vec"][0]), 0.0, places=6)
            self.assertEqual(
                meta["time_alignment_compensation"],
                "preprocessing_signed_delta_only",
            )
            self.assertEqual(meta["calib_fallback_reason"], "thermal_missing")

    def test_livox_calibration_is_recorded_but_not_used_as_real_thermal_calib(self):
        from diffusion_consistency_radar.cm.dataset_loader import NTU4DRadLM_VoxelDataset

        with tempfile.TemporaryDirectory() as tmp:
            config = os.path.join(tmp, "config")
            scene = os.path.join(tmp, "loop3")
            radar = os.path.join(scene, "radar_voxel")
            target = os.path.join(scene, "target_voxel")
            os.makedirs(config)
            os.makedirs(radar)
            os.makedirs(target)

            with open(os.path.join(config, "calib_radar_to_livox.txt"), "w", encoding="utf-8") as f:
                f.write("R: 1 0 0 0 1 0 0 0 1\n")
                f.write("T: 1 2 3\n")

            voxel = np.zeros((4, 4, 4, 4), dtype=np.float32)
            voxel[1, 1, 1, 0] = 1.0
            write_sparse(os.path.join(radar, "000000.npz"), voxel)
            write_sparse(os.path.join(target, "000000.npz"), voxel)

            ds = NTU4DRadLM_VoxelDataset(
                tmp,
                split="train",
                use_augmentation=False,
                allow_legacy_radar_units=True,
            )
            _target, _radar, meta = ds[0]

            self.assertTrue(bool(meta["is_mock_calib"]))
            self.assertTrue(bool(meta["has_livox_calib"]))
            self.assertFalse(bool(meta["has_thermal_calib"]))
            self.assertEqual(meta["calib_source"], "mock_default")
            self.assertEqual(meta["calib_fallback_reason"], "thermal_missing_livox_available_not_used_for_ir")
            self.assertAlmostEqual(float(meta["t_vec"][0]), 0.0, places=6)

    def test_thermal_calibration_is_used_as_real_ir_calib(self):
        from diffusion_consistency_radar.cm.dataset_loader import NTU4DRadLM_VoxelDataset

        with tempfile.TemporaryDirectory() as tmp:
            config = os.path.join(tmp, "config")
            scene = os.path.join(tmp, "loop3")
            radar = os.path.join(scene, "radar_voxel")
            target = os.path.join(scene, "target_voxel")
            os.makedirs(config)
            os.makedirs(radar)
            os.makedirs(target)

            with open(os.path.join(config, "calib_livox_to_thermal.txt"), "w", encoding="utf-8") as f:
                f.write("R: 1 0 0 0 1 0 0 0 1\n")
                f.write("T: 1 2 3\n")

            voxel = np.zeros((4, 4, 4, 4), dtype=np.float32)
            voxel[1, 1, 1, 0] = 1.0
            write_sparse(os.path.join(radar, "000000.npz"), voxel)
            write_sparse(os.path.join(target, "000000.npz"), voxel)

            ds = NTU4DRadLM_VoxelDataset(
                tmp,
                split="train",
                use_augmentation=False,
                allow_legacy_radar_units=True,
            )
            _target, _radar, meta = ds[0]

            self.assertFalse(bool(meta["is_mock_calib"]))
            self.assertEqual(meta["calib_source"], "calib_livox_to_thermal.txt")
            self.assertTrue(bool(meta["calib_is_thermal"]))
            self.assertAlmostEqual(float(meta["t_vec"][0]), 1.0, places=6)

    def test_explicit_scene_names_prevent_dataset_scene_guessing(self):
        from diffusion_consistency_radar.cm.dataset_loader import NTU4DRadLM_VoxelDataset

        with tempfile.TemporaryDirectory() as root:
            for scene_name in ("garden", "loop3"):
                scene = os.path.join(root, scene_name)
                radar = os.path.join(scene, "radar_voxel")
                target = os.path.join(scene, "target_voxel")
                os.makedirs(radar)
                os.makedirs(target)
                voxel = np.zeros((2, 2, 2, 4), dtype=np.float32)
                voxel[0, 0, 0, 0] = 1.0
                write_sparse(os.path.join(radar, "000000.npz"), voxel)
                write_sparse(os.path.join(target, "000000.npz"), voxel)

            dataset = NTU4DRadLM_VoxelDataset(
                root,
                split="train",
                scene_names=["loop3"],
                use_augmentation=False,
                allow_legacy_radar_units=True,
            )

        self.assertEqual(len(dataset), 1)
        self.assertIn("/loop3/", dataset.samples[0][1])

    def test_dataset_audit_reports_ir_coverage_and_mock_calibration(self):
        from diffusion_consistency_radar.scripts.audit_dataset_protocol import audit_scene

        with tempfile.TemporaryDirectory() as tmp:
            scene = os.path.join(tmp, "loop3")
            radar = os.path.join(scene, "radar_voxel")
            target = os.path.join(scene, "target_voxel")
            ir = os.path.join(scene, "ir_image")
            os.makedirs(radar)
            os.makedirs(target)
            os.makedirs(ir)

            with open(os.path.join(scene, "preprocess_policy.json"), "w", encoding="utf-8") as f:
                json.dump({"align_to": "radar"}, f)

            voxel = np.zeros((2, 2, 2, 4), dtype=np.float32)
            voxel[0, 0, 0, 0] = 1.0
            write_sparse(os.path.join(radar, "000000.npz"), voxel)
            write_sparse(os.path.join(target, "000000.npz"), voxel)
            np.save(os.path.join(ir, "000000_ir.npy"), np.ones((4, 4), dtype=np.float32))

            row = audit_scene(tmp, "loop3")

            self.assertEqual(row["radar_frames"], 1)
            self.assertEqual(row["target_frames"], 1)
            self.assertEqual(row["ir_frames"], 1)
            self.assertEqual(row["compatible_ir_frames"], 1)
            self.assertAlmostEqual(float(row["ir_coverage"]), 1.0)
            self.assertAlmostEqual(float(row["mock_ir_ratio"]), 0.0)
            self.assertTrue(row["has_preprocess_policy"])
            self.assertEqual(row["align_to"], "radar")
            self.assertTrue(row["is_mock_calib"])
            self.assertEqual(row["calib_source"], "mock_default")
            self.assertIn("ir_frustum_voxel_ratio", row)

    def test_dataset_audit_distinguishes_dataset_ir_name_from_compatible_names(self):
        from diffusion_consistency_radar.scripts.audit_dataset_protocol import audit_scene

        with tempfile.TemporaryDirectory() as tmp:
            scene = os.path.join(tmp, "loop3")
            radar = os.path.join(scene, "radar_voxel")
            target = os.path.join(scene, "target_voxel")
            ir = os.path.join(scene, "ir_image")
            os.makedirs(radar)
            os.makedirs(target)
            os.makedirs(ir)

            voxel = np.zeros((2, 2, 2, 4), dtype=np.float32)
            voxel[0, 0, 0, 0] = 1.0
            write_sparse(os.path.join(radar, "000000.npz"), voxel)
            write_sparse(os.path.join(target, "000000.npz"), voxel)
            np.save(os.path.join(ir, "000000.npy"), np.ones((4, 4), dtype=np.float32))

            row = audit_scene(tmp, "loop3")

            self.assertEqual(row["ir_frames"], 0)
            self.assertEqual(row["compatible_ir_frames"], 1)
            self.assertAlmostEqual(float(row["ir_coverage"]), 0.0)
            self.assertAlmostEqual(float(row["compatible_ir_coverage"]), 1.0)
            self.assertAlmostEqual(float(row["mock_ir_ratio"]), 1.0)


if __name__ == "__main__":
    unittest.main()
