import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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

            ds = NTU4DRadLM_VoxelDataset(tmp, split="train", use_augmentation=False)

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

            ds = NTU4DRadLM_VoxelDataset(tmp, split="train", use_augmentation=False)
            _target, _radar, meta = ds[0]

            self.assertFalse(bool(meta["is_mock_ir"]))
            self.assertTrue(bool(meta["is_mock_calib"]))
            self.assertEqual(meta["preprocess_policy"]["z_min"], -1.0)
            self.assertEqual(tuple(meta["ir_img"].shape), (3, 480, 640))

    def test_missing_ir_uses_mock_and_fallback_t_has_sync_compensation(self):
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

            ds = NTU4DRadLM_VoxelDataset(tmp, split="train", use_augmentation=False)
            _target, _radar, meta = ds[0]

            self.assertTrue(bool(meta["is_mock_ir"]))
            self.assertTrue(bool(meta["is_mock_calib"]))
            self.assertAlmostEqual(float(meta["t_vec"][0]), 0.01, places=6)

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
            self.assertAlmostEqual(float(row["ir_coverage"]), 1.0)
            self.assertTrue(row["has_preprocess_policy"])
            self.assertEqual(row["align_to"], "radar")
            self.assertTrue(row["is_mock_calib"])


if __name__ == "__main__":
    unittest.main()
