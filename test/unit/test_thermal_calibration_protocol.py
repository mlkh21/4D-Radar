# -*- coding: utf-8 -*-
"""测试 P1-02 的 thermal K/D/S 标定、去畸变和同步补偿共享协议。"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class ThermalCalibrationProtocolTest(unittest.TestCase):
    @staticmethod
    def _write_calibration(config_dir):
        with open(
            os.path.join(config_dir, "calib_radar_to_livox.txt"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write("R: 1 0 0 0 1 0 0 0 1\nT: 0.1 0.2 0.3\n")
        with open(
            os.path.join(config_dir, "calib_radar_to_thermal.txt"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write("R: 1 0 0 0 1 0 0 0 1\nT: 1.1 2.2 3.3\n")
        with open(
            os.path.join(config_dir, "calib_livox_to_thermal.txt"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write("R: 1 0 0 0 1 0 0 0 1\nT: 1 2 3\n")
        with open(
            os.path.join(config_dir, "calib_cam_thermal.txt"),
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(
                "S_00: 640 512\n"
                "K_00: 400 0 320 0 410 256 0 0 1\n"
                "D_00: 0.35 -0.12 0.001 0.002 0.04\n"
            )

    def test_provider_parses_and_scales_thermal_intrinsics(self):
        from diffusion_consistency_radar.cm.dataset_loader import CalibrationProvider

        with tempfile.TemporaryDirectory() as root:
            config_dir = os.path.join(root, "config")
            os.makedirs(config_dir)
            self._write_calibration(config_dir)

            _r_mat, _t_vec, k_mat, metadata = CalibrationProvider(
                root,
                calibration_dir=config_dir,
                require_real=True,
                voxel_coordinate_frame="lidar",
            ).load_with_metadata()

        self.assertAlmostEqual(float(k_mat[0, 0]), 400.0, places=5)
        self.assertAlmostEqual(float(k_mat[0, 2]), 320.0, places=5)
        self.assertAlmostEqual(float(k_mat[1, 1]), 410.0 * 480.0 / 512.0, places=5)
        self.assertAlmostEqual(float(k_mat[1, 2]), 256.0 * 480.0 / 512.0, places=5)
        self.assertEqual(metadata["thermal_source_size"], [640, 512])
        self.assertEqual(metadata["thermal_output_size"], [640, 480])
        np.testing.assert_allclose(
            metadata["thermal_distortion"],
            [0.35, -0.12, 0.001, 0.002, 0.04],
            rtol=1e-6,
            atol=1e-7,
        )
        self.assertEqual(metadata["thermal_intrinsics_source"], "calib_cam_thermal.txt")
        self.assertEqual(metadata["calib_source"], "calib_livox_to_thermal.txt")
        self.assertEqual(metadata["extrinsic_source_frame"], "lidar")
        self.assertTrue(metadata["calibration_closure_available"])
        self.assertAlmostEqual(
            metadata["calibration_closure_rotation_max_abs"],
            0.0,
        )
        self.assertAlmostEqual(
            metadata["calibration_closure_translation_l2_m"],
            0.0,
            places=5,
        )

    def test_ir_resize_applies_distortion_correction_before_tensor_conversion(self):
        from diffusion_consistency_radar.cm.dataset_loader import (
            _resize_or_pad_ir_tensor,
        )

        height, width = 512, 640
        yy, xx = np.indices((height, width), dtype=np.float32)
        image = (xx * 0.001 + yy * 0.002)[None]
        metadata = {
            "thermal_intrinsic_matrix": [
                [400.0, 0.0, 320.0],
                [0.0, 410.0 * 480.0 / 512.0, 256.0 * 480.0 / 512.0],
                [0.0, 0.0, 1.0],
            ],
            "thermal_distortion": [0.35, -0.12, 0.001, 0.002, 0.04],
            "thermal_source_size": [640, 512],
            "thermal_output_size": [640, 480],
        }
        corrected = _resize_or_pad_ir_tensor(torch.from_numpy(image), metadata)
        uncorrected = _resize_or_pad_ir_tensor(
            torch.from_numpy(image),
            dict(metadata, thermal_distortion=[0.0, 0.0, 0.0, 0.0, 0.0]),
        )

        self.assertEqual(tuple(corrected.shape), (3, 480, 640))
        self.assertTrue(torch.isfinite(corrected).all())
        self.assertGreater(float(torch.max(torch.abs(corrected - uncorrected))), 1e-5)

    def test_training_and_inference_do_not_apply_fixed_sync_compensation(self):
        from diffusion_consistency_radar.cm import dataset_loader
        from diffusion_consistency_radar.scripts import inference

        self.assertFalse(hasattr(dataset_loader, "apply_legacy_sync_compensation"))
        self.assertFalse(hasattr(inference, "apply_legacy_sync_compensation"))

    def test_formal_provider_requires_explicit_calibration_directory(self):
        from diffusion_consistency_radar.cm.dataset_loader import CalibrationProvider

        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "calibration_dir"):
                CalibrationProvider(root, require_real=True)

    def test_formal_provider_rejects_non_rotation_extrinsic(self):
        from diffusion_consistency_radar.cm.dataset_loader import CalibrationProvider

        with tempfile.TemporaryDirectory() as root:
            config_dir = os.path.join(root, "config")
            os.makedirs(config_dir)
            self._write_calibration(config_dir)
            with open(
                os.path.join(config_dir, "calib_livox_to_thermal.txt"),
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write("R: 2 0 0 0 1 0 0 0 1\nT: 1 2 3\n")

            with self.assertRaisesRegex(ValueError, "正交|determinant"):
                CalibrationProvider(
                    root,
                    calibration_dir=config_dir,
                    require_real=True,
                ).load_with_metadata()

    def test_formal_provider_rejects_missing_thermal_intrinsics(self):
        from diffusion_consistency_radar.cm.dataset_loader import CalibrationProvider

        with tempfile.TemporaryDirectory() as root:
            config_dir = os.path.join(root, "config")
            os.makedirs(config_dir)
            self._write_calibration(config_dir)
            os.remove(os.path.join(config_dir, "calib_cam_thermal.txt"))

            with self.assertRaisesRegex(RuntimeError, "calib_cam_thermal"):
                CalibrationProvider(
                    root,
                    calibration_dir=config_dir,
                    require_real=True,
                ).load_with_metadata()


if __name__ == "__main__":
    unittest.main()
