#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 Radar normalization artifact 生成器的训练集与发布边界。"""

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


class RadarNormalizationBuilderTest(unittest.TestCase):
    """只使用临时小体素验证统计顺序和副作用约束。"""

    @staticmethod
    def _write_scene(root, scene, intensity_frames):
        radar_dir = os.path.join(root, scene, "radar_voxel")
        os.makedirs(radar_dir)
        for index, intensities in enumerate(intensity_frames):
            values = np.asarray(intensities, dtype=np.float32)
            voxel = np.zeros((values.size, 1, 1, 4), dtype=np.float32)
            voxel[..., 0] = 1.0
            voxel[..., 1] = values.reshape(-1, 1, 1)
            np.save(os.path.join(radar_dir, f"{index:06d}.npy"), voxel)
        return radar_dir

    @staticmethod
    def _kwargs(dataset_dir, output, **overrides):
        kwargs = {
            "dataset_dir": dataset_dir,
            "scenes": ["garden"],
            "output_path": output,
            "target_size": (1, 2, 1),
            "source_pc_range": (0, -1, -1, 2, 1, 1),
            "model_pc_range": (0, -1, -1, 2, 1, 1),
            "doppler_scale_mps": 4.0,
            "max_frames": 0,
        }
        kwargs.update(overrides)
        return kwargs

    def test_builder_uses_only_split_train_frames_and_actual_transform_order(self):
        """只统计唯一 split 的 train frame，并在 crop/resize 后计算分位数。"""
        from diffusion_consistency_radar.scripts import build_radar_normalization

        with tempfile.TemporaryDirectory() as root:
            self._write_scene(
                root,
                "garden",
                [
                    [0.0, np.expm1(1.0)],
                    [np.expm1(2.0), np.expm1(3.0)],
                ],
            )
            self._write_scene(root, "loop3", [[np.expm1(20.0), np.expm1(21.0)]])
            output = os.path.join(root, "artifacts", "radar_normalization.json")
            manifest = {"frame_count": 2, "content_sha256": "a" * 64}

            split_artifact = {
                "scenes": {
                    "garden": {
                        "train_frame_ids": ["000000", "000001"],
                    }
                }
            }
            with mock.patch.object(
                build_radar_normalization,
                "validate_scene_manifest",
                return_value=manifest,
            ) as validate, mock.patch.object(
                build_radar_normalization,
                "load_temporal_split_artifact",
                return_value=(split_artifact, "d" * 64),
            ):
                result_path = build_radar_normalization.build_and_write_artifact(
                    **self._kwargs(
                        root,
                        output,
                        split_artifact_path=os.path.join(root, "split.json"),
                    )
                )

            with open(output, encoding="utf-8") as handle:
                artifact = json.load(handle)

        self.assertEqual(result_path, output)
        validate.assert_called_once_with(
            os.path.join(root, "garden"),
            "garden",
            expected_profile="training",
        )
        self.assertEqual(artifact["training_scenes"], ["garden"])
        self.assertEqual(artifact["frame_count"], 2)
        self.assertTrue(artifact["formal"])
        self.assertAlmostEqual(artifact["intensity"]["log_median"], 1.5, places=5)
        self.assertAlmostEqual(artifact["intensity"]["log_iqr"], 1.5, places=5)
        self.assertEqual(
            artifact["input_provenance"]["dataset_manifest_sha256"],
            {"garden": "a" * 64},
        )
        self.assertEqual(
            artifact["input_provenance"]["split_artifact_sha256"],
            "d" * 64,
        )

    def test_frame_cap_marks_artifact_nonformal(self):
        """抽样 artifact 必须显式标为非正式，不能混入正式链。"""
        from diffusion_consistency_radar.scripts import build_radar_normalization

        with tempfile.TemporaryDirectory() as root:
            self._write_scene(
                root,
                "garden",
                [[0.0, np.expm1(1.0)], [np.expm1(2.0), np.expm1(3.0)]],
            )
            output = os.path.join(root, "sampled.json")
            manifest = {"frame_count": 2, "content_sha256": "b" * 64}
            with mock.patch.object(
                build_radar_normalization,
                "validate_scene_manifest",
                return_value=manifest,
            ):
                build_radar_normalization.build_and_write_artifact(
                    **self._kwargs(root, output, max_frames=1)
                )
            with open(output, encoding="utf-8") as handle:
                artifact = json.load(handle)

        self.assertFalse(artifact["formal"])
        self.assertEqual(artifact["frame_count"], 1)

    def test_existing_or_symlink_output_is_rejected_without_replacement(self):
        """已有目标和 symlink 都不得被覆盖。"""
        from diffusion_consistency_radar.radar_normalization import (
            RadarNormalizationError,
        )
        from diffusion_consistency_radar.scripts import build_radar_normalization

        with tempfile.TemporaryDirectory() as root:
            self._write_scene(root, "garden", [[0.0, np.expm1(1.0)]])
            output = os.path.join(root, "existing.json")
            with open(output, "w", encoding="utf-8") as handle:
                handle.write("sentinel")
            with self.assertRaisesRegex(RadarNormalizationError, "存在|覆盖"):
                build_radar_normalization.build_and_write_artifact(
                    **self._kwargs(root, output)
                )
            with open(output, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "sentinel")

            target = os.path.join(root, "target.json")
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("target")
            link = os.path.join(root, "link.json")
            os.symlink(target, link)
            with self.assertRaisesRegex(RadarNormalizationError, "符号链接|symlink"):
                build_radar_normalization.build_and_write_artifact(
                    **self._kwargs(root, link)
                )

    def test_invalid_scale_or_empty_training_occupancy_writes_nothing(self):
        """参数或统计无效时不得留下 artifact 半成品。"""
        from diffusion_consistency_radar.radar_normalization import (
            RadarNormalizationError,
        )
        from diffusion_consistency_radar.scripts import build_radar_normalization

        with tempfile.TemporaryDirectory() as root:
            self._write_scene(root, "garden", [[0.0, np.expm1(1.0)]])
            invalid_output = os.path.join(root, "invalid", "artifact.json")
            with self.assertRaisesRegex(RadarNormalizationError, "scale|量程"):
                build_radar_normalization.build_and_write_artifact(
                    **self._kwargs(root, invalid_output, doppler_scale_mps=0.0)
                )
            self.assertFalse(os.path.lexists(invalid_output))

        with tempfile.TemporaryDirectory() as root:
            radar_dir = os.path.join(root, "garden", "radar_voxel")
            os.makedirs(radar_dir)
            np.save(
                os.path.join(radar_dir, "000000.npy"),
                np.zeros((2, 1, 1, 4), dtype=np.float32),
            )
            empty_output = os.path.join(root, "empty", "artifact.json")
            manifest = {"frame_count": 1, "content_sha256": "c" * 64}
            with mock.patch.object(
                build_radar_normalization,
                "validate_scene_manifest",
                return_value=manifest,
            ):
                with self.assertRaisesRegex(RadarNormalizationError, "occupied|占据"):
                    build_radar_normalization.build_and_write_artifact(
                        **self._kwargs(root, empty_output)
                    )
            self.assertFalse(os.path.lexists(empty_output))


if __name__ == "__main__":
    unittest.main()
