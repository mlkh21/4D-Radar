#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 Radar 归一化与 Doppler 方差重采样的严格协议。"""

import os
import sys
import hashlib
import json
import tempfile
import unittest
from unittest import mock

import torch
import yaml


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class RadarNormalizationProtocolTest(unittest.TestCase):
    """覆盖 Radar 四通道物理语义的最小确定性用例。"""

    @staticmethod
    def _valid_spec():
        return {
            "protocol": "radar_normalization_v1",
            "formal": True,
            "training_scenes": ["garden"],
            "frame_count": 2,
            "target_size": [1, 1, 3],
            "source_pc_range": [0.0, -2.0, -1.0, 10.0, 2.0, 1.0],
            "model_pc_range": [0.0, -1.0, -1.0, 5.0, 1.0, 1.0],
            "intensity": {
                "transform": "log1p_robust_zscore",
                "log_median": 1.0,
                "log_iqr": 2.0,
                "clip": [-0.5, 0.5],
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
    def _write_spec(root, spec, name="radar_normalization.json"):
        path = os.path.join(root, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(spec, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    def test_resize_radar_variance_uses_total_variance_formula(self):
        """局部方差为零时，合并方差仍须包含细体素均值差。"""
        from diffusion_consistency_radar.cm.dataset_loader import (
            resize_radar_voxel_channels,
        )

        radar = torch.zeros(4, 1, 1, 2, dtype=torch.float32)
        radar[0, 0, 0] = torch.tensor([1.0, 1.0])
        radar[1, 0, 0] = torch.tensor([10.0, 20.0])
        radar[2, 0, 0] = torch.tensor([1.0, 3.0])

        resized = resize_radar_voxel_channels(radar, (1, 1, 1))

        self.assertEqual(tuple(resized.shape), (4, 1, 1, 1))
        self.assertAlmostEqual(float(resized[0, 0, 0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(resized[1, 0, 0, 0]), 15.0, places=6)
        self.assertAlmostEqual(float(resized[2, 0, 0, 0]), 2.0, places=6)
        self.assertAlmostEqual(float(resized[3, 0, 0, 0]), 1.0, places=6)

    def test_resize_radar_variance_combines_local_and_between_voxel_terms(self):
        """按 E[var+mean²]-E[mean]² 合并局部与组间方差。"""
        from diffusion_consistency_radar.cm.dataset_loader import (
            resize_radar_voxel_channels,
        )

        radar = torch.zeros(4, 1, 1, 2, dtype=torch.float32)
        radar[0, 0, 0] = 1.0
        radar[2, 0, 0] = torch.tensor([1.0, 3.0])
        radar[3, 0, 0] = torch.tensor([4.0, 0.0])

        resized = resize_radar_voxel_channels(radar, (1, 1, 1))

        # E[var+mean²]=(4+1+0+9)/2=7，E[mean]²=4。
        self.assertAlmostEqual(float(resized[3, 0, 0, 0]), 3.0, places=6)
        self.assertGreaterEqual(float(resized[3].min()), 0.0)

    def test_resize_radar_uses_same_bins_for_sparse_occupancy_and_attributes(self):
        """稀疏点偏离插值中心时，occupied 输出仍须保留对应物理属性。"""
        from diffusion_consistency_radar.cm.dataset_loader import (
            resize_radar_voxel_channels,
        )

        radar = torch.zeros(4, 1, 1, 8, dtype=torch.float32)
        radar[0, 0, 0, [0, 7]] = 1.0
        radar[1, 0, 0, [0, 7]] = torch.tensor([10.0, 20.0])
        radar[2, 0, 0, [0, 7]] = torch.tensor([1.0, 3.0])
        radar[3, 0, 0, [0, 7]] = torch.tensor([4.0, 0.0])

        resized = resize_radar_voxel_channels(radar, (1, 1, 2))

        torch.testing.assert_close(resized[0, 0, 0], torch.ones(2))
        torch.testing.assert_close(
            resized[1, 0, 0], torch.tensor([10.0, 20.0])
        )
        torch.testing.assert_close(
            resized[2, 0, 0], torch.tensor([1.0, 3.0])
        )
        torch.testing.assert_close(
            resized[3, 0, 0], torch.tensor([4.0, 0.0])
        )

    def test_resize_radar_keeps_empty_output_zero_and_input_unchanged(self):
        """空体素特征必须清零，调用不得原地修改输入。"""
        from diffusion_consistency_radar.cm.dataset_loader import (
            resize_radar_voxel_channels,
        )

        radar = torch.tensor(
            [
                [[[[1.0, 0.0]]]],
                [[[[5.0, 99.0]]]],
                [[[[2.0, 88.0]]]],
                [[[[0.25, 77.0]]]],
            ],
            dtype=torch.float32,
        ).reshape(4, 1, 1, 2)
        original = radar.clone()

        resized = resize_radar_voxel_channels(radar, (1, 1, 2))

        torch.testing.assert_close(radar, original)
        torch.testing.assert_close(resized[:, 0, 0, 1], torch.zeros(4))

    def test_resize_radar_rejects_shape_nonfinite_and_negative_variance(self):
        """Radar 专用接口拒绝未知通道、非有限值和 occupied 负方差。"""
        from diffusion_consistency_radar.cm.dataset_loader import (
            resize_radar_voxel_channels,
        )

        with self.assertRaisesRegex(ValueError, "四通道|\(4"):
            resize_radar_voxel_channels(torch.zeros(5, 1, 1, 1), (1, 1, 1))

        nonfinite = torch.zeros(4, 1, 1, 1)
        nonfinite[1, 0, 0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "有限"):
            resize_radar_voxel_channels(nonfinite, (1, 1, 1))

        negative_variance = torch.zeros(4, 1, 1, 1)
        negative_variance[0, 0, 0, 0] = 1.0
        negative_variance[3, 0, 0, 0] = -0.1
        with self.assertRaisesRegex(ValueError, "variance|方差"):
            resize_radar_voxel_channels(negative_variance, (1, 1, 1))

    def test_apply_normalization_uses_robust_intensity_and_physical_doppler(self):
        """强度使用 log 稳健缩放，Doppler 使用显式物理量程。"""
        from diffusion_consistency_radar.radar_normalization import (
            apply_radar_normalization,
        )

        radar = torch.zeros(4, 1, 1, 3, dtype=torch.float32)
        radar[0, 0, 0] = torch.tensor([1.0, 1.0, 0.0])
        radar[1, 0, 0] = torch.tensor([torch.expm1(torch.tensor(3.0)), 0.0, 99.0])
        radar[2, 0, 0] = torch.tensor([8.0, -8.0, 77.0])
        radar[3, 0, 0] = torch.tensor([0.25, 1.0, 66.0])
        original = radar.clone()

        normalized = apply_radar_normalization(radar, self._valid_spec())

        torch.testing.assert_close(radar, original)
        torch.testing.assert_close(
            normalized[:, 0, 0, 0],
            torch.tensor([1.0, 0.5, 1.0, 0.25]),
        )
        torch.testing.assert_close(
            normalized[:, 0, 0, 1],
            torch.tensor([1.0, -0.5, -1.0, 1.0]),
        )
        torch.testing.assert_close(normalized[:, 0, 0, 2], torch.zeros(4))

    def test_loader_returns_full_spec_and_exact_file_sha256(self):
        """loader 返回完整 JSON 内容和实际 artifact 文件字节 hash。"""
        from diffusion_consistency_radar.radar_normalization import (
            load_radar_normalization_artifact,
        )

        spec = self._valid_spec()
        with tempfile.TemporaryDirectory() as root:
            path = self._write_spec(root, spec)
            with open(path, "rb") as handle:
                expected_sha256 = hashlib.sha256(handle.read()).hexdigest()

            loaded, digest = load_radar_normalization_artifact(
                path,
                target_size=(1, 1, 3),
                source_pc_range=(0, -2, -1, 10, 2, 1),
                model_pc_range=(0, -1, -1, 5, 1, 1),
                doppler_scale_mps=4.0,
            )

        self.assertEqual(loaded, spec)
        self.assertEqual(digest, expected_sha256)

    def test_loader_rejects_invalid_artifacts_and_contract_mismatches(self):
        """正式 loader 对非法 schema、非正式统计和接口不匹配 fail-closed。"""
        from diffusion_consistency_radar.radar_normalization import (
            RadarNormalizationError,
            load_radar_normalization_artifact,
        )

        cases = []
        missing = self._valid_spec()
        missing.pop("variance")
        cases.append(("missing", missing, {}, "字段|variance"))
        nonfinite = self._valid_spec()
        nonfinite["intensity"]["log_median"] = float("nan")
        cases.append(("nonfinite", nonfinite, {}, "有限|finite"))
        zero_iqr = self._valid_spec()
        zero_iqr["intensity"]["log_iqr"] = 0.0
        cases.append(("zero_iqr", zero_iqr, {}, "IQR|iqr"))
        nonformal = self._valid_spec()
        nonformal["formal"] = False
        cases.append(("nonformal", nonformal, {}, "formal|正式"))
        cases.append(
            (
                "grid_mismatch",
                self._valid_spec(),
                {"target_size": (2, 1, 3)},
                "target_size|网格",
            )
        )
        cases.append(
            (
                "scale_mismatch",
                self._valid_spec(),
                {"doppler_scale_mps": 5.0},
                "scale|量程",
            )
        )

        with tempfile.TemporaryDirectory() as root:
            for name, spec, overrides, message in cases:
                with self.subTest(name=name):
                    path = self._write_spec(root, spec, f"{name}.json")
                    kwargs = {
                        "target_size": (1, 1, 3),
                        "source_pc_range": (0, -2, -1, 10, 2, 1),
                        "model_pc_range": (0, -1, -1, 5, 1, 1),
                        "doppler_scale_mps": 4.0,
                    }
                    kwargs.update(overrides)
                    with self.assertRaisesRegex(RadarNormalizationError, message):
                        load_radar_normalization_artifact(path, **kwargs)

            target = self._write_spec(root, self._valid_spec(), "target.json")
            link = os.path.join(root, "link.json")
            os.symlink(target, link)
            with self.assertRaisesRegex(RadarNormalizationError, "符号链接|symlink"):
                load_radar_normalization_artifact(
                    link,
                    target_size=(1, 1, 3),
                    source_pc_range=(0, -2, -1, 10, 2, 1),
                    model_pc_range=(0, -1, -1, 5, 1, 1),
                    doppler_scale_mps=4.0,
                )

    def test_checkpoint_binding_requires_both_spec_and_artifact_hash(self):
        """阶段继承必须同时匹配完整 spec 内容与 artifact 文件身份。"""
        from diffusion_consistency_radar.radar_normalization import (
            RadarNormalizationError,
            assert_same_radar_normalization,
        )

        spec = self._valid_spec()
        assert_same_radar_normalization(
            spec,
            "a" * 64,
            json.loads(json.dumps(spec)),
            "a" * 64,
            context="test chain",
        )

        changed = json.loads(json.dumps(spec))
        changed["frame_count"] += 1
        with self.assertRaisesRegex(RadarNormalizationError, "spec|内容"):
            assert_same_radar_normalization(
                spec,
                "a" * 64,
                changed,
                "a" * 64,
                context="test chain",
            )
        with self.assertRaisesRegex(RadarNormalizationError, "SHA-256|hash"):
            assert_same_radar_normalization(
                spec,
                "a" * 64,
                spec,
                "b" * 64,
                context="test chain",
            )

    def test_condition_noise_preserves_occupancy_and_nonnegative_variance(self):
        """特征噪声不得破坏 occupancy 与 variance 的物理边界。"""
        from diffusion_consistency_radar.cm.augmentation import VoxelAugmentation

        augmentation = VoxelAugmentation(noise_std=1.0)
        voxel = torch.zeros(1, 4, 1, 1, 1)
        voxel[:, 0] = 1.0
        voxel[:, 1] = 2.0
        voxel[:, 2] = 3.0
        voxel[:, 3] = 0.1
        original = voxel.clone()
        with mock.patch(
            "diffusion_consistency_radar.cm.augmentation.torch.randn_like",
            return_value=-torch.ones_like(voxel),
        ):
            augmented = augmentation._add_noise(voxel)

        torch.testing.assert_close(voxel, original)
        torch.testing.assert_close(augmented[:, 0], original[:, 0])
        self.assertGreaterEqual(float(augmented[:, 3].min()), 0.0)

    def test_training_preflight_requires_configured_artifact_and_scale(self):
        """正式训练空配置失败，显式 legacy 只能用于未配置诊断。"""
        from diffusion_consistency_radar.radar_normalization import (
            RadarNormalizationError,
        )
        from diffusion_consistency_radar.scripts.unified_train import (
            resolve_training_radar_normalization,
        )

        geometry = {
            "target_size": (1, 1, 3),
            "source_pc_range": (0, -2, -1, 10, 2, 1),
            "model_pc_range": (0, -1, -1, 5, 1, 1),
        }
        for data_config in ({}, {"radar_normalization_path": "", "doppler_scale_mps": None}):
            with self.subTest(data_config=data_config):
                with self.assertRaisesRegex(
                    RadarNormalizationError,
                    "path|scale|配置|artifact",
                ):
                    resolve_training_radar_normalization(data_config, **geometry)

        spec, digest = resolve_training_radar_normalization(
            {"radar_normalization_path": "", "doppler_scale_mps": None},
            allow_legacy_radar_units=True,
            **geometry,
        )
        self.assertIsNone(spec)
        self.assertEqual(digest, "")

        with self.assertRaisesRegex(RadarNormalizationError, "legacy|同时|互斥"):
            resolve_training_radar_normalization(
                {
                    "radar_normalization_path": "/configured/artifact.json",
                    "doppler_scale_mps": 4.0,
                },
                allow_legacy_radar_units=True,
                **geometry,
            )

    def test_training_preflight_loads_exact_artifact_and_default_yaml_is_formal(self):
        """有效配置返回真实文件 hash；默认 YAML 绑定已验收正式 artifact。"""
        from diffusion_consistency_radar.scripts.unified_train import (
            resolve_training_radar_normalization,
        )

        with tempfile.TemporaryDirectory() as root:
            path = self._write_spec(root, self._valid_spec())
            with open(path, "rb") as handle:
                expected_sha256 = hashlib.sha256(handle.read()).hexdigest()
            loaded, digest = resolve_training_radar_normalization(
                {
                    "radar_normalization_path": path,
                    "doppler_scale_mps": 4.0,
                },
                target_size=(1, 1, 3),
                source_pc_range=(0, -2, -1, 10, 2, 1),
                model_pc_range=(0, -1, -1, 5, 1, 1),
            )

        self.assertEqual(loaded, self._valid_spec())
        self.assertEqual(digest, expected_sha256)
        config_path = os.path.join(
            ROOT,
            "diffusion_consistency_radar",
            "config",
            "default_config.yaml",
        )
        with open(config_path, encoding="utf-8") as handle:
            default_config = yaml.safe_load(handle)
        default_data = default_config["data"]
        self.assertEqual(
            default_data["dataset_dir"],
            "./Data/NTU4DRadLM_Pre_formal_v2_80m_86p8_v1",
        )
        self.assertEqual(default_data["target_size"], [32, 128, 128])
        self.assertEqual(
            default_data["source_pc_range"],
            [0, -20, -6, 80, 20, 10],
        )
        self.assertEqual(
            default_data["model_pc_range"],
            [0, -20, -6, 80, 20, 10],
        )
        self.assertEqual(
            default_data["radar_normalization_path"],
            "./diffusion_consistency_radar/config/"
            "radar_normalization_garden_32x128x128_80m_"
            "train80_purge3s_86p8_v2.json",
        )
        self.assertEqual(default_data["doppler_scale_mps"], 86.8)
        self.assertEqual(default_data["checkpoint_protocol"], "formal_chain_v2")
        self.assertIn("temporal_split_garden_train80_purge3s_v1.json", default_data["temporal_split_artifact"])
        self.assertIn("formal_data_protocol_garden_train80_purge3s_v1.json", default_data["data_protocol_path"])
        self.assertEqual(default_data["scene_names"], ["garden"])
        self.assertTrue(default_data["require_real_ir"])
        self.assertTrue(default_data["require_real_calibration"])
        self.assertTrue(default_data["require_persisted_observed_mask"])
        for stage in ("vae", "ldm", "cd"):
            self.assertEqual(
                default_config[stage]["save_dir"],
                "./Result/train_results/formal_v2_80m_86p8_v1/"
                f"{stage}",
            )


if __name__ == "__main__":
    unittest.main()
