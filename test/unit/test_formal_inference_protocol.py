# -*- coding: utf-8 -*-
"""验证正式部署生成与已保存预测离线评价的边界协议。"""

import csv
import hashlib
import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class FormalInferenceProtocolTest(unittest.TestCase):
    def _read_project_file(self, relative_path):
        with open(os.path.join(ROOT, relative_path), "r", encoding="utf-8") as handle:
            return handle.read()

    def _file_hash(self, path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            digest.update(handle.read())
        return digest.hexdigest()

    def _write_fixture(self, root):
        pred_dir = os.path.join(root, "pred")
        radar_dir = os.path.join(root, "radar")
        target_dir = os.path.join(root, "target")
        lidar_dir = os.path.join(root, "lidar")
        os.makedirs(pred_dir)
        os.makedirs(radar_dir)
        os.makedirs(target_dir)
        os.makedirs(lidar_dir)

        for index in range(2):
            frame_id = f"{index:06d}"
            pred = np.zeros((4, 2, 2, 2), dtype=np.float32)
            pred[0, index, index, index] = 0.9
            pred[1, index, index, index] = 0.5
            np.save(os.path.join(pred_dir, f"{frame_id}_voxel.npy"), pred)
            np.save(
                os.path.join(pred_dir, f"{frame_id}_uncertainty.npy"),
                np.full((2, 2, 2), 0.1 + 0.1 * index, dtype=np.float32),
            )

            radar = np.zeros((2, 2, 2, 4), dtype=np.float32)
            target = np.zeros((2, 2, 2, 4), dtype=np.float32)
            radar[index, index, index, 0] = 1.0
            target[index, index, index, 0] = 1.0
            target[index, index, index, 3] = 1.0
            np.save(os.path.join(radar_dir, f"{frame_id}.npy"), radar)
            np.save(os.path.join(target_dir, f"{frame_id}.npy"), target)

        np.save(
            os.path.join(lidar_dir, "lidar_000.npy"),
            np.array([[1.5, 1.5, 1.5, 1.0]], dtype=np.float32),
        )
        np.save(
            os.path.join(lidar_dir, "lidar_001.npy"),
            np.array([[0.5, 0.5, 0.5, 1.0]], dtype=np.float32),
        )
        lidar_index_file = os.path.join(root, "lidar_index_sequence.txt")
        with open(lidar_index_file, "w", encoding="utf-8") as handle:
            handle.write("1\n0\n")

        metadata = {
            "stage": "deployment_generation",
            "target_size": [2, 2, 2],
            "source_pc_range": [0, 0, 0, 2, 2, 2],
            "model_pc_range": [0, 0, 0, 2, 2, 2],
            "voxel_size": [1, 1, 1],
            "occ_threshold": 0.5,
            "model_type": "ldm",
            "steps": 40,
            "sampler": "heun",
            "model_is_multimodal": True,
            "require_real_ir": True,
            "frame_count": 2,
        }
        metadata_path = os.path.join(pred_dir, "inference_run.json")
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle)
        return {
            "pred": pred_dir,
            "radar": radar_dir,
            "target": target_dir,
            "lidar": lidar_dir,
            "lidar_index": lidar_index_file,
            "metadata": metadata_path,
        }

    def _evaluate(self, paths, output_dir, **overrides):
        from diffusion_consistency_radar.scripts.evaluate_saved_predictions import (
            evaluate_saved_predictions,
        )

        arguments = {
            "pred_voxel_dir": paths["pred"],
            "radar_voxel_dir": paths["radar"],
            "target_voxel_dir": paths["target"],
            "output_dir": output_dir,
            "raw_livox_dir": paths["lidar"],
            "lidar_index_file": paths["lidar_index"],
        }
        arguments.update(overrides)
        return evaluate_saved_predictions(**arguments)

    def test_saved_prediction_evaluator_pairs_frames_and_preserves_predictions(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self._write_fixture(root)
            pred_paths = [
                os.path.join(paths["pred"], f"{index:06d}_voxel.npy")
                for index in range(2)
            ]
            hashes_before = [self._file_hash(path) for path in pred_paths]
            output_dir = os.path.join(root, "evaluation")

            summary = self._evaluate(paths, output_dir)

            hashes_after = [self._file_hash(path) for path in pred_paths]
            self.assertEqual(hashes_before, hashes_after)
            self.assertEqual(summary["stage"], "offline_evaluation")
            self.assertTrue(summary["prediction_unchanged"])
            self.assertEqual(summary["frame_count"], 2)
            self.assertEqual(
                summary["model_pc_range"],
                [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            )
            self.assertTrue(
                os.path.isfile(os.path.join(output_dir, "evaluation_frames.csv"))
            )
            self.assertTrue(
                os.path.isfile(os.path.join(output_dir, "evaluation_summary.json"))
            )
            with open(
                os.path.join(output_dir, "evaluation_frames.csv"),
                "r",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["lidar_file"], "lidar_001.npy")
            self.assertEqual(rows[1]["lidar_file"], "lidar_000.npy")
            self.assertEqual(float(rows[0]["pred_target_chamfer"]), 0.0)

    def test_evaluator_rejects_nonempty_output_before_writing(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self._write_fixture(root)
            output_dir = os.path.join(root, "evaluation")
            os.makedirs(output_dir)
            sentinel = os.path.join(output_dir, "keep.txt")
            with open(sentinel, "w", encoding="utf-8") as handle:
                handle.write("keep")

            with self.assertRaisesRegex(ValueError, "非空|non-empty"):
                self._evaluate(paths, output_dir)

            self.assertTrue(os.path.isfile(sentinel))
            self.assertEqual(os.listdir(output_dir), ["keep.txt"])

    def test_evaluator_rejects_missing_metadata_before_writing(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self._write_fixture(root)
            os.remove(paths["metadata"])
            output_dir = os.path.join(root, "evaluation")

            with self.assertRaisesRegex(ValueError, "inference_run.json|metadata"):
                self._evaluate(paths, output_dir)

            self.assertFalse(os.path.exists(output_dir))

    def test_evaluator_rejects_incomplete_metadata_threshold(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self._write_fixture(root)
            with open(paths["metadata"], "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            del metadata["occ_threshold"]
            with open(paths["metadata"], "w", encoding="utf-8") as handle:
                json.dump(metadata, handle)
            output_dir = os.path.join(root, "evaluation")

            with self.assertRaisesRegex(ValueError, "occ_threshold"):
                self._evaluate(paths, output_dir)

            self.assertFalse(os.path.exists(output_dir))

    def test_voxel_points_use_recorded_voxel_size(self):
        from diffusion_consistency_radar.scripts import evaluate_saved_predictions

        voxel = np.zeros((1, 2, 2, 2), dtype=np.float32)
        voxel[0, 0, 0, 0] = 1.0
        points = evaluate_saved_predictions._voxel_czxy_to_points(
            voxel,
            pc_range=(0, 0, 0, 4, 4, 4),
            threshold=0.5,
            voxel_size=(2, 2, 2),
        )

        np.testing.assert_allclose(points, np.array([[1.0, 1.0, 1.0]]))

    def test_evaluator_rejects_frame_mismatch_and_unknown_prediction_file(self):
        for mutation in ("missing_target", "unknown_prediction"):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as root:
                    paths = self._write_fixture(root)
                    if mutation == "missing_target":
                        os.remove(os.path.join(paths["target"], "000001.npy"))
                    else:
                        with open(
                            os.path.join(paths["pred"], "notes.txt"),
                            "w",
                            encoding="utf-8",
                        ) as handle:
                            handle.write("unexpected")
                    output_dir = os.path.join(root, "evaluation")

                    with self.assertRaisesRegex(
                        ValueError,
                        "frame|帧|unknown|未知",
                    ):
                        self._evaluate(paths, output_dir)

                    self.assertFalse(os.path.exists(output_dir))

    def test_evaluator_rejects_invalid_prediction_before_writing(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self._write_fixture(root)
            invalid = np.zeros((4, 2, 2, 2), dtype=np.float32)
            invalid[0, 0, 0, 0] = np.nan
            np.save(os.path.join(paths["pred"], "000001_voxel.npy"), invalid)
            output_dir = os.path.join(root, "evaluation")

            with self.assertRaisesRegex(ValueError, "非有限|finite"):
                self._evaluate(paths, output_dir)

            self.assertFalse(os.path.exists(output_dir))

    def test_evaluator_requires_paired_raw_lidar_arguments_and_valid_index(self):
        with tempfile.TemporaryDirectory() as root:
            paths = self._write_fixture(root)
            output_dir = os.path.join(root, "evaluation_missing_pair")
            with self.assertRaisesRegex(ValueError, "同时|together"):
                self._evaluate(paths, output_dir, lidar_index_file="")
            self.assertFalse(os.path.exists(output_dir))

            with open(paths["lidar_index"], "w", encoding="utf-8") as handle:
                handle.write("99\n0\n")
            output_dir = os.path.join(root, "evaluation_bad_index")
            with self.assertRaisesRegex(ValueError, "越界|bounds"):
                self._evaluate(paths, output_dir)
            self.assertFalse(os.path.exists(output_dir))

    def test_formal_generation_launchers_are_deployment_only(self):
        """正式生成脚本只消费 Radar+IR，不得接收离线真值参数。"""
        launchers = (
            "diffusion_consistency_radar/launch/inference_ldm.sh",
            "diffusion_consistency_radar/launch/inference_cd.sh",
            "diffusion_consistency_radar/launch/inference_uniified.sh",
        )
        forbidden = (
            "--target_voxel_dir",
            "--compare_with_target",
            "--report_task_metrics",
            "--compare_with_lidar",
            "--raw_livox_dir",
            "--lidar_index_file",
        )
        for launcher in launchers:
            with self.subTest(launcher=launcher):
                text = self._read_project_file(launcher)
                self.assertIn(
                    "NTU4DRadLM_Pre_sensor_aware_p1_04_candidate",
                    text,
                )
                self.assertIn("formal_p1_04_full120_86p8_v1", text)
                self.assertIn(
                    'RESULTS_DIR="${ROOT_DIR}/Result/train_results/${PROTOCOL_TAG}"',
                    text,
                )
                self.assertIn("--require_real_ir", text)
                self.assertIn("--save_voxel", text)
                self.assertIn("--save_pointcloud", text)
                self.assertIn("--save_uncertainty", text)
                self.assertIn("_deploy", text)
                self.assertIn("diagnose_checkpoint_chain.py", text)
                self.assertIn("--vae_ckpt", text)
                self.assertIn("--ldm_ckpt", text)
                self.assertIn("--cd_ckpt", text)
                for token in forbidden:
                    self.assertNotIn(token, text)
                self.assertLess(
                    text.index('"${MANIFEST_SCRIPT}" validate'),
                    text.index('"${INFER_SCRIPT}"'),
                )
                self.assertLess(
                    text.index("diagnose_checkpoint_chain.py"),
                    text.index('"${INFER_SCRIPT}"'),
                )

    def test_unified_launcher_does_not_silently_skip_missing_formal_stages(self):
        text = self._read_project_file(
            "diffusion_consistency_radar/launch/inference_uniified.sh"
        )
        self.assertNotIn('RUN_LDM=false', text)
        self.assertNotIn('RUN_CD=false', text)
        self.assertIn('diagnose_checkpoint_chain.py', text)
        self.assertIn('--cd_ckpt', text)

    def test_formal_evaluation_launcher_never_runs_generation_model(self):
        """独立评价入口只能读取已保存预测，不能引用 checkpoint 或生成脚本。"""
        text = self._read_project_file(
            "diffusion_consistency_radar/launch/evaluate_inference.sh"
        )

        self.assertIn("evaluate_saved_predictions.py", text)
        self.assertIn(
            "NTU4DRadLM_Pre_sensor_aware_p1_04_candidate",
            text,
        )
        self.assertIn("NTU4DRadLM_Raw_p1_01_candidate", text)
        self.assertIn("formal_p1_04_full120_86p8_v1", text)
        self.assertIn('"${MANIFEST_SCRIPT}" validate', text)
        self.assertIn("--target_voxel_dir", text)
        self.assertIn("--raw_livox_dir", text)
        self.assertIn("--lidar_index_file", text)
        self.assertIn("_evaluation", text)
        self.assertIn("ldm|cd|cd4", text)
        self.assertLess(
            text.index('"${MANIFEST_SCRIPT}" validate'),
            text.index('conda run -n Radar-Diffusion python "${EVALUATE_SCRIPT}"'),
        )
        lowered = text.lower()
        for token in ("vae_ckpt", "model_ckpt", ".pt", "scripts/inference.py"):
            self.assertNotIn(token, lowered)


if __name__ == "__main__":
    unittest.main()
