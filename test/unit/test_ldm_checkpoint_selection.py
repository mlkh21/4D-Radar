#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试基于验证集任务/结构指标的 LDM checkpoint 选择。"""

import csv
import os
import sys
import json
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_DIR = os.path.join(PROJECT_ROOT, "test", "evaluation", "ldm")
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

import select_ldm_checkpoint as selector


def candidate(name, *, bev_iou, bev_recall, top, trunk, ratio):
    return {
        "name": name,
        "checkpoint": f"/{name}.pt",
        "bev_iou": bev_iou,
        "bev_recall": bev_recall,
        "top_height_recall": top,
        "trunk_region_recall": trunk,
        "pred_to_target_ratio": ratio,
    }


class LDMCheckpointSelectionTest(unittest.TestCase):
    def test_feasible_candidate_is_preferred_over_better_partial_candidate(self):
        candidates = [
            candidate("partial", bev_iou=0.40, bev_recall=0.95, top=0.20, trunk=0.60, ratio=2.0),
            candidate("feasible", bev_iou=0.30, bev_recall=0.85, top=0.12, trunk=0.70, ratio=3.0),
        ]

        result = selector.select_checkpoint(candidates)

        self.assertEqual(result["selected"]["name"], "feasible")
        self.assertTrue(result["gate_satisfied"])

    def test_no_feasible_candidate_maximizes_worst_normalized_gate(self):
        candidates = [
            candidate("imbalanced", bev_iou=0.40, bev_recall=0.90, top=0.02, trunk=0.80, ratio=2.0),
            candidate("balanced", bev_iou=0.24, bev_recall=0.78, top=0.09, trunk=0.60, ratio=5.0),
        ]

        result = selector.select_checkpoint(candidates)

        self.assertEqual(result["selected"]["name"], "balanced")
        self.assertFalse(result["gate_satisfied"])
        self.assertIn("no candidate", result["reason"])

    def test_summary_loader_requires_exactly_one_real_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "summary.csv")
            checkpoint = os.path.join(tmp, "epoch1.pt")
            with open(checkpoint, "wb") as handle:
                handle.write(b"checkpoint")
            vae_path = os.path.join(tmp, "vae.pt")
            with open(vae_path, "wb") as handle:
                handle.write(b"vae")
            dataset_path = os.path.join(tmp, "dataset")
            os.mkdir(dataset_path)
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "variant", "bev_iou", "bev_recall", "top_height_recall",
                    "trunk_region_recall", "vertical_connectivity_recall",
                    "voxel_iou", "pred_to_target_ratio",
                ])
                writer.writeheader()
                writer.writerow({
                    "variant": "real", "bev_iou": "0.25", "bev_recall": "0.8",
                    "top_height_recall": "0.1", "trunk_region_recall": "0.65",
                    "vertical_connectivity_recall": "0.7", "voxel_iou": "0.12",
                    "pred_to_target_ratio": "4.0",
                })
            protocol = {
                "split": "validation", "max_samples": 32, "required_sample_count": 32,
                "steps": 20, "sampler": "euler", "seed": 42, "occ_threshold": 0.99,
                "target_threshold": 0.5, "variants": ["real"],
                "target_size": [64, 128, 128],
                "source_pc_range": [0, -20, -6, 120, 20, 10],
                "model_pc_range": [0, -20, -6, 40, 20, 10],
                "sample_indices": list(range(32)),
                "dataset_root": dataset_path, "vae_ckpt": vae_path,
                "checkpoint_sha256": selector.sha256_file(checkpoint),
                "vae_sha256": selector.sha256_file(vae_path),
                "dataset_manifest_sha256": selector.dataset_manifest_sha256(dataset_path),
            }
            with open(os.path.join(tmp, "ir_condition_ablation.json"), "w", encoding="utf-8") as handle:
                json.dump({
                    "model_ckpt": checkpoint,
                    "protocol": protocol,
                }, handle)

            loaded = selector.load_candidate("epoch1", checkpoint, path)

        self.assertEqual(loaded["name"], "epoch1")
        self.assertAlmostEqual(loaded["bev_iou"], 0.25)

    def test_runner_uses_fixed_validation_protocol_and_does_not_copy_best_alias(self):
        path = os.path.join(PROJECT_ROOT, "test", "mini-test", "run_ldm_z64_checkpoint_selection.sh")
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding="utf-8") as handle:
            text = handle.read()

        self.assertIn("--split validation", text)
        self.assertIn('--variants real', text)
        self.assertIn('--max_samples 32', text)
        self.assertIn('--require_sample_count 32', text)
        self.assertIn('--steps 20', text)
        self.assertIn('--sampler euler', text)
        self.assertIn('--seed 42', text)
        self.assertIn('--occ_threshold 0.99', text)
        self.assertIn('mkdir -- "${LOCK_PATH}"', text)
        self.assertIn('output already exists and is not empty', text)
        self.assertNotIn('if [[ ! -f "${summary_csv}" ]]', text)
        for variable in ("MAX_SAMPLES", "OCC_THRESHOLD", "STEPS", "SAMPLER", "SEED"):
            self.assertNotIn(f'{variable}="${{{variable}:-', text)
        self.assertNotIn("cp ", text)

    def test_nonfinite_gate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            selector.validate_gates(dict(selector.DEFAULT_GATES, bev_iou=float("nan")))

    def test_changed_checkpoint_content_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = os.path.join(tmp, "model.pt")
            with open(checkpoint, "wb") as handle:
                handle.write(b"before")
            protocol = dict(selector.EXPECTED_FIXED_PROTOCOL)
            protocol.update({
                "dataset_root": tmp, "vae_ckpt": checkpoint,
                "sample_indices": list(range(32)),
                "checkpoint_sha256": selector.sha256_file(checkpoint),
                "vae_sha256": selector.sha256_file(checkpoint),
                "dataset_manifest_sha256": selector.dataset_manifest_sha256(tmp),
            })
            with open(checkpoint, "wb") as handle:
                handle.write(b"after")

            with self.assertRaisesRegex(ValueError, "hash"):
                selector.validate_source_fingerprints("candidate", checkpoint, protocol)

    def test_selection_rejects_candidates_with_different_protocols(self):
        first = candidate("first", bev_iou=0.3, bev_recall=0.8, top=0.1, trunk=0.65, ratio=3.0)
        second = candidate("second", bev_iou=0.31, bev_recall=0.8, top=0.1, trunk=0.65, ratio=3.0)
        first["protocol"] = {"split": "validation", "seed": 42}
        second["protocol"] = {"split": "validation", "seed": 7}

        with self.assertRaisesRegex(ValueError, "protocol"):
            selector.validate_common_protocol([first, second])

    def test_selection_rejects_common_but_nonfixed_protocol(self):
        protocol = dict(selector.EXPECTED_FIXED_PROTOCOL)
        protocol.update({
            "dataset_root": "/dataset", "vae_ckpt": "/vae.pt",
            "sample_indices": list(range(32)),
        })
        protocol["seed"] = 7
        first = candidate("first", bev_iou=0.3, bev_recall=0.8, top=0.1, trunk=0.65, ratio=3.0)
        second = candidate("second", bev_iou=0.31, bev_recall=0.8, top=0.1, trunk=0.65, ratio=3.0)
        first["protocol"] = dict(protocol)
        second["protocol"] = dict(protocol)

        with self.assertRaisesRegex(ValueError, "fixed protocol"):
            selector.validate_common_protocol([first, second])

    def test_selection_rejects_different_dataset_or_vae_sources(self):
        base = dict(selector.EXPECTED_FIXED_PROTOCOL)
        base.update({
            "dataset_root": "/dataset-a", "vae_ckpt": "/vae.pt",
            "sample_indices": list(range(32)),
        })
        first = candidate("first", bev_iou=0.3, bev_recall=0.8, top=0.1, trunk=0.65, ratio=3.0)
        second = candidate("second", bev_iou=0.31, bev_recall=0.8, top=0.1, trunk=0.65, ratio=3.0)
        first["protocol"] = dict(base)
        second["protocol"] = dict(base, dataset_root="/dataset-b")

        with self.assertRaisesRegex(ValueError, "protocol"):
            selector.validate_common_protocol([first, second])

    def test_common_protocol_allows_different_checkpoint_hashes(self):
        base = dict(selector.EXPECTED_FIXED_PROTOCOL)
        base.update({
            "dataset_root": "/dataset", "vae_ckpt": "/vae.pt",
            "sample_indices": list(range(32)), "vae_sha256": "same-vae",
            "dataset_manifest_sha256": "same-data",
        })
        first = candidate("first", bev_iou=0.3, bev_recall=0.8, top=0.1, trunk=0.65, ratio=3.0)
        second = candidate("second", bev_iou=0.31, bev_recall=0.8, top=0.1, trunk=0.65, ratio=3.0)
        first["protocol"] = dict(base, checkpoint_sha256="epoch-one")
        second["protocol"] = dict(base, checkpoint_sha256="epoch-two")

        common = selector.validate_common_protocol([first, second])

        self.assertNotIn("checkpoint_sha256", common)
        self.assertEqual(common["vae_sha256"], "same-vae")


if __name__ == "__main__":
    unittest.main()
