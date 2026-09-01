#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 occupancy threshold sweep、选择规则和 checkpoint 绑定合同。"""

import os
import sys
import tempfile

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion_consistency_radar.occupancy_threshold_artifact import (
    THRESHOLD_SWEEP_PROTOCOL,
    build_threshold_artifact,
    select_threshold_record,
    threshold_sweep_batch_counts,
    threshold_sweep_metrics,
    validate_threshold_artifact,
)


def test_threshold_sweep_excludes_unknown_voxels():
    probability = torch.tensor([[[[[0.9, 0.9, 0.1]]]]])
    target = torch.tensor([[[[[1.0, 0.0, 0.0]]]]])
    observed = torch.tensor([[[[[1, 0, 1]]]]], dtype=torch.bool)
    counts = threshold_sweep_batch_counts(
        probability,
        target,
        observed,
        (0.2, 0.8),
    )
    records = threshold_sweep_metrics(counts, (0.2, 0.8))
    assert records[0]["tp"] == 1
    assert records[0]["fp"] == 0
    assert records[1]["iou"] == 1.0


def test_threshold_selector_uses_iou_recall_then_lower_threshold():
    records = [
        {"threshold": 0.2, "iou": 0.6, "recall": 0.7},
        {"threshold": 0.4, "iou": 0.7, "recall": 0.6},
        {"threshold": 0.6, "iou": 0.7, "recall": 0.8},
    ]
    assert select_threshold_record(records)["threshold"] == 0.6
    records[1]["recall"] = 0.8
    assert select_threshold_record(records)["threshold"] == 0.4


def test_artifact_binds_checkpoint_hash_stage_and_weight_source():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "cd_best.pt")
        torch.save({"weight": torch.ones(1)}, checkpoint_path)
        metrics = [
            {
                "threshold": 0.2,
                "tp": 6,
                "fp": 2,
                "fn": 4,
                "iou": 0.5,
                "precision": 0.75,
                "recall": 0.6,
                "f1": 2 / 3,
            },
            {
                "threshold": 0.5,
                "tp": 5,
                "fp": 0,
                "fn": 5,
                "iou": 0.5,
                "precision": 1.0,
                "recall": 0.5,
                "f1": 2 / 3,
            },
        ]
        checkpoint = {
            "stage": "cd",
            "checkpoint_protocol": "formal_chain_v2",
            "deployment_weight_source": "ema_model_state_dict",
            "data_protocol": {"protocol": "formal_data_v2"},
            "occupancy_threshold_validation": {
                "protocol": THRESHOLD_SWEEP_PROTOCOL,
                "split": "temporal_block_validation_suffix",
                "observation_domain": "persisted_observed_mask_v1",
                "deployment_weight_source": "ema_model_state_dict",
                "candidate_thresholds": [0.2, 0.5],
                "metrics_by_threshold": metrics,
            },
        }
        artifact = build_threshold_artifact(
            checkpoint,
            checkpoint_path=checkpoint_path,
        )
        assert artifact["selected_threshold"] == 0.2
        validate_threshold_artifact(
            artifact,
            checkpoint_path=checkpoint_path,
            expected_stage="cd",
            expected_weight_source="ema_model_state_dict",
        )
        torch.save({"weight": torch.zeros(1)}, checkpoint_path)
        try:
            validate_threshold_artifact(
                artifact,
                checkpoint_path=checkpoint_path,
                expected_stage="cd",
                expected_weight_source="ema_model_state_dict",
            )
        except ValueError as exc:
            assert "SHA-256" in str(exc)
        else:
            raise AssertionError("threshold artifact 必须拒绝 checkpoint 内容漂移")


if __name__ == "__main__":
    test_threshold_sweep_excludes_unknown_voxels()
    test_threshold_selector_uses_iou_recall_then_lower_threshold()
    test_artifact_binds_checkpoint_hash_stage_and_weight_source()
    print("test_occupancy_threshold_artifact passed")
