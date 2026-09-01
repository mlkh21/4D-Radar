#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证正式推理固定随机种子和 CUDA 同步计时合同。"""

import os
import sys
import unittest
from unittest import mock
from types import SimpleNamespace

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffusion_consistency_radar.scripts import inference


class InferenceRuntimeProtocolTest(unittest.TestCase):
    def test_cuda_timing_synchronizes_selected_device(self):
        with mock.patch.object(torch.cuda, "synchronize") as synchronize:
            inference.synchronize_inference_device(torch.device("cuda", 2))
            synchronize.assert_called_once_with(torch.device("cuda", 2))

            synchronize.reset_mock()
            inference.synchronize_inference_device(torch.device("cpu"))
            synchronize.assert_not_called()

    def test_formal_inference_requires_nonnegative_fixed_seed(self):
        self.assertEqual(
            inference.validate_inference_seed(42, require_formal=True),
            42,
        )
        with self.assertRaisesRegex(ValueError, "seed"):
            inference.validate_inference_seed(-1, require_formal=True)
        self.assertEqual(
            inference.validate_inference_seed(-1, require_formal=False),
            -1,
        )

    def test_formal_launchers_pass_explicit_seed(self):
        for name in (
            "inference_ldm.sh",
            "inference_cd.sh",
            "inference_uniified.sh",
        ):
            with self.subTest(name=name):
                path = os.path.join(
                    ROOT,
                    "diffusion_consistency_radar",
                    "launch",
                    name,
                )
                with open(path, "r", encoding="utf-8") as handle:
                    source = handle.read()
                self.assertIn('INFERENCE_SEED="${INFERENCE_SEED:-42}"', source)
                self.assertIn('--seed "${INFERENCE_SEED}"', source)
                self.assertIn("--threshold_artifact", source)
                self.assertNotIn("--occ_threshold", source)

    def test_formal_threshold_requires_artifact_and_rejects_cli_override(self):
        generator = SimpleNamespace(
            deployment_weight_source="model_state_dict",
            checkpoint_protocol="formal_chain_v2",
        )
        base = {
            "threshold_artifact": "",
            "occ_threshold": None,
            "require_real_ir": True,
            "allow_formal_mini_checkpoint": False,
            "model_ckpt": "model.pt",
            "model_type": "ldm",
        }
        with self.assertRaisesRegex(ValueError, "threshold_artifact"):
            inference.resolve_inference_occupancy_threshold(
                SimpleNamespace(**base), generator
            )

        base["occ_threshold"] = 0.3
        with self.assertRaisesRegex(ValueError, "自由"):
            inference.resolve_inference_occupancy_threshold(
                SimpleNamespace(**base), generator
            )

        legacy = SimpleNamespace(
            threshold_artifact="",
            occ_threshold=None,
            require_real_ir=False,
            model_ckpt="legacy.pt",
            model_type="ldm",
        )
        self.assertEqual(
            inference.resolve_inference_occupancy_threshold(legacy, generator),
            0.1,
        )


if __name__ == "__main__":
    unittest.main()
