#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证正式训练单机 1--4 GPU DDP 的批量、采样和启动协议。"""

import os
import sys
import unittest

import torch.nn as nn
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffusion_consistency_radar.distributed_training import (
    DistributedContext,
    DistributedEvalSampler,
    assert_distributed_config_compatible,
    assert_resume_distributed_compatible,
    deterministic_noise_from_sample_ids,
    distributed_checkpoint_metadata,
    reduce_named_sums,
    resolve_world_batch_plan,
    unwrap_model,
)


class DistributedTrainingHelperTest(unittest.TestCase):
    """共享 helper 必须显式表达多卡语义，不能依赖隐式全局状态。"""

    def test_world_batch_plan_preserves_declared_effective_batch(self):
        expected = {
            1: (2, 8, 16),
            2: (1, 8, 16),
            3: (1, 6, 18),
            4: (1, 4, 16),
        }
        for world_size, values in expected.items():
            with self.subTest(world_size=world_size):
                plan = resolve_world_batch_plan(world_size)
                self.assertEqual(
                    (
                        plan.per_rank_batch_size,
                        plan.gradient_accumulation_steps,
                        plan.effective_global_batch_size,
                    ),
                    values,
                )
                self.assertEqual(
                    plan.effective_global_batch_size,
                    world_size
                    * plan.per_rank_batch_size
                    * plan.gradient_accumulation_steps,
                )

    def test_world_batch_plan_rejects_unsupported_process_counts(self):
        for world_size in (0, 5, -1, True):
            with self.subTest(world_size=world_size):
                with self.assertRaises((TypeError, ValueError)):
                    resolve_world_batch_plan(world_size)

    def test_eval_sampler_has_exact_global_coverage_without_padding(self):
        dataset = list(range(11))
        for world_size in (2, 3, 4):
            partitions = [
                list(
                    DistributedEvalSampler(
                        dataset,
                        num_replicas=world_size,
                        rank=rank,
                    )
                )
                for rank in range(world_size)
            ]
            flattened = [index for part in partitions for index in part]
            self.assertEqual(sorted(flattened), list(range(len(dataset))))
            self.assertEqual(len(flattened), len(set(flattened)))

    def test_single_process_reduction_and_unwrap_are_identity(self):
        context = DistributedContext.single_process(device="cpu")
        self.assertTrue(context.is_main_process)
        self.assertEqual(
            reduce_named_sums({"loss": 2.5, "count": 2}, context),
            {"loss": 2.5, "count": 2.0},
        )
        model = nn.Linear(2, 1)
        wrapper = nn.DataParallel(model)
        self.assertIs(unwrap_model(wrapper), model)

    def test_checkpoint_metadata_records_padding_and_global_batch(self):
        context = DistributedContext(
            rank=1,
            local_rank=1,
            world_size=3,
            device="cpu",
            initialized=False,
        )
        plan = resolve_world_batch_plan(3)
        metadata = distributed_checkpoint_metadata(
            context,
            plan,
            train_dataset_size=10,
        )
        self.assertEqual(metadata["protocol"], "single_node_ddp_v1")
        self.assertEqual(metadata["world_size"], 3)
        self.assertEqual(metadata["effective_global_batch_size"], 18)
        self.assertEqual(metadata["train_sampler_padding"], 2)

    def test_sample_identity_noise_is_partition_invariant(self):
        sample_ids = [f"frame-{index}" for index in range(7)]
        reference = torch.zeros(7, 1, 2, 2)
        full = deterministic_noise_from_sample_ids(
            reference,
            sample_ids,
            seed=42,
        )
        reconstructed = torch.empty_like(full)
        for rank in range(3):
            indices = list(range(rank, len(sample_ids), 3))
            reconstructed[indices] = deterministic_noise_from_sample_ids(
                reference[indices],
                [sample_ids[index] for index in indices],
                seed=42,
            )
        self.assertTrue(torch.equal(full, reconstructed))

    def test_dataset_sample_identity_does_not_bind_absolute_root(self):
        dataset_path = os.path.join(
            ROOT, "diffusion_consistency_radar/cm/dataset_loader.py"
        )
        with open(dataset_path, "r", encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn(
            'f"{scene}/{os.path.splitext(os.path.basename(target_path))[0]}"',
            source,
        )
        self.assertNotIn('sample_id = f"{target_path}"', source)

    def test_resume_allows_world_size_change_but_rejects_effective_batch_drift(self):
        checkpoint = {
            "distributed_training": {
                "protocol": "single_node_ddp_v1",
                "world_size": 2,
                "effective_global_batch_size": 16,
            }
        }
        assert_resume_distributed_compatible(
            checkpoint,
            expected_effective_global_batch_size=16,
        )
        with self.assertRaisesRegex(ValueError, "有效全局 batch"):
            assert_resume_distributed_compatible(
                checkpoint,
                expected_effective_global_batch_size=18,
            )

    def test_runtime_rejects_distributed_config_identity_drift(self):
        context = DistributedContext(
            rank=0,
            local_rank=0,
            world_size=2,
            device="cpu",
            initialized=False,
        )
        plan = assert_distributed_config_compatible(
            context,
            per_rank_batch_size=1,
            gradient_accumulation_steps=8,
            configured_protocol="single_node_ddp_v1",
            configured_world_size=2,
            configured_effective_global_batch_size=16,
        )
        self.assertEqual(plan.effective_global_batch_size, 16)
        with self.assertRaisesRegex(ValueError, "有效全局 batch"):
            assert_distributed_config_compatible(
                context,
                per_rank_batch_size=1,
                gradient_accumulation_steps=8,
                configured_protocol="single_node_ddp_v1",
                configured_world_size=2,
                configured_effective_global_batch_size=18,
            )
        with self.assertRaisesRegex(ValueError, "distributed_protocol"):
            assert_distributed_config_compatible(
                context,
                per_rank_batch_size=1,
                gradient_accumulation_steps=8,
                configured_protocol="legacy_ddp",
            )
        with self.assertRaisesRegex(ValueError, "必须显式配置"):
            assert_distributed_config_compatible(
                context,
                per_rank_batch_size=1,
                gradient_accumulation_steps=8,
            )
        with self.assertRaisesRegex(ValueError, "batch/梯度累积"):
            assert_distributed_config_compatible(
                context,
                per_rank_batch_size=2,
                gradient_accumulation_steps=4,
                configured_protocol="single_node_ddp_v1",
                configured_world_size=2,
                configured_effective_global_batch_size=16,
            )

    def test_multimodal_forward_keeps_optional_uncertainty_head_in_ddp_graph(self):
        from diffusion_consistency_radar.cm.multimodal_fusion import (
            CompleteDualModalityPerceptionNet,
        )

        class TinyBackbone(nn.Module):
            def forward(self, fused, _timesteps):
                return fused

        class TinyIRExtractor(nn.Module):
            def forward(self, image):
                return image.new_zeros(image.shape[0], 32, 1, 1)

        class TinyProjection(nn.Module):
            def forward(self, features, *_args, return_mask=False):
                projected = features.new_zeros(features.shape[0], 32, 1, 1, 1)
                mask = torch.ones(
                    features.shape[0], 1, 1, 1, 1, dtype=torch.bool
                )
                return (projected, mask) if return_mask else projected

        model = CompleteDualModalityPerceptionNet(
            TinyBackbone(),
            voxel_shape=(1, 1, 1),
            pc_range=(0, 0, 0, 1, 1, 1),
        )
        model.ir_extractor = TinyIRExtractor()
        model.projection_layer = TinyProjection()
        output = model(
            torch.ones(1, 4, 1, 1, 1),
            torch.ones(1, 3, 2, 2),
            torch.eye(3).unsqueeze(0),
            torch.zeros(1, 3),
            torch.eye(3).unsqueeze(0),
            torch.ones(1),
            return_uncertainty=False,
        )
        output.mean().backward()
        gradients = [
            parameter.grad
            for parameter in model.model_uncertainty_head.parameters()
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(torch.count_nonzero(gradient).item() == 0 for gradient in gradients)
        )


class FormalDistributedLauncherProtocolTest(unittest.TestCase):
    """正式 launcher 应按 stage 创建独立进程组并保持 all 顺序编排。"""

    def test_launcher_validates_one_to_four_unique_devices(self):
        path = os.path.join(
            ROOT, "diffusion_consistency_radar/launch/train_unified.sh"
        )
        with open(path, "r", encoding="utf-8") as handle:
            script = handle.read()
        self.assertIn('GPU_COUNT="${#GPU_IDS[@]}"', script)
        self.assertIn("CUDA_DEVICES 包含重复 GPU 编号", script)
        self.assertIn("正式训练仅支持单机 1--4 个 GPU", script)

    def test_launcher_uses_one_torchrun_job_per_stage(self):
        path = os.path.join(
            ROOT, "diffusion_consistency_radar/launch/train_unified.sh"
        )
        with open(path, "r", encoding="utf-8") as handle:
            script = handle.read()
        self.assertIn("launch_training_stage()", script)
        self.assertIn("python -m torch.distributed.run", script)
        self.assertIn('--nproc_per_node="${GPU_COUNT}"', script)
        self.assertIn('launch_training_stage vae', script)
        self.assertIn('launch_training_stage ldm', script)
        self.assertIn('launch_training_stage cd', script)
        self.assertIn('bash "$0" vae', script)
        self.assertIn('bash "$0" ldm', script)
        self.assertIn('bash "$0" cd', script)

    def test_launcher_writes_resolved_batch_contract_to_config(self):
        path = os.path.join(
            ROOT, "diffusion_consistency_radar/launch/train_unified.sh"
        )
        with open(path, "r", encoding="utf-8") as handle:
            script = handle.read()
        self.assertIn("resolve_world_batch_plan", script)
        self.assertIn("cfg['data']['batch_size'] = batch_plan.per_rank_batch_size", script)
        self.assertIn(
            "cfg['optimization']['gradient_accumulation_steps'] = "
            "batch_plan.gradient_accumulation_steps",
            script,
        )
        self.assertIn(
            "cfg['hardware']['effective_global_batch_size'] = "
            "batch_plan.effective_global_batch_size",
            script,
        )
        self.assertIn("cfg['hardware']['cuda_devices'] = cuda_devices", script)
        self.assertIn("cfg['hardware']['num_gpus'] = int(gpu_count)", script)

    def test_ldm_ddp_rejects_legacy_forward_bypass(self):
        path = os.path.join(
            ROOT, "diffusion_consistency_radar/scripts/unified_train.py"
        )
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn(
            "DDP LDM 不支持缺少 IR/标定的 legacy 旁路 batch",
            source,
        )


if __name__ == "__main__":
    unittest.main()
