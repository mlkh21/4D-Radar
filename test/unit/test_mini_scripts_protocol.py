#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 mini 训练/推理脚本的正式协议与硬件保护门禁。"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class MiniScriptsProtocolTest(unittest.TestCase):
    def _read(self, relative_path):
        with open(os.path.join(ROOT, relative_path), "r", encoding="utf-8") as f:
            return f.read()

    def _write_executable(self, path, content):
        Path(path).write_text(content, encoding="utf-8")
        os.chmod(path, 0o755)

    def _run_guarded_runner_with_fake_gpu(
        self,
        gpu_state,
        extra_env=None,
        runner_args=None,
    ):
        with tempfile.TemporaryDirectory(prefix="formal_mini_guard_") as root:
            fake_bin = os.path.join(root, "bin")
            os.makedirs(fake_bin)
            launch_log = os.path.join(root, "setsid.log")
            self._write_executable(
                os.path.join(fake_bin, "nvidia-smi"),
                "#!/bin/bash\nprintf '%s\\n' " + repr(gpu_state) + "\n",
            )
            self._write_executable(
                os.path.join(fake_bin, "setsid"),
                "#!/bin/bash\nprintf '%s\\n' \"$*\" >> \"${FAKE_SETSID_LOG}\"\n",
            )
            env = os.environ.copy()
            env.update(
                {
                    "PATH": fake_bin + os.pathsep + env["PATH"],
                    "FAKE_SETSID_LOG": launch_log,
                    "MINI_RESULTS_DIR": os.path.join(root, "results"),
                    "MINI_THERMAL_POLL_SECONDS": "1",
                }
            )
            env.update(extra_env or {})
            command = [
                "bash",
                os.path.join(ROOT, "test/mini-test/run_formal_mini_8gb.sh"),
                *(runner_args or ["vae"]),
            ]
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=15,
                check=False,
            )
            launched = Path(launch_log).read_text(encoding="utf-8") if os.path.exists(launch_log) else ""
            return result, launched

    def test_training_script_supports_sensor_aware_root_and_ir_linking(self):
        script = self._read("test/mini-test/train_minimal.sh")

        self.assertIn('PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-', script)
        self.assertIn('SRC_IR_DIR="${SRC_SCENE_DIR}/ir_image"', script)
        self.assertIn('DST_IR_DIR="${DST_SCENE_DIR}/ir_image"', script)
        self.assertIn('ln -s "${SRC_IR_PATH}"', script)
        self.assertIn('preprocess_policy.json', script)

    def test_mini_cd_config_uses_active_ema_consistency_fields(self):
        script = self._read("test/mini-test/train_minimal.sh")

        for token in (
            "'initialization_model_path'",
            "'training_semantics'",
            "'num_scales'",
            "'ema_rate'",
            "'sigma_min'",
            "'sigma_max'",
            "'rho'",
        ):
            self.assertIn(token, script)
        for unused_token in (
            "'training_mode'",
            "'target_ema_mode'",
            "'start_ema'",
            "'scale_mode'",
            "'start_scales'",
            "'end_scales'",
            "'distill_steps_per_iter'",
            "'loss_norm'",
        ):
            self.assertNotIn(unused_token, script)

    def test_inference_script_accepts_matching_data_and_result_roots(self):
        script = self._read("test/mini-test/inference_minimal.sh")

        self.assertIn('PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-', script)
        self.assertIn('MINI_RESULTS_DIR="${MINI_RESULTS_DIR:-', script)
        self.assertIn('--save_uncertainty', script)

    def test_formal_mini_inference_uses_v2_deployment_view_without_truth(self):
        """formal mini smoke 必须读取 0--80 m deployment view 并显式授权 mini 权重。"""
        script = self._read("test/mini-test/inference_minimal.sh")

        for fragment in (
            "NTU4DRadLM_Deploy_formal_v2_80m_86p8_v1",
            'DEFAULT_SOURCE_PC_RANGE="0,-20,-6,80,20,10"',
            'DEFAULT_MODEL_PC_RANGE="0,-20,-6,80,20,10"',
            'CALIBRATION_DIR="${CALIBRATION_DIR:-${ROOT_DIR}/Data/config}"',
            '--deployment_scene_dir',
            '--calibration_dir',
            '--allow_formal_mini_checkpoint',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, script)
        self.assertNotIn("NTU4DRadLM_Pre_sensor_aware_p1_04_candidate", script)
        self.assertNotIn('DEFAULT_MODEL_PC_RANGE="0,-20,-6,120,20,10"', script)

    def test_mini_scripts_explicitly_select_legacy_or_formal_radar_protocol(self):
        train_script = self._read("test/mini-test/train_minimal.sh")
        inference_script = self._read("test/mini-test/inference_minimal.sh")

        for script in (train_script, inference_script):
            self.assertIn('MINI_RADAR_PROTOCOL="${MINI_RADAR_PROTOCOL:-legacy}"', script)
            self.assertIn('case "${MINI_RADAR_PROTOCOL}" in', script)
            self.assertIn("--allow_legacy_radar_units", script)
        self.assertIn("RADAR_PROTOCOL_ARGS", train_script)
        self.assertIn("RADAR_PROTOCOL_ARGS", inference_script)
        self.assertIn(
            "cfg['data']['radar_normalization_path'] = radar_normalization_path",
            train_script,
        )
        self.assertIn(
            "cfg['data']['doppler_scale_mps'] = float(doppler_scale_mps)",
            train_script,
        )
        self.assertIn("formal_mini_chain_v2", train_script)
        self.assertIn("temporal_split_artifact", train_script)
        self.assertIn("data_protocol_path", train_script)
        self.assertIn("mini_train_frames_per_scene", train_script)
        self.assertIn("mini_validation_frames_per_scene", train_script)
        self.assertIn(
            "PYTHON_CMD=(conda run --no-capture-output -n Radar-Diffusion python)",
            train_script,
        )

    def test_formal_preflight_rejects_bad_artifact_sha_before_writing(self):
        script_path = os.path.join(ROOT, "test/mini-test/train_minimal.sh")
        artifact_path = os.path.join(
            ROOT,
            "diffusion_consistency_radar/config/",
            "radar_normalization_garden_32x128x128_80m_train80_purge3s_86p8_v2.json",
        )
        with tempfile.TemporaryDirectory(prefix="formal_bad_sha_") as results_dir:
            scratch = os.path.join(results_dir, ".tmp_vae_train_dataset")
            config_path = os.path.join(results_dir, "mini_vae_config.yaml")
            env = os.environ.copy()
            env.update(
                {
                    "PYTHON_BIN": sys.executable,
                    "MINI_PREFLIGHT_ONLY": "1",
                    "MINI_RADAR_PROTOCOL": "formal",
                    "MINI_RADAR_NORMALIZATION_PATH": artifact_path,
                    "EXPECTED_FORMAL_ARTIFACT_SHA256": "0" * 64,
                    "PREPROCESSED_ROOT": os.path.join(
                        ROOT, "Data/NTU4DRadLM_Pre_formal_v2_80m_86p8_v1"
                    ),
                    "MINI_DATASET_DIR": os.path.join(
                        ROOT, "Data/NTU4DRadLM_Pre_formal_v2_80m_86p8_v1"
                    ),
                    "MINI_RESULTS_DIR": results_dir,
                    "MINI_CONFIG_PATH": config_path,
                    "MINI_REQUIRE_FRESH_SCRATCH": "0",
                    "MINI_REQUIRE_FRESH_CONFIG": "1",
                }
            )
            result = subprocess.run(
                ["bash", script_path, "vae"],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=15,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SHA-256 mismatch", result.stdout)
            self.assertFalse(os.path.exists(scratch))
            self.assertFalse(os.path.exists(config_path))

    def test_formal_preflight_validates_parent_checkpoint_identity_before_exit(self):
        """LDM/CD 预检必须在无输出退出前校验父 checkpoint，而非只检查文件存在。"""
        script = self._read("test/mini-test/train_minimal.sh")

        for fragment in (
            "assert_checkpoint_training_identity",
            "build_formal_mini_selection",
            "checkpoint_state_dict",
            "safe_torch_load",
            'parent_checkpoints["ldm"].get("vae_checkpoint_sha256")',
            "Formal mini parent checkpoint validated",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, script)
        self.assertLess(
            script.index("assert_checkpoint_training_identity"),
            script.index("Mini training preflight passed; no scratch/config/output was created."),
        )

    def test_formal_mini_8gb_runner_has_short_single_stage_hardware_guards(self):
        script = self._read("test/mini-test/run_formal_mini_8gb.sh")

        for fragment in (
            'MODE="${1:-vae}"',
            'PROFILE="${2:-smoke}"',
            'if [[ "$#" -gt 2 ]]; then',
            'MINI_TRAIN_FRAMES_PER_SCENE="${MINI_TRAIN_FRAMES_PER_SCENE:-${PROFILE_TRAIN_FRAMES_PER_SCENE}}"',
            'MINI_VALIDATION_FRAMES_PER_SCENE="${MINI_VALIDATION_FRAMES_PER_SCENE:-${PROFILE_VALIDATION_FRAMES_PER_SCENE}}"',
            'MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-1}"',
            'MINI_NUM_WORKERS="${MINI_NUM_WORKERS:-0}"',
            'MINI_MAX_STAGE_MINUTES="${MINI_MAX_STAGE_MINUTES:-${PROFILE_MAX_STAGE_MINUTES}}"',
            'MINI_MIN_FREE_GPU_MEMORY_MIB="${MINI_MIN_FREE_GPU_MEMORY_MIB:-${PROFILE_MIN_FREE_GPU_MEMORY_MIB}}"',
            'nvidia-smi',
            'formal_mini_chain_v2',
            'NTU4DRadLM_Pre_formal_v2_80m_86p8_v1',
            'radar_normalization_garden_32x128x128_80m_train80_purge3s_86p8_v2.json',
            'temporal_split_garden_train80_purge3s_v1.json',
            'formal_data_protocol_garden_train80_purge3s_v1.json',
            'MINI_RADAR_PROTOCOL="formal"',
            'MINI_CONFIG_PATH="${MINI_RESULTS_DIR}/mini_${MODE}_config.yaml"',
            'MINI_REQUIRE_FRESH_CONFIG="1"',
            'MINI_PREFLIGHT_ONLY="${MINI_PREFLIGHT_ONLY:-0}"',
            'TRAIN_SCENES_OVERRIDE="garden"',
            'EXPECTED_FORMAL_ARTIFACT_SHA256="11f59d84cc186c39256c112154faf458ec9ead5fec9b08b997abd5058b68e97c"',
            'MINI_VAE_LATENT_DIM=""',
            'Formal mini 8 GB preflight passed; training was not started.',
            'bash "${TRAIN_SCRIPT}" "${MODE}"',
            'echo "epochs/batch: ${ACTIVE_STAGE_EPOCHS}/${MINI_BATCH_SIZE}"',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, script)
        self.assertIn('vae|ldm|cd)', script)
        self.assertNotIn('all_with_cd)', script)
        self.assertNotIn('NTU4DRadLM_Pre_sensor_aware_p1_04_candidate', script)
        self.assertNotIn('formal_mini_chain_v1', script)
        self.assertLess(
            script.index('Formal mini 8 GB preflight passed; training was not started.'),
            script.index('setsid bash "${TRAIN_SCRIPT}" "${MODE}"'),
        )
        train_script = self._read("test/mini-test/train_minimal.sh")
        self.assertLess(
            train_script.index("Mini training preflight passed; no scratch/config/output was created."),
            train_script.index('mkdir -- "${MINI_DATASET_DIR}"'),
        )

    def test_short_train_profile_is_isolated_and_more_conservative(self):
        """3 epoch short train 必须使用独立结果根，并收紧温度而非覆盖 smoke。"""
        script = self._read("test/mini-test/run_formal_mini_8gb.sh")

        for fragment in (
            'PROFILE="${2:-smoke}"',
            'short_train)',
            'PROFILE_VAE_EPOCHS=3',
            'PROFILE_MAX_GPU_TEMP_C=75',
            'formal_mini_v2_80m_8gb_short_v1',
            'short_train 目前只允许 VAE',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, script)

        result, launched = self._run_guarded_runner_with_fake_gpu(
            "NVIDIA GeForce RTX 4070 Laptop GPU, 8188, 7000, 50",
            runner_args=["vae", "short_train"],
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("profile: short_train", result.stdout)
        self.assertIn("GPU start/max temperature: 50/75 C", result.stdout)
        self.assertIn("epochs/batch: 3/1", result.stdout)
        self.assertIn("train_minimal.sh vae", launched)

    def test_short_train_rejects_non_vae_and_weaker_temperature_limit(self):
        """short profile 不得扩散到 LDM/CD，也不得把温度上限改回 80 C。"""
        result, launched = self._run_guarded_runner_with_fake_gpu(
            "NVIDIA GeForce RTX 4070 Laptop GPU, 8188, 7000, 50",
            runner_args=["ldm", "short_train"],
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("short_train 目前只允许 VAE", result.stdout)
        self.assertEqual(launched, "")

        result, launched = self._run_guarded_runner_with_fake_gpu(
            "NVIDIA GeForce RTX 4070 Laptop GPU, 8188, 7000, 50",
            {"MINI_MAX_GPU_TEMP_C": "80"},
            runner_args=["vae", "short_train"],
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("不得提高 75 C", result.stdout)
        self.assertEqual(launched, "")

    def test_medium_train_profile_fixes_500_frames_and_20_epochs(self):
        """笔记本 medium profile 必须固定 400/100 帧、20 epoch 和独立结果根。"""
        script = self._read("test/mini-test/run_formal_mini_8gb.sh")

        for fragment in (
            "medium_train)",
            "PROFILE_TRAIN_FRAMES_PER_SCENE=400",
            "PROFILE_VALIDATION_FRAMES_PER_SCENE=100",
            "PROFILE_VAE_EPOCHS=20",
            "PROFILE_LDM_EPOCHS=20",
            "PROFILE_CD_EPOCHS=20",
            "PROFILE_MAX_GPU_TEMP_C=72",
            "PROFILE_MAX_START_TEMP_C=55",
            "PROFILE_MAX_STAGE_MINUTES=180",
            "PROFILE_MIN_FREE_GPU_MEMORY_MIB=6500",
            'PROFILE_REQUIRED_GPU_NAME="NVIDIA GeForce RTX 4070 Laptop GPU"',
            "formal_medium_v2_80m_laptop_500f_20ep_v2",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, script)

        result, launched = self._run_guarded_runner_with_fake_gpu(
            "NVIDIA GeForce RTX 4070 Laptop GPU, 8188, 7000, 50",
            runner_args=["vae", "medium_train"],
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("profile: medium_train", result.stdout)
        self.assertIn("train/validation frames per scene: 400/100", result.stdout)
        self.assertIn("selected frames per scene: 500", result.stdout)
        self.assertIn("GPU start/max temperature: 50/72 C", result.stdout)
        self.assertIn("epochs/batch: 20/1", result.stdout)
        self.assertIn("max stage runtime: 180 min", result.stdout)
        self.assertIn("train_minimal.sh vae", launched)

    def test_guarded_runner_uses_stable_allocator_without_expandable_segments(self):
        """PyTorch 2.4.1 的 laptop 入口不得再次启用已触发内部断言的 allocator。"""
        runner = self._read("test/mini-test/run_formal_mini_8gb.sh")
        memory_efficient = self._read(
            "diffusion_consistency_radar/cm/memory_efficient.py"
        )
        train_script = self._read("test/mini-test/train_minimal.sh")

        self.assertIn(
            'export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"', runner
        )
        self.assertNotIn("expandable_segments", runner)
        self.assertNotIn("expandable_segments", memory_efficient)
        self.assertIn(
            "cfg['hardware']['cuda_allocator_conf'] = allocator_conf",
            train_script,
        )

        result, launched = self._run_guarded_runner_with_fake_gpu(
            "NVIDIA GeForce RTX 4070 Laptop GPU, 8188, 7000, 50",
            {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            runner_args=["vae", "medium_train"],
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("CUDA allocator: max_split_size_mb:128", result.stdout)
        self.assertIn("train_minimal.sh vae", launched)

    def test_medium_train_profile_rejects_data_or_guard_drift(self):
        """500 帧档不得被环境变量改成其他帧数或放宽笔记本保护门槛。"""
        cases = (
            (
                {"MINI_TRAIN_FRAMES_PER_SCENE": "399"},
                "固定 train/validation frames per scene=400/100",
            ),
            ({"MINI_MAX_GPU_TEMP_C": "73"}, "不得提高 72 C"),
            ({"MINI_MAX_STAGE_MINUTES": "181"}, "单阶段时长不得高于 180 分钟"),
            (
                {"MINI_MIN_FREE_GPU_MEMORY_MIB": "6000"},
                "可用显存 6500 MiB 门槛",
            ),
        )
        for extra_env, expected in cases:
            with self.subTest(extra_env=extra_env):
                result, launched = self._run_guarded_runner_with_fake_gpu(
                    "NVIDIA GeForce RTX 4070 Laptop GPU, 8188, 7000, 50",
                    extra_env,
                    runner_args=["vae", "medium_train"],
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stdout)
                self.assertEqual(launched, "")

        result, launched = self._run_guarded_runner_with_fake_gpu(
            "NVIDIA GeForce RTX 4090, 24564, 23000, 40",
            runner_args=["vae", "medium_train"],
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("仅允许 NVIDIA GeForce RTX 4070 Laptop GPU", result.stdout)
        self.assertEqual(launched, "")

    def test_guarded_runner_rejects_unknown_profile_and_extra_arguments(self):
        """未知 profile 和多余位置参数都必须在读取 GPU 或启动训练前拒绝。"""
        result, launched = self._run_guarded_runner_with_fake_gpu(
            "NVIDIA GeForce RTX 4070 Laptop GPU, 8188, 7000, 50",
            runner_args=["vae", "unknown"],
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("profile 必须为 smoke、short_train 或 medium_train", result.stdout)
        self.assertEqual(launched, "")

        result, launched = self._run_guarded_runner_with_fake_gpu(
            "NVIDIA GeForce RTX 4070 Laptop GPU, 8188, 7000, 50",
            runner_args=["vae", "smoke", "unexpected"],
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("最多接受阶段和 profile 两个位置参数", result.stdout)
        self.assertEqual(launched, "")

    def test_formal_mini_8gb_runner_passes_safe_fake_gpu_without_real_training(self):
        result, launched = self._run_guarded_runner_with_fake_gpu(
            "NVIDIA GeForce RTX 4070 Laptop GPU, 8188, 7000, 50"
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Formal mini vae completed", result.stdout)
        self.assertIn("train_minimal.sh vae", launched)

    def test_formal_mini_8gb_runner_rejects_hot_start_before_launch(self):
        result, launched = self._run_guarded_runner_with_fake_gpu(
            "NVIDIA GeForce RTX 4070 Laptop GPU, 8188, 7000, 70"
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("高于启动门槛", result.stdout)
        self.assertEqual(launched, "")

    def test_formal_mini_8gb_runner_refuses_weaker_temperature_limit(self):
        result, launched = self._run_guarded_runner_with_fake_gpu(
            "NVIDIA GeForce RTX 4070 Laptop GPU, 8188, 7000, 50",
            {"MINI_MAX_GPU_TEMP_C": "90"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("不得提高 80 C", result.stdout)
        self.assertEqual(launched, "")

    def test_formal_launchers_never_enable_legacy_radar_units(self):
        formal_launchers = (
            "diffusion_consistency_radar/launch/train_unified.sh",
            "diffusion_consistency_radar/launch/inference_ldm.sh",
            "diffusion_consistency_radar/launch/inference_cd.sh",
            "diffusion_consistency_radar/launch/inference_uniified.sh",
        )
        for launcher in formal_launchers:
            with self.subTest(launcher=launcher):
                self.assertNotIn(
                    "--allow_legacy_radar_units",
                    self._read(launcher),
                )

    def test_formal_training_launcher_binds_candidate_protocol_and_explicit_resume(self):
        """正式训练隔离新结果，已有结果只能显式授权恢复。"""
        script = self._read(
            "diffusion_consistency_radar/launch/train_unified.sh"
        )

        self.assertIn("NTU4DRadLM_Pre_formal_v2_80m_86p8_v1", script)
        self.assertIn(
            "radar_normalization_garden_32x128x128_80m_train80_purge3s_86p8_v2.json",
            script,
        )
        self.assertIn("formal_v2_80m_86p8_v1", script)
        self.assertIn('ALLOW_RESUME="${ALLOW_RESUME:-0}"', script)
        self.assertIn('if [ "${ALLOW_RESUME}" != "1" ]; then', script)
        self.assertIn("拒绝隐式续训", script)
        self.assertIn("cfg['data']['radar_normalization_path']", script)
        self.assertIn("cfg['data']['doppler_scale_mps'] = 86.8", script)
        self.assertNotIn("Please train VAE first: sh ", script)
        self.assertIn(
            'EXPECTED_ARTIFACT_SHA256="${EXPECTED_ARTIFACT_SHA256:-${YAML_EXPECTED_ARTIFACT_SHA256}}"',
            script,
        )
        self.assertIn("temporal_split_garden_train80_purge3s_v1.json", script)
        self.assertIn("formal_data_protocol_garden_train80_purge3s_v1.json", script)
        self.assertIn("cfg['data']['require_persisted_observed_mask'] = True", script)
        self.assertIn("cfg['data']['require_radar_statistics'] = True", script)
        self.assertIn(
            'CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-${YAML_CUDA_DEVICES}}}"',
            script,
        )
        self.assertIn('PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"', script)
        self.assertIn(
            'VAE_EPOCHS="${VAE_EPOCHS:-${FORMAL_EPOCHS:-${YAML_VAE_EPOCHS}}}"',
            script,
        )
        self.assertIn(
            'LDM_EPOCHS="${LDM_EPOCHS:-${FORMAL_EPOCHS:-${YAML_LDM_EPOCHS}}}"',
            script,
        )
        self.assertIn(
            'CD_EPOCHS="${CD_EPOCHS:-${FORMAL_EPOCHS:-${YAML_CD_EPOCHS}}}"',
            script,
        )
        self.assertIn('FORMAL_TRAIN_FRAMES_PER_EPOCH="${FORMAL_TRAIN_FRAMES_PER_EPOCH:-}"', script)
        self.assertIn('FORMAL_VALIDATION_FRAMES_PER_EPOCH="${FORMAL_VALIDATION_FRAMES_PER_EPOCH:-}"', script)
        self.assertIn('export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"', script)
        self.assertIn('if [ "${PREFLIGHT_ONLY}" = "1" ]; then', script)
        self.assertIn("Radar statistics 预检通过", script)
        self.assertIn("cfg['data'].pop('mini_train_frames_per_scene', None)", script)
        self.assertIn("cfg['data'].pop('mini_validation_frames_per_scene', None)", script)
        self.assertIn("cfg[stage]['epochs'] = int(stage_values[stage]['epochs'])", script)
        self.assertIn(
            "cfg[stage]['train_frames_per_epoch'] = int(stage_values[stage]['train_frames'])",
            script,
        )
        self.assertIn(
            "cfg[stage]['validation_frames_per_epoch'] = int(stage_values[stage]['validation_frames'])",
            script,
        )
        self.assertIn(
            "cfg['hardware']['cuda_allocator_conf'] = allocator_conf", script
        )
        self.assertIn('echo "CUDA allocator: ${PYTORCH_CUDA_ALLOC_CONF}"', script)
        self.assertIn('echo "Formal epochs: vae=${VAE_EPOCHS}, ldm=${LDM_EPOCHS}, cd=${CD_EPOCHS}"', script)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=0,1", script)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=0 python", script)
        self.assertLess(
            script.index("拒绝隐式续训"),
            script.index("export CUDA_VISIBLE_DEVICES"),
        )
        self.assertNotIn('rm -rf "${TRAIN_DATASET_DIR}"', script)

    def test_inference_script_rejects_removed_oracle_environment(self):
        """旧 adaptive 环境变量必须在任何推理准备前给出迁移提示。"""
        script_path = os.path.join(ROOT, "test/mini-test/inference_minimal.sh")
        env = os.environ.copy()
        env["ADAPTIVE_OCC_FROM_TARGET"] = "1"
        result = subprocess.run(
            ["bash", script_path, "ldm"],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("diagnose_oracle_target_adaptation.py", result.stdout)

    def test_formal_launchers_bind_each_deployment_scene_directory(self):
        """单模型正式 launcher 不能引用未定义的 SCENE_DIR。"""
        for relative_path in (
            "diffusion_consistency_radar/launch/inference_ldm.sh",
            "diffusion_consistency_radar/launch/inference_cd.sh",
        ):
            script = self._read(relative_path)
            with self.subTest(relative_path=relative_path):
                self.assertIn(
                    '--deployment_scene_dir "${PREPROCESSED_ROOT}/${SCENE}"',
                    script,
                )
                self.assertNotIn('--deployment_scene_dir "${SCENE_DIR}"', script)


if __name__ == "__main__":
    unittest.main()
