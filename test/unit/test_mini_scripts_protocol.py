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

    def _run_guarded_runner_with_fake_gpu(self, gpu_state, extra_env=None):
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
            result = subprocess.run(
                ["bash", os.path.join(ROOT, "test/mini-test/run_formal_mini_8gb.sh"), "vae"],
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

    def test_inference_script_accepts_matching_data_and_result_roots(self):
        script = self._read("test/mini-test/inference_minimal.sh")

        self.assertIn('PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-', script)
        self.assertIn('MINI_RESULTS_DIR="${MINI_RESULTS_DIR:-', script)
        self.assertIn('--save_uncertainty', script)

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
        self.assertIn("formal_mini_chain_v1", train_script)
        self.assertIn(
            "PYTHON_CMD=(conda run --no-capture-output -n Radar-Diffusion python)",
            train_script,
        )

    def test_formal_preflight_rejects_bad_artifact_sha_before_writing(self):
        script_path = os.path.join(ROOT, "test/mini-test/train_minimal.sh")
        artifact_path = os.path.join(
            ROOT,
            "diffusion_consistency_radar/config/",
            "radar_normalization_garden_32x128x128_full120_86p8_v1.json",
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
                    "MINI_MODEL_PC_RANGE": "0,-20,-6,120,20,10",
                    "MINI_RESULTS_DIR": results_dir,
                    "MINI_DATASET_DIR": scratch,
                    "MINI_CONFIG_PATH": config_path,
                    "MINI_REQUIRE_FRESH_SCRATCH": "1",
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

    def test_formal_mini_8gb_runner_has_short_single_stage_hardware_guards(self):
        script = self._read("test/mini-test/run_formal_mini_8gb.sh")

        for fragment in (
            'MODE="${1:-vae}"',
            'SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE:-16}"',
            'MINI_VAE_EPOCHS="${MINI_VAE_EPOCHS:-1}"',
            'MINI_LDM_EPOCHS="${MINI_LDM_EPOCHS:-1}"',
            'MINI_CD_EPOCHS="${MINI_CD_EPOCHS:-1}"',
            'MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-1}"',
            'MINI_NUM_WORKERS="${MINI_NUM_WORKERS:-0}"',
            'MINI_MAX_GPU_TEMP_C="${MINI_MAX_GPU_TEMP_C:-80}"',
            'MINI_MAX_START_TEMP_C="${MINI_MAX_START_TEMP_C:-65}"',
            'MINI_MAX_STAGE_MINUTES="${MINI_MAX_STAGE_MINUTES:-20}"',
            'MINI_MIN_FREE_GPU_MEMORY_MIB="${MINI_MIN_FREE_GPU_MEMORY_MIB:-6000}"',
            'nvidia-smi',
            'formal_mini_chain_v1',
            'NTU4DRadLM_Pre_sensor_aware_p1_04_candidate',
            'radar_normalization_garden_32x128x128_full120_86p8_v1.json',
            'MINI_RADAR_PROTOCOL="formal"',
            'MINI_DATASET_DIR="${MINI_RESULTS_DIR}/.tmp_${MODE}_train_dataset"',
            'MINI_CONFIG_PATH="${MINI_RESULTS_DIR}/mini_${MODE}_config.yaml"',
            'MINI_REQUIRE_FRESH_SCRATCH="1"',
            'MINI_REQUIRE_FRESH_CONFIG="1"',
            'MINI_PREFLIGHT_ONLY="${MINI_PREFLIGHT_ONLY:-0}"',
            'TRAIN_SCENES_OVERRIDE="garden"',
            'EXPECTED_FORMAL_ARTIFACT_SHA256="2c9c92650b98ec686d621b53eccb5e7f376cb6b8ea1047d4fb594349af90c4d5"',
            'MINI_VAE_LATENT_DIM=""',
            'Formal mini 8 GB preflight passed; training was not started.',
            'bash "${TRAIN_SCRIPT}" "${MODE}"',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, script)
        self.assertIn('vae|ldm|cd)', script)
        self.assertNotIn('all_with_cd)', script)
        self.assertLess(
            script.index('Formal mini 8 GB preflight passed; training was not started.'),
            script.index('setsid bash "${TRAIN_SCRIPT}" "${MODE}"'),
        )
        train_script = self._read("test/mini-test/train_minimal.sh")
        self.assertLess(
            train_script.index("Mini training preflight passed; no scratch/config/output was created."),
            train_script.index('mkdir -- "${MINI_DATASET_DIR}"'),
        )

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

        self.assertIn("NTU4DRadLM_Pre_sensor_aware_p1_04_candidate", script)
        self.assertIn(
            "radar_normalization_garden_32x128x128_full120_86p8_v1.json",
            script,
        )
        self.assertIn("formal_p1_04_full120_86p8_v1", script)
        self.assertIn('ALLOW_RESUME="${ALLOW_RESUME:-0}"', script)
        self.assertIn('if [ "${ALLOW_RESUME}" != "1" ]; then', script)
        self.assertIn("拒绝隐式续训", script)
        self.assertIn("cfg['data']['radar_normalization_path']", script)
        self.assertIn("cfg['data']['doppler_scale_mps'] = 86.8", script)
        self.assertNotIn("Please train VAE first: sh ", script)
        self.assertIn(
            'EXPECTED_ARTIFACT_SHA256="2c9c92650b98ec686d621b53eccb5e7f376cb6b8ea1047d4fb594349af90c4d5"',
            script,
        )
        self.assertLess(
            script.index("拒绝隐式续训"),
            script.index("CUDA_VISIBLE_DEVICES"),
        )
        self.assertLess(
            script.index("load_radar_normalization_artifact"),
            script.index('rm -rf "${TRAIN_DATASET_DIR}"'),
        )

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


if __name__ == "__main__":
    unittest.main()
