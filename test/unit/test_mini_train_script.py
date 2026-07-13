#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""静态验证最小训练脚本的实验配置透传契约，不启动任何训练。"""

import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import yaml


TEST_ROOT = Path(__file__).resolve().parents[1]

SCRIPT_PATH = (TEST_ROOT / "mini-test" / "train_minimal.sh").resolve()
LDM_VERTICAL_RUNNER = (
    TEST_ROOT / "mini-test" / "run_ldm_vertical_experiment.sh"
).resolve()
Z64_UPPER_BOUND_RUNNER = (
    TEST_ROOT / "mini-test" / "run_vae_z64_upper_bound.sh"
)
Z64_LDM_V5_RUNNER = (
    TEST_ROOT / "mini-test" / "run_ldm_z64_v5_experiment.sh"
)
Z64_LDM_V6_RUNNER = (
    TEST_ROOT / "mini-test" / "run_ldm_z64_v6_top_experiment.sh"
)
Z64_LDM_V7_RUNNER = (
    TEST_ROOT / "mini-test" / "run_ldm_z64_v7_ir_experiment.sh"
)
Z64_LDM_V8_RUNNER = (
    TEST_ROOT / "mini-test" / "run_ldm_z64_v8_balanced_experiment.sh"
).resolve()
Z64_LDM_V9_RUNNER = (
    TEST_ROOT / "mini-test" / "run_ldm_z64_v9_screen.sh"
).resolve()
Z64_LDM_V9A_WRAPPER = (
    TEST_ROOT / "mini-test" / "run_ldm_z64_v9a_top_screen.sh"
).resolve()
Z64_LDM_V9B_WRAPPER = (
    TEST_ROOT / "mini-test" / "run_ldm_z64_v9b_irneg_screen.sh"
).resolve()
Z64_LDM_V10_RUNNER = (
    TEST_ROOT / "mini-test" / "run_ldm_z64_v10_column_experiment.sh"
).resolve()
SHELL_SCRIPTS_FOR_BASH_N = (
    SCRIPT_PATH,
    LDM_VERTICAL_RUNNER,
    Z64_LDM_V8_RUNNER,
    Z64_LDM_V9_RUNNER,
    Z64_LDM_V9A_WRAPPER,
    Z64_LDM_V9B_WRAPPER,
    Z64_LDM_V10_RUNNER,
)

CONFIG_ARGUMENT_CONTRACT = [
    ("DEFAULT_CONFIG_PATH", "src_cfg"),
    ("MINI_CONFIG_PATH", "dst_cfg"),
    ("MINI_DATASET_DIR", "dataset_dir"),
    ("MINI_BATCH_SIZE", "batch_size"),
    ("MINI_NUM_WORKERS", "num_workers"),
    ("MINI_USE_AUG", "use_aug"),
    ("MINI_VAE_EPOCHS", "vae_epochs"),
    ("MINI_LDM_EPOCHS", "ldm_epochs"),
    ("MINI_CD_EPOCHS", "cd_epochs"),
    ("MINI_GRAD_ACCUM", "grad_accum"),
    ("MINI_RESULTS_DIR", "results_dir"),
    ("MINI_TARGET_SIZE", "target_size_raw"),
    ("MINI_SOURCE_PC_RANGE", "source_pc_range_raw"),
    ("MINI_MODEL_PC_RANGE", "model_pc_range_raw"),
    ("MINI_VAE_CONFIG_TYPE", "vae_config_type"),
    ("MINI_VAE_LATENT_DIM", "vae_latent_dim"),
    ("MINI_VAE_OCC_LOSS", "vae_occ_loss"),
    ("MINI_TRAIN_SPLIT", "train_split"),
    ("MINI_SPLIT_SEED", "split_seed"),
    ("MINI_LDM_DECODED_WEIGHT", "ldm_decoded_weight"),
    ("MINI_LDM_DECODED_FP_WEIGHT", "ldm_decoded_fp_weight"),
    ("MINI_LDM_DECODED_MASS_WEIGHT", "ldm_decoded_mass_weight"),
    ("MINI_LDM_HEIGHT_WEIGHT", "ldm_height_weight"),
    ("MINI_LDM_TOP_WEIGHT", "ldm_top_weight"),
    ("MINI_LDM_TOP_OVERSHOOT_WEIGHT", "ldm_top_overshoot_weight"),
    ("MINI_LDM_CONTINUITY_WEIGHT", "ldm_continuity_weight"),
    ("MINI_LDM_DENSITY_WEIGHT", "ldm_density_weight"),
    ("MINI_LDM_IR_FRUSTUM_OCC_WEIGHT", "ldm_ir_frustum_occ_weight"),
    ("MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT", "ldm_ir_frustum_negative_weight"),
    ("MINI_LDM_IR_FRUSTUM_TOP_WEIGHT", "ldm_ir_frustum_top_weight"),
    ("MINI_LDM_UNCERTAINTY_WEIGHT", "ldm_uncertainty_weight"),
    ("MINI_LDM_COLUMN_POSITIVE_WEIGHT", "ldm_column_positive_weight"),
    ("MINI_LDM_COLUMN_NEGATIVE_WEIGHT", "ldm_column_negative_weight"),
    ("MINI_LDM_COLUMN_TEMPERATURE", "ldm_column_temperature"),
    ("MINI_REQUIRE_FRESH_CONFIG", "require_fresh_config"),
]


class MiniTrainShellSyntaxTest(unittest.TestCase):
    """实际调用 bash -n 检查 Task3 shell 入口语法。"""

    def test_task3_shell_scripts_pass_bash_syntax_check(self):
        for script_path in SHELL_SCRIPTS_FOR_BASH_N:
            with self.subTest(script=script_path.name):
                result = subprocess.run(
                    ["bash", "-n", str(script_path)],
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    msg=f"bash -n failed for {script_path}: {result.stderr}",
                )


class MiniTrainScriptTest(unittest.TestCase):
    """检查 32 帧实验参数从环境变量到 YAML 的完整传递。"""

    @classmethod
    def setUpClass(cls):
        cls.script = SCRIPT_PATH.read_text(encoding="utf-8")
        (
            cls.shell_argument_names,
            cls.python_argument_names,
            cls.config_generator,
        ) = cls._parse_config_call(cls.script)

    @staticmethod
    def _parse_config_call(script):
        matches = re.findall(
            r'"\$\{CONFIG_PYTHON_CMD\[@\]\}"\s+-\s+(.*?)\s+<<\'PY\'\n'
            r'(.*?)\nPY\n',
            script,
            flags=re.DOTALL,
        )
        matches = [
            match for match in matches
            if '"${DEFAULT_CONFIG_PATH}"' in match[0]
        ]
        if len(matches) != 1:
            raise AssertionError("无法唯一定位 YAML 配置生成器")
        shell_arguments, config_generator = matches[0]
        shell_names = re.findall(r'"\$\{([A-Z0-9_]+)\}"', shell_arguments)
        unpack_match = re.search(
            r"\(\s*(.*?)\s*\)\s*=\s*sys\.argv\[1:\d+\]",
            config_generator,
            flags=re.DOTALL,
        )
        if unpack_match is None:
            raise AssertionError("无法定位 YAML 配置生成器的 sys.argv 解包")
        python_names = re.findall(r"\b[a-z][a-z0-9_]*\b", unpack_match.group(1))
        return shell_names, python_names, config_generator

    @staticmethod
    def _assert_config_call_contract(script):
        shell_names, python_names, _generator = MiniTrainScriptTest._parse_config_call(
            script
        )
        actual_contract = list(zip(shell_names, python_names))
        if actual_contract != CONFIG_ARGUMENT_CONTRACT:
            raise AssertionError(
                f"配置生成参数顺序不匹配: {actual_contract!r}"
            )

    def _generate_config(self, latent_dim):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.yaml"
            output_path = temp_path / "generated.yaml"
            source_path.write_text("vae:\n  latent_dim: 4\n", encoding="utf-8")
            values = {
                "DEFAULT_CONFIG_PATH": str(source_path),
                "MINI_CONFIG_PATH": str(output_path),
                "MINI_DATASET_DIR": str(temp_path / "dataset"),
                "MINI_BATCH_SIZE": "1",
                "MINI_NUM_WORKERS": "0",
                "MINI_USE_AUG": "false",
                "MINI_VAE_EPOCHS": "1",
                "MINI_LDM_EPOCHS": "1",
                "MINI_CD_EPOCHS": "1",
                "MINI_GRAD_ACCUM": "1",
                "MINI_RESULTS_DIR": str(temp_path / "results"),
                "MINI_TARGET_SIZE": "32,128,128",
                "MINI_SOURCE_PC_RANGE": "0,-20,-6,120,20,10",
                "MINI_MODEL_PC_RANGE": "0,-20,-6,40,20,10",
                "MINI_VAE_CONFIG_TYPE": "ultra_lightweight",
                "MINI_VAE_LATENT_DIM": latent_dim,
                "MINI_VAE_OCC_LOSS": "bce_dice",
                "MINI_TRAIN_SPLIT": "0.8",
                "MINI_SPLIT_SEED": "42",
                "MINI_LDM_DECODED_WEIGHT": "0.875",
                "MINI_LDM_DECODED_FP_WEIGHT": "0.75",
                "MINI_LDM_DECODED_MASS_WEIGHT": "0.5",
                "MINI_LDM_HEIGHT_WEIGHT": "0.125",
                "MINI_LDM_TOP_WEIGHT": "0.25",
                "MINI_LDM_TOP_OVERSHOOT_WEIGHT": "0.1875",
                "MINI_LDM_CONTINUITY_WEIGHT": "0.375",
                "MINI_LDM_DENSITY_WEIGHT": "0.625",
                "MINI_LDM_IR_FRUSTUM_OCC_WEIGHT": "0.3125",
                "MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT": "0.5625",
                "MINI_LDM_IR_FRUSTUM_TOP_WEIGHT": "0.4375",
                "MINI_LDM_UNCERTAINTY_WEIGHT": "0.25",
                "MINI_LDM_COLUMN_POSITIVE_WEIGHT": "0.03125",
                "MINI_LDM_COLUMN_NEGATIVE_WEIGHT": "0.0625",
                "MINI_LDM_COLUMN_TEMPERATURE": "0.75",
                "MINI_REQUIRE_FRESH_CONFIG": "0",
            }
            arguments = [values[name] for name in self.shell_argument_names]
            subprocess.run(
                [sys.executable, "-", *arguments],
                input=self.config_generator,
                text=True,
                check=True,
                capture_output=True,
            )
            return yaml.safe_load(output_path.read_text(encoding="utf-8"))

    def test_shell_arguments_match_python_argv_unpacking(self):
        self._assert_config_call_contract(self.script)

    def test_argument_contract_rejects_swapped_shell_parameters(self):
        original = '"${MINI_BATCH_SIZE}" "${MINI_NUM_WORKERS}"'
        swapped = '"${MINI_NUM_WORKERS}" "${MINI_BATCH_SIZE}"'
        swapped_script = self.script.replace(original, swapped, 1)
        self.assertNotEqual(swapped_script, self.script)

        with self.assertRaisesRegex(AssertionError, "参数顺序不匹配"):
            self._assert_config_call_contract(swapped_script)

    def test_declares_experiment_defaults(self):
        expected_defaults = {
            "MINI_VAE_CONFIG_TYPE": "ultra_lightweight",
            "MINI_VAE_LATENT_DIM": "",
            "MINI_VAE_OCC_LOSS": "bce_dice",
            "MINI_TRAIN_SPLIT": "0.8",
            "MINI_SPLIT_SEED": "42",
            "MINI_LDM_DECODED_WEIGHT": "",
            "MINI_LDM_DECODED_FP_WEIGHT": "",
            "MINI_LDM_DECODED_MASS_WEIGHT": "",
            "MINI_LDM_HEIGHT_WEIGHT": "0.02",
            "MINI_LDM_TOP_WEIGHT": "0.0",
            "MINI_LDM_TOP_OVERSHOOT_WEIGHT": "0.0",
            "MINI_LDM_CONTINUITY_WEIGHT": "0.02",
            "MINI_LDM_DENSITY_WEIGHT": "0.0",
            "MINI_LDM_IR_FRUSTUM_OCC_WEIGHT": "0.0",
            "MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT": "0.0",
            "MINI_LDM_IR_FRUSTUM_TOP_WEIGHT": "0.0",
            "MINI_LDM_UNCERTAINTY_WEIGHT": "",
            "MINI_LDM_COLUMN_POSITIVE_WEIGHT": "0.0",
            "MINI_LDM_COLUMN_NEGATIVE_WEIGHT": "0.0",
            "MINI_LDM_COLUMN_TEMPERATURE": "1.0",
            "MINI_REQUIRE_FRESH_SCRATCH": "0",
            "MINI_REQUIRE_FRESH_CONFIG": "0",
        }
        for variable, default in expected_defaults.items():
            with self.subTest(variable=variable):
                declaration = f'{variable}="${{{variable}:-{default}}}"'
                self.assertIn(declaration, self.script)

    def test_passes_experiment_values_to_config_generator(self):
        for variable in (
            "MINI_VAE_CONFIG_TYPE",
            "MINI_VAE_LATENT_DIM",
            "MINI_VAE_OCC_LOSS",
            "MINI_TRAIN_SPLIT",
            "MINI_SPLIT_SEED",
            "MINI_LDM_DECODED_WEIGHT",
            "MINI_LDM_DECODED_FP_WEIGHT",
            "MINI_LDM_DECODED_MASS_WEIGHT",
            "MINI_LDM_HEIGHT_WEIGHT",
            "MINI_LDM_TOP_WEIGHT",
            "MINI_LDM_TOP_OVERSHOOT_WEIGHT",
            "MINI_LDM_CONTINUITY_WEIGHT",
            "MINI_LDM_DENSITY_WEIGHT",
            "MINI_LDM_IR_FRUSTUM_OCC_WEIGHT",
            "MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT",
            "MINI_LDM_IR_FRUSTUM_TOP_WEIGHT",
            "MINI_LDM_UNCERTAINTY_WEIGHT",
            "MINI_LDM_COLUMN_POSITIVE_WEIGHT",
            "MINI_LDM_COLUMN_NEGATIVE_WEIGHT",
            "MINI_LDM_COLUMN_TEMPERATURE",
        ):
            with self.subTest(variable=variable):
                self.assertRegex(
                    self.script,
                    rf'"\$\{{{variable}\}}".*<<\'PY\'',
                    msg=f"{variable} 未传给 YAML 配置生成器",
                )

    def test_writes_experiment_values_to_expected_yaml_fields(self):
        expected_assignments = (
            "cfg['vae']['config_type'] = vae_config_type",
            "cfg['vae']['latent_dim'] = int(vae_latent_dim)",
            "cfg['vae']['occupancy_loss_type'] = vae_occ_loss",
            "cfg['data']['train_split'] = float(train_split)",
            "cfg['data']['split_seed'] = int(split_seed)",
            "cfg['ldm']['decoded_loss_weight'] = float(ldm_decoded_weight)",
            "cfg['ldm']['decoded_false_positive_weight'] = float(ldm_decoded_fp_weight)",
            "cfg['ldm']['decoded_mass_weight'] = float(ldm_decoded_mass_weight)",
            "cfg['ldm']['decoded_height_distribution_weight'] = float(ldm_height_weight)",
            "cfg['ldm']['decoded_top_height_weight'] = float(ldm_top_weight)",
            "cfg['ldm']['decoded_top_overshoot_weight'] = float(ldm_top_overshoot_weight)",
            "cfg['ldm']['decoded_vertical_continuity_weight'] = float(ldm_continuity_weight)",
            "cfg['ldm']['decoded_density_weight'] = float(ldm_density_weight)",
            "cfg['ldm']['decoded_ir_frustum_occupancy_weight'] = float(ldm_ir_frustum_occ_weight)",
            "cfg['ldm']['decoded_ir_frustum_negative_weight'] = float(ldm_ir_frustum_negative_weight)",
            "cfg['ldm']['decoded_ir_frustum_top_weight'] = float(ldm_ir_frustum_top_weight)",
            "cfg['ldm']['uncertainty_loss_weight'] = float(ldm_uncertainty_weight)",
            "cfg['ldm']['decoded_column_positive_weight'] = float(ldm_column_positive_weight)",
            "cfg['ldm']['decoded_column_negative_weight'] = float(ldm_column_negative_weight)",
            "cfg['ldm']['decoded_column_temperature'] = float(ldm_column_temperature)",
        )
        for assignment in expected_assignments:
            with self.subTest(assignment=assignment):
                self.assertIn(assignment, self.script)

        self.assertRegex(
            self.script,
            r"if\s+vae_latent_dim\s*:",
            msg="空 MINI_VAE_LATENT_DIM 不应覆盖 preset 的 latent_dim",
        )

    def test_empty_latent_dim_removes_default_config_value(self):
        generated = self._generate_config("")

        self.assertNotIn("latent_dim", generated["vae"])

    def test_nonempty_latent_dim_writes_integer_override(self):
        generated = self._generate_config("8")

        self.assertEqual(generated["vae"]["latent_dim"], 8)
        self.assertIsInstance(generated["vae"]["latent_dim"], int)

    def test_ldm_structure_env_values_are_written_to_generated_yaml(self):
        generated = self._generate_config("")

        self.assertEqual(generated["ldm"]["decoded_loss_weight"], 0.875)
        self.assertEqual(generated["ldm"]["decoded_false_positive_weight"], 0.75)
        self.assertEqual(generated["ldm"]["decoded_mass_weight"], 0.5)
        self.assertEqual(generated["ldm"]["decoded_height_distribution_weight"], 0.125)
        self.assertEqual(generated["ldm"]["decoded_top_height_weight"], 0.25)
        self.assertEqual(generated["ldm"]["decoded_top_overshoot_weight"], 0.1875)
        self.assertEqual(generated["ldm"]["decoded_vertical_continuity_weight"], 0.375)
        self.assertEqual(generated["ldm"]["decoded_density_weight"], 0.625)
        self.assertEqual(generated["ldm"]["decoded_ir_frustum_occupancy_weight"], 0.3125)
        self.assertEqual(generated["ldm"]["decoded_ir_frustum_negative_weight"], 0.5625)
        self.assertEqual(generated["ldm"]["decoded_ir_frustum_top_weight"], 0.4375)
        self.assertEqual(generated["ldm"]["uncertainty_loss_weight"], 0.25)
        self.assertEqual(generated["ldm"]["decoded_column_positive_weight"], 0.03125)
        self.assertEqual(generated["ldm"]["decoded_column_negative_weight"], 0.0625)
        self.assertEqual(generated["ldm"]["decoded_column_temperature"], 0.75)

    def test_fresh_config_generator_exclusively_creates_new_file_and_rejects_race(self):
        shell_names, _, generator = self._parse_config_call(self.script)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source = temp_path / "source.yaml"
            source.write_text("vae: {}\n", encoding="utf-8")
            values = {name: "0" for name in shell_names}
            values.update({
                "DEFAULT_CONFIG_PATH": str(source),
                "MINI_CONFIG_PATH": str(temp_path / "fresh.yaml"),
                "MINI_DATASET_DIR": str(temp_path / "dataset"),
                "MINI_BATCH_SIZE": "1", "MINI_NUM_WORKERS": "0",
                "MINI_USE_AUG": "false", "MINI_VAE_EPOCHS": "1",
                "MINI_LDM_EPOCHS": "1", "MINI_CD_EPOCHS": "1",
                "MINI_GRAD_ACCUM": "1", "MINI_RESULTS_DIR": str(temp_path),
                "MINI_TARGET_SIZE": "32,128,128",
                "MINI_SOURCE_PC_RANGE": "0,-20,-6,120,20,10",
                "MINI_MODEL_PC_RANGE": "0,-20,-6,40,20,10",
                "MINI_VAE_CONFIG_TYPE": "ultra_lightweight",
                "MINI_VAE_LATENT_DIM": "", "MINI_VAE_OCC_LOSS": "bce_dice",
                "MINI_TRAIN_SPLIT": "0.8", "MINI_SPLIT_SEED": "42",
                "MINI_LDM_UNCERTAINTY_WEIGHT": "0",
                "MINI_LDM_COLUMN_TEMPERATURE": "1",
                "MINI_REQUIRE_FRESH_CONFIG": "1",
            })
            args = [values[name] for name in shell_names]
            first = subprocess.run(
                [sys.executable, "-", *args], input=generator,
                text=True, capture_output=True,
            )
            self.assertEqual(first.returncode, 0, msg=first.stdout + first.stderr)
            output = temp_path / "fresh.yaml"
            self.assertTrue(output.is_file())
            output.write_text("race-owner\n", encoding="utf-8")
            second = subprocess.run(
                [sys.executable, "-", *args], input=generator,
                text=True, capture_output=True,
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "race-owner\n")

    def test_prints_effective_experiment_values(self):
        for variable in (
            "MINI_VAE_CONFIG_TYPE",
            "MINI_VAE_LATENT_DIM",
            "MINI_VAE_OCC_LOSS",
            "MINI_TRAIN_SPLIT",
            "MINI_SPLIT_SEED",
            "MINI_LDM_DECODED_WEIGHT",
            "MINI_LDM_DECODED_FP_WEIGHT",
            "MINI_LDM_DECODED_MASS_WEIGHT",
            "MINI_LDM_HEIGHT_WEIGHT",
            "MINI_LDM_TOP_WEIGHT",
            "MINI_LDM_TOP_OVERSHOOT_WEIGHT",
            "MINI_LDM_CONTINUITY_WEIGHT",
            "MINI_LDM_DENSITY_WEIGHT",
            "MINI_LDM_IR_FRUSTUM_OCC_WEIGHT",
            "MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT",
            "MINI_LDM_IR_FRUSTUM_TOP_WEIGHT",
            "MINI_LDM_UNCERTAINTY_WEIGHT",
            "MINI_LDM_COLUMN_POSITIVE_WEIGHT",
            "MINI_LDM_COLUMN_NEGATIVE_WEIGHT",
            "MINI_LDM_COLUMN_TEMPERATURE",
        ):
            with self.subTest(variable=variable):
                self.assertRegex(
                    self.script,
                    rf'echo "[^"]*\$\{{{variable}(?::-[^}}]*)?\}}[^"]*"',
                    msg=f"setup 未打印 {variable} 的有效值",
                )

    def test_does_not_add_unused_patience_setting(self):
        self.assertNotIn("patience", self.script.lower())

    def test_cuda_selection_exports_one_consistent_value(self):
        for fragment in (
            'SELECTED_CUDA_DEVICES="${CUDA_DEVICES}"',
            'SELECTED_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}"',
            'SELECTED_CUDA_DEVICES="0"',
            'export CUDA_DEVICES="${SELECTED_CUDA_DEVICES}"',
            'export CUDA_VISIBLE_DEVICES="${SELECTED_CUDA_DEVICES}"',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.script)

    def test_mode_and_dataset_path_are_validated_before_destructive_setup(self):
        mode_guard = self.script.index('case "${MODE}" in')
        path_guard = self.script.index("validate_mini_dataset_dir")
        destructive_setup = self.script.index('rm -rf "${MINI_DATASET_DIR}"')
        self.assertLess(mode_guard, destructive_setup)
        self.assertLess(path_guard, destructive_setup)

    def test_dataset_delete_allowlist_is_explicit(self):
        for fragment in (
            'realpath -m -- "${MINI_DATASET_DIR}"',
            '"${MINI_DATASET_DIR}" == /tmp/*',
            '"${MINI_DATASET_DIR}" == "${ROOT_DIR}/test/"*',
            '[[ "$(basename "${MINI_DATASET_DIR}")" == .tmp_* ]]',
            '"${MINI_DATASET_DIR}" == "/"',
            '"${MINI_DATASET_DIR}" == "/tmp"',
            '"${MINI_DATASET_DIR}" == "${ROOT_DIR}"',
            '"${MINI_DATASET_DIR}" == "${normalized_preprocessed_root}"',
            '"${MINI_DATASET_DIR}" == "${normalized_results_dir}"',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.script)


class MiniTrainSafetyBehaviorTest(unittest.TestCase):
    """用 marker 验证参数错误不会进入数据集删除阶段。"""

    @staticmethod
    def _run(mode, dataset_dir):
        env = os.environ.copy()
        env.update(
            {
                "MINI_DATASET_DIR": str(dataset_dir),
                "TRAIN_SCENES_OVERRIDE": "missing_scene",
                "PREPROCESSED_ROOT": str(Path(dataset_dir).parent / "missing_source"),
            }
        )
        return subprocess.run(
            ["bash", str(SCRIPT_PATH), mode],
            cwd=SCRIPT_PATH.parents[2],
            env=env,
            text=True,
            capture_output=True,
        )

    def test_invalid_mode_does_not_delete_safe_scratch_marker(self):
        with tempfile.TemporaryDirectory(prefix="radar_task3_mode_") as temp_dir:
            dataset_dir = Path(temp_dir) / "scratch"
            dataset_dir.mkdir()
            marker = dataset_dir / "keep.marker"
            marker.write_text("keep", encoding="utf-8")

            result = self._run("invalid-mode", dataset_dir)

            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(marker.exists(), msg=result.stdout + result.stderr)
            self.assertIn("Usage:", result.stdout + result.stderr)

    def test_unsafe_repo_test_path_does_not_delete_marker(self):
        with tempfile.TemporaryDirectory(
            prefix="unsafe_task3_", dir=SCRIPT_PATH.parents[1]
        ) as temp_dir:
            dataset_dir = Path(temp_dir)
            marker = dataset_dir / "keep.marker"
            marker.write_text("keep", encoding="utf-8")

            result = self._run("vae", dataset_dir)

            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(marker.exists(), msg=result.stdout + result.stderr)
            self.assertIn("unsafe MINI_DATASET_DIR", result.stdout + result.stderr)

    def _run_fresh_scratch(self, dataset_dir, results_dir):
        env = os.environ.copy()
        env.update({
            "MINI_REQUIRE_FRESH_SCRATCH": "1",
            "MINI_DATASET_DIR": str(dataset_dir),
            "MINI_RESULTS_DIR": str(results_dir),
            "TRAIN_SCENES_OVERRIDE": "missing_scene",
            "PREPROCESSED_ROOT": str(results_dir / "missing_source"),
        })
        return subprocess.run(
            ["/bin/bash", str(SCRIPT_PATH), "ldm"], cwd=SCRIPT_PATH.parents[2],
            env=env, text=True, capture_output=True,
        )

    def test_fresh_scratch_rejects_existing_directory_without_deleting_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir) / "experiment"
            scratch = results_dir / ".tmp_mini_train_dataset"
            scratch.mkdir(parents=True)
            marker = scratch / "keep.marker"
            marker.write_text("keep\n", encoding="utf-8")
            result = self._run_fresh_scratch(scratch, results_dir)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(marker.is_file())
            self.assertIn("fresh MINI_DATASET_DIR", result.stdout + result.stderr)

    def test_fresh_scratch_rejects_live_and_dangling_symlinks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            results_dir = temp_path / "experiment"
            results_dir.mkdir()
            target = temp_path / "owned"
            target.mkdir()
            marker = target / "keep.marker"
            marker.write_text("keep\n", encoding="utf-8")
            for name, link_target in (("live", target), ("dangling", temp_path / "missing")):
                with self.subTest(name=name):
                    scratch = results_dir / f".tmp_{name}"
                    scratch.symlink_to(link_target, target_is_directory=True)
                    result = self._run_fresh_scratch(scratch, results_dir)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertTrue(scratch.is_symlink())
                    self.assertIn("fresh MINI_DATASET_DIR", result.stdout + result.stderr)
            self.assertTrue(marker.is_file())

    def test_fresh_scratch_creates_new_path_inside_results_without_rm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir) / "experiment"
            results_dir.mkdir()
            scratch = results_dir / ".tmp_mini_train_dataset"
            result = self._run_fresh_scratch(scratch, results_dir)
            self.assertNotEqual(result.returncode, 0, msg="缺失源场景应在 scratch 创建后失败")
            self.assertTrue(scratch.is_dir(), msg=result.stdout + result.stderr)
            self.assertIn("missing radar_voxel/target_voxel", result.stdout + result.stderr)

    def _run_fresh_config(self, config_path, results_dir):
        scratch = results_dir / ".tmp_dataset"
        env = os.environ.copy()
        env.update({
            "MINI_REQUIRE_FRESH_CONFIG": "1",
            "MINI_CONFIG_PATH": str(config_path),
            "MINI_DATASET_DIR": str(scratch),
            "MINI_RESULTS_DIR": str(results_dir),
            "TRAIN_SCENES_OVERRIDE": "missing_scene",
            "PREPROCESSED_ROOT": str(results_dir / "missing_source"),
        })
        return subprocess.run(
            ["/bin/bash", str(SCRIPT_PATH), "ldm"], cwd=SCRIPT_PATH.parents[2],
            env=env, text=True, capture_output=True,
        )

    def test_fresh_config_rejects_existing_file_and_symlinks_without_modification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            results_dir = temp_path / "experiment"
            results_dir.mkdir()
            external = temp_path / "external.yaml"
            external.write_text("owner: external\n", encoding="utf-8")
            existing = results_dir / ".tmp_existing.yaml"
            existing.write_text("owner: existing\n", encoding="utf-8")
            live = results_dir / ".tmp_live.yaml"
            live.symlink_to(external)
            dangling = results_dir / ".tmp_dangling.yaml"
            dangling.symlink_to(temp_path / "missing.yaml")
            for config_path in (existing, live, dangling):
                with self.subTest(config_path=config_path.name):
                    result = self._run_fresh_config(config_path, results_dir)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("fresh MINI_CONFIG_PATH", result.stdout + result.stderr)
            self.assertEqual(existing.read_text(encoding="utf-8"), "owner: existing\n")
            self.assertEqual(external.read_text(encoding="utf-8"), "owner: external\n")
            self.assertTrue(dangling.is_symlink())


class LDMVerticalRunnerTest(unittest.TestCase):
    """静态检查通用 LDM 竖向实验 runner 的路径契约。"""

    @classmethod
    def setUpClass(cls):
        cls.script = LDM_VERTICAL_RUNNER.read_text(encoding="utf-8")

    def test_runner_normalizes_relative_output_and_vae_paths_to_repo_root(self):
        for fragment in (
            'EXP_DIR="$(realpath -m -- "${EXP_DIR_INPUT}")"',
            'if [[ -n "${BASE_VAE_CKPT}" && "${BASE_VAE_CKPT}" != /* ]]; then',
            'BASE_VAE_CKPT="${ROOT_DIR}/${BASE_VAE_CKPT}"',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.script)

    def test_runner_has_output_allowlist_independent_of_overwrite(self):
        for fragment in (
            'RESULT_ROOT="$(realpath -m -- "${ROOT_DIR}/test/result")"',
            '"${RESULT_ROOT}"/* | /tmp/*',
            'Error: unsafe EXP_DIR',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.script)


class LDMVerticalRunnerSafetyBehaviorTest(unittest.TestCase):
    """验证 generic runner 的输出 allowlist 先于覆盖和训练生效。"""

    script = LDM_VERTICAL_RUNNER.read_text(encoding="utf-8")

    def test_train_only_exit_precedes_inference_eval_and_visualization(self):
        train_only = self.script.index('if [[ "${LDM_TRAIN_ONLY}" == "1" ]]')
        for downstream in (
            'bash "${SELF_DIR}/inference_minimal.sh" ldm',
            '"${ROOT_DIR}/test/evaluation/ldm/evaluate_ldm_vertical_structure.py"',
            '"${ROOT_DIR}/test/visualization/generate_interactive_inference_compare.py"',
        ):
            self.assertLess(train_only, self.script.index(downstream))

    def test_allow_overwrite_cannot_bypass_output_allowlist_or_use_roots(self):
        root_dir = LDM_VERTICAL_RUNNER.parents[2]
        dangerous_paths = (
            Path("/"), Path("/tmp"), root_dir / "test/result",
            root_dir, Path("/var/tmp/task3-generic-outside"),
        )
        for path in dangerous_paths:
            with self.subTest(path=path):
                env = os.environ.copy()
                env.update({"EXP_DIR": str(path), "ALLOW_OVERWRITE": "1"})
                result = subprocess.run(
                    ["/bin/bash", str(LDM_VERTICAL_RUNNER)], cwd=root_dir,
                    env=env, text=True, capture_output=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("unsafe EXP_DIR", result.stdout + result.stderr)

    def test_external_scratch_and_config_are_rejected_without_modifying_content(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            exp_dir = temp_path / "experiment"
            external_dataset = temp_path / "user-data"
            external_dataset.mkdir()
            dataset_marker = external_dataset / "keep.marker"
            dataset_marker.write_text("keep\n", encoding="utf-8")
            external_config = temp_path / "user-config.yaml"
            external_config.write_text("owner: user\n", encoding="utf-8")
            for variable, value in (
                ("MINI_DATASET_DIR", external_dataset),
                ("MINI_CONFIG_PATH", external_config),
            ):
                with self.subTest(variable=variable):
                    env = os.environ.copy()
                    env.update({
                        "EXP_DIR": str(exp_dir), "ALLOW_OVERWRITE": "1",
                        variable: str(value),
                    })
                    result = subprocess.run(
                        ["/bin/bash", str(LDM_VERTICAL_RUNNER)],
                        cwd=LDM_VERTICAL_RUNNER.parents[2], env=env,
                        text=True, capture_output=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(f"unsafe {variable}", result.stdout + result.stderr)
            self.assertEqual(dataset_marker.read_text(encoding="utf-8"), "keep\n")
            self.assertEqual(external_config.read_text(encoding="utf-8"), "owner: user\n")

    def test_legal_relative_exp_scratch_paths_pass_path_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            exp_dir = Path(temp_dir) / "experiment"
            exp_dir.mkdir()
            (exp_dir / "occupied.marker").write_text("keep\n", encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "EXP_DIR": str(exp_dir),
                "MINI_DATASET_DIR": ".tmp_dataset",
                "MINI_CONFIG_PATH": ".tmp_config.yaml",
                "BASE_VAE_CKPT": str(Path(temp_dir) / "missing-vae.pt"),
            })
            result = subprocess.run(
                ["/bin/bash", str(LDM_VERTICAL_RUNNER)],
                cwd=LDM_VERTICAL_RUNNER.parents[2], env=env,
                text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("unsafe MINI_", result.stdout + result.stderr)
            self.assertIn("already exists and is not empty", result.stdout + result.stderr)

    def test_runner_uses_absolute_helper_script_paths(self):
        for path in (
            '"${ROOT_DIR}/test/evaluation/ldm/evaluate_ldm_vertical_structure.py"',
            '"${ROOT_DIR}/test/visualization/generate_interactive_inference_compare.py"',
        ):
            with self.subTest(path=path):
                self.assertIn(path, self.script)

    def test_runner_logs_and_forwards_ir_frustum_weights(self):
        for fragment in (
            'IR frustum occupancy/top weights',
            'MINI_LDM_IR_FRUSTUM_OCC_WEIGHT="${MINI_LDM_IR_FRUSTUM_OCC_WEIGHT:-0.0}"',
            'MINI_LDM_IR_FRUSTUM_TOP_WEIGHT="${MINI_LDM_IR_FRUSTUM_TOP_WEIGHT:-0.0}"',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.script)

    def test_runner_logs_and_forwards_v8_balancing_weights(self):
        for fragment in (
            'MINI_LDM_TOP_OVERSHOOT_WEIGHT="${MINI_LDM_TOP_OVERSHOOT_WEIGHT:-0.0}"',
            'MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT="${MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT:-0.0}"',
            'top overshoot weight: ${MINI_LDM_TOP_OVERSHOOT_WEIGHT}',
            'IR frustum negative weight: ${MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT}',
            'MINI_LDM_TOP_OVERSHOOT_WEIGHT="${MINI_LDM_TOP_OVERSHOOT_WEIGHT}"',
            'MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT="${MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT}"',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.script)

    def test_runner_anchors_all_fallback_vae_candidates_to_repo_root(self):
        for relative_path in (
            "test/result/ldm/vertical_structure/ldm_near40_500_vertical_v1/vae/vae_best.pt",
            "test/mini-test/train_results_near40_loop3/vae/vae_best.pt",
            "test/mini-test/train_results_mini_calibrated/vae/vae_best.pt",
            "test/mini-test/train_results_mini/vae/vae_best.pt",
        ):
            with self.subTest(relative_path=relative_path):
                self.assertIn(f'"${{ROOT_DIR}}/{relative_path}"', self.script)

    def test_runner_forwards_experiment_scratch_and_config_paths(self):
        for variable in ("MINI_DATASET_DIR", "MINI_CONFIG_PATH"):
            with self.subTest(variable=variable):
                self.assertIn(f'{variable}="${{{variable}}}" \\', self.script)

    def test_runner_uses_ifs_read_for_numeric_arrays(self):
        for variable, array_name in (
            ("MINI_TARGET_SIZE", "TARGET_SIZE_ARGS"),
            ("MINI_SOURCE_PC_RANGE", "SOURCE_PC_RANGE_ARGS"),
            ("MINI_MODEL_PC_RANGE", "MODEL_PC_RANGE_ARGS"),
        ):
            with self.subTest(variable=variable):
                self.assertIn(
                    f"IFS=',' read -r -a {array_name} <<< \"${{{variable}}}\"",
                    self.script,
                )


class VAEZ64UpperBoundRunnerTest(unittest.TestCase):
    """静态检查 Z=64 VAE 上界 runner 的默认实验契约。"""

    @classmethod
    def setUpClass(cls):
        cls.script = Z64_UPPER_BOUND_RUNNER.read_text(encoding="utf-8")

    def test_runner_declares_expected_defaults(self):
        expected_defaults = {
            "EXP_DIR": "test/result/vae/reconstruction/vae_near40_500_z64_upper_bound",
            "MINI_TARGET_SIZE": "64,128,128",
            "MINI_MODEL_PC_RANGE": "0,-20,-6,40,20,10",
            "MINI_SOURCE_PC_RANGE": "0,-20,-6,120,20,10",
            "SAMPLES_PER_SCENE": "500",
            "MINI_VAE_EPOCHS": "10",
            "MINI_VAE_CONFIG_TYPE": "lightweight",
            "MINI_VAE_LATENT_DIM": "8",
            "MINI_VAE_OCC_LOSS": "bce_dice",
            "MINI_NUM_WORKERS": "2",
            "MAX_DIAG_FILES": "0",
        }
        for variable, default in expected_defaults.items():
            with self.subTest(variable=variable):
                self.assertIn(f'{variable}="${{{variable}:-{default}}}"', self.script)

    def test_runner_derives_absolute_experiment_dir_from_relative_or_absolute_exp_dir(self):
        expected_fragments = (
            'if [[ "${EXP_DIR}" = /* ]]; then',
            'EXP_DIR_ABS="${EXP_DIR}"',
            'EXP_DIR_ABS="${ROOT_DIR}/${EXP_DIR}"',
        )
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.script)

    def test_runner_default_diagnostic_inputs_follow_training_outputs(self):
        expected_defaults = (
            'TARGET_VOXEL_DIR="${TARGET_VOXEL_DIR:-${MINI_DATASET_DIR}/${SCENE}/target_voxel}"',
            'VAE_CKPT="${VAE_CKPT:-${MINI_RESULTS_DIR}/vae/vae_best.pt}"',
        )
        for default in expected_defaults:
            with self.subTest(default=default):
                self.assertIn(default, self.script)

    def test_runner_keeps_cuda_devices_and_visible_devices_consistent(self):
        expected_fragments = (
            'if [[ -n "${CUDA_DEVICES:-}" ]]; then',
            'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICES}}"',
            'elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then',
            'CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}"',
            'CUDA_DEVICES="0"',
            'CUDA_VISIBLE_DEVICES="0"',
        )
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.script)

    def test_runner_calls_training_then_diagnostic_with_same_grid(self):
        self.assertIn('bash "${TRAIN_SCRIPT}" vae', self.script)
        self.assertLess(
            self.script.index('bash "${TRAIN_SCRIPT}" vae'),
            self.script.index('"${DIAG_SCRIPT}"'),
        )
        for argument in (
            '--vae_ckpt "${VAE_CKPT}"',
            '--target_voxel_dir "${TARGET_VOXEL_DIR}"',
            '--output_dir "${DIAG_OUTPUT_DIR}"',
            '--max_files "${MAX_DIAG_FILES}"',
            '--target_size "${MINI_TARGET_SIZE}"',
            '--source_pc_range "${MINI_SOURCE_PC_RANGE}"',
            '--model_pc_range "${MINI_MODEL_PC_RANGE}"',
        ):
            with self.subTest(argument=argument):
                self.assertIn(argument, self.script)

    def test_runner_checks_checkpoint_before_diagnostic(self):
        checkpoint_check = 'if [[ ! -f "${VAE_CKPT}" ]]; then'
        diagnostic_call = '"${DIAG_SCRIPT}"'
        self.assertIn(checkpoint_check, self.script)
        self.assertLess(
            self.script.index(checkpoint_check),
            self.script.index(diagnostic_call),
        )

    def test_runner_exports_training_environment_overrides(self):
        for variable in (
            "CUDA_VISIBLE_DEVICES",
            "MINI_RESULTS_DIR",
            "MINI_DATASET_DIR",
            "MINI_CONFIG_PATH",
            "MINI_TARGET_SIZE",
            "MINI_SOURCE_PC_RANGE",
            "MINI_MODEL_PC_RANGE",
            "SAMPLES_PER_SCENE",
            "MINI_VAE_EPOCHS",
            "MINI_VAE_CONFIG_TYPE",
            "MINI_VAE_LATENT_DIM",
            "MINI_VAE_OCC_LOSS",
            "MINI_NUM_WORKERS",
            "SCENE",
        ):
            with self.subTest(variable=variable):
                self.assertRegex(
                    self.script,
                    rf"export\s+.*\b{variable}\b",
                    msg=f"{variable} 未导出给训练/诊断流程",
                )


class LDMZ64V5RunnerTest(unittest.TestCase):
    """静态检查 Z=64 LDM v5 runner 的默认实验契约。"""

    @classmethod
    def setUpClass(cls):
        cls.script = Z64_LDM_V5_RUNNER.read_text(encoding="utf-8")

    def test_runner_declares_z64_v5_defaults(self):
        expected_defaults = {
            "EXP_DIR": "test/result/ldm/ablation/ldm_near40_500_z64_v5_empty_column",
            "BASE_VAE_CKPT": "test/result/vae/reconstruction/vae_near40_500_z64_upper_bound/vae/vae_best.pt",
            "MINI_TARGET_SIZE": "64,128,128",
            "MINI_MODEL_PC_RANGE": "0,-20,-6,40,20,10",
            "MINI_SOURCE_PC_RANGE": "0,-20,-6,120,20,10",
            "SAMPLES_PER_SCENE": "500",
            "MINI_LDM_EPOCHS": "10",
            "MINI_LDM_DENSITY_WEIGHT": "0.03",
            "MINI_LDM_UNCERTAINTY_WEIGHT": "0.0",
            "OCC_THRESHOLD": "0.95",
            "ALLOW_OVERWRITE": "0",
        }
        for variable, default in expected_defaults.items():
            with self.subTest(variable=variable):
                self.assertIn(
                    f'export {variable}="${{{variable}:-{default}}}"',
                    self.script,
                )

    def test_runner_calls_generic_vertical_experiment_runner(self):
        self.assertIn('bash "${SELF_DIR}/run_ldm_vertical_experiment.sh"', self.script)

    def test_runner_syncs_cuda_visible_devices_with_cuda_devices(self):
        for fragment in (
            'if [[ -n "${CUDA_DEVICES:-}" ]]; then',
            'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICES}}"',
            'elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then',
            'export CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}"',
            'export CUDA_DEVICES="0"',
            'export CUDA_VISIBLE_DEVICES="0"',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.script)

    def test_runner_refuses_to_overwrite_existing_default_experiment(self):
        for fragment in (
            'EXP_DIR_ABS="${ROOT_DIR}/${EXP_DIR}"',
            'if [[ "${ALLOW_OVERWRITE}" != "1" && -e "${EXP_DIR_ABS}/ldm/ldm_best.pt" ]]; then',
            'Set EXP_DIR to a new directory, or ALLOW_OVERWRITE=1',
            'exit 1',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.script)

    def test_runner_sets_decoded_and_structure_weights(self):
        for variable in (
            "MINI_LDM_DECODED_WEIGHT",
            "MINI_LDM_DECODED_FP_WEIGHT",
            "MINI_LDM_DECODED_MASS_WEIGHT",
            "MINI_LDM_HEIGHT_WEIGHT",
            "MINI_LDM_CONTINUITY_WEIGHT",
            "MINI_LDM_DENSITY_WEIGHT",
        ):
            with self.subTest(variable=variable):
                self.assertIn(f"export {variable}=", self.script)


class LDMZ64V6RunnerTest(unittest.TestCase):
    """静态检查 Z=64 LDM v6 顶部恢复 runner 的默认实验契约。"""

    @classmethod
    def setUpClass(cls):
        cls.script = Z64_LDM_V6_RUNNER.read_text(encoding="utf-8")

    def test_runner_declares_z64_v6_defaults(self):
        expected_defaults = {
            "EXP_DIR": "test/result/ldm/ablation/ldm_near40_500_z64_v6_top",
            "BASE_VAE_CKPT": "test/result/vae/reconstruction/vae_near40_500_z64_upper_bound/vae/vae_best.pt",
            "MINI_TARGET_SIZE": "64,128,128",
            "MINI_MODEL_PC_RANGE": "0,-20,-6,40,20,10",
            "MINI_SOURCE_PC_RANGE": "0,-20,-6,120,20,10",
            "SAMPLES_PER_SCENE": "500",
            "MINI_LDM_EPOCHS": "10",
            "MINI_LDM_TOP_WEIGHT": "0.08",
            "MINI_LDM_DENSITY_WEIGHT": "0.015",
            "MINI_LDM_UNCERTAINTY_WEIGHT": "0.0",
            "OCC_THRESHOLD": "0.85",
            "ALLOW_OVERWRITE": "0",
        }
        for variable, default in expected_defaults.items():
            with self.subTest(variable=variable):
                self.assertIn(
                    f'export {variable}="${{{variable}:-{default}}}"',
                    self.script,
                )

    def test_runner_calls_generic_vertical_experiment_runner(self):
        self.assertIn('bash "${SELF_DIR}/run_ldm_vertical_experiment.sh"', self.script)

    def test_runner_refuses_to_overwrite_existing_default_experiment(self):
        for fragment in (
            'EXP_DIR_ABS="${ROOT_DIR}/${EXP_DIR}"',
            'if [[ "${ALLOW_OVERWRITE}" != "1" && -e "${EXP_DIR_ABS}/ldm/ldm_best.pt" ]]; then',
            'Set EXP_DIR to a new directory, or ALLOW_OVERWRITE=1',
            'exit 1',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.script)

    def test_runner_sets_decoded_and_top_structure_weights(self):
        for variable in (
            "MINI_LDM_DECODED_WEIGHT",
            "MINI_LDM_DECODED_FP_WEIGHT",
            "MINI_LDM_DECODED_MASS_WEIGHT",
            "MINI_LDM_HEIGHT_WEIGHT",
            "MINI_LDM_TOP_WEIGHT",
            "MINI_LDM_CONTINUITY_WEIGHT",
            "MINI_LDM_DENSITY_WEIGHT",
        ):
            with self.subTest(variable=variable):
                self.assertIn(f"export {variable}=", self.script)


class LDMZ64V7RunnerTest(unittest.TestCase):
    """静态检查 Z=64 LDM v7 红外视锥实验 runner 的默认实验契约。"""

    @classmethod
    def setUpClass(cls):
        cls.script = Z64_LDM_V7_RUNNER.read_text(encoding="utf-8")

    def test_runner_declares_z64_v7_defaults(self):
        expected_defaults = {
            "EXP_DIR": "test/result/ldm/ablation/ldm_near40_500_z64_v7_ir",
            "BASE_VAE_CKPT": "test/result/vae/reconstruction/vae_near40_500_z64_upper_bound/vae/vae_best.pt",
            "MINI_TARGET_SIZE": "64,128,128",
            "MINI_MODEL_PC_RANGE": "0,-20,-6,40,20,10",
            "MINI_SOURCE_PC_RANGE": "0,-20,-6,120,20,10",
            "SAMPLES_PER_SCENE": "500",
            "MINI_LDM_EPOCHS": "10",
            "MINI_LDM_IR_FRUSTUM_OCC_WEIGHT": "0.04",
            "MINI_LDM_IR_FRUSTUM_TOP_WEIGHT": "0.05",
            "MINI_LDM_UNCERTAINTY_WEIGHT": "0.0",
            "OCC_THRESHOLD": "0.85",
            "ALLOW_OVERWRITE": "0",
        }
        for variable, default in expected_defaults.items():
            with self.subTest(variable=variable):
                self.assertIn(
                    f'export {variable}="${{{variable}:-{default}}}"',
                    self.script,
                )

    def test_runner_calls_generic_vertical_experiment_runner(self):
        self.assertIn('bash "${SELF_DIR}/run_ldm_vertical_experiment.sh"', self.script)

    def test_runner_sets_ir_frustum_weights(self):
        for variable in (
            "MINI_LDM_IR_FRUSTUM_OCC_WEIGHT",
            "MINI_LDM_IR_FRUSTUM_TOP_WEIGHT",
        ):
            with self.subTest(variable=variable):
                self.assertIn(f"export {variable}=", self.script)


class LDMZ64V8RunnerTest(unittest.TestCase):
    """静态检查 Z=64 LDM v8 平衡监督 runner 的受保护实验契约。"""

    @classmethod
    def setUpClass(cls):
        cls.script = Z64_LDM_V8_RUNNER.read_text(encoding="utf-8")

    def test_runner_declares_z64_v8_defaults(self):
        expected_defaults = {
            "EXP_DIR": "test/result/ldm/ablation/ldm_near40_500_z64_v8_balanced",
            "BASE_VAE_CKPT": "test/result/vae/reconstruction/vae_near40_500_z64_upper_bound/vae/vae_best.pt",
            "MINI_TARGET_SIZE": "64,128,128",
            "MINI_MODEL_PC_RANGE": "0,-20,-6,40,20,10",
            "MINI_SOURCE_PC_RANGE": "0,-20,-6,120,20,10",
            "SAMPLES_PER_SCENE": "500",
            "MINI_LDM_EPOCHS": "10",
            "MINI_NUM_WORKERS": "2",
            "MINI_LDM_DECODED_WEIGHT": "0.12",
            "MINI_LDM_DECODED_FP_WEIGHT": "0.20",
            "MINI_LDM_DECODED_MASS_WEIGHT": "0.08",
            "MINI_LDM_HEIGHT_WEIGHT": "0.04",
            "MINI_LDM_TOP_WEIGHT": "0.08",
            "MINI_LDM_TOP_OVERSHOOT_WEIGHT": "0.05",
            "MINI_LDM_CONTINUITY_WEIGHT": "0.02",
            "MINI_LDM_DENSITY_WEIGHT": "0.015",
            "MINI_LDM_IR_FRUSTUM_OCC_WEIGHT": "0.02",
            "MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT": "0.02",
            "MINI_LDM_IR_FRUSTUM_TOP_WEIGHT": "0.03",
            "MINI_LDM_UNCERTAINTY_WEIGHT": "0.0",
            "OCC_THRESHOLD": "0.99",
            "TARGET_THRESHOLD": "0.5",
            "ALLOW_OVERWRITE": "0",
        }
        for variable, default in expected_defaults.items():
            with self.subTest(variable=variable):
                self.assertIn(
                    f'export {variable}="${{{variable}:-{default}}}"',
                    self.script,
                )

    def test_runner_uses_one_selected_cuda_value_and_nonempty_dir_guard(self):
        for fragment in (
            'if [[ -n "${CUDA_DEVICES:-}" ]]; then',
            'SELECTED_CUDA_DEVICES="${CUDA_DEVICES}"',
            'elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then',
            'SELECTED_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES}"',
            'SELECTED_CUDA_DEVICES="0"',
            'export CUDA_DEVICES="${SELECTED_CUDA_DEVICES}"',
            'export CUDA_VISIBLE_DEVICES="${SELECTED_CUDA_DEVICES}"',
            'EXP_DIR_ABS="$(realpath -m -- "${EXP_DIR_INPUT}")"',
            'find "${EXP_DIR_ABS}" -mindepth 1 -print -quit',
            'Set EXP_DIR to a new directory, or ALLOW_OVERWRITE=1',
            'exit 1',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.script)

    def test_runner_declares_experiment_local_scratch_and_config(self):
        self.assertIn(
            'export MINI_DATASET_DIR="${MINI_DATASET_DIR:-${EXP_DIR_ABS}/.tmp_mini_train_dataset}"',
            self.script,
        )
        self.assertIn(
            'export MINI_CONFIG_PATH="${MINI_CONFIG_PATH:-${EXP_DIR_ABS}/.tmp_ldm_config.yaml}"',
            self.script,
        )

    def test_runner_only_calls_generic_vertical_experiment(self):
        self.assertIn('bash "${SELF_DIR}/run_ldm_vertical_experiment.sh"', self.script)
        self.assertNotIn("target_ablation", self.script)
        self.assertNotRegex(self.script, r"train_minimal\.sh\s+cd\b")


class LDMZ64V8BehaviorTest(unittest.TestCase):
    """用假 bash 验证 v8 早停和 CUDA 环境，不启动训练。"""

    @staticmethod
    def _fake_bash(bin_dir):
        fake_bash = Path(bin_dir) / "bash"
        fake_bash.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n%s\\n%s\\n%s\\n%s\\n' \"$CUDA_DEVICES\" "
            "\"$CUDA_VISIBLE_DEVICES\" \"$MINI_DATASET_DIR\" "
            "\"$MINI_CONFIG_PATH\" \"$*\" > \"$CAPTURE_PATH\"\n",
            encoding="utf-8",
        )
        fake_bash.chmod(0o755)

    def _run(self, exp_dir, capture_path, **overrides):
        bin_dir = Path(capture_path).parent / "bin"
        bin_dir.mkdir()
        self._fake_bash(bin_dir)
        env = os.environ.copy()
        env.update(
            {
                "EXP_DIR": str(exp_dir),
                "CAPTURE_PATH": str(capture_path),
                "PATH": f"{bin_dir}:{env['PATH']}",
            }
        )
        env.update(overrides)
        return subprocess.run(
            ["/bin/bash", str(Z64_LDM_V8_RUNNER)],
            cwd=Path(capture_path).parent,
            env=env,
            text=True,
            capture_output=True,
        )

    @staticmethod
    def _generic_dispatch_env(temp_path, exp_dir):
        bin_dir = temp_path / "dispatch_bin"
        bin_dir.mkdir()
        fake_bash = bin_dir / "bash"
        fake_bash.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  *run_ldm_vertical_experiment.sh)\n"
            "    printf 'generic\\n' >> \"$GENERIC_CAPTURE_PATH\"\n"
            "    exec /bin/bash \"$@\"\n"
            "    ;;\n"
            "  *)\n"
            "    printf 'downstream\\n' >> \"$DOWNSTREAM_CAPTURE_PATH\"\n"
            "    sleep 0.5\n"
            "    exit 97\n"
            "    ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        fake_bash.chmod(0o755)
        base_vae = temp_path / "base_vae.pt"
        base_vae.write_text("checkpoint sentinel\n", encoding="utf-8")
        env = os.environ.copy()
        env.update(
            {
                "EXP_DIR": str(exp_dir),
                "ALLOW_OVERWRITE": "1",
                "BASE_VAE_CKPT": str(base_vae),
                "GENERIC_CAPTURE_PATH": str(temp_path / "generic_calls.txt"),
                "DOWNSTREAM_CAPTURE_PATH": str(temp_path / "downstream_calls.txt"),
                "PATH": f"{bin_dir}:{env['PATH']}",
            }
        )
        return env

    def test_nonempty_exp_dir_with_only_metrics_refuses_before_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            exp_dir = temp_path / "experiment"
            exp_dir.mkdir()
            (exp_dir / "metrics.csv").write_text("metric,value\n", encoding="utf-8")
            capture_path = temp_path / "called.txt"

            result = self._run(exp_dir, capture_path, ALLOW_OVERWRITE="0")

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(capture_path.exists(), msg="非空实验目录仍调用了训练 runner")
            self.assertIn("already exists and is not empty", result.stdout + result.stderr)

    def test_nonempty_exp_dir_with_only_logs_metrics_refuses_before_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            exp_dir = temp_path / "experiment"
            metrics_dir = exp_dir / "logs" / "metrics"
            metrics_dir.mkdir(parents=True)
            (metrics_dir / "epoch.json").write_text("{}\n", encoding="utf-8")
            capture_path = temp_path / "called.txt"

            result = self._run(exp_dir, capture_path, ALLOW_OVERWRITE="0")

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(capture_path.exists(), msg="日志目录非空时仍调用了训练 runner")
            self.assertIn("already exists and is not empty", result.stdout + result.stderr)

    def test_explicit_cuda_devices_wins_and_is_exported_consistently(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            capture_path = temp_path / "called.txt"

            result = self._run(
                temp_path / "new_experiment",
                capture_path,
                CUDA_DEVICES="2,3",
                CUDA_VISIBLE_DEVICES="7",
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            captured = capture_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(captured[:2], ["2,3", "2,3"])
            self.assertIn("run_ldm_vertical_experiment.sh", captured[4])

    def test_cuda_visible_devices_is_used_when_cuda_devices_is_unset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            capture_path = temp_path / "called.txt"

            result = self._run(
                temp_path / "new_experiment",
                capture_path,
                CUDA_DEVICES="",
                CUDA_VISIBLE_DEVICES="4,5",
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            captured = capture_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(captured[:2], ["4,5", "4,5"])

    def test_runner_passes_experiment_local_scratch_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            exp_dir = temp_path / "new_experiment"
            capture_path = temp_path / "called.txt"

            result = self._run(exp_dir, capture_path)

            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            captured = capture_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                captured[2], str(exp_dir / ".tmp_mini_train_dataset")
            )
            self.assertEqual(captured[3], str(exp_dir / ".tmp_ldm_config.yaml"))

    def test_symlink_exp_dir_is_rejected_before_generic_runner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            target_dir = temp_path / "existing_results"
            target_dir.mkdir()
            (target_dir / "metrics.csv").write_text("metric,value\n", encoding="utf-8")
            exp_link = temp_path / "experiment_link"
            exp_link.symlink_to(target_dir, target_is_directory=True)
            capture_path = temp_path / "called.txt"

            result = self._run(exp_link, capture_path, ALLOW_OVERWRITE="1")

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(capture_path.exists(), msg="symlink EXP_DIR 仍调用了 generic runner")
            self.assertIn("must not be a symlink", result.stdout + result.stderr)

    def test_preexisting_lock_rejects_even_when_overwrite_is_allowed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            exp_dir = temp_path / "experiment"
            Path(f"{exp_dir}.lock").mkdir()
            env = self._generic_dispatch_env(temp_path, exp_dir)

            result = subprocess.run(
                ["/bin/bash", str(Z64_LDM_V8_RUNNER)],
                cwd=temp_path,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((temp_path / "downstream_calls.txt").exists())
            self.assertIn("already running", result.stdout + result.stderr)

    def test_wrapper_calls_generic_once_without_lock_handoff(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            exp_dir = temp_path / "new_experiment"
            env = self._generic_dispatch_env(temp_path, exp_dir)

            result = subprocess.run(
                ["/bin/bash", str(Z64_LDM_V8_RUNNER)],
                cwd=temp_path,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 97, msg=result.stdout + result.stderr)
            self.assertEqual(
                (temp_path / "generic_calls.txt").read_text().splitlines(), ["generic"]
            )
            self.assertNotIn("LDM_RUN_LOCK_HELD", Z64_LDM_V8_RUNNER.read_text())
            self.assertNotIn("LDM_RUN_LOCK_PATH", Z64_LDM_V8_RUNNER.read_text())
            self.assertFalse(Path(f"{exp_dir}.lock").exists())

    def test_concurrent_wrappers_allow_only_one_generic_runner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            exp_dir = temp_path / "experiment"
            env = self._generic_dispatch_env(temp_path, exp_dir)
            downstream_capture = temp_path / "downstream_calls.txt"

            first = subprocess.Popen(
                ["/bin/bash", str(Z64_LDM_V8_RUNNER)],
                cwd=temp_path,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for _ in range(100):
                if downstream_capture.exists():
                    break
                time.sleep(0.01)
            second = subprocess.run(
                ["/bin/bash", str(Z64_LDM_V8_RUNNER)],
                cwd=temp_path,
                env=env,
                text=True,
                capture_output=True,
            )
            first_stdout, first_stderr = first.communicate(timeout=5)

            self.assertEqual(first.returncode, 97, msg=first_stdout + first_stderr)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(downstream_capture.read_text().splitlines(), ["downstream"])
            self.assertIn("already running", second.stdout + second.stderr)


class LDMVerticalRunnerBehaviorTest(unittest.TestCase):
    """验证 generic runner 在任何实验写入和训练前执行统一保护。"""

    @staticmethod
    def _run(exp_dir, **overrides):
        sentinel_dir = Path(exp_dir).parent / ".task3_sentinel_bin"
        sentinel_dir.mkdir(exist_ok=True)
        for command in ("bash", "python"):
            command_path = sentinel_dir / command
            command_path.write_text(
                "#!/bin/sh\necho unexpected downstream command: \"$*\" >&2\nexit 97\n",
                encoding="utf-8",
            )
            command_path.chmod(0o755)
        env = os.environ.copy()
        env.update(
            {
                "EXP_DIR": str(exp_dir),
                "ALLOW_OVERWRITE": "0",
                "PATH": f"{sentinel_dir}:{env['PATH']}",
                "PYTHON_BIN": str(sentinel_dir / "python"),
            }
        )
        env.update(overrides)
        return subprocess.run(
            ["/bin/bash", str(LDM_VERTICAL_RUNNER)],
            cwd=LDM_VERTICAL_RUNNER.parents[2],
            env=env,
            text=True,
            capture_output=True,
        )

    def test_direct_generic_refuses_nonempty_exp_dir_before_writing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            exp_dir = Path(temp_dir) / "experiment"
            exp_dir.mkdir()
            marker = exp_dir / "metrics.csv"
            marker.write_text("metric,value\n", encoding="utf-8")

            result = self._run(exp_dir)

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(marker.read_text(encoding="utf-8"), "metric,value\n")
            self.assertFalse((exp_dir / "vae").exists())
            self.assertIn("already exists and is not empty", result.stdout + result.stderr)

    def test_direct_generic_rejects_preexisting_lock_with_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            exp_dir = Path(temp_dir) / "experiment"
            lock_dir = Path(f"{exp_dir}.lock")
            lock_dir.mkdir()

            result = self._run(exp_dir, ALLOW_OVERWRITE="1")

            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(lock_dir.is_dir())
            self.assertFalse(exp_dir.exists())
            self.assertIn("already running", result.stdout + result.stderr)

    def test_forged_legacy_handoff_cannot_bypass_preexisting_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            exp_dir = Path(temp_dir) / "experiment"
            lock_dir = Path(f"{exp_dir}.lock")
            lock_dir.mkdir()

            result = self._run(
                exp_dir,
                ALLOW_OVERWRITE="1",
                LDM_RUN_LOCK_HELD="1",
                LDM_RUN_LOCK_PATH=str(lock_dir),
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(lock_dir.is_dir())
            self.assertFalse(exp_dir.exists())
            self.assertIn("already running", result.stdout + result.stderr)

    def test_generic_cleans_its_lock_after_downstream_exit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            exp_dir = temp_path / "experiment"
            base_vae = temp_path / "vae_best.pt"
            base_vae.write_text("checkpoint sentinel\n", encoding="utf-8")

            result = self._run(
                exp_dir,
                ALLOW_OVERWRITE="1",
                BASE_VAE_CKPT=str(base_vae),
            )

            self.assertEqual(result.returncode, 97)
            self.assertFalse(Path(f"{exp_dir}.lock").exists())
            self.assertIn("unexpected downstream command", result.stderr)


class LDMZ64V9ScreenTest(unittest.TestCase):
    """用假 bash/cp 验证 v9 screen 协议，不启动训练或推理。"""

    @staticmethod
    def _install_fakes(temp_path):
        bin_dir = temp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "cp").write_text(
            "#!/bin/sh\n"
            "printf 'cp|%s\\n' \"$*\" >> \"$CALL_LOG\"\n"
            "[ \"${FAKE_NO_WRITE:-0}\" = 1 ] && exit 0\n"
            "exec /bin/cp \"$@\"\n",
            encoding="utf-8",
        )
        (bin_dir / "bash").write_text(
            "#!/bin/sh\n"
            "printf 'bash|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\\n' \"$*\" "
            "\"$MINI_LDM_TOP_OVERSHOOT_WEIGHT\" \"$MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT\" "
            "\"$MINI_LDM_DECODED_WEIGHT\" \"$MINI_LDM_DECODED_FP_WEIGHT\" "
            "\"$MINI_LDM_DECODED_MASS_WEIGHT\" \"$MINI_LDM_HEIGHT_WEIGHT\" "
            "\"$MINI_LDM_TOP_WEIGHT\" \"$MINI_LDM_CONTINUITY_WEIGHT\" "
            "\"$MINI_LDM_DENSITY_WEIGHT\" \"$MINI_LDM_IR_FRUSTUM_OCC_WEIGHT\" "
            "\"$MINI_LDM_IR_FRUSTUM_TOP_WEIGHT\" \"$MINI_LDM_UNCERTAINTY_WEIGHT\" "
            "\"$OUTPUT_DIR\" \"$ABLATION_MAX_SAMPLES\" \"$ABLATION_STEPS\" "
            "\"$OCC_THRESHOLD\" \"$SAMPLES_PER_SCENE\" \"$MINI_LDM_EPOCHS\" "
            "\"$MINI_TARGET_SIZE\" \"$MINI_SOURCE_PC_RANGE\" \"$MINI_MODEL_PC_RANGE\" "
            "\"$MINI_SPLIT_SEED\" \"$MINI_DATASET_DIR\" \"$MINI_CONFIG_PATH\" "
            ">> \"$CALL_LOG\"\n"
            "[ \"${FAKE_NO_WRITE:-0}\" = 1 ] && exit 0\n"
            "case \"$1\" in\n"
            "  *train_minimal.sh) mkdir -p \"$EXP_DIR/ldm\"; : > \"$EXP_DIR/ldm/ldm_best.pt\" ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        (bin_dir / "mkdir").write_text(
            "#!/bin/sh\n"
            "/bin/mkdir \"$@\" || exit $?\n"
            "case \"$*\" in\n"
            "  *\"$EXP_DIR.lock\"*)\n"
            "    if [ -n \"${RACE_RELATIVE_PATH:-}\" ]; then\n"
            "      /bin/mkdir -p \"$EXP_DIR\" \"$RACE_OUTSIDE\"\n"
            "      ln -s \"$RACE_OUTSIDE\" \"$EXP_DIR/$RACE_RELATIVE_PATH\"\n"
            "    fi\n"
            "    ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        for command in ("bash", "cp", "mkdir"):
            (bin_dir / command).chmod(0o755)
        return bin_dir

    def _run(self, variant="A", **overrides):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        temp_path = Path(temp_dir.name)
        bin_dir = self._install_fakes(temp_path)
        base_vae = temp_path / "base" / "vae_best.pt"
        base_vae.parent.mkdir()
        base_vae.write_text("vae\n", encoding="utf-8")
        exp_dir = temp_path / "experiment"
        call_log = temp_path / "calls.log"
        env = os.environ.copy()
        env.update(
            {
                "V9_VARIANT": variant,
                "EXP_DIR": str(exp_dir),
                "BASE_VAE_CKPT": str(base_vae),
                "CALL_LOG": str(call_log),
                "PATH": f"{bin_dir}:{env['PATH']}",
            }
        )
        env.update(overrides)
        result = subprocess.run(
            ["/bin/bash", str(Z64_LDM_V9_RUNNER)],
            cwd=temp_path,
            env=env,
            text=True,
            capture_output=True,
        )
        calls = call_log.read_text(encoding="utf-8").splitlines() if call_log.exists() else []
        return result, exp_dir, calls

    def _run_with_defaults(self, variant):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        temp_path = Path(temp_dir.name)
        mini_dir = temp_path / "test" / "mini-test"
        mini_dir.mkdir(parents=True)
        runner = mini_dir / Z64_LDM_V9_RUNNER.name
        runner.write_text(Z64_LDM_V9_RUNNER.read_text(encoding="utf-8"), encoding="utf-8")
        runner.chmod(0o755)
        bin_dir = self._install_fakes(temp_path)
        base_vae = temp_path / "test/result/vae/reconstruction/vae_near40_500_z64_upper_bound/vae/vae_best.pt"
        base_vae.parent.mkdir(parents=True)
        base_vae.write_text("default vae\n", encoding="utf-8")
        call_log = temp_path / "calls.log"
        env = os.environ.copy()
        env.pop("EXP_DIR", None)
        env.pop("BASE_VAE_CKPT", None)
        env.update(
            {
                "V9_VARIANT": variant,
                "CALL_LOG": str(call_log),
                "PATH": f"{bin_dir}:{env['PATH']}",
            }
        )
        result = subprocess.run(
            ["/bin/bash", str(runner)], cwd=temp_path, env=env, text=True, capture_output=True
        )
        return result, temp_path, call_log.read_text(encoding="utf-8").splitlines()

    def test_variants_change_only_the_two_required_weights(self):
        result_a, _, calls_a = self._run("A")
        result_b, _, calls_b = self._run("B")
        self.assertEqual(result_a.returncode, 0, msg=result_a.stdout + result_a.stderr)
        self.assertEqual(result_b.returncode, 0, msg=result_b.stdout + result_b.stderr)
        train_a = next(line for line in calls_a if "train_minimal.sh ldm" in line).split("|")
        train_b = next(line for line in calls_b if "train_minimal.sh ldm" in line).split("|")
        self.assertEqual(train_a[2:4], ["0.02", "0.02"])
        self.assertEqual(train_b[2:4], ["0.05", "0.01"])
        self.assertEqual(train_a[4:14], train_b[4:14])
        self.assertEqual(
            train_a[4:14],
            ["0.12", "0.20", "0.08", "0.04", "0.08", "0.02", "0.015", "0.02", "0.03", "0.0"],
        )

    def test_copy_train_ablation_order_and_ablation_parameters(self):
        result, exp_dir, calls = self._run("A")
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertEqual(len(calls), 3, msg="v9 只能复制 VAE、训练 LDM、运行 32 帧消融")
        self.assertTrue(calls[0].startswith("cp|-a "))
        self.assertIn("train_minimal.sh ldm", calls[1])
        self.assertIn("run_ldm_z64_v7_target_ablation.sh", calls[2])
        self.assertIn(str(exp_dir / "ir_target_ablation_32_thr099"), calls[2])
        for fragment in ("32", "20", "0.99"):
            self.assertIn(fragment, calls[2])
        self.assertTrue((exp_dir / "vae" / "vae_best.pt").is_file())

    def test_training_receives_v8_shared_protocol_defaults(self):
        result, exp_dir, calls = self._run("A")
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        train = next(line for line in calls if "train_minimal.sh ldm" in line).split("|")
        self.assertEqual(
            train[18:24],
            ["500", "3", "64,128,128", "0,-20,-6,120,20,10", "0,-20,-6,40,20,10", "42"],
        )
        self.assertEqual(train[24], str(exp_dir / ".tmp_mini_train_dataset"))
        self.assertEqual(train[25], str(exp_dir / ".tmp_ldm_config.yaml"))

    def test_default_variants_use_independent_dirs_and_default_base_vae(self):
        result_a, root_a, calls_a = self._run_with_defaults("A")
        result_b, root_b, calls_b = self._run_with_defaults("B")
        self.assertEqual(result_a.returncode, 0, msg=result_a.stdout + result_a.stderr)
        self.assertEqual(result_b.returncode, 0, msg=result_b.stdout + result_b.stderr)
        self.assertIn(
            str(root_a / "test/result/ldm/ablation/ldm_near40_500_z64_v9a_top_screen/vae/vae_best.pt"),
            calls_a[0],
        )
        self.assertIn(
            str(root_b / "test/result/ldm/ablation/ldm_near40_500_z64_v9b_irneg_screen/vae/vae_best.pt"),
            calls_b[0],
        )
        self.assertIn(
            str(root_a / "test/result/vae/reconstruction/vae_near40_500_z64_upper_bound/vae/vae_best.pt"),
            calls_a[0],
        )

    def test_dangerous_exp_roots_are_rejected_even_with_overwrite(self):
        root_dir = Z64_LDM_V9_RUNNER.parents[2]
        for dangerous in (Path("/"), root_dir, root_dir / "test", root_dir / "test/result", Path("/tmp")):
            with self.subTest(dangerous=dangerous):
                result, _, calls = self._run(
                    "A", EXP_DIR=str(dangerous), ALLOW_OVERWRITE="1", FAKE_NO_WRITE="1"
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(calls)
                self.assertIn("unsafe EXP_DIR", result.stdout + result.stderr)

    def test_exp_outside_result_or_tmp_is_rejected(self):
        result, _, calls = self._run(
            "A", EXP_DIR="/var/tmp/v9-outside", ALLOW_OVERWRITE="1", FAKE_NO_WRITE="1"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(calls)
        self.assertIn("unsafe EXP_DIR", result.stdout + result.stderr)

    def test_scratch_and_config_must_be_restricted_children_of_exp(self):
        invalid_overrides = (
            {"MINI_DATASET_DIR": "/tmp/outside/.tmp_dataset"},
            {"MINI_DATASET_DIR": "dataset"},
            {"MINI_CONFIG_PATH": "/tmp/outside/.tmp_config.yaml"},
            {"MINI_CONFIG_PATH": "config.yaml"},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                result, _, calls = self._run("A", FAKE_NO_WRITE="1", **overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(calls)
                self.assertIn("unsafe MINI_", result.stdout + result.stderr)

    def test_existing_fixed_write_path_symlinks_are_rejected_before_training(self):
        fixed_paths = (
            ("vae", None),
            ("ldm", None),
            ("ir_target_ablation_32_thr099", None),
            (".tmp_mini_train_dataset", "MINI_DATASET_DIR"),
            (".tmp_ldm_config.yaml", "MINI_CONFIG_PATH"),
        )
        for relative_path, variable in fixed_paths:
            with self.subTest(relative_path=relative_path):
                temp_dir = tempfile.TemporaryDirectory()
                self.addCleanup(temp_dir.cleanup)
                temp_path = Path(temp_dir.name)
                exp_dir = temp_path / "experiment"
                exp_dir.mkdir()
                outside = temp_path / "outside" / relative_path
                outside.parent.mkdir(parents=True, exist_ok=True)
                outside.mkdir() if variable != "MINI_CONFIG_PATH" else outside.touch()
                link = exp_dir / relative_path
                link.symlink_to(outside, target_is_directory=outside.is_dir())
                overrides = {
                    "EXP_DIR": str(exp_dir),
                    "ALLOW_OVERWRITE": "1",
                    "FAKE_NO_WRITE": "1",
                }
                if variable:
                    overrides[variable] = str(link)

                result, _, calls = self._run("A", **overrides)

                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(calls, msg=f"symlink {relative_path} 仍调用了 cp/训练")
                self.assertIn("symlink", result.stdout + result.stderr)

    def test_existing_write_parent_must_canonicalize_inside_exp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        temp_path = Path(temp_dir.name)
        exp_dir = temp_path / "experiment"
        exp_dir.mkdir()
        outside = temp_path / "outside"
        outside.mkdir()
        (exp_dir / ".tmp_parent").symlink_to(outside, target_is_directory=True)

        result, _, calls = self._run(
            "A",
            EXP_DIR=str(exp_dir),
            ALLOW_OVERWRITE="1",
            MINI_CONFIG_PATH=str(exp_dir / ".tmp_parent/.tmp_config.yaml"),
            FAKE_NO_WRITE="1",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(calls)
        self.assertIn("unsafe MINI_CONFIG_PATH", result.stdout + result.stderr)

    def test_path_replaced_after_initial_check_is_rejected_before_first_copy(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        outside = Path(temp_dir.name) / "race_outside"

        result, _, calls = self._run(
            "A",
            ALLOW_OVERWRITE="1",
            RACE_RELATIVE_PATH="vae",
            RACE_OUTSIDE=str(outside),
            FAKE_NO_WRITE="1",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(calls, msg="持锁后被替换的 VAE 路径仍进入了 cp/训练")
        self.assertIn("symlink", result.stdout + result.stderr)

    def test_invalid_variant_nonempty_dir_symlink_and_active_lock_are_rejected(self):
        invalid, _, calls = self._run("C")
        self.assertNotEqual(invalid.returncode, 0)
        self.assertFalse(calls)

        nonempty_result, exp_dir, calls = self._run("A", ALLOW_OVERWRITE="0")
        self.assertEqual(nonempty_result.returncode, 0)
        rerun, _, rerun_calls = self._run("A", EXP_DIR=str(exp_dir), ALLOW_OVERWRITE="0")
        self.assertNotEqual(rerun.returncode, 0)
        self.assertFalse(rerun_calls)

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "target"
            target.mkdir()
            link = Path(temp_dir) / "link"
            link.symlink_to(target, target_is_directory=True)
            symlink_result, _, symlink_calls = self._run("A", EXP_DIR=str(link))
            self.assertNotEqual(symlink_result.returncode, 0)
            self.assertFalse(symlink_calls)

        lock_result, lock_exp, _ = self._run("A")
        self.assertEqual(lock_result.returncode, 0)
        Path(f"{lock_exp}.lock").mkdir()
        locked, _, locked_calls = self._run(
            "A", EXP_DIR=str(lock_exp), ALLOW_OVERWRITE="1"
        )
        self.assertNotEqual(locked.returncode, 0)
        self.assertFalse(locked_calls)

    def test_wrappers_only_set_variant_and_exec_common_runner(self):
        for path, variant in ((Z64_LDM_V9A_WRAPPER, "A"), (Z64_LDM_V9B_WRAPPER, "B")):
            with self.subTest(path=path.name):
                script = path.read_text(encoding="utf-8")
                self.assertIn(f'export V9_VARIANT="{variant}"', script)
                self.assertRegex(script, r'exec\s+"\$\{SELF_DIR\}/run_ldm_z64_v9_screen\.sh"')
                self.assertNotIn("MINI_LDM_", script)


class LDMZ64V9RealInterfaceSmokeTest(unittest.TestCase):
    """执行真实 train_minimal 与 ablation shell 接口，替换耗时 Python/conda 进程。"""

    def _run(self, produce_checkpoint):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        temp_path = Path(temp_dir.name)
        exp_dir = temp_path / "experiment"
        base_vae = temp_path / "base_vae.pt"
        base_vae.write_text("vae\n", encoding="utf-8")
        data_root = temp_path / "sensor_data"
        for voxel_dir in ("radar_voxel", "target_voxel"):
            path = data_root / "garden" / voxel_dir
            path.mkdir(parents=True)
            (path / "000000.npy").write_bytes(b"placeholder")

        bin_dir = temp_path / "bin"
        bin_dir.mkdir()
        call_log = temp_path / "interface_calls.log"
        fake_python = bin_dir / "fake-python"
        fake_python.write_text(
            "#!/bin/sh\n"
            "if [ \"$1\" = - ]; then\n"
            "  cat >/dev/null\n"
            "  : > \"$3\"\n"
            "  printf 'config|%s|%s\\n' \"$2\" \"$3\" >> \"$INTERFACE_LOG\"\n"
            "  exit 0\n"
            "fi\n"
            "case \"$1\" in\n"
            "  *unified_train.py)\n"
            "    printf 'train|%s|%s|%s|%s|%s|%s|%s|%s\\n' \"$*\" "
            "\"$SAMPLES_PER_SCENE\" \"$MINI_LDM_EPOCHS\" \"$MINI_SPLIT_SEED\" "
            "\"$MINI_TARGET_SIZE\" \"$MINI_SOURCE_PC_RANGE\" \"$MINI_MODEL_PC_RANGE\" "
            "\"$CUDA_VISIBLE_DEVICES\" >> \"$INTERFACE_LOG\"\n"
            "    if [ \"$PRODUCE_CHECKPOINT\" = 1 ]; then\n"
            "      mkdir -p \"$MINI_RESULTS_DIR/ldm\"\n"
            "      : > \"$MINI_RESULTS_DIR/ldm/ldm_best.pt\"\n"
            "    fi\n"
            "    exit 0\n"
            "    ;;\n"
            "esac\n"
            "exit 97\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        fake_conda = bin_dir / "conda"
        fake_conda.write_text(
            "#!/bin/sh\n"
            "printf 'ablation|%s|cuda=%s\\n' \"$*\" \"$CUDA_VISIBLE_DEVICES\" >> \"$INTERFACE_LOG\"\n",
            encoding="utf-8",
        )
        fake_conda.chmod(0o755)
        env = os.environ.copy()
        env.update(
            {
                "V9_VARIANT": "A",
                "EXP_DIR": str(exp_dir),
                "BASE_VAE_CKPT": str(base_vae),
                "PREPROCESSED_ROOT": str(data_root),
                "PYTHON_BIN": str(fake_python),
                "PRODUCE_CHECKPOINT": "1" if produce_checkpoint else "0",
                "INTERFACE_LOG": str(call_log),
                "CUDA_DEVICES": "6",
                "CUDA_VISIBLE_DEVICES": "9",
                "PATH": f"{bin_dir}:{env['PATH']}",
            }
        )
        result = subprocess.run(
            ["/bin/bash", str(Z64_LDM_V9_RUNNER)],
            cwd=Z64_LDM_V9_RUNNER.parents[2],
            env=env,
            text=True,
            capture_output=True,
        )
        calls = call_log.read_text(encoding="utf-8").splitlines() if call_log.exists() else []
        return result, exp_dir, data_root, calls

    def test_real_shell_interfaces_propagate_training_and_ablation_protocol(self):
        result, exp_dir, data_root, calls = self._run(produce_checkpoint=True)
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        train = next(line for line in calls if line.startswith("train|"))
        for expected in (
            "|500|3|42|64,128,128|0,-20,-6,120,20,10|0,-20,-6,40,20,10|6",
            "--mode ldm",
        ):
            self.assertIn(expected, train)
        ablation = next(line for line in calls if line.startswith("ablation|"))
        for expected in (
            f"--dataset_root {data_root}",
            f"--vae_ckpt {exp_dir}/vae/vae_best.pt",
            f"--model_ckpt {exp_dir}/ldm/ldm_best.pt",
            f"--output_dir {exp_dir}/ir_target_ablation_32_thr099",
            "--max_samples 32",
            "--steps 20",
            "--occ_threshold 0.99",
            "--target_threshold 0.5",
            "cuda=6",
        ):
            self.assertIn(expected, ablation)

    def test_ablation_requires_successful_training_checkpoint(self):
        result, _, _, calls = self._run(produce_checkpoint=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(any(line.startswith("train|") for line in calls))
        self.assertFalse(any(line.startswith("ablation|") for line in calls))
        self.assertIn("LDM checkpoint", result.stdout + result.stderr)


class LDMZ64V10ColumnExperimentTest(unittest.TestCase):
    """验证 v10 列级损失 runner 的隔离变量、训练-only 和输出保护。"""

    @staticmethod
    def _install_fake_bash(temp_path):
        bin_dir = temp_path / "bin"
        bin_dir.mkdir()
        fake_bash = bin_dir / "bash"
        fake_bash.write_text(
            "#!/bin/sh\n"
            "printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\\n' \"$*\" "
            "\"$MINI_LDM_COLUMN_POSITIVE_WEIGHT\" \"$MINI_LDM_COLUMN_NEGATIVE_WEIGHT\" "
            "\"$MINI_LDM_COLUMN_TEMPERATURE\" \"$MINI_LDM_DECODED_WEIGHT\" "
            "\"$MINI_LDM_DECODED_FP_WEIGHT\" \"$MINI_LDM_DECODED_MASS_WEIGHT\" "
            "\"$MINI_LDM_HEIGHT_WEIGHT\" \"$MINI_LDM_TOP_WEIGHT\" "
            "\"$MINI_LDM_TOP_OVERSHOOT_WEIGHT\" \"$MINI_LDM_CONTINUITY_WEIGHT\" "
            "\"$MINI_LDM_DENSITY_WEIGHT\" \"$MINI_LDM_IR_FRUSTUM_OCC_WEIGHT\" "
            "\"$MINI_LDM_IR_FRUSTUM_NEGATIVE_WEIGHT\" \"$MINI_LDM_IR_FRUSTUM_TOP_WEIGHT\" "
            "\"$MINI_LDM_UNCERTAINTY_WEIGHT\" \"$SAMPLES_PER_SCENE\" \"$MINI_LDM_EPOCHS\" "
            "\"$MINI_TARGET_SIZE\" \"$MINI_SPLIT_SEED\" \"$TRAIN_SCENES_OVERRIDE\" "
            "\"$MINI_DATASET_DIR\" \"$MINI_CONFIG_PATH\" \"$CUDA_DEVICES\" \"$CUDA_VISIBLE_DEVICES\" "
            ">> \"$CALL_LOG\"\n"
            "mkdir -p \"$EXP_DIR/ldm\"\n"
            "if [ \"${FAKE_EMPTY_FINAL:-0}\" = 1 ]; then : > \"$EXP_DIR/ldm/ldm_best.pt\"; "
            "else printf 'ldm\\n' > \"$EXP_DIR/ldm/ldm_best.pt\"; fi\n",
            encoding="utf-8",
        )
        fake_bash.chmod(0o755)
        fake_mkdir = bin_dir / "mkdir"
        fake_mkdir.write_text(
            "#!/bin/sh\n"
            "/bin/mkdir \"$@\" || exit $?\n"
            "case \"$*\" in\n"
            "  *\"$EXP_DIR.v10.lock\"*)\n"
            "    if [ -n \"${RACE_AFTER_LOCK_MARKER:-}\" ]; then\n"
            "      /bin/mkdir -p \"$EXP_DIR\"\n"
            "      printf 'race\\n' > \"$EXP_DIR/$RACE_AFTER_LOCK_MARKER\"\n"
            "    fi\n"
            "    ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        fake_mkdir.chmod(0o755)
        return bin_dir

    def _run(self, variant="A", **overrides):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        temp_path = Path(temp_dir.name)
        base_vae = temp_path / "vae_best.pt"
        base_vae.write_text("vae\n", encoding="utf-8")
        exp_dir = temp_path / "experiment"
        call_log = temp_path / "calls.log"
        bin_dir = self._install_fake_bash(temp_path)
        env = os.environ.copy()
        env.update({
            "V10_VARIANT": variant,
            "EXP_DIR": str(exp_dir),
            "BASE_VAE_CKPT": str(base_vae),
            "CALL_LOG": str(call_log),
            "PATH": f"{bin_dir}:{env['PATH']}",
            "CUDA_VISIBLE_DEVICES": "7",
        })
        env.pop("CUDA_DEVICES", None)
        env.update(overrides)
        result = subprocess.run(
            ["/bin/bash", str(Z64_LDM_V10_RUNNER)],
            cwd=Z64_LDM_V10_RUNNER.parents[2], env=env, text=True, capture_output=True,
        )
        calls = call_log.read_text(encoding="utf-8").splitlines() if call_log.exists() else []
        return result, exp_dir, calls

    def test_variants_define_isolated_column_weight_screens(self):
        result_a, _, calls_a = self._run("A")
        result_b, _, calls_b = self._run("B")
        result_c, _, calls_c = self._run("C")
        result_d, _, calls_d = self._run("D")
        self.assertEqual(result_a.returncode, 0, msg=result_a.stdout + result_a.stderr)
        self.assertEqual(result_b.returncode, 0, msg=result_b.stdout + result_b.stderr)
        self.assertEqual(result_c.returncode, 0, msg=result_c.stdout + result_c.stderr)
        self.assertEqual(result_d.returncode, 0, msg=result_d.stdout + result_d.stderr)
        values_a = calls_a[0].split("|")[1:]
        values_b = calls_b[0].split("|")[1:]
        values_c = calls_c[0].split("|")[1:]
        values_d = calls_d[0].split("|")[1:]
        self.assertEqual(values_a[:3], ["0.02", "0.01", "1.0"])
        self.assertEqual(values_b[:3], ["0.02", "0.02", "1.0"])
        self.assertEqual(values_c[:3], ["0.03", "0.01", "1.0"])
        self.assertEqual(values_d[:3], ["0.02", "0.005", "1.0"])
        for candidate in (values_b, values_c, values_d):
            self.assertEqual(values_a[3:20], candidate[3:20])
            self.assertEqual(values_a[22:], candidate[22:])

    def test_unknown_variant_is_rejected_before_writing_or_training(self):
        result, exp_dir, calls = self._run("unknown")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(calls)
        self.assertFalse(exp_dir.exists())
        self.assertIn("must be A, B, C, or D", result.stdout + result.stderr)

    def test_v9a_weights_and_training_protocol_are_preserved(self):
        result, exp_dir, calls = self._run("A")
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        values = calls[0].split("|")
        self.assertEqual(values[4:16], [
            "0.12", "0.20", "0.08", "0.04", "0.08", "0.02", "0.02", "0.0",
            "0.02", "0.02", "0.03", "0.0",
        ])
        self.assertEqual(values[16:21], ["500", "3", "64,128,128", "42", "garden"])
        self.assertEqual(values[21], str(exp_dir / ".tmp_mini_train_dataset"))
        self.assertEqual(values[22], str(exp_dir / ".tmp_ldm_config.yaml"))
        self.assertEqual(values[23:25], ["7", "7"])

    def test_fixed_protocol_ignores_hostile_environment_overrides(self):
        result, _, calls = self._run(
            "A", SAMPLES_PER_SCENE="1", MINI_LDM_EPOCHS="99",
            MINI_TARGET_SIZE="1,2,3", MINI_SOURCE_PC_RANGE="bad",
            MINI_MODEL_PC_RANGE="bad", MINI_SPLIT_SEED="999",
            TRAIN_SCENES_OVERRIDE="loop3",
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        values = calls[0].split("|")
        self.assertEqual(values[16:21], ["500", "3", "64,128,128", "42", "garden"])

    def test_reproducibility_protocol_is_explicitly_fixed(self):
        script = Z64_LDM_V10_RUNNER.read_text(encoding="utf-8")
        expected_exports = (
            'export MINI_BATCH_SIZE="1"',
            'export MINI_NUM_WORKERS="2"',
            'export MINI_GRAD_ACCUM="1"',
            'export MINI_USE_AUG="false"',
            'export MINI_TRAIN_SPLIT="0.8"',
            'export MINI_VAE_CONFIG_TYPE="ultra_lightweight"',
            'export MINI_VAE_LATENT_DIM=""',
            'export MINI_VAE_OCC_LOSS="bce_dice"',
            'export PREPROCESSED_ROOT="${ROOT_DIR}/Data/NTU4DRadLM_Pre_sensor_aware"',
            'export CALIB_CONFIG_DIR="${ROOT_DIR}/Data/config"',
            'export LDM_TRAIN_ONLY="1"',
            'export MINI_REQUIRE_FRESH_SCRATCH="1"',
            'export MINI_REQUIRE_FRESH_CONFIG="1"',
        )
        for export_line in expected_exports:
            with self.subTest(export_line=export_line):
                self.assertIn(export_line, script)

    def test_runner_is_training_only_and_checks_final_checkpoint(self):
        result, _, calls = self._run("A")
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertIn("run_ldm_vertical_experiment.sh", calls[0])
        script = Z64_LDM_V10_RUNNER.read_text(encoding="utf-8")
        for forbidden in ("inference_minimal", "diagnose_ir_condition_ablation", "evaluate_ldm", "cd_train"):
            self.assertNotIn(forbidden, script.lower())

    def test_rejects_unsafe_symlink_nonempty_and_active_lock(self):
        unsafe, _, calls = self._run("A", EXP_DIR="/var/tmp/v10-outside")
        self.assertNotEqual(unsafe.returncode, 0)
        self.assertFalse(calls)

    def test_content_injected_after_lock_is_rejected_before_first_write(self):
        result, exp_dir, calls = self._run("A", RACE_AFTER_LOCK_MARKER="intruder.txt")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(calls)
        self.assertEqual((exp_dir / "intruder.txt").read_text(encoding="utf-8"), "race\n")
        self.assertFalse((exp_dir / "vae").exists())
        script = Z64_LDM_V10_RUNNER.read_text(encoding="utf-8")
        post_lock_check = script.index("check_exp_empty_after_lock")
        self.assertLess(post_lock_check, script.index('mkdir -p -- "${EXP_DIR}/vae"'))

        nonempty_dir = Path(tempfile.mkdtemp(dir="/tmp"))
        self.addCleanup(lambda: nonempty_dir.rmdir() if nonempty_dir.exists() else None)
        (nonempty_dir / "owned.txt").write_text("keep\n", encoding="utf-8")
        self.addCleanup(lambda: (nonempty_dir / "owned.txt").unlink(missing_ok=True))
        nonempty, _, calls = self._run("A", EXP_DIR=str(nonempty_dir))
        self.assertNotEqual(nonempty.returncode, 0)
        self.assertFalse(calls)
        self.assertTrue((nonempty_dir / "owned.txt").is_file())

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "target"
            target.mkdir()
            link = Path(temp_dir) / "link"
            link.symlink_to(target, target_is_directory=True)
            symlink, _, calls = self._run("A", EXP_DIR=str(link))
            self.assertNotEqual(symlink.returncode, 0)
            self.assertFalse(calls)

        locked_dir = Path("/tmp/v10_locked_test")
        lock = Path(f"{locked_dir}.v10.lock")
        lock.mkdir(exist_ok=True)
        self.addCleanup(lambda: lock.rmdir() if lock.exists() else None)
        locked, _, calls = self._run("A", EXP_DIR=str(locked_dir))
        self.assertNotEqual(locked.returncode, 0)
        self.assertFalse(calls)

    def test_missing_base_vae_is_rejected_before_training(self):
        result, _, calls = self._run("A", BASE_VAE_CKPT="/tmp/missing-v10-vae.pt")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(calls)
        self.assertIn("VAE checkpoint", result.stdout + result.stderr)

    def test_empty_base_vae_and_empty_final_checkpoint_are_rejected(self):
        empty_vae = Path(tempfile.mkstemp(prefix="empty-v10-vae-", dir="/tmp")[1])
        self.addCleanup(lambda: empty_vae.unlink(missing_ok=True))
        result, _, calls = self._run("A", BASE_VAE_CKPT=str(empty_vae))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(calls)
        self.assertIn("VAE checkpoint", result.stdout + result.stderr)

        result, _, calls = self._run("A", FAKE_EMPTY_FINAL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(calls), 1)
        self.assertIn("final LDM checkpoint", result.stdout + result.stderr)

    def test_runtime_path_audit_surrounds_each_mutating_stage(self):
        script = Z64_LDM_V10_RUNNER.read_text(encoding="utf-8")
        self.assertIn("audit_runtime_paths()", script)
        self.assertIn('canonical_now="$(realpath -m -- "${EXP_DIR}")"', script)
        self.assertIn('parent="$(realpath -m -- "$(dirname -- "${path}")")"', script)
        self.assertGreaterEqual(script.count("audit_runtime_paths"), 6)
        for mutation in (
            'mkdir -p -- "${EXP_DIR}/vae"',
            'cp -a -- "${BASE_VAE_CKPT}" "${VAE_CKPT_PATH}"',
            'bash "${SELF_DIR}/run_ldm_vertical_experiment.sh"',
        ):
            index = script.index(mutation)
            self.assertIn("audit_runtime_paths", script[max(0, index - 300):index])
            self.assertIn("audit_runtime_paths", script[index:index + 300])

    def test_v10_requires_fresh_scratch_without_precreating_it(self):
        script = Z64_LDM_V10_RUNNER.read_text(encoding="utf-8")
        self.assertIn('export MINI_REQUIRE_FRESH_SCRATCH="1"', script)
        self.assertNotIn('mkdir -p -- "${MINI_DATASET_DIR}"', script)
        self.assertIn('export MINI_REQUIRE_FRESH_CONFIG="1"', script)

    def test_real_shell_interface_reaches_only_final_ldm_training_command(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            exp_dir = temp_path / "experiment"
            base_vae = temp_path / "vae_best.pt"
            base_vae.write_text("vae\n", encoding="utf-8")
            data_root = temp_path / "data"
            for voxel_dir in ("radar_voxel", "target_voxel"):
                path = data_root / "garden" / voxel_dir
                path.mkdir(parents=True)
                (path / "000000.npy").write_bytes(b"placeholder")
            call_log = temp_path / "train.log"
            fake_python = temp_path / "fake-python"
            fake_python.write_text(
                "#!/bin/sh\n"
                "printf '%s|pos=%s|neg=%s|temp=%s|density=%s|cuda=%s\\n' \"$*\" "
                "\"$MINI_LDM_COLUMN_POSITIVE_WEIGHT\" \"$MINI_LDM_COLUMN_NEGATIVE_WEIGHT\" "
                "\"$MINI_LDM_COLUMN_TEMPERATURE\" \"$MINI_LDM_DENSITY_WEIGHT\" "
                "\"$CUDA_VISIBLE_DEVICES\" >> \"$TRAIN_LOG\"\n"
                "mkdir -p \"$MINI_RESULTS_DIR/ldm\"\n"
                "printf 'ldm\\n' > \"$MINI_RESULTS_DIR/ldm/ldm_best.pt\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = os.environ.copy()
            env.update({
                "V10_VARIANT": "B",
                "EXP_DIR": str(exp_dir),
                "BASE_VAE_CKPT": str(base_vae),
                "PREPROCESSED_ROOT": str(data_root),
                "PYTHON_BIN": str(fake_python),
                "TRAIN_LOG": str(call_log),
                "CUDA_DEVICES": "5",
            })
            result = subprocess.run(
                ["/bin/bash", str(Z64_LDM_V10_RUNNER)],
                cwd=Z64_LDM_V10_RUNNER.parents[2], env=env, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            calls = call_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(calls), 1)
            self.assertIn("--mode ldm", calls[0])
            self.assertIn("pos=0.02|neg=0.02|temp=1.0|density=0.0|cuda=5", calls[0])
            self.assertGreater((exp_dir / "ldm" / "ldm_best.pt").stat().st_size, 0)
            for forbidden_output in (
                exp_dir / "loop3_ldm_eval",
                exp_dir / "vertical_structure_eval",
                exp_dir / "raw_lidar_visuals",
            ):
                self.assertFalse(forbidden_output.exists())


if __name__ == "__main__":
    unittest.main()
