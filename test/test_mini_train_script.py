#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""静态验证最小训练脚本的实验配置透传契约，不启动任何训练。"""

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


SCRIPT_PATH = Path(__file__).parent / "mini-test" / "train_minimal.sh"

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
    ("MINI_LDM_HEIGHT_WEIGHT", "ldm_height_weight"),
    ("MINI_LDM_CONTINUITY_WEIGHT", "ldm_continuity_weight"),
]


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
                "MINI_LDM_HEIGHT_WEIGHT": "0.125",
                "MINI_LDM_CONTINUITY_WEIGHT": "0.375",
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
            "MINI_LDM_HEIGHT_WEIGHT": "0.02",
            "MINI_LDM_CONTINUITY_WEIGHT": "0.02",
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
            "MINI_LDM_HEIGHT_WEIGHT",
            "MINI_LDM_CONTINUITY_WEIGHT",
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
            "cfg['ldm']['decoded_height_distribution_weight'] = float(ldm_height_weight)",
            "cfg['ldm']['decoded_vertical_continuity_weight'] = float(ldm_continuity_weight)",
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

        self.assertEqual(generated["ldm"]["decoded_height_distribution_weight"], 0.125)
        self.assertEqual(generated["ldm"]["decoded_vertical_continuity_weight"], 0.375)

    def test_prints_effective_experiment_values(self):
        for variable in (
            "MINI_VAE_CONFIG_TYPE",
            "MINI_VAE_LATENT_DIM",
            "MINI_VAE_OCC_LOSS",
            "MINI_TRAIN_SPLIT",
            "MINI_SPLIT_SEED",
            "MINI_LDM_HEIGHT_WEIGHT",
            "MINI_LDM_CONTINUITY_WEIGHT",
        ):
            with self.subTest(variable=variable):
                self.assertRegex(
                    self.script,
                    rf'echo "[^"]*\$\{{{variable}(?::-[^}}]*)?\}}[^"]*"',
                    msg=f"setup 未打印 {variable} 的有效值",
                )

    def test_does_not_add_unused_patience_setting(self):
        self.assertNotIn("patience", self.script.lower())


if __name__ == "__main__":
    unittest.main()
