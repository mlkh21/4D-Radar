#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion_consistency_radar.scripts.unified_train import (
    ConfigManager,
    resolve_cd_teacher_checkpoint,
)


def _config(path, teacher):
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "cd:\n"
            f"  teacher_model_path: \"{teacher}\"\n"
        )


def test_cd_teacher_checkpoint_prefers_cli_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "config.yaml")
        _config(cfg_path, "/from/config.pt")
        config = ConfigManager(cfg_path)
        assert resolve_cd_teacher_checkpoint("/from/cli.pt", config) == "/from/cli.pt"


def test_cd_teacher_checkpoint_falls_back_to_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = os.path.join(tmpdir, "config.yaml")
        _config(cfg_path, "/from/config.pt")
        config = ConfigManager(cfg_path)
        assert resolve_cd_teacher_checkpoint("", config) == "/from/config.pt"


if __name__ == "__main__":
    test_cd_teacher_checkpoint_prefers_cli_path()
    test_cd_teacher_checkpoint_falls_back_to_config()
    print("test_cd_training_entrypoints passed")
