#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从正式 LDM/CD checkpoint 的 validation sweep 构建部署阈值 artifact。"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(PROJECT_DIR)
for import_root in (PROJECT_DIR, ROOT_DIR):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from diffusion_consistency_radar.checkpoint_chain import (
    FORMAL_CHECKPOINT_PROTOCOL,
    safe_torch_load,
)
from diffusion_consistency_radar.occupancy_threshold_artifact import (
    build_threshold_artifact,
    write_threshold_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建绑定正式 checkpoint SHA-256 的 occupancy threshold artifact"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    output_path = os.path.abspath(args.output)
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    if checkpoint.get("checkpoint_protocol") != FORMAL_CHECKPOINT_PROTOCOL:
        raise ValueError("只允许从 formal_chain_v2 checkpoint 构建正式阈值 artifact")
    artifact = build_threshold_artifact(
        checkpoint,
        checkpoint_path=checkpoint_path,
    )
    write_threshold_artifact(output_path, artifact)
    print(json.dumps({
        "artifact_path": output_path,
        "checkpoint_sha256": artifact["checkpoint_sha256"],
        "selected_threshold": artifact["selected_threshold"],
        "stage": artifact["stage"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
