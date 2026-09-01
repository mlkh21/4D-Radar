#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文件功能：生成正式训练 checkpoint 链使用的数据身份 artifact。"""

import argparse
import json
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion_consistency_radar.formal_data_protocol import (  # noqa: E402
    build_and_write_formal_data_protocol,
    load_formal_data_protocol_artifact,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="从 training manifests、split 和监督模态生成 formal_data_v2/v3"
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--split_artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--protocol_version",
        choices=("v2", "v3"),
        default="v2",
    )
    return parser


def main():
    args = build_parser().parse_args()
    path = build_and_write_formal_data_protocol(
        dataset_dir=args.dataset_dir,
        scenes=args.scene,
        split_artifact_path=args.split_artifact,
        output_path=args.output,
        protocol_version=args.protocol_version,
    )
    _protocol, digest = load_formal_data_protocol_artifact(
        path,
        dataset_dir=args.dataset_dir,
        scenes=args.scene,
        split_artifact_path=args.split_artifact,
        stage="ldm",
    )
    print(
        json.dumps(
            {
                "artifact_path": path,
                "sha256": digest,
                "protocol": _protocol["protocol"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
