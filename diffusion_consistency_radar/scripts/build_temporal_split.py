#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文件功能：从正式训练数据生成唯一 temporal split 与 purge artifact。"""

import argparse
import json
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion_consistency_radar.temporal_split import (  # noqa: E402
    build_and_write_temporal_split,
    load_temporal_split_artifact,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="生成按 Radar 时间排序、带 purge gap 的唯一训练/验证切分"
    )
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train_fraction", type=float, required=True)
    parser.add_argument("--purge_seconds", type=float, required=True)
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="生成 formal=false 诊断切分；正式入口会拒绝",
    )
    return parser


def main():
    args = build_parser().parse_args()
    path = build_and_write_temporal_split(
        dataset_dir=args.dataset_dir,
        scenes=args.scene,
        output_path=args.output,
        train_fraction=args.train_fraction,
        purge_seconds=args.purge_seconds,
        formal=not args.diagnostic,
    )
    artifact, digest = load_temporal_split_artifact(
        path,
        dataset_dir=args.dataset_dir,
        expected_scenes=args.scene,
        require_formal=not args.diagnostic,
    )
    print(
        json.dumps(
            {
                "artifact_path": path,
                "sha256": digest,
                "formal": artifact["formal"],
                "scenes": sorted(artifact["scenes"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
