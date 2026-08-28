#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""文件功能：从候选与跨帧重合诊断发布离线经验 LiDAR 位姿合同。"""

import argparse
import json
import os
import shlex
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffusion_consistency_radar.empirical_pose_contract import (  # noqa: E402
    build_empirical_lidar_pose_contract,
)


def build_parser() -> argparse.ArgumentParser:
    """构造不允许隐式来源猜测的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="发布仅限离线地图、airborne_formal=false 的经验 LiDAR pose receipt"
    )
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--overlap_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command_line = " ".join(shlex.quote(argument) for argument in sys.argv)
    receipt = build_empirical_lidar_pose_contract(
        candidate_dir=args.candidate_dir,
        overlap_dir=args.overlap_dir,
        output_dir=args.output_dir,
        command_line=command_line,
    )
    print(
        json.dumps(
            {
                "receipt_path": os.path.abspath(
                    os.path.join(args.output_dir, "empirical_pose_receipt.json")
                ),
                "selected_pose_frame_count": receipt["coverage"][
                    "selected_pose_frame_count"
                ],
                "airborne_formal": receipt["airborne_formal"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
