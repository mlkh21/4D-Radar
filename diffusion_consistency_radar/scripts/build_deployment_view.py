#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成或只读验证严格 Radar+IR deployment-profile 数据视图。"""

import argparse
import json
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion_consistency_radar.deployment_view import (  # noqa: E402
    build_deployment_dataset,
    validate_deployment_dataset,
)
from diffusion_consistency_radar.dataset_manifest import DatasetManifestError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构建 create/validate 严格子命令。"""
    parser = argparse.ArgumentParser(
        description="从 training v2 生成带父身份收据的 deployment v3 数据视图",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="在 fresh 根发布 deployment view")
    create.add_argument("--training_dataset_dir", required=True)
    create.add_argument("--output_dataset_dir", required=True)
    create.add_argument("--calibration_dir", required=True)
    create.add_argument("--preprocess_script", required=True)
    create.add_argument("--scene", required=True, action="append")
    create.add_argument(
        "--link_mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="默认硬链接节省磁盘；跨文件系统时显式使用 copy",
    )

    validate = subparsers.add_parser("validate", help="只读重算并验证完整视图")
    validate.add_argument("--dataset_dir", required=True)
    validate.add_argument("--scene", required=True, action="append")
    return parser


def main(argv=None) -> None:
    """执行命令并把协议错误转换为稳定的非零退出。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            result = build_deployment_dataset(
                training_dataset_dir=args.training_dataset_dir,
                output_dataset_dir=args.output_dataset_dir,
                scenes=args.scene,
                calibration_dir=args.calibration_dir,
                preprocess_script=args.preprocess_script,
                link_mode=args.link_mode,
            )
        else:
            result = validate_deployment_dataset(
                args.dataset_dir,
                scenes=args.scene,
            )
    except DatasetManifestError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
