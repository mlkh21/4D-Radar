#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提供严格数据集 manifest 的生成与只读验证命令行入口。"""

import argparse
import json
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion_consistency_radar.dataset_manifest import (  # noqa: E402
    DatasetManifestError,
    validate_scene_manifest,
    write_scene_manifest_atomic,
)


def build_parser() -> argparse.ArgumentParser:
    """构建 create/validate 两个严格子命令。"""
    parser = argparse.ArgumentParser(
        description="生成或验证逐帧内容 SHA-256 数据集 manifest",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser(
        "create",
        help="为刚完成且尚无 manifest 的干净场景发布 manifest",
    )
    create_parser.add_argument("--scene_dir", required=True)
    create_parser.add_argument("--scene", required=True)
    create_parser.add_argument(
        "--expected_frame_count",
        required=True,
        type=int,
    )
    create_parser.add_argument("--preprocess_script", required=True)
    create_parser.add_argument("--calibration", required=True)
    create_parser.add_argument("--radar_index", required=True)
    create_parser.add_argument("--lidar_index", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="只读重算场景内容并与已发布 manifest 比较",
    )
    validate_parser.add_argument("--scene_dir", required=True)
    validate_parser.add_argument("--expected_scene", required=True)
    return parser


def main(argv=None) -> None:
    """执行命令；协议错误统一转换为 argparse 的非零退出。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            provenance_paths = {
                "preprocess_script": args.preprocess_script,
                "calibration": args.calibration,
                "radar_index": args.radar_index,
                "lidar_index": args.lidar_index,
            }
            manifest_path = write_scene_manifest_atomic(
                args.scene_dir,
                args.scene,
                args.expected_frame_count,
                provenance_paths,
            )
            manifest = validate_scene_manifest(args.scene_dir, args.scene)
        else:
            manifest_path = os.path.join(
                os.path.abspath(args.scene_dir),
                "dataset_manifest.json",
            )
            manifest = validate_scene_manifest(
                args.scene_dir,
                args.expected_scene,
            )
    except DatasetManifestError as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "manifest_path": manifest_path,
                "scene": manifest["scene"],
                "frame_count": manifest["frame_count"],
                "content_sha256": manifest["content_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
