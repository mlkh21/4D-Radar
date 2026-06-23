#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit preprocessed dataset protocol coverage for IR/calibration metadata."""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cm.dataset_loader import CalibrationProvider  # noqa: E402


def _list_voxel_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.endswith(".npy") or f.endswith(".npz")]
    return sorted(f for f in files if not f.endswith("_pcl.npy"))


def _has_ir(scene_dir: str, voxel_file: str) -> bool:
    stem = os.path.splitext(voxel_file)[0]
    candidates = [
        os.path.join(scene_dir, "ir_image", f"{stem}_ir.npy"),
        os.path.join(scene_dir, "ir_image", f"{stem}.npy"),
        os.path.join(scene_dir, "ir_image", f"{stem}.npz"),
    ]
    return any(os.path.exists(path) for path in candidates)


def audit_scene(dataset_root: str, scene: str) -> Dict[str, object]:
    scene_dir = os.path.join(dataset_root, scene)
    radar_dir = os.path.join(scene_dir, "radar_voxel")
    target_dir = os.path.join(scene_dir, "target_voxel")
    radar_files = _list_voxel_files(radar_dir)
    target_files = _list_voxel_files(target_dir)
    ir_count = sum(1 for f in radar_files if _has_ir(scene_dir, f))
    policy_path = os.path.join(scene_dir, "preprocess_policy.json")
    policy = {}
    if os.path.exists(policy_path):
        with open(policy_path, "r", encoding="utf-8") as f:
            policy = json.load(f)

    _, _, _, is_mock_calib = CalibrationProvider(dataset_root).load()
    frame_count = len(radar_files)
    return {
        "scene": scene,
        "radar_frames": frame_count,
        "target_frames": len(target_files),
        "ir_frames": ir_count,
        "ir_coverage": (ir_count / frame_count) if frame_count else 0.0,
        "has_preprocess_policy": bool(policy),
        "align_to": policy.get("align_to", ""),
        "is_mock_calib": bool(is_mock_calib),
    }


def write_report(rows: List[Dict[str, object]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "dataset_protocol_audit.csv")
    md_path = os.path.join(output_dir, "dataset_protocol_audit.md")
    headers = [
        "scene",
        "radar_frames",
        "target_frames",
        "ir_frames",
        "ir_coverage",
        "has_preprocess_policy",
        "align_to",
        "is_mock_calib",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Protocol Audit\n\n")
        f.write("| scene | radar | target | IR coverage | policy | align_to | mock calib |\n")
        f.write("| --- | ---: | ---: | ---: | --- | --- | --- |\n")
        for row in rows:
            f.write(
                f"| {row['scene']} | {row['radar_frames']} | {row['target_frames']} | "
                f"{float(row['ir_coverage']):.3f} | {row['has_preprocess_policy']} | "
                f"{row['align_to']} | {row['is_mock_calib']} |\n"
            )
    print(f"Saved audit CSV to: {csv_path}")
    print(f"Saved audit report to: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dataset IR/calibration/preprocess protocol coverage")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./Result/dataset_protocol_audit")
    parser.add_argument("--scenes", type=str, default="", help="Comma-separated scene list. Defaults to all scene folders.")
    args = parser.parse_args()

    if args.scenes:
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        scenes = sorted(
            d for d in os.listdir(args.dataset_root)
            if os.path.isdir(os.path.join(args.dataset_root, d))
        )
    rows = [audit_scene(args.dataset_root, scene) for scene in scenes]
    write_report(rows, args.output_dir)


if __name__ == "__main__":
    main()
