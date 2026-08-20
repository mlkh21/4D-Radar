# -*- coding: utf-8 -*-
"""生成 NTU4DRadLM Radar/LiDAR 时间戳索引和实际同步差值记录。"""

import argparse
import csv
import math
import os
import tempfile
from typing import Dict, List

if __package__:
    # 作为包模块导入时使用相对路径，避免同名脚本遮蔽命名空间包。
    from .timestamp_alignment import nearest_timestamp_match
else:
    # 直接执行本脚本时，脚本所在目录位于 sys.path 中。
    from timestamp_alignment import nearest_timestamp_match  # type: ignore[no-redef]


DEFAULT_RADAR_LIDAR_MAX_DELTA = 0.045
DEFAULT_MAX_REJECTED_FRACTION = 0.01
REJECTED_FILENAME = "radar_lidar_rejected.csv"


def _timestamped_files(directory: str) -> List[str]:
    """按文件名中的浮点时间戳排序，拒绝重复或无效时间戳。"""
    names = [name for name in os.listdir(directory) if name.endswith(".npy")]
    try:
        names.sort(key=lambda name: float(os.path.splitext(name)[0]))
    except ValueError as exc:
        raise ValueError(f"目录 {directory} 中存在无法解析时间戳的 .npy 文件") from exc
    timestamps = [float(os.path.splitext(name)[0]) for name in names]
    if timestamps and (
        not all(math.isfinite(value) for value in timestamps)
        or any(after <= before for before, after in zip(timestamps, timestamps[1:]))
    ):
        raise ValueError(f"目录 {directory} 的文件名时间戳必须严格递增且为有限数")
    return names


def find_nearest_index(timestamps, target, max_delta=None):
    """兼容旧调用方的最近邻索引接口；可选地启用时间容差。"""
    index, _ = nearest_timestamp_match(timestamps, target, max_delta=max_delta)
    return index


def _atomic_write(path: str, content: str) -> None:
    """在同目录临时文件中写入后替换，避免单个索引文件只写入一部分。"""
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=os.path.dirname(path)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def _records_to_csv(records: List[Dict[str, object]]) -> str:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "pair_index",
            "radar_index",
            "lidar_index",
            "radar_timestamp",
            "lidar_timestamp",
            "delta_seconds",
            "signed_delta_seconds",
        ),
    )
    writer.writeheader()
    for record in records:
        writer.writerow(record)
    return buffer.getvalue()


def _rejected_records_to_csv(records: List[Dict[str, object]]) -> str:
    """序列化超限候选，确保被跳过的帧仍然可审计。"""
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "candidate_index",
            "radar_index",
            "lidar_index",
            "radar_timestamp",
            "lidar_timestamp",
            "delta_seconds",
            "signed_delta_seconds",
            "reason",
        ),
    )
    writer.writeheader()
    for record in records:
        writer.writerow(record)
    return buffer.getvalue()


def generate_scene_indices(
    scene_path: str,
    radar_lidar_max_delta: float = DEFAULT_RADAR_LIDAR_MAX_DELTA,
    *,
    skip_unmatched: bool = False,
    max_rejected_fraction: float = DEFAULT_MAX_REJECTED_FRACTION,
):
    """生成可审计的 Radar/LiDAR 最近邻索引。

    默认保持 fail-closed：任一候选超限就失败。异步传感器正式重建可显式
    启用 ``skip_unmatched``，把少量超限候选写入 rejected CSV；若拒绝比例
    超过门禁，仍在创建或覆盖任何索引文件前失败。
    """
    try:
        max_delta = float(radar_lidar_max_delta)
    except (TypeError, ValueError) as exc:
        raise ValueError("Radar-LiDAR 时间容差必须是有限非负数") from exc
    if max_delta < 0.0 or not math.isfinite(max_delta):
        raise ValueError("Radar-LiDAR 时间容差必须是有限非负数")
    if type(skip_unmatched) is not bool:
        raise ValueError("skip_unmatched 必须是 bool")
    try:
        rejected_fraction_limit = float(max_rejected_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("最大拒绝比例必须是 [0,1] 内的有限数") from exc
    if (
        not math.isfinite(rejected_fraction_limit)
        or rejected_fraction_limit < 0.0
        or rejected_fraction_limit > 1.0
    ):
        raise ValueError("最大拒绝比例必须是 [0,1] 内的有限数")

    radar_path = os.path.join(scene_path, "radar_pcl")
    lidar_path = os.path.join(scene_path, "livox_lidar")
    if not os.path.isdir(radar_path) or not os.path.isdir(lidar_path):
        raise FileNotFoundError(f"场景缺少 radar_pcl 或 livox_lidar 目录: {scene_path}")

    radar_files = _timestamped_files(radar_path)
    lidar_files = _timestamped_files(lidar_path)
    if not radar_files or not lidar_files:
        raise ValueError(f"场景 {scene_path} 的 Radar/LiDAR 文件不能为空")
    radar_timestamps = [float(os.path.splitext(name)[0]) for name in radar_files]
    lidar_timestamps = [float(os.path.splitext(name)[0]) for name in lidar_files]

    candidates: List[Dict[str, object]] = []
    match_threshold = None if skip_unmatched else max_delta
    if len(radar_timestamps) <= len(lidar_timestamps):
        # 以较稀疏的 Radar 帧为主轴，给每帧匹配最近 LiDAR。
        for radar_index, radar_timestamp in enumerate(radar_timestamps):
            lidar_index, delta = nearest_timestamp_match(
                lidar_timestamps, radar_timestamp, max_delta=match_threshold
            )
            candidates.append(
                {
                    "radar_index": radar_index,
                    "lidar_index": lidar_index,
                    "radar_timestamp": f"{radar_timestamp:.9f}",
                    "lidar_timestamp": f"{lidar_timestamps[lidar_index]:.9f}",
                    "delta_seconds": f"{delta:.9f}",
                    "signed_delta_seconds": f"{lidar_timestamps[lidar_index] - radar_timestamp:.9f}",
                }
            )
    else:
        # NTU4DRadLM 中 LiDAR 更稀疏，以 LiDAR 为主轴避免重复监督帧。
        for lidar_index, lidar_timestamp in enumerate(lidar_timestamps):
            radar_index, delta = nearest_timestamp_match(
                radar_timestamps, lidar_timestamp, max_delta=match_threshold
            )
            candidates.append(
                {
                    "radar_index": radar_index,
                    "lidar_index": lidar_index,
                    "radar_timestamp": f"{radar_timestamps[radar_index]:.9f}",
                    "lidar_timestamp": f"{lidar_timestamp:.9f}",
                    "delta_seconds": f"{delta:.9f}",
                    "signed_delta_seconds": f"{lidar_timestamp - radar_timestamps[radar_index]:.9f}",
                }
            )

    records: List[Dict[str, object]] = []
    rejected_records: List[Dict[str, object]] = []
    for candidate_index, candidate in enumerate(candidates):
        if float(candidate["delta_seconds"]) > max_delta:
            rejected_records.append(
                {
                    "candidate_index": candidate_index,
                    **candidate,
                    "reason": "exceeds_max_delta",
                }
            )
            continue
        records.append({"pair_index": len(records), **candidate})

    if rejected_records and not skip_unmatched:
        raise ValueError("Radar-LiDAR 候选超过时间容差")
    if skip_unmatched:
        rejected_fraction = len(rejected_records) / float(len(candidates))
        if rejected_fraction > rejected_fraction_limit:
            raise ValueError(
                "Radar-LiDAR 超限候选拒绝比例过高: "
                f"rejected={len(rejected_records)}/{len(candidates)} "
                f"({rejected_fraction:.6f}) > {rejected_fraction_limit:.6f}"
            )
    if not records:
        raise ValueError("场景没有满足时间容差的 Radar-LiDAR 配对")

    radar_index_text = "".join(f"{record['radar_index']}\n" for record in records)
    lidar_index_text = "".join(f"{record['lidar_index']}\n" for record in records)
    _atomic_write(os.path.join(scene_path, "radar_index_sequence.txt"), radar_index_text)
    _atomic_write(os.path.join(scene_path, "lidar_index_sequence.txt"), lidar_index_text)
    _atomic_write(os.path.join(scene_path, "radar_lidar_sync.csv"), _records_to_csv(records))
    _atomic_write(
        os.path.join(scene_path, REJECTED_FILENAME),
        _rejected_records_to_csv(rejected_records),
    )
    return records


def generate_new_files(
    directory: str,
    radar_lidar_max_delta: float = DEFAULT_RADAR_LIDAR_MAX_DELTA,
    *,
    skip_unmatched: bool = False,
    max_rejected_fraction: float = DEFAULT_MAX_REJECTED_FRACTION,
):
    """遍历原始数据根目录，为每个完整场景生成对齐索引。"""
    print("start!")
    for scene_dir in sorted(os.listdir(directory)):
        scene_path = os.path.join(directory, scene_dir)
        if not os.path.isdir(scene_path):
            continue
        radar_path = os.path.join(scene_path, "radar_pcl")
        lidar_path = os.path.join(scene_path, "livox_lidar")
        if not os.path.isdir(radar_path) or not os.path.isdir(lidar_path):
            print(f"Skipping {scene_dir}: data directories not found")
            continue
        print(f"Processing scene: {scene_dir}")
        records = generate_scene_indices(
            scene_path,
            radar_lidar_max_delta=radar_lidar_max_delta,
            skip_unmatched=skip_unmatched,
            max_rejected_fraction=max_rejected_fraction,
        )
        rejected_path = os.path.join(scene_path, REJECTED_FILENAME)
        with open(rejected_path, "r", encoding="utf-8", newline="") as handle:
            rejected_count = sum(1 for _ in csv.DictReader(handle))
        print(
            f"Generated {len(records)} aligned pairs for {scene_dir}; "
            f"rejected {rejected_count} candidates"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 NTU4DRadLM Radar/LiDAR 时间对齐索引")
    parser.add_argument(
        "--directory",
        default="./Data/NTU4DRadLM_Raw",
        help="Raw 数据集根目录",
    )
    parser.add_argument(
        "--radar_lidar_max_delta",
        type=float,
        default=DEFAULT_RADAR_LIDAR_MAX_DELTA,
        help="Radar-LiDAR 最近邻最大时间差（秒）",
    )
    parser.add_argument(
        "--skip_unmatched",
        action="store_true",
        help="记录并跳过少量超限候选；默认仍是任意超限即失败",
    )
    parser.add_argument(
        "--max_rejected_fraction",
        type=float,
        default=DEFAULT_MAX_REJECTED_FRACTION,
        help="启用 --skip_unmatched 时允许的最大超限候选比例",
    )
    args = parser.parse_args()
    generate_new_files(
        args.directory,
        radar_lidar_max_delta=args.radar_lidar_max_delta,
        skip_unmatched=args.skip_unmatched,
        max_rejected_fraction=args.max_rejected_fraction,
    )
