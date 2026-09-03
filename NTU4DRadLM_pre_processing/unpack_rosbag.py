# -*- coding: utf-8 -*-

import rosbag
import os
import argparse
import csv
import json
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import glob
import re
import sys
import cv2
from tqdm import tqdm  # ──► 新增引用：工业级流式进度条组件

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion_consistency_radar.extraction_receipt import (
    CRITICAL_EXTRACTION_TOPICS,
    finalize_extraction_receipt,
    mark_bag_processed,
    new_extraction_receipt,
    record_extraction_failure,
    record_extraction_success,
    write_extraction_receipt_atomic,
)

if __package__:
    # 作为项目模块调用时使用相对路径，避免同名脚本遮蔽命名空间包。
    from .timestamp_alignment import preferred_message_timestamp
else:
    # 直接执行脚本时，脚本所在目录位于 sys.path 中。
    from timestamp_alignment import preferred_message_timestamp  # type: ignore[no-redef]

"""NTU4DRadLM 解包脚本。

NOTE: 该脚本会把多分卷 bag 按场景归并到统一目录，并按 topic 拆分输出。
"""

# NOTE: 优先尝试导入 Livox 自定义消息类型；失败时回退到通用点云解析。
try:
    # 根据 ws_livox 工作空间路径，添加正确的 Python 包路径
    sys.path.append('/home/ps/zxj_workspace/src/ws_livox/devel/lib/python3/dist-packages')
    from livox_ros_driver.msg import CustomMsg
    LIVOX_AVAILABLE = True
except ImportError:
    LIVOX_AVAILABLE = False
    print("[Warning] 'livox_ros_driver' not found. Will attempt generic parsing for LiDAR.")

# NOTE: 白名单主题控制导出范围，避免无关 topic 占用磁盘。
ALLOWED_TOPICS = {
    "livox/lidar",
    "radar_pcl",
    "thermal_cam/thermal_image/compressed",
    "ublox/fix",
    "ublox/fix_velocity",
    "vectornav/imu",
}

# PointCloud2 导出协议：无论原消息字段是否完整，保存文件都固定为这五列。
POINTCLOUD_COLUMNS = ("x", "y", "z", "intensity", "doppler")
POINTCLOUD_SCHEMA_FILENAME = "pointcloud_schema.json"
_INTENSITY_FIELD_ALIASES = ("intensity", "reflectivity", "power", "rcs", "snr")
_DOPPLER_FIELD_ALIASES = ("velocity", "doppler", "v_r", "radial_velocity")


def _resolve_pointcloud_field(field_names, aliases):
    """按不区分大小写的别名查找消息中的实际字段名。"""
    by_lower_name = {str(name).lower(): name for name in field_names}
    for alias in aliases:
        if alias in by_lower_name:
            return by_lower_name[alias]
    return None


def _write_pointcloud_schema(save_dir, schema):
    """原子写入点云列协议，避免中断时留下半截 JSON。"""
    schema_path = os.path.join(save_dir, POINTCLOUD_SCHEMA_FILENAME)
    if os.path.isfile(schema_path) and not os.path.islink(schema_path):
        with open(schema_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        stable_existing = {
            key: value for key, value in existing.items() if key != "shape"
        }
        stable_current = {
            key: value for key, value in schema.items() if key != "shape"
        }
        if stable_existing != stable_current:
            raise ValueError("同一输出目录的 PointCloud 字段 layout 发生漂移")
        return
    if os.path.lexists(schema_path):
        raise ValueError("pointcloud schema 路径必须是普通文件或不存在")
    temp_path = schema_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(schema, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, schema_path)


def _build_pointcloud_layout_schema(
    source_field_names,
    field_mapping,
    *,
    topic,
    message_type,
    header_frame_id,
):
    """构建只证明列布局、不声明物理单位或符号的解包收据。"""
    selected_fields = [
        field_mapping[name]
        for name in POINTCLOUD_COLUMNS
        if field_mapping[name] is not None
    ]
    return {
        "schema_version": 2,
        "storage_format": "npy",
        "ros_transport": {
            "topic": str(topic),
            "message_type": str(message_type),
            "header_frame_id": str(header_frame_id),
        },
        "columns": list(POINTCLOUD_COLUMNS),
        "column_indices": {
            name: index for index, name in enumerate(POINTCLOUD_COLUMNS)
        },
        "source_fields": list(source_field_names),
        "selected_fields": selected_fields,
        "field_mapping": field_mapping,
        "missing_fields": [
            name for name in ("intensity", "doppler")
            if field_mapping[name] is None
        ],
        "physical_semantics_status": "unverified_layout_only",
        "missing_value_encoding": "nan",
    }


def _read_pointcloud2_fixed_columns(msg, *, ros_transport):
    """按字段名读取 PointCloud2，并构造固定的 ``[x,y,z,intensity,doppler]``。"""
    source_field_names = [str(field.name) for field in msg.fields]
    field_mapping = {
        "x": _resolve_pointcloud_field(source_field_names, ("x",)),
        "y": _resolve_pointcloud_field(source_field_names, ("y",)),
        "z": _resolve_pointcloud_field(source_field_names, ("z",)),
        "intensity": _resolve_pointcloud_field(
            source_field_names, _INTENSITY_FIELD_ALIASES
        ),
        "doppler": _resolve_pointcloud_field(
            source_field_names, _DOPPLER_FIELD_ALIASES
        ),
    }
    missing_coordinates = [
        name for name in ("x", "y", "z") if field_mapping[name] is None
    ]
    if missing_coordinates:
        raise ValueError(
            "PointCloud2 缺少必需坐标字段: " + ", ".join(missing_coordinates)
        )

    selected_fields = [
        field_mapping[name]
        for name in POINTCLOUD_COLUMNS
        if field_mapping[name] is not None
    ]
    selected_columns = [
        name for name in POINTCLOUD_COLUMNS if field_mapping[name] is not None
    ]
    selected_indices = {name: index for index, name in enumerate(selected_columns)}

    points = []
    for raw_point in pc2.read_points(
        msg, field_names=selected_fields, skip_nans=False
    ):
        values = list(raw_point)
        if len(values) != len(selected_fields):
            raise ValueError(
                "PointCloud2.read_points 返回列数与请求字段不一致: "
                f"expected={len(selected_fields)}, got={len(values)}"
            )
        # XYZ 必须存在；两个可选物理字段缺失时写 NaN，使下游 finite-count
        # 能区分“真实零测量”和“消息根本没有该字段”。
        point = np.full(len(POINTCLOUD_COLUMNS), np.nan, dtype=np.float32)
        for column_index, column_name in enumerate(POINTCLOUD_COLUMNS):
            source_name = field_mapping[column_name]
            if source_name is not None:
                point[column_index] = float(values[selected_indices[column_name]])
        points.append(point.tolist())

    schema = _build_pointcloud_layout_schema(
        source_field_names,
        field_mapping,
        **ros_transport,
    )
    return points, schema


def _pointcloud_transport(msg, *, topic, message_type):
    """提取参与跨帧、跨 bag 漂移检查的 ROS transport 身份。"""
    declared_type = message_type or getattr(msg, "_type", type(msg).__name__)
    header = getattr(msg, "header", None)
    frame_id = getattr(header, "frame_id", "") if header is not None else ""
    return {
        "topic": str(topic or ""),
        "message_type": str(declared_type),
        "header_frame_id": str(frame_id or ""),
    }


def _topic_key(topic_name: str) -> str:
    return topic_name.strip('/')


def _write_csv_records(path, records):
    """使用标准库写入动态字段记录，避免解包入口隐式依赖 pandas。"""
    fieldnames = []
    for record in records:
        for name in record:
            if name not in fieldnames:
                fieldnames.append(name)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

def get_scene_name(bag_path):
    """
    从文件名解析场景名称，用于归并分卷。
    规则：提取第一个下划线前的部分。
    例如:
    loop1_2022-06-03_0.bag -> loop1
    carpark_2022-06-03.bag -> carpark
    """
    filename = os.path.basename(bag_path)
    # HACK: 这里依赖文件命名规则（第一个下划线前即场景名）。
    # TODO: 后续可改为通过目录结构或配置文件确定场景名，减少命名耦合。
    scene_name = filename.split('_')[0]
    return scene_name

def save_pointcloud(
    msg,
    save_dir,
    timestamp,
    *,
    topic="",
    message_type=None,
):
    """
    保存点云为 .npy 文件，保留所有关键属性。
    Livox: [x, y, z, reflectivity]
    Radar (PointCloud v1/PointCloud2): [x, y, z, intensity, doppler]
    """
    points_list = []
    schema = None
    ros_transport = _pointcloud_transport(
        msg,
        topic=topic,
        message_type=message_type,
    )

    # NOTE: 1) Livox CustomMsg
    # 结构: x, y, z, reflectivity, tag, line
    if 'CustomMsg' in str(type(msg)) or 'livox_ros_driver' in str(type(msg)): # 只要名字匹配就尝试处理,不依赖 LIVOX_AVAILABLE
        for p in msg.points:
            # 保存 x, y, z, reflectivity
            points_list.append([p.x, p.y, p.z, float(p.reflectivity)])

    # NOTE: 2) 标准 PointCloud2
    elif hasattr(msg, 'width'): # Duck typing for PointCloud2
        points_list, schema = _read_pointcloud2_fixed_columns(
            msg,
            ros_transport=ros_transport,
        )

    # NOTE: 3) 标准 PointCloud (v1) - 常见于 4D Radar
    # 结构: points[], channels[name, values[]]
    elif hasattr(msg, 'points') and msg.points and hasattr(msg.points[0], 'x'):
        channel_map = {
            str(channel.name): channel.values for channel in msg.channels
        }
        channel_names = list(channel_map)
        intensity_field = _resolve_pointcloud_field(
            channel_names,
            _INTENSITY_FIELD_ALIASES,
        )
        doppler_field = _resolve_pointcloud_field(
            channel_names,
            _DOPPLER_FIELD_ALIASES,
        )
        intensities = (
            channel_map[intensity_field]
            if intensity_field is not None
            else None
        )
        velocities = (
            channel_map[doppler_field]
            if doppler_field is not None
            else None
        )
        schema = _build_pointcloud_layout_schema(
            ["x", "y", "z"] + channel_names,
            {
                "x": "x",
                "y": "y",
                "z": "z",
                "intensity": intensity_field,
                "doppler": doppler_field,
            },
            **ros_transport,
        )

        for i, point in enumerate(msg.points):
            inten = intensities[i] if intensities is not None else np.nan
            vel = velocities[i] if velocities is not None else np.nan
            points_list.append(
                [point.x, point.y, point.z, float(inten), float(vel)]
            )

    else:
        raise ValueError(f"Unknown pointcloud type: {type(msg)} at {timestamp}")

    if not points_list:
        raise ValueError(f"点云帧为空: {timestamp}")

    pc_np = np.array(points_list, dtype=np.float32)

    if schema is not None:
        schema["shape"] = [int(size) for size in pc_np.shape]
        schema["dtype"] = str(pc_np.dtype)
        _write_pointcloud_schema(save_dir, schema)

    # layout 稳定性通过后再发布帧文件，避免漂移帧留下无收据 NPY。
    filename_npy = os.path.join(save_dir, f"{timestamp:.6f}.npy")
    np.save(filename_npy, pc_np)
    return filename_npy

def save_compressed_image(msg, save_dir, timestamp):
    data_np = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(data_np, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to decode image at {timestamp}")
    filename = os.path.join(save_dir, f"{timestamp:.6f}.png")
    if not cv2.imwrite(filename, img):
        raise OSError(f"Failed to write image at {timestamp}: {filename}")
    return filename

def _message_to_csv_row(msg, timestamp, timestamp_source):
    """把非点云 ROS 消息转换为标准库 CSV 可写的扁平记录。"""
    data_row = {"timestamp": timestamp, "timestamp_source": timestamp_source}
    if hasattr(msg, "header"):
        data_row["seq"] = msg.header.seq
    for attr in getattr(msg, "__slots__", dir(msg)):
        if attr.startswith("_") or attr == "header":
            continue
        try:
            value = getattr(msg, attr)
            if all(hasattr(value, key) for key in ("x", "y", "z")):
                data_row[f"{attr}_x"] = value.x
                data_row[f"{attr}_y"] = value.y
                data_row[f"{attr}_z"] = value.z
                if hasattr(value, "w"):
                    data_row[f"{attr}_w"] = value.w
            elif isinstance(value, (list, tuple, np.ndarray)):
                data_row[attr] = str(list(value))
            elif isinstance(value, (int, float, str, bool)):
                data_row[attr] = value
            elif not callable(value):
                data_row[attr] = str(value).replace("\n", " ")
        except Exception:
            # 单个可选属性不可序列化时不影响整条非关键消息。
            continue
    return data_row


def process_ntu_dataset(input_root, output_root):
    """按场景解包 bag，并以关键模态逐帧收据 fail-closed。"""
    search_pattern = os.path.join(input_root, "**", "*.bag")
    bag_files = sorted(glob.glob(search_pattern, recursive=True))
    if not bag_files:
        print(f"No .bag files found in {input_root}")
        return

    expected_bags_by_scene = {}
    for bag_path in bag_files:
        expected_bags_by_scene.setdefault(get_scene_name(bag_path), []).append(
            os.path.basename(bag_path)
        )
    receipts = {
        scene: new_extraction_receipt(scene, expected_bags)
        for scene, expected_bags in expected_bags_by_scene.items()
    }

    print(f"Found {len(bag_files)} bag files. Starting processing...")
    print(f"Output Root: {output_root}\n")
    outer_pbar = tqdm(bag_files, desc="总体包解包进度", unit="bag")

    for bag_path in outer_pbar:
        scene_name = get_scene_name(bag_path)
        scene_output_dir = os.path.join(output_root, scene_name)
        receipt_path = os.path.join(scene_output_dir, "extraction_receipt.json")
        receipt = receipts[scene_name]
        bag_filename = os.path.basename(bag_path)
        outer_pbar.set_description(f"正在处理: {bag_filename}")

        try:
            bag = rosbag.Bag(bag_path)
        except Exception as exc:
            record_extraction_failure(
                receipt,
                bag=bag_filename,
                topic="__bag__",
                timestamp=None,
                timestamp_source=None,
                error=exc,
                critical=True,
            )
            write_extraction_receipt_atomic(receipt_path, receipt)
            raise RuntimeError(
                f"无法打开 rosbag 分卷，拒绝继续生成不完整场景: {bag_path}"
            ) from exc

        csv_buffers = {}
        try:
            info = bag.get_type_and_topic_info()
            pbar_desc = f"  └─ 帧提取流 ({bag_filename[:15]}...)"
            inner_pbar = tqdm(
                bag.read_messages(),
                desc=pbar_desc,
                unit="msg",
                leave=False,
            )
            for topic, msg, bag_time in inner_pbar:
                if topic not in info.topics:
                    continue
                topic_key = _topic_key(topic)
                if topic_key not in ALLOWED_TOPICS:
                    continue

                timestamp = None
                timestamp_source = None
                try:
                    receipt_timestamp = bag_time.to_sec()
                    timestamp, timestamp_source = preferred_message_timestamp(
                        msg,
                        receipt_timestamp,
                    )
                    msg_type = info.topics[topic].msg_type
                    topic_dir = os.path.join(
                        scene_output_dir,
                        topic.strip("/").replace("/", "_"),
                    )
                    if "PointCloud" in msg_type or "CustomMsg" in msg_type:
                        os.makedirs(topic_dir, exist_ok=True)
                        save_pointcloud(
                            msg,
                            topic_dir,
                            timestamp,
                            topic=topic,
                            message_type=msg_type,
                        )
                        record_extraction_success(receipt, topic_key)
                    elif "CompressedImage" in msg_type:
                        os.makedirs(topic_dir, exist_ok=True)
                        save_compressed_image(msg, topic_dir, timestamp)
                        record_extraction_success(receipt, topic_key)
                    else:
                        csv_buffers.setdefault(topic, []).append(
                            _message_to_csv_row(
                                msg,
                                timestamp,
                                timestamp_source,
                            )
                        )
                except Exception as exc:
                    critical = topic_key in CRITICAL_EXTRACTION_TOPICS
                    record_extraction_failure(
                        receipt,
                        bag=bag_filename,
                        topic=topic_key,
                        timestamp=timestamp,
                        timestamp_source=timestamp_source,
                        error=exc,
                        critical=critical,
                    )
                    write_extraction_receipt_atomic(receipt_path, receipt)
                    if critical:
                        raise RuntimeError(
                            "关键模态帧解包失败，拒绝继续生成不完整场景: "
                            f"scene={scene_name}, topic={topic_key}, bag={bag_filename}"
                        ) from exc
        except Exception as exc:
            if receipt["status"] != "failed":
                record_extraction_failure(
                    receipt,
                    bag=bag_filename,
                    topic="__bag__",
                    timestamp=None,
                    timestamp_source=None,
                    error=exc,
                    critical=True,
                )
                write_extraction_receipt_atomic(receipt_path, receipt)
            raise
        finally:
            try:
                bag.close()
            except Exception as exc:
                record_extraction_failure(
                    receipt,
                    bag=bag_filename,
                    topic="__bag__",
                    timestamp=None,
                    timestamp_source=None,
                    error=exc,
                    critical=True,
                )
                write_extraction_receipt_atomic(receipt_path, receipt)
                raise RuntimeError(
                    f"关闭 rosbag 分卷失败，场景完整性未知: {bag_path}"
                ) from exc

        bag_base_name = os.path.splitext(bag_filename)[0]
        for topic, data_list in csv_buffers.items():
            if not data_list:
                continue
            topic_key = _topic_key(topic)
            try:
                topic_dir = os.path.join(
                    scene_output_dir,
                    topic.strip("/").replace("/", "_"),
                )
                os.makedirs(topic_dir, exist_ok=True)
                _write_csv_records(
                    os.path.join(topic_dir, f"data_{bag_base_name}.csv"),
                    data_list,
                )
                for _ in data_list:
                    record_extraction_success(receipt, topic_key)
            except Exception as exc:
                record_extraction_failure(
                    receipt,
                    bag=bag_filename,
                    topic=topic_key,
                    timestamp=None,
                    timestamp_source=None,
                    error=exc,
                    critical=False,
                )

        mark_bag_processed(receipt, bag_filename)
        write_extraction_receipt_atomic(receipt_path, receipt)

    failed_scenes = []
    for scene_name, receipt in receipts.items():
        if not finalize_extraction_receipt(receipt):
            failed_scenes.append(scene_name)
        write_extraction_receipt_atomic(
            os.path.join(output_root, scene_name, "extraction_receipt.json"),
            receipt,
        )
    if failed_scenes:
        raise RuntimeError(
            "关键模态解包不完整，拒绝报告成功: " + ", ".join(failed_scenes)
        )
    print("\nAll extraction tasks completed successfully!")

if __name__ == "__main__":
    # 在这里修改默认路径，或者通过命令行参数传入
    default_input = "./Data/NTU4DRadLM"
    default_output = "./Data/NTU4DRadLM_Raw"

    parser = argparse.ArgumentParser(description="Unpack NTU4DRadLM Bags with Double Progress Bars")
    parser.add_argument("--input", default=default_input, help="Input dataset root directory")
    parser.add_argument("--output", default=default_output, help="Output directory")

    args = parser.parse_args()

    process_ntu_dataset(args.input, args.output)
