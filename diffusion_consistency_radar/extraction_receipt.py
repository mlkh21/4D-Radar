# -*- coding: utf-8 -*-
"""文件功能：定义 rosbag 解包逐场景成功计数、失败帧和关键模态收据。"""

import hashlib
import json
import math
import os
from typing import Dict, Mapping, Sequence, Tuple


EXTRACTION_RECEIPT_PROTOCOL = "rosbag_extraction_receipt_v1"
CRITICAL_EXTRACTION_TOPICS = (
    "radar_pcl",
    "livox/lidar",
    "thermal_cam/thermal_image/compressed",
)
_STATUSES = {"in_progress", "complete", "failed"}


def _critical_summary(receipt: Mapping[str, object]) -> Dict[str, object]:
    successes = receipt["topic_success_counts"]
    failures = receipt["failures"]
    result = {}
    for topic in CRITICAL_EXTRACTION_TOPICS:
        success_count = int(successes.get(topic, 0))
        failure_count = sum(
            1
            for item in failures
            if item.get("topic") == topic and item.get("critical") is True
        )
        result[topic] = {
            "success_count": success_count,
            "failure_count": failure_count,
            "status": (
                "failed"
                if failure_count > 0
                else "present"
                if success_count > 0
                else "missing"
            ),
        }
    return result


def new_extraction_receipt(scene: str, expected_bags: Sequence[str]) -> Dict[str, object]:
    """创建尚未完成的场景收据。"""
    if not isinstance(scene, str) or not scene:
        raise ValueError("extraction receipt scene 必须是非空字符串")
    bags = [str(value) for value in expected_bags]
    if not bags or any(not value for value in bags) or len(set(bags)) != len(bags):
        raise ValueError("extraction receipt expected_bags 必须非空且无重复")
    receipt = {
        "protocol": EXTRACTION_RECEIPT_PROTOCOL,
        "scene": scene,
        "status": "in_progress",
        "expected_bags": bags,
        "processed_bags": [],
        "topic_success_counts": {},
        "failures": [],
        "critical_modalities": {},
    }
    receipt["critical_modalities"] = _critical_summary(receipt)
    return receipt


def record_extraction_success(receipt: Dict[str, object], topic: str) -> None:
    """累加一次成功落盘的消息。"""
    counts = receipt["topic_success_counts"]
    counts[topic] = int(counts.get(topic, 0)) + 1
    receipt["critical_modalities"] = _critical_summary(receipt)


def record_extraction_failure(
    receipt: Dict[str, object],
    *,
    bag: str,
    topic: str,
    timestamp,
    timestamp_source,
    error: BaseException,
    critical: bool,
) -> None:
    """记录一条失败；关键失败立即把场景标为 failed。"""
    receipt["failures"].append(
        {
            "bag": str(bag),
            "topic": str(topic),
            "timestamp": None if timestamp is None else float(timestamp),
            "timestamp_source": timestamp_source,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "critical": bool(critical),
        }
    )
    if critical:
        receipt["status"] = "failed"
    receipt["critical_modalities"] = _critical_summary(receipt)


def mark_bag_processed(receipt: Dict[str, object], bag: str) -> None:
    """在分卷全部消息遍历并保存后登记完成。"""
    bag_name = str(bag)
    if bag_name not in receipt["expected_bags"]:
        raise ValueError("processed bag 不在 expected_bags 中")
    if bag_name in receipt["processed_bags"]:
        raise ValueError("processed bag 不允许重复")
    receipt["processed_bags"].append(bag_name)


def finalize_extraction_receipt(receipt: Dict[str, object]) -> bool:
    """完成场景门禁，返回关键模态是否全部成功。"""
    receipt["critical_modalities"] = _critical_summary(receipt)
    all_bags = receipt["processed_bags"] == receipt["expected_bags"]
    all_critical = all(
        item["status"] == "present"
        for item in receipt["critical_modalities"].values()
    )
    no_critical_failure = not any(
        item.get("critical") is True for item in receipt["failures"]
    )
    passed = all_bags and all_critical and no_critical_failure
    receipt["status"] = "complete" if passed else "failed"
    return passed


def validate_extraction_receipt(
    value: Mapping[str, object],
    *,
    require_complete: bool = False,
) -> Dict[str, object]:
    """验证收据结构及其派生 critical summary。"""
    expected_keys = {
        "protocol",
        "scene",
        "status",
        "expected_bags",
        "processed_bags",
        "topic_success_counts",
        "failures",
        "critical_modalities",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("extraction receipt 顶层字段不匹配")
    if value.get("protocol") != EXTRACTION_RECEIPT_PROTOCOL:
        raise ValueError("extraction receipt protocol 不匹配")
    if value.get("status") not in _STATUSES:
        raise ValueError("extraction receipt status 不支持")
    if require_complete and value.get("status") != "complete":
        raise ValueError("extraction receipt 未通过关键模态完整性门禁")
    expected_bags = value.get("expected_bags")
    processed_bags = value.get("processed_bags")
    if (
        not isinstance(expected_bags, list)
        or not expected_bags
        or len(set(expected_bags)) != len(expected_bags)
        or not isinstance(processed_bags, list)
        or processed_bags != expected_bags[: len(processed_bags)]
    ):
        raise ValueError("extraction receipt bag 顺序/覆盖不匹配")
    counts = value.get("topic_success_counts")
    if not isinstance(counts, Mapping) or any(
        not isinstance(topic, str)
        or type(count) is not int
        or count < 0
        for topic, count in counts.items()
    ):
        raise ValueError("extraction receipt topic success count 无效")
    failures = value.get("failures")
    if not isinstance(failures, list):
        raise ValueError("extraction receipt failures 必须是列表")
    failure_keys = {
        "bag",
        "topic",
        "timestamp",
        "timestamp_source",
        "error_type",
        "error_message",
        "critical",
    }
    for item in failures:
        if not isinstance(item, Mapping) or set(item) != failure_keys:
            raise ValueError("extraction receipt failure 字段不匹配")
        if type(item.get("critical")) is not bool:
            raise ValueError("extraction receipt failure critical 必须是 bool")
        timestamp = item.get("timestamp")
        if timestamp is not None and (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
        ):
            raise ValueError("extraction receipt failure timestamp 必须是有限数或 null")
        if not isinstance(item.get("topic"), str) or not item.get("topic"):
            raise ValueError("extraction receipt failure topic 必须非空")
        if not isinstance(item.get("error_type"), str) or not item.get("error_type"):
            raise ValueError("extraction receipt failure error_type 必须非空")
    expected_critical = _critical_summary(value)
    if value.get("critical_modalities") != expected_critical:
        raise ValueError("extraction receipt critical summary 不一致")
    if value.get("status") == "complete":
        if processed_bags != expected_bags or any(
            item["status"] != "present" for item in expected_critical.values()
        ):
            raise ValueError("complete extraction receipt 实际并不完整")
    return json.loads(json.dumps(value, ensure_ascii=False))


def write_extraction_receipt_atomic(path: str, receipt: Mapping[str, object]) -> None:
    """验证后原子发布收据。"""
    output_path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    validated = validate_extraction_receipt(receipt)
    temp_path = output_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(validated, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, output_path)


def load_extraction_receipt_artifact(
    path: str,
    *,
    require_complete: bool = False,
) -> Tuple[Dict[str, object], str]:
    """加载场景收据并返回内容 SHA-256。"""
    input_path = os.path.abspath(os.fspath(path))
    if os.path.islink(input_path) or not os.path.isfile(input_path):
        raise ValueError("extraction receipt 必须是普通 JSON 文件")
    with open(input_path, "rb") as handle:
        payload = handle.read()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("extraction receipt 不是合法 UTF-8 JSON") from exc
    return (
        validate_extraction_receipt(value, require_complete=require_complete),
        hashlib.sha256(payload).hexdigest(),
    )


__all__ = [
    "CRITICAL_EXTRACTION_TOPICS",
    "EXTRACTION_RECEIPT_PROTOCOL",
    "finalize_extraction_receipt",
    "load_extraction_receipt_artifact",
    "mark_bag_processed",
    "new_extraction_receipt",
    "record_extraction_failure",
    "record_extraction_success",
    "validate_extraction_receipt",
    "write_extraction_receipt_atomic",
]
