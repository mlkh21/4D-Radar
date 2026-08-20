# -*- coding: utf-8 -*-
"""NTU4DRadLM 多传感器时间戳对齐协议。

本模块只依赖 Python 标准库，供 ROS 解包、Radar/LiDAR 索引和红外匹配
共同使用。统一规则是优先采用消息头时间戳，最近邻匹配必须显式检查
时间容差，并返回实际的时间差供调用方记录。
"""

import bisect
import math
from typing import Iterable, Optional, Sequence, Tuple


def _finite_positive(value: object) -> Optional[float]:
    """将候选时间戳转换为有限正数；无效值返回 ``None``。"""
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(candidate) or candidate <= 0.0:
        return None
    return candidate


def preferred_message_timestamp(msg: object, receipt_timestamp: object) -> Tuple[float, str]:
    """返回消息头时间戳，缺失或无效时回退到 bag 接收时间。

    返回值的第二项用于审计实际采用的时间源：``"header"`` 或
    ``"receipt"``。ROS ``Time`` 同时支持 ``to_sec`` 和 ``secs/nsecs``
    两种常见接口，测试桩和真实消息均可使用。
    """
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    header_timestamp = None
    if stamp is not None:
        to_sec = getattr(stamp, "to_sec", None)
        if callable(to_sec):
            try:
                header_timestamp = _finite_positive(to_sec())
            except (TypeError, ValueError, OverflowError):
                header_timestamp = None
        if header_timestamp is None:
            secs = getattr(stamp, "secs", None)
            nsecs = getattr(stamp, "nsecs", 0)
            if secs is not None:
                try:
                    header_timestamp = _finite_positive(float(secs) + float(nsecs) * 1e-9)
                except (TypeError, ValueError, OverflowError):
                    header_timestamp = None

    if header_timestamp is not None:
        return header_timestamp, "header"

    receipt = _finite_positive(receipt_timestamp)
    if receipt is None:
        raise ValueError("消息头和 bag 接收时间均不是有限正时间戳")
    return receipt, "receipt"


def _validated_timestamps(timestamps: Iterable[object]) -> Sequence[float]:
    """验证最近邻查找所需的有限、严格递增时间戳序列。"""
    values = [float(value) for value in timestamps]
    if not values:
        raise ValueError("时间戳序列不能为空")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("时间戳序列必须全部为有限数")
    if any(after <= before for before, after in zip(values, values[1:])):
        raise ValueError("时间戳序列必须严格递增")
    return values


def nearest_timestamp_match(
    timestamps: Iterable[object], target: object, max_delta: Optional[object] = None
) -> Tuple[int, float]:
    """查找最近时间戳并返回 ``(索引, 绝对时间差)``。

    当 ``max_delta`` 不为 ``None`` 且最近帧超出容差时立即抛出
    ``ValueError``，防止调用方静默使用错误模态帧。距离相等时选择较早
    的时间戳，以保持和原有 ``find_nearest_index`` 的行为一致。
    """
    values = _validated_timestamps(timestamps)
    try:
        target_value = float(target)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("目标时间戳必须是有限数") from exc
    if not math.isfinite(target_value):
        raise ValueError("目标时间戳必须是有限数")

    if max_delta is not None:
        try:
            allowed = float(max_delta)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("时间容差必须是有限非负数") from exc
        if not math.isfinite(allowed) or allowed < 0.0:
            raise ValueError("时间容差必须是有限非负数")
    else:
        allowed = None

    insertion = bisect.bisect_left(values, target_value)
    if insertion == 0:
        index = 0
    elif insertion == len(values):
        index = len(values) - 1
    else:
        before_delta = target_value - values[insertion - 1]
        after_delta = values[insertion] - target_value
        index = insertion if after_delta < before_delta else insertion - 1

    delta = abs(values[index] - target_value)
    if allowed is not None and delta > allowed:
        raise ValueError(
            f"最近时间戳偏差 {delta:.9f}s 超过时间容差 {allowed:.9f}s "
            f"(target={target_value:.9f}, matched={values[index]:.9f})"
        )
    return index, delta

