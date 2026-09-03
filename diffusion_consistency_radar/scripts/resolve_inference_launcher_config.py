#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""严格解析正式推理 launcher 使用的少量 YAML 配置。"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping

import yaml


_INFERENCE_KEYS = frozenset({"max_infer_files", "empty_fallback_topk"})


def load_yaml_mapping(path: str) -> dict:
    """读取 YAML 根对象；语法、I/O 或根类型错误均直接失败。"""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML 配置解析失败: {path}: {exc}") from exc
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError(f"YAML 配置根必须是对象: {path}")
    return dict(value)


def _nonnegative_int(value, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"inference.{name} 必须是非负整数，实际为 {value!r}")
    return value


def resolve_inference_defaults(path: str) -> dict:
    """解析 launcher 消费的 inference 字段并拒绝未知拼写。"""
    config = load_yaml_mapping(path)
    inference = config.get("inference", {})
    if inference is None:
        inference = {}
    if not isinstance(inference, Mapping):
        raise ValueError("inference 必须是对象")
    unknown = sorted(set(inference) - _INFERENCE_KEYS)
    if unknown:
        raise ValueError(f"inference 含未知字段: {unknown}")
    return {
        "max_infer_files": _nonnegative_int(
            inference.get("max_infer_files", 0),
            "max_infer_files",
        ),
        "empty_fallback_topk": _nonnegative_int(
            inference.get("empty_fallback_topk", 0),
            "empty_fallback_topk",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="解析正式推理 launcher 配置")
    subparsers = parser.add_subparsers(dest="command", required=True)
    defaults = subparsers.add_parser("defaults")
    defaults.add_argument("--config", required=True)
    args = parser.parse_args()

    try:
        resolved = resolve_inference_defaults(args.config)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(resolved["max_infer_files"])
    print(resolved["empty_fallback_topk"])


if __name__ == "__main__":
    main()
