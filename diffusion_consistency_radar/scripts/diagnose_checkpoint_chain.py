#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立诊断正式 VAE/LDM/CD checkpoint 链，不运行数据推理或训练。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Dict

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(PROJECT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from diffusion_consistency_radar.checkpoint_chain import (
    CheckpointChainError,
    checkpoint_state_dict,
    safe_torch_load,
    validate_formal_checkpoint_chain,
)


def _ensure_empty_report_dir(path: str) -> None:
    """报告目录只能是不存在或空目录，避免覆盖历史诊断。"""
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise ValueError(f"报告路径不是目录: {path}")
        if os.listdir(path):
            raise ValueError(f"报告目录必须为空: {path}")


def _construct_and_load_chain(
    vae_path: str,
    ldm_path: str,
    cd_path: str,
    report: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    """按 checkpoint 自描述配置在指定设备构建并严格加载三阶段模型。"""
    from diffusion_consistency_radar.cm.vae_3d import build_vae_from_checkpoint
    from diffusion_consistency_radar.scripts.inference import (
        build_inference_model,
        resolve_generation_model_config,
    )

    target_size = tuple(int(value) for value in report["grid"]["target_size"])
    model_range = tuple(float(value) for value in report["grid"]["model_pc_range"])
    torch_device = torch.device(device)
    vae_checkpoint = safe_torch_load(vae_path)
    vae, metadata = build_vae_from_checkpoint(vae_checkpoint)
    vae = vae.to(torch_device).eval()
    loaded = {"vae": {"latent_dim": int(vae.latent_dim)}}

    for stage, path in (("ldm", ldm_path), ("cd", cd_path)):
        checkpoint = safe_torch_load(path)
        state_dict = checkpoint_state_dict(checkpoint)
        config = resolve_generation_model_config(
            checkpoint,
            fallback_latent_dim=int(vae.latent_dim),
        )
        model = build_inference_model(
            state_dict,
            torch_device,
            strict=True,
            fusion_voxel_shape=target_size,
            fusion_pc_range=model_range,
            latent_dim=int(config["latent_dim"]),
            model_config=config,
        ).eval()
        loaded[stage] = {
            "latent_dim": int(config["latent_dim"]),
            "is_multimodal": bool(getattr(model, "is_multimodal", False)),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
        del model
    del vae
    report["construct"] = {"device": str(torch_device), "stages_loaded": loaded}
    return report


def diagnose_checkpoint_chain(
    vae_ckpt: str,
    ldm_ckpt: str,
    cd_ckpt: str,
    report_dir: str = "",
    construct: bool = False,
    device: str = "cpu",
) -> Dict[str, Any]:
    """执行协议诊断；成功时可选写入新的 JSON 报告。"""
    report = validate_formal_checkpoint_chain(
        vae_ckpt,
        ldm_ckpt,
        cd_ckpt,
        require_multimodal=True,
    )
    if construct:
        report = _construct_and_load_chain(
            vae_ckpt,
            ldm_ckpt,
            cd_ckpt,
            report,
            device,
        )
    if report_dir:
        _ensure_empty_report_dir(report_dir)
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "checkpoint_chain.json")
        fd, temp_path = tempfile.mkstemp(
            prefix=".checkpoint_chain.", dir=report_dir, text=True
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, report_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="严格诊断正式 VAE/LDM/CD checkpoint 链（不运行推理）"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="校验并可选保存协议报告")
    validate.add_argument("--vae_ckpt", required=True)
    validate.add_argument("--ldm_ckpt", required=True)
    validate.add_argument("--cd_ckpt", required=True)
    validate.add_argument("--report_dir", default="")
    validate.add_argument(
        "--construct",
        action="store_true",
        help="在 CPU/指定设备构建并严格加载 VAE、LDM、CD，不执行 forward",
    )
    validate.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = diagnose_checkpoint_chain(
            args.vae_ckpt,
            args.ldm_ckpt,
            args.cd_ckpt,
            report_dir=args.report_dir,
            construct=args.construct,
            device=args.device,
        )
    except (CheckpointChainError, ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        "正式 checkpoint 链校验通过: "
        f"protocol={report['protocol']}, stages={','.join(report['stages'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
