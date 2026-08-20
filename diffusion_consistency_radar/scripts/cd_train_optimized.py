# -- coding: utf-8 --
"""
LDM 初始化的 EMA Consistency 训练脚本

改进点：
1. 清晰的训练语义 - LDM 仅初始化 CD/EMA，训练目标来自 CD EMA
2. 显存优化 - 梯度累积、检查点、混合精度
3. checkpoint 显式记录初始化来源和一致性目标
4. 模块化设计 - 易于理解和维护
"""

import sys
import os

# 直接执行本文件时同时暴露仓库根和包目录：前者支持
# diffusion_consistency_radar.*，后者兼容既有 cm.* 导入。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)
for _import_root in (_PACKAGE_ROOT, _PROJECT_ROOT):
    if _import_root not in sys.path:
        sys.path.insert(0, _import_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import argparse
import logging
import csv
import time
from typing import Any, Dict, Optional, Tuple
from tqdm import tqdm
import yaml

from diffusion_consistency_radar.cm.unet_optimized import OptimizedUNetModel
from diffusion_consistency_radar.cm.multimodal_fusion import (
    CompleteDualModalityPerceptionNet,
)
from diffusion_consistency_radar.cm.karras_diffusion import KarrasDenoiser
from diffusion_consistency_radar.cm.vae_3d import (
    VAE3D,
    build_vae_from_checkpoint,
    create_lightweight_vae_config,
    create_standard_vae_config,
    create_ultra_lightweight_vae_config,
)
from diffusion_consistency_radar.cm.dataset_loader import (
    NTU4DRadLM_VoxelDataset,
    collate_voxel_samples,
)
from diffusion_consistency_radar.checkpoint_chain import (
    FORMAL_CHECKPOINT_PROTOCOL,
    resolve_training_checkpoint_protocol,
    sha256_file,
)
from diffusion_consistency_radar.radar_normalization import (
    RadarNormalizationError,
    assert_checkpoint_radar_normalization,
    assert_same_radar_normalization,
    load_radar_normalization_artifact,
    radar_normalization_from_checkpoint,
)


def safe_torch_load(path, map_location):
    """兼容不同 PyTorch 版本的 checkpoint 加载逻辑。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # HACK: 低版本 PyTorch 不支持 weights_only，回退到兼容模式。
        return torch.load(path, map_location=map_location)
    except Exception as exc:
        # NOTE: 某些历史权重包含自定义对象，weights_only=True 会拒绝加载。
        msg = str(exc)
        if "Weights only load failed" in msg or "Unsupported global" in msg:
            return torch.load(path, map_location=map_location)
        raise


def checkpoint_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Return the actual model state dict from either raw or wrapped checkpoints."""
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def load_yaml_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_data_grid_config(data_config: Dict[str, Any]):
    """从 data 配置解析原始体素范围、模型裁剪范围和训练张量尺寸。"""
    source_pc_range = data_config.get("source_pc_range", data_config.get("pc_range", [0, -20, -6, 120, 20, 10]))
    model_pc_range = data_config.get("model_pc_range", data_config.get("pc_range", source_pc_range))
    target_size = data_config.get("target_size", data_config.get("voxel_shape", [32, 128, 128]))
    return (
        tuple(int(v) for v in target_size),
        tuple(float(v) for v in source_pc_range),
        tuple(float(v) for v in model_pc_range),
    )


def resolve_cd_radar_normalization(
    ldm_checkpoint,
    configured_spec,
    configured_sha256,
    *,
    data_grid_config,
    allow_legacy_radar_units=False,
):
    """在 CD 输出目录创建前校验配置 artifact 与教师 LDM 完全一致。"""
    if type(allow_legacy_radar_units) is not bool:
        raise RadarNormalizationError("allow_legacy_radar_units 必须是 bool")
    has_embedded = isinstance(ldm_checkpoint, dict) and (
        "radar_normalization" in ldm_checkpoint
        or "radar_normalization_sha256" in ldm_checkpoint
    )
    if allow_legacy_radar_units:
        if configured_spec is not None or configured_sha256 or has_embedded:
            raise RadarNormalizationError(
                "CD legacy 开关与正式 Radar normalization 不能同时启用"
            )
        return None, ""
    if configured_spec is None or not configured_sha256:
        raise RadarNormalizationError("正式 CD 缺少配置 Radar normalization")
    teacher_spec, teacher_sha256 = radar_normalization_from_checkpoint(
        ldm_checkpoint,
        target_size=data_grid_config.get("target_size"),
        source_pc_range=data_grid_config.get("source_pc_range"),
        model_pc_range=data_grid_config.get("model_pc_range"),
        context="CD teacher LDM checkpoint",
    )
    assert_same_radar_normalization(
        configured_spec,
        configured_sha256,
        teacher_spec,
        teacher_sha256,
        context="CD teacher/config",
    )
    return teacher_spec, teacher_sha256


def create_vae_from_config(config: Optional[Dict[str, Any]] = None) -> VAE3D:
    cfg = config or {}
    vae_cfg = cfg.get("vae", {}) if isinstance(cfg.get("vae", {}), dict) else {}
    config_type = vae_cfg.get("config_type", "ultra_lightweight")
    if config_type == "lightweight":
        model_cfg = create_lightweight_vae_config()
    elif config_type == "standard":
        model_cfg = create_standard_vae_config()
    else:
        model_cfg = create_ultra_lightweight_vae_config()
    return VAE3D(**model_cfg)


def build_cd_vae_from_checkpoint(
    checkpoint: Any,
    fallback_config_type: Optional[str] = None,
):
    """按共享 checkpoint 协议构建 CD 训练使用的 VAE。"""
    return build_vae_from_checkpoint(
        checkpoint,
        fallback_config_type=fallback_config_type,
    )


def has_multimodal_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Detect checkpoints saved from CompleteDualModalityPerceptionNet."""
    keys = tuple(state_dict.keys())
    prefixes = ("unet_3d.", "ir_extractor.", "projection_layer.", "fusion_conv.")
    return any(key.startswith(prefixes) for key in keys)


def create_legacy_unet(config: Optional[Dict[str, Any]] = None) -> OptimizedUNetModel:
    """Build the legacy latent denoiser used by historical CD/LDM checkpoints."""
    cfg = config or {}
    latent_dim = int(cfg.get("latent_dim", 4))
    return OptimizedUNetModel(
        image_size=32,
        in_channels=int(cfg.get("in_channels", cfg.get("legacy_in_channels", 2 * latent_dim))),
        model_channels=int(cfg.get("model_channels", 32)),
        out_channels=latent_dim,
        num_res_blocks=int(cfg.get("num_res_blocks", 1)),
        attention_resolutions=tuple(cfg.get("attention_resolutions", [])),
        channel_mult=tuple(cfg.get("channel_mult", [1, 2, 3])),
        use_checkpoint=bool(cfg.get("use_checkpoint", True)),
        attention_type="linear",
    )


def create_multimodal_cd_model(config: Optional[Dict[str, Any]] = None) -> CompleteDualModalityPerceptionNet:
    """Build the multimodal CD/LDM denoiser with a 16-channel latent backbone."""
    cfg = config or {}
    latent_dim = int(cfg.get("latent_dim", 4))
    backbone_in_channels = int(cfg.get("in_channels", max(16, 2 * latent_dim)))
    base_unet = OptimizedUNetModel(
        image_size=32,
        in_channels=backbone_in_channels,
        model_channels=int(cfg.get("model_channels", 32)),
        out_channels=latent_dim,
        num_res_blocks=int(cfg.get("num_res_blocks", 1)),
        attention_resolutions=tuple(cfg.get("attention_resolutions", [])),
        channel_mult=tuple(cfg.get("channel_mult", [1, 2, 3])),
        use_checkpoint=bool(cfg.get("use_checkpoint", True)),
        attention_type="linear",
    )
    fusion_voxel_shape = tuple(int(v) for v in cfg.get("fusion_voxel_shape", [32, 128, 128]))
    fusion_latent_shape = tuple(int(v) for v in cfg.get("fusion_latent_shape", fusion_voxel_shape))
    fusion_pc_range = tuple(float(v) for v in cfg.get("fusion_pc_range", [0, -20, -6, 120, 20, 10]))
    return CompleteDualModalityPerceptionNet(
        base_unet,
        voxel_shape=fusion_voxel_shape,
        pc_range=fusion_pc_range,
        downsample_to_latent=True,
        latent_shape=fusion_latent_shape,
        fused_channels=backbone_in_channels,
    )


def create_cd_model(multimodal: bool, config: Optional[Dict[str, Any]] = None) -> nn.Module:
    return create_multimodal_cd_model(config) if multimodal else create_legacy_unet(config)


def resolve_cd_generation_config(checkpoint, fallback_latent_dim: Optional[int]):
    """读取教师/CD checkpoint 模型配置，旧权重从卷积 shape 推导。"""
    checkpoint_dict = checkpoint if isinstance(checkpoint, dict) else {}
    state_dict = checkpoint_state_dict(checkpoint)
    config = dict(checkpoint_dict.get("model_config") or {})
    latent_dim = checkpoint_dict.get("latent_dim", config.get("latent_dim"))
    for prefix in ("", "unet_3d."):
        input_weight = state_dict.get(f"{prefix}input_blocks.0.0.weight")
        output_weight = state_dict.get(f"{prefix}out.2.weight")
        if "in_channels" not in config and torch.is_tensor(input_weight) and input_weight.ndim >= 2:
            config["in_channels"] = int(input_weight.shape[1])
        if latent_dim is None and torch.is_tensor(output_weight) and output_weight.ndim >= 1:
            latent_dim = int(output_weight.shape[0])
    if latent_dim is None:
        if fallback_latent_dim is None:
            raise ValueError("生成模型 checkpoint 无法推导 latent_dim，请显式提供 fallback")
        latent_dim = fallback_latent_dim
    config["latent_dim"] = int(latent_dim)
    config.setdefault("in_channels", 2 * int(latent_dim))
    config.setdefault("out_channels", int(latent_dim))
    return config


def has_multimodal_meta(meta: Optional[Dict[str, Any]]) -> bool:
    required = ("ir_img", "r_mat", "t_vec", "k_mat")
    return all(torch.is_tensor((meta or {}).get(key)) for key in required)


def encode_cd_training_latents(vae, target, condition, meta_dict):
    """正式多模态只编码 target；legacy batch 才构造 condition latent。"""
    z_target = vae.get_latent(target)
    z_cond = None
    if not has_multimodal_meta(meta_dict):
        z_cond = vae.get_latent(condition)
    return z_target, z_cond


def move_meta_to_device(meta: Optional[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in (meta or {}).items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def unpack_cd_batch(batch):
    """Support both legacy (target, radar) and metadata-rich dataset batches."""
    if len(batch) == 2:
        target, radar = batch
        return target, radar, {}
    if len(batch) == 3:
        target, radar, meta = batch
        return target, radar, meta
    if len(batch) == 4:
        target, radar, meta, _path = batch
        return target, radar, meta
    raise ValueError(f"Unsupported batch format with {len(batch)} elements")


def pad_latent_input_to_sixteen_channels(model_input: torch.Tensor) -> torch.Tensor:
    if model_input.shape[1] == 16:
        return model_input
    if model_input.shape[1] > 16:
        raise ValueError(f"Expected <=16 latent input channels, got {model_input.shape[1]}")
    pad = torch.zeros(
        model_input.shape[0],
        16 - model_input.shape[1],
        *model_input.shape[2:],
        device=model_input.device,
        dtype=model_input.dtype,
    )
    return torch.cat([model_input, pad], dim=1)


def call_cd_denoiser(
    model: nn.Module,
    x_t: torch.Tensor,
    z_cond: Optional[torch.Tensor],
    timesteps: torch.Tensor,
    radar_voxel: Optional[torch.Tensor] = None,
    meta_dict: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Call legacy or multimodal denoisers through one CD training interface."""
    if getattr(model, "is_multimodal", False):
        if has_multimodal_meta(meta_dict) and radar_voxel is not None:
            return model(
                radar_voxel,
                meta_dict["ir_img"],
                meta_dict["r_mat"],
                meta_dict["t_vec"],
                meta_dict["k_mat"],
                timesteps,
                noised_latent=x_t,
            )
        if z_cond is None:
            raise ValueError("缺少 legacy multimodal CD condition latent")
        model_input = pad_latent_input_to_sixteen_channels(torch.cat([x_t, z_cond], dim=1))
        return model.unet_3d(model_input, timesteps)
    if z_cond is None:
        raise ValueError("缺少 legacy CD condition latent")
    return model(torch.cat([x_t, z_cond], dim=1), timesteps)


class ConsistencyDistillationTrainer:
    """
    LDM 初始化的 EMA consistency 训练器
    
    流程：
    1. 加载预训练 LDM 作为一次性初始化来源
    2. 从相同权重创建 CD 模型和 EMA 目标模型
    3. 训练 CD 输出逼近持续更新的 CD EMA 目标
    """
    
    def __init__(
        self,
        ldm_ckpt_path: str,
        vae: nn.Module,
        device: str = "cuda",
        config: dict = None,
    ):
        self.device = device
        self.config = config or {}
        self.data_grid_config = dict(
            self.config.get("data_grid_config", {}) or {}
        )
        self.allow_legacy_radar_units = self.config.get(
            "allow_legacy_radar_units", False
        )
        self.checkpoint_protocol = resolve_training_checkpoint_protocol(
            self.config.get("checkpoint_protocol", FORMAL_CHECKPOINT_PROTOCOL)
        )
        teacher_checkpoint = safe_torch_load(ldm_ckpt_path, map_location="cpu")
        self.radar_normalization, self.radar_normalization_sha256 = (
            resolve_cd_radar_normalization(
                teacher_checkpoint,
                self.config.get("radar_normalization"),
                self.config.get("radar_normalization_sha256", ""),
                data_grid_config=self.data_grid_config,
                allow_legacy_radar_units=self.allow_legacy_radar_units,
            )
        )
        self._preloaded_ldm_checkpoint = teacher_checkpoint
        resume_path = self.config.get('resume_path')
        if resume_path and os.path.exists(resume_path):
            resume_checkpoint = safe_torch_load(resume_path, map_location="cpu")
            assert_checkpoint_radar_normalization(
                resume_checkpoint,
                self.radar_normalization,
                self.radar_normalization_sha256,
                target_size=self.data_grid_config.get("target_size"),
                source_pc_range=self.data_grid_config.get("source_pc_range"),
                model_pc_range=self.data_grid_config.get("model_pc_range"),
                allow_legacy_radar_units=self.allow_legacy_radar_units,
                context="CD resume preflight",
            )
            self._preloaded_resume_checkpoint = resume_checkpoint
        
        # 设置保存目录和日志
        self.save_dir = self.config.get('save_dir', './Result/train_results/cd')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化训练状态
        self.start_epoch = 1
        self.best_loss = float('inf')
        self.is_resumed = False
        self.model_config = dict(self.config.get("ldm", {}) or self.config.get("model", {}) or {})
        self.model_config.setdefault("latent_dim", int(vae.latent_dim))
        self.vae_checkpoint_sha256 = str(self.config.get("vae_checkpoint_sha256", "") or "")
        self.ldm_checkpoint_sha256 = str(self.config.get("ldm_checkpoint_sha256", "") or "")
        self.model_config.setdefault(
            "fusion_voxel_shape",
            list(self.data_grid_config.get("target_size", [32, 128, 128])),
        )
        self.model_config.setdefault(
            "fusion_latent_shape",
            list(self.model_config["fusion_voxel_shape"]),
        )
        self.model_config.setdefault(
            "fusion_pc_range",
            list(self.data_grid_config.get("model_pc_range", [0, -20, -6, 120, 20, 10])),
        )
        self.data_grid_config.setdefault(
            "target_size", list(self.model_config["fusion_voxel_shape"])
        )
        self.data_grid_config.setdefault(
            "source_pc_range", list(self.data_grid_config.get("model_pc_range", [0, -20, -6, 120, 20, 10]))
        )
        self.data_grid_config.setdefault(
            "model_pc_range", list(self.model_config["fusion_pc_range"])
        )
        
        # LDM 仅提供初始权重；后续 consistency target 来自 CD EMA。
        self.ldm_model = self._load_ldm_model(ldm_ckpt_path)
        self.use_multimodal = bool(getattr(self.ldm_model, "is_multimodal", False))
        self.vae = vae.to(device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # 创建 CD 学生模型（与 LDM 同结构）
        self.cd_model = create_cd_model(self.use_multimodal, self.model_config).to(device)
        
        # 创建 EMA 目标模型
        self.cd_model_ema = self._create_ema_model(self.cd_model)
        
        # 初始化 CD 模型为 LDM 的拷贝
        self._initialize_from_ldm()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.cd_model.parameters(),
            lr=self.config.get('lr', 5e-5),
            weight_decay=1e-4,
        )
        
        # Denoiser
        self.denoiser = KarrasDenoiser(
            sigma_data=0.5,
            sigma_max=80.0,
            sigma_min=0.002,
            loss_norm='l2',
        )
        
        # 禁用混合精度（避免 FP16/FP32 类型不匹配）
        self.use_amp = False
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # 设置日志
        self.log_file = os.path.join(self.save_dir, 'training.log')
        self.csv_file = os.path.join(self.save_dir, 'metrics.csv')
        
        # 检查是否恢复训练
        if resume_path and os.path.exists(resume_path):
            self.is_resumed = True
        
        self._setup_logging()
        
        # 恢复训练
        if self.is_resumed:
            self._resume_from_checkpoint(resume_path)
    
    def _load_ldm_model(self, ckpt_path: str) -> nn.Module:
        """加载 LDM 教师模型"""
        # 静默创建模型（避免重复打印）
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        ckpt = getattr(self, "_preloaded_ldm_checkpoint", None)
        if ckpt is None:
            ckpt = safe_torch_load(ckpt_path, map_location='cpu')
        state_dict = checkpoint_state_dict(ckpt)
        resolved_config = resolve_cd_generation_config(
            ckpt,
            fallback_latent_dim=self.model_config.get("latent_dim"),
        )
        resolved_config.update({
            key: value for key, value in self.model_config.items()
            if key not in resolved_config
        })
        self.model_config = resolved_config
        model = create_cd_model(has_multimodal_state_dict(state_dict), self.model_config).to(self.device)
        
        sys.stdout = old_stdout
        
        model.load_state_dict(state_dict)
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        print(f"Loaded LDM initialization checkpoint from {ckpt_path}")
        return model
    
    def _create_ema_model(self, model: nn.Module) -> nn.Module:
        """创建 EMA 目标模型"""
        # 静默创建模型
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        ema_model = create_cd_model(
            bool(getattr(model, "is_multimodal", False)),
            self.model_config,
        ).to(self.device)
        
        sys.stdout = old_stdout
        
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        
        return ema_model
    
    def _setup_logging(self):
        """设置日志系统"""
        # 确定日志文件模式
        log_mode = 'a' if self.is_resumed else 'w'
        
        # 配置文本日志（只写入文件，不输出到终端）
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, mode=log_mode)
            ],
            force=True
        )
        self.logger = logging.getLogger(__name__ + '_cd')
        
        # 添加训练会话分隔符
        if self.is_resumed:
            self.logger.info("\n" + "="*70)
            self.logger.info("RESUMING TRAINING SESSION")
            self.logger.info("="*70)
        
        # 初始化 CSV 文件
        if not os.path.exists(self.csv_file) or not self.is_resumed:
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'loss', 'lr', 'time_seconds'])
    
    def _resume_from_checkpoint(self, ckpt_path: str):
        """从检查点恢复训练"""
        print(f"Resuming CD from checkpoint: {ckpt_path}")
        ckpt = getattr(self, "_preloaded_resume_checkpoint", None)
        if ckpt is None:
            ckpt = safe_torch_load(ckpt_path, map_location=self.device)
        assert_checkpoint_radar_normalization(
            ckpt,
            self.radar_normalization,
            self.radar_normalization_sha256,
            target_size=self.data_grid_config.get("target_size"),
            source_pc_range=self.data_grid_config.get("source_pc_range"),
            model_pc_range=self.data_grid_config.get("model_pc_range"),
            allow_legacy_radar_units=self.allow_legacy_radar_units,
            context="CD resume checkpoint",
        )
        
        # 加载模型
        self.cd_model.load_state_dict(ckpt['model_state_dict'])
        if 'ema_model_state_dict' in ckpt:
            self.cd_model_ema.load_state_dict(ckpt['ema_model_state_dict'])
        
        # 加载优化器
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        # 加载训练状态
        self.start_epoch = ckpt.get('epoch', 0) + 1
        self.best_loss = ckpt.get('best_loss', ckpt.get('loss', float('inf')))
        
        print(f"Resumed from epoch {self.start_epoch - 1}, best loss: {self.best_loss:.4f}")
    
    def _log_metrics(self, epoch: int, loss: float, epoch_time: float):
        """记录指标到 CSV 文件"""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f'{loss:.6f}',
                f'{self.optimizer.param_groups[0]["lr"]:.8f}',
                f'{epoch_time:.2f}'
            ])
    
    def _initialize_from_ldm(self):
        """用 LDM 权重初始化 CD 模型"""
        if not self.is_resumed:
            self.cd_model.load_state_dict(self.ldm_model.state_dict())
            self.cd_model_ema.load_state_dict(self.ldm_model.state_dict())
            print("Initialized CD model and EMA target from LDM checkpoint")
    
    def _update_ema(self, ema_rate: float = 0.999):
        """更新 EMA 模型"""
        with torch.no_grad():
            for src_param, ema_param in zip(
                self.cd_model.parameters(),
                self.cd_model_ema.parameters()
            ):
                ema_param.mul_(ema_rate).add_(src_param.data, alpha=1 - ema_rate)
    
    @torch.no_grad()
    def _euler_solver(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        next_t: torch.Tensor,
        cond: torch.Tensor,
        radar_voxel: Optional[torch.Tensor] = None,
        meta_dict: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Euler ODE 求解器
        
        用于从当前时间步推进到下一时间步
        """
        # 模型输出
        denoised = call_cd_denoiser(
            model,
            x_t,
            cond,
            t,
            radar_voxel=radar_voxel,
            meta_dict=meta_dict,
        )
        
        # NOTE: 这里使用显式 Euler，对应一致性蒸馏中的教师推进步骤。
        # NOTE: 常微分方程（ODE）形式：dx/dt = (x - denoised) / t。
        d = (x_t - denoised) / t.view(-1, 1, 1, 1, 1)
        
        # Euler 步进
        dt = next_t - t
        x_next = x_t + d * dt.view(-1, 1, 1, 1, 1)
        
        return x_next
    
    def train_step(
        self,
        z_target: torch.Tensor,
        z_cond: Optional[torch.Tensor],
        num_scales: int = 40,
        radar_voxel: Optional[torch.Tensor] = None,
        meta_dict: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        单个训练步骤（Consistency Distillation）
        
        核心思想：
        1. 学生模型：从 x(t_n) 直接一步预测去噪后的结果
        2. 教师模型：从 x(t_n) 用 ODE solver 推进到 t_{n+1}，再预测去噪结果
        3. 让学生的一步预测接近教师的多步结果
        """
        batch_size = z_target.shape[0]
        device = z_target.device
        
        # NOTE: 随机采样相邻时间步对 (t_n, t_{n+1})，覆盖不同噪声区间。
        indices = torch.randint(0, num_scales - 1, (batch_size,), device=device)
        
        # 计算噪声水平
        t_n = (
            self.denoiser.sigma_max ** (1 / self.denoiser.rho) +
            indices / (num_scales - 1) * (
                self.denoiser.sigma_min ** (1 / self.denoiser.rho) -
                self.denoiser.sigma_max ** (1 / self.denoiser.rho)
            )
        ) ** self.denoiser.rho
        
        t_next = (
            self.denoiser.sigma_max ** (1 / self.denoiser.rho) +
            (indices + 1) / (num_scales - 1) * (
                self.denoiser.sigma_min ** (1 / self.denoiser.rho) -
                self.denoiser.sigma_max ** (1 / self.denoiser.rho)
            )
        ) ** self.denoiser.rho
        
        # 生成带噪数据 x(t_n)
        noise = torch.randn_like(z_target)
        x_t_n = z_target + noise * t_n.view(-1, 1, 1, 1, 1)
        
        # 学生模型：从 x(t_n) 直接预测去噪结果
        student_denoised = call_cd_denoiser(
            self.cd_model,
            x_t_n,
            z_cond,
            t_n,
            radar_voxel=radar_voxel,
            meta_dict=meta_dict,
        )
        
        # CD EMA 目标模型：从 x(t_n) 推进到 x(t_{n+1})，再预测。
        with torch.no_grad():
            # 该模型由 CD 参数持续 EMA 更新，不是冻结的 LDM 教师。
            x_t_next = self._euler_solver(
                self.cd_model_ema,
                x_t_n,
                t_n,
                t_next,
                z_cond,
                radar_voxel=radar_voxel,
                meta_dict=meta_dict,
            )
            
            # EMA 目标在 t_{n+1} 的预测
            teacher_denoised = call_cd_denoiser(
                self.cd_model_ema,
                x_t_next,
                z_cond,
                t_next,
                radar_voxel=radar_voxel,
                meta_dict=meta_dict,
            )
        
        # NOTE: 一致性目标：CD 一步输出逼近 CD EMA 推进后的输出。
        loss = F.mse_loss(student_denoised, teacher_denoised)
        
        return loss
    
    def train_epoch(
        self,
        epoch: int,
        train_loader: DataLoader,
        num_scales: int = 40,
        grad_accum_steps: int = 8,
    ) -> float:
        """训练一个 epoch"""
        self.cd_model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            target, cond, meta_dict = unpack_cd_batch(batch)
            target = target.to(self.device)
            cond = cond.to(self.device)
            meta_dict = move_meta_to_device(meta_dict, self.device)
            
            # 编码到潜空间
            with torch.no_grad():
                z_target, z_cond = encode_cd_training_latents(
                    self.vae,
                    target,
                    cond,
                    meta_dict,
                )
            
            # 计算损失
            loss = self.train_step(
                z_target,
                z_cond,
                num_scales=num_scales,
                radar_voxel=cond,
                meta_dict=meta_dict,
            )
            loss = loss / grad_accum_steps
            
            # 反向传播
            loss.backward()
            
            # NOTE: 梯度累积用于控制显存；每 grad_accum_steps 次小步做一次参数更新。
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.cd_model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # 更新 EMA 模型
                self._update_ema(ema_rate=0.999)
            
            total_loss += loss.item() * grad_accum_steps
            pbar.set_postfix({'loss': f'{loss.item() * grad_accum_steps:.6f}'})

        if len(train_loader) % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.cd_model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._update_ema(ema_rate=0.999)
        
        return total_loss / len(train_loader)

    def _checkpoint_payload(self, epoch: int, loss: float, best_loss: float) -> Dict[str, Any]:
        """构造带完整网格和父 checkpoint hash 的正式 CD checkpoint。"""
        payload = {
            "epoch": epoch,
            "loss": loss,
            "best_loss": best_loss,
            "model_state_dict": self.cd_model.state_dict(),
            "ema_model_state_dict": self.cd_model_ema.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "latent_dim": int(self.model_config["latent_dim"]),
            "model_config": dict(self.model_config),
            "checkpoint_protocol": (
                getattr(
                    self,
                    "checkpoint_protocol",
                    FORMAL_CHECKPOINT_PROTOCOL,
                )
                if self.use_multimodal and self.radar_normalization is not None
                else "legacy_cd_v0"
            ),
            "stage": "cd",
            "data_grid_config": dict(self.data_grid_config),
            "vae_checkpoint_sha256": self.vae_checkpoint_sha256,
            "ldm_checkpoint_sha256": self.ldm_checkpoint_sha256,
            "model_family": "multimodal" if self.use_multimodal else "legacy",
            "training_semantics": "ldm_initialized_ema_consistency_v1",
            "ldm_role": "initialization_checkpoint",
            "consistency_target_source": "cd_model_ema",
        }
        if self.radar_normalization is not None:
            payload["radar_normalization"] = dict(self.radar_normalization)
            payload["radar_normalization_sha256"] = self.radar_normalization_sha256
        return payload
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 100,
        save_every: int = 10,
        grad_accum_steps: int = 8,
    ):
        """完整训练流程"""
        estimated_total_steps = num_epochs * len(train_loader)
        
        msg = "="*70 + "\n"
        msg += f"Starting CD Training\n"
        msg += f"  Total epochs: {num_epochs}\n"
        msg += f"  Batches per epoch: {len(train_loader)}\n"
        msg += f"  Estimated total steps: {estimated_total_steps:,}\n"
        msg += f"  Start epoch: {self.start_epoch}\n"
        msg += f"  Batch size: {train_loader.batch_size}\n"
        msg += f"  Gradient accumulation: {grad_accum_steps}\n"
        msg += f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}\n"
        msg += f"  Save directory: {self.save_dir}\n"
        msg += f"  Log file: {self.log_file}\n"
        msg += f"  CSV file: {self.csv_file}\n"
        msg += "="*70
        print(msg) 
        self.logger.info(msg)

        for epoch in range(self.start_epoch, num_epochs + 1):
            epoch_start = time.time()
            loss = self.train_epoch(epoch, train_loader, grad_accum_steps=grad_accum_steps)
            epoch_time = time.time() - epoch_start
            
            # 记录日志
            msg = f"\n[Epoch {epoch}/{num_epochs}] Loss: {loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e} | Time: {epoch_time:.1f}s"
            print(msg)
            self.logger.info(msg)
            self._log_metrics(epoch, loss, epoch_time)
            
            # 保存最佳模型
            if loss < self.best_loss:
                self.best_loss = loss
                best_ckpt = os.path.join(self.save_dir, "cd_best.pt")
                torch.save(
                    self._checkpoint_payload(epoch, loss, self.best_loss),
                    best_ckpt,
                )
                msg = f"  ✓ Saved best model (loss: {loss:.4f})"
                self.logger.info(msg)
            
            # 定期保存检查点
            if epoch % save_every == 0:
                ckpt_path = os.path.join(self.save_dir, f"cd_epoch{epoch:04d}.pt")
                torch.save(
                    self._checkpoint_payload(epoch, loss, self.best_loss),
                    ckpt_path,
                )
                self.logger.info(f"  Saved checkpoint: {ckpt_path}")
        
        msg = "\nTraining completed!"
        print(msg)
        self.logger.info(msg)


def main():
    parser = argparse.ArgumentParser(description="Consistency Distillation Training")
    parser.add_argument("--ldm_ckpt", type=str, required=True, help="LDM checkpoint path")
    parser.add_argument("--vae_ckpt", type=str, required=True, help="VAE checkpoint path")
    parser.add_argument("--config", type=str, default="", help="Optional unified YAML config")
    parser.add_argument("--dataset_dir", type=str, default="./Data/NTU4DRadLM_Pre")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--save_dir", type=str, default="./Result/train_results/cd")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument(
        "--allow_legacy_radar_units",
        action="store_true",
        help="仅供 mini-test/旧诊断保留未归一化 Radar；正式 CD 禁止使用",
    )
    
    args = parser.parse_args()
    
    config = load_yaml_config(args.config)
    cd_config = dict(config.get("cd", {}) if isinstance(config.get("cd", {}), dict) else {})
    ldm_config = dict(config.get("ldm", {}) if isinstance(config.get("ldm", {}), dict) else {})
    opt_config = dict(config.get("optimization", {}) if isinstance(config.get("optimization", {}), dict) else {})
    data_config = dict(config.get("data", {}) if isinstance(config.get("data", {}), dict) else {})
    target_size, source_pc_range, model_pc_range = resolve_data_grid_config(data_config)
    ldm_config.setdefault("fusion_voxel_shape", list(target_size))
    ldm_config.setdefault("fusion_pc_range", list(model_pc_range))
    artifact_path = data_config.get("radar_normalization_path")
    scale_mps = data_config.get("doppler_scale_mps")
    path_configured = isinstance(artifact_path, str) and bool(artifact_path.strip())
    scale_configured = scale_mps not in (None, "")
    if args.allow_legacy_radar_units:
        if path_configured or scale_configured:
            raise RadarNormalizationError(
                "CD legacy 开关与正式 normalization 配置不能同时启用"
            )
        radar_normalization, radar_normalization_sha256 = None, ""
    else:
        if not path_configured or not scale_configured:
            raise RadarNormalizationError(
                "正式 CD 必须配置 radar_normalization_path 和 doppler_scale_mps"
            )
        radar_normalization, radar_normalization_sha256 = (
            load_radar_normalization_artifact(
                artifact_path.strip(),
                target_size=target_size,
                source_pc_range=source_pc_range,
                model_pc_range=model_pc_range,
                doppler_scale_mps=scale_mps,
                require_formal=True,
            )
        )

    # 加载 VAE
    ckpt = safe_torch_load(args.vae_ckpt, map_location='cpu')
    vae_config = config.get("vae", {}) if isinstance(config.get("vae", {}), dict) else {}
    vae, _vae_metadata = build_cd_vae_from_checkpoint(
        ckpt,
        fallback_config_type=vae_config.get("config_type"),
    )
    
    # 创建数据加载器
    dataset = NTU4DRadLM_VoxelDataset(
        root_dir=args.dataset_dir,
        split='train',
        use_augmentation=False,
        target_size=target_size,
        source_pc_range=source_pc_range,
        model_pc_range=model_pc_range,
        radar_normalization=radar_normalization,
        radar_normalization_sha256=radar_normalization_sha256,
        allow_legacy_radar_units=args.allow_legacy_radar_units,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        collate_fn=collate_voxel_samples,
    )
    
    cd_save_dir = cd_config.get("save_dir", args.save_dir)

    # 检查是否有检查点可以恢复
    resume_path = os.path.join(cd_save_dir, 'cd_best.pt')
    if not os.path.exists(resume_path):
        resume_path = None
    
    # 创建训练器并训练
    trainer = ConsistencyDistillationTrainer(
        ldm_ckpt_path=args.ldm_ckpt,
        vae=vae,
        config={
            'lr': float(cd_config.get("lr", args.lr)),
            'save_dir': cd_save_dir,
            'resume_path': resume_path,
            'ldm': ldm_config,
            'data_grid_config': {
                'target_size': list(target_size),
                'source_pc_range': list(source_pc_range),
                'model_pc_range': list(model_pc_range),
            },
            'vae_checkpoint_sha256': sha256_file(args.vae_ckpt),
            'ldm_checkpoint_sha256': sha256_file(args.ldm_ckpt),
            'radar_normalization': radar_normalization,
            'radar_normalization_sha256': radar_normalization_sha256,
            'allow_legacy_radar_units': args.allow_legacy_radar_units,
            'checkpoint_protocol': data_config.get(
                'checkpoint_protocol', FORMAL_CHECKPOINT_PROTOCOL
            ),
        },
    )
    
    trainer.train(
        train_loader,
        num_epochs=int(cd_config.get("epochs", args.num_epochs)),
        save_every=int(cd_config.get("save_every", 10)),
        grad_accum_steps=int(opt_config.get("gradient_accumulation_steps", args.grad_accum_steps)),
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()
