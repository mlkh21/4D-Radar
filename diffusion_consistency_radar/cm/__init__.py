# -- coding: utf-8 --

"""
Codebase for "Improved Denoising Diffusion Probabilistic Models".

优化版本 - 针对 4D 雷达数据的改进:
1. 多种高效注意力机制 (Flash/Window/Linear/Sparse)
2. 非对称下采样策略 (保护Z轴分辨率)
3. 多种归一化选项 (Group/Layer/Instance/RMS)
4. Latent Diffusion 支持 (3D VAE/VQ-VAE)
5. 通道瘦身配置

新增模块:
- attention_optimized: 优化的注意力机制
- sampling_optimized: 非对称采样模块
- unet_optimized: 优化的 UNet 模型
- vae_3d: 3D VAE 用于潜空间扩散
"""

# 核心模块
from .nn import (
    SiLU,
    GroupNorm32,
    LayerNorm3D,
    InstanceNorm3D,
    RMSNorm3D,
    AdaptiveNorm3D,
    ConditionalNorm3D,
    conv_nd,
    linear,
    avg_pool_nd,
    normalization,
    timestep_embedding,
    checkpoint,
    zero_module,
    scale_module,
    mean_flat,
    append_dims,
    append_zero,
    update_ema,
    set_norm_type,
    set_norm_groups,
)

# 优化的注意力机制
from .attention_optimized import (
    Window3DAttention,
    FlashAttention3D,
    LinearAttention3D,
    SparseAttention3D,
    HeightSelfAttention3D,
    HybridAttention3D,
    create_attention_block,
)

# 优化的采样模块
from .sampling_optimized import (
    AsymmetricDownsample3D,
    AsymmetricUpsample3D,
    AdaptiveDownsampleScheduler,
    DepthWiseDownsample3D,
    MultiScaleDownsample3D,
    create_downsample_block,
    create_upsample_block,
)

# 模型
from .unet import UNetModel
from .unet_optimized import (
    OptimizedUNetModel,
    OptimizedResBlock,
    create_lightweight_unet_config,
    create_ultra_lightweight_unet_config,
    create_balanced_unet_config,
)

# 3D VAE
from .vae_3d import (
    VAE3D,
    VQVAE3D,
    VAE3DEncoder,
    VAE3DDecoder,
    VectorQuantizer,
    create_ultra_lightweight_vae_config,
    create_lightweight_vae_config,
    create_standard_vae_config,
)

# 扩散过程
from .karras_diffusion import KarrasDenoiser

# 工具函数
from .script_util_cond import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_model,
    cm_train_defaults,
    add_dict_to_argparser,
    args_to_dict,
    str2bool,
)

# 显存优化模块
from .memory_efficient import (
    MemoryManager,
    SparsityAwareProcessor,
    DynamicResolutionProcessor,
    ChunkedProcessor,
    ProgressiveTrainer,
    MemoryEfficientTrainingWrapper,
)

# 数据增强模块
from .augmentation import (
    VoxelAugmentation,
    MixupAugmentation,
    CutoutAugmentation,
    ComposedAugmentation,
)

# 3D损失函数
from .losses_3d import (
    Perceptual3DLoss,
    StructurePreservingLoss,
    OccupancyAwareLoss,
    CompositeLoss3D,
)

__all__ = [
    # NN 工具
    "SiLU", "GroupNorm32", "LayerNorm3D", "InstanceNorm3D", "RMSNorm3D",
    "AdaptiveNorm3D", "ConditionalNorm3D",
    "conv_nd", "linear", "avg_pool_nd", "normalization", "timestep_embedding",
    "checkpoint", "zero_module", "scale_module", "mean_flat", "append_dims",
    "append_zero", "update_ema", "set_norm_type", "set_norm_groups",
    
    # 注意力
    "Window3DAttention", "FlashAttention3D", "LinearAttention3D",
    "SparseAttention3D", "HeightSelfAttention3D", "HybridAttention3D",
    "create_attention_block",
    
    # 采样
    "AsymmetricDownsample3D", "AsymmetricUpsample3D", "AdaptiveDownsampleScheduler",
    "DepthWiseDownsample3D", "MultiScaleDownsample3D",
    "create_downsample_block", "create_upsample_block",
    
    # 模型
    "UNetModel", "OptimizedUNetModel", "OptimizedResBlock",
    "create_lightweight_unet_config", "create_ultra_lightweight_unet_config",
    "create_balanced_unet_config",
    
    # VAE
    "VAE3D", "VQVAE3D", "VAE3DEncoder", "VAE3DDecoder", "VectorQuantizer",
    "create_ultra_lightweight_vae_config", "create_lightweight_vae_config", "create_standard_vae_config",
    
    # 扩散
    "KarrasDenoiser",
    
    # 工具
    "model_and_diffusion_defaults", "create_model_and_diffusion", "create_model",
    "cm_train_defaults", "add_dict_to_argparser", "args_to_dict", "str2bool",
    
    # 显存优化
    "MemoryManager", "SparsityAwareProcessor", "DynamicResolutionProcessor",
    "ChunkedProcessor", "ProgressiveTrainer", "MemoryEfficientTrainingWrapper",
    
    # 数据增强
    "VoxelAugmentation", "MixupAugmentation", "CutoutAugmentation", "ComposedAugmentation",
    
    # 3D损失
    "Perceptual3DLoss", "StructurePreservingLoss", "OccupancyAwareLoss", "CompositeLoss3D",
]
