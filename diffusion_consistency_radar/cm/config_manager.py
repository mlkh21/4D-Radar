#!/usr/bin/env python3
"""
训练配置管理器

功能:
- 集中管理所有训练配置
- 提供预设配置档案（低/中/高显存）
- 支持配置验证和自动调整
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import yaml
import os


@dataclass
class MemoryConfig:
    """显存配置"""
    # 基础设置
    batch_size: int = 2
    microbatch: int = 2
    gradient_accumulation_steps: int = 4
    
    # 混合精度
    use_amp: bool = True
    fp16_scale_growth: float = 1e-3
    
    # 梯度检查点
    use_gradient_checkpointing: bool = True
    
    # 缓存管理
    clear_cache_interval: int = 10
    target_memory_fraction: float = 0.9
    
    # 数据加载
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int = 2


@dataclass
class ModelConfig:
    """模型配置"""
    # UNet 配置
    model_channels: int = 64
    channel_mult: List[int] = field(default_factory=lambda: [1, 2, 4])
    num_res_blocks: int = 2
    attention_resolutions: str = "16,8"
    
    # 注意力类型
    attention_type: str = "linear"  # linear, window, sparse, flash
    
    # 归一化
    norm_type: str = "group"  # group, layer, instance, rms
    norm_groups: int = 16
    
    # VAE 配置
    latent_dim: int = 4
    vae_channels: List[int] = field(default_factory=lambda: [32, 64, 128])


@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径
    data_dir: str = ""
    
    # 数据增强
    use_augmentation: bool = True
    flip_prob: float = 0.5
    rotate_prob: float = 0.3
    noise_prob: float = 0.2
    noise_std: float = 0.02
    dropout_prob: float = 0.1
    point_dropout_rate: float = 0.05
    intensity_jitter_prob: float = 0.1
    doppler_jitter_prob: float = 0.05
    
    # 数据格式
    alignment_size: int = 32


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    lr: float = 1e-4
    weight_decay: float = 0.01
    total_steps: int = 100000
    
    # EMA
    ema_rate: str = "0.9999"
    
    # 日志和保存
    log_interval: int = 100
    save_interval: int = 10000
    
    # 验证
    enable_validation: bool = True
    val_interval: int = 1000
    early_stopping_patience: int = 10
    
    # 扩散参数
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5


@dataclass
class FullConfig:
    """完整配置"""
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def get_preset_config(preset: str = "medium", gpu_memory_gb: float = 24.0) -> FullConfig:
    """
    获取预设配置
    
    输入:
        preset: 预设名称 ("low", "medium", "high", "auto")
        gpu_memory_gb: GPU显存大小 (GB)
        
    返回:
        config: 完整配置对象
    """
    config = FullConfig()
    
    if preset == "auto":
        # 根据GPU显存自动选择
        if gpu_memory_gb < 12:
            preset = "low"
        elif gpu_memory_gb < 24:
            preset = "medium"
        else:
            preset = "high"
    
    if preset == "low":
        # 低显存配置 (8-12GB)
        config.memory.batch_size = 1
        config.memory.microbatch = 1
        config.memory.gradient_accumulation_steps = 8
        config.memory.use_amp = True
        config.memory.use_gradient_checkpointing = True
        config.memory.num_workers = 1
        
        config.model.model_channels = 48
        config.model.channel_mult = [1, 2, 3]
        config.model.num_res_blocks = 1
        config.model.attention_type = "linear"
        
        config.training.log_interval = 50
        
    elif preset == "medium":
        # 中等显存配置 (16-24GB)
        config.memory.batch_size = 2
        config.memory.microbatch = 2
        config.memory.gradient_accumulation_steps = 4
        config.memory.use_amp = True
        config.memory.use_gradient_checkpointing = True
        config.memory.num_workers = 2
        
        config.model.model_channels = 64
        config.model.channel_mult = [1, 2, 4]
        config.model.num_res_blocks = 2
        config.model.attention_type = "linear"
        
    elif preset == "high":
        # 高显存配置 (32GB+)
        config.memory.batch_size = 4
        config.memory.microbatch = 4
        config.memory.gradient_accumulation_steps = 2
        config.memory.use_amp = True
        config.memory.use_gradient_checkpointing = False
        config.memory.num_workers = 4
        
        config.model.model_channels = 96
        config.model.channel_mult = [1, 2, 4, 8]
        config.model.num_res_blocks = 3
        config.model.attention_type = "window"
    
    return config


def estimate_memory_usage(config: FullConfig, input_shape: tuple = (2, 4, 32, 128, 128)) -> Dict[str, float]:
    """
    估算显存使用
    
    输入:
        config: 配置对象
        input_shape: 输入形状 (B, C, Z, H, W)
        
    返回:
        memory_estimate: 显存估算 (GB)
    """
    B, C, Z, H, W = input_shape
    
    # 参数显存（粗略估算）
    base_channels = config.model.model_channels
    total_params = 0
    
    for i, mult in enumerate(config.model.channel_mult):
        ch = base_channels * mult
        # 卷积层参数
        total_params += ch * ch * 27 * config.model.num_res_blocks * 2
        # 注意力层参数
        total_params += ch * ch * 4
    
    param_memory = total_params * 4 / (1024**3)  # float32, GB
    
    # 激活显存（粗略估算）
    activation_size = B * base_channels * Z * H * W * 4 / (1024**3)
    
    # 梯度显存
    gradient_memory = param_memory
    
    # 优化器状态
    optimizer_memory = param_memory * 2  # Adam
    
    # 混合精度节省
    if config.memory.use_amp:
        activation_size *= 0.5
    
    # 梯度检查点节省
    if config.memory.use_gradient_checkpointing:
        activation_size *= 0.3
    
    total = param_memory + activation_size + gradient_memory + optimizer_memory
    
    return {
        'parameters_gb': param_memory,
        'activations_gb': activation_size,
        'gradients_gb': gradient_memory,
        'optimizer_gb': optimizer_memory,
        'total_gb': total,
        'estimated_params': total_params
    }


def save_config(config: FullConfig, path: str):
    """保存配置到YAML文件"""
    config_dict = {
        'memory': config.memory.__dict__,
        'model': config.model.__dict__,
        'data': config.data.__dict__,
        'training': config.training.__dict__
    }
    
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


def load_config(path: str) -> FullConfig:
    """从YAML文件加载配置"""
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = FullConfig()
    
    if 'memory' in config_dict:
        for k, v in config_dict['memory'].items():
            if hasattr(config.memory, k):
                setattr(config.memory, k, v)
    
    if 'model' in config_dict:
        for k, v in config_dict['model'].items():
            if hasattr(config.model, k):
                setattr(config.model, k, v)
    
    if 'data' in config_dict:
        for k, v in config_dict['data'].items():
            if hasattr(config.data, k):
                setattr(config.data, k, v)
    
    if 'training' in config_dict:
        for k, v in config_dict['training'].items():
            if hasattr(config.training, k):
                setattr(config.training, k, v)
    
    return config


def config_to_args(config: FullConfig) -> Dict[str, Any]:
    """将配置转换为命令行参数格式"""
    args = {}
    
    # 显存配置
    args['batch_size'] = config.memory.batch_size
    args['microbatch'] = config.memory.microbatch
    args['gradient_accumulation_steps'] = config.memory.gradient_accumulation_steps
    args['use_amp'] = config.memory.use_amp
    args['fp16_scale_growth'] = config.memory.fp16_scale_growth
    args['use_gradient_checkpointing'] = config.memory.use_gradient_checkpointing
    
    # 模型配置
    args['model_channels'] = config.model.model_channels
    args['channel_mult'] = ','.join(map(str, config.model.channel_mult))
    args['num_res_blocks'] = config.model.num_res_blocks
    args['attention_resolutions'] = config.model.attention_resolutions
    args['attention_type'] = config.model.attention_type
    args['norm_type'] = config.model.norm_type
    
    # 数据配置
    args['data_dir'] = config.data.data_dir
    args['use_augmentation'] = config.data.use_augmentation
    
    # 训练配置
    args['lr'] = config.training.lr
    args['weight_decay'] = config.training.weight_decay
    args['lr_anneal_steps'] = config.training.total_steps
    args['ema_rate'] = config.training.ema_rate
    args['log_interval'] = config.training.log_interval
    args['save_interval'] = config.training.save_interval
    args['sigma_min'] = config.training.sigma_min
    args['sigma_max'] = config.training.sigma_max
    
    return args


def print_config_summary(config: FullConfig, gpu_memory_gb: float = 24.0):
    """打印配置摘要和显存估算"""
    print("\n" + "=" * 60)
    print("训练配置摘要")
    print("=" * 60)
    
    print("\n? 显存配置:")
    print(f"  Batch Size: {config.memory.batch_size}")
    print(f"  Gradient Accumulation: {config.memory.gradient_accumulation_steps}")
    print(f"  有效 Batch Size: {config.memory.batch_size * config.memory.gradient_accumulation_steps}")
    print(f"  混合精度 (AMP): {'?' if config.memory.use_amp else '?'}")
    print(f"  梯度检查点: {'?' if config.memory.use_gradient_checkpointing else '?'}")
    
    print("\n? 模型配置:")
    print(f"  基础通道数: {config.model.model_channels}")
    print(f"  通道倍率: {config.model.channel_mult}")
    print(f"  残差块数: {config.model.num_res_blocks}")
    print(f"  注意力类型: {config.model.attention_type}")
    
    print("\n? 数据配置:")
    print(f"  数据增强: {'?' if config.data.use_augmentation else '?'}")
    print(f"  对齐大小: {config.data.alignment_size}")
    
    print("\n? 训练配置:")
    print(f"  学习率: {config.training.lr}")
    print(f"  总步数: {config.training.total_steps}")
    print(f"  验证间隔: {config.training.val_interval}")
    print(f"  早停耐心: {config.training.early_stopping_patience}")
    
    # 显存估算
    mem_est = estimate_memory_usage(config)
    print("\n? 显存估算:")
    print(f"  参数: {mem_est['parameters_gb']:.2f} GB")
    print(f"  激活: {mem_est['activations_gb']:.2f} GB")
    print(f"  梯度: {mem_est['gradients_gb']:.2f} GB")
    print(f"  优化器: {mem_est['optimizer_gb']:.2f} GB")
    print(f"  预计总量: {mem_est['total_gb']:.2f} GB")
    print(f"  GPU显存: {gpu_memory_gb:.1f} GB")
    
    if mem_est['total_gb'] > gpu_memory_gb * 0.9:
        print("\n?? 警告: 预估显存使用可能超过GPU容量!")
        print("  建议降低 batch_size 或启用更多优化选项")
    else:
        headroom = gpu_memory_gb - mem_est['total_gb']
        print(f"\n? 显存余量: {headroom:.2f} GB")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    # 测试配置系统
    print("测试配置系统...")
    
    # 获取中等显存预设
    config = get_preset_config("medium", gpu_memory_gb=24.0)
    
    # 打印配置摘要
    print_config_summary(config, gpu_memory_gb=24.0)
    
    # 保存配置
    save_config(config, "/tmp/test_config.yaml")
    print("配置已保存到 /tmp/test_config.yaml")
    
    # 加载配置
    loaded_config = load_config("/tmp/test_config.yaml")
    print("配置已加载")
    
    # 转换为命令行参数
    args = config_to_args(loaded_config)
    print("\n命令行参数:")
    for k, v in args.items():
        print(f"  --{k}={v}")
