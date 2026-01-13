#!/usr/bin/env python3
"""
配置验证脚本

检查:
1. 模块导入是否成功
2. 超轻量级配置是否定义正确
3. 显存估算 (如果可能)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("配置验证脚本")
print("="*70)

# 1. 检查导入
print("\n[1/4] 检查模块导入...")
try:
    from diffusion_consistency_radar.cm.vae_3d import (
        VAE3D,
        create_ultra_lightweight_vae_config,
        create_lightweight_vae_config,
        create_standard_vae_config,
    )
    print("? VAE模块导入成功")
except Exception as e:
    print(f"? VAE模块导入失败: {e}")
    sys.exit(1)

try:
    from diffusion_consistency_radar.cm.dataset_loader import NTU4DRadLM_VoxelDataset
    print("? 数据加载器导入成功")
except Exception as e:
    print(f"? 数据加载器导入失败: {e}")

# 2. 检查配置
print("\n[2/4] 检查配置定义...")

configs = {
    "ultra_lightweight": create_ultra_lightweight_vae_config(),
    "lightweight": create_lightweight_vae_config(),
    "standard": create_standard_vae_config(),
}

for name, config in configs.items():
    print(f"\n  {name}:")
    print(f"    base_channels: {config.get('base_channels')}")
    print(f"    channel_mult: {config.get('encoder_channel_mult')}")
    print(f"    num_res_blocks: {config.get('num_res_blocks')}")
    print(f"    use_checkpoint: {config.get('use_checkpoint')}")

# 3. 创建模型并计算参数量
print("\n[3/4] 创建模型并计算参数量...")

try:
    for name, config in configs.items():
        model = VAE3D(**config)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {name:20s}: {params/1e6:6.2f}M 参数")
    print("? 模型创建成功")
except Exception as e:
    print(f"? 模型创建失败: {e}")
    sys.exit(1)

# 4. 检查GPU和显存
print("\n[4/4] 检查GPU和显存...")

try:
    import torch
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"? 检测到 {device_count} 个GPU")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / 1024**3
            print(f"  GPU {i}: {props.name} ({total_mem:.1f} GB)")
            
            # 获取可用显存
            free_mem = torch.cuda.mem_get_info(i)[0] / 1024**3
            used_mem = total_mem - free_mem
            print(f"    已用: {used_mem:.1f} GB / 可用: {free_mem:.1f} GB")
    else:
        print("? 未检测到GPU,CPU模式")
        
except Exception as e:
    print(f"? GPU检查失败: {e}")

# 5. 显存需求估算
print("\n" + "="*70)
print("显存需求估算 (训练模式)")
print("="*70)

est = {
    "ultra_lightweight": "6-9 GB",
    "lightweight": "12-16 GB",
    "standard": "20-25 GB",
}

for name, memory in est.items():
    status = "?" if name == "ultra_lightweight" else ("?" if name == "lightweight" else "?")
    print(f"{status} {name:20s}: {memory}")

print("\n" + "="*70)
print("? 验证完成!")
print("="*70)
print("\n推荐使用超轻量级配置训练:")
print("sh diffusion_consistency_radar/launch/train_latent_diffusion.sh vae")
print()
