#!/usr/bin/env python3
"""
快速参考: 显存优化配置对比

说明: 这个脚本展示不同配置的显存使用对比
"""

import sys

CONFIG_COMPARISON = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    VAE 训练显存优化配置对比                                 ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─ 标准配置 (不推荐 - OOM) ────────────────────────────────────────────────────┐
│ batch_size:              8                                                   │
│ gradient_accumulation:   1                                                   │
│ vae_type:                standard                                             │
│ use_amp:                 False                                                │
│ use_checkpoint:          False                                                │
│ num_workers:             4                                                   │
│ pin_memory:              True                                                 │
│ ────────────────────────────────────────────────────────────────────────────  │
│ 参数量:                  ~7.65M                                              │
│ 预计显存:                20-25 GB ? OOM                                       │
└────────────────────────────────────────────────────────────────────────────────┘

┌─ 轻量级配置 (可能OOM) ────────────────────────────────────────────────────────┐
│ batch_size:              2                                                   │
│ gradient_accumulation:   4                                                   │
│ vae_type:                lightweight                                          │
│ use_amp:                 True                                                 │
│ use_checkpoint:          True                                                 │
│ num_workers:             2                                                   │
│ pin_memory:              False                                                │
│ ────────────────────────────────────────────────────────────────────────────  │
│ 参数量:                  ~1.5M                                               │
│ 预计显存:                14-18 GB ? 可能OOM                                   │
└────────────────────────────────────────────────────────────────────────────────┘

┌─ 超轻量级配置 (推荐) ────────────────────────────────────────────────────────┐
│ batch_size:              1                                                   │
│ gradient_accumulation:   8                                                   │
│ vae_type:                ultra_lightweight  ← 推荐                          │
│ use_amp:                 True                                                 │
│ use_checkpoint:          True                                                 │
│ num_workers:             1                                                   │
│ pin_memory:              False                                                │
│ ────────────────────────────────────────────────────────────────────────────  │
│ 参数量:                  ~0.69M                                              │
│ 预计显存:                6-9 GB ? 可行                                        │
│ 有效batch size:          8 (相同梯度噪声)                                      │
└────────────────────────────────────────────────────────────────────────────────┘

╔════════════════════════════════════════════════════════════════════════════╗
║                         关键优化技巧                                         ║
╚════════════════════════════════════════════════════════════════════════════╝

1??  超轻量级配置 (90% 参数减少)
    base_channels: 32 → 16
    channel_mult: (1,2,4) → (1,2,2)
    num_res_blocks: 2 → 1
    use_attention: True → False

2??  批次大小优化 (8x 激活显存减少)
    batch_size: 8 → 1
    + gradient_accumulation: 1 → 8
    = 有效batch size: 8 (梯度噪声相同)

3??  梯度检查点 (60% 激活显存减少)
    --use_checkpoint in ResBlock, Encoder, Decoder
    权衡: 计算时间 +20%,显存 -60%

4??  混合精度训练 (40% 激活显存减少)
    --use_amp 启用
    权衡: 精度略低,显存节省显著

5??  数据加载优化 (2GB主机内存节省)
    num_workers: 4 → 1
    pin_memory: True → False
    use_augmentation: True → False (VAE)

╔════════════════════════════════════════════════════════════════════════════╗
║                         快速命令                                           ║
╚════════════════════════════════════════════════════════════════════════════╝

? 推荐 - 使用超轻量级配置训练 VAE:
  $ sh diffusion_consistency_radar/launch/train_latent_diffusion.sh vae

? 完整命令 - 带所有参数:
  $ python diffusion_consistency_radar/scripts/train_latent_diffusion.py \\
      --mode train_vae \\
      --vae_type ultra_lightweight \\
      --batch_size 1 \\
      --gradient_accumulation_steps 8 \\
      --use_amp \\
      --vae_epochs 100 \\
      --dataset_dir ./NTU4DRadLM_pre_processing/NTU4DRadLM_Pre

? 其他配置:
  - 轻量级: --vae_type lightweight
  - 标准:   --vae_type standard
  - 不使用AMP: 移除 --use_amp

╔════════════════════════════════════════════════════════════════════════════╗
║                      显存使用估算                                           ║
╚════════════════════════════════════════════════════════════════════════════╝

推理 (inference):
  - 超轻量级: ~0.8 GB
  - 轻量级:   ~1.5 GB
  - 标准:     ~3.0 GB

训练 (training, with gradients + optimizer):
  - 超轻量级 + AMP:   ~6-9 GB
  - 超轻量级 + FP32:  ~9-12 GB
  - 轻量级 + AMP:     ~12-16 GB
  - 标准 + AMP:       ~18-24 GB

╔════════════════════════════════════════════════════════════════════════════╗
║                         环境变量                                           ║
╚════════════════════════════════════════════════════════════════════════════╝

优化CUDA内存分配:
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

禁用cuDNN自动搜索(加速):
  export CUDNN_BENCHMARK=1

设置优先级:
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=0

╔════════════════════════════════════════════════════════════════════════════╗
║                    故障排除 (如果仍然OOM)                                   ║
╚════════════════════════════════════════════════════════════════════════════╝

1. 再次减小batch_size:
   --batch_size 1 --gradient_accumulation_steps 16

2. 禁用pin_memory:
   默认已禁用

3. 更激进的配置:
   base_channels: 16 → 8

4. 使用VQ-VAE (不需要KL loss):
   --vae_type vqvae (尚未实现)

5. 检查GPU:
   nvidia-smi -l 1  # 实时监控

╔════════════════════════════════════════════════════════════════════════════╗
║                        性能影响                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

配置                      速度    显存   质量    综合得分
────────────────────────────────────────────────────────
标准                      快     ?OOM  最好    ? 不可用
轻量级                    中     ?     良     ? 可能失败
超轻量级 ← 推荐           慢    ?好   一般    ? 可行

注: 速度下降主要因为:
  1. batch_size 从8减到1 (~8x减速)
  2. 梯度检查点 (~1.2x减速)
  3. 较小模型优化较差 (~1.1x减速)
  总体: ~10x减速,但显存减少20x+

一旦VAE训练完成,Latent Diffusion可以使用更大配置!

"""

if __name__ == "__main__":
    print(CONFIG_COMPARISON)
    
    # 提示
    print("\n" + "="*80)
    print("? 提示: 复制推荐命令并粘贴到终端运行:")
    print("="*80)
    print("sh diffusion_consistency_radar/launch/train_latent_diffusion.sh vae")
    print("="*80 + "\n")
