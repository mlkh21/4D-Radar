#!/bin/bash
# ==============================================================================
# 优化版 EDM 训练脚本 - 针对 24GB 消费级显卡优化
# ==============================================================================
#
# 主要优化:
# 1. 使用 Flash Attention (PyTorch 2.0+) - 显存效率最优
# 2. 非对称下采样 - 保护 Z 轴分辨率
# 3. 通道瘦身 - 64 通道 + (1,2,2,4) 倍增
# 4. 梯度检查点 + 混合精度 - 大幅节省显存
# 5. 禁用高分辨率注意力 - 避免显存爆炸
#
# 使用方法:
#   sh diffusion_consistency_radar/launch/train_edm.sh
#
# 显存估计 (RTX 4090 24GB):
#   - batch_size=4: ~18GB
#   - batch_size=8: ~22GB
#
# ==============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# # 单卡训练
# python diffusion_consistency_radar/scripts/edm_train_radar.py \
#     --gpu_id 1 \
#     --batch_size 4 \
#     --dims 3 \
#     --image_size 128 \
#     --in_ch 8 \
#     --out_ch 4 \
#     \
#     `# === 模型优化参数 ===` \
#     --num_channels 64 \
#     --channel_mult 1,2,2,4 \
#     --num_res_blocks 2 \
#     --num_heads 4 \
#     --num_head_channels 64 \
#     \
#     `# === 注意力优化 ===` \
#     --attention_type flash \
#     --attention_resolutions 8,4 \
#     \
#     `# === 下采样优化 ===` \
#     --downsample_type asymmetric \
#     --downsample_stride xy_only \
#     \
#     `# === 归一化 ===` \
#     --norm_type group \
#     \
#     `# === 显存优化 ===` \
#     --use_fp16 True \
#     --use_checkpoint True \
#     --use_optimized_unet True \
#     \
#     `# === 训练参数 ===` \
#     --lr 0.0001 \
#     --dropout 0.1 \
#     --resblock_updown True \
#     --use_scale_shift_norm True \
#     \
#     `# === 其他 ===` \
#     --initial_z_size 32 \
#     --window_size 4,4,4

# ==============================================================================
# 备选配置
# ==============================================================================

# --- 超轻量配置 (12GB 显卡或更大 batch) ---
# python diffusion_consistency_radar/scripts/edm_train_radar.py \
#     --gpu_id 0 \
#     --batch_size 2 \
#     --num_channels 32 \
#     --channel_mult 1,2,2,2 \
#     --num_res_blocks 1 \
#     --attention_type linear \
#     --attention_resolutions 4 \
#     --norm_type instance \
#     --use_depthwise True \
#     --use_fp16 True \
#     --use_checkpoint True

# --- 多卡训练 (2x GPU) ---
# 注意: 需要确保有 2 张 GPU，每个进程使用独立的 GPU
# 如果只有 1 张 GPU，请使用单卡训练配置
mpirun -n 2 python diffusion_consistency_radar/scripts/edm_train_radar.py \
    --batch_size 2 \
    --num_channels 32 \
    --channel_mult 1,2,2,2 \
    --num_res_blocks 1 \
    --attention_type linear \
    --attention_resolutions 4 \
    --norm_type instance \
    --use_depthwise True \
    --use_fp16 True \
    --use_checkpoint True

# --- 单卡训练 (推荐，显存不足时) ---
# python diffusion_consistency_radar/scripts/edm_train_radar.py \
#     --gpu_id 0 \
#     --batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --num_channels 32 \
#     --channel_mult 1,2,2,2 \
#     --num_res_blocks 1 \
#     --attention_type linear \
#     --attention_resolutions 4 \
#     --norm_type instance \
#     --use_depthwise True \
#     --use_fp16 True \
#     --use_checkpoint True