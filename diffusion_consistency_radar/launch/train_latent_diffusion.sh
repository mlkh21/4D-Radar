#!/bin/bash
# ==============================================================================
# Latent Diffusion 训练脚本
# ==============================================================================
#
# 这是终极优化方案，分两阶段训练：
#
# 阶段 1: 训练 VAE
#   将 3D 体素 (4, 32, 128, 128) 压缩到潜空间 (4, 8, 32, 32)
#   压缩比: 4x4x4 = 64 倍
#
# 阶段 2: 训练 Latent Diffusion
#   在潜空间中进行扩散训练
#   显存使用降低 8-16 倍
#
# 使用方法:
#   1. 训练 VAE:
#      CUDA_VISIBLE_DEVICES=0,1 sh diffusion_consistency_radar/launch/train_latent_diffusion.sh vae
#
#   2. VAE预编码:
#      CUDA_VISIBLE_DEVICES=0 python diffusion_consistency_radar/scripts/precompute_latents.py
# 
#   3. 训练 Latent Diffusion:
#      sh diffusion_consistency_radar/launch/train_latent_diffusion.sh ldm
#
# ==============================================================================

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null

MODE=${1:-vae}  # 默认训练 VAE
VAE_TYPE=${2:-ultra_lightweight}  # 默认使用超轻量级配置

if [ "$MODE" = "vae" ]; then
    echo "=========================================="
    echo "Stage 1: Training 3D VAE"
    echo "VAE Type: $VAE_TYPE"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=0,1 python diffusion_consistency_radar/scripts/train_latent_diffusion.py \
        --mode train_vae \
        --vae_type $VAE_TYPE \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --vae_epochs 100 \
        --lr 1e-4 \
        --kl_weight 1e-6 \
        --save_every 10 \
        --num_workers 16 \
        --resume diffusion_consistency_radar/train_results/vae/vae_best.pt \
        --dataset_dir ./NTU4DRadLM_pre_processing/NTU4DRadLM_Pre \
        --vae_save_dir ./diffusion_consistency_radar/train_results/vae

elif [ "$MODE" = "ldm" ]; then
    echo "=========================================="
    echo "Stage 2: Training Latent Diffusion"
    echo "=========================================="
    
    # 确保已有训练好的 VAE
    VAE_CKPT="./diffusion_consistency_radar/train_results/vae/vae_best.pt"
    
    if [ ! -f "$VAE_CKPT" ]; then
        echo "Error: VAE checkpoint not found at $VAE_CKPT"
        echo "Please train VAE first: sh $0 vae"
        exit 1
    fi
    
    CUDA_VISIBLE_DEVICES=0,1 python diffusion_consistency_radar/scripts/train_latent_diffusion.py \
        --mode train_ldm \
        --vae_type ultra_lightweight \
        --vae_ckpt $VAE_CKPT \
        --batch_size 1 \
        --gradient_accumulation_steps 16 \
        --ldm_epochs 200 \
        --model_channels 32 \
        --lr 1e-4 \
        --save_every 5000 \
        --num_workers 2 \
        --dataset_dir ./NTU4DRadLM_pre_processing/NTU4DRadLM_Pre \
        --ldm_save_dir ./diffusion_consistency_radar/train_results/ldm

else
    echo "Usage: $0 [vae|ldm]"
    echo "  vae - Train VAE (Stage 1)"
    echo "  ldm - Train Latent Diffusion (Stage 2)"
    exit 1
fi

echo "Training completed!"
