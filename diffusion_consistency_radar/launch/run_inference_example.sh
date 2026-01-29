#!/bin/bash
# 完整推理示例 - 演示如何使用 LDM 和 CD 模型

set -e  # 遇到错误立即退出

echo "=========================================="
echo "4D Radar 推理示例"
echo "=========================================="

# 检查模型是否存在
VAE_CKPT="diffusion_consistency_radar/train_results/vae/vae_best.pt"
LDM_CKPT="diffusion_consistency_radar/train_results/ldm/ldm_best.pt"
CD_CKPT="diffusion_consistency_radar/train_results/cd/cd_best.pt"

if [ ! -f "$VAE_CKPT" ]; then
    echo "错误: VAE 模型不存在: $VAE_CKPT"
    echo "请先运行: bash launch/train_unified.sh vae"
    exit 1
fi

if [ ! -f "$LDM_CKPT" ]; then
    echo "警告: LDM 模型不存在: $LDM_CKPT"
    echo "跳过 LDM 推理..."
    RUN_LDM=false
else
    RUN_LDM=true
fi

if [ ! -f "$CD_CKPT" ]; then
    echo "警告: CD 模型不存在: $CD_CKPT"
    echo "跳过 CD 推理..."
    RUN_CD=false
else
    RUN_CD=true
fi

# LDM 推理
if [ "$RUN_LDM" = true ]; then
    echo ""
    echo "=========================================="
    echo "1. LDM 推理 (40步 Heun 采样)"
    echo "=========================================="
    
    python diffusion_consistency_radar/scripts/inference.py \
        --vae_ckpt "$VAE_CKPT" \
        --model_ckpt "$LDM_CKPT" \
        --model_type ldm \
        --steps 40 \
        --sampler heun \
        --num_samples 10 \
        --output_dir diffusion_consistency_radar/inference_results/ldm \
        --device cuda
    
    echo "✓ LDM 推理完成"
fi

# CD 推理
if [ "$RUN_CD" = true ]; then
    echo ""
    echo "=========================================="
    echo "2. CD 推理 (1步快速生成)"
    echo "=========================================="
    
    python diffusion_consistency_radar/scripts/inference.py \
        --vae_ckpt "$VAE_CKPT" \
        --model_ckpt "$CD_CKPT" \
        --model_type cd \
        --steps 1 \
        --sampler euler \
        --num_samples 10 \
        --output_dir diffusion_consistency_radar/inference_results/cd \
        --device cuda
    
    echo "✓ CD 推理完成"
    
    # CD 4步推理（提升质量）
    echo ""
    echo "=========================================="
    echo "3. CD 推理 (4步高质量生成)"
    echo "=========================================="
    
    python diffusion_consistency_radar/scripts/inference.py \
        --vae_ckpt "$VAE_CKPT" \
        --model_ckpt "$CD_CKPT" \
        --model_type cd \
        --steps 4 \
        --sampler euler \
        --num_samples 10 \
        --output_dir diffusion_consistency_radar/inference_results/cd_4step \
        --device cuda
    
    echo "✓ CD 4步推理完成"
fi

# 可视化
if [ "$RUN_LDM" = true ] && [ "$RUN_CD" = true ]; then
    echo ""
    echo "=========================================="
    echo "4. 可视化生成结果"
    echo "=========================================="
    
    # LDM 可视化
    if [ -f "diffusion_consistency_radar/inference_results/ldm/ldm_samples_40steps.npy" ]; then
        python diffusion_consistency_radar/scripts/visualize_results.py \
            --input diffusion_consistency_radar/inference_results/ldm/ldm_samples_40steps.npy \
            --output_dir diffusion_consistency_radar/visualizations/ldm \
            --num_samples 5
        echo "✓ LDM 可视化完成: diffusion_consistency_radar/visualizations/ldm/"
    fi
    
    # CD 可视化
    if [ -f "diffusion_consistency_radar/inference_results/cd/cd_samples_1steps.npy" ]; then
        python diffusion_consistency_radar/scripts/visualize_results.py \
            --input diffusion_consistency_radar/inference_results/cd/cd_samples_1steps.npy \
            --output_dir diffusion_consistency_radar/visualizations/cd \
            --num_samples 5
        echo "✓ CD 可视化完成: diffusion_consistency_radar/visualizations/cd/"
    fi
    
    # 对比可视化
    if [ -f "diffusion_consistency_radar/inference_results/ldm/ldm_samples_40steps.npy" ] && [ -f "diffusion_consistency_radar/inference_results/cd/cd_samples_1steps.npy" ]; then
        python diffusion_consistency_radar/scripts/visualize_results.py \
            --input diffusion_consistency_radar/inference_results/ldm/ldm_samples_40steps.npy \
            --compare diffusion_consistency_radar/inference_results/cd/cd_samples_1steps.npy \
            --output_dir diffusion_consistency_radar/visualizations/comparison
        echo "✓ 对比可视化完成: diffusion_consistency_radar/visualizations/comparison/"
    fi
fi

echo ""
echo "=========================================="
echo "推理完成！"
echo "=========================================="
echo "生成数据保存在: diffusion_consistency_radar/inference_results/"
echo "可视化结果保存在: diffusion_consistency_radar/visualizations/"
echo ""
