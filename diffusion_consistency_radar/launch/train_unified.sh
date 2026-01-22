#!/bin/bash
# ==============================================================================
# 统一训练脚本 - VAE / LDM / CD 一站式训练
# ==============================================================================
#
# 使用方法:
#   sh diffusion_consistency_radar/launch/train_unified.sh vae      # 训练 VAE
#   sh diffusion_consistency_radar/launch/train_unified.sh ldm      # 训练 LDM
#   sh diffusion_consistency_radar/launch/train_unified.sh cd       # 蒸馏 CD
#   sh diffusion_consistency_radar/launch/train_unified.sh all      # 完整流程
#
# 配置文件: config/default_config.yaml
#
# ==============================================================================

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 默认路径
SCRIPT_DIR="diffusion_consistency_radar/scripts"
CONFIG_PATH="diffusion_consistency_radar/config/default_config.yaml"
DATASET_DIR="./NTU4DRadLM_pre_processing/NTU4DRadLM_Pre"
RESULTS_DIR="./diffusion_consistency_radar/train_results"

MODE=${1:-vae}

case "$MODE" in
    vae)
        echo "=========================================="
        echo "Stage 1: Training VAE"
        echo "=========================================="
        
        CUDA_VISIBLE_DEVICES=0,1 python ${SCRIPT_DIR}/unified_train.py \
            --mode vae \
            --config ${CONFIG_PATH}
        ;;
        
    ldm)
        echo "=========================================="
        echo "Stage 2: Training LDM"
        echo "=========================================="
        
        VAE_CKPT="${RESULTS_DIR}/vae/vae_best.pt"
        if [ ! -f "$VAE_CKPT" ]; then
            echo "Error: VAE checkpoint not found at $VAE_CKPT"
            echo "Please train VAE first: sh $0 vae"
            exit 1
        fi
        
        CUDA_VISIBLE_DEVICES=0,1 python ${SCRIPT_DIR}/unified_train.py \
            --mode ldm \
            --config ${CONFIG_PATH} \
            --vae_ckpt ${VAE_CKPT}
        ;;
        
    cd)
        echo "=========================================="
        echo "Stage 3: Consistency Distillation"
        echo "=========================================="
        
        VAE_CKPT="${RESULTS_DIR}/vae/vae_best.pt"
        LDM_CKPT="${RESULTS_DIR}/ldm/ldm_best.pt"
        
        if [ ! -f "$VAE_CKPT" ]; then
            echo "Error: VAE checkpoint not found at $VAE_CKPT"
            exit 1
        fi
        
        if [ ! -f "$LDM_CKPT" ]; then
            echo "Error: LDM checkpoint not found at $LDM_CKPT"
            exit 1
        fi
        
        CUDA_VISIBLE_DEVICES=0 python ${SCRIPT_DIR}/cd_train_optimized.py \
            --ldm_ckpt ${LDM_CKPT} \
            --vae_ckpt ${VAE_CKPT}
        ;;
        
    all)
        echo "=========================================="
        echo "Running Full Training Pipeline"
        echo "=========================================="
        
        # Stage 1: VAE
        sh $0 vae
        if [ $? -ne 0 ]; then exit 1; fi
        
        # Stage 2: LDM
        sh $0 ldm
        if [ $? -ne 0 ]; then exit 1; fi
        
        # Stage 3: CD
        sh $0 cd
        ;;
        
    *)
        echo "Usage: $0 [vae|ldm|cd|all]"
        echo ""
        echo "  vae  - Train VAE (Stage 1)"
        echo "  ldm  - Train Latent Diffusion (Stage 2)"
        echo "  cd   - Consistency Distillation (Stage 3)"
        echo "  all  - Run full pipeline"
        exit 1
        ;;
esac

echo "Done!"
