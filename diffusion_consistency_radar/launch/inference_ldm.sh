#!/bin/bash
# LDM 推理脚本 - 使用 Karras 采样生成高质量雷达数据

python scripts/inference.py \
    --vae_ckpt train_results/vae/best_model.pth \
    --model_ckpt train_results/ldm/best_model.pth \
    --model_type ldm \
    --steps 40 \
    --sampler heun \
    --num_samples 10 \
    --output_dir inference_results/ldm \
    --device cuda
