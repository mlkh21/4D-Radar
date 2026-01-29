#!/bin/bash
# CD 推理脚本 - 一步快速生成

python scripts/inference.py \
    --vae_ckpt train_results/vae/best_model.pth \
    --model_ckpt train_results/cd/best_model.pth \
    --model_type cd \
    --steps 1 \
    --sampler euler \
    --num_samples 10 \
    --output_dir inference_results/cd \
    --device cuda
