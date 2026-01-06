# 设置显存碎片管理
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python diffusion_consistency_radar/scripts/edm_train_radar.py \
    --config diffusion_consistency_radar/config/default_config.yaml \
    --gpu_id 0 \
    --use_fp16=True \
    --fp16_scale_growth=1e-3