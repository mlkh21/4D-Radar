# 设置显存碎片管理
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mpirun -n 2 python diffusion_consistency_radar/scripts/edm_train_radar.py \
    --gpu_id 2,3 \
    --use_fp16=True \
    --fp16_scale_growth=1e-3