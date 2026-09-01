# -*- coding: utf-8 -*-
"""CD online/EMA 验证与部署权重选择的共享协议常量。"""

CD_VALIDATION_PROTOCOL = "cd_online_ema_denoising_validation_v1"
CD_VALIDATION_SELECTOR = (
    "max_observed_iou_then_min_observed_latent_loss_prefer_ema_v1"
)
CD_VALIDATION_SPLIT = "temporal_block_validation_suffix"
