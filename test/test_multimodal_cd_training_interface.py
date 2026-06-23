#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion_consistency_radar.scripts.cd_train_optimized import (
    call_cd_denoiser,
    has_multimodal_state_dict,
)


class LegacyRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None
        self.last_t = None

    def forward(self, x, t):
        self.last_input = x
        self.last_t = t
        return x[:, :4]


class MultimodalRecorder(nn.Module):
    is_multimodal = True

    def __init__(self):
        super().__init__()
        self.last = {}

    def forward(self, radar_voxel, ir_img, r_mat, t_vec, k_mat, timesteps, noised_latent=None):
        self.last = {
            "radar_voxel": radar_voxel,
            "ir_img": ir_img,
            "r_mat": r_mat,
            "t_vec": t_vec,
            "k_mat": k_mat,
            "timesteps": timesteps,
            "noised_latent": noised_latent,
        }
        return noised_latent + 0.0


def _meta(batch_size=2):
    return {
        "ir_img": torch.randn(batch_size, 3, 16, 16),
        "r_mat": torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1),
        "t_vec": torch.zeros(batch_size, 3),
        "k_mat": torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1),
    }


def test_multimodal_checkpoint_detection():
    assert has_multimodal_state_dict({"ir_extractor.lateral_conv.weight": torch.empty(1)})
    assert has_multimodal_state_dict({"projection_layer.voxel_coords": torch.empty(1)})
    assert has_multimodal_state_dict({"unet_3d.input_blocks.0.0.weight": torch.empty(1)})
    assert not has_multimodal_state_dict({"input_blocks.0.0.weight": torch.empty(1)})


def test_legacy_cd_denoiser_keeps_eight_channel_path():
    model = LegacyRecorder()
    x_t = torch.randn(2, 4, 4, 4, 4)
    z_cond = torch.randn(2, 4, 4, 4, 4)
    t = torch.ones(2)

    out = call_cd_denoiser(model, x_t, z_cond, t)

    assert out.shape == x_t.shape
    assert model.last_input.shape[1] == 8
    assert torch.equal(model.last_t, t)


def test_multimodal_cd_denoiser_passes_radar_ir_and_noised_latent():
    model = MultimodalRecorder()
    x_t = torch.randn(2, 4, 4, 4, 4)
    z_cond = torch.randn(2, 4, 4, 4, 4)
    radar_voxel = torch.randn(2, 4, 4, 4, 4)
    t = torch.ones(2)
    meta = _meta()

    out = call_cd_denoiser(
        model,
        x_t,
        z_cond,
        t,
        radar_voxel=radar_voxel,
        meta_dict=meta,
    )

    assert out.shape == x_t.shape
    assert model.last["radar_voxel"] is radar_voxel
    assert model.last["ir_img"] is meta["ir_img"]
    assert model.last["noised_latent"] is x_t
    assert torch.equal(model.last["timesteps"], t)


if __name__ == "__main__":
    test_multimodal_checkpoint_detection()
    test_legacy_cd_denoiser_keeps_eight_channel_path()
    test_multimodal_cd_denoiser_passes_radar_ir_and_noised_latent()
    print("test_multimodal_cd_training_interface passed")
