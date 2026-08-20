#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffusion_consistency_radar.scripts.cd_train_optimized import (
    build_cd_vae_from_checkpoint,
    call_cd_denoiser,
    encode_cd_training_latents,
    has_multimodal_state_dict,
    create_cd_model,
)
from diffusion_consistency_radar.cm.vae_3d import (
    VAE3D,
    create_lightweight_vae_config,
    create_ultra_lightweight_vae_config,
)


def test_cd_vae_checkpoint_metadata_precedes_fallback_config():
    config = create_lightweight_vae_config()
    config["latent_dim"] = 8
    config["base_channels"] = 32
    checkpoint = {
        "model_state_dict": VAE3D(**config).state_dict(),
        "vae_config": config,
        "vae_config_type": "lightweight",
    }

    model, metadata = build_cd_vae_from_checkpoint(
        checkpoint,
        fallback_config_type="ultra_lightweight",
    )

    assert model.latent_dim == 8
    assert metadata["vae_config_type"] == "lightweight"


def test_cd_legacy_vae_requires_explicit_fallback():
    config = create_ultra_lightweight_vae_config()
    checkpoint = {"model_state_dict": VAE3D(**config).state_dict()}

    try:
        build_cd_vae_from_checkpoint(checkpoint, fallback_config_type=None)
    except ValueError as exc:
        assert "fallback" in str(exc)
    else:
        raise AssertionError("legacy checkpoint without fallback must fail")


def test_cd_z8_legacy_model_has_dynamic_input_and_output_channels():
    model = create_cd_model(
        False,
        {
            "latent_dim": 8,
            "model_channels": 8,
            "channel_mult": [1],
            "use_checkpoint": False,
        },
    )

    output = model(torch.randn(1, 16, 2, 4, 4), torch.ones(1))

    assert model.in_channels == 16
    assert model.out_channels == 8
    assert output.shape == (1, 8, 2, 4, 4)


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


def test_multimodal_cd_encodes_only_target_but_legacy_encodes_condition():
    class CountingVAE:
        def __init__(self):
            self.inputs = []

        def get_latent(self, value):
            self.inputs.append(value)
            return value[:, :1]

    target = torch.randn(1, 4, 2, 2, 2)
    condition = torch.randn_like(target)
    vae = CountingVAE()

    z_target, z_cond = encode_cd_training_latents(
        vae,
        target,
        condition,
        _meta(batch_size=1),
    )
    assert z_target.shape[0] == 1
    assert z_cond is None
    assert vae.inputs == [target]

    vae = CountingVAE()
    _z_target, z_cond = encode_cd_training_latents(vae, target, condition, {})
    assert z_cond is not None
    assert vae.inputs == [target, condition]


if __name__ == "__main__":
    test_cd_vae_checkpoint_metadata_precedes_fallback_config()
    test_cd_legacy_vae_requires_explicit_fallback()
    test_cd_z8_legacy_model_has_dynamic_input_and_output_channels()
    test_multimodal_checkpoint_detection()
    test_legacy_cd_denoiser_keeps_eight_channel_path()
    test_multimodal_cd_denoiser_passes_radar_ir_and_noised_latent()
    test_multimodal_cd_encodes_only_target_but_legacy_encodes_condition()
    print("test_multimodal_cd_training_interface passed")
