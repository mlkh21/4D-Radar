# -*- coding: utf-8 -*-

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as tv_models
except Exception:
    tv_models = None


def heteroscedastic_gaussian_nll(
    prediction: torch.Tensor,
    target: torch.Tensor,
    variance: torch.Tensor,
    detach_residual: bool = False,
) -> torch.Tensor:
    """Gaussian NLL for a per-voxel variance prediction."""
    residual_sq = (prediction - target).pow(2).mean(dim=1, keepdim=True)
    if detach_residual:
        residual_sq = residual_sq.detach()
    if variance.shape[-3:] != residual_sq.shape[-3:]:
        variance = F.interpolate(variance, size=residual_sq.shape[-3:], mode="trilinear", align_corners=False)
    variance = variance.clamp(1e-5, 50.0)
    return 0.5 * (residual_sq / variance + torch.log(variance)).mean()


class IR2DFeatureExtractor(nn.Module):
    """2D infrared feature extractor with a torchvision-free fallback."""

    def __init__(self, out_channels: int = 32, use_resnet18: bool = True):
        super().__init__()
        self.out_channels = out_channels
        if use_resnet18 and tv_models is not None:
            try:
                resnet = tv_models.resnet18(weights=None)
            except TypeError:
                resnet = tv_models.resnet18(pretrained=False)
            self.backbone = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
            )
            self.lateral_conv = nn.Conv2d(128, out_channels, kernel_size=1)
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.lateral_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lateral_conv(self.backbone(x))


class RadarStructureEncoder(nn.Module):
    """Encode radar occupancy, intensity, Doppler, and variance into condition features."""

    def __init__(self, in_channels: int = 4, out_channels: int = 16, hidden_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, min(4, hidden_channels)), num_channels=hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=max(1, min(4, out_channels)), num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, radar_voxel: torch.Tensor) -> torch.Tensor:
        return self.net(radar_voxel)


class UncertaintyHead(nn.Module):
    """Physics-first confidence head driven by Doppler variance and optional metadata."""

    def __init__(self, variance_scale: float = 10.0, min_confidence: float = 0.02):
        super().__init__()
        self.variance_scale = float(variance_scale)
        self.min_confidence = float(min_confidence)

    def forward(
        self,
        radar_voxel: torch.Tensor,
        target_shape: Tuple[int, int, int] = None,
        odom_cov_trace: torch.Tensor = None,
        is_mock_ir: torch.Tensor = None,
        is_mock_calib: torch.Tensor = None,
    ) -> dict:
        if radar_voxel.shape[1] <= 3:
            variance = torch.zeros_like(radar_voxel[:, :1])
        else:
            variance = radar_voxel[:, 3:4].clamp(0.0, 50.0)
        if target_shape is not None and variance.shape[-3:] != tuple(target_shape):
            variance = F.interpolate(variance, size=target_shape, mode="trilinear", align_corners=False)

        physical_variance = variance / max(self.variance_scale, 1e-6)
        bsz = physical_variance.shape[0]
        device = physical_variance.device
        dtype = physical_variance.dtype

        if odom_cov_trace is not None:
            odom_cov_trace = torch.as_tensor(odom_cov_trace, device=device, dtype=dtype).view(bsz, 1, 1, 1, 1)
            physical_variance = physical_variance + 0.35 * odom_cov_trace.clamp_min(0.0)
        if is_mock_ir is not None:
            mock_ir = torch.as_tensor(is_mock_ir, device=device, dtype=dtype).view(bsz, 1, 1, 1, 1)
            physical_variance = physical_variance + 0.10 * mock_ir.clamp(0.0, 1.0)
        if is_mock_calib is not None:
            mock_calib = torch.as_tensor(is_mock_calib, device=device, dtype=dtype).view(bsz, 1, 1, 1, 1)
            physical_variance = physical_variance + 0.20 * mock_calib.clamp(0.0, 1.0)

        physical_variance = physical_variance.clamp(0.0, 50.0)
        confidence = 1.0 / (1.0 + physical_variance)
        confidence = confidence.clamp(self.min_confidence, 1.0)
        log_var = torch.log(physical_variance.clamp_min(1e-6))
        return {"confidence": confidence, "variance": physical_variance, "log_var": log_var}


class DualModalityProjectionLayer(nn.Module):
    """Project 2D IR features onto a 3D voxel grid with frustum masking."""

    def __init__(
        self,
        voxel_shape: Tuple[int, int, int] = (600, 200, 80),
        pc_range: Sequence[float] = (0, -20, -6, 120, 20, 10),
    ):
        super().__init__()
        self.voxel_shape = tuple(int(v) for v in voxel_shape)
        self.pc_range = tuple(float(v) for v in pc_range)
        nx, ny, nz = self.voxel_shape
        x_size = (self.pc_range[3] - self.pc_range[0]) / float(nx)
        y_size = (self.pc_range[4] - self.pc_range[1]) / float(ny)
        z_size = (self.pc_range[5] - self.pc_range[2]) / float(nz)
        xs = self.pc_range[0] + (torch.arange(nx, dtype=torch.float32) + 0.5) * x_size
        ys = self.pc_range[1] + (torch.arange(ny, dtype=torch.float32) + 0.5) * y_size
        zs = self.pc_range[2] + (torch.arange(nz, dtype=torch.float32) + 0.5) * z_size
        grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
        coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
        self.register_buffer("voxel_coords", coords, persistent=False)

    def forward(
        self,
        ir_features: torch.Tensor,
        r_mat: torch.Tensor,
        t_vec: torch.Tensor,
        k_mat: torch.Tensor,
        img_shape: Tuple[int, int],
    ) -> torch.Tensor:
        bsz, channels, _, _ = ir_features.shape
        h_img, w_img = img_shape
        nx, ny, nz = self.voxel_shape

        pts_3d = self.voxel_coords.to(ir_features.device, ir_features.dtype).unsqueeze(0).expand(bsz, -1, -1)
        r_mat = r_mat.to(ir_features.device, ir_features.dtype)
        t_vec = t_vec.to(ir_features.device, ir_features.dtype)
        k_mat = k_mat.to(ir_features.device, ir_features.dtype)

        r_inv = r_mat.transpose(-1, -2)
        pts_cam = torch.bmm(pts_3d - t_vec.unsqueeze(1), r_inv.transpose(-1, -2))
        depth = pts_cam[..., 2].clamp_min(1e-6)
        pts_pixel = torch.bmm(pts_cam, k_mat.transpose(-1, -2))
        uv = pts_pixel[..., :2] / depth.unsqueeze(-1)
        u_norm = (uv[..., 0] / max(float(w_img - 1), 1.0)) * 2.0 - 1.0
        v_norm = (uv[..., 1] / max(float(h_img - 1), 1.0)) * 2.0 - 1.0
        sample_grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(2)

        sampled = F.grid_sample(
            ir_features,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(-1)
        voxel_ir = sampled.view(bsz, channels, nx, ny, nz)
        frustum_mask = (
            (pts_cam[..., 2] > 0.1)
            & (u_norm >= -1.0)
            & (u_norm <= 1.0)
            & (v_norm >= -1.0)
            & (v_norm <= 1.0)
        ).view(bsz, 1, nx, ny, nz)
        return voxel_ir * frustum_mask.to(voxel_ir.dtype)


class CompleteDualModalityPerceptionNet(nn.Module):
    """Fuse radar voxel features and projected IR features before a 3D backbone."""

    def __init__(
        self,
        unet_3d_backbone: nn.Module,
        voxel_shape: Tuple[int, int, int] = (600, 200, 80),
        pc_range: Sequence[float] = (0, -20, -6, 120, 20, 10),
        ir_channels: int = 32,
        fused_channels: int = 16,
        downsample_to_latent: bool = False,
        latent_shape: Tuple[int, int, int] = (32, 128, 128),
    ):
        super().__init__()
        self.unet_3d = unet_3d_backbone
        self.is_multimodal = True
        self.ir_extractor = IR2DFeatureExtractor(out_channels=ir_channels)
        self.projection_layer = DualModalityProjectionLayer(voxel_shape=voxel_shape, pc_range=pc_range)
        self.in_channels = fused_channels
        self.radar_encoder = RadarStructureEncoder(in_channels=4, out_channels=fused_channels)
        self.uncertainty_head = UncertaintyHead()
        self.ir_gate = nn.Sequential(
            nn.Conv3d(fused_channels + 1, ir_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(fused_channels + ir_channels + 1, fused_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        uncertainty_hidden = max(4, fused_channels // 2)
        self.model_uncertainty_head = nn.Sequential(
            nn.Conv3d(fused_channels, uncertainty_hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(uncertainty_hidden, 1, kernel_size=1),
        )
        nn.init.zeros_(self.model_uncertainty_head[-1].weight)
        nn.init.constant_(self.model_uncertainty_head[-1].bias, -3.0)
        self.downsample_to_latent = downsample_to_latent
        self.latent_shape = tuple(int(v) for v in latent_shape)

    def forward(
        self,
        radar_voxel: torch.Tensor,
        ir_img: torch.Tensor,
        r_mat: torch.Tensor,
        t_vec: torch.Tensor,
        k_mat: torch.Tensor,
        timesteps: torch.Tensor,
        noised_latent: torch.Tensor = None,
        odom_cov_trace: torch.Tensor = None,
        is_mock_ir: torch.Tensor = None,
        is_mock_calib: torch.Tensor = None,
        return_uncertainty: bool = False,
    ) -> torch.Tensor:
        _, _, h_img, w_img = ir_img.shape
        ir_feat_2d = self.ir_extractor(ir_img)
        ir_feat_3d = self.projection_layer(ir_feat_2d, r_mat, t_vec, k_mat, (h_img, w_img))
        if ir_feat_3d.shape[-3:] != radar_voxel.shape[-3:]:
            ir_feat_3d = F.interpolate(
                ir_feat_3d,
                size=radar_voxel.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
        radar_cond = self.radar_encoder(radar_voxel)
        uncertainty = self.uncertainty_head(
            radar_voxel,
            target_shape=radar_cond.shape[-3:],
            odom_cov_trace=odom_cov_trace,
            is_mock_ir=is_mock_ir,
            is_mock_calib=is_mock_calib,
        )
        ir_gate = self.ir_gate(torch.cat([radar_cond, uncertainty["confidence"]], dim=1))
        ir_feat_3d = ir_feat_3d * ir_gate
        fused = self.fusion_conv(torch.cat([radar_cond, ir_feat_3d, uncertainty["confidence"]], dim=1))
        if noised_latent is not None:
            noised_latent = noised_latent.to(fused.device, fused.dtype)
            if self.downsample_to_latent and fused.shape[-3:] != noised_latent.shape[-3:]:
                fused = F.interpolate(
                    fused,
                    size=noised_latent.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
            if noised_latent.shape[-3:] != fused.shape[-3:]:
                noised_latent = F.interpolate(
                    noised_latent,
                    size=fused.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )
            channels = min(noised_latent.shape[1], fused.shape[1])
            fused = fused.clone()
            fused[:, :channels] = fused[:, :channels] + noised_latent[:, :channels]
        elif self.downsample_to_latent and fused.shape[-3:] != self.latent_shape:
            fused = F.interpolate(fused, size=self.latent_shape, mode="trilinear", align_corners=False)
        physical_variance = uncertainty["variance"]
        if physical_variance.shape[-3:] != fused.shape[-3:]:
            physical_variance = F.interpolate(
                physical_variance,
                size=fused.shape[-3:],
                mode="trilinear",
                align_corners=False,
            )
        learned_variance = F.softplus(self.model_uncertainty_head(fused)) + 1e-5
        total_variance = (physical_variance + learned_variance).clamp(1e-5, 50.0)
        uncertainty = {
            "confidence": 1.0 / (1.0 + total_variance),
            "variance": total_variance,
            "log_var": torch.log(total_variance),
            "physical_variance": physical_variance,
            "learned_variance": learned_variance,
        }
        out = self.unet_3d(fused, timesteps)
        if return_uncertainty:
            if uncertainty["confidence"].shape[-3:] != out.shape[-3:]:
                uncertainty = {
                    key: F.interpolate(value, size=out.shape[-3:], mode="trilinear", align_corners=False)
                    for key, value in uncertainty.items()
                }
            return out, uncertainty
        return out
