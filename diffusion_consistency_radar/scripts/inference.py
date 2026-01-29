#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的推理脚本 - 使用训练好的 VAE/LDM/CD 模型生成雷达数据
"""

import argparse
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cm.vae_3d import VAE3D, create_ultra_lightweight_vae_config, create_lightweight_vae_config, create_standard_vae_config
from cm.unet_optimized import OptimizedUNetModel
from cm.karras_diffusion import KarrasDenoiser
from cm.dataset_loader import NTU4DRadLM_VoxelDataset
from torch.utils.data import DataLoader


class RadarGenerator:
    """雷达数据生成器"""
    
    def __init__(self, vae_path, model_path, model_type='ldm', device='cuda'):
        """
        Args:
            vae_path: VAE 模型路径
            model_path: LDM 或 CD 模型路径
            model_type: 'ldm' 或 'cd'
            device: 'cuda' 或 'cpu'
        """
        self.device = torch.device(device)
        self.model_type = model_type
        
        # 加载 VAE
        print(f"Loading VAE from {vae_path}...")
        self.vae = self._load_vae(vae_path)
        self.vae.eval()
        
        # 加载生成模型
        print(f"Loading {model_type.upper()} from {model_path}...")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 创建 Denoiser
        self.denoiser = KarrasDenoiser(
            sigma_data=0.5,
            sigma_max=80.0,
            sigma_min=0.002,
            loss_norm='l2',
        )
        
        print("Models loaded successfully!")
    
    def _load_vae(self, ckpt_path):
        """加载 VAE 模型"""
        # 尝试推断 VAE 配置
        vae_config = create_ultra_lightweight_vae_config()
        vae = VAE3D(**vae_config).to(self.device)
        
        ckpt = torch.load(ckpt_path, map_location=self.device)
        if 'model_state_dict' in ckpt:
            vae.load_state_dict(ckpt['model_state_dict'])
        else:
            vae.load_state_dict(ckpt)
        
        return vae
    
    def _load_model(self, ckpt_path):
        """加载 LDM 或 CD 模型"""
        model = OptimizedUNetModel(
            image_size=32,
            in_channels=8,
            model_channels=32,
            out_channels=4,
            num_res_blocks=1,
            attention_resolutions=(),
            channel_mult=(1, 2, 3),
            use_checkpoint=False,
            attention_type="linear",
        ).to(self.device)
        
        ckpt = torch.load(ckpt_path, map_location=self.device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        
        return model
    
    @torch.no_grad()
    def generate(self, condition, num_samples=1, steps=40, sampler='heun'):
        """
        生成雷达数据
        
        Args:
            condition: 条件数据 (B, 4, 32, 128, 128) 或 None
            num_samples: 生成样本数
            steps: 采样步数 (LDM: 40, CD: 1-4)
            sampler: 'heun' 或 'euler'
        
        Returns:
            generated: 生成的雷达数据 (B, 4, 32, 128, 128)
        """
        print(f"\nGenerating {num_samples} samples with {steps} steps ({sampler} sampler)...")
        
        # 编码条件到潜空间
        if condition is not None:
            condition = condition.to(self.device)
            z_cond = self.vae.get_latent(condition)
        else:
            # 无条件生成
            z_cond = torch.zeros(num_samples, 4, 8, 32, 32, device=self.device)
        
        # 初始化随机噪声
        z_shape = (num_samples, 4, 8, 32, 32)
        z_T = torch.randn(z_shape, device=self.device) * self.denoiser.sigma_max
        
        # 采样
        if self.model_type == 'cd' and steps == 1:
            # CD 一步生成
            z_0 = self._cd_sample(z_T, z_cond)
        else:
            # LDM 多步采样
            z_0 = self._ldm_sample(z_T, z_cond, steps, sampler)
        
        # 解码到原始空间
        generated = self.vae.decode(z_0)
        
        return generated
    
    def _cd_sample(self, z_T, z_cond):
        """CD 一步采样"""
        model_input = torch.cat([z_T, z_cond], dim=1)
        sigma = torch.ones(z_T.shape[0], device=self.device) * self.denoiser.sigma_max
        z_0 = self.model(model_input, sigma)
        return z_0
    
    def _ldm_sample(self, z_T, z_cond, steps, sampler):
        """LDM 多步采样 (Heun/Euler)"""
        # 生成时间步调度
        sigmas = self._get_sigmas(steps)
        z_t = z_T
        
        for i in tqdm(range(len(sigmas) - 1), desc="Sampling"):
            sigma_t = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # 预测噪声
            model_input = torch.cat([z_t, z_cond], dim=1)
            sigma_batch = torch.ones(z_t.shape[0], device=self.device) * sigma_t
            denoised = self.model(model_input, sigma_batch)
            
            # 计算导数
            d = (z_t - denoised) / sigma_t
            
            if sampler == 'heun' and i < len(sigmas) - 2:
                # Heun 二阶方法
                z_next = z_t + d * (sigma_next - sigma_t)
                
                # 二次预测
                model_input_2 = torch.cat([z_next, z_cond], dim=1)
                sigma_batch_2 = torch.ones(z_t.shape[0], device=self.device) * sigma_next
                denoised_2 = self.model(model_input_2, sigma_batch_2)
                d_2 = (z_next - denoised_2) / sigma_next
                
                # 校正
                z_t = z_t + (d + d_2) / 2 * (sigma_next - sigma_t)
            else:
                # Euler 一阶方法
                z_t = z_t + d * (sigma_next - sigma_t)
        
        return z_t
    
    def _get_sigmas(self, steps):
        """生成噪声水平调度"""
        rho = 7.0
        sigma_min = self.denoiser.sigma_min
        sigma_max = self.denoiser.sigma_max
        
        step_indices = torch.arange(steps + 1, device=self.device)
        t = step_indices / steps
        sigmas = (sigma_max ** (1 / rho) + t * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        
        return sigmas


def main():
    parser = argparse.ArgumentParser(description="Radar Data Inference")
    parser.add_argument("--vae_ckpt", type=str, required=True, help="Path to VAE checkpoint")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Path to LDM/CD checkpoint")
    parser.add_argument("--model_type", type=str, default="ldm", choices=["ldm", "cd"], help="Model type")
    parser.add_argument("--dataset_dir", type=str, default="./NTU4DRadLM_pre_processing/NTU4DRadLM_Pre", 
                        help="Dataset directory for condition data")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--steps", type=int, default=40, help="Sampling steps (LDM: 40, CD: 1-4)")
    parser.add_argument("--sampler", type=str, default="heun", choices=["heun", "euler"], help="Sampler type")
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--use_condition", action="store_true", help="Use condition data from dataset")
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = RadarGenerator(
        vae_path=args.vae_ckpt,
        model_path=args.model_ckpt,
        model_type=args.model_type,
        device=args.device,
    )
    
    # 准备条件数据
    if args.use_condition:
        print(f"Loading dataset from {args.dataset_dir}...")
        dataset = NTU4DRadLM_VoxelDataset(args.dataset_dir, split='val')
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        condition_data = next(iter(dataloader))[1]  # 获取条件
    else:
        condition_data = None
    
    # 生成数据
    generated = generator.generate(
        condition=condition_data,
        num_samples=args.num_samples,
        steps=args.steps,
        sampler=args.sampler,
    )
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.model_type}_samples_{args.steps}steps.npy")
    np.save(output_path, generated.cpu().numpy())
    print(f"\nSaved {args.num_samples} samples to {output_path}")
    print(f"Shape: {generated.shape}")
    print(f"Range: [{generated.min():.3f}, {generated.max():.3f}]")


if __name__ == "__main__":
    main()
