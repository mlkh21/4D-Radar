# -*- coding: utf-8 -*-
"""
优化的 Consistency Distillation 训练脚本

改进点：
1. 清晰的蒸馏流程 - 从 LDM 教师蒸馏 CD 学生模型
2. 显存优化 - 梯度累积、检查点、混合精度
3. 规范的实现 - 遵循 Consistency Distillation 论文
4. 模块化设计 - 易于理解和维护
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
import argparse
from typing import Tuple
from tqdm import tqdm

from cm.unet_optimized import OptimizedUNetModel
from cm.karras_diffusion import KarrasDenoiser
from cm.vae_3d import VAE3D, create_ultra_lightweight_vae_config
from cm.dataset_loader import NTU4DRadLM_VoxelDataset


class ConsistencyDistillationTrainer:
    """
    Consistency Distillation 训练器
    
    流程：
    1. 加载预训练的 LDM 教师模型
    2. 创建 CD 学生模型
    3. 蒸馏学生模型使其快速生成
    """
    
    def __init__(
        self,
        ldm_ckpt_path: str,
        vae: nn.Module,
        device: str = "cuda",
        config: dict = None,
    ):
        self.device = device
        self.config = config or {}
        
        # 加载 LDM 教师模型
        self.ldm_model = self._load_ldm_model(ldm_ckpt_path)
        self.vae = vae.to(device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # 创建 CD 学生模型（与 LDM 同结构）
        self.cd_model = OptimizedUNetModel(
            image_size=32,
            in_channels=8,
            model_channels=32,
            out_channels=4,
            num_res_blocks=1,
            attention_resolutions=(),
            channel_mult=(1, 2, 3),
            use_checkpoint=True,
            attention_type="linear",
        ).to(device)
        
        # 创建 EMA 目标模型
        self.cd_model_ema = self._create_ema_model(self.cd_model)
        
        # 初始化 CD 模型为 LDM 的拷贝
        self._initialize_from_ldm()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.cd_model.parameters(),
            lr=config.get('lr', 5e-5),
            weight_decay=1e-4,
        )
        
        # Denoiser
        self.denoiser = KarrasDenoiser(
            sigma_data=0.5,
            sigma_max=80.0,
            sigma_min=0.002,
            loss_norm='l2',
        )
        
        # 混合精度
        self.scaler = GradScaler()
    
    def _load_ldm_model(self, ckpt_path: str) -> nn.Module:
        """加载 LDM 教师模型"""
        model = OptimizedUNetModel(
            image_size=32,
            in_channels=8,
            model_channels=32,
            out_channels=4,
            num_res_blocks=1,
            attention_resolutions=(),
            channel_mult=(1, 2, 3),
            use_checkpoint=True,
        ).to(self.device)
        
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        print(f"Loaded LDM teacher model from {ckpt_path}")
        return model
    
    def _create_ema_model(self, model: nn.Module) -> nn.Module:
        """创建 EMA 目标模型"""
        ema_model = OptimizedUNetModel(
            image_size=32,
            in_channels=8,
            model_channels=32,
            out_channels=4,
            num_res_blocks=1,
            attention_resolutions=(),
            channel_mult=(1, 2, 3),
            use_checkpoint=True,
        ).to(self.device)
        
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        
        return ema_model
    
    def _initialize_from_ldm(self):
        """用 LDM 权重初始化 CD 模型"""
        self.cd_model.load_state_dict(self.ldm_model.state_dict())
        self.cd_model_ema.load_state_dict(self.ldm_model.state_dict())
        print("Initialized CD model from LDM teacher")
    
    def _update_ema(self, ema_rate: float = 0.999):
        """更新 EMA 模型"""
        with torch.no_grad():
            for src_param, ema_param in zip(
                self.cd_model.parameters(),
                self.cd_model_ema.parameters()
            ):
                ema_param.mul_(ema_rate).add_(src_param.data, alpha=1 - ema_rate)
    
    @torch.no_grad()
    def _euler_solver(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        next_t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Euler ODE 求解器
        
        用于从当前时间步推进到下一时间步
        """
        # 模型输出
        model_input = torch.cat([x_t, cond], dim=1)
        denoised = model(model_input, t)
        
        # ODE: dx/dt = (x - denoised) / t
        d = (x_t - denoised) / t.view(-1, 1, 1, 1, 1)
        
        # Euler 步进
        dt = next_t - t
        x_next = x_t + d * dt.view(-1, 1, 1, 1, 1)
        
        return x_next
    
    def train_step(
        self,
        z_target: torch.Tensor,
        z_cond: torch.Tensor,
        num_scales: int = 40,
    ) -> float:
        """单个训练步骤（Consistency Distillation）"""
        batch_size = z_target.shape[0]
        device = z_target.device
        
        # 随机采样时间步对 (t_n, t_{n+1})
        indices = torch.randint(0, num_scales - 1, (batch_size,), device=device)
        
        # 计算噪声水平
        t_n = (
            self.denoiser.sigma_max ** (1 / self.denoiser.rho) +
            indices / (num_scales - 1) * (
                self.denoiser.sigma_min ** (1 / self.denoiser.rho) -
                self.denoiser.sigma_max ** (1 / self.denoiser.rho)
            )
        ) ** self.denoiser.rho
        
        t_next = (
            self.denoiser.sigma_max ** (1 / self.denoiser.rho) +
            (indices + 1) / (num_scales - 1) * (
                self.denoiser.sigma_min ** (1 / self.denoiser.rho) -
                self.denoiser.sigma_max ** (1 / self.denoiser.rho)
            )
        ) ** self.denoiser.rho
        
        # 生成带噪数据
        noise = torch.randn_like(z_target)
        x_t_n = z_target + noise * t_n.view(-1, 1, 1, 1, 1)
        
        # LDM 教师推进一步
        with torch.no_grad():
            x_t_next = self._euler_solver(
                self.ldm_model, x_t_n, t_n, t_next, z_cond
            )
        
        # CD 学生模型输出
        model_input = torch.cat([x_t_next, z_cond], dim=1)
        student_output = self.cd_model(model_input, t_next)
        
        # 目标：学生在 t_next 应该输出与教师相同的结果
        with torch.no_grad():
            teacher_output = self.ldm_model(model_input, t_next)
        
        # 一致性损失
        loss = F.mse_loss(student_output, teacher_output)
        
        return loss
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        num_scales: int = 40,
        grad_accum_steps: int = 8,
    ) -> float:
        """训练一个 epoch"""
        self.cd_model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc="Training")
        self.optimizer.zero_grad()
        
        for batch_idx, (target, cond) in enumerate(pbar):
            target = target.to(self.device)
            cond = cond.to(self.device)
            
            # 编码到潜空间
            with torch.no_grad():
                z_target = self.vae.get_latent(target)
                z_cond = self.vae.get_latent(cond)
            
            # 计算损失
            with autocast(enabled=True):
                loss = self.train_step(z_target, z_cond, num_scales=num_scales)
                loss = loss / grad_accum_steps
            
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度累积更新
            if (batch_idx + 1) % grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.cd_model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # 更新 EMA 模型
                self._update_ema(ema_rate=0.999)
            
            total_loss += loss.item() * grad_accum_steps
            pbar.set_postfix({'loss': f'{loss.item() * grad_accum_steps:.4f}'})
        
        return total_loss / len(train_loader)
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 100,
        save_dir: str = "./results/cd",
        save_every: int = 10,
    ):
        """完整训练流程"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting Consistency Distillation training for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            loss = self.train_epoch(train_loader)
            print(f"Epoch {epoch}: loss={loss:.4f}")
            
            if epoch % save_every == 0:
                ckpt_path = os.path.join(save_dir, f"cd_epoch{epoch:04d}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.cd_model.state_dict(),
                    'ema_model_state_dict': self.cd_model_ema.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Consistency Distillation Training")
    parser.add_argument("--ldm_ckpt", type=str, required=True, help="LDM checkpoint path")
    parser.add_argument("--vae_ckpt", type=str, required=True, help="VAE checkpoint path")
    parser.add_argument("--dataset_dir", type=str, default="./NTU4DRadLM_pre_processing/NTU4DRadLM_Pre")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./diffusion_consistency_radar/train_results/cd")
    parser.add_argument("--lr", type=float, default=5e-5)
    
    args = parser.parse_args()
    
    # 加载 VAE
    vae_config = create_ultra_lightweight_vae_config()
    vae = VAE3D(**vae_config)
    ckpt = torch.load(args.vae_ckpt, map_location='cpu')
    if 'model_state_dict' in ckpt:
        vae.load_state_dict(ckpt['model_state_dict'])
    else:
        vae.load_state_dict(ckpt)
    
    # 创建数据加载器
    dataset = NTU4DRadLM_VoxelDataset(
        root_dir=args.dataset_dir,
        split='train',
        use_augmentation=False,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )
    
    # 创建训练器并训练
    trainer = ConsistencyDistillationTrainer(
        ldm_ckpt_path=args.ldm_ckpt,
        vae=vae,
        config={'lr': args.lr},
    )
    
    trainer.train(
        train_loader,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()
