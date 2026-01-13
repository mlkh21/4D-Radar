# -- coding: utf-8 --
"""
Latent Diffusion 训练脚本

这是终极优化方案：
1. 先训练 VAE 将 3D 体素压缩到潜空间
2. 在潜空间中训练扩散模型
3. 显存使用降低 8-16 倍，训练速度提升 10+ 倍

使用方法:
1. 先训练 VAE:
   python diffusion_consistency_radar/scripts/train_vae.py --mode train_vae

2. 再训练 Latent Diffusion:
   python diffusion_consistency_radar/scripts/train_vae.py --mode train_ldm --vae_ckpt <path>

显存优化策略：
- 使用混合精度训练 (AMP)
- 梯度累积支持小 batch 等效大 batch
- 梯度检查点 (gradient checkpointing)
- 定期清理缓存
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import gc

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cm.vae_3d import (
    VAE3D,
    VQVAE3D,
    create_lightweight_vae_config,
    create_standard_vae_config,
)

# 尝试导入超轻量级配置
try:
    from cm.vae_3d import create_ultra_lightweight_vae_config
except ImportError:
    create_ultra_lightweight_vae_config = None

from cm.dataset_loader import NTU4DRadLM_VoxelDataset
from cm import dist_util, logger


class VAETrainer:
    """VAE 训练器 - 带显存优化"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        lr: float = 1e-4,
        kl_weight: float = 1e-6,
        device: str = "cuda",
        save_dir: str = "./train_results/vae",
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.kl_weight = kl_weight
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化日志文件
        self.log_file = open(os.path.join(save_dir, "train_log.txt"), "a", encoding='utf-8')
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        # 混合精度训练的 GradScaler
        self.scaler = GradScaler() if use_amp else None

        # 多GPU支持
        if device == "cuda" and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for VAE training!")
            self.model = nn.DataParallel(self.model)
        
        # 启用梯度检查点以节省显存
        # 注意：如果使用了 DataParallel，需要访问 .module
        _model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(_model, 'encoder'):
            for module in _model.encoder.modules():
                if hasattr(module, 'use_checkpoint'):
                    module.use_checkpoint = True
        if hasattr(_model, 'decoder'):
            for module in _model.decoder.modules():
                if hasattr(module, 'use_checkpoint'):
                    module.use_checkpoint = True
        
    def log(self, message: str, use_tqdm: bool = False, pbar=None):
        """同时记录到控制台和文件"""
        self.log_file.write(message + "\n")
        self.log_file.flush()
        if use_tqdm and pbar is not None:
            pbar.write(message)
        else:
            print(message)

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (target, cond) in enumerate(pbar):
            # target 是要重建的目标（GT）
            target = target.to(self.device, non_blocking=True)
            
            # 检查输入数据是否有 NaN
            if torch.isnan(target).any():
                self.log(f"Skipping batch {batch_idx}: Input contains NaN", use_tqdm=True, pbar=pbar)
                continue
            
            # 使用混合精度
            with autocast(enabled=self.use_amp):
                # 前向传播
                recon, (mean, logvar) = self.model(target)
                
                # 检查输出是否有 NaN
                if torch.isnan(recon).any():
                    self.log(f"Warning batch {batch_idx}: Output reconstruction contains NaN", use_tqdm=True, pbar=pbar)
                    # 为了防止崩溃，跳过此 batch 的反向传播，或者置零损失
                    # continue
                
                # 计算损失
                if isinstance(self.model, nn.DataParallel):
                    loss_fn = self.model.module.compute_loss
                else:
                    loss_fn = self.model.compute_loss

                loss, recon_loss, kl_loss = loss_fn(
                    target, recon, (mean, logvar)
                )
                
                # 梯度累积：对损失进行缩放
                loss = loss / self.gradient_accumulation_steps
            
            # 反向传播
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积：每 N 步更新一次
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # 累积统计（注意要乘回来）
            total_loss += loss.item() * self.gradient_accumulation_steps
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
            # 定期打印详细日志，防止进度条截断看不到
            if batch_idx % 20 == 0:
                log_msg = f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] loss={loss.item() * self.gradient_accumulation_steps:.4f}, rec={recon_loss.item():.4f}, kl={kl_loss.item():.4f}"
                self.log(log_msg, use_tqdm=True, pbar=pbar)

            pbar.set_postfix({
                'loss': f'{loss.item() * self.gradient_accumulation_steps:.3f}',
                'rec': f'{recon_loss.item():.3f}',
                'kl': f'{kl_loss.item():.3f}'
            })
            
            # 定期打印详细日志，防止进度条截断看不到
            if batch_idx % 20 == 0:
                pbar.write(f"Batch {batch_idx}: loss={loss.item() * self.gradient_accumulation_steps:.4f}, recon={recon_loss.item():.4f}, kl={kl_loss.item():.4f}")
            
            # 定期清理显存
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        self.scheduler.step()
        
        # 清理显存
        gc.collect()
        torch.cuda.empty_cache()
        
        n = len(self.train_loader)
        return total_loss / n, total_recon / n, total_kl / n
    
    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        
        for target, cond in self.val_loader:
            target = target.to(self.device, non_blocking=True)
            
            with autocast(enabled=self.use_amp):
                recon, (mean, logvar) = self.model(target, sample_posterior=False)
                
                if isinstance(self.model, nn.DataParallel):
                    loss_fn = self.model.module.compute_loss
                else:
                    loss_fn = self.model.compute_loss
                
                loss, _, _ = loss_fn(target, recon, (mean, logvar))
            
            total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch: int, loss: float):
        path = os.path.join(self.save_dir, f"vae_epoch{epoch:04d}.pt")
        
        # 获取原始模型状态（去除 DataParallel）
        state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)
        print(f"Saved checkpoint: {path}")
    
    def train(self, num_epochs: int = 100, save_every: int = 10, start_epoch: int = 1):
        best_loss = float('inf')
        self.log(f"Starting training for {num_epochs} epochs (starting from {start_epoch})...")

        for epoch in range(start_epoch, num_epochs + 1):
            train_loss, recon_loss, kl_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            log_msg = f"Epoch {epoch}: train_loss={train_loss:.4f}, recon={recon_loss:.4f}, kl={kl_loss:.6f}"
            if val_loss:
                log_msg += f", val_loss={val_loss:.4f}"
            self.log(log_msg)
            
            # 保存检查点
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, train_loss)
            
            # 保存最佳模型
            current_loss = val_loss if val_loss else train_loss
            if current_loss < best_loss:
                best_loss = current_loss
                best_path = os.path.join(self.save_dir, "vae_best.pt")
                
                state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
                
                # 保存完整状态以便 resume
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_loss,
                }, best_path)
                self.log(f"New best model saved: {best_path}")


def train_vae(args):
    """训练 VAE"""
    print("=" * 60)
    print("Training 3D VAE for Latent Diffusion")
    print("=" * 60)
    
    # 显存优化设置
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 清理显存
    gc.collect()
    torch.cuda.empty_cache()
    
    # 禁用数据增强以节省显存
    train_dataset = NTU4DRadLM_VoxelDataset(
        root_dir=args.dataset_dir,
        split='train',
        use_augmentation=False,  # 禁用增强节省显存
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,  # 禁用pin_memory节省显存
        persistent_workers=False,  # 禁用persistent_workers节省内存
        prefetch_factor=None,
    )
    
    # 创建模型 - 根据类型选择配置
    if args.vae_type == "ultra_lightweight" and create_ultra_lightweight_vae_config is not None:
        config = create_ultra_lightweight_vae_config()
        print("Using ultra-lightweight VAE config (minimal memory)")
    elif args.vae_type == "lightweight":
        config = create_lightweight_vae_config()
        print("Using lightweight VAE config")
    else:
        config = create_standard_vae_config()
        print("Using standard VAE config")
    
    config['kl_weight'] = args.kl_weight
    model = VAE3D(**config)
    
    # 注: 直接使用超轻量级配置来减少显存，不使用输入下采样
    
    # 打印模型信息
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VAE parameters: {num_params / 1e6:.2f}M")
    print(f"Batch size: {args.batch_size}, Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Mixed precision (AMP): {args.use_amp}")
    
    # 准备设备
    print(f"GPU Device: {args.device}")
    torch.cuda.empty_cache()
    
    # 训练
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        lr=args.lr,
        kl_weight=args.kl_weight,
        device=args.device,
        save_dir=args.vae_save_dir,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    start_epoch = 1
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, map_location=args.device)
            # 处理 DataParallel 包装
            _model = model.module if hasattr(model, 'module') else model
            
            # 兼容处理：检查是完整 checkpoint 还是仅 state_dict
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                _model.load_state_dict(ckpt['model_state_dict'])
                if 'optimizer_state_dict' in ckpt:
                    trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if 'epoch' in ckpt:
                    start_epoch = ckpt['epoch'] + 1
                print(f"Successfully resumed from epoch {start_epoch-1}")
            else:
                # 假设是纯 state_dict (旧版本保存的 best model)
                print("Warning: Loading raw state_dict (no optimizer state or epoch info found)")
                _model.load_state_dict(ckpt)
                # 尝试推断 epoch，如果是 best 可能就是 epoch 1 或者需要手动指定
                # 这里默认保持 start_epoch = 1，如果用户想继续训练，至少权重是对的
        else:
            print(f"Warning: Checkpoint not found at {args.resume}, starting from scratch.")
    
    trainer.train(
        num_epochs=args.vae_epochs,
        save_every=args.save_every,
        start_epoch=start_epoch,
    )


def train_latent_diffusion(args):
    """在潜空间训练扩散模型"""
    print("=" * 60)
    print("Training Latent Diffusion Model")
    print("=" * 60)
    
    # 加载预训练的 VAE
    if not args.vae_ckpt:
        raise ValueError("Must provide --vae_ckpt for latent diffusion training")
    
    if args.vae_type == "lightweight":
        config = create_lightweight_vae_config()
    else:
        config = create_standard_vae_config()
    
    vae = VAE3D(**config)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location='cpu'))
    vae = vae.to(args.device)
    vae.eval()
    
    # 冻结 VAE
    for param in vae.parameters():
        param.requires_grad = False
    
    print(f"Loaded VAE from: {args.vae_ckpt}")
    
    # 计算潜空间大小
    # 假设输入是 (4, 32, 128, 128)，VAE 压缩 4x4x4
    latent_size = (8, 32, 32)  # (D/4, H/4, W/4)
    latent_channels = config['latent_dim']
    
    print(f"Latent space: {latent_channels} x {latent_size}")
    
    # 创建潜空间扩散模型
    # 这里可以使用更小的模型，因为潜空间已经压缩了
    from cm.unet_optimized import OptimizedUNetModel
    from cm.karras_diffusion import KarrasDenoiser
    
    diffusion_model = OptimizedUNetModel(
        image_size=latent_size[1],  # H 维度
        in_channels=latent_channels * 2,  # 条件 + 噪声
        model_channels=args.model_channels,
        out_channels=latent_channels,
        num_res_blocks=2,
        attention_resolutions=(4, 2),  # 潜空间更小，可以用更多注意力
        dropout=0.1,
        channel_mult=(1, 2, 4),
        use_checkpoint=True,
        use_fp16=True,
        num_heads=4,
        attention_type="flash",
        norm_type="group",
        downsample_type="asymmetric",
        initial_z_size=latent_size[0],
    ).to(args.device)
    
    denoiser = KarrasDenoiser(
        sigma_data=0.5,
        sigma_max=80.0,
        sigma_min=0.002,
    )
    
    # 加载数据
    train_dataset = NTU4DRadLM_VoxelDataset(
        root_dir=args.dataset_dir,
        split='train',
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        diffusion_model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    
    # 训练循环
    print(f"\nStarting Latent Diffusion training...")
    print(f"Model parameters: {sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    global_step = 0
    for epoch in range(1, args.ldm_epochs + 1):
        diffusion_model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (target, cond) in enumerate(pbar):
            target = target.to(args.device)
            cond = cond.to(args.device)
            
            # 编码到潜空间
            with torch.no_grad():
                z_target = vae.get_latent(target)
                z_cond = vae.get_latent(cond)
            
            # 采样噪声水平
            batch_size = z_target.shape[0]
            sigmas = denoiser.sigma_max ** (
                torch.rand(batch_size, device=args.device)
            ) * denoiser.sigma_min ** (1 - torch.rand(batch_size, device=args.device))
            
            # 加噪
            noise = torch.randn_like(z_target)
            noised_z = z_target + noise * sigmas.view(-1, 1, 1, 1, 1)
            
            # 预测去噪
            optimizer.zero_grad()
            
            # 拼接条件
            model_input = torch.cat([noised_z, z_cond], dim=1)
            
            # 模型输出
            denoised = diffusion_model(model_input, sigmas)
            
            # 损失
            loss = F.mse_loss(denoised, z_target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), 1.0)
            optimizer.step()
            
            global_step += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 定期保存
            if global_step % args.save_every == 0:
                save_path = os.path.join(
                    args.ldm_save_dir, f"ldm_step{global_step:06d}.pt"
                )
                torch.save({
                    'step': global_step,
                    'model_state_dict': diffusion_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, save_path)
                print(f"\nSaved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Latent Diffusion Training")
    
    # 模式选择
    parser.add_argument("--mode", type=str, default="train_vae",
                        choices=["train_vae", "train_ldm"],
                        help="Training mode")
    
    # 数据参数
    parser.add_argument("--dataset_dir", type=str,
                        default="./NTU4DRadLM_pre_processing/NTU4DRadLM_Pre")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # VAE 参数
    parser.add_argument("--vae_type", type=str, default="lightweight",
                        choices=["ultra_lightweight", "lightweight", "standard"])
    parser.add_argument("--vae_epochs", type=int, default=100)
    parser.add_argument("--vae_save_dir", type=str,
                        default="./diffusion_consistency_radar/train_results/vae")
    parser.add_argument("--vae_ckpt", type=str, default="",
                        help="VAE checkpoint for LDM training")
    parser.add_argument("--kl_weight", type=float, default=1e-6)
    
    # LDM 参数
    parser.add_argument("--ldm_epochs", type=int, default=200)
    parser.add_argument("--ldm_save_dir", type=str,
                        default="./diffusion_consistency_radar/train_results/ldm")
    parser.add_argument("--model_channels", type=int, default=64)
    
    # 训练参数
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--resume", type=str, default="",
                        help="Resume training from checkpoint")
    
    # 显存优化参数
    parser.add_argument("--use_amp", action="store_true",
                        help="使用混合精度训练 (AMP)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数，有效 batch_size = batch_size * gradient_accumulation_steps")
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.vae_save_dir, exist_ok=True)
    os.makedirs(args.ldm_save_dir, exist_ok=True)
    
    # 执行
    if args.mode == "train_vae":
        train_vae(args)
    elif args.mode == "train_ldm":
        train_latent_diffusion(args)


if __name__ == "__main__":
    main()
