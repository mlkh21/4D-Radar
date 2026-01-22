# -- coding: utf-8 --
"""
统一训练脚本 - 整合 VAE、LDM、CD 训练

优化点：
1. 统一配置系统 - 使用 YAML 配置，避免代码中硬编码参数
2. 显存优化 - 梯度累积、检查点、稀疏处理
3. 蒸馏流程优化 - 清晰的 LDM -> CD 蒸馏步骤
4. 模块化架构 - 易于维护和扩展
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import gc
from tqdm import tqdm
import time
import csv
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cm.vae_3d import (
    VAE3D, 
    create_lightweight_vae_config,
    create_ultra_lightweight_vae_config,
    create_standard_vae_config,
)
from cm.unet_optimized import OptimizedUNetModel, create_lightweight_unet_config
from cm.dataset_loader import NTU4DRadLM_VoxelDataset
from cm.karras_diffusion import KarrasDenoiser


class ConfigManager:
    """配置管理器 - 统一加载和管理配置"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> Dict:
        """加载 YAML 配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default=None):
        """获取配置值，支持点号分隔的嵌套访问"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value


class MemoryOptimizer:
    """显存优化器 - 统一管理显存优化策略"""
    
    def __init__(self, config: ConfigManager):
        self.use_amp = config.get('optimization.use_amp', True)
        self.use_checkpoint = config.get('optimization.use_checkpoint', True)
        self.grad_accum_steps = config.get('optimization.gradient_accumulation_steps', 1)
        device_cfg = config.get('hardware.device', 'cuda') or 'cuda'
        self.device = torch.device(device_cfg)
        
        self.scaler = GradScaler('cuda') if self.use_amp else None
    
    def clear_cache(self):
        """清理显存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_stats(self) -> Dict:
        """获取显存统计"""
        if not torch.cuda.is_available():
            return {}
        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'peak_gb': torch.cuda.max_memory_allocated() / 1024**3,
        }
    
    def print_stats(self, prefix: str = ""):
        """打印显存统计"""
        stats = self.get_memory_stats()
        if stats:
            print(f"{prefix}GPU: {stats['allocated_gb']:.1f}GB allocated, "
                  f"{stats['reserved_gb']:.1f}GB reserved, "
                  f"{stats['peak_gb']:.1f}GB peak")


class OptimizedVAETrainer:
    """优化的 VAE 训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        config: ConfigManager,
        memory_opt: MemoryOptimizer,
        resume_path: str = None,
    ):
        self.config = config
        self.memory_opt = memory_opt
        self.device = memory_opt.device
        
        # 将模型移到设备
        self.model = model.to(self.device)
        
        # 训练参数
        self.vae_config = config.get('vae', {})
        self.lr = self.vae_config.get('lr', 1e-4)
        self.epochs = self.vae_config.get('epochs', 100)
        self.save_dir = self.vae_config.get('save_dir', './results/vae')
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 优化器和调度器
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6
        )
        
        # 初始化训练状态
        self.start_epoch = 1
        self.best_loss = float('inf')
        
        # 设置日志
        self.log_file = os.path.join(self.save_dir, 'training.log')
        self.csv_file = os.path.join(self.save_dir, 'metrics.csv')
        self._setup_logging()
        
        # 恢复训练
        if resume_path and os.path.exists(resume_path):
            self._resume_from_checkpoint(resume_path)
        
        # 初始化训练状态
        self.start_epoch = 1
        self.best_loss = float('inf')
        
        # 设置日志
        self.log_file = os.path.join(self.save_dir, 'training.log')
        self.csv_file = os.path.join(self.save_dir, 'metrics.csv')
        self._setup_logging()
        
        # 恢复训练
        if resume_path and os.path.exists(resume_path):
            self._resume_from_checkpoint(resume_path)
    
    def _setup_logging(self):
        """设置日志系统"""
        # 配置文本日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 初始化 CSV 文件
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'loss', 'recon_loss', 'kl_loss', 'lr', 'time_seconds'])
    
    def _resume_from_checkpoint(self, ckpt_path: str):
        """从检查点恢复训练"""
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        # 加载模型
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt['model_state_dict'])
        
        # 加载优化器
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        # 加载训练状态
        self.start_epoch = ckpt.get('epoch', 0) + 1
        self.best_loss = ckpt.get('best_loss', ckpt.get('loss', float('inf')))
        
        print(f"Resumed from epoch {self.start_epoch - 1}, best loss: {self.best_loss:.4f}")
    
    def _log_metrics(self, epoch: int, loss: float, recon_loss: float, kl_loss: float, epoch_time: float):
        """记录指标到 CSV 文件"""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f'{loss:.6f}',
                f'{recon_loss:.6f}',
                f'{kl_loss:.8f}',
                f'{self.optimizer.param_groups[0]["lr"]:.8f}',
                f'{epoch_time:.2f}'
            ])
    
    def train_epoch(self, epoch: int, train_loader: DataLoader) -> tuple:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        # 创建进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}")
        
        for batch_idx, (target, cond) in enumerate(pbar):
            target = target.to(self.device, non_blocking=True)
            
            if torch.isnan(target).any():
                print(f"Warning: Batch {batch_idx} contains NaN")
                continue
            
            # 前向传播（混合精度）
            with autocast('cuda', enabled=self.memory_opt.use_amp):
                recon, (mean, logvar) = self.model(target)
                loss, recon_loss, kl_loss = self.model.compute_loss(
                    target, recon, (mean, logvar)
                )
                loss = loss / self.memory_opt.grad_accum_steps
            
            # 反向传播
            if self.memory_opt.scaler:
                self.memory_opt.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积更新
            if (batch_idx + 1) % self.memory_opt.grad_accum_steps == 0:
                if self.memory_opt.scaler:
                    self.memory_opt.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.memory_opt.scaler.step(self.optimizer)
                    self.memory_opt.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # 累积损失
            batch_loss = loss.item() * self.memory_opt.grad_accum_steps
            total_loss += batch_loss
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
            # 更新进度条显示
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.6f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # 定期清理显存
            if batch_idx % 50 == 0:
                self.memory_opt.clear_cache()
        
        self.scheduler.step()
        n = len(train_loader)
        return total_loss / n, total_recon / n, total_kl / n
    
    def train(self, train_loader: DataLoader):
        """完整训练流程"""
        msg = "=" * 70 + "\n"
        msg += f"Starting VAE Training\n"
        msg += f"  Total epochs: {self.epochs}\n"
        msg += f"  Start epoch: {self.start_epoch}\n"
        msg += f"  Batch size: {train_loader.batch_size}\n"
        msg += f"  Gradient accumulation: {self.memory_opt.grad_accum_steps}\n"
        msg += f"  Effective batch size: {train_loader.batch_size * self.memory_opt.grad_accum_steps}\n"
        msg += f"  Learning rate: {self.lr}\n"
        msg += f"  Device: {self.device}\n"
        msg += f"  Save directory: {self.save_dir}\n"
        msg += f"  Log file: {self.log_file}\n"
        msg += f"  CSV file: {self.csv_file}\n"
        msg += "=" * 70
        print(msg)
        self.logger.info(msg)
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_start = time.time()
            loss, recon_loss, kl_loss = self.train_epoch(epoch, train_loader)
            epoch_time = time.time() - epoch_start
            
            # 记录到 CSV
            self._log_metrics(epoch, loss, recon_loss, kl_loss, epoch_time)
            
            # 打印和记录 epoch 总结
            summary = (f"\n[Epoch {epoch}/{self.epochs}] "
                      f"Loss: {loss:.4f} | Recon: {recon_loss:.4f} | KL: {kl_loss:.6f} | "
                      f"Time: {epoch_time:.1f}s")
            print(summary)
            self.logger.info(summary)
            
            # 显存统计
            self.memory_opt.print_stats(prefix="  ")
            
            # 保存最佳模型
            if loss < self.best_loss:
                self.best_loss = loss
                best_ckpt = os.path.join(self.save_dir, "vae_best.pt")
                state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    'best_loss': self.best_loss,
                }, best_ckpt)
                msg = f"  ✓ Saved best model (loss: {loss:.4f})"
                print(msg)
                self.logger.info(msg)
            
            # 定期保存
            if epoch % self.vae_config.get('save_every', 10) == 0:
                ckpt_path = os.path.join(self.save_dir, f"vae_epoch{epoch:04d}.pt")
                state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    'best_loss': self.best_loss,
                }, ckpt_path)
                msg = f"  ✓ Saved checkpoint: {ckpt_path}"
                print(msg)
                self.logger.info(msg)
        
        total_time = time.time() - start_time
        final_msg = "\n" + "=" * 70 + "\n"
        final_msg += f"Training completed in {total_time/3600:.2f} hours\n"
        final_msg += f"Best loss: {self.best_loss:.4f}\n"
        final_msg += "=" * 70
        print(final_msg)
        self.logger.info(final_msg)


class OptimizedLDMTrainer:
    """优化的 Latent Diffusion 训练器"""
    
    def __init__(
        self,
        vae: nn.Module,
        config: ConfigManager,
        memory_opt: MemoryOptimizer,
        resume_path: str = None,
    ):
        self.vae = vae.to(memory_opt.device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        
        self.config = config
        self.memory_opt = memory_opt
        self.device = memory_opt.device
        
        ldm_config: Dict[str, Any] = config.get('ldm', {}) or {}
        self.ldm_config = ldm_config
        self.save_dir = ldm_config.get('save_dir', './results/ldm')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 创建潜空间模型
        self.model = OptimizedUNetModel(
            image_size=32,  # 潜空间 H/W
            in_channels=8,  # 2x latent_dim (条件+噪声)
            model_channels=ldm_config.get('model_channels', 32),
            out_channels=4,  # latent_dim
            num_res_blocks=ldm_config.get('num_res_blocks', 1),
            attention_resolutions=tuple(ldm_config.get('attention_resolutions', [])),
            channel_mult=tuple(ldm_config.get('channel_mult', [1, 2, 3])),
            use_checkpoint=True,
            attention_type="linear",
        ).to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=ldm_config.get('lr', 1e-4),
            weight_decay=1e-4,
        )
        
        # Denoiser
        self.denoiser = KarrasDenoiser(
            sigma_data=ldm_config.get('sigma_data', 0.5),
            sigma_max=ldm_config.get('sigma_max', 80.0),
            sigma_min=ldm_config.get('sigma_min', 0.002),
            loss_norm='l2',
        )
        
        # 初始化训练状态
        self.start_epoch = 1
        self.global_step = 0
        self.best_loss = float('inf')
        
        # 设置日志
        self.log_file = os.path.join(self.save_dir, 'training.log')
        self.csv_file = os.path.join(self.save_dir, 'metrics.csv')
        self._setup_logging()
        
        # 恢复训练
        if resume_path and os.path.exists(resume_path):
            self._resume_from_checkpoint(resume_path)
    
    def _setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__ + '_ldm')
        
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'step', 'loss', 'lr', 'time_seconds'])
    
    def _resume_from_checkpoint(self, ckpt_path: str):
        """从检查点恢复训练"""
        print(f"Resuming LDM from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        self.model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        self.start_epoch = ckpt.get('epoch', 0) + 1
        self.global_step = ckpt.get('step', 0)
        self.best_loss = ckpt.get('best_loss', ckpt.get('loss', float('inf')))
        
        print(f"Resumed from epoch {self.start_epoch - 1}, step {self.global_step}, best loss: {self.best_loss:.4f}")
    
    def _log_metrics(self, epoch: int, step: int, loss: float, epoch_time: float):
        """记录指标到 CSV 文件"""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                step,
                f'{loss:.6f}',
                f'{self.optimizer.param_groups[0]["lr"]:.8f}',
                f'{epoch_time:.2f}'
            ])
    
    def train_epoch(self, epoch: int, train_loader: DataLoader) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        
        # 创建进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (target, cond) in enumerate(pbar):
            target = target.to(self.device)
            cond = cond.to(self.device)
            
            # 编码到潜空间
            with torch.no_grad():
                z_target = self.vae.get_latent(target)
                z_cond = self.vae.get_latent(cond)
            
            # 采样噪声水平
            batch_size = z_target.shape[0]
            sigmas = (
                self.denoiser.sigma_max ** torch.rand(batch_size, device=self.device)
            ) * (
                self.denoiser.sigma_min ** (1 - torch.rand(batch_size, device=self.device))
            )
            
            # 生成噪声并加噪
            noise = torch.randn_like(z_target)
            noised_z = z_target + noise * sigmas.view(-1, 1, 1, 1, 1)
            
            # 拼接条件
            model_input = torch.cat([noised_z, z_cond], dim=1)
            
            # 前向传播
            with autocast('cuda', enabled=self.memory_opt.use_amp):
                denoised = self.model(model_input, sigmas)
                loss = torch.nn.functional.mse_loss(denoised, z_target)
                loss = loss / self.memory_opt.grad_accum_steps
            
            # 反向传播
            if self.memory_opt.scaler:
                self.memory_opt.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度更新
            if (batch_idx + 1) % self.memory_opt.grad_accum_steps == 0:
                if self.memory_opt.scaler:
                    self.memory_opt.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.memory_opt.scaler.step(self.optimizer)
                    self.memory_opt.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            batch_loss = loss.item() * self.memory_opt.grad_accum_steps
            total_loss += batch_loss
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        return total_loss / len(train_loader)
    
    def train(self, train_loader: DataLoader):
        """完整训练流程"""
        epochs = self.ldm_config.get('epochs', 200)
        save_every = self.ldm_config.get('save_every', 5000)
        
        msg = "=" * 70 + "\n"
        msg += f"Starting LDM Training\n"
        msg += f"  Total epochs: {epochs}\n"
        msg += f"  Start epoch: {self.start_epoch}\n"
        msg += f"  Batch size: {train_loader.batch_size}\n"
        msg += f"  Gradient accumulation: {self.memory_opt.grad_accum_steps}\n"
        msg += f"  Effective batch size: {train_loader.batch_size * self.memory_opt.grad_accum_steps}\n"
        msg += f"  Save directory: {self.save_dir}\n"
        msg += f"  Log file: {self.log_file}\n"
        msg += f"  CSV file: {self.csv_file}\n"
        msg += "=" * 70
        print(msg)
        self.logger.info(msg)
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, epochs + 1):
            epoch_start = time.time()
            loss = self.train_epoch(epoch, train_loader)
            epoch_time = time.time() - epoch_start
            self.global_step += len(train_loader)
            
            # 记录到 CSV
            self._log_metrics(epoch, self.global_step, loss, epoch_time)
            
            summary = f"\n[Epoch {epoch}/{epochs}] Loss: {loss:.4f} | Step: {self.global_step} | Time: {epoch_time:.1f}s"
            print(summary)
            self.logger.info(summary)
            self.memory_opt.print_stats(prefix="  ")
            
            # 保存最佳模型
            if loss < self.best_loss:
                self.best_loss = loss
                best_ckpt = os.path.join(self.save_dir, "ldm_best.pt")
                torch.save({
                    'epoch': epoch,
                    'step': self.global_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    'best_loss': self.best_loss,
                }, best_ckpt)
                msg = f"  ✓ Saved best model (loss: {loss:.4f})"
                print(msg)
                self.logger.info(msg)
            
            if self.global_step % save_every == 0:
                ckpt_path = os.path.join(self.save_dir, f"ldm_step{self.global_step:06d}.pt")
                torch.save({
                    'epoch': epoch,
                    'step': self.global_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    'best_loss': self.best_loss,
                }, ckpt_path)
                msg = f"  ✓ Saved checkpoint: {ckpt_path}"
                print(msg)
                self.logger.info(msg)
        
        total_time = time.time() - start_time
        final_msg = "\n" + "=" * 70 + "\n"
        final_msg += f"Training completed in {total_time/3600:.2f} hours\n"
        final_msg += f"Best loss: {self.best_loss:.4f}\n"
        final_msg += "=" * 70
        print(final_msg)
        self.logger.info(final_msg)


def main():
    parser = argparse.ArgumentParser(description="Unified Training Script")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["vae", "ldm", "cd"],
                        help="Training mode: vae, ldm, or cd")
    parser.add_argument("--config", type=str, default="./diffusion_consistency_radar/config/default_config.yaml",
                        help="Config file path")
    parser.add_argument("--vae_ckpt", type=str, default="",
                        help="VAE checkpoint path (for LDM training)")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # 加载配置
    config = ConfigManager(args.config)
    memory_opt = MemoryOptimizer(config)
    
    # 创建数据加载器
    data_config = config.get('data', {})
    dataset = NTU4DRadLM_VoxelDataset(
        root_dir=data_config.get('dataset_dir'),
        split='train',
        use_augmentation=data_config.get('use_augmentation', True),
    )
    train_loader = DataLoader(
        dataset,
        batch_size=data_config.get('batch_size', 2),
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=False,
    )
    
    # VAE 训练
    if args.mode == "vae":
        vae_type = config.get('vae.config_type', 'ultra_lightweight')
        if vae_type == 'ultra_lightweight':
            vae_config = create_ultra_lightweight_vae_config()
        elif vae_type == 'lightweight':
            vae_config = create_lightweight_vae_config()
        else:
            vae_config = create_standard_vae_config()
        
        vae = VAE3D(**vae_config)
        trainer = OptimizedVAETrainer(vae, config, memory_opt, resume_path=args.resume)
        trainer.train(train_loader)
    
    # LDM 训练
    elif args.mode == "ldm":
        if not args.vae_ckpt:
            raise ValueError("Must provide --vae_ckpt for LDM, resume_path=args.resume training")
        
        vae_type = config.get('vae.config_type', 'ultra_lightweight')
        if vae_type == 'ultra_lightweight':
            vae_config = create_ultra_lightweight_vae_config()
        elif vae_type == 'lightweight':
            vae_config = create_lightweight_vae_config()
        else:
            vae_config = create_standard_vae_config()
        
        vae = VAE3D(**vae_config)
        ckpt = torch.load(args.vae_ckpt, map_location='cpu')
        if 'model_state_dict' in ckpt:
            vae.load_state_dict(ckpt['model_state_dict'])
        else:
            vae.load_state_dict(ckpt)
        
        trainer = OptimizedLDMTrainer(vae, config, memory_opt)
        trainer.train(train_loader)
    
    print("Training completed!")


if __name__ == "__main__":
    main()
