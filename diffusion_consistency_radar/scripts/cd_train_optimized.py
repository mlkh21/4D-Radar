# -- coding: utf-8 --
"""
优化的 Consistency Distillation 训练脚本

改进点：
1. 清晰的蒸馏流程 - 从 LDM 教师蒸馏 CD 学生模型
2. 显存优化 - 梯度累积、检查点、混合精度
3. 规范的实现 - 遵循 Consistency Distillation 论文
4. 模块化设计 - 易于理解和维护
"""

import sys
import os
# 添加父目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import argparse
import logging
import csv
import time
from typing import Tuple
from tqdm import tqdm

from cm.unet_optimized import OptimizedUNetModel
from cm.karras_diffusion import KarrasDenoiser
from cm.vae_3d import VAE3D, create_ultra_lightweight_vae_config
from cm.dataset_loader import NTU4DRadLM_VoxelDataset


def safe_torch_load(path, map_location):
    """兼容不同 PyTorch 版本的 checkpoint 加载逻辑。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # HACK: 低版本 PyTorch 不支持 weights_only，回退到兼容模式。
        return torch.load(path, map_location=map_location)
    except Exception as exc:
        # NOTE: 某些历史权重包含自定义对象，weights_only=True 会拒绝加载。
        msg = str(exc)
        if "Weights only load failed" in msg or "Unsupported global" in msg:
            return torch.load(path, map_location=map_location)
        raise


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
        
        # 设置保存目录和日志
        self.save_dir = config.get('save_dir', './Result/train_results/cd')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化训练状态
        self.start_epoch = 1
        self.best_loss = float('inf')
        self.is_resumed = False
        
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
        
        # 禁用混合精度（避免 FP16/FP32 类型不匹配）
        self.use_amp = False
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # 设置日志
        self.log_file = os.path.join(self.save_dir, 'training.log')
        self.csv_file = os.path.join(self.save_dir, 'metrics.csv')
        
        # 检查是否恢复训练
        resume_path = config.get('resume_path')
        if resume_path and os.path.exists(resume_path):
            self.is_resumed = True
        
        self._setup_logging()
        
        # 恢复训练
        if self.is_resumed:
            self._resume_from_checkpoint(resume_path)
    
    def _load_ldm_model(self, ckpt_path: str) -> nn.Module:
        """加载 LDM 教师模型"""
        # 静默创建模型（避免重复打印）
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        model = OptimizedUNetModel(
            image_size=32,
            in_channels=8,
            model_channels=32,
            out_channels=4,
            num_res_blocks=1,
            attention_resolutions=(),
            channel_mult=(1, 2, 3),
            use_checkpoint=True,
            attention_type="linear",
        ).to(self.device)
        
        sys.stdout = old_stdout
        
        ckpt = safe_torch_load(ckpt_path, map_location='cpu')
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
        # 静默创建模型
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        ema_model = OptimizedUNetModel(
            image_size=32,
            in_channels=8,
            model_channels=32,
            out_channels=4,
            num_res_blocks=1,
            attention_resolutions=(),
            channel_mult=(1, 2, 3),
            use_checkpoint=True,
            attention_type="linear",
        ).to(self.device)
        
        sys.stdout = old_stdout
        
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        
        return ema_model
    
    def _setup_logging(self):
        """设置日志系统"""
        # 确定日志文件模式
        log_mode = 'a' if self.is_resumed else 'w'
        
        # 配置文本日志（只写入文件，不输出到终端）
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, mode=log_mode)
            ],
            force=True
        )
        self.logger = logging.getLogger(__name__ + '_cd')
        
        # 添加训练会话分隔符
        if self.is_resumed:
            self.logger.info("\n" + "="*70)
            self.logger.info("RESUMING TRAINING SESSION")
            self.logger.info("="*70)
        
        # 初始化 CSV 文件
        if not os.path.exists(self.csv_file) or not self.is_resumed:
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'loss', 'lr', 'time_seconds'])
    
    def _resume_from_checkpoint(self, ckpt_path: str):
        """从检查点恢复训练"""
        print(f"Resuming CD from checkpoint: {ckpt_path}")
        ckpt = safe_torch_load(ckpt_path, map_location=self.device)
        
        # 加载模型
        self.cd_model.load_state_dict(ckpt['model_state_dict'])
        if 'ema_model_state_dict' in ckpt:
            self.cd_model_ema.load_state_dict(ckpt['ema_model_state_dict'])
        
        # 加载优化器
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        # 加载训练状态
        self.start_epoch = ckpt.get('epoch', 0) + 1
        self.best_loss = ckpt.get('best_loss', ckpt.get('loss', float('inf')))
        
        print(f"Resumed from epoch {self.start_epoch - 1}, best loss: {self.best_loss:.4f}")
    
    def _log_metrics(self, epoch: int, loss: float, epoch_time: float):
        """记录指标到 CSV 文件"""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f'{loss:.6f}',
                f'{self.optimizer.param_groups[0]["lr"]:.8f}',
                f'{epoch_time:.2f}'
            ])
    
    def _initialize_from_ldm(self):
        """用 LDM 权重初始化 CD 模型"""
        if not self.is_resumed:
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
        
        # NOTE: 这里使用显式 Euler，对应一致性蒸馏中的教师推进步骤。
        # NOTE: 常微分方程（ODE）形式：dx/dt = (x - denoised) / t。
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
        """
        单个训练步骤（Consistency Distillation）
        
        核心思想：
        1. 学生模型：从 x(t_n) 直接一步预测去噪后的结果
        2. 教师模型：从 x(t_n) 用 ODE solver 推进到 t_{n+1}，再预测去噪结果
        3. 让学生的一步预测接近教师的多步结果
        """
        batch_size = z_target.shape[0]
        device = z_target.device
        
        # NOTE: 随机采样相邻时间步对 (t_n, t_{n+1})，覆盖不同噪声区间。
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
        
        # 生成带噪数据 x(t_n)
        noise = torch.randn_like(z_target)
        x_t_n = z_target + noise * t_n.view(-1, 1, 1, 1, 1)
        
        # 学生模型：从 x(t_n) 直接预测去噪结果
        student_input = torch.cat([x_t_n, z_cond], dim=1)
        student_denoised = self.cd_model(student_input, t_n)
        
        # 教师模型（EMA）：从 x(t_n) 推进到 x(t_{n+1})，再预测
        with torch.no_grad():
            # 使用 EMA 教师模型
            x_t_next = self._euler_solver(
                self.cd_model_ema, x_t_n, t_n, t_next, z_cond
            )
            
            # 教师在 t_{n+1} 的预测
            teacher_input = torch.cat([x_t_next, z_cond], dim=1)
            teacher_denoised = self.cd_model_ema(teacher_input, t_next)
        
        # NOTE: 一致性目标：学生一步输出逼近教师多步推进后的输出。
        loss = F.mse_loss(student_denoised, teacher_denoised)
        
        return loss
    
    def train_epoch(
        self,
        epoch: int,
        train_loader: DataLoader,
        num_scales: int = 40,
        grad_accum_steps: int = 8,
    ) -> float:
        """训练一个 epoch"""
        self.cd_model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        self.optimizer.zero_grad()
        
        for batch_idx, (target, cond) in enumerate(pbar):
            target = target.to(self.device)
            cond = cond.to(self.device)
            
            # 编码到潜空间
            with torch.no_grad():
                z_target = self.vae.get_latent(target)
                z_cond = self.vae.get_latent(cond)
            
            # 计算损失
            loss = self.train_step(z_target, z_cond, num_scales=num_scales)
            loss = loss / grad_accum_steps
            
            # 反向传播
            loss.backward()
            
            # NOTE: 梯度累积用于控制显存；每 grad_accum_steps 次小步做一次参数更新。
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.cd_model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # 更新 EMA 模型
                self._update_ema(ema_rate=0.999)
            
            total_loss += loss.item() * grad_accum_steps
            pbar.set_postfix({'loss': f'{loss.item() * grad_accum_steps:.6f}'})
        
        return total_loss / len(train_loader)
    
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int = 100,
        save_every: int = 10,
    ):
        """完整训练流程"""
        estimated_total_steps = num_epochs * len(train_loader)
        
        msg = "="*70 + "\n"
        msg += f"Starting CD Training\n"
        msg += f"  Total epochs: {num_epochs}\n"
        msg += f"  Batches per epoch: {len(train_loader)}\n"
        msg += f"  Estimated total steps: {estimated_total_steps:,}\n"
        msg += f"  Start epoch: {self.start_epoch}\n"
        msg += f"  Batch size: {train_loader.batch_size}\n"
        msg += f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}\n"
        msg += f"  Save directory: {self.save_dir}\n"
        msg += f"  Log file: {self.log_file}\n"
        msg += f"  CSV file: {self.csv_file}\n"
        msg += "="*70
        print(msg) 
        self.logger.info(msg)

        for epoch in range(self.start_epoch, num_epochs + 1):
            epoch_start = time.time()
            loss = self.train_epoch(epoch, train_loader)
            epoch_time = time.time() - epoch_start
            
            # 记录日志
            msg = f"\n[Epoch {epoch}/{num_epochs}] Loss: {loss:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e} | Time: {epoch_time:.1f}s"
            print(msg)
            self.logger.info(msg)
            self._log_metrics(epoch, loss, epoch_time)
            
            # 保存最佳模型
            if loss < self.best_loss:
                self.best_loss = loss
                best_ckpt = os.path.join(self.save_dir, "cd_best.pt")
                torch.save({
                    'epoch': epoch,
                    'loss': loss,
                    'best_loss': self.best_loss,
                    'model_state_dict': self.cd_model.state_dict(),
                    'ema_model_state_dict': self.cd_model_ema.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, best_ckpt)
                msg = f"  ✓ Saved best model (loss: {loss:.4f})"
                self.logger.info(msg)
            
            # 定期保存检查点
            if epoch % save_every == 0:
                ckpt_path = os.path.join(self.save_dir, f"cd_epoch{epoch:04d}.pt")
                torch.save({
                    'epoch': epoch,
                    'loss': loss,
                    'best_loss': self.best_loss,
                    'model_state_dict': self.cd_model.state_dict(),
                    'ema_model_state_dict': self.cd_model_ema.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, ckpt_path)
                self.logger.info(f"  Saved checkpoint: {ckpt_path}")
        
        msg = "\nTraining completed!"
        print(msg)
        self.logger.info(msg)


def main():
    parser = argparse.ArgumentParser(description="Consistency Distillation Training")
    parser.add_argument("--ldm_ckpt", type=str, required=True, help="LDM checkpoint path")
    parser.add_argument("--vae_ckpt", type=str, required=True, help="VAE checkpoint path")
    parser.add_argument("--dataset_dir", type=str, default="./Data/NTU4DRadLM_Pre")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--save_dir", type=str, default="./Result/train_results/cd")
    parser.add_argument("--lr", type=float, default=5e-5)
    
    args = parser.parse_args()
    
    # 加载 VAE
    vae_config = create_ultra_lightweight_vae_config()
    vae = VAE3D(**vae_config)
    ckpt = safe_torch_load(args.vae_ckpt, map_location='cpu')
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
    
    # 检查是否有检查点可以恢复
    resume_path = os.path.join(args.save_dir, 'cd_best.pt')
    if not os.path.exists(resume_path):
        resume_path = None
    
    # 创建训练器并训练
    trainer = ConsistencyDistillationTrainer(
        ldm_ckpt_path=args.ldm_ckpt,
        vae=vae,
        config={
            'lr': args.lr,
            'save_dir': args.save_dir,
            'resume_path': resume_path,
        },
    )
    
    trainer.train(
        train_loader,
        num_epochs=args.num_epochs,
        save_every=10,
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()
