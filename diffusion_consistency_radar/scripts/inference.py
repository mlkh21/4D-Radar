#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的推理脚本 - 使用训练好的 VAE/LDM/CD 模型生成雷达数据
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cm.vae_3d import VAE3D, create_ultra_lightweight_vae_config, create_lightweight_vae_config, create_standard_vae_config
from cm.unet_optimized import OptimizedUNetModel
from cm.karras_diffusion import KarrasDenoiser
from cm.dataset_loader import NTU4DRadLM_VoxelDataset
from torch.utils.data import DataLoader

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def safe_torch_load(path, map_location):
    """兼容不同 PyTorch 版本的 checkpoint 加载逻辑。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # HACK: 低版本 PyTorch 不支持 weights_only 参数，回退到兼容加载路径。
        return torch.load(path, map_location=map_location)
    except Exception as exc:
        # NOTE: 部分历史权重含有 weights_only=True 不接受的对象，按错误信息回退。
        msg = str(exc)
        if "Weights only load failed" in msg or "Unsupported global" in msg:
            return torch.load(path, map_location=map_location)
        raise


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
        
        # NOTE: 变分自编码器（VAE）负责体素空间与潜空间之间的编码/解码。
        print(f"Loading VAE from {vae_path}...")
        self.vae = self._load_vae(vae_path)
        self.vae.eval()
        
        # NOTE: 生成模型在潜空间进行去噪采样。
        print(f"Loading {model_type.upper()} from {model_path}...")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # NOTE: 复用训练时一致的噪声调度参数。
        self.denoiser = KarrasDenoiser(
            sigma_data=0.5,
            sigma_max=80.0,
            sigma_min=0.002,
            loss_norm='l2',
        )
        
        print("Models loaded successfully!")
    
    def _load_vae(self, ckpt_path):
        """加载 VAE 模型"""
        # TODO: 目前默认使用 ultra lightweight 配置，后续可从 checkpoint 元信息自动恢复。
        vae_config = create_ultra_lightweight_vae_config()
        vae = VAE3D(**vae_config).to(self.device)
        
        ckpt = safe_torch_load(ckpt_path, map_location=self.device)
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
        
        ckpt = safe_torch_load(ckpt_path, map_location=self.device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        
        return model
    
    @torch.no_grad()
    def generate(self, condition, num_samples=1, steps=40, sampler='heun', show_sampling_progress=False):
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
        if show_sampling_progress:
            print(f"\nGenerating {num_samples} samples with {steps} steps ({sampler} sampler)...")
        
        # NOTE: 条件分支将雷达观测编码到潜空间，保证与采样噪声同尺度。
        if condition is not None:
            condition = condition.to(self.device)
            z_cond = self.vae.get_latent(condition)
            # FIXME: 噪声潜变量与条件潜变量形状必须严格一致，否则拼接后会在 UNet 前向报错。
            z_shape = tuple(z_cond.shape)
            num_samples = z_shape[0]
        else:
            # NOTE: 无条件采样时使用固定潜空间尺寸。
            z_shape = (num_samples, 4, 8, 32, 32)
            z_cond = torch.zeros(z_shape, device=self.device)
        
        # NOTE: 按 sigma_max 初始化起始噪声，匹配 Karras 采样假设。
        z_T = torch.randn(z_shape, device=self.device) * self.denoiser.sigma_max
        
        # NOTE: 一致性蒸馏（CD）模式支持一步采样，潜扩散模型（LDM）走多步求解器。
        if self.model_type == 'cd' and steps == 1:
            z_0 = self._cd_sample(z_T, z_cond)
        else:
            z_0 = self._ldm_sample(z_T, z_cond, steps, sampler, show_sampling_progress)
        
        # NOTE: 将潜变量解码回体素空间输出。
        generated = self.vae.decode(z_0)
        
        return generated
    
    def _cd_sample(self, z_T, z_cond):
        """CD 一步采样"""
        model_input = torch.cat([z_T, z_cond], dim=1)
        sigma = torch.ones(z_T.shape[0], device=self.device) * self.denoiser.sigma_max
        z_0 = self.model(model_input, sigma)
        return z_0
    
    def _ldm_sample(self, z_T, z_cond, steps, sampler, show_sampling_progress=False):
        """LDM 多步采样 (Heun/Euler)"""
        # NOTE: 生成单调递减的 sigma 序列，驱动 ODE 采样。
        sigmas = self._get_sigmas(steps)
        z_t = z_T
        
        iterator = range(len(sigmas) - 1)
        if show_sampling_progress:
            iterator = tqdm(iterator, desc="Sampling")

        for i in iterator:
            sigma_t = sigmas[i]
            sigma_next = sigmas[i + 1]
            
            # NOTE: 拼接条件潜变量后预测去噪结果。
            model_input = torch.cat([z_t, z_cond], dim=1)
            sigma_batch = torch.ones(z_t.shape[0], device=self.device) * sigma_t
            denoised = self.model(model_input, sigma_batch)
            
            # NOTE: 常微分方程（ODE）形式导数 d = (x - denoised) / sigma。
            d = (z_t - denoised) / sigma_t
            
            if sampler == 'heun' and i < len(sigmas) - 2:
                # NOTE: 海恩（Heun）二阶法：先欧拉预测，再做一次校正。
                z_next = z_t + d * (sigma_next - sigma_t)
                
                # NOTE: 在预测点重新估计导数。
                model_input_2 = torch.cat([z_next, z_cond], dim=1)
                sigma_batch_2 = torch.ones(z_t.shape[0], device=self.device) * sigma_next
                denoised_2 = self.model(model_input_2, sigma_batch_2)
                d_2 = (z_next - denoised_2) / sigma_next
                
                # NOTE: 两次导数求均值完成校正。
                z_t = z_t + (d + d_2) / 2 * (sigma_next - sigma_t)
            else:
                # NOTE: 欧拉（Euler）一阶法速度快但精度较低。
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


def load_sparse_voxel(filename):
    # NOTE: 历史数据可能以稀疏格式保存，这里恢复为稠密体素以复用统一推理流程。
    data = np.load(filename)
    voxel_grid = np.zeros(data['shape'], dtype=np.float32)
    coords = data['coords']
    if coords.shape[0] > 0:
        voxel_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = data['features']
    return voxel_grid


def load_radar_voxel_as_tensor(path, device):
    if path.endswith('.npz'):
        radar_voxel = load_sparse_voxel(path)
    else:
        radar_voxel = np.load(path).astype(np.float32)

    radar_tensor = torch.from_numpy(radar_voxel).permute(3, 2, 0, 1)
    target_size = (32, 128, 128)
    radar_tensor = F.interpolate(
        radar_tensor.unsqueeze(0),
        size=target_size,
        mode='trilinear',
        align_corners=False
    ).squeeze(0)

    return radar_tensor.to(device)


def voxel_to_pointcloud(voxel, voxel_size, pc_range, occ_threshold=0.5, empty_fallback_topk=0):
    # NOTE: 输入 voxel 形状为 (C, Z, H, W)。
    occ = voxel[0]
    intensity = voxel[1]
    occ_mask = occ > occ_threshold
    used_topk_fallback = False
    if not np.any(occ_mask):
        if empty_fallback_topk <= 0:
            return np.zeros((0, 4), dtype=np.float32), used_topk_fallback

        # HACK: 当阈值筛选后为空点云时，回退到 top-k 占据体素避免评估中断。
        used_topk_fallback = True
        flat_occ = occ.reshape(-1)
        k = int(min(max(empty_fallback_topk, 1), flat_occ.shape[0]))
        topk_idx = np.argpartition(flat_occ, -k)[-k:]
        z_idx, x_idx, y_idx = np.unravel_index(topk_idx, occ.shape)
        x = pc_range[0] + (x_idx + 0.5) * voxel_size[0]
        y = pc_range[1] + (y_idx + 0.5) * voxel_size[1]
        z = pc_range[2] + (z_idx + 0.5) * voxel_size[2]
        inten = intensity[z_idx, x_idx, y_idx]
        pcl = np.stack([x, y, z, inten], axis=1).astype(np.float32)
        return pcl, used_topk_fallback

    z_idx, x_idx, y_idx = np.where(occ_mask)
    x = pc_range[0] + (x_idx + 0.5) * voxel_size[0]
    y = pc_range[1] + (y_idx + 0.5) * voxel_size[1]
    z = pc_range[2] + (z_idx + 0.5) * voxel_size[2]
    inten = intensity[z_idx, x_idx, y_idx]

    return np.stack([x, y, z, inten], axis=1).astype(np.float32), used_topk_fallback


def compute_chamfer(pcl_a, pcl_b):
    # NOTE: 倒角距离（Chamfer）用于衡量生成点云与激光雷达（LiDAR）真值点云的几何一致性。
    if cKDTree is None:
        raise RuntimeError("scipy is required for chamfer distance.")
    if pcl_a.shape[0] == 0 or pcl_b.shape[0] == 0:
        return float('inf')

    tree_a = cKDTree(pcl_a[:, :3])
    tree_b = cKDTree(pcl_b[:, :3])
    dists_a, _ = tree_b.query(pcl_a[:, :3], k=1)
    dists_b, _ = tree_a.query(pcl_b[:, :3], k=1)
    return float(dists_a.mean() + dists_b.mean())


def main():
    parser = argparse.ArgumentParser(description="Radar Data Inference")
    parser.add_argument("--vae_ckpt", type=str, required=True, help="Path to VAE checkpoint")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Path to LDM/CD checkpoint")
    parser.add_argument("--model_type", type=str, default="ldm", choices=["ldm", "cd"], help="Model type")
    parser.add_argument("--dataset_dir", type=str, default="./Data/NTU4DRadLM_Pre", 
                        help="Dataset directory for condition data")
    parser.add_argument("--radar_voxel_dir", type=str, default="",
                        help="If set, load radar_voxel files from this directory and run per-sample inference")
    parser.add_argument("--max_files", type=int, default=0,
                        help="Max number of radar files to run in per-sample mode (0 means all)")
    parser.add_argument("--raw_livox_dir", type=str, default="",
                        help="Raw livox_lidar directory for comparison")
    parser.add_argument("--lidar_index_file", type=str, default="",
                        help="lidar_index_sequence.txt for mapping preprocessed index to raw LiDAR")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--steps", type=int, default=40, help="Sampling steps (LDM: 40, CD: 1-4)")
    parser.add_argument("--sampler", type=str, default="heun", choices=["heun", "euler"], help="Sampler type")
    parser.add_argument("--output_dir", type=str, default="./Result/inference_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--use_condition", action="store_true", help="Use condition data from dataset")
    parser.add_argument("--save_pointcloud", action="store_true", help="Save point cloud .npy per sample")
    parser.add_argument("--save_voxel", action="store_true", help="Save voxel .npy per sample")
    parser.add_argument("--compare_with_lidar", action="store_true", help="Compare with raw livox point clouds")
    parser.add_argument("--occ_threshold", type=float, default=0.5, help="Occupancy threshold for point cloud")
    parser.add_argument("--empty_fallback_topk", type=int, default=0,
                        help="If threshold yields empty point cloud, fallback to top-k occupancy voxels (0 disables)")
    parser.add_argument("--voxel_size", type=float, nargs=3, default=[0.2, 0.2, 0.2],
                        help="Voxel size used in preprocessing")
    parser.add_argument("--pc_range", type=float, nargs=6, default=[0, -20, -6, 120, 20, 10],
                        help="Point cloud range used in preprocessing")
    
    args = parser.parse_args()
    
    # NOTE: 初始化模型与采样器。
    generator = RadarGenerator(
        vae_path=args.vae_ckpt,
        model_path=args.model_ckpt,
        model_type=args.model_type,
        device=args.device,
    )
    
    os.makedirs(args.output_dir, exist_ok=True)

    if args.radar_voxel_dir:
        # NOTE: 逐文件推理模式：每个 radar_voxel 输入生成一个对应输出。
        radar_files = sorted([
            f for f in os.listdir(args.radar_voxel_dir)
            if f.endswith('.npy') or f.endswith('.npz')
        ])

        if args.max_files > 0:
            radar_files = radar_files[:args.max_files]

        if not radar_files:
            raise RuntimeError(f"No radar voxel files found in {args.radar_voxel_dir}")

        lidar_files = []
        lidar_indices = []
        if args.compare_with_lidar:
            # NOTE: 当开启对比评估时，按索引映射原始 LiDAR 帧并导出 CSV 指标。
            if not args.raw_livox_dir:
                raise ValueError("--raw_livox_dir is required when --compare_with_lidar is set")
            lidar_files = sorted([
                f for f in os.listdir(args.raw_livox_dir) if f.endswith('.npy')
            ])
            if args.lidar_index_file:
                with open(args.lidar_index_file, 'r', encoding='utf-8') as f:
                    lidar_indices = [int(line.strip()) for line in f.readlines()]

        csv_path = os.path.join(args.output_dir, "comparison_metrics.csv")
        csv_file = None
        csv_writer = None
        if args.compare_with_lidar:
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["index", "radar_file", "lidar_file", "chamfer"])

        timing_csv_path = os.path.join(args.output_dir, "inference_timing.csv")
        timing_file = open(timing_csv_path, 'w', newline='')
        timing_writer = csv.writer(timing_file)
        timing_writer.writerow(["index", "radar_file", "inference_seconds"])

        log_path = os.path.join(args.output_dir, "inference_runtime.log")
        log_file = open(log_path, 'w', encoding='utf-8')
        log_file.write("=== Inference Runtime Log ===\n")
        log_file.write(f"time: {datetime.now().isoformat()}\n")
        log_file.write(f"model_type: {args.model_type}\n")
        log_file.write(f"vae_ckpt: {args.vae_ckpt}\n")
        log_file.write(f"model_ckpt: {args.model_ckpt}\n")
        log_file.write(f"device: {args.device}\n")
        log_file.write(f"steps: {args.steps}\n")
        log_file.write(f"sampler: {args.sampler}\n")
        log_file.write(f"max_files: {args.max_files}\n")
        log_file.write(f"empty_fallback_topk: {args.empty_fallback_topk}\n")
        log_file.write(f"num_files: {len(radar_files)}\n")
        log_file.write("\n")

        total_infer_sec = 0.0
        fallback_count = 0

        file_pbar = tqdm(
            enumerate(radar_files),
            total=len(radar_files),
            desc="Inferring files",
            unit="file",
        )

        for i, fname in file_pbar:
            radar_path = os.path.join(args.radar_voxel_dir, fname)
            condition_data = load_radar_voxel_as_tensor(radar_path, generator.device)
            condition_data = condition_data.unsqueeze(0)

            file_start = time.perf_counter()

            generated = generator.generate(
                condition=condition_data,
                num_samples=1,
                steps=args.steps,
                sampler=args.sampler,
                show_sampling_progress=False,
            )

            file_infer_sec = time.perf_counter() - file_start
            total_infer_sec += file_infer_sec
            timing_writer.writerow([i, fname, f"{file_infer_sec:.6f}"])
            log_file.write(f"file[{i + 1}/{len(radar_files)}] {fname} infer_sec={file_infer_sec:.6f}\n")
            file_pbar.set_postfix_str(f"{i + 1}/{len(radar_files)} | {file_infer_sec:.3f}s")

            sample = generated[0].detach().cpu().numpy()
            pcl = np.zeros((0, 4), dtype=np.float32)

            if args.save_voxel:
                out_voxel = os.path.join(args.output_dir, f"{os.path.splitext(fname)[0]}_voxel.npy")
                np.save(out_voxel, sample)

            if args.save_pointcloud or args.compare_with_lidar:
                pcl, used_topk_fallback = voxel_to_pointcloud(
                    sample,
                    voxel_size=args.voxel_size,
                    pc_range=args.pc_range,
                    occ_threshold=args.occ_threshold,
                    empty_fallback_topk=args.empty_fallback_topk,
                )
                if used_topk_fallback:
                    fallback_count += 1
                if args.save_pointcloud:
                    out_pcl = os.path.join(args.output_dir, f"{os.path.splitext(fname)[0]}_pcl.npy")
                    np.save(out_pcl, pcl)

            if args.compare_with_lidar:
                lidar_file = None
                if lidar_indices:
                    if i < len(lidar_indices):
                        idx = lidar_indices[i]
                        if idx < len(lidar_files):
                            lidar_file = lidar_files[idx]
                elif i < len(lidar_files):
                    lidar_file = lidar_files[i]

                if lidar_file:
                    lidar_path = os.path.join(args.raw_livox_dir, lidar_file)
                    lidar_pcl = np.load(lidar_path).astype(np.float32)
                    chamfer = compute_chamfer(pcl, lidar_pcl)
                    if csv_writer is not None:
                        csv_writer.writerow([i, fname, lidar_file, f"{chamfer:.6f}"])

        timing_file.flush()
        avg_infer_sec = total_infer_sec / max(len(radar_files), 1)
        log_file.write("\n")
        log_file.write(f"total_infer_sec: {total_infer_sec:.6f}\n")
        log_file.write(f"avg_infer_sec_per_file: {avg_infer_sec:.6f}\n")
        log_file.write(f"topk_fallback_frames: {fallback_count}\n")
        log_file.flush()
        file_pbar.close()

        if csv_file:
            csv_file.close()
            print(f"Saved metrics to {csv_path}")

        timing_file.close()
        log_file.close()
        print(f"Saved timing csv to {timing_csv_path}")
        print(f"Saved runtime log to {log_path}")

    else:
        # NOTE: 简单模式下可从验证集抽取一个条件样本。
        if args.use_condition:
            print(f"Loading dataset from {args.dataset_dir}...")
            dataset = NTU4DRadLM_VoxelDataset(args.dataset_dir, split='val')
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            # FIXME: 当验证集为空时这里会触发 StopIteration，后续可改为显式空集检查。
            condition_data = next(iter(dataloader))[1]
        else:
            condition_data = None

        # NOTE: 运行生成并保存体素结果。
        generated = generator.generate(
            condition=condition_data,
            num_samples=args.num_samples,
            steps=args.steps,
            sampler=args.sampler,
        )

        output_path = os.path.join(args.output_dir, f"{args.model_type}_samples_{args.steps}steps.npy")
        np.save(output_path, generated.cpu().numpy())
        print(f"\nSaved {args.num_samples} samples to {output_path}")
        print(f"Shape: {generated.shape}")
        print(f"Range: [{generated.min():.3f}, {generated.max():.3f}]")


if __name__ == "__main__":
    main()
