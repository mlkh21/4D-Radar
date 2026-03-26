#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化推理结果
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_radar_data(data, save_path=None, title="Radar Voxel"):
    """
    可视化雷达体素数据
    
    Args:
        data: (4, 32, 128, 128) - [Occ, Int, Dop, Var, D, H, W]
        save_path: 保存路径
        title: 图表标题
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(title, fontsize=16)
    
    channel_names = ['Occupancy', 'Intensity', 'Doppler', 'Variance']
    
    for ch in range(4):
        # NOTE: 取中间切片
        mid_depth = data.shape[1] // 2
        slice_data = data[ch, mid_depth]  # (128, 128)
        
        # NOTE: 顶视图
        axes[0, ch].imshow(slice_data, cmap='viridis', aspect='auto')
        axes[0, ch].set_title(f'{channel_names[ch]} - Top View')
        axes[0, ch].axis('off')
        
        # NOTE: 侧视图 (沿宽度方向的最大投影)
        side_view = np.max(data[ch], axis=2)  # (32, 128)
        axes[1, ch].imshow(side_view, cmap='viridis', aspect='auto')
        axes[1, ch].set_title(f'{channel_names[ch]} - Side View')
        axes[1, ch].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_samples(ldm_data, cd_data, save_path=None):
    """
    对比 LDM 和 CD 生成的样本
    
    Args:
        ldm_data: LDM 生成的数据 (4, 32, 128, 128)
        cd_data: CD 生成的数据 (4, 32, 128, 128)
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('LDM vs CD Comparison', fontsize=16)
    
    channel_names = ['Occupancy', 'Intensity', 'Doppler', 'Variance']
    mid_depth = ldm_data.shape[1] // 2
    
    for ch in range(4):
        # NOTE: 潜扩散模型（LDM）模式
        axes[0, ch].imshow(ldm_data[ch, mid_depth], cmap='viridis', aspect='auto')
        axes[0, ch].set_title(f'{channel_names[ch]} - LDM (40 steps)')
        axes[0, ch].axis('off')
        
        # NOTE: 一致性蒸馏（CD）模式
        axes[1, ch].imshow(cd_data[ch, mid_depth], cmap='viridis', aspect='auto')
        axes[1, ch].set_title(f'{channel_names[ch]} - CD (1 step)')
        axes[1, ch].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_statistics(data, name="Sample"):
    """
    分析数据统计信息
    
    Args:
        data: (N, 4, 32, 128, 128)
        name: 数据名称
    """
    print(f"\n{'='*50}")
    print(f"{name} Statistics")
    print(f"{'='*50}")
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"\nChannel Statistics:")
    
    channel_names = ['Occupancy', 'Intensity', 'Doppler', 'Variance']
    for ch, ch_name in enumerate(channel_names):
        ch_data = data[:, ch]
        print(f"\n  {ch_name}:")
        print(f"    Mean: {ch_data.mean():.4f}")
        print(f"    Std:  {ch_data.std():.4f}")
        print(f"    Min:  {ch_data.min():.4f}")
        print(f"    Max:  {ch_data.max():.4f}")
        print(f"    Non-zero ratio: {(ch_data != 0).sum() / ch_data.size * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Visualize Radar Inference Results")
    parser.add_argument("--input", type=str, required=True, help="Path to .npy file")
    parser.add_argument("--compare", type=str, help="Path to comparison .npy file")
    parser.add_argument("--output_dir", type=str, default="./visualizations", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    
    args = parser.parse_args()
    
    # NOTE: 加载数据
    print(f"Loading data from {args.input}...")
    data = np.load(args.input)
    
    if data.ndim == 5:
        # NOTE: (N, 4, 32, 128, 128)
        print(f"Loaded {data.shape[0]} samples")
        analyze_statistics(data, name=Path(args.input).stem)
    else:
        raise ValueError(f"Expected 5D array (N,4,32,128,128), got shape {data.shape}")
    
    # NOTE: 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # NOTE: 可视化多个样本
    num_vis = min(args.num_samples, data.shape[0])
    for i in range(num_vis):
        save_path = output_dir / f"sample_{i:03d}.png"
        visualize_radar_data(
            data[i],
            save_path=save_path,
            title=f"Sample {i} - {Path(args.input).stem}"
        )
    
    # NOTE: 对比模式
    if args.compare:
        print(f"\nLoading comparison data from {args.compare}...")
        compare_data = np.load(args.compare)
        analyze_statistics(compare_data, name=Path(args.compare).stem)
        
        # NOTE: 对比第一个样本
        if data.shape[0] > 0 and compare_data.shape[0] > 0:
            save_path = output_dir / "comparison.png"
            compare_samples(
                data[0],
                compare_data[0],
                save_path=save_path
            )
    
    print(f"\n{'='*50}")
    print(f"Visualization complete! Saved to {output_dir}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
