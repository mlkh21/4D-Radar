import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import logging
from typing import Optional

# NOTE: 可选数据增强模块；当依赖缺失时退化为不增强模式。
try:
    from .augmentation import ComposedAugmentation, VoxelAugmentation, MixupAugmentation
except ImportError:
    ComposedAugmentation = None
    VoxelAugmentation = None
    MixupAugmentation = None

logger = logging.getLogger(__name__)
EPS = 1e-6


def load_sparse_voxel(filename):
    # NOTE: 稀疏存储格式恢复为稠密体素网格，便于后续统一插值与张量变换。
    data = np.load(filename)
    voxel_grid = np.zeros(data['shape'], dtype=np.float32)
    coords = data['coords']
    if coords.shape[0] > 0:
        voxel_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = data['features']
    return voxel_grid


def resize_voxel_channels(voxel_tensor: torch.Tensor, target_size, mask_channel: Optional[int] = None) -> torch.Tensor:
    """
    Resize sparse voxel channels with channel-aware rules.

    Occupancy and mask-like channels use max pooling semantics to preserve sparse positives.
    Feature channels use occupancy-weighted interpolation so zeros outside occupied cells do not dominate.
    """
    if voxel_tensor.ndim != 4:
        raise ValueError(f"Expected (C, Z, H, W), got {tuple(voxel_tensor.shape)}")

    x = voxel_tensor.unsqueeze(0).float()
    occ = x[:, 0:1]

    resized_occ = F.adaptive_max_pool3d(occ, target_size)
    outputs = [resized_occ]

    occ_density = F.interpolate(occ, size=target_size, mode='trilinear', align_corners=False)

    for ch in range(1, x.shape[1]):
        channel = x[:, ch : ch + 1]
        if mask_channel is not None and ch == mask_channel:
            outputs.append(F.adaptive_max_pool3d(channel, target_size))
            continue

        weighted = F.interpolate(channel * occ, size=target_size, mode='trilinear', align_corners=False)
        outputs.append(weighted / occ_density.clamp_min(EPS))

    return torch.cat(outputs, dim=1).squeeze(0)

class NTU4DRadLM_VoxelDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, return_path=False, alignment_size=32,
                 use_augmentation=True, augmentation_config=None):
        """
        输入:
            root_dir (str) - 数据集根目录，例如 ".../NTU4DRadLM_Pre"
            split (str) - 'train', 'val', 或 'test' (目前简单实现为读取所有场景，后续可扩展)
            transform (callable, optional) - 可选的变换
            return_path (bool) - 是否返回文件路径
            alignment_size (int) - 张量填充的对齐大小（U-Net要求32的倍数）
            use_augmentation (bool) - 是否使用数据增强（仅训练集）
            augmentation_config (dict) - 数据增强配置
        输出:
            无
        作用: 初始化数据集，加载所有场景的文件路径。
        逻辑:
        1. 检查根目录是否存在。
        2. 遍历根目录下的所有场景文件夹。
        3. 根据 split 参数划分训练集和验证集。
        4. 遍历目标场景，收集 radar_voxel 和 target_voxel 的文件路径对。
        5. 将路径对存储在 self.samples 列表中。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.return_path = return_path
        self.alignment_size = alignment_size
        self.samples = []
        self.split = split
        
        # NOTE: 仅训练集启用增强，验证/测试保持数据分布稳定。
        self.augmentation = None
        if (
            use_augmentation
            and split == 'train'
            and ComposedAugmentation is not None
            and VoxelAugmentation is not None
        ):
            default_config: dict = {
                'enable_flip': False,
                'enable_rotate': False,
                'flip_prob': 0.0,
                'rotate_prob': 0.0,
                'noise_prob': 0.2,
                'noise_std': 0.02,
                'dropout_prob': 0.1,
                'point_dropout_rate': 0.05,
                'intensity_jitter_prob': 0.1,
                'doppler_jitter_prob': 0.05
            }
            if augmentation_config:
                default_config.update(augmentation_config)
            
            self.augmentation = ComposedAugmentation([
                VoxelAugmentation(**default_config)
            ])
            logger.info(f"数据增强已启用: {default_config}")
        
        # NOTE: 扫描场景目录并构建样本索引。
        if not os.path.exists(root_dir):
            print(f"Warning: Root dir {root_dir} does not exist.")
            return

        scenes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        # TODO: 当前按 8:2 自动切分场景，后续建议改为可配置的显式场景列表。
        if len(scenes) == 1:
            # NOTE: 单场景数据集无法切分，直接复用同一场景。
            target_scenes = scenes
            print(f"Warning: Only 1 scene found. Using it for {split}.")
        else:
            split_idx = int(len(scenes) * 0.8)
            if split_idx == 0:
                split_idx = 1  # NOTE: 保证训练集至少包含 1 个场景。
            
            if split == 'train':
                target_scenes = scenes[:split_idx]
            else:
                target_scenes = scenes[split_idx:]
            
        print(f"Loading {split} dataset from {len(target_scenes)} scenes: {target_scenes}")

        for scene in target_scenes:
            radar_voxel_dir = os.path.join(root_dir, scene, "radar_voxel")
            target_voxel_dir = os.path.join(root_dir, scene, "target_voxel")
            
            if not os.path.exists(radar_voxel_dir) or not os.path.exists(target_voxel_dir):
                continue
                
            files = sorted([f for f in os.listdir(radar_voxel_dir) if f.endswith('.npy') or f.endswith('.npz')])
            
            for f in files:
                radar_path = os.path.join(radar_voxel_dir, f)
                target_path = os.path.join(target_voxel_dir, f)
                
                if not os.path.exists(target_path):
                     # HACK: 兼容历史数据清洗后 .npy/.npz 扩展名不一致的情况。
                     if f.endswith('.npy'):
                         target_path = os.path.join(target_voxel_dir, f.replace('.npy', '.npz'))
                     elif f.endswith('.npz'):
                         target_path = os.path.join(target_voxel_dir, f.replace('.npz', '.npy'))

                if os.path.exists(target_path):
                    self.samples.append((radar_path, target_path))
        
        print(f"Found {len(self.samples)} samples for {split}.")

    def __len__(self):
        """
        输入:
            无
        输出:
            (int) - 数据集样本数量
        作用: 返回数据集的大小。
        逻辑:
        1. 返回 self.samples 列表的长度。
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        输入:
            idx (int) - 样本索引
        输出:
            target_tensor (torch.Tensor) - 目标体素张量 (C, Z, H, W)
            radar_tensor (torch.Tensor) - 雷达体素张量 (C, Z, H, W)
            target_path (str, optional) - 目标体素文件路径 (如果 return_path 为 True)
        作用: 获取指定索引的样本数据。
        逻辑:
        1. 根据索引获取 radar_path 和 target_path。
        2. 加载 .npy 文件为 numpy 数组。
        3. 将维度从 (H, W, Z, C) 转换为 (C, Z, H, W)。
        4. 对 Z, H, W 维度进行 Padding，使其成为 32 的倍数。
        5. 返回目标张量和雷达张量（以及可选的路径）。
        """

        radar_path, target_path = self.samples[idx]

        # NOTE: 加载输入体素与目标体素，支持 .npy/.npz 两种格式。
        try:
            # NOTE: 雷达体素 radar_voxel 形状为 (H, W, Z, 4)，通道为 [Occ, Int, Dop, Var]。
            if radar_path.endswith('.npz'):
                radar_voxel = load_sparse_voxel(radar_path)
            else:
                radar_voxel = np.load(radar_path).astype(np.float32)
            
            # NOTE: 目标体素 target_voxel 形状为 (H, W, Z, 4)，通道为 [Occ, Int, Dop, Mask]。
            if target_path.endswith('.npz'):
                target_voxel = load_sparse_voxel(target_path)
            else:
                target_voxel = np.load(target_path).astype(np.float32)
        except FileNotFoundError as e:
            logger.error(f"文件未找到: {e}")
            raise
        except Exception as e:
            logger.error(f"加载数据失败 (idx={idx}): {e}")
            raise RuntimeError(f"无法加载样本 {idx}") from e

        # NOTE: 形状不一致时仅告警，仍继续走统一插值流程。
        if radar_voxel.shape != target_voxel.shape:
            logger.warning(
                f"形状不匹配: radar={radar_voxel.shape}, target={target_voxel.shape}"
            )
        
        # NOTE: 转为 (C, Z, H, W) 以匹配 3D UNet/VAE 的通道优先输入。
        radar_tensor = torch.from_numpy(radar_voxel).permute(3, 2, 0, 1)
        
        target_tensor = torch.from_numpy(target_voxel).permute(3, 2, 0, 1)
        
        # NOTE: 统一重采样到固定空间尺寸，避免不同场景分辨率导致 batch 拼接失败。
        target_size = (32, 128, 128)  # (Z, H, W)
        
        radar_tensor = resize_voxel_channels(radar_tensor, target_size)
        target_tensor = resize_voxel_channels(target_tensor, target_size, mask_channel=3)

        if self.transform:
            # TODO: 预留 transform 钩子；当前分支尚未接入具体变换实现。
            pass
        
        # NOTE: 训练阶段做成对增强，保持 target 与 condition 的空间一致性。
        if self.augmentation is not None:
            target_tensor, radar_tensor = self.augmentation(target_tensor, radar_tensor)
            
        # NOTE: 返回顺序固定为 (target, radar[, path])，训练代码依赖该约定。
        if self.return_path:
            return target_tensor, radar_tensor, target_path
        else:
            return target_tensor, radar_tensor

if __name__ == "__main__":
    # NOTE: 最小化自检入口，用于验证数据路径与输出形状。
    dataset_path = "./Data/NTU4DRadLM_Pre"
    ds = NTU4DRadLM_VoxelDataset(dataset_path, split='train', return_path=True)
    if len(ds) > 0:
        sample = ds[0]
        if len(sample) == 3:
            t, r, p = sample
        else:
            t, r = sample
            p = ""
        print(f"成功! 加载样本 0。")
        print(f"目标 (GT) 形状: {t.shape}")
        print(f"雷达 (Cond) 形状: {r.shape}")
        print(f"路径: {p}")
    else:
        print("错误: 数据集为空。请检查 dataset_path。")
