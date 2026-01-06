import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class NTU4DRadLM_VoxelDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, return_path=False, alignment_size=32):
        """
        输入:
            root_dir (str) - 数据集根目录，例如 ".../NTU4DRadLM_Pre"
            split (str) - 'train', 'val', 或 'test' (目前简单实现为读取所有场景，后续可扩展)
            transform (callable, optional) - 可选的变换
            return_path (bool) - 是否返回文件路径
            alignment_size (int) - 张量填充的对齐大小（U-Net要求32的倍数）
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
        
        # 遍历所有场景目录
        if not os.path.exists(root_dir):
            print(f"Warning: Root dir {root_dir} does not exist.")
            return

        scenes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        # 简单的划分逻辑示例 (80% train, 20% val)
        # 实际使用时建议手动指定场景列表
        if len(scenes) == 1:
            # 如果只有一个场景，直接用于训练（或测试）
            target_scenes = scenes
            print(f"Warning: Only 1 scene found. Using it for {split}.")
        else:
            split_idx = int(len(scenes) * 0.8)
            if split_idx == 0: split_idx = 1 # 确保至少有一个场景用于训练
            
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
                
            files = sorted([f for f in os.listdir(radar_voxel_dir) if f.endswith('.npy')])
            
            for f in files:
                radar_path = os.path.join(radar_voxel_dir, f)
                target_path = os.path.join(target_voxel_dir, f)
                
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
        5. 返回处理后的张量。
        """
        radar_path, target_path = self.samples[idx]
        
        # 加载数据
        try:
            # radar_voxel 形状: (H, W, Z, 4) -> [Occ, Int, Dop, Var]
            radar_voxel = np.load(radar_path).astype(np.float32)
            
            # target_voxel 形状: (H, W, Z, 4) -> [Occ, Int, Dop, Mask]
            target_voxel = np.load(target_path).astype(np.float32)
        except FileNotFoundError as e:
            logger.error(f"文件未找到: {e}")
            raise
        except Exception as e:
            logger.error(f"加载数据失败 (idx={idx}): {e}")
            raise RuntimeError(f"无法加载样本 {idx}") from e

        # 数据验证
        if radar_voxel.shape != target_voxel.shape:
            logger.warning(
                f"形状不匹配: radar={radar_voxel.shape}, target={target_voxel.shape}"
            )
        
        # 转换为 PyTorch 格式: (C, Z, H, W) 以适配 3D U-Net
        # 输入: (H, W, Z, 4)
        radar_tensor = torch.from_numpy(radar_voxel).permute(3, 2, 0, 1)
        
        # 目标: (C, Z, H, W)
        target_tensor = torch.from_numpy(target_voxel).permute(3, 2, 0, 1)
        
        # 填充至 alignment_size 的倍数以确保与 U-Net 下采样兼容
        # 输入形状: (C, Z, H, W)
        # F.pad 参数: (w_left, w_right, h_left, h_right, z_left, z_right)
        align = self.alignment_size
        pad_z = (align - radar_tensor.shape[1] % align) % align
        pad_h = (align - radar_tensor.shape[2] % align) % align
        pad_w = (align - radar_tensor.shape[3] % align) % align
        
        if pad_h > 0 or pad_w > 0 or pad_z > 0:
            radar_tensor = F.pad(radar_tensor, (0, pad_w, 0, pad_h, 0, pad_z))
            target_tensor = F.pad(target_tensor, (0, pad_w, 0, pad_h, 0, pad_z))

        if self.transform:
            # 如果有变换，则应用变换
            pass
            
        # 先返回目标（作为 batch），然后返回雷达（作为条件）
        if self.return_path:
            return target_tensor, radar_tensor, target_path
        else:
            return target_tensor, radar_tensor

if __name__ == "__main__":
    # Test the dataset
    dataset_path = "./NTU4DRadLM_pre_processing/NTU4DRadLM_Pre"
    ds = NTU4DRadLM_VoxelDataset(dataset_path, split='train')
    if len(ds) > 0:
        t, r = ds[0] # !注意: target, radar
        print(f"Target shape: {t.shape}, Radar shape: {r.shape}")
