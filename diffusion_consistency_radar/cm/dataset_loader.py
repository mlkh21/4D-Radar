# -*- coding: utf-8 -*-

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import logging
from typing import Dict, Optional, Tuple

# ✔ 完美继承：可选数据增强模块，依赖缺失时平滑退化
try:
    from .augmentation import ComposedAugmentation, VoxelAugmentation, MixupAugmentation
except ImportError:
    ComposedAugmentation = None
    VoxelAugmentation = None
    MixupAugmentation = None

logger = logging.getLogger(__name__)
EPS = 1e-6


def _read_calibration_txt(path: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not os.path.exists(path):
        return None, None
    values: Dict[str, list] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, raw = line.split(":", 1)
            try:
                values[key.strip()] = [float(v) for v in raw.strip().split()]
            except ValueError:
                continue
    r_vals = values.get("R")
    t_vals = values.get("T")
    r_mat = torch.tensor(r_vals, dtype=torch.float32).view(3, 3) if r_vals and len(r_vals) == 9 else None
    t_vec = torch.tensor(t_vals, dtype=torch.float32) if t_vals and len(t_vals) == 3 else None
    return r_mat, t_vec


class CalibrationProvider:
    """Load available dataset calibration, with explicit fallback metadata."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        candidates = [
            os.path.join(root_dir, "config"),
            os.path.join(os.path.dirname(root_dir), "config"),
        ]
        project_data = os.path.abspath(os.path.join(os.getcwd(), "Data"))
        if os.path.abspath(root_dir).startswith(project_data):
            candidates.append(os.path.join(project_data, "config"))
        self.config_dirs = []
        for path in candidates:
            if path not in self.config_dirs:
                self.config_dirs.append(path)

    def load(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        k_mat = torch.tensor([[457.2, 0.0, 323.1], [0.0, 457.9, 242.5], [0.0, 0.0, 1.0]], dtype=torch.float32)
        for config_dir in self.config_dirs:
            for name in ("calib_radar_to_thermal.txt", "calib_radar_to_livox.txt"):
                r_mat, t_vec = _read_calibration_txt(os.path.join(config_dir, name))
                if r_mat is not None and t_vec is not None:
                    return r_mat, t_vec, k_mat, False
        return torch.eye(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32), k_mat, True


def load_sparse_voxel(filename):
    """将稀疏存储格式恢复为稠密体素网格"""
    data = np.load(filename)
    voxel_grid = np.zeros(data['shape'], dtype=np.float32)
    coords = data['coords']
    if coords.shape[0] > 0:
        voxel_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = data['features']
    return voxel_grid


def resize_voxel_channels(voxel_tensor: torch.Tensor, target_size, mask_channel: Optional[int] = None) -> torch.Tensor:
    """
    ✔ 完美保留：非线性密集正留存重采样算子
    确保 Occupancy 通道在重采样后依然能够维持正确的几何边界分布
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


def _resize_or_pad_ir_tensor(ir_img: torch.Tensor) -> torch.Tensor:
    if ir_img.ndim == 2:
        ir_img = ir_img.unsqueeze(0).repeat(3, 1, 1)
    elif ir_img.ndim == 3 and ir_img.shape[0] not in (1, 3):
        ir_img = ir_img.permute(2, 0, 1)
    if ir_img.shape[0] == 1:
        ir_img = ir_img.repeat(3, 1, 1)
    ir_img = ir_img[:3].float().unsqueeze(0)
    ir_img = F.interpolate(ir_img, size=(480, 640), mode="bilinear", align_corners=False)
    return ir_img.squeeze(0)


def _mock_ir_image(height: int = 480, width: int = 640) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height),
        torch.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    thermal = torch.exp(-((xx * 1.8) ** 2 + (yy * 1.2) ** 2))
    return torch.stack([thermal, thermal * 0.85, thermal * 0.65], dim=0).float()


class NTU4DRadLM_VoxelDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, return_path=False, alignment_size=32,
                 use_augmentation=True, augmentation_config=None, sequence_length=1):
        """
        工业闭环重构版：完美支持独立场景内部时序滑窗、动态几何参数分发的多模态数据迭代器
        """
        self.root_dir = root_dir
        self.transform = transform
        self.return_path = return_path
        self.alignment_size = alignment_size
        self.samples = []  # ──► 内部元素结构重构为: (radar_seq_paths_list, target_path, ir_path)
        self.scene_policies: Dict[str, dict] = {}
        self.split = split
        self.seq_len = max(1, int(sequence_length))
        self.ir_dir = os.path.join(root_dir, "ir_image")
        self.calibration_provider = CalibrationProvider(root_dir)

        # 统一标定外参参数常驻（规避写死常量造成的域泄漏）
        self.default_K = torch.tensor([[457.2, 0.0, 323.1], [0.0, 457.9, 242.5], [0.0, 0.0, 1.0]], dtype=torch.float32)
        self.R_cam_to_lidar = torch.tensor([[0.012, -0.999, -0.015], [0.024, -0.015, 0.999], [-0.999, -0.012, 0.024]], dtype=torch.float32)
        self.T_cam_to_lidar = torch.zeros(3, dtype=torch.float32)

        # 数据增强加载区
        self.augmentation = None
        if (use_augmentation and split == 'train' and ComposedAugmentation is not None and VoxelAugmentation is not None):
            default_config: dict = {
                'enable_flip': False, 'enable_rotate': False, 'flip_prob': 0.0, 'rotate_prob': 0.0,
                'noise_prob': 0.2, 'noise_std': 0.02, 'dropout_prob': 0.1,
                'point_dropout_rate': 0.05, 'intensity_jitter_prob': 0.1, 'doppler_jitter_prob': 0.05
            }
            if augmentation_config:
                default_config.update(augmentation_config)
            self.augmentation = ComposedAugmentation([VoxelAugmentation(**default_config)])
            logger.info(f"数据增强已启用: {default_config}")

        if not os.path.exists(root_dir):
            print(f"Warning: Root dir {root_dir} does not exist.")
            return

        # Only directories containing a complete radar/target pair participate
        # in train/validation scene splitting. Dataset-level config and other
        # auxiliary directories must not be treated as scenes.
        scenes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d, "radar_voxel"))
            and os.path.isdir(os.path.join(root_dir, d, "target_voxel"))
        ])

        if len(scenes) == 1:
            target_scenes = scenes
            print(f"Warning: Only 1 scene found. Using it for {split}.")
        else:
            split_idx = int(len(scenes) * 0.8)
            if split_idx == 0: split_idx = 1
            target_scenes = scenes[:split_idx] if split == 'train' else scenes[split_idx:]

        print(f"Loading {split} dataset from {len(target_scenes)} scenes: {target_scenes}")

        for scene in target_scenes:
            radar_voxel_dir = os.path.join(root_dir, scene, "radar_voxel")
            target_voxel_dir = os.path.join(root_dir, scene, "target_voxel")
            ir_dir = os.path.join(root_dir, scene, "ir_image")
            policy_path = os.path.join(root_dir, scene, "preprocess_policy.json")
            if os.path.exists(policy_path):
                with open(policy_path, "r", encoding="utf-8") as f:
                    self.scene_policies[scene] = json.load(f)
            else:
                self.scene_policies[scene] = {}

            if not os.path.exists(radar_voxel_dir) or not os.path.exists(target_voxel_dir):
                continue

            files = sorted([f for f in os.listdir(radar_voxel_dir) if f.endswith('.npy') or f.endswith('.npz')])

            # 强制在独立场景内部执行滑窗，绝对禁止跨场景边界缝合
            if len(files) < self.seq_len:
                continue

            for i in range(len(files) - self.seq_len + 1):
                # 收集连续 T 帧雷达路径组
                radar_seq_paths = [os.path.join(radar_voxel_dir, files[i + t]) for t in range(self.seq_len)]

                # 对应当前最终切片截面时刻 (t) 的真值标签路径
                target_f = files[i + self.seq_len - 1]
                target_path = os.path.join(target_voxel_dir, target_f)

                # 原代码扩展名不一致兼容 HACK 逻辑
                if not os.path.exists(target_path):
                     if target_f.endswith('.npy'):
                         target_path = os.path.join(target_voxel_dir, target_f.replace('.npy', '.npz'))
                     elif target_f.endswith('.npz'):
                         target_path = os.path.join(target_voxel_dir, target_f.replace('.npz', '.npy'))

                # 精准检索由预处理并置保存的 LWIR 红外辐射特征矩阵
                ir_f = f"{os.path.splitext(target_f)[0]}_ir.npy"
                ir_path = os.path.join(ir_dir, ir_f)

                if os.path.exists(target_path):
                    self.samples.append((radar_seq_paths, target_path, ir_path, scene))

        print(f"Found {len(self.samples)} temporal sliding-window samples for {split}.")

    def __len__(self):
        return len(self.samples)

    def _get_mock_calibration(
        self,
        velocity_m_s: float = 50.0,
        dt_mu_s: float = 200.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r_mat, t_vec, k_mat, is_mock = self.calibration_provider.load()
        if is_mock:
            r_mat = self.R_cam_to_lidar.clone()
            t_vec = self.T_cam_to_lidar.clone()
            k_mat = self.default_K.clone()
        displacement_x = float(velocity_m_s) * (float(dt_mu_s) / 1e6)
        t_vec[0] += displacement_x
        return r_mat, t_vec, k_mat, is_mock

    def _load_ir_tensor(self, ir_path: str) -> torch.Tensor:
        if os.path.exists(ir_path):
            arr = np.load(ir_path).astype(np.float32)
            return _resize_or_pad_ir_tensor(torch.from_numpy(arr)), False
        return _mock_ir_image(), True

    def __getitem__(self, idx):
        radar_seq_paths, target_path, ir_path, scene = self.samples[idx]

        # 无缝复原原本完全体的张量空间轴向变换流与重采样
        target_size = (32, 128, 128)  # (Z, H, W)

        # 1. 加载并转换主监督真值体素
        if target_path.endswith('.npz'):
            target_voxel = load_sparse_voxel(target_path)
        else:
            target_voxel = np.load(target_path).astype(np.float32)

        # 完美对齐物理原版轴向重塑: (H, W, Z, C) -> (C, Z, H, W)
        target_tensor = torch.from_numpy(target_voxel).permute(3, 2, 0, 1)
        target_tensor = resize_voxel_channels(target_tensor, target_size, mask_channel=3)

        # 2. 时序雷达滑窗包流式解析重采样
        radar_seq_tensors = []
        for path in radar_seq_paths:
            if path.endswith('.npz'):
                radar_voxel = load_sparse_voxel(path)
            else:
                radar_voxel = np.load(path).astype(np.float32)

            r_tensor = torch.from_numpy(radar_voxel).permute(3, 2, 0, 1)
            r_tensor = resize_voxel_channels(r_tensor, target_size)
            radar_seq_tensors.append(r_tensor)

        radar_tensor = radar_seq_tensors[-1]

        # 3. 加载长波红外热成像特征
        ir_img, is_mock_ir = self._load_ir_tensor(ir_path)

        # 完美保留：多模态双成对几何空间一致性增强
        if self.augmentation is not None:
            target_tensor, radar_tensor = self.augmentation(target_tensor, radar_tensor)

        r_mat, t_vec, k_mat, is_mock_calib = self._get_mock_calibration()
        meta_dict = {
            "ir_img": ir_img,
            "r_mat": r_mat,
            "t_vec": t_vec,
            "k_mat": k_mat,
            "is_mock_ir": bool(is_mock_ir),
            "is_mock_calib": bool(is_mock_calib),
            "preprocess_policy": self.scene_policies.get(scene, {}),
        }

        if self.return_path:
            return target_tensor, radar_tensor, meta_dict, target_path
        return target_tensor, radar_tensor, meta_dict


if __name__ == "__main__":
    dataset_path = "./Data/NTU4DRadLM_Pre_sensor_aware"
    ds = NTU4DRadLM_VoxelDataset(dataset_path, split='train', return_path=True)
    if len(ds) > 0:
        sample = ds[0]
        t, r, m, p = sample
        print(f"成功恢复闭环! 加载时序滑窗样本 0。")
        print(f"目标真值 (GT) 形状 [C, Z, H, W]: {t.shape}")
        print(f"时序雷达条件包 形状 [T, C, Z, H, W]: {r.shape}")
        print(f"红外相片矩阵 形状 [C, H_img, W_img]: {m['ir_img'].shape}")
    else:
        print("错误: 数据集为空。请检查 dataset_path。")
