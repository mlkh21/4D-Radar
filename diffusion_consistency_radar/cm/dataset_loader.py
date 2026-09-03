# -*- coding: utf-8 -*-

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, default_collate
import logging
from typing import Dict, Optional, Sequence, Tuple

try:
    from ..observed_mask import (
        PERSISTED_OBSERVED_MASK_SOURCE,
        build_lidar_observed_mask,
        load_observed_mask,
    )
except (ImportError, ValueError):  # 兼容把 diffusion_consistency_radar 加入 sys.path 的旧脚本
    from observed_mask import (  # type: ignore
        PERSISTED_OBSERVED_MASK_SOURCE,
        build_lidar_observed_mask,
        load_observed_mask,
    )

try:
    from ..radar_normalization import (
        LEGACY_RADAR_NORMALIZATION_PROTOCOL,
        RADAR_NORMALIZATION_PROTOCOL,
        RadarNormalizationError,
        apply_radar_normalization,
        validate_radar_normalization_sha256,
        validate_radar_normalization_spec,
    )
except (ImportError, ValueError):  # 兼容把 diffusion_consistency_radar 加入 sys.path 的旧脚本
    from radar_normalization import (  # type: ignore
        LEGACY_RADAR_NORMALIZATION_PROTOCOL,
        RADAR_NORMALIZATION_PROTOCOL,
        RadarNormalizationError,
        apply_radar_normalization,
        validate_radar_normalization_sha256,
        validate_radar_normalization_spec,
    )

try:
    from ..radar_statistics import (
        RADAR_STATISTICS_PROTOCOL,
        RADAR_RESIZE_AGGREGATION,
        RADAR_RESIZE_AGGREGATION_V1,
        SUPPORTED_RADAR_STATISTICS_PROTOCOLS,
        load_sparse_radar_voxel,
        load_sparse_radar_voxel_with_statistics,
        validate_sparse_radar_statistics,
    )
except (ImportError, ValueError):  # 兼容把 diffusion_consistency_radar 加入 sys.path 的旧脚本
    from radar_statistics import (  # type: ignore
        RADAR_STATISTICS_PROTOCOL,
        RADAR_RESIZE_AGGREGATION,
        RADAR_RESIZE_AGGREGATION_V1,
        SUPPORTED_RADAR_STATISTICS_PROTOCOLS,
        load_sparse_radar_voxel,
        load_sparse_radar_voxel_with_statistics,
        validate_sparse_radar_statistics,
    )

try:
    import cv2
except ImportError:  # pragma: no cover - 仅无 OpenCV 的轻量环境触发
    cv2 = None

# ✔ 完美继承：可选数据增强模块，依赖缺失时平滑退化
try:
    from .augmentation import ComposedAugmentation, VoxelAugmentation, MixupAugmentation
except ImportError:
    ComposedAugmentation = None
    VoxelAugmentation = None
    MixupAugmentation = None

logger = logging.getLogger(__name__)
EPS = 1e-6
DEFAULT_PC_RANGE = (0.0, -20.0, -6.0, 120.0, 20.0, 10.0)
DEFAULT_TARGET_SIZE = (32, 128, 128)
THERMAL_OUTPUT_SIZE = (640, 480)  # (width, height)，与模型 IR 输入一致
DEFAULT_THERMAL_K = np.asarray(
    [[457.2, 0.0, 323.1], [0.0, 457.9, 242.5], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


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


def _validate_extrinsic(
    r_mat: torch.Tensor,
    t_vec: torch.Tensor,
    *,
    path: str,
) -> None:
    """拒绝非有限、非旋转或维度错误的外参。"""
    if tuple(r_mat.shape) != (3, 3) or tuple(t_vec.shape) != (3,):
        raise ValueError(f"外参 R/T 维度无效: {path}")
    if not torch.isfinite(r_mat).all() or not torch.isfinite(t_vec).all():
        raise ValueError(f"外参 R/T 含非有限数: {path}")
    identity = torch.eye(3, dtype=r_mat.dtype)
    if not torch.allclose(r_mat @ r_mat.T, identity, atol=5e-3, rtol=0.0):
        raise ValueError(f"外参 R 不是正交旋转矩阵: {path}")
    determinant = float(torch.det(r_mat))
    if abs(determinant - 1.0) > 5e-3:
        raise ValueError(f"外参 R determinant 必须接近 1，实际为 {determinant}: {path}")


def _read_thermal_camera_calibration(path: str):
    """读取 thermal 相机的原始尺寸、K 和畸变系数。"""
    if not os.path.exists(path):
        return None
    values: Dict[str, list] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, raw = line.split(":", 1)
            try:
                values[key.strip()] = [float(value) for value in raw.strip().split()]
            except ValueError:
                continue
    size = values.get("S_00") or values.get("S")
    k_values = values.get("K_00") or values.get("K")
    d_values = values.get("D_00") or values.get("D")
    if size is None or len(size) != 2 or k_values is None or len(k_values) != 9:
        return None
    if d_values is None or len(d_values) < 4:
        return None
    raw_width, raw_height = (int(round(value)) for value in size)
    if raw_width <= 0 or raw_height <= 0:
        return None
    k_raw = np.asarray(k_values, dtype=np.float32).reshape(3, 3)
    distortion = np.asarray(d_values, dtype=np.float32).reshape(-1)
    if not np.isfinite(k_raw).all() or not np.isfinite(distortion).all():
        return None
    output_width, output_height = THERMAL_OUTPUT_SIZE
    scaled_k = k_raw.copy()
    scaled_k[0, :] *= float(output_width) / float(raw_width)
    scaled_k[1, :] *= float(output_height) / float(raw_height)
    return {
        "intrinsic_matrix": scaled_k,
        "distortion": distortion,
        "source_size": [raw_width, raw_height],
        "output_size": [output_width, output_height],
    }


class CalibrationProvider:
    """按体素坐标系加载 IR 外参，并显式区分正式标定与 fallback。"""

    def __init__(
        self,
        root_dir: str,
        calibration_dir: Optional[str] = None,
        *,
        require_real: bool = False,
        voxel_coordinate_frame: str = "lidar",
    ):
        self.root_dir = root_dir
        self.require_real = bool(require_real)
        self.voxel_coordinate_frame = str(voxel_coordinate_frame).strip().lower()
        if self.voxel_coordinate_frame not in ("lidar", "radar"):
            raise ValueError("voxel_coordinate_frame 必须是 lidar 或 radar")
        if self.require_real and not calibration_dir:
            raise ValueError("正式标定模式必须显式提供 calibration_dir")
        if calibration_dir:
            calibration_dir = os.path.abspath(os.fspath(calibration_dir))
            if os.path.islink(calibration_dir) or not os.path.isdir(calibration_dir):
                raise ValueError(f"calibration_dir 必须是普通目录: {calibration_dir}")
            candidates = [calibration_dir]
        else:
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

    def _try_read_named_calib(self, name: str):
        for config_dir in self.config_dirs:
            path = os.path.join(config_dir, name)
            r_mat, t_vec = _read_calibration_txt(path)
            if r_mat is not None and t_vec is not None:
                _validate_extrinsic(r_mat, t_vec, path=path)
                return r_mat, t_vec, path
        return None, None, ""

    def _try_read_thermal_intrinsics(self):
        for config_dir in self.config_dirs:
            path = os.path.join(config_dir, "calib_cam_thermal.txt")
            parsed = _read_thermal_camera_calibration(path)
            if parsed is not None:
                parsed["path"] = path
                return parsed
        return None

    def load_with_metadata(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
        """返回 IR 投影可用标定和来源信息。

        LiDAR 对齐体素只允许 LiDAR-to-Thermal；Radar 对齐的 legacy 数据才使用
        Radar-to-Thermal。时间补偿不在此处修改外参，统一由预处理负责。
        """
        thermal_intrinsics = self._try_read_thermal_intrinsics()
        if thermal_intrinsics is None:
            k_np = DEFAULT_THERMAL_K.copy()
            thermal_distortion = np.zeros(5, dtype=np.float32)
            thermal_source_size = list(THERMAL_OUTPUT_SIZE)
            thermal_intrinsics_source = "default"
            thermal_intrinsics_path = ""
        else:
            k_np = thermal_intrinsics["intrinsic_matrix"]
            thermal_distortion = thermal_intrinsics["distortion"]
            thermal_source_size = thermal_intrinsics["source_size"]
            thermal_intrinsics_source = "calib_cam_thermal.txt"
            thermal_intrinsics_path = thermal_intrinsics["path"]
        k_mat = torch.from_numpy(k_np.copy()).float()
        extrinsic_name = (
            "calib_livox_to_thermal.txt"
            if self.voxel_coordinate_frame == "lidar"
            else "calib_radar_to_thermal.txt"
        )
        thermal_r, thermal_t, thermal_path = self._try_read_named_calib(extrinsic_name)
        livox_r, livox_t, livox_path = self._try_read_named_calib("calib_radar_to_livox.txt")
        has_livox = livox_r is not None and livox_t is not None
        lidar_thermal_r, lidar_thermal_t, _lidar_thermal_path = (
            self._try_read_named_calib("calib_livox_to_thermal.txt")
        )
        radar_thermal_r, radar_thermal_t, _radar_thermal_path = (
            self._try_read_named_calib("calib_radar_to_thermal.txt")
        )
        closure_available = all(
            value is not None
            for value in (
                livox_r,
                livox_t,
                lidar_thermal_r,
                lidar_thermal_t,
                radar_thermal_r,
                radar_thermal_t,
            )
        )
        if closure_available:
            composed_r = lidar_thermal_r @ livox_r
            composed_t = lidar_thermal_r @ livox_t + lidar_thermal_t
            closure_rotation_max_abs = float(
                torch.max(torch.abs(composed_r - radar_thermal_r))
            )
            closure_translation_l2_m = float(
                torch.linalg.vector_norm(composed_t - radar_thermal_t)
            )
        else:
            closure_rotation_max_abs = -1.0
            closure_translation_l2_m = -1.0
        closure_metadata = {
            "calibration_closure_available": bool(closure_available),
            "calibration_closure_rotation_max_abs": closure_rotation_max_abs,
            "calibration_closure_translation_l2_m": closure_translation_l2_m,
            "calibration_closure_composition": (
                "radar_to_livox_then_livox_to_thermal"
            ),
        }
        if self.require_real and thermal_intrinsics is None:
            raise RuntimeError(
                "正式 IR 标定缺失或格式无效: calib_cam_thermal.txt"
            )
        if thermal_r is not None and thermal_t is not None:
            metadata = {
                "is_mock_calib": False,
                "calib_source": extrinsic_name,
                "calib_path": thermal_path,
                "calib_is_thermal": True,
                "extrinsic_source_frame": self.voxel_coordinate_frame,
                "extrinsic_target_frame": "thermal_camera",
                "has_thermal_calib": True,
                "has_livox_calib": bool(has_livox),
                "livox_calib_path": livox_path,
                "calib_fallback_reason": "",
                "thermal_intrinsic_matrix": k_np.tolist(),
                "thermal_distortion": thermal_distortion.tolist(),
                "thermal_source_size": thermal_source_size,
                "thermal_output_size": list(THERMAL_OUTPUT_SIZE),
                "thermal_intrinsics_source": thermal_intrinsics_source,
                "thermal_intrinsics_path": thermal_intrinsics_path,
                "has_thermal_intrinsics": thermal_intrinsics is not None,
                **closure_metadata,
            }
            return thermal_r, thermal_t, k_mat, metadata

        if self.require_real:
            missing = [extrinsic_name]
            if thermal_intrinsics is None:
                missing.append("calib_cam_thermal.txt")
            raise RuntimeError(
                "正式 IR 标定缺失或格式无效: " + ", ".join(missing)
            )

        fallback_reason = (
            "thermal_missing_livox_available_not_used_for_ir"
            if has_livox
            else "thermal_missing"
        )
        metadata = {
            "is_mock_calib": True,
            "calib_source": "mock_default",
            "calib_path": "",
            "calib_is_thermal": False,
            "extrinsic_source_frame": self.voxel_coordinate_frame,
            "extrinsic_target_frame": "thermal_camera",
            "has_thermal_calib": False,
            "has_livox_calib": bool(has_livox),
            "livox_calib_path": livox_path,
            "calib_fallback_reason": fallback_reason,
            "thermal_intrinsic_matrix": k_np.tolist(),
            "thermal_distortion": thermal_distortion.tolist(),
            "thermal_source_size": thermal_source_size,
            "thermal_output_size": list(THERMAL_OUTPUT_SIZE),
            "thermal_intrinsics_source": thermal_intrinsics_source,
            "thermal_intrinsics_path": thermal_intrinsics_path,
            "has_thermal_intrinsics": thermal_intrinsics is not None,
            **closure_metadata,
        }
        return torch.eye(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32), k_mat, metadata

    def load(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """兼容历史调用：返回 R/T/K/is_mock_calib。"""
        r_mat, t_vec, k_mat, metadata = self.load_with_metadata()
        return r_mat, t_vec, k_mat, bool(metadata["is_mock_calib"])


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


def resize_radar_voxel_channels(
    voxel_tensor: torch.Tensor,
    target_size,
    *,
    statistics: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    """按字段有效计数重采样 Radar 均值与 Doppler 总方差。

    输入通道固定为 ``[occupancy, intensity_mean, doppler_mean,
    doppler_variance]``。statistics-v2 使用 intensity/Doppler 各自的有效
    计数；未提供 statistics 时显式保留 legacy occupied 细体素等权语义。
    """
    if voxel_tensor.ndim != 4 or voxel_tensor.shape[0] != 4:
        raise ValueError(
            "Radar voxel 必须是四通道 (4,Z,X,Y)，"
            f"当前为 {tuple(voxel_tensor.shape)}"
        )
    if not torch.isfinite(voxel_tensor).all():
        raise ValueError("Radar voxel 必须全部为有限数")

    x = voxel_tensor.unsqueeze(0).float()
    occupancy = x[:, 0:1]
    occupied = (occupancy > 0).to(dtype=x.dtype)
    local_variance = x[:, 3:4]
    if torch.any((local_variance < 0) & (occupied > 0)):
        raise ValueError("occupied Radar voxel 的 Doppler variance 不得为负")

    resized_occupancy = F.adaptive_max_pool3d(occupancy, target_size)
    if statistics is not None:
        expected_keys = {
            "protocol",
            "point_count",
            "intensity_valid_count",
            "doppler_valid_count",
        }
        if not isinstance(statistics, dict) or set(statistics) != expected_keys:
            raise ValueError(
                f"Radar resize statistics 字段必须精确为 {sorted(expected_keys)}"
            )
        if statistics.get("protocol") != RADAR_STATISTICS_PROTOCOL:
            raise ValueError(
                "Radar count-weighted resize 只接受 "
                f"{RADAR_STATISTICS_PROTOCOL!r}"
            )

        resolved_counts = {}
        spatial_shape = tuple(voxel_tensor.shape[-3:])
        for name in (
            "point_count",
            "intensity_valid_count",
            "doppler_valid_count",
        ):
            count = torch.as_tensor(
                statistics[name],
                device=voxel_tensor.device,
            )
            if tuple(count.shape) != spatial_shape:
                raise ValueError(
                    f"{name} shape 必须为 {spatial_shape}，实际为 {tuple(count.shape)}"
                )
            count = count.to(dtype=torch.float32)
            if (
                not torch.isfinite(count).all()
                or torch.any(count < 0)
                or not torch.equal(count, count.round())
            ):
                raise ValueError(f"{name} 必须是非负有限整数计数")
            resolved_counts[name] = count.unsqueeze(0).unsqueeze(0)

        point_count = resolved_counts["point_count"]
        intensity_count = resolved_counts["intensity_valid_count"]
        doppler_count = resolved_counts["doppler_valid_count"]
        if torch.any(intensity_count > point_count):
            raise ValueError("intensity_valid_count 不得超过 point_count")
        if torch.any(doppler_count > point_count):
            raise ValueError("doppler_valid_count 不得超过 point_count")
        if not torch.equal(point_count > 0, occupied.bool()):
            raise ValueError("point_count support 与 Radar occupancy 不一致")

        def count_weighted_resize(
            channel: torch.Tensor,
            count: torch.Tensor,
        ) -> torch.Tensor:
            pooled_count = F.adaptive_avg_pool3d(count, target_size)
            pooled_sum = F.adaptive_avg_pool3d(
                channel * count,
                target_size,
            )
            merged = pooled_sum / pooled_count.clamp_min(EPS)
            return torch.where(
                pooled_count > EPS,
                merged,
                torch.zeros_like(merged),
            )

        resized_intensity = count_weighted_resize(x[:, 1:2], intensity_count)
        resized_doppler = count_weighted_resize(x[:, 2:3], doppler_count)
        doppler_second_moment = local_variance + x[:, 2:3].square()
        resized_second_moment = count_weighted_resize(
            doppler_second_moment,
            doppler_count,
        )
        resized_variance = (
            resized_second_moment - resized_doppler.square()
        ).clamp_min(0.0)
        coarse_doppler_count = F.adaptive_avg_pool3d(
            doppler_count,
            target_size,
        )
        resized_variance = torch.where(
            coarse_doppler_count > EPS,
            resized_variance,
            torch.zeros_like(resized_variance),
        )
        return torch.cat(
            [
                resized_occupancy,
                resized_intensity,
                resized_doppler,
                resized_variance,
            ],
            dim=1,
        ).squeeze(0)

    # NOTE: occupancy 与物理属性必须使用同一 adaptive 分箱。若 occupancy 用
    # max-pool、属性却用 trilinear 中心采样，稀疏点可能令 coarse voxel 已占用但
    # intensity/Doppler/variance 被错误清零。
    occupied_density = F.adaptive_avg_pool3d(
        occupied,
        target_size,
    )
    has_occupied_input = occupied_density > EPS

    def occupied_weighted_resize(channel: torch.Tensor) -> torch.Tensor:
        weighted = F.adaptive_avg_pool3d(
            channel * occupied,
            target_size,
        )
        merged = weighted / occupied_density.clamp_min(EPS)
        return torch.where(has_occupied_input, merged, torch.zeros_like(merged))

    resized_intensity = occupied_weighted_resize(x[:, 1:2])
    resized_doppler = occupied_weighted_resize(x[:, 2:3])
    doppler_second_moment = local_variance + x[:, 2:3].square()
    resized_second_moment = occupied_weighted_resize(doppler_second_moment)
    resized_variance = (
        resized_second_moment - resized_doppler.square()
    ).clamp_min(0.0)
    resized_variance = torch.where(
        has_occupied_input,
        resized_variance,
        torch.zeros_like(resized_variance),
    )

    return torch.cat(
        [
            resized_occupancy,
            resized_intensity,
            resized_doppler,
            resized_variance,
        ],
        dim=1,
    ).squeeze(0)


def crop_voxel_channels_to_pc_range(
    voxel_tensor: torch.Tensor,
    source_pc_range,
    model_pc_range,
) -> torch.Tensor:
    """Crop a `(C,Z,X,Y)` tensor using physical XYZ bounds."""
    if voxel_tensor.ndim != 4:
        raise ValueError(f"Expected (C,Z,X,Y), got {tuple(voxel_tensor.shape)}")
    source = tuple(float(v) for v in source_pc_range)
    model = tuple(float(v) for v in model_pc_range)
    if len(source) != 6 or len(model) != 6:
        raise ValueError("source_pc_range and model_pc_range must contain 6 values")
    for axis in range(3):
        if model[axis] < source[axis] or model[axis + 3] > source[axis + 3]:
            raise ValueError(f"model_pc_range must lie inside source_pc_range: {model} vs {source}")
        if model[axis] >= model[axis + 3]:
            raise ValueError(f"Invalid model_pc_range: {model}")

    def physical_slice(size: int, low: float, high: float, crop_low: float, crop_high: float):
        step = (high - low) / float(size)
        centers = low + (torch.arange(size, device=voxel_tensor.device) + 0.5) * step
        indices = torch.where((centers >= crop_low) & (centers < crop_high))[0]
        if indices.numel() == 0:
            raise ValueError(f"Physical crop [{crop_low}, {crop_high}) contains no voxel centers")
        return slice(int(indices[0]), int(indices[-1]) + 1)

    z_slice = physical_slice(voxel_tensor.shape[1], source[2], source[5], model[2], model[5])
    x_slice = physical_slice(voxel_tensor.shape[2], source[0], source[3], model[0], model[3])
    y_slice = physical_slice(voxel_tensor.shape[3], source[1], source[4], model[1], model[4])
    return voxel_tensor[:, z_slice, x_slice, y_slice]


def _prepare_ir_array(ir_img, calibration_metadata: Optional[dict] = None) -> np.ndarray:
    """按共享 thermal 标定把图像调整到输出尺寸并执行去畸变。"""
    array = ir_img.detach().cpu().numpy() if torch.is_tensor(ir_img) else np.asarray(ir_img)
    if array.ndim == 2:
        array = array[..., np.newaxis]
    elif array.ndim == 3:
        if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        elif array.shape[-1] not in (1, 3):
            raise ValueError(f"IR 数组通道维度无法识别: {array.shape}")
    else:
        raise ValueError(f"IR 数组必须是 2D 或 3D，当前为 {array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    array = array[..., :3].astype(np.float32, copy=False)

    metadata = calibration_metadata or {}
    output_width, output_height = (int(value) for value in metadata.get(
        "thermal_output_size", THERMAL_OUTPUT_SIZE
    ))
    if output_width <= 0 or output_height <= 0:
        raise ValueError("thermal_output_size 必须是正整数宽高")

    channels = []
    for channel in range(array.shape[-1]):
        image = array[..., channel]
        if image.shape[1] != output_width or image.shape[0] != output_height:
            if cv2 is not None:
                image = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
            else:
                image = (
                    torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
                    .float()
                )
                image = F.interpolate(
                    image,
                    size=(output_height, output_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().numpy()
        channels.append(image.astype(np.float32, copy=False))

    distortion = np.asarray(metadata.get("thermal_distortion", np.zeros(5)), dtype=np.float32)
    intrinsic = np.asarray(
        metadata.get("thermal_intrinsic_matrix", DEFAULT_THERMAL_K),
        dtype=np.float32,
    )
    if distortion.size and not np.allclose(distortion, 0.0):
        if cv2 is None:
            raise RuntimeError("thermal 标定包含畸变系数，但当前环境缺少 OpenCV")
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("thermal_intrinsic_matrix 必须是有限的 3x3 矩阵")
        channels = [
            cv2.undistort(channel, intrinsic, distortion)
            for channel in channels
        ]
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def _resize_or_pad_ir_tensor(
    ir_img: torch.Tensor,
    calibration_metadata: Optional[dict] = None,
) -> torch.Tensor:
    if calibration_metadata is not None:
        prepared = _prepare_ir_array(ir_img, calibration_metadata)
        result = torch.from_numpy(prepared)
        if torch.is_tensor(ir_img) and ir_img.device.type != "cpu":
            result = result.to(ir_img.device)
        return result
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


def collate_voxel_samples(batch):
    """拼接模型输入，并原样保留含 JSON null 的逐样本预处理审计信息。"""
    if not batch:
        raise ValueError("voxel batch 不能为空")
    sample_size = len(batch[0])
    if sample_size not in (3, 4) or any(len(sample) != sample_size for sample in batch):
        raise ValueError("voxel batch 样本必须是等长的三元组或四元组")

    targets, radar_inputs, metadata_items = zip(
        *((sample[0], sample[1], sample[2]) for sample in batch)
    )
    if not all(isinstance(metadata, dict) for metadata in metadata_items):
        raise TypeError("voxel batch metadata 必须全部为字典")
    metadata_keys = tuple(metadata_items[0].keys())
    expected_keys = set(metadata_keys)
    if any(set(metadata.keys()) != expected_keys for metadata in metadata_items[1:]):
        raise ValueError("voxel batch metadata 字段不一致")

    collated_metadata = {}
    for key in metadata_keys:
        values = [metadata[key] for metadata in metadata_items]
        if key in ("preprocess_policy", "radar_statistics"):
            # 审计 provenance 允许 JSON null/嵌套列表，不参与模型张量计算。
            collated_metadata[key] = values
        else:
            collated_metadata[key] = default_collate(values)

    collated = [
        default_collate(list(targets)),
        default_collate(list(radar_inputs)),
        collated_metadata,
    ]
    if sample_size == 4:
        collated.append(default_collate([sample[3] for sample in batch]))
    return tuple(collated)


class NTU4DRadLM_VoxelDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, return_path=False, alignment_size=32,
                 use_augmentation=True, augmentation_config=None, sequence_length=1,
                 target_size=DEFAULT_TARGET_SIZE, source_pc_range=DEFAULT_PC_RANGE,
                 model_pc_range=None, radar_normalization=None,
                 radar_normalization_sha256=None, allow_legacy_radar_units=False,
                 scene_names=None, calibration_dir=None,
                 require_real_ir=False, require_real_calibration=False,
                 require_persisted_observed_mask=False,
                 require_radar_statistics=False,
                 frame_ids_by_scene=None,
                 voxel_coordinate_frame="lidar"):
        """加载单帧 Radar/IR 条件及 LiDAR 离线监督合同。"""
        self.root_dir = root_dir
        self.return_path = return_path
        if transform is not None:
            raise ValueError("Dataset transform 参数未实现；请使用显式 augmentation")
        if alignment_size != 32:
            raise ValueError("Dataset alignment_size 参数未实现且只能保留默认值 32")
        if isinstance(sequence_length, bool) or int(sequence_length) != 1:
            raise ValueError("当前模型没有时序融合，sequence_length 必须严格为 1")
        # 样本包含显式 observed mask 路径，避免正式训练运行时重建监督域。
        self.samples = []  # (radar, target, ir, scene, lidar, observed_mask)
        self.scene_policies: Dict[str, dict] = {}
        self.split = split
        self.require_real_ir = bool(require_real_ir)
        self.require_real_calibration = bool(require_real_calibration)
        self.require_persisted_observed_mask = bool(
            require_persisted_observed_mask
        )
        if type(require_radar_statistics) is not bool:
            raise ValueError("require_radar_statistics 必须是 bool")
        self.require_radar_statistics = require_radar_statistics
        self.radar_statistics_by_path: Dict[str, Dict[str, object]] = {}
        self.frame_ids_by_scene = None
        if frame_ids_by_scene is not None:
            if not isinstance(frame_ids_by_scene, dict) or not frame_ids_by_scene:
                raise ValueError("frame_ids_by_scene 必须是非空场景映射")
            validated_frame_ids = {}
            for frame_scene, frame_ids in frame_ids_by_scene.items():
                if (
                    not isinstance(frame_scene, str)
                    or not frame_scene
                    or os.path.basename(frame_scene) != frame_scene
                    or not isinstance(frame_ids, (list, tuple))
                    or not frame_ids
                    or any(
                        not isinstance(frame_id, str)
                        or len(frame_id) != 6
                        or not frame_id.isdigit()
                        for frame_id in frame_ids
                    )
                    or len(set(frame_ids)) != len(frame_ids)
                ):
                    raise ValueError("frame_ids_by_scene 场景或 frame ID 无效")
                validated_frame_ids[frame_scene] = tuple(frame_ids)
            self.frame_ids_by_scene = validated_frame_ids
        self.voxel_coordinate_frame = str(voxel_coordinate_frame).strip().lower()
        self.target_size = tuple(int(v) for v in target_size)
        self.source_pc_range = tuple(float(v) for v in source_pc_range)
        self.model_pc_range = tuple(
            float(v) for v in (model_pc_range if model_pc_range is not None else source_pc_range)
        )
        if type(allow_legacy_radar_units) is not bool:
            raise RadarNormalizationError("allow_legacy_radar_units 必须是 bool")
        if radar_normalization is None:
            if not allow_legacy_radar_units:
                raise RadarNormalizationError(
                    "Dataset 缺少 Radar normalization；"
                    "旧测试/诊断必须显式设置 allow_legacy_radar_units=True"
                )
            if radar_normalization_sha256 not in (None, ""):
                raise RadarNormalizationError(
                    "legacy Radar 单位不得携带 normalization SHA-256"
                )
            self.radar_normalization = None
            self.radar_normalization_sha256 = ""
            self.radar_normalization_protocol = LEGACY_RADAR_NORMALIZATION_PROTOCOL
            self.legacy_radar_units = True
        else:
            if allow_legacy_radar_units:
                raise RadarNormalizationError(
                    "Radar normalization 与 legacy 开关不能同时启用"
                )
            self.radar_normalization = validate_radar_normalization_spec(
                radar_normalization,
                target_size=self.target_size,
                source_pc_range=self.source_pc_range,
                model_pc_range=self.model_pc_range,
                doppler_scale_mps=radar_normalization.get("doppler", {}).get("scale_mps")
                if isinstance(radar_normalization, dict)
                else None,
                require_formal=True,
            )
            self.radar_normalization_sha256 = validate_radar_normalization_sha256(
                radar_normalization_sha256,
                context="Dataset Radar normalization",
            )
            self.radar_normalization_protocol = self.radar_normalization["protocol"]
            self.legacy_radar_units = False
        self.calibration_provider = CalibrationProvider(
            root_dir,
            calibration_dir=calibration_dir,
            require_real=self.require_real_calibration,
            voxel_coordinate_frame=self.voxel_coordinate_frame,
        )

        # mock 外参只服务显式 legacy 诊断；formal 路径由 CalibrationProvider fail-closed。
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
        discovered_scenes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d, "radar_voxel"))
            and os.path.isdir(os.path.join(root_dir, d, "target_voxel"))
        ])
        if scene_names is not None:
            if isinstance(scene_names, str):
                scene_names = [scene_names]
            target_scenes = [str(scene).strip() for scene in scene_names]
            if (
                not target_scenes
                or len(set(target_scenes)) != len(target_scenes)
                or any(not scene or os.path.basename(scene) != scene for scene in target_scenes)
            ):
                raise ValueError("scene_names 必须是非空、无重复的普通场景名列表")
            missing_scenes = [
                scene for scene in target_scenes if scene not in discovered_scenes
            ]
            if missing_scenes:
                raise FileNotFoundError(f"显式场景不存在或模态不完整: {missing_scenes}")
        elif self.require_real_ir or self.require_real_calibration:
            raise ValueError("正式多模态 Dataset 必须显式提供 scene_names")
        elif len(discovered_scenes) == 1:
            target_scenes = discovered_scenes
            print(f"Warning: Only 1 scene found. Using it for {split}.")
        else:
            split_idx = int(len(discovered_scenes) * 0.8)
            if split_idx == 0: split_idx = 1
            target_scenes = (
                discovered_scenes[:split_idx]
                if split == 'train'
                else discovered_scenes[split_idx:]
            )

        print(f"Loading {split} dataset from {len(target_scenes)} scenes: {target_scenes}")

        if self.frame_ids_by_scene is not None and set(self.frame_ids_by_scene) != set(
            target_scenes
        ):
            raise ValueError(
                "frame_ids_by_scene 场景集合必须与 target scenes 精确一致"
            )
        collected_frame_ids = {scene: set() for scene in target_scenes}

        for scene in target_scenes:
            radar_voxel_dir = os.path.join(root_dir, scene, "radar_voxel")
            target_voxel_dir = os.path.join(root_dir, scene, "target_voxel")
            observed_mask_dir = os.path.join(root_dir, scene, "observed_mask")
            ir_dir = os.path.join(root_dir, scene, "ir_image")
            if self.require_persisted_observed_mask and not os.path.isdir(
                observed_mask_dir
            ):
                raise FileNotFoundError(
                    f"正式 Dataset 缺少 observed mask 目录: {observed_mask_dir}"
                )
            policy_path = os.path.join(root_dir, scene, "preprocess_policy.json")
            if os.path.exists(policy_path):
                with open(policy_path, "r", encoding="utf-8") as f:
                    self.scene_policies[scene] = json.load(f)
            else:
                self.scene_policies[scene] = {}
            policy_frame = self.scene_policies[scene].get("voxel_coordinate_frame")
            if policy_frame is None:
                policy_frame = self.scene_policies[scene].get("align_to")
            if policy_frame is not None and policy_frame != self.voxel_coordinate_frame:
                raise ValueError(
                    f"场景 {scene} voxel frame={policy_frame!r} 与 Dataset "
                    f"{self.voxel_coordinate_frame!r} 不一致"
                )
            policy_statistics_protocol = self.scene_policies[scene].get(
                "radar_statistics_protocol"
            )
            if (
                self.require_radar_statistics
                and policy_statistics_protocol not in SUPPORTED_RADAR_STATISTICS_PROTOCOLS
            ):
                raise ValueError(
                    f"场景 {scene} Radar statistics policy 缺失或协议不匹配"
                )

            if not os.path.exists(radar_voxel_dir) or not os.path.exists(target_voxel_dir):
                continue

            files = sorted([f for f in os.listdir(radar_voxel_dir) if f.endswith('.npy') or f.endswith('.npz')])

            for target_f in files:
                # 当前模型是严格单帧条件；文件名相同即为当前监督时刻。
                radar_path = os.path.join(radar_voxel_dir, target_f)
                target_frame_id = os.path.splitext(target_f)[0]
                if (
                    self.frame_ids_by_scene is not None
                    and target_frame_id not in self.frame_ids_by_scene[scene]
                ):
                    continue
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

                lidar_path = ""
                lidar_dir = os.path.join(root_dir, scene, "lidar_voxel")
                if os.path.isdir(lidar_dir):
                    lidar_path = os.path.join(lidar_dir, target_f)
                    if not os.path.exists(lidar_path):
                        if target_f.endswith('.npy'):
                            lidar_path = os.path.join(
                                lidar_dir, target_f.replace('.npy', '.npz')
                            )
                        elif target_f.endswith('.npz'):
                            lidar_path = os.path.join(
                                lidar_dir, target_f.replace('.npz', '.npy')
                            )
                    if not os.path.exists(lidar_path):
                        lidar_path = ""

                observed_mask_path = os.path.join(
                    observed_mask_dir,
                    f"{target_frame_id}.npz",
                )
                if not os.path.exists(observed_mask_path):
                    observed_mask_path = ""
                if self.require_persisted_observed_mask and not observed_mask_path:
                    raise FileNotFoundError(
                        f"正式 Dataset 缺少 observed mask: scene={scene}, "
                        f"frame={target_f}"
                    )

                if os.path.exists(target_path):
                    if self.require_radar_statistics:
                        if not radar_path.endswith(".npz"):
                            raise ValueError(
                                "正式 Radar statistics 只支持稀疏 NPZ: "
                                f"{radar_path}"
                            )
                        if radar_path not in self.radar_statistics_by_path:
                            summary = validate_sparse_radar_statistics(radar_path)
                            self.radar_statistics_by_path[radar_path] = {
                                **summary,
                                "frame_id": target_frame_id,
                                "reference": (
                                    "pre_augmentation_persisted_radar_voxel"
                                ),
                                "model_consumed": (
                                    summary["protocol"]
                                    == RADAR_STATISTICS_PROTOCOL
                                ),
                            }
                    self.samples.append(
                        (
                            radar_path,
                            target_path,
                            ir_path,
                            scene,
                            lidar_path,
                            observed_mask_path,
                        )
                    )
                    collected_frame_ids[scene].add(target_frame_id)

        if self.frame_ids_by_scene is not None:
            for scene, expected_ids in self.frame_ids_by_scene.items():
                missing = sorted(set(expected_ids) - collected_frame_ids[scene])
                extra = sorted(collected_frame_ids[scene] - set(expected_ids))
                if missing or extra:
                    raise FileNotFoundError(
                        f"Dataset 未精确收集 split frame: scene={scene}, "
                        f"missing={missing}, extra={extra}"
                    )

        print(f"Found {len(self.samples)} single-frame samples for {split}.")

    def __len__(self):
        return len(self.samples)

    def _get_calibration(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
        """读取与体素 frame 一致的标定；不在加载阶段伪造时间位移。"""
        r_mat, t_vec, k_mat, calib_meta = self.calibration_provider.load_with_metadata()
        if calib_meta["is_mock_calib"]:
            r_mat = self.R_cam_to_lidar.clone()
            t_vec = self.T_cam_to_lidar.clone()
        calib_meta = dict(calib_meta)
        calib_meta["time_alignment_compensation"] = "preprocessing_signed_delta_only"
        return r_mat, t_vec, k_mat, calib_meta

    def _load_ir_tensor(
        self,
        ir_path: str,
        calibration_metadata: Optional[dict] = None,
    ) -> torch.Tensor:
        if os.path.exists(ir_path):
            arr = np.load(ir_path).astype(np.float32)
            return _resize_or_pad_ir_tensor(
                torch.from_numpy(arr), calibration_metadata
            ), False
        if self.require_real_ir:
            raise FileNotFoundError(f"正式多模态样本缺少 IR 文件: {ir_path}")
        return _mock_ir_image(), True

    def __getitem__(self, idx):
        (
            radar_path,
            target_path,
            ir_path,
            scene,
            lidar_path,
            observed_mask_path,
        ) = self.samples[idx]

        # 无缝复原原本完全体的张量空间轴向变换流与重采样
        # 1. 加载并转换主监督真值体素
        if target_path.endswith('.npz'):
            target_voxel = load_sparse_voxel(target_path)
        else:
            target_voxel = np.load(target_path).astype(np.float32)

        if observed_mask_path:
            observed_mask_np = load_observed_mask(
                observed_mask_path,
                expected_shape=target_voxel.shape[:3],
                expected_pc_range=self.source_pc_range,
            )
            observed_mask_source = PERSISTED_OBSERVED_MASK_SOURCE
        elif lidar_path:
            if lidar_path.endswith('.npz'):
                lidar_voxel = load_sparse_voxel(lidar_path)
            else:
                lidar_voxel = np.load(lidar_path).astype(np.float32)
            if tuple(lidar_voxel.shape[:3]) != tuple(target_voxel.shape[:3]):
                raise ValueError(
                    "LiDAR/target voxel shape 不匹配: "
                    f"lidar={lidar_voxel.shape}, target={target_voxel.shape}"
                )
            observed_mask_np = build_lidar_observed_mask(
                lidar_voxel,
                self.source_pc_range,
            )
            observed_mask_source = "lidar_ray"
        else:
            # 兼容缺少独立 LiDAR voxel 的旧小样本；只保留 occupied 监督，
            # 不把 target 空白自动解释为可见 free。
            observed_mask_np = target_voxel[..., 0] > 0.5
            observed_mask_source = "occupied_only_fallback"

        # 完美对齐物理原版轴向重塑: (H, W, Z, C) -> (C, Z, H, W)
        target_tensor = torch.from_numpy(target_voxel).permute(3, 2, 0, 1)
        target_tensor = crop_voxel_channels_to_pc_range(
            target_tensor, self.source_pc_range, self.model_pc_range
        )
        target_tensor = resize_voxel_channels(target_tensor, self.target_size, mask_channel=3)

        observed_mask_tensor = torch.from_numpy(
            observed_mask_np.astype(np.float32)
        ).permute(2, 0, 1).unsqueeze(0)
        observed_mask_tensor = crop_voxel_channels_to_pc_range(
            observed_mask_tensor,
            self.source_pc_range,
            self.model_pc_range,
        )
        observed_mask_tensor = resize_voxel_channels(
            observed_mask_tensor,
            self.target_size,
            mask_channel=0,
        )
        observed_mask_tensor = (observed_mask_tensor > 0.5).float()

        # 2. 加载与监督时刻同名的单帧 Radar 条件。
        radar_statistics = []
        radar_resize_statistics = None
        declared_statistics_protocol = self.scene_policies.get(scene, {}).get(
            "radar_statistics_protocol"
        )
        statistics_declared = (
            declared_statistics_protocol in SUPPORTED_RADAR_STATISTICS_PROTOCOLS
        )
        if radar_path.endswith('.npz'):
            if self.require_radar_statistics or statistics_declared:
                radar_voxel, statistics_fields, summary = (
                    load_sparse_radar_voxel_with_statistics(radar_path)
                )
                statistics_consumed = (
                    summary["protocol"] == RADAR_STATISTICS_PROTOCOL
                )
                if statistics_consumed:
                    radar_resize_statistics = statistics_fields
                radar_statistics.append(
                    {
                        **summary,
                        "frame_id": os.path.splitext(os.path.basename(radar_path))[0],
                        # 摘要描述磁盘中的原始稀疏体素，不随数据增强变换。
                        "reference": "pre_augmentation_persisted_radar_voxel",
                        # 计数只控制 resize 聚合；模型接口仍是四通道。
                        "model_consumed": statistics_consumed,
                        "resize_aggregation": (
                            RADAR_RESIZE_AGGREGATION
                            if statistics_consumed
                            else RADAR_RESIZE_AGGREGATION_V1
                        ),
                    }
                )
                if summary["protocol"] != declared_statistics_protocol:
                    raise ValueError(
                        f"Radar statistics payload={summary['protocol']!r} 与 "
                        f"policy={declared_statistics_protocol!r} 不一致: {radar_path}"
                    )
            else:
                radar_voxel = load_sparse_voxel(radar_path)
        else:
            if self.require_radar_statistics or statistics_declared:
                raise ValueError(
                    f"Radar statistics policy 不允许稠密 NPY: {radar_path}"
                )
            radar_voxel = np.load(radar_path).astype(np.float32)

        radar_tensor = torch.from_numpy(radar_voxel).permute(3, 2, 0, 1)
        radar_tensor = crop_voxel_channels_to_pc_range(
            radar_tensor, self.source_pc_range, self.model_pc_range
        )
        if radar_resize_statistics is not None:
            cropped_statistics = {"protocol": radar_resize_statistics["protocol"]}
            for name in (
                "point_count",
                "intensity_valid_count",
                "doppler_valid_count",
            ):
                count_tensor = torch.from_numpy(
                    np.asarray(radar_resize_statistics[name])
                ).permute(2, 0, 1).unsqueeze(0)
                count_tensor = crop_voxel_channels_to_pc_range(
                    count_tensor,
                    self.source_pc_range,
                    self.model_pc_range,
                )
                cropped_statistics[name] = count_tensor.squeeze(0)
            radar_resize_statistics = cropped_statistics
        radar_tensor = resize_radar_voxel_channels(
            radar_tensor,
            self.target_size,
            statistics=radar_resize_statistics,
        )

        # 3. 加载共享 thermal 标定，并按同一 K/D/S 协议准备红外图像
        r_mat, t_vec, k_mat, calib_meta = self._get_calibration()
        ir_img, is_mock_ir = self._load_ir_tensor(ir_path, calib_meta)

        # 完美保留：多模态双成对几何空间一致性增强
        if self.augmentation is not None:
            target_tensor, radar_tensor, observed_mask_tensor = self.augmentation(
                target_tensor,
                radar_tensor,
                observed_mask_tensor,
            )

        if self.radar_normalization is not None:
            expected_resize_aggregation = (
                RADAR_RESIZE_AGGREGATION
                if radar_resize_statistics is not None
                else RADAR_RESIZE_AGGREGATION_V1
            )
            artifact_resize_aggregation = self.radar_normalization[
                "variance"
            ]["aggregation"]
            if artifact_resize_aggregation != expected_resize_aggregation:
                raise RadarNormalizationError(
                    "Radar normalization resize aggregation 与样本统计协议不一致: "
                    f"artifact={artifact_resize_aggregation!r}, "
                    f"sample={expected_resize_aggregation!r}"
                )
            radar_tensor = apply_radar_normalization(
                radar_tensor,
                self.radar_normalization,
            )

        # 样本身份只绑定场景和帧号，不绑定可随机器变化的绝对数据根。
        sample_id = f"{scene}/{os.path.splitext(os.path.basename(target_path))[0]}"
        meta_dict = {
            "sample_id": sample_id,
            "ir_img": ir_img,
            "occupancy_observed_mask": observed_mask_tensor,
            "occupancy_observed_mask_source": observed_mask_source,
            "r_mat": r_mat,
            "t_vec": t_vec,
            "k_mat": k_mat,
            "is_mock_ir": bool(is_mock_ir),
            "is_mock_calib": bool(calib_meta["is_mock_calib"]),
            "calib_source": calib_meta["calib_source"],
            "calib_path": calib_meta["calib_path"],
            "calib_is_thermal": bool(calib_meta["calib_is_thermal"]),
            "has_thermal_calib": bool(calib_meta["has_thermal_calib"]),
            "has_livox_calib": bool(calib_meta["has_livox_calib"]),
            "livox_calib_path": calib_meta["livox_calib_path"],
            "calib_fallback_reason": calib_meta["calib_fallback_reason"],
            "thermal_intrinsics_source": calib_meta["thermal_intrinsics_source"],
            "thermal_intrinsics_path": calib_meta["thermal_intrinsics_path"],
            "thermal_source_size": calib_meta["thermal_source_size"],
            "thermal_output_size": calib_meta["thermal_output_size"],
            "thermal_distortion": calib_meta["thermal_distortion"],
            "has_thermal_intrinsics": bool(calib_meta["has_thermal_intrinsics"]),
            "time_alignment_compensation": calib_meta["time_alignment_compensation"],
            "extrinsic_source_frame": calib_meta["extrinsic_source_frame"],
            "extrinsic_target_frame": calib_meta["extrinsic_target_frame"],
            "calibration_closure_available": bool(
                calib_meta["calibration_closure_available"]
            ),
            "calibration_closure_rotation_max_abs": float(
                calib_meta["calibration_closure_rotation_max_abs"]
            ),
            "calibration_closure_translation_l2_m": float(
                calib_meta["calibration_closure_translation_l2_m"]
            ),
            "calibration_closure_composition": calib_meta[
                "calibration_closure_composition"
            ],
            "voxel_coordinate_frame": self.voxel_coordinate_frame,
            "preprocess_policy": self.scene_policies.get(scene, {}),
            "radar_statistics": radar_statistics if radar_statistics else None,
            "model_pc_range": list(self.model_pc_range),
            "target_size": list(self.target_size),
            "radar_normalization_protocol": self.radar_normalization_protocol,
            "radar_normalization_sha256": self.radar_normalization_sha256,
            "legacy_radar_units": self.legacy_radar_units,
        }

        if self.return_path:
            return target_tensor, radar_tensor, meta_dict, target_path
        return target_tensor, radar_tensor, meta_dict


if __name__ == "__main__":
    dataset_path = "./Data/NTU4DRadLM_Pre_sensor_aware"
    ds = NTU4DRadLM_VoxelDataset(
        dataset_path,
        split='train',
        return_path=True,
        allow_legacy_radar_units=True,
    )
    if len(ds) > 0:
        sample = ds[0]
        t, r, m, p = sample
        print(f"成功加载单帧样本 0。")
        print(f"目标真值 (GT) 形状 [C, Z, H, W]: {t.shape}")
        print(f"单帧雷达条件形状 [C, Z, H, W]: {r.shape}")
        print(f"红外相片矩阵 形状 [C, H_img, W_img]: {m['ir_img'].shape}")
    else:
        print("错误: 数据集为空。请检查 dataset_path。")
