# -*- coding: utf-8 -*-
"""
Probabilistic occupancy mapping primitives.

Includes:
- D-S evidence fusion for occupancy/free/unknown masses
- sliding-window probabilistic map with time decay
- lightweight local lazy proximity query (NanoMap-style)
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np

try:
    from diffusion_consistency_radar.prediction_artifact_protocol import (
        GENERATED_OCCUPANCY_EVIDENCE_SEMANTICS,
    )
except ImportError:  # 兼容 streaming 脚本把包目录加入 sys.path 的直接执行方式。
    from prediction_artifact_protocol import (  # type: ignore
        GENERATED_OCCUPANCY_EVIDENCE_SEMANTICS,
    )

EPS = 1e-6
LEGACY_MULTICHANNEL_EVIDENCE_SEMANTICS = "legacy_multichannel_voxel"
TRAJECTORY_CORRIDOR_QUERY_PROTOCOL = "three_state_trajectory_corridor_v1"
MAX_TRAJECTORY_QUERY_SAMPLES = 100000


def compute_safety_distance_m(
    speed_m_s: float,
    reaction_time_s: float,
    brake_deceleration_m_s2: float,
    margin_m: float,
) -> float:
    """按反应距离、制动距离和余量计算最小安全查询距离。"""
    values = np.asarray(
        [speed_m_s, reaction_time_s, brake_deceleration_m_s2, margin_m],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("安全距离参数必须全部为有限数")
    speed, reaction_time, brake_deceleration, margin = values.tolist()
    if speed < 0.0 or reaction_time < 0.0 or brake_deceleration <= 0.0 or margin < 0.0:
        raise ValueError("速度/反应时间/余量必须非负，制动减速度必须为正")
    return float(
        speed * reaction_time
        + speed * speed / (2.0 * brake_deceleration)
        + margin
    )


@dataclass
class GridMapConfig:
    """2D grid map configuration."""

    x_min: float = 0.0
    y_min: float = -20.0
    x_max: float = 120.0
    y_max: float = 20.0
    x_resolution: float = 0.1
    y_resolution: float = 0.1
    z_min: float = -6.0
    z_max: float = 10.0
    z_resolution: float = 0.2
    window_size: int = 12
    decay_rate: float = 0.12
    dynamic_decay_rate: float = 0.60
    prior_reliability: float = 0.90
    radar_reliability: float = 0.75
    infrared_reliability: float = 0.65
    speed_m_s: float = 50.0
    rolling_enabled: bool = False
    # 模型 evidence 可小于地图范围；范围外地图单元保持初始 unknown。
    evidence_pc_range: Optional[Tuple[float, float, float, float, float, float]] = None

    def __post_init__(self) -> None:
        static_decay = float(self.decay_rate)
        dynamic_decay = float(self.dynamic_decay_rate)
        if not np.isfinite(static_decay) or static_decay < 0.0:
            raise ValueError("decay_rate 必须是有限非负数")
        if not np.isfinite(dynamic_decay) or dynamic_decay < 0.0:
            raise ValueError("dynamic_decay_rate 必须是有限非负数")
        if not np.isfinite(float(self.speed_m_s)) or float(self.speed_m_s) <= 0.0:
            raise ValueError("speed_m_s 必须是有限正数")
        map_range = np.asarray(
            [self.x_min, self.y_min, self.z_min, self.x_max, self.y_max, self.z_max],
            dtype=np.float64,
        )
        resolutions = np.asarray(
            [self.x_resolution, self.y_resolution, self.z_resolution],
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(map_range))
            or not np.all(map_range[3:] > map_range[:3])
            or not np.all(np.isfinite(resolutions))
            or not np.all(resolutions > 0.0)
        ):
            raise ValueError("地图范围与分辨率必须是有效有限数")
        if type(self.rolling_enabled) is not bool:
            raise ValueError("rolling_enabled 必须是 bool")
        if self.evidence_pc_range is None:
            evidence_range = map_range
        else:
            evidence_range = np.asarray(self.evidence_pc_range, dtype=np.float64)
            if evidence_range.shape != (6,) or not np.all(np.isfinite(evidence_range)):
                raise ValueError("evidence_pc_range 必须包含 6 个有限数")
            if not np.all(evidence_range[3:] > evidence_range[:3]):
                raise ValueError("evidence_pc_range 上下界无效")
            if (
                np.any(evidence_range[:3] < map_range[:3] - 1e-9)
                or np.any(evidence_range[3:] > map_range[3:] + 1e-9)
            ):
                raise ValueError("evidence_pc_range 必须完全位于地图范围内")
        self.evidence_pc_range = tuple(float(value) for value in evidence_range)
        # Faster flight leaves fewer frames inside the same local volume, so the
        # map should forget stale observations sooner and keep a shorter window.
        speed_scale = float(np.clip(self.speed_m_s / 50.0, 0.5, 2.0))
        self.window_size = max(4, int(round(float(self.window_size) / speed_scale)))
        self.decay_rate = static_decay * speed_scale
        self.dynamic_decay_rate = dynamic_decay * speed_scale

    @property
    def shape_xy(self) -> Tuple[int, int]:
        width = int(round((self.x_max - self.x_min) / self.x_resolution))
        height = int(round((self.y_max - self.y_min) / self.y_resolution))
        return width, height

    @property
    def shape_xyz(self) -> Tuple[int, int, int]:
        """返回局部地图的三维分层尺寸。"""
        nx, ny = self.shape_xy
        nz = int(round((self.z_max - self.z_min) / self.z_resolution))
        return nx, ny, nz

    @property
    def evidence_shape_xyz(self) -> Tuple[int, int, int]:
        """返回模型 evidence 按地图分辨率对应的源体素尺寸。"""
        values = []
        resolutions = (self.x_resolution, self.y_resolution, self.z_resolution)
        for axis, resolution in enumerate(resolutions):
            span = self.evidence_pc_range[axis + 3] - self.evidence_pc_range[axis]
            cells = int(round(span / resolution))
            if cells <= 0 or not np.isclose(
                cells * resolution,
                span,
                atol=1e-6,
                rtol=0.0,
            ):
                raise ValueError("evidence_pc_range 不能被地图分辨率整除")
            values.append(cells)
        return tuple(values)


class DSEvidenceFusion:
    """Dempster-Shafer fusion for occupancy, free, and unknown masses."""

    @staticmethod
    def prob_to_mass(occ_prob: np.ndarray, reliability) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = np.clip(occ_prob, 0.0, 1.0)
        if isinstance(reliability, np.ndarray):
            r = np.clip(reliability, 0.0, 1.0).astype(np.float32)
        else:
            r = float(np.clip(reliability, 0.0, 1.0))
        m_occ = r * p
        m_free = r * (1.0 - p)
        m_unknown = np.full_like(p, 1.0 - r) if not isinstance(r, np.ndarray) else (1.0 - r).astype(np.float32)
        return m_occ, m_free, m_unknown

    @staticmethod
    def fuse_two(
        mass_a: Tuple[np.ndarray, np.ndarray, np.ndarray],
        mass_b: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        a_occ, a_free, a_unk = mass_a
        b_occ, b_free, b_unk = mass_b

        conflict = a_occ * b_free + a_free * b_occ
        den = np.maximum(1.0 - conflict, EPS)

        m_occ = (a_occ * b_occ + a_occ * b_unk + a_unk * b_occ) / den
        m_free = (a_free * b_free + a_free * b_unk + a_unk * b_free) / den
        m_unknown = (a_unk * b_unk) / den

        belief_occ = m_occ
        plausibility_occ = 1.0 - m_free
        ignorance = np.clip(plausibility_occ - belief_occ, 0.0, 1.0)
        return m_occ, m_free, m_unknown, ignorance


class SlidingProbabilisticGridMap:
    """Sliding probabilistic occupancy map with DEM updates."""

    def __init__(self, config: GridMapConfig):
        self.cfg = config
        self.nx, self.ny = self.cfg.shape_xy
        _, _, self.nz = self.cfg.shape_xyz

        self.occ_prob = np.full((self.nx, self.ny), 0.5, dtype=np.float32)
        self.belief = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.plausibility = np.ones((self.nx, self.ny), dtype=np.float32)
        # 初始地图没有观测；unknown 不应被解释成 free。
        self.unknown_mass = np.ones((self.nx, self.ny), dtype=np.float32)

        # 分高度层状态是 pose-aware 地图的权威三维证据；上面的二维数组继续
        # 保留为旧消费者使用的 BEV 兼容输出。
        layer_shape = (self.nx, self.ny, self.nz)
        self.occ_prob_layers = np.full(layer_shape, 0.5, dtype=np.float32)
        self.belief_layers = np.zeros(layer_shape, dtype=np.float32)
        self.plausibility_layers = np.ones(layer_shape, dtype=np.float32)
        self.unknown_mass_layers = np.ones(layer_shape, dtype=np.float32)

        # 动态状态仅在收到显式 evidence 时惰性分配，默认调用不增加三维内存。
        self.dynamic_occ_prob_layers: Optional[np.ndarray] = None
        self.dynamic_belief_layers: Optional[np.ndarray] = None
        self.dynamic_plausibility_layers: Optional[np.ndarray] = None
        self.dynamic_unknown_mass_layers: Optional[np.ndarray] = None

        self.dem_mean = np.full((self.nx, self.ny), np.nan, dtype=np.float32)
        self.dem_var = np.full((self.nx, self.ny), np.nan, dtype=np.float32)

        self.last_timestamp = 0.0
        self.last_T_local_body = np.eye(4, dtype=np.float32)
        self.last_T_body_voxel = np.eye(4, dtype=np.float32)
        self.last_T_local_voxel = np.eye(4, dtype=np.float32)
        self.last_pose_contract = "identity_legacy"
        self.last_body_pose_available = False
        self._has_voxel_update = False
        self.rolling_protocol = (
            "body_anchored_integer_voxel_roll_v1"
            if self.cfg.rolling_enabled
            else "disabled_legacy_fixed_bounds"
        )
        self._rolling_relative_min = np.asarray(
            [self.cfg.x_min, self.cfg.y_min, self.cfg.z_min],
            dtype=np.float64,
        )
        self._rolling_span = np.asarray(
            [
                self.cfg.x_max - self.cfg.x_min,
                self.cfg.y_max - self.cfg.y_min,
                self.cfg.z_max - self.cfg.z_min,
            ],
            dtype=np.float64,
        )
        self.rolling_recenter_count = 0
        self.last_recenter_shift_cells = np.zeros(3, dtype=np.int64)
        self.last_rolling_anchor_local = np.full(3, np.nan, dtype=np.float64)
        self.history: Deque[Dict[str, np.ndarray]] = deque(maxlen=self.cfg.window_size)
        self.ds_fuser = DSEvidenceFusion()

    @staticmethod
    def _shift_grid_with_fill(
        array: np.ndarray,
        shift_cells: np.ndarray,
        fill_value: float,
    ) -> np.ndarray:
        """按 map bounds 的正向移动平移数组，禁止 `np.roll` 回卷旧状态。"""
        source = np.asarray(array)
        axes = min(source.ndim, 3)
        shift = np.asarray(shift_cells[:axes], dtype=np.int64)
        if any(abs(int(value)) >= source.shape[axis] for axis, value in enumerate(shift)):
            return np.full_like(source, fill_value)
        source_slices = [slice(None)] * source.ndim
        destination_slices = [slice(None)] * source.ndim
        for axis, value in enumerate(shift):
            cells = int(value)
            if cells > 0:
                source_slices[axis] = slice(cells, None)
                destination_slices[axis] = slice(None, -cells)
            elif cells < 0:
                source_slices[axis] = slice(None, cells)
                destination_slices[axis] = slice(-cells, None)
        shifted = np.full_like(source, fill_value)
        shifted[tuple(destination_slices)] = source[tuple(source_slices)]
        return shifted

    def _recenter_rolling_window(self, anchor_local_xyz: np.ndarray) -> None:
        """把局部窗口按整数体素锚定到当前 body/LiDAR 原点。"""
        anchor = np.asarray(anchor_local_xyz, dtype=np.float64)
        if anchor.shape != (3,) or not np.all(np.isfinite(anchor)):
            raise ValueError("rolling anchor 必须是三个有限 local 坐标")
        self.last_rolling_anchor_local = anchor.copy()
        if not self.cfg.rolling_enabled:
            self.last_recenter_shift_cells = np.zeros(3, dtype=np.int64)
            return
        current_min = np.asarray(
            [self.cfg.x_min, self.cfg.y_min, self.cfg.z_min],
            dtype=np.float64,
        )
        resolution = np.asarray(
            [self.cfg.x_resolution, self.cfg.y_resolution, self.cfg.z_resolution],
            dtype=np.float64,
        )
        desired_min = anchor + self._rolling_relative_min
        shift_cells = np.rint((desired_min - current_min) / resolution).astype(
            np.int64
        )
        self.last_recenter_shift_cells = shift_cells.copy()
        if not np.any(shift_cells):
            return

        for name, fill_value in (
            ("occ_prob", 0.5),
            ("belief", 0.0),
            ("plausibility", 1.0),
            ("unknown_mass", 1.0),
            ("occ_prob_layers", 0.5),
            ("belief_layers", 0.0),
            ("plausibility_layers", 1.0),
            ("unknown_mass_layers", 1.0),
            ("dem_mean", np.nan),
            ("dem_var", np.nan),
        ):
            setattr(
                self,
                name,
                self._shift_grid_with_fill(
                    getattr(self, name),
                    shift_cells,
                    fill_value,
                ),
            )
        for name, fill_value in (
            ("dynamic_occ_prob_layers", 0.5),
            ("dynamic_belief_layers", 0.0),
            ("dynamic_plausibility_layers", 1.0),
            ("dynamic_unknown_mass_layers", 1.0),
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(
                    self,
                    name,
                    self._shift_grid_with_fill(value, shift_cells, fill_value),
                )
        new_min = current_min + shift_cells.astype(np.float64) * resolution
        self.cfg.x_min, self.cfg.y_min, self.cfg.z_min = new_min.tolist()
        new_max = new_min + self._rolling_span
        self.cfg.x_max, self.cfg.y_max, self.cfg.z_max = new_max.tolist()
        self.rolling_recenter_count += 1
        self.history.clear()

    def _time_decay(self, timestamp: float) -> None:
        dt = max(0.0, float(timestamp) - float(self.last_timestamp))
        if dt <= 0.0:
            return
        decay = float(np.exp(-self.cfg.decay_rate * dt))
        # 时间衰减必须把已有 occupied/free 质量转回 unknown，而不是只改
        # 概率后保留旧 belief，否则三个输出会彼此矛盾。
        bev_mass = self._discount_mass(
            self.belief,
            self.plausibility,
            decay,
        )
        layer_mass = self._discount_mass(
            self.belief_layers,
            self.plausibility_layers,
            decay,
        )
        (
            self.occ_prob,
            self.belief,
            self.plausibility,
            self.unknown_mass,
        ) = self._mass_to_state(bev_mass)
        (
            self.occ_prob_layers,
            self.belief_layers,
            self.plausibility_layers,
            self.unknown_mass_layers,
        ) = self._mass_to_state(layer_mass)
        if self.dynamic_belief_layers is not None:
            dynamic_decay = float(
                np.exp(-self.cfg.dynamic_decay_rate * dt)
            )
            dynamic_mass = self._discount_mass(
                self.dynamic_belief_layers,
                self.dynamic_plausibility_layers,
                dynamic_decay,
            )
            (
                self.dynamic_occ_prob_layers,
                self.dynamic_belief_layers,
                self.dynamic_plausibility_layers,
                self.dynamic_unknown_mass_layers,
            ) = self._mass_to_state(dynamic_mass)

    def _ensure_dynamic_layers(self) -> None:
        """首次收到显式动态 evidence 时惰性创建三态层。"""
        if self.dynamic_occ_prob_layers is not None:
            return
        layer_shape = (self.nx, self.ny, self.nz)
        self.dynamic_occ_prob_layers = np.full(
            layer_shape,
            0.5,
            dtype=np.float32,
        )
        self.dynamic_belief_layers = np.zeros(layer_shape, dtype=np.float32)
        self.dynamic_plausibility_layers = np.ones(layer_shape, dtype=np.float32)
        self.dynamic_unknown_mass_layers = np.ones(layer_shape, dtype=np.float32)

    @staticmethod
    def _validated_dynamic_evidence(
        dynamic_probability: Optional[np.ndarray],
        dynamic_observed_mask: Optional[np.ndarray],
        voxel_shape: Tuple[int, int, int],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """严格校验成对动态概率/观测域，不从 occupancy 或 Doppler 猜测。"""
        if dynamic_probability is None and dynamic_observed_mask is None:
            return None, None
        if dynamic_probability is None or dynamic_observed_mask is None:
            raise ValueError(
                "dynamic_probability 与 dynamic_observed_mask 必须同时提供"
            )
        probability = np.asarray(dynamic_probability, dtype=np.float32)
        observed = np.asarray(dynamic_observed_mask, dtype=np.float32)
        if probability.shape != voxel_shape or observed.shape != voxel_shape:
            raise ValueError(
                "动态 evidence 必须与 voxel XYZ shape 一致: "
                f"probability={probability.shape}, observed={observed.shape}, "
                f"voxel={voxel_shape}"
            )
        if not np.all(np.isfinite(probability)) or not np.all(np.isfinite(observed)):
            raise ValueError("动态 evidence 必须全部为有限数")
        if np.any(probability < 0.0) or np.any(probability > 1.0):
            raise ValueError("dynamic_probability 必须位于 [0,1]")
        if not np.all((observed == 0.0) | (observed == 1.0)):
            raise ValueError("dynamic_observed_mask 必须是严格 0/1")
        if np.any(probability[observed == 0.0] != 0.0):
            raise ValueError("未观测动态单元的 probability 必须为 0")
        return probability, observed

    @staticmethod
    def _discount_mass(
        belief: np.ndarray,
        plausibility: np.ndarray,
        reliability: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """按可靠度折扣已有 D-S 质量，折扣部分归入 unknown。"""
        factor = float(np.clip(reliability, 0.0, 1.0))
        m_occ = factor * np.clip(belief, 0.0, 1.0)
        m_free = factor * np.clip(1.0 - plausibility, 0.0, 1.0)
        m_unknown = np.clip(1.0 - m_occ - m_free, 0.0, 1.0)
        return (
            m_occ.astype(np.float32),
            m_free.astype(np.float32),
            m_unknown.astype(np.float32),
        )

    @staticmethod
    def _mass_to_state(
        mass: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """把 D-S 质量转为 pignistic occupancy 与可审计三态输出。"""
        m_occ, m_free, m_unknown = mass
        probability = np.clip(m_occ + 0.5 * m_unknown, 0.0, 1.0)
        belief = np.clip(m_occ, 0.0, 1.0)
        plausibility = np.clip(1.0 - m_free, 0.0, 1.0)
        unknown = np.clip(m_unknown, 0.0, 1.0)
        return (
            probability.astype(np.float32),
            belief.astype(np.float32),
            plausibility.astype(np.float32),
            unknown.astype(np.float32),
        )

    @staticmethod
    def _max_probability_state_along_z(
        probability: np.ndarray,
        belief: np.ndarray,
        plausibility: np.ndarray,
        unknown: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """沿 Z 选择最大 occupancy 单元，并保持其整组 D-S 状态。"""
        indices = np.argmax(probability, axis=2)[..., np.newaxis]
        return tuple(
            np.take_along_axis(state, indices, axis=2)[..., 0].astype(np.float32)
            for state in (probability, belief, plausibility, unknown)
        )

    @staticmethod
    def _overlay_dynamic_state(
        static_state: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        dynamic_state: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """仅用明确的动态 occupied 状态覆盖静态状态，忽略未覆盖域。"""
        static_probability = static_state[0]
        dynamic_probability = dynamic_state[0]
        use_dynamic = (
            (dynamic_probability > 0.5)
            & (dynamic_probability > static_probability)
        )
        return tuple(
            np.where(use_dynamic, dynamic_value, static_value).astype(np.float32)
            for static_value, dynamic_value in zip(static_state, dynamic_state)
        )

    @staticmethod
    def _validated_pose(
        transform_value: Optional[np.ndarray],
        label: str = "T_local_body",
    ) -> np.ndarray:
        """校验命名刚体变换；缺省为兼容用单位变换。"""
        if transform_value is None:
            return np.eye(4, dtype=np.float32)
        transform = np.asarray(transform_value, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError(f"{label} 必须是有限的 4x4 齐次刚体矩阵")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError(f"{label} 最后一行必须为 [0,0,0,1]")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError(f"{label} 旋转部分必须正交")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError(f"{label} 旋转部分必须是 det=+1 的右手旋转")
        return transform.astype(np.float32)

    def _validated_timestamp(self, timestamp: float) -> float:
        """验证逐帧时间戳，防止逆序帧先衰减再污染地图。"""
        value = float(timestamp)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("timestamp 必须是有限非负数")
        if self._has_voxel_update and value <= float(self.last_timestamp):
            raise ValueError(
                f"timestamp 必须严格递增: current={value}, last={self.last_timestamp}"
            )
        return value

    @staticmethod
    def _observed_layer_mask(
        observed_mask: Optional[np.ndarray],
        voxel_shape: Tuple[int, int, int],
        occupied_layers: np.ndarray,
        authoritative: bool = False,
    ) -> np.ndarray:
        """解析三维 observed 域；权威 mask 不允许被模型预测扩张。"""
        if observed_mask is None:
            if authoritative:
                raise ValueError("authoritative observed mask 不能为空")
            observed = occupied_layers > 0.0
        else:
            mask = np.asarray(observed_mask, dtype=np.float32)
            if not np.all(np.isfinite(mask)):
                raise ValueError("observed_mask 必须全部为有限数")
            if mask.shape == voxel_shape:
                observed = mask > 0.5
            elif mask.shape == voxel_shape[:2]:
                observed = np.broadcast_to(
                    (mask > 0.5)[..., np.newaxis], voxel_shape
                ).copy()
            else:
                raise ValueError(
                    "observed_mask 必须是 (X,Y) 或 (X,Y,Z)，"
                    f"当前为 {mask.shape}，voxel={voxel_shape}"
                )
            if not authoritative:
                observed = observed | (occupied_layers > 0.0)
        return observed.astype(np.float32)

    def _body_to_local_mapping(
        self,
        voxel_shape: Tuple[int, int, int],
        T_local_body: np.ndarray,
        source_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算已观测源体素中心到 local map 体素的前向映射。"""
        if tuple(voxel_shape) != self.cfg.evidence_shape_xyz:
            raise ValueError(
                f"voxel XYZ shape {voxel_shape} != evidence shape "
                f"{self.cfg.evidence_shape_xyz}"
            )
        mask = np.asarray(source_mask, dtype=np.float32)
        if mask.shape != voxel_shape:
            raise ValueError(f"source_mask shape {mask.shape} != voxel shape {voxel_shape}")
        candidate_indices = np.flatnonzero(mask.reshape(-1) > 0.0).astype(np.int64)
        if candidate_indices.size == 0:
            return candidate_indices, candidate_indices.copy()
        source_x, source_y, source_z = np.unravel_index(
            candidate_indices,
            voxel_shape,
        )
        evidence_range = self.cfg.evidence_pc_range
        source_resolution = np.asarray(
            [
                (evidence_range[3] - evidence_range[0]) / voxel_shape[0],
                (evidence_range[4] - evidence_range[1]) / voxel_shape[1],
                (evidence_range[5] - evidence_range[2]) / voxel_shape[2],
            ],
            dtype=np.float64,
        )
        x = evidence_range[0] + (source_x.astype(np.float64) + 0.5) * source_resolution[0]
        y = evidence_range[1] + (source_y.astype(np.float64) + 0.5) * source_resolution[1]
        z = evidence_range[2] + (source_z.astype(np.float64) + 0.5) * source_resolution[2]
        centers = np.stack(
            [x, y, z, np.ones(candidate_indices.size, dtype=np.float64)],
            axis=0,
        )
        local = np.asarray(T_local_body, dtype=np.float64) @ centers
        ix = np.floor((local[0] - self.cfg.x_min) / self.cfg.x_resolution).astype(np.int64)
        iy = np.floor((local[1] - self.cfg.y_min) / self.cfg.y_resolution).astype(np.int64)
        iz = np.floor((local[2] - self.cfg.z_min) / self.cfg.z_resolution).astype(np.int64)
        valid = (
            (ix >= 0) & (ix < self.nx)
            & (iy >= 0) & (iy < self.ny)
            & (iz >= 0) & (iz < self.nz)
        )
        source_indices = candidate_indices[valid]
        destination_indices = np.ravel_multi_index(
            (ix[valid], iy[valid], iz[valid]),
            (self.nx, self.ny, self.nz),
        ).astype(np.int64)
        return source_indices, destination_indices

    def _warp_voxel_to_local(
        self,
        voxel_xyzc: np.ndarray,
        observed_layers: np.ndarray,
        T_local_body: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """前向投影体素和可靠度辅助映射，越出局部地图的点显式丢弃。"""
        source_indices, destination_indices = self._body_to_local_mapping(
            tuple(int(value) for value in voxel_xyzc.shape[:3]),
            T_local_body,
            observed_layers,
        )
        cell_count = self.nx * self.ny * self.nz
        channels = int(voxel_xyzc.shape[-1])
        source = voxel_xyzc.reshape(-1, channels)
        destination = np.zeros((cell_count, channels), dtype=np.float32)

        source_occ = np.clip(source[source_indices, 0], 0.0, 1.0)
        np.maximum.at(destination[:, 0], destination_indices, source_occ)
        if channels > 1:
            weight_sum = np.zeros(cell_count, dtype=np.float32)
            np.add.at(weight_sum, destination_indices, source_occ)
            for channel in range(1, channels):
                weighted_sum = np.zeros(cell_count, dtype=np.float32)
                np.add.at(
                    weighted_sum,
                    destination_indices,
                    source[source_indices, channel] * source_occ,
                )
                valid_weight = weight_sum > EPS
                destination[valid_weight, channel] = (
                    weighted_sum[valid_weight] / weight_sum[valid_weight]
                )

        warped_observed = np.zeros(cell_count, dtype=np.float32)
        np.maximum.at(
            warped_observed,
            destination_indices,
            observed_layers.reshape(-1)[source_indices],
        )
        return (
            destination.reshape(self.nx, self.ny, self.nz, channels),
            warped_observed.reshape(self.nx, self.ny, self.nz),
            source_indices,
            destination_indices,
        )

    def _scatter_layer_max(
        self,
        source_layers: np.ndarray,
        source_indices: np.ndarray,
        destination_indices: np.ndarray,
    ) -> np.ndarray:
        """把源标量层按同一 body→local 映射做 max-splat。"""
        destination = np.zeros(self.nx * self.ny * self.nz, dtype=np.float32)
        np.maximum.at(
            destination,
            destination_indices,
            np.asarray(source_layers, dtype=np.float32).reshape(-1)[source_indices],
        )
        return destination.reshape(self.nx, self.ny, self.nz)

    def _warp_bev_to_local(
        self,
        bev_xy: np.ndarray,
        T_local_body: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """在 body 的 z=0 平面上把 IR BEV 前向投影到 local XY。"""
        x = self.cfg.x_min + (np.arange(self.nx, dtype=np.float64) + 0.5) * self.cfg.x_resolution
        y = self.cfg.y_min + (np.arange(self.ny, dtype=np.float64) + 0.5) * self.cfg.y_resolution
        xx, yy = np.meshgrid(x, y, indexing="ij")
        centers = np.stack(
            [xx.reshape(-1), yy.reshape(-1), np.zeros(xx.size), np.ones(xx.size)],
            axis=0,
        )
        local = np.asarray(T_local_body, dtype=np.float64) @ centers
        ix = np.floor((local[0] - self.cfg.x_min) / self.cfg.x_resolution).astype(np.int64)
        iy = np.floor((local[1] - self.cfg.y_min) / self.cfg.y_resolution).astype(np.int64)
        valid = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        destination_indices = np.ravel_multi_index(
            (ix[valid], iy[valid]),
            (self.nx, self.ny),
        )
        destination = np.zeros(self.nx * self.ny, dtype=np.float32)
        np.maximum.at(
            destination,
            destination_indices,
            np.asarray(bev_xy, dtype=np.float32).reshape(-1)[valid],
        )
        observed = np.zeros(self.nx * self.ny, dtype=np.float32)
        np.maximum.at(observed, destination_indices, 1.0)
        return (
            destination.reshape(self.nx, self.ny),
            observed.reshape(self.nx, self.ny),
        )

    @staticmethod
    def _odom_confidence(odom_cov: Optional[np.ndarray]) -> float:
        if odom_cov is None:
            return 1.0
        # TODO: 引入显式状态协方差传播(P_k = F P_{k-1} F^T + Q)，替代当前trace经验映射。
        cov = np.asarray(odom_cov, dtype=np.float32)
        trace = float(np.trace(cov)) if cov.ndim == 2 else float(np.sum(cov))
        return float(np.exp(-0.35 * max(0.0, trace)))

    def _fuse_occ(self, obs_occ: np.ndarray, obs_reliability) -> None:
        # 已有地图本身就是 D-S 质量，不能把 unknown 的 0.5 概率重新解释成
        # 高可靠 occupied/free 各半证据。
        prior_mass = self._discount_mass(
            self.belief,
            self.plausibility,
            self.cfg.prior_reliability,
        )
        obs_mass = self.ds_fuser.prob_to_mass(obs_occ, obs_reliability)
        m_occ, m_free, m_unknown, ignorance = self.ds_fuser.fuse_two(
            prior_mass,
            obs_mass,
        )
        (
            self.occ_prob,
            self.belief,
            self.plausibility,
            self.unknown_mass,
        ) = self._mass_to_state((m_occ, m_free, m_unknown))

        self.history.append(
            {
                "occ_prob": self.occ_prob.copy(),
                "belief": self.belief.copy(),
                "ignorance": ignorance.astype(np.float32),
                "unknown_mass": self.unknown_mass.copy(),
            }
        )

    def _fuse_occ_layers(
        self,
        obs_occ_layers: np.ndarray,
        obs_reliability_layers: np.ndarray,
    ) -> None:
        """在固定 local 坐标系内逐高度层融合 occupied/free/unknown 证据。"""
        prior_mass = self._discount_mass(
            self.belief_layers,
            self.plausibility_layers,
            self.cfg.prior_reliability,
        )
        obs_mass = self.ds_fuser.prob_to_mass(
            obs_occ_layers,
            obs_reliability_layers,
        )
        m_occ, m_free, m_unknown, _ignorance = self.ds_fuser.fuse_two(
            prior_mass,
            obs_mass,
        )
        (
            self.occ_prob_layers,
            self.belief_layers,
            self.plausibility_layers,
            self.unknown_mass_layers,
        ) = self._mass_to_state((m_occ, m_free, m_unknown))

    def _fuse_dynamic_layers(
        self,
        dynamic_probability_layers: np.ndarray,
        dynamic_reliability_layers: np.ndarray,
    ) -> None:
        """融合独立动态概率；静态地图状态由调用方单独更新。"""
        self._ensure_dynamic_layers()
        prior_mass = self._discount_mass(
            self.dynamic_belief_layers,
            self.dynamic_plausibility_layers,
            self.cfg.prior_reliability,
        )
        observation_mass = self.ds_fuser.prob_to_mass(
            dynamic_probability_layers,
            dynamic_reliability_layers,
        )
        m_occ, m_free, m_unknown, _ignorance = self.ds_fuser.fuse_two(
            prior_mass,
            observation_mass,
        )
        (
            self.dynamic_occ_prob_layers,
            self.dynamic_belief_layers,
            self.dynamic_plausibility_layers,
            self.dynamic_unknown_mass_layers,
        ) = self._mass_to_state((m_occ, m_free, m_unknown))

    @staticmethod
    def _observed_bev_mask(
        observed_mask: Optional[np.ndarray],
        voxel_shape: Tuple[int, int, int],
        occupied_bev: np.ndarray,
    ) -> np.ndarray:
        """解析显式 observed mask；没有 mask 时仅把 occupied 单元视为已观测。"""
        if observed_mask is None:
            observed_bev = occupied_bev > 0.0
        else:
            mask = np.asarray(observed_mask, dtype=np.float32)
            if not np.all(np.isfinite(mask)):
                raise ValueError("observed_mask 必须全部为有限数")
            if mask.shape == voxel_shape:
                observed_bev = np.any(mask > 0.5, axis=2)
            elif mask.shape == voxel_shape[:2]:
                observed_bev = mask > 0.5
            else:
                raise ValueError(
                    "observed_mask 必须是 (X,Y) 或 (X,Y,Z)，"
                    f"当前为 {mask.shape}，voxel={voxel_shape}"
                )
            # 即使外部 mask 漏标，occupied 单元也必须保持可观测。
            observed_bev = observed_bev | (occupied_bev > 0.0)
        return observed_bev.astype(np.float32)

    def _doppler_variance_bev(self, voxel_xyzc: np.ndarray) -> np.ndarray:
        if voxel_xyzc.ndim != 4 or voxel_xyzc.shape[-1] <= 3:
            return np.zeros(voxel_xyzc.shape[:2], dtype=np.float32)
        occ3d = np.clip(voxel_xyzc[..., 0], 0.0, 1.0)
        var3d = np.clip(voxel_xyzc[..., 3], 0.0, 50.0)
        occ_sum = occ3d.sum(axis=2)
        weighted = (occ3d * var3d).sum(axis=2)
        out = np.zeros_like(weighted, dtype=np.float32)
        valid = occ_sum > EPS
        out[valid] = weighted[valid] / np.maximum(occ_sum[valid], EPS)
        return out

    @staticmethod
    def _uncertainty_to_bev(model_uncertainty: Optional[np.ndarray], target_shape: Tuple[int, int]) -> np.ndarray:
        if model_uncertainty is None:
            return np.zeros(target_shape, dtype=np.float32)
        unc = np.asarray(model_uncertainty, dtype=np.float32)
        unc = np.squeeze(unc)
        if unc.ndim == 3:
            unc = np.nanmean(unc, axis=2)
        if unc.ndim != 2:
            return np.zeros(target_shape, dtype=np.float32)
        if unc.shape != target_shape:
            # Lightweight nearest-neighbor resize without adding image deps.
            x_idx = np.clip(
                np.round(np.linspace(0, unc.shape[0] - 1, target_shape[0])).astype(np.int64),
                0,
                unc.shape[0] - 1,
            )
            y_idx = np.clip(
                np.round(np.linspace(0, unc.shape[1] - 1, target_shape[1])).astype(np.int64),
                0,
                unc.shape[1] - 1,
            )
            unc = unc[x_idx][:, y_idx]
        return np.nan_to_num(np.clip(unc, 0.0, 50.0), nan=50.0).astype(np.float32)

    def observation_reliability_map(
        self,
        voxel_xyzc: np.ndarray,
        sensor: str = "radar",
        odom_cov: Optional[np.ndarray] = None,
        model_uncertainty: Optional[np.ndarray] = None,
        calib_confidence: float = 1.0,
        use_legacy_voxel_doppler_variance: bool = True,
    ) -> np.ndarray:
        base = self.cfg.infrared_reliability if sensor == "infrared" else self.cfg.radar_reliability
        nx = voxel_xyzc.shape[0]
        evidence_range = self.cfg.evidence_pc_range
        x_centers = evidence_range[0] + (
            np.arange(nx, dtype=np.float32) + 0.5
        ) * ((evidence_range[3] - evidence_range[0]) / max(nx, 1))
        max_range = max(
            evidence_range[3] - evidence_range[0],
            EPS,
        )
        speed_scale = float(np.clip(self.cfg.speed_m_s / 50.0, 0.7, 1.4))
        range_conf = np.clip(
            1.0 - 0.65 * speed_scale * ((x_centers - evidence_range[0]) / max_range),
            0.18,
            1.0,
        )[:, np.newaxis]
        # 只有显式 legacy/raw voxel 兼容路径允许把 ch3 解释为 Radar 方差。
        doppler_variance = (
            self._doppler_variance_bev(voxel_xyzc)
            if use_legacy_voxel_doppler_variance
            else np.zeros(voxel_xyzc.shape[:2], dtype=np.float32)
        )
        variance_conf = 1.0 / (1.0 + doppler_variance / 10.0)
        model_unc = self._uncertainty_to_bev(model_uncertainty, voxel_xyzc.shape[:2])
        model_conf = 1.0 / (1.0 + model_unc / 5.0)
        odom_conf = self._odom_confidence(odom_cov)
        calib_conf = float(np.clip(calib_confidence, 0.02, 1.0))
        return np.clip(base * range_conf * variance_conf * model_conf * odom_conf * calib_conf, 0.02, 1.0).astype(np.float32)

    def _update_dem_from_voxel(
        self,
        voxel_xyzc: np.ndarray,
        observed_layers: np.ndarray,
    ) -> None:
        """以 observed occupancy 的 Z 分布更新 DEM，方差单位固定为 m^2。"""
        observed = np.asarray(observed_layers, dtype=np.float32)
        if observed.shape != voxel_xyzc.shape[:3] or not np.all(np.isfinite(observed)):
            raise ValueError("DEM observed_layers 必须是有限且与 voxel XYZ 同形的数组")
        occ3d = np.clip(voxel_xyzc[..., 0], 0.0, 1.0) * (observed > 0.5)
        z_bins = occ3d.shape[2]
        z_values = self.cfg.z_min + (np.arange(z_bins, dtype=np.float32) + 0.5) * self.cfg.z_resolution

        occ_sum = occ3d.sum(axis=2)
        valid = occ_sum > 0.1
        if not np.any(valid):
            return

        z_mean = (occ3d * z_values[np.newaxis, np.newaxis, :]).sum(axis=2) / np.maximum(occ_sum, EPS)
        z_second = (occ3d * (z_values[np.newaxis, np.newaxis, :] ** 2)).sum(axis=2) / np.maximum(occ_sum, EPS)
        z_var = np.maximum(0.0, z_second - z_mean ** 2)

        prev_valid = ~np.isnan(self.dem_mean)
        blend_w = np.clip(self.belief, 0.1, 0.95)

        both = valid & prev_valid
        new_only = valid & (~prev_valid)

        self.dem_mean[both] = (1.0 - blend_w[both]) * self.dem_mean[both] + blend_w[both] * z_mean[both]
        self.dem_var[both] = (1.0 - blend_w[both]) * np.nan_to_num(self.dem_var[both], nan=0.0) + blend_w[both] * z_var[both]

        self.dem_mean[new_only] = z_mean[new_only]
        self.dem_var[new_only] = z_var[new_only]

    def update_from_voxel(
        self,
        voxel_xyzc: np.ndarray,
        timestamp: float,
        sensor: str = "radar",
        odom_cov: Optional[np.ndarray] = None,
        model_uncertainty: Optional[np.ndarray] = None,
        calib_confidence: float = 1.0,
        observed_mask: Optional[np.ndarray] = None,
        T_local_body: Optional[np.ndarray] = None,
        T_body_voxel: Optional[np.ndarray] = None,
        T_local_voxel: Optional[np.ndarray] = None,
        dynamic_probability: Optional[np.ndarray] = None,
        dynamic_observed_mask: Optional[np.ndarray] = None,
        observed_mask_authoritative: bool = False,
        evidence_semantics: str = LEGACY_MULTICHANNEL_EVIDENCE_SEMANTICS,
    ) -> None:
        """把 voxel 系 evidence 经 ``body`` 变换到 ``local`` 后更新地图。

        兼容调用省略 ``T_body_voxel`` 时表示 voxel frame 就是 body；机载正式
        LiDAR-frame inference 必须显式传入 LiDAR→body。仅离线经验合同可直接
        传入 ``T_local_voxel``，并且不得同时伪造 body 链。
        """
        voxel_xyzc = np.asarray(voxel_xyzc, dtype=np.float32)
        if voxel_xyzc.ndim != 4 or voxel_xyzc.shape[-1] < 1:
            raise ValueError(f"voxel_xyzc 必须是 (X,Y,Z,C)，当前为 {voxel_xyzc.shape}")
        if tuple(voxel_xyzc.shape[:3]) != self.cfg.evidence_shape_xyz:
            raise ValueError(
                f"voxel XYZ shape {voxel_xyzc.shape[:3]} != evidence shape "
                f"{self.cfg.evidence_shape_xyz}"
            )
        if not np.all(np.isfinite(voxel_xyzc)):
            raise ValueError("voxel_xyzc 必须全部为有限数")
        evidence_semantics = str(evidence_semantics).strip()
        if evidence_semantics not in {
            LEGACY_MULTICHANNEL_EVIDENCE_SEMANTICS,
            GENERATED_OCCUPANCY_EVIDENCE_SEMANTICS,
        }:
            raise ValueError(f"未知 evidence_semantics: {evidence_semantics}")
        generated_prediction = (
            evidence_semantics == GENERATED_OCCUPANCY_EVIDENCE_SEMANTICS
        )
        if generated_prediction:
            occupancy = voxel_xyzc[..., 0]
            if np.any(occupancy < 0.0) or np.any(occupancy > 1.0):
                raise ValueError("generated occupancy probability 必须位于 [0,1]")
            if not observed_mask_authoritative or observed_mask is None:
                raise ValueError("generated prediction 必须使用外部 authoritative observed mask")

        # 所有外部输入必须在时间衰减或证据融合前完成校验，失败时地图无副作用。
        validated_timestamp = self._validated_timestamp(timestamp)
        if T_local_voxel is not None:
            if T_local_body is not None or T_body_voxel is not None:
                raise ValueError("T_local_voxel 与 T_local_body/T_body_voxel 互斥")
            validated_local_voxel = self._validated_pose(
                T_local_voxel,
                "T_local_voxel",
            )
            validated_body_pose = np.eye(4, dtype=np.float32)
            validated_body_voxel = np.eye(4, dtype=np.float32)
            pose_contract = "direct_local_voxel"
            body_pose_available = False
        else:
            validated_body_pose = self._validated_pose(
                T_local_body,
                "T_local_body",
            )
            validated_body_voxel = self._validated_pose(
                T_body_voxel,
                "T_body_voxel",
            )
            validated_local_voxel = self._validated_pose(
                validated_body_pose @ validated_body_voxel,
                "T_local_voxel",
            )
            pose_contract = (
                "body_chain"
                if T_local_body is not None or T_body_voxel is not None
                else "identity_legacy"
            )
            body_pose_available = T_local_body is not None
        voxel_shape = tuple(int(value) for value in voxel_xyzc.shape[:3])
        dynamic_probability, dynamic_observed = self._validated_dynamic_evidence(
            dynamic_probability,
            dynamic_observed_mask,
            voxel_shape,
        )
        if (
            dynamic_probability is not None
            and self.cfg.dynamic_decay_rate <= self.cfg.decay_rate
        ):
            raise ValueError(
                "启用动态 evidence 时 dynamic_decay_rate 必须严格大于静态 decay_rate"
            )
        source_occ_layers = np.clip(voxel_xyzc[..., 0], 0.0, 1.0)
        source_observed_layers = self._observed_layer_mask(
            observed_mask,
            voxel_shape=voxel_shape,
            occupied_layers=source_occ_layers,
            authoritative=bool(observed_mask_authoritative),
        )
        static_source_occ_layers = source_occ_layers.copy()
        mapping_mask = source_observed_layers
        if dynamic_probability is not None:
            # 显式动态概率从静态证据中扣除；未标为 observed 的位置保持原语义。
            dynamic_cells = dynamic_observed > 0.0
            static_source_occ_layers[dynamic_cells] *= (
                1.0 - dynamic_probability[dynamic_cells]
            )
            mapping_mask = np.maximum(source_observed_layers, dynamic_observed)
        rolling_anchor = (
            validated_local_voxel[:3, 3]
            if pose_contract == "direct_local_voxel"
            else validated_body_pose[:3, 3]
        )
        self._recenter_rolling_window(rolling_anchor)
        source_reliability_bev = self.observation_reliability_map(
            voxel_xyzc,
            sensor=sensor,
            odom_cov=odom_cov,
            model_uncertainty=model_uncertainty,
            calib_confidence=calib_confidence,
            use_legacy_voxel_doppler_variance=not generated_prediction,
        )
        source_reliability_layers = (
            source_reliability_bev[..., np.newaxis] * source_observed_layers
        )
        warped_voxel, _warped_observed, source_indices, destination_indices = (
            self._warp_voxel_to_local(
                voxel_xyzc,
                mapping_mask,
                validated_local_voxel,
            )
        )
        warped_static_occ_layers = self._scatter_layer_max(
            static_source_occ_layers,
            source_indices,
            destination_indices,
        )
        reliability_layers = self._scatter_layer_max(
            source_reliability_layers,
            source_indices,
            destination_indices,
        )
        warped_source_observed_layers = self._scatter_layer_max(
            source_observed_layers,
            source_indices,
            destination_indices,
        )
        warped_dynamic_probability = None
        dynamic_reliability_layers = None
        if dynamic_probability is not None:
            warped_dynamic_probability = self._scatter_layer_max(
                dynamic_probability,
                source_indices,
                destination_indices,
            )
            dynamic_reliability_layers = self._scatter_layer_max(
                self._odom_confidence(odom_cov) * dynamic_observed,
                source_indices,
                destination_indices,
            )

        self._time_decay(validated_timestamp)

        odom_conf = self._odom_confidence(odom_cov)
        obs_occ_layers = np.clip(warped_static_occ_layers, 0.0, 1.0)
        adjusted_occ_layers = 0.5 + odom_conf * (obs_occ_layers - 0.5)
        adjusted_occ_bev = 0.5 + odom_conf * (
            np.max(obs_occ_layers, axis=2) - 0.5
        )
        reliability_bev = np.max(reliability_layers, axis=2)

        self._fuse_occ(
            obs_occ=adjusted_occ_bev,
            obs_reliability=reliability_bev,
        )
        self._fuse_occ_layers(
            obs_occ_layers=adjusted_occ_layers,
            obs_reliability_layers=reliability_layers,
        )
        if warped_dynamic_probability is not None:
            self._fuse_dynamic_layers(
                dynamic_probability_layers=warped_dynamic_probability,
                dynamic_reliability_layers=dynamic_reliability_layers,
            )
        self._update_dem_from_voxel(
            warped_voxel,
            observed_layers=warped_source_observed_layers,
        )
        self.last_timestamp = validated_timestamp
        self.last_T_local_body = validated_body_pose.copy()
        self.last_T_body_voxel = validated_body_voxel.copy()
        self.last_T_local_voxel = validated_local_voxel.copy()
        self.last_pose_contract = pose_contract
        self.last_body_pose_available = body_pose_available
        self._has_voxel_update = True

    def update_from_ir_bev(
        self,
        bev_xy: np.ndarray,
        timestamp: float,
        T_local_body: Optional[np.ndarray] = None,
    ) -> None:
        """把 body 系 IR BEV 使用同一帧位姿投影到 local map 后融合。"""
        bev = np.asarray(bev_xy, dtype=np.float32)
        if bev.ndim == 3:
            bev = bev[..., 0]
        if bev.shape != self.occ_prob.shape:
            raise ValueError(f"Infrared BEV shape {bev.shape} != map shape {self.occ_prob.shape}")
        if not np.all(np.isfinite(bev)):
            raise ValueError("Infrared BEV 必须全部为有限数")
        value = float(timestamp)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("timestamp 必须是有限非负数")
        if self._has_voxel_update and value < float(self.last_timestamp):
            raise ValueError("IR timestamp 不得早于最近 voxel timestamp")
        validated_pose = self._validated_pose(T_local_body)
        self._recenter_rolling_window(validated_pose[:3, 3])
        warped_bev, warped_observed = self._warp_bev_to_local(
            np.clip(bev, 0.0, 1.0),
            validated_pose,
        )
        self._time_decay(value)
        self._fuse_occ(
            obs_occ=warped_bev,
            obs_reliability=self.cfg.infrared_reliability * warped_observed,
        )
        self.last_timestamp = value
        self.last_T_local_body = validated_pose.copy()

    def fuse_with_prior_dem(self, prior_dem: np.ndarray, prior_confidence: float = 0.6) -> None:
        """Fuse a prior DEM with the online DEM estimate."""
        dem = np.asarray(prior_dem, dtype=np.float32)
        # TODO: 增加D-S冲突度驱动的动态先验权重，而不是固定prior_confidence。
        if dem.shape != self.dem_mean.shape:
            raise ValueError(f"prior DEM shape {dem.shape} != map shape {self.dem_mean.shape}")

        valid_online = ~np.isnan(self.dem_mean)
        valid_prior = ~np.isnan(dem)
        both = valid_online & valid_prior
        prior_only = (~valid_online) & valid_prior

        w = float(np.clip(prior_confidence, 0.0, 1.0))
        self.dem_mean[both] = (1.0 - w) * self.dem_mean[both] + w * dem[both]
        self.dem_mean[prior_only] = dem[prior_only]

    def snapshot(self) -> Dict[str, np.ndarray]:
        snapshot = {
            "occ_prob": self.occ_prob.copy(),
            "belief": self.belief.copy(),
            "plausibility": self.plausibility.copy(),
            "unknown_mass": self.unknown_mass.copy(),
            "occ_prob_layers": self.occ_prob_layers.copy(),
            "belief_layers": self.belief_layers.copy(),
            "plausibility_layers": self.plausibility_layers.copy(),
            "unknown_mass_layers": self.unknown_mass_layers.copy(),
            "dem_mean": self.dem_mean.copy(),
            "dem_var": self.dem_var.copy(),
            "last_T_local_body": self.last_T_local_body.copy(),
            "last_T_body_voxel": self.last_T_body_voxel.copy(),
            "last_T_local_voxel": self.last_T_local_voxel.copy(),
            "last_pose_contract": np.asarray(self.last_pose_contract),
            "last_body_pose_available": np.asarray(
                self.last_body_pose_available,
                dtype=np.uint8,
            ),
            "last_timestamp": np.asarray(self.last_timestamp, dtype=np.float64),
            "rolling_enabled": np.asarray(
                self.cfg.rolling_enabled,
                dtype=np.uint8,
            ),
            "rolling_protocol": np.asarray(self.rolling_protocol),
            "rolling_recenter_count": np.asarray(
                self.rolling_recenter_count,
                dtype=np.int64,
            ),
            "last_recenter_shift_cells": self.last_recenter_shift_cells.copy(),
            "last_rolling_anchor_local": self.last_rolling_anchor_local.copy(),
            "map_pc_range_local": np.asarray(
                [
                    self.cfg.x_min,
                    self.cfg.y_min,
                    self.cfg.z_min,
                    self.cfg.x_max,
                    self.cfg.y_max,
                    self.cfg.z_max,
                ],
                dtype=np.float64,
            ),
            "dynamic_layer_enabled": np.asarray(
                self.dynamic_occ_prob_layers is not None,
                dtype=np.uint8,
            ),
        }
        if self.dynamic_occ_prob_layers is None:
            return snapshot

        # 已有键继续表示避障消费者需要的静态∪动态保守组合；新增 static/dynamic
        # 键提供可审计分离状态，内部持久状态仍分别更新和衰减。
        snapshot.update(
            {
                # 基础 snapshot 已与内部状态隔离；复用这些副本可避免再复制
                # 一整套三维静态层，后续替换 legacy 键不会修改其内容。
                "static_occ_prob": snapshot["occ_prob"],
                "static_belief": snapshot["belief"],
                "static_plausibility": snapshot["plausibility"],
                "static_unknown_mass": snapshot["unknown_mass"],
                "static_occ_prob_layers": snapshot["occ_prob_layers"],
                "static_belief_layers": snapshot["belief_layers"],
                "static_plausibility_layers": snapshot["plausibility_layers"],
                "static_unknown_mass_layers": snapshot["unknown_mass_layers"],
                "dynamic_occ_prob_layers": self.dynamic_occ_prob_layers.copy(),
                "dynamic_belief_layers": self.dynamic_belief_layers.copy(),
                "dynamic_plausibility_layers": self.dynamic_plausibility_layers.copy(),
                "dynamic_unknown_mass_layers": self.dynamic_unknown_mass_layers.copy(),
            }
        )
        dynamic_bev_state = self._max_probability_state_along_z(
            self.dynamic_occ_prob_layers,
            self.dynamic_belief_layers,
            self.dynamic_plausibility_layers,
            self.dynamic_unknown_mass_layers,
        )
        (
            dynamic_bev,
            dynamic_belief_bev,
            dynamic_plausibility_bev,
            dynamic_unknown_bev,
        ) = dynamic_bev_state
        combined_bev = self._overlay_dynamic_state(
            (
                snapshot["occ_prob"],
                snapshot["belief"],
                snapshot["plausibility"],
                snapshot["unknown_mass"],
            ),
            dynamic_bev_state,
        )
        combined_layers = self._overlay_dynamic_state(
            (
                snapshot["occ_prob_layers"],
                snapshot["belief_layers"],
                snapshot["plausibility_layers"],
                snapshot["unknown_mass_layers"],
            ),
            (
                self.dynamic_occ_prob_layers,
                self.dynamic_belief_layers,
                self.dynamic_plausibility_layers,
                self.dynamic_unknown_mass_layers,
            ),
        )
        snapshot.update(
            {
                "dynamic_occ_prob": dynamic_bev.copy(),
                "dynamic_belief": dynamic_belief_bev.copy(),
                "dynamic_plausibility": dynamic_plausibility_bev.copy(),
                "dynamic_unknown_mass": dynamic_unknown_bev.copy(),
                "occ_prob": combined_bev[0],
                "belief": combined_bev[1],
                "plausibility": combined_bev[2],
                "unknown_mass": combined_bev[3],
                "occ_prob_layers": combined_layers[0],
                "belief_layers": combined_layers[1],
                "plausibility_layers": combined_layers[2],
                "unknown_mass_layers": combined_layers[3],
            }
        )
        return snapshot


class LazyLocalMapQuery:
    """Lightweight local lazy proximity query helper."""

    def __init__(self, config: GridMapConfig, occ_threshold: float = 0.55):
        self.cfg = config
        self.occ_threshold = float(occ_threshold)
        self._occupied_xy_m: Optional[np.ndarray] = None
        self._belief_map: Optional[np.ndarray] = None
        self._unknown_map: Optional[np.ndarray] = None
        self._occupied_xyz_m: Optional[np.ndarray] = None
        self._belief_layers: Optional[np.ndarray] = None
        self._unknown_layers: Optional[np.ndarray] = None

    def refresh(self, map_snapshot: Dict[str, np.ndarray]) -> None:
        occ = map_snapshot["occ_prob"]
        belief = map_snapshot["belief"]
        unknown = np.asarray(
            map_snapshot.get("unknown_mass", np.ones_like(occ)),
            dtype=np.float32,
        )
        if unknown.shape != occ.shape or not np.all(np.isfinite(unknown)):
            raise ValueError("unknown_mass 必须与 occ_prob 形状一致且全部有限")

        idx = np.argwhere(occ >= self.occ_threshold)
        if idx.shape[0] == 0:
            self._occupied_xy_m = np.zeros((0, 2), dtype=np.float32)
        else:
            x_m = self.cfg.x_min + (idx[:, 0].astype(np.float32) + 0.5) * self.cfg.x_resolution
            y_m = self.cfg.y_min + (idx[:, 1].astype(np.float32) + 0.5) * self.cfg.y_resolution
            self._occupied_xy_m = np.stack([x_m, y_m], axis=1)
        self._belief_map = belief
        self._unknown_map = np.clip(unknown, 0.0, 1.0)

        # 新快照优先缓存三维层；缺少 layers 的旧快照仍保持二维查询能力。
        occ_layers = map_snapshot.get("occ_prob_layers")
        belief_layers = map_snapshot.get("belief_layers")
        if occ_layers is None or belief_layers is None:
            self._occupied_xyz_m = None
            self._belief_layers = None
            self._unknown_layers = None
            return
        occ_layers = np.asarray(occ_layers, dtype=np.float32)
        belief_layers = np.asarray(belief_layers, dtype=np.float32)
        if occ_layers.shape != self.cfg.shape_xyz or belief_layers.shape != occ_layers.shape:
            raise ValueError(
                "occ_prob_layers/belief_layers 必须与局部地图 shape_xyz 一致"
            )
        layer_idx = np.argwhere(occ_layers >= self.occ_threshold)
        if layer_idx.shape[0] == 0:
            self._occupied_xyz_m = np.zeros((0, 3), dtype=np.float32)
        else:
            x_m = self.cfg.x_min + (layer_idx[:, 0].astype(np.float32) + 0.5) * self.cfg.x_resolution
            y_m = self.cfg.y_min + (layer_idx[:, 1].astype(np.float32) + 0.5) * self.cfg.y_resolution
            z_m = self.cfg.z_min + (layer_idx[:, 2].astype(np.float32) + 0.5) * self.cfg.z_resolution
            self._occupied_xyz_m = np.stack([x_m, y_m, z_m], axis=1)
        self._belief_layers = belief_layers
        unknown_layers = np.asarray(
            map_snapshot.get("unknown_mass_layers", np.ones_like(occ_layers)),
            dtype=np.float32,
        )
        if unknown_layers.shape != occ_layers.shape or not np.all(
            np.isfinite(unknown_layers)
        ):
            raise ValueError(
                "unknown_mass_layers 必须与 occ_prob_layers 形状一致且全部有限"
            )
        self._unknown_layers = np.clip(unknown_layers, 0.0, 1.0)

    def query_proximity(
        self,
        x_m: float,
        y_m: float,
        search_radius: float = 15.0,
        z_m: Optional[float] = None,
        speed_m_s: float = 0.0,
        reaction_time_s: float = 0.0,
        brake_deceleration_m_s2: float = 1.0,
        safety_margin_m: float = 5.0,
        max_unknown_mass: float = 0.5,
    ) -> Dict[str, object]:
        """返回 ``clear/obstacle/unknown`` 三态最近障碍查询和可审计原因。"""
        values = [
            float(x_m),
            float(y_m),
            float(search_radius),
            float(max_unknown_mass),
        ]
        if z_m is not None:
            values.append(float(z_m))
        if (
            not np.all(np.isfinite(values))
            or float(search_radius) < 0.0
            or float(max_unknown_mass) < 0.0
            or float(max_unknown_mass) > 1.0
        ):
            raise ValueError(
                "查询坐标/search_radius/max_unknown_mass 必须有限且位于有效范围"
            )
        safety_distance = compute_safety_distance_m(
            speed_m_s,
            reaction_time_s,
            brake_deceleration_m_s2,
            safety_margin_m,
        )

        def result(state, reason, distance=float("inf"), uncertainty=1.0, **extra):
            payload = {
                "state": state,
                "reason": reason,
                "distance": float(distance),
                "uncertainty": float(uncertainty),
                "safety_distance_m": float(safety_distance),
                "is_risky": 0.0 if state == "clear" else 1.0,
            }
            payload.update(extra)
            return payload

        use_layers = (
            z_m is not None
            and self._occupied_xyz_m is not None
            and self._belief_layers is not None
        )
        occupied = self._occupied_xyz_m if use_layers else self._occupied_xy_m
        belief = self._belief_layers if use_layers else self._belief_map
        unknown = self._unknown_layers if use_layers else self._unknown_map
        if occupied is None or belief is None or unknown is None:
            return result("unknown", "map_not_initialized")

        min_dist = float("inf")
        uncertainty = 1.0
        if occupied.shape[0] > 0:
            query_values = [x_m, y_m, z_m] if use_layers else [x_m, y_m]
            q = np.asarray(query_values, dtype=np.float32)
            dists = np.linalg.norm(occupied - q[np.newaxis, :], axis=1)
            min_idx = int(np.argmin(dists))
            min_dist = float(dists[min_idx])
            px = int(np.clip((occupied[min_idx, 0] - self.cfg.x_min) / self.cfg.x_resolution, 0, belief.shape[0] - 1))
            py = int(np.clip((occupied[min_idx, 1] - self.cfg.y_min) / self.cfg.y_resolution, 0, belief.shape[1] - 1))
            if use_layers:
                pz = int(np.clip((occupied[min_idx, 2] - self.cfg.z_min) / self.cfg.z_resolution, 0, belief.shape[2] - 1))
                nearest_belief = belief[px, py, pz]
            else:
                nearest_belief = belief[px, py]
            uncertainty = float(np.clip(1.0 - nearest_belief, 0.0, 1.0))
            if min_dist <= min(float(search_radius), safety_distance):
                return result(
                    "obstacle",
                    "obstacle_within_safety_distance",
                    min_dist,
                    uncertainty,
                )

        if float(search_radius) + EPS < safety_distance:
            return result(
                "unknown",
                "search_radius_below_safety_distance",
                min_dist,
                uncertainty,
            )

        boundary_clearance = min(
            float(x_m) - self.cfg.x_min,
            self.cfg.x_max - float(x_m),
            float(y_m) - self.cfg.y_min,
            self.cfg.y_max - float(y_m),
        )
        if use_layers:
            boundary_clearance = min(
                boundary_clearance,
                float(z_m) - self.cfg.z_min,
                self.cfg.z_max - float(z_m),
            )
        if boundary_clearance + EPS < safety_distance:
            return result(
                "unknown",
                "map_extent_below_safety_distance",
                min_dist,
                uncertainty,
                map_boundary_clearance_m=float(max(boundary_clearance, 0.0)),
            )

        x0 = max(0, int(np.floor((float(x_m) - safety_distance - self.cfg.x_min) / self.cfg.x_resolution)))
        x1 = min(unknown.shape[0], int(np.ceil((float(x_m) + safety_distance - self.cfg.x_min) / self.cfg.x_resolution)))
        y0 = max(0, int(np.floor((float(y_m) - safety_distance - self.cfg.y_min) / self.cfg.y_resolution)))
        y1 = min(unknown.shape[1], int(np.ceil((float(y_m) + safety_distance - self.cfg.y_min) / self.cfg.y_resolution)))
        if use_layers:
            z0 = max(0, int(np.floor((float(z_m) - safety_distance - self.cfg.z_min) / self.cfg.z_resolution)))
            z1 = min(unknown.shape[2], int(np.ceil((float(z_m) + safety_distance - self.cfg.z_min) / self.cfg.z_resolution)))
            unknown_patch = unknown[x0:x1, y0:y1, z0:z1]
        else:
            unknown_patch = unknown[x0:x1, y0:y1]
        if unknown_patch.size == 0:
            return result("unknown", "observed_domain_empty", min_dist, uncertainty)
        unknown_max = float(np.max(unknown_patch))
        if unknown_max > float(max_unknown_mass):
            return result(
                "unknown",
                "unknown_mass_above_threshold",
                min_dist,
                uncertainty,
                observed_domain_unknown_mass_max=unknown_max,
            )
        return result(
            "clear",
            "observed_clear_within_safety_distance",
            min_dist,
            0.0 if not np.isfinite(min_dist) else uncertainty,
            observed_domain_unknown_mass_max=unknown_max,
        )

    def query_trajectory_corridor(
        self,
        waypoints_local_xyz,
        corridor_radius_m: float,
        sample_spacing_m: float,
        speed_m_s: float,
        reaction_time_s: float,
        brake_deceleration_m_s2: float,
        safety_margin_m: float,
        max_unknown_mass: float = 0.5,
    ) -> Dict[str, object]:
        """沿 local-frame 折线检查制动距离内的三态安全走廊。

        每个采样点使用既有三态三维查询；采样间距不大于走廊半径，
        同时将实际查询半径保守扩大半个采样间距，避免在相邻采样球
        中点留下未检查缝隙。轨迹长度不足制动
        距离时，即使已有路段全部 clear 也必须 fail-closed 为 unknown。
        """
        try:
            waypoints = np.asarray(waypoints_local_xyz, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("waypoints_local_xyz 必须是有限数值数组") from exc
        if (
            waypoints.ndim != 2
            or waypoints.shape[0] < 2
            or waypoints.shape[1] != 3
            or not np.all(np.isfinite(waypoints))
        ):
            raise ValueError("waypoints_local_xyz 必须是至少两个有限 local XYZ 点")

        query_parameters = np.asarray(
            [corridor_radius_m, sample_spacing_m, max_unknown_mass],
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(query_parameters))
            or float(corridor_radius_m) <= 0.0
            or float(sample_spacing_m) <= 0.0
            or float(max_unknown_mass) < 0.0
            or float(max_unknown_mass) > 1.0
        ):
            raise ValueError("走廊半径/采样间距必须为有限正数，unknown 阈值必须位于 [0,1]")

        segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        keep = np.concatenate(
            [np.asarray([True]), segment_lengths > EPS],
            axis=0,
        )
        waypoints = waypoints[keep]
        if waypoints.shape[0] < 2:
            raise ValueError("轨迹至少需要两个不同的 local XYZ 点")
        segment_vectors = np.diff(waypoints, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        cumulative_lengths = np.concatenate(
            [np.asarray([0.0], dtype=np.float64), np.cumsum(segment_lengths)]
        )
        trajectory_horizon = float(cumulative_lengths[-1])
        safety_distance = compute_safety_distance_m(
            speed_m_s,
            reaction_time_s,
            brake_deceleration_m_s2,
            safety_margin_m,
        )
        evaluated_horizon = min(trajectory_horizon, safety_distance)
        effective_spacing = min(float(sample_spacing_m), float(corridor_radius_m))
        # 对中心线最近采样点的弧长误差最大是半个 spacing；
        # 用三角不等式做保守膨胀，确保整个连续走廊被覆盖。
        effective_query_radius = float(corridor_radius_m) + 0.5 * effective_spacing
        arc_lengths = np.arange(
            0.0,
            evaluated_horizon,
            effective_spacing,
            dtype=np.float64,
        )
        if arc_lengths.size == 0 or not np.isclose(
            arc_lengths[-1], evaluated_horizon, atol=EPS, rtol=0.0
        ):
            arc_lengths = np.concatenate(
                [arc_lengths, np.asarray([evaluated_horizon], dtype=np.float64)]
            )
        if arc_lengths.size > MAX_TRAJECTORY_QUERY_SAMPLES:
            raise ValueError(
                f"轨迹查询需要 {arc_lengths.size} 个采样点，"
                f"超过上限 {MAX_TRAJECTORY_QUERY_SAMPLES}"
            )

        def interpolate(arc_length: float) -> np.ndarray:
            segment_index = int(
                np.searchsorted(cumulative_lengths[1:], arc_length, side="right")
            )
            segment_index = min(segment_index, segment_lengths.size - 1)
            offset = float(arc_length - cumulative_lengths[segment_index])
            ratio = float(np.clip(offset / segment_lengths[segment_index], 0.0, 1.0))
            return waypoints[segment_index] + ratio * segment_vectors[segment_index]

        minimum_obstacle_distance = float("inf")
        maximum_uncertainty = 0.0
        for sample_index, arc_length in enumerate(arc_lengths.tolist()):
            point = interpolate(float(arc_length))
            proximity = self.query_proximity(
                x_m=float(point[0]),
                y_m=float(point[1]),
                z_m=float(point[2]),
                search_radius=effective_query_radius,
                speed_m_s=0.0,
                reaction_time_s=0.0,
                brake_deceleration_m_s2=1.0,
                safety_margin_m=effective_query_radius,
                max_unknown_mass=float(max_unknown_mass),
            )
            minimum_obstacle_distance = min(
                minimum_obstacle_distance,
                float(proximity["distance"]),
            )
            maximum_uncertainty = max(
                maximum_uncertainty,
                float(proximity["uncertainty"]),
            )
            if proximity["state"] != "clear":
                return {
                    "protocol": TRAJECTORY_CORRIDOR_QUERY_PROTOCOL,
                    "state": str(proximity["state"]),
                    "reason": f"corridor_{proximity['reason']}",
                    "distance": float(proximity["distance"]),
                    "uncertainty": float(proximity["uncertainty"]),
                    "safety_distance_m": float(safety_distance),
                    "trajectory_horizon_m": trajectory_horizon,
                    "evaluated_horizon_m": float(evaluated_horizon),
                    "corridor_radius_m": float(corridor_radius_m),
                    "requested_sample_spacing_m": float(sample_spacing_m),
                    "effective_sample_spacing_m": effective_spacing,
                    "effective_query_radius_m": effective_query_radius,
                    "sample_count": int(arc_lengths.size),
                    "first_risk_sample_index": int(sample_index),
                    "first_risk_arc_length_m": float(arc_length),
                    "first_risk_point_local_m": point.astype(float).tolist(),
                    "is_risky": 1.0,
                }

        common_result = {
            "protocol": TRAJECTORY_CORRIDOR_QUERY_PROTOCOL,
            "distance": float(minimum_obstacle_distance),
            "uncertainty": float(maximum_uncertainty),
            "safety_distance_m": float(safety_distance),
            "trajectory_horizon_m": trajectory_horizon,
            "evaluated_horizon_m": float(evaluated_horizon),
            "corridor_radius_m": float(corridor_radius_m),
            "requested_sample_spacing_m": float(sample_spacing_m),
            "effective_sample_spacing_m": effective_spacing,
            "effective_query_radius_m": effective_query_radius,
            "sample_count": int(arc_lengths.size),
            "first_risk_sample_index": None,
            "first_risk_arc_length_m": None,
            "first_risk_point_local_m": None,
        }
        if trajectory_horizon + EPS < safety_distance:
            return {
                **common_result,
                "state": "unknown",
                "reason": "trajectory_horizon_below_safety_distance",
                "is_risky": 1.0,
            }
        return {
            **common_result,
            "state": "clear",
            "reason": "trajectory_corridor_observed_clear",
            "is_risky": 0.0,
        }


def load_sparse_voxel_npz(path: str) -> np.ndarray:
    """Load sparse voxel npz and restore dense array."""
    data = np.load(path)
    dense = np.zeros(data["shape"], dtype=np.float32)
    coords = data["coords"]
    if coords.shape[0] > 0:
        dense[coords[:, 0], coords[:, 1], coords[:, 2]] = data["features"]
    return dense
