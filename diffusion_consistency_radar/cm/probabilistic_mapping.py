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

EPS = 1e-6


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

    def __post_init__(self) -> None:
        static_decay = float(self.decay_rate)
        dynamic_decay = float(self.dynamic_decay_rate)
        if not np.isfinite(static_decay) or static_decay < 0.0:
            raise ValueError("decay_rate 必须是有限非负数")
        if not np.isfinite(dynamic_decay) or dynamic_decay < 0.0:
            raise ValueError("dynamic_decay_rate 必须是有限非负数")
        if not np.isfinite(float(self.speed_m_s)) or float(self.speed_m_s) <= 0.0:
            raise ValueError("speed_m_s 必须是有限正数")
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
        self._has_voxel_update = False
        self.history: Deque[Dict[str, np.ndarray]] = deque(maxlen=self.cfg.window_size)
        self.ds_fuser = DSEvidenceFusion()

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
    def _validated_pose(T_local_body: Optional[np.ndarray]) -> np.ndarray:
        """校验 body→local 的齐次刚体变换；缺省为兼容用单位变换。"""
        if T_local_body is None:
            return np.eye(4, dtype=np.float32)
        transform = np.asarray(T_local_body, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("T_local_body 必须是有限的 4x4 齐次刚体矩阵")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValueError("T_local_body 最后一行必须为 [0,0,0,1]")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("T_local_body 旋转部分必须正交")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError("T_local_body 旋转部分必须是 det=+1 的右手旋转")
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
    ) -> np.ndarray:
        """把 observed mask 解析为三维层；二维 mask 显式广播到所有高度。"""
        if observed_mask is None:
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
            observed = observed | (occupied_layers > 0.0)
        return observed.astype(np.float32)

    def _body_to_local_mapping(
        self,
        voxel_shape: Tuple[int, int, int],
        T_local_body: np.ndarray,
        source_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算已观测源体素中心到 local map 体素的前向映射。"""
        if tuple(voxel_shape) != (self.nx, self.ny, self.nz):
            raise ValueError(
                f"voxel XYZ shape {voxel_shape} != map shape "
                f"{(self.nx, self.ny, self.nz)}"
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
        x = self.cfg.x_min + (source_x.astype(np.float64) + 0.5) * self.cfg.x_resolution
        y = self.cfg.y_min + (source_y.astype(np.float64) + 0.5) * self.cfg.y_resolution
        z = self.cfg.z_min + (source_z.astype(np.float64) + 0.5) * self.cfg.z_resolution
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
    ) -> np.ndarray:
        base = self.cfg.infrared_reliability if sensor == "infrared" else self.cfg.radar_reliability
        nx = voxel_xyzc.shape[0]
        x_centers = self.cfg.x_min + (np.arange(nx, dtype=np.float32) + 0.5) * self.cfg.x_resolution
        max_range = max(self.cfg.x_max - self.cfg.x_min, EPS)
        speed_scale = float(np.clip(self.cfg.speed_m_s / 50.0, 0.7, 1.4))
        range_conf = np.clip(
            1.0 - 0.65 * speed_scale * ((x_centers - self.cfg.x_min) / max_range),
            0.18,
            1.0,
        )[:, np.newaxis]
        variance_conf = 1.0 / (1.0 + self._doppler_variance_bev(voxel_xyzc) / 10.0)
        model_unc = self._uncertainty_to_bev(model_uncertainty, voxel_xyzc.shape[:2])
        model_conf = 1.0 / (1.0 + model_unc / 5.0)
        odom_conf = self._odom_confidence(odom_cov)
        calib_conf = float(np.clip(calib_confidence, 0.02, 1.0))
        return np.clip(base * range_conf * variance_conf * model_conf * odom_conf * calib_conf, 0.02, 1.0).astype(np.float32)

    def _update_dem_from_voxel(self, voxel_xyzc: np.ndarray, model_uncertainty: Optional[np.ndarray] = None) -> None:
        occ3d = np.clip(voxel_xyzc[..., 0], 0.0, 1.0)
        z_bins = occ3d.shape[2]
        z_values = self.cfg.z_min + (np.arange(z_bins, dtype=np.float32) + 0.5) * self.cfg.z_resolution

        occ_sum = occ3d.sum(axis=2)
        valid = occ_sum > 0.1
        if not np.any(valid):
            return

        z_mean = (occ3d * z_values[np.newaxis, np.newaxis, :]).sum(axis=2) / np.maximum(occ_sum, EPS)
        z_second = (occ3d * (z_values[np.newaxis, np.newaxis, :] ** 2)).sum(axis=2) / np.maximum(occ_sum, EPS)
        z_var = np.maximum(0.0, z_second - z_mean ** 2)
        z_var = z_var + self._doppler_variance_bev(voxel_xyzc)
        z_var = z_var + self._uncertainty_to_bev(model_uncertainty, voxel_xyzc.shape[:2])

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
        dynamic_probability: Optional[np.ndarray] = None,
        dynamic_observed_mask: Optional[np.ndarray] = None,
    ) -> None:
        """把 body 系 `(X,Y,Z,C)` 体素变换到 local 系后更新分层与 BEV 地图。"""
        voxel_xyzc = np.asarray(voxel_xyzc, dtype=np.float32)
        if voxel_xyzc.ndim != 4 or voxel_xyzc.shape[-1] < 1:
            raise ValueError(f"voxel_xyzc 必须是 (X,Y,Z,C)，当前为 {voxel_xyzc.shape}")
        if tuple(voxel_xyzc.shape[:3]) != (self.nx, self.ny, self.nz):
            raise ValueError(
                f"voxel XYZ shape {voxel_xyzc.shape[:3]} != map shape "
                f"{(self.nx, self.ny, self.nz)}"
            )
        if not np.all(np.isfinite(voxel_xyzc)):
            raise ValueError("voxel_xyzc 必须全部为有限数")

        # 所有外部输入必须在时间衰减或证据融合前完成校验，失败时地图无副作用。
        validated_timestamp = self._validated_timestamp(timestamp)
        validated_pose = self._validated_pose(T_local_body)
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
        source_reliability_bev = self.observation_reliability_map(
            voxel_xyzc,
            sensor=sensor,
            odom_cov=odom_cov,
            model_uncertainty=model_uncertainty,
            calib_confidence=calib_confidence,
        )
        source_reliability_layers = (
            source_reliability_bev[..., np.newaxis] * source_observed_layers
        )
        warped_voxel, _warped_observed, source_indices, destination_indices = (
            self._warp_voxel_to_local(
                voxel_xyzc,
                mapping_mask,
                validated_pose,
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
        source_uncertainty_bev = self._uncertainty_to_bev(
            model_uncertainty,
            voxel_xyzc.shape[:2],
        )
        warped_uncertainty_layers = self._scatter_layer_max(
            np.broadcast_to(
                source_uncertainty_bev[..., np.newaxis],
                voxel_xyzc.shape[:3],
            ),
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
            model_uncertainty=np.max(warped_uncertainty_layers, axis=2),
        )
        self.last_timestamp = validated_timestamp
        self.last_T_local_body = validated_pose.copy()
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
            "last_timestamp": np.asarray(self.last_timestamp, dtype=np.float64),
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
        self._occupied_xyz_m: Optional[np.ndarray] = None
        self._belief_layers: Optional[np.ndarray] = None

    def refresh(self, map_snapshot: Dict[str, np.ndarray]) -> None:
        occ = map_snapshot["occ_prob"]
        belief = map_snapshot["belief"]

        idx = np.argwhere(occ >= self.occ_threshold)
        if idx.shape[0] == 0:
            self._occupied_xy_m = np.zeros((0, 2), dtype=np.float32)
        else:
            x_m = self.cfg.x_min + (idx[:, 0].astype(np.float32) + 0.5) * self.cfg.x_resolution
            y_m = self.cfg.y_min + (idx[:, 1].astype(np.float32) + 0.5) * self.cfg.y_resolution
            self._occupied_xy_m = np.stack([x_m, y_m], axis=1)
        self._belief_map = belief

        # 新快照优先缓存三维层；缺少 layers 的旧快照仍保持二维查询能力。
        occ_layers = map_snapshot.get("occ_prob_layers")
        belief_layers = map_snapshot.get("belief_layers")
        if occ_layers is None or belief_layers is None:
            self._occupied_xyz_m = None
            self._belief_layers = None
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

    def query_proximity(
        self,
        x_m: float,
        y_m: float,
        search_radius: float = 15.0,
        z_m: Optional[float] = None,
    ) -> Dict[str, float]:
        """查询最近障碍；提供 `z_m` 且快照含 layers 时执行三维距离查询。"""
        values = [float(x_m), float(y_m), float(search_radius)]
        if z_m is not None:
            values.append(float(z_m))
        if not np.all(np.isfinite(values)) or float(search_radius) < 0.0:
            raise ValueError("查询坐标和 search_radius 必须是有限数，且半径非负")

        use_layers = (
            z_m is not None
            and self._occupied_xyz_m is not None
            and self._belief_layers is not None
        )
        occupied = self._occupied_xyz_m if use_layers else self._occupied_xy_m
        belief = self._belief_layers if use_layers else self._belief_map
        if occupied is None or belief is None:
            return {"distance": float("inf"), "uncertainty": 1.0, "is_risky": 0.0}

        if occupied.shape[0] == 0:
            return {"distance": float("inf"), "uncertainty": 1.0, "is_risky": 0.0}

        query_values = [x_m, y_m, z_m] if use_layers else [x_m, y_m]
        q = np.asarray(query_values, dtype=np.float32)
        dists = np.linalg.norm(occupied - q[np.newaxis, :], axis=1)
        min_idx = int(np.argmin(dists))
        min_dist = float(dists[min_idx])

        if min_dist > search_radius:
            return {"distance": min_dist, "uncertainty": 1.0, "is_risky": 0.0}

        px = int(np.clip((occupied[min_idx, 0] - self.cfg.x_min) / self.cfg.x_resolution, 0, belief.shape[0] - 1))
        py = int(np.clip((occupied[min_idx, 1] - self.cfg.y_min) / self.cfg.y_resolution, 0, belief.shape[1] - 1))
        if use_layers:
            pz = int(np.clip((occupied[min_idx, 2] - self.cfg.z_min) / self.cfg.z_resolution, 0, belief.shape[2] - 1))
            nearest_belief = belief[px, py, pz]
        else:
            nearest_belief = belief[px, py]
        uncertainty = float(np.clip(1.0 - nearest_belief, 0.0, 1.0))
        # TODO: 将固定风险阈值替换为与飞行速度和反应时长相关的动态安全距离模型。

        return {
            "distance": min_dist,
            "uncertainty": uncertainty,
            "is_risky": float(min_dist < 5.0 and uncertainty < 0.7),
        }


def load_sparse_voxel_npz(path: str) -> np.ndarray:
    """Load sparse voxel npz and restore dense array."""
    data = np.load(path)
    dense = np.zeros(data["shape"], dtype=np.float32)
    coords = data["coords"]
    if coords.shape[0] > 0:
        dense[coords[:, 0], coords[:, 1], coords[:, 2]] = data["features"]
    return dense
