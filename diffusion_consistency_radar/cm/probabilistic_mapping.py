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
    prior_reliability: float = 0.90
    radar_reliability: float = 0.75
    infrared_reliability: float = 0.65

    @property
    def shape_xy(self) -> Tuple[int, int]:
        width = int(round((self.x_max - self.x_min) / self.x_resolution))
        height = int(round((self.y_max - self.y_min) / self.y_resolution))
        return width, height


class DSEvidenceFusion:
    """Dempster-Shafer fusion for occupancy, free, and unknown masses."""

    @staticmethod
    def prob_to_mass(occ_prob: np.ndarray, reliability: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = np.clip(occ_prob, 0.0, 1.0)
        r = float(np.clip(reliability, 0.0, 1.0))
        m_occ = r * p
        m_free = r * (1.0 - p)
        m_unknown = np.full_like(p, 1.0 - r)
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

        self.occ_prob = np.full((self.nx, self.ny), 0.5, dtype=np.float32)
        self.belief = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.plausibility = np.ones((self.nx, self.ny), dtype=np.float32)

        self.dem_mean = np.full((self.nx, self.ny), np.nan, dtype=np.float32)
        self.dem_var = np.full((self.nx, self.ny), np.nan, dtype=np.float32)

        self.last_timestamp = 0.0
        self.history: Deque[Dict[str, np.ndarray]] = deque(maxlen=self.cfg.window_size)
        self.ds_fuser = DSEvidenceFusion()

    def _time_decay(self, timestamp: float) -> None:
        dt = max(0.0, float(timestamp) - float(self.last_timestamp))
        if dt <= 0.0:
            return
        decay = float(np.exp(-self.cfg.decay_rate * dt))
        self.occ_prob = 0.5 + decay * (self.occ_prob - 0.5)

    @staticmethod
    def _odom_confidence(odom_cov: Optional[np.ndarray]) -> float:
        if odom_cov is None:
            return 1.0
        cov = np.asarray(odom_cov, dtype=np.float32)
        trace = float(np.trace(cov)) if cov.ndim == 2 else float(np.sum(cov))
        return float(np.exp(-0.35 * max(0.0, trace)))

    def _fuse_occ(self, obs_occ: np.ndarray, obs_reliability: float) -> None:
        prior_mass = self.ds_fuser.prob_to_mass(self.occ_prob, self.cfg.prior_reliability)
        obs_mass = self.ds_fuser.prob_to_mass(obs_occ, obs_reliability)
        m_occ, m_free, _, ignorance = self.ds_fuser.fuse_two(prior_mass, obs_mass)

        self.occ_prob = m_occ / np.maximum(m_occ + m_free, EPS)
        self.belief = np.clip(m_occ, 0.0, 1.0)
        self.plausibility = np.clip(1.0 - m_free, 0.0, 1.0)

        self.history.append(
            {
                "occ_prob": self.occ_prob.copy(),
                "belief": self.belief.copy(),
                "ignorance": ignorance.astype(np.float32),
            }
        )

    def _update_dem_from_voxel(self, voxel_xyzc: np.ndarray) -> None:
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
    ) -> None:
        """Update occupancy map from 3D voxel with shape (X, Y, Z, C)."""
        self._time_decay(timestamp)

        occ3d = np.clip(voxel_xyzc[..., 0], 0.0, 1.0)
        obs_occ = occ3d.max(axis=2)

        odom_conf = self._odom_confidence(odom_cov)
        obs_occ = 0.5 + odom_conf * (obs_occ - 0.5)

        reliability = self.cfg.infrared_reliability if sensor == "infrared" else self.cfg.radar_reliability

        self._fuse_occ(obs_occ=obs_occ, obs_reliability=reliability)
        self._update_dem_from_voxel(voxel_xyzc)
        self.last_timestamp = float(timestamp)

    def update_from_ir_bev(self, bev_xy: np.ndarray, timestamp: float) -> None:
        """Update occupancy map from infrared BEV with shape (X, Y) or (X, Y, C)."""
        self._time_decay(timestamp)
        bev = np.asarray(bev_xy, dtype=np.float32)
        if bev.ndim == 3:
            bev = bev[..., 0]
        bev = np.clip(bev, 0.0, 1.0)

        if bev.shape != self.occ_prob.shape:
            raise ValueError(f"Infrared BEV shape {bev.shape} != map shape {self.occ_prob.shape}")

        self._fuse_occ(obs_occ=bev, obs_reliability=self.cfg.infrared_reliability)
        self.last_timestamp = float(timestamp)

    def fuse_with_prior_dem(self, prior_dem: np.ndarray, prior_confidence: float = 0.6) -> None:
        """Fuse a prior DEM with the online DEM estimate."""
        dem = np.asarray(prior_dem, dtype=np.float32)
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
        return {
            "occ_prob": self.occ_prob.copy(),
            "belief": self.belief.copy(),
            "plausibility": self.plausibility.copy(),
            "dem_mean": self.dem_mean.copy(),
            "dem_var": self.dem_var.copy(),
        }


class LazyLocalMapQuery:
    """Lightweight local lazy proximity query helper."""

    def __init__(self, config: GridMapConfig, occ_threshold: float = 0.55):
        self.cfg = config
        self.occ_threshold = float(occ_threshold)
        self._occupied_xy_m: Optional[np.ndarray] = None
        self._belief_map: Optional[np.ndarray] = None

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

    def query_proximity(self, x_m: float, y_m: float, search_radius: float = 15.0) -> Dict[str, float]:
        if self._occupied_xy_m is None or self._belief_map is None:
            return {"distance": float("inf"), "uncertainty": 1.0, "is_risky": 0.0}

        if self._occupied_xy_m.shape[0] == 0:
            return {"distance": float("inf"), "uncertainty": 1.0, "is_risky": 0.0}

        q = np.array([x_m, y_m], dtype=np.float32)
        dists = np.linalg.norm(self._occupied_xy_m - q[np.newaxis, :], axis=1)
        min_idx = int(np.argmin(dists))
        min_dist = float(dists[min_idx])

        if min_dist > search_radius:
            return {"distance": min_dist, "uncertainty": 1.0, "is_risky": 0.0}

        px = int(np.clip((self._occupied_xy_m[min_idx, 0] - self.cfg.x_min) / self.cfg.x_resolution, 0, self._belief_map.shape[0] - 1))
        py = int(np.clip((self._occupied_xy_m[min_idx, 1] - self.cfg.y_min) / self.cfg.y_resolution, 0, self._belief_map.shape[1] - 1))
        uncertainty = float(np.clip(1.0 - self._belief_map[px, py], 0.0, 1.0))

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
