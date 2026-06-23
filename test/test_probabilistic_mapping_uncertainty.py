import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class ProbabilisticMappingUncertaintyTest(unittest.TestCase):
    def test_high_doppler_variance_reduces_belief_and_raises_dem_variance(self):
        from diffusion_consistency_radar.cm.probabilistic_mapping import GridMapConfig, SlidingProbabilisticGridMap

        cfg = GridMapConfig(x_min=0, x_max=4, y_min=0, y_max=1, x_resolution=1, y_resolution=1, z_min=0, z_max=2, z_resolution=1)
        low = SlidingProbabilisticGridMap(cfg)
        high = SlidingProbabilisticGridMap(cfg)

        low_voxel = np.zeros((4, 1, 2, 4), dtype=np.float32)
        high_voxel = np.zeros((4, 1, 2, 4), dtype=np.float32)
        low_voxel[0, 0, 1, 0] = 1.0
        high_voxel[0, 0, 1, 0] = 1.0
        low_voxel[0, 0, 1, 3] = 0.0
        high_voxel[0, 0, 1, 3] = 50.0

        low.update_from_voxel(low_voxel, timestamp=0.1)
        high.update_from_voxel(high_voxel, timestamp=0.1)

        self.assertGreater(float(low.belief[0, 0]), float(high.belief[0, 0]))
        self.assertGreater(float(high.dem_var[0, 0]), float(low.dem_var[0, 0]))

    def test_far_range_reliability_is_lower_and_query_uncertainty_follows_belief(self):
        from diffusion_consistency_radar.cm.probabilistic_mapping import (
            GridMapConfig,
            LazyLocalMapQuery,
            SlidingProbabilisticGridMap,
        )

        cfg = GridMapConfig(x_min=0, x_max=100, y_min=0, y_max=1, x_resolution=10, y_resolution=1, z_min=0, z_max=1, z_resolution=1)
        grid = SlidingProbabilisticGridMap(cfg)
        voxel = np.zeros((10, 1, 1, 4), dtype=np.float32)
        voxel[1, 0, 0, 0] = 1.0
        voxel[9, 0, 0, 0] = 1.0

        reliability = grid.observation_reliability_map(voxel, sensor="radar")
        self.assertGreater(float(reliability[1, 0]), float(reliability[9, 0]))

        grid.update_from_voxel(voxel, timestamp=0.1)
        query = LazyLocalMapQuery(cfg, occ_threshold=0.5)
        query.refresh(grid.snapshot())
        near = query.query_proximity(x_m=15.0, y_m=0.5, search_radius=20)

        self.assertLess(near["uncertainty"], 1.0)
        self.assertAlmostEqual(near["uncertainty"], 1.0 - float(grid.belief[1, 0]), places=5)

    def test_speed_band_adjusts_window_decay_and_far_reliability(self):
        from diffusion_consistency_radar.cm.probabilistic_mapping import GridMapConfig, SlidingProbabilisticGridMap

        slow_cfg = GridMapConfig(
            x_min=0,
            x_max=100,
            y_min=0,
            y_max=1,
            x_resolution=10,
            y_resolution=1,
            z_min=0,
            z_max=1,
            z_resolution=1,
            window_size=12,
            decay_rate=0.12,
            speed_m_s=35.0,
        )
        fast_cfg = GridMapConfig(
            x_min=0,
            x_max=100,
            y_min=0,
            y_max=1,
            x_resolution=10,
            y_resolution=1,
            z_min=0,
            z_max=1,
            z_resolution=1,
            window_size=12,
            decay_rate=0.12,
            speed_m_s=70.0,
        )
        self.assertGreater(slow_cfg.window_size, fast_cfg.window_size)
        self.assertLess(slow_cfg.decay_rate, fast_cfg.decay_rate)

        voxel = np.zeros((10, 1, 1, 4), dtype=np.float32)
        voxel[9, 0, 0, 0] = 1.0
        slow_rel = SlidingProbabilisticGridMap(slow_cfg).observation_reliability_map(voxel)
        fast_rel = SlidingProbabilisticGridMap(fast_cfg).observation_reliability_map(voxel)
        self.assertGreater(float(slow_rel[9, 0]), float(fast_rel[9, 0]))

    def test_odom_covariance_lowers_belief(self):
        from diffusion_consistency_radar.cm.probabilistic_mapping import GridMapConfig, SlidingProbabilisticGridMap

        cfg = GridMapConfig(x_min=0, x_max=4, y_min=0, y_max=1, x_resolution=1, y_resolution=1, z_min=0, z_max=1, z_resolution=1)
        clean = SlidingProbabilisticGridMap(cfg)
        noisy = SlidingProbabilisticGridMap(cfg)
        voxel = np.zeros((4, 1, 1, 4), dtype=np.float32)
        voxel[0, 0, 0, 0] = 1.0

        clean.update_from_voxel(voxel, timestamp=0.1)
        noisy.update_from_voxel(voxel, timestamp=0.1, odom_cov=np.eye(3, dtype=np.float32) * 4.0)

        self.assertGreater(float(clean.belief[0, 0]), float(noisy.belief[0, 0]))

    def test_model_uncertainty_and_calibration_confidence_lower_reliability(self):
        from diffusion_consistency_radar.cm.probabilistic_mapping import GridMapConfig, SlidingProbabilisticGridMap

        cfg = GridMapConfig(x_min=0, x_max=4, y_min=0, y_max=1, x_resolution=1, y_resolution=1, z_min=0, z_max=1, z_resolution=1)
        grid = SlidingProbabilisticGridMap(cfg)
        voxel = np.zeros((4, 1, 1, 4), dtype=np.float32)
        voxel[0, 0, 0, 0] = 1.0

        confident = np.zeros((4, 1), dtype=np.float32)
        uncertain = np.full((4, 1), 8.0, dtype=np.float32)
        rel_confident = grid.observation_reliability_map(voxel, model_uncertainty=confident, calib_confidence=1.0)
        rel_uncertain = grid.observation_reliability_map(voxel, model_uncertainty=uncertain, calib_confidence=0.5)

        self.assertGreater(float(rel_confident[0, 0]), float(rel_uncertain[0, 0]))

        clean = SlidingProbabilisticGridMap(cfg)
        weak = SlidingProbabilisticGridMap(cfg)
        clean.update_from_voxel(voxel, timestamp=0.1, model_uncertainty=confident, calib_confidence=1.0)
        weak.update_from_voxel(voxel, timestamp=0.1, model_uncertainty=uncertain, calib_confidence=0.5)
        self.assertGreater(float(clean.belief[0, 0]), float(weak.belief[0, 0]))

    def test_streaming_helpers_skip_and_load_uncertainty_files(self):
        from diffusion_consistency_radar.scripts.streaming_map_update import (
            find_uncertainty_file,
            list_voxel_files,
            load_model_uncertainty,
        )

        with tempfile.TemporaryDirectory() as tmp:
            np.save(os.path.join(tmp, "000001_voxel.npy"), np.zeros((4, 2, 3, 3), dtype=np.float32))
            np.save(os.path.join(tmp, "000001_uncertainty.npy"), np.ones((1, 2, 3, 3), dtype=np.float32))

            self.assertEqual(list_voxel_files(tmp), ["000001_voxel.npy"])
            unc_path = find_uncertainty_file(tmp, "000001_voxel.npy")
            loaded = load_model_uncertainty(unc_path)

            self.assertEqual(tuple(loaded.shape), (3, 3, 2))
            self.assertAlmostEqual(float(loaded.mean()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
