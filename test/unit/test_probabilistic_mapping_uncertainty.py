import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class ProbabilisticMappingUncertaintyTest(unittest.TestCase):
    @staticmethod
    def _small_pose_map():
        from diffusion_consistency_radar.cm.probabilistic_mapping import (
            GridMapConfig,
            SlidingProbabilisticGridMap,
        )

        cfg = GridMapConfig(
            x_min=0,
            x_max=4,
            y_min=0,
            y_max=2,
            x_resolution=1,
            y_resolution=1,
            z_min=0,
            z_max=2,
            z_resolution=1,
        )
        return SlidingProbabilisticGridMap(cfg)

    def test_body_to_local_translation_aligns_static_obstacle_across_frames(self):
        """机体前移后，同一静态障碍必须继续融合到同一个 local 体素。"""
        grid = self._small_pose_map()
        first = np.zeros((4, 2, 2, 4), dtype=np.float32)
        second = np.zeros_like(first)
        first[2, 0, 0, 0] = 1.0
        second[1, 0, 0, 0] = 1.0

        identity = np.eye(4, dtype=np.float32)
        translated = identity.copy()
        translated[0, 3] = 1.0
        grid.update_from_voxel(first, timestamp=1.0, T_local_body=identity)
        first_probability = float(grid.snapshot()["occ_prob_layers"][2, 0, 0])
        grid.update_from_voxel(second, timestamp=2.0, T_local_body=translated)
        snapshot = grid.snapshot()

        self.assertGreater(float(snapshot["occ_prob_layers"][2, 0, 0]), first_probability)
        self.assertAlmostEqual(float(snapshot["occ_prob_layers"][1, 0, 0]), 0.5, places=6)
        np.testing.assert_allclose(snapshot["last_T_local_body"], translated)
        self.assertEqual(float(snapshot["last_timestamp"]), 2.0)

    def test_pose_rotation_and_vertical_translation_keep_physical_height_layers(self):
        """刚体旋转和 Z 平移必须作用于三维层，不能只变换 BEV 索引。"""
        grid = self._small_pose_map()
        voxel = np.zeros((4, 2, 2, 4), dtype=np.float32)
        voxel[0, 0, 0, 0] = 1.0  # body 中心坐标为 (0.5, 0.5, 0.5)
        transform = np.asarray(
            [
                [0.0, -1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        grid.update_from_voxel(voxel, timestamp=1.0, T_local_body=transform)
        layers = grid.snapshot()["occ_prob_layers"]

        # 变换后中心为 (0.5, 0.5, 1.5)，落在相同 XY、第二个高度层。
        self.assertGreater(float(layers[0, 0, 1]), 0.5)
        self.assertAlmostEqual(float(layers[0, 0, 0]), 0.5, places=6)

    def test_invalid_pose_or_nonincreasing_timestamp_fails_before_state_change(self):
        grid = self._small_pose_map()
        voxel = np.zeros((4, 2, 2, 4), dtype=np.float32)
        voxel[0, 0, 0, 0] = 1.0
        grid.update_from_voxel(voxel, timestamp=1.0, T_local_body=np.eye(4))
        before = grid.snapshot()

        bad_pose = np.eye(4, dtype=np.float32)
        bad_pose[0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "T_local_body"):
            grid.update_from_voxel(voxel, timestamp=2.0, T_local_body=bad_pose)
        with self.assertRaisesRegex(ValueError, "timestamp"):
            grid.update_from_voxel(voxel, timestamp=1.0, T_local_body=np.eye(4))

        after = grid.snapshot()
        np.testing.assert_array_equal(after["occ_prob_layers"], before["occ_prob_layers"])
        np.testing.assert_array_equal(after["occ_prob"], before["occ_prob"])
        self.assertEqual(float(after["last_timestamp"]), 1.0)

    def test_three_dimensional_observed_mask_preserves_unknown_height_layers(self):
        grid = self._small_pose_map()
        voxel = np.zeros((4, 2, 2, 4), dtype=np.float32)
        observed = np.zeros((4, 2, 2), dtype=np.float32)
        observed[0, 0, 0] = 1.0

        grid.update_from_voxel(voxel, timestamp=1.0, observed_mask=observed)
        snapshot = grid.snapshot()

        self.assertLess(float(snapshot["occ_prob_layers"][0, 0, 0]), 0.5)
        self.assertAlmostEqual(float(snapshot["occ_prob_layers"][0, 0, 1]), 0.5, places=6)
        self.assertGreater(
            float(snapshot["unknown_mass_layers"][0, 0, 1]),
            float(snapshot["unknown_mass_layers"][0, 0, 0]),
        )

    def test_pose_csv_requires_exact_frame_coverage_and_builds_forward_transform(self):
        from diffusion_consistency_radar.scripts.streaming_map_update import load_pose_table

        with tempfile.TemporaryDirectory() as tmp:
            pose_path = os.path.join(tmp, "poses.csv")
            with open(pose_path, "w", encoding="utf-8", newline="") as handle:
                handle.write("frame,timestamp,tx,ty,tz,qx,qy,qz,qw\n")
                handle.write("000001,10.0,1,2,3,0,0,0,1\n")
                handle.write("000002,10.1,2,2,3,0,0,0.70710678,0.70710678\n")

            poses = load_pose_table(
                pose_path,
                ["000001_voxel.npy", "000002_voxel.npy"],
            )
            self.assertEqual(list(poses), ["000001", "000002"])
            self.assertAlmostEqual(float(poses["000002"]["timestamp"]), 10.1)
            np.testing.assert_allclose(
                poses["000002"]["T_local_body"][:2, :2],
                np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32),
                atol=1e-6,
            )

            with self.assertRaisesRegex(ValueError, "帧覆盖"):
                load_pose_table(pose_path, ["000001_voxel.npy", "000003_voxel.npy"])

    def test_unknown_cells_do_not_become_free_without_observed_mask(self):
        from diffusion_consistency_radar.cm.probabilistic_mapping import (
            GridMapConfig,
            SlidingProbabilisticGridMap,
        )

        cfg = GridMapConfig(
            x_min=0,
            x_max=2,
            y_min=0,
            y_max=1,
            x_resolution=1,
            y_resolution=1,
            z_min=0,
            z_max=1,
            z_resolution=1,
        )
        grid = SlidingProbabilisticGridMap(cfg)
        voxel = np.zeros((2, 1, 1, 4), dtype=np.float32)
        voxel[0, 0, 0, 0] = 1.0

        grid.update_from_voxel(voxel, timestamp=0.1)

        self.assertGreater(float(grid.belief[0, 0]), 0.0)
        self.assertAlmostEqual(float(grid.occ_prob[1, 0]), 0.5, places=6)
        self.assertAlmostEqual(float(grid.belief[1, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(grid.unknown_mass[1, 0]), 1.0, places=6)

    def test_time_decay_moves_old_layer_evidence_toward_unknown(self):
        grid = self._small_pose_map()
        first = np.zeros((4, 2, 2, 4), dtype=np.float32)
        second = np.zeros_like(first)
        first[0, 0, 0, 0] = 1.0
        second[1, 0, 0, 0] = 1.0
        grid.update_from_voxel(first, timestamp=1.0)
        before = grid.snapshot()

        grid.update_from_voxel(second, timestamp=2.0)
        after = grid.snapshot()

        self.assertLess(
            float(after["belief_layers"][0, 0, 0]),
            float(before["belief_layers"][0, 0, 0]),
        )
        self.assertGreater(
            float(after["unknown_mass_layers"][0, 0, 0]),
            float(before["unknown_mass_layers"][0, 0, 0]),
        )

    def test_observed_free_mask_allows_free_evidence_only_where_declared(self):
        from diffusion_consistency_radar.cm.probabilistic_mapping import (
            GridMapConfig,
            SlidingProbabilisticGridMap,
        )

        cfg = GridMapConfig(
            x_min=0,
            x_max=2,
            y_min=0,
            y_max=1,
            x_resolution=1,
            y_resolution=1,
            z_min=0,
            z_max=1,
            z_resolution=1,
        )
        grid = SlidingProbabilisticGridMap(cfg)
        voxel = np.zeros((2, 1, 1, 4), dtype=np.float32)
        observed_mask = np.asarray([[1.0], [0.0]], dtype=np.float32)

        grid.update_from_voxel(
            voxel,
            timestamp=0.1,
            observed_mask=observed_mask,
        )

        self.assertLess(float(grid.occ_prob[0, 0]), 0.5)
        self.assertAlmostEqual(float(grid.occ_prob[1, 0]), 0.5, places=6)
        self.assertGreater(float(grid.unknown_mass[1, 0]), 0.0)

    def test_observed_mask_shape_mismatch_fails_before_map_update(self):
        from diffusion_consistency_radar.cm.probabilistic_mapping import (
            GridMapConfig,
            SlidingProbabilisticGridMap,
        )

        cfg = GridMapConfig(
            x_min=0,
            x_max=2,
            y_min=0,
            y_max=1,
            x_resolution=1,
            y_resolution=1,
            z_min=0,
            z_max=1,
            z_resolution=1,
        )
        grid = SlidingProbabilisticGridMap(cfg)
        voxel = np.zeros((2, 1, 1, 4), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "observed_mask"):
            grid.update_from_voxel(
                voxel,
                timestamp=0.1,
                observed_mask=np.zeros((3, 1), dtype=np.float32),
            )

        self.assertTrue(np.allclose(grid.occ_prob, 0.5))

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

    def test_streaming_helpers_load_observed_mask_without_confusing_voxels(self):
        from diffusion_consistency_radar.scripts.streaming_map_update import (
            find_observed_mask_file,
            list_voxel_files,
            load_observed_mask,
        )

        with tempfile.TemporaryDirectory() as tmp:
            np.save(
                os.path.join(tmp, "000001_voxel.npy"),
                np.zeros((2, 1, 1, 4), dtype=np.float32),
            )
            np.save(
                os.path.join(tmp, "000001_observed_mask.npy"),
                np.asarray([[1.0], [0.0]], dtype=np.float32),
            )
            with open(os.path.join(tmp, "000001_observed_mask.npz"), "wb"):
                pass
            self.assertEqual(list_voxel_files(tmp), ["000001_voxel.npy"])
            mask_path = find_observed_mask_file(tmp, "000001_voxel.npy")
            self.assertTrue(mask_path.endswith("000001_observed_mask.npy"))
            mask = load_observed_mask(mask_path, voxel_shape=(2, 1, 1))
            self.assertEqual(mask.shape, (2, 1))
            self.assertEqual(mask.dtype, np.float32)
            self.assertEqual(float(mask[0, 0]), 1.0)

    def test_streaming_observed_mask_shape_mismatch_fails(self):
        from diffusion_consistency_radar.scripts.streaming_map_update import (
            load_observed_mask,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.npy")
            np.save(path, np.zeros((3, 1), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "observed mask"):
                load_observed_mask(path, voxel_shape=(2, 1, 1))

    def test_streaming_sparse_observed_mask_npz_uses_voxel_protocol(self):
        from diffusion_consistency_radar.scripts.streaming_map_update import (
            load_observed_mask,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "000001_observed_mask.npz")
            np.savez_compressed(
                path,
                coords=np.asarray([[0, 0, 0]], dtype=np.int64),
                features=np.asarray([1.0], dtype=np.float32),
                shape=np.asarray([2, 1, 1], dtype=np.int64),
            )
            mask = load_observed_mask(path, voxel_shape=(2, 1, 1))
            np.testing.assert_array_equal(mask, np.asarray([[1.0], [0.0]], dtype=np.float32))

    def test_streaming_pose_mode_outputs_layers_and_auditable_pose_metadata(self):
        from diffusion_consistency_radar.scripts.streaming_map_update import main

        with tempfile.TemporaryDirectory() as tmp:
            voxel_dir = os.path.join(tmp, "voxels")
            output_dir = os.path.join(tmp, "map")
            os.makedirs(voxel_dir)
            first = np.zeros((4, 2, 2, 4), dtype=np.float32)
            second = np.zeros_like(first)
            first[2, 0, 0, 0] = 1.0
            second[1, 0, 0, 0] = 1.0
            np.save(os.path.join(voxel_dir, "000001_voxel.npy"), first)
            np.save(os.path.join(voxel_dir, "000002_voxel.npy"), second)
            pose_path = os.path.join(tmp, "poses.csv")
            with open(pose_path, "w", encoding="utf-8", newline="") as handle:
                handle.write("frame,timestamp,tx,ty,tz,qx,qy,qz,qw\n")
                handle.write("000001,10.0,0,0,0,0,0,0,1\n")
                handle.write("000002,10.1,1,0,0,0,0,0,1\n")

            argv = [
                "streaming_map_update.py",
                "--radar_voxel_dir", voxel_dir,
                "--pose_file", pose_path,
                "--output_dir", output_dir,
                "--radar_voxel_layout", "xyzc",
                "--pc_range", "0", "0", "0", "4", "2", "2",
                "--save_every", "1",
            ]
            with mock.patch.object(sys, "argv", argv):
                main()

            with np.load(os.path.join(output_dir, "map_final.npz")) as result:
                self.assertEqual(result["occ_prob"].shape, (4, 2))
                self.assertEqual(result["occ_prob_layers"].shape, (4, 2, 2))
                self.assertGreater(float(result["occ_prob_layers"][2, 0, 0]), 0.5)
                np.testing.assert_allclose(result["last_T_local_body"][:3, 3], [1, 0, 0])
                self.assertAlmostEqual(float(result["last_timestamp"]), 10.1)
                self.assertEqual(str(result["map_frame"]), "local")
                self.assertEqual(str(result["pose_mode"]), "body_to_local_csv")
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "map_run.json")))

    def test_streaming_pose_missing_frame_fails_before_output_directory_creation(self):
        from diffusion_consistency_radar.scripts.streaming_map_update import main

        with tempfile.TemporaryDirectory() as tmp:
            voxel_dir = os.path.join(tmp, "voxels")
            output_dir = os.path.join(tmp, "map")
            os.makedirs(voxel_dir)
            voxel = np.zeros((2, 1, 1, 4), dtype=np.float32)
            np.save(os.path.join(voxel_dir, "000001_voxel.npy"), voxel)
            np.save(os.path.join(voxel_dir, "000002_voxel.npy"), voxel)
            pose_path = os.path.join(tmp, "poses.csv")
            with open(pose_path, "w", encoding="utf-8", newline="") as handle:
                handle.write("frame,timestamp,tx,ty,tz,qx,qy,qz,qw\n")
                handle.write("000001,10.0,0,0,0,0,0,0,1\n")

            argv = [
                "streaming_map_update.py",
                "--radar_voxel_dir", voxel_dir,
                "--pose_file", pose_path,
                "--output_dir", output_dir,
                "--pc_range", "0", "0", "0", "2", "1", "1",
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "帧覆盖"):
                    main()
            self.assertFalse(os.path.exists(output_dir))

    def test_streaming_observed_mask_can_preserve_height_layers(self):
        from diffusion_consistency_radar.scripts.streaming_map_update import load_observed_mask

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mask.npy")
            expected = np.zeros((2, 1, 2), dtype=np.float32)
            expected[0, 0, 1] = 1.0
            np.save(path, expected)
            actual = load_observed_mask(
                path,
                voxel_shape=(2, 1, 2),
                preserve_height=True,
            )
            np.testing.assert_array_equal(actual, expected)

    def test_voxel_layout_auto_rejects_ambiguous_shape(self):
        from diffusion_consistency_radar.scripts.streaming_map_update import (
            load_voxel,
            to_xyzc,
        )

        ambiguous = np.zeros((4, 2, 2, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "layout 歧义"):
            to_xyzc(ambiguous)
        self.assertEqual(to_xyzc(ambiguous, layout="xyzc").shape, (4, 2, 2, 4))
        self.assertEqual(to_xyzc(ambiguous, layout="czxy").shape, (2, 4, 2, 4))
        with tempfile.TemporaryDirectory() as tmp:
            batch_path = os.path.join(tmp, "batch.npy")
            np.save(batch_path, np.zeros((2, 4, 2, 4, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "恰好一个样本"):
                load_voxel(batch_path, layout="czxy")

    def test_layer_point_export_preserves_physical_height(self):
        from diffusion_consistency_radar.scripts.streaming_map_update import (
            map_occ_to_points,
        )

        grid = self._small_pose_map()
        layers = np.full((4, 2, 2), 0.5, dtype=np.float32)
        layers[1, 0, 1] = 0.8

        points = map_occ_to_points(layers, grid.cfg, threshold=0.55)

        np.testing.assert_allclose(points, [[1.5, 0.5, 1.5]])

    def test_lazy_query_uses_height_layers_when_z_is_supplied(self):
        from diffusion_consistency_radar.cm.probabilistic_mapping import (
            LazyLocalMapQuery,
        )

        grid = self._small_pose_map()
        snapshot = grid.snapshot()
        snapshot["occ_prob"][1, 0] = 0.8
        snapshot["belief"][1, 0] = 0.9
        snapshot["occ_prob_layers"][1, 0, 0] = 0.8
        snapshot["occ_prob_layers"][1, 0, 1] = 0.8
        snapshot["belief_layers"][1, 0, 0] = 0.2
        snapshot["belief_layers"][1, 0, 1] = 0.8
        query = LazyLocalMapQuery(grid.cfg)
        query.refresh(snapshot)

        low = query.query_proximity(1.5, 0.5, search_radius=5.0, z_m=0.5)
        high = query.query_proximity(1.5, 0.5, search_radius=5.0, z_m=1.5)
        legacy = query.query_proximity(1.5, 0.5, search_radius=5.0)

        self.assertAlmostEqual(low["distance"], 0.0, places=6)
        self.assertAlmostEqual(high["distance"], 0.0, places=6)
        self.assertAlmostEqual(low["uncertainty"], 0.8, places=6)
        self.assertAlmostEqual(high["uncertainty"], 0.2, places=6)
        self.assertAlmostEqual(legacy["uncertainty"], 0.1, places=6)

    def test_target_points_follow_body_to_local_pose(self):
        from diffusion_consistency_radar.scripts.streaming_map_update import (
            transform_points,
        )

        points_body = np.asarray([[1.0, 0.0, 0.5]], dtype=np.float32)
        transform = np.asarray(
            [
                [0.0, -1.0, 0.0, 2.0],
                [1.0, 0.0, 0.0, 3.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        actual = transform_points(points_body, transform)

        np.testing.assert_allclose(actual, [[2.0, 4.0, 1.5]])

    def test_streaming_invalid_prior_dem_fails_before_output_creation(self):
        from diffusion_consistency_radar.scripts.streaming_map_update import main

        with tempfile.TemporaryDirectory() as tmp:
            voxel_dir = os.path.join(tmp, "voxels")
            output_dir = os.path.join(tmp, "map")
            os.makedirs(voxel_dir)
            np.save(
                os.path.join(voxel_dir, "000001_voxel.npy"),
                np.zeros((2, 1, 1, 4), dtype=np.float32),
            )
            prior_path = os.path.join(tmp, "prior.npy")
            np.save(prior_path, np.zeros((3, 1), dtype=np.float32))
            argv = [
                "streaming_map_update.py",
                "--radar_voxel_dir", voxel_dir,
                "--radar_voxel_layout", "xyzc",
                "--prior_dem", prior_path,
                "--output_dir", output_dir,
                "--pc_range", "0", "0", "0", "2", "1", "1",
            ]

            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "prior DEM shape"):
                    main()
            self.assertFalse(os.path.exists(output_dir))

    def test_streaming_target_frames_are_preflighted_before_output_creation(self):
        from diffusion_consistency_radar.scripts.streaming_map_update import main

        with tempfile.TemporaryDirectory() as tmp:
            voxel_dir = os.path.join(tmp, "voxels")
            target_dir = os.path.join(tmp, "targets")
            output_dir = os.path.join(tmp, "map")
            os.makedirs(voxel_dir)
            os.makedirs(target_dir)
            np.save(
                os.path.join(voxel_dir, "000001_voxel.npy"),
                np.zeros((2, 1, 1, 4), dtype=np.float32),
            )
            argv = [
                "streaming_map_update.py",
                "--radar_voxel_dir", voxel_dir,
                "--radar_voxel_layout", "xyzc",
                "--target_voxel_dir", target_dir,
                "--output_dir", output_dir,
                "--pc_range", "0", "0", "0", "2", "1", "1",
            ]

            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "target voxel 帧覆盖"):
                    main()
            self.assertFalse(os.path.exists(output_dir))

    def test_dynamic_evidence_is_pose_aligned_and_separated_from_static_map(self):
        grid = self._small_pose_map()
        voxel = np.zeros((4, 2, 2, 4), dtype=np.float32)
        voxel[0, 0, 0, 0] = 1.0
        voxel[1, 0, 0, 0] = 1.0
        dynamic_probability = np.zeros((4, 2, 2), dtype=np.float32)
        dynamic_observed = np.zeros_like(dynamic_probability)
        dynamic_probability[0, 0, 0] = 1.0
        dynamic_observed[0, 0, 0] = 1.0
        dynamic_observed[1, 0, 0] = 1.0
        transform = np.eye(4, dtype=np.float32)
        transform[0, 3] = 1.0

        grid.update_from_voxel(
            voxel,
            timestamp=1.0,
            observed_mask=np.ones((4, 2, 2), dtype=np.float32),
            T_local_body=transform,
            dynamic_probability=dynamic_probability,
            dynamic_observed_mask=dynamic_observed,
        )
        snapshot = grid.snapshot()

        self.assertEqual(int(snapshot["dynamic_layer_enabled"]), 1)
        self.assertLess(float(snapshot["static_occ_prob_layers"][1, 0, 0]), 0.5)
        self.assertGreater(float(snapshot["dynamic_occ_prob_layers"][1, 0, 0]), 0.5)
        self.assertGreater(float(snapshot["occ_prob_layers"][1, 0, 0]), 0.5)
        self.assertGreater(float(snapshot["static_occ_prob_layers"][2, 0, 0]), 0.5)
        self.assertLess(float(snapshot["dynamic_occ_prob_layers"][2, 0, 0]), 0.5)
        self.assertAlmostEqual(
            float(snapshot["dynamic_unknown_mass_layers"][3, 0, 0]),
            1.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(snapshot["occ_prob_layers"][3, 0, 0]),
            float(snapshot["static_occ_prob_layers"][3, 0, 0]),
            places=6,
        )
        self.assertAlmostEqual(
            float(snapshot["unknown_mass_layers"][3, 0, 0]),
            float(snapshot["static_unknown_mass_layers"][3, 0, 0]),
            places=6,
        )
        for suffix in ("", "_layers"):
            np.testing.assert_allclose(
                snapshot[f"unknown_mass{suffix}"],
                snapshot[f"plausibility{suffix}"]
                - snapshot[f"belief{suffix}"],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                snapshot[f"occ_prob{suffix}"],
                snapshot[f"belief{suffix}"]
                + 0.5 * snapshot[f"unknown_mass{suffix}"],
                atol=1e-6,
            )

    def test_dynamic_evidence_reliability_is_external_to_radar_quality(self):
        from diffusion_consistency_radar.cm.probabilistic_mapping import (
            GridMapConfig,
            SlidingProbabilisticGridMap,
        )

        def build_map(radar_reliability):
            return SlidingProbabilisticGridMap(
                GridMapConfig(
                    x_min=0,
                    x_max=2,
                    y_min=0,
                    y_max=1,
                    x_resolution=1,
                    y_resolution=1,
                    z_min=0,
                    z_max=1,
                    z_resolution=1,
                    radar_reliability=radar_reliability,
                )
            )

        voxel = np.zeros((2, 1, 1, 4), dtype=np.float32)
        probability = np.asarray([[[0.8]], [[0.0]]], dtype=np.float32)
        observed = np.asarray([[[1.0]], [[0.0]]], dtype=np.float32)
        low_radar = build_map(0.1)
        high_radar = build_map(0.9)
        noisy_pose = build_map(0.9)

        for grid in (low_radar, high_radar):
            grid.update_from_voxel(
                voxel,
                timestamp=1.0,
                dynamic_probability=probability,
                dynamic_observed_mask=observed,
            )
        noisy_pose.update_from_voxel(
            voxel,
            timestamp=1.0,
            odom_cov=np.eye(3, dtype=np.float32),
            dynamic_probability=probability,
            dynamic_observed_mask=observed,
        )

        low_belief = float(low_radar.snapshot()["dynamic_belief_layers"][0, 0, 0])
        high_belief = float(high_radar.snapshot()["dynamic_belief_layers"][0, 0, 0])
        noisy_belief = float(noisy_pose.snapshot()["dynamic_belief_layers"][0, 0, 0])
        self.assertAlmostEqual(low_belief, high_belief, places=6)
        self.assertLess(noisy_belief, high_belief)

    def test_dynamic_decay_order_is_checked_only_when_evidence_is_enabled(self):
        from diffusion_consistency_radar.cm.probabilistic_mapping import (
            GridMapConfig,
            SlidingProbabilisticGridMap,
        )

        cfg = GridMapConfig(
            x_min=0,
            x_max=1,
            y_min=0,
            y_max=1,
            x_resolution=1,
            y_resolution=1,
            z_min=0,
            z_max=1,
            z_resolution=1,
            decay_rate=0.6,
            dynamic_decay_rate=0.6,
        )
        grid = SlidingProbabilisticGridMap(cfg)
        voxel = np.zeros((1, 1, 1, 1), dtype=np.float32)
        grid.update_from_voxel(voxel, timestamp=1.0)
        before = grid.snapshot()

        with self.assertRaisesRegex(ValueError, "dynamic_decay_rate"):
            grid.update_from_voxel(
                voxel,
                timestamp=2.0,
                dynamic_probability=np.zeros((1, 1, 1), dtype=np.float32),
                dynamic_observed_mask=np.ones((1, 1, 1), dtype=np.float32),
            )
        after = grid.snapshot()
        np.testing.assert_array_equal(after["occ_prob_layers"], before["occ_prob_layers"])
        self.assertEqual(float(after["last_timestamp"]), 1.0)

    def test_dynamic_evidence_pair_validation_is_side_effect_free_and_lazy(self):
        grid = self._small_pose_map()
        voxel = np.zeros((4, 2, 2, 4), dtype=np.float32)
        probability = np.zeros((4, 2, 2), dtype=np.float32)
        before = grid.snapshot()
        self.assertEqual(int(before["dynamic_layer_enabled"]), 0)
        self.assertNotIn("dynamic_occ_prob_layers", before)

        with self.assertRaisesRegex(ValueError, "必须同时提供"):
            grid.update_from_voxel(
                voxel,
                timestamp=1.0,
                dynamic_probability=probability,
            )
        after = grid.snapshot()
        self.assertEqual(int(after["dynamic_layer_enabled"]), 0)
        np.testing.assert_array_equal(after["occ_prob_layers"], before["occ_prob_layers"])

    def test_dynamic_evidence_decays_faster_than_static_evidence(self):
        from diffusion_consistency_radar.cm.probabilistic_mapping import (
            GridMapConfig,
            SlidingProbabilisticGridMap,
        )

        cfg = GridMapConfig(
            x_min=0,
            x_max=3,
            y_min=0,
            y_max=1,
            x_resolution=1,
            y_resolution=1,
            z_min=0,
            z_max=1,
            z_resolution=1,
            decay_rate=0.1,
            dynamic_decay_rate=2.0,
        )
        grid = SlidingProbabilisticGridMap(cfg)
        first = np.zeros((3, 1, 1, 4), dtype=np.float32)
        first[0, 0, 0, 0] = 1.0
        first[1, 0, 0, 0] = 1.0
        probability = np.zeros((3, 1, 1), dtype=np.float32)
        observed = np.zeros_like(probability)
        probability[0, 0, 0] = 1.0
        observed[0, 0, 0] = 1.0
        observed[1, 0, 0] = 1.0
        grid.update_from_voxel(
            first,
            timestamp=1.0,
            dynamic_probability=probability,
            dynamic_observed_mask=observed,
        )
        before = grid.snapshot()

        second = np.zeros_like(first)
        second[2, 0, 0, 0] = 1.0
        grid.update_from_voxel(second, timestamp=2.0)
        after = grid.snapshot()
        dynamic_ratio = (
            float(after["dynamic_belief_layers"][0, 0, 0])
            / float(before["dynamic_belief_layers"][0, 0, 0])
        )
        static_ratio = (
            float(after["static_belief_layers"][1, 0, 0])
            / float(before["static_belief_layers"][1, 0, 0])
        )

        self.assertLess(dynamic_ratio, static_ratio)
        self.assertGreater(
            float(after["dynamic_unknown_mass_layers"][0, 0, 0]),
            float(before["dynamic_unknown_mass_layers"][0, 0, 0]),
        )

    def test_streaming_dynamic_evidence_protocol_and_output(self):
        from diffusion_consistency_radar.scripts.streaming_map_update import main

        with tempfile.TemporaryDirectory() as tmp:
            voxel_dir = os.path.join(tmp, "voxels")
            evidence_dir = os.path.join(tmp, "dynamic")
            output_dir = os.path.join(tmp, "map")
            os.makedirs(voxel_dir)
            os.makedirs(evidence_dir)
            voxel = np.zeros((2, 1, 1, 4), dtype=np.float32)
            voxel[0, 0, 0, 0] = 1.0
            np.save(os.path.join(voxel_dir, "000001_voxel.npy"), voxel)
            probability = np.asarray([[[1.0]], [[0.0]]], dtype=np.float32)
            observed = np.asarray([[[1]], [[0]]], dtype=np.uint8)
            np.savez_compressed(
                os.path.join(evidence_dir, "000001_dynamic_evidence.npz"),
                probability=probability,
                observed=observed,
            )
            metadata = {
                "protocol": "dynamic_occupancy_evidence_v1",
                "coordinate_frame": "body_voxel",
                "value_semantics": "dynamic_probability",
                "observed_semantics": "explicit_boolean_mask",
                "source": "unit_test_tracker",
                "source_artifact_sha256": "a" * 64,
                "frame_count": 1,
                "pc_range": [0, 0, 0, 2, 1, 1],
                "shape_xyz": [2, 1, 1],
            }
            with open(
                os.path.join(evidence_dir, "dynamic_evidence.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(metadata, handle)
            argv = [
                "streaming_map_update.py",
                "--radar_voxel_dir", voxel_dir,
                "--radar_voxel_layout", "xyzc",
                "--dynamic_evidence_dir", evidence_dir,
                "--dynamic_decay_rate", "2.0",
                "--output_dir", output_dir,
                "--pc_range", "0", "0", "0", "2", "1", "1",
            ]

            with mock.patch.object(sys, "argv", argv):
                main()

            with np.load(os.path.join(output_dir, "map_final.npz")) as result:
                self.assertEqual(int(result["dynamic_layer_enabled"]), 1)
                self.assertIn("dynamic_occ_prob_layers", result.files)
                self.assertIn("static_occ_prob_layers", result.files)
                self.assertGreater(float(result["occ_prob_layers"][0, 0, 0]), 0.5)
                self.assertLess(float(result["static_occ_prob_layers"][0, 0, 0]), 0.5)
            with open(
                os.path.join(output_dir, "map_run.json"),
                "r",
                encoding="utf-8",
            ) as handle:
                run = json.load(handle)
            self.assertTrue(run["dynamic_evidence_enabled"])
            self.assertEqual(run["dynamic_evidence_source"], "unit_test_tracker")
            self.assertRegex(run["dynamic_evidence_files_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                run["dynamic_evidence_source_artifact_hash_status"],
                "declared_by_metadata_unresolved",
            )
            self.assertEqual(
                run["dynamic_evidence_reliability"],
                "explicit_observed_times_odometry_confidence",
            )
            self.assertEqual(
                run["combined_static_dynamic_semantics"],
                "dynamic_occupied_pignistic_overlay",
            )
            self.assertEqual(run["decay_rate_base"], 0.12)
            self.assertEqual(run["decay_rate_effective"], 0.12)
            self.assertEqual(run["dynamic_decay_rate_base"], 2.0)
            self.assertEqual(run["dynamic_decay_rate_effective"], 2.0)

    def test_streaming_dynamic_metadata_rejects_numeric_strings_before_output(self):
        from diffusion_consistency_radar.scripts.streaming_map_update import main

        with tempfile.TemporaryDirectory() as tmp:
            voxel_dir = os.path.join(tmp, "voxels")
            evidence_dir = os.path.join(tmp, "dynamic")
            output_dir = os.path.join(tmp, "map")
            os.makedirs(voxel_dir)
            os.makedirs(evidence_dir)
            np.save(
                os.path.join(voxel_dir, "000001_voxel.npy"),
                np.zeros((2, 1, 1, 4), dtype=np.float32),
            )
            np.savez_compressed(
                os.path.join(evidence_dir, "000001_dynamic_evidence.npz"),
                probability=np.zeros((2, 1, 1), dtype=np.float32),
                observed=np.zeros((2, 1, 1), dtype=np.uint8),
            )
            metadata = {
                "protocol": "dynamic_occupancy_evidence_v1",
                "coordinate_frame": "body_voxel",
                "value_semantics": "dynamic_probability",
                "observed_semantics": "explicit_boolean_mask",
                "source": "unit_test_tracker",
                "source_artifact_sha256": "a" * 64,
                "frame_count": 1,
                "pc_range": ["0", 0, 0, 2, 1, 1],
                "shape_xyz": [2, 1, 1],
            }
            with open(
                os.path.join(evidence_dir, "dynamic_evidence.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(metadata, handle)
            argv = [
                "streaming_map_update.py",
                "--radar_voxel_dir", voxel_dir,
                "--radar_voxel_layout", "xyzc",
                "--dynamic_evidence_dir", evidence_dir,
                "--output_dir", output_dir,
                "--pc_range", "0", "0", "0", "2", "1", "1",
            ]

            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "JSON number"):
                    main()
            self.assertFalse(os.path.exists(output_dir))

    def test_streaming_dynamic_npz_is_preflighted_before_output(self):
        from diffusion_consistency_radar.scripts.streaming_map_update import main

        with tempfile.TemporaryDirectory() as tmp:
            voxel_dir = os.path.join(tmp, "voxels")
            evidence_dir = os.path.join(tmp, "dynamic")
            output_dir = os.path.join(tmp, "map")
            os.makedirs(voxel_dir)
            os.makedirs(evidence_dir)
            np.save(
                os.path.join(voxel_dir, "000001_voxel.npy"),
                np.zeros((2, 1, 1, 4), dtype=np.float32),
            )
            # shape 故意错误；正式循环开始前必须拒绝，不能留下半成品目录。
            np.savez_compressed(
                os.path.join(evidence_dir, "000001_dynamic_evidence.npz"),
                probability=np.zeros((1, 1, 1), dtype=np.float32),
                observed=np.zeros((1, 1, 1), dtype=np.uint8),
            )
            metadata = {
                "protocol": "dynamic_occupancy_evidence_v1",
                "coordinate_frame": "body_voxel",
                "value_semantics": "dynamic_probability",
                "observed_semantics": "explicit_boolean_mask",
                "source": "unit_test_tracker",
                "source_artifact_sha256": "a" * 64,
                "frame_count": 1,
                "pc_range": [0, 0, 0, 2, 1, 1],
                "shape_xyz": [2, 1, 1],
            }
            with open(
                os.path.join(evidence_dir, "dynamic_evidence.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(metadata, handle)
            argv = [
                "streaming_map_update.py",
                "--radar_voxel_dir", voxel_dir,
                "--radar_voxel_layout", "xyzc",
                "--dynamic_evidence_dir", evidence_dir,
                "--output_dir", output_dir,
                "--pc_range", "0", "0", "0", "2", "1", "1",
            ]

            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "shape 不匹配"):
                    main()
            self.assertFalse(os.path.exists(output_dir))

    def test_streaming_rejects_duplicate_radar_frame_keys_before_output(self):
        from diffusion_consistency_radar.scripts.streaming_map_update import main

        with tempfile.TemporaryDirectory() as tmp:
            voxel_dir = os.path.join(tmp, "voxels")
            output_dir = os.path.join(tmp, "map")
            os.makedirs(voxel_dir)
            voxel = np.zeros((2, 1, 1, 4), dtype=np.float32)
            np.save(os.path.join(voxel_dir, "000001.npy"), voxel)
            np.save(os.path.join(voxel_dir, "000001_voxel.npy"), voxel)
            argv = [
                "streaming_map_update.py",
                "--radar_voxel_dir", voxel_dir,
                "--radar_voxel_layout", "xyzc",
                "--output_dir", output_dir,
                "--pc_range", "0", "0", "0", "2", "1", "1",
            ]

            with mock.patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "重复 frame 键"):
                    main()
            self.assertFalse(os.path.exists(output_dir))


if __name__ == "__main__":
    unittest.main()
