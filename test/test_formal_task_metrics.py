import csv
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class FormalTaskMetricsTest(unittest.TestCase):
    def test_uncertainty_calibration_rewards_high_uncertainty_on_errors(self):
        from diffusion_consistency_radar.cm.evaluation_metrics import uncertainty_calibration_metrics

        target = np.zeros((2, 2, 2), dtype=np.float32)
        prediction = target.copy()
        prediction[0, 0, 0] = 1.0
        low_uncertainty = np.zeros_like(target)
        informed_uncertainty = np.zeros_like(target)
        informed_uncertainty[0, 0, 0] = 20.0

        low = uncertainty_calibration_metrics(prediction, target, low_uncertainty, occ_threshold=0.5)
        informed = uncertainty_calibration_metrics(prediction, target, informed_uncertainty, occ_threshold=0.5)

        self.assertLess(informed["uncertainty_brier"], low["uncertainty_brier"])
        self.assertLess(informed["uncertainty_nll"], low["uncertainty_nll"])
        self.assertGreater(informed["uncertainty_error_corr"], 0.9)

    def test_threshold_calibration_selects_best_f1_in_task_region(self):
        from diffusion_consistency_radar.scripts.sweep_occ_threshold import evaluate_thresholds

        pred_occ = np.zeros((2, 4, 2), dtype=np.float32)
        target_occ = np.zeros_like(pred_occ)
        target_occ[1, 1, 0] = 1.0
        target_occ[1, 2, 0] = 1.0
        pred_occ[1, 1, 0] = 0.9
        pred_occ[1, 2, 0] = 0.8
        pred_occ[1, 0, 0] = 0.6
        pred_occ[1, 3, 0] = 0.5

        result = evaluate_thresholds(
            pred_occ,
            target_occ,
            thresholds=[0.2, 0.7],
            target_threshold=0.5,
            pc_range=(0, -1, -1, 4, 1, 1),
            x_max=4.0,
            z_min=0.0,
        )

        self.assertEqual(result["best_threshold"], 0.7)
        self.assertEqual(result["metrics"][0.7]["f1"], 1.0)
        self.assertLess(result["metrics"][0.2]["precision"], 1.0)

    def test_overlap_filter_and_precision_recall_metrics(self):
        from diffusion_consistency_radar.cm.evaluation_metrics import (
            bev_iou,
            filter_points_by_band,
            occupancy_prf,
            voxel_to_points,
        )

        voxel = np.zeros((4, 2, 2, 4), dtype=np.float32)
        voxel[0, 0, 0, 0] = 1.0
        voxel[2, 1, 1, 0] = 1.0
        points = voxel_to_points(voxel, pc_range=(0, -1, -1, 4, 1, 1), occ_threshold=0.5)
        high_points = filter_points_by_band(points, pc_range=(0, -1, -1, 4, 1, 1), x_min=0, x_max=4, z_min=0.0)

        self.assertEqual(points.shape[0], 2)
        self.assertEqual(high_points.shape[0], 1)
        self.assertEqual(bev_iou(points, points, pc_range=(0, -1, -1, 4, 1, 1), cell_size=1.0)["bev_iou"], 1.0)

        empty_pred = np.zeros((0, 3), dtype=np.float32)
        prf = occupancy_prf(empty_pred, points, pc_range=(0, -1, -1, 4, 1, 1), cell_size=1.0)
        self.assertEqual(prf["recall"], 0.0)
        self.assertEqual(prf["precision"], 0.0)

    def test_inference_task_metric_header_and_summary_have_same_width(self):
        from diffusion_consistency_radar.scripts.inference import (
            append_task_metric_headers,
            build_task_metric_row,
            build_task_metric_summary_row,
        )

        header = ["index", "radar_file"]
        append_task_metric_headers(header)
        row = ["__summary__", ""]
        build_task_metric_summary_row(row, header, {"task_near_recall_mean": 0.25})

        self.assertEqual(len(header), len(row))
        self.assertIn("task_near_recall_mean", header)
        self.assertIn("uncertainty_brier", header)
        self.assertIn("uncertainty_ece", header)
        self.assertIn("uncertainty_nll", header)
        self.assertIn("uncertainty_error_corr", header)
        self.assertEqual(row[header.index("task_near_recall_mean")], "0.250000")

        frame_row = ["0", "000000.npz"]
        build_task_metric_row(frame_row, header, {"task_near_precision_mean": 0.75})
        self.assertEqual(frame_row[header.index("task_near_precision_mean")], "0.750000")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metrics.csv")
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerow(row)
            with open(path, newline="") as f:
                rows = list(csv.reader(f))
            self.assertEqual(len(rows[0]), len(rows[1]))


if __name__ == "__main__":
    unittest.main()
