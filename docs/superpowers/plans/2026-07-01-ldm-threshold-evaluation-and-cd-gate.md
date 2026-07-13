# LDM 阈值评估修正与 CD 准入 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修正近场 LDM 保存结果的阈值扫描坐标协议，在独立验证子集上按避障任务指标选择阈值，并据此决定是否开始 CD 蒸馏。

**Architecture:** 阈值扫描必须先把原始 `0-120m` target 按物理范围裁剪到模型实际学习的 `0-40m`，再统一缩放到 `(Z,X,Y)=(32,128,128)`。阈值选择从严格的逐体素重合 F1 改为近场分段 BEV F1，同时保留体素 F1、最近邻和点数比例作为诊断；只使用训练时固定随机种子产生的 validation 索引进行选择。

**Tech Stack:** Python 3.8、NumPy、PyTorch、SciPy、CSV/JSON、`unittest`、现有 `cm.dataset_loader` 与 `cm.evaluation_metrics`

---

## 当前判定

- 现有扫描推荐阈值 `0.1`，但精确体素 F1 仅为 `0.007126`，不能作为部署阈值。
- 扫描 JSON 记录的 `pc_range=[0,-20,-6,120,20,10]`，而本轮模型学习范围是 `0-40m`。
- `load_target_occ_resized()` 直接把完整 `0-120m` target 缩放到 128 个 X 栅格，没有先裁剪到 `0-40m`；预测和目标的物理 X 坐标不一致。
- 正式推理在阈值 `0.5` 下得到近场 recall `0.5742`、precision `0.6646`、BEV IoU `0.4478`、2m match ratio `0.9620`，与扫描的近零重合明显冲突。
- 当前 LDM 相对 Radar baseline 的 Chamfer 从 `2.0553` 降到 `1.3749`，说明生成模型已有收益；在修正阈值评估前，不应重新训练 LDM，也不应开始 CD。

### Task 1: 修正 target 的物理裁剪与网格元数据

**Files:**
- Modify: `diffusion_consistency_radar/scripts/sweep_occ_threshold.py:20-42`
- Modify: `diffusion_consistency_radar/scripts/sweep_occ_threshold.py:111-129`
- Test: `test/unit/test_occ_threshold_grid_protocol.py`

- [x] **Step 1: 写出物理裁剪失败测试**

```python
import os
import tempfile
import unittest

import numpy as np
import torch

from diffusion_consistency_radar.scripts.sweep_occ_threshold import load_target_occ_resized


class OccupancyThresholdGridProtocolTest(unittest.TestCase):
    def test_target_is_cropped_before_resize(self):
        voxel = np.zeros((12, 4, 4, 4), dtype=np.float32)
        voxel[1, 1, 1, 0] = 1.0
        voxel[10, 1, 1, 0] = 1.0
        voxel[1, 1, 1, 3] = 1.0
        voxel[10, 1, 1, 3] = 1.0
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "000000.npy")
            np.save(path, voxel)
            result = load_target_occ_resized(
                path,
                torch.device("cpu"),
                source_pc_range=(0, -20, -6, 120, 20, 10),
                model_pc_range=(0, -20, -6, 40, 20, 10),
                target_size=(4, 4, 4),
            )
        self.assertEqual(result.shape, (4, 4, 4))
        self.assertEqual(int(np.count_nonzero(result > 0.1)), 1)
```

- [x] **Step 2: 运行测试并确认 RED**

Run:

```bash
conda run -n Radar-Diffusion python test/unit/test_occ_threshold_grid_protocol.py -v
```

Expected: FAIL，提示 `load_target_occ_resized()` 不接受新增网格参数。

- [x] **Step 3: 最小实现“裁剪后缩放”**

```python
from cm.dataset_loader import crop_voxel_channels_to_pc_range, resize_voxel_channels


def load_target_occ_resized(path, device, source_pc_range, model_pc_range, target_size):
    target = load_sparse_voxel(path) if path.endswith(".npz") else np.load(path).astype(np.float32)
    tensor = torch.from_numpy(target).permute(3, 2, 0, 1).to(device)
    tensor = crop_voxel_channels_to_pc_range(tensor, source_pc_range, model_pc_range)
    resized = resize_voxel_channels(tensor, tuple(target_size), mask_channel=3)
    return resized[0].cpu().numpy()
```

新增 CLI：

```python
parser.add_argument("--source_pc_range", type=float, nargs=6, default=[0, -20, -6, 120, 20, 10])
parser.add_argument("--model_pc_range", type=float, nargs=6, default=[0, -20, -6, 40, 20, 10])
parser.add_argument("--target_size", type=int, nargs=3, default=[32, 128, 128])
```

输出 JSON 必须记录 `source_pc_range`、`model_pc_range` 和 `target_size`，不再使用含义模糊的单一 `pc_range`。

- [x] **Step 4: 运行聚焦测试**

```bash
conda run -n Radar-Diffusion python test/unit/test_occ_threshold_grid_protocol.py -v
conda run -n Radar-Diffusion python -m py_compile diffusion_consistency_radar/scripts/sweep_occ_threshold.py
```

Expected: 测试 PASS，编译无输出。

- [ ] **Step 5: 提交**

```bash
git add diffusion_consistency_radar/scripts/sweep_occ_threshold.py test/unit/test_occ_threshold_grid_protocol.py
git commit -m "fix: align occupancy threshold sweep grids"
```

### Task 2: 仅在 validation 子集上按任务指标选阈值

**Files:**
- Modify: `diffusion_consistency_radar/scripts/sweep_occ_threshold.py:45-108`
- Modify: `diffusion_consistency_radar/scripts/sweep_occ_threshold.py:131-257`
- Test: `test/unit/test_occ_threshold_grid_protocol.py`

- [x] **Step 1: 写出固定划分与任务指标选择测试**

```python
from diffusion_consistency_radar.scripts.sweep_occ_threshold import (
    select_evaluation_files,
    select_recommended_threshold,
)


def test_validation_split_is_deterministic(self):
    files = [f"{index:06d}_voxel.npy" for index in range(10)]
    first = select_evaluation_files(files, "validation", 0.8, 42)
    second = select_evaluation_files(files, "validation", 0.8, 42)
    self.assertEqual(first, second)
    self.assertEqual(len(first), 2)


def test_task_bev_f1_is_primary_selection_metric(self):
    metrics = {
        0.3: {"task_bev_f1": 0.55, "task_bev_iou": 0.40, "pred_to_target_ratio": 1.6},
        0.5: {"task_bev_f1": 0.62, "task_bev_iou": 0.45, "pred_to_target_ratio": 1.1},
        0.7: {"task_bev_f1": 0.58, "task_bev_iou": 0.43, "pred_to_target_ratio": 1.0},
    }
    self.assertEqual(select_recommended_threshold(metrics, "task_bev_f1"), 0.5)
```

- [x] **Step 2: 运行测试并确认 RED**

```bash
conda run -n Radar-Diffusion python test/unit/test_occ_threshold_grid_protocol.py -v
```

Expected: FAIL，提示两个选择函数尚不存在。

- [x] **Step 3: 实现与训练一致的固定划分**

```python
def select_evaluation_files(pred_files, evaluation_split, train_split, split_seed):
    if evaluation_split == "all":
        return list(pred_files)
    if len(pred_files) < 2:
        raise ValueError("validation threshold selection requires at least 2 predictions")
    train_size = int(len(pred_files) * float(train_split))
    if train_size <= 0 or train_size >= len(pred_files):
        raise ValueError("train_split creates an empty partition")
    generator = torch.Generator().manual_seed(int(split_seed))
    indices = torch.randperm(len(pred_files), generator=generator).tolist()
    selected = indices[:train_size] if evaluation_split == "train" else indices[train_size:]
    return [pred_files[index] for index in selected]
```

CLI 增加：

```python
parser.add_argument("--evaluation_split", choices=("train", "validation", "all"), default="validation")
parser.add_argument("--train_split", type=float, default=0.8)
parser.add_argument("--split_seed", type=int, default=42)
parser.add_argument("--range_bins", default="0-20,20-40")
parser.add_argument("--bev_cell_size", type=float, default=0.5)
parser.add_argument("--selection_metric", choices=("task_bev_f1", "voxel_f1"), default="task_bev_f1")
```

- [x] **Step 4: 计算近场分段任务指标**

复用 `cm.evaluation_metrics` 中的 `voxel_to_points()`、`filter_points_by_band()`、`occupancy_prf()`、`bev_iou()`、`nearest_neighbor_metrics()` 和 `parse_range_bins()`。每个阈值按 `model_pc_range` 转点云，对 `0-20m`、`20-40m` 且 `z>=-1m` 分别累计 BEV TP/FP/FN、2m match ratio 和点数。

```python
def select_recommended_threshold(metrics, selection_metric):
    return max(
        metrics,
        key=lambda value: (
            metrics[value][selection_metric],
            metrics[value].get("task_bev_iou", 0.0),
            -abs((metrics[value].get("pred_to_target_ratio") or 1.0) - 1.0),
            -abs(float(value) - 0.5),
        ),
    )
```

CSV/JSON 同时保留严格 voxel F1，明确标注它是辅助诊断，不作为默认选择依据。

- [x] **Step 5: 运行测试与既有指标回归**

```bash
conda run -n Radar-Diffusion python test/unit/test_occ_threshold_grid_protocol.py -v
conda run -n Radar-Diffusion python test/unit/test_formal_task_metrics.py -v
```

Expected: 两组测试全部 PASS。

- [ ] **Step 6: 提交**

```bash
git add diffusion_consistency_radar/scripts/sweep_occ_threshold.py test/unit/test_occ_threshold_grid_protocol.py
git commit -m "feat: calibrate occupancy threshold on validation task metrics"
```

### Task 3: 用已保存的 500 帧输出重新扫描

**Files:**
- Generate: `test/result/ldm/evaluation/ldm_near40_500_v2/loop3_ldm_eval/occ_sweep_validation_metrics.csv`
- Generate: `test/result/ldm/evaluation/ldm_near40_500_v2/loop3_ldm_eval/occ_threshold_validation_recommendation.json`
- Modify: `TODO/findings.md`
- Modify: `TODO/progress.md`

- [x] **Step 1: 运行修正后的扫描，不重新推理**

```bash
conda run -n Radar-Diffusion python diffusion_consistency_radar/scripts/sweep_occ_threshold.py \
  --pred_voxel_dir test/result/ldm/evaluation/ldm_near40_500_v2/loop3_ldm_eval \
  --target_voxel_dir Data/NTU4DRadLM_Pre_sensor_aware/loop3/target_voxel \
  --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --source_pc_range 0 -20 -6 120 20 10 \
  --model_pc_range 0 -20 -6 40 20 10 \
  --target_size 32 128 128 \
  --evaluation_split validation \
  --train_split 0.8 \
  --split_seed 42 \
  --range_bins 0-20,20-40 \
  --z_min -1 \
  --selection_metric task_bev_f1 \
  --output_csv test/result/ldm/evaluation/ldm_near40_500_v2/loop3_ldm_eval/occ_sweep_validation_metrics.csv \
  --output_json test/result/ldm/evaluation/ldm_near40_500_v2/loop3_ldm_eval/occ_threshold_validation_recommendation.json
```

Expected: 只读取已有 `*_voxel.npy`；处理 100 个 validation frame；JSON 中记录 `evaluation_split=validation`、`model_pc_range` 终点为 `40`。

- [x] **Step 2: 检查新推荐值的有效性**

```bash
conda run -n Radar-Diffusion python -c "import json; p='test/result/ldm/evaluation/ldm_near40_500_v2/loop3_ldm_eval/occ_threshold_validation_recommendation.json'; d=json.load(open(p)); print(d['recommended_threshold']); print(d['metrics'][str(d['recommended_threshold'])])"
```

Expected:

- `task_bev_f1` 不再是接近零的异常值。
- `pred_to_target_ratio` 建议落在 `[0.8, 1.3]`；超出时保留结果但不得进入 CD。
- 推荐阈值附近相邻两个阈值的 BEV F1 不应突变；若突变超过 `0.10`，先抽查保存体素和 frame 配对。

- [x] **Step 3: 固定阈值做一次正式复评**

```bash
VALIDATED_THRESHOLD=$(python3 -c "import json; p='test/result/ldm/evaluation/ldm_near40_500_v2/loop3_ldm_eval/occ_threshold_validation_recommendation.json'; print(json.load(open(p))['recommended_threshold'])")
MINI_RESULTS_DIR=test/result/vae/reconstruction/vae_near40_500_v2 \
MINI_INFERENCE_RESULTS_DIR=test/result/ldm/evaluation/ldm_near40_500_v2_threshold_validated \
MAX_INFER_FILES=500 \
OCC_THRESHOLD="${VALIDATED_THRESHOLD}" \
bash test/mini-test/inference_minimal.sh ldm
```

Expected: 新目录生成 500 帧指标；原始结果不被覆盖。

- [x] **Step 4: 记录实验结论**

在 `TODO/findings.md` 记录推荐阈值、两个距离段的 BEV F1/IoU/precision/recall、Chamfer、点数比例；在 `TODO/progress.md` 记录命令和输出目录。

### Task 4: 设置 CD 蒸馏准入门槛

**Files:**
- Modify: `TODO/task_plan.md`
- Modify: `TODO/findings.md`
- Modify: `TODO/progress.md`

- [x] **Step 1: 按固定门槛判断 LDM**

```text
0-20m near BEV IoU >= 0.40
0-20m near recall >= 0.55
0-20m near precision >= 0.60
validation pred/target count ratio in [0.8, 1.3]
mean_pred_target_chamfer < mean_radar_target_chamfer
10 帧 raw LiDAR 对照中至少 8 帧能看到主要障碍物/树干位置
```

固定阈值 `0.1` 的内部数值门槛均通过，但证据来自不同统计集合：near IoU/recall/precision 与 Chamfer 来自包含训练帧的 500 帧内部全量复评，threshold selection 与点数比例来自 100 帧 validation。正式准入必须统一到独立 validation/test 集重新计算。

视觉帧来自按 `split_seed=42` 确定的数据划分；这只保证 validation 帧集合可复现，不代表 LDM 生成采样 seed 固定。视觉检查显示树干连续性和树冠细结构均未稳定恢复，因此针对当前树木结构目标，gate 总判定为 **HOLD / FAIL**，不启动 CD。

树木结构 gate 后续需以最高点高度召回、垂直连通率、树干区域 recall 等指标量化，阈值应依据实验分布确定，本阶段不预设数值。执行顺序为：先实现结构指标并检查 VAE 重建上界；若 VAE 通过而 LDM 失败，再加入垂直结构或高度分布损失重训 LDM；若 VAE 也失败，先提高 Z/X 物理分辨率或调整监督目标。

- [ ] **Step 2: 门槛通过后训练 CD（BLOCKED by gate）**

```bash
SAMPLES_PER_SCENE=500 \
MINI_CD_EPOCHS=20 \
MINI_RESULTS_DIR=test/result/vae/reconstruction/vae_near40_500_v2 \
bash test/mini-test/train_minimal.sh cd
```

Expected: 使用同目录 VAE/LDM checkpoint 生成 `cd/cd_best.pt`。该长训练命令由用户手动执行。

- [ ] **Step 3: 分别评估 CD 1-step 与 4-step（BLOCKED by gate）**

```bash
VALIDATED_THRESHOLD=$(python3 -c "import json; p='test/result/ldm/evaluation/ldm_near40_500_v2/loop3_ldm_eval/occ_threshold_validation_recommendation.json'; print(json.load(open(p))['recommended_threshold'])")
MINI_RESULTS_DIR=test/result/vae/reconstruction/vae_near40_500_v2 \
MINI_INFERENCE_RESULTS_DIR=test/result/ldm/evaluation/ldm_near40_500_v2_threshold_validated/cd_1step \
MAX_INFER_FILES=500 \
OCC_THRESHOLD="${VALIDATED_THRESHOLD}" \
STEPS=1 \
SAMPLER=euler \
bash test/mini-test/inference_minimal.sh cd
```

```bash
VALIDATED_THRESHOLD=$(python3 -c "import json; p='test/result/ldm/evaluation/ldm_near40_500_v2/loop3_ldm_eval/occ_threshold_validation_recommendation.json'; print(json.load(open(p))['recommended_threshold'])")
MINI_RESULTS_DIR=test/result/vae/reconstruction/vae_near40_500_v2 \
MINI_INFERENCE_RESULTS_DIR=test/result/ldm/evaluation/ldm_near40_500_v2_threshold_validated/cd_4step \
MAX_INFER_FILES=500 \
OCC_THRESHOLD="${VALIDATED_THRESHOLD}" \
STEPS=4 \
SAMPLER=heun \
bash test/mini-test/inference_minimal.sh cd
```

Expected: 1-step 作为实时版本，4-step 作为质量版本；两者使用同一个 validation 固定阈值。

- [x] **Step 4: 最终回归检查**

```bash
conda run -n Radar-Diffusion python test/unit/test_occ_threshold_grid_protocol.py -v
conda run -n Radar-Diffusion python test/unit/test_formal_task_metrics.py -v
conda run -n Radar-Diffusion python test/unit/test_multimodal_inference_interface.py -v
conda run -n Radar-Diffusion python -m py_compile \
  diffusion_consistency_radar/scripts/sweep_occ_threshold.py \
  diffusion_consistency_radar/scripts/inference.py
git diff --check
```

Expected: 测试全部 PASS，编译和 `git diff --check` 无错误。
