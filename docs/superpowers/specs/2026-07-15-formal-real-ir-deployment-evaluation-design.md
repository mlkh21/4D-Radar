# 正式真实 IR 与部署/离线评价解耦设计

日期：2026-07-15

## 目标

将正式 LDM/CD 推理统一到 sensor-aware 四模态预处理根，使正式多模态结果只接受真实 IR 和真实 thermal 外参；同时把 target/raw LiDAR 评价从生成入口移到独立的离线评价阶段，保证评价消费的是部署阶段已经保存的同一批预测，而不是重新运行一次随机生成。

本项对应 `TODO/26-7-15.md` 第一阶段第 4 项。它修复正式实验入口，不宣称项目已经具备实时 ROS/PX4 部署系统。

## 已确认根因

- 正式训练读取 `Data/NTU4DRadLM_Pre_sensor_aware`，三个正式推理 launcher 读取旧 `Data/NTU4DRadLM_Pre`；旧根没有 `ir_image`。
- LDM/CD launcher 默认关闭或没有传入 `--use_multimodal_meta`，因此正式生成没有稳定使用真实 IR。
- `load_multimodal_meta_for_radar()` 在 IR 缺失时保留合成 thermal；`is_mock_ir` 只增加 uncertainty，不会把 IR 投影特征或融合 gate 归零。
- 当前正式 launcher 强制检查 target、raw Livox 和 LiDAR index，并把评价参数传入生成命令，导致正式部署入口在结构上依赖离线真值。
- 把评价参数移到另一个、仍会运行 `inference.py` 的 launcher 会重新采样，评价对象不再是部署时保存的预测，因此不是真正解耦。
- Dataset 对真实和 mock 标定都施加固定 `50m/s * 200us = 0.01m` 的 x 向同步位移；逐文件 inference 只对 mock 标定施加该位移，训练和真实 IR 推理的投影协议不一致。

## 修改范围

修改：

- `diffusion_consistency_radar/scripts/inference.py`
- `diffusion_consistency_radar/launch/inference_ldm.sh`
- `diffusion_consistency_radar/launch/inference_cd.sh`
- `diffusion_consistency_radar/launch/inference_uniified.sh`
- 相关 unit tests 和三份 `TODO/*.md`

新增：

- `diffusion_consistency_radar/scripts/evaluate_saved_predictions.py`
- `diffusion_consistency_radar/launch/evaluate_inference.sh`
- `test/unit/test_formal_inference_protocol.py`
- 本规格及对应实施计划

不修改：

- 训练监督、target 生成、Dataset 样本成员、模型结构和 checkpoint 权重；
- thermal K/D 解析、图像去畸变或真实逐帧速度/时间差；这些属于审计第二阶段；
- 通用 Dataset 的 mock fallback、mini-test 和 IR 消融诊断兼容行为；
- 当前 Data、checkpoint、日志和历史结果；
- 正式 checkpoint 链选择；它是第一阶段下一项。

## 方案选择

采用“严格正式生成 + 已保存预测离线评价”的两阶段方案。

未采用仅拆 shell launcher 的方案，因为评价会重新生成随机预测。未采用从 `inference.py` 全面删除评价能力的方案，因为多个 mini-test 和诊断脚本仍依赖兼容接口，超出当前正式入口修复范围。

## 正式生成协议

### 数据根和 launcher

三个正式推理 launcher 的 `PREPROCESSED_ROOT` 统一改为：

```text
Data/NTU4DRadLM_Pre_sensor_aware
```

保留上一项加入的严格 `dataset_manifest.json` gate。该 manifest 验证的是正式离线实验数据集完整性，包含 radar/lidar/target/IR 四模态；生成命令本身不接收 target、raw LiDAR 或 LiDAR index。面向实时流的 Radar+IR 专用输入 manifest 不在本项实现。

launcher 删除以下部署参数和对应目录检查：

```text
--target_voxel_dir
--compare_with_target
--report_task_metrics
--compare_with_lidar
--raw_livox_dir
--lidar_index_file
```

正式生成保留固定全局 `--occ_threshold`，保存 `*_voxel.npy`、`*_pcl.npy` 和可用的 `*_uncertainty.npy`。

### 严格真实 IR 模式

`inference.py` 新增：

```text
--require_real_ir
```

该参数只用于正式多模态生成，并隐含启用 IR meta。它在创建输出目录和生成第一帧之前要求：

- checkpoint 构建出的模型 `is_multimodal=True`；单模态 checkpoint 不能冒充正式 Radar+IR checkpoint；
- 每个待推理 frame 都存在匹配的 `<frame_id>_ir.npy` 普通文件，拒绝 symlink；
- IR 数组维度可被现有 `_resize_or_pad_ir_tensor()` 接受，数值为有限值；
- `CalibrationProvider.load_with_metadata()` 找到真实 `calib_radar_to_thermal.txt`，`is_mock_calib=False` 且 `calib_is_thermal=True`；
- 最终 meta 中 `is_mock_ir=0`、`is_mock_calib=0`。

任一条件不满足时抛出包含场景、frame 或标定来源的错误，并且不创建正式输出目录。正式模式不允许回退 mock thermal、mock 外参或单模态模型。

不带 `--require_real_ir` 的直接 Python、mini-test 和消融入口保持原兼容行为，避免本项破坏诊断能力。

### 训练/推理投影一致性

真实 IR 的逐文件 inference 对真实 thermal 外参同样加入现有训练协议使用的 `0.01m` x 向同步位移，并在 meta/运行记录中标明该值来自 legacy fixed sync compensation。

这只是消除训练/推理不一致，不代表固定 `50m/s` 和 `200us` 是真实飞行状态。逐帧导航速度、姿态和真实时间差仍必须在第二阶段修复。

### 运行输出

无 target/LiDAR 评价参数时，逐文件生成写 `inference_runtime.csv`，只包含：

```text
index, radar_file, radar_point_count, effective_occ_threshold,
inference_seconds, pred_point_count, is_empty_frame,
used_topk_fallback, train_duration_seconds,
total_infer_seconds, avg_infer_seconds,
avg_pred_point_count, empty_frame_rate
```

兼容模式下如果显式传入旧评价参数，继续写历史 `inference_metrics.csv`，以免破坏 mini-test；正式 launcher 不再使用该分支。

正式输出目录后缀由 `_eval` 改为 `_deploy`，避免把尚未评价的预测命名成评价结果：

- `<scene>_ldm_deploy`
- `<scene>_cd_1step_deploy`
- `<scene>_cd_4step_deploy`

每个部署输出目录同时写 `inference_run.json`，至少记录实际解析后的 `target_size`、`source_pc_range`、`model_pc_range`、`voxel_size`、固定 occupancy 阈值、model type、steps、sampler、`model_is_multimodal`、`require_real_ir` 和 frame count。离线 evaluator 必须优先读取该文件，不能用硬编码网格猜测 checkpoint 的实际输出范围。

## 独立离线评价协议

### Python 入口

新增 `evaluate_saved_predictions.py`。它只读取已保存预测和离线真值，不接收 checkpoint，不导入或调用 `RadarGenerator.generate()`。

主要接口：

```python
evaluate_saved_predictions(
    pred_voxel_dir: str,
    radar_voxel_dir: str,
    target_voxel_dir: str,
    output_dir: str,
    run_metadata_path: str = "",
    raw_livox_dir: str = "",
    lidar_index_file: str = "",
    occ_threshold: Optional[float] = None,
    target_threshold: float = 0.5,
    source_pc_range: Optional[Sequence[float]] = None,
    model_pc_range: Optional[Sequence[float]] = None,
    target_size: Optional[Sequence[int]] = None,
    max_files: int = 0,
) -> dict
```

`run_metadata_path` 为空时默认读取 `<pred_voxel_dir>/inference_run.json`。正式评价要求该文件存在且字段完整；显式 CLI 网格/阈值仅作为诊断兼容覆盖值，并在输出 JSON 中记录覆盖来源。

它要求 prediction、radar 和 target frame ID 一一对应。`raw_livox_dir` 与 `lidar_index_file` 必须同时提供或同时省略，防止按排序位置静默错配。

评价沿用当前固定阈值协议，计算：

- prediction/target 和 radar/target 点数、Chamfer、数量比、质心偏移；
- 近场 precision、recall、BEV IoU、NN mean 和 2m match ratio；
- 提供 raw LiDAR 映射时计算 raw-LiDAR Chamfer；
- 存在匹配 `*_uncertainty.npy` 时计算 uncertainty ECE、Brier、NLL 和 error correlation。

评价不得修改 prediction voxel 或点云，不允许 target 改变阈值或输出。输出目录必须不存在或为空，避免覆盖历史评价。

输出：

- `evaluation_frames.csv`：逐帧指标；
- `evaluation_summary.json`：聚合指标、阈值、网格、frame count 和输入路径；
- JSON 固定记录 `stage="offline_evaluation"`、`prediction_unchanged=true`。

### Shell 入口

新增 `evaluate_inference.sh`，接受：

```text
evaluate_inference.sh ldm
evaluate_inference.sh cd
evaluate_inference.sh cd4
```

它按 `data_loading_config.yml` 的 test scene 验证严格 dataset manifest，选择对应 `_deploy` 预测目录及其 `inference_run.json`，然后调用 `evaluate_saved_predictions.py`。该脚本可以要求 target/raw LiDAR，因为它明确是离线评价入口；它不得引用 checkpoint 或 `inference.py`。

评价输出目录分别为：

- `<scene>_ldm_evaluation`
- `<scene>_cd_1step_evaluation`
- `<scene>_cd_4step_evaluation`

## 错误处理

- 正式 launcher 在任何场景不满足 manifest 或真实 IR 协议时整体失败，不开始后续场景生成。
- 严格 IR preflight 在输出目录创建前完成；不得留下“部分正式”结果。
- 离线 evaluator 在写结果前完成目录、run metadata、frame ID、shape、索引长度和参数校验。
- 缺 prediction/target/radar、重复或未知预测文件、非连续/错配 frame、非法 shape、非有限数值和非空输出目录均明确失败。
- raw LiDAR 索引越界或缺文件时失败，不回退为排序位置。
- 不删除失败现场、数据、checkpoint 或历史输出。

## 测试策略

### 真实 IR 契约

在临时目录创建 radar、IR 和真实/缺失 thermal 外参，验证：

- 严格模式加载真实 IR，mock flags 均为 0；
- 缺 IR、IR symlink、非法 shape/非有限值、缺 thermal 外参均失败；
- 真实 thermal 外参得到与 Dataset 相同的 `0.01m` legacy 同步位移；
- 多模态 checkpoint + 缺 meta 在正式策略下失败；
- 单模态 checkpoint 不能通过 `--require_real_ir`；
- 兼容模式仍保留现有 mock 行为。

### 部署/评价边界

静态协议测试验证三个正式 launcher：

- 使用 sensor-aware 根和 `--require_real_ir`；
- 不出现 target/LiDAR 评价参数；
- 输出目录使用 `_deploy`；
- 每次部署写出包含实际网格和阈值的 `inference_run.json`；
- manifest validate 仍在生成前；
- 新 eval launcher 只调用 `evaluate_saved_predictions.py`，不引用 checkpoint 或 `inference.py`。

用两帧临时小体素和 `inference_run.json` 验证离线 evaluator 的 frame 配对、实际网格传播、固定阈值指标、raw LiDAR 索引、JSON 标记、缺文件和非空输出保护。测试不得加载正式 checkpoint 或运行模型采样。

最终运行相关 unit tests、Python 编译、五份 shell 语法检查、真实 launcher 静态失败验证和 `git diff --check`。不运行正式推理、全量评价、训练或预处理。

## 研究协议影响

- 不改变训练监督、target、网格尺寸、输入帧数量、模型结构或 checkpoint。
- 正式多模态预测会从“IR 关闭或 mock”改为真实 IR，且真实投影补齐训练协议中的固定同步位移，因此预测 voxel 值、占据点数和指标可能改变；旧正式推理结果不可与新协议直接合并。
- 评价算法保持固定阈值，不允许 target 改变预测；评价对象明确为部署阶段保存的同一批 voxel。
- runtime CSV 与 evaluation CSV 分离，旧 `inference_metrics.csv` schema 只作为兼容模式保留。
- 当前真实 sensor-aware 数据缺严格 manifest，正式 launcher 在重新干净预处理前仍会被上一项安全 gate 阻断；本项不补签或修改旧数据。

## 非目标和已知限制

- 不解析 `calib_cam_thermal.txt` 的真实 K/D，不做去畸变或 resize 后内参缩放；
- 不实现 IR 过曝、模糊、低对比度或温度动态范围质量门控；
- 不实现缺 IR 时 Radar-only 自动降级；正式模式选择 fail-closed；
- 不建立 Radar+IR 在线流专用 manifest；
- 不修复固定速度/时间差、Doppler、target 独立性或 checkpoint 正式链；
- 不实现 ROS/PX4、地图更新或避障接口；
- 不运行长时间任务。
