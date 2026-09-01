# test 目录说明

本目录存放回归测试、正式评估、问题诊断、消融实验、可视化脚本和实验结果。

## 实际目录结构

```text
test/
├── unit/                  # 单元、接口和 Shell 协议测试
├── evaluation/
│   ├── vae/               # VAE IoU/重建评估
│   ├── ldm/               # LDM 结构评估和 checkpoint 选择
│   └── comparison/        # radar/LiDAR/target 比较指标
├── diagnostics/
│   ├── alignment/         # 坐标、配准和共享可见性诊断
│   ├── occupancy/         # 占用阈值与 oracle 协议诊断
│   ├── radar/             # 雷达坐标轴约定诊断
│   ├── infrared/          # 红外输入诊断预留目录
│   └── vertical_structure/# 垂直结构问题定位预留目录
├── ablation/              # 红外条件及模型损失消融
├── visualization/         # 点云和推理结果 HTML 可视化
├── utils/legacy/          # 一次性或历史脚本
├── mini-test/             # 小规模训练、推理和端到端验证
└── result/                # 按模型和用途分类的结果、checkpoint 和日志
```

`test/AGENTS.md` 是新增文件、移动文件和结果登记的主要规则。历史实验叶目录名称保持不变。

## 正式评估入口

### VAE

当前安全的 VAE 重建评估入口是项目正式脚本：

```bash
conda run -n Radar-Diffusion python \
  diffusion_consistency_radar/scripts/diagnose_vae_reconstruction.py --help
```

`test/evaluation/vae/check_IoU_vae.py` 保留了旧的 IoU 评估逻辑，但导入时会加载模型并执行任务，不支持安全的 `--help`；只能在明确设置环境变量并完成兼容改造后使用。

### LDM

保存结果的垂直结构评估：

```bash
conda run -n Radar-Diffusion python \
  test/evaluation/ldm/evaluate_ldm_vertical_structure.py --help
```

固定验证协议的 checkpoint 选择：

```bash
conda run -n Radar-Diffusion python \
  test/evaluation/ldm/select_ldm_checkpoint.py --help
```

当前没有独立的正式 CD 指标评估脚本；CD 的小规模入口仍是：

```bash
bash test/mini-test/inference_minimal.sh cd
```

## 诊断入口

坐标和配准诊断：

```bash
conda run -n Radar-Diffusion python \
  test/diagnostics/alignment/alignment_sanity_check.py --help
conda run -n Radar-Diffusion python \
  test/diagnostics/alignment/shared_visibility_eval.py --help
```

LiDAR→body 与逐帧 body→local 候选诊断：

```bash
conda run -n Radar-Diffusion python \
  test/diagnostics/alignment/build_mapping_pose_candidates.py \
  --radar_to_imu_matrix Data/config/calib_radar_to_imu.txt \
  --radar_to_lidar_calib Data/config/calib_radar_to_livox.txt \
  --ground_truth Data/NTU4DRadLM/loop3/gt_odom.txt \
  --radar_sync_csv Data/NTU4DRadLM_Pre_sensor_aware_p1_04_candidate/loop3/radar_ir_sync.csv \
  --radar_lidar_sync_csv Data/NTU4DRadLM_Raw_p1_01_candidate/loop3/radar_lidar_sync.csv \
  --pose_reference_sensor lidar \
  --output_dir <全新的诊断结果目录> \
  --max_interpolation_gap_s 0.2
```

该入口只发布 `formal=false` 候选，同时保留 GT-as-IMU 与 GT-as-LiDAR
两种未决假设。LiDAR 对齐数据必须显式选择 `lidar` 时间参考；v2 会把
Radar--LiDAR sync 快照及哈希封存在候选目录。正式地图加载器会按文件内容
拒绝这些候选，不得通过放宽插值间隔、复制首 pose 或修改文件名冒充部署合同。

跨帧 LiDAR 重合反证：

```bash
conda run -n Radar-Diffusion python \
  test/diagnostics/alignment/evaluate_mapping_pose_overlap.py \
  --processed_scene_dir Data/NTU4DRadLM_Pre_sensor_aware_p1_04_candidate/loop3 \
  --candidate_dir <candidate v2 目录> \
  --output_dir <全新的 overlap 诊断目录> \
  --pair_delta_s 1.0 \
  --min_rotation_deg 3.0 \
  --max_pairs 48
```

该入口验证 candidate、manifest 自哈希和所选 LiDAR 体素收据，并比较两种
GT frame 假设的双向最近邻残差。结果始终为 `formal=false`；GT-as-LiDAR
分支会使 LiDAR→body 外参代数消去，因此该指标不能确认 Radar→IMU 方向。

经验 LiDAR→local 离线地图合同：

```bash
conda run -n Radar-Diffusion python \
  diffusion_consistency_radar/scripts/build_empirical_lidar_pose_contract.py \
  --candidate_dir test/result/comparison/alignment_check/mapping_pose_contract_loop3_candidate_v2_lidar_time \
  --overlap_dir test/result/comparison/alignment_check/mapping_pose_frame_overlap_loop3_diagnostic_v2_lidar_time \
  --output_dir <全新的经验位姿合同目录>
```

该入口只接受已经相互绑定的 LiDAR-time candidate 与 overlap 诊断，直接发布
`T_local_lidar`，并把来源快照、逐帧 pose 与 SHA-256 封装成自包含 receipt。
它只允许 `offline_empirical_mapping=true`，明确写入 `airborne_formal=false`、
`avoidance_formal=false`；无位姿覆盖帧保持 uncovered，禁止外推。

新正式推理完成后，可在 fresh 输出目录执行离线经验地图回放：

```bash
conda run -n Radar-Diffusion python \
  diffusion_consistency_radar/scripts/streaming_map_update.py \
  --radar_voxel_dir <新正式推理目录> \
  --radar_voxel_layout czxy \
  --offline_empirical_mapping \
  --empirical_pose_receipt test/result/comparison/alignment_check/empirical_lidar_pose_loop3_v1/empirical_pose_receipt.json \
  --inference_run <新正式推理目录>/inference_run.json \
  --observed_mask_dir <新正式推理目录> \
  --output_dir <全新的离线地图结果目录> \
  --pc_range 0 -20 -6 80 20 10 \
  --map_pc_range 0 -20 -6 120 20 10
```

严格地图会按 receipt 选择 6165 个有 pose 的帧，并重算实际消费的 prediction
voxel 与 observed mask 哈希。旧推理目录缺少 `prediction_voxel` 内容收据，必须
重新推理生成 metadata；不得手工补写 JSON 或删除 267 个 uncovered 文件绕过门禁。

新 prediction artifact 协议为 `generated_voxel_artifact_v2`：地图只消费 ch0
occupancy probability，并要求其位于 `[0,1]`；ch1--3 不得解释为 Radar 方差或
DEM 高度不确定性。formal/经验地图都把逐层 observed mask 作为权威域，mask 外的
正预测继续保持 unknown。`map_run.json` 会记录 prediction mapping contract，以及
DEM mean 为米、DEM variance 为平方米的单位合同。

严格/经验地图现使用随 body/LiDAR 原点移动的整体素 rolling
window；`--map_pc_range` 在这两种模式中表示锚点相对窗口，
`map_run.json` 同时记录最终 local bounds。需要轨迹走廊时，可额外传入：

```json
{
  "protocol": "local_trajectory_frames_v1",
  "coordinate_frame": "local",
  "frame_count": 1,
  "records": [
    {
      "frame_id": "000001",
      "waypoints_local_m": [[1.0, 0.0, 0.0], [20.0, 0.0, 0.0]]
    }
  ]
}
```

使用 `--trajectory_file <JSON> --trajectory_corridor_radius_m <m>
--trajectory_sample_spacing_m <m>` 启用走廊查询。artifact 的帧集和顺序必须与实际
消费帧完全一致，轨迹不足制动距离或走廊包含 unknown 均返回风险。
该脚本仍仅产生 NPZ/JSON/CSV 离线 artifact；ROS1 node/service/action、PX4 bridge
和在线时延验收尚未实现，因此 formal/empirical 输出均不得声称为机载避障正式结果。

雷达轴约定诊断：

```bash
conda run -n Radar-Diffusion python \
  test/diagnostics/radar/check_radar_axis_conventions.py --help
```

红外条件消融：

```bash
conda run -n Radar-Diffusion python \
  test/ablation/diagnose_ir_condition_ablation.py --help
bash test/mini-test/run_ldm_z64_v7_target_ablation.sh
```

垂直结构评估：

```bash
conda run -n Radar-Diffusion python \
  test/evaluation/ldm/evaluate_ldm_vertical_structure.py --help
```

数据协议审计：

```bash
conda run -n Radar-Diffusion python \
  diffusion_consistency_radar/scripts/audit_dataset_protocol.py \
  --dataset_root Data/NTU4DRadLM_Pre_sensor_aware \
  --output_dir test/result/comparison/dataset_protocol_audit_v7
```

训练域与独立验证域的稀疏体素、Doppler 和 IR 分布审计：

```bash
conda run -n Radar-Diffusion python \
  test/diagnostics/radar/audit_scene_distribution_shift.py \
  --dataset_root Data/NTU4DRadLM_Pre_sensor_aware \
  --scenes garden,loop3 \
  --max_frames 500 \
  --output_dir test/result/comparison/scene_distribution_audit_v11
```

该脚本在物理坐标中直接读取稀疏 NPZ，不构造完整稠密体素，也不修改训练数据。

逐帧 target 数量匹配的 oracle 上限诊断：

```bash
conda run -n Radar-Diffusion python \
  test/diagnostics/occupancy/diagnose_oracle_target_adaptation.py \
  --pred_voxel_dir <固定阈值推理输出目录> \
  --target_voxel_dir <对应场景的target_voxel目录> \
  --output_dir <全新的oracle诊断输出目录>
```

该脚本读取已保存的 `*_voxel.npy`，输出逐帧 oracle 阈值、CSV、JSON 和
`XYZ+intensity` 点云。结果使用测试 target 改变逐帧输出，只能用于上限诊断，不能作为正式推理或部署性能；输出目录已存在且非空时脚本会拒绝覆盖。

## 可视化入口

```bash
conda run -n Radar-Diffusion python \
  test/visualization/generate_interactive_raw_compare.py --help
conda run -n Radar-Diffusion python \
  test/visualization/generate_interactive_inference_compare.py --help
```

## Mini-test 入口

```bash
# RTX 4070 Laptop 8 GB：先做不训练的 formal v2 只读预检
MINI_PREFLIGHT_ONLY=1 bash test/mini-test/run_formal_mini_8gb.sh vae

# 用户确认后才启动单阶段 VAE mini
bash test/mini-test/run_formal_mini_8gb.sh vae

# 已验收 1 epoch smoke 后，可先预检独立 3 epoch VAE short profile
MINI_PREFLIGHT_ONLY=1 bash test/mini-test/run_formal_mini_8gb.sh vae short_train
# 用户确认后才启动；不会覆盖上面的 smoke
bash test/mini-test/run_formal_mini_8gb.sh vae short_train

# RTX 4070 Laptop：500 个唯一帧（400 train + 100 validation）的 20 epoch 中型筛查
MINI_PREFLIGHT_ONLY=1 bash test/mini-test/run_formal_mini_8gb.sh vae medium_train
# 预检、散热和空闲显存确认后，才显式启动单阶段
bash test/mini-test/run_formal_mini_8gb.sh vae medium_train

# 历史 legacy mini
bash test/mini-test/train_minimal.sh all
bash test/mini-test/inference_minimal.sh ldm
bash test/mini-test/inference_minimal.sh cd
bash test/mini-test/run_minimal_experiment.sh
bash test/mini-test/diagnose_minimal.sh
```

默认 mini 输出保留在：

```text
test/result/formal_mini_v2_80m_8gb_v1/
test/result/formal_mini_v2_80m_8gb_short_v1/
test/result/formal_medium_v2_80m_laptop_500f_20ep_v2/
test/mini-test/train_results_mini/
test/mini-test/inference_results_mini/
test/mini-test/.tmp_mini_train_dataset/
```

`formal_medium_v2_80m_laptop_500f_20ep_v1/` 是 epoch 1 第 50 个 batch
触发 CUDA expandable-segment 内部断言的失败现场，没有 checkpoint；目录保留用于诊断，
不得续训。修复后的 `medium_train` 固定写入 fresh `v2` 目录。

`run_formal_mini_8gb.sh` 必须按 `vae → ldm → cd` 分阶段执行，并在阶段间冷却。
它使用 formal v2 0--80 m 数据合同，默认从正式 split 读取 8 个 train 和 4 个
validation 帧；checkpoint 标记为 `formal_mini_chain_v2`，只能用于工程 smoke，不能送入
正式 checkpoint/deployment 链。完整门禁与 1 帧推理命令见 `test/mini-test/README.md`。
`short_train` 仅用于 fresh 的 3 epoch VAE 趋势检查，使用独立结果根和 60/75°C
启动/运行温度门禁，不代表正式训练。
short VAE 后续 LDM 必须用 `MINI_RESULTS_DIR=test/result/formal_mini_v2_80m_8gb_short_v1`
显式复用父权重，并先运行 `MINI_PREFLIGHT_ONLY=1`；预检会在零输出条件下校验父
checkpoint 的 stage/protocol/data identity。完整命令见 `test/mini-test/README.md`。

`medium_train` 固定 batch 1、worker 0、400/100 帧和 VAE/LDM/CD 各 20 epoch，不能通过
环境变量放宽为其他样本数或 epoch。它在 RTX 4070 Laptop 上按阶段运行，启动/运行温度
上限为 55/72°C、启动空闲显存至少 6500 MiB、单阶段最多 180 分钟。VAE/LDM 会消费
100 帧 validation；CD 当前只用 400 帧训练，留出的 100 帧在 CD 完成后独立评价。
该 profile 还会严格拒绝设备名不是 `NVIDIA GeForce RTX 4070 Laptop GPU` 的 GPU。
该档用于较强的本地质量筛查，正式服务器仍使用 3210/774 full split、每阶段 20 epoch。

## 结果保存规则

正式结果统一保存到 `test/result/`，并按用途分类：

```text
test/result/vae/evaluation/
test/result/vae/reconstruction/
test/result/vae/diagnostics/
test/result/vae/overfit/
test/result/ldm/evaluation/
test/result/ldm/vertical_structure/
test/result/ldm/ablation/
test/result/ldm/visualization/
test/result/comparison/alignment_check/
test/result/comparison/dataset_protocol_audit_v7/
test/result/comparison/scene_distribution_audit_v11/
test/result/archive/
```

当前已整理的结果叶目录包括：

```text
test/result/vae/reconstruction/vae_near40_500_v2/
test/result/ldm/ablation/ldm_near40_500_z64_v10a_seeded_recheck/
test/result/comparison/alignment_check/near40_raw_lidar_compare/
test/result/comparison/dataset_protocol_audit_v7/
test/result/archive/ldm_sensor_aware_partial_20260713/
```

LDM v5-v10 的消融、训练和固定验证结果统一位于：

```text
test/result/ldm/ablation/
├── ldm_near40_500_z64_v5_empty_column/
├── ldm_near40_500_z64_v6_top/
├── ldm_near40_500_z64_v7_ir/
├── ldm_near40_500_z64_v8_balanced/
├── ldm_near40_500_z64_v9a_top_screen/
├── ldm_near40_500_z64_v9a_top_full/
├── ldm_near40_500_z64_v9b_irneg_screen/
├── ldm_near40_500_z64_v10a_column_screen/
├── ldm_near40_500_z64_v10b_column_screen/
├── ldm_near40_500_z64_v10c_pos003_screen/
├── ldm_near40_500_z64_v10d_neg0005_screen/
└── ldm_near40_500_z64_v10a_seeded_recheck/
```

每个实验叶目录下的 `ldm/` 保存训练 checkpoint、`metrics.csv` 和 `training.log`；已有的
`loop3_ldm_eval/`、`vertical_structure_eval*/`、`raw_lidar_visuals/`、`checkpoint_selection*/`
和 `cd/` 分别保存推理、诊断、可视化、checkpoint 筛选和 CD 结果。具体参数、状态和推荐性以
`test/result/INDEX.md` 为准。

CD 结果必须作为对应 VAE/LDM 实验的 `cd/` 子目录保存；锁目录必须紧邻所属实验，临时 `.tmp_*` 目录必须位于所属实验或 archive 叶目录中。根级 `test/result/cd/`、`test/result/visualization/` 和未归属 `.tmp_*` 不再作为正式结果位置。

临时结果必须使用独立叶目录，不得覆盖已有 checkpoint、日志、metrics、CSV、JSON、图片、点云、NPZ 或 HTML。实验叶目录名称不得为了格式统一而批量重命名。

## 新增文件规则

新增或移动 `test/` 文件前必须完整阅读 `test/AGENTS.md`。新增入口必须使用项目相对路径或 `pathlib.Path` 推导路径，并在 `test/result/INDEX.md` 登记正式实验。
