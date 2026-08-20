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
# RTX 4070 Laptop 8 GB：正式数据协议、单阶段保护运行
bash test/mini-test/run_formal_mini_8gb.sh vae

# 历史 legacy mini
bash test/mini-test/train_minimal.sh all
bash test/mini-test/inference_minimal.sh ldm
bash test/mini-test/inference_minimal.sh cd
bash test/mini-test/run_minimal_experiment.sh
bash test/mini-test/diagnose_minimal.sh
```

默认 mini 输出保留在：

```text
test/result/formal_mini_p1_04_8gb_v1/
test/mini-test/train_results_mini/
test/mini-test/inference_results_mini/
test/mini-test/.tmp_mini_train_dataset/
```

`run_formal_mini_8gb.sh` 必须按 `vae → ldm → cd` 分阶段执行，并在阶段间冷却；它保持 full120 正式监督协议，但 checkpoint 标记为 `formal_mini_chain_v1`，不能送入正式 checkpoint 链。完整门禁与推理烟测命令见 `test/mini-test/README.md`。

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
