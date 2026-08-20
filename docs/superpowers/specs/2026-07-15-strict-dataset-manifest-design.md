# 严格 Dataset Manifest 设计

日期：2026-07-15

## 目标

为每个预处理场景生成可移植、可复验的内容级 `dataset_manifest.json`，并让正式训练和正式推理入口在 manifest 缺失、场景不符、文件混用、符号链接或内容 hash 不一致时 fail-closed。

本项只阻止不可信数据进入正式流程，不删除、覆盖、重链或自动补签现有数据。当前混合数据必须在新的输出目录中完成干净预处理后才能恢复正式流程。

## 已确认根因

- 正式训练读取 `Data/NTU4DRadLM_Pre_sensor_aware`，三个正式推理 launcher 读取旧 `Data/NTU4DRadLM_Pre`。
- 两个真实预处理根均没有 `preprocess_policy.json` 或 manifest。
- sensor-aware garden 的 4014 个 radar 文件全部是指向旧根的绝对符号链接。
- sensor-aware loop3 的前 120 个 radar 文件是旧根符号链接，其余 6330 个是普通文件。
- Dataset 当前只按目录和文件名发现样本，policy 缺失时静默返回空字典，无法证明预处理版本一致。

## 修改边界

新增核心模块与命令行入口：

- `diffusion_consistency_radar/dataset_manifest.py`
- `diffusion_consistency_radar/scripts/dataset_manifest.py`
- `test/unit/test_dataset_manifest_protocol.py`

修改预处理与正式 launcher：

- `NTU4DRadLM_pre_processing/NTU4DRadLM_pre_processing.py`
- `diffusion_consistency_radar/launch/train_unified.sh`
- `diffusion_consistency_radar/launch/inference_ldm.sh`
- `diffusion_consistency_radar/launch/inference_cd.sh`
- `diffusion_consistency_radar/launch/inference_uniified.sh`

同步设计、实施计划和三份 TODO 记录。

不修改通用 Dataset 默认行为、mini-test、诊断入口、checkpoint、数据文件或历史结果；不在本项切换正式推理的数据根，也不修改真实 IR 开关。

## Manifest v1 格式

每个场景根目录保存 `dataset_manifest.json`：

```json
{
  "schema_version": 1,
  "scene": "garden",
  "frame_count": 4014,
  "preprocessing": {
    "policy_path": "preprocess_policy.json",
    "policy_sha256": "...",
    "provenance": {
      "preprocess_script": {"name": "NTU4DRadLM_pre_processing.py", "sha256": "..."},
      "calibration": {"name": "calib_radar_to_livox.txt", "sha256": "..."},
      "radar_index": {"name": "radar_index_sequence.txt", "sha256": "..."},
      "lidar_index": {"name": "lidar_index_sequence.txt", "sha256": "..."}
    }
  },
  "modalities": {
    "radar_voxel": [
      {"frame_id": "000000", "path": "radar_voxel/000000.npz", "size": 123, "sha256": "..."}
    ],
    "lidar_voxel": [],
    "target_voxel": [],
    "ir_image": []
  },
  "content_sha256": "..."
}
```

manifest 不写绝对路径、mtime 或生成时间。`content_sha256` 是对除自身外的规范 JSON 内容计算的 SHA-256，因此同一内容复制到其他目录后仍能验证为同一数据集。

## 文件协议

v1 要求四种模态完整存在：

- `radar_voxel/<六位 frame ID>.npy|npz`
- `lidar_voxel/<六位 frame ID>.npy|npz`
- `target_voxel/<六位 frame ID>.npy|npz`
- `ir_image/<六位 frame ID>_ir.npy`

生成和验证都必须满足：

- 四个模态目录存在且非符号链接；
- 所有参与文件都是普通文件，不允许文件级符号链接；
- 文件名严格符合 v1 格式，同一模态不能有重复 frame ID；
- 四个模态的 frame ID 集合完全一致、非空且从 `000000` 连续到最后一帧；
- 没有额外的 NPY/NPZ 文件或未知目录项；
- `preprocess_policy.json` 存在、为普通文件、JSON 为对象，且 `source_scene` 等于 manifest 场景；
- 预处理自动生成时，实际 frame ID 集合必须精确等于 `frames_written` 所描述的范围，防止旧输出残留被纳入新批次。

任何条件不满足都抛出包含场景、模态和具体路径的错误。

## 核心接口

`diffusion_consistency_radar/dataset_manifest.py` 提供纯标准库文件协议函数。它位于包根，避免预处理为使用 manifest 而触发 `cm/__init__.py` 中整套 Torch/模型导入：

```python
build_scene_manifest(
    scene_dir: str,
    scene: str,
    expected_frame_count: int,
    provenance_paths: Mapping[str, str],
) -> dict

write_scene_manifest_atomic(
    scene_dir: str,
    scene: str,
    expected_frame_count: int,
    provenance_paths: Mapping[str, str],
) -> str

validate_scene_manifest(scene_dir: str, expected_scene: str) -> dict
```

SHA-256 采用分块读取。写入使用同目录临时文件、flush 和 `fsync`，再以同文件系统原子硬链接发布正式文件；正式 manifest 已存在时必须失败，不提供覆盖模式。任何失败都清理临时文件，不留下半成品或替换既有 manifest。

验证器重新枚举目录、拒绝 symlink，逐文件核对 path、size、SHA-256、policy hash、规范内容 hash 和 expected scene。它不信任 manifest 中记录的文件清单。

## CLI

`diffusion_consistency_radar/scripts/dataset_manifest.py` 是单一 manifest 管理入口，带中文文件头和两个子命令：

```text
create --scene_dir DIR --scene NAME --expected_frame_count N
       --preprocess_script FILE --calibration FILE
       --radar_index FILE --lidar_index FILE

validate --scene_dir DIR --expected_scene NAME
```

`create` 只用于预处理集成和显式生成；它不会接受 `--allow-symlink`、`--force` 或 legacy adoption 选项。现有混合目录不能通过该入口补签。

`validate` 成功打印场景、帧数和 `content_sha256`，失败以非零状态退出且不创建文件。

## 预处理集成

`process_scene_task()` 在创建模态目录前先检查目标场景目录：只允许目录不存在或为空。非空目录、符号链接或已有 manifest 都立即失败，防止重新预处理覆盖旧数据。

所有 worker 成功并完整写入 `preprocess_policy.json` 后，流程调用 manifest 原子写入，传入：

- 当前预处理脚本；
- 标定文件；
- 当前场景 radar/lidar index 文件；
- 实际 `written` 帧数。

manifest 构建失败时场景预处理失败。主入口不再吞掉场景异常并以 0 退出；任一请求场景失败都令进程返回非零。失败场景可能保留尚未签署的普通输出文件用于诊断，但不会被自动删除、覆盖或视为正式数据。

## 正式入口强制验证

四个正式 launcher 复用 CLI `validate`：

- `train_unified.sh`：在删除/创建 `.tmp_train_dataset` 前，逐个验证配置中的 train scene；全部成功后才创建场景 symlink。
- `inference_ldm.sh`、`inference_cd.sh`、`inference_uniified.sh`：在调用 `inference.py` 前验证每个 test scene。

launcher 不提供跳过验证的环境变量或兼容开关。缺 manifest、旧 schema、场景不符、symlink、额外/缺失文件或 hash 不符均阻断正式流程。

通用 Dataset、直接 Python 调试入口、mini-test 和诊断脚本暂不强制 manifest，保证小型临时数据仍可测试。正式入口的严格性由 launcher 保证，并通过协议测试固定。

## 测试策略

新增 `test/unit/test_dataset_manifest_protocol.py`，使用临时小文件，不读取真实体素：

- 合法四模态场景可生成并验证 manifest；
- manifest 不含绝对路径或 mtime，复制后仍可验证；
- policy 缺失或 `source_scene` 不匹配时失败；
- 任一模态缺帧、额外帧、非连续 frame ID 或未知文件时失败；
- 目录级或文件级 symlink 在生成和验证阶段都失败；
- 文件内容、大小、manifest 字段或 policy 被修改时失败；
- provenance 缺失或为 symlink 时失败；
- 写入失败不遗留临时 manifest，且不覆盖已有正式 manifest；
- 预处理在目标场景目录非空时于 worker 启动前失败；
- CLI create/validate 返回正确状态；
- 预处理源码在 policy 之后调用 manifest，并传播失败；
- 四个正式 launcher 在破坏性临时目录操作或推理调用前验证场景，且不存在跳过开关。

最终运行新增测试、相关既有 dataset/launcher 测试、Python 编译、四个 shell 语法检查、真实现有场景的只读失败验证和 `git diff --check`。不运行预处理、训练或推理。

## 对研究协议的影响

- 不改变监督信号、target 内容、网格尺寸、每帧体素值、模型或 checkpoint。
- 不改变已经生成的数据数量，但当前不可信场景会从“可被正式加载”变为“被正式拒绝”。
- 模型训练/评价指标算法不变；旧数据产生的指标仍保留，但无法通过新 manifest 证明来源一致，不能作为新正式协议结果。
- 后续在新目录重新预处理会改变实际数据内容和体素统计，届时必须单独审计并重新建立指标基线。

## 非目标

- 不自动为 legacy 数据生成兼容 manifest；
- 不允许符号链接数据通过 formal validation；
- 不删除或重建当前 Data 目录；
- 不切换正式推理到 sensor-aware 根；
- 不启用真实 IR 条件；
- 不修改 checkpoint 加载链；
- 不运行长时间任务。
