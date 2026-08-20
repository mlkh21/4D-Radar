# Radar 物理通道规范化与统计方差重采样实施计划

> **For Codex:** REQUIRED SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目标：** 为正式多模态 LDM/CD 建立冻结的 Radar 归一化协议，修复 Doppler 方差下采样的二阶矩合并，并让训练、蒸馏、推理和 checkpoint 链对同一 artifact fail-closed。

**架构：** 保留现有四通道原始体素和 target 协议，新增一个纯函数协议模块；Dataset、统计生成器和逐文件推理共用 Radar 专用 resize/normalize 边界。LDM/CD checkpoint 嵌入完整 spec 与 artifact 文件 SHA-256，CD 和正式 checkpoint 链校验继承关系，推理只消费 checkpoint 内嵌协议。正式多模态 Radar 只进入专用 `radar_encoder`，不再经过 target VAE；legacy 单模态行为只能由显式诊断开关启用。

**技术栈：** Python 3、PyTorch、NumPy、标准库 JSON/hashlib/tempfile、unittest、YAML 配置。

**设计规格：** `docs/superpowers/specs/2026-07-22-radar-normalization-variance-resampling-design.md`

## 全局执行约束

- 当前是包含用户既有修改的共享 `withir` 工作区；每一步只编辑本计划列出的相关文件，不暂存、不提交、不推送，不覆盖无关差异。
- 所有生产代码修改先写聚焦 RED 测试，再写最小 GREEN 实现；测试数据只使用 `TemporaryDirectory` 或内存张量。
- 不运行完整预处理、artifact 全量统计、VAE/LDM/CD 训练、完整推理或全量评估。
- `data.radar_normalization_path: ""` 和 `data.doppler_scale_mps: null` 是故意的未配置状态，不得解释为 identity/default。
- 每个任务完成后更新 `TODO/findings.md`、`TODO/task_plan.md`、`TODO/progress.md`，并执行相关聚焦测试与 `git diff --check`。
- 正式入口的协议校验必须早于结果目录、训练日志、CSV、checkpoint 目录的创建；诊断 legacy 开关不得出现在三个正式 inference launcher 中。

---

### Task 1：实现 Radar 专用二阶矩 resize 与归一化纯函数

**文件：**

- 新建：`diffusion_consistency_radar/radar_normalization.py`
- 修改：`diffusion_consistency_radar/cm/dataset_loader.py`
- 新建：`test/unit/test_radar_normalization_protocol.py`

**Step 1：写 Radar resize 的 RED 测试**

在新测试文件中构造最小 `(C,Z,X,Y)` 张量，锁定：

```python
def test_resize_radar_variance_uses_total_variance_formula(self):
    # 两个 occupied 细体素：mean=[1, 3]、local variance=[0, 0]
    # 合并后 mean=2、variance=E[mean^2]-E[mean]^2=1。

def test_resize_radar_variance_combines_local_and_between_voxel_terms(self):
    # 验证 E[var + mean^2] - E[mean]^2，并检查非负 clamp。

def test_resize_radar_keeps_empty_output_channels_zero_and_input_unchanged(self):
    # 输出空体素全零，输入 tensor 不被原地修改。
```

先从 `diffusion_consistency_radar.cm.dataset_loader` 导入尚不存在的 `resize_radar_voxel_channels`，运行：

```bash
conda run -n Radar-Diffusion python test/unit/test_radar_normalization_protocol.py -v
```

预期：因接口不存在而失败，证明测试处于 RED。

**Step 2：实现 Radar 专用 resize**

在 `dataset_loader.py` 中保留现有 `resize_voxel_channels()` 给 target/observed mask 使用，新增：

```python
def resize_radar_voxel_channels(
    voxel_tensor: torch.Tensor,
    target_size,
) -> torch.Tensor:
    """按 occupied 权重合并 Radar 均值和 Doppler 二阶矩。"""
```

实现约束：

- 严格要求 `(4,Z,X,Y)`、有限输入和非负局部 variance；occupied 权重为 `occupancy > 0`。
- occupancy 使用 `adaptive_max_pool3d`。
- intensity mean 与 Doppler mean 使用同一个 occupied-weighted interpolation 分母。
- 先计算 `doppler_second = variance + doppler_mean.square()`，再得到：

```python
merged_variance = merged_second - merged_doppler.square()
merged_variance = merged_variance.clamp_min(0.0)
```

- 分母为空的位置四通道全部写 0；不得原地修改输入。
- 只在 Radar condition、builder 和逐文件 Radar inference 中使用新 helper，绝不替换 target/mask 的通用 helper。

**Step 3：写 normalization loader/apply 的 RED 测试**

同一测试文件增加：

```python
def test_apply_normalization_uses_log_robust_intensity_and_physical_doppler_scale(self): ...
def test_apply_normalization_preserves_occupancy_variance_and_zeroes_empty_voxels(self): ...
def test_loader_returns_full_spec_and_exact_file_sha256(self): ...
def test_loader_rejects_symlink_missing_fields_nonfinite_zero_iqr_and_grid_mismatch(self): ...
def test_loader_rejects_nonformal_artifact_and_scale_mismatch(self): ...
def test_explicit_legacy_metadata_is_nonformal(self): ...
```

预期：纯函数模块不存在，测试失败。

**Step 4：实现协议模块**

新文件写中文文件头，并提供唯一共享接口：

```python
RADAR_NORMALIZATION_PROTOCOL = "radar_normalization_v1"

class RadarNormalizationError(ValueError):
    """Radar 归一化 artifact 或绑定关系不满足协议。"""

def validate_radar_normalization_spec(
    spec,
    *,
    target_size,
    source_pc_range,
    model_pc_range,
    doppler_scale_mps=None,
    require_formal=True,
) -> dict: ...

def load_radar_normalization_artifact(
    path,
    *,
    target_size,
    source_pc_range,
    model_pc_range,
    doppler_scale_mps,
    require_formal=True,
) -> tuple[dict, str]: ...

def apply_radar_normalization(radar_tensor, spec): ...

def assert_same_radar_normalization(
    left_spec,
    left_sha256,
    right_spec,
    right_sha256,
    *,
    context,
) -> None: ...
```

校验要求：

- artifact 必须为非 symlink 普通文件，SHA-256 用分块读取真实文件字节。
- 顶层协议、`formal`、训练场景、帧数、网格/量程、三个 transform、clip、IQR、显式 Doppler scale 和 manifest provenance 都需严格校验。
- 所有布尔值/整数/浮点值做严格类型和有限性检查；`log_iqr > 0`、`scale_mps > 0`。
- `apply_radar_normalization()` 不修改输入：occupied intensity 执行 `log1p(clamp_min(0))` 后 median/IQR/clip；Doppler 按 m/s scale 对称裁剪；occupancy/variance 保持；空体素全零。
- `assert_same...` 同时比较完整 canonical JSON 内容和文件 hash，不接受只匹配其中一项。

**Step 5：运行 Task 1 GREEN 与静态检查**

```bash
conda run -n Radar-Diffusion python test/unit/test_radar_normalization_protocol.py -v
conda run -n Radar-Diffusion python -m py_compile diffusion_consistency_radar/radar_normalization.py diffusion_consistency_radar/cm/dataset_loader.py test/unit/test_radar_normalization_protocol.py
git diff --check
```

预期：新增协议测试全通过；无完整数据读取。

---

### Task 2：实现冻结训练集 artifact 生成器

**文件：**

- 新建：`diffusion_consistency_radar/scripts/build_radar_normalization.py`
- 新建：`test/unit/test_radar_normalization_builder.py`
- 修改：`diffusion_consistency_radar/radar_normalization.py`（只增加原子写入/构造所需的小型共享校验时）

**Step 1：写 builder RED 测试**

用临时 scene 和 mock 后的 `validate_scene_manifest`/小体素覆盖：

```python
def test_builder_uses_only_explicit_training_scenes_and_all_frames(self): ...
def test_builder_stats_follow_crop_radar_resize_and_log1p_order(self): ...
def test_builder_records_validated_manifest_content_sha256(self): ...
def test_frame_cap_marks_artifact_nonformal(self): ...
def test_existing_or_symlink_output_is_rejected_without_replacement(self): ...
def test_invalid_scale_or_empty_occupied_training_set_writes_nothing(self): ...
```

运行：

```bash
conda run -n Radar-Diffusion python test/unit/test_radar_normalization_builder.py -v
```

预期：入口/构造函数不存在而失败。

**Step 2：实现可测试 builder 函数和 CLI**

CLI 明确接收：

```text
--dataset_dir
--scene SCENE（可重复）
--output
--target_size Z X Y
--source_pc_range XMIN YMIN ZMIN XMAX YMAX ZMAX
--model_pc_range XMIN YMIN ZMIN XMAX YMAX ZMAX
--doppler_scale_mps
--max_frames（0 表示全部；正数只允许生成 formal=false 的诊断 artifact）
```

核心函数按以下顺序：

1. 在任何输出文件创建前校验参数、场景集合和输出路径。
2. 对每个显式 train scene 调用 `validate_scene_manifest(scene_dir, scene)`，记录返回值的 `content_sha256`。
3. 只扫描该 scene 的 `radar_voxel` 帧，使用实际 crop + `resize_radar_voxel_channels()`。
4. 仅收集 occupied intensity 的 `log1p` 值，计算 median、q25、q75、IQR；空训练集或非正 IQR 失败。
5. `max_frames == 0` 才写 `formal=true`，否则写 `formal=false` 和实际 frame_count。
6. 目标已存在、是 symlink 或不是普通的新路径时失败；成功时同目录临时文件、flush/fsync、`os.replace` 原子发布。

不得自动猜场景、Doppler scale 或从 test/validation 统计。

**Step 3：验证 builder**

```bash
conda run -n Radar-Diffusion python test/unit/test_radar_normalization_builder.py -v
conda run -n Radar-Diffusion python diffusion_consistency_radar/scripts/build_radar_normalization.py --help
conda run -n Radar-Diffusion python -m py_compile diffusion_consistency_radar/scripts/build_radar_normalization.py test/unit/test_radar_normalization_builder.py
git diff --check
```

预期：只运行临时小数据测试和 `--help`，不对真实数据生成 artifact。

---

### Task 3：接入 Dataset、物理增强后置归一化与配置预检

**文件：**

- 修改：`diffusion_consistency_radar/cm/dataset_loader.py`
- 修改：`diffusion_consistency_radar/cm/augmentation.py`
- 修改：`diffusion_consistency_radar/config/default_config.yaml`
- 修改：`diffusion_consistency_radar/scripts/unified_train.py`
- 修改：`test/unit/test_dataset_protocol_metadata.py`
- 修改：`test/unit/test_radar_normalization_protocol.py`

**Step 1：写 Dataset/增强顺序 RED 测试**

增加用例锁定：

```python
def test_dataset_requires_normalization_unless_legacy_is_explicit(self): ...
def test_dataset_uses_radar_specific_resize_and_normalizes_after_augmentation(self): ...
def test_dataset_metadata_records_protocol_hash_and_legacy_status(self): ...
def test_condition_noise_keeps_occupancy_and_nonnegative_variance_physical(self): ...
```

测试替身让 augmentation 对物理 Doppler 增加一个已知 m/s shift，再断言输出按 `scale_mps` 缩放，证明顺序为：

```text
crop -> Radar resize -> physical augmentation -> normalization
```

先运行两个测试文件，预期新增断言失败。

**Step 2：扩展 Dataset 显式接口**

给 `NTU4DRadLM_VoxelDataset.__init__` 增加：

```python
radar_normalization: Optional[dict] = None,
radar_normalization_sha256: Optional[str] = None,
allow_legacy_radar_units: bool = False,
```

实现要求：

- 默认缺 spec 立即失败；仅 `allow_legacy_radar_units=True` 时允许 identity。
- spec 已在入口预检，但 Dataset 仍校验 target/grid 一致性，防止绕过公共构造器。
- Radar 帧用 `resize_radar_voxel_channels()`；target 与 observed mask 继续用 `resize_voxel_channels()`。
- augmentation 返回后调用 `apply_radar_normalization()`。
- meta 只放稳定的小字段：`radar_normalization_protocol`、`radar_normalization_sha256`、`legacy_radar_units`；完整 spec 由 trainer 持有，避免 DataLoader 嵌套 collate 改变类型。
- legacy meta 固定 `protocol=legacy_identity`、`legacy_radar_units=True`，不得标为 formal。

**Step 3：收紧增强后的物理不变量**

`VoxelAugmentation._add_noise()` 保持 occupancy 不变；condition 特征噪声后将 variance clamp 到非负，空体素仍由 occupancy mask 保持全零。该修改只约束 Radar condition 的非法状态，不改变 target、网格或通道数。

**Step 4：配置和统一训练入口预检**

`default_config.yaml` 在 `data` 下增加：

```yaml
radar_normalization_path: ""
doppler_scale_mps: null
```

`unified_train.py` 新增纯 preflight helper，在创建 Dataset、trainer、日志或结果目录前：

- 对正式路径调用 `load_radar_normalization_artifact()`；
- 仅 CLI 显式 `--allow_legacy_radar_units` 时允许 mini-test/诊断运行，三个正式 launcher 不传该 flag；
- 给 train/val Dataset 传完全相同的 spec/hash；
- `mode=vae` 虽不把 normalization 写入 VAE checkpoint，但因当前共享 Dataset 仍读取 Radar condition，统一使用同一预检结果，不新增隐藏 identity 分支；
- 不在 P1-04 顺带接通尚未生效的 `data.augmentation` YAML（保留给 P2-01）；Dataset 对当前默认增强和直接传入的 `augmentation_config` 都保证后置归一化，validation 不增强。

**Step 5：运行聚焦验证**

```bash
conda run -n Radar-Diffusion python test/unit/test_radar_normalization_protocol.py -v
conda run -n Radar-Diffusion python test/unit/test_dataset_protocol_metadata.py -v
conda run -n Radar-Diffusion python -m py_compile diffusion_consistency_radar/cm/augmentation.py diffusion_consistency_radar/cm/dataset_loader.py diffusion_consistency_radar/scripts/unified_train.py
git diff --check
```

预期：Dataset 协议和原有 metadata 回归通过；不启动训练。

---

### Task 4：绑定 LDM/CD checkpoint、resume 和正式多模态编码路径

**文件：**

- 修改：`diffusion_consistency_radar/scripts/unified_train.py`
- 修改：`diffusion_consistency_radar/scripts/cd_train_optimized.py`
- 修改：`test/unit/test_vae_checkpoint_protocol.py`
- 修改：`test/unit/test_multimodal_cd_training_interface.py`
- 修改：`test/unit/test_cd_training_entrypoints.py`
- 修改：`test/unit/test_multimodal_inference_interface.py`（复用轻量 trainer 替身时）

**Step 1：写 checkpoint 与隐藏 VAE 条件编码 RED 测试**

锁定：

```python
def test_ldm_payload_embeds_full_radar_normalization_and_artifact_hash(self): ...
def test_ldm_resume_rejects_normalization_mismatch_before_state_load(self): ...
def test_multimodal_ldm_train_epoch_does_not_encode_radar_with_target_vae(self): ...
def test_cd_preflight_requires_teacher_and_config_normalization_match(self): ...
def test_cd_payload_inherits_exact_teacher_normalization(self): ...
def test_multimodal_cd_train_epoch_does_not_encode_radar_with_target_vae(self): ...
```

VAE 替身对第二次 `get_latent()` 调用抛错；多模态 batch 应只编码 target 一次，legacy batch 仍显式构造 condition latent。

**Step 2：给 LDM trainer 传递并保存协议**

- `OptimizedLDMTrainer` 构造时接收已经验证的 spec/hash，并保存不可变副本。
- `_checkpoint_payload()` 写入 `radar_normalization` 与 `radar_normalization_sha256`。
- resume preflight 在 `model.load_state_dict()`、`optimizer.load_state_dict()` 之前调用 `assert_same_radar_normalization()`；正式 `main()` 更早读取 resume checkpoint 并在 trainer 日志目录创建前做同一检查。
- 多模态 `train_epoch()` 只计算 `z_target`；仅 `not has_multimodal_meta(meta_dict)` 的显式 legacy batch 才计算 `z_cond`。

**Step 3：给 standalone/unified CD 继承协议**

- standalone CD 在构造 Dataset/save dir/trainer 前读取配置 artifact 和教师 LDM checkpoint，并严格比较 spec/hash。
- `ConsistencyDistillationTrainer` 保存教师 normalization；多模态 `_checkpoint_payload()` 原样写入，legacy checkpoint 保持 `legacy_cd_v0` 且不得写成正式协议。
- resume 在加载模型/EMA/optimizer 前比较 normalization。
- 多模态 CD 训练只编码 target；`train_step()`/`call_cd_denoiser()` 的 legacy `z_cond` 改为可选或只在 legacy 分支强制，禁止用伪造零张量掩盖接口错误。

**Step 4：确认 VAE checkpoint 边界**

VAE payload 不添加 Radar normalization；测试明确断言 VAE 不绑定该字段，但统一入口在构造共享 Dataset 前仍完成配置预检。

**Step 5：运行训练接口回归**

```bash
conda run -n Radar-Diffusion python test/unit/test_vae_checkpoint_protocol.py -v
conda run -n Radar-Diffusion python test/unit/test_multimodal_cd_training_interface.py -v
conda run -n Radar-Diffusion python test/unit/test_cd_training_entrypoints.py -v
conda run -n Radar-Diffusion python test/unit/test_multimodal_inference_interface.py -v
conda run -n Radar-Diffusion python -m py_compile diffusion_consistency_radar/scripts/unified_train.py diffusion_consistency_radar/scripts/cd_train_optimized.py
git diff --check
```

预期：只执行函数级/轻量 CPU 接口测试，不训练模型。

---

### Task 5：绑定正式 checkpoint chain 与 inference embedded spec

**文件：**

- 修改：`diffusion_consistency_radar/checkpoint_chain.py`
- 修改：`diffusion_consistency_radar/scripts/inference.py`
- 修改：`test/unit/test_checkpoint_chain_protocol.py`
- 修改：`test/unit/test_multimodal_inference_interface.py`
- 修改：`test/unit/test_formal_inference_protocol.py`

**Step 1：写 formal chain RED 测试**

扩展 fixture：VAE 保持无 normalization；LDM/CD 使用同一完整 spec/hash。增加拒绝用例：

- LDM 缺 spec 或 hash；
- CD 缺 spec 或 hash；
- spec 内容不同但 hash 相同；
- spec 相同但 artifact hash 不同；
- embedded spec 非正式或网格与 checkpoint 不一致。

有效报告增加 normalization protocol/hash 摘要，但不把 VAE 误判为缺字段。

**Step 2：扩展 checkpoint chain 校验**

在 `validate_formal_checkpoint_chain()` 中复用协议模块：

- 分别验证 LDM/CD embedded spec 与各自 `data_grid_config`；
- 调用 `assert_same_radar_normalization()` 比较 LDM/CD；
- 报告写 `radar_normalization_protocol` 与 `radar_normalization_sha256`；
- 继续聚合错误，不因第一项缺失跳过其他父 hash/网格检查。

**Step 3：写 inference RED 测试**

锁定：

```python
def test_model_loader_retains_and_validates_checkpoint_normalization(self): ...
def test_formal_inference_rejects_missing_normalization_before_output_dir(self): ...
def test_radar_file_loader_uses_embedded_spec_after_shared_radar_resize(self): ...
def test_inference_run_metadata_records_full_spec_hash_and_formal_status(self): ...
def test_legacy_checkpoint_requires_explicit_switch_and_is_marked_nonformal(self): ...
def test_multimodal_generate_derives_shape_without_vae_encoding_condition(self): ...
def test_legacy_generate_still_uses_condition_latent(self): ...
```

**Step 4：实现 inference fail-closed 数据流**

- `RadarGenerator.__init__` 增加 `allow_legacy_radar_units=False`；`_load_model()` 保存完整 checkpoint metadata，再验证/保存 `self.radar_normalization` 与 hash。
- 正式多模态 checkpoint 缺字段立即失败；只有显式 `--allow_legacy_radar_units` 才接受旧 checkpoint，并设置 `formal_protocol=False`。
- `load_radar_voxel_as_tensor()` 增加 spec/hash/legacy 参数，调用 `resize_radar_voxel_channels()` 后归一化；不从本机数据或 CLI 重新估计统计。
- Dataset 推理模式接收 generator 已验证的同一 spec/hash。
- 多模态 `generate()` 用 `condition.shape[0]`、`vae.latent_dim` 和 `vae.latent_spatial_shape(target_size)` 构造采样 shape，不调用 `vae.get_latent(condition)`；legacy 单模态继续显式编码 condition。
- `build_inference_run_metadata()` 写完整 spec、artifact hash、protocol 与 `formal_protocol`。
- parser 增加 `--allow_legacy_radar_units`，但所有协议检查都在首次 `os.makedirs(args.output_dir)` 之前完成。

**Step 5：运行 inference/chain 聚焦验证**

```bash
conda run -n Radar-Diffusion python test/unit/test_checkpoint_chain_protocol.py -v
conda run -n Radar-Diffusion python test/unit/test_multimodal_inference_interface.py -v
conda run -n Radar-Diffusion python test/unit/test_formal_inference_protocol.py -v
conda run -n Radar-Diffusion python -m py_compile diffusion_consistency_radar/checkpoint_chain.py diffusion_consistency_radar/scripts/inference.py
git diff --check
```

预期：全部为小型 fixture/静态协议验证，不加载正式 checkpoint、不执行采样。

---

### Task 6：接线正式/mini launcher，完成代码审查和总回归

**文件：**

- 审查并仅在必要时修改：`diffusion_consistency_radar/launch/train_unified.sh`
- 审查并仅在必要时修改：`diffusion_consistency_radar/launch/inference_ldm.sh`
- 审查并仅在必要时修改：`diffusion_consistency_radar/launch/inference_cd.sh`
- 审查并仅在必要时修改：`diffusion_consistency_radar/launch/inference_uniified.sh`
- 修改：`test/mini-test/train_minimal.sh`
- 修改：`test/mini-test/inference_minimal.sh`
- 修改：`test/unit/test_mini_scripts_protocol.py`
- 修改：`TODO/findings.md`
- 修改：`TODO/task_plan.md`
- 修改：`TODO/progress.md`

**Step 1：写 launcher 协议 RED 测试**

静态断言：

- 正式 train launcher 不传 legacy 开关，归一化 artifact/scale 的权威 fail-fast 由 Python 配置入口完成；
- 三个正式 inference launcher 绝不包含 `--allow_legacy_radar_units`；
- mini train/inference 作为明确诊断入口显式传 legacy 开关，直到其测试数据生成独立非正式 artifact；
- legacy 运行输出命名/metadata 不得冒充 formal。

**Step 2：最小接线**

- 正式训练 shell 不重复解析 YAML/JSON schema，只确保不注入 legacy 开关；Python 在任何训练输出副作用前完成唯一权威校验。
- 正式 inference shell 不增加 artifact 参数，也不增加 legacy 开关；它们只依赖正式 checkpoint embedded spec。
- mini 脚本显式声明其 legacy 诊断身份，避免默认配置空值让 smoke 接口静默变成正式结果。

**Step 3：代码审查清单**

逐项用 `rg` 和差异审查确认：

- 不存在第二套 normalization 数学实现；builder、Dataset、inference 只调用公共函数。
- 不存在硬编码 `80/100m/s` 默认值或从 test/validation 动态估计。
- 通用 target/mask resize 未被 Radar 二阶矩语义污染。
- LDM/CD spec/hash、父 checkpoint hash、网格与 model config 的错误消息可定位具体 stage。
- resume 先比 normalization，再加载模型/EMA/optimizer。
- 正式多模态路径没有 Radar→target VAE；legacy 路径没有因 `None` 接口产生 shape mismatch。
- artifact/checkpoint/推理失败发生在输出目录和日志副作用之前。
- 数据集、checkpoint、日志、实验结果均未被创建、覆盖或删除。

发现接口不匹配时，先在上述最接近的既有测试文件补 RED，再做局部修复，不扩大重构范围。

**Step 4：运行最终聚焦回归**

测试前明确范围：仅 normalization/resize、Dataset metadata、VAE/LDM/CD payload、formal chain、multimodal inference、manifest 和 launcher 静态协议；不运行训练或模型采样。

```bash
conda run -n Radar-Diffusion python test/unit/test_radar_normalization_protocol.py -v
conda run -n Radar-Diffusion python test/unit/test_radar_normalization_builder.py -v
conda run -n Radar-Diffusion python test/unit/test_dataset_protocol_metadata.py -v
conda run -n Radar-Diffusion python test/unit/test_vae_checkpoint_protocol.py -v
conda run -n Radar-Diffusion python test/unit/test_multimodal_cd_training_interface.py -v
conda run -n Radar-Diffusion python test/unit/test_cd_training_entrypoints.py -v
conda run -n Radar-Diffusion python test/unit/test_checkpoint_chain_protocol.py -v
conda run -n Radar-Diffusion python test/unit/test_multimodal_inference_interface.py -v
conda run -n Radar-Diffusion python test/unit/test_formal_inference_protocol.py -v
conda run -n Radar-Diffusion python test/unit/test_dataset_manifest_protocol.py -v
conda run -n Radar-Diffusion python test/unit/test_mini_scripts_protocol.py -v
```

静态验证：

```bash
conda run -n Radar-Diffusion python -m py_compile \
  diffusion_consistency_radar/radar_normalization.py \
  diffusion_consistency_radar/cm/dataset_loader.py \
  diffusion_consistency_radar/cm/augmentation.py \
  diffusion_consistency_radar/checkpoint_chain.py \
  diffusion_consistency_radar/scripts/build_radar_normalization.py \
  diffusion_consistency_radar/scripts/unified_train.py \
  diffusion_consistency_radar/scripts/cd_train_optimized.py \
  diffusion_consistency_radar/scripts/inference.py
bash -n diffusion_consistency_radar/launch/train_unified.sh
bash -n diffusion_consistency_radar/launch/inference_ldm.sh
bash -n diffusion_consistency_radar/launch/inference_cd.sh
bash -n diffusion_consistency_radar/launch/inference_uniified.sh
bash -n test/mini-test/train_minimal.sh
bash -n test/mini-test/inference_minimal.sh
git diff --check
git diff --cached --quiet
```

**Step 5：记录影响和后续显式动作**

在 TODO 三文件记录：

- target、observed mask、occupied 坐标、体素尺寸/数量、四通道模型结构和指标公式未改变；
- Radar condition 的 intensity/Doppler 数值尺度与合并 variance 改变，旧 LDM/CD checkpoint 与新正式协议不兼容；
- variance 增加组间均值差贡献，不确定性输入和最终预测/指标可能变化，新旧结果不可直接混合；
- 本轮没有生成正式 artifact、重建数据、训练模型或运行完整推理；下一步需由用户明确给出真实 `doppler_scale_mps` 后，单独执行全训练场景 artifact 生成和正式重训。
