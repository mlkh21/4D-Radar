# P1-04 Radar 物理通道规范化与统计方差重采样设计

## 1. 目标

修复 Radar 四通道在原始高分辨率体素下采样到模型网格时的两个协议问题：

1. intensity 与 Doppler 没有冻结、可审计的输入量纲；
2. Doppler variance 被当作普通均值插值，缺少细体素均值之间的组间方差。

正式训练、CD 蒸馏和推理必须使用同一份 Radar normalization 协议。历史 checkpoint 缺少该协议时正式入口 fail-closed；legacy 诊断只能通过显式开关继续使用原始量纲。

## 2. 非目标

- 不改写现有 `radar_voxel`、target、LiDAR、IR、checkpoint 或实验结果。
- 不自动运行完整统计、预处理、训练或推理。
- 不改变 occupancy/target 生成、四通道模型接口或体素网格尺寸。
- 不把训练集 Doppler 分位数冒充硬件物理量程。
- 不在本项修复 YAML augmentation 未传入 Dataset 的 P2-01 问题。
- 不增加原始点数通道；方差按 occupied 细体素等权合并，不宣称是原始 Radar 点级加权方差。

## 3. 已确认的根因与边界

原始 Radar 体素通道为：

```text
0 occupancy
1 mean intensity
2 egomotion-compensated mean Doppler (m/s)
3 local Doppler variance ((m/s)^2)
```

`resize_voxel_channels()` 当前对通道 1～3 都执行占用加权插值。对于 variance，这只能得到局部方差的均值，无法得到合并后总体方差。

当前体素没有细体素内原始点数，因此本项采用 occupied 细体素等权的全方差公式：

```text
mean_out = E[mean_local]
second_moment_out = E[variance_local + mean_local^2]
variance_out = max(second_moment_out - mean_out^2, 0)
```

现有数据增强在物理通道上执行 intensity scale、Doppler shift 和噪声；Radar normalization 必须在 resize 和增强之后执行。

## 4. 方案

采用“共享统计 artifact + 数据入口归一化”：

```text
物理单位 Radar voxel
  → 物理范围 crop
  → Radar 专用二阶矩 resize
  → 物理单位 augmentation
  → 共享 Radar normalization
  → Radar encoder / LDM / CD
```

target 继续使用通用 `resize_voxel_channels()`，不应用 Radar normalization。Radar Dataset 与逐文件 inference 使用同一个专用 resize 和 normalization helper。

## 5. Radar 专用重采样接口

新增共享函数：

```python
resize_radar_voxel_channels(
    voxel_tensor: torch.Tensor,
    target_size: Sequence[int],
) -> torch.Tensor
```

要求：

- 输入为严格四通道 `(4,Z,X,Y)` 且全部有限；模型与 artifact 协议不接受语义未知的额外通道；
- occupancy 继续使用 adaptive max pooling；
- intensity 和 Doppler mean 使用与当前实现一致的 occupancy-weighted interpolation；
- variance 使用同一权重对 `variance + mean^2` 插值，再减去输出 mean 的平方；
- 空输出体素的 intensity、Doppler 和 variance 都为 0；
- variance 对浮点误差执行 `clamp_min(0)`，不在此处改变 `(m/s)^2` 单位；
- Dataset、inference 和统计生成器不得复制该公式。

## 6. Normalization artifact

新增纯数据协议模块和独立生成入口。artifact 文件名不强制，但内容必须满足 `radar_normalization_v1`：

```json
{
  "protocol": "radar_normalization_v1",
  "formal": true,
  "training_scenes": ["garden"],
  "frame_count": 4014,
  "target_size": [64, 128, 128],
  "source_pc_range": [0, -20, -6, 120, 20, 10],
  "model_pc_range": [0, -20, -6, 40, 20, 10],
  "intensity": {
    "transform": "log1p_robust_zscore",
    "log_median": 0.0,
    "log_iqr": 1.0,
    "clip": [-5.0, 5.0]
  },
  "doppler": {
    "transform": "symmetric_physical_scale",
    "scale_mps": 100.0,
    "clip": [-1.0, 1.0]
  },
  "variance": {
    "transform": "identity",
    "unit": "m2_s2",
    "aggregation": "occupied_voxel_equal_weight_total_variance"
  },
  "input_provenance": {
    "dataset_manifest_sha256": {
      "garden": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    }
  }
}
```

示例中的数值只说明字段类型，不是项目默认值。生成器必须显式收到正有限 `doppler_scale_mps`；代码和默认配置不得猜测 80/100m/s。

artifact 生成规则：

- 只读取显式列出的训练场景，不自动包含 test/validation 场景；
- 按模型实际 `source_pc_range/model_pc_range/target_size` crop 和 Radar 专用 resize 后统计 occupied 输入；
- intensity 先执行 `log1p(clamp_min(value,0))`，再保存 median 和 `q75-q25`；IQR 必须为正有限数；
- 默认扫描全部训练帧；若为测试设置帧数上限，artifact 写入 `formal=false`，正式 loader 拒绝；
- 正式 artifact 要求每个训练场景都有已验证的 dataset manifest，并记录各 manifest 的 SHA-256；
- 目标文件已存在或是 symlink 时拒绝覆盖；允许创建不存在的父目录，成功后原子发布 JSON。

## 7. 运行时变换

共享函数只处理 occupied Radar 体素：

```text
intensity_norm = clip(
  (log1p(max(intensity, 0)) - log_median) / log_iqr,
  intensity_clip_min,
  intensity_clip_max
)

doppler_norm = clip(
  doppler_mps / doppler_scale_mps,
  -1,
  1
)
```

空体素四个通道全部保持 0；occupancy 和 variance 不变。所有输入和统计字段必须为有限数，协议、shape、网格或量程不匹配时在创建训练/推理输出之前失败。

## 8. 配置、Dataset 与 legacy 边界

正式配置增加：

```yaml
data:
  radar_normalization_path: ""
  doppler_scale_mps: null
```

空值用于提示尚未配置，不代表 identity 默认。正式训练要求两项都存在，且 YAML 的 `doppler_scale_mps` 必须与 artifact 完全一致。

`NTU4DRadLM_VoxelDataset` 接收已经验证的 normalization spec。没有 spec 时默认拒绝；测试、旧诊断或迁移工具只有显式设置 `allow_legacy_radar_units=True` 才能保持原始量纲。legacy 标志必须进入 sample metadata，不能伪装成正式输入。

归一化在 augmentation 之后只作用于 Radar condition；target 保持物理/监督协议不变。

现有 condition 噪声增强不得破坏该物理边界：occupancy 不参与高斯噪声，variance 在增强后保持非负，空体素继续为零。该约束只修复 Radar condition 的非法状态，不改变 target 增强、体素网格或通道数量；`data.augmentation` YAML 尚未接线的问题仍保留给 P2-01。

正式多模态 LDM/CD 中，Radar condition 只进入专用 `radar_encoder`，不得再调用用于 target 的 `VAE3D.get_latent()`。训练阶段的潜变量 shape 来自 `z_target`，推理阶段由已加载 VAE 的公开 shape 接口推导；只有显式 legacy 单模态诊断路径继续构造 `z_cond`。这既消除无效计算，也避免把归一化 Radar 四通道误当成 target 四通道送入另一套隐形编码语义。

## 9. Checkpoint 与推理绑定

LDM 和多模态 CD checkpoint 新增：

```text
radar_normalization
radar_normalization_sha256
```

要求：

- LDM 保存实际 Dataset 使用的完整 spec 和 artifact 文件 SHA-256；
- CD 启动时校验配置 artifact 与教师 LDM 完全一致，并把相同字段写入 CD checkpoint；
- formal checkpoint chain 要求 LDM/CD 两项均存在且内容/hash 相同；
- VAE 不消费 Radar，不要求该字段；
- resume 时 normalization 不一致必须在加载 optimizer/model 状态前失败。

inference 从实际 LDM/CD checkpoint 取得 embedded spec，不从当前数据目录重新估计，也不允许 CLI 静默覆盖。逐文件 Radar 输入经共享 resize/normalize helper 后送入模型；`inference_run.json` 记录完整 spec 和 hash。

显式 legacy 推理只能用于旧 checkpoint 诊断，运行 metadata 必须标记：

```text
radar_normalization_protocol = legacy_identity
formal_protocol = false
```

正式 launcher 不暴露 legacy 开关。

## 10. 失败与副作用顺序

以下检查必须发生在创建输出目录、训练日志或 checkpoint 之前：

- artifact 是普通文件而非 symlink，且 JSON 可解析；
- protocol、训练场景、网格、transform、clip、IQR、Doppler scale 和 provenance 合法；
- YAML scale 与 artifact 一致；
- checkpoint embedded spec/hash 与 artifact 或父 checkpoint 一致；
- Radar tensor shape 与有限值合法。

任何失败都不得生成半成品 normalization artifact、训练目录或 inference output。

## 11. 测试策略

采用 RED/GREEN：

1. 两个细体素 Doppler mean 不同但局部 variance 为 0，合并后 variance 必须包含组间方差；
2. 局部 variance 非零时验证完整二阶矩公式、空体素为零和输入不变性；
3. 验证 intensity `log1p+median/IQR`、Doppler 对称缩放/裁剪、variance/occupancy 不变；
4. artifact loader 拒绝缺字段、非有限值、零 IQR、非法 scale、非正式抽样和网格不匹配；
5. Dataset 在物理增强后归一化，训练与 inference 调用同一 helper；
6. LDM/CD payload 保存 spec/hash，checkpoint chain 拒绝缺失或不一致；
7. inference run metadata 固化实际协议，正式入口拒绝 legacy checkpoint；
8. 正式多模态训练/推理不再通过 target VAE 编码 Radar，而 legacy 单模态分支仍可显式使用 `z_cond`；
9. 回归 Dataset metadata、Airborne 多模态、CD/VAE checkpoint、formal inference 和 manifest 协议。

## 12. 监督、体素与指标影响

- target、occupancy observed mask、occupied 体素坐标和体素总数不变；
- Radar intensity/Doppler condition 数值分布改变，LDM/CD 必须重新训练，旧 checkpoint 不兼容；
- 合并后 variance 会增加细体素均值差异贡献，不确定性头和概率地图的输入可能显著增大，但仍保持物理单位；
- 模型结构的四输入通道数量不变；
- 预测值、占据数量和所有下游指标可能变化，新旧 normalization 协议结果不得直接混合汇总。
