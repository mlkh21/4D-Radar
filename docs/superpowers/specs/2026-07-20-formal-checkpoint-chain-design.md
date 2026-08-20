# 正式 VAE/LDM/CD checkpoint 链设计

## 目标

将正式部署可接受的权重固定为一条可审计、可复现的 `VAE → LDM → CD` 链。当前 `Result/train_results` 中的旧权重不能被改写或伪装成新链；在新链生成前，正式入口必须明确失败并给出重训提示。

## 边界

- 只校验普通文件、checkpoint 元数据、state dict 关键结构、父 checkpoint SHA-256 和空间网格一致性。
- 不训练、不覆盖已有 checkpoint，不创建 symlink 作为正式权重，不把历史 legacy CD 自动迁移为多模态 CD。
- 校验脚本独立于推理采样；默认只在 CPU 上安全读取 checkpoint，不加载数据、不创建正式输出目录。
- 可选 `--construct` 模式在 CPU 上按保存的 config 构建三阶段模型并严格加载；正式 launcher 只使用无模型构建的元数据门禁。

## 正式协议 v1

每个新保存的 checkpoint 都必须包含：

```text
checkpoint_protocol: "formal_chain_v1"
stage: "vae" | "ldm" | "cd"
data_grid_config:
  target_size: [Z, X, Y]
  source_pc_range: [xmin, ymin, zmin, xmax, ymax, zmax]
  model_pc_range: [xmin, ymin, zmin, xmax, ymax, zmax]
```

LDM/CD 还必须保存 `model_config`，其中包含 `latent_dim`、`in_channels`、`out_channels`、网络结构参数和 `fusion_voxel_shape`、`fusion_latent_shape`、`fusion_pc_range`；其 `fusion_voxel_shape` 必须等于 VAE 的 `target_size`，`fusion_pc_range` 必须等于 VAE 的 `model_pc_range`。

LDM/CD 必须保存 `vae_checkpoint_sha256`；CD 还必须保存 `ldm_checkpoint_sha256`。LDM/CD state dict 必须含实际持久化的多模态网络前缀（`radar_encoder.`, `model_uncertainty_head.`, `ir_extractor.`, `fusion_conv.`）；投影几何由 `fusion_*` 配置保存，不依赖 `persistent=False` 的几何 buffer，以拒绝旧单模态/legacy CD。

## 失败策略

校验以聚合错误一次性报告缺失字段、父 hash 不匹配、网格不一致、文件类型非法和多模态关键权重缺失。任何错误都返回非零；正式 launcher 在第一帧生成前执行门禁。

## 监督与指标影响

本项不改变 target、监督信号、体素数量、模型前向或评价指标；只改变新 checkpoint 的保存元数据以及正式入口对权重链的接受条件。现有旧结果仍可用于 legacy 诊断，但不能与正式链结果混称。
