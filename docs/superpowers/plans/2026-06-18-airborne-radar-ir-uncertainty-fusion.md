# 机载 Radar-IR 不确定性融合闭环计划

## Summary

当前离线闭环采用 v1 策略：Radar 结构编码、IR 3D 投影条件、不确定性输出、概率地图接入。IR 作为 LDM/CD 去噪条件输入参与生成；Doppler variance、模型不确定性、range、odom covariance 和 calibration confidence 共同影响地图观测可靠度。

## Key Changes

- Radar voxel 4 通道固定为 occupancy/density、intensity、egomotion-compensated Doppler mean、clipped Doppler variance。
- 多模态模型增加 `RadarStructureEncoder` 和 `UncertaintyHead`，保留现有 IR 2D-to-3D frustum projection，暂不引入 BEV cross-attention。
- `CompleteDualModalityPerceptionNet` 支持 `return_uncertainty=True`，默认仍返回 denoised latent，兼容旧训练/推理代码。
- 推理阶段通过 `RadarGenerator.last_uncertainty` 缓存 uncertainty，并在逐文件推理保存 voxel 时同步保存 `*_uncertainty.npy`。
- 概率地图更新支持 `model_uncertainty` 和 `calib_confidence`，观测可靠度由 range、Doppler variance、模型不确定性、odom covariance 和 calibration confidence 联合决定。

## Test Plan

- `test/test_airborne_multimodal_refactor.py` 覆盖 radar encoder shape、uncertainty head 单调性、fusion 输出和 uncertainty 返回。
- `test/test_probabilistic_mapping_uncertainty.py` 覆盖 Doppler variance、range、odom covariance、model uncertainty、calibration confidence 对 belief/reliability 的影响。
- `test/test_multimodal_inference_interface.py` 覆盖新旧 checkpoint 加载和缺失 multimodal meta 的 mock fallback。
- `test/test_multimodal_cd_training_interface.py` 覆盖 CD 多模态调用接口。
- 编译检查覆盖 `multimodal_fusion.py`、`probabilistic_mapping.py`、`inference.py`、`streaming_map_update.py`。

## Assumptions

- 当前优先做离线算法闭环，不做 ROS/PX4/HIL。
- 训练仍使用当前压缩 voxel/latent shape，不直接训练完整 `(600,200,80)` 物理网格。
- mock IR 只能用于通路验证，不能作为真实多模态收益结论。
- 全局 Chamfer 只作为辅助诊断，主看 shared-visible obstacle metrics 和地图更新指标。
