# LDM 列级正负平衡课程设计

## 背景与目标

v10 A/C/D 已将预测与目标的点数比例控制到约 `1.0`，但固定 32 帧验证中的
BEV recall、trunk recall 和 vertical connectivity 仍然过低。固定的空列负样本约束在训练早期
抑制了真实障碍物列的建立。v11 的目标是先学习障碍物主体，再逐步抑制空背景，
不改变 LiDAR target、体素数量、VAE、模型结构或推理接口。

## 方案比较

1. **按 epoch 线性课程（采用）**：训练前期高正列、低负列，随 epoch 线性过渡到最终权重。
   优点是确定、可恢复、容易测试，3 epoch 屏幕实验可精确复现指定曲线。
2. **按 global step 线性课程**：曲线更平滑，但会受 DataLoader 长度、梯度累积和恢复位置影响，
   增加实验协议复杂度。
3. **按在线 recall 动态调整**：根据 batch 指标调权，但 batch size 1 时噪声大，且会破坏固定
   随机种子下的可解释性，暂不采用。

## 课程定义

新增可选的 epoch 线性课程，默认关闭。对第 `epoch` 轮，总轮数为 `epochs`：

```text
progress = (epoch - 1) / max(epochs - 1, 1)
positive = positive_start + progress * (positive_final - positive_start)
negative = negative_start + progress * (negative_final - negative_start)
```

v11 的配置为：

```text
positive_start = 0.03
positive_final = 0.02
negative_start = 0.00
negative_final = 0.01
```

因此 3 epoch 时实际权重为：

| Epoch | 正列权重 | 负列权重 |
|---:|---:|---:|
| 1 | 0.030 | 0.000 |
| 2 | 0.025 | 0.005 |
| 3 | 0.020 | 0.010 |

当课程关闭时，继续使用现有固定权重，保证历史配置和 checkpoint 行为不变。

## 代码边界

- `unified_train.py`：新增纯函数计算当前 epoch 的有效正/负列权重；LDM trainer
  在每轮开始时计算一次，然后将有效权重传入现有 `compute_ldm_loss_components()`。
- checkpoint 元数据：保存课程开关、起点权重、终点权重和当前 epoch 的有效权重，
  便于恢复和实验审计。
- metrics CSV：增加当轮实际 `column_positive_weight` 和 `column_negative_weight`，
  避免将原始损失值误认为加权贡献。
- mini runner：新增独立、带目录锁和非空结果保护的 v11 training-only 入口，
  显式固定 v10 已审计的所有训练参数。

## 验证与停止条件

1. 单元测试覆盖：课程关闭兼容、三轮精确权重、单 epoch、越界 epoch、非法/非有限参数、
   恢复训练 epoch 映射和 checkpoint 元数据。
2. runner 测试确认：固定 seed 42、500 garden frames、Z64、batch 1、无增强、仅训练 LDM，
   不执行推理或 CD。
3. 只允许短时 finite-gradient smoke 自动运行。正式 3 epoch v11 需要用户明确启动。
4. 正式结果使用原固定 32 帧协议：real IR、20 Euler steps、seed 42、threshold `0.99`。
5. 通过条件不变：BEV IoU >= `0.25`、BEV recall >= `0.80`、top >= `0.10`、
   trunk >= `0.65`、count ratio <= `6.0`。通过前不运行 500 帧、可视化验收或 CD。

## 影响说明

该方案只改变启用 v11 时的 LDM 监督权重时序。LiDAR target、radar/IR 输入、
`64x128x128` 体素数量、VAE latent、网络参数量和推理阈值扫描方式均不变。
预期早期预测点数会增加，但最终仍由相同的 count-ratio 和结构门槛约束。
