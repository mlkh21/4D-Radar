<!-- 文件功能：记录当前审查问题按优先顺序实施的长期修复计划。 -->
# 当前审查问题顺序修复计划

## 目标

按已确认的八个阶段依次修复训练配置、LDM observed-mask、多卡状态、推理评价、概率建图、滚动地图、新预处理协议和工程清理问题；每阶段先建立最小回归，再实施并审查接口。

## 当前阶段

八个阶段均已完成；等待服务器短时多卡 smoke 与后续正式训练/评价。

## 阶段

1. [已完成] 统一 GPU 配置唯一来源、修正 YAML 测试，并显式隔离 legacy-only VAE 参数。
2. [已完成] LDM observed-mask 训练、decoded loss、验证指标和 latent 监督合同。
3. [已完成] 多卡 IR normalization 与 CD EMA parameter/buffer 一致性。
4. [已完成] 推理 online/EMA 选择、阈值 artifact、observed-mask evaluator、seed 与 CUDA 计时。
   - [已完成] 4A：固定 formal seed，并对 CUDA 推理计时显式同步。
   - [已完成] 4B：saved evaluator 验证并消费 inference observed-mask 合同。
   - [已完成] 4C：CD validation 比较 online/EMA 并写入 deployment weight source。
   - [已完成] 4D：validation-only threshold artifact 绑定 checkpoint，formal inference 禁止独立阈值。
5. [已完成] 概率地图 observed 权威边界、prediction 通道合同和 DEM 量纲。
6. [已完成] body-centered rolling map、轨迹走廊安全查询与 ROS 接口边界。
7. [已完成] finite Radar 聚合、字段单位 schema、失败收据和 formal v3 门禁。
8. [已完成] sequence、异常处理、formal/legacy 入口、Karras/CD 命名和统一指标清理。

## 约束

- 不覆盖或改写用户现有 `NTU4DRadLM_pre_processing/preprocess.sh` 修改。
- 不删除或覆盖数据集、normalization artifact、checkpoint、训练日志和实验结果。
- 不自动运行正式训练、完整预处理、全量推理或 ROS/PX4 仿真。
- 每阶段先阅读真实调用链，再修改；测试位于 `test/`，只运行短时回归。
- 影响监督信号、体素或指标时必须记录语义变化和 checkpoint 兼容边界。
- formal v2 数据协议保持冻结；需要重建数据的变化另立 formal v3，不静默覆盖。

## 错误记录

| 错误 | 尝试 | 处理 |
|---|---:|---|
| 阶段 1 RED 测试发现默认 YAML 仍含 `num_gpus: 4` | 1 | 符合预期；下一步删除静态字段并显式写入 active BCE+Dice 参数 |
| 阶段 2 新增 6 项 observed-domain RED 测试均因接口缺少 `observed_mask` 失败 | 1 | 符合预期；实现统一解析器并逐层接入 loss/metric/trainer |
