<!-- 文件功能：说明隔离 mini 实验入口，以及 8 GB 笔记本的 formal v2 短训练与 smoke 流程。 -->
# Mini Test

本目录用于小规模验证预处理、训练、checkpoint 串链和推理接口，输出不混入正式
`Result/`。`train_minimal.sh` 与 `inference_minimal.sh` 默认仍为 `legacy`，用于历史回归；
当前正式数据合同只能通过 `run_formal_mini_8gb.sh` 的保护入口运行。

## Formal v2 mini 的边界

保护入口固定使用：

- training root：`Data/NTU4DRadLM_Pre_formal_v2_80m_86p8_v1/`；
- deployment root：`Data/NTU4DRadLM_Deploy_formal_v2_80m_86p8_v1/`；
- 网格：`32×128×128`，source/model range 均为 `0--80 m`；
- Doppler scale：`86.8 m/s`；
- train-only normalization、temporal purge split、真实 IR/标定、persisted observed mask
  和 Radar statistics；
- checkpoint protocol：`formal_mini_chain_v2`。

训练直接只读复用正式数据根，并从正式 split 的有序 ID 中确定性选择帧。默认 `smoke`
和 `short_train` 取每场景前 8 个 train / 4 个 validation；`medium_train` 固定取前
400 个 train / 100 个 validation。选择方式和数量写入每阶段 checkpoint 的
`data_protocol.mini_selection`；不同 profile 或选择不能续训、串链或接入正式 full 链。

每帧仍有 524288 个空间体素，模型结构和单帧显存量没有因为样本数变少而缩小。
mini 只证明数据加载、forward/backward、loss 和 checkpoint 接口能闭合，不能证明收敛，
也不能把 mini 指标作为正式模型结果。正式 checkpoint validator 会拒绝 mini 权重。

## RTX 4070 Laptop 8 GB 推荐流程

接通电源，把机器放在坚硬、通风表面，关闭其他 GPU 程序，首次运行保持有人观察。
脚本可降低持续高温风险，但不能替代厂商散热保护或保证硬件绝对无损。

先执行不训练的只读预检：

```bash
MINI_PREFLIGHT_ONLY=1 \
bash test/mini-test/run_formal_mini_8gb.sh vae
```

预检通过后，才显式启动 VAE mini：

```bash
bash test/mini-test/run_formal_mini_8gb.sh vae
```

上述默认 `smoke` 档固定 1 epoch，只用于验证工程链路。完成并验收 smoke 后，如需观察
极短的 loss 趋势，可使用独立的 `short_train` 档；它从头训练，不续接或覆盖 smoke：

```bash
# 先只读预检，不创建 short 输出
MINI_PREFLIGHT_ONLY=1 \
bash test/mini-test/run_formal_mini_8gb.sh vae short_train

# 用户确认散热和空闲显存后，再显式开始 3 epoch VAE
bash test/mini-test/run_formal_mini_8gb.sh vae short_train
```

`short_train` 目前只允许 VAE，固定 8/4 帧、batch 1、3 epoch（共 24 个训练 batch），
启动温度上限收紧为 60°C，运行中达到 75°C 即中止，最长仍为 20 分钟。默认写入
`test/result/formal_mini_v2_80m_8gb_short_v1/`；它仍只是小样本过拟合/趋势检查，不能
提供正式收敛或泛化结论。

short VAE 验收后，LDM 必须显式复用同一结果根。建议继续保持 60/75°C 温度门禁，
先做无训练预检，再由用户显式启动 1 epoch LDM：

```bash
# 已通过的零训练 LDM preflight；会校验 VAE stage/protocol/data identity
MINI_RESULTS_DIR=test/result/formal_mini_v2_80m_8gb_short_v1 \
MINI_MAX_START_TEMP_C=60 MINI_MAX_GPU_TEMP_C=75 \
MINI_PREFLIGHT_ONLY=1 \
bash test/mini-test/run_formal_mini_8gb.sh ldm smoke

# 用户确认后才执行；不要携带 MINI_PREFLIGHT_ONLY
MINI_RESULTS_DIR=test/result/formal_mini_v2_80m_8gb_short_v1 \
MINI_MAX_START_TEMP_C=60 MINI_MAX_GPU_TEMP_C=75 \
bash test/mini-test/run_formal_mini_8gb.sh ldm smoke
```

formal LDM/CD preflight 会在任何 config/output 创建前安全加载父 checkpoint，验证非空
state、stage、checkpoint protocol 和完整 data protocol；CD 还会核对 LDM 记录的 VAE
文件哈希。预检成功不代表模型质量合格。

## RTX 4070 Laptop 500 帧中型筛查

`medium_train` 用于用户确认的本地质量筛查：从 garden 正式 temporal split 固定选择
400 个 train 和 100 个 validation 帧，共 500 个唯一帧；VAE/LDM/CD 各 20 epoch。
20 epoch 会重复遍历同一训练子集，并不产生 20×500 个唯一帧。每帧仍为
`32×128×128 = 524288` 个空间体素，监督、模型结构和单帧显存没有变化。

必须逐阶段预检、训练和冷却，不能使用 `all`：

```bash
# 阶段 1：VAE；预检不会创建结果或启动训练
MINI_PREFLIGHT_ONLY=1 \
bash test/mini-test/run_formal_mini_8gb.sh vae medium_train
bash test/mini-test/run_formal_mini_8gb.sh vae medium_train

# GPU 冷却到 55°C 以下后，阶段 2：LDM
MINI_PREFLIGHT_ONLY=1 \
bash test/mini-test/run_formal_mini_8gb.sh ldm medium_train
bash test/mini-test/run_formal_mini_8gb.sh ldm medium_train

# 再次冷却并验收 LDM 后，阶段 3：CD
MINI_PREFLIGHT_ONLY=1 \
bash test/mini-test/run_formal_mini_8gb.sh cd medium_train
bash test/mini-test/run_formal_mini_8gb.sh cd medium_train
```

该档固定 batch 1、worker 0、梯度累积 1、启动温度最高 55°C、运行达到 72°C 即中止、
启动空闲显存至少 6500 MiB、单阶段最多 180 分钟；这些上限不能通过环境变量放宽。
入口还会核对 `nvidia-smi` 设备名，只有 `NVIDIA GeForce RTX 4070 Laptop GPU` 才允许
启动 `medium_train`，避免把笔记本子集 profile 误用于服务器或其他显卡。
默认结果根为 `test/result/formal_medium_v2_80m_laptop_500f_20ep_v2/`，原 smoke/short
目录不会被覆盖。根据已完成 short VAE 的吞吐，VAE 线性估算约需 80--100 分钟；
LDM/CD 尚无可靠实测，180 分钟是保护上限而不是完成承诺。训练期间应保持接通电源、
坚硬通风表面并有人观察，阶段之间充分冷却。

旧 `formal_medium_v2_80m_laptop_500f_20ep_v1/` 在 epoch 1 第 50 个 batch 因
`expandable_segments` allocator 内部断言失败，且没有生成 checkpoint。入口现统一使用
`max_split_size_mb:128`，在启动日志中打印并写入生成配置；旧目录仅保留诊断，不续训。

VAE/LDM 会在每个 epoch 消费 100 帧 validation。当前 CD 训练接口只接收 train loader，
所以 CD 的 20 epoch 实际训练 400 帧子集；保留的 100 帧必须在 CD 完成后通过独立推理/
评价入口验收。该结果能比 8/4 smoke 更可靠地检查 loss、验证指标和明显接口问题，但
不能证明满足最终部署需求，也不能替代服务器完整 formal split 训练。

服务器正式训练由 `diffusion_consistency_radar/launch/train_unified.sh` 使用 garden 的
3210 train / 774 validation 完整 split，固定 VAE/LDM/CD 各 20 epoch；launcher 会删除
mini 帧限制并写入 `formal_chain_v2`，不得复用上述 laptop checkpoint。

默认 `smoke` 保护条件如下：

- 单卡、batch 1、worker 0、梯度累积 1，每阶段固定 1 epoch；
- train/validation 默认为 8/4 帧，上限为 32/16 帧；
- 单阶段最多 20 分钟；启动温度不高于 65°C，达到 80°C 时中止；
- 总显存至少 7500 MiB，启动时空闲显存至少 6000 MiB；
- 温度读取失败、过热或超时后按 `INT → TERM → KILL` 结束训练进程组。

可以收紧但不能放宽保护，例如：

```bash
MINI_MAX_GPU_TEMP_C=75 \
MINI_MAX_STAGE_MINUTES=10 \
MINI_TRAIN_FRAMES_PER_SCENE=4 \
MINI_VALIDATION_FRAMES_PER_SCENE=2 \
bash test/mini-test/run_formal_mini_8gb.sh vae
```

默认输出是 `test/result/formal_mini_v2_80m_8gb_v1/`。非空阶段目录会被拒绝，
入口不会自动覆盖或续训。失败后应保留现场并改用新的 `MINI_RESULTS_DIR`，不要删除
checkpoint 或日志。

VAE 完成并冷却到 65°C 以下后，才可依次运行后续阶段：

```bash
bash test/mini-test/run_formal_mini_8gb.sh ldm
# 再次冷却后执行
bash test/mini-test/run_formal_mini_8gb.sh cd
```

LDM 完成后，可在严格 deployment view 上只推理 1 帧。该命令会显式授权读取
`formal_mini_chain_v2`，输出仍标记为 `formal_mini_smoke`，不会冒充正式部署结果；
它不读取 target 或 LiDAR 真值。

```bash
MINI_RADAR_PROTOCOL=formal \
SCENE=loop3 MAX_INFER_FILES=1 \
bash test/mini-test/inference_minimal.sh ldm
```

CD smoke 将最后一个参数改成 `cd`。实际运行前仍应确认对应 checkpoint 已存在，
且输出目录是 fresh。

## 历史 legacy mini

以下命令继续使用旧数据/单位并写入原有 `test/mini-test/*_mini/` 目录：

```bash
bash test/mini-test/train_minimal.sh all
bash test/mini-test/inference_minimal.sh ldm
bash test/mini-test/run_minimal_experiment.sh
bash test/mini-test/diagnose_minimal.sh
```

需要全量训练或正式评价时，使用 `diffusion_consistency_radar/launch/` 下的正式入口，
不能用 mini checkpoint 替代。
