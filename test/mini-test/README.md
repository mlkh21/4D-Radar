<!-- 文件功能：说明隔离 mini 实验入口，以及 8 GB 笔记本的正式协议短训练流程。 -->
# Mini Test

本目录用于小规模验证预处理、训练和推理调用链，输出不混入正式 `Result/`。

## 两种 mini 协议

- `train_minimal.sh` 和 `inference_minimal.sh` 默认仍是 `legacy`，用于保持历史测试兼容。
- `run_formal_mini_8gb.sh` 使用已验收 candidate 数据、正式 normalization artifact 和完整 `32×128×128/full120/86.8` 协议，但只抽取少量帧并运行 1 epoch。

`formal_mini_chain_v1` checkpoint 只用于流程验收，正式 launcher 会拒绝把它当作 `formal_chain_v1` 全量 checkpoint；mini 指标也不能作为正式模型结果。

## RTX 4070 Laptop 8 GB 推荐流程

先接通电源，把机器放在坚硬、通风表面，关闭游戏、浏览器硬件加速等 GPU 程序。首次执行时不要无人值守。脚本只能降低持续高温风险，不能替代厂商散热保护或保证硬件绝对无损。

每次只运行一个阶段：

```bash
# 只读预检：验证 GPU、artifact、数据和输出门禁，不创建训练现场
MINI_PREFLIGHT_ONLY=1 bash test/mini-test/run_formal_mini_8gb.sh vae

# 预检通过后才启动 VAE mini
bash test/mini-test/run_formal_mini_8gb.sh vae
```

VAE 完成且 GPU 冷却到 65°C 以下后，再依次运行：

```bash
bash test/mini-test/run_formal_mini_8gb.sh ldm
# 再次冷却后执行
bash test/mini-test/run_formal_mini_8gb.sh cd
```

受保护入口默认且禁止放宽以下门禁：

- 单卡、batch 1、worker 0、梯度累积 1；每阶段 1 epoch。
- 默认 16 帧，最多 32 帧；单阶段最多 20 分钟。
- 启动温度不高于 65°C，运行达到 80°C 时中止。
- 总显存至少 7500 MiB、启动时可用显存至少 6000 MiB。
- 温度读取失败或进程不响应时，按 `INT → TERM → KILL` 逐级结束整个训练进程组。

可以把门禁调得更保守，例如：

```bash
MINI_MAX_GPU_TEMP_C=75 \
MINI_MAX_STAGE_MINUTES=10 \
SAMPLES_PER_SCENE=8 \
bash test/mini-test/run_formal_mini_8gb.sh vae
```

默认输出位于 `test/result/formal_mini_p1_04_8gb_v1/`。每阶段使用独立且必须全新的 scratch/config；入口拒绝覆盖非空阶段目录或失败现场。若阶段失败，应先保留或移动对应阶段目录、scratch 和 YAML 再重试，不要删除 checkpoint 或日志。

完成 LDM 后只推理 1 帧做接口验证：

```bash
MINI_RADAR_PROTOCOL=formal \
MINI_RESULTS_DIR=test/result/formal_mini_p1_04_8gb_v1 \
MINI_INFERENCE_RESULTS_DIR=test/result/formal_mini_p1_04_8gb_v1/inference \
SCENE=loop3 MAX_INFER_FILES=1 \
bash test/mini-test/inference_minimal.sh ldm
```

## 历史 legacy mini

以下命令继续写入原有 `test/mini-test/*_mini/` 目录：

```bash
bash test/mini-test/train_minimal.sh all
bash test/mini-test/inference_minimal.sh ldm
bash test/mini-test/run_minimal_experiment.sh
bash test/mini-test/diagnose_minimal.sh
```

需要正式训练或正式评价时，使用 `diffusion_consistency_radar/launch/` 下的入口。
