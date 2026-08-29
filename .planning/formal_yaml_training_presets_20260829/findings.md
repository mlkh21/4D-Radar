<!-- 文件功能：记录正式训练 YAML 预设改造中的调用链发现与设计结论。 -->
# 正式训练 YAML 预设发现

## 初始事实

- `default_config.yaml` 当前声明 VAE/LDM/CD 为 `100/200/200` epoch、`data.batch_size=2`、`hardware.num_gpus=2`。
- `train_unified.sh` 当前忽略上述 epoch，强制 `FORMAL_EPOCHS=20`；同时按 GPU 数覆盖 batch/梯度累积，并删除 mini 帧限制字段。
- formal v2 的训练/验证帧来自唯一 temporal split；现有 `mini_*_frames_per_scene` 只在 formal-mini checkpoint 协议下消费，不能直接冒充正式 full/limited 训练合同。
- DDP batch 必须继续由 `resolve_world_batch_plan()` 统一生成，避免 YAML batch 与 GPU 数不匹配。

## 待确认

- 每阶段帧数应定义为每场景静态、确定性 epoch 子集，还是每轮轮换采样；需要沿 checkpoint/data protocol 选择最小安全实现。
- CD 当前没有逐 epoch validation loop，CD 的 validation 帧数只能作为留出身份或独立评价输入，不能伪装成训练期 CD 指标。

## 阶段 1 结论

- 每阶段采用 `train_frames_per_epoch` / `validation_frames_per_epoch`，在当前每场景 formal split 内确定性截取；DataLoader 每轮遍历一次固定 Dataset，因此它们定义“每场景、每 epoch”帧数。默认 `3210/774` 等于 garden full split，不改变现有监督覆盖。
- `0` 表示使用该 partition 全部帧，正整数不得超过 split 容量；多场景时同一上限逐场景应用并在日志中打印全局总数。
- 若帧数小于 full split，checkpoint 必须记录 stage selection protocol、实际 frame IDs hash 和数量；同阶段 resume 必须精确一致。VAE→LDM→CD 父链允许各阶段不同 epoch/帧数，但基础 `formal_data_v2` 必须一致。
- 临时覆盖优先级冻结为：阶段专用环境变量 > `FORMAL_*` 全阶段环境变量 > YAML。`CUDA_DEVICES` > `hardware.cuda_devices`。
- 极简命令为 `bash .../train_unified.sh <stage|all>`。
- 当前正式 normalization artifact 的 SHA256 已核对为 `11f59d84cc186c39256c112154faf458ec9ead5fec9b08b997abd5058b68e97c`；将它作为 YAML 中独立声明的默认身份，`EXPECTED_ARTIFACT_SHA256` 可临时覆盖，避免通过“对当前文件现算 hash”形成自我证明。
- 新增的每阶段帧数只应用于 `formal` 正式协议；`formal_mini` 继续使用其独立 mini selection，避免 YAML 全量默认值覆盖 mini 测试边界。
- 多 GPU 每卡 batch 仍由现有 DDP 安全计划推导，不把它与“每 epoch 帧数”混为一谈，也不绕开显存保护。
- `assert_checkpoint_training_identity()` 已是 VAE/LDM/CD 恢复前的共享门禁；阶段帧选择身份应作为 checkpoint 顶层独立字段接入该门禁，不能塞入跨阶段必须相等的 `data_protocol`。
- launcher 的 `all` 通过三次递归调用同一脚本执行；解析后的 YAML/环境变量值需要显式 export，保证递归阶段与首层采用同一组有效配置。
- 现有 mini runner 从默认 YAML 派生配置，因此正式帧上限的消费必须由 `checkpoint_protocol=formal_chain_v2` 门控，不能仅依赖字段是否存在。
- 代码审查发现临时 `CUDA_DEVICES` 虽能控制进程数，但生成的 override 若继续保留 YAML 的 `hardware.cuda_devices/num_gpus` 会产生审计身份不一致；override 必须同时写入实际设备列表和 GPU 数。
