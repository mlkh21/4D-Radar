<!-- 文件功能：记录正式训练协议切换的调用链证据和设计判断。 -->
# 调用链发现

## 2026-08-21 8 GB 单卡 formal mini 审计

- 现有 `test/mini-test/train_minimal.sh` 默认使用旧 `Data/NTU4DRadLM_Pre_sensor_aware`，并在派生配置中清空正式 artifact/scale、始终传入 `--allow_legacy_radar_units`；它只能验证 legacy mini，不能代表 `formal_p1_04_full120_86p8_v1`。
- 正式 artifact 固定绑定 `[32,128,128]`、source/model range `[0,-20,-6,120,20,10]` 和 `86.8 m/s`，formal mini 不能为省显存擅自缩小网格或裁到 40 m，否则 artifact 校验应失败。
- 候选 garden 有 Radar/target/IR 各 4013 帧，标定位于 `Data/config`；mini 脚本可以继续用少量连续帧软链接，不改变磁盘数据或每帧体素数。
- 默认 mini 会写入已有 `test/mini-test/train_results_mini`（当前约 1.7 GB，含历史 VAE/LDM/CD），正式 mini 必须使用新的独立结果根并默认拒绝复用，避免旧 checkpoint/日志污染。
- 当前训练器默认设备是 CUDA，不会在 CUDA 不可用时安全回退 CPU；针对 8 GB 笔记本应先做设备/显存/温度预检，并把 VAE、LDM、CD 分阶段短时运行，不自动执行长链。
- 正式 VAE/LDM/CD mini 仍使用完整四通道 `32×128×128` 张量；降低负载只能通过样本数、epoch、batch、worker、AMP/gradient checkpoint 和阶段间冷却，不应改变监督或网格协议。
- 受保护 runner 冻结 garden、artifact hash、full120、86.8、轻量 VAE 和单卡配置；外部环境只能把样本、温度或时长上限调得更保守，不能提高 80°C/20 分钟等安全上限。
- Conda 默认捕获模式不会可靠转发 `python -` 的 heredoc stdin，曾使 artifact 校验空跑并返回 0；`--no-capture-output` 是当前直接脚本入口的必要接口条件。
- 正确的 preflight 必须明确输出 artifact SHA，通过后仍不得创建 scratch/config/output；真实 4070 预检满足 8188 MiB 总显存、7186 MiB 空闲和 37°C。

- 用户生成的正式 artifact 已通过加载，SHA-256 为 `2c9c92650b98ec686d621b53eccb5e7f376cb6b8ea1047d4fb594349af90c4d5`。
- 候选 `garden` manifest 为 4013 帧，`loop3` manifest 为 6432 帧；单样本 target/Radar 均为 `(4,32,128,128)`，真实 IR 与标定均可用。
- `default_config.yaml` 当前仍保留空 `radar_normalization_path` 和 null `doppler_scale_mps`。
- 正式训练及三个推理 launcher 当前仍指向旧 `Data/NTU4DRadLM_Pre_sensor_aware`，未消费已验收的候选根。
- `train_unified.sh` 会按固定输出路径自动续训已有 checkpoint；新 LDM/CD 必须避免隐式继承旧 normalization 协议。
- 训练配置覆盖脚本当前只替换 `data.dataset_dir`，因此即使切换数据根，空 artifact/null Doppler 仍会原样进入 `unified_train.py`。
- `unified_train.py` 已在 LDM resume 权重加载前比较 normalization spec/hash；底层 fail-closed 是必要的第二道防线，但 launcher 不应把已知旧 checkpoint 自动塞入正式新链。
- VAE 只消费 target，不绑定 Radar normalization；可继续允许用户显式选择已有 VAE，但 LDM/CD 的新协议训练必须默认从头开始。
- 三个正式推理 launcher 都从 checkpoint 内嵌 normalization 读取推理协议，因此只需统一候选数据根和 manifest；不应从本机 artifact 反向覆盖 checkpoint 协议。
- 现有 normalization 测试明确断言默认 YAML 为空配置；切换正式 artifact 后必须先把该契约改为精确路径、`86.8` 和正式 artifact 可加载，而不是保留过期断言。
- 正式 manifest/inference 测试目前只检查字符串包含 `NTU4DRadLM_Pre_sensor_aware`，无法区分旧根和新 candidate 根，需要收紧为完整目录名。
- 直接在旧 `Result/train_results/{ldm,cd}` 下“从头训练”会覆盖历史 best/log；新正式协议必须使用独立结果根，且已有非空结果目录默认拒绝进入。
- 建议 launcher 用显式 `ALLOW_RESUME=1` 才允许恢复当前协议目录；即便放行，LDM/CD 仍由 Python 层 normalization/hash 门禁复核。
- `unified_train.py` 的 VAE、LDM、CD 三个分支都已经接受统一的 `--resume`；launcher 可以删除隐式探测，并仅在 `ALLOW_RESUME=1` 且对应 best checkpoint 存在时传入该参数，无需改训练器接口。
- 为避免 checkpoint、日志和推理结果混用，正式结果根与推理输出名都应携带稳定协议标识 `formal_p1_04_full120_86p8_v1`。
- default YAML 的 VAE/LDM/CD `save_dir` 也需要切到独立协议根，否则绕过 shell 直接调用 `unified_train.py` 仍可能写入旧结果目录。
- `evaluate_inference.sh` 还同时引用旧 preprocessed 根和旧 `NTU4DRadLM_Raw`；新候选体素的 LiDAR 索引来自 `NTU4DRadLM_Raw_p1_01_candidate`，离线评价若继续读取旧 Raw 会产生跨时间协议错配。
- 正式评价的预测目录必须与带协议标识的部署输出保持同名映射，否则即使生成成功也会读不到或误读旧预测。
- 根 README 仍展示旧 checkpoint/输出目录，完成代码 GREEN 后需同步正式命令；本轮不修改历史诊断 `compare.sh`，其硬编码旧实验输入应保留为历史脚本而非静默改指新结果。
- GREEN 后审查发现 mini 训练会复制正式默认 YAML，再传 `--allow_legacy_radar_units`；默认 YAML 现已配置 artifact/scale，若 mini override 不显式清空，会触发正式/legacy 互斥门禁。这是默认配置切换造成的真实隐形依赖。
- mini inference 不读取 default YAML，继续由 checkpoint/CLI 的显式 legacy 标记控制；只需修复 `train_minimal.sh` 的派生配置。
- 正式 VAE 启动在导入阶段失败：训练入口只把 `diffusion_consistency_radar/` 加入 `sys.path`，可导入顶层 `cm`，但无法解析新增的 `diffusion_consistency_radar.*` 包路径。
- `cd_train_optimized.py` 的 fallback 又把 `checkpoint_chain` 当作顶层模块导入，而 `checkpoint_chain.py` 内部依赖正式包路径，因此 fallback 并未真正提供直接脚本兼容。
- 该错误发生在参数解析和 Trainer 创建前；正式结果根仍不存在，没有生成 checkpoint、训练日志或监督输出。
- 修复后两个入口统一使用 `diffusion_consistency_radar.*` 包名，避免 `scripts.cd_train_optimized` 与正式包名同时加载同一模块；失效的 `ModuleNotFoundError` fallback 已删除。
- 第二次正式启动已进入 VAE epoch 1，但第一个 DataLoader batch 在 default collate 阶段失败，尚未执行优化器更新。
- 真实样本顶层 metadata 没有 `None`；唯一不可默认拼接的值来自审计字段 `preprocess_policy`，其中 `velocity_mode=none` 对应 `v_drone/velocity_file/sha256/record_count` 四个合法 JSON null。
- 模型训练仅消费 observed mask 和多模态张量/布尔标志，不消费 preprocess policy；应在共享 collator 保留逐样本原始 policy，而不是篡改缺失值语义或删除 provenance。
- 统一训练、standalone CD、条件推理共有四个 Dataset DataLoader，若只修 VAE 入口会留下同类隐形依赖。
- 失败目录仅有 header-only `metrics.csv` 与启动头 `training.log`，无 checkpoint；重跑前需无损归档，不能使用 `ALLOW_RESUME=1` 假装存在恢复点。
- 共享 collator 只对 `preprocess_policy` 使用逐样本列表，其余字段继续 default-collate；真实两样本多 worker 烟测验证 null 保留与正式 tensor shape。
- 失败日志已移动到协议根 `failed_starts/vae_20260820_212426_collate_failure/`，active VAE 目录为空，launcher 的非空保护无需放宽。
- 正式训练顺序应使用独立 `vae → ldm → cd` 命令；VAE 已存在后不能再调用 `all`，否则非空门禁会拒绝第一阶段。
- VAE 正式 checkpoint 是按 garden validation IoU 选择的 `vae_best.pt`；LDM `ldm_best.pt` 按固定单步 denoising validation IoU/latent loss 选择，最终完整采样质量仍需独立门禁。
- 旧 `select_ldm_checkpoint.py` 固定要求 Z64、near40 和阈值 0.99，与当前 Z32/full120 正式协议不兼容；不能直接用于新权重，必须先参数化并冻结 full120 协议。
- 当前 VAE reconstruction 诊断和 IR ablation 都未严格复现 garden 3210/803 连续时间块成员；在把结果用于模型选择前需增加 validation suffix 选择接口。
- 三个正式推理 launcher 都先校验完整 VAE/LDM/CD 链；因此正式 LDM/CD 对比应在 CD 完成和 checkpoint chain construct gate 后开始。
- 正式推理当前默认 occupancy threshold 为 0.05，未显式传 `--seed`，`inference.py` 也会对非空目录直接覆盖；阈值、随机种子和输出不可覆盖协议必须在 loop3 test 推理前修复。
- 阈值扫描只消费已保存预测。正式流程需先在 garden validation suffix 生成 LDM/CD1/CD4 固定种子预测，各自标定部署阈值，并额外报告共享阈值下的 CD 忠实度；禁止用 loop3 调参。
- loop3 共有 6432 帧，单个 float32 四通道 Z32 voxel 约 8 MiB；仅三组 voxel 约 151 GiB，另有 uncertainty/点云/日志，因此全量部署前应预留约 220--250 GiB 并先做隔离的 32 帧运行时估算。
- pose-aware layered map 已有消费入口，但正式地图还需要逐帧 body-to-local pose CSV、observed mask 和可信动态 evidence producer；缺这些输入时只能做受限静态/identity 诊断，不能宣称 35--70 m/s 部署闭环。
