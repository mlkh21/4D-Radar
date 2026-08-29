<!-- 文件功能：记录正式训练 YAML 预设改造进展、测试和错误。 -->
# 正式训练 YAML 预设进展

## 2026-08-29

- 用户要求把各阶段 epoch、每 epoch 训练/验证帧数和默认 CUDA devices 放入 YAML，并让环境变量成为单次高优先级覆盖。
- 已读取 planning-with-files、根规则和 test/AGENTS；不会运行长训练。
- 已确认旧 launcher 固定所有阶段 20 epoch、从 GPU 数生成 batch，并删除 mini 帧限制；开始审计 formal 数据选择和 checkpoint 身份。
- 阶段 1 完成：冻结每场景静态 epoch 子集、`0=full`、selection hash/resume 身份、环境变量三级优先级及极简命令；进入 RED 测试。
- 已核对正式 artifact SHA256，确定 YAML 默认身份与环境变量覆盖关系；下一步先补 RED 测试。
- 已定位共享 resume 门禁、三个阶段 checkpoint 写入点和 formal/mini 帧选择分支，具备补充协议测试的条件。
- 决定把阶段帧选择写入独立 `stage_training_selection` checkpoint 字段；父链仍只比较基础 `data_protocol`，同阶段 resume 额外比较选择身份。
- RED 已确认：`test_formal_training_yaml_defaults.py` 因生产代码尚无 `build_formal_stage_training_selection` 而失败（ImportError），证明新合同不是旧实现误通过。
- 已实现正式阶段帧选择的稳定哈希、结构校验和可选 resume 比对门禁。
- 已将 20 epoch、每阶段 3210/774 帧、默认 GPU `0,1,2,3` 及 normalization SHA256 写入默认 YAML。
- launcher 已改为先读取 YAML，再按“阶段专用环境变量 > FORMAL 通用环境变量 > YAML”解析 epoch/帧数；CUDA 与 artifact SHA 也采用环境变量优先。
- override YAML 生成器现在写入三个阶段各自的有效 epoch 和 train/validation 帧上限。
- unified 入口已在 formal split 内按阶段确定性截取帧，实际 ID/数量/hash 写入 trainer 和 checkpoint；VAE/LDM/CD resume 均接入同阶段选择门禁。
- `bash -n` 与三个修改 Python 模块的 `py_compile` 已通过。
- 首轮 37 个 YAML/launcher/DDP 测试通过。
- 扩展 checkpoint 回归发现 Python 3.8 在导入时不支持运行期 `tuple[int, int]` 注解；需要改为已导入的 `Tuple[int, int]`，这也说明仅 `py_compile` 不能替代真实模块导入。
- Python 3.8 注解问题已修复；checkpoint 回归继续发现两个用 `__new__` 构造的最小 VAE trainer 缺少新属性，resume 调用需对该兼容测试路径使用 `getattr(..., None)`。
- 最小 trainer 兼容路径已修复；VAE checkpoint 26 项、CD 入口、LDM validation 5 项回归全部通过。
- 真实 Conda 预检完成：4013 帧 Radar statistics、artifact SHA 和 formal data protocol 均通过，输出确认 YAML 默认值为 20/20/20 epoch、每阶段 3210/774 帧；未生成配置且未启动训练。
- 直接 `bash` 在未激活环境时会调用缺少 PyTorch 的系统 Python；最终极简命令必须保留 `conda run --no-capture-output -n <env> bash ...`，若已激活训练环境才可省略 Conda 前缀。
- 新增动态短测试证明阶段 epoch/帧变量覆盖 FORMAL 通用值、FORMAL 通用值覆盖 YAML，5 项新协议测试全部通过。
- 收尾 200 项 CPU/协议回归全部通过：YAML 5、launcher 20、DDP 14、checkpoint chain 14、VAE 26、LDM validation 5、CD 1、mini config/safety 103、normalization 12。
- README 已改为 Conda 极简命令，并说明 YAML 修改位置、覆盖变量、阶段帧身份及 DDP padding 边界。
- shell 语法、Python 3.8 编译/导入、YAML 解析与 `git diff --check` 全部通过；本次未启动训练或修改任何数据/checkpoint/结果。
