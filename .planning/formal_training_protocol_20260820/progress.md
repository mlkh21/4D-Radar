<!-- 文件功能：记录正式训练协议切换的实施和验证进展。 -->
# 实施进展

## 2026-08-20

- 已收到用户正式输入验收 PASS 输出。
- 已读取项目与测试规则、planning-with-files 技能及相关工作区差异。
- 已确认工作树包含大量既有未提交修改；本轮只在相关配置、launcher、测试和记录上做局部增量。
- 正在审计训练/推理配置覆盖与 checkpoint 恢复调用链，未启动训练或推理。
- 已追踪训练 launcher 的配置生成和 VAE/LDM/CD 分支：VAE/LDM 会自动续训固定路径，CD 底层也存在固定 `cd_best.pt` 自动恢复行为，需要统一改为显式授权。
- 已定位现有 normalization、manifest、formal inference 和 launcher 静态协议测试，下一步先补 RED 契约。
- 已确认统一训练入口三个阶段都支持显式 `--resume`，因此本轮不新增 Python CLI 参数，只收紧 shell 的默认恢复策略与输出隔离。
- 已扩展调用链到独立评价入口，确认 preprocessed/Raw/index/预测目录四者也必须随协议标识同步切换。
- 已写入 4 项 RED 契约；首次具名 unittest 命令因 `test/` 非 package 在测试收集阶段失败，尚未形成行为 RED，已改用直接执行既有测试文件。
- 行为 RED 已确认：normalization 11/12、launcher 5/6，正式推理评价既有 7 项通过且新增协议断言 4 处失败；失败均精确指向旧默认数据根、空 artifact、隐式续训和旧输出命名。
- GREEN 已通过：normalization 12/12、launcher 协议 6/6、正式推理评价 11/11，5 个正式 shell 入口 `bash -n` 通过。
- 已切换默认配置、训练/三种生成/独立评价入口；未运行训练、推理或评价，未创建或覆盖任何新旧结果目录。
- 代码审查命中 mini 配置继承冲突，准备先加 RED 后最小清空 mini 的正式 normalization 字段。
- mini 隐形依赖 RED 已确认：5/6 通过，唯一失败为派生配置未清空正式 artifact；已在 mini 配置生成处显式设置空路径/null scale。
- mini 隐形依赖 GREEN 与扩展回归通过：normalization 12、launcher 6、formal inference 11、mini train 101、manifest 10、checkpoint chain 8，另两份 CD 直接接口测试通过。
- README 已同步 candidate 数据、协议结果根、显式恢复方式和生成/评价分离命令；训练脚本头部用法已纠正为 `bash`。
- 最终补充验证通过：VAE checkpoint 23、多模态 inference 31；candidate preprocessed/Raw/artifact 路径存在，正式结果根为 fresh，`git diff --check` 通过。
- 三份项目 TODO 已同步监督/样本数量、协议可比性、测试结果和下一条显式长训练命令；本计划全部完成。
- 最终文本审查修正 LDM 缺 VAE 时残留的 `sh $0 vae` 提示为 `bash $0 vae`，与脚本的 Bash 数组/mapfile 接口一致。
- 用户首次启动正式 VAE 时在 Python 导入阶段失败；已确认结果根未创建，并定位为直接脚本入口缺少仓库根路径。
- 已开始阶段 4：准备先增加不启动训练的 `--help` 子进程 RED，再小步修复两个训练入口。
- 阶段 4 已完成：RED 精确复现，GREEN 与 checkpoint/VAE/CD/launcher 聚焦回归通过，三份项目 TODO 已同步；未重启正式长训练。
- 用户第二次启动已进入 epoch 1，随后在首 batch 的 metadata default collate 阶段失败；已定位真实 JSON null 和全部 DataLoader 调用端，开始阶段 5 RED/GREEN。
- 阶段 5 已完成：nullable policy RED/GREEN、真实 batch 烟测、四调用端审查、聚焦回归和失败日志归档全部完成；未重启长训练。
- 用户请求训练完成后的完整计划；已启动阶段 6，只审计现有入口并编排 gate，不启动任何训练、推理或评价。
- 阶段 6 审计完成：确认完整链顺序、full120 validation 工具缺口、推理 seed/阈值/覆盖门禁、全量存储预算及地图外部输入前置条件；准备向用户交付分阶段计划。

## 2026-08-21

- 用户要求转为 RTX 4070 Laptop 8 GB 的短时 mini 训练，并优先降低持续热负载。
- 已读取 planning-with-files 与 `test/AGENTS.md`，建立阶段 7；确认本轮不自动启动训练。
- 已审计 legacy mini、正式 normalization、训练器设备/显存配置及 checkpoint 链：formal mini 必须保持 `32×128×128/full120/86.8`，只缩短样本与 epoch。
- 已决定在现有 `train_minimal.sh` 增加显式 formal/legacy 分支，另加 8 GB 单卡受保护 runner；formal mini checkpoint 使用独立协议标识，使正式 checkpoint-chain 入口自动拒绝。
- 已发现默认 AMP 因 VAE 类型不匹配保持关闭；不能为了省显存盲目开启，继续依赖 batch=1、ultra-lightweight、gradient checkpoint、单阶段和运行时温度/时长门禁。
- RED 已确认：mini 静态协议缺 formal 分支与 8 GB runner；配置生成器缺 artifact/scale/checkpoint protocol 参数；新增 payload 断言尚待单独复核。mini 脚本测试其余 101 项保持通过。
- 已完成 mini GREEN：训练/推理入口可显式选择 legacy/formal；formal 分支在修改 scratch 前校验 artifact 的完整 grid/scale/SHA，并写入 `formal_mini_chain_v1`。
- 新增 8 GB 单卡 runner：固定 batch=1、16 帧、每阶段 1 epoch，拒绝链式阶段、非单卡、低可用显存和高启动温度；运行期轮询温度并限制单阶段 20 分钟。
- Bash 语法、mini 协议 7 项和 mini 训练脚本 103 项全部通过；没有启动 GPU 或训练进程。
- 已冻结受保护入口的 scene/artifact/grid/模型与 allocator 配置，增加逐阶段 fresh scratch/config、不可放宽的硬件门禁和 `INT → TERM → KILL` 停止升级。
- 已增加无写入 `MINI_PREFLIGHT_ONLY=1`，并由错误 SHA 负向探针发现 Conda stdin 静默绕过；修复为 `--no-capture-output` 后负向回归与正确 SHA 真实预检均通过。
- 最终聚焦回归：mini 协议 11、mini 配置/安全 103、checkpoint 链 10、VAE payload 23，CD 入口测试通过；没有启动训练。
- 阶段 7 已完成，三份项目 TODO 与 README/mini README 已同步。
