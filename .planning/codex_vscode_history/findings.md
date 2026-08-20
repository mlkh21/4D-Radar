<!-- 文件功能：记录 Codex VS Code 扩展历史会话故障的证据、判断与修复依据。 -->
# 排查发现

## 官方资料

- 已按 `openai-docs` 要求，仅在 `developers.openai.com` 检索 Codex IDE 历史会话相关内容。
- 官方搜索结果确认 Codex/IDE 文档体系存在，但未直接建立“VS Code 扩展必须显示全部历史对话”的具体产品承诺；因此不能只凭 UI 现象下结论，需检查本地会话存储和日志。
- 后续针对 `model_provider` 的官方精确检索没有返回相关配置页，且猜测 URL 未成功取回；最终结论不能伪称为官方已明确记录的行为。

## 安全边界

- 不删除、移动或重建任何历史会话文件，除非先完成只读核验与备份。
- 不触碰项目数据集、checkpoint、训练日志或实验结果。

## 本机证据

- VS Code 当前安装 `openai.chatgpt@26.818.21641`，位置为 `/home/zxj/.vscode/extensions/openai.chatgpt-26.818.21641-linux-x64`。
- `~/.codex/sessions/`、`~/.codex/archived_sessions/` 与 `~/.codex/session_index.jsonl` 均存在；历史数据没有整体丢失。
- `~/.codex/state_5.sqlite` 及其 WAL/SHM 文件正在更新，说明本地状态库处于活动状态。
- `~/.codex/config.toml` 中存在 `disable_response_storage = true`。这是一个重点候选因素，但需结合日志与扩展实现确认其影响范围，暂不直接修改。
- 多个 VS Code 窗口日志目录中都存在 `exthost/openai.chatgpt/Codex.log`，可用于直接定位会话枚举失败。

## 日志初判

- 扩展能识别 ChatGPT 登录：日志为 `authenticatedAccountPresent=true`、`authMethod=chatgpt`、`result=succeeded`。
- 当前会话可正常恢复：日志出现 `resumeState=resumed`，说明会话读取能力本身未整体失效。
- 启动阶段反复出现模型列表请求被发往 `https://1for.cc/models` 并返回 `401 INVALID_API_KEY`。这表明 Codex 进程仍受第三方 API 端点/环境配置影响，与当前 `model_provider = "OpenAI"` 和 ChatGPT 登录路径存在冲突。
- 官方 `/settings/user` 请求反复返回 403；需判断这是权限功能探测还是导致云端历史列表失败的直接原因。
- 日志里的 `fs/readDirectory: No such file or directory` 与 Git diff 输出过大均发生在当前会话恢复之后，更像工作区附属错误，不足以解释历史列表缺失。

## 会话索引与状态库

- `state_5.sqlite` 的只读 `PRAGMA quick_check` 返回 `ok`，数据库没有明显结构损坏。
- `session_index.jsonl` 共 169 行、169 个唯一 ID，全部是合法 JSON，时间覆盖 2026-04-24 至 2026-08-20。
- 物理会话文件共 216 个，另有 3 个归档文件；索引记录少于物理文件，需确认新版 SQLite 回填/迁移是否漏掉部分旧会话。
- 新版 `threads` 表的可见会话索引只包含 `preview <> ''` 的记录。若旧会话迁移后 `preview` 为空，文件仍在但 VS Code 历史列表会不可见，这是目前比 `disable_response_storage` 更直接的根因候选。
- 当前进程环境和常见配置文件中没有发现 `1for.cc`，只有本次排查会话因日志引用而包含该字符串；第三方端点可能来自 VS Code 宿主环境、认证状态或扩展内部运行配置，不能通过普通 shell 环境直接复现。

## 根因范围进一步收敛

- `threads` 表共有 219 条记录（216 活跃、3 归档），所有 219 个 `rollout_path` 都真实存在；214 条拥有非空 `preview`。因此不是数据库损坏、文件丢失或大规模 preview 回填失败。
- 回填状态为 `complete`，物理会话文件与 SQLite 线程总数（216+3）完全对应。
- 当前仓库有 214 条活跃记录，其中 210 条符合 UI 的 `preview <> ''` 可见条件。
- 发现关键命名不一致：旧会话几乎都记录为内置提供方 `openai`（小写），而当前配置显式选择并自定义了 `OpenAI`（大写）；当前新建会话也记录为 `OpenAI`。如果扩展按当前提供方筛选，会只看到新会话而读不到旧历史。
- 当前配置的 `[model_providers.OpenAI]` 实际创建了一个大小写不同的自定义提供方，并非直接使用内置 `openai`；它与日志中第三方端点 401 同时出现，已成为首要根因候选。
- VS Code 扩展捆绑的 Codex CLI 是 `0.148.0-alpha.21`，而终端默认 CLI 是 `0.142.4`；诊断和最终验证应使用扩展捆绑版本，避免把旧 CLI 行为误当成扩展行为。

## 扩展协议与配置兼容性

- 扩展捆绑 CLI 生成的协议明确说明：`thread/list.modelProviders` 非空时按提供方精确过滤，空数组表示全部提供方。
- 当前 Webview 的主要“最近会话”调用传入 `modelProviders: null`，所以提供方大小写分裂虽是配置异常，却不能单独解释主历史列表完全为空；某些扩展宿主路径会按单一提供方过滤，仍可能影响特定入口。
- 使用扩展捆绑 CLI 的 `--strict-config` 启动验证发现：`disable_response_storage` 已是 `0.148.0-alpha.21` 不认识的配置字段。普通启动可能忽略它，但这是明确的陈旧配置，应在备份后移除。
- 对状态库快照实际调用同版本 `thread/list`：不限制提供方时能列出新旧历史；限制为 `OpenAI` 时只返回 2 条 2026-08-20 新会话；限制为 `openai` 时返回旧的交互会话。结果与 SQLite 分组一致，证明大小写分裂会在提供方过滤入口稳定复现历史缺失。
- 这也证明底层 app-server 能读取旧历史元数据；问题位于配置/列表过滤层，而非历史数据丢失。
- 对 2026-04-29 的旧 VS Code 会话执行 `thread/read(includeTurns=true)` 成功返回完整用户消息和助手回复，进一步排除“历史内容不可解析”。
- `config/read` 显示有效历史策略仍是 `history.persistence = "save-all"`；陈旧的 `disable_response_storage = true` 没有关闭本地历史，但会使严格配置校验失败。
- `config/read` 同时确认当前提供方完全来自用户级 `config.toml`：`model_provider = "OpenAI"` 以及自定义 `model_providers.OpenAI`，不是项目配置或系统配置注入。
- 候选配置第一次严格回归继续识别出 `network_access` 也是当前扩展 CLI 的未知旧字段；它在普通模式下未进入有效配置，删除不会改变当前沙箱权限。
- 后续严格校验发现 `windows_wsl_setup_acknowledged` 同样已失效；本机为 Linux，该字段也未进入有效配置，可一并清理。

## 候选修复验证

- `/tmp` 候选配置统一为 `model_provider = "openai"`，删除自定义 `model_providers.OpenAI`，并清理三个未知旧字段：`disable_response_storage`、`network_access`、`windows_wsl_setup_acknowledged`。
- 扩展捆绑 CLI `app-server --strict-config` 已能持续启动，说明候选配置通过当前版本的严格解析。
- 候选配置下 `config/read` 返回 `model_provider = "openai"`、`model_providers = {}`、`history.persistence = "save-all"`。
- 候选配置下按当前提供方 `openai` 调用 `thread/list` 成功分页返回旧 VS Code 会话，并能用 `thread/read(includeTurns=true)` 读取旧会话完整 turns。

## 已实施修复

- 原配置已备份到 `/home/zxj/.codex/config.toml.bak-20260820-codex-history-fix`。
- 已将通过回归的候选配置安装到 `/home/zxj/.codex/config.toml`。
- 未修改 `state_5.sqlite`、`session_index.jsonl`、`sessions/` 或 `archived_sessions/`。

## 最终验证

- 真实配置与已回归候选逐字一致，权限保持 `0600`；备份权限同为 `0600`。
- 真实配置只保留 `model_provider = "openai"`，不存在被清理的三个旧字段或自定义 `model_providers.OpenAI`。
- 修复后状态库再次 `quick_check=ok`：219 条线程、216 活跃、3 归档、214 条非空 preview；提供方分布为 `openai=217`、`OpenAI=2`。
- `session_index.jsonl` 仍为 169 行合法 JSON；`git diff --check` 通过。
- 正在运行的 VS Code 扩展进程需要在交付后重载窗口，才能完全重新读取用户配置；当前对话期间不自动重载，以免中断会话。
