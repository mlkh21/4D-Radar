<!-- 文件功能：记录 Codex VS Code 扩展历史会话故障排查的操作、结果与验证进度。 -->
# 排查进展

## 2026-08-20

- 已完整读取 `openai-docs` 与 `planning-with-files` 技能说明。
- 已搜索并尝试打开官方 Codex IDE 文档入口。
- 已建立独立排查计划，保留旧的 `.planning/project_full_review_20260722` 计划不变。
- 已确认扩展版本、安装位置、本地会话目录、会话索引和状态数据库均存在。
- 已发现配置候选项 `disable_response_storage = true`，尚未修改。
- 已定位 VS Code 的 Codex 专用日志文件。
- 已检查最近多次扩展启动日志：ChatGPT 登录成功、当前会话恢复成功，但第三方 `1for.cc` 模型端点持续 401，官方用户设置接口持续 403。
- 已验证 `session_index.jsonl` 完整合法，统计到 169 条索引、216 个物理会话文件和 3 个归档文件。
- 已验证 `state_5.sqlite` 完整性正常，并发现新版 UI 可见索引依赖非空 `preview` 字段。
- 已核对 SQLite：219 条线程全部对应现存会话文件，回填完成，214 条有可见 preview；排除大规模索引损坏。
- 已发现旧会话提供方 `openai` 与当前自定义提供方 `OpenAI` 的大小写/身份分裂。
- 已确认 VS Code 扩展实际捆绑并使用 Codex CLI `0.148.0-alpha.21`，后续协议验证固定使用该版本。
- 已生成扩展实际 app-server 协议并核对 `thread/list` 过滤语义；确认 `disable_response_storage` 对当前版本是未知字段。
- 已对活动 SQLite 做一致性快照到 `/tmp`，未复制认证文件；第一次协议复现确认初始化正常，但列表请求需按握手时序交互发送。
- 已按协议完成交互式列表复现：`OpenAI` 过滤只见 2 条新会话，`openai` 可见旧历史，不限制提供方则新旧均可列出。
- 已成功读取一条 2026-04-29 旧会话的完整 turns；已确认配置层来源和 `history.persistence = "save-all"`。
- 已在 `/tmp` 应用候选配置；通过严格配置解析、旧历史列表和旧会话全文读取三项回归。
- 已备份并替换真实 `~/.codex/config.toml`；历史数据库和会话文件保持只读未改。
- 最终只读验证全部通过；项目 `TODO/` 三份记录已更新，等待用户重载 VS Code 窗口使扩展进程应用新配置。
