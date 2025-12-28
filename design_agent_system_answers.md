# Agent 系统设计问答与实现映射

此文档基于当前项目代码（backend/*）回答用户提出的设计问题，并指明相关代码落点，便于后续实现与审计。

文件位置示例：
- 核心编排与合成： [backend/travel_agent/agents.py](backend/travel_agent/agents.py)
- 工具与外部集成： [backend/travel_agent/tools.py](backend/travel_agent/tools.py)
- 状态/Schema： [backend/travel_agent/schemas.py](backend/travel_agent/schemas.py)
- 图结构： [backend/travel_agent/graph.py](backend/travel_agent/graph.py)
- 服务入口 / 任务： [backend/main.py](backend/main.py)
- 前端示例（Gradio）： [backend/gradio_app.py](backend/gradio_app.py)

---

## 1) Agent 的成功定义
- 可执行行程：产生 1~3 个 `TravelPackage`（含选中航班/酒店/活动）。实现：`generate_travel_packages()` & `TravelPackage` schema（见 [backend/travel_agent/tools.py](backend/travel_agent/tools.py) 与 [backend/travel_agent/schemas.py](backend/travel_agent/schemas.py)）。
- 可预订链接 / 来源可追溯：工具层保留 `source` 字段并尽量保留供应商 id（见 `HotelOption.source` 等）。
- 价格准确与可审计：价格来自供应商响应或以 `is_error` 标记占位；合成层解析并按公式计算总价（见 `agents.py` 的价格解析段）。
- 满足偏好：`TravelPlan` 包含 `travel_class`、时间偏好、budget 等（见 schemas）。
- 可解释性：合成层生成可读说明并在套餐中给出 `budget_comment` 与成本拆解。

---

## 2) 多轮对话关键槽位
- 必填：`destination`、`departure_date` + `return_date`（或 departure_date + duration）、`adults`、`user_intent`（决定 allowed tools）。
- HITL/可后注入：`customer_info`（name/email/phone/budget）通过 `/chat/customer-info` 存储并注入（see [backend/main.py](backend/main.py)）。
- 可选：`origin`（默认 Shanghai）、`travel_class`、`departure_time_pref`、`arrival_time_pref`、`total_budget`。

---

## 3) LangGraph 节点设计与 I/O schema（子节点/字段）
- 总体：entry → `call_model_and_tools` → conditional → (`collecting_info` END | `synthesizing` → `synthesize_results` | `complete` END)（见 [graph.py](backend/travel_agent/graph.py)）。

- `call_model_and_tools`（逻辑上拆为 Node1~Node4）：
  - Node1 ensure_customer_info
    - 输入：`messages`, `customer_info?`、`is_continuation?`
    - 输出：若缺 customer_info → `current_step="collecting_info"`, `form_to_display="customer_info"`, `execution_plan`={...}
  - Node2 parse_or_update_plan
    - 输入：`messages`、`travel_plan?`
    - 输出：`travel_plan`（`TravelPlan`）、`user_followup_hint`、`execution_plan`
  - Node3 ask_missing_core_fields
    - 输入：`travel_plan`
    - 输出：若缺关键字段 → `messages`(AI ask), `current_step="complete"`
  - Node4 build_execution_plan & run tools
    - 输入：`travel_plan`, `last_tool_args?`, `state.messages`（历史 ToolMessage）
    - 输出：`messages` (ToolMessage list with `tool_call_id`)，`current_step="synthesizing"`, `last_tool_args`

- `synthesize_results_node`:
  - 输入：`messages` (ToolMessage+AI), `travel_plan`, `customer_info`, `last_tool_args`
  - 输出：`messages` (最终 AIMessage)、`current_step="complete"`，并可能触发邮件/CRM 工具。

- 关键 state 字段：`messages`, `travel_plan`, `form_to_display`, `current_step`, `customer_info`, `last_tool_args`, `execution_plan`, `*_error_message`（详见 [schemas.py](backend/travel_agent/schemas.py)）。

---

## 4) thread_id checkpoint 的持久化与版本迁移
- 当前实现：`InMemorySaver()`（开发用），见 [graph.py](backend/travel_agent/graph.py)。
- 生产候选：Redis（快速/TTL/可持久化）、Postgres（关系型可迁移）、S3+索引（审计快照）。
- 版本升级处理：在 state 写入 `schema_version`；入口做好 `migrate_state_to_current(state)` 适配；提供批量迁移脚本或异步迁移任务；灰度发布。建议在 `build_enhanced_graph()` 入口层增加 migration hook。

---

## 5) 工具层归一化 schema 与缺失字段策略
- 内部 schema（见 [schemas.py](backend/travel_agent/schemas.py)）：
  - `FlightOption`: airline, price, departure_time(ISO), arrival_time(ISO), duration, is_error, error_message
  - `HotelOption`: name, category, price_per_night, source, rating, is_error, error_message
  - `ActivityOption`: name, description, price, location, is_error, error_message
- 缺失字段处理：关键字段缺失或 SDK 错误 → 返回 `is_error=True` 的占位；部分字段为空用 `None`/`N/A` 并在合成中弱化显示。
- 单位/币种/时区归一化：价格解析与转换（`parse_price()` + `convert_to_usd`），时间尽量使用 ISO；城市/码靠 `location_utils` 规范化。

---

## 6) 筛选/排序策略（航班/酒店）
- 项目中主要是规则式：价格排序/代表样本抽取（`_get_representative_options()`）和基于价格的套餐规则。
- LLM rerank：在 `generate_travel_packages()` 中把代表样本交给 LLM 做 JSON 格式的“顾问排序”。
- 可解释性保障：保留规则计算（cost breakdown）并在套餐输出中给出预算注释与来源。

---

## 7) 异步 API 实现细节
- 目前：FastAPI + `BackgroundTasks` + `agent_graph.ainvoke()`（异步），工具内有 `asyncio.gather` 与 `run_in_executor` 用法（见 [tools.py](backend/travel_agent/tools.py)）。前端以轮询 `/chat/status/{task_id}` 获取结果（[gradio_app.py](backend/gradio_app.py)）。
- 推荐改进：使用专门的任务队列（Celery/RQ/Arq）或 worker pool，结合 per-tool timeout、circuit breaker 和全局并发限制（semaphores）、监控与追踪。

---

## 8) HITL 触发条件与回填合并流程
- 触发条件：`not state.get("customer_info")` → HITL 表单（Node1）；也会在缺关键槽（Node3）触发文字追问。
- 回填合并：前端 POST `/chat/customer-info` 存入 `customer_data[thread_id]`，随后以 `is_continuation=True` 再次 `/chat`，`run_agent_in_background()` 将 `customer_info` 注入 `initial_state`（见 [main.py](backend/main.py)）。

---

## 9) 区分“无结果”与“故障”方法
- 约定：返回 `[]` 表示业务无库存；异常或 SDK 报错转成 `is_error=True` 的占位 option（含 `error_message`）。
- 判别依据：HTTP 状态码、SDK 异常类型、响应 data 结构，以及统计层面的异常突增检测（用于判定供应商 outage）。

---

## 10) 重试/退避/幂等性
- 当前：`retry_async(retries=3, delay=1.0, backoff=2.0)`（见 [tools.py](backend/travel_agent/tools.py)）。
- 幂等性：读操作天然幂等；写操作（CRM/邮件）需引入 idempotency key（建议生产化实现）。

---

## 11) 允许部分失败继续生成：举例与披露
- 可失败但可交付：航班失败可仍展示酒店/活动；某酒店源失败但另一个源成功；活动失败但仍可提供航班+酒店套餐。
- 披露方式：合成 prompt 明确指出哪个 API 出错、不要编造数据、并给出下一步建议（见 `synthesize_results_node()` 的多个分支）。

---

## 12) 邮件模板与避免写死价格
- 模板要点：标题、套餐清单（来源+价格快照+时间戳）、免责声明（价格为检索时快照，可能变动）、CTA（预订链接）。
- 不把价格写死：在邮件标注检索时间并提供供应商链接/booking reference；可在邮件正文声明“价格参考，最终以供应商为准”。

---

## 13) 安全与风控要点
- 防 Prompt Injection：工具参数仅用受控 schema，不直接执行 LLM 文本生成的命令；对 LLM 输出做 Pydantic 校验。
- 防越权：工具白名单 + execution_plan 驱动工具执行。
- PII 处理：最小化存储、存储加密、日志脱敏。

---

## 14) 线上观测（建议监控指标）
- 对话成功率、工具失败率（按 provider）、平均轮次、平均耗时、成本（LLM token，API 调用次数）。
- 实现：Prometheus + OpenTelemetry traces + structured logs。埋点位置建议：`call_model_node()` 前后、每工具执行前后、`run_agent_in_background` 开始/结束、邮件/CRM 成功失败点。

---

## 后续建议/改进项（摘录）
- 持久化 checkpointer（Redis/Postgres）并实现 state migration。
- 在工具执行点增加 metrics/trace 埋点。
- 邮件模板增强并加入 price_snapshot metadata。
- 工具调用写操作（CRM/邮件）增加幂等 key。
- 日志脱敏与审计导出。

---

如需我把上述改进拆成具体补丁并逐步实现（Redis checkpointer、metrics、邮件模板），我可以继续创建 TODO 并生成 patch。