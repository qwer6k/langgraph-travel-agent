# Examples

## LangGraph 原生 HITL（interrupt/resume）PoC

这个示例演示 langgraph 的原生“暂停等待人类输入 → 提交输入 → 恢复执行”的最小闭环。

- 第一次 `invoke` 触发 `interrupt(payload)`：返回值会包含 `__interrupt__`，其中携带 `payload`（可视为表单 schema）。
- 使用同一个 `thread_id` 调用 `invoke(Command(resume=...))`：图从暂停点继续执行。

### 运行

非交互模式（推荐用于验证/CI）：

```bash
python examples/langgraph_hitl_poc.py --demo
```

交互模式：

```bash
python examples/langgraph_hitl_poc.py
```

### 和本仓库当前 HITL 的关系

- 当前实现：用自定义字段（例如 `form_to_display`）+ 前端轮询/后端合并 state 来模拟“等待”。
- 原生实现：用 `interrupt()` 在图内产生等待点，用 `Command(resume=...)` 恢复；配合 checkpointer（如 `MemorySaver`/Redis）可跨请求恢复。
