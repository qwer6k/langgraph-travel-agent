"""LangGraph 原生 HITL（interrupt/resume）最小可运行示例。

兼容当前仓库环境：langgraph==1.0.4。

运行方式：
- 非交互 demo：python examples/langgraph_hitl_poc.py --demo
- 交互模式：python examples/langgraph_hitl_poc.py

说明：
- 第一次 invoke 若缺少必填字段，会返回 {'__interrupt__': [Interrupt(...)]}
- 通过同一 thread_id + Command(resume=...) 恢复执行
"""

from __future__ import annotations

import argparse
import json
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt


class PocState(TypedDict, total=False):
    destination_city: str
    start_date: str
    end_date: str
    summary: str


def _need_dates(state: PocState) -> bool:
    return not state.get("start_date") or not state.get("end_date")


def collect_missing_info(state: PocState) -> PocState:
    """缺槽位时触发 interrupt，返回 resume 后写入的字段。"""

    missing_fields: list[dict[str, Any]] = []
    if not state.get("destination_city"):
        missing_fields.append({"name": "destination_city", "label": "目的地城市"})
    if not state.get("start_date"):
        missing_fields.append({"name": "start_date", "label": "出发日期(YYYY-MM-DD)"})
    if not state.get("end_date"):
        missing_fields.append({"name": "end_date", "label": "返程日期(YYYY-MM-DD)"})

    if not missing_fields:
        return {}

    payload = {
        "type": "form",
        "title": "请补全行程信息",
        "fields": missing_fields,
    }

    # interrupt 的返回值来自后续 Command(resume=...)。
    # 这里期望 resume 是 dict，例如：{"destination_city": "上海", "start_date": "2025-12-30", "end_date": "2026-01-02"}
    resume_value = interrupt(payload)
    if not isinstance(resume_value, dict):
        raise TypeError(
            "Expected resume value to be a dict, got "
            f"{type(resume_value).__name__}: {resume_value!r}"
        )
    return resume_value


def synthesize(state: PocState) -> PocState:
    """模拟最终合成节点：把收集到的信息写成 summary。"""

    destination_city = state.get("destination_city", "(unknown)")
    start_date = state.get("start_date", "(unknown)")
    end_date = state.get("end_date", "(unknown)")

    summary = f"行程：{destination_city}，{start_date} -> {end_date}"
    return {"summary": summary}


def build_app() -> Any:
    sg: StateGraph = StateGraph(PocState)
    sg.add_node("collect_missing_info", collect_missing_info)
    sg.add_node("synthesize", synthesize)

    sg.set_entry_point("collect_missing_info")
    sg.add_edge("collect_missing_info", "synthesize")
    sg.add_edge("synthesize", END)

    # 为了演示可恢复执行，必须启用 checkpointer 并使用 thread_id
    return sg.compile(checkpointer=MemorySaver())


def _pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return repr(obj)


def run_demo(thread_id: str) -> None:
    app = build_app()
    config = {"configurable": {"thread_id": thread_id}}

    print("[1/2] first invoke -> expect interrupt")
    out1 = app.invoke({}, config=config)
    print(_pretty(out1))

    if "__interrupt__" not in out1:
        raise RuntimeError("Expected an interrupt on first invoke, but got none")

    resume_payload = {
        "destination_city": "上海",
        "start_date": "2025-12-30",
        "end_date": "2026-01-02",
    }
    print("\n[2/2] resume invoke -> expect completion")
    out2 = app.invoke(Command(resume=resume_payload), config=config)
    print(_pretty(out2))

    final_state = app.get_state(config)
    print("\nfinal snapshot:")
    print(final_state)


def run_interactive(thread_id: str) -> None:
    app = build_app()
    config = {"configurable": {"thread_id": thread_id}}

    out1 = app.invoke({}, config=config)
    print(_pretty(out1))

    interrupts = out1.get("__interrupt__")
    if not interrupts:
        print("No interrupt; graph already finished.")
        return

    intr = interrupts[0]
    # intr.value 是 interrupt(payload) 里的 payload
    payload = getattr(intr, "value", None)
    print("\n=== interrupt payload ===")
    print(_pretty(payload))

    if not isinstance(payload, dict) or payload.get("type") != "form":
        raise RuntimeError("This PoC expects a form-style interrupt payload")

    fields = payload.get("fields") or []
    if not isinstance(fields, list):
        raise RuntimeError("Invalid fields in interrupt payload")

    resume_payload: dict[str, Any] = {}
    for f in fields:
        name = f.get("name")
        label = f.get("label") or name
        if not name:
            continue
        val = input(f"{label}: ").strip()
        if val:
            resume_payload[name] = val

    out2 = app.invoke(Command(resume=resume_payload), config=config)
    print("\n=== resumed output ===")
    print(_pretty(out2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread-id", default="hitl-poc-thread")
    parser.add_argument("--demo", action="store_true", help="Run non-interactive demo")
    args = parser.parse_args()

    if args.demo:
        run_demo(args.thread_id)
    else:
        run_interactive(args.thread_id)


if __name__ == "__main__":
    main()
