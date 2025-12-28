import asyncio
import os
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, END

from .schemas import TravelAgentState
from .agents import (
    ensure_customer_info_node,
    parse_or_update_plan_node,
    ask_missing_core_fields_node,
    execute_tools_node,
    synthesize_results_node,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQLITE_PATH = os.getenv(
    "LANGGRAPH_SQLITE_PATH",
    str(PROJECT_ROOT / ".langgraph_checkpoints.sqlite"),
)


def build_enhanced_graph(checkpointer: AsyncSqliteSaver):
    """
    构建生产级 LangGraph：
    - 入口：ensure_customer_info（必要时 interrupt 触发表单）
    - 流转：parse/update plan → ask missing core fields → execute tools → synthesize
    - 分支（按 current_step）：
    - collecting_info → END（前端去收集客户信息 / HITL）
    - synthesizing    → synthesize_results（综合 + 套餐 + CRM + 邮件）
    - complete        → END
    """

    workflow = StateGraph(TravelAgentState)

    workflow.add_node("ensure_customer_info", ensure_customer_info_node)
    workflow.add_node("parse_or_update_plan", parse_or_update_plan_node)
    workflow.add_node("ask_missing_core_fields", ask_missing_core_fields_node)
    workflow.add_node("execute_tools", execute_tools_node)
    workflow.add_node("synthesize_results", synthesize_results_node)

    workflow.set_entry_point("ensure_customer_info")

    workflow.add_conditional_edges(
        "ensure_customer_info",
        lambda state: state["current_step"],
        {
            "collecting_info": END,
            "continue": "parse_or_update_plan",
            "complete": END,
        },
    )

    workflow.add_edge("parse_or_update_plan", "ask_missing_core_fields")

    workflow.add_conditional_edges(
        "ask_missing_core_fields",
        lambda state: state["current_step"],
        {
            "continue": "execute_tools",
            "complete": END,
        },
    )

    workflow.add_conditional_edges(
        "execute_tools",
        lambda state: state["current_step"],
        {
            "synthesizing": "synthesize_results",
            "complete": END,
        },
    )

    workflow.add_edge("synthesize_results", END)

    print("✓ Graph compiled successfully")
    return workflow.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    async def main():
        print("=" * 80)
        print("Multi-Agent Travel Booking System")
        print("Production-ready LangGraph implementation (SQLite checkpoints)")
        print("=" * 80)

        async with AsyncSqliteSaver.from_conn_string(DEFAULT_SQLITE_PATH) as saver:
            graph = build_enhanced_graph(checkpointer=saver)
            print("\n✓ Graph ready for production use")
            print("\nTo integrate:")
            print("1. Import: from backend.travel_agent import build_enhanced_graph")
            print("2. Provide an AsyncSqliteSaver (e.g., AsyncSqliteSaver.from_conn_string(db_path))")
            print("3. Invoke: await graph.ainvoke({'messages': [HumanMessage(content=query)]}, config)")

    asyncio.run(main())
