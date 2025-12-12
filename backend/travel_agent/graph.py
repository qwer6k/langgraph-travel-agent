from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from .schemas import TravelAgentState
from .agents import call_model_node, synthesize_results_node


def build_enhanced_graph(checkpointer: Any | None = None):
    """
    构建生产级 LangGraph：
    - 入口：call_model_and_tools（分析 + 工具并发）
    - 分支：
      - collecting_info → END（前端去收集客户信息）
      - synthesizing    → synthesize_results（综合 + 套餐 + CRM + 邮件）
      - complete        → END
    """
    if checkpointer is None:
        checkpointer = InMemorySaver()

    workflow = StateGraph(TravelAgentState)

    workflow.add_node("call_model_and_tools", call_model_node)
    workflow.add_node("synthesize_results", synthesize_results_node)

    workflow.set_entry_point("call_model_and_tools")

    workflow.add_conditional_edges(
        "call_model_and_tools",
        lambda state: state["current_step"],
        {
            "collecting_info": END,
            "synthesizing": "synthesize_results",
            "complete": END,
        },
    )

    workflow.add_edge("synthesize_results", END)

    print("✓ Graph compiled successfully")
    return workflow.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    print("=" * 80)
    print("Multi-Agent Travel Booking System")
    print("Production-ready LangGraph implementation")
    print("=" * 80)

    graph = build_enhanced_graph()
    print("\n✓ Graph ready for production use")
    print("\nTo integrate:")
    print("1. Import: from backend.travel_agent import build_enhanced_graph")
    print("2. Initialize: graph = build_enhanced_graph()")
    print(
        "3. Invoke: await graph.ainvoke({'messages': [HumanMessage(content=query)]})",
    )
