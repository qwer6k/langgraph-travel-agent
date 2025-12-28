# tests/test_tool_reuse_keyed.py
# ------------------------------------------------------------
# Usage:
#   1) 修改下面 TRAVEL_AGENT_MODULE 为你那份代码所在的模块路径
#   2) pip install -U pytest pytest-asyncio
#   3) pytest -q
#
# 你也可以用环境变量覆盖：
#   TRAVEL_AGENT_MODULE=your.pkg.module pytest -q
# ------------------------------------------------------------

import os
import json
import importlib
import pytest

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

import types
import pytest
from langchain_core.messages import AIMessage

@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch, m):
    async def fake_ainvoke(prompt: str):
        # 直接回显 prompt，保证断言稳定（prompt 里会包含你塞进去的 JSON）
        return AIMessage(content=prompt)

    # m 是你测试里的被测模块（backend.travel_agent）
    monkeypatch.setattr(m, "llm", types.SimpleNamespace(ainvoke=fake_ainvoke), raising=False)

TRAVEL_AGENT_MODULE = os.getenv("TRAVEL_AGENT_MODULE", "CHANGE_ME_TO_YOUR_MODULE_PATH")
def _flight_item(airline: str):
    return {
        "airline": airline,
        "price": "123.45",  # str
        "departure_time": "2026-02-14T08:00:00",
        "arrival_time": "2026-02-14T12:00:00",
        "duration": "4h",   # optional but ok
        "is_error": False,
        "error_message": None,
    }

def _hotel_item(name: str):
    return {
        "name": name,
        "category": "4EST",       # str
        "price_per_night": "200", # str
        "source": "TEST",         # str
        "rating": 4.5,            # Optional[float]
        "is_error": False,
        "error_message": None,
    }

def _activity_item(name: str):
    return {
        "name": name,
        "description": "test activity",
        "price": "20",
        "location": "TEST",
        "is_error": False,
        "error_message": None,
    }


# -----------------------------
# Test Doubles / Monkeypatches
# -----------------------------

class DummyLLM:
    async def ainvoke(self, prompt: str):
        # 直接把 prompt 回显，便于断言 “最终用了哪条 tool result”
        return AIMessage(content=prompt)


class DummyAsyncTool:
    async def ainvoke(self, *args, **kwargs):
        return None


class DummyOption:
    """
    用来替代 FlightOption/HotelOption/ActivityOption 的实例：
    - 兼容 getattr(..., "is_error", False)
    - 兼容 .model_dump()
    """
    def __init__(self, data: dict):
        self._data = dict(data)

    def model_dump(self):
        return dict(self._data)

    def __getattr__(self, item):
        if item in self._data:
            return self._data[item]
        raise AttributeError(item)


class DummyOptionSchema:
    """
    用来替代 FlightOption/HotelOption/ActivityOption 的 class：
    - 提供 .model_validate() -> DummyOption
    """
    @classmethod
    def model_validate(cls, data: dict):
        return DummyOption(data)


@pytest.fixture(scope="session")
def m():
    """
    动态导入你的模块；如果路径没改对，会直接 skip 并提示你改 TRAVEL_AGENT_MODULE。
    """
    if TRAVEL_AGENT_MODULE == "CHANGE_ME_TO_YOUR_MODULE_PATH":
        pytest.skip("请先把 TRAVEL_AGENT_MODULE 改成你项目里那份代码的真实模块路径")

    try:
        mod = importlib.import_module(TRAVEL_AGENT_MODULE)
    except Exception as e:
        pytest.skip(f"无法 import 模块 {TRAVEL_AGENT_MODULE}: {e}")

    return mod


@pytest.fixture(autouse=True)
def patch_external(monkeypatch, m):
    """
    全局 patch：
    - llm：回显 prompt，方便断言
    - 发邮件/CRM/套餐生成：全部 no-op
    - 选项 schema：用 DummyOptionSchema 绕过必填字段不确定的问题
    """
    # LLM
    monkeypatch.setattr(m, "llm", DummyLLM(), raising=False)

    # outbound side effects
    monkeypatch.setattr(m, "send_email_notification", DummyAsyncTool(), raising=False)
    monkeypatch.setattr(m, "send_to_hubspot", DummyAsyncTool(), raising=False)

    async def _no_packages(*args, **kwargs):
        return []
    monkeypatch.setattr(m, "generate_travel_packages", _no_packages, raising=False)

    # schemas for parsing tool results
    monkeypatch.setattr(m, "FlightOption", DummyOptionSchema, raising=False)
    monkeypatch.setattr(m, "HotelOption", DummyOptionSchema, raising=False)
    monkeypatch.setattr(m, "ActivityOption", DummyOptionSchema, raising=False)


# -----------------------------
# Helpers
# -----------------------------

def _make_toolmsg(tool_name: str, key: str, payload: list[dict], idx: int = 0) -> ToolMessage:
    return ToolMessage(
        name=tool_name,
        content=json.dumps(payload, ensure_ascii=False),
        tool_call_id=f"call_{tool_name}:{key}:{idx}",
    )


def _make_plan(m, **overrides):
    """
    兼容 pydantic / dataclass 两种 TravelPlan 定义方式。
    """
    base = dict(
        origin="HKG",
        destination="NRT",
        departure_date="2025-12-20",
        return_date="2025-12-25",
        adults=1,
        travel_class="BUSINESS",
        departure_time_pref=None,
        arrival_time_pref=None,
        duration_days=None,
        total_budget=600.0,
        user_intent="flights_only",
    )
    base.update(overrides)

    TP = m.TravelPlan
    if hasattr(TP, "model_validate"):
        # pydantic v2
        return TP.model_validate(base)
    return TP(**base)


# ============================================================
# 1) 目的地变化：不是“最近一条”，而是“key 匹配那条”
# ============================================================

@pytest.mark.asyncio
async def test_pick_flights_by_key_not_latest__destination_change(m):
    planA = _make_plan(m, destination="NRT", user_intent="flights_only", adults=5)
    planB = _make_plan(m, destination="KIX", user_intent="flights_only", adults=5)

    keyA = m._compute_tool_key("search_flights", planA, one_way=False)
    keyB = m._compute_tool_key("search_flights", planB, one_way=False)

    msgA = _make_toolmsg("search_flights", keyA, [_flight_item("AIRLINE_A")])
    msgB = _make_toolmsg("search_flights", keyB, [{"airline": "AIRLINE_B", "is_error": False}])

    # 注意：B 更“近”（在最后），但当前 plan 是 A，应该选 A
    state = {
        "travel_plan": planA,
        "one_way": False,
        "tools_used": ["search_flights"],
        "messages": [msgA, msgB],
        "customer_info": {"email": None},
    }

    out = await m.synthesize_results_node(state)
    text = out["messages"][0].content

    assert "AIRLINE_A" in text
    assert "AIRLINE_B" not in text


# ============================================================
# 2) 日期变化：hotel key 必须跟 checkin/checkout 匹配
# ============================================================

@pytest.mark.asyncio
async def test_pick_hotels_by_key__date_change(m):
    planA = _make_plan(m, destination="PAR", departure_date="2026-02-14", return_date="2026-02-17",
                       user_intent="hotels_only")
    planB = _make_plan(m, destination="PAR", departure_date="2026-02-15", return_date="2026-02-18",
                       user_intent="hotels_only")

    # 建议按你创建 ToolMessage 的方式算 key（带 kwargs）
    keyA = m._compute_tool_key(
        "search_and_compare_hotels",
        planA,
        city_code=planA.destination,
        check_in_date=planA.departure_date,
        check_out_date=planA.return_date,
    )
    keyB = m._compute_tool_key(
        "search_and_compare_hotels",
        planB,
        city_code=planB.destination,
        check_in_date=planB.departure_date,
        check_out_date=planB.return_date,
    )

    msgA = _make_toolmsg("search_and_compare_hotels", keyA, [_hotel_item("HOTEL_A")])
    msgB = _make_toolmsg("search_and_compare_hotels", keyB, [{"name": "HOTEL_B", "is_error": False}])

    state = {
        "travel_plan": planA,
        "tools_used": ["search_and_compare_hotels"],
        "messages": [msgA, msgB],  # B 更近，但当前 planA 应该命中 A
        "customer_info": {"email": None},
    }

    out = await m.synthesize_results_node(state)
    text = out["messages"][0].content

    assert "HOTEL_A" in text
    assert "HOTEL_B" not in text


# ============================================================
# 3) 预算变化：不 rerun 工具，但仍能复用同 key 的历史结果
#    （这里会用到 diff gating）
# ============================================================

@pytest.mark.asyncio
async def test_budget_change_reuse_history_no_tool_rerun(m, monkeypatch):
    prev_plan = _make_plan(m, destination="NRT", user_intent="flights_only", total_budget=600.0)
    new_plan = _make_plan(m, destination="NRT", user_intent="flights_only", total_budget=800.0)

    async def fake_update(prev, last_text):
        return new_plan

    # 只测 gating：确保不会去跑外部工具
    monkeypatch.setattr(m, "update_travel_plan", fake_update, raising=False)

    key_prev = m._compute_tool_key("search_flights", prev_plan, one_way=False)
    msg_prev = _make_toolmsg(
    "search_flights",
    key_prev,
    [{
        "airline": "AIRLINE_BUDGET_REUSE",
        "price": "100",
        "departure_time": "2026-01-01T10:00:00",
        "arrival_time": "2026-01-01T14:00:00",
        "duration": "4h",
        "is_error": False,
    }],
)


    state = {
        "travel_plan": prev_plan,
        "original_request": "HKG to NRT",
        "customer_info": {"email": None, "budget": "600"},
        "messages": [
            msg_prev,
            HumanMessage(content="预算改成 800 USD"),
        ],
    }

    # 新版多节点：parse/update plan -> execute tools（此用例只验证 gating，不跑工具）
    u1 = await m.parse_or_update_plan_node(state)
    state2 = {**state, **u1}
    res = await m.execute_tools_node(state2)

    # 预期：只改预算 => 不跑工具，进入 synthesizing
    assert res["current_step"] == "synthesizing"
    assert res.get("messages") == []  # 没有新 ToolMessage

    # 把节点输出写回 state（模拟 graph 行为）
    state["travel_plan"] = res["travel_plan"]
    state["tools_used"] = res.get("tools_used", ["search_flights"])
    state["one_way"] = state.get("one_way", False)

    out = await m.synthesize_results_node(state)
    text = out["messages"][0].content
    assert "AIRLINE_BUDGET_REUSE" in text


# ============================================================
# 4) one_way 变化：当 one_way=True 时，应命中 one_way 版本的 key
# ============================================================

@pytest.mark.asyncio
async def test_one_way_key_selects_correct_flights(m):
    plan = _make_plan(m, destination="NRT", user_intent="flights_only")

    key_round = m._compute_tool_key("search_flights", plan, one_way=False)
    key_oneway = m._compute_tool_key("search_flights", plan, one_way=True)

    msg_round = _make_toolmsg(
    "search_flights",
    key_round,
    [{
        "airline": "AIRLINE_ROUND",
        "price": "200",
        "departure_time": "2026-01-01T10:00:00",
        "arrival_time": "2026-01-01T18:00:00",
        "duration": "8h",
        "is_error": False,
    }],
)

    msg_oneway = _make_toolmsg(
        "search_flights",
        key_oneway,
        [{
            "airline": "AIRLINE_ONEWAY",
            "price": "180",
            "departure_time": "2026-01-02T09:00:00",
            "arrival_time": "2026-01-02T13:00:00",
            "duration": "4h",
            "is_error": False,
        }],
    )

    state = {
        "travel_plan": plan,
        "one_way": True,
        "tools_used": ["search_flights"],
        "messages": [msg_round, msg_oneway],  # round 更早，oneway 更晚
        "customer_info": {"email": None},
    }

    out = await m.synthesize_results_node(state)
    text = out["messages"][0].content
    assert "AIRLINE_ONEWAY" in text
    assert "AIRLINE_ROUND" not in text


# （可选）如果你还没实现 “one_way 变化 => rerun_flights”，这个测试会暴露问题
@pytest.mark.asyncio
@pytest.mark.xfail(reason="如果 one_way 从 False -> True 但没有 rerun_flights，会因 key 不匹配导致 flights 为空。实现 rerun 后应改为 pass。")
async def test_one_way_flip_without_rerun_will_miss_history(m):
    plan = _make_plan(m, destination="NRT", user_intent="flights_only")
    key_round = m._compute_tool_key("search_flights", plan, one_way=False)
    msg_round = _make_toolmsg("search_flights", key_round, [{"airline": "AIRLINE_ONLY_ROUND", "is_error": False}])

    # 当前 state one_way=True，但历史只有 round_trip key
    state = {
        "travel_plan": plan,
        "one_way": True,
        "tools_used": ["search_flights"],
        "messages": [msg_round],
        "customer_info": {"email": None},
    }
    out = await m.synthesize_results_node(state)
    text = out["messages"][0].content
    assert "AIRLINE_ONLY_ROUND" in text  # 期望失败：不应匹配到


# ============================================================
# 5) intent 切换：allowed_tools 过滤是否生效（只输出 activities）
# ============================================================

@pytest.mark.asyncio
async def test_intent_switch_filters_allowed_tools__activities_only(m):
    plan = _make_plan(m, destination="SIN", user_intent="activities_only")

    # 三种工具各放一条（混在 history）
    kf = m._compute_tool_key("search_flights", plan, one_way=False)
    kh = m._compute_tool_key("search_and_compare_hotels", plan)
    ka = m._compute_tool_key("search_activities_by_city", plan)

    msg_f = _make_toolmsg("search_flights", kf, [{"airline": "AIRLINE_SHOULD_NOT_SHOW", "is_error": False}])
    msg_h = _make_toolmsg("search_and_compare_hotels", kh, [{"name": "HOTEL_SHOULD_NOT_SHOW", "is_error": False}])
    msg_a = _make_toolmsg("search_activities_by_city", ka, [_activity_item("ACTIVITY_SHOULD_SHOW")])

    state = {
        "travel_plan": plan,
        "one_way": False,
        "tools_used": ["search_activities_by_city"],  # intent 只允许活动
        "messages": [msg_f, msg_h, msg_a],
        "customer_info": {"email": None},
    }

    out = await m.synthesize_results_node(state)
    text = out["messages"][0].content

    assert "ACTIVITY_SHOULD_SHOW" in text
    assert "AIRLINE_SHOULD_NOT_SHOW" not in text
    assert "HOTEL_SHOULD_NOT_SHOW" not in text
