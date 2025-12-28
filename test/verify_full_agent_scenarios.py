"""全链路多场景验证（不依赖真实 LLM/API Key）。

目标：验证整个 agent 系统的关键链路与边界情况：
- 多轮对话（thread_id checkpoint 恢复）
- 原生 HITL：/chat interrupt -> /chat/resume Command(resume=...)
- 缺日期追问（非 interrupt，而是对话 ASK）
- 续聊不重跑工具（复用 ToolMessage）
- 仅改预算不重跑工具
- 工具失败时的 error placeholder 不导致系统崩溃
- 非法日期格式/顺序的追问
- 低信息输入拦截
- intent 切换后的工具执行范围变化

运行：
    conda run -n agents python test/verify_full_agent_scenarios.py

说明：
- 脚本会在进程内 patch `backend.travel_agent.agents` 的 LLM 与 tools。
- 通过 httpx.ASGITransport 直接调用 FastAPI app，不需要启动 uvicorn。
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import httpx


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _poll_status(client: httpx.AsyncClient, task_id: str, timeout_s: float = 20.0) -> dict:
    deadline = time.time() + timeout_s
    last: dict | None = None

    while time.time() < deadline:
        resp = await client.get(f"/chat/status/{task_id}")
        resp.raise_for_status()
        last = resp.json()
        if last.get("status") in ("completed", "failed"):
            return last
        await asyncio.sleep(0.1)

    raise TimeoutError(f"Polling timeout for task_id={task_id}. last={last}")


@dataclass
class ToolStub:
    name: str
    handler: Callable[[dict], Any]
    calls: list[dict]

    async def ainvoke(self, args: dict) -> Any:
        self.calls.append(args)
        return self.handler(args)


class FakeLLM:
    """最小 LLM stub：返回 AIMessage(content=...)。"""

    async def ainvoke(self, prompt: str) -> Any:
        # 延迟一点点模拟 async
        from langchain_core.messages import AIMessage

        text = str(prompt)
        # 用非常稳定的输出，便于断言
        if "temporarily unavailable" in text.lower():
            return AIMessage(content="(FAKE_LLM) Tooling outage message")
        return AIMessage(content="(FAKE_LLM) Synthesis OK")


def _parse_dates_from_text(text: str) -> dict[str, Any]:
    """从 follow-up 文本中提取 YYYY-MM-DD / duration days（非常简化）。"""

    out: dict[str, Any] = {}

    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if m:
        out["departure_date"] = m.group(1)

    m2 = re.search(r"\bto\b\s*(20\d{2}-\d{2}-\d{2})\b", text, flags=re.I)
    if m2:
        out["return_date"] = m2.group(1)

    md = re.search(r"\bfor\s+(\d+)\s+days\b", text, flags=re.I)
    if md:
        out["duration_days"] = int(md.group(1))

    # budget
    mb = re.search(r"\bbudget\s*(?:is|=)?\s*(\d{2,6})\b", text, flags=re.I)
    if mb:
        out["total_budget"] = float(mb.group(1))

    return out


async def main() -> None:
    # Make repo root importable
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    # Use a temp SQLite file to avoid clobbering local dev data.
    default_db = Path("/tmp") / f"langgraph_checkpoints_full_verify_{uuid.uuid4().hex}.sqlite"
    os.environ.setdefault("LANGGRAPH_SQLITE_PATH", str(default_db))

    # Import models for deterministic tool outputs
    from backend.travel_agent.schemas import ActivityOption, FlightOption, HotelOption, TravelPlan

    # ---- Tool call counters ----
    flight_calls: list[dict] = []
    hotel_calls: list[dict] = []
    activity_calls: list[dict] = []

    # ---- Tool handlers ----
    def flights_ok(_: dict) -> list[FlightOption]:
        return [
            FlightOption(
                airline="FAKE_AIR",
                price="$500",
                departure_time="2026-04-10T08:00:00",
                arrival_time="2026-04-10T18:00:00",
                duration="10H",
            )
        ]

    def hotels_ok(_: dict) -> list[HotelOption]:
        return [
            HotelOption(
                name="FAKE_HOTEL",
                category="5EST",
                price_per_night="$200",
                source="FAKE",
                rating=4.6,
            )
        ]

    def activities_ok(_: dict) -> list[ActivityOption]:
        return [
            ActivityOption(
                name="FAKE_ACTIVITY",
                description="FAKE_DESC",
                price="$50",
                location="FAKE_LOC",
            )
        ]

    def flights_fail(_: dict) -> Any:
        raise RuntimeError("Injected flight API failure")

    # ---- Analysis stubs ----
    async def fake_enhanced_travel_analysis(user_request: str) -> TravelPlan:
        text = (user_request or "").lower()
        # 默认 full_plan，但不提供日期，强制触发“缺日期追问”
        if "paris" in text and "tokyo" in text:
            return TravelPlan(origin="Paris", destination="Tokyo")
        if "activities" in text:
            return TravelPlan(destination="Tokyo", user_intent="activities_only")
        # fallback: destination must exist for schema
        return TravelPlan(destination="Tokyo")

    async def fake_update_travel_plan(prev: TravelPlan, user_text: str) -> TravelPlan:
        data = prev.model_dump()
        data.update(_parse_dates_from_text(user_text))

        # intent switch examples
        t = (user_text or "").lower()
        if "only hotels" in t:
            data["user_intent"] = "hotels_only"
        if "only activities" in t:
            data["user_intent"] = "activities_only"
        if "only flights" in t:
            data["user_intent"] = "flights_only"

        return TravelPlan(**data)

    # ---- location resolver stub ----
    async def fake_location_to_airport_code(_: Any, location: str) -> str:
        # Keep it stable; agents uses it for tool args, not for tool key.
        mapping = {"Paris": "PAR", "Tokyo": "TYO", "Shanghai": "SHA"}
        return mapping.get(location, "XXX")

    # ---- senders (no-op) ----
    async def noop_sender(_: dict) -> Any:
        return {"ok": True}

    # Create stub tool objects
    search_flights_stub = ToolStub("search_flights", flights_ok, flight_calls)
    hotels_stub = ToolStub("search_and_compare_hotels", hotels_ok, hotel_calls)
    activities_stub = ToolStub("search_activities_by_city", activities_ok, activity_calls)

    send_email_stub = ToolStub("send_email_notification", lambda _: {"ok": True}, [])
    hubspot_stub = ToolStub("send_to_hubspot", lambda _: {"ok": True}, [])

    # Patch targets in-process
    with (
        patch("backend.travel_agent.agents.llm", new=FakeLLM()),
        patch("backend.travel_agent.agents.enhanced_travel_analysis", new=fake_enhanced_travel_analysis),
        patch("backend.travel_agent.agents.update_travel_plan", new=fake_update_travel_plan),
        patch("backend.travel_agent.agents.search_flights", new=search_flights_stub),
        patch("backend.travel_agent.agents.search_and_compare_hotels", new=hotels_stub),
        patch("backend.travel_agent.agents.search_activities_by_city", new=activities_stub),
        patch("backend.travel_agent.agents.send_email_notification", new=send_email_stub),
        patch("backend.travel_agent.agents.send_to_hubspot", new=hubspot_stub),
        patch("backend.travel_agent.location_utils.location_to_airport_code", new=fake_location_to_airport_code),
    ):
        # Import app after patching so background tasks see patched modules.
        from backend.main import app

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # -----------------------------------------------------------------
            # Scenario 0: low-signal input
            # -----------------------------------------------------------------
            thread0 = f"t0_{_now_ms()}"
            resp = await client.post("/chat", json={"message": "ok", "thread_id": thread0, "is_continuation": False})
            resp.raise_for_status()
            st0 = await _poll_status(client, resp.json()["task_id"])
            assert st0["status"] == "completed", st0
            assert st0.get("form_to_display") is None, st0

            # -----------------------------------------------------------------
            # Scenario 1: full multi-turn happy path
            #  - interrupt for customer_info
            #  - resume
            #  - ask dates
            #  - provide dates -> tools -> synth
            #  - refresh + budget change -> no tool reruns
            # -----------------------------------------------------------------
            thread1 = f"t1_{_now_ms()}"

            # 1a) start -> interrupt
            start = await client.post(
                "/chat",
                json={
                    "message": "I want a trip from Paris to Tokyo.",
                    "thread_id": thread1,
                    "is_continuation": False,
                },
            )
            start.raise_for_status()
            st1 = await _poll_status(client, start.json()["task_id"])
            assert st1["status"] == "completed", st1
            assert st1.get("form_to_display") == "customer_info", st1

            # 1b) resume -> should ask for dates (complete reply, no interrupt)
            resume = await client.post(
                "/chat/resume",
                json={
                    "thread_id": thread1,
                    "resume": {
                        "name": "Alice",
                        "email": "alice@example.com",
                        "phone": "+123456789",
                        "budget": "2000",
                    },
                },
            )
            resume.raise_for_status()
            st1b = await _poll_status(client, resume.json()["task_id"])
            assert st1b["status"] == "completed", st1b
            assert st1b.get("form_to_display") is None, st1b
            assert "travel dates" in (st1b.get("result", {}).get("reply", "").lower()), st1b

            # 1c) provide dates -> tools executed
            before_calls = (len(flight_calls), len(hotel_calls), len(activity_calls))
            dates = await client.post(
                "/chat",
                json={
                    "message": "depart 2026-04-10 for 4 days",
                    "thread_id": thread1,
                    "is_continuation": True,
                },
            )
            dates.raise_for_status()
            st1c = await _poll_status(client, dates.json()["task_id"])
            assert st1c["status"] == "completed", st1c
            after_calls = (len(flight_calls), len(hotel_calls), len(activity_calls))
            assert after_calls[0] > before_calls[0] and after_calls[1] > before_calls[1] and after_calls[2] > before_calls[2]
            assert "reply" in (st1c.get("result") or {}), st1c

            # 1d) refresh -> should NOT rerun tools
            before_calls = (len(flight_calls), len(hotel_calls), len(activity_calls))
            refresh = await client.post(
                "/chat",
                json={
                    "message": "refresh recommendations",
                    "thread_id": thread1,
                    "is_continuation": True,
                },
            )
            refresh.raise_for_status()
            st1d = await _poll_status(client, refresh.json()["task_id"])
            assert st1d["status"] == "completed", st1d
            after_calls = (len(flight_calls), len(hotel_calls), len(activity_calls))
            assert after_calls == before_calls, (before_calls, after_calls)

            # 1e) budget-only change -> should NOT rerun tools
            before_calls = (len(flight_calls), len(hotel_calls), len(activity_calls))
            budget = await client.post(
                "/chat",
                json={
                    "message": "budget is 3000",
                    "thread_id": thread1,
                    "is_continuation": True,
                },
            )
            budget.raise_for_status()
            st1e = await _poll_status(client, budget.json()["task_id"])
            assert st1e["status"] == "completed", st1e
            after_calls = (len(flight_calls), len(hotel_calls), len(activity_calls))
            assert after_calls == before_calls, (before_calls, after_calls)

            # -----------------------------------------------------------------
            # Scenario 2: invalid date format -> asks for correct format
            # -----------------------------------------------------------------
            thread2 = f"t2_{_now_ms()}"
            s2 = await client.post(
                "/chat",
                json={"message": "Trip Paris to Tokyo", "thread_id": thread2, "is_continuation": False},
            )
            s2.raise_for_status()
            s2a = await _poll_status(client, s2.json()["task_id"])
            assert s2a.get("form_to_display") == "customer_info", s2a

            r2 = await client.post(
                "/chat/resume",
                json={"thread_id": thread2, "resume": {"name": "Bob", "email": "b@b.com", "phone": "+1", "budget": "1000"}},
            )
            r2.raise_for_status()
            _ = await _poll_status(client, r2.json()["task_id"])

            bad = await client.post(
                "/chat",
                json={"message": "depart 2026-99-99 for 4 days", "thread_id": thread2, "is_continuation": True},
            )
            bad.raise_for_status()
            badst = await _poll_status(client, bad.json()["task_id"])
            assert "departure date" in (badst.get("result", {}).get("reply", "").lower()), badst

            # -----------------------------------------------------------------
            # Scenario 3: tool failure placeholder does not crash
            # -----------------------------------------------------------------
            # Switch flights handler to failing for this scenario
            search_flights_stub.handler = flights_fail
            thread3 = f"t3_{_now_ms()}"
            s3 = await client.post(
                "/chat",
                json={"message": "I want a trip from Paris to Tokyo.", "thread_id": thread3, "is_continuation": False},
            )
            s3.raise_for_status()
            s3a = await _poll_status(client, s3.json()["task_id"])
            assert s3a.get("form_to_display") == "customer_info", s3a

            r3 = await client.post(
                "/chat/resume",
                json={"thread_id": thread3, "resume": {"name": "C", "email": "c@c.com", "phone": "+2", "budget": "2000"}},
            )
            r3.raise_for_status()
            _ = await _poll_status(client, r3.json()["task_id"])

            d3 = await client.post(
                "/chat",
                json={"message": "depart 2026-04-10 to 2026-04-14", "thread_id": thread3, "is_continuation": True},
            )
            d3.raise_for_status()
            d3st = await _poll_status(client, d3.json()["task_id"])
            assert d3st["status"] == "completed", d3st
            assert d3st.get("result", {}).get("reply"), d3st

            # Reset flights handler
            search_flights_stub.handler = flights_ok

            # -----------------------------------------------------------------
            # Scenario 4: intent switch (activities_only) limits tool runs
            # -----------------------------------------------------------------
            thread4 = f"t4_{_now_ms()}"
            s4 = await client.post(
                "/chat",
                json={"message": "I want a trip from Paris to Tokyo.", "thread_id": thread4, "is_continuation": False},
            )
            s4.raise_for_status()
            s4a = await _poll_status(client, s4.json()["task_id"])
            assert s4a.get("form_to_display") == "customer_info", s4a

            r4 = await client.post(
                "/chat/resume",
                json={"thread_id": thread4, "resume": {"name": "D", "email": "d@d.com", "phone": "+3", "budget": "2000"}},
            )
            r4.raise_for_status()
            _ = await _poll_status(client, r4.json()["task_id"])

            before = (len(flight_calls), len(hotel_calls), len(activity_calls))
            # Switch intent to activities_only, should only run activities tool
            i4 = await client.post(
                "/chat",
                json={"message": "only activities", "thread_id": thread4, "is_continuation": True},
            )
            i4.raise_for_status()
            i4st = await _poll_status(client, i4.json()["task_id"])
            assert i4st["status"] == "completed", i4st
            after = (len(flight_calls), len(hotel_calls), len(activity_calls))
            assert after[2] >= before[2]  # activities may run
            assert after[0] == before[0] and after[1] == before[1], (before, after)

    print("OK: full agent multi-scenario verification passed")
    print(f"SQLite: {os.environ['LANGGRAPH_SQLITE_PATH']}")
    print(f"Tool calls: flights={len(flight_calls)}, hotels={len(hotel_calls)}, activities={len(activity_calls)}")


if __name__ == "__main__":
    asyncio.run(main())
