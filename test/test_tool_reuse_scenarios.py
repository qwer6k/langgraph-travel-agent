import asyncio
import json
import sys
import os
from copy import deepcopy

# Ensure project root is importable
sys.path.insert(0, os.getcwd())

from backend.travel_agent import agents, schemas, location_utils
from langchain_core.messages import ToolMessage, AIMessage


class FakeTool:
    def __init__(self, func):
        self._func = func

    async def ainvoke(self, args):
        # allow either dict or kwargs
        if isinstance(args, dict):
            return await self._func(**args)
        return await self._func(args)


class FakeLLM:
    async def ainvoke(self, prompt):
        return AIMessage(content="[FAKE_LLM_RESPONSE] " + (prompt[:200] if prompt else ""))


async def run():
    calls = []

    async def fake_flights(**kwargs):
        calls.append(("search_flights", deepcopy(kwargs)))
        return [
            schemas.FlightOption(
                airline="TESTAIR",
                price="100 USD",
                departure_time=(kwargs.get("departureDate") or "2026-05-01") + "T08:00:00",
                arrival_time=(kwargs.get("departureDate") or "2026-05-01") + "T12:00:00",
            )
        ]

    async def fake_hotels(**kwargs):
        calls.append(("search_and_compare_hotels", deepcopy(kwargs)))
        return [
            schemas.HotelOption(
                name="Test Hotel",
                category="4EST",
                price_per_night="120 USD",
                source="TEST",
            )
        ]

    async def fake_activities(**kwargs):
        calls.append(("search_activities_by_city", deepcopy(kwargs)))
        return [
            schemas.ActivityOption(
                name="Test Activity",
                description="Fun",
                price="20 USD",
            )
        ]

    # Patch agents' tool objects and LLM
    agents.search_flights = FakeTool(fake_flights)
    agents.search_and_compare_hotels = FakeTool(fake_hotels)
    agents.search_activities_by_city = FakeTool(fake_activities)
    agents.llm = FakeLLM()

    # Patch location_utils to avoid external API calls
    async def fake_loc(am, loc):
        if not loc:
            return "XXX"
        return loc[:3].upper()

    location_utils.location_to_airport_code = fake_loc
    location_utils.location_to_city_code = fake_loc

    summary = []

    # Helper to run execute node given state
    async def exec_tools(state):
        out = await agents.execute_tools_node(state)
        # append returned messages into state.messages to simulate persistence
        state_messages = state.get("messages") or []
        state_messages.extend(out.get("messages") or [])
        state["messages"] = state_messages
        # update prev plan
        state["_prev_travel_plan"] = deepcopy(state.get("travel_plan"))
        return out

    # Scenario 1: initial full_plan run (prev=None) -> expect 3 calls
    state = {
        "travel_plan": schemas.TravelPlan(origin="Beijing", destination="Paris", departure_date="2026-05-01", return_date="2026-05-05"),
        "messages": [],
        "last_tool_args": {},
        "current_step": "continue",
        "one_way": False,
    }
    calls.clear()
    await exec_tools(state)
    summary.append(("initial_run", [c[0] for c in calls]))

    # Scenario 2: repeat identical -> expect no new calls
    calls.clear()
    await exec_tools(state)
    summary.append(("repeat_identical", [c[0] for c in calls]))

    # Scenario 3: change departure_date -> expect flights/hotels/activities rerun (dates affect all)
    state["travel_plan"].departure_date = "2026-05-03"
    state["travel_plan"].return_date = "2026-05-07"
    calls.clear()
    await exec_tools(state)
    summary.append(("date_change", [c[0] for c in calls]))

    # Scenario 4: change origin only -> expect flights only
    state["travel_plan"].origin = "Shanghai"
    state["travel_plan"].departure_date = "2026-05-03"
    state["travel_plan"].return_date = "2026-05-07"
    calls.clear()
    await exec_tools(state)
    summary.append(("origin_change", [c[0] for c in calls]))

    # Scenario 5: reuse error placeholder -> create prior error ToolMessage matching current key
    # Build a key for flights using current travel_plan
    cur_plan = state["travel_plan"]
    key = agents._compute_tool_key("search_flights", cur_plan, **agents._semantic_key_kwargs_for_tool(cur_plan, "search_flights", state.get("one_way", False)))
    err_content = json.dumps([{"is_error": True, "error_message": "API down"}], ensure_ascii=False)
    err_msg = ToolMessage(content=err_content, name="search_flights", tool_call_id=f"call_search_flights:{key}:0")
    # Put error message into state and set prev plan to same
    state["messages"].append(err_msg)
    state["_prev_travel_plan"] = deepcopy(cur_plan)
    calls.clear()
    out = await exec_tools(state)
    summary.append(("reuse_error_placeholder", [c[0] for c in calls], {
        "synth_messages": [type(m).__name__ for m in state.get("messages", [])[-2:]]
    }))

    # Print human-readable summary
    print("TEST SUMMARY")
    for item in summary:
        print(item)


if __name__ == "__main__":
    asyncio.run(run())
