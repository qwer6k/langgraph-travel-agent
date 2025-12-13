import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Awaitable
from langchain_core.messages import AIMessage, ToolMessage

from .config import llm
from .schemas import (
    TravelAgentState,
    TravelPlan,
    FlightOption,
    HotelOption,
    ActivityOption,
    TravelPackage,
)
from .tools import (
    enhanced_travel_analysis,
    update_travel_plan, 
    search_flights,
    search_and_compare_hotels,
    search_activities_by_city,
    generate_travel_packages,
    send_to_hubspot,
    send_email_notification,
)


def _calculate_default_dates(travel_plan: TravelPlan) -> tuple[str, str]:
    """
    根据当前时间 + duration 自动兜底出发/返回日期。
    """
    today = datetime.now()
    default_checkin = today + timedelta(days=15)
    default_checkout = default_checkin + timedelta(days=3)

    departure_date = travel_plan.departure_date
    return_date = travel_plan.return_date

    if not departure_date:
        departure_date = default_checkin.strftime("%Y-%m-%d")

    if not return_date:
        if travel_plan.duration_days:
            try:
                dep_dt = datetime.strptime(departure_date, "%Y-%m-%d")
                return_dt = dep_dt + timedelta(days=travel_plan.duration_days)
                return_date = return_dt.strftime("%Y-%m-%d")
            except ValueError:
                return_date = default_checkout.strftime("%Y-%m-%d")
        else:
            return_date = default_checkout.strftime("%Y-%m-%d")

    return departure_date, return_date


def _changed_fields(prev: TravelPlan, new: TravelPlan) -> set[str]:
    """
    粗粒度 diff：比较 model_dump 后 key 的值是否变化。
    """
    a = prev.model_dump()
    b = new.model_dump()
    return {k for k in a.keys() if a.get(k) != b.get(k)}


def _compute_rerun_flags(prev: Optional[TravelPlan], new: TravelPlan) -> tuple[bool, bool, bool]:
    """
    返回 (rerun_flights, rerun_hotels, rerun_activities)
    prev 为 None 表示首次规划：全跑。
    """
    if prev is None:
        return True, True, True

    changed = _changed_fields(prev, new)

    flights_deps = {
        "origin", "destination", "departure_date", "return_date",
        "adults", "travel_class", "departure_time_pref", "arrival_time_pref",
    }
    hotels_deps = {"destination", "departure_date", "return_date", "adults"}
    activities_deps = {"destination"}

    rerun_flights = bool(changed & flights_deps)
    rerun_hotels = bool(changed & hotels_deps)
    rerun_activities = bool(changed & activities_deps)

    # ✅ 只改预算：不重跑工具，复用历史 ToolMessage
    if changed == {"total_budget"}:
        rerun_flights = rerun_hotels = rerun_activities = False

    return rerun_flights, rerun_hotels, rerun_activities


def _is_one_way_request(text: str) -> bool:
    import re
    t = (text or "").strip().lower()
    patterns = [
        r"单程",
        r"one[-\s]?way",
        r"\boneway\b",
        r"只要去程",
        r"只看去程",
        r"不返程",
    ]
    return any(re.search(p, t, flags=re.IGNORECASE) for p in patterns)


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------

async def call_model_node(state: TravelAgentState) -> Dict[str, Any]:
    """
    分析节点（Analysis Agent）：
    1. 决定是否先要用户填写 customer_info 表单（多轮期间持续锁定）
    2. 基于 original_request / 上一版 travel_plan 做增量解析
    3. diff gating：决定哪些工具需要重跑（预算/措辞更新不重跑）
    4. 串行执行需要重跑的工具（带间隔，保护 QPS），写入 ToolMessage
    """
    print("━━━ NODE: Analysis & Execution ━━━")

    # 兼容保留（前端可能传，也可能不传）
    _ = state.get("is_continuation", False)

    # ------------------------------------------------------------------
    # collecting_info：只要 customer_info 缺失，就一直停在这里
    # original_request 只记录一次，后续用户继续说话不要覆盖它
    # ------------------------------------------------------------------
    if not state.get("customer_info"):
        original_request = state.get("original_request")
        if not original_request:
            original_request = state["messages"][-1].content

        return {
            "messages": [],
            "current_step": "collecting_info",
            "form_to_display": "customer_info",
            "original_request": original_request,
        }

    customer_info = state.get("customer_info", {}) or {}

    try:
        # ==============================
        # Phase 1: 解析/更新 TravelPlan
        # ==============================
        print("→ Phase 1: Analyzing request")

        last_user_text = state["messages"][-1].content
        prev_plan: Optional[TravelPlan] = state.get("travel_plan")

        if prev_plan is None:
            user_request = state.get("original_request") or last_user_text
            travel_plan = await enhanced_travel_analysis(user_request)
        else:
            travel_plan = await update_travel_plan(prev_plan, last_user_text)

        # 默认出发地
        if not travel_plan.origin:
            travel_plan.origin = "Shanghai"
            print("→ Origin not provided, defaulting to Shanghai")

        if customer_info.get("budget"):
            print(f"→ Budget captured (not injected in analysis): {customer_info.get('budget')}")

        # diff gating（在写回 state 之前用 prev_plan）
        rerun_flights, rerun_hotels, rerun_activities = _compute_rerun_flags(prev_plan, travel_plan)
        print(f"→ Rerun flags: flights={rerun_flights}, hotels={rerun_hotels}, activities={rerun_activities}")

        # 写回
        state["travel_plan"] = travel_plan

        # ------------------------------------------------------------------
        # ✅ 给 synthesize 用：按 intent 决定“本轮应当读取/复用”的工具范围
        #    注意：这是“范围”，不是“本轮实际执行的工具集合”
        # ------------------------------------------------------------------
        intent = travel_plan.user_intent if travel_plan else "full_plan"
        reuse_tools = {
            "flights_only": ["search_flights"],
            "hotels_only": ["search_and_compare_hotels"],
            "activities_only": ["search_activities_by_city"],
            "full_plan": ["search_flights", "search_and_compare_hotels", "search_activities_by_city"],
        }.get(intent, [])

        # ==============================
        # Phase 2: 准备要调用的工具（按 rerun gate）
        # ==============================
        print(f"→ Phase 2: Preparing tools (intent: {travel_plan.user_intent})")

        tasks_and_names: List[tuple[Awaitable, str]] = []

        default_checkin, default_checkout = _calculate_default_dates(travel_plan)
        departure_date = travel_plan.departure_date or default_checkin
        return_date = travel_plan.return_date or default_checkout

        # 日期合法性兜底
        try:
            datetime.strptime(departure_date, "%Y-%m-%d")
            if return_date:
                datetime.strptime(return_date, "%Y-%m-%d")
        except ValueError as e:
            print(f"⚠ Invalid date, using defaults: {e}")
            departure_date = default_checkin
            return_date = default_checkout

        # ---- flights ----
        if (
            rerun_flights
            and travel_plan.user_intent in ["full_plan", "flights_only"]
            and travel_plan.origin
            and travel_plan.destination
        ):
            latest_user_text = (
                state["messages"][-1].content
                or state.get("original_request")
                or ""
            )
            one_way = _is_one_way_request(latest_user_text)

            flight_args = {
                "originLocationCode": travel_plan.origin,
                "destinationLocationCode": travel_plan.destination,
                "departureDate": departure_date,
                "adults": travel_plan.adults,
                "currencyCode": "USD",
                "travelClass": travel_plan.travel_class,
                "departureTime": travel_plan.departure_time_pref,
                "arrivalTime": travel_plan.arrival_time_pref,
            }

            if (not one_way) and return_date:
                flight_args["returnDate"] = return_date
            else:
                travel_plan.return_date = None

            tasks_and_names.append((search_flights.ainvoke(flight_args), "search_flights"))

        # ---- hotels ----
        if (
            rerun_hotels
            and travel_plan.user_intent in ["full_plan", "hotels_only"]
            and travel_plan.destination
        ):
            hotel_args = {
                "city_code": travel_plan.destination,
                "check_in_date": departure_date,
                "check_out_date": return_date,
                "adults": travel_plan.adults,
            }
            tasks_and_names.append((search_and_compare_hotels.ainvoke(hotel_args), "search_and_compare_hotels"))

        # ---- activities ----
        if (
            rerun_activities
            and travel_plan.user_intent in ["full_plan", "activities_only"]
            and travel_plan.destination
        ):
            tasks_and_names.append(
                (search_activities_by_city.ainvoke({"city_name": travel_plan.destination}), "search_activities_by_city")
            )

        # ==============================
        # Phase 2.5: 工具复用入口（关键）
        # ==============================
        has_any_tool_history = any(
            isinstance(m, ToolMessage) for m in state.get("messages", [])
        )

        if not tasks_and_names:
            if has_any_tool_history:
                print("→ No tools needed this turn; reusing previous tool results")
                return {
                    "messages": [],
                    "current_step": "synthesizing",
                    "travel_plan": travel_plan,
                    "form_to_display": None,
                    "tools_used": reuse_tools,  # ✅ 按 intent 给全范围
                }

            print("⚠ No tools to call and no previous tool history")
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I've understood your request, but there's no specific "
                            "search I can perform. How else can I help?"
                        )
                    )
                ],
                "current_step": "complete",
                "travel_plan": travel_plan,
                "form_to_display": None,
            }

        # ==============================
        # Phase 3: 串行执行工具（带间隔）
        # ==============================
        print(f"→ Phase 3: Executing {len(tasks_and_names)} tools sequentially (rate-limit safe)")

        processed_messages: List[ToolMessage] = []

        def _tool_error_placeholder(tool_name: str, err: Exception) -> str:
            msg = f"{type(err).__name__}: {err}"
            msg = (msg[:500] + "…") if len(msg) > 500 else msg

            if tool_name == "search_flights":
                payload = [{
                    "airline": "API_ERROR",
                    "price": "N/A",
                    "departure_time": "N/A",
                    "arrival_time": "N/A",
                    "duration": None,
                    "is_error": True,
                    "error_message": msg,
                }]
            elif tool_name == "search_and_compare_hotels":
                payload = [{
                    "name": "API_ERROR",
                    "category": "N/A",
                    "price_per_night": "N/A",
                    "source": "SYSTEM",
                    "rating": None,
                    "is_error": True,
                    "error_message": msg,
                }]
            elif tool_name == "search_activities_by_city":
                payload = [{
                    "name": "API_ERROR",
                    "description": "Activity API error",
                    "price": "N/A",
                    "location": None,
                    "is_error": True,
                    "error_message": msg,
                }]
            else:
                payload = [{"is_error": True, "error_message": msg}]

            return json.dumps(payload, ensure_ascii=False)

        for i, (task_coro, tool_name) in enumerate(tasks_and_names):
            print(f"→ [{i+1}/{len(tasks_and_names)}] Running tool: {tool_name}")
            try:
                result = await task_coro
                try:
                    content = json.dumps([item.model_dump() for item in result], ensure_ascii=False)
                except Exception as e:
                    print(f"✗ Serialization failed for {tool_name}: {e}")
                    content = _tool_error_placeholder(tool_name, e)

            except Exception as e:
                print(f"✗ Tool {tool_name} failed: {e}")
                content = _tool_error_placeholder(tool_name, e)

            processed_messages.append(
                ToolMessage(
                    content=content,
                    name=tool_name,
                    tool_call_id=f"call_{tool_name}_{i}",
                )
            )

            if i < len(tasks_and_names) - 1:
                await asyncio.sleep(1.2)

        print("✓ All tools executed")

        # ✅ tools_used 不用“本轮跑了哪些”，而是“本轮 synthesize 应读取的范围”
        tools_used = reuse_tools

        return {
            "messages": processed_messages,
            "current_step": "synthesizing",
            "travel_plan": travel_plan,
            "form_to_display": None,
            "tools_used": tools_used,
        }

    except ValueError as e:
        print(f"✗ Analysis failed: {e}")
        return {
            "messages": [
                AIMessage(content="I'm sorry, I had trouble understanding your request. Could you rephrase it?")
            ],
            "current_step": "complete",
            "form_to_display": None,
        }

    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return {
            "messages": [
                AIMessage(content="I apologize, but a system error occurred. Please try again.")
            ],
            "current_step": "complete",
            "form_to_display": None,
        }




import re
from typing import Optional

def _parse_budget_to_float(raw: object) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s = s.upper().replace("USD", "").replace("$", "").replace(",", "").strip()
    m = re.search(r"\d+(\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None

def _ensure_budget_for_packages(travel_plan: TravelPlan, customer_info: dict) -> Optional[float]:
    # 优先使用用户话里解析到的预算
    if travel_plan.total_budget is not None and travel_plan.total_budget > 0:
        return travel_plan.total_budget

    # 兜底使用表单预算
    fallback = _parse_budget_to_float(customer_info.get("budget"))
    if fallback is not None and fallback > 0:
        travel_plan.total_budget = fallback  # ✅ 只为生成套餐写回
        return fallback

    return None

async def synthesize_results_node(state: TravelAgentState) -> Dict[str, Any]: 
    """
    综合节点（Synthesis Agent）：
    1. 把工具 ToolMessage 的 JSON 解析成 Flight/Hotel/Activity 对象
    2.（如果有预算 && 有机票 && 有酒店）调用套餐生成器
    3. 调用 LLM 生成最终用户话术
    4. 把结果同步到 CRM + 给用户发邮件

    额外处理：
    - 识别 FlightOption / ActivityOption 中的 is_error / error_message，
      在航班 / 活动 API 挂掉时优雅降级，不编造数据。
    """
    print("━━━ NODE: Synthesis & Response ━━━")
    travel_plan = state.get("travel_plan")
    ##新增:synthesize 只读「本轮该读」的工具
    tools_used = state.get("tools_used", [])
    if tools_used:
        allowed_tools = set(tools_used)
    else:                       # 兜底用 intent
        intent = travel_plan.user_intent if travel_plan else "full_plan"
        allowed_tools = {
                "flights_only": {"search_flights"},
                "hotels_only": {"search_and_compare_hotels"},
                "activities_only": {"search_activities_by_city"},
                "full_plan": {"search_flights", "search_and_compare_hotels", "search_activities_by_city"},
            }.get(intent, set())

    tool_results: Dict[str, str] = {}
    for msg in state.get("messages", []):
        if isinstance(msg, ToolMessage) and msg.name in allowed_tools:
            try:
                tool_results[msg.name] = msg.content
            except Exception as e:
                print(f"⚠ Failed to process {msg.name}: {e}")
                tool_results[msg.name] = "[]"

    customer_info = state.get("customer_info") or {}
    

    # 解析工具返回为结构化 options
    all_options: Dict[str, list] = {"flights": [], "hotels": [], "activities": []}
    for tool_name, content in tool_results.items():
        try:
            if content and content != "[]":
                parsed_data = json.loads(content)
                if tool_name == "search_flights":
                    all_options["flights"] = [
                        FlightOption.model_validate(f) for f in parsed_data
                    ]
                elif tool_name == "search_and_compare_hotels":
                    all_options["hotels"] = [
                        HotelOption.model_validate(h) for h in parsed_data
                    ]
                elif tool_name == "search_activities_by_city":
                    all_options["activities"] = [
                        ActivityOption.model_validate(a) for a in parsed_data
                    ]
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"✗ Failed to parse {tool_name}: {e}")

    # ------------------------------------------------------------------
    # 额外处理：区分“正常航班结果”和“航班 API 错误占位”
    # ------------------------------------------------------------------
    flights_all: List[FlightOption] = all_options.get("flights", [])
    normal_flights: List[FlightOption] = []
    flight_error_message: Optional[str] = None

    for f in flights_all:
        # 兼容：没有 is_error 字段时，默认 False
        if getattr(f, "is_error", False):
            if not flight_error_message and getattr(f, "error_message", None):
                flight_error_message = f.error_message
        else:
            normal_flights.append(f)

    # 用处理后的“正常航班”覆盖回去
    all_options["flights"] = normal_flights

    # 你也可以把错误信息挂回 state，方便以后调试
    if flight_error_message:
        state["flight_error_message"] = flight_error_message

    # ------------------------------------------------------------------
    # 额外处理：区分“正常活动结果”和“活动 API 错误占位”
    # ------------------------------------------------------------------
    activities_all: List[ActivityOption] = all_options.get("activities", [])
    normal_activities: List[ActivityOption] = []
    activity_error_message: Optional[str] = None

    for a in activities_all:
        # 同样兼容：没有 is_error 字段时，默认 False
        if getattr(a, "is_error", False):
            if not activity_error_message and getattr(a, "error_message", None):
                activity_error_message = a.error_message
        else:
            normal_activities.append(a)

    # 覆盖回“正常活动”
    all_options["activities"] = normal_activities

    if activity_error_message:
        state["activity_error_message"] = activity_error_message

    # ------------------------------------------------------------------
    # 额外处理：区分“正常酒店结果”和“酒店 API 错误占位”
    # ------------------------------------------------------------------
    hotels_all: List[HotelOption] = all_options.get("hotels", [])
    normal_hotels: List[HotelOption] = []
    hotel_error_message: Optional[str] = None

    for h in hotels_all:
        if getattr(h, "is_error", False):
            if not hotel_error_message and getattr(h, "error_message", None):
                hotel_error_message = h.error_message
        else:
            normal_hotels.append(h)

    all_options["hotels"] = normal_hotels

    if hotel_error_message:
        state["hotel_error_message"] = hotel_error_message


    # ------------------------------------------------------------------
    # 尝试生成套餐（仅在真实有机票 + 酒店时）
    # ------------------------------------------------------------------
    packages: List[TravelPackage] = []
    if (
        travel_plan
        and travel_plan.user_intent == "full_plan"
        and all_options["flights"]
        and all_options["hotels"]
    ):
        budget_for_packages = _ensure_budget_for_packages(travel_plan, customer_info)
        if budget_for_packages:
            print(f"→ Generating travel packages (budget=${budget_for_packages})")
            try:
                packages = await generate_travel_packages(travel_plan, all_options)
            except Exception as e:
                print(f"✗ Package generation failed: {e}")
                packages = []
        else:
            print("→ Skip package generation: no budget available")


    synthesis_prompt = ""
    hubspot_recommendations: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1) 有完整套餐
    # ------------------------------------------------------------------
    if packages:
        has_balanced = any(getattr(p, "grade", None) == "Balanced" for p in packages)

        if has_balanced:
            recommend_line = '- Highlight the "Balanced" package as recommended'
        else:
            # 没有 Balanced 时：推荐一个替代策略（例如推荐第一个/中位/最便宜）
            recommend_line = f'- Recommend the "{packages[0].name}" package as the best choice'

        synthesis_prompt = f"""You are an AI travel assistant. You MUST respond in **English**.

Present these custom travel packages professionally.
**GENERATED PACKAGES:**
{json.dumps([p.model_dump() for p in packages], indent=2)}

**YOUR TASK:**
- Start with a warm greeting
- Present ALL packages with clear details (flight, hotel, activities)
{recommend_line}
- End with clear call to action
"""


    # ------------------------------------------------------------------
    # 2) 没有套餐，根据工具结果和错误信息降级
    # ------------------------------------------------------------------
    else:
        flights_exist = bool(all_options["flights"])
        hotels_exist = bool(all_options["hotels"])
        activities_exist = bool(all_options["activities"])
        has_any_results = flights_exist or hotels_exist or activities_exist

        # 2.1 航班 API 挂了，但酒店 / 活动有结果
        if flight_error_message and (hotels_exist or activities_exist):
            tool_results_for_prompt = {
                "flights": [],  # 没有真实航班数据可展示
                "hotels": [h.model_dump() for h in all_options.get("hotels", [])],
                "activities": [
                    a.model_dump() for a in all_options.get("activities", [])
                ],
            }

            destination = travel_plan.destination if travel_plan else ""

            # 如果活动也挂了，就一起在技术说明里提一下
            activity_error_note = (
                f'\nActivity search also failed with internal error:\n"{activity_error_message}"'
                if activity_error_message
                else ""
            )

            synthesis_prompt = f"""You are an AI travel assistant.You MUST respond in **English**.

IMPORTANT:
- The live **flight search API failed**, so you DO NOT have any concrete flight options to show.
- You DO have real-time results for hotels and/or activities.
{activity_error_note}

Destination: {destination}

Technical note about the flight error (summarize in simple terms if needed):
"{flight_error_message}"

Using the structured data below:

{json.dumps(tool_results_for_prompt, indent=2)}

YOUR TASK:
- Clearly explain to the user that flight search is temporarily unavailable.
- DO NOT invent or guess any specific flight numbers, schedules or prices.
- Present the available hotel and activity options in a clear, friendly way.
- Suggest how the user can independently look up flights
  (e.g. airline websites or common OTAs), while using these hotels/activities as a base plan.
- Keep the tone reassuring and practical.
"""
            hubspot_recommendations = {
                "flights": [],
                "hotels": tool_results_for_prompt["hotels"],
                "activities": tool_results_for_prompt["activities"],
                "note": [
                    "Flight API temporarily unavailable",
                    flight_error_message,
                    activity_error_message,
                ],
            }

        # 2.2 活动 API 挂了，但有航班 / 酒店结果
        elif activity_error_message and (flights_exist or hotels_exist):
            tool_results_for_prompt = {
                "flights": [f.model_dump() for f in all_options.get("flights", [])],
                "hotels": [h.model_dump() for h in all_options.get("hotels", [])],
                "activities": [],  # 没有真实活动可展示
            }

            destination = travel_plan.destination if travel_plan else ""

            synthesis_prompt = f"""You are an AI travel assistant.You MUST respond in **English**.

IMPORTANT:
- The live **activity search API failed**, so you DO NOT have any concrete activity options to show.
- You DO have real-time results for flights and/or hotels.

Destination: {destination}

Technical note about the activity error (summarize in simple terms if needed):
"{activity_error_message}"

Using the structured data below:

{json.dumps(tool_results_for_prompt, indent=2)}

YOUR TASK:
- Clearly explain to the user that activity search is temporarily unavailable.
- DO NOT invent or guess any specific activity names, schedules or prices.
- Present the available flight and/or hotel options in a clear, friendly way.
- Give some high-level suggestions on what types of activities are usually popular in {destination},
  but make it clear these are generic ideas, not live offers.
- Keep the tone reassuring and practical.
"""
            hubspot_recommendations = {
                "flights": tool_results_for_prompt["flights"],
                "hotels": tool_results_for_prompt["hotels"],
                "activities": [],
                "note": [
                    "Activity API temporarily unavailable",
                    activity_error_message,
                ],
            }
        # 2.x 酒店 API 挂了，但有航班 / 活动结果
        elif hotel_error_message and (flights_exist or activities_exist):
            tool_results_for_prompt = {
                "flights": [f.model_dump() for f in all_options.get("flights", [])],
                "hotels": [],  # 没有真实酒店可展示
                "activities": [a.model_dump() for a in all_options.get("activities", [])],
            }

            destination = travel_plan.destination if travel_plan else ""

            synthesis_prompt = f"""You are an AI travel assistant.You MUST respond in **English**.

        IMPORTANT:
        - The live **hotel availability search failed**, so you DO NOT have any concrete hotel options to show.
        - You DO have real-time results for flights and/or activities.

        Destination: {destination}

        Technical note about the hotel error (summarize in simple terms if needed):
        "{hotel_error_message}"

        Using the structured data below:

        {json.dumps(tool_results_for_prompt, indent=2)}

        YOUR TASK:
        - Clearly explain to the user that hotel search is temporarily unavailable.
        - DO NOT invent or guess any specific hotel names, availability, or prices.
        - Present the available flight and/or activity options in a clear, friendly way.
        - Suggest concrete next steps for hotels (e.g. OTAs), or changing dates to retry.
        - Keep the tone reassuring and practical.
        """
            hubspot_recommendations = {
                "flights": tool_results_for_prompt["flights"],
                "hotels": [],
                "activities": tool_results_for_prompt["activities"],
                "note": ["Hotel API temporarily unavailable", hotel_error_message],
            }

        # 2.3 有机票 & 活动，但没有酒店（你原来的逻辑，保留）
        elif flights_exist and not hotels_exist:
            tool_results_for_prompt = {
                "flights": [f.model_dump() for f in all_options.get("flights", [])],
                "activities": [
                    a.model_dump() for a in all_options.get("activities", [])
                ],
            }

            destination = travel_plan.destination if travel_plan else ""

            synthesis_prompt = f"""You are an AI travel assistant.You MUST respond in **English**.

We successfully found **flight options and activities**, but **no real-time hotel availability** for the requested dates from our inventory providers (Amadeus / Hotelbeds).

Destination: {destination}

Using the structured data below:

**SEARCH RESULTS (no real-time hotels):**
{json.dumps(tool_results_for_prompt, indent=2)}

YOUR TASK:
- Clearly present the available flight options (prices, times, airlines).
- Clearly present the available activity options (what they are, prices if any).
- Explain in simple language that:
  - For these exact dates, our suppliers did not return any bookable hotel rooms.
  - This is likely due to fully booked inventory or supplier restrictions (e.g. stay too long, blackout dates).
- Give 2–3 suggestions of popular areas/neighbourhoods to stay in {destination}, with approximate nightly budget ranges
  (make it clear these are **guidelines only**, not live offers).
- Suggest concrete next steps, e.g.:
  - Try adjusting the travel dates (shorten the stay or shift by a few days), and we can search again.
  - Or book hotels manually on common OTA platforms while keeping these flights & activities as a reference.
- Keep the tone reassuring and helpful.
"""
            hubspot_recommendations = {
                "flights": tool_results_for_prompt["flights"],
                "hotels": [],
                "activities": tool_results_for_prompt["activities"],
                "note": ["No real-time hotel inventory for requested dates"],
            }

        # 2.4 有部分结果（不一定三种都有），且没有航班 / 活动 API 错误
        elif has_any_results:
            tool_results_for_prompt = {
                "flights": [f.model_dump() for f in all_options.get("flights", [])],
                "hotels": [h.model_dump() for h in all_options.get("hotels", [])],
                "activities": [
                    a.model_dump() for a in all_options.get("activities", [])
                ],
            }
            synthesis_prompt = f"""You are an AI travel assistant.You MUST respond in **English**.
Present these search results clearly.
**SEARCH RESULTS:**
{json.dumps(tool_results_for_prompt, indent=2)}

Organize and present options in a user-friendly format.
- Group by Flights / Hotels / Activities.
- If any category is empty, clearly mention it.
- Highlight a few good-value choices (but do not invent any data).
"""
            hubspot_recommendations = tool_results_for_prompt

        # 2.5 完全没有结果：区分“真没结果”与“API 挂了”（航班 / 活动任意一个）
        else:
            if flight_error_message or activity_error_message:
                failure_msgs = []
                if flight_error_message:
                    failure_msgs.append(f'Flights: "{flight_error_message}"')
                if activity_error_message:
                    failure_msgs.append(f'Activities: "{activity_error_message}"')
                failure_str = "\n".join(f"- {m}" for m in failure_msgs)

                synthesis_prompt = f"""You are an AI travel assistant.You MUST respond in **English**.

The real-time travel search system returned internal errors:

{failure_str}

There are currently **no structured results** for flights, hotels or activities.

YOUR TASK:
- Explain in simple, user-friendly language that the travel search system
  is temporarily unavailable for this request.
- Do NOT invent or guess any concrete flight numbers, activity details, times or prices.
- Offer concrete next steps:
  - Try again later.
  - Or check flights / hotels / activities directly on common OTA platforms,
    then come back with the chosen dates so you can help plan the rest.
- Keep the tone apologetic but proactive and reassuring.
"""
                hubspot_recommendations = {
                    "error": "Supplier API failure",
                    "details": {
                        "flight_error_message": flight_error_message,
                        "activity_error_message": activity_error_message,
                    },
                }
            else:
                synthesis_prompt = """You are an AI travel assistant. You MUST respond in **English**.
Apologize that no options were found and offer to help refine the search.
Explain that supplier inventory may be empty for these exact parameters.
Offer specific ways to adjust:
- Change travel dates
- Try nearby airports or cities
- Adjust budget or constraints.
"""
                hubspot_recommendations = {"error": "No results found"}

    # ------------------------------------------------------------------
    # 调用 LLM 生成最终回复
    # ------------------------------------------------------------------
    try:
        final_response = await llm.ainvoke(synthesis_prompt)
    except Exception as e:
        print(f"✗ Response generation failed: {e}")
        final_response = AIMessage(
            content=(
                "I apologize, but I encountered an issue generating your "
                "recommendations. Please try again."
            ),
        )

    # 邮件通知
    # customer_info = state.get("customer_info") or {}
    to_email = customer_info.get("email")

    if to_email:
        try:
            subject = (
                f"Your AI travel plan to {travel_plan.destination}"
                if travel_plan
                else "Your AI travel plan"
            )
            body = getattr(final_response, "content", str(final_response))

            await send_email_notification.ainvoke(
                {
                    "to_email": to_email,
                    "subject": subject,
                    "body": body,
                },
            )
            print(f"✓ Email sent to customer email: {to_email}")
        except Exception as e:
            print(f"✗ Failed to send email to customer: {e}")
    else:
        print("⚠ No email found in customer_info, skip email notification.")

    return {
        "messages": [final_response],
        "current_step": "complete",
        "form_to_display": None,
    }

