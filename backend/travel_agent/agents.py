# backend/travel_agent/agents.py

import asyncio
import json
import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Awaitable, Tuple

from pydantic import ValidationError
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
    _is_refresh_recommendation,
)


# =============================================================================
# Low-signal guard
# =============================================================================

def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
    )


def _is_low_signal_user_input(text: Optional[str]) -> bool:
    t = (text or "").strip()
    if not t:
        return True

    meaningful_count = sum(1 for c in t if c.isalnum() or _is_cjk_char(c))
    if meaningful_count == 0:
        return True

    if re.fullmatch(r"(hi|hello|hey|ok|okay|thanks|thank\s+you|test)\W*", t, flags=re.I):
        return True
    if t in {"好的", "谢谢", "OK", "ok", "嗯", "哈", "哈哈", "收到"}:
        return True

    has_time = bool(re.search(
        r"(\d{4}-\d{1,2}-\d{1,2})|(\d{1,2}\s*(月|/|-)\s*\d{1,2})|"
        r"(today|tomorrow|next\s+week|next\s+\w+day)|"
        r"(今天|明天|后天|下周|周[一二三四五六日天])",
        t, flags=re.I
    ))

    has_travel_kw = bool(re.search(
        r"\b(flight|flights|hotel|hotels|activity|activities|tour|itinerary|airport|"
        r"business|economy|one[-\s]?way|round[-\s]?trip|budget|price)\b",
        t, flags=re.I
    )) or bool(re.search(
        r"(机票|航班|酒店|住宿|活动|行程|预算|商务舱|经济舱|单程|往返|机场|高铁|火车)",
        t
    ))

    has_cjk = any(_is_cjk_char(c) for c in t)

    if len(t) <= 4 and not (has_time or has_travel_kw or has_cjk):
        return True

    ratio = meaningful_count / max(1, len(t))
    if ratio < 0.35 and not (has_time or has_travel_kw) and len(t) < 20:
        return True

    return False


# =============================================================================
# Diff / rerun flags
# =============================================================================

def _changed_fields(prev: TravelPlan, new: TravelPlan) -> set[str]:
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
        "adults", "travel_class", "departure_time_pref", "arrival_time_pref", "user_intent",
    }
    hotels_deps = {"destination", "departure_date", "return_date", "adults", "user_intent"}
    activities_deps = {"destination", "user_intent"}

    rerun_flights = bool(changed & flights_deps)
    rerun_hotels = bool(changed & hotels_deps)
    rerun_activities = bool(changed & activities_deps)

    # ✅ 只改预算：不重跑工具
    if changed == {"total_budget"}:
        rerun_flights = rerun_hotels = rerun_activities = False

    return rerun_flights, rerun_hotels, rerun_activities


def _is_one_way_request(text: str) -> bool:
    t = (text or "").strip().lower()
    patterns = [
        r"单程",
        r"单向",
        r"one[-\s]?way",
        r"\boneway\b",
        r"只要去程",
        r"只看去程",
        r"不返程",
    ]
    return any(re.search(p, t, flags=re.IGNORECASE) for p in patterns)


def _compute_tool_key(tool_name: str, travel_plan: TravelPlan, **kwargs) -> str:
    """
    为工具调用生成唯一指纹 key（由该工具依赖的 plan 字段值拼接后 hash）
    NOTE:
    - key 始终基于“语义参数”（origin/destination 文本），而不是 IATA/city_code。
    - one_way 只使用“最终执行策略”（final policy），不使用 one_way_detected。
      （你现在策略强制往返，则 key 永远 round_trip）
    """
    parts: List[str] = []
    if tool_name == "search_flights":
        one_way_final = bool(kwargs.get("one_way", False))
        parts.extend([
            str(kwargs.get("originLocationCode") or travel_plan.origin or ""),
            str(kwargs.get("destinationLocationCode") or travel_plan.destination or ""),
            str(kwargs.get("departureDate") or travel_plan.departure_date or ""),
            str(kwargs.get("returnDate") or travel_plan.return_date or ""),
            str(travel_plan.adults),
            str(travel_plan.travel_class or ""),
            str(travel_plan.departure_time_pref or ""),
            str(travel_plan.arrival_time_pref or ""),
            "one_way" if one_way_final else "round_trip",
        ])
    elif tool_name == "search_and_compare_hotels":
        parts.extend([
            str(kwargs.get("city_code") or travel_plan.destination or ""),
            str(kwargs.get("check_in_date") or travel_plan.departure_date or ""),
            str(kwargs.get("check_out_date") or travel_plan.return_date or ""),
            str(travel_plan.adults),
        ])
    elif tool_name == "search_activities_by_city":
        parts.extend([
            str(kwargs.get("city_name") or travel_plan.destination or ""),
        ])

    key_str = "|".join(parts)
    return hashlib.md5(key_str.encode()).hexdigest()[:8]


def _extract_tool_key_from_call_id(tool_call_id: str) -> Optional[str]:
    """
    解析 tool_call_id，期望格式：call_<tool>:<key>:<index>
    """
    if not tool_call_id:
        return None
    parts = tool_call_id.split(":")
    if len(parts) >= 3:
        return parts[1]
    return None


def _semantic_key_kwargs_for_tool(travel_plan: TravelPlan, tool_name: str, one_way: bool) -> Dict[str, Any]:
    """
    ✅ 用于 key 的参数永远是“语义参数”，不使用 IATA/city_code。
    ✅ one_way 是“最终执行策略”（final policy），不使用 one_way_detected。
    """
    if tool_name == "search_flights":
        return {
            "originLocationCode": travel_plan.origin or "",
            "destinationLocationCode": travel_plan.destination or "",
            "departureDate": travel_plan.departure_date or "",
            "returnDate": travel_plan.return_date or "",
            "adults": travel_plan.adults,
            "travelClass": travel_plan.travel_class,
            "departureTime": travel_plan.departure_time_pref,
            "arrivalTime": travel_plan.arrival_time_pref,
            "one_way": bool(one_way),
        }
    if tool_name == "search_and_compare_hotels":
        return {
            "city_code": travel_plan.destination or "",
            "check_in_date": travel_plan.departure_date or "",
            "check_out_date": travel_plan.return_date or "",
            "adults": travel_plan.adults,
        }
    if tool_name == "search_activities_by_city":
        return {"city_name": travel_plan.destination or ""}
    return {}


def _safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _tool_content_is_all_error_placeholders(tool_content: str) -> bool:
    data = _safe_json_loads(tool_content or "")
    if not isinstance(data, list) or not data:
        return False
    for item in data:
        if not isinstance(item, dict):
            return False
        if not item.get("is_error", False):
            return False
    return True



# =============================================================================
# PR2: date gate (no more default +15d)
# =============================================================================

def _parse_ymd(date_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None


def _normalize_dates_or_ask(travel_plan: TravelPlan) -> Tuple[bool, str]:
    """
    PR2 关键变化：缺日期就追问，不再默认给“15天后”。
    返回 (ok, ask_message)。
    同时会尽量自动补齐 return_date / departure_date（当信息足够时）。
    """
    if not getattr(travel_plan, "destination", None):
        return False, "Where are you traveling to (destination city/airport)?"

    dep = travel_plan.departure_date
    ret = travel_plan.return_date
    dur = travel_plan.duration_days

    dep_dt = _parse_ymd(dep) if dep else None
    ret_dt = _parse_ymd(ret) if ret else None

    if dep and not dep_dt:
        return False, "What is your departure date? Please use YYYY-MM-DD (e.g., 2026-04-10)."
    if ret and not ret_dt:
        return False, "What is your return date? Please use YYYY-MM-DD (e.g., 2026-04-14)."

    # dep + ret -> 补 duration
    if dep_dt and ret_dt:
        if ret_dt <= dep_dt:
            return False, "Your return date must be after your departure date. Could you confirm the dates?"
        if not dur:
            travel_plan.duration_days = (ret_dt - dep_dt).days
        return True, ""

    # dep + duration -> 补 return
    if dep_dt and dur:
        if dur <= 0:
            return False, "How many days is your trip (a positive number)?"
        travel_plan.return_date = (dep_dt + timedelta(days=dur)).strftime("%Y-%m-%d")
        return True, ""

    # ret + duration -> 补 departure
    if ret_dt and dur:
        if dur <= 0:
            return False, "How many days is your trip (a positive number)?"
        travel_plan.departure_date = (ret_dt - timedelta(days=dur)).strftime("%Y-%m-%d")
        return True, ""

    return (
        False,
        "What are your travel dates (departure & return), or at least a departure date + trip duration (days)? "
        "Example: '2026-04-10 to 2026-04-14' or 'depart 2026-04-10 for 4 days'."
    )


# =============================================================================
# Main node: call_model_node (PR2 Node1~Node4 internally)
# =============================================================================

async def call_model_node(state: TravelAgentState) -> Dict[str, Any]:
    """
    不改 graph、不改对外接口：仍然只有一个 node。
    但内部逻辑按 Node1~Node4 分段组织，便于 PR3 再做自治循环。
    """
    print("━━━ NODE: Analysis & Execution (PR2 Node1-4) ━━━")

    # ------------------------------
    # 获取 last_user_text（必须最先做）
    # ------------------------------
    last_user_text = ""
    try:
        if state.get("messages"):
            last_user_text = state["messages"][-1].content
    except Exception:
        last_user_text = ""

    # ------------------------------
    # one-way 检测：仅用于日志/解释
    # 产品策略：强制往返
    # ------------------------------
    one_way_detected = _is_one_way_request(last_user_text)
    one_way = False  # ✅ final execution policy (forced round-trip)

    if one_way_detected:
        print("→ one_way_detected=True, but product policy forces round-trip (one_way=False)")

    # ------------------------------
    # 低信息拦截
    # ------------------------------
    if _is_low_signal_user_input(last_user_text):
        return {
            "messages": [AIMessage(content="Sorry—I didn't catch that. Could you please repeat your request in English?")],
            "current_step": "complete",
            "form_to_display": None,
            "one_way": one_way,
            "one_way_detected": one_way_detected,
            "last_tool_args": state.get("last_tool_args") or {},
            "execution_plan": state.get("execution_plan"),
        }

    # =========================================================================
    # Node1: ensure_customer_info
    # =========================================================================
    if not state.get("customer_info"):
        original_request = state.get("original_request") or last_user_text
        return {
            "messages": [],
            "current_step": "collecting_info",
            "form_to_display": "customer_info",
            "original_request": original_request,
            "one_way": one_way,
            "one_way_detected": one_way_detected,
            "last_tool_args": state.get("last_tool_args") or {},
            "execution_plan": {"decision": "ASK", "ask": "customer_info", "tasks": []},
        }

    customer_info = state.get("customer_info", {}) or {}

    try:
        # =========================================================================
        # Node2: parse_or_update_plan
        # =========================================================================
        prev_plan: Optional[TravelPlan] = state.get("travel_plan")

        if prev_plan is not None and _is_refresh_recommendation(last_user_text):
            travel_plan = prev_plan
            user_followup_hint = "refresh_recommendation"
        else:
            user_followup_hint = None
            if prev_plan is None:
                user_request = state.get("original_request") or last_user_text
                travel_plan = await enhanced_travel_analysis(user_request)
            else:
                travel_plan = await update_travel_plan(prev_plan, last_user_text)

        # update 后兜底（防 destination/origin 被清空）
        if prev_plan is not None:
            if getattr(travel_plan, "destination", None) in (None, ""):
                travel_plan.destination = prev_plan.destination
            if getattr(travel_plan, "origin", None) in (None, ""):
                travel_plan.origin = prev_plan.origin


        # ------------------------------
        # NEW: deterministic intent override (防 LLM patch 把活动问询写成 full_plan)
        # ------------------------------
        override_intent = _infer_intent_override(last_user_text)
        if override_intent and override_intent != travel_plan.user_intent:
            old = travel_plan.user_intent
            travel_plan.user_intent = override_intent
            print(f"→ intent overridden by heuristics: {old} → {override_intent}")

        # ------------------------------
        # CHANGED: intent change cleanup（改成统一走 helper）
        # ------------------------------
        if prev_plan is not None:
            prev_intent = prev_plan.user_intent
            new_intent = travel_plan.user_intent
            intent_changed = prev_intent != new_intent

            if intent_changed:
                changed = _changed_fields(prev_plan, travel_plan)
                _cleanup_inherited_fields_on_intent(
                    travel_plan,
                    new_intent,
                    changed_fields=changed,
                    user_text=last_user_text,
                )


        # CHANGED: 只有需要航班时才默认 origin
        if travel_plan.user_intent in ["full_plan", "flights_only"] and not travel_plan.origin:
            travel_plan.origin = "Shanghai"
            print("→ Origin not provided, defaulting to Shanghai")


        # =========================================================================
        # Node3: ask_missing_core_fields (destination / dates)
        #   - PR2：缺日期优先追问（不再默认 15 天后）
        # =========================================================================
        needs_dates = travel_plan.user_intent in ["full_plan", "flights_only", "hotels_only"]

        if needs_dates:
            ok, ask_msg = _normalize_dates_or_ask(travel_plan)
            if not ok:
                return {
                    "messages": [AIMessage(content=ask_msg)],
                    "current_step": "complete",
                    "form_to_display": None,
                    "travel_plan": travel_plan,
                    "one_way": one_way,
                    "one_way_detected": one_way_detected,
                    "last_tool_args": state.get("last_tool_args") or {},
                    "user_followup_hint": user_followup_hint,
                    "execution_plan": {"decision": "ASK", "ask": ask_msg, "intent": travel_plan.user_intent, "tasks": []},
                }
        else:
            # activities_only：只需 destination
            if not getattr(travel_plan, "destination", None):
                ask_msg = "Where are you traveling to (destination city/airport)?"
                return {
                    "messages": [AIMessage(content=ask_msg)],
                    "current_step": "complete",
                    "form_to_display": None,
                    "travel_plan": travel_plan,
                    "one_way": one_way,
                    "one_way_detected": one_way_detected,
                    "last_tool_args": state.get("last_tool_args") or {},
                    "user_followup_hint": user_followup_hint,
                    "execution_plan": {"decision": "ASK", "ask": ask_msg, "intent": travel_plan.user_intent, "tasks": []},
                }

        # =========================================================================
        # Diff / rerun flags
        # =========================================================================
        rerun_flights, rerun_hotels, rerun_activities = _compute_rerun_flags(prev_plan, travel_plan)
        intent = travel_plan.user_intent if travel_plan else "full_plan"

        # ✅ effective rerun：只表示“本轮真的会执行的工具”
        eff_rerun_flights = rerun_flights and intent in ["full_plan", "flights_only"]
        eff_rerun_hotels = rerun_hotels and intent in ["full_plan", "hotels_only"]
        eff_rerun_activities = rerun_activities and intent in ["full_plan", "activities_only"]

        print(f"→ Rerun flags (raw): flights={rerun_flights}, hotels={rerun_hotels}, activities={rerun_activities}")
        print(f"→ Rerun flags (effective): flights={eff_rerun_flights}, hotels={eff_rerun_hotels}, activities={eff_rerun_activities}")

        reuse_tools = {
            "flights_only": ["search_flights"],
            "hotels_only": ["search_and_compare_hotels"],
            "activities_only": ["search_activities_by_city"],
            "full_plan": ["search_flights", "search_and_compare_hotels", "search_activities_by_city"],
        }.get(intent, [])

        # =========================================================================
        # Node4: build_execution_plan
        # =========================================================================
        planned_tasks: List[str] = []
        if eff_rerun_flights:
            planned_tasks.append("search_flights")
        if eff_rerun_hotels:
            planned_tasks.append("search_and_compare_hotels")
        if eff_rerun_activities:
            planned_tasks.append("search_activities_by_city")

        execution_plan = {"decision": "EXECUTE", "intent": intent, "tasks": planned_tasks}

        # ---------------------------------------------------------------------
        # 准备工具 + 复用 + 串行执行
        # ---------------------------------------------------------------------
        tasks_and_names: List[tuple[Awaitable, str, Dict[str, Any]]] = []

        departure_date = travel_plan.departure_date
        return_date = travel_plan.return_date

        raw_origin = travel_plan.origin
        raw_dest = travel_plan.destination

        key_args_update: Dict[str, Dict[str, Any]] = {}

        # ---- flights ----
        if eff_rerun_flights and raw_origin and raw_dest and departure_date:
            from .location_utils import location_to_airport_code
            from .config import amadeus as amadeus_client

            origin_iata = await location_to_airport_code(amadeus_client, raw_origin)
            dest_iata = await location_to_airport_code(amadeus_client, raw_dest)

            flight_args = {
                "originLocationCode": origin_iata,
                "destinationLocationCode": dest_iata,
                "departureDate": departure_date,
                "returnDate": return_date,
                "adults": travel_plan.adults,
                "currencyCode": "USD",
                "travelClass": travel_plan.travel_class,
                "departureTime": travel_plan.departure_time_pref,
                "arrivalTime": travel_plan.arrival_time_pref,
            }
            tasks_and_names.append((search_flights.ainvoke(flight_args), "search_flights", flight_args))

            # ✅ key 用语义参数（raw_origin/raw_dest），one_way 用最终执行策略（forced round-trip）
            key_args_update["search_flights"] = {
                "originLocationCode": raw_origin,
                "destinationLocationCode": raw_dest,
                "departureDate": departure_date,
                "returnDate": return_date,
                "adults": travel_plan.adults,
                "travelClass": travel_plan.travel_class,
                "departureTime": travel_plan.departure_time_pref,
                "arrivalTime": travel_plan.arrival_time_pref,
                "one_way": one_way,
            }

        # ---- hotels ----
        if eff_rerun_hotels and raw_dest and departure_date and return_date:
            hotel_args = {
                "city_code": raw_dest,
                "check_in_date": departure_date,
                "check_out_date": return_date,
                "adults": travel_plan.adults,
            }
            tasks_and_names.append((search_and_compare_hotels.ainvoke(hotel_args), "search_and_compare_hotels", hotel_args))

            key_args_update["search_and_compare_hotels"] = {
                "city_code": raw_dest,
                "check_in_date": departure_date,
                "check_out_date": return_date,
                "adults": travel_plan.adults,
            }

        # ---- activities ----
        if eff_rerun_activities and raw_dest:
            act_args = {"city_name": raw_dest}
            tasks_and_names.append((search_activities_by_city.ainvoke(act_args), "search_activities_by_city", act_args))
            key_args_update["search_activities_by_city"] = {"city_name": raw_dest}

        # 合并 last_tool_args（并按 intent 过滤，避免无关缓存污染）
        allowed_tools_for_intent = {
            "flights_only": {"search_flights"},
            "hotels_only": {"search_and_compare_hotels"},
            "activities_only": {"search_activities_by_city"},
            "full_plan": {"search_flights", "search_and_compare_hotels", "search_activities_by_city"},
        }.get(intent, set())

        prev_last_args = state.get("last_tool_args") or {}
        prev_last_args = {k: v for k, v in prev_last_args.items() if k in allowed_tools_for_intent}

        merged_last_args = dict(prev_last_args)
        merged_last_args.update(key_args_update)

        # 工具复用入口
        has_any_tool_history = any(isinstance(m, ToolMessage) for m in state.get("messages", []))

        if not tasks_and_names:
            if has_any_tool_history:
                print("→ No tools needed this turn; reusing previous tool results")
                return {
                    "messages": [],
                    "current_step": "synthesizing",
                    "travel_plan": travel_plan,
                    "form_to_display": None,
                    "tools_used": reuse_tools,
                    "one_way": one_way,
                    "one_way_detected": one_way_detected,
                    "last_tool_args": merged_last_args,
                    "user_followup_hint": user_followup_hint,
                    "execution_plan": {**execution_plan, "tasks": []},
                }

            return {
                "messages": [AIMessage(content="I've understood your request, but there's no specific search I can perform. How else can I help?")],
                "current_step": "complete",
                "travel_plan": travel_plan,
                "form_to_display": None,
                "one_way": one_way,
                "one_way_detected": one_way_detected,
                "last_tool_args": merged_last_args,
                "user_followup_hint": user_followup_hint,
                "execution_plan": {**execution_plan, "tasks": []},
            }

        # 串行执行工具
        print(f"→ Phase: Executing {len(tasks_and_names)} tools sequentially (rate-limit safe)")
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

        for i, (task_coro, tool_name, tool_args) in enumerate(tasks_and_names):
            print(f"→ [{i+1}/{len(tasks_and_names)}] Running tool: {tool_name}")

            key_kwargs = dict((merged_last_args or {}).get(tool_name, {}) or {})
            if tool_name == "search_flights":
                key_kwargs["one_way"] = one_way  # ✅ use final policy only

            current_tool_key = _compute_tool_key(tool_name, travel_plan, **key_kwargs)

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
                    tool_call_id=f"call_{tool_name}:{current_tool_key}:{i}",
                )
            )

            if i < len(tasks_and_names) - 1:
                await asyncio.sleep(1.2)

        print("✓ All tools executed")

        return {
            "messages": processed_messages,
            "current_step": "synthesizing",
            "travel_plan": travel_plan,
            "form_to_display": None,
            "tools_used": reuse_tools,
            "one_way": one_way,
            "one_way_detected": one_way_detected,
            "last_tool_args": merged_last_args,
            "user_followup_hint": user_followup_hint,
            "execution_plan": execution_plan,
        }

    except (ValueError, ValidationError) as e:
        print(f"✗ Analysis failed: {e}")
        return {
            "messages": [AIMessage(content="I'm sorry, I had trouble understanding your request. Could you rephrase it?")],
            "current_step": "complete",
            "form_to_display": None,
            "one_way": one_way,
            "one_way_detected": one_way_detected,
            "last_tool_args": state.get("last_tool_args") or {},
            "execution_plan": state.get("execution_plan"),
        }

    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return {
            "messages": [AIMessage(content="I apologize, but a system error occurred. Please try again.")],
            "current_step": "complete",
            "form_to_display": None,
            "one_way": one_way,
            "one_way_detected": one_way_detected,
            "last_tool_args": state.get("last_tool_args") or {},
            "execution_plan": state.get("execution_plan"),
        }



# ------------------------------------------------------------------------------
# Budget helpers（保持你原实现）
# ------------------------------------------------------------------------------
import re
from typing import Optional, Literal, Set

Intent = Literal["full_plan", "flights_only", "hotels_only", "activities_only"]

# -----------------------------
# NEW: deterministic intent inference
# -----------------------------
def _hit_any(patterns: list[str], text: str) -> bool:
    t = (text or "")
    return any(re.search(p, t, flags=re.IGNORECASE) for p in patterns)

def _text_mentions_date(text: str) -> bool:
    # 简单够用：YYYY-MM-DD / 常见中文相对日期
    return _hit_any(
        [
            r"\b20\d{2}-\d{2}-\d{2}\b",
            r"明天|后天|今天|今晚|下周|下星期|周一|周二|周三|周四|周五|周六|周日",
            r"\d+\s*晚|\d+\s*天",
        ],
        text,
    )

def _infer_intent_override(text: str) -> Optional[Intent]:
    """
    ✅ 防止 LLM patch 乱改 user_intent：当用户文本语义非常明确时，用规则覆盖 intent。
    """
    # 关键词你可以按业务再补
    act_kw = [r"当地体验", r"体验项目", r"体验", r"有什么好玩", r"玩什么", r"活动", r"things to do", r"\bactivities?\b"]
    flight_kw = [r"航班", r"机票", r"飞", r"单程", r"往返", r"商务舱", r"经济舱", r"\bflight(s)?\b", r"one[-\s]?way"]
    hotel_kw = [r"酒店", r"住宿", r"\bhotel(s)?\b", r"stay", r"四星|五星|星级"]

    has_act = _hit_any(act_kw, text)
    has_flight = _hit_any(flight_kw, text)
    has_hotel = _hit_any(hotel_kw, text)

    # ✅ 只有当“单一目标非常明确”时才覆盖（避免误伤）
    if has_act and not has_flight and not has_hotel:
        return "activities_only"
    if has_flight and not has_hotel and not has_act:
        return "flights_only"
    if has_hotel and not has_flight and not has_act:
        return "hotels_only"
    return None

def _cleanup_inherited_fields_on_intent(
    travel_plan: TravelPlan,
    new_intent: Intent,
    *,
    changed_fields: Set[str],
    user_text: str,
) -> None:
    """
    ✅ intent 变化时清理“不该继承 / 高风险继承”的字段
    规则：
    - 只清理本轮没有被用户显式更新的字段（不覆盖用户刚给的值）
    - hotels_only / activities_only：origin 一律不要继承（否则后面 full_plan 会拿到旧出发地）
    - flights_only / hotels_only：dates/duration 高风险继承（没更新就清掉，迫使追问）
    - activities_only：dates 不强制需要；若用户没提日期则清掉（避免污染后续 full_plan）
    """
    def clear_if_not_changed(field: str):
        if field not in changed_fields:
            setattr(travel_plan, field, None)

    if new_intent == "activities_only":
        # 不需要的“航班相关”全部不要继承
        clear_if_not_changed("origin")
        clear_if_not_changed("travel_class")
        clear_if_not_changed("departure_time_pref")
        clear_if_not_changed("arrival_time_pref")
        clear_if_not_changed("total_budget")

        # dates：用户没提日期就清掉，避免继承污染；用户提了日期就保留（以后切 full_plan 可用）
        if not _text_mentions_date(user_text):
            clear_if_not_changed("departure_date")
            clear_if_not_changed("return_date")
            clear_if_not_changed("duration_days")

    elif new_intent == "hotels_only":
        # 酒店不需要 origin / 航班偏好
        clear_if_not_changed("origin")
        clear_if_not_changed("travel_class")
        clear_if_not_changed("departure_time_pref")
        clear_if_not_changed("arrival_time_pref")

        # dates 高风险继承：没更新就清掉 -> 触发追问
        clear_if_not_changed("departure_date")
        clear_if_not_changed("return_date")
        clear_if_not_changed("duration_days")

    elif new_intent == "flights_only":
        # flights 仍然需要 origin/destination/dates，但 dates 也属于高风险继承
        clear_if_not_changed("departure_date")
        clear_if_not_changed("return_date")
        clear_if_not_changed("duration_days")

    # full_plan：不额外清理（你已有逻辑会在 rerun + ask gate 控制）


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
    if travel_plan.total_budget is not None and travel_plan.total_budget > 0:
        return travel_plan.total_budget

    fallback = _parse_budget_to_float(customer_info.get("budget"))
    if fallback is not None and fallback > 0:
        travel_plan.total_budget = fallback
        return fallback

    return None


# ------------------------------------------------------------------------------
# Synthesis node（保持你给的版本）
# ------------------------------------------------------------------------------

async def synthesize_results_node(state: TravelAgentState) -> Dict[str, Any]:
    """
    你的原版本（我未改动逻辑，只确保依赖的 helper 在本文件上半部分都存在）
    """
    print("━━━ NODE: Synthesis & Response ━━━")

    travel_plan = state.get("travel_plan")
    customer_info = state.get("customer_info") or {}

    tools_used = state.get("tools_used", [])
    if tools_used:
        allowed_tools = set(tools_used)
    else:
        intent = travel_plan.user_intent if travel_plan else "full_plan"
        allowed_tools = {
            "flights_only": {"search_flights"},
            "hotels_only": {"search_and_compare_hotels"},
            "activities_only": {"search_activities_by_city"},
            "full_plan": {"search_flights", "search_and_compare_hotels", "search_activities_by_city"},
        }.get(intent, set())

    one_way = state.get("one_way", False)

    # ✅ PR2: derive allowed categories (用于合成裁剪)
    allow_flights = "search_flights" in allowed_tools
    allow_hotels = "search_and_compare_hotels" in allowed_tools
    allow_activities = "search_activities_by_city" in allowed_tools

    current_keys: Dict[str, str] = {}
    if travel_plan:
        for tool_name in ["search_flights", "search_and_compare_hotels", "search_activities_by_city"]:
            key_kwargs = _semantic_key_kwargs_for_tool(travel_plan, tool_name, one_way)
            current_keys[tool_name] = _compute_tool_key(tool_name, travel_plan, **key_kwargs)

    tool_results: Dict[str, str] = {}
    pending = set(allowed_tools)

    messages = state.get("messages", []) or []

    for msg in reversed(messages):
        if not pending:
            break
        if isinstance(msg, ToolMessage) and msg.name in pending:
            stored_key = _extract_tool_key_from_call_id(getattr(msg, "tool_call_id", "") or "")
            if stored_key and stored_key == current_keys.get(msg.name):
                tool_results[msg.name] = msg.content
                pending.remove(msg.name)

    print("🔍 allowed_tools:", allowed_tools)
    print("🔍 current_keys:", {k: current_keys.get(k) for k in allowed_tools})
    print("📦 stored_keys  :", [getattr(m, "tool_call_id", None) for m in messages if isinstance(m, ToolMessage)])
    print("✅ matched tools:", list(tool_results.keys()))
    print("🧪 pending left:", pending)

    if pending:
        for tool_name in list(pending):
            for msg in reversed(messages):
                if isinstance(msg, ToolMessage) and msg.name == tool_name:
                    if _tool_content_is_all_error_placeholders(msg.content):
                        tool_results[tool_name] = msg.content
                        pending.remove(tool_name)
                    break

    if not tool_results and allowed_tools:
        has_any_relevant_toolmsg = any(
            isinstance(m, ToolMessage) and m.name in allowed_tools for m in messages
        )

        if not has_any_relevant_toolmsg:
            synthesis_prompt = """You are an AI travel assistant. You MUST respond in **English**.

IMPORTANT:
- The live **travel search system is temporarily unavailable**, so no concrete flight/hotel/activity options could be retrieved.
- This is a technical issue, not a lack of inventory.

YOUR TASK:
- Clearly explain that the search system is experiencing a temporary outage.
- DO NOT invent or guess any schedules, prices, or availability.
- Suggest the user try again in a few minutes, or book components separately on common OTAs.
- Keep the tone reassuring and practical.
"""
        else:
            synthesis_prompt = """You are an AI travel assistant. You MUST respond in **English**.

IMPORTANT:
- I did run a travel search, but I couldn't reliably associate the returned results with your latest request parameters due to an internal consistency issue.
- This is NOT a claim that airlines/hotels are sold out.

YOUR TASK:
- Apologize briefly.
- Ask the user to retry the same request once (or slightly rephrase).
- Do NOT invent or guess any schedules, prices, or availability.
- Keep the tone reassuring and practical.
"""

        try:
            final_response = await llm.ainvoke(synthesis_prompt)
        except Exception as e:
            print(f"✗ Response generation failed: {e}")
            final_response = AIMessage(
                content="I apologize, but I encountered an issue generating your recommendations. Please try again."
            )

        # ✅ PR2: prune output by allowed_tools (不改 prompt，只裁剪输出段落)
        def _prune_response_by_allowed_tools(text: str) -> str:
            import re
            if not text:
                return text

            out = text

            def _strip_section_md(title_regex: str) -> None:
                nonlocal out
                # Match markdown headings like "## Hotels" until next heading or end
                pat = re.compile(rf"(?is)\n#+\s*{title_regex}\b.*?(?=\n#+\s|\Z)")
                out = re.sub(pat, "\n", out)

            def _strip_section_emoji(emoji: str) -> None:
                nonlocal out
                # Match blocks starting with emoji line until next emoji/heading/end
                pat = re.compile(rf"(?is)\n{re.escape(emoji)}.*?(?=\n(?:✈️|🏨|🎡|\n#+\s)|\Z)")
                out = re.sub(pat, "\n", out)

            # remove disallowed categories
            if not allow_flights:
                _strip_section_md(r"(Flights?|Flight Options?)")
                _strip_section_emoji("✈️")
            if not allow_hotels:
                _strip_section_md(r"(Hotels?|Hotel Availability|Hotel Options?)")
                _strip_section_emoji("🏨")
            if not allow_activities:
                _strip_section_md(r"(Activities|Things to do)")
                _strip_section_emoji("🎡")

            # cleanup excessive blank lines
            out2 = re.sub(r"\n{3,}", "\n\n", out).strip()
            return out2 if out2 else text

        pruned = _prune_response_by_allowed_tools(getattr(final_response, "content", str(final_response)))
        final_response = AIMessage(content=pruned)

        to_email = customer_info.get("email")
        if to_email:
            try:
                await send_email_notification.ainvoke({
                    "to_email": to_email,
                    "subject": "Your AI travel plan",
                    "body": final_response.content,
                })
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

    all_options: Dict[str, list] = {"flights": [], "hotels": [], "activities": []}

    intent = travel_plan.user_intent if travel_plan else "full_plan"
    if intent == "flights_only":
        all_options["hotels"] = []
        all_options["activities"] = []
    elif intent == "hotels_only":
        all_options["flights"] = []
        all_options["activities"] = []
    elif intent == "activities_only":
        all_options["flights"] = []
        all_options["hotels"] = []

    for tool_name, content in tool_results.items():
        try:
            if content and content != "[]":
                parsed_data = json.loads(content)
                if tool_name == "search_flights":
                    all_options["flights"] = [FlightOption.model_validate(f) for f in parsed_data]
                elif tool_name == "search_and_compare_hotels":
                    all_options["hotels"] = [HotelOption.model_validate(h) for h in parsed_data]
                elif tool_name == "search_activities_by_city":
                    all_options["activities"] = [ActivityOption.model_validate(a) for a in parsed_data]
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"✗ Failed to parse {tool_name}: {e}")

    flights_all: List[FlightOption] = all_options.get("flights", [])
    normal_flights: List[FlightOption] = []
    flight_error_message: Optional[str] = None
    for f in flights_all:
        if getattr(f, "is_error", False):
            if not flight_error_message and getattr(f, "error_message", None):
                flight_error_message = f.error_message
        else:
            normal_flights.append(f)
    all_options["flights"] = normal_flights
    if flight_error_message:
        state["flight_error_message"] = flight_error_message

    activities_all: List[ActivityOption] = all_options.get("activities", [])
    normal_activities: List[ActivityOption] = []
    activity_error_message: Optional[str] = None
    for a in activities_all:
        if getattr(a, "is_error", False):
            if not activity_error_message and getattr(a, "error_message", None):
                activity_error_message = a.error_message
        else:
            normal_activities.append(a)
    all_options["activities"] = normal_activities
    if activity_error_message:
        state["activity_error_message"] = activity_error_message

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

    packages: List[TravelPackage] = []
    import re
    from decimal import Decimal

    _PRICE_RE = re.compile(r"(?P<amount>[\d,]+(?:\.\d+)?)\s*(?P<ccy>[A-Z]{3})?$")

    def parse_price(s: str, default_ccy: str = "USD"):
        s = (s or "").strip()
        if s.startswith("$"):
            s = s[1:].strip()
            default_ccy = "USD"
        m = _PRICE_RE.search(s.replace("USD", " USD").replace("JPY", " JPY").strip())
        if not m:
            return None
        amt = Decimal(m.group("amount").replace(",", ""))
        ccy = (m.group("ccy") or default_ccy).upper()
        return amt, ccy

    def convert_to_usd(amount: Decimal, ccy: str, fx: dict[str, Decimal]) -> Decimal:
        ccy = ccy.upper()
        if ccy == "USD":
            return amount
        if ccy not in fx:
            raise ValueError(f"Missing FX rate for {ccy}->USD")
        return amount * fx[ccy]

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

    if packages:
        has_balanced = any(getattr(p, "grade", None) == "Balanced" for p in packages)
        if has_balanced:
            recommend_line = '- Highlight the "Balanced" package as recommended'
        else:
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
    else:
        flights_exist = bool(all_options["flights"])
        hotels_exist = bool(all_options["hotels"])
        activities_exist = bool(all_options["activities"])
        has_any_results = flights_exist or hotels_exist or activities_exist

        if flight_error_message and (hotels_exist or activities_exist):
            tool_results_for_prompt = {
                "flights": [],
                "hotels": [h.model_dump() for h in all_options.get("hotels", [])],
                "activities": [a.model_dump() for a in all_options.get("activities", [])],
            }
            destination = travel_plan.destination if travel_plan else ""
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
                "note": ["Flight API temporarily unavailable", flight_error_message, activity_error_message],
            }

        elif activity_error_message and (flights_exist or hotels_exist):
            tool_results_for_prompt = {
                "flights": [f.model_dump() for f in all_options.get("flights", [])],
                "hotels": [h.model_dump() for h in all_options.get("hotels", [])],
                "activities": [],
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
                "note": ["Activity API temporarily unavailable", activity_error_message],
            }

        elif hotel_error_message and (flights_exist or activities_exist):
            tool_results_for_prompt = {
                "flights": [f.model_dump() for f in all_options.get("flights", [])],
                "hotels": [],
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

        # ✅ PR2: 仅在“允许酒店的意图场景”才进入“无酒店库存”解释分支，避免 flights_only 误触发
        elif flights_exist and (allow_hotels) and not hotels_exist:
            tool_results_for_prompt = {
                "flights": [f.model_dump() for f in all_options.get("flights", [])],
                "activities": [a.model_dump() for a in all_options.get("activities", [])],
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

        elif has_any_results:
            tool_results_for_prompt = {
                "flights": [f.model_dump() for f in all_options.get("flights", [])],
                "hotels": [h.model_dump() for h in all_options.get("hotels", [])],
                "activities": [a.model_dump() for a in all_options.get("activities", [])],
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

    try:
        final_response = await llm.ainvoke(synthesis_prompt)
    except Exception as e:
        print(f"✗ Response generation failed: {e}")
        final_response = AIMessage(
            content="I apologize, but I encountered an issue generating your recommendations. Please try again."
        )

    # ✅ PR2: prune output by allowed_tools (不改 prompt，只裁剪输出段落)
    def _prune_response_by_allowed_tools(text: str) -> str:
        import re
        if not text:
            return text

        out = text

        def _strip_section_md(title_regex: str) -> None:
            nonlocal out
            pat = re.compile(rf"(?is)\n#+\s*{title_regex}\b.*?(?=\n#+\s|\Z)")
            out = re.sub(pat, "\n", out)

        def _strip_section_emoji(emoji: str) -> None:
            nonlocal out
            pat = re.compile(rf"(?is)\n{re.escape(emoji)}.*?(?=\n(?:✈️|🏨|🎡|\n#+\s)|\Z)")
            out = re.sub(pat, "\n", out)

        if not allow_flights:
            _strip_section_md(r"(Flights?|Flight Options?)")
            _strip_section_emoji("✈️")
        if not allow_hotels:
            _strip_section_md(r"(Hotels?|Hotel Availability|Hotel Options?)")
            _strip_section_emoji("🏨")
        if not allow_activities:
            _strip_section_md(r"(Activities|Things to do)")
            _strip_section_emoji("🎡")

        out2 = re.sub(r"\n{3,}", "\n\n", out).strip()
        return out2 if out2 else text

    pruned = _prune_response_by_allowed_tools(getattr(final_response, "content", str(final_response)))
    final_response = AIMessage(content=pruned)

    to_email = customer_info.get("email")
    if to_email:
        try:
            subject = f"Your AI travel plan to {travel_plan.destination}" if travel_plan else "Your AI travel plan"
            body = getattr(final_response, "content", str(final_response))
            await send_email_notification.ainvoke({"to_email": to_email, "subject": subject, "body": body})
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

