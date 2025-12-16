import asyncio
import json
import random
import functools
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Literal, Any, Awaitable
import httpx
from amadeus import ResponseError
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, ToolMessage
from .config import amadeus, llm, EMAIL_SENDER, EMAIL_PASSWORD, HUBSPOT_API_KEY, hotelbeds_headers
from .schemas import (
    FlightOption,
    HotelOption,
    ActivityOption,
    TravelAgentState,
    TravelPlan,
    TravelPackage,
    TravelPackageList,
)
from .location_utils import (
    location_to_airport_code,
    flexible_city_code,
)

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def retry_async(retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """通用异步重试装饰器，可以按需套在外部你自己新增的工具上。"""
    def deco(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for i in range(1, retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if i == retries:
                        raise
                    wait = delay * (backoff ** (i - 1)) * (1 + random.random())
                    print(f"Retry {func.__name__} in {wait:.1f}s: {e}")
                    await asyncio.sleep(wait)
        return wrapper
    return deco

def _hotel_error_placeholder(source: str, message: str) -> List[HotelOption]:
    return [
        HotelOption(
            name="HOTEL_API_ERROR",
            category="N/A",
            price_per_night="N/A",
            source=source,
            rating=None,
            is_error=True,
            error_message=message,
        )
    ]


def _safe_price_to_float(price: str) -> float | None:
    if not price:
        return None
    first = price.split()[0]
    try:
        return float(first)
    except ValueError:
        return None


def _get_representative_options(
    options: List,
    key_attr: str,
    max_items: int = 7,
) -> List:
    """
    从大量 options 中抽取“代表性样本”给 LLM，控制 prompt 长度。
    """
    if not options or len(options) <= max_items:
        return options

    try:
        if key_attr == "price":
            options.sort(key=lambda x: float(getattr(x, key_attr).split(" ")[0]))
    except (ValueError, TypeError, IndexError):
        pass

    cheapest = options[:2]
    most_expensive = options[-2:]
    mid_index = len(options) // 2
    mid_range = options[mid_index - 1 : mid_index + 2]

    representative_sample = cheapest + mid_range + most_expensive
    seen = set()
    unique_sample = []
    for item in representative_sample:
        val = getattr(item, key_attr)
        if val not in seen:
            unique_sample.append(item)
            seen.add(val)
    return unique_sample


def _extract_json_object(raw: str) -> str:
    """
    从 LLM 输出中提取单个 JSON 对象字符串：
    - 去除 ```json``` 包裹
    - 若前后有解释文字，用正则只抠 “{ ... }” 部分
    """
    import re

    if not raw:
        raise ValueError("LLM returned empty content when JSON object was expected")

    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("Could not find JSON object in LLM output")

    return match.group(0)


# ---------------------------------------------------------------------------
# Location / coordinates
# ---------------------------------------------------------------------------

async def location_to_coordinates(location_name: str) -> tuple[float, float]:
    """
    用 LLM 粗略把城市/机场名转成城市中心坐标，用于 activities 搜索。
    """
    conversion_prompt = f"""
Provide the city center coordinates for this location.

Examples:
- "Seoul" → 37.566, 126.978
- "ICN" → 37.566, 126.978 (Seoul city center)
- "Tokyo" → 35.676, 139.650
- "Paris" → 48.8566, 2.3522

Location: "{location_name}"
Coordinates:
"""
    try:
        response = await llm.ainvoke(conversion_prompt)
        coords_text = response.content.strip()

        import re

        coords = re.findall(r"-?\d+\.?\d*", coords_text)
        if len(coords) >= 2:
            return float(coords[0]), float(coords[1])
        return 0.0, 0.0

    except Exception as e:
        print(f"✗ Coordinate conversion failed for {location_name}: {e}")
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Travel analysis: 自然语言 → TravelPlan
# ---------------------------------------------------------------------------

async def enhanced_travel_analysis(user_request: str) -> TravelPlan:
    """
    把用户自然语言需求解析成结构化 TravelPlan。
    """
    analysis_prompt = f"""
You are a world-class travel analyst AI. Extract structured trip information
from the user's request and output valid JSON matching the provided schema.

**User Request:** "{user_request}"

**Today's Date:** {datetime.now().strftime('%Y-%m-%d')}

**Instructions:**

1. Determine user_intent:
   - "full_plan": Combination of flights, hotels, or activities
   - "flights_only": Only asking for flights
   - "hotels_only": Only asking for hotels
   - "activities_only": Only asking for activities

2. Extract core details:
   - origin: Starting location (can be null)
   - destination: Final destination (mandatory)
   - departure_date & return_date: Calculate absolute dates in YYYY-MM-DD format
   - duration_days: Calculate days between departure and return
   - adults: Number of travelers (default 1)

3. Extract preferences:
   - travel_class
   - departure_time_pref & arrival_time_pref
   - total_budget as float

CRITICAL: Output MUST be valid JSON matching this schema:
{TravelPlan.model_json_schema()}

JSON Output:
"""
    try:
        response = await llm.ainvoke(analysis_prompt)

        raw_content = getattr(response, "content", "")
        if not isinstance(raw_content, str):
            raw_content = str(raw_content)

        # ✅ 允许模型输出前后夹带解释文字 / code fences：只抽取 {...}
        json_str = _extract_json_object(raw_content)

        extracted_plan = TravelPlan.model_validate_json(json_str)
        print(f"✓ Travel plan extracted: intent={extracted_plan.user_intent}")
        return extracted_plan

    except Exception as e:
        print(f"✗ Travel analysis failed: {e}")
        raise ValueError(f"Could not understand the travel request: {e}") from e

# ---------------------------------------------------------------------------
# Travel plan update based on user feedback
# ---------------------------------------------------------------------------
# --- update_travel_plan.py (replace your function with this) ---
import json
import re
from typing import Any, Dict, Optional
from pydantic import ValidationError

def _safe_load_json_obj(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def _is_refresh_recommendation(text: str) -> bool:
    t = (text or "").strip().lower()
    # 中英常见“换一个/再给我几个/换个推荐”
    if re.search(r"(换(一)?个|再(来|给)个|换个|再推荐|更多推荐|别的推荐|another|different)\s*(推荐|recommend)", t):
        return True
    # 极短但明确的“换一个推荐/换一个/再来一个”
    if t in {"换一个推荐", "换个推荐", "换一个", "再来一个", "再推荐", "another one", "another", "different one"}:
        return True
    return False

import re
import json
from typing import Optional
from pydantic import ValidationError

def _infer_intent_from_text(user_update: str) -> Optional[str]:
    """
    规则推断 intent：
    - 明确 only 指令优先
    - 否则出现航班/路线语义 -> 倾向 full_plan（除非明确 only hotels）
    """
    t = (user_update or "").strip().lower()

    # only patterns
    hotel_only = re.search(r"(只|仅|只想|只要).{0,6}(酒店|住宿|宾馆|hotel)", t)
    flight_only = re.search(r"(只|仅|只想|只要).{0,6}(航班|机票|flight|flights)", t)
    act_only = re.search(r"(只|仅|只想|只要).{0,6}(活动|行程|景点|tour|things to do|activity)", t)

    if hotel_only:
        return "hotels_only"
    if flight_only:
        return "flights_only"
    if act_only:
        return "activities_only"

    # signals
    has_flight_signal = bool(re.search(r"(往返|单程|飞|航班|机票|商务舱|经济舱|头等舱|round\s*trip|one\s*way)", t))
    has_route_signal = bool(re.search(r"(从.+到.+)", t)) or ("到" in t and "飞" in t)
    has_stay_signal = bool(re.search(r"(住|入住|酒店|住宿|待\s*\d+\s*(晚|天))", t))
    has_date_signal = bool(re.search(r"20\d{2}[-/]\d{1,2}[-/]\d{1,2}", t))

    # 只要出现明显“航班/路线”语义，且没有 only 限制，通常用户期待 full_plan
    if (has_flight_signal or has_route_signal) and (has_stay_signal or has_date_signal):
        return "full_plan"

    # 如果只有航班语义但没提住宿/活动，也可保守给 flights_only
    if has_flight_signal or has_route_signal:
        return "flights_only"

    # 默认不推断（保持原 intent）
    return None


async def update_travel_plan(prev: TravelPlan, user_update: str) -> TravelPlan:
    """
    ✅ 稳健增量更新：
    - LLM 输出 patch
    - merge 到 prev
    - 额外：merge 后做 intent 纠偏（更符合产品直觉）
    - 可选：大变更时直接重建 plan
    """
    if _is_refresh_recommendation(user_update):
        return prev

    prompt = f"""
You are updating an existing travel plan based on a user's new message.

PREVIOUS PLAN (JSON):
{json.dumps(prev.model_dump(), ensure_ascii=False)}

USER UPDATE:
"{user_update}"

RULES:
- Keep any fields not mentioned by the user unchanged.
- Only modify fields the user explicitly changes.
- Output MUST be a JSON OBJECT that contains ONLY the fields that changed (a patch).
- If the user did not change anything, output {{}}.
- NEVER output null for any field. If unsure, OMIT the field.
- Do NOT wrap in markdown. Output JSON only.

For reference, the full schema is:
{TravelPlan.model_json_schema()}

JSON PATCH Output:
"""
    try:
        resp = await llm.ainvoke(prompt)
        raw = getattr(resp, "content", "") or ""
        json_str = _extract_json_object(raw)
        patch = _safe_load_json_obj(json_str) or {}
    except Exception:
        return prev

    allowed = set(prev.model_dump().keys())
    patch = {k: v for k, v in patch.items() if k in allowed}

    # -----------------------------
    # (可选) 大变更 => 视为新需求
    # -----------------------------
    BIG_FIELDS = {"origin", "destination", "departure_date", "return_date", "duration_days", "adults", "total_budget"}
    big_changed = len(set(patch.keys()) & BIG_FIELDS) >= 4 and ("destination" in patch or "origin" in patch)
    if big_changed:
        # 你也可以选择：return await enhanced_travel_analysis(user_update)
        # 这里先给“更符合逻辑”的方案：直接重建
        try:
            new_plan = await enhanced_travel_analysis(user_update)
            # 如果 new_plan 缺关键字段，再回退 prev（避免模型发散）
            if not new_plan.destination:
                return prev
            return new_plan
        except Exception:
            pass  # fallback to patch merge

    merged = prev.model_dump()
    for k, v in patch.items():
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        merged[k] = v

    for k in ("origin", "destination"):
        if not merged.get(k):
            merged[k] = getattr(prev, k)



    # -----------------------------
    # ✅ intent 纠偏（核心）
    # -----------------------------
    inferred = _infer_intent_from_text(user_update)
    if inferred:
        merged["user_intent"] = inferred
    print("→ patch keys:", sorted(patch.keys()))
    print("→ final intent:", merged.get("user_intent"), "inferred:", inferred)
    try:
        return TravelPlan.model_validate(merged)
    except ValidationError:
        return prev




# ---------------------------------------------------------------------------
# Flight search
# ---------------------------------------------------------------------------

class FlightSearchArgs(BaseModel):
    originLocationCode: str = Field(description="Departure city IATA code")
    destinationLocationCode: str = Field(description="Arrival city IATA code")
    departureDate: str = Field(description="Departure date (YYYY-MM-DD)")
    returnDate: Optional[str] = Field(default=None, description="Return date (YYYY-MM-DD)")
    adults: int = Field(default=1, description="Number of adult passengers")
    currencyCode: str = Field(default="USD", description="Preferred currency")



def _parse_and_prepare_offers(response_data: dict) -> List[Dict[str, Any]]:
    if "data" not in response_data or not response_data["data"]:
        return []

    prepared_offers: List[Dict[str, Any]] = []
    carriers = response_data.get("dictionaries", {}).get("carriers", {})

    for offer in response_data["data"]:
        try:
            price_float = float(offer["price"]["total"])

            itinerary = offer["itineraries"][0]
            first_segment = itinerary["segments"][0]
            last_segment = itinerary["segments"][-1]

            option_obj = FlightOption(
                airline=carriers.get(
                    first_segment["carrierCode"],
                    first_segment["carrierCode"],
                ),
                price=f"{offer['price']['total']} {offer['price']['currency']}",
                departure_time=first_segment["departure"]["at"],
                arrival_time=last_segment["arrival"]["at"],
                duration=itinerary.get("duration"),
            )

            prepared_offers.append(
                {"price_numeric": price_float, "option_object": option_obj},
            )
        except (ValueError, KeyError, IndexError, TypeError) as e:
            print(f"⚠ Skipping malformed flight offer: {e}")
            continue

    return prepared_offers


def _find_closest_flight(offers: List[Dict[str, Any]], target_time_str: str) -> List[Dict[str, Any]]:
    try:
        target_hour = int(target_time_str.split(":")[0])
    except (ValueError, IndexError):
        print(f"⚠ Invalid target time: {target_time_str}")
        return offers

    def get_time_difference(prepared_offer: Dict[str, Any]) -> float:
        try:
            departure_dt = datetime.fromisoformat(
                prepared_offer["option_object"].departure_time,
            )
            return abs(departure_dt.hour - target_hour)
        except (ValueError, TypeError):
            return float("inf")

    return sorted(offers, key=get_time_difference)




@tool(args_schema=FlightSearchArgs)
async def search_flights(
    originLocationCode: str,
    destinationLocationCode: str,
    departureDate: str,
    returnDate: Optional[str] = None,
    adults: int = 1,
    travelClass: Optional[str] = None,
    departureTime: Optional[str] = None,
    arrivalTime: Optional[str] = None,
    currencyCode: str = "USD",
) -> List[FlightOption]:
    """
    航班查询工具，使用 Amadeus。

    设计要点：
    - 正常情况下返回真实的航班列表（用于套餐生成 / 展示）。
    - 如果是“条件下确实没有票” → 返回 []，表示业务正常但无结果。
    - 如果 Amadeus 报错 / 网络异常 / 内部异常：
        - 尝试最多 3 次（1s -> 2s -> 4s 指数退避）。
        - 如果最终还是失败，则返回一个带 is_error=True 的 FlightOption，
          供综合节点判断是“接口挂了”，而不是“查不到票”。
    """
    print(f"→ Flight search: {originLocationCode} → {destinationLocationCode}")

    # ------------------------------------------------------------------
    # 1. 城市/机场名 → 三字码
    # ------------------------------------------------------------------
    try:
        origin_task = location_to_airport_code(amadeus, originLocationCode)
        destination_task = location_to_airport_code(amadeus, destinationLocationCode)
        actual_origin, actual_destination = await asyncio.gather(
            origin_task,
            destination_task,
        )
        print(f"→ Converted to: {actual_origin} → {actual_destination}")
    except Exception as e:
        print(f"✗ Location conversion failed: {e}")
        return [
            FlightOption(
                airline="LOCATION_ERROR",
                price="N/A",
                departure_time="N/A",
                arrival_time="N/A",
                is_error=True,
                error_message=f"Failed to convert locations to airport codes: {e}",
            ),
        ]

    # ------------------------------------------------------------------
    # 2. Amadeus 客户端检查
    # ------------------------------------------------------------------
    if not amadeus:
        print("✗ Amadeus client not available.")
        return [
            FlightOption(
                airline="API_NOT_AVAILABLE",
                price="N/A",
                departure_time="N/A",
                arrival_time="N/A",
                is_error=True,
                error_message="Amadeus client not available in current environment.",
            ),
        ]

    # ------------------------------------------------------------------
    # 3. 构建 Amadeus 查询参数
    # ------------------------------------------------------------------
    search_params: Dict[str, Any] = {
        "originLocationCode": actual_origin,
        "destinationLocationCode": actual_destination,
        "departureDate": departureDate,
        "adults": adults,
        "currencyCode": currencyCode,
        "max": 25,
    }

    if returnDate:
        search_params["returnDate"] = returnDate

    if travelClass and travelClass.upper() in [
        "ECONOMY",
        "PREMIUM_ECONOMY",
        "BUSINESS",
        "FIRST",
    ]:
        search_params["travelClass"] = travelClass.upper()

    time_windows = {
        "morning": "06:00-12:00",
        "afternoon": "12:00-18:00",
        "evening": "18:00-23:59",
    }
    if departureTime and departureTime.lower() in time_windows:
        search_params["departureWindow"] = time_windows[departureTime.lower()]
    if arrivalTime and arrivalTime.lower() in time_windows:
        search_params["arrivalWindow"] = time_windows[arrivalTime.lower()]

    print(f"→ Calling Amadeus with params: {search_params}")

    # ------------------------------------------------------------------
    # 4. 指数退避重试：最多 3 次，1s -> 2s -> 4s
    # ------------------------------------------------------------------
    max_attempts = 3
    delay = 1.0
    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"→ Amadeus attempt {attempt}/{max_attempts}")
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: amadeus.shopping.flight_offers_search.get(**search_params),
            )

            # 这里表示 API 正常工作，只是这一组条件下没有航班
            if not response.data:
                print("→ Amadeus returned no data (no matching flights).")
                return []

            all_offers = _parse_and_prepare_offers(response.result)
            if not all_offers:
                print("→ Amadeus parsed 0 offers from response.")
                return []

            final_sorted_offers = sorted(
                all_offers,
                key=lambda x: x["price_numeric"],
            )

            # 如果用户给了具体时间（如“15:30”），再做一次按时间接近度排序
            if departureTime and ":" in departureTime:
                print(f"→ Re-sorting by proximity to {departureTime}")
                final_sorted_offers = _find_closest_flight(
                    final_sorted_offers,
                    departureTime,
                )

            top_3_offers = [item["option_object"] for item in final_sorted_offers[:3]]
            print(f"✓ Returning top 3 of {len(all_offers)} flight options")
            return top_3_offers

        except ResponseError as error:
            # Amadeus 返回 4xx/5xx 错误（包括你遇到的 141）
            last_error = error
            print(
                f"✗ Amadeus API error (attempt {attempt}/{max_attempts}): {error}",
            )
            try:
                status = getattr(error.response, "status_code", None)
                body = getattr(error.response, "body", None)
                print(f"  status: {status}")
                print(f"  body: {body}")
            except Exception:
                pass

            if attempt < max_attempts:
                print(f"→ Waiting {delay:.1f}s before retry...")
                await asyncio.sleep(delay)
                delay *= 2
            else:
                print("✗ Amadeus failed after max retries.")

        except Exception as e:
            # 代码 bug / 网络错误 等
            last_error = e
            print(
                f"✗ Flight search error (attempt {attempt}/{max_attempts}): {e}",
            )
            if attempt < max_attempts:
                print(f"→ Waiting {delay:.1f}s before retry...")
                await asyncio.sleep(delay)
                delay *= 2
            else:
                print("✗ Flight search failed after max retries.")

    # ------------------------------------------------------------------
    # 5. 所有重试都失败：返回 is_error=True 的占位，交给综合节点兜底
    # ------------------------------------------------------------------
    if last_error:
        return [
            FlightOption(
                airline="API_ERROR",
                price="N/A",
                departure_time="N/A",
                arrival_time="N/A",
                is_error=True,
                error_message=f"Flight API failed after retries: {last_error}",
            ),
        ]

    # 理论上不会走到这里，为了类型安全兜底一下
    return []




# ---------------------------------------------------------------------------
# Hotel search (Amadeus + Hotelbeds)
# ---------------------------------------------------------------------------

class HotelSearchArgs(BaseModel):
    city_code: str = Field(description="City IATA code (e.g., 'PAR', 'NYC')")
    check_in_date: str = Field(description="Check-in date (YYYY-MM-DD)")
    check_out_date: str = Field(description="Check-out date (YYYY-MM-DD)")
    adults: int = Field(description="Number of guests", default=1)


async def _clip_for_hotelbeds(check_in: str, check_out: str) -> tuple[str, str]:
    ci = datetime.strptime(check_in, "%Y-%m-%d")
    co = datetime.strptime(check_out, "%Y-%m-%d")
    nights = (co - ci).days
    if nights > 30:
        print(f"⚠ Hotelbeds stay too long: {nights} nights, clipping to 30.")
        co = ci + timedelta(days=30)
    return ci.strftime("%Y-%m-%d"), co.strftime("%Y-%m-%d")


async def _search_hotelbeds_hotels(
    city_code: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 1,
) -> List[HotelOption]:
    print(f"→ Hotelbeds: Searching {city_code} ({check_in_date} to {check_out_date})")

    headers = hotelbeds_headers()
    if not headers:
        print("⚠ Hotelbeds API keys not configured")
        return _hotel_error_placeholder(
            "Hotelbeds",
            "Hotelbeds API keys not configured in environment.",
        )


    api_url = "https://api.test.hotelbeds.com/hotel-api/1.0/hotels"
    check_in_date, check_out_date = await _clip_for_hotelbeds(
        check_in_date,
        check_out_date,
    )
    request_body = {
        "stay": {"checkIn": check_in_date, "checkOut": check_out_date},
        "occupancies": [
            {
                "rooms": 1,
                "adults": adults,
                "children": 0,
            },
        ],
        "destination": {"code": city_code},
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(api_url, headers=headers, json=request_body)
            response.raise_for_status()
            data = response.json()

        hotels: List[HotelOption] = []
        hotels_data = data.get("hotels", {})
        hotel_list = (
            hotels_data.get("hotels", [])
            if isinstance(hotels_data, dict)
            else hotels_data
        )

        for hotel in hotel_list[:5]:
            min_rate = hotel.get("minRate", "N/A")
            currency = hotel.get("currency", "USD")

            hotels.append(
                HotelOption(
                    name=hotel.get("name", "N/A"),
                    category=hotel.get("categoryName", "N/A"),
                    price_per_night=f"{min_rate} {currency}",
                    source="Hotelbeds",
                ),
            )

        print(f"✓ Hotelbeds: {len(hotels)} hotels found")
        return hotels

    except httpx.HTTPStatusError as e:
        print(f"✗ Hotelbeds API error: {e.response.status_code}")
        try:
            print("  Hotelbeds response body:", e.response.text)
        except Exception:
            pass
        return _hotel_error_placeholder(
            "Hotelbeds",
            f"Hotelbeds HTTP error {e.response.status_code}: {e.response.text if hasattr(e.response, 'text') else str(e)}",
        )
    except Exception as e:
        print(f"✗ Hotelbeds error: {e}")
        return _hotel_error_placeholder("Hotelbeds", f"Hotelbeds error: {e!r}")



async def _fallback_individual_hotel_search(
    hotel_ids: List[str],
    check_in_date: str,
    check_out_date: str,
    adults: int,
) -> List[HotelOption]:
    print("→ Using fallback individual hotel search")

    if not amadeus:
        print("⚠ Amadeus client not initialized")
        return _hotel_error_placeholder(
            "Amadeus",
            "Amadeus client not initialized (fallback individual search).",
        )


    offers: List[HotelOption] = []
    loop = asyncio.get_running_loop()

    for hotel_id in hotel_ids:
        try:
            offer_response = await loop.run_in_executor(
                None,
                lambda: amadeus.shopping.hotel_offers_search.get(
                    hotelIds=hotel_id,
                    checkInDate=check_in_date,
                    checkOutDate=check_out_date,
                    adults=adults,
                    roomQuantity=1,
                    currency="USD",
                ),
            )

            if not offer_response.data:
                continue

            for hotel_offer in offer_response.data:
                if not hotel_offer.get("available", True):
                    continue

                hotel_info = hotel_offer.get("hotel", {})
                offer_list = hotel_offer.get("offers", [])
                if not offer_list:
                    continue

                offer = offer_list[0]
                price_info = offer.get("price", {})

                offers.append(
                    HotelOption(
                        name=hotel_info.get("name", "N/A"),
                        category=f"{hotel_info.get('rating', 'N/A')}-star",
                        price_per_night=f"{price_info.get('total', 'N/A')} {price_info.get('currency', 'USD')}",
                        source="Amadeus",
                    ),
                )

        except Exception as e:
            print(f"✗ Individual search failed for {hotel_id}: {e}")
            continue

    return offers


async def _search_amadeus_hotels(
    city_code: str,
    check_in_date: str,
    check_out_date: str,
    adults: int,
) -> List[HotelOption]:
    print(f"→ Amadeus: Searching {city_code}")

    if not amadeus:
        print("⚠ Amadeus client not initialized")
        return _hotel_error_placeholder(
            "Amadeus",
            "Amadeus client not available in current environment.",
        )


    try:
        loop = asyncio.get_running_loop()
        list_response = await loop.run_in_executor(
            None,
            lambda: amadeus.reference_data.locations.hotels.by_city.get(
                cityCode=city_code,
                radius=5,
            ),
        )

        if not list_response.data:
            print(f"✗ Amadeus: No hotels found for {city_code}")
            return []

        hotel_ids = [hotel["hotelId"] for hotel in list_response.data[:5]]
        print(f"→ Amadeus: Found {len(hotel_ids)} hotel IDs")

        try:
            datetime.strptime(check_in_date, "%Y-%m-%d")
            datetime.strptime(check_out_date, "%Y-%m-%d")
        except ValueError as e:
            print(f"✗ Invalid date format: {e}")
            return _hotel_error_placeholder(
                "Input",
                f"Invalid date format: {e}",
            )


        try:
            offer_response = await loop.run_in_executor(
                None,
                lambda: amadeus.shopping.hotel_offers_search.get(
                    hotelIds=",".join(hotel_ids),
                    checkInDate=check_in_date,
                    checkOutDate=check_out_date,
                    adults=adults,
                    roomQuantity=1,
                    currency="USD",
                ),
            )
        except Exception as api_error:
            print(f"✗ Amadeus API error: {api_error}")
            return await _fallback_individual_hotel_search(
                hotel_ids[:3],
                check_in_date,
                check_out_date,
                adults,
            )

        offers: List[HotelOption] = []
        if offer_response.data:
            for hotel_offer in offer_response.data:
                if not hotel_offer.get("available", True):
                    continue

                hotel_info = hotel_offer.get("hotel", {})
                offer_list = hotel_offer.get("offers", [])

                if not offer_list:
                    continue

                offer = offer_list[0]
                price_info = offer.get("price", {})

                offers.append(
                    HotelOption(
                        name=hotel_info.get("name", "N/A"),
                        category=f"{hotel_info.get('rating', 'N/A')}-star",
                        price_per_night=f"{price_info.get('total', 'N/A')} {price_info.get('currency', 'USD')}",
                        source="Amadeus",
                    ),
                )

        print(f"✓ Amadeus: {len(offers)} hotels found")
        return offers

    except ResponseError as e:
        print(f"✗ Amadeus error: {e}")
        return _hotel_error_placeholder("Amadeus", f"Amadeus ResponseError: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return _hotel_error_placeholder("Amadeus", f"Amadeus hotel search error: {e!r}")



@tool(args_schema=HotelSearchArgs)
async def search_and_compare_hotels(
    city_code: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 1,
) -> List[HotelOption]:
    """
    酒店查询工具：自动将机场/城市名转为 city code，Amadeus + Hotelbeds 并发查询。
    """
    try:
        actual_city_code = await flexible_city_code(amadeus, city_code)
    except ValueError as e:
        print(f"✗ Entry validation: {e}")
        return _hotel_error_placeholder("Input", f"Invalid city_code: {e}")

    print(f"→ Hotel search: {city_code} → {actual_city_code}")

    amadeus_task = _search_amadeus_hotels(actual_city_code, check_in_date, check_out_date, adults)
    hotelbeds_task = _search_hotelbeds_hotels(actual_city_code, check_in_date, check_out_date, adults)

    results = await asyncio.gather(amadeus_task, hotelbeds_task, return_exceptions=True)

    combined_list: List[HotelOption] = []
    for r in results:
        if isinstance(r, Exception):
            combined_list.extend(_hotel_error_placeholder("HotelSearch", f"Unexpected error: {r!r}"))
        else:
            combined_list.extend(r)

    print(f"✓ Total hotels found: {len(combined_list)}")
    return combined_list


# ---------------------------------------------------------------------------
# Activity search
# ---------------------------------------------------------------------------

class ActivitySearchArgs(BaseModel):
    city_name: str = Field(
        description="Full city name for activity search (e.g., 'Paris', 'London')",
    )


@tool(args_schema=ActivitySearchArgs)
async def search_activities_by_city(city_name: str) -> List[ActivityOption]:
    """
    活动/景点查询工具，基于城市中心坐标。
    """
    print(f"→ Activity search: {city_name}")

    lat, lng = await location_to_coordinates(city_name)
    print(f"→ Coordinates: ({lat}, {lng})")

    if lat == 0.0 and lng == 0.0:
        return [
            ActivityOption(
                name="COORDINATE_ERROR",
                description="Could not determine city-center coordinates (required for activity search).",
                price="N/A",
                location=city_name,
                is_error=True,
                error_message=f"Could not determine coordinates for '{city_name}'",
            ),
        ]


    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: amadeus.shopping.activities.get(latitude=lat, longitude=lng, radius=15),
        )

        qualified_activities: List[ActivityOption] = []
        for act in response.data[:10]:
            price_info = act.get("price")
            description = act.get("shortDescription") or act.get("description")
            activity_name = act.get("name", "Unnamed Activity")

            if price_info or description:
                if price_info:
                    amount = price_info.get("amount", "N/A")
                    currency = price_info.get("currencyCode", "")
                    price_str = f"{amount} {currency}".strip()
                else:
                    price_str = "Price on request"

                if not description:
                    description = "Experience this popular local activity"

                qualified_activities.append(
                    ActivityOption(
                        name=activity_name,
                        description=description,
                        price=price_str,
                        location=city_name,
                    ),
                )

            if len(qualified_activities) >= 8:
                break

        if not qualified_activities:
            return []

        print(f"✓ Found {len(qualified_activities)} activities")
        return qualified_activities

    except Exception as e:
        print(f"✗ Activity search failed: {e!r}")
        return [
            ActivityOption(
                name="ERROR_PLACEHOLDER",
                description="Activity API error",
                price="0",
                location=None,
                is_error=True,
                error_message=str(e),
            ),
        ]


# ---------------------------------------------------------------------------
# Email notification
# ---------------------------------------------------------------------------

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class EmailArgs(BaseModel):
    to_email: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content (travel plan details)")


@tool(args_schema=EmailArgs)
def send_email_notification(to_email: str, subject: str, body: str) -> str:
    """
    Send an email notification via Gmail SMTP.

    Uses port 587 + STARTTLS because 465 is blocked in the current environment.
    """
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        print(f"→ Email (Mock): TO={to_email}, SUB={subject}")
        return "Email configuration missing. Sent mock email to console."

    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        # 关键修改在这里
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)  # 必须是 App Password
            server.send_message(msg)

        print(f"✓ Email sent to {to_email}")
        return "Email notification sent successfully."

    except Exception as e:
        print(f"✗ Email error: {e}")
        print(repr(e))
        return f"Failed to send email: {e}"



# ---------------------------------------------------------------------------
# CRM (HubSpot)
# ---------------------------------------------------------------------------

class HubSpotArgs(BaseModel):
    customer_info: Dict[str, str]
    travel_plan: TravelPlan
    recommendations: Dict[str, List[Any]]
    original_request: str


@tool(args_schema=HubSpotArgs)
async def send_to_hubspot(
    customer_info: Dict[str, str],
    travel_plan: TravelPlan,
    recommendations: Dict[str, List[Any]],
    original_request: str,
) -> str:
    """
    CRM 集成工具：默认对接 HubSpot，你也可以在这里替换成其他 CRM。
    """
    if not HUBSPOT_API_KEY:
        return "CRM integration is disabled."

    print("→ Preparing CRM data")

    description = f"""**Original Request:**\n{original_request}\n\n---
**AI-Generated Travel Plan:**
- **Origin:** {travel_plan.origin or 'N/A'}
- **Destination:** {travel_plan.destination}
- **Dates:** {travel_plan.departure_date} to {travel_plan.return_date}
- **Travelers:** {travel_plan.adults} adult(s)
- **Budget:** ${travel_plan.total_budget or 'Not specified'}
---
"""

    if "packages" in recommendations and recommendations["packages"]:
        description += "\n**AI-Generated Packages:**\n"
        packages = [TravelPackage.model_validate(p) for p in recommendations["packages"]]
        for i, pkg in enumerate(packages):
            description += (
                f"\n**{i+1}. {pkg.name} - ${pkg.total_cost:.2f}** ({pkg.budget_comment})\n"
                f"- **Flight:** {pkg.selected_flight.airline} ({pkg.selected_flight.price})\n"
                f"- **Hotel:** {pkg.selected_hotel.name} ({pkg.selected_hotel.price_per_night})\n"
                f"- **Activities:** {', '.join([a.name for a in pkg.selected_activities]) or 'None'}\n"
            )
    else:
        description += "\n**AI Search Results:**\n"
        if recommendations.get("flights"):
            description += f"- {len(recommendations['flights'])} flight option(s)\n"
        if recommendations.get("hotels"):
            description += f"- {len(recommendations['hotels'])} hotel option(s)\n"
        if recommendations.get("activities"):
            description += f"- {len(recommendations['activities'])} activity option(s)\n"

    hubspot_data = {
        "properties": {
            "dealname": f"AI Plan: {travel_plan.destination} for {customer_info.get('name', 'New Lead')}",
            "amount": str(travel_plan.total_budget or 0),
            "dealstage": "appointmentscheduled",
            "description": description,
            "customer_name": customer_info.get("name", ""),
            "customer_email": customer_info.get("email", ""),
            "customer_phone": customer_info.get("phone", ""),
            "original_travel_request": original_request,
            "travel_origin": travel_plan.origin or "Not specified",
            "travel_destination": travel_plan.destination,
            "departure_date": travel_plan.departure_date,
            "return_date": travel_plan.return_date,
            "number_of_travelers": travel_plan.adults,
            "flight_class_preference": travel_plan.travel_class,
            "ai_generated_content": json.dumps(recommendations),
        },
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.hubapi.com/crm/v3/objects/deals",
                headers={"Authorization": f"Bearer {HUBSPOT_API_KEY}"},
                json=hubspot_data,
            )
            response.raise_for_status()
            print("✓ Data sent to CRM successfully")
            return "Customer data sent to CRM successfully"
    except Exception as e:
        print(f"✗ CRM integration failed: {e}")
        return f"Failed to send to CRM: {e}"


# ---------------------------------------------------------------------------
# Package generation (LLM + rule-based fallback)
# ---------------------------------------------------------------------------

def _generate_rule_based_packages(
    trip_plan: TravelPlan,
    flights: List[FlightOption],
    hotels: List[HotelOption],
    activities: List[ActivityOption],
) -> List[TravelPackage]:
    """
    纯规则兜底版套餐生成：保证在 LLM 出问题时依然有结果。
    """
    if not flights or not hotels:
        print("⚠ Fallback: not enough flights or hotels")
        return []

    nights = trip_plan.duration_days or 1
    budget = trip_plan.total_budget or 0.0

    flights_sorted = sorted(
        flights,
        key=lambda f: _safe_price_to_float(f.price) or 999_999.0,
    )
    hotels_sorted = sorted(
        hotels,
        key=lambda h: _safe_price_to_float(h.price_per_night) or 999_999.0,
    )

    def _pick_activities(max_count: int) -> List[ActivityOption]:
        valid = [
            a for a in activities if _safe_price_to_float(a.price) is not None
        ]
        return valid[:max_count]

    def _build_package(
        name: str,
        grade: Literal["Budget", "Balanced", "Premium"],
        flight: FlightOption,
        hotel: HotelOption,
        acts: List[ActivityOption],
    ) -> TravelPackage:
        flight_cost = _safe_price_to_float(flight.price) or 0.0
        hotel_cost = (_safe_price_to_float(hotel.price_per_night) or 0.0) * nights
        act_cost = sum(_safe_price_to_float(a.price) or 0.0 for a in acts)
        total = flight_cost + hotel_cost + act_cost

        if budget > 0:
            diff = total - budget
            if diff <= 0:
                comment = f"总价在预算内，约节省 {abs(diff):.0f} USD。"
            else:
                comment = f"总价超出预算约 {diff:.0f} USD，可调整航班或酒店以降低价格。"
        else:
            comment = "用户未提供预算，按价格从低到高组合。"

        return TravelPackage(
            name=name,
            grade=grade,
            total_cost=total,
            budget_comment=comment,
            selected_flight=flight,
            selected_hotel=hotel,
            selected_activities=acts,
        )

    packages: List[TravelPackage] = []

    budget_pkg = _build_package(
        name="Budget 基础版",
        grade="Budget",
        flight=flights_sorted[0],
        hotel=hotels_sorted[0],
        acts=_pick_activities(1),
    )
    packages.append(budget_pkg)

    if len(flights_sorted) >= 2 and len(hotels_sorted) >= 2:
        mid_flight = flights_sorted[len(flights_sorted) // 2]
        mid_hotel = hotels_sorted[len(hotels_sorted) // 2]
        balanced_pkg = _build_package(
            name="Balanced 均衡版",
            grade="Balanced",
            flight=mid_flight,
            hotel=mid_hotel,
            acts=_pick_activities(2),
        )
        packages.append(balanced_pkg)

    if len(flights_sorted) >= 3 and len(hotels_sorted) >= 3:
        premium_pkg = _build_package(
            name="Premium 进阶版",
            grade="Premium",
            flight=flights_sorted[-1],
            hotel=hotels_sorted[-1],
            acts=_pick_activities(2),
        )
        packages.append(premium_pkg)

    print(f"✓ Rule-based fallback generated {len(packages)} packages")
    return packages


async def generate_travel_packages(
    trip_plan: TravelPlan,
    all_options: Dict[str, List[Any]],
) -> List[TravelPackage]:
    """
    LLM + 规则兜底的套餐生成主函数。
    """
    if not trip_plan.total_budget or trip_plan.total_budget <= 0:
        print("⚠ Cannot generate packages without valid budget")
        return []

    sorted_flights: List[FlightOption] = sorted(
        [
            f
            for f in all_options.get("flights", [])
            if _safe_price_to_float(f.price) is not None
        ],
        key=lambda x: _safe_price_to_float(x.price),
    )
    sorted_hotels: List[HotelOption] = sorted(
        [
            h
            for h in all_options.get("hotels", [])
            if _safe_price_to_float(h.price_per_night) is not None
        ],
        key=lambda x: _safe_price_to_float(x.price_per_night),
    )
    sorted_activities: List[ActivityOption] = sorted(
        [
            a
            for a in all_options.get("activities", [])
            if _safe_price_to_float(a.price) is not None
        ],
        key=lambda x: _safe_price_to_float(x.price),
    )

    if not sorted_flights or not sorted_hotels:
        print("⚠ Insufficient options for package generation")
        return []

    rep_flights = _get_representative_options(sorted_flights, "price")
    rep_hotels = _get_representative_options(sorted_hotels, "name")
    rep_activities = _get_representative_options(
        sorted_activities,
        "name",
        max_items=10,
    )

    schema_json = json.dumps(
        TravelPackageList.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )

    generation_prompt = f"""
You are an expert travel consultant. Create up to 3 compelling travel packages
for a client based on their plan and available options.

CLIENT PLAN:
- Destination: {trip_plan.destination}
- Duration: {trip_plan.duration_days} nights
- Budget: ${trip_plan.total_budget}

AVAILABLE OPTIONS (you MUST only pick from these lists):
- Flights: {json.dumps([f.model_dump() for f in rep_flights])}
- Hotels: {json.dumps([h.model_dump() for h in rep_hotels])}
- Activities: {json.dumps([a.model_dump() for a in rep_activities])}

Your job:
1. First check if a basic trip is possible within the budget.
2. Then create 1~3 packages:
   - If even the cheapest combination is OVER budget: create ONE "Budget" package only.
   - If budget is reasonable: create THREE packages (Budget, Balanced, Premium).
3. Each package must contain:
   - EXACTLY ONE selected_flight
   - EXACTLY ONE selected_hotel
   - 0~2 selected_activities
4. For each package:
   - total_cost = flight.price + hotel.price_per_night * {trip_plan.duration_days} + sum(activity.price)
   - budget_comment: compare total_cost vs client budget and briefly comment.

OUTPUT REQUIREMENTS:
- You MUST output a single JSON object that matches the following JSON schema:

{schema_json}

- The top-level object must match the `TravelPackageList` schema.
- Do NOT include any explanation, markdown, or text outside of the JSON.
- Do NOT wrap the JSON in ```json fences.
"""

    try:
        ai_msg = await llm.ainvoke(generation_prompt)
        raw_content = getattr(ai_msg, "content", ai_msg)
        if not isinstance(raw_content, str):
            raw_content = str(raw_content)

        json_str = _extract_json_object(raw_content)
        package_list = TravelPackageList.model_validate_json(json_str)
        packages = package_list.packages or []

        print(f"✓ Generated {len(packages)} packages via JSON mode")
        return packages

    except Exception as e:
        print(f"✗ LLM JSON package generation failed, fallback to rule-based: {e}")

        fallback_packages = _generate_rule_based_packages(
            trip_plan=trip_plan,
            flights=sorted_flights,
            hotels=sorted_hotels,
            activities=sorted_activities,
        )

        if fallback_packages:
            print("✓ Using rule-based fallback packages")
            return fallback_packages

        print("⚠ Rule-based fallback also failed, return [] to caller")
        return []
