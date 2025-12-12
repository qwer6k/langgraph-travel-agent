"""
location_utils.py

地点解析工具：只用本地映射表 + Amadeus 的 reference_data.locations，
不依赖 LLM。

提供两个异步函数：

- location_to_airport_code(amadeus_client, location_name) -> str
- location_to_city_code(amadeus_client, location_name) -> str
"""

import asyncio
from typing import Optional
from amadeus import Client, ResponseError
# ====================== 新增依赖 ======================
import asyncio
from typing import Optional
from pypinyin import lazy_pinyin, Style          # pyright: ignore[reportMissingImports] # pip install pypinyin
import re
from .city_maps import (
    CITY_NAME_TO_MAIN_AIRPORT,
    CITY_NAME_TO_CITY_CODE,
    AIRPORT_TO_CITY_CODE,
)



# -----------------------------------------------------------------------------
# 内部工具函数
# -----------------------------------------------------------------------------

def _is_iata_code(text: str) -> bool:
    """判断字符串是不是形如 'PEK' 这样的 3 位大写字母 IATA 码。"""
    if not text:
        return False
    text = text.strip()
    return len(text) == 3 and text.isalpha() and text.upper() == text


def _norm_key(text: str) -> str:
    """归一化城市名，用于查映射表：去空格 + 小写。中文不会受影响。"""
    return text.strip().lower()

def _to_pinyin(text: str) -> str:
    """中文→无音标小写拼音；非中文原样返回"""
    if re.search(r'[\u4e00-\u9fff]', text):
        return ''.join(lazy_pinyin(text, style=Style.NORMAL)).lower()
    return text.lower()

# -----------------------------------------------------------------------------
# 内部工具：统一 Amadeus 查询逻辑
# -----------------------------------------------------------------------------

async def _resolve_with_amadeus(
    amadeus_client: Optional[Client],
    keyword_candidates: list[str],
    subtype: str,
    raw_location: str,
) -> Optional[str]:
    """
    封装 Amadeus reference_data.locations.get 的通用查询逻辑。
    - keyword_candidates: 已经过滤过的候选 keyword（尽量是 ASCII，避免 400）
    - subtype: "AIRPORT" / "CITY"
    - raw_location: 仅用于日志
    返回：成功解析到的三字码，否则 None
    """
    if not amadeus_client:
        raise ValueError(f"Amadeus client not initialized, cannot resolve {subtype.lower()} for '{raw_location}'")

    loop = asyncio.get_running_loop()

    # 去重 & 去空
    seen_kw: set[str] = set()
    clean_candidates = []
    for kw in keyword_candidates:
        if not kw:
            continue
        kw = kw.strip()
        if not kw:
            continue
        if kw in seen_kw:
            continue
        seen_kw.add(kw)

        # 避免再把中文 keyword 丢给 Amadeus 导致 400
        if re.search(r'[\u4e00-\u9fff]', kw):
            continue

        clean_candidates.append(kw)

    for keyword in clean_candidates:
        for attempt in range(3):
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: amadeus_client.reference_data.locations.get(
                        keyword=keyword,
                        subType=subtype,
                    ),
                )
                data = getattr(response, "data", None) or []
                if not data:
                    # 当前 keyword 没有任何结果，换下一个 keyword
                    break

                # 优先挑 subType 精确匹配的结果
                chosen = None
                for item in data:
                    if item.get("subType") == subtype:
                        chosen = item
                        break
                if not chosen:
                    chosen = data[0]

                code = (chosen.get("iataCode") or "").upper().strip()
                if _is_iata_code(code):
                    print(f"→ {subtype.title()} code from Amadeus: '{raw_location}' / '{keyword}' → {code}")
                    return code
                else:
                    # 数据结构不符合预期，尝试下一个 keyword
                    print(f"⚠ Amadeus returned invalid {subtype} code '{code}' for '{raw_location}'")
                    break

            except ResponseError as e:
                # 400 之类的问题，通常没必要重复同一个 keyword
                print(f"✗ Amadeus {subtype.lower()} lookup error for '{raw_location}' (keyword='{keyword}'): {e}")
                break
            except Exception as e:
                print(
                    f"✗ Unexpected Amadeus {subtype.lower()} lookup error for "
                    f"'{raw_location}' (keyword='{keyword}', attempt={attempt+1}): {e}"
                )
                # 小退避重试
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                break

    return None


# -----------------------------------------------------------------------------
# 对外函数：location_to_airport_code
# -----------------------------------------------------------------------------

async def location_to_airport_code(
    amadeus_client: Optional[Client],
    location_name: str,
) -> str:
    """
    将用户输入的地点（中文 / 英文 / 城市名 / 机场名）转为主机场 IATA 码。

    优先级：
      1. 已经是 IATA 码：直接返回（例如 "PEK", "PVG"）
      2. 命中本地 CITY_NAME_TO_MAIN_AIRPORT 映射表（支持中文 + 拼音/英文）
      3. 使用 Amadeus reference_data.locations.get(subType="AIRPORT")
         - 会对 keyword 做规范化（避免中文直接丢给 Amadeus）
      4. 全部失败：抛出 ValueError（不再返回原始字符串）
    """
    if not location_name or not location_name.strip():
        raise ValueError("Airport name is empty")

    text = location_name.strip()
    # 1) 已经是 IATA 码
    if _is_iata_code(text):
        return text.upper()

    # 统一归一化信息
    norm_raw = _norm_key(text)          # 原文小写去空格，如 "Hong Kong" -> "hong kong" / "上海" -> "上海"
    pinyin   = _to_pinyin(text)         # 中文 -> 拼音，非中文保持小写，如 "上海" -> "shanghai"

    # 2) 先查本地映射表（原文 + 拼音 两种 key）
    local_keys = {norm_raw, pinyin}
    for key in local_keys:
        if key and key in CITY_NAME_TO_MAIN_AIRPORT:
            code = (CITY_NAME_TO_MAIN_AIRPORT[key] or "").upper().strip()
            if not _is_iata_code(code):
                raise ValueError(f"Local airport map returned invalid code '{code}' for '{location_name}'")
            print(f"→ Airport code from local map: '{location_name}' → {code}")
            return code

    # 3) 使用 Amadeus 查询
    # 对 Amadeus 来说，避免中文 keyword，优先使用英文 / 拼音 / 归一化英文
    keyword_candidates: list[str] = []
    # 对中文：_to_pinyin 已经是无音标小写拼音；对英文：就是 lower
    # 这里尽量多给几个变体，让 Amadeus 有机会命中
    keyword_candidates.append(text)      # 原文，例如 "Hong Kong"
    if norm_raw != text.lower():
        keyword_candidates.append(norm_raw)
    if pinyin and pinyin != norm_raw:
        keyword_candidates.append(pinyin)

    airport_code = await _resolve_with_amadeus(
        amadeus_client,
        keyword_candidates,
        subtype="AIRPORT",
        raw_location=location_name,
    )
    if airport_code:
        return airport_code

    # 4) 全部失败：明确抛错
    raise ValueError(f"Cannot resolve airport code for '{location_name}'")


# -----------------------------------------------------------------------------
# 对外函数：location_to_city_code
# -----------------------------------------------------------------------------

async def location_to_city_code(
    amadeus_client: Optional[Client],
    location_name: str,
) -> str:
    """
    将用户输入的地点转为“城市码”（用于酒店，例如 BJS, SHA, PAR）。

    优先级：
      1. 如果是 IATA 机场码，先查 AIRPORT_TO_CITY_CODE（PEK -> BJS, PVG -> SHA 等）
         - 查不到则认为该三字码本身就是城市码（如 HKG），直接返回
      2. 命中本地 CITY_NAME_TO_CITY_CODE 映射表（支持中文 + 拼音/英文）
      3. 使用 Amadeus reference_data.locations.get(subType="CITY")
      4. 全部失败：抛出 ValueError
    """
    if not location_name or not location_name.strip():
        raise ValueError("City name is empty")

    text = location_name.strip()

    # 1) 已经是 IATA 码：先尝试机场->城市映射
    if _is_iata_code(text):
        upper = text.upper()
        if upper in AIRPORT_TO_CITY_CODE:
            code = (AIRPORT_TO_CITY_CODE[upper] or "").upper().strip()
            if not _is_iata_code(code):
                raise ValueError(f"AIRPORT_TO_CITY_CODE returned invalid code '{code}' for '{location_name}'")
            print(f"→ City code from airport map: '{location_name}' → {code}")
            return code

        # 没有映射，就直接把这个三字码当作城市码用（比如 HKG）
        print(f"⚠ No explicit city mapping for airport '{location_name}', using '{upper}' as city code")
        return upper

    # 2) 本地城市名映射表（支持原文 + 拼音）
    norm_raw = _norm_key(text)
    pinyin   = _to_pinyin(text)
    local_keys = {norm_raw, pinyin}

    for key in local_keys:
        if key and key in CITY_NAME_TO_CITY_CODE:
            code = (CITY_NAME_TO_CITY_CODE[key] or "").upper().strip()
            if not _is_iata_code(code):
                raise ValueError(f"Local city map returned invalid code '{code}' for '{location_name}'")
            print(f"→ City code from local map: '{location_name}' → {code}")
            return code

    # 3) 使用 Amadeus 查询 CITY
    keyword_candidates: list[str] = []
    keyword_candidates.append(text)
    if norm_raw != text.lower():
        keyword_candidates.append(norm_raw)
    if pinyin and pinyin != norm_raw:
        keyword_candidates.append(pinyin)

    city_code = await _resolve_with_amadeus(
        amadeus_client,
        keyword_candidates,
        subtype="CITY",
        raw_location=location_name,
    )
    if city_code:
        return city_code

    # 4) 全部失败
    raise ValueError(f"Cannot resolve city code for '{location_name}'")


# -----------------------------------------------------------------------------
# 对外函数：flexible_city_code
# -----------------------------------------------------------------------------

async def flexible_city_code(
    amadeus_client: Optional[Client],
    location_name: str,
) -> str:
    """
    渐进式城市码解析（用于酒店搜索）：

    1. 如果用户已经给了 3 位 IATA 码（不论是机场还是城市码），
       统一走 location_to_city_code，处理 AIRPORT -> CITY 的转换。
    2. 否则交给 location_to_city_code 做：本地映射 + Amadeus 查询。
    3. 全部失败时抛出明确的 ValueError。
    """
    if not location_name or not location_name.strip():
        raise ValueError("City name is empty")

    text = location_name.strip()

    try:
        # 逻辑全部复用 location_to_city_code，保证行为一致
        return await location_to_city_code(amadeus_client, text)
    except ValueError as e:
        # 保持原先 flexible_city_code 的错误信息感觉
        raise ValueError(f"Cannot resolve city '{location_name}' – please use 3-letter code") from e
