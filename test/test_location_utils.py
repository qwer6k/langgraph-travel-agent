# tests/test_location_utils.py

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from .location_utils import (
    location_to_airport_code,
    location_to_city_code,
    flexible_city_code,
)


# =============== 工具类：构造假 Amadeus client ==================

class DummyAmadeusNoCall:
    """
    用于测试“本地映射命中时不会调用 Amadeus”。
    一旦 location.get 被调用，就抛异常让测试失败。
    """
    def __init__(self):
        self.reference_data = SimpleNamespace(
            locations=SimpleNamespace(
                get=self._get  # 如果被调用，测试直接挂掉
            )
        )

    def _get(self, *args, **kwargs):
        raise AssertionError("Amadeus should not be called for this test")


class FakeAmadeus:
    """
    用于模拟 Amadeus API 的返回值，不出网。
    fixtures: dict[(keyword, subType) -> list[dict]]
    """
    def __init__(self, fixtures):
        self._fixtures = fixtures
        self.reference_data = SimpleNamespace(
            locations=SimpleNamespace(
                get=self._get
            )
        )

    def _get(self, keyword: str, subType: str):
        # 模拟 amadeus SDK 的 Response 对象，只要有 .data 即可
        data = self._fixtures.get((keyword, subType), [])
        return SimpleNamespace(data=data)


# =============== 1. 本地映射路径测试 ==================

@pytest.mark.asyncio
async def test_airport_code_local_shanghai_in_chinese():
    """
    输入中文『上海』，应直接命中本地映射为 PVG，且不会调用 Amadeus。
    """
    amadeus = DummyAmadeusNoCall()

    code = await location_to_airport_code(amadeus, "上海")

    assert code == "PVG"   # 来自 CITY_NAME_TO_MAIN_AIRPORT
    # 如果 DummyAmadeusNoCall._get 被调用，上面已经会抛 AssertionError 让测试失败


@pytest.mark.asyncio
async def test_airport_code_local_osaka_in_english():
    """
    输入英文『osaka』，应命中本地映射为 KIX。
    """
    amadeus = DummyAmadeusNoCall()

    code = await location_to_airport_code(amadeus, "osaka")

    assert code == "KIX"   # CITY_NAME_TO_MAIN_AIRPORT["osaka"] = "KIX"


@pytest.mark.asyncio
async def test_city_code_local_osaka_in_chinese():
    """
    输入中文『大阪』，应命中本地映射为 OSA。
    """
    amadeus = DummyAmadeusNoCall()

    code = await location_to_city_code(amadeus, "大阪")

    assert code == "OSA"   # CITY_NAME_TO_CITY_CODE["大阪"] = "OSA"


@pytest.mark.asyncio
async def test_hongkong_local_mapping_airport_and_city():
    """
    验证你刚刚在 city_maps 里加的『香港』映射：
    - 机场码 HKG
    - 城市码 HKG
    """
    amadeus = DummyAmadeusNoCall()

    airport_code = await location_to_airport_code(amadeus, "香港")
    city_code = await location_to_city_code(amadeus, "香港")

    assert airport_code == "HKG"
    assert city_code == "HKG"


# =============== 2. 外部 API 路径（使用 FakeAmadeus，不出网） ==================

@pytest.mark.asyncio
async def test_airport_code_via_amadeus_fallback():
    """
    输入一个本地映射表里没有的城市，例如 'Boston'，
    应该走 Amadeus（这里用 FakeAmadeus 模拟）。
    """
    fixtures = {
        ("Boston", "AIRPORT"): [
            {"subType": "AIRPORT", "iataCode": "BOS"}
        ]
    }
    amadeus = FakeAmadeus(fixtures)

    code = await location_to_airport_code(amadeus, "Boston")

    assert code == "BOS"


@pytest.mark.asyncio
async def test_city_code_via_amadeus_fallback():
    """
    同样用 'Boston'，但测试 CITY 路径。
    """
    fixtures = {
        ("Boston", "CITY"): [
            {"subType": "CITY", "iataCode": "BOS"}
        ]
    }
    amadeus = FakeAmadeus(fixtures)

    code = await location_to_city_code(amadeus, "Boston")

    assert code == "BOS"


@pytest.mark.asyncio
async def test_flexible_city_code_from_iata_airport():
    """
    flexible_city_code 传入机场三字码时，应该通过 AIRPORT_TO_CITY_CODE 映射成城市码。
    例如：KIX -> OSA, HND -> TYO 等。
    """
    amadeus = DummyAmadeusNoCall()   # 这里完全不需要调用 Amadeus

    code_kix = await flexible_city_code(amadeus, "KIX")
    code_hnd = await flexible_city_code(amadeus, "HND")

    assert code_kix == "OSA"   # AIRPORT_TO_CITY_CODE["KIX"] = "OSA"
    assert code_hnd == "TYO"   # AIRPORT_TO_CITY_CODE["HND"] = "TYO"
