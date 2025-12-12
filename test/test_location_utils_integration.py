# tests/test_location_utils_integration.py

import os
import pytest
from amadeus import Client

from backend.location_utils import (
    location_to_airport_code,
    location_to_city_code,
)

AMADEUS_API_KEY = "TURgfZn2pIWkak17TZG7pLqbGVF4qbRx"
AMADEUS_API_SECRET = "GaGDmuUUmpJHOaZa"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not (AMADEUS_API_KEY and AMADEUS_API_SECRET),
    reason="Amadeus credentials not configured",
)
async def test_airport_code_real_amadeus_boston():
    """
    真正打 Amadeus API（如果配置了 key）：
    用一个不在本地映射表里的城市，例如 'Boston'。
    """
    client = Client(
        client_id=AMADEUS_API_KEY,
        client_secret=AMADEUS_API_SECRET,
    )

    code = await location_to_airport_code(client, "Boston")

    # 实际上 BOS 是波士顿的主机场
    assert code == "BOS"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not (AMADEUS_API_KEY and AMADEUS_API_SECRET),
    reason="Amadeus credentials not configured",
)
async def test_city_code_real_amadeus_boston():
    client = Client(
        client_id=AMADEUS_API_KEY,
        client_secret=AMADEUS_API_SECRET,
    )

    code = await location_to_city_code(client, "Boston")

    # Amadeus 城市码一般也会给 BOS，这里简单地断言是合法三字码即可
    assert isinstance(code, str)
    assert len(code) == 3
    assert code.isalpha()
    assert code.upper() == code
