# test/test_flights.py
import asyncio
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from backend.travel_agent.tools import search_flights

async def main():
    result = await search_flights.ainvoke({
        "originLocationCode": "LON",  # 替换为 test 环境常用的组合
        "destinationLocationCode": "NYC",
        "departureDate": "2026-01-15",
        "returnDate": "2026-01-20",
        "adults": 1,
        "currencyCode": "USD",
        "travelClass": "ECONOMY",
    })
    print("flights:", result)

if __name__ == "__main__":
    asyncio.run(main())
