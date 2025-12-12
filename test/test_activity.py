# test/test_activities.py
import asyncio
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from backend.travel_agent.tools import search_activities_by_city
async def main():
    result = await search_activities_by_city.ainvoke({
        "city_name": "Osaka",
    })
    print("activities:", result)

if __name__ == "__main__":
    asyncio.run(main())
