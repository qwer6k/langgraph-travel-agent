# test/test_amadeus_flights.py
import os
from amadeus import Client, ResponseError

AMADEUS_API_KEY = "TURgfZn2pIWkak17TZG7pLqbGVF4qbRx"
AMADEUS_API_SECRET = "GaGDmuUUmpJHOaZa"

amadeus = Client(
    client_id=AMADEUS_API_KEY,
    client_secret=AMADEUS_API_SECRET,
)

def test_flights():
    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode="PVG",
            destinationLocationCode="KIX",
            departureDate="2025-12-17",
            returnDate="2025-12-20",
            adults=1,
            currencyCode="USD",
            max=5,
        )
        print("status:", response.status_code)
        print("data length:", len(response.data))
        print("sample:", response.data[0] if response.data else "NO DATA")
    except ResponseError as error:
        print("ResponseError:", error)
        if error.response:
            print("status:", error.response.status_code)
            print("body:", error.response.body)

def test_activities():
    try:
        response = amadeus.shopping.activities.get(
            latitude=34.6937,
            longitude=135.5022,
            radius=15,
        )
        print("status:", response.status_code)
        print("data length:", len(response.data))
        print("sample:", response.data[0] if response.data else "NO DATA")
    except ResponseError as error:
        print("ResponseError:", error)
        if error.response:
            print("status:", error.response.status_code)
            print("body:", error.response.body)

if __name__ == "__main__":
    print("=== FLIGHTS ===")
    test_flights()
    print("\n=== ACTIVITIES ===")
    test_activities()
