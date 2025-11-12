"""
Multi-Agent Travel Booking System
Production-ready LangGraph implementation with async tool orchestration.

Features:
- Multi-API integration (Amadeus, Hotelbeds)
- Async parallel tool execution
- Human-in-the-loop workflows
- CRM integration (HubSpot by default, easily replaceable)
- SMS notifications via Twilio
"""

import os
import hashlib
import time
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Optional, Dict, Any, Literal
import operator
from pydantic import BaseModel, Field 
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
import json
from datetime import datetime, timedelta
from amadeus import Client, ResponseError
import asyncio
import httpx 
from twilio.rest import Client as TwilioClient

# ==============================================================================
# ENVIRONMENT & CLIENT INITIALIZATION
# ==============================================================================

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# Core API Keys (Required)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET")

# Optional API Keys
HOTELBEDS_API_KEY = os.getenv("HOTELBEDS_API_KEY")
HOTELBEDS_API_SECRET = os.getenv("HOTELBEDS_API_SECRET")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_SENDER_PHONE = os.getenv("TWILIO_SENDER_PHONE")
HUBSPOT_API_KEY = os.getenv("HUBSPOT_API_KEY")  # Change to your CRM's API key

if not all([GOOGLE_API_KEY, AMADEUS_API_KEY, AMADEUS_API_SECRET]):
    raise ValueError("Required API keys missing: GOOGLE_API_KEY, AMADEUS_API_KEY, AMADEUS_API_SECRET")

# Initialize clients
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0, 
    google_api_key=GOOGLE_API_KEY
)

amadeus = None
twilio_client = None

try:
    if AMADEUS_API_KEY and AMADEUS_API_SECRET:
        amadeus = Client(
            client_id=AMADEUS_API_KEY, 
            client_secret=AMADEUS_API_SECRET
        )
        print("✓ Amadeus client initialized")
    
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
        twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        print("✓ Twilio client initialized")
except Exception as e:
    print(f"⚠ Client initialization warning: {e}")

# ==============================================================================
# PYDANTIC MODELS - Data Structures
# ==============================================================================

class FlightOption(BaseModel):
    """Structured flight offer data"""
    airline: str = Field(description="Airline name")
    price: str = Field(description="Total flight cost")
    departure_time: str = Field(description="Departure time (YYYY-MM-DDTHH:MM:SS)")
    arrival_time: str = Field(description="Arrival time (YYYY-MM-DDTHH:MM:SS)")
    duration: Optional[str] = Field(description="Flight duration", default=None)

class HotelOption(BaseModel):
    """Structured hotel offer data"""
    name: str = Field(description="Hotel name")
    category: str = Field(description="Star rating, e.g., '5EST' for 5-star")
    price_per_night: str = Field(description="Price per night")
    source: str = Field(description="Data source (e.g., 'Amadeus', 'Hotelbeds')")
    rating: Optional[float] = Field(description="Hotel rating", default=None)

class ActivityOption(BaseModel):
    """Structured activity offer data"""
    name: str = Field(description="Activity name")
    description: str = Field(description="Brief description")
    price: str = Field(description="Activity price")
    location: Optional[str] = Field(description="Activity location", default=None)

class TravelPackage(BaseModel):
    """Complete travel package combining flight, hotel, and activities"""
    name: str = Field(description="Package name, e.g., 'Smart Explorer'")
    grade: Literal["Budget", "Balanced", "Premium"] = Field(description="Package tier")
    total_cost: float = Field(description="Total package cost in USD")
    budget_comment: str = Field(description="Budget comparison comment")
    selected_flight: FlightOption = Field(description="Selected flight option")
    selected_hotel: HotelOption = Field(description="Selected hotel option")
    selected_activities: List[ActivityOption] = Field(
        default_factory=list,
        max_items=2,
        description="0-2 activity options"
    )

class TravelPlan(BaseModel):
    """Structured travel plan extracted from user request"""
    origin: Optional[str] = Field(None, description="Origin city or airport code")
    destination: str = Field(..., description="Destination city or airport code")
    departure_date: Optional[str] = Field(None, description="Departure date (YYYY-MM-DD)")
    return_date: Optional[str] = Field(None, description="Return date (YYYY-MM-DD)")
    duration_days: Optional[int] = Field(None, description="Trip duration in days")
    adults: int = Field(1, description="Number of adult travelers")    
    travel_class: Optional[Literal["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"]] = "ECONOMY"
    departure_time_pref: Optional[str] = Field(None, description="Preferred departure time")
    arrival_time_pref: Optional[str] = Field(None, description="Preferred arrival time")    
    total_budget: Optional[float] = Field(None, description="Total budget in USD")
    user_intent: Literal["full_plan", "flights_only", "hotels_only", "activities_only"] = Field(
        "full_plan", 
        description="User's primary goal"
    )

# ==============================================================================
# STATE MANAGEMENT
# ==============================================================================

class TravelAgentState(TypedDict):
    """Graph state with complete conversation context"""
    messages: Annotated[List[AnyMessage], operator.add]
    travel_plan: Optional[TravelPlan]
    user_preferences: Optional[Dict[str, Any]] 
    form_to_display: Optional[str]
    current_step: str  # "initial", "tools_called", "synthesizing", "complete"
    errors: List[str]
    customer_info: Optional[Dict[str, str]]
    trip_details: Optional[Dict[str, Any]]
    original_request: Optional[str]
    is_continuation: Optional[bool]

# ==============================================================================
# HOTEL SEARCH TOOLS
# ==============================================================================

def _get_hotelbeds_headers():
    """
    Generates X-Signature authentication for Hotelbeds API.
    
    Returns:
        dict: Authentication headers with API key and signature
    """
    api_key = HOTELBEDS_API_KEY
    secret = HOTELBEDS_API_SECRET
    if not api_key or not secret:
        return None
    
    utc_timestamp = int(time.time())
    signature = hashlib.sha256(
        f"{api_key}{secret}{utc_timestamp}".encode()
    ).hexdigest()
    
    return {
        "Api-key": api_key,
        "X-Signature": signature,
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
    }

async def _search_hotelbeds_hotels(
    city_code: str, 
    check_in_date: str, 
    check_out_date: str, 
    adults: int = 1
) -> List[HotelOption]:
    """
    Search real-time hotel availability on Hotelbeds.
    
    Args:
        city_code: Destination city code
        check_in_date: Check-in date (YYYY-MM-DD)
        check_out_date: Check-out date (YYYY-MM-DD)
        adults: Number of guests
        
    Returns:
        List of available hotel options
    """
    print(f"→ Hotelbeds: Searching {city_code} ({check_in_date} to {check_out_date})")
    
    headers = _get_hotelbeds_headers()
    if not headers: 
        print("⚠ Hotelbeds API keys not configured")
        return []

    api_url = "https://api.test.hotelbeds.com/hotel-api/1.0/hotels"
    
    request_body = {
        "stay": {
            "checkIn": check_in_date,
            "checkOut": check_out_date
        },
        "occupancies": [{
            "rooms": 1,
            "adults": adults,
            "children": 0
        }],
        "destination": {
            "code": city_code
        }
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(api_url, headers=headers, json=request_body)
            response.raise_for_status()
            data = response.json()
            
            hotels = []
            hotels_data = data.get('hotels', {})
            hotel_list = hotels_data.get('hotels', []) if isinstance(hotels_data, dict) else hotels_data
            
            for hotel in hotel_list[:5]:
                min_rate = hotel.get('minRate', 'N/A')
                currency = hotel.get('currency', 'USD')
                
                hotels.append(HotelOption(
                    name=hotel.get('name', 'N/A'),
                    category=hotel.get('categoryName', 'N/A'),
                    price_per_night=f"{min_rate} {currency}",
                    source="Hotelbeds"
                ))
            
            print(f"✓ Hotelbeds: {len(hotels)} hotels found")
            return hotels
            
    except httpx.HTTPStatusError as e:
        print(f"✗ Hotelbeds API error: {e.response.status_code}")
        return []
    except Exception as e:
        print(f"✗ Hotelbeds error: {e}")
        return []

async def _search_amadeus_hotels(
    city_code: str, 
    check_in_date: str, 
    check_out_date: str, 
    adults: int
) -> List[HotelOption]:
    """
    Search hotels via Amadeus API with fallback logic.
    
    Args:
        city_code: Destination city code
        check_in_date: Check-in date (YYYY-MM-DD)
        check_out_date: Check-out date (YYYY-MM-DD)
        adults: Number of guests
        
    Returns:
        List of available hotel options
    """
    print(f"→ Amadeus: Searching {city_code}")
    
    if not amadeus: 
        print("⚠ Amadeus client not initialized")
        return []
    
    try:
        loop = asyncio.get_running_loop()
        
        # Step 1: Get hotel IDs in the city
        list_response = await loop.run_in_executor(
            None, 
            lambda: amadeus.reference_data.locations.hotels.by_city.get(
                cityCode=city_code, 
                radius=5
            )
        )
        
        if not list_response.data:
            print(f"✗ Amadeus: No hotels found for {city_code}")
            return []
        
        hotel_ids = [hotel['hotelId'] for hotel in list_response.data[:5]]
        print(f"→ Amadeus: Found {len(hotel_ids)} hotel IDs")

        # Step 2: Validate date format
        try:
            datetime.strptime(check_in_date, '%Y-%m-%d')
            datetime.strptime(check_out_date, '%Y-%m-%d')
        except ValueError as e:
            print(f"✗ Invalid date format: {e}")
            return []
        
        # Step 3: Get offers for hotels
        try:
            offer_response = await loop.run_in_executor(
                None, 
                lambda: amadeus.shopping.hotel_offers_search.get(
                    hotelIds=','.join(hotel_ids),
                    checkInDate=check_in_date,
                    checkOutDate=check_out_date,
                    adults=adults,
                    roomQuantity=1,
                    currency='USD'
                )
            )
        except Exception as api_error:
            print(f"✗ Amadeus API error: {api_error}")
            return await _fallback_individual_hotel_search(
                hotel_ids[:3], 
                check_in_date, 
                check_out_date, 
                adults
            )
        
        offers = []
        if offer_response.data:
            for hotel_offer in offer_response.data:
                if not hotel_offer.get('available', True):
                    continue

                hotel_info = hotel_offer.get('hotel', {})
                offer_list = hotel_offer.get('offers', [])
                
                if not offer_list:
                    continue
                    
                offer = offer_list[0]
                price_info = offer.get('price', {})
                
                offers.append(HotelOption(
                    name=hotel_info.get('name', 'N/A'),
                    category=f"{hotel_info.get('rating', 'N/A')}-star",
                    price_per_night=f"{price_info.get('total', 'N/A')} {price_info.get('currency', 'USD')}",
                    source="Amadeus"
                ))
        
        print(f"✓ Amadeus: {len(offers)} hotels found")
        return offers
        
    except ResponseError as e:
        print(f"✗ Amadeus error: {e}")
        return []
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return []

async def _fallback_individual_hotel_search(
    hotel_ids: List[str], 
    check_in_date: str, 
    check_out_date: str, 
    adults: int
) -> List[HotelOption]:
    """
    Fallback: Search hotels individually when batch search fails.
    
    Args:
        hotel_ids: List of hotel IDs to search
        check_in_date: Check-in date (YYYY-MM-DD)
        check_out_date: Check-out date (YYYY-MM-DD)
        adults: Number of guests
        
    Returns:
        List of available hotel options
    """
    print("→ Using fallback individual hotel search")
    
    offers = []
    loop = asyncio.get_running_loop()
    
    for hotel_id in hotel_ids:
        try:
            offer_response = await loop.run_in_executor(
                None,
                lambda: amadeus.shopping.hotel_offers_by_hotel.get(
                    hotelId=hotel_id,
                    checkInDate=check_in_date,
                    checkOutDate=check_out_date,
                    adults=adults
                )
            )
            
            if offer_response.data and offer_response.data.get('offers'):
                hotel_info = offer_response.data.get('hotel', {})
                offer = offer_response.data['offers'][0]
                price_info = offer.get('price', {})
                
                offers.append(HotelOption(
                    name=hotel_info.get('name', 'N/A'),
                    category=f"{hotel_info.get('rating', 'N/A')}-star",
                    price_per_night=f"{price_info.get('total', 'N/A')} {price_info.get('currency', 'USD')}",
                    source="Amadeus"
                ))
                
        except Exception as e:
            print(f"✗ Individual search failed for {hotel_id}: {e}")
            continue
    
    return offers

# ==============================================================================
# LOCATION CONVERSION HELPERS
# ==============================================================================

async def location_to_airport_code(location_name: str) -> str:
    """
    Convert location name to IATA airport code using LLM.
    
    Args:
        location_name: City name or existing airport code
        
    Returns:
        3-letter IATA airport code
    """
    if not location_name:
        return ""
        
    # Already an airport code
    if len(location_name) == 3 and location_name.isalpha() and location_name.isupper():
        return location_name
    
    conversion_prompt = f"""
    Convert this location to the main international airport IATA code.
    
    Examples:
    - "Seoul" → "ICN"
    - "Tokyo" → "NRT" 
    - "Paris" → "CDG"
    - "New York" → "JFK"
    - "London" → "LHR"
    
    Location: "{location_name}"
    IATA Code:
    """
    
    try:
        response = await llm.ainvoke(conversion_prompt)
        airport_code = response.content.strip().upper()
        
        if len(airport_code) == 3 and airport_code.isalpha():
            return airport_code
        else:
            import re
            codes = re.findall(r'[A-Z]{3}', response.content.upper())
            return codes[0] if codes else location_name
            
    except Exception as e:
        print(f"✗ Location conversion failed for {location_name}: {e}")
        return location_name

async def location_to_city_code(location_name: str) -> str:
    """
    Convert location to city code for hotel search.
    
    Args:
        location_name: Location name or airport code
        
    Returns:
        3-letter city code
    """
    conversion_prompt = f"""
    Convert this location to the appropriate city code for hotel booking.
    
    Examples:
    - "Seoul" → "SEL"
    - "ICN" → "SEL" (Seoul city for hotels)
    - "Tokyo" → "TYO" 
    - "NRT" → "TYO" (Tokyo city for hotels)
    - "Paris" → "PAR"
    - "CDG" → "PAR" (Paris city for hotels)
    
    Location: "{location_name}"
    City Code:
    """
    
    try:
        response = await llm.ainvoke(conversion_prompt)
        city_code = response.content.strip().upper()
        
        if len(city_code) == 3 and city_code.isalpha():
            return city_code
        else:
            import re
            codes = re.findall(r'[A-Z]{3}', response.content.upper())
            return codes[0] if codes else location_name
                
    except Exception as e:
        print(f"✗ City code conversion failed for {location_name}: {e}")
        return location_name

async def location_to_coordinates(location_name: str) -> tuple:
    """
    Convert location to city center coordinates for activity search.
    
    Args:
        location_name: Location name or airport code
        
    Returns:
        Tuple of (latitude, longitude)
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
        coords = re.findall(r'-?\d+\.?\d*', coords_text)
        if len(coords) >= 2:
            return float(coords[0]), float(coords[1])
        else:
            return 0.0, 0.0
            
    except Exception as e:
        print(f"✗ Coordinate conversion failed for {location_name}: {e}")
        return 0.0, 0.0

# ==============================================================================
# TRAVEL PLAN ANALYSIS
# ==============================================================================

async def enhanced_travel_analysis(user_request: str) -> TravelPlan:
    """
    Extract structured trip information from natural language request.
    
    Uses LLM to analyze user intent, dates, preferences, and budget.
    
    Args:
        user_request: Natural language travel request
        
    Returns:
        Structured TravelPlan object
        
    Raises:
        ValueError: If request cannot be understood
    """
    analysis_prompt = f"""
    You are a world-class travel analyst AI. Extract structured trip information
    from the user's request and output valid JSON matching the provided schema.

    **User Request:** "{user_request}"
    
    **Today's Date:** {datetime.now().strftime('%Y-%m-%d')}

    **Instructions:**

    1. **Determine User Intent (`user_intent`):**
        - "full_plan": Combination of flights, hotels, or activities
        - "flights_only": Only asking for flights
        - "hotels_only": Only asking for hotels
        - "activities_only": Only asking for activities

    2. **Extract Core Details:**
        - `origin`: Starting location (can be null)
        - `destination`: Final destination (mandatory)
        - `departure_date` & `return_date`: Calculate absolute dates in YYYY-MM-DD format
        - `duration_days`: Calculate days between departure and return
        - `adults`: Number of travelers (default 1)

    3. **Extract Preferences:**
        - `travel_class`: Look for "business", "first class", etc. (default "ECONOMY")
        - `departure_time_pref` & `arrival_time_pref`: Look for time preferences
        - `total_budget`: Extract monetary value as float

    **CRITICAL: Output MUST be valid JSON matching this schema:**
    {TravelPlan.model_json_schema()}

    **JSON Output:**
    """
    
    try:
        response = await llm.ainvoke(analysis_prompt)
        
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:]  
        if content.endswith('```'):
            content = content[:-3]  
        content = content.strip()
        
        extracted_plan = TravelPlan.model_validate_json(content)
        print(f"✓ Travel plan extracted: intent={extracted_plan.user_intent}")
        return extracted_plan

    except Exception as e:
        print(f"✗ Travel analysis failed: {e}")
        raise ValueError(f"Could not understand the travel request: {e}")

# ==============================================================================
# FLIGHT SEARCH TOOL
# ==============================================================================

def _parse_and_prepare_offers(response_data: dict) -> List[Dict]:
    """
    Parse Amadeus flight search response into sortable format.
    
    Args:
        response_data: Raw Amadeus API response
        
    Returns:
        List of dicts with price_numeric and option_object
    """
    if 'data' not in response_data or not response_data['data']:
        return []
    
    prepared_offers = []
    carriers = response_data.get('dictionaries', {}).get('carriers', {})

    for offer in response_data['data']:
        try:
            price_float = float(offer['price']['total'])
            
            itinerary = offer['itineraries'][0]
            first_segment = itinerary['segments'][0]
            last_segment = itinerary['segments'][-1]
            
            option_obj = FlightOption(
                airline=carriers.get(
                    first_segment['carrierCode'], 
                    first_segment['carrierCode']
                ),
                price=f"{offer['price']['total']} {offer['price']['currency']}",
                departure_time=first_segment['departure']['at'],
                arrival_time=last_segment['arrival']['at'],
                duration=itinerary.get('duration')
            )
            
            prepared_offers.append({
                "price_numeric": price_float,
                "option_object": option_obj
            })
        except (ValueError, KeyError, IndexError, TypeError) as e:
            print(f"⚠ Skipping malformed flight offer: {e}")
            continue
            
    return prepared_offers

def find_closest_flight(offers: List[Dict], target_time_str: str) -> List[Dict]:
    """
    Sort flights by proximity to target departure time.
    
    Args:
        offers: List of prepared flight offers
        target_time_str: Target time in HH:MM format
        
    Returns:
        Sorted list of flight offers
    """
    try:
        target_hour = int(target_time_str.split(':')[0])
    except (ValueError, IndexError):
        print(f"⚠ Invalid target time: {target_time_str}")
        return offers

    def get_time_difference(prepared_offer):
        try:
            departure_dt = datetime.fromisoformat(
                prepared_offer['option_object'].departure_time
            )
            return abs(departure_dt.hour - target_hour)
        except (ValueError, TypeError):
            return float('inf')

    return sorted(offers, key=get_time_difference)

class FlightSearchArgs(BaseModel):
    """Flight search parameters"""
    originLocationCode: str = Field(description="Departure city IATA code")
    destinationLocationCode: str = Field(description="Arrival city IATA code")
    departureDate: str = Field(description="Departure date (YYYY-MM-DD)")
    returnDate: Optional[str] = Field(description="Return date (YYYY-MM-DD)") 
    adults: int = Field(description="Number of adult passengers", default=1)
    currencyCode: str = Field(description="Preferred currency", default="USD") 

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
    currencyCode: str = "USD"
) -> List[FlightOption]:
    """
    Search for flight offers using Amadeus API.
    
    Supports preferences for cabin class and departure time.
    Automatically converts location names to IATA codes.
    
    Args:
        originLocationCode: Origin city or airport code
        destinationLocationCode: Destination city or airport code
        departureDate: Departure date (YYYY-MM-DD)
        returnDate: Return date (YYYY-MM-DD) for round trips
        adults: Number of passengers
        travelClass: Cabin class (ECONOMY, BUSINESS, FIRST)
        departureTime: Preferred departure time or window
        arrivalTime: Preferred arrival time or window
        currencyCode: Currency for prices
        
    Returns:
        List of top 3 flight options sorted by relevance
    """
    print(f"→ Flight search: {originLocationCode} → {destinationLocationCode}")
    
    # Convert locations to IATA codes
    try:
        origin_task = location_to_airport_code(originLocationCode)
        destination_task = location_to_airport_code(destinationLocationCode)
        actual_origin, actual_destination = await asyncio.gather(
            origin_task, 
            destination_task
        )
        print(f"→ Converted to: {actual_origin} → {actual_destination}")
    except Exception as e:
        print(f"✗ Location conversion failed: {e}")
        return [FlightOption(
            airline="Location Error", 
            price="N/A", 
            departure_time="N/A", 
            arrival_time=str(e)
        )]

    if not amadeus: 
        return [FlightOption(
            airline="Error", 
            price="N/A", 
            departure_time="N/A", 
            arrival_time="Amadeus client not available"
        )]
    
    try:
        # Build search parameters
        search_params = {
            'originLocationCode': actual_origin,
            'destinationLocationCode': actual_destination,
            'departureDate': departureDate,
            'adults': adults,
            'nonStop': False, 
            'currencyCode': currencyCode,
            'max': 25
        }
        
        if returnDate:
            search_params['returnDate'] = returnDate
        
        if travelClass and travelClass.upper() in [
            "ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"
        ]:
            search_params['travelClass'] = travelClass.upper()

        # Add time window preferences
        time_windows = {
            "morning": "06:00-12:00", 
            "afternoon": "12:00-18:00", 
            "evening": "18:00-23:59"
        }
        if departureTime and departureTime.lower() in time_windows:
            search_params['departureWindow'] = time_windows[departureTime.lower()]
        if arrivalTime and arrivalTime.lower() in time_windows:
            search_params['arrivalWindow'] = time_windows[arrivalTime.lower()]

        # Execute search asynchronously
        print(f"→ Calling Amadeus with params: {search_params}")
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: amadeus.shopping.flight_offers_search.get(**search_params)
        )
        
        if not response.data: 
            return []
        
        # Parse and sort results
        all_offers = _parse_and_prepare_offers(response.result)
        
        if not all_offers:
            return []
        
        # Default sort by price
        final_sorted_offers = sorted(all_offers, key=lambda x: x['price_numeric'])

        # Re-sort by time if specific time requested
        if departureTime and ":" in departureTime:
            print(f"→ Re-sorting by proximity to {departureTime}")
            final_sorted_offers = find_closest_flight(final_sorted_offers, departureTime)

        # Return top 3 options
        top_3_offers = [item['option_object'] for item in final_sorted_offers[:3]]
        
        print(f"✓ Returning top 3 of {len(all_offers)} flight options")
        return top_3_offers

    except ResponseError as error:
        print(f"✗ Amadeus API error: {error}")
        return [FlightOption(
            airline="API Error", 
            price="N/A", 
            departure_time="N/A", 
            arrival_time=str(error)
        )]
    except Exception as e:
        print(f"✗ Flight search error: {e}")
        return [FlightOption(
            airline="System Error", 
            price="N/A", 
            departure_time="N/A", 
            arrival_time=str(e)
        )]

# ==============================================================================
# HOTEL SEARCH TOOL
# ==============================================================================

class HotelSearchArgs(BaseModel):
    """Hotel search parameters"""
    city_code: str = Field(description="City IATA code (e.g., 'PAR', 'NYC')")
    check_in_date: str = Field(description="Check-in date (YYYY-MM-DD)")
    check_out_date: str = Field(description="Check-out date (YYYY-MM-DD)")
    adults: int = Field(description="Number of guests", default=1)

@tool(args_schema=HotelSearchArgs)
async def search_and_compare_hotels(
    city_code: str, 
    check_in_date: str, 
    check_out_date: str, 
    adults: int = 1
) -> List[HotelOption]:
    """
    Search hotels across multiple providers (Amadeus + Hotelbeds).
    
    Automatically converts airport codes to city codes.
    Runs parallel searches across providers for best coverage.
    
    Args:
        city_code: Destination city or airport code
        check_in_date: Check-in date (YYYY-MM-DD)
        check_out_date: Check-out date (YYYY-MM-DD)
        adults: Number of guests
        
    Returns:
        Combined list of hotel options from all providers
    """
    # Convert to city code if needed
    actual_city_code = await location_to_city_code(city_code)
    print(f"→ Hotel search: {city_code} → {actual_city_code}")
    
    # Parallel search across providers
    amadeus_task = _search_amadeus_hotels(
        actual_city_code, 
        check_in_date, 
        check_out_date, 
        adults
    )
    hotelbeds_task = _search_hotelbeds_hotels(
        actual_city_code, 
        check_in_date, 
        check_out_date, 
        adults
    )
    
    results = await asyncio.gather(amadeus_task, hotelbeds_task)
    
    # Combine results
    combined_list = []
    for result_list in results:
        combined_list.extend(result_list)
    
    print(f"✓ Total hotels found: {len(combined_list)}")
    return combined_list

# ==============================================================================
# ACTIVITY SEARCH TOOL
# ==============================================================================

class ActivitySearchArgs(BaseModel):
    """Activity search parameters"""
    city_name: str = Field(
        description="Full city name for activity search (e.g., 'Paris', 'London')"
    )

@tool(args_schema=ActivitySearchArgs)
async def search_activities_by_city(city_name: str) -> List[ActivityOption]:
    """
    Search for activities and attractions in a city.
    
    Uses city center coordinates to find nearby activities.
    
    Args:
        city_name: Full city name
        
    Returns:
        List of activity options with pricing
    """
    print(f"→ Activity search: {city_name}")
    
    # Convert to coordinates
    lat, lng = await location_to_coordinates(city_name)
    print(f"→ Coordinates: ({lat}, {lng})")
    
    if lat == 0.0 and lng == 0.0:
        return [ActivityOption(
            name="Error", 
            description=f"Could not determine coordinates for {city_name}", 
            price="N/A"
        )]
    
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: amadeus.shopping.activities.get(
                latitude=lat, 
                longitude=lng, 
                radius=15
            )
        )
        
        qualified_activities = []
        for act in response.data[:10]:
            price_info = act.get('price')
            description = act.get('shortDescription') or act.get('description')
            activity_name = act.get('name', 'Unnamed Activity')

            if price_info or description:
                if price_info:
                    amount = price_info.get('amount', 'N/A')
                    currency = price_info.get('currencyCode', '')
                    price_str = f"{amount} {currency}".strip()
                else:
                    price_str = "Price on request"
                
                if not description:
                    description = "Experience this popular local activity"
                
                qualified_activities.append(ActivityOption(
                    name=activity_name, 
                    description=description, 
                    price=price_str,
                    location=city_name
                ))
            
            if len(qualified_activities) >= 8:
                break
        
        if not qualified_activities:
            return [ActivityOption(
                name="No activities found", 
                description="Unable to find activities", 
                price="N/A"
            )]
        
        print(f"✓ Found {len(qualified_activities)} activities")
        return qualified_activities

    except Exception as e:
        print(f"✗ Activity search failed: {e}")
        return [ActivityOption(
            name="Error", 
            description=f"Search failed: {e}", 
            price="N/A"
        )]

# ==============================================================================
# SMS NOTIFICATION TOOL (OPTIONAL)
# ==============================================================================

class SmsArgs(BaseModel):
    """SMS notification parameters"""
    to_number: str = Field(description="Recipient phone in E.164 format (+15551234567)")
    message: str = Field(description="SMS text message")

@tool(args_schema=SmsArgs)
def send_sms_notification(to_number: str, message: str) -> str:
    """
    Send SMS notification via Twilio.
    
    Optional feature - falls back to console logging if not configured.
    
    Args:
        to_number: Recipient's phone number
        message: Message text
        
    Returns:
        Success/failure message
    """
    if not twilio_client or not TWILIO_SENDER_PHONE:
        mock_message = f"→ SMS (Mock): TO={to_number}, MSG={message}"
        print(mock_message)
        return "SMS sent successfully (Mock)."
    
    try:
        sent_message = twilio_client.messages.create(
            to=to_number,
            from_=TWILIO_SENDER_PHONE,
            body=message
        )
        print(f"✓ SMS sent: SID={sent_message.sid}")
        return "SMS notification sent successfully."
    except Exception as e:
        print(f"✗ Twilio error: {e}")
        return f"Failed to send SMS: {e}"

# ==============================================================================
# CRM INTEGRATION TOOL (HubSpot by default)
# ==============================================================================
# NOTE: To use a different CRM (Salesforce, Pipedrive, etc.):
# 1. Replace HUBSPOT_API_KEY with your CRM's API key
# 2. Update the API endpoint URL below
# 3. Modify the payload structure to match your CRM's API
# 4. Update custom property names as needed
# ==============================================================================

class HubSpotArgs(BaseModel):
    """
    CRM integration data structure.
    Works with HubSpot by default - easily adaptable to other CRMs.
    """
    customer_info: Dict[str, str]
    travel_plan: TravelPlan
    recommendations: Dict[str, List] 
    original_request: str

@tool(args_schema=HubSpotArgs)
async def send_to_hubspot(
    customer_info: Dict[str, str],
    travel_plan: TravelPlan,
    recommendations: Dict[str, List],
    original_request: str
) -> str:
    """
    Send final travel plan to CRM (HubSpot by default).
    
    Creates a deal with rich travel data for tracking and reporting.
    
    CUSTOMIZATION: To use a different CRM:
    - Replace API endpoint and authentication
    - Modify payload structure
    - Update custom property names
    
    Args:
        customer_info: Customer contact details
        travel_plan: Structured travel plan
        recommendations: Generated packages or search results
        original_request: Original user message
        
    Returns:
        Success/failure message
    """
    if not HUBSPOT_API_KEY:
        print("⚠ CRM integration disabled (no API key)")
        return "CRM integration is not configured."
        
    print("→ Preparing CRM data")

    # Build human-readable description
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

    # Construct CRM payload
    # CUSTOMIZATION POINT: Modify this structure for your CRM
    hubspot_data = {
        "properties": {
            # Standard HubSpot properties
            "dealname": f"AI Plan: {travel_plan.destination} for {customer_info.get('name', 'New Lead')}",
            "amount": str(travel_plan.total_budget or 0),
            "dealstage": "appointmentscheduled",
            "description": description,
            
            # Custom properties for filtering & reporting
            # CUSTOMIZATION POINT: Replace with your CRM's custom fields
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
            "ai_generated_content": json.dumps(recommendations)
        }
    }
    
    try:
        # CUSTOMIZATION POINT: Replace with your CRM's API endpoint
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.hubapi.com/crm/v3/objects/deals",
                headers={"Authorization": f"Bearer {HUBSPOT_API_KEY}"},
                json=hubspot_data
            )
            response.raise_for_status()
            print("✓ Data sent to CRM successfully")
            return "Customer data sent to CRM successfully"
    except Exception as e:
        print(f"✗ CRM integration failed: {e}")
        return f"Failed to send to CRM: {e}"

# ==============================================================================
# PACKAGE GENERATION
# ==============================================================================

def _get_representative_options(
    options: List, 
    key_attr: str, 
    max_items: int = 7
) -> List:
    """
    Select representative sample (cheapest, mid-range, priciest).
    
    Keeps prompt concise while maintaining price range diversity.
    
    Args:
        options: List of options to sample
        key_attr: Attribute to sort by
        max_items: Maximum items to return
        
    Returns:
        Representative sample of options
    """
    if not options or len(options) <= max_items:
        return options
        
    try:
        if key_attr == 'price':
            options.sort(key=lambda x: float(getattr(x, key_attr).split(' ')[0]))
    except (ValueError, TypeError, IndexError):
        pass

    cheapest = options[:2]
    most_expensive = options[-2:]
    mid_index = len(options) // 2
    mid_range = options[mid_index-1 : mid_index+2]
    
    # Combine and deduplicate
    representative_sample = cheapest + mid_range + most_expensive
    seen = set()
    unique_sample = []
    for item in representative_sample:
        val = getattr(item, key_attr)
        if val not in seen:
            unique_sample.append(item)
            seen.add(val)
    return unique_sample

async def generate_travel_packages(
    trip_plan: TravelPlan, 
    all_options: Dict
) -> List[TravelPackage]:
    """
    Generate up to 3 travel packages (Budget, Balanced, Premium).
    
    Uses LLM to intelligently combine flights, hotels, and activities
    based on user's budget and preferences.
    
    Args:
        trip_plan: Structured travel plan with budget
        all_options: Dict containing flights, hotels, activities
        
    Returns:
        List of TravelPackage objects (1-3 packages)
    """
    if not trip_plan.total_budget or trip_plan.total_budget <= 0:
        print("⚠ Cannot generate packages without valid budget")
        return []

    # Sort and sample options
    sorted_flights = sorted(
        all_options.get('flights', []), 
        key=lambda x: float(x.price.split(' ')[0])
    )
    sorted_hotels = sorted(
        all_options.get('hotels', []), 
        key=lambda x: float(x.price.split(' ')[0])
    )
    sorted_activities = sorted(
        all_options.get('activities', []), 
        key=lambda x: float(x.price.split(' ')[0])
    )

    if not sorted_flights or not sorted_hotels:
        print("⚠ Insufficient options for package generation")
        return []

    rep_flights = _get_representative_options(sorted_flights, 'price')
    rep_hotels = _get_representative_options(sorted_hotels, 'name')
    rep_activities = _get_representative_options(sorted_activities, 'name', max_items=10)
    
    # Generate packages with LLM
    generation_prompt = f"""
    You are an expert travel consultant. Create up to 3 compelling travel packages
    for a client based on their plan and available options.

    **CLIENT'S PLAN:**
    - Destination: {trip_plan.destination}
    - Duration: {trip_plan.duration_days} nights
    - Budget: ${trip_plan.total_budget}

    **AVAILABLE OPTIONS (choose from these lists):**
    - Flights: {json.dumps([f.model_dump() for f in rep_flights])}
    - Hotels: {json.dumps([h.model_dump() for h in rep_hotels])}
    - Activities: {json.dumps([a.model_dump() for a in rep_activities])}

    **YOUR TASK:**
    1. Check if basic trip is possible within budget
    2. Create packages:
       - If cheapest combo is OVER budget: Create ONE "Budget" package only
       - If budget is reasonable: Create THREE packages (Budget, Balanced, Premium)
    3. Each package must contain:
       - EXACTLY ONE flight
       - EXACTLY ONE hotel
       - 0 to 2 activities
    4. Calculate `total_cost` = flight + (hotel × {trip_plan.duration_days} nights) + activities
    5. Calculate `budget_comment` based on difference from total budget
    6. Create creative `name` for each package

    **OUTPUT: Valid JSON array matching this schema:**
    {TravelPackage.model_json_schema()}

    **JSON Array Output:**
    """
    
    try:
        response = await llm.ainvoke(generation_prompt)
        packages = [
            TravelPackage.model_validate(p) 
            for p in json.loads(response.content)
        ]
        
        print(f"✓ Generated {len(packages)} packages")
        return packages

    except Exception as e:
        print(f"✗ Package generation failed: {e}")
        return []

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _calculate_default_dates(travel_plan: TravelPlan) -> tuple:
    """
    Calculate reasonable default dates for searches.
    
    Args:
        travel_plan: TravelPlan with possibly incomplete dates
        
    Returns:
        Tuple of (departure_date, return_date) in YYYY-MM-DD format
    """
    today = datetime.now()
    default_checkin = today + timedelta(days=30)
    default_checkout = default_checkin + timedelta(days=3)
    
    departure_date = travel_plan.departure_date
    return_date = travel_plan.return_date
    
    if not departure_date:
        departure_date = default_checkin.strftime('%Y-%m-%d')
    
    if not return_date:
        if travel_plan.duration_days:
            try:
                dep_dt = datetime.strptime(departure_date, '%Y-%m-%d')
                return_dt = dep_dt + timedelta(days=travel_plan.duration_days)
                return_date = return_dt.strftime('%Y-%m-%d')
            except ValueError:
                return_date = default_checkout.strftime('%Y-%m-%d')
        else:
            return_date = default_checkout.strftime('%Y-%m-%d')
    
    return departure_date, return_date

# ==============================================================================
# GRAPH NODES - Core Business Logic
# ==============================================================================

async def call_model_node(state: TravelAgentState) -> dict:
    """
    Analysis and execution node.
    
    Workflow:
    1. Check if customer info is needed
    2. Analyze user request with LLM
    3. Prepare tool calls based on intent
    4. Execute tools in parallel
    5. Return results for synthesis
    
    Args:
        state: Current graph state
        
    Returns:
        Updated state with tool results
    """
    print("━━━ NODE: Analysis & Execution ━━━")
    
    is_continuation = state.get("is_continuation", False)

    # Check if customer info form is needed
    if (not is_continuation and
        not state.get('customer_info') and 
        state.get('current_step') in [None, "initial"] and 
        len(state.get('messages', [])) <= 1):
        
        return {
            "messages": [],
            "current_step": "collecting_info", 
            "form_to_display": "customer_info",
            "original_request": state['messages'][-1].content,
        }

    user_request = state['messages'][-1].content
    customer_info = state.get('customer_info', {})
    
    try:
        # Phase 1: Analyze request
        print("→ Phase 1: Analyzing request")
        travel_plan = await enhanced_travel_analysis(user_request)

        # Inject budget from customer form
        if customer_info.get('budget'):
            try:
                budget_str = customer_info['budget'].upper().replace("USD", "").replace("$", "").strip()
                travel_plan.total_budget = float(budget_str)
                print(f"→ Budget injected: ${travel_plan.total_budget}")
            except (ValueError, TypeError):
                print(f"⚠ Could not parse budget: {customer_info.get('budget')}")
        
        state['travel_plan'] = travel_plan

        # Phase 2: Prepare tool calls
        print(f"→ Phase 2: Preparing tools (intent: {travel_plan.user_intent})")
        
        tasks_and_names = []
        default_checkin, default_checkout = _calculate_default_dates(travel_plan)
        
        departure_date = travel_plan.departure_date or default_checkin
        return_date = travel_plan.return_date or default_checkout

        # Validate dates
        try:
            datetime.strptime(departure_date, '%Y-%m-%d')
            if return_date:
                datetime.strptime(return_date, '%Y-%m-%d')
        except ValueError as e:
            print(f"⚠ Invalid date, using defaults: {e}")
            departure_date = default_checkin
            return_date = default_checkout

        # Prepare flight search
        if travel_plan.user_intent in ["full_plan", "flights_only"] and travel_plan.origin and travel_plan.destination:
            task = search_flights.ainvoke({
                "originLocationCode": travel_plan.origin,
                "destinationLocationCode": travel_plan.destination,
                "departureDate": departure_date,
                "returnDate": return_date,
                "adults": travel_plan.adults,
                "currencyCode": "USD",
                "travelClass": travel_plan.travel_class,
                "departureTime": travel_plan.departure_time_pref,
                "arrivalTime": travel_plan.arrival_time_pref
            })
            tasks_and_names.append((task, "search_flights"))
        
        # Prepare hotel search
        if travel_plan.user_intent in ["full_plan", "hotels_only"] and travel_plan.destination:
            task = search_and_compare_hotels.ainvoke({
                "city_code": travel_plan.destination,
                "check_in_date": departure_date,
                "check_out_date": return_date,
                "adults": travel_plan.adults
            })
            tasks_and_names.append((task, "search_and_compare_hotels"))

        # Prepare activity search
        if travel_plan.user_intent in ["full_plan", "activities_only"] and travel_plan.destination:
            task = search_activities_by_city.ainvoke({
                "city_name": travel_plan.destination
            })
            tasks_and_names.append((task, "search_activities_by_city"))
        
        if not tasks_and_names:
            print("⚠ No tools to call")
            return {
                "messages": [AIMessage(
                    content="I've understood your request, but there's no specific search I can perform. How else can I help?"
                )],
                "current_step": "complete",
                "travel_plan": travel_plan
            }

        # Phase 3: Execute tools in parallel
        print(f"→ Phase 3: Executing {len(tasks_and_names)} tools in parallel")
        
        tasks = [task for task, name in tasks_and_names]
        tool_results = await asyncio.gather(*tasks, return_exceptions=True)
        processed_messages = []

        for i, (result, (_, tool_name)) in enumerate(zip(tool_results, tasks_and_names)):
            if isinstance(result, Exception):
                print(f"✗ Tool {tool_name} failed: {result}")
                content = "[]"
            else:
                try:
                    content = json.dumps([item.model_dump() for item in result])
                except Exception as e:
                    print(f"✗ Serialization failed for {tool_name}: {e}")
                    content = "[]"
            
            processed_messages.append(ToolMessage(
                content=content, 
                name=tool_name,
                tool_call_id=f"call_{tool_name}_{i}" 
            ))
        
        print("✓ All tools executed")
        return {
            "messages": processed_messages,
            "current_step": "synthesizing",
            "travel_plan": travel_plan
        }
    
    except ValueError as e:
        print(f"✗ Analysis failed: {e}")
        response = AIMessage(
            content="I'm sorry, I had trouble understanding your request. Could you rephrase it?"
        )
        return {"messages": [response], "current_step": "complete"}
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        response = AIMessage(
            content="I apologize, but a system error occurred. Please try again."
        )
        return {"messages": [response], "current_step": "complete"}

async def synthesize_results_node(state: TravelAgentState) -> dict:
    """
    Package generation and final response node.
    
    Workflow:
    1. Parse tool results from state
    2. Generate travel packages (if applicable)
    3. Create final user response
    4. Send to CRM
    
    Args:
        state: Current graph state with tool results
        
    Returns:
        Updated state with final response
    """
    print("━━━ NODE: Synthesis & Response ━━━")
    
    # Parse tool results
    tool_results = {}
    for msg in state['messages']:
        if isinstance(msg, ToolMessage):
            try:
                tool_results[msg.name] = msg.content
            except Exception as e:
                print(f"⚠ Failed to process {msg.name}: {e}")
                tool_results[msg.name] = "[]"
    
    travel_plan = state.get('travel_plan')
    
    # Organize options by type
    all_options = {'flights': [], 'hotels': [], 'activities': []}
    for tool_name, content in tool_results.items():
        try:
            if content and content != "[]":
                parsed_data = json.loads(content)
                if tool_name == "search_flights":
                    all_options['flights'] = [
                        FlightOption.model_validate(f) for f in parsed_data
                    ]
                elif tool_name == "search_and_compare_hotels":
                    all_options['hotels'] = [
                        HotelOption.model_validate(h) for h in parsed_data
                    ]
                elif tool_name == "search_activities_by_city":
                    all_options['activities'] = [
                        ActivityOption.model_validate(a) for a in parsed_data
                    ]
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"✗ Failed to parse {tool_name}: {e}")

    # Generate packages if applicable
    packages = []
    if (travel_plan and 
        travel_plan.user_intent == "full_plan" and 
        travel_plan.total_budget and 
        all_options['flights'] and 
        all_options['hotels']):
        
        print("→ Generating travel packages")
        try:
            packages = await generate_travel_packages(travel_plan, all_options)
        except Exception as e:
            print(f"✗ Package generation failed: {e}")
            packages = []

    # Create final response
    synthesis_prompt = ""
    hubspot_recommendations = {}

    if packages:
        print(f"→ Preparing response with {len(packages)} packages")
        synthesis_prompt = f"""You are an AI travel assistant. Present these custom travel packages professionally.

**GENERATED PACKAGES:**
{json.dumps([p.model_dump() for p in packages], indent=2)}

**YOUR TASK:**
- Start with a warm greeting
- Present ALL packages with clear details (flight, hotel, activities)
- Highlight the "Balanced" package as recommended
- End with clear call to action
"""
        hubspot_recommendations = {"packages": [p.model_dump() for p in packages]}
    else:
        print("→ Preparing response with search results")
        has_results = any(all_options.values())
        
        if has_results:
            tool_results_for_prompt = {
                "flights": [f.model_dump() for f in all_options.get('flights', [])],
                "hotels": [h.model_dump() for h in all_options.get('hotels', [])],
                "activities": [a.model_dump() for a in all_options.get('activities', [])]
            }
            synthesis_prompt = f"""You are an AI travel assistant. Present these search results clearly.

**SEARCH RESULTS:**
{json.dumps(tool_results_for_prompt, indent=2)}

Organize and present options in a user-friendly format.
"""
            hubspot_recommendations = tool_results_for_prompt
        else:
            synthesis_prompt = """You are an AI travel assistant. 
Apologize that no options were found and offer to help refine the search."""
            hubspot_recommendations = {"error": "No results found"}

    # Generate final response
    try:
        final_response = await llm.ainvoke(synthesis_prompt)
    except Exception as e:
        print(f"✗ Response generation failed: {e}")
        final_response = AIMessage(
            content="I apologize, but I encountered an issue generating your recommendations. Please try again."
        )

    # Send to CRM
    if state.get('customer_info') and travel_plan:
        try:
            await send_to_hubspot.ainvoke({
                'customer_info': state['customer_info'],
                'travel_plan': travel_plan,
                'recommendations': hubspot_recommendations,
                'original_request': state.get('original_request', '')
            })
        except Exception as e:
            print(f"⚠ CRM integration warning: {e}")

    return {
        "messages": [final_response],
        "current_step": "complete"
    }

# ==============================================================================
# GRAPH CONSTRUCTION
# ==============================================================================

# Bind tools to LLM
tools = [
    search_flights, 
    search_and_compare_hotels, 
    search_activities_by_city, 
    send_sms_notification, 
    send_to_hubspot
]
tool_llm = llm.bind_tools(tools)

def build_enhanced_graph(checkpointer=None):
    """
    Build the production LangGraph workflow.
    
    Graph structure:
    - Entry: call_model_and_tools (analysis & execution)
    - Conditional: Based on current_step
      - collecting_info → END (wait for customer form)
      - synthesizing → synthesize_results
      - complete → END
    - Exit: Always END after synthesis
    
    Args:
        checkpointer: Optional checkpointer for conversation persistence
        
    Returns:
        Compiled LangGraph
    """
    if checkpointer is None:
        checkpointer = InMemorySaver()
    
    workflow = StateGraph(TravelAgentState)
    
    # Define nodes
    workflow.add_node("call_model_and_tools", call_model_node)
    workflow.add_node("synthesize_results", synthesize_results_node)

    # Define flow
    workflow.set_entry_point("call_model_and_tools")

    # Conditional routing based on step
    workflow.add_conditional_edges(
        "call_model_and_tools",
        lambda state: state["current_step"],
        {
            "collecting_info": END,
            "synthesizing": "synthesize_results",
            "complete": END
        }
    )
    
    # Synthesis always ends
    workflow.add_edge("synthesize_results", END)
    
    print("✓ Graph compiled successfully")
    return workflow.compile(checkpointer=checkpointer)

# ==============================================================================
# PRODUCTION USAGE
# ==============================================================================

if __name__ == "__main__":
    """
    Example usage for testing.
    
    In production, this graph should be invoked by a FastAPI server
    or other web framework.
    """
    print("=" * 80)
    print("Multi-Agent Travel Booking System")
    print("Production-ready LangGraph implementation")
    print("=" * 80)
    
    graph = build_enhanced_graph()
    print("\n✓ Graph ready for production use")
    print("\nTo integrate:")
    print("1. Import: from agent_graph import build_enhanced_graph")
    print("2. Initialize: graph = build_enhanced_graph()")
    print("3. Invoke: await graph.ainvoke({'messages': [HumanMessage(content=query)]})")
