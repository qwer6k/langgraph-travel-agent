from typing import TypedDict, Annotated, List, Optional, Dict, Any, Literal
import operator

from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage


class FlightOption(BaseModel):
    airline: str = Field(description="Airline name")
    price: str = Field(description="Total flight cost")
    departure_time: str = Field(description="Departure time (YYYY-MM-DDTHH:MM:SS)")
    arrival_time: str = Field(description="Arrival time (YYYY-MM-DDTHH:MM:SS)")
    duration: Optional[str] = Field(description="Flight duration", default=None)
    # 新增：用于标记这是一个“错误占位”，不是正常航班
    is_error: bool = False
    error_message: Optional[str] = None
    


class HotelOption(BaseModel):
    name: str = Field(description="Hotel name")
    category: str = Field(description="Star rating, e.g., '5EST' for 5-star")
    price_per_night: str = Field(description="Price per night")
    source: str = Field(description="Data source (e.g., 'Amadeus', 'Hotelbeds')")
    rating: Optional[float] = Field(description="Hotel rating", default=None)
    # ✅ 新增：用于区分“接口挂了” vs “确实无库存”
    is_error: bool = False
    error_message: Optional[str] = None


class ActivityOption(BaseModel):
    name: str = Field(description="Activity name")
    description: str = Field(description="Brief description")
    price: str = Field(description="Activity price")
    location: Optional[str] = Field(description="Activity location", default=None)
    is_error: bool = False
    error_message: Optional[str] = None


class TravelPackage(BaseModel):
    name: str = Field(description="Package name, e.g., 'Smart Explorer'")
    grade: Literal["Budget", "Balanced", "Premium"] = Field(description="Package tier")
    total_cost: float = Field(description="Total package cost in USD")
    budget_comment: str = Field(description="Budget comparison comment")
    selected_flight: FlightOption = Field(description="Selected flight option")
    selected_hotel: HotelOption = Field(description="Selected hotel option")
    selected_activities: List[ActivityOption] = Field(
        default_factory=list,
        max_items=2,
        description="0-2 activity options",
    )


class TravelPackageList(BaseModel):
    packages: List[TravelPackage]


class TravelPlan(BaseModel):
    """Structured travel plan extracted from user request"""

    origin: Optional[str] = Field(None, description="Origin city or airport code")
    destination: str = Field(..., description="Destination city or airport code")
    departure_date: Optional[str] = Field(
        None, description="Departure date (YYYY-MM-DD)"
    )
    return_date: Optional[str] = Field(
        None, description="Return date (YYYY-MM-DD)"
    )
    duration_days: Optional[int] = Field(
        None, description="Trip duration in days"
    )
    adults: int = Field(1, description="Number of adult travelers")
    travel_class: Optional[
        Literal["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"]
    ] = "ECONOMY"
    departure_time_pref: Optional[str] = Field(
        None, description="Preferred departure time"
    )
    arrival_time_pref: Optional[str] = Field(
        None, description="Preferred arrival time"
    )
    total_budget: Optional[float] = Field(
        None, description="Total budget in USD"
    )
    user_intent: Literal[
        "full_plan", "flights_only", "hotels_only", "activities_only"
    ] = Field("full_plan", description="User's primary goal")


class TravelAgentState(TypedDict):
    """Graph state with complete conversation context."""

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

    # ✅ synthesize_results_node() 里会写入的字段，补进契约避免后续踩雷
    flight_error_message: Optional[str]
    activity_error_message: Optional[str]
    hotel_error_message: Optional[str]

