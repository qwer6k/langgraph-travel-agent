from typing import TypedDict, Annotated, List, Optional, Dict, Any, Literal, Union
import operator

from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage


# ---------------------------------------------------------------------------
# Core option schemas (unchanged)
# ---------------------------------------------------------------------------

class FlightOption(BaseModel):
    airline: str = Field(description="Airline name")
    price: str = Field(description="Total flight cost")
    departure_time: str = Field(description="Departure time (YYYY-MM-DDTHH:MM:SS)")
    arrival_time: str = Field(description="Arrival time (YYYY-MM-DDTHH:MM:SS)")
    duration: Optional[str] = Field(description="Flight duration", default=None)
    is_error: bool = False
    error_message: Optional[str] = None


class HotelOption(BaseModel):
    name: str = Field(description="Hotel name")
    category: str = Field(description="Star rating, e.g., '5EST' for 5-star")
    price_per_night: str = Field(description="Price per night")
    source: str = Field(description="Data source (e.g., 'Amadeus', 'Hotelbeds')")
    rating: Optional[float] = Field(description="Hotel rating", default=None)
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
        description="User's primary goal",
    )


# ---------------------------------------------------------------------------
# PR2/PR3-ready: execution plan schemas (NEW, optional)
# ---------------------------------------------------------------------------

ToolName = Literal[
    "search_flights",
    "search_and_compare_hotels",
    "search_activities_by_city",
]

DecisionType = Literal[
    "ASK",      # ask user for missing info (dates, etc.)
    "EXECUTE",  # run tools
    "RELAX",    # relax constraints (PR3)
    "RETRY",    # retry tool call (PR3)
    "DONE",     # finish
]


class ExecutionTask(BaseModel):
    """
    单个可执行任务（未来 PR3 可以扩展：失败策略、重试次数、RELAX 参数等）
    """
    tool_name: ToolName
    reason: Optional[str] = None
    args: Dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(BaseModel):
    """
    执行计划：PR2 先用于“可解释”，PR3 用于驱动自治循环。
    """
    intent: Literal["full_plan", "flights_only", "hotels_only", "activities_only"]
    tasks: List[Union[ToolName, ExecutionTask]] = Field(default_factory=list)
    decision: Optional[DecisionType] = None
    ask: Optional[str] = None
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# LangGraph state contract
# ---------------------------------------------------------------------------

class TravelAgentState(TypedDict):
    """Graph state with complete conversation context."""

    messages: Annotated[List[AnyMessage], operator.add]
    travel_plan: Optional[TravelPlan]
    user_preferences: Optional[Dict[str, Any]]
    form_to_display: Optional[str]

    # "collecting_info" / "synthesizing" / "complete" etc.
    current_step: str

    errors: List[str]
    customer_info: Optional[Dict[str, str]]
    trip_details: Optional[Dict[str, Any]]
    original_request: Optional[str]
    is_continuation: Optional[bool]
    one_way_detected: Optional[bool]

    # ---- PR2 additions (important) ----
    one_way: Optional[bool]

    # 你现在代码里会读写这个（用于 tool key 复用）
    last_tool_args: Optional[Dict[str, Dict[str, Any]]]

    # 你 PR2 在 call_model_node 里写入了 execution_plan（目前是 dict，也允许未来换成 model_dump）
    execution_plan: Optional[Dict[str, Any]]

    # 你现在用于 refresh 的 hint（synthesis 可选用）
    user_followup_hint: Optional[str]

    # ---- synthesize_results_node() 写入的字段（你原来就有） ----
    flight_error_message: Optional[str]
    activity_error_message: Optional[str]
    hotel_error_message: Optional[str]
