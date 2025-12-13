"""
FastAPI Server - Multi-Agent Travel Booking System
Production-ready async server with background task processing.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
from langchain_core.messages import HumanMessage
import uuid
from backend.travel_agent import build_enhanced_graph 
import asyncio

# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Travel AI Assistant API",
    description="Async multi-agent system for intelligent travel planning",
    version="1.0.0"
)

# Initialize agent graph
agent_graph = build_enhanced_graph()

# In-memory job store for async task tracking
# PRODUCTION: Replace with Redis for scalability
jobs = {}

# In-memory customer data storage
# PRODUCTION: Replace with database or Redis
customer_data = {}

# ============================================================================
# CORS CONFIGURATION
# ============================================================================

# Configure allowed origins for frontend integration
# PRODUCTION: Replace with your actual frontend domain(s)
origins = [
    "http://localhost:3000",           # Local development
    "http://localhost:3001",           # Alternative local port
    # Add your production frontend URL here:
    # "https://your-frontend-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Incoming chat request from frontend"""
    message: str = Field(min_length=1, description="User's travel query")
    thread_id: str = Field(min_length=5, description="Conversation thread ID")
    is_continuation: Optional[bool] = Field(False, description="Is this continuing a previous conversation")

class TaskResponse(BaseModel):
    """Response with async task ID for status polling"""
    task_id: str

class StatusResponse(BaseModel):
    """Status response for async task polling"""
    status: str  # "running", "completed", "failed"
    result: dict | None = None
    form_to_display: str | None = None

class CustomerInfoRequest(BaseModel):
    """Customer information submission"""
    thread_id: str = Field(min_length=5)
    customer_info: dict

# ============================================================================
# BACKGROUND TASK PROCESSING
# ============================================================================

async def run_agent_in_background(
    task_id: str,
    thread_id: str,
    message: str,
    is_continuation: bool = False,
):
    print(f"→ Background task {task_id} started (continuation: {is_continuation})")

    try:
        # ✅ checkpoint key 永远稳定：只用 thread_id
        graph_thread_id = thread_id
        config = {"configurable": {"thread_id": graph_thread_id}}

        # ✅ 只传本轮用户消息 + is_continuation +（可选）customer_info
        #    不要传 current_step，避免覆盖 checkpoint 内状态
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "is_continuation": is_continuation,
        }

        if thread_id in customer_data:
            initial_state["customer_info"] = customer_data[thread_id]
            print(f"→ Using stored customer info for thread {thread_id}")

        # 调 LangGraph
        final_state = await agent_graph.ainvoke(initial_state, config)

        # ============================================================
        # 计算 reply
        # ============================================================
        reply: str | None = None
        msgs = final_state.get("messages", []) or []

        # 是否已经有“非 HumanMessage”的回复（ToolMessage / AIMessage）
        has_non_human = any(not isinstance(m, HumanMessage) for m in msgs)

        # ✅ collecting_info 的提示：只依赖 final_state（更稳，不依赖 is_continuation）
        if (
            final_state.get("form_to_display") == "customer_info"
            and final_state.get("current_step") == "collecting_info"
            and not has_non_human
        ):
            reply = (
                "✅ I have noted your travel needs.\n\n"
                "Please fill in your contact information below (name, email, budget, etc.). "
                "After that, I will immediately search for suitable flights, hotels, and activities."
            )
        else:
            # 从最后往前找一条“非 HumanMessage”的消息作为回复
            for msg in reversed(msgs):
                if isinstance(msg, HumanMessage):
                    continue
                content = getattr(msg, "content", None)
                if content:
                    reply = str(content)
                    break

        if reply is None:
            reply = "I've processed the information."

        result_data = {
            "status": "completed",
            "result": {"reply": reply},
        }

        # 仍然把 form_to_display 返回给前端，维持交互逻辑
        if final_state.get("form_to_display"):
            result_data["form_to_display"] = final_state["form_to_display"]

        jobs[task_id] = result_data
        print(f"✓ Background task {task_id} completed")

    except Exception as e:
        import traceback

        traceback.print_exc()
        jobs[task_id] = {
            "status": "failed",
            "result": {"error": str(e)},
        }
        print(f"✗ Background task {task_id} failed: {e}")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Status"])
def root():
    """Root endpoint - health check"""
    return {
        "status": "ok",
        "service": "Travel AI Assistant",
        "architecture": "async",
        "version": "1.0.0"
    }

@app.get("/health", tags=["Status"])
def health():
    """Health check endpoint for monitoring"""
    return {"status": "healthy"}

@app.post("/chat", response_model=TaskResponse, tags=["AI Agent"])
async def start_chat_task(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Start an async chat task with the AI agent.
    
    Returns a task_id immediately for status polling.
    The actual processing happens in the background.
    
    Flow:
    1. POST /chat → Get task_id
    2. Poll GET /chat/status/{task_id} until completed
    3. Extract result from status response
    """
    task_id = str(uuid.uuid4())
    jobs[task_id] = {"status": "running"}
    
    background_tasks.add_task(
        run_agent_in_background,
        task_id,
        request.thread_id,
        request.message,
        request.is_continuation
    )
    
    print(f"→ Chat task created: {task_id}")
    return TaskResponse(task_id=task_id)

@app.get("/chat/status/{task_id}", response_model=StatusResponse, tags=["AI Agent"])
async def get_task_status(task_id: str):
    """
    Poll the status of an async chat task.
    
    Frontend should poll this endpoint every 2-3 seconds until status is "completed" or "failed".
    
    Returns:
        StatusResponse with status and optional result
    """
    job = jobs.get(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")
    return StatusResponse(**job)

@app.post("/chat/customer-info", tags=["AI Agent"])
async def submit_customer_info(request: CustomerInfoRequest):
    """
    Submit customer information for a conversation thread.
    
    This is called when the frontend collects customer details
    (name, email, phone, budget) via a form mid-conversation.
    
    The information is stored and injected into subsequent agent calls.
    """
    customer_data[request.thread_id] = request.customer_info
    print(f"→ Customer info stored for thread {request.thread_id}")
    
    return {
        "status": "received",
        "message": "Customer information saved successfully"
    }

@app.delete("/chat/thread/{thread_id}", tags=["AI Agent"])
async def clear_thread(thread_id: str):
    """
    Clear stored data for a conversation thread.
    同时重建 agent_graph，把 InMemorySaver 里的所有 checkpoint 一起清掉。
    """
    # 1. 清业务数据
    customer_data.pop(thread_id, None)

    # 2. 重建图 = 清掉所有 thread 的 checkpoint（单用户最粗暴有效）
    global agent_graph
    agent_graph = build_enhanced_graph()

    print(f"→ Thread {thread_id} customer data cleared & graph rebuilt (all checkpoints dropped)")
    return {"status": "cleared"}

# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    print("=" * 80)
    print("Travel AI Assistant - Server Starting")
    print("=" * 80)
    print("✓ Agent graph initialized")
    print("✓ CORS configured")
    print("✓ Ready to accept requests")
    print("=" * 80)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n" + "=" * 80)
    print("Server shutting down")
    print("=" * 80)

# ============================================================================
# LOCAL DEVELOPMENT
# ============================================================================

if __name__ == "__main__":
    """
    Run with: python main.py
    
    For production deployment, use:
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
    """
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
