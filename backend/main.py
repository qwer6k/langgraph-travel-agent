"""FastAPI Server - Multi-Agent Travel Booking System.

This server uses LangGraph with async background task processing.

2025-12: Migrated customer_info HITL to LangGraph native interrupt/resume.
- First /chat may return an interrupt (customer form request)
- Frontend submits to /chat/resume (or /chat/customer-info alias), which resumes
    execution via Command(resume=...).
- State is persisted to SQLite via SqliteSaver.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
from langchain_core.messages import HumanMessage
import uuid
from pathlib import Path
import os

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from backend.travel_agent import build_enhanced_graph

# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

app = FastAPI(
    title="Travel AI Assistant API",
    description="Async multi-agent system for intelligent travel planning",
    version="1.0.0"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# SQLite checkpoint DB (simple external persistence)
LANGGRAPH_SQLITE_PATH = os.getenv(
    "LANGGRAPH_SQLITE_PATH",
    str(PROJECT_ROOT / ".langgraph_checkpoints.sqlite"),
)

# In-memory job store for async task tracking
# PRODUCTION: Replace with Redis for scalability
jobs = {}
# Track threads that are currently waiting for a resume (HITL)
# Maps thread_id -> task_id (the task that produced the interrupt)
waiting_for_resume: dict[str, str] = {}

def _ensure_sqlite_parent_dir() -> None:
    db_path = Path(LANGGRAPH_SQLITE_PATH)
    if db_path.parent and str(db_path.parent) not in (".", ""):
        db_path.parent.mkdir(parents=True, exist_ok=True)

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


class ResumeRequest(BaseModel):
    """Resume a paused LangGraph execution via Command(resume=...)."""

    thread_id: str = Field(min_length=5)
    resume: dict

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
        config = {"configurable": {"thread_id": thread_id}}

        _ensure_sqlite_parent_dir()

        initial_state = {
            "messages": [HumanMessage(content=message)],
            "is_continuation": is_continuation,
        }

        # Compile + invoke with SQLite checkpointer
        async with AsyncSqliteSaver.from_conn_string(LANGGRAPH_SQLITE_PATH) as saver:
            graph = build_enhanced_graph(checkpointer=saver)
            final_state = await graph.ainvoke(initial_state, config)

        # ============================================================
        # 计算 reply
        # ============================================================
        # If graph interrupted (HITL), return a stable prompt + form trigger
        if isinstance(final_state, dict) and final_state.get("__interrupt__"):
            # mark this thread as waiting for resume so backend can enforce resume-only flow
            try:
                waiting_for_resume[thread_id] = task_id
            except Exception:
                pass

            jobs[task_id] = {
                "status": "completed",
                "result": {
                    "reply": (
                        "✅ I have noted your travel needs.\n\n"
                        "Please fill in your contact information below (name, email, budget, etc.). "
                        "After that, I will immediately continue from where we paused."
                    )
                },
                "form_to_display": "customer_info",
            }
            print(f"✓ Background task {task_id} interrupted (customer_info)")
            return

        reply: str | None = None
        msgs = (final_state.get("messages", []) if isinstance(final_state, dict) else []) or []

        # 是否已经有“非 HumanMessage”的回复（ToolMessage / AIMessage）
        has_non_human = any(not isinstance(m, HumanMessage) for m in msgs)

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

        # Backward-compatible: allow agents to signal a form trigger
        if isinstance(final_state, dict) and final_state.get("form_to_display"):
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


async def run_resume_in_background(task_id: str, thread_id: str, resume: dict):
    print(f"→ Resume task {task_id} started")
    try:
        # clear waiting flag as we're about to consume the resume for this thread
        try:
            waiting_for_resume.pop(thread_id, None)
        except Exception:
            pass
        config = {"configurable": {"thread_id": thread_id}}

        _ensure_sqlite_parent_dir()

        async with AsyncSqliteSaver.from_conn_string(LANGGRAPH_SQLITE_PATH) as saver:
            graph = build_enhanced_graph(checkpointer=saver)
            final_state = await graph.ainvoke(Command(resume=resume), config)

        # If still interrupted, keep asking (rare but possible)
        if isinstance(final_state, dict) and final_state.get("__interrupt__"):
            jobs[task_id] = {
                "status": "completed",
                "result": {
                    "reply": "Still need more information. Please complete the form.",
                },
                "form_to_display": "customer_info",
            }
            print(f"✓ Resume task {task_id} interrupted again")
            return

        reply: str | None = None
        msgs = (final_state.get("messages", []) if isinstance(final_state, dict) else []) or []
        for msg in reversed(msgs):
            if isinstance(msg, HumanMessage):
                continue
            content = getattr(msg, "content", None)
            if content:
                reply = str(content)
                break
        if reply is None:
            reply = "I've processed the information."

        jobs[task_id] = {"status": "completed", "result": {"reply": reply}}
        print(f"✓ Resume task {task_id} completed")
    except Exception as e:
        import traceback

        traceback.print_exc()
        jobs[task_id] = {"status": "failed", "result": {"error": str(e)}}
        print(f"✗ Resume task {task_id} failed: {e}")


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

    # include metadata for better traceability and cleanup
    jobs[task_id] = {"status": "running", "thread_id": request.thread_id, "is_continuation": bool(request.is_continuation)}
    # If this thread currently awaits a resume (HITL), block non-resume starts
    if waiting_for_resume.get(request.thread_id):
        # Client should call /chat/resume to continue the interrupted flow
        raise HTTPException(status_code=409, detail="This thread is waiting for a form response. Please submit via /chat/resume or /chat/customer-info to resume the interrupted session.")

    # If the client explicitly requests a fresh start (is_continuation=False),
    # delete any previous checkpoint for this thread so the graph starts clean.
    if not request.is_continuation:
        _ensure_sqlite_parent_dir()
        async def _delete_checkpoint():
            try:
                async with AsyncSqliteSaver.from_conn_string(LANGGRAPH_SQLITE_PATH) as saver:
                    await saver.adelete_thread(request.thread_id)
            except Exception as e:
                # If the DB is empty or schema not yet created, ignore deletion errors
                print(f"⚠ Could not delete checkpoint for thread {request.thread_id}: {e}")
        # Execute deletion immediately (await here to ensure deletion before background run)
        await _delete_checkpoint()

    background_tasks.add_task(
        run_agent_in_background,
        task_id,
        request.thread_id,
        request.message,
        request.is_continuation,
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

@app.post("/chat/customer-info", response_model=TaskResponse, tags=["AI Agent"])
async def submit_customer_info(request: CustomerInfoRequest, background_tasks: BackgroundTasks):
    """
    Submit customer information for a conversation thread.
    
    This is called when the frontend collects customer details
    (name, email, phone, budget) via a form mid-conversation.
    
    The information is stored and injected into subsequent agent calls.
    """
    # Backward-compatible alias: resume the interrupted graph.
    task_id = str(uuid.uuid4())
    # clear waiting flag before enqueueing resume
    waiting_for_resume.pop(request.thread_id, None)
    jobs[task_id] = {"status": "running", "thread_id": request.thread_id, "is_continuation": True}
    print(f"→ Customer info received for thread {request.thread_id}, resume task: {task_id}")

    background_tasks.add_task(run_resume_in_background, task_id, request.thread_id, request.customer_info)

    return TaskResponse(task_id=task_id)


@app.post("/chat/resume", response_model=TaskResponse, tags=["AI Agent"])
async def resume_chat(request: ResumeRequest, background_tasks: BackgroundTasks):
    """Resume a paused execution.

    Frontend should call this after it receives a customer_info form trigger.
    """
    task_id = str(uuid.uuid4())
    # clear waiting flag before enqueueing resume
    waiting_for_resume.pop(request.thread_id, None)
    jobs[task_id] = {"status": "running", "thread_id": request.thread_id, "is_continuation": True}
    background_tasks.add_task(run_resume_in_background, task_id, request.thread_id, request.resume)
    print(f"→ Resume task created: {task_id}")
    return TaskResponse(task_id=task_id)

@app.delete("/chat/thread/{thread_id}", tags=["AI Agent"])
async def clear_thread(thread_id: str):
    """
    Clear stored data for a conversation thread.
    同时重建 agent_graph，把 InMemorySaver 里的所有 checkpoint 一起清掉。
    """
    # 1) clear checkpoints for this thread
    _ensure_sqlite_parent_dir()
    async with AsyncSqliteSaver.from_conn_string(LANGGRAPH_SQLITE_PATH) as saver:
        await saver.adelete_thread(thread_id)

    # 2) clear in-memory job statuses (optional)
    for k, v in list(jobs.items()):
        try:
            if isinstance(v, dict) and v.get("thread_id") == thread_id:
                jobs.pop(k, None)
        except Exception:
            pass

    # clear waiting flag if any
    waiting_for_resume.pop(thread_id, None)

    return {"status": "cleared", "thread_id": thread_id}

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
