"""最小端到端验证：FastAPI(/chat + /chat/resume) + LangGraph interrupt/resume + SQLite checkpointer。

特点：
- 不需要启动 uvicorn；直接用 httpx 的 ASGITransport 调用 app。
- 只验证 HITL 的 pause/resume 与任务轮询链路不报错。

运行：
    conda run -n agents python test/verify_hitl_flow.py

可选环境变量：
- LANGGRAPH_SQLITE_PATH: SQLite checkpoint 文件路径（默认会用 /tmp 下的临时文件）
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

import httpx


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _poll_status(client: httpx.AsyncClient, task_id: str, timeout_s: float = 15.0) -> dict:
    deadline = time.time() + timeout_s
    last: dict | None = None

    while time.time() < deadline:
        resp = await client.get(f"/chat/status/{task_id}")
        resp.raise_for_status()
        last = resp.json()
        if last.get("status") in ("completed", "failed"):
            return last
        await asyncio.sleep(0.1)

    raise TimeoutError(f"Polling timeout for task_id={task_id}. last={last}")


async def main() -> None:
    # Make repo root importable (so `import backend` works when running from test/).
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    # Use a temp SQLite file to avoid clobbering local dev data.
    default_db = Path("/tmp") / f"langgraph_checkpoints_verify_{uuid.uuid4().hex}.sqlite"
    os.environ.setdefault("LANGGRAPH_SQLITE_PATH", str(default_db))

    # Ensure parent exists + clean file (best-effort)
    db_path = Path(os.environ["LANGGRAPH_SQLITE_PATH"]).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        db_path.unlink(missing_ok=True)
    except Exception:
        pass

    # Import after env is set so backend picks up LANGGRAPH_SQLITE_PATH.
    from backend.main import app

    thread_id = f"verify_{_now_ms()}_{uuid.uuid4().hex[:6]}"

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1) Start chat task -> should interrupt and ask for customer form.
        start = await client.post(
            "/chat",
            json={"message": "Plan me a trip.", "thread_id": thread_id, "is_continuation": False},
        )
        start.raise_for_status()
        task_id = start.json()["task_id"]

        status1 = await _poll_status(client, task_id)
        assert status1.get("status") == "completed", status1
        assert status1.get("form_to_display") == "customer_info", status1
        assert status1.get("result", {}).get("reply"), status1

        # 2) Resume with customer_info payload -> should complete.
        resume_payload = {
            "name": "Test User",
            "email": "test@example.com",
            "phone": "+1234567890",
            "budget": "2000",
        }

        resume = await client.post(
            "/chat/resume",
            json={"thread_id": thread_id, "resume": resume_payload},
        )
        resume.raise_for_status()
        resume_task_id = resume.json()["task_id"]

        status2 = await _poll_status(client, resume_task_id)
        assert status2.get("status") == "completed", status2
        assert status2.get("result", {}).get("reply"), status2

        # 3) Clear thread (should delete SQLite checkpoints for this thread).
        cleared = await client.delete(f"/chat/thread/{thread_id}")
        cleared.raise_for_status()

    print("OK: HITL interrupt/resume flow works")
    print(f"- thread_id: {thread_id}")
    print(f"- sqlite: {db_path}")


if __name__ == "__main__":
    asyncio.run(main())
