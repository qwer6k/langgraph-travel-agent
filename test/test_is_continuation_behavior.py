import os
import sys
import asyncio
import uuid
from pathlib import Path
import httpx

# Make repo importable
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from backend.main import app

async def _poll(client, task_id, timeout=10.0):
    import time
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = await client.get(f"/chat/status/{task_id}")
        r.raise_for_status()
        last = r.json()
        if last.get("status") in ("completed","failed"):
            return last
        await asyncio.sleep(0.1)
    raise TimeoutError(last)

def test_is_continuation_deletes_checkpoint():
    async def _inner():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            thread = f"t_{uuid.uuid4().hex[:6]}"
            # 1) Start -> interrupt for customer_info
            r = await client.post("/chat", json={"message":"I want a trip from Paris to Tokyo.", "thread_id":thread, "is_continuation":False})
            r.raise_for_status()
            st = await _poll(client, r.json()["task_id"])
            assert st.get("form_to_display")=="customer_info"

            # 2) Resume
            r2 = await client.post("/chat/resume", json={"thread_id":thread, "resume":{"name":"A","email":"a@a.com","phone":"+1","budget":"1000"}})
            r2.raise_for_status()
            _ = await _poll(client, r2.json()["task_id"])

            # 3) Now call /chat with is_continuation=True -> should NOT interrupt (customer_info exists)
            r3 = await client.post("/chat", json={"message":"Check my plan","thread_id":thread, "is_continuation":True})
            r3.raise_for_status()
            st3 = await _poll(client, r3.json()["task_id"])
            assert st3.get("form_to_display") is None

            # 4) Call /chat with same thread but is_continuation=False -> should delete checkpoint -> interrupt again
            r4 = await client.post("/chat", json={"message":"Start over","thread_id":thread, "is_continuation":False})
            r4.raise_for_status()
            st4 = await _poll(client, r4.json()["task_id"])
            assert st4.get("form_to_display")=="customer_info"

    asyncio.run(_inner())

if __name__ == '__main__':
    test_is_continuation_deletes_checkpoint()
