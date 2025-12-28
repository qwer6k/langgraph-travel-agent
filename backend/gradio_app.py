import os
import time
import uuid
from typing import Any, Dict, List, Tuple

import gradio as gr
import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
POLL_INTERVAL_S = float(os.getenv("POLL_INTERVAL_S", "1.0"))
POLL_TIMEOUT_S = float(os.getenv("POLL_TIMEOUT_S", "180"))

ChatHistory = List[Tuple[str, str]]


def _new_thread_id() -> str:
    return f"session_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


def _poll_task(client: httpx.Client, task_id: str) -> Dict[str, Any]:
    deadline = time.time() + POLL_TIMEOUT_S
    last_payload: Dict[str, Any] | None = None

    while time.time() < deadline:
        resp = client.get(f"/chat/status/{task_id}")
        resp.raise_for_status()
        payload = resp.json()
        last_payload = payload

        status = payload.get("status")
        if status in ("completed", "failed"):
            return payload

        time.sleep(POLL_INTERVAL_S)

    return {
        "status": "failed",
        "result": {"error": f"Polling timeout after {POLL_TIMEOUT_S:.0f}s"},
        "debug": last_payload,
    }


def _chat_once(message: str, thread_id: str, is_continuation: bool) -> Dict[str, Any]:
    with httpx.Client(base_url=BACKEND_URL, timeout=60.0) as client:
        resp = client.post(
            "/chat",
            json={
                "message": message,
                "thread_id": thread_id,
                "is_continuation": is_continuation,
            },
        )
        resp.raise_for_status()
        task_id = resp.json()["task_id"]
        return _poll_task(client, task_id)


def _submit_customer_info(thread_id: str, customer_info: Dict[str, Any]) -> None:
    with httpx.Client(base_url=BACKEND_URL, timeout=30.0) as client:
        resp = client.post(
            "/chat/customer-info",
            json={"thread_id": thread_id, "customer_info": customer_info},
        )
        resp.raise_for_status()


def _resume(thread_id: str, resume: Dict[str, Any]) -> Dict[str, Any]:
    with httpx.Client(base_url=BACKEND_URL, timeout=60.0) as client:
        resp = client.post(
            "/chat/resume",
            json={"thread_id": thread_id, "resume": resume},
        )
        resp.raise_for_status()
        task_id = resp.json()["task_id"]
        return _poll_task(client, task_id)


def _clear_thread(thread_id: str) -> None:
    with httpx.Client(base_url=BACKEND_URL, timeout=30.0) as client:
        resp = client.delete(f"/chat/thread/{thread_id}")
        resp.raise_for_status()


def on_send(
    user_message: str,
    history: ChatHistory,
    thread_id: str,
    pending_user_message: str,
    needs_customer_info: bool,
):
    user_message = (user_message or "").strip()
    history = history or []

    if not user_message:
        return history, "", thread_id, pending_user_message, needs_customer_info, gr.update()

    if needs_customer_info:
        # 不额外发起请求，避免状态混乱
        history = history + [(user_message, "请先提交下方客户信息表单，然后我会继续处理上一条需求。")]
        return history, "", thread_id, pending_user_message, needs_customer_info, gr.update()

    try:
        payload = _chat_once(user_message, thread_id, is_continuation=False)
    except Exception as e:
        history = history + [(user_message, f"后端请求失败：{e}")]
        return history, "", thread_id, pending_user_message, needs_customer_info, gr.update()

    status = payload.get("status")
    form_to_display = payload.get("form_to_display")
    result = payload.get("result") or {}

    if status == "failed":
        history = history + [(user_message, f"任务失败：{result.get('error', 'unknown error')}")]
        return history, "", thread_id, pending_user_message, needs_customer_info, gr.update()

    reply = (result.get("reply") or "").strip() or "（无回复）"
    history = history + [(user_message, reply)]

    if form_to_display == "customer_info":
        return history, "", thread_id, user_message, True, gr.update(visible=True)

    return history, "", thread_id, "", False, gr.update(visible=False)


def on_submit_customer_info(
    name: str,
    email: str,
    phone: str,
    budget: str,
    history: ChatHistory,
    thread_id: str,
    pending_user_message: str,
):
    history = history or []
    name = (name or "").strip()
    email = (email or "").strip().lower()
    phone = (phone or "").strip().replace(" ", "")
    budget = (budget or "").strip().replace("$", "").replace(",", "")

    customer_info = {"name": name, "email": email, "phone": phone, "budget": budget}

    try:
        payload = _resume(thread_id, customer_info)
    except Exception as e:
        history = history + [("", f"提交/续聊失败：{e}")]
        return history, thread_id, pending_user_message, True, gr.update(visible=True)

    status = payload.get("status")
    result = payload.get("result") or {}

    if status == "failed":
        history = history + [("", f"任务失败：{result.get('error', 'unknown error')}")]
        return history, thread_id, pending_user_message, True, gr.update(visible=True)

    reply = (result.get("reply") or "").strip() or "（无回复）"

    # 由于 resume 不需要重发用户消息，这里用空 user 字段展示 bot 回复
    history = history + [("", reply)]
    return history, thread_id, "", False, gr.update(visible=False)


def on_reset():
    thread_id = _new_thread_id()
    # 尝试清理旧 thread（失败也不影响新会话）
    return [], thread_id, "", False, gr.update(visible=False)


with gr.Blocks(title="Travel Agent (Minimal Gradio)") as demo:
    gr.Markdown("### Travel Agent（最小 Gradio 替代前端）")
    gr.Markdown(f"后端：`{BACKEND_URL}`")

    thread_id_state = gr.State(_new_thread_id())
    pending_user_message_state = gr.State("")
    needs_customer_info_state = gr.State(False)

    chatbot = gr.Chatbot(height=420)
    user_box = gr.Textbox(label="你的需求", placeholder="例如：我想下周从上海去东京，3天，预算2000美金")
    send_btn = gr.Button("发送", variant="primary")

    customer_group = gr.Group(visible=False)
    with customer_group:
        gr.Markdown("#### 请先填写客户信息（后端要求 HITL）")
        name = gr.Textbox(label="Name")
        email = gr.Textbox(label="Email")
        phone = gr.Textbox(label="Phone (+country code)")
        budget = gr.Textbox(label="Budget (number)")
        submit_btn = gr.Button("提交信息并继续", variant="primary")

    reset_btn = gr.Button("重置会话（新 thread）")

    send_btn.click(
        on_send,
        inputs=[user_box, chatbot, thread_id_state, pending_user_message_state, needs_customer_info_state],
        outputs=[chatbot, user_box, thread_id_state, pending_user_message_state, needs_customer_info_state, customer_group],
    )

    submit_btn.click(
        on_submit_customer_info,
        inputs=[name, email, phone, budget, chatbot, thread_id_state, pending_user_message_state],
        outputs=[chatbot, thread_id_state, pending_user_message_state, needs_customer_info_state, customer_group],
    )

    reset_btn.click(
        on_reset,
        inputs=[],
        outputs=[chatbot, thread_id_state, pending_user_message_state, needs_customer_info_state, customer_group],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)