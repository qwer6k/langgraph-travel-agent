#!/bin/bash
set -euo pipefail

HOST="${HOST:-http://127.0.0.1:8000}"
TIMEOUT="${TIMEOUT:-90}"

echo "============================================================"
echo "PR2 Smoke Tests (robust)"
echo "HOST=$HOST  TIMEOUT=${TIMEOUT}s"
echo "============================================================"

need_jq() {
  command -v jq >/dev/null 2>&1 || { echo "❌ jq not found"; exit 1; }
}

post_json() {
  local url="$1"
  local data="$2"
  # 输出：body + httpcode(最后一行)
  curl -sS -X POST "$url" -H "Content-Type: application/json" -d "$data" -w "\n%{http_code}\n"
}

get_json() {
  local url="$1"
  curl -sS "$url"
}

start_chat() {
  local thread_id="$1"
  local message="$2"
  local resp http body
  resp="$(post_json "$HOST/chat" "{\"thread_id\":\"$thread_id\",\"message\":\"$message\"}")"
  http="$(echo "$resp" | tail -n 1)"
  body="$(echo "$resp" | sed '$d')"

  # debug 打到 stderr，别污染 stdout
  echo "$body" | jq . >&2

  if [[ "$http" != "200" ]]; then
    echo "❌ /chat HTTP=$http" >&2
    echo "$body" >&2
    exit 1
  fi

  # stdout 只返回 task_id
  echo "$body" | jq -r '.task_id'
}



poll_status() {
  local task_id="$1"
  local start now elapsed status
  start="$(date +%s)"

  while true; do
    now="$(date +%s)"
    elapsed=$((now - start))
    if [[ "$elapsed" -ge "$TIMEOUT" ]]; then
      echo "❌ timeout after ${TIMEOUT}s" >&2
      return 2
    fi

    status="$(get_json "$HOST/chat/status/$task_id")"

    # debug 打到 stderr
    echo "$status" | jq . >&2

    if echo "$status" | jq -e '.status == "completed" or .status == "complete" or .status == "failed"' >/dev/null; then
      # stdout 只输出一次 raw JSON，给外面 capture
      echo "$status"
      return 0
    fi
    sleep 1
  done
}


extract_form_to_display() {
  local status_json="$1"
  # 兼容 form_to_display 在顶层或 result 里
  echo "$status_json" | jq -r '.form_to_display // .result.form_to_display // ""'
}

extract_reply() {
  local status_json="$1"
  echo "$status_json" | jq -r '.result.reply // .reply // ""'
}

save_customer_info() {
  local thread_id="$1"
  local resp http body
  resp="$(post_json "$HOST/chat/customer-info" \
    "{\"thread_id\":\"$thread_id\",\"customer_info\":{\"name\":\"Test\",\"email\":\"test@example.com\",\"budget\":\"600\"}}")"
  http="$(echo "$resp" | tail -n 1)"
  body="$(echo "$resp" | sed '$d')"
  echo "$body" | jq . || true
  if [[ "$http" != "200" ]]; then
    echo "❌ /chat/customer-info HTTP=$http"
    echo "$body"
    exit 1
  fi
}

need_jq

# -----------------------
# CASE 1: customer_info gate
# -----------------------
echo -e "\n=== CASE 1: customer_info gate ==="
THREAD_ID="session_pr2_case1_$(date +%s)"
TASK_ID="$(start_chat "$THREAD_ID" "Plan me a trip to Tokyo")"
STATUS_JSON="$(poll_status "$TASK_ID")" || exit 1
FORM="$(extract_form_to_display "$STATUS_JSON")"

if [[ "$FORM" != "customer_info" ]]; then
  echo "❌ Expected form_to_display=customer_info, got: '$FORM'"
  echo "➡️ 你需要检查：是否真的跑到 PR2 的 call_model_node？是否重启后端？thread_id 是否复用？"
  exit 2
fi
echo "✅ CASE1 OK (form_to_display=customer_info)"

# -----------------------
# CASE 2: missing dates should ASK (no default +15d)
# -----------------------
echo -e "\n=== CASE 2: missing dates should ASK ==="
THREAD_ID="session_pr2_case2_$(date +%s)"
save_customer_info "$THREAD_ID"
TASK_ID="$(start_chat "$THREAD_ID" "Hong Kong to Tokyo, one-way business class, 5 adults")"
STATUS_JSON="$(poll_status "$TASK_ID")" || exit 1
REPLY="$(extract_reply "$STATUS_JSON")"
echo "REPLY: $REPLY"

# 只要包含 ask dates 的关键句就算通过（不要过度依赖 exact 文案）
if [[ "$REPLY" != *"travel dates"* && "$REPLY" != *"departure date"* && "$REPLY" != *"trip duration"* ]]; then
  echo "❌ Expected an ASK for dates/duration, but reply doesn't look like it."
  exit 3
fi
echo "✅ CASE2 OK (asks for dates/duration)"

# -----------------------
# CASE 3: activities_only should NOT ask dates
# -----------------------
echo -e "\n=== CASE 3: activities_only should not ask dates ==="
THREAD_ID="session_pr2_case3_$(date +%s)"
save_customer_info "$THREAD_ID"
TASK_ID="$(start_chat "$THREAD_ID" "Recommend activities in Tokyo")"
STATUS_JSON="$(poll_status "$TASK_ID")" || exit 1
REPLY="$(extract_reply "$STATUS_JSON")"
echo "REPLY: $REPLY"

if [[ "$REPLY" == *"travel dates"* || "$REPLY" == *"departure date"* ]]; then
  echo "❌ activities_only unexpectedly asked for dates"
  exit 4
fi
echo "✅ CASE3 OK (no date ask for activities_only)"

echo -e "\n🎉 PR2 smoke tests passed."
