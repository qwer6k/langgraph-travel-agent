#!/bin/bash
set -euo pipefail

HOST="http://127.0.0.1:8000"
THREAD_ID="session_test_$(date +%s)"
TIMEOUT=60

echo "=== 1. 存 customer_info ==="
curl -s -X POST "$HOST/chat/customer-info" \
  -H "Content-Type: application/json" \
  -d "{\"thread_id\": \"$THREAD_ID\", \"customer_info\": {\"name\": \"Test\", \"email\": \"test@example.com\", \"budget\": \"600\"}}" | jq .

echo -e "\n=== 2. 发 chat（拿 task_id）==="
RESP=$(curl -s -X POST "$HOST/chat" \
  -H "Content-Type: application/json" \
  -d "{\"thread_id\": \"$THREAD_ID\", \"message\": \"香港-东京 12月20日 单程 商务舱，5人\"}")
echo "$RESP" | jq .
TASK_ID=$(echo "$RESP" | jq -r '.task_id')
echo "TASK_ID=$TASK_ID"

echo -e "\n=== 3. 轮询状态（直到 completed/failed 或超时）==="
START_TIME=$(date +%s)
STATUS=""
TIMED_OUT=false

while true; do
  CURRENT_TIME=$(date +%s)
  ELAPSED=$((CURRENT_TIME - START_TIME))

  if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "❌ 错误：轮询超时（${TIMEOUT}秒）"
    TIMED_OUT=true
    break
  fi

  STATUS=$(curl -s "$HOST/chat/status/$TASK_ID")
  echo "$STATUS" | jq .

  # ✅ 兼容 completed / complete / failed
  if echo "$STATUS" | jq -e '.status == "completed" or .status == "complete" or .status == "failed"' >/dev/null; then
    break
  fi

  sleep 1
done

echo -e "\n=== 4. 验证 ==="
if [ "$TIMED_OUT" = true ]; then
  echo "❌ 测试因超时中断"
  exit 1
fi

FINAL_STATUS=$(echo "$STATUS" | jq -r '.status')
REPLY=$(echo "$STATUS" | jq -r '.result.reply // ""')

echo "最终状态: $FINAL_STATUS"
echo "耗时: ${ELAPSED}s"

# ✅ 你的当前逻辑里，这段 outage 文案通常意味着：synthesize 没匹配到任何 ToolMessage（key 对不齐 / last_tool_args 没写入 / allowed_tools 不对）
if [[ "$REPLY" == *"temporary technical outage"* ]]; then
  echo "❌ 发现 outage 降级回复：这通常不是供应商真的挂了，而是 tool_results 匹配失败（key 对不齐）"
  echo "回复片段："
  echo "$REPLY" | head -c 400; echo
  exit 2
fi

echo "✅ 未触发 outage 降级（说明 tool_results 至少匹配到了某些 ToolMessage）"
