#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

_cfg_py="$(awk -F': *' '/^python_executable:/{v=$2; sub(/[[:space:]]+$/,"",v); gsub(/"/,"",v); print v; exit}' unified_config.yaml 2>/dev/null)"
PY="${PY:-${_cfg_py:-.venv/bin/python}}"
if [ ! -x "$PY" ]; then
  echo "[dev.sh] 警告：解释器 '$PY' 不存在或不可执行；请检查 unified_config.yaml 的 python_executable。"
fi
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-0}"

free_port() {
  local port="$1" pids
  command -v lsof >/dev/null 2>&1 || return 0
  pids="$(lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  [ -z "$pids" ] && return 0
  echo "[dev.sh] 端口 $port 被占用（PID: $(echo "$pids" | tr '\n' ' ')），正在释放…"
  kill $pids 2>/dev/null || true
  sleep 1
  pids="$(lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  [ -n "$pids" ] && { kill -9 $pids 2>/dev/null || true; sleep 0.5; }
  echo "[dev.sh] 端口 $port 已释放。"
}

if ! "$PY" -c "import watchfiles" >/dev/null 2>&1; then
  echo "[dev.sh] 提示：未检测到 watchfiles，--reload 可能无法工作。"
  echo "[dev.sh] 安装：$PY -m pip install watchfiles"
fi

free_port "$PORT"
if [ "$RELOAD" = "0" ]; then
  echo "[dev.sh] 启动稳定模式：$PY -m uvicorn backend.app:app --port $PORT"
  exec "$PY" -m uvicorn backend.app:app --port "$PORT"
fi

echo "[dev.sh] 启动：$PY -m uvicorn backend.app:app --reload --reload-dir backend --port $PORT"
exec "$PY" -m uvicorn backend.app:app \
  --reload \
  --reload-dir backend \
  --port "$PORT"
