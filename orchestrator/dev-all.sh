#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

_cfg_py="$(awk -F': *' '/^python_executable:/{v=$2; sub(/[[:space:]]+$/,"",v); gsub(/"/,"",v); print v; exit}' unified_config.yaml 2>/dev/null)"
PY="${PY:-${_cfg_py:-.venv/bin/python}}"
if [ ! -x "$PY" ]; then
  echo "[dev-all] 警告：解释器 '$PY' 不存在或不可执行；请检查 unified_config.yaml 的 python_executable。"
fi
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-0}"
DIST_DIR="frontend/dist"
DIST_ASSETS="$DIST_DIR/assets"
FE_DIR="$(pwd)/frontend"

free_port() {
  local port="$1" pids
  command -v lsof >/dev/null 2>&1 || return 0
  pids="$(lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  [ -z "$pids" ] && return 0
  echo "[dev-all] 端口 $port 被占用（PID: $(echo "$pids" | tr '\n' ' ')），正在释放…"
  kill $pids 2>/dev/null || true
  sleep 1
  pids="$(lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  [ -n "$pids" ] && { kill -9 $pids 2>/dev/null || true; sleep 0.5; }
  echo "[dev-all] 端口 $port 已释放。"
}

if ! "$PY" -c "import watchfiles" >/dev/null 2>&1; then
  echo "[dev-all] 提示：未检测到 watchfiles，后端 --reload 可能无法工作。"
  echo "[dev-all]   安装：$PY -m pip install watchfiles"
fi

pids=()
cleanup() {
  echo
  echo "[dev-all] 正在停止前后端…"
  for pid in "${pids[@]:-}"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  done
  pkill -f "$FE_DIR/node_modules" 2>/dev/null || true
  wait 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

rm -rf "$DIST_DIR"

echo "[dev-all] 前端：cd frontend && npm run build -- --watch"
( cd frontend && exec npm run build -- --watch ) &
pids+=($!)
fe_pid="${pids[0]}"

echo "[dev-all] 等待前端首次构建产出 $DIST_DIR …"
ready=0
for _ in $(seq 1 240); do
  if [ -d "$DIST_ASSETS" ] && [ -f "$DIST_DIR/index.html" ]; then
    ready=1
    echo "[dev-all] 前端已就绪。"
    break
  fi
  if ! kill -0 "$fe_pid" 2>/dev/null; then
    echo "[dev-all] 前端构建进程已退出，请检查上方构建错误。"
    cleanup
  fi
  sleep 0.5
done
if [ "$ready" -ne 1 ]; then
  echo "[dev-all] 等待超时：$DIST_ASSETS 仍不存在；为避免后端崩溃，已中止。"
  cleanup
fi

free_port "$PORT"
if [ "$RELOAD" = "0" ]; then
  echo "[dev-all] 后端稳定模式：$PY -m uvicorn backend.app:app --port $PORT"
  "$PY" -m uvicorn backend.app:app --port "$PORT" &
else
  echo "[dev-all] 后端：$PY -m uvicorn backend.app:app --reload --reload-dir backend --port $PORT"
  "$PY" -m uvicorn backend.app:app --reload --reload-dir backend --port "$PORT" &
fi
be_pid=$!
pids+=("$be_pid")

echo "[dev-all] 等待后端 http://127.0.0.1:$PORT/api/health （最近任务同步恢复，其余后台加载；最多等 ~120s）…"
be_ready=0
for _ in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
    be_ready=1
    break
  fi
  if ! kill -0 "$be_pid" 2>/dev/null; then
    echo "[dev-all] 后端进程已退出，请检查上方 uvicorn 报错。"
    cleanup
  fi
  sleep 0.5
done
if [ "$be_ready" -ne 1 ]; then
  echo "[dev-all] 等待后端超时（${PORT} 无响应）。"
  cleanup
fi

if [ "$RELOAD" = "0" ]; then
  echo "[dev-all] 已启动（前端 watch + 后端稳定模式）。打开 http://127.0.0.1:$PORT ，Ctrl+C 退出。"
else
  echo "[dev-all] 已启动（前端 watch + 后端 reload）。打开 http://127.0.0.1:$PORT ，Ctrl+C 退出。"
fi
wait
