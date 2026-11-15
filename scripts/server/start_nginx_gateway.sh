#!/bin/bash
# Copyright 2026 OPPO
#
# Modified by OPPO for this release.
# OPPO copyright applies only to the modifications.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ============================================================
# mobile-gym 生产环境一键启动 (Nginx + API backend)
#
# 架构:
#   Nginx (:PORT) ──┬── 静态文件 (dist/)  ← sendfile 零拷贝
#                   └── /api/gw/* → Python (:PORT+1)
#
# 使用:
#   ./scripts/server/start_nginx_gateway.sh          # 默认 4180
#   ./scripts/server/start_nginx_gateway.sh 4185     # 自定义端口
#   ./scripts/server/start_nginx_gateway.sh stop     # 停止默认端口
#   ./scripts/server/start_nginx_gateway.sh stop 4185
# ============================================================

ACTION=start
if [ "${1:-}" = "stop" ]; then
    ACTION=stop
    PORT=${2:-4180}
else
    PORT=${1:-4180}
fi
case "$PORT" in
    ''|*[!0-9]*)
        echo "[error] Port must be numeric: $PORT" >&2
        exit 2
        ;;
esac
API_PORT=$((PORT + 1))
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR" || exit 1
STATIC_DIR="${MOBILEGYM_STATIC_DIR:-$ROOT_DIR/dist}"
if [ "$ACTION" = "start" ] && [ ! -f "$STATIC_DIR/index.html" ]; then
    echo "[error] Missing frontend build: $STATIC_DIR/index.html" >&2
    exit 2
fi

# Uvicorn imports ``scripts.server.api_gateway`` again in every spawned
# worker.  Executing api_gateway.py by path alone does not reliably retain the
# repository root on sys.path in those workers.
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

find_bin() {
    local names=("$@")
    # 1. Try command -v for each candidate name
    for name in "${names[@]}"; do
        local found
        found="$(command -v "$name" 2>/dev/null)"
        if [ -n "$found" ]; then echo "$found"; return; fi
    done
    # 2. Image/system sbin directories are not always present in task PATH.
    for name in "${names[@]}"; do
        local system_dir
        for system_dir in /starfire/sbin /usr/local/sbin /usr/sbin; do
            [ -x "$system_dir/$name" ] && { echo "$system_dir/$name"; return; }
        done
    done
    # 3. conda / well-known fallback
    for name in "${names[@]}"; do
        local dir
        for dir in \
            "${CONDA_PREFIX:-}" \
            "${CONDA_EXE%/bin/conda}" \
            "$HOME/miniconda3" \
            "$HOME/anaconda3" \
            "$HOME/miniforge3"; do
            [ -n "$dir" ] && [ -x "$dir/bin/$name" ] && { echo "$dir/bin/$name"; return; }
        done
    done
    echo "[error] Cannot find '${names[*]}'. Install it or set ${names[0]^^}_BIN." >&2
    exit 1
}

NGINX_BIN="${NGINX_BIN:-}"
if [ -z "$NGINX_BIN" ]; then
    NGINX_BIN="$(find_bin nginx)" || exit 1
fi
NGINX_DIR="$ROOT_DIR/.nginx"
NGINX_SOURCE_CONFIG="$ROOT_DIR/.nginx/nginx.source.conf"
# Runtime state must be container-local.  The repository can be mounted into
# multiple task containers at once; PID files under ROOT_DIR can therefore
# refer to an unrelated process in another PID namespace (or a newly reused
# PID in this namespace).
RUNTIME_BASE="${MOBILEGYM_GATEWAY_RUNTIME_BASE:-/tmp}"
RUNTIME_DIR="${MOBILEGYM_GATEWAY_RUNTIME_DIR:-$RUNTIME_BASE/mobilegym-gateway-$(id -u)-$PORT}"
GATEWAY_LOG_DIR="${MOBILEGYM_GATEWAY_LOG_DIR:-$NGINX_DIR/logs}"
NGINX_PREFIX="$RUNTIME_DIR/"
NGINX_RUN_CONFIG="$RUNTIME_DIR/nginx.run.conf"
PIDFILE_NGINX="$RUNTIME_DIR/nginx.pid"
PIDFILE_API="$RUNTIME_DIR/api_gateway.pid"

mkdir -p "$RUNTIME_DIR" "$RUNTIME_DIR/logs" "$GATEWAY_LOG_DIR" \
    "$RUNTIME_DIR"/temp/{client_body,proxy,fastcgi,uwsgi,scgi}
# Nginx workers drop privileges to ``nobody`` and must be able to traverse the
# runtime directory to reach their per-container temp paths.
chmod 0755 "$RUNTIME_DIR"

escape_sed_replacement() {
    printf '%s' "$1" | sed -e 's/[\/&]/\\&/g'
}

render_nginx_config() {
    local output="$NGINX_RUN_CONFIG"
    if [ ! -f "$NGINX_SOURCE_CONFIG" ]; then
        echo "[error] Missing nginx source config: $NGINX_SOURCE_CONFIG" >&2
        exit 1
    fi

    local esc_nginx_dir esc_nginx_runtime_dir esc_mime_types esc_root esc_static esc_logs
    esc_nginx_dir="$(escape_sed_replacement "$NGINX_DIR")"
    esc_nginx_runtime_dir="$(escape_sed_replacement "$RUNTIME_DIR")"
    esc_mime_types="$(escape_sed_replacement "$MIME_TYPES")"
    esc_root="$(escape_sed_replacement "$ROOT_DIR")"
    esc_static="$(escape_sed_replacement "$STATIC_DIR")"
    esc_logs="$(escape_sed_replacement "$GATEWAY_LOG_DIR")"

    sed \
        -e "s/__NGINX_DIR__/$esc_nginx_dir/g" \
        -e "s/__NGINX_RUNTIME_DIR__/$esc_nginx_runtime_dir/g" \
        -e "s/__NGINX_LOG_DIR__/$esc_logs/g" \
        -e "s/__MIME_TYPES__/$esc_mime_types/g" \
        -e "s/__ROOT__/$esc_root/g" \
        -e "s/__STATIC_ROOT__/$esc_static/g" \
        -e "s/__PORT__/$PORT/g" \
        -e "s/__API_PORT__/$API_PORT/g" \
        "$NGINX_SOURCE_CONFIG" > "$output"
}

read_pidfile() {
    local pidfile="$1"
    local pid
    [ -r "$pidfile" ] || return 1
    pid="$(cat "$pidfile" 2>/dev/null)"
    case "$pid" in
        ''|*[!0-9]*) return 1 ;;
    esac
    printf '%s' "$pid"
}

pid_cmdline() {
    local pid="$1"
    [ -r "/proc/$pid/cmdline" ] || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline"
}

is_our_api_gateway() {
    local pid="$1"
    local cmdline
    cmdline="$(pid_cmdline "$pid")" || return 1
    case "$cmdline" in
        *"$SCRIPT_DIR/api_gateway.py"*"--port $API_PORT"*) return 0 ;;
        *) return 1 ;;
    esac
}

is_our_nginx() {
    local pid="$1"
    local cmdline
    cmdline="$(pid_cmdline "$pid")" || return 1
    case "$cmdline" in
        *nginx*"$NGINX_RUN_CONFIG"*) return 0 ;;
        *) return 1 ;;
    esac
}

wait_for_exit() {
    local pid="$1"
    local attempt
    for attempt in $(seq 1 50); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 0.1
    done
    return 1
}

stop_api_gateway() {
    local pid pgid
    pid="$(read_pidfile "$PIDFILE_API")" || {
        rm -f "$PIDFILE_API"
        return
    }
    if ! is_our_api_gateway "$pid"; then
        echo "  api pidfile stale; refusing to signal pid=$pid"
        rm -f "$PIDFILE_API"
        return
    fi

    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
    if [ "$pgid" = "$pid" ]; then
        kill -- "-$pid" 2>/dev/null || true
    else
        kill "$pid" 2>/dev/null || true
    fi
    if wait_for_exit "$pid"; then
        echo "  api stopped (pid=$pid)"
    else
        echo "  api did not exit within 5s (pid=$pid)" >&2
    fi
    rm -f "$PIDFILE_API"
}

stop_nginx() {
    local pid
    pid="$(read_pidfile "$PIDFILE_NGINX")" || {
        rm -f "$PIDFILE_NGINX"
        return
    }
    if ! is_our_nginx "$pid"; then
        echo "  nginx pidfile stale; refusing to signal pid=$pid"
        rm -f "$PIDFILE_NGINX"
        return
    fi

    kill "$pid" 2>/dev/null || true
    if wait_for_exit "$pid"; then
        echo "  nginx stopped (pid=$pid)"
    else
        echo "  nginx did not exit within 5s (pid=$pid)" >&2
    fi
    rm -f "$PIDFILE_NGINX"
}

stop_servers() {
    echo "[stop] Stopping servers..."
    stop_nginx
    stop_api_gateway
}

if [ "$ACTION" = "stop" ]; then
    stop_servers
    exit 0
fi

# Stop any previous instances
stop_servers

# Create dirs
mkdir -p "$NGINX_DIR"/{logs,ssl}

# Auto-generate self-signed TLS certificate for HTTP/2 (one-time)
if [ ! -f "$NGINX_DIR/ssl/localhost.crt" ]; then
    echo "[setup] Generating self-signed TLS certificate for HTTP/2..."
    if ! openssl req -x509 -nodes -days 3650 -newkey rsa:2048 \
        -keyout "$NGINX_DIR/ssl/localhost.key" \
        -out "$NGINX_DIR/ssl/localhost.crt" \
        -subj '/CN=localhost'; then
        echo "[error] Failed to generate TLS certificate. Is openssl installed?" >&2
        exit 1
    fi
    echo "  cert: $NGINX_DIR/ssl/localhost.crt"
fi

# Generate Nginx config
MIME_TYPES="$NGINX_DIR/mime.types"
if [ ! -f "$MIME_TYPES" ]; then
    for p in \
        "$(dirname "$NGINX_BIN")/../etc/nginx/mime.types" \
        "/starfire/etc/nginx/conf/mime.types" \
        "/etc/nginx/mime.types"; do
        if [ -f "$p" ]; then cp "$p" "$MIME_TYPES"; break; fi
    done
fi
render_nginx_config

# 1. Start API backend (uvicorn multi-worker via conda rllm)
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(find_bin python python3)" || exit 1
fi
API_WORKERS=${API_WORKERS:-8}
echo "[start] API gateway on :${API_PORT} (workers=${API_WORKERS})"
# setsid 在 Linux 上给 API gateway 起独立进程组，便于 stop 时 kill -- "-$pid" 一次清理 uvicorn 全家。
# macOS 默认没有 setsid，退回 nohup（同样能脱离终端 SIGHUP）；stop_servers 里的 `|| kill "$pid"` 会兜底。
if command -v setsid >/dev/null 2>&1; then
    setsid "$PYTHON_BIN" "$SCRIPT_DIR/api_gateway.py" --port "$API_PORT" --workers "$API_WORKERS" \
        >> "$GATEWAY_LOG_DIR/api_gateway.log" 2>&1 &
else
    nohup "$PYTHON_BIN" "$SCRIPT_DIR/api_gateway.py" --port "$API_PORT" --workers "$API_WORKERS" \
        >> "$GATEWAY_LOG_DIR/api_gateway.log" 2>&1 &
fi
echo $! > "$PIDFILE_API"

# 2. Start Nginx
echo "[start] Nginx on :${PORT} (workers=8, bin=${NGINX_BIN})"
if ! "$NGINX_BIN" -p "$NGINX_PREFIX" -c "$NGINX_RUN_CONFIG"; then
    echo "[error] Failed to start Nginx with $NGINX_BIN" >&2
    stop_servers
    exit 1
fi

sleep 0.5
echo ""
echo "✅ mobile-gym serving at https://0.0.0.0:${PORT}  (HTTP/2 + TLS)"
echo "   Static:  Nginx (sendfile + HTTP/2 + 8 workers)"
echo "   API:     uvicorn + starlette (:${API_PORT}, ${API_WORKERS} workers)"
echo ""
echo "   Stop:    $0 stop $PORT"
echo "   Logs:    $GATEWAY_LOG_DIR/{error,access,api_gateway}.log"
echo "   Runtime: $RUNTIME_DIR"
