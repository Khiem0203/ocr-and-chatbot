#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
STOP_TIMEOUT="${STOP_TIMEOUT:-60}"

if [ ! -d /proc/self ]; then
    echo "Error: stop.sh needs a Linux /proc filesystem." >&2
    exit 1
fi

_belongs_to_project() {
    local pid=$1
    [ "$(readlink "/proc/$pid/cwd" 2>/dev/null)" = "$PROJECT_DIR" ] && return 0
    tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | grep -qF "$PROJECT_DIR/" && return 0
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -qxF "CHATBOT_RAG_HOME=$PROJECT_DIR" && return 0
    return 1
}

_project_pids() {
    local pid
    for pid in $(pgrep -f "$1" 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        _belongs_to_project "$pid" && echo "$pid"
    done
}

_top_level_pids() {
    local pids=$1 pid ppid
    for pid in $pids; do
        ppid=$(awk '/^PPid:/ {print $2}' "/proc/$pid/status" 2>/dev/null)
        case " $(echo $pids) " in
            *" $ppid "*) continue ;;
        esac
        echo "$pid"
    done
}

stop_group() {
    local label=$1 pattern=$2
    local pids top waited=0 remaining

    pids=$(_project_pids "$pattern")
    if [ -z "$pids" ]; then
        echo "  [skip] $label: not running"
        return
    fi

    top=$(_top_level_pids "$pids" | tr '\n' ' ')
    echo "  [stop] $label: SIGTERM to $top"
    kill -TERM $top 2>/dev/null

    while [ -n "$(_project_pids "$pattern")" ] && [ "$waited" -lt "$STOP_TIMEOUT" ]; do
        sleep 1
        waited=$((waited + 1))
    done

    remaining=$(_project_pids "$pattern" | tr '\n' ' ')
    if [ -n "$remaining" ]; then
        echo "  [kill] $label: still running after ${STOP_TIMEOUT}s, SIGKILL to $remaining"
        kill -KILL $remaining 2>/dev/null
        sleep 1
    else
        echo "  [done] $label stopped"
    fi
}

echo "========================================"
echo "  Stopping chatbot-rag services"
echo "  Project: $PROJECT_DIR"
echo "========================================"

stop_group "Celery Beat"    'app\.ocr\.celery_app beat|celery beat'
stop_group "OCR and Chatbot APIs" 'uvicorn app\.(ocr|embedding)\.main:app'
stop_group "Celery workers" 'app\.ocr\.celery_app worker|celeryd:'

echo "--------------------------------"
echo "Application services stopped."
echo "A worker killed after the timeout has its unfinished task redelivered (task_acks_late is enabled)."
echo ""
echo "Not stopped by this script:"
echo "  Infrastructure containers:  docker compose stop"
echo "  Model servers:              sudo systemctl stop vllm-chandra llm-chatbot"
echo "--------------------------------"
