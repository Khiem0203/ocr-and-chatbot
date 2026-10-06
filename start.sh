#!/bin/bash

ENV_FILE="$(dirname "$0")/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
    echo "Loaded .env from $ENV_FILE"
else
    echo "Warning: .env file not found at $ENV_FILE"
fi

echo "Using CELERY_RABBITMQ: $CELERY_RABBITMQ"
echo "Using CELERY_POSTGRES: $CELERY_POSTGRES"

DEFAULT_PREPARE_WORKERS=2
DEFAULT_PROCESS_WORKERS=1
DEFAULT_WEBHOOK_WORKERS=0
DEFAULT_DB_WORKERS=2
DEFAULT_SYNC_WORKERS=1
DEFAULT_EMBED_WORKERS=1

CPU_CONCURRENCY=1
GPU_CONCURRENCY=1

OCR_GPU_ID="${OCR_GPU_ID:-0}"
CHAT_GPU_ID="${CHAT_GPU_ID:-1}"

VENV_DIR="/path/to/venv"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
export CHATBOT_RAG_HOME="$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR/api"
mkdir -p "$LOG_DIR/chat_api"
mkdir -p "$LOG_DIR/prepare"
mkdir -p "$LOG_DIR/process"
mkdir -p "$LOG_DIR/db"
mkdir -p "$LOG_DIR/sync"
mkdir -p "$LOG_DIR/embed"
mkdir -p "$LOG_DIR/webhook"
mkdir -p "$LOG_DIR/beat"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Error: venv not found at $VENV_DIR. Check VENV_DIR in start.sh."
    exit 1
fi

for i in {1..30}; do
    if pg_isready -h localhost -p 5432 -U your_celery_db_user -d celery_results --timeout=1; then
        echo "PostgreSQL is ready!"
        break
    fi
    echo "PostgreSQL not ready yet, waiting 5 seconds... (Attempt $i/30)"
    sleep 5
done

if ! pg_isready -h localhost -p 5432 -U your_celery_db_user -d celery_results --timeout=1; then
    echo "Error: Timed out waiting for PostgreSQL to start." >&2
    exit 1
fi

source "$VENV_DIR/bin/activate"
if [ $? -ne 0 ]; then echo "Error: Failed to activate venv at '$VENV_DIR'."; exit 1; fi

cd "$PROJECT_DIR" || { echo "Error: Could not change to project directory $PROJECT_DIR"; exit 1; }

start_ocr_api_server() {
    local log_file="$LOG_DIR/api/ocr_api.log"
    echo "Starting OCR API Server on port 9090..."
    nohup "$VENV_DIR/bin/uvicorn" app.ocr.main:app --host 0.0.0.0 --port 9090 > "$log_file" 2>&1 &
    echo "OCR API server started. Log: $log_file. PID: $!"
}

start_chat_api_server() {
    local log_file="$LOG_DIR/chat_api/embedding_api.log"
    echo "Starting Embedding/Chatbot API Server on port 9091..."
    CUDA_VISIBLE_DEVICES="$CHAT_GPU_ID" nohup "$VENV_DIR/bin/uvicorn" app.embedding.main:app --host 0.0.0.0 --port 9091 > "$log_file" 2>&1 &
    echo "Embedding/Chatbot API server started. Log: $log_file. PID: $!"
}

start_celery_beat() {
    local log_file="$LOG_DIR/beat/beat.log"
    echo "Starting Celery Beat (MinIO sync scheduler)..."
    nohup "$VENV_DIR/bin/celery" -A app.ocr.celery_app beat --loglevel=INFO --logfile="$log_file" --detach
    echo "Celery Beat started. Log: $log_file"
}

_get_log_subdir() {
    case "$1" in
        prepare_queue) echo "prepare" ;;
        process_queue) echo "process" ;;
        webhook_queue) echo "webhook" ;;
        db_queue)      echo "db" ;;
        sync_queue)    echo "sync" ;;
        embed_queue)   echo "embed" ;;
        *)             echo "prepare" ;;
    esac
}

_get_gpu_id() {
    case "$1" in
        process_queue) echo "$OCR_GPU_ID" ;;
        embed_queue)   echo "$CHAT_GPU_ID" ;;
        *)             echo "" ;;
    esac
}

start_celery_workers() {
    local queue_name=$1; local num_workers=$2; local concurrency=$3
    local subdir; subdir=$(_get_log_subdir "$queue_name")
    local gpu_id; gpu_id=$(_get_gpu_id "$queue_name")
    local gpu_env=()
    if [ -n "$gpu_id" ]; then gpu_env=(env CUDA_VISIBLE_DEVICES="$gpu_id"); fi
    if [ "$num_workers" -eq 0 ]; then
        echo "Skipping workers for queue '$queue_name' (count is 0)."
        return
    fi
    echo "Starting $num_workers worker(s) for queue: '$queue_name' (Concurrency: $concurrency)"
    for ((i=1; i<=num_workers; i++)); do
        local worker_hostname="${queue_name}_worker_${i}@%h"
        local log_file="$LOG_DIR/$subdir/${queue_name}_worker_${i}.log"
        "${gpu_env[@]}" "$VENV_DIR/bin/celery" -A app.ocr.celery_app worker --loglevel=INFO --queues="$queue_name" --hostname="$worker_hostname" --concurrency="$concurrency" --logfile="$log_file" --detach
        echo "  -> Worker $i for queue '$queue_name' started. Log: $log_file"
    done
}

echo "========================================"
echo "  Starting chatbot-rag Services"
echo "========================================"
echo "This script assumes RabbitMQ, PostgreSQL and Milvus are running (docker compose up -d)."
echo "This script assumes vLLM services (Chandra, RAG) are running (systemctl start vllm-chandra vllm-rag)."

start_ocr_api_server
sleep 3

start_chat_api_server
sleep 3

echo "--- Starting Celery Worker Groups ---"

start_celery_workers "prepare_queue" "$DEFAULT_PREPARE_WORKERS" "$CPU_CONCURRENCY"
start_celery_workers "process_queue" "$DEFAULT_PROCESS_WORKERS" "$GPU_CONCURRENCY"
start_celery_workers "webhook_queue" "$DEFAULT_WEBHOOK_WORKERS" "$CPU_CONCURRENCY"
start_celery_workers "db_queue"      "$DEFAULT_DB_WORKERS"      "$CPU_CONCURRENCY"
start_celery_workers "sync_queue"    "$DEFAULT_SYNC_WORKERS"    "$CPU_CONCURRENCY"
start_celery_workers "embed_queue"   "$DEFAULT_EMBED_WORKERS"   "$GPU_CONCURRENCY"

start_celery_beat

echo ""
echo "All application services have been started."
echo "--------------------------------"
echo "Resources Utilization"
echo "RAM Usage:"
free -h | awk '/^Mem:/ {print "  Used: "$3" / Total: "$2" (Free: "$4")"}'
echo "VRAM Usage (GPU):"
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits | awk -F', ' '{print "  Total: "$1" MiB, Used: "$2" MiB, Free: "$3" MiB"}'
echo "--------------------------------"
echo "GPU assignment: OCR=GPU $OCR_GPU_ID (process_queue), Chatbot=GPU $CHAT_GPU_ID (chat API, embed_queue)"
echo "OCR API:       http://localhost:9090"
echo "Chatbot API:   http://localhost:9091"
echo "--------------------------------"
echo "To monitor workers:"
echo "  celery -A app.ocr.celery_app inspect active"
echo "To see stats:"
echo "  celery -A app.ocr.celery_app status"
echo ""
echo "To stop the application services, run:"
echo "  pkill -f 'celery -A app.ocr.celery_app worker'"
echo "  pkill -f 'celery -A app.ocr.celery_app beat'"
echo "  pkill -f 'uvicorn app.ocr.main:app'"
echo "  pkill -f 'uvicorn app.embedding.main:app'"
echo "--------------------------------"
