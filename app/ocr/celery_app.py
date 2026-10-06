from celery import Celery
from celery.signals import worker_ready
from kombu import Exchange, Queue
import os
import logging
import traceback

from dotenv import load_dotenv
load_dotenv()

_log = logging.getLogger(__name__)

_STARTUP_LOG = os.path.join(
    os.getenv("PROJECT_DIR", "/path/to/chatbot-rag"),
    "logs", "startup_connections.log",
)


def _startup_log(msg: str):
    _log.info(msg)
    try:
        os.makedirs(os.path.dirname(_STARTUP_LOG), exist_ok=True)
        if os.path.exists(_STARTUP_LOG) and os.path.getsize(_STARTUP_LOG) > 5 * 1024 * 1024:
            open(_STARTUP_LOG, "w").close()
        with open(_STARTUP_LOG, "a", encoding="utf-8") as f:
            from datetime import datetime
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except Exception:
        pass


@worker_ready.connect
def _check_connections(sender, **kwargs):
    import psycopg2
    import pika
    from urllib.parse import urlparse

    worker_name = getattr(sender, "hostname", "unknown")
    _startup_log("=" * 60)
    _startup_log(f"  WORKER STARTUP — {worker_name}")
    _startup_log("=" * 60)

    backend_url = os.getenv("CELERY_POSTGRES", "")
    url = backend_url.replace("db+postgresql://", "postgresql://", 1).split("?")[0]
    p = urlparse(url)
    pg_host = p.hostname or "localhost"
    pg_port = p.port or 5432
    pg_db   = p.path.lstrip("/")
    try:
        conn = psycopg2.connect(
            host=pg_host, port=pg_port, database=pg_db,
            user=p.username, password=p.password, connect_timeout=5,
        )
        conn.close()
        _startup_log(f"  [OK]   PostgreSQL (celery_results)  {pg_host}:{pg_port}/{pg_db}")
    except Exception:
        _startup_log(f"  [FAIL] PostgreSQL (celery_results)  {pg_host}:{pg_port}/{pg_db}")
        for line in traceback.format_exc().splitlines():
            _startup_log(f"         {line}")

    ocrdb_host = os.getenv("POSTGRES_HOST", "192.0.2.10")
    ocrdb_port = int(os.getenv("POSTGRES_PORT", "5432"))
    ocrdb_name   = os.getenv("POSTGRES_DB", "ocr_db")
    ocrdb_user = os.getenv("POSTGRES_USER", "your_db_user")
    ocrdb_pw   = os.getenv("POSTGRES_PW", "your_db_password")
    try:
        conn = psycopg2.connect(
            host=ocrdb_host, port=ocrdb_port, database=ocrdb_name,
            user=ocrdb_user, password=ocrdb_pw, connect_timeout=5,
        )
        conn.close()
        _startup_log(f"  [OK]   PostgreSQL (ocrdb remote)     {ocrdb_host}:{ocrdb_port}/{ocrdb_name}")
    except Exception:
        _startup_log(f"  [FAIL] PostgreSQL (ocrdb remote)     {ocrdb_host}:{ocrdb_port}/{ocrdb_name}")
        for line in traceback.format_exc().splitlines():
            _startup_log(f"         {line}")

    broker_url = os.getenv("CELERY_RABBITMQ", "")
    b = urlparse(broker_url)
    rmq_host = b.hostname or "localhost"
    rmq_port = b.port or 5672
    pika_logger = logging.getLogger("pika")
    pika_level  = pika_logger.level
    pika_logger.setLevel(logging.CRITICAL)
    try:
        creds = pika.PlainCredentials(b.username or "your_rabbit_user", b.password or "your_rabbit_password")
        conn = pika.BlockingConnection(
            pika.ConnectionParameters(host=rmq_host, port=rmq_port,
                                      credentials=creds, socket_timeout=5)
        )
        conn.close()
        _startup_log(f"  [OK]   RabbitMQ                    {rmq_host}:{rmq_port}")
    except Exception:
        _startup_log(f"  [FAIL] RabbitMQ                    {rmq_host}:{rmq_port}")
        for line in traceback.format_exc().splitlines():
            _startup_log(f"         {line}")
    finally:
        pika_logger.setLevel(pika_level)

    _startup_log("=" * 60)


CELERY_RABBITMQ = os.getenv("CELERY_RABBITMQ")
CELERY_POSTGRES = os.getenv("CELERY_POSTGRES")
if not CELERY_RABBITMQ or not CELERY_POSTGRES:
    raise ValueError("CELERY_RABBITMQ and CELERY_POSTGRES must be set.")

celery_app = Celery(
    "worker",
    broker=CELERY_RABBITMQ,
    backend=CELERY_POSTGRES,
    include=['app.ocr.task', 'app.ocr.sync_task', 'app.embedding.task']
)

prepare_exchange          = Exchange('prepare_exchange',          type='direct')
process_exchange          = Exchange('process_exchange',          type='direct')
webhook_exchange          = Exchange('webhook_exchange',          type='direct')
db_exchange               = Exchange('db_exchange',               type='direct')
sync_exchange             = Exchange('sync_exchange',             type='direct')
embed_exchange            = Exchange('embed_exchange',            type='direct')


celery_app.conf.task_queues = (
    Queue('prepare_queue',          prepare_exchange,          routing_key='prepare_task'),
    Queue('process_queue',          process_exchange,          routing_key='process_task'),
    Queue('webhook_queue',          webhook_exchange,          routing_key='webhook_task'),
    Queue('db_queue',               db_exchange,               routing_key='db_task'),
    Queue('sync_queue',             sync_exchange,             routing_key='sync_task'),
    Queue('embed_queue',            embed_exchange,            routing_key='embed_task'),
)

celery_app.conf.task_routes = {
    'prepare_ocr_task':              {'queue': 'prepare_queue',          'routing_key': 'prepare_task'},
    'ocr_task':                      {'queue': 'process_queue',          'routing_key': 'process_task'},
    'save_result_task':              {'queue': 'db_queue',               'routing_key': 'db_task'},
    'sync_minio_task':               {'queue': 'sync_queue',             'routing_key': 'sync_task'},
    'chunk_embed_task':              {'queue': 'embed_queue',            'routing_key': 'embed_task'},
    'cleanup_chat_history_task':     {'queue': 'db_queue',               'routing_key': 'db_task'},
}

MINIO_SYNC_INTERVAL_SECONDS = int(os.getenv("MINIO_SYNC_INTERVAL_SECONDS", "300"))
CHAT_HISTORY_CLEANUP_INTERVAL_SECONDS = int(os.getenv("CHAT_HISTORY_CLEANUP_INTERVAL_SECONDS", "1800"))

celery_app.conf.beat_schedule = {
    'sync-minio-periodic': {
        'task': 'sync_minio_task',
        'schedule': MINIO_SYNC_INTERVAL_SECONDS,
        'options': {'queue': 'sync_queue'},
    },
    'cleanup-chat-history-periodic': {
        'task': 'cleanup_chat_history_task',
        'schedule': CHAT_HISTORY_CLEANUP_INTERVAL_SECONDS,
        'options': {'queue': 'db_queue'},
    },
}

celery_app.conf.update(
    task_track_started=True,
    broker_connection_retry_on_startup=True,
    database_engine_options={'pool_recycle': 3600},

    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)