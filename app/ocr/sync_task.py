import logging
import os

from celery import chain
from celery.utils.log import get_task_logger
from dotenv import load_dotenv

from app.embedding import milvus_store
from app.embedding.task import chunk_embed_task

from . import db
from .celery_app import celery_app
from .task import ocr_task, prepare_ocr_task, save_result_task
from .utils import list_s3_objects

load_dotenv()

celery_logger = get_task_logger(__name__)

MINIO_SYNC_BUCKET = os.getenv("MINIO_SYNC_BUCKET", "")
MINIO_SYNC_PREFIX = os.getenv("MINIO_SYNC_PREFIX", "")


@celery_app.task(name="sync_minio_task", bind=True, max_retries=0)
def sync_minio_task(self):
    if not MINIO_SYNC_BUCKET:
        celery_logger.warning("sync_minio_task: MINIO_SYNC_BUCKET chưa cấu hình trong .env — bỏ qua.")
        return {"status": "skipped", "reason": "MINIO_SYNC_BUCKET not set"}

    access_key = os.getenv("S3_ACCESS_KEY")
    secret_key = os.getenv("S3_SECRET_KEY")
    endpoint   = os.getenv("S3_ENDPOINT")

    db.ensure_schema()

    try:
        current = list_s3_objects(MINIO_SYNC_BUCKET, MINIO_SYNC_PREFIX, access_key, secret_key, endpoint)
    except Exception as e:
        celery_logger.error(f"sync_minio_task: list_s3_objects failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

    tracked = db.get_tracked_etags(MINIO_SYNC_BUCKET)

    new_or_updated = [k for k, etag in current.items() if tracked.get(k) != etag]
    deleted        = [k for k in tracked if k not in current]

    celery_logger.info(
        f"sync_minio_task: bucket={MINIO_SYNC_BUCKET} total={len(current)} "
        f"new/updated={len(new_or_updated)} deleted={len(deleted)}"
    )

    for key in new_or_updated:
        etag    = current[key]
        s3_path = f"s3://{MINIO_SYNC_BUCKET}/{key}"
        db.mark_processing(MINIO_SYNC_BUCKET, key, etag)
        chain(
            prepare_ocr_task.s(
                s3_path=s3_path, access_key=access_key, secret_key=secret_key, endpoint=endpoint,
            ).set(queue="prepare_queue"),
            ocr_task.s().set(queue="process_queue"),
            save_result_task.s().set(queue="db_queue"),
            chunk_embed_task.s().set(queue="embed_queue"),
        ).apply_async()

    for key in deleted:
        try:
            db.delete_document(MINIO_SYNC_BUCKET, key)
            milvus_store.delete_document_chunks(MINIO_SYNC_BUCKET, key)
            celery_logger.info(f"sync_minio_task: deleted {MINIO_SYNC_BUCKET}/{key} (không còn trên MinIO)")
        except Exception as e:
            celery_logger.error(f"sync_minio_task: delete_document failed for {key}: {e}", exc_info=True)

    return {"status": "ok", "total": len(current), "new_or_updated": len(new_or_updated), "deleted": len(deleted)}
