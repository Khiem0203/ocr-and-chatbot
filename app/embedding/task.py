from celery.utils.log import get_task_logger

from app.ocr.celery_app import celery_app

from . import chunking, embedding_service, history_db, milvus_store

celery_logger = get_task_logger(__name__)


@celery_app.task(name="cleanup_chat_history_task", bind=True, max_retries=0)
def cleanup_chat_history_task(self):
    deleted = history_db.cleanup_expired_sessions()
    celery_logger.info(f"cleanup_chat_history_task: removed {deleted} expired message(s)")
    return {"deleted": deleted}


@celery_app.task(name="chunk_embed_task", bind=True, max_retries=3)
def chunk_embed_task(self, result: dict):
    bucket = result.get("bucket")
    object_key = result.get("object_key")
    status = result.get("status")
    corrected_text = result.get("corrected_text", "")

    if not bucket or not object_key:
        return result

    if status != "success":
        return result

    try:
        chunks = chunking.split_text(corrected_text)
        if not chunks:
            milvus_store.delete_document_chunks(bucket, object_key)
            celery_logger.info(f"chunk_embed_task: {bucket}/{object_key} no chunks, cleared vectors")
            return result

        texts = [c["text"] for c in chunks]
        embeddings = embedding_service.embed_texts(texts)
        milvus_store.upsert_chunks(bucket, object_key, chunks, embeddings)
        celery_logger.info(f"chunk_embed_task: {bucket}/{object_key} {len(chunks)} chunks embedded")
    except Exception as e:
        celery_logger.error(f"chunk_embed_task failed for {bucket}/{object_key}: {e}", exc_info=True)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=10)

    return result
