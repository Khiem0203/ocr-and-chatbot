import logging
import os

import psycopg2
from botocore.exceptions import ClientError
from celery.utils.log import get_task_logger
from datetime import datetime
from dotenv import load_dotenv
from urllib.parse import urlparse

from . import db
from .celery_app import celery_app
from .flow import ocr_flow, preprocess_file
from .utils import TEXT_DOC_EXTS, download_from_s3, extract_document_text, get_pdf_metadata, mask_sensitive_words, parse_s3_path

load_dotenv()

celery_logger = get_task_logger(__name__)

_SUPPORTED_CONTENT_TYPES = {
    "application/pdf",
    "image/jpeg", "image/jpg", "image/png", "image/bmp",
    "image/tiff", "image/webp", "image/gif",
    "text/plain",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
_SUPPORTED_EXTS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"} | TEXT_DOC_EXTS


def _celery_db_connect(connect_timeout: int = 5):
    url = os.getenv("CELERY_POSTGRES")
    url = url.replace("db+postgresql://", "postgresql://", 1)
    url = url.split("?")[0]
    p = urlparse(url)
    return psycopg2.connect(
        host=p.hostname,
        port=p.port or 5432,
        database=p.path.lstrip("/"),
        user=p.username,
        password=p.password,
        connect_timeout=connect_timeout,
    )


def _error_result(document_id, s3_path, request_id, error: str, bucket: str = None, object_key: str = None) -> dict:
    return {
        "document_id": document_id, "s3_path": s3_path, "request_id": request_id,
        "bucket": bucket, "object_key": object_key,
        "status": "error", "error": error,
    }


@celery_app.task(name="prepare_ocr_task", bind=True, max_retries=3)
def prepare_ocr_task(self, s3_path: str, access_key: str, secret_key: str, endpoint: str, document_id: int = None):

    request_id  = self.request.id
    temp_folder = f"./temp_processing/{request_id}"
    celery_logger.info(f"Prepare Task {request_id} for file: {s3_path}")

    try:
        bucket, object_key = parse_s3_path(s3_path)
    except ValueError as e:
        celery_logger.error(f"Prepare Task ({request_id}) failed: {e}")
        return _error_result(document_id, s3_path, request_id, str(e))

    try:
        info = download_from_s3(s3_path, temp_folder, access_key, secret_key, endpoint)
        file_path    = info["file_path"]
        content_type = info["content_type"]
        etag         = info["etag"]
        ext          = os.path.splitext(file_path)[1].lower()

        if content_type not in _SUPPORTED_CONTENT_TYPES and ext not in _SUPPORTED_EXTS:
            raise ValueError(f"Unsupported file type: content_type={content_type!r} ext={ext!r}")

        creator, created_at = "", ""
        if ext == ".pdf":
            meta = get_pdf_metadata(file_path) or {}
            creator = meta.get("/Creator", "") or ""
            created_at_raw = (meta.get("/CreationDate", "") or "").replace("D:", "").replace("'", "")
            created_at_raw = created_at_raw[:14] if len(created_at_raw) >= 14 else ""
            if len(created_at_raw) == 14:
                try:
                    created_at = datetime.strptime(created_at_raw, "%Y%m%d%H%M%S").strftime("%H:%M:%S %d/%m/%Y")
                except ValueError:
                    celery_logger.warning(f"Prepare Task ({request_id}): Invalid CreationDate '{created_at_raw}'")

        if ext in TEXT_DOC_EXTS:
            direct_text = extract_document_text(file_path, temp_folder)
            celery_logger.info(
                f"Prepare Task ({request_id}) done: file_path={file_path} "
                f"direct-extract {len(direct_text)} chars"
            )
            return {
                "file_path":   file_path,
                "direct_text": direct_text,
                "temp_folder": temp_folder,
                "s3_path":     s3_path,
                "bucket":      bucket,
                "object_key":  object_key,
                "etag":        etag,
                "request_id":  request_id,
                "document_id": document_id,
                "creator":     creator,
                "created_at":  created_at,
            }

        page_image_paths = preprocess_file(file_path, temp_folder)
        celery_logger.info(
            f"Prepare Task ({request_id}) done: file_path={file_path} "
            f"creator={creator!r} pages={len(page_image_paths)}"
        )

        return {
            "file_path":        file_path,
            "page_image_paths": page_image_paths,
            "temp_folder":      temp_folder,
            "s3_path":          s3_path,
            "bucket":           bucket,
            "object_key":       object_key,
            "etag":             etag,
            "request_id":       request_id,
            "document_id":      document_id,
            "creator":          creator,
            "created_at":       created_at,
        }

    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchKey":
            celery_logger.error(f"Prepare Task ({request_id}) failed: File not found on S3 at {s3_path}.")
            return _error_result(document_id, s3_path, request_id, f"File not found on S3: {s3_path}", bucket, object_key)
        celery_logger.error(f"Prepare Task ({request_id}) failed with S3 error: {e}", exc_info=True)
        raise self.retry(exc=e, countdown=10)

    except ValueError as e:
        celery_logger.error(f"Prepare Task ({request_id}) failed: {e}")
        return _error_result(document_id, s3_path, request_id, str(e), bucket, object_key)

    except Exception as e:
        celery_logger.error(f"Prepare Task ({request_id}) failed: {e}", exc_info=True)
        if self.request.retries >= self.max_retries:
            return _error_result(document_id, s3_path, request_id, str(e), bucket, object_key)
        raise self.retry(exc=e, countdown=10)


@celery_app.task(name="ocr_task", bind=True, max_retries=0)
def ocr_task(self, context: dict):

    bucket     = context.get("bucket")
    object_key = context.get("object_key")
    etag       = context.get("etag")

    if context.get("status") == "error":
        celery_logger.warning(f"OCR Task skip — prepare stage failed: {context.get('error')}")
        return context

    request_id  = context.get("request_id", self.request.id)
    document_id = context.get("document_id")
    s3_path     = context.get("s3_path")

    if "direct_text" in context:
        direct_text = context.get("direct_text", "")
        masked_text = mask_sensitive_words(direct_text)
        celery_logger.info(f"OCR Task ({request_id}) skip OCR — direct text extraction ({len(direct_text)} chars)")
        return {
            "document_id":    document_id,
            "s3_path":        s3_path,
            "bucket":         bucket,
            "object_key":     object_key,
            "etag":           etag,
            "request_id":     request_id,
            "creator":        context.get("creator", ""),
            "created_at":     context.get("created_at", ""),
            "raw_text":       direct_text,
            "corrected_text": masked_text,
            "status":         "success" if masked_text else "empty",
        }

    try:
        result = ocr_flow(context)
        corrected_text = result.get("corrected_text", "")
        celery_logger.info(f"OCR Task ({request_id}) done: {len(corrected_text)} chars corrected")

        return {
            "document_id":    document_id,
            "s3_path":        s3_path,
            "bucket":         bucket,
            "object_key":     object_key,
            "etag":           etag,
            "request_id":     request_id,
            "creator":        context.get("creator", ""),
            "created_at":     context.get("created_at", ""),
            "raw_text":       result.get("ocr_raw_text", ""),
            "corrected_text": corrected_text,
            "status":         "success" if corrected_text else "empty",
        }
    except Exception as e:
        celery_logger.error(f"OCR Task ({request_id}) failed: {e}", exc_info=True)
        return _error_result(document_id, s3_path, request_id, str(e), bucket, object_key)


@celery_app.task(name="save_result_task", bind=True, max_retries=3)
def save_result_task(self, result: dict):
    bucket     = result.get("bucket")
    object_key = result.get("object_key")
    request_id = result.get("request_id")

    if not bucket or not object_key:
        celery_logger.warning(f"save_result_task ({request_id}): thiếu bucket/object_key, bỏ qua lưu DB.")
        return result

    try:
        db.ensure_schema()
        db.save_result(bucket, object_key, result.get("etag", ""), result)
        celery_logger.info(f"save_result_task ({request_id}): saved {bucket}/{object_key} status={result.get('status')}")
    except Exception as e:
        celery_logger.error(f"save_result_task ({request_id}) failed for {bucket}/{object_key}: {e}", exc_info=True)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=10)

    return result
