import asyncio
import datetime
import logging
from functools import partial

from celery import chain
from celery.result import AsyncResult
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth import CORS_ORIGINS, require_user
from app.embedding.task import chunk_embed_task

from . import db
from .celery_app import celery_app
from .models import OCRRequest, StatusResponse, SubmitResponse
from .sync_task import sync_minio_task
from .task import ocr_task, prepare_ocr_task, save_result_task

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OCR Service (Celery)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/submit-ocr", response_model=SubmitResponse, status_code=202, dependencies=[Depends(require_user)])
async def submit_ocr(request: OCRRequest):
    logger.info(f"Received OCR request: {request.s3_path}")

    task_chain = chain(
        prepare_ocr_task.s(
            s3_path=request.s3_path,
            access_key=request.access_key,
            secret_key=request.secret_key,
            endpoint=request.endpoint,
            document_id=request.document_id,
        ).set(queue="prepare_queue"),
        ocr_task.s().set(queue="process_queue"),
        save_result_task.s().set(queue="db_queue"),
        chunk_embed_task.s().set(queue="embed_queue"),
    ).apply_async()

    return SubmitResponse(
        error_code=202,
        error_message="Đã nhận yêu cầu OCR. Dùng /api/request-status/{request_id} để lấy kết quả.",
        request_id=task_chain.id,
    )


@app.get("/api/request-status/{request_id}", response_model=StatusResponse, dependencies=[Depends(require_user)])
async def get_request_status(request_id: str):
    task_result = AsyncResult(request_id, app=celery_app)

    result_info = task_result.info
    if isinstance(task_result.info, Exception):
        result_info = str(task_result.info)

    return StatusResponse(
        request_id=request_id,
        status=task_result.state,
        result=result_info,
    )


@app.get("/api/documents", dependencies=[Depends(require_user)])
async def list_documents(bucket: str = None, status: str = None, limit: int = 200):
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, partial(db.list_documents, bucket, status, limit))
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        return []


@app.post("/api/sync-minio", dependencies=[Depends(require_user)])
async def trigger_minio_sync():
    task = sync_minio_task.apply_async(queue="sync_queue")
    return {"request_id": task.id, "message": "Đã kích hoạt đồng bộ MinIO — xem log worker sync_queue."}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.now().isoformat()}
