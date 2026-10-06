import asyncio
import logging
import os
import uuid
from functools import partial

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.auth import CORS_ORIGINS, require_user

from . import embedding_service, milvus_store, reranker_service
from .chat_service import answer_question
from .history_db import ensure_schema

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CHAT_GPU_CONCURRENCY = int(os.getenv("CHAT_GPU_CONCURRENCY", "1"))
_gpu_semaphore = asyncio.Semaphore(CHAT_GPU_CONCURRENCY)

app = FastAPI(title="RAG Chatbot Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "true").strip().lower() in ("1", "true", "yes")


@app.on_event("startup")
def on_startup():
    ensure_schema()
    if not PRELOAD_MODELS:
        return
    try:
        embedding_service.load_model()
        reranker_service.load_model()
        milvus_store.ensure_collection()
        logger.info("Preload xong: embedding, reranker, Milvus collection.")
    except Exception as e:
        logger.error(f"Preload thất bại (sẽ nạp lười ở request đầu tiên): {e}", exc_info=True)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    answer: str
    sources: list
    session_id: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: str = Depends(require_user)):
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"Received chat question (session={session_id}): {request.question}")
    loop = asyncio.get_event_loop()
    async with _gpu_semaphore:
        result = await loop.run_in_executor(None, partial(answer_question, request.question, session_id=session_id))
    return ChatResponse(session_id=session_id, **result)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
