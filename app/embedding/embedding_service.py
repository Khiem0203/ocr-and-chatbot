import logging
import os
import threading

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("embedding")

MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "AITeamVN/Vietnamese_Embedding")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

_model = None
_lock = threading.Lock()


def load_model() -> None:
    global _model
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model from {MODEL_PATH} ...")
        _model = SentenceTransformer(MODEL_PATH)
        logger.info("Embedding model loaded.")


def embed_texts(texts: list) -> list:
    if not texts:
        return []
    load_model()
    vectors = _model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(text: str) -> list:
    return embed_texts([text])[0]
