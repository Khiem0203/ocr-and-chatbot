import logging
import os
import threading

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("reranker")

MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", "AITeamVN/Vietnamese_Reranker")
MAX_LENGTH = 2304

_tokenizer = None
_model = None
_device = None
_lock = threading.Lock()


def load_model() -> None:
    global _tokenizer, _model, _device
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        logger.info(f"Loading reranker model from {MODEL_PATH} ...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(_device)
        _model.eval()
        logger.info(f"Reranker model loaded on {_device}.")


def rerank(query: str, passages: list) -> list:
    if not passages:
        return []
    load_model()
    import torch
    pairs = [[query, p] for p in passages]
    inputs = _tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=MAX_LENGTH).to(_device)
    with torch.no_grad():
        scores = _model(**inputs, return_dict=True).logits.view(-1).float()
    return scores.cpu().tolist()
