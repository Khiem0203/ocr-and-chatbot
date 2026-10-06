import logging
import os
import threading

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("spelling_correction")

MODEL_PATH  = os.getenv("SPELL_MODEL_PATH", "yammdd/vietnamese-error-correction")
MAX_LENGTH  = 256
BATCH_SIZE  = int(os.getenv("SPELL_BATCH_SIZE", "16"))

_tokenizer = None
_model     = None
_device    = None
_lock      = threading.Lock()


def load_model() -> None:
    global _tokenizer, _model, _device
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info(f"Loading spelling-correction model from {MODEL_PATH} ...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        _device    = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(_device)
        _model.eval()
        logger.info(f"Spelling-correction model loaded on {_device}.")


def _generate_batch(texts: list) -> list:
    import torch

    inputs = _tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH,
    ).to(_device)
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs, max_new_tokens=MAX_LENGTH, num_beams=1, do_sample=False,
        )
    return _tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def correct_batch(texts: list) -> list:
    if not texts:
        return []
    load_model()

    results = [None] * len(texts)
    idx_map = [i for i, t in enumerate(texts) if t and t.strip()]
    to_run  = [texts[i] for i in idx_map]

    for start in range(0, len(to_run), BATCH_SIZE):
        batch     = to_run[start:start + BATCH_SIZE]
        batch_idx = idx_map[start:start + BATCH_SIZE]
        try:
            corrected = _generate_batch(batch)
        except Exception as e:
            logger.warning(f"Spelling correction batch failed ({e}) — giữ nguyên text gốc cho batch này")
            corrected = batch
        for i, c in zip(batch_idx, corrected):
            results[i] = c if c and c.strip() else texts[i]

    for i, t in enumerate(texts):
        if results[i] is None:
            results[i] = t
    return results


def unload_model() -> None:
    global _tokenizer, _model
    import gc

    _model     = None
    _tokenizer = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    logger.info("Spelling-correction model unloaded.")
