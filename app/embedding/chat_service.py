import logging
import os
import re

import requests
from dotenv import load_dotenv

from . import embedding_service, history_db, milvus_store, reranker_service

load_dotenv()

logger = logging.getLogger("chat_service")

RAG_LLM_URL = os.getenv("RAG_LLM_URL")
RAG_LLM_MODEL = os.getenv("RAG_LLM_MODEL")

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "20"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
RETRIEVAL_HISTORY_QUESTIONS = int(os.getenv("RETRIEVAL_HISTORY_QUESTIONS", "2"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "-5.0"))
SUMMARY_MAX_PAGES = int(os.getenv("SUMMARY_MAX_PAGES", "3"))
SUMMARY_MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", "9000"))
SUMMARY_MAX_TOKENS = int(os.getenv("SUMMARY_MAX_TOKENS", "1024"))
SUMMARY_AMBIGUITY_MARGIN = 2.0

_SUMMARY_INTENT_RE = re.compile(r"\b(tóm tắt|tóm lược|tóm gọn)\b")
_SUMMARY_NOT_DOCUMENT_RE = re.compile(
    r"\b(điều|khoản|chương|mục|điểm)\s*\d+|câu trả lời|trả lời (trên|vừa)|cuộc (trò chuyện|hội thoại)|đoạn chat|những gì (vừa|bạn)"
)
_REFERS_TO_PREVIOUS_RE = re.compile(r"\b(này|đó|ấy|trên|vừa (rồi|nói|hỏi))\b")
_OVERLAP_SCAN_CHARS = 400

_SUMMARY_TEMPLATE = (
    "Hãy tóm tắt nội dung văn bản dưới đây bằng tiếng Việt, ngắn gọn và rõ ràng: "
    "văn bản về vấn đề gì, các nội dung chính, đối tượng và thời hạn áp dụng nếu có. "
    "Chỉ sử dụng thông tin có trong văn bản, không suy diễn.\n"
    "### Văn bản :\n{document}\n\n"
    "### Tóm tắt :"
)

_SYSTEM_PROMPT = (
    "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể.\n\n"
    "Bạn hãy trả lời theo định dạng sau:\n"
    "<think>\n[Suy nghĩ, phân tích của bạn]\n</think>\n[Câu trả lời của bạn]"
)

_RAG_TEMPLATE = (
    "Chú ý các yêu cầu sau:\n"
    "- Câu trả lời phải chính xác và đầy đủ nếu ngữ cảnh có câu trả lời.\n"
    "- Chỉ sử dụng các thông tin có trong ngữ cảnh được cung cấp.\n"
    "- Chỉ cần từ chối trả lời và không suy luận gì thêm nếu ngữ cảnh không có câu trả lời.\n"
    "Hãy trả lời câu hỏi dựa trên ngữ cảnh:\n"
    "### Ngữ cảnh :\n{context}\n\n"
    "### Câu hỏi :\n{question}\n\n"
    "### Trả lời :"
)


def _retrieve(question: str, collection_name: str = None) -> list:
    query_vec = embedding_service.embed_query(question)
    candidates = milvus_store.search(query_vec, top_k=RETRIEVAL_TOP_K, collection_name=collection_name)
    if not candidates:
        return []

    passages = [c["chunk_text"] for c in candidates]
    scores = reranker_service.rerank(question, passages)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = s
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

    selected, seen = [], set()
    for c in candidates:
        if c["rerank_score"] < RERANK_MIN_SCORE:
            break
        key = " ".join(c["chunk_text"].split())
        if key in seen:
            continue
        seen.add(key)
        selected.append(c)
        if len(selected) >= RERANK_TOP_N:
            break

    logger.info(
        f"retrieve: {len(candidates)} candidate(s), top score={candidates[0]['rerank_score']:.3f}, "
        f"kept {len(selected)} (min_score={RERANK_MIN_SCORE}, top_n={RERANK_TOP_N})"
    )
    return selected


def _post_llm(messages: list, max_tokens: int) -> str:
    payload = {
        "model": RAG_LLM_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    resp = requests.post(RAG_LLM_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_rag_llm(context: str, question: str, history: list = None) -> str:
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for turn in history or []:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": _RAG_TEMPLATE.format(context=context, question=question)})
    return _post_llm(messages, 4096)


def _call_summary_llm(document: str) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _SUMMARY_TEMPLATE.format(document=document)},
    ]
    return _post_llm(messages, SUMMARY_MAX_TOKENS)


def _is_summary_request(question: str) -> bool:
    q = question.lower()
    return bool(_SUMMARY_INTENT_RE.search(q)) and not _SUMMARY_NOT_DOCUMENT_RE.search(q)


def _rank_documents(chunks: list) -> list:
    best = {}
    for c in chunks:
        key = (c["bucket"], c["object_key"])
        if key not in best or c["rerank_score"] > best[key]:
            best[key] = c["rerank_score"]
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)


def _merge_chunk_texts(texts: list) -> str:
    merged = ""
    for text in texts:
        if not merged:
            merged = text
            continue
        k = min(len(merged), len(text), _OVERLAP_SCAN_CHARS)
        while k >= 20 and not merged.endswith(text[:k]):
            k -= 1
        merged += text[k:] if k >= 20 else "\n" + text
    return merged


def _build_retrieval_query(question: str, history: list) -> str:
    prior_questions = [turn["content"] for turn in history if turn["role"] == "user"]
    prior_questions = prior_questions[-RETRIEVAL_HISTORY_QUESTIONS:]
    if not prior_questions:
        return question
    return " ".join(prior_questions + [question])


def _answer_summary(question: str, history: list, collection_name: str, session_id: str) -> dict:
    last_doc = history_db.get_last_document(session_id) if session_id else None
    target, alternatives = None, []

    if last_doc and _REFERS_TO_PREVIOUS_RE.search(question.lower()):
        target = last_doc
    else:
        chunks = _retrieve(_build_retrieval_query(question, history), collection_name=collection_name)
        ranked = _rank_documents(chunks)
        if ranked:
            target = ranked[0][0]
            alternatives = [k for k, s in ranked[1:3] if ranked[0][1] - s <= SUMMARY_AMBIGUITY_MARGIN]
        elif last_doc:
            target = last_doc

    if target is None:
        return {"answer": "Xin lỗi, tôi không xác định được văn bản cần tóm tắt. Bạn hãy nêu rõ tên hoặc số hiệu văn bản.", "sources": []}

    bucket, object_key = target
    rows, truncated = milvus_store.get_document_chunks(bucket, object_key, SUMMARY_MAX_PAGES, collection_name)
    if not rows:
        return {"answer": f"Xin lỗi, tôi không đọc được nội dung của văn bản {object_key}.", "sources": []}

    document = _merge_chunk_texts([r["chunk_text"] for r in rows])
    if len(document) > SUMMARY_MAX_CHARS:
        document = document[:SUMMARY_MAX_CHARS]
        truncated = True

    try:
        raw = _call_summary_llm(document)
    except Exception as e:
        logger.error(f"Summary LLM call failed: {e}", exc_info=True)
        return {"answer": "Xin lỗi, hệ thống đang gặp sự cố khi tóm tắt. Vui lòng thử lại sau.", "sources": []}

    summary = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    parts = [f"Tóm tắt văn bản: {object_key}", "", summary]
    if truncated:
        parts += ["", f"(Lưu ý: chỉ dựa trên {SUMMARY_MAX_PAGES} trang đầu của văn bản.)"]
    if alternatives:
        parts += ["", "Nếu bạn muốn tóm tắt văn bản khác, hãy nêu rõ tên: " + ", ".join(k[1] for k in alternatives)]
    answer = "\n".join(parts)

    if session_id:
        history_db.save_turn(session_id, question, answer, doc=target)

    sources = [
        {"bucket": bucket, "object_key": object_key, "chunk_index": r["chunk_index"], "score": None}
        for r in rows
    ]
    return {"answer": answer, "sources": sources}


def answer_question(question: str, collection_name: str = None, session_id: str = None) -> dict:
    history = history_db.get_recent_turns(session_id) if session_id else []

    if _is_summary_request(question):
        return _answer_summary(question, history, collection_name, session_id)

    retrieval_query = _build_retrieval_query(question, history)

    top_chunks = _retrieve(retrieval_query, collection_name=collection_name)
    if not top_chunks:
        return {
            "answer": "Xin lỗi, tôi không tìm thấy tài liệu liên quan để trả lời câu hỏi này.",
            "sources": [],
        }

    context = "\n\n".join(
        f"[{i+1}] ({c['object_key']}) {c['chunk_text']}" for i, c in enumerate(top_chunks)
    )

    try:
        raw = _call_rag_llm(context, question, history=history)
    except Exception as e:
        logger.error(f"RAG LLM call failed: {e}", exc_info=True)
        return {
            "answer": "Xin lỗi, hệ thống đang gặp sự cố khi tạo câu trả lời. Vui lòng thử lại sau.",
            "sources": [],
        }

    answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    if session_id:
        history_db.save_turn(session_id, question, answer, doc=(top_chunks[0]["bucket"], top_chunks[0]["object_key"]))

    sources = [
        {
            "bucket": c["bucket"],
            "object_key": c["object_key"],
            "chunk_index": c["chunk_index"],
            "score": c["rerank_score"],
        }
        for c in top_chunks
    ]
    return {"answer": answer, "sources": sources}
