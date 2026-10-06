import os

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

_PAGE_BREAK_SEP = "\n[PAGE_BREAK]\n"


def _build_clean_text_with_page_offsets(raw_text: str):
    pages = raw_text.split(_PAGE_BREAK_SEP)
    clean_parts = []
    page_starts = []
    cursor = 0
    for page_text in pages:
        page_starts.append(cursor)
        clean_parts.append(page_text)
        cursor += len(page_text) + 1
    return "\n".join(clean_parts), page_starts


def _page_index_for_offset(offset: int, page_starts: list) -> int:
    page_idx = 0
    for i, start in enumerate(page_starts):
        if start <= offset:
            page_idx = i
        else:
            break
    return page_idx


def split_text(text: str) -> list:
    if not text or not text.strip():
        return []

    clean_text, page_starts = _build_clean_text_with_page_offsets(text)
    if not clean_text.strip():
        return []

    chunks = []
    search_from = 0
    for piece in _splitter.split_text(clean_text):
        piece_stripped = piece.strip()
        if not piece_stripped:
            continue

        found_at = clean_text.find(piece, search_from)
        if found_at == -1:
            found_at = clean_text.find(piece)
        if found_at == -1:
            found_at = search_from

        chunks.append({
            "chunk_index": len(chunks),
            "page_index": _page_index_for_offset(found_at, page_starts),
            "text": piece_stripped,
        })
        search_from = found_at

    return chunks
