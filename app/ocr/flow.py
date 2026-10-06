import os
import re
import json
import base64
import logging
import unicodedata
import time
import requests
import numpy as np
import cv2
from PIL import Image, ImageSequence, UnidentifiedImageError
from dotenv import load_dotenv
from . import spelling_service
from .utils import find_skew, clean_text, mask_sensitive_words

load_dotenv()

CHANDRA_URL        = os.getenv("CHANDRA_URL")
CHANDRA_MODEL      = os.getenv("CHANDRA_MODEL")
CHANDRA_MAX_TOKENS = int(os.getenv("CHANDRA_MAX_TOKENS", "2000"))
MAX_LONG_SIDE      = 5800

SPELL_CHUNK_WORDS = 40

_log = logging.getLogger("ocr")
_log.setLevel(logging.INFO)
_log.propagate = True

PDF_EXTS = {".pdf"}

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    return _clahe.apply(gray)


def _deskew(gray: np.ndarray, page_idx: int) -> np.ndarray:
    angle = find_skew(gray)
    if abs(angle) >= 0.5:
        h, w = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        _log.info(f"  page {page_idx+1} deskew: {angle:.2f}°")
    return gray


def _pdf_to_images(pdf_path: str) -> list:
    from pdf2image import convert_from_path
    pil_images = convert_from_path(pdf_path, dpi=200)
    result = []
    for i, pil in enumerate(pil_images):
        arr  = np.array(pil.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = _apply_clahe(gray)
        result.append(_deskew(gray, i))
    return result


def _image_file_to_images(image_path: str) -> list:
    result = []
    try:
        with Image.open(image_path) as im:
            for i, frame in enumerate(ImageSequence.Iterator(im)):
                arr  = np.array(frame.convert("RGB"))
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
                gray = _apply_clahe(gray)
                result.append(_deskew(gray, i))
    except UnidentifiedImageError as e:
        raise ValueError(f"Định dạng file không được hỗ trợ: {image_path}") from e
    return result


def _file_to_images(file_path: str) -> list:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in PDF_EXTS:
        return _pdf_to_images(file_path)
    return _image_file_to_images(file_path)


def _resize_for_ocr(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def preprocess_file(file_path: str, output_dir: str) -> list:
    os.makedirs(output_dir, exist_ok=True)
    pages = _file_to_images(file_path)
    paths = []
    for i, img in enumerate(pages):
        img = _resize_for_ocr(img)
        page_path = os.path.join(output_dir, f"page_{i}.jpg")
        cv2.imwrite(page_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        paths.append(page_path)
    return paths


def _img_file_to_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _ocr_page(image_path: str, page_idx: int) -> str:
    b64 = _img_file_to_b64(image_path)
    payload = {
        "model": CHANDRA_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": "<image>"},
            ],
        }],
        "max_tokens": CHANDRA_MAX_TOKENS,
        "temperature": 0.0,
    }
    try:
        resp = requests.post(CHANDRA_URL, json=payload, timeout=600)
        if not resp.ok:
            _log.error(f"  page {page_idx+1} OCR error: {resp.status_code} {resp.text[:2000]}")
            return ""
        content = resp.json()["choices"][0]["message"]["content"].strip()
        _log.info(f"  page {page_idx+1} OCR: {len(content)} chars")
        return content
    except Exception as e:
        _log.error(f"  page {page_idx+1} OCR error: {e}")
        return ""


def _strip_chandra_json(page_text: str) -> str:
    stripped = page_text.strip()
    if not stripped.startswith("[{"):
        return page_text

    try:
        parts = []
        first_line = stripped.split("\n")[0]
        objs = json.loads(first_line)
        if isinstance(objs, list):
            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                for field in ("titles", "text", "table", "other"):
                    val = obj.get(field, "")
                    if not val or not isinstance(val, str):
                        continue
                    viet_phrases = re.findall(r"'([^']{2,200})'", val)
                    for phrase in viet_phrases:
                        if re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', phrase, re.IGNORECASE):
                            parts.append(phrase)

        rest = "\n".join(stripped.split("\n")[1:])
        for m in re.finditer(r'"text"\s*:\s*"([^"]{1,200})"', rest):
            val = m.group(1)
            if val and not val.startswith("\\"):
                parts.append(val)

        if parts:
            return "\n".join(parts)
    except Exception:
        pass

    return page_text


def _strip_html(text: str) -> str:
    text = re.sub(r'<div[^>]*data-label="Figure"[^>]*>.*?(?:</div>|$)', " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<img[^>]*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<su[pb][^>]*>(.*?)</su[pb]>", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"</t[dh]>\s*<t[dh][^>]*>", " | ", text, flags=re.IGNORECASE)
    text = re.sub(r"<t[dh][^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</t[dh]>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(?:p|div|h[1-6]|li|tr)>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<(?:p|div|h[1-6]|li|tr)[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[*_#`]{1,3}", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_text(page_image_paths: list) -> str:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(page_image_paths)) as executor:
        futures = [executor.submit(_ocr_page, p, i) for i, p in enumerate(page_image_paths)]
        pages   = [_strip_chandra_json(f.result()) for f in futures]
    text = "\n[PAGE_BREAK]\n".join(pages)
    text = unicodedata.normalize("NFC", text).replace("\xa0", " ")
    return text


_PAGE_BREAK_MARK = "[PAGE_BREAK]"


def _chunk_line(line: str, max_words: int = SPELL_CHUNK_WORDS) -> list:
    words = line.split(" ")
    if len(words) <= max_words:
        return [line]
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _correct_text(raw_text: str) -> str:
    lines = raw_text.split("\n")

    line_chunks = [None] * len(lines)
    flat_chunks = []
    positions   = []

    for li, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped == _PAGE_BREAK_MARK:
            line_chunks[li] = [line]
            continue
        chunks = _chunk_line(line)
        line_chunks[li] = chunks
        for ci, c in enumerate(chunks):
            flat_chunks.append(c)
            positions.append((li, ci))

    t0 = time.time()
    corrected_flat = spelling_service.correct_batch(flat_chunks)
    _log.info(f"  spelling correct: {len(flat_chunks)} chunk(s) ({time.time()-t0:.2f}s)")

    for (li, ci), corrected in zip(positions, corrected_flat):
        line_chunks[li][ci] = corrected

    return "\n".join(" ".join(chunks) for chunks in line_chunks)


def ocr_stage(context: dict) -> dict:
    request_id        = context.get("request_id", "unknown")
    page_image_paths  = context.get("page_image_paths")

    if not page_image_paths:
        file_path = context.get("file_path") or context.get("pdf_path", "")
        if not file_path or not os.path.exists(file_path):
            _log.warning(f"[{request_id}] File not found: {file_path}")
            context["ocr_raw_text"]   = ""
            context["ocr_plain_text"] = ""
            return context
        _log.info(f"[{request_id}] Không có page_image_paths — tiền xử lý tại chỗ (fallback).")
        temp_dir = context.get("temp_folder") or os.path.dirname(file_path)
        page_image_paths = preprocess_file(file_path, temp_dir)

    t0 = time.time()
    _log.info(f"[{request_id}] Start OCR: {len(page_image_paths)} trang")

    raw_text   = _extract_text(page_image_paths)
    plain_text = _strip_html(raw_text)
    plain_text = unicodedata.normalize("NFC", plain_text)

    _log.info(f"[{request_id}] OCR done: {len(plain_text)} chars ({time.time()-t0:.2f}s)")

    context["ocr_raw_text"]   = raw_text
    context["ocr_plain_text"] = plain_text
    return context


def spelling_correct_stage(context: dict) -> dict:
    request_id = context.get("request_id", "unknown")
    plain_text = context.get("ocr_plain_text", "")

    if not plain_text:
        _log.warning(f"[{request_id}] No OCR text, skipping spelling correction.")
        context["corrected_text"] = ""
        return context

    t0 = time.time()
    corrected = _correct_text(plain_text)
    _log.info(f"[{request_id}] Spelling correction done: {len(corrected)} chars ({time.time()-t0:.2f}s)")

    context["corrected_text"] = corrected
    return context


def sensitive_filter_stage(context: dict) -> dict:
    context["corrected_text"] = mask_sensitive_words(context.get("corrected_text", ""))
    return context


def ocr_flow(context: dict) -> dict:
    context = ocr_stage(context)
    context = spelling_correct_stage(context)
    context = sensitive_filter_stage(context)
    return context
