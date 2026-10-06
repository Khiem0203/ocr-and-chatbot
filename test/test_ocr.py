import argparse
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ocr.flow import ocr_flow
from app.ocr.utils import TEXT_DOC_EXTS, extract_document_text

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

IMAGE_EXTS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
SUPPORTED_EXTS = IMAGE_EXTS | TEXT_DOC_EXTS


def process_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in TEXT_DOC_EXTS:
        return extract_document_text(file_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        context = ocr_flow({
            "file_path": file_path,
            "temp_folder": temp_dir,
            "request_id": os.path.splitext(os.path.basename(file_path))[0],
        })
    return context.get("corrected_text", "")


def run_folder(data_dir: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    for name in sorted(os.listdir(data_dir)):
        ext = os.path.splitext(name)[1].lower()
        if ext not in SUPPORTED_EXTS:
            continue
        src = os.path.join(data_dir, name)
        text = process_file(src)
        out_path = os.path.join(output_dir, os.path.splitext(name)[0] + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        results[name] = out_path
    return results


class OcrFolderBatchTest(unittest.TestCase):
    def test_process_all_files_in_data_folder(self):
        if not os.path.isdir(DATA_DIR) or not any(
            os.path.splitext(n)[1].lower() in SUPPORTED_EXTS for n in os.listdir(DATA_DIR)
        ):
            self.skipTest(f"Không có file mẫu trong {DATA_DIR} — bỏ file PDF/ảnh/docx/txt vào đó rồi chạy lại.")

        results = run_folder(DATA_DIR, OUTPUT_DIR)
        self.assertGreater(len(results), 0)
        for name, out_path in results.items():
            self.assertTrue(os.path.exists(out_path), f"Thiếu output cho {name}")


class ProcessFileTempDirTest(unittest.TestCase):
    def test_page_images_not_written_next_to_input_file(self):
        captured = {}

        def fake_ocr_flow(context):
            captured.update(context)
            return {"corrected_text": "ok"}

        with tempfile.TemporaryDirectory() as data_dir:
            src = os.path.join(data_dir, "scan.pdf")
            with open(src, "wb") as f:
                f.write(b"%PDF-1.4")
            with mock.patch.object(sys.modules[__name__], "ocr_flow", fake_ocr_flow):
                text = process_file(src)

        self.assertEqual(text, "ok")
        self.assertIn("temp_folder", captured)
        self.assertNotEqual(os.path.abspath(captured["temp_folder"]), os.path.abspath(data_dir))


class SensitiveMaskTest(unittest.TestCase):
    def _mask_with_words(self, words: list, text: str) -> str:
        from app.ocr import utils

        with tempfile.TemporaryDirectory() as tmp:
            words_path = os.path.join(tmp, "words.txt")
            with open(words_path, "w", encoding="utf-8") as f:
                f.write("# comment\n" + "\n".join(words) + "\n")
            with mock.patch.object(utils, "_SENSITIVE_WORDS_PATH", utils._Path(words_path)), \
                 mock.patch.object(utils, "_sensitive_cache", None), \
                 mock.patch.object(utils, "_sensitive_mtime", 0.0):
                return utils.mask_sensitive_words(text)

    def test_single_word_is_masked_keeping_length(self):
        self.assertEqual(self._mask_with_words(["xấu"], "điều này xấu quá"), "điều này *** quá")

    def test_word_boundary_is_respected(self):
        self.assertEqual(self._mask_with_words(["vl"], "vlog và vl"), "vlog và **")

    def test_phrase_wins_over_its_single_word_regardless_of_order(self):
        result = self._mask_with_words(["xấu", "rất xấu xa"], "anh ta rất xấu xa lắm")
        self.assertEqual(result, "anh ta " + "*" * len("rất xấu xa") + " lắm")

    def test_phrase_matches_across_irregular_whitespace(self):
        result = self._mask_with_words(["rất xấu xa"], "rất   xấu\nxa")
        self.assertNotIn("xấu", result)

    def test_empty_and_clean_text_unchanged(self):
        self.assertEqual(self._mask_with_words(["xấu"], ""), "")
        self.assertEqual(self._mask_with_words(["xấu"], "văn bản bình thường"), "văn bản bình thường")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR toàn bộ file trong 1 thư mục local, xuất .txt tương ứng.")
    parser.add_argument("data_dir", nargs="?", default=DATA_DIR, help="Thư mục chứa file PDF/ảnh/docx/txt")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Thư mục ghi file .txt kết quả")
    args = parser.parse_args()

    results = run_folder(args.data_dir, args.output)
    if not results:
        print(f"Không tìm thấy file hỗ trợ nào trong {args.data_dir}")
    for name, path in results.items():
        size = os.path.getsize(path)
        print(f"{name} -> {path} ({size} bytes)")
