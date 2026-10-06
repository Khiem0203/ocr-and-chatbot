import argparse
import os
import sys
import unittest
import uuid

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS_DIR, "..")))
sys.path.insert(0, _THIS_DIR)

from test_ocr import DATA_DIR, OUTPUT_DIR, run_folder
from test_embedding import TEST_COLLECTION, _print_answer, ask, ingest_txt_folder, preview_retrieval_query


def run_full_pipeline(data_dir: str = DATA_DIR, output_dir: str = OUTPUT_DIR, collection_name: str = TEST_COLLECTION) -> dict:
    ocr_results = run_folder(data_dir, output_dir)
    ingest_summary = ingest_txt_folder(output_dir, collection_name)
    return {"ocr": ocr_results, "ingest": ingest_summary}


class FullLocalPipelineTest(unittest.TestCase):
    def test_ocr_then_embedding_on_local_data_folder(self):
        if not os.path.isdir(DATA_DIR) or not os.listdir(DATA_DIR):
            self.skipTest(f"Không có file mẫu trong {DATA_DIR} — bỏ file PDF/ảnh/docx/txt vào đó rồi chạy lại.")

        result = run_full_pipeline()
        self.assertGreater(len(result["ocr"]), 0)
        self.assertEqual(len(result["ocr"]), len(result["ingest"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy toàn bộ pipeline local: OCR thư mục -> nạp Milvus test_embedding -> (tuỳ chọn) hỏi thử.")
    parser.add_argument("data_dir", nargs="?", default=DATA_DIR, help="Thư mục chứa file PDF/ảnh/docx/txt")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Thư mục ghi file .txt trung gian")
    parser.add_argument("--collection", default=TEST_COLLECTION, help="Tên collection Milvus dùng để test")
    parser.add_argument("--question", default=None, help="Câu hỏi để thử luôn chatbot RAG sau khi chạy xong")
    parser.add_argument("--chat", action="store_true", help="Vào chế độ hỏi-đáp liên tục với chatbot sau khi chạy xong (gõ 'exit' hoặc để trống để thoát)")
    parser.add_argument("--debug", action="store_true", help="In ra câu truy vấn tìm kiếm thực tế (đã ghép lịch sử) trước mỗi câu trả lời")
    args = parser.parse_args()

    result = run_full_pipeline(args.data_dir, args.output, args.collection)

    print("=== OCR ===")
    for name, path in result["ocr"].items():
        print(f"{name} -> {path}")

    print("\n=== Embedding ===")
    for name, n_chunks in result["ingest"].items():
        print(f"{name}: {n_chunks} chunk(s) -> collection '{args.collection}'")

    if args.question:
        print(f"\n=== Chatbot ===\nCâu hỏi: {args.question}")
        _print_answer(ask(args.question, collection_name=args.collection))

    if args.chat:
        session_id = str(uuid.uuid4())
        print(f"\n=== Chế độ chat với collection '{args.collection}' (session={session_id}, gõ 'exit' hoặc Enter trống để thoát) ===")
        while True:
            question = input("\nBạn hỏi: ").strip()
            if not question or question.lower() in ("exit", "quit"):
                break
            if args.debug:
                print(f"[debug] retrieval query: {preview_retrieval_query(question, session_id)}")
            _print_answer(ask(question, collection_name=args.collection, session_id=session_id))
