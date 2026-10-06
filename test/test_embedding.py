import argparse
import glob
import os
import sys
import unittest
import uuid
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.embedding import chat_service, chunking, embedding_service, milvus_store, reranker_service

TXT_DIR = os.path.join(os.path.dirname(__file__), "output")
TEST_COLLECTION = "test_embedding"
TEST_BUCKET = "test_embedding"


def ingest_txt_folder(txt_dir: str, collection_name: str = TEST_COLLECTION) -> dict:
    milvus_store.ensure_collection(collection_name)

    summary = {}
    for path in sorted(glob.glob(os.path.join(txt_dir, "*.txt"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        object_key = os.path.basename(path)
        chunks = chunking.split_text(text)
        if not chunks:
            milvus_store.delete_document_chunks(TEST_BUCKET, object_key, collection_name)
            summary[object_key] = 0
            continue

        texts = [c["text"] for c in chunks]
        embeddings = embedding_service.embed_texts(texts)
        milvus_store.upsert_chunks(TEST_BUCKET, object_key, chunks, embeddings, collection_name)
        summary[object_key] = len(chunks)

    return summary


def demo_rerank(question: str, collection_name: str = TEST_COLLECTION, top_k: int = 20) -> list:
    query_vec = embedding_service.embed_query(question)
    candidates = milvus_store.search(query_vec, top_k=top_k, collection_name=collection_name)
    if not candidates:
        return []

    passages = [c["chunk_text"] for c in candidates]
    scores = reranker_service.rerank(question, passages)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = s
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates


def ask(question: str, collection_name: str = TEST_COLLECTION, session_id: str = None) -> dict:
    return chat_service.answer_question(question, collection_name=collection_name, session_id=session_id)


def preview_retrieval_query(question: str, session_id: str) -> str:
    history = chat_service.history_db.get_recent_turns(session_id) if session_id else []
    return chat_service._build_retrieval_query(question, history)


class EmbeddingIngestTest(unittest.TestCase):
    def test_ingest_txt_folder_into_test_collection(self):
        txt_files = glob.glob(os.path.join(TXT_DIR, "*.txt"))
        if not txt_files:
            self.skipTest(f"Không có file .txt trong {TXT_DIR} — chạy test_ocr.py trước để tạo output.")

        summary = ingest_txt_folder(TXT_DIR, TEST_COLLECTION)
        self.assertGreater(len(summary), 0)
        for name, n_chunks in summary.items():
            self.assertGreaterEqual(n_chunks, 0, f"{name} có số chunk âm bất thường")


class ChatbotQueryTest(unittest.TestCase):
    def test_ask_question_against_test_collection(self):
        txt_files = glob.glob(os.path.join(TXT_DIR, "*.txt"))
        if not txt_files:
            self.skipTest(f"Không có file .txt trong {TXT_DIR} — chạy test_ocr.py trước để tạo output.")

        summary = ingest_txt_folder(TXT_DIR, TEST_COLLECTION)
        if not any(summary.values()):
            self.skipTest("Không có chunk nào được nạp vào collection test.")

        result = ask("Văn bản này nói về nội dung gì?", collection_name=TEST_COLLECTION)
        self.assertIn("answer", result)
        self.assertIsInstance(result["answer"], str)
        self.assertGreater(len(result["answer"]), 0)


def _fake_candidate(text: str, chunk_index: int = 0, key: str = "doc.txt") -> dict:
    return {"bucket": "b", "object_key": key, "chunk_index": chunk_index, "page_index": 0, "chunk_text": text}


class RetrieveLogicTest(unittest.TestCase):
    def _run_retrieve(self, candidates: list, scores: list, min_score: float = -5.0, top_n: int = 5) -> list:
        with mock.patch.object(chat_service.embedding_service, "embed_query", return_value=[0.0]), \
             mock.patch.object(chat_service.milvus_store, "search", return_value=candidates), \
             mock.patch.object(chat_service.reranker_service, "rerank", return_value=scores), \
             mock.patch.object(chat_service, "RERANK_MIN_SCORE", min_score), \
             mock.patch.object(chat_service, "RERANK_TOP_N", top_n):
            return chat_service._retrieve("câu hỏi")

    def test_chunks_below_min_score_are_dropped(self):
        result = self._run_retrieve([_fake_candidate("a"), _fake_candidate("b", 1)], [2.5, -9.0])
        self.assertEqual([c["chunk_text"] for c in result], ["a"])

    def test_returns_empty_when_all_below_min_score(self):
        self.assertEqual(self._run_retrieve([_fake_candidate("a"), _fake_candidate("b", 1)], [-6.0, -9.0]), [])

    def test_duplicate_text_from_different_files_kept_once(self):
        candidates = [
            _fake_candidate("cùng nội dung", 0, "page_0.txt"),
            _fake_candidate("cùng  nội\ndung", 0, "full.txt"),
            _fake_candidate("khác", 1, "full.txt"),
        ]
        result = self._run_retrieve(candidates, [3.0, 2.9, 1.0])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["object_key"], "page_0.txt")

    def test_top_n_limit_applied_after_filtering(self):
        candidates = [_fake_candidate(f"t{i}", i) for i in range(6)]
        result = self._run_retrieve(candidates, [6, 5, 4, 3, 2, 1], top_n=3)
        self.assertEqual([c["chunk_text"] for c in result], ["t0", "t1", "t2"])

    def test_results_sorted_by_rerank_score(self):
        result = self._run_retrieve([_fake_candidate("thấp"), _fake_candidate("cao", 1)], [1.0, 4.0])
        self.assertEqual(result[0]["chunk_text"], "cao")

    def test_no_candidates_from_milvus(self):
        self.assertEqual(self._run_retrieve([], []), [])


class BuildRetrievalQueryTest(unittest.TestCase):
    def test_no_history_returns_question_only(self):
        self.assertEqual(chat_service._build_retrieval_query("câu 2", []), "câu 2")

    def test_prior_user_questions_are_prepended_but_answers_are_not(self):
        history = [
            {"role": "user", "content": "câu 1"},
            {"role": "assistant", "content": "trả lời 1"},
        ]
        self.assertEqual(chat_service._build_retrieval_query("câu 2", history), "câu 1 câu 2")

    def test_only_last_n_prior_questions_are_used(self):
        history = []
        for i in range(1, 5):
            history += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
        with mock.patch.object(chat_service, "RETRIEVAL_HISTORY_QUESTIONS", 2):
            self.assertEqual(chat_service._build_retrieval_query("q5", history), "q3 q4 q5")


class AnswerQuestionFlowTest(unittest.TestCase):
    def _chunk(self) -> dict:
        return {"bucket": "b", "object_key": "doc.txt", "chunk_index": 0, "chunk_text": "nội dung", "rerank_score": 3.0}

    def test_no_relevant_chunks_skips_llm_call(self):
        with mock.patch.object(chat_service, "_retrieve", return_value=[]), \
             mock.patch.object(chat_service, "_call_rag_llm") as llm:
            result = chat_service.answer_question("hỏi gì đó")
        llm.assert_not_called()
        self.assertEqual(result["sources"], [])

    def test_llm_sees_original_question_and_history_is_saved(self):
        history = [{"role": "user", "content": "câu 1"}, {"role": "assistant", "content": "trả lời 1"}]
        with mock.patch.object(chat_service.history_db, "get_recent_turns", return_value=history), \
             mock.patch.object(chat_service.history_db, "save_turn") as save, \
             mock.patch.object(chat_service, "_retrieve", return_value=[self._chunk()]) as retrieve, \
             mock.patch.object(chat_service, "_call_rag_llm", return_value="<think>x</think>Đáp án") as llm:
            result = chat_service.answer_question("câu 2", session_id="s1")

        self.assertEqual(retrieve.call_args.args[0], "câu 1 câu 2")
        self.assertEqual(llm.call_args.args[1], "câu 2")
        self.assertEqual(result["answer"], "Đáp án")
        save.assert_called_once_with("s1", "câu 2", "Đáp án", doc=("b", "doc.txt"))

    def test_without_session_id_history_is_not_touched(self):
        with mock.patch.object(chat_service.history_db, "get_recent_turns") as get_h, \
             mock.patch.object(chat_service.history_db, "save_turn") as save, \
             mock.patch.object(chat_service, "_retrieve", return_value=[self._chunk()]), \
             mock.patch.object(chat_service, "_call_rag_llm", return_value="ok"):
            chat_service.answer_question("câu hỏi")
        get_h.assert_not_called()
        save.assert_not_called()


class SummaryIntentTest(unittest.TestCase):
    def test_summary_keywords_detected(self):
        for q in ["Tóm tắt Quyết định 1780", "tóm lược văn bản này giúp tôi", "Hãy TÓM GỌN thông tư 05"]:
            self.assertTrue(chat_service._is_summary_request(q), q)

    def test_normal_questions_not_summary(self):
        for q in ["Quyết định 1780 là về gì?", "Chủ tịch hội đồng là ai?"]:
            self.assertFalse(chat_service._is_summary_request(q), q)

    def test_section_or_conversation_summary_goes_to_normal_rag(self):
        for q in ["Tóm tắt Điều 2", "tóm tắt khoản 3 điều 5", "Tóm tắt lại câu trả lời trên", "tóm tắt cuộc trò chuyện"]:
            self.assertFalse(chat_service._is_summary_request(q), q)


class MergeChunkTextsTest(unittest.TestCase):
    def test_overlap_between_chunks_is_removed(self):
        overlap = "phần chồng lấn giữa hai đoạn liên tiếp"
        merged = chat_service._merge_chunk_texts([f"đoạn một {overlap}", f"{overlap} đoạn hai"])
        self.assertEqual(merged, f"đoạn một {overlap} đoạn hai")

    def test_chunks_without_overlap_joined_by_newline(self):
        self.assertEqual(chat_service._merge_chunk_texts(["aaa", "bbb"]), "aaa\nbbb")

    def test_empty_list(self):
        self.assertEqual(chat_service._merge_chunk_texts([]), "")


class SummaryFlowTest(unittest.TestCase):
    def _rows(self) -> list:
        return [{"chunk_index": 0, "page_index": 0, "chunk_text": "nội dung trang một"},
                {"chunk_index": 1, "page_index": 1, "chunk_text": "nội dung trang hai"}]

    def _chunk(self, key: str, score: float) -> dict:
        return {"bucket": "b", "object_key": key, "chunk_index": 0, "chunk_text": key, "rerank_score": score}

    def test_summary_resolves_document_by_retrieval_and_reads_first_pages_only(self):
        with mock.patch.object(chat_service, "_retrieve", return_value=[self._chunk("qd1780.txt", 3.0)]), \
             mock.patch.object(chat_service.milvus_store, "get_document_chunks", return_value=(self._rows(), True)) as get_chunks, \
             mock.patch.object(chat_service, "_call_summary_llm", return_value="<think>x</think>Bản tóm tắt") as llm:
            result = chat_service.answer_question("Tóm tắt Quyết định 1780")

        self.assertEqual(get_chunks.call_args.args[:3], ("b", "qd1780.txt", chat_service.SUMMARY_MAX_PAGES))
        self.assertIn("nội dung trang một", llm.call_args.args[0])
        self.assertIn("Tóm tắt văn bản: qd1780.txt", result["answer"])
        self.assertIn("Bản tóm tắt", result["answer"])
        self.assertIn("trang đầu", result["answer"])

    def test_this_document_uses_last_document_of_session(self):
        with mock.patch.object(chat_service.history_db, "get_recent_turns", return_value=[]), \
             mock.patch.object(chat_service.history_db, "get_last_document", return_value=("b", "prev.txt")), \
             mock.patch.object(chat_service.history_db, "save_turn") as save, \
             mock.patch.object(chat_service, "_retrieve") as retrieve, \
             mock.patch.object(chat_service.milvus_store, "get_document_chunks", return_value=(self._rows(), False)) as get_chunks, \
             mock.patch.object(chat_service, "_call_summary_llm", return_value="ok"):
            result = chat_service.answer_question("Tóm tắt văn bản này", session_id="s1")

        retrieve.assert_not_called()
        self.assertEqual(get_chunks.call_args.args[1], "prev.txt")
        self.assertNotIn("trang đầu", result["answer"])
        save.assert_called_once()
        self.assertEqual(save.call_args.kwargs["doc"], ("b", "prev.txt"))

    def test_close_second_document_is_listed_as_alternative(self):
        chunks = [self._chunk("a.txt", 3.0), self._chunk("b.txt", 2.0), self._chunk("c.txt", -3.0)]
        with mock.patch.object(chat_service, "_retrieve", return_value=chunks), \
             mock.patch.object(chat_service.milvus_store, "get_document_chunks", return_value=(self._rows(), False)), \
             mock.patch.object(chat_service, "_call_summary_llm", return_value="ok"):
            result = chat_service.answer_question("tóm tắt thông tư")

        self.assertIn("Tóm tắt văn bản: a.txt", result["answer"])
        self.assertIn("b.txt", result["answer"].split("Nếu bạn muốn")[-1])
        self.assertNotIn("c.txt", result["answer"])

    def test_text_longer_than_char_limit_is_cut_and_flagged(self):
        long_rows = [{"chunk_index": 0, "page_index": 0, "chunk_text": "x" * 500}]
        with mock.patch.object(chat_service, "_retrieve", return_value=[self._chunk("a.txt", 3.0)]), \
             mock.patch.object(chat_service.milvus_store, "get_document_chunks", return_value=(long_rows, False)), \
             mock.patch.object(chat_service, "SUMMARY_MAX_CHARS", 100), \
             mock.patch.object(chat_service, "_call_summary_llm", return_value="ok") as llm:
            result = chat_service.answer_question("tóm tắt văn bản")

        self.assertEqual(len(llm.call_args.args[0]), 100)
        self.assertIn("trang đầu", result["answer"])

    def test_unknown_document_asks_user_to_name_it(self):
        with mock.patch.object(chat_service, "_retrieve", return_value=[]), \
             mock.patch.object(chat_service, "_call_summary_llm") as llm:
            result = chat_service.answer_question("tóm tắt văn bản nào đó")
        llm.assert_not_called()
        self.assertEqual(result["sources"], [])


def _print_answer(result: dict) -> None:
    print(f"Trả lời: {result['answer']}")
    print("Nguồn:")
    for src in result["sources"]:
        score = src.get("score")
        score_text = f", score={score:.4f}" if score is not None else ""
        print(f"  - {src['object_key']} (chunk {src['chunk_index']}{score_text})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nạp toàn bộ .txt vào Milvus collection test_embedding, thử rerank + hỏi chatbot.")
    parser.add_argument("--txt-dir", default=TXT_DIR, help="Thư mục chứa file .txt (mặc định: output của test_ocr.py)")
    parser.add_argument("--collection", default=TEST_COLLECTION, help="Tên collection Milvus dùng để test")
    parser.add_argument("--question", default=None, help="Câu hỏi để thử luôn chatbot RAG sau khi nạp xong")
    parser.add_argument("--skip-ingest", action="store_true", help="Bỏ qua bước nạp lại dữ liệu, chỉ hỏi chatbot trên collection đã có sẵn")
    parser.add_argument("--chat", action="store_true", help="Vào chế độ hỏi-đáp liên tục với chatbot (gõ 'exit' hoặc để trống để thoát)")
    parser.add_argument("--debug", action="store_true", help="In ra câu truy vấn tìm kiếm thực tế (đã ghép lịch sử) trước mỗi câu trả lời, dùng để kiểm tra retrieval theo ngữ cảnh")
    args = parser.parse_args()

    if args.skip_ingest:
        print(f"Bỏ qua nạp dữ liệu, dùng collection có sẵn '{args.collection}'")
    else:
        summary = ingest_txt_folder(args.txt_dir, args.collection)
        if not summary:
            print(f"Không tìm thấy file .txt nào trong {args.txt_dir}")
        for name, n_chunks in summary.items():
            print(f"{name}: {n_chunks} chunk(s) -> collection '{args.collection}'")

    if args.question:
        print(f"\nCâu hỏi: {args.question}")
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


# python test_embedding.py --skip-ingest --chat --debug