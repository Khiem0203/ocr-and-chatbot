import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app import auth


class BasicAuthTest(unittest.TestCase):
    def setUp(self):
        app = FastAPI()

        @app.get("/protected")
        def protected(user: str = Depends(auth.require_user)):
            return {"user": user}

        self.client = TestClient(app)
        patches = [
            mock.patch.object(auth, "USERNAME", "admin"),
            mock.patch.object(auth, "PASSWORD", "mat-khau-test"),
            mock.patch.object(auth, "AUTH_ENABLED", True),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_correct_credentials(self):
        r = self.client.get("/protected", auth=("admin", "mat-khau-test"))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"user": "admin"})

    def test_no_credentials_gets_basic_challenge(self):
        r = self.client.get("/protected")
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.headers.get("WWW-Authenticate"), "Basic")

    def test_wrong_password_or_username(self):
        self.assertEqual(self.client.get("/protected", auth=("admin", "sai")).status_code, 401)
        self.assertEqual(self.client.get("/protected", auth=("khac", "mat-khau-test")).status_code, 401)

    def test_vietnamese_characters_do_not_crash_comparison(self):
        self.assertEqual(self.client.get("/protected", auth=("quản-trị", "mật-khẩu")).status_code, 401)

    def test_unconfigured_credentials_fail_closed(self):
        with mock.patch.object(auth, "USERNAME", ""), mock.patch.object(auth, "PASSWORD", ""):
            self.assertEqual(self.client.get("/protected", auth=("", "")).status_code, 401)

    def test_auth_disabled_allows_access(self):
        with mock.patch.object(auth, "AUTH_ENABLED", False):
            r = self.client.get("/protected")
        self.assertEqual(r.json(), {"user": "anonymous"})


class ChatEndpointAuthTest(unittest.TestCase):
    def setUp(self):
        from app.embedding import main as chat_main

        self.chat_main = chat_main
        self.client = TestClient(chat_main.app)
        for p in (mock.patch.object(auth, "USERNAME", "admin"), mock.patch.object(auth, "PASSWORD", "pw-test-123"),
                  mock.patch.object(auth, "AUTH_ENABLED", True)):
            p.start()
            self.addCleanup(p.stop)

    def test_chat_requires_authentication(self):
        with mock.patch.object(self.chat_main, "answer_question") as answer:
            r = self.client.post("/api/chat", json={"question": "xin chào"})
        self.assertEqual(r.status_code, 401)
        answer.assert_not_called()

    def test_chat_works_with_credentials(self):
        with mock.patch.object(self.chat_main, "answer_question", return_value={"answer": "ok", "sources": []}):
            r = self.client.post("/api/chat", json={"question": "xin chào", "session_id": "abc"}, auth=("admin", "pw-test-123"))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["session_id"], "abc")

    def test_question_and_session_length_limits(self):
        creds = ("admin", "pw-test-123")
        self.assertEqual(self.client.post("/api/chat", json={"question": ""}, auth=creds).status_code, 422)
        self.assertEqual(self.client.post("/api/chat", json={"question": "x" * 2001}, auth=creds).status_code, 422)
        self.assertEqual(self.client.post("/api/chat", json={"question": "hi", "session_id": "s" * 129}, auth=creds).status_code, 422)

    def test_health_stays_open(self):
        self.assertEqual(self.client.get("/health").status_code, 200)


if __name__ == "__main__":
    unittest.main()
