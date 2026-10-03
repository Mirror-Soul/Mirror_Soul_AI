"""Realtime calls search the single RAG store on the AI API server."""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-only")

import chromadb  # noqa: E402
import httpx  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from model_calling.clients import rag_search  # noqa: E402
from model_training import services  # noqa: E402
from model_training.routers import internal_rag  # noqa: E402
from shared.config import settings  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
KEY = "internal-rag-test-key"
USER_A = "5f0154ef-7d83-4ae4-a724-ae591e1c985e"
USER_B = "66506c15-c42a-455f-8af7-a23f76c03bb2"


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(internal_rag.router)
    return app


def _embed(text: str) -> list[float]:
    return [1.0, 0.0, 0.0] if "카페" in text else [0.0, 1.0, 0.0] if "갈등" in text else [0.0, 0.0, 1.0]


class InternalSearchApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(_app())
        self.body = {"userId": USER_A, "query": "비밀 질문 문장", "topK": 3}

    def test_rejects_when_server_key_is_not_configured(self) -> None:
        with patch.object(settings, "RAG_INTERNAL_API_KEY", ""):
            response = self.client.post("/internal/rag/search", json=self.body, headers={"X-Rag-Internal-Key": "x"})
        self.assertEqual(response.status_code, 503)

    def test_rejects_missing_or_wrong_key(self) -> None:
        with patch.object(settings, "RAG_INTERNAL_API_KEY", KEY):
            missing = self.client.post("/internal/rag/search", json=self.body)
            wrong = self.client.post("/internal/rag/search", json=self.body, headers={"X-Rag-Internal-Key": "nope"})
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(wrong.status_code, 401)

    def test_searches_store_with_valid_key_without_logging_query(self) -> None:
        memories = [{"documentId": "d1", "text": "t", "metadata": {"userId": USER_A}, "distance": None}]
        buffer = io.StringIO()
        with patch.object(settings, "RAG_INTERNAL_API_KEY", KEY), patch.object(
            services, "search_user_memories", return_value=memories
        ) as search, contextlib.redirect_stdout(buffer):
            response = self.client.post("/internal/rag/search", json=self.body, headers={"X-Rag-Internal-Key": KEY})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["memories"][0]["documentId"], "d1")
        search.assert_called_once_with(user_id=USER_A, query="비밀 질문 문장", top_k=3)
        self.assertNotIn("비밀 질문 문장", buffer.getvalue())
        self.assertNotIn(KEY, buffer.getvalue())

    def test_validates_top_k(self) -> None:
        with patch.object(settings, "RAG_INTERNAL_API_KEY", KEY):
            response = self.client.post(
                "/internal/rag/search",
                json={**self.body, "topK": 500},
                headers={"X-Rag-Internal-Key": KEY},
            )
        self.assertEqual(response.status_code, 422)


class RemoteClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self._patches = [
            patch.object(settings, "RAG_SEARCH_BASE_URL", "http://ai.internal:8000/"),
            patch.object(settings, "RAG_INTERNAL_API_KEY", KEY),
            patch.object(settings, "RAG_SEARCH_TIMEOUT_SECONDS", 3.0),
        ]
        for item in self._patches:
            item.start()
        self.requests: list[httpx.Request] = []

    def tearDown(self) -> None:
        rag_search._client = None
        for item in reversed(self._patches):
            item.stop()

    def _use(self, handler) -> None:
        def recording(request: httpx.Request):
            self.requests.append(request)
            return handler(request)

        rag_search._client = httpx.Client(transport=httpx.MockTransport(recording))

    def test_posts_to_internal_api_with_key(self) -> None:
        self._use(lambda request: httpx.Response(200, json={"success": True, "memories": [{"documentId": "d1"}]}))
        result = rag_search.search_user_memories(USER_A, "카페", 4)
        self.assertEqual(result, [{"documentId": "d1"}])
        request = self.requests[0]
        self.assertEqual(str(request.url), "http://ai.internal:8000/internal/rag/search")
        self.assertEqual(request.headers["X-Rag-Internal-Key"], KEY)
        self.assertIn(b'"topK":4', request.content.replace(b" ", b""))
        self.assertEqual(rag_search.search_mode(), "remote")

    def test_error_mapping(self) -> None:
        cases = [
            (lambda r: httpx.Response(401), rag_search.RagSearchConfigurationError),
            (lambda r: httpx.Response(503), rag_search.RagSearchConfigurationError),
            (lambda r: httpx.Response(500), rag_search.RagSearchUnavailable),
            (lambda r: httpx.Response(422), rag_search.RagSearchRejected),
            (lambda r: httpx.Response(200, text="not json"), rag_search.RagSearchError),
        ]
        for handler, error in cases:
            with self.subTest(error=error.__name__):
                self._use(handler)
                with self.assertRaises(error):
                    rag_search.search_user_memories(USER_A, "q", 3)

    def test_timeout_is_unavailable_and_key_not_leaked(self) -> None:
        def timeout(request: httpx.Request):
            raise httpx.ReadTimeout("timed out", request=request)

        self._use(timeout)
        with self.assertRaises(rag_search.RagSearchUnavailable) as caught:
            rag_search.search_user_memories(USER_A, "q", 3)
        self.assertEqual(caught.exception.code, "RAG_SEARCH_UNAVAILABLE")
        self.assertNotIn(KEY, str(caught.exception))

    def test_missing_key_is_configuration_error(self) -> None:
        with patch.object(settings, "RAG_INTERNAL_API_KEY", ""):
            with self.assertRaises(rag_search.RagSearchConfigurationError):
                rag_search.search_user_memories(USER_A, "q", 3)


class LocalModeTests(unittest.TestCase):
    def test_local_mode_uses_local_store(self) -> None:
        with patch.object(settings, "RAG_SEARCH_BASE_URL", ""), patch.object(
            services, "search_user_memories", return_value=[{"documentId": "local"}]
        ) as local:
            result = rag_search.search_user_memories(USER_A, "q", 2)
        self.assertEqual(result, [{"documentId": "local"}])
        local.assert_called_once_with(USER_A, "q", 2)
        self.assertEqual(rag_search.search_mode(), "local")


class EndToEndTests(unittest.TestCase):
    """Call-server client -> internal API -> AI server ChromaDB."""

    def test_call_server_client_reads_ai_server_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = chromadb.PersistentClient(path=directory).get_or_create_collection("e2e")
            with patch.object(services, "collection", store), patch.object(
                services, "create_embeddings", side_effect=lambda texts: [_embed(t) for t in texts]
            ), patch.object(services, "create_embedding", side_effect=_embed), patch.object(
                settings, "RAG_INTERNAL_API_KEY", KEY
            ), patch.object(settings, "RAG_SEARCH_BASE_URL", "http://testserver"), patch.object(
                settings, "RAG_MAX_DISTANCE", 0.75
            ):
                for user in (USER_A, USER_B):
                    services.add_member_profile_to_rag(
                        user_id=user,
                        ai_profile_id="clone-1",
                        mbti="ENFP",
                        interview_samples=[
                            {"questionId": 1, "questionText": "쉬는 날?", "transcript": "카페에 가요"},
                            {"questionId": 2, "questionText": "갈등은?", "transcript": "갈등이 생기면 대화해요"},
                        ],
                    )
                rag_search._client = TestClient(_app())
                try:
                    memories = rag_search.search_user_memories(USER_A, "카페 좋아해?", 6)
                finally:
                    rag_search._client = None

        ids = [memory["documentId"] for memory in memories]
        self.assertEqual(
            ids,
            [f"{USER_A}:profile_snapshot:clone-1", f"{USER_A}:interview_memory:question-1"],
        )
        self.assertTrue(all(memory["metadata"]["userId"] == USER_A for memory in memories))


class CallServerIsolationTests(unittest.TestCase):
    def test_call_server_does_not_open_a_local_store_in_remote_mode(self) -> None:
        env = {key: value for key, value in os.environ.items()}
        with tempfile.TemporaryDirectory() as directory:
            env.update(
                {
                    "OPENAI_API_KEY": "test-only",
                    "PYTHONPATH": str(REPO_ROOT),
                    "RAG_SEARCH_BASE_URL": "http://10.0.1.201:8000",
                    "RAG_INTERNAL_API_KEY": "k",
                    "RAG_DB_PATH": str(Path(directory) / "chroma"),
                }
            )
            code = (
                "import sys, json\n"
                "import call_server.main as m\n"
                "from fastapi.testclient import TestClient\n"
                "health = TestClient(m.app).get('/health').json()\n"
                "print(json.dumps({'services': 'model_training.services' in sys.modules,"
                " 'chromadb': 'chromadb' in sys.modules, 'rag': health.get('ragSearch')}))\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", code], cwd=directory, env=env,
                capture_output=True, text=True, timeout=120,
            )
            store_created = (Path(directory) / "chroma").exists()
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        last = result.stdout.strip().splitlines()[-1]
        self.assertIn('"services": false', last)
        self.assertIn('"chromadb": false', last)
        self.assertIn('"rag": "remote"', last)
        self.assertFalse(store_created)


if __name__ == "__main__":
    unittest.main()
