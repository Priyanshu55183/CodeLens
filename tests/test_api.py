"""
Tests for the FastAPI REST API (Phase 6).

Design notes:
- We use FastAPI's TestClient (backed by httpx) for all tests.
  TestClient runs the app in-process without a real server.
- All heavy I/O is mocked: index_repo, VectorStore, BM25Index, ask_llm.
  We test routing, serialisation, and error handling — not the pipeline.
- The fixtures create a fresh app instance per test to avoid shared state.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from codelens.app.api.main import create_app
from codelens.app.ingestion.indexer import IndexResult
from codelens.app.retrieval.vector_store import RetrievedChunk


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """Return a TestClient for a fresh app instance."""
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


def _make_chunk(n: int = 1) -> RetrievedChunk:
    """Create a dummy RetrievedChunk for use in mocked return values."""
    return RetrievedChunk(
        chunk_id=f"chunk-{n}",
        content=f"def foo_{n}(): pass",
        file_path=f"src/module_{n}.py",
        start_line=10 * n,
        end_line=10 * n + 5,
        language="python",
        distance=0.123,
        symbol_name=f"foo_{n}",
        symbol_type="function",
        parent_class="",
        imports="",
    )


# ── GET /health ───────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_required_fields(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "provider" in data
        assert "model" in data
        assert "retrieval_mode" in data

    def test_root_redirects_to_docs(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 302, 307, 308)
        assert resp.headers["location"].endswith("/docs")


# ── POST /repos/index ─────────────────────────────────────────────────────────

class TestIndexEndpoint:
    VALID_URL = "https://github.com/pallets/click"

    def _mock_index_result(self):
        return IndexResult(
            owner="pallets",
            repo="click",
            commit_sha="abc1234def5678",
            files_indexed=42,
            chunks_created=310,
            duration_seconds=12.5,
        )

    def test_valid_index_returns_201(self, client):
        with patch("codelens.app.api.routes.index_repo", return_value=self._mock_index_result()):
            resp = client.post("/repos/index", json={"repo_url": self.VALID_URL})
        assert resp.status_code == 201

    def test_valid_index_response_schema(self, client):
        with patch("codelens.app.api.routes.index_repo", return_value=self._mock_index_result()):
            data = client.post("/repos/index", json={"repo_url": self.VALID_URL}).json()
        assert data["owner"] == "pallets"
        assert data["repo"] == "click"
        assert data["files_indexed"] == 42
        assert data["chunks_created"] == 310
        assert "message" in data

    def test_invalid_url_returns_400(self, client):
        resp = client.post("/repos/index", json={"repo_url": "not-a-url"})
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_repo_url"

    def test_missing_repo_url_returns_422(self, client):
        resp = client.post("/repos/index", json={})
        assert resp.status_code == 422

    def test_indexing_failure_returns_500(self, client):
        with patch("codelens.app.api.routes.index_repo", side_effect=RuntimeError("disk full")):
            resp = client.post("/repos/index", json={"repo_url": self.VALID_URL})
        assert resp.status_code == 500
        assert resp.json()["error"] == "indexing_failed"


# ── POST /repos/ask ───────────────────────────────────────────────────────────

class TestAskEndpoint:
    VALID_BODY = {
        "repo_url": "https://github.com/pallets/click",
        "question": "How does Click parse arguments?",
    }

    def _setup_retriever(self, mock_cls, chunks=None):
        """Helper to mock HybridRetriever and its query method."""
        if chunks is None:
            chunks = [_make_chunk(1), _make_chunk(2)]
        instance = mock_cls.return_value
        instance.query.return_value = chunks
        return instance

    def test_valid_ask_returns_200(self, client):
        chunks = [_make_chunk(1)]
        with (
            patch("codelens.app.api.routes.HybridRetriever") as MockHR,
            patch("codelens.app.api.routes.VectorStore"),
            patch("codelens.app.api.routes.BM25Index"),
            patch("codelens.app.api.routes.ask_llm", return_value="The answer."),
        ):
            self._setup_retriever(MockHR, chunks)
            resp = client.post("/repos/ask", json=self.VALID_BODY)
        assert resp.status_code == 200

    def test_ask_response_schema(self, client):
        chunks = [_make_chunk(1)]
        with (
            patch("codelens.app.api.routes.HybridRetriever") as MockHR,
            patch("codelens.app.api.routes.VectorStore"),
            patch("codelens.app.api.routes.BM25Index"),
            patch("codelens.app.api.routes.ask_llm", return_value="The answer."),
        ):
            self._setup_retriever(MockHR, chunks)
            data = client.post("/repos/ask", json=self.VALID_BODY).json()

        assert data["answer"] == "The answer."
        assert data["chunks_retrieved"] == 1
        assert len(data["sources"]) == 1
        assert data["sources"][0]["file_path"] == "src/module_1.py"
        assert "citation" in data["sources"][0]

    def test_no_indexed_data_returns_404(self, client):
        with (
            patch("codelens.app.api.routes.HybridRetriever") as MockHR,
            patch("codelens.app.api.routes.VectorStore"),
            patch("codelens.app.api.routes.BM25Index"),
        ):
            self._setup_retriever(MockHR, [])  # empty list = not indexed
            resp = client.post("/repos/ask", json=self.VALID_BODY)
        assert resp.status_code == 404
        assert resp.json()["error"] == "repo_not_indexed"

    def test_invalid_mode_returns_400(self, client):
        body = {**self.VALID_BODY, "retrieval_mode": "magic"}
        resp = client.post("/repos/ask", json=body)
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_retrieval_mode"

    def test_top_k_out_of_range_returns_422(self, client):
        body = {**self.VALID_BODY, "top_k": 0}
        resp = client.post("/repos/ask", json=body)
        assert resp.status_code == 422

    def test_llm_failure_returns_500(self, client):
        chunks = [_make_chunk(1)]
        with (
            patch("codelens.app.api.routes.HybridRetriever") as MockHR,
            patch("codelens.app.api.routes.VectorStore"),
            patch("codelens.app.api.routes.BM25Index"),
            patch("codelens.app.api.routes.ask_llm", side_effect=RuntimeError("API down")),
        ):
            self._setup_retriever(MockHR, chunks)
            resp = client.post("/repos/ask", json=self.VALID_BODY)
        assert resp.status_code == 500
        assert resp.json()["error"] == "llm_failed"


# ── POST /repos/find ─────────────────────────────────────────────────────────

class TestFindEndpoint:
    VALID_BODY = {
        "repo_url": "https://github.com/pallets/click",
        "symbol": "OptionParser",
    }

    def test_valid_find_returns_200(self, client):
        chunks = [_make_chunk(1)]
        with (
            patch("codelens.app.api.routes.SymbolFinder") as MockSF,
            patch("codelens.app.api.routes.VectorStore"),
            patch("codelens.app.api.routes.BM25Index"),
        ):
            MockSF.return_value.find.return_value = chunks
            resp = client.post("/repos/find", json=self.VALID_BODY)
        assert resp.status_code == 200

    def test_find_response_schema(self, client):
        chunks = [_make_chunk(1), _make_chunk(2)]
        with (
            patch("codelens.app.api.routes.SymbolFinder") as MockSF,
            patch("codelens.app.api.routes.VectorStore"),
            patch("codelens.app.api.routes.BM25Index"),
        ):
            MockSF.return_value.find.return_value = chunks
            data = client.post("/repos/find", json=self.VALID_BODY).json()

        assert data["symbol"] == "OptionParser"
        assert data["total_found"] == 2
        assert len(data["results"]) == 2

    def test_symbol_not_found_returns_404(self, client):
        with (
            patch("codelens.app.api.routes.SymbolFinder") as MockSF,
            patch("codelens.app.api.routes.VectorStore"),
            patch("codelens.app.api.routes.BM25Index"),
        ):
            MockSF.return_value.find.return_value = []
            resp = client.post("/repos/find", json=self.VALID_BODY)
        assert resp.status_code == 404
        assert resp.json()["error"] == "symbol_not_found"

    def test_exact_match_flag_is_respected(self, client):
        """fuzzy=False should be passed to SymbolFinder.find()."""
        body = {**self.VALID_BODY, "fuzzy": False}
        chunks = [_make_chunk(1)]
        with (
            patch("codelens.app.api.routes.SymbolFinder") as MockSF,
            patch("codelens.app.api.routes.VectorStore"),
            patch("codelens.app.api.routes.BM25Index"),
        ):
            MockSF.return_value.find.return_value = chunks
            client.post("/repos/find", json=body)
            call_kwargs = MockSF.return_value.find.call_args[1]
            assert call_kwargs["fuzzy"] is False


# ── GET /repos/status ─────────────────────────────────────────────────────────

class TestStatusEndpoint:
    def test_indexed_repo_returns_indexed_true(self, client):
        with patch("codelens.app.api.routes.VectorStore") as MockVS:
            MockVS.return_value.count.return_value = 150
            resp = client.get(
                "/repos/status",
                params={"repo_url": "https://github.com/pallets/click"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["indexed"] is True
        assert data["chunk_count"] == 150

    def test_unindexed_repo_returns_indexed_false(self, client):
        with patch("codelens.app.api.routes.VectorStore") as MockVS:
            MockVS.return_value.count.return_value = 0
            resp = client.get(
                "/repos/status",
                params={"repo_url": "https://github.com/pallets/click"},
            )
        data = resp.json()
        assert data["indexed"] is False
        assert data["chunk_count"] == 0

    def test_invalid_url_returns_400(self, client):
        resp = client.get("/repos/status", params={"repo_url": "not-a-url"})
        assert resp.status_code == 400
