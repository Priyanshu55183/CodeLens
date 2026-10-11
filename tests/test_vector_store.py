"""
Tests for VectorStore and RetrievedChunk.
"""

from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from codelens.app.retrieval.vector_store import VectorStore, RetrievedChunk
from codelens.app.ingestion.chunker import Chunk


class TestRetrievedChunk:
    """Test RetrievedChunk dataclass and citation formatting."""

    def test_citation_without_symbol(self):
        c = RetrievedChunk(
            chunk_id="test.py:1-10",
            content="print('hello')",
            file_path="src/app.py",
            start_line=1,
            end_line=10,
            language="python",
            distance=0.12,
        )
        assert c.citation == "[src/app.py:1-10]"

    def test_citation_with_function_symbol(self):
        c = RetrievedChunk(
            chunk_id="test.py:1-10",
            content="def greet(): pass",
            file_path="src/app.py",
            start_line=1,
            end_line=10,
            language="python",
            distance=0.05,
            symbol_name="greet",
            symbol_type="function",
        )
        assert c.citation == "[src/app.py:1-10 (greet)]"

    def test_citation_with_method_and_parent_class(self):
        c = RetrievedChunk(
            chunk_id="test.py:15-30",
            content="def run(self): pass",
            file_path="src/runner.py",
            start_line=15,
            end_line=30,
            language="python",
            distance=0.08,
            symbol_name="run",
            symbol_type="method",
            parent_class="Runner",
        )
        assert c.citation == "[src/runner.py:15-30 (Runner.run)]"


class TestVectorStoreCollectionNaming:
    """Test ChromaDB collection name sanitation."""

    @patch("codelens.app.retrieval.vector_store.chromadb.PersistentClient")
    @patch("codelens.app.retrieval.vector_store.OpenAIEmbeddingFunction")
    def test_collection_name_sanitization(self, mock_embed, mock_client, tmp_path: Path):
        store = VectorStore(chroma_dir=tmp_path, api_key="sk-test")

        # Standard owner/repo
        assert store._collection_name("pallets", "click") == "pallets__click"

        # Special characters sanitized
        assert store._collection_name("my.org", "repo@v1") == "my_org__repo_v1"

        # Short names padded to >= 3 chars
        short_name = store._collection_name("a", "b")
        assert len(short_name) >= 3

        # Length capped at 63 characters
        long_owner = "a" * 40
        long_repo = "b" * 40
        assert len(store._collection_name(long_owner, long_repo)) <= 63
