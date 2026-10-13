"""
Tests for BM25 sparse keyword indexer and code tokenizer (Phase 3).
"""

from pathlib import Path
import pytest

from codelens.app.ingestion.chunker import Chunk
from codelens.app.retrieval.bm25 import BM25Index, tokenize_code


class TestTokenizeCode:
    """Test code-aware tokenization."""

    def test_snake_case_splitting(self):
        tokens = tokenize_code("def parse_cli_args(repo_url):")
        assert "parse_cli_args" in tokens
        assert "parse" in tokens
        assert "cli" in tokens
        assert "args" in tokens
        assert "repo_url" in tokens
        assert "repo" in tokens
        assert "url" in tokens

    def test_camel_case_splitting(self):
        tokens = tokenize_code("class OptionParser(BaseCommand):")
        assert "optionparser" in tokens
        assert "option" in tokens
        assert "parser" in tokens
        assert "basecommand" in tokens
        assert "base" in tokens
        assert "command" in tokens

    def test_numbers_and_symbols(self):
        tokens = tokenize_code("port_8080 = 8080")
        assert "port_8080" in tokens
        assert "port" in tokens
        assert "8080" in tokens

    def test_empty_string(self):
        assert tokenize_code("") == []


class TestBM25Index:
    """Test BM25 indexing and querying."""

    @pytest.fixture
    def sample_chunks(self) -> list[Chunk]:
        c1 = Chunk(
            content="# File: core.py\nclass Command:\n    def invoke(self): pass\n",
            raw_content="class Command:\n    def invoke(self): pass\n",
            file_path="src/click/core.py",
            start_line=1,
            end_line=3,
            language="python",
            chunk_id="core.py:1-3",
            symbol_name="Command",
            symbol_type="class",
        )
        c2 = Chunk(
            content="# File: decorators.py\ndef option(*args):\n    return decorator\n",
            raw_content="def option(*args):\n    return decorator\n",
            file_path="src/click/decorators.py",
            start_line=1,
            end_line=3,
            language="python",
            chunk_id="decorators.py:1-3",
            symbol_name="option",
            symbol_type="function",
        )
        c3 = Chunk(
            content="# File: utils.py\ndef echo(message):\n    print(message)\n",
            raw_content="def echo(message):\n    print(message)\n",
            file_path="src/click/utils.py",
            start_line=1,
            end_line=3,
            language="python",
            chunk_id="utils.py:1-3",
            symbol_name="echo",
            symbol_type="function",
        )
        return [c1, c2, c3]

    def test_add_and_query_exact_symbol(self, tmp_path: Path, sample_chunks: list[Chunk]):
        index = BM25Index(bm25_dir=tmp_path)
        count = index.add_chunks("pallets", "click", sample_chunks)
        assert count == 3

        # Query for "Command"
        results = index.query("pallets", "click", "Command", top_k=2)
        assert len(results) >= 1
        assert results[0].file_path == "src/click/core.py"
        assert results[0].symbol_name == "Command"
        assert results[0].distance > 0.0  # BM25 score

    def test_query_decorator_option(self, tmp_path: Path, sample_chunks: list[Chunk]):
        index = BM25Index(bm25_dir=tmp_path)
        index.add_chunks("pallets", "click", sample_chunks)

        results = index.query("pallets", "click", "option decorator", top_k=2)
        assert len(results) >= 1
        assert results[0].file_path == "src/click/decorators.py"

    def test_query_no_matches_returns_empty(self, tmp_path: Path, sample_chunks: list[Chunk]):
        index = BM25Index(bm25_dir=tmp_path)
        index.add_chunks("pallets", "click", sample_chunks)

        results = index.query("pallets", "click", "quantum physics superposition", top_k=5)
        assert results == []

    def test_persistence_and_reloading(self, tmp_path: Path, sample_chunks: list[Chunk]):
        index1 = BM25Index(bm25_dir=tmp_path)
        index1.add_chunks("pallets", "click", sample_chunks)

        # Create a fresh index instance pointing to the same directory
        index2 = BM25Index(bm25_dir=tmp_path)
        results = index2.query("pallets", "click", "echo print message", top_k=1)
        assert len(results) == 1
        assert results[0].file_path == "src/click/utils.py"

    def test_delete_index(self, tmp_path: Path, sample_chunks: list[Chunk]):
        index = BM25Index(bm25_dir=tmp_path)
        index.add_chunks("pallets", "click", sample_chunks)

        index.delete_index("pallets", "click")
        assert index.query("pallets", "click", "Command") == []
