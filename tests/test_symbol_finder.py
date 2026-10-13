"""
Tests for SymbolFinder — Phase 4 symbol definition lookup.

Tests cover:
- BM25 record scan: exact match, fuzzy/substring match, type filter, exact-flag
- Merge & dedup: vector hit overwrites BM25 placeholder distance
- Ranking: exact-name match ranks before fuzzy match
- Edge cases: empty query, symbol not found, missing BM25 index
- Integration smoke test with real BM25Index (no ChromaDB needed)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codelens.app.ingestion.chunker import Chunk
from codelens.app.retrieval.bm25 import BM25Index
from codelens.app.retrieval.vector_store import RetrievedChunk, VectorStore
from codelens.app.retrieval.symbol_finder import SymbolFinder, _BM25_ONLY_DISTANCE


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_chunk(
    chunk_id: str,
    file_path: str,
    symbol_name: str,
    symbol_type: str = "function",
    parent_class: str = "",
    content: str = "",
    start_line: int = 1,
    end_line: int = 10,
    language: str = "python",
) -> Chunk:
    return Chunk(
        content=content or f"def {symbol_name}():\n    pass\n",
        raw_content=content or f"def {symbol_name}():\n    pass\n",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        language=language,
        chunk_id=chunk_id,
        symbol_name=symbol_name,
        symbol_type=symbol_type,
        parent_class=parent_class,
    )


def make_retrieved(
    chunk_id: str,
    file_path: str,
    symbol_name: str,
    symbol_type: str = "function",
    parent_class: str = "",
    distance: float = 0.1,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=f"def {symbol_name}(): pass",
        file_path=file_path,
        start_line=1,
        end_line=5,
        language="python",
        distance=distance,
        symbol_name=symbol_name,
        symbol_type=symbol_type,
        parent_class=parent_class,
    )


@pytest.fixture
def click_chunks() -> list[Chunk]:
    """A small fake indexed repo simulating click library symbols."""
    return [
        make_chunk("core.py:1-10", "src/core.py", "Command", "class"),
        make_chunk("core.py:20-30", "src/core.py", "BaseCommand", "class"),
        make_chunk("decorators.py:1-5", "src/decorators.py", "option", "function"),
        make_chunk("decorators.py:10-20", "src/decorators.py", "add_option", "function"),
        make_chunk("utils.py:1-5", "src/utils.py", "echo", "function"),
        make_chunk("utils.py:10-20", "src/utils.py", "echo_error", "function"),
        make_chunk(
            "core.py:50-60", "src/core.py", "invoke",
            symbol_type="method", parent_class="Command",
        ),
    ]


# ── BM25-backed tests (no ChromaDB) ─────────────────────────────────────────

class TestBM25SymbolScan:
    """Test _bm25_symbol_scan directly via full SymbolFinder.find() with mocked VectorStore."""

    def _make_finder(self, tmp_path: Path, chunks: list[Chunk]) -> SymbolFinder:
        bm25 = BM25Index(bm25_dir=tmp_path)
        bm25.add_chunks("pallets", "click", chunks)
        mock_vs = MagicMock(spec=VectorStore)
        mock_vs.query.return_value = []  # No vector hits — pure BM25 scan
        return SymbolFinder(vector_store=mock_vs, bm25_index=bm25)

    def test_exact_match_by_name(self, tmp_path: Path, click_chunks: list[Chunk]):
        finder = self._make_finder(tmp_path, click_chunks)
        results = finder.find("pallets", "click", "Command", fuzzy=False)

        assert len(results) == 1
        assert results[0].symbol_name == "Command"
        assert results[0].file_path == "src/core.py"
        assert results[0].symbol_type == "class"

    def test_fuzzy_substring_match(self, tmp_path: Path, click_chunks: list[Chunk]):
        finder = self._make_finder(tmp_path, click_chunks)
        results = finder.find("pallets", "click", "echo", fuzzy=True)

        # Should return both "echo" and "echo_error"
        names = {r.symbol_name for r in results}
        assert "echo" in names
        assert "echo_error" in names

    def test_exact_flag_excludes_substrings(self, tmp_path: Path, click_chunks: list[Chunk]):
        finder = self._make_finder(tmp_path, click_chunks)
        results = finder.find("pallets", "click", "echo", fuzzy=False)

        assert len(results) == 1
        assert results[0].symbol_name == "echo"

    def test_type_filter_class(self, tmp_path: Path, click_chunks: list[Chunk]):
        finder = self._make_finder(tmp_path, click_chunks)
        results = finder.find("pallets", "click", "Command", symbol_type="class")

        assert all(r.symbol_type == "class" for r in results)
        assert any(r.symbol_name == "Command" for r in results)

    def test_type_filter_excludes_wrong_type(self, tmp_path: Path, click_chunks: list[Chunk]):
        finder = self._make_finder(tmp_path, click_chunks)
        # "Command" is a class; asking for function type should find nothing
        results = finder.find("pallets", "click", "Command", symbol_type="function", fuzzy=False)
        assert results == []

    def test_method_with_parent_class(self, tmp_path: Path, click_chunks: list[Chunk]):
        finder = self._make_finder(tmp_path, click_chunks)
        results = finder.find("pallets", "click", "invoke", symbol_type="method")

        assert len(results) == 1
        assert results[0].symbol_name == "invoke"
        assert results[0].parent_class == "Command"
        assert results[0].symbol_type == "method"

    def test_no_results_for_unknown_symbol(self, tmp_path: Path, click_chunks: list[Chunk]):
        finder = self._make_finder(tmp_path, click_chunks)
        results = finder.find("pallets", "click", "nonexistent_quantum_xyz")
        assert results == []

    def test_empty_query_returns_empty(self, tmp_path: Path, click_chunks: list[Chunk]):
        finder = self._make_finder(tmp_path, click_chunks)
        assert finder.find("pallets", "click", "") == []
        assert finder.find("pallets", "click", "   ") == []

    def test_missing_bm25_index_returns_empty(self, tmp_path: Path):
        bm25 = BM25Index(bm25_dir=tmp_path)  # Empty, nothing indexed
        mock_vs = MagicMock(spec=VectorStore)
        mock_vs.query.return_value = []
        finder = SymbolFinder(vector_store=mock_vs, bm25_index=bm25)

        results = finder.find("pallets", "click", "Command")
        assert results == []

    def test_top_k_limits_results(self, tmp_path: Path, click_chunks: list[Chunk]):
        finder = self._make_finder(tmp_path, click_chunks)
        # "o" substring matches: option, add_option, echo, echo_error, Command(no), ...
        results = finder.find("pallets", "click", "o", fuzzy=True, top_k=2)
        assert len(results) <= 2


# ── Merge & ranking tests ────────────────────────────────────────────────────

class TestSymbolFinderMerge:
    """Test merge, dedup and ranking logic in SymbolFinder."""

    def test_exact_match_ranks_before_fuzzy(self, tmp_path: Path):
        chunks = [
            make_chunk("a.py:1-5", "a.py", "echo_extended", "function"),
            make_chunk("b.py:1-5", "b.py", "echo", "function"),
        ]
        bm25 = BM25Index(bm25_dir=tmp_path)
        bm25.add_chunks("owner", "repo", chunks)
        mock_vs = MagicMock(spec=VectorStore)
        mock_vs.query.return_value = []

        finder = SymbolFinder(vector_store=mock_vs, bm25_index=bm25)
        results = finder.find("owner", "repo", "echo", fuzzy=True)

        # "echo" (exact) must come before "echo_extended" (fuzzy)
        assert results[0].symbol_name == "echo"
        assert results[1].symbol_name == "echo_extended"

    def test_vector_distance_overwrites_bm25_placeholder(self, tmp_path: Path):
        """When a chunk appears in both BM25 scan and vector results, keep real distance."""
        chunk = make_chunk("core.py:1-5", "src/core.py", "Command", "class")
        bm25 = BM25Index(bm25_dir=tmp_path)
        bm25.add_chunks("owner", "repo", [chunk])

        real_distance = 0.05
        vector_result = make_retrieved("core.py:1-5", "src/core.py", "Command", distance=real_distance)
        mock_vs = MagicMock(spec=VectorStore)
        mock_vs.query.return_value = [vector_result]

        finder = SymbolFinder(vector_store=mock_vs, bm25_index=bm25)
        results = finder.find("owner", "repo", "Command", fuzzy=False)

        assert len(results) == 1
        # Real vector distance, not the BM25 sentinel
        assert results[0].distance == real_distance
        assert results[0].distance != _BM25_ONLY_DISTANCE

    def test_bm25_only_hit_appears_with_sentinel_distance(self, tmp_path: Path):
        """Hits only found by BM25 scan have the sentinel _BM25_ONLY_DISTANCE."""
        chunk = make_chunk("utils.py:1-5", "src/utils.py", "echo", "function")
        bm25 = BM25Index(bm25_dir=tmp_path)
        bm25.add_chunks("owner", "repo", [chunk])

        mock_vs = MagicMock(spec=VectorStore)
        mock_vs.query.return_value = []  # Vector returns nothing

        finder = SymbolFinder(vector_store=mock_vs, bm25_index=bm25)
        results = finder.find("owner", "repo", "echo", fuzzy=False)

        assert len(results) == 1
        assert results[0].distance == _BM25_ONLY_DISTANCE

    def test_deduplication_by_chunk_id(self, tmp_path: Path):
        """Same chunk_id from both BM25 and vector should appear only once."""
        chunk = make_chunk("core.py:1-5", "src/core.py", "Command", "class")
        bm25 = BM25Index(bm25_dir=tmp_path)
        bm25.add_chunks("owner", "repo", [chunk])

        vector_result = make_retrieved("core.py:1-5", "src/core.py", "Command", distance=0.1)
        mock_vs = MagicMock(spec=VectorStore)
        mock_vs.query.return_value = [vector_result]

        finder = SymbolFinder(vector_store=mock_vs, bm25_index=bm25)
        results = finder.find("owner", "repo", "Command")

        assert len(results) == 1

    def test_case_insensitive_matching(self, tmp_path: Path):
        """Symbol lookup is case-insensitive."""
        chunk = make_chunk("core.py:1-5", "src/core.py", "OptionParser", "class")
        bm25 = BM25Index(bm25_dir=tmp_path)
        bm25.add_chunks("owner", "repo", [chunk])
        mock_vs = MagicMock(spec=VectorStore)
        mock_vs.query.return_value = []

        finder = SymbolFinder(vector_store=mock_vs, bm25_index=bm25)

        # All of these should find OptionParser
        for query in ["optionparser", "OPTIONPARSER", "OptionParser"]:
            results = finder.find("owner", "repo", query, fuzzy=False)
            assert len(results) == 1, f"Expected 1 result for query '{query}'"
            assert results[0].symbol_name == "OptionParser"
