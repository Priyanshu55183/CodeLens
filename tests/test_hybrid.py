"""
Tests for HybridRetriever and Reciprocal Rank Fusion (Phase 3).
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from codelens.app.retrieval.vector_store import RetrievedChunk, VectorStore
from codelens.app.retrieval.bm25 import BM25Index
from codelens.app.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


def make_chunk(cid: str, path: str, dist: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        content=f"content of {cid}",
        file_path=path,
        start_line=1,
        end_line=10,
        language="python",
        distance=dist,
    )


class TestReciprocalRankFusion:
    """Test RRF algorithm logic."""

    def test_item_in_both_ranks_higher_than_single_modality(self):
        # Chunk A is #1 in dense, #1 in BM25
        # Chunk B is #2 in dense, not in BM25
        # Chunk C is not in dense, #2 in BM25
        chunk_a = make_chunk("A", "a.py")
        chunk_b = make_chunk("B", "b.py")
        chunk_c = make_chunk("C", "c.py")

        dense = [chunk_a, chunk_b]
        bm25 = [chunk_a, chunk_c]

        fused = reciprocal_rank_fusion(dense, bm25, k=60, top_k=3)
        assert len(fused) == 3
        # Chunk A must be #1 with the highest score
        assert fused[0].chunk_id == "A"
        assert fused[0].distance > fused[1].distance

    def test_top_k_limiting(self):
        dense = [make_chunk(f"D{i}", f"d{i}.py") for i in range(10)]
        bm25 = [make_chunk(f"B{i}", f"b{i}.py") for i in range(10)]

        fused = reciprocal_rank_fusion(dense, bm25, k=60, top_k=5)
        assert len(fused) == 5

    def test_weighting_favors_heavier_source(self):
        # Chunk D is rank 1 in dense only
        # Chunk B is rank 1 in BM25 only
        chunk_d = make_chunk("D", "d.py")
        chunk_b = make_chunk("B", "b.py")

        # Dense has weight 2.0, BM25 has weight 0.5
        fused = reciprocal_rank_fusion([chunk_d], [chunk_b], k=60, dense_weight=2.0, bm25_weight=0.5)
        assert fused[0].chunk_id == "D"


class TestHybridRetriever:
    """Test HybridRetriever orchestration and fallbacks."""

    def test_vector_mode_delegates_to_vector_store(self):
        mock_vs = MagicMock(spec=VectorStore)
        mock_bm25 = MagicMock(spec=BM25Index)

        expected = [make_chunk("V1", "v1.py")]
        mock_vs.query.return_value = expected

        retriever = HybridRetriever(mock_vs, mock_bm25, default_mode="vector")
        results = retriever.query("owner", "repo", "query", top_k=5)

        assert results == expected
        mock_vs.query.assert_called_once_with("owner", "repo", "query", top_k=5)
        mock_bm25.query.assert_not_called()

    def test_bm25_mode_delegates_to_bm25_index(self):
        mock_vs = MagicMock(spec=VectorStore)
        mock_bm25 = MagicMock(spec=BM25Index)

        expected = [make_chunk("B1", "b1.py")]
        mock_bm25.query.return_value = expected

        retriever = HybridRetriever(mock_vs, mock_bm25, default_mode="bm25")
        results = retriever.query("owner", "repo", "query", top_k=5)

        assert results == expected
        mock_bm25.query.assert_called_once_with("owner", "repo", "query", top_k=5)
        mock_vs.query.assert_not_called()

    def test_hybrid_mode_fuses_candidates(self):
        mock_vs = MagicMock(spec=VectorStore)
        mock_bm25 = MagicMock(spec=BM25Index)

        c1 = make_chunk("C1", "c1.py")
        c2 = make_chunk("C2", "c2.py")

        mock_vs.query.return_value = [c1]
        mock_bm25.query.return_value = [c1, c2]

        retriever = HybridRetriever(mock_vs, mock_bm25, default_mode="hybrid")
        results = retriever.query("owner", "repo", "query", top_k=5)

        assert len(results) == 2
        # C1 was in both, so it must rank first
        assert results[0].chunk_id == "C1"

    def test_fallback_when_bm25_empty(self):
        mock_vs = MagicMock(spec=VectorStore)
        mock_bm25 = MagicMock(spec=BM25Index)

        c1 = make_chunk("V1", "v1.py")
        mock_vs.query.return_value = [c1]
        mock_bm25.query.return_value = []

        retriever = HybridRetriever(mock_vs, mock_bm25, default_mode="hybrid")
        results = retriever.query("owner", "repo", "query", top_k=5)

        assert len(results) == 1
        assert results[0].chunk_id == "V1"

    def test_fallback_when_vector_empty(self):
        mock_vs = MagicMock(spec=VectorStore)
        mock_bm25 = MagicMock(spec=BM25Index)

        c1 = make_chunk("B1", "b1.py")
        mock_vs.query.return_value = []
        mock_bm25.query.return_value = [c1]

        retriever = HybridRetriever(mock_vs, mock_bm25, default_mode="hybrid")
        results = retriever.query("owner", "repo", "query", top_k=5)

        assert len(results) == 1
        assert results[0].chunk_id == "B1"
