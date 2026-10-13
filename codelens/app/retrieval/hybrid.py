"""
Hybrid retrieval combining Dense Vector Search and Sparse BM25 (Phase 3).

Design decisions:
- Reciprocal Rank Fusion (RRF): We fuse rankings rather than raw scores.
  Dense distance (cosine/L2) and BM25 scores have completely different scales
  and distributions. Trying to normalize and add raw scores is fragile and sensitive
  to outliers. RRF uses ONLY ranks:
      RRF_score(d) = sum_m ( w_m / (k + rank_m(d)) )
  where k=60 (standard constant). This is robust, parameter-free, and proven
  in information retrieval benchmarks (Cormack et al., 2009).
- Oversampling before fusion: We fetch top 2x candidates from both vector search
  and BM25, fuse them with RRF, and take the top K. This ensures items that are
  strong in one modality but missed in the top K of another still get considered.
- Graceful degradation: If BM25 has no index or no matches, it smoothly falls back
  to dense vector results (and vice versa).
"""

import logging
from dataclasses import replace

from codelens.app.retrieval.bm25 import BM25Index
from codelens.app.retrieval.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    k: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    top_k: int = 10,
) -> list[RetrievedChunk]:
    """Combine dense and BM25 search rankings using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score(chunk) = dense_weight / (k + rank_dense) + bm25_weight / (k + rank_bm25)

    Args:
        dense_results: RetrievedChunk list from vector search (ordered best-first).
        bm25_results: RetrievedChunk list from BM25 search (ordered best-first).
        k: Smoothing constant to control impact of top vs lower ranks (default 60).
        dense_weight: Relative weight for dense vector retrieval.
        bm25_weight: Relative weight for sparse BM25 retrieval.
        top_k: Maximum number of merged results to return.

    Returns:
        List of fused RetrievedChunk instances, sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    # Accumulate dense ranks (1-indexed)
    for rank, chunk in enumerate(dense_results, start=1):
        cid = chunk.chunk_id
        scores[cid] = scores.get(cid, 0.0) + (dense_weight / (k + rank))
        chunk_map[cid] = chunk

    # Accumulate BM25 ranks (1-indexed)
    for rank, chunk in enumerate(bm25_results, start=1):
        cid = chunk.chunk_id
        scores[cid] = scores.get(cid, 0.0) + (bm25_weight / (k + rank))
        if cid not in chunk_map:
            chunk_map[cid] = chunk

    # Sort chunk IDs by descending RRF score
    sorted_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    top_ids = sorted_chunk_ids[:top_k]

    fused_results: list[RetrievedChunk] = []
    for cid in top_ids:
        original = chunk_map[cid]
        # Store the fused RRF score in distance for downstream inspection
        fused_results.append(
            replace(
                original,
                distance=round(scores[cid], 6),
            )
        )

    logger.debug(
        "RRF fused %d dense + %d BM25 candidates into %d final chunks",
        len(dense_results),
        len(bm25_results),
        len(fused_results),
    )
    return fused_results


class HybridRetriever:
    """Orchestrates hybrid retrieval combining VectorStore and BM25Index."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        default_mode: str = "hybrid",
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        """Initialize the hybrid retriever.

        Args:
            vector_store: VectorStore instance for dense semantic search.
            bm25_index: BM25Index instance for sparse keyword search.
            default_mode: Default retrieval strategy ("hybrid", "vector", "bm25").
            rrf_k: Smoothing constant for RRF.
            dense_weight: Weight for dense vector search in RRF.
            bm25_weight: Weight for sparse BM25 search in RRF.
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.default_mode = default_mode
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def query(
        self,
        owner: str,
        repo: str,
        query_text: str,
        top_k: int = 10,
        mode: str | None = None,
    ) -> list[RetrievedChunk]:
        """Search the codebase using the selected retrieval mode.

        Args:
            owner: Repo owner.
            repo: Repo name.
            query_text: User question or code identifier.
            top_k: Number of chunks to return.
            mode: Override mode ("hybrid", "vector", "bm25"). If None, uses default_mode.

        Returns:
            List of RetrievedChunk instances.
        """
        strategy = (mode or self.default_mode).lower()

        if strategy == "vector":
            return self.vector_store.query(owner, repo, query_text, top_k=top_k)

        if strategy == "bm25":
            return self.bm25_index.query(owner, repo, query_text, top_k=top_k)

        # Hybrid strategy: oversample then fuse
        candidate_k = max(top_k * 2, 20)

        dense_chunks = self.vector_store.query(owner, repo, query_text, top_k=candidate_k)
        bm25_chunks = self.bm25_index.query(owner, repo, query_text, top_k=candidate_k)

        # Fallback if one search modality has no results
        if not bm25_chunks:
            return dense_chunks[:top_k]
        if not dense_chunks:
            return bm25_chunks[:top_k]

        return reciprocal_rank_fusion(
            dense_results=dense_chunks,
            bm25_results=bm25_chunks,
            k=self.rrf_k,
            dense_weight=self.dense_weight,
            bm25_weight=self.bm25_weight,
            top_k=top_k,
        )
