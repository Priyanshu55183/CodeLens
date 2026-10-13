"""
Symbol definition finder (Phase 4).

Design decisions:
- Two-source lookup: We query BOTH ChromaDB (semantic search + metadata post-filter
  on symbol_name) and the BM25 JSON store (direct record scan on symbol_name field).
  ChromaDB gives us semantic proximity; BM25 gives us exact record access without
  an additional embedding call. Together they cover: exact matches, partial matches
  (e.g. "add" finds "add_chunks"), and cross-file duplicates.
- Fuzzy matching: We support substring and case-insensitive symbol name matching so
  users don't need to remember exact capitalisation ("optionParser" finds "OptionParser").
- symbol_type filter: Optional. If supplied, restricts results to "function", "class",
  "method", "interface", or "type". Useful when a name is overloaded (e.g. a class
  and a function both named "command").
- Deduplication by chunk_id: Both sources may return the same chunk. We merge on
  chunk_id, keeping the ChromaDB version (has real vector distance) when both appear.
- Result ordering:
    1. Exact symbol_name matches (case-insensitive) rank before partial matches.
    2. Within each group: ascending vector distance (closer = better).
    3. BM25-only hits (no vector distance) appear after all vector hits.
- Graceful failure: If ChromaDB collection is empty or BM25 has no index, we still
  return results from whichever source is available.

This enables the `codelens find <repo> <symbol>` CLI command added in Phase 4.
"""

import logging

from codelens.app.retrieval.bm25 import BM25Index
from codelens.app.retrieval.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)

# Sentinel distance for BM25-only hits that have no vector distance.
# High value so they sort after real vector hits in the merged list.
_BM25_ONLY_DISTANCE = 999.0


class SymbolFinder:
    """Find symbol definitions (functions, classes, methods) by name.

    Usage:
        finder = SymbolFinder(vector_store, bm25_index)
        results = finder.find("pallets", "click", "option", symbol_type="function")
    """

    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index):
        """Initialize the symbol finder.

        Args:
            vector_store: VectorStore instance for metadata-filtered semantic search.
            bm25_index: BM25Index instance for direct record scanning.
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def find(
        self,
        owner: str,
        repo: str,
        symbol_name: str,
        symbol_type: str | None = None,
        top_k: int = 10,
        fuzzy: bool = True,
    ) -> list[RetrievedChunk]:
        """Find symbol definitions in the indexed repo.

        Args:
            owner: Repo owner.
            repo: Repo name.
            symbol_name: The symbol name to look up (function, class, method name).
            symbol_type: Optional type filter: "function", "class", "method",
                         "interface", "type". None returns all types.
            top_k: Maximum number of results to return.
            fuzzy: If True, includes substring matches (e.g. "add" matches "add_chunks").
                   If False, only exact case-insensitive matches are returned.

        Returns:
            List of RetrievedChunk sorted by: exact match first, then vector
            distance ascending. Returns [] if nothing found.
        """
        query_lower = symbol_name.strip().lower()
        if not query_lower:
            return []

        # ── Source 1: ChromaDB metadata-filtered semantic search ─────────────
        # Embed the symbol name and query — chunks whose context headers include
        # the symbol name will have close embeddings. Post-filter by metadata.
        vector_hits = self._vector_symbol_search(
            owner, repo, symbol_name, symbol_type, top_k=top_k * 3
        )

        # ── Source 2: BM25 record scan ───────────────────────────────────────
        # Directly iterate the in-memory doc_records and match symbol_name field.
        # Zero embedding cost; O(N) scan which is fast for typical repos.
        bm25_hits = self._bm25_symbol_scan(
            owner, repo, query_lower, symbol_type, fuzzy=fuzzy
        )

        # ── Merge, dedup, rank ───────────────────────────────────────────────
        merged = self._merge(vector_hits, bm25_hits, query_lower, top_k)

        logger.info(
            "SymbolFinder '%s' (type=%s, fuzzy=%s) -> %d results for %s/%s",
            symbol_name,
            symbol_type or "any",
            fuzzy,
            len(merged),
            owner,
            repo,
        )
        return merged

    # ── Private helpers ──────────────────────────────────────────────────────

    def _vector_symbol_search(
        self,
        owner: str,
        repo: str,
        symbol_name: str,
        symbol_type: str | None,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Query ChromaDB with symbol name, post-filter by symbol_name metadata."""
        try:
            chunks = self.vector_store.query(owner, repo, symbol_name, top_k=top_k)
        except Exception as e:
            logger.warning("VectorStore.query failed during symbol search: %s", e)
            return []

        query_lower = symbol_name.strip().lower()
        filtered: list[RetrievedChunk] = []
        for chunk in chunks:
            name_lower = chunk.symbol_name.lower()
            if not name_lower:
                continue
            # Accept exact or substring match
            if name_lower == query_lower or query_lower in name_lower:
                if symbol_type is None or chunk.symbol_type.lower() == symbol_type.lower():
                    filtered.append(chunk)

        return filtered

    def _bm25_symbol_scan(
        self,
        owner: str,
        repo: str,
        query_lower: str,
        symbol_type: str | None,
        fuzzy: bool,
    ) -> list[RetrievedChunk]:
        """Scan BM25 doc_records for chunks whose symbol_name matches the query."""
        if not self.bm25_index._ensure_loaded(owner, repo):
            logger.debug("BM25 index not found for %s/%s; skipping symbol scan", owner, repo)
            return []

        repo_key = f"{owner}/{repo}".lower()
        doc_records = self.bm25_index._cache[repo_key]["doc_records"]

        hits: list[RetrievedChunk] = []
        for rec in doc_records:
            sname = rec.get("symbol_name", "").lower()
            if not sname:
                continue

            exact_match = sname == query_lower
            fuzzy_match = fuzzy and query_lower in sname
            if not (exact_match or fuzzy_match):
                continue

            if symbol_type and rec.get("symbol_type", "").lower() != symbol_type.lower():
                continue

            hits.append(
                RetrievedChunk(
                    chunk_id=rec["chunk_id"],
                    content=rec["content"],
                    file_path=rec["file_path"],
                    start_line=rec["start_line"],
                    end_line=rec["end_line"],
                    language=rec["language"],
                    distance=_BM25_ONLY_DISTANCE,
                    symbol_name=rec.get("symbol_name", ""),
                    symbol_type=rec.get("symbol_type", ""),
                    parent_class=rec.get("parent_class", ""),
                    imports=rec.get("imports", ""),
                )
            )

        return hits

    def _merge(
        self,
        vector_hits: list[RetrievedChunk],
        bm25_hits: list[RetrievedChunk],
        query_lower: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Merge hits from both sources, deduplicate by chunk_id, sort, and cap."""
        # BM25 first so vector hits overwrite with real distance on conflict
        seen: dict[str, RetrievedChunk] = {}
        for chunk in bm25_hits:
            seen[chunk.chunk_id] = chunk
        for chunk in vector_hits:
            seen[chunk.chunk_id] = chunk

        def sort_key(c: RetrievedChunk) -> tuple:
            # Exact match (0) before fuzzy/partial (1)
            exact = int(c.symbol_name.lower() != query_lower)
            return (exact, c.distance)

        ranked = sorted(seen.values(), key=sort_key)
        return ranked[:top_k]
