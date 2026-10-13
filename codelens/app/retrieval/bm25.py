"""
BM25 sparse keyword indexer for code chunks (Phase 3).

Design decisions:
- Code Tokenizer: Splits on identifiers, snake_case, camelCase, and numbers.
  Emits both exact symbols and subwords so both exact variable names and
  natural language descriptions match.
- Lucene BM25 Formulation: Uses Robertson BM25 with Lucene-style IDF:
    IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))
  This avoids negative or zero IDF scores on small corpuses while maintaining
  exact BM25 term weighting (k1=1.5, b=0.75).
- Persistent JSON Index: Chunks are stored in a simple JSON file per repo,
  allowing instant loading across restarts without re-chunking or re-embedding.
"""

import json
import logging
import math
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from codelens.app.ingestion.chunker import Chunk
from codelens.app.retrieval.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)


def tokenize_code(text: str) -> list[str]:
    """Tokenize source code or queries for BM25 keyword matching.

    Extracts:
    1. Full lowercased tokens (e.g. "calculate_sum", "OptionParser")
    2. Underscore-separated parts (e.g. "calculate", "sum")
    3. CamelCase / PascalCase subwords (e.g. "option", "parser")

    Args:
        text: Code string or search query.

    Returns:
        List of string tokens.
    """
    if not text:
        return []

    raw_tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+", text)
    tokens: list[str] = []

    for tok in raw_tokens:
        lower = tok.lower()
        tokens.append(lower)

        # Split snake_case
        if "_" in tok:
            for sub in tok.split("_"):
                if sub:
                    tokens.append(sub.lower())

        # Split camelCase / PascalCase
        subwords = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)", tok)
        if len(subwords) > 1:
            for sub in subwords:
                tokens.append(sub.lower())

    return tokens


class BM25Index:
    """Persistent BM25 index for repository code chunks."""

    def __init__(self, bm25_dir: Path, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25 index manager.

        Args:
            bm25_dir: Directory where index files are stored.
            k1: Term frequency saturation parameter (default 1.5).
            b: Document length normalization parameter (default 0.75).
        """
        self.bm25_dir = bm25_dir
        self.bm25_dir.mkdir(parents=True, exist_ok=True)
        self.k1 = k1
        self.b = b

        # In-memory cache for loaded repo indexes:
        # repo_key -> dict with doc_records, corpus_tokens, doc_freqs, idf, avgdl
        self._cache: dict[str, dict] = {}

    def _index_path(self, owner: str, repo: str) -> Path:
        """Get the path to a repo's persistent BM25 index file."""
        safe_name = f"{owner}__{repo}".lower()
        safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in safe_name)
        return self.bm25_dir / f"{safe_name}.json"

    def add_chunks(self, owner: str, repo: str, chunks: list[Chunk]) -> int:
        """Index chunks and persist to disk.

        Args:
            owner: Repo owner.
            repo: Repo name.
            chunks: List of Chunk objects to index.

        Returns:
            Number of indexed chunks.
        """
        if not chunks:
            logger.warning("No chunks to index for %s/%s", owner, repo)
            return 0

        doc_records = []
        corpus_tokens = []
        doc_freqs = []
        doc_lengths = []
        term_doc_count: Counter[str] = Counter()

        for c in chunks:
            record = {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "file_path": c.file_path,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "language": c.language,
                "symbol_name": c.symbol_name,
                "symbol_type": c.symbol_type,
                "parent_class": c.parent_class,
                "imports": c.imports,
            }
            doc_records.append(record)

            tokens = tokenize_code(c.content)
            corpus_tokens.append(tokens)
            doc_lengths.append(len(tokens))

            tf = Counter(tokens)
            doc_freqs.append(tf)
            for term in tf:
                term_doc_count[term] += 1

        n_docs = len(doc_records)
        avgdl = sum(doc_lengths) / max(1, n_docs)

        # Compute Lucene-style IDF: ln(1 + (N - n + 0.5) / (n + 0.5))
        idf = {}
        for term, n in term_doc_count.items():
            idf[term] = math.log(1.0 + (n_docs - n + 0.5) / (n + 0.5))

        # Save to disk
        out_path = self._index_path(owner, repo)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "owner": owner,
                    "repo": repo,
                    "doc_records": doc_records,
                },
                f,
            )

        # Cache in memory
        repo_key = f"{owner}/{repo}".lower()
        self._cache[repo_key] = {
            "doc_records": doc_records,
            "corpus_tokens": corpus_tokens,
            "doc_freqs": doc_freqs,
            "doc_lengths": doc_lengths,
            "idf": idf,
            "avgdl": avgdl,
        }

        logger.info("BM25 indexed %d chunks for %s/%s at %s", n_docs, owner, repo, out_path)
        return n_docs

    def _ensure_loaded(self, owner: str, repo: str) -> bool:
        """Load index from disk into memory cache if not already loaded."""
        repo_key = f"{owner}/{repo}".lower()
        if repo_key in self._cache:
            return True

        index_file = self._index_path(owner, repo)
        if not index_file.exists():
            return False

        try:
            with open(index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            doc_records = data.get("doc_records", [])
        except Exception as e:
            logger.error("Failed to load BM25 index from %s: %s", index_file, e)
            return False

        corpus_tokens = []
        doc_freqs = []
        doc_lengths = []
        term_doc_count: Counter[str] = Counter()

        for rec in doc_records:
            tokens = tokenize_code(rec["content"])
            corpus_tokens.append(tokens)
            doc_lengths.append(len(tokens))

            tf = Counter(tokens)
            doc_freqs.append(tf)
            for term in tf:
                term_doc_count[term] += 1

        n_docs = len(doc_records)
        avgdl = sum(doc_lengths) / max(1, n_docs)

        idf = {}
        for term, n in term_doc_count.items():
            idf[term] = math.log(1.0 + (n_docs - n + 0.5) / (n + 0.5))

        self._cache[repo_key] = {
            "doc_records": doc_records,
            "corpus_tokens": corpus_tokens,
            "doc_freqs": doc_freqs,
            "doc_lengths": doc_lengths,
            "idf": idf,
            "avgdl": avgdl,
        }
        return True

    def query(
        self,
        owner: str,
        repo: str,
        query_text: str,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Search for code chunks using BM25 scoring.

        Args:
            owner: Repo owner.
            repo: Repo name.
            query_text: User question or code symbol query.
            top_k: Max chunks to return.

        Returns:
            List of RetrievedChunk sorted by BM25 score (descending).
        """
        if not self._ensure_loaded(owner, repo):
            logger.warning("BM25 index not found for %s/%s", owner, repo)
            return []

        repo_key = f"{owner}/{repo}".lower()
        cached = self._cache[repo_key]
        doc_records = cached["doc_records"]
        doc_freqs = cached["doc_freqs"]
        doc_lengths = cached["doc_lengths"]
        idf = cached["idf"]
        avgdl = cached["avgdl"]

        query_tokens = tokenize_code(query_text)
        if not query_tokens:
            return []

        # Filter query tokens to those with non-zero IDF
        active_terms = [t for t in query_tokens if t in idf]
        if not active_terms:
            return []

        scores: list[float] = [0.0] * len(doc_records)

        for i, (tf, doc_len) in enumerate(zip(doc_freqs, doc_lengths)):
            doc_score = 0.0
            len_norm = 1.0 - self.b + self.b * (doc_len / avgdl) if avgdl > 0 else 1.0

            for term in active_terms:
                freq = tf.get(term, 0)
                if freq > 0:
                    term_idf = idf[term]
                    numerator = freq * (self.k1 + 1.0)
                    denominator = freq + self.k1 * len_norm
                    doc_score += term_idf * (numerator / denominator)

            scores[i] = doc_score

        # Collect and rank candidates with score > 0
        ranked_indices = [i for i, s in enumerate(scores) if s > 0.0]
        ranked_indices.sort(key=lambda idx: scores[idx], reverse=True)
        top_indices = ranked_indices[:top_k]

        results: list[RetrievedChunk] = []
        for idx in top_indices:
            rec = doc_records[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=rec["chunk_id"],
                    content=rec["content"],
                    file_path=rec["file_path"],
                    start_line=rec["start_line"],
                    end_line=rec["end_line"],
                    language=rec["language"],
                    distance=scores[idx],  # BM25 score
                    symbol_name=rec.get("symbol_name", ""),
                    symbol_type=rec.get("symbol_type", ""),
                    parent_class=rec.get("parent_class", ""),
                    imports=rec.get("imports", ""),
                )
            )

        logger.info("BM25 query '%s' returned %d results", query_text, len(results))
        return results

    def delete_index(self, owner: str, repo: str) -> None:
        """Delete a repo's persistent BM25 index."""
        repo_key = f"{owner}/{repo}".lower()
        self._cache.pop(repo_key, None)
        path = self._index_path(owner, repo)
        if path.exists():
            path.unlink()
            logger.info("Deleted BM25 index at %s", path)
