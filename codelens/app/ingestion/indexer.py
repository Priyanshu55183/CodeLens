"""
Indexer: orchestrates clone → filter → chunk → embed.

This is the "pipeline" module that ties ingestion steps together.
Each step is its own module; this module just calls them in order.

Design decision: We return an IndexResult dataclass so the CLI
can display stats (files indexed, chunks created, time taken)
without reaching into internals.
"""

import time
import logging
from pathlib import Path
from dataclasses import dataclass

from codelens.app.config import Settings
from codelens.app.ingestion.cloner import clone_repo, parse_github_url
from codelens.app.ingestion.filters import collect_files
from codelens.app.ingestion.chunker import chunk_all_files, chunk_all_files_smart, Chunk
from codelens.app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    """Stats from an indexing run."""

    owner: str
    repo: str
    commit_sha: str
    files_indexed: int
    chunks_created: int
    duration_seconds: float


def index_repo(url: str, settings: Settings) -> IndexResult:
    """Run the full indexing pipeline for a GitHub repo.

    Steps:
        1. Clone the repo (or reuse existing clone)
        2. Collect files (skip junk)
        3. Chunk all files (tree-sitter syntax-aware or sliding-window fallback)
        4. Embed and store in ChromaDB

    Args:
        url: GitHub repository URL.
        settings: Application settings.

    Returns:
        IndexResult with stats.
    """
    start_time = time.time()
    owner, repo_name = parse_github_url(url)

    # Step 1: Clone
    repo_dir, commit_sha = clone_repo(
        url, settings.repos_dir, settings.github_token
    )
    logger.info("Step 1/4: Clone complete (SHA: %s)", commit_sha[:8])

    # Step 2: Collect files
    files = collect_files(repo_dir)
    logger.info("Step 2/4: Collected %d files", len(files))

    # Step 3: Chunk (tree-sitter smart chunking when enabled, sliding-window fallback)
    if getattr(settings, "use_treesitter", True):
        chunks = chunk_all_files_smart(
            files, repo_dir, settings.chunk_size_lines, settings.chunk_overlap_lines
        )
    else:
        chunks = chunk_all_files(
            files, repo_dir, settings.chunk_size_lines, settings.chunk_overlap_lines
        )
    logger.info("Step 3/4: Created %d chunks", len(chunks))

    # Step 4: Embed and store
    store = VectorStore(
        chroma_dir=settings.chroma_dir,
        api_key=settings.openai_api_key,
        embedding_model=settings.embedding_model,
    )
    store.add_chunks(owner, repo_name, chunks)
    logger.info("Step 4/4: Stored in ChromaDB")

    duration = time.time() - start_time
    return IndexResult(
        owner=owner,
        repo=repo_name,
        commit_sha=commit_sha,
        files_indexed=len(files),
        chunks_created=len(chunks),
        duration_seconds=round(duration, 2),
    )
