"""
Pydantic schemas for CodeLens REST API (Phase 6).

Design decisions:
- Request and response models are strictly typed via Pydantic v2.
- Response models always include the same fields regardless of which
  retrieval path was taken. Callers never get different JSON shapes.
- We expose only the minimum needed for external clients. Internal
  dataclasses (Chunk, RetrievedChunk) stay internal.
- 'retrieval_mode' defaults are intentionally None in requests — the
  server resolves the default from settings so callers never need to
  specify it unless overriding.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Shared types ─────────────────────────────────────────────────────────────

class ChunkInfo(BaseModel):
    """A single retrieved code chunk, returned by /ask and /find."""

    chunk_id: str = Field(..., description="Unique identifier for this chunk.")
    file_path: str = Field(..., description="Relative file path within the repo.")
    start_line: int = Field(..., description="1-indexed starting line number.")
    end_line: int = Field(..., description="1-indexed ending line number.")
    language: str = Field(..., description="Programming language (e.g., 'python').")
    content: str = Field(..., description="The source code text of this chunk.")
    score: float = Field(
        ...,
        description=(
            "Retrieval score. For hybrid/BM25 this is the RRF score (higher = better). "
            "For vector-only this is cosine distance (lower = better)."
        ),
    )
    symbol_name: str = Field("", description="Symbol name if the chunk is a definition.")
    symbol_type: str = Field("", description="Symbol type: function, class, method, etc.")
    parent_class: str = Field("", description="Enclosing class name for methods.")
    citation: str = Field(..., description="Human-readable citation string.")


# ── /repos/index ─────────────────────────────────────────────────────────────

class IndexRequest(BaseModel):
    """Request body for POST /repos/index."""

    repo_url: str = Field(
        ...,
        description="GitHub repository URL (e.g., https://github.com/owner/repo).",
        examples=["https://github.com/pallets/click"],
    )


class IndexResponse(BaseModel):
    """Response for POST /repos/index."""

    owner: str = Field(..., description="Repository owner.")
    repo: str = Field(..., description="Repository name.")
    commit_sha: str = Field(..., description="HEAD commit SHA that was indexed.")
    files_indexed: int = Field(..., description="Number of source files processed.")
    chunks_created: int = Field(..., description="Number of code chunks created.")
    duration_seconds: float = Field(..., description="Total indexing time in seconds.")
    message: str = Field(..., description="Human-readable status message.")


# ── /repos/ask ───────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    """Request body for POST /repos/ask."""

    repo_url: str = Field(
        ...,
        description="GitHub repository URL (must be indexed first).",
        examples=["https://github.com/pallets/click"],
    )
    question: str = Field(
        ...,
        description="Natural-language question about the codebase.",
        examples=["How does Click handle type conversion for parameters?"],
    )
    retrieval_mode: str | None = Field(
        default=None,
        description="Retrieval strategy: 'hybrid' (default), 'vector', or 'bm25'.",
        examples=["hybrid"],
    )
    top_k: int | None = Field(
        default=None,
        description="Number of code chunks to retrieve (default from settings).",
        ge=1,
        le=50,
    )


class AskResponse(BaseModel):
    """Response for POST /repos/ask."""

    question: str = Field(..., description="The original question.")
    answer: str = Field(..., description="LLM-generated answer grounded in retrieved code.")
    retrieval_mode: str = Field(..., description="Retrieval strategy that was used.")
    chunks_retrieved: int = Field(..., description="Number of code chunks used as context.")
    sources: list[ChunkInfo] = Field(..., description="Code chunks used to generate the answer.")
    provider: str = Field(..., description="LLM provider used for answering.")
    model: str = Field(..., description="Specific LLM model used for answering.")


# ── /repos/find ──────────────────────────────────────────────────────────────

class FindRequest(BaseModel):
    """Request body for POST /repos/find."""

    repo_url: str = Field(
        ...,
        description="GitHub repository URL (must be indexed first).",
        examples=["https://github.com/pallets/click"],
    )
    symbol: str = Field(
        ...,
        description="Symbol name to look up (function, class, or method name).",
        examples=["OptionParser"],
    )
    symbol_type: str | None = Field(
        default=None,
        description="Optional type filter: 'function', 'class', 'method', 'interface', 'type'.",
        examples=["function"],
    )
    fuzzy: bool = Field(
        default=True,
        description=(
            "If True (default), substring matches are included "
            "(e.g. 'parse' matches 'parse_args'). "
            "If False, only exact case-insensitive matches are returned."
        ),
    )
    top_k: int | None = Field(
        default=None,
        description="Maximum number of results (default from settings).",
        ge=1,
        le=50,
    )


class FindResponse(BaseModel):
    """Response for POST /repos/find."""

    symbol: str = Field(..., description="The queried symbol name.")
    symbol_type: str | None = Field(..., description="Type filter that was applied.")
    fuzzy: bool = Field(..., description="Whether fuzzy matching was used.")
    total_found: int = Field(..., description="Number of symbol definitions found.")
    results: list[ChunkInfo] = Field(..., description="Matching symbol definitions.")


# ── /repos/status ────────────────────────────────────────────────────────────

class RepoStatusResponse(BaseModel):
    """Response for GET /repos/status."""

    owner: str = Field(..., description="Repository owner.")
    repo: str = Field(..., description="Repository name.")
    indexed: bool = Field(..., description="Whether the repo has been indexed.")
    chunk_count: int = Field(..., description="Number of indexed chunks (0 if not indexed).")


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = Field(..., description="Service health status: 'ok' or 'degraded'.")
    version: str = Field(..., description="CodeLens version.")
    provider: str = Field(..., description="Configured LLM provider.")
    model: str = Field(..., description="Configured LLM model.")
    retrieval_mode: str = Field(..., description="Default retrieval mode.")


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    """Standard error response body."""

    error: str = Field(..., description="Machine-readable error type.")
    message: str = Field(..., description="Human-readable error description.")
    detail: str | None = Field(None, description="Optional additional context.")
