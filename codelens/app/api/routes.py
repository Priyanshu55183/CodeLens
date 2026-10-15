"""
/repos router — index, ask, find, status endpoints (Phase 6).

Design decisions:
- BackgroundTasks is NOT used for indexing. Indexing can take 30-120 s for
  large repos; we keep it synchronous so the caller knows when it's done.
  A real production system would use Celery/RQ and return a job_id — we
  add a TODO comment for that upgrade path.
- We construct retrieval objects per-request rather than as app-level
  singletons. ChromaDB's PersistentClient is thread-safe; BM25Index
  deserialises from disk (fast, ~10 ms) and is stateless for queries.
  This keeps the server stateless and easier to test.
- HTTP status codes:
    200  — successful read/query
    201  — successful index (resource created)
    400  — bad request (invalid repo URL, unsupported mode, etc.)
    404  — repo not indexed yet
    422  — Pydantic validation failure (FastAPI handles automatically)
    500  — unexpected server error (logged + returned)
- We catch ValueError/RuntimeError separately from generic Exception so
  the error category is surfaced in the response.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from codelens.app.config import get_settings
from codelens.app.ingestion.cloner import parse_github_url
from codelens.app.ingestion.indexer import index_repo
from codelens.app.retrieval import VectorStore, BM25Index, HybridRetriever, SymbolFinder, RetrievedChunk
from codelens.app.llm_client import ask_llm
from codelens.app.api.schemas import (
    IndexRequest,
    IndexResponse,
    AskRequest,
    AskResponse,
    FindRequest,
    FindResponse,
    RepoStatusResponse,
    ChunkInfo,
    ErrorDetail,
)

# Re-use the system prompt from cli.py (single source of truth)
from codelens.cli import SYSTEM_PROMPT, build_context, build_prompt

router = APIRouter(prefix="/repos", tags=["Repositories"])
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_to_info(chunk: RetrievedChunk) -> ChunkInfo:
    """Convert an internal RetrievedChunk to a public ChunkInfo schema."""
    return ChunkInfo(
        chunk_id=chunk.chunk_id,
        file_path=chunk.file_path,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        language=chunk.language,
        content=chunk.content,
        score=chunk.distance,
        symbol_name=chunk.symbol_name,
        symbol_type=chunk.symbol_type,
        parent_class=chunk.parent_class,
        citation=chunk.citation,
    )


def _make_retriever(settings) -> tuple[VectorStore, BM25Index, HybridRetriever]:
    """Build retrieval stack from settings. Called per-request."""
    vector_store = VectorStore(
        chroma_dir=settings.chroma_dir,
        api_key=settings.openai_api_key,
        embedding_model=settings.embedding_model,
    )
    bm25_index = BM25Index(bm25_dir=settings.bm25_dir)
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        default_mode=settings.retrieval_mode,
        rrf_k=settings.rrf_k,
        dense_weight=settings.dense_weight,
        bm25_weight=settings.bm25_weight,
    )
    return vector_store, bm25_index, retriever


# ── POST /repos/index ────────────────────────────────────────────────────────

@router.post(
    "/index",
    response_model=IndexResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Index a GitHub repository",
    description=(
        "Clone, chunk, embed, and persist a GitHub repository. "
        "This is a synchronous operation — it completes before returning. "
        "Re-indexing the same repo is idempotent (existing data is overwritten)."
    ),
    responses={
        400: {"model": ErrorDetail, "description": "Invalid repository URL."},
        500: {"model": ErrorDetail, "description": "Indexing failed."},
    },
)
def index_repo_endpoint(body: IndexRequest) -> IndexResponse:
    """Index a GitHub repository."""
    settings = get_settings()

    # Validate URL before kicking off the long operation
    try:
        owner, repo_name = parse_github_url(body.repo_url)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_repo_url", "message": str(exc), "detail": body.repo_url},
        ) from exc

    logger.info("POST /repos/index — %s/%s", owner, repo_name)

    try:
        result = index_repo(body.repo_url, settings)
    except Exception as exc:
        logger.exception("Indexing failed for %s", body.repo_url)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "indexing_failed", "message": str(exc), "detail": None},
        ) from exc

    return IndexResponse(
        owner=result.owner,
        repo=result.repo,
        commit_sha=result.commit_sha,
        files_indexed=result.files_indexed,
        chunks_created=result.chunks_created,
        duration_seconds=result.duration_seconds,
        message=f"Successfully indexed {result.files_indexed} files into {result.chunks_created} chunks.",
    )


# ── POST /repos/ask ──────────────────────────────────────────────────────────

@router.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about a repository",
    description=(
        "Retrieve relevant code chunks and generate a grounded answer using an LLM. "
        "The repo must be indexed first via POST /repos/index."
    ),
    responses={
        400: {"model": ErrorDetail, "description": "Invalid request parameters."},
        404: {"model": ErrorDetail, "description": "Repository not indexed."},
        500: {"model": ErrorDetail, "description": "Retrieval or LLM call failed."},
    },
)
def ask_endpoint(body: AskRequest) -> AskResponse:
    """Ask a question about a repository's codebase."""
    settings = get_settings()

    try:
        owner, repo_name = parse_github_url(body.repo_url)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_repo_url", "message": str(exc), "detail": body.repo_url},
        ) from exc

    mode = body.retrieval_mode or settings.retrieval_mode
    if mode not in ("hybrid", "vector", "bm25"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_retrieval_mode",
                "message": f"Unknown retrieval mode '{mode}'.",
                "detail": "Valid values: 'hybrid', 'vector', 'bm25'.",
            },
        )

    top_k = body.top_k or settings.top_k
    logger.info("POST /repos/ask — %s/%s q=%r mode=%s", owner, repo_name, body.question, mode)

    _, _, retriever = _make_retriever(settings)

    try:
        chunks = retriever.query(owner, repo_name, body.question, top_k=top_k, mode=mode)
    except Exception as exc:
        logger.exception("Retrieval failed for %s/%s", owner, repo_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "retrieval_failed", "message": str(exc), "detail": None},
        ) from exc

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "repo_not_indexed",
                "message": f"No indexed data found for {owner}/{repo_name}.",
                "detail": "Index the repository first using POST /repos/index.",
            },
        )

    context = build_context(chunks)
    prompt = build_prompt(body.question, context)

    try:
        answer = ask_llm(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            provider=settings.llm_provider,
            model=settings.resolved_llm_model,
            api_key=settings.active_api_key,
            temperature=0.0,
        )
    except Exception as exc:
        logger.exception("LLM call failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "llm_failed", "message": str(exc), "detail": None},
        ) from exc

    return AskResponse(
        question=body.question,
        answer=answer,
        retrieval_mode=mode,
        chunks_retrieved=len(chunks),
        sources=[_chunk_to_info(c) for c in chunks],
        provider=settings.llm_provider,
        model=settings.resolved_llm_model,
    )


# ── POST /repos/find ─────────────────────────────────────────────────────────

@router.post(
    "/find",
    response_model=FindResponse,
    status_code=status.HTTP_200_OK,
    summary="Find a symbol definition",
    description=(
        "Look up a function, class, or method definition by name. "
        "Returns matching code chunks ranked by exactness and relevance."
    ),
    responses={
        400: {"model": ErrorDetail, "description": "Invalid request parameters."},
        404: {"model": ErrorDetail, "description": "Repository not indexed."},
        500: {"model": ErrorDetail, "description": "Search failed."},
    },
)
def find_endpoint(body: FindRequest) -> FindResponse:
    """Find symbol definitions in an indexed repository."""
    settings = get_settings()

    try:
        owner, repo_name = parse_github_url(body.repo_url)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_repo_url", "message": str(exc), "detail": body.repo_url},
        ) from exc

    top_k = body.top_k or settings.top_k
    logger.info(
        "POST /repos/find — %s/%s symbol=%r type=%s fuzzy=%s",
        owner, repo_name, body.symbol, body.symbol_type, body.fuzzy,
    )

    vector_store, bm25_index, _ = _make_retriever(settings)
    finder = SymbolFinder(vector_store=vector_store, bm25_index=bm25_index)

    try:
        results = finder.find(
            owner=owner,
            repo=repo_name,
            symbol_name=body.symbol,
            symbol_type=body.symbol_type,
            top_k=top_k,
            fuzzy=body.fuzzy,
        )
    except Exception as exc:
        logger.exception("Symbol search failed for %s/%s", owner, repo_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "find_failed", "message": str(exc), "detail": None},
        ) from exc

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "symbol_not_found",
                "message": f"No definitions found for symbol '{body.symbol}' in {owner}/{repo_name}.",
                "detail": "Check the symbol name, or index the repo first.",
            },
        )

    return FindResponse(
        symbol=body.symbol,
        symbol_type=body.symbol_type,
        fuzzy=body.fuzzy,
        total_found=len(results),
        results=[_chunk_to_info(c) for c in results],
    )


# ── GET /repos/status ────────────────────────────────────────────────────────

@router.get(
    "/status",
    response_model=RepoStatusResponse,
    summary="Check if a repository has been indexed",
    description="Returns whether the given repo is indexed and how many chunks it has.",
    responses={
        400: {"model": ErrorDetail, "description": "Invalid repository URL."},
    },
)
def repo_status(repo_url: str) -> RepoStatusResponse:
    """Check index status of a repository.

    Query param: ?repo_url=https://github.com/owner/repo
    """
    settings = get_settings()

    try:
        owner, repo_name = parse_github_url(repo_url)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_repo_url", "message": str(exc), "detail": repo_url},
        ) from exc

    vector_store = VectorStore(
        chroma_dir=settings.chroma_dir,
        api_key=settings.openai_api_key,
        embedding_model=settings.embedding_model,
    )

    try:
        count = vector_store.count(owner, repo_name)
        indexed = count > 0
    except Exception:
        count = 0
        indexed = False

    return RepoStatusResponse(
        owner=owner,
        repo=repo_name,
        indexed=indexed,
        chunk_count=count,
    )
