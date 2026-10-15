"""
FastAPI application factory for CodeLens (Phase 6).

Design decisions:
- We use a factory function `create_app()` rather than a module-level
  `app = FastAPI(...)`. This makes testing trivial: tests call create_app()
  to get a fresh app without starting a real server.
- CORS is enabled with allow_origins=["*"] for development. Production
  deployments should restrict this to the actual frontend origin.
- Custom exception handlers for HTTPException convert our dict-based
  `detail` fields into the structured ErrorDetail schema. This ensures
  consistent error JSON across all routes.
- Lifespan context: FastAPI 0.93+ recommends lifespan over @app.on_event.
  We log server start/stop for observability.
- Docs are enabled at /docs (Swagger UI) and /redoc (ReDoc). They can be
  disabled in production by setting docs_url=None in FastAPI().
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from codelens.app.api.routes import router as repos_router
from codelens.app.api.schemas import HealthResponse, ErrorDetail
from codelens.app.config import get_settings

logger = logging.getLogger(__name__)

VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    settings = get_settings()
    logger.info(
        "CodeLens API v%s starting — provider=%s model=%s mode=%s",
        VERSION,
        settings.llm_provider,
        settings.resolved_llm_model,
        settings.retrieval_mode,
    )
    yield
    logger.info("CodeLens API shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Fully configured FastAPI app ready to be served.
    """
    app = FastAPI(
        title="CodeLens API",
        description=(
            "**CodeLens** — Accuracy-first Codebase Q&A as a REST API.\n\n"
            "Index any GitHub repository, then ask natural-language questions "
            "and look up symbol definitions. Answers are grounded in retrieved "
            "source code with inline citations.\n\n"
            "## Workflow\n"
            "1. `POST /repos/index` — Clone, chunk, and embed a repo.\n"
            "2. `POST /repos/ask` — Ask a question; get an LLM answer with sources.\n"
            "3. `POST /repos/find` — Look up a function/class definition by name.\n"
            "4. `GET /repos/status` — Check if a repo is indexed.\n\n"
            "## Authentication\n"
            "Set `OPENAI_API_KEY` (and optionally `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`) "
            "in your `.env` file before starting the server."
        ),
        version=VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Custom exception handler ──────────────────────────────────────────────
    from fastapi.exceptions import HTTPException

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Return structured ErrorDetail for all HTTP errors."""
        # If detail is already a dict (from our routers), pass it through.
        if isinstance(exc.detail, dict):
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        # Otherwise wrap it in the standard shape.
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": "http_error", "message": str(exc.detail), "detail": None},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """Catch-all for unexpected errors — log and return 500."""
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred.",
                "detail": str(exc),
            },
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(repos_router)

    # ── Health endpoint ───────────────────────────────────────────────────────
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Utilities"],
        summary="Health check",
        description="Returns service status and current configuration.",
    )
    def health() -> HealthResponse:
        """Return service health and current configuration."""
        settings = get_settings()
        return HealthResponse(
            status="ok",
            version=VERSION,
            provider=settings.llm_provider,
            model=settings.resolved_llm_model,
            retrieval_mode=settings.retrieval_mode,
        )

    @app.get("/", include_in_schema=False)
    def root():
        """Redirect root to docs."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/docs")

    return app


# Module-level app instance — used by uvicorn when running:
#   uvicorn codelens.app.api.main:app --reload
app = create_app()
