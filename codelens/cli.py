"""
CodeLens CLI — index a repo and ask questions from the command line.

Commands:
  index  <repo_url>              — clone, chunk, embed a repo
  ask    <repo_url> <question>   — hybrid search + LLM answer
  find   <repo_url> <symbol>     — look up a symbol definition
  serve                          — start the FastAPI REST API server (Phase 6)

Design decision: We use argparse (stdlib) instead of click/typer because
it's zero-dependency and we only have a few commands. The real API surface
is the FastAPI server started with the 'serve' command.

The ask flow:
  1. Hybrid retrieval (dense + BM25 + RRF) for top-K relevant chunks
  2. Build a prompt with the chunks as context
  3. Send to LLM with a system prompt that enforces grounded answers
  4. Print the answer with citations"""

import argparse
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from codelens.app.config import get_settings
from codelens.app.ingestion.cloner import parse_github_url
from codelens.app.ingestion.indexer import index_repo
from codelens.app.retrieval import (
    VectorStore,
    BM25Index,
    HybridRetriever,
    SymbolFinder,
    RetrievedChunk,
)
from codelens.app.llm_client import ask_llm

console = Console()

# ── System prompt for grounded Q&A ──────────────────────────────
SYSTEM_PROMPT = """\
You are CodeLens, a code-analysis assistant. You answer questions about a codebase
using ONLY the code snippets provided below as context.

RULES — follow these strictly:
1. Base every claim on the provided code snippets. Cite your sources using
   the format [file_path:start_line-end_line].
2. If the snippets do not contain enough information to answer the question,
   say: "I couldn't find enough evidence in the retrieved code to answer that."
   Do NOT guess or make up information.
3. If you are partially sure, state what you found and explicitly flag what
   is uncertain: "Based on the retrieved code, X appears to be the case, but
   I could not confirm Y."
4. Never execute, run, or evaluate any code. Only read and analyze it.
5. Keep your answer concise and focused on the question asked.
"""


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a context string for the LLM.
    
    Each chunk is wrapped in a clear delimiter so the LLM can
    distinguish between chunks and cite them properly.
    """
    if not chunks:
        return "(No relevant code was found.)"

    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"--- Snippet {i} [{chunk.file_path}:{chunk.start_line}-{chunk.end_line}] ---\n"
            f"{chunk.content}\n"
        )
    return "\n".join(parts)


def build_prompt(question: str, context: str) -> str:
    """Build the user prompt with question and code context."""
    return (
        f"## Code Context\n\n{context}\n\n"
        f"## Question\n\n{question}\n\n"
        f"## Instructions\n\n"
        f"Answer the question using ONLY the code snippets above. "
        f"Cite every claim with [file_path:start-end]. "
        f"If evidence is insufficient, say so explicitly."
    )


def cmd_index(args: argparse.Namespace) -> None:
    """Handle the 'index' command."""
    settings = get_settings()

    console.print(f"\n[bold blue]Indexing:[/bold blue] {args.repo_url}\n")

    result = index_repo(args.repo_url, settings)

    # Display results in a nice table
    table = Table(title="Indexing Complete ✓", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Repository", f"{result.owner}/{result.repo}")
    table.add_row("Commit SHA", result.commit_sha[:12])
    table.add_row("Files indexed", str(result.files_indexed))
    table.add_row("Chunks created", str(result.chunks_created))
    table.add_row("Duration", f"{result.duration_seconds}s")
    console.print(table)


def cmd_ask(args: argparse.Namespace) -> None:
    """Handle the 'ask' command."""
    settings = get_settings()
    owner, repo_name = parse_github_url(args.repo_url)

    console.print(f"\n[bold blue]Question:[/bold blue] {args.question}\n")

    # Step 1: Retrieve relevant chunks (Hybrid dense + sparse)
    mode = getattr(args, "mode", None) or settings.retrieval_mode
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
    chunks = retriever.query(owner, repo_name, args.question, top_k=settings.top_k, mode=mode)

    if not chunks:
        console.print("[yellow]No indexed data found. Run 'index' first.[/yellow]")
        return

    # Show retrieved sources
    console.print(f"[dim]Retrieved {len(chunks)} code snippets (mode: {mode})[/dim]\n")
    for i, chunk in enumerate(chunks, 1):
        console.print(f"  [dim]{i}. {chunk.citation} (score/dist: {chunk.distance:.4f})[/dim]")
    console.print()

    # Step 2: Build prompt and ask LLM
    context = build_context(chunks)
    prompt = build_prompt(args.question, context)

    console.print("[dim]Thinking...[/dim]\n")
    answer = ask_llm(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        provider=settings.llm_provider,
        model=settings.resolved_llm_model,
        api_key=settings.active_api_key,
        temperature=0.0,
    )

    # Step 3: Display answer
    console.print(Panel(
        Markdown(answer),
        title="[bold green]CodeLens Answer[/bold green]",
        border_style="green",
    ))


def cmd_find(args: argparse.Namespace) -> None:
    """Handle the 'find' command — look up a symbol definition by name."""
    settings = get_settings()
    owner, repo_name = parse_github_url(args.repo_url)

    symbol = args.symbol
    symbol_type = getattr(args, "type", None) or None
    fuzzy = not getattr(args, "exact", False)

    console.print(
        f"\n[bold blue]Finding symbol:[/bold blue] [bold]{symbol}[/bold]"
        + (f" [dim](type: {symbol_type})[/dim]" if symbol_type else "")
        + (" [dim](exact)[/dim]" if not fuzzy else "")
        + "\n"
    )

    vector_store = VectorStore(
        chroma_dir=settings.chroma_dir,
        api_key=settings.openai_api_key,
        embedding_model=settings.embedding_model,
    )
    bm25_index = BM25Index(bm25_dir=settings.bm25_dir)
    finder = SymbolFinder(vector_store=vector_store, bm25_index=bm25_index)

    results = finder.find(
        owner=owner,
        repo=repo_name,
        symbol_name=symbol,
        symbol_type=symbol_type,
        top_k=settings.top_k,
        fuzzy=fuzzy,
    )

    if not results:
        console.print("[yellow]No matching symbols found. Did you index the repo first?[/yellow]")
        return

    console.print(f"[dim]Found {len(results)} definition(s):[/dim]\n")

    from rich.syntax import Syntax

    for i, chunk in enumerate(results, 1):
        # Build a rich header line
        kind = chunk.symbol_type or "symbol"
        parent = f"[dim]{chunk.parent_class}.[/dim]" if chunk.parent_class else ""
        title = (
            f"  {i}. [cyan]{kind}[/cyan] {parent}[bold]{chunk.symbol_name}[/bold] "
            f"[dim]— {chunk.citation}[/dim]"
        )
        console.print(title)

        # Show the code with syntax highlighting (first 20 lines only for readability)
        code_lines = chunk.content.splitlines()
        preview = "\n".join(code_lines[:20])
        if len(code_lines) > 20:
            preview += f"\n[dim]... ({len(code_lines) - 20} more lines)[/dim]"

        syntax = Syntax(
            preview,
            chunk.language or "text",
            theme="monokai",
            line_numbers=True,
            start_line=chunk.start_line,
            word_wrap=False,
        )
        console.print(Panel(syntax, border_style="dim"))
        console.print()


def cmd_serve(args: argparse.Namespace) -> None:
    """Handle the 'serve' command — start the FastAPI REST API server."""
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]Error:[/red] uvicorn is required to run the server.\n"
            "Install it with: [bold]pip install uvicorn[/bold]"
        )
        return

    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)
    reload = getattr(args, "reload", False)

    console.print(Panel(
        f"[bold green]CodeLens API[/bold green]\n"
        f"  Docs:  [link=http://{host}:{port}/docs]http://{host}:{port}/docs[/link]\n"
        f"  ReDoc: [link=http://{host}:{port}/redoc]http://{host}:{port}/redoc[/link]\n"
        f"  Health:[link=http://{host}:{port}/health]http://{host}:{port}/health[/link]",
        title="Starting Server",
        border_style="blue",
    ))

    uvicorn.run(
        "codelens.app.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def main() -> None:
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog="codelens",
        description="CodeLens: Accuracy-first Codebase Q&A",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index command
    index_parser = subparsers.add_parser("index", help="Index a GitHub repository")
    index_parser.add_argument("repo_url", help="GitHub repository URL")

    # ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a question about a repo")
    ask_parser.add_argument("repo_url", help="GitHub repository URL")
    ask_parser.add_argument("question", help="Your question about the codebase")
    ask_parser.add_argument(
        "--mode",
        choices=["hybrid", "vector", "bm25"],
        default=None,
        help="Retrieval strategy: hybrid (default), vector, or bm25",
    )

    # find command
    find_parser = subparsers.add_parser("find", help="Find a symbol definition by name")
    find_parser.add_argument("repo_url", help="GitHub repository URL")
    find_parser.add_argument("symbol", help="Symbol name to find (function, class, method)")
    find_parser.add_argument(
        "--type",
        dest="type",
        choices=["function", "class", "method", "interface", "type"],
        default=None,
        help="Restrict search to a specific symbol type",
    )
    find_parser.add_argument(
        "--exact",
        action="store_true",
        default=False,
        help="Only return exact name matches (no substring/fuzzy matching)",
    )

    # serve command (Phase 6)
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI REST API server")
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "index":
        cmd_index(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "find":
        cmd_find(args)
    elif args.command == "serve":
        cmd_serve(args)


if __name__ == "__main__":
    main()
