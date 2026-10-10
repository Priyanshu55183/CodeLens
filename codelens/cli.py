"""
CodeLens CLI — index a repo and ask questions from the command line.

This is the main entry point for Phase 1. It provides two commands:
  1. `python -m codelens.cli index <repo_url>` — clone, chunk, embed
  2. `python -m codelens.cli ask <repo_url> "<question>"` — search + answer

Design decision: We use argparse (stdlib) instead of click/typer because
it's zero-dependency and we only have 2 commands. We'll move to FastAPI
for the real API in Phase 9.

The ask flow:
  1. Vector search for top-K relevant chunks
  2. Build a prompt with the chunks as context
  3. Send to LLM with a system prompt that enforces grounded answers
  4. Print the answer with citations
"""

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
from codelens.app.retrieval.vector_store import VectorStore, RetrievedChunk
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

    # Step 1: Retrieve relevant chunks
    store = VectorStore(
        chroma_dir=settings.chroma_dir,
        api_key=settings.openai_api_key,
        embedding_model=settings.embedding_model,
    )
    chunks = store.query(owner, repo_name, args.question, top_k=settings.top_k)

    if not chunks:
        console.print("[yellow]No indexed data found. Run 'index' first.[/yellow]")
        return

    # Show retrieved sources
    console.print(f"[dim]Retrieved {len(chunks)} code snippets[/dim]\n")
    for i, chunk in enumerate(chunks, 1):
        console.print(f"  [dim]{i}. {chunk.citation} (distance: {chunk.distance:.4f})[/dim]")
    console.print()

    # Step 2: Build prompt and ask LLM
    context = build_context(chunks)
    prompt = build_prompt(args.question, context)

    console.print("[dim]Thinking...[/dim]\n")
    answer = ask_llm(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0.0,
    )

    # Step 3: Display answer
    console.print(Panel(
        Markdown(answer),
        title="[bold green]CodeLens Answer[/bold green]",
        border_style="green",
    ))


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


if __name__ == "__main__":
    main()
