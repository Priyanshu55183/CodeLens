"""
Baseline evaluation script for CodeLens Phase 1.

Metrics computed:
- Retrieval Hit@5: Does at least one of the top 5 retrieved chunks come
  from an expected file? This measures whether our retrieval finds the
  right code.
- Retrieval Hit@10: Same but top 10 (looser).
- MRR (Mean Reciprocal Rank): 1/rank of the first correct chunk.
  Higher is better. If the correct file is in position 1, MRR=1.0.

For unanswerable questions, we don't measure retrieval (there's no
correct file to find). We'll add abstention metrics in Phase 7.

Usage:
    python -m eval.run_eval --repo-url https://github.com/pallets/click
"""

import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from codelens.app.config import get_settings
from codelens.app.ingestion.cloner import parse_github_url
from codelens.app.retrieval.vector_store import VectorStore, RetrievedChunk

console = Console()
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
QUESTIONS_FILE = EVAL_DIR / "questions.json"


@dataclass
class QuestionResult:
    """Result for a single eval question."""

    question_id: str
    category: str
    question: str
    expected_files: list[str]
    retrieved_files: list[str] = field(default_factory=list)
    hit_at_5: bool = False
    hit_at_10: bool = False
    reciprocal_rank: float = 0.0


def evaluate_retrieval(
    repo_url: str,
    questions_file: Path = QUESTIONS_FILE,
    top_k: int = 10,
) -> list[QuestionResult]:
    """Run retrieval evaluation on all questions.

    Args:
        repo_url: The indexed repo URL.
        questions_file: Path to questions.json.
        top_k: Number of chunks to retrieve.

    Returns:
        List of QuestionResult objects.
    """
    settings = get_settings()
    owner, repo_name = parse_github_url(repo_url)

    store = VectorStore(
        chroma_dir=settings.chroma_dir,
        api_key=settings.openai_api_key,
        embedding_model=settings.embedding_model,
    )

    with open(questions_file) as f:
        data = json.load(f)

    results: list[QuestionResult] = []

    for q in data["questions"]:
        qr = QuestionResult(
            question_id=q["id"],
            category=q["category"],
            question=q["question"],
            expected_files=q["expected_files"],
        )

        # Skip retrieval metrics for unanswerable questions
        if q["category"] == "unanswerable":
            results.append(qr)
            continue

        # Run vector search
        chunks = store.query(owner, repo_name, q["question"], top_k=top_k)
        qr.retrieved_files = [c.file_path for c in chunks]

        # Check hits
        expected_set = set(q["expected_files"])
        for rank, chunk in enumerate(chunks, 1):
            if chunk.file_path in expected_set:
                if rank <= 5:
                    qr.hit_at_5 = True
                qr.hit_at_10 = True
                if qr.reciprocal_rank == 0.0:
                    qr.reciprocal_rank = 1.0 / rank
                break  # Only need the first hit for MRR

        results.append(qr)

    return results


def print_results(results: list[QuestionResult]) -> None:
    """Display evaluation results in a formatted table."""
    # Filter to answerable questions only for retrieval metrics
    answerable = [r for r in results if r.category != "unanswerable"]

    if not answerable:
        console.print("[yellow]No answerable questions to evaluate.[/yellow]")
        return

    # Summary metrics
    hit5 = sum(1 for r in answerable if r.hit_at_5) / len(answerable)
    hit10 = sum(1 for r in answerable if r.hit_at_10) / len(answerable)
    mrr = sum(r.reciprocal_rank for r in answerable) / len(answerable)

    # Summary table
    summary = Table(title="Retrieval Evaluation — Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_row("Questions (answerable)", str(len(answerable)))
    summary.add_row("Hit@5", f"{hit5:.1%}")
    summary.add_row("Hit@10", f"{hit10:.1%}")
    summary.add_row("MRR", f"{mrr:.3f}")
    console.print(summary)

    # Per-question detail table
    detail = Table(title="Per-Question Results")
    detail.add_column("ID", style="cyan")
    detail.add_column("Category")
    detail.add_column("Hit@5")
    detail.add_column("Hit@10")
    detail.add_column("RR")
    detail.add_column("Top Retrieved Files")

    for r in results:
        if r.category == "unanswerable":
            detail.add_row(r.question_id, r.category, "N/A", "N/A", "N/A", "(unanswerable)")
        else:
            h5 = "[green]✓[/green]" if r.hit_at_5 else "[red]✗[/red]"
            h10 = "[green]✓[/green]" if r.hit_at_10 else "[red]✗[/red]"
            top_files = ", ".join(dict.fromkeys(r.retrieved_files[:5]))  # Unique, ordered
            detail.add_row(r.question_id, r.category, h5, h10, f"{r.reciprocal_rank:.2f}", top_files)

    console.print(detail)


def main() -> None:
    """Run the evaluation."""
    parser = argparse.ArgumentParser(description="Run CodeLens retrieval evaluation")
    parser.add_argument("--repo-url", required=True, help="GitHub repo URL (must be indexed)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    console.print(f"\n[bold blue]Running evaluation for:[/bold blue] {args.repo_url}\n")

    results = evaluate_retrieval(args.repo_url)
    print_results(results)

    # Save results to JSON
    results_dir = EVAL_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / "phase1_baseline.json"
    with open(output_file, "w") as f:
        json.dump(
            [
                {
                    "question_id": r.question_id,
                    "category": r.category,
                    "hit_at_5": r.hit_at_5,
                    "hit_at_10": r.hit_at_10,
                    "reciprocal_rank": r.reciprocal_rank,
                    "retrieved_files": r.retrieved_files[:5],
                }
                for r in results
            ],
            f,
            indent=2,
        )
    console.print(f"\n[dim]Results saved to {output_file}[/dim]")


if __name__ == "__main__":
    main()
