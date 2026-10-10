"""
Sliding-window text chunker + smart dispatch.

Phase 1: Sliding-window line-based chunker. Still the FALLBACK for
unsupported languages, markdown, configs, etc.

Phase 2 adds: tree-sitter-based syntax-aware chunking for Python, JS, TS.
The smart dispatcher `chunk_all_files_smart()` tries tree-sitter first,
falls back to sliding-window when tree-sitter doesn't support the language.

Design decisions:
- We chunk by LINES (not tokens) because:
  (a) line numbers map directly to file positions (for citations),
  (b) code is line-oriented — splitting mid-line is always wrong,
  (c) it's easy to understand and debug.
- Each chunk gets a CONTEXT HEADER prepended before embedding. This tells
  the embedding model what file this code comes from. Without it, a chunk
  like `def run():` could match any "run" function across the repo.
- Overlap ensures that if a concept spans a chunk boundary, it still
  appears in at least one chunk.

Chunk metadata (stored in ChromaDB alongside the text):
- file_path: relative path from repo root
- start_line, end_line: 1-indexed line range
- language: inferred from file extension
- symbol_name, symbol_type, parent_class: from tree-sitter (Phase 2)
- imports: comma-separated imports of the file (Phase 2)
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Map file extensions to language names (used in context headers)
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".txt": "text",
    ".sql": "sql",
    ".r": "r",
    ".R": "r",
    ".dockerfile": "dockerfile",
}


@dataclass
class Chunk:
    """A piece of a file, ready to be embedded.

    Interview note: We use a dataclass (not a dict) because it gives us
    type safety, auto-generated __init__/__repr__, and is self-documenting.

    Phase 2 adds symbol_name/symbol_type/parent_class/imports for
    tree-sitter-based chunks. These are empty strings for sliding-window
    chunks — backward compatible.
    """

    content: str             # The chunk text WITH context header
    raw_content: str         # The chunk text WITHOUT context header
    file_path: str           # Relative path from repo root
    start_line: int          # 1-indexed
    end_line: int            # 1-indexed, inclusive
    language: str            # Inferred from extension
    chunk_id: str = ""       # Set by the caller (e.g. "path/to/file:10-45")
    # Phase 2: tree-sitter metadata (empty for sliding-window chunks)
    symbol_name: str = ""    # e.g. "parse_args", "Command"
    symbol_type: str = ""    # e.g. "function", "class", "method", "module"
    parent_class: str = ""   # e.g. "Command" if this is a method of Command
    imports: str = ""        # Comma-separated imports of the file

    @property
    def metadata(self) -> dict:
        """Return metadata dict for ChromaDB storage.

        Interview note: ChromaDB only supports str/int/float/bool in
        metadata — no lists. That's why imports is a comma-separated
        string, not a list.
        """
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type,
            "parent_class": self.parent_class,
            "imports": self.imports,
        }


def detect_language(file_path: Path) -> str:
    """Detect programming language from file extension.
    
    Returns "unknown" for unrecognized extensions. We never skip a file
    just because we don't know the language — we still chunk and index it.
    """
    # Handle special file names
    name = file_path.name.lower()
    if name == "dockerfile":
        return "dockerfile"
    if name == "makefile":
        return "makefile"

    suffix = file_path.suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(suffix, "unknown")


def make_context_header(
    file_path: str,
    start_line: int,
    end_line: int,
    language: str,
    symbol_name: str = "",
    symbol_type: str = "",
    parent_class: str = "",
) -> str:
    """Create a context header for a chunk.
    
    This is prepended to the chunk text before embedding. It helps the
    embedding model understand what this code is and where it lives.

    Phase 2 enhancement: if symbol info is available, include it in the
    header. This gives the embedding model even more context about what
    this code does.

    Example (Phase 1):
        # File: src/utils/helpers.py | Lines: 10-45 | Language: python
    Example (Phase 2):
        # File: src/core.py | Lines: 10-45 | Language: python | method Command.invoke
    """
    parts = [f"# File: {file_path} | Lines: {start_line}-{end_line} | Language: {language}"]
    if symbol_type and symbol_name:
        if parent_class:
            parts.append(f" | {symbol_type} {parent_class}.{symbol_name}")
        else:
            parts.append(f" | {symbol_type} {symbol_name}")
    return "".join(parts) + "\n"


def chunk_file(
    file_path: Path,
    repo_dir: Path,
    chunk_size: int = 80,
    chunk_overlap: int = 10,
) -> list[Chunk]:
    """Split a file into overlapping line-based chunks.

    Args:
        file_path: Absolute path to the file.
        repo_dir: Root of the repo (to compute relative paths).
        chunk_size: Number of lines per chunk.
        chunk_overlap: Number of overlapping lines between chunks.

    Returns:
        List of Chunk objects. Empty list if the file can't be read.

    Interview note: We use a sliding window — start at line 0, take
    `chunk_size` lines, then advance by `chunk_size - chunk_overlap`
    lines. The overlap ensures boundary code appears in 2 chunks.
    """
    # Read file, skip binary/unreadable files gracefully
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("Could not read %s: %s", file_path, e)
        return []

    lines = text.splitlines(keepends=True)  # Keep \n for accurate content
    if not lines:
        return []

    rel_path = str(file_path.relative_to(repo_dir)).replace("\\", "/")
    language = detect_language(file_path)
    stride = max(1, chunk_size - chunk_overlap)  # How far we advance each step

    chunks: list[Chunk] = []

    for start_idx in range(0, len(lines), stride):
        end_idx = min(start_idx + chunk_size, len(lines))

        raw_content = "".join(lines[start_idx:end_idx])
        if not raw_content.strip():
            continue  # Skip empty chunks

        start_line = start_idx + 1        # Convert to 1-indexed
        end_line = end_idx                 # Inclusive

        header = make_context_header(rel_path, start_line, end_line, language)
        content = header + raw_content

        chunk_id = f"{rel_path}:{start_line}-{end_line}"

        chunks.append(Chunk(
            content=content,
            raw_content=raw_content,
            file_path=rel_path,
            start_line=start_line,
            end_line=end_line,
            language=language,
            chunk_id=chunk_id,
        ))

        # If we've consumed all lines, stop
        if end_idx >= len(lines):
            break

    logger.debug("Chunked %s into %d chunks", rel_path, len(chunks))
    return chunks


def chunk_all_files(
    files: list[Path],
    repo_dir: Path,
    chunk_size: int = 80,
    chunk_overlap: int = 10,
) -> list[Chunk]:
    """Chunk all files using sliding-window only (Phase 1 fallback).

    Args:
        files: List of absolute file paths to chunk.
        repo_dir: Root of the repo.
        chunk_size: Lines per chunk.
        chunk_overlap: Overlap lines.

    Returns:
        List of all Chunk objects across all files.
    """
    all_chunks: list[Chunk] = []
    for file_path in files:
        all_chunks.extend(chunk_file(file_path, repo_dir, chunk_size, chunk_overlap))

    logger.info("Total chunks across %d files: %d", len(files), len(all_chunks))
    return all_chunks


# ── Languages supported by tree-sitter ────────────────────────
TREESITTER_LANGUAGES: set[str] = {"python", "javascript", "typescript"}


def chunk_all_files_smart(
    files: list[Path],
    repo_dir: Path,
    chunk_size: int = 80,
    chunk_overlap: int = 10,
) -> list[Chunk]:
    """Chunk files using tree-sitter when possible, sliding-window otherwise.

    Design decision: We import ts_chunker here (not at module level) to
    avoid import errors if tree-sitter is not installed. The sliding-window
    chunker always works as a fallback.

    Args:
        files: List of absolute file paths to chunk.
        repo_dir: Root of the repo.
        chunk_size: Lines per chunk.
        chunk_overlap: Overlap lines.

    Returns:
        List of all Chunk objects across all files.
    """
    # Lazy import to keep tree-sitter optional
    from codelens.app.ingestion.ts_chunker import chunk_file_treesitter

    all_chunks: list[Chunk] = []
    ts_count = 0
    sw_count = 0

    for file_path in files:
        language = detect_language(file_path)

        if language in TREESITTER_LANGUAGES:
            # Try tree-sitter; fall back to sliding-window on failure
            chunks = chunk_file_treesitter(
                file_path, repo_dir, chunk_size, chunk_overlap
            )
            if chunks:
                all_chunks.extend(chunks)
                ts_count += 1
                continue

        # Fallback: sliding-window
        all_chunks.extend(
            chunk_file(file_path, repo_dir, chunk_size, chunk_overlap)
        )
        sw_count += 1

    logger.info(
        "Chunked %d files: %d via tree-sitter, %d via sliding-window. Total chunks: %d",
        len(files), ts_count, sw_count, len(all_chunks),
    )
    return all_chunks
