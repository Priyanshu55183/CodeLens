"""
Filter out files we don't want to index.

Design decisions:
- We skip files early (before reading content) using directory names,
  file extensions, and file names. This is much faster than reading
  every file and then deciding.
- The skip lists are intentionally conservative — we'd rather miss some
  junk than accidentally skip real code.
- We also skip files > 1 MB (likely generated or binary).
- Each filter function is pure (no side effects) so it's easy to test.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Directories to always skip ────────────────────────────────
SKIP_DIRS: set[str] = {
    ".git",
    "node_modules",
    "__pycache__",
    ".next",
    ".nuxt",
    "dist",
    "build",
    "out",
    ".cache",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "venv",
    ".venv",
    "env",
    ".env",
    ".eggs",
    "egg-info",
    "vendor",
    "coverage",
    ".nyc_output",
    ".terraform",
    "target",        # Rust / Java build
    "bin",           # Go / compiled output
}

# ── Files to skip by exact name ───────────────────────────────
SKIP_FILES: set[str] = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Pipfile.lock",
    "poetry.lock",
    "composer.lock",
    "Gemfile.lock",
    "Cargo.lock",
    ".DS_Store",
    "Thumbs.db",
}

# ── Extensions that are binary or not useful ──────────────────
SKIP_EXTENSIONS: set[str] = {
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".bmp",
    # Fonts
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    # Compiled / binary
    ".exe", ".dll", ".so", ".dylib", ".o", ".a",
    ".pyc", ".pyo", ".class", ".jar",
    ".wasm",
    # Data (usually large / generated)
    ".sqlite", ".db", ".sqlite3",
    ".pkl", ".pickle", ".npy", ".npz",
    ".parquet", ".arrow", ".feather",
    # Media
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv",
    # Documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    # Maps
    ".map",
    # Min files
    ".min.js", ".min.css",
}

# Maximum file size in bytes (1 MB)
MAX_FILE_SIZE: int = 1_000_000


def should_skip_dir(dir_name: str) -> bool:
    """Return True if this directory should be skipped entirely.
    
    Args:
        dir_name: The name of the directory (not the full path).
    """
    return dir_name in SKIP_DIRS or dir_name.startswith(".")


def should_skip_file(file_path: Path) -> bool:
    """Return True if this file should be skipped.
    
    Checks (in order, cheapest first):
    1. File name is in the skip list
    2. File extension is in the skip list
    3. File is too large

    Args:
        file_path: Path to the file (must exist for size check).
    """
    # Check name
    if file_path.name in SKIP_FILES:
        return True

    # Check extension (handle compound extensions like .min.js)
    suffix = file_path.suffix.lower()
    if suffix in SKIP_EXTENSIONS:
        return True
    # Check compound extension
    if file_path.name.endswith(".min.js") or file_path.name.endswith(".min.css"):
        return True

    # Check size
    try:
        if file_path.stat().st_size > MAX_FILE_SIZE:
            logger.debug("Skipping large file: %s", file_path)
            return True
    except OSError:
        # File might have been deleted or be unreadable
        return True

    return False


def collect_files(repo_dir: Path) -> list[Path]:
    """Walk the repo and return all files that pass our filters.
    
    Uses os.walk-style traversal with directory pruning — when we hit
    a directory in SKIP_DIRS, we don't recurse into it at all (faster
    than collecting everything and filtering after).

    Args:
        repo_dir: Root directory of the cloned repo.

    Returns:
        List of absolute paths to files worth indexing.
    """
    files: list[Path] = []

    for item in sorted(repo_dir.rglob("*")):
        # Skip files inside skipped directories
        # We check the RELATIVE path parts (not absolute) to avoid
        # false positives from parent directories like C:\Users etc.
        try:
            rel_parts = item.relative_to(repo_dir).parts
        except ValueError:
            continue
        if any(should_skip_dir(part) for part in rel_parts):
            continue

        if item.is_file() and not should_skip_file(item):
            files.append(item)

    logger.info("Collected %d files from %s", len(files), repo_dir)
    return files
