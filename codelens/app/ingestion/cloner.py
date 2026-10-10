"""
Clone a GitHub repo and pin the exact commit SHA.

Design decisions:
- We clone with depth=1 (shallow) to save time and disk space.
  Full history is only needed for get_file_history (Phase 4+).
- We PIN the HEAD commit SHA immediately after cloning. Every chunk,
  citation, and verification step uses this SHA. This guarantees that
  even if the remote repo changes, our indexed data stays consistent.
- If the repo was already cloned, we skip re-cloning and reuse the
  existing directory (idempotent).
"""

import re
import logging
from pathlib import Path

from git import Repo

logger = logging.getLogger(__name__)


def parse_github_url(url: str) -> tuple[str, str]:
    """Extract (owner, repo_name) from a GitHub URL.

    Supports:
        https://github.com/owner/repo
        https://github.com/owner/repo.git
        https://github.com/owner/repo/tree/branch

    Returns:
        Tuple of (owner, repo_name) with no trailing .git

    Raises:
        ValueError: If the URL doesn't match expected GitHub patterns.
    """
    # Strip trailing slashes and .git suffix
    url = url.rstrip("/")
    pattern = r"github\.com/([^/]+)/([^/.]+)"
    match = re.search(pattern, url)
    if not match:
        raise ValueError(
            f"Could not parse GitHub URL: {url}. "
            "Expected format: https://github.com/owner/repo"
        )
    return match.group(1), match.group(2)


def clone_repo(
    url: str,
    repos_dir: Path,
    github_token: str = "",
) -> tuple[Path, str]:
    """Clone a GitHub repository and return (local_path, commit_sha).

    Args:
        url: GitHub repository URL.
        repos_dir: Parent directory where repos are stored.
        github_token: Optional token for private repos.

    Returns:
        Tuple of (path to cloned repo, pinned commit SHA).

    Interview note: We return the SHA so every downstream component
    (chunks, citations, verification) references the exact same snapshot.
    This is critical for the accuracy guarantee.
    """
    owner, repo_name = parse_github_url(url)
    repo_dir = repos_dir / owner / repo_name

    if repo_dir.exists() and (repo_dir / ".git").exists():
        # Already cloned — reuse it
        logger.info("Repo already cloned at %s, reusing", repo_dir)
        repo = Repo(repo_dir)
        sha = repo.head.commit.hexsha
        logger.info("Pinned commit: %s", sha)
        return repo_dir, sha

    # Build the clone URL (with token if provided)
    if github_token:
        # Insert token for authentication
        clone_url = url.replace(
            "https://github.com",
            f"https://{github_token}@github.com",
        )
    else:
        clone_url = url

    logger.info("Cloning %s/%s into %s ...", owner, repo_name, repo_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)

    repo = Repo.clone_from(
        clone_url,
        str(repo_dir),
        depth=1,  # Shallow clone — saves time and disk
    )

    sha = repo.head.commit.hexsha
    logger.info("Cloned successfully. Pinned commit: %s", sha)
    return repo_dir, sha
