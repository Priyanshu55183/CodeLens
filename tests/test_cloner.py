"""
Tests for the GitHub URL parser.

Parsing is critical — a bad parse means cloning the wrong repo
or crashing. We test many URL formats here.
"""

import pytest

from codelens.app.ingestion.cloner import parse_github_url


class TestParseGithubUrl:
    """Test GitHub URL parsing."""

    def test_standard_url(self):
        owner, repo = parse_github_url("https://github.com/pallets/click")
        assert owner == "pallets"
        assert repo == "click"

    def test_url_with_git_suffix(self):
        owner, repo = parse_github_url("https://github.com/pallets/click.git")
        assert owner == "pallets"
        assert repo == "click"

    def test_url_with_trailing_slash(self):
        owner, repo = parse_github_url("https://github.com/pallets/click/")
        assert owner == "pallets"
        assert repo == "click"

    def test_url_with_branch(self):
        owner, repo = parse_github_url(
            "https://github.com/pallets/click/tree/main"
        )
        assert owner == "pallets"
        assert repo == "click"

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            parse_github_url("https://gitlab.com/user/repo")

    def test_random_string_raises(self):
        with pytest.raises(ValueError, match="Could not parse"):
            parse_github_url("not a url at all")

    def test_url_with_hyphen_in_name(self):
        owner, repo = parse_github_url(
            "https://github.com/my-org/my-cool-project"
        )
        assert owner == "my-org"
        assert repo == "my-cool-project"
