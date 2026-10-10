"""
Tests for the file filter module.

These tests verify that our filter logic correctly skips junk files
and directories without accidentally filtering out real code.
"""

import tempfile
from pathlib import Path

import pytest

from codelens.app.ingestion.filters import (
    should_skip_dir,
    should_skip_file,
    collect_files,
    MAX_FILE_SIZE,
)


class TestShouldSkipDir:
    """Test directory filtering."""

    def test_skips_node_modules(self):
        assert should_skip_dir("node_modules") is True

    def test_skips_git(self):
        assert should_skip_dir(".git") is True

    def test_skips_pycache(self):
        assert should_skip_dir("__pycache__") is True

    def test_skips_venv(self):
        assert should_skip_dir("venv") is True

    def test_skips_dot_dirs(self):
        """Any directory starting with . should be skipped."""
        assert should_skip_dir(".hidden") is True
        assert should_skip_dir(".config") is True

    def test_allows_src(self):
        assert should_skip_dir("src") is False

    def test_allows_lib(self):
        assert should_skip_dir("lib") is False

    def test_allows_tests(self):
        assert should_skip_dir("tests") is False


class TestShouldSkipFile:
    """Test file filtering."""

    def test_skips_lockfiles(self, tmp_path: Path):
        lockfile = tmp_path / "package-lock.json"
        lockfile.write_text("{}")
        assert should_skip_file(lockfile) is True

    def test_skips_images(self, tmp_path: Path):
        img = tmp_path / "logo.png"
        img.write_bytes(b"\x89PNG")
        assert should_skip_file(img) is True

    def test_skips_large_files(self, tmp_path: Path):
        big_file = tmp_path / "huge.py"
        big_file.write_text("x" * (MAX_FILE_SIZE + 1))
        assert should_skip_file(big_file) is True

    def test_allows_python(self, tmp_path: Path):
        py_file = tmp_path / "main.py"
        py_file.write_text("print('hello')")
        assert should_skip_file(py_file) is False

    def test_allows_javascript(self, tmp_path: Path):
        js_file = tmp_path / "index.js"
        js_file.write_text("console.log('hi')")
        assert should_skip_file(js_file) is False

    def test_allows_markdown(self, tmp_path: Path):
        md = tmp_path / "README.md"
        md.write_text("# Hello")
        assert should_skip_file(md) is False

    def test_skips_minified_js(self, tmp_path: Path):
        minjs = tmp_path / "bundle.min.js"
        minjs.write_text("var a=1;")
        assert should_skip_file(minjs) is True


class TestCollectFiles:
    """Test the full file collection pipeline."""

    def test_collects_python_files(self, tmp_path: Path):
        """Should find .py files in subdirectories."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("print('hi')")
        (src / "utils.py").write_text("def foo(): pass")

        files = collect_files(tmp_path)
        names = [f.name for f in files]
        assert "main.py" in names
        assert "utils.py" in names

    def test_skips_node_modules_contents(self, tmp_path: Path):
        """Files inside node_modules should never appear."""
        nm = tmp_path / "node_modules" / "express"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("module.exports = {}")
        (tmp_path / "app.js").write_text("const app = require('express')")

        files = collect_files(tmp_path)
        # Use relative paths to avoid false matches from the pytest
        # tmp dir name (which ironically contains "node_modules")
        rel_paths = [str(f.relative_to(tmp_path)) for f in files]
        assert not any("node_modules" in p for p in rel_paths)
        assert any("app.js" in p for p in rel_paths)

    def test_empty_dir_returns_empty(self, tmp_path: Path):
        """Empty directory should return empty list."""
        assert collect_files(tmp_path) == []
