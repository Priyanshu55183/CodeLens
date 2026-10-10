"""
Tests for the sliding-window chunker.

These tests verify chunk boundaries, overlap, context headers,
metadata, and edge cases (empty files, single-line files, binary).
"""

from pathlib import Path

import pytest

from codelens.app.ingestion.chunker import (
    chunk_file,
    chunk_all_files,
    detect_language,
    make_context_header,
    Chunk,
)


class TestDetectLanguage:
    """Test language detection from file extensions."""

    def test_python(self):
        assert detect_language(Path("foo.py")) == "python"

    def test_javascript(self):
        assert detect_language(Path("index.js")) == "javascript"

    def test_typescript(self):
        assert detect_language(Path("app.ts")) == "typescript"

    def test_tsx(self):
        assert detect_language(Path("Component.tsx")) == "typescript"

    def test_unknown_extension(self):
        assert detect_language(Path("data.xyz")) == "unknown"

    def test_dockerfile(self):
        assert detect_language(Path("Dockerfile")) == "dockerfile"

    def test_makefile(self):
        assert detect_language(Path("Makefile")) == "makefile"

    def test_markdown(self):
        assert detect_language(Path("README.md")) == "markdown"


class TestContextHeader:
    """Test the context header prepended to chunks."""

    def test_format(self):
        header = make_context_header("src/main.py", 1, 80, "python")
        assert header == "# File: src/main.py | Lines: 1-80 | Language: python\n"

    def test_contains_file_path(self):
        header = make_context_header("lib/utils.js", 10, 50, "javascript")
        assert "lib/utils.js" in header


class TestChunkFile:
    """Test the core chunking logic."""

    def test_small_file_single_chunk(self, tmp_path: Path):
        """A file smaller than chunk_size should produce exactly 1 chunk."""
        f = tmp_path / "small.py"
        f.write_text("line1\nline2\nline3\n")

        chunks = chunk_file(f, tmp_path, chunk_size=80, chunk_overlap=10)
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 3

    def test_chunk_overlap(self, tmp_path: Path):
        """Chunks should overlap by chunk_overlap lines."""
        # Create a 20-line file, chunk_size=10, overlap=3
        f = tmp_path / "overlap.py"
        lines = [f"line {i}\n" for i in range(1, 21)]
        f.write_text("".join(lines))

        chunks = chunk_file(f, tmp_path, chunk_size=10, chunk_overlap=3)

        # With 20 lines, chunk_size=10, stride=7: chunks start at 0, 7, 14
        assert len(chunks) == 3
        # First chunk: lines 1-10
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 10
        # Second chunk: lines 8-17 (overlaps with first)
        assert chunks[1].start_line == 8
        assert chunks[1].end_line == 17
        # Third chunk: lines 15-20 (last chunk may be shorter)
        assert chunks[2].start_line == 15
        assert chunks[2].end_line == 20

    def test_chunk_has_context_header(self, tmp_path: Path):
        """Chunk content should start with the context header."""
        f = tmp_path / "header.py"
        f.write_text("x = 1\n")

        chunks = chunk_file(f, tmp_path, chunk_size=80)
        assert chunks[0].content.startswith("# File:")

    def test_raw_content_no_header(self, tmp_path: Path):
        """raw_content should NOT include the context header."""
        f = tmp_path / "raw.py"
        f.write_text("x = 1\n")

        chunks = chunk_file(f, tmp_path, chunk_size=80)
        assert not chunks[0].raw_content.startswith("# File:")
        assert "x = 1" in chunks[0].raw_content

    def test_chunk_metadata(self, tmp_path: Path):
        """Metadata dict should contain all required fields."""
        f = tmp_path / "meta.py"
        f.write_text("a = 1\nb = 2\n")

        chunks = chunk_file(f, tmp_path, chunk_size=80)
        meta = chunks[0].metadata
        assert meta["file_path"] == "meta.py"
        assert meta["start_line"] == 1
        assert meta["end_line"] == 2
        assert meta["language"] == "python"

    def test_chunk_id_format(self, tmp_path: Path):
        """chunk_id should be 'relative_path:start-end'."""
        f = tmp_path / "idtest.py"
        f.write_text("hello\n")

        chunks = chunk_file(f, tmp_path, chunk_size=80)
        assert chunks[0].chunk_id == "idtest.py:1-1"

    def test_empty_file_no_chunks(self, tmp_path: Path):
        """An empty file should produce no chunks."""
        f = tmp_path / "empty.py"
        f.write_text("")

        chunks = chunk_file(f, tmp_path, chunk_size=80)
        assert chunks == []

    def test_whitespace_only_file_no_chunks(self, tmp_path: Path):
        """A file with only whitespace should produce no chunks."""
        f = tmp_path / "blank.py"
        f.write_text("   \n  \n\n")

        chunks = chunk_file(f, tmp_path, chunk_size=80)
        assert chunks == []

    def test_relative_path_uses_forward_slashes(self, tmp_path: Path):
        """File paths in chunks should always use forward slashes."""
        subdir = tmp_path / "src" / "utils"
        subdir.mkdir(parents=True)
        f = subdir / "helpers.py"
        f.write_text("def helper(): pass\n")

        chunks = chunk_file(f, tmp_path, chunk_size=80)
        assert "\\" not in chunks[0].file_path
        assert chunks[0].file_path == "src/utils/helpers.py"


class TestChunkAllFiles:
    """Test chunking across multiple files."""

    def test_combines_chunks(self, tmp_path: Path):
        """Should return chunks from all files."""
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("y = 2\n")
        files = [tmp_path / "a.py", tmp_path / "b.py"]

        chunks = chunk_all_files(files, tmp_path, chunk_size=80)
        assert len(chunks) == 2
        paths = {c.file_path for c in chunks}
        assert paths == {"a.py", "b.py"}

    def test_empty_file_list(self, tmp_path: Path):
        """Empty input should return empty output."""
        assert chunk_all_files([], tmp_path) == []
