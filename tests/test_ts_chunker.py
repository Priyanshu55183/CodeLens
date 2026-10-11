"""
Unit tests for the Tree-sitter syntax-aware chunker (Phase 2).

Verifies AST-based chunking across Python, JavaScript, and TypeScript,
testing symbol name extraction, class/method extraction, decorator handling,
import extraction, module-level statement isolation, large symbol splitting,
and fallback behaviors.
"""

from pathlib import Path
import pytest

from codelens.app.ingestion.ts_chunker import (
    chunk_file_treesitter,
    extract_imports,
    _get_parser,
    _get_symbol_name,
    _get_symbol_type,
)
from codelens.app.ingestion.chunker import chunk_all_files_smart


class TestPythonChunking:
    """Tests for Python AST chunking."""

    def test_python_function_extraction(self, tmp_path: Path):
        f = tmp_path / "simple.py"
        f.write_text("def calculate_sum(a, b):\n    \"\"\"Sum two numbers.\"\"\"\n    return a + b\n")

        chunks = chunk_file_treesitter(f, tmp_path)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.symbol_name == "calculate_sum"
        assert c.symbol_type == "function"
        assert c.start_line == 1
        assert c.end_line == 3
        assert "calculate_sum" in c.content

    def test_python_decorated_function(self, tmp_path: Path):
        f = tmp_path / "decorated.py"
        f.write_text("@my_decorator\n@other_decorator\ndef decorated_fn():\n    pass\n")

        chunks = chunk_file_treesitter(f, tmp_path)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.symbol_name == "decorated_fn"
        assert c.symbol_type == "function"
        assert c.start_line == 1
        assert c.end_line == 4
        assert "@my_decorator" in c.content

    def test_python_class_and_methods(self, tmp_path: Path):
        code = (
            "class BankAccount:\n"
            "    \"\"\"A bank account.\"\"\"\n"
            "    rate = 0.05\n\n"
            "    def __init__(self, balance):\n"
            "        self.balance = balance\n\n"
            "    @property\n"
            "    def is_positive(self):\n"
            "        return self.balance > 0\n"
        )
        f = tmp_path / "bank.py"
        f.write_text(code)

        chunks = chunk_file_treesitter(f, tmp_path)
        # Should have: class header chunk, __init__ method chunk, is_positive method chunk
        assert len(chunks) == 3

        header = chunks[0]
        assert header.symbol_name == "BankAccount"
        assert header.symbol_type == "class"
        assert "rate = 0.05" in header.content

        init_method = chunks[1]
        assert init_method.symbol_name == "__init__"
        assert init_method.symbol_type == "method"
        assert init_method.parent_class == "BankAccount"

        prop_method = chunks[2]
        assert prop_method.symbol_name == "is_positive"
        assert prop_method.symbol_type == "method"
        assert prop_method.parent_class == "BankAccount"
        assert "@property" in prop_method.content

    def test_python_class_without_methods(self, tmp_path: Path):
        f = tmp_path / "empty_class.py"
        f.write_text("class EmptyConfig:\n    pass\n")

        chunks = chunk_file_treesitter(f, tmp_path)
        assert len(chunks) == 1
        assert chunks[0].symbol_name == "EmptyConfig"
        assert chunks[0].symbol_type == "class"

    def test_python_imports_attached_to_all_chunks(self, tmp_path: Path):
        code = (
            "import os\n"
            "from pathlib import Path\n\n"
            "def worker():\n"
            "    return os.name\n"
        )
        f = tmp_path / "worker.py"
        f.write_text(code)

        chunks = chunk_file_treesitter(f, tmp_path)
        # Chunks: module chunk (imports) and function chunk (worker)
        fn_chunks = [c for c in chunks if c.symbol_type == "function"]
        assert len(fn_chunks) == 1
        assert "import os" in fn_chunks[0].imports
        assert "from pathlib import Path" in fn_chunks[0].imports

    def test_python_module_level_separation(self, tmp_path: Path):
        """Module statements should not duplicate intermediate functions."""
        code = (
            "import math\n\n"
            "def square(x):\n"
            "    return x * x\n\n"
            "GLOBAL_CONST = 100\n"
        )
        f = tmp_path / "calc.py"
        f.write_text(code)

        chunks = chunk_file_treesitter(f, tmp_path)
        types_and_names = [(c.symbol_type, c.symbol_name) for c in chunks]
        assert ("function", "square") in types_and_names

        # The function should NOT be duplicated inside a module chunk
        module_chunks = [c for c in chunks if c.symbol_type == "module"]
        for mc in module_chunks:
            assert "def square" not in mc.raw_content


class TestJavaScriptChunking:
    """Tests for JavaScript AST chunking."""

    def test_js_function_declaration(self, tmp_path: Path):
        f = tmp_path / "app.js"
        f.write_text("function add(a, b) {\n  return a + b;\n}\n")

        chunks = chunk_file_treesitter(f, tmp_path)
        assert len(chunks) == 1
        assert chunks[0].symbol_name == "add"
        assert chunks[0].symbol_type == "function"

    def test_js_arrow_function(self, tmp_path: Path):
        f = tmp_path / "arrow.js"
        f.write_text("const multiply = (a, b) => {\n  return a * b;\n};\n")

        chunks = chunk_file_treesitter(f, tmp_path)
        assert len(chunks) == 1
        assert chunks[0].symbol_name == "multiply"
        assert chunks[0].symbol_type == "function"

    def test_js_exported_function(self, tmp_path: Path):
        f = tmp_path / "export_fn.js"
        f.write_text("export function greet(name) {\n  return `Hello ${name}`;\n}\n")

        chunks = chunk_file_treesitter(f, tmp_path)
        assert len(chunks) == 1
        assert chunks[0].symbol_name == "greet"
        assert chunks[0].symbol_type == "function"

    def test_js_exported_class(self, tmp_path: Path):
        code = (
            "export class Service {\n"
            "  constructor(url) {\n"
            "    this.url = url;\n"
            "  }\n\n"
            "  fetch() {\n"
            "    return true;\n"
            "  }\n"
            "}\n"
        )
        f = tmp_path / "service.js"
        f.write_text(code)

        chunks = chunk_file_treesitter(f, tmp_path)
        assert len(chunks) == 3
        assert chunks[0].symbol_type == "class"
        assert chunks[0].symbol_name == "Service"
        assert "export class Service" in chunks[0].content

        methods = [c for c in chunks if c.symbol_type == "method"]
        assert len(methods) == 2
        method_names = {m.symbol_name for m in methods}
        assert method_names == {"constructor", "fetch"}
        assert all(m.parent_class == "Service" for m in methods)


class TestTypeScriptChunking:
    """Tests for TypeScript AST chunking."""

    def test_ts_interface_and_type(self, tmp_path: Path):
        code = (
            "export interface UserProfile {\n"
            "  id: string;\n"
            "  name: string;\n"
            "}\n\n"
            "export type Status = 'active' | 'inactive';\n"
        )
        f = tmp_path / "types.ts"
        f.write_text(code)

        chunks = chunk_file_treesitter(f, tmp_path)
        assert len(chunks) == 2

        interface_chunk = chunks[0]
        assert interface_chunk.symbol_name == "UserProfile"
        assert interface_chunk.symbol_type == "interface"

        type_chunk = chunks[1]
        assert type_chunk.symbol_name == "Status"
        assert type_chunk.symbol_type == "type"

    def test_ts_class_with_typed_methods(self, tmp_path: Path):
        code = (
            "export class OrderProcessor {\n"
            "  private status: string = 'idle';\n\n"
            "  async process(orderId: string): Promise<boolean> {\n"
            "    this.status = 'processing';\n"
            "    return true;\n"
            "  }\n"
            "}\n"
        )
        f = tmp_path / "orders.ts"
        f.write_text(code)

        chunks = chunk_file_treesitter(f, tmp_path)
        assert len(chunks) == 2
        assert chunks[0].symbol_type == "class"
        assert chunks[0].symbol_name == "OrderProcessor"

        method_chunk = chunks[1]
        assert method_chunk.symbol_name == "process"
        assert method_chunk.symbol_type == "method"
        assert method_chunk.parent_class == "OrderProcessor"


class TestLargeNodeSubdivision:
    """Tests splitting large functions/methods exceeding chunk_size."""

    def test_large_function_subdivision(self, tmp_path: Path):
        lines = ["def huge_function():\n"]
        for i in range(120):
            lines.append(f"    step_{i} = {i}\n")
        lines.append("    return True\n")

        f = tmp_path / "large.py"
        f.write_text("".join(lines))

        # Chunk size 50, overlap 10 -> 122 lines should result in multiple sub-chunks
        chunks = chunk_file_treesitter(f, tmp_path, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1
        for c in chunks:
            assert c.symbol_name == "huge_function"
            assert c.symbol_type == "function"
            assert c.start_line <= c.end_line


class TestFallbackAndSmartChunker:
    """Tests fallback behaviors and chunk_all_files_smart."""

    def test_unsupported_language_returns_empty(self, tmp_path: Path):
        f = tmp_path / "notes.txt"
        f.write_text("Just some plain text notes.\n")
        assert chunk_file_treesitter(f, tmp_path) == []

    def test_empty_or_whitespace_file(self, tmp_path: Path):
        f1 = tmp_path / "empty.py"
        f1.write_text("")
        f2 = tmp_path / "blank.py"
        f2.write_text("   \n\n\t\n")

        assert chunk_file_treesitter(f1, tmp_path) == []
        assert chunk_file_treesitter(f2, tmp_path) == []

    def test_nonexistent_file(self, tmp_path: Path):
        f = tmp_path / "does_not_exist.py"
        assert chunk_file_treesitter(f, tmp_path) == []

    def test_chunk_all_files_smart_mixed_repo(self, tmp_path: Path):
        py_file = tmp_path / "main.py"
        py_file.write_text("def run():\n    pass\n")

        ts_file = tmp_path / "index.ts"
        ts_file.write_text("export const answer = () => 42;\n")

        md_file = tmp_path / "README.md"
        md_file.write_text("# Project\nDescription here\n")

        files = [py_file, ts_file, md_file]
        chunks = chunk_all_files_smart(files, tmp_path)

        # Python & TS chunks should have symbols
        py_chunks = [c for c in chunks if c.file_path == "main.py"]
        ts_chunks = [c for c in chunks if c.file_path == "index.ts"]
        md_chunks = [c for c in chunks if c.file_path == "README.md"]

        assert len(py_chunks) >= 1
        assert py_chunks[0].symbol_name == "run"

        assert len(ts_chunks) >= 1
        assert ts_chunks[0].symbol_name == "answer"

        # Markdown chunk should fall back to sliding window (symbol_name empty)
        assert len(md_chunks) == 1
        assert md_chunks[0].symbol_name == ""
