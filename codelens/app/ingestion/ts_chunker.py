"""
Tree-sitter syntax-aware chunker (Phase 2).

Design decisions:
- We parse each file into an AST and split at SYMBOL BOUNDARIES (functions,
  classes, methods) instead of arbitrary line windows. This means each chunk
  is a complete semantic unit — the embedding captures one whole function
  rather than half of two functions.
- WHY THIS IS BETTER: A sliding-window chunk might split a function at
  line 80, putting the signature in one chunk and the return statement in
  another. When someone asks "what does function X return?", neither chunk
  has the full answer. Tree-sitter chunking keeps the entire function
  together, so the embedding and retrieval work on meaningful units.
- We extract METADATA for each chunk: symbol_name, symbol_type (function,
  class, method), parent_class. This enables the find_definition tool
  in Phase 4 and enriches the context header for better embeddings.
- We extract FILE-LEVEL IMPORTS and attach them to every chunk from that
  file. This helps the LLM understand dependencies without extra lookups.
- For LARGE SYMBOLS (e.g. a 200-line function), we fall back to sliding-
  window within that symbol. This is rare but avoids embedding size issues.
- For MODULE-LEVEL CODE (imports, constants, top-level statements that
  aren't inside any function/class), we create a single "module" chunk.
- If tree-sitter parsing fails for any reason, we return an empty list
  so the caller falls back to sliding-window. Fail-safe, never crash.

Supported languages: Python, JavaScript, TypeScript.
"""

import logging
from pathlib import Path

from tree_sitter import Language, Parser, Node

from codelens.app.ingestion.chunker import (
    Chunk,
    detect_language,
    make_context_header,
    chunk_file,
)

logger = logging.getLogger(__name__)


# ── Language Loading ──────────────────────────────────────────
# We load language objects lazily and cache them. Each language
# grammar is a compiled C library provided by tree-sitter-<lang>.

_PARSERS: dict[str, Parser] = {}


def _get_parser(language: str) -> Parser | None:
    """Get or create a tree-sitter parser for the given language.

    Returns None if the language is not supported.
    Caches parsers since creating them is relatively expensive.
    """
    if language in _PARSERS:
        return _PARSERS[language]

    try:
        if language == "python":
            import tree_sitter_python as tspy
            lang = Language(tspy.language())
        elif language == "javascript":
            import tree_sitter_javascript as tsjs
            lang = Language(tsjs.language())
        elif language == "typescript":
            import tree_sitter_typescript as tsts
            lang = Language(tsts.language_typescript())
        else:
            return None

        parser = Parser(lang)
        _PARSERS[language] = parser
        return parser

    except Exception as e:
        logger.warning("Failed to load tree-sitter for %s: %s", language, e)
        return None


# ── Node Type Mappings ────────────────────────────────────────
# Which AST node types represent "top-level definitions" we want
# as individual chunks, for each supported language.

# Maps language -> set of node types that are top-level definitions
DEFINITION_TYPES: dict[str, set[str]] = {
    "python": {
        "function_definition",
        "class_definition",
        "decorated_definition",  # @decorator above a function/class
    },
    "javascript": {
        "function_declaration",
        "class_declaration",
        "export_statement",       # export function / export class
        "lexical_declaration",    # const foo = () => {} (arrow functions)
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "export_statement",
        "lexical_declaration",
        "interface_declaration",
        "type_alias_declaration",
    },
}

# Maps language -> set of node types that are methods inside a class
METHOD_TYPES: dict[str, set[str]] = {
    "python": {"function_definition"},
    "javascript": {"method_definition"},
    "typescript": {"method_definition"},
}

# Maps language -> set of node types that are import statements
IMPORT_TYPES: dict[str, set[str]] = {
    "python": {"import_statement", "import_from_statement"},
    "javascript": {"import_statement"},
    "typescript": {"import_statement"},
}


# ── Symbol Name Extraction ────────────────────────────────────

def _get_symbol_name(node: Node, language: str) -> str:
    """Extract the symbol name from a definition node.

    Examples:
        function_definition -> the function name
        class_definition    -> the class name
        decorated_definition -> recurse into the inner definition

    Returns empty string if name can't be determined.
    """
    # For decorated definitions, look at the actual definition inside
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return _get_symbol_name(child, language)
        return ""

    # For export statements, look at the declaration inside
    if node.type == "export_statement":
        for child in node.children:
            name = _get_symbol_name(child, language)
            if name:
                return name
        return ""

    # Look for the 'name' or 'identifier' child
    for child in node.children:
        if child.type == "name":          # Python
            return child.text.decode("utf-8") if child.text else ""
        if child.type == "identifier":    # JS/TS
            return child.text.decode("utf-8") if child.text else ""
        if child.type == "property_identifier":  # JS/TS methods
            return child.text.decode("utf-8") if child.text else ""

    return ""


def _get_symbol_type(node: Node, language: str, is_method: bool = False) -> str:
    """Classify the node into a symbol type string.

    Returns one of: "function", "class", "method", "interface", "type", "module".
    """
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return _get_symbol_type(child, language, is_method)
        return "function"

    if node.type == "export_statement":
        for child in node.children:
            st = _get_symbol_type(child, language, is_method)
            if st:
                return st
        return "function"

    if is_method or node.type == "method_definition":
        return "method"

    type_map = {
        "function_definition": "function",
        "function_declaration": "function",
        "class_definition": "class",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
        "lexical_declaration": "function",  # const x = () => {} is treated as function
    }
    return type_map.get(node.type, "function")


def _get_class_name(node: Node, language: str) -> str:
    """If this node is a class definition, return its name."""
    if node.type in ("class_definition", "class_declaration"):
        return _get_symbol_name(node, language)
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("class_definition", "class_declaration"):
                return _get_symbol_name(child, language)
    return ""


# ── Import Extraction ─────────────────────────────────────────

def extract_imports(node: Node, language: str) -> list[str]:
    """Extract all import statements from the file's AST root.

    Returns a list of import strings (the raw text of each import line).
    These get attached to every chunk from this file so the LLM knows
    what modules are available without reading the whole file.
    """
    import_types = IMPORT_TYPES.get(language, set())
    imports: list[str] = []

    for child in node.children:
        if child.type in import_types:
            text = child.text.decode("utf-8") if child.text else ""
            if text:
                imports.append(text.strip())

    return imports


# ── Methods from Class Body ───────────────────────────────────

def _extract_methods(
    class_node: Node,
    class_name: str,
    language: str,
    source_lines: list[str],
    rel_path: str,
    imports_str: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Extract individual method chunks from a class body.

    Design decision: We chunk methods individually (not the whole class
    as one chunk) because:
    1. Large classes can be hundreds of lines — one chunk would be too big.
    2. When someone asks "what does Foo.bar do?", we want the bar method
       specifically, not the entire Foo class.
    3. Each method gets metadata linking it back to its parent class.

    We also create a "class header" chunk containing the class signature,
    docstring, and class-level attributes (everything before the first method).
    """
    method_types = METHOD_TYPES.get(language, set())
    chunks: list[Chunk] = []

    # Find the class body
    body = None
    for child in class_node.children:
        if child.type in ("block", "class_body"):  # Python uses "block", JS/TS use "class_body"
            body = child
            break

    if body is None:
        # No body found — treat the whole class as one chunk
        return []

    # Class header: everything from class start to first method
    first_method_line = None
    methods = []
    for child in body.children:
        if child.type in method_types or child.type == "decorated_definition":
            if first_method_line is None:
                first_method_line = child.start_point[0]  # 0-indexed
            methods.append(child)

    # Create class header chunk (signature + docstring + class vars)
    header_end = first_method_line if first_method_line is not None else class_node.end_point[0] + 1
    header_start = class_node.start_point[0]  # 0-indexed

    if header_end > header_start:
        raw = "".join(source_lines[header_start:header_end])
        if raw.strip():
            start_line = header_start + 1
            end_line = header_end
            header_text = make_context_header(
                rel_path, start_line, end_line, language,
                symbol_name=class_name, symbol_type="class",
            )
            chunks.append(Chunk(
                content=header_text + raw,
                raw_content=raw,
                file_path=rel_path,
                start_line=start_line,
                end_line=end_line,
                language=language,
                chunk_id=f"{rel_path}:{start_line}-{end_line}",
                symbol_name=class_name,
                symbol_type="class",
                parent_class="",
                imports=imports_str,
            ))

    # Create a chunk for each method
    for method_node in methods:
        method_name = _get_symbol_name(method_node, language)
        m_start = method_node.start_point[0]  # 0-indexed
        m_end = method_node.end_point[0] + 1  # exclusive

        raw = "".join(source_lines[m_start:m_end])
        if not raw.strip():
            continue

        start_line = m_start + 1
        end_line = m_end
        line_count = end_line - start_line + 1

        # If method is too large, subdivide with sliding window
        if line_count > chunk_size:
            sub_chunks = _subdivide_large_node(
                source_lines, m_start, m_end, rel_path, language,
                method_name, "method", class_name, imports_str,
                chunk_size, chunk_overlap,
            )
            chunks.extend(sub_chunks)
        else:
            header_text = make_context_header(
                rel_path, start_line, end_line, language,
                symbol_name=method_name, symbol_type="method",
                parent_class=class_name,
            )
            chunks.append(Chunk(
                content=header_text + raw,
                raw_content=raw,
                file_path=rel_path,
                start_line=start_line,
                end_line=end_line,
                language=language,
                chunk_id=f"{rel_path}:{start_line}-{end_line}",
                symbol_name=method_name,
                symbol_type="method",
                parent_class=class_name,
                imports=imports_str,
            ))

    return chunks


# ── Large Node Subdivision ────────────────────────────────────

def _subdivide_large_node(
    source_lines: list[str],
    start_idx: int,
    end_idx: int,
    rel_path: str,
    language: str,
    symbol_name: str,
    symbol_type: str,
    parent_class: str,
    imports_str: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split a large function/method into sliding-window sub-chunks.

    Design decision: When a single function is too big (>chunk_size lines),
    we still want to preserve the symbol metadata. So we apply sliding-window
    WITHIN the function boundaries, and each sub-chunk inherits the function's
    name and type.
    """
    stride = max(1, chunk_size - chunk_overlap)
    chunks: list[Chunk] = []

    for i in range(start_idx, end_idx, stride):
        sub_end = min(i + chunk_size, end_idx)
        raw = "".join(source_lines[i:sub_end])
        if not raw.strip():
            continue

        start_line = i + 1
        end_line = sub_end

        header = make_context_header(
            rel_path, start_line, end_line, language,
            symbol_name=symbol_name, symbol_type=symbol_type,
            parent_class=parent_class,
        )
        chunks.append(Chunk(
            content=header + raw,
            raw_content=raw,
            file_path=rel_path,
            start_line=start_line,
            end_line=end_line,
            language=language,
            chunk_id=f"{rel_path}:{start_line}-{end_line}",
            symbol_name=symbol_name,
            symbol_type=symbol_type,
            parent_class=parent_class,
            imports=imports_str,
        ))

        if sub_end >= end_idx:
            break

    return chunks


# ── Main Entry Point ──────────────────────────────────────────

def chunk_file_treesitter(
    file_path: Path,
    repo_dir: Path,
    chunk_size: int = 80,
    chunk_overlap: int = 10,
) -> list[Chunk]:
    """Chunk a file using tree-sitter syntax-aware parsing.

    High-level algorithm:
    1. Parse the file into an AST.
    2. Walk top-level nodes:
       - Import statements → collect into imports list
       - Function/class definitions → extract as individual chunks
       - Classes → further split into method-level chunks
       - Everything else → collect into "module-level" chunk
    3. Attach import info to every chunk from this file.
    4. For very large symbols, subdivide with sliding window.

    Args:
        file_path: Absolute path to the file.
        repo_dir: Root of the repo.
        chunk_size: Max lines per chunk (large symbols get split).
        chunk_overlap: Overlap lines when splitting large symbols.

    Returns:
        List of Chunk objects. Empty list if parsing fails (caller
        should fall back to sliding-window).
    """
    # Read the file
    try:
        source_text = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError) as e:
        logger.warning("Could not read %s: %s", file_path, e)
        return []

    if not source_text.strip():
        return []

    rel_path = str(file_path.relative_to(repo_dir)).replace("\\", "/")
    language = detect_language(file_path)

    # Get the parser for this language
    parser = _get_parser(language)
    if parser is None:
        return []  # Unsupported language → fall back to sliding-window

    # Parse
    source_bytes = source_text.encode("utf-8")
    try:
        tree = parser.parse(source_bytes)
    except Exception as e:
        logger.warning("Tree-sitter parse failed for %s: %s", file_path, e)
        return []

    root = tree.root_node
    source_lines = source_text.splitlines(keepends=True)

    # Extract imports
    imports = extract_imports(root, language)
    imports_str = ", ".join(imports)

    # Get the definition types for this language
    def_types = DEFINITION_TYPES.get(language, set())

    chunks: list[Chunk] = []
    module_level_ranges: list[tuple[int, int]] = []  # Ranges of non-definition top-level code

    for child in root.children:
        # Skip import statements (already extracted)
        if child.type in IMPORT_TYPES.get(language, set()):
            # Still include imports in module-level chunk
            module_level_ranges.append((child.start_point[0], child.end_point[0] + 1))
            continue

        # Skip comments (but include in module-level chunk)
        if child.type == "comment":
            module_level_ranges.append((child.start_point[0], child.end_point[0] + 1))
            continue

        # Is this a definition?
        if child.type in def_types:
            class_name = _get_class_name(child, language)

            if class_name:
                # It's a class — extract methods separately
                method_chunks = _extract_methods(
                    child, class_name, language, source_lines,
                    rel_path, imports_str, chunk_size, chunk_overlap,
                )
                if method_chunks:
                    chunks.extend(method_chunks)
                else:
                    # Class with no methods — treat as single chunk
                    _add_definition_chunk(
                        child, source_lines, rel_path, language,
                        imports_str, chunk_size, chunk_overlap, chunks,
                    )
            else:
                # It's a function/other definition
                _add_definition_chunk(
                    child, source_lines, rel_path, language,
                    imports_str, chunk_size, chunk_overlap, chunks,
                )
        else:
            # Module-level code (constants, expressions, etc.)
            module_level_ranges.append((child.start_point[0], child.end_point[0] + 1))

    # Create a single "module" chunk for all module-level code
    if module_level_ranges:
        _add_module_chunk(
            module_level_ranges, source_lines, rel_path, language,
            imports_str, chunk_size, chunk_overlap, chunks,
        )

    logger.debug(
        "Tree-sitter chunked %s: %d chunks (%d imports)",
        rel_path, len(chunks), len(imports),
    )
    return chunks


def _add_definition_chunk(
    node: Node,
    source_lines: list[str],
    rel_path: str,
    language: str,
    imports_str: str,
    chunk_size: int,
    chunk_overlap: int,
    chunks: list[Chunk],
) -> None:
    """Add a single definition (function/class) as a chunk."""
    symbol_name = _get_symbol_name(node, language)
    symbol_type = _get_symbol_type(node, language)

    start_idx = node.start_point[0]  # 0-indexed
    end_idx = node.end_point[0] + 1  # exclusive

    raw = "".join(source_lines[start_idx:end_idx])
    if not raw.strip():
        return

    start_line = start_idx + 1
    end_line = end_idx
    line_count = end_line - start_line + 1

    if line_count > chunk_size:
        # Too big — subdivide
        sub_chunks = _subdivide_large_node(
            source_lines, start_idx, end_idx, rel_path, language,
            symbol_name, symbol_type, "", imports_str,
            chunk_size, chunk_overlap,
        )
        chunks.extend(sub_chunks)
    else:
        header = make_context_header(
            rel_path, start_line, end_line, language,
            symbol_name=symbol_name, symbol_type=symbol_type,
        )
        chunks.append(Chunk(
            content=header + raw,
            raw_content=raw,
            file_path=rel_path,
            start_line=start_line,
            end_line=end_line,
            language=language,
            chunk_id=f"{rel_path}:{start_line}-{end_line}",
            symbol_name=symbol_name,
            symbol_type=symbol_type,
            parent_class="",
            imports=imports_str,
        ))


def _add_module_chunk(
    ranges: list[tuple[int, int]],
    source_lines: list[str],
    rel_path: str,
    language: str,
    imports_str: str,
    chunk_size: int,
    chunk_overlap: int,
    chunks: list[Chunk],
) -> None:
    """Create a chunk for module-level code (imports, constants, etc.).

    We merge all non-definition ranges into one chunk. If it's too big,
    we subdivide.
    """
    if not ranges:
        return

    # Find the overall range
    min_start = min(r[0] for r in ranges)
    max_end = max(r[1] for r in ranges)

    raw = "".join(source_lines[min_start:max_end])
    if not raw.strip():
        return

    start_line = min_start + 1
    end_line = max_end
    line_count = end_line - start_line + 1

    if line_count > chunk_size:
        sub_chunks = _subdivide_large_node(
            source_lines, min_start, max_end, rel_path, language,
            "", "module", "", imports_str, chunk_size, chunk_overlap,
        )
        chunks.extend(sub_chunks)
    else:
        header = make_context_header(
            rel_path, start_line, end_line, language,
            symbol_name="", symbol_type="module",
        )
        chunks.append(Chunk(
            content=header + raw,
            raw_content=raw,
            file_path=rel_path,
            start_line=start_line,
            end_line=end_line,
            language=language,
            chunk_id=f"{rel_path}:{start_line}-{end_line}",
            symbol_name="",
            symbol_type="module",
            parent_class="",
            imports=imports_str,
        ))
