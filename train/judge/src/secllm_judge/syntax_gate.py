"""Deterministic tree-sitter gate: a patch may not add syntax errors.

Runs before any LLM call. A patched file whose ERROR/MISSING node count exceeds
the original's is a syntax regression and gets ``verdict=fail`` without a judge
call (brace mismatches were the most common failure in external evaluation).
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_go
import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_python
import tree_sitter_typescript
from tree_sitter import Language, Parser

from secllm_judge.schemas import SyntaxGateResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tree_sitter import Node


def normalize_language(language: str) -> str:
    """Reject unsupported grammars instead of silently parsing them as Java."""
    aliases = {
        "java": "java",
        "javascript": "javascript",
        "js": "javascript",
        "c": "c",
        "cpp": "cpp",
        "c++": "cpp",
        "go": "go",
        "golang": "go",
        "python": "python",
        "py": "python",
        "typescript": "typescript",
        "ts": "typescript",
        "tsx": "tsx",
    }
    try:
        return aliases[language.strip().casefold()]
    except KeyError as error:
        msg = f"unsupported judge language: {language}"
        raise ValueError(msg) from error


@cache
def _parser(language: str) -> Parser:
    grammars = {
        "java": tree_sitter_java.language,
        "javascript": tree_sitter_javascript.language,
        "c": tree_sitter_c.language,
        "cpp": tree_sitter_cpp.language,
        "go": tree_sitter_go.language,
        "python": tree_sitter_python.language,
        "typescript": tree_sitter_typescript.language_typescript,
        "tsx": tree_sitter_typescript.language_tsx,
    }
    return Parser(Language(grammars[language]()))


def count_error_nodes(source: str, *, language: str = "java") -> int:
    """Number of ERROR or MISSING nodes in the parse tree (0 = clean)."""
    tree = _parser(normalize_language(language)).parse(source.encode("utf-8"))
    root: Node = tree.root_node
    if not root.has_error:
        return 0
    count = 0
    stack: list[Node] = [root]
    while stack:
        node = stack.pop()
        if node.is_error or node.is_missing:
            count += 1
        stack.extend(child for child in node.children if child.has_error)
    return count


def syntax_gate(
    original: Sequence[str], patched: Sequence[str], *, language: str = "java"
) -> SyntaxGateResult:
    before = count_error_nodes("\n".join(original), language=language)
    after = count_error_nodes("\n".join(patched), language=language)
    return SyntaxGateResult(
        passed=after <= before,
        error_nodes_before=before,
        error_nodes_after=after,
    )
