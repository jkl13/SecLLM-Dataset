"""Apply add/delete wire operations and render what the judge sees.

The wire contract mirrors the harness consumer contract: all coordinates are
original-file, 1-based; ``add`` inserts before original ``start_line``;
``delete`` removes ``start_line..end_line`` inclusive. Operations never chain
(an ``add`` after a ``delete`` at the same line is a replacement).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from typing import TYPE_CHECKING, final

from secllm_judge.schemas import (
    CodeMode,
    CodeModification,
    ModificationType,
    RenderedPatch,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

FULL_FILE_MAX_LINES = 300
DIFF_CONTEXT_LINES = 15
ISSUE_WINDOW_LINES = 40
MODIFICATION_WINDOW_LINES = 15
HEADER_MAX_LINES = 120

_WS = re.compile(r"\s+")


@final
class ModificationError(ValueError):
    """A wire operation does not fit the original file."""


def _delete_range(mod: CodeModification, total: int) -> tuple[int, int]:
    end = mod.end_line if mod.end_line is not None else mod.start_line
    if end < mod.start_line or end > total:
        msg = f"delete {mod.start_line}-{end} outside 1..{total}"
        raise ModificationError(msg)
    return mod.start_line, end


def _add_lines(mod: CodeModification, total: int) -> list[str]:
    if mod.content is None:
        msg = f"add at line {mod.start_line} has no content"
        raise ModificationError(msg)
    if mod.start_line > total + 1:
        msg = f"add at line {mod.start_line} beyond EOF+1 ({total + 1})"
        raise ModificationError(msg)
    return mod.content.split("\n")


def apply_modifications(
    original: Sequence[str], modifications: Iterable[CodeModification]
) -> list[str]:
    """Apply wire operations to ``original`` lines and return patched lines."""
    deletes: dict[int, int] = {}
    additions: dict[int, list[str]] = {}
    total = len(original)
    for mod in modifications:
        if mod.type is ModificationType.DELETE:
            start, end = _delete_range(mod, total)
            if start in deletes:
                msg = f"duplicate delete at line {start}"
                raise ModificationError(msg)
            deletes[start] = end
        else:
            additions.setdefault(mod.start_line, []).extend(_add_lines(mod, total))

    result: list[str] = []
    line = 1
    while line <= total + 1:
        result.extend(additions.get(line, []))
        end = deletes.get(line)
        if end is not None:
            line = end + 1
            continue
        if line <= total:
            result.append(original[line - 1])
        line += 1
    return result


def changed_original_ranges(
    modifications: Iterable[CodeModification],
) -> tuple[tuple[int, int], ...]:
    """Original-file line ranges touched by the operations (inclusive)."""
    ranges: list[tuple[int, int]] = []
    for mod in modifications:
        if mod.type is ModificationType.DELETE:
            end = mod.end_line if mod.end_line is not None else mod.start_line
            ranges.append((mod.start_line, end))
        else:
            ranges.append((mod.start_line, mod.start_line))
    return tuple(sorted(ranges))


def patch_hash(modifications: Iterable[CodeModification]) -> str:
    """Whitespace-insensitive content hash used as the cache identity."""
    normalized = [
        (
            mod.type.value,
            mod.start_line,
            mod.end_line,
            _WS.sub(" ", mod.content).strip() if mod.content is not None else None,
        )
        for mod in modifications
    ]
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def unified_diff(
    original: Sequence[str],
    patched: Sequence[str],
    *,
    context: int = DIFF_CONTEXT_LINES,
) -> str:
    lines = difflib.unified_diff(
        list(original),
        list(patched),
        fromfile="original",
        tofile="patched",
        n=context,
        lineterm="",
    )
    return "\n".join(lines)


def _numbered(original: Sequence[str], start: int, end: int) -> str:
    """Render original lines ``start..end`` (inclusive, 1-based) with numbers."""
    return "\n".join(f"{n}. {original[n - 1]}" for n in range(start, end + 1))


def _merge(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def header_range(original: Sequence[str]) -> tuple[int, int] | None:
    """``package``/``import`` block so a windowed view still shows dependencies."""
    limit = min(len(original), HEADER_MAX_LINES)
    last = 0
    for number in range(1, limit + 1):
        stripped = original[number - 1].lstrip()
        if stripped.startswith(("package ", "import ")):
            last = number
    return (1, last) if last else None


def render_code(
    original: Sequence[str],
    *,
    issue_line: int,
    modifications: Sequence[CodeModification],
) -> tuple[CodeMode, str]:
    """Full numbered file when short; otherwise windows around issue and edits."""
    total = len(original)
    if total <= FULL_FILE_MAX_LINES:
        return CodeMode.FULL, _numbered(original, 1, total)

    windows = [
        (
            max(1, issue_line - ISSUE_WINDOW_LINES),
            min(total, issue_line + ISSUE_WINDOW_LINES),
        )
    ]
    header = header_range(original)
    if header is not None:
        windows.append(header)
    for start, end in changed_original_ranges(modifications):
        windows.append(
            (
                max(1, start - MODIFICATION_WINDOW_LINES),
                min(total, end + MODIFICATION_WINDOW_LINES),
            )
        )
    parts: list[str] = []
    cursor = 1
    for start, end in _merge(windows):
        if start > cursor:
            parts.append(f"... (lines {cursor}-{start - 1} omitted) ...")
        parts.append(_numbered(original, start, end))
        cursor = end + 1
    if cursor <= total:
        parts.append(f"... (lines {cursor}-{total} omitted) ...")
    return CodeMode.WINDOW, "\n".join(parts)


def render_patch(
    entire_code: str,
    *,
    issue_line: int,
    modifications: Sequence[CodeModification],
) -> RenderedPatch:
    """Build every judge-facing artifact for one (issue, patch) pair."""
    original = entire_code.split("\n")
    patched = apply_modifications(original, modifications)
    code_mode, code_text = render_code(
        original, issue_line=issue_line, modifications=modifications
    )
    return RenderedPatch(
        code_mode=code_mode,
        code_text=code_text,
        diff_text=unified_diff(original, patched),
        patch_hash=patch_hash(modifications),
        original_line_count=len(original),
        patched_line_count=len(patched),
        window_truncated=code_mode is CodeMode.WINDOW,
    )
