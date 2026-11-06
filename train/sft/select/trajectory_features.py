"""Per-trajectory features read from teacher generation rows (stdlib only).

Everything here is derived from what the harness recorded (``trace``, ``loop``,
``messages``, ``response.codeModifications``) plus the SAST re-verification fields
(``sast_result`` / ``sast_verdict`` / ``sast_grader``). No LLM is involved.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

# generations/<model>_run01.jsonl, <model>_run02.jsonl, ... (one file per run)
GENERATION_GLOB = "*_run[0-9][0-9].jsonl"
VALID_TOOLS = frozenset({"read_code", "edit_code", "submit_fix"})

# Pagination rule from the agent system prompt.
WHOLE_FILE_MAX_LINES = 1500
AROUND_ISSUE_BEFORE = 100
AROUND_ISSUE_LIMIT = 200
AROUND_ISSUE_TOLERANCE = 5

_WS = re.compile(r"\s+")
_ISSUE_LINE = re.compile(r'"issueLineNumber":\s*(\d+)')
_TOTAL_LINES = re.compile(r'"totalLines":\s*(\d+)')
_CHUNK_RANGE = re.compile(r"^lines (\d+)-(\d+)$", re.M)
_TRACE_LINE = re.compile(r'"lineNo":\s*(\d+)')
_ERROR_RESULT = re.compile(r"^(ERROR[:\s]|Error[:\s]|Error invoking tool|FAIL \(attempt)")


def patch_key(modifications: list[dict[str, Any]]) -> str:
    """Whitespace-insensitive identity of a ``codeModifications`` list.

    Mirrors ``secllm_judge.diff_render.patch_hash`` so the judge stage and the
    deterministic stage agree on which patches are "the same".
    """
    normalized = [
        (
            str(mod.get("type")),
            int(mod["startLine"]),
            mod.get("endLine"),
            _WS.sub(" ", mod["content"]).strip() if mod.get("content") is not None else None,
        )
        for mod in modifications
    ]
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class TrajectoryFeatures:
    dataset_id: str
    idx: int
    run: int
    language: str
    sast_result: str | None
    sast_verdict: str | None
    terminal_cause: str
    passed: bool
    conversation_valid: bool
    patch_key: str
    hunks: int
    added_lines: int
    deleted_lines: int
    touches_trace_line: bool
    turns: int
    input_tokens: int
    output_tokens: int
    edit_errors: int
    fuzzy_matches: int
    nudges: int
    attempts: int
    first_attempt_passed: bool
    error_tool_results: int
    invalid_tool_calls: int
    unknown_tools: int
    read_calls: int
    edit_calls: int
    total_lines: int
    issue_line: int
    chunk_covers_file: bool
    read_rule_ok: bool
    artifact_matches_last_candidate: bool
    final_call_is_submit: bool
    desc_len: int
    hard_reasons: tuple[str, ...] = field(default=())

    @property
    def sast_pass(self) -> bool:
        return self.sast_result == "pass"

    @property
    def hard_ok(self) -> bool:
        return not self.hard_reasons

    @property
    def patch_lines(self) -> int:
        return self.added_lines + self.deleted_lines

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hard_reasons"] = list(self.hard_reasons)
        data["hard_ok"] = self.hard_ok
        data["patch_lines"] = self.patch_lines
        return data


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_rule_ok(
    reads: list[dict[str, Any]],
    *,
    total_lines: int,
    issue_line: int,
    chunk_covers_file: bool,
) -> bool:
    if not reads:
        return chunk_covers_file
    first = reads[0]
    offset = _int(first.get("offset"), 0)
    limit_raw = first.get("limit")
    limit = _int(limit_raw, 100) if limit_raw not in (None, "") else 100
    if total_lines <= WHOLE_FILE_MAX_LINES:
        return offset == 0 and limit >= total_lines
    expected = max(0, issue_line - AROUND_ISSUE_BEFORE)
    return abs(offset - expected) <= AROUND_ISSUE_TOLERANCE and limit >= AROUND_ISSUE_LIMIT


def _hard_reasons(
    *,
    terminal_cause: str,
    passed: bool,
    conversation_valid: bool,
    edit_errors: int,
    fuzzy: int,
    nudges: int,
    attempts: int,
    first_attempt_passed: bool,
    error_results: int,
    invalid_calls: int,
    unknown_tools: int,
    added_lines: int,
    hunks: int,
    artifact_ok: bool,
    final_is_submit: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if terminal_cause != "pass" or not passed:
        reasons.append("terminal_not_pass")
    if not conversation_valid:
        reasons.append("conversation_invalid")
    if edit_errors:
        reasons.append("edit_errors")
    if fuzzy:
        reasons.append("fuzzy_match")
    if nudges:
        reasons.append("nudged")
    if attempts != 1 or not first_attempt_passed:
        reasons.append("verifier_retry")
    if error_results:
        reasons.append("tool_error_result")
    if invalid_calls:
        reasons.append("invalid_tool_call")
    if unknown_tools:
        reasons.append("unknown_tool")
    if hunks == 0:
        reasons.append("empty_patch")
    elif added_lines == 0:
        reasons.append("deletion_only")
    if not artifact_ok:
        reasons.append("artifact_mismatch")
    if not final_is_submit:
        reasons.append("final_call_not_submit")
    return tuple(reasons)


def extract_features(row: dict[str, Any], dataset_id: str) -> TrajectoryFeatures:
    trace = row.get("trace") or {}
    response = row.get("response") or {}
    mods: list[dict[str, Any]] = list(response.get("codeModifications") or [])
    messages = row.get("messages") or []
    # 20 connection_error rows carry no conversation; they never pass SAST.
    user_text = str(messages[1]["content"]) if len(messages) > 1 else ""
    issue_line = _int((_ISSUE_LINE.search(user_text) or [None, 0])[1], 0)
    total_lines = _int((_TOTAL_LINES.search(user_text) or [None, 0])[1], 0)
    chunks = [(int(a), int(b)) for a, b in _CHUNK_RANGE.findall(user_text)]
    chunk_covers_file = any(a <= 1 and b >= total_lines for a, b in chunks)
    trace_lines = {int(x) for x in _TRACE_LINE.findall(user_text)} | ({issue_line} if issue_line else set())

    added = sum((m.get("content") or "").count("\n") + 1 for m in mods if m.get("type") == "add")
    deleted = sum(
        _int(m.get("endLine"), _int(m.get("startLine"))) - _int(m.get("startLine")) + 1
        for m in mods
        if m.get("type") == "delete"
    )
    touches = any(
        _int(m.get("startLine")) <= ln <= _int(m.get("endLine"), _int(m.get("startLine")))
        for m in mods
        for ln in trace_lines
    )

    loop = row.get("loop") or []
    model_turns = [e for e in loop if e.get("ev") == "model_turn"]
    calls = [tc for e in model_turns for tc in (e.get("tool_calls") or [])]
    reads = [tc.get("args") or {} for tc in calls if tc.get("tool") == "read_code"]
    edits = sum(1 for tc in calls if tc.get("tool") == "edit_code")
    unknown = sum(1 for tc in calls if tc.get("tool") not in VALID_TOOLS)
    invalid = sum(len(e.get("invalid_tool_calls") or []) for e in model_turns)
    error_results = sum(
        1
        for e in loop
        if e.get("ev") == "tool_result" and isinstance(e.get("result"), str) and _ERROR_RESULT.match(e["result"])
    ) + sum(1 for e in loop if e.get("ev") == "tool_error")

    attempts = list(trace.get("attempts") or [])
    first_passed = bool(attempts[0].get("passed")) if attempts else False
    candidates = list(trace.get("candidates") or [])
    artifact_ok = bool(candidates) and patch_key(list(candidates[-1].get("codeModifications") or [])) == patch_key(mods)
    last_assistant = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
    final_is_submit = bool(
        last_assistant
        and last_assistant.get("tool_calls")
        and last_assistant["tool_calls"][-1]["function"]["name"] == "submit_fix"
    )

    edit_errors = _int(trace.get("edit_errors"))
    fuzzy = _int(trace.get("fuzzy_matches"))
    nudges = _int(trace.get("nudges"))
    reasons = _hard_reasons(
        terminal_cause=str(row.get("terminal_cause")),
        passed=bool(row.get("passed")),
        conversation_valid=bool(row.get("conversation_valid", True)),
        edit_errors=edit_errors,
        fuzzy=fuzzy,
        nudges=nudges,
        attempts=len(attempts),
        first_attempt_passed=first_passed,
        error_results=error_results,
        invalid_calls=invalid,
        unknown_tools=unknown,
        added_lines=added,
        hunks=len(mods),
        artifact_ok=artifact_ok,
        final_is_submit=final_is_submit,
    )
    return TrajectoryFeatures(
        dataset_id=dataset_id,
        idx=int(row["idx"]),
        run=int(row["run"]),
        language=str(row.get("language") or ""),
        sast_result=row.get("sast_result"),
        sast_verdict=row.get("sast_verdict"),
        terminal_cause=str(row.get("terminal_cause")),
        passed=bool(row.get("passed")),
        conversation_valid=bool(row.get("conversation_valid", True)),
        patch_key=patch_key(mods),
        hunks=len(mods),
        added_lines=added,
        deleted_lines=deleted,
        touches_trace_line=touches,
        turns=_int(trace.get("model_turns"), len(model_turns)),
        input_tokens=_int(trace.get("input_tokens")),
        output_tokens=_int(trace.get("output_tokens")),
        edit_errors=edit_errors,
        fuzzy_matches=fuzzy,
        nudges=nudges,
        attempts=len(attempts),
        first_attempt_passed=first_passed,
        error_tool_results=error_results,
        invalid_tool_calls=invalid,
        unknown_tools=unknown,
        read_calls=len(reads),
        edit_calls=edits,
        total_lines=total_lines,
        issue_line=issue_line,
        chunk_covers_file=chunk_covers_file,
        read_rule_ok=_read_rule_ok(
            reads, total_lines=total_lines, issue_line=issue_line, chunk_covers_file=chunk_covers_file
        ),
        artifact_matches_last_candidate=artifact_ok,
        final_call_is_submit=final_is_submit,
        desc_len=len(str(response.get("desc") or "")),
        hard_reasons=reasons,
    )


def training_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Conversation as trained: drop the trailing ``submit_fix`` tool result.

    Every SAST-passing row ends with the verifier's ``PASS (attempt 1/3)`` tool
    message. At inference the harness stops right after ``submit_fix`` succeeds,
    so that message is never seen by the policy and carries no loss.
    """
    messages = list(row.get("messages") or [])
    while messages and messages[-1].get("role") == "tool":
        messages.pop()
    return messages


def generation_files(dataset_dir: Path) -> list[Path]:
    files = sorted((dataset_dir / "generations").glob(GENERATION_GLOB))
    if not files:
        msg = f"no generation files under {dataset_dir / 'generations'}"
        raise FileNotFoundError(msg)
    return files


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                msg = f"{path}:{line_no}: invalid JSON ({error.msg})"
                raise ValueError(msg) from error


def load_row(path: Path, *, idx: int, run: int) -> dict[str, Any]:
    for row in iter_rows(path):
        if int(row["idx"]) == idx and int(row["run"]) == run:
            return row
    msg = f"idx {idx} run {run} not found in {path}"
    raise KeyError(msg)


def deterministic_order(seed: int, dataset_id: str, idx: int, run: int) -> str:
    """Last-resort tie-break: a seeded hash, so run order never biases the choice."""
    return hashlib.sha256(f"{seed}|{dataset_id}|{idx}|{run}".encode()).hexdigest()
