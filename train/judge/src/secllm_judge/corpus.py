"""Load judge inputs from request records and generation files.

``issue_from_record`` builds an ``IssueInput`` from one request record whose
``userRequest`` is a dict or a Python-literal string. The file-level loaders assume
that a generation row's ``idx`` equals the 0-based line index of the request file.
"""

from __future__ import annotations

import ast
import json
from typing import TYPE_CHECKING, ClassVar, final

from pydantic import BaseModel, ConfigDict, Field

from secllm_judge.schemas import (
    CodeModification,
    ContextEvent,
    ContextEventType,
    IssueInput,
    ModificationType,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@final
class GenerationRow(BaseModel):
    """The subset of a generation row the judge pipeline consumes."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="ignore")

    idx: int = Field(ge=0)
    passed: bool
    terminal_cause: str
    modifications: tuple[CodeModification, ...]

    @property
    def is_pass(self) -> bool:
        return self.terminal_cause == "pass" and self.passed


def parse_user_request(raw: object) -> dict[str, object]:
    """``userRequest`` is stored as a Python-literal string; JSON is a fallback."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        msg = f"userRequest has unexpected type {type(raw).__name__}"
        raise TypeError(msg)
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        msg = "userRequest did not parse to a mapping"
        raise TypeError(msg)
    return parsed


def _contexts(raw: object) -> tuple[ContextEvent, ...]:
    if not isinstance(raw, list):
        return ()
    events: list[ContextEvent] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        events.append(
            ContextEvent(
                line_no=int(item["lineNo"]),
                event_type=ContextEventType(str(item["eventType"])),
                message=str(item.get("message", "")),
            )
        )
    return tuple(events)


def issue_from_record(idx: int, record: dict[str, object]) -> IssueInput:
    request = parse_user_request(record["userRequest"])
    return IssueInput(
        idx=idx,
        checker_id=int(str(request["checkerId"])),
        language=str(request.get("programmingLanguage", "Java")),
        issue_name_ko=str(request["issueNameKo"]),
        issue_name_en=str(request["issueNameEn"]),
        issue_description_ko=str(request["issueDescriptionKo"]),
        issue_line_number=int(str(request["issueLineNumber"])),
        contexts=_contexts(request.get("contexts")),
        dangerous_example_ko=str(request.get("dangerousExampleKo", "")),
        safe_example_ko=str(request.get("safeExampleKo", "")),
        entire_code=str(request["entireCode"]),
    )


def iter_issues(training_set: Path) -> Iterator[IssueInput]:
    with training_set.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if line.strip():
                yield issue_from_record(idx, json.loads(line))


def load_issue(training_set: Path, idx: int) -> IssueInput:
    with training_set.open(encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == idx:
                return issue_from_record(idx, json.loads(line))
    msg = f"idx {idx} not found in {training_set}"
    raise IndexError(msg)


def modifications_from_record(raw: object) -> tuple[CodeModification, ...]:
    if not isinstance(raw, list):
        return ()
    mods: list[CodeModification] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        end = item.get("endLine")
        mods.append(
            CodeModification(
                type=ModificationType(str(item["type"])),
                start_line=int(str(item["startLine"])),
                end_line=int(str(end)) if end is not None else None,
                content=item.get("content"),
            )
        )
    return tuple(mods)


def generation_from_record(record: dict[str, object]) -> GenerationRow:
    response = record.get("response")
    raw_mods = response.get("codeModifications") if isinstance(response, dict) else None
    return GenerationRow(
        idx=int(str(record["idx"])),
        passed=bool(record["passed"]),
        terminal_cause=str(record["terminal_cause"]),
        modifications=modifications_from_record(raw_mods),
    )


def iter_generations(generation_file: Path) -> Iterator[GenerationRow]:
    with generation_file.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield generation_from_record(json.loads(line))


def load_generation(generation_file: Path, idx: int) -> GenerationRow:
    for row in iter_generations(generation_file):
        if row.idx == idx:
            return row
    msg = f"idx {idx} not found in {generation_file}"
    raise IndexError(msg)
