"""Listwise judge prompt: rank 2-4 SAST-passing patches for one finding.

Imports the finding/dataflow/example blocks and the diff renderer from
``secllm_judge`` so the judge sees exactly what the Fix-Rate judge sees, plus
one labelled unified diff per candidate. The agent's description, reasoning,
timeline and run numbers are deliberately withheld.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

from secllm_judge.diff_render import (
    apply_modifications,
    patch_hash,
    render_code,
    unified_diff,
)
from secllm_judge.prompt import (
    Message,
    _dataflow_block,
    _examples_block,
    _issue_block,
)
from secllm_judge.schemas import CodeMode, CodeModification, IssueInput, ModificationType
from secllm_judge.syntax_gate import normalize_language

PROMPT_VERSION = "sft-select-listwise.v1"
LABELS = ("A", "B", "C", "D")
REASON_MAX = 200
NOTE_MAX = 160

SYSTEM_PROMPT = """\
You are comparing candidate fixes for ONE static-analysis (SAST) finding in {display} code. \
Every candidate has ALREADY been re-scanned by the same checker and is no longer reported. \
Your task is to rank them so that the best one can be used as a training example for a \
fix-suggestion agent.

Judge each candidate against the checker's own detection rule (its description, its \
dangerous/safe examples, and the dataflow trace), not against security best practice or \
your personal ideal fix. Evaluate ONLY the code you are given; you are not shown the \
authors' explanations on purpose. Line numbers refer to the ORIGINAL file. Do not \
speculate about callers or code in other files. Candidate labels are random and carry \
no information.

For EVERY candidate answer:
- not_redetected: would the checker's pattern stay absent at the reported sink and along \
the reported dataflow for ordinary inputs, i.e. is the fix robust rather than a \
coincidence of the scan (for example only a comment or logging was added, or the flagged \
code was moved but still reachable with tainted data)? Answer false only when the flagged \
pattern clearly remains or is clearly still reachable.
- compiles: would the patched file still compile/execute? The patched file has ALREADY \
passed a parser check, so never report brace/semicolon/syntax problems. Answer false only \
for a certain semantic break visible in this file: an undefined symbol, a missing import \
for a used type, a type mismatch, a variable used out of scope, or a call that no longer \
matches a visible definition.
- scope_ok: is the change limited to what the finding needs, without deleting or rewriting \
legitimate behaviour for benign inputs and without unrelated edits? The insecure \
behaviour itself is not functionality to preserve.
- note: one short English sentence naming the decisive fact for this candidate.

Then produce:
- ranking: ALL candidate labels, best first. Order by: (1) candidates with \
not_redetected and compiles both true come before any other; (2) among those, prefer \
scope_ok true; (3) then prefer the fix that most directly follows the checker's safe \
example and would most plausibly survive a stricter scan; (4) then prefer the smaller, \
cleaner change. If two candidates are equivalent in all of these, keep their given order.
- confidence_score: integer 0-100 for the FIRST place. Go below 70 when the choice \
rests on interpretation or the top candidates are near-equivalent.
- reason: one English sentence (max {reason_max} characters) saying why the first-ranked \
candidate wins.

Respond with a single JSON object matching the schema you are given and nothing else.
"""

_DISPLAY = {
    "java": "Java",
    "javascript": "JavaScript",
    "c": "C",
    "cpp": "C++",
    "go": "Go",
    "python": "Python",
    "typescript": "TypeScript",
    "tsx": "TypeScript",
}


def output_json_schema(labels: Sequence[str]) -> dict[str, Any]:
    """Flat JSON schema (no $ref) for vLLM structured output."""
    label_enum = {"type": "string", "enum": list(labels)}
    return {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": len(labels),
                "maxItems": len(labels),
                "items": {
                    "type": "object",
                    "properties": {
                        "label": label_enum,
                        "not_redetected": {"type": "boolean"},
                        "compiles": {"type": "boolean"},
                        "scope_ok": {"type": "boolean"},
                        "note": {"type": "string", "maxLength": NOTE_MAX},
                    },
                    "required": ["label", "not_redetected", "compiles", "scope_ok", "note"],
                    "additionalProperties": False,
                },
            },
            "ranking": {
                "type": "array",
                "minItems": len(labels),
                "maxItems": len(labels),
                "items": label_enum,
            },
            "confidence_score": {"type": "integer", "minimum": 0, "maximum": 100},
            "reason": {"type": "string", "minLength": 1, "maxLength": REASON_MAX},
        },
        "required": ["candidates", "ranking", "confidence_score", "reason"],
        "additionalProperties": False,
    }


def to_modifications(raw: Sequence[dict[str, Any]]) -> tuple[CodeModification, ...]:
    return tuple(
        CodeModification(
            type=ModificationType(str(m["type"])),
            start_line=int(m["startLine"]),
            end_line=(int(m["endLine"]) if m.get("endLine") is not None else None),
            content=m.get("content"),
        )
        for m in raw
    )


@dataclass(frozen=True)
class Candidate:
    patch_key: str
    modifications: tuple[CodeModification, ...]


@dataclass(frozen=True)
class RenderedPrompt:
    messages: tuple[Message, ...]
    label_to_patch: dict[str, str]
    schema: dict[str, Any]
    code_mode: CodeMode
    prompt_sha256: str


def build_messages(issue: IssueInput, candidates: Sequence[Candidate], order: Sequence[int]) -> RenderedPrompt:
    """``order[i]`` is the candidate index shown under ``LABELS[i]``."""
    if not 2 <= len(candidates) <= len(LABELS):
        msg = f"listwise judge takes 2..{len(LABELS)} candidates, got {len(candidates)}"
        raise ValueError(msg)
    if sorted(order) != list(range(len(candidates))):
        msg = f"order {order!r} is not a permutation of {len(candidates)} candidates"
        raise ValueError(msg)
    language = normalize_language(issue.language)
    labels = LABELS[: len(candidates)]
    original = issue.entire_code.split("\n")
    union_mods = tuple(m for c in candidates for m in c.modifications)
    code_mode, code_text = render_code(original, issue_line=issue.issue_line_number, modifications=union_mods)
    heading = (
        f"Original file ({len(original)} lines, complete):"
        if code_mode is CodeMode.FULL
        else (
            f"Original file ({len(original)} lines; only windows around the reported line "
            "and the edited regions are shown, other lines are omitted):"
        )
    )
    diffs: list[str] = []
    label_to_patch: dict[str, str] = {}
    for label, cand_index in zip(labels, order, strict=True):
        cand = candidates[cand_index]
        patched = apply_modifications(original, cand.modifications)
        diffs.append(f"### Candidate {label}\n```diff\n{unified_diff(original, patched)}\n```")
        label_to_patch[label] = cand.patch_key
    schema = output_json_schema(labels)
    system = SYSTEM_PROMPT.format(display=_DISPLAY.get(language, issue.language), reason_max=REASON_MAX)
    user = (
        "## Finding\n"
        f"{_issue_block(issue)}\n\n"
        "## Dataflow trace from the checker (original line numbers)\n"
        f"{_dataflow_block(issue)}\n\n"
        "## Checker examples\n"
        f"{_examples_block(issue)}\n\n"
        f"## {heading}\n"
        f"```{language}\n{code_text}\n```\n\n"
        f"## Candidate patches ({len(candidates)}; unified diff, original -> patched)\n"
        + "\n\n".join(diffs)
        + "\n\n## Output JSON schema\n"
        f"```json\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n```\n\n"
        "Assess every candidate, rank them all, and respond with the JSON object only."
    )
    messages = (Message(role="system", content=system), Message(role="user", content=user))
    sha = hashlib.sha256(
        json.dumps({"version": PROMPT_VERSION, "messages": list(messages)}, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    return RenderedPrompt(messages=messages, label_to_patch=label_to_patch, schema=schema, code_mode=code_mode, prompt_sha256=sha)


def candidate_from_raw(raw_modifications: Sequence[dict[str, Any]]) -> Candidate:
    mods = to_modifications(raw_modifications)
    return Candidate(patch_key=patch_hash(mods), modifications=mods)
