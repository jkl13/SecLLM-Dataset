"""English judge prompt built only from the selected metadata.

Deliberately excluded: the agent's own fix description, its reasoning text,
tool timelines, behavioural trace metrics, and verifier attempt history. The
judge evaluates the resulting patch, nothing about how it was produced.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, TypedDict

from secllm_judge.schemas import (
    CodeMode,
    FixKind,
    RenderedPatch,
    judge_output_json_schema,
)
from secllm_judge.syntax_gate import normalize_language

# Shown verbatim in the user message: with thinking enabled the request uses
# ``response_format: json_object`` (no server-side schema), so the model must see
# the exact schema here.
_SCHEMA_TEXT = json.dumps(judge_output_json_schema(), ensure_ascii=False, indent=2)

if TYPE_CHECKING:
    from secllm_judge.schemas import IssueInput

PROMPT_VERSION = "judge-prompt"

_LANGUAGE_SEMANTICS = {
    "c": ("C", "Respect preprocessing, header declarations, pointers, and C types."),
    "cpp": (
        "C++",
        "Respect preprocessing, headers, namespaces, templates, overloads, "
        "and C++ types.",
    ),
    "go": ("Go", "Respect Go packages, imports, static types, scope, and interfaces."),
    "python": (
        "Python",
        "Respect dynamic typing, Python builtins, imports, "
        "and runtime name resolution.",
    ),
    "typescript": (
        "TypeScript",
        "Respect TypeScript types, ambient declarations, module resolution, "
        "and browser/Node globals. Do not assume unseen tsconfig or dependency types.",
    ),
    "tsx": (
        "TypeScript",
        "Respect TypeScript types, ambient declarations, module resolution, JSX, "
        "and browser/Node globals. Do not assume unseen tsconfig or dependency types.",
    ),
}


class Message(TypedDict):
    role: str
    content: str


SYSTEM_PROMPT = """\
You are reviewing candidate fixes for static-analysis (SAST) findings in Java code. \
You will be shown one finding reported by a specific checker and one candidate patch.

The goal of a fix is narrow and concrete: after the patch, the SAME checker, run again \
on the patched file, must no longer report this finding. Judge against the checker's \
own detection rule, as described by its description, its dangerous/safe examples, \
and the dataflow trace. Do NOT judge against security best practice, completeness, \
elegance, or what you personally consider the ideal fix. A patch that is imperfect \
from a security standpoint but removes what the checker looks for is a fix.

Evaluate ONLY the code you are given. You are not shown the patch author's \
explanation on purpose. Line numbers refer to the ORIGINAL file, before the patch. \
Do not speculate about callers or code in other files.

Answer these questions:

1. not_redetected - Would the checker's pattern be absent from the patched file at the \
reported sink and along the reported dataflow (if any)? Any change that achieves this \
counts: removing or replacing the flagged call, adding the validation, sanitisation, \
permission, modifier, or wrapper the checker's safe example uses, breaking the taint \
flow between source and sink, or catching/handling what the checker demands. When the \
checker's rule does not clearly say whether the patched form is still detected, \
decide by the closest safe/dangerous example; if it is still ambiguous, answer true. \
Answer false only when the flagged pattern clearly remains (for example the same \
call is still made with the same tainted data, or only a comment or log line was \
added).

2. compiles - Would the patched file still compile? The patched file has ALREADY \
passed a parser check: braces, parentheses, and semicolons are balanced and the file \
structure is valid. Never report a syntax or brace/structure problem; the diff is a \
fragment and counting braces in it is unreliable. Answer false only for a certain \
semantic break that a parser cannot see: a class, method, or variable that is used \
but defined nowhere in the file or its imports, a type used without an import, a \
type mismatch, a variable used outside its scope, or a call that no longer matches \
a definition visible in this file. A changed signature by itself is not a break.

3. functionality_preserved - Does the patched code still perform the program's \
legitimate behaviour for benign inputs? The insecure behaviour itself is not \
functionality to preserve. This answer is recorded for analysis only and does NOT \
affect the verdict.

Then set:
- fix_kind: one of {fix_kinds}. This is a descriptive label only.
- verdict: "pass" if not_redetected and compiles are both true, otherwise "fail".
- confidence_score: an integer 0-100 expressing how sure you are of the verdict. \
Use the whole range: reserve 95-100 for verdicts backed by a certain fact (a \
compile error, an unchanged sink), and go below 70 when the decision rests on \
interpretation.
- reason: one English sentence (max {reason_max} characters) naming the decisive fact.

Respond with a single JSON object matching the schema you are given and nothing else.
"""


def _issue_block(issue: IssueInput) -> str:
    return (
        f"Checker: {issue.issue_name_en} ({issue.issue_name_ko})\n"
        f"Language: {issue.language}\n"
        f"Reported line: {issue.issue_line_number}\n\n"
        f"Checker description (Korean):\n{issue.issue_description_ko}"
    )


def _dataflow_block(issue: IssueInput) -> str:
    if not issue.contexts:
        return "(no dataflow trace; the finding is a single-line pattern match)"
    return "\n".join(
        f"L{event.line_no} [{event.event_type.value}] {event.message}"
        for event in issue.contexts
    )


def _examples_block(issue: IssueInput) -> str:
    parts: list[str] = []
    if issue.dangerous_example_ko.strip():
        parts.append(f"Dangerous example:\n{issue.dangerous_example_ko}")
    if issue.safe_example_ko.strip():
        parts.append(f"Safe example:\n{issue.safe_example_ko}")
    return "\n\n".join(parts) if parts else "(no examples provided by the checker)"


def _code_heading(rendered: RenderedPatch) -> str:
    if rendered.code_mode is CodeMode.FULL:
        return f"Original file ({rendered.original_line_count} lines, complete):"
    return (
        f"Original file ({rendered.original_line_count} lines; only windows around the "
        "reported line and the edited regions are shown, other lines are omitted):"
    )


def build_messages(issue: IssueInput, rendered: RenderedPatch) -> tuple[Message, ...]:
    language = normalize_language(issue.language)
    system = SYSTEM_PROMPT.format(
        fix_kinds=", ".join(f'"{kind.value}"' for kind in FixKind),
        reason_max=200,
    )
    if language == "javascript":
        system = system.replace("in Java code", "in JavaScript code")
        start = system.index("2. compiles -")
        end = system.index("3. functionality_preserved", start)
        system = (
            system[:start]
            + "2. compiles - Is the patch free of a certain JavaScript execution "
            "or module-loading break introduced by the edit? The parser gate checks "
            "that the edit adds no syntax errors; existing parser errors may remain. "
            "Respect dynamic typing, automatic semicolon insertion, browser/Node "
            "globals, CommonJS/ES modules, and JSX. Do not require Java types, "
            "classes or imports. Answer false only for a definite new break supported "
            "by the visible code, not assumptions about unavailable dependencies "
            "or runtime context.\n\n" + system[end:]
        )
    if language in _LANGUAGE_SEMANTICS:
        display_name, semantics = _LANGUAGE_SEMANTICS[language]
        system = system.replace("in Java code", f"in {display_name} code")
        start = system.index("2. compiles -")
        end = system.index("3. functionality_preserved", start)
        system = (
            system[:start]
            + f"2. compiles - Is the patch free of a certain {display_name} compile, "
            "execution, or module-loading break introduced by the edit? The parser "
            "gate checks that the edit adds no syntax errors; existing parser errors "
            f"may remain. {semantics} Answer false only for a definite new break "
            "supported by the visible code, not assumptions about unavailable "
            "dependencies or runtime context. Do not report syntax errors from "
            "counting delimiters in a partial diff.\n\n" + system[end:]
        )
    user = (
        "## Finding\n"
        f"{_issue_block(issue)}\n\n"
        "## Dataflow trace from the checker (original line numbers)\n"
        f"{_dataflow_block(issue)}\n\n"
        "## Checker examples\n"
        f"{_examples_block(issue)}\n\n"
        f"## {_code_heading(rendered)}\n"
        f"```{language}\n{rendered.code_text}\n```\n\n"
        "## Candidate patch (unified diff, original -> patched)\n"
        f"```diff\n{rendered.diff_text}\n```\n\n"
        "## Output JSON schema\n"
        f"```json\n{_SCHEMA_TEXT}\n```\n\n"
        "Evaluate the patch and respond with the JSON object only."
    )
    return (
        Message(role="system", content=system),
        Message(role="user", content=user),
    )


def prompt_sha256(messages: tuple[Message, ...]) -> str:
    payload = json.dumps(
        {"version": PROMPT_VERSION, "messages": list(messages)},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_key(idx: int, patch_hash: str, code_mode: CodeMode) -> str:
    """Identity of one judged patch (prompt edits are tracked via prompt_sha256)."""
    payload = f"{idx}|{patch_hash}|{code_mode.value}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
