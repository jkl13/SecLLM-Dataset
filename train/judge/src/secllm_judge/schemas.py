"""Typed contracts for judge inputs, LLM output, and stored records."""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Verdict(StrEnum):
    PASS = "pass"  # noqa: S105 - verdict label, not a credential.
    FAIL = "fail"


class FixKind(StrEnum):
    """Coarse category of the fix. Logged for bias analysis, never for reward."""

    SANITIZE = "sanitize"
    VALIDATE = "validate"
    API_REPLACE = "api_replace"
    REMOVE_CODE = "remove_code"
    CONFIG = "config"
    OTHER = "other"


class CodeMode(StrEnum):
    """How much of the original file the judge saw."""

    FULL = "full"
    WINDOW = "window"


class VerdictSource(StrEnum):
    SYNTAX_GATE = "syntax_gate"
    LLM = "llm"


class ModificationType(StrEnum):
    ADD = "add"
    DELETE = "delete"


class ContextEventType(StrEnum):
    SOURCE = "source"
    BRANCH = "branch"
    SINK = "sink"


class _Frozen(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Inputs (subset of userRequest / codeModifications selected for the judge)
# ---------------------------------------------------------------------------


class ContextEvent(_Frozen):
    """One SAST dataflow event (source -> branch -> sink)."""

    line_no: int = Field(ge=1)
    event_type: ContextEventType
    message: str


class IssueInput(_Frozen):
    """Metadata the judge is allowed to see about the finding."""

    idx: int = Field(ge=0)
    checker_id: int
    language: str
    issue_name_ko: str
    issue_name_en: str
    issue_description_ko: str
    issue_line_number: int = Field(ge=1)
    contexts: tuple[ContextEvent, ...]
    dangerous_example_ko: str
    safe_example_ko: str
    entire_code: str


class CodeModification(_Frozen):
    """Public add/delete wire operation in original-file coordinates.

    ``add`` inserts ``content`` before original line ``start_line``; ``delete``
    removes original lines ``start_line..end_line`` inclusive.
    """

    type: ModificationType
    start_line: int = Field(ge=1)
    end_line: int | None = None
    content: str | None = None


# ---------------------------------------------------------------------------
# Rendered patch (renderer output)
# ---------------------------------------------------------------------------


class RenderedPatch(_Frozen):
    code_mode: CodeMode
    code_text: str
    diff_text: str
    patch_hash: str = Field(min_length=16, max_length=16)
    original_line_count: int = Field(ge=0)
    patched_line_count: int = Field(ge=0)
    window_truncated: bool


# ---------------------------------------------------------------------------
# LLM output (enforced through vLLM response_format json_schema)
# ---------------------------------------------------------------------------

REASON_MAX_CHARS = 200


class JudgeOutput(_Frozen):
    """Exact JSON the judge model must emit. Kept minimal and discrete.

    The verdict targets SAST fix rate: ``pass`` iff the same checker would no
    longer report the finding on the patched file AND the patch still compiles.
    ``functionality_preserved`` is logged for analysis and never affects the verdict.
    """

    not_redetected: bool
    compiles: bool
    functionality_preserved: bool
    fix_kind: FixKind
    verdict: Verdict
    confidence_score: int = Field(ge=0, le=100)
    reason: str = Field(min_length=1, max_length=REASON_MAX_CHARS)

    def derived_verdict(self) -> Verdict:
        """Verdict recomputed from the gating sub-checks."""
        return Verdict.PASS if self.not_redetected and self.compiles else Verdict.FAIL

    def is_consistent(self) -> bool:
        return self.verdict is self.derived_verdict()


def judge_output_json_schema() -> dict[str, object]:
    """JSON schema for the vLLM ``response_format`` structured-output request."""
    return JudgeOutput.model_json_schema()


# ---------------------------------------------------------------------------
# Stored records
# ---------------------------------------------------------------------------

JudgeFlag = Literal[
    "judge_inconsistent",
    "schema_retry",
    "window_truncated",
    "syntax_claim_overridden",
]


class SyntaxGateResult(_Frozen):
    passed: bool
    error_nodes_before: int = Field(ge=0)
    error_nodes_after: int = Field(ge=0)


class JudgeProvenance(_Frozen):
    judge_model: str
    temperature: float = Field(ge=0.0)
    prompt_version: str
    prompt_sha256: str = Field(min_length=64, max_length=64)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    attempts: int = Field(ge=1)
    created_at: str


class JudgeRecord(_Frozen):
    """One judged patch (one per cache key)."""

    schema_version: Literal["judge-record.v1"] = "judge-record.v1"
    cache_key: str = Field(min_length=24, max_length=24)
    idx: int = Field(ge=0)
    checker_id: int
    patch_hash: str = Field(min_length=16, max_length=16)
    code_mode: CodeMode
    syntax_gate: SyntaxGateResult
    judge: JudgeOutput | None
    verdict: Verdict
    verdict_source: VerdictSource
    flags: tuple[JudgeFlag, ...] = ()
    provenance: JudgeProvenance | None


class TrajectoryMapping(_Frozen):
    """One generated trajectory (run x idx) joined to its judge verdict."""

    run: int = Field(ge=1)
    idx: int = Field(ge=0)
    terminal_cause: str
    cache_key: str | None
    verdict: Verdict | None
    confidence_score: int | None = Field(default=None, ge=0, le=100)
    reward_v2: float | None
    reward_3tier: float | None
    reward_hybrid: float | None
