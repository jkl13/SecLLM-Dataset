"""Turn-boundary rendering and assistant-only label masking for Qwen3.5 SFT rows.

Two row shapes are accepted:

* Plain rows ``{messages, tools, ...}``: every assistant turn is supervised. System,
  user and tool-result tokens (tool results render as ``<tool_response>`` blocks
  inside user turns) and the empty think block opening each assistant turn get
  label ``IGNORE``.
* SRFT rows that also carry ``schema_version`` / ``policy_version`` /
  ``assistant_supervision`` / ``training_status``: each assistant message has a
  per-turn label (``good`` and ``unnecessary`` weight 1, ``harmful`` weight 0) bound
  to the message by its SHA-256; a mismatch is reported, never silently trained.

Nothing is truncated: rows longer than ``max_length`` come back as ``overlength``.
Only the tokenizer is needed (no judge, no model).
"""

from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

IGNORE: Final = -100
Status = Literal[
    "ready",
    "pending_judgment",
    "needs_review",
    "excluded_generation",
    "overlength",
    "zero_supervised_tokens",
    "invalid_annotations",
]
Index = Annotated[int, Field(strict=True, ge=0)]
Nonempty = Annotated[str, Field(min_length=1)]


class Example(TypedDict):
    input_ids: list[int]
    labels: list[int]
    length: int


@dataclass(frozen=True, slots=True)
class TokenizationResult:
    training_status: Status
    reason: str
    example: Example | None = None
    token_count: int = 0
    supervised_tokens: int = 0


class AnnotationError(ValueError):
    """An SRFT row cannot be safely interpreted as training supervision."""


class Evidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    location: Nonempty
    observation: Nonempty
    basis: Literal["recorded", "code_inference"]


class Supervision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    message_index: Index
    message_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    event_index: Index | None = None
    turn: Index | None = None
    call_ids: tuple[Nonempty, ...] = ()
    label: Literal["good", "unnecessary", "harmful", "needs_review"] | None
    weight: Annotated[int, Field(strict=True, ge=0, le=1)] | None
    origin: Literal["resolved_rule", "critic"]
    reason: Nonempty
    evidence: tuple[Evidence, ...] = Field(min_length=1)
    judgment_ref: Nonempty


class Row(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    messages: list[dict[str, JsonValue]]
    tools: list[dict[str, JsonValue]]


class SrftRow(Row):
    schema_version: Literal["secllm-srft.v1"]
    policy_version: Literal["security-srft.v1"]
    assistant_supervision: tuple[Supervision, ...]
    training_status: Literal[
        "ready",
        "pending_judgment",
        "pending_tokenization",
        "needs_review",
        "excluded_generation",
        "overlength",
        "zero_supervised_tokens",
    ]


def evaluate_example(
    row: dict[str, JsonValue],
    tok: PreTrainedTokenizerBase,
    max_length: int = 12288,
) -> TokenizationResult:
    """Validate and tokenize without mutating canonical content or truncating context.

    ``invalid_annotations`` is an adapter diagnostic: exporters must hold the row
    as ``needs_review`` with this reason, never publish it as ready.
    """
    srft = bool(
        {"schema_version", "policy_version", "assistant_supervision", "training_status"}
        & row.keys()
    )
    try:
        parsed = SrftRow.model_validate(row) if srft else Row.model_validate(row)
    except ValidationError as exc:
        return TokenizationResult(
            "invalid_annotations", f"row_schema:{exc.error_count()}"
        )
    weights: dict[int, int | None] = {}
    if srft:
        annotated = SrftRow.model_validate(row)
        expected = {
            i for i, m in enumerate(parsed.messages) if m.get("role") == "assistant"
        }
        for item in annotated.assistant_supervision:
            i = item.message_index
            if i not in expected or i in weights:
                return TokenizationResult(
                    "invalid_annotations", "assistant_index_coverage"
                )
            digest = hashlib.sha256(
                json.dumps(
                    parsed.messages[i],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            if digest != item.message_sha256:
                return TokenizationResult(
                    "invalid_annotations", "message_hash_mismatch"
                )
            valid = (
                (item.label is None and item.weight == 1)
                if item.origin == "resolved_rule"
                else (
                    item.label is not None
                    and item.weight
                    == {
                        "good": 1,
                        "unnecessary": 1,
                        "harmful": 0,
                        "needs_review": None,
                    }[item.label]
                )
            )
            if not valid:
                return TokenizationResult(
                    "invalid_annotations", "label_weight_mismatch"
                )
            weights[i] = item.weight
        if set(weights) != expected:
            return TokenizationResult("invalid_annotations", "assistant_index_coverage")
        if annotated.training_status in {
            "pending_judgment",
            "needs_review",
            "excluded_generation",
        }:
            return TokenizationResult(annotated.training_status, "canonical_hold")
        if any(weight is None for weight in weights.values()):
            return TokenizationResult("needs_review", "unknown_weight")
    messages, tools = parsed.messages, parsed.tools
    full_text = tok.apply_chat_template(messages, tools=tools, tokenize=False)
    enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > max_length:
        return TokenizationResult(
            "overlength", "max_length_exceeded", token_count=len(ids)
        )
    starts = [s for s, _ in offsets]
    ends = [e for _, e in offsets]
    labels = [IGNORE] * len(ids)
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = tok.apply_chat_template(
            messages[:i],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not full_text.startswith(prefix):
            prefix = tok.apply_chat_template(
                messages[:i],
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        upto = tok.apply_chat_template(messages[: i + 1], tools=tools, tokenize=False)
        if not (full_text.startswith(prefix) and full_text.startswith(upto)):
            return TokenizationResult(
                "invalid_annotations", f"template_prefix:{i}", token_count=len(ids)
            )
        lo = bisect.bisect_left(starts, len(prefix))
        hi = bisect.bisect_left(ends, len(upto))
        if (
            lo >= len(ids)
            or hi >= len(ids)
            or starts[lo] != len(prefix)
            or ends[hi] != len(upto)
        ):
            return TokenizationResult(
                "invalid_annotations", f"token_boundary:{i}", token_count=len(ids)
            )
        if not srft or weights[i] == 1:
            labels[lo : hi + 1] = ids[lo : hi + 1]
    supervised = sum(label != IGNORE for label in labels)
    if not supervised:
        return TokenizationResult(
            "zero_supervised_tokens", "no_targets", token_count=len(ids)
        )
    return TokenizationResult(
        "ready",
        "eligible",
        Example(input_ids=ids, labels=labels, length=len(ids)),
        len(ids),
        supervised,
    )


def build_example(
    row: dict[str, JsonValue],
    tok: PreTrainedTokenizerBase,
    max_length: int,
) -> Example | None:
    """``evaluate_example`` returning only the example; raises on malformed SRFT annotations."""
    result = evaluate_example(row, tok, max_length)
    if result.training_status == "invalid_annotations":
        raise AnnotationError(result.reason)
    return result.example
