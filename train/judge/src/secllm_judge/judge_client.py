"""OpenAI-compatible judge client with structured output and a JSONL cache.

Settings come from the repository-root ``.env`` (``JUDGE_API_URL``,
``JUDGE_API_KEY``, ``JUDGE_MODEL``); exported environment variables win.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, final

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from secllm_judge.prompt import PROMPT_VERSION, build_messages, prompt_sha256
from secllm_judge.schemas import (
    JudgeFlag,
    JudgeOutput,
    JudgeProvenance,
    JudgeRecord,
    Verdict,
    VerdictSource,
    judge_output_json_schema,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from secllm_judge.prompt import Message
    from secllm_judge.schemas import IssueInput, RenderedPatch, SyntaxGateResult

_ENV_KEYS = ("JUDGE_API_URL", "JUDGE_API_KEY", "JUDGE_MODEL")

# Structural-syntax claims the tree-sitter gate can refute with certainty. The judge
# model miscounts braces in diff fragments; the harness already guarantees that a
# passed trajectory introduced no parse errors (verify_fix V401).
_STRUCTURAL_CLAIM = re.compile(
    r"brace|semicolon|parenthes|unclosed|unbalanced|dangling|stray|"
    r"outside the class|nest(ed|ing) (a |the )?method|not properly closed",
    re.IGNORECASE,
)
MAX_ATTEMPTS = 2
# Non-thinking budget (the judge answers with ~80-100 tokens). Thinking mode needs
# ``max_tokens`` of 16384: reasoning tokens count against it, a trivial patch used
# ~3,800 and a real Go patch exhausted 8,192 in measured probes.
MAX_COMPLETION_TOKENS = 400
THINKING_MAX_COMPLETION_TOKENS = 16384
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


@final
class JudgeSampling(BaseModel):
    """Sampling parameters sent with every judge request.

    Defaults are the judge model's recommended non-thinking sampling. Measured on
    one judge server, with thinking on (temperature 1.0, top_p 0.95, presence_penalty 0.0,
    ``reasoning_effort`` xhigh, max_tokens 16384) the same judge server scored
    ~290 patches/hour at 3,600 completion tokens each and failed to parse 2-3% of
    answers, against ~3,800/hour, ~100 tokens and no failures without thinking;
    ``reasoning_effort`` (low vs xhigh) made no measurable difference. Thinking
    mode stays available by passing those values explicitly. ``top_k``, ``min_p``
    and ``repetition_penalty`` are vLLM extensions accepted at the top level of the
    chat-completions body; ``reasoning_effort`` is forwarded by vLLM to the chat
    template as a variable, so it only has an effect when the served model's
    template reads it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    temperature: float = Field(default=0.7, ge=0.0, allow_inf_nan=False)
    top_p: float = Field(default=0.80, gt=0.0, le=1.0, allow_inf_nan=False)
    top_k: int = Field(default=20, ge=-1)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    presence_penalty: float = Field(default=1.5, ge=-2.0, le=2.0, allow_inf_nan=False)
    repetition_penalty: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    enable_thinking: bool = False
    reasoning_effort: ReasoningEffort | None = None
    max_tokens: int = Field(default=MAX_COMPLETION_TOKENS, ge=1)

    def request_fields(self) -> dict[str, float | int]:
        """Body fields exactly as sent to the server."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
        }

    def generation_fields(self) -> dict[str, object]:
        """Thinking/budget body fields, also part of the online cache identity."""
        fields: dict[str, object] = {
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if self.reasoning_effort is not None:
            fields["reasoning_effort"] = self.reasoning_effort
        return fields

    def response_format(self, schema: dict[str, object]) -> dict[str, object] | None:
        r"""Structured output request, or ``None`` to send no ``response_format``.

        With thinking enabled, vLLM structured output (``json_schema`` and
        ``json_object`` alike) duplicated the opening tokens after the reasoning
        block (``{{"not_redetected"``, ``{\n{\n  "not_redetected"``, in 5 of 12
        probes and 3 of 14 smoke patches), while unconstrained
        decoding stayed well-formed. The schema is carried in the prompt and
        ``parse_judge_content`` tolerates the duplicated prefix anyway.
        """
        if self.enable_thinking:
            return None
        return {
            "type": "json_schema",
            "json_schema": {"name": "judge_output", "schema": schema},
        }


JUDGE_SAMPLING = JudgeSampling()


def find_dotenv(start: Path) -> Path | None:
    for parent in (start, *start.parents):
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


@final
@dataclass(frozen=True, slots=True)
class JudgeSettings:
    api_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls, start: Path | None = None) -> JudgeSettings:
        merged: dict[str, str] = {}
        dotenv = find_dotenv(start or Path.cwd())
        if dotenv is not None:
            merged.update(load_dotenv(dotenv))
        merged.update({k: os.environ[k] for k in _ENV_KEYS if k in os.environ})
        missing = [k for k in _ENV_KEYS if not merged.get(k)]
        if missing:
            msg = f"missing judge settings: {', '.join(missing)}"
            raise RuntimeError(msg)
        return cls(
            api_url=merged["JUDGE_API_URL"].rstrip("/"),
            api_key=merged["JUDGE_API_KEY"],
            model=merged["JUDGE_MODEL"],
        )


@final
class JudgeParseError(ValueError):
    """The judge response was not a valid ``JudgeOutput``."""


@final
class JudgeClient:
    """Thin synchronous client; one call = one ``(JudgeOutput, usage)``."""

    __slots__: ClassVar[tuple[str, ...]] = ("_http", "_schema", "sampling", "settings")

    def __init__(
        self,
        settings: JudgeSettings,
        *,
        timeout: float = 120.0,
        sampling: JudgeSampling = JUDGE_SAMPLING,
    ) -> None:
        """Open a pooled HTTP client against the judge endpoint."""
        self.settings = settings
        self.sampling = sampling
        self._schema = judge_output_json_schema()
        self._http = httpx.Client(
            base_url=settings.api_url,
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=timeout,
        )

    def close(self) -> None:
        self._http.close()

    def complete(
        self, messages: tuple[Message, ...]
    ) -> tuple[JudgeOutput, dict[str, int]]:
        payload: dict[str, object] = {
            "model": self.settings.model,
            "messages": list(messages),
            **self.sampling.request_fields(),
            **self.sampling.generation_fields(),
        }
        response_format = self.sampling.response_format(self._schema)
        if response_format is not None:
            payload["response_format"] = response_format
        response = self._http.post("/chat/completions", json=payload)
        response.raise_for_status()
        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            usage = body.get("usage") or {}
            output = JudgeOutput.model_validate(parse_judge_content(content))
            tokens = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            }
        except (
            ValidationError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            AttributeError,
        ) as error:
            raise JudgeParseError(type(error).__name__) from error
        return output, tokens


_MAX_PREFIX_SCAN = 8


def parse_judge_content(content: str) -> dict[str, object]:
    r"""Decode the judge's JSON object, tolerating a duplicated opening brace.

    Thinking-mode responses sometimes start with ``{{`` or ``{\n{`` (see
    ``JudgeSampling.response_format``). The object is decoded from the first
    ``{`` within the first few characters that yields a complete JSON object;
    anything after the object (whitespace, stray text) is ignored. Raises
    ``ValueError`` when no object decodes.
    """
    text = content.strip()
    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(text[:_MAX_PREFIX_SCAN]) if ch == "{"]
    for start in starts or [0]:
        try:
            value, _ = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    msg = "judge content holds no JSON object"
    raise ValueError(msg)


def override_structural_claim(
    output: JudgeOutput, gate: SyntaxGateResult
) -> tuple[JudgeOutput, bool]:
    """Trust the parser over the judge for brace/structure compile claims."""
    if output.compiles or not gate.passed:
        return output, False
    if not _STRUCTURAL_CLAIM.search(output.reason):
        return output, False
    fixed = output.model_copy(update={"compiles": True})
    return fixed.model_copy(update={"verdict": fixed.derived_verdict()}), True


def judge_once(  # noqa: PLR0913 - preserve the existing offline keyword API.
    client: JudgeClient,
    *,
    cache_key: str,
    issue: IssueInput,
    rendered: RenderedPatch,
    gate: SyntaxGateResult,
    max_attempts: int = MAX_ATTEMPTS,
) -> JudgeRecord:
    """Produce the record for one patch: syntax gate first, then the LLM."""
    flags: list[JudgeFlag] = []
    if rendered.window_truncated:
        flags.append("window_truncated")
    if not gate.passed:
        return JudgeRecord(
            cache_key=cache_key,
            idx=issue.idx,
            checker_id=issue.checker_id,
            patch_hash=rendered.patch_hash,
            code_mode=rendered.code_mode,
            syntax_gate=gate,
            judge=None,
            verdict=Verdict.FAIL,
            verdict_source=VerdictSource.SYNTAX_GATE,
            flags=tuple(flags),
            provenance=None,
        )

    messages = build_messages(issue, rendered)
    prompt_tokens = completion_tokens = 0
    output: JudgeOutput | None = None
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        try:
            candidate, usage = client.complete(messages)
        except JudgeParseError:
            flags.append("schema_retry")
            continue
        prompt_tokens += usage["prompt_tokens"]
        completion_tokens += usage["completion_tokens"]
        if candidate.is_consistent():
            output = candidate
            break
        flags.append("judge_inconsistent")
        output = candidate  # keep the last one; verdict is recomputed below
    if output is None:
        msg = f"judge returned no parseable output after {max_attempts} attempts"
        raise JudgeParseError(msg)
    output, overridden = override_structural_claim(output, gate)
    if overridden:
        flags.append("syntax_claim_overridden")

    return JudgeRecord(
        cache_key=cache_key,
        idx=issue.idx,
        checker_id=issue.checker_id,
        patch_hash=rendered.patch_hash,
        code_mode=rendered.code_mode,
        syntax_gate=gate,
        judge=output,
        verdict=output.derived_verdict(),
        verdict_source=VerdictSource.LLM,
        flags=tuple(flags),
        provenance=JudgeProvenance(
            judge_model=client.settings.model,
            temperature=client.sampling.temperature,
            prompt_version=PROMPT_VERSION,
            prompt_sha256=prompt_sha256(messages),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            attempts=attempts,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        ),
    )


@final
class RecordCache:
    """Append-only JSONL cache keyed by ``cache_key``."""

    __slots__: ClassVar[tuple[str, ...]] = ("_path", "_records")

    def __init__(self, path: Path) -> None:
        """Load every record already stored at ``path`` (if it exists)."""
        self._path = path
        self._records: dict[str, JudgeRecord] = {}
        if path.is_file():
            for record in self._iter_file(path):
                self._records[record.cache_key] = record

    @staticmethod
    def _iter_file(path: Path) -> Iterator[JudgeRecord]:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield JudgeRecord.model_validate_json(line)

    def __len__(self) -> int:
        return len(self._records)

    def get(self, cache_key: str) -> JudgeRecord | None:
        return self._records.get(cache_key)

    def put(self, record: JudgeRecord) -> None:
        if record.cache_key in self._records:
            return
        self._records[record.cache_key] = record
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
