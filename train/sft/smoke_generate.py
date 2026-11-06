#!/usr/bin/env python3
"""Generation smoke test for a trained adapter (read_code / edit_code / submit_fix format).

Qwen3.5-family models have been reported to truncate tool calls as `<tool_call><|im_end|>`
after SFT (ms-swift #9234), so this generates the next turn at three points of each val
trajectory and checks it.
  T1 first turn:  given system+user only → is there one complete read_code or edit_code
                  tool call?
  T2 edit turn:   given everything before the last edit_code → is there one complete
                  edit_code tool call with old_string/new_string filled in?
  T3 submit turn: given everything before the last assistant (submit_fix) → is there one
                  complete submit_fix tool call with a Korean desc (≤300 chars)?
Every turn must produce exactly one tool call (every assistant turn in the training data
carries one call). T2 is checked only on trajectories where the teacher took at least one
turn before editing (otherwise SKIP).

The val file is expected in the train/sft/select/export_sft.py output format (the last
message is the submit_fix call). Prompts are rendered with enable_thinking=False, the same
as training-time masking. Results are written to <adapter>/smoke_test.json as
[{test, idx, dataset_id, language, result}].
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

TOOL_CALL_RE = re.compile(r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RE = re.compile(r"<parameter=(\w+)>\n(.*?)\n</parameter>", re.S)
HANGUL_RE = re.compile(r"[가-힣]")
DESC_MAX_CHARS = 300  # submit_fix desc limit (from the harness system prompt)
DEFAULT_VAL = "datas/sft/select/export/val.jsonl"


def parse_single_call(text: str, allowed: set[str]) -> tuple[str, dict[str, str]] | str:
    """Return the single complete tool call as (name, params), or else a failure reason."""
    calls = TOOL_CALL_RE.findall(text)
    if not calls:
        return f"no complete tool call block: {text[:120]!r}"
    if len(calls) > 1:
        return f"{len(calls)} tool calls (expected 1)"
    name, body = calls[0]
    if name not in allowed:
        return f"unexpected tool {name!r} (allowed: {sorted(allowed)})"
    return name, dict(PARAM_RE.findall(body))


def check_read(params: dict[str, str]) -> str | None:
    for key in ("offset", "limit"):
        if key in params and not params[key].strip().isdigit():
            return f"read_code {key} is not an integer: {params[key][:20]!r}"
    return None


def check_edit(params: dict[str, str]) -> str | None:
    missing = {"old_string", "new_string"} - params.keys()
    if missing:
        return f"edit_code is missing parameters: {sorted(missing)}"
    if not params["old_string"].strip():
        return "old_string is empty"
    if params["old_string"] == params["new_string"]:
        return "old_string equals new_string (no-op)"
    if re.match(r"^\s*\d+: ", params["old_string"]):
        return "old_string includes a line-number prefix"
    return None


def check_submit(params: dict[str, str]) -> str | None:
    desc = params.get("desc", "").strip()
    if not desc:
        return "submit_fix desc is empty"
    if len(desc) > DESC_MAX_CHARS:
        return f"desc is {len(desc)} chars (must be ≤{DESC_MAX_CHARS})"
    if not HANGUL_RE.search(desc):
        return f"desc is not in Korean: {desc[:60]!r}"
    return None


def check(text: str, allowed: set[str]) -> str | None:
    parsed = parse_single_call(text, allowed)
    if isinstance(parsed, str):
        return parsed
    name, params = parsed
    return {"read_code": check_read, "edit_code": check_edit, "submit_fix": check_submit}[name](params)


def called(message: dict, name: str) -> bool:
    return any(tc["function"]["name"] == name for tc in message.get("tool_calls") or [])


def pick_rows(rows: list[dict], n: int) -> list[dict]:
    """Pick n rows round-robin by language.

    val is sorted by dataset/idx, so its first rows all come from one language.
    """
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_lang[row.get("language", "?")].append(row)
    ordered = [r for group in zip_longest(*by_lang.values()) for r in group if r is not None]
    return ordered[:n]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="adapters/<run_name>")
    p.add_argument("--val-file", default=DEFAULT_VAL)
    p.add_argument("--n", type=int, default=6, help="Number of val trajectories to check (round-robin by language)")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-9B", dtype=torch.bfloat16, device_map="cuda:0"
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = pick_rows([json.loads(l) for l in open(args.val_file, encoding="utf-8")], args.n)
    results: list[dict] = []
    # A turn ends at <|im_end|>. When the base checkpoint has no generation_config.json,
    # generate() only watches the config eos (<|endoftext|>) and does not stop, so the
    # model keeps writing tool results and further turns. Stop explicitly on both tokens.
    stop_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]

    def generate(messages, tools):
        prompt = tok.apply_chat_template(
            messages, tools=tools, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=stop_ids,
            )
        return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def record(test: str, row: dict, err: str | None, skipped: str | None = None) -> None:
        result = f"SKIP: {skipped}" if skipped else ("PASS" if err is None else f"FAIL: {err}")
        results.append({
            "test": test, "idx": row["idx"], "dataset_id": row.get("dataset_id"),
            "language": row.get("language"), "result": result,
        })
        print(f"{test} {row.get('dataset_id')}:{row['idx']} ({row.get('language')}): {results[-1]['result']}")

    for row in rows:
        ms, tools = row["messages"], row["tools"]
        if not (ms[-1]["role"] == "assistant" and called(ms[-1], "submit_fix")):
            record("T0-format", row, "last message is not a submit_fix call (is this export_sft.py output?)")
            continue

        # T1: first turn (system + user only)
        record("T1-first", row, check(generate(ms[:2], tools), {"read_code", "edit_code"}))

        # T2: everything before the last edit_code turn. When the teacher edited in its
        # first turn, this prompt equals T1's (system+user) and starting with read_code
        # is also valid behavior, so the check is skipped.
        edit_at = max(i for i, m in enumerate(ms) if m["role"] == "assistant" and called(m, "edit_code"))
        if edit_at > 2:
            record("T2-edit", row, check(generate(ms[:edit_at], tools), {"edit_code"}))
        else:
            record("T2-edit", row, None, skipped="teacher's first turn is edit_code (same prompt as T1)")

        # T3: everything before the last submit_fix turn (after the edit OK result)
        record("T3-submit", row, check(generate(ms[:-1], tools), {"submit_fix"}))

    n_pass = sum(1 for r in results if r["result"] == "PASS")
    n_run = sum(1 for r in results if not r["result"].startswith("SKIP"))
    print(f"\nResult: {n_pass}/{n_run} PASS ({len(results) - n_run} skipped)")
    out = Path(args.adapter) / "smoke_test.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
