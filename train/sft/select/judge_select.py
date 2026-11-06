#!/usr/bin/env python3
"""Stage 2 — listwise LLM judge for inputs whose passing runs disagree on the patch.

Run from the repository root with the judge package environment::

    PYTHONPATH=train/judge/src train/judge/.venv/bin/python \
        train/sft/select/judge_select.py --select-dir datas/sft/select

Requests go to an OpenAI-compatible judge server (for example vLLM). Its endpoint,
model name and key come from ``JUDGE_API_URL`` / ``JUDGE_MODEL`` / ``JUDGE_API_KEY``:
exported environment variables first, then the first ``.env`` found walking up from
this script's directory (see ``train/.env.example``). ``--judge-url`` /
``--judge-model`` / ``--api-key`` override them. The URL and model are required;
the key defaults to ``EMPTY``, which a local vLLM server accepts.

Decision per ``needs_judge`` input:

* Each vote shows the competing patches under shuffled labels A-D and asks for
  per-candidate ``not_redetected`` / ``compiles`` / ``scope_ok`` and a full ranking.
* A vote's winner is its highest-ranked candidate with ``not_redetected`` and
  ``compiles`` both true (none if every candidate is rejected).
* Two votes with different label orders; if their winners agree the decision is
  final. Otherwise a third vote decides by majority. No majority, or all votes
  rejecting every candidate, falls back to the deterministic pick and is flagged.
* Transport / parse failures are retried; an input that still fails is recorded
  as ``judge_error`` and also falls back (re-run the script to retry them).

Results are appended to ``judging/decisions_<model>_<profile>.jsonl`` keyed by
an identity hash (dataset, idx, patch set, prompt version, model, sampling), so
the script resumes where it stopped and never re-asks a decided input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
from secllm_judge.corpus import issue_from_record
from secllm_judge.judge_client import JudgeSampling, find_dotenv, load_dotenv, parse_judge_content

sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge_prompt import (  # noqa: E402
    LABELS,
    PROMPT_VERSION,
    Candidate,
    build_messages,
    candidate_from_raw,
)
from trajectory_features import generation_files, iter_rows, patch_key  # noqa: E402

SCHEMA_VERSION = "sft-select-judge.v1"
# argparse destination -> environment variable that supplies its default.
JUDGE_ENV = {"judge_url": "JUDGE_API_URL", "judge_model": "JUDGE_MODEL", "api_key": "JUDGE_API_KEY"}

PROFILES: dict[str, JudgeSampling] = {
    # Fix-Rate judge defaults with room for 4 candidate assessments + ranking.
    "nonthinking": JudgeSampling(max_tokens=1024),
    # Thinking-mode sampling; about 13x slower per call, but in a 20-input
    # calibration its rankings were far more stable than the non-thinking profile.
    "thinking": JudgeSampling(
        temperature=1.0,
        top_p=0.95,
        presence_penalty=0.0,
        enable_thinking=True,
        max_tokens=16384,
    ),
}


def judge_env() -> dict[str, str]:
    """Judge settings from the environment; exported variables win over ``.env``."""
    dotenv = find_dotenv(Path(__file__).resolve().parent)
    values = load_dotenv(dotenv) if dotenv is not None else {}
    values.update({k: os.environ[k] for k in JUDGE_ENV.values() if k in os.environ})
    # Empty entries (e.g. copied unchanged from .env.example) count as unset.
    return {k: values[k] for k in JUDGE_ENV.values() if values.get(k)}


def parse_args() -> argparse.Namespace:
    env = judge_env()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--select-dir", default="datas/sft/select")
    p.add_argument("--datasets-root", default="datas/sft", help="where <dataset_id>/ directories live")
    p.add_argument("--judge-url", default=env.get("JUDGE_API_URL"),
                   help="OpenAI-compatible base URL ending in /v1 (default: $JUDGE_API_URL)")
    p.add_argument("--judge-model", default=env.get("JUDGE_MODEL"),
                   help="model name served at --judge-url (default: $JUDGE_MODEL)")
    p.add_argument("--api-key", default=env.get("JUDGE_API_KEY", "EMPTY"),
                   help="bearer token for the judge server (default: $JUDGE_API_KEY, else EMPTY; "
                   "a local vLLM server accepts any value)")
    p.add_argument("--profile", choices=sorted(PROFILES), default="nonthinking")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--max-retries", type=int, default=3, help="per vote, for HTTP and parse failures")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample", type=int, default=0, help="judge only N randomly sampled inputs (calibration)")
    p.add_argument("--only", default="", help="comma-separated dataset_id:idx list to judge")
    p.add_argument("--retry-errors", action="store_true", help="re-judge inputs whose last record is judge_error")
    args = p.parse_args()
    missing = [f"--{dest.replace('_', '-')} or {var}" for dest, var in JUDGE_ENV.items()
               if dest != "api_key" and not getattr(args, dest)]
    if missing:
        p.error("judge server not configured; set " + ", ".join(missing))
    return args


# ---------------------------------------------------------------- inputs


def load_needs_judge(select_dir: Path) -> list[dict[str, Any]]:
    path = select_dir / "candidates.jsonl"
    return [r for r in iter_rows(path) if r["status"] == "needs_judge"]


def load_issue_records(datasets_root: Path, dataset_ids: set[str]) -> dict[tuple[str, int], dict[str, Any]]:
    """Map ``(dataset_id, idx)`` to its request record.

    ``sft_candidate_inputs.jsonl`` line number equals the generation ``idx``;
    ``decide`` re-checks this by comparing ``entireCode`` with the generation row.
    """
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for dataset_id in sorted(dataset_ids):
        path = datasets_root / dataset_id / "sft_candidate_inputs.jsonl"
        with path.open(encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if line.strip():
                    records[(dataset_id, idx)] = json.loads(line)
    return records


def load_patches(
    datasets_root: Path, wanted: set[tuple[str, int, int]]
) -> dict[tuple[str, int, int], dict[str, Any]]:
    """One pass over each generation file; keep only the representative rows."""
    found: dict[tuple[str, int, int], dict[str, Any]] = {}
    for dataset_id in sorted({w[0] for w in wanted}):
        for path in generation_files(datasets_root / dataset_id):
            for row in iter_rows(path):
                key = (dataset_id, int(row["idx"]), int(row["run"]))
                if key in wanted:
                    found[key] = {
                        "modifications": list(row["response"]["codeModifications"] or []),
                        "entireCode": row["entireCode"],
                        "prompt_id": row["prompt_id"],
                    }
    missing = wanted - found.keys()
    if missing:
        msg = f"{len(missing)} representative rows not found, e.g. {sorted(missing)[:3]}"
        raise KeyError(msg)
    return found


# ---------------------------------------------------------------- judge call


class JudgeCallError(RuntimeError):
    pass


def validate_vote(content: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    cands = content.get("candidates")
    ranking = content.get("ranking")
    if not isinstance(cands, list) or sorted(c.get("label") for c in cands) != sorted(labels):
        raise ValueError("candidates must assess every label exactly once")
    if not isinstance(ranking, list) or sorted(ranking) != sorted(labels):
        raise ValueError("ranking must be a permutation of the labels")
    for c in cands:
        for key in ("not_redetected", "compiles", "scope_ok"):
            if not isinstance(c.get(key), bool):
                raise ValueError(f"candidate {c.get('label')} missing boolean {key}")
    conf = content.get("confidence_score")
    if not isinstance(conf, int) or not 0 <= conf <= 100:
        raise ValueError("confidence_score must be an integer 0-100")
    if not isinstance(content.get("reason"), str) or not content["reason"].strip():
        raise ValueError("reason must be a non-empty string")
    return content


def call_judge(
    http: httpx.Client,
    *,
    model: str,
    sampling: JudgeSampling,
    messages: tuple[dict[str, str], ...],
    schema: dict[str, Any],
    labels: tuple[str, ...],
    max_retries: int,
) -> tuple[dict[str, Any], dict[str, int], int]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        **sampling.request_fields(),
        **sampling.generation_fields(),
    }
    response_format = sampling.response_format(schema)
    if response_format is not None:
        response_format["json_schema"]["name"] = "listwise_selection"
        payload["response_format"] = response_format
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = http.post("/chat/completions", json=payload)
            response.raise_for_status()
            body = response.json()
            choice = body["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("completion truncated (finish_reason=length)")
            content = validate_vote(parse_judge_content(choice["message"]["content"] or ""), labels)
            usage = body.get("usage") or {}
            tokens = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            }
            return content, tokens, attempt
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as error:
            last_error = f"{type(error).__name__}: {str(error)[:200]}"
            time.sleep(min(2**attempt, 20))
    raise JudgeCallError(last_error)


def vote_winner(content: dict[str, Any], label_to_patch: dict[str, str]) -> str | None:
    ok = {c["label"] for c in content["candidates"] if c["not_redetected"] and c["compiles"]}
    for label in content["ranking"]:
        if label in ok:
            return label_to_patch[label]
    return None


def permutations_for(n: int, rng: random.Random) -> list[list[int]]:
    """Vote 1 random order, vote 2 its reverse (every pair flips), vote 3 random."""
    first = list(range(n))
    rng.shuffle(first)
    third = list(range(n))
    rng.shuffle(third)
    return [first, list(reversed(first)), third]


# ---------------------------------------------------------------- decision


def decide(
    entry: dict[str, Any],
    issue_record: dict[str, Any],
    patches: dict[tuple[str, int, int], dict[str, Any]],
    *,
    http: httpx.Client,
    args: argparse.Namespace,
    sampling: JudgeSampling,
    identity: str,
) -> dict[str, Any]:
    dataset_id, idx = entry["dataset_id"], entry["idx"]
    judge_patches = entry["judge_patches"][: len(LABELS)]
    candidates: list[Candidate] = []
    for jp in judge_patches:
        row = patches[(dataset_id, idx, jp["representative_run"])]
        cand = candidate_from_raw(row["modifications"])
        if cand.patch_key != jp["patch_key"] or patch_key(row["modifications"]) != jp["patch_key"]:
            msg = f"patch key mismatch for {dataset_id}:{idx} run {jp['representative_run']}"
            raise ValueError(msg)
        if row["entireCode"] != issue_record["userRequest"]["entireCode"]:
            msg = f"entireCode mismatch between input and generation for {dataset_id}:{idx}"
            raise ValueError(msg)
        candidates.append(cand)
    issue = issue_from_record(idx, issue_record)
    rng = random.Random(f"{args.seed}|{dataset_id}|{idx}")
    orders = permutations_for(len(candidates), rng)

    votes: list[dict[str, Any]] = []
    winners: list[str | None] = []
    base = {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "dataset_id": dataset_id,
        "idx": idx,
        "patch_keys": [c.patch_key for c in candidates],
        "deterministic_pick": entry["pick"],
        "judge": {
            "url": args.judge_url,
            "model": args.judge_model,
            "profile": args.profile,
            "sampling": {**sampling.request_fields(), **sampling.generation_fields()},
            "prompt_version": PROMPT_VERSION,
        },
    }
    try:
        for vote_index, order in enumerate(orders):
            if vote_index == 2 and winners[0] == winners[1]:
                break
            rendered = build_messages(issue, candidates, order)
            labels = tuple(rendered.label_to_patch)
            started = time.time()
            content, tokens, attempts = call_judge(
                http,
                model=args.judge_model,
                sampling=sampling,
                messages=rendered.messages,
                schema=rendered.schema,
                labels=labels,
                max_retries=args.max_retries,
            )
            winner = vote_winner(content, rendered.label_to_patch)
            winners.append(winner)
            votes.append(
                {
                    "order": order,
                    "label_to_patch": rendered.label_to_patch,
                    "code_mode": rendered.code_mode.value,
                    "prompt_sha256": rendered.prompt_sha256,
                    "output": content,
                    "winner_patch": winner,
                    "tokens": tokens,
                    "attempts": attempts,
                    "elapsed_sec": round(time.time() - started, 2),
                }
            )
    except JudgeCallError as error:
        return {**base, "status": "judge_error", "error": str(error), "votes": votes, "selected_patch": None}

    tally = Counter(w for w in winners if w is not None)
    top = tally.most_common(1)
    if top and top[0][1] >= 2:
        selected, status = top[0][0], "judge"
    elif not tally:
        selected, status = None, "judge_all_rejected"
    else:
        selected, status = None, "judge_no_majority"
    selected_run = None
    if selected is not None:
        selected_run = next(jp["representative_run"] for jp in judge_patches if jp["patch_key"] == selected)
    rejected = Counter(
        rendered_patch
        for v in votes
        for c in v["output"]["candidates"]
        if not (c["not_redetected"] and c["compiles"])
        for rendered_patch in [v["label_to_patch"][c["label"]]]
    )
    return {
        **base,
        "status": status,
        "votes": votes,
        "winners": winners,
        "selected_patch": selected,
        "selected_run": selected_run,
        "agrees_with_deterministic": selected == entry["pick"]["patch_key"] if selected else None,
        "rejected_votes_by_patch": dict(rejected),
        "confidence_first_place": [v["output"]["confidence_score"] for v in votes],
    }


def identity_for(entry: dict[str, Any], args: argparse.Namespace, sampling: JudgeSampling) -> str:
    payload = json.dumps(
        {
            "dataset_id": entry["dataset_id"],
            "idx": entry["idx"],
            "patches": sorted(jp["patch_key"] for jp in entry["judge_patches"]),
            "prompt_version": PROMPT_VERSION,
            "model": args.judge_model,
            "url": args.judge_url,
            "sampling": {**sampling.request_fields(), **sampling.generation_fields()},
            "seed": args.seed,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def load_done(path: Path, *, retry_errors: bool) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for record in iter_rows(path):
            done[record["identity"]] = record
    if retry_errors:
        done = {k: v for k, v in done.items() if v["status"] not in {"judge_error", "input_error"}}
    return done


def main() -> None:
    args = parse_args()
    sampling = PROFILES[args.profile]
    select_dir = Path(args.select_dir)
    datasets_root = Path(args.datasets_root)
    entries = load_needs_judge(select_dir)
    if args.only:
        wanted = {tuple(item.rsplit(":", 1)) for item in args.only.split(",") if item}
        entries = [e for e in entries if (e["dataset_id"], str(e["idx"])) in wanted]
    if args.sample:
        rng = random.Random(args.seed)
        entries = rng.sample(entries, min(args.sample, len(entries)))

    out_dir = select_dir / "judging"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Model names such as "Org/Model" would otherwise create a subdirectory.
    model_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", args.judge_model)
    out_path = out_dir / f"decisions_{model_tag}_{args.profile}.jsonl"
    done = load_done(out_path, retry_errors=args.retry_errors)
    todo = []
    for entry in entries:
        identity = identity_for(entry, args, sampling)
        if identity not in done:
            todo.append((entry, identity))
    print(f"needs_judge={len(entries)} already_decided={len(entries) - len(todo)} todo={len(todo)} -> {out_path}")
    if not todo:
        return

    issue_records = load_issue_records(datasets_root, {e["dataset_id"] for e, _ in todo})
    wanted_rows = {
        (e["dataset_id"], e["idx"], jp["representative_run"]) for e, _ in todo for jp in e["judge_patches"]
    }
    patches = load_patches(datasets_root, wanted_rows)

    lock = threading.Lock()
    status_count: Counter = Counter()
    started = time.time()
    http = httpx.Client(
        base_url=args.judge_url.rstrip("/"),
        headers={"Authorization": f"Bearer {args.api_key}"},
        timeout=args.timeout,
        limits=httpx.Limits(max_connections=args.concurrency * 2),
    )
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool, out_path.open("a", encoding="utf-8") as out:
            futures = {
                pool.submit(
                    decide,
                    entry,
                    issue_records[(entry["dataset_id"], entry["idx"])],
                    patches,
                    http=http,
                    args=args,
                    sampling=sampling,
                    identity=identity,
                ): entry
                for entry, identity in todo
            }
            for n, future in enumerate(as_completed(futures), 1):
                entry = futures[future]
                try:
                    result = future.result()
                except (ValueError, KeyError) as error:
                    # Input/patch inconsistency or an unrenderable patch: not a judge
                    # outcome. Recorded so the export falls back and the report shows it.
                    result = {
                        "schema_version": SCHEMA_VERSION,
                        "identity": identity_for(entry, args, sampling),
                        "dataset_id": entry["dataset_id"],
                        "idx": entry["idx"],
                        "status": "input_error",
                        "error": f"{type(error).__name__}: {str(error)[:300]}",
                        "deterministic_pick": entry["pick"],
                        "selected_patch": None,
                    }
                result["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                with lock:
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out.flush()
                    status_count[result["status"]] += 1
                if n % 10 == 0 or n == len(todo):
                    rate = n / max(time.time() - started, 1e-6) * 3600
                    print(f"{n}/{len(todo)} {dict(status_count)} {rate:.0f}/h", flush=True)
    finally:
        http.close()
    print(f"done in {time.time() - started:.0f}s: {dict(status_count)}")


if __name__ == "__main__":
    main()
