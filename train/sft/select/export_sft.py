#!/usr/bin/env python3
"""Stage 3 — merge deterministic picks and judge decisions into SFT train/val files.

Run from the repository root with the root ``.venv``::

    .venv/bin/python train/sft/select/export_sft.py \
        --select-dir datas/sft/select \
        --decisions datas/sft/select/judging/decisions_<judge-model>_<profile>.jsonl

Final pick per input:

* ``single`` / ``same_patch`` — the deterministic pick from stage 1.
* ``needs_judge`` — the judge's majority patch (its best-ranked trajectory).
  ``judge_no_majority`` / ``judge_error`` / ``input_error`` follow ``--unresolved``
  (default: deterministic fallback). ``judge_all_rejected`` follows
  ``--all-rejected`` (default: drop). An input without any decision aborts the
  export unless ``--allow-missing-judge`` is given.

Rows keep the teacher conversation without the trailing verifier ``PASS`` tool
message, ``tool_calls[*].function.arguments`` as dicts (as stored), and the
``tools`` list. The split is grouped by original code hash: sha256 of
``"42|<language>|<code_hash>"``, first 8 hex digits mod 10,000 below 300 go to
validation (3%; the rule earlier SRFT exports used). Every exported row is
re-tokenized with the training masker; any non-``ready`` row aborts the export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajectory_features import generation_files, iter_rows, training_messages  # noqa: E402

SCHEMA_VERSION = "sft-select-export.v1"
VALIDATION_NUMERATOR = 300  # out of 10,000
_LANG = {
    "java": "java",
    "javascript": "javascript",
    "js": "javascript",
    "c": "c",
    "cpp": "cpp",
    "c++": "cpp",
    "go": "go",
    "golang": "go",
    "python": "python",
    "py": "python",
    "typescript": "typescript",
    "ts": "typescript",
    "tsx": "tsx",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--select-dir", default="datas/sft/select")
    p.add_argument("--datasets-root", default="datas/sft")
    p.add_argument(
        "--decisions",
        required=True,
        help="judge decisions JSONL written by judge_select.py (judging/decisions_<model>_<profile>.jsonl)",
    )
    p.add_argument("--unresolved", choices=("fallback", "drop"), default="fallback")
    p.add_argument("--all-rejected", choices=("keep", "drop"), default="drop")
    p.add_argument("--allow-missing-judge", action="store_true")
    p.add_argument("--out", default="", help="output dir (default: <select-dir>/export)")
    p.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    p.add_argument("--max-length", type=int, default=12288)
    return p.parse_args()


def split_for(language: str, code_hash: str) -> str:
    lang = _LANG.get(language.strip().casefold())
    if lang is None:
        msg = f"unsupported language for split: {language}"
        raise ValueError(msg)
    digest = hashlib.sha256(f"42|{lang}|{code_hash}".encode()).hexdigest()
    return "val" if int(digest[:8], 16) % 10000 < VALIDATION_NUMERATOR else "train"


def load_decisions(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    if path.is_file():
        for record in iter_rows(path):
            latest[(record["dataset_id"], int(record["idx"]))] = record  # later lines win
    return latest


def resolve(entry: dict[str, Any], decision: dict[str, Any] | None, args: argparse.Namespace) -> tuple[int | None, str]:
    status = entry["status"]
    if status == "dropped":
        return None, f"dropped:{entry['drop_reason']}"
    if status in {"single", "same_patch"}:
        return entry["pick"]["run"], status
    if decision is None:
        if not args.allow_missing_judge:
            msg = f"no judge decision for {entry['dataset_id']}:{entry['idx']} (run judge_select.py first)"
            raise SystemExit(msg)
        return entry["pick"]["run"], "judge_missing_fallback"
    jstatus = decision["status"]
    if jstatus == "judge":
        return int(decision["selected_run"]), "judge"
    if jstatus == "judge_all_rejected":
        return (entry["pick"]["run"], "judge_all_rejected_kept") if args.all_rejected == "keep" else (None, "dropped:judge_all_rejected")
    # judge_no_majority, judge_error, input_error
    return (entry["pick"]["run"], f"{jstatus}_fallback") if args.unresolved == "fallback" else (None, f"dropped:{jstatus}")


def main() -> None:
    args = parse_args()
    select_dir = Path(args.select_dir)
    datasets_root = Path(args.datasets_root)
    out_dir = Path(args.out) if args.out else select_dir / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = Path(args.decisions)
    decisions = load_decisions(decisions_path)

    entries = list(iter_rows(select_dir / "candidates.jsonl"))
    picks: dict[tuple[str, int, int], dict[str, Any]] = {}
    stage_count: Counter = Counter()
    for entry in entries:
        key = (entry["dataset_id"], int(entry["idx"]))
        run, stage = resolve(entry, decisions.get(key), args)
        stage_count[stage] += 1
        if run is not None:
            picks[(*key, run)] = {"stage": stage, "entry": entry, "decision": decisions.get(key)}

    from srft_masking import evaluate_example
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    token_counts: list[int] = []
    target_counts: list[int] = []
    lang_split: Counter = Counter()
    issue_names: dict[tuple[str, int], str] = {}
    for dataset_id in sorted({k[0] for k in picks}):
        for meta in iter_rows(datasets_root / dataset_id / "sft_candidate_metadata.jsonl"):
            issue_names[(dataset_id, int(meta["idx"]))] = meta.get("issueNameKo")
    found = 0
    for dataset_id in sorted({k[0] for k in picks}):
        for path in generation_files(datasets_root / dataset_id):
            for raw in iter_rows(path):
                key = (dataset_id, int(raw["idx"]), int(raw["run"]))
                if key not in picks:
                    continue
                found += 1
                info = picks[key]
                entry = info["entry"]
                row = {
                    "messages": training_messages(raw),
                    "tools": raw["tools"],
                    "idx": key[1],
                    "run": key[2],
                    "dataset_id": dataset_id,
                    "language": raw["language"],
                    "quality_verdict": "sast_pass",
                    "issueNameKo": issue_names.get((dataset_id, key[1])),
                    "selection": {
                        "stage": info["stage"],
                        "patch_key": next(c["patch_key"] for c in entry["candidates"] if c["run"] == key[2]),
                        "sast_verdict": raw.get("sast_verdict"),
                        "sast_grader": raw.get("sast_grader"),
                    },
                }
                result = evaluate_example({"messages": row["messages"], "tools": row["tools"]}, tok, args.max_length)
                if result.training_status != "ready":
                    msg = f"{dataset_id}:{key[1]} run {key[2]} is {result.training_status} ({result.reason})"
                    raise SystemExit(msg)
                token_counts.append(result.token_count)
                target_counts.append(result.supervised_tokens)
                split = split_for(raw["language"], entry.get("code_hash") or hashlib.sha256(raw["entireCode"].encode()).hexdigest())
                lang_split[(split, raw["language"])] += 1
                rows[split].append(row)
    if found != len(picks):
        msg = f"found {found} of {len(picks)} picked rows"
        raise SystemExit(msg)

    for split, items in rows.items():
        items.sort(key=lambda r: (r["dataset_id"], r["idx"]))
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    stats = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "args": vars(args),
        "decisions_file": str(decisions_path),
        "decisions_loaded": len(decisions),
        "inputs": len(entries),
        "stage": dict(stage_count),
        "rows": {k: len(v) for k, v in rows.items()},
        "by_split_language": {f"{s}/{lang}": n for (s, lang), n in sorted(lang_split.items())},
        "tokens": {
            "total_p50": statistics.median(token_counts) if token_counts else 0,
            "total_max": max(token_counts, default=0),
            "target_mean": round(statistics.mean(target_counts), 1) if target_counts else 0,
            "target_sum": sum(target_counts),
            "signal_ratio": round(sum(target_counts) / max(sum(token_counts), 1), 4),
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
