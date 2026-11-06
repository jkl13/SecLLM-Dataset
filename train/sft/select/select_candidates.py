#!/usr/bin/env python3
"""Stage 1 — deterministic selection of one teacher trajectory per input.

Run from the repository root with the root ``.venv`` (needs the Qwen tokenizer
for the max_length filter; pass ``--no-token-length`` to skip it)::

    .venv/bin/python train/sft/select/select_candidates.py \
        --dataset datas/sft/<teacher_set_a> \
        --dataset datas/sft/<teacher_set_b> \
        --out datas/sft/select

Per input (``dataset_id``, ``idx``):

1. hard filter — SAST re-verification ``pass``; harness terminal ``pass``; zero
   edit errors / fuzzy matches / nudges / tool error results / invalid calls;
   exactly one verifier attempt; non-empty patch with at least one added line;
   final ``codeModifications`` equal to the last verifier candidate; rendered
   token length within ``--max-length``.
2. read rule — keep trajectories whose first ``read_code`` follows the prompt's
   pagination rule (or that skipped reading because the target chunk already
   covered the whole file); fall back to all survivors when none complies.
3. patch consensus — keep trajectories whose patch received the most votes
   among hard-filtered runs.
4. ranking — fewer model turns, then fewer cumulative input tokens, then fewer
   patch lines, then patches touching a SAST trace line, then a seeded hash.

If step 3 leaves more than one distinct patch, the input is marked
``needs_judge`` and every distinct patch (with its best trajectory) is listed for
stage 2. The deterministic pick is always recorded as the fallback.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trajectory_features import (  # noqa: E402
    TrajectoryFeatures,
    deterministic_order,
    extract_features,
    generation_files,
    iter_rows,
    training_messages,
)

SCHEMA_VERSION = "sft-select-candidates.v1"
# "unchecked" = --no-token-length; the export stage re-checks length anyway.
TOKEN_OK = frozenset({"ready", "unchecked"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", action="append", required=True, help="teacher generation directory (repeatable)")
    p.add_argument("--out", default="datas/sft/select")
    p.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    p.add_argument("--max-length", type=int, default=12288)
    p.add_argument("--no-token-length", action="store_true", help="skip tokenizer-based length filter")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="only the first N idx per dataset (smoke runs)")
    return p.parse_args()


def rank_key(f: TrajectoryFeatures, seed: int) -> tuple:
    return (
        f.turns,
        f.input_tokens,
        f.patch_lines,
        0 if f.touches_trace_line else 1,
        deterministic_order(seed, f.dataset_id, f.idx, f.run),
    )


def load_tokenizer(name: str):  # noqa: ANN201 - transformers type kept lazy
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def token_status(row: dict[str, Any], tok: Any, max_length: int) -> tuple[str, int]:
    from srft_masking import evaluate_example

    result = evaluate_example({"messages": training_messages(row), "tools": row["tools"]}, tok, max_length)
    return result.training_status, int(result.token_count)


def load_code_hashes(dataset_dir: Path) -> dict[int, str]:
    path = dataset_dir / "sft_candidate_metadata.jsonl"
    hashes: dict[int, str] = {}
    if path.is_file():
        for row in iter_rows(path):
            hashes[int(row["idx"])] = str(row.get("code_hash") or "")
    return hashes


def select_for_idx(
    feats: list[TrajectoryFeatures],
    token_info: dict[int, tuple[str, int]],
    *,
    seed: int,
) -> dict[str, Any]:
    dataset_id, idx = feats[0].dataset_id, feats[0].idx
    sast_pass = [f for f in feats if f.sast_pass]
    hard = [f for f in sast_pass if f.hard_ok and token_info[f.run][0] in TOKEN_OK]
    cand_rows = []
    for f in feats:
        d = f.to_dict()
        d["token_status"], d["token_count"] = token_info[f.run]
        cand_rows.append(d)

    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "idx": idx,
        "language": feats[0].language,
        "runs_total": len(feats),
        "runs_sast_pass": len(sast_pass),
        "runs_hard_ok": len(hard),
        "candidates": cand_rows,
    }
    if not sast_pass:
        return {**base, "status": "dropped", "drop_reason": "no_sast_pass", "pick": None, "judge_patches": []}
    if not hard:
        tally = Counter(r for f in sast_pass for r in f.hard_reasons)
        for f in sast_pass:
            if f.hard_ok and token_info[f.run][0] not in TOKEN_OK:
                tally[f"token_{token_info[f.run][0]}"] += 1
        return {
            **base,
            "status": "dropped",
            "drop_reason": "all_hard_filtered",
            "drop_tally": dict(tally),
            "pick": None,
            "judge_patches": [],
        }

    rule = [f for f in hard if f.read_rule_ok]
    rule_relaxed = not rule
    if rule_relaxed:
        rule = hard
    votes = Counter(f.patch_key for f in hard)
    max_votes = max(votes[f.patch_key] for f in rule)
    top = [f for f in rule if votes[f.patch_key] == max_votes]
    pick = min(top, key=lambda f: rank_key(f, seed))
    groups: dict[str, list[TrajectoryFeatures]] = defaultdict(list)
    for f in top:
        groups[f.patch_key].append(f)

    if len(hard) == 1:
        status = "single"
    elif len(groups) == 1:
        status = "same_patch"
    else:
        status = "needs_judge"

    judge_patches = []
    if status == "needs_judge":
        for key, members in groups.items():
            rep = min(members, key=lambda f: rank_key(f, seed))
            judge_patches.append(
                {
                    "patch_key": key,
                    "votes": votes[key],
                    "representative_run": rep.run,
                    "runs": sorted(m.run for m in members),
                }
            )
        judge_patches.sort(key=lambda d: (-d["votes"], d["representative_run"]))

    return {
        **base,
        "status": status,
        "rule_relaxed": rule_relaxed,
        "max_votes": max_votes,
        "distinct_top_patches": len(groups),
        "pick": {
            "run": pick.run,
            "patch_key": pick.patch_key,
            "source": "deterministic",
            "turns": pick.turns,
            "input_tokens": pick.input_tokens,
            "token_count": token_info[pick.run][1],
        },
        "judge_patches": judge_patches,
    }


def main() -> None:
    args = parse_args()
    datasets = [Path(d) for d in args.dataset]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = None if args.no_token_length else load_tokenizer(args.tokenizer)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "args": vars(args),
        "datasets": {},
        "status": Counter(),
        "drop_reason": Counter(),
        "hard_reason_tally": Counter(),
        "token_status": Counter(),
        "rule_relaxed": 0,
        "judge_patch_count": Counter(),
    }
    out_path = out_dir / "candidates.jsonl"
    started = time.time()
    with out_path.open("w", encoding="utf-8") as out:
        for dataset_dir in datasets:
            dataset_id = dataset_dir.name
            files = generation_files(dataset_dir)
            code_hashes = load_code_hashes(dataset_dir)
            per_idx: dict[int, list[TrajectoryFeatures]] = defaultdict(list)
            token_info: dict[int, dict[int, tuple[str, int]]] = defaultdict(dict)
            rows_seen = 0
            for path in files:
                for row in iter_rows(path):
                    idx = int(row["idx"])
                    if args.limit and idx >= args.limit:
                        continue
                    f = extract_features(row, dataset_id)
                    per_idx[idx].append(f)
                    if f.sast_pass and f.hard_ok and tok is not None:
                        token_info[idx][f.run] = token_status(row, tok, args.max_length)
                    else:
                        token_info[idx][f.run] = ("unchecked" if tok is None else "skipped", 0)
                    rows_seen += 1
                    if rows_seen % 1000 == 0:
                        print(f"[{dataset_id}] {rows_seen} rows, {time.time() - started:.0f}s", flush=True)
            n_status: Counter = Counter()
            for idx in sorted(per_idx):
                feats = sorted(per_idx[idx], key=lambda f: f.run)
                result = select_for_idx(feats, token_info[idx], seed=args.seed)
                result["code_hash"] = code_hashes.get(idx, "")
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                n_status[result["status"]] += 1
                report["status"][result["status"]] += 1
                if result["status"] == "dropped":
                    report["drop_reason"][result["drop_reason"]] += 1
                if result.get("rule_relaxed"):
                    report["rule_relaxed"] += 1
                if result["status"] == "needs_judge":
                    report["judge_patch_count"][str(len(result["judge_patches"]))] += 1
                for c in result["candidates"]:
                    report["token_status"][c["token_status"]] += 1
                    if c["sast_result"] == "pass":
                        for r in c["hard_reasons"]:
                            report["hard_reason_tally"][r] += 1
            report["datasets"][dataset_id] = {
                "files": [p.name for p in files],
                "rows": rows_seen,
                "inputs": len(per_idx),
                "status": dict(n_status),
            }
            print(f"[{dataset_id}] inputs={len(per_idx)} status={dict(n_status)}", flush=True)

    for key in ("status", "drop_reason", "hard_reason_tally", "token_status", "judge_patch_count"):
        report[key] = dict(report[key])
    report["elapsed_sec"] = round(time.time() - started, 1)
    (out_dir / "selection_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "drop_reason", "rule_relaxed", "judge_patch_count")}, ensure_ascii=False))
    print(f"wrote {out_path} and selection_report.json ({report['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
