#!/usr/bin/env python3
"""Pre-training sanity check: chat template rendering + assistant label masking.

Rows are tokenized with the same srft_masking.evaluate_example() as train_sft.py, so if
every row comes out ready here, the training inputs are built as intended. The
"Training signal ratio" line in the output is the share of all tokens that are
assistant tokens carrying loss.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from srft_masking import evaluate_example
from transformers import AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datas/sft/select/export/train.jsonl")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--max-length", type=int, default=12288)
    p.add_argument("--show", type=int, default=1, help="Number of samples whose rendering is printed")
    p.add_argument("--report", type=Path, help="Write per-row token eligibility JSON")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    if "generation %}" in (tok.chat_template or ""):
        print("NOTE: template has a {% generation %} marker -> TRL assistant_only_loss is usable")
    else:
        print("NOTE: no {% generation %} marker -> the manual masking path is the right one")

    with open(args.data, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    tot, tgt, results = [], [], []

    for n, row in enumerate(rows):
        result = evaluate_example(row, tok, args.max_length)
        results.append({"row_index": n, "training_status": result.training_status,
                        "reason": result.reason, "token_count": result.token_count,
                        "supervised_tokens": result.supervised_tokens})
        ex = result.example
        if ex is None:
            continue
        ids, labels = ex["input_ids"], ex["labels"]
        assert len(ids) == len(labels), "input_ids/labels length mismatch"
        n_tgt = sum(1 for x in labels if x != -100)
        assert n_tgt > 0, "no target tokens"
        tot.append(len(ids))
        tgt.append(n_tgt)

        if n < args.show:
            print("=" * 70)
            print(tok.decode(ids))
            print("-" * 30, "decoded loss-target spans only", "-" * 30)
            print(tok.decode([i for i, l in zip(ids, labels) if l != -100]))
            print("=" * 70)

    counts = dict(Counter(item["training_status"] for item in results))
    print(f"dispositions: {json.dumps(counts, sort_keys=True)}")
    if args.report:
        args.report.write_text(json.dumps({"counts": counts, "rows": results}, indent=2) + "\n")
    if counts.get("invalid_annotations"):
        raise SystemExit("invalid SRFT annotations; see dispositions/report")
    if not tot:
        print("No trainable rows")
        return
    def q(values: list[int], ratio: float) -> int:
        return sorted(values)[int(len(values) * ratio)]

    print(f"Passed: {len(tot)} rows")
    print(f"TOTAL  p50={q(tot, .5)} p90={q(tot, .9)} p99={q(tot, .99)} max={max(tot)}")
    print(f"TARGET p50={q(tgt, .5)} p90={q(tgt, .9)} max={max(tgt)} mean={statistics.mean(tgt):.0f}")
    print(f"Training signal ratio: {100 * sum(tgt) / sum(tot):.1f}%  (total target tokens {sum(tgt):,})")


if __name__ == "__main__":
    main()
