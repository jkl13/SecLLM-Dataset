#!/usr/bin/env python3
"""Dump every non-``judge`` decision (all rejected / no majority / errors) for review.

Run from the repository root with the judge environment::

    PYTHONPATH=train/judge/src train/judge/.venv/bin/python \
        train/sft/select/audit_unresolved.py \
        --select-dir datas/sft/select \
        --decisions datas/sft/select/judging/decisions_<judge-model>_<profile>.jsonl

Writes ``<select-dir>/audit/<dataset_id>__<idx>.md`` (finding, original code window,
every candidate diff, every vote with per-candidate flags and notes) plus
``audit/index.jsonl`` with one summary line per input.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from secllm_judge.corpus import issue_from_record
from secllm_judge.diff_render import apply_modifications, render_code, unified_diff
from secllm_judge.prompt import _dataflow_block, _examples_block, _issue_block

sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge_prompt import candidate_from_raw  # noqa: E402
from judge_select import load_issue_records, load_patches  # noqa: E402
from trajectory_features import iter_rows  # noqa: E402

UNRESOLVED = ("judge_all_rejected", "judge_no_majority", "judge_error", "input_error")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--select-dir", default="datas/sft/select")
    p.add_argument("--datasets-root", default="datas/sft")
    p.add_argument("--decisions", required=True, help="judge decisions JSONL written by judge_select.py")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sel = Path(args.select_dir)
    decisions_path = Path(args.decisions)
    latest: dict[tuple[str, int], dict] = {}
    for record in iter_rows(decisions_path):
        latest[(record["dataset_id"], int(record["idx"]))] = record
    targets = {k: v for k, v in latest.items() if v["status"] in UNRESOLVED}
    entries = {(e["dataset_id"], int(e["idx"])): e for e in iter_rows(sel / "candidates.jsonl") if (e["dataset_id"], int(e["idx"])) in targets}

    issues = load_issue_records(Path(args.datasets_root), {k[0] for k in targets})
    wanted = {(k[0], k[1], jp["representative_run"]) for k, e in entries.items() for jp in e["judge_patches"]}
    patches = load_patches(Path(args.datasets_root), wanted)

    out_dir = sel / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "index.jsonl").open("w", encoding="utf-8") as index:
        for key in sorted(targets):
            decision, entry = targets[key], entries[key]
            issue = issue_from_record(key[1], issues[key])
            original = issue.entire_code.split("\n")
            cands = []
            for jp in entry["judge_patches"]:
                raw = patches[(key[0], key[1], jp["representative_run"])]["modifications"]
                cands.append((jp, candidate_from_raw(raw)))
            union = tuple(m for _, c in cands for m in c.modifications)
            _, code_text = render_code(original, issue_line=issue.issue_line_number, modifications=union)

            per_patch_flags: dict[str, Counter] = {jp["patch_key"]: Counter() for jp, _ in cands}
            lines = [
                f"# {key[0]} idx {key[1]} — {decision['status']}",
                "",
                f"- language: {issue.language} / checker: {issue.issue_name_en} ({issue.issue_name_ko})",
                f"- reported line: {issue.issue_line_number} / file lines: {len(original)}",
                f"- deterministic pick: run {entry['pick']['run']} patch {entry['pick']['patch_key']}",
                f"- winners per vote: {decision.get('winners')}",
            ]
            if decision.get("error"):
                lines.append(f"- error: {decision['error']}")
            lines += ["", "## Finding", _issue_block(issue), "", "## Dataflow", _dataflow_block(issue), "",
                      "## Checker examples", _examples_block(issue), "",
                      f"## Original code\n```\n{code_text}\n```", ""]
            for jp, cand in cands:
                patched = apply_modifications(original, cand.modifications)
                lines += [
                    f"## Candidate {jp['patch_key']} (runs {jp['runs']}, votes {jp['votes']}, representative run {jp['representative_run']})",
                    f"```diff\n{unified_diff(original, patched)}\n```",
                    "",
                ]
            for vi, vote in enumerate(decision.get("votes") or [], 1):
                out = vote["output"]
                lines.append(f"## Vote {vi} (order {vote['order']}, confidence {out['confidence_score']}, winner {vote['winner_patch']})")
                lines.append("| label | patch | not_redetected | compiles | scope_ok | note |")
                lines.append("|---|---|---|---|---|---|")
                for c in out["candidates"]:
                    pk = vote["label_to_patch"][c["label"]]
                    per_patch_flags[pk][("nr" if c["not_redetected"] else "NR_FALSE")] += 1
                    per_patch_flags[pk][("co" if c["compiles"] else "CO_FALSE")] += 1
                    lines.append(
                        f"| {c['label']} | {pk} | {c['not_redetected']} | {c['compiles']} | {c['scope_ok']} | {c['note'].replace('|', '/')} |"
                    )
                lines += [f"\nranking: {out['ranking']}  ", f"reason: {out['reason']}", ""]
            (out_dir / f"{key[0]}__{key[1]}.md").write_text("\n".join(lines), encoding="utf-8")
            index.write(json.dumps({
                "dataset_id": key[0], "idx": key[1], "status": decision["status"], "language": issue.language,
                "checker": issue.issue_name_en, "n_candidates": len(cands), "winners": decision.get("winners"),
                "deterministic_pick": entry["pick"]["patch_key"],
                "flags_by_patch": {k: dict(v) for k, v in per_patch_flags.items()},
                "file": f"{key[0]}__{key[1]}.md",
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(targets)} audit files to {out_dir}: {dict(Counter(v['status'] for v in targets.values()))}")


if __name__ == "__main__":
    main()
