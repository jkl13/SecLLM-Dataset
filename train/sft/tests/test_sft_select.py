"""Pure-logic tests for train/sft/select (no tokenizer, no judge server).

Run from the repository root:

    PYTHONPATH=train/judge/src:train/sft/select:train/sft \
        train/judge/.venv/bin/python -m pytest train/sft/tests/test_sft_select.py -q
"""

import random

import pytest
from export_sft import resolve, split_for
from judge_prompt import build_messages, candidate_from_raw
from judge_select import permutations_for, validate_vote, vote_winner
from secllm_judge.diff_render import patch_hash
from secllm_judge.schemas import ContextEvent, ContextEventType, IssueInput
from select_candidates import select_for_idx
from trajectory_features import extract_features, patch_key, training_messages

CODE = "\n".join(f"line{i}" for i in range(1, 21))


def gen_row(run, *, content="fixed", sast="pass", turns=3, in_tok=1000, read=True, fuzzy=0, attempts=1):
    mods = [
        {"type": "delete", "startLine": 5, "endLine": 5, "content": None},
        {"type": "add", "startLine": 5, "content": content},
    ]
    user = (
        '{"programmingLanguage": "java", "totalLines": 20, "issueLineNumber": 5}\n'
        "## SAST trace\n"
        '[{"lineNo": 5, "eventType": "sink"}]\n'
        "## Target chunks\nlines 1-20\n"
    )
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": user}]
    loop = []
    if read:
        loop.append({"ev": "model_turn", "tool_calls": [{"tool": "read_code", "args": {"offset": "0", "limit": "20"}}]})
    loop.append({"ev": "model_turn", "tool_calls": [{"tool": "edit_code", "args": {}}]})
    loop.append({"ev": "tool_result", "result": "OK. Working copy updated."})
    loop.append({"ev": "model_turn", "tool_calls": [{"tool": "submit_fix", "args": {}}]})
    msgs.append({"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "submit_fix", "arguments": {"desc": "d"}}}]})
    msgs.append({"role": "tool", "content": "PASS (attempt 1/3)."})
    return {
        "idx": 7,
        "run": run,
        "language": "Java",
        "passed": True,
        "terminal_cause": "pass",
        "conversation_valid": True,
        "sast_result": sast,
        "sast_verdict": "fixed" if sast == "pass" else "persist",
        "response": {"codeModifications": mods, "desc": "d"},
        "trace": {
            "model_turns": turns,
            "input_tokens": in_tok,
            "output_tokens": 10,
            "edit_errors": 0,
            "fuzzy_matches": fuzzy,
            "nudges": 0,
            "attempts": [{"attempt": k + 1, "passed": k + 1 == attempts} for k in range(attempts)],
            "candidates": [{"codeModifications": mods}],
        },
        "loop": loop,
        "messages": msgs,
        "entireCode": CODE,
    }


def feats(*rows):
    return [extract_features(r, "ds") for r in rows]


def ready(fs):
    return {f.run: ("ready", 100) for f in fs}


def test_patch_key_matches_judge_patch_hash():
    raw = gen_row(1)["response"]["codeModifications"]
    assert patch_key(raw) == patch_hash(candidate_from_raw(raw).modifications)


def test_training_messages_drops_trailing_verifier_result():
    msgs = training_messages(gen_row(1))
    assert msgs[-1]["role"] == "assistant"


def test_hard_filters():
    fuzzy, retry, fine = feats(gen_row(1, fuzzy=1), gen_row(2, attempts=2), gen_row(3))
    assert "fuzzy_match" in fuzzy.hard_reasons
    assert "verifier_retry" in retry.hard_reasons
    assert fine.hard_ok


def test_same_patch_prefers_fewer_turns_then_tokens():
    fs = feats(gen_row(1, turns=4), gen_row(2, turns=3, in_tok=900), gen_row(3, turns=3, in_tok=800))
    result = select_for_idx(fs, ready(fs), seed=42)
    assert result["status"] == "same_patch"
    assert result["pick"]["run"] == 3


def test_majority_patch_beats_shorter_singleton():
    fs = feats(gen_row(1, content="a", turns=2), gen_row(2, content="b", turns=5), gen_row(3, content="b", turns=6))
    result = select_for_idx(fs, ready(fs), seed=42)
    assert result["status"] == "same_patch"
    assert result["pick"]["run"] == 2


def test_competing_patches_need_judge():
    fs = feats(gen_row(1, content="a"), gen_row(2, content="b"), gen_row(3, sast="fail"))
    result = select_for_idx(fs, ready(fs), seed=42)
    assert result["status"] == "needs_judge"
    assert sorted(p["representative_run"] for p in result["judge_patches"]) == [1, 2]


def test_overlength_and_no_pass_drop():
    fs = feats(gen_row(1), gen_row(2, sast="fail"))
    result = select_for_idx(fs, {1: ("overlength", 13000), 2: ("skipped", 0)}, seed=42)
    assert result["status"] == "dropped"
    assert result["drop_tally"] == {"token_overlength": 1}


def issue():
    return IssueInput(
        idx=7,
        checker_id=1,
        language="Java",
        issue_name_ko="k",
        issue_name_en="e",
        issue_description_ko="d",
        issue_line_number=5,
        contexts=(ContextEvent(line_no=5, event_type=ContextEventType.SINK, message="m"),),
        dangerous_example_ko="",
        safe_example_ko="",
        entire_code=CODE,
    )


def test_prompt_labels_follow_order_and_hide_runs():
    a = candidate_from_raw(gen_row(1, content="alpha")["response"]["codeModifications"])
    b = candidate_from_raw(gen_row(2, content="beta")["response"]["codeModifications"])
    rendered = build_messages(issue(), [a, b], [1, 0])
    assert rendered.label_to_patch == {"A": b.patch_key, "B": a.patch_key}
    user = rendered.messages[1]["content"]
    assert user.index("+beta") < user.index("+alpha")
    assert "run" not in user.lower().split("## candidate patches")[1].split("## output")[0]


def vote(ranking, rejected=()):
    return {
        "candidates": [
            {"label": x, "not_redetected": x not in rejected, "compiles": True, "scope_ok": True, "note": "n"}
            for x in ranking
        ],
        "ranking": list(ranking),
        "confidence_score": 80,
        "reason": "r",
    }


def test_vote_winner_skips_rejected_candidates():
    assert vote_winner(vote("AB", rejected="A"), {"A": "p1", "B": "p2"}) == "p2"
    assert vote_winner(vote("AB", rejected="AB"), {"A": "p1", "B": "p2"}) is None


def test_validate_vote_rejects_incomplete_ranking():
    bad = vote("AB")
    bad["ranking"] = ["A"]
    with pytest.raises(ValueError, match="permutation"):
        validate_vote(bad, ("A", "B"))


def test_second_vote_reverses_first():
    orders = permutations_for(4, random.Random(0))
    assert orders[1] == list(reversed(orders[0]))


def test_resolve_policies():
    class Args:
        allow_missing_judge = False
        unresolved = "fallback"
        all_rejected = "drop"

    entry = {"dataset_id": "ds", "idx": 7, "status": "needs_judge", "pick": {"run": 1}}
    assert resolve(entry, {"status": "judge", "selected_run": 3}, Args) == (3, "judge")
    assert resolve(entry, {"status": "judge_no_majority"}, Args) == (1, "judge_no_majority_fallback")
    assert resolve(entry, {"status": "judge_all_rejected"}, Args) == (None, "dropped:judge_all_rejected")
    with pytest.raises(SystemExit):
        resolve(entry, None, Args)


def test_split_is_stable_per_code_hash():
    assert split_for("C++", "abc") == split_for("cpp", "abc")
