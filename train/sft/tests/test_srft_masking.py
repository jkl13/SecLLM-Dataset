"""Actual cached Qwen tokenizer regressions for canonical SRFT masking."""

import copy
import hashlib
import json

import pytest
import torch
from srft_masking import AnnotationError, build_example, evaluate_example
from train_sft import PadCollator
from transformers import AutoTokenizer


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", local_files_only=True)


@pytest.fixture
def row():
    messages = [
        {"role": "system", "content": "Repair safely."},
        {"role": "user", "content": "Fix validation."},
        {
            "role": "assistant",
            "content": "Inspect input.",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "edit_code",
                        "arguments": {"replacement": "unsafe"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "Validation failed."},
        {"role": "assistant", "content": "Keep validation."},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "edit_code",
                "description": "Edit code",
                "parameters": {
                    "type": "object",
                    "properties": {"replacement": {"type": "string"}},
                    "required": ["replacement"],
                },
            },
        }
    ]
    annotations = [
        {
            "message_index": i,
            "message_sha256": hashlib.sha256(
                json.dumps(
                    m, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "label": "harmful" if i == 2 else "good",
            "weight": 0 if i == 2 else 1,
            "origin": "critic",
            "reason": "Fixture observation",
            "evidence": [
                {
                    "location": f"messages[{i}]",
                    "observation": "Fixture",
                    "basis": "recorded",
                }
            ],
            "judgment_ref": "fixture-review",
        }
        for i, m in enumerate(messages)
        if m["role"] == "assistant"
    ]
    return {
        "schema_version": "secllm-srft.v1",
        "policy_version": "security-srft.v1",
        "training_status": "pending_tokenization",
        "messages": messages,
        "tools": tools,
        "assistant_supervision": annotations,
    }


def test_context_and_attention_preserved_when_harmful_turn_masked(row, tok):
    # Given: the complete same conversation with and without annotations.
    original = copy.deepcopy(row)
    legacy = build_example(
        {"messages": row["messages"], "tools": row["tools"]}, tok, 12288
    )
    # When: SRFT masks the first assistant action.
    result = evaluate_example(row, tok)
    # Then: only labels change, and later recovery has the entire observed history.
    assert result.training_status == "ready"
    assert row == original
    assert result.example["input_ids"] == legacy["input_ids"]
    collator = PadCollator(tok.pad_token_id)
    assert torch.equal(
        collator([result.example])["attention_mask"],
        collator([legacy])["attention_mask"],
    )
    assert result.supervised_tokens < sum(v != -100 for v in legacy["labels"])
    text = tok.decode(result.example["input_ids"])
    assert (
        "unsafe" in text and "Validation failed." in text and "Keep validation." in text
    )
    supervised = tok.decode([v for v in result.example["labels"] if v != -100])
    assert "Keep validation." in supervised and "unsafe" not in supervised
    assert all(
        new == -100
        for old, new in zip(legacy["labels"], result.example["labels"])
        if old == -100
    )


def test_direct_shifted_ce_when_harmful_then_recovery(row, tok):
    # Given: real token labels and positions predicting harmful/recovery tokens.
    legacy = build_example(
        {"messages": row["messages"], "tools": row["tools"]}, tok, 12288
    )
    result = evaluate_example(row, tok)
    positions = [
        i
        for i, (old, new) in enumerate(zip(legacy["labels"], result.example["labels"]))
        if old != -100 and new == -100
    ][:2]
    positions += [
        i for i, value in enumerate(result.example["labels"]) if value != -100
    ][:2]
    shift_labels = torch.tensor(result.example["labels"][1:])
    logits = torch.zeros((4, len(tok)), requires_grad=True)
    # When: causal LM cross entropy predicts each selected token from its predecessor.
    losses = torch.nn.functional.cross_entropy(
        logits,
        shift_labels[torch.tensor(positions) - 1],
        ignore_index=-100,
        reduction="none",
    )
    losses.sum().backward()
    # Then: harmful tokens have zero direct loss/gradient; recovery stays supervised.
    assert losses[:2].tolist() == [0.0, 0.0]
    assert torch.all(losses[2:] > 0)
    assert torch.count_nonzero(logits.grad[:2]) == 0
    assert torch.count_nonzero(logits.grad[2:]) > 0


def test_labels_equal_legacy_when_all_resolved(row, tok):
    # Given: every assistant uses the resolved rule.
    for item in row["assistant_supervision"]:
        item.update(origin="resolved_rule", label=None, weight=1)
    legacy = build_example(
        {"messages": row["messages"], "tools": row["tools"]}, tok, 12288
    )
    # When: tokenizing the resolved canonical row.
    result = evaluate_example(row, tok)
    # Then: all token IDs and labels equal legacy behavior.
    assert result.example == legacy


@pytest.mark.parametrize("index", [2, 4])
def test_reasoning_prefix_preserves_context_and_supervision(row, tok, index):
    message = row["messages"][index]
    message["content"] = "Recorded reasoning.</think>\n\n" + message["content"]
    annotation = next(
        item for item in row["assistant_supervision"] if item["message_index"] == index
    )
    annotation["message_sha256"] = hashlib.sha256(
        json.dumps(
            message, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    original = copy.deepcopy(row)
    result = evaluate_example(row, tok)
    assert result.training_status == "ready", result.reason
    assert row == original
    full_text = tok.apply_chat_template(
        row["messages"], tools=row["tools"], tokenize=False
    )
    ids = tok(full_text, add_special_tokens=False)["input_ids"]
    assert result.example["input_ids"] == ids
    targets = tok.decode([label for label in result.example["labels"] if label != -100])
    assert result.supervised_tokens > 0
    assert "Keep validation." in targets and "unsafe" not in targets
    assert ("Recorded reasoning." in targets) == (index == 4)


@pytest.mark.parametrize(
    "fault",
    [
        "version",
        "policy",
        "missing",
        "duplicate",
        "nonassistant",
        "hash",
        "bool",
        "float",
        "negative",
        "unknown_label",
        "mismatch",
        "origin",
        "evidence",
    ],
)
def test_rejects_malformed_annotations_when_mapping_invalid(row, tok, fault):
    # Given: one independently malformed canonical field.
    item = row["assistant_supervision"][0]
    changes = {
        "version": lambda: row.update(schema_version="other"),
        "policy": lambda: row.update(policy_version="other"),
        "missing": lambda: row.pop("assistant_supervision"),
        "duplicate": lambda: row["assistant_supervision"].append(copy.deepcopy(item)),
        "nonassistant": lambda: item.update(message_index=1),
        "hash": lambda: item.update(message_sha256="0" * 64),
        "bool": lambda: item.update(weight=False),
        "float": lambda: item.update(weight=0.0),
        "negative": lambda: item.update(weight=-1),
        "unknown_label": lambda: item.update(label="other"),
        "mismatch": lambda: item.update(weight=1),
        "origin": lambda: item.update(origin="resolved_rule"),
        "evidence": lambda: item.update(evidence=[]),
    }
    changes[fault]()
    # When: tokenizing malformed annotations through the legacy public entry point.
    # Then: no fallback to ordinary assistant supervision is possible.
    with pytest.raises(AnnotationError):
        build_example(row, tok, 12288)


@pytest.mark.parametrize(
    "status", ["pending_judgment", "needs_review", "excluded_generation"]
)
def test_holds_whole_row_when_canonical_status_pending(row, tok, status):
    # Given: a hold despite fully assigned weights.
    row["training_status"] = status
    # When: eligibility is evaluated.
    result = evaluate_example(row, tok)
    # Then: the explicit hold survives.
    assert result.training_status == status and result.example is None


def test_holds_whole_row_when_one_weight_unknown(row, tok):
    # Given: a pending action and a retained recovery action.
    row["assistant_supervision"][0].update(label="needs_review", weight=None)
    # When: eligibility is evaluated.
    result = evaluate_example(row, tok)
    # Then: partial supervision is not published.
    assert result.training_status == "needs_review" and result.example is None


def test_reports_no_targets_when_all_masked(row, tok):
    # Given: every assistant action masked.
    row["assistant_supervision"][1].update(label="harmful", weight=0)
    # When: eligibility is evaluated.
    result = evaluate_example(row, tok)
    # Then: zero targets is distinct from length exclusion.
    assert result.training_status == "zero_supervised_tokens"
    assert result.token_count > 0 and result.example is None


def test_reports_overlength_without_truncation_when_limit_exceeded(row, tok):
    # Given: original input exceeds the production cap.
    row["messages"][1]["content"] = "input " * 13000
    original = copy.deepcopy(row)
    # When: eligibility is evaluated at the production limit.
    result = evaluate_example(row, tok, 12288)
    # Then: it reports the full length and preserves canonical input.
    assert result.training_status == "overlength" and result.token_count > 12288
    assert row == original and result.example is None


def test_rejects_unstable_template_when_later_user_changes_history(row, tok):
    # Given: Qwen changes historical thinking rendering after another user turn.
    legacy = {"messages": copy.deepcopy(row["messages"]), "tools": row["tools"]}
    legacy["messages"][3] = {"role": "user", "content": "Continue."}
    # When: the original prefix invariant cannot identify assistant boundaries.
    # Then: the public trainer API still raises instead of supervising wrong tokens.
    with pytest.raises(AnnotationError, match="template_prefix"):
        build_example(legacy, tok, 12288)


def test_includes_exact_length_when_at_limit(row, tok):
    # Given: the real, complete rendered sequence length.
    full = evaluate_example(row, tok)
    # When: the cap equals that length.
    result = evaluate_example(row, tok, full.token_count)
    # Then: equality is eligible; no end token is truncated.
    assert result.training_status == "ready" and result.example == full.example


@pytest.mark.parametrize("num_proc", [1, 2])
def test_trainer_preserves_original_json_when_arrow_maps_nested_messages(row, tok, num_proc):
    # Given: heterogeneous message keys and both canonical and legacy rows.
    from train_sft import prepare_dataset

    original = copy.deepcopy(row)
    legacy = {"messages": row["messages"], "tools": row["tools"]}
    expected = [build_example(item, tok, 12288) for item in (row, legacy)]
    # When: the same dataset preparation entry point used by main runs Arrow mapping.
    dataset = prepare_dataset([row, legacy], tok, 12288, num_proc, "test")
    # Then: original hashes remain valid and both examples retain exact token labels.
    assert list(dataset) == expected
    assert row == original


def test_trainer_filters_held_rows_when_mapped_dataset_contains_no_targets(row, tok):
    # Given: a valid ready row plus an all-masked canonical row.
    from train_sft import prepare_dataset

    masked = copy.deepcopy(row)
    masked["assistant_supervision"][1].update(label="harmful", weight=0)
    expected = build_example(row, tok, 12288)
    # When: the production dataset boundary evaluates dispositions and filters rows.
    dataset = prepare_dataset([row, masked], tok, 12288, 1, "test")
    # Then: only the ready row reaches the training dataset.
    assert list(dataset) == [expected]
