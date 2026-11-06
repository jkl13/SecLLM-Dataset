# train/ — SFT for a SAST fix-suggestion agent

Code that fine-tunes **Qwen3.5-9B** with LoRA so it fixes SAST findings as a tool-calling
agent, plus the pipeline that selects the supervised trajectories.

The agent works in a three-tool loop:

1. `read_code(offset, limit)` reads the working copy of the source file.
2. `edit_code(old_string, new_string)` applies an edit.
3. `submit_fix(desc)` submits the fix with a one-sentence Korean description (≤300 characters).

Training data are trajectories produced by a teacher model in that loop. Each finding is a
record shaped like the `input` field of `data/<lang>/input.jsonl` in this repository. A
trajectory is used only if its final patch passed a SAST re-scan.

## Layout

```
train/
├── pyproject.toml, uv.lock            # training environment (torch cu128, transformers, peft, liger-kernel)
├── .env.example                       # environment variables (copy to train/.env)
├── sft/
│   ├── train_sft.py                   # LoRA SFT
│   ├── srft_masking.py                # turn-boundary rendering + assistant-only label masking
│   ├── verify_template.py             # pre-training rendering/masking check
│   ├── smoke_generate.py              # post-training generation check (T1–T3)
│   ├── select/                        # one trajectory per finding
│   │   ├── select_candidates.py       # stage 1: deterministic filters and ranking
│   │   ├── judge_select.py            # stage 2: listwise LLM judge for competing patches
│   │   ├── export_sft.py              # stage 3: train/val export
│   │   ├── audit_unresolved.py        # dump judge cases without a decision for review
│   │   ├── judge_prompt.py, trajectory_features.py
│   └── tests/
└── judge/                             # judge environment + the judge modules stage 2 imports
    ├── pyproject.toml, uv.lock
    └── src/secllm_judge/              # schemas, corpus, diff_render, syntax_gate, prompt, judge_client
```

## Inputs (not included in this repository)

Every path is relative to the working directory the scripts run from. Each teacher set is a
directory `datas/sft/<dataset_id>/`:

| File | Contents |
|---|---|
| `sft_candidate_inputs.jsonl` | one request per line, finding fields under `userRequest`; line number = `idx` |
| `sft_candidate_metadata.jsonl` | `idx`, `language`, `code_hash` (SHA-256 of the original file), `issueNameKo`, … |
| `generations/<model>_runNN.jsonl` | one trajectory per `idx` per run: `messages`, `tools`, `trace`, `loop`, `response.codeModifications`, `terminal_cause`, and the SAST re-scan fields `sast_result` / `sast_verdict` / `sast_grader` |

Pass one `--dataset` per teacher set to stage 1. The selection output defaults to
`datas/sft/select/`; change it with `--out` / `--select-dir`. Nothing in the code is tied to
a particular teacher or judge model: the teacher is whatever produced the generation files,
and the judge is whichever model the `JUDGE_API_URL` server serves.

## Environments

```bash
uv sync --project train          # -> train/.venv        (stages 1 and 3, training, smoke test)
uv sync --project train/judge    # -> train/judge/.venv  (stage 2, tests)
```

Settings come from environment variables. Copy [`.env.example`](.env.example) to
`train/.env` and fill it in:

| Variable | Used by | Required |
|---|---|---|
| `WANDB_API_KEY` | `train_sft.py` | required unless `WANDB_MODE=offline` or `disabled` |
| `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE` | `train_sft.py` | optional (project default `qwen35-secfix-sft`) |
| `HF_TOKEN` | model/tokenizer downloads | optional (the base model is public) |
| `JUDGE_API_URL`, `JUDGE_MODEL` | `judge_select.py` (stage 2) | required for stage 2 |
| `JUDGE_API_KEY` | `judge_select.py` (stage 2) | optional (default `EMPTY`, accepted by local vLLM) |

`train_sft.py` and `judge_select.py` read the first `.env` found walking up from their own
directory. Already-exported environment variables take precedence, and `judge_select.py`'s
`--judge-url` / `--judge-model` / `--api-key` flags override both. The other scripts do not
read `.env`; export the variables first if they need them (`set -a; . train/.env; set +a`).
`.env` is git-ignored.

## Pipeline

```bash
SEL=datas/sft/select

# 1. Deterministic selection -> $SEL/candidates.jsonl, $SEL/selection_report.json
train/.venv/bin/python train/sft/select/select_candidates.py \
    --dataset datas/sft/<teacher_set_a> --dataset datas/sft/<teacher_set_b> --out $SEL

# 2. Judge only the findings whose SAST-passing runs produced different patches.
#    Needs an OpenAI-compatible server (vLLM) serving the judge model, set by
#    JUDGE_API_URL / JUDGE_MODEL in train/.env; resumable.
PYTHONPATH=train/judge/src train/judge/.venv/bin/python \
    train/sft/select/judge_select.py --select-dir $SEL --profile thinking

# 3. Export one trajectory per finding -> $SEL/export/{train,val}.jsonl, stats.json
train/.venv/bin/python train/sft/select/export_sft.py --select-dir $SEL \
    --decisions $SEL/judging/decisions_<model_tag>_thinking.jsonl --all-rejected keep
# (<model_tag> = judge model name with characters outside [A-Za-z0-9._-] replaced by "_";
#  stage 2 prints the exact path on start)

# 4. Check rendering and masking (every row must be `ready`)
train/.venv/bin/python train/sft/verify_template.py --data $SEL/export/train.jsonl

# 5. Train (run name: qwen35-9b-secfix-<data_tag>-r<r>-lr<lr>-<MMDD-HHMM>)
CUDA_VISIBLE_DEVICES=0 train/.venv/bin/python train/sft/train_sft.py --data-tag <tag> \
    --train-file $SEL/export/train.jsonl --val-file $SEL/export/val.jsonl

# 6. Smoke test -> adapters/<run_name>/smoke_test.json
train/.venv/bin/python train/sft/smoke_generate.py --adapter adapters/<run_name> \
    --val-file $SEL/export/val.jsonl
```

## Selection rules

- **Hard filter** (per trajectory):
  - the SAST re-scan passed and the harness ended with `pass`;
  - no edit errors, fuzzy matches, nudges, tool error results or invalid calls;
  - exactly one verifier attempt;
  - at least one added line;
  - the final patch equals the last verifier candidate;
  - the rendered length is within `--max-length`.
- **Ranking** (per finding):
  1. trajectories whose first `read_code` follows the prompt's pagination rule;
  2. then the patch most runs agree on;
  3. then fewer turns, fewer cumulative input tokens, fewer patch lines, a patch that touches a SAST trace line, and a seeded hash.
- **Judge**: competing patches are shown under shuffled labels A–D, with run numbers and agent text hidden.
  - Two votes are taken in reversed order, and a third if they disagree.
  - The winner is the best-ranked candidate the judge marks both `not_redetected` and `compiles`.
- **Export**: SAST pass is the acceptance criterion, so the judge only chooses and never removes a finding.
  - No-majority, judge-error and all-rejected cases use the stage-1 pick (`--unresolved fallback --all-rejected keep`).
  - The trailing verifier `PASS` tool message is removed, so every row ends with the `submit_fix` call.
  - 3% of findings go to validation, grouped by original-code hash.

## Constraints to keep when editing

- **Masking**: Qwen3.5's chat template has no `{% generation %}` marker. `srft_masking.py` renders turn boundaries itself and labels only assistant tokens. A broken prefix property is reported as `invalid_annotations`. Never suppress it.
- **Tool-call format**: tool calls render as `<function=...><parameter=...>` XML, so `arguments` must be dicts.
- **Thinking mode**: assistant turns render with an empty `<think>` block, and training masks match `enable_thinking=False`. Inference and serving must use `enable_thinking=False` too.
- **Length**: keep `--max-length` equal in `select_candidates.py`, `export_sft.py` and `train_sft.py`. The default is 12288.
- **Liger**: the Liger kernel is required. With a 248k vocabulary, disabling it costs about 42 GiB at batch size 1.

## Tests

```bash
PYTHONPATH=train/judge/src:train/sft/select:train/sft \
    train/judge/.venv/bin/python -m pytest train/sft/tests/test_sft_select.py -q
# needs torch/transformers and the cached Qwen3.5-9B tokenizer; pytest is not in the training lock
cd train/sft && uv run --project .. --with pytest python -m pytest tests/test_srft_masking.py -q
```

## License

The license for the code under `train/` has not been decided yet. The repository-level
[LICENSE](../LICENSE) (CC BY 4.0) was written for the dataset.
