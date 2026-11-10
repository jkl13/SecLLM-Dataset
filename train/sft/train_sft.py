#!/usr/bin/env python3
"""Qwen3.5-9B LoRA SFT for a security-vulnerability-fixing tool-call agent.

Input: {train,val}.jsonl produced by train/sft/select/export_sft.py
       (three-tool trajectories with read_code / edit_code / submit_fix; function.arguments
        is a dict, every assistant turn carries one tool call, and the last message is the
        submit_fix call).
       Set the paths with --train-file / --val-file. The defaults are the selection output
       paths relative to the working directory.

TRL's assistant_only_loss is not used because of how the Qwen3.5 template behaves.
  - The template has no {% generation %} marker, so assistant_only_loss does not work.
  - Turn boundaries are therefore rendered here, and labels are masked through a
    char offset -> token index mapping.
  - Tool calls render as <function=...><parameter=...> XML, not JSON.
    arguments must be passed as a dict (the template iterates over `arguments|items`).
  - Assistant turns always render starting with `<think>\n\n</think>\n\n`.
    This matches add_generation_prompt=True + enable_thinking=False exactly, so
    inference must also be served with enable_thinking=False.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import JsonValue
from srft_masking import IGNORE, AnnotationError, evaluate_example
from srft_masking import build_example as build_example  # noqa: PLC0414 - public compatibility export

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase

# Reduces memory fragmentation caused by the wide spread in sequence lengths.
# Must be set before torch's first CUDA allocation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Load the credentials in .env (WANDB_API_KEY, HF_TOKEN) into the process environment.
# Python does not read .env on its own, so they are injected here. Values already set are kept.
# Uses the first .env found walking up from the script's location (train/sft/).
_env = next(
    (p / ".env" for p in Path(__file__).resolve().parents if (p / ".env").exists()),
    None,
)
if _env is not None:
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
os.environ.setdefault("WANDB_PROJECT", "qwen35-secfix-sft")

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_pt_utils import LengthGroupedSampler

# ---------------------------------------------------------------- data


def load_rows(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def prepare_dataset(
    rows: list[dict[str, JsonValue]], tok: PreTrainedTokenizerBase,
    max_length: int, num_proc: int, split: str,
) -> Dataset:
    """Keep original JSON intact until hash validation and token masking finish."""
    import datasets

    def tokenize_serialized(row):
        result = evaluate_example(json.loads(row["row_json"]), tok, max_length)
        if result.training_status == "invalid_annotations":
            raise AnnotationError(result.reason)
        return {**(result.example or {"input_ids": None, "labels": None, "length": 0}),
                "training_status": result.training_status}

    ds = datasets.Dataset.from_dict({"row_json": [json.dumps(row) for row in rows]})
    ds = ds.map(
        tokenize_serialized, remove_columns=ds.column_names,
        num_proc=num_proc, desc=f"tokenize:{split}",
    )
    print(f"[{split}] dispositions: {dict(Counter(ds['training_status']))}")
    ds = ds.filter(lambda row: row["input_ids"] is not None, num_proc=num_proc)
    ds = ds.remove_columns("training_status")
    n_tgt = sum(sum(token != IGNORE for token in labels) for labels in ds["labels"])
    print(f"[{split}] {len(ds)} rows, total tokens {sum(ds['length']):,}, target tokens {n_tgt:,}")
    return ds


class LengthGroupedTrainer(Trainer):
    """Length-grouped sampling, added here because transformers 5.x removed group_by_length.

    Sequence lengths vary widely (about 2.6k to 12k tokens), so random sampling wastes
    a lot of compute on padding.
    """

    def _get_train_sampler(self, train_dataset=None):
        ds = train_dataset if train_dataset is not None else self.train_dataset
        if ds is None or "length" not in getattr(ds, "column_names", []):
            return super()._get_train_sampler(train_dataset)
        return LengthGroupedSampler(
            batch_size=self.args.train_batch_size * self.args.gradient_accumulation_steps,
            lengths=list(ds["length"]),
        )


@dataclass
class PadCollator:
    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        n = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            pad = n - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["labels"].append(f["labels"] + [IGNORE] * pad)
            batch["attention_mask"].append([1] * len(f["input_ids"]) + [0] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


# ---------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--train-file", default="datas/sft/select/export/train.jsonl")
    p.add_argument("--val-file", default="datas/sft/select/export/val.jsonl")
    p.add_argument("--run-name", default=None, help="W&B run name. Generated from the settings when omitted")
    p.add_argument(
        "--data-tag",
        default="merged",
        help="Dataset identifier included in the run_name and adapter folder. "
        "The selection output is overwritten when it is rebuilt, so this tag records "
        "which data the run was trained on.",
    )
    p.add_argument("--output-dir", default=None, help="Checkpoint directory. Default: outputs/<run_name>")
    p.add_argument("--adapters-dir", default="adapters", help="Archive folder for final adapters")
    # Must match --max-length in select_candidates.py / export_sft.py
    # (the selection stage already drops trajectories longer than this).
    p.add_argument("--max-length", type=int, default=12288)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=-1, help="For smoke tests (-1 = disabled)")
    p.add_argument("--lr", type=float, default=1e-4)
    # The default bs=4 / grad-accum 4 is the setting for an 80GB GPU (with Liger,
    # max_length 12288; under 60GiB even for a batch of the longest sequences).
    # On 183GB-class GPUs we trained with bs=12 / grad-accum 2.
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.05)  # ms-swift's recommended value for Qwen3.5
    p.add_argument(
        "--no-liger",
        action="store_true",
        help="Disable Liger fused linear CE. With a 248k vocabulary, the fp32 logits of one "
             "8k sequence alone take 16GiB. Without it, even bs=1 uses 42GiB and bs=2 "
             "runs out of memory.",
    )
    p.add_argument("--num-proc", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.run_name is None:
        from datetime import datetime

        args.run_name = (
            f"qwen35-9b-secfix-{args.data_tag}-r{args.lora_r}"
            f"-lr{args.lr:g}-{datetime.now():%m%d-%H%M}"
        )
    if args.output_dir is None:
        args.output_dir = f"outputs/{args.run_name}"

    # Write the training log into the run folder automatically. The tee works at the fd
    # level, so all of stdout/stderr (including tqdm, HF logging and CUDA warnings) goes
    # to both the terminal and the file. At the end it is also copied to the adapter
    # folder, so it is kept with the run's other outputs.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.output_dir) / "train.log"
    import subprocess
    import sys

    _tee = subprocess.Popen(["tee", "-a", str(log_path)], stdin=subprocess.PIPE)
    sys.stdout.flush(); sys.stderr.flush()
    os.dup2(_tee.stdin.fileno(), 1)
    os.dup2(_tee.stdin.fileno(), 2)
    print(f"run: {args.run_name}\nlog: {log_path}")

    tok = AutoTokenizer.from_pretrained(args.model)

    train_ds = prepare_dataset(
        load_rows(args.train_file), tok, args.max_length, args.num_proc, "train",
    )
    eval_ds = (
        prepare_dataset(load_rows(args.val_file), tok, args.max_length, args.num_proc, "val")
        if Path(args.val_file).exists() else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",  # flash_attention_2 if flash-attn is installed
        device_map=None,
    )
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Language-model linear layers only; the vision tower is excluded because
        # this data does not use it.
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.enable_input_require_grads()  # needed for gradient checkpointing + LoRA

    # transformers 5.x has no warmup_ratio, so compute the total steps and pass warmup_steps.
    eff_batch = args.batch_size * args.grad_accum
    total_steps = (
        args.max_steps
        if args.max_steps > 0
        else max(1, int(len(train_ds) / eff_batch * args.epochs))
    )
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    print(f"{total_steps} total steps (effective batch {eff_batch}), {warmup_steps} warmup steps")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        use_liger_kernel=not args.no_liger,
        logging_steps=10,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=["wandb"],
        run_name=args.run_name,
        seed=args.seed,
    )

    trainer = LengthGroupedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PadCollator(tok.pad_token_id),
    )
    trainer.train()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"peak GPU memory: {peak:.1f} GiB")

    # load_best_model_at_end has loaded the lowest-eval_loss checkpoint.
    # Save it to the final adapter archive, kept apart from the checkpoints (outputs/).
    adapter_dir = Path(args.adapters_dir) / args.run_name
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    (adapter_dir / "training_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str)
    )
    data_stats = Path(args.train_file).parent / "stats.json"
    if data_stats.exists():
        (adapter_dir / "data_stats.json").write_text(data_stats.read_text())

    try:
        import wandb

        if wandb.run is not None:
            wandb.run.summary["peak_gpu_gib"] = round(peak, 1)
            wandb.run.summary["adapter_dir"] = str(adapter_dir)
    except ImportError as exc:
        print(
            f"warning: wandb is not installed; skipped writing the run summary ({exc})",
            file=sys.stderr,
        )
    print(f"Adapter saved: {adapter_dir}")

    # Copy the training log to the final archive (outputs/ is a cleanup target)
    import shutil

    sys.stdout.flush(); sys.stderr.flush()
    shutil.copy2(log_path, adapter_dir / "train.log")


if __name__ == "__main__":
    main()
