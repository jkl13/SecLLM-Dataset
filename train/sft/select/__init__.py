"""Teacher-trajectory selection for SFT: one trajectory per input.

Stages (run from the repository root):

1. ``select_candidates.py`` (root ``.venv``) — deterministic filters and ranking,
   writes ``candidates.jsonl`` and marks inputs whose passing runs produced
   materially different patches as ``needs_judge``.
2. ``judge_select.py`` (``train/judge/.venv`` + ``PYTHONPATH=train/judge/src``) —
   listwise LLM comparison of the competing patches on the local judge server.
3. ``export_sft.py`` — merges both results into ``train.jsonl`` / ``val.jsonl``.
"""
