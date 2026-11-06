"""LLM-judge design package: prompt, diff renderer, and typed output contracts.

This is the subset used by ``train/sft/select``: typed contracts (``schemas``),
request loading (``corpus``), diff rendering (``diff_render``), the tree-sitter syntax
gate (``syntax_gate``), the Fix-Rate prompt blocks (``prompt``) and the
OpenAI-compatible client (``judge_client``).
"""
