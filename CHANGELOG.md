# Changelog

All notable changes to this dataset are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).
Dataset versions use `MAJOR.MINOR.PATCH`:

- **MAJOR**: breaking schema changes (fields renamed, removed, or retyped)
- **MINOR**: records added or removed, or new languages or tracks, without breaking the schema
- **PATCH**: corrections to existing records or documentation

Each release lists the SHA-256 of its data files. Verify with `sha256sum data/java/input.jsonl`.

## [Unreleased]

### Added

- C dataset (`data/c/input.jsonl`): 1,794 records across 86 vulnerability types
- TypeScript dataset (`data/typescript/input.jsonl`): 137 records across 12 vulnerability types
  Excludes 1 records flagged by GitHub push protection for credential patterns.
- Python dataset (`data/python/input.jsonl`): 114 records across 21 vulnerability types
  Excludes 2 records flagged by GitHub push protection for credential patterns.
- Go dataset (`data/go/input.jsonl`): 222 records across 12 vulnerability types
- C++ dataset (`data/cpp/input.jsonl`): 2,479 records across 86 vulnerability types
- JavaScript dataset (`data/javascript/input.jsonl`): 332 records across 26 vulnerability types
  Excludes 3 records flagged by GitHub push protection for credential patterns.
- Project policies: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, GOVERNANCE, MAINTAINERS
- Datasheet (`DATASHEET.md`), English README with Korean translation (`README.ko.md`), `CITATION.cff`
- Validator (`scripts/validate.py`) and CI workflows
- Validator support for JavaScript, TypeScript, Go, Python, C, and C++ datasets,
  in addition to Java
- StarCoderData download scripts (`scripts/starcoderdata/`) that fetch the upstream parquet files and extract
  source files for SAST analysis
- Community data track (`data/community/`) for data produced with third-party tools
- Links to the published Hugging Face model (`sparrow-co-ltd/qwen35-9b-secfix-grpo-step180`), the project portal, and the K-OSSC projects
  listing in the README, COMMUNITY.md, the issue chooser, and `CITATION.cff`
- The current model, `sparrow-co-ltd/qwen35-9b-secfix-grpo-lead-step300-merged` (SFT then GRPO, merged full weights), in the README, COMMUNITY.md,
  the issue chooser, `CITATION.cff`, and the weekly HF KPIs; the step-180 adapter stays listed as the earlier model
- Weekly Hugging Face model KPIs (`collect_metrics.py --hf`, `metrics/hf-kpi.csv` on the `metrics` branch)
- Test inputs (`data/test/<lang>.jsonl`): 1,231 records in 7 files (Java 315, Python 149, C 210, C++ 287,
  JavaScript 120, Go 75, TypeScript 75). Same `input` schema as core records; `output` is always `[]`
- Validator support for `data/test/<lang>.jsonl` (`output` must be `[]`) and matching `--self-test` cases
- Core records that share a source file (`entireCode`) with a test record are removed: Python 31, C 69, C++ 73.
  Core records with the same vulnerability type and flagged line as a test record in another version of the same file
  are also removed: Go 1
- Core records keep only findings labeled true positive in a per-finding review (at least two independent LLM reviews,
  more on disagreement). Findings labeled policy or false positive, or without a review label, are removed:
  JavaScript 805, TypeScript 390, Go 767, Python 1,242, C 810, C++ 1,335. Test inputs are not filtered

### Changed

- NOTICE: English translation added; file-level license headers in `entireCode` take precedence
  (the upstream corpus filtered by repository-level license); removal requests go through an issue form

### Removed

- `hf/model-card.md` draft. The card published on Hugging Face is the authoritative model description
- Java (`data/java/input.jsonl`): 2 records (4,617 → 4,615) with the same vulnerability type and flagged line as a test
  record in another version of the same file
- Java (`data/java/input.jsonl`): 2,291 records (4,615 → 2,324, 101 → 88 vulnerability types) whose finding was labeled
  policy (1,733) or false positive (388), or has no review label (170)

### Deprecated

- Records whose `entireCode` carries a GPL-family license header (roughly 80 files, under 2%) are scheduled for
  removal in a future release. Until then, the license in each file's header governs that file.

## [1.0.0] - 2026-09-20

### Added

- Initial Java release: `data/java/input.jsonl`, 4,617 records, 101 vulnerability types
- SHA-256 `9a03c0a04b39814be14876ae9683f6f450c497a9712f11d9f3b8be593880af37` (56,475,076 bytes)

[Unreleased]: https://github.com/Sparrow-Co-Ltd/SecLLM-Dataset/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Sparrow-Co-Ltd/SecLLM-Dataset/releases/tag/v1.0.0
