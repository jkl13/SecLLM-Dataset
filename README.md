# SecLLM-Dataset

English | [한국어](README.ko.md)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/Sparrow-Co-Ltd/SecLLM-Dataset/badge)](https://scorecard.dev/viewer/?uri=github.com/Sparrow-Co-Ltd/SecLLM-Dataset)
[![Hugging Face model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-qwen35--9b--secfix--lead-yellow)](https://huggingface.co/sparrow-co-ltd/qwen35-9b-secfix-grpo-lead-step300-merged)
[![Portal](https://img.shields.io/badge/Portal-SecLLM-blue)](https://sparrow-co-ltd.github.io/SecLLM-Dataset-Portal/)

> **A static-analysis-based dataset for training LLMs to detect, explain, and patch security vulnerabilities**

SecLLM-Dataset collects security vulnerabilities that a static application security testing (SAST) tool
found in real open-source code, packaged as JSON Lines for LLM training and evaluation.

Each record is more than a `(code, label)` pair. It provides:

- The **entire source file** where the vulnerability occurs and the **vulnerable line**
- A **Korean/English description** of the vulnerability type
- A **vulnerable example** and a **safe example** of the same type (Korean/English)
- The **taint flow** the analyzer followed to reach its verdict: source → branch → sink
- A **code patch** that actually fixes the vulnerability. Only patches confirmed by re-running SAST after applying them are included

Besides detection, the data supports tasks such as **explanation generation**, **patch generation**,
and **reasoning-path learning**.

For motivation, collection process, limitations, and responsible use, see the [datasheet](DATASHEET.md).

## Model, portal, and listings

- **Model:** [sparrow-co-ltd/qwen35-9b-secfix-grpo-lead-step300-merged](https://huggingface.co/sparrow-co-ltd/qwen35-9b-secfix-grpo-lead-step300-merged) on Hugging Face, the SecLLM project's current model.
  `Qwen/Qwen3.5-9B` was trained with LoRA SFT on teacher agent trajectories, then with GRPO, and both stages are
  merged into full weights under Apache-2.0. Given one confirmed SAST finding, the agent reads the source file with
  `read_code`, patches it with `edit_code`, and submits the fix with `submit_fix`. It covers C, C++, Java, Python,
  JavaScript, Go, and TypeScript. It does not detect vulnerabilities, and its patches need review before merging.
  The model card is the authoritative description, including how to serve it and its limitations.
- **Earlier model:** [sparrow-co-ltd/qwen35-9b-secfix-grpo-step180](https://huggingface.co/sparrow-co-ltd/qwen35-9b-secfix-grpo-step180), a LoRA adapter (GRPO step 180) that uses a single
  `edit_src_code` tool and gets the source file in the prompt
- **Project portal:** <https://sparrow-co-ltd.github.io/SecLLM-Dataset-Portal/> (Korean and English overview)
- **Listed in:** [K-OSSC projects](https://github.com/k-ossc/kossc-projects), the curated list of open source projects by K-OSSC members and member companies
  (Korea Open Source Security Community, 한국오픈소스보안커뮤니티), since 2026-09

---

## Dataset layout

```
SecLLM-Dataset/
└── data/
    ├── java/
    │   └── input.jsonl        # Java vulnerabilities + verified patches, 2,324 records (~25 MB)
    ├── javascript/
    │   └── input.jsonl        # JavaScript vulnerabilities + patches, 332 records (~3.3 MB)
    ├── c/
    │   └── input.jsonl        # C vulnerabilities + patches, 1,794 records (~20 MB)
    ├── cpp/
    │   └── input.jsonl        # C++ vulnerabilities + patches, 2,479 records (~24 MB)
    ├── go/
    │   └── input.jsonl        # Go vulnerabilities + patches, 222 records (~1.7 MB)
    ├── python/
    │   └── input.jsonl        # Python vulnerabilities + patches, 114 records (~1.4 MB)
    ├── typescript/
    │   └── input.jsonl        # TypeScript vulnerabilities + patches, 137 records (~1.0 MB)
    ├── test/                  # Test inputs, 1,231 records in 7 files (no patches)
    └── community/             # Community track: data from third-party tools (see below)
```

Files are **JSON Lines**: one line is one vulnerability instance (a JSON object).
Each record pairs vulnerability information (`input`) with a patch that fixes it (`output`).

> Data is split by language. Java, JavaScript, C, C++, Go, Python and TypeScript datasets are available.

`data/test/<lang>.jsonl` holds test inputs (1,231 records: Java 315, Python 149, C 210, C++ 287,
JavaScript 120, Go 75, TypeScript 75). A test record has the same `input` as a core record, but no verified
patch is published, so `output` is always `[]`.
No test source file appears in the core data: core records sharing an `entireCode` with a test record are removed.
Core records with the same vulnerability type and flagged line as a test record in another version of the same file
are also removed (Java 2, Go 1).

Core records keep only **true-positive findings**. Each finding was reviewed against the real source file by at least two
independent LLM reviewers, with additional reviews where they disagreed, and labeled true positive (the defect the
rule describes really exists at the reported line), policy (the construct is there but the rule's concern does not apply
to this code), or false positive (the analyzer misread the code). Findings labeled policy or false positive, and
findings without a review label, are removed (Java 2,291, JavaScript 805, TypeScript 390, Go 767, Python 1,242,
C 810, C++ 1,335). Test inputs are not filtered by this review.

## Data tiers

| Tier | Path | Produced by | Verification |
| --- | --- | --- | --- |
| Core | `data/java/`, `data/javascript/`, `data/c/`, `data/cpp/`, `data/go/`, `data/python/`, `data/typescript/` | Sparrow SAST detection and annotation | Patches re-verified by the maintainers with Sparrow SAST |
| Community | `data/community/<lang>/` | Third-party tools (e.g. Semgrep, CodeQL), contributed by the community | Attested by the contributor; CI checks schema and provenance only |

Community records use the same `input`/`output` schema plus a required `meta` object with provenance
(tool, rule, source repository, license). See [data/community/README.md](data/community/README.md).

---

## Statistics

### By file

| File | Records | Format | Size |
| --- | --- | --- | --- |
| `data/java/input.jsonl` | 2,324 | input + output | ~25 MB |
| `data/javascript/input.jsonl` | 332 | input + output | ~3.3 MB (3,276,977 bytes) |
| `data/typescript/input.jsonl` | 137 | input + output | ~1.0 MB (979,877 bytes) |
| `data/python/input.jsonl` | 114 | input + output | ~1.4 MB (1,422,445 bytes) |
| `data/go/input.jsonl` | 222 | input + output | ~1.7 MB (1,742,037 bytes) |
| `data/cpp/input.jsonl` | 2,479 | input + output | ~24 MB (24,233,051 bytes) |
| `data/c/input.jsonl` | 1,794 | input + output | ~20 MB (20,314,910 bytes) |
| `data/test/java.jsonl` | 315 | input only (`output` is `[]`) | 2,901,996 bytes |
| `data/test/python.jsonl` | 149 | input only (`output` is `[]`) | 1,548,686 bytes |
| `data/test/c.jsonl` | 210 | input only (`output` is `[]`) | 2,083,727 bytes |
| `data/test/cpp.jsonl` | 287 | input only (`output` is `[]`) | 2,444,046 bytes |
| `data/test/javascript.jsonl` | 120 | input only (`output` is `[]`) | 1,229,428 bytes |
| `data/test/go.jsonl` | 75 | input only (`output` is `[]`) | 1,324,727 bytes |
| `data/test/typescript.jsonl` | 75 | input only (`output` is `[]`) | 659,954 bytes |

### Java (`data/java/input.jsonl`)

| Item | Value |
| --- | --- |
| Records | 2,324 |
| Language | Java (`.java`) |
| Unique vulnerability types | 88 |
| File size | ~25 MB (25,386,476 bytes) |
| Code length (lines) | min 9 / median 105 / max 2,929 |

Recompute with `python scripts/validate.py --stats data/java/input.jsonl`.

### JavaScript (`data/javascript/input.jsonl`)

| Item | Value |
| --- | --- |
| Records | 332 |
| Language | JavaScript (`.js`) |
| Unique vulnerability types | 26 |
| File size | ~3.3 MB (3,276,977 bytes) |
| Code length (lines) | min 9 / median 77 / max 2,728 |

Recompute with `python scripts/validate.py --stats data/javascript/input.jsonl`.

---

### C++ (`data/cpp/input.jsonl`)

| Item | Value |
| --- | --- |
| Records | 2,479 |
| Language | C++ (`.cpp`) |
| Unique vulnerability types | 86 |
| File size | ~24 MB (24,233,051 bytes) |
| Code length (lines) | min 9 / median 93 / max 2,992 |

Recompute with `python scripts/validate.py --stats data/cpp/input.jsonl`.

---

### Go (`data/go/input.jsonl`)

| Item | Value |
| --- | --- |
| Records | 222 |
| Language | Go (`.go`) |
| Unique vulnerability types | 12 |
| File size | ~1.7 MB (1,742,037 bytes) |
| Code length (lines) | min 14 / median 111.5 / max 1,902 |

Recompute with `python scripts/validate.py --stats data/go/input.jsonl`.

---

### Python (`data/python/input.jsonl`)

| Item | Value |
| --- | --- |
| Records | 114 |
| Language | Python (`.py`) |
| Unique vulnerability types | 21 |
| File size | ~1.4 MB (1,422,445 bytes) |
| Code length (lines) | min 14 / median 100.5 / max 2,802 |

Recompute with `python scripts/validate.py --stats data/python/input.jsonl`.

---

### TypeScript (`data/typescript/input.jsonl`)

| Item | Value |
| --- | --- |
| Records | 137 |
| Language | TypeScript (`.ts`) |
| Unique vulnerability types | 12 |
| File size | ~1.0 MB (979,877 bytes) |
| Code length (lines) | min 10 / median 87 / max 1,036 |

Recompute with `python scripts/validate.py --stats data/typescript/input.jsonl`.

---

### C (`data/c/input.jsonl`)

| Item | Value |
| --- | --- |
| Records | 1,794 |
| Language | C (`.c`) |
| Unique vulnerability types | 86 |
| File size | ~20 MB (20,314,910 bytes) |
| Code length (lines) | min 9 / median 126.5 / max 2,996 |

Recompute with `python scripts/validate.py --stats data/c/input.jsonl`.

---

## Schema

Each record has two top-level fields: `input` (vulnerability information) and `output` (patch).

| Field | Type | Description |
| --- | --- | --- |
| `input` | object | Vulnerability information. See the 14 fields below |
| `output` | array | Code modifications that fix the vulnerability. See below |

### `input` fields

| Field | Type | Description |
| --- | --- | --- |
| `fileExtension` | string | Source file extension (e.g. `.java`) |
| `programmingLanguage` | string | Programming language (e.g. `Java`) |
| `issueRisk` | string | Risk level: `매우 높음` (very high) / `높음` (high) / `보통` (medium) / `낮음` (low) / `매우 낮음` (very low) |
| `issueNameEn` | string | Vulnerability type name (English) |
| `issueNameKo` | string | Vulnerability type name (Korean) |
| `issueDescriptionEn` | string | Vulnerability type description (English) |
| `issueDescriptionKo` | string | Vulnerability type description (Korean) |
| `issueLineNumber` | int | Reported line number (1-based, relative to `entireCode`) |
| `entireCode` | string | **Full original source file** containing the vulnerability |
| `dangerousExampleEn` | string | Vulnerable example for this type (English comments, with line numbers) |
| `dangerousExampleKo` | string | Vulnerable example for this type (Korean comments, with line numbers) |
| `safeExampleEn` | string | Safe example for this type (English comments, with line numbers) |
| `safeExampleKo` | string | Safe example for this type (Korean comments, with line numbers) |
| `contexts` | array | Analyzer trace. See below |

`dangerousExample*` / `safeExample*` are **predefined examples** per vulnerability type, not excerpts from `entireCode`.

#### `contexts[]` fields

`contexts` lists, in order, the data-flow events that led to the finding.

| Field | Type | Description |
| --- | --- | --- |
| `lineNo` | int | Line where the event occurs (relative to `entireCode`) |
| `eventType` | string | `source` / `branch` / `sink` |
| `messageKey` | string | Analyzer message identifier (e.g. `jfsyn.DIRECT_USE_OF_THREADS.defect`) |
| `message` | string | Human-readable explanation (Korean) |
| `params` | object | Values substituted into `message`; `{}` if none |

### `output[]` fields

| Field | Type | Description |
| --- | --- | --- |
| `type` | string | `delete` / `add` |
| `startLine` | int | Target line (1-based, **relative to the original `entireCode`**) |
| `endLine` | int \| null | For `delete`, last deleted line (inclusive). `null` for `add` |
| `content` | string \| null | For `add`, code inserted before `startLine` (may span lines). `null` for `delete` |

All line numbers refer to the original code before modification. `delete` removes lines `startLine` through `endLine`;
`add` inserts `content` before line `startLine` of the original (appended at the end if past the last line).

---

## Example

```json
{
  "input": {
    "fileExtension": ".java",
    "programmingLanguage": "Java",
    "issueRisk": "높음",
    "issueNameEn": "Direct Use of Threads",
    "issueNameKo": "스레드의 직접 사용",
    "issueDescriptionEn": "The Direct Use of Threads checker finds instances of a J2EE Web application directly using threads. ...",
    "issueDescriptionKo": "스레드의 직접 사용 체커는 J2EE 웹 애플리케이션에서 직접적으로 스레드를 사용하는 경우를 검출합니다. ...",
    "issueLineNumber": 23,
    "entireCode": "package io.opentracing.contrib.specialagent.test.servlet.jetty;\n\nimport java.io.PrintWriter;\n...",
    "dangerousExampleEn": "1. public class U383 extends HttpServlet {\n2.   protected void doGet(...)\n...",
    "dangerousExampleKo": "1. public class U383 extends HttpServlet {\n2.   protected void doGet(...)\n...",
    "safeExampleEn": "1. public class S383 extends HttpServlet {\n...",
    "safeExampleKo": "1. public class S383 extends HttpServlet {\n...",
    "contexts": [
      {
        "lineNo": 23,
        "eventType": "sink",
        "messageKey": "jfsyn.DIRECT_USE_OF_THREADS.defect",
        "message": "Thread 를 직접 생성하거나 사용하지 않도록 합니다.",
        "params": {}
      }
    ]
  },
  "output": [
    {
      "type": "delete",
      "startLine": 23,
      "endLine": 40,
      "content": null
    },
    {
      "type": "add",
      "startLine": 23,
      "endLine": null,
      "content": "    asyncContext.start(new Runnable() {\n      @Override\n      public void run() {\n..."
    }
  ]
}
```

---

## Usage

### Python (standard library)

```python
import json

records = []
with open("data/java/input.jsonl", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(len(records))                      # 2324
r = records[0]
inp = r["input"]
print(inp["issueNameEn"], inp["issueRisk"])  # Direct Use of Threads 높음

# Show the vulnerable line
lines = inp["entireCode"].split("\n")
print(lines[inp["issueLineNumber"] - 1])

# Show the patch (code modifications)
for mod in r["output"]:
    print(mod["type"], mod["startLine"], mod["endLine"])
```

### pandas

```python
import pandas as pd

df = pd.read_json("data/java/input.jsonl", lines=True)
inputs = pd.json_normalize(df["input"])
print(inputs["issueNameEn"].value_counts())
```

> The file is about 25 MB and fits in memory, but streaming is recommended for large-scale processing.

---

## Validation

`scripts/validate.py` checks the schema and cross-field rules (line ranges, `delete`/`add` semantics, community provenance). It needs only the Python standard library.

```bash
python scripts/validate.py --self-test                     # validator checks itself
python scripts/validate.py                                 # validate every data/**/*.jsonl
python scripts/validate.py --stats data/java/input.jsonl   # print statistics
```

## Upstream source corpus

[scripts/starcoderdata/](scripts/starcoderdata/README.md) downloads the
[bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata) parquet files for a language and
extracts one source file per row, in the layout used to run Sparrow SAST. It needs [uv](https://docs.astral.sh/uv/)
and a HuggingFace token with access to the gated dataset.

```bash
uv run scripts/starcoderdata/download_starcoderdata.py --language java --max-rows 5000
```

## Citation

Citation metadata is in [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this repository" button).

```bibtex
@misc{secllm_dataset_2026,
  title        = {SecLLM-Dataset: A Static-Analysis-Based Dataset for Vulnerability Detection, Explanation, and Repair with LLMs},
  author       = {{Sparrow Co., Ltd.}},
  year         = {2026},
  howpublished = {\url{https://github.com/Sparrow-Co-Ltd/SecLLM-Dataset}},
  note         = {Version 1.0.0}
}
```

## Contributing

Core data is changed only by the maintainers after re-verification with Sparrow SAST. You can help by reporting
data errors, sending documentation or script improvements, or contributing third-party-tool data to the community track.

- [CONTRIBUTING.md](CONTRIBUTING.md): how to contribute (commits must be signed off under the DCO)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md): community standards
- [SECURITY.md](SECURITY.md): how to report security issues
- [GOVERNANCE.md](GOVERNANCE.md): how decisions are made
- [CHANGELOG.md](CHANGELOG.md): release history

## Community

See [COMMUNITY.md](COMMUNITY.md) for the full guide to every channel.

- Usage questions: [Discussions Q&A](https://github.com/Sparrow-Co-Ltd/SecLLM-Dataset/discussions/categories/q-a)
- Data errors (wrong label, false positive, wrong trace, bad patch): [data error form](https://github.com/Sparrow-Co-Ltd/SecLLM-Dataset/issues/new?template=data-error.yml)
- Security vulnerabilities: follow [SECURITY.md](SECURITY.md) (do not open a public issue)
- Removal requests from rights holders: [removal request form](https://github.com/Sparrow-Co-Ltd/SecLLM-Dataset/issues/new?template=removal-request.yml)
- Model questions: the [Community tab of the model](https://huggingface.co/sparrow-co-ltd/qwen35-9b-secfix-grpo-lead-step300-merged/discussions); see [Hugging Face in COMMUNITY.md](COMMUNITY.md#hugging-face)
- First triage within 2 business days (Monday-Friday, KST). Answers are best effort, not a support contract.

> GitHub is the source of record for data, bugs, and decisions; the Hugging Face Community tab covers questions about a specific Hugging Face revision; Discord is for chat only and never makes decisions; vulnerabilities and personal data are never posted publicly on any of them.

## License

The compilation and annotations of SecLLM-Dataset are licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE).

The `entireCode` field contains original source files from public open-source projects. That code is **not**
covered by CC BY 4.0 and remains under its original licenses. See [NOTICE.md](NOTICE.md) for the exact scope.

## Acknowledgments

This dataset was produced with support from the Ministry of Science and ICT (MSIT) and the
National IT Industry Promotion Agency (NIPA) of Korea under the
"2026 Open Source AI/SW Development and Utilization Support Program (Utilization Track)".
