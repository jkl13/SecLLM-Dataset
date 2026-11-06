# SecLLM-Dataset

[English](README.md) | 한국어

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/Sparrow-Co-Ltd/SecLLM-Dataset/badge)](https://scorecard.dev/viewer/?uri=github.com/Sparrow-Co-Ltd/SecLLM-Dataset)
[![Hugging Face model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-qwen35--9b--secfix--lead-yellow)](https://huggingface.co/sparrow-co-ltd/qwen35-9b-secfix-grpo-lead-step300-merged)
[![Portal](https://img.shields.io/badge/Portal-SecLLM-blue)](https://sparrow-co-ltd.github.io/SecLLM-Dataset-Portal/)

> **보안 취약점 탐지·설명·패치 LLM 학습을 위한 정적분석 기반 데이터셋**

SecLLM-Dataset은 정적 분석 도구(SAST)가 실제 오픈소스 코드에서 탐지한 보안 취약점을,
LLM 학습·평가에 사용할 수 있는 JSON Lines 형식으로 정리한 데이터셋입니다.

각 레코드는 단순한 `(코드, 라벨)` 쌍이 아니라 다음을 함께 제공합니다.

- 취약점이 발생한 **전체 소스 파일**과 **취약점 발생 라인**
- 취약점 유형에 대한 **한/영 설명**
- 같은 유형의 **취약한 예시 코드**와 **안전한 예시 코드** (한/영)
- 정적 분석기가 판단에 이르기까지의 **추적 경로(taint flow)** — source → branch → sink
- 취약점을 실제로 해결하는 **코드 수정 패치** — 패치 적용 후 SAST 재분석으로 해결이 확인된 수정만 수록

취약점 탐지뿐 아니라, 취약점 **설명 생성**, **패치 생성**,
**추론 경로 학습** 등 다양한 태스크에 활용할 수 있습니다.

제작 동기, 수집 과정, 한계, 책임 있는 사용에 대해서는 [데이터시트(영문)](DATASHEET.md)를 참고하세요.

## 모델, 포털, 소개처

- **모델:** Hugging Face의 [sparrow-co-ltd/qwen35-9b-secfix-grpo-lead-step300-merged](https://huggingface.co/sparrow-co-ltd/qwen35-9b-secfix-grpo-lead-step300-merged)는 SecLLM 프로젝트의 현재 모델입니다.
  `Qwen/Qwen3.5-9B`를 교사 에이전트 궤적으로 LoRA SFT한 뒤 GRPO로 학습하고, 두 단계를 전체 가중치로 병합했습니다.
  라이선스는 Apache-2.0입니다. 확인된 SAST 탐지 결과 하나를 받아 `read_code`로 소스 파일을 읽고, `edit_code`로
  수정한 뒤 `submit_fix`로 제출합니다. C, C++, Java, Python, JavaScript, Go, TypeScript를 다룹니다.
  취약점을 탐지하지는 않으며, 생성한 패치는 병합 전에 검토해야 합니다. 서빙 방법과 한계는 모델 카드가 기준입니다.
- **이전 모델:** [sparrow-co-ltd/qwen35-9b-secfix-grpo-step180](https://huggingface.co/sparrow-co-ltd/qwen35-9b-secfix-grpo-step180)는 GRPO 180스텝의 LoRA 어댑터로, `edit_src_code` 도구 하나를 쓰고
  소스 파일을 프롬프트로 받습니다.
- **프로젝트 포털:** <https://sparrow-co-ltd.github.io/SecLLM-Dataset-Portal/> (한국어·영어 소개)
- **소개처:** 한국오픈소스보안커뮤니티(K-OSSC) 회원·회원사 프로젝트 목록인 [K-OSSC projects](https://github.com/k-ossc/kossc-projects)에
  2026년 9월부터 등재되어 있습니다.

---

## 데이터셋 구성

```
SecLLM-Dataset/
└── data/
    ├── java/
    │   └── input.jsonl        # Java 취약점 + 해결 확인된 패치 2,324건 (약 25MB)
    ├── javascript/
    │   └── input.jsonl        # JavaScript 취약점 + 패치 332건 (약 3.3MB)
    ├── c/
    │   └── input.jsonl        # C 취약점 + 패치 1,794건 (약 20MB)
    ├── cpp/
    │   └── input.jsonl        # C++ 취약점 + 패치 2,479건 (약 24MB)
    ├── go/
    │   └── input.jsonl        # Go 취약점 + 패치 222건 (약 1.7MB)
    ├── python/
    │   └── input.jsonl        # Python 취약점 + 패치 114건 (약 1.4MB)
    ├── typescript/
    │   └── input.jsonl        # TypeScript 취약점 + 패치 137건 (약 1.0MB)
    ├── test/                  # 평가용 test 입력 1,231건 (7개 파일, 패치 없음)
    └── community/             # 커뮤니티 트랙: 서드파티 도구로 만든 데이터 (아래 참조)
```

파일 형식은 **JSON Lines** 입니다. 한 줄이 하나의 취약점 인스턴스(JSON 객체)에 해당하며,
각 레코드는 취약점 정보(`input`)와 해당 취약점을 해결하는 패치(`output`)의 쌍으로 구성됩니다.

`data/test/<언어>.jsonl` 은 평가용 test 입력입니다 (1,231건: Java 315, Python 149, C 210, C++ 287, JavaScript 120, Go 75, TypeScript 75).
`input` 은 core 레코드와 같지만 검증된 패치는 공개하지 않으므로 `output` 은 항상 `[]` 입니다.
test 소스 파일은 core 데이터에 들어 있지 않습니다. test 레코드와 `entireCode` 가 같은 core 레코드는 제외했습니다.
같은 파일의 다른 버전에서 test 레코드와 취약점 유형·검출 줄이 같은 core 레코드도 제외했습니다 (Java 2, Go 1).

core 레코드에는 **정탐(true positive) 검출만** 남겼습니다. 각 검출을 실제 소스 파일과 대조해 LLM 검토자가 2회 이상 독립적으로
판정했고, 판정이 엇갈리면 추가로 검토했습니다. 판정은 정탐(규칙이 설명하는 결함이 검출 줄에 실제로 존재), policy(구문은 있으나
이 코드에는 규칙의 우려가 해당하지 않음), 오탐(분석기가 코드를 잘못 읽음) 세 가지입니다. policy·오탐 판정 검출과 판정이 없는 검출은
제외했습니다 (Java 2,291, JavaScript 805, TypeScript 390, Go 767, Python 1,242, C 810, C++ 1,335). test 입력은 이 판정으로 거르지 않았습니다.

> 언어별로 디렉터리를 분리하며, Java, JavaScript, C, C++, Go, Python, TypeScript 데이터를 제공합니다.

## 데이터 등급

| 등급 | 경로 | 생성 주체 | 검증 |
| --- | --- | --- | --- |
| 코어 | `data/java/`, `data/javascript/`, `data/c/`, `data/cpp/`, `data/go/`, `data/python/`, `data/typescript/` | Sparrow SAST 탐지·주석 | 메인테이너가 Sparrow SAST로 패치를 재검증 |
| 커뮤니티 | `data/community/<언어>/` | 커뮤니티가 서드파티 도구(Semgrep, CodeQL 등)로 생성해 기여 | 기여자가 보증. CI는 스키마와 출처 정보만 검사 |

커뮤니티 레코드는 같은 `input`/`output` 스키마에 출처(도구, 규칙, 원본 저장소, 라이선스)를 담은
`meta` 객체가 필수로 추가됩니다. 자세한 내용은 [data/community/README.md](data/community/README.md)를 참고하세요.

---

## 데이터 통계

### 파일별

| 파일 | 레코드 수 | 형식 | 크기 |
| --- | --- | --- | --- |
| `data/java/input.jsonl` | 2,324 | input + output | 약 25 MB |
| `data/javascript/input.jsonl` | 332 | input + output | 약 3.3 MB (3,276,977 bytes) |
| `data/typescript/input.jsonl` | 137 | input + output | 약 1.0 MB (979,877 bytes) |
| `data/python/input.jsonl` | 114 | input + output | 약 1.4 MB (1,422,445 bytes) |
| `data/go/input.jsonl` | 222 | input + output | 약 1.7 MB (1,742,037 bytes) |
| `data/cpp/input.jsonl` | 2,479 | input + output | 약 24 MB (24,233,051 bytes) |
| `data/c/input.jsonl` | 1,794 | input + output | 약 20 MB (20,314,910 bytes) |
| `data/test/java.jsonl` | 315 | input only (`output` 은 `[]`) | 2,901,996 bytes |
| `data/test/python.jsonl` | 149 | input only (`output` 은 `[]`) | 1,548,686 bytes |
| `data/test/c.jsonl` | 210 | input only (`output` 은 `[]`) | 2,083,727 bytes |
| `data/test/cpp.jsonl` | 287 | input only (`output` 은 `[]`) | 2,444,046 bytes |
| `data/test/javascript.jsonl` | 120 | input only (`output` 은 `[]`) | 1,229,428 bytes |
| `data/test/go.jsonl` | 75 | input only (`output` 은 `[]`) | 1,324,727 bytes |
| `data/test/typescript.jsonl` | 75 | input only (`output` 은 `[]`) | 659,954 bytes |

### Java (`data/java/input.jsonl` 기준)

| 항목 | 값 |
| --- | --- |
| 레코드 수 | 2,324 |
| 언어 | Java (`.java`) |
| 고유 취약점 유형 | 88종 |
| 파일 크기 | 약 25 MB (25,386,476 bytes) |
| 코드 길이(라인) | 최소 9 / 중앙값 105 / 최대 2,929 |

`python scripts/validate.py --stats data/java/input.jsonl` 로 다시 계산할 수 있습니다.

### JavaScript (`data/javascript/input.jsonl` 기준)

| 항목 | 값 |
| --- | --- |
| 레코드 수 | 332 |
| 언어 | JavaScript (`.js`) |
| 고유 취약점 유형 | 26종 |
| 파일 크기 | 약 3.3 MB (3,276,977 bytes) |
| 코드 길이(라인) | 최소 9 / 중앙값 77 / 최대 2,728 |

`python scripts/validate.py --stats data/javascript/input.jsonl` 로 다시 계산할 수 있습니다.

---

### C++ (`data/cpp/input.jsonl` 기준)

| 항목 | 값 |
| --- | --- |
| 레코드 수 | 2,479 |
| 언어 | C++ (`.cpp`) |
| 고유 취약점 유형 | 86종 |
| 파일 크기 | 약 24 MB (24,233,051 bytes) |
| 코드 길이(라인) | 최소 9 / 중앙값 93 / 최대 2,992 |

`python scripts/validate.py --stats data/cpp/input.jsonl` 로 다시 계산할 수 있습니다.

---

### Go (`data/go/input.jsonl` 기준)

| 항목 | 값 |
| --- | --- |
| 레코드 수 | 222 |
| 언어 | Go (`.go`) |
| 고유 취약점 유형 | 12종 |
| 파일 크기 | 약 1.7 MB (1,742,037 bytes) |
| 코드 길이(라인) | 최소 14 / 중앙값 111.5 / 최대 1,902 |

`python scripts/validate.py --stats data/go/input.jsonl` 로 다시 계산할 수 있습니다.

---

### Python (`data/python/input.jsonl` 기준)

| 항목 | 값 |
| --- | --- |
| 레코드 수 | 114 |
| 언어 | Python (`.py`) |
| 고유 취약점 유형 | 21종 |
| 파일 크기 | 약 1.4 MB (1,422,445 bytes) |
| 코드 길이(라인) | 최소 14 / 중앙값 100.5 / 최대 2,802 |

`python scripts/validate.py --stats data/python/input.jsonl` 로 다시 계산할 수 있습니다.

---

### TypeScript (`data/typescript/input.jsonl` 기준)

| 항목 | 값 |
| --- | --- |
| 레코드 수 | 137 |
| 언어 | TypeScript (`.ts`) |
| 고유 취약점 유형 | 12종 |
| 파일 크기 | 약 1.0 MB (979,877 bytes) |
| 코드 길이(라인) | 최소 10 / 중앙값 87 / 최대 1,036 |

`python scripts/validate.py --stats data/typescript/input.jsonl` 로 다시 계산할 수 있습니다.

---

### C (`data/c/input.jsonl` 기준)

| 항목 | 값 |
| --- | --- |
| 레코드 수 | 1,794 |
| 언어 | C (`.c`) |
| 고유 취약점 유형 | 86종 |
| 파일 크기 | 약 20 MB (20,314,910 bytes) |
| 코드 길이(라인) | 최소 9 / 중앙값 126.5 / 최대 2,996 |

`python scripts/validate.py --stats data/c/input.jsonl` 로 다시 계산할 수 있습니다.

---

## 데이터 스키마

각 레코드는 최상위에 `input`(취약점 정보)과 `output`(패치) 2개 필드를 가집니다.

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `input` | object | 취약점 정보. 아래 14개 필드 참조 |
| `output` | array | 취약점을 해결하는 코드 수정 목록. 아래 참조 |

### `input` 필드

`input` 객체는 아래 14개 필드를 가집니다.

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `fileExtension` | string | 원본 파일 확장자 (예: `.java`) |
| `programmingLanguage` | string | 프로그래밍 언어 (예: `Java`) |
| `issueRisk` | string | 위험도. `매우 높음` / `높음` / `보통` / `낮음` / `매우 낮음` |
| `issueNameEn` | string | 취약점 유형명 (영문) |
| `issueNameKo` | string | 취약점 유형명 (한글) |
| `issueDescriptionEn` | string | 취약점 유형 설명 (영문) |
| `issueDescriptionKo` | string | 취약점 유형 설명 (한글) |
| `issueLineNumber` | int | 취약점이 보고된 라인 번호 (1-based, `entireCode` 기준) |
| `entireCode` | string | 취약점이 포함된 **전체 소스 파일 원문** |
| `dangerousExampleEn` | string | 해당 유형의 취약한 예시 코드 (영문 주석, 라인 번호 포함) |
| `dangerousExampleKo` | string | 해당 유형의 취약한 예시 코드 (한글 주석, 라인 번호 포함) |
| `safeExampleEn` | string | 해당 유형의 안전한 예시 코드 (영문 주석, 라인 번호 포함) |
| `safeExampleKo` | string | 해당 유형의 안전한 예시 코드 (한글 주석, 라인 번호 포함) |
| `contexts` | array | 정적 분석기의 추적 경로. 아래 참조 |

`dangerousExample*` / `safeExample*` 는 **사전에 정의된 예시**이며, `entireCode`에서 발췌한 코드가 아닙니다.

#### `contexts[]` 필드

`contexts`는 취약점 판정에 이르기까지의 데이터 흐름을 순서대로 담은 배열입니다.

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `lineNo` | int | 해당 이벤트가 발생한 라인 번호 (`entireCode` 기준) |
| `eventType` | string | `source` / `branch` / `sink` |
| `messageKey` | string | 분석기 내부 메시지 식별자 (예: `jfsyn.DIRECT_USE_OF_THREADS.defect`) |
| `message` | string | 사람이 읽을 수 있는 설명 (한글) |
| `params` | object | `message`에 삽입된 변수 값. 없으면 `{}` |

### `output[]` 필드

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `type` | string | `delete` / `add` |
| `startLine` | int | 대상 라인 번호 (1-based, **수정 전 `entireCode` 기준**) |
| `endLine` | int \| null | `delete` 시 삭제 구간의 끝 라인(포함). `add`는 `null` |
| `content` | string \| null | `add` 시 `startLine` 앞에 삽입할 코드(여러 줄 가능). `delete`는 `null` |

모든 라인 번호는 수정 전 원본 코드 기준입니다. `delete`는 `startLine`~`endLine` 라인을 삭제하고,
`add`는 원본의 `startLine` 위치 앞에 `content`를 삽입합니다(파일 끝을 넘는 경우 맨 뒤에 추가).

---

## 데이터 예시

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

## 사용 방법

### Python (표준 라이브러리)

```python
import json

records = []
with open("data/java/input.jsonl", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(len(records))                      # 2324
r = records[0]
inp = r["input"]
print(inp["issueNameKo"], inp["issueRisk"])  # 스레드의 직접 사용 높음

# 취약점이 발생한 라인 확인
lines = inp["entireCode"].split("\n")
print(lines[inp["issueLineNumber"] - 1])

# 패치(코드 수정 목록) 확인
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

> 파일이 약 25MB이므로 전체를 메모리에 올릴 수 있지만, 대용량 처리 시에는 스트리밍 방식을 권장합니다.

---

## 데이터 검증

`scripts/validate.py`는 스키마와 필드 간 규칙(라인 범위, `delete`/`add` 의미, 커뮤니티 출처 정보)을 검사합니다. Python 표준 라이브러리만 사용합니다.

```bash
python scripts/validate.py --self-test                     # 검증기 자체 점검
python scripts/validate.py                                 # data/**/*.jsonl 전체 검증
python scripts/validate.py --stats data/java/input.jsonl   # 통계 출력
```

## 원본 소스코드 코퍼스

[scripts/starcoderdata/](scripts/starcoderdata/README.md)는
[bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata)에서 언어별 parquet 파일을 내려받아
행마다 소스 파일 하나로 추출합니다(Sparrow SAST 분석에 쓰는 폴더 구조). [uv](https://docs.astral.sh/uv/)와
gated 데이터셋 접근 권한이 있는 HuggingFace 토큰이 필요합니다.

```bash
uv run scripts/starcoderdata/download_starcoderdata.py --language java --max-rows 5000
```

## 인용

인용 정보는 [CITATION.cff](CITATION.cff)에 있습니다(GitHub의 "Cite this repository" 버튼으로도 확인할 수 있습니다).

```bibtex
@misc{secllm_dataset_2026,
  title        = {SecLLM-Dataset: A Static-Analysis-Based Dataset for Vulnerability Detection, Explanation, and Repair with LLMs},
  author       = {{Sparrow Co., Ltd.}},
  year         = {2026},
  howpublished = {\url{https://github.com/Sparrow-Co-Ltd/SecLLM-Dataset}},
  note         = {Version 1.0.0}
}
```

## 기여하기

코어 데이터는 메인테이너가 Sparrow SAST로 재검증한 뒤에만 변경합니다. 데이터 오류 제보, 문서·스크립트 개선,
커뮤니티 트랙에 서드파티 도구 데이터 기여로 참여할 수 있습니다. 정책 문서는 영문으로 작성되어 있습니다.

- [CONTRIBUTING.md](CONTRIBUTING.md): 기여 방법 (커밋에 DCO 서명 필요)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md): 행동 강령
- [SECURITY.md](SECURITY.md): 보안 문제 신고 방법
- [GOVERNANCE.md](GOVERNANCE.md): 의사결정 방식
- [CHANGELOG.md](CHANGELOG.md): 릴리스 이력

## 커뮤니티

모든 채널에 대한 전체 안내는 [COMMUNITY.md](COMMUNITY.md)(영문)를 참고하세요.

- 사용 방법 질문: [Discussions Q&A](https://github.com/Sparrow-Co-Ltd/SecLLM-Dataset/discussions/categories/q-a)
- 데이터 오류(잘못된 라벨, 오탐, 잘못된 추적 경로, 잘못된 패치): [데이터 오류 신고 양식](https://github.com/Sparrow-Co-Ltd/SecLLM-Dataset/issues/new?template=data-error.yml)
- 보안 취약점: [SECURITY.md](SECURITY.md)의 절차를 따라 주세요 (공개 이슈로 올리지 마세요)
- 권리자의 삭제 요청: [삭제 요청 양식](https://github.com/Sparrow-Co-Ltd/SecLLM-Dataset/issues/new?template=removal-request.yml)
- 모델 관련 질문: [모델의 Community 탭](https://huggingface.co/sparrow-co-ltd/qwen35-9b-secfix-grpo-lead-step300-merged/discussions), [COMMUNITY.md의 Hugging Face 섹션](COMMUNITY.md#hugging-face) 참고
- 첫 응대는 영업일 기준 2일 이내입니다 (월~금, 한국 표준시). 답변은 최선을 다해 제공하며, 보장된 지원 계약은 아닙니다.

> GitHub은 데이터·버그·의사결정의 공식 기록소이며, Hugging Face Community 탭은 특정 Hugging Face 리비전에 대한 질문을 다루고, Discord는 채팅 전용으로 의사결정을 하지 않으며, 취약점과 개인정보는 어떤 채널에도 공개적으로 게시되지 않습니다.

## 라이선스

SecLLM-Dataset의 편집물과 주석 데이터는
[크리에이티브 커먼즈 저작자표시 4.0 국제 라이선스(CC BY 4.0)](LICENSE)를 따릅니다.

`entireCode` 필드에는 공개 오픈소스 프로젝트의 소스 파일 원문이 들어 있습니다. 이 코드는 CC BY 4.0의
적용 대상이 **아니며** 각 원저작물의 라이선스를 따릅니다. 정확한 적용 범위는 [NOTICE.md](NOTICE.md)를 참고하세요.

## 지원 사업 표기

본 데이터셋은 과학기술정보통신부·정보통신산업진흥원(NIPA)
「2026년도 오픈소스 AI·SW 개발·활용 지원사업(활용트랙)」의 지원으로 제작되었습니다.
