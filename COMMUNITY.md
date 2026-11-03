# Community

SecLLM-Dataset is GitHub-first. This page tells you where to ask, report, or discuss, and what to
expect in return. Everyone in every channel follows the [Code of Conduct](CODE_OF_CONDUCT.md).

> GitHub is the source of record for data, bugs, and decisions; the Hugging Face Community tab covers questions about a specific Hugging Face revision; Discord is for chat only and never makes decisions; vulnerabilities and personal data are never posted publicly on any of them.

## Where to go

| I want to… | Go to |
|---|---|
| Ask how to use the dataset | [Discussions Q&A](https://github.com/Sparrow-Co-Ltd/SecLLM-Dataset/discussions/categories/q-a) |
| Report a wrong label, false positive, wrong trace, or bad patch | [Data error form](https://github.com/Sparrow-Co-Ltd/SecLLM-Dataset/issues/new?template=data-error.yml) |
| Suggest an idea or a new language | [Discussions Ideas](https://github.com/Sparrow-Co-Ltd/SecLLM-Dataset/discussions/categories/ideas) |
| Contribute docs, scripts, or community data | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Ask about a specific Hugging Face revision | The Community tab of the Hugging Face repository (see [Hugging Face](#hugging-face)) |
| Ask a general ML question | Out of scope here; use a general forum such as the Hugging Face Forum |
| Chat with other users | Discord (see [Discord](#discord)); chat only, not authoritative |
| Report a security vulnerability | Private Vulnerability Reporting, see [SECURITY.md](SECURITY.md) |
| Remove your code or personal data | [Removal request form](https://github.com/Sparrow-Co-Ltd/SecLLM-Dataset/issues/new?template=removal-request.yml) or the private contact in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md#how-to-report) |

The full matrix, with switch conditions and response targets, is in
[docs/community/CHANNEL_MATRIX.md](docs/community/CHANNEL_MATRIX.md).

## Channels

### Discussions Q&A

**Covers:** usage questions, loading and parsing the data, schema questions.
**Does not cover:** data errors (use the form), vulnerabilities, personal data.

### Discussions Ideas

**Covers:** proposals for new languages, fields, tooling, or documentation.
**Does not cover:** committed work. An idea becomes an issue only once scope is agreed.

### Discussions Announcements

**Covers:** releases, schedule changes, and outages of other channels. Maintainers post here.
**Does not cover:** questions. Replies are off-topic; start a Q&A thread instead.

### GitHub Issues

**Covers:** data errors, removal requests, community-data proposals, and work items promoted from discussions or chat.
**Does not cover:** usage questions, vulnerabilities in this repository's tooling.

### Pull requests

**Covers:** documentation, scripts, and community-track data, as described in [CONTRIBUTING.md](CONTRIBUTING.md).
**Does not cover:** changes to core data in `data/java/`; report those as data errors.

### Hugging Face

**Covers:** questions about a specific revision of the Hugging Face dataset or model: loading, the card, and the files on the Hub.
**Does not cover:** data errors and decisions (they go to GitHub), general ML questions, vulnerabilities, personal data.

Repositories:

- Model: [sparrow-co-ltd/qwen35-9b-secfix-grpo-step180](https://huggingface.co/sparrow-co-ltd/qwen35-9b-secfix-grpo-step180) ([Community tab](https://huggingface.co/sparrow-co-ltd/qwen35-9b-secfix-grpo-step180/discussions))
- Dataset: [TBD: HF dataset repo ID]

Rules for the Community tabs: [docs/community/HUGGINGFACE_COMMUNITY.md](docs/community/HUGGINGFACE_COMMUNITY.md).

### Discord

**Covers:** quick help between users, contributor chat, events.
**Does not cover:** official answers, decisions, bug reports, vulnerabilities, personal data.

Invite:

[TBD: Discord invite URL]

Rules: [docs/community/CHANNEL_POLICY.md](docs/community/CHANNEL_POLICY.md).
Launch plan: [docs/community/DISCORD_LAUNCH.md](docs/community/DISCORD_LAUNCH.md).

### Private reports

**Covers:** vulnerabilities in scripts or workflows, leaked secrets or personal data ([SECURITY.md](SECURITY.md)); conduct reports ([CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md#how-to-report)).
**Does not cover:** vulnerabilities inside `entireCode` samples; they are the dataset itself.

## Promotion to GitHub

Chat and Hugging Face threads are not a record. When a thread finds a data error, reaches a
decision, or answers a question that keeps coming back, a maintainer or moderator summarizes it on
GitHub (an issue, a Discussion, or a documentation pull request) and links both ways. Each topic has
one source of record; other channels hold only a link and a short context.

## Response expectations

- First triage within **2 business days** (Monday to Friday, Korean business hours, KST).
- Answers are best effort. This is not a support contract, and an answer is not guaranteed.
- Security reports follow the targets in [SECURITY.md](SECURITY.md).

## If a channel is down

If Discord or Hugging Face is unavailable, or maintainers are away, the notice goes to
[Discussions Announcements](https://github.com/Sparrow-Co-Ltd/SecLLM-Dataset/discussions/categories/announcements).
GitHub Discussions Announcements is the fallback announcement channel.

## More

- People and roles: [MAINTAINERS.md](MAINTAINERS.md), [GOVERNANCE.md](GOVERNANCE.md)
- Moderation: [docs/community/CHANNEL_POLICY.md](docs/community/CHANNEL_POLICY.md), [docs/community/MODERATOR_RUNBOOK.md](docs/community/MODERATOR_RUNBOOK.md)
