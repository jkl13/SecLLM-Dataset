# Metrics

How SecLLM-Dataset measures its community and project health, and how the numbers are reviewed.
Metrics exist to allocate maintainer time and improve the project, never to rank or monitor individuals.

## Where the data lives

| File | Contents | Updated by |
|---|---|---|
| [`github-kpi.csv` on the `metrics` branch](https://github.com/Sparrow-Co-Ltd/SecLLM-Dataset/blob/metrics/metrics/github-kpi.csv) | One row per week of GitHub KPIs (columns below) | **Automated.** The [`metrics` workflow](../.github/workflows/metrics.yml) runs [`collect_metrics.py`](../scripts/collect_metrics.py) every Monday 00:00 UTC and appends a row through the GitHub Contents API. |
| [`hf-kpi.csv` on the `metrics` branch](https://github.com/Sparrow-Co-Ltd/SecLLM-Dataset/blob/metrics/metrics/hf-kpi.csv) | One row per week for the Hugging Face model: `date`, `model`, `revision`, `downloads_30d`, `likes` | **Automated.** Same workflow, `collect_metrics.py --hf`, from the public Hub API. `downloads_30d` is the Hub's rolling 30-day count: report the month's last row, never sum rows. |
| [`COMMUNITY_KPI.csv`](COMMUNITY_KPI.csv) | One row per month of community KPIs | **Manual**, during the monthly review. Automated columns are copied from `github-kpi.csv`; the rest are counted by hand. |
| `metrics/reviews/YYYY-MM.md` | Minutes of each monthly review | **Manual**, via pull request after each review. |

`main` only accepts changes through reviewed pull requests, so the weekly rows are kept on the separate `metrics` branch.
That branch holds only the KPI CSVs (`github-kpi.csv`, `hf-kpi.csv`) and is protected against deletion and force-push, so its commit history is the audit trail.

The 2026-09 row of `COMMUNITY_KPI.csv` is the baseline, taken from the first `collect_metrics.py` run (2026-09-27, 7-day window).
A blank cell means "not measured"; it never means zero.

Run it locally (read-only):

```sh
python scripts/collect_metrics.py --self-test
GITHUB_TOKEN=$(gh auth token) python scripts/collect_metrics.py            # print one row
GITHUB_TOKEN=$(gh auth token) python scripts/collect_metrics.py --out k.csv # append to a local file
```

## Weekly GitHub KPIs (automated)

Source: GitHub REST API (`/repos/{repo}`, `/issues?state=all`, issue comments, PR reviews). Cadence: weekly. The window is the 7 days before the run.

| Column | Formula |
|---|---|
| `date` | Run date (UTC) |
| `window_days` | Length of the window (default 7) |
| `stars`, `forks` | Current totals |
| `watchers` | Current subscribers (people watching the repository) |
| `open_issues` | Open issues plus open pull requests, as GitHub reports it |
| `issues_opened`, `issues_closed` | Issues (not PRs) created / closed in the window |
| `prs_opened`, `prs_merged` | Pull requests created / merged in the window |
| `median_first_human_response_hours` | For issues and PRs created in the window: hours from creation to the first comment or review by someone who is neither the author nor a bot (account type `Bot` or login ending in `[bot]`). Median over the items that got a response; blank if none did. |
| `first_time_contributors` | Human PR authors active in the window whose first PR ever falls in the window |
| `repeat_contributors` | Human PR authors active in the window whose PRs span 2 or more calendar months |

## Monthly community KPIs (`COMMUNITY_KPI.csv`)

| Column | Definition and formula | Source | Automated? |
|---|---|---|---|
| `new_members` | New members of the chat community in the month | Chat platform admin view | Manual |
| `active_chatters` | People who posted in the chat community in the month | Chat platform admin view | Manual |
| `first_time_contributors` | People whose first contribution (PR, docs, triage) falls in the month | `collect_metrics.py --days` recomputed to cover the month, for PRs; docs/triage by hand | Partly |
| `repeat_contributors` | Human PR authors active in the month whose PRs span 2 or more calendar months | `collect_metrics.py --days` recomputed to cover the month | Partly |
| `questions_received` | Questions opened in the month on any channel | Discussions Q&A, issues, chat, Hugging Face | Manual |
| `questions_answered` | Questions answered or closed in the month; closing without an answer does not count | Same as above | Manual |
| `median_first_human_response_hours` | Median time from a question to the first non-bot, non-author response; report business-hours and weekend cases separately | `github-kpi.csv` for GitHub; other channels by hand | Partly |
| `chat_to_github_conversions` | Chat or forum threads promoted to a Discussion or issue | Links in the promoted items | Manual |
| `issues_opened`, `prs_opened`, `prs_merged` | Sum of the weekly rows for the month | `github-kpi.csv` | Automated |
| `moderation_incidents` | Code of conduct reports received, by severity | Moderator log | Manual |
| `moderator_hours` | Hours spent on triage duty, events and moderation | Self-reported by operators | Manual |

Also reviewed monthly (manual, recorded in the minutes, not in the CSV):

- **Multi-channel:** share of questions resolved in the first channel used; time from a chat finding to its issue or Discussion; duplicate items for the same problem across channels; share of model or dataset questions that state the exact revision; share of repeated questions turned into card or FAQ updates; items without a human response for 7 or 14 days, per platform.
- **Governance:** time to first PR review, distribution of merge approvers, bus factor, contributors promoted to maintainer, time to reach a decision, share of stale permissions removed, policy compliance rate.
- **Hugging Face:** model downloads and likes come from `hf-kpi.csv`; derived models (adapters, merges, quantizations)
  are counted by hand from the model page. The dataset repository is not published yet.

## Monthly review (60 minutes, 6 steps)

1. Check whether last month's commitments and improvement issues were completed.
2. Look at the trends: question backlog, response times, contributions, operator hours, incidents.
3. Read two conversations that went well and two that got stuck.
4. Turn one repeated question into a documentation improvement.
5. Review channels and permissions that should be closed or merged.
6. Pick one or two improvements for next month, each with an owner and a due date.

Write the minutes to `metrics/reviews/YYYY-MM.md` (for example `metrics/reviews/2026-10.md`) and open a pull request that also adds the month's row to `COMMUNITY_KPI.csv`.
The first minutes file is created by the maintainers after the first review; it is not generated.
Minutes never include per-person activity rankings.

## Forbidden metrics

- Member counts, message counts or emoji counts as targets.
- Ranking platforms against each other by response time.
- Per-person activity rankings, or any metric that works as individual surveillance.
- Bot or automated comments counted as a first human response.
