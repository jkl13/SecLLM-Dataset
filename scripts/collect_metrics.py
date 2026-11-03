#!/usr/bin/env python3
"""Collect weekly GitHub and Hugging Face KPIs for SecLLM-Dataset (Python 3.9+, standard library only).

Usage:
  python scripts/collect_metrics.py                  print one KPI row (dry run)
  python scripts/collect_metrics.py --out FILE       append the row to a local CSV
  python scripts/collect_metrics.py --publish BRANCH append the row to metrics/github-kpi.csv on BRANCH
  python scripts/collect_metrics.py --hf MODEL_ID    collect the Hugging Face model row instead
                                                     (metrics/hf-kpi.csv; works with --out and --publish)
  python scripts/collect_metrics.py --self-test      check the calculations offline

The token is read from GITHUB_TOKEN (optional for a dry run, required for --publish).
When GITHUB_STEP_SUMMARY is set, the row is also appended there.
Column definitions: metrics/README.md.
"""
import argparse
import base64
import csv
import io
import json
import os
import re
import statistics
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

API = "https://api.github.com"
CSV_PATH = "metrics/github-kpi.csv"
COLUMNS = ["date", "window_days", "stars", "forks", "watchers", "open_issues", "issues_opened",
           "issues_closed", "prs_opened", "prs_merged", "median_first_human_response_hours",
           "first_time_contributors", "repeat_contributors"]
HF_API = "https://huggingface.co/api/models"
HF_CSV_PATH = "metrics/hf-kpi.csv"
HF_COLUMNS = ["date", "model", "revision", "downloads_30d", "likes"]


def ts(s):
    """Parse a GitHub timestamp ('2026-09-28T01:02:03Z'); None stays None."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def is_bot(user):
    return not user or user.get("type") == "Bot" or user.get("login", "").endswith("[bot]")


def request(url, token, method="GET", body=None):
    """Return (parsed JSON, next-page URL or None)."""
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
               "User-Agent": "SecLLM-Dataset-collect-metrics"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        m = re.search(r'<([^>]+)>;\s*rel="next"', resp.headers.get("Link", ""))
        return json.load(resp), (m.group(1) if m else None)


def get_all(url, token):
    items = []
    while url:
        page, url = request(url, token)
        items += page
    return items


def build_row(repo, items, responses, now, days):
    """Compute one KPI row.

    repo:      GET /repos/{r} JSON
    items:     every issue and PR (GET /repos/{r}/issues?state=all JSON)
    responses: {item number: [{"user": {...}, "at": timestamp}, ...]} comments and reviews,
               needed only for items created inside the window
    """
    start = now - timedelta(days=days)
    in_win = lambda t: t is not None and start <= t < now  # noqa: E731
    issues = [i for i in items if "pull_request" not in i]
    prs = [i for i in items if "pull_request" in i]

    hours = []
    for i in items:
        if not in_win(ts(i["created_at"])):
            continue
        author = i["user"]["login"] if i.get("user") else None
        human = [ts(r["at"]) for r in responses.get(i["number"], [])
                 if r["at"] and not is_bot(r["user"]) and r["user"]["login"] != author]
        if human:
            hours.append((min(human) - ts(i["created_at"])).total_seconds() / 3600)

    # Contributors = human PR authors; month history covers every PR ever opened.
    months, first_pr = {}, {}
    for p in prs:
        if is_bot(p.get("user")):
            continue
        login, created = p["user"]["login"], ts(p["created_at"])
        months.setdefault(login, set()).add((created.year, created.month))
        first_pr[login] = min(first_pr.get(login, created), created)
    active = {p["user"]["login"] for p in prs
              if not is_bot(p.get("user")) and in_win(ts(p["created_at"]))}

    return {
        "date": now.date().isoformat(),
        "window_days": days,
        "stars": repo["stargazers_count"],
        "forks": repo["forks_count"],
        "watchers": repo["subscribers_count"],
        "open_issues": repo["open_issues_count"],
        "issues_opened": sum(in_win(ts(i["created_at"])) for i in issues),
        "issues_closed": sum(in_win(ts(i.get("closed_at"))) for i in issues),
        "prs_opened": sum(in_win(ts(p["created_at"])) for p in prs),
        "prs_merged": sum(in_win(ts(p["pull_request"].get("merged_at"))) for p in prs),
        "median_first_human_response_hours": round(statistics.median(hours), 1) if hours else "",
        "first_time_contributors": sum(in_win(first_pr[x]) for x in active),
        "repeat_contributors": sum(len(months[x]) >= 2 for x in active),
    }


def collect(repo_name, days, token, now):
    base = f"{API}/repos/{repo_name}"
    repo, _ = request(base, token)
    # ponytail: fetches the full issue/PR history every run; add `since=` paging if the repo grows past a few thousand items.
    items = get_all(f"{base}/issues?state=all&per_page=100", token)
    start = now - timedelta(days=days)
    responses = {}
    for i in items:
        if ts(i["created_at"]) < start:
            continue
        n = i["number"]
        rs = [{"user": c["user"], "at": c["created_at"]}
              for c in get_all(f"{base}/issues/{n}/comments?per_page=100", token)]
        if "pull_request" in i:
            rs += [{"user": r["user"], "at": r.get("submitted_at")}
                   for r in get_all(f"{base}/pulls/{n}/reviews?per_page=100", token)]
        responses[n] = rs
    return build_row(repo, items, responses, now, days)


def hf_row(model, now):
    """One Hugging Face row from GET https://huggingface.co/api/models/{id}.

    The Hub's `downloads` is a rolling 30-day count, not a lifetime total, so rows are not summed.
    """
    return {"date": now.date().isoformat(), "model": model["id"], "revision": model.get("sha", ""),
            "downloads_30d": model.get("downloads", 0), "likes": model.get("likes", 0)}


def collect_hf(model_id, now):
    # Public endpoint, no token: the GitHub token is never sent to another host.
    model, _ = request(f"{HF_API}/{model_id}", None)
    return hf_row(model, now)


def csv_text(row, header, columns=COLUMNS):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    if header:
        w.writerow(columns)
    w.writerow([row[c] for c in columns])
    return buf.getvalue()


def append_file(path, row, columns=COLUMNS):
    path = Path(path)
    header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(csv_text(row, header, columns))


def publish(repo_name, branch, row, token, csv_path=CSV_PATH, columns=COLUMNS):
    """Append the row to csv_path on BRANCH via the Contents API (fails on any conflict)."""
    url = f"{API}/repos/{repo_name}/contents/{csv_path}"
    try:
        cur, _ = request(f"{url}?ref={branch}", token)
        old, sha = base64.b64decode(cur["content"]).decode("utf-8"), cur["sha"]
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        old, sha = "", None
    new = old + ("" if not old or old.endswith("\n") else "\n") + csv_text(row, not old, columns)
    body = {"message": f"metrics: KPI row {row['date']}", "branch": branch,
            "content": base64.b64encode(new.encode("utf-8")).decode("ascii")}
    if sha:
        body["sha"] = sha
    request(url, token, method="PUT", body=body)


def self_test():
    now = datetime(2026, 9, 28, tzinfo=timezone.utc)
    human = lambda login: {"login": login, "type": "User"}  # noqa: E731
    repo = {"stargazers_count": 5, "forks_count": 2, "subscribers_count": 3, "open_issues_count": 4}
    pr = lambda merged=None: {"merged_at": merged}  # noqa: E731
    items = [
        # issue in window: bot at +1h, author self-reply at +2h, human at +10h -> 10h
        {"number": 1, "user": human("alice"), "created_at": "2026-09-25T00:00:00Z", "closed_at": None},
        # issue in window, closed in window: [bot]-suffixed login at +1h, human at +8h -> 8h
        {"number": 2, "user": human("bob"), "created_at": "2026-09-26T00:00:00Z",
         "closed_at": "2026-09-27T00:00:00Z"},
        # PR in window, merged: review by a human at +6h -> 6h. carol's first PR ever -> first-time
        {"number": 3, "user": human("carol"), "created_at": "2026-09-22T00:00:00Z", "closed_at": None,
         "pull_request": pr("2026-09-23T00:00:00Z")},
        # PR in window by dave, who also had a PR in July -> repeat, not first-time
        {"number": 4, "user": human("dave"), "created_at": "2026-09-27T00:00:00Z", "closed_at": None,
         "pull_request": pr()},
        {"number": 5, "user": human("dave"), "created_at": "2026-07-01T00:00:00Z", "closed_at": None,
         "pull_request": pr("2026-07-02T00:00:00Z")},
        # outside the window: issue closed in window still counts as closed, nothing else
        {"number": 6, "user": human("erin"), "created_at": "2026-08-01T00:00:00Z",
         "closed_at": "2026-09-24T00:00:00Z"},
        # bot PR in window: counted as a PR, never as a contributor
        {"number": 7, "user": {"login": "dependabot[bot]", "type": "Bot"}, "created_at": "2026-09-26T00:00:00Z",
         "closed_at": None, "pull_request": pr()},
    ]
    bot = {"login": "github-actions[bot]", "type": "Bot"}
    responses = {
        1: [{"user": bot, "at": "2026-09-25T01:00:00Z"}, {"user": human("alice"), "at": "2026-09-25T02:00:00Z"},
            {"user": human("maint"), "at": "2026-09-25T10:00:00Z"}],
        2: [{"user": {"login": "helper[bot]", "type": "User"}, "at": "2026-09-26T01:00:00Z"},
            {"user": human("maint"), "at": "2026-09-26T08:00:00Z"}],
        3: [{"user": human("maint"), "at": "2026-09-22T06:00:00Z"}, {"user": human("maint"), "at": None}],
        4: [{"user": human("dave"), "at": "2026-09-27T01:00:00Z"}],  # self-reply only -> no response
        7: [{"user": bot, "at": "2026-09-26T00:01:00Z"}],
    }
    row = build_row(repo, items, responses, now, 7)
    expected = {"date": "2026-09-28", "window_days": 7, "stars": 5, "forks": 2, "watchers": 3, "open_issues": 4,
                "issues_opened": 2, "issues_closed": 2, "prs_opened": 3, "prs_merged": 1,
                "median_first_human_response_hours": 8.0,
                "first_time_contributors": 1, "repeat_contributors": 1}
    failures = [f"{k}: expected {v!r}, got {row[k]!r}" for k, v in expected.items() if row[k] != v]
    if build_row(repo, items, {1: responses[1][:2]}, now, 7)["median_first_human_response_hours"] != "":
        failures.append("bot-only / self-only responses must leave the median blank")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "k.csv"
        append_file(path, row)
        append_file(path, row)
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != 3 or lines.count(",".join(COLUMNS)) != 1:
            failures.append(f"header must be written once: {lines}")
    hf = hf_row({"id": "org/m", "sha": "abc", "downloads": 17, "likes": 1}, now)
    if csv_text(hf, True, HF_COLUMNS) != "date,model,revision,downloads_30d,likes\n2026-09-28,org/m,abc,17,1\n":
        failures.append(f"HF row: {hf}")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "hf.csv"
        append_file(path, hf, HF_COLUMNS)
        append_file(path, hf, HF_COLUMNS)
        if path.read_text(encoding="utf-8").splitlines().count(",".join(HF_COLUMNS)) != 1:
            failures.append("HF header must be written once")
    if failures:
        print("self-test FAILED:\n  " + "\n  ".join(failures))
        return 1
    print(f"self-test OK: {len(expected)} columns match (bots and self-replies ignored, median, window, "
          "first-time vs repeat), header written once, HF row")
    return 0


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY") or "Sparrow-Co-Ltd/SecLLM-Dataset")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out", metavar="FILE")
    ap.add_argument("--publish", metavar="BRANCH")
    ap.add_argument("--hf", metavar="MODEL_ID", help="collect the Hugging Face model row instead")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    token = os.environ.get("GITHUB_TOKEN")
    if args.publish and not token:
        print("--publish needs GITHUB_TOKEN")
        return 1
    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    if args.hf:
        row, csv_path, columns = collect_hf(args.hf, now), HF_CSV_PATH, HF_COLUMNS
    else:
        row, csv_path, columns = collect(args.repo, args.days, token, now), CSV_PATH, COLUMNS
    line = csv_text(row, False, columns).strip()
    if args.out:
        append_file(args.out, row, columns)
    if args.publish:
        publish(args.repo, args.publish, row, token, csv_path, columns)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(f"KPI row ({','.join(columns)}): `{line}`\n")
    print(",".join(columns))
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
