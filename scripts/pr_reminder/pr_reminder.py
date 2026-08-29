# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Post a daily Slack reminder listing the stalest open pull requests.

Open PRs with no activity within STALE_HOURS are sorted oldest-activity-first and
the top MAX_PRS are posted to SLACK_CHANNEL_ID.

Posting needs either SLACK_WEBHOOK_URL, or SLACK_BOT_TOKEN plus SLACK_CHANNEL_ID.

Environment variables:
  GITHUB_TOKEN            GitHub REST API token; required to file a failure issue,
                          optional (but rate-limit-easing) for reading a public repo
  GITHUB_REPOSITORY       owner/repo to scan (default torch-spyre/spyre-inference)
  SLACK_WEBHOOK_URL       Slack incoming-webhook URL, channel is fixed by the hook (secret)
  SLACK_BOT_TOKEN         Slack bot token, needs chat:write (secret)
  SLACK_CHANNEL_ID        channel to post into on the bot-token path, e.g. C0123456789
  SLACK_MENTION_GROUP_ID  who to ping in the header: a user group ID, or one of
                          here/channel/everyone; empty pings nobody
  MAX_PRS                 how many PRs to list (default 10)
  STALE_HOURS             inactivity threshold in hours (default 24)
  MAX_IDLE_DAYS           skip PRs idle longer than this; 0 disables (default 30)
  EXEMPT_LABELS           comma-separated labels that exclude a PR
  EXEMPT_TITLE_TEXT       comma-separated title substrings that exclude a PR
                          (default "do not merge")
  SLACK_HEADER_TEMPLATE   overrides HEADER_TEMPLATE
  SLACK_PR_LINE_TEMPLATE  overrides PR_LINE_TEMPLATE
  SLACK_FOOTER_TEMPLATE   overrides FOOTER_TEMPLATE

On failure the job escalates: report to Slack, else open a GitHub issue, else just
fail. Every failure path exits non-zero so the Actions run goes red.
"""

import logging
import os
import re
import sys
import traceback
from argparse import ArgumentParser, Namespace
from datetime import UTC, datetime, timedelta

import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackClientError
from slack_sdk.webhook import WebhookClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("pr_reminder")


def normalize_title(text: str) -> str:
    """Lowercase and fold runs of whitespace/-/_ to single spaces, for matching."""
    return re.sub(r"[\s_-]+", " ", text).strip().lower()


GITHUB_API = "https://api.github.com"
GITHUB_REPO = os.environ.get("GITHUB_REPOSITORY") or "torch-spyre/spyre-inference"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# Either an incoming-webhook URL (channel is baked into the hook) or a bot token
# plus channel id. The webhook wins when both are configured.
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID", "")

SLACK_MENTION_GROUP_ID = os.environ.get("SLACK_MENTION_GROUP_ID", "")

# Slack only notifies on mention *entities*; plain "@handle" text renders as
# literal characters and pings nobody.
BROADCAST_MENTIONS = ("here", "channel", "everyone")

MAX_PRS = int(os.environ.get("MAX_PRS") or "10")
STALE_HOURS = float(os.environ.get("STALE_HOURS") or "24")
# Upper bound on idleness: a PR nobody has touched in a month is not a review
# backlog item any more, it is abandoned, and stale.yml already chases those.
# Set to 0 to disable the upper bound.
MAX_IDLE_DAYS = float(os.environ.get("MAX_IDLE_DAYS") or "30")
# Normalized like titles so one entry covers "do-not-merge", "do not merge" and
# "Do Not Merge" — GitHub label names carry spaces the exact string wouldn't match.
EXEMPT_LABELS = {
    normalized
    for label in (os.environ.get("EXEMPT_LABELS") or "keep-open,stale,do-not-merge").split(",")
    if (normalized := normalize_title(label))
}
# Authors flag "not ready" in the title far more often than with a label, so the
# title is checked too. Patterns are matched against a normalized title, which
# lets one entry cover "DO NOT MERGE", "do-not-merge" and "[Do Not Merge]".
EXEMPT_TITLE_TEXT = {
    normalized
    for text in (os.environ.get("EXEMPT_TITLE_TEXT") or "do not merge").split(",")
    if (normalized := normalize_title(text))
}

TITLE_MAX_CHARS = 120
# The pulls endpoint caps per_page at 100; the ascending `updated` sort puts the
# stalest PRs on page 1, so this is only a safety net for huge backlogs.
MAX_PAGES = 10

FAILURE_ISSUE_TITLE = "[pr-reminder] daily PR reminder job failed"

HEADER_TEMPLATE = os.environ.get("SLACK_HEADER_TEMPLATE") or (
    "{mention} :eyes: *{count} open PRs with no activity in the last {stale_hours:g}h* "
    "in <{repo_url}|{repo}> — oldest first:"
)
PR_LINE_TEMPLATE = os.environ.get("SLACK_PR_LINE_TEMPLATE") or (
    "{rank}. <{url}|#{number}> *{title}* — {author} · idle {idle_days:.1f}d{labels}"
)
FOOTER_TEMPLATE = (
    os.environ.get("SLACK_FOOTER_TEMPLATE")
    or "_Showing {shown} of {total} stale PRs. Please pick one up._"
)
FAILURE_TEMPLATE = ":rotating_light: *PR reminder job failed* for {repo}\n```{error}```\n{run_url}"


def parse_args() -> Namespace:
    parser = ArgumentParser("Post a Slack reminder about stale open PRs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="render and print the message instead of posting to Slack",
    )
    return parser.parse_args()


def github_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    # Reading a public repo works unauthenticated (60 req/h), which keeps local
    # dry runs setup-free. Writing the failure issue does need the token.
    if GITHUB_TOKEN:
        session.headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    else:
        log.warning("GITHUB_TOKEN is not set; using unauthenticated (rate-limited) API access")
    return session


def run_url() -> str:
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    run_id = os.environ.get("GITHUB_RUN_ID")
    return f"{server}/{GITHUB_REPO}/actions/runs/{run_id}" if run_id else "(local run)"


def fetch_open_prs(session: requests.Session) -> list[dict]:
    prs: list[dict] = []
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls"
    params = {"state": "open", "sort": "updated", "direction": "asc", "per_page": 100}
    for _ in range(MAX_PAGES):
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        page = response.json()
        prs.extend(page)
        next_url = response.links.get("next", {}).get("url")
        if not next_url:
            break
        url, params = next_url, {}
    log.info("Fetched %d open PRs from %s", len(prs), GITHUB_REPO)
    return prs


def select_stale(prs: list[dict], now: datetime) -> tuple[list[dict], int]:
    """Return (top MAX_PRS stale PRs, total eligible), oldest activity first."""
    newest = now - timedelta(hours=STALE_HOURS)
    oldest = now - timedelta(days=MAX_IDLE_DAYS) if MAX_IDLE_DAYS > 0 else None
    eligible = []
    abandoned = 0
    title_flagged = 0
    for pr in prs:
        if pr.get("draft"):
            continue
        labels = {normalize_title(label["name"]) for label in pr.get("labels") or []}
        if labels & EXEMPT_LABELS:
            continue
        title_key = normalize_title(pr["title"])
        if any(text in title_key for text in EXEMPT_TITLE_TEXT):
            title_flagged += 1
            continue
        updated = parse_timestamp(pr["updated_at"])
        if updated > newest:
            continue
        if oldest and updated < oldest:
            abandoned += 1
            continue
        eligible.append(pr)

    eligible.sort(key=lambda pr: pr["updated_at"])
    log.info("%d of %d open PRs are stale (>= %gh idle)", len(eligible), len(prs), STALE_HOURS)
    if title_flagged:
        log.info("Skipped %d PRs flagged in their title", title_flagged)
    if abandoned:
        log.info("Skipped %d PRs idle for more than %g days", abandoned, MAX_IDLE_DAYS)
    return eligible[:MAX_PRS], len(eligible)


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def escape_mrkdwn(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_mention(value: str) -> str:
    """Render SLACK_MENTION_GROUP_ID as a Slack mention entity, or "" for none.

    "here"/"channel"/"everyone" become the broadcast forms; anything else is
    taken as a user group ID (handles get renamed, IDs don't).
    """
    value = value.strip().lstrip("@")
    if not value:
        return ""
    if value.lower() in BROADCAST_MENTIONS:
        return f"<!{value.lower()}>"
    return f"<!subteam^{value}>"


def render_message(top: list[dict], total: int, now: datetime) -> str:
    header = HEADER_TEMPLATE.format(
        mention=build_mention(SLACK_MENTION_GROUP_ID),
        count=len(top),
        stale_hours=STALE_HOURS,
        repo=GITHUB_REPO,
        repo_url=f"https://github.com/{GITHUB_REPO}",
    ).lstrip()  # no leading gap when the mention is disabled
    lines = [header]
    for rank, pr in enumerate(top, start=1):
        title = pr["title"].strip()
        if len(title) > TITLE_MAX_CHARS:
            title = title[: TITLE_MAX_CHARS - 1].rstrip() + "…"
        names = [label["name"] for label in pr.get("labels") or []]
        lines.append(
            PR_LINE_TEMPLATE.format(
                rank=rank,
                number=pr["number"],
                url=pr["html_url"],
                title=escape_mrkdwn(title),
                author=(pr.get("user") or {}).get("login", "unknown"),
                idle_days=(now - parse_timestamp(pr["updated_at"])).total_seconds() / 86400,
                labels=" · " + ", ".join(f"`{n}`" for n in names) if names else "",
            )
        )
    lines.append(FOOTER_TEMPLATE.format(shown=len(top), total=total))
    return "\n".join(lines)


def slack_destination() -> str:
    """Describe where a post would go, without ever revealing the webhook URL."""
    if SLACK_WEBHOOK_URL:
        return "the incoming webhook's channel"
    return SLACK_CHANNEL_ID or "<unset>"


def post_to_slack(text: str) -> None:
    # An incoming webhook is bound to the channel it was created for, so
    # SLACK_CHANNEL_ID is only consulted on the bot-token path.
    if SLACK_WEBHOOK_URL:
        response = WebhookClient(SLACK_WEBHOOK_URL).send(
            text=text,
            unfurl_links=False,
            unfurl_media=False,
        )
        # WebhookClient.send() reports transport failures in the response rather
        # than raising, so a bad URL or revoked hook is silent unless checked.
        if response.status_code != 200 or response.body != "ok":
            raise RuntimeError(f"Slack webhook returned {response.status_code}: {response.body}")
        log.info("Posted reminder to Slack via incoming webhook")
        return

    if not SLACK_BOT_TOKEN or not SLACK_CHANNEL_ID:
        raise RuntimeError(
            "Configure either SLACK_WEBHOOK_URL, or both SLACK_BOT_TOKEN and SLACK_CHANNEL_ID"
        )
    WebClient(token=SLACK_BOT_TOKEN).chat_postMessage(
        channel=SLACK_CHANNEL_ID,
        text=text,
        unfurl_links=False,
        unfurl_media=False,
    )
    log.info("Posted reminder to Slack channel %s", SLACK_CHANNEL_ID)


def open_failure_issue(session: requests.Session, error: BaseException) -> None:
    """File (or comment on) the tracking issue for a reminder-job failure."""
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN is not set; cannot open a failure issue")
    body = (
        f"The daily PR reminder job failed and could not report to Slack.\n\n"
        f"Run: {run_url()}\n\n```\n{format_error(error)}\n```\n"
    )
    existing = find_failure_issue(session)
    if existing:
        session.post(
            f"{GITHUB_API}/repos/{GITHUB_REPO}/issues/{existing}/comments",
            json={"body": body},
            timeout=30,
        ).raise_for_status()
        log.info("Commented on existing failure issue #%d", existing)
        return
    response = session.post(
        f"{GITHUB_API}/repos/{GITHUB_REPO}/issues",
        json={"title": FAILURE_ISSUE_TITLE, "body": body},
        timeout=30,
    )
    response.raise_for_status()
    log.info("Opened failure issue %s", response.json().get("html_url"))


def find_failure_issue(session: requests.Session) -> int | None:
    """Look for an already-open failure issue so we file one, not one per day."""
    try:
        response = session.get(
            f"{GITHUB_API}/search/issues",
            params={"q": f'repo:{GITHUB_REPO} is:issue is:open in:title "{FAILURE_ISSUE_TITLE}"'},
            timeout=30,
        )
        response.raise_for_status()
        for item in response.json().get("items", []):
            if item["title"] == FAILURE_ISSUE_TITLE:
                return item["number"]
    except requests.RequestException:
        # A search outage must not stop us from reporting the real failure.
        log.warning("Could not search for an existing failure issue; filing a new one")
    return None


def redact(text: str) -> str:
    """Strip credentials from anything bound for a log, Slack, or a GitHub issue.

    Error text from requests/slack_sdk embeds the URL it failed to reach, and on
    the webhook path that URL *is* the credential. Everything that reports a
    failure must go through here.
    """
    for secret in (SLACK_WEBHOOK_URL, SLACK_BOT_TOKEN, GITHUB_TOKEN):
        if secret:
            text = text.replace(secret, "***")
    return text


class RedactingFilter(logging.Filter):
    """Redact secrets from every record reaching a handler, ours or a library's.

    Redacting only our own messages is not enough: on a connection failure
    slack_sdk logs the full request URL at INFO ("Going to retry the same
    request: POST <url>"), and on the webhook path that URL is the credential.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        # Only strings are rewritten; coercing numeric args would break %d/%g.
        if isinstance(record.args, tuple):
            record.args = tuple(redact(a) if isinstance(a, str) else a for a in record.args)
        return True


for _handler in logging.getLogger().handlers:
    _handler.addFilter(RedactingFilter())


def format_error(error: BaseException) -> str:
    return redact(
        "".join(traceback.format_exception(type(error), error, error.__traceback__)).strip()
    )


def write_step_summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a") as handle:
        handle.write(f"### PR reminder\n\n{text}\n")


def report_failure(error: BaseException) -> int:
    """Escalate a pre-Slack failure: Slack, else a GitHub issue, else just fail."""
    log.error("PR reminder failed:\n%s", format_error(error))
    failure_text = FAILURE_TEMPLATE.format(
        repo=GITHUB_REPO, error=format_error(error), run_url=run_url()
    )
    try:
        post_to_slack(failure_text)
        return 1
    except (SlackClientError, RuntimeError, OSError) as slack_error:
        log.error("Could not report the failure to Slack: %s", redact(str(slack_error)))
    try:
        open_failure_issue(github_session(), error)
    except Exception as issue_error:  # last rung of the ladder
        log.error("Could not open a failure issue either: %s", redact(str(issue_error)))
    return 1


def main() -> int:
    args = parse_args()
    now = datetime.now(UTC)

    try:
        session = github_session()
        top, total = select_stale(fetch_open_prs(session), now)
        if not top:
            log.info("No stale PRs; nothing to remind about")
            write_step_summary(f"No open PRs idle for {STALE_HOURS:g}h or more.")
            return 0
        text = render_message(top, total, now)
    except Exception as error:  # any failure before the post must escalate
        return report_failure(error)

    write_step_summary(text)
    if args.dry_run:
        log.info("Dry run; would post to %s:", slack_destination())
        print(text)
        return 0

    try:
        post_to_slack(text)
    except Exception as error:  # Slack unreachable, escalate to a GitHub issue
        log.error("Failed to post to Slack:\n%s", format_error(error))
        try:
            open_failure_issue(session, error)
        except Exception as issue_error:  # last rung of the ladder
            log.error("Could not open a failure issue either: %s", redact(str(issue_error)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
