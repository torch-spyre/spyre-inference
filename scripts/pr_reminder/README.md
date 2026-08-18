# PR reminder bot

Posts a Slack reminder on weekdays listing the open pull requests that reviewers have
gone quiet on, so long-lived contributions don't rot into rebase-only work.

On each run the bot lists open PRs, drops the ones that are drafts, carry an exempt
label, or are flagged in their title, keeps those idle between `STALE_HOURS` and
`MAX_IDLE_DAYS`, sorts them oldest-activity-first, and posts the top `MAX_PRS`. When
nothing is stale it posts nothing — an "all clear" would only train people to mute the
channel.

The upper bound matters: a PR untouched for over a month is abandoned rather than
review-blocked, and [stale.yml](../../.github/workflows/stale.yml) already chases those
at 90 days. Without it, the same few fossils would head the list every single day and
the reminder would stop being actionable.

"Activity" is GitHub's `updated_at` on the PR, which advances on pushes, comments,
reviews and label changes.

Driven by [pr_reminder_slack.yaml](../../.github/workflows/pr_reminder_slack.yaml)
twice a day on weekdays, plus manual dispatch:

| Cron (UTC) | Zurich | US East | Audience |
|---|---|---|---|
| `3 8 * * 1-5` | 10:03 CEST / 09:03 CET | 04:03 EDT / 03:03 EST | European morning |
| `3 15 * * 1-5` | 17:03 CEST / 16:03 CET | 11:03 EDT / 10:03 EST | US morning |

Day-of-week is evaluated in UTC. Both runs sit mid-day UTC, so `1-5` lands on Mon–Fri
local time on both sides of the Atlantic with no date-boundary surprises. Monday's post
covers the weekend, so anything untouched since Friday shows ~3 days idle.

Both fire off the hour on purpose — GitHub delays scheduled runs under load and the top
of the hour is its documented peak. Cron has no DST handling, so each slides an hour
later in local terms during summer time.

## Setup

Pick one of the two Slack paths. The webhook is simpler; the bot token is worth it if
you ever want to retarget the channel without minting a new hook, or to post to several
channels. When both are configured the webhook wins.

**Incoming webhook (simplest).** Create the hook in Slack (*Your apps → Incoming
Webhooks → Add New Webhook to Workspace*), pick the channel, and add the resulting URL
as the repository secret `SLACK_WEBHOOK_URL`. The channel is baked into the hook, so
`SLACK_CHANNEL_ID` is ignored on this path. Treat the URL as a credential — anyone
holding it can post to that channel.

**Bot token.** Create a Slack app with the `chat:write` bot scope, install it, and
invite it to the channel (`/invite @your-app`). Add the bot token as the repository
secret `SLACK_BOT_TOKEN` and the channel ID (e.g. `C0123456789`, from the channel's
*View channel details*) as the repository variable `SLACK_CHANNEL_ID`.

The workflow's `GITHUB_TOKEN` covers the GitHub side; no PAT is needed.

## Configuration

Everything is read from the environment, so it can be changed either in the workflow
file or from **Settings → Secrets and variables → Actions** without a commit.

| Env var | Repo variable | Default | Meaning |
|---|---|---|---|
| `SLACK_WEBHOOK_URL` | secret of the same name | — | Incoming-webhook URL; takes precedence |
| `SLACK_BOT_TOKEN` | secret of the same name | — | Slack bot token (`chat:write`) |
| `SLACK_CHANNEL_ID` | `SLACK_CHANNEL_ID` | — | Channel to post into (bot-token path only) |
| `SLACK_MENTION_GROUP_ID` | `PR_REMINDER_SLACK_GROUP_ID` | see workflow | Who to ping: a user group ID, or `here`/`channel`/`everyone`; empty pings nobody |
| `MAX_PRS` | `PR_REMINDER_MAX_PRS` | `10` | How many PRs to list |
| `STALE_HOURS` | `PR_REMINDER_STALE_HOURS` | `24` | Inactivity threshold, in hours |
| `MAX_IDLE_DAYS` | `PR_REMINDER_MAX_IDLE_DAYS` | `30` | Skip PRs idle longer than this; `0` disables |
| `EXEMPT_LABELS` | `PR_REMINDER_EXEMPT_LABELS` | `keep-open,stale,do-not-merge` | Labels that exclude a PR |

## Message templates

`HEADER_TEMPLATE`, `PR_LINE_TEMPLATE` and `FOOTER_TEMPLATE` live at the top of
[pr_reminder.py](pr_reminder.py). Edit them there — the workflow deliberately does not
plumb them through repo variables, so wording changes go through review like any other
code. Each still honours a `SLACK_HEADER_TEMPLATE` / `SLACK_PR_LINE_TEMPLATE` /
`SLACK_FOOTER_TEMPLATE` env override, which is handy for trying wording locally.

Templates are Slack `mrkdwn` and use `str.format` placeholders. The header takes
`mention`, `count`, `stale_hours`, `repo`, `repo_url`; each PR line takes `rank`,
`number`, `url`, `title`, `author`, `idle_days`, `labels`; the footer takes `shown`,
`total`.

`{mention}` is built from `SLACK_MENTION_GROUP_ID`:

| Value | Renders as | Effect |
|---|---|---|
| `S0BKBNPFP0W` | `<!subteam^S0BKBNPFP0W>` | Notifies every member of that user group |
| `here` | `<!here>` | Notifies active members of the channel |
| `channel` | `<!channel>` | Notifies every member of the channel |
| `everyone` | `<!everyone>` | Notifies the whole workspace via `#general` |
| empty | nothing | No ping; the header starts at the emoji |

These entity forms are the only ones Slack notifies on — plain `@handle` text renders as
literal characters and pings no one
([Slack docs](https://docs.slack.dev/messaging/formatting-message-text)). For a user
group, use the *ID* rather than the handle: handles get renamed, IDs don't. A leading `@`
is tolerated, so `@here` and `here` behave the same.

## On failure

The job escalates rather than failing silently:

1. Report the failure to Slack.
2. If Slack itself is the problem, open (or comment on) a GitHub issue titled
   `[pr-reminder] daily PR reminder job failed` — one issue, not one per day.
3. If neither works, just fail the run.

Every one of these paths still exits non-zero, so a broken run is always red in the
Actions UI.

## Running it locally

`--dry-run` renders the message and prints it instead of calling Slack:

```bash
GITHUB_TOKEN=$(gh auth token) \
  uv run --no-project --with requests --with slack-sdk \
  python scripts/pr_reminder/pr_reminder.py --dry-run
```

Reading a public repo works without a token too, at GitHub's 60 requests/hour
unauthenticated rate limit.

To post for real from a laptop, keep the webhook URL out of your shell history and out
of the repo — e.g. in `~/.pr_reminder_webhook` (mode `600`):

```bash
SLACK_WEBHOOK_URL="$(cat ~/.pr_reminder_webhook)" \
  uv run --no-project --with requests --with slack-sdk \
  python scripts/pr_reminder/pr_reminder.py
```

Cross-check the selection with:

```bash
gh pr list --state open --limit 100 \
  --json number,title,updatedAt,isDraft,labels --jq 'sort_by(.updatedAt)'
```
