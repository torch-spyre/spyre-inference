---
name: cleanup-monthly-labels
description: Reconcile monthly labels (April2026, May2026, June2026, …) on the torch-spyre vLLM project board (project 2, view 7, repo torch-spyre/spyre-inference) at the end of each month. Use whenever the user asks to "clean up the board", "do the monthly label cleanup", "do the May/June/July cleanup", or otherwise refresh which monthly bucket each issue lives in. The skill enforces three rules — May-closed → MayYYYY, open MayYYYY → JuneYYYY swap, and Q2-relevant unlabeled issues → current month — while flagging anomalies for the user instead of guessing.
---

# Monthly board-label cleanup

Reconcile the monthly labels (`April2026`, `May2026`, `June2026`, …) on the torch-spyre
project board so each tracked issue carries exactly one monthly label, reflecting either
the month the issue closed or the month the team is actively pursuing it.

The board this maintains:

- Project: <https://github.com/orgs/torch-spyre/projects/2> — "Torch-Spyre Device Enablement"
- View: 7 — "vLLM" — filter `label:vllm-spyre-next,vllm-spyre-old`
- Issue scope: `torch-spyre/spyre-inference` only

The user runs this near the end of each month (e.g., after the May→June rollover).

## Before you start

### Get the month right

Ask the user which month they're closing out and which month they're rolling into,
unless they said it explicitly ("do the June cleanup", "May→June pass"). Do not infer
from the system date alone — cleanups often run a few days late, and the wrong month
silently mislabels every issue.

Throughout this skill:

- `<closing>` = the month being closed (e.g., `May`, `June`).
- `<active>` = the new active month (e.g., `June`, `July`).
- Labels are `<closing>YYYY` and `<active>YYYY`, matching the existing label names.

### Verify GitHub auth scopes

Reading and writing the project board requires `read:project,project` token scopes.
The default `gh` token lacks them.

```bash
gh auth status 2>&1 | grep 'scopes'
```

If `project` is missing, ask the user to run:

```bash
gh auth refresh -h github.com -s read:project,project
```

This is interactive (browser flow). Wait for them to confirm before proceeding.

### Verify monthly labels exist

```bash
gh label list --repo torch-spyre/spyre-inference --limit 200 \
  | grep -E '<closing>YYYY|<active>YYYY'
```

If a label is missing, **stop and report**. Do not auto-create — the user may want to
choose color/description, or the missing label may indicate a typo in the cleanup
inputs.

## The three rules

Apply these in order. Each issue matches at most one rule.

### Rule 1 — closed in `<closing>` → ensure `<closing>YYYY`

For every issue on the board with `state == CLOSED` and `closedAt` in the closing month
(`YYYY-MM-01` ≤ closedAt < first-of-`<active>`):

- If the issue lacks `<closing>YYYY`, add it.
- If the issue has any *other* monthly label (`<previous>YYYY`, `<active>YYYY`),
  remove it. One monthly per issue.

### Rule 2 — open with `<closing>YYYY` → swap to `<active>YYYY`

For every open issue on the board still carrying `<closing>YYYY` (work that started
last month and continues): add `<active>YYYY`, remove `<closing>YYYY`. Also remove any
other stale monthlies.

### Rule 3 — open, no monthly, Q2-relevant → add `<active>YYYY`

For every open issue on the board with **no** monthly label, decide whether it relates
to the active quarter's milestone goals (see "Q2-relevance judgement" below). If yes,
add `<active>YYYY`. If no (future work, off-quarter), leave it alone.

**Important:** rule 3 is *not* "add `<active>YYYY` to every unlabeled board issue." It
is gated on quarterly relevance. Many issues on the board are tracking Q3+ work
(Mamba2/hybrid models, MoE, encoder support, prefill/decode disaggregation, sliding
window attention) and should stay unlabeled until their quarter arrives.

## Source-of-truth queries

### Fetch the view's filter (don't hardcode — re-fetch each run)

```bash
gh api graphql -f query='
{
  organization(login: "torch-spyre") {
    projectV2(number: 2) {
      title
      view(number: 7) { name filter }
    }
  }
}'
```

Today this returns `filter: "label:vllm-spyre-next,vllm-spyre-old"`. If it changes,
update the local filter logic accordingly.

### Enumerate board items in `spyre-inference` matching the view filter

```bash
gh api graphql --paginate -f query='
query($endCursor: String) {
  organization(login: "torch-spyre") {
    projectV2(number: 2) {
      items(first: 100, after: $endCursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          content {
            __typename
            ... on Issue {
              number title state closedAt
              repository { nameWithOwner }
              labels(first: 30) { nodes { name } }
            }
          }
        }
      }
    }
  }
}' --jq '.data.organization.projectV2.items.nodes[]
  | select(.content.__typename == "Issue")
  | select(.content.repository.nameWithOwner == "torch-spyre/spyre-inference")
  | select((.content.labels.nodes | map(.name)
            | (contains(["vllm-spyre-next"]) or contains(["vllm-spyre-old"]))))
  | {n: .content.number, t: .content.title, s: .content.state,
     c: .content.closedAt, labels: [.content.labels.nodes[].name]}'
```

Save to `/tmp/board_items.jsonl` for the classifier.

### Read the active quarter's milestone goals

```bash
gh api repos/torch-spyre/torch-spyre/milestones --jq \
  '.[] | select(.title == "2026 Q2") | .description'
```

The milestone numbers each goal (1, 2, 3, …). Use the description verbatim when
explaining Q2-relevance to the user. Switch the title to the current quarter
(`2026 Q3`, `2026 Q4`, `2027 Q1`, …) when the calendar moves on.

## Classifier (jq)

Replace `<closing>YYYY`, `<active>YYYY`, `<previous>YYYY`, and the date bounds before
running. The skeleton:

```bash
CLOSING=May2026
ACTIVE=June2026
START=2026-05-01
END=2026-06-01

jq -r --arg CL "$CLOSING" --arg AC "$ACTIVE" --arg ST "$START" --arg EN "$END" '
def has(x): .labels | index(x);
def monthlies: ["April2026","May2026","June2026","July2026","August2026"];

. as $i |
if $i.s == "CLOSED" and ($i.c >= $ST) and ($i.c < $EN) then
  {rule: 1, n: $i.n, t: $i.t, current: $i.labels,
   add:    (if has($CL) then [] else [$CL] end),
   remove: [monthlies[] | select(. != $CL) | select(. as $m | $i.labels | index($m))]}
elif $i.s == "OPEN" and (has($CL)) then
  {rule: 2, n: $i.n, t: $i.t, current: $i.labels,
   add:    (if has($AC) then [] else [$AC] end),
   remove: [monthlies[] | select(. != $AC) | select(. as $m | $i.labels | index($m))]}
elif $i.s == "OPEN" and (any(monthlies[]; . as $m | $i.labels | index($m)) | not) then
  {rule: 3, n: $i.n, t: $i.t, current: $i.labels,
   add: [$AC], remove: []}
else
  {rule: 0, n: $i.n, t: $i.t, current: $i.labels, add: [], remove: []}
end
| @json
' /tmp/board_items.jsonl > /tmp/classified.jsonl
```

Then group by rule and present.

## Q2-relevance judgement (rule 3)

Read the milestone description first, then for each rule-3 candidate propose a goal
mapping. Group the dry-run by goal so the user can spot mistakes per category. For
example:

```text
Goal 3: Functional vLLM support for Granite 3 8b / Granite 4 8b
  #6   ParallelLMHead compatible with SpyreRunner    | Q2 enablement
  #22  RMSNorm error when hidden size not mult of 64 | Q2 model bring-up bug
  …
Goal 4: Functional multi-AIU PyTorch support (TP>1)
  #13  Add Support for tp>1 in spyre-next  | Goal-4 EPIC
  …
NOT Q2 (recommend leave unlabeled):
  #120, #114, #113  Mamba2 backend  | Q3 hybrid models
  #192  MoE                         | Not in Q2 goals
  …
```

**Borderline cases worth surfacing explicitly:**

- Anything around "Granite 4 8b" — Q2 goal 3 names it, but Granite 4 is hybrid, which
  is also Q3 goal 2. Mamba2 issues are the usual victims; ask the user.
- Helion exploration issues — the team has tagged some Helion work May2026 in
  practice, but the Q2 milestone doesn't list Helion. Ask.
- Generic CI/type-check hygiene — defensible either way; mention which call you made.

Don't decide these yourself. Lay out the reasoning and ask.

## Anomaly playbook

These cases fall outside the 3 rules. Detect them, surface them in the dry-run, and
**ask the user** what to do — don't bake in defaults. The same anomaly may have a
different answer next month.

### A. Closed in `<active>` carrying old `<closing>YYYY` label

Issue closed in the *new* active month but still tagged with the closing month's label.
Could mean: the work belonged to last month and bled over (keep `<closing>YYYY`), or
the work belonged to the new month (swap to `<active>YYYY`).

```bash
jq -c 'select(.s == "CLOSED" and .c >= "<active-start>" and (.labels | index("<closing>YYYY")))' \
  /tmp/board_items.jsonl
```

### B. Closed in `<active>` with no monthly label

Issue closed in the new month but never received any monthly label. Probably should be
labeled — but with which month? Could be `<closing>YYYY` (work was done in May, merged
in early June) or `<active>YYYY` (work was done quickly in June). Ask.

```bash
jq -c 'select(.s == "CLOSED" and .c >= "<active-start>") |
       select((.labels | index("<closing>YYYY")) or (.labels | index("<active>YYYY")) | not)' \
  /tmp/board_items.jsonl
```

### C. Open with a stale monthly label (2+ months old)

Issue still open and still tagged `<previous>YYYY` (e.g., `April2026` in June). Probably
needs swapping to `<active>YYYY`, but could also be a deliberate "park". Ask.

```bash
jq -c 'select(.s == "OPEN" and (.labels | index("<previous>YYYY")) and
              ((.labels | index("<closing>YYYY")) | not) and
              ((.labels | index("<active>YYYY")) | not))' /tmp/board_items.jsonl
```

### Other anomalies to flag

- Multiple monthly labels on a single issue (the cleanup itself violates the
  one-monthly invariant, so this should be empty after — but pre-cleanup it can
  surface team mistakes).
- Issues on the board in repos *other* than `spyre-inference`. The current scope is
  `spyre-inference` only; flag cross-repo board items so the user can decide whether
  to extend the cleanup.

## Workflow

1. **Confirm month** — ask if not stated. Set `<closing>`, `<active>`, date bounds.
2. **Verify scopes** — `gh auth status`. If missing, prompt user to refresh.
3. **Verify labels** — `gh label list … | grep`. Halt if missing.
4. **Re-fetch view filter** — don't trust a cached filter, the view may have changed.
5. **Enumerate board items** — paginated GraphQL, save to file.
6. **Classify** — jq classifier, group by rule.
7. **Detect anomalies** — A/B/C buckets above.
8. **Dry-run report** — present rules 1, 2, 3 (with Q2 grouping) and anomalies. Wait
   for user approval. Make borderline Q2 calls explicit.
9. **Apply** — `gh issue edit <n> --repo torch-spyre/spyre-inference --add-label X
   --remove-label Y`, sequentially, one log line per issue. On any failure, halt.
10. **Verify** — re-fetch the board and re-run the classifier; output should be empty
    (no rule-violating issues remain). Spot-check 2–3 issues in the GitHub UI.

## Apply step (template)

```bash
REPO=torch-spyre/spyre-inference
apply() {
  local n=$1 add=$2 rem=$3
  local args="--repo $REPO"
  [ -n "$add" ] && args="$args --add-label $add"
  [ -n "$rem" ] && args="$args --remove-label $rem"
  echo "#$n: +[$add] -[$rem]"
  gh issue edit "$n" $args >/dev/null
}

# Rule 1: + <closing>YYYY (and remove other monthlies if any)
for n in <list>; do apply $n <closing>YYYY ""; done

# Rule 2: + <active>YYYY  - <closing>YYYY
for n in <list>; do apply $n <active>YYYY <closing>YYYY; done

# Rule 3: + <active>YYYY (Q2-relevant only, after user confirms)
for n in <list>; do apply $n <active>YYYY ""; done
```

Run sequentially, not in parallel — these are writes against shared GitHub state.

## Post-cleanup verification

```bash
# Re-fetch and re-classify; results should be empty for each check.

# Every closed-in-<closing> issue has <closing>YYYY exactly once
jq -c 'select(.s == "CLOSED" and .c >= "<start>" and .c < "<end>") |
       select((.labels | index("<closing>YYYY")) | not)' /tmp/board_after.jsonl

# No open issue still has <closing>YYYY
jq -c 'select(.s == "OPEN" and (.labels | index("<closing>YYYY")))' /tmp/board_after.jsonl

# No issue has more than one monthly label
jq -c '. as $i |
  ([.labels[] | select(. == "April2026" or . == "May2026" or
                       . == "June2026" or . == "July2026")] | length) as $c |
  select($c > 1) | "VIOLATION: #\($i.n) labels=\($i.labels)"' /tmp/board_after.jsonl
```

If any of those produce output, surface and resolve before declaring done.

## What to keep out of this skill

- Specific issue numbers from past cleanups (ephemeral).
- Hardcoded month names in the rules — the rules are parametric.
- A list of "Q2 goals" — re-read the milestone description each run; the team edits it.
- A baked-in answer for anomaly A/B/C — those are user judgement calls, not policy.
