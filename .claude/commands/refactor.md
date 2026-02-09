---
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, Task, TeamCreate, TeamDelete, TaskCreate, TaskUpdate, TaskList, TaskGet, SendMessage
description: 3-agent PR code review debate -- spawns Devil's Advocate, Neutral Judge, and Approval Advocate to review current PR
---

# /refactor -- 3-Agent PR Code Review Debate

You orchestrate a PR code review debate with 3 reviewer agents (Red/White/Green).
Follow these 10 steps exactly. All output is in English.

## Step 1: Verify PR Exists

Run:
```bash
gh pr view --json number,url,title,body,headRefName,baseRefName
```

If the command fails, exit immediately:
> No PR found on the current branch. Create a PR first.

Extract and remember these values:
- `PR_NUMBER`, `PR_URL`, `PR_TITLE`, `HEAD_REF`, `BASE_REF`

## Step 2: Gather PR Diff and Context

Run these commands (in parallel where possible):
```bash
gh pr diff
gh pr diff --stat
git rev-parse HEAD
git remote get-url origin
```

Store results as:
- `FULL_DIFF` -- the full diff output
- `DIFF_STAT` -- the diff stat summary
- `HEAD_SHA` -- current HEAD commit SHA
- `OWNER_REPO` -- parse the remote URL to extract `OWNER/REPO` (handle both HTTPS and SSH formats)

If `FULL_DIFF` is empty, exit immediately:
> Nothing to review. The PR diff is empty.

## Step 3: Create Team and Spawn 3 Reviewers (Parallel)

### 3a. Create the team
```
TeamCreate(team_name="pr-review-{PR_NUMBER}")
```

### 3b. Create 3 tracking tasks via TaskCreate
- "Red Reviewer: Devil's Advocate analysis"
- "White Reviewer: Neutral Judge analysis"
- "Green Reviewer: Approval Advocate analysis"

### 3c. Spawn 3 reviewer agents in a SINGLE message (parallel)

Use the `Task` tool three times in the same message. Each agent:
- `subagent_type`: `"general-purpose"`
- `team_name`: `"pr-review-{PR_NUMBER}"`
- `mode`: `"bypassPermissions"`

Read `~/.claude/skills/refactor/references/reviewer-prompts.md` for the 3 role-specific prompt templates.
Fill in `FULL_DIFF`, `DIFF_STAT`, `PR_TITLE` into each template.

| Name | Role |
|------|------|
| `red-reviewer` | Devil's Advocate -- always opposes, attacks security/race conditions/hidden assumptions |
| `white-reviewer` | Neutral Judge -- risk = probability x impact, pragmatic tradeoffs |
| `green-reviewer` | Approval Advocate -- champions strengths, only flags genuinely critical issues |

## Step 4: Collect Results

- Messages arrive automatically from teammates
- Each agent returns structured findings (see prompt templates)
- Mark each tracking task completed as agents finish
- If an agent has not responded within 5 minutes, proceed with available results

## Step 5: Parse and Cluster Findings

Read `~/.claude/skills/refactor/references/synthesis-algorithm.md` and apply:

1. Parse structured findings from all 3 agents
2. Cluster by (file_path, line_range within 5 lines, related category)
3. Classify each cluster:
   - **Unanimous (3/3)** -> auto-fix
   - **Majority (2/3)** -> post PR comment
   - **Single (1/3, severity >= MEDIUM)** -> post PR comment
   - **Single (1/3, severity LOW)** -> skip

If an agent's response cannot be parsed, include raw text in summary and skip for clustering.

## Step 6: Auto-Fix Unanimous Findings

For each unanimous finding:
1. Read the target file
2. Apply the White Reviewer's (Neutral Judge) suggested fix using `Edit`
3. Run `make check` to validate
4. If `make check` fails: revert with `git checkout -- {file_path}`, downgrade to PR comment
5. If fix > 20 lines: skip auto-fix, downgrade to PR comment

After all fixes:
```bash
git add <fixed files>
git commit -m "fix: address unanimous review findings from /refactor"
git push
```

If `git push` fails, inform user to push manually.

## Step 7: Post Inline PR Comments (Disagreements)

Read `~/.claude/skills/refactor/references/pr-comment-format.md` for templates.

For each majority/significant single finding:
```bash
gh api repos/{OWNER_REPO}/pulls/{PR_NUMBER}/comments \
  -f body='{FORMATTED_COMMENT}' \
  -f path='{FILE_PATH}' \
  -f line={LINE_NUMBER} \
  -f commit_id='{HEAD_SHA}' \
  -f side='RIGHT'
```

If `gh api` fails for a comment, log the error and continue.

## Step 8: Post Summary Comment

Read `~/.claude/skills/refactor/references/pr-comment-format.md` for the summary template.

Post a top-level PR comment:
```bash
gh api repos/{OWNER_REPO}/issues/{PR_NUMBER}/comments \
  -f body='{SUMMARY_BODY}'
```

Include: results table, auto-fixed list, inline comments list, top 3 strengths, reviewer verdicts.

## Step 9: Cleanup Team

```
SendMessage(type="shutdown_request") to each agent
TeamDelete() after all agents shut down
```

## Step 10: Report to User

Print a final summary. No emojis. Include:
- Number of findings auto-fixed
- Number of inline comments posted
- Number of findings skipped
- PR URL

## Error Handling

| Scenario | Action |
|----------|--------|
| No PR found | Exit with message |
| Empty diff | Exit with message |
| Agent timeout (5 min) | Proceed with available results |
| `make check` fails after fix | Revert file, downgrade to PR comment |
| `gh api` comment fails | Log error, continue, note in summary |
| Agent response unparseable | Include raw text in summary, skip for clustering |
| `git push` fails | Inform user to push manually |
