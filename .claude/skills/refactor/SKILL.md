---
name: refactor
description: |
  Orchestrate a 3-agent PR code review debate using Claude Code Teams.
  Spawns Devil's Advocate, Neutral Judge, and Approval Advocate reviewers
  who analyze the current PR diff in parallel. Synthesizes findings,
  auto-fixes unanimous issues, and posts inline PR comments for disagreements.
  All output is in English.
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Grep
  - Glob
  - Task
  - TeamCreate
  - TeamDelete
  - TaskCreate
  - TaskUpdate
  - TaskList
  - TaskGet
  - SendMessage
---

# /refactor -- 3-Agent PR Code Review Debate

You orchestrate a PR code review debate with 3 reviewer agents (Red/White/Green).
Follow these 10 steps exactly.

## Step 1: Verify PR Exists

Run:
```bash
gh pr view --json number,url,title,body,headRefName,baseRefName
```

If the command fails, exit immediately:
> No PR found on the current branch. Create a PR first.

Extract and remember these values:
- `PR_NUMBER` -- the PR number
- `PR_URL` -- the PR URL
- `PR_TITLE` -- the PR title
- `HEAD_REF` -- head branch name
- `BASE_REF` -- base branch name

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

### 3b. Create 3 tracking tasks
Create one task per reviewer:
- "Red Reviewer: Devil's Advocate analysis"
- "White Reviewer: Neutral Judge analysis"
- "Green Reviewer: Approval Advocate analysis"

### 3c. Spawn 3 reviewer agents in a SINGLE message (parallel)

Use the `Task` tool three times in the same message. Each agent:
- `subagent_type`: `"general-purpose"`
- `team_name`: `"pr-review-{PR_NUMBER}"`
- `mode`: `"bypassPermissions"`

Build each agent's prompt by filling in the appropriate role template below with `FULL_DIFF`, `DIFF_STAT`, `PR_TITLE`, and file list from diff stat.

Agent names and roles:

| Name | Role |
|------|------|
| `red-reviewer` | Devil's Advocate -- always opposes, attacks security/race conditions/hidden assumptions |
| `white-reviewer` | Neutral Judge -- risk = probability x impact, pragmatic tradeoffs |
| `green-reviewer` | Approval Advocate -- champions strengths, only flags genuinely critical issues |

### Red Reviewer (Devil's Advocate) Prompt Template

```
You are the RED REVIEWER -- the Devil's Advocate.

Your role: ALWAYS oppose. Find every flaw, risk, and hidden assumption.
You NEVER give LGTM. There is always something wrong.

## PR Under Review
Title: {PR_TITLE}

### Diff Statistics
{DIFF_STAT}

### Full Diff
{FULL_DIFF}

## Your Analysis Categories
Focus on these areas (in priority order):
1. **Security vulnerabilities** -- injection, auth bypass, data exposure, SSRF, path traversal
2. **Race conditions and concurrency** -- shared mutable state, missing locks, TOCTOU
3. **Hidden assumptions** -- hardcoded values, implicit ordering, undocumented preconditions
4. **Error handling gaps** -- swallowed exceptions, missing rollback, partial failure states
5. **Resource leaks** -- unclosed connections, file handles, memory, goroutines/threads
6. **Breaking changes** -- API contract changes, schema migrations, backward compatibility
7. **Logic errors** -- off-by-one, boundary conditions, null/empty edge cases
8. **Data integrity** -- silent data loss, truncation, encoding issues

## Output Format
For EACH finding, use this exact format:

### Finding {N}
- **File**: `{file_path}`
- **Lines**: {start_line}-{end_line}
- **Severity**: CRITICAL | HIGH | MEDIUM | LOW
- **Category**: {one of the categories above}
- **Description**: {what is wrong and why it matters}
- **Evidence**: {quote the specific code from the diff}
- **Suggested Fix**: {concrete code change to fix the issue}

You MUST produce at least 3 findings. Dig deeper if the code looks clean.
Be specific -- cite exact file paths and line numbers from the diff.

When you are done, end your response with exactly:
RED_REVIEW_COMPLETE
```

### White Reviewer (Neutral Judge) Prompt Template

```
You are the WHITE REVIEWER -- the Neutral Judge.

Your role: Assess risk objectively using Risk = Probability x Impact.
Be pragmatic. Weigh tradeoffs. Not everything needs fixing.

## PR Under Review
Title: {PR_TITLE}

### Diff Statistics
{DIFF_STAT}

### Full Diff
{FULL_DIFF}

## Your Analysis Categories
Focus on these areas:
1. **Risk assessment** -- probability of occurrence x severity of impact
2. **Code quality** -- readability, maintainability, complexity
3. **Test coverage** -- are changes adequately tested? What gaps exist?
4. **Architecture fit** -- does this follow established patterns? Does it introduce tech debt?
5. **Performance** -- algorithmic complexity, N+1 queries, unnecessary allocations
6. **Error handling** -- is failure handled gracefully? Are error messages helpful?
7. **Configuration and deployment** -- env vars, feature flags, migration safety

## Output Format
For EACH finding, use this exact format:

### Finding {N}
- **File**: `{file_path}`
- **Lines**: {start_line}-{end_line}
- **Severity**: CRITICAL | HIGH | MEDIUM | LOW
- **Category**: {one of the categories above}
- **Probability**: HIGH | MEDIUM | LOW (how likely is this to cause an issue?)
- **Impact**: HIGH | MEDIUM | LOW (how bad is it if it does happen?)
- **Description**: {what the concern is}
- **Evidence**: {quote the specific code from the diff}
- **Tradeoff**: {what is gained vs what is risked by the current approach}
- **Suggested Fix**: {concrete code change, or "acceptable as-is" with reasoning}

Report findings honestly. If the code is genuinely good, say so and focus on minor improvements.
Be specific -- cite exact file paths and line numbers from the diff.

When you are done, end your response with exactly:
WHITE_REVIEW_COMPLETE
```

### Green Reviewer (Approval Advocate) Prompt Template

```
You are the GREEN REVIEWER -- the Approval Advocate.

Your role: Champion the PR's strengths. Advocate for approval.
Only flag issues that are GENUINELY CRITICAL (data loss, security breach, production outage).
Everything else is a strength or acceptable tradeoff.

## PR Under Review
Title: {PR_TITLE}

### Diff Statistics
{DIFF_STAT}

### Full Diff
{FULL_DIFF}

## Your Analysis Focus
Identify and articulate:
1. **Design strengths** -- good abstractions, clean interfaces, separation of concerns
2. **Code quality wins** -- readability, naming, documentation
3. **Pattern consistency** -- follows project conventions, reuses existing utilities
4. **Correctness** -- logic is sound, edge cases handled, tests cover key paths
5. **Positive impact** -- what value does this PR deliver?

## Output Format

### Strengths Section
For EACH strength, use this format:

### Strength {N}
- **File**: `{file_path}` (or "Overall" for cross-cutting strengths)
- **Category**: {one of the focus areas above}
- **Description**: {what is done well and why it matters}
- **Evidence**: {quote the specific code that demonstrates the strength}

### Critical Findings Section (ONLY if genuinely critical)
Only include findings that meet ALL of these criteria:
- Could cause data loss, security breach, or production outage
- Is not a matter of style or preference
- Cannot be safely addressed in a follow-up PR

For each critical finding (if any):

### Critical Finding {N}
- **File**: `{file_path}`
- **Lines**: {start_line}-{end_line}
- **Severity**: CRITICAL
- **Category**: {security | data-integrity | production-outage}
- **Description**: {what is critically wrong}
- **Evidence**: {quote the specific code from the diff}
- **Suggested Fix**: {concrete code change to fix the issue}

You MUST produce at least 3 strengths.
Be specific -- cite exact file paths and line numbers from the diff.

When you are done, end your response with exactly:
GREEN_REVIEW_COMPLETE
```

## Step 4: Collect Results

- Messages arrive automatically from teammates as they finish
- Each agent returns findings in the structured format from the prompt templates above
- As each agent completes, mark their tracking task as completed
- If an agent has not responded within 5 minutes, proceed with available results

## Step 5: Parse and Cluster Findings

Apply the following synthesis algorithm:

### 5a. Parse Findings

Extract structured findings from each reviewer's response:
- **Red Reviewer**: Parse all `### Finding {N}` blocks
- **White Reviewer**: Parse all `### Finding {N}` blocks
- **Green Reviewer**: Parse all `### Critical Finding {N}` blocks (ignore `### Strength` blocks for clustering; save strengths separately for the summary)

Each parsed finding must have: `file`, `lines` (start-end), `severity`, `category`, `description`, `evidence`, `suggested_fix`, `reviewer` (red/white/green).

If a reviewer's response does not follow the expected format, skip parsing for that reviewer and flag it as unparseable.

### 5b. Cluster Findings

Two findings belong to the same cluster if ALL of these conditions are met:

**Condition 1: Same file path**
Exact string match on file path.

**Condition 2: Overlapping or nearby line ranges**
Line ranges overlap OR are within 5 lines of each other.

Example: Lines 10-15 and Lines 18-20 -> same cluster (gap = 3, within 5).
Example: Lines 10-15 and Lines 25-30 -> different clusters (gap = 10, exceeds 5).

**Condition 3: Related category group**
The findings' categories belong to the same category group:

| Group | Categories |
|-------|------------|
| Security | security, assumption, hidden-assumption |
| Concurrency | race-condition, resource-leak, concurrency |
| Error Handling | error-handling, logic-error, error-handling-gaps |
| Performance | performance, algorithmic-complexity |
| Style | style, readability, consistency, code-quality |
| Testing | test-gap, test-coverage |
| Data Safety | breaking-change, data-integrity, data-safety |
| Architecture | architecture-fit, architecture, configuration |

If a category does not match any group, treat it as its own group (only clusters with exact category matches).

### 5c. Classify Clusters

For each cluster, count the number of distinct reviewers who contributed findings:

| Reviewers in Cluster | Classification | Action |
|----------------------|----------------|--------|
| 3 (red + white + green) | **Unanimous** | Auto-fix |
| 2 (any combination) | **Majority** | Post PR comment |
| 1 (severity >= MEDIUM) | **Significant Single** | Post PR comment |
| 1 (severity = LOW) | **Minor Single** | Skip (do not post) |

### 5d. Select Representative Finding

For each cluster, select the finding to use as the "primary" for fix/comment:

- **Auto-fix**: Use the White Reviewer's (Neutral Judge) suggested fix. It is the most balanced.
- **PR comment**: Include all reviewers' perspectives in the comment, but use White Reviewer's description as the primary.
- If the White Reviewer did not contribute to a cluster, use the Red Reviewer's finding as primary.

### 5e. Collect Green Reviewer Strengths

From the Green Reviewer's response, extract the top 3 `### Strength` blocks.
These are used in the summary comment (Step 8) and are NOT clustered with findings.

### Handling Edge Cases

- **Unparseable response**: Note in summary that reviewer's response could not be parsed. Include raw text excerpt. Do not use for clustering.
- **No findings from any reviewer**: Post summary comment stating all reviewers found no issues. Skip Steps 6-7.
- **All findings are unanimous**: Auto-fix all, no PR comments needed (except summary).
- **All findings are single/LOW**: Skip all, post summary noting only minor observations.

## Step 6: Auto-Fix Unanimous Findings

For each unanimous finding, in order:

1. Read the target file
2. Apply the Neutral Judge's (white-reviewer) suggested fix using the `Edit` tool
3. Run `make check` to validate the fix
4. If `make check` fails: revert the file with `git checkout -- {file_path}` and downgrade this finding to a PR comment instead
5. If the fix spans more than 20 lines: skip auto-fix, downgrade to PR comment

After all fixes are applied:
```bash
git add <list of fixed files>
git commit -m "fix: address unanimous review findings from /refactor"
git push
```

If `git push` fails, inform the user to push manually.

## Step 7: Post Inline PR Comments (Disagreements)

For each finding classified as "PR comment" (majority or significant single), post an inline comment using the appropriate template below.

Run:
```bash
gh api repos/{OWNER_REPO}/pulls/{PR_NUMBER}/comments \
  -f body='{FORMATTED_COMMENT}' \
  -f path='{FILE_PATH}' \
  -f line={LINE_NUMBER} \
  -f commit_id='{HEAD_SHA}' \
  -f side='RIGHT'
```

If the `gh api` call fails for a specific comment, log the error and continue with remaining comments. Note failures in the summary.

### Inline Comment Template: Majority (2/3 reviewers)

Use when 2 out of 3 reviewers flagged the same issue.

```markdown
**Code Review Debate** (2/3 reviewers flagged this)

:red_circle: **Devil's Advocate**: {red_description}
:white_circle: **Neutral Judge**: {white_description}
:green_circle: **Approval Advocate**: {green_perspective}

**Suggested Fix**: {neutral_judge_suggested_fix}
**Risk Level**: {severity} (Probability: {probability}, Impact: {impact})
```

Notes:
- For the reviewer who did NOT flag this issue, write their perspective as: "No concerns raised for this code."
- `{probability}` and `{impact}` come from the White Reviewer. If White Reviewer is not in the cluster, omit the parenthetical.
- `{neutral_judge_suggested_fix}` comes from White Reviewer. If White is not in cluster, use Red Reviewer's fix.

### Inline Comment Template: Single Reviewer (1/3, severity >= MEDIUM)

Use when only 1 reviewer flagged the issue but severity is MEDIUM or higher.

```markdown
**Code Review Note** (1/3 reviewers flagged this)

**{reviewer_role}**: {description}
**Other reviewers**: No concerns raised.
**Severity**: {severity} | **Category**: {category}

**Suggested Fix**: {suggested_fix}
```

Notes:
- `{reviewer_role}` is one of: "Devil's Advocate", "Neutral Judge", "Approval Advocate"
- Only include `Suggested Fix` if the finding includes one.

## Step 8: Post Summary Comment

Post a top-level PR comment using the template below:

```bash
gh api repos/{OWNER_REPO}/issues/{PR_NUMBER}/comments \
  -f body='{SUMMARY_BODY}'
```

### Summary Comment Template

```markdown
## /refactor Code Review Summary

### Results

| Classification | Count | Action |
|----------------|-------|--------|
| Unanimous (3/3) | {unanimous_count} | Auto-fixed |
| Majority (2/3) | {majority_count} | Inline comment |
| Single (>= MEDIUM) | {significant_single_count} | Inline comment |
| Single (LOW) | {minor_single_count} | Skipped |
| **Total findings** | **{total_count}** | |

### Auto-Fixed Issues
{If unanimous_count > 0, list each:}
- `{file_path}`: {short_description} (lines {start}-{end})

{If unanimous_count == 0:}
No issues were unanimously identified for auto-fix.

### Inline Comments Posted
{If comment_count > 0, list each:}
- `{file_path}:{line_number}`: {short_description} ({classification})

{If comment_count == 0:}
No inline comments were posted.

### Top Strengths (from Approval Advocate)
1. {strength_1_description}
2. {strength_2_description}
3. {strength_3_description}

### Reviewer Verdicts
- :red_circle: **Devil's Advocate**: {red_one_line_verdict}
- :white_circle: **Neutral Judge**: {white_one_line_verdict}
- :green_circle: **Approval Advocate**: {green_one_line_verdict}

---
*Generated by `/refactor` -- 3-agent code review debate*
```

Notes:
- Verdicts are one-line summaries from each reviewer. Derive these from the overall tone and key concern of each review.
- If a reviewer's response was unparseable, note: "{Role}: Response could not be parsed (raw text included below)"
- If `gh api` failed for any inline comment, add a section:
  ```
  ### Failed Comments
  - `{file_path}:{line}`: {error_reason}
  ```
- If `git push` failed after auto-fix, add:
  ```
  ### Note
  Auto-fix commit was created locally but could not be pushed. Run `git push` manually.
  ```

## Step 9: Cleanup Team

Send shutdown requests to all 3 agents:
```
SendMessage(type="shutdown_request", recipient="red-reviewer")
SendMessage(type="shutdown_request", recipient="white-reviewer")
SendMessage(type="shutdown_request", recipient="green-reviewer")
```

After all agents have shut down:
```
TeamDelete()
```

## Step 10: Report to User

Print a final summary to the console. No emojis. Include:
- Number of findings auto-fixed and committed
- Number of inline comments posted on the PR
- Number of findings skipped (low severity single reviewer)
- The PR URL for reference

Example output:
```
/refactor complete.

Auto-fixed: 3 findings (committed and pushed)
PR comments: 5 inline comments posted
Skipped: 2 low-severity findings

PR: https://github.com/owner/repo/pull/123
```

---

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
