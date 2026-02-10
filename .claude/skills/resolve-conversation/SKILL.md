---
name: resolve-conversation
description: |
  Process [APPROVE] and [IGNORE] replies on /refactor review threads.
  Applies approved fixes to the codebase, resolves all responded threads on GitHub,
  commits and pushes changes. Sequential single-agent workflow.
  All output is in English.
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Grep
  - Glob
---

# /resolve-conversation -- Process /refactor Review Thread Replies

You process human replies on /refactor inline PR comments. Follow these 8 steps exactly.

## Step 1: Verify PR Exists

Run:
```bash
gh pr view --json number,url,title,headRefName,baseRefName
```

If the command fails, exit immediately:
> No PR found on the current branch. Create a PR first.

Extract and remember these values:
- `PR_NUMBER` -- the PR number
- `PR_URL` -- the PR URL
- `PR_TITLE` -- the PR title
- `HEAD_REF` -- head branch name
- `BASE_REF` -- base branch name

## Step 2: Get Repository Identity

Run these commands in parallel:
```bash
git remote get-url origin
git rev-parse HEAD
```

Parse the remote URL to extract `OWNER/REPO` (handle both HTTPS and SSH formats).
Store:
- `OWNER_REPO` -- e.g. `owner/repo`
- `HEAD_SHA` -- current HEAD commit SHA

## Step 3: Fetch All Review Threads via GraphQL

Run a single GraphQL query to fetch all review threads with their comments:

```bash
gh api graphql -f query='
query($owner: String!, $repo: String!, $pr: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $pr) {
      reviewThreads(first: 100) {
        nodes {
          id
          isResolved
          comments(first: 50) {
            nodes {
              body
              author {
                login
              }
            }
          }
          path
          line
          startLine
        }
      }
    }
  }
}' -F owner='{OWNER}' -F repo='{REPO}' -F pr={PR_NUMBER}
```

Replace `{OWNER}`, `{REPO}`, and `{PR_NUMBER}` with the values from Steps 1-2.

If the query fails, exit immediately:
> GraphQL query failed. Check your GitHub authentication and permissions.

## Step 4: Identify Actionable Threads

From the GraphQL response, filter threads that meet ALL of these criteria:

1. **Thread is unresolved** (`isResolved == false`)
2. **First comment is from /refactor** -- the first comment body contains either:
   - `**Code Review Debate**` (majority finding)
   - `**Code Review Note**` (single reviewer finding)
3. **A reply exists with a human action** -- any comment after the first contains:
   - `[APPROVE]` -- human approves the suggested fix
   - `[IGNORE]` -- human wants to ignore this finding

**Precedence rule:** If a thread has both `[APPROVE]` and `[IGNORE]` in different replies, `[APPROVE]` takes precedence.

Classify each actionable thread as either `APPROVE` or `IGNORE`.

If no actionable threads are found, exit immediately:
> No actionable threads found. Reply with [APPROVE] or [IGNORE] on /refactor review comments first.

## Step 5: Process APPROVE Threads

For each thread classified as `APPROVE`, in order:

### 5a. Extract the Suggested Fix

From the first comment (the /refactor comment), extract the text after `**Suggested Fix**:`.
This is the code change to apply.

Also extract:
- `path` -- the file path from the thread
- `line` / `startLine` -- the line number(s) from the thread

### 5b. Check File Exists

If the target file no longer exists:
- Note in report: "File `{path}` no longer exists, skipping fix"
- Still mark this thread for resolution in Step 6
- Continue to next thread

### 5c. Apply the Fix

1. Read the target file using the `Read` tool
2. Apply the suggested fix using the `Edit` tool
3. Run `make check` to validate

### 5d. Validate

If `make check` fails:
- Revert the file: `git checkout -- {path}`
- Note in report: "Fix for `{path}:{line}` failed validation, reverted"
- Still mark this thread for resolution in Step 6

If `make check` passes:
- Note in report: "Fix applied to `{path}:{line}`"

## Step 6: Resolve All Actionable Threads

For EVERY actionable thread (both APPROVE and IGNORE), resolve it via GraphQL mutation:

```bash
gh api graphql -f query='
mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) {
    thread {
      isResolved
    }
  }
}' -f threadId='{THREAD_ID}'
```

Replace `{THREAD_ID}` with the thread's `id` from the GraphQL response.

Process all threads. If a resolve mutation fails for a specific thread:
- Log the error
- Continue with remaining threads
- Note in report: "Failed to resolve thread for `{path}:{line}`"

## Step 7: Commit and Push

If any fixes were successfully applied (not reverted) in Step 5:

```bash
git add <list of fixed files>
git commit -m "fix: apply approved review suggestions from /resolve-conversation"
git push
```

If `git push` fails:
- Inform user: "Commit created locally but push failed. Run `git push` manually."

If no fixes were applied (all were IGNORE or all APPROVE fixes failed validation):
- Skip this step entirely

## Step 8: Report to User

Print a final summary to the console. No emojis. Include:

```
/resolve-conversation complete.

Threads processed: {total_actionable}
  APPROVE: {approve_count} ({fixes_applied} applied, {fixes_failed} failed validation)
  IGNORE: {ignore_count}
Threads resolved: {resolved_count} ({resolve_failed} failed to resolve)

{If fixes_applied > 0:}
Changes committed and pushed.

{If fixes_applied > 0 and push_failed:}
Changes committed locally. Run `git push` manually.

PR: {PR_URL}
```

If any fixes failed validation, list them:
```
Failed fixes:
  - {path}:{line}: {reason}
```

If any files no longer exist, list them:
```
Skipped (file missing):
  - {path}
```

---

## Error Handling

| Scenario | Action |
|----------|--------|
| No PR found | Exit with message |
| GraphQL query fails | Exit with message |
| No actionable threads | Exit with message |
| `make check` fails after fix | Revert file, still resolve thread, note in report |
| Resolve mutation fails | Log error, continue with remaining |
| `git push` fails | Inform user to push manually |
| File no longer exists | Skip fix, still resolve, note in report |
