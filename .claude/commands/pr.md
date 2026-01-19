---
allowed-tools: Bash(git status), Bash(git diff:*), Bash(git log:*), Bash(git branch:*), Bash(gh pr:*)
description: Create a pull request to main branch
---

Create a pull request from the current branch to main:

1. Check current branch with `git branch --show-current`
2. Ensure all changes are committed (run `git status`)
3. Review the commits that will be included: `git log main..HEAD --oneline`
4. Review the diff against main: `git diff main...HEAD --stat`
5. Create PR using GitHub CLI:

```bash
gh pr create --base main --title "<descriptive title>" --body "<summary of changes>"
```

PR body should include:
- Summary of what was changed
- Any relevant context
- Testing notes if applicable

After creating, output the PR URL.
