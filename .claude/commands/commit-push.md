---
allowed-tools: Bash(git add:*), Bash(git status), Bash(git diff:*), Bash(git commit:*), Bash(git push:*), Bash(git log:*), Read
description: Stage changes, commit with descriptive message, and push to current branch
---

Perform a checkpoint commit and push to the current branch:

1. Run `git status` to review both modified and untracked files
2. For untracked files: Check file contents to ensure they don't contain private info (API keys, tokens, passwords, credentials). Skip files like `.env`, `*credentials*`, `*secret*`, `*token*` files.
3. Stage all appropriate files including new files with `git add` (exclude sensitive files)
4. Run `git diff --staged` to review what will be committed
5. Create a clear, descriptive commit message based on the actual changes
6. Push to the current remote branch

**Files to NEVER add:**
- `.env`, `.env.*`
- Files containing API keys, tokens, passwords, or credentials
- `*secret*`, `*credential*`, `*token*` files

Commit message format:
- Use imperative mood (e.g., "Add feature" not "Added feature")
- First line should be concise summary (max 72 chars)
- Add body if changes are complex

This is a checkpoint commit for tracking progress.
