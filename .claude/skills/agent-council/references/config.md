# Configure members

Edit `council.config.yaml` to set chairman and members:

```yaml
council:
  chairman:
    role: "auto"
  members:
    - name: claude
      command: "claude -p"
      emoji: "🧠"
      color: "CYAN"
    - name: codex
      command: "codex exec"
      emoji: "🤖"
      color: "BLUE"
    - name: gemini
      command: "gemini"
      emoji: "💎"
      color: "GREEN"
```

Add custom members by appending entries to `members`:

- Use a stable `name` (lowercase, short).
- Set `command` to a runnable CLI invocation.
- Provide `emoji` and `color` for readability (optional but recommended).
- Note that the installer filters members to detected CLIs only when it first generates `council.config.yaml`. After that, missing CLIs are not auto-removed and will report `missing_cli` at runtime; remove unavailable members or install the CLI before running.
