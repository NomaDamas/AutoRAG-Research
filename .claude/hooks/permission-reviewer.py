# ruff: noqa: S110, S603, S607, TRY300
#!/usr/bin/env python3
"""
PermissionRequest hook: Sonnet API to auto-judge gray-zone commands.
- Only receives requests that didn't match allow/deny rules
- Default stance: APPROVE (dev context, most operations are safe)
- Only deny genuine, irreversible risks
Requires ANTHROPIC_API_KEY environment variable.
"""

import json
import os
import subprocess
import sys


def load_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        result = subprocess.run(
            ["zsh", "-lc", "echo $ANTHROPIC_API_KEY"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        key = result.stdout.strip()
        if key:
            return key
    except Exception:
        pass
    return None


SYSTEM_PROMPT = """\
You are a permissive permission reviewer for a developer's coding agent.
The request below already passed rule-based allow/deny filters without matching — \
it's an edge case that needs your contextual judgment.

Your default stance is to APPROVE. Developers move fast and most operations are \
safe in a dev context. Only deny something if it poses a genuine, concrete risk — \
not a theoretical one.

Approve liberally:
- Deleting files/dirs is almost always fine (build artifacts, caches, temp files, \
even source files during refactoring)
- Unfamiliar commands or scripts are fine if they look like normal dev work
- Git operations, package management, builds, tests — just let them through
- When in doubt, approve. The developer can always undo.

Only deny when the risk is real and irreversible:
- Destroying the home directory, root filesystem, or OS-level system files
- Leaking credentials or secrets to external services
- Publishing packages or deploying to production
- Commands that clearly exfiltrate data to unknown destinations

Respond ONLY with a JSON object: {"ok": true} to approve or {"ok": false, "reason": "..."} to deny.
No other text.\
"""


def call_sonnet(request_info, api_key):
    """Call Anthropic API via curl (no external dependencies)."""
    payload = json.dumps({
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 150,
        "messages": [{"role": "user", "content": "Review this request:\n\n" + request_info}],
        "system": SYSTEM_PROMPT,
    })
    try:
        result = subprocess.run(
            [
                "curl",
                "-s",
                "--max-time",
                "15",
                "https://api.anthropic.com/v1/messages",
                "-H",
                "content-type: application/json",
                "-H",
                "x-api-key: " + api_key,
                "-H",
                "anthropic-version: 2023-06-01",
                "-d",
                payload,
            ],
            capture_output=True,
            text=True,
            timeout=18,
        )
        resp = json.loads(result.stdout)
        text = resp["content"][0]["text"]
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return {"ok": True}
    except Exception:
        return {"ok": True}


def emit(behavior, message=""):
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PermissionRequest",
            "decision": {"behavior": behavior},
        }
    }
    if behavior == "deny" and message:
        output["hookSpecificOutput"]["decision"]["message"] = message
    sys.stdout.write(json.dumps(output))


def main():
    payload = json.load(sys.stdin)
    tool_name = payload.get("tool_name", "unknown")
    tool_input = payload.get("tool_input", {})

    # matcher가 Bash|Edit|Write만 통과시키므로 여기엔 해당 도구만 옴
    if tool_name == "Bash":
        request_info = "Tool: Bash\nCommand: " + tool_input.get("command", "")
    elif tool_name in ("Edit", "Write"):
        request_info = "Tool: " + tool_name + "\nFile: " + tool_input.get("file_path", "")
    else:
        request_info = "Tool: " + tool_name + "\nInput: " + json.dumps(tool_input, ensure_ascii=False)[:500]

    api_key = load_api_key()
    if not api_key:
        emit("allow")
        return

    result = call_sonnet(request_info, api_key)

    if result.get("ok", True):
        emit("allow")
    else:
        emit("deny", result.get("reason", "Sonnet denied this request"))


if __name__ == "__main__":
    main()
