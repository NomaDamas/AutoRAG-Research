#!/usr/bin/env python3
import json
import re
import shlex
import sys

WRITEY = [
    "insert",
    "update",
    "delete",
    "truncate",
    "alter",
    "create",
    "grant",
    "revoke",
    "vacuum",
    "analyze",
    "comment",
    "do",
    "call",
    "copy",
    "refresh",
    "reindex",
    "cluster",
    "lock",
    "set",
    "reset",
    "begin",
    "commit",
    "rollback",
    "savepoint",
    "prepare",
    "execute",
]


def emit(decision, reason):
    sys.stdout.write(
        json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": decision,
                "permissionDecisionReason": reason,
            }
        })
    )


def tokens(cmd):
    try:
        return shlex.split(cmd)
    except Exception:
        return cmd.split()


def env_ref(cmd):
    return bool(re.search(r'(^|[ \t"\'/])\.env(\.[^ \t"\'/]+)?($|[ \t"\'/])', cmd))


def is_psql(ts):
    i = 0
    while i < len(ts) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", ts[i]):
        i += 1
    if i < len(ts) and (ts[i] == "psql" or ts[i].endswith("/psql") or ts[i].endswith("\\psql.exe")):
        return i
    return None


def extract_sql(ts, psql_i):
    i = psql_i + 1
    while i < len(ts):
        t = ts[i]
        if t in ("-c", "--command"):
            return ts[i + 1] if i + 1 < len(ts) else ""
        if t.startswith("--command="):
            return t.split("=", 1)[1]
        if t in ("-f", "--file"):
            return None
        i += 1
    return None


def classify(sql):
    s = sql.strip()
    low = s.lower()

    if re.search(r"\bdrop\b", low):
        return "deny", "psql: DROP denied"

    starts = bool(re.match(r"^(select|with)\b", low))
    has_select = bool(re.search(r"\bselect\b", low))
    has_into = bool(re.search(r"\binto\b", low))
    has_writey = any(re.search(rf"\b{k}\b", low) for k in WRITEY)

    if starts and has_select and (not has_into) and (not has_writey):
        return "allow", "psql: read-only query auto-allowed"

    return "ask", "psql: non-SELECT query requires confirmation"


def main():
    payload = json.load(sys.stdin)
    if payload.get("tool_name") != "Bash":
        return

    cmd = payload.get("tool_input", {}).get("command", "")
    if not isinstance(cmd, str) or not cmd.strip():
        return

    if env_ref(cmd):
        emit("deny", "sensitive file access denied")
        return

    ts = tokens(cmd)
    p = is_psql(ts)
    if p is not None:
        sql = extract_sql(ts, p)
        if sql is None:
            emit("ask", "psql: interactive/file execution requires confirmation")
            return
        decision, reason = classify(sql)
        emit(decision, reason)
        return


if __name__ == "__main__":
    main()
