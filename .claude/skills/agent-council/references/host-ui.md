# Host UI Checklist Guidance

Use these steps only when a host agent UI supports native checklist updates.

## Checklist flow

1. Run `council.sh wait` once to seed the cursor and get the JSON payload.
2. Update the host's native checklist UI using the payload (if provided).
3. Repeat `wait` until progress changes, then update the UI again.
4. Finish with `results` and `clean`.

## Behavior notes

- Do not run a blocking wait before the first checklist update, or the Plan UI may not appear.
- Keep exactly one in_progress item while work remains.
- Preserve existing checklist items and append the [Council] section.
- Avoid a long while loop in a single tool call; update after each wait return.
- Use `--bucket 1` for per-member updates when needed.
