# Requirements

- Install and authenticate the CLIs listed under `council.members` in `council.config.yaml`.
- Note that the installer filters members to detected CLIs only on initial config generation; afterward, missing CLIs show as `missing_cli` in status output.
- Install Node.js (plugins cannot bundle or auto-install it).
- Verify each member’s base command exists (for example, `command -v <binary>` or `<binary> --version`).
