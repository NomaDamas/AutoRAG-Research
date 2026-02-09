Scan recently modified Python files for utility function duplication.

## Instructions

1. **Read the catalog**: Read `ai_instructions/utility_reference.md` — specifically the "Common Duplication Patterns" section. This is the single source of truth for what patterns to detect and what replacements to suggest. Do NOT use a hard-coded list of patterns.

2. **Find target files**: Use `git diff --name-only HEAD` via Bash to find recently modified `.py` files. If the working tree is clean, fall back to `git diff --name-only HEAD~1`. Only include files under `autorag_research/` (exclude `util.py` itself and test files).

3. **Scan for duplication**: For each file found, check whether it contains code that reimplements any pattern listed in the "Common Duplication Patterns" section of the catalog. Use Grep to search for the key indicators shown in each pattern's "BAD" code block (e.g., the distinctive function calls, import patterns, or code shapes that signal reimplementation).

4. **Report findings** in this format:
   ```
   ## Duplication Check Results

   ### <filename>
   - **Line X**: Found `<pattern>` — use `<replacement>` from `<location>` instead

   ### Summary
   - Files scanned: N
   - Potential duplications found: N
   ```

5. If no duplications are found, report:
   ```
   ## Duplication Check Results

   No duplications detected. All scanned files properly reuse existing utilities.
   Files scanned: N
   ```
