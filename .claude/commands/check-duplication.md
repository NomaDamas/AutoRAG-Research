Scan recently modified Python files for utility function duplication.

## Instructions

1. Find recently modified Python files using `git diff --name-only HEAD` via Bash. If the working tree is clean, fall back to `git diff --name-only HEAD~1` to check the last commit. Only include `.py` files under `autorag_research/` (exclude `util.py` itself and test files).

2. For each file found, search for these duplication indicators using Grep:

   **Pattern 1: Manual asyncio.Semaphore** (should use `run_with_concurrency_limit`)
   - Search for `asyncio.Semaphore` in the file

   **Pattern 2: Manual asyncio.to_thread** (should use `to_async_func`)
   - Search for `asyncio.to_thread` in the file

   **Pattern 3: Manual min/max normalization** (should use `normalize_minmax` or family)
   - Search for patterns like `min(scores)` or `max(scores)` combined with division

   **Pattern 4: BytesIO + image.save** (should use `pil_image_to_bytes`)
   - Search for `BytesIO()` combined with `.save(` in proximity

   **Pattern 5: Base64 data URI parsing** (should use `extract_image_from_data_uri`)
   - Search for `base64.b64decode` outside of `util.py`

   **Pattern 6: Direct UoW repository calls from pipeline files** (should use service methods)
   - In files under `pipelines/`, search for `uow.queries.get_by_id` or `uow.chunks.bm25_search` or `uow.chunks.vector_search`

   **Pattern 7: Manual token summation** (should use `aggregate_token_usage`)
   - Search for `["token_usage"]` combined with manual summation patterns

3. Report findings in this format:
   ```
   ## Duplication Check Results

   ### <filename>
   - **Line X**: Found `<pattern>` — use `<replacement>` from `<location>` instead

   ### Summary
   - Files scanned: N
   - Potential duplications found: N
   ```

4. If no duplications are found, report:
   ```
   ## Duplication Check Results

   No duplications detected. All scanned files properly reuse existing utilities.
   Files scanned: N
   ```

5. Reference `ai_instructions/utility_reference.md` for the full catalog of available utilities and services.
