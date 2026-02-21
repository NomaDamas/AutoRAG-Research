# Schema Migration Guide

## Why Migrations Are Needed

SQLAlchemy's `Base.metadata.create_all()` only creates **new** tables — it does NOT alter existing tables to add missing columns. When a user restores a pre-ingested dataset via `pg_restore`, they get the exact schema from the dump, which may lack columns added after the dump was created.

This means Python-level fallbacks (e.g., `rel.score if rel.score is not None else 1`) never execute because the SQL query itself fails with `UndefinedColumn`.

## How It Works

All schema migrations live in a single `_MIGRATION_SQL` constant in `autorag_research/orm/util.py`. This is a PostgreSQL `DO $$ ... END $$` block containing idempotent `ALTER TABLE` statements, each annotated with version number, date, and issue reference.

The `run_migrations()` function executes this SQL block. It is called from three entry points in `DBConnection`:

| Entry Point | When It Runs |
|---|---|
| `create_schema()` | After `create_all()` + BM25 indexes — ensures new schemas are complete |
| `restore_database()` | After `pg_restore` + BM25 indexes — patches restored dumps |
| `get_schema()` | Before returning schema — safety net for externally-restored databases |

## How to Add a New Migration

1. **Add a new annotated SQL block** to `_MIGRATION_SQL` in `autorag_research/orm/util.py`:
   ```sql
   -- ============================================================
   -- Migration vN: <description>
   -- Date: YYYY-MM-DD
   -- Issue: #NNN — <context>
   -- ============================================================
   IF EXISTS (
       SELECT 1 FROM information_schema.tables
       WHERE table_schema = 'public' AND table_name = '<table>'
   ) THEN
       BEGIN
           ALTER TABLE <table>
               ADD COLUMN IF NOT EXISTS <column> <type> DEFAULT <value>;
       EXCEPTION WHEN others THEN PERFORM 1;
       END;
   END IF;
   ```

2. **Update `001-schema.sql`** and **`schema_factory.py`** to include the column in new schemas (so fresh databases have it from the start).

3. **Do not duplicate schema docs**: keep `ai_instructions/db_schema.md` as a redirect-only file that points to `001-schema.sql`.

4. **Keep Python-level fallbacks** as belt-and-suspenders safety (e.g., `value if value is not None else default`).

5. **Add a test** in `tests/autorag_research/orm/test_connection.py` to verify the column exists after migration.

## Design Principles

- **Idempotent**: Every migration uses `IF NOT EXISTS` or equivalent guards so it can run multiple times safely.
- **Exception-safe**: Each `ALTER TABLE` is wrapped in `BEGIN ... EXCEPTION WHEN others THEN PERFORM 1; END` so one failed migration doesn't block others.
- **Annotated**: Each migration block includes version number, date, and issue reference for traceability.
- **Single entry point**: All migrations are in one SQL block, executed by one function, called from three places.
