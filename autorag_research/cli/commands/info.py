"""info command - Show detailed information about a database schema."""

import hydra
from omegaconf import DictConfig

from autorag_research.cli.utils import get_schema_info


@hydra.main(version_base=None, config_name="info_config", config_path=None)
def info(cfg: DictConfig) -> None:
    """Show detailed information about a database schema."""
    database = cfg.get("database", "")

    if not database:
        print("Error: database parameter is required")
        print("Usage: autorag-research info database=<schema_name>")
        return

    print(f"\nSchema Information: {database}")
    print("=" * 60)

    try:
        schema_info = get_schema_info(cfg, database)

        if not schema_info["tables"]:
            print(f"  Schema '{database}' not found or has no tables.")
            return

        print(f"\nTables in schema '{database}':")
        print("-" * 60)

        total_rows = 0
        for table_name, table_info in sorted(schema_info["tables"].items()):
            row_count = table_info["row_count"]
            total_rows += row_count
            print(f"  {table_name:<30} {row_count:>10,} rows")

        print("-" * 60)
        print(f"  {'Total':<30} {total_rows:>10,} rows")

        print("\nKey Statistics:")
        print("-" * 60)

        key_tables = ["query", "chunk", "document", "page", "retrieval_relation"]
        for table in key_tables:
            if table in schema_info["tables"]:
                count = schema_info["tables"][table]["row_count"]
                print(f"  {table.replace('_', ' ').title():<20} {count:>10,}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("  1. PostgreSQL is running")
        print("  2. Database credentials are correct")
        print("  3. The schema exists")
