import os
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database connection configuration for reporting service."""

    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "postgres"

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables."""
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        )

    def get_duckdb_connection_string(self, db_name: str) -> str:
        """Generate DuckDB ATTACH connection string for PostgreSQL.

        Warning:
            The returned string contains credentials in plain text.
            Do not log or display this value.
        """
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{db_name}"
