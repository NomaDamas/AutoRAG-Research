# Installation

This guide covers the installation of AutoRAG-Research.

## Requirements

Before installing AutoRAG-Research, ensure you have the following prerequisites:

- **Python 3.10+**
- **PostgreSQL 18** client tools (required for database operations)

### PostgreSQL Client Tools

AutoRAG-Research requires PostgreSQL 18 client tools, specifically `pg_restore-18`, for database restore functionality.

Check the installation of PostgreSQL 18 version in their official [documentation](https://www.postgresql.org/download/).

Verify the installation:

```bash
pg_restore-18 --version
```

## Package Installation

Install AutoRAG-Research using pip or uv:

=== "pip"

    ```bash
    pip install AutoRAG-Research
    ```

=== "uv"

    ```bash
    uv pip install AutoRAG-Research
    ```

Highly recommend to use `uv` for better dependency resolution.

## Development Installation

For development, clone the repository and install with development dependencies:

```bash
git clone https://github.com/vkehfdl1/AutoRAG-Research.git
cd AutoRAG-Research
uv sync --all-extras
```
