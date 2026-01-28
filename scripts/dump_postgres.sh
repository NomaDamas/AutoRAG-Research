#!/usr/bin/env bash
# Dump a PostgreSQL database and optionally upload to HuggingFace Hub.
#
# Usage:
#   # Dump only
#   ./dump_postgres.sh --dbname mydb --output ./mydb.dump
#
#   # Dump and upload to HuggingFace Hub
#   ./dump_postgres.sh --dbname scifact_openai-small --output ./scifact.dump \
#       --ingestor beir --dataset scifact --embedding openai-small

set -euo pipefail

usage() {
	cat <<EOF >&2
Usage: $0 --dbname NAME --output FILE [OPTIONS] [-- EXTRA_PG_DUMP_ARGS...]

Required:
  --dbname NAME       Database name to dump
  --output FILE       Output dump file path

PostgreSQL connection (or use environment variables):
  --host HOST         PostgreSQL host (default: \$PGHOST or localhost)
  --port PORT         PostgreSQL port (default: \$PGPORT or 5432)
  --user USER         PostgreSQL user (default: \$PGUSER)
  --password PASS     PostgreSQL password (default: \$PGPASSWORD)

HuggingFace upload (all three required to enable upload):
  --ingestor NAME     Ingestor family (beir, mrtydi, ragbench, etc.)
  --dataset NAME      Dataset subset name (scifact, nfcorpus, etc.)
  --embedding NAME    Embedding model name (openai-small, colpali-v1.2, etc.)

Examples:
  # Dump only
  $0 --dbname scifact_openai --output ./scifact.dump

  # Dump and upload
  $0 --dbname scifact_openai --output ./scifact.dump \\
      --ingestor beir --dataset scifact --embedding openai-small
EOF
	exit 1
}

HOST="${PGHOST:-localhost}"
PORT="${PGPORT:-5432}"
USER="${PGUSER:-}"
PASSWORD="${PGPASSWORD:-}"
DBNAME=""
OUTPUT=""
INGESTOR=""
DATASET=""
EMBEDDING=""

while [[ $# -gt 0 ]]; do
	case "$1" in
		--host)
			HOST="$2"; shift 2 ;;
		--port)
			PORT="$2"; shift 2 ;;
		--user)
			USER="$2"; shift 2 ;;
		--password)
			PASSWORD="$2"; shift 2 ;;
		--dbname)
			DBNAME="$2"; shift 2 ;;
		--output)
			OUTPUT="$2"; shift 2 ;;
		--ingestor)
			INGESTOR="$2"; shift 2 ;;
		--dataset)
			DATASET="$2"; shift 2 ;;
		--embedding)
			EMBEDDING="$2"; shift 2 ;;
		--)
			shift
			break ;;
		-h|--help)
			usage ;;
		*)
			echo "Unknown option: $1" >&2
			usage ;;
	esac
done

if [[ -z "$DBNAME" || -z "$OUTPUT" ]]; then
	echo "Error: --dbname and --output are required" >&2
	usage
fi

OUTPUT_DIR="$(dirname "$OUTPUT")"
mkdir -p "$OUTPUT_DIR"

# Build pg_dump command
CMD=("pg_dump" "--format=custom" "--file=$OUTPUT" "--dbname=$DBNAME")
[[ -n "$HOST" ]] && CMD+=("--host=$HOST")
[[ -n "$PORT" ]] && CMD+=("--port=$PORT")
[[ -n "$USER" ]] && CMD+=("--username=$USER")

# Add any extra args passed after --
CMD+=("$@")

if [[ -n "$PASSWORD" ]]; then
	export PGPASSWORD="$PASSWORD"
fi

echo "Dumping database '$DBNAME' to '$OUTPUT'..."
"${CMD[@]}"

if [[ ! -f "$OUTPUT" ]]; then
	echo "Error: dump file not created" >&2
	exit 1
fi

echo "pg_dump completed successfully: $OUTPUT"

# Upload to HuggingFace Hub if all upload params provided
if [[ -n "$INGESTOR" && -n "$DATASET" && -n "$EMBEDDING" ]]; then
	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	echo "Uploading to HuggingFace Hub: NomaDamas/${INGESTOR}-dumps/${DATASET}_${EMBEDDING}.dump"
	uv run python "$SCRIPT_DIR/upload_postgres.py" \
		--file-path "$OUTPUT" \
		--ingestor "$INGESTOR" \
		--dataset "$DATASET" \
		--embedding-model "$EMBEDDING"
	echo "Upload completed successfully"
elif [[ -n "$INGESTOR" || -n "$DATASET" || -n "$EMBEDDING" ]]; then
	echo "Warning: To upload, all three are required: --ingestor, --dataset, --embedding" >&2
	echo "Skipping upload." >&2
fi
