#!/usr/bin/env bash
# Simple wrapper around pg_dump.

set -euo pipefail

usage() {
	echo "Usage: $0 --host HOST --port PORT --user USER --password PASS --dbname NAME --output FILE [-- EXTRA_ARGS...]" >&2
	exit 1
}

HOST="${PGHOST:-}"
PORT="${PGPORT:-5432}"
USER="${PGUSER:-}"
PASSWORD="${PGPASSWORD:-}"
DBNAME=""
OUTPUT=""

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
		--)
			shift
			break ;;
		*)
			echo "Unknown option: $1" >&2
			usage ;;
	esac
done

if [[ -z "$DBNAME" || -z "$OUTPUT" ]]; then
	echo "--dbname and --output are required" >&2
	usage
fi

OUTPUT_DIR="$(dirname "$OUTPUT")"
mkdir -p "$OUTPUT_DIR"

CMD=("pg_dump" "--format=custom" "--file=$OUTPUT" "--dbname=$DBNAME")
[[ -n "$HOST" ]] && CMD+=("--host=$HOST")
[[ -n "$PORT" ]] && CMD+=("--port=$PORT")
[[ -n "$USER" ]] && CMD+=("--username=$USER")

CMD+=("$@")

if [[ -n "$PASSWORD" ]]; then
	export PGPASSWORD="$PASSWORD"
fi

"${CMD[@]}"

# pg_dump 성공 시 업로드 진행
if [[ -f "$OUTPUT" ]]; then
	echo "pg_dump completed successfully: $OUTPUT"
	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	KEY_NAME="$(basename "$OUTPUT")"
	python "$SCRIPT_DIR/upload_postgres.py" --file-path "$OUTPUT" --key-name "$KEY_NAME"
	echo "Upload completed successfully"
else
	echo "Error: dump file not created" >&2
	exit 1
fi
