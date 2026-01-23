#!/usr/bin/env python
"""Build a minimal Lucene index for BM25 integration tests."""

import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
CORPUS_PATH = SCRIPT_DIR / "bm25_test_corpus"
INDEX_PATH = SCRIPT_DIR / "bm25_test_index"


def build_index():
    if INDEX_PATH.exists():
        shutil.rmtree(INDEX_PATH)

    cmd = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(CORPUS_PATH),
        "--index",
        str(INDEX_PATH),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        "1",
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]

    print(f"Building index at {INDEX_PATH}...")
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(f"Index built successfully at {INDEX_PATH}")


if __name__ == "__main__":
    build_index()
