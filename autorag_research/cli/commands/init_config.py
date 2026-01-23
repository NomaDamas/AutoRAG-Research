"""init-config command - Download default configuration files."""

import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/vkehfdl1/AutoRAG-Research/main/configs"

# Configuration files to download (datasets are handled via CLI, not YAML)
CONFIG_FILES = [
    "db/default.yaml",
    "experiment.yaml",
    "pipelines/bm25_baseline.yaml",
    "pipelines/vector_search.yaml",
    "pipelines/hybrid_search.yaml",
    "pipelines/naive_rag.yaml",
    "metrics/recall.yaml",
    "metrics/recall_5.yaml",
    "metrics/precision.yaml",
    "metrics/ndcg.yaml",
    "metrics/mrr.yaml",
    "metrics/f1.yaml",
    "metrics/rouge.yaml",
]


def init_config() -> None:
    """Download default configuration files to ./configs/ directory."""
    config_dir = Path.cwd() / "configs"

    print(f"Initializing configuration files in {config_dir}")

    # Create subdirectories (datasets are handled via CLI, not YAML)
    (config_dir / "db").mkdir(parents=True, exist_ok=True)
    (config_dir / "pipelines").mkdir(parents=True, exist_ok=True)
    (config_dir / "metrics").mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0

    with httpx.Client(timeout=30.0) as client:
        for file_path in CONFIG_FILES:
            local_path = config_dir / file_path
            url = f"{GITHUB_RAW_BASE}/{file_path}"

            if local_path.exists():
                print(f"  [skip] {file_path} (already exists)")
                skipped += 1
                continue

            try:
                response = client.get(url)
                if response.status_code == 200:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    local_path.write_text(response.text)
                    print(f"  [ok] {file_path}")
                    downloaded += 1
                elif response.status_code == 404:
                    create_default_config(local_path, file_path)
                    print(f"  [created] {file_path} (default)")
                    downloaded += 1
                else:
                    print(f"  [error] {file_path} (HTTP {response.status_code})")
                    failed += 1
            except httpx.RequestError as e:
                logger.warning(f"Failed to download {file_path}: {e}")
                create_default_config(local_path, file_path)
                print(f"  [created] {file_path} (default, offline)")
                downloaded += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed")
    print(f"\nConfiguration files are in: {config_dir}")
    print("\nNext steps:")
    print("  1. Edit configs/db/default.yaml with your database credentials")
    print("  2. Ingest a dataset: autorag-research ingest beir --dataset=scifact")
    print("  3. Run experiment: autorag-research run --db-name=beir_scifact_test")


def create_default_config(path: Path, file_name: str) -> None:
    """Create a default configuration file."""
    defaults = get_default_configs()
    if file_name in defaults:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(defaults[file_name])


def get_default_configs() -> dict[str, str]:
    """Return default configuration file contents."""
    return {
        "db/default.yaml": """# Database connection configuration
host: localhost
port: 5432
user: postgres
password: ${oc.env:PGPASSWORD,postgres}
database: autorag_research
""",
        "experiment.yaml": """# Experiment configuration
#
# Usage:
#   autorag-research run                           # Run this config
#   autorag-research run pipelines.0.k1=1.2       # Override parameter
#   autorag-research run -m pipelines.0.k1=0.5,0.9,1.2  # Multirun
#

defaults:
  - db: default
  - pipelines/bm25_baseline@pipelines.0
  - metrics/recall@metrics.0
  - metrics/ndcg@metrics.1
  - _self_

schema: beir_scifact_test
embedding_dim: 1536
max_retries: 3
eval_batch_size: 100
""",
        "pipelines/bm25_baseline.yaml": """# BM25 Retrieval Pipeline
_target_: autorag_research.pipelines.retrieval.bm25.BM25PipelineConfig
name: bm25_baseline
index_path: ${hydra:runtime.cwd}/indices/bm25
k1: 0.9
b: 0.4
language: en
top_k: 10
batch_size: 100
""",
        "pipelines/vector_search.yaml": """# Vector Search Pipeline
_target_: autorag_research.pipelines.retrieval.vector_search.VectorSearchPipelineConfig
name: vector_search
similarity_top_k: 10
batch_size: 100
""",
        "pipelines/hybrid_search.yaml": """# Hybrid Search Pipeline
_target_: autorag_research.pipelines.retrieval.hybrid.HybridPipelineConfig
name: hybrid_search
bm25_weight: 0.5
vector_weight: 0.5
top_k: 10
batch_size: 100
""",
        "pipelines/naive_rag.yaml": """# Naive RAG Pipeline
_target_: autorag_research.pipelines.generation.naive_rag.NaiveRAGPipelineConfig
name: naive_rag
retrieval_pipeline_name: bm25_baseline
llm_model: gpt-4o-mini
system_prompt: null
top_k: 5
batch_size: 10
""",
        "metrics/recall.yaml": """# Recall@k Metric
_target_: autorag_research.metrics.retrieval.recall.RecallConfig
k: 10
""",
        "metrics/recall_5.yaml": """# Recall@5 Metric
_target_: autorag_research.metrics.retrieval.recall.RecallConfig
k: 5
""",
        "metrics/precision.yaml": """# Precision@k Metric
_target_: autorag_research.metrics.retrieval.precision.PrecisionConfig
k: 10
""",
        "metrics/ndcg.yaml": """# NDCG@k Metric
_target_: autorag_research.metrics.retrieval.ndcg.NDCGConfig
k: 10
""",
        "metrics/mrr.yaml": """# MRR Metric
_target_: autorag_research.metrics.retrieval.mrr.MRRConfig
k: 10
""",
        "metrics/f1.yaml": """# F1 Score Metric
_target_: autorag_research.metrics.generation.f1.F1Config
""",
        "metrics/rouge.yaml": """# ROUGE Score Metric
_target_: autorag_research.metrics.generation.rouge.RougeConfig
rouge_type: rougeL
""",
    }
