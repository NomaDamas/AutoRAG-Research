"""Ingestor registry for CLI - maps ingestor names to their specifications."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IngestorOption:
    """CLI option specification for an ingestor."""

    name: str  # CLI option name without -- (e.g., "batch-size")
    param_name: str  # Python parameter name (e.g., "batch_size")
    type: type  # Python type
    default: Any = None
    required: bool = False
    help: str = ""
    is_list: bool = False  # For comma-separated values like --configs=a,b,c


@dataclass
class IngestorSpec:
    """Ingestor CLI specification."""

    ingestor_class: str  # Full class path (e.g., "autorag_research.data.beir.BEIRIngestor")
    dataset_param: str  # Parameter name for dataset selection (e.g., "dataset_name")
    cli_option: str  # CLI option name (e.g., "dataset")
    description: str  # Help text description
    available_values: list[str] = field(default_factory=list)  # Available dataset values
    extra_options: list[IngestorOption] = field(default_factory=list)


# Common options shared by most ingestors
COMMON_OPTIONS = [
    IngestorOption(
        name="subset",
        param_name="subset",
        type=str,
        default="test",
        help="Dataset split: train, dev, or test",
    ),
    IngestorOption(
        name="query-limit",
        param_name="query_limit",
        type=int,
        default=None,
        help="Maximum number of queries to ingest",
    ),
    IngestorOption(
        name="min-corpus-cnt",
        param_name="min_corpus_cnt",
        type=int,
        default=None,
        help="Minimum number of corpus documents to ingest, which means at least these many chunks will be ingested.",
    ),
    IngestorOption(
        name="db-name",
        param_name="db_name",
        type=str,
        default=None,
        help="Custom database schema name (auto-generated if not specified)",
    ),
]


INGESTOR_REGISTRY: dict[str, IngestorSpec] = {
    "beir": IngestorSpec(
        ingestor_class="autorag_research.data.beir.BEIRIngestor",
        dataset_param="dataset_name",
        cli_option="dataset",
        description="BEIR benchmark datasets for information retrieval",
        available_values=[
            "msmarco",
            "trec-covid",
            "nfcorpus",
            "nq",
            "hotpotqa",
            "fiqa",
            "arguana",
            "webis-touche2020",
            "cqadupstack",
            "quora",
            "dbpedia-entity",
            "scidocs",
            "fever",
            "climate-fever",
            "scifact",
            "germanquad",
            "robust04",
            "signal1m",
        ],
    ),
    "mrtydi": IngestorSpec(
        ingestor_class="autorag_research.data.mrtydi.MrTyDiIngestor",
        dataset_param="language",
        cli_option="language",
        description="Mr. TyDi multilingual retrieval benchmark",
        available_values=[
            "arabic",
            "bengali",
            "english",
            "finnish",
            "indonesian",
            "japanese",
            "korean",
            "russian",
            "swahili",
            "telugu",
            "thai",
        ],
    ),
    "ragbench": IngestorSpec(
        ingestor_class="autorag_research.data.ragbench.RAGBenchIngestor",
        dataset_param="configs",
        cli_option="configs",
        description="RAGBench benchmark for RAG evaluation",
        available_values=[
            "covidqa",
            "cuad",
            "delucionqa",
            "emanual",
            "expertqa",
            "finqa",
            "hagrid",
            "hotpotqa",
            "msmarco",
            "pubmedqa",
            "tatqa",
            "techqa",
        ],
        extra_options=[
            IngestorOption(
                name="batch-size",
                param_name="batch_size",
                type=int,
                default=1000,
                help="Batch size for streaming ingestion",
            ),
        ],
    ),
    "mteb": IngestorSpec(
        ingestor_class="autorag_research.data.mteb.TextMTEBDatasetIngestor",
        dataset_param="task_name",
        cli_option="task-name",
        description="MTEB (Massive Text Embedding Benchmark) retrieval tasks",
        available_values=[
            "NFCorpus",
            "SciFact",
            "ArguAna",
            "FiQA2018",
            "SCIDOCS",
            "MSMARCO",
            "HotpotQA",
            "FEVER",
            "NQ",
            "QuoraRetrieval",
            "IFEval",
            "IFIRNFCorpus",
            "IFIRSciFact",
        ],
        extra_options=[
            IngestorOption(
                name="score-threshold",
                param_name="score_threshold",
                type=int,
                default=1,
                help="Minimum relevance score threshold (0-2)",
            ),
            IngestorOption(
                name="include-instruction",
                param_name="include_instruction",
                type=bool,
                default=True,
                help="Include instruction prefix for InstructionRetrieval tasks",
            ),
        ],
    ),
    "bright": IngestorSpec(
        ingestor_class="autorag_research.data.bright.BRIGHTIngestor",
        dataset_param="domains",
        cli_option="domains",
        description="BRIGHT benchmark for reasoning-intensive retrieval",
        available_values=[
            "biology",
            "earth_science",
            "economics",
            "psychology",
            "robotics",
            "stackoverflow",
            "sustainable_living",
            "leetcode",
            "pony",
            "aops",
            "theoremqa_theorems",
            "theoremqa_questions",
            "math",
        ],
        extra_options=[
            IngestorOption(
                name="document-mode",
                param_name="document_mode",
                type=str,
                default="short",
                help="Document mode: 'short' or 'long'",
            ),
        ],
    ),
    "vidorev2": IngestorSpec(
        ingestor_class="autorag_research.data.vidorev2.ViDoReV2Ingestor",
        dataset_param="dataset_name",
        cli_option="dataset",
        description="ViDoRe v2 visual document retrieval benchmark",
        available_values=[
            "vidore_esg_reports_v2",
            "vidore_biomedical_lectures_v2",
            "vidore_economics_reports_v2",
        ],
    ),
}


def generate_db_name(ingestor_name: str, params: dict, subset: str) -> str:
    """Generate database schema name from ingestor parameters.

    Examples:
        beir + scifact + test → beir_scifact_test
        ragbench + [covidqa, msmarco] + test → ragbench_covidqa_msmarco_test
    """
    parts = [ingestor_name]

    spec = INGESTOR_REGISTRY.get(ingestor_name)
    if spec:
        param_value = params.get(spec.dataset_param)
        if isinstance(param_value, list):
            parts.extend([v.lower().replace("-", "_") for v in param_value])
        elif isinstance(param_value, str):
            parts.append(param_value.lower().replace("-", "_"))

    parts.append(subset)
    return "_".join(parts)


def get_ingestor_help() -> str:
    """Generate help text listing all available ingestors."""
    lines = ["Available ingestors:"]
    for name, spec in INGESTOR_REGISTRY.items():
        lines.append(f"  {name:12} - {spec.description}")
    return "\n".join(lines)
