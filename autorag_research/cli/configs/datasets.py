"""Dataset configurations - available ingestors for listing."""

from autorag_research.cli.configs.ingestors import INGESTOR_REGISTRY

# Available datasets for listing (generated from ingestor registry)
# Format: ingestor_name -> description with available values
AVAILABLE_DATASETS = {
    name: f"{spec.description} (values: {', '.join(spec.available_values[:3])}...)"
    for name, spec in INGESTOR_REGISTRY.items()
}
