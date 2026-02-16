"""health-check command - Health check embedding, LLM, and reranker models."""

import logging
from typing import Annotated, Literal

import typer

logger = logging.getLogger("AutoRAG-Research")

ModelType = Literal["embedding", "llm", "reranker"]


def health_check_command(
    model_type: Annotated[
        ModelType,
        typer.Argument(help="Model type to check: embedding, llm, or reranker"),
    ],
    name: Annotated[
        str,
        typer.Argument(help="Config name (YAML filename without extension)"),
    ],
) -> None:
    """Health check a specific model config.

    Loads the model config via Hydra and runs a health check to verify
    the model is functional.

    Examples:
      autorag-research health-check embedding mock
      autorag-research health-check llm openai-gpt4
      autorag-research health-check reranker cohere
    """
    from autorag_research.injection import (
        health_check_embedding,
        load_embedding_model,
        load_llm,
        load_reranker,
    )

    try:
        if model_type == "embedding":
            model = load_embedding_model(name)
            dim = health_check_embedding(model)
            typer.echo(f"[PASS] {name} (dimension: {dim})")
        elif model_type == "llm":
            load_llm(name)
            typer.echo(f"[PASS] {name}")
        elif model_type == "reranker":
            load_reranker(name)
            typer.echo(f"[PASS] {name}")
    except Exception as e:
        typer.echo(f"[FAIL] {name} -- {e}")
        raise typer.Exit(1) from None
