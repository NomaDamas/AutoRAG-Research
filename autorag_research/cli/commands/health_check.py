"""health-check command - Health check embedding, LLM, and reranker models."""

import logging
from typing import Annotated, Literal

import typer

logger = logging.getLogger("AutoRAG-Research")

ModelType = Literal["embedding", "llm", "reranker", "all"]

MODEL_TYPES: tuple[str, ...] = ("embedding", "llm", "reranker")

DISCOVER_FUNCS: dict[str, str] = {
    "embedding": "discover_embedding_configs",
    "llm": "discover_llm_configs",
    "reranker": "discover_reranker_configs",
}


def health_check_command(
    model_type: Annotated[
        ModelType,
        typer.Argument(help="Model type to check: embedding, llm, reranker, or all"),
    ],
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Specific config name to check"),
    ] = None,
) -> None:
    """Health check embedding, LLM, and reranker models.

    Loads each model config via Hydra and runs a health check to verify
    the model is functional.

    Examples:
      autorag-research health-check all
      autorag-research health-check embedding
      autorag-research health-check embedding --name mock
      autorag-research health-check llm -n openai-gpt4
    """
    types_to_check = MODEL_TYPES if model_type == "all" else (model_type,)

    total_passed = 0
    total_failed = 0

    for mt in types_to_check:
        passed, failed = _check_model_type(mt, name)
        total_passed += passed
        total_failed += failed

    typer.echo(f"\nResults: {total_passed} passed, {total_failed} failed")

    if total_failed > 0:
        raise typer.Exit(1)


def _check_model_type(model_type: str, name: str | None) -> tuple[int, int]:
    """Discover configs and check each model.

    Args:
        model_type: One of "embedding", "llm", "reranker".
        name: Optional specific config name to check.

    Returns:
        Tuple of (passed_count, failed_count).
    """
    from autorag_research.cli import utils

    discover_func = getattr(utils, DISCOVER_FUNCS[model_type])

    try:
        configs = discover_func()
    except FileNotFoundError:
        typer.echo(f"\n[{model_type}]")
        typer.echo("  No config directory found. Run 'autorag-research init' to create configs.")
        return 0, 0

    if name is not None:
        if name not in configs:
            typer.echo(f"\n[{model_type}]")
            typer.echo(f"  Config '{name}' not found. Available: {', '.join(sorted(configs))}")
            return 0, 1
        configs = {name: configs[name]}

    typer.echo(f"\n[{model_type}]")

    passed = 0
    failed = 0

    for config_name in sorted(configs):
        success = _check_single_model(model_type, config_name)
        if success:
            passed += 1
        else:
            failed += 1

    return passed, failed


def _check_single_model(model_type: str, config_name: str) -> bool:
    """Load and health-check a single model config.

    Args:
        model_type: One of "embedding", "llm", "reranker".
        config_name: Config name (without .yaml extension).

    Returns:
        True if health check passed, False otherwise.
    """
    from autorag_research.injection import (
        health_check_embedding,
        load_embedding_model,
        load_llm,
        load_reranker,
    )

    try:
        if model_type == "embedding":
            model = load_embedding_model(config_name)
            dim = health_check_embedding(model)
            typer.echo(f"  [PASS] {config_name} (dimension: {dim})")
        elif model_type == "llm":
            load_llm(config_name)
            typer.echo(f"  [PASS] {config_name}")
        elif model_type == "reranker":
            load_reranker(config_name)
            typer.echo(f"  [PASS] {config_name}")
    except Exception as e:
        typer.echo(f"  [FAIL] {config_name} -- {e}")
        return False
    else:
        return True
