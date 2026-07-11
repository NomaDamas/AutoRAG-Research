"""Gradio MTEB-style leaderboard UI for AutoRAG evaluation results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from typing import Literal

import gradio as gr
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from autorag_research.cli.config_resolver import ConfigResolver
from autorag_research.cli.utils import get_config_dir
from autorag_research.reporting.service import ReportingService

METRIC_TYPES = Literal["retrieval", "generation"]


@dataclass(frozen=True)
class LeaderboardScope:
    """Experiment-defined database, pipeline, and metric allowlists."""

    db_name: str
    pipeline_names: dict[str, tuple[str, ...]]
    metric_names: dict[str, tuple[str, ...]]


def _config_references(config: DictConfig, section: str, metric_type: METRIC_TYPES) -> list[str]:
    """Return normalized config references for one experiment section and type."""
    section_config = config.get(section, {})
    references = section_config.get(metric_type, []) if isinstance(section_config, DictConfig) else []
    if isinstance(references, str):
        return [references]
    return [str(reference) for reference in references]


def load_leaderboard_scope(config_name: str) -> LeaderboardScope:
    """Load leaderboard allowlists from an experiment YAML config."""
    config_dir = get_config_dir()
    experiment_path = config_dir / f"{config_name}.yaml"
    if not experiment_path.exists():
        raise FileNotFoundError(f"Config file not found: {experiment_path}")  # noqa: TRY003

    experiment_config = OmegaConf.load(experiment_path)
    if not isinstance(experiment_config, DictConfig):
        raise TypeError(f"Experiment config must be a YAML mapping: {experiment_path}")  # noqa: TRY003

    db_name = experiment_config.get("db_name")
    if not db_name:
        raise ValueError(f"Experiment config must define db_name: {experiment_path}")  # noqa: TRY003

    resolver = ConfigResolver(config_dir=config_dir)
    pipeline_names: dict[str, tuple[str, ...]] = {}
    metric_names: dict[str, tuple[str, ...]] = {}
    for metric_type in ("retrieval", "generation"):
        pipeline_references = _config_references(experiment_config, "pipelines", metric_type)
        resolved_pipeline_names = []
        for reference in pipeline_references:
            pipeline_config = resolver.resolve_config(["pipelines", metric_type], reference)
            resolved_pipeline_names.append(str(pipeline_config.get("name", reference)))
        pipeline_names[metric_type] = tuple(resolved_pipeline_names)

        metric_references = _config_references(experiment_config, "metrics", metric_type)
        resolved_metric_names = []
        for reference in metric_references:
            metric_config = resolver.resolve_config(["metrics", metric_type], reference)
            resolved_metric_names.append(str(instantiate(metric_config).get_metric_name()))
        metric_names[metric_type] = tuple(resolved_metric_names)

    return LeaderboardScope(
        db_name=str(db_name),
        pipeline_names=pipeline_names,
        metric_names=metric_names,
    )


# === Service Management ===
class _ServiceManager:
    """Manages ReportingService singleton without global variables."""

    _instance: ReportingService | None = None

    @classmethod
    def get(cls) -> ReportingService:
        """Get or create ReportingService singleton."""
        if cls._instance is None:
            cls._instance = ReportingService()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the service singleton (useful for testing)."""
        if cls._instance is not None:
            cls._instance.close()
            cls._instance = None


def get_service() -> ReportingService:
    """Get or create ReportingService singleton."""
    return _ServiceManager.get()


def reset_service() -> None:
    """Reset the service singleton (useful for testing)."""
    _ServiceManager.reset()


def format_dataset_stats(db_name: str) -> str:
    """Format dataset statistics as display string."""
    if not db_name:
        return ""
    stats = get_service().get_dataset_stats(db_name)
    return f"📊 {stats['query_count']} queries | {stats['chunk_count']} chunks | {stats['document_count']} documents"


# === UI Update Handlers ===


def _scope_filters(
    scope: LeaderboardScope | None, metric_type: METRIC_TYPES
) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
    """Return pipeline and metric allowlists for the selected metric type."""
    if scope is None:
        return None, None
    return scope.pipeline_names.get(metric_type, ()), scope.metric_names.get(metric_type, ())


def on_dataset_change(
    db_name: str, metric_type: METRIC_TYPES, scope: LeaderboardScope | None = None
) -> tuple[pd.DataFrame, dict]:
    """Handle dataset selection change - returns leaderboard DataFrame and stats."""
    if not db_name:
        return pd.DataFrame(), gr.update(value="")
    pipeline_names, metric_names = _scope_filters(scope, metric_type)
    df = get_service().get_all_metrics_leaderboard(
        db_name,
        metric_type,
        pipeline_names=pipeline_names,
        metric_names=metric_names,
    )
    stats = format_dataset_stats(db_name)
    return df, gr.update(value=stats)


def on_metric_type_change(
    db_name: str, metric_type: METRIC_TYPES, scope: LeaderboardScope | None = None
) -> pd.DataFrame:
    """Handle metric type selection change - returns leaderboard DataFrame."""
    if not db_name:
        return pd.DataFrame()
    pipeline_names, metric_names = _scope_filters(scope, metric_type)
    return get_service().get_all_metrics_leaderboard(
        db_name,
        metric_type,
        pipeline_names=pipeline_names,
        metric_names=metric_names,
    )


def on_refresh_leaderboard(
    db_name: str, metric_type: METRIC_TYPES, scope: LeaderboardScope | None = None
) -> tuple[pd.DataFrame, dict]:
    """Refresh leaderboard data."""
    if not db_name:
        return pd.DataFrame(), gr.update(value="")
    pipeline_names, metric_names = _scope_filters(scope, metric_type)
    df = get_service().get_all_metrics_leaderboard(
        db_name,
        metric_type,
        pipeline_names=pipeline_names,
        metric_names=metric_names,
    )
    stats = format_dataset_stats(db_name)
    return df, gr.update(value=stats)


def on_datasets_select_for_pipelines(db_names: list[str]) -> dict:
    """Update pipeline dropdown when datasets are selected."""
    if not db_names:
        return gr.update(choices=[], value=None)
    # Get pipelines common to all selected datasets (intersection)
    service = get_service()
    pipeline_sets = [set(service.list_pipelines(db_name)) for db_name in db_names]
    common_pipelines = sorted(set.intersection(*pipeline_sets)) if pipeline_sets else []
    return gr.update(choices=common_pipelines, value=common_pipelines[0] if common_pipelines else None)


def on_datasets_select_for_metrics(db_names: list[str]) -> dict:
    """Update metrics checkbox when datasets are selected."""
    if not db_names:
        return gr.update(choices=[], value=[])
    # Get metrics common to all selected datasets (intersection)
    service = get_service()
    metric_sets = [set(service.list_metrics(db_name)) for db_name in db_names]
    common_metrics = sorted(set.intersection(*metric_sets)) if metric_sets else []
    return gr.update(choices=common_metrics, value=[])


# === UI Component Builders ===


def build_single_dataset_tab(
    scope: LeaderboardScope | None = None,
) -> tuple[gr.Tab, gr.Dropdown, gr.Dropdown, gr.Dataframe, gr.Textbox]:
    """Build the single dataset leaderboard tab."""
    metric_types = [
        metric_type
        for metric_type in ("retrieval", "generation")
        if scope is None or (scope.pipeline_names.get(metric_type) and scope.metric_names.get(metric_type))
    ]
    selected_metric_type = metric_types[0] if metric_types else "retrieval"
    dataset_change_handler = partial(on_dataset_change, scope=scope)
    metric_type_change_handler = partial(on_metric_type_change, scope=scope)
    refresh_handler = partial(on_refresh_leaderboard, scope=scope)

    with gr.Tab("Single Dataset") as tab:
        with gr.Row():
            dataset_dropdown = gr.Dropdown(
                label="Dataset",
                choices=[],
                interactive=scope is None,
                scale=2,
            )
            metric_type_dropdown = gr.Dropdown(
                label="Metric Type",
                choices=metric_types,
                value=selected_metric_type,
                interactive=len(metric_types) > 1,
                scale=1,
            )
            refresh_btn = gr.Button("🔄 Refresh", scale=1)

        leaderboard_table = gr.Dataframe(
            label="Leaderboard",
            interactive=False,
        )

        stats_display = gr.Textbox(
            label="Dataset Stats",
            interactive=False,
            show_label=False,
        )

        # Event handlers
        dataset_dropdown.change(
            fn=dataset_change_handler,
            inputs=[dataset_dropdown, metric_type_dropdown],
            outputs=[leaderboard_table, stats_display],
        )
        metric_type_dropdown.change(
            fn=metric_type_change_handler,
            inputs=[dataset_dropdown, metric_type_dropdown],
            outputs=[leaderboard_table],
        )
        refresh_btn.click(
            fn=refresh_handler,
            inputs=[dataset_dropdown, metric_type_dropdown],
            outputs=[leaderboard_table, stats_display],
        )

    return tab, dataset_dropdown, metric_type_dropdown, leaderboard_table, stats_display


def build_cross_dataset_tab(all_datasets: list[str]) -> tuple[gr.Tab, gr.CheckboxGroup]:
    """Build the cross-dataset comparison tab."""
    with gr.Tab("Cross-Dataset") as tab:
        with gr.Row():
            datasets_checkbox = gr.CheckboxGroup(
                label="Datasets",
                choices=all_datasets,
                interactive=True,
                scale=2,
            )
        with gr.Row():
            pipeline_dropdown = gr.Dropdown(
                label="Pipeline",
                choices=[],
                value=None,
                interactive=True,
                preserved_by_key=None,  # Disable browser state restoration for dynamic choices
                scale=2,
            )
            compare_btn = gr.Button("🔄 Compare", scale=1)

        comparison_table = gr.Dataframe(
            label="Cross-Dataset Comparison",
            interactive=False,
        )

        # Event handlers
        # When datasets change -> update pipelines
        datasets_checkbox.change(
            fn=on_datasets_select_for_pipelines,
            inputs=[datasets_checkbox],
            outputs=[pipeline_dropdown],
        )
        compare_btn.click(
            fn=lambda dbs, p: get_service().compare_pipeline_all_metrics(dbs, p) if dbs and p else pd.DataFrame(),
            inputs=[datasets_checkbox, pipeline_dropdown],
            outputs=[comparison_table],
        )

    return tab, datasets_checkbox


def build_borda_ranking_tab(all_datasets: list[str]) -> tuple[gr.Tab, gr.CheckboxGroup, gr.CheckboxGroup]:
    """Build the Borda count ranking tab (MTEB-style)."""
    with gr.Tab("Borda Ranking") as tab:
        with gr.Row():
            datasets_checkbox = gr.CheckboxGroup(
                label="Datasets",
                choices=all_datasets,
                interactive=True,
            )
        with gr.Row():
            metrics_checkbox = gr.CheckboxGroup(
                label="Metrics",
                choices=[],
                interactive=True,
            )
        compute_btn = gr.Button("🔄 Compute Ranking")

        borda_table = gr.Dataframe(
            label="Borda Count Leaderboard",
            headers=["pipeline", "total_rank", "avg_rank", "num_rankings"],
            interactive=False,
        )

        # Event handlers
        datasets_checkbox.change(
            fn=on_datasets_select_for_metrics,
            inputs=[datasets_checkbox],
            outputs=[metrics_checkbox],
        )
        compute_btn.click(
            fn=lambda dbs, ms: get_service().get_borda_count_leaderboard(dbs, ms) if dbs and ms else pd.DataFrame(),
            inputs=[datasets_checkbox, metrics_checkbox],
            outputs=[borda_table],
        )

    return tab, datasets_checkbox, metrics_checkbox


# === Main App Factory ===


def create_leaderboard_app(scope: LeaderboardScope | None = None) -> gr.Blocks:
    """Create the Gradio leaderboard application."""
    datasets = [scope.db_name] if scope is not None else get_service().list_available_datasets()
    initial_metric_type: METRIC_TYPES = "retrieval"
    if scope is not None and not (scope.pipeline_names.get("retrieval") and scope.metric_names.get("retrieval")):
        initial_metric_type = "generation"

    with gr.Blocks(title="AutoRAG-Research Leaderboard") as app:
        gr.Markdown("# 🏆 AutoRAG-Research Leaderboard")

        with gr.Tabs():
            (
                _single_tab,
                single_dataset_dropdown,
                _single_metric_type_dropdown,
                single_leaderboard_table,
                single_stats_display,
            ) = build_single_dataset_tab(scope)
            if scope is None:
                _cross_tab, _cross_datasets_checkbox = build_cross_dataset_tab(datasets)
                _borda_tab, _borda_datasets_checkbox, _borda_metrics_checkbox = build_borda_ranking_tab(datasets)

        # Initialize single dataset dropdown with available datasets
        app.load(
            fn=lambda: gr.update(choices=datasets, value=datasets[0] if datasets else None),
            outputs=[single_dataset_dropdown],
        )

        # Auto-load leaderboard when app starts if datasets exist
        if datasets:
            app.load(
                fn=partial(on_dataset_change, datasets[0], initial_metric_type, scope=scope),
                outputs=[single_leaderboard_table, single_stats_display],
            )

    return app


def main() -> None:
    """Launch the leaderboard application."""
    parser = argparse.ArgumentParser(description="Launch the AutoRAG-Research leaderboard.")
    parser.add_argument(
        "--config-name",
        help="Experiment config name used to restrict the database, pipelines, and metrics.",
    )
    args = parser.parse_args()
    scope = load_leaderboard_scope(args.config_name) if args.config_name else None
    app = create_leaderboard_app(scope)
    app.launch()


if __name__ == "__main__":
    main()
