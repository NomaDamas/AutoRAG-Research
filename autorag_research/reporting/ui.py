"""Gradio MTEB-style leaderboard UI for AutoRAG evaluation results."""

from __future__ import annotations

import argparse
from functools import partial

import gradio as gr
import pandas as pd

from autorag_research.reporting.scope import (
    LeaderboardScope,
    MetricType,
    initial_scope_metric_type,
    load_leaderboard_scope,
    scope_filters,
    scope_metric_types,
)
from autorag_research.reporting.service import ReportingService
from autorag_research.reporting.service_manager import get_service as _get_service
from autorag_research.reporting.service_manager import reset_service as _reset_service


def get_service() -> ReportingService:
    """Get or create the reporting service."""
    return _get_service()


def reset_service() -> None:
    """Close and clear the reporting service."""
    _reset_service()


def format_dataset_stats(db_name: str) -> str:
    """Format dataset statistics as display string."""
    if not db_name:
        return ""
    stats = get_service().get_dataset_stats(db_name)
    return f"📊 {stats['query_count']} queries | {stats['chunk_count']} chunks | {stats['document_count']} documents"


# === UI Update Handlers ===


def on_dataset_change(
    db_name: str, metric_type: MetricType, scope: LeaderboardScope | None = None
) -> tuple[pd.DataFrame, dict]:
    """Handle dataset selection change - returns leaderboard DataFrame and stats."""
    if not db_name:
        return pd.DataFrame(), gr.update(value="")
    pipeline_names, metric_names = scope_filters(scope, metric_type)
    df = get_service().get_all_metrics_leaderboard(
        db_name,
        metric_type,
        pipeline_names=pipeline_names,
        metric_names=metric_names,
    )
    stats = format_dataset_stats(db_name)
    return df, gr.update(value=stats)


def on_metric_type_change(db_name: str, metric_type: MetricType, scope: LeaderboardScope | None = None) -> pd.DataFrame:
    """Handle metric type selection change - returns leaderboard DataFrame."""
    if not db_name:
        return pd.DataFrame()
    pipeline_names, metric_names = scope_filters(scope, metric_type)
    return get_service().get_all_metrics_leaderboard(
        db_name,
        metric_type,
        pipeline_names=pipeline_names,
        metric_names=metric_names,
    )


def on_refresh_leaderboard(
    db_name: str, metric_type: MetricType, scope: LeaderboardScope | None = None
) -> tuple[pd.DataFrame, dict]:
    """Refresh leaderboard data."""
    if not db_name:
        return pd.DataFrame(), gr.update(value="")
    pipeline_names, metric_names = scope_filters(scope, metric_type)
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
    metric_types = scope_metric_types(scope)
    selected_metric_type = initial_scope_metric_type(scope)
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
    initial_metric_type = initial_scope_metric_type(scope)

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
