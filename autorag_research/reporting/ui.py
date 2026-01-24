"""Gradio MTEB-style leaderboard UI for AutoRAG evaluation results."""

from __future__ import annotations

from typing import Literal

import gradio as gr
import pandas as pd

from autorag_research.reporting.service import ReportingService


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


# === Data Fetchers (Event Handlers) ===
def fetch_datasets() -> list[str]:
    """Fetch list of available datasets."""
    try:
        return get_service().list_available_datasets()
    except Exception as e:
        gr.Warning(f"Failed to fetch datasets: {e}")
        return []


def fetch_metrics(db_name: str, metric_type: Literal["retrieval", "generation"]) -> list[str]:
    """Fetch metrics for a dataset filtered by type."""
    if not db_name:
        return []
    try:
        result = get_service().list_metrics_by_type(db_name, metric_type)
    except Exception as e:
        gr.Warning(f"Failed to fetch metrics: {e}")
        print(f"[ERROR] fetch_metrics({db_name}, {metric_type}) failed: {e}")  # Terminal log
        return []
    if not result:
        gr.Warning(f"No metrics found for type '{metric_type}' in {db_name}")
    return result


def fetch_pipelines(db_name: str) -> list[str]:
    """Fetch pipelines for a dataset."""
    if not db_name:
        return []
    try:
        return get_service().list_pipelines(db_name)
    except Exception as e:
        gr.Warning(f"Failed to fetch pipelines: {e}")
        return []


def fetch_all_metrics(db_name: str) -> list[str]:
    """Fetch all metrics for a dataset."""
    if not db_name:
        return []
    try:
        return get_service().list_metrics(db_name)
    except Exception as e:
        gr.Warning(f"Failed to fetch metrics: {e}")
        return []


def fetch_leaderboard(db_name: str, metric_name: str) -> pd.DataFrame:
    """Fetch leaderboard data for a dataset and metric."""
    if not db_name or not metric_name:
        return pd.DataFrame()
    try:
        return get_service().get_leaderboard(db_name, metric_name)
    except Exception as e:
        gr.Warning(f"Failed to fetch leaderboard: {e}")
        return pd.DataFrame()


def fetch_dataset_stats(db_name: str) -> str:
    """Fetch dataset statistics and format as string."""
    if not db_name:
        return ""
    try:
        stats = get_service().get_dataset_stats(db_name)
        return (
            f"📊 {stats['query_count']} queries | {stats['chunk_count']} chunks | {stats['document_count']} documents"
        )
    except Exception as e:
        gr.Warning(f"Failed to fetch stats: {e}")
        return ""


def fetch_cross_dataset(db_names: list[str], pipeline: str, metric: str) -> pd.DataFrame:
    """Fetch cross-dataset comparison data."""
    if not db_names or not pipeline or not metric:
        return pd.DataFrame()
    try:
        return get_service().compare_across_datasets(db_names, pipeline, metric)
    except Exception as e:
        gr.Warning(f"Failed to fetch cross-dataset comparison: {e}")
        return pd.DataFrame()


def fetch_borda_ranking(db_names: list[str], metrics: list[str]) -> pd.DataFrame:
    """Fetch Borda count leaderboard."""
    if not db_names or not metrics:
        return pd.DataFrame()
    try:
        return get_service().get_borda_count_leaderboard(db_names, metrics)
    except Exception as e:
        gr.Warning(f"Failed to fetch Borda ranking: {e}")
        return pd.DataFrame()


# === UI Update Handlers ===


def on_dataset_change(db_name: str, metric_type: Literal["retrieval", "generation"]) -> tuple[dict, dict]:
    """Handle dataset selection change."""
    metrics = fetch_metrics(db_name, metric_type)
    stats = fetch_dataset_stats(db_name)
    return gr.update(choices=metrics, value=metrics[0] if metrics else None), gr.update(value=stats)


def on_metric_type_change(db_name: str, metric_type: Literal["retrieval", "generation"]) -> dict:
    """Handle metric type selection change."""
    metrics = fetch_metrics(db_name, metric_type)
    return gr.update(choices=metrics, value=metrics[0] if metrics else None)


def on_refresh_leaderboard(db_name: str, metric_name: str) -> tuple[pd.DataFrame, dict]:
    """Refresh leaderboard data."""
    df = fetch_leaderboard(db_name, metric_name)
    stats = fetch_dataset_stats(db_name)
    return df, gr.update(value=stats)


def on_datasets_select_for_pipelines(db_names: list[str]) -> dict:
    """Update pipeline dropdown when datasets are selected."""
    if not db_names:
        return gr.update(choices=[], value=None)
    # Get pipelines from first selected dataset
    pipelines = fetch_pipelines(db_names[0])
    return gr.update(choices=pipelines, value=pipelines[0] if pipelines else None)


def on_datasets_select_for_metrics(db_names: list[str]) -> dict:
    """Update metrics checkbox when datasets are selected."""
    if not db_names:
        return gr.update(choices=[], value=[])
    # Get metrics from first selected dataset
    metrics = fetch_all_metrics(db_names[0])
    return gr.update(choices=metrics, value=[])


# === UI Component Builders ===


def build_single_dataset_tab() -> tuple[gr.Tab, gr.Dropdown, gr.Dropdown, gr.Textbox]:
    """Build the single dataset leaderboard tab."""
    with gr.Tab("Single Dataset") as tab:
        with gr.Row():
            dataset_dropdown = gr.Dropdown(
                label="Dataset",
                choices=[],
                interactive=True,
                scale=2,
            )
            metric_type_dropdown = gr.Dropdown(
                label="Metric Type",
                choices=["retrieval", "generation"],
                value="retrieval",
                interactive=True,
                scale=1,
            )
        with gr.Row():
            metric_dropdown = gr.Dropdown(
                label="Metric",
                choices=[],
                interactive=True,
                scale=2,
            )
            refresh_btn = gr.Button("🔄 Refresh", scale=1)

        leaderboard_table = gr.Dataframe(
            label="Leaderboard",
            headers=["rank", "pipeline", "score", "time_ms"],
            interactive=False,
        )

        stats_display = gr.Textbox(
            label="Dataset Stats",
            interactive=False,
            show_label=False,
        )

        # Event handlers
        dataset_dropdown.change(
            fn=on_dataset_change,
            inputs=[dataset_dropdown, metric_type_dropdown],
            outputs=[metric_dropdown, stats_display],
        )
        metric_type_dropdown.change(
            fn=on_metric_type_change,
            inputs=[dataset_dropdown, metric_type_dropdown],
            outputs=[metric_dropdown],
        )
        metric_dropdown.change(
            fn=fetch_leaderboard,
            inputs=[dataset_dropdown, metric_dropdown],
            outputs=[leaderboard_table],
        )
        refresh_btn.click(
            fn=on_refresh_leaderboard,
            inputs=[dataset_dropdown, metric_dropdown],
            outputs=[leaderboard_table, stats_display],
        )

    return tab, dataset_dropdown, metric_dropdown, stats_display


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
                interactive=True,
                scale=1,
            )
            metric_dropdown = gr.Dropdown(
                label="Metric",
                choices=[],
                interactive=True,
                scale=1,
            )
            compare_btn = gr.Button("🔄 Compare", scale=1)

        comparison_table = gr.Dataframe(
            label="Cross-Dataset Comparison",
            headers=["dataset", "score", "time_ms"],
            interactive=False,
        )

        # Event handlers
        datasets_checkbox.change(
            fn=on_datasets_select_for_pipelines,
            inputs=[datasets_checkbox],
            outputs=[pipeline_dropdown],
        )
        datasets_checkbox.change(
            fn=lambda dbs: gr.update(
                choices=fetch_all_metrics(dbs[0]) if dbs else [],
                value=None,
            ),
            inputs=[datasets_checkbox],
            outputs=[metric_dropdown],
        )
        compare_btn.click(
            fn=fetch_cross_dataset,
            inputs=[datasets_checkbox, pipeline_dropdown, metric_dropdown],
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
            fn=fetch_borda_ranking,
            inputs=[datasets_checkbox, metrics_checkbox],
            outputs=[borda_table],
        )

    return tab, datasets_checkbox, metrics_checkbox


# === Main App Factory ===


def create_leaderboard_app() -> gr.Blocks:
    """Create the Gradio leaderboard application."""
    # Pre-fetch available datasets
    datasets = fetch_datasets()

    with gr.Blocks(title="AutoRAG Leaderboard") as app:
        gr.Markdown("# 🏆 AutoRAG Leaderboard")

        with gr.Tabs():
            _single_tab, single_dataset_dropdown, single_metric_dropdown, single_stats_display = (
                build_single_dataset_tab()
            )
            _cross_tab, _cross_datasets_checkbox = build_cross_dataset_tab(datasets)
            _borda_tab, _borda_datasets_checkbox, _borda_metrics_checkbox = build_borda_ranking_tab(datasets)

        # Initialize single dataset dropdown with available datasets
        app.load(
            fn=lambda: gr.update(choices=datasets, value=datasets[0] if datasets else None),
            outputs=[single_dataset_dropdown],
        )

        # Auto-load metrics when app starts if datasets exist
        if datasets:
            app.load(
                fn=lambda: on_dataset_change(datasets[0], "retrieval"),
                outputs=[single_metric_dropdown, single_stats_display],
            )

    return app


def main() -> None:
    """Launch the leaderboard application."""
    app = create_leaderboard_app()
    app.launch()


if __name__ == "__main__":
    main()
