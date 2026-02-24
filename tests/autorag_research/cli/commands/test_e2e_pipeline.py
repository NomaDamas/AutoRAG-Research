"""End-to-end CLI integration test for ingest -> run -> drop workflow."""

import logging
from pathlib import Path
from textwrap import dedent
from uuid import uuid4

import pytest
from sqlalchemy import text

from autorag_research.cli.app import app
from autorag_research.orm.connection import DBConnection


def _combined_output(result) -> str:
    return f"{result.stdout}\n{result.stderr}"


def _prepare_test_configs(tmp_path: Path) -> Path:
    """Prepare a minimal isolated config tree for e2e CLI test."""
    repo_root = Path(__file__).resolve().parents[4]
    source_configs = repo_root / "configs"
    config_dir = tmp_path / "configs"

    (config_dir / "embedding").mkdir(parents=True, exist_ok=True)
    (config_dir / "llm").mkdir(parents=True, exist_ok=True)
    (config_dir / "pipelines" / "retrieval").mkdir(parents=True, exist_ok=True)
    (config_dir / "pipelines" / "generation").mkdir(parents=True, exist_ok=True)
    (config_dir / "metrics" / "retrieval").mkdir(parents=True, exist_ok=True)
    (config_dir / "metrics" / "generation").mkdir(parents=True, exist_ok=True)

    (config_dir / "db.yaml").write_text((source_configs / "db.yaml").read_text())
    (config_dir / "embedding" / "mock.yaml").write_text((source_configs / "embedding" / "mock.yaml").read_text())
    (config_dir / "llm" / "mock.yaml").write_text((source_configs / "llm" / "mock.yaml").read_text())
    (config_dir / "pipelines" / "retrieval" / "bm25.yaml").write_text(
        (source_configs / "pipelines" / "retrieval" / "bm25.yaml").read_text()
    )
    (config_dir / "metrics" / "retrieval" / "recall.yaml").write_text(
        (source_configs / "metrics" / "retrieval" / "recall.yaml").read_text()
    )
    (config_dir / "metrics" / "generation" / "rouge.yaml").write_text(
        (source_configs / "metrics" / "generation" / "rouge.yaml").read_text()
    )

    (config_dir / "pipelines" / "generation" / "basic_rag.yaml").write_text(
        dedent(
            """
            _target_: autorag_research.pipelines.generation.basic_rag.BasicRAGPipelineConfig
            description: "Simple retrieve-then-generate RAG"
            name: basic_rag
            retrieval_pipeline_name: bm25
            llm: mock
            top_k: 10
            batch_size: 128
            max_concurrency: 8
            max_retries: 3
            retry_delay: 1.0
            prompt_template: |
              You are an AI assistant that provides answers based on the provided context.

              Context:
              {context}

              Question:
              {query}

              Answer:
            """
        ).strip()
        + "\n"
    )

    (config_dir / "experiment.yaml").write_text(
        dedent(
            """
            db_name: placeholder
            max_retries: 1
            eval_batch_size: 16
            pipelines:
              retrieval: [bm25]
              generation: [basic_rag]
            metrics:
              retrieval: [recall]
              generation: [rouge]
            """
        ).strip()
        + "\n"
    )

    return config_dir


def _force_drop_database(config_dir: Path, db_name: str) -> None:
    """Best-effort cleanup to avoid leaked test databases."""
    try:
        db_conn = DBConnection.from_config(config_dir)
        db_conn.database = db_name
        db_conn.terminate_connections()
        db_conn.drop_database()
    except Exception:
        logging.warning(f"Failed to clean up test database {db_name}", exc_info=True)
        return


@pytest.mark.ci_skip
def test_cli_end_to_end_ingest_run_and_drop(cli_runner, tmp_path: Path) -> None:
    """Runs real CLI ingestion + execution with small BEIR sample and verifies DB results."""
    config_dir = _prepare_test_configs(tmp_path)
    db_name = f"e2e_beir_scifact_{uuid4().hex[:8]}"
    db_created = False

    try:
        ingest_result = cli_runner.invoke(
            app,
            [
                "--config-path",
                str(config_dir),
                "ingest",
                "--name",
                "beir",
                "--extra",
                "dataset-name=scifact",
                "--subset",
                "test",
                "--query-limit",
                "10",
                "--min-corpus-cnt",
                "100",
                "--db-name",
                db_name,
                "--embedding-model",
                "mock",
                "--embed-batch-size",
                "32",
                "--embed-concurrency",
                "4",
            ],
        )

        ingest_output = _combined_output(ingest_result)
        assert ingest_result.exit_code == 0, ingest_output
        db_created = True  # DB was created by the CLI command
        assert "Ingesting dataset: beir" in ingest_output
        assert "Ingestion complete" in ingest_output
        assert "Embedding complete" in ingest_output

        db_conn = DBConnection.from_config(config_dir)
        db_conn.database = db_name
        ingest_engine = db_conn.get_engine()
        try:
            with ingest_engine.connect() as conn:
                query_count = conn.execute(text("SELECT COUNT(*) FROM query")).scalar_one()
                chunk_count = conn.execute(text("SELECT COUNT(*) FROM chunk")).scalar_one()
                retrieval_gt_count = conn.execute(text("SELECT COUNT(*) FROM retrieval_relation")).scalar_one()
        finally:
            ingest_engine.dispose()

        assert query_count == 10
        assert chunk_count >= 100
        assert retrieval_gt_count > 0

        run_result = cli_runner.invoke(
            app,
            [
                "--config-path",
                str(config_dir),
                "run",
                "--db-name",
                db_name,
            ],
        )

        run_output = _combined_output(run_result)
        assert run_result.exit_code == 0, run_output
        assert "EXPERIMENT RESULTS" in run_output
        assert "Summary:" in run_output

        run_engine = db_conn.get_engine()
        try:
            with run_engine.connect() as conn:
                pipeline_count = conn.execute(text("SELECT COUNT(*) FROM pipeline")).scalar_one()
                metric_count = conn.execute(text("SELECT COUNT(*) FROM metric")).scalar_one()
                executor_result_count = conn.execute(text("SELECT COUNT(*) FROM executor_result")).scalar_one()
                evaluation_result_count = conn.execute(text("SELECT COUNT(*) FROM evaluation_result")).scalar_one()
        finally:
            run_engine.dispose()

        assert pipeline_count >= 2
        assert metric_count >= 2
        assert executor_result_count > 0
        assert evaluation_result_count > 0

        drop_result = cli_runner.invoke(
            app,
            [
                "--config-path",
                str(config_dir),
                "drop",
                "database",
                "--db-name",
                db_name,
                "--yes",
            ],
        )

        drop_output = _combined_output(drop_result)
        assert drop_result.exit_code == 0, drop_output
        assert f"Database '{db_name}' dropped successfully." in drop_output
        db_created = False
    finally:
        if db_created:
            _force_drop_database(config_dir, db_name)
