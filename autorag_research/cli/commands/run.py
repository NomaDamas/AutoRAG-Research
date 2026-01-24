"""run command - Execute experiment pipelines with metrics evaluation."""

import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from autorag_research.cli.utils import setup_logging
from autorag_research.config import ExecutorConfig
from autorag_research.executor import Executor
from autorag_research.orm.schema_factory import create_schema

logger = logging.getLogger("AutoRAG-Research")


# config_path=None means Hydra will use --config-path CLI option (injected by main.py)
@hydra.main(version_base=None, config_path=None, config_name="experiment")
def run(cfg: DictConfig) -> None:
    """Run experiment pipelines with metrics evaluation.

    Configuration is loaded from configs/experiment.yaml (or specified --config-name).
    Pipelines and metrics are instantiated via _target_ in YAML.
    """
    setup_logging(verbose=cfg.get("verbose", False))

    db_name = cfg.get("db_name", "")
    if not db_name:
        print("Error: db_name is required in config")
        print("Add 'db_name: <db_name>' to your experiment YAML or pass db_name=xxx")
        return

    pipelines_cfg = cfg.get("pipelines", [])
    metrics_cfg = cfg.get("metrics", [])

    if not pipelines_cfg:
        print("Error: at least one pipeline is required")
        print("Add pipelines to your experiment YAML using defaults")
        return

    print("\nAutoRAG-Research Experiment Runner")
    print("=" * 60)

    print(f"\nDatabase name: {db_name}")
    print(f"Database: {cfg.db.host}:{cfg.db.port}/{cfg.db.database}")

    print("\nPipelines:")
    for i, p in enumerate(pipelines_cfg):
        if hasattr(p, "_target_"):
            print(f"  [{i}] {p._target_.split('.')[-1]}")
        print(f"       {OmegaConf.to_yaml(p).strip()}")

    print("\nMetrics:")
    for i, m in enumerate(metrics_cfg):
        if hasattr(m, "_target_"):
            print(f"  [{i}] {m._target_.split('.')[-1]}")
        print(f"       {OmegaConf.to_yaml(m).strip()}")

    print("\nConfiguration:")
    print(f"  max_retries: {cfg.get('max_retries', 3)}")
    print(f"  eval_batch_size: {cfg.get('eval_batch_size', 100)}")

    # Build executor config using instantiate
    try:
        executor_config = build_executor_config(cfg)
    except Exception as e:
        logger.exception("Failed to build executor config")
        print(f"\nError building configuration: {e}")
        return

    # Create database session
    try:
        session_factory = create_session_factory(cfg)
    except Exception as e:
        logger.exception("Failed to connect to database")
        print(f"\nError connecting to database: {e}")
        return

    # Create schema object
    schema = create_schema(dim=cfg.get("embedding_dim", 1536))

    print("\n" + "-" * 60)
    print("Starting experiment...")
    print("-" * 60)

    try:
        executor = Executor(
            session_factory=session_factory,
            config=executor_config,
            schema=schema,
        )
        result = executor.run()
        print_results(result)
    except Exception as e:
        logger.exception("Experiment failed")
        print(f"\nExperiment failed: {e}")


def build_executor_config(cfg: DictConfig) -> ExecutorConfig:
    """Build ExecutorConfig from Hydra configuration using instantiate."""
    pipelines = []
    metrics = []

    # Instantiate pipeline configs via _target_
    for p_cfg in cfg.get("pipelines", []):
        pipeline_config = instantiate(p_cfg)
        pipelines.append(pipeline_config)

    # Instantiate metric configs via _target_
    for m_cfg in cfg.get("metrics", []):
        metric_config = instantiate(m_cfg)
        metrics.append(metric_config)

    return ExecutorConfig(
        pipelines=pipelines,
        metrics=metrics,
        max_retries=cfg.get("max_retries", 3),
        eval_batch_size=cfg.get("eval_batch_size", 100),
    )


def create_session_factory(cfg: DictConfig) -> sessionmaker:
    """Create SQLAlchemy session factory from config."""
    password = cfg.db.password
    if isinstance(password, str) and password.startswith("${") and password.endswith("}"):
        password = os.environ.get("PGPASSWORD", "postgres")

    db_url = f"postgresql+psycopg://{cfg.db.user}:{password}@{cfg.db.host}:{cfg.db.port}/{cfg.db.database}"
    engine = create_engine(db_url)
    return sessionmaker(bind=engine)


def print_results(result) -> None:
    """Print experiment results."""
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)

    print("\nPipeline Results:")
    print("-" * 60)
    for pr in result.pipeline_results:
        status = "✓" if pr.success else "✗"
        print(f"  {status} {pr.pipeline_name}")
        if pr.error:
            print(f"      Error: {pr.error}")

    print("\nMetric Results:")
    print("-" * 60)
    for mr in result.metric_results:
        print(f"  {mr.metric_name} @ {mr.pipeline_name}")
        print(f"      Score: {mr.score:.4f}")

    successful_pipelines = sum(1 for pr in result.pipeline_results if pr.success)
    total_pipelines = len(result.pipeline_results)
    print(f"\nSummary: {successful_pipelines}/{total_pipelines} pipelines completed")

    if result.metric_results:
        avg_score = sum(mr.score for mr in result.metric_results) / len(result.metric_results)
        print(f"Average metric score: {avg_score:.4f}")
