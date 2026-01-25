"""Config resolver for name-based configuration loading.

This module provides a ConfigResolver class that resolves YAML config names
to actual file paths, mirroring the folder structure from YAML keys.

Example YAML structure:
    pipelines:
      retrieval: [bm25]        # → configs/pipelines/retrieval/bm25.yaml
      generation: [basic_rag]  # → configs/pipelines/generation/basic_rag.yaml

    metrics:
      retrieval: [recall, ndcg]  # → configs/metrics/retrieval/*.yaml
      generation: [rouge]        # → configs/metrics/generation/rouge.yaml
"""

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


@dataclass
class ConfigResolver:
    """Resolves config names to file paths based on YAML structure."""

    config_dir: Path

    def resolve_configs(self, base_dir: str, config_dict: DictConfig) -> list[DictConfig]:
        """Resolve YAML dict structure to file system paths and load configs.

        Args:
            base_dir: Base directory name ("pipelines" or "metrics").
            config_dict: Dict mapping subdirectory to list of config names.
                Example: {"retrieval": ["bm25"], "generation": ["basic_rag"]}

        Returns:
            List of loaded DictConfig objects.

        Raises:
            FileNotFoundError: If a config file doesn't exist.
        """
        configs = []
        for subdir, names in config_dict.items():
            # Handle both list and single string
            if isinstance(names, str):
                names = [names]
            for name in names:
                path = self.config_dir / base_dir / subdir / f"{name}.yaml"
                if not path.exists():
                    raise FileNotFoundError(f"Config not found: {path}")
                configs.append(OmegaConf.load(path))
        return configs

    def resolve_pipelines(self, pipelines_cfg: DictConfig) -> list[DictConfig]:
        """Resolve pipeline configs from YAML dict.

        Args:
            pipelines_cfg: Dict mapping pipeline type to list of names.
                Example: {"retrieval": ["bm25"], "generation": ["basic_rag"]}

        Returns:
            List of loaded pipeline DictConfig objects.
        """
        return self.resolve_configs("pipelines", pipelines_cfg)

    def resolve_metrics(self, metrics_cfg: DictConfig) -> list[DictConfig]:
        """Resolve metric configs from YAML dict.

        Args:
            metrics_cfg: Dict mapping metric type to list of names.
                Example: {"retrieval": ["recall", "ndcg"], "generation": ["rouge"]}

        Returns:
            List of loaded metric DictConfig objects.
        """
        return self.resolve_configs("metrics", metrics_cfg)

    def load_db_config(self, db_name: str = "default") -> DictConfig:
        """Load database config by name.

        Args:
            db_name: Database config name (default: "default").

        Returns:
            Loaded database DictConfig.

        Raises:
            FileNotFoundError: If db config doesn't exist.
            TypeError: If config is not a dict.
        """
        path = self.config_dir / "db" / f"{db_name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"DB config not found: {path}")
        cfg = OmegaConf.load(path)
        if not isinstance(cfg, DictConfig):
            raise TypeError(f"DB config must be a mapping, not a list: {path}")
        return cfg
