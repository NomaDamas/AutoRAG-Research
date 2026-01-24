"""Tests for autorag_research.cli.utils module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import autorag_research.cli as cli
from autorag_research.cli.utils import (
    APP_NAME,
    DatabaseConfig,
    discover_configs,
    discover_embedding_configs,
    discover_metrics,
    discover_pipelines,
    get_config_dir,
    get_db_url,
    get_user_data_dir,
    health_check_embedding,
    list_schemas_with_connection,
    load_db_config_from_yaml,
    load_embedding_model,
    setup_logging,
)


class TestDatabaseConfig:
    """Tests for DatabaseConfig dataclass."""

    def test_default_values(self) -> None:
        """DatabaseConfig has sensible default values."""
        config = DatabaseConfig()

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.user == "postgres"
        # password default is OmegaConf template for env var
        assert "PGPASSWORD" in config.password or config.password == "postgres"  # noqa: S105
        assert config.database == "autorag_research"

    def test_custom_values(self) -> None:
        """DatabaseConfig accepts custom values."""
        config = DatabaseConfig(
            host="db.example.com",
            port=5433,
            user="myuser",
            password="mypass",  # noqa: S106
            database="mydb",
        )

        assert config.host == "db.example.com"
        assert config.port == 5433
        assert config.user == "myuser"
        assert config.password == "mypass"  # noqa: S105
        assert config.database == "mydb"


class TestDiscoverConfigs:
    """Tests for discover_configs function."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Returns empty dict for empty directory."""
        result = discover_configs(tmp_path)

        assert result == {}

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """Returns empty dict for non-existent directory."""
        nonexistent = tmp_path / "does_not_exist"

        result = discover_configs(nonexistent)

        assert result == {}

    def test_yaml_with_description(self, tmp_path: Path) -> None:
        """Uses description field from YAML file."""
        (tmp_path / "test.yaml").write_text("description: Test pipeline config")

        result = discover_configs(tmp_path)

        assert "test" in result
        assert result["test"] == "Test pipeline config"

    def test_yaml_falls_back_to_target(self, tmp_path: Path) -> None:
        """Falls back to _target_ if no description field."""
        (tmp_path / "dense.yaml").write_text("_target_: some.module.DenseRetrieval\nk: 10")

        result = discover_configs(tmp_path)

        assert "dense" in result
        assert "some.module.DenseRetrieval" in result["dense"]

    def test_yaml_without_description_or_target(self, tmp_path: Path) -> None:
        """Uses filename as fallback if no description or target."""
        (tmp_path / "simple.yaml").write_text("key: value")

        result = discover_configs(tmp_path)

        assert "simple" in result

    def test_returns_sorted_by_name(self, tmp_path: Path) -> None:
        """Results are sorted alphabetically by name."""
        (tmp_path / "zebra.yaml").write_text("description: Z config")
        (tmp_path / "alpha.yaml").write_text("description: A config")
        (tmp_path / "middle.yaml").write_text("description: M config")

        result = discover_configs(tmp_path)

        keys = list(result.keys())
        assert keys == sorted(keys)

    def test_handles_invalid_yaml(self, tmp_path: Path) -> None:
        """Handles malformed YAML gracefully."""
        (tmp_path / "invalid.yaml").write_text("invalid: yaml: content: [")

        result = discover_configs(tmp_path)

        assert "invalid" in result
        assert "Error" in result["invalid"] or "error" in result["invalid"].lower() or result["invalid"] != ""


class TestDiscoverPipelines:
    """Tests for discover_pipelines function."""

    def test_discover_pipelines_uses_config_dir(self, sample_config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """discover_pipelines looks in pipelines/ subdirectory."""
        monkeypatch.setattr(cli, "CONFIG_PATH", sample_config_dir)

        result = discover_pipelines()

        assert "bm25" in result
        assert "dense" in result


class TestDiscoverMetrics:
    """Tests for discover_metrics function."""

    def test_discover_metrics_uses_config_dir(self, sample_config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """discover_metrics looks in metrics/ subdirectory."""
        monkeypatch.setattr(cli, "CONFIG_PATH", sample_config_dir)

        result = discover_metrics()

        assert "ndcg" in result


class TestDiscoverEmbeddingConfigs:
    """Tests for discover_embedding_configs function."""

    def test_discover_embedding_configs(self, sample_config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """discover_embedding_configs looks in embedding/ subdirectory."""
        monkeypatch.setattr(cli, "CONFIG_PATH", sample_config_dir)

        result = discover_embedding_configs()

        assert "openai-small" in result


class TestGetUserDataDir:
    """Tests for get_user_data_dir function."""

    def test_returns_path(self) -> None:
        """Returns a Path object."""
        result = get_user_data_dir()

        assert isinstance(result, Path)

    def test_contains_app_name(self) -> None:
        """Path contains app name."""
        result = get_user_data_dir()

        assert APP_NAME.lower() in str(result).lower() or "autorag" in str(result).lower()


class TestGetConfigDir:
    """Tests for get_config_dir function."""

    def test_returns_cli_config_path_when_set(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns cli.CONFIG_PATH when it's set."""
        monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path)

        result = get_config_dir()

        assert result == tmp_path

    def test_falls_back_to_cwd_configs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Falls back to cwd/configs when CONFIG_PATH is None."""
        monkeypatch.setattr(cli, "CONFIG_PATH", None)

        result = get_config_dir()

        assert result == Path.cwd() / "configs"


class TestGetDbUrl:
    """Tests for get_db_url function."""

    def test_formats_url_correctly(self) -> None:
        """Constructs correct PostgreSQL URL."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "db": {
                "host": "localhost",
                "port": 5432,
                "user": "testuser",
                "password": "testpass",
                "database": "testdb",
            }
        })

        result = get_db_url(cfg)

        assert result == "postgresql+psycopg://testuser:testpass@localhost:5432/testdb"


class TestLoadDbConfigFromYaml:
    """Tests for load_db_config_from_yaml function."""

    def test_returns_defaults_when_no_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns default DatabaseConfig when db.yaml doesn't exist."""
        monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path)

        result = load_db_config_from_yaml()

        assert result.host == "localhost"
        assert result.port == 5432

    def test_loads_values_from_yaml(self, sample_config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Loads values from db.yaml file."""
        monkeypatch.setattr(cli, "CONFIG_PATH", sample_config_dir)

        result = load_db_config_from_yaml()

        assert result.host == "localhost"
        assert result.user == "autorag"
        assert result.database == "autorag_db"

    def test_cli_args_override_yaml(self, sample_config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI arguments override YAML values."""
        monkeypatch.setattr(cli, "CONFIG_PATH", sample_config_dir)

        result = load_db_config_from_yaml(host="override.example.com", port=5433)

        assert result.host == "override.example.com"
        assert result.port == 5433
        # Non-overridden values come from YAML
        assert result.user == "autorag"


class TestLoadEmbeddingModel:
    """Tests for load_embedding_model function."""

    def test_raises_file_not_found_for_missing_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises FileNotFoundError when config doesn't exist."""
        monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path)

        with pytest.raises(FileNotFoundError):
            load_embedding_model("nonexistent")

    @patch("hydra.utils.instantiate")
    def test_raises_type_error_for_wrong_type(
        self, mock_instantiate: MagicMock, sample_config_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raises TypeError when instantiated object is not BaseEmbedding."""
        monkeypatch.setattr(cli, "CONFIG_PATH", sample_config_dir)
        mock_instantiate.return_value = "not an embedding"

        with pytest.raises(TypeError, match="BaseEmbedding"):
            load_embedding_model("openai-small")


class TestHealthCheckEmbedding:
    """Tests for health_check_embedding function."""

    def test_returns_dimension_on_success(self) -> None:
        """Returns embedding dimension on success."""
        mock_model = MagicMock()
        mock_model.get_text_embedding.return_value = [0.1] * 384

        result = health_check_embedding(mock_model)

        assert result == 384
        mock_model.get_text_embedding.assert_called_once()

    def test_raises_on_embedding_failure(self) -> None:
        """Raises EmbeddingNotSetError when embedding fails."""
        from autorag_research.exceptions import EmbeddingNotSetError

        mock_model = MagicMock()
        mock_model.get_text_embedding.side_effect = Exception("API Error")

        with pytest.raises(EmbeddingNotSetError):
            health_check_embedding(mock_model)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_verbose_sets_debug_level(self) -> None:
        """verbose=True sets DEBUG level on root logger."""
        import logging

        # Reset root logger handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        setup_logging(verbose=True)

        # basicConfig sets root logger level
        assert root_logger.level == logging.DEBUG

    def test_default_sets_info_level(self) -> None:
        """verbose=False sets INFO level on root logger."""
        import logging

        # Reset root logger handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        setup_logging(verbose=False)

        assert root_logger.level == logging.INFO


class TestListSchemasWithConnection:
    """Tests for list_schemas_with_connection function using mocks."""

    @patch("autorag_research.cli.utils.create_engine")
    def test_excludes_system_schemas(self, mock_create_engine: MagicMock) -> None:
        """Excludes pg_*, information_schema schemas."""
        # Mock the engine and connection
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_connection)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        # Mock result set with system and user schemas
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [
            "pg_catalog",
            "pg_toast",
            "information_schema",
            "public",
            "user_schema",
        ]
        mock_connection.execute.return_value = mock_result

        result = list_schemas_with_connection("localhost", 5432, "user", "pass", "db")

        # System schemas should be excluded
        assert "pg_catalog" not in result
        assert "pg_toast" not in result
        assert "information_schema" not in result

    @patch("autorag_research.cli.utils.create_engine")
    def test_returns_sorted_list(self, mock_create_engine: MagicMock) -> None:
        """Returns schemas sorted alphabetically."""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_connection)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = ["zebra_schema", "alpha_schema", "middle_schema"]
        mock_connection.execute.return_value = mock_result

        result = list_schemas_with_connection("localhost", 5432, "user", "pass", "db")

        assert result == sorted(result)
