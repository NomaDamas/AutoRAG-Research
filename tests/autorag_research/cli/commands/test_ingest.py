"""Tests for autorag_research.cli.commands.ingest module."""

from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
import typer
from langchain_core.embeddings import Embeddings
from typer.testing import CliRunner

from autorag_research.cli.app import app
from autorag_research.cli.commands.ingest import (
    _convert_param_value,
    _parse_extra_params,
    _validate_required_params,
    generate_db_name,
)
from autorag_research.data.base import TextEmbeddingDataIngestor
from autorag_research.data.registry import IngestorMeta, ParamMeta, discover_ingestors


@pytest.fixture
def beir_ingestor_meta() -> IngestorMeta:
    """Return the real beir ingestor metadata."""
    return discover_ingestors()["beir"]


@pytest.fixture
def mteb_ingestor_meta() -> IngestorMeta:
    """Return the real mteb ingestor metadata."""
    return discover_ingestors()["mteb"]


@pytest.fixture
def param_meta_list() -> ParamMeta:
    """Create ParamMeta for list type testing.

    Note: No real ingestors currently have list params, so we create a mock.
    """
    return ParamMeta(
        name="configs",
        cli_option="configs",
        param_type=str,
        choices=None,
        required=True,
        default=None,
        help="Config names",
        is_list=True,
    )


class TestIngestCommand:
    """Tests for the ingest CLI command."""

    def test_ingest_help_shows_options(self, cli_runner: CliRunner) -> None:
        """'ingest --help' shows available options."""
        result = cli_runner.invoke(app, ["ingest", "--help"])

        assert result.exit_code == 0
        assert "--name" in result.stdout or "-n" in result.stdout

    def test_ingest_help_shows_overwrite_option(self, cli_runner: CliRunner) -> None:
        """'ingest --help' shows --overwrite option."""
        result = cli_runner.invoke(app, ["ingest", "--help"])

        assert result.exit_code == 0
        assert "--overwrite" in result.stdout

    def test_ingest_without_name_shows_ingestors(self, cli_runner: CliRunner) -> None:
        """'ingest' without --name shows available ingestors."""
        result = cli_runner.invoke(app, ["ingest"])

        # Should show ingestors, embedding configs, or help (exit code 0 is acceptable)
        assert result.exit_code == 0 or "embedding" in result.stdout.lower() or "ingestor" in result.stdout.lower()


class TestParseExtraParams:
    """Tests for _parse_extra_params function."""

    def test_empty_list_returns_empty_dict(self) -> None:
        """Empty input returns empty dict."""
        result = _parse_extra_params([])

        assert result == {}

    def test_single_key_value_pair(self) -> None:
        """Parses single key=value pair."""
        result = _parse_extra_params(["dataset=scifact"])

        assert result == {"dataset": "scifact"}

    def test_multiple_pairs(self) -> None:
        """Parses multiple key=value pairs."""
        result = _parse_extra_params(["key1=value1", "key2=value2"])

        assert result == {"key1": "value1", "key2": "value2"}

    def test_kebab_to_snake_case(self) -> None:
        """Converts kebab-case keys to snake_case."""
        result = _parse_extra_params(["dataset-name=scifact"])

        assert "dataset_name" in result
        assert result["dataset_name"] == "scifact"

    def test_value_with_equals_sign(self) -> None:
        """Handles values containing = sign."""
        result = _parse_extra_params(["url=http://example.com?foo=bar"])

        assert result["url"] == "http://example.com?foo=bar"

    def test_invalid_format_raises_error(self) -> None:
        """Invalid format (no =) raises BadParameter."""
        with pytest.raises(typer.BadParameter):
            _parse_extra_params(["invalid_no_equals"])


class TestValidateRequiredParams:
    """Tests for _validate_required_params function using real ingestors."""

    def test_all_required_present_passes(self, beir_ingestor_meta: IngestorMeta) -> None:
        """Passes when all required params are provided."""
        # beir requires dataset_name
        extra_params = {"dataset_name": "scifact"}

        # Should not raise
        _validate_required_params(beir_ingestor_meta, extra_params)

    def test_missing_required_raises_exit(self, beir_ingestor_meta: IngestorMeta) -> None:
        """Raises typer.Exit when required param is missing."""
        import click.exceptions

        extra_params = {}  # Missing dataset_name which is required for beir

        with pytest.raises((SystemExit, click.exceptions.Exit)):
            _validate_required_params(beir_ingestor_meta, extra_params)

    def test_optional_missing_passes(self, mteb_ingestor_meta: IngestorMeta) -> None:
        """Passes when optional params are missing."""
        # mteb requires task_name, but score_threshold and include_instruction are optional
        extra_params = {"task_name": "test_task"}

        # Should not raise
        _validate_required_params(mteb_ingestor_meta, extra_params)


class TestConvertParamValue:
    """Tests for _convert_param_value function using real param types."""

    def test_string_type_returns_string(self, beir_ingestor_meta: IngestorMeta) -> None:
        """String type returns value as-is."""
        # beir's dataset_name is a string param
        dataset_name_param = beir_ingestor_meta.params[0]

        result = _convert_param_value("scifact", dataset_name_param)

        assert result == "scifact"
        assert isinstance(result, str)

    def test_int_type_converts_to_int(self, mteb_ingestor_meta: IngestorMeta) -> None:
        """Int type converts string to int."""
        # mteb's score_threshold is an int param
        score_threshold_param = next(p for p in mteb_ingestor_meta.params if p.name == "score_threshold")

        result = _convert_param_value("42", score_threshold_param)

        assert result == 42
        assert isinstance(result, int)

    def test_bool_type_true_values(self, mteb_ingestor_meta: IngestorMeta) -> None:
        """Bool type recognizes true values."""
        # mteb's include_instruction is a bool param
        include_instruction_param = next(p for p in mteb_ingestor_meta.params if p.name == "include_instruction")

        for true_val in ["true", "True", "1", "yes", "Yes"]:
            result = _convert_param_value(true_val, include_instruction_param)
            assert result is True

    def test_bool_type_false_values(self, mteb_ingestor_meta: IngestorMeta) -> None:
        """Bool type treats other values as False."""
        # mteb's include_instruction is a bool param
        include_instruction_param = next(p for p in mteb_ingestor_meta.params if p.name == "include_instruction")

        for false_val in ["false", "False", "0", "no", "No", ""]:
            result = _convert_param_value(false_val, include_instruction_param)
            assert result is False

    def test_list_type_splits_comma(self, param_meta_list: ParamMeta) -> None:
        """List type splits comma-separated values."""
        result = _convert_param_value("a,b,c", param_meta_list)

        assert result == ["a", "b", "c"]
        assert isinstance(result, list)

    def test_list_type_strips_whitespace(self, param_meta_list: ParamMeta) -> None:
        """List type strips whitespace around values."""
        result = _convert_param_value("a, b, c", param_meta_list)

        assert result == ["a", "b", "c"]


class TestGenerateDbName:
    """Tests for generate_db_name function."""

    def test_simple_case(self) -> None:
        """Generates name from simple parameters."""
        result = generate_db_name(
            ingestor_name="beir",
            params={"dataset_name": "scifact"},
            subset="test",
            embedding_model="bge-small",
        )

        assert "beir" in result
        assert "scifact" in result
        assert "test" in result
        assert "bge" in result

    def test_converts_kebab_to_snake(self) -> None:
        """Converts kebab-case to snake_case."""
        result = generate_db_name(
            ingestor_name="beir",
            params={"dataset_name": "nf-corpus"},
            subset="test",
            embedding_model="openai-large",
        )

        assert "-" not in result
        assert "_" in result

    def test_preserves_case_from_params(self) -> None:
        """Preserves case from parameter values (lowercase recommended by convention)."""
        result = generate_db_name(
            ingestor_name="beir",
            params={"dataset_name": "scifact"},
            subset="test",
            embedding_model="bge-small",
        )

        # Function preserves case; users should provide lowercase
        assert "beir" in result
        assert "scifact" in result

    def test_list_params_joined(self) -> None:
        """List parameters are joined."""
        result = generate_db_name(
            ingestor_name="ragbench",
            params={"configs": ["covidqa", "msmarco"]},
            subset="test",
            embedding_model="bge",
        )

        assert "covidqa" in result
        assert "msmarco" in result

    def test_none_values_skipped(self) -> None:
        """None values are skipped."""
        result = generate_db_name(
            ingestor_name="beir",
            params={"dataset_name": "scifact", "optional": None},
            subset="test",
            embedding_model="bge",
        )

        assert "none" not in result.lower()

    def test_parts_joined_with_underscore(self) -> None:
        """Parts are joined with underscore."""
        result = generate_db_name(
            ingestor_name="beir",
            params={"dataset_name": "scifact"},
            subset="test",
            embedding_model="bge",
        )

        # Should have underscores separating parts
        parts = result.split("_")
        assert len(parts) >= 3


class _StubTextIngestor(TextEmbeddingDataIngestor):
    """Stub ingestor for testing CLI flow without real data."""

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        min_corpus_cnt: int | None = None,
    ) -> None:
        pass

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        return "bigint"


def _make_stub_meta() -> IngestorMeta:
    """Create an IngestorMeta pointing to _StubTextIngestor with no required params."""
    return IngestorMeta(
        name="stub",
        ingestor_class=_StubTextIngestor,
        description="stub ingestor for testing",
        params=[],
        hf_repo=None,
    )


def _make_mock_db_conn(create_returns: bool | list[bool]) -> MagicMock:
    """Create a mock DBConnection with controlled create_database behavior."""
    mock_conn = MagicMock()
    if isinstance(create_returns, list):
        mock_conn.create_database.side_effect = create_returns
    else:
        mock_conn.create_database.return_value = create_returns
    mock_conn.host = "localhost"
    mock_conn.port = 5432
    mock_conn.database = "test_db"
    mock_conn.create_schema.return_value = MagicMock()
    mock_conn.get_session_factory.return_value = MagicMock()
    return mock_conn


class TestIngestOverwriteBehavior:
    """Tests for --overwrite flag behavior during database creation."""

    @patch("autorag_research.data.registry.get_ingestor", return_value=_make_stub_meta())
    @patch("autorag_research.cli.commands.ingest.DBConnection")
    @patch("autorag_research.cli.commands.ingest.load_embedding_model", return_value=MagicMock(spec=Embeddings))
    @patch("autorag_research.cli.commands.ingest.health_check_embedding", return_value=768)
    def test_existing_db_without_overwrite_exits_with_error(
        self,
        _mock_health: MagicMock,
        _mock_embed: MagicMock,
        mock_db_cls: MagicMock,
        _mock_get_ingestor: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Exits with error when database already exists and --overwrite is not set."""
        mock_conn = _make_mock_db_conn(create_returns=False)
        mock_db_cls.from_config.return_value = mock_conn

        result = cli_runner.invoke(app, ["ingest", "-n", "stub"])

        assert result.exit_code == 1
        assert "already exists" in result.output
        assert "--overwrite" in result.output
        mock_conn.drop_database.assert_not_called()

    @patch("autorag_research.data.registry.get_ingestor", return_value=_make_stub_meta())
    @patch("autorag_research.cli.commands.ingest.DBConnection")
    @patch("autorag_research.cli.commands.ingest.load_embedding_model", return_value=MagicMock(spec=Embeddings))
    @patch("autorag_research.cli.commands.ingest.health_check_embedding", return_value=768)
    def test_existing_db_with_overwrite_drops_and_recreates(
        self,
        _mock_health: MagicMock,
        _mock_embed: MagicMock,
        mock_db_cls: MagicMock,
        _mock_get_ingestor: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Drops and recreates database when --overwrite is set and database exists."""
        mock_conn = _make_mock_db_conn(create_returns=[False, True])
        mock_db_cls.from_config.return_value = mock_conn

        result = cli_runner.invoke(app, ["ingest", "-n", "stub", "--overwrite", "--skip-embedding"])

        assert result.exit_code == 0, result.output
        mock_conn.terminate_connections.assert_called_once()
        mock_conn.drop_database.assert_called_once()
        assert mock_conn.create_database.call_count == 2

    @patch("autorag_research.data.registry.get_ingestor", return_value=_make_stub_meta())
    @patch("autorag_research.cli.commands.ingest.DBConnection")
    @patch("autorag_research.cli.commands.ingest.load_embedding_model", return_value=MagicMock(spec=Embeddings))
    @patch("autorag_research.cli.commands.ingest.health_check_embedding", return_value=768)
    def test_new_db_proceeds_without_overwrite(
        self,
        _mock_health: MagicMock,
        _mock_embed: MagicMock,
        mock_db_cls: MagicMock,
        _mock_get_ingestor: MagicMock,
        cli_runner: CliRunner,
    ) -> None:
        """Proceeds normally when database is newly created."""
        mock_conn = _make_mock_db_conn(create_returns=True)
        mock_db_cls.from_config.return_value = mock_conn

        result = cli_runner.invoke(app, ["ingest", "-n", "stub", "--skip-embedding"])

        assert result.exit_code == 0, result.output
        mock_conn.drop_database.assert_not_called()
        mock_conn.terminate_connections.assert_not_called()
        mock_conn.create_database.assert_called_once()
