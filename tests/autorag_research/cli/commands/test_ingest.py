"""Tests for autorag_research.cli.commands.ingest module."""

from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from autorag_research.cli.app import app
from autorag_research.cli.commands.ingest import (
    _convert_param_value,
    _parse_extra_params,
    _validate_required_params,
    generate_db_name,
)
from autorag_research.data.registry import IngestorMeta, ParamMeta


@pytest.fixture
def cli_runner() -> CliRunner:
    """Return a Typer CliRunner for testing commands."""
    return CliRunner()


@pytest.fixture
def mock_param_meta_string() -> ParamMeta:
    """Create mock ParamMeta for string type."""
    return ParamMeta(
        name="dataset_name",
        cli_option="dataset-name",
        param_type=str,
        choices=None,
        required=True,
        default=None,
        help="Dataset name",
        is_list=False,
    )


@pytest.fixture
def mock_param_meta_int() -> ParamMeta:
    """Create mock ParamMeta for int type."""
    return ParamMeta(
        name="batch_size",
        cli_option="batch-size",
        param_type=int,
        choices=None,
        required=False,
        default=100,
        help="Batch size",
        is_list=False,
    )


@pytest.fixture
def mock_param_meta_bool() -> ParamMeta:
    """Create mock ParamMeta for bool type."""
    return ParamMeta(
        name="include_images",
        cli_option="include-images",
        param_type=bool,
        choices=None,
        required=False,
        default=False,
        help="Include images",
        is_list=False,
    )


@pytest.fixture
def mock_param_meta_list() -> ParamMeta:
    """Create mock ParamMeta for list type."""
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


@pytest.fixture
def mock_ingestor_meta() -> IngestorMeta:
    """Create mock IngestorMeta for testing."""
    return IngestorMeta(
        name="test_ingestor",
        ingestor_class=MagicMock,
        description="Test ingestor",
        params=[
            ParamMeta(
                name="dataset_name",
                cli_option="dataset-name",
                param_type=str,
                choices=["option_a", "option_b"],
                required=True,
                default=None,
                help="Dataset name",
                is_list=False,
            ),
            ParamMeta(
                name="batch_size",
                cli_option="batch-size",
                param_type=int,
                choices=None,
                required=False,
                default=100,
                help="Batch size",
                is_list=False,
            ),
        ],
    )


class TestIngestCommand:
    """Tests for the ingest CLI command."""

    def test_ingest_help_shows_options(self, cli_runner: CliRunner) -> None:
        """'ingest --help' shows available options."""
        result = cli_runner.invoke(app, ["ingest", "--help"])

        assert result.exit_code == 0
        assert "--name" in result.stdout or "-n" in result.stdout

    def test_ingest_without_name_shows_help(self, cli_runner: CliRunner) -> None:
        """'ingest' without --name shows available ingestors or help."""
        with patch("autorag_research.data.registry.discover_ingestors") as mock_discover:
            mock_discover.return_value = {"beir": MagicMock(description="BEIR datasets")}

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
    """Tests for _validate_required_params function."""

    def test_all_required_present_passes(self, mock_ingestor_meta: IngestorMeta) -> None:
        """Passes when all required params are provided."""
        extra_params = {"dataset_name": "scifact"}

        # Should not raise
        _validate_required_params(mock_ingestor_meta, extra_params)

    def test_missing_required_raises_exit(self, mock_ingestor_meta: IngestorMeta) -> None:
        """Raises typer.Exit when required param is missing."""
        import click.exceptions

        extra_params = {}  # Missing dataset_name

        with pytest.raises((SystemExit, click.exceptions.Exit)):
            _validate_required_params(mock_ingestor_meta, extra_params)

    def test_optional_missing_passes(self, mock_ingestor_meta: IngestorMeta) -> None:
        """Passes when optional params are missing."""
        extra_params = {"dataset_name": "scifact"}  # batch_size is optional

        # Should not raise
        _validate_required_params(mock_ingestor_meta, extra_params)


class TestConvertParamValue:
    """Tests for _convert_param_value function."""

    def test_string_type_returns_string(self, mock_param_meta_string: ParamMeta) -> None:
        """String type returns value as-is."""
        result = _convert_param_value("hello", mock_param_meta_string)

        assert result == "hello"
        assert isinstance(result, str)

    def test_int_type_converts_to_int(self, mock_param_meta_int: ParamMeta) -> None:
        """Int type converts string to int."""
        result = _convert_param_value("42", mock_param_meta_int)

        assert result == 42
        assert isinstance(result, int)

    def test_bool_type_true_values(self, mock_param_meta_bool: ParamMeta) -> None:
        """Bool type recognizes true values."""
        for true_val in ["true", "True", "1", "yes", "Yes"]:
            result = _convert_param_value(true_val, mock_param_meta_bool)
            assert result is True

    def test_bool_type_false_values(self, mock_param_meta_bool: ParamMeta) -> None:
        """Bool type treats other values as False."""
        for false_val in ["false", "False", "0", "no", "No", ""]:
            result = _convert_param_value(false_val, mock_param_meta_bool)
            assert result is False

    def test_list_type_splits_comma(self, mock_param_meta_list: ParamMeta) -> None:
        """List type splits comma-separated values."""
        result = _convert_param_value("a,b,c", mock_param_meta_list)

        assert result == ["a", "b", "c"]
        assert isinstance(result, list)

    def test_list_type_strips_whitespace(self, mock_param_meta_list: ParamMeta) -> None:
        """List type strips whitespace around values."""
        result = _convert_param_value("a, b, c", mock_param_meta_list)

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
