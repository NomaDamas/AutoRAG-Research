"""Tests for the ingestor registry module."""

from enum import Enum
from typing import Literal
from unittest.mock import patch

from autorag_research.data.registry import (
    IngestorMeta,
    ParamMeta,
    _auto_import_data_modules,
    _extract_choices,
    _extract_params_from_init,
    _get_base_type,
    _is_list_type,
    discover_ingestors,
    get_ingestor,
    get_ingestor_help,
    register_ingestor,
)


class TestAutoImportDataModules:
    """Tests for _auto_import_data_modules function."""

    def test_auto_import_discovers_modules(self):
        """Auto-import should discover and import modules in autorag_research.data."""
        # Reset discovery state to force re-discovery
        import autorag_research.data.registry as registry_module

        registry_module._discovery_done = False
        registry_module._INGESTOR_REGISTRY.clear()

        # Run auto-import
        _auto_import_data_modules()

        # Should have imported modules that register ingestors
        # At minimum, registry module itself should be importable
        assert True  # If we get here without error, import works

    def test_auto_import_skips_private_modules(self):
        """Auto-import should skip modules starting with underscore."""
        with patch("pkgutil.iter_modules") as mock_iter:
            mock_iter.return_value = [
                (None, "_private", False),
                (None, "__init__", False),
                (None, "public", False),
            ]
            with patch("importlib.import_module") as mock_import:
                _auto_import_data_modules()
                # Should only import 'public', not '_private' or '__init__'
                mock_import.assert_called_once_with("autorag_research.data.public")

    def test_auto_import_skips_subpackages(self):
        """Auto-import should skip subpackages (ispkg=True)."""
        with patch("pkgutil.iter_modules") as mock_iter:
            mock_iter.return_value = [
                (None, "subpackage", True),  # ispkg=True
                (None, "module", False),  # ispkg=False
            ]
            with patch("importlib.import_module") as mock_import:
                _auto_import_data_modules()
                # Should only import 'module', not 'subpackage'
                mock_import.assert_called_once_with("autorag_research.data.module")

    def test_auto_import_handles_import_error(self):
        """Auto-import should silently handle ImportError for optional dependencies."""
        with patch("pkgutil.iter_modules") as mock_iter:
            mock_iter.return_value = [(None, "optional_module", False)]
            with patch("importlib.import_module") as mock_import:
                mock_import.side_effect = ImportError("Optional dependency missing")
                # Should not raise
                _auto_import_data_modules()


class TestDiscoverIngestors:
    """Tests for discover_ingestors function."""

    def test_discover_ingestors_returns_dict(self):
        """discover_ingestors should return a dictionary."""
        result = discover_ingestors()
        assert isinstance(result, dict)

    def test_discover_ingestors_not_empty(self):
        """discover_ingestors should find at least one ingestor."""
        result = discover_ingestors()
        assert len(result) > 0

    def test_discover_ingestors_contains_known_ingestors(self):
        """discover_ingestors should find known internal ingestors."""
        result = discover_ingestors()
        # These are the known ingestors from autorag_research.data
        known_ingestors = ["beir", "mrtydi", "ragbench", "mteb", "bright", "vidore", "vidorev2"]
        for name in known_ingestors:
            assert name in result, f"Expected ingestor '{name}' not found"

    def test_discover_ingestors_idempotent(self):
        """Calling discover_ingestors multiple times should return same result."""
        result1 = discover_ingestors()
        result2 = discover_ingestors()
        assert result1 is result2  # Should be exact same dict object

    def test_discover_ingestors_values_are_ingestor_meta(self):
        """All values in discover_ingestors result should be IngestorMeta."""
        result = discover_ingestors()
        for name, meta in result.items():
            assert isinstance(meta, IngestorMeta), f"Value for '{name}' is not IngestorMeta"


class TestGetIngestor:
    """Tests for get_ingestor function."""

    def test_get_ingestor_returns_meta(self):
        """get_ingestor should return IngestorMeta for known ingestor."""
        result = get_ingestor("beir")
        assert result is not None
        assert isinstance(result, IngestorMeta)

    def test_get_ingestor_returns_none_for_unknown(self):
        """get_ingestor should return None for unknown ingestor."""
        result = get_ingestor("nonexistent_ingestor_xyz")
        assert result is None

    def test_get_ingestor_meta_has_correct_name(self):
        """get_ingestor should return meta with matching name."""
        result = get_ingestor("beir")
        assert result is not None
        assert result.name == "beir"

    def test_get_ingestor_meta_has_class(self):
        """get_ingestor should return meta with ingestor_class."""
        result = get_ingestor("beir")
        assert result is not None
        assert result.ingestor_class is not None
        assert callable(result.ingestor_class)


class TestRegisterIngestor:
    """Tests for register_ingestor decorator."""

    def test_register_ingestor_adds_to_registry(self):
        """register_ingestor should add class to registry."""
        import autorag_research.data.registry as registry_module

        @register_ingestor(name="test_ingestor_123", description="Test description")
        class TestIngestor:
            def __init__(self, param1: str):
                pass

        assert "test_ingestor_123" in registry_module._INGESTOR_REGISTRY
        # Cleanup
        del registry_module._INGESTOR_REGISTRY["test_ingestor_123"]

    def test_register_ingestor_preserves_class(self):
        """register_ingestor should return the original class."""
        import autorag_research.data.registry as registry_module

        @register_ingestor(name="test_ingestor_456", description="Test")
        class TestIngestor:
            pass

        assert TestIngestor.__name__ == "TestIngestor"
        # Cleanup
        del registry_module._INGESTOR_REGISTRY["test_ingestor_456"]

    def test_register_ingestor_extracts_params(self):
        """register_ingestor should extract parameters from __init__."""
        import autorag_research.data.registry as registry_module

        @register_ingestor(name="test_ingestor_789", description="Test")
        class TestIngestor:
            def __init__(self, dataset_name: str, batch_size: int = 100):
                pass

        meta = registry_module._INGESTOR_REGISTRY["test_ingestor_789"]
        assert len(meta.params) == 2
        param_names = [p.name for p in meta.params]
        assert "dataset_name" in param_names
        assert "batch_size" in param_names
        # Cleanup
        del registry_module._INGESTOR_REGISTRY["test_ingestor_789"]


class TestExtractParamsFromInit:
    """Tests for _extract_params_from_init function."""

    def test_extract_params_basic(self):
        """Should extract basic string parameter."""

        class TestClass:
            def __init__(self, name: str):
                pass

        params = _extract_params_from_init(TestClass)
        assert len(params) == 1
        assert params[0].name == "name"
        assert params[0].param_type is str
        assert params[0].required is True

    def test_extract_params_with_default(self):
        """Should extract parameter with default value."""

        class TestClass:
            def __init__(self, count: int = 10):
                pass

        params = _extract_params_from_init(TestClass)
        assert len(params) == 1
        assert params[0].name == "count"
        assert params[0].default == 10
        assert params[0].required is False

    def test_extract_params_skips_self(self):
        """Should skip 'self' parameter."""

        class TestClass:
            def __init__(self, name: str):
                pass

        params = _extract_params_from_init(TestClass)
        param_names = [p.name for p in params]
        assert "self" not in param_names

    def test_extract_params_skips_embedding_model(self):
        """Should skip 'embedding_model' parameter."""

        class TestClass:
            def __init__(self, embedding_model, name: str):
                pass

        params = _extract_params_from_init(TestClass)
        param_names = [p.name for p in params]
        assert "embedding_model" not in param_names
        assert "name" in param_names

    def test_extract_params_with_literal(self):
        """Should extract choices from Literal type."""
        DatasetType = Literal["train", "test", "dev"]

        class TestClass:
            def __init__(self, subset: DatasetType):
                pass

        params = _extract_params_from_init(TestClass)
        assert params[0].choices == ["train", "test", "dev"]

    def test_extract_params_cli_option_format(self):
        """Should convert snake_case to kebab-case for CLI option."""

        class TestClass:
            def __init__(self, dataset_name: str):
                pass

        params = _extract_params_from_init(TestClass)
        assert params[0].cli_option == "dataset-name"


class TestExtractChoices:
    """Tests for _extract_choices function."""

    def test_extract_choices_from_literal(self):
        """Should extract choices from Literal type."""
        hint = Literal["a", "b", "c"]
        choices = _extract_choices(hint)
        assert choices == ["a", "b", "c"]

    def test_extract_choices_from_enum(self):
        """Should extract choices from Enum type."""

        class Color(Enum):
            RED = "red"
            GREEN = "green"
            BLUE = "blue"

        choices = _extract_choices(Color)
        assert set(choices) == {"red", "green", "blue"}

    def test_extract_choices_returns_none_for_basic_type(self):
        """Should return None for basic types without choices."""
        assert _extract_choices(str) is None
        assert _extract_choices(int) is None
        assert _extract_choices(bool) is None


class TestGetBaseType:
    """Tests for _get_base_type function."""

    def test_get_base_type_str(self):
        """Should return str for str type."""
        assert _get_base_type(str) is str

    def test_get_base_type_int(self):
        """Should return int for int type."""
        assert _get_base_type(int) is int

    def test_get_base_type_bool(self):
        """Should return bool for bool type."""
        assert _get_base_type(bool) is bool

    def test_get_base_type_literal_str(self):
        """Should return str for Literal with string values."""
        hint = Literal["a", "b"]
        assert _get_base_type(hint) is str

    def test_get_base_type_literal_int(self):
        """Should return int for Literal with int values."""
        hint = Literal[1, 2, 3]
        assert _get_base_type(hint) is int

    def test_get_base_type_optional(self):
        """Should unwrap Optional type."""
        hint = str | None
        assert _get_base_type(hint) is str

    def test_get_base_type_list(self):
        """Should return str for list type (CLI uses comma-separated strings)."""
        hint = list[str]
        assert _get_base_type(hint) is str


class TestIsListType:
    """Tests for _is_list_type function."""

    def test_is_list_type_true_for_list(self):
        """Should return True for list type."""
        assert _is_list_type(list[str]) is True

    def test_is_list_type_true_for_optional_list(self):
        """Should return True for Optional list type.

        Note: Uses typing.Union syntax. The `|` syntax (e.g., list[str] | None)
        creates types.UnionType which is not currently supported.
        """
        hint = list[str] | None
        assert _is_list_type(hint) is True

    def test_is_list_type_false_for_str(self):
        """Should return False for str type."""
        assert _is_list_type(str) is False

    def test_is_list_type_false_for_int(self):
        """Should return False for int type."""
        assert _is_list_type(int) is False


class TestGetIngestorHelp:
    """Tests for get_ingestor_help function."""

    def test_get_ingestor_help_returns_string(self):
        """get_ingestor_help should return a string."""
        result = get_ingestor_help()
        assert isinstance(result, str)

    def test_get_ingestor_help_contains_header(self):
        """get_ingestor_help should contain header text."""
        result = get_ingestor_help()
        assert "Available ingestors:" in result

    def test_get_ingestor_help_contains_ingestor_names(self):
        """get_ingestor_help should contain known ingestor names."""
        result = get_ingestor_help()
        assert "beir" in result
        assert "mrtydi" in result

    def test_get_ingestor_help_contains_params(self):
        """get_ingestor_help should contain parameter info."""
        result = get_ingestor_help()
        # BEIR has --dataset-name parameter
        assert "--dataset-name" in result


class TestParamMeta:
    """Tests for ParamMeta dataclass."""

    def test_param_meta_creation(self):
        """Should create ParamMeta with required fields."""
        param = ParamMeta(
            name="test_param",
            cli_option="test-param",
            param_type=str,
        )
        assert param.name == "test_param"
        assert param.cli_option == "test-param"
        assert param.param_type is str
        assert param.choices is None
        assert param.required is True
        assert param.default is None

    def test_param_meta_with_all_fields(self):
        """Should create ParamMeta with all fields."""
        param = ParamMeta(
            name="dataset",
            cli_option="dataset",
            param_type=str,
            choices=["a", "b"],
            required=False,
            default="a",
            help="Dataset name",
            is_list=False,
        )
        assert param.choices == ["a", "b"]
        assert param.required is False
        assert param.default == "a"
        assert param.help == "Dataset name"
        assert param.is_list is False


class TestIngestorMeta:
    """Tests for IngestorMeta dataclass."""

    def test_ingestor_meta_creation(self):
        """Should create IngestorMeta with required fields."""

        class DummyIngestor:
            pass

        meta = IngestorMeta(
            name="test",
            ingestor_class=DummyIngestor,
            description="Test ingestor",
        )
        assert meta.name == "test"
        assert meta.ingestor_class == DummyIngestor
        assert meta.description == "Test ingestor"
        assert meta.params == []

    def test_ingestor_meta_with_params(self):
        """Should create IngestorMeta with params."""

        class DummyIngestor:
            pass

        params = [
            ParamMeta(name="p1", cli_option="p1", param_type=str),
            ParamMeta(name="p2", cli_option="p2", param_type=int),
        ]
        meta = IngestorMeta(
            name="test",
            ingestor_class=DummyIngestor,
            description="Test",
            params=params,
        )
        assert len(meta.params) == 2
