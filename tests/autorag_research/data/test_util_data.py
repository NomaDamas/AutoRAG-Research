"""Tests for autorag_research.data.util module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from autorag_research.data.util import (
    DATASET_REGISTRY,
    _find_ingestor_for_dataset,
    setup_dataset,
)


class TestFindIngestorForDataset:
    """Tests for _find_ingestor_for_dataset function."""

    def test_find_ingestor_for_known_dataset(self):
        """Test finding ingestor for a dataset in the registry."""
        # scifact is in the DATASET_REGISTRY under "beir"
        result = _find_ingestor_for_dataset("scifact")
        assert result == "beir"

    def test_find_ingestor_for_unknown_dataset(self):
        """Test that None is returned for unknown datasets."""
        result = _find_ingestor_for_dataset("nonexistent_dataset")
        assert result is None


class TestSetupDataset:
    """Tests for setup_dataset function."""

    @patch("autorag_research.data.util.restore_database")
    @patch("autorag_research.data.util.download_dump")
    def test_setup_dataset_with_auto_detect(self, mock_download, mock_restore):
        """Test setup_dataset with auto-detected ingestor."""
        mock_download.return_value = Path("/cache/scifact_embeddinggemma-300m.dump")

        setup_dataset(
            dataset_name="scifact",
            embedding_model_name="embeddinggemma-300m",
            host="localhost",
            user="postgres",
            password="postgres",  # noqa: S106
        )

        mock_download.assert_called_once_with(
            ingestor="beir",
            dataset="scifact",
            embedding="embeddinggemma-300m",
        )
        mock_restore.assert_called_once()
        call_kwargs = mock_restore.call_args[1]
        assert call_kwargs["database"] == "scifact_embeddinggemma-300m"

    @patch("autorag_research.data.util.restore_database")
    @patch("autorag_research.data.util.download_dump")
    def test_setup_dataset_with_explicit_ingestor(self, mock_download, mock_restore):
        """Test setup_dataset with explicitly provided ingestor."""
        mock_download.return_value = Path("/cache/nfcorpus_openai-small.dump")

        setup_dataset(
            dataset_name="nfcorpus",
            embedding_model_name="openai-small",
            host="localhost",
            user="postgres",
            password="postgres",  # noqa: S106
            ingestor_name="beir",
        )

        mock_download.assert_called_once_with(
            ingestor="beir",
            dataset="nfcorpus",
            embedding="openai-small",
        )

    def test_setup_dataset_unknown_dataset_no_ingestor(self):
        """Test that KeyError is raised for unknown dataset without ingestor."""
        with pytest.raises(KeyError, match="not found"):
            setup_dataset(
                dataset_name="nonexistent",
                embedding_model_name="test",
                host="localhost",
                user="postgres",
                password="postgres",  # noqa: S106
            )

    def test_setup_dataset_unknown_ingestor(self):
        """Test that ValueError is raised for unknown ingestor."""
        with pytest.raises(ValueError, match="Unknown ingestor"):
            setup_dataset(
                dataset_name="test",
                embedding_model_name="test",
                host="localhost",
                user="postgres",
                password="postgres",  # noqa: S106
                ingestor_name="unknown_ingestor",
            )

    @patch("autorag_research.data.util.restore_database")
    @patch("autorag_research.data.util.download_dump")
    def test_setup_dataset_passes_kwargs_to_restore(self, mock_download, mock_restore):
        """Test that extra kwargs are passed to restore_database."""
        mock_download.return_value = Path("/cache/test.dump")

        setup_dataset(
            dataset_name="scifact",
            embedding_model_name="embeddinggemma-300m",
            host="localhost",
            user="postgres",
            password="postgres",  # noqa: S106
            port=5433,
            clean=True,
            no_owner=True,
        )

        mock_restore.assert_called_once()
        call_kwargs = mock_restore.call_args[1]
        assert call_kwargs["port"] == 5433
        assert call_kwargs["clean"] is True
        assert call_kwargs["no_owner"] is True


class TestDatasetRegistry:
    """Tests for DATASET_REGISTRY constant."""

    def test_registry_structure(self):
        """Test that registry has expected structure."""
        assert isinstance(DATASET_REGISTRY, dict)
        for ingestor, datasets in DATASET_REGISTRY.items():
            assert isinstance(ingestor, str)
            assert isinstance(datasets, dict)
            for dataset, embeddings in datasets.items():
                assert isinstance(dataset, str)
                assert isinstance(embeddings, dict)

    def test_registry_has_beir(self):
        """Test that registry contains beir ingestor."""
        assert "beir" in DATASET_REGISTRY
        assert "scifact" in DATASET_REGISTRY["beir"]
