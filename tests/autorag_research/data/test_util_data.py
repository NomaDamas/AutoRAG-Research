"""Tests for autorag_research.data.util module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from autorag_research.data.util import setup_dataset


class TestSetupDataset:
    """Tests for setup_dataset function."""

    @patch("autorag_research.data.util.restore_database")
    @patch("autorag_research.data.util.download_dump")
    def test_setup_dataset_with_explicit_ingestor(self, mock_download, mock_restore):
        """Test setup_dataset with explicitly provided ingestor."""
        mock_download.return_value = Path("/cache/nfcorpus_openai-small.dump")

        setup_dataset(
            ingestor_name="beir",
            dataset_name="nfcorpus",
            embedding_model_name="openai-small",
            host="localhost",
            user="postgres",
            password="postgres",  # noqa: S106
        )

        mock_download.assert_called_once_with(
            ingestor="beir",
            filename="nfcorpus_openai-small",
        )

    def test_setup_dataset_unknown_ingestor(self):
        """Test that ValueError is raised for unknown ingestor."""
        with pytest.raises(ValueError, match="Unknown ingestor"):
            setup_dataset(
                ingestor_name="unknown_ingestor",
                dataset_name="test",
                embedding_model_name="test",
                host="localhost",
                user="postgres",
                password="postgres",  # noqa: S106
            )

    @patch("autorag_research.data.util.restore_database")
    @patch("autorag_research.data.util.download_dump")
    def test_setup_dataset_passes_kwargs_to_restore(self, mock_download, mock_restore):
        """Test that extra kwargs are passed to restore_database."""
        mock_download.return_value = Path("/cache/test.dump")

        setup_dataset(
            ingestor_name="beir",
            dataset_name="scifact",
            embedding_model_name="embeddinggemma-300m",
            host="localhost",
            user="postgres",
            password="postgres",  # noqa: S106
            port=5433,
            clean=True,
            no_owner=True,
        )

        mock_download.assert_called_once_with(
            ingestor="beir",
            filename="scifact_embeddinggemma-300m",
        )
        mock_restore.assert_called_once()
        call_kwargs = mock_restore.call_args[1]
        assert call_kwargs["port"] == 5433
        assert call_kwargs["clean"] is True
        assert call_kwargs["no_owner"] is True
