"""Test cases for HuggingFace Hub storage integration.

Tests the hf_storage module functions with mocked HuggingFace Hub API calls.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from autorag_research.data.hf_storage import (
    HF_ORG,
    download_dump,
    dump_exists,
    get_repo_id,
    list_available_dumps,
    upload_dump,
)


class TestGetRepoId:
    """Tests for get_repo_id function."""

    def test_get_repo_id_beir(self):
        """Test getting repo ID for beir ingestor."""
        result = get_repo_id("beir")
        assert result == f"{HF_ORG}/beir-dumps"

    def test_get_repo_id_all_ingestors_with_hf_repo(self):
        """Test that all registered ingestors with hf_repo return valid repo IDs."""
        from autorag_research.data.registry import discover_ingestors

        registry = discover_ingestors()
        for name, meta in registry.items():
            if meta.hf_repo is not None:
                result = get_repo_id(name)
                assert result == f"{HF_ORG}/{meta.hf_repo}"

    def test_get_repo_id_unknown_ingestor(self):
        """Test that unknown ingestor raises KeyError."""
        with pytest.raises(KeyError, match="Unknown ingestor or no HF repo configured"):
            get_repo_id("havertz")


class TestDownloadDump:
    """Tests for download_dump function."""

    @patch("autorag_research.data.hf_storage.hf_hub_download")
    def test_download_dump_success(self, mock_download):
        """Test successful dump download."""
        mock_download.return_value = "/cache/path/scifact_openai-small.dump"

        result = download_dump("beir", "scifact_openai-small")

        assert result == Path("/cache/path/scifact_openai-small.dump")
        mock_download.assert_called_once_with(
            repo_id=f"{HF_ORG}/beir-dumps",
            filename="scifact_openai-small.dump",
            revision=None,
            repo_type="dataset",
            cache_dir=None,
        )

    @patch("autorag_research.data.hf_storage.hf_hub_download")
    def test_download_dump_with_revision(self, mock_download):
        """Test download with specific revision."""
        mock_download.return_value = "/cache/path/scifact_openai-small.dump"

        download_dump("beir", "scifact_openai-small", revision="v1.0")

        mock_download.assert_called_once_with(
            repo_id=f"{HF_ORG}/beir-dumps",
            filename="scifact_openai-small.dump",
            revision="v1.0",
            repo_type="dataset",
            cache_dir=None,
        )

    @patch("autorag_research.data.hf_storage.hf_hub_download")
    def test_download_dump_with_cache_dir(self, mock_download):
        """Test download with custom cache directory."""
        mock_download.return_value = "/custom/cache/scifact_openai-small.dump"

        download_dump("beir", "scifact_openai-small", cache_dir="/custom/cache")

        mock_download.assert_called_once_with(
            repo_id=f"{HF_ORG}/beir-dumps",
            filename="scifact_openai-small.dump",
            revision=None,
            repo_type="dataset",
            cache_dir="/custom/cache",
        )

    def test_download_dump_unknown_ingestor(self):
        """Test that unknown ingestor raises KeyError."""
        with pytest.raises(KeyError, match="Unknown ingestor"):
            download_dump("unknown", "some_filename")


class TestUploadDump:
    """Tests for upload_dump function."""

    @patch("autorag_research.data.hf_storage.upload_file")
    def test_upload_dump_success(self, mock_upload, tmp_path):
        """Test successful dump upload."""
        # Create a temporary dump file
        dump_file = tmp_path / "test.dump"
        dump_file.write_text("test content")

        mock_upload.return_value = (
            "https://huggingface.co/datasets/NomaDamas/beir-dumps/blob/main/scifact_openai-small.dump"
        )

        result = upload_dump(dump_file, "beir", "scifact_openai-small")

        assert "huggingface.co" in result
        mock_upload.assert_called_once_with(
            path_or_fileobj=str(dump_file),
            path_in_repo="scifact_openai-small.dump",
            repo_id=f"{HF_ORG}/beir-dumps",
            repo_type="dataset",
            commit_message="Add scifact_openai-small dump",
        )

    @patch("autorag_research.data.hf_storage.upload_file")
    def test_upload_dump_custom_message(self, mock_upload, tmp_path):
        """Test upload with custom commit message."""
        dump_file = tmp_path / "test.dump"
        dump_file.write_text("test content")

        mock_upload.return_value = "https://example.com"

        upload_dump(
            dump_file,
            "beir",
            "scifact_openai-small",
            commit_message="Custom commit message",
        )

        mock_upload.assert_called_once()
        call_kwargs = mock_upload.call_args[1]
        assert call_kwargs["commit_message"] == "Custom commit message"

    def test_upload_dump_file_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Dump file not found"):
            upload_dump("/nonexistent/file.dump", "beir", "scifact_openai-small")

    def test_upload_dump_unknown_ingestor(self, tmp_path):
        """Test that unknown ingestor raises KeyError."""
        dump_file = tmp_path / "test.dump"
        dump_file.write_text("test content")

        with pytest.raises(KeyError, match="Unknown ingestor"):
            upload_dump(dump_file, "unknown", "some_filename")


class TestListAvailableDumps:
    """Tests for list_available_dumps function."""

    @patch("autorag_research.data.hf_storage.list_repo_files")
    def test_list_available_dumps_success(self, mock_list):
        """Test listing dumps from repository."""
        mock_list.return_value = [
            "scifact_openai-small.dump",
            "nfcorpus_openai-small.dump",
            "README.md",
            ".gitattributes",
        ]

        result = list_available_dumps("beir")

        # Should return filenames without .dump extension
        assert result == ["scifact_openai-small", "nfcorpus_openai-small"]
        mock_list.assert_called_once_with(
            repo_id=f"{HF_ORG}/beir-dumps",
            repo_type="dataset",
        )

    @patch("autorag_research.data.hf_storage.list_repo_files")
    def test_list_available_dumps_empty(self, mock_list):
        """Test listing dumps from empty repository."""
        mock_list.return_value = ["README.md", ".gitattributes"]

        result = list_available_dumps("beir")

        assert result == []

    def test_list_available_dumps_unknown_ingestor(self):
        """Test that unknown ingestor raises KeyError."""
        with pytest.raises(KeyError, match="Unknown ingestor"):
            list_available_dumps("unknown")


class TestDumpExists:
    """Tests for dump_exists function."""

    @patch("autorag_research.data.hf_storage.list_repo_files")
    @patch("autorag_research.data.hf_storage.repo_exists")
    def test_dump_exists_true(self, mock_repo_exists, mock_list):
        """Test that dump_exists returns True when file exists."""
        mock_repo_exists.return_value = True
        mock_list.return_value = ["scifact_openai-small.dump", "nfcorpus_openai-small.dump"]

        result = dump_exists("beir", "scifact_openai-small")

        assert result is True

    @patch("autorag_research.data.hf_storage.list_repo_files")
    @patch("autorag_research.data.hf_storage.repo_exists")
    def test_dump_exists_false(self, mock_repo_exists, mock_list):
        """Test that dump_exists returns False when file doesn't exist."""
        mock_repo_exists.return_value = True
        mock_list.return_value = ["other_openai-small.dump"]

        result = dump_exists("beir", "scifact_openai-small")

        assert result is False

    @patch("autorag_research.data.hf_storage.repo_exists")
    def test_dump_exists_repo_not_found(self, mock_repo_exists):
        """Test that dump_exists returns False when repo doesn't exist."""
        mock_repo_exists.return_value = False

        result = dump_exists("beir", "scifact_openai-small")

        assert result is False

    @patch("autorag_research.data.hf_storage.list_repo_files")
    @patch("autorag_research.data.hf_storage.repo_exists")
    def test_dump_exists_exception(self, mock_repo_exists, mock_list):
        """Test that dump_exists returns False on exception."""
        mock_repo_exists.return_value = True
        mock_list.side_effect = Exception("Network error")

        result = dump_exists("beir", "scifact_openai-small")

        assert result is False

    def test_dump_exists_unknown_ingestor(self):
        """Test that unknown ingestor raises KeyError."""
        with pytest.raises(KeyError, match="Unknown ingestor"):
            dump_exists("unknown", "some_filename")


class TestConstants:
    """Tests for module constants."""

    def test_hf_org_value(self):
        """Test HF_ORG constant."""
        assert HF_ORG == "NomaDamas"

    def test_registry_has_expected_ingestors_with_hf_repo(self):
        """Test that registry has expected ingestors with hf_repo configured."""
        from autorag_research.data.registry import discover_ingestors

        registry = discover_ingestors()
        ingestors_with_hf_repo = {name for name, meta in registry.items() if meta.hf_repo is not None}

        # These are the ingestors that should have hf_repo configured
        expected = {
            "beir",
            "mrtydi",
            "ragbench",
            "bright",
            "visrag",
            "vidorev2",
            "open-ragbench",
            "mteb",
        }
        assert expected.issubset(ingestors_with_hf_repo)

    def test_hf_repo_values_end_with_dumps(self):
        """Test that all hf_repo values end with '-dumps'."""
        from autorag_research.data.registry import discover_ingestors

        registry = discover_ingestors()
        for name, meta in registry.items():
            if meta.hf_repo is not None:
                assert meta.hf_repo.endswith("-dumps"), f"{name}: {meta.hf_repo} should end with '-dumps'"
