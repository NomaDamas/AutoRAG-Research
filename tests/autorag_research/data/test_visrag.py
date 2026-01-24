"""Tests for VisRAG Dataset Ingestor.

Unit tests for helper functions.
Integration tests use real data subsets against PostgreSQL.

The VisRAG benchmark includes 6 datasets:
- ArxivQA: 816 queries, 8,070 images (academic papers)
- ChartQA: Chart comprehension
- MP-DocVQA: Multi-page document VQA
- InfoVQA: Infographics VQA
- PlotQA: Scientific plots
- SlideVQA: Slide presentations
"""

import pytest

from autorag_research.data.visrag import (
    _DATASET_CONFIGS,
    VisRAGDatasetName,
    VisRAGIngestor,
    _format_query_with_options,
)
from autorag_research.orm.service.multi_modal_ingestion import MultiModalIngestionService
from tests.autorag_research.data.ingestor_test_utils import (
    IngestorTestConfig,
    IngestorTestVerifier,
    create_test_database,
)

# ==================== Unit Tests: Query Formatting ====================


class TestFormatQueryWithOptions:
    """Test _format_query_with_options function."""

    def test_format_query_basic(self):
        question = "What color is the sky?"
        options = ["A. Red", "B. Blue", "C. Green", "D. Yellow"]
        result = _format_query_with_options(question, options)

        assert "Given the following query and options" in result
        assert "Query: What color is the sky?" in result
        assert "Options:" in result
        assert "A. Red" in result
        assert "B. Blue" in result
        assert "C. Green" in result
        assert "D. Yellow" in result

    def test_format_query_with_five_options(self):
        question = "Sample question?"
        options = ["A. One", "B. Two", "C. Three", "D. Four", "E. Five"]
        result = _format_query_with_options(question, options)

        assert "E. Five" in result

    def test_format_query_options_joined_with_newlines(self):
        question = "Question?"
        options = ["A. First", "B. Second"]
        result = _format_query_with_options(question, options)

        # Options should be on separate lines
        assert "A. First\nB. Second" in result


# ==================== Unit Tests: Dataset Configuration ====================


class TestDatasetConfigs:
    """Test dataset configuration mapping."""

    def test_all_datasets_have_configs(self):
        """All enum values should have corresponding configs."""
        for dataset_name in VisRAGDatasetName:
            assert dataset_name in _DATASET_CONFIGS, f"Missing config for {dataset_name}"

    def test_hf_paths_follow_pattern(self):
        """All HuggingFace paths should follow the VisRAG pattern."""
        for dataset_name, config in _DATASET_CONFIGS.items():
            assert config.hf_path.startswith("openbmb/VisRAG-Ret-Test-"), (
                f"Unexpected HF path for {dataset_name}: {config.hf_path}"
            )

    def test_arxivqa_config(self):
        config = _DATASET_CONFIGS[VisRAGDatasetName.ARXIV_QA]
        assert config.hf_path == "openbmb/VisRAG-Ret-Test-ArxivQA"
        assert config.has_options is True
        assert config.supports_multi_answer is False

    def test_mp_docvqa_config(self):
        config = _DATASET_CONFIGS[VisRAGDatasetName.MP_DOCVQA]
        assert config.hf_path == "openbmb/VisRAG-Ret-Test-MP-DocVQA"
        assert config.has_options is False
        assert config.supports_multi_answer is True

    def test_infovqa_config(self):
        config = _DATASET_CONFIGS[VisRAGDatasetName.INFO_VQA]
        assert config.hf_path == "openbmb/VisRAG-Ret-Test-InfographicsVQA"
        assert config.has_options is False
        assert config.supports_multi_answer is True


# ==================== Unit Tests: Ingestor Initialization ====================


class TestVisRAGIngestorInit:
    """Test VisRAGIngestor initialization."""

    def test_creates_with_dataset_name(self):
        ingestor = VisRAGIngestor(VisRAGDatasetName.CHART_QA)
        assert ingestor.dataset_name == VisRAGDatasetName.CHART_QA
        assert ingestor._config == _DATASET_CONFIGS[VisRAGDatasetName.CHART_QA]

    def test_detect_primary_key_type_is_string(self):
        ingestor = VisRAGIngestor(VisRAGDatasetName.ARXIV_QA)
        assert ingestor.detect_primary_key_type() == "string"


class TestVisRAGIngestorFormatQuery:
    """Test _format_query method with different dataset configurations."""

    def test_format_query_with_options(self):
        """Datasets with options should format query with options."""
        ingestor = VisRAGIngestor(VisRAGDatasetName.ARXIV_QA)
        row = {"query": "What is shown?", "options": ["A. Cat", "B. Dog"]}
        result = ingestor._format_query(row)

        assert "Given the following query and options" in result
        assert "A. Cat" in result

    def test_format_query_without_options(self):
        """Datasets without options should return plain query."""
        ingestor = VisRAGIngestor(VisRAGDatasetName.MP_DOCVQA)
        row = {"query": "What is the document about?", "options": None}
        result = ingestor._format_query(row)

        assert result == "What is the document about?"
        assert "Options" not in result

    def test_format_query_missing_options_field(self):
        """Should handle missing options field gracefully."""
        ingestor = VisRAGIngestor(VisRAGDatasetName.SLIDE_VQA)
        row = {"query": "What is shown in the slide?"}
        result = ingestor._format_query(row)

        assert result == "What is shown in the slide?"


class TestVisRAGIngestorExtractAnswers:
    """Test _extract_answers method with different dataset configurations."""

    def test_extract_single_answer(self):
        """Single answer datasets should return list with one answer."""
        ingestor = VisRAGIngestor(VisRAGDatasetName.ARXIV_QA)
        row = {"answer": "B"}
        result = ingestor._extract_answers(row)

        assert result == ["B"]

    def test_extract_multi_answer(self):
        """Multi-answer datasets should return all answers."""
        ingestor = VisRAGIngestor(VisRAGDatasetName.MP_DOCVQA)
        row = {"answer": ["answer1", "answer2", "answer3"]}
        result = ingestor._extract_answers(row)

        assert result == ["answer1", "answer2", "answer3"]

    def test_extract_answer_none(self):
        """Should handle None answer gracefully."""
        ingestor = VisRAGIngestor(VisRAGDatasetName.ARXIV_QA)
        row = {"answer": None}
        result = ingestor._extract_answers(row)

        assert result == []

    def test_extract_answer_missing_field(self):
        """Should handle missing answer field gracefully."""
        ingestor = VisRAGIngestor(VisRAGDatasetName.ARXIV_QA)
        row = {}
        result = ingestor._extract_answers(row)

        assert result == []

    def test_single_answer_dataset_with_list_input(self):
        """Single-answer dataset receiving list should return first as string."""
        ingestor = VisRAGIngestor(VisRAGDatasetName.ARXIV_QA)
        # This tests edge case - single answer dataset receiving list
        row = {"answer": ["A"]}
        result = ingestor._extract_answers(row)
        # Since supports_multi_answer is False, it treats answer as-is (converts to string)
        assert result == ["['A']"]


# ==================== Integration Tests ====================


def create_visrag_test_config(dataset_name: VisRAGDatasetName) -> IngestorTestConfig:
    """Create test config for a VisRAG dataset.

    All VisRAG datasets have BEIR-style format with qrels,
    generation ground truth (answers), and use string primary keys.
    """
    return IngestorTestConfig(
        expected_query_count=10,
        expected_image_chunk_count=50,  # 10+ gold + sampled
        chunk_count_is_minimum=True,  # At least this many (gold IDs always included)
        check_retrieval_relations=True,
        check_generation_gt=True,
        generation_gt_required_for_all=True,
        primary_key_type="string",
        db_name=f"visrag_{dataset_name.value.lower().replace('-', '_')}_test",
    )


@pytest.mark.data
class TestVisRAGIngestorIntegration:
    """Integration tests using real VisRAG dataset subsets."""

    @pytest.mark.parametrize(
        "dataset_name",
        [
            VisRAGDatasetName.ARXIV_QA,
            VisRAGDatasetName.CHART_QA,
            VisRAGDatasetName.MP_DOCVQA,
            VisRAGDatasetName.INFO_VQA,
            VisRAGDatasetName.PLOT_QA,
            VisRAGDatasetName.SLIDE_VQA,
        ],
        ids=["ArxivQA", "ChartQA", "MP-DocVQA", "InfoVQA", "PlotQA", "SlideVQA"],
    )
    def test_ingest_dataset_subset(self, dataset_name: VisRAGDatasetName):
        """Test ingestion for each VisRAG dataset."""
        config = create_visrag_test_config(dataset_name)

        with create_test_database(config) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = VisRAGIngestor(dataset_name)
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=config.expected_query_count,
                min_corpus_cnt=config.expected_image_chunk_count,
            )

            verifier = IngestorTestVerifier(service, db.schema, config)
            verifier.verify_all()


@pytest.mark.data
class TestVisRAGIngestorDatasetSpecific:
    """Dataset-specific integration tests."""

    def test_arxivqa_query_format_has_options(self):
        """ArxivQA queries should include multiple choice options."""
        config = IngestorTestConfig(
            expected_query_count=3,
            expected_image_chunk_count=20,
            chunk_count_is_minimum=True,
            check_retrieval_relations=True,
            check_generation_gt=True,
            generation_gt_required_for_all=True,
            primary_key_type="string",
            db_name="visrag_arxivqa_format_test",
        )

        with create_test_database(config) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = VisRAGIngestor(VisRAGDatasetName.ARXIV_QA)
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=config.expected_query_count,
                min_corpus_cnt=config.expected_image_chunk_count,
            )

            with service._create_uow() as uow:
                queries = uow.queries.get_all(limit=10)
                for query in queries:
                    assert "Query:" in query.contents
                    assert "Options:" in query.contents
                    assert "Given the following query and options" in query.contents

    def test_mp_docvqa_query_format_no_options(self):
        """MP-DocVQA queries should NOT include multiple choice options."""
        config = IngestorTestConfig(
            expected_query_count=3,
            expected_image_chunk_count=20,
            chunk_count_is_minimum=True,
            check_retrieval_relations=True,
            check_generation_gt=True,
            generation_gt_required_for_all=True,
            primary_key_type="string",
            db_name="visrag_mp_docvqa_format_test",
        )

        with create_test_database(config) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = VisRAGIngestor(VisRAGDatasetName.MP_DOCVQA)
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=config.expected_query_count,
                min_corpus_cnt=config.expected_image_chunk_count,
            )

            with service._create_uow() as uow:
                queries = uow.queries.get_all(limit=10)
                for query in queries:
                    # MP-DocVQA should NOT have formatted options
                    assert "Given the following query and options" not in query.contents

    def test_corpus_always_includes_gold_ids(self):
        """Test that corpus filtering always includes gold IDs from qrels."""
        config = IngestorTestConfig(
            expected_query_count=5,
            expected_image_chunk_count=5,  # Minimal: approximately gold IDs
            chunk_count_is_minimum=True,
            check_retrieval_relations=True,
            check_generation_gt=True,
            generation_gt_required_for_all=True,
            primary_key_type="string",
            db_name="visrag_gold_ids_test",
        )

        with create_test_database(config) as db:
            service = MultiModalIngestionService(db.session_factory, schema=db.schema)

            ingestor = VisRAGIngestor(VisRAGDatasetName.ARXIV_QA)
            ingestor.set_service(service)
            ingestor.ingest(
                query_limit=config.expected_query_count,
                min_corpus_cnt=config.expected_query_count,  # Forces minimal corpus
            )

            # Verify all queries have valid retrieval relations
            with service._create_uow() as uow:
                queries = uow.queries.get_all(limit=100)
                relations = uow.retrieval_relations.get_all(limit=1000)

                query_ids_with_relations = {r.query_id for r in relations}
                for query in queries:
                    assert query.id in query_ids_with_relations, f"Query {query.id} has no retrieval relations"
