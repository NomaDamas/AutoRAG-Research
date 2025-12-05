from dataclasses import dataclass

from autorag_research.orm.service.retrieval_evaluation import build_retrieval_gt_from_relations


@dataclass
class MockRetrievalRelation:
    """Mock RetrievalRelation for testing."""

    group_index: int
    group_order: int
    chunk_id: int | None


class TestBuildRetrievalGtFromRelations:
    def test_empty_relations(self):
        """Empty input returns empty list."""
        result = build_retrieval_gt_from_relations([])
        assert result == []

    def test_single_group_single_item(self):
        """Single group with single item."""
        relations = [MockRetrievalRelation(group_index=0, group_order=0, chunk_id=1)]
        result = build_retrieval_gt_from_relations(relations)
        assert result == [["1"]]

    def test_single_group_multiple_items_or_condition(self):
        """Multiple items in same group = OR condition."""
        relations = [
            MockRetrievalRelation(group_index=0, group_order=0, chunk_id=1),
            MockRetrievalRelation(group_index=0, group_order=1, chunk_id=2),
            MockRetrievalRelation(group_index=0, group_order=2, chunk_id=3),
        ]
        result = build_retrieval_gt_from_relations(relations)
        # All in same inner list = OR condition
        assert result == [["1", "2", "3"]]

    def test_multiple_groups_single_items_and_condition(self):
        """Multiple groups with single items = AND condition."""
        relations = [
            MockRetrievalRelation(group_index=0, group_order=0, chunk_id=1),
            MockRetrievalRelation(group_index=1, group_order=0, chunk_id=2),
            MockRetrievalRelation(group_index=2, group_order=0, chunk_id=3),
        ]
        result = build_retrieval_gt_from_relations(relations)
        # Each in different inner list = AND condition
        assert result == [["1"], ["2"], ["3"]]

    def test_mixed_and_or_conditions(self):
        """Mixed AND/OR: (1 OR 2) AND 3."""
        relations = [
            MockRetrievalRelation(group_index=0, group_order=0, chunk_id=1),
            MockRetrievalRelation(group_index=0, group_order=1, chunk_id=2),
            MockRetrievalRelation(group_index=1, group_order=0, chunk_id=3),
        ]
        result = build_retrieval_gt_from_relations(relations)
        assert result == [["1", "2"], ["3"]]

    def test_group_order_sorting(self):
        """Items within group are sorted by group_order."""
        relations = [
            MockRetrievalRelation(group_index=0, group_order=2, chunk_id=3),
            MockRetrievalRelation(group_index=0, group_order=0, chunk_id=1),
            MockRetrievalRelation(group_index=0, group_order=1, chunk_id=2),
        ]
        result = build_retrieval_gt_from_relations(relations)
        # Should be sorted by group_order: 0, 1, 2 -> chunk_ids: 1, 2, 3
        assert result == [["1", "2", "3"]]

    def test_group_index_sorting(self):
        """Groups are sorted by group_index."""
        relations = [
            MockRetrievalRelation(group_index=2, group_order=0, chunk_id=30),
            MockRetrievalRelation(group_index=0, group_order=0, chunk_id=10),
            MockRetrievalRelation(group_index=1, group_order=0, chunk_id=20),
        ]
        result = build_retrieval_gt_from_relations(relations)
        assert result == [["10"], ["20"], ["30"]]

    def test_none_chunk_id_ignored(self):
        """Relations with None chunk_id are ignored."""
        relations = [
            MockRetrievalRelation(group_index=0, group_order=0, chunk_id=1),
            MockRetrievalRelation(group_index=0, group_order=1, chunk_id=None),
            MockRetrievalRelation(group_index=0, group_order=2, chunk_id=2),
        ]
        result = build_retrieval_gt_from_relations(relations)
        assert result == [["1", "2"]]

    def test_all_none_chunk_ids(self):
        """All None chunk_ids returns empty list."""
        relations = [
            MockRetrievalRelation(group_index=0, group_order=0, chunk_id=None),
            MockRetrievalRelation(group_index=1, group_order=0, chunk_id=None),
        ]
        result = build_retrieval_gt_from_relations(relations)
        assert result == []

    def test_complex_scenario(self):
        """Complex: (1 OR 2 OR 3) AND (4 OR 5) AND 6."""
        relations = [
            # Group 0: OR condition
            MockRetrievalRelation(group_index=0, group_order=0, chunk_id=1),
            MockRetrievalRelation(group_index=0, group_order=1, chunk_id=2),
            MockRetrievalRelation(group_index=0, group_order=2, chunk_id=3),
            # Group 1: OR condition
            MockRetrievalRelation(group_index=1, group_order=0, chunk_id=4),
            MockRetrievalRelation(group_index=1, group_order=1, chunk_id=5),
            # Group 2: single item
            MockRetrievalRelation(group_index=2, group_order=0, chunk_id=6),
        ]
        result = build_retrieval_gt_from_relations(relations)
        assert result == [["1", "2", "3"], ["4", "5"], ["6"]]
