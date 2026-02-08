"""Tests for the unified retrieval ground truth API."""

import pytest

from autorag_research.exceptions import EmptyIterableError
from autorag_research.orm.models.retrieval_gt import (
    AndChain,
    ImageId,
    OrGroup,
    TextId,
    and_all,
    and_all_mixed,
    gt_to_relations,
    image,
    normalize_gt,
    or_all,
    or_all_mixed,
    text,
)


class TestOrOperator:
    """Test the | (OR) operator."""

    def test_text_id_or_text_id(self):
        result = TextId(1) | TextId(2)
        assert isinstance(result, OrGroup)
        assert len(result.items) == 2
        assert result.items[0] == TextId(1)
        assert result.items[1] == TextId(2)

    def test_image_id_or_image_id(self):
        result = ImageId(1) | ImageId(2)
        assert isinstance(result, OrGroup)
        assert len(result.items) == 2
        assert result.items[0] == ImageId(1)
        assert result.items[1] == ImageId(2)

    def test_text_id_or_image_id(self):
        result = TextId(1) | ImageId(2)
        assert isinstance(result, OrGroup)
        assert len(result.items) == 2
        assert result.items[0] == TextId(1)
        assert result.items[1] == ImageId(2)

    def test_chained_or(self):
        result = TextId(1) | TextId(2) | TextId(3)
        assert isinstance(result, OrGroup)
        assert len(result.items) == 3
        assert result.items == (TextId(1), TextId(2), TextId(3))

    def test_or_group_or_chunk(self):
        group = TextId(1) | TextId(2)
        result = group | TextId(3)
        assert isinstance(result, OrGroup)
        assert len(result.items) == 3


class TestAndOperator:
    """Test the & (AND) operator."""

    def test_text_id_and_text_id(self):
        result = TextId(1) & TextId(2)
        assert isinstance(result, AndChain)
        assert len(result.groups) == 2
        assert result.groups[0].items == (TextId(1),)
        assert result.groups[1].items == (TextId(2),)

    def test_or_group_and_chunk(self):
        result = (TextId(1) | TextId(2)) & TextId(3)
        assert isinstance(result, AndChain)
        assert len(result.groups) == 2
        assert result.groups[0].items == (TextId(1), TextId(2))
        assert result.groups[1].items == (TextId(3),)

    def test_or_group_and_or_group(self):
        result = (TextId(1) | TextId(2)) & (TextId(3) | TextId(4))
        assert isinstance(result, AndChain)
        assert len(result.groups) == 2
        assert result.groups[0].items == (TextId(1), TextId(2))
        assert result.groups[1].items == (TextId(3), TextId(4))

    def test_chained_and(self):
        result = TextId(1) & TextId(2) & TextId(3)
        assert isinstance(result, AndChain)
        assert len(result.groups) == 3


class TestIntWrapper:
    """Test the text() and image() wrapper functions."""

    def test_text_wrapper(self):
        t = text(10)
        assert t.value == 10
        assert t.chunk_type == "text"

    def test_image_wrapper(self):
        i = image(20)
        assert i.value == 20
        assert i.chunk_type == "image"

    def test_text_or_text(self):
        result = text(1) | text(2)
        assert isinstance(result, OrGroup)
        assert len(result.items) == 2
        assert result.items[0] == TextId(1)
        assert result.items[1] == TextId(2)

    def test_image_or_image(self):
        result = image(1) | image(2)
        assert isinstance(result, OrGroup)
        assert len(result.items) == 2
        assert result.items[0] == ImageId(1)
        assert result.items[1] == ImageId(2)

    def test_text_and_text(self):
        result = text(1) & text(2)
        assert isinstance(result, AndChain)
        assert len(result.groups) == 2

    def test_chained_text_or(self):
        result = text(1) | text(2) | text(3)
        assert isinstance(result, OrGroup)
        assert len(result.items) == 3

    def test_complex_expression(self):
        result = (text(1) | text(2)) & (text(3) | text(4))
        assert isinstance(result, AndChain)
        assert len(result.groups) == 2


class TestOrAll:
    """Test the or_all() helper function."""

    def test_or_all_single(self):
        result = or_all([1])
        # Single element returns _IntWrapper, not OrGroup
        assert result._to_chunk_id() == TextId(1)

    def test_or_all_multiple(self):
        result = or_all([1, 2, 3])
        assert isinstance(result, OrGroup)
        assert len(result.items) == 3
        assert result.items == (TextId(1), TextId(2), TextId(3))

    def test_or_all_with_image(self):
        result = or_all([1, 2], image)
        assert isinstance(result, OrGroup)
        assert result.items == (ImageId(1), ImageId(2))

    def test_or_all_empty_raises(self):
        with pytest.raises(ValueError, match="ids cannot be empty"):
            or_all([])


class TestAndAll:
    """Test the and_all() helper function."""

    def test_and_all_single(self):
        result = and_all([1])
        # Single element returns _IntWrapper
        assert result._to_chunk_id() == TextId(1)

    def test_and_all_multiple(self):
        result = and_all([1, 2, 3])
        assert isinstance(result, AndChain)
        assert len(result.groups) == 3

    def test_and_all_with_image(self):
        result = and_all([1, 2], image)
        assert isinstance(result, AndChain)
        assert len(result.groups) == 2

    def test_and_all_empty_raises(self):
        with pytest.raises(ValueError, match="ids cannot be empty"):
            and_all([])


class TestOrAllMixed:
    """Test the or_all_mixed() helper function."""

    def test_or_all_mixed_single(self):
        result = or_all_mixed([TextId(1)])
        assert result == TextId(1)

    def test_or_all_mixed_multiple(self):
        result = or_all_mixed([TextId(1), ImageId(2), TextId(3)])
        assert isinstance(result, OrGroup)
        assert len(result.items) == 3

    def test_or_all_mixed_empty_raises(self):
        with pytest.raises(EmptyIterableError):
            or_all_mixed([])


class TestAndAllMixed:
    """Test the and_all_mixed() helper function."""

    def test_and_all_mixed_single(self):
        result = and_all_mixed([TextId(1)])
        assert result == TextId(1)

    def test_and_all_mixed_multiple(self):
        result = and_all_mixed([TextId(1), ImageId(2), TextId(3)])
        assert isinstance(result, AndChain)
        assert len(result.groups) == 3

    def test_and_all_mixed_empty_raises(self):
        with pytest.raises(EmptyIterableError):
            and_all_mixed([])


class TestNormalizeGT:
    """Test the normalize_gt() function."""

    def test_normalize_int_text_mode(self):
        result = normalize_gt(42, chunk_type="text")
        assert isinstance(result, AndChain)
        assert len(result.groups) == 1
        assert result.groups[0].items == (TextId(42),)

    def test_normalize_int_image_mode(self):
        result = normalize_gt(42, chunk_type="image")
        assert isinstance(result, AndChain)
        assert len(result.groups) == 1
        assert result.groups[0].items == (ImageId(42),)

    def test_normalize_int_mixed_mode_raises(self):
        with pytest.raises(ValueError, match="Mixed mode requires"):
            normalize_gt(42, chunk_type="mixed")

    def test_normalize_text_id(self):
        result = normalize_gt(TextId(10), chunk_type="mixed")
        assert isinstance(result, AndChain)
        assert len(result.groups) == 1
        assert result.groups[0].items == (TextId(10),)

    def test_normalize_or_group(self):
        og = TextId(1) | TextId(2)
        result = normalize_gt(og, chunk_type="mixed")
        assert isinstance(result, AndChain)
        assert len(result.groups) == 1
        assert result.groups[0] == og

    def test_normalize_and_chain(self):
        ac = TextId(1) & TextId(2)
        result = normalize_gt(ac, chunk_type="mixed")
        assert result == ac

    def test_normalize_int_wrapper(self):
        result = normalize_gt(text(10), chunk_type="text")
        assert isinstance(result, AndChain)
        assert len(result.groups) == 1
        assert result.groups[0].items == (TextId(10),)


class TestGTToRelations:
    """Test the gt_to_relations() function."""

    def test_single_chunk(self):
        gt = AndChain(groups=(OrGroup(items=(TextId(10),)),))
        relations = gt_to_relations(query_id=1, gt=gt)

        assert len(relations) == 1
        assert relations[0] == {
            "query_id": 1,
            "chunk_id": 10,
            "image_chunk_id": None,
            "group_index": 0,
            "group_order": 0,
            "score": None,
        }

    def test_single_image_chunk(self):
        gt = AndChain(groups=(OrGroup(items=(ImageId(20),)),))
        relations = gt_to_relations(query_id=1, gt=gt)

        assert len(relations) == 1
        assert relations[0] == {
            "query_id": 1,
            "chunk_id": None,
            "image_chunk_id": 20,
            "group_index": 0,
            "group_order": 0,
            "score": None,
        }

    def test_or_group(self):
        gt = AndChain(groups=(OrGroup(items=(TextId(1), TextId(2), TextId(3))),))
        relations = gt_to_relations(query_id=1, gt=gt)

        assert len(relations) == 3
        assert relations[0]["group_index"] == 0
        assert relations[0]["group_order"] == 0
        assert relations[0]["chunk_id"] == 1
        assert relations[1]["group_order"] == 1
        assert relations[1]["chunk_id"] == 2
        assert relations[2]["group_order"] == 2
        assert relations[2]["chunk_id"] == 3

    def test_and_chain(self):
        gt = AndChain(
            groups=(
                OrGroup(items=(TextId(1),)),
                OrGroup(items=(TextId(2),)),
                OrGroup(items=(TextId(3),)),
            )
        )
        relations = gt_to_relations(query_id=1, gt=gt)

        assert len(relations) == 3
        assert relations[0]["group_index"] == 0
        assert relations[1]["group_index"] == 1
        assert relations[2]["group_index"] == 2

    def test_complex_and_or(self):
        # (1 OR 2) AND (3 OR 4)
        gt = AndChain(
            groups=(
                OrGroup(items=(TextId(1), TextId(2))),
                OrGroup(items=(TextId(3), TextId(4))),
            )
        )
        relations = gt_to_relations(query_id=1, gt=gt)

        assert len(relations) == 4
        # Group 0: chunks 1, 2
        assert relations[0] == {
            "query_id": 1,
            "chunk_id": 1,
            "image_chunk_id": None,
            "group_index": 0,
            "group_order": 0,
            "score": None,
        }
        assert relations[1] == {
            "query_id": 1,
            "chunk_id": 2,
            "image_chunk_id": None,
            "group_index": 0,
            "group_order": 1,
            "score": None,
        }
        # Group 1: chunks 3, 4
        assert relations[2] == {
            "query_id": 1,
            "chunk_id": 3,
            "image_chunk_id": None,
            "group_index": 1,
            "group_order": 0,
            "score": None,
        }
        assert relations[3] == {
            "query_id": 1,
            "chunk_id": 4,
            "image_chunk_id": None,
            "group_index": 1,
            "group_order": 1,
            "score": None,
        }

    def test_mixed_modality(self):
        gt = AndChain(groups=(OrGroup(items=(TextId(1), ImageId(2))),))
        relations = gt_to_relations(query_id=1, gt=gt)

        assert len(relations) == 2
        assert relations[0]["chunk_id"] == 1
        assert relations[0]["image_chunk_id"] is None
        assert relations[1]["chunk_id"] is None
        assert relations[1]["image_chunk_id"] == 2

    def test_with_scores(self):
        """Test that scores are correctly passed through to relations."""
        gt = AndChain(
            groups=(
                OrGroup(items=(TextId(1, score=2), TextId(2, score=1))),
                OrGroup(items=(ImageId(3, score=2),)),
            )
        )
        relations = gt_to_relations(query_id=1, gt=gt)

        assert len(relations) == 3
        assert relations[0]["chunk_id"] == 1
        assert relations[0]["score"] == 2
        assert relations[1]["chunk_id"] == 2
        assert relations[1]["score"] == 1
        assert relations[2]["image_chunk_id"] == 3
        assert relations[2]["score"] == 2

    def test_with_scores_from_wrapper(self):
        """Test that scores work with text() and image() wrappers."""
        expr = text(1, score=2) | text(2, score=1)
        gt = normalize_gt(expr, chunk_type="text")
        relations = gt_to_relations(1, gt)

        assert len(relations) == 2
        assert relations[0]["chunk_id"] == 1
        assert relations[0]["score"] == 2
        assert relations[1]["chunk_id"] == 2
        assert relations[1]["score"] == 1


class TestEndToEndExpressions:
    """Test complete expressions from user-facing syntax to relations."""

    def test_single_text_chunk(self):
        # service.add_retrieval_gt(query_id=1, gt=10, chunk_type="text")
        gt = normalize_gt(10, chunk_type="text")
        relations = gt_to_relations(1, gt)

        assert len(relations) == 1
        assert relations[0]["chunk_id"] == 10

    def test_or_alternatives(self):
        # service.add_retrieval_gt(query_id=1, gt=text(1) | text(2) | text(3), chunk_type="text")
        expr = text(1) | text(2) | text(3)
        gt = normalize_gt(expr, chunk_type="text")
        relations = gt_to_relations(1, gt)

        assert len(relations) == 3
        assert all(r["group_index"] == 0 for r in relations)

    def test_multi_hop_chain(self):
        # service.add_retrieval_gt(query_id=1, gt=text(1) & text(2) & text(3), chunk_type="text")
        expr = text(1) & text(2) & text(3)
        gt = normalize_gt(expr, chunk_type="text")
        relations = gt_to_relations(1, gt)

        assert len(relations) == 3
        assert relations[0]["group_index"] == 0
        assert relations[1]["group_index"] == 1
        assert relations[2]["group_index"] == 2

    def test_and_or_combination(self):
        # service.add_retrieval_gt(query_id=1, gt=(text(1) | text(2)) & (text(3) | text(4)), chunk_type="text")
        expr = (text(1) | text(2)) & (text(3) | text(4))
        gt = normalize_gt(expr, chunk_type="text")
        relations = gt_to_relations(1, gt)

        assert len(relations) == 4
        group0 = [r for r in relations if r["group_index"] == 0]
        group1 = [r for r in relations if r["group_index"] == 1]
        assert len(group0) == 2
        assert len(group1) == 2

    def test_or_all_helper(self):
        # For loops with varying lengths
        expr = or_all([1, 2, 3])
        gt = normalize_gt(expr, chunk_type="text")
        relations = gt_to_relations(1, gt)

        assert len(relations) == 3
        assert all(r["group_index"] == 0 for r in relations)

    def test_and_all_helper(self):
        # Multi-hop from list
        expr = and_all([1, 2, 3])
        gt = normalize_gt(expr, chunk_type="text")
        relations = gt_to_relations(1, gt)

        assert len(relations) == 3
        assert relations[0]["group_index"] == 0
        assert relations[1]["group_index"] == 1
        assert relations[2]["group_index"] == 2

    def test_mixed_modality_expression(self):
        # service.add_retrieval_gt(query_id=1, gt=TextId(1) | ImageId(2))
        expr = TextId(1) | ImageId(2)
        gt = normalize_gt(expr, chunk_type="mixed")
        relations = gt_to_relations(1, gt)

        assert len(relations) == 2
        assert relations[0]["chunk_id"] == 1
        assert relations[0]["image_chunk_id"] is None
        assert relations[1]["chunk_id"] is None
        assert relations[1]["image_chunk_id"] == 2
