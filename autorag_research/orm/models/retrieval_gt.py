"""Unified Retrieval Ground Truth API using `|` (or) and `&` (and) operators.

This module provides an intuitive, type-safe API for specifying retrieval ground truth
with explicit AND-OR semantics.

Examples:
    # Mixed modality (default, explicit TextId/ImageId wrappers required)
    service.add_retrieval_gt(query_id=1, gt=(TextId(1) | TextId(2) | ImageId(3)) & TextId(4))

    # Text-only (use chunk_type="text")
    service.add_retrieval_gt(query_id=1, gt=text(1) | text(2) | text(3), chunk_type="text")

    # Image-only (use chunk_type="image")
    service.add_retrieval_gt(query_id=1, gt=image(1) | image(2), chunk_type="image")

    # Single chunk (no wrapper needed with text/image mode)
    service.add_retrieval_gt(query_id=1, gt=42, chunk_type="text")

    # Multi-hop chain (just &)
    service.add_retrieval_gt(query_id=1, gt=text(1) & text(2) & text(3), chunk_type="text")

    # Dynamic list -> OR expression (for loops with varying lengths)
    service.add_retrieval_gt(query_id=1, gt=or_all([1, 2, 3]), chunk_type="text")
    service.add_retrieval_gt(query_id=1, gt=and_all([1, 2, 3]), chunk_type="text")

Semantics:
    - `|` (OR): Any chunk in the group is a valid answer (group_order)
    - `&` (AND): All groups must be satisfied (group_index) - for multi-hop or AND-OR logic
    - Outer structure = AND groups (group_index)
    - Inner structure = OR alternatives (group_order)
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce

from autorag_research.exceptions import EmptyIterableError

# =============================================================================
# Chunk ID Wrappers
# =============================================================================


@dataclass(frozen=True)
class TextId:
    """Text chunk ID wrapper for mixed modality operations."""

    id: int | str
    score: int | None = None  # Graded relevance: 0=not relevant, 1=somewhat, 2=highly

    def __or__(self, other: TextId | ImageId | OrGroup) -> OrGroup:
        """Create OR group: TextId(1) | TextId(2)."""
        return OrGroup._create(self) | other

    def __and__(self, other: TextId | ImageId | OrGroup | AndChain) -> AndChain:
        """Create AND chain: TextId(1) & TextId(2)."""
        return AndChain._create(OrGroup._create(self)) & other


@dataclass(frozen=True)
class ImageId:
    """Image chunk ID wrapper for mixed modality operations."""

    id: int | str
    score: int | None = None  # Graded relevance: 0=not relevant, 1=somewhat, 2=highly

    def __or__(self, other: TextId | ImageId | OrGroup) -> OrGroup:
        """Create OR group: ImageId(1) | ImageId(2)."""
        return OrGroup._create(self) | other

    def __and__(self, other: TextId | ImageId | OrGroup | AndChain) -> AndChain:
        """Create AND chain: ImageId(1) & ImageId(2)."""
        return AndChain._create(OrGroup._create(self)) & other


# Union type for any chunk reference
ChunkId = TextId | ImageId


# =============================================================================
# OR Group and AND Chain
# =============================================================================


@dataclass
class OrGroup:
    """OR group of chunk IDs (any is valid).

    Represents alternatives where ANY chunk in the group is a correct answer.
    Maps to group_order in the database.
    """

    items: tuple[ChunkId, ...]

    @classmethod
    def _create(cls, item: ChunkId) -> OrGroup:
        """Create OrGroup from a single ChunkId."""
        return cls(items=(item,))

    def __or__(self, other: ChunkId | OrGroup) -> OrGroup:
        """Extend OR group: (TextId(1) | TextId(2)) | TextId(3)."""
        if isinstance(other, OrGroup):
            return OrGroup(items=self.items + other.items)
        # Handle _IntWrapper (has _to_chunk_id method)
        if hasattr(other, "_to_chunk_id"):
            return OrGroup(items=(*self.items, other._to_chunk_id()))  # ty: ignore
        return OrGroup(items=(*self.items, other))

    def __and__(self, other: ChunkId | OrGroup | AndChain) -> AndChain:
        """Create AND chain from OR group: (TextId(1) | TextId(2)) & TextId(3)."""
        return AndChain._create(self) & other


@dataclass
class AndChain:
    """AND chain of OR groups (all groups must be satisfied).

    Represents multi-hop retrieval or AND-OR conditions where ALL groups
    must have at least one match. Maps to group_index in the database.
    """

    groups: tuple[OrGroup, ...]

    @classmethod
    def _create(cls, group: OrGroup) -> AndChain:
        """Create AndChain from a single OrGroup."""
        return cls(groups=(group,))

    def __and__(self, other: ChunkId | OrGroup | AndChain) -> AndChain:
        """Extend AND chain: (A & B) & C."""
        if isinstance(other, AndChain):
            return AndChain(groups=(*self.groups, other.groups))  # ty: ignore
        elif isinstance(other, OrGroup):
            return AndChain(groups=(*self.groups, other))
        # Handle _IntWrapper (has _to_chunk_id method)
        elif hasattr(other, "_to_chunk_id"):
            return AndChain(groups=(*self.groups, OrGroup._create(other._to_chunk_id())))  # ty: ignore
        else:  # ChunkId (TextId or ImageId)
            return AndChain(groups=(*self.groups, OrGroup._create(other)))


# =============================================================================
# Integer Wrapper for text()/image() functions
# =============================================================================


class _IntWrapper:
    """Wrapper to enable | and & operators on integers.

    It provides operator overloading for integers wrapped via text() or image() functions.

    The returned types (OrGroup, AndChain) are specific to the retrieval GT
    system and do not affect any other code.
    """

    def __init__(self, value: int | str, chunk_type: str, score: int | None = None):
        self.value = value
        self.chunk_type = chunk_type
        self.score = score

    def _to_chunk_id(self) -> ChunkId:
        """Convert to appropriate ChunkId type."""
        if self.chunk_type == "text":
            return TextId(self.value, score=self.score)
        return ImageId(self.value, score=self.score)

    def __or__(self, other: _IntWrapper | OrGroup) -> OrGroup:
        """Create OR group: text(1) | text(2)."""
        if isinstance(other, _IntWrapper):
            return OrGroup(items=(self._to_chunk_id(), other._to_chunk_id()))
        return OrGroup._create(self._to_chunk_id()) | other

    def __ror__(self, other: _IntWrapper) -> OrGroup:
        """Handle reverse OR for _IntWrapper."""
        return OrGroup(items=(other._to_chunk_id(), self._to_chunk_id()))

    def __and__(self, other: _IntWrapper | OrGroup | AndChain) -> AndChain:
        """Create AND chain: text(1) & text(2)."""
        chain = AndChain._create(OrGroup._create(self._to_chunk_id()))
        if isinstance(other, _IntWrapper):
            return chain & OrGroup._create(other._to_chunk_id())
        return chain & other

    def __rand__(self, other: _IntWrapper) -> AndChain:
        """Handle reverse AND for _IntWrapper."""
        return AndChain._create(OrGroup._create(other._to_chunk_id())) & OrGroup._create(self._to_chunk_id())


# Type alias for ground truth expression
RetrievalGT = int | ChunkId | OrGroup | AndChain | _IntWrapper

# =============================================================================
# Factory Functions
# =============================================================================


def text(_id: int | str, score: int | None = None) -> _IntWrapper:
    """Wrap int for text-only mode with | and & support.

    Args:
        _id: Text chunk ID
        score: Optional graded relevance score (0=not relevant, 1=somewhat, 2=highly)

    Returns:
        Wrapper that supports | (OR) and & (AND) operators

    Examples:
        text(1) | text(2) | text(3)  # OR: any of 1, 2, 3
        text(1) & text(2) & text(3)  # AND: multi-hop chain
        (text(1) | text(2)) & text(3)  # (1 OR 2) AND 3
        text(1, score=2) | text(2, score=1)  # with graded relevance
    """
    return _IntWrapper(_id, "text", score=score)


def image(_id: int | str, score: int | None = None) -> _IntWrapper:
    """Wrap int for image-only mode with | and & support.

    Args:
        _id: Image chunk ID
        score: Optional graded relevance score (0=not relevant, 1=somewhat, 2=highly)

    Returns:
        Wrapper that supports | (OR) and & (AND) operators

    Examples:
        image(1) | image(2)  # OR: either image 1 or 2
        image(1) & image(2)  # AND: multi-hop with images
        image(1, score=2) | image(2, score=1)  # with graded relevance
    """
    return _IntWrapper(_id, "image", score=score)


# =============================================================================
# Helper Functions for Building from Lists
# =============================================================================


def or_all(ids: list[int | str], wrapper_fn: Callable[[int | str], _IntWrapper] = text) -> OrGroup | _IntWrapper:
    """Build OR expression from list: [1, 2, 3] -> wrapper(1) | wrapper(2) | wrapper(3).

    Use this when you have a dynamic list of IDs (e.g., in a for loop with varying lengths).

    Args:
        ids: List of chunk IDs
        wrapper_fn: text (default) or image

    Returns:
        OrGroup expression, or single _IntWrapper if only one ID

    Raises:
        ValueError: If ids is empty

    Examples:
        or_all([1, 2, 3])           # text(1) | text(2) | text(3)
        or_all([1, 2], image)       # image(1) | image(2)
        or_all([1])                 # text(1) - single element
    """
    if not ids:
        raise ValueError("ids cannot be empty")  # noqa: TRY003
    if len(ids) == 1:
        return wrapper_fn(ids[0])
    return reduce(operator.or_, [wrapper_fn(_id) for _id in ids])


def and_all(ids: list[int | str], wrapper_fn: Callable[[int | str], _IntWrapper] = text) -> AndChain | _IntWrapper:
    """Build AND chain from list: [1, 2, 3] -> wrapper(1) & wrapper(2) & wrapper(3).

    Use this for multi-hop retrieval where each ID represents a sequential hop.

    Args:
        ids: List of chunk IDs (each becomes a hop in multi-hop)
        wrapper_fn: text (default) or image

    Returns:
        AndChain expression, or single _IntWrapper if only one ID

    Raises:
        ValueError: If ids is empty

    Examples:
        and_all([1, 2, 3])          # text(1) & text(2) & text(3) (3-hop chain)
        and_all([1, 2], image)      # image(1) & image(2)
        and_all([1])                # text(1) - single element
    """
    if not ids:
        raise ValueError("ids cannot be empty")  # noqa: TRY003
    if len(ids) == 1:
        return wrapper_fn(ids[0])
    return reduce(operator.and_, [wrapper_fn(_id) for _id in ids])


def or_all_mixed(items: list[TextId | ImageId]) -> OrGroup | ChunkId:
    """Build OR expression from mixed TextId/ImageId list.

    Args:
        items: List of TextId or ImageId objects

    Returns:
        OrGroup expression, or single ChunkId if only one item

    Raises:
        ValueError: If items is empty

    Examples:
        or_all_mixed([TextId(1), ImageId(2), TextId(3)])
    """
    if not items:
        raise EmptyIterableError("items")
    if len(items) == 1:
        return items[0]
    return reduce(operator.or_, items)


def and_all_mixed(items: list[TextId | ImageId]) -> AndChain | ChunkId:
    """Build AND chain from mixed TextId/ImageId list.

    Args:
        items: List of TextId or ImageId objects

    Returns:
        AndChain expression, or single ChunkId if only one item

    Raises:
        ValueError: If items is empty

    Examples:
        and_all_mixed([TextId(1), ImageId(2), TextId(3)])  # 3-hop mixed chain
    """
    if not items:
        raise EmptyIterableError("items")
    if len(items) == 1:
        return items[0]
    return reduce(operator.and_, items)


# =============================================================================
# Conversion Functions
# =============================================================================


def normalize_gt(gt: RetrievalGT, chunk_type: str = "mixed") -> AndChain:
    """Normalize any GT expression to AndChain.

        This function handles all valid GT input types and converts them to a
        normalized AndChain structure for database insertion
    .
        Args:
            gt: The ground truth expression (int, ChunkId, _IntWrapper, OrGroup, or AndChain)
            chunk_type: "mixed", "text", or "image" - for auto-wrapping ints

        Returns:
            Normalized AndChain structure

        Raises:
            ValueError: If chunk_type is "mixed" but gt is a plain int
            TypeError: If gt is an invalid type
    """
    # Handle _IntWrapper first (from text()/image() functions)
    if isinstance(gt, _IntWrapper):
        chunk_id = gt._to_chunk_id()
        return AndChain(groups=(OrGroup(items=(chunk_id,)),))

    # Auto-wrap int to ChunkId
    if type(gt) is int or type(gt) is str:
        if chunk_type == "text":
            gt = TextId(gt)
        elif chunk_type == "image":
            gt = ImageId(gt)
        else:
            raise ValueError(  # noqa: TRY003
                "Mixed mode requires explicit TextId/ImageId wrappers. "
                "Use chunk_type='text' or chunk_type='image' for plain ints."
            )

    # ChunkId -> OrGroup -> AndChain
    if isinstance(gt, (TextId, ImageId)):
        return AndChain(groups=(OrGroup(items=(gt,)),))
    elif isinstance(gt, OrGroup):
        return AndChain(groups=(gt,))
    elif isinstance(gt, AndChain):
        return gt
    else:
        raise TypeError


def gt_to_relations(query_id: int | str, gt: AndChain) -> list[dict]:
    """Convert AndChain to list of relation dicts for DB insertion.

    Args:
        query_id: The query ID this ground truth applies to
        gt: Normalized AndChain structure

    Returns:
        List of dicts with keys: query_id, chunk_id, image_chunk_id,
        group_index, group_order, score
    """
    relations = []
    for group_index, or_group in enumerate(gt.groups):
        for group_order, chunk_id in enumerate(or_group.items):
            relations.append({
                "query_id": query_id,
                "chunk_id": chunk_id.id if isinstance(chunk_id, TextId) else None,
                "image_chunk_id": chunk_id.id if isinstance(chunk_id, ImageId) else None,
                "group_index": group_index,
                "group_order": group_order,
                "score": chunk_id.score,
            })
    return relations
