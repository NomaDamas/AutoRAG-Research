import pytest

from autorag_research.orm.repository.text_uow import TextOnlyUnitOfWork
from autorag_research.orm.schema import Chunk, Query, RetrievalRelation
from autorag_research.orm.service.base import BaseService


class ConcreteTestService(BaseService):
    def _create_uow(self) -> TextOnlyUnitOfWork:
        return TextOnlyUnitOfWork(self.session_factory, self._schema)

    def _get_schema_classes(self) -> dict[str, type]:
        return {
            "Query": Query,
            "Chunk": Chunk,
            "RetrievalRelation": RetrievalRelation,
        }


class TestBaseService:
    def test_add_returns_ids(self, session_factory):
        service = ConcreteTestService(session_factory)
        chunks = [
            {"contents": "test chunk 1", "parent_caption": None},
            {"contents": "test chunk 2", "parent_caption": None},
        ]
        ids = service._add(chunks, table_name="Chunk", repository_property="chunks")
        assert len(ids) == 2
        assert all(isinstance(id_, int) for id_ in ids)

        with service._create_uow() as uow:
            for id_ in ids:
                uow.chunks.delete_by_id(id_)
            uow.commit()

    def test_add_invalid_table_raises_error(self, session_factory):
        service = ConcreteTestService(session_factory)
        with pytest.raises(ValueError, match="Table 'InvalidTable' not found"):
            service._add([{"data": "test"}], table_name="InvalidTable", repository_property="chunks")
