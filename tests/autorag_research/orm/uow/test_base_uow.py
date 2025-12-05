import pytest

from autorag_research.orm.uow import (
    MultiModalUnitOfWork,
    RetrievalUnitOfWork,
    TextOnlyUnitOfWork,
)


@pytest.fixture(
    params=[
        TextOnlyUnitOfWork,
        MultiModalUnitOfWork,
        RetrievalUnitOfWork,
    ],
    ids=["TextOnly", "MultiModal", "Retrieval"],
)
def uow_class(request):
    return request.param


def test_context_manager_lifecycle(session_factory, uow_class):
    uow = uow_class(session_factory)
    assert uow.session is None

    with uow:
        assert uow.session is not None

    assert uow.session is None


def test_transaction_operations(session_factory, uow_class):
    with uow_class(session_factory) as uow:
        uow.flush()
        uow.commit()
        uow.rollback()
