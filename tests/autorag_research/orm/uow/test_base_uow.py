import pytest

from autorag_research.exceptions import SessionNotSetError
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

    assert uow.session.is_active is False


def test_repository_access_without_session_raises_error(session_factory, uow_class):
    uow = uow_class(session_factory)
    available_repos = uow.available_repositories()

    with pytest.raises(SessionNotSetError):
        getattr(uow, available_repos[0])


def test_repository_lazy_initialization_and_caching(session_factory, uow_class):
    with uow_class(session_factory) as uow:
        available_repos = uow.available_repositories()
        repo_name = available_repos[0]
        private_attr = f"_{repo_name.rstrip('s')}_repo"

        assert getattr(uow, private_attr, "NOT_SET") in (None, "NOT_SET")

        first_access = getattr(uow, repo_name)
        second_access = getattr(uow, repo_name)

        assert first_access is second_access


def test_repository_reset_after_exit(session_factory, uow_class):
    uow = uow_class(session_factory)

    with uow:
        available_repos = uow.available_repositories()
        for repo_name in available_repos[:2]:
            _ = getattr(uow, repo_name)

    for attr_name in dir(uow):
        if attr_name.endswith("_repo") and not attr_name.startswith("__"):
            assert getattr(uow, attr_name) is None


def test_transaction_operations(session_factory, uow_class):
    with uow_class(session_factory) as uow:
        uow.flush()
        uow.commit()
        uow.rollback()


def test_available_repositories_returns_list(session_factory, uow_class):
    uow = uow_class(session_factory)
    repos = uow.available_repositories()

    assert isinstance(repos, list)
    assert len(repos) > 0
    assert all(isinstance(name, str) for name in repos)
    assert repos == sorted(repos)


def test_repr_shows_repositories(session_factory, uow_class):
    uow = uow_class(session_factory)
    repr_str = repr(uow)

    assert uow_class.__name__ in repr_str
    assert "repositories=" in repr_str
