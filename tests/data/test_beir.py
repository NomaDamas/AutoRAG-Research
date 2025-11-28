import os

import pytest
from llama_index.core import MockEmbedding
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from autorag_research.data.beir import BEIRIngestor
from autorag_research.orm.schema import Base
from autorag_research.orm.service.text_ingestion import TextDataIngestionService
from autorag_research.orm.util import create_database, drop_database, install_vector_extensions


@pytest.fixture(scope="session")
def beir_db_engine():
    host = os.getenv("POSTGRES_HOST")
    user = os.getenv("POSTGRES_USER")
    pwd = os.getenv("POSTGRES_PASSWORD")
    port = int(os.getenv("POSTGRES_PORT"))
    db_name = "autorag_research_beir_test"

    create_database(host, user, pwd, db_name, port=port)
    install_vector_extensions(host, user, pwd, db_name, port=port)
    url = f"postgresql+psycopg://{user}:{pwd}@{host}:{port}/autorag_research_beir_test"

    engine = create_engine(
        url,
        pool_pre_ping=True,  # Check connection health
        pool_size=10,
        max_overflow=20,
    )
    Base.metadata.create_all(engine)  # make all schema
    yield engine
    engine.dispose()

    drop_database(host, user, pwd, db_name, port=port)


@pytest.fixture(scope="session")
def session_factory_beir(beir_db_engine):
    return sessionmaker(bind=beir_db_engine)


@pytest.fixture
def service(session_factory_beir):
    service = TextDataIngestionService(session_factory_beir)
    yield service


@pytest.fixture
def beir_ingestor(db_session, service):
    ingestor = BEIRIngestor(service, MockEmbedding(768), "scifact")
    yield ingestor


@pytest.mark.data
def test_beir_ingest_embed_all(beir_ingestor):
    beir_ingestor.ingest(subset="test")
    stats = beir_ingestor.service.get_statistics()
    assert stats["queries"]["total"] == 300
    assert stats["chunks"]["total"] == 5183
    assert stats["chunks"]["with_embeddings"] == 0

    beir_ingestor.embed_all(max_concurrency=16)
    stats = beir_ingestor.service.get_statistics()
    assert stats["queries"]["total"] == 300
    assert stats["chunks"]["total"] == 5183
    assert stats["chunks"]["with_embeddings"] == 5183
    assert stats["chunks"]["without_embeddings"] == 0
