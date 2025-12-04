-- Prefer VectorChord's extension; load alternatives only if needed
DO $$
BEGIN
	BEGIN
		CREATE EXTENSION IF NOT EXISTS vchord CASCADE;
	EXCEPTION WHEN others THEN
		PERFORM 1;
	END;
	BEGIN
		CREATE EXTENSION IF NOT EXISTS vectors;
	EXCEPTION WHEN others THEN
		PERFORM 1;
	END;
	BEGIN
		CREATE EXTENSION IF NOT EXISTS vector;
	EXCEPTION WHEN others THEN
		PERFORM 1;
	END;
END $$;

-- Schema DDL matching the provided design

-- File
CREATE TABLE IF NOT EXISTS file (
	id BIGSERIAL PRIMARY KEY,
	type VARCHAR(255) NOT NULL,
	path VARCHAR(255) NOT NULL
);

-- Document
CREATE TABLE IF NOT EXISTS document (
	id BIGSERIAL PRIMARY KEY,
	path BIGINT REFERENCES file(id),
	filename TEXT,
	author TEXT,
	title TEXT,
	doc_metadata JSONB
);

-- Page
CREATE TABLE IF NOT EXISTS page (
	id BIGSERIAL PRIMARY KEY,
	page_num INT NOT NULL,
	document_id BIGINT NOT NULL REFERENCES document(id),
	image_contents BYTEA,
	mimetype VARCHAR(255),
	page_metadata JSONB,
	CONSTRAINT uq_page_per_doc UNIQUE (document_id, page_num)
);

-- Caption
CREATE TABLE IF NOT EXISTS caption (
	id BIGSERIAL PRIMARY KEY,
	page_id BIGINT NOT NULL REFERENCES page(id),
	contents TEXT NOT NULL
);

-- Chunk
-- embeddings column supports VectorChord's MaxSim operator (@#) for late interaction models
CREATE TABLE IF NOT EXISTS chunk (
	id BIGSERIAL PRIMARY KEY,
	parent_caption BIGINT REFERENCES caption(id),
	contents TEXT NOT NULL,
	embedding VECTOR(768),
	embeddings VECTOR(768)[]  -- Multi-vector for ColBERT/ColPali style retrieval
);

-- ImageChunk
-- embeddings column supports VectorChord's MaxSim operator (@#) for late interaction models
CREATE TABLE IF NOT EXISTS image_chunk (
	id BIGSERIAL PRIMARY KEY,
	parent_page BIGINT REFERENCES page(id),
	contents BYTEA NOT NULL,
	mimetype VARCHAR(255) NOT NULL,
	embedding VECTOR(768),
	embeddings VECTOR(768)[]  -- Multi-vector for ColPali style image retrieval
);

-- CaptionChunkRelation
CREATE TABLE IF NOT EXISTS caption_chunk_relation (
	caption_id BIGINT NOT NULL REFERENCES caption(id),
	chunk_id BIGINT NOT NULL REFERENCES chunk(id),
	PRIMARY KEY (caption_id, chunk_id)
);

-- Query
-- embeddings column supports VectorChord's MaxSim operator (@#) for late interaction models
CREATE TABLE IF NOT EXISTS query (
	id BIGSERIAL PRIMARY KEY,
	contents TEXT NOT NULL,
	generation_gt TEXT[],
	embedding VECTOR(768),
	embeddings VECTOR(768)[]  -- Multi-vector for ColBERT/ColPali style retrieval
);

-- RetrievalRelation
CREATE TABLE IF NOT EXISTS retrieval_relation (
	query_id BIGINT NOT NULL REFERENCES query(id),
	group_index INT NOT NULL,
	group_order INT NOT NULL,
	chunk_id BIGINT REFERENCES chunk(id),
	image_chunk_id BIGINT REFERENCES image_chunk(id),
	PRIMARY KEY (query_id, group_index, group_order),
	CONSTRAINT ck_rr_one_only CHECK ((chunk_id IS NULL) <> (image_chunk_id IS NULL))
);

-- Pipeline
CREATE TABLE IF NOT EXISTS pipeline (
	id BIGSERIAL PRIMARY KEY,
	name VARCHAR(255) NOT NULL,
	config JSONB NOT NULL
);

-- Metric
CREATE TABLE IF NOT EXISTS metric (
	id BIGSERIAL PRIMARY KEY,
	name VARCHAR(255) NOT NULL,
	type VARCHAR(255) NOT NULL
);

-- ExperimentResult
CREATE TABLE IF NOT EXISTS executor_result (
	query_id BIGINT NOT NULL REFERENCES query(id),
	pipeline_id BIGINT NOT NULL REFERENCES pipeline(id),
	generation_result TEXT,
	token_usage INT,
	execution_time INT,
	result_metadata JSONB,
	PRIMARY KEY (query_id, pipeline_id)
);

CREATE TABLE IF NOT EXISTS evaluation_result (
    query_id BIGINT NOT NULL REFERENCES query(id),
    pipeline_id BIGINT NOT NULL REFERENCES pipeline(id),
    metric_id BIGINT NOT NULL REFERENCES metric(id),
    metric_result FLOAT NOT NULL,
    PRIMARY KEY (query_id, pipeline_id, metric_id)
);

-- ImageChunkRetrievedResult
CREATE TABLE IF NOT EXISTS image_chunk_retrieved_result (
	query_id BIGINT NOT NULL REFERENCES query(id),
	pipeline_id BIGINT NOT NULL REFERENCES pipeline(id),
	metric_id BIGINT NOT NULL REFERENCES metric(id),
	image_chunk_id BIGINT NOT NULL REFERENCES image_chunk(id),
    rel_score FLOAT,
	PRIMARY KEY (query_id, pipeline_id, metric_id, image_chunk_id)
);

-- ChunkRetrievedResult
CREATE TABLE IF NOT EXISTS chunk_retrieved_result (
	query_id BIGINT NOT NULL REFERENCES query(id),
	pipeline_id BIGINT NOT NULL REFERENCES pipeline(id),
	chunk_id BIGINT NOT NULL REFERENCES chunk(id),
    rel_score FLOAT,
	PRIMARY KEY (query_id, pipeline_id, chunk_id)
);

-- Summary
CREATE TABLE IF NOT EXISTS summary (
	pipeline_id BIGINT NOT NULL REFERENCES pipeline(id),
	metric_id BIGINT NOT NULL REFERENCES metric(id),
	metric_result FLOAT NOT NULL,
	token_usage INT,
	execution_time INT,
	result_metadata JSONB,
	PRIMARY KEY (pipeline_id, metric_id)
);
