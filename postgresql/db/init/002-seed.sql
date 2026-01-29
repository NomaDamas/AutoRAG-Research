-- Seed test data for all tables assuming schema already created in 001-schema.sql
-- NOTE: explicit IDs are used for FK wiring; sequences are advanced at the end.

BEGIN;

-- Files (raw docs and images)
INSERT INTO file (id, type, path) VALUES
	(1, 'raw', '/data/doc1.pdf'),
	(2, 'raw', '/data/doc2.pdf'),
	(3, 'raw', '/data/doc3.pdf'),
	(4, 'raw', '/data/doc4.pdf'),
	(5, 'raw', '/data/doc5.pdf'),
	(6, 'image', '/data/img1.png'),
	(7, 'image', '/data/img2.png'),
	(8, 'image', '/data/img3.png'),
	(9, 'image', '/data/img4.png'),
	(10, 'image', '/data/img5.png')
ON CONFLICT DO NOTHING;

-- Documents (each points to a file)
INSERT INTO document (id, path, filename, author, title, doc_metadata) VALUES
	(1, 1, 'doc1.pdf', 'alice', 'Doc One', '{"topic": "alpha"}'),
	(2, 2, 'doc2.pdf', 'bob', 'Doc Two', '{"topic": "beta"}'),
	(3, 3, 'doc3.pdf', 'carol', 'Doc Three', '{"topic": "gamma"}'),
	(4, 4, 'doc4.pdf', 'dave', 'Doc Four', '{"topic": "delta"}'),
	(5, 5, 'doc5.pdf', 'erin', 'Doc Five', '{"topic": "epsilon"}')
ON CONFLICT DO NOTHING;

-- Pages (two pages per document)
INSERT INTO page (id, page_num, document_id, image_contents, mimetype, page_metadata) VALUES
	(1, 1, 1, NULL, NULL, '{"dpi": 300}'),
	(2, 2, 1, NULL, NULL, '{"dpi": 300}'),
	(3, 1, 2, NULL, NULL, '{"dpi": 300}'),
	(4, 2, 2, NULL, NULL, '{"dpi": 300}'),
	(5, 1, 3, NULL, NULL, '{"dpi": 300}'),
	(6, 2, 3, NULL, NULL, '{"dpi": 300}'),
	(7, 1, 4, NULL, NULL, '{"dpi": 300}'),
	(8, 2, 4, NULL, NULL, '{"dpi": 300}'),
	(9, 1, 5, NULL, NULL, '{"dpi": 300}'),
	(10, 2, 5, NULL, NULL, '{"dpi": 300}')
ON CONFLICT DO NOTHING;

-- Chunks (text chunks from pages)
-- embeddings column uses VectorChord-compatible array format: ARRAY['[...]'::vector, ...]
INSERT INTO chunk (id, contents, embedding, embeddings, is_table, table_type) VALUES
	(1, 'Chunk 1-1', NULL, NULL, FALSE, NULL),
	(2, 'Chunk 1-2', NULL, NULL, FALSE, NULL),
	(3, 'Chunk 2-1', NULL, NULL, FALSE, NULL),
	(4, 'Chunk 3-1', NULL, NULL, FALSE, NULL),
	(5, 'Chunk 4-1', NULL, NULL, FALSE, NULL),
	(6, 'Chunk 5-1', NULL, NULL, FALSE, NULL),
	-- Table sample data
	(7, '| Col1 | Col2 |\n|------|------|\n| A | B |', NULL, NULL, TRUE, 'markdown'),
	(8, '<table><tr><td>Data</td></tr></table>', NULL, NULL, TRUE, 'html')
ON CONFLICT DO NOTHING;

-- ImageChunks (image regions from pages)
-- embeddings column uses VectorChord-compatible array format for multi-vector retrieval
INSERT INTO image_chunk (id, parent_page, contents, mimetype, embedding, embeddings) VALUES
	(1, 1, E'\\x00'::bytea, 'image/png', NULL, NULL),
	(2, 2, E'\\x00'::bytea, 'image/png', NULL, NULL),
	(3, 3, E'\\x00'::bytea, 'image/png', NULL, NULL),
	(4, 4, E'\\x00'::bytea, 'image/png', NULL, NULL),
	(5, 5, E'\\x00'::bytea, 'image/png', NULL, NULL)
ON CONFLICT DO NOTHING;

-- Page-Chunk relations
INSERT INTO page_chunk_relation (page_id, chunk_id) VALUES
	(1, 1), (1, 2), (3, 3), (5, 4), (7, 5), (9, 6)
ON CONFLICT DO NOTHING;

-- Queries
-- embeddings column uses VectorChord-compatible array format for multi-vector retrieval
INSERT INTO query (id, contents, query_to_llm, generation_gt, embedding, embeddings) VALUES
	(1, 'What is Doc One about?', NULL, ARRAY['alpha'], NULL, NULL),
	(2, 'Find details in Doc Two', NULL, ARRAY['beta'], NULL, NULL),
	(3, 'Summarize Doc Three',  'Three Three', ARRAY['gamma'], NULL, NULL),
	(4, 'Topics in Doc Four', 'Doc Four Four',ARRAY['delta'], NULL, NULL),
	(5, 'Explain Doc Five', 'Doc Five Five', ARRAY['epsilon'], NULL, NULL)
ON CONFLICT DO NOTHING;

-- Retrieval relations (exactly one of chunk_id, image_chunk_id)
INSERT INTO retrieval_relation (query_id, group_index, group_order, chunk_id, image_chunk_id) VALUES
	(1, 0, 0, 1, NULL),
	(2, 0, 0, 3, NULL),
	(3, 0, 0, NULL, 1),
	(4, 0, 0, NULL, 2),
	(5, 0, 0, 6, NULL)
ON CONFLICT DO NOTHING;

-- Pipelines
INSERT INTO pipeline (id, name, config) VALUES
	(1, 'baseline', '{"k": 5}'),
	(2, 'rerank', '{"k": 10, "reranker": true}')
ON CONFLICT DO NOTHING;

-- Metrics
INSERT INTO metric (id, name, type) VALUES
	(1, 'retrieval@k', 'retrieval'),
	(2, 'bleu', 'generation')
ON CONFLICT DO NOTHING;

-- Executor results
INSERT INTO executor_result (query_id, pipeline_id, generation_result, token_usage, execution_time, result_metadata) VALUES
	(1, 1,  NULL, '{"prompt_tokens": 50, "completion_tokens": 50, "total_tokens": 100, "embedding_tokens": 0}', 1200, '{"notes": "ok"}'),
	(1, 2,  NULL, '{"prompt_tokens": 60, "completion_tokens": 60, "total_tokens": 120, "embedding_tokens": 0}', 1400, '{"notes": "better"}'),
	(2, 1,  'Generated text 1', '{"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200, "embedding_tokens": 0}', 2500, '{"len": 20}')
ON CONFLICT DO NOTHING;


-- Experiment results
INSERT INTO evaluation_result (query_id, pipeline_id, metric_id, metric_result) VALUES
	(1, 1, 1, 0.8),
	(1, 2, 1, 0.85),
	(2, 1, 2, 0.6)
ON CONFLICT DO NOTHING;

-- Retrieved results (image/text)
INSERT INTO image_chunk_retrieved_result (query_id, pipeline_id, image_chunk_id, rel_score) VALUES
	(3, 1, 1, 0.9),
	(4, 2, 2, 0.7)
ON CONFLICT DO NOTHING;

INSERT INTO chunk_retrieved_result (query_id, pipeline_id, chunk_id, rel_score) VALUES
	(1, 1, 1, 0.85),
	(2, 2, 3, 0.4)
ON CONFLICT DO NOTHING;

-- Summary (retrieval metrics)
INSERT INTO summary (pipeline_id, metric_id, metric_result, token_usage, execution_time, result_metadata) VALUES
	(1, 1, 0.82, '{"prompt_tokens": 110, "completion_tokens": 110, "total_tokens": 220, "embedding_tokens": 0}', 2600, '{"run": 1}'),
	(2, 1, 0.85, '{"prompt_tokens": 60, "completion_tokens": 60, "total_tokens": 120, "embedding_tokens": 0}', 1400, '{"run": 1}')
ON CONFLICT DO NOTHING;

-- Summary (generation metrics - bleu)
INSERT INTO summary (pipeline_id, metric_id, metric_result, token_usage, execution_time, result_metadata) VALUES
	(1, 2, 0.55, '{"prompt_tokens": 100, "completion_tokens": 150, "total_tokens": 250, "embedding_tokens": 0}', 3000, '{"run": 1}'),
	(2, 2, 0.62, '{"prompt_tokens": 120, "completion_tokens": 180, "total_tokens": 300, "embedding_tokens": 0}', 3200, '{"run": 1}')
ON CONFLICT DO NOTHING;

-- Advance sequences to max IDs to prevent conflicts on future inserts
SELECT setval(pg_get_serial_sequence('file','id'), (SELECT COALESCE(MAX(id), 1) FROM file), true);
SELECT setval(pg_get_serial_sequence('document','id'), (SELECT COALESCE(MAX(id), 1) FROM document), true);
SELECT setval(pg_get_serial_sequence('page','id'), (SELECT COALESCE(MAX(id), 1) FROM page), true);
SELECT setval(pg_get_serial_sequence('chunk','id'), (SELECT COALESCE(MAX(id), 1) FROM chunk), true);
SELECT setval(pg_get_serial_sequence('image_chunk','id'), (SELECT COALESCE(MAX(id), 1) FROM image_chunk), true);
SELECT setval(pg_get_serial_sequence('query','id'), (SELECT COALESCE(MAX(id), 1) FROM query), true);
SELECT setval(pg_get_serial_sequence('pipeline','id'), (SELECT COALESCE(MAX(id), 1) FROM pipeline), true);
SELECT setval(pg_get_serial_sequence('metric','id'), (SELECT COALESCE(MAX(id), 1) FROM metric), true);

COMMIT;
