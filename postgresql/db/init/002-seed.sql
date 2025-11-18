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
INSERT INTO document (id, filepath, filename, author, title, doc_metadata) VALUES
	(1, 1, 'doc1.pdf', 'alice', 'Doc One', '{"topic": "alpha"}'),
	(2, 2, 'doc2.pdf', 'bob', 'Doc Two', '{"topic": "beta"}'),
	(3, 3, 'doc3.pdf', 'carol', 'Doc Three', '{"topic": "gamma"}'),
	(4, 4, 'doc4.pdf', 'dave', 'Doc Four', '{"topic": "delta"}'),
	(5, 5, 'doc5.pdf', 'erin', 'Doc Five', '{"topic": "epsilon"}')
ON CONFLICT DO NOTHING;

-- Pages (two pages per document)
INSERT INTO page (id, page_num, document_id, image_path, page_metadata) VALUES
	(1, 1, 1, 6, '{"dpi": 300}'),
	(2, 2, 1, 7, '{"dpi": 300}'),
	(3, 1, 2, 8, '{"dpi": 300}'),
	(4, 2, 2, 9, '{"dpi": 300}'),
	(5, 1, 3, 10, '{"dpi": 300}'),
	(6, 2, 3, NULL, '{"dpi": 300}'),
	(7, 1, 4, NULL, '{"dpi": 300}'),
	(8, 2, 4, NULL, '{"dpi": 300}'),
	(9, 1, 5, NULL, '{"dpi": 300}'),
	(10, 2, 5, NULL, '{"dpi": 300}')
ON CONFLICT DO NOTHING;

-- Captions (one per page)
INSERT INTO caption (id, page_id, contents) VALUES
	(1, 1, 'Caption for page 1 of doc1'),
	(2, 2, 'Caption for page 2 of doc1'),
	(3, 3, 'Caption for page 1 of doc2'),
	(4, 4, 'Caption for page 2 of doc2'),
	(5, 5, 'Caption for page 1 of doc3'),
	(6, 6, 'Caption for page 2 of doc3'),
	(7, 7, 'Caption for page 1 of doc4'),
	(8, 8, 'Caption for page 2 of doc4'),
	(9, 9, 'Caption for page 1 of doc5'),
	(10, 10, 'Caption for page 2 of doc5')
ON CONFLICT DO NOTHING;

-- Chunks (text chunks from captions)
INSERT INTO chunk (id, parent_caption, contents, embedding, embeddings) VALUES
	(1, 1, 'Chunk 1-1', NULL, NULL),
	(2, 1, 'Chunk 1-2', NULL, NULL),
	(3, 3, 'Chunk 2-1', NULL, NULL),
	(4, 5, 'Chunk 3-1', NULL, NULL),
	(5, 7, 'Chunk 4-1', NULL, NULL),
	(6, 9, 'Chunk 5-1', NULL, NULL)
ON CONFLICT DO NOTHING;

-- ImageChunks (image regions from pages)
INSERT INTO image_chunk (id, parent_page, image_path, embedding, embeddings) VALUES
	(1, 1, 6, NULL, NULL),
	(2, 2, 7, NULL, NULL),
	(3, 3, 8, NULL, NULL),
	(4, 4, 9, NULL, NULL),
	(5, 5, 10, NULL, NULL)
ON CONFLICT DO NOTHING;

-- Caption-Chunk relations
INSERT INTO caption_chunk_relation (caption_id, chunk_id) VALUES
	(1, 1), (1, 2), (3, 3), (5, 4), (7, 5), (9, 6)
ON CONFLICT DO NOTHING;

-- Queries
INSERT INTO query (id, query, generation_gt, embedding, embeddings) VALUES
	(1, 'What is Doc One about?', ARRAY['alpha'], NULL, NULL),
	(2, 'Find details in Doc Two', ARRAY['beta'], NULL, NULL),
	(3, 'Summarize Doc Three', ARRAY['gamma'], NULL, NULL),
	(4, 'Topics in Doc Four', ARRAY['delta'], NULL, NULL),
	(5, 'Explain Doc Five', ARRAY['epsilon'], NULL, NULL)
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
	(1, 1,  NULL, 100, 1200, '{"notes": "ok"}'),
	(1, 2,  NULL,  120, 1400, '{"notes": "better"}'),
	(2, 1,  'Generated text 1', 200, 2500, '{"len": 20}')
ON CONFLICT DO NOTHING;


-- Experiment results
INSERT INTO evaluation_result (query_id, pipeline_id, metric_id, metric_result) VALUES
	(1, 1, 1, 0.8),
	(1, 2, 1, 0.85),
	(2, 1, 2, 0.6)
ON CONFLICT DO NOTHING;

-- Retrieved results (image/text)
INSERT INTO image_chunk_retrieved_result (query_id, pipeline_id, metric_id, image_chunk_id) VALUES
	(3, 1, 1, 1), (4, 2, 1, 2)
ON CONFLICT DO NOTHING;

INSERT INTO chunk_retrieved_result (query_id, pipeline_id, metric_id, chunk_id) VALUES
	(1, 1, 1, 1), (2, 2, 1, 3)
ON CONFLICT DO NOTHING;

-- Summary
INSERT INTO summary (pipeline_id, metric_id, metric_result, token_usage, execution_time, result_metadata) VALUES
	(1, 1, 0.82, 220, 2600, '{"run": 1}'),
	(2, 1, 0.85, 120, 1400, '{"run": 1}')
ON CONFLICT DO NOTHING;

-- Advance sequences to max IDs to prevent conflicts on future inserts
SELECT setval(pg_get_serial_sequence('file','id'), (SELECT COALESCE(MAX(id), 1) FROM file), true);
SELECT setval(pg_get_serial_sequence('document','id'), (SELECT COALESCE(MAX(id), 1) FROM document), true);
SELECT setval(pg_get_serial_sequence('page','id'), (SELECT COALESCE(MAX(id), 1) FROM page), true);
SELECT setval(pg_get_serial_sequence('caption','id'), (SELECT COALESCE(MAX(id), 1) FROM caption), true);
SELECT setval(pg_get_serial_sequence('chunk','id'), (SELECT COALESCE(MAX(id), 1) FROM chunk), true);
SELECT setval(pg_get_serial_sequence('image_chunk','id'), (SELECT COALESCE(MAX(id), 1) FROM image_chunk), true);
SELECT setval(pg_get_serial_sequence('query','id'), (SELECT COALESCE(MAX(id), 1) FROM query), true);

COMMIT;
