Table Document {
  id bigserial [pk]
  path bigint [ref: - File.id]
  filename text
  author text
  title text
  doc_metadata jsonb
}

Table Page {
  id bigserial [pk]
  page_num int [not null]
  document_id bigint [ref: > Document.id, not null]
  image_contents bytea
  mimetype varchar(255)
  page_metadata jsonb

  indexes {
    (document_id, page_num) [unique]
  }
}

Table File {
  id bigserial [pk]
  type varchar(255) [not null] // raw, image, audio, video
  path varchar(255) [not null]
}

Table Chunk {
  id bigserial [pk]
  contents text [not null]
  embedding vector(768)
  embeddings vector[](768)
  bm25_tokens bm25vector // tokenized sparse vector (TF) for BM25 search (requires vchord_bm25 extension, optional)
  is_table boolean [default: false]
  table_type varchar(255) // markdown, xml, html
}

Table ImageChunk {
  id bigserial [pk]
  parent_page bigint [ref: > Page.id]
  contents bytea [not null]
  mimetype varchar(255) [not null]
  embedding vector(768)
  embeddings vector[](768)
}

Table PageChunkRelation {
  page_id bigint [ref: > Page.id, pk]
  chunk_id bigint [ref: > Chunk.id, pk]
}

Table Query {
  id bigserial [pk]
  contents text [not null]
  query_to_llm text
  generation_gt text[]
  embedding vector(768)
  embeddings vector[](768)
  bm25_tokens bm25vector // pre-computed tokens for BM25 search (no index needed - index is on chunks only)
}

Table RetrievalRelation {
  query_id bigint [ref: > Query.id, not null]
  group_index int [not null]
  group_order int [not null]

  chunk_id bigint [ref: > Chunk.id]
  image_chunk_id bigint [ref: > ImageChunk.id]
  score int [default: 1]

  indexes {
    (query_id, group_index, group_order) [pk]
  }

  // chunk_id, image_chunk_id 둘 중 하나만 null인 제약 추가 필요
}

Table Pipeline {
  id bigint [pk]
  name varchar(255) [not null]
  config jsonb [not null]
}

Table Metric {
  id bigint [pk]
  name varchar(255) [not null]
  type varchar(255) [not null] // retrieval, generation
}

Table ExecutorResult {
  query_id bigint [ref: > Query.id, not null]
  pipeline_id bigint [ref: > Pipeline.id, not null]

  generation_result text
  token_usage jsonb
  execution_time int //아무튼 시간임
  result_metadata jsonb

  indexes {
    (query_id, pipeline_id) [pk]
  }
}

Table EvaluationResult {
  query_id bigint [ref: > Query.id, not null]
  pipeline_id bigint [ref: > Pipeline.id, not null]
  metric_id bigint [ref: > Metric.id, not null]

  metric_result float [not null]

  indexes {
    (query_id, pipeline_id, metric_id) [pk]
    }
}

Table ImageChunkRetrievedResult {
  query_id bigint [ref: > Query.id, not null]
  pipeline_id bigint [ref: > Pipeline.id, not null]
  image_chunk_id bigint [ref: > ImageChunk.id, not null]
  rel_score float

  indexes {
    (query_id, pipeline_id, image_chunk_id) [pk]
  }
}

Table ChunkRetrievedResult {
  query_id bigint [ref: > Query.id, not null]
  pipeline_id bigint [ref: > Pipeline.id, not null]
  chunk_id bigint [ref: > Chunk.id, not null]
  rel_score float

  indexes {
    (query_id, pipeline_id, chunk_id) [pk]
  }
}

Table Summary {
  pipeline_id bigint [ref: > Pipeline.id, not null]
  metric_id bigint [ref: > Metric.id, not null]
  metric_result float [not null]
  token_usage jsonb
  execution_time int //아무튼 시간임
  result_metadata jsonb

  indexes {
    (pipeline_id, metric_id) [pk]
  }
}
