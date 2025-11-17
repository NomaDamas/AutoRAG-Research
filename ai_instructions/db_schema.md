Table Document {
  id bigserial [pk]
  filepath bigint [ref: - File.id]
  filename text
  author text
  title text
  doc_metadata jsonb
}

Table Page {
  id bigserial [pk]
  page_num int [not null]
  document_id bigint [ref: > Document.id, not null]
  image_path bigint [ref: - File.id]
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

Table Caption {
  id bigserial [pk]
  page_id bigint [ref: > Page.id, not null]
  contents text [not null]
}

Table Chunk {
  id bigserial [pk]
  parent_caption bigint [ref: > Caption.id]
  contents text [not null]
  embedding vector(768)
  embeddings vector[](768)
}

Table ImageChunk {
  id bigserial [pk]
  parent_page bigint [ref: > Page.id]
  image_path bigint [ref: - File.id, not null]
  embedding vector(768)
  embeddings vector[](768)
}

Table CaptionChunkRelation {
  caption_id bigint [ref: > Caption.id, pk]
  chunk_id bigint [ref: > Chunk.id, pk]
}

Table Query {
  id bigserial [pk]
  query text [not null]
  generation_gt text[] [not null]
}

Table RetrievalRelation {
  query_id bigint [ref: > Query.id, not null]
  group_index int [not null]
  group_order int [not null]

  chunk_id bigint [ref: > Chunk.id]
  image_chunk_id bigint [ref: > ImageChunk.id]

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

Table ExperimentResult {
  query_id bigint [ref: > Query.id, not null]
  pipeline_id bigint [ref: > Pipeline.id, not null]
  metric_id bigint [ref: > Metric.id, not null]

  generation_result text
  metric_result float
  token_usage int
  execution_time int //아무튼 시간임
  result_metadata jsonb

  indexes {
    (query_id, pipeline_id, metric_id) [pk]
  }
}

Table ImageChunkRetrievedResult {
  query_id bigint [ref: > Query.id, not null]
  pipeline_id bigint [ref: > Pipeline.id, not null]
  metric_id bigint [ref: > Metric.id, not null]
  image_chunk_id bigint [ref: > ImageChunk.id, not null]

  indexes {
    (query_id, pipeline_id, metric_id, image_chunk_id) [pk]
  }
}

Table ChunkRetrievedResult {
  query_id bigint [ref: > Query.id, not null]
  pipeline_id bigint [ref: > Pipeline.id, not null]
  metric_id bigint [ref: > Metric.id, not null]
  chunk_id bigint [ref: > Chunk.id, not null]

  indexes {
    (query_id, pipeline_id, metric_id, chunk_id) [pk]
  }
}

Table Summary {
  pipeline_id bigint [ref: > Pipeline.id, not null]
  metric_id bigint [ref: > Metric.id, not null]
  metric_result float [not null]
  token_usage int
  execution_time int // This has to be time
  result_metadata jsonb

  indexes {
    (pipeline_id, metric_id) [pk]
  }
}
