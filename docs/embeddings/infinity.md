# Infinity

Multi-vector embeddings via an [Infinity Embedding API](https://github.com/michaelfeil/infinity) server.

## Overview

| Field | Value |
|-------|-------|
| Type | API |
| Modality | Text + Image |
| Provider | [Infinity](https://github.com/michaelfeil/infinity) (self-hosted) |
| Default Model | `michaelfeil/colqwen2-v0.1` |
| Env Variable | `INFINITY_API_URL` |
| GPU Required | No (client-side) |

`InfinityEmbeddings` connects to a running Infinity server that serves ColPali/ColQwen2 models. The server handles GPU inference; this client only needs HTTP access. The embeddings are multi-vector (one vector per token/patch), suitable for MaxSim late interaction retrieval.

## Prerequisites

You need a running Infinity server. Start one with Docker:

```bash
docker run -it --gpus all \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -p 7997:7997 \
  michaelf34/infinity:latest \
  v2 \
  --model-id michaelfeil/colqwen2-v0.1 \
  --port 7997
```

## Configuration

```yaml
_target_: autorag_research.embeddings.infinity.InfinityEmbeddings
model_name: michaelfeil/colqwen2-v0.1
url: ${oc.env:INFINITY_API_URL,http://localhost:7997}
encoding: base64
hidden_dim: 128
timeout: 60.0
max_retries: 3
embed_batch_size: 10
```

## Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `url` | str | `http://localhost:7997` | Infinity API server URL |
| `model_name` | str | `michaelfeil/colqwen2-v0.1` | Model name served by Infinity |
| `encoding` | str | `base64` | Response encoding: `base64` or `float` |
| `hidden_dim` | int | `128` | Hidden dimension for reshaping flat arrays |
| `timeout` | float | `60.0` | HTTP request timeout in seconds |
| `max_retries` | int | `3` | Max retry attempts with exponential backoff |
| `embed_batch_size` | int | `10` | Batch size for `embed_documents_batch` / `embed_images_batch` |

## Models

Any model supported by the Infinity server can be used. Common ColEncoder models:

| Model | Type | Description |
|-------|------|-------------|
| `michaelfeil/colqwen2-v0.1` | ColQwen2 | Multi-modal (text + image), 128-dim |
| `vidore/colpali-v1.3` | ColPali | Multi-modal (text + image), 128-dim |
| `colbert-ir/colbertv2.0` | ColBERT | Text-only, 128-dim |

## Usage

### Text Embedding

```python
from autorag_research.embeddings.infinity import InfinityEmbeddings

embeddings = InfinityEmbeddings(
    url="http://localhost:7997",
    model_name="michaelfeil/colqwen2-v0.1",
)

# Single text
text_emb = embeddings.embed_text("What is retrieval augmented generation?")
# Returns: list[list[float]] -- shape (num_tokens, 128)

# Single query (delegates to embed_text)
query_emb = embeddings.embed_query("What is RAG?")

# Batch -- single API call
doc_embs = embeddings.embed_documents(["Document 1", "Document 2", "Document 3"])
# Returns: list[list[list[float]]] -- one multi-vector per document
```

### Image Embedding

```python
# From file path
image_emb = embeddings.embed_image("path/to/image.png")

# From bytes
with open("image.jpg", "rb") as f:
    image_emb = embeddings.embed_image(f.read())

# Batch images -- single API call
image_embs = embeddings.embed_images(["img1.png", "img2.png", "img3.png"])
```

### Async

```python
import asyncio

async def main():
    emb = InfinityEmbeddings()

    text_emb = await emb.aembed_text("async text")
    query_emb = await emb.aembed_query("async query")
    image_emb = await emb.aembed_image("image.png")

    doc_embs = await emb.aembed_documents(["doc1", "doc2"])
    img_embs = await emb.aembed_images(["img1.png", "img2.png"])

asyncio.run(main())
```

### With Automatic Batching

For large inputs, use the batch methods which chunk inputs by `embed_batch_size`:

```python
embeddings = InfinityEmbeddings(embed_batch_size=5)

# Processes 100 texts in batches of 5
all_embs = embeddings.embed_documents_batch(texts_list)

# Processes 50 images in batches of 5
all_img_embs = embeddings.embed_images_batch(image_paths)
```

## API Protocol

The Infinity server exposes `POST /embeddings` with the following request format:

```json
{
    "input": ["text1", "text2"],
    "model": "michaelfeil/colqwen2-v0.1",
    "modality": "text",
    "encoding": "base64"
}
```

For images, `modality` is set to `"image"` and `input` contains base64-encoded JPEG strings.

The response contains multi-vector embeddings (one embedding per input, each embedding is a list of vectors).

## Encoding Formats

| Encoding | Description | Performance |
|----------|-------------|-------------|
| `base64` | Base64-encoded float32 arrays | Faster transfer, smaller payload |
| `float` | Plain JSON float arrays | Easier to debug, larger payload |

The `base64` encoding (default) is recommended for production use as it reduces network transfer size.
