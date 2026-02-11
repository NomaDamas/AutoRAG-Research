# Embeddings

Multi-vector embedding models for late interaction retrieval (MaxSim).

## Overview

Unlike single-vector embeddings that produce one vector per input, multi-vector models produce **one vector per token/patch**. This enables late interaction retrieval where query-document similarity is computed as a sum of maximum similarities between individual token vectors (MaxSim scoring).

## Available Embeddings

| Embedding | Type | Modality | GPU Required |
|-----------|------|----------|--------------|
| [Infinity](infinity.md) | API | Text + Image | No (server-side) |
| ColPali | Local | Text + Image | Yes |
| BiPali | Local | Text + Image | Yes |

## Base Classes

All multi-vector embeddings extend from the base classes in `autorag_research.embeddings.base`:

```python
from autorag_research.embeddings.base import (
    MultiVectorBaseEmbedding,       # Text-only multi-vector
    MultiVectorMultiModalEmbedding, # Text + Image multi-vector
)
```

## Methods

### Text Embedding

| Method | Description |
|--------|-------------|
| `embed_text(text)` | Embed a single text |
| `aembed_text(text)` | Async embed a single text |
| `embed_query(query)` | Embed a single query |
| `aembed_query(query)` | Async embed a single query |
| `embed_documents(texts)` | Embed multiple texts |
| `aembed_documents(texts)` | Async embed multiple texts |
| `embed_documents_batch(texts)` | Embed with automatic batching |
| `aembed_documents_batch(texts)` | Async embed with automatic batching |

### Image Embedding (MultiModal only)

| Method | Description |
|--------|-------------|
| `embed_image(img)` | Embed a single image |
| `aembed_image(img)` | Async embed a single image |
| `embed_images(imgs)` | Embed multiple images |
| `aembed_images(imgs)` | Async embed multiple images |
| `embed_images_batch(imgs)` | Embed with automatic batching |
| `aembed_images_batch(imgs)` | Async embed with automatic batching |

## Image Input Types

Image methods accept any of the following types:

```python
from autorag_research.types import ImageType
# ImageType = str | bytes | Path | BytesIO
```

- `str` -- file path as string
- `Path` -- `pathlib.Path` object
- `bytes` -- raw image bytes
- `BytesIO` -- in-memory file-like object
