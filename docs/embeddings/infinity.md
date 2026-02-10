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

Connects to a running Infinity server that serves ColPali/ColQwen2 models. The server handles GPU inference; this client only needs HTTP access. Produces multi-vector embeddings (one vector per token/patch) for MaxSim late interaction retrieval.

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

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | str | `http://localhost:7997` | Infinity API server URL |
| `model_name` | str | `michaelfeil/colqwen2-v0.1` | Model name served by Infinity |
| `encoding` | str | `base64` | Response encoding: `base64` or `float` |
| `hidden_dim` | int | `128` | Hidden dimension for reshaping flat arrays |
| `timeout` | float | `60.0` | HTTP request timeout in seconds |
| `max_retries` | int | `3` | Max retry attempts with exponential backoff |
| `embed_batch_size` | int | `10` | Batch size for batch embedding methods |

## Supported Models

Any model supported by the Infinity server can be used. Common ColEncoder models:

| Model | Type | Description |
|-------|------|-------------|
| `michaelfeil/colqwen2-v0.1` | ColQwen2 | Multi-modal (text + image), 128-dim |
| `vidore/colpali-v1.3` | ColPali | Multi-modal (text + image), 128-dim |
| `colbert-ir/colbertv2.0` | ColBERT | Text-only, 128-dim |
