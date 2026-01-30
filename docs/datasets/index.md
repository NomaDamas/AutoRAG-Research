# Datasets

Available benchmarks for RAG evaluation.

## Comparison

| Dataset | Modality | Queries | Documents | Generation GT | Use |
|---------|----------|---------|-----------|---------------|-----|
| BEIR | Text | 300-15k | 3k-5M | No | Retrieval |
| MTEB | Text | varies | varies | No | Retrieval |
| RAGBench | Text | varies | varies | Yes | RAG |
| MrTyDi | Text | varies | varies | No | Multilingual |
| BRIGHT | Text | varies | varies | No | Retrieval |
| Open-RAGBench | Text | varies | varies | Yes | RAG |
| ViDoRe | Multimodal | varies | varies | No | Visual |
| ViDoRe v2 | Multimodal | varies | varies | No | Visual |
| ViDoRe v3 | Multimodal | varies | varies | No | Visual |
| VisRAG | Multimodal | varies | varies | No | Visual |

## Text vs Multimodal

**Text datasets**:

- Documents as plain text
- Text embeddings + BM25 tokens
- Use: [Text Retrieval Tutorial](../tutorial/text-retrieval.md)

**Multimodal datasets**:

- Documents as images (PDF pages)
- Image embeddings (ColPali)
- Use: [Multimodal Tutorial](../tutorial/multimodal.md)

## Browse

- [Text Datasets](text/beir.md)
- [Multimodal Datasets](multimodal/vidore.md)
