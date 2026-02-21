# Metrics

Evaluation measures for retrieval and generation.

## Retrieval Metrics

| Metric | Measures | Range |
|--------|----------|-------|
| [Recall@k](retrieval/recall.md) | Ground truth coverage | [0, 1] |
| [Precision@k](retrieval/precision.md) | Retrieved relevance | [0, 1] |
| [F1@k](retrieval/f1.md) | Recall + Precision balance | [0, 1] |
| [NDCG@k](retrieval/ndcg.md) | Ranking quality | [0, 1] |
| [MRR](retrieval/mrr.md) | First relevant position | [0, 1] |
| [MAP](retrieval/map.md) | Average precision | [0, 1] |

## Generation Metrics

| Metric | Measures | Range |
|--------|----------|-------|
| [BLEU](generation/bleu.md) | N-gram overlap | [0, 1] |
| [METEOR](generation/meteor.md) | Alignment | [0, 1] |
| [ROUGE](generation/rouge.md) | Recall overlap | [0, 1] |
| [BERTScore](generation/bert-score.md) | Semantic similarity | [-1, 1] |
| [SemScore](generation/sem-score.md) | Embedding similarity | [-1, 1] |
| [Grounded Refusal F1](generation/grounded-refusal-f1.md) | Refusal correctness | [0, 1] |
| [Answer Correctness F1](generation/answer-correctness-f1.md) | Calibrated correctness | [0, 1] |

## Choosing Metrics

| Goal | Metrics |
|------|---------|
| Find all relevant docs | Recall |
| Rank correctly | NDCG |
| Find relevant quickly | MRR |
| Text similarity | ROUGE, BLEU |
| Semantic correctness | BERTScore |
| Refusal behavior quality | Grounded Refusal F1 |
| Paper-aligned calibrated correctness | Answer Correctness F1 |

## Browse

- [Retrieval Metrics](retrieval/index.md)
- [Generation Metrics](generation/index.md)
