#!/usr/bin/env python3
import csv
import json
from pathlib import Path

base = Path(__file__).resolve().parent / "results"
out_csv = Path(__file__).resolve().parent / "summary.csv"
rows = []
for path in sorted(base.glob("*/*.json")):
    with path.open() as f:
        data = json.load(f)
    metrics = data.get("metrics", {})
    rows.append({
        "model_name": data.get("model_name", path.parent.name),
        "domain": data.get("domain", path.stem),
        "ndcg@5": metrics.get("ndcg@5"),
        "ndcg@10": metrics.get("ndcg@10"),
        "recall@1": metrics.get("recall@1"),
        "recall@5": metrics.get("recall@5"),
        "recall@10": metrics.get("recall@10"),
        "num_eval_queries": metrics.get("num_eval_queries"),
        "num_queries": data.get("num_queries"),
        "num_corpus_scored": data.get("num_corpus_scored"),
        "path": str(path),
    })
fields = [
    "model_name",
    "domain",
    "ndcg@5",
    "ndcg@10",
    "recall@1",
    "recall@5",
    "recall@10",
    "num_eval_queries",
    "num_queries",
    "num_corpus_scored",
    "path",
]
with out_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} rows to {out_csv}")
for row in rows[-20:]:
    print(f"{row['model_name']:<32} {row['domain']:<18} nDCG@5={row['ndcg@5']:.6f} R@10={row['recall@10']:.6f}")
