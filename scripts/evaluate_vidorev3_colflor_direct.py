from __future__ import annotations

import argparse, json, math, os, shutil, tempfile, time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm


def load_model(model_path: str, device: str):
    from colpali_engine.models import ColFlor, ColFlorProcessor
    for attr in ("_supports_sdpa", "_supports_flash_attn", "_supports_flash_attn_2"):
        if not hasattr(ColFlor, attr):
            setattr(ColFlor, attr, False)
    model_path = _prepare_colflor_checkpoint(model_path)
    model = ColFlor.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device).eval()
    processor = ColFlorProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def _prepare_colflor_checkpoint(model_path: str) -> str:
    """Make older saved ColFlor checkpoints load with the ColFlor git package.

    Some locally saved checkpoints have ``vision_config.model_type`` missing
    after ``save_pretrained``.  The ColFlor implementation asserts that this
    value is ``davit``.  Rather than mutating the original artifacts, create a
    lightweight temporary checkpoint directory with an adjusted config and
    symlinks to all other files.
    """
    src = Path(model_path)
    cfg_path = src / "config.json"
    if not cfg_path.exists():
        return model_path
    cfg = json.loads(cfg_path.read_text())
    changed = False
    vision = cfg.setdefault("vision_config", {})
    if vision.get("model_type") != "davit":
        vision["model_type"] = "davit"
        changed = True
    text = cfg.setdefault("text_config", {})
    if "forced_bos_token_id" not in text:
        text["forced_bos_token_id"] = cfg.get("bos_token_id", 0)
        changed = True
    if not changed:
        return model_path
    tmp = Path(tempfile.mkdtemp(prefix=f"colflor_ckpt_{src.name}_"))
    for child in src.iterdir():
        target = tmp / child.name
        if child.name == "config.json":
            target.write_text(json.dumps(cfg, indent=2))
        elif child.is_dir():
            shutil.copytree(child, target, symlinks=True)
        else:
            target.symlink_to(child)
    return str(tmp)


def to_device(batch: dict[str, Any], device: str, dtype: torch.dtype | None = None):
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            if dtype is not None and torch.is_floating_point(v):
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def embed_queries(model, processor, queries: list[str], device: str, batch_size: int):
    out=[]
    for i in tqdm(range(0, len(queries), batch_size), desc="embed queries"):
        batch = processor.process_queries(queries[i:i+batch_size])
        emb = model(**to_device(batch, device))
        out.extend([e.detach().float().cpu() for e in emb])
    return out


@torch.no_grad()
def embed_images(model, processor, images: list[Any], device: str):
    batch = processor.process_images(images)
    model_dtype = next(model.parameters()).dtype
    emb = model(**to_device(batch, device, dtype=model_dtype))
    return [e.detach().float().cpu() for e in emb]


def maxsim_scores(query_embs: list[torch.Tensor], doc_emb: torch.Tensor, device: str, query_batch: int = 128):
    d = doc_emb.to(device)
    scores=[]
    for i in range(0, len(query_embs), query_batch):
        qs = query_embs[i:i+query_batch]
        # Variable query token counts: loop, but keep doc on GPU.
        vals=[]
        for q in qs:
            qd = q.to(device)
            vals.append((qd @ d.T).max(dim=1).values.sum().float().cpu())
        scores.append(torch.stack(vals))
    return torch.cat(scores)


def dcg(rels: list[float]) -> float:
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def compute_metrics(top_ids_by_q: dict[int, list[int]], qrels: dict[int, dict[int, float]]):
    ks=[1,5,10]
    sums={f"recall@{k}":0.0 for k in ks}
    sums.update({f"ndcg@{k}":0.0 for k in [5,10]})
    n=0
    per_query=[]
    for qid, relmap in qrels.items():
        if not relmap: continue
        pred=top_ids_by_q.get(qid, [])
        rel_ids=set(relmap)
        n+=1
        row={"query_id":qid}
        for k in ks:
            hit=len(set(pred[:k]) & rel_ids)
            val=hit/len(rel_ids)
            row[f"recall@{k}"]=val; sums[f"recall@{k}"]+=val
        for k in [5,10]:
            gains=[float(relmap.get(pid,0.0)) for pid in pred[:k]]
            ideal=sorted([float(v) for v in relmap.values()], reverse=True)[:k]
            val=dcg(gains)/(dcg(ideal) or 1.0)
            row[f"ndcg@{k}"]=val; sums[f"ndcg@{k}"]+=val
        per_query.append(row)
    avg={k:(v/n if n else 0.0) for k,v in sums.items()}
    avg["num_eval_queries"]=n
    return avg, per_query


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model-name', required=True)
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--domain', required=True)
    ap.add_argument('--output-dir', default='experiments/vidorev3_colflor/results')
    ap.add_argument('--query-batch-size', type=int, default=16)
    ap.add_argument('--image-batch-size', type=int, default=8)
    ap.add_argument('--top-k', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit-queries', type=int)
    ap.add_argument('--limit-corpus', type=int)
    args=ap.parse_args()

    out_dir=Path(args.output_dir)/args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file=out_dir/f'{args.domain}.json'
    if out_file.exists():
        print(f'Already exists: {out_file}')
        print(out_file.read_text())
        return

    t0=time.time(); dataset_path=f'vidore/vidore_v3_{args.domain}'
    qrels_ds=load_dataset(dataset_path, 'qrels', split='test')
    queries_ds=load_dataset(dataset_path, 'queries', split='test')
    qrels: dict[int, dict[int,float]]={}
    for r in qrels_ds:
        if int(r.get('score',0)) <= 0: continue
        qid=int(r['query_id']); cid=int(r['corpus_id'])
        qrels.setdefault(qid,{})[cid]=float(r.get('score',1))
    query_rows=[r for r in queries_ds if int(r['query_id']) in qrels]
    if args.limit_queries: query_rows=query_rows[:args.limit_queries]
    qids=[int(r['query_id']) for r in query_rows]
    queries=[str(r['query']) for r in query_rows]
    qrels={qid:qrels[qid] for qid in qids if qid in qrels}
    qid_to_idx={qid:i for i,qid in enumerate(qids)}
    print(f'Loaded {dataset_path}: queries={len(queries)} qrels={len(qrels)}')

    model, processor=load_model(args.model_path, args.device)
    query_embs=embed_queries(model, processor, queries, args.device, args.query_batch_size)
    top_scores=torch.full((len(qids), args.top_k), -1e30)
    top_ids=torch.full((len(qids), args.top_k), -1, dtype=torch.long)

    corpus=load_dataset(dataset_path, 'corpus', streaming=True, split='test')
    images=[]; ids=[]; seen=0
    pbar=tqdm(desc='score corpus')
    def flush():
        nonlocal images, ids
        if not images: return
        doc_embs=embed_images(model, processor, images, args.device)
        for cid, demb in zip(ids, doc_embs, strict=True):
            scores=maxsim_scores(query_embs, demb, args.device)
            cand_scores=torch.cat([top_scores, scores[:,None]], dim=1)
            cand_ids=torch.cat([top_ids, torch.full((len(qids),1), int(cid), dtype=torch.long)], dim=1)
            vals, idx=cand_scores.topk(args.top_k, dim=1)
            top_scores.copy_(vals)
            top_ids.copy_(cand_ids.gather(1, idx))
        images=[]; ids=[]
    for row in corpus:
        if args.limit_corpus and seen >= args.limit_corpus: break
        cid=int(row['corpus_id'])
        images.append(row['image']); ids.append(cid); seen += 1
        if len(images) >= args.image_batch_size: flush(); pbar.update(args.image_batch_size)
    flush(); pbar.close()

    top_ids_by_q={qid:[int(x) for x in top_ids[i].tolist() if int(x)>=0] for qid,i in qid_to_idx.items()}
    avg, per_query=compute_metrics(top_ids_by_q, qrels)
    result={
        'model_name': args.model_name, 'model_path': args.model_path, 'domain': args.domain,
        'dataset': dataset_path, 'metrics': avg, 'elapsed_sec': time.time()-t0,
        'num_queries': len(qids), 'num_corpus_scored': seen, 'top_k': args.top_k,
    }
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (out_dir/f'{args.domain}.per_query.jsonl').write_text('\n'.join(json.dumps(x) for x in per_query)+"\n")
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
