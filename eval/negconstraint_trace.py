"""
TRACE NegConstraint Evaluation

Runs TRACE and its controlled variants on NegConstraint dataset.
Supports multiple encoders: RepLLaMA, BGE-large-en-v1.5

Variants (Table 8 in paper):
  1. Base retriever:   S_final = S_base
  2. + Positive:       S_final = z_full + p  (ablation=pos_only)
  3. + Raw Negative:   S_final = z_full + p - z_neg  (ablation=raw_neg_subtract)
  4. TRACE:            S_final = z_full + p*g - h  (ablation=full)

Usage:
  cd /home/luwa/Documents/DSCLR-remote && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/luwa/.conda/envs/dsclr/bin/python -m eval.negconstraint_trace --encoder repllama --ablation full

  cd /home/luwa/Documents/DSCLR-remote && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/luwa/.conda/envs/dsclr/bin/python -m eval.negconstraint_trace --encoder repllama --ablation pos_only
"""

import os, sys, json, csv, argparse, time, logging
import numpy as np, torch, torch.nn.functional as F

torch.cuda._lazy_init()

import pytrec_eval
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "dataset/NegConstraint/NegConstraint"
DUAL_QUERIES_PATH = "dataset/NegConstraint/NegConstraint/dual_queries/NegConstraint_TSC_BALANCED_t01.jsonl"

CANDIDATE_DEPTH = 100  # Top-K candidates per query

# Import TRACE scoring components
from eval.engine_trace import TRACEEvaluator, robust_standardize, _mean_std_standardize, _mad, fit_huber_regression

ENCODER_CONFIGS = {
    "repllama": {
        "class": "repllama",
        "model_name": "samaya-ai/RepLLaMA-reproduced",
        "embedding_cache": "dataset/NegConstraint/NegConstraint/embeddings/negconstraint_repllama_corpus.pt",
        "query_prefix": "",
        "doc_prefix": "",
    },
    "bge": {
        "class": "sentence_transformer",
        "model_name": "BAAI/bge-large-en-v1.5",
        "embedding_cache": "dataset/NegConstraint/NegConstraint/embeddings/negconstraint_bge_corpus.pt",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "",
    },
    "bge-m3": {
        "class": "sentence_transformer",
        "model_name": "BAAI/bge-m3",
        "embedding_cache": "dataset/NegConstraint/NegConstraint/embeddings/negconstraint_bge_m3_corpus.pt",
        "query_prefix": "",
        "doc_prefix": "",
    },
    "bge-small": {
        "class": "sentence_transformer",
        "model_name": "BAAI/bge-small-en-v1.5",
        "embedding_cache": "dataset/NegConstraint/NegConstraint/embeddings/negconstraint_bge_small_corpus.pt",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "",
    },
}


def is_none_query(q):
    return not q or q.strip().lower() in ['none', '[none]', 'n/a', 'null', '']


def load_data():
    corpus = {}
    with open(os.path.join(DATA_DIR, "corpus.jsonl"), encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            corpus[str(doc['_id'])] = doc.get('text', '')

    queries = {}
    with open(os.path.join(DATA_DIR, "queries.jsonl"), encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            queries[str(q['_id'])] = q.get('text', '')

    qrels = {}
    with open(os.path.join(DATA_DIR, "test.tsv"), encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            qid, did, score = str(row[0]), str(row[1]), int(row[2])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = score

    dual_data = {}
    with open(DUAL_QUERIES_PATH) as f:
        for line in f:
            r = json.loads(line)
            dual_data[r['qid']] = r

    return corpus, queries, qrels, dual_data


def create_encoder(config, encode_batch_size=0):
    bs = encode_batch_size if encode_batch_size > 0 else 64
    if config["class"] == "repllama":
        from eval.models.repllama_encoder import RepLLaMAEncoder
        return RepLLaMAEncoder(
            model_name=config["model_name"],
            device="cuda", batch_size=bs
        )
    else:
        from eval.models.encoder import SentenceTransformerEncoder
        return SentenceTransformerEncoder(
            model_name=config["model_name"],
            device="cuda", batch_size=bs if bs > 0 else 256,
            normalize_embeddings=True
        )


def encode_and_cache_corpus(corpus, encoder, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        logger.info(f"Loading cached corpus embeddings from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        doc_ids = cache["doc_ids"]
        doc_embeddings = cache["embeddings"]
        logger.info(f"Cached: {len(doc_ids)} docs, shape={doc_embeddings.shape}")
        return doc_ids, doc_embeddings

    doc_ids = sorted(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]
    logger.info(f"Encoding {len(doc_texts)} documents...")
    doc_embeddings = encoder.encode_documents(doc_texts, batch_size=encoder.batch_size)
    if doc_embeddings.dim() == 2:
        doc_embeddings = F.normalize(doc_embeddings.float(), p=2, dim=1)

    torch.save({
        "doc_ids": doc_ids,
        "embeddings": doc_embeddings.cpu().float(),
    }, cache_path)
    logger.info(f"Cached corpus embeddings to {cache_path}")
    return doc_ids, doc_embeddings


def evaluate(run_data, qrels, eval_qids):
    evaluator = pytrec_eval.RelevanceEvaluator(
        {qid: qrels[qid] for qid in eval_qids},
        {'ndcg_cut_5', 'ndcg_cut_10', 'map_cut_100', 'recall_5', 'recall_10', 'recall_100'}
    )
    results = evaluator.evaluate(run_data)
    metrics = {}
    for m in ['ndcg_cut_5', 'ndcg_cut_10', 'map_cut_100', 'recall_5', 'recall_10', 'recall_100']:
        vals = [results[qid][m] for qid in eval_qids if qid in results]
        metrics[m] = np.mean(vals) if vals else 0.0
    return metrics, results


def apply_trace_scoring(s_base, s_pos, s_neg, has_neg,
                        lambda_boundary=1.0, tau_decay=0.2,
                        huber_delta=1.345, eps=1e-6,
                        ablation="full"):
    """Apply TRACE scoring for a single query's candidate set (NegConstraint)."""
    n = s_base.numel()

    # Robust standardization
    z_full = robust_standardize(s_base.float(), eps)
    z_pos = robust_standardize(s_pos.float(), eps)
    z_neg = robust_standardize(s_neg.float(), eps)

    if not has_neg or n < 3:
        p = torch.clamp(z_pos, min=0)
        if ablation == "z_full_only":
            return z_full
        elif ablation == "raw_neg_subtract":
            return z_full + p
        else:
            return z_full + p

    # Fit Huber regression
    a_hat, b_hat = fit_huber_regression(z_neg, z_pos, delta=huber_delta)

    # Compute residual
    e = z_neg.float() - a_hat - b_hat * z_pos.float()
    e_median = e.median()
    e_mad = _mad(e, eps)
    r = (e - e_median) / e_mad

    # Composition
    p = torch.clamp(z_pos, min=0)
    h = torch.clamp(r - lambda_boundary, min=0)
    g = torch.exp(-h / tau_decay)

    if ablation == "z_full_only":
        return z_full
    elif ablation == "full":
        return z_full + p * g - h
    elif ablation == "pos_only":
        return z_full + p
    elif ablation == "raw_neg_subtract":
        return z_full + p - z_neg
    elif ablation == "linear":
        return z_full + p - r
    elif ablation == "no_gate":
        return z_full + p - h
    elif ablation == "gate_only":
        return z_full + p * g
    else:
        return z_full + p * g - h


def main():
    parser = argparse.ArgumentParser(description="TRACE NegConstraint Evaluation")
    parser.add_argument("--encoder", type=str, default="repllama", choices=["repllama", "bge", "bge-m3", "bge-small"])
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "z_full_only", "pos_only", "raw_neg_subtract", "linear", "no_gate", "gate_only"])
    parser.add_argument("--lambda_boundary", type=float, default=1.0)
    parser.add_argument("--tau_decay", type=float, default=0.2)
    parser.add_argument("--huber_delta", type=float, default=1.345)
    parser.add_argument("--candidate_depth", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--encode_batch_size", type=int, default=0)
    args = parser.parse_args()

    config = ENCODER_CONFIGS[args.encoder]
    encoder_name = args.encoder
    logger.info("=" * 60)
    logger.info(f"TRACE NegConstraint Evaluation: {encoder_name} encoder, ablation={args.ablation}")
    logger.info("=" * 60)

    corpus, queries, qrels, dual_data = load_data()
    eval_qids = sorted(set(qrels.keys()) & set(queries.keys()))
    logger.info(f"Corpus: {len(corpus)} docs, Queries: {len(queries)}, Eval queries: {len(eval_qids)}")
    logger.info(f"Dual queries loaded: {len(dual_data)}")

    encoder = create_encoder(config, args.encode_batch_size)
    enc_bs = args.encode_batch_size if args.encode_batch_size > 0 else 256

    doc_ids, doc_embeddings = encode_and_cache_corpus(corpus, encoder, config["embedding_cache"])
    doc_embeddings = F.normalize(doc_embeddings.float(), p=2, dim=1).to('cuda')

    q_prefix = config["query_prefix"]
    K = args.candidate_depth

    # Build query lists
    q_base_list = []
    q_pos_list = []
    q_neg_list = []
    has_neg_mask = []

    for qid in eval_qids:
        q_base_list.append(queries[qid])
        d = dual_data.get(qid, {})
        q_plus = d.get('q_plus', '')
        q_minus = d.get('q_minus', '')

        if not q_plus or is_none_query(q_plus):
            q_plus = queries[qid]

        if not q_minus or is_none_query(q_minus):
            q_minus = ""
            has_neg_mask.append(0.0)
        else:
            has_neg_mask.append(1.0)

        q_pos_list.append(q_plus)
        q_neg_list.append(q_minus)

    logger.info(f"Q_minus available: {int(sum(has_neg_mask))}/{len(has_neg_mask)}")

    # Encode queries
    logger.info("Encoding Q_base...")
    q_base_prefixed = [q_prefix + q for q in q_base_list] if q_prefix else q_base_list
    q_base_emb = F.normalize(encoder.encode_queries(q_base_prefixed, batch_size=enc_bs).float(), p=2, dim=1)

    logger.info("Encoding Q_pos (Q+)...")
    q_pos_prefixed = [q_prefix + q for q in q_pos_list] if q_prefix else q_pos_list
    q_pos_emb = F.normalize(encoder.encode_queries(q_pos_prefixed, batch_size=enc_bs).float(), p=2, dim=1)

    logger.info("Encoding Q_neg (Q-)...")
    # Replace empty q_neg with a dummy string for encoding
    q_neg_for_enc = [q if q else "none dummy query" for q in q_neg_list]
    q_neg_prefixed = [q_prefix + q for q in q_neg_for_enc] if q_prefix else q_neg_for_enc
    q_neg_emb = F.normalize(encoder.encode_queries(q_neg_prefixed, batch_size=enc_bs).float(), p=2, dim=1)

    # Compute full score matrix
    logger.info("Computing S_base for all queries...")
    S_base_full = torch.matmul(q_base_emb.to('cuda'), doc_embeddings.T).float()

    # Get top-K candidates per query
    top_k_indices = torch.zeros(len(eval_qids), K, dtype=torch.long)
    top_k_scores = torch.zeros(len(eval_qids), K)
    for i in range(len(eval_qids)):
        scores = S_base_full[i]
        k = min(K, len(scores))
        topk = torch.topk(scores, k)
        top_k_indices[i, :k] = topk.indices
        top_k_scores[i, :k] = topk.values

    # Compute S_pos and S_neg for top-K candidates
    logger.info("Computing S_pos and S_neg for top-K candidates...")
    S_pos_topk = torch.zeros(len(eval_qids), K)
    S_neg_topk = torch.zeros(len(eval_qids), K)

    for i in tqdm(range(len(eval_qids)), desc="S_pos/S_neg"):
        indices = top_k_indices[i]
        valid_mask = indices >= 0
        if valid_mask.sum() == 0:
            continue
        valid_indices = indices[valid_mask].to('cuda')

        doc_emb_selected = doc_embeddings[valid_indices]
        s_pos = torch.matmul(q_pos_emb[i].unsqueeze(0).to('cuda'), doc_emb_selected.T).squeeze(0)
        S_pos_topk[i, valid_mask] = s_pos.float().cpu()

        if has_neg_mask[i] > 0:
            s_neg = torch.matmul(q_neg_emb[i].unsqueeze(0).to('cuda'), doc_emb_selected.T).squeeze(0)
            S_neg_topk[i, valid_mask] = s_neg.float().cpu()

    # Apply TRACE scoring
    logger.info(f"Applying TRACE scoring (ablation={args.ablation})...")
    S_final_topk = torch.zeros(len(eval_qids), K)

    for i in range(len(eval_qids)):
        valid_mask = top_k_indices[i] >= 0
        if valid_mask.sum() == 0:
            continue
        k = valid_mask.sum().item()
        s_b = S_base_full[i, top_k_indices[i][:k]].cpu()
        s_p = S_pos_topk[i, :k]
        s_n = S_neg_topk[i, :k]
        has_neg = bool(has_neg_mask[i] > 0)

        s_final = apply_trace_scoring(
            s_base=s_b, s_pos=s_p, s_neg=s_n, has_neg=has_neg,
            lambda_boundary=args.lambda_boundary, tau_decay=args.tau_decay,
            huber_delta=args.huber_delta,
            ablation=args.ablation,
        )
        S_final_topk[i, :k] = s_final

    # Build run dict and evaluate
    run_trace = {}
    for i, qid in enumerate(eval_qids):
        run_trace[qid] = {}
        for j in range(K):
            if top_k_indices[i, j] < 0:
                break
            did = doc_ids[top_k_indices[i, j].item()]
            run_trace[qid][did] = S_final_topk[i, j].item()

    trace_metrics, _ = evaluate(run_trace, qrels, eval_qids)

    # Also compute baseline metrics
    run_baseline = {}
    for i, qid in enumerate(eval_qids):
        run_baseline[qid] = {}
        for j in range(K):
            if top_k_indices[i, j] < 0:
                break
            did = doc_ids[top_k_indices[i, j].item()]
            run_baseline[qid][did] = top_k_scores[i, j].item()

    baseline_metrics, _ = evaluate(run_baseline, qrels, eval_qids)

    # Print results
    logger.info("Results:")
    logger.info(f"  Baseline: MAP={baseline_metrics['map_cut_100']:.4f}, nDCG@10={baseline_metrics['ndcg_cut_10']:.4f}")
    logger.info(f"  TRACE ({args.ablation}): MAP={trace_metrics['map_cut_100']:.4f}, nDCG@10={trace_metrics['ndcg_cut_10']:.4f}")
    logger.info(f"  Delta: MAP={trace_metrics['map_cut_100']-baseline_metrics['map_cut_100']:+.4f}, nDCG@10={trace_metrics['ndcg_cut_10']-baseline_metrics['ndcg_cut_10']:+.4f}")

    # Save results
    output = {
        "encoder": encoder_name,
        "model_name": config["model_name"],
        "ablation": args.ablation,
        "lambda_boundary": args.lambda_boundary,
        "tau_decay": args.tau_decay,
        "huber_delta": args.huber_delta,
        "candidate_depth": K,
        "baseline": baseline_metrics,
        "trace": trace_metrics,
        "delta_map": trace_metrics['map_cut_100'] - baseline_metrics['map_cut_100'],
        "delta_ndcg10": trace_metrics['ndcg_cut_10'] - baseline_metrics['ndcg_cut_10'],
        "q_minus_rate": f"{int(sum(has_neg_mask))}/{len(has_neg_mask)}",
    }

    output_dir = args.output_dir or f"results/negconstraint_trace/{encoder_name}/{args.ablation}"
    output_path = os.path.join(output_dir, "metrics_summary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print(f"TRACE NegConstraint EVALUATION SUMMARY ({encoder_name}, ablation={args.ablation})")
    print("=" * 80)
    print(f"{'Method':<35} {'MAP@100':>10} {'nDCG@10':>10}")
    print("-" * 80)
    print(f"{'Baseline':<35} {baseline_metrics['map_cut_100']:>10.4f} {baseline_metrics['ndcg_cut_10']:>10.4f}")
    print(f"{f'TRACE ({args.ablation})':<35} {trace_metrics['map_cut_100']:>10.4f} {trace_metrics['ndcg_cut_10']:>10.4f}")
    print(f"{'Delta':<35} {trace_metrics['map_cut_100']-baseline_metrics['map_cut_100']:>+10.4f} {trace_metrics['ndcg_cut_10']-baseline_metrics['ndcg_cut_10']:>+10.4f}")
    print("-" * 80)

    del encoder
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
