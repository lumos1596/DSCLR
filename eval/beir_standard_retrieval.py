"""
TRACE Standard Retrieval Preservation on BEIR

Tests whether TRACE's no-exclusion branch perturbs ordinary retrieval.
Conditions:
  1. Base query: Original query text (no instruction)
  2. + Neutral instruction: Query + neutral task instruction
  3. + TRACE: TRACE scoring with q_minus=[NONE] (no exclusion)

Datasets: TREC-COVID, NFCorpus, FiQA-2018, ArguAna, SciFact, Quora
Metrics: nDCG@10, MAP@100

Usage:
  cd /home/luwa/Documents/DSCLR-remote && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/luwa/.conda/envs/dsclr/bin/python -m eval.beir_standard_retrieval \
      --model_name samaya-ai/RepLLaMA-reproduced \
      --dual_queries_path dataset/BEIR/dual_queries/trec-covid_CONSERVATIVE_t01.jsonl \
      --output_dir results/beir_standard/trec-covid_repllama
"""

import os, sys, json, argparse, time, logging
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force online mode for BEIR datasets
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ.pop("HF_ENDPOINT", None)

try:
    import huggingface_hub.constants as _hf_const
    _hf_const.HF_HUB_OFFLINE = False
except Exception:
    pass

import datasets
try:
    datasets.config.HF_DATASETS_OFFLINE = False
except Exception:
    pass

import pytrec_eval
from collections import defaultdict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from eval.models.encoder import ModelFactory
from eval.engine_trace import TRACEEvaluator, robust_standardize, _mad, fit_huber_regression

BEIR_DATASET_MAP = {
    "trec-covid": "BeIR/trec-covid",
    "nfcorpus": "BeIR/nfcorpus",
    "fiqa": "BeIR/fiqa",
    "arguana": "BeIR/arguana",
    "scifact": "BeIR/scifact",
    "quora": "BeIR/quora",
}

NEUTRAL_INSTRUCTION_MAP = {
    "trec-covid": "Find relevant scientific articles about COVID-19.",
    "nfcorpus": "Find relevant nutrition articles.",
    "fiqa": "Find relevant financial Q&A.",
    "arguana": "Find relevant arguments.",
    "scifact": "Find relevant scientific facts.",
    "quora": "Find relevant similar questions.",
}


def load_beir_dataset(dataset_name):
    """Load BEIR dataset from HuggingFace."""
    hf_name = BEIR_DATASET_MAP.get(dataset_name, dataset_name)

    from datasets import load_dataset

    corpus = {}
    queries = {}
    qrels = {}

    # Load each split separately to avoid configuration ambiguity
    logger.info(f"Loading corpus from {hf_name}...")
    corpus_ds = load_dataset(hf_name, "corpus")
    for doc in corpus_ds['corpus']:
        text = doc.get('text', '') or ''
        title = doc.get('title', '') or ''
        corpus[str(doc['_id'])] = f"{title} {text}".strip() if title else text

    logger.info(f"Loading queries from {hf_name}...")
    queries_ds = load_dataset(hf_name, "queries")
    for q in queries_ds['queries']:
        queries[str(q['_id'])] = q.get('text', '')

    logger.info(f"Loading qrels from {hf_name}...")
    # BEIR qrels are stored in a separate dataset: BeIR/<name>-qrels
    qrels_hf_name = hf_name + "-qrels"
    try:
        qrels_ds = load_dataset(qrels_hf_name)
        # Find the split that contains the test data
        split_name = None
        for s in ['test', 'train', 'validation']:
            if s in qrels_ds:
                split_name = s
                break
        if split_name is None:
            split_name = list(qrels_ds.keys())[0]
        for item in qrels_ds[split_name]:
            qid = str(item['query-id'])
            did = str(item['corpus-id'])
            score = int(item['score'])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = score
    except Exception as e:
        logger.error(f"Failed to load qrels from {qrels_hf_name}: {e}")

    return corpus, queries, qrels


def load_dual_queries(dual_path):
    """Load dual queries for BEIR datasets."""
    dual_data = {}
    if dual_path and os.path.exists(dual_path):
        with open(dual_path) as f:
            for line in f:
                item = json.loads(line)
                dual_data[item['qid']] = item
    return dual_data


def is_none_query(q):
    return not q or q.strip().lower() in ['none', '[none]', 'n/a', 'null', '', '[NONE]']


def compute_scores_and_evaluate(q_embs, doc_embs, eval_qids, doc_ids, qrels, K=100):
    """Compute top-K scores and evaluate with pytrec_eval."""
    # For large corpora, compute similarity in batches to avoid OOM
    n_docs = doc_embs.shape[0]
    n_queries = q_embs.shape[0]
    if n_docs > 100000:
        # Batched computation with running top-K to avoid accumulating all scores
        batch_size = 50000
        topk_scores = torch.full((n_queries, K), float('-inf'), device='cpu')
        topk_indices = torch.full((n_queries, K), -1, dtype=torch.long, device='cpu')
        for start in range(0, n_docs, batch_size):
            end = min(start + batch_size, n_docs)
            doc_batch = doc_embs[start:end].to('cuda')
            q_batch = q_embs.to('cuda')
            S_batch = torch.matmul(q_batch, doc_batch.T).float().cpu()
            del doc_batch
            torch.cuda.empty_cache()
            # Get local top-K for this batch
            local_k = min(K, S_batch.shape[1])
            local_scores, local_idx = torch.topk(S_batch, k=local_k, dim=1)
            local_indices = local_idx + start
            del S_batch
            # Merge with running top-K
            merged_scores = torch.cat([topk_scores, local_scores], dim=1)
            merged_indices = torch.cat([topk_indices, local_indices], dim=1)
            topk_scores, merge_idx = torch.topk(merged_scores, k=K, dim=1)
            topk_indices = torch.gather(merged_indices, 1, merge_idx)
            del merged_scores, merged_indices, local_scores, local_idx, local_indices
    else:
        S = torch.matmul(q_embs.to('cuda'), doc_embs.to('cuda').T).float()
        topk_scores, topk_indices = torch.topk(S, k=min(K, S.shape[1]), dim=1)

    run_data = {}
    for i, qid in enumerate(eval_qids):
        run_data[qid] = {}
        for j in range(topk_scores.shape[1]):
            did = doc_ids[topk_indices[i, j].item()]
            run_data[qid][did] = topk_scores[i, j].item()
    
    evaluator = pytrec_eval.RelevanceEvaluator(
        {qid: qrels[qid] for qid in eval_qids if qid in qrels},
        {'ndcg_cut_10', 'map_cut_100'}
    )
    results = evaluator.evaluate(run_data)
    metrics = {}
    for m in ['ndcg_cut_10', 'map_cut_100']:
        vals = [results[qid][m] for qid in eval_qids if qid in results]
        metrics[m] = np.mean(vals) if vals else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser(description="TRACE Standard Retrieval on BEIR")
    parser.add_argument("--dataset", type=str, required=True, choices=list(BEIR_DATASET_MAP.keys()))
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--dual_queries_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=100)
    args = parser.parse_args()

    logger.info(f"Loading BEIR dataset: {args.dataset}")
    corpus, queries, qrels = load_beir_dataset(args.dataset)
    eval_qids = sorted(set(qrels.keys()) & set(queries.keys()))
    logger.info(f"Corpus: {len(corpus)}, Queries: {len(queries)}, Eval: {len(eval_qids)}")

    # Load dual queries (for TRACE condition)
    dual_data = load_dual_queries(args.dual_queries_path)

    # Create encoder
    encoder = ModelFactory.create(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        normalize_embeddings=True
    )

    # Encode corpus
    doc_ids = sorted(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]

    # Try to load cached embeddings
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "dataset", "BEIR", "embeddings", args.dataset)
    doc_embs = None

    if "bge" in args.model_name.lower():
        cache_path = os.path.join(cache_dir, f"{args.dataset}_bge_large_en_v1.5_corpus.pt")
        if os.path.exists(cache_path):
            logger.info(f"Loading cached BGE embeddings from {cache_path}...")
            cached = torch.load(cache_path, map_location='cpu')
            if isinstance(cached, dict) and 'embeddings' in cached:
                doc_embs = F.normalize(cached['embeddings'].float(), p=2, dim=1)
                if 'doc_ids' in cached:
                    doc_ids = [str(d) for d in cached['doc_ids']]
            elif isinstance(cached, torch.Tensor):
                doc_embs = F.normalize(cached.float(), p=2, dim=1)
            logger.info(f"Loaded cached embeddings: {doc_embs.shape}")

    if doc_embs is None:
        logger.info(f"Encoding {len(doc_texts)} documents...")
        doc_embs = F.normalize(encoder.encode_documents(doc_texts, batch_size=args.batch_size).float(), p=2, dim=1).to('cuda')

    # Determine query prefix based on model
    q_prefix = ""
    if "bge" in args.model_name.lower():
        q_prefix = "Represent this sentence for searching relevant passages: "
    elif "e5" in args.model_name.lower() and "mistral" in args.model_name.lower():
        q_prefix = "Instruct: "
    elif "gritlm" in args.model_name.lower():
        q_prefix = ""

    # Condition 1: Base query (original text, no instruction)
    logger.info("Condition 1: Base query")
    q_base_list = [queries[qid] for qid in eval_qids]
    q_base_prefixed = [q_prefix + q for q in q_base_list] if q_prefix else q_base_list
    q_base_embs = F.normalize(encoder.encode_queries(q_base_prefixed, batch_size=args.batch_size).float(), p=2, dim=1)
    base_metrics = compute_scores_and_evaluate(q_base_embs, doc_embs, eval_qids, doc_ids, qrels, args.top_k)
    logger.info(f"Base: nDCG@10={base_metrics['ndcg_cut_10']:.4f}, MAP@100={base_metrics['map_cut_100']:.4f}")

    # Condition 2: Base query + neutral instruction
    logger.info("Condition 2: + Neutral instruction")
    neutral_instr = NEUTRAL_INSTRUCTION_MAP.get(args.dataset, "")
    q_neutral_list = [f"{queries[qid]} {neutral_instr}".strip() for qid in eval_qids]
    q_neutral_prefixed = [q_prefix + q for q in q_neutral_list] if q_prefix else q_neutral_list
    q_neutral_embs = F.normalize(encoder.encode_queries(q_neutral_prefixed, batch_size=args.batch_size).float(), p=2, dim=1)
    neutral_metrics = compute_scores_and_evaluate(q_neutral_embs, doc_embs, eval_qids, doc_ids, qrels, args.top_k)
    logger.info(f"Neutral: nDCG@10={neutral_metrics['ndcg_cut_10']:.4f}, MAP@100={neutral_metrics['map_cut_100']:.4f}")

    # Condition 3: + TRACE (with q_minus=[NONE], using TRACE scoring)
    logger.info("Condition 3: + TRACE (no exclusion)")
    # For TRACE, we use the neutral instruction as q_base, q_plus from dual data, and q_minus=""
    # If dual data exists, use q_plus; otherwise use neutral query as q_plus
    q_pos_list = []
    for qid in eval_qids:
        d = dual_data.get(qid, {})
        q_plus = d.get('q_plus', '')
        if not q_plus or is_none_query(q_plus):
            q_plus = q_neutral_list[eval_qids.index(qid)] if qid in eval_qids else ""
        q_pos_list.append(q_plus)
    
    q_pos_prefixed = [q_prefix + q for q in q_pos_list] if q_prefix else q_pos_list
    q_pos_embs = F.normalize(encoder.encode_queries(q_pos_prefixed, batch_size=args.batch_size).float(), p=2, dim=1)

    # q_minus is [NONE] for all queries → no exclusion
    # Compute TRACE scores with batched approach for large corpora
    n_docs = doc_embs.shape[0]
    n_queries = len(eval_qids)
    K = args.top_k

    if n_docs > 100000:
        # Batched TRACE scoring for large corpora
        # We need top-K indices from neutral scores first
        topk_indices_base = torch.full((n_queries, K), -1, dtype=torch.long, device='cpu')
        topk_scores_neutral = torch.full((n_queries, K), float('-inf'), device='cpu')
        topk_scores_pos = torch.full((n_queries, K), float('-inf'), device='cpu')

        doc_batch_size = 50000
        for start in range(0, n_docs, doc_batch_size):
            end = min(start + doc_batch_size, n_docs)
            doc_batch = doc_embs[start:end].to('cuda')
            q_neutral_cuda = q_neutral_embs.to('cuda')
            q_pos_cuda = q_pos_embs.to('cuda')

            S_full_batch = torch.matmul(q_neutral_cuda, doc_batch.T).float().cpu()
            S_pos_batch = torch.matmul(q_pos_cuda, doc_batch.T).float().cpu()
            del doc_batch
            torch.cuda.empty_cache()

            # For each query, we need top-K from neutral scores AND corresponding pos scores
            # Strategy: find local top-K by neutral, merge with running top-K
            local_k = min(K, S_full_batch.shape[1])
            local_neutral_scores, local_idx = torch.topk(S_full_batch, k=local_k, dim=1)
            local_pos_scores = torch.gather(S_pos_batch, 1, local_idx)
            local_indices = local_idx + start

            del S_full_batch, S_pos_batch

            # Merge neutral scores with running top-K
            merged_neutral = torch.cat([topk_scores_neutral, local_neutral_scores], dim=1)
            merged_pos = torch.cat([topk_scores_pos, local_pos_scores], dim=1)
            merged_indices = torch.cat([topk_indices_base, local_indices], dim=1)

            merge_topk_vals, merge_topk_idx = torch.topk(merged_neutral, k=K, dim=1)
            topk_scores_neutral = merge_topk_vals
            topk_indices_base = torch.gather(merged_indices, 1, merge_topk_idx)
            topk_scores_pos = torch.gather(merged_pos, 1, merge_topk_idx)

            del merged_neutral, merged_pos, merged_indices, local_neutral_scores, local_pos_scores, local_indices

        # Now apply TRACE scoring on the top-K results
        run_trace = {}
        for i, qid in enumerate(eval_qids):
            s_f = topk_scores_neutral[i]
            s_p = topk_scores_pos[i]
            z_full = robust_standardize(s_f.float(), 1e-6)
            z_pos = robust_standardize(s_p.float(), 1e-6)
            p = torch.clamp(z_pos, min=0)
            s_final = z_full + p
            run_trace[qid] = {}
            for j in range(K):
                did = doc_ids[topk_indices_base[i, j].item()]
                run_trace[qid][did] = s_final[j].item()
    else:
        doc_embs_cuda = doc_embs.to('cuda')
        S_full = torch.matmul(q_neutral_embs.to('cuda'), doc_embs_cuda.T).float()
        S_pos = torch.matmul(q_pos_embs.to('cuda'), doc_embs_cuda.T).float()
        del doc_embs_cuda
        torch.cuda.empty_cache()

        S_final = S_full.clone()
        topk_indices_base = torch.topk(S_full, k=min(K, S_full.shape[1]), dim=1).indices

        for i, qid in enumerate(eval_qids):
            indices = topk_indices_base[i]
            s_f = S_full[i, indices]
            s_p = S_pos[i, indices]
            z_full = robust_standardize(s_f.float(), 1e-6)
            z_pos = robust_standardize(s_p.float(), 1e-6)
            p = torch.clamp(z_pos, min=0)
            s_final = z_full + p
            S_final[i, indices] = s_final.to(dtype=S_final.dtype)

        topk_scores_final, topk_indices_final = torch.topk(S_final, k=min(K, S_final.shape[1]), dim=1)
        run_trace = {}
        for i, qid in enumerate(eval_qids):
            run_trace[qid] = {}
            for j in range(topk_scores_final.shape[1]):
                did = doc_ids[topk_indices_final[i, j].item()]
                run_trace[qid][did] = topk_scores_final[i, j].item()

    evaluator = pytrec_eval.RelevanceEvaluator(
        {qid: qrels[qid] for qid in eval_qids if qid in qrels},
        {'ndcg_cut_10', 'map_cut_100'}
    )
    trace_results = evaluator.evaluate(run_trace)
    trace_metrics = {}
    for m in ['ndcg_cut_10', 'map_cut_100']:
        vals = [trace_results[qid][m] for qid in eval_qids if qid in trace_results]
        trace_metrics[m] = np.mean(vals) if vals else 0.0
    logger.info(f"TRACE: nDCG@10={trace_metrics['ndcg_cut_10']:.4f}, MAP@100={trace_metrics['map_cut_100']:.4f}")

    # Save results
    output = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "base_query": base_metrics,
        "neutral_instruction": neutral_metrics,
        "trace": trace_metrics,
        "delta_neutral_vs_base": {
            "ndcg_cut_10": neutral_metrics['ndcg_cut_10'] - base_metrics['ndcg_cut_10'],
            "map_cut_100": neutral_metrics['map_cut_100'] - base_metrics['map_cut_100'],
        },
        "delta_trace_vs_base": {
            "ndcg_cut_10": trace_metrics['ndcg_cut_10'] - base_metrics['ndcg_cut_10'],
            "map_cut_100": trace_metrics['map_cut_100'] - base_metrics['map_cut_100'],
        },
        "delta_trace_vs_neutral": {
            "ndcg_cut_10": trace_metrics['ndcg_cut_10'] - neutral_metrics['ndcg_cut_10'],
            "map_cut_100": trace_metrics['map_cut_100'] - neutral_metrics['map_cut_100'],
        },
    }

    output_dir = args.output_dir or f"results/beir_standard/{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "metrics_summary.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print(f"BEIR Standard Retrieval ({args.dataset}, {args.model_name})")
    print("=" * 80)
    print(f"{'Condition':<35} {'nDCG@10':>10} {'MAP@100':>10}")
    print("-" * 80)
    print(f"{'Base query':<35} {base_metrics['ndcg_cut_10']:>10.4f} {base_metrics['map_cut_100']:>10.4f}")
    print(f"{'+ Neutral instruction':<35} {neutral_metrics['ndcg_cut_10']:>10.4f} {neutral_metrics['map_cut_100']:>10.4f}")
    print(f"{'+ TRACE (no exclusion)':<35} {trace_metrics['ndcg_cut_10']:>10.4f} {trace_metrics['map_cut_100']:>10.4f}")
    print("-" * 80)
    print(f"{'Δ Neutral vs Base':<35} {neutral_metrics['ndcg_cut_10']-base_metrics['ndcg_cut_10']:>+10.4f} {neutral_metrics['map_cut_100']-base_metrics['map_cut_100']:>+10.4f}")
    print(f"{'Δ TRACE vs Base':<35} {trace_metrics['ndcg_cut_10']-base_metrics['ndcg_cut_10']:>+10.4f} {trace_metrics['map_cut_100']-base_metrics['map_cut_100']:>+10.4f}")
    print(f"{'Δ TRACE vs Neutral':<35} {trace_metrics['ndcg_cut_10']-neutral_metrics['ndcg_cut_10']:>+10.4f} {trace_metrics['map_cut_100']-neutral_metrics['map_cut_100']:>+10.4f}")

    del encoder
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
