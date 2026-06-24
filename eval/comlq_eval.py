"""
ComLQ Evaluation: Baseline vs DeIR-Dual V2

ComLQ (Complex Logical Queries) benchmark.
14 query types: 9 without negation (1p/2p/3p/2i/3i/pi/ip/2u/up) + 5 with negation (2in/3in/inp/pin/pni)

Usage:
  # Baseline only
  cd /home/luwa/Documents/DSCLR && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/luwa/.conda/envs/dsclr/bin/python -m eval.comlq_eval --encoder bge

  # With dual queries (default params)
  cd /home/luwa/Documents/DSCLR && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/luwa/.conda/envs/dsclr/bin/python -m eval.comlq_eval --encoder bge --dual_queries

  # Custom params
  cd /home/luwa/Documents/DSCLR && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/luwa/.conda/envs/dsclr/bin/python -m eval.comlq_eval --encoder bge --dual_queries --alpha 0.3 --beta 0.5 --delta 0.10

  # Grid search
  cd /home/luwa/Documents/DSCLR && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/luwa/.conda/envs/dsclr/bin/python -m eval.comlq_eval --encoder bge --dual_queries --grid_search
"""

import os, sys, json, csv, argparse, logging, math
import numpy as np, torch, torch.nn.functional as F

torch.cuda._lazy_init()

import pytrec_eval
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.models.encoder import SentenceTransformerEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "dataset/ComLQ/dataset"
DUAL_QUERIES_DIR = "dataset/ComLQ/dual_queries"

DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 1.5
DEFAULT_DELTA = 0.05
DEFAULT_T_SAFETY = 20.0
DEFAULT_TOP_K = 100

NEGATION_TYPES = {"2in", "3in", "inp", "pin", "pni"}

ENCODER_CONFIGS = {
    "bge": {
        "class": "sentence_transformer",
        "model_name": "BAAI/bge-large-en-v1.5",
        "embedding_cache": "dataset/ComLQ/embeddings/comlq_bge_corpus.pt",
        "results_path": "results/comlq/comlq_bge_results.json",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "",
    },
    "repllama": {
        "class": "repllama",
        "model_name": "samaya-ai/RepLLaMA-reproduced",
        "embedding_cache": "dataset/ComLQ/embeddings/comlq_repllama_corpus.pt",
        "results_path": "results/comlq/comlq_repllama_results.json",
        "query_prefix": "",
        "doc_prefix": "",
    },
}


def get_base_type(qtype):
    return qtype.split("_")[0]


def is_negation_type(qtype):
    return get_base_type(qtype) in NEGATION_TYPES


def load_data():
    corpus = {}
    with open(os.path.join(DATA_DIR, "corpus.jsonl"), encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            corpus[str(doc['_id'])] = doc.get('text', '')

    queries = {}
    query_types = {}
    with open(os.path.join(DATA_DIR, "queries.jsonl"), encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            qid = str(q['_id'])
            queries[qid] = q.get('text', '')
            query_types[qid] = q.get('type', '1p')

    qrels = {}
    with open(os.path.join(DATA_DIR, "qrels", "test.tsv"), encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            qid, did, score = str(row[0]), str(row[1]), int(row[2])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = score

    return corpus, queries, query_types, qrels


def load_dual_queries():
    dual_data = {}
    for fname in os.listdir(DUAL_QUERIES_DIR):
        if fname.endswith('.jsonl'):
            fpath = os.path.join(DUAL_QUERIES_DIR, fname)
            with open(fpath) as f:
                for line in f:
                    r = json.loads(line)
                    dual_data[r['qid']] = r
    return dual_data


def create_encoder(config):
    if config["class"] == "repllama":
        from eval.models.repllama_encoder import RepLLaMAEncoder
        return RepLLaMAEncoder(
            model_name=config["model_name"],
            device="cuda", batch_size=64
        )
    else:
        return SentenceTransformerEncoder(
            model_name=config["model_name"],
            device="cuda", batch_size=256,
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
    doc_embeddings = encoder.encode_documents(doc_texts, batch_size=256 if encoder.batch_size >= 256 else 64)
    if doc_embeddings.dim() == 2:
        doc_embeddings = F.normalize(doc_embeddings.float(), p=2, dim=1)

    torch.save({
        "doc_ids": doc_ids,
        "embeddings": doc_embeddings.cpu().float(),
    }, cache_path)
    logger.info(f"Cached corpus embeddings to {cache_path}")
    return doc_ids, doc_embeddings


def is_none_query(q):
    return not q or q.strip().lower() in ['none', '[none]', 'n/a', 'null', '']


def score_deir_dual_v2(s_base, s_req, s_neg, cos_qbase_qneg, has_req, has_neg,
                       alpha, beta, delta, t_safety):
    if not has_neg:
        s_req_eff = s_req if has_req else torch.zeros_like(s_base)
        return s_base + beta * s_req_eff
    tau = cos_qbase_qneg + delta
    overflow = s_neg - tau
    smooth_penalty = F.softplus(overflow)
    raw_penalty = alpha * smooth_penalty
    safety = 1.0 - torch.sigmoid((s_neg - tau) * t_safety)
    s_req_eff = s_req if has_req else torch.zeros_like(s_base)
    return s_base + beta * s_req_eff * safety - raw_penalty


def evaluate(run_data, qrels, eval_qids):
    evaluator = pytrec_eval.RelevanceEvaluator(
        {qid: qrels[qid] for qid in eval_qids if qid in qrels},
        {'ndcg_cut_5', 'ndcg_cut_10', 'map_cut_100', 'recall_5', 'recall_10', 'recall_100'}
    )
    results = evaluator.evaluate(run_data)
    metrics = {}
    for m in ['ndcg_cut_5', 'ndcg_cut_10', 'map_cut_100', 'recall_5', 'recall_10', 'recall_100']:
        vals = [results[qid][m] for qid in eval_qids if qid in results]
        metrics[m] = np.mean(vals) if vals else 0.0
    return metrics, results


def compute_lsnc(run_data, qrels, query_types, eval_qids, K=100):
    neg_qids = [qid for qid in eval_qids if is_negation_type(query_types.get(qid, ''))]
    if not neg_qids:
        return 0.0

    lsnc_vals = []
    for qid in neg_qids:
        if qid not in run_data or qid not in qrels:
            continue

        sorted_docs = sorted(run_data[qid].items(), key=lambda x: x[1], reverse=True)
        top_k_docs = sorted_docs[:K]

        relevant_docs = {did for did, score in qrels[qid].items() if score > 0}

        neg_violations = 0
        for did, _ in top_k_docs:
            if did not in relevant_docs:
                neg_violations += 1

        total = len(top_k_docs)
        if total > 0:
            consistency = 1.0 - (neg_violations / total)
            lsnc_val = math.log(1 + consistency) / math.log(2)
            lsnc_vals.append(lsnc_val)

    return np.mean(lsnc_vals) if lsnc_vals else 0.0


def apply_scoring_and_evaluate(eval_qids, doc_ids, top_k_indices, S_base_full,
                                S_req_topk, S_neg_topk, cos_qbase_qneg,
                                has_req_mask, has_neg_mask,
                                qrels, query_types, neg_qids, nonneg_qids,
                                alpha, beta, delta, t_safety, top_k):
    S_final_topk = torch.zeros(len(eval_qids), top_k)
    for i in range(len(eval_qids)):
        valid_mask = top_k_indices[i] >= 0
        if valid_mask.sum() == 0:
            continue
        k = valid_mask.sum().item()
        s_b = S_base_full[i, top_k_indices[i][:k]]
        s_r = S_req_topk[i, :k]
        s_n = S_neg_topk[i, :k]

        s_final = score_deir_dual_v2(
            s_base=s_b, s_req=s_r, s_neg=s_n,
            cos_qbase_qneg=float(cos_qbase_qneg[i].item()),
            has_req=bool(has_req_mask[i] > 0),
            has_neg=bool(has_neg_mask[i] > 0),
            alpha=alpha, beta=beta, delta=delta, t_safety=t_safety,
        )
        S_final_topk[i, :k] = s_final

    run_deir = {}
    for i, qid in enumerate(eval_qids):
        run_deir[qid] = {}
        for j in range(top_k):
            if top_k_indices[i, j] < 0:
                break
            did = doc_ids[top_k_indices[i, j].item()]
            run_deir[qid][did] = S_final_topk[i, j].item()

    all_metrics, _ = evaluate(run_deir, qrels, eval_qids)
    neg_metrics, _ = evaluate(run_deir, qrels, neg_qids)
    nonneg_metrics, _ = evaluate(run_deir, qrels, nonneg_qids)
    lsnc = compute_lsnc(run_deir, qrels, query_types, eval_qids, K=100)
    neg_lsnc = compute_lsnc(run_deir, qrels, query_types, neg_qids, K=100)

    return all_metrics, neg_metrics, nonneg_metrics, lsnc, neg_lsnc, run_deir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="bge", choices=["bge", "repllama"])
    parser.add_argument("--dual_queries", action="store_true")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--t_safety", type=float, default=DEFAULT_T_SAFETY)
    parser.add_argument("--grid_search", action="store_true")
    args = parser.parse_args()

    config = ENCODER_CONFIGS[args.encoder]
    encoder_name = args.encoder
    logger.info("=" * 60)
    logger.info(f"ComLQ Evaluation: {encoder_name} encoder")
    logger.info("=" * 60)

    corpus, queries, query_types, qrels = load_data()
    eval_qids = sorted(set(qrels.keys()) & set(queries.keys()))
    logger.info(f"Corpus: {len(corpus)} docs, Queries: {len(queries)}, Eval queries: {len(eval_qids)}")

    neg_qids = [qid for qid in eval_qids if is_negation_type(query_types.get(qid, ''))]
    nonneg_qids = [qid for qid in eval_qids if not is_negation_type(query_types.get(qid, ''))]
    logger.info(f"Negation queries: {len(neg_qids)}, Non-negation queries: {len(nonneg_qids)}")

    type_counts = defaultdict(int)
    for qid in eval_qids:
        type_counts[get_base_type(query_types.get(qid, ''))] += 1
    logger.info(f"Query type distribution: {dict(sorted(type_counts.items()))}")

    dual_data = {}
    if args.dual_queries:
        dual_data = load_dual_queries()
        logger.info(f"Dual queries loaded: {len(dual_data)}")

    encoder = create_encoder(config)
    doc_ids, doc_embeddings = encode_and_cache_corpus(corpus, encoder, config["embedding_cache"])
    doc_embeddings = F.normalize(doc_embeddings.float(), p=2, dim=1).to('cuda')

    q_prefix = config["query_prefix"]

    # ============================================================
    # Baseline
    # ============================================================
    logger.info("\n--- Baseline: Original Query ---")
    q_base_list = [queries[qid] for qid in eval_qids]
    q_base_prefixed = [q_prefix + q for q in q_base_list] if q_prefix else q_base_list
    q_base_emb = F.normalize(encoder.encode_queries(q_base_prefixed, batch_size=256).float(), p=2, dim=1)

    S_base = torch.matmul(q_base_emb.to('cuda'), doc_embeddings.T).float().cpu()
    topk_scores, topk_indices = torch.topk(S_base, k=min(args.top_k, S_base.shape[1]), dim=1)

    run_baseline = {}
    for i, qid in enumerate(eval_qids):
        run_baseline[qid] = {}
        for j in range(topk_scores.shape[1]):
            did = doc_ids[topk_indices[i, j].item()]
            run_baseline[qid][did] = topk_scores[i, j].item()

    baseline_metrics, baseline_per_query = evaluate(run_baseline, qrels, eval_qids)
    baseline_lsnc = compute_lsnc(run_baseline, qrels, query_types, eval_qids, K=100)
    logger.info("Baseline Results (All):")
    for m, v in sorted(baseline_metrics.items()):
        logger.info(f"  {m}: {v:.4f}")
    logger.info(f"  LSNC@100: {baseline_lsnc:.4f}")

    baseline_neg_metrics, _ = evaluate(run_baseline, qrels, neg_qids)
    baseline_neg_lsnc = compute_lsnc(run_baseline, qrels, query_types, neg_qids, K=100)
    logger.info("Baseline Results (Negation only):")
    for m, v in sorted(baseline_neg_metrics.items()):
        logger.info(f"  {m}: {v:.4f}")
    logger.info(f"  LSNC@100: {baseline_neg_lsnc:.4f}")

    baseline_nonneg_metrics, _ = evaluate(run_baseline, qrels, nonneg_qids)
    logger.info("Baseline Results (Non-negation only):")
    for m, v in sorted(baseline_nonneg_metrics.items()):
        logger.info(f"  {m}: {v:.4f}")

    baseline_type_metrics = {}
    for qtype in sorted(type_counts.keys()):
        type_qids = [qid for qid in eval_qids if get_base_type(query_types.get(qid, '')) == qtype]
        if len(type_qids) >= 2:
            type_metrics, _ = evaluate(run_baseline, qrels, type_qids)
            baseline_type_metrics[qtype] = type_metrics

    # ============================================================
    # DeIR-Dual V2
    # ============================================================
    if not args.dual_queries or not dual_data:
        logger.info("\nNo dual queries provided. Skipping DeIR-Dual V2 evaluation.")
        output = {
            "encoder": encoder_name,
            "model_name": config["model_name"],
            "baseline_all": baseline_metrics,
            "baseline_negation": baseline_neg_metrics,
            "baseline_nonnegation": baseline_nonneg_metrics,
            "baseline_lsnc100": baseline_lsnc,
            "baseline_negation_lsnc100": baseline_neg_lsnc,
            "baseline_per_type": baseline_type_metrics,
            "params": {"alpha": args.alpha, "beta": args.beta, "delta": args.delta, "t_safety": args.t_safety, "top_k": args.top_k},
        }
        output_path = config["results_path"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")
        return

    q_req_list = []
    q_neg_list = []
    has_req_mask = []
    has_neg_mask = []

    for qid in eval_qids:
        d = dual_data.get(qid, {})
        q_plus = d.get('q_plus', '')
        q_minus = d.get('q_minus', '')

        if not q_plus or is_none_query(q_plus):
            q_plus = queries[qid]
            has_req_mask.append(0.0)
        else:
            has_req_mask.append(1.0)

        if not q_minus or is_none_query(q_minus):
            q_minus = ""
            has_neg_mask.append(0.0)
        else:
            has_neg_mask.append(1.0)

        q_req_list.append(q_plus)
        q_neg_list.append(q_minus)

    logger.info(f"Q_plus available: {int(sum(has_req_mask))}/{len(has_req_mask)}")
    logger.info(f"Q_minus available: {int(sum(has_neg_mask))}/{len(has_neg_mask)}")

    neg_qid_set = set(neg_qids)
    neg_minus_count = sum(1 for i, qid in enumerate(eval_qids) if qid in neg_qid_set and has_neg_mask[i] > 0)
    logger.info(f"Q_minus in negation queries: {neg_minus_count}/{len(neg_qids)}")

    logger.info("Encoding Q_req (Q+)...")
    q_req_prefixed = [q_prefix + q for q in q_req_list] if q_prefix else q_req_list
    q_req_emb = F.normalize(encoder.encode_queries(q_req_prefixed, batch_size=256).float(), p=2, dim=1)

    logger.info("Encoding Q_neg (Q-)...")
    q_neg_prefixed = [q_prefix + q for q in q_neg_list] if q_prefix else q_neg_list
    q_neg_emb = F.normalize(encoder.encode_queries(q_neg_prefixed, batch_size=256).float(), p=2, dim=1)

    logger.info("Computing S_base for top-k candidates...")
    S_base_full = torch.matmul(q_base_emb.to('cuda'), doc_embeddings.T).float().cpu()

    top_k_indices = torch.zeros(len(eval_qids), args.top_k, dtype=torch.long)
    top_k_scores = torch.zeros(len(eval_qids), args.top_k)
    for i in range(len(eval_qids)):
        scores = S_base_full[i]
        k = min(args.top_k, len(scores))
        topk = torch.topk(scores, k)
        top_k_indices[i, :k] = topk.indices
        top_k_scores[i, :k] = topk.values

    logger.info("Computing S_req and S_neg for top-k candidates...")
    S_req_topk = torch.zeros(len(eval_qids), args.top_k)
    S_neg_topk = torch.zeros(len(eval_qids), args.top_k)
    cos_qbase_qneg = F.cosine_similarity(q_base_emb.cpu(), q_neg_emb.cpu(), dim=1)

    for i in tqdm(range(len(eval_qids)), desc="S_req/S_neg"):
        indices = top_k_indices[i]
        valid_mask = indices >= 0
        if valid_mask.sum() == 0:
            continue
        valid_indices = indices[valid_mask].to('cuda')

        doc_emb_selected = doc_embeddings[valid_indices]
        s_req = torch.matmul(q_req_emb[i].unsqueeze(0).to('cuda'), doc_emb_selected.T).squeeze(0)
        S_req_topk[i, valid_mask] = s_req.float().cpu()

        if has_neg_mask[i] > 0:
            s_neg = torch.matmul(q_neg_emb[i].unsqueeze(0).to('cuda'), doc_emb_selected.T).squeeze(0)
            S_neg_topk[i, valid_mask] = s_neg.float().cpu()

    del encoder
    torch.cuda.empty_cache()

    # ============================================================
    # Grid Search
    # ============================================================
    if args.grid_search:
        logger.info("\n" + "=" * 60)
        logger.info("GRID SEARCH for optimal parameters")
        logger.info("=" * 60)

        alphas = [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0]
        betas = [0.0, 0.3, 0.5, 0.8, 1.0, 1.3, 1.5]
        deltas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

        best_neg_ndcg = -1
        best_params = None
        best_results = None
        grid_results = []

        total_combos = len(alphas) * len(betas) * len(deltas)
        logger.info(f"Total combinations: {total_combos}")

        for alpha in alphas:
            for beta in betas:
                for delta in deltas:
                    all_m, neg_m, nonneg_m, lsnc, neg_lsnc, _ = apply_scoring_and_evaluate(
                        eval_qids, doc_ids, top_k_indices, S_base_full,
                        S_req_topk, S_neg_topk, cos_qbase_qneg,
                        has_req_mask, has_neg_mask,
                        qrels, query_types, neg_qids, nonneg_qids,
                        alpha, beta, delta, args.t_safety, args.top_k
                    )

                    neg_ndcg = neg_m['ndcg_cut_10']
                    all_ndcg = all_m['ndcg_cut_10']

                    result = {
                        "alpha": alpha, "beta": beta, "delta": delta,
                        "all_ndcg10": all_ndcg, "neg_ndcg10": neg_ndcg,
                        "neg_map100": neg_m['map_cut_100'],
                        "neg_lsnc100": neg_lsnc,
                        "all_map100": all_m['map_cut_100'],
                    }
                    grid_results.append(result)

                    if neg_ndcg > best_neg_ndcg:
                        best_neg_ndcg = neg_ndcg
                        best_params = (alpha, beta, delta)
                        best_results = result

        logger.info(f"\nBest params (neg nDCG@10): alpha={best_params[0]}, beta={best_params[1]}, delta={best_params[2]}")
        logger.info(f"  neg nDCG@10={best_results['neg_ndcg10']:.4f}, all nDCG@10={best_results['all_ndcg10']:.4f}")
        logger.info(f"  neg MAP@100={best_results['neg_map100']:.4f}, neg LSNC@100={best_results['neg_lsnc100']:.4f}")

        grid_output = {
            "encoder": encoder_name,
            "model_name": config["model_name"],
            "baseline_all": baseline_metrics,
            "baseline_negation": baseline_neg_metrics,
            "baseline_negation_lsnc100": baseline_neg_lsnc,
            "grid_search": grid_results,
            "best_params": {
                "alpha": best_params[0], "beta": best_params[1], "delta": best_params[2],
                "neg_ndcg10": best_results['neg_ndcg10'],
                "all_ndcg10": best_results['all_ndcg10'],
            },
        }
        grid_path = "results/comlq/comlq_bge_grid_search.json"
        os.makedirs(os.path.dirname(grid_path), exist_ok=True)
        with open(grid_path, 'w') as f:
            json.dump(grid_output, f, indent=2)
        logger.info(f"Grid search results saved to {grid_path}")

        print("\n" + "=" * 80)
        print(f"GRID SEARCH SUMMARY ({encoder_name})")
        print("=" * 80)
        print(f"Baseline neg nDCG@10: {baseline_neg_metrics['ndcg_cut_10']:.4f}")
        print(f"Best: alpha={best_params[0]}, beta={best_params[1]}, delta={best_params[2]}")
        print(f"  neg nDCG@10={best_results['neg_ndcg10']:.4f} (Δ={best_results['neg_ndcg10']-baseline_neg_metrics['ndcg_cut_10']:+.4f})")
        print(f"  all nDCG@10={best_results['all_ndcg10']:.4f} (Δ={best_results['all_ndcg10']-baseline_metrics['ndcg_cut_10']:+.4f})")
        return

    # ============================================================
    # Single parameter evaluation
    # ============================================================
    alpha, beta, delta, t_safety = args.alpha, args.beta, args.delta, args.t_safety
    logger.info(f"\n--- DeIR-Dual V2 (alpha={alpha}, beta={beta}, delta={delta}) ---")

    deir_all, deir_neg, deir_nonneg, deir_lsnc, deir_neg_lsnc, run_deir = apply_scoring_and_evaluate(
        eval_qids, doc_ids, top_k_indices, S_base_full,
        S_req_topk, S_neg_topk, cos_qbase_qneg,
        has_req_mask, has_neg_mask,
        qrels, query_types, neg_qids, nonneg_qids,
        alpha, beta, delta, t_safety, args.top_k
    )

    logger.info("DeIR-Dual V2 Results (All):")
    for m, v in sorted(deir_all.items()):
        delta_v = v - baseline_metrics[m]
        logger.info(f"  {m}: {v:.4f} (Δ={delta_v:+.4f})")
    logger.info(f"  LSNC@100: {deir_lsnc:.4f} (Δ={deir_lsnc - baseline_lsnc:+.4f})")

    logger.info("DeIR-Dual V2 Results (Negation only):")
    for m, v in sorted(deir_neg.items()):
        delta_v = v - baseline_neg_metrics[m]
        logger.info(f"  {m}: {v:.4f} (Δ={delta_v:+.4f})")
    logger.info(f"  LSNC@100: {deir_neg_lsnc:.4f} (Δ={deir_neg_lsnc - baseline_neg_lsnc:+.4f})")

    logger.info("DeIR-Dual V2 Results (Non-negation only):")
    for m, v in sorted(deir_nonneg.items()):
        delta_v = v - baseline_nonneg_metrics[m]
        logger.info(f"  {m}: {v:.4f} (Δ={delta_v:+.4f})")

    deir_type_metrics = {}
    for qtype in sorted(type_counts.keys()):
        type_qids = [qid for qid in eval_qids if get_base_type(query_types.get(qid, '')) == qtype]
        if len(type_qids) >= 2:
            type_metrics, _ = evaluate(run_deir, qrels, type_qids)
            deir_type_metrics[qtype] = type_metrics

    # Q_plus only
    logger.info("\n--- Q_plus Only ---")
    S_qplus_topk = torch.zeros(len(eval_qids), args.top_k)
    for i in range(len(eval_qids)):
        valid_mask = top_k_indices[i] >= 0
        if valid_mask.sum() == 0:
            continue
        k = valid_mask.sum().item()
        s_b = S_base_full[i, top_k_indices[i][:k]]
        s_r = S_req_topk[i, :k]
        has_req = bool(has_req_mask[i] > 0)
        s_req_eff = s_r if has_req else torch.zeros_like(s_b)
        S_qplus_topk[i, :k] = s_b + beta * s_req_eff

    run_qplus = {}
    for i, qid in enumerate(eval_qids):
        run_qplus[qid] = {}
        for j in range(args.top_k):
            if top_k_indices[i, j] < 0:
                break
            did = doc_ids[top_k_indices[i, j].item()]
            run_qplus[qid][did] = S_qplus_topk[i, j].item()

    qplus_metrics, _ = evaluate(run_qplus, qrels, eval_qids)
    qplus_lsnc = compute_lsnc(run_qplus, qrels, query_types, eval_qids, K=100)
    qplus_neg_metrics, _ = evaluate(run_qplus, qrels, neg_qids)
    qplus_neg_lsnc = compute_lsnc(run_qplus, qrels, query_types, neg_qids, K=100)

    logger.info("Q_plus Only Results (All):")
    for m, v in sorted(qplus_metrics.items()):
        delta_v = v - baseline_metrics[m]
        logger.info(f"  {m}: {v:.4f} (Δ={delta_v:+.4f})")
    logger.info(f"  LSNC@100: {qplus_lsnc:.4f}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 80)
    print(f"ComLQ EVALUATION SUMMARY ({encoder_name})")
    print(f"Params: alpha={alpha}, beta={beta}, delta={delta}")
    print("=" * 80)
    print(f"\n--- All Queries ({len(eval_qids)}) ---")
    print(f"{'Method':<35} {'nDCG@10':>8} {'MAP@100':>8} {'R@10':>8} {'LSNC@100':>9}")
    print("-" * 80)
    print(f"{'Baseline':<35} {baseline_metrics['ndcg_cut_10']:>8.4f} {baseline_metrics['map_cut_100']:>8.4f} {baseline_metrics['recall_10']:>8.4f} {baseline_lsnc:>9.4f}")
    print(f"{'Q_plus only':<35} {qplus_metrics['ndcg_cut_10']:>8.4f} {qplus_metrics['map_cut_100']:>8.4f} {qplus_metrics['recall_10']:>8.4f} {qplus_lsnc:>9.4f}")
    print(f"{'DeIR-Dual V2':<35} {deir_all['ndcg_cut_10']:>8.4f} {deir_all['map_cut_100']:>8.4f} {deir_all['recall_10']:>8.4f} {deir_lsnc:>9.4f}")

    print(f"\n--- Negation Queries ({len(neg_qids)}) ---")
    print(f"{'Method':<35} {'nDCG@10':>8} {'MAP@100':>8} {'R@10':>8} {'LSNC@100':>9}")
    print("-" * 80)
    print(f"{'Baseline':<35} {baseline_neg_metrics['ndcg_cut_10']:>8.4f} {baseline_neg_metrics['map_cut_100']:>8.4f} {baseline_neg_metrics['recall_10']:>8.4f} {baseline_neg_lsnc:>9.4f}")
    print(f"{'Q_plus only':<35} {qplus_neg_metrics['ndcg_cut_10']:>8.4f} {qplus_neg_metrics['map_cut_100']:>8.4f} {qplus_neg_metrics['recall_10']:>8.4f} {qplus_neg_lsnc:>9.4f}")
    print(f"{'DeIR-Dual V2':<35} {deir_neg['ndcg_cut_10']:>8.4f} {deir_neg['map_cut_100']:>8.4f} {deir_neg['recall_10']:>8.4f} {deir_neg_lsnc:>9.4f}")

    print(f"\n--- Per-Type nDCG@10 ---")
    print(f"{'Type':<8} {'Baseline':>10} {'DeIR-Dual':>10} {'Δ':>8}")
    print("-" * 40)
    for qtype in sorted(type_counts.keys()):
        b_val = baseline_type_metrics.get(qtype, {}).get('ndcg_cut_10', 0.0)
        d_val = deir_type_metrics.get(qtype, {}).get('ndcg_cut_10', 0.0)
        delta_val = d_val - b_val
        marker = " *" if is_negation_type(qtype) else ""
        print(f"{qtype:<8} {b_val:>10.4f} {d_val:>10.4f} {delta_val:>+8.4f}{marker}")

    output = {
        "encoder": encoder_name,
        "model_name": config["model_name"],
        "num_queries": len(eval_qids),
        "num_negation_queries": len(neg_qids),
        "baseline_all": baseline_metrics,
        "baseline_negation": baseline_neg_metrics,
        "baseline_nonnegation": baseline_nonneg_metrics,
        "baseline_lsnc100": baseline_lsnc,
        "baseline_negation_lsnc100": baseline_neg_lsnc,
        "qplus_all": qplus_metrics,
        "qplus_negation": qplus_neg_metrics,
        "qplus_lsnc100": qplus_lsnc,
        "qplus_negation_lsnc100": qplus_neg_lsnc,
        "deir_all": deir_all,
        "deir_negation": deir_neg,
        "deir_nonnegation": deir_nonneg,
        "deir_lsnc100": deir_lsnc,
        "deir_negation_lsnc100": deir_neg_lsnc,
        "baseline_per_type": baseline_type_metrics,
        "deir_per_type": deir_type_metrics,
        "params": {"alpha": alpha, "beta": beta, "delta": delta, "t_safety": t_safety, "top_k": args.top_k},
        "q_minus_rate": f"{int(sum(has_neg_mask))}/{len(has_neg_mask)}",
        "q_minus_negation_rate": f"{neg_minus_count}/{len(neg_qids)}",
    }
    output_path = config["results_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
