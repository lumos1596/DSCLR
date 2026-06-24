"""
NegConstraint Evaluation: Baseline vs DeIR-Dual V2

Supports multiple encoders: RepLLaMA, BGE-large-en-v1.5

Usage:
  # RepLLaMA
  cd /home/luwa/Documents/DSCLR && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/luwa/.conda/envs/dsclr/bin/python -m eval.negconstraint_eval --encoder repllama

  # BGE-large-en-v1.5
  cd /home/luwa/Documents/DSCLR && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/luwa/.conda/envs/dsclr/bin/python -m eval.negconstraint_eval --encoder bge
"""

import os, sys, json, csv, argparse, time, logging
import numpy as np, torch, torch.nn.functional as F

torch.cuda._lazy_init()

import pytrec_eval
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.models.encoder import SentenceTransformerEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "dataset/NegConstraint/NegConstraint"
DUAL_QUERIES_PATH = "dataset/NegConstraint/NegConstraint/dual_queries/NegConstraint_TSC_BALANCED_t01.jsonl"

ALPHA = 1.0
BETA = 1.5
DELTA = 0.05
T_SAFETY = 20.0
TOP_K = 100

ENCODER_CONFIGS = {
    "repllama": {
        "class": "repllama",
        "model_name": "samaya-ai/RepLLaMA-reproduced",
        "embedding_cache": "dataset/NegConstraint/NegConstraint/embeddings/negconstraint_repllama_corpus.pt",
        "results_path": "results/negconstraint/negconstraint_repllama_results.json",
        "query_prefix": "",
        "doc_prefix": "",
    },
    "bge": {
        "class": "sentence_transformer",
        "model_name": "BAAI/bge-large-en-v1.5",
        "embedding_cache": "dataset/NegConstraint/NegConstraint/embeddings/negconstraint_bge_corpus.pt",
        "results_path": "results/negconstraint/negconstraint_bge_results.json",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "",
    },
}


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
                       alpha=ALPHA, beta=BETA, delta=DELTA):
    if not has_neg:
        s_req_eff = s_req if has_req else torch.zeros_like(s_base)
        return s_base + beta * s_req_eff
    tau = cos_qbase_qneg + delta
    overflow = s_neg - tau
    smooth_penalty = F.softplus(overflow)
    raw_penalty = alpha * smooth_penalty
    safety = 1.0 - torch.sigmoid((s_neg - tau) * T_SAFETY)
    s_req_eff = s_req if has_req else torch.zeros_like(s_base)
    return s_base + beta * s_req_eff * safety - raw_penalty


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="bge", choices=["repllama", "bge"])
    args = parser.parse_args()

    config = ENCODER_CONFIGS[args.encoder]
    encoder_name = args.encoder
    logger.info("=" * 60)
    logger.info(f"NegConstraint Evaluation: {encoder_name} encoder")
    logger.info("=" * 60)

    corpus, queries, qrels, dual_data = load_data()
    eval_qids = sorted(set(qrels.keys()) & set(queries.keys()))
    logger.info(f"Corpus: {len(corpus)} docs, Queries: {len(queries)}, Eval queries: {len(eval_qids)}")
    logger.info(f"Dual queries loaded: {len(dual_data)}")

    encoder = create_encoder(config)

    doc_ids, doc_embeddings = encode_and_cache_corpus(corpus, encoder, config["embedding_cache"])
    doc_embeddings = F.normalize(doc_embeddings.float(), p=2, dim=1).to('cuda')

    q_prefix = config["query_prefix"]

    # ============================================================
    # Baseline: Q_base = original query
    # ============================================================
    logger.info("\n--- Baseline: Original Query ---")
    q_base_list = [queries[qid] for qid in eval_qids]
    q_base_prefixed = [q_prefix + q for q in q_base_list] if q_prefix else q_base_list
    q_base_emb = F.normalize(encoder.encode_queries(q_base_prefixed, batch_size=256).float(), p=2, dim=1)

    S_base = torch.matmul(q_base_emb.to('cuda'), doc_embeddings.T).float().cpu()

    topk_scores, topk_indices = torch.topk(S_base, k=min(TOP_K, S_base.shape[1]), dim=1)

    run_baseline = {}
    for i, qid in enumerate(eval_qids):
        run_baseline[qid] = {}
        for j in range(topk_scores.shape[1]):
            did = doc_ids[topk_indices[i, j].item()]
            run_baseline[qid][did] = topk_scores[i, j].item()

    baseline_metrics, baseline_per_query = evaluate(run_baseline, qrels, eval_qids)
    logger.info("Baseline Results:")
    for m, v in sorted(baseline_metrics.items()):
        logger.info(f"  {m}: {v:.4f}")

    # ============================================================
    # DeIR-Dual V2
    # ============================================================
    logger.info(f"\n--- DeIR-Dual V2 (alpha={ALPHA}, beta={BETA}, delta={DELTA}) ---")

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

    logger.info("Encoding Q_req (Q+)...")
    q_req_prefixed = [q_prefix + q for q in q_req_list] if q_prefix else q_req_list
    q_req_emb = F.normalize(encoder.encode_queries(q_req_prefixed, batch_size=256).float(), p=2, dim=1)

    logger.info("Encoding Q_neg (Q-)...")
    q_neg_prefixed = [q_prefix + q for q in q_neg_list] if q_prefix else q_neg_list
    q_neg_emb = F.normalize(encoder.encode_queries(q_neg_prefixed, batch_size=256).float(), p=2, dim=1)

    logger.info("Computing S_base for top-k candidates...")
    S_base_full = torch.matmul(q_base_emb.to('cuda'), doc_embeddings.T).float().cpu()

    top_k_indices = torch.zeros(len(eval_qids), TOP_K, dtype=torch.long)
    top_k_scores = torch.zeros(len(eval_qids), TOP_K)
    for i in range(len(eval_qids)):
        scores = S_base_full[i]
        k = min(TOP_K, len(scores))
        topk = torch.topk(scores, k)
        top_k_indices[i, :k] = topk.indices
        top_k_scores[i, :k] = topk.values

    logger.info("Computing S_req and S_neg for top-k candidates...")
    S_req_topk = torch.zeros(len(eval_qids), TOP_K)
    S_neg_topk = torch.zeros(len(eval_qids), TOP_K)
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

    logger.info("Applying DeIR-Dual V2 scoring...")
    S_final_topk = torch.zeros(len(eval_qids), TOP_K)
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
        )
        S_final_topk[i, :k] = s_final

    run_deir = {}
    for i, qid in enumerate(eval_qids):
        run_deir[qid] = {}
        for j in range(TOP_K):
            if top_k_indices[i, j] < 0:
                break
            did = doc_ids[top_k_indices[i, j].item()]
            run_deir[qid][did] = S_final_topk[i, j].item()

    deir_metrics, deir_per_query = evaluate(run_deir, qrels, eval_qids)
    logger.info("DeIR-Dual V2 Results:")
    for m, v in sorted(deir_metrics.items()):
        delta_v = v - baseline_metrics[m]
        logger.info(f"  {m}: {v:.4f} (Δ={delta_v:+.4f})")

    # ============================================================
    # Q_plus only (no penalty)
    # ============================================================
    logger.info("\n--- Q_plus Only (S_base + β*S_req, no penalty) ---")
    S_qplus_topk = torch.zeros(len(eval_qids), TOP_K)
    for i in range(len(eval_qids)):
        valid_mask = top_k_indices[i] >= 0
        if valid_mask.sum() == 0:
            continue
        k = valid_mask.sum().item()
        s_b = S_base_full[i, top_k_indices[i][:k]]
        s_r = S_req_topk[i, :k]
        has_req = bool(has_req_mask[i] > 0)
        s_req_eff = s_r if has_req else torch.zeros_like(s_b)
        S_qplus_topk[i, :k] = s_b + BETA * s_req_eff

    run_qplus = {}
    for i, qid in enumerate(eval_qids):
        run_qplus[qid] = {}
        for j in range(TOP_K):
            if top_k_indices[i, j] < 0:
                break
            did = doc_ids[top_k_indices[i, j].item()]
            run_qplus[qid][did] = S_qplus_topk[i, j].item()

    qplus_metrics, _ = evaluate(run_qplus, qrels, eval_qids)
    logger.info("Q_plus Only Results:")
    for m, v in sorted(qplus_metrics.items()):
        delta_v = v - baseline_metrics[m]
        logger.info(f"  {m}: {v:.4f} (Δ={delta_v:+.4f})")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 80)
    print(f"NegConstraint EVALUATION SUMMARY ({encoder_name})")
    print("=" * 80)
    print(f"{'Method':<35} {'nDCG@10':>8} {'MAP@100':>8} {'R@10':>8} {'R@100':>8}")
    print("-" * 80)
    print(f"{'Baseline (original query)':<35} {baseline_metrics['ndcg_cut_10']:>8.4f} {baseline_metrics['map_cut_100']:>8.4f} {baseline_metrics['recall_10']:>8.4f} {baseline_metrics['recall_100']:>8.4f}")
    print(f"{'Q_plus only (no penalty)':<35} {qplus_metrics['ndcg_cut_10']:>8.4f} {qplus_metrics['map_cut_100']:>8.4f} {qplus_metrics['recall_10']:>8.4f} {qplus_metrics['recall_100']:>8.4f}")
    print(f"{'DeIR-Dual V2 (full)':<35} {deir_metrics['ndcg_cut_10']:>8.4f} {deir_metrics['map_cut_100']:>8.4f} {deir_metrics['recall_10']:>8.4f} {deir_metrics['recall_100']:>8.4f}")
    print("-" * 80)
    print(f"{'Δ DeIR-Dual V2 vs Baseline':<35} {deir_metrics['ndcg_cut_10']-baseline_metrics['ndcg_cut_10']:>+8.4f} {deir_metrics['map_cut_100']-baseline_metrics['map_cut_100']:>+8.4f} {deir_metrics['recall_10']-baseline_metrics['recall_10']:>+8.4f} {deir_metrics['recall_100']-baseline_metrics['recall_100']:>+8.4f}")

    output = {
        "encoder": encoder_name,
        "model_name": config["model_name"],
        "baseline": baseline_metrics,
        "qplus_only": qplus_metrics,
        "deir_dual_v2": deir_metrics,
        "params": {"alpha": ALPHA, "beta": BETA, "delta": DELTA, "t_safety": T_SAFETY, "top_k": TOP_K},
        "q_minus_rate": f"{int(sum(has_neg_mask))}/{len(has_neg_mask)}",
    }
    output_path = config["results_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    del encoder
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
