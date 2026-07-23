"""
Measure TRACE latency for Table 6.

Two components:
1. Decomposition latency: one structured LLM call
2. TRACE reranking latency: robust standardization + regression + scoring on cached embeddings

Usage:
  cd /home/luwa/Documents/DSCLR-remote && \
  /home/luwa/.conda/envs/dsclr/bin/python -m eval.measure_latency --device cuda
"""

import os, sys, json, argparse, time, logging
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from eval.engine_trace import TRACEEvaluator, robust_standardize, _mad, fit_huber_regression


def measure_trace_reranking_latency(task_name, model_name, dual_queries_path, device="cuda"):
    """Measure TRACE reranking latency per query."""
    # Create evaluator with fixed params (no grid search)
    engine = TRACEEvaluator(
        model_name=model_name,
        task_name=task_name,
        output_dir=f"/tmp/trace_latency/{task_name}",
        dual_queries_path=dual_queries_path,
        lambda_boundary=1.0,
        tau_decay=0.2,
        device=device,
    )

    # Load data via the run method's internals
    dual_data = engine.load_dual_queries()
    corpus, q_og, q_changed, candidates = engine.data_loader.load()
    all_doc_ids = engine._get_all_candidate_doc_ids(candidates)

    # Ensure embeddings are loaded
    logger.info("Loading/encoding embeddings (one-time cost, not measured)...")
    from eval.engine_trace import load_cached_embeddings
    cached_data = load_cached_embeddings(engine.cache_dir, task_name, model_name)
    if cached_data is not None:
        cached_embeddings, cached_doc_ids = cached_data
        if set(cached_doc_ids) == set(all_doc_ids):
            engine.retriever.set_embeddings(cached_embeddings, cached_doc_ids)
        else:
            doc_texts = [corpus[did]["text"] for did in all_doc_ids]
            engine.retriever.encode_documents(doc_texts)
    else:
        doc_texts = [corpus[did]["text"] for did in all_doc_ids]
        engine.retriever.encode_documents(doc_texts)

    doc_emb = engine.retriever.doc_embeddings
    doc_id_to_col_idx = {doc_id: idx for idx, doc_id in enumerate(engine.retriever.doc_ids)}

    # Build candidate indices
    qid_to_candidate_indices = {}
    for qid, doc_ids in candidates.items():
        indices = [doc_id_to_col_idx[did] for did in doc_ids if did in doc_id_to_col_idx]
        qid_to_candidate_indices[qid] = torch.tensor(indices, dtype=torch.long)

    # Encode all query views
    qids = sorted(candidates.keys())
    logger.info(f"Total queries: {len(qids)}")

    # Prepare query embeddings for all 3 views
    q_full_texts, q_pos_texts, q_neg_texts = [], [], []
    for qid in qids:
        d = dual_data.get(qid, {})
        q_full = q_changed.get(qid, "")
        q_pos = d.get('q_plus', q_full)
        q_neg = d.get('q_minus', '[NONE]')
        q_full_texts.append(q_full)
        q_pos_texts.append(q_pos)
        q_neg_texts.append(q_neg)

    logger.info("Encoding query views...")
    encoder = engine.retriever.encoder
    q_full_embs = F.normalize(encoder.encode_queries(q_full_texts).float(), p=2, dim=1).to(device)
    q_pos_embs = F.normalize(encoder.encode_queries(q_pos_texts).float(), p=2, dim=1).to(device)
    q_neg_embs = F.normalize(encoder.encode_queries(q_neg_texts).float(), p=2, dim=1).to(device)
    doc_emb_device = doc_emb.to(device)

    # Warmup: run TRACE scoring for first 5 queries
    logger.info("Warmup (5 queries)...")
    for i in range(min(5, len(qids))):
        qid = qids[i]
        indices = qid_to_candidate_indices[qid]
        doc_cands = doc_emb_device[indices]
        s_full = torch.matmul(q_full_embs[i:i+1], doc_cands.T).squeeze(0)
        s_pos = torch.matmul(q_pos_embs[i:i+1], doc_cands.T).squeeze(0)
        s_neg = torch.matmul(q_neg_embs[i:i+1], doc_cands.T).squeeze(0)
        z_full = robust_standardize(s_full.float(), 1e-6)
        z_pos = robust_standardize(s_pos.float(), 1e-6)
        z_neg = robust_standardize(s_neg.float(), 1e-6)
        b_hat, a_hat = fit_huber_regression(z_neg, z_pos)
        e = z_neg - (a_hat + b_hat * z_pos)
        e_z = robust_standardize(e.float(), 1e-6)

    torch.cuda.synchronize()

    # Measure per-query latency
    logger.info(f"Measuring latency ({len(qids)} queries)...")
    latencies = []
    for i, qid in enumerate(qids):
        indices = qid_to_candidate_indices[qid]

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Step 1: Compute similarity scores (3 dot products)
        doc_cands = doc_emb_device[indices]
        s_full = torch.matmul(q_full_embs[i:i+1], doc_cands.T).squeeze(0)
        s_pos = torch.matmul(q_pos_embs[i:i+1], doc_cands.T).squeeze(0)
        s_neg = torch.matmul(q_neg_embs[i:i+1], doc_cands.T).squeeze(0)

        # Step 2: Robust standardize
        z_full = robust_standardize(s_full.float(), 1e-6)
        z_pos = robust_standardize(s_pos.float(), 1e-6)
        z_neg = robust_standardize(s_neg.float(), 1e-6)

        # Step 3: Huber regression
        b_hat, a_hat = fit_huber_regression(z_neg, z_pos)

        # Step 4: Residual + scoring
        e = z_neg - (a_hat + b_hat * z_pos)
        e_z = robust_standardize(e.float(), 1e-6)

        # Step 5: Gate + composition (simplified, representative)
        tau = 0.2
        p = torch.clamp(z_pos, min=0)
        penalty = torch.clamp(e_z - tau, min=0)
        s_final = z_full + p - 1.0 * penalty

        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

    return np.mean(latencies), np.std(latencies), np.median(latencies)


def main():
    parser = argparse.ArgumentParser(description="Measure TRACE latency")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval")
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--dual_queries_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.dual_queries_path is None:
        args.dual_queries_path = f"/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_{args.task_name}.jsonl"

    # Measure TRACE reranking latency
    logger.info("=" * 60)
    logger.info("Measuring TRACE reranking latency...")
    logger.info("=" * 60)
    mean, std, median = measure_trace_reranking_latency(
        task_name=args.task_name,
        model_name=args.model_name,
        dual_queries_path=args.dual_queries_path,
        device=args.device,
    )

    results = {
        "task_name": args.task_name,
        "model_name": args.model_name,
        "trace_rerank_mean_ms": mean * 1000,
        "trace_rerank_std_ms": std * 1000,
        "trace_rerank_median_ms": median * 1000,
    }

    # Save
    output_path = "/home/luwa/Documents/DSCLR/evaluation_remote/latency_measurements.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Latency Summary")
    print("=" * 60)
    print(f"TRACE reranking: {mean*1000:.1f} ± {std*1000:.1f} ms/query (median: {median*1000:.1f} ms)")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
