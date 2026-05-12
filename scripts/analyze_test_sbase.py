import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.engine_dscrl import DSCLREvaluatorEngine

def analyze_test_sbase(model_name, task_name, dual_queries_path):
    engine = DSCLREvaluatorEngine(
        model_name=model_name,
        task_name=task_name,
        output_dir=f"/tmp/analysis_{model_name.replace('/', '_')}_{task_name}",
        use_cache=True,
        device="cuda",
    )

    dual_queries = []
    with open(dual_queries_path) as f:
        for line in f:
            dual_queries.append(json.loads(line.strip()))

    q_base_texts = [dq["q_base"] for dq in dual_queries]
    q_changed_texts = [dq["q_changed"] for dq in dual_queries]

    q_base_embs = engine.retriever.encode_queries(q_base_texts)
    q_base_embs = F.normalize(q_base_embs, p=2, dim=1)

    doc_embs = engine.retriever.doc_embeddings
    if isinstance(doc_embs, np.ndarray):
        doc_embs = torch.from_numpy(doc_embs)
    doc_embs = F.normalize(doc_embs.float(), p=2, dim=1)

    S_base = q_base_embs @ doc_embs.T

    all_sbase = S_base.cpu().numpy().flatten()

    n_queries = S_base.shape[0]
    n_docs = S_base.shape[1]

    print(f"\n  {task_name} ({model_name}):")
    print(f"    Queries: {n_queries}, Docs: {n_docs}")
    print(f"    S_base distribution:")
    print(f"      mean={all_sbase.mean():.4f}, std={all_sbase.std():.4f}")
    print(f"      min={all_sbase.min():.4f}, max={all_sbase.max():.4f}")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"      p{p}={np.percentile(all_sbase, p):.4f}")

    # Per-query top-k S_base
    for k in [1, 10, 100, 1000]:
        topk_vals = np.sort(all_sbase.reshape(n_queries, n_docs), axis=1)[:, -k:]
        print(f"    Top-{k} S_base: mean={topk_vals.mean():.4f}, min={topk_vals.min():.4f}")

    # Fraction of docs with S_base > various thresholds
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
        frac = (all_sbase > thresh).mean()
        print(f"    S_base > {thresh}: {frac * 100:.2f}%")

    del engine
    torch.cuda.empty_cache()

    return all_sbase

for model_name, model_label in [
    ("repllama-reproduced", "Repllama"),
    ("intfloat/e5-mistral-7b-instruct", "Mistral"),
]:
    print(f"\n{'=' * 60}")
    print(f"  {model_label} - Test Set S_base Distribution")
    print(f"{'=' * 60}")

    for ds, task in [
        ("Core17", "Core17InstructionRetrieval"),
        ("Robust04", "Robust04InstructionRetrieval"),
        ("News21", "News21InstructionRetrieval"),
    ]:
        dual_path = f"dataset/FollowIR_test/dual_queries_v5/dual_queries_v5_{task}.jsonl"
        try:
            analyze_test_sbase(model_name, task, dual_path)
        except Exception as e:
            print(f"  {ds}: Error - {e}")
