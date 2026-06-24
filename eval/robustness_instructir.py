import os, sys, json, numpy as np, torch, torch.nn.functional as F
import datasets, pytrec_eval
from collections import defaultdict
from tqdm import tqdm

os.environ.pop("HF_ENDPOINT", None)
os.environ.pop("HF_HUB_OFFLINE", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.models.repllama_encoder import RepLLaMAEncoder

ALPHA = 1.0
BETA = 1.5
DELTA = 0.05
T_SAFETY = 20.0
TOP_K = 100

DUAL_QUERIES_PATH = "dataset/InstructIR/dual_queries/InstructIR_TSC_BALANCED_t01.jsonl"

print("=" * 60)
print("InstructIR Robustness@10 Evaluation (with DeIR-Dual V2)")
print("=" * 60)

ds_queries = datasets.load_dataset('mteb/InstructIR-mteb', 'queries')['queries']
ds_inst = datasets.load_dataset('mteb/InstructIR-mteb', 'instruction')['instruction']
ds_qrels = datasets.load_dataset('mteb/InstructIR-mteb', 'default', split='test')

query_map = {str(q['_id']): str(q.get('text', '')) for q in ds_queries}
inst_map = {str(item['query-id']): str(item.get('instruction', '')) for item in ds_inst}

qrel_map = defaultdict(dict)
for item in ds_qrels:
    qid = str(item['query-id'])
    did = str(item['corpus-id'])
    score = int(item.get('score', 1))
    if score > 0:
        qrel_map[qid][did] = score

base_to_variants = defaultdict(list)
for qid in sorted(query_map.keys()):
    base_qid = qid.rsplit('_', 1)[0] if '_' in qid else qid
    base_to_variants[base_qid].append(qid)

eval_qids = sorted(query_map.keys())
print(f"Queries: {len(eval_qids)}, Base queries: {len(base_to_variants)}")

dual_data = {}
with open(DUAL_QUERIES_PATH) as f:
    for line in f:
        r = json.loads(line)
        dual_data[r['qid']] = r
print(f"Dual queries loaded: {len(dual_data)}")

cache = torch.load(
    "dataset/InstructIR/embeddings/instructir_repllama_corpus.pt",
    map_location="cpu", weights_only=False
)
doc_ids = cache["doc_ids"]
doc_id_to_idx = {did: idx for idx, did in enumerate(doc_ids)}
doc_embeddings = F.normalize(cache["embeddings"].float(), p=2, dim=1).to('cuda')
print(f"Doc embeddings: {doc_embeddings.shape}")

encoder = RepLLaMAEncoder(
    model_name="samaya-ai/RepLLaMA-reproduced",
    device="cuda", batch_size=64
)


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


def compute_robustness(per_query_ndcg10, base_to_variants):
    robustness_scores = []
    for base_qid, variant_qids in base_to_variants.items():
        variant_scores = [per_query_ndcg10.get(vq, 0) for vq in variant_qids]
        robustness_scores.append(min(variant_scores))
    return np.mean(robustness_scores), np.median(robustness_scores)


def evaluate_and_report(run_data, label):
    evaluator = pytrec_eval.RelevanceEvaluator(
        {qid: qrel_map[qid] for qid in eval_qids},
        {'ndcg_cut_5', 'ndcg_cut_10'}
    )
    results = evaluator.evaluate(run_data)
    per_query_ndcg10 = {qid: v['ndcg_cut_10'] for qid, v in results.items()}
    mean_ndcg5 = np.mean([v['ndcg_cut_5'] for v in results.values()])
    mean_ndcg10 = np.mean(list(per_query_ndcg10.values()))
    robustness_10, robustness_median = compute_robustness(per_query_ndcg10, base_to_variants)

    print(f"  nDCG@5:   {mean_ndcg5:.4f}")
    print(f"  nDCG@10:  {mean_ndcg10:.4f}")
    print(f"  **Robustness@10: {robustness_10:.4f}** (median={robustness_median:.4f})")

    return {
        'label': label,
        'nDCG@5': mean_ndcg5,
        'nDCG@10': mean_ndcg10,
        'Robustness@10': robustness_10,
        'Robustness@10_median': robustness_median,
    }


def full_retrieval(q_embs, doc_embeddings, doc_ids, eval_qids, top_k=TOP_K):
    scores_list = []
    bs = 500
    for i in range(0, len(q_embs), bs):
        end = min(i + bs, len(q_embs))
        batch_scores = torch.matmul(q_embs[i:end].to('cuda'), doc_embeddings.T).cpu()
        scores_list.append(batch_scores)
    scores = torch.cat(scores_list, dim=0)
    topk_scores, topk_indices = torch.topk(scores, k=min(top_k, scores.shape[1]), dim=1)

    run_data = {}
    for i, qid in enumerate(eval_qids):
        run_data[qid] = {}
        for j in range(topk_scores.shape[1]):
            did = doc_ids[topk_indices[i, j].item()]
            run_data[qid][did] = topk_scores[i, j].item()
    return run_data, scores, topk_scores, topk_indices


results_summary = []

# ============================================================
# Format A: Baseline (instruction:inst [SEP] query)
# ============================================================
print(f"\n--- Evaluating: A: Baseline (instruction:inst [SEP] query) ---")
q_base_list = []
for qid in eval_qids:
    q_base_list.append(f"instruction: {inst_map.get(qid,'')} [SEP] {query_map.get(qid,'')}")

q_base_emb = F.normalize(encoder.encode_queries(q_base_list, batch_size=64).float(), p=2, dim=1)
run_baseline, S_base_full, topk_scores_base, topk_indices_base = full_retrieval(
    q_base_emb, doc_embeddings, doc_ids, eval_qids
)
results_summary.append(evaluate_and_report(run_baseline, "A: Baseline (inst+query)"))

# ============================================================
# Format B: Query only (no instruction)
# ============================================================
print(f"\n--- Evaluating: B: Query only ---")
q_only_list = [query_map.get(qid, '') for qid in eval_qids]
q_only_emb = F.normalize(encoder.encode_queries(q_only_list, batch_size=64).float(), p=2, dim=1)
run_query_only, _, _, _ = full_retrieval(q_only_emb, doc_embeddings, doc_ids, eval_qids)
results_summary.append(evaluate_and_report(run_query_only, "B: Query only"))

# ============================================================
# Format C: DeIR-Dual V2 (S_base + Q_plus/Q_minus reranking)
# ============================================================
print(f"\n--- Evaluating: C: DeIR-Dual V2 (alpha={ALPHA}, beta={BETA}, delta={DELTA}) ---")

q_req_list = []
q_neg_list = []
has_req_mask = []
has_neg_mask = []

def is_none_query(q):
    return not q or q.strip().lower() in ['none', '[none]', 'n/a', 'null', '']

for qid in eval_qids:
    d = dual_data.get(qid, {})
    q_plus = d.get('q_plus', '')
    q_minus = d.get('q_minus', '')

    if not q_plus or is_none_query(q_plus):
        q_plus = q_base_list[eval_qids.index(qid)]
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

print(f"  Q_plus available: {sum(has_req_mask)}/{len(has_req_mask)}")
print(f"  Q_minus available: {sum(has_neg_mask)}/{len(has_neg_mask)}")

print("  Encoding Q_req (Q+)...")
q_req_emb = F.normalize(encoder.encode_queries(q_req_list, batch_size=64).float(), p=2, dim=1)

print("  Encoding Q_neg (Q-)...")
q_neg_emb = F.normalize(encoder.encode_queries(q_neg_list, batch_size=64).float(), p=2, dim=1)

print("  Computing S_base for top-k candidates...")
S_base = torch.matmul(q_base_emb.to('cuda'), doc_embeddings.T).float().cpu()

top_k_indices = torch.zeros(len(eval_qids), TOP_K, dtype=torch.long)
top_k_scores = torch.zeros(len(eval_qids), TOP_K)
for i in range(len(eval_qids)):
    scores = S_base[i]
    k = min(TOP_K, len(scores))
    topk = torch.topk(scores, k)
    top_k_indices[i, :k] = topk.indices
    top_k_scores[i, :k] = topk.values

print("  Computing S_req and S_neg for top-k candidates...")
S_req_topk = torch.zeros(len(eval_qids), TOP_K)
S_neg_topk = torch.zeros(len(eval_qids), TOP_K)
cos_qbase_qneg = F.cosine_similarity(q_base_emb.cpu(), q_neg_emb.cpu(), dim=1)

for i in tqdm(range(len(eval_qids)), desc="  S_req/S_neg"):
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

print("  Applying DeIR-Dual V2 scoring...")
S_final_topk = torch.zeros(len(eval_qids), TOP_K)
for i in range(len(eval_qids)):
    valid_mask = top_k_indices[i] >= 0
    if valid_mask.sum() == 0:
        continue
    k = valid_mask.sum().item()
    s_b = S_base[i, top_k_indices[i][:k]]
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

results_summary.append(evaluate_and_report(run_deir, f"C: DeIR-Dual V2 (a={ALPHA},b={BETA},d={DELTA})"))

del encoder
torch.cuda.empty_cache()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 80)
print("ROBUSTNESS@10 SUMMARY")
print("=" * 80)
print(f"{'Format':<50} {'nDCG@10':>8} {'Robust@10':>12} {'Drop':>8}")
print("-" * 80)
for r in results_summary:
    drop = r['nDCG@10'] - r['Robustness@10']
    print(f"{r['label']:<50} {r['nDCG@10']:>8.4f} {r['Robustness@10']:>12.4f} {drop:>+8.4f}")

output_path = "results/instructir/robustness_results.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f"\nResults saved to {output_path}")