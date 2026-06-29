"""
分析 delta_S_req = S_req(changed) - S_req(og) 对 FP_nomatch 的区分能力

假设：FP_nomatch 不满足 Changed 新增的 q_plus 要求
     → S_req(changed) < S_req(og) → delta_S_req < 0
     而 TP 满足新增要求 → delta_S_req >= 0
"""

import sys, os, json, torch, torch.nn.functional as F
import numpy as np
import logging
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, "/home/luwa/Documents/DSCLR")

DUAL_QUERIES_DIR = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_4B/TSC_BALANCED_t01"
MODEL_NAME = "samaya-ai/RepLLaMA-reproduced"

def is_none(text):
    if not text: return True
    return str(text).strip().upper() in ("[NONE]", "NONE", "NULL", "N/A", "")

def extract_keywords(q_minus):
    if not q_minus or is_none(q_minus): return []
    phrases = [p.strip().lower() for p in q_minus.split(',')]
    keywords = []
    for p in phrases:
        words = [w for w in p.split() if w not in ('the','a','an','in','of','and','or','to','for','with','that','this','is','are','be','should','avoid','documents','document','about','discussing','discuss','not','merely','mention','mentions')]
        if words: keywords.append(' '.join(words))
    return keywords

def doc_matches_qminus(doc_text, keywords):
    if not doc_text or not keywords: return False
    doc_lower = doc_text.lower()
    return any(kw in doc_lower for kw in keywords)

def analyze_task(task_name):
    logger.info(f"\n{'='*80}\n分析任务: {task_name}\n{'='*80}")
    from eval.engine_dscrl import DSCLREvaluatorEngine, load_cached_embeddings
    from eval.metrics.evaluator import DataLoader

    dual_path = os.path.join(DUAL_QUERIES_DIR, f"dual_queries_TSC_BALANCED_t01_{task_name}.jsonl")
    engine = DSCLREvaluatorEngine(model_name=MODEL_NAME, task_name=task_name,
        output_dir=f"/tmp/analysis_{task_name}", device="cuda", batch_size=64, use_cache=True)
    corpus, q_og, q_changed, candidates = engine.data_loader.load()
    q_raw_og, q_raw_changed = engine.data_loader.load_raw_queries()
    data_loader = DataLoader(task_name)
    qrels = data_loader.load_qrels()

    dual_data = {}
    with open(dual_path) as f:
        for line in f:
            item = json.loads(line.strip())
            dual_data[item["qid"]] = item

    cache_dir = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/embeddings"
    cached_data = load_cached_embeddings(cache_dir, task_name, MODEL_NAME)
    if cached_data is not None:
        cached_embeddings, cached_doc_ids = cached_data
        engine.retriever.set_embeddings(cached_embeddings, cached_doc_ids)

    doc_emb = engine.retriever.doc_embeddings
    doc_ids = engine.retriever.doc_ids
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    qids = list(q_raw_og.keys())
    # 收集 og 和 changed 的 q_plus
    qp_og_list, qp_ch_list, q_minus_list, q_base_list = [], [], [], []
    has_neg_list = []
    valid_qids = []

    for qid in qids:
        base_qid = qid.replace('-og', '')
        og_qid = f"{base_qid}-og"
        changed_qid = f"{base_qid}-changed"
        if og_qid not in qrels: continue

        dual_og = dual_data.get(og_qid, {})
        dual_ch = dual_data.get(changed_qid, {})

        qp_og = dual_og.get("q_plus", "")
        qp_ch = dual_ch.get("q_plus", "")
        q_minus = dual_og.get("q_minus", "")

        if is_none(qp_og) or is_none(qp_ch): continue
        if is_none(q_minus): continue  # 只分析有 neg 的

        raw = q_raw_og.get(qid, ("", ""))
        q_base = f"{raw[0]} {raw[1]}".strip() if raw[0] else ""

        qp_og_list.append(qp_og)
        qp_ch_list.append(qp_ch)
        q_minus_list.append(q_minus)
        q_base_list.append(q_base)
        has_neg_list.append(True)
        valid_qids.append(qid)

    logger.info(f"有效 query: {len(valid_qids)}")

    # 编码
    logger.info("编码 q_base, q_plus_og, q_plus_changed...")
    emb_base = F.normalize(engine.encoder.encode_queries(q_base_list, batch_size=64), p=2, dim=1).to(doc_emb.device)
    emb_qp_og = F.normalize(engine.encoder.encode_queries(qp_og_list, batch_size=64), p=2, dim=1).to(doc_emb.device)
    emb_qp_ch = F.normalize(engine.encoder.encode_queries(qp_ch_list, batch_size=64), p=2, dim=1).to(doc_emb.device)

    doc_emb_norm = F.normalize(doc_emb, p=2, dim=1)
    S_base = torch.matmul(emb_base, doc_emb_norm.T).cpu()
    S_req_og = torch.matmul(emb_qp_og, doc_emb_norm.T).cpu()
    S_req_ch = torch.matmul(emb_qp_ch, doc_emb_norm.T).cpu()
    delta_S_req = S_req_ch - S_req_og  # 关键信号

    # 分类
    tp_delta, fp_match_delta, fp_nomatch_delta = [], [], []
    tp_s_req_ch, fp_nomatch_s_req_ch = [], []
    tp_s_base, fp_nomatch_s_base = [], []

    for i, qid in enumerate(valid_qids):
        base_qid = qid.replace('-og', '')
        og_qid = f"{base_qid}-og"
        changed_qid = f"{base_qid}-changed"
        relevant_og = set(d for d, r in qrels.get(og_qid, {}).items() if r > 0)
        relevant_changed = set(d for d, r in qrels.get(changed_qid, {}).items() if r > 0)
        keywords = extract_keywords(q_minus_list[i])

        for did in candidates.get(base_qid, []):
            if did not in doc_id_to_idx: continue
            doc_idx = doc_id_to_idx[did]
            doc_text = corpus.get(did, {}).get('text', '')
            in_og = did in relevant_og
            in_changed = did in relevant_changed
            d_s_req = delta_S_req[i, doc_idx].item()
            s_req_ch = S_req_ch[i, doc_idx].item()
            s_base = S_base[i, doc_idx].item()

            if in_og and in_changed:
                tp_delta.append(d_s_req)
                tp_s_req_ch.append(s_req_ch)
                tp_s_base.append(s_base)
            elif in_og and not in_changed:
                if doc_matches_qminus(doc_text, keywords):
                    fp_match_delta.append(d_s_req)
                else:
                    fp_nomatch_delta.append(d_s_req)
                    fp_nomatch_s_req_ch.append(s_req_ch)
                    fp_nomatch_s_base.append(s_base)

    tp_delta = np.array(tp_delta)
    fp_match_delta = np.array(fp_match_delta) if fp_match_delta else np.array([])
    fp_nomatch_delta = np.array(fp_nomatch_delta)
    tp_s_req_ch = np.array(tp_s_req_ch)
    fp_nomatch_s_req_ch = np.array(fp_nomatch_s_req_ch)
    tp_s_base = np.array(tp_s_base)
    fp_nomatch_s_base = np.array(fp_nomatch_s_base)

    logger.info(f"\n{'='*60}")
    logger.info(f"delta_S_req = S_req(changed) - S_req(og) 统计")
    logger.info(f"{'='*60}")
    print(f"\n{'类别':<20} {'数量':<10} {'delta_S_req均值':<18} {'S_req_ch均值':<15} {'S_base均值':<15}")
    print("-" * 80)
    print(f"{'TP':<20} {len(tp_delta):<10} {tp_delta.mean():<+18.6f} {tp_s_req_ch.mean():<15.4f} {tp_s_base.mean():<15.4f}")
    if len(fp_match_delta) > 0:
        print(f"{'FP_match':<20} {len(fp_match_delta):<10} {fp_match_delta.mean():<+18.6f}")
    print(f"{'FP_nomatch':<20} {len(fp_nomatch_delta):<10} {fp_nomatch_delta.mean():<+18.6f} {fp_nomatch_s_req_ch.mean():<15.4f} {fp_nomatch_s_base.mean():<15.4f}")

    # TP vs FP_nomatch 的 delta_S_req 区分能力
    if len(fp_nomatch_delta) > 0:
        logger.info(f"\n--- TP vs FP_nomatch: delta_S_req 区分能力 ---")
        tp_m, fp_m = tp_delta.mean(), fp_nomatch_delta.mean()
        tp_s, fp_s = tp_delta.std(), fp_nomatch_delta.std()
        diff = fp_m - tp_m
        pooled = np.sqrt((tp_s**2 + fp_s**2) / 2)
        d = diff / pooled if pooled > 1e-8 else 0
        print(f"  TP delta: {tp_m:+.6f}, FP_nomatch delta: {fp_m:+.6f}, diff: {diff:+.6f}, Cohen-d: {d:+.4f}")
        labels = np.concatenate([np.ones(len(fp_nomatch_delta)), np.zeros(len(tp_delta))])
        scores = np.concatenate([fp_nomatch_delta, tp_delta])
        if len(set(labels)) > 1:
            print(f"  AUC (delta_S_req): {roc_auc_score(labels, scores):.4f}")

    # 也看 S_req_ch 本身的区分能力（对比）
    if len(fp_nomatch_delta) > 0:
        logger.info(f"\n--- TP vs FP_nomatch: S_req(changed) 区分能力（对比）---")
        tp_m, fp_m = tp_s_req_ch.mean(), fp_nomatch_s_req_ch.mean()
        diff = fp_m - tp_m
        pooled = np.sqrt((tp_s_req_ch.std()**2 + fp_nomatch_s_req_ch.std()**2) / 2)
        d = diff / pooled if pooled > 1e-8 else 0
        print(f"  TP S_req_ch: {tp_m:.6f}, FP_nomatch S_req_ch: {fp_m:.6f}, diff: {diff:+.6f}, Cohen-d: {d:+.4f}")
        labels = np.concatenate([np.ones(len(fp_nomatch_s_req_ch)), np.zeros(len(tp_s_req_ch))])
        scores = np.concatenate([fp_nomatch_s_req_ch, tp_s_req_ch])
        if len(set(labels)) > 1:
            print(f"  AUC (S_req_ch): {roc_auc_score(labels, scores):.4f}")

def main():
    for task in ["Core17InstructionRetrieval", "Robust04InstructionRetrieval"]:
        analyze_task(task)

if __name__ == "__main__":
    main()
