"""
FollowIR FP 细粒度语义分析 - 深入版

关键发现：FollowIR 的 FP 是异质的
- FP_type1: 触发了 q_minus 排除条件（如 temperature）→ S_neg 应该高
- FP_type2: 不满足 q_plus 增强要求（如缺少 SAR）→ S_neg 不一定高
- FP_type3: 人工标注差异 → 随机

通过关键词匹配细分 FP，验证 S_neg 能否区分 FP_type1 和 TP
"""

import sys
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import logging
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, "/home/luwa/Documents/DSCLR")

DUAL_QUERIES_DIR = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_4B/TSC_BALANCED_t01"
MODEL_NAME = "samaya-ai/RepLLaMA-reproduced"


def is_none(text):
    if not text:
        return True
    return str(text).strip().upper() in ("[NONE]", "NONE", "NULL", "N/A", "")


def extract_keywords_from_qminus(q_minus):
    """从 q_minus 提取关键词（简单方法：取名词短语）"""
    if not q_minus or is_none(q_minus):
        return []
    # 简单提取：按逗号分割，取每个短语的核心词
    phrases = [p.strip().lower() for p in q_minus.split(',')]
    keywords = []
    for p in phrases:
        # 去掉常见停用词
        words = [w for w in p.split() if w not in ('the', 'a', 'an', 'in', 'of', 'and', 'or', 'to', 'for', 'with', 'that', 'this', 'is', 'are', 'be', 'should', 'avoid', 'documents', 'document', 'about', 'discussing', 'discuss', 'not', 'merely', 'mention', 'mentions')]
        if words:
            keywords.append(' '.join(words))
    return keywords


def doc_matches_qminus_keywords(doc_text, keywords):
    """检查文档是否包含 q_minus 的关键词"""
    if not doc_text or not keywords:
        return False, []
    doc_lower = doc_text.lower()
    matched = []
    for kw in keywords:
        if kw in doc_lower:
            matched.append(kw)
    return len(matched) > 0, matched


def analyze_task(task_name):
    logger.info(f"\n{'='*80}")
    logger.info(f"分析任务: {task_name}")
    logger.info(f"{'='*80}")

    from eval.engine_dscrl import DSCLREvaluatorEngine, load_cached_embeddings
    from eval.metrics.evaluator import DataLoader

    dual_path = os.path.join(DUAL_QUERIES_DIR, f"dual_queries_TSC_BALANCED_t01_{task_name}.jsonl")

    engine = DSCLREvaluatorEngine(
        model_name=MODEL_NAME, task_name=task_name,
        output_dir=f"/tmp/analysis_{task_name}", device="cuda", batch_size=64, use_cache=True,
    )

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
    else:
        all_doc_ids = engine._get_all_candidate_doc_ids(candidates)
        doc_texts = [corpus[did]["text"] for did in all_doc_ids]
        engine.retriever.index_documents(all_doc_ids, doc_texts, engine.batch_size)

    doc_emb = engine.retriever.doc_embeddings
    doc_ids = engine.retriever.doc_ids
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    qids = list(q_raw_og.keys())
    q_base_list, q_req_list, q_neg_list = [], [], []
    has_neg_list = []

    for qid in qids:
        raw = q_raw_og.get(qid, ("", ""))
        q_base = f"{raw[0]} {raw[1]}".strip() if raw[0] else ""
        q_base_list.append(q_base)
        dual = dual_data.get(qid, {})
        q_plus = dual.get("q_plus", "")
        q_minus = dual.get("q_minus", "")
        q_req_list.append(q_plus if q_plus and not is_none(q_plus) else "")
        q_neg_list.append(q_minus if q_minus and not is_none(q_minus) else "")
        has_neg_list.append(bool(q_minus and not is_none(q_minus)))

    logger.info("编码查询...")
    emb_base = F.normalize(engine.encoder.encode_queries(q_base_list, batch_size=64), p=2, dim=1).to(doc_emb.device)
    emb_req = F.normalize(engine.encoder.encode_queries(q_req_list, batch_size=64), p=2, dim=1).to(doc_emb.device)
    emb_neg = F.normalize(engine.encoder.encode_queries(q_neg_list, batch_size=64), p=2, dim=1).to(doc_emb.device)

    doc_emb_norm = F.normalize(doc_emb, p=2, dim=1)
    S_base = torch.matmul(emb_base, doc_emb_norm.T).cpu()
    S_req = torch.matmul(emb_req, doc_emb_norm.T).cpu()
    S_neg = torch.matmul(emb_neg, doc_emb_norm.T).cpu()
    cos_qbase_qneg = torch.nan_to_num(F.cosine_similarity(emb_base, emb_neg, dim=1).cpu(), nan=0.0)

    # ===== 分类并细分 FP =====
    tp_data = {"S_base": [], "S_req": [], "S_neg": [], "cos": []}
    fp_match = {"S_base": [], "S_req": [], "S_neg": [], "cos": [], "matched_kw": []}  # FP 触发 q_minus
    fp_nomatch = {"S_base": [], "S_req": [], "S_neg": [], "cos": []}  # FP 未触发 q_minus
    tn_data = {"S_base": [], "S_req": [], "S_neg": [], "cos": []}

    fp_match_examples = []
    fp_nomatch_examples = []
    tp_examples = []

    for i, qid in enumerate(qids):
        base_qid = qid.replace('-og', '')
        og_qid = f"{base_qid}-og"
        changed_qid = f"{base_qid}-changed"
        if og_qid not in qrels:
            continue

        relevant_og = set(d for d, r in qrels.get(og_qid, {}).items() if r > 0)
        relevant_changed = set(d for d, r in qrels.get(changed_qid, {}).items() if r > 0)
        cos_i = cos_qbase_qneg[i].item()
        q_has_neg = has_neg_list[i]
        q_minus_text = q_neg_list[i] if q_has_neg else ""
        keywords = extract_keywords_from_qminus(q_minus_text) if q_has_neg else []

        if not q_has_neg:
            continue

        candidate_docs = candidates.get(base_qid, [])
        for did in candidate_docs:
            if did not in doc_id_to_idx:
                continue
            doc_idx = doc_id_to_idx[did]
            s_base = S_base[i, doc_idx].item()
            s_req = S_req[i, doc_idx].item()
            s_neg = S_neg[i, doc_idx].item()
            doc_text = corpus.get(did, {}).get('text', '')

            in_og = did in relevant_og
            in_changed = did in relevant_changed

            if in_og and in_changed:
                cat = "tp"
            elif in_og and not in_changed:
                # FP: 细分
                matches, matched_kw = doc_matches_qminus_keywords(doc_text, keywords)
                if matches:
                    cat = "fp_match"
                else:
                    cat = "fp_nomatch"
            elif not in_og and in_changed:
                continue  # skip new_positive
            else:
                cat = "tn"

            target = {"tp": tp_data, "fp_match": fp_match, "fp_nomatch": fp_nomatch, "tn": tn_data}[cat]
            target["S_base"].append(s_base)
            target["S_req"].append(s_req)
            target["S_neg"].append(s_neg)
            target["cos"].append(cos_i)
            if cat == "fp_match":
                target["matched_kw"].append(matched_kw)

            # 保存例子
            if cat == "fp_match" and len(fp_match_examples) < 10:
                fp_match_examples.append({
                    "qid": base_qid, "query": q_base_list[i][:120],
                    "q_minus": q_minus_text[:150], "keywords": keywords,
                    "matched_kw": matched_kw, "did": did,
                    "s_base": s_base, "s_req": s_req, "s_neg": s_neg,
                    "doc_snippet": doc_text[:250],
                })
            elif cat == "fp_nomatch" and len(fp_nomatch_examples) < 10:
                fp_nomatch_examples.append({
                    "qid": base_qid, "query": q_base_list[i][:120],
                    "q_minus": q_minus_text[:150], "keywords": keywords,
                    "did": did, "s_base": s_base, "s_req": s_req, "s_neg": s_neg,
                    "doc_snippet": doc_text[:250],
                })
            elif cat == "tp" and len(tp_examples) < 10:
                tp_examples.append({
                    "qid": base_qid, "query": q_base_list[i][:120],
                    "q_minus": q_minus_text[:150], "did": did,
                    "s_base": s_base, "s_req": s_req, "s_neg": s_neg,
                    "doc_snippet": doc_text[:250],
                })

    # ===== 统计 =====
    logger.info(f"\n{'='*60}")
    logger.info(f"FP 细分统计：fp_match（触发q_minus）vs fp_nomatch（未触发q_minus）")
    logger.info(f"{'='*60}")
    print(f"\n{'类别':<20} {'数量':<10} {'S_base':<12} {'S_req':<12} {'S_neg':<12} {'S_neg_std':<12}")
    print("-" * 80)
    for name, data in [("TP", tp_data), ("FP_match", fp_match), ("FP_nomatch", fp_nomatch), ("TN", tn_data)]:
        if not data["S_neg"]:
            continue
        s_base = np.array(data["S_base"])
        s_req = np.array(data["S_req"])
        s_neg = np.array(data["S_neg"])
        print(f"{name:<20} {len(s_neg):<10} {s_base.mean():<12.4f} {s_req.mean():<12.4f} {s_neg.mean():<12.4f} {s_neg.std():<12.4f}")

    # TP vs FP_match
    if tp_data["S_neg"] and fp_match["S_neg"]:
        logger.info(f"\n--- TP vs FP_match（触发q_minus的FP）---")
        tp_neg = np.array(tp_data["S_neg"])
        fp_neg = np.array(fp_match["S_neg"])
        tp_base = np.array(tp_data["S_base"])
        fp_base = np.array(fp_match["S_base"])
        print(f"  {'指标':<15} {'TP均值':<12} {'FP_match均值':<12} {'差异':<12} {'Cohen-d':<12}")
        for name, tp_v, fp_v in [("S_base", tp_base, fp_base), ("S_neg", tp_neg, fp_neg)]:
            diff = fp_v.mean() - tp_v.mean()
            pooled = np.sqrt((tp_v.std()**2 + fp_v.std()**2) / 2)
            d = diff / pooled if pooled > 1e-8 else 0
            print(f"  {name:<15} {tp_v.mean():<12.4f} {fp_v.mean():<12.4f} {diff:<+12.4f} {d:<+12.4f}")
        labels = np.concatenate([np.ones(len(fp_neg)), np.zeros(len(tp_neg))])
        scores = np.concatenate([fp_neg, tp_neg])
        if len(set(labels)) > 1:
            print(f"  AUC (S_neg): {roc_auc_score(labels, scores):.4f}")

    # TP vs FP_nomatch
    if tp_data["S_neg"] and fp_nomatch["S_neg"]:
        logger.info(f"\n--- TP vs FP_nomatch（未触发q_minus的FP）---")
        tp_neg = np.array(tp_data["S_neg"])
        fp_neg = np.array(fp_nomatch["S_neg"])
        print(f"  TP S_neg: {tp_neg.mean():.4f}, FP_nomatch S_neg: {fp_neg.mean():.4f}, diff: {fp_neg.mean()-tp_neg.mean():+.4f}")

    # ===== 具体例子 =====
    if fp_match_examples:
        logger.info(f"\n{'='*60}")
        logger.info(f"FP_match 例子（文档包含q_minus关键词 - 真正违反约束）")
        logger.info(f"{'='*60}")
        for ex in fp_match_examples[:5]:
            print(f"\n--- Q{ex['qid']} FP_match ---")
            print(f"  Query: {ex['query']}")
            print(f"  Q_minus: {ex['q_minus']}")
            print(f"  Keywords: {ex['keywords']}")
            print(f"  Matched: {ex['matched_kw']}")
            print(f"  Doc {ex['did']}: S_base={ex['s_base']:.4f}, S_neg={ex['s_neg']:.4f}")
            print(f"  Snippet: {ex['doc_snippet']}...")

    if fp_nomatch_examples:
        logger.info(f"\n{'='*60}")
        logger.info(f"FP_nomatch 例子（文档不含q_minus关键词 - 因不满足q_plus被降级）")
        logger.info(f"{'='*60}")
        for ex in fp_nomatch_examples[:5]:
            print(f"\n--- Q{ex['qid']} FP_nomatch ---")
            print(f"  Query: {ex['query']}")
            print(f"  Q_minus: {ex['q_minus']}")
            print(f"  Keywords: {ex['keywords']}")
            print(f"  Doc {ex['did']}: S_base={ex['s_base']:.4f}, S_neg={ex['s_neg']:.4f}")
            print(f"  Snippet: {ex['doc_snippet']}...")

    if tp_examples:
        logger.info(f"\n{'='*60}")
        logger.info(f"TP 例子（真相关文档）")
        logger.info(f"{'='*60}")
        for ex in tp_examples[:3]:
            print(f"\n--- Q{ex['qid']} TP ---")
            print(f"  Query: {ex['query']}")
            print(f"  Q_minus: {ex['q_minus']}")
            print(f"  Doc {ex['did']}: S_base={ex['s_base']:.4f}, S_neg={ex['s_neg']:.4f}")
            print(f"  Snippet: {ex['doc_snippet']}...")


def main():
    for task in ["Core17InstructionRetrieval", "Robust04InstructionRetrieval"]:
        analyze_task(task)


if __name__ == "__main__":
    main()
