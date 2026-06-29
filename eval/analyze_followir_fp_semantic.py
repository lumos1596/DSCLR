"""
FollowIR FP vs TP 细粒度语义分析

正确的 FP 定义（基于 qrels 差集）：
- TP: OG相关 AND Changed相关
- FP: OG相关 BUT Changed不相关  ← 被指令惩罚的假阳性
- TN: OG不相关 AND Changed不相关
- NP: OG不相关 BUT Changed相关  ← 指令要求的新相关

目标：手动分析 FP 文档的语义内容，理解为什么 S_neg 无法区分 FP 和 TP
"""

import sys
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import logging
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, "/home/luwa/Documents/DSCLR")

DUAL_QUERIES_DIR = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_4B/TSC_BALANCED_t01"
MODEL_NAME = "samaya-ai/RepLLaMA-reproduced"


def is_none(text):
    if not text:
        return True
    return str(text).strip().upper() in ("[NONE]", "NONE", "NULL", "N/A", "")


def analyze_task(task_name):
    """分析单个任务"""
    logger.info(f"\n{'='*80}")
    logger.info(f"分析任务: {task_name}")
    logger.info(f"{'='*80}")

    from eval.engine_dscrl import DSCLREvaluatorEngine, load_cached_embeddings
    from eval.metrics.evaluator import DataLoader

    dual_path = os.path.join(DUAL_QUERIES_DIR, f"dual_queries_TSC_BALANCED_t01_{task_name}.jsonl")

    engine = DSCLREvaluatorEngine(
        model_name=MODEL_NAME,
        task_name=task_name,
        output_dir=f"/tmp/analysis_{task_name}",
        device="cuda",
        batch_size=64,
        use_cache=True,
    )

    corpus, q_og, q_changed, candidates = engine.data_loader.load()
    q_raw_og, q_raw_changed = engine.data_loader.load_raw_queries()

    data_loader = DataLoader(task_name)
    qrels = data_loader.load_qrels()

    dual_data = {}
    with open(dual_path, "r") as f:
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
    logger.info(f"文档数: {len(doc_ids)}, 维度: {doc_emb.shape[1]}")

    qids = list(q_raw_og.keys())
    q_base_list, q_req_list, q_neg_list = [], [], []
    has_neg_list = []

    for qid in qids:
        raw = q_raw_og.get(qid, ("", ""))
        query_text, instruction = raw[0], raw[1]
        q_base = f"{query_text} {instruction}".strip() if query_text else ""
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

    # ===== 分类文档 =====
    categories = {
        "true_positive": {"S_base": [], "S_req": [], "S_neg": [], "cos": [], "count": 0},
        "false_positive": {"S_base": [], "S_req": [], "S_neg": [], "cos": [], "count": 0},
        "true_negative": {"S_base": [], "S_req": [], "S_neg": [], "cos": [], "count": 0},
        "new_positive": {"S_base": [], "S_req": [], "S_neg": [], "cos": [], "count": 0},
    }

    per_query_data = []
    fp_examples = []  # 保存 FP 例子用于语义分析
    tp_examples = []  # 保存 TP 例子用于对比

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

        if not q_has_neg:
            continue

        q_tp_s_neg = []
        q_fp_s_neg = []
        q_tp_s_base = []
        q_fp_s_base = []

        candidate_docs = candidates.get(base_qid, [])
        for did in candidate_docs:
            if did not in doc_id_to_idx:
                continue
            doc_idx = doc_id_to_idx[did]

            s_base = S_base[i, doc_idx].item()
            s_req = S_req[i, doc_idx].item()
            s_neg = S_neg[i, doc_idx].item()

            in_og = did in relevant_og
            in_changed = did in relevant_changed

            if in_og and in_changed:
                cat = "true_positive"
                q_tp_s_neg.append(s_neg)
                q_tp_s_base.append(s_base)
            elif in_og and not in_changed:
                cat = "false_positive"
                q_fp_s_neg.append(s_neg)
                q_fp_s_base.append(s_base)
                # 保存前 5 个 query 的 FP 例子
                if len(fp_examples) < 30 and base_qid in ['341', '344', '355', '356', '305']:
                    doc_text = corpus.get(did, {}).get('text', '')[:300]
                    fp_examples.append({
                        "qid": base_qid,
                        "query": q_base_list[i][:150],
                        "q_minus": q_minus_text[:200],
                        "did": did,
                        "s_base": s_base,
                        "s_req": s_req,
                        "s_neg": s_neg,
                        "doc_snippet": doc_text,
                    })
            elif not in_og and in_changed:
                cat = "new_positive"
            else:
                cat = "true_negative"

            categories[cat]["S_base"].append(s_base)
            categories[cat]["S_req"].append(s_req)
            categories[cat]["S_neg"].append(s_neg)
            categories[cat]["cos"].append(cos_i)
            categories[cat]["count"] += 1

        if q_tp_s_neg and q_fp_s_neg:
            per_query_data.append({
                "qid": base_qid,
                "cos": cos_i,
                "q_minus": q_minus_text[:100],
                "tp_s_neg": q_tp_s_neg,
                "fp_s_neg": q_fp_s_neg,
                "tp_s_base": q_tp_s_base,
                "fp_s_base": q_fp_s_base,
            })

        # 保存 TP 例子用于对比
        if len(tp_examples) < 15 and base_qid in ['341', '344', '355', '356', '305'] and q_tp_s_neg:
            for did in candidate_docs:
                if did in relevant_og and did in relevant_changed and did in doc_id_to_idx:
                    doc_idx = doc_id_to_idx[did]
                    doc_text = corpus.get(did, {}).get('text', '')[:300]
                    tp_examples.append({
                        "qid": base_qid,
                        "query": q_base_list[i][:150],
                        "q_minus": q_minus_text[:200],
                        "did": did,
                        "s_base": S_base[i, doc_idx].item(),
                        "s_req": S_req[i, doc_idx].item(),
                        "s_neg": S_neg[i, doc_idx].item(),
                        "doc_snippet": doc_text,
                    })
                    break

    # ===== 统计 =====
    logger.info(f"\n{'='*60}")
    logger.info(f"文档分类统计（基于 OG/Changed qrels 差集）")
    logger.info(f"{'='*60}")
    print(f"\n{'类别':<20} {'数量':<10} {'S_base均值':<12} {'S_req均值':<12} {'S_neg均值':<12} {'S_neg标准差':<12}")
    print("-" * 80)

    stats = {}
    for cat, data in categories.items():
        if data["count"] == 0:
            continue
        s_base_arr = np.array(data["S_base"])
        s_req_arr = np.array(data["S_req"])
        s_neg_arr = np.array(data["S_neg"])
        stats[cat] = {
            "count": data["count"],
            "S_base": s_base_arr,
            "S_req": s_req_arr,
            "S_neg": s_neg_arr,
        }
        print(f"{cat:<20} {data['count']:<10} {s_base_arr.mean():<12.4f} {s_req_arr.mean():<12.4f} {s_neg_arr.mean():<12.4f} {s_neg_arr.std():<12.4f}")

    # ===== TP vs FP =====
    if "true_positive" in stats and "false_positive" in stats:
        logger.info(f"\n{'='*60}")
        logger.info(f"TP vs FP 对比")
        logger.info(f"{'='*60}")
        tp = stats["true_positive"]
        fp = stats["false_positive"]
        print(f"\n{'指标':<20} {'TP均值':<15} {'FP均值':<15} {'差异':<15} {'Cohen-d':<12}")
        print("-" * 80)
        for metric in ["S_base", "S_req", "S_neg"]:
            tp_m = tp[metric].mean()
            fp_m = fp[metric].mean()
            tp_s = tp[metric].std()
            fp_s = fp[metric].std()
            diff = fp_m - tp_m
            pooled = np.sqrt((tp_s**2 + fp_s**2) / 2)
            d = diff / pooled if pooled > 1e-8 else 0
            print(f"{metric:<20} {tp_m:<15.4f} {fp_m:<15.4f} {diff:<+15.4f} {d:<+12.4f}")

        labels = np.concatenate([np.ones(len(fp["S_neg"])), np.zeros(len(tp["S_neg"]))])
        scores = np.concatenate([fp["S_neg"], tp["S_neg"]])
        if len(set(labels)) > 1:
            auc = roc_auc_score(labels, scores)
            print(f"\nAUC (S_neg 区分 FP vs TP): {auc:.4f}")

    # ===== 逐 query 分析 =====
    if per_query_data:
        logger.info(f"\n{'='*60}")
        logger.info(f"逐 query 分析：FP vs TP 的 S_neg 差异")
        logger.info(f"{'='*60}")

        diffs_neg = np.array([np.mean(qd["fp_s_neg"]) - np.mean(qd["tp_s_neg"]) for qd in per_query_data])
        diffs_base = np.array([np.mean(qd["fp_s_base"]) - np.mean(qd["tp_s_base"]) for qd in per_query_data])

        print(f"\n  [全部 query] (n={len(per_query_data)})")
        print(f"  {'指标':<25} {'均值差(FP-TP)':<15} {'标准差':<12} {'t统计量':<12} {'p值':<12} {'FP>TP比例':<12}")
        print(f"  {'-'*93}")
        for name, diffs in [("S_neg", diffs_neg), ("S_base", diffs_base)]:
            mean_d = diffs.mean()
            std_d = diffs.std()
            n = len(diffs)
            t_stat = mean_d / (std_d / np.sqrt(n)) if std_d > 1e-8 else 0
            p_val = 1 - sp_stats.t.cdf(t_stat, df=n-1)
            fp_gt_tp = (diffs > 0).mean()
            print(f"  {name:<25} {mean_d:<+15.6f} {std_d:<12.6f} {t_stat:<+12.4f} {p_val:<12.6f} {fp_gt_tp:<12.4f}")

    # ===== 具体例子分析 =====
    if fp_examples:
        logger.info(f"\n{'='*60}")
        logger.info(f"FP 例子（OG相关但Changed不相关 - 被指令惩罚的文档）")
        logger.info(f"{'='*60}")
        for ex in fp_examples:
            print(f"\n--- Query {ex['qid']} (FP) ---")
            print(f"  Query: {ex['query']}")
            print(f"  Q_minus: {ex['q_minus']}")
            print(f"  Doc {ex['did']}: S_base={ex['s_base']:.4f}, S_req={ex['s_req']:.4f}, S_neg={ex['s_neg']:.4f}")
            print(f"  Doc snippet: {ex['doc_snippet']}...")

    if tp_examples:
        logger.info(f"\n{'='*60}")
        logger.info(f"TP 例子（OG和Changed都相关 - 真相关文档）")
        logger.info(f"{'='*60}")
        for ex in tp_examples:
            print(f"\n--- Query {ex['qid']} (TP) ---")
            print(f"  Query: {ex['query']}")
            print(f"  Q_minus: {ex['q_minus']}")
            print(f"  Doc {ex['did']}: S_base={ex['s_base']:.4f}, S_req={ex['s_req']:.4f}, S_neg={ex['s_neg']:.4f}")
            print(f"  Doc snippet: {ex['doc_snippet']}...")

    # ===== cos 分布 =====
    logger.info(f"\n{'='*60}")
    logger.info(f"cos(Q_base, Q_neg) 分布")
    logger.info(f"{'='*60}")
    has_neg_cos = cos_qbase_qneg[np.array(has_neg_list)]
    print(f"\n  has_neg=True (n={len(has_neg_cos)}):")
    print(f"    mean={has_neg_cos.mean():.4f}, std={has_neg_cos.std():.4f}")
    print(f"    min={has_neg_cos.min():.4f}, max={has_neg_cos.max():.4f}")
    print(f"    median={np.median(has_neg_cos):.4f}")
    print(f"    <0.3: {(has_neg_cos < 0.3).float().mean()*100:.1f}%, <0.5: {(has_neg_cos < 0.5).float().mean()*100:.1f}%, <0.7: {(has_neg_cos < 0.7).float().mean()*100:.1f}%")


def main():
    for task in ["Core17InstructionRetrieval", "Robust04InstructionRetrieval"]:
        analyze_task(task)


if __name__ == "__main__":
    main()
