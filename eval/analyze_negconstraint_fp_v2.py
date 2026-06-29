"""
NegConstraint FP vs TP 语义分析（修正版）

正确的 FP 定义：
- TP: qrels 中的正例文档（相关，数据集保证不包含 q_minus）
- FP_violation: 不在 qrels 但文档文本包含 q_minus（违反否定约束的假阳性）
  ← 这才是真正的"假相关"，对应 FollowIR 中"相关于OG但不相关于Changed"
- FP_clean: 不在 qrels 且不包含 q_minus（普通不相关文档）

关键问题：S_neg 能否区分 TP 和 FP_violation？
假设：FP_violation 包含 q_minus 主题词 → S_neg 应该更高
"""

import os
import sys
import json
import csv
import torch
import torch.nn.functional as F
import numpy as np
import logging
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, "/home/luwa/Documents/DSCLR")

DATA_DIR = "dataset/NegConstraint/NegConstraint"
DUAL_QUERIES_PATH = "dataset/NegConstraint/NegConstraint/dual_queries/NegConstraint_TSC_BALANCED_t01.jsonl"
MODEL_NAME = "samaya-ai/RepLLaMA-reproduced"
TOP_K = 100


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


def doc_contains_qminus(doc_text, q_minus):
    """检查文档是否包含 q_minus（不区分大小写的子串匹配）"""
    if not q_minus or not doc_text:
        return False
    return q_minus.lower() in doc_text.lower()


def main():
    from eval.models.repllama_encoder import RepLLaMAEncoder

    logger.info("=" * 60)
    logger.info("NegConstraint FP vs TP 语义分析（修正版）")
    logger.info("FP_violation = 不在qrels但包含q_minus的文档")
    logger.info("=" * 60)

    corpus, queries, qrels, dual_data = load_data()
    eval_qids = sorted(set(qrels.keys()) & set(queries.keys()))
    logger.info(f"Corpus: {len(corpus)} docs, Queries: {len(queries)}, Eval queries: {len(eval_qids)}")

    # 加载编码器
    encoder = RepLLaMAEncoder(model_name=MODEL_NAME, device="cuda", batch_size=64)

    # 加载文档向量
    cache_path = "dataset/NegConstraint/NegConstraint/embeddings/negconstraint_repllama_corpus.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    doc_ids = cache["doc_ids"]
    doc_embeddings = cache["embeddings"]
    doc_embeddings = F.normalize(doc_embeddings.float(), p=2, dim=1).to('cuda')
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}
    logger.info(f"Docs: {len(doc_ids)}, dim={doc_embeddings.shape[1]}")

    # 准备查询
    q_base_list, q_req_list, q_neg_list = [], [], []
    has_neg_list = []

    for qid in eval_qids:
        q_base_list.append(queries[qid])
        dual = dual_data.get(qid, {})
        q_plus = dual.get('q_plus', '')
        q_minus = dual.get('q_minus', '')
        q_req_list.append(q_plus if q_plus and not is_none_query(q_plus) else "")
        q_neg_list.append(q_minus if q_minus and not is_none_query(q_minus) else "")
        has_neg_list.append(bool(q_minus and not is_none_query(q_minus)))

    logger.info(f"Q_minus available: {sum(has_neg_list)}/{len(has_neg_list)}")

    # 编码查询
    logger.info("Encoding queries...")
    emb_base = F.normalize(encoder.encode_queries(q_base_list, batch_size=64).float(), p=2, dim=1).to('cuda')
    emb_req = F.normalize(encoder.encode_queries(q_req_list, batch_size=64).float(), p=2, dim=1).to('cuda')
    emb_neg = F.normalize(encoder.encode_queries(q_neg_list, batch_size=64).float(), p=2, dim=1).to('cuda')

    # 相似度
    S_base = torch.matmul(emb_base, doc_embeddings.T).cpu()
    S_req = torch.matmul(emb_req, doc_embeddings.T).cpu()
    S_neg = torch.matmul(emb_neg, doc_embeddings.T).cpu()
    cos_qbase_qneg = torch.nan_to_num(F.cosine_similarity(emb_base, emb_neg, dim=1).cpu(), nan=0.0)

    # ===== 分类文档 =====
    categories = {
        "true_positive": {"S_base": [], "S_req": [], "S_neg": [], "cos": [], "count": 0},
        "fp_violation": {"S_base": [], "S_req": [], "S_neg": [], "cos": [], "count": 0},
        "fp_clean": {"S_base": [], "S_req": [], "S_neg": [], "cos": [], "count": 0},
    }

    per_query_data = []
    violation_examples = []  # 保存一些具体的例子用于语义分析

    for i, qid in enumerate(eval_qids):
        if not has_neg_list[i]:
            continue
        relevant = set(d for d, r in qrels.get(qid, {}).items() if r > 0)
        cos_i = cos_qbase_qneg[i].item()
        q_minus = q_neg_list[i]

        scores_i = S_base[i]
        k = min(TOP_K, len(scores_i))
        topk_vals, topk_idxs = torch.topk(scores_i, k)

        q_tp_s_neg = []
        q_fp_viol_s_neg = []
        q_fp_clean_s_neg = []
        q_tp_s_base = []
        q_fp_viol_s_base = []

        for j in range(k):
            doc_idx = topk_idxs[j].item()
            did = doc_ids[doc_idx]
            s_base = topk_vals[j].item()
            s_req = S_req[i, doc_idx].item()
            s_neg = S_neg[i, doc_idx].item()
            doc_text = corpus.get(did, '')

            if did in relevant:
                cat = "true_positive"
                q_tp_s_neg.append(s_neg)
                q_tp_s_base.append(s_base)
            elif doc_contains_qminus(doc_text, q_minus):
                cat = "fp_violation"  # 违反否定约束
                q_fp_viol_s_neg.append(s_neg)
                q_fp_viol_s_base.append(s_base)
                # 保存前几个 query 的例子
                if len(violation_examples) < 15 and qid in ['0', '1', '2', '10', '100']:
                    violation_examples.append({
                        "qid": qid,
                        "query": queries[qid],
                        "q_minus": q_minus,
                        "did": did,
                        "s_base": s_base,
                        "s_neg": s_neg,
                        "s_req": s_req,
                        "doc_snippet": doc_text[:200],
                    })
            else:
                cat = "fp_clean"
                q_fp_clean_s_neg.append(s_neg)

            categories[cat]["S_base"].append(s_base)
            categories[cat]["S_req"].append(s_req)
            categories[cat]["S_neg"].append(s_neg)
            categories[cat]["cos"].append(cos_i)
            categories[cat]["count"] += 1

        if q_tp_s_neg and q_fp_viol_s_neg:
            per_query_data.append({
                "qid": qid,
                "cos": cos_i,
                "q_minus": q_minus,
                "tp_s_neg": q_tp_s_neg,
                "fp_viol_s_neg": q_fp_viol_s_neg,
                "tp_s_base": q_tp_s_base,
                "fp_viol_s_base": q_fp_viol_s_base,
            })

    # ===== 全部统计 =====
    logger.info(f"\n{'='*60}")
    logger.info(f"文档分类统计（基于语义：是否包含 q_minus）")
    logger.info(f"{'='*60}")
    print(f"\n{'类别':<25} {'数量':<10} {'S_base均值':<12} {'S_req均值':<12} {'S_neg均值':<12} {'S_neg标准差':<12} {'cos均值':<12}")
    print("-" * 110)

    stats = {}
    for cat, data in categories.items():
        if data["count"] == 0:
            continue
        s_base_arr = np.array(data["S_base"])
        s_req_arr = np.array(data["S_req"])
        s_neg_arr = np.array(data["S_neg"])
        cos_arr = np.array(data["cos"])
        stats[cat] = {
            "count": data["count"],
            "S_base": s_base_arr,
            "S_req": s_req_arr,
            "S_neg": s_neg_arr,
        }
        print(f"{cat:<25} {data['count']:<10} {s_base_arr.mean():<12.4f} {s_req_arr.mean():<12.4f} {s_neg_arr.mean():<12.4f} {s_neg_arr.std():<12.4f} {cos_arr.mean():<12.4f}")

    # ===== 关键对比：TP vs FP_violation =====
    if "true_positive" in stats and "fp_violation" in stats:
        logger.info(f"\n{'='*60}")
        logger.info(f"关键对比：TP vs FP_violation（违反否定约束的假阳性）")
        logger.info(f"{'='*60}")
        tp = stats["true_positive"]
        fp = stats["fp_violation"]
        print(f"\n{'指标':<20} {'TP均值':<15} {'FP_viol均值':<15} {'差异':<15} {'Cohen-d':<12}")
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

        # AUC
        labels = np.concatenate([np.ones(len(fp["S_neg"])), np.zeros(len(tp["S_neg"]))])
        scores = np.concatenate([fp["S_neg"], tp["S_neg"]])
        if len(set(labels)) > 1:
            auc = roc_auc_score(labels, scores)
            print(f"\nAUC (S_neg 区分 FP_violation vs TP): {auc:.4f}")

            # 最佳 F1
            best_f1 = 0
            best_thr = 0
            for thr in np.arange(0.0, 1.0, 0.001):
                pred = (scores > thr).astype(int)
                f1 = f1_score(labels, pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = thr
            print(f"最佳 F1 (S_neg): {best_f1:.4f} (阈值={best_thr:.3f})")

    # ===== 逐 query 分析 =====
    if per_query_data:
        logger.info(f"\n{'='*60}")
        logger.info(f"逐 query 分析：FP_violation vs TP 的 S_neg 差异")
        logger.info(f"{'='*60}")

        diffs_neg = np.array([np.mean(qd["fp_viol_s_neg"]) - np.mean(qd["tp_s_neg"]) for qd in per_query_data])
        diffs_base = np.array([np.mean(qd["fp_viol_s_base"]) - np.mean(qd["tp_s_base"]) for qd in per_query_data])

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

        # 按 cos 分组
        cos_vals = np.array([qd["cos"] for qd in per_query_data])
        cos_median = np.median(cos_vals)
        for label_str, mask in [(f"cos<median({cos_median:.3f})", cos_vals < cos_median),
                                (f"cos>=median({cos_median:.3f})", cos_vals >= cos_median)]:
            sub = [qd for qd, m in zip(per_query_data, mask) if m]
            if len(sub) < 3:
                continue
            diffs_neg = np.array([np.mean(qd["fp_viol_s_neg"]) - np.mean(qd["tp_s_neg"]) for qd in sub])
            diffs_base = np.array([np.mean(qd["fp_viol_s_base"]) - np.mean(qd["tp_s_base"]) for qd in sub])

            print(f"\n  [{label_str}] (n={len(sub)})")
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
    if violation_examples:
        logger.info(f"\n{'='*60}")
        logger.info(f"具体例子：违反否定约束的文档（FP_violation）")
        logger.info(f"{'='*60}")
        for ex in violation_examples:
            print(f"\n--- Query {ex['qid']} ---")
            print(f"  Query: {ex['query']}")
            print(f"  Q_minus: {ex['q_minus']}")
            print(f"  Doc {ex['did']}: S_base={ex['s_base']:.4f}, S_req={ex['s_req']:.4f}, S_neg={ex['s_neg']:.4f}")
            print(f"  Doc snippet: {ex['doc_snippet']}...")

    # ===== cos(Q_base, Q_neg) 分布 =====
    logger.info(f"\n{'='*60}")
    logger.info(f"cos(Q_base, Q_neg) 分布")
    logger.info(f"{'='*60}")
    has_neg_cos = cos_qbase_qneg[np.array(has_neg_list)]
    print(f"\n  has_neg=True (n={len(has_neg_cos)}):")
    print(f"    mean={has_neg_cos.mean():.4f}, std={has_neg_cos.std():.4f}")
    print(f"    min={has_neg_cos.min():.4f}, max={has_neg_cos.max():.4f}")
    print(f"    median={np.median(has_neg_cos):.4f}")
    print(f"    <0.3: {(has_neg_cos < 0.3).float().mean()*100:.1f}%, <0.5: {(has_neg_cos < 0.5).float().mean()*100:.1f}%, <0.7: {(has_neg_cos < 0.7).float().mean()*100:.1f}%")

    logger.info("\n分析完成。")


if __name__ == "__main__":
    main()
