"""
分析 NegConstraint 测试集中 FP vs TP 的可区分性。

NegConstraint 数据结构（与 FollowIR 不同）：
- 每个查询只有 1 个正例文档（qrels score=1）
- 没有显式负例标签
- 查询包含显式否定约束（如 "don't mention Moses"）
- q_minus 是具体实体（如 "Moses", "Denmark"）

文档分类定义：
- TP (True Positive): qrels 中的正例文档（相关，不违反否定约束）
- FP (False Positive): 不在 qrels 但 S_base 排名靠前的文档（表面匹配 base query）
  - 其中 S_neg 高的是"应被否定的假阳性"（违反否定约束）
  - 其中 S_neg 低的是"普通不相关文档"
- TN (True Negative): 不在 qrels 且 S_base 低的文档

关键问题：S_neg 能否区分 TP 和 FP？
假设：NegConstraint 中否定信号显式且强（q_minus 是具体实体），
     TP 应有低 S_neg（不提及被排除的主题），FP 应有高 S_neg（提及被排除的主题）。
"""

import os
import sys
import json
import csv
import torch
import torch.nn.functional as F
import numpy as np
import logging
from collections import defaultdict
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, "/home/luwa/Documents/DSCLR")

DATA_DIR = "dataset/NegConstraint/NegConstraint"
DUAL_QUERIES_PATH = "dataset/NegConstraint/NegConstraint/dual_queries/NegConstraint_TSC_BALANCED_t01.jsonl"
MODEL_NAME = "samaya-ai/RepLLaMA-reproduced"
TOP_K = 100  # 分析 top-k 候选文档


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


def main():
    from eval.models.repllama_encoder import RepLLaMAEncoder

    logger.info("=" * 60)
    logger.info("NegConstraint FP vs TP 分析")
    logger.info("=" * 60)

    corpus, queries, qrels, dual_data = load_data()
    eval_qids = sorted(set(qrels.keys()) & set(queries.keys()))
    logger.info(f"Corpus: {len(corpus)} docs, Queries: {len(queries)}, Eval queries: {len(eval_qids)}")

    # 加载编码器
    encoder = RepLLaMAEncoder(model_name=MODEL_NAME, device="cuda", batch_size=64)

    # 加载或编码文档
    cache_path = "dataset/NegConstraint/NegConstraint/embeddings/negconstraint_repllama_corpus.pt"
    if os.path.exists(cache_path):
        logger.info(f"Loading cached corpus embeddings from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        doc_ids = cache["doc_ids"]
        doc_embeddings = cache["embeddings"]
        logger.info(f"Cached: {len(doc_ids)} docs, shape={doc_embeddings.shape}")
    else:
        doc_ids = sorted(corpus.keys())
        doc_texts = [corpus[did] for did in doc_ids]
        logger.info(f"Encoding {len(doc_texts)} documents...")
        doc_embeddings = encoder.encode_documents(doc_texts, batch_size=64)
        torch.save({"doc_ids": doc_ids, "embeddings": doc_embeddings.cpu().float()}, cache_path)

    doc_embeddings = F.normalize(doc_embeddings.float(), p=2, dim=1).to('cuda')
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    # 准备查询
    q_base_list, q_req_list, q_neg_list = [], [], []
    has_req_list, has_neg_list = [], []

    for qid in eval_qids:
        q_base_list.append(queries[qid])
        dual = dual_data.get(qid, {})
        q_plus = dual.get('q_plus', '')
        q_minus = dual.get('q_minus', '')

        q_req_list.append(q_plus if q_plus and not is_none_query(q_plus) else "")
        q_neg_list.append(q_minus if q_minus and not is_none_query(q_minus) else "")
        has_req_list.append(bool(q_plus and not is_none_query(q_plus)))
        has_neg_list.append(bool(q_minus and not is_none_query(q_minus)))

    logger.info(f"Q_plus available: {sum(has_req_list)}/{len(has_req_list)}")
    logger.info(f"Q_minus available: {sum(has_neg_list)}/{len(has_neg_list)}")

    # 编码查询
    logger.info("Encoding Q_base...")
    emb_base = F.normalize(encoder.encode_queries(q_base_list, batch_size=64).float(), p=2, dim=1).to('cuda')
    logger.info("Encoding Q_req...")
    emb_req = F.normalize(encoder.encode_queries(q_req_list, batch_size=64).float(), p=2, dim=1).to('cuda')
    logger.info("Encoding Q_neg...")
    emb_neg = F.normalize(encoder.encode_queries(q_neg_list, batch_size=64).float(), p=2, dim=1).to('cuda')

    # 计算相似度
    S_base = torch.matmul(emb_base, doc_embeddings.T).cpu()  # (n_queries, n_docs)
    S_req = torch.matmul(emb_req, doc_embeddings.T).cpu()
    S_neg = torch.matmul(emb_neg, doc_embeddings.T).cpu()
    cos_qbase_qneg = torch.nan_to_num(F.cosine_similarity(emb_base, emb_neg, dim=1).cpu(), nan=0.0)

    logger.info(f"S_base shape: {S_base.shape}")
    logger.info(f"cos(Q_base, Q_neg) stats: mean={cos_qbase_qneg.mean():.4f}, std={cos_qbase_qneg.std():.4f}")

    # ===== 分类文档 =====
    # TP: qrels 中的正例文档
    # FP: 不在 qrels 但在 top-k by S_base 的文档
    # TN: 不在 qrels 且不在 top-k by S_base 的文档
    categories = {
        "true_positive": {"S_base": [], "S_req": [], "S_neg": [], "cos": [], "has_neg": [], "count": 0},
        "false_positive": {"S_base": [], "S_req": [], "S_neg": [], "cos": [], "has_neg": [], "count": 0},
        "true_negative": {"S_base": [], "S_req": [], "S_neg": [], "cos": [], "has_neg": [], "count": 0},
    }

    # 逐 query 收集，用于 per-query 分析
    per_query_data = []

    for i, qid in enumerate(eval_qids):
        relevant = set(d for d, r in qrels.get(qid, {}).items() if r > 0)
        cos_i = cos_qbase_qneg[i].item()
        q_has_neg = has_neg_list[i]

        # 获取 top-k 文档索引
        scores_i = S_base[i]
        k = min(TOP_K, len(scores_i))
        topk_vals, topk_idxs = torch.topk(scores_i, k)

        q_tp_s_neg = []
        q_fp_s_neg = []
        q_tp_s_base = []
        q_fp_s_base = []

        for j in range(k):
            doc_idx = topk_idxs[j].item()
            did = doc_ids[doc_idx]
            s_base = topk_vals[j].item()
            s_req = S_req[i, doc_idx].item()
            s_neg = S_neg[i, doc_idx].item()

            if did in relevant:
                cat = "true_positive"
                q_tp_s_neg.append(s_neg)
                q_tp_s_base.append(s_base)
            else:
                cat = "false_positive"  # top-k 但不在 qrels
                q_fp_s_neg.append(s_neg)
                q_fp_s_base.append(s_base)

            categories[cat]["S_base"].append(s_base)
            categories[cat]["S_req"].append(s_req)
            categories[cat]["S_neg"].append(s_neg)
            categories[cat]["cos"].append(cos_i)
            categories[cat]["has_neg"].append(q_has_neg)
            categories[cat]["count"] += 1

        # TN: 不在 top-k 的文档（采样以控制数量）
        non_topk_mask = torch.ones(len(scores_i), dtype=torch.bool)
        non_topk_mask[topk_idxs] = False
        non_topk_indices = non_topk_mask.nonzero(as_tuple=True)[0]
        # 采样 50 个 TN
        if len(non_topk_indices) > 50:
            sample_indices = non_topk_indices[torch.randperm(len(non_topk_indices))[:50]]
        else:
            sample_indices = non_topk_indices

        for doc_idx in sample_indices:
            did = doc_ids[doc_idx.item()]
            if did in relevant:
                continue
            s_base = S_base[i, doc_idx].item()
            s_req = S_req[i, doc_idx].item()
            s_neg = S_neg[i, doc_idx].item()
            categories["true_negative"]["S_base"].append(s_base)
            categories["true_negative"]["S_req"].append(s_req)
            categories["true_negative"]["S_neg"].append(s_neg)
            categories["true_negative"]["cos"].append(cos_i)
            categories["true_negative"]["has_neg"].append(q_has_neg)
            categories["true_negative"]["count"] += 1

        # per-query 数据
        if q_tp_s_neg and q_fp_s_neg:
            per_query_data.append({
                "qid": qid,
                "has_neg": q_has_neg,
                "cos": cos_i,
                "tp_s_neg": q_tp_s_neg,
                "fp_s_neg": q_fp_s_neg,
                "tp_s_base": q_tp_s_base,
                "fp_s_base": q_fp_s_base,
            })

    # ===== 全部统计 =====
    logger.info(f"\n{'='*60}")
    logger.info(f"文档分类统计 (全部)")
    logger.info(f"{'='*60}")
    print(f"\n{'类别':<20} {'数量':<10} {'S_base均值':<12} {'S_req均值':<12} {'S_neg均值':<12} {'S_neg标准差':<12} {'cos均值':<12}")
    print("-" * 100)

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
            "S_base": {"mean": float(s_base_arr.mean()), "std": float(s_base_arr.std())},
            "S_req": {"mean": float(s_req_arr.mean()), "std": float(s_req_arr.std())},
            "S_neg": {"mean": float(s_neg_arr.mean()), "std": float(s_neg_arr.std())},
            "cos": {"mean": float(cos_arr.mean()), "std": float(cos_arr.std())},
        }
        print(f"{cat:<20} {data['count']:<10} {s_base_arr.mean():<12.4f} {s_req_arr.mean():<12.4f} {s_neg_arr.mean():<12.4f} {s_neg_arr.std():<12.4f} {cos_arr.mean():<12.4f}")

    # ===== 按 has_neg 分组 =====
    logger.info(f"\n{'='*60}")
    logger.info(f"按 has_neg 分组分析")
    logger.info(f"{'='*60}")

    for neg_flag, label in [(True, "has_neg=True"), (False, "has_neg=False")]:
        sub_cats = {}
        for cat in categories:
            arr = np.array(categories[cat]["has_neg"])
            mask = arr == neg_flag
            if mask.sum() == 0:
                continue
            sub_cats[cat] = {
                "count": int(mask.sum()),
                "S_base": np.array(categories[cat]["S_base"])[mask],
                "S_req": np.array(categories[cat]["S_req"])[mask],
                "S_neg": np.array(categories[cat]["S_neg"])[mask],
                "cos": np.array(categories[cat]["cos"])[mask],
            }

        print(f"\n  [{label}]")
        print(f"  {'类别':<20} {'数量':<10} {'S_base':<12} {'S_req':<12} {'S_neg':<12} {'cos':<12}")
        print(f"  {'-'*76}")
        for cat, data in sub_cats.items():
            print(f"  {cat:<20} {data['count']:<10} {data['S_base'].mean():<12.4f} {data['S_req'].mean():<12.4f} {data['S_neg'].mean():<12.4f} {data['cos'].mean():<12.4f}")

        # FP vs TP 对比
        if "false_positive" in sub_cats and "true_positive" in sub_cats:
            fp_data = sub_cats["false_positive"]
            tp_data = sub_cats["true_positive"]
            print(f"\n  [{label}] FP vs TP:")
            print(f"  {'指标':<20} {'TP均值':<12} {'FP均值':<12} {'差异':<12} {'Cohen-d':<12}")
            print(f"  {'-'*68}")
            for metric in ["S_base", "S_req", "S_neg"]:
                tp_m = tp_data[metric].mean()
                fp_m = fp_data[metric].mean()
                tp_s = tp_data[metric].std()
                fp_s = fp_data[metric].std()
                diff = fp_m - tp_m
                pooled = np.sqrt((tp_s**2 + fp_s**2) / 2)
                d = diff / pooled if pooled > 1e-8 else 0
                print(f"  {metric:<20} {tp_m:<12.4f} {fp_m:<12.4f} {diff:<+12.4f} {d:<+12.4f}")

            # AUC
            labels = np.concatenate([np.ones(len(fp_data["S_neg"])), np.zeros(len(tp_data["S_neg"]))])
            scores = np.concatenate([fp_data["S_neg"], tp_data["S_neg"]])
            if len(set(labels)) > 1:
                auc = roc_auc_score(labels, scores)
                print(f"  AUC (S_neg 区分 FP vs TP): {auc:.4f}")

            # 最佳 F1
            best_f1 = 0
            best_thr = 0
            for thr in np.arange(0.0, 1.0, 0.001):
                pred = (scores > thr).astype(int)
                f1 = f1_score(labels, pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = thr
            print(f"  最佳 F1 (S_neg): {best_f1:.4f} (阈值={best_thr:.3f})")

    # ===== 逐 query 分析 =====
    if per_query_data:
        logger.info(f"\n{'='*60}")
        logger.info(f"逐 query 分析：FP vs TP 的 S_neg 差异")
        logger.info(f"{'='*60}")

        # 全部 query
        diffs_neg = []
        diffs_base = []
        for qd in per_query_data:
            tp_mean_neg = np.mean(qd["tp_s_neg"])
            fp_mean_neg = np.mean(qd["fp_s_neg"])
            tp_mean_base = np.mean(qd["tp_s_base"])
            fp_mean_base = np.mean(qd["fp_s_base"])
            diffs_neg.append(fp_mean_neg - tp_mean_neg)
            diffs_base.append(fp_mean_base - tp_mean_base)

        diffs_neg = np.array(diffs_neg)
        diffs_base = np.array(diffs_base)

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

        # 按 has_neg 分组
        for neg_flag, label in [(True, "has_neg=True"), (False, "has_neg=False")]:
            sub = [qd for qd in per_query_data if qd["has_neg"] == neg_flag]
            if len(sub) < 3:
                continue
            diffs_neg = np.array([np.mean(qd["fp_s_neg"]) - np.mean(qd["tp_s_neg"]) for qd in sub])
            diffs_base = np.array([np.mean(qd["fp_s_base"]) - np.mean(qd["tp_s_base"]) for qd in sub])

            print(f"\n  [{label}] (n={len(sub)})")
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
            diffs_neg = np.array([np.mean(qd["fp_s_neg"]) - np.mean(qd["tp_s_neg"]) for qd in sub])
            diffs_base = np.array([np.mean(qd["fp_s_base"]) - np.mean(qd["tp_s_base"]) for qd in sub])

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

    # ===== FP vs TN 对比（验证 S_base 区分能力）=====
    if "false_positive" in stats and "true_negative" in stats:
        logger.info(f"\n{'='*60}")
        logger.info(f"FP vs TN 对比（验证 S_base 区分能力）")
        logger.info(f"{'='*60}")
        fp = stats["false_positive"]
        tn = stats["true_negative"]
        print(f"\n{'指标':<20} {'TN均值':<25} {'FP均值':<25} {'差异':<15}")
        print("-" * 85)
        for metric in ["S_base", "S_req", "S_neg"]:
            tn_m = tn[metric]["mean"]
            fp_m = fp[metric]["mean"]
            diff = fp_m - tn_m
            print(f"{metric+'均值':<20} {tn_m:<25.4f} {fp_m:<25.4f} {diff:<+15.4f}")

    # ===== 关键分析：将 FP 细分为"高S_neg FP"（违反否定约束）vs"低S_neg FP" =====
    # 在 FollowIR 中，FP = 相关于OG但不相关于Changed（违反否定约束）
    # 在 NegConstraint 中，我们用 S_neg 高低来近似"是否违反否定约束"
    logger.info(f"\n{'='*60}")
    logger.info(f"关键分析：TP vs 高S_neg FP vs 低S_neg FP")
    logger.info(f"（高S_neg FP = 违反否定约束的假阳性，最接近 FollowIR 的 FP 定义）")
    logger.info(f"{'='*60}")

    # 只分析 has_neg=True 的 query
    for neg_flag, label in [(True, "has_neg=True")]:
        tp_s_neg_all = []
        tp_s_base_all = []
        fp_high_s_neg = []  # 违反否定约束的 FP
        fp_high_s_base = []
        fp_low_s_neg = []   # 普通不相关 FP
        fp_low_s_base = []

        for i, qid in enumerate(eval_qids):
            if not has_neg_list[i]:
                continue
            relevant = set(d for d, r in qrels.get(qid, {}).items() if r > 0)
            scores_i = S_base[i]
            k = min(TOP_K, len(scores_i))
            topk_vals, topk_idxs = torch.topk(scores_i, k)

            # 用中位数作为 S_neg 高低分界
            q_s_neg_vals = []
            for j in range(k):
                doc_idx = topk_idxs[j].item()
                q_s_neg_vals.append(S_neg[i, doc_idx].item())
            q_s_neg_median = np.median(q_s_neg_vals) if q_s_neg_vals else 0

            for j in range(k):
                doc_idx = topk_idxs[j].item()
                did = doc_ids[doc_idx]
                s_base = topk_vals[j].item()
                s_neg = S_neg[i, doc_idx].item()

                if did in relevant:
                    tp_s_neg_all.append(s_neg)
                    tp_s_base_all.append(s_base)
                else:
                    if s_neg >= q_s_neg_median:
                        fp_high_s_neg.append(s_neg)
                        fp_high_s_base.append(s_base)
                    else:
                        fp_low_s_neg.append(s_neg)
                        fp_low_s_base.append(s_base)

        tp_s_neg_arr = np.array(tp_s_neg_all)
        tp_s_base_arr = np.array(tp_s_base_all)
        fp_high_arr = np.array(fp_high_s_neg)
        fp_high_base_arr = np.array(fp_high_s_base)
        fp_low_arr = np.array(fp_low_s_neg)
        fp_low_base_arr = np.array(fp_low_s_base)

        print(f"\n  [{label}]")
        print(f"  {'类别':<25} {'数量':<10} {'S_base均值':<12} {'S_neg均值':<12} {'S_neg标准差':<12}")
        print(f"  {'-'*71}")
        print(f"  {'TP (真相关)':<25} {len(tp_s_neg_arr):<10} {tp_s_base_arr.mean():<12.4f} {tp_s_neg_arr.mean():<12.4f} {tp_s_neg_arr.std():<12.4f}")
        print(f"  {'FP_high (违反否定约束)':<25} {len(fp_high_arr):<10} {fp_high_base_arr.mean():<12.4f} {fp_high_arr.mean():<12.4f} {fp_high_arr.std():<12.4f}")
        print(f"  {'FP_low (普通不相关)':<25} {len(fp_low_arr):<10} {fp_low_base_arr.mean():<12.4f} {fp_low_arr.mean():<12.4f} {fp_low_arr.std():<12.4f}")

        # TP vs FP_high（最接近 FollowIR 的 FP 定义）
        print(f"\n  [{label}] TP vs FP_high (违反否定约束):")
        print(f"  {'指标':<20} {'TP均值':<12} {'FP_high均值':<12} {'差异':<12} {'Cohen-d':<12}")
        print(f"  {'-'*68}")
        for name, tp_vals, fp_vals in [("S_base", tp_s_base_arr, fp_high_base_arr), ("S_neg", tp_s_neg_arr, fp_high_arr)]:
            tp_m = tp_vals.mean()
            fp_m = fp_vals.mean()
            tp_s = tp_vals.std()
            fp_s = fp_vals.std()
            diff = fp_m - tp_m
            pooled = np.sqrt((tp_s**2 + fp_s**2) / 2)
            d = diff / pooled if pooled > 1e-8 else 0
            print(f"  {name:<20} {tp_m:<12.4f} {fp_m:<12.4f} {diff:<+12.4f} {d:<+12.4f}")

        # AUC
        labels = np.concatenate([np.ones(len(fp_high_arr)), np.zeros(len(tp_s_neg_arr))])
        scores = np.concatenate([fp_high_arr, tp_s_neg_arr])
        if len(set(labels)) > 1:
            auc = roc_auc_score(labels, scores)
            print(f"  AUC (S_neg 区分 FP_high vs TP): {auc:.4f}")

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
