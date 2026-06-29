"""
分析测试集中"假相关文档"的统计特性。

假相关文档定义：
- 在 OG qrels 中相关（S_base 高）
- 在 Changed qrels 中不相关（被指令惩罚）
- 即：表面符合基础查询，但违反指令约束的文档

对比三类文档：
1. 真相关 (True Positive): OG相关 AND Changed相关
2. 假相关 (False Positive): OG相关 BUT Changed不相关  ← 应被惩罚
3. 不相关 (True Negative): OG不相关 AND Changed不相关
4. 新相关 (New Positive): OG不相关 BUT Changed相关  ← 指令要求的
"""

import sys
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 复用实验脚本的路径和加载逻辑
sys.path.insert(0, "/home/luwa/Documents/DSCLR")

DUAL_QUERIES_DIR = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/dual_queries_4B/TSC_BALANCED_t01"
MODEL_NAME = "samaya-ai/RepLLaMA-reproduced"
TASKS = ["Core17InstructionRetrieval", "Robust04InstructionRetrieval", "News21InstructionRetrieval"]


def analyze_task(task_name):
    """分析单个任务的假相关文档特性"""
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

    # 加载 qrels
    data_loader = DataLoader(task_name)
    qrels = data_loader.load_qrels()

    # 加载 dual queries
    dual_data = {}
    with open(dual_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            dual_data[item["qid"]] = item

    # 加载缓存的文档向量
    cache_dir = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/embeddings"
    cached_data = load_cached_embeddings(cache_dir, task_name, MODEL_NAME)
    if cached_data is not None:
        cached_embeddings, cached_doc_ids = cached_data
        engine.retriever.set_embeddings(cached_embeddings, cached_doc_ids)
    else:
        all_doc_ids = engine._get_all_candidate_doc_ids(candidates)
        doc_texts = [corpus[did]["text"] for did in all_doc_ids]
        engine.retriever.index_documents(all_doc_ids, doc_texts, engine.batch_size)

    # 获取文档嵌入
    doc_emb = engine.retriever.doc_embeddings  # (n_docs, dim)
    doc_ids = engine.retriever.doc_ids
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}
    logger.info(f"文档数: {len(doc_ids)}, 维度: {doc_emb.shape[1]}")

    def is_none(text):
        if not text:
            return True
        return str(text).strip().upper() in ("[NONE]", "NONE", "NULL", "N/A", "")

    # 编码查询
    qids = list(q_raw_og.keys())
    q_base_list, q_req_list, q_neg_list = [], [], []
    has_req_list, has_neg_list = [], []

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
        has_req_list.append(bool(q_plus and not is_none(q_plus)))
        has_neg_list.append(bool(q_minus and not is_none(q_minus)))

    # 编码
    logger.info("编码 base 查询...")
    emb_base = engine.encoder.encode_queries(q_base_list, batch_size=64)
    logger.info("编码 req 查询...")
    emb_req = engine.encoder.encode_queries(q_req_list, batch_size=64)
    logger.info("编码 neg 查询...")
    emb_neg = engine.encoder.encode_queries(q_neg_list, batch_size=64)

    # 归一化
    emb_base = F.normalize(emb_base, p=2, dim=1).to(doc_emb.device)
    emb_req = F.normalize(emb_req, p=2, dim=1).to(doc_emb.device)
    emb_neg = F.normalize(emb_neg, p=2, dim=1).to(doc_emb.device)
    doc_emb_norm = F.normalize(doc_emb, p=2, dim=1)

    # 计算相似度
    S_base = torch.matmul(emb_base, doc_emb_norm.T)  # (n_queries, n_docs)
    S_req = torch.matmul(emb_req, doc_emb_norm.T)
    S_neg = torch.matmul(emb_neg, doc_emb_norm.T)

    # Cos(Q_base, Q_neg)
    cos_qbase_qneg = torch.nan_to_num(F.cosine_similarity(emb_base, emb_neg, dim=1), nan=0.0)

    # 分类文档
    # 对每个查询，根据 OG/Changed qrels 分类文档
    # 关键：分别统计 has_neg 和 no_neg 的 query，因为只有 has_neg 的 query
    # 其 FP 文档才可能是因为违反负面约束而被降级
    categories = {
        "true_positive": {"S_base": [], "S_req": [], "S_neg": [], "cos": [], "count": 0, "has_neg": []},
        "false_positive": {"S_base": [], "S_req": [], "S_neg": [], "cos": [], "count": 0, "has_neg": []},
        "true_negative": {"S_base": [], "S_req": [], "S_neg": [], "cos": [], "count": 0, "has_neg": []},
        "new_positive": {"S_base": [], "S_req": [], "S_neg": [], "cos": [], "count": 0, "has_neg": []},
    }

    # 逐 query 收集 FP/TP 的 S_neg，用于 per-query 分析
    per_query_data = []  # list of {qid, has_neg, cos, tp_s_neg_list, fp_s_neg_list, tp_s_base_list, fp_s_base_list}

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

        # 逐 query 收集
        q_tp_s_neg = []
        q_fp_s_neg = []
        q_tp_s_base = []
        q_fp_s_base = []
        q_tp_s_req = []
        q_fp_s_req = []

        # 只分析候选文档 (candidates 键是 base_qid，不带 -og 后缀)
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
            elif in_og and not in_changed:
                cat = "false_positive"  # 假相关：被指令惩罚
            elif not in_og and in_changed:
                cat = "new_positive"
            else:
                cat = "true_negative"

            categories[cat]["S_base"].append(s_base)
            categories[cat]["S_req"].append(s_req)
            categories[cat]["S_neg"].append(s_neg)
            categories[cat]["cos"].append(cos_i)
            categories[cat]["has_neg"].append(q_has_neg)
            categories[cat]["count"] += 1

            # 逐 query 收集
            if cat == "true_positive":
                q_tp_s_neg.append(s_neg)
                q_tp_s_base.append(s_base)
                q_tp_s_req.append(s_req)
            elif cat == "false_positive":
                q_fp_s_neg.append(s_neg)
                q_fp_s_base.append(s_base)
                q_fp_s_req.append(s_req)

        # 只有同时有 FP 和 TP 的 query 才记录
        if q_tp_s_neg and q_fp_s_neg:
            per_query_data.append({
                "qid": base_qid,
                "has_neg": q_has_neg,
                "cos": cos_i,
                "tp_s_neg": q_tp_s_neg,
                "fp_s_neg": q_fp_s_neg,
                "tp_s_base": q_tp_s_base,
                "fp_s_base": q_fp_s_base,
                "tp_s_req": q_tp_s_req,
                "fp_s_req": q_fp_s_req,
            })

    # ===== 逐 query 分析：FP vs TP 的 S_neg 差异 =====
    # 关键：跨 query 聚合会掩盖信号，因为不同 query 的 S_neg 基线不同
    # 逐 query 分析可以消除 query 间变异，暴露真实的 FP vs TP 差异
    if per_query_data:
        logger.info(f"\n{'='*60}")
        logger.info(f"逐 query 分析：FP vs TP 的 S_neg 差异 ({task_name})")
        logger.info(f"{'='*60}")

        # 全部 query
        diffs_neg = []
        diffs_base = []
        diffs_req = []
        for qd in per_query_data:
            tp_mean_neg = np.mean(qd["tp_s_neg"])
            fp_mean_neg = np.mean(qd["fp_s_neg"])
            tp_mean_base = np.mean(qd["tp_s_base"])
            fp_mean_base = np.mean(qd["fp_s_base"])
            tp_mean_req = np.mean(qd["tp_s_req"])
            fp_mean_req = np.mean(qd["fp_s_req"])
            diffs_neg.append(fp_mean_neg - tp_mean_neg)
            diffs_base.append(fp_mean_base - tp_mean_base)
            diffs_req.append(fp_mean_req - tp_mean_req)

        diffs_neg = np.array(diffs_neg)
        diffs_base = np.array(diffs_base)
        diffs_req = np.array(diffs_req)

        print(f"\n  [全部 query] (n={len(per_query_data)})")
        print(f"  {'指标':<25} {'均值差(FP-TP)':<15} {'标准差':<12} {'t统计量':<12} {'p值':<12} {'FP>TP比例':<12}")
        print(f"  {'-'*93}")
        for name, diffs in [("S_neg", diffs_neg), ("S_base", diffs_base), ("S_req", diffs_req)]:
            mean_d = diffs.mean()
            std_d = diffs.std()
            n = len(diffs)
            t_stat = mean_d / (std_d / np.sqrt(n)) if std_d > 1e-8 else 0
            # 单尾 p-value: H1 = FP > TP
            from scipy import stats as sp_stats
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
            diffs_req = np.array([np.mean(qd["fp_s_req"]) - np.mean(qd["tp_s_req"]) for qd in sub])

            print(f"\n  [{label}] (n={len(sub)})")
            print(f"  {'指标':<25} {'均值差(FP-TP)':<15} {'标准差':<12} {'t统计量':<12} {'p值':<12} {'FP>TP比例':<12}")
            print(f"  {'-'*93}")
            for name, diffs in [("S_neg", diffs_neg), ("S_base", diffs_base), ("S_req", diffs_req)]:
                mean_d = diffs.mean()
                std_d = diffs.std()
                n = len(diffs)
                t_stat = mean_d / (std_d / np.sqrt(n)) if std_d > 1e-8 else 0
                p_val = 1 - sp_stats.t.cdf(t_stat, df=n-1)
                fp_gt_tp = (diffs > 0).mean()
                print(f"  {name:<25} {mean_d:<+15.6f} {std_d:<12.6f} {t_stat:<+12.4f} {p_val:<12.6f} {fp_gt_tp:<12.4f}")

        # 按 cos 分组：低 cos vs 高 cos
        # 假设：cos 低时 Q_neg 与 Q_base 正交，S_neg 能区分 FP/TP
        cos_vals = np.array([qd["cos"] for qd in per_query_data])
        cos_median = np.median(cos_vals)
        for label, mask in [(f"cos<median({cos_median:.3f})", cos_vals < cos_median),
                            (f"cos>=median({cos_median:.3f})", cos_vals >= cos_median)]:
            sub = [qd for qd, m in zip(per_query_data, mask) if m]
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

    # ===== 分别分析 has_neg 和 no_neg 的 query =====
    logger.info(f"\n{'='*60}")
    logger.info(f"文档分类统计 ({task_name}) - 全部")
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
            "S_base": {"mean": float(s_base_arr.mean()), "std": float(s_base_arr.std()),
                       "median": float(np.median(s_base_arr)),
                       "p25": float(np.percentile(s_base_arr, 25)),
                       "p75": float(np.percentile(s_base_arr, 75))},
            "S_req": {"mean": float(s_req_arr.mean()), "std": float(s_req_arr.std())},
            "S_neg": {"mean": float(s_neg_arr.mean()), "std": float(s_neg_arr.std()),
                      "median": float(np.median(s_neg_arr)),
                      "p25": float(np.percentile(s_neg_arr, 25)),
                      "p75": float(np.percentile(s_neg_arr, 75)),
                      "p90": float(np.percentile(s_neg_arr, 90)),
                      "p95": float(np.percentile(s_neg_arr, 95))},
            "cos": {"mean": float(cos_arr.mean()), "std": float(cos_arr.std())},
        }

        print(f"{cat:<20} {data['count']:<10} {s_base_arr.mean():<12.4f} {s_req_arr.mean():<12.4f} {s_neg_arr.mean():<12.4f} {s_neg_arr.std():<12.4f} {cos_arr.mean():<12.4f}")

    # ===== 关键：按 has_neg 分组分析 =====
    # 只有 has_neg=True 的 query，其 FP 文档才可能是因为违反负面约束被降级
    # 对于 has_neg=False 的 query，FP 被降级的原因与 S_neg 无关
    logger.info(f"\n{'='*60}")
    logger.info(f"按 has_neg 分组分析 ({task_name})")
    logger.info(f"{'='*60}")

    has_neg_stats = {}
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

        has_neg_stats[label] = sub_cats

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
            from sklearn.metrics import roc_auc_score
            labels = np.concatenate([np.ones(len(fp_data["S_neg"])), np.zeros(len(tp_data["S_neg"]))])
            scores = np.concatenate([fp_data["S_neg"], tp_data["S_neg"]])
            if len(set(labels)) > 1:
                auc = roc_auc_score(labels, scores)
                print(f"  AUC (S_neg): {auc:.4f}")

    # 关键分析：假相关文档 vs 真相关文档的区别 (全部)
    if "false_positive" in stats and "true_positive" in stats:
        logger.info(f"\n{'='*60}")
        logger.info(f"关键对比：假相关 vs 真相关 (全部)")
        logger.info(f"{'='*60}")
        fp = stats["false_positive"]
        tp = stats["true_positive"]
        print(f"\n{'指标':<20} {'真相关(TP)':<25} {'假相关(FP)':<25} {'差异':<15}")
        print("-" * 85)
        for metric in ["S_base", "S_req", "S_neg"]:
            tp_m = tp[metric]["mean"]
            fp_m = fp[metric]["mean"]
            diff = fp_m - tp_m
            print(f"{metric+'均值':<20} {tp_m:<25.4f} {fp_m:<25.4f} {diff:<+15.4f}")

        # S_neg 的分布对比
        print(f"\n{'S_neg分布':<20} {'真相关(TP)':<25} {'假相关(FP)':<25}")
        print("-" * 70)
        for p in ["p25", "median", "p75", "p90", "p95"]:
            tp_p = tp["S_neg"][p]
            fp_p = fp["S_neg"][p]
            print(f"  {p:<18} {tp_p:<25.4f} {fp_p:<25.4f}")

    # 关键分析：假相关文档 vs 不相关文档的区别
    if "false_positive" in stats and "true_negative" in stats:
        logger.info(f"\n{'='*60}")
        logger.info(f"关键对比：假相关 vs 不相关")
        logger.info(f"{'='*60}")
        fp = stats["false_positive"]
        tn = stats["true_negative"]
        print(f"\n{'指标':<20} {'不相关(TN)':<25} {'假相关(FP)':<25} {'差异':<15}")
        print("-" * 85)
        for metric in ["S_base", "S_req", "S_neg"]:
            tn_m = tn[metric]["mean"]
            fp_m = fp[metric]["mean"]
            diff = fp_m - tn_m
            print(f"{metric+'均值':<20} {tn_m:<25.4f} {fp_m:<25.4f} {diff:<+15.4f}")

    # 分析假相关文档的 S_neg vs S_base 关系
    if categories["false_positive"]["count"] > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"假相关文档的 S_neg vs S_base 关系")
        logger.info(f"{'='*60}")
        fp_s_base = np.array(categories["false_positive"]["S_base"])
        fp_s_neg = np.array(categories["false_positive"]["S_neg"])
        fp_s_req = np.array(categories["false_positive"]["S_req"])

        # S_neg - S_base
        diff_neg_base = fp_s_neg - fp_s_base
        # S_neg / S_base
        ratio_neg_base = fp_s_neg / (fp_s_base + 1e-8)
        # S_neg - cos
        fp_cos = np.array(categories["false_positive"]["cos"])
        diff_neg_cos = fp_s_neg - fp_cos

        print(f"\n{'指标':<30} {'均值':<12} {'标准差':<12} {'中位数':<12}")
        print("-" * 66)
        print(f"{'S_neg - S_base':<30} {diff_neg_base.mean():<12.4f} {diff_neg_base.std():<12.4f} {np.median(diff_neg_base):<12.4f}")
        print(f"{'S_neg / S_base':<30} {ratio_neg_base.mean():<12.4f} {ratio_neg_base.std():<12.4f} {np.median(ratio_neg_base):<12.4f}")
        print(f"{'S_neg - cos(Q,Q_neg)':<30} {diff_neg_cos.mean():<12.4f} {diff_neg_cos.std():<12.4f} {np.median(diff_neg_cos):<12.4f}")
        print(f"{'S_neg':<30} {fp_s_neg.mean():<12.4f} {fp_s_neg.std():<12.4f} {np.median(fp_s_neg):<12.4f}")
        print(f"{'S_base':<30} {fp_s_base.mean():<12.4f} {fp_s_base.std():<12.4f} {np.median(fp_s_base):<12.4f}")
        print(f"{'cos(Q,Q_neg)':<30} {fp_cos.mean():<12.4f} {fp_cos.std():<12.4f} {np.median(fp_cos):<12.4f}")

    # ===== 新增：深入分析 FP vs TP 的可区分性 =====
    if categories["false_positive"]["count"] > 0 and categories["true_positive"]["count"] > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"深入分析：FP vs TP 的可区分性")
        logger.info(f"{'='*60}")

        fp_s_base = np.array(categories["false_positive"]["S_base"])
        fp_s_neg = np.array(categories["false_positive"]["S_neg"])
        fp_s_req = np.array(categories["false_positive"]["S_req"])
        fp_cos = np.array(categories["false_positive"]["cos"])

        tp_s_base = np.array(categories["true_positive"]["S_base"])
        tp_s_neg = np.array(categories["true_positive"]["S_neg"])
        tp_s_req = np.array(categories["true_positive"]["S_req"])
        tp_cos = np.array(categories["true_positive"]["cos"])

        # 各种组合特征
        print(f"\n{'特征':<35} {'TP均值':<12} {'FP均值':<12} {'差异':<12} {'Cohen-d':<12}")
        print("-" * 85)

        features = {
            "S_base": (tp_s_base, fp_s_base),
            "S_req": (tp_s_req, fp_s_req),
            "S_neg": (tp_s_neg, fp_s_neg),
            "cos(Q,Q_neg)": (tp_cos, fp_cos),
            "S_neg - S_base": (tp_s_neg - tp_s_base, fp_s_neg - fp_s_base),
            "S_neg / S_base": (tp_s_neg / (tp_s_base+1e-8), fp_s_neg / (fp_s_base+1e-8)),
            "S_neg - cos": (tp_s_neg - tp_cos, fp_s_neg - fp_cos),
            "S_base - S_neg": (tp_s_base - tp_s_neg, fp_s_base - fp_s_neg),
            "S_base - S_req": (tp_s_base - tp_s_req, fp_s_base - fp_s_req),
            "S_req - S_neg": (tp_s_req - tp_s_neg, fp_s_req - fp_s_neg),
            "(S_base+S_req)/2 - S_neg": ((tp_s_base+tp_s_req)/2 - tp_s_neg, (fp_s_base+fp_s_req)/2 - fp_s_neg),
            "S_base * (1 - S_neg)": (tp_s_base * (1 - tp_s_neg), fp_s_base * (1 - fp_s_neg)),
        }

        for name, (tp_vals, fp_vals) in features.items():
            tp_m, tp_s = tp_vals.mean(), tp_vals.std()
            fp_m, fp_s = fp_vals.mean(), fp_vals.std()
            diff = fp_m - tp_m
            pooled_std = np.sqrt((tp_s**2 + fp_s**2) / 2)
            cohen_d = diff / pooled_std if pooled_std > 1e-8 else 0
            print(f"{name:<35} {tp_m:<12.4f} {fp_m:<12.4f} {diff:<+12.4f} {cohen_d:<+12.4f}")

        # S_neg 分布重叠分析
        logger.info(f"\n{'='*60}")
        logger.info(f"S_neg 分布重叠分析 (FP vs TP)")
        logger.info(f"{'='*60}")
        print(f"\n{'分位数':<10} {'TP':<12} {'FP':<12} {'差异':<12}")
        print("-" * 46)
        for p in [10, 25, 50, 75, 90, 95, 99]:
            tp_p = np.percentile(tp_s_neg, p)
            fp_p = np.percentile(fp_s_neg, p)
            print(f"{p}th{'':<6} {tp_p:<12.4f} {fp_p:<12.4f} {fp_p-tp_p:<+12.4f}")

        # 如果用 S_neg 阈值区分 FP/TP，最佳 F1 是多少？
        logger.info(f"\n{'='*60}")
        logger.info(f"用 S_neg 阈值区分 FP/TP 的理论上限")
        logger.info(f"{'='*60}")
        from sklearn.metrics import roc_auc_score, f1_score
        labels = np.concatenate([np.ones(len(fp_s_neg)), np.zeros(len(tp_s_neg))])
        scores = np.concatenate([fp_s_neg, tp_s_neg])
        auc = roc_auc_score(labels, scores)
        print(f"\nAUC (S_neg 区分 FP vs TP): {auc:.4f}")
        # 最佳 F1
        best_f1 = 0
        best_thr = 0
        for thr in np.arange(0.3, 0.8, 0.001):
            pred = (scores > thr).astype(int)
            f1 = f1_score(labels, pred)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        print(f"最佳 F1: {best_f1:.4f} (阈值={best_thr:.3f})")

        # 同样分析 S_base - S_neg
        labels2 = np.concatenate([np.ones(len(fp_s_neg)), np.zeros(len(tp_s_neg))])
        scores2 = np.concatenate([fp_s_base - fp_s_neg, tp_s_base - tp_s_neg])
        auc2 = roc_auc_score(labels2, scores2)
        print(f"\nAUC (S_base-S_neg 区分 FP vs TP): {auc2:.4f}")

        # 分析: 假相关文档 vs 不相关文档
        logger.info(f"\n{'='*60}")
        logger.info(f"用 S_base 区分 FP vs TN 的理论上限")
        logger.info(f"{'='*60}")
        tn_s_base = np.array(categories["true_negative"]["S_base"])
        tn_s_neg = np.array(categories["true_negative"]["S_neg"])
        labels3 = np.concatenate([np.ones(len(fp_s_base)), np.zeros(len(tn_s_base))])
        scores3 = np.concatenate([fp_s_base, tn_s_base])
        auc3 = roc_auc_score(labels3, scores3)
        print(f"\nAUC (S_base 区分 FP vs TN): {auc3:.4f}")

    return stats


def main():
    all_stats = {}
    for task in TASKS:
        stats = analyze_task(task)
        all_stats[task] = stats

    # 汇总
    logger.info(f"\n{'='*80}")
    logger.info(f"汇总分析")
    logger.info(f"{'='*80}")

    print(f"\n{'任务':<35} {'类别':<20} {'数量':<8} {'S_base':<10} {'S_neg':<10} {'S_neg-S_base':<14}")
    print("-" * 100)
    for task, stats in all_stats.items():
        for cat, data in stats.items():
            s_base = data["S_base"]["mean"]
            s_neg = data["S_neg"]["mean"]
            diff = s_neg - s_base
            print(f"{task:<35} {cat:<20} {data['count']:<8} {s_base:<10.4f} {s_neg:<10.4f} {diff:<+14.4f}")

    # 保存结果
    output_path = "/home/luwa/Documents/DSCLR/results/false_positive_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
