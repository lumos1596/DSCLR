"""细粒度分析 safe-anchor 导致 MAP 下降的原因。
对比 V5 (cos+δ) vs V6 (safe-anchor) 在 Core17 上的逐 query AP 变化，
关联阈值和惩罚信息。
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics.evaluator import FollowIREvaluator


def compute_ap_at_k(ranked_doc_ids, qrel_for_query, k=1000):
    """计算 AP@k (Average Precision)。
    qrel_for_query: [doc_id, ...] (均相关) 或 {doc_id: score}。
    """
    if isinstance(qrel_for_query, dict):
        relevant = {d for d, s in qrel_for_query.items() if s > 0}
    else:
        relevant = set(qrel_for_query)
    if not relevant:
        return None  # 无相关文档，跳过
    hits = 0
    precision_sum = 0.0
    for i, did in enumerate(ranked_doc_ids[:k]):
        if did in relevant:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / len(relevant)


def main():
    task = "Core17InstructionRetrieval"
    # 1. 加载 V5 / V6 ranking
    v5_rank_path = "evaluation/deir_dual_v2/Core17InstructionRetrieval/ranking_changed.json"
    v6_rank_path = "results/safe_anchor/Core17InstructionRetrieval/ranking_changed.json"
    with open(v5_rank_path) as f:
        v5_ch = json.load(f)
    with open(v6_rank_path) as f:
        v6_ch = json.load(f)

    # 2. 加载 V6 debug 阈值信息
    with open("results/safe_anchor/Core17InstructionRetrieval/debug_anchor_logs.json") as f:
        v6_debug = {r["query_id"]: r for r in json.load(f)}

    # 3. 加载 qrels (changed)
    evaluator = FollowIREvaluator(task)
    changed_qrels = evaluator.data_loader.load_qrel_diff()

    # 4. 逐 query 计算 AP@1000
    # ranking key: '310-changed' -> qrel key: '310'
    rows = []
    all_qids = sorted(set(v5_ch.keys()) | set(v6_ch.keys()))
    v5_aps, v6_aps = [], []
    for qid in all_qids:
        qrel_key = qid.replace("-changed", "").replace("-og", "")
        qrel = changed_qrels.get(qrel_key, [])
        if not qrel:
            continue
        v5_rank = sorted(v5_ch.get(qid, {}).items(), key=lambda x: -x[1])
        v6_rank = sorted(v6_ch.get(qid, {}).items(), key=lambda x: -x[1])
        v5_ap = compute_ap_at_k([d for d, _ in v5_rank], qrel)
        v6_ap = compute_ap_at_k([d for d, _ in v6_rank], qrel)
        if v5_ap is None or v6_ap is None:
            continue
        dbg = v6_debug.get(qid, {})
        rows.append({
            "qid": qid,
            "v5_ap": v5_ap,
            "v6_ap": v6_ap,
            "delta_ap": v6_ap - v5_ap,
            "rel_count": len(qrel) if isinstance(qrel, list) else sum(1 for s in qrel.values() if s > 0),
            "tau_anchor": dbg.get("tau_anchor"),
            "cos_qbase_qneg": dbg.get("cos_qbase_qneg"),
            "v6_threshold": dbg.get("threshold_base_used"),
            "v5_threshold": (dbg.get("cos_qbase_qneg") or 0) + 0.02,  # V5: cos+0.02
            "num_penalized": dbg.get("num_penalized_docs"),
            "cand_s_neg_max": dbg.get("candidate_s_neg_max"),
            "has_anchor": qid in v6_debug,
            "q_neg": dbg.get("q_neg", "")[:60],
        })
        v5_aps.append(v5_ap)
        v6_aps.append(v6_ap)

    # 5. 汇总
    print("=" * 90)
    print(f"Core17 逐 query AP 对比 (V5 cos+δ vs V6 safe-anchor)")
    print(f"  V5 参数: α=0.72, β=1.32, δ=0.02  |  V6 参数: α=0.99, β=1.96, ad=-0.05")
    print(f"  changed queries: {len(rows)} 个")
    print(f"  V5 MAP@1000 = {sum(v5_aps)/len(v5_aps):.4f}")
    print(f"  V6 MAP@1000 = {sum(v6_aps)/len(v6_aps):.4f}")
    print(f"  变化 = {(sum(v6_aps)-sum(v5_aps))/len(v5_aps)*100:+.1f}%")
    print()

    # 6. 按 delta_ap 排序
    rows_sorted = sorted(rows, key=lambda x: x["delta_ap"])

    print("=" * 90)
    print("【AP 下降最严重的 query (top 8)】")
    print(f"{'qid':<14} {'V5_AP':>7} {'V6_AP':>7} {'ΔAP':>7} {'rel#':>4} | {'τ_anchor':>9} {'cos(Qb,Qn)':>11} {'V5_τ':>7} {'V6_τ':>7} {'pen#':>5} | q_neg")
    print("-" * 110)
    for r in rows_sorted[:8]:
        ta = f"{r['tau_anchor']:.4f}" if r['tau_anchor'] is not None else "  N/A"
        cn = f"{r['cos_qbase_qneg']:.4f}" if r['cos_qbase_qneg'] is not None else "  N/A"
        v5t = f"{r['v5_threshold']:.4f}" if r['cos_qbase_qneg'] is not None else "  N/A"
        v6t = f"{r['v6_threshold']:.4f}" if r['v6_threshold'] is not None else "  N/A"
        print(f"{r['qid']:<14} {r['v5_ap']:>7.4f} {r['v6_ap']:>7.4f} {r['delta_ap']:>+7.4f} {r['rel_count']:>4} | {ta:>9} {cn:>11} {v5t:>7} {v6t:>7} {r['num_penalized'] or 0:>5} | {r['q_neg']}")

    print()
    print("=" * 90)
    print("【AP 上升的 query】")
    print(f"{'qid':<14} {'V5_AP':>7} {'V6_AP':>7} {'ΔAP':>7} {'rel#':>4} | {'τ_anchor':>9} {'cos(Qb,Qn)':>11} {'V5_τ':>7} {'V6_τ':>7} {'pen#':>5} | q_neg")
    print("-" * 110)
    for r in rows_sorted:
        if r["delta_ap"] > 0:
            ta = f"{r['tau_anchor']:.4f}" if r['tau_anchor'] is not None else "  N/A"
            cn = f"{r['cos_qbase_qneg']:.4f}" if r['cos_qbase_qneg'] is not None else "  N/A"
            v5t = f"{r['v5_threshold']:.4f}" if r['cos_qbase_qneg'] is not None else "  N/A"
            v6t = f"{r['v6_threshold']:.4f}" if r['v6_threshold'] is not None else "  N/A"
            print(f"{r['qid']:<14} {r['v5_ap']:>7.4f} {r['v6_ap']:>7.4f} {r['delta_ap']:>+7.4f} {r['rel_count']:>4} | {ta:>9} {cn:>11} {v5t:>7} {v6t:>7} {r['num_penalized'] or 0:>5} | {r['q_neg']}")

    # 7. 阈值统计
    print()
    print("=" * 90)
    print("【阈值变化统计】")
    anchor_rows = [r for r in rows if r["has_anchor"]]
    no_anchor_rows = [r for r in rows if not r["has_anchor"]]
    print(f"  有 safe_anchor 的 query: {len(anchor_rows)} 个")
    if anchor_rows:
        v5_ts = [r["v5_threshold"] for r in anchor_rows]
        v6_ts = [r["v6_threshold"] for r in anchor_rows]
        taus = [r["tau_anchor"] for r in anchor_rows]
        coss = [r["cos_qbase_qneg"] for r in anchor_rows]
        print(f"    V5 阈值 (cos+0.02):  mean={sum(v5_ts)/len(v5_ts):.4f}, min={min(v5_ts):.4f}, max={max(v5_ts):.4f}")
        print(f"    V6 阈值 (max(anchor,cos)-0.05): mean={sum(v6_ts)/len(v6_ts):.4f}, min={min(v6_ts):.4f}, max={max(v6_ts):.4f}")
        print(f"    τ_anchor:            mean={sum(taus)/len(taus):.4f}, min={min(taus):.4f}, max={max(taus):.4f}")
        print(f"    cos(Q_base,Q_neg):   mean={sum(coss)/len(coss):.4f}, min={min(coss):.4f}, max={max(coss):.4f}")
        higher_anchor = sum(1 for r in anchor_rows if r["tau_anchor"] > r["cos_qbase_qneg"])
        print(f"    τ_anchor > cos(Q_base,Q_neg) 的 query: {higher_anchor}/{len(anchor_rows)} ({higher_anchor/len(anchor_rows)*100:.0f}%)")
        # 阈值升高 vs 降低
        raised = sum(1 for r in anchor_rows if r["v6_threshold"] > r["v5_threshold"])
        lowered = sum(1 for r in anchor_rows if r["v6_threshold"] < r["v5_threshold"])
        print(f"    V6阈值 > V5阈值 (更严格): {raised}/{len(anchor_rows)} ({raised/len(anchor_rows)*100:.0f}%)")
        print(f"    V6阈值 < V5阈值 (更宽松): {lowered}/{len(anchor_rows)} ({lowered/len(anchor_rows)*100:.0f}%)")
    print(f"  无 safe_anchor 的 query: {len(no_anchor_rows)} 个 (回退到 cos+ad)")

    # 8. 惩罚文档数统计
    print()
    print("=" * 90)
    print("【惩罚强度统计 (候选集内被惩罚的文档数)】")
    if anchor_rows:
        pens = [r["num_penalized"] or 0 for r in anchor_rows]
        print(f"  有 anchor query 的平均惩罚文档数: {sum(pens)/len(pens):.1f}")
        print(f"  惩罚文档数分布: min={min(pens)}, max={max(pens)}, median={sorted(pens)[len(pens)//2]}")

    # 9. 相关文档被误伤分析（针对下降最严重的 query）
    print()
    print("=" * 90)
    print("【下降最严重 query 的相关文档排名变化】")
    worst = rows_sorted[0]
    qid = worst["qid"]
    qrel_key = qid.replace("-changed", "").replace("-og", "")
    qrel = changed_qrels.get(qrel_key, [])
    relevant_docs = set(qrel) if isinstance(qrel, list) else {d for d, s in qrel.items() if s > 0}
    v5_rank = sorted(v5_ch.get(qid, {}).items(), key=lambda x: -x[1])
    v6_rank = sorted(v6_ch.get(qid, {}).items(), key=lambda x: -x[1])
    v5_pos = {d: i + 1 for i, (d, _) in enumerate(v5_rank)}
    v6_pos = {d: i + 1 for i, (d, _) in enumerate(v6_rank)}
    print(f"  qid={qid}, q_neg='{worst['q_neg']}', rel_docs={len(relevant_docs)}")
    print(f"  V5_AP={worst['v5_ap']:.4f}, V6_AP={worst['v6_ap']:.4f}, ΔAP={worst['delta_ap']:+.4f}")
    print(f"  V5_τ={worst['v5_threshold']:.4f}, V6_τ={worst['v6_threshold']:.4f}, τ_anchor={worst['tau_anchor']:.4f}, cos={worst['cos_qbase_qneg']:.4f}")
    print(f"  {'doc_id':<14} {'V5_rank':>8} {'V6_rank':>8} {'Δrank':>8} | 是否被误伤")
    for d in sorted(relevant_docs, key=lambda x: v5_pos.get(x, 9999)):
        v5r = v5_pos.get(d, 9999)
        v6r = v6_pos.get(d, 9999)
        flag = "⚠️大幅下降" if (v6r - v5r) > 50 else ("下降" if v6r > v5r else ("上升" if v6r < v5r else "不变"))
        print(f"  {d:<14} {v5r:>8} {v6r:>8} {v6r-v5r:>+8} | {flag}")


if __name__ == "__main__":
    main()
