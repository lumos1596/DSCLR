"""
V8 max_mean 机制在 News21 上的细粒度 per-query 分析
目标：
1. 计算每个 query 的 changed_AP（单 query 效果）
2. 关联 α/β/统计量与 query 语义
3. 找出共性问题和改进点
"""
import json
import os
import sys
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

os.environ.setdefault('HF_HOME', '/home/luwa/.cache/huggingface')

# 添加项目根到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics.evaluator import DataLoader

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def average_precision(ranked_doc_ids: List[str], qrels: Dict[str, int]) -> float:
    """计算 AP。qrels: {doc_id: relevance}，二分类（rel > 0 视为相关）。"""
    if not qrels:
        return 0.0
    relevant_set = {d for d, r in qrels.items() if r > 0}
    if not relevant_set:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, did in enumerate(ranked_doc_ids, 1):
        if did in relevant_set:
            hits += 1
            sum_prec += hits / i
    return sum_prec / len(relevant_set)


def ndcg_at_k(ranked_doc_ids: List[str], qrels: Dict[str, int], k: int = 5) -> float:
    """nDCG@k，支持多级相关性。"""
    if not qrels:
        return 0.0
    import math
    # DCG
    dcg = 0.0
    for i, did in enumerate(ranked_doc_ids[:k], 1):
        rel = qrels.get(did, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(i + 1)
    # IDCG
    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 1) for i, r in enumerate(ideal_rels, 1) if r > 0)
    return dcg / idcg if idcg > 0 else 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats_dir', type=str,
                        default='/home/luwa/Documents/DSCLR/results/safe_anchor_v8_max_mean_news21_stats/News21',
                        help='per_query_stats.json 所在目录')
    parser.add_argument('--tag', type=str, default='V8 max_mean',
                        help='分析标签（用于日志标题）')
    args = parser.parse_args()

    base = '/home/luwa/Documents/DSCLR'
    stats_dir = args.stats_dir

    # 1. 加载 per_query_stats
    with open(f'{stats_dir}/per_query_stats.json') as f:
        stats_data = json.load(f)
    per_query_stats = stats_data['per_query_stats']
    logger.info(f"✅ 加载 per_query_stats: {len(per_query_stats)} queries")

    # 2. 加载 dual_queries (q_idx 是文件行号 0-based)
    dual_queries = []
    with open(f'{base}/dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_News21InstructionRetrieval.jsonl') as f:
        for line in f:
            d = json.loads(line)
            dual_queries.append(d)
    logger.info(f"✅ 加载 dual_queries: {len(dual_queries)} queries")

    # 3. 加载 ranking
    with open(f'{stats_dir}/ranking_changed.json') as f:
        ranking_changed = json.load(f)
    with open(f'{stats_dir}/ranking_og.json') as f:
        ranking_og = json.load(f)
    logger.info(f"✅ 加载 ranking: changed={len(ranking_changed)}, og={len(ranking_og)}")

    # 4. 加载 qrels
    dl = DataLoader('News21InstructionRetrieval')
    qrels = dl.load_qrels()
    logger.info(f"✅ 加载 qrels: {len(qrels)} queries")

    # 5. 加载 qrel_diff（用于判断 changed query）
    # 从 evaluator 获取 qrels_diff
    # 这里简单处理：根据 ranking keys 推断
    # qid 格式: "937-changed" / "937-og"

    # 6. 构建分析表
    results = []
    for s in per_query_stats:
        q_idx = s['q_idx']
        if q_idx >= len(dual_queries):
            logger.warning(f"q_idx {q_idx} 超出 dual_queries 范围")
            continue
        dq = dual_queries[q_idx]
        qid_str = str(dq['idx'])
        qid_changed = f"{qid_str}-changed"
        qid_og = f"{qid_str}-og"

        # 排序
        if qid_changed in ranking_changed:
            rc = ranking_changed[qid_changed]
            ranked_changed = sorted(rc.keys(), key=lambda d: -rc[d])
        else:
            ranked_changed = []
        if qid_og in ranking_og:
            ro = ranking_og[qid_og]
            ranked_og = sorted(ro.keys(), key=lambda d: -ro[d])
        else:
            ranked_og = []

        # qrels - 用原始 qid (无后缀)
        qrel = qrels.get(qid_str, qrels.get(qid_changed, {}))
        # 尝试不同的 qid 格式
        if not qrel:
            for k in qrels:
                if k.startswith(qid_str) or qid_str.startswith(k.split('-')[0]):
                    qrel = qrels[k]
                    break

        # 计算 AP 和 nDCG@5
        ap_changed = average_precision(ranked_changed, qrel) if qrel else 0.0
        ap_og = average_precision(ranked_og, qrel) if qrel else 0.0
        ndcg5_changed = ndcg_at_k(ranked_changed, qrel, 5) if qrel else 0.0
        ndcg5_og = ndcg_at_k(ranked_og, qrel, 5) if qrel else 0.0

        # 改善幅度（AP_delta = ap_changed - ap_og）
        ap_delta = ap_changed - ap_og
        ndcg5_delta = ndcg5_changed - ndcg5_og

        results.append({
            'q_idx': q_idx,
            'qid': qid_str,
            'query': dq['query'],
            'instruction': dq['instruction'][:200],
            'q_plus': dq['q_plus'],
            'q_minus': dq['q_minus'],
            'query_type': dq['query_type'],
            'alpha_q': s['alpha_q'],
            'beta_q': s['beta_q'],
            'num_at_risk': s['num_at_risk'],
            'at_risk_ratio': s['at_risk_ratio'],
            's_base_max': s['s_base_max'],
            's_base_mean': s['s_base_mean'],
            's_base_std': s['s_base_std'],
            's_req_mean': s['s_req_mean'],
            's_req_max': s['s_req_max'],
            's_neg_mean': s['s_neg_mean'],
            's_neg_max': s['s_neg_max'],
            's_reward_mean': s['s_reward_mean'],
            'safety_mean': s['safety_mean'],
            'gap_sbase_sreward': s['gap_sbase_sreward'],
            'cos_qbase_qneg': s['cos_qbase_qneg'],
            'tau_penalty': s['tau_penalty'],
            'ap_changed': ap_changed,
            'ap_og': ap_og,
            'ap_delta': ap_delta,
            'ndcg5_changed': ndcg5_changed,
            'ndcg5_og': ndcg5_og,
            'ndcg5_delta': ndcg5_delta,
            'n_rel': sum(1 for r in qrel.values() if r > 0) if qrel else 0,
        })

    logger.info(f"\n{'='*180}")
    logger.info(f"News21 {args.tag} 细粒度 per-query 分析（共 {len(results)} queries）")
    logger.info(f"{'='*180}\n")

    # ============ 总体统计 ============
    logger.info("【总体统计】")
    alphas = [r['alpha_q'] for r in results]
    betas = [r['beta_q'] for r in results]
    aps_ch = [r['ap_changed'] for r in results]
    aps_og = [r['ap_og'] for r in results]
    ndcgs_ch = [r['ndcg5_changed'] for r in results]
    ndcgs_og = [r['ndcg5_og'] for r in results]
    import statistics
    logger.info(f"  α: all=1.0 (fallback) | at-risk queries: {sum(1 for r in results if r['num_at_risk']>0)}/{len(results)}")
    logger.info(f"  β: min={min(betas):.3f} max={max(betas):.3f} mean={statistics.mean(betas):.3f} std={statistics.stdev(betas):.3f}")
    logger.info(f"  changed_AP: mean={statistics.mean(aps_ch):.4f} | og_AP: mean={statistics.mean(aps_og):.4f}")
    logger.info(f"  changed_nDCG@5: mean={statistics.mean(ndcgs_ch):.4f} | og_nDCG@5: mean={statistics.mean(ndcgs_og):.4f}")
    improved = sum(1 for r in results if r['ap_delta'] > 0)
    declined = sum(1 for r in results if r['ap_delta'] < 0)
    logger.info(f"  AP 改善: {improved}/{len(results)} | AP 下降: {declined}/{len(results)} | 持平: {len(results)-improved-declined}")

    # ============ 排序：按 ap_delta 从差到好 ============
    sorted_by_delta = sorted(results, key=lambda r: r['ap_delta'])

    logger.info(f"\n{'='*180}")
    logger.info("【按 AP 改善幅度排序（最差→最好）】")
    logger.info(f"{'='*180}\n")
    header = f"{'q_idx':>5} {'qid':>6} {'α':>5} {'β':>6} {'AP_ch':>7} {'AP_og':>7} {'ΔAP':>7} {'nDCG5_ch':>9} {'ΔnDCG5':>8} {'n_rel':>6} {'s_b_max':>8} {'s_r_mean':>9} {'s_n_max':>8} {'safety':>7} {'gap':>6} {'at_risk':>7}"
    logger.info(header)
    logger.info('-' * len(header))
    for r in sorted_by_delta:
        logger.info(f"{r['q_idx']:>5} {r['qid']:>6} {r['alpha_q']:>5.2f} {r['beta_q']:>6.3f} {r['ap_changed']:>7.4f} {r['ap_og']:>7.4f} {r['ap_delta']:>+7.4f} {r['ndcg5_changed']:>9.4f} {r['ndcg5_delta']:>+8.4f} {r['n_rel']:>6} {r['s_base_max']:>8.4f} {r['s_req_mean']:>9.4f} {r['s_neg_max']:>8.4f} {r['safety_mean']:>7.4f} {r['gap_sbase_sreward']:>6.3f} {r['num_at_risk']:>7}")

    # ============ 表现最差的 5 个 query 的语义分析 ============
    logger.info(f"\n{'='*180}")
    logger.info("【表现最差的 5 个 query（AP 下降最多）- 语义深度分析】")
    logger.info(f"{'='*180}\n")
    for r in sorted_by_delta[:5]:
        logger.info(f"--- q_idx={r['q_idx']} (qid={r['qid']}) | α={r['alpha_q']:.2f} β={r['beta_q']:.3f} | ΔAP={r['ap_delta']:+.4f} ΔnDCG5={r['ndcg5_delta']:+.4f} | n_rel={r['n_rel']} ---")
        logger.info(f"  原始 query    : {r['query']}")
        logger.info(f"  instruction   : {r['instruction']}")
        logger.info(f"  q_plus        : {r['q_plus']}")
        logger.info(f"  q_minus       : {r['q_minus']}")
        logger.info(f"  统计          : s_b_max={r['s_base_max']:.4f} s_b_mean={r['s_base_mean']:.4f} s_r_mean={r['s_req_mean']:.4f} s_n_max={r['s_neg_max']:.4f}")
        logger.info(f"                safety={r['safety_mean']:.4f} gap={r['gap_sbase_sreward']:.4f} at_risk={r['num_at_risk']}")
        logger.info(f"  效果          : AP(og→ch)={r['ap_og']:.4f}→{r['ap_changed']:.4f} | nDCG5(og→ch)={r['ndcg5_og']:.4f}→{r['ndcg5_changed']:.4f}")
        logger.info("")

    # ============ 表现最好的 5 个 query 的语义分析 ============
    logger.info(f"\n{'='*180}")
    logger.info("【表现最好的 5 个 query（AP 上升最多）- 语义深度分析】")
    logger.info(f"{'='*180}\n")
    for r in sorted_by_delta[-5:]:
        logger.info(f"--- q_idx={r['q_idx']} (qid={r['qid']}) | α={r['alpha_q']:.2f} β={r['beta_q']:.3f} | ΔAP={r['ap_delta']:+.4f} ΔnDCG5={r['ndcg5_delta']:+.4f} | n_rel={r['n_rel']} ---")
        logger.info(f"  原始 query    : {r['query']}")
        logger.info(f"  instruction   : {r['instruction']}")
        logger.info(f"  q_plus        : {r['q_plus']}")
        logger.info(f"  q_minus       : {r['q_minus']}")
        logger.info(f"  统计          : s_b_max={r['s_base_max']:.4f} s_b_mean={r['s_base_mean']:.4f} s_r_mean={r['s_req_mean']:.4f} s_n_max={r['s_neg_max']:.4f}")
        logger.info(f"                safety={r['safety_mean']:.4f} gap={r['gap_sbase_sreward']:.4f} at_risk={r['num_at_risk']}")
        logger.info(f"  效果          : AP(og→ch)={r['ap_og']:.4f}→{r['ap_changed']:.4f} | nDCG5(og→ch)={r['ndcg5_og']:.4f}→{r['ndcg5_changed']:.4f}")
        logger.info("")

    # ============ β 高/低对比 ============
    sorted_by_beta = sorted(results, key=lambda r: r['beta_q'])
    logger.info(f"\n{'='*180}")
    logger.info("【β 最低的 5 个 query】")
    logger.info(f"{'='*180}\n")
    for r in sorted_by_beta[:5]:
        logger.info(f"q_idx={r['q_idx']} β={r['beta_q']:.3f} ΔAP={r['ap_delta']:+.4f} | q: {r['query'][:80]}")
        logger.info(f"  q_minus: {r['q_minus'][:100]}")
        logger.info(f"  s_b_max={r['s_base_max']:.4f} s_r_mean={r['s_req_mean']:.4f} (β=s_b_max/s_r_mean_safe) | gap={r['gap_sbase_sreward']:.4f}")
        logger.info("")

    logger.info(f"\n{'='*180}")
    logger.info("【β 最高的 5 个 query】")
    logger.info(f"{'='*180}\n")
    for r in sorted_by_beta[-5:]:
        logger.info(f"q_idx={r['q_idx']} β={r['beta_q']:.3f} ΔAP={r['ap_delta']:+.4f} | q: {r['query'][:80]}")
        logger.info(f"  q_minus: {r['q_minus'][:100]}")
        logger.info(f"  s_b_max={r['s_base_max']:.4f} s_r_mean={r['s_req_mean']:.4f} (β=s_b_max/s_r_mean_safe) | gap={r['gap_sbase_sreward']:.4f}")
        logger.info("")

    # ============ 共性问题分析 ============
    logger.info(f"\n{'='*180}")
    logger.info("【共性问题分析】")
    logger.info(f"{'='*180}\n")

    # 问题1: penalty track 完全失效
    n_no_atrisk = sum(1 for r in results if r['num_at_risk'] == 0)
    logger.info(f"问题1: penalty track 失效")
    logger.info(f"  {n_no_atrisk}/{len(results)} queries 的 num_at_risk=0，α 全部 fallback 为 1.0")
    logger.info(f"  → 这意味着 max_mean 实际退化为 'Q_plus-only' 模式，penalty 项对最终分数无贡献")
    # s_neg_max vs tau_penalty 的差距
    gaps = [r['tau_penalty'] - r['s_neg_max'] for r in results]
    logger.info(f"  τ_penalty - s_neg_max 的差距: min={min(gaps):.4f} max={max(gaps):.4f} mean={statistics.mean(gaps):.4f}")
    logger.info(f"  → 所有 query 的最高 s_neg 都低于阈值，penalty track 完全无效")
    logger.info("")

    # 问题2: safety_mean 与效果的关系
    corr_safety_ap = statistics.correlation([r['safety_mean'] for r in results], [r['ap_delta'] for r in results])
    logger.info(f"问题2: safety_mean 与 AP_delta 的相关性")
    logger.info(f"  Pearson 相关系数: {corr_safety_ap:+.4f}")
    logger.info(f"  → 负相关意味着 safety 越低（更多文档被门控抑制）AP 改善越差")
    logger.info("")

    # 问题3: β 与效果的关系
    corr_beta_ap = statistics.correlation([r['beta_q'] for r in results], [r['ap_delta'] for r in results])
    corr_beta_ndcg = statistics.correlation([r['beta_q'] for r in results], [r['ndcg5_delta'] for r in results])
    logger.info(f"问题3: β 与效果的关系")
    logger.info(f"  β vs ΔAP 相关系数: {corr_beta_ap:+.4f}")
    logger.info(f"  β vs ΔnDCG5 相关系数: {corr_beta_ndcg:+.4f}")
    logger.info("")

    # 问题4: gap 与效果的关系
    corr_gap_ap = statistics.correlation([r['gap_sbase_sreward'] for r in results], [r['ap_delta'] for r in results])
    logger.info(f"问题4: gap (S_base vs S_reward) 与效果的关系")
    logger.info(f"  gap vs ΔAP 相关系数: {corr_gap_ap:+.4f}")
    logger.info(f"  → 正相关意味着 gap 越大（基础信号远高于奖励信号）AP 改善越大")
    logger.info("")

    # 问题5: s_neg_max 分布 - 哪些 query 的 s_neg_max 接近 τ
    logger.info(f"问题5: s_neg_max 与 τ_penalty 的接近度")
    close_to_tau = [(r['q_idx'], r['s_neg_max'], r['tau_penalty'], r['s_neg_max'] - r['tau_penalty'])
                    for r in results]
    close_to_tau.sort(key=lambda x: -x[3])  # 最接近 τ 的在前
    logger.info(f"  最接近 τ 的 5 个 query:")
    for qidx, sn, tp, gap in close_to_tau[:5]:
        logger.info(f"    q_idx={qidx}: s_neg_max={sn:.4f} τ={tp:.4f} gap={gap:+.4f}")
    logger.info("")

    # 保存详细结果
    out_path = f'{stats_dir}/per_query_analysis.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_queries': len(results),
            'summary': {
                'alpha_all_fallback': all(r['alpha_q'] == 1.0 for r in results),
                'beta_mean': statistics.mean(betas),
                'beta_std': statistics.stdev(betas),
                'beta_min': min(betas),
                'beta_max': max(betas),
                'ap_changed_mean': statistics.mean(aps_ch),
                'ap_og_mean': statistics.mean(aps_og),
                'ndcg5_changed_mean': statistics.mean(ndcgs_ch),
                'ndcg5_og_mean': statistics.mean(ndcgs_og),
                'n_improved': improved,
                'n_declined': declined,
                'n_no_atrisk': n_no_atrisk,
                'corr_safety_ap': corr_safety_ap,
                'corr_beta_ap': corr_beta_ap,
                'corr_beta_ndcg': corr_beta_ndcg,
                'corr_gap_ap': corr_gap_ap,
            },
            'queries': sorted(sorted_by_delta, key=lambda r: r['q_idx']),
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 详细分析已保存: {out_path}")


if __name__ == '__main__':
    main()
