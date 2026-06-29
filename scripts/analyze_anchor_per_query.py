"""Per-query 锚点阈值有效性分析。

对每个 changed 查询分两类文档：
  A. 翻转文档（og 相关 → changed 不相关）：penalty 应该打击 → 期望 S_neg > τ
  B. 保持相关文档（changed 中仍相关）：penalty 应该保护 → 期望 S_neg < τ

对比 τ_anchor vs τ_V5，判断锚点阈值偏高/偏低，并找出"漏打"和"误伤"案例。
"""
import json
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/luwa/Documents/DSCLR")

from eval.experiment_safe_anchor_threshold import (
    SafeAnchorDeIREvaluator,
    load_safe_anchors,
    compute_safe_anchor_threshold,
)
from eval.metrics.evaluator import DataLoader

TASK = "Core17InstructionRetrieval"
DUAL_PATH = "dataset/FollowIR_test/dual_queries_4B/TSC_BALANCED_t01/dual_queries_TSC_BALANCED_t01_Core17InstructionRetrieval.jsonl"
ANCHOR_PATH = "dataset/FollowIR_test/safe_anchors/safe_anchors_core17.json"
DELTA_V5 = 0.02


def main():
    engine = SafeAnchorDeIREvaluator(
        model_name="samaya-ai/RepLLaMA-reproduced",
        task_name=TASK,
        output_dir="results/safe_anchor_analysis/tmp",
        dual_queries_path=DUAL_PATH,
        safe_anchors_path=ANCHOR_PATH,
        anchor_stat="max",
        anchor_delta=0.0,
        anchor_mix_mode="replace",
        device="cuda",
        batch_size=32,
        use_cache=True,
    )

    dl = DataLoader(TASK)
    qrels = dl.load_qrels()  # {qid-og/qid-changed: {doc_id: rel}}
    diff = dl.load_qrel_diff()  # {base_qid: [翻转文档]}
    candidates: Dict[str, List[str]] = dl.load_candidates()

    # 加载文档嵌入
    corpus, _, _, _ = engine.data_loader.load()
    all_doc_ids = engine._get_all_candidate_doc_ids(candidates)
    from eval.engine_dscrl import load_cached_embeddings
    cached_data = load_cached_embeddings(engine.cache_dir, engine.task_name, engine.model_name)
    if cached_data is not None:
        ce, cdids = cached_data
        if set(cdids) == set(all_doc_ids):
            engine.retriever.set_embeddings(ce, cdids)
        else:
            engine.retriever.index_documents(all_doc_ids, [corpus[d]["text"] for d in all_doc_ids], engine.batch_size)
    else:
        engine.retriever.index_documents(all_doc_ids, [corpus[d]["text"] for d in all_doc_ids], engine.batch_size)

    # 编码 changed 查询
    with open(DUAL_PATH) as f:
        dual_records = [json.loads(l) for l in f]
    dual_map = {r["qid"]: r for r in dual_records}
    q_og_raw, q_ch_raw = engine.data_loader.load_raw_queries()
    qids_ch = [q for q in dual_map.keys() if q.endswith("-changed")]
    base_ch, req_ch, neg_ch = [], [], []
    for qid in qids_ch:
        r = dual_map[qid]
        base_ch.append(q_ch_raw.get(qid.replace("-changed", ""), ""))
        req_ch.append(r.get("q_plus", ""))
        neg_ch.append(r.get("q_minus", ""))
    print(f"Changed queries: {len(qids_ch)}")

    q_base_emb = engine._encode_queries(base_ch).to("cuda")
    q_neg_emb = engine._encode_queries(neg_ch).to("cuda")
    doc_emb = engine.retriever.doc_embeddings
    doc_ids = engine.retriever.doc_ids
    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}

    S_neg_all = (q_neg_emb @ doc_emb.T)  # [Q, N]
    cos_qbase_qneg = F.cosine_similarity(q_base_emb, q_neg_emb, dim=1)
    tau_v5 = cos_qbase_qneg + DELTA_V5

    anchors_map = load_safe_anchors(ANCHOR_PATH)
    tau_anchor, per_q_scores = compute_safe_anchor_threshold(
        q_neg_emb, qids_ch, anchors_map, encoder_fn=engine._encode_queries, stat="max"
    )

    # ============ Per-query 分析 ============
    print("\n" + "=" * 140)
    print(f"{'qid':<14}{'τ_V5':>8}{'τ_anc':>8} | {'翻转数':>6}{'打中':>6}{'率':>7} "
          f"{'翻转S_neg中位':>14}{'翻转S_neg最小':>14} | {'保相数':>6}{'保护':>6}{'率':>7} "
          f"{'保相S_neg中位':>14}{'保相S_neg最大':>14} | {'判定':>12}")
    print("-" * 140)

    summary = {
        "flip_total": 0, "flip_hit_anchor": 0, "flip_hit_v5": 0,
        "keep_total": 0, "keep_safe_anchor": 0, "keep_safe_v5": 0,
        "per_query": []
    }

    for qi, qid in enumerate(qids_ch):
        base_qid = qid.replace("-changed", "")
        if tau_anchor[qi] == float("-inf"):
            continue
        if base_qid not in candidates or base_qid not in diff:
            continue
        if f"{base_qid}-changed" not in qrels:
            continue

        cand = candidates[base_qid]
        cand_idx = [doc_id_to_idx[d] for d in cand if d in doc_id_to_idx]
        s_neg_cand = S_neg_all[qi, cand_idx].cpu().numpy()

        # A. 翻转文档（应被惩罚）
        flip_docs = set(diff[base_qid])
        flip_mask = np.array([d in flip_docs for d in cand if d in doc_id_to_idx])
        # B. 保持相关文档（应被保护）
        ch_rel_docs = set(d for d, r in qrels[f"{base_qid}-changed"].items() if r > 0)
        keep_mask = np.array([d in ch_rel_docs for d in cand if d in doc_id_to_idx])

        if flip_mask.sum() == 0 or keep_mask.sum() == 0:
            continue

        s_neg_flip = s_neg_cand[flip_mask]
        s_neg_keep = s_neg_cand[keep_mask]

        t_a = float(tau_anchor[qi])
        t_v = float(tau_v5[qi])

        # 翻转文档命中率（S_neg > τ 应被惩罚）
        flip_hit_a = int((s_neg_flip > t_a).sum())
        flip_hit_v = int((s_neg_flip > t_v).sum())
        # 保持相关文档保护率（S_neg < τ 应被保护）
        keep_safe_a = int((s_neg_keep < t_a).sum())
        keep_safe_v = int((s_neg_keep < t_v).sum())

        summary["flip_total"] += len(s_neg_flip)
        summary["flip_hit_anchor"] += flip_hit_a
        summary["flip_hit_v5"] += flip_hit_v
        summary["keep_total"] += len(s_neg_keep)
        summary["keep_safe_anchor"] += keep_safe_a
        summary["keep_safe_v5"] += keep_safe_v

        # 判定锚点阈值偏高/偏低
        flip_med = float(np.median(s_neg_flip))
        keep_max = float(np.max(s_neg_keep))
        if t_a < flip_med and t_a < keep_max:
            verdict = "偏低(双误)"
        elif t_a < flip_med:
            verdict = "偏低(漏打)"
        elif t_a > keep_max and t_a > flip_med:
            verdict = "偏高(误伤)"
        elif t_a > keep_max:
            verdict = "偏高(过保)"
        else:
            verdict = "合理"

        row = {
            "qid": qid, "tau_v5": t_v, "tau_anchor": t_a,
            "n_flip": len(s_neg_flip), "flip_hit_a": flip_hit_a, "flip_hit_v": flip_hit_v,
            "flip_med": flip_med, "flip_min": float(np.min(s_neg_flip)),
            "n_keep": len(s_neg_keep), "keep_safe_a": keep_safe_a, "keep_safe_v": keep_safe_v,
            "keep_med": float(np.median(s_neg_keep)), "keep_max": keep_max,
            "verdict": verdict,
            "anchor_scores": per_q_scores[qi],
        }
        summary["per_query"].append(row)
        print(f"{qid:<14}{t_v:>8.4f}{t_a:>8.4f} | {len(s_neg_flip):>6}{flip_hit_a:>6}{flip_hit_a/len(s_neg_flip)*100:>6.1f}% "
              f"{flip_med:>14.4f}{float(np.min(s_neg_flip)):>14.4f} | {len(s_neg_keep):>6}{keep_safe_a:>6}{keep_safe_a/len(s_neg_keep)*100:>6.1f}% "
              f"{float(np.median(s_neg_keep)):>14.4f}{keep_max:>14.4f} | {verdict:>12}")

    print("\n" + "=" * 140)
    print("汇总:")
    print(f"  翻转文档（应惩罚）:")
    print(f"    τ_anchor 命中率: {summary['flip_hit_anchor']}/{summary['flip_total']} = {summary['flip_hit_anchor']/summary['flip_total']*100:.1f}%")
    print(f"    τ_V5     命中率: {summary['flip_hit_v5']}/{summary['flip_total']} = {summary['flip_hit_v5']/summary['flip_total']*100:.1f}%")
    print(f"  保持相关文档（应保护）:")
    print(f"    τ_anchor 保护率: {summary['keep_safe_anchor']}/{summary['keep_total']} = {summary['keep_safe_anchor']/summary['keep_total']*100:.1f}%")
    print(f"    τ_V5     保护率: {summary['keep_safe_v5']}/{summary['keep_total']} = {summary['keep_safe_v5']/summary['keep_total']*100:.1f}%")

    # 漏打与误伤的具体案例
    print("\n漏打案例（翻转文档 S_neg < τ_anchor，应打未打）:")
    print(f"{'qid':<14}{'τ_anc':>8}{'翻转文档S_neg':>30}")
    for r in summary["per_query"]:
        if r["flip_hit_a"] < r["n_flip"]:
            print(f"{r['qid']:<14}{r['tau_anchor']:>8.4f}  {r['n_flip']-r['flip_hit_a']}/{r['n_flip']} 个漏打, flip_min={r['flip_min']:.4f}")

    print("\n误伤案例（保持相关文档 S_neg > τ_anchor，应保护未保护）:")
    print(f"{'qid':<14}{'τ_anc':>8}{'保相文档S_neg':>30}")
    for r in summary["per_query"]:
        if r["keep_safe_a"] < r["n_keep"]:
            print(f"{r['qid']:<14}{r['tau_anchor']:>8.4f}  {r['n_keep']-r['keep_safe_a']}/{r['n_keep']} 个误伤, keep_max={r['keep_max']:.4f}")

    # 分离度分析：τ 是否落在 flip 和 keep 之间
    print("\n分离度分析（τ 是否在 flip 分布和 keep 分布之间形成分界）:")
    sep_anchor = sum(1 for r in summary["per_query"] if r["tau_anchor"] >= r["keep_med"] and r["tau_anchor"] <= r["flip_med"])
    sep_v5 = sum(1 for r in summary["per_query"] if r["tau_v5"] >= r["keep_med"] and r["tau_v5"] <= r["flip_med"])
    print(f"  τ_anchor 落在 keep_med 与 flip_med 之间: {sep_anchor}/{len(summary['per_query'])}")
    print(f"  τ_V5     落在 keep_med 与 flip_med 之间: {sep_v5}/{len(summary['per_query'])}")

    out = "results/safe_anchor_analysis/per_query_effectiveness_core17.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
