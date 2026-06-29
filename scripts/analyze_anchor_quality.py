"""分析 safe-anchor 文档质量：锚点 S_neg 是否对齐真实"边界相关文档"。

对比每个 changed 查询的三类 S_neg：
  1. 锚点文档 S_neg_anchor = cos(q_neg, anchor_doc)
  2. 真实相关文档 S_neg（changed 后仍相关的文档）
  3. 真实"边界相关文档" S_neg（最接近被惩罚阈值的相关文档）

并对比 τ_anchor vs τ_V5 = cos(Q_base,Q_neg)+δ vs 真实边界 S_neg。
"""
import json
import sys
from typing import Dict, List

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
    # 复用 experiment engine 的初始化（加载模型 + 缓存文档嵌入 + dual_queries）
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

    # 加载 qrels / candidates
    dl = DataLoader(TASK)
    changed_qrels: Dict[str, List[str]] = dl.load_qrel_diff()  # base_qid -> [changed doc ids]
    candidates: Dict[str, List[str]] = dl.load_candidates()

    # 触发文档嵌入加载（复用 engine.run 的逻辑）
    corpus, _, _, _ = engine.data_loader.load()
    all_doc_ids = engine._get_all_candidate_doc_ids(candidates)
    from eval.engine_dscrl import load_cached_embeddings
    cached_data = load_cached_embeddings(engine.cache_dir, engine.task_name, engine.model_name)
    if cached_data is not None:
        cached_embeddings, cached_doc_ids = cached_data
        if set(cached_doc_ids) == set(all_doc_ids):
            engine.retriever.set_embeddings(cached_embeddings, cached_doc_ids)
        else:
            doc_texts = [corpus[did]["text"] for did in all_doc_ids]
            engine.retriever.index_documents(all_doc_ids, doc_texts, engine.batch_size)
    else:
        doc_texts = [corpus[did]["text"] for did in all_doc_ids]
        engine.retriever.index_documents(all_doc_ids, doc_texts, engine.batch_size)

    # 复用 engine.run() 的前半部分：编码 queries、算 S_*、算 tau_anchor
    q_og_raw, q_ch_raw = engine.data_loader.load_raw_queries()
    # 构建查询列表（与 engine.run 一致）
    import json as _json
    with open(DUAL_PATH) as f:
        dual_records = [_json.loads(l) for l in f]
    dual_map = {r["qid"]: r for r in dual_records}

    def build(qids, raw_map):
        ids, base_l, req_l, neg_l = [], [], [], []
        for qid in qids:
            r = dual_map.get(qid, {})
            base = raw_map.get(qid.replace("-changed", "").replace("-og", ""), "")
            ids.append(qid)
            base_l.append(base)
            req_l.append(r.get("q_plus", ""))
            neg_l.append(r.get("q_minus", ""))
        return ids, base_l, req_l, neg_l

    qids_ch, base_ch, req_ch, neg_ch = build(
        [q for q in dual_map.keys() if q.endswith("-changed")], q_ch_raw
    )
    print(f"Changed queries: {len(qids_ch)}")

    print("编码 queries...")
    q_base_emb = engine._encode_queries(base_ch).to("cuda")
    q_req_emb = engine._encode_queries(req_ch).to("cuda")
    q_neg_emb = engine._encode_queries(neg_ch).to("cuda")

    doc_emb = engine.retriever.doc_embeddings  # [N, D]
    doc_ids = engine.retriever.doc_ids
    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}

    # S_base / S_req / S_neg 对全部文档
    S_base = q_base_emb @ doc_emb.T  # [Q, N]
    S_req = q_req_emb @ doc_emb.T
    S_neg = (q_neg_emb @ doc_emb.T)  # [Q, N]

    cos_qbase_qneg = F.cosine_similarity(q_base_emb, q_neg_emb, dim=1)  # [Q]
    tau_v5 = cos_qbase_qneg + DELTA_V5  # [Q]

    # 锚点 tau
    anchors_map = load_safe_anchors(ANCHOR_PATH)
    tau_anchor, per_q_anchor_scores = compute_safe_anchor_threshold(
        q_neg_emb, qids_ch, anchors_map, encoder_fn=engine._encode_queries, stat="max"
    )

    print("\n" + "=" * 110)
    print(f"{'qid':<14}{'cos(Qb,Qn)':>11}{'τ_V5':>8}{'τ_anchor':>10}{'gap(τa-τV5)':>13}"
          f"{'S_neg_rel_max':>15}{'S_neg_rel_med':>15}{'S_neg_nonrel_p90':>18}")
    print("-" * 110)

    rows = []
    for qi, qid in enumerate(qids_ch):
        base_qid = qid.replace("-changed", "")
        if tau_anchor[qi] == float("-inf"):
            continue  # 无锚点
        if base_qid not in candidates or base_qid not in changed_qrels:
            continue

        cand = candidates[base_qid]
        cand_idx = [doc_id_to_idx[d] for d in cand if d in doc_id_to_idx]
        rel_docs = set(changed_qrels[base_qid])

        s_neg_cand = S_neg[qi, cand_idx].cpu().numpy()
        rel_mask = [d in rel_docs for d in cand if d in doc_id_to_idx]
        import numpy as np
        s_neg_rel = s_neg_cand[rel_mask]
        s_neg_nonrel = s_neg_cand[[not m for m in rel_mask]]

        if len(s_neg_rel) == 0:
            continue

        row = {
            "qid": qid,
            "cos": float(cos_qbase_qneg[qi]),
            "tau_v5": float(tau_v5[qi]),
            "tau_anchor": float(tau_anchor[qi]),
            "gap": float(tau_anchor[qi] - tau_v5[qi]),
            "s_neg_rel_max": float(np.max(s_neg_rel)),
            "s_neg_rel_med": float(np.median(s_neg_rel)),
            "s_neg_nonrel_p90": float(np.percentile(s_neg_nonrel, 90)) if len(s_neg_nonrel) > 0 else float("nan"),
            "n_rel": len(s_neg_rel),
            "anchor_scores": per_q_anchor_scores[qi],
        }
        rows.append(row)
        print(f"{qid:<14}{row['cos']:>11.4f}{row['tau_v5']:>8.4f}{row['tau_anchor']:>10.4f}"
              f"{row['gap']:>+13.4f}{row['s_neg_rel_max']:>15.4f}{row['s_neg_rel_med']:>15.4f}"
              f"{row['s_neg_nonrel_p90']:>18.4f}")

    # 汇总统计
    print("\n" + "=" * 110)
    print("汇总统计:")
    import numpy as np
    gaps = np.array([r["gap"] for r in rows])
    relmax_minus_taua = np.array([r["s_neg_rel_max"] - r["tau_anchor"] for r in rows])
    relmax_minus_tauv5 = np.array([r["s_neg_rel_max"] - r["tau_v5"] for r in rows])
    print(f"  τ_anchor - τ_V5  gap: mean={gaps.mean():+.4f}  median={np.median(gaps):+.4f}  "
          f"min={gaps.min():+.4f}  max={gaps.max():+.4f}")
    print(f"  S_neg_rel_max - τ_anchor: mean={relmax_minus_taua.mean():+.4f}  "
          f"(>0 表示相关文档最高 S_neg 超过锚点阈值 → 会被误惩罚)")
    print(f"  S_neg_rel_max - τ_V5:     mean={relmax_minus_tauv5.mean():+.4f}  "
          f"(>0 表示相关文档最高 S_neg 超过 V5 阈值 → 会被误惩罚)")
    # 锚点 S_neg vs 真实边界相关文档 S_neg 的差异
    print("\n逐 query 锚点得分 vs 真实相关文档 S_neg:")
    print(f"{'qid':<14}{'anchor_scores':>30}{'S_neg_rel_max':>16}{'anchor<rel_max?':>18}")
    for r in rows:
        flag = "YES(锚点偏低)" if max(r["anchor_scores"]) < r["s_neg_rel_max"] else "no"
        scores_str = "[" + ",".join(f"{s:.3f}" for s in r["anchor_scores"]) + "]"
        print(f"{r['qid']:<14}{scores_str:>30}{r['s_neg_rel_max']:>16.4f}{flag:>18}")

    # 保存
    out = "results/safe_anchor_analysis/anchor_quality_core17.json"
    with open(out, "w") as f:
        json.dump({"rows": rows, "summary": {
            "gap_mean": float(gaps.mean()),
            "gap_median": float(np.median(gaps)),
            "relmax_minus_taua_mean": float(relmax_minus_taua.mean()),
            "relmax_minus_tauv5_mean": float(relmax_minus_tauv5.mean()),
        }}, f, indent=2)
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
