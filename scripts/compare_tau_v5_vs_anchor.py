"""对比 Core17 每个 query 的 V5 甜点区 τ = cos(Q_base, Q_neg) 与锚点 S_neg_anchor。

输出每个 query:
  - cos(Q_base, Q_neg)  [V5 目标]
  - 每个锚点文档的 S_neg = cos(Q_neg, anchor_doc)
  - τ_anchor (max/min/mean)
  - 差值 = τ_anchor - cos(Q_base, Q_neg)
  - 指导：锚点需调高(增加neg语义) 还是 调低(减少neg语义)
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/luwa/Documents/DSCLR")
from eval.experiment_safe_anchor_threshold import (
    SafeAnchorDeIREvaluator,
    load_safe_anchors,
)
from eval.metrics.evaluator import DataLoader

DELTA = 0.02
TASK = "Core17InstructionRetrieval"
DUAL = "dataset/FollowIR_test/dual_queries_4B/TSC_BALANCED_t01/dual_queries_TSC_BALANCED_t01_Core17InstructionRetrieval.jsonl"
ANCHOR = "dataset/FollowIR_test/safe_anchors/safe_anchors_core17.json"


def main():
    engine = SafeAnchorDeIREvaluator(
        model_name="samaya-ai/RepLLaMA-reproduced",
        task_name=TASK,
        output_dir="results/safe_anchor_analysis/tmp",
        dual_queries_path=DUAL,
        safe_anchors_path=ANCHOR,
        anchor_stat="max",
        device="cuda",
        batch_size=32,
        use_cache=True,
    )

    dl = DataLoader(TASK)
    candidates = dl.load_candidates()
    corpus, _, _, _ = engine.data_loader.load()
    all_doc_ids = engine._get_all_candidate_doc_ids(candidates)
    from eval.engine_dscrl import load_cached_embeddings
    cached_data = load_cached_embeddings(engine.cache_dir, TASK, engine.model_name)
    if cached_data is not None:
        ce, cdids = cached_data
        if set(cdids) == set(all_doc_ids):
            engine.retriever.set_embeddings(ce, cdids)
        else:
            engine.retriever.index_documents(all_doc_ids, [corpus[d]["text"] for d in all_doc_ids], engine.batch_size)
    else:
        engine.retriever.index_documents(all_doc_ids, [corpus[d]["text"] for d in all_doc_ids], engine.batch_size)

    with open(DUAL) as f:
        dual_records = [json.loads(l) for l in f]
    dual_map = {r["qid"]: r for r in dual_records}
    qids_ch = [q for q in dual_map.keys() if q.endswith("-changed")]

    base_ch = [dual_map[q]["query"] for q in qids_ch]
    neg_ch = [dual_map[q].get("q_minus", "") for q in qids_ch]

    q_base_emb = engine._encode_queries(base_ch).to("cuda")
    q_neg_emb = engine._encode_queries(neg_ch).to("cuda")

    cos_qbase_qneg = F.cosine_similarity(q_base_emb, q_neg_emb, dim=1)  # [Q]
    tau_v5 = cos_qbase_qneg + DELTA

    anchors_map = load_safe_anchors(ANCHOR)

    print("\n" + "=" * 140)
    print(f"{'qid':<16}{'τ_V5目标':>10} | {'anc_max':>9}{'anc_min':>9}{'anc_mean':>9} | "
          f"{'差(max)':>9}{'指导':>12} | 各锚点 S_neg")
    print("-" * 140)

    all_anchor_sneg = {}
    for qi, qid in enumerate(qids_ch):
        anchors = anchors_map.get(qid, [])
        if not anchors:
            continue
        anc_emb = engine._encode_queries(anchors).to("cuda")
        anc_emb = F.normalize(anc_emb, p=2, dim=1)
        qn = F.normalize(q_neg_emb[qi].unsqueeze(0), p=2, dim=1)
        sneg = F.cosine_similarity(qn, anc_emb).cpu().numpy()

        t_v = float(tau_v5[qi])
        a_max, a_min, a_mean = float(sneg.max()), float(sneg.min()), float(sneg.mean())
        diff = a_max - t_v
        if a_max < t_v - 0.02:
            guide = "需调高(+neg)"
        elif a_max > t_v + 0.02:
            guide = "需调低(-neg)"
        else:
            guide = "OK"

        sneg_str = "[" + ", ".join(f"{x:.3f}" for x in sneg) + "]"
        print(f"{qid:<16}{t_v:>10.4f} | {a_max:>9.4f}{a_min:>9.4f}{a_mean:>9.4f} | "
              f"{diff:>+9.4f}{guide:>12} | {sneg_str}")
        all_anchor_sneg[qid] = {
            "tau_v5": t_v, "cos_qbase_qneg": t_v - DELTA,
            "anchor_sneg": [float(x) for x in sneg],
            "anchor_max": a_max, "anchor_min": a_min, "anchor_mean": a_mean,
            "diff_max": diff, "guide": guide,
            "anchors": anchors,
        }

    # 汇总
    diffs = [v["diff_max"] for v in all_anchor_sneg.values()]
    print(f"\n汇总 (n={len(diffs)}):")
    print(f"  差值(max) mean={np.mean(diffs):+.4f}  median={np.median(diffs):+.4f}")
    print(f"  锚点偏低(需+neg): {sum(1 for d in diffs if d < -0.02)}")
    print(f"  锚点偏高(需-neg): {sum(1 for d in diffs if d > 0.02)}")
    print(f"  已在甜点区:       {sum(1 for d in diffs if -0.02 <= d <= 0.02)}")

    out = "results/safe_anchor_analysis/core17_tau_calibration.json"
    with open(out, "w") as f:
        json.dump(all_anchor_sneg, f, indent=2, ensure_ascii=False)
    print(f"  保存: {out}")


if __name__ == "__main__":
    main()
