"""细粒度逐 query 分析：判断负面约束强度，只对真有负面约束的 query 分析 S_neg 区分度。

每个测试集分别分析，每个 query 输出：
  1. 负面约束判定（基于 instruction/q_minus 语义）
  2. 翻转文档 vs 保持相关文档的 S_neg 分布
  3. τ_anchor / τ_V5 是否落在两分布间
  4. 区分度 AUC（翻转文档 S_neg 排序后能否与保持相关文档分开）
"""
import json
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/home/luwa/Documents/DSCLR")

from eval.experiment_safe_anchor_threshold import (
    SafeAnchorDeIREvaluator,
    load_safe_anchors,
    compute_safe_anchor_threshold,
)
from eval.metrics.evaluator import DataLoader

DELTA_V5 = 0.02

# 三个测试集的路径配置
DATASETS = {
    "Core17": {
        "task": "Core17InstructionRetrieval",
        "dual": "dataset/FollowIR_test/dual_queries_4B/TSC_BALANCED_t01/dual_queries_TSC_BALANCED_t01_Core17InstructionRetrieval.jsonl",
        "anchor": "dataset/FollowIR_test/safe_anchors/safe_anchors_core17.json",
    },
    "Robust04": {
        "task": "Robust04InstructionRetrieval",
        "dual": "dataset/FollowIR_test/dual_queries_4B/TSC_BALANCED_t01/dual_queries_TSC_BALANCED_t01_Robust04InstructionRetrieval.jsonl",
        "anchor": "dataset/FollowIR_test/safe_anchors/safe_anchors_robust04.json",
    },
    "News21": {
        "task": "News21InstructionRetrieval",
        "dual": "dataset/FollowIR_test/dual_queries_4B/TSC_BALANCED_t01/dual_queries_TSC_BALANCED_t01_News21InstructionRetrieval.jsonl",
        "anchor": "dataset/FollowIR_test/safe_anchors/safe_anchors_news21.json",
    },
}


def judge_neg_constraint(q_minus: str, instruction: str) -> str:
    """基于 q_minus 和 instruction 判断负面约束强度。

    Returns:
        "strong" / "weak" / "none"
    """
    qm = (q_minus or "").strip()
    # q_minus 为空或显式标记 NONE -> 无负面约束
    if not qm or qm.upper() in ("[NONE]", "NONE", "N/A", ""):
        return "none"
    # 单纯年份、数字 -> 弱约束
    if qm.isdigit() or (len(qm) <= 4 and qm.replace("/", "").isdigit()):
        return "weak"
    # instruction 中是否出现明确排除词
    instr_lower = (instruction or "").lower()
    exclude_keywords = ["not relevant", "exclude", "not include", "must not", "are not",
                        "is not relevant", "should not", "do not", "rather than",
                        "instead of", "without", "ignore"]
    has_exclude = any(k in instr_lower for k in exclude_keywords)
    if has_exclude and len(qm) > 5:
        return "strong"
    # 默认有 q_minus 文本就视为强约束
    return "strong"


def analyze_dataset(name: str, cfg: dict, engine_cache: dict) -> dict:
    """分析单个测试集。"""
    task = cfg["task"]
    dual_path = cfg["dual"]
    anchor_path = cfg["anchor"]

    print("\n" + "#" * 120)
    print(f"# 测试集: {name}  (task={task})")
    print("#" * 120)

    # 复用模型（同 model_name）
    if engine_cache.get("engine") is None:
        engine = SafeAnchorDeIREvaluator(
            model_name="samaya-ai/RepLLaMA-reproduced",
            task_name=task,
            output_dir="results/safe_anchor_analysis/tmp",
            dual_queries_path=dual_path,
            safe_anchors_path=anchor_path,
            anchor_stat="max",
            anchor_delta=0.0,
            anchor_mix_mode="replace",
            device="cuda",
            batch_size=32,
            use_cache=True,
        )
        engine_cache["engine"] = engine
    else:
        # 复用模型，但换数据加载器
        engine = engine_cache["engine"]

    dl = DataLoader(task)
    qrels = dl.load_qrels()
    diff = dl.load_qrel_diff()
    candidates: Dict[str, List[str]] = dl.load_candidates()

    # 加载文档嵌入
    corpus, _, _, _ = engine.data_loader.load()
    all_doc_ids = engine._get_all_candidate_doc_ids(candidates)
    from eval.engine_dscrl import load_cached_embeddings
    cached_data = load_cached_embeddings(engine.cache_dir, task, engine.model_name)
    if cached_data is not None:
        ce, cdids = cached_data
        if set(cdids) == set(all_doc_ids):
            engine.retriever.set_embeddings(ce, cdids)
        else:
            engine.retriever.index_documents(all_doc_ids, [corpus[d]["text"] for d in all_doc_ids], engine.batch_size)
    else:
        engine.retriever.index_documents(all_doc_ids, [corpus[d]["text"] for d in all_doc_ids], engine.batch_size)

    # 加载 dual queries
    with open(dual_path) as f:
        dual_records = [json.loads(l) for l in f]
    dual_map = {r["qid"]: r for r in dual_records}
    qids_ch = [q for q in dual_map.keys() if q.endswith("-changed")]

    base_ch, req_ch, neg_ch = [], [], []
    for qid in qids_ch:
        r = dual_map[qid]
        base_ch.append(r.get("query", ""))
        req_ch.append(r.get("q_plus", ""))
        neg_ch.append(r.get("q_minus", ""))

    print(f"Changed queries 总数: {len(qids_ch)}")

    # 编码
    q_base_emb = engine._encode_queries(base_ch).to("cuda")
    q_neg_emb = engine._encode_queries(neg_ch).to("cuda")
    doc_emb = engine.retriever.doc_embeddings
    doc_ids = engine.retriever.doc_ids
    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}

    S_neg_all = (q_neg_emb @ doc_emb.T)  # [Q, N]
    cos_qbase_qneg = F.cosine_similarity(q_base_emb, q_neg_emb, dim=1)
    tau_v5 = cos_qbase_qneg + DELTA_V5

    anchors_map = load_safe_anchors(anchor_path)
    tau_anchor, per_q_scores = compute_safe_anchor_threshold(
        q_neg_emb, qids_ch, anchors_map, encoder_fn=engine._encode_queries, stat="max"
    )

    # ===== 逐 query 分析 =====
    print("\n" + "=" * 130)
    print(f"{'qid':<14}{'neg约束':>8} | {'τ_V5':>7}{'τ_anc':>7} | "
          f"{'flip_n':>7}{'flip_med':>9}{'flip_min':>9} | "
          f"{'keep_n':>7}{'keep_med':>9}{'keep_max':>9} | "
          f"{'AUC':>6}{'τ在间?':>7}{'判定':>10}")
    print("-" * 130)

    summary = {
        "strong": {"queries": [], "aucs": [], "flip_med": [], "keep_med": [], "tau_in_gap": 0, "n": 0},
        "weak":   {"queries": [], "aucs": [], "flip_med": [], "keep_med": [], "tau_in_gap": 0, "n": 0},
        "none":   {"queries": [], "aucs": [], "flip_med": [], "keep_med": [], "tau_in_gap": 0, "n": 0},
    }
    all_rows = []

    for qi, qid in enumerate(qids_ch):
        base_qid = qid.replace("-changed", "")
        r = dual_map[qid]
        q_minus = r.get("q_minus", "")
        instruction = r.get("instruction", "")

        neg_type = judge_neg_constraint(q_minus, instruction)

        if base_qid not in candidates or base_qid not in diff or f"{base_qid}-changed" not in qrels:
            continue
        if tau_anchor[qi] == float("-inf"):
            continue

        cand = candidates[base_qid]
        cand_idx = [doc_id_to_idx[d] for d in cand if d in doc_id_to_idx]
        s_neg_cand = S_neg_all[qi, cand_idx].cpu().numpy()

        flip_docs = set(diff[base_qid])
        flip_mask = np.array([d in flip_docs for d in cand if d in doc_id_to_idx])
        ch_rel_docs = set(d for d, rel in qrels[f"{base_qid}-changed"].items() if rel > 0)
        keep_mask = np.array([d in ch_rel_docs for d in cand if d in doc_id_to_idx])

        if flip_mask.sum() == 0 or keep_mask.sum() == 0:
            continue

        s_neg_flip = s_neg_cand[flip_mask]
        s_neg_keep = s_neg_cand[keep_mask]

        t_a = float(tau_anchor[qi])
        t_v = float(tau_v5[qi])

        # AUC: 翻转文档应 S_neg 高（label=1），保持相关应 S_neg 低（label=0）
        labels = np.concatenate([np.ones(len(s_neg_flip)), np.zeros(len(s_neg_keep))])
        scores = np.concatenate([s_neg_flip, s_neg_keep])
        try:
            auc = roc_auc_score(labels, scores)
        except ValueError:
            auc = 0.5

        # τ 是否落在 flip_min 与 keep_max 之间（理想分界区）
        flip_min = float(np.min(s_neg_flip))
        keep_max = float(np.max(s_neg_keep))
        in_gap_a = flip_min <= t_a <= keep_max or keep_max <= t_a <= flip_min
        # 更严格：τ 应 >= keep_max 且 <= flip_min（如果 flip S_neg > keep S_neg）
        # 或 τ 落在两分布重叠区外
        if np.median(s_neg_flip) > np.median(s_neg_keep):
            # 理想：keep_max < τ < flip_min
            ideal_gap = keep_max < t_a < flip_min
        else:
            ideal_gap = flip_min < t_a < keep_max
        # 判定
        if np.median(s_neg_flip) > np.median(s_neg_keep) and t_a > keep_max and t_a < flip_min:
            verdict = "理想"
        elif t_a > max(keep_max, np.max(s_neg_flip)):
            verdict = "偏高"
        elif t_a < min(flip_min, keep_max):
            verdict = "偏低"
        else:
            verdict = "重叠区"

        row = {
            "qid": qid, "neg_type": neg_type, "q_minus": q_minus[:50],
            "tau_v5": t_v, "tau_anchor": t_a,
            "flip_n": len(s_neg_flip), "flip_med": float(np.median(s_neg_flip)),
            "flip_min": float(np.min(s_neg_flip)),
            "keep_n": len(s_neg_keep), "keep_med": float(np.median(s_neg_keep)),
            "keep_max": float(np.max(s_neg_keep)),
            "auc": auc, "verdict": verdict,
            "flip_gt_keep_med": float(np.median(s_neg_flip)) > float(np.median(s_neg_keep)),
        }
        all_rows.append(row)

        s = summary[neg_type]
        s["queries"].append(qid)
        s["aucs"].append(auc)
        s["flip_med"].append(float(np.median(s_neg_flip)))
        s["keep_med"].append(float(np.median(s_neg_keep)))
        s["n"] += 1
        if verdict in ("理想",):
            s["tau_in_gap"] += 1

        print(f"{qid:<14}{neg_type:>8} | {t_v:>7.4f}{t_a:>7.4f} | "
              f"{len(s_neg_flip):>7}{float(np.median(s_neg_flip)):>9.4f}{float(np.min(s_neg_flip)):>9.4f} | "
              f"{len(s_neg_keep):>7}{float(np.median(s_neg_keep)):>9.4f}{float(np.max(s_neg_keep)):>9.4f} | "
              f"{auc:>6.3f}{verdict:>10}")

    # ===== 按负面约束强度汇总 =====
    print("\n" + "=" * 130)
    print(f"[{name}] 按负面约束强度汇总:")
    for neg_type in ["strong", "weak", "none"]:
        s = summary[neg_type]
        if s["n"] == 0:
            continue
        aucs = np.array(s["aucs"])
        flip_meds = np.array(s["flip_med"])
        keep_meds = np.array(s["keep_med"])
        sep = flip_meds - keep_meds  # 正值表示翻转文档 S_neg > 保持相关文档
        print(f"  {neg_type:>6} (n={s['n']}):")
        print(f"    AUC: mean={aucs.mean():.3f}  median={np.median(aucs):.3f}  "
              f"min={aucs.min():.3f}  max={aucs.max():.3f}  "
              f"(>0.5={int((aucs>0.5).sum())}/{s['n']}, >0.6={int((aucs>0.6).sum())}/{s['n']})")
        print(f"    S_neg 区分度 (flip_med - keep_med): mean={sep.mean():+.4f}  "
              f">0 的 query: {int((sep>0).sum())}/{s['n']}")
        print(f"    τ_anchor 落在理想分界区: {s['tau_in_gap']}/{s['n']}")

    out = f"results/safe_anchor_analysis/fine_grained_{name.lower()}.json"
    with open(out, "w") as f:
        json.dump({"rows": all_rows, "summary": {
            nt: {"n": s["n"], "auc_mean": float(np.mean(s["aucs"])) if s["aucs"] else 0,
                 "sep_mean": float(np.mean(np.array(s["flip_med"]) - np.array(s["keep_med"]))) if s["flip_med"] else 0}
            for nt, s in summary.items()
        }}, f, indent=2, ensure_ascii=False)
    print(f"  保存: {out}")
    return {"rows": all_rows, "summary": summary}


def main():
    import os
    os.makedirs("results/safe_anchor_analysis", exist_ok=True)
    engine_cache = {}
    all_results = {}
    for name, cfg in DATASETS.items():
        all_results[name] = analyze_dataset(name, cfg, engine_cache)
    # 总汇总
    print("\n\n" + "#" * 120)
    print("# 三测试集总汇总 (仅 strong 负面约束)")
    print("#" * 120)
    for name in DATASETS:
        s = all_results[name]["summary"]["strong"]
        if s["n"] > 0:
            aucs = np.array(s["aucs"])
            print(f"  {name}: strong n={s['n']}, AUC mean={aucs.mean():.3f}, "
                  f">0.5: {int((aucs>0.5).sum())}/{s['n']}, >0.6: {int((aucs>0.6).sum())}/{s['n']}")


if __name__ == "__main__":
    main()
