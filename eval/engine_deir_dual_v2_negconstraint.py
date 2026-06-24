"""
NegConstraint evaluation based on DeIR-Dual-V2 scoring architecture.

This script keeps the same core scoring logic as `eval/engine_deir_dual_v2.py`:

    tau = cos(Q_base, Q_neg) + delta
    gap_w = sigmoid((S_neg - S_base) * T_gap)
    safety = 1 - sigmoid((S_neg - tau) * T_safety)
    penalty = min(alpha * softplus(S_neg - tau) * gap_w, S_base * max_penalty_ratio)
    S_final = S_base + beta * S_req * safety - penalty

It evaluates NegConstraint (queries.jsonl / corpus.jsonl / test.tsv) with fixed
parameters and exports detailed metrics for baseline comparison.
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from datetime import datetime
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.models import ModelFactory
from model.reformulator import QueryReformulator

logger = logging.getLogger(__name__)


PAPER_NEGCONSTRAINT_BASELINES = {
    "BM25": {"MAP": 31.4, "nDCG@10": 33.7},
    "Contriever": {"MAP": 31.8, "nDCG@10": 35.7},
    "HyDE": {"MAP": 47.8, "nDCG@10": 53.1},
    "InterR": {"MAP": 52.3, "nDCG@10": 54.5},
    "BGE": {"MAP": 36.3, "nDCG@10": 40.8},
    "BGE w/ LA": {"MAP": 40.8, "nDCG@10": 47.6},
    "BGE w/ CC": {"MAP": 47.8, "nDCG@10": 46.9},
    "NS-IR (LogicLLaMA)": {"MAP": 50.7, "nDCG@10": 55.2},
    "NS-IR (GPT-4o)": {"MAP": 53.3, "nDCG@10": 56.5},
}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_qrels_tsv(path: str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = str(row["query-id"])
            doc_id = str(row["corpus-id"])
            score = int(row["score"])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = score
    return qrels


def clean_segment(text: str) -> str:
    text = text.strip(" \t\n\r,.;:!?()[]{}\"'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_dual_query(query: str) -> Tuple[str, str]:
    """Heuristically split a negative-constraint query into q_plus / q_minus."""
    q = query.strip()
    lower = q.lower()

    # Priority-ordered markers commonly used by NegConstraint queries.
    markers = [
        "but don't mention",
        "but do not mention",
        "do not mention",
        "don't mention",
        "excluding",
        "without",
        "other than",
        "excluding the role of",
        "avoid",
        "avoiding",
        "omitting",
        "except",
    ]

    pos = -1
    marker_used = ""
    for marker in markers:
        m_pos = lower.find(marker)
        if m_pos >= 0 and (pos == -1 or m_pos < pos):
            pos = m_pos
            marker_used = marker

    if pos == -1:
        # No explicit negative clause found; keep only base semantics.
        return clean_segment(q), ""

    q_plus = clean_segment(q[:pos])
    q_minus = clean_segment(q[pos + len(marker_used):])

    # Remove common wrappers in trailing negative clause.
    q_minus = re.sub(r"^(the\s+role\s+of\s+)", "", q_minus, flags=re.IGNORECASE)
    q_minus = clean_segment(q_minus)

    # Fallback safety.
    if not q_plus:
        q_plus = clean_segment(q)
    return q_plus, q_minus


def load_dual_queries_jsonl(path: str) -> Dict[str, Dict[str, str]]:
    dual: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = str(item.get("qid", "")).strip()
            if not qid:
                continue
            dual[qid] = {
                "q_plus": str(item.get("q_plus", "")).strip(),
                "q_minus": str(item.get("q_minus", "")).strip(),
            }
    return dual


def build_dual_queries_for_negconstraint(
    qids_eval: List[str],
    query_map: Dict[str, str],
    args: argparse.Namespace,
) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []

    if args.dual_queries_path:
        logger.info("Loading dual queries from file: %s", args.dual_queries_path)
        dual_map = load_dual_queries_jsonl(args.dual_queries_path)
        missing = [qid for qid in qids_eval if qid not in dual_map]
        if missing:
            raise ValueError(
                f"dual_queries_path 缺少 {len(missing)} 条 qid，例如: {missing[:5]}"
            )

        for qid in qids_eval:
            q = query_map[qid]
            q_plus = dual_map[qid].get("q_plus", "")
            q_minus = dual_map[qid].get("q_minus", "")
            records.append({"qid": qid, "q_base": q, "q_plus": q_plus, "q_minus": q_minus})
        return records

    if args.use_reformulator:
        api_key = args.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        logger.info("Generating dual queries via QueryReformulator (DeepSeek API)")
        reformulator_kwargs = {
            "task_name": args.reformulator_task_name,
            "use_cache": not args.disable_reformulator_cache,
            "cache_dir": args.reformulator_cache_dir,
            "prompt_version": args.reformulator_prompt_version,
        }
        # 优先使用命令行/环境变量提供的 key；若为空则回退到 reformulator 默认值。
        if api_key:
            reformulator_kwargs["api_key"] = api_key

        reformulator = QueryReformulator(**reformulator_kwargs)

        for idx, qid in enumerate(qids_eval):
            q = query_map[qid]
            q_plus, q_minus = reformulator.reformulate(
                qid=qid,
                idx=idx,
                query=q,
                instruction="",
                query_type="changed",
            )
            records.append({"qid": qid, "q_base": q, "q_plus": q_plus, "q_minus": q_minus})

        failed = reformulator.get_failed_summary()
        if failed.get("total_failed", 0) > 0:
            logger.warning("Reformulator failed queries: %d (see %s)", failed["total_failed"], failed["log_file"])
        return records

    raise ValueError(
        "必须二选一提供 dual query 来源: --dual_queries_path 或 --use_reformulator。"
    )


def score_query_dual_v2(
    s_base: torch.Tensor,
    s_req: torch.Tensor,
    s_neg: torch.Tensor,
    cos_qbase_qneg: float,
    has_req: bool,
    has_neg: bool,
    alpha: float,
    beta: float,
    delta: float,
    t_gap: float,
    t_safety: float,
    max_penalty_ratio: float,
) -> Tuple[torch.Tensor, float]:
    if not has_neg:
        s_req_eff = s_req if has_req else torch.zeros_like(s_base)
        s_final = s_base + beta * s_req_eff
        return s_final, 0.0

    tau = cos_qbase_qneg + delta
    overflow = s_neg - tau
    smooth_penalty = F.softplus(overflow)
    gap_w = torch.sigmoid((s_neg - s_base) * t_gap)
    raw_penalty = alpha * smooth_penalty * gap_w

    if max_penalty_ratio > 0:
        penalty_cap = s_base * max_penalty_ratio
        penalty = torch.min(raw_penalty, penalty_cap)
    else:
        penalty = raw_penalty

    safety = 1.0 - torch.sigmoid((s_neg - tau) * t_safety)
    s_req_eff = s_req if has_req else torch.zeros_like(s_base)
    s_final = s_base + beta * s_req_eff * safety - penalty
    return s_final, float(penalty.mean().item())


def query_type_from_qid(qid: str) -> str:
    q = int(qid)
    if q < 136:
        return "A-a"
    if q < 259:
        return "(A-a)U(B)"
    return "(A-a)U(B-b)"


def evaluate_scores(
    score_by_query: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    cutoffs: List[int],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    agg = {
        "MAP": 0.0,
        "MRR@10": 0.0,
    }
    for k in cutoffs:
        agg[f"nDCG@{k}"] = 0.0
        agg[f"Recall@{k}"] = 0.0
        agg[f"P@{k}"] = 0.0

    per_query: List[Dict[str, Any]] = []

    qids = sorted(qrels.keys(), key=lambda x: int(x))
    for qid in qids:
        rels = qrels[qid]
        ranked = sorted(score_by_query[qid].items(), key=lambda x: x[1], reverse=True)
        ranked_doc_ids = [d for d, _ in ranked]
        n_rel = sum(1 for v in rels.values() if v > 0)

        ap_sum = 0.0
        hit_count = 0
        rr10 = 0.0
        first_rel_rank = None

        for rank, doc_id in enumerate(ranked_doc_ids, start=1):
            rel = 1 if rels.get(doc_id, 0) > 0 else 0
            if rel:
                if first_rel_rank is None:
                    first_rel_rank = rank
                hit_count += 1
                ap_sum += hit_count / rank
                if rank <= 10 and rr10 == 0.0:
                    rr10 = 1.0 / rank

        map_q = ap_sum / n_rel if n_rel > 0 else 0.0

        q_metrics: Dict[str, Any] = {
            "qid": qid,
            "query_type": query_type_from_qid(qid),
            "num_relevant": n_rel,
            "first_relevant_rank": first_rel_rank,
            "MAP": map_q,
            "MRR@10": rr10,
        }

        agg["MAP"] += map_q
        agg["MRR@10"] += rr10

        for k in cutoffs:
            topk = ranked_doc_ids[:k]
            rel_hits = sum(1 for d in topk if rels.get(d, 0) > 0)
            precision = rel_hits / k
            recall = rel_hits / n_rel if n_rel > 0 else 0.0

            dcg = 0.0
            for i, d in enumerate(topk):
                rel = 1 if rels.get(d, 0) > 0 else 0
                if rel:
                    dcg += rel / np.log2(i + 2)

            ideal_hits = min(n_rel, k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits)) if ideal_hits > 0 else 0.0
            ndcg = dcg / idcg if idcg > 0 else 0.0

            q_metrics[f"nDCG@{k}"] = ndcg
            q_metrics[f"Recall@{k}"] = recall
            q_metrics[f"P@{k}"] = precision

            agg[f"nDCG@{k}"] += ndcg
            agg[f"Recall@{k}"] += recall
            agg[f"P@{k}"] += precision

        per_query.append(q_metrics)

    denom = float(len(qids)) if qids else 1.0
    final_metrics = {k: v / denom for k, v in agg.items()}
    return final_metrics, per_query


def metrics_by_type(per_query: List[Dict[str, Any]], cutoffs: List[int]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for item in per_query:
        buckets.setdefault(item["query_type"], []).append(item)

    out: Dict[str, Dict[str, float]] = {}
    keys = ["MAP", "MRR@10"] + [f"nDCG@{k}" for k in cutoffs]
    for t, rows in buckets.items():
        out[t] = {}
        for key in keys:
            out[t][key] = float(np.mean([r[key] for r in rows])) if rows else 0.0
        out[t]["count"] = len(rows)
    return out


def run(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.output_dir, exist_ok=True)

    queries_rows = load_jsonl(os.path.join(args.dataset_dir, "queries.jsonl"))
    corpus_rows = load_jsonl(os.path.join(args.dataset_dir, "corpus.jsonl"))
    qrels_raw = load_qrels_tsv(os.path.join(args.dataset_dir, "test.tsv"))

    query_map = {str(x["_id"]): str(x["text"]) for x in queries_rows}
    if args.eval_scope == "all_queries":
        qids_eval = sorted(query_map.keys(), key=lambda x: int(x))
    else:
        qids_eval = sorted(qrels_raw.keys(), key=lambda x: int(x))

    qrels: Dict[str, Dict[str, int]] = {}
    for qid in qids_eval:
        qrels[qid] = qrels_raw.get(qid, {})

    labeled_qids = [qid for qid in qids_eval if qid in qrels_raw]
    unlabeled_qids = [qid for qid in qids_eval if qid not in qrels_raw]

    qids_with_labels = sum(1 for qid in qids_eval if qid in qrels_raw)
    qids_without_labels = len(qids_eval) - qids_with_labels

    unlabeled_type_counter = Counter(query_type_from_qid(qid) for qid in unlabeled_qids)

    if qids_without_labels > 0:
        logger.warning(
            "Eval scope=%s includes %d queries without qrels labels; their metrics contribute as 0.",
            args.eval_scope,
            qids_without_labels,
        )

    doc_ids = [str(x["_id"]) for x in corpus_rows]
    doc_texts = [str(x.get("text", "")) for x in corpus_rows]

    logger.info(
        "Loaded NegConstraint: %d queries (eval=%d, labeled=%d, unlabeled=%d), %d docs",
        len(query_map),
        len(qids_eval),
        qids_with_labels,
        qids_without_labels,
        len(doc_ids),
    )

    q_base_list: List[str] = []
    q_req_list: List[str] = []
    q_neg_list: List[str] = []
    has_req_mask: List[float] = []
    has_neg_mask: List[float] = []

    dual_queries_records = build_dual_queries_for_negconstraint(qids_eval, query_map, args)
    for row in dual_queries_records:
        qid = row["qid"]
        q = row["q_base"]
        q_plus = row["q_plus"]
        q_minus = row["q_minus"]
        q_base_list.append(q)
        q_req_list.append(q_plus if q_plus else q)
        q_neg_list.append(q_minus if q_minus else q)
        has_req_mask.append(1.0 if q_plus else 0.0)
        has_neg_mask.append(1.0 if q_minus else 0.0)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = ModelFactory.create(
        model_name=args.model_name,
        device=device,
        batch_size=args.batch_size,
        normalize_embeddings=True,
    )

    logger.info("Encoding corpus...")
    d_emb = encoder.encode_documents(doc_texts, batch_size=args.batch_size).to(device)

    logger.info("Encoding queries: base / req / neg...")
    q_base_emb = encoder.encode_queries(q_base_list, batch_size=args.batch_size).to(device)
    q_req_emb = encoder.encode_queries(q_req_list, batch_size=args.batch_size).to(device)
    q_neg_emb = encoder.encode_queries(q_neg_list, batch_size=args.batch_size).to(device)

    has_req = torch.tensor(has_req_mask, dtype=torch.float32, device=device)
    has_neg = torch.tensor(has_neg_mask, dtype=torch.float32, device=device)

    s_base = torch.matmul(q_base_emb, d_emb.T)
    s_req = torch.matmul(q_req_emb, d_emb.T)
    s_neg = torch.matmul(q_neg_emb, d_emb.T) * has_neg.unsqueeze(1)

    cos_qbase_qneg = F.cosine_similarity(q_base_emb, q_neg_emb, dim=1)

    logger.info("Scoring with DeIR-Dual-V2 fixed params alpha=%.3f beta=%.3f delta=%.3f", args.alpha, args.beta, args.delta)
    s_final = s_base.clone()
    penalties: List[float] = []

    top_k = min(args.top_k, len(doc_ids))
    for i, qid in enumerate(qids_eval):
        idx_tensor = torch.topk(s_base[i], k=top_k, largest=True).indices
        s_b = s_base[i].index_select(0, idx_tensor)
        s_r = s_req[i].index_select(0, idx_tensor)
        s_n = s_neg[i].index_select(0, idx_tensor)

        s_local, avg_penalty = score_query_dual_v2(
            s_base=s_b,
            s_req=s_r,
            s_neg=s_n,
            cos_qbase_qneg=float(cos_qbase_qneg[i].item()),
            has_req=bool(has_req[i].item() > 0),
            has_neg=bool(has_neg[i].item() > 0),
            alpha=args.alpha,
            beta=args.beta,
            delta=args.delta,
            t_gap=args.t_gap,
            t_safety=args.t_safety,
            max_penalty_ratio=args.max_penalty_ratio,
        )

        s_final[i, idx_tensor] = s_local.to(dtype=s_final.dtype)
        penalties.append(avg_penalty)

    base_scores_by_query: Dict[str, Dict[str, float]] = {}
    deir_scores_by_query: Dict[str, Dict[str, float]] = {}
    for i, qid in enumerate(qids_eval):
        base_row = s_base[i].detach().float().cpu().numpy()
        deir_row = s_final[i].detach().float().cpu().numpy()
        base_scores_by_query[qid] = {doc_ids[j]: float(base_row[j]) for j in range(len(doc_ids))}
        deir_scores_by_query[qid] = {doc_ids[j]: float(deir_row[j]) for j in range(len(doc_ids))}

    cutoffs = [1, 5, 10, 100, 1000]
    baseline_metrics, baseline_per_query = evaluate_scores(base_scores_by_query, qrels, cutoffs)
    deir_metrics, deir_per_query = evaluate_scores(deir_scores_by_query, qrels, cutoffs)

    labeled_qrels = {qid: qrels_raw[qid] for qid in labeled_qids}
    labeled_base_scores = {qid: base_scores_by_query[qid] for qid in labeled_qids}
    labeled_deir_scores = {qid: deir_scores_by_query[qid] for qid in labeled_qids}

    baseline_metrics_labeled, baseline_per_query_labeled = evaluate_scores(
        labeled_base_scores,
        labeled_qrels,
        cutoffs,
    )
    deir_metrics_labeled, deir_per_query_labeled = evaluate_scores(
        labeled_deir_scores,
        labeled_qrels,
        cutoffs,
    )

    baseline_by_type = metrics_by_type(baseline_per_query, cutoffs)
    deir_by_type = metrics_by_type(deir_per_query, cutoffs)
    baseline_by_type_labeled = metrics_by_type(baseline_per_query_labeled, cutoffs)
    deir_by_type_labeled = metrics_by_type(deir_per_query_labeled, cutoffs)

    summary = {
        "dataset": "NegConstraint",
        "timestamp": datetime.now().isoformat(),
        "model": args.model_name,
        "device": device,
        "eval_scope": args.eval_scope,
        "eval_query_count": len(qids_eval),
        "eval_queries_with_labels": qids_with_labels,
        "eval_queries_without_labels": qids_without_labels,
        "eval_unlabeled_queries_by_type": dict(unlabeled_type_counter),
        "corpus_size": len(doc_ids),
        "candidate_top_k": top_k,
        "deir_dual_v2_params": {
            "alpha": args.alpha,
            "beta": args.beta,
            "delta": args.delta,
            "t_gap": args.t_gap,
            "t_safety": args.t_safety,
            "max_penalty_ratio": args.max_penalty_ratio,
        },
        "dual_query_source": {
            "from_file": args.dual_queries_path,
            "use_reformulator": bool(args.use_reformulator),
            "reformulator_task_name": args.reformulator_task_name if args.use_reformulator else None,
            "reformulator_cache_dir": args.reformulator_cache_dir if args.use_reformulator else None,
            "reformulator_cache_enabled": (not args.disable_reformulator_cache) if args.use_reformulator else None,
        },
        "deir_dual_v2": {
            **deir_metrics,
            "avg_penalty": float(np.mean(penalties)) if penalties else 0.0,
            "by_type": deir_by_type,
        },
        "deir_dual_v2_labeled_only": {
            **deir_metrics_labeled,
            "avg_penalty": float(np.mean(penalties)) if penalties else 0.0,
            "query_count": len(labeled_qids),
            "by_type": deir_by_type_labeled,
        },
        "baseline_base_only": {
            **baseline_metrics,
            "by_type": baseline_by_type,
        },
        "baseline_base_only_labeled_only": {
            **baseline_metrics_labeled,
            "query_count": len(labeled_qids),
            "by_type": baseline_by_type_labeled,
        },
        "paper_table3_baselines": PAPER_NEGCONSTRAINT_BASELINES,
        "comparison_to_paper": {
            "vs_BGE_MAP": float(deir_metrics["MAP"] * 100.0 - PAPER_NEGCONSTRAINT_BASELINES["BGE"]["MAP"]),
            "vs_BGE_nDCG@10": float(deir_metrics["nDCG@10"] * 100.0 - PAPER_NEGCONSTRAINT_BASELINES["BGE"]["nDCG@10"]),
            "vs_NSIR_GPT4o_MAP": float(deir_metrics["MAP"] * 100.0 - PAPER_NEGCONSTRAINT_BASELINES["NS-IR (GPT-4o)"]["MAP"]),
            "vs_NSIR_GPT4o_nDCG@10": float(deir_metrics["nDCG@10"] * 100.0 - PAPER_NEGCONSTRAINT_BASELINES["NS-IR (GPT-4o)"]["nDCG@10"]),
        },
    }

    with open(os.path.join(args.output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "dual_queries_auto.jsonl"), "w", encoding="utf-8") as f:
        for row in dual_queries_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(os.path.join(args.output_dir, "per_query_deir.jsonl"), "w", encoding="utf-8") as f:
        for row in deir_per_query:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(os.path.join(args.output_dir, "per_query_baseline.jsonl"), "w", encoding="utf-8") as f:
        for row in baseline_per_query:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Saved results to %s", args.output_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NegConstraint DeIR-Dual-V2 fixed-parameter evaluation")
    parser.add_argument("--dataset_dir", type=str, default="dataset/NegConstraint/NegConstraint")
    parser.add_argument("--output_dir", type=str, default="evaluation/deir_dual_v2/negconstraint_fixed_a0.5_b1.1_d0.0")
    parser.add_argument("--eval_scope", type=str, choices=["test", "all_queries"], default="test")
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.1)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--t_gap", type=float, default=10.0)
    parser.add_argument("--t_safety", type=float, default=10.0)
    parser.add_argument("--max_penalty_ratio", type=float, default=0.5)

    parser.add_argument("--dual_queries_path", type=str, default=None)
    parser.add_argument("--use_reformulator", action="store_true")
    parser.add_argument("--deepseek_api_key", type=str, default=None)
    parser.add_argument("--reformulator_task_name", type=str, default="NegConstraint")
    parser.add_argument("--reformulator_cache_dir", type=str, default="dataset/NegConstraint/dual_queries_v4")
    parser.add_argument("--reformulator_prompt_version", type=str, default="v4")
    parser.add_argument("--disable_reformulator_cache", action="store_true")

    return parser


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = build_parser().parse_args()
    summary = run(args)
    print("\n=== NegConstraint DeIR-Dual-V2 Fixed Run ===")
    print(f"MAP: {summary['deir_dual_v2']['MAP']:.4f}")
    print(f"nDCG@10: {summary['deir_dual_v2']['nDCG@10']:.4f}")
    print(f"MRR@10: {summary['deir_dual_v2']['MRR@10']:.4f}")
    print(f"vs Paper BGE MAP: {summary['comparison_to_paper']['vs_BGE_MAP']:.2f} points")
    print(f"vs Paper BGE nDCG@10: {summary['comparison_to_paper']['vs_BGE_nDCG@10']:.2f} points")