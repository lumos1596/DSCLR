"""
TRACE-BM25: BM25-based TRACE Evaluator for FollowIR

Applies the TRACE scoring framework (robust standardization + Huber regression +
residual boundary mechanism) using BM25 sparse retrieval scores instead of dense
neural encoder scores. This provides a non-neural baseline for TRACE.

Usage (base mode — z_full_only):
  cd /home/luwa/Documents/DSCLR-remote && \
  /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_trace_bm25 \
    --task_name Core17InstructionRetrieval \
    --dual_queries_path dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_Core17InstructionRetrieval.jsonl \
    --output_dir /home/luwa/Documents/DSCLR/evaluation_remote/cross_retriever_trace/bm25/base/Core17InstructionRetrieval \
    --ablation z_full_only

Usage (trace mode — full):
  cd /home/luwa/Documents/DSCLR-remote && \
  /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_trace_bm25 \
    --task_name Core17InstructionRetrieval \
    --dual_queries_path dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_Core17InstructionRetrieval.jsonl \
    --output_dir /home/luwa/Documents/DSCLR/evaluation_remote/cross_retriever_trace/bm25/trace/Core17InstructionRetrieval \
    --ablation full \
    --huber_delta 1.345 --lambda_boundary 1.0 --tau_decay 0.2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from eval.engine import FollowIRDataLoader
from eval.engine_trace import (
    robust_standardize,
    _mad,
    fit_huber_regression,
    HuberFitResult,
    TRACEQueryResult,
)
from eval.metrics import FollowIREvaluator

logger = logging.getLogger(__name__)


def simple_tokenize(text: str) -> List[str]:
    """Whitespace tokenization + lowercasing (same as engine_beir_bm25.py)."""
    return text.lower().split()


class TRACEBM25Evaluator:
    """BM25-based TRACE evaluator for FollowIR."""

    def __init__(
        self,
        task_name: str,
        output_dir: str,
        dual_queries_path: str,
        huber_delta: float = 1.345,
        lambda_boundary: float = 1.0,
        tau_decay: float = 0.2,
        eps: float = 1e-6,
        ablation: str = "full",
        device: str = "cpu",
    ):
        self.task_name = task_name
        self.output_dir = output_dir
        self.dual_queries_path = dual_queries_path
        self.huber_delta = huber_delta
        self.lambda_boundary = lambda_boundary
        self.tau_decay = tau_decay
        self.eps = eps
        self.ablation = ablation
        self.device = device

        self.data_loader = FollowIRDataLoader(task_name)

        logger.info("TRACE-BM25 evaluator initialized")
        logger.info(f"  huber_delta={self.huber_delta}, lambda={self.lambda_boundary}, tau={self.tau_decay}")
        logger.info(f"  ablation={self.ablation}, device={self.device}")

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def load_dual_queries(self) -> Dict[str, Dict[str, Any]]:
        if not self.dual_queries_path:
            raise ValueError("dual_queries_path is required")
        dual_data: Dict[str, Dict[str, Any]] = {}
        with open(self.dual_queries_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                qid = item["qid"]
                dual_data[qid] = item
        logger.info(f"Loaded {len(dual_data)} dual queries")
        return dual_data

    @staticmethod
    def _is_none_query(text: str) -> bool:
        if not text:
            return True
        t = str(text).strip().upper()
        return t in ("[NONE]", "NONE", "NULL", "N/A", "")

    @staticmethod
    def _get_all_candidate_doc_ids(candidates: Dict[str, List[str]]) -> List[str]:
        all_doc_ids_set = set()
        for doc_ids in candidates.values():
            all_doc_ids_set.update(doc_ids)
        return list(all_doc_ids_set)

    # ------------------------------------------------------------------
    # BM25 index building & scoring
    # ------------------------------------------------------------------

    def build_bm25_index(
        self,
        doc_ids: List[str],
        corpus: Dict[str, Dict[str, str]],
    ) -> Tuple[BM25Okapi, List[str]]:
        """Build a BM25 index over the corpus.

        Returns (bm25, doc_ids) where doc_ids defines the column order.
        """
        doc_texts = [corpus[did]["text"] for did in doc_ids]
        logger.info(f"Tokenizing {len(doc_ids)} documents for BM25...")
        tokenized_corpus = [simple_tokenize(text) for text in tqdm(doc_texts, desc="Tokenizing corpus")]
        logger.info("Building BM25 index...")
        bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built successfully")
        return bm25, doc_ids

    def compute_bm25_scores(
        self,
        bm25: BM25Okapi,
        query_text: str,
    ) -> np.ndarray:
        """Compute BM25 scores for a single query against ALL documents."""
        tokenized_query = simple_tokenize(query_text)
        scores = bm25.get_scores(tokenized_query)
        return scores

    # ------------------------------------------------------------------
    # TRACE scoring (delegates to engine_trace functions)
    # ------------------------------------------------------------------

    def trace_score_query(
        self,
        s_full: torch.Tensor,
        s_pos: torch.Tensor,
        s_neg: torch.Tensor,
        has_neg: bool,
    ) -> TRACEQueryResult:
        """Apply TRACE scoring for a single query's candidate set.

        Mirrors TRACEEvaluator.trace_score_query but is self-contained so we
        don't need to inherit from DSCLREvaluatorEngine.
        """
        stats: Dict[str, Any] = {}
        n = s_full.numel()

        # Step 1: Channel normalization (Eq. 3)
        z_full = robust_standardize(s_full.float(), self.eps)
        z_pos = robust_standardize(s_pos.float(), self.eps)
        z_neg = robust_standardize(s_neg.float(), self.eps)

        stats["n_candidates"] = n
        stats["s_full_median"] = float(s_full.float().median().item())
        stats["s_pos_median"] = float(s_pos.float().median().item())
        stats["s_neg_median"] = float(s_neg.float().median().item())
        stats["s_full_mad"] = float(_mad(s_full.float(), self.eps).item())
        stats["s_pos_mad"] = float(_mad(s_pos.float(), self.eps).item())
        stats["s_neg_mad"] = float(_mad(s_neg.float(), self.eps).item())

        if not has_neg or n < 3:
            p = torch.clamp(z_pos, min=0)
            if self.ablation == "z_full_only":
                s_final = z_full
            else:
                s_final = z_full + p
            return TRACEQueryResult(
                s_final=s_final,
                z_full=z_full,
                z_pos=z_pos,
                z_neg=z_neg,
                r=torch.zeros_like(z_neg),
                p=p,
                h=torch.zeros_like(z_neg),
                g=torch.ones_like(z_neg),
                huber_fit=HuberFitResult(0, 0, 0, 0, 0, n, 0),
                stats=stats,
            )

        # Step 2: Fit Huber regression (Eq. 4)
        a_hat, b_hat = fit_huber_regression(z_neg, z_pos, delta=self.huber_delta)

        # Compute residual
        e = z_neg.float() - a_hat - b_hat * z_pos.float()

        # Pseudo R^2
        ss_res = (e ** 2).sum()
        ss_tot = ((z_neg.float() - z_neg.float().mean()) ** 2).sum()
        r_squared = 1.0 - float(ss_res.item()) / float(ss_tot.item()) if ss_tot > 1e-12 else 0.0

        # Step 3: Re-standardize residual (Eq. 5)
        e_median = e.median()
        e_mad = _mad(e, self.eps)
        r = (e - e_median) / e_mad

        # Step 4: Monotone residual-aware composition (Eqs. 6-8)
        p = torch.clamp(z_pos, min=0)                              # Eq. 6
        h = torch.clamp(r - self.lambda_boundary, min=0)           # Eq. 7
        g = torch.exp(-h / self.tau_decay)                          # Eq. 7

        n_above = int((r > self.lambda_boundary).sum().item())

        # Eq. 8: final score
        if self.ablation == "z_full_only":
            s_final = z_full
        elif self.ablation == "full":
            s_final = z_full + p * g - h
        elif self.ablation == "no_regression":
            r_raw = z_neg
            h_raw = torch.clamp(r_raw - self.lambda_boundary, min=0)
            g_raw = torch.exp(-h_raw / self.tau_decay)
            s_final = z_full + p * g_raw - h_raw
        elif self.ablation == "no_gate":
            s_final = z_full + p - h
        elif self.ablation == "pos_only":
            s_final = z_full + p
        elif self.ablation == "linear":
            s_final = z_full + p - r
        elif self.ablation == "raw_neg_subtract":
            s_final = z_full + p - z_neg
        elif self.ablation == "raw_neg_fusion":
            h_raw = torch.clamp(z_neg - self.lambda_boundary, min=0)
            s_final = z_full + p - h_raw
        elif self.ablation == "gate_only":
            s_final = z_full + p * g
        else:
            s_final = z_full + p * g - h

        _resid_median = float(e_median.item())
        _resid_mad = float(e_mad.item())

        huber_fit = HuberFitResult(
            a_hat=a_hat,
            b_hat=b_hat,
            r_squared=r_squared,
            residual_median=_resid_median,
            residual_mad=_resid_mad,
            n_candidates=n,
            n_above_boundary=n_above,
        )

        stats.update({
            "a_hat": a_hat,
            "b_hat": b_hat,
            "r_squared": r_squared,
            "residual_mad": _resid_mad,
            "n_above_boundary": n_above,
            "above_boundary_ratio": n_above / max(n, 1),
            "p_mean": float(p.mean().item()),
            "p_max": float(p.max().item()),
            "h_mean": float(h.mean().item()),
            "h_max": float(h.max().item()),
            "g_mean": float(g.mean().item()),
            "r_mean": float(r.mean().item()),
            "r_std": float(r.std().item()) if n > 1 else 0.0,
            "r_min": float(r.min().item()),
            "r_max": float(r.max().item()),
        })

        return TRACEQueryResult(
            s_final=s_final,
            z_full=z_full,
            z_pos=z_pos,
            z_neg=z_neg,
            r=r,
            p=p,
            h=h,
            g=g,
            huber_fit=huber_fit,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Core run method
    # ------------------------------------------------------------------

    def run(
        self,
        lambda_list: Optional[List[float]] = None,
        tau_list: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Run TRACE-BM25 evaluation on FollowIR."""
        logger.info("=" * 60)
        logger.info("Starting TRACE-BM25 evaluation")
        logger.info("=" * 60)

        start_time = time.time()

        if lambda_list is None:
            lambda_list = [0.5, 1.0, 1.5, 2.0]
        if tau_list is None:
            tau_list = [0.1, 0.2, 0.5, 1.0]

        total_trials = len(lambda_list) * len(tau_list)
        logger.info(f"Grid search: {total_trials} combinations (lambda x tau)")

        # ------------------------------------------------------------------
        # 1. Load data
        # ------------------------------------------------------------------
        dual_data = self.load_dual_queries()

        corpus, q_og, q_changed, candidates = self.data_loader.load()
        q_raw_og, q_raw_changed = self.data_loader.load_raw_queries()

        # ------------------------------------------------------------------
        # 2. Build BM25 index from corpus
        # ------------------------------------------------------------------
        all_doc_ids = self._get_all_candidate_doc_ids(candidates)
        bm25, doc_id_list = self.build_bm25_index(all_doc_ids, corpus)
        doc_id_to_idx = {did: idx for idx, did in enumerate(doc_id_list)}

        # ------------------------------------------------------------------
        # 3. Build query lists for OG and changed
        # ------------------------------------------------------------------
        def build_query_lists(queries, raw_queries):
            query_ids, q_full_list, q_pos_list, q_neg_list = [], [], [], []
            has_neg_list = []
            for qid in queries.keys():
                query_ids.append(qid)
                raw = raw_queries.get(qid, ("", ""))
                query_text, instruction = raw[0], raw[1]
                q_full = f"{query_text} {instruction}".strip() if query_text else queries.get(qid, "")
                q_full_list.append(q_full)

                d = dual_data.get(qid, {})
                q_plus = d.get("q_plus", "")
                q_minus = d.get("q_minus", "")

                q_pos_list.append(q_plus if not self._is_none_query(q_plus) else "")
                q_neg_list.append(q_minus if not self._is_none_query(q_minus) else "")
                has_neg_list.append(0.0 if self._is_none_query(q_minus) else 1.0)
            return query_ids, q_full_list, q_pos_list, q_neg_list, has_neg_list

        query_ids_og, q_full_og, q_pos_og, q_neg_og, has_neg_og = build_query_lists(q_og, q_raw_og)
        query_ids_ch, q_full_ch, q_pos_ch, q_neg_ch, has_neg_ch = build_query_lists(q_changed, q_raw_changed)

        has_neg_mask_og = torch.tensor(has_neg_og, dtype=torch.float32)
        has_neg_mask_ch = torch.tensor(has_neg_ch, dtype=torch.float32)

        n_docs = len(doc_id_list)

        # ------------------------------------------------------------------
        # 4. Compute BM25 scores for each query channel
        # ------------------------------------------------------------------
        logger.info("Computing BM25 scores for OG queries (S_full, S_pos, S_neg)...")
        S_full_og = torch.zeros(len(query_ids_og), n_docs, dtype=torch.float32)
        S_pos_og = torch.zeros(len(query_ids_og), n_docs, dtype=torch.float32)
        S_neg_og = torch.zeros(len(query_ids_og), n_docs, dtype=torch.float32)

        for i, qid in enumerate(tqdm(query_ids_og, desc="BM25 OG queries")):
            if q_full_og[i]:
                S_full_og[i] = torch.from_numpy(self.compute_bm25_scores(bm25, q_full_og[i]))
            if q_pos_og[i]:
                S_pos_og[i] = torch.from_numpy(self.compute_bm25_scores(bm25, q_pos_og[i]))
            if q_neg_og[i]:
                S_neg_og[i] = torch.from_numpy(self.compute_bm25_scores(bm25, q_neg_og[i]))

        # Apply has_neg mask to S_neg
        S_neg_og = S_neg_og * has_neg_mask_og.unsqueeze(1)

        logger.info("Computing BM25 scores for Changed queries (S_full, S_pos, S_neg)...")
        S_full_ch = torch.zeros(len(query_ids_ch), n_docs, dtype=torch.float32)
        S_pos_ch = torch.zeros(len(query_ids_ch), n_docs, dtype=torch.float32)
        S_neg_ch = torch.zeros(len(query_ids_ch), n_docs, dtype=torch.float32)

        for i, qid in enumerate(tqdm(query_ids_ch, desc="BM25 Changed queries")):
            if q_full_ch[i]:
                S_full_ch[i] = torch.from_numpy(self.compute_bm25_scores(bm25, q_full_ch[i]))
            if q_pos_ch[i]:
                S_pos_ch[i] = torch.from_numpy(self.compute_bm25_scores(bm25, q_pos_ch[i]))
            if q_neg_ch[i]:
                S_neg_ch[i] = torch.from_numpy(self.compute_bm25_scores(bm25, q_neg_ch[i]))

        # Apply has_neg mask to S_neg
        S_neg_ch = S_neg_ch * has_neg_mask_ch.unsqueeze(1)

        # ------------------------------------------------------------------
        # 5. Build candidate index mapping
        # ------------------------------------------------------------------
        qid_to_candidate_indices: Dict[str, List[int]] = {}
        for qid, doc_ids in candidates.items():
            indices = [doc_id_to_idx[d] for d in doc_ids if d in doc_id_to_idx]
            qid_to_candidate_indices[qid] = indices

        # ------------------------------------------------------------------
        # 6. Grid search over lambda x tau
        # ------------------------------------------------------------------
        best_metrics = None
        best_params = None
        best_results_og = None
        best_results_changed = None
        best_per_query_stats: List[Dict[str, Any]] = []
        all_results: List[Dict[str, Any]] = []

        trial_idx = 0
        for lam in lambda_list:
            for tau_d in tau_list:
                trial_idx += 1
                self.lambda_boundary = lam
                self.tau_decay = tau_d

                S_final_ch, per_query_stats = self._compute_trace_scores(
                    S_full_ch, S_pos_ch, S_neg_ch,
                    has_neg_mask_ch, query_ids_ch, qid_to_candidate_indices,
                )

                results_og = self._extract_results(S_full_og, query_ids_og, candidates, doc_id_list)
                results_changed = self._extract_results(S_final_ch, query_ids_ch, candidates, doc_id_list)

                evaluator = FollowIREvaluator(self.task_name)
                metrics = evaluator.evaluate(results_og, results_changed)

                p_mrr = metrics.get("p-MRR", 0.0)
                og_map = metrics.get("original", {}).get("map_at_1000", 0.0)
                changed_map = metrics.get("changed", {}).get("map_at_1000", 0.0)
                changed_ndcg = metrics.get("changed", {}).get("ndcg_at_5", 0.0)

                logger.info(
                    "[%d/%d] lambda=%.1f, tau=%.2f: p-MRR=%.4f, Changed_MAP=%.4f, Changed_nDCG@5=%.4f",
                    trial_idx, total_trials,
                    lam, tau_d,
                    p_mrr, changed_map, changed_ndcg,
                )

                # Compute target_avg for this single task
                if "News21" in self.task_name:
                    target_avg = changed_ndcg
                else:
                    target_avg = changed_map

                all_results.append({
                    "lambda": lam,
                    "tau_decay": tau_d,
                    "huber_delta": self.huber_delta,
                    "ablation": self.ablation,
                    "p-MRR": p_mrr,
                    "target_avg": target_avg,
                    "og_MAP@1000": og_map,
                    "changed_MAP@1000": changed_map,
                    "changed_nDCG@5": changed_ndcg,
                })

                composite = p_mrr + changed_map + changed_ndcg
                if best_metrics is None:
                    best_metrics = metrics
                    best_params = {"lambda": lam, "tau_decay": tau_d}
                    best_results_og = results_og
                    best_results_changed = results_changed
                    best_per_query_stats = list(per_query_stats)
                else:
                    best_composite = (
                        best_metrics.get("p-MRR", 0.0)
                        + best_metrics.get("changed", {}).get("map_at_1000", 0.0)
                        + best_metrics.get("changed", {}).get("ndcg_at_5", 0.0)
                    )
                    if composite > best_composite:
                        best_metrics = metrics
                        best_params = {"lambda": lam, "tau_decay": tau_d}
                        best_results_og = results_og
                        best_results_changed = results_changed
                        best_per_query_stats = list(per_query_stats)

        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info("TRACE-BM25 evaluation complete")
        logger.info(f"  Best: lambda={best_params['lambda']}, tau={best_params['tau_decay']}")
        logger.info(f"  p-MRR: {best_metrics.get('p-MRR', 0.0):.4f}")
        logger.info(f"  Changed MAP@1000: {best_metrics.get('changed', {}).get('map_at_1000', 0.0):.4f}")
        logger.info(f"  Elapsed: {elapsed:.1f}s")
        logger.info("=" * 60)

        self._save_results(
            best_params, best_metrics, all_results,
            best_results_og, best_results_changed,
            best_per_query_stats,
        )

        return {
            "best_params": best_params,
            "best_metrics": best_metrics,
            "all_results": all_results,
            "elapsed": elapsed,
        }

    # ------------------------------------------------------------------
    # TRACE batch scoring
    # ------------------------------------------------------------------

    def _compute_trace_scores(
        self,
        S_full: torch.Tensor,
        S_pos: torch.Tensor,
        S_neg: torch.Tensor,
        has_neg_mask: torch.Tensor,
        query_ids: List[str],
        qid_to_candidate_indices: Dict[str, List[int]],
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Apply TRACE scoring to all changed queries."""
        S_final = S_full.clone()
        all_stats: List[Dict[str, Any]] = []

        for q_idx, qid in enumerate(query_ids):
            base_qid = qid.replace("-og", "").replace("-changed", "")
            cand_indices = qid_to_candidate_indices.get(base_qid, [])
            if not cand_indices:
                continue

            idx_tensor = torch.tensor(cand_indices, dtype=torch.long)
            s_f = S_full[q_idx].index_select(0, idx_tensor)
            s_p = S_pos[q_idx].index_select(0, idx_tensor)
            s_n = S_neg[q_idx].index_select(0, idx_tensor)
            has_neg = bool(has_neg_mask[q_idx].item() > 0)

            result = self.trace_score_query(s_f, s_p, s_n, has_neg)

            s_final_local = result.s_final.to(dtype=S_final.dtype)
            S_final[q_idx, idx_tensor] = s_final_local

            stat = result.stats
            stat["qid"] = qid
            stat["has_neg"] = has_neg
            all_stats.append(stat)

        return S_final, all_stats

    # ------------------------------------------------------------------
    # Result extraction (mirrors DSCLREvaluatorEngine._extract_results)
    # ------------------------------------------------------------------

    def _extract_results(
        self,
        S_final: torch.Tensor,
        query_ids: List[str],
        candidates: Dict[str, List[str]],
        doc_id_list: List[str],
        top_k: int = 1000,
    ) -> Dict[str, Dict[str, float]]:
        """Extract ranking results from the score matrix."""
        results = {}
        doc_id_to_col_idx = {doc_id: idx for idx, doc_id in enumerate(doc_id_list)}

        for idx, qid in enumerate(query_ids):
            base_qid = qid.replace("-og", "").replace("-changed", "")

            if base_qid not in candidates or not candidates[base_qid]:
                continue

            doc_ids = candidates[base_qid]
            scores = S_final[idx].cpu().float().numpy()

            doc_scores = {}
            for doc_id in doc_ids:
                if doc_id in doc_id_to_col_idx:
                    col_idx = doc_id_to_col_idx[doc_id]
                    doc_scores[doc_id] = float(scores[col_idx])

            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            if top_k > 0:
                results[qid] = dict(sorted_docs[:top_k])
            else:
                results[qid] = dict(sorted_docs)

        return results

    # ------------------------------------------------------------------
    # Save results (follows engine_trace.py _save_results format)
    # ------------------------------------------------------------------

    def _save_results(
        self,
        best_params: Dict[str, Any],
        best_metrics: Dict[str, Any],
        all_results: List[Dict[str, Any]],
        results_og: Dict[str, Dict[str, float]],
        results_changed: Dict[str, Dict[str, float]],
        per_query_stats: List[Dict[str, Any]],
    ) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        summary = {
            "task": self.task_name,
            "model": "BM25",
            "mode": "TRACE-BM25",
            "dual_queries_source": self.dual_queries_path,
            "fixed_params": {
                "huber_delta": self.huber_delta,
                "ablation": self.ablation,
            },
            "timestamp": datetime.now().isoformat(),
            "best_params": best_params,
            "metrics": {
                "p-MRR": best_metrics.get("p-MRR", 0.0),
                "original": best_metrics.get("original", {}),
                "changed": best_metrics.get("changed", {}),
            },
        }

        summary_path = os.path.join(self.output_dir, "trace_metrics_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        all_results_path = os.path.join(self.output_dir, "trace_all_results.json")
        with open(all_results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        if per_query_stats:
            pqs_path = os.path.join(self.output_dir, "trace_per_query_stats.json")
            with open(pqs_path, "w", encoding="utf-8") as f:
                json.dump(per_query_stats, f, indent=2, ensure_ascii=False)

        out_og = os.path.join(self.output_dir, "ranking_og.json")
        out_changed = os.path.join(self.output_dir, "ranking_changed.json")
        with open(out_og, "w", encoding="utf-8") as f:
            json.dump(results_og, f, ensure_ascii=False)
        with open(out_changed, "w", encoding="utf-8") as f:
            json.dump(results_changed, f, ensure_ascii=False)

        logger.info(f"Results saved to {self.output_dir}")


def run_trace_bm25(
    task_name: str = "Core17InstructionRetrieval",
    output_dir: Optional[str] = None,
    dual_queries_path: Optional[str] = None,
    huber_delta: float = 1.345,
    lambda_boundary: float = 1.0,
    tau_decay: float = 0.2,
    eps: float = 1e-6,
    ablation: str = "full",
    device: str = "cpu",
) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = f"evaluation/trace_bm25/{task_name}"
    if not dual_queries_path:
        raise ValueError("dual_queries_path is required")

    engine = TRACEBM25Evaluator(
        task_name=task_name,
        output_dir=output_dir,
        dual_queries_path=dual_queries_path,
        huber_delta=huber_delta,
        lambda_boundary=lambda_boundary,
        tau_decay=tau_decay,
        eps=eps,
        ablation=ablation,
        device=device,
    )

    return engine.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TRACE-BM25 evaluator for FollowIR")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dual_queries_path", type=str, required=True)
    parser.add_argument("--huber_delta", type=float, default=1.345)
    parser.add_argument("--lambda_boundary", type=float, default=1.0)
    parser.add_argument("--tau_decay", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=1e-6,
                        help="Numerical floor for MAD standardization")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "z_full_only", "no_regression", "no_gate",
                                 "pos_only", "linear", "raw_neg_subtract",
                                 "raw_neg_fusion", "gate_only"])
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (BM25 is CPU-only, use 'cpu')")

    args = parser.parse_args()

    result = run_trace_bm25(
        task_name=args.task_name,
        output_dir=args.output_dir,
        dual_queries_path=args.dual_queries_path,
        huber_delta=args.huber_delta,
        lambda_boundary=args.lambda_boundary,
        tau_decay=args.tau_decay,
        eps=args.eps,
        ablation=args.ablation,
        device=args.device,
    )

    print(f"\nFinal p-MRR: {result['best_metrics'].get('p-MRR', 0.0):.4f}")
