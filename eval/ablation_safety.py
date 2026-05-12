"""
Safety Gate Ablation: 对比有/无 safety 门控的 DeIR-Dual V2 性能

有 safety:   S_final = S_base + β × S_req × safety - penalty
无 safety:   S_final = S_base + β × S_req         - penalty   (safety=1.0 恒定)

控制变量: 相同的编码器、相同的 dual_queries、相同的参数网格
只改变: safety 门控是否启用
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import logging
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

from eval.engine_dscrl import DSCLREvaluatorEngine, load_cached_embeddings
from eval.metrics import FollowIREvaluator

logger = logging.getLogger(__name__)


class SafetyAblationEngine(DSCLREvaluatorEngine):
    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_dir: str,
        dual_queries_path: str,
        use_safety: bool = True,
        t_gap: float = 20.0,
        t_safety: float = 20.0,
        max_penalty_ratio: float = 0.0,
        device: str = "auto",
        **kwargs,
    ):
        self.dual_queries_path = dual_queries_path
        self.use_safety = use_safety
        self.t_gap = t_gap
        self.t_safety = t_safety
        self.max_penalty_ratio = max_penalty_ratio
        if device == "auto":
            import torch as _torch
            try:
                _torch.cuda._lazy_init()
                device = "cuda" if _torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        kwargs.setdefault("device", device)
        kwargs.setdefault("batch_size", kwargs.pop("batch_size", 64))
        super().__init__(model_name, task_name, output_dir, **kwargs)

        mode = "WITH_SAFETY" if use_safety else "NO_SAFETY"
        logger.info(f"🔬 Safety Ablation: {mode}")
        logger.info(f"📁 Dual queries: {self.dual_queries_path}")

    def load_dual_queries(self) -> Dict[str, Dict[str, Any]]:
        dual_data: Dict[str, Dict[str, Any]] = {}
        with open(self.dual_queries_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                qid = item.get("qid", item.get("query_id", ""))
                dual_data[qid] = item
        logger.info(f"✅ Loaded dual queries: {len(dual_data)} entries")
        return dual_data

    def _build_candidate_indices(
        self,
        candidates: Dict[str, List[str]],
        doc_id_to_col_idx: Dict[str, int],
    ) -> Dict[str, List[int]]:
        qid_to_indices: Dict[str, List[int]] = {}
        for qid, doc_ids in candidates.items():
            indices = [doc_id_to_col_idx[d] for d in doc_ids if d in doc_id_to_col_idx]
            qid_to_indices[qid] = indices
        return qid_to_indices

    def _is_none_query(self, text: str) -> bool:
        if not text:
            return True
        t = str(text).strip().upper()
        return t in ("[NONE]", "NONE", "NULL", "N/A", "")

    def _score_query(
        self,
        s_base: torch.Tensor,
        s_req: torch.Tensor,
        s_neg: torch.Tensor,
        cos_qbase_qneg: float,
        has_req: bool,
        has_neg: bool,
        alpha: float,
        beta: float,
        delta: float,
    ) -> Tuple[torch.Tensor, float]:
        if not has_neg:
            s_req_eff = s_req if has_req else torch.zeros_like(s_base)
            s_final = s_base + beta * s_req_eff
            return s_final, 0.0

        tau = cos_qbase_qneg + delta

        overflow = s_neg - tau
        smooth_penalty = F.softplus(overflow)

        gap_w = torch.sigmoid((s_neg - s_base) * self.t_gap)

        raw_penalty = alpha * smooth_penalty * gap_w

        if self.max_penalty_ratio > 0:
            penalty_cap = s_base * self.max_penalty_ratio
            penalty = torch.min(raw_penalty, penalty_cap)
        else:
            penalty = raw_penalty

        if self.use_safety:
            safety = 1.0 - torch.sigmoid((s_neg - tau) * self.t_safety)
        else:
            safety = 1.0

        s_req_eff = s_req if has_req else torch.zeros_like(s_base)
        s_final = s_base + beta * s_req_eff * safety - penalty

        avg_penalty = float(penalty.mean().item())
        return s_final, avg_penalty

    def compute_scores(
        self,
        S_base: torch.Tensor,
        S_req: torch.Tensor,
        S_neg: torch.Tensor,
        cos_qbase_qneg: torch.Tensor,
        has_req_mask: torch.Tensor,
        has_neg_mask: torch.Tensor,
        query_ids: List[str],
        qid_to_candidate_indices: Dict[str, List[int]],
        alpha: float,
        beta: float,
        delta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        S_final = S_base.clone()
        penalty_scores = torch.zeros(len(query_ids), device=S_base.device)

        for q_idx, qid in enumerate(query_ids):
            base_qid = qid.replace("-og", "").replace("-changed", "")
            cand_indices = qid_to_candidate_indices.get(base_qid, [])
            if not cand_indices:
                continue

            idx_tensor = torch.tensor(cand_indices, device=S_base.device, dtype=torch.long)
            s_b = S_base[q_idx].index_select(0, idx_tensor)
            s_r = S_req[q_idx].index_select(0, idx_tensor)
            s_n = S_neg[q_idx].index_select(0, idx_tensor)

            has_req = bool(has_req_mask[q_idx].item() > 0)
            has_neg = bool(has_neg_mask[q_idx].item() > 0)
            cos_val = float(cos_qbase_qneg[q_idx].item())

            s_final_local, avg_penalty = self._score_query(
                s_base=s_b, s_req=s_r, s_neg=s_n,
                cos_qbase_qneg=cos_val,
                has_req=has_req, has_neg=has_neg,
                alpha=alpha, beta=beta, delta=delta,
            )

            s_final_local = s_final_local.to(dtype=S_final.dtype)
            S_final[q_idx, idx_tensor] = s_final_local
            penalty_scores[q_idx] = avg_penalty

        return S_final, penalty_scores

    def run(
        self,
        alphas: Optional[List[float]] = None,
        betas: Optional[List[float]] = None,
        deltas: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        alpha_list = alphas if alphas else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        beta_list = betas if betas else [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        delta_list = deltas if deltas else [-0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

        total_trials = len(alpha_list) * len(beta_list) * len(delta_list)
        mode_str = "WITH_SAFETY" if self.use_safety else "NO_SAFETY"
        logger.info(f"🔬 Safety Ablation [{mode_str}]: {total_trials} trials")

        dual_data = self.load_dual_queries()
        corpus, q_og, q_changed, candidates = self.data_loader.load()
        q_raw_og, q_raw_changed = self.data_loader.load_raw_queries()

        all_doc_ids = self._get_all_candidate_doc_ids(candidates)

        cached_data = None
        if self.use_cache:
            cached_data = load_cached_embeddings(self.cache_dir, self.task_name, self.model_name)

        if cached_data is not None:
            cached_embeddings, cached_doc_ids = cached_data
            if set(cached_doc_ids) == set(all_doc_ids):
                logger.info(f"✅ Using cached embeddings ({len(cached_doc_ids)})")
                self.retriever.set_embeddings(cached_embeddings, cached_doc_ids)
            else:
                logger.warning("⚠️ Cache mismatch, re-encoding...")
                doc_texts = [corpus[did]["text"] for did in all_doc_ids]
                self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)
        else:
            doc_texts = [corpus[did]["text"] for did in all_doc_ids]
            self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)

        def prepare_queries(q_dict, q_raw):
            qids, q_bases, q_reqs, q_negs = [], [], [], []
            req_masks, neg_masks = [], []
            for qid in q_dict.keys():
                qids.append(qid)
                raw = q_raw.get(qid, ("", ""))
                query_text, instruction = raw[0], raw[1]
                q_base = f"{query_text} {instruction}".strip() if query_text else q_dict.get(qid, "")
                q_bases.append(q_base)
                d = dual_data.get(qid, {})
                q_plus = d.get("q_plus", "")
                q_minus = d.get("q_minus", "")
                q_reqs.append(q_plus if not self._is_none_query(q_plus) else "")
                q_negs.append(q_minus if not self._is_none_query(q_minus) else "")
                req_masks.append(0.0 if self._is_none_query(q_plus) else 1.0)
                neg_masks.append(0.0 if self._is_none_query(q_minus) else 1.0)
            return qids, q_bases, q_reqs, q_negs, torch.tensor(req_masks), torch.tensor(neg_masks)

        query_ids_og, q_base_og, q_req_og, q_neg_og, has_req_og, has_neg_og = prepare_queries(q_og, q_raw_og)
        query_ids_ch, q_base_ch, q_req_ch, q_neg_ch, has_req_ch, has_neg_ch = prepare_queries(q_changed, q_raw_changed)

        logger.info("📊 Encoding queries...")
        q_base_emb_og = self._encode_queries(q_base_og)
        q_req_emb_og = self._encode_queries(q_req_og)
        q_neg_emb_og = self._encode_queries(q_neg_og)
        q_base_emb_ch = self._encode_queries(q_base_ch)
        q_req_emb_ch = self._encode_queries(q_req_ch)
        q_neg_emb_ch = self._encode_queries(q_neg_ch)

        device = self.retriever.doc_embeddings.device
        q_base_emb_og = q_base_emb_og.to(device)
        q_req_emb_og = q_req_emb_og.to(device)
        q_neg_emb_og = q_neg_emb_og.to(device)
        q_base_emb_ch = q_base_emb_ch.to(device)
        q_req_emb_ch = q_req_emb_ch.to(device)
        q_neg_emb_ch = q_neg_emb_ch.to(device)
        has_req_og = has_req_og.to(device)
        has_neg_og = has_neg_og.to(device)
        has_req_ch = has_req_ch.to(device)
        has_neg_ch = has_neg_ch.to(device)

        logger.info("📊 Computing similarity scores...")
        S_base_og = torch.matmul(q_base_emb_og, self.retriever.doc_embeddings.T)
        S_req_og = torch.matmul(q_req_emb_og, self.retriever.doc_embeddings.T)
        S_neg_og = torch.matmul(q_neg_emb_og, self.retriever.doc_embeddings.T) * has_neg_og.unsqueeze(1)
        S_base_ch = torch.matmul(q_base_emb_ch, self.retriever.doc_embeddings.T)
        S_req_ch = torch.matmul(q_req_emb_ch, self.retriever.doc_embeddings.T)
        S_neg_ch = torch.matmul(q_neg_emb_ch, self.retriever.doc_embeddings.T) * has_neg_ch.unsqueeze(1)

        cos_qbase_qneg_og = F.cosine_similarity(q_base_emb_og, q_neg_emb_og, dim=1)
        cos_qbase_qneg_ch = F.cosine_similarity(q_base_emb_ch, q_neg_emb_ch, dim=1)

        doc_id_to_col_idx = {doc_id: idx for idx, doc_id in enumerate(self.retriever.doc_ids)}
        qid_to_candidate_indices = self._build_candidate_indices(candidates, doc_id_to_col_idx)

        all_results: List[Dict[str, Any]] = []
        trial_idx = 0

        for alpha in alpha_list:
            for beta in beta_list:
                for delta in delta_list:
                    trial_idx += 1

                    S_final_ch, penalty_scores = self.compute_scores(
                        S_base=S_base_ch, S_req=S_req_ch, S_neg=S_neg_ch,
                        cos_qbase_qneg=cos_qbase_qneg_ch,
                        has_req_mask=has_req_ch, has_neg_mask=has_neg_ch,
                        query_ids=query_ids_ch,
                        qid_to_candidate_indices=qid_to_candidate_indices,
                        alpha=alpha, beta=beta, delta=delta,
                    )

                    results_og = self._extract_results(S_base_og, query_ids_og, candidates)
                    results_changed = self._extract_results(S_final_ch, query_ids_ch, candidates)

                    evaluator = FollowIREvaluator(self.task_name)
                    metrics = evaluator.evaluate(results_og, results_changed)

                    p_mrr = metrics.get("p-MRR", 0.0)
                    og_map = metrics.get("original", {}).get("map_at_1000", 0.0)
                    changed_map = metrics.get("changed", {}).get("map_at_1000", 0.0)
                    changed_ndcg5 = metrics.get("changed", {}).get("ndcg_at_5", 0.0)

                    all_results.append({
                        "alpha": alpha, "beta": beta, "delta": delta,
                        "use_safety": self.use_safety,
                        "p-MRR": p_mrr,
                        "og_MAP@1000": og_map,
                        "changed_MAP@1000": changed_map,
                        "changed_nDCG@5": changed_ndcg5,
                        "avg_penalty": float(penalty_scores.mean().item()),
                    })

                    if trial_idx % 50 == 0:
                        logger.info(f"[{trial_idx}/{total_trials}] α={alpha}, β={beta}, δ={delta}: p-MRR={p_mrr:.4f}, ch_MAP={changed_map:.5f}")

        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, "all_results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"💾 Saved {len(all_results)} results to {out_path}")

        return {"all_results": all_results}


def load_dual_queries(path: str) -> Dict[str, Dict[str, Any]]:
    dual_data: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            dual_data[item["query_id"]] = item
    return dual_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Safety Gate Ablation")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--dual_queries_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_safety", type=str, default="true")
    parser.add_argument("--alphas", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--betas", type=str, default="0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5")
    parser.add_argument("--deltas", type=str, default="-0.10,-0.05,0.00,0.05,0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--use_cache", type=str, default="true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    use_safety = args.use_safety.lower() == "true"
    alphas_list = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    betas_list = [float(x.strip()) for x in args.betas.split(",") if x.strip()]
    deltas_list = [float(x.strip()) for x in args.deltas.split(",") if x.strip()]
    use_cache = args.use_cache.lower() == "true"

    engine = SafetyAblationEngine(
        model_name=args.model_name,
        task_name=args.task_name,
        output_dir=args.output_dir,
        dual_queries_path=args.dual_queries_path,
        use_safety=use_safety,
        use_cache=use_cache,
        device=args.device,
        batch_size=args.batch_size,
    )

    engine.run(alphas=alphas_list, betas=betas_list, deltas=deltas_list)
