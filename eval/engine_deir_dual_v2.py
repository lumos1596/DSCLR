"""
DeIR-Dual V2: 免训练奖惩双轨制评估引擎

##在conda环境dsclr中运行

## 评测相关：两个重要指标
### target_avg 定义（重要！）
**target_avg = (Core17_changed_MAP@1000 + Robust04_changed_MAP@1000 + News21_changed_nDCG@5) / 3**
除非用户特别说明，否则"平均指标"均指此定义，而非三个数据集 MAP 的简单平均。

### pMRR: 衡量指令敏感度

核心公式:
    τ = Cos(Q_base, Q_neg) + δ                           (动态语义阈值)
    safety = 1 - sigmoid((S_neg - τ) × T_safety)          (安全门控)
    penalty = α × Softplus(S_neg - τ)                      (平滑惩罚)
    S_final = S_base + β × S_req × safety - penalty        (条件性奖励)

V1 → V2 三大升级:
    1. 动态语义阈值: τ = Cos(Q_base, Q_neg) + δ (替代 mean(S_neg) + δ)
    2. Softplus 平滑惩罚: 替代 ReLU 硬截断
    3. 条件性奖励: safety 门控防止踩雷文档被推高
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F

from eval.engine_dscrl import DSCLREvaluatorEngine, load_cached_embeddings
from eval.metrics import FollowIREvaluator

logger = logging.getLogger(__name__)


class DeIRDualV2Evaluator(DSCLREvaluatorEngine):
    """DeIR-Dual V2 免训练奖惩双轨制评估引擎"""

    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_dir: str,
        dual_queries_path: str,
        t_safety: float = 20.0,
        device: str = "auto",
        **kwargs,
    ):
        self.dual_queries_path = dual_queries_path
        self.t_safety = t_safety
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

        logger.info("🏛️ DeIR-Dual V2 奖惩双轨制模式已启用")
        logger.info(f"📁 Dual queries 数据路径: {self.dual_queries_path}")
        logger.info(f"⚙️ T_safety=%.1f", self.t_safety)

    def load_dual_queries(self) -> Dict[str, Dict[str, Any]]:
        if not self.dual_queries_path:
            raise ValueError("必须提供 dual_queries_path")

        dual_data: Dict[str, Dict[str, Any]] = {}
        with open(self.dual_queries_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                qid = item["qid"]
                dual_data[qid] = item

        logger.info(f"✅ 加载 dual queries 数据: {len(dual_data)} 条")
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

    def _score_query_dual_v2(
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
        """DeIR-Dual V2 核心打分函数。

        τ = Cos(Q_base, Q_neg) + δ
        safety = 1 - sigmoid((S_neg - τ) × T_safety)
        penalty = α × Softplus(S_neg - τ)
        S_final = S_base + β × S_req × safety - penalty
        """
        if not has_neg:
            s_req_eff = s_req if has_req else torch.zeros_like(s_base)
            s_final = s_base + beta * s_req_eff
            return s_final, 0.0

        tau = cos_qbase_qneg + delta

        overflow = s_neg - tau
        smooth_penalty = F.softplus(overflow)

        raw_penalty = alpha * smooth_penalty

        safety = 1.0 - torch.sigmoid((s_neg - tau) * self.t_safety)

        s_req_eff = s_req if has_req else torch.zeros_like(s_base)
        s_final = s_base + beta * s_req_eff * safety - raw_penalty

        avg_penalty = float(raw_penalty.mean().item())
        return s_final, avg_penalty

    def compute_deir_dual_v2_scores(
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

            s_final_local, avg_penalty = self._score_query_dual_v2(
                s_base=s_b,
                s_req=s_r,
                s_neg=s_n,
                cos_qbase_qneg=cos_val,
                has_req=has_req,
                has_neg=has_neg,
                alpha=alpha,
                beta=beta,
                delta=delta,
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
        logger.info("=" * 60)
        logger.info("🚀 开始 DeIR-Dual V2 奖惩双轨制评测")
        logger.info("=" * 60)

        start_time = time.time()

        alpha_list = alphas if alphas else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        beta_list = betas if betas else [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        delta_list = deltas if deltas else [-0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

        total_trials = len(alpha_list) * len(beta_list) * len(delta_list)
        logger.info(f"🔬 DeIR-Dual V2 3D 网格搜索规模: {total_trials} 组")
        logger.info(f"   α 范围: {alpha_list}")
        logger.info(f"   β 范围: {beta_list}")
        logger.info(f"   δ 范围: {delta_list}")
        logger.info(f"   T_safety=%.1f", self.t_safety)

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
                logger.info(f"✅ 使用缓存文档向量 ({len(cached_doc_ids)} 个)")
                self.retriever.set_embeddings(cached_embeddings, cached_doc_ids)
            else:
                logger.warning("⚠️ 缓存文档ID不匹配，重新编码...")
                doc_texts = [corpus[did]["text"] for did in all_doc_ids]
                self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)
        else:
            logger.info("📚 编码候选文档...")
            doc_texts = [corpus[did]["text"] for did in all_doc_ids]
            self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)

        query_ids_og: List[str] = []
        q_base_list_og: List[str] = []
        q_req_list_og: List[str] = []
        q_neg_list_og: List[str] = []
        has_req_mask_og_list: List[float] = []
        has_neg_mask_og_list: List[float] = []

        for qid in q_og.keys():
            query_ids_og.append(qid)
            raw = q_raw_og.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            q_base = f"{query_text} {instruction}".strip() if query_text else q_og.get(qid, "")
            q_base_list_og.append(q_base)

            d = dual_data.get(qid, {})
            q_plus = d.get("q_plus", "")
            q_minus = d.get("q_minus", "")

            q_req_list_og.append(q_plus if not self._is_none_query(q_plus) else "")
            q_neg_list_og.append(q_minus if not self._is_none_query(q_minus) else "")
            has_req_mask_og_list.append(0.0 if self._is_none_query(q_plus) else 1.0)
            has_neg_mask_og_list.append(0.0 if self._is_none_query(q_minus) else 1.0)

        query_ids_changed: List[str] = []
        q_base_list_changed: List[str] = []
        q_req_list_changed: List[str] = []
        q_neg_list_changed: List[str] = []
        has_req_mask_changed_list: List[float] = []
        has_neg_mask_changed_list: List[float] = []

        for qid in q_changed.keys():
            query_ids_changed.append(qid)
            raw = q_raw_changed.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            q_base = f"{query_text} {instruction}".strip() if query_text else q_changed.get(qid, "")
            q_base_list_changed.append(q_base)

            d = dual_data.get(qid, {})
            q_plus = d.get("q_plus", "")
            q_minus = d.get("q_minus", "")

            q_req_list_changed.append(q_plus if not self._is_none_query(q_plus) else "")
            q_neg_list_changed.append(q_minus if not self._is_none_query(q_minus) else "")
            has_req_mask_changed_list.append(0.0 if self._is_none_query(q_plus) else 1.0)
            has_neg_mask_changed_list.append(0.0 if self._is_none_query(q_minus) else 1.0)

        has_req_mask_og = torch.tensor(has_req_mask_og_list, dtype=torch.float32)
        has_neg_mask_og = torch.tensor(has_neg_mask_og_list, dtype=torch.float32)
        has_req_mask_changed = torch.tensor(has_req_mask_changed_list, dtype=torch.float32)
        has_neg_mask_changed = torch.tensor(has_neg_mask_changed_list, dtype=torch.float32)

        logger.info("📊 编码 OG Q_base...")
        q_base_emb_og = self._encode_queries(q_base_list_og)
        logger.info("📊 编码 OG Q_req (Q+)...")
        q_req_emb_og = self._encode_queries(q_req_list_og)
        logger.info("📊 编码 OG Q_neg (Q-)...")
        q_neg_emb_og = self._encode_queries(q_neg_list_og)

        logger.info("📊 编码 Changed Q_base...")
        q_base_emb_changed = self._encode_queries(q_base_list_changed)
        logger.info("📊 编码 Changed Q_req (Q+)...")
        q_req_emb_changed = self._encode_queries(q_req_list_changed)
        logger.info("📊 编码 Changed Q_neg (Q-)...")
        q_neg_emb_changed = self._encode_queries(q_neg_list_changed)

        device = self.retriever.doc_embeddings.device
        q_base_emb_og = q_base_emb_og.to(device)
        q_req_emb_og = q_req_emb_og.to(device)
        q_neg_emb_og = q_neg_emb_og.to(device)
        q_base_emb_changed = q_base_emb_changed.to(device)
        q_req_emb_changed = q_req_emb_changed.to(device)
        q_neg_emb_changed = q_neg_emb_changed.to(device)
        has_req_mask_og = has_req_mask_og.to(device)
        has_neg_mask_og = has_neg_mask_og.to(device)
        has_req_mask_changed = has_req_mask_changed.to(device)
        has_neg_mask_changed = has_neg_mask_changed.to(device)

        logger.info("📊 计算 S_base / S_req / S_neg...")
        S_base_og = torch.matmul(q_base_emb_og, self.retriever.doc_embeddings.T)
        S_req_og = torch.matmul(q_req_emb_og, self.retriever.doc_embeddings.T)
        S_neg_og = torch.matmul(q_neg_emb_og, self.retriever.doc_embeddings.T) * has_neg_mask_og.unsqueeze(1)

        S_base_changed = torch.matmul(q_base_emb_changed, self.retriever.doc_embeddings.T)
        S_req_changed = torch.matmul(q_req_emb_changed, self.retriever.doc_embeddings.T)
        S_neg_changed = torch.matmul(q_neg_emb_changed, self.retriever.doc_embeddings.T) * has_neg_mask_changed.unsqueeze(1)

        logger.info("📊 计算 Cos(Q_base, Q_neg) 动态语义阈值...")
        cos_qbase_qneg_og = F.cosine_similarity(q_base_emb_og, q_neg_emb_og, dim=1)
        cos_qbase_qneg_changed = F.cosine_similarity(q_base_emb_changed, q_neg_emb_changed, dim=1)

        logger.info(f"   Cos(Q_base, Q_neg) 统计 (changed): min=%.4f, max=%.4f, mean=%.4f",
                     cos_qbase_qneg_changed.min().item(),
                     cos_qbase_qneg_changed.max().item(),
                     cos_qbase_qneg_changed.mean().item())

        doc_id_to_col_idx = {doc_id: idx for idx, doc_id in enumerate(self.retriever.doc_ids)}
        qid_to_candidate_indices = self._build_candidate_indices(candidates, doc_id_to_col_idx)

        best_metrics = None
        best_params = None
        best_results_og = None
        best_results_changed = None
        all_results: List[Dict[str, Any]] = []

        trial_idx = 0
        for alpha in alpha_list:
            for beta in beta_list:
                for delta in delta_list:
                    trial_idx += 1

                    S_final_changed, penalty_scores = self.compute_deir_dual_v2_scores(
                        S_base=S_base_changed,
                        S_req=S_req_changed,
                        S_neg=S_neg_changed,
                        cos_qbase_qneg=cos_qbase_qneg_changed,
                        has_req_mask=has_req_mask_changed,
                        has_neg_mask=has_neg_mask_changed,
                        query_ids=query_ids_changed,
                        qid_to_candidate_indices=qid_to_candidate_indices,
                        alpha=alpha,
                        beta=beta,
                        delta=delta,
                    )

                    results_og = self._extract_results(S_base_og, query_ids_og, candidates)
                    results_changed = self._extract_results(S_final_changed, query_ids_changed, candidates)

                    evaluator = FollowIREvaluator(self.task_name)
                    metrics = evaluator.evaluate(results_og, results_changed)

                    p_mrr = metrics.get("p-MRR", 0.0)
                    og_ndcg = metrics.get("original", {}).get("ndcg_at_5", 0.0)
                    changed_ndcg = metrics.get("changed", {}).get("ndcg_at_5", 0.0)
                    og_map = metrics.get("original", {}).get("map_at_1000", 0.0)
                    changed_map = metrics.get("changed", {}).get("map_at_1000", 0.0)
                    og_mrr = metrics.get("original", {}).get("mrr_at_10", 0.0)
                    changed_mrr = metrics.get("changed", {}).get("mrr_at_10", 0.0)

                    logger.info(
                        "[%d/%d] α=%.1f, β=%.1f, δ=%.2f: "
                        "p-MRR=%.4f, OG_MAP=%.4f, Changed_MAP=%.4f, Changed_MRR=%.4f",
                        trial_idx, total_trials,
                        alpha, beta, delta,
                        p_mrr, og_map, changed_map, changed_mrr,
                    )

                    all_results.append({
                        "alpha": alpha,
                        "beta": beta,
                        "delta": delta,
                        "t_safety": self.t_safety,
                        "p-MRR": p_mrr,
                        "og_nDCG@5": og_ndcg,
                        "og_nDCG@10": metrics.get("original", {}).get("ndcg_at_10", 0.0),
                        "og_MAP@1000": og_map,
                        "og_MRR@10": og_mrr,
                        "changed_nDCG@5": changed_ndcg,
                        "changed_nDCG@10": metrics.get("changed", {}).get("ndcg_at_10", 0.0),
                        "changed_MAP@1000": changed_map,
                        "changed_MRR@10": changed_mrr,
                        "avg_penalty": float(penalty_scores.mean().item()),
                    })

                    composite_score = p_mrr + changed_map + changed_ndcg
                    if best_metrics is None:
                        best_metrics = metrics
                        best_params = {"alpha": alpha, "beta": beta, "delta": delta}
                        best_results_og = results_og
                        best_results_changed = results_changed
                    else:
                        best_composite = (
                            best_metrics.get("p-MRR", 0.0)
                            + best_metrics.get("changed", {}).get("map_at_1000", 0.0)
                            + best_metrics.get("changed", {}).get("ndcg_at_5", 0.0)
                        )
                        if composite_score > best_composite:
                            best_metrics = metrics
                            best_params = {"alpha": alpha, "beta": beta, "delta": delta}
                            best_results_og = results_og
                            best_results_changed = results_changed

        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info("📊 DeIR-Dual V2 3D 搜索完成")
        logger.info(
            "   最佳参数: α=%.1f, β=%.1f, δ=%.2f",
            best_params["alpha"], best_params["beta"], best_params["delta"],
        )
        logger.info("   p-MRR: %.4f", best_metrics.get("p-MRR", 0.0))
        logger.info("   OG MAP@1000: %.4f", best_metrics.get("original", {}).get("map_at_1000", 0.0))
        logger.info("   Changed MAP@1000: %.4f", best_metrics.get("changed", {}).get("map_at_1000", 0.0))
        logger.info("   Changed MRR@10: %.4f", best_metrics.get("changed", {}).get("mrr_at_10", 0.0))
        logger.info("   耗时: %.1f 秒", elapsed)
        logger.info("=" * 60)

        self._save_results(
            best_params=best_params,
            best_metrics=best_metrics,
            all_results=all_results,
            results_og=best_results_og,
            results_changed=best_results_changed,
        )

        return {
            "best_params": best_params,
            "best_metrics": best_metrics,
            "all_results": all_results,
            "elapsed": elapsed,
        }

    def _save_results(
        self,
        best_params: Dict[str, Any],
        best_metrics: Dict[str, Any],
        all_results: List[Dict[str, Any]],
        results_og: Dict[str, Dict[str, float]],
        results_changed: Dict[str, Dict[str, float]],
    ) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        summary = {
            "task": self.task_name,
            "model": self.model_name,
            "mode": "DeIR-Dual-V2",
            "dual_queries_source": self.dual_queries_path,
            "fixed_params": {
                "t_safety": self.t_safety,
            },
            "timestamp": datetime.now().isoformat(),
            "best_params": best_params,
            "metrics": {
                "p-MRR": best_metrics.get("p-MRR", 0.0),
                "original": best_metrics.get("original", {}),
                "changed": best_metrics.get("changed", {}),
                "full_scores": best_metrics.get("full_scores", {}),
            },
        }

        summary_path = os.path.join(self.output_dir, "metrics_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        metrics_path = os.path.join(self.output_dir, "metrics_deir_dual_v2.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        all_results_path = os.path.join(self.output_dir, "all_results.json")
        with open(all_results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        out_og = os.path.join(self.output_dir, "ranking_og.json")
        out_changed = os.path.join(self.output_dir, "ranking_changed.json")
        with open(out_og, "w", encoding="utf-8") as f:
            json.dump(results_og, f, ensure_ascii=False)
        with open(out_changed, "w", encoding="utf-8") as f:
            json.dump(results_changed, f, ensure_ascii=False)

        logger.info(f"💾 指标已保存: {metrics_path}")
        logger.info(f"💾 参数明细已保存: {all_results_path}")


def run_deir_dual_v2(
    task_name: str = "Core17InstructionRetrieval",
    model_name: str = "samaya-ai/RepLLaMA-reproduced",
    output_dir: Optional[str] = None,
    dual_queries_path: Optional[str] = None,
    alphas: Optional[List[float]] = None,
    betas: Optional[List[float]] = None,
    deltas: Optional[List[float]] = None,
    t_safety: float = 20.0,
    use_cache: bool = True,
    device: str = "auto",
    batch_size: int = 64,
) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = f"evaluation/deir_dual_v2/{task_name}"

    if not dual_queries_path:
        raise ValueError("dual_queries_path 不能为空")

    engine = DeIRDualV2Evaluator(
        model_name=model_name,
        task_name=task_name,
        output_dir=output_dir,
        dual_queries_path=dual_queries_path,
        t_safety=t_safety,
        use_cache=use_cache,
        device=device,
        batch_size=batch_size,
    )

    return engine.run(
        alphas=alphas,
        betas=betas,
        deltas=deltas,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeIR-Dual V2 奖惩双轨制评估引擎")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval")
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dual_queries_path", type=str, required=True)

    parser.add_argument("--alphas", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--betas", type=str, default="0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5")
    parser.add_argument("--deltas", type=str, default="-0.10,-0.05,0.00,0.05,0.10,0.15,0.20,0.25,0.30")

    parser.add_argument("--t_safety", type=float, default=20.0)

    parser.add_argument("--use_cache", type=str, default="true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    alphas_list = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    betas_list = [float(x.strip()) for x in args.betas.split(",") if x.strip()]
    deltas_list = [float(x.strip()) for x in args.deltas.split(",") if x.strip()]
    use_cache = args.use_cache.lower() == "true"

    result = run_deir_dual_v2(
        task_name=args.task_name,
        model_name=args.model_name,
        output_dir=args.output_dir,
        dual_queries_path=args.dual_queries_path,
        alphas=alphas_list,
        betas=betas_list,
        deltas=deltas_list,
        t_safety=args.t_safety,
        use_cache=use_cache,
        device=args.device,
        batch_size=args.batch_size,
    )

    print(f"\n最终 p-MRR: {result['best_metrics'].get('p-MRR', 0.0):.4f}")
