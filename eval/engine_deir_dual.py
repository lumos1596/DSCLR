"""
DeIR-Dual: 免训练奖惩双轨制评估引擎

核心公式:
    S_final = S_base + β × S_req - α × ReLU(S_neg - τ)
    τ = mean(S_neg) + δ

三流意图特征:
    S_base = sim(Q_base, D)   基础意图得分 (原始查询+指令)
    S_req  = sim(Q_req, D)    正向奖励得分 (Q+ 编码)
    S_neg  = sim(Q_neg, D)    负向踩雷得分 (Q- 编码)

数据源: dual_queries_v5 格式 (每查询含 q_plus / q_minus 字段)

三维网格搜索:
    α (alpha): 惩罚力度 [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    β (beta):  奖励力度 [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    δ (delta): 底噪偏移 [-0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
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

from eval.engine_dscrl import DSCLREvaluatorEngine, load_cached_embeddings
from eval.metrics import FollowIREvaluator

logger = logging.getLogger(__name__)


class DeIRDualEvaluator(DSCLREvaluatorEngine):
    """DeIR-Dual 免训练奖惩双轨制评估引擎"""

    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_dir: str,
        dual_queries_path: str,
        device: str = "auto",
        **kwargs,
    ):
        self.dual_queries_path = dual_queries_path
        if device == "auto":
            import torch as _torch
            device = "cuda" if _torch.cuda.is_available() else "cpu"
        kwargs.setdefault("device", device)
        super().__init__(model_name, task_name, output_dir, **kwargs)

        logger.info("🏛️ DeIR-Dual 奖惩双轨制模式已启用")
        logger.info(f"📁 Dual queries 数据路径: {self.dual_queries_path}")

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

    def _score_query_dual(
        self,
        s_base: torch.Tensor,
        s_req: torch.Tensor,
        s_neg: torch.Tensor,
        has_req: bool,
        has_neg: bool,
        alpha: float,
        beta: float,
        delta: float,
    ) -> Tuple[torch.Tensor, float]:
        """DeIR-Dual 核心打分函数。

        S_final = S_base + β × S_req - α × ReLU(S_neg - τ)
        τ = mean(S_neg) + δ
        """
        if not has_neg:
            s_req_eff = s_req if has_req else torch.zeros_like(s_base)
            s_final = s_base + beta * s_req_eff
            return s_final, 0.0

        baseline_noise = s_neg.mean()
        tau = baseline_noise + delta

        penalty = torch.relu(s_neg - tau)

        s_req_eff = s_req if has_req else torch.zeros_like(s_base)
        s_final = s_base + beta * s_req_eff - alpha * penalty

        avg_penalty = float(penalty.mean().item())
        return s_final, avg_penalty

    def compute_deir_dual_scores(
        self,
        S_base: torch.Tensor,
        S_req: torch.Tensor,
        S_neg: torch.Tensor,
        has_req_mask: torch.Tensor,
        has_neg_mask: torch.Tensor,
        query_ids: List[str],
        qid_to_candidate_indices: Dict[str, List[int]],
        alpha: float,
        beta: float,
        delta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """按 query 在候选池内计算 DeIR-Dual 最终得分。"""
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

            s_final_local, avg_penalty = self._score_query_dual(
                s_base=s_b,
                s_req=s_r,
                s_neg=s_n,
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
        logger.info("🚀 开始 DeIR-Dual 奖惩双轨制评测")
        logger.info("=" * 60)

        start_time = time.time()

        alpha_list = alphas if alphas else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        beta_list = betas if betas else [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
        delta_list = deltas if deltas else [-0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

        total_trials = len(alpha_list) * len(beta_list) * len(delta_list)
        logger.info(f"🔬 DeIR-Dual 3D 网格搜索规模: {total_trials} 组")
        logger.info(f"   α 范围: {alpha_list}")
        logger.info(f"   β 范围: {beta_list}")
        logger.info(f"   δ 范围: {delta_list}")

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

        # ── 模块一：三流意图特征提取 ──

        # OG 查询: Q_base + Q_req + Q_neg
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

        # Changed 查询: Q_base + Q_req + Q_neg
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

        # 编码三流查询
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

        # ── 计算三流得分矩阵 ──
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

        doc_id_to_col_idx = {doc_id: idx for idx, doc_id in enumerate(self.retriever.doc_ids)}
        qid_to_candidate_indices = self._build_candidate_indices(candidates, doc_id_to_col_idx)

        # ── 模块三：3D 网格搜索 ──
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

                    S_final_changed, penalty_scores = self.compute_deir_dual_scores(
                        S_base=S_base_changed,
                        S_req=S_req_changed,
                        S_neg=S_neg_changed,
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
        logger.info("📊 DeIR-Dual 3D 搜索完成")
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
            "mode": "DeIR-Dual",
            "dual_queries_source": self.dual_queries_path,
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

        metrics_path = os.path.join(self.output_dir, "metrics_deir_dual.json")
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


def run_deir_dual(
    task_name: str = "Core17InstructionRetrieval",
    model_name: str = "samaya-ai/RepLLaMA-reproduced",
    output_dir: Optional[str] = None,
    dual_queries_path: Optional[str] = None,
    alphas: Optional[List[float]] = None,
    betas: Optional[List[float]] = None,
    deltas: Optional[List[float]] = None,
    use_cache: bool = True,
    device: str = "auto",
) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = f"evaluation/deir_dual/{task_name}"

    if not dual_queries_path:
        raise ValueError("dual_queries_path 不能为空")

    engine = DeIRDualEvaluator(
        model_name=model_name,
        task_name=task_name,
        output_dir=output_dir,
        dual_queries_path=dual_queries_path,
        use_cache=use_cache,
        device=device,
    )

    return engine.run(
        alphas=alphas,
        betas=betas,
        deltas=deltas,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DeIR-Dual 奖惩双轨制评估引擎")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval")
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dual_queries_path", type=str, required=True)

    parser.add_argument("--alphas", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--betas", type=str, default="0.1,0.3,0.5,0.7,0.9,1.1,1.3,1.5")
    parser.add_argument("--deltas", type=str, default="-0.10,-0.05,0.00,0.05,0.10,0.15,0.20,0.25,0.30")

    parser.add_argument("--use_cache", type=str, default="true")
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    alphas_list = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    betas_list = [float(x.strip()) for x in args.betas.split(",") if x.strip()]
    deltas_list = [float(x.strip()) for x in args.deltas.split(",") if x.strip()]
    use_cache = args.use_cache.lower() == "true"

    result = run_deir_dual(
        task_name=args.task_name,
        model_name=args.model_name,
        output_dir=args.output_dir,
        dual_queries_path=args.dual_queries_path,
        alphas=alphas_list,
        betas=betas_list,
        deltas=deltas_list,
        use_cache=use_cache,
        device=args.device,
    )

    print(f"\n最终 p-MRR: {result['best_metrics'].get('p-MRR', 0.0):.4f}")
