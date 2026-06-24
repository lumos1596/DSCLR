"""
DeIR-Dual V2 InstructIR Evaluation Engine

Two-stage evaluation pipeline on InstructIR benchmark:
  Stage 1: Dense retrieval with Q_base (query + instruction) -> top-k candidates
  Stage 2: DeIR-Dual V2 reranking on top-k candidates

InstructIR has per-query instance-specific instructions (not og/changed pairs).
Each query-instruction pair has its own set of relevant documents.

Usage:
  cd /home/luwa/Documents/DSCLR && /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_instructir \
    --model_name samaya-ai/RepLLaMA-reproduced \
    --dual_queries_path dataset/InstructIR/dual_queries/InstructIR_TSC_BALANCED_t01.jsonl \
    --alphas 1.0 --betas 1.5 --deltas 0.05 \
    --top_k 100 --device cuda \
    --output_dir results/instructir
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.pop("HF_ENDPOINT", None)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

import json
import logging
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import datasets
import pytrec_eval

logger = logging.getLogger(__name__)


class InstructIRDataLoader:
    def __init__(self):
        self.dataset_path = "mteb/InstructIR-mteb"

    def load_corpus(self) -> Dict[str, Dict[str, str]]:
        logger.info(f"Loading corpus from {self.dataset_path}...")
        ds = datasets.load_dataset(self.dataset_path, 'corpus')['corpus']
        corpus = {}
        for d in tqdm(ds, desc="Loading corpus"):
            doc_id = str(d["_id"])
            text = str(d.get("text", ""))
            corpus[doc_id] = {"text": text}
        logger.info(f"Loaded {len(corpus)} documents")
        return corpus

    def load_queries_and_instructions(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        logger.info(f"Loading queries from {self.dataset_path}...")
        ds_queries = datasets.load_dataset(self.dataset_path, 'queries')['queries']
        ds_inst = datasets.load_dataset(self.dataset_path, 'instruction')['instruction']

        queries = {}
        for q in ds_queries:
            qid = str(q["_id"])
            queries[qid] = str(q.get("text", ""))

        instructions = {}
        for item in ds_inst:
            qid = str(item["query-id"])
            instructions[qid] = str(item.get("instruction", ""))

        logger.info(f"Loaded {len(queries)} queries, {len(instructions)} instructions")
        return queries, instructions

    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        logger.info(f"Loading qrels from {self.dataset_path}...")
        ds = datasets.load_dataset(self.dataset_path, 'default', split='test')
        qrels = {}
        for item in ds:
            qid = str(item["query-id"])
            doc_id = str(item["corpus-id"])
            score = int(item.get("score", 1))
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = score
        logger.info(f"Loaded qrels for {len(qrels)} queries")
        return qrels


class InstructIREvaluator:
    def __init__(
        self,
        model_name: str,
        dual_queries_path: str,
        output_dir: str,
        top_k: int = 100,
        t_safety: float = 10.0,
        device: str = "auto",
        batch_size: int = 64,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.dual_queries_path = dual_queries_path
        self.output_dir = output_dir
        self.top_k = top_k
        self.t_safety = t_safety
        self.batch_size = batch_size

        if device == "auto":
            try:
                torch.cuda._lazy_init()
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device

        self.cache_dir = cache_dir or "dataset/InstructIR/embeddings"
        self.data_loader = InstructIRDataLoader()

        self._init_encoder()

    def _init_encoder(self):
        from eval.models import ModelFactory
        self.encoder = ModelFactory.create(
            model_name=self.model_name,
            device=self.device,
            batch_size=self.batch_size,
            normalize_embeddings=True,
        )
        logger.info(f"Encoder initialized: {self.model_name}")

    def _get_model_short_name(self) -> str:
        if "repllama" in self.model_name.lower():
            return "repllama"
        elif "mistral" in self.model_name.lower():
            return "e5-mistral-7b"
        else:
            return self.model_name.split("/")[-1].replace("-", "_")

    def _get_corpus_cache_path(self) -> str:
        model_short = self._get_model_short_name()
        return os.path.join(self.cache_dir, f"instructir_{model_short}_corpus.pt")

    def _encode_and_cache_corpus(self, corpus: Dict[str, Dict[str, str]]) -> Tuple[torch.Tensor, List[str]]:
        cache_path = self._get_corpus_cache_path()
        if os.path.exists(cache_path):
            logger.info(f"Loading cached corpus embeddings from {cache_path}")
            data = torch.load(cache_path, map_location="cpu")
            doc_ids = data["doc_ids"]
            embeddings = data["embeddings"]
            if len(doc_ids) == len(corpus):
                logger.info(f"Cache hit: {len(doc_ids)} documents, shape={embeddings.shape}")
                return embeddings, doc_ids
            else:
                logger.warning(f"Cache size mismatch (cache={len(doc_ids)}, corpus={len(corpus)}), re-encoding")

        doc_ids = list(corpus.keys())
        doc_texts = [corpus[did]["text"] for did in doc_ids]

        logger.info(f"Encoding {len(doc_ids)} documents...")
        embeddings_list = []
        for i in tqdm(range(0, len(doc_ids), self.batch_size), desc="Encoding corpus"):
            batch_texts = doc_texts[i:i + self.batch_size]
            batch_emb = self.encoder.encode_documents(batch_texts, batch_size=self.batch_size)
            if batch_emb.dim() == 2:
                batch_emb = F.normalize(batch_emb, p=2, dim=1)
            embeddings_list.append(batch_emb.cpu())

        embeddings = torch.cat(embeddings_list, dim=0)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"doc_ids": doc_ids, "embeddings": embeddings}, cache_path)
        logger.info(f"Corpus embeddings cached to {cache_path} (shape={embeddings.shape})")

        return embeddings, doc_ids

    def load_dual_queries(self) -> Dict[str, Dict[str, Any]]:
        if not self.dual_queries_path or not os.path.exists(self.dual_queries_path):
            logger.warning("No dual queries file provided, using Q_plus=Q_base, Q_minus=[NONE]")
            return {}

        dual_data = {}
        with open(self.dual_queries_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                qid = item["qid"]
                dual_data[qid] = item
        logger.info(f"Loaded dual queries: {len(dual_data)} entries")
        return dual_data

    def _is_none_query(self, text: str) -> bool:
        if not text:
            return True
        t = str(text).strip().upper()
        return t in ("[NONE]", "NONE", "NULL", "N/A", "")

    def _score_deir_dual_v2(
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
    ) -> torch.Tensor:
        if not has_neg:
            s_req_eff = s_req if has_req else torch.zeros_like(s_base)
            return s_base + beta * s_req_eff

        tau = cos_qbase_qneg + delta
        overflow = s_neg - tau
        smooth_penalty = F.softplus(overflow)
        raw_penalty = alpha * smooth_penalty
        safety = 1.0 - torch.sigmoid((s_neg - tau) * self.t_safety)
        s_req_eff = s_req if has_req else torch.zeros_like(s_base)
        s_final = s_base + beta * s_req_eff * safety - raw_penalty
        return s_final

    def compute_metrics(
        self,
        results: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
    ) -> Dict[str, float]:
        qrels_eval = {}
        for qid, rel_dict in qrels.items():
            if qid in results:
                qrels_eval[qid] = rel_dict

        if not qrels_eval:
            logger.warning("No overlapping queries between results and qrels")
            return {}

        results_str = {qid: {did: float(score) for did, score in scores.items()} for qid, scores in results.items()}
        qrels_str = {qid: {did: int(rel) for did, rel in rel_dict.items()} for qid, rel_dict in qrels_eval.items()}

        evaluator = pytrec_eval.RelevanceEvaluator(qrels_str, {
            "ndcg_cut.5", "ndcg_cut.10", "ndcg_cut.100",
            "map_cut.100", "map_cut.1000",
            "recall.5", "recall.10", "recall.100", "recall.1000",
            "recip_rank.5", "recip_rank.10",
        })

        scores = evaluator.evaluate(results_str)

        metrics = {}
        metric_keys = [
            "ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_100",
            "map_cut_100", "map_cut_1000",
            "recall_5", "recall_10", "recall_100", "recall_1000",
        ]
        for key in metric_keys:
            values = [s.get(key, 0.0) for s in scores.values()]
            metrics[key] = np.mean(values) if values else 0.0

        mrr_values = [s.get("recip_rank", 0.0) for s in scores.values()]
        metrics["recip_rank_10"] = np.mean(mrr_values) if mrr_values else 0.0

        return metrics

    def _extract_topk_results(
        self,
        S: torch.Tensor,
        top_k_indices: torch.Tensor,
        query_ids: List[str],
        doc_ids: List[str],
        subset_indices: Optional[List[int]] = None,
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        indices = subset_indices if subset_indices else range(len(query_ids))
        for i in indices:
            qid = query_ids[i]
            results[qid] = {}
            for j in range(top_k_indices.shape[1]):
                idx = int(top_k_indices[i, j].item())
                if idx < 0 or idx >= len(doc_ids):
                    continue
                did = doc_ids[idx]
                results[qid][did] = float(S[i, j].item())
        return results

    def run(
        self,
        alphas: List[float],
        betas: List[float],
        deltas: List[float],
    ) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("DeIR-Dual V2 InstructIR Evaluation")
        logger.info("=" * 60)

        start_time = time.time()

        corpus = self.data_loader.load_corpus()
        queries, instructions = self.data_loader.load_queries_and_instructions()
        qrels = self.data_loader.load_qrels()

        eval_qids = sorted(set(qrels.keys()) & set(queries.keys()))
        logger.info(f"Evaluating on {len(eval_qids)} queries (with qrels)")

        doc_embeddings, doc_ids = self._encode_and_cache_corpus(corpus)
        doc_id_to_idx = {did: idx for idx, did in enumerate(doc_ids)}
        doc_embeddings = doc_embeddings.to(self.device)

        dual_data = self.load_dual_queries()

        query_ids = eval_qids
        q_base_list = []
        q_req_list = []
        q_neg_list = []
        has_req_mask = []
        has_neg_mask = []

        for qid in query_ids:
            query_text = queries[qid]
            instruction = instructions.get(qid, "")
            if instruction:
                q_base = f"instruction: {instruction} [SEP] {query_text}"
            else:
                q_base = query_text
            q_base_list.append(q_base)

            d = dual_data.get(qid, {})
            q_plus = d.get("q_plus", "")
            q_minus = d.get("q_minus", "")

            if not q_plus or self._is_none_query(q_plus):
                q_plus = q_base
                has_req_mask.append(0.0)
            else:
                has_req_mask.append(1.0)

            if not q_minus or self._is_none_query(q_minus):
                q_minus = ""
                has_neg_mask.append(0.0)
            else:
                has_neg_mask.append(1.0)

            q_req_list.append(q_plus)
            q_neg_list.append(q_minus)

        has_req_mask_t = torch.tensor(has_req_mask, dtype=torch.float32, device=self.device)
        has_neg_mask_t = torch.tensor(has_neg_mask, dtype=torch.float32, device=self.device)

        logger.info(f"Q_minus statistics: {sum(has_neg_mask)}/{len(has_neg_mask)} queries have Q_minus")

        logger.info("Encoding Q_base...")
        q_base_emb = self.encoder.encode_queries(q_base_list, batch_size=self.batch_size)
        if q_base_emb.dim() == 2:
            q_base_emb = F.normalize(q_base_emb, p=2, dim=1)
        q_base_emb = q_base_emb.to(self.device)

        logger.info("Encoding Q_req (Q+)...")
        q_req_emb = self.encoder.encode_queries(q_req_list, batch_size=self.batch_size)
        if q_req_emb.dim() == 2:
            q_req_emb = F.normalize(q_req_emb, p=2, dim=1)
        q_req_emb = q_req_emb.to(self.device)

        logger.info("Encoding Q_neg (Q-)...")
        q_neg_emb = self.encoder.encode_queries(q_neg_list, batch_size=self.batch_size)
        if q_neg_emb.dim() == 2:
            q_neg_emb = F.normalize(q_neg_emb, p=2, dim=1)
        q_neg_emb = q_neg_emb.to(self.device)

        logger.info("Computing S_base...")
        S_base = torch.matmul(q_base_emb, doc_embeddings.T).float().cpu()

        logger.info(f"Stage 1: Retrieving top-{self.top_k} candidates per query...")
        top_k_indices = torch.zeros(len(query_ids), self.top_k, dtype=torch.long)
        top_k_scores = torch.zeros(len(query_ids), self.top_k)
        for i in range(len(query_ids)):
            scores = S_base[i]
            k = min(self.top_k, len(scores))
            topk = torch.topk(scores, k)
            top_k_indices[i, :k] = topk.indices
            top_k_scores[i, :k] = topk.values

        logger.info("Computing S_req and S_neg for top-k candidates...")
        S_req_topk = torch.zeros(len(query_ids), self.top_k)
        S_neg_topk = torch.zeros(len(query_ids), self.top_k)
        cos_qbase_qneg = F.cosine_similarity(q_base_emb.cpu(), q_neg_emb.cpu(), dim=1)

        for i in tqdm(range(len(query_ids)), desc="Computing S_req/S_neg"):
            indices = top_k_indices[i]
            valid_mask = indices >= 0
            if valid_mask.sum() == 0:
                continue
            valid_indices = indices[valid_mask].to(self.device)

            doc_emb_selected = doc_embeddings[valid_indices]
            s_req = torch.matmul(q_req_emb[i].unsqueeze(0), doc_emb_selected.T).squeeze(0)
            S_req_topk[i, valid_mask] = s_req.float().cpu()

            if has_neg_mask[i] > 0:
                s_neg = torch.matmul(q_neg_emb[i].unsqueeze(0), doc_emb_selected.T).squeeze(0)
                S_neg_topk[i, valid_mask] = s_neg.float().cpu()

        baseline_results = self._extract_topk_results(S_base, top_k_indices, query_ids, doc_ids)
        baseline_metrics = self.compute_metrics(baseline_results, qrels)
        logger.info("Baseline (S_base only):")
        for k, v in sorted(baseline_metrics.items()):
            logger.info(f"   {k}: {v:.4f}")

        all_results = []
        best_metrics = None
        best_params = None

        total_trials = len(alphas) * len(betas) * len(deltas)
        trial_idx = 0

        for alpha in alphas:
            for beta in betas:
                for delta in deltas:
                    trial_idx += 1

                    S_final_topk = torch.zeros(len(query_ids), self.top_k)
                    for i in range(len(query_ids)):
                        valid_mask = top_k_indices[i] >= 0
                        if valid_mask.sum() == 0:
                            continue
                        k = valid_mask.sum().item()
                        s_b = S_base[i, top_k_indices[i][:k]]
                        s_r = S_req_topk[i, :k]
                        s_n = S_neg_topk[i, :k]

                        s_final = self._score_deir_dual_v2(
                            s_base=s_b,
                            s_req=s_r,
                            s_neg=s_n,
                            cos_qbase_qneg=float(cos_qbase_qneg[i].item()),
                            has_req=bool(has_req_mask[i] > 0),
                            has_neg=bool(has_neg_mask[i] > 0),
                            alpha=alpha,
                            beta=beta,
                            delta=delta,
                        )
                        S_final_topk[i, :k] = s_final

                    reranked_results = self._extract_topk_results(
                        S_final_topk, top_k_indices, query_ids, doc_ids
                    )
                    metrics = self.compute_metrics(reranked_results, qrels)

                    ndcg10 = metrics.get("ndcg_cut_10", 0.0)
                    logger.info(
                        "[%d/%d] alpha=%.1f, beta=%.1f, delta=%.2f: "
                        "nDCG@10=%.4f, MAP@100=%.4f, Recall@100=%.4f",
                        trial_idx, total_trials,
                        alpha, beta, delta,
                        ndcg10,
                        metrics.get("map_cut_100", 0.0),
                        metrics.get("recall_100", 0.0),
                    )

                    result_entry = {
                        "alpha": alpha, "beta": beta, "delta": delta,
                        "t_safety": self.t_safety,
                        **metrics,
                    }
                    all_results.append(result_entry)

                    if best_metrics is None or ndcg10 > best_metrics.get("ndcg_cut_10", 0.0):
                        best_metrics = metrics
                        best_params = {"alpha": alpha, "beta": beta, "delta": delta}

        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info("DeIR-Dual V2 InstructIR Evaluation Complete")
        logger.info(f"Best params: alpha={best_params['alpha']}, beta={best_params['beta']}, delta={best_params['delta']}")
        logger.info("Best metrics:")
        for k, v in sorted(best_metrics.items()):
            logger.info(f"   {k}: {v:.4f}")
        logger.info("Baseline vs Best delta:")
        for k in sorted(baseline_metrics.keys()):
            delta_v = best_metrics.get(k, 0.0) - baseline_metrics.get(k, 0.0)
            logger.info(f"   {k}: {baseline_metrics.get(k, 0.0):.4f} -> {best_metrics.get(k, 0.0):.4f} (delta={delta_v:+.4f})")
        logger.info(f"Elapsed: {elapsed:.1f}s")
        logger.info("=" * 60)

        self._save_results(
            baseline_metrics=baseline_metrics,
            best_params=best_params,
            best_metrics=best_metrics,
            all_results=all_results,
            elapsed=elapsed,
        )

        return {
            "baseline_metrics": baseline_metrics,
            "best_params": best_params,
            "best_metrics": best_metrics,
            "all_results": all_results,
            "elapsed": elapsed,
        }

    def _save_results(
        self,
        baseline_metrics: Dict[str, float],
        best_params: Dict[str, Any],
        best_metrics: Dict[str, float],
        all_results: List[Dict[str, Any]],
        elapsed: float,
    ) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        summary = {
            "task": "InstructIR",
            "model": self.model_name,
            "mode": "DeIR-Dual-V2",
            "dual_queries_source": self.dual_queries_path,
            "top_k": self.top_k,
            "fixed_params": {
                "t_safety": self.t_safety,
            },
            "timestamp": datetime.now().isoformat(),
            "baseline_metrics": baseline_metrics,
            "best_params": best_params,
            "best_metrics": best_metrics,
            "improvement": {
                k: best_metrics.get(k, 0.0) - baseline_metrics.get(k, 0.0)
                for k in baseline_metrics.keys()
            },
        }

        summary_path = os.path.join(self.output_dir, "metrics_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        all_results_path = os.path.join(self.output_dir, "all_results.json")
        with open(all_results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DeIR-Dual V2 InstructIR Evaluation")
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--dual_queries_path", type=str, required=True,
                        help="Path to dual queries JSONL file")
    parser.add_argument("--output_dir", type=str, default="results/instructir")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--t_safety", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    parser.add_argument("--betas", type=float, nargs="+", default=[1.5])
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.05])

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    evaluator = InstructIREvaluator(
        model_name=args.model_name,
        dual_queries_path=args.dual_queries_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
        t_safety=args.t_safety,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )

    results = evaluator.run(
        alphas=args.alphas,
        betas=args.betas,
        deltas=args.deltas,
    )


if __name__ == "__main__":
    main()
