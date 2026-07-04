"""
DeIR-Dual V2 BEIR Evaluation Engine — BM25 Initial Retrieval + RepLLaMA Reranking

Two-stage evaluation pipeline:
  Stage 1: BM25 sparse retrieval -> top-1000 candidates
  Stage 2: DeIR-Dual V2 reranking on candidates with RepLLaMA encoder

Usage (small-scale test):
  cd /home/luwa/Documents/DSCLR && \
  /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_beir_bm25 \
    --dataset nq \
    --model_name samaya-ai/RepLLaMA-reproduced \
    --dual_queries_path dataset/BEIR/dual_queries/nq_TSC_BALANCED_t01.jsonl \
    --top_k 1000 --max_corpus 50000 --max_queries 50 \
    --alphas 1.0 --betas 1.5 --deltas 0.05 \
    --device cuda --output_dir results/beir_bm25/nq_test

Usage (full-scale):
  cd /home/luwa/Documents/DSCLR && \
  /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_beir_bm25 \
    --dataset nq \
    --model_name samaya-ai/RepLLaMA-reproduced \
    --dual_queries_path dataset/BEIR/dual_queries/nq_TSC_BALANCED_t01.jsonl \
    --top_k 1000 \
    --alphas 1.0 --betas 1.5 --deltas 0.05 \
    --device cuda --output_dir results/beir_bm25/nq

Usage (diagnostic analysis only, no reranking):
  cd /home/luwa/Documents/DSCLR && \
  /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_beir_bm25 \
    --dataset nq \
    --model_name samaya-ai/RepLLaMA-reproduced \
    --dual_queries_path dataset/BEIR/dual_queries/nq_TSC_BALANCED_t01.jsonl \
    --top_k 1000 --max_corpus 50000 --max_queries 50 \
    --diagnose_only \
    --device cuda --output_dir results/beir_bm25/nq_diag
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import logging
import argparse
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import datasets
import pytrec_eval
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

BEIR_DATASET_MAP = {
    "nq": "BeIR/nq",
    "hotpotqa": "BeIR/hotpotqa",
    "quora": "BeIR/quora",
    "fiqa": "BeIR/fiqa",
    "arguana": "BeIR/arguana",
    "scidocs": "BeIR/scidocs",
    "scifact": "BeIR/scifact",
    "nfcorpus": "BeIR/nfcorpus",
    "trec-covid": "BeIR/trec-covid",
    "msmarco": "BeIR/msmarco",
    "fever": "BeIR/fever",
    "climate-fever": "BeIR/climate-fever",
    "dbpedia-entity": "BeIR/dbpedia-entity",
    "webis-touche2020": "BeIR/webis-touche2020",
}


def resolve_dataset_name(name: str) -> str:
    if name in BEIR_DATASET_MAP:
        return BEIR_DATASET_MAP[name]
    if "/" in name:
        return name
    raise ValueError(f"Unknown BEIR dataset: {name}. Available: {list(BEIR_DATASET_MAP.keys())}")


def simple_tokenize(text: str) -> List[str]:
    return text.lower().split()


class BEIRDataLoader:
    def __init__(self, dataset_name: str, split: str = "test"):
        self.dataset_name = resolve_dataset_name(dataset_name)
        self.split = split
        self.short_name = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name

    def load_corpus(self, max_corpus: int = 0, required_doc_ids: Optional[set] = None) -> Dict[str, Dict[str, str]]:
        logger.info(f"Loading corpus from {self.dataset_name}...")
        ds = datasets.load_dataset(self.dataset_name, "corpus", split="corpus")
        corpus = {}
        missing_required = set(required_doc_ids) if required_doc_ids else set()

        for d in tqdm(ds, desc="Loading corpus"):
            doc_id = str(d["_id"])
            is_required = doc_id in missing_required

            if not is_required and max_corpus > 0 and len(corpus) >= max_corpus:
                if not missing_required:
                    break
                continue

            title = str(d.get("title", ""))
            text = str(d.get("text", ""))
            if title and title != "None":
                full_text = f"{title} {text}"
            else:
                full_text = text
            corpus[doc_id] = {"text": full_text, "title": title, "body": text}

            if is_required:
                missing_required.discard(doc_id)

        total_required = len(required_doc_ids) if required_doc_ids else 0
        found_required = total_required - len(missing_required)
        logger.info(f"Loaded {len(corpus)} documents (required: {found_required}/{total_required})")
        if missing_required:
            logger.warning(f"Missing {len(missing_required)} required documents")
        return corpus

    def load_queries(self) -> Dict[str, str]:
        logger.info(f"Loading queries from {self.dataset_name}...")
        ds = datasets.load_dataset(self.dataset_name, "queries", split="queries")
        queries = {}
        for q in ds:
            qid = str(q["_id"])
            text = str(q.get("text", ""))
            queries[qid] = text
        logger.info(f"Loaded {len(queries)} queries")
        return queries

    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        qrel_dataset = f"{self.dataset_name}-qrels"
        logger.info(f"Loading qrels from {qrel_dataset}...")
        ds = datasets.load_dataset(qrel_dataset, split=self.split)
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


class BM25Retriever:
    def __init__(self, corpus: Dict[str, Dict[str, str]], tokenize_fn=simple_tokenize):
        self.doc_ids = list(corpus.keys())
        self.doc_texts = [corpus[did]["text"] for did in self.doc_ids]
        self.tokenize_fn = tokenize_fn
        self.bm25 = None
        self.tokenized_corpus = None

    def build_index(self):
        logger.info(f"Tokenizing {len(self.doc_ids)} documents for BM25...")
        self.tokenized_corpus = [self.tokenize_fn(text) for text in tqdm(self.doc_texts, desc="Tokenizing corpus")]
        logger.info("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("BM25 index built successfully")

    def search(self, query: str, top_k: int = 1000) -> List[Tuple[str, float]]:
        tokenized_query = self.tokenize_fn(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(self.doc_ids[idx], float(scores[idx])) for idx in top_indices]
        return results

    def batch_search(self, queries: Dict[str, str], top_k: int = 1000) -> Dict[str, List[Tuple[str, float]]]:
        results = {}
        for qid, query_text in tqdm(queries.items(), desc="BM25 search"):
            results[qid] = self.search(query_text, top_k)
        return results


class BEIRBM25Evaluator:
    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        dual_queries_path: str,
        output_dir: str,
        top_k: int = 1000,
        t_safety: float = 20.0,
        device: str = "auto",
        batch_size: int = 32,
        max_seq_length: Optional[int] = None,
        max_corpus: int = 0,
        max_queries: int = 0,
        split: str = "test",
        cache_dir: Optional[str] = None,
        diagnose_only: bool = False,
    ):
        self.dataset_name = dataset_name
        self.short_name = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
        self.model_name = model_name
        self.dual_queries_path = dual_queries_path
        self.output_dir = output_dir
        self.top_k = top_k
        self.t_safety = t_safety
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.max_corpus = max_corpus
        self.max_queries = max_queries
        self.diagnose_only = diagnose_only

        if device == "auto":
            try:
                torch.cuda._lazy_init()
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device

        self.cache_dir = cache_dir or f"dataset/BEIR/embeddings/{self.short_name}"
        self.data_loader = BEIRDataLoader(dataset_name, split=split)

    def _get_model_short_name(self, model_name: str) -> str:
        if "repllama" in model_name.lower() or "promptriever" in model_name.lower():
            return "repllama"
        elif "e5-mistral" in model_name.lower():
            return "e5_mistral_7b"
        elif "bge" in model_name.lower():
            return model_name.split("/")[-1].replace("-", "_")
        else:
            return model_name.split("/")[-1].replace("-", "_")

    def _get_doc_cache_path(self, model_name: str) -> str:
        model_short = self._get_model_short_name(model_name)
        return os.path.join(self.cache_dir, f"{self.short_name}_{model_short}_candidates.pt")

    def _create_encoder(self, model_name: str):
        from eval.models import ModelFactory
        logger.info(f"Initializing encoder: {model_name}")
        encoder_kwargs = {
            "model_name": model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "normalize_embeddings": True,
        }
        if self.max_seq_length:
            encoder_kwargs["max_seq_length"] = self.max_seq_length
        encoder = ModelFactory.create(**encoder_kwargs)
        logger.info(f"Encoder initialized: {model_name}")
        return encoder

    def _encode_and_cache_candidates(
        self,
        doc_ids: List[str],
        doc_texts: List[str],
        encoder,
        model_name: str,
    ) -> torch.Tensor:
        cache_path = self._get_doc_cache_path(model_name)

        if os.path.exists(cache_path):
            logger.info(f"Loading cached candidate embeddings from {cache_path}")
            data = torch.load(cache_path, map_location="cpu", weights_only=False)
            cached_ids = data["doc_ids"]
            cached_emb = data["embeddings"]
            if set(cached_ids) == set(doc_ids):
                id_to_idx = {did: idx for idx, did in enumerate(cached_ids)}
                ordered_emb = torch.zeros(len(doc_ids), cached_emb.shape[1])
                for i, did in enumerate(doc_ids):
                    ordered_emb[i] = cached_emb[id_to_idx[did]]
                logger.info(f"Cache hit: {len(doc_ids)} documents, shape={ordered_emb.shape}")
                return ordered_emb
            else:
                cached_set = set(cached_ids)
                needed_set = set(doc_ids)
                if needed_set.issubset(cached_set):
                    id_to_idx = {did: idx for idx, did in enumerate(cached_ids)}
                    ordered_emb = torch.zeros(len(doc_ids), cached_emb.shape[1])
                    for i, did in enumerate(doc_ids):
                        ordered_emb[i] = cached_emb[id_to_idx[did]]
                    logger.info(f"Cache superset hit: {len(doc_ids)} / {len(cached_ids)} documents")
                    return ordered_emb
                logger.warning(f"Cache mismatch (cache={len(cached_ids)}, need={len(doc_ids)}), re-encoding")

        logger.info(f"Encoding {len(doc_ids)} candidate documents...")
        doc_embeddings = encoder.encode_documents(doc_texts, batch_size=self.batch_size)
        if doc_embeddings.dim() == 2:
            doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"doc_ids": doc_ids, "embeddings": doc_embeddings}, cache_path)
        logger.info(f"Candidate embeddings cached to {cache_path} (shape={doc_embeddings.shape})")

        return doc_embeddings

    def load_dual_queries(self) -> Dict[str, Dict[str, Any]]:
        if not self.dual_queries_path or not os.path.exists(self.dual_queries_path):
            logger.warning("No dual queries file provided, using Q_plus=Q_base, Q_minus=[NONE]")
            return {}
        dual_data = {}
        with open(self.dual_queries_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                qid = str(item["qid"])
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

    def _run_diagnosis(
        self,
        query_ids: List[str],
        eval_queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        dual_data: Dict[str, Dict[str, Any]],
        bm25_results: Dict[str, List[Tuple[str, float]]],
        bm25_topk_indices: torch.Tensor,
        candidate_doc_id_list: List[str],
        doc_embeddings: torch.Tensor,
        q_base_emb: torch.Tensor,
        q_req_emb: torch.Tensor,
        q_neg_emb: torch.Tensor,
        S_base_topk: torch.Tensor,
        S_req_topk: torch.Tensor,
        S_neg_topk: torch.Tensor,
        has_req_mask: List[float],
        has_neg_mask: List[float],
        cos_qbase_qneg: torch.Tensor,
        alpha: float,
        beta: float,
        delta: float,
    ):
        logger.info("=" * 60)
        logger.info("DIAGNOSTIC ANALYSIS")
        logger.info("=" * 60)

        n_queries = len(query_ids)
        n_has_neg = sum(1 for m in has_neg_mask if m > 0)
        n_has_req = sum(1 for m in has_req_mask if m > 0)
        logger.info(f"Total queries: {n_queries}")
        logger.info(f"Has Q_minus (negation): {n_has_neg} ({100*n_has_neg/n_queries:.1f}%)")
        logger.info(f"Has Q_plus (enhanced): {n_has_req} ({100*n_has_req/n_queries:.1f}%)")

        neg_indices = [i for i in range(n_queries) if has_neg_mask[i] > 0]
        nonneg_indices = [i for i in range(n_queries) if has_neg_mask[i] == 0]

        if neg_indices:
            cos_neg = cos_qbase_qneg[neg_indices]
            logger.info(f"\n--- Cos(Q_base, Q_neg) distribution (negation queries) ---")
            logger.info(f"   Mean: {cos_neg.mean():.4f}")
            logger.info(f"   Median: {cos_neg.median():.4f}")
            logger.info(f"   Std: {cos_neg.std():.4f}")
            logger.info(f"   Min: {cos_neg.min():.4f}, Max: {cos_neg.max():.4f}")
            logger.info(f"   >0.5: {(cos_neg > 0.5).sum().item()}/{len(neg_indices)} ({100*(cos_neg > 0.5).float().mean():.1f}%)")
            logger.info(f"   >0.6: {(cos_neg > 0.6).sum().item()}/{len(neg_indices)} ({100*(cos_neg > 0.6).float().mean():.1f}%)")
            logger.info(f"   >0.7: {(cos_neg > 0.7).sum().item()}/{len(neg_indices)} ({100*(cos_neg > 0.7).float().mean():.1f}%)")

        rel_s_neg_all = []
        irr_s_neg_all = []
        rel_s_base_all = []
        irr_s_base_all = []
        at_risk_count = 0
        total_neg_candidates = 0
        safety_active_count = 0

        for i in neg_indices:
            qid = query_ids[i]
            rel_docs = qrels.get(qid, {})
            valid_mask = bm25_topk_indices[i] >= 0
            k = valid_mask.sum().item()

            for j in range(k):
                idx = bm25_topk_indices[i, j].item()
                did = candidate_doc_id_list[idx]
                s_neg_val = S_neg_topk[i, j].item()
                s_base_val = S_base_topk[i, j].item()
                tau = cos_qbase_qneg[i].item() + delta

                total_neg_candidates += 1
                if s_neg_val > tau:
                    at_risk_count += 1
                if s_neg_val > tau - 0.05:
                    safety_active_count += 1

                if did in rel_docs:
                    rel_s_neg_all.append(s_neg_val)
                    rel_s_base_all.append(s_base_val)
                else:
                    irr_s_neg_all.append(s_neg_val)
                    irr_s_base_all.append(s_base_val)

        if neg_indices:
            at_risk_ratio = at_risk_count / max(total_neg_candidates, 1)
            safety_active_ratio = safety_active_count / max(total_neg_candidates, 1)
            logger.info(f"\n--- At-risk analysis (negation queries, alpha={alpha}, delta={delta}) ---")
            logger.info(f"   tau = Cos(Q_base, Q_neg) + {delta}")
            logger.info(f"   At-risk (S_neg > tau): {at_risk_count}/{total_neg_candidates} ({100*at_risk_ratio:.1f}%)")
            logger.info(f"   Safety-active (S_neg > tau-0.05): {safety_active_count}/{total_neg_candidates} ({100*safety_active_ratio:.1f}%)")

        if rel_s_neg_all and irr_s_neg_all:
            logger.info(f"\n--- S_neg distribution on relevant vs irrelevant docs ---")
            logger.info(f"   Relevant docs:   S_neg mean={np.mean(rel_s_neg_all):.4f}, median={np.median(rel_s_neg_all):.4f}")
            logger.info(f"   Irrelevant docs: S_neg mean={np.mean(irr_s_neg_all):.4f}, median={np.median(irr_s_neg_all):.4f}")
            diff = np.mean(rel_s_neg_all) - np.mean(irr_s_neg_all)
            logger.info(f"   Delta (rel - irr): {diff:+.4f}")
            if diff > 0:
                logger.info(f"   *** WARNING: S_neg is HIGHER on relevant docs! Negation penalty hurts relevant docs. ***")

        if rel_s_base_all and irr_s_base_all:
            logger.info(f"\n--- S_base distribution on relevant vs irrelevant docs ---")
            logger.info(f"   Relevant docs:   S_base mean={np.mean(rel_s_base_all):.4f}")
            logger.info(f"   Irrelevant docs: S_base mean={np.mean(irr_s_base_all):.4f}")

        logger.info(f"\n--- Score decomposition with alpha={alpha}, beta={beta}, delta={delta} ---")
        neg_qid_set = set(query_ids[i] for i in neg_indices)
        nonneg_qid_set = set(query_ids[i] for i in nonneg_indices)

        for label, qid_set in [("ALL", set(query_ids)), ("NEGATION", neg_qid_set), ("NON-NEGATION", nonneg_qid_set)]:
            if not qid_set:
                continue
            indices = [i for i, qid in enumerate(query_ids) if qid in qid_set]
            delta_scores = []
            req_contributions = []
            neg_contributions = []

            for i in indices:
                valid_mask = bm25_topk_indices[i] >= 0
                k = valid_mask.sum().item()
                if k == 0:
                    continue

                s_b = S_base_topk[i, :k]
                s_r = S_req_topk[i, :k]
                s_n = S_neg_topk[i, :k]

                s_final = self._score_deir_dual_v2(
                    s_base=s_b, s_req=s_r, s_neg=s_n,
                    cos_qbase_qneg=float(cos_qbase_qneg[i].item()),
                    has_req=bool(has_req_mask[i] > 0),
                    has_neg=bool(has_neg_mask[i] > 0),
                    alpha=alpha, beta=beta, delta=delta,
                )
                delta_s = s_final - s_b
                delta_scores.append(delta_s.mean().item())

                if has_req_mask[i] > 0:
                    tau = cos_qbase_qneg[i].item() + delta if has_neg_mask[i] > 0 else 1.0
                    if has_neg_mask[i] > 0:
                        safety = 1.0 - torch.sigmoid((s_n - tau) * self.t_safety)
                        req_contrib = (beta * s_r * safety).mean().item()
                    else:
                        req_contrib = (beta * s_r).mean().item()
                    req_contributions.append(req_contrib)

                if has_neg_mask[i] > 0:
                    tau = cos_qbase_qneg[i].item() + delta
                    penalty = (alpha * F.softplus(s_n - tau)).mean().item()
                    neg_contributions.append(penalty)

            logger.info(f"   [{label}] Avg score change: {np.mean(delta_scores):+.4f}")
            if req_contributions:
                logger.info(f"   [{label}] Avg req contribution (+beta*S_req*safety): {np.mean(req_contributions):+.4f}")
            if neg_contributions:
                logger.info(f"   [{label}] Avg neg penalty (-alpha*Softplus(S_neg-tau)): {-np.mean(neg_contributions):+.4f}")

        logger.info(f"\n--- Per-query rank change analysis (top-10) ---")
        rank_changes = []
        for i in range(n_queries):
            valid_mask = bm25_topk_indices[i] >= 0
            k = min(10, valid_mask.sum().item())
            if k == 0:
                continue

            s_b = S_base_topk[i, :k]
            s_r = S_req_topk[i, :k]
            s_n = S_neg_topk[i, :k]

            s_final = self._score_deir_dual_v2(
                s_base=s_b, s_req=s_r, s_neg=s_n,
                cos_qbase_qneg=float(cos_qbase_qneg[i].item()),
                has_req=bool(has_req_mask[i] > 0),
                has_neg=bool(has_neg_mask[i] > 0),
                alpha=alpha, beta=beta, delta=delta,
            )

            base_top10 = set(s_b.argsort(descending=True)[:10].tolist())
            final_top10 = set(s_final.argsort(descending=True)[:10].tolist())
            overlap = len(base_top10 & final_top10)
            rank_changes.append(overlap)

        logger.info(f"   Avg top-10 overlap (base vs final): {np.mean(rank_changes):.1f}/10")
        logger.info(f"   Queries with <5 overlap: {sum(1 for o in rank_changes if o < 5)}/{len(rank_changes)}")

        diag_results = {
            "total_queries": n_queries,
            "has_negation": n_has_neg,
            "has_enhanced": n_has_req,
            "negation_rate": n_has_neg / n_queries,
            "cos_qbase_qneg_mean": float(cos_qbase_qneg[neg_indices].mean()) if neg_indices else None,
            "cos_qbase_qneg_median": float(cos_qbase_qneg[neg_indices].median()) if neg_indices else None,
            "cos_qbase_qneg_gt06": float((cos_qbase_qneg[neg_indices] > 0.6).float().mean()) if neg_indices else None,
            "at_risk_ratio": at_risk_count / max(total_neg_candidates, 1) if neg_indices else None,
            "s_neg_rel_mean": float(np.mean(rel_s_neg_all)) if rel_s_neg_all else None,
            "s_neg_irr_mean": float(np.mean(irr_s_neg_all)) if irr_s_neg_all else None,
            "s_neg_rel_irr_delta": float(np.mean(rel_s_neg_all) - np.mean(irr_s_neg_all)) if (rel_s_neg_all and irr_s_neg_all) else None,
            "alpha": alpha, "beta": beta, "delta": delta,
        }

        diag_path = os.path.join(self.output_dir, "diagnosis.json")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diag_results, f, indent=2, ensure_ascii=False)
        logger.info(f"\nDiagnosis saved to {diag_path}")

    def run(
        self,
        alphas: List[float],
        betas: List[float],
        deltas: List[float],
    ) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("DeIR-Dual V2 BEIR Evaluation (BM25 + RepLLaMA)")
        logger.info("=" * 60)

        start_time = time.time()

        queries = self.data_loader.load_queries()
        qrels = self.data_loader.load_qrels()

        eval_queries = {qid: text for qid, text in queries.items() if qid in qrels}
        logger.info(f"Evaluating on {len(eval_queries)} queries (with qrels)")

        dual_data = self.load_dual_queries()

        if self.max_queries > 0:
            if dual_data:
                dual_qids = [qid for qid in eval_queries if qid in dual_data]
                if len(dual_qids) > self.max_queries:
                    dual_qids = dual_qids[:self.max_queries]
                eval_queries = {qid: eval_queries[qid] for qid in dual_qids}
                logger.info(f"Limited to {len(eval_queries)} queries (with dual data, max={self.max_queries})")
            else:
                eval_queries = dict(list(eval_queries.items())[:self.max_queries])
                logger.info(f"Limited to {len(eval_queries)} queries (max={self.max_queries})")

        qrel_doc_ids = set()
        for qid in eval_queries:
            if qid in qrels:
                qrel_doc_ids.update(qrels[qid].keys())
        logger.info(f"Required documents from qrels: {len(qrel_doc_ids)}")

        corpus = self.data_loader.load_corpus(
            max_corpus=self.max_corpus,
            required_doc_ids=qrel_doc_ids,
        )

        qrel_docs_in_corpus = qrel_doc_ids & set(corpus.keys())
        logger.info(f"Qrel documents found in corpus: {len(qrel_docs_in_corpus)}/{len(qrel_doc_ids)}")

        query_ids = list(eval_queries.keys())

        bm25_retriever = BM25Retriever(corpus)
        bm25_retriever.build_index()

        logger.info(f"Stage 1: BM25 top-{self.top_k} candidate retrieval...")
        bm25_results = bm25_retriever.batch_search(eval_queries, top_k=self.top_k)

        bm25_run = {}
        for qid, doc_score_list in bm25_results.items():
            bm25_run[qid] = {did: score for did, score in doc_score_list}

        bm25_metrics = self.compute_metrics(bm25_run, qrels)
        logger.info("BM25 Baseline:")
        for k, v in sorted(bm25_metrics.items()):
            logger.info(f"   {k}: {v:.4f}")

        candidate_doc_ids = set()
        for qid, doc_score_list in bm25_results.items():
            for did, _ in doc_score_list:
                candidate_doc_ids.add(did)
        logger.info(f"Unique candidate documents across all queries: {len(candidate_doc_ids)}")

        candidate_doc_id_list = sorted(candidate_doc_ids)
        did_to_idx = {did: idx for idx, did in enumerate(candidate_doc_id_list)}
        candidate_doc_texts = [corpus[did]["text"] for did in candidate_doc_id_list]

        encoder = self._create_encoder(self.model_name)

        doc_embeddings = self._encode_and_cache_candidates(
            candidate_doc_id_list, candidate_doc_texts, encoder, self.model_name
        )
        doc_embeddings = doc_embeddings.to(self.device)
        logger.info(f"Document embeddings shape: {doc_embeddings.shape}")

        q_base_list = [eval_queries[qid] for qid in query_ids]
        q_req_list = []
        q_neg_list = []
        has_req_mask = []
        has_neg_mask = []

        for qid in query_ids:
            d = dual_data.get(qid, {})
            q_plus = d.get("q_plus", "")
            q_minus = d.get("q_minus", "")
            if not q_plus or self._is_none_query(q_plus):
                q_plus = eval_queries[qid]
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

        logger.info("Encoding Q_base / Q_req / Q_neg...")
        q_base_emb = encoder.encode_queries(q_base_list, batch_size=self.batch_size)
        if q_base_emb.dim() == 2:
            q_base_emb = F.normalize(q_base_emb, p=2, dim=1)
        q_base_emb = q_base_emb.to(self.device)

        q_req_emb = encoder.encode_queries(q_req_list, batch_size=self.batch_size)
        if q_req_emb.dim() == 2:
            q_req_emb = F.normalize(q_req_emb, p=2, dim=1)
        q_req_emb = q_req_emb.to(self.device)

        q_neg_emb = encoder.encode_queries(q_neg_list, batch_size=self.batch_size)
        if q_neg_emb.dim() == 2:
            q_neg_emb = F.normalize(q_neg_emb, p=2, dim=1)
        q_neg_emb = q_neg_emb.to(self.device)

        cos_qbase_qneg = F.cosine_similarity(q_base_emb.cpu(), q_neg_emb.cpu(), dim=1)

        logger.info("Computing S_base / S_req / S_neg for BM25 candidates...")

        top_k_per_query = self.top_k
        bm25_topk_indices = torch.full((len(query_ids), top_k_per_query), -1, dtype=torch.long)
        S_base_topk = torch.zeros(len(query_ids), top_k_per_query)
        S_req_topk = torch.zeros(len(query_ids), top_k_per_query)
        S_neg_topk = torch.zeros(len(query_ids), top_k_per_query)

        for i in tqdm(range(len(query_ids)), desc="Computing scores"):
            qid = query_ids[i]
            doc_score_list = bm25_results[qid]
            k = min(top_k_per_query, len(doc_score_list))

            candidate_indices = []
            for j, (did, bm25_score) in enumerate(doc_score_list[:k]):
                if did in did_to_idx:
                    candidate_indices.append(did_to_idx[did])
                else:
                    candidate_indices.append(-1)

            for j in range(k):
                idx = candidate_indices[j]
                bm25_topk_indices[i, j] = idx
                if idx >= 0:
                    doc_emb = doc_embeddings[idx].unsqueeze(0)
                    S_base_topk[i, j] = torch.matmul(q_base_emb[i].unsqueeze(0), doc_emb.T).squeeze().float().cpu().item()
                    S_req_topk[i, j] = torch.matmul(q_req_emb[i].unsqueeze(0), doc_emb.T).squeeze().float().cpu().item()
                    if has_neg_mask[i] > 0:
                        S_neg_topk[i, j] = torch.matmul(q_neg_emb[i].unsqueeze(0), doc_emb.T).squeeze().float().cpu().item()

            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

        rerank_baseline_results = self._extract_topk_results_from_scores(
            S_base_topk, bm25_topk_indices, query_ids, candidate_doc_id_list
        )
        rerank_baseline_metrics = self.compute_metrics(rerank_baseline_results, qrels)
        logger.info("RepLLaMA Baseline (dense scores on BM25 candidates):")
        for k, v in sorted(rerank_baseline_metrics.items()):
            logger.info(f"   {k}: {v:.4f}")

        default_alpha = alphas[0]
        default_beta = betas[0]
        default_delta = deltas[0]

        if self.diagnose_only:
            self._run_diagnosis(
                query_ids=query_ids,
                eval_queries=eval_queries,
                qrels=qrels,
                dual_data=dual_data,
                bm25_results=bm25_results,
                bm25_topk_indices=bm25_topk_indices,
                candidate_doc_id_list=candidate_doc_id_list,
                doc_embeddings=doc_embeddings,
                q_base_emb=q_base_emb,
                q_req_emb=q_req_emb,
                q_neg_emb=q_neg_emb,
                S_base_topk=S_base_topk,
                S_req_topk=S_req_topk,
                S_neg_topk=S_neg_topk,
                has_req_mask=has_req_mask,
                has_neg_mask=has_neg_mask,
                cos_qbase_qneg=cos_qbase_qneg,
                alpha=default_alpha,
                beta=default_beta,
                delta=default_delta,
            )
            elapsed = time.time() - start_time
            return {
                "diagnose_only": True,
                "bm25_metrics": bm25_metrics,
                "rerank_baseline_metrics": rerank_baseline_metrics,
                "elapsed": elapsed,
            }

        all_results = []
        best_metrics = None
        best_params = None

        total_trials = len(alphas) * len(betas) * len(deltas)
        trial_idx = 0

        for alpha in alphas:
            for beta in betas:
                for delta in deltas:
                    trial_idx += 1

                    S_final_topk = torch.zeros(len(query_ids), top_k_per_query)
                    for i in range(len(query_ids)):
                        valid_mask = bm25_topk_indices[i] >= 0
                        if valid_mask.sum() == 0:
                            continue
                        k = valid_mask.sum().item()
                        s_b = S_base_topk[i, :k]
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

                    reranked_results = self._extract_topk_results_from_scores(
                        S_final_topk, bm25_topk_indices, query_ids, candidate_doc_id_list
                    )
                    metrics = self.compute_metrics(reranked_results, qrels)

                    ndcg10 = metrics.get("ndcg_cut_10", 0.0)
                    logger.info(
                        "[%d/%d] alpha=%.1f, beta=%.1f, delta=%.2f: nDCG@10=%.4f, MAP@100=%.4f, Recall@100=%.4f",
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

        self._run_diagnosis(
            query_ids=query_ids,
            eval_queries=eval_queries,
            qrels=qrels,
            dual_data=dual_data,
            bm25_results=bm25_results,
            bm25_topk_indices=bm25_topk_indices,
            candidate_doc_id_list=candidate_doc_id_list,
            doc_embeddings=doc_embeddings,
            q_base_emb=q_base_emb,
            q_req_emb=q_req_emb,
            q_neg_emb=q_neg_emb,
            S_base_topk=S_base_topk,
            S_req_topk=S_req_topk,
            S_neg_topk=S_neg_topk,
            has_req_mask=has_req_mask,
            has_neg_mask=has_neg_mask,
            cos_qbase_qneg=cos_qbase_qneg,
            alpha=best_params["alpha"],
            beta=best_params["beta"],
            delta=best_params["delta"],
        )

        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info("DeIR-Dual V2 BEIR Evaluation Complete (BM25 + RepLLaMA)")
        logger.info("   Rerank model: %s", self.model_name)
        logger.info("   Best params: alpha=%.1f, beta=%.1f, delta=%.2f",
                     best_params["alpha"], best_params["beta"], best_params["delta"])
        logger.info("   nDCG@10: %.4f", best_metrics.get("ndcg_cut_10", 0.0))
        logger.info("   MAP@100: %.4f", best_metrics.get("map_cut_100", 0.0))
        logger.info("   Recall@100: %.4f", best_metrics.get("recall_100", 0.0))
        logger.info("   MRR@10: %.4f", best_metrics.get("recip_rank_10", 0.0))
        logger.info("   BM25 baseline nDCG@10: %.4f", bm25_metrics.get("ndcg_cut_10", 0.0))
        logger.info("   RepLLaMA baseline nDCG@10: %.4f", rerank_baseline_metrics.get("ndcg_cut_10", 0.0))
        logger.info("   Elapsed: %.1f seconds", elapsed)
        logger.info("=" * 60)

        self._save_results(
            best_params,
            best_metrics,
            bm25_metrics,
            rerank_baseline_metrics,
            all_results,
            elapsed,
        )

        return {
            "best_params": best_params,
            "best_metrics": best_metrics,
            "bm25_metrics": bm25_metrics,
            "rerank_baseline_metrics": rerank_baseline_metrics,
            "rerank_model": self.model_name,
            "all_results": all_results,
            "elapsed": elapsed,
        }

    def _extract_topk_results_from_scores(
        self,
        S_final_topk: torch.Tensor,
        top_k_indices: torch.Tensor,
        query_ids: List[str],
        doc_ids: List[str],
        subset_indices: Optional[List[int]] = None,
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        indices = subset_indices if subset_indices is not None else range(len(query_ids))
        for i in indices:
            qid = query_ids[i]
            doc_scores = {}
            scored_pairs = []
            for j in range(self.top_k):
                idx = top_k_indices[i, j].item()
                if idx < 0:
                    continue
                did = doc_ids[idx]
                score = float(S_final_topk[i, j].item())
                scored_pairs.append((did, score))
            scored_pairs.sort(key=lambda x: x[1], reverse=True)
            results[qid] = dict(scored_pairs)
        return results

    def _save_results(
        self,
        best_params: Dict[str, Any],
        best_metrics: Dict[str, float],
        bm25_metrics: Dict[str, float],
        rerank_baseline_metrics: Dict[str, float],
        all_results: List[Dict[str, Any]],
        elapsed: float,
    ):
        os.makedirs(self.output_dir, exist_ok=True)

        summary = {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "mode": "BM25-then-DeIR-Dual-V2-BEIR",
            "dual_queries_source": self.dual_queries_path,
            "top_k": self.top_k,
            "t_safety": self.t_safety,
            "max_corpus": self.max_corpus,
            "max_queries": self.max_queries,
            "timestamp": datetime.now().isoformat(),
            "best_params": best_params,
            "best_metrics": best_metrics,
            "bm25_metrics": bm25_metrics,
            "rerank_baseline_metrics": rerank_baseline_metrics,
            "elapsed_seconds": elapsed,
        }

        summary_path = os.path.join(self.output_dir, "metrics_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        all_results_path = os.path.join(self.output_dir, "all_results.json")
        with open(all_results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DeIR-Dual V2 BEIR Evaluation (BM25 + RepLLaMA)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="BEIR dataset name (e.g., nq, hotpotqa, quora, BeIR/nq)")
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--dual_queries_path", type=str, default="",
                        help="Path to dual queries JSONL file")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=1000,
                        help="Number of candidates to retrieve in Stage 1 (BM25)")
    parser.add_argument("--alphas", type=str, default="1.0",
                        help="Comma-separated alpha values (default: training-set-derived optimal)")
    parser.add_argument("--betas", type=str, default="1.5",
                        help="Comma-separated beta values (default: training-set-derived optimal)")
    parser.add_argument("--deltas", type=str, default="0.05",
                        help="Comma-separated delta values (default: training-set-derived optimal)")
    parser.add_argument("--t_safety", type=float, default=20.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_corpus", type=int, default=0,
                        help="Max corpus documents for BM25 (0 = all). Use small value for testing.")
    parser.add_argument("--max_queries", type=int, default=0,
                        help="Max queries to evaluate (0 = all, prefer dual queries)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--cache_dir", type=str, default="",
                        help="Directory for caching document embeddings")
    parser.add_argument("--diagnose_only", action="store_true",
                        help="Only run diagnostic analysis, skip parameter search")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    alphas = [float(a.strip()) for a in args.alphas.split(",")]
    betas = [float(b.strip()) for b in args.betas.split(",")]
    deltas = [float(d.strip()) for d in args.deltas.split(",")]

    evaluator = BEIRBM25Evaluator(
        dataset_name=args.dataset,
        model_name=args.model_name,
        dual_queries_path=args.dual_queries_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
        t_safety=args.t_safety,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        max_corpus=args.max_corpus,
        max_queries=args.max_queries,
        split=args.split,
        cache_dir=args.cache_dir if args.cache_dir else None,
        diagnose_only=args.diagnose_only,
    )

    result = evaluator.run(alphas=alphas, betas=betas, deltas=deltas)

    if result.get("diagnose_only"):
        print("\n" + "=" * 60)
        print("Diagnostic Analysis Complete")
        print(f"   BM25 baseline nDCG@10: {result['bm25_metrics'].get('ndcg_cut_10', 0.0):.4f}")
        print(f"   RepLLaMA baseline nDCG@10: {result.get('rerank_baseline_metrics', {}).get('ndcg_cut_10', 0.0):.4f}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Final Results (BM25 + DeIR-Dual V2)")
        print(f"   Dataset: {args.dataset}")
        print(f"   Rerank model: {result.get('rerank_model', args.model_name)}")
        print(f"   Best params: alpha={result['best_params']['alpha']}, "
              f"beta={result['best_params']['beta']}, delta={result['best_params']['delta']}")
        print(f"   nDCG@10: {result['best_metrics'].get('ndcg_cut_10', 0.0):.4f}")
        print(f"   MAP@100: {result['best_metrics'].get('map_cut_100', 0.0):.4f}")
        print(f"   Recall@100: {result['best_metrics'].get('recall_100', 0.0):.4f}")
        print(f"   MRR@10: {result['best_metrics'].get('recip_rank_10', 0.0):.4f}")
        print(f"   BM25 baseline nDCG@10: {result['bm25_metrics'].get('ndcg_cut_10', 0.0):.4f}")
        print(f"   RepLLaMA baseline nDCG@10: {result.get('rerank_baseline_metrics', {}).get('ndcg_cut_10', 0.0):.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
