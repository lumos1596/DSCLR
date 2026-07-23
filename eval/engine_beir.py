"""
DeIR-Dual V2 BEIR Evaluation Engine

Two-stage evaluation pipeline:
  Stage 1: Dense retrieval with Q_base -> top-k candidates
  Stage 2: DeIR-Dual V2 reranking on top-k candidates

Supports BEIR datasets via HuggingFace (BeIR/nq, BeIR/hotpotqa, BeIR/quora, etc.)

Usage:
  cd /home/luwa/Documents/DSCLR && /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_beir \
    --dataset BeIR/nq \
    --model_name samaya-ai/RepLLaMA-reproduced \
    --dual_queries_path dataset/BEIR/dual_queries/nq_TSC_BALANCED_t01.jsonl \
    --alphas 0.5 --betas 1.3 --deltas 0.10 \
    --top_k 100 --device cuda \
    --output_dir results/beir/nq
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force online mode BEFORE importing datasets/huggingface_hub
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ.pop("HF_ENDPOINT", None)

# Also patch huggingface_hub constants before any import
try:
    import huggingface_hub.constants as _hf_const
    _hf_const.HF_HUB_OFFLINE = False
except Exception:
    pass

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
# Patch datasets offline mode after import
try:
    datasets.config.HF_DATASETS_OFFLINE = False
except Exception:
    pass
import pytrec_eval

from eval.residual_boundary import compute_background_residual_boundary

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


class BEIRDataLoader:
    def __init__(self, dataset_name: str, split: str = "test"):
        self.dataset_name = resolve_dataset_name(dataset_name)
        self.split = split
        self.short_name = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name

    def load_corpus(self, max_corpus: int = 0, required_doc_ids: Optional[set] = None) -> Dict[str, Dict[str, str]]:
        logger.info(f"📂 Loading corpus from {self.dataset_name}...")
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
        logger.info(f"✅ Loaded {len(corpus)} documents (required: {found_required}/{total_required})")
        if missing_required:
            logger.warning(f"⚠️ Missing {len(missing_required)} required documents")
        return corpus

    def load_queries(self) -> Dict[str, str]:
        logger.info(f"📂 Loading queries from {self.dataset_name}...")
        ds = datasets.load_dataset(self.dataset_name, "queries", split="queries")
        queries = {}
        for q in ds:
            qid = str(q["_id"])
            text = str(q.get("text", ""))
            queries[qid] = text
        logger.info(f"✅ Loaded {len(queries)} queries")
        return queries

    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        qrel_dataset = f"{self.dataset_name}-qrels"
        logger.info(f"📂 Loading qrels from {qrel_dataset}...")
        ds = datasets.load_dataset(qrel_dataset, split=self.split)
        qrels = {}
        for item in ds:
            qid = str(item["query-id"])
            doc_id = str(item["corpus-id"])
            score = int(item.get("score", 1))
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = score
        logger.info(f"✅ Loaded qrels for {len(qrels)} queries")
        return qrels


class BEIREvaluator:
    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        dual_queries_path: str,
        output_dir: str,
        candidate_model_name: Optional[str] = None,
        top_k: int = 100,
        t_safety: float = 10.0,
        boundary_mode: str = "semantic",
        residual_margin_scale: float = 1.0,
        safety_kappa: float = 0.0,
        per_query_ab: bool = False,
        beta_derive_mode: str = "max_mean",
        device: str = "auto",
        batch_size: int = 64,
        max_seq_length: Optional[int] = None,
        cache_checkpoint_interval: int = 200,
        cache_dir: Optional[str] = None,
        split: str = "test",
        max_queries: int = 0,
        max_corpus: int = 0,
    ):
        self.dataset_name = dataset_name
        self.short_name = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
        self.model_name = model_name
        self.candidate_model_name = candidate_model_name or model_name
        self.dual_queries_path = dual_queries_path
        self.output_dir = output_dir
        self.top_k = top_k
        self.t_safety = t_safety
        self.boundary_mode = boundary_mode
        self.residual_margin_scale = residual_margin_scale
        self.safety_kappa = safety_kappa
        self.per_query_ab = per_query_ab
        self.beta_derive_mode = beta_derive_mode
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.cache_checkpoint_interval = max(1, cache_checkpoint_interval)
        self.max_queries = max_queries
        self.max_corpus = max_corpus

        if device == "auto":
            try:
                torch.cuda._lazy_init()
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device

        self.cache_dir = cache_dir or f"dataset/BEIR/embeddings/{self.short_name}"
        self.data_loader = BEIRDataLoader(dataset_name, split=split)

    def _create_encoder(self, model_name: str):
        from eval.models import ModelFactory
        logger.info(f"📥 Initializing encoder: {model_name}")
        encoder_kwargs = {
            "model_name": model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "normalize_embeddings": True,
        }
        if self.max_seq_length:
            encoder_kwargs["max_seq_length"] = self.max_seq_length

        encoder = ModelFactory.create(**encoder_kwargs)
        logger.info(f"✅ Encoder initialized: {model_name}")
        return encoder

    def _get_model_short_name(self, model_name: str) -> str:
        if "repllama" in model_name.lower():
            return "repllama"
        elif "mistral" in model_name.lower():
            return "e5-mistral-7b"
        else:
            return model_name.split("/")[-1].replace("-", "_")

    def _get_corpus_cache_path(self, model_name: str) -> str:
        model_short = self._get_model_short_name(model_name)
        return os.path.join(self.cache_dir, f"{self.short_name}_{model_short}_corpus.pt")

    def _encode_and_cache_corpus(
        self,
        corpus: Dict[str, Dict[str, str]],
        encoder,
        model_name: str,
    ) -> Tuple[torch.Tensor, List[str]]:
        cache_path = self._get_corpus_cache_path(model_name)
        checkpoint_dir = f"{cache_path}.checkpoint"
        shard_dir = os.path.join(checkpoint_dir, "shards")
        meta_path = os.path.join(checkpoint_dir, "meta.json")
        if os.path.exists(cache_path):
            logger.info(f"📂 Loading cached corpus embeddings from {cache_path}")
            data = torch.load(cache_path, map_location="cpu")
            doc_ids = data["doc_ids"]
            embeddings = data["embeddings"]
            if len(doc_ids) == len(corpus):
                logger.info(f"✅ Cache hit: {len(doc_ids)} documents, shape={embeddings.shape}")
                return embeddings, doc_ids
            else:
                logger.warning(f"⚠️ Cache size mismatch (cache={len(doc_ids)}, corpus={len(corpus)}), re-encoding")

        doc_ids = list(corpus.keys())
        doc_texts = [corpus[did]["text"] for did in doc_ids]

        processed_count = 0
        shards: List[Dict[str, int]] = []
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                processed_count = int(meta.get("processed_count", 0))
                shards = meta.get("shards", [])
                logger.info("📌 Found checkpoint: %d docs encoded across %d shards", processed_count, len(shards))
            except Exception as exc:
                logger.warning("⚠️ Failed to read checkpoint meta (%s); re-encoding", exc)
                processed_count = 0
                shards = []

        os.makedirs(shard_dir, exist_ok=True)

        logger.info(f"📚 Encoding {len(doc_ids)} documents (resume from {processed_count})...")
        shard_buffers: List[torch.Tensor] = []
        shard_start = processed_count
        batches_in_shard = 0

        for i in tqdm(range(processed_count, len(doc_ids), self.batch_size), desc="Encoding corpus"):
            batch_texts = doc_texts[i:i + self.batch_size]
            batch_emb = encoder.encode_documents(batch_texts, batch_size=self.batch_size)
            if batch_emb.dim() == 2:
                batch_emb = F.normalize(batch_emb, p=2, dim=1)
            shard_buffers.append(batch_emb.cpu())
            batches_in_shard += 1

            if batches_in_shard >= self.cache_checkpoint_interval:
                shard_end = i + self.batch_size
                shard_tensor = torch.cat(shard_buffers, dim=0)
                shard_path = os.path.join(shard_dir, f"shard_{shard_start}_{shard_end}.pt")
                torch.save(shard_tensor, shard_path)
                shards.append({"start": shard_start, "end": shard_end, "path": shard_path})
                processed_count = shard_end
                shard_buffers = []
                batches_in_shard = 0
                shard_start = processed_count

                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump({"processed_count": processed_count, "shards": shards}, f)

        if shard_buffers:
            shard_end = len(doc_ids)
            shard_tensor = torch.cat(shard_buffers, dim=0)
            shard_path = os.path.join(shard_dir, f"shard_{shard_start}_{shard_end}.pt")
            torch.save(shard_tensor, shard_path)
            shards.append({"start": shard_start, "end": shard_end, "path": shard_path})
            processed_count = shard_end
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"processed_count": processed_count, "shards": shards}, f)

        logger.info("📦 Consolidating %d shards into final embedding tensor...", len(shards))
        shards_sorted = sorted(shards, key=lambda x: x["start"])
        embeddings_list = [torch.load(s["path"], map_location="cpu") for s in shards_sorted]
        embeddings = torch.cat(embeddings_list, dim=0)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"doc_ids": doc_ids, "embeddings": embeddings}, cache_path)
        logger.info(f"💾 Corpus embeddings cached to {cache_path} (shape={embeddings.shape})")

        try:
            for shard in shards_sorted:
                if os.path.exists(shard["path"]):
                    os.remove(shard["path"])
            if os.path.exists(meta_path):
                os.remove(meta_path)
            if os.path.isdir(shard_dir):
                os.rmdir(shard_dir)
            if os.path.isdir(checkpoint_dir):
                os.rmdir(checkpoint_dir)
        except Exception as exc:
            logger.warning("⚠️ Failed to clean checkpoint cache: %s", exc)

        return embeddings, doc_ids

    def load_dual_queries(self) -> Dict[str, Dict[str, Any]]:
        if not self.dual_queries_path or not os.path.exists(self.dual_queries_path):
            logger.warning("⚠️ No dual queries file provided, using Q_plus=Q_base, Q_minus=[NONE]")
            return {}

        dual_data = {}
        with open(self.dual_queries_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                qid = item["qid"]
                dual_data[qid] = item
        logger.info(f"✅ Loaded dual queries: {len(dual_data)} entries")
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
            if self.per_query_ab and has_req and s_base.numel() > 0:
                safety = torch.ones_like(s_base)
                if self.beta_derive_mode == "max_mean":
                    max_b = s_base.max()
                    mean_r = s_req.mean()
                    beta = float((max_b / mean_r).item()) if mean_r > 1e-8 else beta
                    beta = min(beta, 50.0)
            return s_base + beta * s_req_eff

        # V8 residual_bg boundary
        if self.boundary_mode == "residual_bg":
            boundary = compute_background_residual_boundary(
                s_base=s_base,
                s_neg=s_neg,
                cos_qbase_qneg=cos_qbase_qneg,
                margin_scale=self.residual_margin_scale,
            )
            overflow = boundary.overflow
            smooth_penalty = F.softplus(overflow)
            tau = cos_qbase_qneg + delta

            # safety_kappa gate: MAD-normalized residual safety
            if self.safety_kappa > 0 and boundary.mad > 1e-8:
                safety = 1.0 - torch.sigmoid(
                    boundary.residual / boundary.mad * self.safety_kappa
                )
            else:
                safety = 1.0 - torch.sigmoid((s_neg - tau) * self.t_safety)
        else:
            tau = cos_qbase_qneg + delta
            overflow = s_neg - tau
            smooth_penalty = F.softplus(overflow)
            safety = 1.0 - torch.sigmoid(overflow * self.t_safety)

        # per_query_ab derivation
        if self.per_query_ab:
            at_risk_mask = overflow > 0
            safe_mask = ~at_risk_mask
            if at_risk_mask.any():
                mean_base_risk = s_base[at_risk_mask].mean()
                mean_penalty_risk = smooth_penalty[at_risk_mask].mean()
                if mean_penalty_risk > 1e-8:
                    alpha = float((mean_base_risk / mean_penalty_risk).item())
            if has_req and safe_mask.any():
                if self.beta_derive_mode == "max_mean":
                    max_b = s_base[safe_mask].max()
                    mean_r = s_req[safe_mask].mean()
                    beta = float((max_b / mean_r).item()) if mean_r > 1e-8 else beta
            alpha = min(alpha, 50.0)
            beta = min(beta, 50.0)

        raw_penalty = alpha * smooth_penalty
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
            logger.warning("⚠️ No overlapping queries between results and qrels")
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

    def run(
        self,
        alphas: List[float],
        betas: List[float],
        deltas: List[float],
    ) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("🚀 DeIR-Dual V2 BEIR Evaluation")
        logger.info("=" * 60)

        start_time = time.time()

        queries = self.data_loader.load_queries()
        qrels = self.data_loader.load_qrels()

        required_doc_ids: set = set()
        for qid, doc_dict in qrels.items():
            for did in doc_dict.keys():
                required_doc_ids.add(did)
        logger.info(f"📋 Required doc ids (from qrels): {len(required_doc_ids)}")

        corpus = self.data_loader.load_corpus(
            max_corpus=self.max_corpus,
            required_doc_ids=required_doc_ids,
        )

        eval_queries = {qid: text for qid, text in queries.items() if qid in qrels}
        logger.info(f"📊 Evaluating on {len(eval_queries)} queries (with qrels)")

        dual_data = self.load_dual_queries()

        if self.max_queries > 0:
            if dual_data:
                dual_qids = [qid for qid in eval_queries if qid in dual_data]
                if len(dual_qids) > self.max_queries:
                    dual_qids = dual_qids[:self.max_queries]
                eval_queries = {qid: eval_queries[qid] for qid in dual_qids}
                logger.info(f"📊 Limited to {len(eval_queries)} queries (with dual data, max={self.max_queries})")
            else:
                eval_queries = dict(list(eval_queries.items())[:self.max_queries])
                logger.info(f"📊 Limited to {len(eval_queries)} queries (max={self.max_queries})")

        query_ids = list(eval_queries.keys())
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

        has_req_mask_t = torch.tensor(has_req_mask, dtype=torch.float32, device=self.device)
        has_neg_mask_t = torch.tensor(has_neg_mask, dtype=torch.float32, device=self.device)

        candidate_encoder = self._create_encoder(self.candidate_model_name)
        candidate_doc_embeddings, candidate_doc_ids = self._encode_and_cache_corpus(
            corpus,
            candidate_encoder,
            self.candidate_model_name,
        )

        logger.info("📊 Encoding candidate Q_base...")
        candidate_q_base_emb = candidate_encoder.encode_queries(q_base_list, batch_size=self.batch_size)
        if candidate_q_base_emb.dim() == 2:
            candidate_q_base_emb = F.normalize(candidate_q_base_emb, p=2, dim=1)
        candidate_q_base_emb = candidate_q_base_emb.to(self.device)

        logger.info("📊 Computing candidate S_base (chunked for memory efficiency)...")
        q_chunk_size = 50
        d_chunk_size = 500000
        candidate_S_base = torch.zeros(len(query_ids), len(candidate_doc_ids))
        for di in tqdm(range(0, len(candidate_doc_ids), d_chunk_size), desc="Computing candidate S_base (doc chunks)"):
            d_end = min(di + d_chunk_size, len(candidate_doc_ids))
            doc_chunk = candidate_doc_embeddings[di:d_end].to(self.device)
            for qi in range(0, len(query_ids), q_chunk_size):
                q_end = min(qi + q_chunk_size, len(query_ids))
                q_chunk = candidate_q_base_emb[qi:q_end]
                chunk_scores = torch.matmul(q_chunk, doc_chunk.T)
                candidate_S_base[qi:q_end, di:d_end] = chunk_scores.float().cpu()
            del doc_chunk
            torch.cuda.empty_cache()

        logger.info(f"📊 Stage 1: Promptriever top-{self.top_k} candidate retrieval...")
        top_k_indices = torch.zeros(len(query_ids), self.top_k, dtype=torch.long)
        top_k_scores = torch.zeros(len(query_ids), self.top_k)
        for i in range(len(query_ids)):
            scores = candidate_S_base[i]
            k = min(self.top_k, len(scores))
            topk = torch.topk(scores, k)
            top_k_indices[i, :k] = topk.indices
            top_k_scores[i, :k] = topk.values

        baseline_results = self._extract_topk_results(candidate_S_base, top_k_indices, query_ids, candidate_doc_ids)
        baseline_metrics = self.compute_metrics(baseline_results, qrels)
        logger.info("📊 Baseline (stage 1 / candidate model only):")
        for k, v in sorted(baseline_metrics.items()):
            logger.info(f"   {k}: {v:.4f}")

        if self.candidate_model_name == self.model_name:
            rerank_encoder = candidate_encoder
            rerank_doc_embeddings = candidate_doc_embeddings
            rerank_doc_ids = candidate_doc_ids
        else:
            del candidate_encoder
            torch.cuda.empty_cache()
            rerank_encoder = self._create_encoder(self.model_name)
            rerank_doc_embeddings, rerank_doc_ids = self._encode_and_cache_corpus(
                corpus,
                rerank_encoder,
                self.model_name,
            )

        if rerank_doc_ids != candidate_doc_ids:
            logger.warning("⚠️ Candidate and rerank doc orders differ; rerank will follow rerank model ordering.")

        logger.info("📊 Encoding rerank Q_base / Q_req / Q_neg...")
        q_base_emb = rerank_encoder.encode_queries(q_base_list, batch_size=self.batch_size)
        if q_base_emb.dim() == 2:
            q_base_emb = F.normalize(q_base_emb, p=2, dim=1)
        q_base_emb = q_base_emb.to(self.device)

        q_req_emb = rerank_encoder.encode_queries(q_req_list, batch_size=self.batch_size)
        if q_req_emb.dim() == 2:
            q_req_emb = F.normalize(q_req_emb, p=2, dim=1)
        q_req_emb = q_req_emb.to(self.device)

        q_neg_emb = rerank_encoder.encode_queries(q_neg_list, batch_size=self.batch_size)
        if q_neg_emb.dim() == 2:
            q_neg_emb = F.normalize(q_neg_emb, p=2, dim=1)
        q_neg_emb = q_neg_emb.to(self.device)

        logger.info("📊 Computing rerank scores for top-k candidates...")
        S_req_topk = torch.zeros(len(query_ids), self.top_k)
        S_neg_topk = torch.zeros(len(query_ids), self.top_k)
        S_base_topk = torch.zeros(len(query_ids), self.top_k)
        cos_qbase_qneg = F.cosine_similarity(q_base_emb.cpu(), q_neg_emb.cpu(), dim=1)

        for i in tqdm(range(len(query_ids)), desc="Computing S_req/S_neg"):
            indices = top_k_indices[i]
            valid_mask = indices >= 0
            if valid_mask.sum() == 0:
                continue
            valid_indices = indices[valid_mask]

            doc_emb_selected = rerank_doc_embeddings[valid_indices].to(self.device)
            s_base = torch.matmul(q_base_emb[i].unsqueeze(0), doc_emb_selected.T).squeeze(0)
            S_base_topk[i, valid_mask] = s_base.float().cpu()
            s_req = torch.matmul(q_req_emb[i].unsqueeze(0), doc_emb_selected.T).squeeze(0)
            S_req_topk[i, valid_mask] = s_req.float().cpu()

            if has_neg_mask[i] > 0:
                s_neg = torch.matmul(q_neg_emb[i].unsqueeze(0), doc_emb_selected.T).squeeze(0)
                S_neg_topk[i, valid_mask] = s_neg.float().cpu()

        rerank_baseline_results = self._extract_topk_results_from_scores(
            S_base_topk, top_k_indices, query_ids, rerank_doc_ids
        )
        rerank_baseline_metrics = self.compute_metrics(rerank_baseline_results, qrels)
        logger.info("📊 Baseline (rerank model scores on candidate set):")
        for k, v in sorted(rerank_baseline_metrics.items()):
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
                        S_final_topk, top_k_indices, query_ids, rerank_doc_ids
                    )
                    metrics = self.compute_metrics(reranked_results, qrels)

                    ndcg10 = metrics.get("ndcg_cut_10", 0.0)
                    logger.info(
                        "[%d/%d] α=%.1f, β=%.1f, δ=%.2f: nDCG@10=%.4f, MAP@100=%.4f, Recall@100=%.4f",
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

        dual_qid_set = set(dual_data.keys())
        dual_query_indices = [i for i, qid in enumerate(query_ids) if qid in dual_qid_set]

        if dual_query_indices:
            logger.info(f"📊 Subset evaluation on {len(dual_query_indices)} queries with dual data...")
            dual_baseline_results = self._extract_topk_results(
                candidate_S_base, top_k_indices, query_ids, candidate_doc_ids, dual_query_indices
            )
            dual_baseline_metrics = self.compute_metrics(dual_baseline_results, qrels)
            logger.info("📊 Dual-subset Baseline (stage 1 / candidate model only):")
            for k, v in sorted(dual_baseline_metrics.items()):
                logger.info(f"   {k}: {v:.4f}")

            best_alpha = best_params["alpha"]
            best_beta = best_params["beta"]
            best_delta = best_params["delta"]
            S_final_best = torch.zeros(len(query_ids), self.top_k)
            for i in range(len(query_ids)):
                valid_mask = top_k_indices[i] >= 0
                if valid_mask.sum() == 0:
                    continue
                k = valid_mask.sum().item()
                s_b = S_base_topk[i, :k]
                s_r = S_req_topk[i, :k]
                s_n = S_neg_topk[i, :k]
                s_final = self._score_deir_dual_v2(
                    s_base=s_b, s_req=s_r, s_neg=s_n,
                    cos_qbase_qneg=float(cos_qbase_qneg[i].item()),
                    has_req=bool(has_req_mask[i] > 0),
                    has_neg=bool(has_neg_mask[i] > 0),
                    alpha=best_alpha, beta=best_beta, delta=best_delta,
                )
                S_final_best[i, :k] = s_final

            dual_reranked_results = self._extract_topk_results_from_scores(
                S_final_best, top_k_indices, query_ids, rerank_doc_ids, dual_query_indices
            )
            dual_reranked_metrics = self.compute_metrics(dual_reranked_results, qrels)
            logger.info(f"📊 Dual-subset DeIR-Dual V2 (α={best_alpha}, β={best_beta}, δ={best_delta}):")
            for k, v in sorted(dual_reranked_metrics.items()):
                logger.info(f"   {k}: {v:.4f}")
                delta_v = v - dual_baseline_metrics.get(k, 0.0)
                sign = "+" if delta_v >= 0 else ""
                logger.info(f"   {k}: {v:.4f} ({sign}{delta_v:.4f})")

        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info("📊 DeIR-Dual V2 BEIR Evaluation Complete")
        logger.info("   Candidate model: %s", self.candidate_model_name)
        logger.info("   Rerank model: %s", self.model_name)
        logger.info("   Best params: α=%.1f, β=%.1f, δ=%.2f",
                     best_params["alpha"], best_params["beta"], best_params["delta"])
        logger.info("   nDCG@10: %.4f", best_metrics.get("ndcg_cut_10", 0.0))
        logger.info("   MAP@100: %.4f", best_metrics.get("map_cut_100", 0.0))
        logger.info("   Recall@100: %.4f", best_metrics.get("recall_100", 0.0))
        logger.info("   MRR@10: %.4f", best_metrics.get("recip_rank_10", 0.0))
        logger.info("   Elapsed: %.1f seconds", elapsed)
        logger.info("=" * 60)

        self._save_results(
            best_params,
            best_metrics,
            baseline_metrics,
            rerank_baseline_metrics,
            all_results,
            elapsed,
        )

        return {
            "best_params": best_params,
            "best_metrics": best_metrics,
            "baseline_metrics": baseline_metrics,
            "rerank_baseline_metrics": rerank_baseline_metrics,
            "candidate_model": self.candidate_model_name,
            "rerank_model": self.model_name,
            "all_results": all_results,
            "elapsed": elapsed,
        }

    def _extract_topk_results(
        self,
        S_base: torch.Tensor,
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
            for j in range(self.top_k):
                idx = top_k_indices[i, j].item()
                if idx < 0:
                    continue
                did = doc_ids[idx]
                doc_scores[did] = float(S_base[i, idx].item())
            results[qid] = doc_scores
        return results

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
        baseline_metrics: Dict[str, float],
        rerank_baseline_metrics: Dict[str, float],
        all_results: List[Dict[str, Any]],
        elapsed: float,
    ):
        os.makedirs(self.output_dir, exist_ok=True)

        summary = {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "candidate_model": self.candidate_model_name,
            "mode": "TwoStage-Promptriever-then-DeIR-Dual-V2-BEIR",
            "dual_queries_source": self.dual_queries_path,
            "top_k": self.top_k,
            "t_safety": self.t_safety,
            "timestamp": datetime.now().isoformat(),
            "best_params": best_params,
            "best_metrics": best_metrics,
            "baseline_metrics": baseline_metrics,
            "rerank_baseline_metrics": rerank_baseline_metrics,
            "elapsed_seconds": elapsed,
        }

        summary_path = os.path.join(self.output_dir, "metrics_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        all_results_path = os.path.join(self.output_dir, "all_results.json")
        with open(all_results_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DeIR-Dual V2 BEIR Evaluation")
    parser.add_argument("--dataset", type=str, required=True,
                        help="BEIR dataset name (e.g., nq, hotpotqa, quora, BeIR/nq)")
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--candidate_model_name", type=str, default=None,
                        help="Stage-1 candidate retriever model (defaults to --model_name).")
    parser.add_argument("--dual_queries_path", type=str, default="",
                        help="Path to dual queries JSONL file")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=100,
                        help="Number of candidates to retrieve in Stage 1")
    parser.add_argument("--alphas", type=str, default="0.5",
                        help="Comma-separated alpha values")
    parser.add_argument("--betas", type=str, default="1.3",
                        help="Comma-separated beta values")
    parser.add_argument("--deltas", type=str, default="0.10",
                        help="Comma-separated delta values")
    parser.add_argument("--t_safety", type=float, default=10.0)
    parser.add_argument("--boundary_mode", type=str, default="semantic",
                        choices=["semantic", "residual_bg"],
                        help="Boundary computation mode")
    parser.add_argument("--residual_margin_scale", type=float, default=1.0,
                        help="Margin scale for residual_bg boundary")
    parser.add_argument("--safety_kappa", type=float, default=0.0,
                        help="Safety kappa for MAD-normalized residual safety gate")
    parser.add_argument("--per_query_ab", type=str, default="false",
                        help="Enable per-query alpha/beta derivation (true/false)")
    parser.add_argument("--beta_derive_mode", type=str, default="max_mean",
                        help="Beta derivation mode for per_query_ab")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Max sequence length for Promptriever encoding")
    parser.add_argument("--cache_checkpoint_interval", type=int, default=200,
                        help="Number of batches per checkpoint shard")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_queries", type=int, default=0,
                        help="Max queries to evaluate (0 = all, prefer dual queries)")
    parser.add_argument("--max_corpus", type=int, default=0,
                        help="Max corpus documents to load (0 = full corpus)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    alphas = [float(a.strip()) for a in args.alphas.split(",")]
    betas = [float(b.strip()) for b in args.betas.split(",")]
    deltas = [float(d.strip()) for d in args.deltas.split(",")]

    evaluator = BEIREvaluator(
        dataset_name=args.dataset,
        model_name=args.model_name,
        candidate_model_name=args.candidate_model_name,
        dual_queries_path=args.dual_queries_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
        t_safety=args.t_safety,
        boundary_mode=args.boundary_mode,
        residual_margin_scale=args.residual_margin_scale,
        safety_kappa=args.safety_kappa,
        per_query_ab=args.per_query_ab.lower() == "true",
        beta_derive_mode=args.beta_derive_mode,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        cache_checkpoint_interval=args.cache_checkpoint_interval,
        cache_dir=args.cache_dir if args.cache_dir else None,
        split=args.split,
        max_queries=args.max_queries,
        max_corpus=args.max_corpus,
    )

    result = evaluator.run(alphas=alphas, betas=betas, deltas=deltas)

    print("\n" + "=" * 60)
    print("📊 Final Results")
    print(f"   Dataset: {args.dataset}")
    print(f"   Candidate model: {result.get('candidate_model', args.candidate_model_name or args.model_name)}")
    print(f"   Rerank model: {result.get('rerank_model', args.model_name)}")
    print(f"   Best params: α={result['best_params']['alpha']}, "
          f"β={result['best_params']['beta']}, δ={result['best_params']['delta']}")
    print(f"   nDCG@10: {result['best_metrics'].get('ndcg_cut_10', 0.0):.4f}")
    print(f"   MAP@100: {result['best_metrics'].get('map_cut_100', 0.0):.4f}")
    print(f"   Recall@100: {result['best_metrics'].get('recall_100', 0.0):.4f}")
    print(f"   MRR@10: {result['best_metrics'].get('recip_rank_10', 0.0):.4f}")
    print(f"   Stage-1 baseline nDCG@10: {result['baseline_metrics'].get('ndcg_cut_10', 0.0):.4f}")
    print(f"   Rerank baseline nDCG@10: {result.get('rerank_baseline_metrics', {}).get('ndcg_cut_10', 0.0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
