"""
DeIR-Dual V2 BEIR Full-Corpus Evaluation Engine

Extends engine_beir.py for full 8.8M MS MARCO corpus evaluation:
  - Multi-GPU parallel corpus encoding (2-4 GPUs)
  - Chunked similarity computation (memory-efficient for 8.8M docs)
  - Custom qrels/queries loading for DL20 (not in BEIR)
  - DL20 passage relevance handling (rel=1 → 0, as per TREC guidelines)

Usage:
  cd /home/luwa/Documents/DSCLR && CUDA_VISIBLE_DEVICES=2,3 /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_beir_full \
    --dataset msmarco \
    --split test \
    --model_name samaya-ai/RepLLaMA-reproduced \
    --dual_queries_path dataset/BEIR/dual_queries/msmarco_CONSERVATIVE_t01.jsonl \
    --alphas 1.0 --betas 1.5 --deltas 0.05 \
    --top_k 1000 --device cuda \
    --output_dir results/beir/msmarco_full_dl19 \
    --cache_dir dataset/BEIR/embeddings/msmarco_full

  # DL20 evaluation:
  cd /home/luwa/Documents/DSCLR && CUDA_VISIBLE_DEVICES=2,3 /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_beir_full \
    --dataset msmarco \
    --split dl20 \
    --qrels_path dataset/BEIR/msmarco_dl20/dl20_qrels.json \
    --queries_path dataset/BEIR/msmarco_dl20/dl20_queries.json \
    --dual_queries_path dataset/BEIR/dual_queries/msmarco_dl20_CONSERVATIVE_t01.jsonl \
    --alphas 1.0 --betas 1.5 --deltas 0.05 \
    --top_k 1000 --device cuda \
    --output_dir results/beir/msmarco_full_dl20 \
    --cache_dir dataset/BEIR/embeddings/msmarco_full
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
import torch.multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import datasets
import pytrec_eval

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


class BEIRFullDataLoader:
    """Extended BEIRDataLoader with support for custom qrels/queries files."""

    def __init__(self, dataset_name: str, split: str = "test",
                 qrels_path: str = "", queries_path: str = ""):
        self.dataset_name = resolve_dataset_name(dataset_name)
        self.split = split
        self.short_name = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
        self.qrels_path = qrels_path
        self.queries_path = queries_path

    def load_corpus(self) -> Dict[str, Dict[str, str]]:
        """Load full corpus (no max_corpus limit for full evaluation)."""
        logger.info(f"📂 Loading full corpus from {self.dataset_name}...")
        ds = datasets.load_dataset(self.dataset_name, "corpus", split="corpus")
        corpus = {}
        for d in tqdm(ds, desc="Loading corpus"):
            doc_id = str(d["_id"])
            title = str(d.get("title", ""))
            text = str(d.get("text", ""))
            if title and title != "None":
                full_text = f"{title} {text}"
            else:
                full_text = text
            corpus[doc_id] = {"text": full_text, "title": title, "body": text}
        logger.info(f"✅ Loaded {len(corpus)} documents")
        return corpus

    def load_queries(self) -> Dict[str, str]:
        if self.queries_path:
            return self._load_custom_queries()
        logger.info(f"📂 Loading queries from {self.dataset_name}...")
        ds = datasets.load_dataset(self.dataset_name, "queries", split="queries")
        queries = {}
        for q in ds:
            qid = str(q["_id"])
            text = str(q.get("text", ""))
            queries[qid] = text
        logger.info(f"✅ Loaded {len(queries)} queries")
        return queries

    def _load_custom_queries(self) -> Dict[str, str]:
        logger.info(f"📂 Loading custom queries from {self.queries_path}...")
        with open(self.queries_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        queries = {str(k): str(v) for k, v in data.items()}
        logger.info(f"✅ Loaded {len(queries)} custom queries")
        return queries

    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        if self.qrels_path:
            return self._load_custom_qrels()
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

    def _load_custom_qrels(self) -> Dict[str, Dict[str, int]]:
        logger.info(f"📂 Loading custom qrels from {self.qrels_path}...")
        with open(self.qrels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        qrels = {}
        for qid, docs in data.items():
            qid_str = str(qid)
            qrels[qid_str] = {}
            for doc_id, rel in docs.items():
                # For DL20 passage ranking: rel=1 ("Related") is NOT relevant
                # Convert rel=1 → 0 for binary relevance metrics
                rel_int = int(rel)
                if self.split == "dl20" and rel_int == 1:
                    rel_int = 0
                qrels[qid_str][str(doc_id)] = rel_int
        logger.info(f"✅ Loaded custom qrels for {len(qrels)} queries ({self.split})")
        return qrels


def _encode_corpus_worker(
    gpu_id: int,
    doc_ids_shard: List[str],
    doc_texts_shard: List[str],
    model_name: str,
    cache_dir: str,
    shard_name: str,
    batch_size: int,
    max_seq_length: int,
    cache_checkpoint_interval: int,
    result_queue: mp.Queue,
):
    """Worker function for multi-GPU corpus encoding."""
    # Fix sys.path for subprocess
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    device = f"cuda:{gpu_id}"
    logger.info(f"🖥️ GPU {gpu_id}: Encoding {len(doc_ids_shard)} docs...")

    from eval.models import ModelFactory
    encoder_kwargs = {
        "model_name": model_name,
        "device": device,
        "batch_size": batch_size,
        "normalize_embeddings": True,
    }
    if max_seq_length:
        encoder_kwargs["max_seq_length"] = max_seq_length
    encoder = ModelFactory.create(**encoder_kwargs)

    shard_dir = os.path.join(cache_dir, f"shard_{shard_name}")
    meta_path = os.path.join(shard_dir, "meta.json")
    os.makedirs(shard_dir, exist_ok=True)

    # Check for existing checkpoint
    processed_count = 0
    shards_info = []
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            processed_count = int(meta.get("processed_count", 0))
            shards_info = meta.get("shards", [])
            logger.info(f"GPU {gpu_id}: Resuming from {processed_count} docs")
        except Exception:
            processed_count = 0

    # Encode in batches with checkpointing
    shard_buffers = []
    shard_start = processed_count
    batches_in_shard = 0

    for i in tqdm(range(processed_count, len(doc_ids_shard), batch_size),
                  desc=f"GPU{gpu_id}", position=gpu_id):
        batch_texts = doc_texts_shard[i:i + batch_size]
        batch_emb = encoder.encode_documents(batch_texts, batch_size=batch_size)
        if batch_emb.dim() == 2:
            batch_emb = F.normalize(batch_emb, p=2, dim=1)
        shard_buffers.append(batch_emb.cpu())
        batches_in_shard += 1

        if batches_in_shard >= cache_checkpoint_interval:
            shard_end = min(i + batch_size, len(doc_ids_shard))
            shard_tensor = torch.cat(shard_buffers, dim=0)
            shard_path = os.path.join(shard_dir, f"chunk_{shard_start}_{shard_end}.pt")
            torch.save(shard_tensor, shard_path)
            shards_info.append({"start": shard_start, "end": shard_end, "path": shard_path})
            processed_count = shard_end
            shard_buffers = []
            batches_in_shard = 0
            shard_start = processed_count

            with open(meta_path, "w") as f:
                json.dump({"processed_count": processed_count, "shards": shards_info}, f)

    # Save remaining
    if shard_buffers:
        shard_end = len(doc_ids_shard)
        shard_tensor = torch.cat(shard_buffers, dim=0)
        shard_path = os.path.join(shard_dir, f"chunk_{shard_start}_{shard_end}.pt")
        torch.save(shard_tensor, shard_path)
        shards_info.append({"start": shard_start, "end": shard_end, "path": shard_path})
        processed_count = shard_end
        with open(meta_path, "w") as f:
            json.dump({"processed_count": processed_count, "shards": shards_info}, f)

    # Consolidate shards
    logger.info(f"GPU {gpu_id}: Consolidating {len(shards_info)} shards...")
    shards_sorted = sorted(shards_info, key=lambda x: x["start"])
    embeddings_list = [torch.load(s["path"], map_location="cpu") for s in shards_sorted]
    embeddings = torch.cat(embeddings_list, dim=0)

    # Save final consolidated embedding
    final_path = os.path.join(cache_dir, f"{shard_name}_embeddings.pt")
    torch.save({"doc_ids": doc_ids_shard, "embeddings": embeddings}, final_path)

    logger.info(f"GPU {gpu_id}: Done. Shape={embeddings.shape}")
    result_queue.put((gpu_id, shard_name, final_path))


class BEIRFullEvaluator:
    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        dual_queries_path: str,
        output_dir: str,
        candidate_model_name: Optional[str] = None,
        top_k: int = 1000,
        t_safety: float = 20.0,
        device: str = "auto",
        batch_size: int = 64,
        max_seq_length: Optional[int] = None,
        cache_checkpoint_interval: int = 200,
        cache_dir: Optional[str] = None,
        split: str = "test",
        qrels_path: str = "",
        queries_path: str = "",
        gpus: str = "",
    ):
        self.dataset_name = dataset_name
        self.short_name = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
        self.model_name = model_name
        self.candidate_model_name = candidate_model_name or model_name
        self.dual_queries_path = dual_queries_path
        self.output_dir = output_dir
        self.top_k = top_k
        self.t_safety = t_safety
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.cache_checkpoint_interval = max(1, cache_checkpoint_interval)
        self.split = split
        self.qrels_path = qrels_path
        self.queries_path = queries_path

        if device == "auto":
            try:
                torch.cuda._lazy_init()
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device

        self.cache_dir = cache_dir or f"dataset/BEIR/embeddings/{self.short_name}_full"

        # Parse GPU list for multi-GPU encoding
        if gpus:
            self.gpu_ids = [int(g.strip()) for g in gpus.split(",")]
        else:
            # Auto-detect from CUDA_VISIBLE_DEVICES
            vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if vis:
                self.gpu_ids = [int(g.strip()) for g in vis.split(",")]
            else:
                self.gpu_ids = list(range(torch.cuda.device_count()))

        self.data_loader = BEIRFullDataLoader(
            dataset_name, split=split,
            qrels_path=qrels_path, queries_path=queries_path,
        )

    def _create_encoder(self, model_name: str):
        from eval.models import ModelFactory
        encoder_kwargs = {
            "model_name": model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "normalize_embeddings": True,
        }
        if self.max_seq_length:
            encoder_kwargs["max_seq_length"] = self.max_seq_length
        encoder = ModelFactory.create(**encoder_kwargs)
        return encoder

    def _encode_and_cache_corpus_multi_gpu(
        self,
        corpus: Dict[str, Dict[str, str]],
        model_name: str,
    ) -> Tuple[torch.Tensor, List[str]]:
        """Load pre-encoded corpus embeddings or fall back to single-GPU encoding.

        For multi-GPU encoding, use encode_corpus_shard.py separately:
          CUDA_VISIBLE_DEVICES=0 python -m eval.encode_corpus_shard --shard 0 --num_shards 4 ...
          CUDA_VISIBLE_DEVICES=1 python -m eval.encode_corpus_shard --shard 1 --num_shards 4 ...
          Then: python -m eval.encode_corpus_shard --merge --num_shards 4 ...
        """
        # Try loading from merged shard cache first
        cache_path = os.path.join(self.cache_dir, f"{self.short_name}_full_corpus.pt")
        if os.path.exists(cache_path):
            logger.info(f"📂 Loading cached corpus embeddings from {cache_path}")
            data = torch.load(cache_path, map_location="cpu", weights_only=False)
            doc_ids = data["doc_ids"]
            embeddings = data["embeddings"]
            logger.info(f"✅ Cache hit: {len(doc_ids)} documents, shape={embeddings.shape}")
            return embeddings, doc_ids

        # Try loading from single-file cache
        cache_path = os.path.join(self.cache_dir, f"{self.short_name}_repllama_corpus.pt")
        if os.path.exists(cache_path):
            logger.info(f"📂 Loading cached corpus embeddings from {cache_path}")
            data = torch.load(cache_path, map_location="cpu", weights_only=False)
            doc_ids = data["doc_ids"]
            embeddings = data["embeddings"]
            if len(doc_ids) == len(corpus):
                logger.info(f"✅ Cache hit: {len(doc_ids)} documents, shape={embeddings.shape}")
                return embeddings, doc_ids
            else:
                logger.warning(f"⚠️ Cache size mismatch ({len(doc_ids)} vs {len(corpus)}), re-encoding")

        # Fall back to single-GPU encoding
        logger.info("⚠️ No cached embeddings found, falling back to single-GPU encoding")
        logger.info("💡 For faster multi-GPU encoding, use: python -m eval.encode_corpus_shard")
        return self._encode_and_cache_corpus_single_gpu(corpus, model_name)

    def _encode_and_cache_corpus_single_gpu(
        self,
        corpus: Dict[str, Dict[str, str]],
        model_name: str,
    ) -> Tuple[torch.Tensor, List[str]]:
        """Single GPU corpus encoding with checkpointing (original approach)."""
        cache_path = os.path.join(self.cache_dir, f"{self.short_name}_replmma_corpus.pt")
        checkpoint_dir = f"{cache_path}.checkpoint"
        shard_dir = os.path.join(checkpoint_dir, "shards")
        meta_path = os.path.join(checkpoint_dir, "meta.json")

        if os.path.exists(cache_path):
            logger.info(f"📂 Loading cached corpus embeddings from {cache_path}")
            data = torch.load(cache_path, map_location="cpu", weights_only=False)
            doc_ids = data["doc_ids"]
            embeddings = data["embeddings"]
            if len(doc_ids) == len(corpus):
                logger.info(f"✅ Cache hit: {len(doc_ids)} documents")
                return embeddings, doc_ids
            else:
                logger.warning(f"⚠️ Cache size mismatch, re-encoding")

        doc_ids = list(corpus.keys())
        doc_texts = [corpus[did]["text"] for did in doc_ids]

        processed_count = 0
        shards = []
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                processed_count = int(meta.get("processed_count", 0))
                shards = meta.get("shards", [])
                logger.info(f"📌 Resuming from {processed_count} docs")
            except Exception:
                processed_count = 0

        os.makedirs(shard_dir, exist_ok=True)

        encoder = self._create_encoder(model_name)
        logger.info(f"📚 Encoding {len(doc_ids)} docs (resume from {processed_count})...")

        shard_buffers = []
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
                shard_end = min(i + self.batch_size, len(doc_ids))
                shard_tensor = torch.cat(shard_buffers, dim=0)
                shard_path = os.path.join(shard_dir, f"shard_{shard_start}_{shard_end}.pt")
                torch.save(shard_tensor, shard_path)
                shards.append({"start": shard_start, "end": shard_end, "path": shard_path})
                processed_count = shard_end
                shard_buffers = []
                batches_in_shard = 0
                shard_start = processed_count

                with open(meta_path, "w") as f:
                    json.dump({"processed_count": processed_count, "shards": shards}, f)

        if shard_buffers:
            shard_end = len(doc_ids)
            shard_tensor = torch.cat(shard_buffers, dim=0)
            shard_path = os.path.join(shard_dir, f"shard_{shard_start}_{shard_end}.pt")
            torch.save(shard_tensor, shard_path)
            shards.append({"start": shard_start, "end": shard_end, "path": shard_path})
            with open(meta_path, "w") as f:
                json.dump({"processed_count": len(doc_ids), "shards": shards}, f)

        logger.info(f"📦 Consolidating {len(shards)} shards...")
        shards_sorted = sorted(shards, key=lambda x: x["start"])
        embeddings_list = [torch.load(s["path"], map_location="cpu") for s in shards_sorted]
        embeddings = torch.cat(embeddings_list, dim=0)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"doc_ids": doc_ids, "embeddings": embeddings}, cache_path)
        logger.info(f"💾 Cached to {cache_path} (shape={embeddings.shape})")

        return embeddings, doc_ids

    def load_dual_queries(self) -> Dict[str, Dict[str, Any]]:
        if not self.dual_queries_path or not os.path.exists(self.dual_queries_path):
            logger.warning("⚠️ No dual queries file, using Q_plus=Q_base, Q_minus=[NONE]")
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
        qrels_eval = {qid: rel for qid, rel in qrels.items() if qid in results}
        if not qrels_eval:
            return {}

        results_str = {qid: {did: float(s) for did, s in scores.items()} for qid, scores in results.items()}
        qrels_str = {qid: {did: int(r) for did, r in rel_dict.items()} for qid, rel_dict in qrels_eval.items()}

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
        logger.info("🚀 DeIR-Dual V2 Full-Corpus BEIR Evaluation")
        logger.info("=" * 60)

        start_time = time.time()

        # Load data
        queries = self.data_loader.load_queries()
        qrels = self.data_loader.load_qrels()

        logger.info(f"📋 Qrels: {len(qrels)} queries, split={self.split}")

        # Load full corpus
        corpus = self.data_loader.load_corpus()
        logger.info(f"📚 Full corpus: {len(corpus)} documents")

        eval_queries = {qid: text for qid, text in queries.items() if qid in qrels}
        logger.info(f"📊 Evaluating on {len(eval_queries)} queries (with qrels)")

        dual_data = self.load_dual_queries()

        # Prepare query lists
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

        # Encode corpus with multi-GPU (no encoder loaded in main process yet)
        candidate_doc_embeddings, candidate_doc_ids = self._encode_and_cache_corpus_multi_gpu(
            corpus, self.candidate_model_name,
        )
        del corpus  # Free memory
        import gc
        gc.collect()

        # Encode queries (load encoder now for query encoding)
        candidate_encoder = self._create_encoder(self.candidate_model_name)
        logger.info("📊 Encoding candidate Q_base...")
        candidate_q_base_emb = candidate_encoder.encode_queries(q_base_list, batch_size=self.batch_size)
        if candidate_q_base_emb.dim() == 2:
            candidate_q_base_emb = F.normalize(candidate_q_base_emb, p=2, dim=1)
        candidate_q_base_emb = candidate_q_base_emb.to(self.device)

        # Chunked similarity for top-k retrieval (8.8M docs)
        logger.info(f"📊 Stage 1: Dense retrieval top-{self.top_k} from {len(candidate_doc_ids)} docs...")
        d_chunk_size = 500000
        q_chunk_size = 500  # Process queries in chunks to avoid OOM on large query sets

        # For 8.8M docs with many queries, we need memory-efficient top-k
        # Process in (query_chunk, doc_chunk) blocks, maintaining top-k per query
        top_k_scores = torch.full((len(query_ids), self.top_k), float('-inf'))
        top_k_indices = torch.zeros(len(query_ids), self.top_k, dtype=torch.long)

        for qi_start in tqdm(range(0, len(query_ids), q_chunk_size), desc="Stage 1 retrieval (queries)"):
            qi_end = min(qi_start + q_chunk_size, len(query_ids))
            q_emb = candidate_q_base_emb[qi_start:qi_end]  # (q_chunk, dim)

            cur_top_k_scores = top_k_scores[qi_start:qi_end].clone()
            cur_top_k_indices = top_k_indices[qi_start:qi_end].clone()

            for di in range(0, len(candidate_doc_ids), d_chunk_size):
                d_end = min(di + d_chunk_size, len(candidate_doc_ids))
                doc_chunk = candidate_doc_embeddings[di:d_end].to(self.device)

                # (q_chunk, dim) x (dim, d_chunk) -> (q_chunk, d_chunk)
                chunk_scores = torch.matmul(q_emb, doc_chunk.T)
                del doc_chunk
                torch.cuda.empty_cache()

                # Merge top-k per query in this query chunk
                chunk_offset = torch.arange(di, d_end)
                # combined: (q_chunk, top_k + d_chunk_size)
                combined_scores = torch.cat([cur_top_k_scores, chunk_scores.cpu()], dim=1)
                combined_indices = torch.cat([
                    cur_top_k_indices,
                    chunk_offset.unsqueeze(0).expand(qi_end - qi_start, -1)
                ], dim=1)

                k = min(self.top_k, combined_scores.shape[1])
                topk = torch.topk(combined_scores, k, dim=1)
                cur_top_k_scores = topk.values
                cur_top_k_indices = combined_indices.gather(1, topk.indices)

                del chunk_scores, combined_scores, combined_indices
                torch.cuda.empty_cache()

            top_k_scores[qi_start:qi_end] = cur_top_k_scores
            top_k_indices[qi_start:qi_end] = cur_top_k_indices

        # Baseline metrics
        baseline_results = self._extract_topk_results_from_full(
            top_k_scores, top_k_indices, query_ids, candidate_doc_ids
        )
        baseline_metrics = self.compute_metrics(baseline_results, qrels)
        logger.info("📊 Baseline (dense retrieval):")
        for k, v in sorted(baseline_metrics.items()):
            logger.info(f"   {k}: {v:.4f}")

        # Rerank encoder
        if self.candidate_model_name == self.model_name:
            rerank_encoder = candidate_encoder
            rerank_doc_embeddings = candidate_doc_embeddings
            rerank_doc_ids = candidate_doc_ids
        else:
            del candidate_encoder
            torch.cuda.empty_cache()
            rerank_encoder = self._create_encoder(self.model_name)
            rerank_doc_embeddings, rerank_doc_ids = self._encode_and_cache_corpus_multi_gpu(
                {}, self.model_name,
            )

        # Encode rerank queries
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

        # Compute rerank scores for top-k candidates
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
        logger.info("📊 Baseline (rerank model):")
        for k, v in sorted(rerank_baseline_metrics.items()):
            logger.info(f"   {k}: {v:.4f}")

        # Parameter search
        all_results = []
        best_metrics = None
        best_params = None

        for alpha in alphas:
            for beta in betas:
                for delta in deltas:
                    S_final_topk = torch.zeros(len(query_ids), self.top_k)
                    for i in range(len(query_ids)):
                        valid_mask = top_k_indices[i] >= 0
                        if valid_mask.sum() == 0:
                            continue
                        k = valid_mask.sum().item()
                        s_final = self._score_deir_dual_v2(
                            s_base=S_base_topk[i, :k],
                            s_req=S_req_topk[i, :k],
                            s_neg=S_neg_topk[i, :k],
                            cos_qbase_qneg=float(cos_qbase_qneg[i].item()),
                            has_req=bool(has_req_mask[i] > 0),
                            has_neg=bool(has_neg_mask[i] > 0),
                            alpha=alpha, beta=beta, delta=delta,
                        )
                        S_final_topk[i, :k] = s_final

                    reranked_results = self._extract_topk_results_from_scores(
                        S_final_topk, top_k_indices, query_ids, rerank_doc_ids
                    )
                    metrics = self.compute_metrics(reranked_results, qrels)

                    ndcg10 = metrics.get("ndcg_cut_10", 0.0)
                    logger.info(f"α={alpha}, β={beta}, δ={delta}: nDCG@10={ndcg10:.4f}")

                    all_results.append({"alpha": alpha, "beta": beta, "delta": delta, **metrics})
                    if best_metrics is None or ndcg10 > best_metrics.get("ndcg_cut_10", 0.0):
                        best_metrics = metrics
                        best_params = {"alpha": alpha, "beta": beta, "delta": delta}

        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info("📊 Full-Corpus Evaluation Complete")
        logger.info("   Split: %s", self.split)
        logger.info("   Best: α=%.1f, β=%.1f, δ=%.02f",
                     best_params["alpha"], best_params["beta"], best_params["delta"])
        logger.info("   nDCG@10: %.4f", best_metrics.get("ndcg_cut_10", 0.0))
        logger.info("   MRR@10: %.4f", best_metrics.get("recip_rank_10", 0.0))
        logger.info("   Elapsed: %.1f sec", elapsed)
        logger.info("=" * 60)

        self._save_results(best_params, best_metrics, baseline_metrics,
                           rerank_baseline_metrics, all_results, elapsed)

        return {
            "best_params": best_params,
            "best_metrics": best_metrics,
            "baseline_metrics": baseline_metrics,
            "rerank_baseline_metrics": rerank_baseline_metrics,
            "all_results": all_results,
            "elapsed": elapsed,
        }

    def _extract_topk_results_from_full(
        self,
        top_k_scores: torch.Tensor,
        top_k_indices: torch.Tensor,
        query_ids: List[str],
        doc_ids: List[str],
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        for i, qid in enumerate(query_ids):
            doc_scores = {}
            scored_pairs = []
            for j in range(self.top_k):
                idx = top_k_indices[i, j].item()
                if idx < 0:
                    continue
                did = doc_ids[idx]
                score = float(top_k_scores[i, j].item())
                scored_pairs.append((did, score))
            scored_pairs.sort(key=lambda x: x[1], reverse=True)
            results[qid] = dict(scored_pairs)
        return results

    def _extract_topk_results_from_scores(
        self,
        S_final_topk: torch.Tensor,
        top_k_indices: torch.Tensor,
        query_ids: List[str],
        doc_ids: List[str],
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        for i, qid in enumerate(query_ids):
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

    def _save_results(self, best_params, best_metrics, baseline_metrics,
                      rerank_baseline_metrics, all_results, elapsed):
        os.makedirs(self.output_dir, exist_ok=True)
        summary = {
            "dataset": self.dataset_name,
            "split": self.split,
            "corpus": "full",
            "model": self.model_name,
            "candidate_model": self.candidate_model_name,
            "mode": "TwoStage-DenseRetrieval-then-DeIR-Dual-V2-Full",
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
        with open(os.path.join(self.output_dir, "metrics_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DeIR-Dual V2 Full-Corpus BEIR Evaluation")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--candidate_model_name", type=str, default=None)
    parser.add_argument("--dual_queries_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--alphas", type=str, default="1.0")
    parser.add_argument("--betas", type=str, default="1.5")
    parser.add_argument("--deltas", type=str, default="0.05")
    parser.add_argument("--t_safety", type=float, default=20.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--cache_checkpoint_interval", type=int, default=500)
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--qrels_path", type=str, default="",
                        help="Custom qrels JSON file (for DL20)")
    parser.add_argument("--queries_path", type=str, default="",
                        help="Custom queries JSON file (for DL20)")
    parser.add_argument("--gpus", type=str, default="",
                        help="Comma-separated GPU IDs for multi-GPU encoding (e.g., '2,3')")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    alphas = [float(a.strip()) for a in args.alphas.split(",")]
    betas = [float(b.strip()) for b in args.betas.split(",")]
    deltas = [float(d.strip()) for d in args.deltas.split(",")]

    evaluator = BEIRFullEvaluator(
        dataset_name=args.dataset,
        model_name=args.model_name,
        candidate_model_name=args.candidate_model_name,
        dual_queries_path=args.dual_queries_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
        t_safety=args.t_safety,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        cache_checkpoint_interval=args.cache_checkpoint_interval,
        cache_dir=args.cache_dir if args.cache_dir else None,
        split=args.split,
        qrels_path=args.qrels_path,
        queries_path=args.queries_path,
        gpus=args.gpus,
    )

    result = evaluator.run(alphas=alphas, betas=betas, deltas=deltas)

    print("\n" + "=" * 60)
    print("📊 Final Results")
    print(f"   Dataset: {args.dataset}, Split: {args.split}")
    print(f"   Best: α={result['best_params']['alpha']}, "
          f"β={result['best_params']['beta']}, δ={result['best_params']['delta']}")
    print(f"   nDCG@10: {result['best_metrics'].get('ndcg_cut_10', 0.0):.4f}")
    print(f"   MRR@10: {result['best_metrics'].get('recip_rank_10', 0.0):.4f}")
    print(f"   Baseline nDCG@10: {result['baseline_metrics'].get('ndcg_cut_10', 0.0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
