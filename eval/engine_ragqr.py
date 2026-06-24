"""
RAG-Query-Rewriting Evaluation Engine for FollowIR

Reference: Ma et al. "Query Rewriting for Retrieval-Augmented Large Language Models" (EMNLP 2023)
Paper: https://aclanthology.org/2023.emnlp-main.322
Official code: https://github.com/xbmxb/RAG-query-rewriting
Model weights: https://drive.google.com/drive/folders/1NpvRC0TivgCYEgui9XrXhIrUO7x4SX5V
HF mirror: catyung/t5l-turbo-hotpot-0331

Faithfully reproduced from the official code:
    - Model: T5-large (770M) fine-tuned via PPO on HotpotQA (t5l-turbo-hotpot-0331)
    - Prompt prefix: "rewrite a better search query: " (from rl config datapool.args.prompt_prefix)
    - Generation: num_beams=4, max_length=50 (from rl config generation_kwargs)
    - Output: Direct text output (seq2seq model, no JSON parsing needed)
    - The rewriter does NOT incorporate instructions into rewriting (trained for web search query rewriting)
    - However, instructions ARE appended to rewritten queries before RepLLaMA encoding
      (consistent with V2 engine's q_base = query_text + instruction approach)

Evaluation pipeline:
    - Phase 1: T5-large rewriter rewrites queries (replaces V2's Q_plus/Q_minus dual-track)
    - Phase 2: RepLLaMA encodes rewritten queries + documents (same as V2 engine)
    - Phase 3: Matrix multiplication for similarity (same as V2 engine)
    - Phase 4: FollowIR metrics computation (same as V2 engine)

Usage:
    cd /home/luwa/Documents/DSCLR && /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_ragqr \
        --task_name Core17InstructionRetrieval \
        --model_name samaya-ai/RepLLaMA-reproduced \
        --ragqr_model_path /home/luwa/Documents/models/rag-query-rewriting/t5l-turbo-hotpot-0331 \
        --device cuda \
        --output_dir results/ragqr/Core17
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any

import torch
from tqdm import tqdm

from eval.engine_dscrl import DSCLREvaluatorEngine, load_cached_embeddings
from eval.metrics import FollowIREvaluator

logger = logging.getLogger(__name__)

RAGQR_PROMPT_PREFIX = "rewrite a better search query: "


class RAGQRGenerator:
    """RAG-Query-Rewriting T5-large rewriter

    Official code (rl/RL4LMs/scripts/training/task_configs/hotpot/t5_ppo_0314_v2.yml):
        datapool.args.prompt_prefix: "rewrite a better search query: "
        generation_kwargs:
            min_length: 1
            max_length: 50
            num_return_sequences: 1
            num_beams: 4
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_length: int = 50,
        num_beams: int = 4,
    ):
        self.max_length = max_length
        self.num_beams = num_beams
        self.device = device

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info(f"📥 Loading RAG-QR T5 rewriter from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        effective_device = device
        if device.startswith("cuda"):
            try:
                torch.cuda._lazy_init()
                if not torch.cuda.is_available():
                    effective_device = "cpu"
            except Exception:
                effective_device = "cpu"

        if effective_device == "cpu":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path, torch_dtype=torch.float32
            ).to("cpu")
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        self.model.eval()
        logger.info(f"✅ RAG-QR T5 rewriter loaded on {effective_device}")

    def generate(self, query: str) -> str:
        input_text = RAGQR_PROMPT_PREFIX + query
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                num_return_sequences=1,
                min_length=1,
            )

        rewritten_query = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return rewritten_query.strip()


class RAGQREvaluator(DSCLREvaluatorEngine):
    """RAG-Query-Rewriting evaluation engine for FollowIR

    Inherits from DSCLREvaluatorEngine to reuse the exact same evaluation pipeline:
    - Same encoder (RepLLaMA)
    - Same document indexing (L2 normalized embeddings)
    - Same score computation (matrix multiplication)
    - Same result extraction (_extract_results)
    - Same metrics computation (FollowIREvaluator)

    The ONLY difference is the query rewriting method:
    - V2: Uses Q_plus/Q_minus dual-track with reward-penalty formula
    - RAG-QR: Uses T5-large rewriter (PPO-trained) to rewrite queries, then encodes with RepLLaMA
    """

    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_dir: str,
        ragqr_model_path: str,
        ragqr_max_length: int = 50,
        ragqr_num_beams: int = 4,
        ragqr_cache_dir: str = "dataset/FollowIR_test/ragqr_queries",
        gpu_id: int = 0,
        **kwargs,
    ):
        self.ragqr_model_path = ragqr_model_path
        self.ragqr_max_length = ragqr_max_length
        self.ragqr_num_beams = ragqr_num_beams
        self.ragqr_cache_dir = ragqr_cache_dir
        self.gpu_id = gpu_id

        kwargs.setdefault("device", f"cuda:{gpu_id}")
        kwargs.setdefault("batch_size", 64)
        kwargs.setdefault("use_cache", True)

        super().__init__(
            model_name=model_name,
            task_name=task_name,
            output_dir=output_dir,
            **kwargs,
        )

        os.makedirs(self.ragqr_cache_dir, exist_ok=True)

        logger.info("🏛️ RAG-Query-Rewriting 改写模式已启用")
        logger.info(f"📁 RAG-QR 模型: {self.ragqr_model_path}")
        logger.info(f"📁 改写缓存目录: {self.ragqr_cache_dir}")

    def _get_ragqr_cache_path(self) -> str:
        model_name = os.path.basename(self.ragqr_model_path.rstrip("/"))
        return os.path.join(
            self.ragqr_cache_dir,
            f"{self.task_name}_ragqr_{model_name}.jsonl",
        )

    def _load_ragqr_cache(self) -> Dict[str, str]:
        cache_path = self._get_ragqr_cache_path()
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        cache[item["qid"]] = item["rewritten_query"]
            logger.info(f"📂 Loaded RAG-QR cache: {len(cache)} entries from {cache_path}")
        return cache

    def _save_ragqr_cache(self, cache: Dict[str, str], queries_info: Dict[str, Dict]):
        cache_path = self._get_ragqr_cache_path()
        with open(cache_path, "w", encoding="utf-8") as f:
            for qid, rewritten_query in cache.items():
                info = queries_info.get(qid, {})
                f.write(json.dumps({
                    "qid": qid,
                    "query": info.get("query", ""),
                    "instruction": info.get("instruction", ""),
                    "rewritten_query": rewritten_query,
                }, ensure_ascii=False) + "\n")
        logger.info(f"💾 RAG-QR cache saved: {cache_path}")

    def run(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("🚀 Starting RAG-Query-Rewriting Evaluation on FollowIR")
        logger.info(f"   Paper: Ma et al. EMNLP 2023")
        logger.info(f"   Official code: https://github.com/xbmxb/RAG-query-rewriting")
        logger.info(f"   Rewriter model: {self.ragqr_model_path}")
        logger.info(f"   Encoder: {self.model_name} (same as V2 engine)")
        logger.info(f"   Prompt prefix: '{RAGQR_PROMPT_PREFIX}'")
        logger.info(f"   num_beams={self.ragqr_num_beams}, max_length={self.ragqr_max_length}")
        logger.info(f"   Instruction: incorporated into encoding (rewritten_query + instruction)")
        logger.info("=" * 60)

        start_time = time.time()

        corpus, q_og, q_changed, candidates = self.data_loader.load()
        q_raw_og, q_raw_changed = self.data_loader.load_raw_queries()

        ragqr_cache = self._load_ragqr_cache()
        queries_info = {}

        query_ids_og: List[str] = []
        query_ids_changed: List[str] = []
        queries_to_generate: List[Tuple[str, str]] = []

        for qid in q_og.keys():
            query_ids_og.append(qid)
            raw = q_raw_og.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            queries_info[qid] = {"query": query_text, "instruction": instruction}
            if qid not in ragqr_cache:
                queries_to_generate.append((qid, query_text))

        for qid in q_changed.keys():
            query_ids_changed.append(qid)
            raw = q_raw_changed.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            queries_info[qid] = {"query": query_text, "instruction": instruction}
            if qid not in ragqr_cache:
                queries_to_generate.append((qid, query_text))

        if queries_to_generate:
            logger.info(f"🔄 Phase 1: RAG-QR rewriting {len(queries_to_generate)} queries...")
            generator = RAGQRGenerator(
                model_path=self.ragqr_model_path,
                device=f"cuda:{self.gpu_id}",
                max_length=self.ragqr_max_length,
                num_beams=self.ragqr_num_beams,
            )

            for qid, query_text in tqdm(queries_to_generate, desc="RAG-QR generation"):
                rewritten_query = generator.generate(query_text)
                ragqr_cache[qid] = rewritten_query

            self._save_ragqr_cache(ragqr_cache, queries_info)
            del generator
            torch.cuda.empty_cache()
            logger.info("✅ Rewritten queries generated and cached, generator unloaded")
        else:
            logger.info(f"✅ All rewritten queries already cached ({len(ragqr_cache)} entries)")

        logger.info("🔄 Phase 2: Loading encoder and computing retrieval (same pipeline as V2)...")
        all_doc_ids = self._get_all_candidate_doc_ids(candidates)

        cached_data = None
        if self.use_cache:
            cached_data = load_cached_embeddings(self.cache_dir, self.task_name, self.model_name)

        if cached_data is not None:
            cached_embeddings, cached_doc_ids = cached_data
            if set(cached_doc_ids) == set(all_doc_ids):
                logger.info(f"✅ Using cached document embeddings ({len(cached_doc_ids)} docs)")
                self.retriever.set_embeddings(cached_embeddings, cached_doc_ids)
            else:
                logger.warning("⚠️ Cached doc IDs mismatch, re-encoding...")
                doc_texts = [corpus[did]["text"] for did in all_doc_ids]
                self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)
        else:
            logger.info("📚 Encoding candidate documents...")
            doc_texts = [corpus[did]["text"] for did in all_doc_ids]
            self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)

        og_rewritten_queries = []
        for qid in query_ids_og:
            rewritten = ragqr_cache[qid]
            instruction = queries_info[qid].get("instruction", "")
            og_rewritten_queries.append(f"{rewritten} {instruction}".strip() if instruction else rewritten)

        changed_rewritten_queries = []
        for qid in query_ids_changed:
            rewritten = ragqr_cache[qid]
            instruction = queries_info[qid].get("instruction", "")
            changed_rewritten_queries.append(f"{rewritten} {instruction}".strip() if instruction else rewritten)

        logger.info("📊 Encoding OG rewritten queries with RepLLaMA...")
        q_emb_og = self._encode_queries(og_rewritten_queries)

        logger.info("📊 Encoding Changed rewritten queries with RepLLaMA...")
        q_emb_changed = self._encode_queries(changed_rewritten_queries)

        device = self.retriever.doc_embeddings.device
        q_emb_og = q_emb_og.to(device)
        q_emb_changed = q_emb_changed.to(device)

        logger.info("📊 Computing similarity scores (matrix multiplication, same as V2)...")
        S_og = torch.matmul(q_emb_og, self.retriever.doc_embeddings.T)
        S_changed = torch.matmul(q_emb_changed, self.retriever.doc_embeddings.T)

        logger.info("📊 Extracting results and computing FollowIR metrics...")
        results_og = self._extract_results(S_og, query_ids_og, candidates)
        results_changed = self._extract_results(S_changed, query_ids_changed, candidates)

        evaluator = FollowIREvaluator(self.task_name)
        metrics = evaluator.evaluate(results_og, results_changed)

        elapsed = time.time() - start_time

        p_mrr = metrics.get("p-MRR", 0.0)
        og_map = metrics.get("original", {}).get("map_at_1000", 0.0)
        changed_map = metrics.get("changed", {}).get("map_at_1000", 0.0)
        og_ndcg5 = metrics.get("original", {}).get("ndcg_at_5", 0.0)
        changed_ndcg5 = metrics.get("changed", {}).get("ndcg_at_5", 0.0)

        logger.info("=" * 60)
        logger.info("📊 RAG-Query-Rewriting Evaluation Results")
        logger.info(f"   Task: {self.task_name}")
        logger.info(f"   Rewriter: {self.ragqr_model_path}")
        logger.info(f"   Encoder: {self.model_name}")
        logger.info(f"   p-MRR: {p_mrr:.4f}")
        logger.info(f"   OG MAP@1000: {og_map:.4f}")
        logger.info(f"   Changed MAP@1000: {changed_map:.4f}")
        logger.info(f"   OG nDCG@5: {og_ndcg5:.4f}")
        logger.info(f"   Changed nDCG@5: {changed_ndcg5:.4f}")
        logger.info(f"   Elapsed: {elapsed:.1f}s")
        logger.info("=" * 60)

        self._save_results(
            metrics=metrics,
            results_og=results_og,
            results_changed=results_changed,
            ragqr_cache=ragqr_cache,
            query_ids_og=query_ids_og,
            query_ids_changed=query_ids_changed,
            elapsed=elapsed,
        )

        return {
            "metrics": metrics,
            "elapsed": elapsed,
        }

    def _save_results(
        self,
        metrics: Dict[str, Any],
        results_og: Dict[str, Dict[str, float]],
        results_changed: Dict[str, Dict[str, float]],
        ragqr_cache: Dict[str, str],
        query_ids_og: List[str],
        query_ids_changed: List[str],
        elapsed: float,
    ):
        os.makedirs(self.output_dir, exist_ok=True)

        summary = {
            "task": self.task_name,
            "model": self.model_name,
            "mode": "RAG-Query-Rewriting (Ma et al. EMNLP 2023)",
            "ragqr_model": self.ragqr_model_path,
            "ragqr_prompt_prefix": RAGQR_PROMPT_PREFIX,
            "ragqr_max_length": self.ragqr_max_length,
            "ragqr_num_beams": self.ragqr_num_beams,
            "ragqr_instruction_aware": False,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "p-MRR": metrics.get("p-MRR", 0.0),
                "original": metrics.get("original", {}),
                "changed": metrics.get("changed", {}),
                "full_scores": metrics.get("full_scores", {}),
            },
            "elapsed_seconds": elapsed,
        }

        summary_path = os.path.join(self.output_dir, "metrics_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        out_og = os.path.join(self.output_dir, "ranking_og.json")
        out_changed = os.path.join(self.output_dir, "ranking_changed.json")
        with open(out_og, "w", encoding="utf-8") as f:
            json.dump(results_og, f, ensure_ascii=False)
        with open(out_changed, "w", encoding="utf-8") as f:
            json.dump(results_changed, f, ensure_ascii=False)

        rewrite_samples_path = os.path.join(self.output_dir, "ragqr_samples.json")
        samples = {}
        for qid in query_ids_og[:5]:
            samples[qid] = {
                "type": "og",
                "original_query": ragqr_cache.get(qid, "")[:500],
            }
        for qid in query_ids_changed[:5]:
            samples[qid] = {
                "type": "changed",
                "original_query": ragqr_cache.get(qid, "")[:500],
            }
        with open(rewrite_samples_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="RAG-Query-Rewriting Evaluation on FollowIR")
    parser.add_argument("--task_name", type=str, required=True,
                        choices=["Core17InstructionRetrieval", "Robust04InstructionRetrieval", "News21InstructionRetrieval"])
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--ragqr_model_path", type=str,
                        default="/home/luwa/Documents/models/rag-query-rewriting/t5l-turbo-hotpot-0331",
                        help="RAG-QR T5 rewriter model path (local path)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ragqr_max_length", type=int, default=50)
    parser.add_argument("--ragqr_num_beams", type=int, default=4)
    parser.add_argument("--cache_dir", type=str, default="dataset/FollowIR_test/embeddings")
    parser.add_argument("--ragqr_cache_dir", type=str, default="dataset/FollowIR_test/ragqr_queries")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    evaluator = RAGQREvaluator(
        model_name=args.model_name,
        task_name=args.task_name,
        output_dir=args.output_dir,
        ragqr_model_path=args.ragqr_model_path,
        ragqr_max_length=args.ragqr_max_length,
        ragqr_num_beams=args.ragqr_num_beams,
        ragqr_cache_dir=args.ragqr_cache_dir,
        gpu_id=args.gpu_id,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
