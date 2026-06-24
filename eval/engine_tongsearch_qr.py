"""
TongSearch-QR Evaluation Engine for FollowIR

Reference: TongSearch-QR: Query Reasoning for Retrieval via Reinforcement Learning
Official code: https://github.com/bigai-nlco/TongSearch-QR
Model weights: https://huggingface.co/TongSearch/TongSearch-QR-3B
              https://huggingface.co/TongSearch/TongSearch-QR-7B

Faithfully reproduced from the official code:
    - Model: TongSearch-QR-3B (Qwen2.5-3B-Instruct fine-tuned via GRPO)
             TongSearch-QR-7B (Qwen2.5-7B-Instruct fine-tuned via GRPO)
    - Prompt: "Instructions:\\n1. Identify the essential problem.\\n2. Think step by step..."
    - System prompt (think mode): reasoning in <think /> tags, answer in <answer /> tags
    - Sampling: max_tokens=500, top_p=0.8, top_k=20, temperature=0.7, repetition_penalty=1.05
    - Output: Extract <answer> content, then use original_query + reasoned_content for retrieval
    - Encoder: RepLLaMA (same as other baselines)

Evaluation pipeline:
    - Phase 1: TongSearch-QR model reasons about queries (generates reasoning + expanded query)
    - Phase 2: RepLLaMA encodes reasoned queries + documents (same as V2 engine)
    - Phase 3: Matrix multiplication for similarity (same as V2 engine)
    - Phase 4: FollowIR metrics computation (same as V2 engine)

Usage:
    cd /home/luwa/Documents/DSCLR && /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_tongsearch_qr \
        --task_name Core17InstructionRetrieval \
        --model_name samaya-ai/RepLLaMA-reproduced \
        --qr_model_path TongSearch/TongSearch-QR-3B \
        --device cuda \
        --output_dir results/tongsearch_qr/Core17_3B
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
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from eval.engine_dscrl import DSCLREvaluatorEngine, load_cached_embeddings, save_embeddings_cache
from eval.metrics import FollowIREvaluator

logger = logging.getLogger(__name__)


# TongSearch-QR official prompt from scripts/evaluation/question_rewriting.py
TONGSEARCH_PROMPT_TEMPLATE = (
    "Instructions:\n"
    "1. Identify the essential problem.\n"
    "2. Think step by step to reason and describe what information could be relevant "
    "and helpful to address the questions in detail.\n"
    "3. Draft an answer with as many thoughts as you have.\n"
    "Query: {query}\n\n"
)

# System prompt for think mode (from official vllm_model.py)
TONGSEARCH_THINK_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. "
    "The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind "
    "and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think > </think > and <answer> </answer> tags, "
    "i.e., <think > reasoning process here </think > <answer> answer here </answer>."
)


def extract_answer_from_response(response_text: str) -> str:
    """Extract content from <answer> tags (from official vllm_model.py _parse_think_output)

    Official pattern: re.compile(r"<think >.*?</think >\\s*<answer>(.*?)</answer>", re.DOTALL)
    """
    think_pattern = re.compile(r"<think >.*?</think >\s*<answer>(.*?)</answer>", re.DOTALL)
    match = think_pattern.search(response_text)
    if match:
        return match.group(1).strip()

    # Fallback: just look for <answer> tags
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    match = answer_pattern.search(response_text)
    if match:
        return match.group(1).strip()

    # No tags found, return raw text
    return response_text.strip()


class TongSearchQRGenerator:
    """TongSearch-QR model for query reasoning

    Official code uses vllm for inference. We use HuggingFace transformers as fallback
    since vllm may not be available.

    Official sampling params from vllm_model.py:
        max_tokens=500, top_p=0.8, top_k=20, temperature=0.7,
        repetition_penalty=1.05, stop_token_ids=[151645, 151643]
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 500,
        use_think: bool = True,
    ):
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.use_think = use_think
        self.system_prompt = TONGSEARCH_THINK_SYSTEM_PROMPT if use_think else ""

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading TongSearch-QR model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        effective_device = device
        if device.startswith("cuda"):
            try:
                torch.cuda._lazy_init()
                if not torch.cuda.is_available():
                    effective_device = "cpu"
            except Exception:
                effective_device = "cpu"

        # Always load QR model to CPU to avoid GPU memory fragmentation
        # that can cause OOM when loading RepLLaMA later
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        if effective_device != "cpu":
            self.model = self.model.to(device)
        self.model.eval()
        logger.info(f"TongSearch-QR model loaded on {effective_device}")

    def generate(self, query: str) -> str:
        """Generate reasoned query content.

        Returns the extracted answer content (from <answer> tags if use_think=True).
        The caller should combine original_query + reasoned_content for retrieval.
        """
        user_content = TONGSEARCH_PROMPT_TEMPLATE.format(query=query)

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Official sampling params: temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.05,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        full_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if self.use_think:
            return extract_answer_from_response(full_response)
        return full_response.strip()


class TongSearchQREvaluator(DSCLREvaluatorEngine):
    """TongSearch-QR evaluation engine for FollowIR

    Inherits from DSCLREvaluatorEngine to reuse the exact same evaluation pipeline:
    - Same encoder (RepLLaMA)
    - Same document indexing (L2 normalized embeddings)
    - Same score computation (matrix multiplication)
    - Same result extraction (_extract_results)
    - Same metrics computation (FollowIREvaluator)

    The ONLY difference is the query rewriting method:
    - V2: Uses Q_plus/Q_minus dual-track with reward-penalty formula
    - TongSearch-QR: Uses GRPO-trained rewriter to reason about queries, then encodes with RepLLaMA
    """

    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_dir: str,
        qr_model_path: str,
        qr_max_new_tokens: int = 500,
        qr_use_think: bool = True,
        qr_cache_dir: str = "dataset/FollowIR_test/tongsearch_qr_queries",
        gpu_id: int = 0,
        **kwargs,
    ):
        self.qr_model_path = qr_model_path
        self.qr_max_new_tokens = qr_max_new_tokens
        self.qr_use_think = qr_use_think
        self.qr_cache_dir = qr_cache_dir
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

        os.makedirs(self.qr_cache_dir, exist_ok=True)

        logger.info("TongSearch-QR reasoning mode enabled")
        logger.info(f"QR model: {self.qr_model_path}")
        logger.info(f"Think mode: {self.qr_use_think}")
        logger.info(f"Cache dir: {self.qr_cache_dir}")

    def _get_qr_cache_path(self) -> str:
        qr_name = self.qr_model_path.replace("/", "_")
        return os.path.join(
            self.qr_cache_dir,
            f"{self.task_name}_tongsearch_qr_{qr_name}.jsonl",
        )

    def _load_qr_cache(self) -> Dict[str, str]:
        cache_path = self._get_qr_cache_path()
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        cache[item["qid"]] = item["reasoned_query"]
            logger.info(f"Loaded TongSearch-QR cache: {len(cache)} entries from {cache_path}")
        return cache

    def _save_qr_cache(self, cache: Dict[str, str], queries_info: Dict[str, Dict]):
        cache_path = self._get_qr_cache_path()
        with open(cache_path, "w", encoding="utf-8") as f:
            for qid, reasoned_query in cache.items():
                info = queries_info.get(qid, {})
                f.write(json.dumps({
                    "qid": qid,
                    "query": info.get("query", ""),
                    "instruction": info.get("instruction", ""),
                    "reasoned_query": reasoned_query,
                }, ensure_ascii=False) + "\n")
        logger.info(f"TongSearch-QR cache saved: {cache_path}")

    def run(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("Starting TongSearch-QR Evaluation on FollowIR")
        logger.info(f"Official code: https://github.com/bigai-nlco/TongSearch-QR")
        logger.info(f"Rewriter model: {self.qr_model_path}")
        logger.info(f"Encoder: {self.model_name} (same as V2 engine)")
        logger.info(f"Think mode: {self.qr_use_think}")
        logger.info(f"Sampling: temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05")
        logger.info("=" * 60)

        start_time = time.time()

        corpus, q_og, q_changed, candidates = self.data_loader.load()
        q_raw_og, q_raw_changed = self.data_loader.load_raw_queries()

        qr_cache = self._load_qr_cache()
        queries_info = {}

        query_ids_og: List[str] = []
        query_ids_changed: List[str] = []
        queries_to_generate: List[Tuple[str, str, str]] = []

        for qid in q_og.keys():
            query_ids_og.append(qid)
            raw = q_raw_og.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            queries_info[qid] = {"query": query_text, "instruction": instruction}
            if qid not in qr_cache:
                queries_to_generate.append((qid, query_text, instruction))

        for qid in q_changed.keys():
            query_ids_changed.append(qid)
            raw = q_raw_changed.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            queries_info[qid] = {"query": query_text, "instruction": instruction}
            if qid not in qr_cache:
                queries_to_generate.append((qid, query_text, instruction))

        if queries_to_generate:
            logger.info(f"Phase 1: TongSearch-QR reasoning {len(queries_to_generate)} queries...")
            generator = TongSearchQRGenerator(
                model_path=self.qr_model_path,
                device=f"cuda:{self.gpu_id}",
                max_new_tokens=self.qr_max_new_tokens,
                use_think=self.qr_use_think,
            )

            for qid, query_text, instruction in tqdm(queries_to_generate, desc="TongSearch-QR generation"):
                # TongSearch-QR reasons about the query text
                # The instruction is appended to the reasoned query for RepLLaMA encoding
                reasoned_content = generator.generate(query_text)
                # Combine: original query + reasoned content + instruction
                # This follows the same pattern as other rewriters in our framework
                if instruction:
                    reasoned_query = f"{query_text} {reasoned_content} {instruction}"
                else:
                    reasoned_query = f"{query_text} {reasoned_content}"
                qr_cache[qid] = reasoned_query

            self._save_qr_cache(qr_cache, queries_info)
            # Thoroughly release QR model memory
            del generator.model
            del generator.tokenizer
            del generator
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("Reasoned queries generated and cached, generator unloaded")
        else:
            logger.info(f"All reasoned queries already cached ({len(qr_cache)} entries)")

        logger.info("Phase 2: Loading encoder and computing retrieval (same pipeline as V2)...")
        all_doc_ids = self._get_all_candidate_doc_ids(candidates)

        cached_data = None
        if self.use_cache:
            cached_data = load_cached_embeddings(self.cache_dir, self.task_name, self.model_name)

        if cached_data is not None:
            cached_embeddings, cached_doc_ids = cached_data
            if set(cached_doc_ids) == set(all_doc_ids):
                logger.info(f"Using cached document embeddings ({len(cached_doc_ids)} docs)")
                self.retriever.set_embeddings(cached_embeddings, cached_doc_ids)
            else:
                logger.warning("Cached doc IDs mismatch, re-encoding...")
                doc_texts = [corpus[did]["text"] for did in all_doc_ids]
                self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)
        else:
            logger.info("Encoding candidate documents...")
            doc_texts = [corpus[did]["text"] for did in all_doc_ids]
            self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)

        og_reasoned_queries = [qr_cache[qid] for qid in query_ids_og]
        changed_reasoned_queries = [qr_cache[qid] for qid in query_ids_changed]

        logger.info("Encoding OG reasoned queries with RepLLaMA...")
        q_emb_og = self._encode_queries(og_reasoned_queries)

        logger.info("Encoding Changed reasoned queries with RepLLaMA...")
        q_emb_changed = self._encode_queries(changed_reasoned_queries)

        device = self.retriever.doc_embeddings.device
        q_emb_og = q_emb_og.to(device)
        q_emb_changed = q_emb_changed.to(device)

        logger.info("Computing similarity scores (matrix multiplication, same as V2)...")
        S_og = torch.matmul(q_emb_og, self.retriever.doc_embeddings.T)
        S_changed = torch.matmul(q_emb_changed, self.retriever.doc_embeddings.T)

        logger.info("Extracting results and computing FollowIR metrics...")
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
        logger.info("TongSearch-QR Evaluation Results (Official Reproduction)")
        logger.info(f"   Task: {self.task_name}")
        logger.info(f"   Rewriter: {self.qr_model_path}")
        logger.info(f"   Encoder: {self.model_name}")
        logger.info(f"   Think mode: {self.qr_use_think}")
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
            qr_cache=qr_cache,
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
        qr_cache: Dict[str, str],
        query_ids_og: List[str],
        query_ids_changed: List[str],
        elapsed: float,
    ):
        os.makedirs(self.output_dir, exist_ok=True)

        summary = {
            "task": self.task_name,
            "model": self.model_name,
            "mode": "TongSearch-QR (official reproduction)",
            "qr_model": self.qr_model_path,
            "qr_use_think": self.qr_use_think,
            "qr_max_new_tokens": self.qr_max_new_tokens,
            "qr_sampling": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.05,
            },
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

        rewrite_samples_path = os.path.join(self.output_dir, "tongsearch_qr_samples.json")
        samples = {}
        for qid in query_ids_og[:5]:
            samples[qid] = {
                "type": "og",
                "reasoned_query": qr_cache.get(qid, "")[:500],
            }
        for qid in query_ids_changed[:5]:
            samples[qid] = {
                "type": "changed",
                "reasoned_query": qr_cache.get(qid, "")[:500],
            }
        with open(rewrite_samples_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="TongSearch-QR Evaluation on FollowIR")
    parser.add_argument("--task_name", type=str, required=True,
                        choices=["Core17InstructionRetrieval", "Robust04InstructionRetrieval", "News21InstructionRetrieval"])
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--qr_model_path", type=str, required=True,
                        help="TongSearch-QR model path (HuggingFace model ID or local path)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--qr_max_new_tokens", type=int, default=500)
    parser.add_argument("--qr_use_think", action="store_true", default=True,
                        help="Enable think mode (default: True)")
    parser.add_argument("--no_qr_use_think", action="store_true",
                        help="Disable think mode")
    parser.add_argument("--cache_dir", type=str, default="dataset/FollowIR_test/embeddings")
    parser.add_argument("--qr_cache_dir", type=str, default="dataset/FollowIR_test/tongsearch_qr_queries")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    use_think = not args.no_qr_use_think

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    evaluator = TongSearchQREvaluator(
        model_name=args.model_name,
        task_name=args.task_name,
        output_dir=args.output_dir,
        qr_model_path=args.qr_model_path,
        qr_max_new_tokens=args.qr_max_new_tokens,
        qr_use_think=use_think,
        qr_cache_dir=args.qr_cache_dir,
        gpu_id=args.gpu_id,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
