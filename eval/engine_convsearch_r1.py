"""
ConvSearch-R1 Evaluation Engine for FollowIR

Reference: Zhu et al. "ConvSearch-R1: Enhancing Query Reformulation for Conversational
           Search with Reasoning via Reinforcement Learning" (EMNLP 2025)
Paper: https://arxiv.org/abs/2505.15776
Official code: https://github.com/BeastyZ/ConvSearch-R1
Model weights: https://huggingface.co/BeastyZ/Qwen2.5-3B-ConvSearch-R1-TopiOCQA

ConvSearch-R1 is a two-stage alignment framework for conversational query reformulation (CQR).
It uses GRPO (Group Relative Policy Optimization) to train a query rewriter without any
external supervised data (reference rewrite).

Key reproduction details:
    - Model: Qwen2.5-3B-ConvSearch-R1-TopiOCQA (fine-tuned via GRPO on TopiOCQA)
    - Prompt: Conversational context decontextualization with reasoning
    - Output format: <think reasoning> </think<rewrite> rewritten query </rewrite>
    - temperature=0.7 (official inference setting)
    - max_tokens=4096 (official inference setting)

Evaluation pipeline:
    - Phase 1: ConvSearch-R1 model rewrites queries (decontextualizes with reasoning)
    - Phase 2: RepLLaMA encodes rewritten queries + documents (same as other baselines)
    - Phase 3: Matrix multiplication for similarity (same as V2 engine)
    - Phase 4: FollowIR metrics computation (same as V2 engine)

Usage:
    cd /home/luwa/Documents/DSCLR && /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_convsearch_r1 \
        --task_name Core17InstructionRetrieval \
        --model_name samaya-ai/RepLLaMA-reproduced \
        --csr_model_path /home/luwa/Documents/models/Qwen2.5-3B-ConvSearch-R1-TopiOCQA \
        --device cuda \
        --output_dir results/convsearch_r1/Core17
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


# ConvSearch-R1 prompt template (from official README)
CONVSEARCH_R1_PROMPT_TEMPLATE = """Given a query and its context, you must first think about the reasoning process in the mind to decontextualize the query by resolving coreference and omission issues. Then, provide the user with a rewrite that retains its original meaning and is as informative as possible to help search engines retrieve relevant documents effectively. The reasoning process and rewrite should be enclosed within <thinkthink> and <rewrite> </rewrite> tags, respectively, i.e., \
<think reasoning process here </think<rewrite> rewrite here </rewrite>.

### Context Begin ###
{context}
### Context End ###

Query: {query}
Rewrite:"""


def extract_rewrite_from_response(response_text: str) -> str:
    """Extract rewritten query from ConvSearch-R1 response.

    ConvSearch-R1 outputs in the format:
        <think reasoning process </think<rewrite> rewritten query </rewrite>

    We extract the content within <rewrite>...</rewrite> tags.
    """
    # Try to find <rewrite>...</rewrite> pattern
    rewrite_pattern = r'<rewrite>(.*?)</rewrite>'
    matches = re.findall(rewrite_pattern, response_text, re.DOTALL)

    if matches:
        return matches[-1].strip()

    # Fallback: if <rewrite> tag exists but no closing tag
    if "<rewrite>" in response_text:
        after_rewrite = response_text.split("<rewrite>")[-1]
        # Try to find end of rewrite content
        for end_tag in ["</rewrite>", "<|im_end|>", "\n\n", "<think"]:
            if end_tag in after_rewrite:
                after_rewrite = after_rewrite.split(end_tag)[0]
        return after_rewrite.strip()

    # Last resort: return the full response (stripped)
    return response_text.strip()


class ConvSearchR1Promptor:
    """ConvSearch-R1 prompt builder for FollowIR queries.

    ConvSearch-R1 is designed for conversational query reformulation.
    For FollowIR, we adapt it by:
    - Treating the instruction as "context" for the query
    - The query itself is the user's current question
    - The model decontextualizes the query considering the instruction
    """

    def build_messages(self, query: str, instruction: str = "") -> List[Dict[str, str]]:
        """Build chat messages for ConvSearch-R1.

        FollowIR queries have:
        - query: the base query text
        - instruction: an instruction that modifies the search intent

        We format the instruction as "context" since ConvSearch-R1 expects
        conversational context to decontextualize the query.
        """
        if instruction:
            context = f"Q1: {instruction}"
        else:
            context = "(No additional context)"

        prompt = CONVSEARCH_R1_PROMPT_TEMPLATE.format(
            context=context,
            query=query,
        )

        return [
            {"role": "user", "content": prompt},
        ]


class ConvSearchR1Generator:
    """ConvSearch-R1 model for query rewriting.

    Official code uses vllm for inference:
        llm = LLM(model=model_name_or_path, tensor_parallel_size=1,
                   enforce_eager=False, gpu_memory_utilization=0.8, dtype='bfloat16')
        outputs = llm.chat(conv, sampling_params, add_generation_prompt=True)

    We use HuggingFace transformers for compatibility with the existing eval pipeline.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading ConvSearch-R1 model from {model_path}...")
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

        if effective_device == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float32, trust_remote_code=True
            ).to("cpu")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        self.model.eval()
        logger.info(f"ConvSearch-R1 model loaded on {effective_device}")

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate rewritten query from ConvSearch-R1.

        Uses the chat template from the model (Qwen2.5 format).
        temperature=0.7 as per official inference code.
        """
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        full_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        rewritten_query = extract_rewrite_from_response(full_response)
        return rewritten_query


class ConvSearchR1Evaluator(DSCLREvaluatorEngine):
    """ConvSearch-R1 evaluation engine for FollowIR

    Inherits from DSCLREvaluatorEngine to reuse the exact same evaluation pipeline:
    - Same encoder (RepLLaMA)
    - Same document indexing (L2 normalized embeddings)
    - Same score computation (matrix multiplication)
    - Same result extraction (_extract_results)
    - Same metrics computation (FollowIREvaluator)

    The ONLY difference is the query rewriting method:
    - V2: Uses Q_plus/Q_minus dual-track with reward-penalty formula
    - ConvSearch-R1: Uses GRPO-trained rewriter to decontextualize queries, then encodes with RepLLaMA
    """

    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_dir: str,
        csr_model_path: str,
        csr_max_new_tokens: int = 4096,
        csr_temperature: float = 0.7,
        csr_cache_dir: str = "dataset/FollowIR_test/convsearch_r1_queries",
        gpu_id: int = 0,
        **kwargs,
    ):
        self.csr_model_path = csr_model_path
        self.csr_max_new_tokens = csr_max_new_tokens
        self.csr_temperature = csr_temperature
        self.csr_cache_dir = csr_cache_dir
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

        os.makedirs(self.csr_cache_dir, exist_ok=True)

        logger.info("ConvSearch-R1 rewrite mode enabled")
        logger.info(f"  ConvSearch-R1 model: {self.csr_model_path}")
        logger.info(f"  Rewrite cache dir: {self.csr_cache_dir}")

    def _get_csr_cache_path(self) -> str:
        csr_name = os.path.basename(self.csr_model_path)
        return os.path.join(
            self.csr_cache_dir,
            f"{self.task_name}_convsearch_r1_{csr_name}.jsonl",
        )

    def _load_csr_cache(self) -> Dict[str, str]:
        cache_path = self._get_csr_cache_path()
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        cache[item["qid"]] = item["rewritten_query"]
            logger.info(f"Loaded ConvSearch-R1 cache: {len(cache)} entries from {cache_path}")
        return cache

    def _save_csr_cache(self, cache: Dict[str, str], queries_info: Dict[str, Dict]):
        cache_path = self._get_csr_cache_path()
        with open(cache_path, "w", encoding="utf-8") as f:
            for qid, rewritten_query in cache.items():
                info = queries_info.get(qid, {})
                f.write(json.dumps({
                    "qid": qid,
                    "query": info.get("query", ""),
                    "instruction": info.get("instruction", ""),
                    "rewritten_query": rewritten_query,
                }, ensure_ascii=False) + "\n")
        logger.info(f"ConvSearch-R1 cache saved: {cache_path}")

    def run(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("Starting ConvSearch-R1 Evaluation on FollowIR")
        logger.info(f"  Paper: Zhu et al. 2025 (EMNLP 2025, arXiv 2505.15776)")
        logger.info(f"  Official code: https://github.com/BeastyZ/ConvSearch-R1")
        logger.info(f"  Rewriter model: {self.csr_model_path}")
        logger.info(f"  Encoder: {self.model_name} (same as V2 engine)")
        logger.info(f"  temperature={self.csr_temperature} (official inference setting)")
        logger.info(f"  max_new_tokens={self.csr_max_new_tokens}")
        logger.info("=" * 60)

        start_time = time.time()

        corpus, q_og, q_changed, candidates = self.data_loader.load()
        q_raw_og, q_raw_changed = self.data_loader.load_raw_queries()

        csr_cache = self._load_csr_cache()
        queries_info = {}

        query_ids_og: List[str] = []
        query_ids_changed: List[str] = []
        queries_to_generate: List[Tuple[str, str, str]] = []

        for qid in q_og.keys():
            query_ids_og.append(qid)
            raw = q_raw_og.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            queries_info[qid] = {"query": query_text, "instruction": instruction}
            if qid not in csr_cache:
                queries_to_generate.append((qid, query_text, instruction))

        for qid in q_changed.keys():
            query_ids_changed.append(qid)
            raw = q_raw_changed.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            queries_info[qid] = {"query": query_text, "instruction": instruction}
            if qid not in csr_cache:
                queries_to_generate.append((qid, query_text, instruction))

        if queries_to_generate:
            logger.info(f"Phase 1: ConvSearch-R1 rewriting {len(queries_to_generate)} queries...")
            promptor = ConvSearchR1Promptor()
            generator = ConvSearchR1Generator(
                model_path=self.csr_model_path,
                device=f"cuda:{self.gpu_id}",
                max_new_tokens=self.csr_max_new_tokens,
                temperature=self.csr_temperature,
            )

            for qid, query_text, instruction in tqdm(queries_to_generate, desc="ConvSearch-R1 generation"):
                messages = promptor.build_messages(query_text, instruction)
                rewritten_query = generator.generate(messages)
                csr_cache[qid] = rewritten_query

            self._save_csr_cache(csr_cache, queries_info)
            del generator
            torch.cuda.empty_cache()
            logger.info("Rewritten queries generated and cached, generator unloaded")
        else:
            logger.info(f"All rewritten queries already cached ({len(csr_cache)} entries)")

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

        og_rewritten_queries = [csr_cache[qid] for qid in query_ids_og]
        changed_rewritten_queries = [csr_cache[qid] for qid in query_ids_changed]

        logger.info("Encoding OG rewritten queries with RepLLaMA...")
        q_emb_og = self._encode_queries(og_rewritten_queries)

        logger.info("Encoding Changed rewritten queries with RepLLaMA...")
        q_emb_changed = self._encode_queries(changed_rewritten_queries)

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
        logger.info("ConvSearch-R1 Evaluation Results (Official Reproduction)")
        logger.info(f"  Task: {self.task_name}")
        logger.info(f"  Rewriter: {self.csr_model_path}")
        logger.info(f"  Encoder: {self.model_name}")
        logger.info(f"  p-MRR: {p_mrr:.4f}")
        logger.info(f"  OG MAP@1000: {og_map:.4f}")
        logger.info(f"  Changed MAP@1000: {changed_map:.4f}")
        logger.info(f"  OG nDCG@5: {og_ndcg5:.4f}")
        logger.info(f"  Changed nDCG@5: {changed_ndcg5:.4f}")
        logger.info(f"  Elapsed: {elapsed:.1f}s")
        logger.info("=" * 60)

        self._save_results(
            metrics=metrics,
            results_og=results_og,
            results_changed=results_changed,
            csr_cache=csr_cache,
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
        csr_cache: Dict[str, str],
        query_ids_og: List[str],
        query_ids_changed: List[str],
        elapsed: float,
    ):
        os.makedirs(self.output_dir, exist_ok=True)

        summary = {
            "task": self.task_name,
            "model": self.model_name,
            "mode": "ConvSearch-R1 (official reproduction from Zhu et al. 2025, EMNLP 2025)",
            "csr_model": self.csr_model_path,
            "csr_temperature": self.csr_temperature,
            "csr_max_new_tokens": self.csr_max_new_tokens,
            "csr_paper": "https://arxiv.org/abs/2505.15776",
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
            json.dump(summary, f, indent=2, ensure_ascii=False, default=lambda o: float(o))

        out_og = os.path.join(self.output_dir, "ranking_og.json")
        out_changed = os.path.join(self.output_dir, "ranking_changed.json")
        with open(out_og, "w", encoding="utf-8") as f:
            json.dump(results_og, f, ensure_ascii=False)
        with open(out_changed, "w", encoding="utf-8") as f:
            json.dump(results_changed, f, ensure_ascii=False)

        rewrite_samples_path = os.path.join(self.output_dir, "convsearch_r1_samples.json")
        samples = {}
        for qid in query_ids_og[:5]:
            samples[qid] = {
                "type": "og",
                "rewritten_query": csr_cache.get(qid, "")[:500],
            }
        for qid in query_ids_changed[:5]:
            samples[qid] = {
                "type": "changed",
                "rewritten_query": csr_cache.get(qid, "")[:500],
            }
        with open(rewrite_samples_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="ConvSearch-R1 Evaluation on FollowIR")
    parser.add_argument("--task_name", type=str, required=True,
                        choices=["Core17InstructionRetrieval", "Robust04InstructionRetrieval", "News21InstructionRetrieval"])
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--csr_model_path", type=str,
                        default="/home/luwa/Documents/models/Qwen2.5-3B-ConvSearch-R1-TopiOCQA",
                        help="ConvSearch-R1 model path (HuggingFace model ID or local path)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--csr_max_new_tokens", type=int, default=4096)
    parser.add_argument("--csr_temperature", type=float, default=0.7)
    parser.add_argument("--cache_dir", type=str, default="dataset/FollowIR_test/embeddings")
    parser.add_argument("--csr_cache_dir", type=str, default="dataset/FollowIR_test/convsearch_r1_queries")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    evaluator = ConvSearchR1Evaluator(
        model_name=args.model_name,
        task_name=args.task_name,
        output_dir=args.output_dir,
        csr_model_path=args.csr_model_path,
        csr_max_new_tokens=args.csr_max_new_tokens,
        csr_temperature=args.csr_temperature,
        csr_cache_dir=args.csr_cache_dir,
        gpu_id=args.gpu_id,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )

    result = evaluator.run()
    metrics = result["metrics"]

    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS: ConvSearch-R1 on FollowIR")
    logger.info("=" * 80)
    logger.info(f"  Task: {args.task_name}")
    logger.info(f"  p-MRR: {metrics.get('p-MRR', 0):.4f}")
    logger.info(f"  OG MAP@1000: {metrics.get('original', {}).get('map_at_1000', 0):.4f}")
    logger.info(f"  Changed MAP@1000: {metrics.get('changed', {}).get('map_at_1000', 0):.4f}")
    logger.info(f"  OG nDCG@5: {metrics.get('original', {}).get('ndcg_at_5', 0):.4f}")
    logger.info(f"  Changed nDCG@5: {metrics.get('changed', {}).get('ndcg_at_5', 0):.4f}")


if __name__ == "__main__":
    main()
