"""
DeepRetrieval Evaluation Engine for FollowIR

Reference: Jiang et al. "DeepRetrieval: Hacking Real Search Engines and Retrievers
           with Large Language Models via Reinforcement Learning" (2025)
Paper: https://arxiv.org/abs/2503.00223
Official code: https://github.com/pat-jj/DeepRetrieval
Model weights: https://huggingface.co/DeepRetrieval/DeepRetrieval-NQ-BM25-3B

Faithfully reproduced from the official code:
    - Model: DeepRetrieval/DeepRetrieval-NQ-BM25-3B (Qwen2.5-3B-Instruct fine-tuned via RL)
    - Prompt: Qwen2.5 chat format with <|im_start|>/<|im_end|> tokens
    - Instruction: Dense retrieval instruction from data_preprocess/dense/*.py
    - Reasoning: Model generates 思考和 reasoning before <answer> JSON
    - Output: {"query": "expanded query text"} within <answer> tags
    - Extraction: Parse <answer> tags, then JSON parse the "query" field
    - do_sample=False for deterministic generation (official eval code uses do_sample=False)

Evaluation pipeline:
    - Phase 1: DeepRetrieval model rewrites queries (replaces V2's Q_plus/Q_minus dual-track)
    - Phase 2: RepLLaMA encodes rewritten queries + documents (same as V2 engine)
    - Phase 3: Matrix multiplication for similarity (same as V2 engine)
    - Phase 4: FollowIR metrics computation (same as V2 engine)

Usage:
    cd /home/luwa/Documents/DSCLR && /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_deepretrieval \
        --task_name Core17InstructionRetrieval \
        --model_name samaya-ai/RepLLaMA-reproduced \
        --dr_model_path DeepRetrieval/DeepRetrieval-NQ-BM25-3B \
        --device cuda \
        --output_dir results/deepretrieval/Core17
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


DEEPRETRIEVAL_DENSE_INSTRUCTION = (
    "You are an expert in generating queries for dense retrieval. "
    "Given a question, your task is to retain the original query while expanding it "
    "with additional semantically relevant information, to retrieve relevant documents "
    "that best answer the question. If no useful expansion is needed, return the "
    "original query as is."
)

DEEPRETRIEVAL_DENSE_INSTRUCTION_WITH_INSTRUCTION = (
    "You are an expert in generating queries for dense retrieval. "
    "Given a question and an instruction, your task is to retain the original query "
    "while expanding it with additional semantically relevant information and following "
    "the instruction, to retrieve relevant documents that best answer the question. "
    "If no useful expansion is needed, return the original query as is."
)


class DeepRetrievalPromptor:
    """Faithful reproduction of DeepRetrieval prompting from official code

    Official code (data_preprocess/dense/nfcorpus.py, data_preprocess/dense/msmarco_beir.py):
        input_str = <|im_start|>system\\nYou are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|im_end|>
        <|im_start|>user\\n{INSTRUCTION}
        Show your work in 思考和  tags. Your final response must be in JSON format within <answer> </answer> tags. For example,
        <answer>
        {
            "query": "...."
        }
        </answer>.
        Here's the question:
        {query}
        Assistant: Let me think step by step.
    """

    def build_messages(self, query: str, instruction: str = "") -> List[Dict[str, str]]:
        system_msg = "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer."

        if instruction:
            dr_instruction = DEEPRETRIEVAL_DENSE_INSTRUCTION_WITH_INSTRUCTION
        else:
            dr_instruction = DEEPRETRIEVAL_DENSE_INSTRUCTION

        user_content = dr_instruction + "\n\n"
        user_content += 'Show your work in 思考和  tags. Your final response must be in JSON format within <answer> </answer> tags. For example,\n'
        user_content += '<answer>\n{\n    "query": "...."\n} \n</answer>.\n\n'
        user_content += f"Here's the question:\n{query}\n"
        if instruction:
            user_content += f"Here's the instruction:\n{instruction}\n"

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]


def extract_query_from_response(response_text: str) -> str:
    """Faithful extraction from official code (src/eval/Dense/baselines/model_generate/model_generate.py)

    Official extraction logic:
        1. Split by "\\nAssistant:" to get the assistant's response
        2. Find all <answer>...</answer> matches
        3. If matches found, try json.loads and extract 'query' key
        4. If JSON parsing fails, return the raw match content
    """
    try:
        if "\nAssistant:" in response_text:
            response_text = response_text.split("\nAssistant:")[1]
        elif "\nassistant:" in response_text:
            response_text = response_text.split("\nassistant:")[1]
    except Exception:
        pass

    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, response_text, re.DOTALL)

    if matches:
        last_match = matches[-1].strip()
        try:
            answer_json = json.loads(last_match)
            return answer_json.get('query', last_match)
        except json.JSONDecodeError:
            return last_match

    if "<answer>" in response_text:
        after_answer = response_text.split("<answer>")[-1]
        after_answer = after_answer.split("</answer>")[0] if "</answer>" in after_answer else after_answer
        try:
            answer_json = json.loads(after_answer.strip())
            return answer_json.get('query', after_answer.strip())
        except json.JSONDecodeError:
            return after_answer.strip()

    return response_text.strip()


class DeepRetrievalGenerator:
    """DeepRetrieval model for query rewriting

    Official code loads the model with:
        AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2",
                                              torch_dtype=torch.bfloat16, device_map='auto')
        model.generate(max_new_tokens=1024, do_sample=False)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 1024,
    ):
        self.max_new_tokens = max_new_tokens
        self.device = device

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"📥 Loading DeepRetrieval model from {model_path}...")
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
        logger.info(f"✅ DeepRetrieval model loaded on {effective_device}")

    def generate(self, messages: List[Dict[str, str]]) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        full_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        rewritten_query = extract_query_from_response(full_response)
        return rewritten_query


class DeepRetrievalEvaluator(DSCLREvaluatorEngine):
    """DeepRetrieval evaluation engine for FollowIR

    Inherits from DSCLREvaluatorEngine to reuse the exact same evaluation pipeline:
    - Same encoder (RepLLaMA)
    - Same document indexing (L2 normalized embeddings)
    - Same score computation (matrix multiplication)
    - Same result extraction (_extract_results)
    - Same metrics computation (FollowIREvaluator)

    The ONLY difference is the query rewriting method:
    - V2: Uses Q_plus/Q_minus dual-track with reward-penalty formula
    - DeepRetrieval: Uses DeepRetrieval model to rewrite queries, then encodes with RepLLaMA
    """

    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_dir: str,
        dr_model_path: str,
        dr_max_new_tokens: int = 1024,
        dr_cache_dir: str = "dataset/FollowIR_test/deepretrieval_queries",
        gpu_id: int = 0,
        **kwargs,
    ):
        self.dr_model_path = dr_model_path
        self.dr_max_new_tokens = dr_max_new_tokens
        self.dr_cache_dir = dr_cache_dir
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

        os.makedirs(self.dr_cache_dir, exist_ok=True)

        logger.info("🏛️ DeepRetrieval 改写模式已启用")
        logger.info(f"📁 DeepRetrieval 模型: {self.dr_model_path}")
        logger.info(f"📁 改写缓存目录: {self.dr_cache_dir}")

    def _get_dr_cache_path(self) -> str:
        dr_name = self.dr_model_path.replace("/", "_")
        return os.path.join(
            self.dr_cache_dir,
            f"{self.task_name}_deepretrieval_{dr_name}.jsonl",
        )

    def _load_dr_cache(self) -> Dict[str, str]:
        cache_path = self._get_dr_cache_path()
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        cache[item["qid"]] = item["rewritten_query"]
            logger.info(f"📂 Loaded DeepRetrieval cache: {len(cache)} entries from {cache_path}")
        return cache

    def _save_dr_cache(self, cache: Dict[str, str], queries_info: Dict[str, Dict]):
        cache_path = self._get_dr_cache_path()
        with open(cache_path, "w", encoding="utf-8") as f:
            for qid, rewritten_query in cache.items():
                info = queries_info.get(qid, {})
                f.write(json.dumps({
                    "qid": qid,
                    "query": info.get("query", ""),
                    "instruction": info.get("instruction", ""),
                    "rewritten_query": rewritten_query,
                }, ensure_ascii=False) + "\n")
        logger.info(f"💾 DeepRetrieval cache saved: {cache_path}")

    def run(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("🚀 Starting DeepRetrieval Evaluation on FollowIR")
        logger.info(f"   Paper: Jiang et al. 2025 (arXiv 2503.00223)")
        logger.info(f"   Official code: https://github.com/pat-jj/DeepRetrieval")
        logger.info(f"   Rewriter model: {self.dr_model_path}")
        logger.info(f"   Encoder: {self.model_name} (same as V2 engine)")
        logger.info(f"   Instruction: dense retrieval (query expansion)")
        logger.info(f"   do_sample=False (official eval setting)")
        logger.info("=" * 60)

        start_time = time.time()

        corpus, q_og, q_changed, candidates = self.data_loader.load()
        q_raw_og, q_raw_changed = self.data_loader.load_raw_queries()

        dr_cache = self._load_dr_cache()
        queries_info = {}

        query_ids_og: List[str] = []
        query_ids_changed: List[str] = []
        queries_to_generate: List[Tuple[str, str, str]] = []

        for qid in q_og.keys():
            query_ids_og.append(qid)
            raw = q_raw_og.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            queries_info[qid] = {"query": query_text, "instruction": instruction}
            if qid not in dr_cache:
                queries_to_generate.append((qid, query_text, instruction))

        for qid in q_changed.keys():
            query_ids_changed.append(qid)
            raw = q_raw_changed.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            queries_info[qid] = {"query": query_text, "instruction": instruction}
            if qid not in dr_cache:
                queries_to_generate.append((qid, query_text, instruction))

        if queries_to_generate:
            logger.info(f"🔄 Phase 1: DeepRetrieval rewriting {len(queries_to_generate)} queries...")
            promptor = DeepRetrievalPromptor()
            generator = DeepRetrievalGenerator(
                model_path=self.dr_model_path,
                device=f"cuda:{self.gpu_id}",
                max_new_tokens=self.dr_max_new_tokens,
            )

            for qid, query_text, instruction in tqdm(queries_to_generate, desc="DeepRetrieval generation"):
                messages = promptor.build_messages(query_text, instruction)
                rewritten_query = generator.generate(messages)
                dr_cache[qid] = rewritten_query

            self._save_dr_cache(dr_cache, queries_info)
            del generator
            torch.cuda.empty_cache()
            logger.info("✅ Rewritten queries generated and cached, generator unloaded")
        else:
            logger.info(f"✅ All rewritten queries already cached ({len(dr_cache)} entries)")

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

        og_rewritten_queries = [dr_cache[qid] for qid in query_ids_og]
        changed_rewritten_queries = [dr_cache[qid] for qid in query_ids_changed]

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
        logger.info("📊 DeepRetrieval Evaluation Results (Official Reproduction)")
        logger.info(f"   Task: {self.task_name}")
        logger.info(f"   Rewriter: {self.dr_model_path}")
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
            dr_cache=dr_cache,
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
        dr_cache: Dict[str, str],
        query_ids_og: List[str],
        query_ids_changed: List[str],
        elapsed: float,
    ):
        os.makedirs(self.output_dir, exist_ok=True)

        summary = {
            "task": self.task_name,
            "model": self.model_name,
            "mode": "DeepRetrieval (official reproduction from Jiang et al. 2025)",
            "dr_model": self.dr_model_path,
            "dr_instruction": "dense retrieval query expansion",
            "dr_max_new_tokens": self.dr_max_new_tokens,
            "dr_do_sample": False,
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

        rewrite_samples_path = os.path.join(self.output_dir, "deepretrieval_samples.json")
        samples = {}
        for qid in query_ids_og[:5]:
            samples[qid] = {
                "type": "og",
                "rewritten_query": dr_cache.get(qid, "")[:500],
            }
        for qid in query_ids_changed[:5]:
            samples[qid] = {
                "type": "changed",
                "rewritten_query": dr_cache.get(qid, "")[:500],
            }
        with open(rewrite_samples_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DeepRetrieval Evaluation on FollowIR")
    parser.add_argument("--task_name", type=str, required=True,
                        choices=["Core17InstructionRetrieval", "Robust04InstructionRetrieval", "News21InstructionRetrieval"])
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--dr_model_path", type=str, default="DeepRetrieval/DeepRetrieval-NQ-BM25-3B",
                        help="DeepRetrieval model path (HuggingFace model ID or local path)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dr_max_new_tokens", type=int, default=1024)
    parser.add_argument("--cache_dir", type=str, default="dataset/FollowIR_test/embeddings")
    parser.add_argument("--dr_cache_dir", type=str, default="dataset/FollowIR_test/deepretrieval_queries")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    evaluator = DeepRetrievalEvaluator(
        model_name=args.model_name,
        task_name=args.task_name,
        output_dir=args.output_dir,
        dr_model_path=args.dr_model_path,
        dr_max_new_tokens=args.dr_max_new_tokens,
        dr_cache_dir=args.dr_cache_dir,
        gpu_id=args.gpu_id,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
