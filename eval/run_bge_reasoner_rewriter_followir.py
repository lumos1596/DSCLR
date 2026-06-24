"""
BGE-Reasoner-Rewriter (ReasonEmbed) + BGE FollowIR 评测引擎

评估 BGE-Reasoner-Rewriter 的查询改写效果，使用 BGE-large-en 作为编码器。
BGE-Reasoner-Rewriter 基于 Qwen2.5-7B-Instruct，生成 5 个改写查询。

参考论文: ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval
参考框架: BGE-Reasoner (https://github.com/hanhainebula/FlagEmbedding/tree/research/BGE_Reasoner)

BGE-Reasoner 官方流程:
    - Query Rewrite: BGE-Reasoner-Rewriter generates 5 rewritten queries for each original query
    - Retrieval: For each rewritten query, BGE-Reasoner-Embed retrieves top-2000 documents
    - Aggregation: Sum the scores across the 5 rewrites to produce a final score

用法:
    cd /home/luwa/Documents/DSCLR && /home/luwa/.conda/envs/dsclr/bin/python -m eval.run_bge_reasoner_rewriter_followir \
        --task_name Core17InstructionRetrieval \
        --device cuda:1

    cd /home/luwa/Documents/DSCLR && /home/luwa/.conda/envs/dsclr/bin/python -m eval.run_bge_reasoner_rewriter_followir \
        --task_name Robust04InstructionRetrieval \
        --device cuda:2
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
import gc
from datetime import datetime
from typing import Dict, List, Any

import torch

from eval.engine_dscrl import DSCLREvaluatorEngine, load_cached_embeddings
from eval.metrics import FollowIREvaluator

logger = logging.getLogger(__name__)

REWRITER_PATH = "/home/luwa/Documents/models/reasoner-rewriter-qwen2.5-7b-0821"
NUM_REWRITES = 5  # BGE-Reasoner-Rewriter generates 5 rewritten queries per query


class BGEReasonerRewriter:
    """BGE-Reasoner-Rewriter - 基于 Qwen2.5-7B-Instruct 的查询改写模型

    官方说明: BGE-Reasoner-Rewriter generates 5 rewritten queries for each original query.
    官方 prompt 模板来自: https://huggingface.co/cfli/reasoner-rewriter-qwen2.5-7b-0821
    输出格式: <think >reasoning</think > <response>rewritten query</response>
    """

    # Official prompt template from model README
    PROMPT_TEMPLATE = (
        "Given a task and an input, first analyze the task and the input within the `<think >` and `</think >` tags. "
        "In your analysis:\n"
        "- Break down the requirements of the task\n"
        "- Identify key components from the input\n"
        "- Think step by step to reason about what should be included in the output\n\n"
        "Then, within the `<response>` and `</response>` tags, present the complete long output.\n\n"
        "## Task\n{task}\n\n## Input\n{query}"
    )

    # Default task description for general query rewriting
    DEFAULT_TASK = "Generate information that is relevant and helpful to address the questions in detail."

    def __init__(self, model_path: str = REWRITER_PATH, batch_size: int = 1, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading BGE-Reasoner-Rewriter from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load to CPU first to avoid GPU memory fragmentation
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",  # Use PyTorch native SDPA for faster inference
        )
        effective_device = device
        if device.startswith("cuda"):
            try:
                torch.cuda._lazy_init()
                if not torch.cuda.is_available():
                    effective_device = "cpu"
            except Exception:
                effective_device = "cpu"
        if effective_device != "cpu":
            self.model = self.model.to(device)
        self.model.eval()
        self.batch_size = batch_size
        self.device = effective_device
        logger.info(f"BGE-Reasoner-Rewriter loaded on {effective_device} (batch_size={batch_size})")

    def _build_messages(self, query: str, task: str = None) -> List[Dict[str, str]]:
        """构建官方格式的 prompt (user role only, no system)"""
        task_desc = task or self.DEFAULT_TASK
        content = self.PROMPT_TEMPLATE.format(task=task_desc, query=query)
        return [{"role": "user", "content": content}]

    @staticmethod
    def _extract_response(text: str) -> str:
        """从模型输出中提取 <response> 标签内的内容"""
        import re
        # Try <response>...</response> tags
        match = re.search(r'<response>(.*?)</response>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: return full text if no tags found
        return text.strip()

    @torch.no_grad()
    def rewrite_batch(self, queries: List[str], num_rewrites: int = NUM_REWRITES) -> List[List[str]]:
        """改写查询列表，每个查询生成 num_rewrites 个改写版本

        官方生成参数: temperature=0.6, top_p=0.9, max_new_tokens=4096

        Returns:
            List[List[str]]: 每个查询对应 num_rewrites 个改写版本的列表
        """
        all_rewritten = []

        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            batch_rewrites = [[] for _ in range(len(batch))]

            for _ in range(num_rewrites):
                messages_list = [self._build_messages(q) for q in batch]
                input_list = [
                    self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    for msgs in messages_list
                ]

                model_inputs = self.tokenizer(
                    input_list, padding=True, truncation=True, max_length=8192, return_tensors="pt"
                ).to(self.device)

                input_len = model_inputs['attention_mask'].shape[1]

                # Official sampling params: temperature=0.6, top_p=0.9
                # Reduced max_new_tokens from 4096 to 2048 (P95 ~1569 tokens, P99 ~1733)
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=2048,
                    temperature=0.6,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

                trimmed = generated_ids[:, input_len:]
                responses = self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)

                for j, resp in enumerate(responses):
                    extracted = self._extract_response(resp)
                    batch_rewrites[j].append(extracted)

            all_rewritten.extend(batch_rewrites)
            logger.info(f"  Rewriting: {min(i + self.batch_size, len(queries))}/{len(queries)} queries")

        return all_rewritten

    def cleanup(self):
        """释放显存"""
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("BGE-Reasoner-Rewriter memory released")


class BGEReasonerRewriterEvaluator(DSCLREvaluatorEngine):
    """BGE-Reasoner-Rewriter + RepLLaMA FollowIR 评测引擎

    继承 DSCLREvaluatorEngine 复用:
    - RepLLaMA 编码器加载和文档索引
    - _encode_queries 方法
    - _extract_results 方法
    - FollowIRDataLoader 数据加载

    关键区别:
    - 使用 BGE-Reasoner-Rewriter 生成 5 个改写查询
    - 对每个改写查询编码，然后聚合分数（求和）
    """

    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_dir: str,
        rewriter_path: str = REWRITER_PATH,
        num_rewrites: int = NUM_REWRITES,
        qr_cache_dir: str = "dataset/FollowIR_test/bge_reasoner_rewriter_queries",
        gpu_id: int = 0,
        **kwargs,
    ):
        self.rewriter_path = rewriter_path
        self.num_rewrites = num_rewrites
        self.qr_cache_dir = qr_cache_dir
        self.gpu_id = gpu_id

        kwargs.setdefault("device", f"cuda:{gpu_id}")
        kwargs.setdefault("batch_size", 16)  # Smaller batch to avoid OOM with 7B encoder
        kwargs.setdefault("use_cache", True)

        super().__init__(
            model_name=model_name,
            task_name=task_name,
            output_dir=output_dir,
            **kwargs,
        )

        os.makedirs(self.qr_cache_dir, exist_ok=True)

        logger.info("BGE-Reasoner-Rewriter evaluation mode enabled")
        logger.info(f"Rewriter: {self.rewriter_path}")
        logger.info(f"Num rewrites: {self.num_rewrites}")
        logger.info(f"Aggregation: score sum (official BGE-Reasoner method)")

    def _get_qr_cache_path(self) -> str:
        return os.path.join(
            self.qr_cache_dir,
            f"{self.task_name}_bge_reasoner_rewriter.jsonl",
        )

    def _load_qr_cache(self) -> Dict[str, List[str]]:
        """加载改写查询缓存 (qid -> list of 5 rewritten queries)"""
        cache_path = self._get_qr_cache_path()
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        cache[item["qid"]] = item["rewritten_queries"]
            logger.info(f"Loaded BGE-Reasoner-Rewriter cache: {len(cache)} entries")
        return cache

    def _save_qr_cache(self, cache: Dict[str, List[str]]):
        cache_path = self._get_qr_cache_path()
        with open(cache_path, "w", encoding="utf-8") as f:
            for qid, rewrites in cache.items():
                f.write(json.dumps({
                    "qid": qid,
                    "rewritten_queries": rewrites,
                }, ensure_ascii=False) + "\n")
        logger.info(f"BGE-Reasoner-Rewriter cache saved: {len(cache)} entries to {cache_path}")

    def run(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("BGE-Reasoner-Rewriter + RepLLaMA FollowIR Evaluation")
        logger.info(f"Reference: ReasonEmbed (BGE-Reasoner)")
        logger.info(f"Rewriter: {self.rewriter_path}")
        logger.info(f"Encoder: {self.model_name}")
        logger.info(f"Num rewrites: {self.num_rewrites}")
        logger.info(f"Aggregation: score sum across {self.num_rewrites} rewrites")
        logger.info("=" * 60)

        start_time = time.time()

        # 加载数据
        corpus, q_og, q_changed, candidates = self.data_loader.load()
        query_ids_og = list(q_og.keys())
        query_ids_changed = list(q_changed.keys())

        # Phase 1: 只改写 changed 查询（OG 查询用原始文本编码）
        qr_cache = self._load_qr_cache()

        missing_changed = [qid for qid in query_ids_changed if qid not in qr_cache]

        if missing_changed:
            logger.info(f"Phase 1: Generating {len(missing_changed)} changed rewritten queries (OG queries use original text)...")
            rewriter = BGEReasonerRewriter(
                model_path=self.rewriter_path,
                device=f"cuda:{self.gpu_id}",
                batch_size=1,  # 7B model + max_new_tokens=2048, use batch_size=1 to avoid OOM
            )

            changed_texts = [q_changed[qid] for qid in missing_changed]
            changed_rewritten = rewriter.rewrite_batch(changed_texts, self.num_rewrites)
            for qid, rewrites in zip(missing_changed, changed_rewritten):
                qr_cache[qid] = rewrites

            self._save_qr_cache(qr_cache)
            rewriter.cleanup()
            logger.info("Changed queries rewritten and cached, rewriter unloaded")
        else:
            logger.info(f"All changed queries already cached ({len(qr_cache)} entries)")

        # Phase 2: 加载 RepLLaMA 编码器并索引文档
        logger.info("Phase 2: Loading RepLLaMA encoder and indexing documents...")
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

        device = self.retriever.doc_embeddings.device

        # Phase 3: 编码查询并聚合分数
        logger.info(f"Phase 3: Encoding queries (OG: original text, Changed: {self.num_rewrites} rewrites)...")

        # OG queries - 直接用原始文本编码（不改写）
        og_queries = [q_og[qid] for qid in query_ids_og]
        q_emb_og = self._encode_queries(og_queries).to(device)
        S_og = torch.matmul(q_emb_og, self.retriever.doc_embeddings.T)
        logger.info(f"  OG queries encoded (original text, {len(og_queries)} queries)")

        # Changed queries
        S_changed_accum = None
        for rewrite_idx in range(self.num_rewrites):
            changed_queries = [qr_cache[qid][rewrite_idx] for qid in query_ids_changed]
            q_emb = self._encode_queries(changed_queries).to(device)
            S = torch.matmul(q_emb, self.retriever.doc_embeddings.T)
            if S_changed_accum is None:
                S_changed_accum = S
            else:
                S_changed_accum += S
            logger.info(f"  Changed rewrite {rewrite_idx + 1}/{self.num_rewrites} encoded")

        # Phase 4: 提取结果并计算 FollowIR 指标
        logger.info("Phase 4: Computing FollowIR metrics...")
        results_og = self._extract_results(S_og, query_ids_og, candidates)
        results_changed = self._extract_results(S_changed_accum, query_ids_changed, candidates)

        evaluator = FollowIREvaluator(self.task_name)
        metrics = evaluator.evaluate(results_og, results_changed)

        elapsed = time.time() - start_time

        p_mrr = metrics.get("p-MRR", 0.0)
        og_map = metrics.get("original", {}).get("map_at_1000", 0.0)
        changed_map = metrics.get("changed", {}).get("map_at_1000", 0.0)
        og_ndcg5 = metrics.get("original", {}).get("ndcg_at_5", 0.0)
        changed_ndcg5 = metrics.get("changed", {}).get("ndcg_at_5", 0.0)

        logger.info("=" * 60)
        logger.info("BGE-Reasoner-Rewriter + RepLLaMA Results")
        logger.info(f"   Task: {self.task_name}")
        logger.info(f"   Rewriter: {self.rewriter_path}")
        logger.info(f"   Encoder: {self.model_name}")
        logger.info(f"   Num rewrites: {self.num_rewrites}")
        logger.info(f"   Aggregation: score sum")
        logger.info(f"   p-MRR: {p_mrr:.4f}")
        logger.info(f"   OG MAP@1000: {og_map:.4f}")
        logger.info(f"   Changed MAP@1000: {changed_map:.4f}")
        logger.info(f"   OG nDCG@5: {og_ndcg5:.4f}")
        logger.info(f"   Changed nDCG@5: {changed_ndcg5:.4f}")
        logger.info(f"   Elapsed: {elapsed:.1f}s")
        logger.info("=" * 60)

        self._save_results(metrics, results_og, results_changed, qr_cache,
                           query_ids_og, query_ids_changed, elapsed)

        return {"metrics": metrics, "elapsed": elapsed}

    def _save_results(
        self,
        metrics: Dict[str, Any],
        results_og: Dict[str, Dict[str, float]],
        results_changed: Dict[str, Dict[str, float]],
        qr_cache: Dict[str, List[str]],
        query_ids_og: List[str],
        query_ids_changed: List[str],
        elapsed: float,
    ):
        os.makedirs(self.output_dir, exist_ok=True)

        summary = {
            "task": self.task_name,
            "model": self.model_name,
            "mode": "BGE-Reasoner-Rewriter + RepLLaMA",
            "rewriter": self.rewriter_path,
            "num_rewrites": self.num_rewrites,
            "aggregation": "score_sum",
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

        # 保存排名结果
        with open(os.path.join(self.output_dir, "ranking_og.json"), "w", encoding="utf-8") as f:
            json.dump(results_og, f, ensure_ascii=False)
        with open(os.path.join(self.output_dir, "ranking_changed.json"), "w", encoding="utf-8") as f:
            json.dump(results_changed, f, ensure_ascii=False)

        # 保存改写样本
        samples = {}
        for qid in query_ids_og[:5]:
            samples[qid] = {"type": "og", "rewrites": qr_cache.get(qid, [])}
        for qid in query_ids_changed[:5]:
            samples[qid] = {"type": "changed", "rewrites": qr_cache.get(qid, [])}
        with open(os.path.join(self.output_dir, "rewrite_samples.json"), "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="BGE-Reasoner-Rewriter + RepLLaMA FollowIR Evaluation")
    parser.add_argument("--task_name", type=str, required=True,
                        choices=["Core17InstructionRetrieval", "Robust04InstructionRetrieval", "News21InstructionRetrieval"])
    parser.add_argument("--model_name", type=str, default="/home/luwa/Documents/models/BGE-large-en-v1.5")
    parser.add_argument("--rewriter_path", type=str, default=REWRITER_PATH)
    parser.add_argument("--num_rewrites", type=int, default=NUM_REWRITES)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--qr_cache_dir", type=str,
                        default="dataset/FollowIR_test/bge_reasoner_rewriter_queries")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    evaluator = BGEReasonerRewriterEvaluator(
        model_name=args.model_name,
        task_name=args.task_name,
        output_dir=args.output_dir,
        rewriter_path=args.rewriter_path,
        num_rewrites=args.num_rewrites,
        qr_cache_dir=args.qr_cache_dir,
        gpu_id=args.gpu_id,
    )

    evaluator.run()


if __name__ == "__main__":
    main()
