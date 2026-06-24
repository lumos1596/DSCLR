"""
INF-X-Retriever FollowIR 评测脚本

评测两种模式:
1. 纯检索器模式 (inf-retriever-v1-pro only)
2. Query Aligner + Retriever 模式 (inf-query-aligner + inf-retriever-v1-pro)

用法:
    python eval/run_infx_followir.py --mode retriever_only
    python eval/run_infx_followir.py --mode full
    python eval/run_infx_followir.py --mode both
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import argparse
import copy
import time
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

import torch
import torch.nn as nn

os.environ.setdefault('HF_HOME', '/home/luwa/.cache/huggingface')
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TASKS = [
    "Core17InstructionRetrieval",
    "Robust04InstructionRetrieval",
    "News21InstructionRetrieval",
]

ALIGNER_PATH = "/home/luwa/Documents/models/inf-query-aligner"
RETRIEVER_PATH = "/home/luwa/Documents/models/inf-retriever-v1-pro"


class INFQueryAligner:
    """INF-X Query Aligner - 基于 inf-query-aligner (Qwen2.5-7B-Instruct RL)"""

    QUERY_WRITER_PROMPT = (
        "For the input query, formulating a concise search query for dense retrieval "
        "by distilling the core intent from a complex user prompt and ignoring LLM instructions."
        "The response should be less than 200 words"
    )

    def __init__(self, model_path: str = ALIGNER_PATH, batch_size: int = 8, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading INF Query Aligner from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device,
        )
        self.model.eval()
        self.batch_size = batch_size
        self.device = device
        logger.info("INF Query Aligner loaded successfully")

    def _build_prompt(self, query: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": f"{self.QUERY_WRITER_PROMPT}\n\n**Input Query:**\n{query}\n**Your Output:**\n"},
        ]

    @torch.no_grad()
    def rewrite(self, queries: List[str]) -> List[str]:
        """改写查询列表"""
        rewritten = []

        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            prompts = [self._build_prompt(q) for q in batch]

            input_list = [
                self.tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                for p in prompts
            ]

            model_inputs = self.tokenizer(
                input_list, padding=True, truncation=True, max_length=8192, return_tensors="pt"
            ).to(self.device)

            input_len = model_inputs['attention_mask'].shape[1]

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512,
            )

            # 去掉输入部分
            trimmed = generated_ids[:, input_len:]
            response = self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)
            rewritten.extend(response)

            if (i + self.batch_size) % (self.batch_size * 5) == 0:
                logger.info(f"  改写进度: {min(i + self.batch_size, len(queries))}/{len(queries)}")

        return rewritten


def run_retriever_only(task_name: str, output_dir: str, device: str = "cuda", batch_size: int = 16):
    """纯检索器模式 - 直接用 inf-retriever-v1-pro"""
    from eval.engine import FollowIREvaluatorEngine

    logger.info(f"=" * 60)
    logger.info(f"INF-X-Retriever (retriever only) - {task_name}")
    logger.info(f"=" * 60)

    engine = FollowIREvaluatorEngine(
        model_name="inf-retriever-v1-pro",
        task_name=task_name,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        use_cache=True,
    )

    metrics = engine.run()
    return metrics


def run_full_pipeline(task_name: str, output_dir: str, device: str = "cuda", batch_size: int = 16):
    """完整模式 - Query Aligner + Retriever"""
    from eval.models.infx_encoder import INFXRetrieverEncoder
    from eval.metrics.evaluator import DataLoader, FollowIREvaluator
    from eval.engine import FollowIRDataLoader
    from eval.models.encoder import DenseRetriever

    logger.info(f"=" * 60)
    logger.info(f"INF-X-Retriever (full: aligner + retriever) - {task_name}")
    logger.info(f"=" * 60)

    # 1. 加载数据
    data_loader = FollowIRDataLoader(task_name)
    corpus, q_og, q_changed, candidates = data_loader.load()

    # 2. 加载 Query Aligner
    aligner = INFQueryAligner(device=device)

    # 3. 改写查询
    logger.info("改写 og 查询...")
    og_query_ids = list(q_og.keys())
    og_query_texts = [q_og[qid] for qid in og_query_ids]
    og_rewritten = aligner.rewrite(og_query_texts)
    q_og_rewritten = {qid: r for qid, r in zip(og_query_ids, og_rewritten)}

    logger.info("改写 changed 查询...")
    changed_query_ids = list(q_changed.keys())
    changed_query_texts = [q_changed[qid] for qid in changed_query_ids]
    changed_rewritten = aligner.rewrite(changed_query_texts)
    q_changed_rewritten = {qid: r for qid, r in zip(changed_query_ids, changed_rewritten)}

    # 保存改写结果
    rewrite_path = os.path.join(output_dir, "rewritten_queries.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(rewrite_path, 'w') as f:
        json.dump({
            "og": q_og_rewritten,
            "changed": q_changed_rewritten,
        }, f, indent=2, ensure_ascii=False)

    # 4. 释放 aligner 显存
    del aligner
    torch.cuda.empty_cache()

    # 5. 加载检索器
    encoder = INFXRetrieverEncoder(
        model_name="inf-retriever-v1-pro",
        device=device,
        batch_size=batch_size,
    )
    retriever = DenseRetriever(encoder)

    # 6. 编码文档
    all_doc_ids = list(set(did for docs in candidates.values() for did in docs))
    doc_texts = [corpus[did]['text'] for did in all_doc_ids]
    logger.info(f"编码 {len(all_doc_ids)} 个文档...")
    retriever.index_documents(all_doc_ids, doc_texts, batch_size=batch_size)

    # 7. 检索 og 查询
    logger.info("检索 og 查询 (改写后)...")
    results_og = {}
    og_embeddings = encoder.encode_queries(list(q_og_rewritten.values()), batch_size=batch_size)
    for idx, qid in enumerate(og_query_ids):
        base_qid = qid.replace('-og', '')
        if base_qid in candidates:
            results_og[qid] = retriever.compute_scores(og_embeddings[idx], candidates[base_qid])

    # 8. 检索 changed 查询
    logger.info("检索 changed 查询 (改写后)...")
    results_changed = {}
    changed_embeddings = encoder.encode_queries(list(q_changed_rewritten.values()), batch_size=batch_size)
    for idx, qid in enumerate(changed_query_ids):
        base_qid = qid.replace('-changed', '')
        if base_qid in candidates:
            results_changed[qid] = retriever.compute_scores(changed_embeddings[idx], candidates[base_qid])

    # 9. 计算指标
    evaluator = FollowIREvaluator(task_name)
    metrics = evaluator.evaluate(results_og, results_changed)

    # 保存结果
    metrics_path = os.path.join(output_dir, "metrics_infx_full.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            "model": "INF-X-Retriever (full: aligner + retriever)",
            "task": task_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"p-MRR: {metrics.get('p-MRR', 0):.4f}")
    logger.info(f"OG MAP@1000: {metrics.get('original', {}).get('map_at_1000', 0):.4f}")
    logger.info(f"Changed MAP@1000: {metrics.get('changed', {}).get('map_at_1000', 0):.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="INF-X-Retriever FollowIR Evaluation")
    parser.add_argument("--mode", type=str, default="both", choices=["retriever_only", "full", "both"],
                        help="Evaluation mode")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task names (default: all FollowIR tasks)")
    parser.add_argument("--output_dir", type=str, default="results/infx_retriever")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    tasks = args.tasks.split(",") if args.tasks else TASKS

    all_results = {}

    for task_name in tasks:
        logger.info(f"\n{'='*80}")
        logger.info(f"Task: {task_name}")
        logger.info(f"{'='*80}")

        if args.mode in ["retriever_only", "both"]:
            output_dir = os.path.join(args.output_dir, "retriever_only", task_name)
            metrics = run_retriever_only(task_name, output_dir, args.device, args.batch_size)
            all_results[f"{task_name}_retriever_only"] = metrics

        if args.mode in ["full", "both"]:
            output_dir = os.path.join(args.output_dir, "full", task_name)
            metrics = run_full_pipeline(task_name, output_dir, args.device, args.batch_size)
            all_results[f"{task_name}_full"] = metrics

    # 汇总
    logger.info("\n" + "=" * 80)
    logger.info("INF-X-Retriever FollowIR Results Summary")
    logger.info("=" * 80)
    for key, metrics in all_results.items():
        p_mrr = metrics.get('p-MRR', 0)
        og_map = metrics.get('original', {}).get('map_at_1000', 0)
        ch_map = metrics.get('changed', {}).get('map_at_1000', 0)
        logger.info(f"  {key}: p-MRR={p_mrr:.4f}, OG_MAP@1000={og_map:.4f}, Changed_MAP@1000={ch_map:.4f}")

    # 保存汇总
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
