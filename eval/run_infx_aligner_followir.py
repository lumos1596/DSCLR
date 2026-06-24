"""
INF-Query-Aligner + RepLLaMA FollowIR 评测脚本

独立评估 INF-Query-Aligner 的查询改写效果，使用 RepLLaMA 作为编码器。
与 INF-X-Retriever 完整 pipeline 不同，这里只评估 aligner 的改写质量。

用法:
    python eval/run_infx_aligner_followir.py --tasks Core17InstructionRetrieval
    python eval/run_infx_aligner_followir.py --tasks all
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import argparse
import gc
from typing import Dict, List
from datetime import datetime

import torch

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

QUERY_WRITER_PROMPT = (
    "For the input query, formulating a concise search query for dense retrieval "
    "by distilling the core intent from a complex user prompt and ignoring LLM instructions."
    "The response should be less than 200 words"
)


class INFQueryAligner:
    """INF-X Query Aligner - 基于 inf-query-aligner (Qwen2.5-7B-Instruct RL)"""

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
            {"role": "user", "content": f"{QUERY_WRITER_PROMPT}\n\n**Input Query:**\n{query}\n**Your Output:**\n"},
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

            trimmed = generated_ids[:, input_len:]
            response = self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)
            rewritten.extend(response)

            logger.info(f"  Rewriting: {min(i + self.batch_size, len(queries))}/{len(queries)}")

        return rewritten

    def cleanup(self):
        """释放显存"""
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("Aligner memory released")


def run_aligner_eval(task_name: str, output_dir: str, device: str = "cuda", batch_size: int = 64):
    """INF-Query-Aligner + RepLLaMA 评测"""
    from eval.engine_dscrl import DSCLREvaluatorEngine, load_cached_embeddings
    from eval.metrics import FollowIREvaluator
    from eval.engine import FollowIRDataLoader

    logger.info(f"=" * 60)
    logger.info(f"INF-Query-Aligner + RepLLaMA - {task_name}")
    logger.info(f"=" * 60)

    # 缓存目录
    cache_dir = "dataset/FollowIR_test/infx_aligner_queries"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{task_name}_infx_aligner.jsonl")

    # 加载数据
    data_loader = FollowIRDataLoader(task_name)
    corpus, q_og, q_changed, candidates = data_loader.load()
    og_ids = list(q_og.keys())
    changed_ids = list(q_changed.keys())

    # 加载或生成改写查询
    qr_cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    qr_cache[item['qid']] = item['rewritten_query']
        logger.info(f"Loaded {len(qr_cache)} cached queries")

    missing_og = [qid for qid in og_ids if qid not in qr_cache]
    missing_changed = [qid for qid in changed_ids if qid not in qr_cache]

    if missing_og or missing_changed:
        logger.info(f"Missing {len(missing_og)} og + {len(missing_changed)} changed queries, generating...")
        aligner = INFQueryAligner(device=device, batch_size=8)

        if missing_og:
            og_texts = [q_og[qid] for qid in missing_og]
            og_rewritten = aligner.rewrite(og_texts)
            for qid, r in zip(missing_og, og_rewritten):
                qr_cache[qid] = r

        if missing_changed:
            changed_texts = [q_changed[qid] for qid in missing_changed]
            changed_rewritten = aligner.rewrite(changed_texts)
            for qid, r in zip(missing_changed, changed_rewritten):
                qr_cache[qid] = r

        # 保存缓存
        with open(cache_path, 'w') as f:
            for qid, r in qr_cache.items():
                f.write(json.dumps({'qid': qid, 'rewritten_query': r}, ensure_ascii=False) + '\n')
        logger.info(f"Cached {len(qr_cache)} queries to {cache_path}")

        aligner.cleanup()
    else:
        logger.info("All queries cached, skipping aligner")

    # 用 RepLLaMA 编码
    logger.info("Loading RepLLaMA encoder...")
    engine = DSCLREvaluatorEngine(
        model_name='samaya-ai/RepLLaMA-reproduced',
        task_name=task_name,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        use_cache=True,
    )
    corpus2, q_og2, q_changed2, candidates2 = engine.data_loader.load()
    cached_data = load_cached_embeddings(engine.cache_dir, engine.task_name, engine.model_name)
    if cached_data:
        engine.retriever.set_embeddings(cached_data[0], cached_data[1])

    og_queries = [qr_cache[qid] for qid in og_ids]
    changed_queries = [qr_cache[qid] for qid in changed_ids]

    logger.info(f"Encoding {len(og_queries)} og + {len(changed_queries)} changed queries with RepLLaMA...")
    q_emb_og = engine._encode_queries(og_queries)
    q_emb_changed = engine._encode_queries(changed_queries)

    dev = engine.retriever.doc_embeddings.device
    q_emb_og = q_emb_og.to(dev)
    q_emb_changed = q_emb_changed.to(dev)

    S_og = torch.matmul(q_emb_og, engine.retriever.doc_embeddings.T)
    S_changed = torch.matmul(q_emb_changed, engine.retriever.doc_embeddings.T)

    results_og = engine._extract_results(S_og, og_ids, candidates)
    results_changed = engine._extract_results(S_changed, changed_ids, candidates)

    evaluator = FollowIREvaluator(task_name)
    metrics = evaluator.evaluate(results_og, results_changed)

    p_mrr = metrics.get('p-MRR', 0.0)
    og_map = metrics.get('original', {}).get('map_at_1000', 0.0)
    ch_map = metrics.get('changed', {}).get('map_at_1000', 0.0)
    og_ndcg5 = metrics.get('original', {}).get('ndcg_at_5', 0.0)
    ch_ndcg5 = metrics.get('changed', {}).get('ndcg_at_5', 0.0)

    logger.info(f"p-MRR: {p_mrr:.4f}")
    logger.info(f"OG MAP@1000: {og_map:.4f}, Changed MAP@1000: {ch_map:.4f}")
    logger.info(f"OG nDCG@5: {og_ndcg5:.4f}, Changed nDCG@5: {ch_ndcg5:.4f}")

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        'task': task_name,
        'encoder': 'samaya-ai/RepLLaMA-reproduced',
        'qr_model': 'inf-query-aligner (Qwen2.5-7B-Instruct RL)',
        'mode': 'INF-Query-Aligner + RepLLaMA',
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
    }
    with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 释放 RepLLaMA
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="INF-Query-Aligner + RepLLaMA FollowIR Evaluation")
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated task names or 'all'")
    parser.add_argument("--output_dir", type=str, default="results/infx_aligner")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    tasks = TASKS if args.tasks == "all" else args.tasks.split(",")

    all_results = {}

    for task_name in tasks:
        out_dir = os.path.join(args.output_dir, task_name)
        metrics = run_aligner_eval(task_name, out_dir, args.device, args.batch_size)
        all_results[task_name] = {
            'p-MRR': metrics.get('p-MRR', 0.0),
            'OG_MAP@1000': metrics.get('original', {}).get('map_at_1000', 0.0),
            'Changed_MAP@1000': metrics.get('changed', {}).get('map_at_1000', 0.0),
        }

    # 汇总
    logger.info("\n" + "=" * 60)
    logger.info("INF-Query-Aligner + RepLLaMA FollowIR Results")
    logger.info("=" * 60)
    for task, res in all_results.items():
        logger.info(f"  {task}: p-MRR={res['p-MRR']:.4f}, OG_MAP={res['OG_MAP@1000']:.4f}, Changed_MAP={res['Changed_MAP@1000']:.4f}")


if __name__ == "__main__":
    main()
