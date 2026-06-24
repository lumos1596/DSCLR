"""
Promptriever 基线在 FollowIR 上的评测脚本

Promptriever 是基于 LLaMA 3.1 8B Instruct 的指令感知检索模型。
本脚本复现其在 FollowIR 基准上的表现，作为基线对比。

模型: samaya-ai/promptriever-llama3.1-8b-instruct-v1
基础模型: meta-llama/Meta-Llama-3.1-8B-Instruct

用法:
    cd /home/luwa/Documents/DSCLR
    /home/luwa/.conda/envs/dsclr/bin/python eval/eval_promptriever.py --device cuda
"""

import os
import sys
import argparse
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel, PeftConfig
from tqdm import tqdm

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import DataLoader as MetricsDataLoader, FollowIREvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 模型路径
PROMPTRIEVER_ADAPTER_PATH = "/home/luwa/Documents/models/promptriever-llama3.1-8b-instruct-v1"
BASE_MODEL_PATH = "/home/luwa/Documents/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"

# 缓存目录
CACHE_DIR = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/embeddings/promptriever_llama31_8b_instruct"

TASK_NAMES = [
    "Core17InstructionRetrieval",
    "Robust04InstructionRetrieval",
    "News21InstructionRetrieval",
]


class PromptrieverEncoder:
    """Promptriever 编码器 - 基于 LLaMA 3.1 8B Instruct + LoRA"""

    def __init__(
        self,
        adapter_path: str = PROMPTRIEVER_ADAPTER_PATH,
        base_model_path: str = BASE_MODEL_PATH,
        device: str = "cuda",
        batch_size: int = 4,
        max_seq_length: int = 2048,
    ):
        self.adapter_path = adapter_path
        self.base_model_path = base_model_path
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        logger.info(f"Loading Promptriever model...")
        logger.info(f"  Adapter: {adapter_path}")
        logger.info(f"  Base model: {base_model_path}")

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        # 加载基础模型
        base_model = AutoModel.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map=device,
        )

        # 加载并合并 LoRA adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model = model.merge_and_unload()
        self.model.eval()

        logger.info("Promptriever model loaded successfully")

    def _create_batch_dict(self, input_texts: List[str]):
        """创建 batch 输入，遵循 Promptriever 官方代码的 token 处理方式"""
        batch_dict = self.tokenizer(
            input_texts,
            max_length=self.max_seq_length - 1,
            return_token_type_ids=False,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        # 追加 EOS token
        batch_dict["input_ids"] = [
            ids + [self.tokenizer.eos_token_id]
            for ids in batch_dict["input_ids"]
        ]
        return self.tokenizer.pad(
            batch_dict,
            padding=True,
            pad_to_multiple_of=8,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def encode(self, texts: List[str], batch_size: Optional[int] = None, show_progress: bool = True) -> torch.Tensor:
        """编码文本列表，返回 L2 归一化的嵌入向量"""
        batch_size = batch_size or self.batch_size
        all_embeddings = []

        num_batches = (len(texts) + batch_size - 1) // batch_size
        pbar = tqdm(total=num_batches, desc="Encoding", unit="batch", disable=not show_progress)

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            batch_dict = self._create_batch_dict(batch_texts)
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = self.model(**batch_dict)
                    last_hidden_state = outputs.last_hidden_state
                    # 取最后一个 token 的表示（EOS pooling）
                    sequence_lengths = batch_dict["attention_mask"].sum(dim=1) - 1
                    batch_size_cur = last_hidden_state.shape[0]
                    reps = last_hidden_state[
                        torch.arange(batch_size_cur, device=last_hidden_state.device),
                        sequence_lengths,
                    ]
                    embeddings = F.normalize(reps, p=2, dim=-1)
                    all_embeddings.append(embeddings.cpu())

            pbar.update(1)

        pbar.close()
        return torch.cat(all_embeddings, dim=0)

    def encode_queries(self, texts: List[str], **kwargs) -> torch.Tensor:
        """编码查询 - 使用 'query:  ' 前缀（双空格）"""
        formatted = [f"query:  {text.strip()}" for text in texts]
        return self.encode(formatted, **kwargs)

    def encode_documents(self, texts: List[str], **kwargs) -> torch.Tensor:
        """编码文档 - 使用 'passage:  ' 前缀（双空格）"""
        formatted = [f"passage:  {text.strip()}" for text in texts]
        return self.encode(formatted, **kwargs)


def load_cached_embeddings(cache_dir: str, task_name: str) -> Optional[Tuple[torch.Tensor, List[str]]]:
    """加载缓存的文档向量"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{task_name}_corpus_embeddings.npy")
    ids_file = os.path.join(cache_dir, f"{task_name}_corpus_ids.json")

    if os.path.exists(cache_file) and os.path.exists(ids_file):
        logger.info(f"Loading cached embeddings: {cache_file}")
        embeddings = np.load(cache_file)
        with open(ids_file, 'r') as f:
            doc_ids = json.load(f)
        logger.info(f"Cached: {len(doc_ids)} docs, shape={embeddings.shape}")
        return torch.tensor(embeddings), doc_ids

    return None


def save_embeddings_cache(cache_dir: str, task_name: str, embeddings: torch.Tensor, doc_ids: List[str]):
    """保存文档向量到缓存"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{task_name}_corpus_embeddings.npy")
    ids_file = os.path.join(cache_dir, f"{task_name}_corpus_ids.json")

    np.save(cache_file, embeddings.cpu().numpy())
    with open(ids_file, 'w') as f:
        json.dump(doc_ids, f)
    logger.info(f"Embeddings cached: {cache_file}")


def get_all_candidate_doc_ids(candidates: Dict[str, List[str]]) -> List[str]:
    """获取所有候选文档 ID（去重并保持顺序）"""
    seen = set()
    all_ids = []
    for doc_ids in candidates.values():
        for did in doc_ids:
            if did not in seen:
                seen.add(did)
                all_ids.append(did)
    return all_ids


def run_retrieval(
    query_embeddings: torch.Tensor,
    query_ids: List[str],
    doc_embeddings: torch.Tensor,
    doc_ids: List[str],
    candidates: Dict[str, List[str]],
) -> Dict[str, Dict[str, float]]:
    """执行检索，返回每个查询的文档得分"""
    # 计算所有得分
    scores = torch.matmul(query_embeddings, doc_embeddings.T)  # (n_queries, n_docs)

    results = {}
    for i, qid in enumerate(query_ids):
        base_qid = qid.replace('-og', '').replace('-changed', '')
        cand_ids = candidates.get(base_qid, doc_ids)

        # 只返回候选文档的得分
        id_to_idx = {did: j for j, did in enumerate(doc_ids)}
        q_results = {}
        for did in cand_ids:
            if did in id_to_idx:
                q_results[did] = scores[i, id_to_idx[did]].item()
        results[qid] = q_results

    return results


def evaluate_task(
    task_name: str,
    encoder: PromptrieverEncoder,
    device: str,
    batch_size: int,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """在单个 FollowIR 任务上评测 Promptriever"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {task_name}")
    logger.info(f"{'='*60}")

    data_loader = MetricsDataLoader(task_name)

    # 加载数据
    corpus = data_loader.load_corpus()
    q_og, q_changed = data_loader.load_queries()
    candidates = data_loader.load_candidates()

    logger.info(f"Corpus: {len(corpus)} docs, OG queries: {len(q_og)}, Changed queries: {len(q_changed)}")

    # 获取候选文档
    all_doc_ids = get_all_candidate_doc_ids(candidates)
    logger.info(f"Total candidate docs: {len(all_doc_ids)}")

    # 编码或加载文档向量
    cached_data = None
    if use_cache:
        cached_data = load_cached_embeddings(CACHE_DIR, task_name)

    if cached_data is not None:
        cached_embeddings, cached_doc_ids = cached_data
        if set(cached_doc_ids) == set(all_doc_ids):
            logger.info("Using cached document embeddings")
            id_to_idx = {did: i for i, did in enumerate(cached_doc_ids)}
            doc_embeddings = torch.stack([cached_embeddings[id_to_idx[did]] for did in all_doc_ids])
        else:
            logger.warning("Cache doc IDs mismatch, re-encoding...")
            doc_texts = [corpus[did]['text'] for did in all_doc_ids]
            doc_embeddings = encoder.encode_documents(doc_texts, batch_size=batch_size)
            save_embeddings_cache(CACHE_DIR, task_name, doc_embeddings, all_doc_ids)
    else:
        logger.info("Encoding documents...")
        doc_texts = [corpus[did]['text'] for did in all_doc_ids]
        doc_embeddings = encoder.encode_documents(doc_texts, batch_size=batch_size)
        if use_cache:
            save_embeddings_cache(CACHE_DIR, task_name, doc_embeddings, all_doc_ids)

    # 编码 og 查询
    logger.info("Encoding OG queries...")
    og_query_ids = list(q_og.keys())
    og_query_texts = [q_og[qid] for qid in og_query_ids]
    og_query_embeddings = encoder.encode_queries(og_query_texts, batch_size=batch_size)

    # 编码 changed 查询
    logger.info("Encoding changed queries...")
    changed_query_ids = list(q_changed.keys())
    changed_query_texts = [q_changed[qid] for qid in changed_query_ids]
    changed_query_embeddings = encoder.encode_queries(changed_query_texts, batch_size=batch_size)

    # 执行检索
    logger.info("Running retrieval for OG queries...")
    results_og = run_retrieval(og_query_embeddings, og_query_ids, doc_embeddings, all_doc_ids, candidates)

    logger.info("Running retrieval for changed queries...")
    results_changed = run_retrieval(changed_query_embeddings, changed_query_ids, doc_embeddings, all_doc_ids, candidates)

    # 计算评测指标
    logger.info("Computing metrics...")
    evaluator = FollowIREvaluator(task_name)
    metrics = evaluator.evaluate(results_og, results_changed)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Promptriever on FollowIR")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for encoding")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--use_cache", action="store_true", default=True, help="Use cached embeddings")
    parser.add_argument("--no_cache", action="store_true", help="Do not use cached embeddings")
    parser.add_argument("--tasks", nargs="+", default=None, help="Specific tasks to evaluate")
    args = parser.parse_args()

    device = args.device
    if device == "cuda":
        torch.cuda._lazy_init()
        if not torch.cuda.is_available():
            device = "cpu"
    logger.info(f"Device: {device}")

    use_cache = args.use_cache and not args.no_cache
    tasks = args.tasks or TASK_NAMES

    # 加载模型
    encoder = PromptrieverEncoder(
        adapter_path=PROMPTRIEVER_ADAPTER_PATH,
        base_model_path=BASE_MODEL_PATH,
        device=device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )

    # 评测所有任务
    all_metrics = {}
    for task_name in tasks:
        start_time = time.time()
        metrics = evaluate_task(task_name, encoder, device, args.batch_size, use_cache)
        elapsed = time.time() - start_time

        all_metrics[task_name] = metrics

        logger.info(f"\n{'='*60}")
        logger.info(f"Results for {task_name}:")
        logger.info(f"  p-MRR: {metrics.get('p-MRR', 0):.4f}")
        logger.info(f"  OG MAP@1000: {metrics.get('original', {}).get('map_at_1000', 0):.4f}")
        logger.info(f"  OG nDCG@5: {metrics.get('original', {}).get('ndcg_at_5', 0):.4f}")
        logger.info(f"  Changed MAP@1000: {metrics.get('changed', {}).get('map_at_1000', 0):.4f}")
        logger.info(f"  Changed nDCG@5: {metrics.get('changed', {}).get('ndcg_at_5', 0):.4f}")
        logger.info(f"  Time: {elapsed:.1f}s")
        logger.info(f"{'='*60}")

    # 汇总结果
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: Promptriever (LLaMA 3.1 8B Instruct) on FollowIR")
    logger.info("=" * 80)

    pmrr_values = []
    c17_cmap = r04_cmap = n21_cndcg = 0.0

    for task_name, metrics in all_metrics.items():
        pmrr = metrics.get('p-MRR', 0)
        pmrr_values.append(pmrr)

        if "Core17" in task_name:
            c17_cmap = metrics.get('changed', {}).get('map_at_1000', 0)
        elif "Robust04" in task_name:
            r04_cmap = metrics.get('changed', {}).get('map_at_1000', 0)
        elif "News21" in task_name:
            n21_cndcg = metrics.get('changed', {}).get('ndcg_at_5', 0)

        logger.info(f"  {task_name}: p-MRR={pmrr:.4f}")

    mean_pmrr = sum(pmrr_values) / len(pmrr_values) if pmrr_values else 0
    target_avg = (c17_cmap + r04_cmap + n21_cndcg) / 3 if (c17_cmap + r04_cmap + n21_cndcg) > 0 else 0

    logger.info(f"\n  Mean p-MRR: {mean_pmrr:.4f}")
    logger.info(f"  target_avg: {target_avg:.4f}")
    logger.info(f"    Core17 changed_MAP@1000: {c17_cmap:.4f}")
    logger.info(f"    Robust04 changed_MAP@1000: {r04_cmap:.4f}")
    logger.info(f"    News21 changed_nDCG@5: {n21_cndcg:.4f}")

    # 保存结果
    output_path = "/home/luwa/Documents/DSCLR/evaluation/promptriever_followir_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = {
        "model": "samaya-ai/promptriever-llama3.1-8b-instruct-v1",
        "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mean_pMRR": mean_pmrr,
        "target_avg": target_avg,
        "Core17_changed_MAP@1000": c17_cmap,
        "Robust04_changed_MAP@1000": r04_cmap,
        "News21_changed_nDCG@5": n21_cndcg,
        "per_task": {
            task: {
                "p-MRR": m.get("p-MRR", 0),
                "og_MAP@1000": m.get("original", {}).get("map_at_1000", 0),
                "og_nDCG@5": m.get("original", {}).get("ndcg_at_5", 0),
                "changed_MAP@1000": m.get("changed", {}).get("map_at_1000", 0),
                "changed_nDCG@5": m.get("changed", {}).get("ndcg_at_5", 0),
            }
            for task, m in all_metrics.items()
        },
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
