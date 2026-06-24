"""
IBM Granite aLoRA Query Rewrite + BGE FollowIR 评测引擎

评估 IBM Granite 3.2 8B aLoRA Query Rewrite 的查询改写效果，使用 BGE-large-en 作为编码器。

参考论文: A Library of LLM Intrinsics for Retrieval-Augmented Generation (IBM Research)
参考模型: https://huggingface.co/ibm-granite/granite-3.2-8b-alora-rag-query-rewrite
基础模型: ibm-granite/granite-3.2-8b-instruct

Granite aLoRA QR 原始流程:
    - 输入: 多轮对话 (user/assistant turns)
    - 输出: JSON {"rewritten_question": <REWRITE>}
    - 特殊触发: <|start_of_role|>rewrite: ...<|end_of_role|>
    - 目的: 将对话中最后一个用户查询去上下文化为独立查询

FollowIR 适配:
    - FollowIR 是单轮查询 + instruction，无对话历史
    - 将 (query, instruction) 作为单轮 user 输入
    - aLoRA 的 rewrite 触发器会尝试改写为独立查询
    - OG 查询用原始文本编码（不改写）
    - Changed 查询改写后编码
    - 使用 BGE-large-en-v1.5 作为编码器（Granite 8B + BGE 1.3GB 可共存于 24GB GPU）

用法:
    cd /home/luwa/Documents/DSCLR && CUDA_VISIBLE_DEVICES=0 /home/luwa/.conda/envs/dsclr/bin/python -m eval.run_granite_alora_qr_followir \
        --task_name Core17InstructionRetrieval \
        --gpu_id 0 \
        --output_dir results/granite_alora_qr/Core17InstructionRetrieval
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import json
import re
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

BASE_MODEL_PATH = "/home/luwa/Documents/models/granite-3.2-8b-instruct"
LORA_ADAPTER_PATH = "/home/luwa/Documents/models/granite-3.2-8b-alora-rag-query-rewrite"


class GraniteALoRAQueryRewriter:
    """IBM Granite 3.2 8B aLoRA Query Rewrite

    基于 granite-3.2-8b-instruct + LoRA adapter (rank=32, alpha=32)
    使用标准 peft 加载（aLoRA 的推理优化在 peft 0.18+ 不兼容，但 LoRA 权重完全兼容）

    原始用途: 多轮对话查询去上下文化
    FollowIR 适配: 将 (query, instruction) 改写为独立检索查询

    特殊触发 prompt:
        <|start_of_role|>rewrite: Reword the final utterance from the USER into a
        single utterance that doesn't need the prior conversation history to understand
        the user's intent. If the final utterance is a clear and standalone question,
        please DO NOT attempt to rewrite it, rather output the last user utterance as is.
        Your output format should be in JSON: { "rewritten_question": <REWRITE> }<|end_of_role|>
    """

    REWRITE_PROMPT = (
        '<|start_of_role|>rewrite: Reword the final utterance from the USER into a '
        'single utterance that doesn\'t need the prior conversation history to understand '
        'the user\'s intent. If the final utterance is a clear and standalone question, '
        'please DO NOT attempt to rewrite it, rather output the last user utterance as is. '
        'Your output format should be in JSON: { "rewritten_question": <REWRITE> }<|end_of_role|>'
    )

    def __init__(
        self,
        base_model_path: str = BASE_MODEL_PATH,
        lora_adapter_path: str = LORA_ADAPTER_PATH,
        batch_size: int = 1,
        device: str = "cuda",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        logger.info(f"Loading Granite aLoRA QR base model from {base_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, padding_side='left', trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
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
            model_base = model_base.to(effective_device)

        logger.info(f"Loading LoRA adapter from {lora_adapter_path}...")
        self.model = PeftModel.from_pretrained(model_base, lora_adapter_path)
        self.model.eval()

        self.batch_size = batch_size
        self.device = effective_device
        logger.info(f"Granite aLoRA QR loaded on {effective_device} (batch_size={batch_size})")

    def _build_input_text(self, query: str, instruction: str = "") -> str:
        """构建 Granite chat template + rewrite 触发器

        FollowIR 适配:
        - 单轮对话场景：user 包含 query + instruction
        - aLoRA 的 rewrite 触发器会尝试将查询改写为独立版本
        """
        user_content = query
        if instruction:
            user_content = f"{query}\n{instruction}"

        conv = [
            {"role": "system", "content": ""},
            {"role": "user", "content": user_content},
        ]
        chat_text = self.tokenizer.apply_chat_template(conv, tokenize=False)
        return chat_text + self.REWRITE_PROMPT

    @staticmethod
    def _extract_rewrite(output_text: str) -> str:
        """从模型输出中提取改写查询

        输出格式: JSON {"rewritten_question": <REWRITE>}
        """
        # 尝试 JSON 匹配
        pattern = r'\{\s*"rewritten_question"\s*:\s*"([^"]*)"\s*\}'
        m = re.search(pattern, output_text)
        if m:
            return m.group(1)

        # 尝试更宽松的匹配（处理转义字符等）
        pattern2 = r'"rewritten_question":\s*"([^"]*)"'
        m2 = re.search(pattern2, output_text)
        if m2:
            return m2.group(1)

        # 尝试提取 JSON 块
        try:
            pattern3 = r'\{[^{}]*"rewritten_question"[^{}]*\}'
            m3 = re.search(pattern3, output_text)
            if m3:
                data = json.loads(m3.group(0))
                return data.get("rewritten_question", "")
        except (json.JSONDecodeError, KeyError):
            pass

        # 最后尝试：取 <|end_of_role|> 后到 <|end_of_text|> 前的内容
        if "<|end_of_role|>" in output_text:
            after_role = output_text.split("<|end_of_role|>")[-1]
            if "<|end_of_text|>" in after_role:
                after_role = after_role.split("<|end_of_text|>")[0]
            return after_role.strip()

        return ""

    @torch.no_grad()
    def rewrite_batch(self, queries: List[str], instructions: List[str] = None) -> List[str]:
        """改写查询列表

        Returns:
            List[str]: 每个查询对应 1 个改写版本
        """
        if instructions is None:
            instructions = [""] * len(queries)

        all_rewritten = []

        for i in range(0, len(queries), self.batch_size):
            batch_queries = queries[i:i + self.batch_size]
            batch_instructions = instructions[i:i + self.batch_size]

            input_list = [
                self._build_input_text(q, instr)
                for q, instr in zip(batch_queries, batch_instructions)
            ]

            model_inputs = self.tokenizer(
                input_list, padding=True, truncation=True, max_length=4096, return_tensors="pt"
            ).to(self.device)

            input_len = model_inputs['attention_mask'].shape[1]

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=256,
                do_sample=False,  # greedy decoding for reproducibility
                pad_token_id=self.tokenizer.pad_token_id,
            )

            trimmed = generated_ids[:, input_len:]
            responses = self.tokenizer.batch_decode(trimmed, skip_special_tokens=False)

            for resp in responses:
                rewritten = self._extract_rewrite(resp)
                all_rewritten.append(rewritten)

            logger.info(f"  Rewriting: {min(i + self.batch_size, len(queries))}/{len(queries)} queries")

        return all_rewritten

    def cleanup(self):
        """释放显存"""
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("Granite aLoRA QR memory released")


class GraniteALoRAQREvaluator(DSCLREvaluatorEngine):
    """Granite aLoRA QR + BGE FollowIR 评测引擎

    继承 DSCLREvaluatorEngine 复用:
    - BGE 编码器加载和文档索引
    - _encode_queries 方法
    - _extract_results 方法
    - FollowIRDataLoader 数据加载

    关键区别:
    - 使用 Granite aLoRA QR 改写 changed 查询（1 个改写版本）
    - OG 查询用原始文本编码
    - Changed 查询用改写后的文本编码
    - 使用 BGE-large-en 编码器（与 Granite 8B 共存于 24GB GPU）
    """

    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_dir: str,
        base_model_path: str = BASE_MODEL_PATH,
        lora_adapter_path: str = LORA_ADAPTER_PATH,
        qr_cache_dir: str = "dataset/FollowIR_test/granite_alora_qr_queries",
        gpu_id: int = 0,
        **kwargs,
    ):
        self.base_model_path = base_model_path
        self.lora_adapter_path = lora_adapter_path
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

        logger.info("Granite aLoRA QR evaluation mode enabled")
        logger.info(f"Base model: {self.base_model_path}")
        logger.info(f"LoRA adapter: {self.lora_adapter_path}")
        logger.info(f"Mode: single rewrite (1 rewritten query per changed query)")

    def _get_qr_cache_path(self) -> str:
        return os.path.join(
            self.qr_cache_dir,
            f"{self.task_name}_granite_alora_qr.jsonl",
        )

    def _load_qr_cache(self) -> Dict[str, str]:
        """加载改写查询缓存 (qid -> rewritten query)"""
        cache_path = self._get_qr_cache_path()
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        cache[item["qid"]] = item["rewritten_query"]
            logger.info(f"Loaded Granite aLoRA QR cache: {len(cache)} entries")
        return cache

    def _save_qr_cache(self, cache: Dict[str, str]):
        cache_path = self._get_qr_cache_path()
        with open(cache_path, "w", encoding="utf-8") as f:
            for qid, rewritten in cache.items():
                f.write(json.dumps({
                    "qid": qid,
                    "rewritten_query": rewritten,
                }, ensure_ascii=False) + "\n")
        logger.info(f"Granite aLoRA QR cache saved: {len(cache)} entries to {cache_path}")

    def run(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("Granite aLoRA QR + BGE FollowIR Evaluation")
        logger.info(f"Reference: A Library of LLM Intrinsics for RAG (IBM Research)")
        logger.info(f"Rewriter: {self.base_model_path} + {self.lora_adapter_path}")
        logger.info(f"Encoder: {self.model_name}")
        logger.info(f"Mode: OG=original, Changed=single rewrite")
        logger.info("=" * 60)

        start_time = time.time()

        # 加载数据
        corpus, q_og, q_changed, candidates = self.data_loader.load()
        query_ids_og = list(q_og.keys())
        query_ids_changed = list(q_changed.keys())

        # Phase 1: 只改写 changed 查询
        qr_cache = self._load_qr_cache()

        missing_changed = [qid for qid in query_ids_changed if qid not in qr_cache]

        if missing_changed:
            logger.info(f"Phase 1: Generating {len(missing_changed)} changed rewritten queries...")
            rewriter = GraniteALoRAQueryRewriter(
                base_model_path=self.base_model_path,
                lora_adapter_path=self.lora_adapter_path,
                device=f"cuda:{self.gpu_id}",
                batch_size=1,
            )

            changed_texts = [q_changed[qid] for qid in missing_changed]
            changed_rewritten = rewriter.rewrite_batch(changed_texts)
            for qid, rewritten in zip(missing_changed, changed_rewritten):
                qr_cache[qid] = rewritten

            self._save_qr_cache(qr_cache)
            rewriter.cleanup()
            logger.info("Changed queries rewritten and cached, rewriter unloaded")
        else:
            logger.info(f"All changed queries already cached ({len(qr_cache)} entries)")

        # Phase 2: 加载 BGE 编码器并索引文档
        logger.info("Phase 2: Loading BGE encoder and indexing documents...")
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

        # Phase 3: 编码查询
        logger.info("Phase 3: Encoding queries (OG: original text, Changed: rewritten)...")

        # OG queries - 直接用原始文本编码
        og_queries = [q_og[qid] for qid in query_ids_og]
        q_emb_og = self._encode_queries(og_queries).to(device)
        S_og = torch.matmul(q_emb_og, self.retriever.doc_embeddings.T)
        logger.info(f"  OG queries encoded (original text, {len(og_queries)} queries)")

        # Changed queries - 用改写后的文本编码
        changed_queries = [qr_cache[qid] for qid in query_ids_changed]
        q_emb_changed = self._encode_queries(changed_queries).to(device)
        S_changed = torch.matmul(q_emb_changed, self.retriever.doc_embeddings.T)
        logger.info(f"  Changed queries encoded (rewritten, {len(changed_queries)} queries)")

        # Phase 4: 提取结果并计算 FollowIR 指标
        logger.info("Phase 4: Computing FollowIR metrics...")
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
        logger.info("Granite aLoRA QR + BGE Results")
        logger.info(f"   Task: {self.task_name}")
        logger.info(f"   Rewriter: {self.base_model_path} + {self.lora_adapter_path}")
        logger.info(f"   Encoder: {self.model_name}")
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
        qr_cache: Dict[str, str],
        query_ids_og: List[str],
        query_ids_changed: List[str],
        elapsed: float,
    ):
        os.makedirs(self.output_dir, exist_ok=True)

        summary = {
            "task": self.task_name,
            "model": self.model_name,
            "mode": "Granite aLoRA QR + BGE",
            "base_model": self.base_model_path,
            "lora_adapter": self.lora_adapter_path,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "p-MRR": metrics.get("p-MRR", 0.0),
                "original": metrics.get("original", {}),
                "changed": metrics.get("changed", {}),
                "full_scores": metrics.get("full_scores", {}),
            },
            "elapsed_seconds": elapsed,
        }

        summary_path = os.path.join(self.output_dir, "metrics_granite_alora_qr.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 保存排名结果
        with open(os.path.join(self.output_dir, "ranking_og.json"), "w", encoding="utf-8") as f:
            json.dump(results_og, f, ensure_ascii=False)
        with open(os.path.join(self.output_dir, "ranking_changed.json"), "w", encoding="utf-8") as f:
            json.dump(results_changed, f, ensure_ascii=False)

        # 保存改写样本
        samples = {}
        for qid in query_ids_changed[:10]:
            samples[qid] = qr_cache.get(qid, "")
        with open(os.path.join(self.output_dir, "rewrite_samples.json"), "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Granite aLoRA QR + BGE FollowIR Evaluation")
    parser.add_argument("--task_name", type=str, required=True,
                        choices=["Core17InstructionRetrieval", "Robust04InstructionRetrieval", "News21InstructionRetrieval"])
    parser.add_argument("--model_name", type=str, default="/home/luwa/Documents/models/BGE-large-en-v1.5")
    parser.add_argument("--base_model_path", type=str, default=BASE_MODEL_PATH)
    parser.add_argument("--lora_adapter_path", type=str, default=LORA_ADAPTER_PATH)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--qr_cache_dir", type=str,
                        default="dataset/FollowIR_test/granite_alora_qr_queries")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    evaluator = GraniteALoRAQREvaluator(
        model_name=args.model_name,
        task_name=args.task_name,
        output_dir=args.output_dir,
        base_model_path=args.base_model_path,
        lora_adapter_path=args.lora_adapter_path,
        qr_cache_dir=args.qr_cache_dir,
        gpu_id=args.gpu_id,
    )

    evaluator.run()


if __name__ == "__main__":
    main()
