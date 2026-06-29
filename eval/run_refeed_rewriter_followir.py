"""
ReFeed-8B Rewriter + RepLLaMA FollowIR 评测引擎

评估 ReFeed-8B 的查询改写效果，使用 RepLLaMA 作为编码器。
ReFeed-8B 基于 LLaMA-3.1-8B-Instruct 微调，专为摘要精炼设计（COLM'25）。

参考论文: ReFeed: Multi-dimensional Summarization Refinement with Reflective Reasoning on Feedback
参考模型: https://huggingface.co/DISLab/ReFeed-8B
参考代码: https://github.com/DISL-Lab/ReFeed

ReFeed 原始流程:
    - Input: document + summary + feedback (multi-dimensional)
    - Output: refined summary with reflective reasoning (Long-CoT)

FollowIR 适配:
    - 将 query 视为 "summary"，instruction 视为 "feedback"
    - ReFeed 精炼查询使其更好地遵循指令约束
    - OG 查询用原始文本编码（不改写）
    - Changed 查询改写后编码
    - 使用 RepLLaMA 作为编码器

用法:
    cd /home/luwa/Documents/DSCLR && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        /home/luwa/.conda/envs/dsclr/bin/python -m eval.run_refeed_rewriter_followir \
        --task_name Core17InstructionRetrieval \
        --output_dir results/refeed_rewriter/Core17InstructionRetrieval \
        --device cuda --batch_size 64
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

REFEED_MODEL_PATH = "/home/luwa/Documents/models/ReFeed-8B"


class ReFeedRewriter:
    """ReFeed-8B Rewriter - 基于 LLaMA-3.1-8B-Instruct 的摘要精炼模型

    原始用途: 接收 document + summary + feedback，输出精炼后的摘要
    FollowIR 适配: 将 query 视为 "summary"，instruction 视为 "feedback"，
                   让 ReFeed 精炼查询使其更好地遵循指令约束

    ReFeed 使用 Long-CoT (Chain-of-Thought) 进行反思推理，
    最终输出格式: <answer> {refined text} </answer>
    """

    SYSTEM_PROMPT = (
        "Your role as an assistant involves thoroughly exploring questions through "
        "a systematic long thinking process before providing the final precise and "
        "accurate solutions. This requires engaging in a comprehensive cycle of "
        "analysis, summarizing, exploration, reassessment, reflection, backtracing, "
        "and iteration to develop well-considered thinking process. Please structure "
        "your response into two main sections: Think and Answer. In the Think section, "
        "detail your reasoning process using the specified format: <think) {thought with "
        "steps separated with '\\n\\n'} </think) Each step should include detailed "
        "considerations such as analyzing questions, summarizing relevant findings, "
        "brainstorming new ideas, verifying the accuracy of the current steps, refining "
        "any errors, and revisiting previous steps. In the Answer section, based on "
        "various attempts, explorations, and reflections from the Think section, "
        "systematically present the final solution that you deem correct. The solution "
        "should remain a logical, accurate, concise expression style and detail necessary "
        "step needed to reach the conclusion, formatted as follows: <answer> {final "
        "formatted, precise, and clear solution} </answer> Now, try to solve the "
        "following question through the above guidelines:"
    )

    def __init__(self, model_path: str = REFEED_MODEL_PATH, batch_size: int = 1, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading ReFeed-8B from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        self.batch_size = batch_size
        self.device = device
        logger.info(f"ReFeed-8B loaded with device_map=auto (batch_size={batch_size})")

    def _build_user_prompt(self, query: str, instruction: str) -> str:
        """构建 FollowIR 适配的 user prompt

        将 query 视为 "summary"，instruction 视为 "feedback"，
        让 ReFeed 精炼查询使其更好地遵循指令约束。
        """
        return f"""Your goal is to deliberate on the provided feedback and propose actionable and specific aggregated feedback based on it.

Instructions:
1. Deliberate on the characteristics an ideal search query should achieve.
2. Assess and choose the validity of the given feedback in improving the query considering feedback quality criteria:
- Faithfulness: The query should accurately reflect the user's information need
- Completeness: The query should include all key constraints from the instructions
- Conciseness: The query should not include unnecessary or redundant content
3. Aggregate the valid feedback and revise the query by incorporating it.
4. Check whether revisions harm other quality dimensions.

Original Query:
{query}

Feedback (Instructions to incorporate):
{instruction}"""

    def _extract_answer(self, response: str) -> str:
        """从 ReFeed 的 Long-CoT 输出中提取 <answer> 标签内容"""
        # 尝试提取 <answer>...</answer>
        import re
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 如果没有 <answer> 标签，返回整个响应（去掉 think 部分）
        match = re.search(r'</think)\s*(.*)', response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 最后回退：返回完整响应
        return response.strip()

    @torch.no_grad()
    def rewrite_batch(self, queries: List[str], instructions: List[str]) -> List[str]:
        """改写查询列表"""
        all_rewritten = []

        for i in range(0, len(queries), self.batch_size):
            batch_queries = queries[i:i + self.batch_size]
            batch_instructions = instructions[i:i + self.batch_size]

            messages_list = []
            for q, instr in zip(batch_queries, batch_instructions):
                messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_user_prompt(q, instr)},
                ]
                messages_list.append(messages)

            input_list = [
                self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                for msgs in messages_list
            ]

            model_inputs = self.tokenizer(
                input_list, padding=True, truncation=True, max_length=4096, return_tensors="pt"
            ).to(self.model.device)

            input_len = model_inputs['attention_mask'].shape[1]

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=4096,  # ReFeed 使用 Long-CoT，需要较长输出
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,  # ReFeed 官方推荐
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            trimmed = generated_ids[:, input_len:]
            responses = self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)

            for resp in responses:
                rewritten = self._extract_answer(resp)
                all_rewritten.append(rewritten)

            logger.info(f"  Rewriting: {min(i + self.batch_size, len(queries))}/{len(queries)} queries")

        return all_rewritten

    def cleanup(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("ReFeed-8B memory released")


class ReFeedRewriterEvaluator(DSCLREvaluatorEngine):
    """ReFeed-8B Rewriter + RepLLaMA FollowIR 评测引擎"""

    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_dir: str,
        rewriter_path: str = REFEED_MODEL_PATH,
        qr_cache_dir: str = "dataset/FollowIR_test/refeed_rewriter_queries",
        **kwargs,
    ):
        self.rewriter_path = rewriter_path
        self.qr_cache_dir = qr_cache_dir

        # 确保 RepLLaMA adapter 本地路径正确
        from eval.models.repllama_encoder import RepLLaMAEncoder
        if "samaya-ai/RepLLaMA-reproduced" not in RepLLaMAEncoder.ADAPTER_LOCAL_MAP or \
           not os.path.isdir(RepLLaMAEncoder.ADAPTER_LOCAL_MAP.get("samaya-ai/RepLLaMA-reproduced", "")):
            # 使用 HuggingFace cache 路径
            import glob
            cache_pattern = os.path.expanduser(
                "~/.cache/huggingface/hub/models--samaya-ai--RepLLaMA-reproduced/snapshots/*"
            )
            snapshots = sorted(glob.glob(cache_pattern))
            if snapshots:
                RepLLaMAEncoder.ADAPTER_LOCAL_MAP["samaya-ai/RepLLaMA-reproduced"] = snapshots[-1]
                logger.info(f"Updated RepLLaMA adapter path to: {snapshots[-1]}")

        kwargs.setdefault("device", "cuda")
        kwargs.setdefault("batch_size", 64)
        kwargs.setdefault("use_cache", True)

        super().__init__(
            model_name=model_name,
            task_name=task_name,
            output_dir=output_dir,
            **kwargs,
        )

        os.makedirs(self.qr_cache_dir, exist_ok=True)

    def _get_qr_cache_path(self) -> str:
        return os.path.join(self.qr_cache_dir, f"{self.task_name}_refeed_rewriter.jsonl")

    def _load_qr_cache(self) -> Dict[str, str]:
        cache_path = self._get_qr_cache_path()
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        cache[item["qid"]] = item["rewritten_query"]
            logger.info(f"Loaded ReFeed rewriter cache: {len(cache)} entries")
        return cache

    def _save_qr_cache(self, cache: Dict[str, str]):
        cache_path = self._get_qr_cache_path()
        with open(cache_path, "w", encoding="utf-8") as f:
            for qid, rewritten in cache.items():
                f.write(json.dumps({
                    "qid": qid,
                    "rewritten_query": rewritten,
                }, ensure_ascii=False) + "\n")
        logger.info(f"ReFeed rewriter cache saved: {len(cache)} entries")

    def run(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("ReFeed-8B Rewriter + RepLLaMA FollowIR Evaluation")
        logger.info(f"Rewriter: {self.rewriter_path}")
        logger.info(f"Encoder: {self.model_name}")
        logger.info("=" * 60)

        start_time = time.time()

        # 加载数据
        corpus, q_og, q_changed, candidates = self.data_loader.load()
        query_ids_og = list(q_og.keys())
        query_ids_changed = list(q_changed.keys())

        # 加载原始 query 和 instruction（分离的）
        q_raw_og, q_raw_changed = self.data_loader.load_raw_queries()

        # Phase 1: 改写 changed 查询
        qr_cache = self._load_qr_cache()
        missing_changed = [qid for qid in query_ids_changed if qid not in qr_cache]

        if missing_changed:
            logger.info(f"Phase 1: Generating {len(missing_changed)} changed rewritten queries...")
            rewriter = ReFeedRewriter(
                model_path=self.rewriter_path,
                device="cuda",
                batch_size=1,
            )

            # 从 q_raw_changed 获取分离的 query_text 和 instruction
            changed_queries_only = []
            changed_instructions = []
            for qid in missing_changed:
                if qid in q_raw_changed:
                    query_text, instruction = q_raw_changed[qid]
                else:
                    # 回退：用完整文本作为 query，无 instruction
                    query_text = q_changed[qid]
                    instruction = ""
                changed_queries_only.append(query_text)
                changed_instructions.append(instruction)

            changed_rewritten = rewriter.rewrite_batch(changed_queries_only, changed_instructions)
            for qid, rewritten in zip(missing_changed, changed_rewritten):
                qr_cache[qid] = rewritten

            self._save_qr_cache(qr_cache)
            rewriter.cleanup()
            logger.info("Changed queries rewritten and cached, rewriter unloaded")
        else:
            logger.info(f"All changed queries already cached ({len(qr_cache)} entries)")

        # Phase 2: 编码和检索
        logger.info("Phase 2: Encoding and retrieval...")

        # 加载文档嵌入
        all_doc_ids = self._get_all_candidate_doc_ids(candidates)
        cached_data = None
        if self.use_cache:
            cached_data = load_cached_embeddings(self.cache_dir, self.task_name, self.model_name)

        if cached_data is not None:
            cached_embeddings, cached_doc_ids = cached_data
            if set(cached_doc_ids) == set(all_doc_ids):
                logger.info(f"✅ 使用缓存的文档向量 ({len(cached_doc_ids)} 个)")
                self.retriever.set_embeddings(cached_embeddings, cached_doc_ids)
            else:
                logger.warning("⚠️ 缓存文档ID不匹配，重新编码...")
                doc_texts = [corpus[did]['text'] for did in all_doc_ids]
                self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)
        else:
            logger.info("📚 编码候选文档...")
            doc_texts = [corpus[did]['text'] for did in all_doc_ids]
            self.retriever.index_documents(all_doc_ids, doc_texts, self.batch_size)

        # OG 查询用原始文本编码
        og_queries = [q_og[qid] for qid in query_ids_og]
        q_emb_og = self._encode_queries(og_queries)

        # Changed 查询用改写后的文本编码
        changed_rewritten_queries = [qr_cache[qid] for qid in query_ids_changed]
        q_emb_changed = self._encode_queries(changed_rewritten_queries)

        # 确保所有张量在同一设备上
        device = self.retriever.doc_embeddings.device
        q_emb_og = q_emb_og.to(device)
        q_emb_changed = q_emb_changed.to(device)

        # 计算相似度
        S_og = torch.matmul(q_emb_og, self.retriever.doc_embeddings.T)
        S_changed = torch.matmul(q_emb_changed, self.retriever.doc_embeddings.T)

        # 提取结果
        results_og = self._extract_results(S_og, query_ids_og, candidates)
        results_changed = self._extract_results(S_changed, query_ids_changed, candidates)

        # 计算 FollowIR 指标
        evaluator = FollowIREvaluator(self.task_name)
        metrics = evaluator.evaluate(results_og, results_changed)

        elapsed = time.time() - start_time
        logger.info(f"Total time: {elapsed:.1f}s")
        logger.info(f"p-MRR: {metrics.get('p-MRR', 0.0):.4f}")

        # 保存结果
        output = {
            "rewriter": "ReFeed-8B",
            "rewriter_path": self.rewriter_path,
            "encoder": self.model_name,
            "task_name": self.task_name,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "p-MRR": metrics.get("p-MRR", 0.0),
            "og_MAP@1000": metrics.get("original", {}).get("map_at_1000", 0.0),
            "og_nDCG@5": metrics.get("original", {}).get("ndcg_at_5", 0.0),
            "changed_MAP@1000": metrics.get("changed", {}).get("map_at_1000", 0.0),
            "changed_nDCG@5": metrics.get("changed", {}).get("ndcg_at_5", 0.0),
            "n_og_queries": len(query_ids_og),
            "n_changed_queries": len(query_ids_changed),
            "sample_rewrites": {qid: qr_cache[qid] for qid in list(qr_cache.keys())[:3]},
        }

        os.makedirs(self.output_dir, exist_ok=True)
        result_path = os.path.join(self.output_dir, "refeed_rewriter_results.json")
        with open(result_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Results saved to {result_path}")

        return metrics


def run_refeed_rewriter(task_name: str, output_dir: str, device: str = "cuda", batch_size: int = 64):
    """运行 ReFeed-8B Rewriter 评测"""
    model_name = "samaya-ai/RepLLaMA-reproduced"
    engine = ReFeedRewriterEvaluator(
        model_name=model_name,
        task_name=task_name,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        use_cache=True,
    )
    return engine.run()


def main():
    parser = argparse.ArgumentParser(description="ReFeed-8B Rewriter FollowIR Evaluation")
    parser.add_argument("--task_name", type=str, required=True,
                        choices=["Core17InstructionRetrieval", "Robust04InstructionRetrieval",
                                 "News21InstructionRetrieval"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    run_refeed_rewriter(args.task_name, args.output_dir, args.device, args.batch_size)


if __name__ == "__main__":
    main()
