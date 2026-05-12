"""
使用本地 Qwen3-0.6B 模型进行指令改写
复用 reformulator.py 中的 v4 提示词模板
输出格式与 DeepSeek API 版本完全一致
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.reformulator import get_prompt_templates, DEFAULT_PROMPT_VERSION

logger = logging.getLogger(__name__)


class Qwen3LocalReformulator:
    def __init__(
        self,
        model_path: str = "/home/luwa/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B",
        device: str = "cuda",
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        self.prompt_version = prompt_version
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        logger.info(f"Loading Qwen3-0.6B from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if device == "cuda":
            try:
                torch.cuda._lazy_init()
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available, falling back to CPU")
                    device = "cpu"
            except Exception as e:
                logger.warning(f"CUDA init failed ({e}), falling back to CPU")
                device = "cpu"

        if device == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).to("cpu")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        self.model.eval()
        self.device = device
        logger.info(f"Qwen3-0.6B loaded on {device}")

        self.system_prompt, self.user_prompt_template = get_prompt_templates(prompt_version)

    def reformulate(self, query: str, instruction: str) -> Tuple[str, str]:
        user_prompt = self.user_prompt_template.format(
            query=query,
            instruction=instruction
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                top_p=0.9,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        result_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        q_plus, q_minus = self._parse_result(result_text, query)
        return q_plus, q_minus

    def _parse_result(self, result_text: str, fallback_query: str) -> Tuple[str, str]:
        try:
            json_str = result_text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            brace_start = json_str.find("{")
            brace_end = json_str.rfind("}")
            if brace_start >= 0 and brace_end > brace_start:
                json_str = json_str[brace_start:brace_end + 1]

            parsed = json.loads(json_str)
            q_plus = parsed.get("Q_plus", "").strip()
            q_minus = parsed.get("Q_minus", "[NONE]").strip()

            if not q_plus:
                q_plus = fallback_query

            return q_plus, q_minus

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"JSON parse failed: {e}, raw: {result_text[:200]}")
            return fallback_query, "[NONE]"


def load_followir_queries(task_name: str) -> Tuple[Dict, Dict]:
    import datasets

    path_map = {
        "Core17InstructionRetrieval": "jhu-clsp/core17-instructions-mteb",
        "Robust04InstructionRetrieval": "jhu-clsp/robust04-instructions-mteb",
        "News21InstructionRetrieval": "jhu-clsp/news21-instructions-mteb",
    }
    dataset_path = path_map.get(task_name, "")
    if not dataset_path:
        raise ValueError(f"Unknown task: {task_name}")

    ds_q = datasets.load_dataset(dataset_path, 'queries', trust_remote_code=True)
    ds_inst = datasets.load_dataset(dataset_path, 'instruction', trust_remote_code=True)

    q_split = 'queries' if 'queries' in ds_q else list(ds_q.keys())[0]
    i_split = 'instruction' if 'instruction' in ds_inst else list(ds_inst.keys())[0]

    instruction_dict = {}
    for item in ds_inst[i_split]:
        qid = str(item.get('query-id', ''))
        instruction_dict[qid] = str(item.get('instruction', ''))

    q_og = {}
    q_changed = {}

    for q in ds_q[q_split]:
        full_qid = str(q.get('_id', q.get('id', '')))
        query_text = q.get('text', '')
        inst = instruction_dict.get(full_qid, "")

        if full_qid.endswith('-og'):
            q_og[full_qid] = (query_text, inst)
        elif full_qid.endswith('-changed'):
            q_changed[full_qid] = (query_text, inst)

    return q_og, q_changed


def run_reformulation(
    task_name: str,
    model_path: str,
    output_dir: str,
    device: str = "cuda",
    prompt_version: str = DEFAULT_PROMPT_VERSION,
):
    reformulator = Qwen3LocalReformulator(
        model_path=model_path,
        device=device,
        prompt_version=prompt_version,
    )

    q_og, q_changed = load_followir_queries(task_name)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"dual_queries_qwen3_{task_name}.jsonl")

    existing_qids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing_qids.add(item['qid'])
        logger.info(f"Found {len(existing_qids)} existing records, will skip them")

    all_queries = []
    for qid, (query_text, instruction) in q_og.items():
        all_queries.append((qid, query_text, instruction, "og"))
    for qid, (query_text, instruction) in q_changed.items():
        all_queries.append((qid, query_text, instruction, "changed"))

    total = len(all_queries)
    new_count = sum(1 for qid, _, _, _ in all_queries if qid not in existing_qids)
    logger.info(f"Task: {task_name}, Total queries: {total}, New to process: {new_count}")

    processed = 0
    failed = 0
    start_time = time.time()

    with open(output_file, 'a', encoding='utf-8') as f:
        for qid, query_text, instruction, query_type in all_queries:
            if qid in existing_qids:
                continue

            try:
                idx = int(qid.split('-')[0])
            except ValueError:
                idx = 0

            try:
                q_plus, q_minus = reformulator.reformulate(query_text, instruction)
            except Exception as e:
                logger.error(f"Failed for {qid}: {e}")
                q_plus, q_minus = query_text, "[NONE]"
                failed += 1

            record = {
                "task_name": task_name,
                "qid": qid,
                "idx": idx,
                "query": query_text,
                "instruction": instruction,
                "query_type": query_type,
                "prompt_version": prompt_version,
                "q_plus": q_plus,
                "q_minus": q_minus,
                "reformulator": "Qwen3-0.6B",
                "created_at": datetime.now().isoformat(),
            }

            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()

            processed += 1
            if processed % 10 == 0:
                elapsed = time.time() - start_time
                speed = processed / elapsed
                eta = (new_count - processed) / speed if speed > 0 else 0
                logger.info(
                    f"Progress: {processed}/{new_count} ({processed/new_count*100:.1f}%), "
                    f"speed: {speed:.1f} q/s, ETA: {eta:.0f}s, failed: {failed}"
                )

    elapsed = time.time() - start_time
    logger.info(f"Done! Processed: {processed}, Failed: {failed}, Time: {elapsed:.1f}s")
    logger.info(f"Output: {output_file}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Qwen3-0.6B Local Reformulator")
    parser.add_argument("--task_name", type=str, required=True,
                        choices=["Core17InstructionRetrieval", "Robust04InstructionRetrieval", "News21InstructionRetrieval"])
    parser.add_argument("--model_path", type=str,
                        default="/home/luwa/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str,
                        default="dataset/FollowIR_test/dual_queries_qwen3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt_version", type=str, default=DEFAULT_PROMPT_VERSION)

    args = parser.parse_args()

    run_reformulation(
        task_name=args.task_name,
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        prompt_version=args.prompt_version,
    )
