"""
使用本地 Qwen3-0.6B 模型进行指令改写
V8 混合方法：规则提取 Q_minus + 模型生成 Q_plus

核心思路：
- Q_minus 用规则从 instruction 中提取（更可靠，0.6B 模型推理能力不足）
- Q_plus 用模型生成（模型擅长理解和重组语义）
"""

import os
import re
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_V8 = """You are an expert Information Retrieval query optimizer. Your task is to analyze a query and instruction, then create Q_plus (what to find).

## Rules:
1. Extract ALL topics the instruction marks as relevant
2. Include ALL relevant topics mentioned, do not miss any
3. Remove format words like "documents", "articles", "reports", "studies"
4. Write as a natural, fluent phrase
5. If instruction limits scope (e.g., "only in UK", "enhanced screening"), reflect that in Q_plus

## Output format (JSON):
{"Q_plus": "your optimized query here"}"""

USER_PROMPT_TEMPLATE_V8 = """Query: "{query}"
Instruction: "{instruction}"

Create Q_plus:"""


def _clean_exclusion(text: str) -> str:
    text = re.sub(r'^(any\s+)?(?:mentions?\s+(?:of\s+)?)?(?:references?\s+(?:to\s+)?)?(?:discussions?\s+(?:of\s+|about\s+)?)?(?:documents?\s+(?:about\s+|on\s+)?)?(?:information\s+(?:about\s+|on\s+)?)?(?:details?\s+(?:about\s+|on\s+)?)?(?:accounts?\s+(?:of\s+)?)?(?:reports?\s+(?:of\s+|on\s+)?)?(?:descriptions?\s+(?:of\s+)?)?', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'^(that\s+)?(?:discuss(?:es|ed|ing)?)\s+', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\s+(?:is|are)\s+(?:all\s+)?(?:irrelevant|not\s+relevant)\s*$', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\s*,\s*$', '', text)
    text = re.sub(r'^\s*,\s*', '', text)
    if len(text) > 120:
        sentences = re.split(r'[.;]', text)
        text = sentences[0].strip()
    return text


def extract_q_minus_rule_based(instruction: str) -> str:
    if not instruction:
        return "[NONE]"

    exclusions = []

    neg_sent_patterns = [
        r'(?:Any|All|Any and all)\s+(?:mentions?\s+(?:of\s+)?|references?\s+(?:to\s+)?|discussions?\s+(?:of\s+|about\s+)?|documents?\s+(?:about\s+|on\s+)?|information\s+(?:about\s+|on\s+)?|details?\s+(?:about\s+|on\s+)?)?(.+?)\s+(?:is|are)\s+not\s+relevant',
        r'(.+?)\s+(?:is|are)\s+(?:all\s+)?irrelevant',
        r'(.+?)\s+(?:is|are)\s+not\s+relevant',
    ]

    for pattern in neg_sent_patterns:
        matches = re.finditer(pattern, instruction, re.IGNORECASE)
        for m in matches:
            excluded = m.group(1).strip()
            excluded = _clean_exclusion(excluded)
            if excluded and len(excluded) > 2 and excluded not in exclusions:
                exclusions.append(excluded)

    while_pattern = r'while\s+(.+?)\s+(?:is|are)\s+(?:all\s+)?(?:irrelevant|not\s+relevant)'
    while_matches = re.finditer(while_pattern, instruction, re.IGNORECASE)
    for m in while_matches:
        excluded = m.group(1).strip()
        excluded = _clean_exclusion(excluded)
        if excluded and len(excluded) > 2 and excluded not in exclusions:
            exclusions.append(excluded)

    nor_pattern = r',?\s*nor\s+(?:are\s+)?(?:mentions?\s+(?:of\s+)?|references?\s+(?:to\s+)?|discussions?\s+(?:of\s+|about\s+)?)?(.+?)(?:\.|;|$)'
    nor_matches = re.finditer(nor_pattern, instruction, re.IGNORECASE)
    for m in nor_matches:
        excluded = m.group(1).strip().rstrip('.')
        excluded = _clean_exclusion(excluded)
        if excluded and len(excluded) > 2 and excluded not in exclusions:
            exclusions.append(excluded)

    not_directly_pattern = r'(.+?)\s+not\s+directly\s+(?:attributable|related|connected)\s+to\s+(.+?)\s+(?:is|are)\s+not\s+relevant'
    not_directly_matches = re.finditer(not_directly_pattern, instruction, re.IGNORECASE)
    for m in not_directly_matches:
        excluded = m.group(1).strip()
        excluded = _clean_exclusion(excluded)
        if excluded and len(excluded) > 2 and excluded not in exclusions:
            exclusions.append(excluded)

    do_not_pattern = r'(?:do\s+not|does\s+not)\s+(?:include|consider|count)\s+(.+?)(?:\.|,|;|$)'
    do_not_matches = re.finditer(do_not_pattern, instruction, re.IGNORECASE)
    for m in do_not_matches:
        excluded = m.group(1).strip().rstrip('.,;')
        excluded = _clean_exclusion(excluded)
        if excluded and len(excluded) > 2 and excluded not in exclusions:
            exclusions.append(excluded)

    cleaned = []
    for e in exclusions:
        if len(e) > 100:
            continue
        if any(kw in e.lower() for kw in ['relevant document', 'a relevant', 'relevant doc']):
            continue
        cleaned.append(e)

    if not cleaned:
        return "[NONE]"

    return ", ".join(cleaned)


class Qwen3LocalReformulatorV8:
    def __init__(
        self,
        model_path: str = "/home/luwa/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B",
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ):
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

        self.system_prompt = SYSTEM_PROMPT_V8
        self.user_prompt_template = USER_PROMPT_TEMPLATE_V8

    def reformulate(self, query: str, instruction: str) -> Tuple[str, str]:
        q_minus = extract_q_minus_rule_based(instruction)

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

        q_plus = self._parse_q_plus(result_text, query)
        return q_plus, q_minus

    def _parse_q_plus(self, result_text: str, original_query: str) -> str:
        try:
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                result = json.loads(json_str)
                q_plus = result.get('Q_plus', result.get('q_plus', '')).strip()
                if q_plus:
                    return q_plus
        except json.JSONDecodeError:
            pass

        for line in result_text.split('\n'):
            line = line.strip()
            if 'Q_plus' in line or 'q_plus' in line:
                parts = line.split(':', 1)
                if len(parts) > 1:
                    val = parts[1].strip().strip('",')
                    if val:
                        return val

        return original_query


def load_followir_queries(task_name: str):
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


def main():
    parser = argparse.ArgumentParser(description="Qwen3-0.6B Local Reformulator V8 (Hybrid)")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="/home/luwa/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default="dataset/FollowIR_test/dual_queries_qwen3_v8")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"dual_queries_qwen3_v8_{args.task_name}.jsonl")

    q_og, q_changed = load_followir_queries(args.task_name)

    reformulator = Qwen3LocalReformulatorV8(
        model_path=args.model_path,
        device=args.device,
    )

    all_queries = []
    for qid, (query_text, instruction) in q_og.items():
        all_queries.append((qid, query_text, instruction, "og"))
    for qid, (query_text, instruction) in q_changed.items():
        all_queries.append((qid, query_text, instruction, "changed"))

    results = []
    start_time = time.time()
    failed = 0

    for i, (qid, query, instruction, query_type) in enumerate(all_queries):
        if not query:
            continue

        try:
            idx = int(qid.split('-')[0])
        except ValueError:
            idx = i

        try:
            q_plus, q_minus = reformulator.reformulate(query, instruction)
            results.append({
                "task_name": args.task_name,
                "qid": qid,
                "idx": idx,
                "query": query,
                "query_type": query_type,
                "instruction": instruction,
                "prompt_version": "v8",
                "q_plus": q_plus,
                "q_minus": q_minus,
                "reformulator": "Qwen3-0.6B-v8-hybrid",
                "created_at": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Failed to process {qid}: {e}")
            failed += 1
            results.append({
                "task_name": args.task_name,
                "qid": qid,
                "idx": idx,
                "query": query,
                "query_type": query_type,
                "instruction": instruction,
                "prompt_version": "v8",
                "q_plus": query,
                "q_minus": extract_q_minus_rule_based(instruction),
                "reformulator": "Qwen3-0.6B-v8-hybrid",
                "error": str(e),
                "created_at": datetime.now().isoformat()
            })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(all_queries) - i - 1) / speed if speed > 0 else 0
            logger.info(f"Progress: {i+1}/{len(all_queries)} ({(i+1)/len(all_queries)*100:.1f}%), speed: {speed:.1f} q/s, ETA: {eta:.0f}s, failed: {failed}")

    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    logger.info(f"Done! Processed: {len(results)}, Failed: {failed}, Time: {time.time() - start_time:.1f}s")
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
