"""
使用本地 Qwen3-0.6B 模型进行指令改写
使用改进的 v6 提示词，专门针对小模型优化
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

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_V6 = """You are an expert Information Retrieval query optimizer. Your task is to analyze a query and instruction, then extract Q_plus (what to find) and Q_minus (what to exclude).

## STEP-BY-STEP ANALYSIS (MUST FOLLOW):

### Step 1: Identify RELEVANT Topics (for Q_plus)
- Read the instruction carefully
- Extract ALL topics that are marked as "relevant" or should be included
- Include ALL relevant topics mentioned, do not miss any
- Remove format words like "documents", "articles", "reports", "studies"
- Write as a natural, fluent phrase

### Step 2: Identify EXPLICIT EXCLUSIONS (for Q_minus)
- Look for phrases like "is not relevant", "are not relevant", "ignore", "exclude"
- ONLY extract what is EXPLICITLY marked as irrelevant
- If NO explicit exclusion exists, output "[NONE]"
- DO NOT put relevant topics in Q_minus!

### Step 3: Verify Before Output
- Check: Is everything in Q_plus actually relevant according to instruction?
- Check: Is everything in Q_minus actually marked as irrelevant?
- Check: Did I miss any relevant topics that should be in Q_plus?

## CRITICAL WARNINGS:

1. NEVER put relevant topics in Q_minus
2. If instruction says "X is relevant", X MUST go in Q_plus, NOT Q_minus
3. If instruction says "X is not relevant", X goes in Q_minus
4. If no explicit "not relevant" statement, Q_minus = "[NONE]"
5. Do NOT generate Q_minus content just to fill space

## OUTPUT FORMAT (JSON):
{
  "Analysis": {
    "Relevant_Topics": "list all topics marked as relevant",
    "Explicit_Exclusions": "list topics marked as not relevant, or NONE"
  },
  "Q_plus": "natural language query covering all relevant topics",
  "Q_minus": "excluded topics or [NONE]"
}

## EXAMPLES:

[Example 1 - Clear Exclusion]
Query: Evidence that radio waves from radio towers affect brain cancer.
Instruction: Relevant documents include experiments, studies about radio waves and brain cancer. Any mentions of leukemia is not relevant.

Analysis:
- Relevant_Topics: radio waves from towers/phones, brain cancer
- Explicit_Exclusions: leukemia

Output:
{
  "Analysis": {"Relevant_Topics": "radio waves, brain cancer", "Explicit_Exclusions": "leukemia"},
  "Q_plus": "Association between radio waves from towers or phones and brain cancer incidence",
  "Q_minus": "leukemia"
}

[Example 2 - Multiple Relevant Topics, NO Exclusion]
Query: Identify documents discussing the use of estrogen by postmenopausal women in Britain.
Instruction: The use of hormone replacement therapy outside of the United Kingdom is not relevant. United Kingdom and British development and marketing of estrogen suppressing drugs are relevant.

Analysis:
- Relevant_Topics: estrogen use by postmenopausal women in Britain, hormone replacement therapy, UK/British development of estrogen suppressing drugs
- Explicit_Exclusions: hormone replacement therapy OUTSIDE UK (not the therapy itself!)

Output:
{
  "Analysis": {"Relevant_Topics": "estrogen use in Britain, hormone replacement therapy, UK development of estrogen suppressing drugs", "Explicit_Exclusions": "hormone replacement therapy outside UK"},
  "Q_plus": "Use of estrogen and hormone replacement therapy by postmenopausal women in Britain, including UK development of estrogen suppressing drugs",
  "Q_minus": "United States, global, international"
}

[Example 3 - NO Exclusion Case]
Query: What is the status of The Three Gorges Project?
Instruction: A relevant document will provide the completion date, total cost, and electrical output.

Analysis:
- Relevant_Topics: Three Gorges Project status, completion date, cost, electrical output
- Explicit_Exclusions: NONE (no "not relevant" statement)

Output:
{
  "Analysis": {"Relevant_Topics": "Three Gorges Project, completion date, cost, output", "Explicit_Exclusions": "NONE"},
  "Q_plus": "Status of the Three Gorges Project including completion date, total cost, and electrical output",
  "Q_minus": "[NONE]"
}

[Example 4 - Complex Instruction]
Query: Identify documents discussing space-borne ocean remote sensing.
Instruction: Documents about oceanography and seabed prospecting are relevant. Documents about geography, agriculture, forestry, mining are not relevant. Marketing and temperature references are not relevant.

Analysis:
- Relevant_Topics: space-borne ocean remote sensing, oceanography, seabed prospecting
- Explicit_Exclusions: geography, agriculture, forestry, mining, marketing, temperature

Output:
{
  "Analysis": {"Relevant_Topics": "space-borne ocean remote sensing, oceanography, seabed prospecting", "Explicit_Exclusions": "geography, agriculture, forestry, mining, marketing, temperature"},
  "Q_plus": "Development and application of space-borne ocean remote sensing in oceanography and seabed prospecting",
  "Q_minus": "land-bound sciences, agriculture, forestry, mining, marketing, temperature"
}

Remember: When in doubt, Q_minus should be "[NONE]" rather than guessing."""

USER_PROMPT_TEMPLATE_V6 = """Now analyze this query and instruction:

Query: "{query}"
Instruction: "{instruction}"

Follow the step-by-step analysis and output JSON:"""


class Qwen3LocalReformulatorV6:
    def __init__(
        self,
        model_path: str = "/home/luwa/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B",
        device: str = "cuda",
        max_new_tokens: int = 1024,
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

        self.system_prompt = SYSTEM_PROMPT_V6
        self.user_prompt_template = USER_PROMPT_TEMPLATE_V6

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

    def _parse_result(self, result_text: str, original_query: str) -> Tuple[str, str]:
        try:
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                result = json.loads(json_str)
                
                q_plus = result.get('Q_plus', '').strip()
                q_minus = result.get('Q_minus', '').strip()
                
                if not q_plus:
                    q_plus = original_query
                if not q_minus:
                    q_minus = '[NONE]'
                    
                return q_plus, q_minus
        except json.JSONDecodeError:
            pass
        
        q_plus_match = result_text
        q_minus_match = '[NONE]'
        
        for line in result_text.split('\n'):
            line = line.strip()
            if line.startswith('"Q_plus"') or line.startswith('Q_plus'):
                try:
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        q_plus_match = parts[1].strip().strip('",')
                except:
                    pass
            elif line.startswith('"Q_minus"') or line.startswith('Q_minus'):
                try:
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        q_minus_match = parts[1].strip().strip('",')
                except:
                    pass
        
        return q_plus_match if q_plus_match else original_query, q_minus_match if q_minus_match else '[NONE]'


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
    parser = argparse.ArgumentParser(description="Qwen3-0.6B Local Reformulator V6")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="/home/luwa/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default="dataset/FollowIR_test/dual_queries_qwen3_v6")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"dual_queries_qwen3_v6_{args.task_name}.jsonl")

    q_og, q_changed = load_followir_queries(args.task_name)

    reformulator = Qwen3LocalReformulatorV6(
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
                "prompt_version": "v6",
                "q_plus": q_plus,
                "q_minus": q_minus,
                "reformulator": "Qwen3-0.6B-v6",
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
                "prompt_version": "v6",
                "q_plus": query,
                "q_minus": "[NONE]",
                "reformulator": "Qwen3-0.6B-v6",
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
