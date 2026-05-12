"""
使用本地 Qwen3-0.6B 模型进行指令改写
使用改进的 v7 提示词，平衡 v4(过度生成)和 v6(过度保守)
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


SYSTEM_PROMPT_V7 = """You are an expert Information Retrieval query optimizer. Your task is to analyze a query and instruction, then extract Q_plus (what to find) and Q_minus (what to exclude).

## STEP-BY-STEP ANALYSIS (MUST FOLLOW):

### Step 1: Understand the SCOPE of the instruction
- What topics does the instruction say are relevant? These go in Q_plus.
- What topics does the instruction say are NOT relevant? These go in Q_minus.
- What topics are OUTSIDE the defined scope? These also go in Q_minus.

### Step 2: Extract Q_plus (what to find)
- Include ALL topics the instruction marks as relevant
- Include ALL topics the query asks about that the instruction confirms
- Remove format words like "documents", "articles", "reports"
- Write as a natural, fluent phrase

### Step 3: Extract Q_minus (what to exclude)
There are THREE types of exclusions to look for:

Type A - Explicit "not relevant": The instruction directly says "X is not relevant", "X are not relevant", "ignore X", "exclude X"
Type B - Scope limitation: The instruction limits the scope (e.g., "only in the UK" means outside UK is excluded; "enhanced screening" means normal screening is excluded)
Type C - Contrast exclusion: The instruction contrasts relevant vs irrelevant (e.g., "actual references count as interest while productions are irrelevant")

If NO exclusion of any type exists, output "[NONE]"

### Step 4: Verify
- Is everything in Q_plus actually relevant? (no irrelevant topics in Q_plus)
- Is everything in Q_minus actually excluded? (no relevant topics in Q_minus)
- Did I miss any relevant topics?

## CRITICAL RULES:

1. NEVER put relevant topics in Q_minus
2. If instruction says "X is relevant", X MUST go in Q_plus, NOT Q_minus
3. If instruction says "X is not relevant", X goes in Q_minus
4. If instruction limits scope, topics outside that scope go in Q_minus
5. If instruction contrasts relevant vs irrelevant, the irrelevant part goes in Q_minus
6. If truly no exclusion exists, Q_minus = "[NONE]"

## OUTPUT FORMAT (JSON):
{
  "Analysis": {
    "Relevant_Topics": "list all topics that are relevant",
    "Excluded_Topics": "list topics that are not relevant or outside scope, or NONE"
  },
  "Q_plus": "natural language query covering all relevant topics",
  "Q_minus": "excluded topics or [NONE]"
}

## EXAMPLES:

[Example 1 - Type A: Explicit "not relevant"]
Query: Evidence that radio waves from radio towers affect brain cancer.
Instruction: Relevant documents include experiments, studies about radio waves and brain cancer. Any mentions of leukemia is not relevant.

Analysis:
- Relevant_Topics: radio waves from towers/phones, brain cancer
- Excluded_Topics: leukemia (explicitly "not relevant")

Output:
{
  "Analysis": {"Relevant_Topics": "radio waves, brain cancer", "Excluded_Topics": "leukemia"},
  "Q_plus": "Association between radio waves from towers or phones and brain cancer incidence",
  "Q_minus": "leukemia"
}

[Example 2 - Type B: Scope limitation]
Query: How often were the peace talks in Ireland delayed by acts of violence?
Instruction: Any interruptions to the peace process not directly attributable to acts of violence are not relevant.

Analysis:
- Relevant_Topics: peace talks in Ireland, delays caused by acts of violence
- Excluded_Topics: non-violent interruptions (scope limited to violence-caused delays)

Output:
{
  "Analysis": {"Relevant_Topics": "Irish peace talks, violence-caused delays", "Excluded_Topics": "non-violent interruptions"},
  "Q_plus": "Frequency of delays or disruptions to peace talks in Ireland resulting from acts of violence",
  "Q_minus": "non-violent interruptions"
}

[Example 3 - Type C: Contrast exclusion]
Query: Is there contemporary interest in the Greek philosophy of stoicism?
Instruction: Actual references to the philosophy or philosophers in books or by professors counts as interest while productions of Greek stoic plays and new stoic artistic productions are all irrelevant.

Analysis:
- Relevant_Topics: contemporary interest in Greek stoicism philosophy, references by professors, books about the philosophy
- Excluded_Topics: Greek stoic plays, stoic artistic productions (contrasted as irrelevant)

Output:
{
  "Analysis": {"Relevant_Topics": "contemporary interest in stoicism philosophy, references in books, professors", "Excluded_Topics": "Greek stoic plays, stoic artistic productions"},
  "Q_plus": "Contemporary interest in the Greek philosophy of stoicism including references in books and by professors",
  "Q_minus": "Greek stoic plays, stoic artistic productions, modern people described as stoics"
}

[Example 4 - Type B: Geographic scope]
Query: Identify documents discussing the use of estrogen by postmenopausal women in Britain.
Instruction: The use of hormone replacement therapy outside of the United Kingdom is not relevant. United Kingdom and British development and marketing of estrogen suppressing drugs are relevant.

Analysis:
- Relevant_Topics: estrogen use by postmenopausal women in Britain, hormone replacement therapy in UK, UK development of estrogen suppressing drugs
- Excluded_Topics: hormone replacement therapy outside UK (geographic scope)

Output:
{
  "Analysis": {"Relevant_Topics": "estrogen use in Britain, hormone replacement therapy in UK, UK development of estrogen suppressing drugs", "Excluded_Topics": "hormone replacement therapy outside UK"},
  "Q_plus": "Use of estrogen and hormone replacement therapy by postmenopausal women in Britain, including UK development of estrogen suppressing drugs",
  "Q_minus": "United States, global, international"
}

[Example 5 - Type B: Scope limitation - enhanced vs normal]
Query: A relevant document would discuss the effectiveness of efforts to better scrutinize passengers and luggage.
Instruction: A relevant document would contain reports on what steps airports worldwide have taken to better scrutinize passengers and their luggage on domestic and international flights and to step up screening of all persons and goods boarding aircraft.

Analysis:
- Relevant_Topics: enhanced passenger and luggage scrutiny at airports worldwide, stepped-up screening
- Excluded_Topics: normal security screening methods (instruction focuses on "better scrutinize" and "step up"), mere mention without details

Output:
{
  "Analysis": {"Relevant_Topics": "enhanced passenger scrutiny at airports, stepped-up screening", "Excluded_Topics": "normal screening, mere mention of enhanced security"},
  "Q_plus": "Effectiveness of enhanced passenger and luggage scrutiny at airports worldwide, particularly for international flights",
  "Q_minus": "normal security screening methods, mere mention of enhanced security"
}

[Example 6 - NO Exclusion]
Query: What is the status of The Three Gorges Project?
Instruction: A relevant document will provide the completion date, total cost, and electrical output.

Analysis:
- Relevant_Topics: Three Gorges Project status, completion date, cost, electrical output
- Excluded_Topics: NONE (no exclusion stated, no scope limitation)

Output:
{
  "Analysis": {"Relevant_Topics": "Three Gorges Project, completion date, cost, output", "Excluded_Topics": "NONE"},
  "Q_plus": "Status of the Three Gorges Project including completion date, total cost, and electrical output",
  "Q_minus": "[NONE]"
}"""

USER_PROMPT_TEMPLATE_V7 = """Now analyze this query and instruction:

Query: "{query}"
Instruction: "{instruction}"

Follow the step-by-step analysis and output JSON:"""


class Qwen3LocalReformulatorV7:
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

        self.system_prompt = SYSTEM_PROMPT_V7
        self.user_prompt_template = USER_PROMPT_TEMPLATE_V7

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
                q_minus = result.get('q_minus', result.get('Q_minus', '')).strip()

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
            elif line.startswith('"Q_minus"') or line.startswith('Q_minus') or line.startswith('"q_minus"') or line.startswith('q_minus'):
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
    parser = argparse.ArgumentParser(description="Qwen3-0.6B Local Reformulator V7")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="/home/luwa/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default="dataset/FollowIR_test/dual_queries_qwen3_v7")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"dual_queries_qwen3_v7_{args.task_name}.jsonl")

    q_og, q_changed = load_followir_queries(args.task_name)

    reformulator = Qwen3LocalReformulatorV7(
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
                "prompt_version": "v7",
                "q_plus": q_plus,
                "q_minus": q_minus,
                "reformulator": "Qwen3-0.6B-v7",
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
                "prompt_version": "v7",
                "q_plus": query,
                "q_minus": "[NONE]",
                "reformulator": "Qwen3-0.6B-v7",
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
