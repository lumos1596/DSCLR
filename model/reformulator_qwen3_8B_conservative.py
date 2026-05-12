"""
Qwen3-8B Reformulator with TSC_CONSERVATIVE prompt

Designed specifically for larger models (8B+) that tend to over-extract Q_minus.
Key differences from TSC_BALANCED:
  1. Emphasizes CONSERVATIVE extraction - only direct, explicit exclusions
  2. Limits Q_minus to 1-3 keywords max
  3. Explicitly forbids inferred/implicit exclusions
  4. Requires "not relevant" / "irrelevant" exact phrasing for exclusion
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
from collections import OrderedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

TSC_CONSERVATIVE_SYSTEM = """Your task: Extract Q_plus and Q_minus from the query and instruction.

BE CONSERVATIVE WITH Q_MINUS. Only extract exclusions when the instruction uses EXACT exclusion language.

RULE FOR Q_MINUS — ONLY extract if the instruction contains ONE of these EXACT patterns:
  - "X is not relevant" / "X are not relevant"
  - "X is irrelevant"
  - "outside of [scope] is not relevant"
  - "Discussions of X are not relevant"

DO NOT extract Q_minus for:
  - Inferred or implied exclusions (e.g., if instruction says "focus on X", other topics are NOT excluded)
  - Scope hints without "not relevant" (e.g., "particularly X" does NOT exclude others)
  - Requirements phrased as conditions (e.g., "as long as X" is a requirement, not exclusion)
  - Background context that seems tangential (ALL mentioned topics are RELEVANT)
  - Contrasts or comparisons (e.g., "X rather than Y" — Y is still context, not excluded)

CRITICAL: When in doubt, Q_minus = [NONE]. False negatives (missing an exclusion) are MUCH less harmful than false positives (over-extracting).

RULE FOR Q_PLUS — Be COMPREHENSIVE:
  - Include ALL instruction content that describes what to find
  - Include background context, examples, and requirements
  - "must include X" → X goes in Q_plus
  - "X is relevant" / "X are also relevant" → X goes in Q_plus

Q_MINUS FORMAT RULES:
  - Maximum 1-3 short keywords, separated by commas
  - Each keyword: 1-4 words only
  - Do NOT add explanatory phrases or qualifiers
  - If no exclusion → Q_minus = [NONE] (with brackets)"""

TSC_CONSERVATIVE_EXAMPLES = """

Example 1 (explicit "not relevant" — YES, extract Q_minus):
Query: Evidence that radio waves from radio towers affect brain cancer.
Instruction: Relevant documents include experiments about radio waves and brain cancer. Any mentions of leukemia is not relevant.
Decision: YES — "leukemia is not relevant" is an explicit exclusion.
Output: {"Q_plus": "Association between radio waves and brain cancer", "Q_minus": "leukemia"}

Example 2 (scope limitation with "not relevant" — YES, extract Q_minus):
Query: How often were the peace talks in Ireland delayed by acts of violence?
Instruction: Any interruptions to the peace process not directly attributable to acts of violence are not relevant.
Decision: YES — "not directly attributable to acts of violence are not relevant" is an explicit exclusion.
Output: {"Q_plus": "Delays to peace talks in Ireland caused by acts of violence", "Q_minus": "non-violent interruptions"}

Example 3 (NO "not relevant" — NO exclusion):
Query: What is the status of The Three Gorges Project?
Instruction: A relevant document will provide the completion date, total cost, and electrical output.
Decision: NO — no "not relevant" phrase. The instruction only describes what IS relevant.
Output: {"Q_plus": "Status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "[NONE]"}

Example 4 (condition, not exclusion — NO Q_minus):
Query: How did France respond to protests against its nuclear testing in the South Pacific?
Instruction: A document containing information regarding any type of protest by any one is relevant as long as France's response is also included in the document.
Decision: NO — "as long as" is a condition/requirement, not an exclusion. No "not relevant" phrase.
Output: {"Q_plus": "France's response to protests against nuclear testing in the South Pacific", "Q_minus": "[NONE]"}

Example 5 (background context — ALL relevant, NO Q_minus):
Query: Find information about troops accused of war crimes in Afghanistan.
Instruction: Australian forces are accused of war crimes while on duty in Afghanistan. Some other countries are also investigating what their troops may have done. American contractors have also been accused and convicted. The ICC has been investigating these accusations.
Decision: NO — all mentioned groups are relevant context. No "not relevant" phrase.
Output: {"Q_plus": "Troops accused of war crimes in Afghanistan including Australian forces, other countries, American contractors, and ICC investigations", "Q_minus": "[NONE]"}

Example 6 (must include = requirement — NO Q_minus):
Query: Find information about increasing numbers of coyotes in suburban Maryland.
Instruction: As coyotes have moved into the area other animals such as feral cats have been driven out. Find information on the growing coyote population in Maryland and its impact on other species. Relevant documents must include references to Virginia.
Decision: NO — "must include references to Virginia" is a requirement. No "not relevant" phrase.
Output: {"Q_plus": "Increasing coyote population in suburban Maryland and impact on other species, must include references to Virginia", "Q_minus": "[NONE]"}

Example 7 (explicit "not relevant" with multiple items — YES, but keep Q_minus SHORT):
Query: What is the ongoing status of The Three Gorges Project?
Instruction: A relevant document will provide the projected or actual date of completion, its estimated or actual total cost, or the estimated or ongoing electrical output. Discussions of the social, political, or ecological impact of the project are not relevant.
Decision: YES — "social, political, or ecological impact are not relevant" is an explicit exclusion.
Output: {"Q_plus": "Ongoing status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "social, political, ecological impact"}

Example 8 (explicit "not relevant" + "are relevant" — extract ONLY the excluded part):
Query: Identify documents discussing the use of estrogen by postmenopausal women in Britain.
Instruction: The use of hormone replacement therapy outside of the United Kingdom is not relevant. United Kingdom and British development and marketing of estrogen suppressing drugs are relevant.
Decision: YES — "outside of the United Kingdom is not relevant" is an explicit exclusion. "UK development" and "British drugs" are RELEVANT.
Output: {"Q_plus": "Use of estrogen by postmenopausal women in Britain including UK hormone replacement therapy and British estrogen suppressing drugs", "Q_minus": "HRT outside UK"}

Example 9 (focus/hint but NO "not relevant" — NO Q_minus):
Query: What are the pros and cons of Great Britain's universal health care system?
Instruction: Recommendations for change and criticisms of the system are relevant. Individual's experience with the health care system is not what is sought.
Decision: NO — "is not what is sought" is a hint about focus, but does NOT use "not relevant". The individual experiences are still contextual. Be conservative.
Output: {"Q_plus": "Pros and cons of Great Britain's universal health care system including recommendations and criticisms", "Q_minus": "[NONE]"}

Example 10 (implicit scope limitation but NO "not relevant" — NO Q_minus):
Query: What progress is being made in the effort to map and sequence the human genetic code?
Instruction: Find documents that report on progress in mapping the human genome. Also of interest are new techniques and technologies being used. Applications of this research are not what is sought.
Decision: NO — "are not what is sought" is a focus hint, NOT "not relevant". Be conservative. Applications are still contextual.
Output: {"Q_plus": "Progress in mapping the human genome including new techniques and technologies", "Q_minus": "[NONE]"}"""

USER_TEMPLATE = """

Now process this:

Query: "{query}"
Instruction: "{instruction}"

Output:"""


VARIANTS = OrderedDict()

for temp in [0.1]:
    variant_id = f"TSC_CONSERVATIVE_t{str(temp).replace('.', '')}"
    VARIANTS[variant_id] = {
        'system_prompt': TSC_CONSERVATIVE_SYSTEM + TSC_CONSERVATIVE_EXAMPLES,
        'user_template': USER_TEMPLATE,
        'temperature': temp,
    }


class Qwen3Reformulator:
    def __init__(self, model_path, device="cuda", max_new_tokens=512):
        self.max_new_tokens = max_new_tokens
        logger.info(f"Loading Qwen3-8B from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if device == "cuda":
            try:
                torch.cuda._lazy_init()
                if not torch.cuda.is_available():
                    device = "cpu"
            except Exception:
                device = "cpu"

        if device == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float32, trust_remote_code=True
            ).to("cpu")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            )
        self.model.eval()
        logger.info(f"Qwen3-8B loaded on {device}")

    def reformulate(self, query, instruction, system_prompt, user_template, temperature):
        user_prompt = user_template.format(query=query, instruction=instruction)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        result_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return self._parse_result(result_text, query)

    def _parse_result(self, result_text, original_query):
        try:
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                result = json.loads(result_text[json_start:json_end])
                q_plus = result.get('Q_plus', result.get('q_plus', '')).strip()
                q_minus = result.get('Q_minus', result.get('q_minus', '')).strip()
                if q_plus:
                    if q_minus.lower() in ['none', '[none]']:
                        q_minus = '[NONE]'
                    return q_plus, (q_minus if q_minus else '[NONE]')
        except json.JSONDecodeError:
            pass

        q_plus, q_minus = original_query, '[NONE]'
        for line in result_text.split('\n'):
            line = line.strip()
            if 'Q_plus' in line or 'q_plus' in line:
                parts = line.split(':', 1)
                if len(parts) > 1 and parts[1].strip().strip('",'):
                    q_plus = parts[1].strip().strip('",')
            elif 'Q_minus' in line or 'q_minus' in line:
                parts = line.split(':', 1)
                if len(parts) > 1 and parts[1].strip().strip('",'):
                    q_minus = parts[1].strip().strip('",')
                    if q_minus.lower() in ['none', '[none]']:
                        q_minus = '[NONE]'
        return q_plus, q_minus


def load_followir_queries(task_name):
    import datasets
    path_map = {
        "Core17InstructionRetrieval": "jhu-clsp/core17-instructions-mteb",
        "Robust04InstructionRetrieval": "jhu-clsp/robust04-instructions-mteb",
        "News21InstructionRetrieval": "jhu-clsp/news21-instructions-mteb",
    }
    dataset_path = path_map.get(task_name, "")
    if not dataset_path:
        raise ValueError(f"Unknown task: {task_name}")

    ds_q = datasets.load_dataset(dataset_path, 'queries')
    ds_inst = datasets.load_dataset(dataset_path, 'instruction')

    q_split = 'queries' if 'queries' in ds_q else list(ds_q.keys())[0]
    i_split = 'instruction' if 'instruction' in ds_inst else list(ds_inst.keys())[0]

    instruction_dict = {}
    for item in ds_inst[i_split]:
        qid = str(item.get('query-id', ''))
        instruction_dict[qid] = str(item.get('instruction', ''))

    q_og, q_changed = {}, {}
    for q in ds_q[q_split]:
        full_qid = str(q.get('_id', q.get('id', '')))
        query_text = q.get('text', '')
        inst = instruction_dict.get(full_qid, "")
        if full_qid.endswith('-og'):
            q_og[full_qid] = (query_text, inst)
        elif full_qid.endswith('-changed'):
            q_changed[full_qid] = (query_text, inst)
    return q_og, q_changed


def run_variant(reformulator, variant_id, variant_config, queries, output_dir, task_name):
    system_prompt = variant_config['system_prompt']
    user_template = variant_config['user_template']
    temperature = variant_config['temperature']

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"dual_queries_{variant_id}_{task_name}.jsonl")

    if os.path.exists(output_path):
        existing = {}
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing[r['qid']] = r
        if len(existing) == len(queries):
            logger.info(f"[{variant_id}] Already completed ({len(existing)} queries), skipping")
            return output_path

    results = []
    start_time = time.time()
    failed = 0

    for i, (qid, query, instruction, query_type) in enumerate(queries):
        if not query:
            continue
        try:
            idx = int(qid.split('-')[0])
        except ValueError:
            idx = i

        try:
            q_plus, q_minus = reformulator.reformulate(
                query, instruction, system_prompt, user_template, temperature
            )
            results.append({
                "task_name": task_name,
                "qid": qid, "idx": idx, "query": query,
                "query_type": query_type, "instruction": instruction,
                "prompt_version": variant_id,
                "q_plus": q_plus, "q_minus": q_minus,
                "reformulator": f"Qwen3-8B-{variant_id}",
                "created_at": datetime.now().isoformat()
            })
        except Exception as e:
            failed += 1
            results.append({
                "task_name": task_name,
                "qid": qid, "idx": idx, "query": query,
                "query_type": query_type, "instruction": instruction,
                "prompt_version": variant_id,
                "q_plus": query, "q_minus": "[NONE]",
                "reformulator": f"Qwen3-8B-{variant_id}",
                "error": str(e),
                "created_at": datetime.now().isoformat()
            })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(queries) - i - 1) / speed if speed > 0 else 0
            logger.info(f"[{variant_id}] {i+1}/{len(queries)} ({(i+1)/len(queries)*100:.1f}%), speed: {speed:.1f} q/s, ETA: {eta:.0f}s, failed: {failed}")

    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    logger.info(f"[{variant_id}] Done! Processed: {len(results)}, Failed: {failed}, Time: {time.time()-start_time:.1f}s")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Qwen3-8B TSC_CONSERVATIVE Reformulator")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval")
    parser.add_argument("--model_path", type=str, default="/home/luwa/Documents/models/Qwen3-8B")
    parser.add_argument("--output_dir", type=str, default="dataset/FollowIR_test/dual_queries_8B")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    q_og, q_changed = load_followir_queries(args.task_name)
    all_queries = []
    for qid, (query_text, instruction) in q_og.items():
        all_queries.append((qid, query_text, instruction, "og"))
    for qid, (query_text, instruction) in q_changed.items():
        all_queries.append((qid, query_text, instruction, "changed"))

    reformulator = Qwen3Reformulator(args.model_path, args.device)

    for variant_id, variant_config in VARIANTS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running variant: {variant_id}")
        logger.info(f"{'='*60}")

        variant_output_dir = os.path.join(args.output_dir, variant_id)
        output_path = run_variant(
            reformulator, variant_id, variant_config,
            all_queries, variant_output_dir, args.task_name
        )
        logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
