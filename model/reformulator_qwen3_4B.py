"""
Qwen3-4B Reformulator with TSC_BALANCED prompt (best from 1.7B optimization)

Only uses TSC_BALANCED strategy - the best performing prompt from 1.7B experiments.
Model: Qwen3-4B (local path: /home/luwa/Documents/models/Qwen3-4B)
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


TSC_BALANCED_SYSTEM = """Your task: Decide whether the instruction contains any exclusion information, then extract Q_plus and Q_minus.

STEP 1 — Binary decision: Does the instruction explicitly state that something is NOT relevant, irrelevant, or outside scope?

YES signals (exclusion EXISTS):
  - "X is not relevant" / "X is irrelevant" / "X are not relevant"
  - "outside of [scope]" / "not directly attributable to"
  - "Discussions of X are not relevant"
  - "only in [region]" (implies other regions excluded)

NO signals (NO exclusion — these are all RELEVANT content descriptions):
  - "A relevant document will include/provide/contain X" → X is what to FIND
  - "Particularly sought are X" → X is what to FIND
  - "Relevant documents may identify X" → X is RELEVANT
  - "must include X" / "must quote X" / "must discuss X" → X is a REQUIREMENT, not exclusion
  - "Documents must include X" → X is REQUIRED content
  - The instruction provides background context or examples → ALL context is RELEVANT
  - The instruction lists subtopics or details → ALL are RELEVANT
  - "X are also relevant" / "X is relevant as well" → X is RELEVANT

CRITICAL DISTINCTION:
  - "X is not relevant" → X goes in Q_minus (excluded)
  - "X is relevant" / "X must be included" → X goes in Q_plus (required)
  - Instruction describes background → background is RELEVANT context
  - Instruction says "must include X" → X is a requirement, NOT an exclusion

If NO → Q_minus = [NONE]. Put ALL instruction content into Q_plus.
If YES → Extract ONLY the explicitly excluded topics into Q_minus as short keywords.

Output JSON: {"Q_plus": "...", "Q_minus": "[NONE]"} or {"Q_plus": "...", "Q_minus": "keyword1, keyword2"}

FORMAT RULES:
- Q_minus must be exactly [NONE] (with brackets) when no exclusion exists
- Never write "none" or "NONE" without brackets
- Q_minus uses only short keywords (2-5 words per item)
- Q_minus must NOT contain anything that is also in Q_plus"""

TSC_BALANCED_EXAMPLES = """

Example 1 (explicit "not relevant"):
Query: Evidence that radio waves from radio towers affect brain cancer.
Instruction: Relevant documents include experiments about radio waves and brain cancer. Any mentions of leukemia is not relevant.
Decision: YES — "leukemia is not relevant" is an exclusion signal.
Output: {"Q_plus": "Association between radio waves and brain cancer", "Q_minus": "leukemia"}

Example 2 (scope limitation):
Query: How often were the peace talks in Ireland delayed by acts of violence?
Instruction: Any interruptions to the peace process not directly attributable to acts of violence are not relevant.
Decision: YES — "not directly attributable to acts of violence are not relevant" is an exclusion signal.
Output: {"Q_plus": "Delays to peace talks in Ireland caused by acts of violence", "Q_minus": "non-violent interruptions"}

Example 3 (NO exclusion — instruction only describes relevant content):
Query: What is the status of The Three Gorges Project?
Instruction: A relevant document will provide the completion date, total cost, and electrical output.
Decision: NO — the instruction only describes what IS relevant. No exclusion signal.
Output: {"Q_plus": "Status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "[NONE]"}

Example 4 (NO exclusion — "must include" is a requirement):
Query: How did France respond to protests against its nuclear testing in the South Pacific?
Instruction: A document containing information regarding any type of protest by any one is relevant as long as France's response is also included in the document.
Decision: NO — "as long as France's response is also included" is a REQUIREMENT, not an exclusion. No exclusion signal.
Output: {"Q_plus": "France's response to protests against nuclear testing in the South Pacific", "Q_minus": "[NONE]"}

Example 5 (NO exclusion — instruction provides background context):
Query: Find information about troops accused of war crimes in Afghanistan.
Instruction: Australian forces are accused of war crimes while on duty in Afghanistan. Some other countries are also investigating what their troops may have done. American contractors have also been accused and convicted. The ICC has been investigating these accusations.
Decision: NO — all mentioned groups are RELEVANT examples and context. "Also investigating" and "also been accused" signal RELEVANCE, not exclusion. No exclusion signal.
Output: {"Q_plus": "Troops accused of war crimes in Afghanistan including Australian forces, other countries, American contractors, and ICC investigations", "Q_minus": "[NONE]"}

Example 6 (NO exclusion — "must include" requirement):
Query: Find information about increasing numbers of coyotes in suburban Maryland.
Instruction: As coyotes have moved into the area other animals such as feral cats have been driven out. Find information on the growing coyote population in Maryland and its impact on other species. Relevant documents must include references to Virginia.
Decision: NO — "must include references to Virginia" is a REQUIREMENT, not an exclusion. Virginia is RELEVANT. No exclusion signal.
Output: {"Q_plus": "Increasing coyote population in suburban Maryland and impact on other species, must include references to Virginia", "Q_minus": "[NONE]"}

Example 7 (YES — explicit "not relevant" with multiple items):
Query: What is the ongoing status of The Three Gorges Project?
Instruction: A relevant document will provide the projected or actual date of completion, its estimated or actual total cost, or the estimated or ongoing electrical output. Discussions of the social, political, or ecological impact of the project are not relevant.
Decision: YES — "social, political, or ecological impact are not relevant" is an exclusion signal.
Output: {"Q_plus": "Ongoing status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "social impact, political impact, ecological impact"}

Example 8 (YES — scope limitation + "are relevant"):
Query: Identify documents discussing the use of estrogen by postmenopausal women in Britain.
Instruction: The use of hormone replacement therapy outside of the United Kingdom is not relevant. United Kingdom and British development and marketing of estrogen suppressing drugs are relevant.
Decision: YES — "outside of the United Kingdom is not relevant" is an exclusion signal. "UK development" and "British drugs" are RELEVANT.
Output: {"Q_plus": "Use of estrogen by postmenopausal women in Britain including UK hormone replacement therapy and British estrogen suppressing drugs", "Q_minus": "hormone replacement therapy outside UK"}

Example 9 (NO exclusion — instruction describes what to find):
Query: What steps have been taken world-wide by those bearing the cost of E-mail to prevent excesses?
Instruction: To be relevant, a document will concern dissatisfaction by an entity paying for the cost of electronic mail. Particularly sought are items which relate to system users who abuse the system.
Decision: NO — "dissatisfaction" and "system users who abuse" are what to FIND. "Particularly sought" means RELEVANT. No exclusion signal.
Output: {"Q_plus": "Steps taken to prevent email abuses by cost-bearing entities, including dissatisfaction and system user abuses", "Q_minus": "[NONE]"}

Example 10 (NO exclusion — "must include" + background):
Query: I'm looking for information pertaining to Ethiopia's Abiy Ahmed's winning of the Nobel Peace Prize.
Instruction: Please provide details explaining what accomplishments led to Abiy Ahmed being awarded the Nobel Peace Prize. Background information regarding the history of the Ethiopia-Eritrea conflict are relevant. Eritrean reaction to the award and information about current relations between the two countries is relevant as well. Documents must include context that mentions death.
Decision: NO — "are relevant" and "is relevant as well" signal RELEVANCE. "must include context that mentions death" is a REQUIREMENT. No exclusion signal.
Output: {"Q_plus": "Abiy Ahmed Nobel Peace Prize including accomplishments, Ethiopia-Eritrea conflict background, Eritrean reaction, current relations, and death context", "Q_minus": "[NONE]"}"""


USER_TEMPLATE = """

Now process this:

Query: "{query}"
Instruction: "{instruction}"

Output:"""


VARIANTS = OrderedDict()

for temp in [0.1]:
    variant_id = f"TSC_BALANCED_t{str(temp).replace('.', '')}"
    VARIANTS[variant_id] = {
        'system_prompt': TSC_BALANCED_SYSTEM + TSC_BALANCED_EXAMPLES,
        'user_template': USER_TEMPLATE,
        'temperature': temp,
    }


class Qwen3Reformulator:
    def __init__(self, model_path, device="cuda", max_new_tokens=512):
        self.max_new_tokens = max_new_tokens
        logger.info(f"Loading Qwen3-4B from {model_path}...")
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
        logger.info(f"Qwen3-4B loaded on {device}")

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
                "reformulator": f"Qwen3-4B-{variant_id}",
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
                "reformulator": f"Qwen3-4B-{variant_id}",
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
    parser = argparse.ArgumentParser(description="Qwen3-4B TSC_BALANCED Reformulator")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval")
    parser.add_argument("--model_path", type=str, default="/home/luwa/Documents/models/Qwen3-4B")
    parser.add_argument("--output_dir", type=str, default="dataset/FollowIR_test/dual_queries_4B")
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
