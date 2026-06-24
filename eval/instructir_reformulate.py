"""
InstructIR Dual Query Generation with Qwen3-4B

Generates Q_plus and Q_minus for InstructIR queries using TSC_BALANCED prompt.
InstructIR has per-query instance-specific instructions, so we use the FollowIR-style
prompt that takes both query and instruction.

Usage:
  cd /home/luwa/Documents/DSCLR && /home/luwa/.conda/envs/dsclr/bin/python -m eval.instructir_reformulate \
    --model_path /home/luwa/Documents/models/Qwen3-4B --device cuda \
    --output_dir dataset/InstructIR/dual_queries
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.pop("HF_ENDPOINT", None)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
from collections import OrderedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import datasets

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

Example 6 (InstructIR-style instruction — user context, no exclusion):
Query: where is padre island tx
Instruction: I am a geography teacher preparing for my next class on U.S. coastal regions, and I want to find the precise location of Padre Island in Texas.
Decision: NO — the instruction provides user context (geography teacher) and search goal (precise location). No exclusion signal.
Output: {"Q_plus": "Precise geographical location of Padre Island in Texas for U.S. coastal regions study", "Q_minus": "[NONE]"}

Example 7 (InstructIR-style instruction — with implicit exclusion):
Query: wine cabinets definition
Instruction: I am an interior designer focused on high-end kitchen remodels, and my current project includes a request for a custom wine storage solution. I need to search for the definition of "wine cabinets" to ensure that my understanding aligns with industry standards and that the furniture I suggest will meet my client's expectations in terms of form, function, and climate control features.
Decision: NO — the instruction specifies user context (interior designer) and search goal (definition of wine cabinets for high-end kitchen). All content is RELEVANT. No exclusion signal.
Output: {"Q_plus": "Definition of wine cabinets for high-end kitchen remodels including industry standards for form, function, and climate control features", "Q_minus": "[NONE]"}"""

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
    def __init__(self, model_path, device="cuda", max_new_tokens=512, batch_size=16):
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        logger.info(f"Loading model from {model_path} (batch_size={batch_size})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

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
                model_path, torch_dtype=torch.float16, trust_remote_code=True
            ).cuda()
        self.model.eval()
        self.device = self.model.device
        logger.info(f"Model loaded on {self.device}")

    def _build_prompt(self, query, instruction, system_prompt, user_template):
        user_prompt = user_template.format(query=query, instruction=instruction)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def reformulate_batch(self, queries, instructions, system_prompt, user_template, temperature):
        prompts = [
            self._build_prompt(q, i, system_prompt, user_template)
            for q, i in zip(queries, instructions)
        ]
        all_q_plus = []
        all_q_minus = []

        for batch_start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[batch_start:batch_start + self.batch_size]
            batch_queries = queries[batch_start:batch_start + self.batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            for j, output in enumerate(outputs):
                input_len = inputs["input_ids"].shape[1]
                generated_ids = output[input_len:]
                result_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                q_plus, q_minus = self._parse_result(result_text, batch_queries[j])
                all_q_plus.append(q_plus)
                all_q_minus.append(q_minus)

        return all_q_plus, all_q_minus

    def reformulate(self, query, instruction, system_prompt, user_template, temperature):
        q_plus_list, q_minus_list = self.reformulate_batch(
            [query], [instruction], system_prompt, user_template, temperature
        )
        return q_plus_list[0], q_minus_list[0]

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


def load_instructir_queries() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, int]]]:
    logger.info("Loading InstructIR dataset from mteb/InstructIR-mteb...")

    ds_queries = datasets.load_dataset('mteb/InstructIR-mteb', 'queries')['queries']
    ds_inst = datasets.load_dataset('mteb/InstructIR-mteb', 'instruction')['instruction']
    ds_qrels = datasets.load_dataset('mteb/InstructIR-mteb', 'default', split='test')

    queries = {}
    for q in ds_queries:
        qid = str(q['_id'])
        queries[qid] = str(q.get('text', ''))

    instructions = {}
    for item in ds_inst:
        qid = str(item['query-id'])
        instructions[qid] = str(item.get('instruction', ''))

    qrels = {}
    for item in ds_qrels:
        qid = str(item['query-id'])
        doc_id = str(item['corpus-id'])
        score = int(item.get('score', 1))
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][doc_id] = score

    logger.info(f"Loaded {len(queries)} queries, {len(instructions)} instructions, {len(qrels)} qrels")

    qrel_qids = set(qrels.keys())
    query_qids = set(queries.keys())
    eval_qids = qrel_qids & query_qids
    logger.info(f"Queries with qrels: {len(eval_qids)}")

    return queries, instructions, qrels


def run_variant(reformulator, variant_id, variant_config, queries, instructions, qrels, output_dir):
    system_prompt = variant_config['system_prompt']
    user_template = variant_config['user_template']
    temperature = variant_config['temperature']
    batch_size = reformulator.batch_size

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"InstructIR_{variant_id}.jsonl")

    existing = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing[r['qid']] = r
        logger.info(f"[{variant_id}] Found {len(existing)} existing entries")

    eval_qids = sorted(set(qrels.keys()) & set(queries.keys()))

    results = []
    pending_qids = []
    pending_queries = []
    pending_instructions = []

    for qid in eval_qids:
        query_text = queries[qid]
        instruction = instructions.get(qid, "")
        if not query_text:
            continue
        if qid in existing:
            results.append(existing[qid])
        else:
            pending_qids.append(qid)
            pending_queries.append(query_text)
            pending_instructions.append(instruction)

    skipped = len(results)
    logger.info(f"[{variant_id}] Skipped {skipped} existing, processing {len(pending_qids)} new queries in batches of {batch_size}")

    start_time = time.time()
    failed = 0

    for batch_start in tqdm(range(0, len(pending_qids), batch_size), desc=f"Reformulating InstructIR"):
        batch_end = min(batch_start + batch_size, len(pending_qids))
        b_qids = pending_qids[batch_start:batch_end]
        b_queries = pending_queries[batch_start:batch_end]
        b_instructions = pending_instructions[batch_start:batch_end]

        try:
            q_plus_list, q_minus_list = reformulator.reformulate_batch(
                b_queries, b_instructions, system_prompt, user_template, temperature
            )
            for j, qid in enumerate(b_qids):
                results.append({
                    "task_name": "InstructIR",
                    "qid": qid,
                    "query": b_queries[j],
                    "instruction": b_instructions[j],
                    "prompt_version": variant_id,
                    "q_plus": q_plus_list[j],
                    "q_minus": q_minus_list[j],
                    "reformulator": f"Qwen3-4B-{variant_id}",
                    "created_at": datetime.now().isoformat()
                })
        except Exception as e:
            failed += len(b_qids)
            logger.warning(f"Batch failed ({len(b_qids)} queries): {e}")
            for j, qid in enumerate(b_qids):
                results.append({
                    "task_name": "InstructIR",
                    "qid": qid,
                    "query": b_queries[j],
                    "instruction": b_instructions[j],
                    "prompt_version": variant_id,
                    "q_plus": b_queries[j],
                    "q_minus": "[NONE]",
                    "reformulator": f"Qwen3-4B-{variant_id}",
                    "error": str(e),
                    "created_at": datetime.now().isoformat()
                })

        processed = batch_end
        if processed % (batch_size * 10) == 0 or processed == len(pending_qids):
            with open(output_path, 'w') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')

            elapsed = time.time() - start_time
            if elapsed > 0:
                speed = processed / elapsed
                remaining = len(pending_qids) - processed
                eta = remaining / speed if speed > 0 else 0
                logger.info(
                    f"[{variant_id}] {processed}/{len(pending_qids)} "
                    f"(failed: {failed}), speed: {speed:.1f} q/s, ETA: {eta:.0f}s"
                )

    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    logger.info(
        f"[{variant_id}] Done! Total: {len(results)}, "
        f"New: {len(pending_qids)}, Skipped: {skipped}, "
        f"Failed: {failed}, Time: {time.time()-start_time:.1f}s"
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="InstructIR Dual Query Generation with Qwen3-4B")
    parser.add_argument("--model_path", type=str, default="/home/luwa/Documents/models/Qwen3-4B")
    parser.add_argument("--output_dir", type=str, default="dataset/InstructIR/dual_queries")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    queries, instructions, qrels = load_instructir_queries()

    reformulator = Qwen3Reformulator(args.model_path, args.device)

    for variant_id, variant_config in VARIANTS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running variant: {variant_id}")
        logger.info(f"{'='*60}")

        output_path = run_variant(
            reformulator, variant_id, variant_config,
            queries, instructions, qrels, args.output_dir
        )
        logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
