"""
系统性提示词工程实验框架
用于 Qwen3-0.6B 指令改写任务

实验维度:
A. 指令风格: concise / structured / example_dominant
B. 示例覆盖: 4ex / 6ex / 8ex
C. 温度: 0.1 / 0.3 / 0.5

评估指标:
1. p-MRR (主要 - 指令遵循能力)
2. [NONE]率偏差 (与V5的偏差, 越小越好)
3. Q_minus长度偏差 (与V5的偏差)
4. target_avg (检索性能)
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


# ============================================================
# PROMPT VARIANTS
# ============================================================

# --- Dimension A: Instruction Style ---

CONCISE_SYSTEM = """Given a query and instruction, extract Q_plus (what to find) and Q_minus (what to exclude).

Rules:
- Q_plus: ALL relevant topics from the instruction, written as a natural phrase
- Q_minus: Topics the instruction excludes or marks as outside scope, or [NONE] if no exclusion
- Never put relevant topics in Q_minus
- If instruction limits scope (e.g., "only in UK", "enhanced screening"), topics outside that scope go in Q_minus

Output JSON: {"Q_plus": "...", "Q_minus": "..."}"""

STRUCTURED_SYSTEM = """Analyze the query and instruction step by step:

Step 1 - Find relevant topics: What does the instruction say is relevant? These go in Q_plus.
Step 2 - Find excluded topics: What does the instruction say is NOT relevant or outside scope? These go in Q_minus.
Step 3 - Check: Are all relevant topics in Q_plus? Are all excluded topics in Q_minus? No mixing?

Rules:
- Q_plus includes ALL topics the instruction confirms as relevant
- Q_minus includes topics marked as "not relevant", "irrelevant", or outside the defined scope
- If no exclusion exists, Q_minus = "[NONE]"
- NEVER put relevant topics in Q_minus

Output JSON: {"Q_plus": "...", "Q_minus": "..."}"""

EXAMPLE_DOMINANT_SYSTEM = """Extract Q_plus and Q_minus from the query and instruction.

Q_plus = what to find (all relevant topics)
Q_minus = what to exclude (not relevant or outside scope), or [NONE]

Output JSON: {"Q_plus": "...", "Q_minus": "..."}"""


# --- Dimension B: Example Coverage ---

EXAMPLES_4 = """

Example 1:
Query: Evidence that radio waves from radio towers affect brain cancer.
Instruction: Relevant documents include experiments about radio waves and brain cancer. Any mentions of leukemia is not relevant.
Output: {"Q_plus": "Association between radio waves from radio towers or phones and brain cancer incidence", "Q_minus": "leukemia"}

Example 2:
Query: How often were the peace talks in Ireland delayed by acts of violence?
Instruction: Any interruptions to the peace process not directly attributable to acts of violence are not relevant.
Output: {"Q_plus": "Frequency of delays to peace talks in Ireland caused by acts of violence", "Q_minus": "non-violent interruptions, political delays"}

Example 3:
Query: What is the status of The Three Gorges Project?
Instruction: A relevant document will provide the completion date, total cost, and electrical output.
Output: {"Q_plus": "Status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "[NONE]"}

Example 4:
Query: Is there contemporary interest in the Greek philosophy of stoicism?
Instruction: Actual references to the philosophy or philosophers in books or by professors counts as interest while productions of Greek stoic plays and new stoic artistic productions are all irrelevant.
Output: {"Q_plus": "Contemporary interest in Greek stoicism philosophy including references in books and by professors", "Q_minus": "Greek stoic plays, stoic artistic productions, modern people described as stoics"}"""

EXAMPLES_6 = EXAMPLES_4 + """

Example 5:
Query: Identify documents discussing space-borne ocean remote sensing.
Instruction: Documents about oceanography and seabed prospecting are relevant. Documents about geography, agriculture, forestry, mining are not relevant. Marketing and temperature references are not relevant.
Output: {"Q_plus": "Development and application of space-borne ocean remote sensing in oceanography and seabed prospecting", "Q_minus": "land-bound sciences, agriculture, forestry, mining, marketing, temperature"}

Example 6:
Query: A relevant document would discuss the effectiveness of efforts to better scrutinize passengers and luggage.
Instruction: A relevant document would contain reports on what steps airports worldwide have taken to better scrutinize passengers and their luggage on domestic and international flights and to step up screening of all persons and goods boarding aircraft.
Output: {"Q_plus": "Effectiveness of enhanced passenger and luggage scrutiny at airports worldwide, stepped-up screening measures", "Q_minus": "normal security screening methods, mere mention of enhanced security without details"}"""

EXAMPLES_8 = EXAMPLES_6 + """

Example 7:
Query: What measures are being taken by local South American authorities to preserve the Amazon tropical rain forest?
Instruction: Relevant documents may identify the official organizations and individuals of the countries included in the Amazon rain forest; the measures being taken by them to preserve the rain forest; and the effects of these measures on the indigenous Indian populations.
Output: {"Q_plus": "Measures taken by local South American authorities to preserve the Amazon rain forest and their effects on indigenous populations", "Q_minus": "native Indians without preservation context"}

Example 8:
Query: What new uses have been developed for old automobile tires as a means of tire recycling?
Instruction: A relevant document must show advantageous uses of recycled tires, such as: destructive distillation of scrap rubber for valuable chemicals, reef building for fish habitats, filler or binder in asphalt road surfaces.
Output: {"Q_plus": "New advantageous uses for recycled automobile tires including destructive distillation, reef building, and asphalt filler", "Q_minus": "rubber as raw material, tire disposal without recycling"}"""


USER_TEMPLATE = """

Now process this:

Query: "{query}"
Instruction: "{instruction}"

Output:"""


# --- Build all variants ---

VARIANTS = OrderedDict()

for style_name, style_prompt in [('concise', CONCISE_SYSTEM), ('structured', STRUCTURED_SYSTEM), ('example_dominant', EXAMPLE_DOMINANT_SYSTEM)]:
    for ex_name, ex_prompt in [('4ex', EXAMPLES_4), ('6ex', EXAMPLES_6), ('8ex', EXAMPLES_8)]:
        for temp in [0.1, 0.3, 0.5]:
            variant_id = f"{style_name}_{ex_name}_t{str(temp).replace('.', '')}"
            VARIANTS[variant_id] = {
                'system_prompt': style_prompt + ex_prompt,
                'user_template': USER_TEMPLATE,
                'temperature': temp,
            }


class Qwen3Reformulator:
    def __init__(self, model_path, device="cuda", max_new_tokens=512):
        self.max_new_tokens = max_new_tokens
        logger.info(f"Loading Qwen3-0.6B from {model_path}...")
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
        logger.info(f"Qwen3-0.6B loaded on {device}")

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

    ds_q = datasets.load_dataset(dataset_path, 'queries', trust_remote_code=True)
    ds_inst = datasets.load_dataset(dataset_path, 'instruction', trust_remote_code=True)

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
                "reformulator": f"Qwen3-0.6B-{variant_id}",
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
                "reformulator": f"Qwen3-0.6B-{variant_id}",
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


def compute_quality_metrics(output_path, v5_path):
    def load_jsonl(path):
        records = {}
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    records[r['qid']] = r
        return records

    variant_data = load_jsonl(output_path)
    v5_data = load_jsonl(v5_path)

    total = len(variant_data)
    none_count = sum(1 for r in variant_data.values() if r['q_minus'] == '[NONE]')
    v5_none_count = sum(1 for r in v5_data.values() if r['q_minus'] == '[NONE]')

    qm_lens = [len(r['q_minus']) for r in variant_data.values() if r['q_minus'] != '[NONE]']
    v5_qm_lens = [len(r['q_minus']) for r in v5_data.values() if r['q_minus'] != '[NONE]']

    none_rate = none_count / total if total > 0 else 0
    v5_none_rate = v5_none_count / len(v5_data) if len(v5_data) > 0 else 0
    none_rate_diff = abs(none_rate - v5_none_rate)

    avg_qm_len = sum(qm_lens) / len(qm_lens) if qm_lens else 0
    v5_avg_qm_len = sum(v5_qm_lens) / len(v5_qm_lens) if v5_qm_lens else 0

    qm_match = sum(1 for qid in variant_data if qid in v5_data and variant_data[qid]['q_minus'] == v5_data[qid]['q_minus'])
    qm_match_rate = qm_match / total if total > 0 else 0

    return {
        'none_rate': none_rate,
        'v5_none_rate': v5_none_rate,
        'none_rate_diff': none_rate_diff,
        'avg_qm_len': avg_qm_len,
        'v5_avg_qm_len': v5_avg_qm_len,
        'qm_match_rate': qm_match_rate,
        'total': total,
    }


def main():
    parser = argparse.ArgumentParser(description="Systematic Prompt Engineering Experiments")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval")
    parser.add_argument("--model_path", type=str, default="/home/luwa/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default="dataset/FollowIR_test/dual_queries_experiments")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--variants", type=str, default="all",
                        help="Comma-separated variant IDs, or 'all', or 'phase1' (3 styles x 6ex x t01)")
    parser.add_argument("--eval_only", action="store_true", help="Only compute quality metrics, skip reformulation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if args.variants == 'phase1':
        selected = {k: v for k, v in VARIANTS.items() if '_6ex_t01' in k}
    elif args.variants == 'phase2':
        selected = {k: v for k, v in VARIANTS.items() if '_6ex_' in k}
    elif args.variants == 'all':
        selected = VARIANTS
    else:
        variant_ids = [v.strip() for v in args.variants.split(',')]
        selected = OrderedDict((k, VARIANTS[k]) for k in variant_ids if k in VARIANTS)

    logger.info(f"Selected {len(selected)} variants: {list(selected.keys())}")

    q_og, q_changed = load_followir_queries(args.task_name)
    all_queries = []
    for qid, (query_text, instruction) in q_og.items():
        all_queries.append((qid, query_text, instruction, "og"))
    for qid, (query_text, instruction) in q_changed.items():
        all_queries.append((qid, query_text, instruction, "changed"))

    v5_path_map = {
        "Core17InstructionRetrieval": "dataset/FollowIR_test/dual_queries_v5/dual_queries_v5_Core17InstructionRetrieval.jsonl",
        "Robust04InstructionRetrieval": "dataset/FollowIR_test/dual_queries_v5/dual_queries_v5_Robust04InstructionRetrieval.jsonl",
        "News21InstructionRetrieval": "dataset/FollowIR_test/dual_queries_v5/dual_queries_v5_News21InstructionRetrieval.jsonl",
    }
    v5_path = v5_path_map.get(args.task_name, "")

    if not args.eval_only:
        reformulator = Qwen3Reformulator(args.model_path, args.device)

    results_summary = []

    for variant_id, variant_config in selected.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running variant: {variant_id}")
        logger.info(f"{'='*60}")

        variant_output_dir = os.path.join(args.output_dir, variant_id)

        if not args.eval_only:
            output_path = run_variant(
                reformulator, variant_id, variant_config,
                all_queries, variant_output_dir, args.task_name
            )
        else:
            output_path = os.path.join(variant_output_dir, f"dual_queries_{variant_id}_{args.task_name}.jsonl")

        if os.path.exists(output_path) and v5_path and os.path.exists(v5_path):
            metrics = compute_quality_metrics(output_path, v5_path)
            metrics['variant_id'] = variant_id
            metrics['temperature'] = variant_config['temperature']
            results_summary.append(metrics)
            logger.info(f"[{variant_id}] none_rate={metrics['none_rate']:.3f} (V5={metrics['v5_none_rate']:.3f}, diff={metrics['none_rate_diff']:.3f}), qm_match={metrics['qm_match_rate']:.3f}")

    if results_summary:
        print("\n" + "=" * 120)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 120)
        print(f"{'Variant':<35} {'NONE_rate':>10} {'V5_NONE':>10} {'NONE_diff':>10} {'QM_match':>10} {'Avg_QM_len':>12} {'V5_QM_len':>12}")
        print("-" * 100)
        for m in sorted(results_summary, key=lambda x: x['none_rate_diff']):
            print(f"{m['variant_id']:<35} {m['none_rate']:>10.3f} {m['v5_none_rate']:>10.3f} {m['none_rate_diff']:>10.3f} {m['qm_match_rate']:>10.3f} {m['avg_qm_len']:>12.1f} {m['v5_avg_qm_len']:>12.1f}")

        summary_path = os.path.join(args.output_dir, f"experiment_summary_{args.task_name}.json")
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
