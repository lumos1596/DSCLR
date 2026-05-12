"""
Phase 3: 改进提示词实验
基于Phase 1-2错误分析，设计三种针对性策略

错误模式:
1. 相关内容误入Q_minus (356-og: UK estrogen drugs被排除)
2. 模式混淆 (433-og: "are relevant"被误读为"are irrelevant")
3. 遗漏隐式排除 (416-og: "social/political/ecological not relevant"→[NONE])
4. 泛化占位符 (445-og: "non-relevant or outside scope")

策略:
A. Contrastive Few-Shot (CFS): 对比示例,区分relevant vs irrelevant
B. Two-Step Classification (TSC): 先列举再分类
C. Error-Corrected + Scope-Aware (ECSA): 修正示例+scope引导
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
# PHASE 3 PROMPT VARIANTS
# ============================================================

# --- Strategy A: Contrastive Few-Shot ---
# 核心创新: 对比示例对, 明确区分"X is relevant" vs "X is irrelevant"

CFS_SYSTEM = """Extract Q_plus and Q_minus from the query and instruction.

Q_plus = what to find (all relevant topics from the instruction)
Q_minus = what to exclude (topics the instruction says are NOT relevant or outside scope), or [NONE] if nothing is excluded

CRITICAL: Read the instruction carefully. "X is relevant" means X goes in Q_plus. "X is irrelevant" or "X is not relevant" means X goes in Q_minus. These are opposites!

Output JSON: {"Q_plus": "...", "Q_minus": "..."}"""

CFS_EXAMPLES = """

Example 1 (explicit exclusion):
Query: Evidence that radio waves from radio towers affect brain cancer.
Instruction: Relevant documents include experiments about radio waves and brain cancer. Any mentions of leukemia is not relevant.
Output: {"Q_plus": "Association between radio waves from radio towers or phones and brain cancer incidence", "Q_minus": "leukemia"}

Example 2 (scope limitation → implicit exclusion):
Query: How often were the peace talks in Ireland delayed by acts of violence?
Instruction: Any interruptions to the peace process not directly attributable to acts of violence are not relevant.
Output: {"Q_plus": "Frequency of delays to peace talks in Ireland caused by acts of violence", "Q_minus": "non-violent interruptions, political delays"}

Example 3 (no exclusion - everything mentioned is relevant):
Query: What is the status of The Three Gorges Project?
Instruction: A relevant document will provide the completion date, total cost, and electrical output.
Output: {"Q_plus": "Status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "[NONE]"}

Example 4 (CONTRAST: same topic, opposite relevance):
Query A: Is there contemporary interest in stoicism philosophy?
Instruction A: References to the philosophy in books, productions of stoic plays, and new stoic artistic productions are all relevant.
Output A: {"Q_plus": "Contemporary interest in stoicism including philosophy references, stoic plays, and stoic artistic productions", "Q_minus": "[NONE]"}

Query B: Is there contemporary interest in stoicism philosophy?
Instruction B: References to the philosophy in books counts as interest while productions of stoic plays and new stoic artistic productions are all irrelevant.
Output B: {"Q_plus": "Contemporary interest in stoicism philosophy including references in books", "Q_minus": "stoic plays, stoic artistic productions"}

Example 5 (scope limitation creates exclusion):
Query: Identify documents discussing the use of estrogen by postmenopausal women in Britain.
Instruction: The use of hormone replacement therapy outside of the United Kingdom is not relevant. United Kingdom and British development and marketing of estrogen suppressing drugs are relevant.
Output: {"Q_plus": "Use of estrogen by postmenopausal women in Britain including UK hormone replacement therapy and British estrogen suppressing drugs", "Q_minus": "hormone replacement therapy outside the United Kingdom"}"""


# --- Strategy B: Two-Step Classification ---
# 核心创新: 先列举所有提及的主题, 再逐个分类为relevant/excluded

TSC_SYSTEM = """Analyze the query and instruction to extract Q_plus and Q_minus.

Step 1: List all topics mentioned in the instruction.
Step 2: For each topic, decide: is it RELEVANT (included) or EXCLUDED (not relevant/outside scope)?
Step 3: Q_plus = all RELEVANT topics. Q_minus = all EXCLUDED topics, or [NONE] if none.

Key signals for EXCLUDED:
- "X is not relevant" / "X is irrelevant" / "X are not relevant"
- "outside of [scope]" / "not directly attributable to"
- "Discussions of X are not relevant"
- Scope limitation: if instruction says "only in [region]", then other regions are excluded

Key signals for RELEVANT:
- "X is relevant" / "X are relevant" / "X counts as interest"
- "A relevant document will provide/contain/include X"
- "Relevant documents may identify X"

Output JSON: {"Q_plus": "...", "Q_minus": "..."}"""

TSC_EXAMPLES = """

Example 1:
Query: Evidence that radio waves from radio towers affect brain cancer.
Instruction: Relevant documents include experiments about radio waves and brain cancer. Any mentions of leukemia is not relevant.
Analysis: Topics: [radio waves & brain cancer → RELEVANT, leukemia → EXCLUDED]
Output: {"Q_plus": "Association between radio waves from radio towers or phones and brain cancer incidence", "Q_minus": "leukemia"}

Example 2:
Query: How often were the peace talks in Ireland delayed by acts of violence?
Instruction: Any interruptions to the peace process not directly attributable to acts of violence are not relevant.
Analysis: Topics: [delays caused by violence → RELEVANT, non-violent interruptions → EXCLUDED]
Output: {"Q_plus": "Frequency of delays to peace talks in Ireland caused by acts of violence", "Q_minus": "non-violent interruptions, political delays"}

Example 3:
Query: What is the status of The Three Gorges Project?
Instruction: A relevant document will provide the completion date, total cost, and electrical output.
Analysis: Topics: [completion date → RELEVANT, total cost → RELEVANT, electrical output → RELEVANT]. No exclusions.
Output: {"Q_plus": "Status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "[NONE]"}

Example 4:
Query: What is the ongoing status of The Three Gorges Project?
Instruction: A relevant document will provide the projected or actual date of completion, its estimated or actual total cost, or the estimated or ongoing electrical output. Discussions of the social, political, or ecological impact of the project are not relevant.
Analysis: Topics: [completion date → RELEVANT, total cost → RELEVANT, electrical output → RELEVANT, social impact → EXCLUDED, political impact → EXCLUDED, ecological impact → EXCLUDED]
Output: {"Q_plus": "Ongoing status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "social impact, political impact, ecological impact"}

Example 5:
Query: Identify documents discussing the use of estrogen by postmenopausal women in Britain.
Instruction: The use of hormone replacement therapy outside of the United Kingdom is not relevant. United Kingdom and British development and marketing of estrogen suppressing drugs are relevant.
Analysis: Topics: [UK hormone replacement therapy → RELEVANT, British estrogen suppressing drugs → RELEVANT, hormone therapy outside UK → EXCLUDED]
Output: {"Q_plus": "Use of estrogen by postmenopausal women in Britain including UK hormone replacement therapy and British estrogen suppressing drugs", "Q_minus": "hormone replacement therapy outside the United Kingdom"}"""


# --- Strategy C: Error-Corrected + Scope-Aware ---
# 核心创新: 修正有问题的示例 + 添加scope limitation引导

ECSA_SYSTEM = """Extract Q_plus and Q_minus from the query and instruction.

Q_plus = what to find (all relevant topics from the instruction)
Q_minus = what to exclude, or [NONE] if nothing is excluded

Rules:
1. Q_minus only contains topics the instruction explicitly marks as NOT relevant, irrelevant, or outside the defined scope
2. If the instruction says "X is relevant" or "X are relevant", X goes in Q_plus, NOT Q_minus
3. If the instruction limits scope (e.g., "only in UK", "must include X"), topics outside that scope go in Q_minus
4. If the instruction says "discussions of X are not relevant", X goes in Q_minus
5. Never put vague phrases like "non-relevant" or "outside scope" in Q_minus - only specific excluded topics

Output JSON: {"Q_plus": "...", "Q_minus": "..."}"""

ECSA_EXAMPLES = """

Example 1 (explicit "not relevant"):
Query: Evidence that radio waves from radio towers affect brain cancer.
Instruction: Relevant documents include experiments about radio waves and brain cancer. Any mentions of leukemia is not relevant.
Output: {"Q_plus": "Association between radio waves from radio towers or phones and brain cancer incidence", "Q_minus": "leukemia"}

Example 2 (scope limitation):
Query: How often were the peace talks in Ireland delayed by acts of violence?
Instruction: Any interruptions to the peace process not directly attributable to acts of violence are not relevant.
Output: {"Q_plus": "Frequency of delays to peace talks in Ireland caused by acts of violence", "Q_minus": "non-violent interruptions, political delays"}

Example 3 (no exclusion):
Query: What is the status of The Three Gorges Project?
Instruction: A relevant document will provide the completion date, total cost, and electrical output.
Output: {"Q_plus": "Status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "[NONE]"}

Example 4 (explicit "not relevant" with multiple items):
Query: What is the ongoing status of The Three Gorges Project?
Instruction: A relevant document will provide the projected or actual date of completion, its estimated or actual total cost, or the estimated or ongoing electrical output. Discussions of the social, political, or ecological impact of the project are not relevant.
Output: {"Q_plus": "Ongoing status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "social impact, political impact, ecological impact"}

Example 5 (scope limitation + "are relevant"):
Query: Identify documents discussing the use of estrogen by postmenopausal women in Britain.
Instruction: The use of hormone replacement therapy outside of the United Kingdom is not relevant. United Kingdom and British development and marketing of estrogen suppressing drugs are relevant.
Output: {"Q_plus": "Use of estrogen by postmenopausal women in Britain including UK hormone replacement therapy and British estrogen suppressing drugs", "Q_minus": "hormone replacement therapy outside the United Kingdom"}

Example 6 ("are all relevant" - no exclusion):
Query: Is there contemporary interest in the Greek philosophy of stoicism?
Instruction: Actual references to the philosophy or philosophers, productions of Greek stoic plays, and new stoic artistic productions are all relevant.
Output: {"Q_plus": "Contemporary interest in Greek stoicism including philosophy references, stoic plays, and stoic artistic productions", "Q_minus": "[NONE]"}"""


USER_TEMPLATE = """

Now process this:

Query: "{query}"
Instruction: "{instruction}"

Output:"""


# --- Build Phase 3 variants ---

PHASE3_VARIANTS = OrderedDict()

for strategy_name, (system_prompt, examples_prompt) in [
    ('CFS', (CFS_SYSTEM, CFS_EXAMPLES)),
    ('TSC', (TSC_SYSTEM, TSC_EXAMPLES)),
    ('ECSA', (ECSA_SYSTEM, ECSA_EXAMPLES)),
]:
    for temp in [0.1, 0.3]:
        variant_id = f"{strategy_name}_t{str(temp).replace('.', '')}"
        PHASE3_VARIANTS[variant_id] = {
            'system_prompt': system_prompt + examples_prompt,
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

    fp_count = sum(1 for qid in variant_data if qid in v5_data
                   and variant_data[qid]['q_minus'] != '[NONE]'
                   and v5_data[qid]['q_minus'] == '[NONE]')
    fn_count = sum(1 for qid in variant_data if qid in v5_data
                   and variant_data[qid]['q_minus'] == '[NONE]'
                   and v5_data[qid]['q_minus'] != '[NONE]')

    fp_rate = fp_count / total if total > 0 else 0
    fn_rate = fn_count / total if total > 0 else 0

    def jaccard_similarity(s1, s2):
        if s1 == '[NONE]' and s2 == '[NONE]':
            return 1.0
        if s1 == '[NONE]' or s2 == '[NONE]':
            return 0.0
        set1 = set(s1.lower().split())
        set2 = set(s2.lower().split())
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    jaccard_scores = []
    for qid in variant_data:
        if qid in v5_data:
            jaccard_scores.append(jaccard_similarity(variant_data[qid]['q_minus'], v5_data[qid]['q_minus']))
    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0

    return {
        'none_rate': none_rate,
        'v5_none_rate': v5_none_rate,
        'none_rate_diff': none_rate_diff,
        'avg_qm_len': avg_qm_len,
        'v5_avg_qm_len': v5_avg_qm_len,
        'qm_match_rate': qm_match_rate,
        'fp_rate': fp_rate,
        'fn_rate': fn_rate,
        'avg_jaccard': avg_jaccard,
        'total': total,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Improved Prompt Engineering Experiments")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval")
    parser.add_argument("--model_path", type=str, default="/home/luwa/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default="dataset/FollowIR_test/dual_queries_experiments_p3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--variants", type=str, default="all",
                        help="Comma-separated variant IDs, or 'all'")
    parser.add_argument("--eval_only", action="store_true", help="Only compute quality metrics, skip reformulation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if args.variants == 'all':
        selected = PHASE3_VARIANTS
    else:
        variant_ids = [v.strip() for v in args.variants.split(',')]
        selected = OrderedDict((k, PHASE3_VARIANTS[k]) for k in variant_ids if k in PHASE3_VARIANTS)

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
            logger.info(f"[{variant_id}] none_rate={metrics['none_rate']:.3f} (V5={metrics['v5_none_rate']:.3f}, diff={metrics['none_rate_diff']:.3f}), qm_match={metrics['qm_match_rate']:.3f}, FP={metrics['fp_rate']:.3f}, FN={metrics['fn_rate']:.3f}, Jaccard={metrics['avg_jaccard']:.3f}")

    if results_summary:
        print("\n" + "=" * 140)
        print("PHASE 3 EXPERIMENT RESULTS SUMMARY")
        print("=" * 140)
        print(f"{'Variant':<20} {'NONE_rate':>10} {'V5_NONE':>10} {'NONE_diff':>10} {'QM_match':>10} {'FP_rate':>10} {'FN_rate':>10} {'Jaccard':>10} {'QM_len':>10} {'V5_len':>10}")
        print("-" * 120)
        for m in sorted(results_summary, key=lambda x: x['avg_jaccard'], reverse=True):
            print(f"{m['variant_id']:<20} {m['none_rate']:>10.3f} {m['v5_none_rate']:>10.3f} {m['none_rate_diff']:>10.3f} {m['qm_match_rate']:>10.3f} {m['fp_rate']:>10.3f} {m['fn_rate']:>10.3f} {m['avg_jaccard']:>10.3f} {m['avg_qm_len']:>10.1f} {m['v5_avg_qm_len']:>10.1f}")

        summary_path = os.path.join(args.output_dir, f"experiment_summary_p3_{args.task_name}.json")
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
