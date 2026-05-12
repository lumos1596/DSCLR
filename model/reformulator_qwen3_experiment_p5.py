"""
Phase 5: 精炼策略实验
基于TSC_t01的错误分析，设计针对性优化

核心发现:
- TSC_t01在target_avg上最优(+7.8%)，但p-MRR远低于V5
- 主要错误类型:
  1. 误将相关内容放入Q_minus (310-og, 344-og, 404-og)
  2. Q_minus过长/逐字复制 (355-og, 367-og)
  3. 遗漏隐式排除项 (445-og)
  4. 排除项不够精炼 (356-og: "hormone replacement therapy outside UK" vs V5: "United States, France, global")

Phase 5策略:
A. TSC_REFINE: TSC + 精炼步骤(缩短Q_minus到关键词)
B. TSC_GUARD: TSC + 守卫规则(防止相关内容误入Q_minus)
C. TSC_REFINE_GUARD: TSC + 精炼 + 守卫(综合)
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


TSC_REFINE_SYSTEM = """Analyze the query and instruction to extract Q_plus and Q_minus.

Step 1: List all topics mentioned in the instruction.
Step 2: For each topic, classify it as RELEVANT or EXCLUDED:
  - RELEVANT signals: "X is relevant", "X are relevant", "X counts as", "A relevant document will provide/contain/include X"
  - EXCLUDED signals: "X is not relevant", "X is irrelevant", "X are not relevant", "outside of [scope]", "Discussions of X are not relevant"
  - Scope limitation: if instruction says "only in [region]", other regions are EXCLUDED
Step 3: Q_plus = all RELEVANT topics combined naturally. Q_minus = all EXCLUDED topics, or [NONE] if none.
Step 4: REFINE Q_minus - express each exclusion as a short keyword or phrase (2-4 words), not a full sentence or clause.

Rules:
- Q_minus should contain only short keywords/phrases, separated by commas
- Never copy long phrases from the instruction verbatim
- If the instruction describes what IS relevant but does NOT say anything is excluded, Q_minus = [NONE]
- Describing what a relevant document "will provide" or "must include" defines relevance, NOT exclusion

Output JSON: {"Q_plus": "...", "Q_minus": "..."}"""

TSC_REFINE_EXAMPLES = """

Example 1 (explicit "not relevant" → short keywords):
Query: Evidence that radio waves from radio towers affect brain cancer.
Instruction: Relevant documents include experiments about radio waves and brain cancer. Any mentions of leukemia is not relevant.
Analysis: [radio waves & brain cancer → RELEVANT, leukemia → EXCLUDED]
Refine: "leukemia" (already short)
Output: {"Q_plus": "Association between radio waves and brain cancer", "Q_minus": "leukemia"}

Example 2 (scope limitation → short keywords):
Query: How often were the peace talks in Ireland delayed by acts of violence?
Instruction: Any interruptions to the peace process not directly attributable to acts of violence are not relevant.
Analysis: [delays by violence → RELEVANT, non-violent interruptions → EXCLUDED]
Refine: "non-violent interruptions" (short phrase)
Output: {"Q_plus": "Delays to peace talks in Ireland caused by acts of violence", "Q_minus": "non-violent interruptions"}

Example 3 (no exclusion → [NONE]):
Query: What is the status of The Three Gorges Project?
Instruction: A relevant document will provide the completion date, total cost, and electrical output.
Analysis: [completion date → RELEVANT, total cost → RELEVANT, electrical output → RELEVANT]. No exclusion language.
Output: {"Q_plus": "Status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "[NONE]"}

Example 4 (multiple exclusions → short keywords):
Query: What is the ongoing status of The Three Gorges Project?
Instruction: A relevant document will provide the projected or actual date of completion, its estimated or actual total cost, or the estimated or ongoing electrical output. Discussions of the social, political, or ecological impact of the project are not relevant.
Analysis: [completion date → RELEVANT, total cost → RELEVANT, electrical output → RELEVANT, social impact → EXCLUDED, political impact → EXCLUDED, ecological impact → EXCLUDED]
Refine: "social impact, political impact, ecological impact" (short phrases)
Output: {"Q_plus": "Ongoing status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "social impact, political impact, ecological impact"}

Example 5 (scope + "are relevant" → short keywords):
Query: Identify documents discussing the use of estrogen by postmenopausal women in Britain.
Instruction: The use of hormone replacement therapy outside of the United Kingdom is not relevant. United Kingdom and British development and marketing of estrogen suppressing drugs are relevant.
Analysis: [UK hormone therapy → RELEVANT, British estrogen drugs → RELEVANT, outside UK → EXCLUDED]
Refine: "outside United Kingdom" (short phrase, not full clause)
Output: {"Q_plus": "Use of estrogen by postmenopausal women in Britain including UK hormone replacement therapy and British estrogen suppressing drugs", "Q_minus": "outside United Kingdom"}

Example 6 (long exclusion list → short keywords):
Query: Identify documents discussing the development and application of space-borne ocean remote sensing.
Instruction: Documents discussing the development and application of space-borne ocean remote sensing in oceanography, seabed prospecting and mining, or any marine-science activity are relevant. Documents that discuss the application of satellite remote sensing in geography, agriculture, forestry, mining and mineral prospecting or any land-bound science are not relevant, nor are references to international marketing or promotional advertising of any remote-sensing technology.
Analysis: [ocean remote sensing → RELEVANT, oceanography → RELEVANT, seabed → RELEVANT, geography → EXCLUDED, agriculture → EXCLUDED, forestry → EXCLUDED, land-bound science → EXCLUDED, marketing → EXCLUDED, advertising → EXCLUDED]
Refine: "geography, agriculture, forestry, land-bound science, marketing, advertising" (short keywords)
Output: {"Q_plus": "Development and application of space-borne ocean remote sensing in oceanography and marine sciences", "Q_minus": "geography, agriculture, forestry, land-bound science, marketing, advertising"}

Example 7 (describing relevance ≠ exclusion):
Query: Persons living near radio towers and more recently persons using car (mobile) phones have been diagnosed with brain cancer.
Instruction: Persons living near radio towers and more recently persons using car (mobile) phones have been diagnosed with brain cancer. The argument rages regarding the direct association of one with the other. The statistical evidence is not clear.
Analysis: [radio towers & brain cancer → RELEVANT, mobile phones & brain cancer → RELEVANT, statistical evidence → RELEVANT]. The instruction describes the topic but does NOT exclude anything.
Output: {"Q_plus": "Association between radio towers or mobile phones and brain cancer", "Q_minus": "[NONE]"}"""


TSC_GUARD_SYSTEM = """Analyze the query and instruction to extract Q_plus and Q_minus.

Step 1: List all topics mentioned in the instruction.
Step 2: For each topic, classify it as RELEVANT or EXCLUDED.
Step 3: Q_plus = all RELEVANT topics. Q_minus = all EXCLUDED topics, or [NONE] if none.

GUARD RULES (check before finalizing Q_minus):
1. If the instruction says "X is relevant" or "X are relevant", X MUST go in Q_plus, NEVER in Q_minus
2. If the instruction describes what a relevant document "will provide", "must include", or "must describe", those are RELEVANCE CRITERIA, not exclusions
3. If the instruction only describes what IS relevant without saying anything is NOT relevant, Q_minus = [NONE]
4. "X is not relevant" means X goes in Q_minus. "X is relevant" means X goes in Q_plus. These are OPPOSITES.
5. If instruction says "not directly attributable to X are not relevant", then things NOT attributable to X are excluded (not X itself)

Output JSON: {"Q_plus": "...", "Q_minus": "..."}"""

TSC_GUARD_EXAMPLES = """

Example 1:
Query: Evidence that radio waves from radio towers affect brain cancer.
Instruction: Relevant documents include experiments about radio waves and brain cancer. Any mentions of leukemia is not relevant.
Analysis: [radio waves & brain cancer → RELEVANT, leukemia → EXCLUDED by "not relevant"]
Guard: "leukemia" is explicitly "not relevant" ✓
Output: {"Q_plus": "Association between radio waves and brain cancer", "Q_minus": "leukemia"}

Example 2:
Query: How often were the peace talks in Ireland delayed by acts of violence?
Instruction: Any interruptions to the peace process not directly attributable to acts of violence are not relevant.
Analysis: [delays by violence → RELEVANT, non-violent interruptions → EXCLUDED]
Guard: "not directly attributable to acts of violence" excludes non-violent interruptions, NOT violence itself ✓
Output: {"Q_plus": "Delays to peace talks in Ireland caused by acts of violence", "Q_minus": "non-violent interruptions"}

Example 3 (no exclusion):
Query: What is the status of The Three Gorges Project?
Instruction: A relevant document will provide the completion date, total cost, and electrical output.
Analysis: [completion date → RELEVANT, total cost → RELEVANT, electrical output → RELEVANT]
Guard: "will provide" defines relevance criteria, no exclusion language → Q_minus = [NONE] ✓
Output: {"Q_plus": "Status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "[NONE]"}

Example 4:
Query: What is the ongoing status of The Three Gorges Project?
Instruction: A relevant document will provide the projected or actual date of completion, its estimated or actual total cost, or the estimated or ongoing electrical output. Discussions of the social, political, or ecological impact of the project are not relevant.
Analysis: [completion date → RELEVANT, total cost → RELEVANT, electrical output → RELEVANT, social impact → EXCLUDED, political impact → EXCLUDED, ecological impact → EXCLUDED]
Guard: "are not relevant" applies to social/political/ecological impact ✓
Output: {"Q_plus": "Ongoing status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "social impact, political impact, ecological impact"}

Example 5 (CRITICAL: "are relevant" → Q_plus, NOT Q_minus):
Query: Identify documents discussing the use of estrogen by postmenopausal women in Britain.
Instruction: The use of hormone replacement therapy outside of the United Kingdom is not relevant. United Kingdom and British development and marketing of estrogen suppressing drugs are relevant.
Analysis: [UK hormone therapy → RELEVANT, British estrogen drugs → RELEVANT, outside UK → EXCLUDED]
Guard: "are relevant" confirms UK drugs → Q_plus ✓. "outside UK is not relevant" → Q_minus ✓. British estrogen suppressing drugs are NOT excluded ✓
Output: {"Q_plus": "Use of estrogen by postmenopausal women in Britain including UK hormone replacement therapy and British estrogen suppressing drugs", "Q_minus": "outside United Kingdom"}

Example 6 (describing relevance ≠ exclusion):
Query: Persons living near radio towers and using mobile phones have been diagnosed with brain cancer.
Instruction: Persons living near radio towers and more recently persons using car (mobile) phones have been diagnosed with brain cancer. The argument rages regarding the direct association of one with the other.
Analysis: [radio towers & brain cancer → RELEVANT, mobile phones & brain cancer → RELEVANT]
Guard: Instruction only describes the topic, no exclusion language → Q_minus = [NONE] ✓
Output: {"Q_plus": "Association between radio towers or mobile phones and brain cancer", "Q_minus": "[NONE]"}

Example 7 (scope limitation with "must"):
Query: To be relevant, a document must indicate either a country where a woman has been installed as clergy or a country that is considering such an installation. The clergy position must be as church pastor.
Instruction: To be relevant, a document must indicate either a country where a woman has been installed as clergy or a country that is considering such an installation. The clergy position must be as church pastor. Non-pastor positions such as nuns or choir members are not relevant. Documents that discuss the situation in the United States are not relevant.
Analysis: [women as clergy/pastor → RELEVANT, non-pastor positions → EXCLUDED, United States → EXCLUDED]
Guard: "must be as church pastor" defines relevance scope → non-pastor excluded ✓. "United States are not relevant" → excluded ✓
Output: {"Q_plus": "Countries where women have been installed as clergy or considering such installation as church pastor", "Q_minus": "non-pastor positions, United States"}"""


TSC_REFINE_GUARD_SYSTEM = """Analyze the query and instruction to extract Q_plus and Q_minus.

Step 1: List all topics mentioned in the instruction.
Step 2: For each topic, classify it as RELEVANT or EXCLUDED.
Step 3: Q_plus = all RELEVANT topics. Q_minus = all EXCLUDED topics as SHORT KEYWORDS, or [NONE] if none.

GUARD RULES:
1. "X is relevant" / "X are relevant" → X goes in Q_plus, NEVER in Q_minus
2. "A relevant document will provide/must include" → defines relevance, NOT exclusion
3. If no exclusion language exists, Q_minus = [NONE]
4. "X is not relevant" / "X are not relevant" → X goes in Q_minus
5. "Not directly attributable to X are not relevant" → excludes non-X, NOT X itself

REFINE RULES:
- Express each exclusion as 2-4 word keywords/phrases, not full clauses
- Never copy long phrases verbatim from the instruction
- Separate multiple exclusions with commas

Output JSON: {"Q_plus": "...", "Q_minus": "..."}"""

TSC_REFINE_GUARD_EXAMPLES = """

Example 1:
Query: Evidence that radio waves from radio towers affect brain cancer.
Instruction: Relevant documents include experiments about radio waves and brain cancer. Any mentions of leukemia is not relevant.
Analysis: [radio waves & brain cancer → RELEVANT, leukemia → EXCLUDED]
Guard: "not relevant" → leukemia excluded ✓. Refine: "leukemia" (already short) ✓
Output: {"Q_plus": "Association between radio waves and brain cancer", "Q_minus": "leukemia"}

Example 2:
Query: How often were the peace talks in Ireland delayed by acts of violence?
Instruction: Any interruptions to the peace process not directly attributable to acts of violence are not relevant.
Analysis: [delays by violence → RELEVANT, non-violent interruptions → EXCLUDED]
Guard: "not directly attributable to violence" excludes non-violent, NOT violence ✓. Refine: "non-violent interruptions" ✓
Output: {"Q_plus": "Delays to peace talks in Ireland caused by acts of violence", "Q_minus": "non-violent interruptions"}

Example 3:
Query: What is the status of The Three Gorges Project?
Instruction: A relevant document will provide the completion date, total cost, and electrical output.
Analysis: [completion date → RELEVANT, total cost → RELEVANT, electrical output → RELEVANT]
Guard: "will provide" = relevance criteria, no exclusion → [NONE] ✓
Output: {"Q_plus": "Status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "[NONE]"}

Example 4:
Query: What is the ongoing status of The Three Gorges Project?
Instruction: A relevant document will provide the projected or actual date of completion, its estimated or actual total cost, or the estimated or ongoing electrical output. Discussions of the social, political, or ecological impact of the project are not relevant.
Analysis: [completion date → RELEVANT, total cost → RELEVANT, electrical output → RELEVANT, social impact → EXCLUDED, political impact → EXCLUDED, ecological impact → EXCLUDED]
Guard: "are not relevant" ✓. Refine: "social impact, political impact, ecological impact" ✓
Output: {"Q_plus": "Ongoing status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "social impact, political impact, ecological impact"}

Example 5:
Query: Identify documents discussing the use of estrogen by postmenopausal women in Britain.
Instruction: The use of hormone replacement therapy outside of the United Kingdom is not relevant. United Kingdom and British development and marketing of estrogen suppressing drugs are relevant.
Analysis: [UK hormone therapy → RELEVANT, British estrogen drugs → RELEVANT, outside UK → EXCLUDED]
Guard: "are relevant" → UK drugs in Q_plus ✓. Refine: "outside United Kingdom" ✓
Output: {"Q_plus": "Use of estrogen by postmenopausal women in Britain including UK hormone replacement therapy and British estrogen suppressing drugs", "Q_minus": "outside United Kingdom"}

Example 6 (long exclusion → short keywords):
Query: Identify documents discussing the development and application of space-borne ocean remote sensing.
Instruction: Documents discussing the development and application of space-borne ocean remote sensing in oceanography, seabed prospecting and mining, or any marine-science activity are relevant. Documents that discuss the application of satellite remote sensing in geography, agriculture, forestry, mining and mineral prospecting or any land-bound science are not relevant, nor are references to international marketing or promotional advertising of any remote-sensing technology.
Analysis: [ocean remote sensing → RELEVANT, geography → EXCLUDED, agriculture → EXCLUDED, forestry → EXCLUDED, land-bound science → EXCLUDED, marketing → EXCLUDED, advertising → EXCLUDED]
Guard: "are not relevant" + "nor are" ✓. Refine: "geography, agriculture, forestry, land-bound science, marketing, advertising" ✓
Output: {"Q_plus": "Development and application of space-borne ocean remote sensing in oceanography and marine sciences", "Q_minus": "geography, agriculture, forestry, land-bound science, marketing, advertising"}

Example 7 (describing topic ≠ exclusion):
Query: Persons living near radio towers and using mobile phones have been diagnosed with brain cancer.
Instruction: Persons living near radio towers and more recently persons using car (mobile) phones have been diagnosed with brain cancer. The argument rages regarding the direct association of one with the other.
Analysis: [radio towers & brain cancer → RELEVANT, mobile phones → RELEVANT]
Guard: No exclusion language → [NONE] ✓
Output: {"Q_plus": "Association between radio towers or mobile phones and brain cancer", "Q_minus": "[NONE]"}

Example 8 (scope + multiple exclusions):
Query: To be relevant, a document must indicate either a country where a woman has been installed as clergy or a country that is considering such an installation. The clergy position must be as church pastor.
Instruction: To be relevant, a document must indicate either a country where a woman has been installed as clergy or a country that is considering such an installation. The clergy position must be as church pastor. Non-pastor positions such as nuns or choir members are not relevant. Documents that discuss the situation in the United States are not relevant.
Analysis: [women as clergy → RELEVANT, non-pastor positions → EXCLUDED, United States → EXCLUDED]
Guard: "are not relevant" ✓. Refine: "non-pastor positions, United States" ✓
Output: {"Q_plus": "Countries where women have been installed as clergy or considering such installation as church pastor", "Q_minus": "non-pastor positions, United States"}"""


USER_TEMPLATE = """

Now process this:

Query: "{query}"
Instruction: "{instruction}"

Output:"""


PHASE5_VARIANTS = OrderedDict()

for strategy_name, (system_prompt, examples_prompt) in [
    ('TSC_REFINE', (TSC_REFINE_SYSTEM, TSC_REFINE_EXAMPLES)),
    ('TSC_GUARD', (TSC_GUARD_SYSTEM, TSC_GUARD_EXAMPLES)),
    ('TSC_REFINE_GUARD', (TSC_REFINE_GUARD_SYSTEM, TSC_REFINE_GUARD_EXAMPLES)),
]:
    for temp in [0.1]:
        variant_id = f"{strategy_name}_t{str(temp).replace('.', '')}"
        PHASE5_VARIANTS[variant_id] = {
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
    parser = argparse.ArgumentParser(description="Phase 5: Refined Strategy Experiments")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval")
    parser.add_argument("--model_path", type=str, default="/home/luwa/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default="dataset/FollowIR_test/dual_queries_experiments_p5")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--variants", type=str, default="all",
                        help="Comma-separated variant IDs, or 'all'")
    parser.add_argument("--eval_only", action="store_true", help="Only compute quality metrics, skip reformulation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if args.variants == 'all':
        selected = PHASE5_VARIANTS
    else:
        variant_ids = [v.strip() for v in args.variants.split(',')]
        selected = OrderedDict((k, PHASE5_VARIANTS[k]) for k in variant_ids if k in PHASE5_VARIANTS)

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
        print("PHASE 5 EXPERIMENT RESULTS SUMMARY")
        print("=" * 140)
        print(f"{'Variant':<25} {'NONE_rate':>10} {'V5_NONE':>10} {'NONE_diff':>10} {'QM_match':>10} {'FP_rate':>10} {'FN_rate':>10} {'Jaccard':>10} {'QM_len':>10} {'V5_len':>10}")
        print("-" * 125)
        for m in sorted(results_summary, key=lambda x: x['avg_jaccard'], reverse=True):
            print(f"{m['variant_id']:<25} {m['none_rate']:>10.3f} {m['v5_none_rate']:>10.3f} {m['none_rate_diff']:>10.3f} {m['qm_match_rate']:>10.3f} {m['fp_rate']:>10.3f} {m['fn_rate']:>10.3f} {m['avg_jaccard']:>10.3f} {m['avg_qm_len']:>10.1f} {m['v5_avg_qm_len']:>10.1f}")

        summary_path = os.path.join(args.output_dir, f"experiment_summary_p5_{args.task_name}.json")
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
