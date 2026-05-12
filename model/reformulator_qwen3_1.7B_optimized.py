"""
Qwen3-1.7B Prompt Optimization: 针对 1.7B 模型的提示词优化实验

核心问题诊断:
1. [NONE] 格式不一致: 1.7B 输出 none/NONE/[NONE] 三种变体
2. 过度提取: 把指令中描述相关内容的短语当作排除项
3. 自矛盾率高: Q_minus 与 Q_plus 词汇重叠 (Core17 40%)
4. Q_minus 过长: 平均 64 字符 vs v5 的 48 字符

策略:
A. TSC_STRICT: TSC + 严格 [NONE] 判定 + 格式约束 + 简洁性要求
B. TSC_DECIDE: 前置决策步骤 + 严格分离相关/排除 + 简洁关键词
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
# Strategy A: TSC_STRICT
# 核心改进:
#   1. 明确 [NONE] 判定规则（必须看到排除信号词才输出排除项）
#   2. 严格格式要求（必须输出 [NONE]，不是 none/NONE）
#   3. 简洁性约束（Q_minus 只用关键词，不用完整句子）
#   4. 自矛盾检查（Q_minus 不能包含 Q_plus 中已有的内容）
# ============================================================

TSC_STRICT_SYSTEM = """Analyze the query and instruction to extract Q_plus and Q_minus.

Step 1: Does the instruction contain ANY explicit exclusion signal? Look ONLY for these signals:
  - "X is not relevant" / "X is irrelevant" / "X are not relevant"
  - "X is outside the scope" / "not directly attributable to X"
  - "Discussions of X are not relevant"
  - "only in [region]" (implies other regions are excluded)
  - "X is irrelevant" / "while X is irrelevant"

Step 2: If NO exclusion signal found → Q_minus = [NONE]. STOP HERE.
  Do NOT infer exclusions from what the instruction does NOT mention.
  Do NOT treat descriptions of relevant content as exclusions.
  "A relevant document will include X" means X is RELEVANT, not excluded.

Step 3: If exclusion signal found, list ONLY the explicitly excluded topics as Q_minus.
  - Use short keywords only (2-5 words per item), NOT full sentences
  - Do NOT include anything that also appears in Q_plus
  - If the exclusion is about a scope (e.g., "outside UK"), only put the scope boundary in Q_minus

Output JSON: {"Q_plus": "...", "Q_minus": "[NONE]"} or {"Q_plus": "...", "Q_minus": "keyword1, keyword2"}

CRITICAL RULES:
- Q_minus MUST be exactly [NONE] (with brackets) when no exclusion signal exists
- Do NOT write "none" or "NONE" — always use [NONE]
- Q_minus should contain ONLY excluded topics, never relevant topics
- Keep Q_minus concise: use keywords, not sentences"""

TSC_STRICT_EXAMPLES = """

Example 1 (explicit "not relevant"):
Query: Evidence that radio waves from radio towers affect brain cancer.
Instruction: Relevant documents include experiments about radio waves and brain cancer. Any mentions of leukemia is not relevant.
Decision: Exclusion signal found ("leukemia is not relevant")
Output: {"Q_plus": "Association between radio waves and brain cancer", "Q_minus": "leukemia"}

Example 2 (scope limitation):
Query: How often were the peace talks in Ireland delayed by acts of violence?
Instruction: Any interruptions to the peace process not directly attributable to acts of violence are not relevant.
Decision: Exclusion signal found ("not directly attributable to acts of violence are not relevant")
Output: {"Q_plus": "Delays to peace talks in Ireland caused by acts of violence", "Q_minus": "non-violent interruptions"}

Example 3 (NO exclusion — everything is relevant):
Query: What is the status of The Three Gorges Project?
Instruction: A relevant document will provide the completion date, total cost, and electrical output.
Decision: No exclusion signal. The instruction only describes what IS relevant.
Output: {"Q_plus": "Status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "[NONE]"}

Example 4 (NO exclusion — instruction describes relevant content only):
Query: Identify documents that discuss mainstreaming children with physical or mental impairments.
Instruction: A relevant document will include the pros and cons of mainstreaming children with physical or mental impairments, the benefits to the impaired child, as well as the attitude, beliefs and concerns of teachers and school administrators.
Decision: No exclusion signal. The instruction only describes what IS relevant. "Pros and cons" and "attitudes" are all RELEVANT.
Output: {"Q_plus": "Mainstreaming children with physical or mental impairments including pros and cons, benefits, and attitudes of teachers and administrators", "Q_minus": "[NONE]"}

Example 5 (NO exclusion — instruction provides context, not exclusions):
Query: What steps have been taken world-wide by those bearing the cost of E-mail to prevent excesses?
Instruction: To be relevant, a document will concern dissatisfaction by an entity paying for the cost of electronic mail. Particularly sought are items which relate to system users who abuse the system.
Decision: No exclusion signal. "Dissatisfaction" and "system users who abuse" are what to FIND, not what to exclude.
Output: {"Q_plus": "Steps taken to prevent email abuses by those bearing the cost, including dissatisfaction of paying entities and system user abuses", "Q_minus": "[NONE]"}

Example 6 (explicit "not relevant" with multiple items):
Query: What is the ongoing status of The Three Gorges Project?
Instruction: A relevant document will provide the projected or actual date of completion, its estimated or actual total cost, or the estimated or ongoing electrical output. Discussions of the social, political, or ecological impact of the project are not relevant.
Decision: Exclusion signal found ("social, political, or ecological impact are not relevant")
Output: {"Q_plus": "Ongoing status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "social impact, political impact, ecological impact"}

Example 7 (scope limitation + "are relevant"):
Query: Identify documents discussing the use of estrogen by postmenopausal women in Britain.
Instruction: The use of hormone replacement therapy outside of the United Kingdom is not relevant. United Kingdom and British development and marketing of estrogen suppressing drugs are relevant.
Decision: Exclusion signal found ("outside of the United Kingdom is not relevant"). "UK development" and "British drugs" are RELEVANT.
Output: {"Q_plus": "Use of estrogen by postmenopausal women in Britain including UK hormone replacement therapy and British estrogen suppressing drugs", "Q_minus": "hormone replacement therapy outside UK"}

Example 8 (NO exclusion — instruction only adds context):
Query: Find information about troops who have been accused of war crimes during their service in Afghanistan.
Instruction: Australian forces are accused of war crimes while on duty in Afghanistan. Some other countries are also investigating what their troops may have done while serving there. There have also been American contractors accused and convicted of war crimes while fighting the War on Terror. The International Criminal Court (ICC) has been investigating these accusations for years.
Decision: No exclusion signal. ALL mentioned groups (Australian forces, other countries, American contractors, ICC) are RELEVANT examples of the topic.
Output: {"Q_plus": "Troops accused of war crimes in Afghanistan including Australian forces, other countries, American contractors, and ICC investigations", "Q_minus": "[NONE]"}"""


# ============================================================
# Strategy B: TSC_DECIDE
# 核心改进:
#   1. 前置二分类决策: 指令是否包含排除信息?
#   2. 如果否, 直接输出 [NONE], 不进入主题分析
#   3. 如果是, 用严格规则提取排除关键词
# ============================================================

TSC_DECIDE_SYSTEM = """Your task: Decide whether the instruction contains any exclusion information, then extract Q_plus and Q_minus.

FIRST, answer this question: Does the instruction explicitly state that something is NOT relevant, irrelevant, or outside scope?

YES signals (the instruction DOES contain exclusions):
  - "X is not relevant" / "X is irrelevant" / "X are not relevant"
  - "outside of [scope]" / "not directly attributable to"
  - "Discussions of X are not relevant"
  - "only in [region]" (implies other regions excluded)

NO signals (the instruction does NOT contain exclusions):
  - "A relevant document will include/provide/contain X" → this describes what IS relevant
  - "Particularly sought are X" → X is what to find
  - "Relevant documents may identify X" → X is relevant
  - The instruction only lists examples or context of relevant content
  - The instruction adds requirements (e.g., "must include X") → X is a requirement, not an exclusion

If NO → Q_minus = [NONE]. Put ALL instruction content into Q_plus.
If YES → Extract ONLY the explicitly excluded topics into Q_minus as short keywords.

Output JSON: {"Q_plus": "...", "Q_minus": "[NONE]"} or {"Q_plus": "...", "Q_minus": "keyword1, keyword2"}

FORMAT RULES:
- Q_minus must be exactly [NONE] (with brackets) when no exclusion exists
- Never write "none" or "NONE" without brackets
- Q_minus uses only short keywords (2-5 words per item)
- Q_minus must NOT contain anything that is also in Q_plus"""

TSC_DECIDE_EXAMPLES = """

Example 1:
Query: Evidence that radio waves from radio towers affect brain cancer.
Instruction: Relevant documents include experiments about radio waves and brain cancer. Any mentions of leukemia is not relevant.
Decision: YES — "leukemia is not relevant" is an exclusion signal.
Output: {"Q_plus": "Association between radio waves and brain cancer", "Q_minus": "leukemia"}

Example 2:
Query: How often were the peace talks in Ireland delayed by acts of violence?
Instruction: Any interruptions to the peace process not directly attributable to acts of violence are not relevant.
Decision: YES — "not directly attributable to acts of violence are not relevant" is an exclusion signal.
Output: {"Q_plus": "Delays to peace talks in Ireland caused by acts of violence", "Q_minus": "non-violent interruptions"}

Example 3:
Query: What is the status of The Three Gorges Project?
Instruction: A relevant document will provide the completion date, total cost, and electrical output.
Decision: NO — the instruction only describes what IS relevant. No exclusion signal.
Output: {"Q_plus": "Status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "[NONE]"}

Example 4:
Query: Identify documents that discuss mainstreaming children with physical or mental impairments.
Instruction: A relevant document will include the pros and cons of mainstreaming children with physical or mental impairments, the benefits to the impaired child, as well as the attitude, beliefs and concerns of teachers and school administrators.
Decision: NO — the instruction only describes what IS relevant (pros, cons, benefits, attitudes). No exclusion signal.
Output: {"Q_plus": "Mainstreaming children with impairments including pros and cons, benefits, and teacher/administrator attitudes", "Q_minus": "[NONE]"}

Example 5:
Query: What steps have been taken world-wide by those bearing the cost of E-mail to prevent excesses?
Instruction: To be relevant, a document will concern dissatisfaction by an entity paying for the cost of electronic mail. Particularly sought are items which relate to system users who abuse the system.
Decision: NO — "dissatisfaction" and "system users who abuse" are what to FIND, not what to exclude. No exclusion signal.
Output: {"Q_plus": "Steps taken to prevent email abuses by cost-bearing entities, including dissatisfaction and system user abuses", "Q_minus": "[NONE]"}

Example 6:
Query: What is the ongoing status of The Three Gorges Project?
Instruction: A relevant document will provide the projected or actual date of completion, its estimated or actual total cost, or the estimated or ongoing electrical output. Discussions of the social, political, or ecological impact of the project are not relevant.
Decision: YES — "social, political, or ecological impact are not relevant" is an exclusion signal.
Output: {"Q_plus": "Ongoing status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "social impact, political impact, ecological impact"}

Example 7:
Query: Identify documents discussing the use of estrogen by postmenopausal women in Britain.
Instruction: The use of hormone replacement therapy outside of the United Kingdom is not relevant. United Kingdom and British development and marketing of estrogen suppressing drugs are relevant.
Decision: YES — "outside of the United Kingdom is not relevant" is an exclusion signal. "UK development" and "British drugs" are RELEVANT.
Output: {"Q_plus": "Use of estrogen by postmenopausal women in Britain including UK hormone replacement therapy and British estrogen suppressing drugs", "Q_minus": "hormone replacement therapy outside UK"}

Example 8:
Query: Find information about troops accused of war crimes in Afghanistan.
Instruction: Australian forces are accused of war crimes while on duty in Afghanistan. Some other countries are also investigating what their troops may have done. American contractors have also been accused and convicted. The ICC has been investigating these accusations.
Decision: NO — all mentioned groups are RELEVANT examples of the topic. No exclusion signal.
Output: {"Q_plus": "Troops accused of war crimes in Afghanistan including Australian forces, other countries, American contractors, and ICC investigations", "Q_minus": "[NONE]"}

Example 9:
Query: How did France respond to protests against its nuclear testing in the South Pacific?
Instruction: A document containing information regarding any type of protest by any one is relevant as long as France's response is also included in the document.
Decision: NO — the instruction describes a requirement (France's response must be included), not an exclusion. No exclusion signal.
Output: {"Q_plus": "France's response to protests against nuclear testing in the South Pacific", "Q_minus": "[NONE]"}

Example 10:
Query: What new uses have been developed for old automobile tires as a means of tire recycling?
Instruction: A relevant document must show advantageous uses of recycled tires, such as: destructive distillation of scrap rubber for valuable chemicals, reef building for fish habitats, filler or binder in asphalt roadway mixes, and burning in a controlled environment for heat generation.
Decision: NO — the instruction only lists examples of relevant content. No exclusion signal.
Output: {"Q_plus": "New uses for recycled tires including chemical distillation, reef building, asphalt mixes, and heat generation", "Q_minus": "[NONE]"}"""


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

Example 7 (NO exclusion — background context is relevant):
Query: How accurate was the 2020 reporting of COVID-19 in Mexico?
Instruction: Several factors contributed to under-reporting COVID-related cases and deaths in Mexico. One factor was that Mexico only included fatalities confirmed with a lab test. Documents must include relevant information about the US-Mexico border.
Decision: NO — "under-reporting factors" are RELEVANT context, and "must include US-Mexico border" is a REQUIREMENT. No exclusion signal.
Output: {"Q_plus": "Accuracy of 2020 COVID-19 reporting in Mexico including under-reporting factors and US-Mexico border information", "Q_minus": "[NONE]"}

Example 8 (YES — explicit "not relevant"):
Query: What is the ongoing status of The Three Gorges Project?
Instruction: A relevant document will provide the projected or actual date of completion, its estimated or actual total cost, or the estimated or ongoing electrical output. Discussions of the social, political, or ecological impact of the project are not relevant.
Decision: YES — "social, political, or ecological impact are not relevant" is an exclusion signal.
Output: {"Q_plus": "Ongoing status of the Three Gorges Project including completion date, total cost, and electrical output", "Q_minus": "social impact, political impact, ecological impact"}

Example 9 (YES — scope limitation + "are relevant"):
Query: Identify documents discussing the use of estrogen by postmenopausal women in Britain.
Instruction: The use of hormone replacement therapy outside of the United Kingdom is not relevant. United Kingdom and British development and marketing of estrogen suppressing drugs are relevant.
Decision: YES — "outside of the United Kingdom is not relevant" is an exclusion signal. "UK development" and "British drugs" are RELEVANT.
Output: {"Q_plus": "Use of estrogen by postmenopausal women in Britain including UK hormone replacement therapy and British estrogen suppressing drugs", "Q_minus": "hormone replacement therapy outside UK"}

Example 10 (NO exclusion — instruction describes what to find):
Query: What steps have been taken world-wide by those bearing the cost of E-mail to prevent excesses?
Instruction: To be relevant, a document will concern dissatisfaction by an entity paying for the cost of electronic mail. Particularly sought are items which relate to system users who abuse the system.
Decision: NO — "dissatisfaction" and "system users who abuse" are what to FIND. "Particularly sought" means RELEVANT. No exclusion signal.
Output: {"Q_plus": "Steps taken to prevent email abuses by cost-bearing entities, including dissatisfaction and system user abuses", "Q_minus": "[NONE]"}

Example 11 (NO exclusion — "must include" + background):
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

for strategy_name, (system_prompt, examples_prompt) in [
    ('TSC_STRICT', (TSC_STRICT_SYSTEM, TSC_STRICT_EXAMPLES)),
    ('TSC_DECIDE', (TSC_DECIDE_SYSTEM, TSC_DECIDE_EXAMPLES)),
    ('TSC_BALANCED', (TSC_BALANCED_SYSTEM, TSC_BALANCED_EXAMPLES)),
]:
    for temp in [0.1]:
        variant_id = f"{strategy_name}_t{str(temp).replace('.', '')}"
        VARIANTS[variant_id] = {
            'system_prompt': system_prompt + examples_prompt,
            'user_template': USER_TEMPLATE,
            'temperature': temp,
        }


class Qwen3Reformulator:
    def __init__(self, model_path, device="cuda", max_new_tokens=512):
        self.max_new_tokens = max_new_tokens
        logger.info(f"Loading Qwen3-1.7B from {model_path}...")
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
        logger.info(f"Qwen3-1.7B loaded on {device}")

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
                "reformulator": f"Qwen3-1.7B-{variant_id}",
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
                "reformulator": f"Qwen3-1.7B-{variant_id}",
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
    parser = argparse.ArgumentParser(description="Qwen3-1.7B Prompt Optimization")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="dataset/FollowIR_test/dual_queries_1.7B_optimized")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--variants", type=str, default="TSC_STRICT_t01,TSC_DECIDE_t01",
                        help="Comma-separated variant IDs, or 'all'")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if args.variants == 'all':
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
            logger.info(f"[{variant_id}] none_rate={metrics['none_rate']:.3f} (V5={metrics['v5_none_rate']:.3f}, diff={metrics['none_rate_diff']:.3f}), qm_match={metrics['qm_match_rate']:.3f}, FP={metrics['fp_rate']:.3f}, FN={metrics['fn_rate']:.3f}, Jaccard={metrics['avg_jaccard']:.3f}")

    if results_summary:
        print("\n" + "=" * 140)
        print("QWEN3-1.7B PROMPT OPTIMIZATION RESULTS")
        print("=" * 140)
        print(f"{'Variant':<25} {'NONE_rate':>10} {'V5_NONE':>10} {'NONE_diff':>10} {'QM_match':>10} {'FP_rate':>10} {'FN_rate':>10} {'Jaccard':>10} {'QM_len':>10} {'V5_len':>10}")
        print("-" * 125)
        for m in sorted(results_summary, key=lambda x: x['avg_jaccard'], reverse=True):
            print(f"{m['variant_id']:<25} {m['none_rate']:>10.3f} {m['v5_none_rate']:>10.3f} {m['none_rate_diff']:>10.3f} {m['qm_match_rate']:>10.3f} {m['fp_rate']:>10.3f} {m['fn_rate']:>10.3f} {m['avg_jaccard']:>10.3f} {m['avg_qm_len']:>10.1f} {m['v5_avg_qm_len']:>10.1f}")

        summary_path = os.path.join(args.output_dir, f"experiment_summary_1.7B_opt_{args.task_name}.json")
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
