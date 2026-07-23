"""
Qwen3-4B Reformulator V7 — Multi-Exclusion Support

基于 v6 提示词升级，仅在 Q- 侧做最小幅度调整：
  - Q_plus 维持单字符串（复用 v6 的 Step 1 逻辑）
  - Q_minus 从单字符串升级为 Q_minus_list（每项为 independently sufficient exclusion unit）
  - 输出格式: {"Q_plus": "...", "Q_minus_list": [...]}
  - 空列表表示无 exclusion（不再使用 [NONE] 占位符）

参考方案: paper/模拟评审/TRACE_rewriter_prompt_adjustment_plan.md
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


SYSTEM_PROMPT_V7 = """You are an expert Information Retrieval query optimizer. Your task is to analyze a query and instruction, then extract Q_plus (what to find) and Q_minus_list (a list of exclusion units).

## STEP-BY-STEP ANALYSIS (MUST FOLLOW):

### Step 1: Identify RELEVANT Topics (for Q_plus)
- Read the instruction carefully
- Extract ALL topics that are marked as "relevant" or should be included
- Include ALL relevant topics mentioned, do not miss any
- Remove format words like "documents", "articles", "reports", "studies"
- Write as a natural, fluent phrase
- Q_plus is ONE single string covering all relevant topics jointly

### Step 2: Identify EXPLICIT EXCLUSIONS (for Q_minus_list)
- Look for phrases like "is not relevant", "are not relevant", "ignore", "exclude"
- ONLY extract what is EXPLICITLY marked as irrelevant
- If NO explicit exclusion exists, output an empty list []
- DO NOT put relevant topics in Q_minus_list!

### Step 3: Decompose Exclusions into Independently Sufficient Units
Each item in Q_minus_list must represent an exclusion condition that is INDEPENDENTLY SUFFICIENT to exclude a document.

Determine this from the intended violation semantics, NOT from surface conjunction words like "and"/"or":

1. Independent excluded categories:
   If a document should be excluded when it matches ANY ONE of several categories,
   output EACH category as a SEPARATE exclusion unit.
   Example: "exclude social, political, and ecological impacts" -> 3 units
   (because matching any one category is sufficient to exclude)

2. Jointly required attributes:
   If several attributes must occur TOGETHER before the document should be excluded,
   KEEP them together in ONE exclusion unit.
   Example: "exclude documents discussing both cost overruns and construction delays" -> 1 unit
   (because cost overruns ALONE or construction delays ALONE do not trigger exclusion)

3. Alternative compound exclusions:
   If any one of several compound conditions is sufficient for exclusion,
   output ONE unit for each compound condition while preserving the internal conjunction.
   Example: "exclude both cost overruns and delays, or both pollution and relocation" -> 2 units

4. Context preservation:
   Include the relevant topic, entity, location, time, or relation needed to make each
   exclusion unit meaningful and directly comparable to documents.
   Rewrite each unit as a description of the content to exclude; do NOT retain only
   a negation token such as "not", "without", or "exclude".

5. Fidelity:
   - Do not invent new constraints
   - Do not remove required qualifiers
   - Do not broaden or narrow the exclusion scope
   - Do not split a condition whose meaning depends on its components occurring together

### Step 4: Verify Before Output
- Check: Is everything in Q_plus actually relevant according to instruction?
- Check: Is everything in Q_minus_list actually marked as irrelevant?
- Check: Did I miss any relevant topics that should be in Q_plus?
- Check: Is each Q_minus_list item independently sufficient to exclude a document?
- Check: Did I incorrectly split jointly required attributes?

## CRITICAL WARNINGS:

1. NEVER put relevant topics in Q_minus_list
2. If instruction says "X is relevant", X MUST go in Q_plus, NOT in Q_minus_list
3. If instruction says "X is not relevant", X goes in Q_minus_list
4. If no explicit "not relevant" statement, Q_minus_list = []
5. Do NOT generate Q_minus_list content just to fill space
6. Each Q_minus_list item must be a descriptive phrase, NOT a single keyword
7. Each Q_minus_list item should preserve topic/entity context for direct document matching

## OUTPUT FORMAT (JSON):
{
  "Analysis": {
    "Relevant_Topics": "list all topics marked as relevant",
    "Explicit_Exclusions": "list topics marked as not relevant, or NONE"
  },
  "Q_plus": "natural language query covering all relevant topics",
  "Q_minus_list": ["independently sufficient exclusion unit 1", "exclusion unit 2"]
}

## EXAMPLES:

[Example 1 - Clear Single Exclusion]
Query: Evidence that radio waves from radio towers affect brain cancer.
Instruction: Relevant documents include experiments, studies about radio waves and brain cancer. Any mentions of leukemia is not relevant.

Analysis:
- Relevant_Topics: radio waves from towers/phones, brain cancer
- Explicit_Exclusions: leukemia

Output:
{
  "Analysis": {"Relevant_Topics": "radio waves, brain cancer", "Explicit_Exclusions": "leukemia"},
  "Q_plus": "Association between radio waves from towers or phones and brain cancer incidence",
  "Q_minus_list": ["leukemia mentions in the context of radio waves and brain cancer studies"]
}

[Example 2 - Multiple Independent Exclusion Categories]
Query: What is the ongoing status of The Three Gorges Project?
Instruction: A relevant document will provide the projected or actual date of completion, its estimated or actual total cost, or the estimated or ongoing electrical output. Discussions of the social, political, or ecological impact of the project are not relevant.

Analysis:
- Relevant_Topics: Three Gorges Project status, completion date, total cost, electrical output
- Explicit_Exclusions: social impact, political impact, ecological impact (each independently sufficient)

Output:
{
  "Analysis": {"Relevant_Topics": "Three Gorges Project, completion date, cost, output", "Explicit_Exclusions": "social, political, ecological impacts (independent)"},
  "Q_plus": "Ongoing status of the Three Gorges Project including completion date, total cost, and electrical output",
  "Q_minus_list": [
    "Social impacts of the Three Gorges Project",
    "Political impacts of the Three Gorges Project",
    "Ecological impacts of the Three Gorges Project"
  ]
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
  "Q_minus_list": []
}

[Example 4 - Complex Instruction with Multiple Independent Exclusions]
Query: Identify documents discussing space-borne ocean remote sensing.
Instruction: Documents about oceanography and seabed prospecting are relevant. Documents about geography, agriculture, forestry, mining are not relevant. Marketing and temperature references are not relevant.

Analysis:
- Relevant_Topics: space-borne ocean remote sensing, oceanography, seabed prospecting
- Explicit_Exclusions: geography, agriculture, forestry, mining, marketing, temperature (each independently sufficient)

Output:
{
  "Analysis": {"Relevant_Topics": "space-borne ocean remote sensing, oceanography, seabed prospecting", "Explicit_Exclusions": "geography, agriculture, forestry, mining, marketing, temperature"},
  "Q_plus": "Development and application of space-borne ocean remote sensing in oceanography and seabed prospecting",
  "Q_minus_list": [
    "Geography content in the context of remote sensing",
    "Agriculture content in the context of remote sensing",
    "Forestry content in the context of remote sensing",
    "Mining content in the context of remote sensing",
    "Marketing references in the context of remote sensing",
    "Temperature references in the context of remote sensing"
  ]
}

[Example 5 - Scope Limitation Exclusion]
Query: Identify documents discussing the use of estrogen by postmenopausal women in Britain.
Instruction: The use of hormone replacement therapy outside of the United Kingdom is not relevant. United Kingdom and British development and marketing of estrogen suppressing drugs are relevant.

Analysis:
- Relevant_Topics: estrogen use in Britain, hormone replacement therapy, UK development of estrogen suppressing drugs
- Explicit_Exclusions: hormone replacement therapy OUTSIDE UK (single condition)

Output:
{
  "Analysis": {"Relevant_Topics": "estrogen use in Britain, hormone replacement therapy, UK development of estrogen suppressing drugs", "Explicit_Exclusions": "hormone replacement therapy outside UK"},
  "Q_plus": "Use of estrogen and hormone replacement therapy by postmenopausal women in Britain, including UK development of estrogen suppressing drugs",
  "Q_minus_list": [
    "Hormone replacement therapy use outside the United Kingdom"
  ]
}

[Example 6 - Jointly Required Attributes (DO NOT SPLIT)]
Query: Find reports on the project, but exclude documents that discuss both cost overruns and construction delays.

Analysis:
- Relevant_Topics: project reports
- Explicit_Exclusions: cost overruns AND construction delays together (jointly required)

Output:
{
  "Analysis": {"Relevant_Topics": "project reports", "Explicit_Exclusions": "cost overruns and construction delays together (joint)"},
  "Q_plus": "Reports on the project",
  "Q_minus_list": [
    "Project reports discussing both cost overruns and construction delays"
  ]
}

Remember: When in doubt, Q_minus_list should be an empty list [] rather than guessing. Each non-empty item MUST be independently sufficient to exclude a document and MUST preserve topic/entity context for direct matching."""

USER_PROMPT_TEMPLATE_V7 = """Now analyze this query and instruction:

Query: "{query}"
Instruction: "{instruction}"

Follow the step-by-step analysis and output JSON:"""


class Qwen3LocalReformulatorV7:
    def __init__(
        self,
        model_path: str = "/home/luwa/Documents/models/Qwen3-4B",
        device: str = "cuda",
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        logger.info(f"Loading Qwen3-4B from {model_path}...")
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
        logger.info(f"Qwen3-4B loaded on {device}")

        self.system_prompt = SYSTEM_PROMPT_V7
        self.user_prompt_template = USER_PROMPT_TEMPLATE_V7

    def reformulate(self, query: str, instruction: str) -> Tuple[str, List[str]]:
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

        q_plus, q_minus_list = self._parse_result(result_text, query)
        return q_plus, q_minus_list

    def _parse_result(self, result_text: str, original_query: str) -> Tuple[str, List[str]]:
        """Parse model output into (Q_plus, Q_minus_list).

        Returns:
            q_plus: str (fallback to original_query if missing)
            q_minus_list: List[str] (empty list if no exclusion)
        """
        # First attempt: strict JSON parsing
        try:
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                result = json.loads(json_str)

                q_plus = str(result.get('Q_plus', result.get('q_plus', ''))).strip()
                if not q_plus:
                    q_plus = original_query

                # Parse Q_minus_list (preferred) or fallback to Q_minus (legacy)
                q_minus_raw = result.get('Q_minus_list', result.get('q_minus_list', None))
                if q_minus_raw is not None:
                    if isinstance(q_minus_raw, list):
                        q_minus_list = [
                            str(item).strip()
                            for item in q_minus_raw
                            if str(item).strip() and str(item).strip().upper() not in ("[NONE]", "NONE", "NULL", "N/A")
                        ]
                    else:
                        # Model returned a string instead of a list (fallback)
                        text = str(q_minus_raw).strip()
                        if text.upper() in ("[NONE]", "NONE", "NULL", "N/A", ""):
                            q_minus_list = []
                        else:
                            # Split by common separators as fallback
                            import re
                            parts = [p.strip() for p in re.split(r'[,;]\s*', text) if p.strip()]
                            q_minus_list = parts if parts else [text]
                else:
                    # Legacy Q_minus field
                    q_minus_legacy = str(result.get('Q_minus', result.get('q_minus', ''))).strip()
                    if q_minus_legacy.upper() in ("[NONE]", "NONE", "NULL", "N/A", ""):
                        q_minus_list = []
                    else:
                        q_minus_list = [q_minus_legacy]

                return q_plus, q_minus_list
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: line-by-line parsing
        logger.warning(f"JSON parsing failed, falling back to line parsing. Result text: {result_text[:200]}")
        q_plus = original_query
        q_minus_list = []

        # Try to extract Q_plus from line
        for line in result_text.split('\n'):
            line = line.strip()
            if line.startswith('"Q_plus"') or line.startswith('Q_plus'):
                try:
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        val = parts[1].strip().strip('",')
                        if val:
                            q_plus = val
                except Exception:
                    pass

        # Try to extract Q_minus_list items (heuristic: look for quoted strings after Q_minus_list)
        # This is a best-effort fallback; JSON parsing should normally succeed
        import re
        list_match = re.search(r'Q_minus_list["\']?\s*:\s*\[(.*?)\]', result_text, re.DOTALL)
        if list_match:
            items = re.findall(r'"([^"]+)"', list_match.group(1))
            q_minus_list = [item.strip() for item in items if item.strip() and item.strip().upper() not in ("[NONE]", "NONE")]

        return q_plus, q_minus_list


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
    parser = argparse.ArgumentParser(description="Qwen3-4B Reformulator V7 (Multi-Exclusion)")
    parser.add_argument("--task_name", type=str, required=True,
                        choices=["Core17InstructionRetrieval", "Robust04InstructionRetrieval", "News21InstructionRetrieval"])
    parser.add_argument("--model_path", type=str, default="/home/luwa/Documents/models/Qwen3-4B")
    parser.add_argument("--output_dir", type=str, default="dataset/FollowIR_test/dual_queries_v7")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"dual_queries_v7_{args.task_name}.jsonl")

    q_og, q_changed = load_followir_queries(args.task_name)

    reformulator = Qwen3LocalReformulatorV7(
        model_path=args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    all_queries = []
    for qid, (query_text, instruction) in q_og.items():
        all_queries.append((qid, query_text, instruction, "og"))
    for qid, (query_text, instruction) in q_changed.items():
        all_queries.append((qid, query_text, instruction, "changed"))

    # Resume support: load existing results
    existing = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing[r['qid']] = r
        logger.info(f"Resuming: found {len(existing)} existing entries in {output_path}")

    results = []
    start_time = time.time()
    failed = 0

    for i, (qid, query, instruction, query_type) in enumerate(all_queries):
        if not query:
            continue

        # Skip already-processed queries
        if qid in existing:
            results.append(existing[qid])
            continue

        try:
            idx = int(qid.split('-')[0])
        except ValueError:
            idx = i

        try:
            q_plus, q_minus_list = reformulator.reformulate(query, instruction)
            results.append({
                "task_name": args.task_name,
                "qid": qid,
                "idx": idx,
                "query": query,
                "query_type": query_type,
                "instruction": instruction,
                "prompt_version": "v7",
                "q_plus": q_plus,
                "q_minus_list": q_minus_list,
                "q_minus_count": len(q_minus_list),
                "reformulator": "Qwen3-4B-v7",
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
                "q_minus_list": [],
                "q_minus_count": 0,
                "reformulator": "Qwen3-4B-v7",
                "error": str(e),
                "created_at": datetime.now().isoformat()
            })

        # Incremental save every 10 queries
        if (i + 1) % 10 == 0:
            with open(output_path, 'w') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            elapsed = time.time() - start_time
            processed = len(results) - len(existing)
            speed = processed / elapsed if elapsed > 0 else 0
            remaining = len(all_queries) - i - 1
            eta = remaining / speed if speed > 0 else 0
            logger.info(f"Progress: {i+1}/{len(all_queries)} ({(i+1)/len(all_queries)*100:.1f}%), "
                        f"new: {processed}, speed: {speed:.1f} q/s, ETA: {eta:.0f}s, failed: {failed}")

    # Final save
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Summary statistics
    multi_exclusion_count = sum(1 for r in results if r.get("q_minus_count", 0) > 1)
    no_exclusion_count = sum(1 for r in results if r.get("q_minus_count", 0) == 0)
    single_exclusion_count = sum(1 for r in results if r.get("q_minus_count", 0) == 1)

    logger.info(f"\n{'='*60}")
    logger.info(f"V7 Generation Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Total: {len(results)}, Failed: {failed}, Time: {time.time()-start_time:.1f}s")
    logger.info(f"Q_minus_list distribution:")
    logger.info(f"  No exclusion (M=0):    {no_exclusion_count}")
    logger.info(f"  Single exclusion (M=1): {single_exclusion_count}")
    logger.info(f"  Multi exclusion (M>1):  {multi_exclusion_count}")
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
