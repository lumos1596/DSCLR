"""
NegConstraint Dual Query Generation with Qwen3-4B

Generates Q_plus and Q_minus for NegConstraint queries using TSC_BALANCED prompt.
NegConstraint queries inherently contain negation patterns (e.g., "but don't mention",
"without", "excluding"), so we extract Q_minus directly from the query itself.

Usage:
  cd /home/luwa/Documents/DSCLR && /home/luwa/.conda/envs/dsclr/bin/python -m eval.negconstraint_reformulate \
    --model_path /home/luwa/Documents/models/Qwen3-4B --device cuda \
    --output_dir dataset/NegConstraint/NegConstraint/dual_queries
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

logger = logging.getLogger(__name__)

NEG_SYSTEM = """Your task: Extract Q_plus and Q_minus from a query that contains negation or exclusion.

The input query contains explicit negation/exclusion signals such as:
- "but don't mention X" / "but do not mention X"
- "without X" / "excluding X"
- "other than X" / "apart from X"
- "not X" / "no X"

STEP 1 — Identify the exclusion part of the query.
STEP 2 — Q_plus = the positive search intent (what to FIND), rephrased as a clear search query
STEP 3 — Q_minus = the excluded topics as short keywords (2-5 words per item)

If the query has NO negation/exclusion → Q_minus = [NONE], Q_plus = rephrased query.

Output JSON: {"Q_plus": "...", "Q_minus": "keyword1, keyword2"} or {"Q_plus": "...", "Q_minus": "[NONE]"}

FORMAT RULES:
- Q_minus must be exactly [NONE] (with brackets) when no exclusion exists
- Never write "none" or "NONE" without brackets
- Q_minus uses only short keywords (2-5 words per item)
- Q_minus must NOT contain anything that is also in Q_plus
- Q_plus should be a self-contained search query that captures the positive intent"""

NEG_EXAMPLES = """

Example 1 (explicit "don't mention"):
Query: Aaron's profile, but don't mention Moses.
Output: {"Q_plus": "Profile and biography of Aaron", "Q_minus": "Moses"}

Example 2 (explicit "without"):
Query: Provide an introduction to Aalborg Municipality without Denmark.
Output: {"Q_plus": "Introduction to Aalborg Municipality", "Q_minus": "Denmark"}

Example 3 (explicit "excluding"):
Query: Examine the theme of justice in To Kill a Mockingbird, excluding the trial of Tom Robinson.
Output: {"Q_plus": "Theme of justice in To Kill a Mockingbird", "Q_minus": "trial of Tom Robinson"}

Example 4 (explicit "other than"):
Query: What themes are expressed in Allen Ginsberg's works other than 'Howl' and Edgar Allan Poe's works?
Output: {"Q_plus": "Themes expressed in Allen Ginsberg's works", "Q_minus": "Howl, Edgar Allan Poe"}

Example 5 (double exclusion):
Query: What themes do Allen Ginsberg's works other than 'Howl' and Edgar Allan Poe's works other than 'The Raven' express?
Output: {"Q_plus": "Themes expressed in Allen Ginsberg's works and Edgar Allan Poe's works", "Q_minus": "Howl, The Raven"}

Example 6 (no exclusion):
Query: How did José Canale's leadership influence the success of the teams he coached in the QMJHL?
Output: {"Q_plus": "José Canale's leadership influence on QMJHL team success", "Q_minus": "[NONE]"}"""

NEG_USER_TEMPLATE = """

Now process this query:

Query: "{query}"

Output:"""

VARIANTS = OrderedDict()

for temp in [0.1]:
    variant_id = f"TSC_BALANCED_t{str(temp).replace('.', '')}"
    VARIANTS[variant_id] = {
        'system_prompt': NEG_SYSTEM + NEG_EXAMPLES,
        'user_template': NEG_USER_TEMPLATE,
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

    def _build_prompt(self, query, system_prompt, user_template):
        user_prompt = user_template.format(query=query)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def reformulate_batch(self, queries, system_prompt, user_template, temperature):
        prompts = [
            self._build_prompt(q, system_prompt, user_template)
            for q in queries
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


def load_negconstraint_queries(data_dir):
    queries = {}
    with open(os.path.join(data_dir, "queries.jsonl"), encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            queries[str(q['_id'])] = q.get('text', '')

    qrels = {}
    import csv
    with open(os.path.join(data_dir, "test.tsv"), encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            qid, did, score = str(row[0]), str(row[1]), int(row[2])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = score

    logger.info(f"Loaded {len(queries)} queries, {len(qrels)} qrels")
    return queries, qrels


def run_variant(reformulator, variant_id, variant_config, queries, qrels, output_dir):
    system_prompt = variant_config['system_prompt']
    user_template = variant_config['user_template']
    temperature = variant_config['temperature']
    batch_size = reformulator.batch_size

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"NegConstraint_{variant_id}.jsonl")

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

    for qid in eval_qids:
        query_text = queries[qid]
        if not query_text:
            continue
        if qid in existing:
            results.append(existing[qid])
        else:
            pending_qids.append(qid)
            pending_queries.append(query_text)

    skipped = len(results)
    logger.info(f"[{variant_id}] Skipped {skipped} existing, processing {len(pending_qids)} new queries in batches of {batch_size}")

    start_time = time.time()
    failed = 0

    for batch_start in tqdm(range(0, len(pending_qids), batch_size), desc=f"Reformulating NegConstraint"):
        batch_end = min(batch_start + batch_size, len(pending_qids))
        b_qids = pending_qids[batch_start:batch_end]
        b_queries = pending_queries[batch_start:batch_end]

        try:
            q_plus_list, q_minus_list = reformulator.reformulate_batch(
                b_queries, system_prompt, user_template, temperature
            )
            for j, qid in enumerate(b_qids):
                results.append({
                    "task_name": "NegConstraint",
                    "qid": qid,
                    "query": b_queries[j],
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
                    "task_name": "NegConstraint",
                    "qid": qid,
                    "query": b_queries[j],
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
    parser = argparse.ArgumentParser(description="NegConstraint Dual Query Generation with Qwen3-4B")
    parser.add_argument("--model_path", type=str, default="/home/luwa/Documents/models/Qwen3-4B")
    parser.add_argument("--output_dir", type=str, default="dataset/NegConstraint/NegConstraint/dual_queries")
    parser.add_argument("--data_dir", type=str, default="dataset/NegConstraint/NegConstraint")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    queries, qrels = load_negconstraint_queries(args.data_dir)

    reformulator = Qwen3Reformulator(args.model_path, args.device)

    for variant_id, variant_config in VARIANTS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running variant: {variant_id}")
        logger.info(f"{'='*60}")

        output_path = run_variant(
            reformulator, variant_id, variant_config,
            queries, qrels, args.output_dir
        )
        logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
