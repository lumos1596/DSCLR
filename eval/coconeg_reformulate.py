"""
COCO-Neg Dual Query Generation with Qwen3-4B

Generates Q_plus and Q_minus for COCO-Neg negated captions using TSC_BALANCED prompt.
COCO-Neg captions contain negation patterns (e.g., "no X", "without X", "but there is no X").

Usage:
  cd /home/luwa/Documents/DSCLR && /home/luwa/.conda/envs/dsclr/bin/python -m eval.coconeg_reformulate \
    --model_path /home/luwa/Documents/models/Qwen3-4B --device cuda \
    --output_dir dataset/COCO-Neg/dual_queries
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
import pandas as pd
import ast

logger = logging.getLogger(__name__)

NEG_SYSTEM = """Your task: Extract Q_plus and Q_minus from a caption that contains negation or exclusion.

The input caption contains explicit negation/exclusion signals such as:
- "but there is no X" / "but there are no X"
- "without X" / "no X is visible"
- "X is not present" / "X is absent"
- "excluding X" / "apart from X"

STEP 1 — Identify the exclusion part of the caption.
STEP 2 — Q_plus = the positive visual description (what IS in the image), rephrased as a clear image search query
STEP 3 — Q_minus = the excluded visual elements as short keywords (1-3 words per item)

If the caption has NO negation/exclusion → Q_minus = [NONE], Q_plus = rephrased caption.

Output JSON: {"Q_plus": "...", "Q_minus": "keyword1, keyword2"} or {"Q_plus": "...", "Q_minus": "[NONE]"}

FORMAT RULES:
- Q_minus must be exactly [NONE] (with brackets) when no exclusion exists
- Never write "none" or "NONE" without brackets
- Q_minus uses only short keywords (1-3 words per item)
- Q_minus must NOT contain anything that is also in Q_plus
- Q_plus should be a self-contained visual description that captures the positive content"""

NEG_EXAMPLES = """

Example 1 (explicit "no X"):
Caption: A man in a kitchen is making pizzas, but there is no chair in sight.
Output: {"Q_plus": "A man in a kitchen making pizzas", "Q_minus": "chair"}

Example 2 (explicit "without"):
Caption: A dog running through a grassy field without a leash.
Output: {"Q_plus": "A dog running through a grassy field", "Q_minus": "leash"}

Example 3 (explicit "no X is visible"):
Caption: No car is visible in the image, which shows city dwellers passing by a homeless man.
Output: {"Q_plus": "City dwellers passing by a homeless man", "Q_minus": "car"}

Example 4 (double exclusion):
Caption: A person with a shopping cart on a city street, with no car and no dining table.
Output: {"Q_plus": "A person with a shopping cart on a city street", "Q_minus": "car, dining table"}

Example 5 (no exclusion):
Caption: A cat sitting on a windowsill looking outside.
Output: {"Q_plus": "A cat sitting on a windowsill looking outside", "Q_minus": "[NONE]"}"""

NEG_USER_TEMPLATE = """

Now process this caption:

Caption: "{query}"

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


def load_coconeg_queries():
    csv_path = os.path.join("dataset/COCO-Neg", "evaluation data/images/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv")
    df = pd.read_csv(csv_path)

    queries = {}
    q_idx = 0
    for _, row in df.iterrows():
        img_id = str(row['image_id'])
        caps = ast.literal_eval(row['captions'])
        for cap in caps:
            qid = f"q{q_idx}"
            queries[qid] = cap
            q_idx += 1

    logger.info(f"Loaded {len(queries)} negated caption queries from COCO-Neg")
    return queries


def run_variant(reformulator, variant_id, variant_config, queries, output_dir):
    system_prompt = variant_config['system_prompt']
    user_template = variant_config['user_template']
    temperature = variant_config['temperature']
    batch_size = reformulator.batch_size

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"COCO-Neg_{variant_id}.jsonl")

    existing = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing[r['qid']] = r
        logger.info(f"[{variant_id}] Found {len(existing)} existing entries")

    eval_qids = sorted(queries.keys())

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

    for batch_start in tqdm(range(0, len(pending_qids), batch_size), desc=f"Reformulating COCO-Neg"):
        batch_end = min(batch_start + batch_size, len(pending_qids))
        b_qids = pending_qids[batch_start:batch_end]
        b_queries = pending_queries[batch_start:batch_end]

        try:
            q_plus_list, q_minus_list = reformulator.reformulate_batch(
                b_queries, system_prompt, user_template, temperature
            )

            for j in range(len(b_qids)):
                result = {
                    'qid': b_qids[j],
                    'original_query': b_queries[j],
                    'q_plus': q_plus_list[j],
                    'q_minus': q_minus_list[j],
                    'variant': variant_id,
                }
                results.append(result)

            if (batch_start // batch_size) % 10 == 0 and batch_start > 0:
                with open(output_path, 'w') as f:
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + '\n')
                elapsed = time.time() - start_time
                rate = len(results) / elapsed if elapsed > 0 else 0
                logger.info(f"[{variant_id}] Progress: {len(results)}/{len(eval_qids)} ({rate:.1f} q/s)")

        except Exception as e:
            logger.error(f"Batch failed at {batch_start}: {e}")
            failed += len(b_qids)
            for j in range(len(b_qids)):
                results.append({
                    'qid': b_qids[j],
                    'original_query': b_queries[j],
                    'q_plus': b_queries[j],
                    'q_minus': '[NONE]',
                    'variant': variant_id,
                })

    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    elapsed = time.time() - start_time
    q_minus_count = sum(1 for r in results if r['q_minus'] != '[NONE]')
    logger.info(f"[{variant_id}] Done! {len(results)} queries, {failed} failed, {elapsed:.1f}s")
    logger.info(f"[{variant_id}] Q_minus rate: {q_minus_count}/{len(results)} ({q_minus_count/len(results)*100:.1f}%)")
    logger.info(f"[{variant_id}] Saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/luwa/Documents/models/Qwen3-4B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="dataset/COCO-Neg/dual_queries")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    queries = load_coconeg_queries()

    reformulator = Qwen3Reformulator(
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
    )

    for variant_id, variant_config in VARIANTS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running variant: {variant_id}")
        logger.info(f"{'='*60}")
        run_variant(reformulator, variant_id, variant_config, queries, args.output_dir)


if __name__ == "__main__":
    main()
