"""
ComLQ Dual Query Generation with Qwen3-4B

Generates Q_plus and Q_minus for ComLQ queries.
ComLQ has 14 query types: 5 with negation (2in/3in/inp/pin/pni) + 9 without.

For negation queries: extract Q_minus from the negation/exclusion part.
For non-negation queries: Q_minus = [NONE], Q_plus = rephrased query.

Usage:
  cd /home/luwa/Documents/DSCLR && /home/luwa/.conda/envs/dsclr/bin/python -m eval.comlq_reformulate \
    --model_path /home/luwa/Documents/models/Qwen3-4B --device cuda \
    --output_dir dataset/ComLQ/dual_queries
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

COMLQ_SYSTEM = """Your task: Extract Q_plus and Q_minus from a complex logical query.

The input query may contain negation/exclusion signals such as:
- "but not X" / "but are not X" / "but have not X"
- "without X" / "excluding X" / "other than X"
- "never X" / "not X" / "no X"

It may also be a purely positive query with NO negation.

STEP 1 — Determine if the query contains negation/exclusion.
STEP 2 — Q_plus = the positive search intent (what to FIND), rephrased as a clear search query
STEP 3 — Q_minus = the excluded topics as short keywords (2-5 words per item)

If the query has NO negation/exclusion → Q_minus = [NONE], Q_plus = rephrased query.

Output JSON: {"Q_plus": "...", "Q_minus": "keyword1, keyword2"} or {"Q_plus": "...", "Q_minus": "[NONE]"}

FORMAT RULES:
- Q_minus must be exactly [NONE] (with brackets) when no exclusion exists
- Never write "none" or "NONE" without brackets
- Q_minus uses only short keywords (2-5 words per item)
- Q_minus must NOT contain anything that is also in Q_plus
- Q_plus should be a self-contained search query that captures the positive intent
- For queries with multiple conditions (AND/OR), Q_plus should combine all positive conditions"""

COMLQ_EXAMPLES = """

Example 1 (negation "but not"):
Query: Which ecclesiastical titles refer to the male head of a monastery but not to a female head?
Output: {"Q_plus": "Ecclesiastical titles for male head of a monastery", "Q_minus": "female head, abbess"}

Example 2 (negation "never"):
Query: Which foundations are classified as a 501(c)(3) organization but never as a for-profit corporation?
Output: {"Q_plus": "Foundations classified as 501(c)(3) organization", "Q_minus": "for-profit corporation"}

Example 3 (negation "not"):
Query: Which concepts are key to internetworking but are not components interconnecting participating networks?
Output: {"Q_plus": "Key concepts of internetworking", "Q_minus": "components interconnecting networks, network interconnection hardware"}

Example 4 (no negation - conjunction):
Query: Who is the brother of Moses?
Output: {"Q_plus": "Brother of Moses", "Q_minus": "[NONE]"}

Example 5 (no negation - complex):
Query: What are the nationalities of directors who have made open-source contributions?
Output: {"Q_plus": "Nationalities of directors with open-source contributions", "Q_minus": "[NONE]"}

Example 6 (negation with "other than"):
Query: What themes are expressed in Allen Ginsberg's works other than 'Howl'?
Output: {"Q_plus": "Themes expressed in Allen Ginsberg's works", "Q_minus": "Howl"}"""

COMLQ_USER_TEMPLATE = """

Now process this query:

Query: "{query}"

Output:"""

VARIANTS = OrderedDict()

for temp in [0.1]:
    variant_id = f"TSC_BALANCED_t{str(temp).replace('.', '')}"
    VARIANTS[variant_id] = {
        'system_prompt': COMLQ_SYSTEM + COMLQ_EXAMPLES,
        'user_template': COMLQ_USER_TEMPLATE,
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
        q_plus = original_query
        q_minus = "[NONE]"

        try:
            if '{' in result_text:
                json_str = result_text[result_text.index('{'):]
                if '}' in json_str:
                    json_str = json_str[:json_str.rindex('}') + 1]
                    parsed = json.loads(json_str)
                    if 'Q_plus' in parsed:
                        q_plus = parsed['Q_plus'].strip()
                    if 'Q_minus' in parsed:
                        q_minus = parsed['Q_minus'].strip()
        except (json.JSONDecodeError, ValueError):
            pass

        if not q_plus or q_plus.lower() in ['none', '[none]', 'n/a', 'null']:
            q_plus = original_query
        if not q_minus:
            q_minus = "[NONE]"

        return q_plus, q_minus


def load_comlq_queries():
    queries = {}
    query_types = {}
    data_dir = "dataset/ComLQ/dataset"
    with open(os.path.join(data_dir, "queries.jsonl"), encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            qid = str(q['_id'])
            queries[qid] = q.get('text', '')
            query_types[qid] = q.get('type', '1p')
    logger.info(f"Loaded {len(queries)} queries from ComLQ")
    return queries, query_types


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/luwa/Documents/models/Qwen3-4B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="dataset/ComLQ/dual_queries")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--variant", type=str, default="TSC_BALANCED_t01")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    queries, query_types = load_comlq_queries()

    reformulator = Qwen3Reformulator(
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
    )

    variant_config = VARIANTS[args.variant]
    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(args.output_dir, f"ComLQ_{args.variant}.jsonl")

    qids = sorted(queries.keys())
    query_texts = [queries[qid] for qid in qids]

    logger.info(f"Generating dual queries for {len(query_texts)} queries...")
    logger.info(f"Variant: {args.variant}, Temperature: {variant_config['temperature']}")

    start_time = time.time()
    q_plus_list, q_minus_list = reformulator.reformulate_batch(
        query_texts,
        variant_config['system_prompt'],
        variant_config['user_template'],
        variant_config['temperature'],
    )
    elapsed = time.time() - start_time
    logger.info(f"Generation completed in {elapsed:.1f}s ({len(query_texts)/elapsed:.1f} queries/s)")

    neg_types = {"2in", "3in", "inp", "pin", "pni"}
    neg_count = 0
    neg_minus_count = 0
    total_minus_count = 0

    with open(output_path, 'w') as f:
        for i, qid in enumerate(qids):
            q_plus = q_plus_list[i]
            q_minus = q_minus_list[i]
            qtype = query_types.get(qid, '1p')
            base_type = qtype.split("_")[0]

            is_neg = base_type in neg_types
            if is_neg:
                neg_count += 1
            if q_minus and q_minus.strip().lower() not in ['none', '[none]', 'n/a', 'null', '']:
                total_minus_count += 1
                if is_neg:
                    neg_minus_count += 1

            record = {
                "qid": qid,
                "query": queries[qid],
                "q_plus": q_plus,
                "q_minus": q_minus,
                "type": qtype,
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Total queries: {len(qids)}")
    logger.info(f"Negation queries: {neg_count}")
    logger.info(f"Q_minus available: {total_minus_count}/{len(qids)} ({100*total_minus_count/len(qids):.1f}%)")
    logger.info(f"Q_minus in negation queries: {neg_minus_count}/{neg_count} ({100*neg_minus_count/neg_count:.1f}%)" if neg_count > 0 else "No negation queries")

    del reformulator
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
