"""
BEIR Dual Query Generation with Qwen3-4B

Generates Q_plus and Q_minus for BEIR queries (no instruction).
Uses a BEIR-adapted prompt that infers implicit exclusions from the query.

Usage:
  cd /home/luwa/Documents/DSCLR && /home/luwa/.conda/envs/dsclr/bin/python -m eval.beir_reformulate \
    --dataset nq --model_path /home/luwa/Documents/models/Qwen3-4B --device cuda
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.pop("HF_ENDPOINT", None)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

import json
import re
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import datasets

logger = logging.getLogger(__name__)

BEIR_DATASET_MAP = {
    "nq": "BeIR/nq",
    "hotpotqa": "BeIR/hotpotqa",
    "quora": "BeIR/quora",
    "fiqa": "BeIR/fiqa",
    "arguana": "BeIR/arguana",
    "scidocs": "BeIR/scidocs",
    "scifact": "BeIR/scifact",
    "nfcorpus": "BeIR/nfcorpus",
    "trec-covid": "BeIR/trec-covid",
    "msmarco": "BeIR/msmarco",
    "fever": "BeIR/fever",
    "climate-fever": "BeIR/climate-fever",
    "dbpedia-entity": "BeIR/dbpedia-entity",
}

BEIR_SYSTEM_PROMPT = """You are an expert Information Retrieval query optimizer for Dense Vector Models.
Your task is to analyze a search query and produce two enhanced representations: Q_plus and Q_minus.

⚠️ CRITICAL RULES FOR Q_plus (Positive Enhancement):
1. Rewrite the query as a fluent, natural language phrase that captures the core information need.
2. Add relevant context, synonyms, or specificity that would help a dense retriever find relevant documents.
3. Remove format-related noise (e.g., "find documents about", "what is", "how does").
4. Keep it focused — do NOT add unrelated information.

⚠️ CRITICAL RULES FOR Q_minus (Negative Exclusion) — STRICT CONSERVATIVE POLICY:
Q_minus should ONLY be generated when the query contains an EXPLICIT negation/exclusion signal.
Most queries do NOT have such signals — for those, Q_minus MUST be exactly "[NONE]".

YES signals (Q_minus is warranted — the query EXPLICITLY excludes something):
  - "but not X" / "but are not X" / "but have not X"
  - "without X" / "excluding X" / "other than X"
  - "never X" / "not X" / "no X" (as exclusion, not as part of a proper noun)
  - "instead of X" / "rather than X"
  - "X is not relevant" / "X is irrelevant" / "outside of X"
  - "apart from X" / "aside from X" (when used to exclude)

NO signals (Q_minus MUST be [NONE] — these are NOT exclusions):
  - The query asks a factual question (who/what/when/where/how many) → [NONE]
  - The query mentions a concept that could be confused with something else → [NONE]
  - The query is about a specific entity (person, place, organization) → [NONE]
  - "non-X" as part of a compound term (e.g., "non-controlling interest" is a SINGLE concept) → [NONE]
  - The query is broad or ambiguous but does not explicitly exclude anything → [NONE]
  - Potential false positive directions exist but the user did not explicitly exclude them → [NONE]

THE KEY DISTINCTION:
  - "Potential confusion" ≠ "Explicit exclusion". A dense retriever might retrieve wrong results for many reasons, but Q_minus is only for cases where the user EXPLICITLY states what they do NOT want.
  - When in doubt, output [NONE]. It is ALWAYS safer to output [NONE] than to generate a spurious Q_minus.

FORMAT RULES for Q_minus:
1. Use concise keywords or short phrases (2-5 words per item).
2. NEVER use negation words in Q_minus (e.g., "no", "not", "non-", "without").
   Convert logical negations into AFFIRMATIVE entities.
   - BAD: "non-fiction" → GOOD: "fiction, novels"
   - BAD: "not about humans" → GOOD: "animal studies, plant biology"
3. Q_minus must be exactly [NONE] (with brackets) when no exclusion exists.

Output strictly in JSON format:
{
  "Reasoning_Steps": "1. Identify core information need. 2. Does the query contain an EXPLICIT negation/exclusion signal? If yes, what is excluded? If no, Q_minus=[NONE]. 3. Formulate Q_plus and Q_minus.",
  "Q_plus": "Enhanced natural language query",
  "Q_minus": "Exclusion keywords or [NONE]"
}

---
EXAMPLES:

[Example 1: Factual question — NO exclusion]
Query: what is non controlling interest on balance sheet
Output:
{
  "Reasoning_Steps": "1. Core need: definition and explanation of non-controlling interest in accounting/finance. 2. Does the query contain an explicit exclusion? NO — 'non-controlling interest' is a single compound term, not an exclusion of 'controlling interest'. The user is asking what this concept IS, not asking to exclude anything. 3. Q_plus captures the concept; Q_minus is [NONE].",
  "Q_plus": "Non-controlling interest minority interest on balance sheet accounting definition and explanation",
  "Q_minus": "[NONE]"
}

[Example 2: Specific entity question — NO exclusion]
Query: who is the governor of california
Output:
{
  "Reasoning_Steps": "1. Core need: current governor of California. 2. Does the query contain an explicit exclusion? NO — this is a straightforward factual question about a specific entity. 3. Q_plus adds context; Q_minus is [NONE].",
  "Q_plus": "Current governor of California state government executive",
  "Q_minus": "[NONE]"
}

[Example 3: Comparison question — NO exclusion]
Query: What country of origin does House of Cosbys and Bill Cosby have in common?
Output:
{
  "Reasoning_Steps": "1. Core need: the shared country of origin between the TV show House of Cosbys and Bill Cosby. 2. Does the query contain an explicit exclusion? NO — the user is asking about a shared attribute, not excluding anything. 3. Q_plus focuses on the comparison; Q_minus is [NONE].",
  "Q_plus": "Country of origin shared by House of Cosbys television show and Bill Cosby comedian actor",
  "Q_minus": "[NONE]"
}

[Example 4: Platform-specific question — NO exclusion]
Query: How does Quora look to a moderator?
Output:
{
  "Reasoning_Steps": "1. Core need: Quora platform from a moderator's perspective. 2. Does the query contain an explicit exclusion? NO — the user is asking about a specific perspective, not excluding other perspectives. 3. Q_plus adds context; Q_minus is [NONE].",
  "Q_plus": "Quora platform moderation interface tools and perspective for content moderators",
  "Q_minus": "[NONE]"
}

[Example 5: Scientific query — NO exclusion]
Query: causes of climate change
Output:
{
  "Reasoning_Steps": "1. Core need: scientific causes and drivers of climate change. 2. Does the query contain an explicit exclusion? NO — the user is asking about causes. While 'effects' might be a potential confusion, the user did not explicitly exclude it. 3. Q_plus focuses on causes; Q_minus is [NONE].",
  "Q_plus": "Scientific causes and drivers of climate change greenhouse gases natural and anthropogenic factors",
  "Q_minus": "[NONE]"
}

[Example 6: Query WITH explicit exclusion — YES, generate Q_minus]
Query: Find movies directed by Christopher Nolan but not The Dark Knight trilogy
Output:
{
  "Reasoning_Steps": "1. Core need: movies directed by Christopher Nolan. 2. Does the query contain an explicit exclusion? YES — 'but not The Dark Knight trilogy' explicitly excludes those films. 3. Q_plus captures the director; Q_minus excludes the trilogy.",
  "Q_plus": "Films directed by Christopher Nolan filmography",
  "Q_minus": "Dark Knight trilogy, Batman Begins, The Dark Knight, The Dark Knight Rises"
}

[Example 7: Query WITH explicit exclusion — YES, generate Q_minus]
Query: What are the health benefits of green tea without caffeine content information
Output:
{
  "Reasoning_Steps": "1. Core need: health benefits of green tea. 2. Does the query contain an explicit exclusion? YES — 'without caffeine content information' explicitly excludes caffeine-related content. 3. Q_plus focuses on health benefits; Q_minus excludes caffeine content.",
  "Q_plus": "Health benefits of green tea antioxidants and medicinal properties",
  "Q_minus": "caffeine content, caffeine levels"
}"""

BEIR_USER_TEMPLATE = """

Now process this query:

Query: "{query}"

Output:"""


BEIR_VARIANTS = OrderedDict()

for temp in [0.1]:
    variant_id = f"CONSERVATIVE_t{str(temp).replace('.', '')}"
    BEIR_VARIANTS[variant_id] = {
        'system_prompt': BEIR_SYSTEM_PROMPT,
        'user_template': BEIR_USER_TEMPLATE,
        'temperature': temp,
    }


class Qwen3Reformulator:
    def __init__(self, model_path, device="cuda", max_new_tokens=512):
        self.max_new_tokens = max_new_tokens
        self.batch_size = 16
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

    def reformulate(self, query, system_prompt, user_template, temperature):
        user_prompt = user_template.format(query=query)
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

    def reformulate_batch(self, queries, system_prompt, user_template, temperature):
        """Process multiple queries in a single forward pass for speed."""
        if not queries:
            return []
        # Build prompts
        texts = []
        for query in queries:
            user_prompt = user_template.format(query=query)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            texts.append(text)

        # Tokenize with left padding for batch generation
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Decode each output, stripping the prompt tokens
        input_len = inputs["input_ids"].shape[1]
        results = []
        for i, (query, output) in enumerate(zip(queries, outputs)):
            generated_ids = output[input_len:]
            result_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            results.append(self._parse_result(result_text, query))
        return results

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


def load_beir_queries(dataset_name: str) -> Dict[str, str]:
    hf_name = BEIR_DATASET_MAP.get(dataset_name, dataset_name)
    logger.info(f"Loading queries from {hf_name}...")
    ds = datasets.load_dataset(hf_name, "queries", split="queries")
    queries = {}
    for q in ds:
        qid = str(q["_id"])
        text = str(q.get("text", ""))
        queries[qid] = text
    logger.info(f"Loaded {len(queries)} queries")
    return queries


def run_variant(reformulator, variant_id, variant_config, queries, output_dir, dataset_name, output_suffix=""):
    system_prompt = variant_config['system_prompt']
    user_template = variant_config['user_template']
    temperature = variant_config['temperature']

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}{output_suffix}_{variant_id}.jsonl")

    existing = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing[r['qid']] = r
        logger.info(f"[{variant_id}] Found {len(existing)} existing entries")

    results = []
    start_time = time.time()
    failed = 0
    skipped = 0

    # Separate existing from queries needing generation
    pending = []  # list of (qid, query_text)
    for qid, query_text in queries.items():
        if not query_text:
            continue
        if qid in existing:
            results.append(existing[qid])
            skipped += 1
        else:
            pending.append((qid, query_text))

    logger.info(f"[{variant_id}] {skipped} skipped (existing), {len(pending)} to generate")

    # Process in batches
    batch_size = reformulator.batch_size
    total = len(pending)
    for batch_start in range(0, total, batch_size):
        batch = pending[batch_start:batch_start + batch_size]
        batch_queries = [q[1] for q in batch]

        try:
            batch_results = reformulator.reformulate_batch(
                batch_queries, system_prompt, user_template, temperature
            )
        except Exception as e:
            logger.warning(f"[{variant_id}] Batch failed at {batch_start}: {e}, falling back to sequential")
            batch_results = []
            for q in batch_queries:
                try:
                    batch_results.append(reformulator.reformulate(q, system_prompt, user_template, temperature))
                except Exception as e2:
                    failed += 1
                    batch_results.append((q, "[NONE]"))

        for (qid, query_text), (q_plus, q_minus) in zip(batch, batch_results):
            result = {
                "task_name": dataset_name,
                "qid": qid,
                "query": query_text,
                "instruction": "",
                "prompt_version": variant_id,
                "q_plus": q_plus,
                "q_minus": q_minus,
                "reformulator": f"Qwen3-4B-{variant_id}",
                "created_at": datetime.now().isoformat()
            }
            results.append(result)

        i = skipped + batch_start + len(batch)
        if (i) % 50 == 0 or i == len(queries) or batch_start + batch_size >= total:
            with open(output_path, 'w') as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')

            elapsed = time.time() - start_time
            new_processed = (batch_start + len(batch))
            if new_processed > 0 and elapsed > 0:
                speed = new_processed / elapsed
                remaining = total - new_processed
                eta = remaining / speed if speed > 0 else 0
                logger.info(
                    f"[{variant_id}] {i}/{len(queries)} "
                    f"(new: {new_processed}, skipped: {skipped}, failed: {failed}), "
                    f"speed: {speed:.1f} q/s, ETA: {eta:.0f}s"
                )

    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    logger.info(
        f"[{variant_id}] Done! Total: {len(results)}, "
        f"New: {len(results) - skipped}, Skipped: {skipped}, "
        f"Failed: {failed}, Time: {time.time()-start_time:.1f}s"
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="BEIR Dual Query Generation with Qwen3-4B")
    parser.add_argument("--dataset", type=str, required=True,
                        help="BEIR dataset short name (e.g., nq, hotpotqa, quora)")
    parser.add_argument("--model_path", type=str, default="/home/luwa/Documents/models/Qwen3-4B")
    parser.add_argument("--output_dir", type=str, default="dataset/BEIR/dual_queries")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_queries", type=int, default=0,
                        help="Max queries to process (0 = all)")
    parser.add_argument("--filter_qrels", action="store_true",
                        help="Only process queries that appear in qrels split")
    parser.add_argument("--qrels_split", type=str, default="test",
                        help="qrels split for filtering (test/validation/train)")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix appended to dataset name in output filename (e.g. '_dev')")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max new tokens for generation (default 256, outputs are short JSON)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for batched generation")
    parser.add_argument("--custom_queries", type=str, default=None,
                        help="Path to custom queries JSON file (format: {qid: text, ...})")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.custom_queries:
        logger.info(f"Loading custom queries from {args.custom_queries}...")
        with open(args.custom_queries) as f:
            queries = json.load(f)
        logger.info(f"Loaded {len(queries)} custom queries")
    else:
        queries = load_beir_queries(args.dataset)

    if args.filter_qrels:
        hf_name = BEIR_DATASET_MAP.get(args.dataset, args.dataset)
        qrel_dataset = f"{hf_name}-qrels"
        logger.info(f"Loading qrels from {qrel_dataset} split={args.qrels_split} for filtering...")
        ds_qrels = datasets.load_dataset(qrel_dataset, split=args.qrels_split)
        qrel_qids = set(str(item["query-id"]) for item in ds_qrels)
        before = len(queries)
        queries = {qid: text for qid, text in queries.items() if qid in qrel_qids}
        logger.info(f"Filtered to {len(queries)} queries (from {before}) that appear in qrels {args.qrels_split}")

    if args.max_queries > 0:
        queries = dict(list(queries.items())[:args.max_queries])
        logger.info(f"Limited to {len(queries)} queries")

    reformulator = Qwen3Reformulator(args.model_path, args.device, max_new_tokens=args.max_new_tokens)
    reformulator.batch_size = args.batch_size

    for variant_id, variant_config in BEIR_VARIANTS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Running variant: {variant_id}")
        logger.info(f"{'='*60}")

        output_path = run_variant(
            reformulator, variant_id, variant_config,
            queries, args.output_dir, args.dataset, args.output_suffix
        )
        logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
