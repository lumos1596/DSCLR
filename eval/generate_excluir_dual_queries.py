"""
为 ExcluIR 基准生成 dual queries (Q_plus, Q_minus)

使用本地 Qwen3-4B 模型，将 ExcluIR 的排他性查询分解为:
  - Q_plus: 用户实际想要检索的内容
  - Q_minus: 需要排除的内容

ExcluIR 查询格式: "What are some tourist attractions in Paris besides the Eiffel Tower?"
  → Q_plus: "Tourist attractions in Paris"
  → Q_minus: "Eiffel Tower"

Usage:
  python -m eval.generate_excluir_dual_queries \
    --data_dir dataset/ExcluIR \
    --model_path /home/luwa/Documents/models/Qwen3-4B \
    --output_path dataset/ExcluIR/dual_queries/dual_queries_excluir.jsonl \
    --device cuda
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


EXCLUIR_SYSTEM_PROMPT = """Your task: Decompose an exclusionary query into Q_plus (what the user wants) and Q_minus (what the user wants to exclude).

An exclusionary query explicitly states that something should NOT be included in the search results.

STEP 1 — Identify the exclusion signal in the query:
  - "besides X" / "other than X" / "apart from X" → X is excluded
  - "not including X" / "excluding X" / "without X" → X is excluded
  - "instead of X" / "rather than X" → X is excluded
  - "different from X" / "other X" → X is excluded
  - "aside from X" → X is excluded

STEP 2 — Extract:
  - Q_plus: What the user is ACTUALLY looking for (the main topic, WITHOUT the exclusion)
  - Q_minus: The specific entity/topic being excluded (short keywords or phrases)

Output JSON: {"Q_plus": "...", "Q_minus": "..."}

FORMAT RULES:
- Q_plus should be a natural search query describing what the user wants to find
- Q_minus should contain ONLY the excluded entity/topic as short keywords
- Q_plus must NOT contain the exclusion signal (no "besides", "other than", etc.)
- Q_minus must NOT contain anything that is also in Q_plus

Examples:

Query: "What are some tourist attractions in Paris besides the Eiffel Tower?"
Output: {"Q_plus": "Tourist attractions in Paris", "Q_minus": "Eiffel Tower"}

Query: "What other sci-fi movies were released in 2019 besides Avengers: Endgame?"
Output: {"Q_plus": "Sci-fi movies released in 2019", "Q_minus": "Avengers: Endgame"}

Query: "Find information about world capitals other than London"
Output: {"Q_plus": "World capitals", "Q_minus": "London"}

Query: "What are some programming languages besides Python that are used for data science?"
Output: {"Q_plus": "Programming languages used for data science", "Q_minus": "Python"}

Query: "Name some famous painters apart from Picasso"
Output: {"Q_plus": "Famous painters", "Q_minus": "Picasso"}

Query: "What other countries in Europe have a monarchy besides the United Kingdom?"
Output: {"Q_plus": "Countries in Europe with a monarchy", "Q_minus": "United Kingdom"}"""


USER_TEMPLATE = """

Now decompose this exclusionary query:

Query: "{query}"

Output:"""


class ExcluIRDualQueryGenerator:
    def __init__(self, model_path: str, device: str = "cuda", max_new_tokens: int = 512):
        self.max_new_tokens = max_new_tokens
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

    def generate(self, query: str, temperature: float = 0.1) -> tuple:
        """生成 Q_plus 和 Q_minus"""
        user_prompt = USER_TEMPLATE.format(query=query)
        messages = [
            {"role": "system", "content": EXCLUIR_SYSTEM_PROMPT},
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

    def _parse_result(self, result_text: str, original_query: str) -> tuple:
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


def main():
    parser = argparse.ArgumentParser(description="Generate ExcluIR dual queries with Qwen3-4B")
    parser.add_argument("--data_dir", type=str, default="dataset/ExcluIR")
    parser.add_argument("--model_path", type=str, default="/home/luwa/Documents/models/Qwen3-4B")
    parser.add_argument("--output_path", type=str,
                        default="dataset/ExcluIR/dual_queries/dual_queries_excluir.jsonl")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load ExcluIR data
    queries_path = os.path.join(args.data_dir, "test_manual_final.json")
    logger.info(f"Loading queries from {queries_path}...")
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    logger.info(f"Loaded {len(queries)} queries")

    # Resume support
    existing = {}
    if args.resume and os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing[r['qid']] = r
        logger.info(f"Resuming from {len(existing)} existing entries")

    # Initialize generator
    generator = ExcluIRDualQueryGenerator(args.model_path, args.device)

    # Generate
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    results = list(existing.values())
    start_time = time.time()
    failed = 0

    for i, q in enumerate(queries):
        if i in existing:
            continue

        query_text = q.get("ExcluQ", q.get("RQ_rewrite", ""))

        if not query_text:
            results.append({
                "qid": i,
                "query": "",
                "q_plus": "",
                "q_minus": "[NONE]",
                "reformulator": "Qwen3-4B-ExcluIR",
                "error": "empty query",
                "created_at": datetime.now().isoformat(),
            })
            failed += 1
            continue

        try:
            q_plus, q_minus = generator.generate(query_text, temperature=args.temperature)
            results.append({
                "qid": i,
                "query": query_text,
                "q_plus": q_plus,
                "q_minus": q_minus,
                "reformulator": "Qwen3-4B-ExcluIR",
                "created_at": datetime.now().isoformat(),
            })
        except Exception as e:
            failed += 1
            results.append({
                "qid": i,
                "query": query_text,
                "q_plus": query_text,
                "q_minus": "[NONE]",
                "reformulator": "Qwen3-4B-ExcluIR",
                "error": str(e),
                "created_at": datetime.now().isoformat(),
            })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(queries) - i - 1) / speed if speed > 0 else 0
            logger.info(
                f"Progress: {i+1}/{len(queries)} ({(i+1)/len(queries)*100:.1f}%), "
                f"speed: {speed:.1f} q/s, ETA: {eta:.0f}s, failed: {failed}"
            )

        # Save incrementally
        if (i + 1) % 50 == 0:
            with open(args.output_path, "w", encoding="utf-8") as f:
                for r in sorted(results, key=lambda x: x['qid']):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Final save
    with open(args.output_path, "w", encoding="utf-8") as f:
        for r in sorted(results, key=lambda x: x['qid']):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    logger.info(f"Done! Processed: {len(results)}, Failed: {failed}, Time: {elapsed:.1f}s")
    logger.info(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
