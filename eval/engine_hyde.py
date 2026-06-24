"""
HyDE (Hypothetical Document Embeddings) Evaluation Engine for FollowIR

Reference: Gao et al. "Precise Zero-Shot Dense Retrieval without Relevance Labels" (ACL 2023)
Official code: https://github.com/texttron/hyde

Reproduced faithfully from the official implementation:
    - Promptor: task-specific prompts from official promptor.py (adapted for FollowIR instructions)
    - Generator: n=8 hypothesis documents, temperature=0.7 (official defaults)
    - Encoder: average of [query_emb] + [hypo_doc_emb_1, ..., hypo_doc_emb_n] (official hyde.py encode method)

FollowIR adaptation:
    - FollowIR queries have both query text and instruction
    - Prompt includes instruction to guide hypothesis document generation
    - Both og and changed queries are processed independently
    - p-MRR and target_avg metrics are computed

Usage:
    cd /home/luwa/Documents/DSCLR && /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_hyde \
        --task_name Core17InstructionRetrieval \
        --model_name samaya-ai/RepLLaMA-reproduced \
        --llm_model_path /home/luwa/Documents/models/Qwen3-4B \
        --device cuda \
        --output_dir results/hyde/Core17

    # Run all three datasets:
    cd /home/luwa/Documents/DSCLR && for task in Core17InstructionRetrieval Robust04InstructionRetrieval News21InstructionRetrieval; do
        /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_hyde \
            --task_name $task \
            --model_name samaya-ai/RepLLaMA-reproduced \
            --llm_model_path /home/luwa/Documents/models/Qwen3-4B \
            --device cuda \
            --output_dir results/hyde/${task/InstructionRetrieval/}
    done
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from eval.metrics import DataLoader as MetricsDataLoader, FollowIREvaluator
from eval.models.repllama_encoder import RepLLaMAEncoder
from eval.engine_dscrl import load_cached_embeddings, save_embeddings_cache

logger = logging.getLogger(__name__)


WEB_SEARCH_PROMPT = """Please write a passage to answer the question.
Question: {}
Passage:"""

WEB_SEARCH_PROMPT_WITH_INSTRUCTION = """Please write a passage to answer the question.
Question: {}
Instruction: {}
Passage:"""


class Promptor:
    """Faithful reproduction of official HyDE Promptor (texttron/hyde/src/hyde/promptor.py)

    Extended to support FollowIR instructions.
    """

    def __init__(self, task: str = "web search"):
        self.task = task

    def build_prompt(self, query: str, instruction: str = "") -> str:
        if instruction:
            return WEB_SEARCH_PROMPT_WITH_INSTRUCTION.format(query, instruction)
        if self.task == "web search":
            return WEB_SEARCH_PROMPT.format(query)
        elif self.task == "scifact":
            return "Please write a scientific paper passage to support/refute the claim.\nClaim: {}\nPassage:".format(query)
        elif self.task == "arguana":
            return "Please write a counter argument for the passage.\nPassage: {}\nCounter Argument:".format(query)
        elif self.task == "trec-covid":
            return "Please write a scientific paper passage to answer the question.\nQuestion: {}\nPassage:".format(query)
        elif self.task == "fiqa":
            return "Please write a financial article passage to answer the question.\nQuestion: {}\nPassage:".format(query)
        elif self.task == "dbpedia-entity":
            return "Please write a passage to answer the question.\nQuestion: {}\nPassage:".format(query)
        elif self.task == "trec-news":
            return "Please write a news passage about the topic.\nTopic: {}\nPassage:".format(query)
        else:
            return WEB_SEARCH_PROMPT.format(query)


class LocalGenerator:
    """Local LLM generator replacing OpenAI/Cohere in official code (texttron/hyde/src/hyde/generator.py)

    Faithful to official defaults: n=8, max_tokens=512, temperature=0.7
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        n: int = 8,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.device = device

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"📥 Loading HyDE generator LLM from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        effective_device = device
        if device == "cuda":
            try:
                torch.cuda._lazy_init()
                if not torch.cuda.is_available():
                    effective_device = "cpu"
            except Exception:
                effective_device = "cpu"

        if effective_device == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float32, trust_remote_code=True
            ).to("cpu")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            )
        self.model.eval()
        logger.info(f"✅ HyDE generator loaded on {effective_device} (n={n}, temp={temperature})")

    def generate(self, prompt: str) -> List[str]:
        """Generate n hypothesis documents from a single prompt.

        Faithful to official OpenAIGenerator.generate() which returns List[str] of n texts.
        """
        texts = []
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        for _ in range(self.n):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    top_p=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            texts.append(text)

        return texts


class HyDE:
    """Faithful reproduction of official HyDE class (texttron/hyde/src/hyde/hyde.py)

    Core encode method: average of [query_emb] + [hypo_doc_emb_1, ..., hypo_doc_emb_n]
    """

    def __init__(self, promptor: Promptor, generator: LocalGenerator, encoder: RepLLaMAEncoder):
        self.promptor = promptor
        self.generator = generator
        self.encoder = encoder

    def prompt(self, query: str, instruction: str = "") -> str:
        return self.promptor.build_prompt(query, instruction)

    def generate(self, query: str, instruction: str = "") -> List[str]:
        prompt = self.promptor.build_prompt(query, instruction)
        hypothesis_documents = self.generator.generate(prompt)
        return hypothesis_documents

    def encode(self, query_text: str, hypothesis_documents: List[str]) -> torch.Tensor:
        """Faithful reproduction of official HyDE.encode()

        Official code:
            all_emb_c = []
            for c in [query] + hypothesis_documents:
                c_emb = self.encoder.encode(c)
                all_emb_c.append(np.array(c_emb))
            all_emb_c = np.array(all_emb_c)
            avg_emb_c = np.mean(all_emb_c, axis=0)
            hyde_vector = avg_emb_c.reshape((1, len(avg_emb_c)))
            return hyde_vector

        Adaptation for RepLLaMA (which has separate query/document templates):
            - query_text is encoded with query template (it IS a query)
            - hypothesis_documents are encoded with document template (they ARE documents)
        """
        query_emb = self.encoder.encode_queries([query_text])
        hypo_embs = self.encoder.encode_documents(hypothesis_documents)

        all_embs = torch.cat([query_emb, hypo_embs], dim=0)
        avg_emb = torch.mean(all_embs, dim=0, keepdim=True)
        avg_emb = F.normalize(avg_emb, p=2, dim=1)
        return avg_emb


class HyDEDataLoader:
    """FollowIR data loader for HyDE evaluation"""

    def __init__(self, task_name: str):
        self.task_name = task_name
        self.metrics_loader = MetricsDataLoader(task_name)

    def load(self):
        logger.info(f"📂 Loading dataset: {self.task_name}")
        self.corpus = self.metrics_loader.load_corpus()
        self.q_og, self.q_changed = self.metrics_loader.load_queries()
        self.candidates = self.metrics_loader.load_candidates()
        logger.info(
            f"✅ Data loaded: {len(self.corpus)} docs, "
            f"{len(self.q_og)} og queries, {len(self.q_changed)} changed queries"
        )
        return self.corpus, self.q_og, self.q_changed, self.candidates

    def load_raw_queries(self):
        return self.metrics_loader.load_raw_queries()


class HyDEEvaluator:
    """HyDE evaluation engine for FollowIR

    Faithfully reproduces the official HyDE pipeline from texttron/hyde
    """

    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_dir: str,
        llm_model_path: str,
        device: str = "cuda",
        batch_size: int = 32,
        hyde_n: int = 8,
        hyde_temperature: float = 0.7,
        hyde_max_tokens: int = 512,
        cache_dir: str = "dataset/FollowIR_test/embeddings",
        hyde_cache_dir: str = "dataset/FollowIR_test/hyde_docs",
        use_cache: bool = True,
        gpu_id: int = 0,
    ):
        self.model_name = model_name
        self.task_name = task_name
        self.output_dir = output_dir
        self.llm_model_path = llm_model_path
        self.device = device
        self.batch_size = batch_size
        self.hyde_n = hyde_n
        self.hyde_temperature = hyde_temperature
        self.hyde_max_tokens = hyde_max_tokens
        self.cache_dir = cache_dir
        self.hyde_cache_dir = hyde_cache_dir
        self.use_cache = use_cache
        self.gpu_id = gpu_id

        self.data_loader = HyDEDataLoader(task_name)
        self.encoder = None

        self._init_hyde_cache()

    def _init_encoder(self):
        logger.info(f"📥 Loading encoder: {self.model_name} on GPU {self.gpu_id}")
        self.encoder = RepLLaMAEncoder(
            model_name=self.model_name,
            device=f"cuda:{self.gpu_id}",
            batch_size=self.batch_size,
        )
        logger.info("✅ Encoder loaded")

    def _init_hyde_cache(self):
        os.makedirs(self.hyde_cache_dir, exist_ok=True)

    def _get_hyde_cache_path(self) -> str:
        llm_name = os.path.basename(self.llm_model_path)
        return os.path.join(
            self.hyde_cache_dir,
            f"{self.task_name}_hyde_{llm_name}_n{self.hyde_n}_t{self.hyde_temperature}.jsonl",
        )

    def _load_hyde_cache(self) -> Dict[str, List[str]]:
        cache_path = self._get_hyde_cache_path()
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        cache[item["qid"]] = item["hypo_docs"]
            logger.info(f"📂 Loaded HyDE cache: {len(cache)} entries from {cache_path}")
        return cache

    def _save_hyde_cache(self, cache: Dict[str, List[str]], queries_info: Dict[str, Dict]):
        cache_path = self._get_hyde_cache_path()
        with open(cache_path, "w", encoding="utf-8") as f:
            for qid, hypo_docs in cache.items():
                info = queries_info.get(qid, {})
                f.write(json.dumps({
                    "qid": qid,
                    "query": info.get("query", ""),
                    "instruction": info.get("instruction", ""),
                    "hypo_docs": hypo_docs,
                    "n": len(hypo_docs),
                }, ensure_ascii=False) + "\n")
        logger.info(f"💾 HyDE cache saved: {cache_path}")

    def _get_all_candidate_doc_ids(self, candidates: Dict[str, List[str]]) -> List[str]:
        all_ids = set()
        for doc_ids in candidates.values():
            all_ids.update(doc_ids)
        return sorted(all_ids)

    def _encode_documents(self, doc_ids: List[str], doc_texts: List[str]) -> torch.Tensor:
        cached_data = load_cached_embeddings(self.cache_dir, self.task_name, self.model_name)
        if cached_data is not None:
            cached_embeddings, cached_doc_ids = cached_data
            if set(cached_doc_ids) == set(doc_ids):
                logger.info(f"✅ Using cached document embeddings ({len(cached_doc_ids)} docs)")
                id_to_idx = {did: i for i, did in enumerate(cached_doc_ids)}
                ordered = torch.stack([cached_embeddings[id_to_idx[did]] for did in doc_ids])
                return ordered
            else:
                logger.warning("⚠️ Cached doc IDs mismatch, re-encoding...")
        else:
            logger.info("📚 Encoding candidate documents...")

        embeddings = self.encoder.encode_documents(doc_texts, batch_size=self.batch_size)
        if embeddings.dim() == 2:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        if self.use_cache:
            save_embeddings_cache(self.cache_dir, self.task_name, embeddings, doc_ids, self.model_name)

        return embeddings

    def _extract_results(
        self,
        hyde_vectors: torch.Tensor,
        query_ids: List[str],
        doc_embeddings: torch.Tensor,
        candidates: Dict[str, List[str]],
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        score_matrix = torch.matmul(hyde_vectors, doc_embeddings.T)

        for q_idx, qid in enumerate(query_ids):
            base_qid = qid.replace("-og", "").replace("-changed", "")
            cand_ids = candidates.get(base_qid, [])
            if not cand_ids:
                continue

            doc_id_to_col_idx = {did: i for i, did in enumerate(self.doc_ids)}
            scores = {}
            for did in cand_ids:
                if did in doc_id_to_col_idx:
                    scores[did] = float(score_matrix[q_idx, doc_id_to_col_idx[did]].item())
            results[qid] = scores
        return results

    def run(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("🚀 Starting HyDE Evaluation on FollowIR")
        logger.info(f"   Official code: https://github.com/texttron/hyde")
        logger.info(f"   n_hypo_docs={self.hyde_n}, temperature={self.hyde_temperature}")
        logger.info("=" * 60)

        start_time = time.time()

        corpus, q_og, q_changed, candidates = self.data_loader.load()
        q_raw_og, q_raw_changed = self.data_loader.load_raw_queries()

        hyde_cache = self._load_hyde_cache()
        queries_info = {}

        query_ids_og: List[str] = []
        query_ids_changed: List[str] = []
        queries_to_generate: List[Tuple[str, str, str]] = []

        for qid in q_og.keys():
            query_ids_og.append(qid)
            raw = q_raw_og.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            queries_info[qid] = {"query": query_text, "instruction": instruction}
            if qid not in hyde_cache:
                queries_to_generate.append((qid, query_text, instruction))

        for qid in q_changed.keys():
            query_ids_changed.append(qid)
            raw = q_raw_changed.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            queries_info[qid] = {"query": query_text, "instruction": instruction}
            if qid not in hyde_cache:
                queries_to_generate.append((qid, query_text, instruction))

        if queries_to_generate:
            logger.info(f"🔄 Phase 1: Generating hypothesis documents for {len(queries_to_generate)} queries (n={self.hyde_n} each)...")
            promptor = Promptor(task="web search")
            generator = LocalGenerator(
                model_path=self.llm_model_path,
                device=f"cuda:{self.gpu_id}",
                n=self.hyde_n,
                max_tokens=self.hyde_max_tokens,
                temperature=self.hyde_temperature,
            )

            for qid, query_text, instruction in tqdm(queries_to_generate, desc="HyDE generation"):
                prompt = promptor.build_prompt(query_text, instruction)
                hypo_docs = generator.generate(prompt)
                hyde_cache[qid] = hypo_docs

            self._save_hyde_cache(hyde_cache, queries_info)
            del generator
            torch.cuda.empty_cache()
            logger.info("✅ Hypothesis documents generated and cached, generator unloaded")
        else:
            logger.info(f"✅ All hypothesis documents already cached ({len(hyde_cache)} entries)")

        logger.info("🔄 Phase 2: Loading encoder and computing HyDE vectors...")
        self._init_encoder()

        all_doc_ids = self._get_all_candidate_doc_ids(candidates)
        self.doc_ids = all_doc_ids

        doc_texts = [corpus[did]["text"] for did in all_doc_ids]
        doc_embeddings = self._encode_documents(all_doc_ids, doc_texts)
        doc_embeddings = doc_embeddings.to(f"cuda:{self.gpu_id}")

        logger.info("📊 Computing HyDE vectors (official encode method: avg of query + hypo_docs)...")
        hyde_vectors_og = []
        for qid in tqdm(query_ids_og, desc="Encoding OG HyDE vectors"):
            combined_query = q_og[qid]
            hypo_docs = hyde_cache[qid]

            query_emb = self.encoder.encode_queries([combined_query])
            hypo_embs = self.encoder.encode_documents(hypo_docs, batch_size=self.batch_size)
            all_embs = torch.cat([query_emb, hypo_embs], dim=0)
            avg_emb = torch.mean(all_embs, dim=0, keepdim=True)
            avg_emb = F.normalize(avg_emb, p=2, dim=1)
            hyde_vectors_og.append(avg_emb)

        hyde_vectors_og = torch.cat(hyde_vectors_og, dim=0).to(f"cuda:{self.gpu_id}")

        hyde_vectors_changed = []
        for qid in tqdm(query_ids_changed, desc="Encoding Changed HyDE vectors"):
            combined_query = q_changed[qid]
            hypo_docs = hyde_cache[qid]

            query_emb = self.encoder.encode_queries([combined_query])
            hypo_embs = self.encoder.encode_documents(hypo_docs, batch_size=self.batch_size)
            all_embs = torch.cat([query_emb, hypo_embs], dim=0)
            avg_emb = torch.mean(all_embs, dim=0, keepdim=True)
            avg_emb = F.normalize(avg_emb, p=2, dim=1)
            hyde_vectors_changed.append(avg_emb)

        hyde_vectors_changed = torch.cat(hyde_vectors_changed, dim=0).to(f"cuda:{self.gpu_id}")

        logger.info("📊 Computing FollowIR metrics...")
        results_og = self._extract_results(hyde_vectors_og, query_ids_og, doc_embeddings, candidates)
        results_changed = self._extract_results(hyde_vectors_changed, query_ids_changed, doc_embeddings, candidates)

        evaluator = FollowIREvaluator(self.task_name)
        metrics = evaluator.evaluate(results_og, results_changed)

        elapsed = time.time() - start_time

        p_mrr = metrics.get("p-MRR", 0.0)
        og_map = metrics.get("original", {}).get("map_at_1000", 0.0)
        changed_map = metrics.get("changed", {}).get("map_at_1000", 0.0)
        og_ndcg5 = metrics.get("original", {}).get("ndcg_at_5", 0.0)
        changed_ndcg5 = metrics.get("changed", {}).get("ndcg_at_5", 0.0)

        logger.info("=" * 60)
        logger.info("📊 HyDE Evaluation Results (Official Reproduction)")
        logger.info(f"   Task: {self.task_name}")
        logger.info(f"   Encoder: {self.model_name}")
        logger.info(f"   HyDE LLM: {os.path.basename(self.llm_model_path)}")
        logger.info(f"   n_hypo_docs: {self.hyde_n}, temperature: {self.hyde_temperature}")
        logger.info(f"   p-MRR: {p_mrr:.4f}")
        logger.info(f"   OG MAP@1000: {og_map:.4f}")
        logger.info(f"   Changed MAP@1000: {changed_map:.4f}")
        logger.info(f"   OG nDCG@5: {og_ndcg5:.4f}")
        logger.info(f"   Changed nDCG@5: {changed_ndcg5:.4f}")
        logger.info(f"   Elapsed: {elapsed:.1f}s")
        logger.info("=" * 60)

        self._save_results(
            metrics=metrics,
            results_og=results_og,
            results_changed=results_changed,
            hyde_cache=hyde_cache,
            query_ids_og=query_ids_og,
            query_ids_changed=query_ids_changed,
            elapsed=elapsed,
        )

        return {
            "metrics": metrics,
            "elapsed": elapsed,
        }

    def _save_results(
        self,
        metrics: Dict[str, Any],
        results_og: Dict[str, Dict[str, float]],
        results_changed: Dict[str, Dict[str, float]],
        hyde_cache: Dict[str, List[str]],
        query_ids_og: List[str],
        query_ids_changed: List[str],
        elapsed: float,
    ):
        os.makedirs(self.output_dir, exist_ok=True)

        summary = {
            "task": self.task_name,
            "model": self.model_name,
            "mode": "HyDE (official reproduction from texttron/hyde)",
            "hyde_llm": self.llm_model_path,
            "hyde_n": self.hyde_n,
            "hyde_temperature": self.hyde_temperature,
            "hyde_max_tokens": self.hyde_max_tokens,
            "hyde_encode_method": "avg(query_emb, hypo_doc_emb_1, ..., hypo_doc_emb_n) - official",
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "p-MRR": metrics.get("p-MRR", 0.0),
                "original": metrics.get("original", {}),
                "changed": metrics.get("changed", {}),
                "full_scores": metrics.get("full_scores", {}),
            },
            "elapsed_seconds": elapsed,
        }

        summary_path = os.path.join(self.output_dir, "metrics_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        out_og = os.path.join(self.output_dir, "ranking_og.json")
        out_changed = os.path.join(self.output_dir, "ranking_changed.json")
        with open(out_og, "w", encoding="utf-8") as f:
            json.dump(results_og, f, ensure_ascii=False)
        with open(out_changed, "w", encoding="utf-8") as f:
            json.dump(results_changed, f, ensure_ascii=False)

        hypo_samples_path = os.path.join(self.output_dir, "hyde_samples.json")
        samples = {}
        for qid in query_ids_og[:5]:
            samples[qid] = {
                "type": "og",
                "hypo_docs": hyde_cache.get(qid, [])[:2],
            }
        for qid in query_ids_changed[:5]:
            samples[qid] = {
                "type": "changed",
                "hypo_docs": hyde_cache.get(qid, [])[:2],
            }
        with open(hypo_samples_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="HyDE Evaluation Engine for FollowIR (Official Reproduction)")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval",
                        choices=["Core17InstructionRetrieval", "Robust04InstructionRetrieval",
                                 "News21InstructionRetrieval"])
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--llm_model_path", type=str, default="/home/luwa/Documents/models/Qwen3-4B")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hyde_n", type=int, default=8,
                        help="Number of hypothesis documents per query (official default: 8)")
    parser.add_argument("--hyde_temperature", type=float, default=0.7,
                        help="Generation temperature (official default: 0.7)")
    parser.add_argument("--hyde_max_tokens", type=int, default=512,
                        help="Max tokens for hypothesis document generation (official default: 512)")
    parser.add_argument("--cache_dir", type=str, default="dataset/FollowIR_test/embeddings")
    parser.add_argument("--hyde_cache_dir", type=str, default="dataset/FollowIR_test/hyde_docs")
    parser.add_argument("--use_cache", type=str, default="true")

    args = parser.parse_args()

    if args.output_dir is None:
        short_name = args.task_name.replace("InstructionRetrieval", "")
        args.output_dir = f"results/hyde/{short_name}"

    use_cache = args.use_cache.lower() in ("true", "1", "yes")

    evaluator = HyDEEvaluator(
        model_name=args.model_name,
        task_name=args.task_name,
        output_dir=args.output_dir,
        llm_model_path=args.llm_model_path,
        device=args.device,
        batch_size=args.batch_size,
        hyde_n=args.hyde_n,
        hyde_temperature=args.hyde_temperature,
        hyde_max_tokens=args.hyde_max_tokens,
        cache_dir=args.cache_dir,
        hyde_cache_dir=args.hyde_cache_dir,
        use_cache=use_cache,
        gpu_id=args.gpu_id,
    )

    evaluator.run()


if __name__ == "__main__":
    main()
