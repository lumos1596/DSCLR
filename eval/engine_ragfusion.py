"""
RAG-Fusion Evaluation Engine for FollowIR

Reference: Raudaschl, "RAG-Fusion: a New Take on Retrieval-Augmented Generation" (2024)
Paper: https://arxiv.org/abs/2402.03367
Official code: https://github.com/Raudaschl/rag-fusion

Faithfully reproduced from the official code (main.py):
    - Query Generation: LLM generates 4 diverse search queries (official: GPT-3.5, here: Qwen3-4B)
    - Vector Search: Each query (original + 4 generated) independently retrieves documents
    - Reciprocal Rank Fusion: RRF combines ranked lists with score = Σ 1/(rank + k), k=60
    - Original query is included in the search alongside generated queries

FollowIR adaptation:
    - Both og and changed queries are processed independently
    - Each generates 4 additional queries, then RRF fuses 5 retrieval results
    - p-MRR and target_avg metrics are computed

Usage:
    cd /home/luwa/Documents/DSCLR && /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_ragfusion \
        --task_name Core17InstructionRetrieval \
        --model_name samaya-ai/RepLLaMA-reproduced \
        --llm_model_path /home/luwa/Documents/models/Qwen3-4B \
        --device cuda \
        --output_dir results/ragfusion/Core17
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

import torch
import torch.nn.functional as F
from tqdm import tqdm

from eval.metrics import DataLoader as MetricsDataLoader, FollowIREvaluator
from eval.models.repllama_encoder import RepLLaMAEncoder
from eval.engine_dscrl import load_cached_embeddings, save_embeddings_cache

logger = logging.getLogger(__name__)


RAGFUSION_PROMPT_DEFAULT = [
    {"role": "system", "content": "You are a helpful assistant that generates multiple search queries based on a single input query."},
    {"role": "user", "content": "Generate multiple search queries related to: {query}"},
    {"role": "user", "content": "OUTPUT (4 queries):"},
]

RAGFUSION_PROMPT_DIVERSE = [
    {"role": "system", "content": "You are a search expert. Generate diverse search queries that explore different aspects of the user's question. Each query should target a different angle: use synonyms, vary specificity (broader/narrower), and consider related sub-topics. Avoid generating queries that are just minor rewordings of each other."},
    {"role": "user", "content": "Generate 4 diverse search queries for: {query}"},
    {"role": "user", "content": "OUTPUT (4 queries):"},
]

RAGFUSION_PROMPT_WITH_INSTRUCTION = [
    {"role": "system", "content": "You are a helpful assistant that generates multiple search queries based on a single input query and its instruction."},
    {"role": "user", "content": "Generate multiple search queries related to: {query}\nInstruction: {instruction}"},
    {"role": "user", "content": "OUTPUT (4 queries):"},
]


class RAGFusionPromptor:
    """Faithful reproduction of RAG-Fusion prompting (Raudaschl/rag-fusion main.py)

    Official code has two modes:
    - diverse=False (default): simple prompt "Generate multiple search queries related to: {query}"
    - diverse=True: detailed prompt with synonyms, specificity, sub-topics
    Both generate 4 queries.
    """

    def __init__(self, diverse: bool = False, n_queries: int = 4):
        self.diverse = diverse
        self.n_queries = n_queries

    def build_prompt(self, query: str, instruction: str = "") -> List[Dict[str, str]]:
        if instruction:
            messages = []
            for msg in RAGFUSION_PROMPT_WITH_INSTRUCTION:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"].format(query=query, instruction=instruction)
                })
            return messages

        template = RAGFUSION_PROMPT_DIVERSE if self.diverse else RAGFUSION_PROMPT_DEFAULT
        messages = []
        for msg in template:
            messages.append({
                "role": msg["role"],
                "content": msg["content"].format(query=query)
            })
        return messages


class RAGFusionGenerator:
    """Local LLM generator for RAG-Fusion multi-query generation

    Official code uses GPT-3.5-turbo. Here we use local Qwen3-4B for fair comparison.
    Generates 4 diverse search queries per input query.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_tokens: int = 256,
        temperature: float = 0.7,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.device = device

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"📥 Loading RAG-Fusion generator LLM from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        effective_device = device
        if device.startswith("cuda"):
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
        logger.info(f"✅ RAG-Fusion generator loaded on {effective_device}")

    def generate(self, messages: List[Dict[str, str]]) -> List[str]:
        """Generate multiple search queries from chat-style messages.

        Official code: response.choices[0].message.content.strip().split("\\n")
        Returns list of generated queries.
        """
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        queries = [q.strip().lstrip("0123456789.-) ") for q in text.split("\n") if q.strip()]
        return queries


def reciprocal_rank_fusion(
    search_results_dict: Dict[str, Dict[str, float]],
    k: int = 60,
) -> Dict[str, float]:
    """Faithful reproduction of RRF from Raudaschl/rag-fusion main.py

    Official code:
        for query, doc_scores in search_results_dict.items():
            weight = query_weights.get(query, 1.0) if query_weights else 1.0
            for rank, (doc, _) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
                if doc not in fused_scores:
                    fused_scores[doc] = 0
                previous_score = fused_scores[doc]
                fused_scores[doc] += weight * (1 / (rank + k))

    Args:
        search_results_dict: {query_name: {doc_id: score}} for each query
        k: RRF constant (official default: 60)

    Returns:
        Dict of {doc_id: fused_score}, sorted descending
    """
    fused_scores = {}

    for query, doc_scores in search_results_dict.items():
        for rank, (doc, _) in enumerate(
            sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        ):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            fused_scores[doc] += 1.0 / (rank + k)

    return dict(sorted(fused_scores.items(), key=lambda x: x[1], reverse=True))


class RAGFusionDataLoader:
    """FollowIR data loader for RAG-Fusion evaluation"""

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


class RAGFusionEvaluator:
    """RAG-Fusion evaluation engine for FollowIR

    Faithfully reproduces the official RAG-Fusion pipeline from Raudaschl/rag-fusion
    """

    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_dir: str,
        llm_model_path: str,
        device: str = "cuda",
        batch_size: int = 32,
        rf_n_queries: int = 4,
        rf_diverse: bool = False,
        rf_temperature: float = 0.7,
        rf_max_tokens: int = 256,
        rrf_k: int = 60,
        cache_dir: str = "dataset/FollowIR_test/embeddings",
        rf_cache_dir: str = "dataset/FollowIR_test/ragfusion_queries",
        use_cache: bool = True,
        gpu_id: int = 0,
    ):
        self.model_name = model_name
        self.task_name = task_name
        self.output_dir = output_dir
        self.llm_model_path = llm_model_path
        self.device = device
        self.batch_size = batch_size
        self.rf_n_queries = rf_n_queries
        self.rf_diverse = rf_diverse
        self.rf_temperature = rf_temperature
        self.rf_max_tokens = rf_max_tokens
        self.rrf_k = rrf_k
        self.cache_dir = cache_dir
        self.rf_cache_dir = rf_cache_dir
        self.use_cache = use_cache
        self.gpu_id = gpu_id

        self.data_loader = RAGFusionDataLoader(task_name)
        self.encoder = None

        self._init_rf_cache()

    def _init_encoder(self):
        logger.info(f"📥 Loading encoder: {self.model_name} on GPU {self.gpu_id}")
        self.encoder = RepLLaMAEncoder(
            model_name=self.model_name,
            device=f"cuda:{self.gpu_id}",
            batch_size=self.batch_size,
        )
        logger.info("✅ Encoder loaded")

    def _init_rf_cache(self):
        os.makedirs(self.rf_cache_dir, exist_ok=True)

    def _get_rf_cache_path(self) -> str:
        llm_name = os.path.basename(self.llm_model_path)
        mode = "diverse" if self.rf_diverse else "default"
        return os.path.join(
            self.rf_cache_dir,
            f"{self.task_name}_ragfusion_{llm_name}_{mode}_t{self.rf_temperature}.jsonl",
        )

    def _load_rf_cache(self) -> Dict[str, List[str]]:
        cache_path = self._get_rf_cache_path()
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        cache[item["qid"]] = item["generated_queries"]
            logger.info(f"📂 Loaded RAG-Fusion cache: {len(cache)} entries from {cache_path}")
        return cache

    def _save_rf_cache(self, cache: Dict[str, List[str]], queries_info: Dict[str, Dict]):
        cache_path = self._get_rf_cache_path()
        with open(cache_path, "w", encoding="utf-8") as f:
            for qid, gen_queries in cache.items():
                info = queries_info.get(qid, {})
                f.write(json.dumps({
                    "qid": qid,
                    "query": info.get("query", ""),
                    "instruction": info.get("instruction", ""),
                    "generated_queries": gen_queries,
                }, ensure_ascii=False) + "\n")
        logger.info(f"💾 RAG-Fusion cache saved: {cache_path}")

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

    def _retrieve_with_single_query(
        self,
        query_vector: torch.Tensor,
        doc_embeddings: torch.Tensor,
        candidate_doc_ids: List[str],
    ) -> Dict[str, float]:
        scores = torch.matmul(query_vector, doc_embeddings.T).squeeze(0)
        results = {}
        for idx, did in enumerate(candidate_doc_ids):
            results[did] = float(scores[idx].item())
        return results

    def _rrf_retrieve(
        self,
        original_query: str,
        generated_queries: List[str],
        doc_embeddings: torch.Tensor,
        candidate_doc_ids: List[str],
    ) -> Dict[str, float]:
        """RAG-Fusion retrieval: encode each query, retrieve, then RRF fuse.

        Faithful to official code:
            all_queries = [original_query] + generated_queries
            for query in all_queries:
                search_results = vector_search(query, collection)
                all_results[query] = search_results
            ranked_results = reciprocal_rank_fusion(all_results)
        """
        all_queries = [original_query] + generated_queries
        all_results = {}

        query_vectors = self.encoder.encode_queries(all_queries)
        query_vectors = F.normalize(query_vectors, p=2, dim=1).to(doc_embeddings.device)

        for i, query in enumerate(all_queries):
            scores = torch.matmul(query_vectors[i:i+1], doc_embeddings.T).squeeze(0)
            doc_scores = {}
            for idx, did in enumerate(candidate_doc_ids):
                doc_scores[did] = float(scores[idx].item())
            all_results[query] = doc_scores

        fused = reciprocal_rank_fusion(all_results, k=self.rrf_k)
        return fused

    def run(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("🚀 Starting RAG-Fusion Evaluation on FollowIR")
        logger.info(f"   Official code: https://github.com/Raudaschl/rag-fusion")
        logger.info(f"   n_generated_queries={self.rf_n_queries}, diverse={self.rf_diverse}")
        logger.info(f"   RRF k={self.rrf_k}, temperature={self.rf_temperature}")
        logger.info("=" * 60)

        start_time = time.time()

        corpus, q_og, q_changed, candidates = self.data_loader.load()
        q_raw_og, q_raw_changed = self.data_loader.load_raw_queries()

        rf_cache = self._load_rf_cache()
        queries_info = {}

        query_ids_og: List[str] = []
        query_ids_changed: List[str] = []
        queries_to_generate: List[Tuple[str, str, str]] = []

        for qid in q_og.keys():
            query_ids_og.append(qid)
            raw = q_raw_og.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            queries_info[qid] = {"query": query_text, "instruction": instruction}
            if qid not in rf_cache:
                queries_to_generate.append((qid, query_text, instruction))

        for qid in q_changed.keys():
            query_ids_changed.append(qid)
            raw = q_raw_changed.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            queries_info[qid] = {"query": query_text, "instruction": instruction}
            if qid not in rf_cache:
                queries_to_generate.append((qid, query_text, instruction))

        if queries_to_generate:
            logger.info(f"🔄 Phase 1: Generating multi-queries for {len(queries_to_generate)} queries...")
            promptor = RAGFusionPromptor(diverse=self.rf_diverse, n_queries=self.rf_n_queries)
            generator = RAGFusionGenerator(
                model_path=self.llm_model_path,
                device=f"cuda:{self.gpu_id}",
                max_tokens=self.rf_max_tokens,
                temperature=self.rf_temperature,
            )

            for qid, query_text, instruction in tqdm(queries_to_generate, desc="RAG-Fusion generation"):
                messages = promptor.build_prompt(query_text, instruction)
                gen_queries = generator.generate(messages)
                gen_queries = gen_queries[:self.rf_n_queries]
                while len(gen_queries) < self.rf_n_queries:
                    gen_queries.append(query_text)
                rf_cache[qid] = gen_queries

            self._save_rf_cache(rf_cache, queries_info)
            del generator
            torch.cuda.empty_cache()
            logger.info("✅ Multi-queries generated and cached, generator unloaded")
        else:
            logger.info(f"✅ All multi-queries already cached ({len(rf_cache)} entries)")

        logger.info("🔄 Phase 2: Loading encoder and computing RAG-Fusion retrieval...")
        self._init_encoder()

        all_doc_ids = self._get_all_candidate_doc_ids(candidates)
        self.doc_ids = all_doc_ids

        doc_texts = [corpus[did]["text"] for did in all_doc_ids]
        doc_embeddings = self._encode_documents(all_doc_ids, doc_texts)
        doc_embeddings = doc_embeddings.to(f"cuda:{self.gpu_id}")

        logger.info("📊 Computing RAG-Fusion retrieval for OG queries...")
        results_og = {}
        for qid in tqdm(query_ids_og, desc="RAG-Fusion OG retrieval"):
            combined_query = q_og[qid]
            gen_queries = rf_cache[qid]
            base_qid = qid.replace("-og", "").replace("-changed", "")
            cand_ids = candidates.get(base_qid, [])
            if not cand_ids:
                continue

            doc_id_to_idx = {did: i for i, did in enumerate(self.doc_ids)}
            cand_indices = [doc_id_to_idx[did] for did in cand_ids if did in doc_id_to_idx]
            cand_doc_ids = [self.doc_ids[i] for i in cand_indices]
            cand_doc_embs = doc_embeddings[cand_indices]

            fused_scores = self._rrf_retrieve(combined_query, gen_queries, cand_doc_embs, cand_doc_ids)
            results_og[qid] = fused_scores

        logger.info("📊 Computing RAG-Fusion retrieval for Changed queries...")
        results_changed = {}
        for qid in tqdm(query_ids_changed, desc="RAG-Fusion Changed retrieval"):
            combined_query = q_changed[qid]
            gen_queries = rf_cache[qid]
            base_qid = qid.replace("-og", "").replace("-changed", "")
            cand_ids = candidates.get(base_qid, [])
            if not cand_ids:
                continue

            doc_id_to_idx = {did: i for i, did in enumerate(self.doc_ids)}
            cand_indices = [doc_id_to_idx[did] for did in cand_ids if did in doc_id_to_idx]
            cand_doc_ids = [self.doc_ids[i] for i in cand_indices]
            cand_doc_embs = doc_embeddings[cand_indices]

            fused_scores = self._rrf_retrieve(combined_query, gen_queries, cand_doc_embs, cand_doc_ids)
            results_changed[qid] = fused_scores

        logger.info("📊 Computing FollowIR metrics...")
        evaluator = FollowIREvaluator(self.task_name)
        metrics = evaluator.evaluate(results_og, results_changed)

        elapsed = time.time() - start_time

        p_mrr = metrics.get("p-MRR", 0.0)
        og_map = metrics.get("original", {}).get("map_at_1000", 0.0)
        changed_map = metrics.get("changed", {}).get("map_at_1000", 0.0)
        og_ndcg5 = metrics.get("original", {}).get("ndcg_at_5", 0.0)
        changed_ndcg5 = metrics.get("changed", {}).get("ndcg_at_5", 0.0)

        logger.info("=" * 60)
        logger.info("📊 RAG-Fusion Evaluation Results (Official Reproduction)")
        logger.info(f"   Task: {self.task_name}")
        logger.info(f"   Encoder: {self.model_name}")
        logger.info(f"   RAG-Fusion LLM: {os.path.basename(self.llm_model_path)}")
        logger.info(f"   n_generated_queries: {self.rf_n_queries}")
        logger.info(f"   RRF k: {self.rrf_k}")
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
            rf_cache=rf_cache,
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
        rf_cache: Dict[str, List[str]],
        query_ids_og: List[str],
        query_ids_changed: List[str],
        elapsed: float,
    ):
        os.makedirs(self.output_dir, exist_ok=True)

        summary = {
            "task": self.task_name,
            "model": self.model_name,
            "mode": "RAG-Fusion (official reproduction from Raudaschl/rag-fusion)",
            "rf_llm": self.llm_model_path,
            "rf_n_queries": self.rf_n_queries,
            "rf_diverse": self.rf_diverse,
            "rf_temperature": self.rf_temperature,
            "rrf_k": self.rrf_k,
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

        samples_path = os.path.join(self.output_dir, "ragfusion_samples.json")
        samples = {}
        for qid in query_ids_og[:5]:
            samples[qid] = {
                "type": "og",
                "generated_queries": rf_cache.get(qid, []),
            }
        for qid in query_ids_changed[:5]:
            samples[qid] = {
                "type": "changed",
                "generated_queries": rf_cache.get(qid, []),
            }
        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="RAG-Fusion Evaluation Engine for FollowIR (Official Reproduction)")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval",
                        choices=["Core17InstructionRetrieval", "Robust04InstructionRetrieval",
                                 "News21InstructionRetrieval"])
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--llm_model_path", type=str, default="/home/luwa/Documents/models/Qwen3-4B")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--rf_n_queries", type=int, default=4,
                        help="Number of generated queries per input (official default: 4)")
    parser.add_argument("--rf_diverse", action="store_true",
                        help="Use diverse prompt mode (official default: False)")
    parser.add_argument("--rf_temperature", type=float, default=0.7,
                        help="Generation temperature (official default: 0.7)")
    parser.add_argument("--rf_max_tokens", type=int, default=256,
                        help="Max tokens for query generation")
    parser.add_argument("--rrf_k", type=int, default=60,
                        help="RRF constant k (official default: 60)")
    parser.add_argument("--cache_dir", type=str, default="dataset/FollowIR_test/embeddings")
    parser.add_argument("--rf_cache_dir", type=str, default="dataset/FollowIR_test/ragfusion_queries")
    parser.add_argument("--use_cache", type=str, default="true")

    args = parser.parse_args()

    if args.output_dir is None:
        short_name = args.task_name.replace("InstructionRetrieval", "")
        args.output_dir = f"results/ragfusion/{short_name}"

    use_cache = args.use_cache.lower() in ("true", "1", "yes")

    evaluator = RAGFusionEvaluator(
        model_name=args.model_name,
        task_name=args.task_name,
        output_dir=args.output_dir,
        llm_model_path=args.llm_model_path,
        device=args.device,
        batch_size=args.batch_size,
        rf_n_queries=args.rf_n_queries,
        rf_diverse=args.rf_diverse,
        rf_temperature=args.rf_temperature,
        rf_max_tokens=args.rf_max_tokens,
        rrf_k=args.rrf_k,
        cache_dir=args.cache_dir,
        rf_cache_dir=args.rf_cache_dir,
        use_cache=use_cache,
        gpu_id=args.gpu_id,
    )

    evaluator.run()


if __name__ == "__main__":
    main()
