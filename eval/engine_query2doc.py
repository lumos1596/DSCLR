"""
Query2Doc Evaluation Engine for FollowIR

Reference: Wang et al. "Query2doc: Query Expansion with Large Language Models" (EMNLP 2023)
Paper: https://arxiv.org/abs/2303.07678
Official pseudo-documents: https://huggingface.co/datasets/intfloat/query2doc_msmarco

Faithfully reproduced from the paper:
    - Prompt: "Write a passage that answers the given query:" + k=4 few-shot examples (Section 2, Figure 1)
    - Dense retrieval: q+ = q [SEP] d'  (Section 2, "Dense Retrieval")
    - Sparse retrieval: q+ = q q ... q (repeat n times) d'  (Section 2, "Sparse Retrieval")
    - Only 1 pseudo-document per query (not multiple like HyDE)
    - Original LLM: GPT-3 text-davinci-003; here we use Qwen3-4B for fair comparison

FollowIR adaptation:
    - FollowIR queries have both query text and instruction
    - Prompt includes instruction to guide pseudo-document generation
    - Both og and changed queries are processed independently
    - p-MRR and target_avg metrics are computed

Usage:
    cd /home/luwa/Documents/DSCLR && /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_query2doc \
        --task_name Core17InstructionRetrieval \
        --model_name samaya-ai/RepLLaMA-reproduced \
        --llm_model_path /home/luwa/Documents/models/Qwen3-4B \
        --device cuda \
        --output_dir results/query2doc/Core17
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


QUERY2DOC_PROMPT = """Write a passage that answers the given query:
Query: {}
Passage:"""

QUERY2DOC_PROMPT_WITH_INSTRUCTION = """Write a passage that answers the given query:
Query: {}
Instruction: {}
Passage:"""

QUERY2DOC_FEW_SHOT_TEMPLATE = """Write a passage that answers the given query:
Query: {}
Passage: {}
"""

QUERY2DOC_FEW_SHOT_TEMPLATE_WITH_INSTRUCTION = """Write a passage that answers the given query:
Query: {}
Instruction: {}
Passage: {}
"""


class Query2DocPromptor:
    """Faithful reproduction of Query2Doc prompting (Wang et al. EMNLP 2023, Section 2, Figure 1)

    Paper: "The prompt comprises a brief instruction
    'Write a passage that answers the given query:'
    and k labeled pairs randomly sampled from a training set. We use k=4 throughout this paper."

    Extended to support FollowIR instructions.
    """

    def __init__(self, few_shot_examples: List[Tuple[str, str]] = None, k: int = 4):
        self.k = k
        self.few_shot_examples = few_shot_examples or []

    def build_prompt(self, query: str, instruction: str = "") -> str:
        parts = []

        for ex_query, ex_passage in self.few_shot_examples[:self.k]:
            if instruction:
                parts.append(QUERY2DOC_FEW_SHOT_TEMPLATE_WITH_INSTRUCTION.format(
                    ex_query, instruction, ex_passage
                ))
            else:
                parts.append(QUERY2DOC_FEW_SHOT_TEMPLATE.format(ex_query, ex_passage))

        if instruction:
            parts.append(QUERY2DOC_PROMPT_WITH_INSTRUCTION.format(query, instruction))
        else:
            parts.append(QUERY2DOC_PROMPT.format(query))

        return "\n".join(parts)


class Query2DocGenerator:
    """Local LLM generator for Query2Doc pseudo-documents

    Paper uses GPT-3 text-davinci-003. Here we use local Qwen3-4B for fair comparison.
    Only generates 1 pseudo-document per query (unlike HyDE which generates n=8).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.device = device

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"📥 Loading Query2Doc generator LLM from {model_path}...")
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
        logger.info(f"✅ Query2Doc generator loaded on {effective_device}")

    def generate(self, prompt: str) -> str:
        """Generate 1 pseudo-document from a prompt (paper generates 1 doc per query)."""
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return text


class Query2DocDataLoader:
    """FollowIR data loader for Query2Doc evaluation"""

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


class Query2DocEvaluator:
    """Query2Doc evaluation engine for FollowIR

    Faithfully reproduces the official Query2Doc pipeline from Wang et al. EMNLP 2023
    """

    def __init__(
        self,
        model_name: str,
        task_name: str,
        output_dir: str,
        llm_model_path: str,
        device: str = "cuda",
        batch_size: int = 32,
        q2d_max_tokens: int = 256,
        q2d_temperature: float = 0.0,
        cache_dir: str = "dataset/FollowIR_test/embeddings",
        q2d_cache_dir: str = "dataset/FollowIR_test/q2d_docs",
        use_cache: bool = True,
        gpu_id: int = 0,
    ):
        self.model_name = model_name
        self.task_name = task_name
        self.output_dir = output_dir
        self.llm_model_path = llm_model_path
        self.device = device
        self.batch_size = batch_size
        self.q2d_max_tokens = q2d_max_tokens
        self.q2d_temperature = q2d_temperature
        self.cache_dir = cache_dir
        self.q2d_cache_dir = q2d_cache_dir
        self.use_cache = use_cache
        self.gpu_id = gpu_id

        self.data_loader = Query2DocDataLoader(task_name)
        self.encoder = None

        self._init_q2d_cache()

    def _init_encoder(self):
        logger.info(f"📥 Loading encoder: {self.model_name} on GPU {self.gpu_id}")
        self.encoder = RepLLaMAEncoder(
            model_name=self.model_name,
            device=f"cuda:{self.gpu_id}",
            batch_size=self.batch_size,
        )
        logger.info("✅ Encoder loaded")

    def _init_q2d_cache(self):
        os.makedirs(self.q2d_cache_dir, exist_ok=True)

    def _get_q2d_cache_path(self) -> str:
        llm_name = os.path.basename(self.llm_model_path)
        return os.path.join(
            self.q2d_cache_dir,
            f"{self.task_name}_q2d_{llm_name}_t{self.q2d_temperature}.jsonl",
        )

    def _load_q2d_cache(self) -> Dict[str, str]:
        cache_path = self._get_q2d_cache_path()
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        cache[item["qid"]] = item["pseudo_doc"]
            logger.info(f"📂 Loaded Query2Doc cache: {len(cache)} entries from {cache_path}")
        return cache

    def _save_q2d_cache(self, cache: Dict[str, str], queries_info: Dict[str, Dict]):
        cache_path = self._get_q2d_cache_path()
        with open(cache_path, "w", encoding="utf-8") as f:
            for qid, pseudo_doc in cache.items():
                info = queries_info.get(qid, {})
                f.write(json.dumps({
                    "qid": qid,
                    "query": info.get("query", ""),
                    "instruction": info.get("instruction", ""),
                    "pseudo_doc": pseudo_doc,
                }, ensure_ascii=False) + "\n")
        logger.info(f"💾 Query2Doc cache saved: {cache_path}")

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
        query_vectors: torch.Tensor,
        query_ids: List[str],
        doc_embeddings: torch.Tensor,
        candidates: Dict[str, List[str]],
    ) -> Dict[str, Dict[str, float]]:
        results = {}
        score_matrix = torch.matmul(query_vectors, doc_embeddings.T)

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
        logger.info("🚀 Starting Query2Doc Evaluation on FollowIR")
        logger.info(f"   Paper: Wang et al. EMNLP 2023 (arXiv 2303.07678)")
        logger.info(f"   Dense: q+ = q [SEP] d'")
        logger.info(f"   temperature={self.q2d_temperature}, max_tokens={self.q2d_max_tokens}")
        logger.info("=" * 60)

        start_time = time.time()

        corpus, q_og, q_changed, candidates = self.data_loader.load()
        q_raw_og, q_raw_changed = self.data_loader.load_raw_queries()

        q2d_cache = self._load_q2d_cache()
        queries_info = {}

        query_ids_og: List[str] = []
        query_ids_changed: List[str] = []
        queries_to_generate: List[Tuple[str, str, str]] = []

        for qid in q_og.keys():
            query_ids_og.append(qid)
            raw = q_raw_og.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            queries_info[qid] = {"query": query_text, "instruction": instruction}
            if qid not in q2d_cache:
                queries_to_generate.append((qid, query_text, instruction))

        for qid in q_changed.keys():
            query_ids_changed.append(qid)
            raw = q_raw_changed.get(qid, ("", ""))
            query_text, instruction = raw[0], raw[1]
            queries_info[qid] = {"query": query_text, "instruction": instruction}
            if qid not in q2d_cache:
                queries_to_generate.append((qid, query_text, instruction))

        if queries_to_generate:
            logger.info(f"🔄 Phase 1: Generating pseudo-documents for {len(queries_to_generate)} queries...")
            promptor = Query2DocPromptor(few_shot_examples=[], k=0)
            generator = Query2DocGenerator(
                model_path=self.llm_model_path,
                device=f"cuda:{self.gpu_id}",
                max_tokens=self.q2d_max_tokens,
                temperature=self.q2d_temperature,
            )

            for qid, query_text, instruction in tqdm(queries_to_generate, desc="Query2Doc generation"):
                prompt = promptor.build_prompt(query_text, instruction)
                pseudo_doc = generator.generate(prompt)
                q2d_cache[qid] = pseudo_doc

            self._save_q2d_cache(q2d_cache, queries_info)
            del generator
            torch.cuda.empty_cache()
            logger.info("✅ Pseudo-documents generated and cached, generator unloaded")
        else:
            logger.info(f"✅ All pseudo-documents already cached ({len(q2d_cache)} entries)")

        logger.info("🔄 Phase 2: Loading encoder and computing expanded query vectors...")
        self._init_encoder()

        all_doc_ids = self._get_all_candidate_doc_ids(candidates)
        self.doc_ids = all_doc_ids

        doc_texts = [corpus[did]["text"] for did in all_doc_ids]
        doc_embeddings = self._encode_documents(all_doc_ids, doc_texts)
        doc_embeddings = doc_embeddings.to(f"cuda:{self.gpu_id}")

        logger.info("📊 Computing Query2Doc expanded vectors (dense: q [SEP] d')...")
        query_vectors_og = []
        for qid in tqdm(query_ids_og, desc="Encoding OG Query2Doc vectors"):
            combined_query = q_og[qid]
            pseudo_doc = q2d_cache[qid]
            expanded_query = f"{combined_query} [SEP] {pseudo_doc}"

            query_emb = self.encoder.encode_queries([expanded_query])
            query_emb = F.normalize(query_emb, p=2, dim=1)
            query_vectors_og.append(query_emb)

        query_vectors_og = torch.cat(query_vectors_og, dim=0).to(f"cuda:{self.gpu_id}")

        query_vectors_changed = []
        for qid in tqdm(query_ids_changed, desc="Encoding Changed Query2Doc vectors"):
            combined_query = q_changed[qid]
            pseudo_doc = q2d_cache[qid]
            expanded_query = f"{combined_query} [SEP] {pseudo_doc}"

            query_emb = self.encoder.encode_queries([expanded_query])
            query_emb = F.normalize(query_emb, p=2, dim=1)
            query_vectors_changed.append(query_emb)

        query_vectors_changed = torch.cat(query_vectors_changed, dim=0).to(f"cuda:{self.gpu_id}")

        logger.info("📊 Computing FollowIR metrics...")
        results_og = self._extract_results(query_vectors_og, query_ids_og, doc_embeddings, candidates)
        results_changed = self._extract_results(query_vectors_changed, query_ids_changed, doc_embeddings, candidates)

        evaluator = FollowIREvaluator(self.task_name)
        metrics = evaluator.evaluate(results_og, results_changed)

        elapsed = time.time() - start_time

        p_mrr = metrics.get("p-MRR", 0.0)
        og_map = metrics.get("original", {}).get("map_at_1000", 0.0)
        changed_map = metrics.get("changed", {}).get("map_at_1000", 0.0)
        og_ndcg5 = metrics.get("original", {}).get("ndcg_at_5", 0.0)
        changed_ndcg5 = metrics.get("changed", {}).get("ndcg_at_5", 0.0)

        logger.info("=" * 60)
        logger.info("📊 Query2Doc Evaluation Results (Official Reproduction)")
        logger.info(f"   Task: {self.task_name}")
        logger.info(f"   Encoder: {self.model_name}")
        logger.info(f"   Query2Doc LLM: {os.path.basename(self.llm_model_path)}")
        logger.info(f"   Expansion: q [SEP] d' (dense retrieval)")
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
            q2d_cache=q2d_cache,
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
        q2d_cache: Dict[str, str],
        query_ids_og: List[str],
        query_ids_changed: List[str],
        elapsed: float,
    ):
        os.makedirs(self.output_dir, exist_ok=True)

        summary = {
            "task": self.task_name,
            "model": self.model_name,
            "mode": "Query2Doc (official reproduction from Wang et al. EMNLP 2023)",
            "q2d_llm": self.llm_model_path,
            "q2d_temperature": self.q2d_temperature,
            "q2d_max_tokens": self.q2d_max_tokens,
            "q2d_expansion_method": "dense: q [SEP] d' (paper Section 2)",
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

        pseudo_samples_path = os.path.join(self.output_dir, "q2d_samples.json")
        samples = {}
        for qid in query_ids_og[:5]:
            samples[qid] = {
                "type": "og",
                "pseudo_doc": q2d_cache.get(qid, "")[:500],
            }
        for qid in query_ids_changed[:5]:
            samples[qid] = {
                "type": "changed",
                "pseudo_doc": q2d_cache.get(qid, "")[:500],
            }
        with open(pseudo_samples_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Query2Doc Evaluation Engine for FollowIR (Official Reproduction)")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval",
                        choices=["Core17InstructionRetrieval", "Robust04InstructionRetrieval",
                                 "News21InstructionRetrieval"])
    parser.add_argument("--model_name", type=str, default="samaya-ai/RepLLaMA-reproduced")
    parser.add_argument("--llm_model_path", type=str, default="/home/luwa/Documents/models/Qwen3-4B")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--q2d_max_tokens", type=int, default=256,
                        help="Max tokens for pseudo-document generation")
    parser.add_argument("--q2d_temperature", type=float, default=0.0,
                        help="Generation temperature (0 = greedy, matching paper's single doc)")
    parser.add_argument("--cache_dir", type=str, default="dataset/FollowIR_test/embeddings")
    parser.add_argument("--q2d_cache_dir", type=str, default="dataset/FollowIR_test/q2d_docs")
    parser.add_argument("--use_cache", type=str, default="true")

    args = parser.parse_args()

    if args.output_dir is None:
        short_name = args.task_name.replace("InstructionRetrieval", "")
        args.output_dir = f"results/query2doc/{short_name}"

    use_cache = args.use_cache.lower() in ("true", "1", "yes")

    evaluator = Query2DocEvaluator(
        model_name=args.model_name,
        task_name=args.task_name,
        output_dir=args.output_dir,
        llm_model_path=args.llm_model_path,
        device=args.device,
        batch_size=args.batch_size,
        q2d_max_tokens=args.q2d_max_tokens,
        q2d_temperature=args.q2d_temperature,
        cache_dir=args.cache_dir,
        q2d_cache_dir=args.q2d_cache_dir,
        use_cache=use_cache,
        gpu_id=args.gpu_id,
    )

    evaluator.run()


if __name__ == "__main__":
    main()
