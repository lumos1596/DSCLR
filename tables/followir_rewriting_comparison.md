# Query Rewriting Methods on FollowIR Benchmark

## Evaluation Setup

- **Benchmark**: FollowIR (Core17, Robust04, News21)
- **Metrics**: p-MRR (instruction sensitivity), MAP@1000 (Core17/Robust04), nDCG@5 (News21)
- **target_avg** = (Core17_changed_MAP@1000 + Robust04_changed_MAP@1000 + News21_changed_nDCG@5) / 3
- All rewriting methods generate rewritten queries for changed queries only; OG queries use original text

## Results

| Method | Encoder | Core17 p-MRR | Core17 MAP | Robust04 p-MRR | Robust04 MAP | News21 p-MRR | News21 nDCG@5 | Avg p-MRR | target_avg |
|--------|---------|-------------|-----------|----------------|-------------|-------------|--------------|-----------|------------|
| HyDE (2023) | RepLLaMA | 0.0651 | 0.2364 | -0.0292 | 0.2487 | 0.0071 | 0.2137 | 0.0143 | 0.2343 |
| Query2Doc (2023) | RepLLaMA | 0.0798 | 0.2588 | -0.0749 | 0.2792 | -0.0375 | 0.2489 | -0.0109 | 0.2622 |
| RAG-QR (2023) | RepLLaMA | 0.0228 | 0.2228 | -0.1029 | 0.2763 | -0.0214 | 0.2290 | -0.0338 | 0.2427 |
| RAG-Fusion (2023) | RepLLaMA | 0.0540 | 0.2187 | -0.0810 | 0.2109 | 0.0180 | 0.2105 | -0.0030 | 0.2134 |
| DeepRetrieval (2025) | RepLLaMA | 0.0663 | 0.2149 | -0.0532 | 0.2594 | -0.0218 | 0.2394 | -0.0029 | 0.2379 |
| ConvSearch-R1 (2025) | RepLLaMA | 0.0304 | 0.2234 | -0.0251 | 0.2589 | 0.0018 | 0.2201 | 0.0024 | 0.2341 |
| TongSearch-QR (7B) (2025) | RepLLaMA | 0.0328 | 0.2166 | -0.0456 | 0.2453 | 0.0019 | 0.2178 | -0.0036 | 0.2266 |
| TongSearch-QR (3B) (2025) | RepLLaMA | -0.0310 | 0.2206 | -0.0559 | 0.2712 | -0.0186 | 0.2239 | -0.0352 | 0.2386 |
| Granite aLoRA QR (2025) | BGE-large-en | -0.0376 | 0.1878 | -0.0827 | 0.1781 | 0.0511 | 0.2002 | -0.0231 | 0.1887 |
| INF-X-Retriever (Full) (2025) | INF-Retriever | 0.0718 | 0.2703 | -0.0141 | 0.3206 | 0.0439 | 0.2234 | 0.0339 | 0.2704 |
| INF-X Aligner (2025) | RepLLaMA | 0.0748 | 0.2291 | -0.0236 | 0.2420 | -0.0073 | 0.2238 | 0.0146 | 0.2274 |
| BGE-Reasoner-Rewriter (2026) | BGE-large-en | 0.0071 | 0.2153 | -0.0762 | 0.2101 | 0.0336 | 0.1977 | -0.0118 | 0.2077 |
| mTRAG Rewriter (2026) | BGE-large-en | -0.0040 | 0.1725 | -0.0996 | 0.1938 | 0.0814 | 0.2101 | -0.0074 | 0.1921 |
| **DeIR-Dual V2 (Ours)** | **RepLLaMA** | **0.1162** | **0.2597** | **0.0826** | **0.2657** | **0.1871** | **0.3229** | **0.1286** | **0.2828** |

## References

| Method | Paper | Venue | Year |
|--------|-------|-------|------|
| HyDE | Precise Zero-Shot Dense Retrieval without Relevance Labels (Luyu Gao et al.) | ACL | 2023 |
| Query2Doc | Query2doc: Query Expansion with Large Language Models (Liang Wang, Nan Yang, Furu Wei) | EMNLP | 2023 |
| RAG-QR | Query Rewriting for Retrieval-Augmented Large Language Models (Xinbei Ma et al.) | EMNLP | 2023 |
| RAG-Fusion | RAG-Fusion: The Next Frontier of Search Technology (Adrian H. Raudaschl) | Blog | 2023 |
| DeepRetrieval | DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning (Pengcheng Jiang et al.) | COLM | 2025 |
| ConvSearch-R1 | ConvSearch-R1: Enhancing Query Reformulation for Conversational Search with Reasoning via Reinforcement Learning (Changtai Zhu et al.) | EMNLP | 2025 |
| TongSearch-QR | Reinforced Query Reasoners for Reasoning-intensive Retrieval Tasks (Xubo Qin et al.) | EMNLP | 2025 |
| Granite aLoRA QR | Activated LoRA: Fine-tuned LLMs for Intrinsics (Kristjan Greenewald et al.) | NeurIPS | 2025 |
| INF-X-Retriever | INF-X-Retriever: A Pragmatic Framework for Reasoning-Intensive Retrieval | arXiv | 2025 |
| BGE-Reasoner-Rewriter | ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval (Jianlyu Chen et al.) | ACL | 2026 |
| mTRAG Rewriter | Caraman at SemEval-2026 Task 8: Three-Stage Multi-Turn Retrieval with Query Rewriting, Hybrid Search, and Cross-Encoder Reranking | SemEval | 2026 |

## Method Categories

| Category | Methods | Avg p-MRR Range | Key Pattern |
|----------|---------|-----------------|-------------|
| Instruction-aware rewriting | DeIR-Dual V2 | 0.1286 | Reward-penalty dual-track: Q_plus enhancement + Q_minus penalty |
| End-to-end instruction-tuned | Promptriever (p-MRR=0.1001) | ~0.10 | Instruction-finetuned retriever |
| RL-tuned rewriting | DeepRetrieval, ConvSearch-R1 | -0.003 ~ 0.002 | RL training for recall, not instruction sensitivity |
| Pseudo-document expansion | HyDE, Query2Doc | -0.011 ~ 0.014 | Semantic enhancement dilutes instruction signal |
| Multi-query fusion | RAG-Fusion, BGE-Reasoner-Rewriter | -0.012 ~ 0.034 | Score aggregation/RRF averages out instruction signal |
| Conversational decontextualization | mTRAG, TongSearch-QR, Granite aLoRA QR | -0.035 ~ -0.004 | Rewriting removes context, not responds to instructions |
