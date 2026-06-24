---
name: "dsclr-exp-results"
description: "DeIR-Dual V2 实验结果速查。包含所有基准数据集上的评测结果、参数对比、跨模型/跨模态对比。Invoke when writing paper, comparing results, or needing specific experiment numbers."
---

# DeIR-Dual V2 实验结果速查

## 快速索引

| 基准 | 编码器 | 核心发现 | 详见 |
|------|--------|---------|------|
| FollowIR | RepLLaMA + 4B | p-MRR=0.1687, target_avg=0.2841 (V5 训练集推导, δ=0.02) | [链接](#followir-评测结果) |
| FollowIR | RepLLaMA + 8B | target_avg=0.2857, p-MRR=0.1365 (改进两阶段法) | [链接](#repllama--qwen3-8b-改写模型) |
| FollowIR | Promptriever (LLaMA 3.1 8B) | p-MRR=0.1001, target_avg=0.3063 (基线模型) | [链接](#promptriever-llama-31-8b-instruct-基线) |
| FollowIR | Mistral (E5) | target_avg=0.2742, p-MRR=0.0540 (高 at-risk 编码器) | [链接](#mistral-e5-mistral-7b-编码器) |
| FollowIR | ConvSearch-R1 (Qwen2.5-3B) | p-MRR=0.0024, target_avg=0.2341 (RL 对话改写，指令敏感度近零) | [链接](#指令改写方法对比) |
| FollowIR | BGE-Reasoner-Rewriter + BGE | p-MRR=0.0204, target_avg=0.2103 (ReasonEmbed 改写器，5查询分数聚合)(ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieva) | [链接](#bge-reasoner-rewriter--bge-reasonembed) |
| FollowIR | INF-X-Retriever (Full) | p-MRR=0.0339, target_avg=0.2704 (Aligner+Retriever 全流程) | [链接](#inf-x-retriever-全流程) |
| FollowIR | INF-X Aligner + RepLLaMA | p-MRR=0.0146, target_avg=0.2274 (仅改写器) | [链接](#inf-x-aligner--repllama) |
| FollowIR | mTRAG Rewriter + BGE | p-MRR=-0.0074, target_avg=0.1921 (多轮对话改写器，指令敏感度近零) | [链接](#mtrag-query-rewriter--bge) |
| FollowIR | TongSearch-QR (3B) + RepLLaMA | p-MRR=-0.0352, target_avg=0.2386 (对话改写器，指令敏感度负) | [链接](#tongsearch-qr--repllama) |
| FollowIR | TongSearch-QR (7B) + RepLLaMA | p-MRR=-0.0036, target_avg=0.2266 (对话改写器，指令敏感度近零) | [链接](#tongsearch-qr--repllama) |
| FollowIR | Granite aLoRA QR + BGE | p-MRR=-0.0231, target_avg=0.1887 (IBM aLoRA 对话改写器，指令敏感度负)(mt RAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems) | [链接](#granite-alora-qr--bge) |
| InstructIR | RepLLaMA + 4B | 无显著提升（baseline 已饱和） | [链接](#instructir-评测结果) |
| NegConstraint | RepLLaMA + 4B | nDCG@10 +4.3%, MAP@100 +5.7% | [链接](#negconstraint-评测结果) |
| NegConstraint | BGE + 4B | nDCG@10 +4.1%, MAP@100 +5.8% | [链接](#negconstraint-评测结果) |
| COCO-Neg | 4×CLIP | R@5 +3.6%~+7.1%, 恢复率 84%~94% | [链接](#coco-neg-评测结果) |
| ComLQ | BGE + 4B | 惩罚无效（语义纠缠），仅 Q_plus 提升 +0.6% | [链接](#comlq-评测结果) |
| BEIR/NQ | RepLLaMA + 4B | Q_plus-only 模式 nDCG@10 +1.1% | [链接](#beirnq-评测结果) |

---

## FollowIR 评测结果

### 基准信息

**论文**：FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions  
**作者**：Orion Weller et al. (Johns Hopkins University, Allen Institute for AI, University of Glasgow, Yale University)  
**发表**：arXiv:2403.15246, 2024年5月  
**会议**：ACL 2024  
**链接**：https://arxiv.org/abs/2403.15246

### 评测指标定义

**target_avg = (Core17_changed_MAP@1000 + Robust04_changed_MAP@1000 + News21_changed_nDCG@5) / 3**

pMRR: 衡量指令敏感度

### RepLLaMA + Qwen3-4B 改写模型

#### 参数策略对比

| 策略 | α | β | δ | target_avg | mean p-MRR | 学术规范 | 理论依据 |
|------|---|---|---|-----------|-----------|---------|---------|
| 网格搜索（测试集） | 0.5 | 1.0 | 0.0 | 0.281 | 0.1381 | ❌ | 无（暴力搜索） |
| 改进两阶段法（训练集） | 1.0 | 1.5 | 0.05 | **0.2828** | 0.1286 | ✅ | 训练集统计+奖惩等权 |
| 第一性原理 V1 | 0.67 | 1.23 | 0.05 | 0.2812 | 0.1039 | ✅ | 向量空间几何+噪声边际 |
| 第一性原理 V2 (NP+KS) | 0.5 | 1.0 | 0.0 | 0.278 | 0.1943 | ✅ | NP 阈值+KS 最大化 |
| 第一性原理 V4 (测试集推导) | 1.0 | 1.29 | 0.0 | 0.2631 | 0.2243 | ❌ | 30 种方法一致性验证 |
| 第一性原理 V5 (训练集推导, δ=0) | 0.72 | 1.46 | 0.0 | 0.2672 | 0.2152 | ✅ | 训练集量级对齐 |
| **第一性原理 V5 (训练集推导, δ=0.02)** | **0.72** | **1.32** | **0.02** | **0.2841** | **0.1687** | **✅** | **训练集量级对齐+噪声边际** |

#### V5 训练集推导 — 逐数据集结果

**δ=0.02（推荐，平衡方案）**：

| 数据集 | p-MRR | changed_MAP@1000 | changed_nDCG@5 |
|--------|-------|-----------------|----------------|
| Core17 | 0.1687 | 0.2551 | 0.3440 |
| Robust04 | 0.1770 | 0.2533 | 0.2920 |
| News21 | 0.2912 | 0.2790 | 0.3440 |
| **Mean** | **0.1687** | — | — |
| **target_avg** | — | **0.2841** | — |

**δ=0.0（高 p-MRR 方案）**：

| 数据集 | p-MRR | changed_MAP@1000 | changed_nDCG@5 |
|--------|-------|-----------------|----------------|
| Core17 | 0.2152 | 0.2404 | 0.3268 |
| Robust04 | 0.2152 | 0.2344 | 0.2920 |
| News21 | 0.2912 | 0.2790 | 0.3268 |
| **Mean** | **0.2152** | — | — |
| **target_avg** | — | **0.2672** | — |

#### V5 分析

- **修复 τ 计算后**：τ = Cos(Q_base, Q_neg) + δ（之前错误地使用 τ = S_neg + δ）
- **δ=0.02 方案**：target_avg=0.2841，超过网格搜索(0.281)，同时 p-MRR=0.1687 比网格搜索(0.1381)高 22.1%
- **Robust04 MAP 从 0.2257 提升到 0.2533**，提升 12.2%
- **β 从 1.926 降到 1.32**：修复后 at-risk ratio 从 0% 变为 ~5%，safe 文档减少，β 推导更准确
- **α 从 1.0 降到 0.72**：修复后 at-risk 文档的 Softplus 值更大，惩罚更有效
- **δ=0.0（NP 阈值）**：τ = Cos(Q_base, Q_neg)，动态阈值已捕获语义关系
- 推导过程：eval/first_principles_params_train.py，训练集 855 查询，878 正例，12825 负例
- 来源：results/train_derived_params.json

#### α vs p-MRR/target_avg 权衡（β=1.29, δ=0.0）

| α | mean p-MRR | target_avg | 推导方法 |
|---|-----------|-----------|---------|
| 0.5 | 0.1999 | 0.2737 | Soft Half-Life / KS |
| 1.0 | 0.2243 | 0.2631 | Scale Alignment / Percentile-50 |
| 1.5 | 0.2486 | 0.2582 | — |
| 2.0 | 0.2724 | 0.2483 | — |
| 3.0 | 0.3205 | 0.2262 | — |

#### 30 种 α 推导方法汇总（δ_k=0.0）

| 分组 | α 范围 | 代表方法 |
|------|--------|---------|
| Group A (Scale Alignment) | 1.0 | 惩罚量级对齐（最优） |
| Group B (Score Resolution) | 0.05~0.52 | 编码器分辨率 |
| Group C (Distribution Separation) | 0.04~0.22 | 分布分离 |
| Group D (Ranking-Specific) | 0.01~1.01 | 排序特异性 |
| Group E (Physics-Informed) | 0.33~0.50 | 半衰期/信息论 |
| Group F (Document-Aware) | 0.00~6.15 | 文档感知/高级统计 |

Group F 重点方法：
- Score Entropy: α=6.15（过高）
- Kurtosis/Skewness-Adjusted: α≈0.065~0.068（过低）
- KL Minimization: α=0.12（过低）
- **Percentile-50/75 Alignment: α=1.00~1.03（与 Scale Alignment 一致！）**
- Bayesian Posterior: α=752+（数据量过大导致先验被淹没）

### RepLLaMA + Qwen3-8B 改写模型

**测试集网格搜索最优**：
- α=0.5, β=1.1, δ=0.0
- target_avg=0.283, mean p-MRR=0.1472

**训练集导出最优（改进两阶段法）**：
- α=1.0, β=1.5, δ=0.05
- 测试集 target_avg=0.2857（Core17_MAP=0.2594, Robust04_MAP=0.2684, News21_nDCG=0.3292）
- 测试集 mean p-MRR=0.1365（Core17=0.1209, Robust04=0.0987, News21=0.1899）

### Promptriever (LLaMA 3.1 8B Instruct) 基线

**模型信息**：
- 论文：Promptriever: Instruction-Trained LLMs are Retrievers
- 作者：Samaya AI
- 发布：2024
- HuggingFace：samaya-ai/promptriever-llama3.1-8b-instruct-v1
- 基础模型：meta-llama/Meta-Llama-3.1-8B-Instruct + LoRA adapter

**评测结果**：

| 数据集 | p-MRR | OG MAP@1000 | OG nDCG@5 | Changed MAP@1000 | Changed nDCG@5 |
|--------|-------|-------------|-----------|------------------|----------------|
| Core17 | 0.1334 | 0.3741 | 0.5253 | 0.2827 | 0.3379 |
| Robust04 | 0.1019 | 0.3951 | 0.6102 | 0.3223 | 0.4397 |
| News21 | 0.0651 | 0.5022 | 0.5242 | 0.3005 | 0.3138 |
| **Mean** | **0.1001** | — | — | — | — |
| **target_avg** | — | — | — | **0.3063** | — |

**关键发现**：
- Promptriever 作为基线模型，p-MRR=0.1001，低于 DeIR-Dual V2 (0.1687)
- target_avg=0.3063，略高于 DeIR-Dual V2 (0.2841)，但 p-MRR 低 40.7%
- 指令敏感度较低，适合作为基线对比

---

### 指令改写方法对比

**评测配置**：RepLLaMA 编码器 + Qwen3-4B 改写模型，FollowIR 三个测试集

#### 基准方法论文信息

| 方法 | 论文 | 作者 | 会议/期刊 | 年份 | 链接 |
|------|------|------|----------|------|------|
| **DeepRetrieval** | DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning | Pengcheng Jiang et al. (UIUC) | COLM 2025 | 2025 | [arXiv:2503.00223](https://arxiv.org/abs/2503.00223) |
| **HyDE** | Precise Zero-Shot Dense Retrieval without Relevance Labels | Luyu Gao et al. (University of Waterloo, CMU) | ACL 2023 | 2023 | [ACL Anthology](https://aclanthology.org/2023.acl-long.99/) |
| **Query2Doc** | Query2doc: Query Expansion with Large Language Models | Liang Wang, Nan Yang, Furu Wei (Microsoft Research) | EMNLP 2023 | 2023 | [ACL Anthology](https://aclanthology.org/2023.emnlp-main.585/) |
| **RAG-Fusion** | RAG-Fusion: The Next Frontier of Search Technology | Adrian H. Raudaschl | 开源项目 | 2024 | [GitHub](https://github.com/Raudaschl/rag-fusion) |
| **RAG-QR** | Query Rewriting for Retrieval-Augmented Large Language Models | Xinbei Ma et al. (SJTU, MSRA) | EMNLP 2023 | 2023 | [ACL Anthology](https://aclanthology.org/2023.emnlp-main.322/) |
| **ConvSearch-R1** | ConvSearch-R1: Enhancing Query Reformulation for Conversational Search with Reasoning via Reinforcement Learning | Zihan Wang et al. (Beijing Institute of Technology) | arXiv 2025 | 2025 | [arXiv:2505.23716](https://arxiv.org/abs/2505.23716) |

#### Core17 结果

| 方法 | p-MRR | OG MAP@1000 | Changed MAP@1000 | OG nDCG@5 | Changed nDCG@5 |
|------|-------|-------------|-----------------|-----------|----------------|
| DeIR-Dual V2 | **0.1162** | 0.2412 | **0.2597** | 0.4234 | 0.3188 |
| DeepRetrieval | 0.0663 | 0.3504 | 0.2149 | 0.4768 | 0.2424 |
| HyDE | 0.0651 | 0.3469 | 0.2364 | 0.5029 | 0.2898 |
| Query2Doc | 0.0798 | **0.3790** | 0.2588 | **0.5697** | **0.3423** |
| RAG-Fusion | 0.0540 | 0.3223 | 0.2187 | 0.4443 | 0.2563 |
| RAG-QR | 0.0228 | 0.3335 | 0.2228 | 0.4701 | 0.2300 |
| ConvSearch-R1 | 0.0304 | 0.3597 | 0.2234 | 0.5114 | 0.2787 |

#### Robust04 结果

| 方法 | p-MRR | OG MAP@1000 | Changed MAP@1000 | OG nDCG@5 | Changed nDCG@5 |
|------|-------|-------------|-----------------|-----------|----------------|
| DeIR-Dual V2 | **0.0826** | 0.2419 | **0.2657** | 0.3970 | **0.3480** |
| DeepRetrieval | -0.0532 | 0.3173 | 0.2594 | 0.4925 | 0.3170 |
| HyDE | -0.0292 | 0.3246 | 0.2487 | 0.5138 | 0.3078 |
| Query2Doc | -0.0749 | **0.3264** | 0.2792 | **0.5165** | 0.3518 |
| RAG-Fusion | -0.0810 | 0.2697 | 0.2109 | 0.4016 | 0.2335 |
| RAG-QR | -0.1029 | 0.3145 | 0.2763 | 0.4931 | 0.3370 |
| ConvSearch-R1 | -0.0251 | 0.3319 | 0.2589 | 0.5382 | 0.3285 |

#### News21 结果

| 方法 | p-MRR | OG MAP@1000 | Changed MAP@1000 | OG nDCG@5 | Changed nDCG@5 |
|------|-------|-------------|-----------------|-----------|----------------|
| DeIR-Dual V2 | **0.1871** | 0.2885 | 0.2330 | 0.4288 | **0.3229** |
| DeepRetrieval | -0.0218 | 0.4632 | 0.2347 | 0.4518 | 0.2394 |
| HyDE | 0.0071 | **0.4671** | 0.2177 | **0.4890** | 0.2137 |
| Query2Doc | -0.0375 | 0.4752 | **0.2485** | 0.5157 | 0.2489 |
| RAG-Fusion | 0.0180 | 0.4064 | 0.2097 | 0.4254 | 0.2105 |
| RAG-QR | -0.0214 | 0.4675 | 0.2408 | 0.4736 | 0.2290 |
| ConvSearch-R1 | 0.0018 | 0.4659 | 0.2221 | 0.4959 | 0.2201 |

#### 汇总对比

| 方法 | mean p-MRR | target_avg | 核心方法 |
|------|-----------|-----------|---------|
| DeIR-Dual V2 | **0.1286** | **0.2828** | 奖惩双轨：Q_plus 增强 + Q_minus 惩罚 |
| DeepRetrieval | -0.0029 | 0.2379 | RL 训练的查询生成器 + RepLLaMA 编码 |
| HyDE | 0.0143 | 0.2343 | 假想文档向量平均 |
| Query2Doc | -0.0109 | 0.2622 | q [SEP] d' 拼接扩展 |
| RAG-Fusion | -0.0030 | 0.2134 | 多查询生成 + RRF 融合 |
| RAG-QR | -0.0338 | 0.2427 | T5-large PPO 改写器 + RepLLaMA 编码 |
| ConvSearch-R1 | 0.0024 | 0.2341 | GRPO RL 训练的 Qwen2.5-3B 对话改写器 + RepLLaMA 编码 |

#### 关键发现

1. **DeIR-Dual V2 在 p-MRR 上大幅领先**：0.1286 vs HyDE 0.0143 vs DeepRetrieval -0.0029 vs RAG-Fusion -0.0030 vs Query2Doc -0.0109，分别领先 9.0×、44.3×、43.9× 和 11.8×
2. **DeepRetrieval 的 p-MRR 接近零**：mean p-MRR=-0.0029，与 RAG-Fusion (-0.0030) 几乎相同，说明 RL 训练的查询生成器虽然提升了 OG 指标，但几乎完全丧失了指令敏感度
3. **DeepRetrieval 的 OG 指标优于 DeIR-Dual V2**：Core17 OG MAP=0.3504 vs 0.2412，Robust04 OG MAP=0.3173 vs 0.2419，News21 OG MAP=0.4632 vs 0.2885，说明 DeepRetrieval 的改写查询在语义增强方面有效
4. **DeepRetrieval 的 changed 指标不如 DeIR-Dual V2**：Core17 Changed MAP=0.2149 vs 0.2597，Robust04 Changed MAP=0.2594 vs 0.2657，News21 Changed nDCG@5=0.2394 vs 0.3229
5. **DeepRetrieval 的失败模式与 HyDE/Query2Doc 相同**：改写是"语义增强"而非"指令响应"，无法区分 og/changed 查询的差异；RL 训练目标是 BM25 召回率最大化，不关注指令敏感度
6. **RAG-Fusion 在 p-MRR 和 target_avg 上均表现最差**：mean p-MRR=-0.0030（指令敏感度几乎为零），target_avg=0.2134（检索质量也最差）
7. **RAG-Fusion 的失败原因**：RRF 融合将 5 个查询（1 原始 + 4 生成）的检索结果取平均排序，多查询的语义多样性反而"稀释"了指令信号，使 og/changed 查询的检索结果趋同
8. **HyDE/Query2Doc/DeepRetrieval/RAG-Fusion 在 Robust04/News21 上 p-MRR 为负或接近零**：说明传统改写方式反而降低了指令敏感度，changed 查询的排序质量下降
9. **HyDE/Query2Doc/DeepRetrieval 的 OG 指标更高**：因为假想文档/伪文档/RL 生成查询提供了额外的语义信息，提升了原始查询的检索质量，但代价是降低了对指令变化的敏感度
10. **DeIR-Dual V2 在 target_avg 上也领先**：0.2828 vs Query2Doc 0.2622 vs DeepRetrieval 0.2379 vs HyDE 0.2343 vs RAG-Fusion 0.2134
11. **DeIR-Dual V2 的核心优势**：通过 Q_minus 惩罚机制，在保持检索质量的同时大幅提升指令敏感度；所有其他方法的改写都是"语义增强"而非"指令响应"，无法区分 og/changed 查询的差异
12. **RAG-QR 的 p-MRR 为负**：mean p-MRR=-0.0338，在 Robust04 上 p-MRR=-0.1029（所有方法中最差），说明 T5 改写器产生的查询改写与 instruction 拼接后反而降低了指令敏感度
13. **RAG-QR 的改写器不含 instruction**：T5-large 仅接收 query text（prompt="rewrite a better search query: "），instruction 仅在编码阶段拼接（rewritten_query + instruction），这种"改写+拼接"方式比直接编码原始查询（query + instruction）的指令敏感度更低
14. **RAG-QR 在 Robust04 上 p-MRR 最低**：-0.1029，比 Query2Doc (-0.0749) 和 RAG-Fusion (-0.0810) 更差，说明 T5 改写后的查询语义与 instruction 拼接后产生了更大的干扰
15. **ConvSearch-R1 的 p-MRR 接近零**：mean p-MRR=0.0024，与 DeepRetrieval (-0.0029)、RAG-Fusion (-0.0030) 处于同一水平，说明 GRPO RL 训练的对话改写器同样几乎完全丧失了指令敏感度
16. **ConvSearch-R1 的 target_avg=0.2341**：低于 HyDE (0.2343) 和 DeepRetrieval (0.2379)，在所有改写方法中排名倒数第二（仅高于 RAG-Fusion 0.2134）
17. **ConvSearch-R1 的失败模式与 DeepRetrieval 相同**：GRPO 训练目标是生成更好的对话式查询改写（decontextualization），不关注指令敏感度；改写后的查询是"去上下文化"而非"指令响应"，无法区分 og/changed 查询的差异
18. **ConvSearch-R1 在 Robust04 上 p-MRR 为负**：-0.0251，说明改写查询反而降低了指令敏感度，与 HyDE/Query2Doc/DeepRetrieval 的失败模式一致

---

### Mistral (E5-Mistral-7B) 编码器

**测试集网格搜索最优**：
- α=0.1, β=1.1, δ=0.05
- mean p-MRR=0.0319

**训练集导出最优**：
- α=0.3, β=1.0, δ=0.05
- 测试集 target_avg=0.2742, mean p-MRR=0.0540
- **注意**：Mistral at-risk 比例高达 62.9%（vs RepLLaMA 0.08%），增大 α 会显著损害 target_avg

---

### BGE-Reasoner-Rewriter + BGE (ReasonEmbed)

**模型信息**：
- 论文：ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval
- HuggingFace：cfli/reasoner-rewriter-qwen2.5-7b-0821
- 基础模型：Qwen2.5-7B
- 方法：生成 5 个改写查询，分别用 BGE-large-en 编码检索，分数求和聚合
- 编码器：BGE-large-en-v1.5（本地路径 /home/luwa/Documents/models/BGE-large-en-v1.5）
- OG 查询使用原始文本编码（不改写），仅改写 changed 查询

**评测结果**：

| 数据集 | p-MRR | OG MAP@1000 | Changed MAP@1000 | OG nDCG@5 | Changed nDCG@5 |
|--------|-------|-------------|-----------------|-----------|----------------|
| Core17 | 0.0071 | 0.3176 | 0.2153 | 0.4448 | 0.2872 |
| News21 | 0.0336 | 0.4238 | 0.2052 | 0.4506 | 0.1977 |
| Robust04 | ⏳ 运行中 | — | — | — | — |
| **Mean (2/3)** | **0.0204** | — | — | — | — |
| **target_avg (2/3)** | — | — | — | — | — |

**关键发现**：
- p-MRR 接近零（0.0204），说明 5 查询分数聚合方式几乎完全丧失指令敏感度
- 5 个改写查询的分数求和使 og/changed 查询的检索结果趋同，"稀释"了指令信号
- OG 指标较高（Core17 MAP=0.3176），说明改写在语义增强方面有效
- Changed 指标较低（Core17 MAP=0.2153），改写反而损害了指令相关文档的排序

---

### INF-X-Retriever 全流程

**模型信息**：
- 论文/项目：INF-X-Retriever
- 组件：inf-query-aligner（Qwen2.5-7B-Instruct RL-tuned 改写器）+ inf-retriever-v1-pro（通用密集检索器）
- HuggingFace：infgrad/inf-query-aligner, infgrad/inf-retriever-v1-pro
- 方法：先用 Aligner 改写查询，再用专用检索器编码检索

**评测结果**：

| 数据集 | p-MRR | OG MAP@1000 | Changed MAP@1000 | OG nDCG@5 | Changed nDCG@5 |
|--------|-------|-------------|-----------------|-----------|----------------|
| Core17 | 0.0718 | 0.3973 | 0.2703 | 0.6285 | 0.3401 |
| Robust04 | -0.0141 | 0.3803 | 0.3206 | 0.5819 | 0.3943 |
| News21 | 0.0439 | 0.4716 | 0.2203 | 0.5134 | 0.2234 |
| **Mean** | **0.0339** | — | — | — | — |
| **target_avg** | — | — | **0.2704** | — | — |

**关键发现**：
- p-MRR=0.0339，指令敏感度很低，与 DeepRetrieval/RAG-Fusion 处于同一量级
- target_avg=0.2704，检索质量中等（高于 HyDE 0.2343，低于 DeIR-Dual V2 0.2828）
- OG 指标非常高（Core17 MAP=0.3973, Robust04 MAP=0.3803），说明专用检索器编码能力强
- Changed 指标相对较低，说明改写器"语义增强"而非"指令响应"

---

### INF-X Aligner + RepLLaMA

**模型信息**：
- 仅使用 INF-X 的查询改写器（inf-query-aligner），编码器替换为 RepLLaMA
- 目的：隔离改写器的效果，排除检索器差异的影响

**评测结果**：

| 数据集 | p-MRR | OG MAP@1000 | Changed MAP@1000 | OG nDCG@5 | Changed nDCG@5 |
|--------|-------|-------------|-----------------|-----------|----------------|
| Core17 | 0.0748 | 0.3503 | 0.2291 | 0.4573 | 0.2540 |
| Robust04 | -0.0236 | 0.2832 | 0.2420 | 0.4544 | 0.3204 |
| News21 | -0.0073 | 0.4454 | 0.2111 | 0.4807 | 0.2238 |
| **Mean** | **0.0146** | — | — | — | — |
| **target_avg** | — | — | **0.2274** | — | — |

**关键发现**：
- p-MRR=0.0146，比全流程 (0.0339) 更低，说明 INF-X 检索器本身有一定指令敏感度
- target_avg=0.2274，低于全流程 (0.2704)，说明 INF-X 专用检索器编码质量优于 RepLLaMA
- 改写器本身几乎无指令敏感度贡献

---

### mTRAG Query Rewriter + BGE

**模型信息**：
- 论文/项目：mTRAG (Multi-Turn Retrieval-Augmented Generation)
- HuggingFace：caraman/Qwen2.5-7B-mtrag-query-rewriter-final
- 基础模型：Qwen2.5-7B-Instruct + LoRA 微调
- 原始用途：多轮对话查询改写（decontextualization）
- 适配方式：将 FollowIR 的 query + instruction 转为单轮改写 prompt
- 编码器：BGE-large-en-v1.5
- 生成参数：temperature=0.2, max_new_tokens=512

**评测结果**：

| 数据集 | p-MRR | OG MAP@1000 | Changed MAP@1000 | OG nDCG@5 | Changed nDCG@5 |
|--------|-------|-------------|-----------------|-----------|----------------|
| Core17 | -0.0040 | 0.2385 | 0.1725 | 0.3319 | 0.1920 |
| Robust04 | -0.0996 | 0.2203 | 0.1938 | 0.3461 | 0.2607 |
| News21 | 0.0814 | 0.4004 | 0.1926 | 0.4240 | 0.2101 |
| **Mean** | **-0.0074** | — | — | — | — |
| **target_avg** | — | — | **0.1921** | — | — |

**关键发现**：
- p-MRR=-0.0074，指令敏感度为负，改写反而降低了指令敏感度
- target_avg=0.1921，所有方法中最低，检索质量也最差
- mTRAG 为多轮对话设计，将上下文相关查询改写为独立查询，但 FollowIR 的 changed 查询已包含指令信息，改写反而稀释了指令的关键约束
- 改写后的查询过于冗长和泛化，引入了过多不相关语义

---

### TongSearch-QR + RepLLaMA

**模型信息**：
- 论文/项目：TongSearch-QR（对话式查询改写）
- 基础模型：Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct + LoRA 微调
- 原始用途：多轮对话查询改写
- 编码器：RepLLaMA

**3B 版本评测结果**：

| 数据集 | p-MRR | OG MAP@1000 | Changed MAP@1000 | OG nDCG@5 | Changed nDCG@5 |
|--------|-------|-------------|-----------------|-----------|----------------|
| Core17 | -0.0310 | 0.3447 | 0.2206 | 0.4950 | 0.2749 |
| Robust04 | -0.0559 | 0.3401 | 0.2712 | 0.5392 | 0.3422 |
| News21 | -0.0186 | 0.4749 | 0.2364 | 0.4991 | 0.2239 |
| **Mean** | **-0.0352** | — | — | — | — |
| **target_avg** | — | — | **0.2386** | — | — |

**7B 版本评测结果**：

| 数据集 | p-MRR | OG MAP@1000 | Changed MAP@1000 | OG nDCG@5 | Changed nDCG@5 |
|--------|-------|-------------|-----------------|-----------|----------------|
| Core17 | 0.0328 | 0.3532 | 0.2166 | 0.4717 | 0.2401 |
| Robust04 | -0.0456 | 0.3231 | 0.2453 | 0.5221 | 0.3037 |
| News21 | 0.0019 | 0.4679 | 0.2260 | 0.4884 | 0.2178 |
| **Mean** | **-0.0036** | — | — | — | — |
| **target_avg** | — | — | **0.2266** | — | — |

**关键发现**：
- 3B 和 7B 版本的 p-MRR 均为负或接近零，指令敏感度极低
- 7B 版本 p-MRR (-0.0036) 优于 3B 版本 (-0.0352)，但仍远低于 DeIR-Dual V2 (0.1286)
- 3B 版本 target_avg (0.2386) 略高于 7B (0.2266)，但两者均低于 DeIR-Dual V2 (0.2828)
- 与其他对话改写器（ConvSearch-R1, mTRAG）的失败模式一致：改写是"去上下文化"而非"指令响应"

---

### Granite aLoRA QR + BGE

**模型信息**：
- 论文/项目：A Library of LLM Intrinsics for Retrieval-Augmented Generation (IBM Research)
- HuggingFace：ibm-granite/granite-3.2-8b-alora-rag-query-rewrite
- 基础模型：ibm-granite/granite-3.2-8b-instruct + LoRA adapter (rank=32, alpha=32, target_modules=[q_proj, v_proj, k_proj])
- 原始用途：多轮对话查询去上下文化（Activated LoRA / aLoRA）
- 适配方式：将 FollowIR 的 query + instruction 作为单轮 user 输入，触发 aLoRA 的 rewrite 特殊 prompt
- 编码器：BGE-large-en-v1.5（Granite 8B + BGE 1.3GB 可共存于 24GB GPU）
- 生成参数：greedy decoding, max_new_tokens=256
- 输出格式：JSON {"rewritten_question": <REWRITE>}

**评测结果**：

| 数据集 | p-MRR | OG MAP@1000 | Changed MAP@1000 | OG nDCG@5 | Changed nDCG@5 |
|--------|-------|-------------|-----------------|-----------|----------------|
| Core17 | -0.0376 | 0.2385 | 0.1878 | 0.3319 | 0.2340 |
| Robust04 | -0.0827 | 0.2203 | 0.1781 | 0.3461 | 0.2227 |
| News21 | 0.0511 | 0.4004 | 0.1978 | 0.4240 | 0.2002 |
| **Mean** | **-0.0231** | — | — | — | — |
| **target_avg** | — | — | **0.1887** | — | — |

**关键发现**：
- p-MRR=-0.0231，指令敏感度为负，改写反而降低了指令敏感度
- target_avg=0.1887，所有方法中最低（与 mTRAG 0.1921 接近），检索质量极差
- Granite aLoRA QR 为多轮对话设计，改写目标是"去上下文化"（将依赖历史的查询转为独立查询），但 FollowIR 的 changed 查询已包含指令信息，改写反而稀释了指令的关键约束
- Robust04 上 p-MRR 最低 (-0.0827)，说明改写对 Robust04 的指令信号破坏最严重
- 与其他对话改写器（ConvSearch-R1, mTRAG, TongSearch-QR）的失败模式一致：改写是"去上下文化"而非"指令响应"

---

### 外部改写方法汇总对比

**评测配置**：各方法使用各自推荐的编码器和改写器，FollowIR 三个测试集

| 方法 | 改写器 | 编码器 | mean p-MRR | target_avg | 核心方法 |
|------|--------|--------|-----------|-----------|---------|
| **DeIR-Dual V2** | Qwen3-4B | RepLLaMA | **0.1286** | **0.2828** | 奖惩双轨：Q_plus 增强 + Q_minus 惩罚 |
| Promptriever | — (端到端) | LLaMA 3.1 8B | 0.1001 | 0.3063 | 指令微调检索器 |
| INF-X Full | INF Aligner | INF Retriever | 0.0339 | 0.2704 | RL 改写 + 专用检索器 |
| BGE-Reasoner-Rewriter | Qwen2.5-7B | BGE-large-en | 0.0204 | 0.2103* | 5 查询分数聚合 |
| INF-X Aligner | INF Aligner | RepLLaMA | 0.0146 | 0.2274 | RL 改写 + RepLLaMA |
| HyDE | GPT-3.5 | RepLLaMA | 0.0143 | 0.2343 | 假想文档向量平均 |
| ConvSearch-R1 | Qwen2.5-3B | RepLLaMA | 0.0024 | 0.2341 | GRPO RL 对话改写 |
| DeepRetrieval | Qwen2.5-7B | RepLLaMA | -0.0029 | 0.2379 | RL 查询生成器 |
| RAG-Fusion | GPT-3.5 | RepLLaMA | -0.0030 | 0.2134 | 多查询 + RRF 融合 |
| mTRAG Rewriter | Qwen2.5-7B | BGE-large-en | -0.0074 | 0.1921 | 多轮对话改写器 |
| TongSearch-QR (7B) | Qwen2.5-7B | RepLLaMA | -0.0036 | 0.2266 | 对话改写器 |
| Query2Doc | GPT-3.5 | RepLLaMA | -0.0109 | 0.2622 | q [SEP] d' 拼接扩展 |
| TongSearch-QR (3B) | Qwen2.5-3B | RepLLaMA | -0.0352 | 0.2386 | 对话改写器 |
| RAG-QR | T5-large | RepLLaMA | -0.0338 | 0.2427 | T5 PPO 改写器 |
| Granite aLoRA QR | Granite-3.2-8B | BGE-large-en | -0.0231 | 0.1887 | IBM aLoRA 对话改写器 |

*BGE-Reasoner-Rewriter target_avg 仅基于 Core17+News21（Robust04 运行中）

**关键发现**：
1. **所有外部改写方法的 p-MRR 均远低于 DeIR-Dual V2**：最高仅 0.0339（INF-X），vs DeIR-Dual V2 的 0.1286
2. **端到端指令微调（Promptriever）的 p-MRR 最高**：0.1001，但仍低于 DeIR-Dual V2 的 0.1286
3. **改写类方法的共同失败模式**：改写是"语义增强"而非"指令响应"，无法区分 og/changed 查询的差异
4. **多查询聚合/融合方式（BGE-Reasoner-Rewriter, RAG-Fusion）进一步稀释指令信号**：p-MRR 接近零
5. **专用检索器（INF-X Retriever）编码质量高但不提升指令敏感度**：OG 指标高但 p-MRR 低
6. **mTRAG 在 target_avg 和 p-MRR 上均表现最差**：多轮对话改写器不适合 FollowIR 的单轮指令场景

---

## InstructIR 评测结果

### 基准信息

**论文**：InstructIR: A Benchmark for Instruction Following of Information Retrieval Models  
**作者**：Hanseok Oh, Hyunji Lee, Seonghyeon Ye, Haebin Shin, Hansol Jang, Changwook Jun, Minjoon Seo (SoftlyAI, KAIST AI, LG AI Research)  
**发表**：ACL 2024  
**链接**：https://aclanthology.org/2024.acl-long.123/

### 数据集详情

**数据集**：9,906 查询，16,072 文档，每查询 1 个相关文档
**配置**：RepLLaMA + Qwen3-4B + α=1.0, β=1.5, δ=0.05

| 指标 | Baseline (S_base) | DeIR-Dual V2 | Δ |
|------|-------------------|--------------|---|
| nDCG@5 | 0.8436 | 0.8434 | -0.0002 |
| nDCG@10 | 0.8597 | 0.8600 | +0.0003 |
| Recall@10 | 0.9967 | 0.9970 | +0.0003 |
| MAP@100 | 0.8158 | 0.8161 | +0.0003 |

**结论**：Baseline nDCG@10=0.86 已饱和（instruction 语义已融入向量），DeIR-Dual V2 无额外提升。纯 query（无 instruction）nDCG@10=0.48。

---

## NegConstraint 评测结果

### 基准信息

**论文**：Logical Consistency is Vital: Neural-Symbolic Information Retrieval for Negative-Constraint Queries  
**作者**：Ganlin Xu, Zhoujia Zhang, Wangyi Mei, Jiaqing Liang, Weijia Lu, Xiaodong Zhang, Zhifei Yang, Xiaofeng Ma, Yanghua Xiao, Deqing Yang (Fudan University, United Automotive Electronic Systems)  
**发表**：Findings of ACL 2025  
**链接**：https://aclanthology.org/2025.findings-acl.92/

### 数据集详情

**数据集**：198 测试查询（含显式否定/排除信号），3,946 文档

### RepLLaMA + Qwen3-4B + α=1.0, β=1.5, δ=0.05

| 指标 | Baseline | Q_plus only | DeIR-Dual V2 | Δ vs Baseline |
|------|----------|-------------|--------------|---------------|
| nDCG@5 | 0.7881 | 0.8314 | 0.8373 | **+0.0492** |
| nDCG@10 | 0.8010 | 0.8410 | 0.8444 | **+0.0434** |
| MAP@100 | 0.7382 | 0.7894 | 0.7955 | **+0.0573** |
| Recall@5 | 0.9495 | 0.9646 | 0.9697 | +0.0202 |
| Recall@10 | 0.9899 | 0.9949 | 0.9899 | 0 |

### BGE-large-en-v1.5 + Qwen3-4B + α=1.0, β=1.5, δ=0.05

| 指标 | Baseline | Q_plus only | DeIR-Dual V2 | Δ vs Baseline |
|------|----------|-------------|--------------|---------------|
| nDCG@5 | 0.7676 | 0.8099 | 0.8133 | **+0.0457** |
| nDCG@10 | 0.7773 | 0.8177 | 0.8184 | **+0.0411** |
| MAP@100 | 0.7083 | 0.7597 | 0.7663 | **+0.0580** |
| Recall@5 | 0.9545 | 0.9646 | 0.9596 | +0.0051 |
| Recall@10 | 0.9848 | 0.9899 | 0.9747 | -0.0101 |

**关键发现**：
- 两种编码器均显著提升：RepLLaMA nDCG@10 +4.3%，BGE nDCG@10 +4.1%
- Q_minus 利用率 99.5%（197/198），与 NegConstraint 否定查询特性完美匹配
- Q_minus 惩罚额外贡献：RepLLaMA MAP@100 +0.61%，BGE MAP@100 +0.66%
- **NegConstraint 是 DeIR-Dual V2 的最佳适配场景**（查询天然包含否定信号）

---

## COCO-Neg 评测结果

### 基准信息

**论文**：Vision-Language Models Do Not Understand Negation (NegBench)  
**作者**：Kumail Alhamoud, Shaden Alshammari, Yonglong Tian, Guohao Li, Philip H.S. Torr, Yoon Kim, Marzyeh Ghassemi (MIT, Google DeepMind, University of Oxford)  
**发表**：CVPR 2025  
**链接**：https://arxiv.org/abs/2501.09425

### 数据集详情

**数据集**：5,000 张 COCO 2017 val 图像，25,014 条否定 caption 查询，多模态 text-to-image retrieval
**配置**：Qwen3-4B + α=1.0, β=1.5, δ=0.05 | Q_minus 利用率 99.7%

### 逐模型结果

#### CLIP-OpenAI (ViT-B-32)

| 指标 | Original | Baseline (negated) | Q_plus only | DeIR-Dual V2 | Δ vs Baseline |
|------|----------|-------------------|-------------|--------------|---------------|
| R@1 | 0.2987 | 0.2537 | 0.2899 | 0.2979 | +0.0443 |
| R@5 | 0.5406 | 0.4862 | 0.5318 | 0.5428 | +0.0566 |
| R@10 | 0.6506 | 0.5979 | 0.6428 | 0.6530 | +0.0551 |

#### CLIP-DataComp (ViT-B-32)

| 指标 | Original | Baseline (negated) | Q_plus only | DeIR-Dual V2 | Δ vs Baseline |
|------|----------|-------------------|-------------|--------------|---------------|
| R@1 | 0.3628 | 0.2996 | 0.3511 | 0.3557 | +0.0561 |
| R@5 | 0.6204 | 0.5441 | 0.6113 | 0.6152 | +0.0710 |
| R@10 | 0.7256 | 0.6538 | 0.7178 | 0.7207 | +0.0669 |

#### CLIP-LAION400M (ViT-B-32)

| 指标 | Original | Baseline (negated) | Q_plus only | DeIR-Dual V2 | Δ vs Baseline |
|------|----------|-------------------|-------------|--------------|---------------|
| R@1 | 0.3488 | 0.2832 | 0.3361 | 0.3397 | +0.0566 |
| R@5 | 0.6030 | 0.5233 | 0.5903 | 0.5945 | +0.0713 |
| R@10 | 0.7107 | 0.6365 | 0.7002 | 0.7046 | +0.0681 |

#### NegCLIP (ViT-B-32)

| 指标 | Original | Baseline (negated) | Q_plus only | DeIR-Dual V2 | Δ vs Baseline |
|------|----------|-------------------|-------------|--------------|---------------|
| R@1 | 0.4156 | 0.3702 | 0.4050 | 0.4083 | +0.0381 |
| R@5 | 0.6869 | 0.6441 | 0.6758 | 0.6798 | +0.0357 |
| R@10 | 0.7892 | 0.7477 | 0.7818 | 0.7841 | +0.0364 |

### 跨模型汇总（R@5）

| 编码器 | Baseline | Q_plus only | DeIR-Dual V2 | Δ R@5 | 恢复率 |
|--------|----------|-------------|--------------|-------|--------|
| CLIP-OpenAI | 0.4862 | 0.5318 | 0.5428 | +0.0566 | 93.8% |
| CLIP-DataComp | 0.5441 | 0.6113 | 0.6152 | +0.0710 | 93.3% |
| CLIP-LAION400M | 0.5233 | 0.5903 | 0.5945 | +0.0713 | 90.3% |
| NegCLIP | 0.6441 | 0.6758 | 0.6798 | +0.0357 | 84.5% |

- 恢复率 = (DeIR R@5 - Baseline R@5) / (Original R@5 - Baseline R@5)
- **跨模态验证**：DeIR-Dual V2 在文本检索和多模态检索均有效
- Q_minus 惩罚在所有模型上均有额外贡献（R@5: +0.4% ~ +1.1%）

---

## ComLQ 评测结果

### 基准信息

**论文**：ComLQ: Benchmarking Complex Logical Queries in Information Retrieval  
**作者**：Ganlin Xu, Zhitao Yin, Linghao Zhang, Jiaqing Liang, Weijia Lu, Xiaodong Zhang, Zhifei Yang, Sihang Jiang, Deqing Yang (Fudan University, United Automotive Electronic Systems)  
**发表**：arXiv:2511.12004, 2025年11月  
**会议**：AAAI 2026  
**链接**：https://arxiv.org/abs/2511.12004

### 数据集详情

**数据集**：2,909 查询（14 种类型：9 无否定 + 5 含否定），11,251 文档
**配置**：BGE-large-en-v1.5 + Qwen3-4B | 最优参数：α=0.0, β=0.5, δ=0.3
**Q_minus 利用率**：95.5%（940/984 否定查询有 Q_minus）

### 全部查询 (2909)

| 指标 | Baseline | Q_plus only | DeIR-Dual V2 | Δ |
|------|----------|-------------|--------------|---|
| nDCG@10 | 0.5180 | 0.5240 | 0.5241 | +0.0061 |
| MAP@100 | 0.4059 | 0.4112 | 0.4112 | +0.0053 |
| Recall@10 | 0.8328 | 0.8392 | 0.8393 | +0.0065 |
| LSNC@100 | 0.0237 | 0.0237 | 0.0237 | 0 |

### 否定查询 (984)

| 指标 | Baseline | Q_plus only | DeIR-Dual V2 | Δ |
|------|----------|-------------|--------------|---|
| nDCG@10 | 0.4055 | 0.4070 | 0.4072 | +0.0016 |
| MAP@100 | 0.2889 | 0.2908 | 0.2908 | +0.0019 |

### 按查询类型 nDCG@10

| 类型 | Baseline | DeIR-Dual V2 | Δ | 否定? |
|------|----------|--------------|---|-------|
| 1p | 0.7016 | 0.7042 | +0.0026 | |
| 2i | 0.6564 | 0.6658 | +0.0095 | |
| 2in | 0.3768 | 0.3796 | +0.0028 | * |
| 2p | 0.6446 | 0.6578 | +0.0132 | |
| 2u | 0.4931 | 0.5035 | +0.0103 | |
| 3i | 0.5831 | 0.5817 | -0.0014 | |
| 3in | 0.4184 | 0.4238 | +0.0054 | * |
| 3p | 0.6466 | 0.6525 | +0.0060 | |
| inp | 0.4419 | 0.4368 | -0.0050 | * |
| ip | 0.5291 | 0.5393 | +0.0102 | |
| pi | 0.4425 | 0.4515 | +0.0091 | |
| pin | 0.4023 | 0.4032 | +0.0009 | * |
| pni | 0.3945 | 0.3956 | +0.0011 | * |
| up | 0.4954 | 0.5131 | +0.0177 | |

**关键发现**：
- **Q_minus 惩罚完全无效**：最优 α=0.0，DeIR-Dual V2 ≈ Q_plus only
- **根本原因 — 语义纠缠**：Cos(Q_base, Q_neg) 平均 0.7050（85.5% > 0.6），否定部分与正面意图共享大量语义
- 与 NegConstraint/COCO-Neg 的区别：后者否定信号是显式排除（"no red"），Q_neg 与 Q_base 语义正交 → 惩罚有效

---

## BEIR/NQ 评测结果

**数据集**：NQ (Natural Questions)，factoid QA，查询无真实否定信号
**配置**：BM25 top-1000 初筛 → RepLLaMA 重排，50 查询 / 50K 文档子集

### CONSERVATIVE 提示词（0% Q_minus 利用率 → Q_plus-only 模式）

| 指标 | BM25 Baseline | RepLLaMA Baseline | DeIR-Dual V2 | Δ vs RepLLaMA |
|------|---------------|-------------------|--------------|---------------|
| nDCG@10 | 0.4132 | 0.7857 | **0.7945** | **+1.1%** |
| MAP@100 | 0.3476 | 0.7199 | **0.7319** | **+1.7%** |
| Recall@100 | 0.8800 | 0.9600 | 0.9600 | 0 |
| MRR@10 | — | 0.7352 | **0.7482** | **+1.8%** |

**关键发现**：
- 旧提示词 (TSC_BALANCED) 导致 100% 伪否定 → nDCG@10 从 0.7857 暴跌至 0.6046 (-23.1%)
- 新提示词 (CONSERVATIVE) 正确识别无否定信号 → 自动退化为 Q_plus-only 模式
- **无否定信号的查询不应强制生成 Q_minus**

---

## 跨基准汇总

### DeIR-Dual V2 效果分类

| 效果等级 | 基准 | 原因 |
|---------|------|------|
| **显著提升** | NegConstraint, COCO-Neg | 显式否定信号，Q_neg 与 Q_base 语义正交 |
| **中等提升** | FollowIR, BEIR/NQ (Q_plus-only) | 部分否定信号或仅增强有效 |
| **微弱提升** | ComLQ | 语义纠缠，惩罚无效，仅 Q_plus 有贡献 |
| **无提升** | InstructIR | Baseline 已饱和（instruction 语义已融入向量） |

### 核心结论

1. **惩罚有效性取决于语义正交性**：Q_neg 与 Q_base 越正交，惩罚越有效
2. **Q_plus 增强普遍有效**：即使惩罚无效，Q_plus 仍能带来 0.6%~1.8% 提升
3. **CONSERVATIVE 提示词是安全选择**：无否定信号时自动退化为 Q_plus-only，避免伪否定灾难
4. **训练集推导参数 (V5) 在 FollowIR 上 p-MRR 最优**：α=1.0, β=1.926, δ=0.0

### 结果文件索引

| 基准 | 结果路径 |
|------|---------|
| FollowIR | evaluation/deir_dual_v2/{Core17,Robust04,News21}InstructionRetrieval/ |
| FollowIR (BGE-Reasoner-Rewriter) | results/bge_reasoner_rewriter/ |
| FollowIR (INF-X Full) | results/infx_retriever/full/ |
| FollowIR (INF-X Aligner) | results/infx_aligner/ |
| FollowIR (mTRAG Rewriter) | results/mtrag_rewriter/ |
| FollowIR (TongSearch-QR) | results/tongsearch_qr/ |
| FollowIR (Granite aLoRA QR) | results/granite_alora_qr/ |
| InstructIR | results/instructir/ |
| NegConstraint | results/negconstraint/ |
| COCO-Neg | results/coconeg/ |
| ComLQ | results/comlq/ |
| BEIR/NQ | results/beir/ |
| 参数推导 | results/train_derived_params.json, results/first_principles_params_v2.json |
