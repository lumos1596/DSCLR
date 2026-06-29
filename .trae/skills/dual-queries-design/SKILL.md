---
name: "dual-queries-design"
description: "Dual-queries (q_plus/q_minus) semantic design principles. Invoke when designing/refining q_plus/q_minus, analyzing q_minus quality issues, or debugging safety gate coupling problems."
---

# Dual-Queries Semantic Design Principles

This skill provides comprehensive guidelines for designing q_plus/q_minus in DeIR-Dual V2, including the "Specificity Paradox" problem, encoding noise mitigation, and experiment-validated best practices.

## Core Concepts

### Dual-Queries Structure

Each dual_query entry (JSONL format) contains:
- `qid`: Query ID
- `query`: Original query text
- `instruction`: User instruction
- `q_plus`: Enhancement query (positive requirements)
- `q_minus`: Negation query (exclusion criteria)
- `query_type`: Query classification

**Line number mapping**: Line `i` in JSONL corresponds to `q_idx=i` in evaluation logs.

## The Specificity Paradox (Critical Finding)

### Problem Definition

**Refining q_minus → More specific semantics → Overlap with query topic → tau_anchor rises → Safety gate becomes more aggressive → Mistakenly suppresses S_req enhancement**

This is a fundamental coupling issue in V8's safety gate mechanism:

```
τ_safety = tau_anchor = max(cos(q_minus, safe_anchor))
```

When q_minus semantics overlap with query topic, safe_anchor (which are relevant documents) naturally have high similarity with q_minus, causing tau_anchor to rise.

### Experimental Evidence (News21, batch_size=1 deterministic encoding)

| qidx | qid | q_minus refinement | tau_anchor Δ | threshold Δ | s_neg_max Δ | penalized Δ | nDCG@5 Δ | Outcome |
|------|-----|-------------------|--------------|-------------|-------------|-------------|----------|---------|
| 9 | 949 (Khashoggi) | "Khashoggi's death, investigations into his murder" → "murder investigation details, suspects and trial proceedings" | 0.709→0.626 | 0.729→0.646 | 0.702→0.652 | 0→6 | 0.626→0.000 | **Triggered penalty, killed relevant docs** |
| 16 | 958 (plant-based) | "non-plant-based products" → "traditional grocery items without plant-based alternatives" | 0.640→0.725 | 0.660→0.745 | 0.527→0.682 | 0→0 | 0.339→0.000 | **Safety gate suppressed S_req by 30%** |
| 7 | 947 (telescopes) | "documents without images" → "space telescope history without Hubble or Webb" | 0.457→0.712 | 0.477→0.732 | 0.427→0.669 | 0→0 | 0.737→0.391 | **Safety gate semantic coupling** |

### Root Cause Analysis

#### q9 (Khashoggi): Mis-triggered Penalty Track

- Original q_minus: "Khashoggi's death, investigations into his murder" (general negation)
- Refined q_minus: "murder investigation details, suspects and trial proceedings" (specific negation)
- **Problem**: Refined q_minus overlaps with query topic (Khashoggi family + murder context)
- **Effect**: tau_anchor dropped (0.709→0.626), s_neg_max exceeded threshold (0.652 > 0.646), triggered penalty on 6 documents, mistakenly killed relevant docs
- **nDCG@5 collapse**: 0.626 → 0.000

#### q16 (plant-based foods): Safety Gate Over-suppression

- Original q_minus: "non-plant-based products" (broad negation)
- Refined q_minus: "traditional grocery items without plant-based alternatives" (contextualized negation)
- **Problem**: Refined q_minus overlaps with query topic (plant-based foods for grocery stores)
- **Effect**: tau_anchor rose (0.640→0.725), safety gate dropped from ~0.93 to ~0.71, S_req enhancement suppressed by 30%
- **nDCG@5 collapse**: 0.339 → 0.000

#### q7 (telescopes): Safety Gate Semantic Coupling

- Original q_minus: "documents without images" (format-based exclusion)
- Refined q_minus: "space telescope history without Hubble or Webb" (content-based exclusion)
- **Problem**: Refined q_minus overlaps with query topic (Hubble/James Webb comparison)
- **Effect**: tau_anchor rose sharply (0.457→0.712), safety gate suppressed S_req enhancement
- **Mixed outcome**: nDCG@5 dropped (0.737→0.391), but MAP improved (0.297→0.369) — overall ranking improved but top-5 damaged

## Design Principles

### q_minus Design Rule (Avoid Specificity Paradox)

**CRITICAL**: q_minus should describe "**domains completely unrelated to the query**", NOT "**sub-topics related to query but should be excluded**"

#### Correct Examples

| Query | Correct q_minus | Rationale |
|-------|----------------|-----------|
| "Hubble vs James Webb comparison" | "medical diagnosis reports, financial quarterly earnings, sports game recaps" | Completely unrelated domains, no semantic overlap with telescope comparison |
| "Jamal Khashoggi's family" | "technology product reviews, travel destination guides, cooking recipes" | Unrelated domains, no overlap with Khashoggi/murder context |
| "Plant-based foods rebranding" | "automotive industry news, real estate market trends, political campaign updates" | Unrelated domains, no overlap with plant-based grocery |

#### Incorrect Examples (Trigger Specificity Paradox)

| Query | Incorrect q_minus | Problem |
|-------|------------------|---------|
| "Hubble vs James Webb comparison" | "space telescope history without Hubble or Webb, NASA missions unrelated to telescopes" | Semantic overlap with query topic (telescopes, NASA) |
| "Jamal Khashoggi's family" | "murder investigation details, suspects and trial proceedings" | Semantic overlap with query context (Khashoggi murder) |
| "Plant-based foods rebranding" | "traditional grocery items without plant-based alternatives" | Semantic overlap with query topic (grocery, plant-based) |

### q_plus Design Principles

q_plus should capture **positive requirements from instruction** with semantic specificity:

1. **Extract key entities**: Identify entities mentioned in instruction that should appear in relevant documents
2. **Add contextual constraints**: Include temporal, spatial, or format constraints when instruction specifies them
3. **Avoid over-specificity**: q_plus should be specific enough to guide retrieval but not so narrow that it excludes borderline relevant documents

#### Examples

| Instruction | q_plus Design |
|-------------|---------------|
| "On the day he was murdered, Saudi journalist Jamal Khashoggi entered the Saudi consulate in Istanbul" | "Jamal Khashoggi's family members, fiancee Hatice Cengiz, children from first marriage, Saudi government relations with family" |
| "Several factors have contributed to the current plant-based trend, formerly vegan" | "Rebranding and positioning of plant-based foods for mainstream grocery stores, wellness culture influence, environmental sustainability marketing, clean label trends" |

## Encoding Noise Problem (Critical Technical Issue)

### Root Cause

RepLLaMA uses **GPU float16 batch encoding**. When batch composition changes:
- Different texts → Different padding lengths → Different GPU float accumulation order → Encoding differences (~0.001-0.003)

**Effect**: Even if some q_minus texts remain unchanged, changing other q_minus in the same batch causes padding differences, which propagates through the encoder and produces different embeddings for the **unchanged** q_minus.

### Experimental Evidence

Before batch_size=1 fix:
- OG ranking (no q_minus usage): 0 queries changed — encoding is deterministic for same input
- Changed ranking (uses q_minus): **22 queries changed** — but only 8 q_minus were modified
- Unmodified group (24 queries): mean |dN5| = 0.0309, max |dN5| = 0.3857 — significant noise
- Modified group (8 queries): mean |dN5| = 0.1780
- **Signal-to-noise ratio**: 5.75 (modified/unmodified) — q_minus refinement effect exists but encoding noise is significant confounder

After batch_size=1 fix:
- Unmodified group: **0 queries changed, all metrics identical** — encoding noise completely eliminated
- Modified group: 7/8 queries changed — pure q_minus refinement effect
- **Deterministic encoding validated**: Same input always produces same output, regardless of other texts in "batch"

### Solution

**Encode queries with batch_size=1 to eliminate batch padding coupling**

Implementation (already integrated in `experiment_safe_anchor_threshold.py`):

```python
# 编码查询时使用 batch_size=1，消除 batch padding 导致的 float16 编码噪声
# 确保相同输入永远产生相同输出，无论其他 query 文本如何变化
_orig_batch_size = self.batch_size
self.batch_size = 1
logger.info("📊 编码 OG Q_base/Q_req/Q_neg (batch_size=1, 消除编码噪声)...")
q_base_emb_og = self._encode_queries(q_base_list_og)
q_req_emb_og = self._encode_queries(q_req_list_og)
q_neg_emb_og = self._encode_queries(q_neg_list_og)
logger.info("📊 编码 Changed Q_base/Q_req/Q_neg (batch_size=1, 消除编码噪声)...")
q_base_emb_ch = self._encode_queries(q_base_list_ch)
q_req_emb_ch = self._encode_queries(q_req_list_ch)
q_neg_emb_ch = self._encode_queries(q_neg_list_ch)
self.batch_size = _orig_batch_size
```

**Performance impact**: Minimal — only ~266 query texts to encode (News21 has 32 queries × 3 query types = 96 texts for changed set + OG set), batch_size=1 takes ~50 seconds total vs ~10 seconds with batch_size=64.

**Why not float32 or deterministic algorithms?**:
- float32 encoding would require modifying the encoder's internal model loading (RepLLaMA is float16 by design)
- `torch.use_deterministic_algorithms(True)` doesn't solve padding-induced accumulation order differences
- batch_size=1 is the **simplest and most reliable** solution — guarantees same input → same output

## Three Types of q_minus Quality Issues

### 1. Overly Broad Negation

**Problem**: q_minus too general, excludes entire semantic category that may contain relevant documents

**Example** (q14, 956-changed):
- Query: "What are the advantages and disadvantages of reverse mortgages?"
- Instruction: "Reverse mortgages can be a financial tool for homeowners aged 62 and older..."
- Original q_minus: "loans"
- **Issue**: "loans" excludes all loan-related content, including reverse mortgage explanations

**Refinement**: "payday loans, personal loans, loan interest rates, lending institutions" — specifies subcategories unrelated to reverse mortgages

**Effect**: nDCG@5: 0.391→0.248 (仍下降，因为 refined q_minus 与 query 主题"reverse mortgages"仍有语义重叠)

### 2. Format-based Exclusion

**Problem**: q_minus excludes based on document format rather than content, may remove relevant text-only documents

**Example** (q7, 947-changed):
- Query: "How do the Hubble and James Webb space telescopes compare?"
- Instruction: "The Hubble space telescope was launched in 1990 and is providing the world with dazzling images..."
- Original q_minus: "documents without images"
- **Issue**: Excludes text-only telescope comparison articles that may contain detailed technical comparisons

**Refinement**: "general astronomy without telescope comparison, space telescope history without Hubble or Webb, NASA missions unrelated to telescopes"

**Effect**: Triggered Specificity Paradox (see above) — nDCG@5 dropped from 0.737 to 0.391

### 3. Underspecified Entities

**Problem**: q_minus mentions entities without sufficient context, fails to distinguish relevant from irrelevant instances

**Example** (q23, 966-changed):
- Query: "When is the next total solar eclipse in the United States?"
- Instruction: "A total solar eclipse will be visible from the United States on April 8, 2024..."
- Original q_minus: "historical eclipses before the most recent"
- **Issue**: "historical eclipses" includes medieval eclipse records (irrelevant) but also recent historical eclipses (2017, may be relevant for comparison)

**Refinement**: "ancient eclipse records, medieval eclipse observations, historical eclipse mythology and superstitions" — explicitly excludes only ancient/mythological content

**Effect**: nDCG@5 unchanged (0.000→0.000), MAP dropped (0.097→0.031) — refined q_minus 与 query 主题"United States eclipse 2024"仍有语义重叠

## Mechanism Coupling Analysis

### Why V8 Safety Gate Coupled with q_minus Semantics?

V8's safety gate formula:

```
safety = 1 - sigmoid((S_neg - τ_safety) × t_safety)
τ_safety = tau_anchor = max(cos(q_minus, safe_anchor))
```

safe_anchor are LLM-generated "innocent documents" that are **relevant to the query**. When q_minus semantics overlap with query topic:
- safe_anchor (relevant docs) naturally have high similarity with q_minus
- tau_anchor rises
- τ_safety rises
- Safety gate becomes more aggressive (lower safety values)
- S_req enhancement is suppressed

### Two Potential Solutions

#### 1. Decouple Safety Gate from tau_anchor (Mechanism Fix)

**Proposal**: Safety gate should not depend on tau_anchor. Instead, use absolute S_req threshold:

```
safety = 1 - sigmoid((S_neg - τ_safety) × t_safety)
τ_safety = fixed_threshold OR S_req-based_threshold
```

**Rationale**: High S_req documents should not be suppressed by safety gate, regardless of tau_anchor

#### 2. q_minus Design Principle Adjustment (Design Fix)

**Proposal**: q_minus should describe "completely unrelated domains" to avoid semantic overlap with query topic

**Rationale**: If q_minus has no semantic overlap with query, tau_anchor will be low (safe_anchor are relevant to query, not to unrelated domains), safety gate will be less aggressive

**Current recommendation**: Use Design Fix (principle 2) until Mechanism Fix is implemented and validated

## Best Practices Summary

### q_minus Design Checklist

1. ❌ **Avoid semantic overlap with query topic**: q_minus should describe domains completely unrelated to the query
2. ❌ **Avoid format-based exclusion**: Exclude based on content, not document format
3. ✅ **Specify unrelated subcategories**: If query mentions "reverse mortgages", q_minus can exclude "payday loans, auto loans" (unrelated loan types)
4. ✅ **Use concrete entities with context**: "ancient eclipse records, medieval observations" (not just "historical eclipses")
5. ❌ **Avoid "without X" formulations**: "documents without images" → use "general astronomy without telescope comparison" (content-based)
6. ✅ **Test with batch_size=1 encoding**: Validate q_minus refinement effect without encoding noise

### q_plus Design Checklist

1. ✅ **Extract key entities from instruction**: Identify entities that should appear in relevant documents
2. ✅ **Add contextual constraints**: Temporal, spatial, format constraints when instruction specifies them
3. ❌ **Avoid over-specificity**: q_plus should guide retrieval, not exclude borderline relevant documents
4. ✅ **Balance specificity and breadth**: Enough detail to guide, enough breadth to include relevant variations

### Evaluation Validation

When testing q_plus/q_minus refinements:

1. ✅ **Use batch_size=1 encoding**: Eliminate encoding noise, ensure deterministic comparison
2. ✅ **Check unmodified queries**: Should have identical metrics (dN5=0, dAP=0) — validates encoding determinism
3. ✅ **Analyze tau_anchor changes**: If tau_anchor rises significantly after refinement, check for semantic overlap
4. ✅ **Check penalty track activation**: If `num_penalized_docs > 0` after refinement, investigate whether penalty is appropriate
5. ✅ **Check safety gate suppression**: If safety_mean drops significantly, investigate whether S_req enhancement is being over-suppressed

## Related Files

- dual_queries_v6 (News21): `dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_News21InstructionRetrieval.jsonl`
- dual_queries_v6 refined: `dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_News21InstructionRetrieval_qminus_refined.jsonl`
- safe_anchors (News21): `dataset/FollowIR_test/safe_anchors/safe_anchors_news21.json`
- Evaluation script: `eval/experiment_safe_anchor_threshold.py`
- Encoding noise fix: Line 695-715 in `experiment_safe_anchor_threshold.py`
- Baseline results (deterministic): `results/safe_anchor_v8_qminus_baseline_det/`
- Refined results (deterministic): `results/safe_anchor_v8_qminus_refined_det/`

## References

- dsclr-param-derivation skill: V8 per-query parameter derivation mechanism
- V8 safety gate formula: `safety = 1 - sigmoid((S_neg - τ_safety) × t_safety)`
- tau_anchor definition: `max(cos(q_minus, safe_anchor))`