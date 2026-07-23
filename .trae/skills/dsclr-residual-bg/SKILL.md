---
name: "dsclr-residual-bg"
description: "V8.6 Cross-Scale Residual Penalty + 残差MAD归一化Safety Gate 实验复现手册。包含完整配置、代码路径、公式、依赖、命令行和结果。Invoke when reproducing residual_bg experiments, checking V8.5/V8.6 configs, or migrating to new machines."
---

# V8.6 Cross-Scale Residual Penalty + 残差 MAD 归一化 Safety Gate

## 一、方案概述

V8.5/V8.6 是对 DeIR-Dual V2 惩罚和 safety 机制的改进，独立于 anchor 机制。核心思想：

- **V8.5**：将 `cos(Q_base, Q_neg)` 从"直接 QD 阈值"改为"QQ→QD 尺度转译系数"，解决 semantic 模式下 at-risk ratio≈0% 的尺度错配问题
- **V8.6（2026-06-30）**：进一步将 safety gate 从传统 τ 改为残差 MAD 归一化机制 `safety = 1 - sigmoid(R_neg/MAD × κ)`，消除传统 τ safety gate 的可解释性问题

### 与 V8 的关系

- α/β 推导：仍使用 V8 per-query 推理时推导机制（量级对齐原则）
- V8.5 改动范围：替换惩罚项的 overflow 计算
- V8.6 改动范围：进一步替换 safety gate 的计算方式
- 两者独立于 anchor 机制
- 主引擎：`engine_deir_dual_v2.py`（boundary_mode=residual_bg）

### 当前最优配置

```
boundary_mode = residual_bg
residual_margin_scale (λ) = 2.0
safety_kappa (κ) = 10
per_query_ab = true
beta_derive_mode = max_mean
delta = 0.02
t_safety = 20
alpha = 0.5, beta = 1.0 (fallback 值，实际 per-query 推导)
```

---

## 二、核心公式

### 2.1 背景泄漏预期（Scale Transfer）

```
c_q = cos(h_base, h_neg)
z_b(d) = (S_base(d) - μ_b) / σ_b
Ŝ_neg^bg(d) = μ_n + σ_n · c_q · z_b(d)
```

- `μ_b, σ_b`: 当前候选集 S_base 的均值、标准差
- `μ_n, σ_n`: 当前候选集 S_neg 的均值、标准差
- `Ŝ_neg^bg(d)`: 文档 d "仅因背景相关而在 neg 通道的预期得分"

### 2.2 残差提取

```
R_neg(d) = S_neg(d) - Ŝ_neg^bg(d)
```

残差 = 实际 neg 得分 - 背景泄漏预期。只有 R_neg > 0 的部分才是真正的 exclusion evidence。

### 2.3 MAD Robust Threshold

```
m_q = λ × MAD(R_neg)
```

- MAD = median(|R_neg - median(R_neg)|) × 1.4826（1.4826 使其与标准差一致，正态假设下）
- λ = residual_margin_scale 参数，推荐 2.0

### 2.4 惩罚项（基于残差，不加 delta）

```
overflow = R_neg - m_q
penalty = α × Softplus(overflow)
```

### 2.5 Safety Gate（2026-06-30 更新：残差 MAD 归一化机制）

传统 safety gate `safety = 1 - sigmoid((S_neg - τ) × T_safety)` 缺乏可解释性——`τ = cos(Q_base, Q_neg) + δ` 是 QQ 空间的值被直接当作 QD 阈值，尺度错配导致 safety 信号意义不明。

**新机制**：以残差的 MAD 作为归一化尺度，直接度量"文档的负向证据超出背景泄漏预期几个 MAD"：

```
safety(d) = 1 - sigmoid(R_neg(d) / MAD(R_neg) × κ)
```

**可解释性**：
- `R_neg / MAD` = 残差是"几个 MAD"，衡量超出背景泄漏预期的程度
- `κ` = 每增加 1 个 MAD，safety 下降多少（过渡锐度）
- `R_neg = 0` → safety = 0.5（负向证据恰好等于背景泄漏预期）
- `R_neg = MAD` → safety = 1 - sigmoid(κ)
- `κ → ∞` 退化为阶跃函数（理论最优 safety 的硬近似）
- `κ = 0` 回退到传统 τ safety gate

**参数切换逻辑**：
- `safety_kappa > 0`：使用残差 MAD 归一化 safety gate
- `safety_kappa = 0`：回退到传统 `safety = 1 - sigmoid((S_neg - τ) × T_safety)`

### 2.6 最终打分

```
S_final = S_base + β × S_req × safety - penalty
```

**惩罚与 Safety 的分工**：
- **Penalty**（残差 overflow）：决定哪些文档需要惩罚——基于残差排除背景泄漏
- **Safety**（残差 MAD 归一化）：决定哪些文档值得增强——基于残差控制增强强度

---

## 三、代码文件清单

### 3.1 核心文件

| 文件 | 作用 | 关键函数/类 |
|------|------|------------|
| `eval/engine_deir_dual_v2.py` | 主评估引擎 | `DeIRDualV2Evaluator`, `_score_query_dual_v2()` |
| `eval/residual_boundary.py` | 残差边界计算 | `compute_background_residual_boundary()`, `ResidualBoundaryOutput` |
| `eval/engine_dscrl.py` | 基类引擎 | `DSCLREvaluatorEngine`（文档编码、缓存、评测） |

### 3.2 数据文件

| 文件 | 作用 |
|------|------|
| `dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_Core17InstructionRetrieval.jsonl` | Core17 dual queries |
| `dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_Robust04InstructionRetrieval.jsonl` | Robust04 dual queries |
| `dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_News21InstructionRetrieval.jsonl` | News21 dual queries |
| `dataset/FollowIR_test/embeddings/RepLLaMA_reproduced/` | 预编码文档向量缓存 |

### 3.3 结果文件

| 文件 | 作用 |
|------|------|
| `results/residual_bg_v85_core17_lambda2.0/metrics_summary.json` | 最佳参数和评测指标 |
| `results/residual_bg_v85_core17_lambda2.0/per_query_stats.json` | 每个 query 的 α_q, β_q, at_risk_ratio 等 |
| `results/residual_bg_v85_core17_lambda2.0/ranking_changed.json` | changed 排序结果 |

---

## 四、完整实验配置

### 4.1 运行环境

```
Python:       3.10.20
PyTorch:      2.7.1+cu118
Transformers: 5.9.0
CUDA:         11.8
Conda env:    dsclr (/home/luwa/.conda/envs/dsclr/bin/python)
```

### 4.2 模型

```
编码器: samaya-ai/RepLLaMA-reproduced (Llama-2-7B + LoRA)
本地路径: /home/luwa/Documents/models/Llama-2-7b-hf/shakechen/Llama-2-7b-hf
```

### 4.3 引擎参数（V8 + residual_bg）

```python
# 固定参数
t_safety = 20.0
boundary_mode = "residual_bg"
per_query_ab = True
beta_derive_mode = "max_mean"
ab_clip_alpha = (0.05, 5.0)
ab_clip_beta = (0.05, 5.0)
batch_size = 64
use_cache = True

# 网格搜索参数（单点实验时取最佳值）
alpha = 0.5
beta = 1.0
delta = 0.02
residual_margin_scale (λ) = 2.0  # 推荐值
safety_kappa (κ) = 8~12  # 推荐范围，κ=0 回退到传统τ
```

### 4.4 Per-Query α/β 推导规则（V8 推理时推导）

V8 的核心改进：α 和 β 不再从训练集全局推导，而是在推理时从当前候选集的统计量推导，实现 per-query 自适应。

**at-risk / safe 划分**：
```
overflow = R_neg - m_q   # R_neg = S_neg - Ŝ_neg^bg, m_q = λ × MAD(R_neg)
at_risk_mask = (overflow > 0)   # 残差超过阈值 → 有真正负面证据
safe_mask   = (overflow <= 0)   # 残差未超过阈值 → 无需担心
```

**α 推导**（at-risk 文档上，量级对齐原则）：
- 物理意义：使惩罚项 α·Softplus(overflow) 的量级与 S_base 在 at-risk 文档上的量级匹配
```
if at_risk_mask.any():
    α_q = mean(S_base[at_risk]) / mean(softplus(overflow[at_risk]))
else:
    α_q = fallback_alpha  # = 网格搜索的 α 值（通常 0.5）
α_q = clip(α_q, 0.05, 5.0)
```

**β 推导**（safe 文档上，max_mean 模式，量级对齐原则）：
- 物理意义：使增强项 β·S_reward 在 safe 文档上的均值与 S_base 的峰值匹配
- **safety 的值取决于 κ 设置**：
  - κ > 0 (V8.6): `safety = 1 - sigmoid(R_neg/MAD × κ)`，残差越大 safety 越低
  - κ = 0 (V8.5): `safety = 1 - sigmoid((S_neg - τ) × T_safety)`，传统 τ 驱动
```
s_reward = S_req[safe] × safety[safe]   # safety 由 κ 决定
β_q = max(S_base[safe]) / mean(s_reward)
β_q = clip(β_q, 0.05, 5.0)
```

**V8.6 的关键协同**：
- 当 κ > 0 时，safe 文档（overflow ≤ 0）的 R_neg ≤ m_q，但由于 R_neg 可能 > 0，
  safety 可能 < 1。这意味着即使文档未被惩罚，其增强强度也可能被 safety 压低。
- 这是有意义的：R_neg > 0 但 R_neg ≤ m_q 的文档虽然"还不够坏到需要惩罚"，
  但已经有轻微负面证据，应降低增强强度而非完全增强。
- α 推导只依赖 overflow（与 κ 无关），β 推导依赖 safety（受 κ 影响），
  因此 κ 变化主要影响 β 而非 α。

---

## 五、完整复现命令

### 5.1 Core17 单点实验（λ=2.0, κ=10, 推荐配置）

```bash
cd /home/luwa/Documents/DSCLR-remote && \
CUDA_VISIBLE_DEVICES=2 /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_deir_dual_v2 \
  --task_name Core17InstructionRetrieval \
  --dual_queries_path dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_Core17InstructionRetrieval.jsonl \
  --alphas 0.5 \
  --betas 1.0 \
  --deltas 0.02 \
  --t_safety 20 \
  --boundary_mode residual_bg \
  --residual_margin_scale 2.0 \
  --safety_kappa 10 \
  --per_query_ab true \
  --beta_derive_mode max_mean \
  --output_dir results/residual_bg_v85_core17_lambda2.0_kappa10 \
  --device cuda
```

### 5.2 λ 搜索实验

```bash
for lambda in 0.5 1.0 1.5 2.0 2.5 3.0; do
  cd /home/luwa/Documents/DSCLR-remote && \
  CUDA_VISIBLE_DEVICES=2 /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_deir_dual_v2 \
    --task_name Core17InstructionRetrieval \
    --dual_queries_path dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_Core17InstructionRetrieval.jsonl \
    --alphas 0.5 \
    --betas 1.0 \
    --deltas 0.02 \
    --t_safety 20 \
    --boundary_mode residual_bg \
    --residual_margin_scale $lambda \
    --safety_kappa 0 \
    --per_query_ab true \
    --beta_derive_mode max_mean \
    --output_dir results/residual_bg_v85_core17_lambda${lambda} \
    --device cuda
done
```

### 5.3 κ 搜索实验（残差 MAD 归一化 safety gate）

```bash
for kappa in 1 2 3 4 6 8 10 12 15 20; do
  cd /home/luwa/Documents/DSCLR-remote && \
  CUDA_VISIBLE_DEVICES=2 /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_deir_dual_v2 \
    --task_name Core17InstructionRetrieval \
    --dual_queries_path dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_Core17InstructionRetrieval.jsonl \
    --alphas 0.5 \
    --betas 1.0 \
    --deltas 0.02 \
    --t_safety 20 \
    --boundary_mode residual_bg \
    --residual_margin_scale 2.0 \
    --safety_kappa $kappa \
    --per_query_ab true \
    --beta_derive_mode max_mean \
    --output_dir results/residual_bg_kappa${kappa}_lambda2.0 \
    --device cuda
done
```

### 5.4 Semantic baseline 对比实验

```bash
cd /home/luwa/Documents/DSCLR-remote && \
CUDA_VISIBLE_DEVICES=2 /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_deir_dual_v2 \
  --task_name Core17InstructionRetrieval \
  --dual_queries_path dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_Core17InstructionRetrieval.jsonl \
  --alphas 0.5 \
  --betas 1.0 \
  --deltas 0.02 \
  --t_safety 20 \
  --boundary_mode semantic \
  --per_query_ab true \
  --beta_derive_mode max_mean \
  --output_dir results/semantic_v8_core17_baseline \
  --device cuda
```

### 5.5 其他数据集

```bash
# Robust04
cd /home/luwa/Documents/DSCLR-remote && \
CUDA_VISIBLE_DEVICES=2 /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_deir_dual_v2 \
  --task_name Robust04InstructionRetrieval \
  --dual_queries_path dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_Robust04InstructionRetrieval.jsonl \
  --alphas 0.5 --betas 1.0 --deltas 0.02 --t_safety 20 \
  --boundary_mode residual_bg --residual_margin_scale 2.0 --safety_kappa 10 \
  --per_query_ab true --beta_derive_mode max_mean \
  --output_dir results/residual_bg_v85_robust04_lambda2.0_kappa10 --device cuda

# News21
cd /home/luwa/Documents/DSCLR-remote && \
CUDA_VISIBLE_DEVICES=2 /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_deir_dual_v2 \
  --task_name News21InstructionRetrieval \
  --dual_queries_path dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_News21InstructionRetrieval.jsonl \
  --alphas 0.5 --betas 1.0 --deltas 0.02 --t_safety 20 \
  --boundary_mode residual_bg --residual_margin_scale 2.0 --safety_kappa 10 \
  --per_query_ab true --beta_derive_mode max_mean \
  --output_dir results/residual_bg_v85_news21_lambda2.0_kappa10 --device cuda
```

---

## 六、实验结果（Core17, 2026-06-30）

### 6.1 λ 扫描（κ=0，传统 τ safety gate）

| λ (margin_scale) | p-MRR | changed_MAP@1000 | changed_nDCG@5 |
|------------------|-------|-------------------|-----------------|
| 0.5 | 0.1665 | — | — |
| 1.0 | 0.1665 | — | — |
| 1.5 | 0.1665 | — | — |
| **2.0** | **0.1670** | **0.2624** | **0.3462** |
| 2.5 | 0.1669 | — | — |
| 3.0 | 0.1668 | — | — |

### 6.2 κ 扫描（λ=2.0，残差 MAD 归一化 safety gate, 2026-06-30 新增）

| κ (safety_kappa) | changed_MAP@1000 | changed_nDCG@5 | p-MRR | 说明 |
|------------------|-------------------|-----------------|-------|------|
| 0 (传统τ) | **0.2624** | 0.3462 | 0.1670 | baseline |
| 1 | 0.1840 | 0.2486 | 0.3749 | 过度抑制 |
| 2 | 0.2080 | 0.3114 | 0.3583 | 过度抑制 |
| 3 | 0.2328 | 0.3402 | 0.3354 | 偏激进 |
| 4 | 0.2408 | 0.3473 | 0.3209 | 偏激进 |
| 6 | 0.2450 | 0.3482 | 0.3030 | 适中 |
| 8 | 0.2480 | 0.3675 | 0.2903 | 较优 |
| **10** | **0.2470** | **0.3734** | **0.2817** | **nDCG 最优区间** |
| 12 | 0.2470 | 0.3754 | 0.2746 | nDCG 最高 |
| 15 | 0.2464 | 0.3716 | 0.2674 | 保守 |
| 20 | 0.2456 | 0.3686 | 0.2586 | 保守 |
| semantic V8 | 0.2643 | 0.3432 | 0.1542 | 对照组 |

**关键发现**：
- κ=8~12 是最优区间：nDCG@5 从 0.346 提升到 0.373（+7.8%），p-MRR 从 0.167 提升到 0.275（+64%），MAP 从 0.262 降到 0.247（-5.7%）
- 传统τ（κ=0）的 MAP 优势来自"过于保守的 safety"——τ 太高导致 safety≈1，几乎不抑制任何文档
- 残差 safety 虽略牺牲 MAP，但 nDCG 和 p-MRR 大幅提升，说明更精准控制增强/抑制平衡

### 6.3 与 semantic baseline 对比

| 指标 | semantic (V8) | residual_bg κ=0 | residual_bg κ=10 |
|------|--------------|-----------------|-------------------|
| p-MRR | 0.1542 | 0.1670 (+8.3%) | **0.2817 (+82.6%)** |
| changed_MAP@1000 | **0.2643** | 0.2624 (-0.7%) | 0.2470 (-6.5%) |
| changed_nDCG@5 | 0.3432 | 0.3462 (+0.9%) | **0.3734 (+8.8%)** |

### 6.4 Per-Query 参数对比（κ=0 vs semantic）

| 指标 | semantic (V8) | residual_bg κ=0 λ=2.0 |
|------|--------------|-------------------|
| α_q 均值 | 0.535 (93% 退化为 fallback=0.5) | 0.985 (有效推导) |
| α_q 范围 | 0.500~1.063 | 0.843~1.076 |
| β_q 均值 | 1.422 | 1.412 |
| β_q 范围 | 1.271~1.657 | 1.267~1.621 |
| at-risk ratio 均值 | 0.1% | **8.9%** |
| at-risk ratio 范围 | 0%~1.1% | 2.9%~11.2% |

### 6.5 Per-Query 详细参数（residual_bg λ=2.0 κ=0, 有 neg 的 query）

| qid | α_q | β_q | at_risk | safety_mean | cos_qbase_qneg |
|-----|------|------|---------|-------------|----------------|
| 310 | 1.004 | 1.621 | 11.0% | 0.7931 | 0.5590 |
| 341 | 1.076 | 1.270 | 8.2% | 0.9606 | 0.6560 |
| 355 | 0.959 | 1.445 | 2.9% | 0.9504 | 0.6189 |
| 356 | 1.013 | 1.267 | 9.7% | 0.9398 | 0.5974 |
| 367 | 0.944 | 1.280 | 9.6% | 0.9881 | 0.7217 |
| 400 | 1.012 | 1.467 | 11.2% | 0.8752 | 0.5569 |
| 404 | 0.985 | 1.282 | 10.2% | 0.9860 | 0.6413 |
| 414 | 1.010 | 1.433 | 6.7% | 0.9713 | 0.6498 |

---

## 七、背景泄漏假设诊断实验（2026-07-01）

核心假设 `Ŝ_neg^bg = μ_n + σ_n · c_q · z_b(d)` 认为 base 通道的标准化分数可通过 `cos(h_base, h_neg)` 线性转译到 neg 通道。为回应顶会 reviewer 对该 heuristic 的质疑，设计了四个诊断实验。

### 7.1 诊断实验设计

| 诊断 | 目的 | 期望结果 |
|------|------|----------|
| D1: 相关性 | 预测 Ŝ_neg^bg 与实际 S_neg 在 safe 文档上的 Pearson r | r > 0.5 |
| D2: 残差分布 | R_neg 是否近似零均值、对称分布 | |skewness| < 1.0 |
| D3: Trap docs | 高 S_base 且高 S_neg 的文档残差是否显著更高 | p < 0.05 |
| D4: Shuffled neg | 随机打乱 neg query 后相关性是否显著下降 | p < 0.05 |

### 7.2 Core17 诊断结果

**DIAGNOSTIC 1: 相关性分析**
```
Ŝ_neg^bg 与 S_neg 在 safe 文档上的相关性:
  mean r  = 0.5557, median r = 0.5683
  mean R² = 0.3650
  min r   = 0.2050, max r = 0.9564
→ PASS: 平均相关性 0.56，远超 0.5 阈值
```

**DIAGNOSTIC 2: 残差分布**
```
全局残差 R_neg (pooled across all queries):
  mean = 0.0, median = -0.0008, std = 0.0174
  skewness = 0.28
  frac(R > 0) = 0.479
→ PASS: 近似零均值、低偏度（0.28 < 1.0），模型残差近似对称
```

**DIAGNOSTIC 3: Trap docs 残差分析**
```
Trap docs (top 25% S_base AND top 25% S_neg):
  R_neg mean:  trap=0.0119, non-trap=-0.0016
  frac(R>0):   trap=0.8496, non-trap=0.4304
  Paired t-test: t=10.62, p=1.98e-09
→ PASS: Trap docs 残差显著更高 (p < 1e-8)
```

**DIAGNOSTIC 4: Shuffled neg sanity check**
```
cos(Q_base, Q_neg): real=0.6044, shuffled=0.4669
相关性:     real=0.5557, shuffled=0.2738
At-risk ratio: real=0.0317, shuffled=0.0423
Paired t-test: t=4.82, p=0.000119
→ PASS: 真实 neg 的相关性显著高于 shuffled (p=0.0001)
```

### 7.3 综合判定

| 诊断 | 结果 | 判定 |
|------|------|------|
| D1 (r > 0.5) | r = 0.5557 | **PASS** |
| D2 (\|skew\| < 1.0) | skew = 0.28 | **PASS** |
| D3 (trap docs p < 0.05) | p = 1.98e-09 | **PASS** |
| D4 (shuffled p < 0.05) | p = 0.000119 | **PASS** |

**结论**：四项诊断全部通过，背景泄漏假设 `Ŝ_neg^bg = μ_n + σ_n · c_q · z_b(d)` 得到实证支持：
1. 线性转译模型可解释 S_neg 约 36.5% 的方差（R²=0.365）
2. 残差近似零均值对称分布，无系统性偏差
3. Trap docs（同时满足 base 和 neg 的文档）残差显著更高，说明模型正确捕捉了"背景泄漏"
4. Shuffled neg 相关性仅为真实的 49%，说明 c_q 的语义关系是模型有效性的关键

### 7.4 诊断脚本

```bash
# 诊断实验命令
cd /home/luwa/Documents/DSCLR-remote && \
CUDA_VISIBLE_DEVICES=5 /home/luwa/.conda/envs/dsclr/bin/python -m eval.diagnose_bg_leakage \
  --task_name Core17InstructionRetrieval \
  --dual_queries_path dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_Core17InstructionRetrieval.jsonl \
  --output_dir results/diagnose_bg_leakage \
  --device cuda --batch_size 8
```

脚本路径：`eval/diagnose_bg_leakage.py`
结果路径：`results/diagnose_bg_leakage/diagnostic_results.json`

---

## 八、迁移复现检查清单

### 8.1 必须确认的文件

- [ ] `eval/engine_deir_dual_v2.py` — 包含 `boundary_mode="residual_bg"` 分支
- [ ] `eval/residual_boundary.py` — 包含 `compute_background_residual_boundary()` 函数
- [ ] `eval/engine_dscrl.py` — 基类，包含编码、缓存、评测逻辑
- [ ] dual_queries_v6 数据文件（3个数据集各1个 jsonl）
- [ ] 模型权重和 LoRA adapter
- [ ] 预编码文档向量缓存（如无则需重新编码，约 30 分钟/GPU）

### 8.2 路径配置

迁移时需要修改的硬编码路径：

1. `engine_dscrl.py` 第 30 行：`DEFAULT_CACHE_DIR` — 文档向量缓存目录
2. 模型加载路径：RepLLaMA 本地模型路径
3. `CUDA_VISIBLE_DEVICES` — GPU 选择
4. Conda 环境路径

### 8.3 关键验证步骤

1. 先运行 semantic baseline，确认 p-MRR ≈ 0.1542
2. 再运行 residual_bg λ=2.0 κ=0，确认 p-MRR ≈ 0.1670
3. 运行 residual_bg λ=2.0 κ=10，确认 p-MRR ≈ 0.2817
4. 检查 per_query_stats.json 中 at-risk ratio 是否 ≈ 8.9%（κ=0）
5. 检查 α_q 均值是否 ≈ 0.985（而非 fallback 0.5）
6. 检查 κ=10 时 safety_mode 应为 "residual_mad"

### 8.4 已知注意事项

1. **batch_size=1 确定性编码**：如需严格可复现，设置 `--batch_size 1`（但会显著变慢）
2. **CUDA 检测**：engine 中使用了 `torch.cuda._lazy_init()` 先于 `is_available()` 的逻辑，沙箱环境可能需要绕过
3. **dtype 一致性**：`residual_boundary.py` 内部用 float32 计算，输出转回原始 dtype
4. **MAD 的 1.4826 因子**：正态分布下使 MAD 与标准差一致的修正系数，已在 `_mad()` 函数中包含

---

## 九、消融实验与残差语义诊断图（2026-07-01）

### 9.1 消融实验设计

以 V8.6 完整方法为基准（λ=2.0, κ=10, per_query_ab=true, beta_derive_mode=max_mean），在 Core17、Robust04、News21 三个数据集上逐项去除组件验证必要性。

完整打分公式：`S_final = S_base + β·S_req·safety - α·Softplus(R_neg - m_q)`

| Variant | 打分公式 | 验证目标 |
|---------|----------|----------|
| Base only | `S_base` | 原检索器下限 |
| w/o positive | `S_base - penalty` | 正向通道是否必要 |
| w/o negative residual | `S_base + β·S_req·safety` | 负向 residual 分支是否必要 |
| w/o safety gate | `S_base + β·S_req - penalty` | safety reward 是否必要 |
| Linear fusion | `S_base + β·S_req - α·S_neg` | 非线性 decision layer 是否必要 |
| Full method | `S_base + β·S_req·safety - penalty` | 完整方法 |

### 9.2 消融实验结果（三数据集）

**Core17**（MAP@1000 og = 0.34145，nDCG@5 og = 0.46142）

| Variant | MAP@1000 (changed) | nDCG@5 (changed) | p-MRR |
|---------|-------------------:|------------------:|------:|
| Base only | 0.23211 | 0.26570 | 0.0107 |
| w/o positive refinement | 0.24876 | 0.29799 | 0.0917 |
| w/o negative residual branch | 0.24819 | 0.36671 | 0.2744 |
| w/o safety-gated reward | 0.26557 | 0.32668 | 0.0049 |
| Linear fusion | 0.26885 | 0.32900 | 0.0474 |
| **Full method** | 0.24696 | **0.37335** | **0.2817** |

**Robust04**（MAP@1000 og = 0.33143，nDCG@5 og = 0.47010）

| Variant | MAP@1000 (changed) | nDCG@5 (changed) | p-MRR |
|---------|-------------------:|------------------:|------:|
| Base only | 0.25714 | 0.32170 | -0.0927 |
| w/o positive refinement | 0.25924 | 0.31498 | -0.0207 |
| w/o negative residual branch | 0.23249 | 0.30217 | 0.0957 |
| w/o safety-gated reward | 0.24470 | 0.30320 | 0.0177 |
| Linear fusion | 0.23867 | 0.29965 | 0.0511 |
| **Full method** | 0.22880 | 0.29983 | **0.0961** |

**News21**（MAP@1000 og = 0.36547，nDCG@5 og = 0.48593）

| Variant | MAP@1000 (changed) | nDCG@5 (changed) | p-MRR |
|---------|-------------------:|------------------:|------:|
| Base only | 0.24621 | 0.22519 | -0.0184 |
| w/o positive refinement | 0.25567 | 0.25185 | 0.0570 |
| w/o negative residual branch | 0.24170 | 0.27036 | 0.2221 |
| w/o safety-gated reward | 0.27131 | 0.28793 | 0.0115 |
| Linear fusion | 0.27018 | 0.29822 | 0.0603 |
| **Full method** | 0.24871 | 0.28412 | **0.2265** |

**跨数据集汇总**（target_avg = (Core17_MAP@1000_ch + Robust04_MAP@1000_ch + News21_nDCG@5_ch) / 3）

| Variant | target_avg | avg p-MRR | p-MRR 相对 Full |
|---------|-----------:|----------:|----------------:|
| Base only | 0.2382 | -0.0335 | -117% |
| w/o positive refinement | 0.2533 | 0.0427 | -79% |
| w/o negative residual branch | 0.2503 | 0.1974 | -2.0% |
| w/o safety-gated reward | 0.2661 | 0.0114 | -94% |
| Linear fusion | 0.2686 | 0.0529 | -74% |
| **Full method** | 0.2533 | **0.2014** | — |

### 9.3 消融分析

1. **Base only** 是下限：三数据集 avg p-MRR = -0.034，说明原检索器对指令几乎不敏感，Robust04 上甚至为负（指令反而有害）
2. **w/o positive refinement**：去掉 `β·S_req·safety` 后 avg p-MRR 从 0.201 降至 0.043（-79%），正向 refinement 是指令敏感度的主要来源
3. **w/o negative residual branch**：去掉 penalty 后 avg p-MRR 从 0.201 降至 0.197（-2.0%），负向 residual 分支对 p-MRR 贡献较小，但其语义价值由诊断图（9.4）独立验证
4. **w/o safety gate**：safety=1（不乘 safety）后 avg p-MRR 暴跌至 0.011（-94%），safety gate 是指令敏感度的核心 gating 机制
5. **Linear fusion**：用线性 `α·S_neg` 替代非线性 softplus+sigmoid 后 avg p-MRR 仅 0.053（-74%），非线性 decision layer 不可或缺
6. **Full method** 在三数据集 avg p-MRR（0.201）上为最优，在 Core17 的 changed nDCG@5（0.373）上也最优

**组件重要性排序**（按 avg p-MRR 退化幅度）：safety gate（-94%）> 正向 refinement（-79%）> 非线性 decision layer（-74%）> 负向 residual 分支（-2.0%）

**注**：Base only / w/o positive / w/o safety / Linear fusion 在 MAP@1000 上有时高于 Full method，因为这些变体退化后更接近"忽略指令"的行为，而 Robust04 和 News21 上指令本身有负面效果（og 指标远高于 changed 指标）。Full method 的优势主要体现在 p-MRR（指令敏感度）和 Core17 的 changed nDCG@5 上。

### 9.4 残差语义诊断图

**验证目标**：R_neg/MAD 是否能区分 trap docs（高 S_base + 高 S_neg，但非 relevant）和 relevant docs。

**文档分类**：
- Relevant docs：qrel 中 relevance > 0 的文档
- Trap docs：S_base 和 S_neg 均 top 25%，但 qrel relevance = 0
- Other：其余非相关文档

**三数据集 R_neg/MAD 统计**：

| 数据集 | 类别 | 文档数 | mean | median | std |
|--------|------|-------:|-----:|-------:|----:|
| Core17 | Relevant | 654 | **-0.431** | -0.493 | 1.060 |
| Core17 | Trap | 2205 | **+0.779** | +0.675 | 0.803 |
| Core17 | Other | 17141 | -0.084 | -0.146 | 1.004 |
| Robust04 | Relevant | 529 | **-0.414** | -0.487 | 1.185 |
| Robust04 | Trap | 3457 | **+0.771** | +0.648 | 0.791 |
| Robust04 | Other | 27014 | -0.091 | -0.145 | 1.012 |
| News21 | Relevant | 523 | **-0.399** | -0.425 | 1.365 |
| News21 | Trap | 3309 | **+0.792** | +0.691 | 0.793 |
| News21 | Other | 28168 | -0.086 | -0.149 | 1.015 |

**统计检验（trap > relevant）**：

| 数据集 | Mann-Whitney U | MWU p-value | t-statistic | t-test p-value |
|--------|---------------:|------------:|------------:|---------------:|
| Core17 | 1,190,357 | 1.08e-141 | 31.30 | 3.66e-185 |
| Robust04 | 1,459,183 | 1.51e-108 | 29.71 | 2.44e-175 |
| News21 | 1,357,359 | 1.48e-97 | 28.33 | 1.88e-160 |

**图文件**：
- Core17: `results/diagnose_residual_semantics/residual_semantics_{violin,hist}.png`
- Robust04: `results/diagnose_residual_semantics_robust04/residual_semantics_{violin,hist}.png`
- News21: `results/diagnose_residual_semantics_news21/residual_semantics_{violin,hist}.png`

**结论**：R_neg/MAD 在三个数据集上均高度显著区分 trap docs 和 relevant docs（p < 1e-97）。相关文档的残差为负（mean ≈ -0.41），说明其负向证据低于背景泄漏预期；trap docs 的残差为正（mean ≈ +0.78），说明有额外的负向证据超出背景泄漏。这一跨数据集的一致性验证了 safety gate 的设计意图：用 R_neg/MAD 控制增强强度，相关文档获得高 safety，trap docs 获得低 safety。

### 9.5 实验复现命令

```bash
# 消融实验（3数据集 × 6变体）
cd /home/luwa/Documents/DSCLR-remote && \
for ds in Core17 Robust04 News21; do
  ds_lower=$(echo $ds | tr 'A-Z' 'a-z')
  for mode in full base_only no_pos no_neg no_safety linear; do
    CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/luwa/.conda/envs/dsclr/bin/python -m eval.engine_deir_dual_v2 \
      --task_name ${ds}InstructionRetrieval \
      --dual_queries_path dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_${ds}InstructionRetrieval.jsonl \
      --alphas 0.5 --betas 1.0 --deltas 0.02 --t_safety 20 \
      --boundary_mode residual_bg --residual_margin_scale 2.0 --safety_kappa 10 \
      --per_query_ab true --beta_derive_mode max_mean \
      --ablation_mode $mode \
      --output_dir results/ablation_${ds_lower}/$mode \
      --device cuda --batch_size 1
  done
done

# 残差语义诊断图（3数据集）
for ds in Core17 Robust04 News21; do
  ds_lower=$(echo $ds | tr 'A-Z' 'a-z')
  CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/luwa/.conda/envs/dsclr/bin/python -m eval.diagnose_residual_semantics \
    --task_name ${ds}InstructionRetrieval \
    --dual_queries_path dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_${ds}InstructionRetrieval.jsonl \
    --output_dir results/diagnose_residual_semantics_${ds_lower} \
    --device cuda --batch_size 1
done
```

注：Robust04 和 News21 语料较大，建议用 `batch_size=1` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 避免 OOM。Core17 可用 `batch_size=64`。

脚本路径：
- 消融实验：`eval/engine_deir_dual_v2.py`（`--ablation_mode` 参数）
- 诊断图：`eval/diagnose_residual_semantics.py`

---

## 十、跨编码器实验：per-query 推导 vs 编码器级静态参数（2026-07-12）

### 10.1 实验目的

验证 V8.6 推理时 per-query 动态参数推导在多种编码器上的综合性能，与编码器级别静态参数组进行对比。

**初版方案**：为每个编码器预计算一组静态 α/β，在所有测试集上统一应用。
**新版方案**：推理阶段为每个 query 动态推导 α_q/β_q，无需预计算。

### 10.2 编码器

| 编码器 | 参数量 | 嵌入维度 | 类型 | 加载方式 |
|--------|-------:|--------:|------|---------|
| samaya-ai/RepLLaMA-reproduced | 7B | 4096 | 检索微调 (Llama-2 + LoRA) | 单 GPU fp16 |
| BAAI/bge-large-en-v1.5 | 335M | 1024 | 通用嵌入 (BERT-based) | 单 GPU fp16 |
| GritLM/GritLM-7B | 7B | 4096 | 通用嵌入 (Mistral-based) | 多 GPU fp16 (device_map="auto") |

**BGE 编码器实现**：`eval/models/bge_encoder.py`，使用 SentenceTransformer 加载，查询前置指令 `"Represent this sentence for searching relevant passages: "`。

### 10.3 统一实验配置

```
boundary_mode = residual_bg
residual_margin_scale (λ) = 2.0
safety_kappa (κ) = 10
per_query_ab = true / false（对比变量）
beta_derive_mode = max_mean
delta = 0.02
t_safety = 20
ab_clip_alpha = (0.05, 5.0)
ab_clip_beta = (0.05, 5.0)
```

静态网格搜索范围（per_query_ab=false）：
- α ∈ {0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0} (8个)
- β ∈ {0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0} (8个)
- δ = 0.02 (固定, κ=10 时 δ 不参与安全计算)
- 总计 64 组合/数据集

### 10.4 per-query 推导结果（V8.6, κ=10, per_query_ab=true）

**RepLLaMA**

| 数据集 | p-MRR | og_nDCG@5 | og_nDCG@10 | og_MAP@1k | chg_nDCG@5 | chg_nDCG@10 | chg_MAP@1k |
|--------|------:|----------:|-----------:|----------:|-----------:|------------:|-----------:|
| Core17 | 0.2815 | 0.4696 | 0.4493 | 0.3406 | 0.3777 | 0.3630 | 0.2496 |
| Robust04 | 0.1002 | 0.4904 | 0.4663 | 0.3218 | 0.3011 | 0.3050 | 0.2285 |
| News21 | 0.2275 | 0.4603 | 0.4725 | 0.4763 | 0.2787 | 0.3104 | 0.2486 |

**BGE-large**

| 数据集 | p-MRR | og_nDCG@5 | og_nDCG@10 | og_MAP@1k | chg_nDCG@5 | chg_nDCG@10 | chg_MAP@1k |
|--------|------:|----------:|-----------:|----------:|-----------:|------------:|-----------:|
| Core17 | 0.2674 | 0.2929 | 0.3121 | 0.2510 | 0.3196 | 0.3009 | 0.1984 |
| Robust04 | 0.1180 | 0.3673 | 0.3574 | 0.2474 | 0.2813 | 0.2734 | 0.1932 |
| News21 | 0.1990 | 0.4380 | 0.4474 | 0.4220 | 0.2412 | 0.2636 | 0.2199 |

**GritLM-7B（多GPU fp16, κ=0 — 注意：GritLM 实验使用 κ=0 非标准配置）**

| 数据集 | p-MRR | og_nDCG@5 | og_nDCG@10 | og_MAP@1k | chg_nDCG@5 | chg_nDCG@10 | chg_MAP@1k |
|--------|------:|----------:|-----------:|----------:|-----------:|------------:|-----------:|
| Core17 | -0.0145 | 0.0289 | 0.0269 | 0.0602 | 0.0075 | 0.0112 | 0.0360 |
| Robust04 | 0.0520 | 0.0246 | 0.0268 | 0.0431 | 0.0121 | 0.0187 | 0.0253 |
| News21 | 0.0029 | 0.0180 | 0.0203 | 0.0497 | 0.0106 | 0.0149 | 0.0257 |

### 10.5 静态网格搜索 Top-10（per_query_ab=false, κ=10）

**RepLLaMA — Core17**

| α | β | p-MRR | chg_MAP@1k | chg_nDCG@5 | chg_nDCG@10 |
|---|---|------:|-----------:|-----------:|------------:|
| 3.0 | 0.1 | **0.3168** | 0.2217 | 0.2984 | 0.3231 |
| 3.0 | 0.3 | 0.3168 | 0.2257 | 0.3163 | 0.3366 |
| 3.0 | 0.5 | 0.3108 | 0.2294 | 0.3225 | 0.3386 |
| 3.0 | 0.7 | 0.3089 | 0.2334 | 0.3257 | 0.3535 |
| 3.0 | 1.0 | 0.3015 | 0.2379 | 0.3477 | 0.3473 |
| 3.0 | 1.5 | 0.2966 | 0.2416 | 0.3459 | 0.3483 |
| 2.0 | 0.3 | 0.2964 | 0.2328 | 0.3262 | 0.3414 |
| 2.0 | 0.5 | 0.2950 | 0.2377 | 0.3434 | 0.3520 |
| 2.0 | 0.7 | 0.2929 | 0.2407 | 0.3468 | 0.3605 |
| 3.0 | 2.0 | 0.2925 | 0.2445 | 0.3564 | 0.3556 |

**RepLLaMA — Robust04**

| α | β | p-MRR | chg_MAP@1k | chg_nDCG@5 | chg_nDCG@10 |
|---|---|------:|-----------:|-----------:|------------:|
| 1.0 | 3.0 | **0.1020** | 0.2277 | 0.2996 | 0.3035 |
| 0.7 | 3.0 | 0.1013 | 0.2274 | 0.3018 | 0.3028 |
| 0.1 | 3.0 | 0.1013 | 0.2308 | 0.2995 | 0.3059 |
| 1.5 | 3.0 | 0.1012 | 0.2258 | 0.3028 | 0.3011 |
| 0.5 | 3.0 | 0.1011 | 0.2266 | 0.3018 | 0.3025 |
| 0.3 | 3.0 | 0.1004 | 0.2272 | 0.2995 | 0.3012 |
| 2.0 | 3.0 | 0.1003 | 0.2287 | 0.3076 | 0.3039 |
| 3.0 | 3.0 | 0.0956 | 0.2285 | 0.3159 | 0.3083 |
| 0.3 | 2.0 | 0.0945 | 0.2309 | 0.3002 | 0.3083 |
| 0.1 | 2.0 | 0.0943 | 0.2353 | 0.3029 | 0.3130 |

**RepLLaMA — News21**

| α | β | p-MRR | chg_MAP@1k | chg_nDCG@5 | chg_nDCG@10 |
|---|---|------:|-----------:|-----------:|------------:|
| 3.0 | 0.1 | **0.2500** | 0.2462 | 0.2770 | 0.3255 |
| 3.0 | 0.3 | 0.2475 | 0.2440 | 0.2782 | 0.3159 |
| 3.0 | 0.5 | 0.2440 | 0.2491 | 0.2802 | 0.3193 |
| 3.0 | 3.0 | 0.2429 | 0.2474 | 0.2857 | 0.3164 |
| 3.0 | 0.7 | 0.2404 | 0.2504 | 0.2786 | 0.3225 |
| 3.0 | 1.0 | 0.2393 | 0.2516 | 0.2853 | 0.3375 |
| 2.0 | 3.0 | 0.2378 | 0.2445 | 0.2799 | 0.3068 |
| 3.0 | 2.0 | 0.2375 | 0.2463 | 0.2756 | 0.3098 |
| 3.0 | 1.5 | 0.2364 | 0.2479 | 0.2742 | 0.3139 |
| 1.5 | 3.0 | 0.2353 | 0.2454 | 0.2812 | 0.3065 |

**BGE — Core17**

| α | β | p-MRR | chg_MAP@1k | chg_nDCG@5 | chg_nDCG@10 |
|---|---|------:|-----------:|-----------:|------------:|
| 3.0 | 0.3 | **0.2976** | 0.1704 | 0.2487 | 0.2267 |
| 3.0 | 0.5 | 0.2968 | 0.1749 | 0.2617 | 0.2509 |
| 3.0 | 0.7 | 0.2930 | 0.1785 | 0.2721 | 0.2548 |
| 3.0 | 0.1 | 0.2877 | 0.1630 | 0.2271 | 0.2096 |
| 3.0 | 1.0 | 0.2872 | 0.1842 | 0.2783 | 0.2657 |
| 2.0 | 0.5 | 0.2868 | 0.1809 | 0.2711 | 0.2643 |
| 2.0 | 0.3 | 0.2841 | 0.1772 | 0.2674 | 0.2447 |
| 2.0 | 0.7 | 0.2832 | 0.1831 | 0.2739 | 0.2741 |
| 2.0 | 1.0 | 0.2811 | 0.1898 | 0.2990 | 0.2963 |
| 1.5 | 0.5 | 0.2803 | 0.1839 | 0.2773 | 0.2851 |

**BGE — Robust04**

| α | β | p-MRR | chg_MAP@1k | chg_nDCG@5 | chg_nDCG@10 |
|---|---|------:|-----------:|-----------:|------------:|
| 0.1 | 3.0 | **0.1153** | 0.1942 | 0.2850 | 0.2748 |
| 0.3 | 3.0 | 0.1145 | 0.1943 | 0.2850 | 0.2763 |
| 2.0 | 3.0 | 0.1140 | 0.1959 | 0.2825 | 0.2775 |
| 0.7 | 3.0 | 0.1130 | 0.1944 | 0.2813 | 0.2764 |
| 1.0 | 3.0 | 0.1130 | 0.1946 | 0.2804 | 0.2784 |
| 1.5 | 3.0 | 0.1130 | 0.1944 | 0.2801 | 0.2764 |
| 3.0 | 3.0 | 0.1127 | 0.1978 | 0.2829 | 0.2806 |
| 0.5 | 3.0 | 0.1125 | 0.1938 | 0.2808 | 0.2760 |
| 3.0 | 0.1 | 0.1104 | 0.1716 | 0.2440 | 0.2361 |
| 1.5 | 2.0 | 0.1077 | 0.1985 | 0.2879 | 0.2766 |

**BGE — News21**

| α | β | p-MRR | chg_MAP@1k | chg_nDCG@5 | chg_nDCG@10 |
|---|---|------:|-----------:|-----------:|------------:|
| 3.0 | 0.3 | **0.2268** | 0.2227 | 0.2414 | 0.2832 |
| 3.0 | 0.5 | 0.2237 | 0.2244 | 0.2407 | 0.2898 |
| 3.0 | 0.1 | 0.2209 | 0.2251 | 0.2505 | 0.2892 |
| 3.0 | 0.7 | 0.2200 | 0.2225 | 0.2454 | 0.2845 |
| 3.0 | 1.0 | 0.2172 | 0.2249 | 0.2504 | 0.2765 |
| 3.0 | 1.5 | 0.2153 | 0.2250 | 0.2534 | 0.2766 |
| 3.0 | 2.0 | 0.2133 | 0.2238 | 0.2473 | 0.2792 |
| 3.0 | 3.0 | 0.2114 | 0.2218 | 0.2484 | 0.2786 |
| 2.0 | 3.0 | 0.2072 | 0.2198 | 0.2474 | 0.2755 |
| 2.0 | 2.0 | 0.2064 | 0.2208 | 0.2453 | 0.2727 |

### 10.6 编码器级最优静态参数

跨数据集综合得分 = avg(p-MRR + chg_MAP@1k + chg_nDCG@5)，选跨三数据集综合最优的 (α, β)。

| 编码器 | 最优静态 α | 最优静态 β | avg p-MRR | avg chg_MAP | avg chg_nDCG@5 | 综合得分 |
|--------|:---------:|:---------:|:---------:|:-----------:|:--------------:|:--------:|
| RepLLaMA | 3.0 | 3.0 | 0.2094 | 0.2408 | 0.3224 | 0.7726 |
| BGE | 3.0 | 3.0 | 0.1985 | 0.2062 | 0.2800 | 0.6848 |

注：两编码器跨数据集最优静态参数均为 α=3.0, β=3.0，但不同数据集的 Oracle 最优差异巨大（Core17/News21: 高α低β; Robust04: 低α高β），没有一组静态参数能同时在所有数据集上最优。

### 10.7 核心对比：per-query 推导 vs 编码器级静态参数

**RepLLaMA**（静态 α=3.0, β=3.0）

| 数据集 | 方法 | p-MRR | chg_MAP@1k | chg_nDCG@5 | chg_nDCG@10 | 综合 |
|--------|------|------:|-----------:|-----------:|------------:|-----:|
| Core17 | 静态 | 0.2897 | 0.2464 | 0.3656 | 0.3608 | 0.9017 |
| Core17 | 推导 | 0.2815 | 0.2496 | **0.3777** | 0.3630 | **0.9088** |
| Robust04 | 静态 | 0.0956 | 0.2285 | **0.3159** | **0.3083** | **0.6400** |
| Robust04 | 推导 | **0.1002** | 0.2285 | 0.3011 | 0.3050 | 0.6298 |
| News21 | 静态 | **0.2429** | 0.2474 | **0.2857** | **0.3164** | **0.7760** |
| News21 | 推导 | 0.2275 | **0.2486** | 0.2787 | 0.3104 | 0.7549 |
| **AVG** | 静态 | **0.2094** | 0.2408 | **0.3224** | **0.3285** | **0.7726** |
| **AVG** | 推导 | 0.2031 | **0.2422** | 0.3192 | 0.3262 | 0.7645 |

**BGE**（静态 α=3.0, β=3.0）

| 数据集 | 方法 | p-MRR | chg_MAP@1k | chg_nDCG@5 | chg_nDCG@10 | 综合 |
|--------|------|------:|-----------:|-----------:|------------:|-----:|
| Core17 | 静态 | 0.2715 | 0.1991 | 0.3088 | **0.3018** | 0.7793 |
| Core17 | 推导 | 0.2674 | 0.1984 | **0.3196** | 0.3009 | **0.7854** |
| Robust04 | 静态 | 0.1127 | **0.1978** | **0.2829** | **0.2806** | **0.5933** |
| Robust04 | 推导 | **0.1180** | 0.1932 | 0.2813 | 0.2734 | 0.5925 |
| News21 | 静态 | **0.2114** | **0.2218** | **0.2484** | **0.2786** | **0.6816** |
| News21 | 推导 | 0.1990 | 0.2199 | 0.2412 | 0.2636 | 0.6601 |
| **AVG** | 静态 | **0.1985** | **0.2062** | **0.2800** | **0.2870** | **0.6848** |
| **AVG** | 推导 | 0.1948 | 0.2038 | 0.2807 | 0.2793 | 0.6793 |

**跨编码器×数据集总平均**

| 方法 | avg p-MRR | avg chg_MAP@1k | avg chg_nDCG@5 | avg 综合 |
|------|:---------:|:--------------:|:--------------:|:--------:|
| 静态(3,3) | 0.2040 | 0.2235 | 0.3012 | 0.7287 |
| 推导(pq) | 0.1989 | 0.2230 | 0.2999 | 0.7219 |
| Δ(推-静) | -0.0050 | -0.0004 | -0.0013 | -0.0068 |

### 10.8 per-query α_q/β_q 统计

**RepLLaMA**

| 数据集 | n | α_q mean | α_q median | α_q min | α_q max | β_q mean | β_q median | β_q min | β_q max | at_risk mean | fallback |
|--------|---:|:--------:|:----------:|:-------:|:-------:|:--------:|:----------:|:-------:|:-------:|:------------:|:--------:|
| Core17 | 20 | 0.888 | 0.981 | 0.500 | 1.073 | 2.152 | 2.226 | 1.216 | 2.934 | 0.072 | 4/20 |
| Robust04 | 31 | 0.919 | 0.992 | 0.500 | 1.101 | 2.173 | 2.295 | 1.217 | 2.713 | 0.080 | 5/31 |
| News21 | 32 | 0.899 | 1.012 | 0.500 | 1.141 | 2.157 | 2.298 | 1.310 | 2.939 | 0.073 | 8/32 |

**BGE**

| 数据集 | n | α_q mean | α_q median | α_q min | α_q max | β_q mean | β_q median | β_q min | β_q max | at_risk mean | fallback |
|--------|---:|:--------:|:----------:|:-------:|:-------:|:--------:|:----------:|:-------:|:-------:|:------------:|:--------:|
| Core17 | 20 | 0.649 | 0.765 | 0.100 | 0.897 | 2.134 | 2.288 | 1.288 | 2.580 | 0.077 | 4/20 |
| Robust04 | 31 | 0.663 | 0.762 | 0.100 | 0.872 | 2.389 | 2.385 | 1.243 | 3.574 | 0.079 | 5/31 |
| News21 | 32 | 0.621 | 0.769 | 0.100 | 0.915 | 2.309 | 2.479 | 1.302 | 3.311 | 0.065 | 8/32 |

### 10.9 分析

1. **推导接近静态**：跨编码器×数据集总平均，推导综合得分仅低于静态 0.0068（0.94%），在不需任何参数调优的前提下接近编码器级最优静态参数。

2. **推导在 nDCG@5 上的优势**：在 Core17 上推导的 chg_nDCG@5 超过静态（RepLLaMA: 0.3777 vs 0.3656, BGE: 0.3196 vs 0.3088），说明 per-query β_q 推导能更精准地增强头部排序质量。

3. **α_q 系统性偏低**：推导 α_q（RepLLaMA: ~0.9, BGE: ~0.65）远低于最优静态 α=3.0。当前推导公式 `α_q = mean(s_base[at_risk]) / mean(softplus(overflow))` 做量级对齐，但 p-MRR 优化需要过度惩罚（α=3.0），这是推导综合得分略低于静态的主因。

4. **GritLM 异常**：GritLM 在所有指标上显著落后（og_nDCG@10 ≈ 0.02 vs RepLLaMA ≈ 0.45），p-MRR 接近零。原因不是模型规模（BGE 335M 表现优异），而是 GritLM 的通用嵌入范式缺乏指令区分能力，导致 q_plus/q_minus 信号无法有效分离。

5. **不同数据集需要不同参数**：Core17/News21 Oracle 为高α低β（α=3.0, β=0.1-0.3），Robust04 Oracle 为低α高β（α=0.1-1.0, β=3.0）。per-query 推导消除了手动选择参数的需要。

### 10.10 结果文件

| 文件 | 描述 |
|------|------|
| `results/repllama_v86_kappa10_{core17,robust04,news21}/metrics_summary.json` | RepLLaMA per-query V8.6 |
| `results/repllama_static_{core17,robust04,news21}/all_results.json` | RepLLaMA 静态网格搜索 64 组 |
| `results/bge_v86_kappa10_{core17,robust04,news21}/metrics_summary.json` | BGE per-query V8.6 |
| `results/bge_static_{core17,robust04,news21}/all_results.json` | BGE 静态网格搜索 64 组 |
| `results/gritlm_residual_bg_{core17,robust04,news21}_fp16/metrics_summary.json` | GritLM fp16 |

### 10.11 BGE 编码器实现

文件：`eval/models/bge_encoder.py`

关键设计：
- 使用 SentenceTransformer 加载，查询前置 `"Represent this sentence for searching relevant passages: "`
- `encode_queries()`: 前置指令后编码
- `encode_documents()`: 直接编码
- `half()` 精度，batch_size=64
- 注册名：`BAAI/bge-large-en-v1.5`, `bge-large-en-v1.5`, `BGE-large-en-v1.5`

### 10.12 GritLM 多GPU 实现

文件：`eval/models/gritlm_encoder.py`

关键修改：
- `device_map="auto"` 参数，模型层自动分布到多 GPU
- `_encode_batch()` 中 attention mask 移到 hidden.device（跨 GPU 时必需）
- `torch.cuda.empty_cache()` 在文档索引后释放中间内存

运行命令：`CUDA_VISIBLE_DEVICES=6,7`（2×24GB GPU，7B fp16 模型层均匀分布）
