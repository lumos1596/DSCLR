---
name: "dsclr-overview"
description: "DeIR-Dual V2 技术体系总览索引。提供四个核心 skill 的导航：参数推导、dual-queries 设计、技术细节、实验结果。Agent 可通过此索引快速定位所需信息的具体 skill。"
---

# DeIR-Dual V2 技术体系总览索引

本 skill 提供四个核心 skill 的导航和快速索引，帮助 Agent 根据任务需求快速定位到具体 skill。

## Skill 体系架构

```
dsclr-overview (总览索引)
    ├── dsclr-param-derivation (参数推导方案)
    ├── dual-queries-design (dual-queries 语义设计)
    ├── dsclr-tech-details (技术细节与公式)
    ├── dsclr-exp-results (实验结果速查)
    └── dsclr-residual-bg (V8.5 residual_bg 实验复现手册)
```

## 1. dsclr-param-derivation（参数推导方案）

**用途**：提供第一性原理参数推导方法，确定 DeIR-Dual V2 的核心参数 α/β/δ。

**核心内容**：
- V5 基础版（动态语义阈值 τ = Cos(Q_base, Q_neg) + δ）
- V6/V7 safe-anchor 扩展版（LLM 无辜文档锚点 + 覆盖率校正）
- V8 per-query 推理时推导版（测试时逐 query 动态计算 α/β）
- V8.3 编码噪声问题与解决方案（batch_size=1 确定性编码）
- V8.4 q_minus 语义质量实验（失败探索：具体化悖论）
- V8.5 cross-scale residual penalty（背景泄漏预期 + 残差惩罚，惩罚/safety 解耦）
- V8.6 残差 MAD 归一化 safety gate（2026-06-30，safety = 1 - sigmoid(R_neg/MAD × κ)，替代传统 τ safety gate）
- 跨系列编码器泛化（RepLLaMA / E5-Mistral / BGE）

**适用场景**：
- 需要推导或验证参数 α/β/δ
- 需要了解参数推导的物理意义和数学公式
- 需要为新编码器推导适配参数
- 需要了解编码噪声问题及其解决方案
- 需要了解 safety gate 与 q_minus 语义耦合问题

**调用方式**：
```
invoke Skill: "dsclr-param-derivation"
```

**关键文件位置**：
- 推导脚本：`eval/first_principles_params_safe_anchor.py`
- 评测脚本：`eval/experiment_safe_anchor_threshold.py`
- 结果目录：`results/safe_anchor_v8_*/`

---

## 2. dual-queries-design（Dual-Queries 语义设计）

**用途**：提供 q_plus/q_minus 的语义设计原则和最佳实践。

**核心内容**：
- q_plus 设计原则（明确增强目标、避免冗余）
- q_minus 设计原则（避免"与 query 相关但应排除"，改用"完全无关领域"）
- 具体化悖论问题（细化 q_minus → tau_anchor 上升 → safety gate 过度抑制）
- 编码噪声影响分析
- 实验验证的设计案例

**适用场景**：
- 设计或优化 q_plus/q_minus 文本
- 分析 q_minus 语义质量问题
- 诊断 safety gate 与 q_minus 语义耦合问题
- 需要了解 dual-queries 的设计最佳实践

**调用方式**：
```
invoke Skill: "dual-queries-design"
```

**关键文件位置**：
- dual_queries 文件：`dataset/FollowIR_test/dual_queries_v6/`
- refined 实验文件：`dataset/FollowIR_test/dual_queries_v6/dual_queries_v6_News21InstructionRetrieval_qminus_refined.jsonl`

---

## 3. dsclr-tech-details（技术细节与公式）

**用途**：提供 DeIR-Dual V2 的核心技术细节、公式和参数搜索策略。

**核心内容**：
- Scoring formula（S_final = S_base + α×penalty + β×S_req×safety）
- Safety gate 公式（safety = 1 - sigmoid((S_neg - τ) × t_safety））
- 最优参数配置（t_safety=20, anchor_delta=+0.02, beta_derive_mode=max_mean）
- 参数搜索策略（grid search vs first-principles derivation）
- 三数据集最优参数对比

**适用场景**：
- 需要查阅核心公式和技术细节
- 需要了解最优参数配置
- 需要了解参数搜索策略
- Writing paper 需要技术细节引用

**调用方式**：
```
invoke Skill: "dsclr-tech-details"
```

---

## 4. dsclr-exp-results（实验结果速查）

**用途**：提供所有基准数据集上的评测结果、参数对比和跨模型对比。

**核心内容**：
- Core17 / Robust04 / News21 三数据集完整评测结果
- target_avg = (Core17_cMAP + Robust04_cMAP + News21_cnDCG@5) / 3
- p-MRR（指令敏感度指标）
- V5/V6/V7/V8 各版本对比
- RepLLaMA / E5-Mistral / BGE 跨编码器对比
- 网格搜索最优 vs 推导参数对比

**适用场景**：
- 需要查询具体实验结果数值
- 需要对比不同版本/编码器的效果
- Writing paper 需要实验数据引用
- 需要计算 target_avg 或 p-MRR

**调用方式**：
```
invoke Skill: "dsclr-exp-results"
```

---

## 5. dsclr-residual-bg（V8.5 Cross-Scale Residual Penalty 实验复现手册）

**用途**：提供 V8.5 residual_bg 机制的完整复现配置，确保迁移后可复现。

**核心内容**：
- 核心公式（背景泄漏预期、残差提取、MAD 阈值、惩罚/safety 解耦）
- **V8.6 残差 MAD 归一化 safety gate**（2026-06-30 新增）：`safety = 1 - sigmoid(R_neg/MAD × κ)`，替代缺乏可解释性的传统 τ safety gate
- κ 参数扫描实验结果（κ=8~12 最优区间）
- 代码文件清单及关键函数
- 完整实验配置（环境、参数、命令行）
- Core17 实验结果（λ 扫描、κ 扫描、与 semantic 对比、per-query 参数）
- 迁移复现检查清单

**适用场景**：
- 迁移到新机器后复现 residual_bg 实验
- 查阅 V8.5 的具体配置和命令行
- 验证 residual_bg 效果是否一致
- 了解 residual_bg 与 semantic 模式的差异

**调用方式**：
```
invoke Skill: "dsclr-residual-bg"
```

**关键文件位置**：
- 主引擎：`eval/engine_deir_dual_v2.py`（boundary_mode=residual_bg）
- 残差计算：`eval/residual_boundary.py`
- 结果目录：`results/residual_bg_v85_*/`

---

## 快速决策指南

**问题类型 → 推荐 Skill**：

| 问题类型 | 推荐 Skill | 说明 |
|---------|-----------|------|
| "如何推导参数 α/β？" | dsclr-param-derivation | 第一性原理推导方法 |
| "最优参数是什么？" | dsclr-tech-details | 查阅最优配置 |
| "实验结果数值是多少？" | dsclr-exp-results | 速查具体数值 |
| "如何设计 q_plus/q_minus？" | dual-queries-design | 设计原则和案例 |
| "编码噪声怎么解决？" | dsclr-param-derivation (V8.3) | batch_size=1 方案 |
| "safety gate 有什么问题？" | dsclr-param-derivation (V8.4) + dual-queries-design | 语义耦合问题 |
| "residual penalty 机制？" | dsclr-param-derivation (V8.5) + dsclr-tech-details + dsclr-residual-bg | 背景泄漏预期+残差惩罚 |
| "safety gate 可解释性？" | dsclr-residual-bg (V8.6) | 残差 MAD 归一化 safety gate |
| "跨编码器泛化效果如何？" | dsclr-param-derivation + dsclr-exp-results | 推导方法 + 结果对比 |
| "target_avg 怎么计算？" | dsclr-exp-results | 定义和数值 |
| "如何复现 residual_bg 实验？" | dsclr-residual-bg | 完整配置和复现命令 |

---

## Skill 依赖关系

```
参数推导 (dsclr-param-derivation)
    ↓ (生成参数)
技术细节 (dsclr-tech-details)
    ↓ (应用公式)
Dual-Queries 设计 (dual-queries-design)
    ↓ (提供 q_plus/q_minus)
实验结果 (dsclr-exp-results)
    ← (验证效果)
```

**逻辑链**：
1. 参数推导 skill 提供推导方法 → 得到 α/β/δ
2. 技术细节 skill 应用公式 + 参数 → 得到 scoring function
3. Dual-Queries 设计 skill 优化 q_plus/q_minus → 改善语义质量
4. 实验结果 skill 验证效果 → 反馈到参数推导和 dual-queries 设计

---

## 典型工作流示例

### 工作流 1：参数推导与验证

```
1. invoke dsclr-param-derivation → 了解推导方法
2. 运行推导脚本 → 得到 α=0.74, β=2.55
3. invoke dsclr-tech-details → 查阅公式和最优配置
4. 运行评测脚本 → 得到实验结果
5. invoke dsclr-exp-results → 对比 baseline 和推导参数效果
```

### 工作流 2：Dual-Queries 优化

```
1. invoke dual-queries-design → 了解设计原则
2. 分析现有 q_minus 问题 → 发现过度泛化
3. 设计 refined q_minus → 细化否定实体
4. invoke dsclr-param-derivation (V8.3) → 使用 batch_size=1 编码
5. 运行评测 → invoke dsclr-exp-results → 分析效果变化
6. invoke dual-queries-design → 分析具体化悖论问题
```

### 工作流 3：跨编码器泛化

```
1. invoke dsclr-param-derivation → 了解跨编码器推导方法
2. 准备新编码器训练集 embeddings
3. 运行推导脚本 → 得到编码器适配参数
4. invoke dsclr-exp-results → 对比各编码器结果
5. invoke dsclr-tech-details → 验证公式适用性
```

---

## 注意事项

1. **Skill 调用优先级**：总览索引 → 具体 skill。先通过此索引确定目标 skill，再调用具体 skill。

2. **Skill 互补性**：四个 skill 内容互补，无重复。参数推导 focus 推导方法；技术细节 focus 公式和配置；dual-queries-design focus 语义设计；实验结果 focus 数据速查。

3. **语言一致性**：dsclr-param-derivation 和 dsclr-exp-results 为中文；dual-queries-design 和 dsclr-tech-details 为英文。调用时应注意语言匹配。

4. **版本追踪**：dsclr-param-derivation 包含完整的 V5→V8.5 演进链；dsclr-residual-bg 包含 V8.5+V8.6 safety gate 更新；dsclr-exp-results 包含各版本结果对比。

5. **失败探索记录**：V8.4（q_minus 语义质量实验）记录在 dsclr-param-derivation 和 dual-queries-design 中，是重要的负面实验结果。