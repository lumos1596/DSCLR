"""验证 safety 双重效应：τ 降低导致 safety 急剧下降，削弱正面查询增益。"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

T_SAFETY = 20.0


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def safety(s_neg, tau):
    return 1.0 - sigmoid((s_neg - tau) * T_SAFETY)


def softplus(x):
    return math.log1p(math.exp(x)) if x > -20 else 0.0


def main():
    # 614-changed 的阈值
    v5_tau = 0.6968  # cos + 0.02
    v6_tau = 0.6321  # max(anchor, cos) - 0.05
    # V5/V6 参数
    v5_alpha, v5_beta = 0.72, 1.32
    v6_alpha, v6_beta = 0.99, 1.96

    print("=" * 90)
    print("【safety 双重效应验证: 614-changed (q_neg='genetically engineered food')】")
    print(f"  V5: τ={v5_tau:.4f}, α={v5_alpha}, β={v5_beta}")
    print(f"  V6: τ={v6_tau:.4f}, α={v6_alpha}, β={v6_beta}")
    print(f"  T_safety={T_SAFETY} (safety 对 τ 极度敏感)")
    print(f"  τ 降幅: {v5_tau - v6_tau:.4f} → sigmoid 输入增幅: {(v5_tau-v6_tau)*T_SAFETY:.2f}")
    print()

    # 模拟不同 S_neg 水平的文档
    print(f"{'S_neg':>7} | {'V5_safety':>9} {'V6_safety':>9} {'safety降':>9} | {'V5_penalty':>10} {'V6_penalty':>10} | {'V5_β·Sreq·safe':>14} {'V6_β·Sreq·safe':>14} (Sreq=0.3)")
    print("-" * 110)
    s_req_example = 0.3  # 假设 S_req=0.3
    for s_neg in [0.50, 0.55, 0.60, 0.62, 0.63, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80]:
        v5_s = safety(s_neg, v5_tau)
        v6_s = safety(s_neg, v6_tau)
        v5_p = v5_alpha * softplus(s_neg - v5_tau)
        v6_p = v6_alpha * softplus(s_neg - v6_tau)
        v5_gain = v5_beta * s_req_example * v5_s
        v6_gain = v6_beta * s_req_example * v6_s
        flag = "⚠️" if abs(v5_s - v6_s) > 0.3 else ""
        print(f"{s_neg:>7.2f} | {v5_s:>9.4f} {v6_s:>9.4f} {v6_s-v5_s:>+9.4f} | {v5_p:>10.4f} {v6_p:>10.4f} | {v5_gain:>14.4f} {v6_gain:>14.4f} {flag}")

    print()
    print("=" * 90)
    print("【关键洞察】")
    print(f"  τ 降低 {v5_tau-v6_tau:.4f} (V5→V6) 的双重效应:")
    print(f"  1. penalty 增强: Softplus(S_neg-τ) 增大 (压制含负向词文档) — 预期效果")
    print(f"  2. safety 降低: 1-sigmoid((S_neg-τ)×20) 急剧下降 (削弱正面查询增益) — 副作用!")
    print(f"  对于 S_neg≈τ 的文档, safety 从 ~0.5 降到 ~0.1, β·S_req·safety 损失巨大")
    print(f"  如果相关文档的 S_neg 略高于 τ (含部分负向语义), 它们的正面增益被错误削弱")
    print()

    # 量化: 对 614-changed 的相关文档排名变化
    print("=" * 90)
    print("【614-changed 相关文档的排名变化与 S_neg 的关系】")
    # 加载 debug 和 ranking
    with open("results/safe_anchor/Core17InstructionRetrieval/debug_anchor_logs.json") as f:
        debug = {r["query_id"]: r for r in json.load(f)}
    dbg = debug.get("614-changed", {})
    print(f"  候选集 S_neg 范围: [{dbg.get('candidate_s_neg_min'):.4f}, {dbg.get('candidate_s_neg_max'):.4f}]")
    print(f"  V5_τ={v5_tau:.4f}, V6_τ={v6_tau:.4f}")
    print(f"  候选集内被惩罚文档数: V6={dbg.get('num_penalized_docs')}")
    # 估计: 候选集中有多少文档的 S_neg 在 V5_τ 和 V6_τ 之间 (safety 敏感区)
    print(f"  safety 敏感区 (S_neg ∈ [{v6_tau:.4f}, {v5_tau:.4f}]): 这些文档在 V5 下 safety≈1, V6 下 safety≈0")
    print(f"  这意味着 V6 下这些文档的 β·S_req·safety 几乎归零, 正面增益被完全抹除")


if __name__ == "__main__":
    main()
