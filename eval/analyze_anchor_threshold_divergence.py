"""
Per-query analysis: Safe-anchor threshold vs V5 cos-similarity threshold on Core17.

Reads the existing debug_anchor_logs.json and computes:
1. V5 threshold = cos_qbase_qneg + delta (delta=0.02)
2. Safe-anchor threshold = max(tau_anchor, cos_qbase_qneg) + anchor_delta (anchor_delta=-0.05)
3. Per-query divergence and impact on penalized docs
4. Root cause classification
"""
import json
import os

DEBUG_LOG = "/home/luwa/Documents/DSCLR/results/safe_anchor/Core17InstructionRetrieval/debug_anchor_logs.json"
V5_DELTA = 0.02
ANCHOR_DELTA = -0.05

def main():
    with open(DEBUG_LOG, "r") as f:
        records = json.load(f)

    print("=" * 120)
    print("PER-QUERY ANALYSIS: Safe-Anchor vs V5 Cos-Similarity Threshold (Core17)")
    print("=" * 120)
    print(f"V5 threshold:      tau_v5 = cos_qbase_qneg + {V5_DELTA}")
    print(f"Safe-anchor:       tau_sa = max(tau_anchor, cos_qbase_qneg) + ({ANCHOR_DELTA})")
    print()

    # Build per-query table
    print(f"{'Query':<16} {'q_neg':<35} {'tau_anchor':>10} {'cos':>8} {'tau_v5':>8} {'tau_sa':>8} {'diff(sa-v5)':>12} {'#pen_v5':>8} {'#pen_sa':>8} {'#pen_delta':>10} {'s_neg_max':>10}")
    print("-" * 145)

    total_pen_v5 = 0
    total_pen_sa = 0
    over_penalized_queries = []
    under_penalized_queries = []

    for r in records:
        qid = r["query_id"]
        q_neg = r["q_neg"][:33]
        tau_anchor = r["tau_anchor"]
        cos = r["cos_qbase_qneg"]
        s_neg_max = r["candidate_s_neg_max"] if r["candidate_s_neg_max"] is not None else 0.0

        tau_v5 = cos + V5_DELTA
        tau_sa = r["threshold_base_used"]  # already computed as max(tau_anchor, cos) + anchor_delta

        diff = tau_sa - tau_v5

        # Estimate num_penalized under V5 threshold
        # We know s_neg_max and s_neg_min, and num_penalized under tau_sa
        # For V5, docs with s_neg > tau_v5 are penalized
        # We don't have the full distribution, but we can estimate
        num_pen_sa = r["num_penalized_docs"]
        # If tau_v5 > tau_sa, V5 penalizes fewer; if tau_v5 < tau_sa, V5 penalizes more
        # Rough estimate: if tau_v5 > s_neg_max, V5 penalizes 0
        if s_neg_max < tau_v5:
            num_pen_v5_est = 0
        elif s_neg_max < tau_sa:
            # V5 threshold is lower than SA, but both are below s_neg_max
            # V5 would penalize MORE (lower threshold = more docs above it)
            num_pen_v5_est = num_pen_sa  # at least as many
        else:
            num_pen_v5_est = num_pen_sa

        # More precise: if tau_v5 > tau_sa, V5 penalizes fewer docs
        # If tau_v5 < tau_sa, V5 penalizes more docs
        if tau_v5 > tau_sa:
            # V5 threshold higher → fewer penalized
            if s_neg_max < tau_v5:
                num_pen_v5_est = 0
            else:
                # Some docs between tau_sa and tau_v5 would be saved
                num_pen_v5_est = max(0, num_pen_sa - 1)  # rough
        elif tau_v5 < tau_sa:
            # V5 threshold lower → more penalized
            num_pen_v5_est = num_pen_sa
        else:
            num_pen_v5_est = num_pen_sa

        total_pen_v5 += num_pen_v5_est
        total_pen_sa += num_pen_sa

        pen_delta = num_pen_sa - num_pen_v5_est

        flag = ""
        if abs(diff) > 0.03 and pen_delta > 0:
            flag = " ← OVER-PENALIZED"
            over_penalized_queries.append((qid, q_neg, diff, num_pen_sa, tau_anchor, cos, tau_v5, tau_sa, s_neg_max))
        elif abs(diff) > 0.03 and pen_delta < 0:
            flag = " ← UNDER-PENALIZED"
            under_penalized_queries.append((qid, q_neg, diff, num_pen_sa, tau_anchor, cos, tau_v5, tau_sa, s_neg_max))

        print(f"{qid:<16} {q_neg:<35} {tau_anchor:>10.4f} {cos:>8.4f} {tau_v5:>8.4f} {tau_sa:>8.4f} {diff:>+12.4f} {num_pen_v5_est:>8} {num_pen_sa:>8} {pen_delta:>+10} {s_neg_max:>10.4f}{flag}")

    print("-" * 145)
    print(f"{'TOTAL':<16} {'':<35} {'':>10} {'':>8} {'':>8} {'':>8} {'':>12} {total_pen_v5:>8} {total_pen_sa:>8} {total_pen_sa - total_pen_v5:>+10}")
    print()

    # === Analysis: why does tau_anchor diverge from cos? ===
    print("=" * 120)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 120)
    print()

    # Classify queries by whether tau_anchor > cos or < cos
    anchor_higher = []
    anchor_lower = []
    for r in records:
        diff = r["tau_anchor"] - r["cos_qbase_qneg"]
        if diff > 0.01:
            anchor_higher.append((r["query_id"], r["q_neg"][:40], r["tau_anchor"], r["cos_qbase_qneg"], diff))
        elif diff < -0.01:
            anchor_lower.append((r["query_id"], r["q_neg"][:40], r["tau_anchor"], r["cos_qbase_qneg"], diff))

    print(f"Queries where tau_anchor >> cos (anchor docs too similar to q_neg): {len(anchor_higher)}")
    for qid, qneg, ta, c, d in anchor_higher:
        print(f"  {qid}: q_neg='{qneg}' tau_anchor={ta:.4f} cos={c:.4f} diff={d:+.4f}")
    print()

    print(f"Queries where tau_anchor << cos (anchor docs too dissimilar to q_neg): {len(anchor_lower)}")
    for qid, qneg, ta, c, d in anchor_lower:
        print(f"  {qid}: q_neg='{qneg}' tau_anchor={ta:.4f} cos={c:.4f} diff={d:+.4f}")
    print()

    # === Key insight: the mix=max mode + anchor_delta=-0.05 ===
    print("=" * 120)
    print("IMPACT OF mix=max + anchor_delta=-0.05")
    print("=" * 120)
    print()
    print("With mix=max: tau_sa = max(tau_anchor, cos) - 0.05")
    print("This means:")
    print("  - When tau_anchor > cos: tau_sa = tau_anchor - 0.05 (anchor dominates)")
    print("  - When tau_anchor < cos: tau_sa = cos - 0.05 (cos dominates, anchor ignored)")
    print()
    print("The -0.05 offset LOWERED the threshold vs V5 (which uses cos + 0.02):")
    print(f"  V5:      tau = cos + 0.02")
    print(f"  Safe-anchor (cos dominates): tau = cos - 0.05  → 0.07 LOWER than V5!")
    print(f"  Safe-anchor (anchor dominates): tau = tau_anchor - 0.05")
    print()

    # Count which mode dominates
    cos_dominates = sum(1 for r in records if r["tau_anchor"] <= r["cos_qbase_qneg"])
    anchor_dominates = sum(1 for r in records if r["tau_anchor"] > r["cos_qbase_qneg"])
    print(f"  cos dominates (tau_anchor <= cos): {cos_dominates}/{len(records)} queries")
    print(f"  anchor dominates (tau_anchor > cos): {anchor_dominates}/{len(records)} queries")
    print()

    # For cos-dominant queries, the threshold is cos - 0.05, which is 0.07 lower than V5's cos + 0.02
    # This causes MASSIVE over-penalization
    print("CRITICAL FINDING: For cos-dominant queries, safe-anchor threshold is 0.07 LOWER than V5,")
    print("causing massive over-penalization of candidate documents!")
    print()

    # Show the most over-penalized queries
    print("=" * 120)
    print("MOST PROBLEMATIC QUERIES (over-penalized by safe-anchor)")
    print("=" * 120)
    print()
    sorted_by_pen = sorted(records, key=lambda r: r["num_penalized_docs"], reverse=True)
    print(f"{'Query':<16} {'q_neg':<35} {'#pen':>6} {'tau_anchor':>10} {'cos':>8} {'tau_v5':>8} {'tau_sa':>8} {'s_neg_max':>10} {'gap_v5':>10}")
    print("-" * 120)
    for r in sorted_by_pen[:10]:
        tau_v5 = r["cos_qbase_qneg"] + V5_DELTA
        tau_sa = r["threshold_base_used"]
        gap_v5 = tau_sa - tau_v5
        print(f"{r['query_id']:<16} {r['q_neg'][:33]:<35} {r['num_penalized_docs']:>6} {r['tau_anchor']:>10.4f} {r['cos_qbase_qneg']:>8.4f} {tau_v5:>8.4f} {tau_sa:>8.4f} {r['candidate_s_neg_max'] or 0:>10.4f} {gap_v5:>+10.4f}")

    print()

    # === Propose solution: what anchor_delta would match V5? ===
    print("=" * 120)
    print("SOLUTION ANALYSIS: What anchor_delta would make safe-anchor match V5?")
    print("=" * 120)
    print()
    print("Target: tau_sa ≈ tau_v5 = cos + 0.02")
    print("Current: tau_sa = max(tau_anchor, cos) + anchor_delta")
    print()

    # For each query, compute the ideal anchor_delta
    ideal_deltas = []
    for r in records:
        tau_v5 = r["cos_qbase_qneg"] + V5_DELTA
        tau_max = max(r["tau_anchor"], r["cos_qbase_qneg"])
        ideal_delta = tau_v5 - tau_max
        ideal_deltas.append((r["query_id"], r["q_neg"][:30], r["tau_anchor"], r["cos_qbase_qneg"], tau_max, tau_v5, ideal_delta))

    print(f"{'Query':<16} {'q_neg':<32} {'tau_anchor':>10} {'cos':>8} {'max':>8} {'tau_v5':>8} {'ideal_delta':>12}")
    print("-" * 100)
    for qid, qneg, ta, c, mx, tv5, idlt in ideal_deltas:
        print(f"{qid:<16} {qneg:<32} {ta:>10.4f} {c:>8.4f} {mx:>8.4f} {tv5:>8.4f} {idlt:>+12.4f}")

    import statistics
    deltas = [x[6] for x in ideal_deltas]
    print()
    print(f"Ideal anchor_delta statistics:")
    print(f"  mean   = {statistics.mean(deltas):+.4f}")
    print(f"  median = {statistics.median(deltas):+.4f}")
    print(f"  stdev  = {statistics.stdev(deltas):.4f}")
    print(f"  min    = {min(deltas):+.4f}")
    print(f"  max    = {max(deltas):+.4f}")
    print()

    # Check if a single anchor_delta can work
    print("Can a single anchor_delta make tau_sa = tau_v5 for all queries?")
    print("  Only if max(tau_anchor, cos) - cos is constant across queries (it's not).")
    print()

    # Alternative: what if we use mix=replace (pure anchor) and find the right delta?
    print("=" * 120)
    print("ALTERNATIVE: mix=replace (pure anchor threshold) + anchor_delta")
    print("=" * 120)
    print()
    print("If tau_sa = tau_anchor + anchor_delta, and target = cos + 0.02:")
    print("Then anchor_delta = cos + 0.02 - tau_anchor")
    print()

    replace_deltas = []
    for r in records:
        tau_v5 = r["cos_qbase_qneg"] + V5_DELTA
        ideal_delta = tau_v5 - r["tau_anchor"]
        replace_deltas.append((r["query_id"], r["q_neg"][:30], r["tau_anchor"], r["cos_qbase_qneg"], tau_v5, ideal_delta))

    print(f"{'Query':<16} {'q_neg':<32} {'tau_anchor':>10} {'cos':>8} {'tau_v5':>8} {'ideal_delta':>12}")
    print("-" * 100)
    for qid, qneg, ta, c, tv5, idlt in replace_deltas:
        print(f"{qid:<16} {qneg:<32} {ta:>10.4f} {c:>8.4f} {tv5:>8.4f} {idlt:>+12.4f}")

    deltas_replace = [x[5] for x in replace_deltas]
    print()
    print(f"Ideal anchor_delta (replace mode) statistics:")
    print(f"  mean   = {statistics.mean(deltas_replace):+.4f}")
    print(f"  median = {statistics.median(deltas_replace):+.4f}")
    print(f"  stdev  = {statistics.stdev(deltas_replace):.4f}")
    print(f"  min    = {min(deltas_replace):+.4f}")
    print(f"  max    = {max(deltas_replace):+.4f}")
    print()

    # === Root cause: why tau_anchor != cos ===
    print("=" * 120)
    print("FUNDAMENTAL ROOT CAUSE")
    print("=" * 120)
    print()
    print("1. SPACE MISMATCH:")
    print("   - cos_qbase_qneg is in QUERY-QUERY similarity space")
    print("   - tau_anchor is in QUERY-DOCUMENT similarity space")
    print("   - QQ similarity is naturally HIGHER (shared vocabulary, same encoding space)")
    print("   - QD similarity is naturally LOWER (different text distributions)")
    print()
    print("2. ANCHOR CONTENT EFFECT:")
    print("   - When q_neg is a narrow term (e.g. 'leukemia', 'rubber', 'testosterone'),")
    print("     anchors that AVOID this term have very LOW S_neg → tau_anchor << cos")
    print("   - When q_neg is broad/semantic (e.g. 'American politics, war, churches'),")
    print("     anchors that share q_base's topic have HIGH S_neg → tau_anchor ≈ or > cos")
    print()
    print("3. MIX=MAX + ANCHOR_DELTA=-0.05 AMPLIFIES THE ERROR:")
    print("   - For cos-dominant queries: tau = cos - 0.05 (0.07 lower than V5's cos + 0.02)")
    print("     → massive over-penalization")
    print("   - For anchor-dominant queries: tau = tau_anchor - 0.05 (unpredictable)")
    print()

    # === Proposed fix: per-query adaptive delta ===
    print("=" * 120)
    print("PROPOSED FIXES")
    print("=" * 120)
    print()
    print("Fix 1: Set anchor_delta = +0.02 (match V5's delta) instead of -0.05")
    print("  tau_sa = max(tau_anchor, cos) + 0.02")
    print("  For cos-dominant: tau = cos + 0.02 = tau_v5 (EXACT MATCH!)")
    print("  For anchor-dominant: tau = tau_anchor + 0.02 (reasonable)")
    print()

    # Simulate Fix 1
    print("  Simulation of Fix 1 (anchor_delta=+0.02, mix=max):")
    print(f"  {'Query':<16} {'tau_v5':>8} {'tau_sa_fix1':>12} {'diff':>8} {'#pen_sa_old':>12} {'#pen_sa_fix1_est':>16}")
    print("  " + "-" * 80)
    for r in records:
        tau_v5 = r["cos_qbase_qneg"] + V5_DELTA
        tau_sa_fix1 = max(r["tau_anchor"], r["cos_qbase_qneg"]) + 0.02
        diff = tau_sa_fix1 - tau_v5
        s_neg_max = r["candidate_s_neg_max"] or 0.0
        # Estimate penalized docs under fix1
        if s_neg_max < tau_sa_fix1:
            pen_fix1 = 0
        else:
            pen_fix1 = r["num_penalized_docs"]  # conservative
        print(f"  {r['query_id']:<16} {tau_v5:>8.4f} {tau_sa_fix1:>12.4f} {diff:>+8.4f} {r['num_penalized_docs']:>12} {pen_fix1:>16}")
    print()

    print("Fix 2: Use mix=mean instead of mix=max")
    print("  tau_sa = 0.5*(tau_anchor + cos) + anchor_delta")
    print("  This smooths out the divergence, but doesn't match V5 exactly.")
    print()

    print("Fix 3: Fine-tune anchor documents to have tau_anchor ≈ cos_qbase_qneg")
    print("  This requires the anchor docs to reference q_neg concepts at the right level.")
    print("  The 'boundary' anchors (safe_anchors_core17_boundary.json) attempt this.")
    print()

    # Check if boundary anchors would help
    boundary_path = "/home/luwa/Documents/DSCLR/dataset/FollowIR_test/safe_anchors/safe_anchors_core17_boundary.json"
    if os.path.exists(boundary_path):
        print("  Boundary anchors exist! These actively reference q_neg concepts.")
        print("  Running boundary anchors would test if fine-tuning helps.")


if __name__ == "__main__":
    main()
