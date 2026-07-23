"""
TRACE failure mode analysis: Why are MAP and nDCG low?

Analyzes per-query scoring dynamics and compares with V8.6 baseline.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from collections import defaultdict

DATASETS = ["Core17InstructionRetrieval", "Robust04InstructionRetrieval", "News21InstructionRetrieval"]
TRACE_DIR = "evaluation/trace_v2"
V86_DIR = "results/repllama_v86_kappa10"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def analyze_scoring_dynamics():
    """Analyze how p, g, h interact at the per-query level."""
    print("=" * 80)
    print("1. SCORING DYNAMICS ANALYSIS")
    print("=" * 80)
    
    for ds in DATASETS:
        stats = load_json(f"{TRACE_DIR}/{ds}/trace_per_query_stats.json")
        neg_stats = [s for s in stats if s.get("has_neg", False)]
        
        if not neg_stats:
            continue
        
        print(f"\n--- {ds} ({len(neg_stats)} queries with neg) ---")
        
        # Key dynamics
        p_means = [s["p_mean"] for s in neg_stats]
        h_means = [s["h_mean"] for s in neg_stats]
        g_means = [s["g_mean"] for s in neg_stats]
        r2s = [s["r_squared"] for s in neg_stats]
        above_ratios = [s["above_boundary_ratio"] for s in neg_stats]
        p_maxes = [s["p_max"] for s in neg_stats]
        h_maxes = [s["h_max"] for s in neg_stats]
        
        print(f"  Reward p(d):     mean={np.mean(p_means):.3f}, max_mean={max(p_means):.3f}")
        print(f"  Penalty h(d):    mean={np.mean(h_means):.3f}, max_mean={max(h_means):.3f}")
        print(f"  Gate g(d):       mean={np.mean(g_means):.3f} (1.0=no suppression)")
        print(f"  Net reward p*g:  mean={np.mean([p*g for p,g in zip(p_means, g_means)]):.3f}")
        print(f"  Net effect p*g-h: mean={np.mean([p*g - h for p,g,h in zip(p_means, g_means, h_means)]):.3f}")
        print(f"  Above boundary:  mean={np.mean(above_ratios):.1%}")
        print(f"  R² distribution:")
        
        r2_bins = {"<0.1": 0, "0.1-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, ">0.7": 0}
        for r2 in r2s:
            if r2 < 0.1: r2_bins["<0.1"] += 1
            elif r2 < 0.3: r2_bins["0.1-0.3"] += 1
            elif r2 < 0.5: r2_bins["0.3-0.5"] += 1
            elif r2 < 0.7: r2_bins["0.5-0.7"] += 1
            else: r2_bins[">0.7"] += 1
        print(f"    {r2_bins}")
        
        # Score magnitude analysis
        print(f"  Max reward p_max:  mean={np.mean(p_maxes):.1f}, range=[{min(p_maxes):.1f}, {max(p_maxes):.1f}]")
        print(f"  Max penalty h_max:  mean={np.mean(h_maxes):.1f}, range=[{min(h_maxes):.1f}, {max(h_maxes):.1f}]")
        print(f"  p_max/h_max ratio:  {np.mean([p/h for p,h in zip(p_maxes, h_maxes)]):.2f}")
        
        # R² vs scoring effect
        print(f"\n  R² vs net reward correlation:")
        net_effects = [p*g - h for p,g,h in zip(p_means, g_means, h_means)]
        corr = np.corrcoef(r2s, net_effects)[0,1]
        print(f"    corr(R², net_reward) = {corr:.3f}")


def analyze_per_query_ranking():
    """Compare TRACE vs V8.6 rankings per query."""
    print("\n" + "=" * 80)
    print("2. PER-QUERY RANKING COMPARISON (TRACE vs V8.6)")
    print("=" * 80)
    
    for ds in DATASETS:
        trace_changed = load_json(f"{TRACE_DIR}/{ds}/ranking_changed.json")
        v86_changed_path = f"{V86_DIR}/{ds.split('Instruction')[0].lower()}/ranking_changed.json"
        
        # Try different path patterns for V8.6
        for path_pattern in [
            f"{V86_DIR}/{ds.split('Instruction')[0].lower()}/ranking_changed.json",
            f"results/repllama_v86_kappa10_{ds.split('Instruction')[0].lower()}/ranking_changed.json",
        ]:
            if os.path.exists(path_pattern):
                v86_changed_path = path_pattern
                break
        
        if not os.path.exists(v86_changed_path):
            # Check the actual directory structure
            import glob
            matches = glob.glob(f"results/repllama_v86_kappa10*{ds.split('Instruction')[0].lower()}*/ranking_changed.json")
            if matches:
                v86_changed_path = matches[0]
            else:
                print(f"  V8.6 results not found for {ds}")
                continue
        
        v86_changed = load_json(v86_changed_path)
        trace_stats = load_json(f"{TRACE_DIR}/{ds}/trace_per_query_stats.json")
        
        # Build stat lookup
        stat_lookup = {s["qid"]: s for s in trace_stats}
        
        print(f"\n--- {ds} ---")
        
        # Compare top-5 overlap between TRACE and V8.6
        trace_better = 0
        v86_better = 0
        overlap_top5_list = []
        
        for qid in trace_changed:
            if qid not in v86_changed:
                continue
            
            # Get top-5 doc IDs
            trace_top5 = set(list(trace_changed[qid].keys())[:5])
            v86_top5 = set(list(v86_changed[qid].keys())[:5])
            overlap = len(trace_top5 & v86_top5)
            overlap_top5_list.append(overlap)
            
            # Compare top-1
            trace_top1 = list(trace_changed[qid].keys())[0]
            v86_top1 = list(v86_changed[qid].keys())[0]
            
            s = stat_lookup.get(qid, {})
            has_neg = s.get("has_neg", False)
            r2 = s.get("r_squared", 0)
            
        print(f"  Top-5 overlap (TRACE vs V8.6): mean={np.mean(overlap_top5_list):.2f}/5")


def analyze_penalty_magnitude():
    """Analyze whether the penalty h(d) is too aggressive."""
    print("\n" + "=" * 80)
    print("3. PENALTY MAGNITUDE ANALYSIS")
    print("=" * 80)
    
    for ds in DATASETS:
        stats = load_json(f"{TRACE_DIR}/{ds}/trace_per_query_stats.json")
        neg_stats = [s for s in stats if s.get("has_neg", False)]
        
        if not neg_stats:
            continue
        
        print(f"\n--- {ds} ---")
        
        # Compare effective reward vs penalty
        for s in neg_stats[:3]:  # Show first 3 queries as examples
            p, g, h = s["p_mean"], s["g_mean"], s["h_mean"]
            r2 = s["r_squared"]
            above = s["above_boundary_ratio"]
            print(f"  qid={s['qid'][:15]}: R²={r2:.3f}, above={above:.1%}")
            print(f"    p_mean={p:.3f}, g_mean={g:.3f}, h_mean={h:.3f}")
            print(f"    effective_reward=p*g={p*g:.3f}, net=p*g-h={p*g-h:.3f}")
            print(f"    z_full range implied by p_max={s['p_max']:.1f}, h_max={s['h_max']:.1f}")
            
            # The S_final = z_full + p*g - h
            # z_full is ~N(0,1) after robust standardization
            # p*g can be up to ~p_max*1.0 for low-h documents
            # h can be up to ~h_max for high-r documents
            # Net effect: documents with high residual (r > lambda) get:
            #   S_final ≈ z_full + p*exp(-h/tau) - h
            #   For large h: S_final ≈ z_full - h (gate kills reward)
            print(f"    For top-penalized doc: S_final ≈ z_full - {s['h_max']:.1f} (gate kills reward)")
            print(f"    For top-rewarded doc: S_final ≈ z_full + {s['p_max']:.1f} (no penalty)")
            print()


def diagnose_root_cause():
    """Summarize the root causes of low MAP/nDCG."""
    print("\n" + "=" * 80)
    print("4. ROOT CAUSE DIAGNOSIS")
    print("=" * 80)
    
    print("""
ROOT CAUSE 1: z-score space destroys score separation
-----------------------------------------------------
After robust standardization, z_full has median=0, MAD=1.
Original cosine scores have narrow range (e.g., 0.55-0.80), so:
  - z_full range ≈ [-3, +8] (very spread)
  - A small absolute score difference (0.01) becomes z ≈ 0.5-1.0
  - This amplifies noise in the original retrieval scores

V8.6 operates in ORIGINAL score space:
  S = S_base + β·S_req·safety - α·Softplus(S_neg - τ)
  Score differences are proportional to actual cosine similarities.

ROOT CAUSE 2: Penalty h(d) is too aggressive
---------------------------------------------
h(d) = [r(d) - λ]+ operates on normalized residuals.
  - r(d) has MAD-normalized scale, so r > 1 is common
  - λ=0.5 means ~36-40% of documents are penalized
  - h_max can reach 5-10, completely destroying those docs' scores
  - But some of those documents are actually relevant!

V8.6's Softplus(S_neg - τ) penalty:
  - Only triggers when S_neg exceeds a threshold τ
  - τ is derived from the candidate set distribution
  - Safety gate prevents penalty on documents where S_req is high

ROOT CAUSE 3: Gate g(d) kills reward for wrong documents
---------------------------------------------------------
g(d) = exp(-h(d)/τ_decay)
  - When h > 0 (38% of docs), g < 1, reducing reward p(d)
  - But many documents with high residual r(d) may still be relevant
  - The gate punishes them by reducing their positive signal

V8.6's safety gate:
  - safety = sigmoid((S_req - t_safety) * t_safety_slope)
  - Only gates on S_req (positive signal), not on S_neg
  - Documents with strong S_req keep their reward regardless of S_neg

ROOT CAUSE 4: Low R² for many queries → noisy residual
-------------------------------------------------------
Queries with R² < 0.1 (regression explains <10% of variance):
  - Residual r(d) ≈ z_neg (regression didn't help)
  - The penalty is essentially random noise
  - These queries would be better with pos_only mode

Improvement ideas:
1. R²-conditional gating: only apply regression penalty when R² > threshold
2. Scale-aware scoring: operate in original score space, not z-space
3. Adaptive λ: set λ based on R² (high R² → lower λ, more aggressive penalty)
4. Hybrid: use V8.6 scoring but replace τ derivation with regression-based τ
""")


def propose_improvements():
    """Propose concrete improvements based on analysis."""
    print("\n" + "=" * 80)
    print("5. PROPOSED IMPROVEMENTS")
    print("=" * 80)
    
    print("""
IMPROVEMENT A: Regression-informed boundary in original score space
--------------------------------------------------------------------
Instead of z-score TRACE, use Huber regression to set the S_neg boundary
in the ORIGINAL cosine score space:

  e(d) = S_neg(d) - â - b̂·S_pos(d)   (regression residual in raw space)
  τ_trace(d) = median(e) + MAD(e) * λ   (data-adaptive threshold)
  S_final = S_base + β·S_pos·safety - α·Softplus(e(d) - τ_trace)

This preserves V8.6's score-space advantages while using regression
to get a per-query adaptive boundary.

IMPROVEMENT B: R²-gated penalty
--------------------------------
Only apply the negative penalty when R² > R²_min (e.g., 0.3):
  if R² < R²_min:
      S_final = S_base + β·S_pos·safety   (pos_only fallback)
  else:
      S_final = S_base + β·S_pos·safety - α·Softplus(e - τ)

IMPROVEMENT C: Residual-aware safety gate
------------------------------------------
Use regression residual to modulate safety gate:
  safety_trace = safety * sigmoid(-γ·r(d))  (reduce safety for high residual)

This means documents with high regression residual (likely to violate
exclusion) get their reward suppressed even when S_req is high.
""")


if __name__ == "__main__":
    os.chdir("/home/luwa/Documents/DSCLR-remote")
    analyze_scoring_dynamics()
    analyze_per_query_ranking()
    analyze_penalty_magnitude()
    diagnose_root_cause()
    propose_improvements()
