"""
Generate Figure 3: Reward–Penalty Effect Decomposition.

Three panels showing how TRIX's reward and penalty components
promote relevant documents and suppress violating ones.

Panel A: Score contribution decomposition by document type
Panel B: Rank change vs residual exclusion evidence scatter
Panel C: Top-k entry/exit statistics

Usage:
  cd /home/luwa/Documents/DSCLR-remote && \
  /home/luwa/.conda/envs/dsclr/bin/python -m eval.generate_figure3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DATA_PATH = "/home/luwa/Documents/DSCLR-remote/results/figure3/figure3_reward_penalty_data.json"
OUTPUT_PATH = "/home/luwa/Documents/DSCLR-remote/paper/AuthorKit27/AuthorKit27/Figures/figure3.pdf"

# Color scheme
COLOR_SATISFYING = '#2196F3'   # Blue
COLOR_AFFECTED = '#F44336'     # Red
COLOR_OTHER = '#9E9E9E'        # Gray
COLOR_REWARD = '#4CAF50'       # Green
COLOR_PENALTY = '#FF5722'      # Orange-red
COLOR_BASELINE = '#78909C'     # Blue-gray

CATEGORY_LABELS = {
    'constraint_satisfying': 'Satisfying\n(og: rel, ch: rel)',
    'constraint_affected': 'Affected\n(og: rel, ch: not rel)',
    'other': 'Other\n(og: not rel)',
}

CATEGORY_SHORT = {
    'constraint_satisfying': 'Satisfying',
    'constraint_affected': 'Affected',
    'other': 'Other',
}


def load_data():
    with open(DATA_PATH, 'r') as f:
        return json.load(f)


def panel_a_score_decomposition(ax, docs):
    """Panel A: Score contribution decomposition by document type."""
    categories = ['constraint_satisfying', 'constraint_affected', 'other']

    # Compute mean contributions per category
    data = {}
    for cat in categories:
        cat_docs = [d for d in docs if d['category'] == cat]
        if not cat_docs:
            data[cat] = {'p_g': 0, 'neg_h': 0, 'delta': 0, 'n': 0}
            continue
        data[cat] = {
            'p_g': np.mean([d['p_g'] for d in cat_docs]),
            'neg_h': np.mean([d['neg_h'] for d in cat_docs]),
            'delta': np.mean([d['delta'] for d in cat_docs]),
            'n': len(cat_docs),
        }

    x = np.arange(len(categories))
    width = 0.6

    # Stacked bars: reward (positive) and penalty (negative)
    rewards = [data[cat]['p_g'] for cat in categories]
    penalties = [-data[cat]['neg_h'] for cat in categories]  # negative for display

    bars_reward = ax.bar(x, rewards, width, label='Reward ($p \\cdot g$)',
                         color=COLOR_REWARD, alpha=0.85, edgecolor='white', linewidth=0.5)
    bars_penalty = ax.bar(x, penalties, width, label='Penalty ($-h$)',
                          color=COLOR_PENALTY, alpha=0.85, edgecolor='white', linewidth=0.5)

    # Zero line
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')

    # Add total delta markers
    for i, cat in enumerate(categories):
        delta = data[cat]['delta']
        marker = '^' if delta >= 0 else 'v'
        ax.plot(i, delta, marker=marker, color='black', markersize=8, zorder=5)
        # Annotate
        offset = 0.02 if delta >= 0 else -0.04
        ax.annotate(f'$\\Delta$={delta:+.3f}',
                    xy=(i, delta), ha='center', va='bottom' if delta >= 0 else 'top',
                    fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_SHORT[cat] for cat in categories], fontsize=9)
    ax.set_ylabel('Mean score contribution', fontsize=10)
    ax.set_title('(a) Score contribution by document type', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add sample counts
    for i, cat in enumerate(categories):
        n = data[cat]['n']
        ax.annotate(f'n={n}', xy=(i, ax.get_ylim()[0]), ha='center', va='bottom',
                    fontsize=7, color='gray')


def panel_b_rank_change_scatter(ax, docs):
    """Panel B: Rank change vs residual exclusion evidence."""
    # Only plot docs with exclusion (h > 0) or interesting rank changes
    categories = ['constraint_satisfying', 'constraint_affected', 'other']
    colors = [COLOR_SATISFYING, COLOR_AFFECTED, COLOR_OTHER]

    for cat, color in zip(categories, colors):
        cat_docs = [d for d in docs if d['category'] == cat]
        if not cat_docs:
            continue

        h_vals = [d['h'] for d in cat_docs]
        rank_changes = [d['rank_change'] for d in cat_docs]

        alpha = 0.6 if cat == 'other' else 0.8
        size = 8 if cat == 'other' else 15
        zorder = 1 if cat == 'other' else 2
        label = f'{CATEGORY_SHORT[cat]} (n={len(cat_docs)})'

        ax.scatter(h_vals, rank_changes, c=color, alpha=alpha, s=size,
                   label=label, zorder=zorder, edgecolors='none')

    # Add trend lines for satisfying and affected
    for cat, color in [('constraint_satisfying', COLOR_SATISFYING),
                       ('constraint_affected', COLOR_AFFECTED)]:
        cat_docs = [d for d in docs if d['category'] == cat]
        if len(cat_docs) > 5:
            h_vals = np.array([d['h'] for d in cat_docs])
            rank_changes = np.array([d['rank_change'] for d in cat_docs])
            # Binned trend line
            if h_vals.max() > h_vals.min():
                n_bins = min(20, len(cat_docs) // 5)
                bins = np.linspace(h_vals.min(), h_vals.max(), n_bins + 1)
                bin_centers = []
                bin_means = []
                for j in range(n_bins):
                    mask = (h_vals >= bins[j]) & (h_vals < bins[j+1])
                    if mask.sum() > 2:
                        bin_centers.append((bins[j] + bins[j+1]) / 2)
                        bin_means.append(rank_changes[mask].mean())
                if bin_centers:
                    ax.plot(bin_centers, bin_means, color=color, linewidth=2,
                            linestyle='--', alpha=0.8, zorder=3)

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-', alpha=0.5)
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle=':', alpha=0.3)

    ax.set_xlabel('Exclusion evidence $h(d) = [r(d) - \\lambda]_+$', fontsize=10)
    ax.set_ylabel('Rank change (base $\\rightarrow$ TRIX)', fontsize=10)
    ax.set_title('(b) Rank change vs exclusion evidence', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9, markerscale=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate regions
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.text(xlim[0] + 0.02 * (xlim[1] - xlim[0]), ylim[1] * 0.9,
            'Promoted\n(reward effect)', fontsize=7, color=COLOR_REWARD,
            alpha=0.7, ha='left', va='top')
    ax.text(xlim[1] * 0.7, ylim[0] * 0.9,
            'Suppressed\n(penalty effect)', fontsize=7, color=COLOR_PENALTY,
            alpha=0.7, ha='center', va='bottom')


def panel_c_topk_entry_exit(ax, docs):
    """Panel C: Top-k entry/exit statistics."""
    top_k_values = [5, 10, 20, 50, 100]

    categories = ['constraint_satisfying', 'constraint_affected']
    cat_labels = ['Satisfying docs\nentered top-$k$',
                  'Affected docs\nexited top-$k$']
    cat_colors = [COLOR_SATISFYING, COLOR_AFFECTED]

    # Group docs by query
    query_docs = defaultdict(list)
    for d in docs:
        query_docs[d['qid']].append(d)

    for cat, label, color in zip(categories, cat_labels, cat_colors):
        entered_or_exited = []
        for k in top_k_values:
            count = 0
            total_possible = 0
            for qid, q_docs in query_docs.items():
                cat_docs = [d for d in q_docs if d['category'] == cat]
                if not cat_docs:
                    continue

                # Sort by base rank to find top-k baseline
                base_sorted = sorted(q_docs, key=lambda d: d['base_rank'])
                trace_sorted = sorted(q_docs, key=lambda d: d['trace_rank'])

                base_topk = set(d['doc_id'] for d in base_sorted[:k])
                trace_topk = set(d['doc_id'] for d in trace_sorted[:k])

                for d in cat_docs:
                    if cat == 'constraint_satisfying':
                        # Entered: not in base top-k but in trace top-k
                        if d['doc_id'] not in base_topk and d['doc_id'] in trace_topk:
                            count += 1
                        # Possible: not in base top-k
                        if d['doc_id'] not in base_topk:
                            total_possible += 1
                    else:  # constraint_affected
                        # Exited: in base top-k but not in trace top-k
                        if d['doc_id'] in base_topk and d['doc_id'] not in trace_topk:
                            count += 1
                        # Possible: in base top-k
                        if d['doc_id'] in base_topk:
                            total_possible += 1

            ratio = count / max(total_possible, 1) * 100
            entered_or_exited.append((count, total_possible, ratio))

        x = np.arange(len(top_k_values))
        width = 0.35

        # Offset for the two categories
        offset = -width/2 if cat == 'constraint_satisfying' else width/2

        ratios = [e[2] for e in entered_or_exited]
        counts = [e[0] for e in entered_or_exited]
        totals = [e[1] for e in entered_or_exited]

        bars = ax.bar(x + offset, ratios, width, label=label,
                      color=color, alpha=0.85, edgecolor='white', linewidth=0.5)

        # Annotate with count/total
        for i, (bar, cnt, tot) in enumerate(zip(bars, counts, totals)):
            if tot > 0:
                ax.annotate(f'{cnt}/{tot}',
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            ha='center', va='bottom', fontsize=6.5, color='gray')

    ax.set_xticks(np.arange(len(top_k_values)))
    ax.set_xticklabels([f'$k$={k}' for k in top_k_values], fontsize=9)
    ax.set_ylabel('Ratio (%)', fontsize=10)
    ax.set_title('(c) Top-$k$ entry/exit rate', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7.5, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(ax.get_ylim()[1], 5))


def main():
    data = load_data()
    docs = data['docs']
    logger.info(f"Loaded {len(docs)} documents from {data['n_total_docs']}")

    # Summary
    for cat in ['constraint_satisfying', 'constraint_affected', 'other']:
        n = sum(1 for d in docs if d['category'] == cat)
        if n > 0:
            avg_delta = np.mean([d['delta'] for d in docs if d['category'] == cat])
            avg_rank = np.mean([d['rank_change'] for d in docs if d['category'] == cat])
            logger.info(f"  {cat}: n={n}, avg_delta={avg_delta:.4f}, avg_rank_change={avg_rank:.2f}")

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=300)
    plt.subplots_adjust(wspace=0.35, left=0.06, right=0.97, bottom=0.15, top=0.88)

    panel_a_score_decomposition(axes[0], docs)
    panel_b_rank_change_scatter(axes[1], docs)
    panel_c_topk_entry_exit(axes[2], docs)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches='tight', dpi=300)
    logger.info(f"\nFigure saved to {OUTPUT_PATH}")

    # Also save a PNG for quick preview
    png_path = OUTPUT_PATH.replace('.pdf', '.png')
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    logger.info(f"PNG preview saved to {png_path}")


if __name__ == "__main__":
    main()
