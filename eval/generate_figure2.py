"""
Generate Figure 2 PDF for the TRACE paper:
  Left:  z_pos-z_neg scatter with Huber fit and residual annotations
  Right: z_pos-r scatter showing spurious correlation is removed

Also prints Pearson r and mispenalty rate for filling TBD values in the paper.

Usage:
  cd /home/luwa/Documents/DSCLR-remote && \
  /home/luwa/.conda/envs/dsclr/bin/python -m eval.generate_figure2
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======== Paths ========

SCATTER_DATA_PATH = "/home/luwa/Documents/DSCLR-remote/results/figure2/figure2_scatter_data.json"
OUTPUT_PATH = "/home/luwa/Documents/DSCLR-remote/paper/AuthorKit27/AuthorKit27/Figures/figure2.pdf"


def _compute_stats(data):
    """Compute Pearson correlation and mispenalty rate."""
    candidates = data["candidates"]

    all_z_pos = np.array([c["z_pos"] for c in candidates])
    all_z_neg = np.array([c["z_neg"] for c in candidates])
    all_r = np.array([c["r"] for c in candidates])

    # Pearson correlations
    pearson_zneg = float(np.corrcoef(all_z_pos, all_z_neg)[0, 1])
    pearson_r = float(np.corrcoef(all_z_pos, all_r)[0, 1])

    # Mispenalty rate: among constraint-satisfying docs, fraction that are
    # scored below at least one constraint-affected doc by the exclusion signal
    satisfying = [(i, c) for i, c in enumerate(candidates) if c["category"] == "constraint_satisfying"]
    affected = [(i, c) for i, c in enumerate(candidates) if c["category"] == "constraint_affected"]

    if not satisfying or not affected:
        mispenalty_zneg = float("nan")
        mispenalty_r = float("nan")
    else:
        zneg_affected = np.array([c["z_neg"] for _, c in affected])
        zneg_satisfying = np.array([c["z_neg"] for _, c in satisfying])
        r_affected = np.array([c["r"] for _, c in affected])
        r_satisfying = np.array([c["r"] for _, c in satisfying])

        # Pairwise mispenalty rate: over all (satisfying, affected) pairs,
        # the fraction where the satisfying doc has a higher exclusion score
        # (i.e., is more penalized) than the affected doc.
        # Higher exclusion score -> more penalized -> should not happen for satisfying docs
        n_pairs = len(zneg_satisfying) * len(zneg_affected)
        zneg_pairs = zneg_satisfying[:, None] > zneg_affected[None, :]  # (n_sat, n_aff)
        r_pairs = r_satisfying[:, None] > r_affected[None, :]
        mispenalty_zneg = float(zneg_pairs.sum() / n_pairs)
        mispenalty_r = float(r_pairs.sum() / n_pairs)

    print(f"Pearson r(z_pos, z_neg) = {pearson_zneg:.3f}")
    print(f"Pearson r(z_pos, r)     = {pearson_r:.3f}")
    print(f"Mispenalty rate (z_neg)  = {mispenalty_zneg:.3f}")
    print(f"Mispenalty rate (r)      = {mispenalty_r:.3f}")

    return pearson_zneg, pearson_r, mispenalty_zneg, mispenalty_r


def plot_left_panel(ax, data):
    """Plot z_pos-z_neg scatter with Huber fit and residual annotations."""
    candidates = data["candidates"]
    a_hat = data["huber_a"]
    b_hat = data["huber_b"]

    # Separate by category
    satisfying = [c for c in candidates if c["category"] == "constraint_satisfying"]
    affected = [c for c in candidates if c["category"] == "constraint_affected"]
    other = [c for c in candidates if c["category"] == "other"]

    # Plot "other" points (gray, small)
    if other:
        ax.scatter(
            [c["z_pos"] for c in other],
            [c["z_neg"] for c in other],
            c="#B0B0B0", s=18, alpha=0.5, zorder=2, edgecolors='none',
        )

    # Plot constraint-satisfying (blue)
    if satisfying:
        ax.scatter(
            [c["z_pos"] for c in satisfying],
            [c["z_neg"] for c in satisfying],
            c="#4472C4", s=28, alpha=0.85, zorder=3, edgecolors='white',
            linewidths=0.3, label="Constraint-satisfying",
        )

    # Plot constraint-affected (red triangles)
    if affected:
        ax.scatter(
            [c["z_pos"] for c in affected],
            [c["z_neg"] for c in affected],
            c="#C0504D", s=36, alpha=0.9, zorder=4, marker="^",
            edgecolors='white', linewidths=0.3, label="Constraint-affected",
        )

    # Huber fit line
    all_z_pos = [c["z_pos"] for c in candidates]
    x_min, x_max = min(all_z_pos), max(all_z_pos)
    x_line = np.linspace(x_min - 0.3, x_max + 0.3, 100)
    y_line = a_hat + b_hat * x_line
    ax.plot(x_line, y_line, 'k--', linewidth=1.2, alpha=0.7, zorder=5,
            label=r"Huber fit $\hat{a}+\hat{b}\,z_{\mathrm{pos}}$")

    # Annotate residuals for affected candidates (pick up to 3 with largest residuals)
    if affected:
        affected_sorted = sorted(affected, key=lambda c: c["r"], reverse=True)
        for c in affected_sorted[:3]:
            z_p, z_n = c["z_pos"], c["z_neg"]
            y_pred = a_hat + b_hat * z_p
            # Draw vertical arrow from fit line to point
            ax.annotate(
                "", xy=(z_p, z_n), xytext=(z_p, y_pred),
                arrowprops=dict(arrowstyle="->", color="#C0504D", lw=1.0, shrinkA=0, shrinkB=0),
                zorder=6,
            )

    ax.set_xlabel(r"$z_{\mathrm{pos}}$", fontsize=10)
    ax.set_ylabel(r"$z_{\mathrm{neg}}$", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9, handletextpad=0.4)
    ax.tick_params(labelsize=8)

    # Subtle grid
    ax.grid(True, alpha=0.15, linewidth=0.5)

    # Annotate correlation
    all_z_pos_arr = np.array([c["z_pos"] for c in candidates])
    all_z_neg_arr = np.array([c["z_neg"] for c in candidates])
    pearson = np.corrcoef(all_z_pos_arr, all_z_neg_arr)[0, 1]
    ax.text(0.97, 0.03, f"Pearson $r$ = {pearson:.2f}",
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    qid = data.get("qid_base", data.get("qid_changed", ""))
    ax.set_title(f"Raw: $z_{{\\mathrm{{neg}}}}$ vs $z_{{\\mathrm{{pos}}}}$ (qid {qid.replace('-changed','')})",
                 fontsize=9, pad=4)


def plot_right_panel(ax, data):
    """Plot z_pos-r scatter showing spurious correlation is removed."""
    candidates = data["candidates"]

    # Separate by category
    satisfying = [c for c in candidates if c["category"] == "constraint_satisfying"]
    affected = [c for c in candidates if c["category"] == "constraint_affected"]
    other = [c for c in candidates if c["category"] == "other"]

    # Plot "other" points (gray, small)
    if other:
        ax.scatter(
            [c["z_pos"] for c in other],
            [c["r"] for c in other],
            c="#B0B0B0", s=18, alpha=0.5, zorder=2, edgecolors='none',
        )

    # Plot constraint-satisfying (blue)
    if satisfying:
        ax.scatter(
            [c["z_pos"] for c in satisfying],
            [c["r"] for c in satisfying],
            c="#4472C4", s=28, alpha=0.85, zorder=3, edgecolors='white',
            linewidths=0.3, label="Constraint-satisfying",
        )

    # Plot constraint-affected (red triangles)
    if affected:
        ax.scatter(
            [c["z_pos"] for c in affected],
            [c["r"] for c in affected],
            c="#C0504D", s=36, alpha=0.9, zorder=4, marker="^",
            edgecolors='white', linewidths=0.3, label="Constraint-affected",
        )

    # Zero line (residuals should be centered around 0)
    all_z_pos = [c["z_pos"] for c in candidates]
    x_min, x_max = min(all_z_pos), max(all_z_pos)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1.0, alpha=0.5, zorder=1)

    ax.set_xlabel(r"$z_{\mathrm{pos}}$", fontsize=10)
    ax.set_ylabel(r"$r$", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9, handletextpad=0.4)
    ax.tick_params(labelsize=8)

    # Subtle grid
    ax.grid(True, alpha=0.15, linewidth=0.5)

    # Annotate correlation
    all_z_pos_arr = np.array([c["z_pos"] for c in candidates])
    all_r_arr = np.array([c["r"] for c in candidates])
    pearson = np.corrcoef(all_z_pos_arr, all_r_arr)[0, 1]
    ax.text(0.97, 0.03, f"Pearson $r$ = {pearson:.2f}",
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    ax.set_title(r"Residualized: $r$ vs $z_{\mathrm{pos}}$",
                 fontsize=9, pad=4)


def main():
    # Load scatter data
    if not os.path.exists(SCATTER_DATA_PATH):
        print(f"ERROR: Scatter data not found at {SCATTER_DATA_PATH}. "
              "Run generate_figure2_data.py first.")
        sys.exit(1)

    with open(SCATTER_DATA_PATH, 'r') as f:
        scatter_data = json.load(f)

    # Compute and print statistics for paper TBD values
    print("=" * 50)
    print("Statistics for Mechanism Analysis section:")
    print("=" * 50)
    _compute_stats(scatter_data)
    print("=" * 50)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

    plot_left_panel(ax1, scatter_data)
    plot_right_panel(ax2, scatter_data)

    plt.tight_layout(w_pad=2.0)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Figure 2 saved to {OUTPUT_PATH}")

    # Also save PNG for quick preview
    png_path = OUTPUT_PATH.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print(f"Preview saved to {png_path}")


if __name__ == "__main__":
    main()
