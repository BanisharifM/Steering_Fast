"""Analyze GCG prefix optimization results across all concepts.

Generates:
1. Summary statistics (CSV + text)
2. Correlation: handcrafted baseline vs improvement
3. Token pattern analysis: which tokens are kept/replaced most often
4. Per-layer analysis from all-layers experiments
5. Matplotlib plots saved as PNG

Usage:
    python -m steering_fast.prefix_optimization.analyze_results \
        --results_dir outputs/prefix_optimization/llama_3.1_8b/fears
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from typing import Dict, List

def load_all_results(results_dir: str) -> List[Dict]:
    """Load summary.json from each concept directory."""
    rows = []
    for concept_dir in sorted(os.listdir(results_dir)):
        summary_path = os.path.join(results_dir, concept_dir, "summary.json")
        if not os.path.isfile(summary_path):
            continue
        with open(summary_path) as f:
            d = json.load(f)
        try:
            rows.append({
                "concept": d["concept"],
                "l16_handcrafted": float(d["gcg_layer16_handcrafted"]),
                "l16_no_prefix": float(d.get("gcg_layer16_no_prefix", 0)),
                "l16_optimized": float(d["gcg_layer16_optimized"]),
                "l16_improvement": float(d["gcg_layer16_improvement"]),
                "l16_prefix": d.get("gcg_layer16_prefix", ""),
                "l16_time": float(d.get("gcg_layer16_time", 0)),
                "al_handcrafted": float(d["gcg_all_layers_handcrafted"]),
                "al_no_prefix": float(d.get("gcg_all_layers_no_prefix", 0)),
                "al_optimized": float(d["gcg_all_layers_optimized"]),
                "al_improvement": float(d["gcg_all_layers_improvement"]),
                "al_prefix": d.get("gcg_all_layers_prefix", ""),
            })
        except (KeyError, TypeError) as e:
            print(f"Skipping {concept_dir}: {e}")
    return rows


def compute_statistics(rows: List[Dict]) -> str:
    """Compute and format all statistics."""
    import statistics as stats

    n = len(rows)
    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"GCG PREFIX OPTIMIZATION: FULL ANALYSIS ({n} concepts)")
    lines.append(f"{'='*70}")
    lines.append("")

    # Layer 16
    l16_h = [r["l16_handcrafted"] for r in rows]
    l16_o = [r["l16_optimized"] for r in rows]
    l16_i = [r["l16_improvement"] for r in rows]
    l16_np = [r["l16_no_prefix"] for r in rows]
    improved_16 = sum(1 for x in l16_i if x > 0)

    lines.append("LAYER 16 (SINGLE LAYER)")
    lines.append(f"  Concepts improved: {improved_16}/{n} ({improved_16/n*100:.1f}%)")
    lines.append(f"  Mean no-prefix baseline:  {stats.mean(l16_np):.4f}")
    lines.append(f"  Mean handcrafted cos_sim: {stats.mean(l16_h):.4f}")
    lines.append(f"  Mean optimized cos_sim:   {stats.mean(l16_o):.4f}")
    lines.append(f"  Mean improvement:         {stats.mean(l16_i):.4f}")
    lines.append(f"  Median improvement:       {stats.median(l16_i):.4f}")
    lines.append(f"  Std improvement:          {stats.stdev(l16_i):.4f}")
    lines.append(f"  Min improvement:          {min(l16_i):.4f}")
    lines.append(f"  Max improvement:          {max(l16_i):.4f}")
    lines.append(f"  Mean delta (vs no-prefix):{stats.mean(l16_o) - stats.mean(l16_np):.4f}")
    lines.append("")

    # All layers
    al_h = [r["al_handcrafted"] for r in rows]
    al_o = [r["al_optimized"] for r in rows]
    al_i = [r["al_improvement"] for r in rows]
    improved_al = sum(1 for x in al_i if x > 0)

    lines.append("ALL 31 LAYERS (MEAN)")
    lines.append(f"  Concepts improved: {improved_al}/{n} ({improved_al/n*100:.1f}%)")
    lines.append(f"  Mean handcrafted cos_sim: {stats.mean(al_h):.4f}")
    lines.append(f"  Mean optimized cos_sim:   {stats.mean(al_o):.4f}")
    lines.append(f"  Mean improvement:         {stats.mean(al_i):.4f}")
    lines.append(f"  Median improvement:       {stats.median(al_i):.4f}")
    lines.append("")

    # Correlation: handcrafted baseline vs improvement
    # Pearson correlation
    mean_h = stats.mean(l16_h)
    mean_i = stats.mean(l16_i)
    cov = sum((h - mean_h) * (i - mean_i) for h, i in zip(l16_h, l16_i)) / n
    std_h = stats.stdev(l16_h)
    std_i = stats.stdev(l16_i)
    corr = cov / (std_h * std_i) if std_h > 0 and std_i > 0 else 0

    lines.append("CORRELATION ANALYSIS")
    lines.append(f"  Pearson(handcrafted, improvement) at L16: {corr:.4f}")
    if corr < -0.3:
        lines.append("  -> Strong negative: concepts with high baselines improve less (diminishing returns)")
    elif corr > 0.3:
        lines.append("  -> Strong positive: concepts with high baselines improve more")
    else:
        lines.append("  -> Weak correlation: improvement is relatively independent of baseline")
    lines.append("")

    # Top/bottom concepts
    sorted_by_improvement = sorted(rows, key=lambda r: r["l16_improvement"], reverse=True)

    lines.append("TOP 10 MOST IMPROVED (Layer 16)")
    for r in sorted_by_improvement[:10]:
        lines.append(f"  {r['concept']:25s}  {r['l16_handcrafted']:+.3f} -> {r['l16_optimized']:+.3f}  (+{r['l16_improvement']:.3f})")
    lines.append("")

    lines.append("BOTTOM 10 LEAST IMPROVED (Layer 16)")
    for r in sorted_by_improvement[-10:]:
        lines.append(f"  {r['concept']:25s}  {r['l16_handcrafted']:+.3f} -> {r['l16_optimized']:+.3f}  (+{r['l16_improvement']:.3f})")
    lines.append("")

    # Token analysis
    lines.append("TOKEN PATTERN ANALYSIS")

    # Which original tokens survive optimization?
    original_tokens = ["Personify", "someone", "who", "is", "terrified", "of"]
    survived_count = Counter()
    total = 0

    for r in rows:
        prefix = r["l16_prefix"]
        total += 1
        for tok in original_tokens:
            if tok.lower() in prefix.lower():
                survived_count[tok] += 1

    lines.append("  Original tokens that survive optimization (Layer 16):")
    for tok in original_tokens:
        pct = survived_count[tok] / total * 100
        bar = "#" * int(pct / 2)
        lines.append(f"    {tok:15s}  {survived_count[tok]:3d}/{total}  ({pct:5.1f}%)  {bar}")

    lines.append("")

    # Most common replacement tokens
    all_tokens = Counter()
    for r in rows:
        prefix = r["l16_prefix"]
        for tok in prefix.split():
            tok_clean = tok.strip(".,!()[]{}\"'")
            if tok_clean and tok_clean not in original_tokens and tok_clean.lower() not in [t.lower() for t in original_tokens]:
                all_tokens[tok_clean] += 1

    lines.append("  Most common NEW tokens in optimized prefixes:")
    for tok, count in all_tokens.most_common(15):
        lines.append(f"    {tok:20s}  appears in {count}/{total} concepts ({count/total*100:.1f}%)")

    lines.append("")

    # Timing
    times = [r["l16_time"] for r in rows if r["l16_time"] > 0]
    if times:
        lines.append("TIMING")
        lines.append(f"  Mean time per concept (L16): {stats.mean(times):.1f}s")
        lines.append(f"  Total time (L16 only): {sum(times)/60:.1f} min")
        lines.append(f"  Total time (all experiments): ~{sum(times)*2/60:.1f} min")

    return "\n".join(lines)


def save_csv(rows: List[Dict], output_path: str):
    """Save all results as a CSV for external analysis."""
    keys = ["concept", "l16_handcrafted", "l16_no_prefix", "l16_optimized", "l16_improvement",
            "al_handcrafted", "al_no_prefix", "al_optimized", "al_improvement",
            "l16_prefix", "al_prefix", "l16_time"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["concept"]))


def generate_plots(rows: List[Dict], output_dir: str):
    """Generate matplotlib plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    concepts = [r["concept"] for r in rows]
    l16_h = [r["l16_handcrafted"] for r in rows]
    l16_o = [r["l16_optimized"] for r in rows]
    l16_i = [r["l16_improvement"] for r in rows]

    # Sort by improvement for the bar chart
    sorted_idx = np.argsort(l16_i)[::-1]

    # Plot 1: Handcrafted vs Optimized scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(l16_h, l16_o, alpha=0.6, s=40, c=l16_i, cmap="viridis", edgecolors="black", linewidth=0.5)
    ax.plot([min(l16_h)-0.1, max(l16_o)+0.1], [min(l16_h)-0.1, max(l16_o)+0.1], "k--", alpha=0.3, label="y=x (no improvement)")
    ax.set_xlabel("Hand-crafted cos_sim", fontsize=12)
    ax.set_ylabel("GCG optimized cos_sim", fontsize=12)
    ax.set_title("GCG Prefix Optimization: 100 Fears Concepts (Layer 16)", fontsize=14)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Improvement", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_handcrafted_vs_optimized.png"), dpi=150)
    plt.close()
    print(f"Saved scatter plot")

    # Plot 2: Improvement distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(l16_i, bins=20, color="#58a6ff", edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(l16_i), color="red", linestyle="--", label=f"Mean: {np.mean(l16_i):.3f}")
    ax.axvline(np.median(l16_i), color="orange", linestyle="--", label=f"Median: {np.median(l16_i):.3f}")
    ax.set_xlabel("Improvement (cos_sim)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of GCG Improvement over Hand-crafted (Layer 16)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "histogram_improvement.png"), dpi=150)
    plt.close()
    print(f"Saved histogram")

    # Plot 3: Improvement vs handcrafted baseline
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(l16_h, l16_i, alpha=0.6, s=40, c="#bc8cff", edgecolors="black", linewidth=0.5)
    # Fit a trend line
    z = np.polyfit(l16_h, l16_i, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(l16_h), max(l16_h), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.5, label=f"Trend (slope={z[0]:.3f})")
    ax.set_xlabel("Hand-crafted cos_sim (baseline)", fontsize=12)
    ax.set_ylabel("GCG improvement", fontsize=12)
    ax.set_title("Improvement vs Baseline Strength (Layer 16)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_baseline_vs_improvement.png"), dpi=150)
    plt.close()
    print(f"Saved baseline vs improvement scatter")

    # Plot 4: Top 20 concepts bar chart
    top20_idx = sorted_idx[:20]
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = range(20)
    concepts_20 = [concepts[i] for i in top20_idx]
    h_20 = [l16_h[i] for i in top20_idx]
    o_20 = [l16_o[i] for i in top20_idx]

    bars_h = ax.barh(y_pos, h_20, height=0.35, color="#8b949e", label="Hand-crafted", align="edge")
    bars_o = ax.barh([y + 0.35 for y in y_pos], o_20, height=0.35, color="#58a6ff", label="GCG optimized", align="edge")
    ax.set_yticks([y + 0.35 for y in y_pos])
    ax.set_yticklabels(concepts_20, fontsize=9)
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_title("Top 20 Most Improved Concepts (Layer 16)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis="x")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bar_top20_concepts.png"), dpi=150)
    plt.close()
    print(f"Saved top 20 bar chart")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "analysis")

    os.makedirs(args.output_dir, exist_ok=True)

    rows = load_all_results(args.results_dir)
    print(f"Loaded {len(rows)} concept results")

    # Statistics
    stats_text = compute_statistics(rows)
    print(stats_text)

    stats_path = os.path.join(args.output_dir, "analysis_report.txt")
    with open(stats_path, "w") as f:
        f.write(stats_text)
    print(f"\nSaved report to {stats_path}")

    # CSV
    csv_path = os.path.join(args.output_dir, "all_results.csv")
    save_csv(rows, csv_path)
    print(f"Saved CSV to {csv_path}")

    # Plots
    generate_plots(rows, args.output_dir)


if __name__ == "__main__":
    main()
