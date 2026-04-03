"""
Re-analysis for OPT-125m Last-K Blocks QKVO quantization sweep
Better statistics + paper-friendly visualization

Expected input:
- outputs_opt_qkvo_lastk_sweep_v1/lastk_qkvo_sweep_results.json
- outputs_opt_qkvo_lastk_sweep_v1/last_{k}_blocks/all_blocks_qkvo_results.json
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import textwrap

# =========================
# Config
# =========================
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

WORKSPACE_DIR = Path(__file__).parent
OUTPUT_BASE_DIR = WORKSPACE_DIR / "outputs_opt_qkvo_lastk_sweep_v1"
ANALYSIS_DIR = OUTPUT_BASE_DIR / "analysis_v2"
ANALYSIS_DIR.mkdir(exist_ok=True, parents=True)

SUMMARY_FILE = OUTPUT_BASE_DIR / "lastk_qkvo_sweep_results.json"


# =========================
# IO
# =========================
def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_sweep_results():
    return load_json(SUMMARY_FILE)

def load_block_results(last_k):
    return load_json(OUTPUT_BASE_DIR / f"last_{last_k}_blocks" / "all_blocks_qkvo_results.json")


# =========================
# Helpers
# =========================
def safe_get(d, key, default=np.nan):
    return d.get(key, default) if isinstance(d, dict) else default

def parse_proj_type(layer_name: str):
    name = layer_name.lower()
    for p in ["q_proj", "k_proj", "v_proj", "out_proj"]:
        if p in name:
            return p
    return "other"

def parse_block_id(layer_name: str):
    # try to parse "model.decoder.layers.11.self_attn.k_proj"
    import re
    m = re.search(r"layers\.(\d+)", layer_name)
    if m:
        return int(m.group(1))
    return -1


# =========================
# Build DataFrames
# =========================
def build_summary_df():
    sweep = load_sweep_results()
    rows = []

    for r in sorted(sweep["results"], key=lambda x: x["last_k"]):
        baseline = r["baseline_ppl"]
        sq = r["sq_ppl"]
        ours = r["ours_ppl"]

        rows.append({
            "last_k": r["last_k"],
            "baseline_ppl": baseline,
            "sq_ppl": sq,
            "ours_ppl": ours,
            "sq_delta_ppl": sq - baseline,
            "ours_delta_ppl": ours - baseline,
            "ours_minus_sq": ours - sq,
            "ours_improve_over_sq_abs": sq - ours,
            "ours_improve_over_sq_pct": (sq - ours) / sq * 100 if sq != 0 else np.nan,
        })

    df = pd.DataFrame(rows).sort_values("last_k").reset_index(drop=True)
    return df

def build_layer_df():
    summary_df = build_summary_df()
    all_rows = []

    for last_k in summary_df["last_k"]:
        block_data = load_block_results(last_k)

        for method_key, method_name in [("sq_metrics", "SQ"), ("quant_metrics", "Ours")]:
            metrics_dict = block_data.get(method_key, {})
            for layer_name, metrics in metrics_dict.items():
                all_rows.append({
                    "last_k": last_k,
                    "method": method_name,
                    "layer_name": layer_name,
                    "block_id": parse_block_id(layer_name),
                    "proj_type": parse_proj_type(layer_name),
                    "rel_recon_error_x": safe_get(metrics, "rel_recon_error_x"),
                    "rel_recon_error_w": safe_get(metrics, "rel_recon_error_w"),
                    "rel_linear_error": safe_get(metrics, "rel_linear_error"),
                })

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["last_k", "method", "block_id", "layer_name"]).reset_index(drop=True)
    return df


# =========================
# Plot 1: PPL curves
# =========================
def plot_ppl(summary_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = summary_df["last_k"]

    # Absolute PPL
    ax = axes[0]
    ax.plot(x, summary_df["baseline_ppl"], marker='o', linewidth=2, label='FP Baseline')
    ax.plot(x, summary_df["sq_ppl"], marker='s', linewidth=2, label='SQ')
    ax.plot(x, summary_df["ours_ppl"], marker='^', linewidth=2, label='Ours')
    ax.set_title("Perplexity vs Last-K")
    ax.set_xlabel("Last-K blocks quantized")
    ax.set_ylabel("PPL")
    ax.grid(alpha=0.3)
    ax.legend()

    # Delta PPL
    ax = axes[1]
    ax.plot(x, summary_df["sq_delta_ppl"], marker='s', linewidth=2, label='SQ ΔPPL')
    ax.plot(x, summary_df["ours_delta_ppl"], marker='^', linewidth=2, label='Ours ΔPPL')
    ax.set_title("Perplexity Degradation vs Last-K")
    ax.set_xlabel("Last-K blocks quantized")
    ax.set_ylabel("ΔPPL (vs FP)")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "01_ppl_vs_lastk.png", bbox_inches="tight")
    plt.close()


# =========================
# Plot 2: Improvement over SQ
# =========================
def plot_improvement(summary_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = summary_df["last_k"]

    ax = axes[0]
    ax.bar(x, summary_df["ours_improve_over_sq_abs"])
    ax.set_title("Absolute PPL Improvement over SQ")
    ax.set_xlabel("Last-K")
    ax.set_ylabel("SQ PPL - Ours PPL")
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x, summary_df["ours_improve_over_sq_pct"])
    ax.set_title("Relative Improvement over SQ (%)")
    ax.set_xlabel("Last-K")
    ax.set_ylabel("Improvement (%)")
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "02_delta_ppl_vs_lastk.png", bbox_inches="tight")
    plt.close()


# =========================
# Plot 3: Heatmaps
# =========================
def plot_heatmaps(layer_df, metric="rel_linear_error"):
    for method in ["SQ", "Ours"]:
        sub = layer_df[layer_df["method"] == method].copy()

        pivot = sub.pivot_table(
            index="layer_name",
            columns="last_k",
            values=metric,
            aggfunc="mean"
        ).sort_index()

        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.35)))
        im = ax.imshow(pivot.values, aspect='auto')
        ax.set_title(f"{method} {metric} Heatmap")
        ax.set_xlabel("Last-K")
        ax.set_ylabel("Layer")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(ANALYSIS_DIR / f"03_{method.lower()}_{metric}_heatmap.png", bbox_inches="tight")
        plt.close()

    # Ours - SQ gap
    sq_pivot = layer_df[layer_df["method"] == "SQ"].pivot_table(
        index="layer_name", columns="last_k", values=metric, aggfunc="mean"
    )
    ours_pivot = layer_df[layer_df["method"] == "Ours"].pivot_table(
        index="layer_name", columns="last_k", values=metric, aggfunc="mean"
    )

    common_idx = sorted(set(sq_pivot.index) & set(ours_pivot.index))
    common_cols = sorted(set(sq_pivot.columns) & set(ours_pivot.columns))

    gap = ours_pivot.loc[common_idx, common_cols] - sq_pivot.loc[common_idx, common_cols]

    fig, ax = plt.subplots(figsize=(10, max(6, len(gap) * 0.35)))
    im = ax.imshow(gap.values, aspect='auto')
    ax.set_title(f"Ours - SQ {metric} Heatmap (negative is better)")
    ax.set_xlabel("Last-K")
    ax.set_ylabel("Layer")
    ax.set_xticks(np.arange(len(gap.columns)))
    ax.set_xticklabels(gap.columns)
    ax.set_yticks(np.arange(len(gap.index)))
    ax.set_yticklabels(gap.index, fontsize=8)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / f"04_gap_{metric}_heatmap.png", bbox_inches="tight")
    plt.close()


# =========================
# Plot 4: proj-type trends
# =========================
def plot_projtype_trends(layer_df):
    grouped = layer_df.groupby(["last_k", "method", "proj_type"])[
        ["rel_recon_error_x", "rel_recon_error_w", "rel_linear_error"]
    ].mean().reset_index()

    metrics = ["rel_recon_error_x", "rel_recon_error_w", "rel_linear_error"]
    titles = ["Activation Reconstruction Error", "Weight Reconstruction Error", "Linear Output Error"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric, title in zip(axes, metrics, titles):
        for method in ["SQ", "Ours"]:
            for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                sub = grouped[(grouped["method"] == method) & (grouped["proj_type"] == proj)]
                if len(sub) == 0:
                    continue
                linestyle = "-" if method == "Ours" else "--"
                marker = {"q_proj": "o", "k_proj": "s", "v_proj": "^", "out_proj": "d"}[proj]
                ax.plot(sub["last_k"], sub[metric], linestyle=linestyle, marker=marker,
                        label=f"{method}-{proj}")

        ax.set_title(title)
        ax.set_xlabel("Last-K")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(ANALYSIS_DIR / "05_projtype_error_trends.png", bbox_inches="tight")
    plt.close()


# =========================
# Plot 5: error vs PPL
# =========================
def plot_error_vs_ppl(summary_df, layer_df):
    avg_err = layer_df.groupby(["last_k", "method"])[
        ["rel_recon_error_x", "rel_recon_error_w", "rel_linear_error"]
    ].mean().reset_index()

    merged = avg_err.merge(
        summary_df[["last_k", "sq_delta_ppl", "ours_delta_ppl"]],
        on="last_k",
        how="left"
    )
    merged["delta_ppl"] = np.where(
        merged["method"] == "SQ",
        merged["sq_delta_ppl"],
        merged["ours_delta_ppl"]
    )

    metrics = ["rel_recon_error_x", "rel_recon_error_w", "rel_linear_error"]
    titles = ["Avg rel_recon_error_x", "Avg rel_recon_error_w", "Avg rel_linear_error"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric, title in zip(axes, metrics, titles):
        for method, marker in [("SQ", "s"), ("Ours", "^")]:
            sub = merged[merged["method"] == method]
            ax.scatter(sub[metric], sub["delta_ppl"], marker=marker, label=method)
            for _, row in sub.iterrows():
                ax.annotate(f"k={int(row['last_k'])}", (row[metric], row["delta_ppl"]), fontsize=8)

        corr = merged[[metric, "delta_ppl"]].corr().iloc[0, 1]
        ax.set_title(f"{title} vs ΔPPL\ncorr={corr:.3f}")
        ax.set_xlabel(metric)
        ax.set_ylabel("ΔPPL")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "06_error_vs_ppl_scatter.png", bbox_inches="tight")
    plt.close()


# =========================
# Plot 6: layer-wise correlation with PPL
# =========================
def plot_layer_ppl_correlation(summary_df, layer_df, metric="rel_linear_error"):
    rows = []

    for method in ["SQ", "Ours"]:
        delta_col = "sq_delta_ppl" if method == "SQ" else "ours_delta_ppl"

        for layer_name, sub in layer_df[layer_df["method"] == method].groupby("layer_name"):
            merged = sub.merge(summary_df[["last_k", delta_col]], on="last_k", how="left")
            if merged[metric].notna().sum() >= 2:
                corr = merged[[metric, delta_col]].corr().iloc[0, 1]
                rows.append({
                    "method": method,
                    "layer_name": layer_name,
                    "corr_with_delta_ppl": corr
                })

    corr_df = pd.DataFrame(rows)
    corr_df = corr_df.sort_values("corr_with_delta_ppl", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(corr_df["layer_name"].unique()) * 0.3)))

    for ax, method in zip(axes, ["SQ", "Ours"]):
        sub = corr_df[corr_df["method"] == method].sort_values("corr_with_delta_ppl")
        ax.barh(sub["layer_name"], sub["corr_with_delta_ppl"])
        ax.set_title(f"{method}: layer {metric} correlation with ΔPPL")
        ax.set_xlabel("Pearson correlation")
        ax.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / "07_layer_ppl_correlation.png", bbox_inches="tight")
    plt.close()

    return corr_df


# =========================
# Report
# =========================
def generate_report(summary_df, layer_df, corr_df):
    report_path = ANALYSIS_DIR / "analysis_summary.txt"

    # global averages
    grouped = layer_df.groupby("method")[["rel_recon_error_x", "rel_recon_error_w", "rel_linear_error"]].mean()

    best_ours_row = summary_df.loc[summary_df["ours_ppl"].idxmin()]
    best_sq_row = summary_df.loc[summary_df["sq_ppl"].idxmin()]
    best_improve_row = summary_df.loc[summary_df["ours_improve_over_sq_abs"].idxmax()]

    # projection sensitivity
    proj_group = layer_df.groupby(["method", "proj_type"])[
        ["rel_recon_error_x", "rel_recon_error_w", "rel_linear_error"]
    ].mean().reset_index()

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("OPT-125m Last-K Blocks QKVO Sweep - Reanalysis Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write("[1] PPL Summary\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")

        f.write("[2] Best / Worst Cases\n")
        f.write(f"Best Ours PPL: {best_ours_row['ours_ppl']:.6f} at last_k={int(best_ours_row['last_k'])}\n")
        f.write(f"Best SQ   PPL: {best_sq_row['sq_ppl']:.6f} at last_k={int(best_sq_row['last_k'])}\n")
        f.write(f"Largest improvement over SQ: {best_improve_row['ours_improve_over_sq_abs']:.6f} "
                f"at last_k={int(best_improve_row['last_k'])}\n\n")

        f.write("[3] Global Average Errors\n")
        f.write(grouped.to_string())
        f.write("\n\n")

        f.write("[4] Projection-Type Average Errors\n")
        f.write(proj_group.to_string(index=False))
        f.write("\n\n")

        f.write("[5] Top Layers Most Correlated with PPL Degradation\n")
        for method in ["SQ", "Ours"]:
            f.write(f"\nMethod = {method}\n")
            top = corr_df[corr_df["method"] == method].sort_values("corr_with_delta_ppl", ascending=False).head(10)
            f.write(top.to_string(index=False))
            f.write("\n")

        f.write("\n[6] Interpretation Template\n")
        interpretation = """
        Suggested interpretation:
        1. Check whether ΔPPL increases monotonically with last_k.
        2. Identify whether Ours consistently reduces rel_linear_error compared with SQ.
        3. Check which projection type (q/k/v/out) contributes most to degradation.
        4. Use layer-PPL correlation to identify the most performance-critical layers.
        5. If Ours mainly reduces activation-side error (rel_recon_error_x), then the improvement
           likely comes from better activation subspace preservation rather than only weight fidelity.
        """
        f.write(textwrap.dedent(interpretation))

    return report_path


# =========================
# Main
# =========================
def main():
    print("=" * 80)
    print("Building data tables...")
    print("=" * 80)

    summary_df = build_summary_df()
    layer_df = build_layer_df()

    summary_df.to_csv(ANALYSIS_DIR / "summary_lastk.csv", index=False, encoding="utf-8-sig")
    layer_df.to_csv(ANALYSIS_DIR / "layer_metrics_long.csv", index=False, encoding="utf-8-sig")

    print("summary_lastk.csv saved")
    print("layer_metrics_long.csv saved")

    print("\nGenerating plots...")
    plot_ppl(summary_df)
    plot_improvement(summary_df)
    plot_heatmaps(layer_df, metric="rel_linear_error")
    plot_heatmaps(layer_df, metric="rel_recon_error_x")
    plot_heatmaps(layer_df, metric="rel_recon_error_w")
    plot_projtype_trends(layer_df)
    plot_error_vs_ppl(summary_df, layer_df)
    corr_df = plot_layer_ppl_correlation(summary_df, layer_df, metric="rel_linear_error")

    corr_df.to_csv(ANALYSIS_DIR / "layer_ppl_correlation.csv", index=False, encoding="utf-8-sig")

    report_path = generate_report(summary_df, layer_df, corr_df)

    print("\n" + "=" * 80)
    print("Done.")
    print(f"Results saved to: {ANALYSIS_DIR}")
    print(f"Report: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()