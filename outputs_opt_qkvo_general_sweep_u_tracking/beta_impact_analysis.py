import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


SWEEP_DIR = "outputs_opt_qkvo_general_sweep_u_tracking"
SUMMARY_FILE = os.path.join(SWEEP_DIR, "sweep_summary.json")
FIG_DIR = os.path.join(SWEEP_DIR, "visualizations_beta")
TABLE_DIR = os.path.join(SWEEP_DIR, "analysis_tables")


def load_summary(summary_file: str) -> pd.DataFrame:
    with open(summary_file, "r", encoding="utf-8") as f:
        runs = json.load(f)

    records = []
    for run in runs:
        combo = run["combo"]
        blocks = combo["target.block_indices"]
        metrics = run.get("quant_metrics_avg", {})
        records.append(
            {
                "run_name": run["run_name"],
                "blocks": "-".join(str(v) for v in blocks),
                "k": len(blocks),
                "beta": float(combo["quant.beta"]),
                "init_mode": combo["quant_ext.init_mode"],
                "baseline_ppl": float(run["baseline_ppl"]),
                "sq_baseline_ppl": float(run["sq_baseline_ppl"]),
                "quantized_ppl": float(run["quantized_ppl"]),
                "delta_ppl": float(run["quantized_ppl"]) - float(run["baseline_ppl"]),
                "sq_gain": float(run["sq_baseline_ppl"]) - float(run["quantized_ppl"]),
                "rel_linear_error": metrics.get("rel_linear_error"),
                "recon_w": metrics.get("relative_recon_error_w"),
                "recon_x": metrics.get("relative_recon_error_x"),
            }
        )

    df = pd.DataFrame(records).sort_values(["k", "blocks", "init_mode", "beta"]).reset_index(drop=True)
    beta_values = sorted(df["beta"].drop_duplicates().tolist())
    non_zero = [v for v in beta_values if v > 0]
    zero_plot_val = (non_zero[0] / 10.0) if non_zero else 0.001
    df["beta_plot"] = df["beta"].map(lambda x: zero_plot_val if x == 0 else x)
    df.attrs["plot_ticks"] = [zero_plot_val] + non_zero
    df.attrs["plot_labels"] = ["0"] + [f"{v:g}" for v in non_zero]
    return df


def ensure_dirs() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)


def plot_ppl(df: pd.DataFrame) -> None:
    block_groups = list(df["blocks"].drop_duplicates())
    fig, axes = plt.subplots(1, len(block_groups), figsize=(8 * len(block_groups), 6), sharey=False)
    if len(block_groups) == 1:
        axes = [axes]

    palette = {"pca": "#1f77b4", "random": "#d62728"}
    for ax, blocks in zip(axes, block_groups):
        sub = df[df["blocks"] == blocks]
        sns.lineplot(
            data=sub,
            x="beta_plot",
            y="quantized_ppl",
            hue="init_mode",
            marker="o",
            linewidth=2,
            palette=palette,
            ax=ax,
        )
        ax.axhline(sub["baseline_ppl"].iloc[0], color="#444444", linestyle="--", linewidth=1.5, label="baseline")
        ax.axhline(sub["sq_baseline_ppl"].iloc[0], color="#888888", linestyle=":", linewidth=1.8, label="sq")
        ax.set_xscale("log")
        ax.set_xticks(df.attrs["plot_ticks"])
        ax.set_xticklabels(df.attrs["plot_labels"])
        ax.set_xlabel("beta")
        ax.set_ylabel("quantized PPL")
        ax.set_title(f"Blocks {blocks}")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    fig.legend(dedup.values(), dedup.keys(), ncol=len(dedup), loc="upper center", frameon=False)
    fig.suptitle("Impact of beta on quantized PPL", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "beta_vs_ppl_overview.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(df: pd.DataFrame) -> None:
    metrics = [
        ("rel_linear_error", "relative linear error"),
        ("recon_w", "weight recon error"),
        ("recon_x", "activation recon error"),
    ]
    block_groups = list(df["blocks"].drop_duplicates())
    fig, axes = plt.subplots(len(metrics), len(block_groups), figsize=(8 * len(block_groups), 4.2 * len(metrics)))
    palette = {"pca": "#1f77b4", "random": "#d62728"}

    for row_idx, (metric_col, metric_title) in enumerate(metrics):
        for col_idx, blocks in enumerate(block_groups):
            ax = axes[row_idx][col_idx] if len(block_groups) > 1 else axes[row_idx]
            sub = df[df["blocks"] == blocks]
            sns.lineplot(
                data=sub,
                x="beta_plot",
                y=metric_col,
                hue="init_mode",
                marker="o",
                linewidth=2,
                palette=palette,
                legend=(row_idx == 0 and col_idx == 0),
                ax=ax,
            )
            ax.set_xscale("log")
            ax.set_xticks(df.attrs["plot_ticks"])
            ax.set_xticklabels(df.attrs["plot_labels"])
            ax.set_xlabel("beta")
            ax.set_ylabel(metric_title)
            ax.set_title(f"{metric_title} | Blocks {blocks}")
            ax.grid(True, alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels() if len(block_groups) > 1 else axes[0].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    fig.legend(dedup.values(), dedup.keys(), ncol=len(dedup), loc="upper center", frameon=False)
    fig.suptitle("Impact of beta on quantization metrics", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "beta_vs_quant_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(df: pd.DataFrame) -> None:
    block_groups = list(df["blocks"].drop_duplicates())
    fig, axes = plt.subplots(1, len(block_groups), figsize=(7 * len(block_groups), 6), sharey=True)
    if len(block_groups) == 1:
        axes = [axes]

    palette = {"pca": "#1f77b4", "random": "#d62728"}
    for ax, blocks in zip(axes, block_groups):
        sub = df[df["blocks"] == blocks]
        sns.scatterplot(
            data=sub,
            x="rel_linear_error",
            y="quantized_ppl",
            hue="init_mode",
            size="beta",
            sizes=(40, 220),
            palette=palette,
            ax=ax,
        )
        for _, row in sub.iterrows():
            ax.text(row["rel_linear_error"], row["quantized_ppl"], f"{row['beta']:g}", fontsize=8, alpha=0.75)
        ax.set_title(f"Blocks {blocks}")
        ax.set_xlabel("relative linear error")
        ax.set_ylabel("quantized PPL")
        ax.grid(True, alpha=0.25)

    fig.suptitle("Trade-off between error and PPL", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "beta_tradeoff_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_tables(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for (blocks, k, init_mode), g in df.groupby(["blocks", "k", "init_mode"]):
        best_idx = g["quantized_ppl"].idxmin()
        summary_rows.append(
            {
                "blocks": blocks,
                "k": k,
                "init_mode": init_mode,
                "best_beta_by_ppl": g.loc[best_idx, "beta"],
                "best_quantized_ppl": g["quantized_ppl"].min(),
                "best_delta_ppl": g.loc[best_idx, "delta_ppl"],
                "best_sq_gain": g.loc[best_idx, "sq_gain"],
                "spearman_beta_vs_ppl": g["beta"].corr(g["quantized_ppl"], method="spearman"),
                "spearman_beta_vs_linear_error": g["beta"].corr(g["rel_linear_error"], method="spearman"),
            }
        )
    summary = pd.DataFrame(summary_rows)

    ranking = df.sort_values(["blocks", "init_mode", "quantized_ppl", "beta"]).reset_index(drop=True)
    summary.to_csv(os.path.join(TABLE_DIR, "beta_impact_summary.csv"), index=False, encoding="utf-8-sig")
    ranking.to_csv(os.path.join(TABLE_DIR, "beta_impact_ranking.csv"), index=False, encoding="utf-8-sig")
    return summary


def export_findings(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = []
    lines.append("# Beta Impact Analysis")
    lines.append("")
    lines.append("## Summary")
    for _, row in summary.sort_values(["blocks", "init_mode"]).iterrows():
        lines.append(
            (
                f"- Blocks `{row['blocks']}` / `{row['init_mode']}`: "
                f"best beta = `{row['best_beta_by_ppl']:g}`, "
                f"best quantized PPL = `{row['best_quantized_ppl']:.4f}`, "
                f"delta vs baseline = `{row['best_delta_ppl']:.4f}`, "
                f"gain vs SQ = `{row['best_sq_gain']:.4f}`."
            )
        )
    lines.append("")
    lines.append("## Trend Notes")

    for (blocks, init_mode), sub in df.groupby(["blocks", "init_mode"]):
        best_row = sub.loc[sub["quantized_ppl"].idxmin()]
        worst_row = sub.loc[sub["quantized_ppl"].idxmax()]
        lines.append(
            (
                f"- Blocks `{blocks}` / `{init_mode}`: as beta moves from `{sub['beta'].min():g}` "
                f"to `{sub['beta'].max():g}`, linear error drops from "
                f"`{sub.iloc[0]['rel_linear_error']:.4f}` to `{sub.iloc[-1]['rel_linear_error']:.4f}`; "
                f"however, the best PPL occurs at beta = `{best_row['beta']:g}` and the worst at "
                f"beta = `{worst_row['beta']:g}`, so lower reconstruction error does not always "
                f"translate into lower PPL."
            )
        )

    with open(os.path.join(TABLE_DIR, "beta_impact_findings.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    ensure_dirs()
    df = load_summary(SUMMARY_FILE)
    plot_ppl(df)
    plot_metrics(df)
    plot_tradeoff(df)
    summary = export_tables(df)
    export_findings(df, summary)
    print("Saved figures to", FIG_DIR)
    print("Saved tables to", TABLE_DIR)


if __name__ == "__main__":
    main()
