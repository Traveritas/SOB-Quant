from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt


DEFAULT_SWEEP_DIRS = (
    "qkvo_sweep_abs_gamma",
    "qkvo_sweep_abs_gamma_again",
    "qkvo_sweep_abs_gamma_again2",
)
DEFAULT_OUTPUT_DIR = "analysis_qkvo_gamma_ppl"
GAMMA_PATTERN = re.compile(r"gamma_([-+]?\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class RunRecord:
    sweep_dir: str
    run_name: str
    gamma: float
    seed: int | None
    baseline_ppl: float | None
    quantized_ppl: float
    quantized_delta: float | None
    results_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate qkvo gamma sweep results across multiple seed directories "
            "and plot gamma vs quantized PPL."
        )
    )
    parser.add_argument(
        "--sweep-dirs",
        nargs="+",
        default=list(DEFAULT_SWEEP_DIRS),
        help="Directories that contain per-gamma experiment subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the analysis tables and plots will be written.",
    )
    parser.add_argument(
        "--results-file",
        default="results.json",
        help="Result filename expected inside each experiment directory.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def extract_gamma(run_name: str, payload: dict[str, Any]) -> float:
    gamma = payload.get("config", {}).get("quant", {}).get("ip_reg_gamma")
    if gamma is not None:
        return float(gamma)

    match = GAMMA_PATTERN.search(run_name)
    if not match:
        raise ValueError(f"Could not infer gamma from run name: {run_name}")
    return float(match.group(1))


def load_run_record(results_path: Path, sweep_dir_name: str) -> RunRecord:
    payload = load_json(results_path)
    run_name = results_path.parent.name
    gamma = extract_gamma(run_name, payload)
    config = payload.get("config", {})
    seed_value = config.get("seed")
    baseline_ppl = maybe_float(payload.get("baseline_ppl"))
    quantized_ppl = maybe_float(payload.get("quantized_ppl"))
    if quantized_ppl is None:
        raise ValueError(f"Missing quantized_ppl in {results_path}")
    delta = None if baseline_ppl is None else quantized_ppl - baseline_ppl
    return RunRecord(
        sweep_dir=sweep_dir_name,
        run_name=run_name,
        gamma=gamma,
        seed=None if seed_value is None else int(seed_value),
        baseline_ppl=baseline_ppl,
        quantized_ppl=quantized_ppl,
        quantized_delta=delta,
        results_path=results_path,
    )


def iter_results_paths(sweep_dir: Path, results_file: str) -> Iterable[Path]:
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")
    for child in sorted(entry for entry in sweep_dir.iterdir() if entry.is_dir()):
        results_path = child / results_file
        if results_path.exists():
            yield results_path


def collect_records(root: Path, sweep_dirs: list[str], results_file: str) -> list[RunRecord]:
    records: list[RunRecord] = []
    for sweep_dir_name in sweep_dirs:
        sweep_dir = root / sweep_dir_name
        for results_path in iter_results_paths(sweep_dir, results_file):
            records.append(load_run_record(results_path, sweep_dir_name))
    records.sort(key=lambda item: (item.gamma, item.seed or -1, item.sweep_dir, item.run_name))
    if not records:
        joined = ", ".join(sweep_dirs)
        raise RuntimeError(f"No result files were found under: {joined}")
    return records


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_by_gamma(records: list[RunRecord]) -> list[dict[str, Any]]:
    grouped: dict[float, list[RunRecord]] = {}
    for record in records:
        grouped.setdefault(record.gamma, []).append(record)

    summary_rows: list[dict[str, Any]] = []
    for gamma in sorted(grouped):
        group = grouped[gamma]
        ppl_values = [item.quantized_ppl for item in group]
        baseline_values = [item.baseline_ppl for item in group if item.baseline_ppl is not None]
        delta_values = [item.quantized_delta for item in group if item.quantized_delta is not None]
        seeds = [item.seed for item in group if item.seed is not None]
        summary_rows.append(
            {
                "gamma": gamma,
                "n_runs": len(group),
                "seeds": seeds,
                "ppl_mean": mean(ppl_values),
                "ppl_std": sample_std(ppl_values),
                "ppl_min": min(ppl_values),
                "ppl_max": max(ppl_values),
                "baseline_ppl_mean": mean(baseline_values) if baseline_values else None,
                "baseline_ppl_std": sample_std(baseline_values) if baseline_values else None,
                "quantized_delta_mean": mean(delta_values) if delta_values else None,
                "quantized_delta_std": sample_std(delta_values) if delta_values else None,
            }
        )
    return summary_rows


def write_raw_csv(path: Path, records: list[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sweep_dir",
                "run_name",
                "gamma",
                "seed",
                "baseline_ppl",
                "quantized_ppl",
                "quantized_delta",
                "results_path",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sweep_dir": record.sweep_dir,
                    "run_name": record.run_name,
                    "gamma": record.gamma,
                    "seed": record.seed,
                    "baseline_ppl": record.baseline_ppl,
                    "quantized_ppl": record.quantized_ppl,
                    "quantized_delta": record.quantized_delta,
                    "results_path": record.results_path.as_posix(),
                }
            )


def write_summary_csv(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "gamma",
                "n_runs",
                "seeds",
                "ppl_mean",
                "ppl_std",
                "ppl_min",
                "ppl_max",
                "baseline_ppl_mean",
                "baseline_ppl_std",
                "quantized_delta_mean",
                "quantized_delta_std",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writable = dict(row)
            writable["seeds"] = ",".join(str(seed) for seed in row["seeds"])
            writer.writerow(writable)


def write_summary_json(path: Path, records: list[RunRecord], summary_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_row = min(summary_rows, key=lambda item: item["ppl_mean"])
    payload = {
        "input_sweep_dirs": sorted({record.sweep_dir for record in records}),
        "num_runs": len(records),
        "num_unique_gammas": len(summary_rows),
        "best_gamma_by_mean_ppl": best_row["gamma"],
        "best_mean_ppl": best_row["ppl_mean"],
        "records": [
            {
                "sweep_dir": record.sweep_dir,
                "run_name": record.run_name,
                "gamma": record.gamma,
                "seed": record.seed,
                "baseline_ppl": record.baseline_ppl,
                "quantized_ppl": record.quantized_ppl,
                "quantized_delta": record.quantized_delta,
                "results_path": record.results_path.as_posix(),
            }
            for record in records
        ],
        "summary_by_gamma": summary_rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def configure_x_axis(ax: plt.Axes, gammas: list[float], scale_mode: str) -> None:
    if scale_mode == "linear":
        ax.set_xscale("linear")
        return

    positive = [gamma for gamma in gammas if gamma > 0]
    linthresh = min(positive) / 2 if positive else 1.0
    ax.set_xscale("symlog", linthresh=linthresh)


def plot_seed_curves(ax: plt.Axes, records: list[RunRecord], color_map: dict[int | None, Any]) -> None:
    grouped: dict[int | None, list[RunRecord]] = {}
    for record in records:
        grouped.setdefault(record.seed, []).append(record)

    for seed in sorted(grouped, key=lambda value: (-1 if value is None else value)):
        seed_records = sorted(grouped[seed], key=lambda item: item.gamma)
        xs = [item.gamma for item in seed_records]
        ys = [item.quantized_ppl for item in seed_records]
        label = "seed=unknown" if seed is None else f"seed={seed}"
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.2,
            linestyle="--",
            alpha=0.8,
            markersize=4.5,
            color=color_map[seed],
            label=label,
        )


def plot_mean_band(ax: plt.Axes, summary_rows: list[dict[str, Any]]) -> None:
    gammas = [row["gamma"] for row in summary_rows]
    means = [row["ppl_mean"] for row in summary_rows]
    stds = [row["ppl_std"] for row in summary_rows]
    lower = [value - std for value, std in zip(means, stds)]
    upper = [value + std for value, std in zip(means, stds)]
    ax.fill_between(gammas, lower, upper, color="black", alpha=0.12, label="mean ± 1 std")
    ax.plot(gammas, means, color="black", linewidth=2.4, marker="o", markersize=6, label="mean quantized_ppl")


def add_baseline_line(ax: plt.Axes, records: list[RunRecord]) -> None:
    baseline_values = [record.baseline_ppl for record in records if record.baseline_ppl is not None]
    if not baseline_values:
        return
    baseline_mean = mean(baseline_values)
    ax.axhline(
        baseline_mean,
        color="#666666",
        linewidth=1.8,
        linestyle=":",
        label=f"baseline mean = {baseline_mean:.3f}",
    )


def configure_x_axis(ax: plt.Axes, gammas: list[float], scale_mode: str) -> None:
    if scale_mode == "linear":
        ax.set_xscale("linear")
        return

    positive = [gamma for gamma in gammas if gamma > 0]
    linthresh = min(positive) / 2 if positive else 1.0
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_xlim(left=0.0, right=max(gammas) * 1.05 if gammas else None)


def plot_mean_band(ax: plt.Axes, summary_rows: list[dict[str, Any]]) -> None:
    gammas = [row["gamma"] for row in summary_rows]
    means = [row["ppl_mean"] for row in summary_rows]
    stds = [row["ppl_std"] for row in summary_rows]
    lower = [value - std for value, std in zip(means, stds)]
    upper = [value + std for value, std in zip(means, stds)]
    ax.fill_between(gammas, lower, upper, color="black", alpha=0.12, label="mean +/- 1 std")
    ax.plot(gammas, means, color="black", linewidth=2.4, marker="o", markersize=6, label="mean quantized_ppl")


def render_plot(path: Path, records: list[RunRecord], summary_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gammas = [row["gamma"] for row in summary_rows]
    seeds = sorted({record.seed for record in records}, key=lambda value: (-1 if value is None else value))
    cmap = plt.get_cmap("tab10")
    color_map = {seed: cmap(index % 10) for index, seed in enumerate(seeds)}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, scale_mode, subtitle in zip(
        axes,
        ("linear", "symlog"),
        ("Linear x-axis", "Symlog x-axis"),
        strict=True,
    ):
        plot_seed_curves(ax, records, color_map)
        plot_mean_band(ax, summary_rows)
        add_baseline_line(ax, records)
        configure_x_axis(ax, gammas, scale_mode)
        ax.set_title(subtitle)
        ax.set_xlabel("gamma")
        ax.set_ylabel("PPL")
        ax.grid(True, which="both", alpha=0.25)

    best_row = min(summary_rows, key=lambda item: item["ppl_mean"])
    fig.suptitle(
        "QKVO gamma sweep: quantized PPL across seeds\n"
        f"best gamma = {best_row['gamma']:.0f}, mean PPL = {best_row['ppl_mean']:.3f}",
        fontsize=13,
        y=0.98,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(top=0.84, bottom=0.12, wspace=0.18)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(labels), 5),
        frameon=False,
        bbox_to_anchor=(0.5, 0.915),
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_text_summary(path: Path, records: list[RunRecord], summary_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_row = min(summary_rows, key=lambda item: item["ppl_mean"])
    baseline_values = [record.baseline_ppl for record in records if record.baseline_ppl is not None]
    baseline_mean = mean(baseline_values) if baseline_values else None
    lines = [
        "QKVO Gamma Sweep PPL Analysis",
        f"- input_sweep_dirs: {', '.join(sorted({record.sweep_dir for record in records}))}",
        f"- num_runs: {len(records)}",
        f"- num_unique_gammas: {len(summary_rows)}",
        f"- best_gamma_by_mean_ppl: {best_row['gamma']}",
        f"- best_mean_ppl: {best_row['ppl_mean']:.6f}",
    ]
    if baseline_mean is not None:
        lines.append(f"- overall_baseline_ppl_mean: {baseline_mean:.6f}")
    lines.extend(
        [
            "",
            "Per-gamma summary",
        ]
    )
    for row in summary_rows:
        std = row["ppl_std"]
        delta_mean = row["quantized_delta_mean"]
        baseline = row["baseline_ppl_mean"]
        lines.append(
            "- gamma={gamma:g}: mean={mean:.6f}, std={std:.6f}, min={min:.6f}, max={max:.6f}, "
            "baseline_mean={baseline}, delta_mean={delta}, seeds={seeds}".format(
                gamma=row["gamma"],
                mean=row["ppl_mean"],
                std=std,
                min=row["ppl_min"],
                max=row["ppl_max"],
                baseline="None" if baseline is None else f"{baseline:.6f}",
                delta="None" if delta_mean is None else f"{delta_mean:.6f}",
                seeds=row["seeds"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    output_dir = root / args.output_dir

    records = collect_records(root=root, sweep_dirs=args.sweep_dirs, results_file=args.results_file)
    summary_rows = summarize_by_gamma(records)

    write_raw_csv(output_dir / "gamma_ppl_runs.csv", records)
    write_summary_csv(output_dir / "gamma_ppl_summary.csv", summary_rows)
    write_summary_json(output_dir / "gamma_ppl_summary.json", records, summary_rows)
    write_text_summary(output_dir / "gamma_ppl_summary.txt", records, summary_rows)
    render_plot(output_dir / "gamma_vs_ppl_analysis.png", records, summary_rows)

    best_row = min(summary_rows, key=lambda item: item["ppl_mean"])
    print(f"Wrote analysis to: {output_dir}")
    print(
        "Best gamma by mean quantized PPL: "
        f"{best_row['gamma']:g} (mean PPL={best_row['ppl_mean']:.6f}, std={best_row['ppl_std']:.6f})"
    )


if __name__ == "__main__":
    main()
