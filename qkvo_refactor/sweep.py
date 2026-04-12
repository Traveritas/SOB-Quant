from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .cli import add_experiment_args
from .config import CODEBOOKS, ExperimentConfig, parse_block_indices, parse_codebook, parse_target_linear_names
from . import sweep_config as default_sweep_config

DEFAULT_SWEEP_GRID: Dict[str, Tuple[Any, ...]] = default_sweep_config.SWEEP_GRID


@dataclass
class SweepOutputOptions:
    save_manifest: bool = True
    save_summary_json: bool = True
    save_summary_text: bool = True
    save_ranking_text: bool = True
    save_run_combo_json: bool = True
    save_run_tracking_json: bool = True
    save_u_trace_plots: bool = False


@dataclass
class SweepSummaryRow:
    combo_index: int
    run_name: str
    run_dir: str
    combo: Dict[str, Any]
    baseline_ppl: float
    sq_baseline_ppl: float
    quantized_ppl: float
    quantized_delta: float
    sq_delta: float
    quant_metrics_avg: Dict[str, float]
    sq_metrics_avg: Dict[str, float]
    convergence_iters: Dict[str, int]
    fit_quantizer_sec_total: float
    objective_last: Dict[str, Optional[float]]
    tracking_last: Dict[str, Dict[str, Any]]
    tracking_length: Dict[str, int]


@dataclass
class SweepFileConfig:
    experiment_overrides: Dict[str, Any]
    grid: Dict[str, Iterable[Any]]
    tracking: Dict[str, Any]
    output_options: Dict[str, Any]
    run_control: Dict[str, Any]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_python_config_module(path: Path):
    spec = importlib.util.spec_from_file_location("qkvo_refactor_user_sweep_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sweep_file_config(config_path: Optional[str]) -> SweepFileConfig:
    if config_path is None:
        module = default_sweep_config
    else:
        path = Path(config_path)
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            return SweepFileConfig(
                experiment_overrides=dict(payload.get("EXPERIMENT_CONFIG_OVERRIDES", {})),
                grid=dict(payload.get("SWEEP_GRID", {})),
                tracking=dict(payload.get("TRACKING_CONFIG", {})),
                output_options=dict(payload.get("OUTPUT_OPTIONS", {})),
                run_control=dict(payload.get("RUN_CONTROL", {})),
            )
        module = load_python_config_module(path)

    return SweepFileConfig(
        experiment_overrides=dict(getattr(module, "EXPERIMENT_CONFIG_OVERRIDES", {})),
        grid=dict(getattr(module, "SWEEP_GRID", {})),
        tracking=dict(getattr(module, "TRACKING_CONFIG", {})),
        output_options=dict(getattr(module, "OUTPUT_OPTIONS", {})),
        run_control=dict(getattr(module, "RUN_CONTROL", {})),
    )


def setup_sweep_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"qkvo_refactor_sweep_{output_dir.resolve()}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(output_dir / "sweep.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


def set_dotted_attr(obj: Any, dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    current = obj
    for part in parts[:-1]:
        if not hasattr(current, part):
            raise AttributeError(f"Config has no field '{dotted_path}' (stopped at '{part}')")
        current = getattr(current, part)
    leaf = parts[-1]
    if not hasattr(current, leaf):
        raise AttributeError(f"Config has no field '{dotted_path}'")
    setattr(current, leaf, value)


def _ensure_iterable(name: str, values: Iterable[Any]) -> Tuple[Any, ...]:
    if isinstance(values, (str, bytes)):
        return (values,)
    try:
        seq = tuple(values)
    except TypeError as error:
        raise TypeError(f"Sweep values for {name!r} must be iterable") from error
    if not seq:
        raise ValueError(f"Sweep values for {name!r} cannot be empty")
    if name == "quant.codebook" and all(isinstance(value, (int, float)) for value in seq):
        return (tuple(float(value) for value in seq),)
    if name == "target.block_indices" and all(isinstance(value, int) for value in seq):
        return (tuple(int(value) for value in seq),)
    return seq


def normalize_grid_value(path: str, value: Any) -> Any:
    if path == "quant.codebook":
        if isinstance(value, str):
            return parse_codebook(value)
        return tuple(float(item) for item in value)
    if path == "target.block_indices":
        if value is None:
            return None
        if isinstance(value, str):
            return parse_block_indices(value)
        return tuple(int(item) for item in value)
    if path == "target.target_linear_names":
        if isinstance(value, str):
            return parse_target_linear_names(value)
        return tuple(str(item) for item in value)
    if path == "output_dir":
        return str(value)
    return value


def normalize_override_value(path: str, value: Any) -> Any:
    if path == "quant.codebook":
        if isinstance(value, str):
            return parse_codebook(value)
        return tuple(float(item) for item in value)
    if path == "target.block_indices":
        if value is None:
            return None
        if isinstance(value, str):
            return parse_block_indices(value)
        return tuple(int(item) for item in value)
    if path == "target.target_linear_names":
        if isinstance(value, str):
            return parse_target_linear_names(value)
        return tuple(str(item) for item in value)
    return value


def load_grid(args: argparse.Namespace, file_config: SweepFileConfig) -> List[Tuple[str, Tuple[Any, ...]]]:
    if args.grid_json and args.grid_file:
        raise ValueError("Use either --grid-json or --grid-file, not both.")

    if args.grid_file:
        payload = json.loads(Path(args.grid_file).read_text(encoding="utf-8"))
    elif args.grid_json:
        payload = json.loads(args.grid_json)
    else:
        payload = {key: list(values) for key, values in file_config.grid.items()}

    if not isinstance(payload, dict):
        raise ValueError("Sweep grid must be a JSON object: {dotted.path: [values...]}")

    grid_items: List[Tuple[str, Tuple[Any, ...]]] = []
    for path, values in payload.items():
        seq = _ensure_iterable(path, values)
        grid_items.append((path, tuple(normalize_grid_value(path, value) for value in seq)))
    return grid_items


def sanitize_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "t" if value else "f"
    if isinstance(value, float):
        text = f"{value:.6g}"
    elif isinstance(value, (tuple, list)):
        text = "-".join(sanitize_value(item) for item in value)
    else:
        text = str(value)
    for old, new in [("/", "-"), (" ", ""), ("(", ""), (")", ""), (",", "-"), ("'", ""), (".", "p")]:
        text = text.replace(old, new)
    return text


def block_indices_tag(value: Any) -> str:
    if value is None:
        return "all"
    seq = tuple(int(item) for item in value)
    if not seq:
        return "none"
    if len(seq) == 1:
        return f"b{seq[0]}"
    if all(seq[index] + 1 == seq[index + 1] for index in range(len(seq) - 1)):
        return f"b{seq[0]}-{seq[-1]}"
    return "b" + "-".join(str(item) for item in seq)


def codebook_tag_from_values(codebook: Tuple[float, ...]) -> str:
    for key, values in CODEBOOKS.items():
        if tuple(values) == tuple(codebook):
            return key
    preview = "-".join(f"{item:.3g}" for item in codebook[:4])
    return f"c{len(codebook)}_{sanitize_value(preview)}"


def combo_to_name(combo: Dict[str, Any]) -> str:
    parts: List[str] = []
    for path, value in combo.items():
        if path == "target.block_indices":
            parts.append(block_indices_tag(value))
        elif path == "quant.codebook":
            parts.append(f"codebook_{codebook_tag_from_values(tuple(value))}")
        else:
            short = (
                path.replace("quant.", "q_")
                .replace("target.", "t_")
                .replace("data.", "d_")
                .replace("eval.", "e_")
                .replace(".", "_")
            )
            parts.append(f"{short}_{sanitize_value(value)}")
    return "__".join(parts)


def enumerate_runs(grid_items: List[Tuple[str, Tuple[Any, ...]]]) -> List[Tuple[int, Dict[str, Any], str]]:
    if not grid_items:
        return [(0, {}, "default")]
    keys = [key for key, _ in grid_items]
    values = [values for _, values in grid_items]
    runs: List[Tuple[int, Dict[str, Any], str]] = []
    for combo_index, values_combo in enumerate(itertools.product(*values)):
        combo = dict(zip(keys, values_combo))
        runs.append((combo_index, combo, combo_to_name(combo)))
    return runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parameter sweep runner for qkvo_refactor.")
    add_experiment_args(parser)
    parser.add_argument("--config", default=None, help="Python or JSON config file. Defaults to qkvo_refactor/sweep_config.py")
    parser.add_argument("--grid-json", default=None, help="JSON object of dotted config path -> list of values.")
    parser.add_argument("--grid-file", default=None, help="Path to a JSON file containing the sweep grid.")
    parser.add_argument("--run-index", type=int, default=None, help="Run only one combo index from the sweep.")
    parser.add_argument("--max-runs", type=int, default=None, help="Execute only the first N selected runs.")
    parser.add_argument("--list-runs", action="store_true", help="List the resolved run names without executing them.")
    parser.add_argument("--print-num-combos", action="store_true", help="Print the number of resolved sweep combinations and exit.")
    parser.add_argument("--array-task-id", type=int, default=None, help="Run exactly one combo using the given array task id.")
    parser.add_argument("--rebuild-summary", action="store_true", help="Rebuild summary files from existing run outputs and exit.")

    parser.add_argument("--track-u", action="store_true", help="Record U trajectory during quantizer fitting.")
    parser.add_argument("--track-u-every", type=int, default=1)
    parser.add_argument("--track-u-full-matrix", action="store_true", help="Save selected U matrices to disk.")
    parser.add_argument("--track-u-save-interval", type=int, default=10)
    parser.add_argument("--track-u-save-first", type=int, default=5)
    parser.add_argument("--no-track-z-flip-stats", action="store_true")
    parser.add_argument("--save-u-trace-plots", action="store_true", help="Save plots for tracked U metrics.")

    parser.add_argument("--no-save-manifest", action="store_true")
    parser.add_argument("--no-save-summary-json", action="store_true")
    parser.add_argument("--no-save-summary-text", action="store_true")
    parser.add_argument("--no-save-ranking-text", action="store_true")
    parser.add_argument("--no-save-run-combo-json", action="store_true")
    parser.add_argument("--no-save-run-tracking-json", action="store_true")
    return parser


def build_output_options(args: argparse.Namespace) -> SweepOutputOptions:
    return SweepOutputOptions(
        save_manifest=not args.no_save_manifest,
        save_summary_json=not args.no_save_summary_json,
        save_summary_text=not args.no_save_summary_text,
        save_ranking_text=not args.no_save_ranking_text,
        save_run_combo_json=not args.no_save_run_combo_json,
        save_run_tracking_json=not args.no_save_run_tracking_json,
        save_u_trace_plots=args.save_u_trace_plots,
    )


def build_tracking_options(args: argparse.Namespace, file_config: SweepFileConfig):
    from .quantizer import QuantizerTrackingOptions

    tracking = {
        "track_u": file_config.tracking.get("track_u", False),
        "track_u_every": file_config.tracking.get("track_u_every", 1),
        "track_u_full_matrix": file_config.tracking.get("track_u_full_matrix", False),
        "track_u_save_interval": file_config.tracking.get("track_u_save_interval", 10),
        "track_u_save_first": file_config.tracking.get("track_u_save_first", 5),
        "track_z_flip_stats": file_config.tracking.get("track_z_flip_stats", True),
    }
    if args.track_u:
        tracking["track_u"] = True
    if args.track_u_every != 1:
        tracking["track_u_every"] = args.track_u_every
    if args.track_u_full_matrix:
        tracking["track_u_full_matrix"] = True
    if args.track_u_save_interval != 10:
        tracking["track_u_save_interval"] = args.track_u_save_interval
    if args.track_u_save_first != 5:
        tracking["track_u_save_first"] = args.track_u_save_first
    if args.no_track_z_flip_stats:
        tracking["track_z_flip_stats"] = False

    return QuantizerTrackingOptions(
        track_u=tracking["track_u"],
        track_u_every=tracking["track_u_every"],
        track_u_full_matrix=tracking["track_u_full_matrix"],
        track_u_save_interval=tracking["track_u_save_interval"],
        track_u_save_first=tracking["track_u_save_first"],
        track_z_flip_stats=tracking["track_z_flip_stats"],
    )


def extract_tracking_summary(tracking_info: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    tracking_last: Dict[str, Dict[str, Any]] = {}
    tracking_length: Dict[str, int] = {}
    for layer_name, layer_tracking in tracking_info.items():
        u_trace = layer_tracking.get("u_trace", [])
        tracking_length[layer_name] = len(u_trace)
        if u_trace:
            tracking_last[layer_name] = u_trace[-1]
    return tracking_last, tracking_length


def save_u_trace_plots(output_dir: Path, layer_name: str, u_trace: List[Dict[str, Any]]) -> Dict[str, str]:
    if not u_trace:
        return {}

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = layer_name.replace(".", "_")
    xs = [int(point["iteration"]) for point in u_trace]
    paths: Dict[str, str] = {}

    def plot_metric(key: str, ylabel: str, title: str, filename: str) -> None:
        ys = [float(point[key]) for point in u_trace]
        plt.figure(figsize=(8, 5))
        plt.plot(xs, ys, marker="o", linewidth=1.5)
        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        path = output_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths[key] = str(path)

    specs = [
        ("delta_fro_from_prev", "Frobenius delta", f"{layer_name} U change vs previous", f"{prefix}_u_prev.png"),
        ("delta_fro_from_init", "Frobenius delta", f"{layer_name} U change vs init", f"{prefix}_u_init.png"),
        ("orthogonality_error", "Orthogonality error", f"{layer_name} U orthogonality error", f"{prefix}_ortho.png"),
        ("principal_angle_mean_prev_deg", "Degrees", f"{layer_name} mean principal angle vs previous", f"{prefix}_angle_prev.png"),
        ("principal_angle_mean_init_deg", "Degrees", f"{layer_name} mean principal angle vs init", f"{prefix}_angle_init.png"),
        ("z_x_flip_rate", "Flip rate", f"{layer_name} Z_x flip rate", f"{prefix}_zx_flip.png"),
        ("z_w_flip_rate", "Flip rate", f"{layer_name} Z_w flip rate", f"{prefix}_zw_flip.png"),
        ("lambda_x_delta_from_prev", "L2 delta", f"{layer_name} lambda_x delta", f"{prefix}_lambda_x.png"),
        ("lambda_w_delta_from_prev", "L2 delta", f"{layer_name} lambda_w delta", f"{prefix}_lambda_w.png"),
    ]
    for spec in specs:
        plot_metric(*spec)
    return paths


def save_run_metadata(
    run_dir: Path,
    combo_index: int,
    run_name: str,
    combo: Dict[str, Any],
    artifacts,
    output_options: SweepOutputOptions,
) -> None:
    if output_options.save_run_combo_json:
        write_json(
            run_dir / "sweep_combo.json",
            {
                "combo_index": combo_index,
                "run_name": run_name,
                "combo": jsonable(combo),
                "config": artifacts.config,
            },
        )

    tracking_last, tracking_length = extract_tracking_summary(artifacts.tracking_info)
    tracking_plot_paths: Dict[str, Dict[str, str]] = {}
    if output_options.save_u_trace_plots:
        plots_root = run_dir / "tracking_plots"
        for layer_name, layer_tracking in artifacts.tracking_info.items():
            tracking_plot_paths[layer_name] = save_u_trace_plots(
                plots_root,
                layer_name,
                layer_tracking.get("u_trace", []),
            )

    if output_options.save_run_tracking_json:
        write_json(
            run_dir / "tracking_summary.json",
            {
                "tracking_last": tracking_last,
                "tracking_length": tracking_length,
                "tracking_plot_paths": tracking_plot_paths,
                "tracking_info": artifacts.tracking_info,
            },
        )


def build_summary_row(combo_index: int, run_name: str, run_dir: Path, combo: Dict[str, Any], artifacts) -> SweepSummaryRow:
    tracking_last, tracking_length = extract_tracking_summary(artifacts.tracking_info)
    objective_last = {
        layer_name: (history[-1] if history else None)
        for layer_name, history in artifacts.objective_histories.items()
    }
    sq_delta = float("nan")
    if not math.isnan(artifacts.sq_baseline_ppl):
        sq_delta = artifacts.sq_baseline_ppl - artifacts.baseline_ppl

    return SweepSummaryRow(
        combo_index=combo_index,
        run_name=run_name,
        run_dir=str(run_dir),
        combo=jsonable(combo),
        baseline_ppl=artifacts.baseline_ppl,
        sq_baseline_ppl=artifacts.sq_baseline_ppl,
        quantized_ppl=artifacts.quantized_ppl,
        quantized_delta=artifacts.quantized_ppl - artifacts.baseline_ppl,
        sq_delta=sq_delta,
        quant_metrics_avg=artifacts.quant_metrics_avg,
        sq_metrics_avg=artifacts.sq_metrics_avg,
        convergence_iters=artifacts.convergence_iters,
        fit_quantizer_sec_total=float(artifacts.timing_info.get("fit_quantizer_sec_total", 0.0)),
        objective_last=objective_last,
        tracking_last=tracking_last,
        tracking_length=tracking_length,
    )


def select_runs(
    runs: List[Tuple[int, Dict[str, Any], str]],
    run_index: Optional[int],
    max_runs: Optional[int],
) -> List[Tuple[int, Dict[str, Any], str]]:
    if run_index is not None:
        runs = [item for item in runs if item[0] == run_index]
    if max_runs is not None:
        runs = runs[:max_runs]
    return runs


def build_manifest_payload(
    base_config: ExperimentConfig,
    grid_items: List[Tuple[str, Tuple[Any, ...]]],
    runs: List[Tuple[int, Dict[str, Any], str]],
) -> Dict[str, Any]:
    return {
        "base_config": jsonable(asdict(base_config)),
        "grid": {path: jsonable(values) for path, values in grid_items},
        "runs": [
            {
                "combo_index": combo_index,
                "run_name": run_name,
                "combo": jsonable(combo),
            }
            for combo_index, combo, run_name in runs
        ],
    }


def build_summary_text(rows: List[SweepSummaryRow]) -> str:
    lines = [
        "QKVO Sweep Summary",
        f"completed_runs={len(rows)}",
        "",
    ]
    for row in rows:
        lines.append(
            f"[{row.combo_index:03d}] {row.run_name} | baseline={row.baseline_ppl:.6f} | "
            f"quantized={row.quantized_ppl:.6f} | delta={row.quantized_delta:.6f}"
        )
        if not math.isnan(row.sq_baseline_ppl):
            lines.append(f"sq={row.sq_baseline_ppl:.6f} | sq_delta={row.sq_delta:.6f}")
        lines.append(f"combo={json.dumps(row.combo, ensure_ascii=False)}")
        if row.tracking_length:
            lines.append(f"tracking_length={json.dumps(row.tracking_length, ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_ranking_text(rows: List[SweepSummaryRow]) -> str:
    ranked = sorted(rows, key=lambda row: (row.quantized_ppl, row.quantized_delta))
    lines = [
        "QKVO Sweep Ranking",
        "",
    ]
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"{rank:02d}. [{row.combo_index:03d}] {row.run_name} | "
            f"quantized={row.quantized_ppl:.6f} | delta={row.quantized_delta:.6f}"
        )
    return "\n".join(lines) + "\n"


def list_runs(runs: List[Tuple[int, Dict[str, Any], str]]) -> None:
    for combo_index, combo, run_name in runs:
        print(f"[{combo_index:03d}] {run_name} {json.dumps(jsonable(combo), ensure_ascii=False)}")


def apply_combo(base_config: ExperimentConfig, combo: Dict[str, Any], run_dir: Path) -> ExperimentConfig:
    config = copy.deepcopy(base_config)
    for path, value in combo.items():
        set_dotted_attr(config, path, value)
    config.output_dir = str(run_dir)
    return config


def build_base_config(args: argparse.Namespace, file_config: SweepFileConfig) -> ExperimentConfig:
    config = ExperimentConfig()
    for path, value in file_config.experiment_overrides.items():
        set_dotted_attr(config, path, normalize_override_value(path, value))

    cli_override_fields = {
        "model_name": ("data.model_name", args.model_name, "facebook/opt-125m"),
        "dataset_name": ("data.dataset_name", args.dataset_name, "wikitext"),
        "dataset_config": ("data.dataset_config", args.dataset_config, "wikitext-2-raw-v1"),
        "calib_split": ("data.calib_split", args.calib_split, "train"),
        "eval_split": ("data.eval_split", args.eval_split, "test"),
        "calib_num_tokens": ("data.calib_num_tokens", args.calib_num_tokens, 4096),
        "eval_num_tokens": ("data.eval_num_tokens", args.eval_num_tokens, None),
        "block_indices": ("target.block_indices", parse_block_indices(args.block_indices), (8, 9, 10, 11)),
        "target_linear_names": ("target.target_linear_names", parse_target_linear_names(args.target_linear_names), ("q_proj", "k_proj", "v_proj", "out_proj")),
        "beta": ("quant.beta", args.beta, 1.0),
        "max_iters": ("quant.max_iters", args.max_iters, 80),
        "tol": ("quant.tol", args.tol, 1e-5),
        "codebook": ("quant.codebook", parse_codebook(args.codebook), parse_codebook("d5")),
        "dtype": ("quant.dtype", args.dtype, "float32"),
        "init_mode": ("quant.init_mode", args.init_mode, "random"),
        "error_mode": ("quant.error_mode", args.error_mode, "relative"),
        "latent_mode": ("quant.latent_mode", args.latent_mode, "discrete"),
        "ip_reg_gamma": ("quant.ip_reg_gamma", args.ip_reg_gamma, 0.0),
        "ip_reg_inner_iters": ("quant.ip_reg_inner_iters", args.ip_reg_inner_iters, 1),
        "fit_device": ("quant.fit_device", args.fit_device, "cpu"),
        "stride": ("eval.stride", args.stride, 512),
        "output_dir": ("output_dir", args.output_dir, "./qkvo_refactor_outputs"),
        "seed": ("seed", args.seed, 42),
        "run_sq_baseline": ("run_sq_baseline", not args.skip_sq_baseline, True),
        "save_plots": ("save_plots", not args.no_plots, True),
    }
    for _, (path, value, default) in cli_override_fields.items():
        if value != default:
            set_dotted_attr(config, path, value)

    if args.device is not None:
        set_dotted_attr(config, "eval.device", args.device)
    return config


def find_array_task_id(args: argparse.Namespace) -> Optional[int]:
    if args.array_task_id is not None:
        return args.array_task_id
    env_value = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_value is None or env_value == "":
        return None
    return int(env_value)


def build_row_from_saved_run(run_dir: Path) -> SweepSummaryRow:
    results = read_json(run_dir / "results.json")

    combo_payload: Dict[str, Any] = {}
    combo_path = run_dir / "sweep_combo.json"
    if combo_path.exists():
        combo_payload = read_json(combo_path)

    tracking_payload: Dict[str, Any] = {}
    tracking_path = run_dir / "tracking_summary.json"
    if tracking_path.exists():
        tracking_payload = read_json(tracking_path)

    combo_index = int(combo_payload.get("combo_index", -1))
    run_name = str(combo_payload.get("run_name", run_dir.name))
    combo = combo_payload.get("combo", {})
    sq_baseline_ppl = float(results.get("sq_baseline_ppl", float("nan")))
    baseline_ppl = float(results["baseline_ppl"])
    quantized_ppl = float(results["quantized_ppl"])
    sq_delta = float("nan") if math.isnan(sq_baseline_ppl) else sq_baseline_ppl - baseline_ppl

    objective_last = {
        layer_name: (history[-1] if history else None)
        for layer_name, history in results.get("objective_histories", {}).items()
    }
    tracking_last = tracking_payload.get("tracking_last", {})
    tracking_length = tracking_payload.get("tracking_length", {})
    if not tracking_last or not tracking_length:
        tracking_info = results.get("tracking_info", {})
        tracking_last, tracking_length = extract_tracking_summary(tracking_info)

    return SweepSummaryRow(
        combo_index=combo_index,
        run_name=run_name,
        run_dir=str(run_dir),
        combo=combo,
        baseline_ppl=baseline_ppl,
        sq_baseline_ppl=sq_baseline_ppl,
        quantized_ppl=quantized_ppl,
        quantized_delta=quantized_ppl - baseline_ppl,
        sq_delta=sq_delta,
        quant_metrics_avg=results.get("quant_metrics_avg", {}),
        sq_metrics_avg=results.get("sq_metrics_avg", {}),
        convergence_iters=results.get("convergence_iters", {}),
        fit_quantizer_sec_total=float(results.get("timing_info", {}).get("fit_quantizer_sec_total", 0.0)),
        objective_last=objective_last,
        tracking_last=tracking_last,
        tracking_length=tracking_length,
    )


def rebuild_summary_from_disk(output_dir: Path, logger: Optional[logging.Logger] = None) -> List[SweepSummaryRow]:
    rows: List[SweepSummaryRow] = []
    for child in sorted(output_dir.iterdir()):
        if not child.is_dir():
            continue
        results_path = child / "results.json"
        if not results_path.exists():
            continue
        try:
            rows.append(build_row_from_saved_run(child))
        except Exception as error:
            if logger is not None:
                logger.warning("Skip run dir %s while rebuilding summary: %s", child, error)

    rows = sorted(rows, key=lambda row: (row.combo_index, row.run_name))
    write_json(output_dir / "summary.json", [asdict(row) for row in rows])
    write_text(output_dir / "summary.txt", build_summary_text(rows))
    write_text(output_dir / "ranking.txt", build_ranking_text(rows))
    return rows


def run_sweep(args: argparse.Namespace) -> List[SweepSummaryRow]:
    file_config = load_sweep_file_config(args.config)
    base_config = build_base_config(args, file_config)
    output_dir = Path(base_config.output_dir)
    logger = setup_sweep_logger(output_dir)

    grid_items = load_grid(args, file_config)
    all_runs = enumerate_runs(grid_items)
    array_task_id = find_array_task_id(args)
    if args.print_num_combos:
        print(len(all_runs))
        return []

    if args.rebuild_summary:
        rows = rebuild_summary_from_disk(output_dir, logger=logger)
        logger.info("Rebuilt summary from disk | completed_runs=%d", len(rows))
        return rows

    run_index = args.run_index if args.run_index is not None else file_config.run_control.get("run_index")
    if array_task_id is not None:
        run_index = array_task_id
    max_runs = args.max_runs if args.max_runs is not None else file_config.run_control.get("max_runs")
    if array_task_id is not None:
        max_runs = None
    selected_runs = select_runs(all_runs, run_index, max_runs)

    if args.list_runs:
        list_runs(selected_runs)
        return []

    if not selected_runs:
        raise ValueError("No runs selected. Check --run-index / --max-runs / grid settings.")

    output_options = build_output_options(args)
    for key, value in file_config.output_options.items():
        if hasattr(output_options, key):
            setattr(output_options, key, value)
    if args.save_u_trace_plots:
        output_options.save_u_trace_plots = True
    if args.no_save_manifest:
        output_options.save_manifest = False
    if args.no_save_summary_json:
        output_options.save_summary_json = False
    if args.no_save_summary_text:
        output_options.save_summary_text = False
    if args.no_save_ranking_text:
        output_options.save_ranking_text = False
    if args.no_save_run_combo_json:
        output_options.save_run_combo_json = False
    if args.no_save_run_tracking_json:
        output_options.save_run_tracking_json = False

    tracking_options = build_tracking_options(args, file_config)

    if output_options.save_manifest:
        write_json(output_dir / "manifest.json", build_manifest_payload(base_config, grid_items, all_runs))

    from .experiment import run_experiment

    rows: List[SweepSummaryRow] = []
    logger.info("Sweep start | total_runs=%d selected_runs=%d", len(all_runs), len(selected_runs))
    for combo_index, combo, run_name in selected_runs:
        run_dir = output_dir / f"{combo_index:03d}_{run_name}"
        logger.info("Run start | index=%d name=%s", combo_index, run_name)
        logger.info("Combo: %s", json.dumps(jsonable(combo), ensure_ascii=False))
        config = apply_combo(base_config, combo, run_dir)
        artifacts = run_experiment(config, tracking_options=tracking_options)
        save_run_metadata(run_dir, combo_index, run_name, combo, artifacts, output_options)
        row = build_summary_row(combo_index, run_name, run_dir, combo, artifacts)
        rows.append(row)
        logger.info(
            "Run done | index=%d quantized_ppl=%.6f delta=%.6f",
            combo_index,
            row.quantized_ppl,
            row.quantized_delta,
        )

    if output_options.save_summary_json:
        write_json(output_dir / "summary.json", [asdict(row) for row in rows])
    if output_options.save_summary_text:
        write_text(output_dir / "summary.txt", build_summary_text(rows))
    if output_options.save_ranking_text:
        write_text(output_dir / "ranking.txt", build_ranking_text(rows))

    logger.info("Sweep end | completed_runs=%d", len(rows))
    return rows


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rows = run_sweep(args)
    if rows:
        print(json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
