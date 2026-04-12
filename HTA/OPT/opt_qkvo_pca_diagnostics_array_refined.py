from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ============================================================
# User config area
# ============================================================

BASE_SCRIPT_PATH = Path(__file__).with_name("opt_all_blocks_qkvo_experiment_v1.py")
OUTPUT_ROOT = Path("./out_qkvo_pca_diag_array")

CUSTOM_CODEBOOKS: Dict[str, Tuple[float, ...]] = {
    "d5": (-2.0, -1.0, 0.0, 1.0, 2.0),
    "t3": (-1.0, 0.0, 1.0),
    "s7": (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
    "s8": (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
    "4b": (-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
    "2b": (-2.0, -1.0, 0.0, 1.0),
    "3b": (-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0),
    "4b2": (-16.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
}
DEFAULT_CODEBOOK_SPEC: Any = "s8"

# ------------------------------------------------------------
# Sweep config: 平时只改这里
# ------------------------------------------------------------
SWEEP_GRID: Dict[str, Iterable[Any]] = {
    "target.block_indices": (
        (11,),
        (11,10,9,8,),
        (11,10,9,8,7,6,5,4,3,2,1,0,),
    ),
    "quant.beta": (1.0,),
    "quant_ext.codebook_spec": ("s8",),
}

# ------------------------------------------------------------
# Diagnostic settings: 放在 sweep 旁边，和 sweep 一样通过改代码控制
# 不需要改 sbatch 脚本。
# ------------------------------------------------------------
DIAGNOSTIC_USER_CONFIG: Dict[str, Any] = {
    "target.target_linear_names": ("q_proj", "k_proj", "v_proj", "out_proj"),
    "data.calib_num_tokens": 4096,
    "data.eval_num_tokens": None,
    "eval.stride": 512,
    "quant.max_iters": 80,
    "quant.tol": 1e-5,
    "quant.convergence_check_every": 1,
    "quant.log_every": 1,
    "quant.dtype": "float32",
    "quant.eps": 1e-8,
    "quant.error_mode": "relative",
    "diag.eval_metric_num_tokens": 4096,
    "diag.lambda_track_iters": 10,
    "diag.szz_zero_tol": 1e-10,
    "diag.lambda_near_zero_tol": 1e-8,
    "diag.save_plots": True,
    "diag.run_ppl": True,
    "diag.warm_start_mode": "weighted_rms",
}

DIAGNOSTIC_VARIANTS_RAW: Tuple[Tuple[str, str, bool], ...] = (
    ("random", "random", False),
    ("pca", "pca", False),
    ("pca_warm", "pca", True),
)

BASE_EXPERIMENT_CONFIG_OVERRIDES: Dict[str, Any] = {
    "target.block_indices": (11,),
    "quant.beta": 1.0,
    "quant.codebook": CUSTOM_CODEBOOKS[DEFAULT_CODEBOOK_SPEC],
    **DIAGNOSTIC_USER_CONFIG,
}

LOG_NAME = "diag.log"
RUN_JSON_NAME = "diagnostic_results.json"
RUN_SUMMARY_NAME = "diagnostic_summary.txt"
SWEEP_JSON_NAME = "sum.json"
SWEEP_TEXT_NAME = "sum.txt"
SWEEP_MANIFEST_NAME = "manifest.json"


# ============================================================
# Base module loading
# ============================================================


def load_base_module(script_path: Path):
    import sys

    module_name = "opt_all_blocks_qkvo_experiment_v1"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import base script from: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module(BASE_SCRIPT_PATH)


# ============================================================
# Config dataclasses
# ============================================================


@dataclass(frozen=True)
class DiagnosticVariantConfig:
    name: str
    init_mode: str
    lambda_warm_start: bool = False


DEFAULT_VARIANTS: Tuple[DiagnosticVariantConfig, ...] = tuple(
    DiagnosticVariantConfig(name=name, init_mode=init_mode, lambda_warm_start=lambda_warm_start)
    for name, init_mode, lambda_warm_start in DIAGNOSTIC_VARIANTS_RAW
)


@dataclass
class DiagnosticConfig:
    eval_metric_num_tokens: int = 4096
    lambda_track_iters: int = 10
    szz_zero_tol: float = 1e-10
    lambda_near_zero_tol: float = 1e-8
    save_plots: bool = True
    run_ppl: bool = True
    warm_start_mode: str = "weighted_rms"   # weighted_rms | rms
    variants: Tuple[DiagnosticVariantConfig, ...] = DEFAULT_VARIANTS


@dataclass
class ExperimentConfig(base.ExperimentConfig):
    diag: DiagnosticConfig = field(default_factory=DiagnosticConfig)


# ============================================================
# Result dataclasses
# ============================================================


@dataclass
class DiagnosticQuantizationState(base.QuantizationState):
    init_mode: str = "pca"
    lambda_warm_start: bool = False
    first_round: Dict[str, Any] = field(default_factory=dict)
    lambda_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class VariantRunArtifacts:
    variant_name: str
    init_mode: str
    lambda_warm_start: bool
    quantized_ppl: Optional[float]
    ppl_eval_info: Dict[str, float]
    per_layer_metrics: Dict[str, Dict[str, float]]
    per_layer_first_round: Dict[str, Dict[str, Any]]
    per_layer_lambda_history: Dict[str, List[Dict[str, Any]]]
    convergence_iters: Dict[str, int]
    fit_time_sec_total: float


@dataclass
class DiagnosticArtifacts:
    config: Dict[str, Any]
    combo: Dict[str, Any]
    combo_resolved: Dict[str, Any]
    codebook: Tuple[float, ...]
    codebook_tag: str
    baseline_ppl: Optional[float]
    baseline_ppl_eval_info: Dict[str, float]
    target_info: Dict[str, Any]
    variants: List[VariantRunArtifacts]


@dataclass
class SweepSummaryRow:
    combo_index: int
    run_name: str
    run_dir: str
    combo: Dict[str, Any]
    combo_resolved: Dict[str, Any]
    baseline_ppl: Optional[float]
    variant_ppl: Dict[str, Optional[float]]
    variant_avg_metrics: Dict[str, Dict[str, float]]
    variant_fit_time_sec_total: Dict[str, float]


# ============================================================
# Generic utilities
# ============================================================


def _ensure_iterable(name: str, values: Iterable[Any]) -> Tuple[Any, ...]:
    if isinstance(values, (str, bytes)):
        return (values,)
    try:
        seq = tuple(values)
    except TypeError as e:
        raise TypeError(f"Sweep values for {name!r} must be iterable") from e
    if not seq:
        raise ValueError(f"Sweep values for {name!r} cannot be empty")
    if name == "quant_ext.codebook_spec" and all(isinstance(v, (int, float)) for v in seq):
        return (tuple(float(v) for v in seq),)
    return seq


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}_{time.time_ns()}")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}_{time.time_ns()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def build_sweep_grid_items() -> List[Tuple[str, Tuple[Any, ...]]]:
    return [(k, _ensure_iterable(k, v)) for k, v in SWEEP_GRID.items()]


def enumerate_sweep_runs() -> List[Tuple[int, Dict[str, Any], str]]:
    grid_items = build_sweep_grid_items()
    keys = [k for k, _ in grid_items]
    values = [v for _, v in grid_items]
    runs: List[Tuple[int, Dict[str, Any], str]] = []
    for combo_index, values_combo in enumerate(itertools.product(*values)):
        combo = dict(zip(keys, values_combo))
        run_name = combo_to_name(combo)
        runs.append((combo_index, combo, run_name))
    return runs


def set_dotted_attr(obj: Any, dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    cur = obj
    for p in parts[:-1]:
        if not hasattr(cur, p):
            raise AttributeError(f"Config has no field '{dotted_path}' (stopped at '{p}')")
        cur = getattr(cur, p)
    leaf = parts[-1]
    if not hasattr(cur, leaf):
        raise AttributeError(f"Config has no field '{dotted_path}'")
    setattr(cur, leaf, value)


def sanitize_value(v: Any) -> str:
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "t" if v else "f"
    if isinstance(v, float):
        s = f"{v:.6g}"
    else:
        s = str(v)
    for old, new in [("/", "-"), (" ", ""), ("(", ""), (")", ""), (",", "-"), ("'", ""), (".", "p")]:
        s = s.replace(old, new)
    return s


def normalize_codebook(values: Iterable[Any]) -> Tuple[float, ...]:
    codebook = tuple(sorted(float(v) for v in values))
    if not codebook:
        raise ValueError("Custom codebook cannot be empty")
    if len(set(codebook)) != len(codebook):
        raise ValueError(f"Codebook contains duplicated values: {codebook}")
    for v in codebook:
        if not math.isfinite(v):
            raise ValueError(f"Codebook contains non-finite value: {v}")
    return codebook


def codebook_tag_from_values(codebook: Tuple[float, ...]) -> str:
    for k, v in CUSTOM_CODEBOOKS.items():
        if tuple(v) == tuple(codebook):
            return k
    preview = "-".join(f"{x:.3g}" for x in codebook[:4])
    return f"c{len(codebook)}_{sanitize_value(preview)}"


def resolve_codebook_spec(spec: Any) -> Tuple[Tuple[float, ...], str]:
    if spec is None:
        spec = DEFAULT_CODEBOOK_SPEC
    if isinstance(spec, str):
        if spec not in CUSTOM_CODEBOOKS:
            raise KeyError(f"Unknown codebook key: {spec}. Available: {sorted(CUSTOM_CODEBOOKS)}")
        codebook = normalize_codebook(CUSTOM_CODEBOOKS[spec])
        return codebook, spec
    codebook = normalize_codebook(spec)
    return codebook, codebook_tag_from_values(codebook)


def block_indices_tag(v: Any) -> str:
    if v is None:
        return "all"
    seq = tuple(int(x) for x in v)
    if not seq:
        return "none"
    if len(seq) == 1:
        return f"b{seq[0]}"
    if all(seq[i] + 1 == seq[i + 1] for i in range(len(seq) - 1)):
        return f"b{seq[0]}-{seq[-1]}"
    return "b" + "-".join(str(x) for x in seq)


def combo_to_name(combo: Dict[str, Any]) -> str:
    pieces: List[str] = []
    for k, v in combo.items():
        if k == "target.block_indices":
            pieces.append(block_indices_tag(v))
        elif k == "quant.beta":
            pieces.append(f"bt{sanitize_value(v)}")
        elif k == "quant_ext.codebook_spec":
            _, tag = resolve_codebook_spec(v)
            pieces.append(f"z{tag}")
        else:
            pieces.append(f"{k.replace('quant_ext.', '').replace('quant.', 'q').replace('target.', 't').replace('data.', 'd')}{sanitize_value(v)}")
    return "__".join(pieces)


def layer_tag(layer_name: str) -> str:
    block_part, proj_part = layer_name.split('.')
    block_idx = int(block_part.replace('block', ''))
    proj_map = {'q_proj': 'q', 'k_proj': 'k', 'v_proj': 'v', 'out_proj': 'o'}
    return f"b{block_idx}_{proj_map.get(proj_part, proj_part)}"


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"qkvo_pca_diag_array_{output_dir.resolve()}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(output_dir / LOG_NAME, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def vector_stats(x: torch.Tensor, near_zero_tol: float) -> Dict[str, float]:
    x = x.detach().float()
    abs_x = x.abs()
    return {
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "mean_abs": float(abs_x.mean().item()),
        "max_abs": float(abs_x.max().item()),
        "near_zero_frac": float((abs_x <= near_zero_tol).float().mean().item()),
        "negative_frac": float((x < 0).float().mean().item()),
    }


def codebook_hist(Z: torch.Tensor, codebook: torch.Tensor) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in codebook.detach().cpu().tolist():
        key = sanitize_value(float(v))
        out[key] = int((Z == float(v)).sum().item())
    return out


def weighted_rms(values: torch.Tensor, weights: Optional[torch.Tensor], eps: float) -> torch.Tensor:
    if weights is None:
        return torch.sqrt(torch.clamp(torch.mean(values * values, dim=1), min=eps))
    w = weights.to(device=values.device, dtype=values.dtype)
    denom = torch.clamp(torch.sum(w), min=eps)
    mean_sq = torch.sum((values * values) * w.unsqueeze(0), dim=1) / denom
    return torch.sqrt(torch.clamp(mean_sq, min=eps))


def analyze_first_round(
    z_tilde: torch.Tensor,
    Z: torch.Tensor,
    weights: torch.Tensor,
    codebook: torch.Tensor,
    szz_zero_tol: float,
) -> Dict[str, Any]:
    z_tilde = z_tilde.detach().float()
    Z = Z.detach().float()
    weights = weights.detach().float()
    codebook = codebook.detach().float()

    code_min = float(codebook.min().item())
    code_max = float(codebook.max().item())
    has_zero = bool(torch.any(codebook == 0.0).item())
    is_zero = (Z == 0.0) if has_zero else torch.zeros_like(Z, dtype=torch.bool)
    is_min = (Z == code_min)
    is_max = (Z == code_max)
    is_edge = is_min | is_max

    per_dim_zero_ratio = is_zero.float().mean(dim=1) if has_zero else torch.zeros(Z.shape[0], dtype=torch.float32)
    per_dim_edge_ratio = is_edge.float().mean(dim=1)
    szz = (Z * weights.unsqueeze(0)) @ Z.T
    szz_diag = torch.diag(szz)
    bad_dim = (szz_diag.abs() <= szz_zero_tol)
    if has_zero:
        bad_dim = bad_dim | (is_zero.all(dim=1))

    return {
        "shape": [int(Z.shape[0]), int(Z.shape[1])],
        "z_tilde_mean": float(z_tilde.mean().item()),
        "z_tilde_std": float(z_tilde.std(unbiased=False).item()),
        "z_tilde_abs_mean": float(z_tilde.abs().mean().item()),
        "z_tilde_outside_codebook_ratio": float(((z_tilde < code_min) | (z_tilde > code_max)).float().mean().item()),
        "z_tilde_below_min_ratio": float((z_tilde < code_min).float().mean().item()),
        "z_tilde_above_max_ratio": float((z_tilde > code_max).float().mean().item()),
        "zero_ratio": float(is_zero.float().mean().item()) if has_zero else 0.0,
        "min_edge_ratio": float(is_min.float().mean().item()),
        "max_edge_ratio": float(is_max.float().mean().item()),
        "edge_ratio": float(is_edge.float().mean().item()),
        "per_dim_zero_ratio_mean": float(per_dim_zero_ratio.mean().item()) if has_zero else 0.0,
        "per_dim_edge_ratio_mean": float(per_dim_edge_ratio.mean().item()),
        "per_dim_all_zero_ratio": float(is_zero.all(dim=1).float().mean().item()) if has_zero else 0.0,
        "per_dim_edge_ge_95_ratio": float((per_dim_edge_ratio >= 0.95).float().mean().item()),
        "szz_diag_min": float(szz_diag.min().item()),
        "szz_diag_max": float(szz_diag.max().item()),
        "szz_diag_mean": float(szz_diag.mean().item()),
        "szz_diag_near_zero_ratio": float((szz_diag.abs() <= szz_zero_tol).float().mean().item()),
        "bad_dim_ratio": float(bad_dim.float().mean().item()),
        "code_hist": codebook_hist(Z, codebook),
    }


def record_lambda_snapshot(
    history: List[Dict[str, Any]],
    iter_idx: int,
    lambda_x: torch.Tensor,
    lambda_w: torch.Tensor,
    prev_lambda_x: Optional[torch.Tensor],
    prev_lambda_w: Optional[torch.Tensor],
    near_zero_tol: float,
) -> None:
    history.append(
        {
            "iter_idx": int(iter_idx),
            "lambda_x": vector_stats(lambda_x, near_zero_tol),
            "lambda_w": vector_stats(lambda_w, near_zero_tol),
            "lambda_x_delta_l2": 0.0 if prev_lambda_x is None else float(torch.linalg.norm(lambda_x - prev_lambda_x).item()),
            "lambda_w_delta_l2": 0.0 if prev_lambda_w is None else float(torch.linalg.norm(lambda_w - prev_lambda_w).item()),
        }
    )


def save_lambda_plots(output_dir: Path, layer_name: str, lambda_history: List[Dict[str, Any]]) -> Dict[str, str]:
    if not lambda_history:
        return {}
    output_dir.mkdir(parents=True, exist_ok=True)
    xs = [int(p["iter_idx"]) for p in lambda_history]
    short = layer_tag(layer_name)
    paths: Dict[str, str] = {}

    def series(key_root: str, stat_key: str) -> List[float]:
        return [float(p[key_root][stat_key]) for p in lambda_history]

    for which in ("lambda_x", "lambda_w"):
        plt.figure(figsize=(8, 5))
        plt.plot(xs, series(which, "min"), marker="o", label="min")
        plt.plot(xs, series(which, "max"), marker="o", label="max")
        plt.plot(xs, series(which, "mean_abs"), marker="o", label="mean_abs")
        plt.xlabel("Iteration")
        plt.ylabel(which)
        plt.title(f"{layer_name} {which} range")
        plt.legend()
        plt.grid(True, alpha=0.3)
        path = output_dir / f"{short}_{which}_range.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths[f"{which}_range"] = str(path)

        plt.figure(figsize=(8, 5))
        plt.plot(xs, series(which, "near_zero_frac"), marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("near_zero_frac")
        plt.title(f"{layer_name} {which} near-zero fraction")
        plt.grid(True, alpha=0.3)
        path = output_dir / f"{short}_{which}_near_zero.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths[f"{which}_near_zero"] = str(path)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, [float(p["lambda_x_delta_l2"]) for p in lambda_history], marker="o", label="lambda_x_delta_l2")
    plt.plot(xs, [float(p["lambda_w_delta_l2"]) for p in lambda_history], marker="o", label="lambda_w_delta_l2")
    plt.xlabel("Iteration")
    plt.ylabel("L2 delta")
    plt.title(f"{layer_name} lambda deltas")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = output_dir / f"{short}_lambda_delta.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    paths["lambda_delta"] = str(path)
    return paths


def average_metrics(metrics_by_layer: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if not metrics_by_layer:
        return {}
    keys = sorted(next(iter(metrics_by_layer.values())).keys())
    return {
        key: float(sum(layer_metrics[key] for layer_metrics in metrics_by_layer.values()) / len(metrics_by_layer))
        for key in keys
    }


# ============================================================
# Diagnostic quantizer
# ============================================================


class DiagnosticQuantizer(base.LatticeLinearQuantizer):
    def __init__(self, config: base.QuantizerConfig, diag_config: DiagnosticConfig, logger: Optional[logging.Logger] = None):
        super().__init__(config=config, logger=logger)
        self.diag_config = diag_config

    def _init_U(self, X: torch.Tensor) -> torch.Tensor:
        mode = getattr(self.config, "init_mode", "pca").lower()
        if mode == "random":
            return self._random_init(X)
        if mode == "pca":
            return self._pca_init(X)
        raise ValueError(f"Unsupported init_mode: {mode}")

    def _init_lambdas(
        self,
        U: torch.Tensor,
        X: torch.Tensor,
        W: torch.Tensor,
        inv_norms_x: torch.Tensor,
        inv_norms_w: torch.Tensor,
        lambda_warm_start: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not lambda_warm_start:
            d = X.shape[0]
            return (
                torch.ones(d, dtype=X.dtype, device=X.device),
                torch.ones(d, dtype=W.dtype, device=W.device),
            )

        s_x = U.T @ X
        s_w = U.T @ W
        if self.diag_config.warm_start_mode == "weighted_rms":
            lam_x = weighted_rms(s_x, inv_norms_x, self.config.eps)
            lam_w = weighted_rms(s_w, inv_norms_w, self.config.eps)
        elif self.diag_config.warm_start_mode == "rms":
            lam_x = weighted_rms(s_x, None, self.config.eps)
            lam_w = weighted_rms(s_w, None, self.config.eps)
        else:
            raise ValueError(f"Unsupported warm_start_mode: {self.diag_config.warm_start_mode}")
        return lam_x, lam_w

    def fit(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        tag: str = "",
        lambda_warm_start: bool = False,
    ) -> DiagnosticQuantizationState:
        fit_start = time.perf_counter()
        X = X.to(dtype=self.dtype)
        W = W.to(dtype=self.dtype)
        self.device = X.device
        self.codebook = self.codebook.to(self.device)

        d, n = X.shape
        _, m = W.shape
        assert W.shape[0] == d, "X and W must have same feature dimension"

        inv_norms_x = 1.0 / (torch.sum(X * X, dim=0) + self.config.eps)
        inv_norms_w = 1.0 / (torch.sum(W * W, dim=0) + self.config.eps)
        if getattr(self.config, "error_mode", "relative") == "absolute":
            inv_norms_x = torch.ones_like(inv_norms_x)
            inv_norms_w = torch.ones_like(inv_norms_w)

        U = self._init_U(X)
        lambda_x, lambda_w = self._init_lambdas(U, X, W, inv_norms_x, inv_norms_w, lambda_warm_start=lambda_warm_start)

        lambda_history: List[Dict[str, Any]] = []
        record_lambda_snapshot(
            lambda_history,
            iter_idx=0,
            lambda_x=lambda_x,
            lambda_w=lambda_w,
            prev_lambda_x=None,
            prev_lambda_w=None,
            near_zero_tol=self.diag_config.lambda_near_zero_tol,
        )

        s_x0 = U.T @ X
        s_w0 = U.T @ W
        safe_lambda_x0 = torch.where(lambda_x.abs() < self.config.eps, torch.full_like(lambda_x, self.config.eps), lambda_x)
        safe_lambda_w0 = torch.where(lambda_w.abs() < self.config.eps, torch.full_like(lambda_w, self.config.eps), lambda_w)
        z_tilde_x0 = s_x0 / safe_lambda_x0.unsqueeze(1)
        z_tilde_w0 = s_w0 / safe_lambda_w0.unsqueeze(1)
        Z_x0 = base.quantize_nearest(z_tilde_x0, self.codebook)
        Z_w0 = base.quantize_nearest(z_tilde_w0, self.codebook)
        first_round = {
            "X": analyze_first_round(z_tilde_x0, Z_x0, inv_norms_x, self.codebook, self.diag_config.szz_zero_tol),
            "W": analyze_first_round(z_tilde_w0, Z_w0, inv_norms_w, self.codebook, self.diag_config.szz_zero_tol),
        }

        if self.logger is not None:
            self.logger.info(
                "diag first-round | tag=%s init=%s warm=%s | X[outside=%.4f zero=%.4f edge=%.4f bad_dim=%.4f] W[outside=%.4f zero=%.4f edge=%.4f bad_dim=%.4f]",
                tag,
                getattr(self.config, "init_mode", "pca"),
                lambda_warm_start,
                first_round["X"]["z_tilde_outside_codebook_ratio"],
                first_round["X"]["zero_ratio"],
                first_round["X"]["edge_ratio"],
                first_round["X"]["bad_dim_ratio"],
                first_round["W"]["z_tilde_outside_codebook_ratio"],
                first_round["W"]["zero_ratio"],
                first_round["W"]["edge_ratio"],
                first_round["W"]["bad_dim_ratio"],
            )

        J_old = float("inf")
        hist_J: List[float] = []
        hist_Jx: List[float] = []
        hist_Jw: List[float] = []
        convergence_iter = self.config.max_iters

        Z_x = Z_x0
        Z_w = Z_w0
        for t in range(1, self.config.max_iters + 1):
            Z_x = self._e_step(X, U, lambda_x)
            Z_w = self._e_step(W, U, lambda_w)

            prev_lambda_x = lambda_x.detach().clone()
            prev_lambda_w = lambda_w.detach().clone()
            lambda_x, SXZx, _ = self._update_lambda(X, U, Z_x, inv_norms_x)
            lambda_w, SWZw, _ = self._update_lambda(W, U, Z_w, inv_norms_w)
            U = self._update_U(lambda_x, lambda_w, SXZx, SWZw)

            if t <= max(1, self.diag_config.lambda_track_iters):
                record_lambda_snapshot(
                    lambda_history,
                    iter_idx=t,
                    lambda_x=lambda_x,
                    lambda_w=lambda_w,
                    prev_lambda_x=prev_lambda_x,
                    prev_lambda_w=prev_lambda_w,
                    near_zero_tol=self.diag_config.lambda_near_zero_tol,
                )

            X_hat = U @ (lambda_x.unsqueeze(1) * Z_x)
            W_hat = U @ (lambda_w.unsqueeze(1) * Z_w)
            J_x = base.relative_weighted_reconstruction_error(X, X_hat, inv_norms_x, average=True)
            J_w = base.relative_weighted_reconstruction_error(W, W_hat, inv_norms_w, average=True)
            J = J_x + self.config.beta * J_w

            hist_J.append(float(J.item()))
            hist_Jx.append(float(J_x.item()))
            hist_Jw.append(float(J_w.item()))

            rel_change = float("inf") if not math.isfinite(J_old) else abs(float(J.item()) - J_old) / max(1.0, abs(J_old))
            if self.logger is not None and (t == 1 or t % self.config.log_every == 0):
                self.logger.info(
                    "tag=%s iter=%03d | J=%.6f J_x=%.6f J_w=%.6f rel_change=%.6e | lambda_x[min,max]=[%.6f, %.6f] lambda_w[min,max]=[%.6f, %.6f]",
                    tag,
                    t,
                    float(J.item()),
                    float(J_x.item()),
                    float(J_w.item()),
                    rel_change,
                    float(lambda_x.min().item()),
                    float(lambda_x.max().item()),
                    float(lambda_w.min().item()),
                    float(lambda_w.max().item()),
                )
            if t % self.config.convergence_check_every == 0:
                if rel_change < self.config.tol:
                    convergence_iter = t
                    break
                J_old = float(J.item())

        fit_time_sec = time.perf_counter() - fit_start
        return DiagnosticQuantizationState(
            U=U,
            lambda_x=lambda_x,
            lambda_w=lambda_w,
            Z_x=Z_x,
            Z_w=Z_w,
            codebook=self.codebook,
            objective_history=hist_J,
            objective_x_history=hist_Jx,
            objective_w_history=hist_Jw,
            convergence_iter=convergence_iter,
            fit_time_sec=fit_time_sec,
            init_mode=getattr(self.config, "init_mode", "pca"),
            lambda_warm_start=bool(lambda_warm_start),
            first_round=first_round,
            lambda_history=lambda_history,
        )


# ============================================================
# Metrics
# ============================================================


@torch.no_grad()
def compute_metrics_on_split(
    X: torch.Tensor,
    W: torch.Tensor,
    state: DiagnosticQuantizationState,
    quantizer: DiagnosticQuantizer,
) -> Dict[str, float]:
    X = X.to(device=state.U.device, dtype=state.U.dtype)
    W = W.to(device=state.U.device, dtype=state.U.dtype)
    X_hat = quantizer.reconstruct_X(X, state)
    return {
        "rel_recon_error_x": float(torch.sum((X - X_hat) ** 2).item() / max(torch.sum(X ** 2).item(), 1e-12)),
        "rel_linear_error": float(base.compute_linear_relative_error(X, W, state, chunk_tokens=128)),
    }


# ============================================================
# Config resolver
# ============================================================


class RunConfigResolver:
    def __init__(self, custom_codebooks: Dict[str, Tuple[float, ...]]):
        self.custom_codebooks = custom_codebooks

    def make_config(self, combo: Dict[str, Any]) -> Tuple[ExperimentConfig, Dict[str, Any], Tuple[float, ...], str]:
        cfg = ExperimentConfig()
        for path, value in BASE_EXPERIMENT_CONFIG_OVERRIDES.items():
            set_dotted_attr(cfg, path, value)

        resolved_combo: Dict[str, Any] = {}
        codebook = tuple(cfg.quant.codebook)
        codebook_tag = codebook_tag_from_values(codebook)

        for path, value in combo.items():
            if path == "quant_ext.codebook_spec":
                codebook, codebook_tag = resolve_codebook_spec(value)
                cfg.quant.codebook = codebook
                resolved_combo[path] = list(codebook)
                continue
            set_dotted_attr(cfg, path, value)
            resolved_combo[path] = value

        if "quant_ext.codebook_spec" not in combo:
            codebook = tuple(cfg.quant.codebook)
            codebook_tag = codebook_tag_from_values(codebook)
            resolved_combo["quant_ext.codebook_spec"] = list(codebook)

        return cfg, resolved_combo, codebook, codebook_tag


# ============================================================
# Diagnostic runner
# ============================================================


class PCADiagnosticExperimentRunner:
    def __init__(self, config_resolver: Optional[RunConfigResolver] = None):
        self.config_resolver = config_resolver or RunConfigResolver(CUSTOM_CODEBOOKS)

    def _collect_inputs(
        self,
        cfg: ExperimentConfig,
        logger: logging.Logger,
        model,
        tokenizer,
        target_specs: Dict[str, base.TargetModuleSpec],
        text: str,
        max_tokens: int,
        tag: str,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
        t0 = time.perf_counter()
        input_ids = base.tokenize_text(text, tokenizer, max_tokens=max_tokens)
        X_by_layer, counts = base.collect_target_inputs(
            model=model,
            input_ids=input_ids,
            target_modules={k: spec.module for k, spec in target_specs.items()},
            max_tokens=max_tokens,
            device=cfg.eval.device,
            logger=logger,
        )
        logger.info("collected %s inputs | max_tokens=%d elapsed=%.3fs", tag, max_tokens, time.perf_counter() - t0)
        return X_by_layer, counts

    def _fit_variant(
        self,
        cfg: ExperimentConfig,
        logger: logging.Logger,
        variant_dir: Path,
        variant: DiagnosticVariantConfig,
        target_specs: Dict[str, base.TargetModuleSpec],
        X_calib_by_layer: Dict[str, torch.Tensor],
        X_eval_by_layer: Dict[str, torch.Tensor],
        eval_text: str,
        model,
        tokenizer,
    ) -> VariantRunArtifacts:
        variant_dir.mkdir(parents=True, exist_ok=True)
        modules: Dict[str, nn.Module] = {}
        per_layer_metrics: Dict[str, Dict[str, float]] = {}
        per_layer_first_round: Dict[str, Dict[str, Any]] = {}
        per_layer_lambda_history: Dict[str, List[Dict[str, Any]]] = {}
        convergence_iters: Dict[str, int] = {}
        fit_time_sec_total = 0.0
        ppl_eval_info: Dict[str, float] = {}

        quant_cfg = copy.deepcopy(cfg.quant)
        quant_cfg.init_mode = variant.init_mode

        for layer_name, spec in target_specs.items():
            layer_dir = variant_dir / layer_tag(layer_name)
            layer_dir.mkdir(parents=True, exist_ok=True)
            quantizer = DiagnosticQuantizer(quant_cfg, cfg.diag, logger=logger)

            X_calib = X_calib_by_layer[layer_name].to(device=cfg.eval.device)
            X_eval = X_eval_by_layer[layer_name].to(device=cfg.eval.device)
            W = spec.module.weight.detach().T.to(device=cfg.eval.device, dtype=base.get_torch_dtype(quant_cfg.dtype))
            bias = None if spec.module.bias is None else spec.module.bias.detach().to(device=cfg.eval.device, dtype=base.get_torch_dtype(quant_cfg.dtype))

            state = quantizer.fit(X_calib, W, tag=layer_name, lambda_warm_start=variant.lambda_warm_start)
            fit_time_sec_total += state.fit_time_sec

            calib_metrics = compute_metrics_on_split(X_calib, W, state, quantizer)
            eval_metrics = compute_metrics_on_split(X_eval, W, state, quantizer)
            metrics = {
                "calib_rel_recon_error_x": calib_metrics["rel_recon_error_x"],
                "calib_rel_linear_error": calib_metrics["rel_linear_error"],
                "eval_rel_recon_error_x": eval_metrics["rel_recon_error_x"],
                "eval_rel_linear_error": eval_metrics["rel_linear_error"],
            }

            per_layer_metrics[layer_name] = metrics
            per_layer_first_round[layer_name] = state.first_round
            per_layer_lambda_history[layer_name] = state.lambda_history
            convergence_iters[layer_name] = state.convergence_iter
            modules[layer_name] = base.QuantizedLinear(state, bias=bias).to(cfg.eval.device)

            _atomic_write_json(layer_dir / "first_round.json", state.first_round)
            _atomic_write_json(layer_dir / "lambda_history.json", state.lambda_history)
            _atomic_write_json(layer_dir / "metrics.json", metrics)
            if cfg.diag.save_plots:
                plot_paths = save_lambda_plots(layer_dir, layer_name, state.lambda_history)
                _atomic_write_json(layer_dir / "plot_paths.json", plot_paths)

        quantized_ppl: Optional[float] = None
        if cfg.diag.run_ppl:
            originals = base.replace_target_modules(target_specs, modules)
            try:
                quantized_ppl, ppl_eval_info = base.evaluate_perplexity_sliding_window(
                    model=model,
                    tokenizer=tokenizer,
                    text=eval_text,
                    device=cfg.eval.device,
                    stride=cfg.eval.stride,
                    max_eval_tokens=cfg.data.eval_num_tokens,
                    logger=logger,
                    tag=f"diag_{variant.name}",
                )
            finally:
                base.restore_target_modules(target_specs, originals)

        return VariantRunArtifacts(
            variant_name=variant.name,
            init_mode=variant.init_mode,
            lambda_warm_start=variant.lambda_warm_start,
            quantized_ppl=quantized_ppl,
            ppl_eval_info=ppl_eval_info,
            per_layer_metrics=per_layer_metrics,
            per_layer_first_round=per_layer_first_round,
            per_layer_lambda_history=per_layer_lambda_history,
            convergence_iters=convergence_iters,
            fit_time_sec_total=fit_time_sec_total,
        )

    def _build_summary(self, cfg: ExperimentConfig, artifacts: DiagnosticArtifacts) -> str:
        lines: List[str] = []
        lines.append("PCA initialization diagnostics")
        if artifacts.baseline_ppl is not None:
            lines.append(f"- baseline_ppl: {artifacts.baseline_ppl:.6f}")
        lines.append(f"- target_blocks: {artifacts.target_info['block_indices']}")
        lines.append(f"- eval_metric_tokens: {cfg.diag.eval_metric_num_tokens}")
        lines.append(f"- codebook_tag: {artifacts.codebook_tag}")
        lines.append("")

        for variant in artifacts.variants:
            lines.append(
                f"[{variant.variant_name}] init={variant.init_mode} warm={variant.lambda_warm_start} ppl={variant.quantized_ppl if variant.quantized_ppl is not None else 'not_run'}"
            )
            for layer_name, metrics in variant.per_layer_metrics.items():
                fr_x = variant.per_layer_first_round[layer_name]["X"]
                fr_w = variant.per_layer_first_round[layer_name]["W"]
                lines.append(
                    f"- {layer_name}: calib_linear={metrics['calib_rel_linear_error']:.6f} eval_linear={metrics['eval_rel_linear_error']:.6f} "
                    f"X[out={fr_x['z_tilde_outside_codebook_ratio']:.4f}, zero={fr_x['zero_ratio']:.4f}, edge={fr_x['edge_ratio']:.4f}, bad={fr_x['bad_dim_ratio']:.4f}] "
                    f"W[out={fr_w['z_tilde_outside_codebook_ratio']:.4f}, zero={fr_w['zero_ratio']:.4f}, edge={fr_w['edge_ratio']:.4f}, bad={fr_w['bad_dim_ratio']:.4f}]"
                )
            lines.append("")
        return "\n".join(lines)

    def run(self, combo: Dict[str, Any], run_dir: Path) -> DiagnosticArtifacts:
        cfg, combo_resolved, codebook, codebook_tag = self.config_resolver.make_config(combo)
        cfg.output_dir = str(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logger(run_dir)

        logger.info("========== PCA diagnostic run starts ==========")
        logger.info("config=%s", json.dumps(asdict(cfg), ensure_ascii=False, indent=2, default=str))
        base.set_seed(cfg.seed)

        model, tokenizer, model_load_sec = base.load_model_and_tokenizer(cfg, logger)
        target_specs = base.get_all_block_attention_targets(
            model,
            target_linear_names=cfg.target.target_linear_names,
            block_indices=cfg.target.block_indices,
        )
        resolved_block_indices = sorted({spec.block_index for spec in target_specs.values()})

        calib_text = base.load_text_split(cfg, cfg.data.calib_split, logger)
        eval_text = base.load_text_split(cfg, cfg.data.eval_split, logger)

        baseline_ppl: Optional[float] = None
        baseline_ppl_eval_info: Dict[str, float] = {}
        if cfg.diag.run_ppl:
            baseline_ppl, baseline_ppl_eval_info = base.evaluate_perplexity_sliding_window(
                model=model,
                tokenizer=tokenizer,
                text=eval_text,
                device=cfg.eval.device,
                stride=cfg.eval.stride,
                max_eval_tokens=cfg.data.eval_num_tokens,
                logger=logger,
                tag="baseline_fp",
            )

        X_calib_by_layer, calib_counts = self._collect_inputs(
            cfg=cfg,
            logger=logger,
            model=model,
            tokenizer=tokenizer,
            target_specs=target_specs,
            text=calib_text,
            max_tokens=cfg.data.calib_num_tokens,
            tag="calib",
        )
        X_eval_by_layer, eval_counts = self._collect_inputs(
            cfg=cfg,
            logger=logger,
            model=model,
            tokenizer=tokenizer,
            target_specs=target_specs,
            text=eval_text,
            max_tokens=cfg.diag.eval_metric_num_tokens,
            tag="eval_metric",
        )

        variants: List[VariantRunArtifacts] = []
        for variant in cfg.diag.variants:
            logger.info("running variant=%s init=%s warm=%s", variant.name, variant.init_mode, variant.lambda_warm_start)
            variants.append(
                self._fit_variant(
                    cfg=cfg,
                    logger=logger,
                    variant_dir=run_dir / variant.name,
                    variant=variant,
                    target_specs=target_specs,
                    X_calib_by_layer=X_calib_by_layer,
                    X_eval_by_layer=X_eval_by_layer,
                    eval_text=eval_text,
                    model=model,
                    tokenizer=tokenizer,
                )
            )

        artifacts = DiagnosticArtifacts(
            config=asdict(cfg),
            combo=combo,
            combo_resolved=combo_resolved,
            codebook=codebook,
            codebook_tag=codebook_tag,
            baseline_ppl=baseline_ppl,
            baseline_ppl_eval_info=baseline_ppl_eval_info,
            target_info={
                "model_name": cfg.data.model_name,
                "block_indices": resolved_block_indices,
                "num_blocks": len(resolved_block_indices),
                "target_linear_names": list(cfg.target.target_linear_names),
                "calib_collected_token_counts": calib_counts,
                "eval_metric_collected_token_counts": eval_counts,
                "model_load_sec": model_load_sec,
            },
            variants=variants,
        )
        _atomic_write_json(run_dir / RUN_JSON_NAME, asdict(artifacts))
        _atomic_write_text(run_dir / RUN_SUMMARY_NAME, self._build_summary(cfg, artifacts))
        logger.info("========== PCA diagnostic run ends ==========")
        return artifacts


# ============================================================
# Artifact writer and sweep runner
# ============================================================


class ArtifactWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def write_sweep_manifest(self, runs: List[Tuple[int, Dict[str, Any], str]]) -> None:
        payload = [
            {
                "combo_index": combo_index,
                "run_name": run_name,
                "combo": combo,
            }
            for combo_index, combo, run_name in runs
        ]
        _atomic_write_json(self.output_dir / SWEEP_MANIFEST_NAME, payload)

    def write_sweep_summary(self, rows: List[SweepSummaryRow]) -> None:
        rows = sorted(rows, key=lambda r: r.combo_index)
        _atomic_write_json(self.output_dir / SWEEP_JSON_NAME, [asdict(r) for r in rows])

        lines = [
            "===== PCA diagnostic sweep summary =====",
            f"completed_runs={len(rows)}",
            "",
        ]
        for row in rows:
            pca_ppl = row.variant_ppl.get("pca")
            rnd_ppl = row.variant_ppl.get("random")
            pca_warm_ppl = row.variant_ppl.get("pca_warm")
            lines.append(
                f"[{row.combo_index:03d}] {row.run_name} | baseline={row.baseline_ppl} | random={rnd_ppl} | pca={pca_ppl} | pca_warm={pca_warm_ppl}"
            )
            lines.append(f"  combo={json.dumps(row.combo_resolved, ensure_ascii=False)}")
        _atomic_write_text(self.output_dir / SWEEP_TEXT_NAME, "\n".join(lines))


class SweepRunner:
    def __init__(self, runner: Optional[PCADiagnosticExperimentRunner] = None):
        self.runner = runner or PCADiagnosticExperimentRunner()

    def enumerate_runs(self) -> List[Tuple[int, Dict[str, Any], str]]:
        return enumerate_sweep_runs()

    def num_combos(self) -> int:
        return len(self.enumerate_runs())

    def _validate_indices(self, indices: Sequence[int], total: int) -> List[int]:
        unique = sorted({int(i) for i in indices})
        if not unique:
            raise ValueError("indices 不能为空")
        bad = [i for i in unique if i < 0 or i >= total]
        if bad:
            raise IndexError(f"组合索引越界: {bad}; 合法范围为 0..{total - 1}")
        return unique

    def _row_from_artifacts(
        self,
        combo_index: int,
        combo: Dict[str, Any],
        run_dir: Path,
        run_name: str,
        artifacts: DiagnosticArtifacts,
    ) -> SweepSummaryRow:
        return SweepSummaryRow(
            combo_index=combo_index,
            run_name=run_name,
            run_dir=str(run_dir),
            combo=combo,
            combo_resolved=artifacts.combo_resolved,
            baseline_ppl=artifacts.baseline_ppl,
            variant_ppl={v.variant_name: v.quantized_ppl for v in artifacts.variants},
            variant_avg_metrics={v.variant_name: average_metrics(v.per_layer_metrics) for v in artifacts.variants},
            variant_fit_time_sec_total={v.variant_name: float(v.fit_time_sec_total) for v in artifacts.variants},
        )

    def collect_existing_rows(self, output_root: Path = OUTPUT_ROOT) -> List[SweepSummaryRow]:
        output_root.mkdir(parents=True, exist_ok=True)
        rows: List[SweepSummaryRow] = []
        for combo_index, combo, run_name in self.enumerate_runs():
            run_dir = output_root / run_name
            result_file = run_dir / RUN_JSON_NAME
            if not result_file.exists():
                continue
            with open(result_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            variants = [VariantRunArtifacts(**v) for v in loaded["variants"]]
            loaded["variants"] = variants
            artifacts = DiagnosticArtifacts(**loaded)
            rows.append(self._row_from_artifacts(combo_index, combo, run_dir, run_name, artifacts))
        return sorted(rows, key=lambda r: r.combo_index)

    def rebuild_summary(self, output_root: Path = OUTPUT_ROOT) -> List[SweepSummaryRow]:
        output_root.mkdir(parents=True, exist_ok=True)
        runs = self.enumerate_runs()
        writer = ArtifactWriter(output_root)
        writer.write_sweep_manifest(runs)
        rows = self.collect_existing_rows(output_root)
        writer.write_sweep_summary(rows)
        return rows

    def run(self, output_root: Path = OUTPUT_ROOT, indices: Optional[Sequence[int]] = None) -> List[SweepSummaryRow]:
        output_root.mkdir(parents=True, exist_ok=True)
        runs = self.enumerate_runs()
        total = len(runs)
        selected_indices = list(range(total)) if indices is None else self._validate_indices(indices, total)
        selected_set = set(selected_indices)

        rows: List[SweepSummaryRow] = []
        for combo_index, combo, run_name in runs:
            if combo_index not in selected_set:
                continue
            run_dir = output_root / run_name
            result_file = run_dir / RUN_JSON_NAME

            if result_file.exists():
                print(f"跳过已完成的实验: idx={combo_index} name={run_name}")
                with open(result_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                variants = [VariantRunArtifacts(**v) for v in loaded["variants"]]
                loaded["variants"] = variants
                artifacts = DiagnosticArtifacts(**loaded)
            else:
                print(f"开始实验: idx={combo_index} name={run_name}")
                artifacts = self.runner.run(combo, run_dir)

            rows.append(self._row_from_artifacts(combo_index, combo, run_dir, run_name, artifacts))

        self.rebuild_summary(output_root)
        return sorted(rows, key=lambda r: r.combo_index)


# ============================================================
# Entry points
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCA diagnostic sweep runner with sbatch-array support")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT, help="实验输出根目录")
    parser.add_argument("--combo-index", type=int, default=None, help="只运行第几个 sweep 组合")
    parser.add_argument("--array-task-id", type=int, default=None, help="等价于 --combo-index，便于 sbatch --array 调用")
    parser.add_argument("--print-num-combos", action="store_true", help="仅打印 sweep 组合总数")
    parser.add_argument("--print-array-range", action="store_true", help="仅打印建议的 array 范围，如 0-15")
    parser.add_argument("--rebuild-summary", action="store_true", help="不跑实验，只扫描已有 diagnostic_results.json 重建 sweep 汇总")
    return parser.parse_args()


def resolve_requested_index(args: argparse.Namespace) -> Optional[int]:
    requested = []
    if args.combo_index is not None:
        requested.append(args.combo_index)
    if args.array_task_id is not None:
        requested.append(args.array_task_id)
    env_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if env_task_id is not None and args.combo_index is None and args.array_task_id is None:
        requested.append(int(env_task_id))
    if not requested:
        return None
    if len(set(requested)) != 1:
        raise ValueError(f"收到互相冲突的组合索引: {requested}")
    return requested[0]


def main() -> None:
    args = parse_args()
    sweep_runner = SweepRunner()

    if args.print_num_combos:
        print(sweep_runner.num_combos())
        return

    if args.print_array_range:
        num_combos = sweep_runner.num_combos()
        if num_combos <= 0:
            raise ValueError("SWEEP_GRID 为空，无法生成 array 范围")
        print(f"0-{num_combos - 1}")
        return

    if args.rebuild_summary:
        rows = sweep_runner.rebuild_summary(args.output_root)
        print(json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2))
        return

    requested_index = resolve_requested_index(args)
    if requested_index is None:
        rows = sweep_runner.run(args.output_root)
    else:
        rows = sweep_runner.run(args.output_root, indices=[requested_index])
    print(json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
