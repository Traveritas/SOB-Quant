from __future__ import annotations

import importlib.util
import itertools
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ============================================================
# User config area
# ============================================================

BASE_SCRIPT_PATH = Path(__file__).with_name("opt_all_blocks_qkvo_experiment_v1.py")
OUTPUT_ROOT = Path("./out_qkvo_ref")

CUSTOM_CODEBOOKS: Dict[str, Tuple[float, ...]] = {
    "d5": (-2.0, -1.0, 0.0, 1.0, 2.0),
    "t3": (-1.0, 0.0, 1.0),
    "s7": (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
    "s8": (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
}
DEFAULT_CODEBOOK_SPEC: Any = "d5"

BASE_EXPERIMENT_CONFIG_OVERRIDES: Dict[str, Any] = {
    "data.calib_num_tokens": 4096,
    "data.eval_num_tokens": None,
    "eval.stride": 512,
    "target.block_indices": None,
    "target.target_linear_names": ("q_proj", "k_proj", "v_proj", "out_proj"),
    "quant.max_iters": 80,
    "quant.tol": 1e-5,
    "quant.convergence_check_every": 1,
    "quant.log_every": 1,
    "quant.codebook": CUSTOM_CODEBOOKS["d5"],
    "quant.dtype": "float32",
    "quant.eps": 1e-8,
}

SWEEP_GRID: Dict[str, Iterable[Any]] = {
    "target.block_indices": (
        (11,),
    ),
    "quant.beta": (1,),
    "quant_ext.init_mode": ("random"),
    "quant_ext.error_mode": ("relative",),
    "quant_ext.codebook_spec": ("s8",),
}

SAVE_PER_RUN_JSON = True
SAVE_PER_RUN_PLOTS = True
LOG_NAME = "log.txt"
RUN_JSON_NAME = "res.json"
RUN_SUMMARY_NAME = "sum.txt"
SWEEP_JSON_NAME = "sum.json"
SWEEP_TEXT_NAME = "sum.txt"
U_DIR_NAME = "u"


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
# Config / state dataclasses
# ============================================================


@dataclass
class ExtendedQuantizerConfig(base.QuantizerConfig):
    init_mode: str = "pca"          # pca | random
    error_mode: str = "relative"    # relative | absolute
    track_u: bool = True
    track_u_every: int = 1
    track_u_full_matrix: bool = False
    track_u_save_interval: int = 10
    track_u_save_first: int = 5
    track_z_flip_stats: bool = True
    random_init_orthogonal: bool = True
    random_init_scale: float = 1.0


@dataclass
class RefactorExperimentConfig(base.ExperimentConfig):
    quant: ExtendedQuantizerConfig = field(default_factory=ExtendedQuantizerConfig)


@dataclass
class TrackingQuantizationState(base.QuantizationState):
    u_trace: List[Dict[str, Any]] = field(default_factory=list)
    init_mode: str = "pca"
    error_mode: str = "relative"


@dataclass
class PreparedRun:
    logger: logging.Logger
    model: Any
    tokenizer: Any
    target_specs: Dict[str, base.TargetModuleSpec]
    resolved_block_indices: List[int]
    eval_text: str
    X_calib_by_layer: Dict[str, torch.Tensor]
    collected_token_counts: Dict[str, int]
    baseline_ppl: float
    ppl_eval_info: Dict[str, Dict[str, float]]
    timing_info: Dict[str, float]


@dataclass
class QuantBuildResult:
    modules: Dict[str, nn.Module]
    states: Dict[str, TrackingQuantizationState]
    metrics_by_layer: Dict[str, Dict[str, float]]
    tensor_info: Dict[str, Dict[str, object]]
    plot_paths: Dict[str, Dict[str, str]]
    u_traces: Dict[str, List[Dict[str, Any]]]


@dataclass
class RunArtifacts:
    config: Dict[str, Any]
    combo: Dict[str, Any]
    combo_resolved: Dict[str, Any]
    codebook: Tuple[float, ...]
    codebook_tag: str
    quantization_completed: bool
    baseline_ppl: float
    sq_baseline_ppl: float
    quantized_ppl: float
    sq_metrics: Dict[str, Dict[str, float]]
    sq_metrics_avg: Dict[str, float]
    quant_metrics: Dict[str, Dict[str, float]]
    quant_metrics_avg: Dict[str, float]
    convergence_iters: Dict[str, int]
    objective_histories: Dict[str, List[float]]
    objective_x_histories: Dict[str, List[float]]
    objective_w_histories: Dict[str, List[float]]
    u_traces: Dict[str, List[Dict[str, Any]]]
    tensor_info: Dict[str, Dict[str, object]]
    timing_info: Dict[str, float]
    ppl_eval_info: Dict[str, Dict[str, float]]
    plot_paths: Dict[str, Dict[str, str]]
    target_info: Dict[str, object]


@dataclass
class SweepSummaryRow:
    run_name: str
    run_dir: str
    combo: Dict[str, Any]
    combo_resolved: Dict[str, Any]
    baseline_ppl: float
    sq_baseline_ppl: float
    quantized_ppl: float
    quant_metrics_avg: Dict[str, float]
    sq_metrics_avg: Dict[str, float]
    convergence_iters: Dict[str, int]
    u_trace_last: Dict[str, Dict[str, Any]]
    u_trace_length: Dict[str, int]


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



def layer_tag(layer_name: str) -> str:
    block_part, proj_part = layer_name.split('.')
    block_idx = int(block_part.replace('block', ''))
    proj_map = {
        'q_proj': 'q',
        'k_proj': 'k',
        'v_proj': 'v',
        'out_proj': 'o',
    }
    return f"b{block_idx}_{proj_map.get(proj_part, proj_part)}"



def combo_to_name(combo: Dict[str, Any]) -> str:
    pieces: List[str] = []
    for k, v in combo.items():
        if k == "target.block_indices":
            pieces.append(block_indices_tag(v))
        elif k == "quant.beta":
            pieces.append(f"bt{sanitize_value(v)}")
        elif k == "quant_ext.init_mode":
            pieces.append(f"i{str(v)[0].lower()}")
        elif k == "quant_ext.error_mode":
            pieces.append(f"e{str(v)[0].lower()}")
        elif k == "quant_ext.codebook_spec":
            _, tag = resolve_codebook_spec(v)
            pieces.append(f"z{tag}")
        else:
            pieces.append(f"{k.replace('quant_ext.', '').replace('quant.', 'q').replace('target.', 't').replace('data.', 'd')}{sanitize_value(v)}")
    return "__".join(pieces)



def setup_short_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"qkvo_ref_{output_dir.resolve()}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(output_dir / LOG_NAME, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger



def objective_reconstruction_error(
    X: torch.Tensor,
    X_hat: torch.Tensor,
    inv_norms: torch.Tensor,
    error_mode: str,
    average: bool = False,
) -> torch.Tensor:
    residual = X - X_hat
    per_col_err = torch.sum(residual * residual, dim=0)
    if error_mode == "relative":
        loss = torch.sum(inv_norms * per_col_err)
    elif error_mode == "absolute":
        loss = torch.sum(per_col_err)
    else:
        raise ValueError(f"Unsupported error_mode: {error_mode}")
    if average:
        loss = loss / max(X.shape[1], 1)
    return loss



def save_loss_plots_short(
    output_dir: Path,
    layer_name: str,
    objective_history: List[float],
    objective_x_history: List[float],
    objective_w_history: List[float],
) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    if not objective_history:
        return paths
    xs = list(range(1, len(objective_history) + 1))
    prefix = layer_tag(layer_name)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, objective_history, marker="o", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("J")
    plt.title(f"{layer_name} total objective")
    plt.grid(True, alpha=0.3)
    total_path = output_dir / f"{prefix}_J.png"
    plt.tight_layout()
    plt.savefig(total_path, dpi=160)
    plt.close()
    paths["J"] = str(total_path)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, objective_x_history, marker="o", linewidth=1.5, label="J_x")
    plt.plot(xs, objective_w_history, marker="o", linewidth=1.5, label="J_w")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"{layer_name} objective components")
    plt.legend()
    plt.grid(True, alpha=0.3)
    comp_path = output_dir / f"{prefix}_Jxw.png"
    plt.tight_layout()
    plt.savefig(comp_path, dpi=160)
    plt.close()
    paths["Jxw"] = str(comp_path)
    return paths


# ============================================================
# Tracking helpers + observer
# ============================================================


def _safe_cosine_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    denom = torch.linalg.norm(aa) * torch.linalg.norm(bb)
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float(torch.dot(aa, bb).item() / denom.item())



def _principal_angle_stats(A: torch.Tensor, B: torch.Tensor) -> Dict[str, float]:
    A64 = A.to(dtype=torch.float64)
    B64 = B.to(dtype=torch.float64)
    M = A64.T @ B64
    svals = torch.linalg.svdvals(M).clamp(-1.0, 1.0)
    angles = torch.arccos(svals)
    angles_deg = angles * (180.0 / math.pi)
    return {
        "mean_deg": float(torch.mean(angles_deg).item()),
        "max_deg": float(torch.max(angles_deg).item()),
        "geodesic": float(torch.linalg.norm(angles).item()),
    }



def _column_cosine_stats(A: torch.Tensor, B: torch.Tensor) -> Dict[str, float]:
    cos = torch.sum(A * B, dim=0)
    denom = torch.linalg.norm(A, dim=0) * torch.linalg.norm(B, dim=0)
    denom = torch.clamp(denom, min=1e-12)
    cos = torch.abs(cos / denom)
    return {
        "mean": float(torch.mean(cos).item()),
        "min": float(torch.min(cos).item()),
    }



def _flip_rate(prev_codes: Optional[torch.Tensor], new_codes: Optional[torch.Tensor]) -> Tuple[float, float]:
    if prev_codes is None or new_codes is None:
        return 0.0, 0.0
    changed = (prev_codes != new_codes)
    if changed.numel() == 0:
        return 0.0, 0.0
    overall = float(changed.to(torch.float32).mean().item())
    per_dim = changed.to(torch.float32).mean(dim=1)
    return overall, float(per_dim.mean().item())



def _should_save_u_matrix(iter_idx: int, max_iters: int, cfg: ExtendedQuantizerConfig) -> bool:
    if not cfg.track_u_full_matrix:
        return False
    if iter_idx <= max(0, cfg.track_u_save_first):
        return True
    if iter_idx == max_iters:
        return True
    interval = max(1, cfg.track_u_save_interval)
    return (iter_idx % interval) == 0



def compute_u_trace_point(
    U: torch.Tensor,
    U_prev: torch.Tensor,
    U_init: torch.Tensor,
    iter_idx: int,
    lambda_x: Optional[torch.Tensor] = None,
    lambda_x_prev: Optional[torch.Tensor] = None,
    lambda_w: Optional[torch.Tensor] = None,
    lambda_w_prev: Optional[torch.Tensor] = None,
    Z_x: Optional[torch.Tensor] = None,
    Z_x_prev: Optional[torch.Tensor] = None,
    Z_w: Optional[torch.Tensor] = None,
    Z_w_prev: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    eye = torch.eye(U.shape[1], device=U.device, dtype=U.dtype)
    gram = U.T @ U
    diag = torch.diag(gram)
    offdiag = gram - torch.diag(diag)
    svals = torch.linalg.svdvals(U)

    angle_prev = _principal_angle_stats(U, U_prev)
    angle_init = _principal_angle_stats(U, U_init)
    col_prev = _column_cosine_stats(U, U_prev)
    z_x_flip, z_x_flip_per_dim = _flip_rate(Z_x_prev, Z_x)
    z_w_flip, z_w_flip_per_dim = _flip_rate(Z_w_prev, Z_w)

    lambda_x_delta = 0.0 if lambda_x is None or lambda_x_prev is None else float(torch.linalg.norm(lambda_x - lambda_x_prev).item())
    lambda_w_delta = 0.0 if lambda_w is None or lambda_w_prev is None else float(torch.linalg.norm(lambda_w - lambda_w_prev).item())

    return {
        "iter_idx": int(iter_idx),
        "delta_fro_from_prev": float(torch.linalg.norm(U - U_prev).item()),
        "delta_fro_from_init": float(torch.linalg.norm(U - U_init).item()),
        "cosine_to_prev": _safe_cosine_flat(U, U_prev),
        "cosine_to_init": _safe_cosine_flat(U, U_init),
        "orthogonality_error": float(torch.linalg.norm(gram - eye).item()),
        "singular_min": float(svals.min().item()),
        "singular_max": float(svals.max().item()),
        "diag_mean_abs": float(torch.mean(torch.abs(diag)).item()),
        "offdiag_fro": float(torch.linalg.norm(offdiag).item()),
        "principal_angle_mean_prev_deg": angle_prev["mean_deg"],
        "principal_angle_max_prev_deg": angle_prev["max_deg"],
        "principal_angle_mean_init_deg": angle_init["mean_deg"],
        "principal_angle_max_init_deg": angle_init["max_deg"],
        "subspace_geodesic_prev": angle_prev["geodesic"],
        "subspace_geodesic_init": angle_init["geodesic"],
        "column_cosine_mean_prev": col_prev["mean"],
        "column_cosine_min_prev": col_prev["min"],
        "lambda_x_delta_from_prev": lambda_x_delta,
        "lambda_w_delta_from_prev": lambda_w_delta,
        "z_x_flip_rate": z_x_flip,
        "z_w_flip_rate": z_w_flip,
        "z_x_flip_ratio_per_dim_mean": z_x_flip_per_dim,
        "z_w_flip_ratio_per_dim_mean": z_w_flip_per_dim,
    }



def save_u_trace_plots(output_dir: Path, layer_name: str, u_trace: List[Dict[str, Any]]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    if not u_trace:
        return paths

    xs = [int(p["iter_idx"]) for p in u_trace]
    short = layer_tag(layer_name)

    def _plot(y_key: str, ylabel: str, title: str, filename: str) -> None:
        ys = [float(p[y_key]) for p in u_trace]
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
        paths[y_key] = str(path)

    plot_specs = [
        ("delta_fro_from_prev", "Frobenius delta", f"{layer_name} U change vs previous", f"{short}_up.png"),
        ("delta_fro_from_init", "Frobenius delta", f"{layer_name} U change vs init", f"{short}_ui.png"),
        ("cosine_to_prev", "Cosine similarity", f"{layer_name} U cosine vs previous", f"{short}_cp.png"),
        ("cosine_to_init", "Cosine similarity", f"{layer_name} U cosine vs init", f"{short}_ci.png"),
        ("orthogonality_error", "Orthogonality error", f"{layer_name} U orthogonality error", f"{short}_oe.png"),
        ("principal_angle_mean_prev_deg", "Degrees", f"{layer_name} mean principal angle vs previous", f"{short}_apm.png"),
        ("principal_angle_max_prev_deg", "Degrees", f"{layer_name} max principal angle vs previous", f"{short}_apx.png"),
        ("principal_angle_mean_init_deg", "Degrees", f"{layer_name} mean principal angle vs init", f"{short}_aim.png"),
        ("principal_angle_max_init_deg", "Degrees", f"{layer_name} max principal angle vs init", f"{short}_aix.png"),
        ("subspace_geodesic_prev", "Geodesic distance", f"{layer_name} subspace geodesic vs previous", f"{short}_gp.png"),
        ("subspace_geodesic_init", "Geodesic distance", f"{layer_name} subspace geodesic vs init", f"{short}_gi.png"),
        ("column_cosine_mean_prev", "Abs cosine", f"{layer_name} column cosine mean vs previous", f"{short}_cmp.png"),
        ("column_cosine_min_prev", "Abs cosine", f"{layer_name} column cosine min vs previous", f"{short}_cnp.png"),
        ("lambda_x_delta_from_prev", "L2 delta", f"{layer_name} lambda_x change vs previous", f"{short}_lxp.png"),
        ("lambda_w_delta_from_prev", "L2 delta", f"{layer_name} lambda_w change vs previous", f"{short}_lwp.png"),
        ("z_x_flip_rate", "Flip rate", f"{layer_name} Z_x flip rate", f"{short}_zxf.png"),
        ("z_w_flip_rate", "Flip rate", f"{layer_name} Z_w flip rate", f"{short}_zwf.png"),
    ]
    for args in plot_specs:
        _plot(*args)
    return paths


class QuantizerObserver:
    def on_fit_start(self, *, tag: str, U_init: torch.Tensor, config: ExtendedQuantizerConfig) -> None:
        return None

    def on_iteration_end(
        self,
        *,
        tag: str,
        iter_idx: int,
        max_iters: int,
        U_new: torch.Tensor,
        U_prev: torch.Tensor,
        U_init: torch.Tensor,
        lambda_x: torch.Tensor,
        lambda_x_prev: torch.Tensor,
        lambda_w: torch.Tensor,
        lambda_w_prev: torch.Tensor,
        Z_x: Optional[torch.Tensor],
        Z_x_prev: Optional[torch.Tensor],
        Z_w: Optional[torch.Tensor],
        Z_w_prev: Optional[torch.Tensor],
        config: ExtendedQuantizerConfig,
    ) -> None:
        return None

    def build_state_fields(self) -> Dict[str, Any]:
        return {}


class UTraceObserver(QuantizerObserver):
    def __init__(self, config: ExtendedQuantizerConfig, trace_dir: Optional[Path] = None):
        self.config = config
        self.trace_dir = trace_dir
        self.trace: List[Dict[str, Any]] = []

    def on_fit_start(self, *, tag: str, U_init: torch.Tensor, config: ExtendedQuantizerConfig) -> None:
        self.trace = []

    def on_iteration_end(
        self,
        *,
        tag: str,
        iter_idx: int,
        max_iters: int,
        U_new: torch.Tensor,
        U_prev: torch.Tensor,
        U_init: torch.Tensor,
        lambda_x: torch.Tensor,
        lambda_x_prev: torch.Tensor,
        lambda_w: torch.Tensor,
        lambda_w_prev: torch.Tensor,
        Z_x: Optional[torch.Tensor],
        Z_x_prev: Optional[torch.Tensor],
        Z_w: Optional[torch.Tensor],
        Z_w_prev: Optional[torch.Tensor],
        config: ExtendedQuantizerConfig,
    ) -> None:
        if not config.track_u or iter_idx % max(1, config.track_u_every) != 0:
            return
        point = compute_u_trace_point(
            U=U_new,
            U_prev=U_prev,
            U_init=U_init,
            iter_idx=iter_idx,
            lambda_x=lambda_x,
            lambda_x_prev=lambda_x_prev,
            lambda_w=lambda_w,
            lambda_w_prev=lambda_w_prev,
            Z_x=Z_x if config.track_z_flip_stats else None,
            Z_x_prev=Z_x_prev if config.track_z_flip_stats else None,
            Z_w=Z_w if config.track_z_flip_stats else None,
            Z_w_prev=Z_w_prev if config.track_z_flip_stats else None,
        )
        if self.trace_dir is not None and _should_save_u_matrix(iter_idx, max_iters, config):
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            full_path = self.trace_dir / f"{layer_tag(tag)}_t{iter_idx:04d}.pt"
            torch.save(U_new.detach().cpu(), full_path)
            point["full_matrix_path"] = str(full_path)
        self.trace.append(point)

    def build_state_fields(self) -> Dict[str, Any]:
        return {"u_trace": self.trace}


# ============================================================
# Quantizer: one implementation, optional observers
# ============================================================


class ObservableLatticeLinearQuantizer(base.LatticeLinearQuantizer):
    def __init__(
        self,
        config: ExtendedQuantizerConfig,
        logger: Optional[logging.Logger] = None,
        observers: Optional[Sequence[QuantizerObserver]] = None,
    ):
        super().__init__(config=config, logger=logger)
        self.config = config
        self.observers = list(observers or [])

    def _random_init(self, X: torch.Tensor) -> torch.Tensor:
        d = X.shape[0]
        if self.config.random_init_orthogonal:
            A = torch.randn(d, d, device=X.device, dtype=X.dtype)
            Q, _ = torch.linalg.qr(A)
            return Q
        return self.config.random_init_scale * torch.randn(d, d, device=X.device, dtype=X.dtype)

    def _init_U(self, X: torch.Tensor) -> torch.Tensor:
        mode = self.config.init_mode.lower()
        if mode == "pca":
            return self._pca_init(X)
        if mode == "random":
            return self._random_init(X)
        raise ValueError(f"Unsupported init_mode: {self.config.init_mode}")

    def fit(self, X: torch.Tensor, W: torch.Tensor, tag: str = "") -> TrackingQuantizationState:
        fit_start = time.perf_counter()
        X = X.to(dtype=self.dtype)
        W = W.to(dtype=self.dtype)
        device = X.device
        self.device = device
        self.codebook = self.codebook.to(device)

        d, n = X.shape
        _, m = W.shape
        assert W.shape[0] == d, "X and W must have same feature dimension"

        inv_norms_x = 1.0 / (torch.sum(X * X, dim=0) + self.config.eps)
        inv_norms_w = 1.0 / (torch.sum(W * W, dim=0) + self.config.eps)
        if self.config.error_mode == "absolute":
            inv_norms_x = torch.ones_like(inv_norms_x)
            inv_norms_w = torch.ones_like(inv_norms_w)

        if self.logger is not None:
            self.logger.info(
                "开始拟合量化器 | tag=%s d=%d N=%d M=%d beta=%.4f max_iters=%d tol=%.2e init_mode=%s error_mode=%s codebook=%s",
                tag,
                d,
                n,
                m,
                self.config.beta,
                self.config.max_iters,
                self.config.tol,
                self.config.init_mode,
                self.config.error_mode,
                list(self.config.codebook),
            )

        U = self._init_U(X)
        U_init = U.detach().clone()
        U_prev = U.detach().clone()
        lambda_x = torch.ones(d, dtype=X.dtype, device=device)
        lambda_w = torch.ones(d, dtype=W.dtype, device=device)
        lambda_x_prev = lambda_x.detach().clone()
        lambda_w_prev = lambda_w.detach().clone()
        Z_x = None
        Z_w = None
        Z_x_prev = None
        Z_w_prev = None

        for observer in self.observers:
            observer.on_fit_start(tag=tag, U_init=U_init, config=self.config)

        J_old = float("inf")
        hist_J: List[float] = []
        hist_Jx: List[float] = []
        hist_Jw: List[float] = []
        convergence_iter = self.config.max_iters

        for t in range(1, self.config.max_iters + 1):
            iter_start = time.perf_counter()

            Z_x = self._e_step(X, U, lambda_x)
            Z_w = self._e_step(W, U, lambda_w)

            lambda_x, SXZx, _ = self._update_lambda(X, U, Z_x, inv_norms_x)
            lambda_w, SWZw, _ = self._update_lambda(W, U, Z_w, inv_norms_w)
            U_new = self._update_U(lambda_x, lambda_w, SXZx, SWZw)

            for observer in self.observers:
                observer.on_iteration_end(
                    tag=tag,
                    iter_idx=t,
                    max_iters=self.config.max_iters,
                    U_new=U_new,
                    U_prev=U_prev,
                    U_init=U_init,
                    lambda_x=lambda_x,
                    lambda_x_prev=lambda_x_prev,
                    lambda_w=lambda_w,
                    lambda_w_prev=lambda_w_prev,
                    Z_x=Z_x,
                    Z_x_prev=Z_x_prev,
                    Z_w=Z_w,
                    Z_w_prev=Z_w_prev,
                    config=self.config,
                )

            U = U_new
            U_prev = U.detach().clone()
            lambda_x_prev = lambda_x.detach().clone()
            lambda_w_prev = lambda_w.detach().clone()
            Z_x_prev = Z_x.detach().clone()
            Z_w_prev = Z_w.detach().clone()

            X_hat = U @ (lambda_x.unsqueeze(1) * Z_x)
            W_hat = U @ (lambda_w.unsqueeze(1) * Z_w)
            J_x = objective_reconstruction_error(X, X_hat, inv_norms_x, self.config.error_mode, average=True)
            J_w = objective_reconstruction_error(W, W_hat, inv_norms_w, self.config.error_mode, average=True)
            J = J_x + self.config.beta * J_w

            hist_J.append(float(J.item()))
            hist_Jx.append(float(J_x.item()))
            hist_Jw.append(float(J_w.item()))

            rel_change = float("inf") if not math.isfinite(J_old) else abs(float(J.item()) - J_old) / max(1.0, abs(J_old))
            iter_time = time.perf_counter() - iter_start
            if self.logger is not None and (t == 1 or t % self.config.log_every == 0):
                self.logger.info(
                    "tag=%s iter=%03d | J=%.6f J_x=%.6f J_w=%.6f rel_change=%.6e | time=%.3fs",
                    tag,
                    t,
                    float(J.item()),
                    float(J_x.item()),
                    float(J_w.item()),
                    rel_change,
                    iter_time,
                )

            if t % self.config.convergence_check_every == 0:
                if rel_change < self.config.tol:
                    convergence_iter = t
                    if self.logger is not None:
                        self.logger.info("tag=%s 量化器在第 %d 轮收敛。", tag, t)
                    break
                J_old = float(J.item())

        fit_time_sec = time.perf_counter() - fit_start
        if self.logger is not None:
            self.logger.info("量化器拟合完成 | tag=%s convergence_iter=%d total_fit_time=%.3fs", tag, convergence_iter, fit_time_sec)

        assert Z_x is not None and Z_w is not None
        state_fields: Dict[str, Any] = {}
        for observer in self.observers:
            state_fields.update(observer.build_state_fields())

        return TrackingQuantizationState(
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
            u_trace=state_fields.get("u_trace", []),
            init_mode=self.config.init_mode,
            error_mode=self.config.error_mode,
        )


# ============================================================
# Builders and runner
# ============================================================


class RunConfigResolver:
    def __init__(self, custom_codebooks: Dict[str, Tuple[float, ...]]):
        self.custom_codebooks = custom_codebooks

    def make_config(self, combo: Dict[str, Any]) -> Tuple[RefactorExperimentConfig, Dict[str, Any], Tuple[float, ...], str]:
        cfg = RefactorExperimentConfig()
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
            mapped = path.replace("quant_ext.", "quant.")
            set_dotted_attr(cfg, mapped, value)
            resolved_combo[path] = value

        if "quant_ext.codebook_spec" not in combo:
            codebook = tuple(cfg.quant.codebook)
            codebook_tag = codebook_tag_from_values(codebook)
            resolved_combo["quant_ext.codebook_spec"] = list(codebook)

        return cfg, resolved_combo, codebook, codebook_tag


class QuantizedModuleBuilder:
    def __init__(self, output_dir: Path, logger: logging.Logger):
        self.output_dir = output_dir
        self.logger = logger

    def compute_reconstruction_errors(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        state: TrackingQuantizationState,
        quantizer: ObservableLatticeLinearQuantizer,
        error_mode: str,
        layer_name: str,
    ) -> Dict[str, float]:
        self.logger.info("开始计算重建/近似误差 | layer=%s error_mode=%s", layer_name, error_mode)
        t0 = time.perf_counter()

        device = state.U.device
        dtype = state.U.dtype
        X = X.to(device=device, dtype=dtype)
        W = W.to(device=device, dtype=dtype)
        X_hat = quantizer.reconstruct_X(X, state)
        W_hat = quantizer.reconstruct_W(state)

        if error_mode == "relative":
            err_x = float(torch.sum((X - X_hat) ** 2).item() / max(torch.sum(X ** 2).item(), 1e-12))
            err_w = float(torch.sum((W - W_hat) ** 2).item() / max(torch.sum(W ** 2).item(), 1e-12))
        elif error_mode == "absolute":
            err_x = float(torch.mean((X - X_hat) ** 2).item())
            err_w = float(torch.mean((W - W_hat) ** 2).item())
        else:
            raise ValueError(f"Unsupported error_mode: {error_mode}")

        err_linear = float(base.compute_linear_relative_error(X, W, state, chunk_tokens=128))
        elapsed = time.perf_counter() - t0
        self.logger.info(
            "误差计算完成 | layer=%s err_x=%.6f err_w=%.6f rel_linear_error=%.6f elapsed=%.3fs",
            layer_name,
            err_x,
            err_w,
            err_linear,
            elapsed,
        )
        return {
            f"{error_mode}_recon_error_x": err_x,
            f"{error_mode}_recon_error_w": err_w,
            "rel_linear_error": err_linear,
        }

    def build_ours(
        self,
        target_specs: Dict[str, base.TargetModuleSpec],
        X_calib_by_layer: Dict[str, torch.Tensor],
        quant_config: ExtendedQuantizerConfig,
        device: str,
    ) -> QuantBuildResult:
        self.logger.info("开始构建 Ours 量化模块(observer 版)。")
        modules: Dict[str, nn.Module] = {}
        states: Dict[str, TrackingQuantizationState] = {}
        metrics_by_layer: Dict[str, Dict[str, float]] = {}
        tensor_info: Dict[str, Dict[str, object]] = {}
        plot_paths: Dict[str, Dict[str, str]] = {}
        u_traces: Dict[str, List[Dict[str, Any]]] = {}

        for layer_name, spec in target_specs.items():
            X_calib = X_calib_by_layer[layer_name].to(device=device)
            W = spec.module.weight.detach().T.to(device=device, dtype=base.get_torch_dtype(quant_config.dtype))
            bias = None if spec.module.bias is None else spec.module.bias.detach().to(device=device, dtype=base.get_torch_dtype(quant_config.dtype))

            base.log_tensor_stats(self.logger, f"{layer_name}.X_calib", X_calib)
            base.log_tensor_stats(self.logger, f"{layer_name}.W", W)

            observer = UTraceObserver(quant_config, trace_dir=self.output_dir / U_DIR_NAME)
            quantizer = ObservableLatticeLinearQuantizer(quant_config, logger=self.logger, observers=[observer])
            state = quantizer.fit(X_calib, W, tag=layer_name)
            metrics = self.compute_reconstruction_errors(X_calib, W, state, quantizer, quant_config.error_mode, layer_name)
            quantized_module = base.QuantizedLinear(state, bias=bias)
            quantized_module.to(device)

            modules[layer_name] = quantized_module
            states[layer_name] = state
            metrics_by_layer[layer_name] = metrics
            u_traces[layer_name] = state.u_trace

            tensor_info[f"{layer_name}.X_calib"] = base.tensor_stats(X_calib)
            tensor_info[f"{layer_name}.W"] = base.tensor_stats(W)
            tensor_info[f"{layer_name}.U"] = base.tensor_stats(state.U)
            tensor_info[f"{layer_name}.lambda_x"] = base.tensor_stats(state.lambda_x)
            tensor_info[f"{layer_name}.lambda_w"] = base.tensor_stats(state.lambda_w)
            tensor_info[f"{layer_name}.Z_w"] = base.tensor_stats(state.Z_w)
            if bias is not None:
                tensor_info[f"{layer_name}.bias"] = base.tensor_stats(bias)

            if SAVE_PER_RUN_PLOTS:
                plot_paths[layer_name] = save_loss_plots_short(
                    output_dir=self.output_dir,
                    layer_name=layer_name,
                    objective_history=state.objective_history,
                    objective_x_history=state.objective_x_history,
                    objective_w_history=state.objective_w_history,
                )
                plot_paths[layer_name].update(save_u_trace_plots(self.output_dir, layer_name, state.u_trace))
            else:
                plot_paths[layer_name] = {}

        self.logger.info("Ours 量化模块构建完成。")
        return QuantBuildResult(
            modules=modules,
            states=states,
            metrics_by_layer=metrics_by_layer,
            tensor_info=tensor_info,
            plot_paths=plot_paths,
            u_traces=u_traces,
        )


class ArtifactWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def write_run(self, artifacts: RunArtifacts) -> None:
        if SAVE_PER_RUN_JSON:
            with open(self.output_dir / RUN_JSON_NAME, "w", encoding="utf-8") as f:
                json.dump(asdict(artifacts), f, ensure_ascii=False, indent=2)

        summary_base = base.build_analysis_summary(
            base.ExperimentArtifacts(
                config=artifacts.config,
                quantization_completed=artifacts.quantization_completed,
                baseline_ppl=artifacts.baseline_ppl,
                sq_baseline_ppl=artifacts.sq_baseline_ppl,
                quantized_ppl=artifacts.quantized_ppl,
                sq_metrics=artifacts.sq_metrics,
                sq_metrics_avg=artifacts.sq_metrics_avg,
                quant_metrics=artifacts.quant_metrics,
                quant_metrics_avg=artifacts.quant_metrics_avg,
                convergence_iters=artifacts.convergence_iters,
                objective_histories=artifacts.objective_histories,
                objective_x_histories=artifacts.objective_x_histories,
                objective_w_histories=artifacts.objective_w_histories,
                tensor_info=artifacts.tensor_info,
                timing_info=artifacts.timing_info,
                ppl_eval_info=artifacts.ppl_eval_info,
                plot_paths=artifacts.plot_paths,
                target_info=artifacts.target_info,
            )
        )

        extra_lines = [
            "",
            "附加信息",
            f"- codebook_tag: {artifacts.codebook_tag}",
            f"- codebook: {list(artifacts.codebook)}",
            "",
            "U 变化摘要",
        ]
        for layer_name, trace in artifacts.u_traces.items():
            if trace:
                last = trace[-1]
                extra_lines.append(
                    f"- {layer_name}: len={len(trace)} delta_init={last['delta_fro_from_init']:.6f} "
                    f"angle_init_mean={last['principal_angle_mean_init_deg']:.4f} "
                    f"zx_flip={last['z_x_flip_rate']:.6f} zw_flip={last['z_w_flip_rate']:.6f} "
                    f"orth_err={last['orthogonality_error']:.6f}"
                )
            else:
                extra_lines.append(f"- {layer_name}: len=0")
        (self.output_dir / RUN_SUMMARY_NAME).write_text(summary_base + "\n" + "\n".join(extra_lines), encoding="utf-8")

    def write_sweep_summary(self, rows: List[SweepSummaryRow]) -> None:
        with open(self.output_dir / SWEEP_JSON_NAME, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=2)

        lines = ["===== Refactored sweep summary =====", ""]
        for row in rows:
            lines.append(
                f"{row.run_name} | baseline={row.baseline_ppl:.6f} | ours={row.quantized_ppl:.6f} | sq={row.sq_baseline_ppl:.6f} | "
                f"ours_delta={row.quantized_ppl - row.baseline_ppl:.6f}"
            )
            lines.append(f"  combo={json.dumps(row.combo_resolved, ensure_ascii=False)}")
        (self.output_dir / SWEEP_TEXT_NAME).write_text("\n".join(lines), encoding="utf-8")


class ExperimentRunner:
    def __init__(self, config_resolver: Optional[RunConfigResolver] = None):
        self.config_resolver = config_resolver or RunConfigResolver(CUSTOM_CODEBOOKS)

    def prepare(self, cfg: RefactorExperimentConfig, run_dir: Path) -> PreparedRun:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_short_logger(run_dir)
        timing_info: Dict[str, float] = {}
        ppl_eval_info: Dict[str, Dict[str, float]] = {}

        base.set_seed(cfg.seed)
        model, tokenizer, model_load_time = base.load_model_and_tokenizer(cfg, logger)
        timing_info["model_load_sec"] = model_load_time

        target_specs = base.get_all_block_attention_targets(
            model,
            target_linear_names=cfg.target.target_linear_names,
            block_indices=cfg.target.block_indices,
        )
        resolved_block_indices = sorted({spec.block_index for spec in target_specs.values()})

        calib_text = base.load_text_split(cfg, cfg.data.calib_split, logger)
        eval_text = base.load_text_split(cfg, cfg.data.eval_split, logger)

        baseline_ppl, baseline_eval_stats = base.evaluate_perplexity_sliding_window(
            model=model,
            tokenizer=tokenizer,
            text=eval_text,
            device=cfg.eval.device,
            stride=cfg.eval.stride,
            max_eval_tokens=cfg.data.eval_num_tokens,
            logger=logger,
            tag="baseline_fp",
        )
        ppl_eval_info["baseline_fp"] = baseline_eval_stats

        t0 = time.perf_counter()
        calib_input_ids = base.tokenize_text(calib_text, tokenizer, max_tokens=cfg.data.calib_num_tokens)
        timing_info["tokenize_calib_sec"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        X_calib_by_layer, collected_token_counts = base.collect_target_inputs(
            model=model,
            input_ids=calib_input_ids,
            target_modules={layer_name: spec.module for layer_name, spec in target_specs.items()},
            max_tokens=cfg.data.calib_num_tokens,
            device=cfg.eval.device,
            logger=logger,
        )
        timing_info["collect_target_inputs_sec"] = time.perf_counter() - t0

        return PreparedRun(
            logger=logger,
            model=model,
            tokenizer=tokenizer,
            target_specs=target_specs,
            resolved_block_indices=resolved_block_indices,
            eval_text=eval_text,
            X_calib_by_layer=X_calib_by_layer,
            collected_token_counts=collected_token_counts,
            baseline_ppl=baseline_ppl,
            ppl_eval_info=ppl_eval_info,
            timing_info=timing_info,
        )

    def run_sq_baseline(self, ctx: PreparedRun, cfg: RefactorExperimentConfig) -> Tuple[float, Dict[str, Dict[str, float]], Dict[str, Dict[str, object]], int]:
        t0 = time.perf_counter()
        sq_modules, sq_bits, sq_metrics_by_layer, sq_tensor_info = base.build_sq_target_modules(
            target_specs=ctx.target_specs,
            X_calib_by_layer=ctx.X_calib_by_layer,
            quant_config=cfg.quant,
            logger=ctx.logger,
            device=cfg.eval.device,
        )
        ctx.timing_info["build_sq_baseline_sec"] = time.perf_counter() - t0

        original_modules = base.replace_target_modules(ctx.target_specs, sq_modules)
        try:
            sq_baseline_ppl, sq_eval_stats = base.evaluate_perplexity_sliding_window(
                model=ctx.model,
                tokenizer=ctx.tokenizer,
                text=ctx.eval_text,
                device=cfg.eval.device,
                stride=cfg.eval.stride,
                max_eval_tokens=cfg.data.eval_num_tokens,
                logger=ctx.logger,
                tag="sq_all_blocks_qkvo",
            )
            ctx.ppl_eval_info["sq_all_blocks_qkvo"] = sq_eval_stats
        finally:
            base.restore_target_modules(ctx.target_specs, original_modules)
        return sq_baseline_ppl, sq_metrics_by_layer, sq_tensor_info, sq_bits

    def run_ours(self, ctx: PreparedRun, cfg: RefactorExperimentConfig, run_dir: Path) -> Tuple[float, QuantBuildResult]:
        builder = QuantizedModuleBuilder(run_dir, ctx.logger)
        t0 = time.perf_counter()
        build_result = builder.build_ours(
            target_specs=ctx.target_specs,
            X_calib_by_layer=ctx.X_calib_by_layer,
            quant_config=cfg.quant,
            device=cfg.eval.device,
        )
        ctx.timing_info["fit_quantizer_and_metrics_sec"] = time.perf_counter() - t0
        ctx.timing_info["fit_quantizer_sec_total"] = sum(state.fit_time_sec for state in build_result.states.values())

        original_modules = base.replace_target_modules(ctx.target_specs, build_result.modules)
        try:
            quantized_ppl, quantized_eval_stats = base.evaluate_perplexity_sliding_window(
                model=ctx.model,
                tokenizer=ctx.tokenizer,
                text=ctx.eval_text,
                device=cfg.eval.device,
                stride=cfg.eval.stride,
                max_eval_tokens=cfg.data.eval_num_tokens,
                logger=ctx.logger,
                tag="ours_all_blocks_qkvo",
            )
            ctx.ppl_eval_info["ours_all_blocks_qkvo"] = quantized_eval_stats
        finally:
            base.restore_target_modules(ctx.target_specs, original_modules)
        return quantized_ppl, build_result

    def run(self, combo: Dict[str, Any], run_dir: Path) -> RunArtifacts:
        cfg, combo_resolved, codebook, codebook_tag = self.config_resolver.make_config(combo)
        cfg.output_dir = str(run_dir)
        ctx = self.prepare(cfg, run_dir)

        sq_baseline_ppl, sq_metrics_by_layer, sq_tensor_info, sq_bits = self.run_sq_baseline(ctx, cfg)
        quantized_ppl, ours_result = self.run_ours(ctx, cfg, run_dir)

        merged_tensor_info = dict(ours_result.tensor_info)
        merged_tensor_info.update(sq_tensor_info)
        merged_tensor_info["sq_bitwidth"] = {"value": sq_bits}

        convergence_iters = {layer_name: state.convergence_iter for layer_name, state in ours_result.states.items()}
        objective_histories = {layer_name: state.objective_history for layer_name, state in ours_result.states.items()}
        objective_x_histories = {layer_name: state.objective_x_history for layer_name, state in ours_result.states.items()}
        objective_w_histories = {layer_name: state.objective_w_history for layer_name, state in ours_result.states.items()}

        target_info = {
            "block_indices": ctx.resolved_block_indices,
            "num_blocks": len(ctx.resolved_block_indices),
            "target_linear_names": list(cfg.target.target_linear_names),
            "collected_token_counts": ctx.collected_token_counts,
            "model_name": cfg.data.model_name,
        }

        artifacts = RunArtifacts(
            config=asdict(cfg),
            combo=combo,
            combo_resolved=combo_resolved,
            codebook=codebook,
            codebook_tag=codebook_tag,
            quantization_completed=True,
            baseline_ppl=ctx.baseline_ppl,
            sq_baseline_ppl=sq_baseline_ppl,
            quantized_ppl=quantized_ppl,
            sq_metrics=sq_metrics_by_layer,
            sq_metrics_avg=base.average_metrics(sq_metrics_by_layer),
            quant_metrics=ours_result.metrics_by_layer,
            quant_metrics_avg=base.average_metrics(ours_result.metrics_by_layer),
            convergence_iters=convergence_iters,
            objective_histories=objective_histories,
            objective_x_histories=objective_x_histories,
            objective_w_histories=objective_w_histories,
            u_traces=ours_result.u_traces,
            tensor_info=merged_tensor_info,
            timing_info=ctx.timing_info,
            ppl_eval_info=ctx.ppl_eval_info,
            plot_paths=ours_result.plot_paths,
            target_info=target_info,
        )
        ArtifactWriter(run_dir).write_run(artifacts)
        return artifacts


class SweepRunner:
    def __init__(self, runner: Optional[ExperimentRunner] = None):
        self.runner = runner or ExperimentRunner()

    def run(self, output_root: Path = OUTPUT_ROOT) -> List[SweepSummaryRow]:
        output_root.mkdir(parents=True, exist_ok=True)
        grid_items = [(k, _ensure_iterable(k, v)) for k, v in SWEEP_GRID.items()]
        keys = [k for k, _ in grid_items]
        values = [v for _, v in grid_items]

        rows: List[SweepSummaryRow] = []
        for values_combo in itertools.product(*values):
            combo = dict(zip(keys, values_combo))
            run_name = combo_to_name(combo)
            run_dir = output_root / run_name
            result_file = run_dir / RUN_JSON_NAME

            if result_file.exists():
                print(f"跳过已完成的实验: {run_name}")
                with open(result_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                artifacts = RunArtifacts(**loaded)
            else:
                artifacts = self.runner.run(combo, run_dir)

            rows.append(
                SweepSummaryRow(
                    run_name=run_name,
                    run_dir=str(run_dir),
                    combo=combo,
                    combo_resolved=artifacts.combo_resolved,
                    baseline_ppl=float(artifacts.baseline_ppl),
                    sq_baseline_ppl=float(artifacts.sq_baseline_ppl),
                    quantized_ppl=float(artifacts.quantized_ppl),
                    quant_metrics_avg=artifacts.quant_metrics_avg,
                    sq_metrics_avg=artifacts.sq_metrics_avg,
                    convergence_iters=artifacts.convergence_iters,
                    u_trace_last={layer_name: (trace[-1] if trace else {}) for layer_name, trace in artifacts.u_traces.items()},
                    u_trace_length={layer_name: len(trace) for layer_name, trace in artifacts.u_traces.items()},
                )
            )

        ArtifactWriter(output_root).write_sweep_summary(rows)
        return rows


# ============================================================
# Entry points
# ============================================================


def run_single_example() -> RunArtifacts:
    combo = {
        "target.block_indices": (11,),
        "quant.beta": 1,
        "quant_ext.init_mode": "random",
        "quant_ext.error_mode": "relative",
        "quant_ext.codebook_spec": DEFAULT_CODEBOOK_SPEC,
    }
    run_dir = OUTPUT_ROOT / combo_to_name(combo)
    return ExperimentRunner().run(combo, run_dir)



def main() -> None:
    rows = SweepRunner().run(OUTPUT_ROOT)
    print(json.dumps([asdict(r) for r in rows], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
