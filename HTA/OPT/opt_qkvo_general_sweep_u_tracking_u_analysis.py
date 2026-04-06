from __future__ import annotations

import importlib.util
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


# ============================================================
# User config area
# Edit here directly.
# ============================================================

BASE_SCRIPT_PATH = Path(__file__).with_name("opt_all_blocks_qkvo_experiment_v1.py")
OUTPUT_ROOT = Path("./outputs_opt_qkvo_general_sweep_u_tracking")

BASE_EXPERIMENT_CONFIG_OVERRIDES: Dict[str, Any] = {
    "data.calib_num_tokens": 4096,
    "data.eval_num_tokens": None,
    "eval.stride": 512,
    "target.block_indices": None,  # e.g. (10, 11)
    "target.target_linear_names": ("q_proj", "k_proj", "v_proj", "out_proj"),
    "quant.max_iters": 80,
    "quant.tol": 1e-5,
    "quant.convergence_check_every": 1,
    "quant.log_every": 1,
    "quant.codebook": (-2.0, -1.0, 0.0, 1.0, 2.0),
    "quant.dtype": "float32",
    "quant.eps": 1e-8,
}

# Sweep any dotted-path parameter below. Values must be iterable.
# Common examples:
# - "quant.beta"
# - "quant.max_iters"
# - "quant.tol"
# - "target.block_indices"
# - "data.calib_num_tokens"
# - "quant_ext.init_mode"
# - "quant_ext.error_mode"
SWEEP_GRID: Dict[str, Iterable[Any]] = {
    "target.block_indices": (
        (11,),
        (8, 9, 10, 11),
        (0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    ),
    "quant.beta": (1,),
    "quant_ext.init_mode": ("pca", "random"),
    "quant_ext.error_mode": ("relative",),
}

TRACK_U_FULL_MATRIX = False         # False by default: avoid storing every U snapshot
TRACK_U_EVERY = 1                    # record every n iterations
TRACK_U_SAVE_INTERVAL = 10           # when saving full U, keep every n tracked steps
TRACK_U_SAVE_FIRST = 5               # always keep early iterations up to this index
TRACK_U_SAVE_LAST = True             # always save the final U
TRACK_Z_FLIP_STATS = True            # record code flip statistics induced by U updates
RANDOM_INIT_ORTHOGONAL = True        # random init as orthogonal basis
RANDOM_INIT_SCALE = 1.0              # used only if RANDOM_INIT_ORTHOGONAL=False
SAVE_PER_RUN_JSON = True
SAVE_PER_RUN_PLOTS = True


# ============================================================
# Load base module
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
# Extended configs / states
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
class SweepExperimentConfig(base.ExperimentConfig):
    quant: ExtendedQuantizerConfig = field(default_factory=ExtendedQuantizerConfig)


@dataclass
class UTracePoint:
    iter_idx: int
    delta_fro_from_prev: float
    delta_fro_from_init: float
    cosine_to_prev: float
    cosine_to_init: float
    orthogonality_error: float
    singular_min: float
    singular_max: float
    diag_mean_abs: float
    offdiag_fro: float
    principal_angle_mean_prev_deg: float
    principal_angle_max_prev_deg: float
    principal_angle_mean_init_deg: float
    principal_angle_max_init_deg: float
    subspace_geodesic_prev: float
    subspace_geodesic_init: float
    column_cosine_mean_prev: float
    column_cosine_min_prev: float
    lambda_x_delta_from_prev: float = 0.0
    lambda_w_delta_from_prev: float = 0.0
    z_x_flip_rate: float = 0.0
    z_w_flip_rate: float = 0.0
    z_x_flip_ratio_per_dim_mean: float = 0.0
    z_w_flip_ratio_per_dim_mean: float = 0.0
    full_matrix_path: Optional[str] = None


@dataclass
class TrackingQuantizationState(base.QuantizationState):
    u_trace: List[Dict[str, Any]] = field(default_factory=list)
    init_mode: str = "pca"
    error_mode: str = "relative"


# ============================================================
# Utilities
# ============================================================


def _ensure_iterable(name: str, values: Iterable[Any]) -> Tuple[Any, ...]:
    if isinstance(values, (str, bytes)):
        return (values,)
    try:
        seq = tuple(values)
    except TypeError as e:
        raise TypeError(f"Sweep values for {name!r} must be iterable, got {type(values).__name__}") from e
    if len(seq) == 0:
        raise ValueError(f"Sweep values for {name!r} cannot be empty")
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



def get_dotted_attr(obj: Any, dotted_path: str) -> Any:
    cur = obj
    for p in dotted_path.split("."):
        cur = getattr(cur, p)
    return cur



def sanitize_value(v: Any) -> str:
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        s = f"{v:.6g}"
    else:
        s = str(v)
    for old, new in [("/", "-"), (" ", ""), ("(", ""), (")", ""), (",", "-"), ("'", "")]:
        s = s.replace(old, new)
    return s



def combo_to_name(combo: Dict[str, Any]) -> str:
    pieces: List[str] = []
    for k, v in combo.items():
        short = k.replace("quant_ext.", "").replace("quant.", "q_").replace("target.", "t_").replace("data.", "d_")
        pieces.append(f"{short}_{sanitize_value(v)}")
    return "__".join(pieces)



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

    lambda_x_delta = 0.0
    if lambda_x is not None and lambda_x_prev is not None:
        lambda_x_delta = float(torch.linalg.norm(lambda_x - lambda_x_prev).item())
    lambda_w_delta = 0.0
    if lambda_w is not None and lambda_w_prev is not None:
        lambda_w_delta = float(torch.linalg.norm(lambda_w - lambda_w_prev).item())

    point = UTracePoint(
        iter_idx=int(iter_idx),
        delta_fro_from_prev=float(torch.linalg.norm(U - U_prev).item()),
        delta_fro_from_init=float(torch.linalg.norm(U - U_init).item()),
        cosine_to_prev=_safe_cosine_flat(U, U_prev),
        cosine_to_init=_safe_cosine_flat(U, U_init),
        orthogonality_error=float(torch.linalg.norm(gram - eye).item()),
        singular_min=float(svals.min().item()),
        singular_max=float(svals.max().item()),
        diag_mean_abs=float(torch.mean(torch.abs(diag)).item()),
        offdiag_fro=float(torch.linalg.norm(offdiag).item()),
        principal_angle_mean_prev_deg=angle_prev["mean_deg"],
        principal_angle_max_prev_deg=angle_prev["max_deg"],
        principal_angle_mean_init_deg=angle_init["mean_deg"],
        principal_angle_max_init_deg=angle_init["max_deg"],
        subspace_geodesic_prev=angle_prev["geodesic"],
        subspace_geodesic_init=angle_init["geodesic"],
        column_cosine_mean_prev=col_prev["mean"],
        column_cosine_min_prev=col_prev["min"],
        lambda_x_delta_from_prev=lambda_x_delta,
        lambda_w_delta_from_prev=lambda_w_delta,
        z_x_flip_rate=z_x_flip,
        z_w_flip_rate=z_w_flip,
        z_x_flip_ratio_per_dim_mean=z_x_flip_per_dim,
        z_w_flip_ratio_per_dim_mean=z_w_flip_per_dim,
    )
    return asdict(point)



def save_u_trace_plots(output_dir: Path, prefix: str, u_trace: List[Dict[str, Any]]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    if not u_trace:
        return paths
    xs = [int(p["iter_idx"]) for p in u_trace]

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
        ("delta_fro_from_prev", "Frobenius delta", f"{prefix} U change vs previous", f"{prefix}_u_delta_prev.png"),
        ("delta_fro_from_init", "Frobenius delta", f"{prefix} U change vs init", f"{prefix}_u_delta_init.png"),
        ("cosine_to_prev", "Cosine similarity", f"{prefix} U cosine vs previous", f"{prefix}_u_cos_prev.png"),
        ("cosine_to_init", "Cosine similarity", f"{prefix} U cosine vs init", f"{prefix}_u_cos_init.png"),
        ("orthogonality_error", "Orthogonality error", f"{prefix} U orthogonality error", f"{prefix}_u_orth_error.png"),
        ("principal_angle_mean_prev_deg", "Degrees", f"{prefix} mean principal angle vs previous", f"{prefix}_u_angle_prev_mean.png"),
        ("principal_angle_max_prev_deg", "Degrees", f"{prefix} max principal angle vs previous", f"{prefix}_u_angle_prev_max.png"),
        ("principal_angle_mean_init_deg", "Degrees", f"{prefix} mean principal angle vs init", f"{prefix}_u_angle_init_mean.png"),
        ("principal_angle_max_init_deg", "Degrees", f"{prefix} max principal angle vs init", f"{prefix}_u_angle_init_max.png"),
        ("subspace_geodesic_prev", "Geodesic distance", f"{prefix} subspace geodesic vs previous", f"{prefix}_u_geodesic_prev.png"),
        ("subspace_geodesic_init", "Geodesic distance", f"{prefix} subspace geodesic vs init", f"{prefix}_u_geodesic_init.png"),
        ("column_cosine_mean_prev", "Abs cosine", f"{prefix} column cosine mean vs previous", f"{prefix}_u_col_cos_mean_prev.png"),
        ("column_cosine_min_prev", "Abs cosine", f"{prefix} column cosine min vs previous", f"{prefix}_u_col_cos_min_prev.png"),
        ("lambda_x_delta_from_prev", "L2 delta", f"{prefix} lambda_x change vs previous", f"{prefix}_lambda_x_delta.png"),
        ("lambda_w_delta_from_prev", "L2 delta", f"{prefix} lambda_w change vs previous", f"{prefix}_lambda_w_delta.png"),
        ("z_x_flip_rate", "Flip rate", f"{prefix} Z_x flip rate", f"{prefix}_zx_flip_rate.png"),
        ("z_w_flip_rate", "Flip rate", f"{prefix} Z_w flip rate", f"{prefix}_zw_flip_rate.png"),
    ]
    for args in plot_specs:
        _plot(*args)
    return paths


# ============================================================
# Extended quantizer
# ============================================================


class TrackingLatticeLinearQuantizer(base.LatticeLinearQuantizer):
    def __init__(self, config: ExtendedQuantizerConfig, logger=None, trace_dir: Optional[Path] = None):
        super().__init__(config=config, logger=logger)
        self.config = config
        self.trace_dir = trace_dir

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

        if self.logger is not None:
            self.logger.info(
                "开始拟合量化器 | tag=%s d=%d N=%d M=%d beta=%.4f max_iters=%d tol=%.2e init_mode=%s error_mode=%s",
                tag, d, n, m, self.config.beta, self.config.max_iters, self.config.tol,
                self.config.init_mode, self.config.error_mode,
            )

        U = self._init_U(X)
        U_init = U.detach().clone()
        U_prev = U.detach().clone()
        lambda_x = torch.ones(d, dtype=X.dtype, device=device)
        lambda_w = torch.ones(d, dtype=W.dtype, device=device)
        lambda_x_prev = lambda_x.detach().clone()
        lambda_w_prev = lambda_w.detach().clone()

        J_old = float("inf")
        hist_J: List[float] = []
        hist_Jx: List[float] = []
        hist_Jw: List[float] = []
        u_trace: List[Dict[str, Any]] = []
        convergence_iter = self.config.max_iters
        Z_x = None
        Z_w = None
        Z_x_prev = None
        Z_w_prev = None

        for t in range(1, self.config.max_iters + 1):
            iter_start = time.perf_counter()

            Z_x = self._e_step(X, U, lambda_x)
            Z_w = self._e_step(W, U, lambda_w)

            lambda_x, SXZx, _ = self._update_lambda(X, U, Z_x, inv_norms_x)
            lambda_w, SWZw, _ = self._update_lambda(W, U, Z_w, inv_norms_w)

            U_new = self._update_U(lambda_x, lambda_w, SXZx, SWZw)

            if self.config.track_u and (t % max(1, self.config.track_u_every) == 0):
                point = compute_u_trace_point(
                    U_new,
                    U_prev,
                    U_init,
                    iter_idx=t,
                    lambda_x=lambda_x,
                    lambda_x_prev=lambda_x_prev,
                    lambda_w=lambda_w,
                    lambda_w_prev=lambda_w_prev,
                    Z_x=Z_x if self.config.track_z_flip_stats else None,
                    Z_x_prev=Z_x_prev if self.config.track_z_flip_stats else None,
                    Z_w=Z_w if self.config.track_z_flip_stats else None,
                    Z_w_prev=Z_w_prev if self.config.track_z_flip_stats else None,
                )
                if self.trace_dir is not None and _should_save_u_matrix(t, self.config.max_iters, self.config):
                    self.trace_dir.mkdir(parents=True, exist_ok=True)
                    full_path = self.trace_dir / f"{tag}_U_iter_{t:04d}.pt"
                    torch.save(U_new.detach().cpu(), full_path)
                    point["full_matrix_path"] = str(full_path)
                u_trace.append(point)

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

            if math.isfinite(J_old):
                rel_change = abs(float(J.item()) - J_old) / max(1.0, abs(J_old))
            else:
                rel_change = float("inf")
            iter_time = time.perf_counter() - iter_start

            if self.logger is not None and (t == 1 or t % self.config.log_every == 0):
                self.logger.info(
                    "tag=%s iter=%03d | J=%.6f J_x=%.6f J_w=%.6f rel_change=%.6e | time=%.3fs",
                    tag, t, float(J.item()), float(J_x.item()), float(J_w.item()), rel_change, iter_time,
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
            u_trace=u_trace,
            init_mode=self.config.init_mode,
            error_mode=self.config.error_mode,
        )


# ============================================================
# Extended metrics / builders
# ============================================================


def compute_reconstruction_errors_extended(
    X: torch.Tensor,
    W: torch.Tensor,
    state: TrackingQuantizationState,
    quantizer: TrackingLatticeLinearQuantizer,
    error_mode: str,
    logger,
    layer_name: str,
) -> Dict[str, float]:
    logger.info("开始计算重建/近似误差 | layer=%s error_mode=%s", layer_name, error_mode)
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
    logger.info(
        "误差计算完成 | layer=%s err_x=%.6f err_w=%.6f rel_linear_error=%.6f elapsed=%.3fs",
        layer_name, err_x, err_w, err_linear, elapsed,
    )
    return {
        f"{error_mode}_recon_error_x": err_x,
        f"{error_mode}_recon_error_w": err_w,
        "rel_linear_error": err_linear,
    }



def build_quantized_target_modules_extended(
    target_specs: Dict[str, base.TargetModuleSpec],
    X_calib_by_layer: Dict[str, torch.Tensor],
    quantizer: TrackingLatticeLinearQuantizer,
    logger,
    device: str,
    output_dir: Path,
) -> Tuple[
    Dict[str, torch.nn.Module],
    Dict[str, TrackingQuantizationState],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, object]],
    Dict[str, Dict[str, str]],
]:
    logger.info("开始为目标层构建扩展量化器。")
    quantized_modules: Dict[str, torch.nn.Module] = {}
    states: Dict[str, TrackingQuantizationState] = {}
    metrics_by_layer: Dict[str, Dict[str, float]] = {}
    tensor_info: Dict[str, Dict[str, object]] = {}
    u_plot_paths: Dict[str, Dict[str, str]] = {}

    for layer_name, spec in target_specs.items():
        module = spec.module
        X_calib = X_calib_by_layer[layer_name].to(device=device)
        W = module.weight.detach().T.to(device=device, dtype=base.get_torch_dtype(quantizer.config.dtype))
        bias = None if module.bias is None else module.bias.detach().to(device=device, dtype=base.get_torch_dtype(quantizer.config.dtype))

        base.log_tensor_stats(logger, f"{layer_name}.X_calib", X_calib)
        base.log_tensor_stats(logger, f"{layer_name}.W", W)

        state = quantizer.fit(X_calib, W, tag=layer_name)
        metrics = compute_reconstruction_errors_extended(
            X_calib, W, state, quantizer, quantizer.config.error_mode, logger, layer_name
        )
        quantized_module = base.QuantizedLinear(state, bias=bias)
        quantized_module.to(device)

        quantized_modules[layer_name] = quantized_module
        states[layer_name] = state
        metrics_by_layer[layer_name] = metrics

        tensor_info[f"{layer_name}.X_calib"] = base.tensor_stats(X_calib)
        tensor_info[f"{layer_name}.W"] = base.tensor_stats(W)
        tensor_info[f"{layer_name}.U"] = base.tensor_stats(state.U)
        tensor_info[f"{layer_name}.lambda_x"] = base.tensor_stats(state.lambda_x)
        tensor_info[f"{layer_name}.lambda_w"] = base.tensor_stats(state.lambda_w)
        tensor_info[f"{layer_name}.Z_w"] = base.tensor_stats(state.Z_w)
        if bias is not None:
            tensor_info[f"{layer_name}.bias"] = base.tensor_stats(bias)

        if SAVE_PER_RUN_PLOTS:
            u_plot_paths[layer_name] = save_u_trace_plots(output_dir, layer_name, state.u_trace)

    logger.info("扩展量化模块构建完成。")
    return quantized_modules, states, metrics_by_layer, tensor_info, u_plot_paths


# ============================================================
# Single run / sweep
# ============================================================


@dataclass
class SweepRunArtifacts:
    run_name: str
    run_dir: str
    combo: Dict[str, Any]
    baseline_ppl: float
    sq_baseline_ppl: float
    quantized_ppl: float
    quant_metrics_avg: Dict[str, float]
    sq_metrics_avg: Dict[str, float]
    convergence_iters: Dict[str, int]
    u_trace_last: Dict[str, Dict[str, Any]]
    u_trace_length: Dict[str, int]



def make_config_for_combo(combo: Dict[str, Any]) -> SweepExperimentConfig:
    cfg = SweepExperimentConfig()
    for path, value in BASE_EXPERIMENT_CONFIG_OVERRIDES.items():
        set_dotted_attr(cfg, path, value)
    cfg.quant.track_u = True
    cfg.quant.track_u_every = TRACK_U_EVERY
    cfg.quant.track_u_full_matrix = TRACK_U_FULL_MATRIX
    cfg.quant.track_u_save_interval = TRACK_U_SAVE_INTERVAL
    cfg.quant.track_u_save_first = TRACK_U_SAVE_FIRST
    cfg.quant.track_z_flip_stats = TRACK_Z_FLIP_STATS
    cfg.quant.random_init_orthogonal = RANDOM_INIT_ORTHOGONAL
    cfg.quant.random_init_scale = RANDOM_INIT_SCALE

    for path, value in combo.items():
        mapped = path.replace("quant_ext.", "quant.")
        set_dotted_attr(cfg, mapped, value)
    return cfg



def run_single_combo(combo: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    cfg = make_config_for_combo(combo)
    cfg.output_dir = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = base.setup_logger(run_dir)

    logger.info("========== Sweep 单次实验开始 ==========")
    logger.info("combo = %s", json.dumps(combo, ensure_ascii=False, indent=2, default=str))
    logger.info("config = %s", json.dumps(asdict(cfg), ensure_ascii=False, indent=2, default=str))

    base.set_seed(cfg.seed)
    timing_info: Dict[str, float] = {}
    ppl_eval_info: Dict[str, Dict[str, float]] = {}
    quantization_completed = False

    model, tokenizer, model_load_time = base.load_model_and_tokenizer(cfg, logger)
    timing_info["model_load_sec"] = model_load_time

    target_specs = base.get_all_block_attention_targets(
        model,
        target_linear_names=cfg.target.target_linear_names,
        block_indices=cfg.target.block_indices,
    )
    target_modules = {layer_name: spec.module for layer_name, spec in target_specs.items()}
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
        target_modules=target_modules,
        max_tokens=cfg.data.calib_num_tokens,
        device=cfg.eval.device,
        logger=logger,
    )
    timing_info["collect_target_inputs_sec"] = time.perf_counter() - t0

    # SQ baseline
    t0 = time.perf_counter()
    sq_modules, sq_bits, sq_metrics_by_layer, sq_tensor_info = base.build_sq_target_modules(
        target_specs=target_specs,
        X_calib_by_layer=X_calib_by_layer,
        quant_config=cfg.quant,
        logger=logger,
        device=cfg.eval.device,
    )
    timing_info["build_sq_baseline_sec"] = time.perf_counter() - t0

    original_modules = base.replace_target_modules(target_specs, sq_modules)
    try:
        sq_baseline_ppl, sq_eval_stats = base.evaluate_perplexity_sliding_window(
            model=model,
            tokenizer=tokenizer,
            text=eval_text,
            device=cfg.eval.device,
            stride=cfg.eval.stride,
            max_eval_tokens=cfg.data.eval_num_tokens,
            logger=logger,
            tag="sq_all_blocks_qkvo",
        )
        ppl_eval_info["sq_all_blocks_qkvo"] = sq_eval_stats
    finally:
        base.restore_target_modules(target_specs, original_modules)

    # Ours
    quantizer = TrackingLatticeLinearQuantizer(cfg.quant, logger=logger, trace_dir=run_dir / "u_matrices")
    t0 = time.perf_counter()
    quantized_modules, states, quant_metrics_by_layer, tensor_info, u_plot_paths = build_quantized_target_modules_extended(
        target_specs=target_specs,
        X_calib_by_layer=X_calib_by_layer,
        quantizer=quantizer,
        logger=logger,
        device=cfg.eval.device,
        output_dir=run_dir,
    )
    timing_info["fit_quantizer_and_metrics_sec"] = time.perf_counter() - t0
    timing_info["fit_quantizer_sec_total"] = sum(state.fit_time_sec for state in states.values())

    original_modules = base.replace_target_modules(target_specs, quantized_modules)
    try:
        quantized_ppl, quantized_eval_stats = base.evaluate_perplexity_sliding_window(
            model=model,
            tokenizer=tokenizer,
            text=eval_text,
            device=cfg.eval.device,
            stride=cfg.eval.stride,
            max_eval_tokens=cfg.data.eval_num_tokens,
            logger=logger,
            tag="ours_all_blocks_qkvo",
        )
        ppl_eval_info["ours_all_blocks_qkvo"] = quantized_eval_stats
        quantization_completed = True
    finally:
        base.restore_target_modules(target_specs, original_modules)

    plot_paths: Dict[str, Dict[str, str]] = {}
    if SAVE_PER_RUN_PLOTS:
        for layer_name, state in states.items():
            plot_paths[layer_name] = base.save_loss_plots(
                output_dir=run_dir,
                prefix=layer_name,
                objective_history=state.objective_history,
                objective_x_history=state.objective_x_history,
                objective_w_history=state.objective_w_history,
            )
            plot_paths[layer_name].update(u_plot_paths.get(layer_name, {}))

    merged_tensor_info = dict(tensor_info)
    merged_tensor_info.update(sq_tensor_info)
    merged_tensor_info["sq_bitwidth"] = {"value": sq_bits}

    convergence_iters = {layer_name: state.convergence_iter for layer_name, state in states.items()}
    objective_histories = {layer_name: state.objective_history for layer_name, state in states.items()}
    objective_x_histories = {layer_name: state.objective_x_history for layer_name, state in states.items()}
    objective_w_histories = {layer_name: state.objective_w_history for layer_name, state in states.items()}
    u_traces = {layer_name: state.u_trace for layer_name, state in states.items()}

    target_info = {
        "block_indices": resolved_block_indices,
        "num_blocks": len(resolved_block_indices),
        "target_linear_names": list(cfg.target.target_linear_names),
        "collected_token_counts": collected_token_counts,
        "model_name": cfg.data.model_name,
    }

    artifacts = {
        "config": asdict(cfg),
        "combo": combo,
        "quantization_completed": quantization_completed,
        "baseline_ppl": baseline_ppl,
        "sq_baseline_ppl": sq_baseline_ppl,
        "quantized_ppl": quantized_ppl,
        "sq_metrics": sq_metrics_by_layer,
        "sq_metrics_avg": base.average_metrics(sq_metrics_by_layer),
        "quant_metrics": quant_metrics_by_layer,
        "quant_metrics_avg": base.average_metrics(quant_metrics_by_layer),
        "convergence_iters": convergence_iters,
        "objective_histories": objective_histories,
        "objective_x_histories": objective_x_histories,
        "objective_w_histories": objective_w_histories,
        "u_traces": u_traces,
        "tensor_info": merged_tensor_info,
        "timing_info": timing_info,
        "ppl_eval_info": ppl_eval_info,
        "plot_paths": plot_paths,
        "target_info": target_info,
    }

    if SAVE_PER_RUN_JSON:
        with open(run_dir / "general_sweep_results.json", "w", encoding="utf-8") as f:
            json.dump(artifacts, f, ensure_ascii=False, indent=2)

    summary_text = base.build_analysis_summary(
        base.ExperimentArtifacts(
            config=artifacts["config"],
            quantization_completed=quantization_completed,
            baseline_ppl=baseline_ppl,
            sq_baseline_ppl=sq_baseline_ppl,
            quantized_ppl=quantized_ppl,
            sq_metrics=sq_metrics_by_layer,
            sq_metrics_avg=artifacts["sq_metrics_avg"],
            quant_metrics=quant_metrics_by_layer,
            quant_metrics_avg=artifacts["quant_metrics_avg"],
            convergence_iters=convergence_iters,
            objective_histories=objective_histories,
            objective_x_histories=objective_x_histories,
            objective_w_histories=objective_w_histories,
            tensor_info=merged_tensor_info,
            timing_info=timing_info,
            ppl_eval_info=ppl_eval_info,
            plot_paths=plot_paths,
            target_info=target_info,
        )
    )
    extra_lines = ["", "U 变化摘要"]
    for layer_name, trace in u_traces.items():
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
    (run_dir / "analysis_summary.txt").write_text(summary_text + "\n" + "\n".join(extra_lines), encoding="utf-8")
    logger.info("========== Sweep 单次实验结束 ==========")
    return artifacts



def save_sweep_summary(output_root: Path, run_summaries: List[SweepRunArtifacts]) -> None:
    rows = [asdict(r) for r in run_summaries]
    with open(output_root / "sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    lines = ["===== General sweep summary =====", ""]
    for r in run_summaries:
        lines.append(
            f"{r.run_name} | baseline={r.baseline_ppl:.6f} | ours={r.quantized_ppl:.6f} | sq={r.sq_baseline_ppl:.6f} | "
            f"ours_delta={r.quantized_ppl - r.baseline_ppl:.6f}"
        )
        lines.append(f"  combo={json.dumps(r.combo, ensure_ascii=False)}")
    (output_root / "sweep_summary.txt").write_text("\n".join(lines), encoding="utf-8")



def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    grid_items = [(k, _ensure_iterable(k, v)) for k, v in SWEEP_GRID.items()]
    keys = [k for k, _ in grid_items]
    values = [v for _, v in grid_items]

    run_summaries: List[SweepRunArtifacts] = []
    for values_combo in itertools.product(*values):
        combo = dict(zip(keys, values_combo))
        run_name = combo_to_name(combo)
        run_dir = OUTPUT_ROOT / run_name

        result_file = run_dir / "general_sweep_results.json"
        if result_file.exists():
            print(f"跳过已完成的实验: {run_name}")
            with open(result_file, "r", encoding="utf-8") as f:
                artifacts = json.load(f)
        else:
            artifacts = run_single_combo(combo, run_dir)

        u_trace_last = {
            layer_name: (trace[-1] if trace else {})
            for layer_name, trace in artifacts["u_traces"].items()
        }
        u_trace_length = {
            layer_name: len(trace)
            for layer_name, trace in artifacts["u_traces"].items()
        }
        run_summaries.append(
            SweepRunArtifacts(
                run_name=run_name,
                run_dir=str(run_dir),
                combo=combo,
                baseline_ppl=float(artifacts["baseline_ppl"]),
                sq_baseline_ppl=float(artifacts["sq_baseline_ppl"]),
                quantized_ppl=float(artifacts["quantized_ppl"]),
                quant_metrics_avg=artifacts["quant_metrics_avg"],
                sq_metrics_avg=artifacts["sq_metrics_avg"],
                convergence_iters=artifacts["convergence_iters"],
                u_trace_last=u_trace_last,
                u_trace_length=u_trace_length,
            )
        )

    save_sweep_summary(OUTPUT_ROOT, run_summaries)
    print(json.dumps([asdict(r) for r in run_summaries], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
