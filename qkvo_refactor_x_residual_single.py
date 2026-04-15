from __future__ import annotations


# ==============================================================================
# BEGIN common.py
# ==============================================================================

import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def setup_logger(output_dir: Path, log_name: str = "experiment.log") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"qkvo_refactor_{output_dir.resolve()}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(output_dir / log_name, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def quantize_nearest(z_continuous: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(z_continuous.unsqueeze(-1) - codebook)
    indices = torch.argmin(diff, dim=-1)
    return codebook[indices]


def infer_sq_bitwidth_from_codebook(codebook: Tuple[float, ...]) -> int:
    return int(math.ceil(math.log2(max(len(codebook), 2))))


def scalar_quant_scale_maxabs(x: torch.Tensor, bits: int, eps: float = 1e-8) -> torch.Tensor:
    qmax = float((2 ** (bits - 1)) - 1)
    maxabs = torch.max(torch.abs(x))
    scale = maxabs / qmax
    return torch.clamp(scale, min=eps)


def scalar_quantize_maxabs(
    x: torch.Tensor,
    bits: int,
    scale: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = scalar_quant_scale_maxabs(x, bits=bits, eps=eps)
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    return q * scale, scale


def weighted_cross(X: torch.Tensor, weights: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    return (X * weights.unsqueeze(0)) @ Z.T


def weighted_gram(Z: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (Z * weights.unsqueeze(0)) @ Z.T


def reconstruction_objective(
    X: torch.Tensor,
    X_hat: torch.Tensor,
    weights: torch.Tensor,
    error_mode: str,
    average: bool = False,
) -> torch.Tensor:
    residual = X - X_hat
    per_column_error = torch.sum(residual * residual, dim=0)
    if error_mode == "relative":
        loss = torch.sum(weights * per_column_error)
    elif error_mode == "absolute":
        loss = torch.sum(per_column_error)
    else:
        raise ValueError(f"Unsupported error_mode: {error_mode}")
    if average:
        loss = loss / max(X.shape[1], 1)
    return loss


def tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    data = tensor.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "mean": float(data.mean().item()),
        "std": float(data.std(unbiased=False).item()),
        "min": float(data.min().item()),
        "max": float(data.max().item()),
        "fro_norm": float(torch.linalg.norm(data).item()),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ==============================================================================
# BEGIN config.py
# ==============================================================================

from dataclasses import dataclass, field
from typing import Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


Codebook = Tuple[float, ...]

CODEBOOKS: dict[str, Codebook] = {
    "d5": (-2.0, -1.0, 0.0, 1.0, 2.0),
    "t3": (-1.0, 0.0, 1.0),
    "s7": (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
    "s8": (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
    "2b": (-2.0, -1.0, 0.0, 1.0),
    "3b": (-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0),
    "4b": (-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
    "4b2": (-16.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
}


def parse_codebook(spec: str) -> Codebook:
    spec = spec.strip()
    if spec in CODEBOOKS:
        return CODEBOOKS[spec]

    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if not parts:
        raise ValueError("Codebook cannot be empty.")

    codebook = tuple(sorted(float(part) for part in parts))
    if len(set(codebook)) != len(codebook):
        raise ValueError(f"Codebook has duplicated values: {codebook}")
    return codebook


def parse_block_indices(spec: Optional[str]) -> Optional[Tuple[int, ...]]:
    if spec is None:
        return None
    text = spec.strip().lower()
    if not text or text == "all":
        return None
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def parse_target_linear_names(spec: str) -> Tuple[str, ...]:
    values = tuple(part.strip() for part in spec.split(",") if part.strip())
    if not values:
        raise ValueError("target_linear_names cannot be empty.")
    return values


@dataclass
class DataConfig:
    model_name: str = "facebook/opt-125m"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    calib_split: str = "train"
    eval_split: str = "test"
    calib_num_tokens: int = 4096
    eval_num_tokens: Optional[int] = None
    tokenizer_use_fast: bool = True


@dataclass
class QuantizerConfig:
    beta: float = 1.0
    max_iters: int = 300
    tol: float = 1e-5
    convergence_check_every: int = 1
    log_every: int = 1
    codebook: Codebook = (-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7)
    dtype: str = "float32"
    eps: float = 1e-8
    init_mode: str = "random"
    error_mode: str = "relative"
    latent_mode: str = "discrete"
    latent_bits: int = 4
    ip_reg_gamma: float = 100
    ip_reg_inner_iters: int = 3
    lambda_quantile_init_enable: bool = True
    lambda_quantile_rebalance_enable: bool = False
    lambda_quantile_p: float = 0.95
    lambda_quantile_rho: float = 0.8
    lambda_quantile_alpha: float = 0
    lambda_rebalance_ratio_min: float = 0.8
    lambda_rebalance_ratio_max: float = 1.25
    lambda_min_value: float = 1e-4
    lambda_max_value: float = 1e4
    fit_device: str = "cuda"
    freeze_u_after_init: bool = False

    enable_x_residual: bool = False
    x_residual_codebook: Codebook = (-1.0, 0.0, 1.0)
    x_residual_max_iters: int = 50
    x_residual_tol: float = 1e-6
    x_residual_init_scale: float = 0.25


@dataclass
class EvalConfig:
    stride: int = 512
    device: str = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


@dataclass
class TargetConfig:
    block_indices: Optional[Tuple[int, ...]] = None
    target_linear_names: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    quant: QuantizerConfig = field(default_factory=QuantizerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    output_dir: str = "./origin"
    seed: int = 42
    save_plots: bool = True


# ==============================================================================
# BEGIN quantizer.py
# ==============================================================================

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch



@dataclass
class QuantizationState:
    U: torch.Tensor
    lambda_x: torch.Tensor
    lambda_w: torch.Tensor
    Z_x: torch.Tensor
    Z_w: torch.Tensor
    codebook: torch.Tensor
    objective_history: List[float]
    objective_x_history: List[float]
    objective_w_history: List[float]
    convergence_iter: int
    fit_time_sec: float
    latent_mode: str
    delta_x: Optional[torch.Tensor] = None
    Z_x_res: Optional[torch.Tensor] = None
    residual_codebook: Optional[torch.Tensor] = None
    x_residual_objective_history: List[float] = field(default_factory=list)
    tracking: Dict[str, Any] = field(default_factory=dict)

    @property
    def coeff(self) -> torch.Tensor:
        return self.lambda_x * self.lambda_w

    @property
    def coeff_res(self) -> Optional[torch.Tensor]:
        if self.delta_x is None:
            return None
        return self.delta_x * self.lambda_w


@dataclass
class QuantizerTrackingOptions:
    track_u: bool = False
    track_u_every: int = 1
    track_u_full_matrix: bool = False
    track_u_save_interval: int = 10
    track_u_save_first: int = 5
    track_z_flip_stats: bool = True


def _safe_cosine_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denominator = torch.linalg.norm(a_flat) * torch.linalg.norm(b_flat)
    if float(denominator.item()) <= 1e-12:
        return 0.0
    return float(torch.dot(a_flat, b_flat).item() / denominator.item())


def _principal_angle_stats(A: torch.Tensor, B: torch.Tensor) -> Dict[str, float]:
    A64 = A.to(dtype=torch.float64)
    B64 = B.to(dtype=torch.float64)
    singular_values = torch.linalg.svdvals(A64.T @ B64).clamp(-1.0, 1.0)
    angles = torch.arccos(singular_values)
    angles_deg = angles * (180.0 / math.pi)
    return {
        "mean_deg": float(torch.mean(angles_deg).item()),
        "max_deg": float(torch.max(angles_deg).item()),
        "geodesic": float(torch.linalg.norm(angles).item()),
    }


def _column_cosine_stats(A: torch.Tensor, B: torch.Tensor) -> Dict[str, float]:
    cosine = torch.sum(A * B, dim=0)
    denominator = torch.linalg.norm(A, dim=0) * torch.linalg.norm(B, dim=0)
    denominator = torch.clamp(denominator, min=1e-12)
    cosine = torch.abs(cosine / denominator)
    return {
        "mean": float(torch.mean(cosine).item()),
        "min": float(torch.min(cosine).item()),
    }


def _flip_rate(prev_codes: Optional[torch.Tensor], new_codes: Optional[torch.Tensor]) -> Tuple[float, float]:
    if prev_codes is None or new_codes is None:
        return 0.0, 0.0
    changed = prev_codes != new_codes
    if changed.numel() == 0:
        return 0.0, 0.0
    overall = float(changed.to(torch.float32).mean().item())
    per_dim = changed.to(torch.float32).mean(dim=1)
    return overall, float(per_dim.mean().item())


def _should_save_u_matrix(iteration: int, max_iters: int, options: QuantizerTrackingOptions) -> bool:
    if not options.track_u_full_matrix:
        return False
    if iteration <= max(0, options.track_u_save_first):
        return True
    if iteration == max_iters:
        return True
    interval = max(1, options.track_u_save_interval)
    return iteration % interval == 0


def compute_u_trace_point(
    U: torch.Tensor,
    U_prev: torch.Tensor,
    U_init: torch.Tensor,
    iteration: int,
    lambda_x: Optional[torch.Tensor] = None,
    lambda_x_prev: Optional[torch.Tensor] = None,
    lambda_w: Optional[torch.Tensor] = None,
    lambda_w_prev: Optional[torch.Tensor] = None,
    Z_x: Optional[torch.Tensor] = None,
    Z_x_prev: Optional[torch.Tensor] = None,
    Z_w: Optional[torch.Tensor] = None,
    Z_w_prev: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    identity = torch.eye(U.shape[1], device=U.device, dtype=U.dtype)
    gram = U.T @ U
    diagonal = torch.diag(gram)
    off_diagonal = gram - torch.diag(diagonal)
    singular_values = torch.linalg.svdvals(U)

    angle_prev = _principal_angle_stats(U, U_prev)
    angle_init = _principal_angle_stats(U, U_init)
    column_prev = _column_cosine_stats(U, U_prev)
    z_x_flip, z_x_flip_per_dim = _flip_rate(Z_x_prev, Z_x)
    z_w_flip, z_w_flip_per_dim = _flip_rate(Z_w_prev, Z_w)

    lambda_x_delta = 0.0 if lambda_x is None or lambda_x_prev is None else float(torch.linalg.norm(lambda_x - lambda_x_prev).item())
    lambda_w_delta = 0.0 if lambda_w is None or lambda_w_prev is None else float(torch.linalg.norm(lambda_w - lambda_w_prev).item())

    return {
        "iteration": int(iteration),
        "delta_fro_from_prev": float(torch.linalg.norm(U - U_prev).item()),
        "delta_fro_from_init": float(torch.linalg.norm(U - U_init).item()),
        "cosine_to_prev": _safe_cosine_flat(U, U_prev),
        "cosine_to_init": _safe_cosine_flat(U, U_init),
        "orthogonality_error": float(torch.linalg.norm(gram - identity).item()),
        "singular_min": float(singular_values.min().item()),
        "singular_max": float(singular_values.max().item()),
        "diag_mean_abs": float(torch.mean(torch.abs(diagonal)).item()),
        "offdiag_fro": float(torch.linalg.norm(off_diagonal).item()),
        "principal_angle_mean_prev_deg": angle_prev["mean_deg"],
        "principal_angle_max_prev_deg": angle_prev["max_deg"],
        "principal_angle_mean_init_deg": angle_init["mean_deg"],
        "principal_angle_max_init_deg": angle_init["max_deg"],
        "subspace_geodesic_prev": angle_prev["geodesic"],
        "subspace_geodesic_init": angle_init["geodesic"],
        "column_cosine_mean_prev": column_prev["mean"],
        "column_cosine_min_prev": column_prev["min"],
        "lambda_x_delta_from_prev": lambda_x_delta,
        "lambda_w_delta_from_prev": lambda_w_delta,
        "z_x_flip_rate": z_x_flip,
        "z_w_flip_rate": z_w_flip,
        "z_x_flip_ratio_per_dim_mean": z_x_flip_per_dim,
        "z_w_flip_ratio_per_dim_mean": z_w_flip_per_dim,
    }


class QuantizerObserver:
    def on_fit_start(self, *, tag: str, U_init: torch.Tensor, options: QuantizerTrackingOptions) -> None:
        return None

    def on_iteration_end(
        self,
        *,
        tag: str,
        iteration: int,
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
        options: QuantizerTrackingOptions,
    ) -> None:
        return None

    def build_state_fields(self) -> Dict[str, Any]:
        return {}


class UTraceObserver(QuantizerObserver):
    def __init__(self, options: QuantizerTrackingOptions, trace_dir: Optional[Path] = None):
        self.options = options
        self.trace_dir = trace_dir
        self.trace: List[Dict[str, Any]] = []

    def _layer_prefix(self, layer_name: str) -> str:
        return layer_name.replace(".", "_")

    def on_fit_start(self, *, tag: str, U_init: torch.Tensor, options: QuantizerTrackingOptions) -> None:
        self.trace = []

    def on_iteration_end(
        self,
        *,
        tag: str,
        iteration: int,
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
        options: QuantizerTrackingOptions,
    ) -> None:
        if not options.track_u or iteration % max(1, options.track_u_every) != 0:
            return

        point = compute_u_trace_point(
            U=U_new,
            U_prev=U_prev,
            U_init=U_init,
            iteration=iteration,
            lambda_x=lambda_x,
            lambda_x_prev=lambda_x_prev,
            lambda_w=lambda_w,
            lambda_w_prev=lambda_w_prev,
            Z_x=Z_x if options.track_z_flip_stats else None,
            Z_x_prev=Z_x_prev if options.track_z_flip_stats else None,
            Z_w=Z_w if options.track_z_flip_stats else None,
            Z_w_prev=Z_w_prev if options.track_z_flip_stats else None,
        )

        if self.trace_dir is not None and _should_save_u_matrix(iteration, max_iters, options):
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            matrix_path = self.trace_dir / f"{self._layer_prefix(tag)}_iter{iteration:04d}.pt"
            torch.save(U_new.detach().cpu(), matrix_path)
            point["u_matrix_path"] = str(matrix_path)

        self.trace.append(point)

    def build_state_fields(self) -> Dict[str, Any]:
        return {"u_trace": self.trace}


class LatticeLinearQuantizer:
    def __init__(
        self,
        config: QuantizerConfig,
        logger: Optional[logging.Logger] = None,
        observers: Optional[Sequence[QuantizerObserver]] = None,
        tracking_options: Optional[QuantizerTrackingOptions] = None,
    ):
        self.config = config
        self.dtype = get_torch_dtype(config.dtype)
        self.codebook = torch.tensor(config.codebook, dtype=self.dtype)
        self.residual_codebook = torch.tensor(config.x_residual_codebook, dtype=self.dtype)
        self.logger = logger
        self.observers = list(observers or [])
        self.tracking_options = tracking_options or QuantizerTrackingOptions()

    def _pca_init(self, X: torch.Tensor) -> torch.Tensor:
        mean = X.mean(dim=1, keepdim=True)
        centered = X - mean
        covariance = (centered @ centered.T) / max(X.shape[1], 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        return eigenvectors[:, order]

    def _random_init(self, X: torch.Tensor) -> torch.Tensor:
        random_matrix = torch.randn((X.shape[0], X.shape[0]), dtype=X.dtype, device=X.device)
        Q, _ = torch.linalg.qr(random_matrix)
        return Q

    def _latent_step(self, data: torch.Tensor, U: torch.Tensor, lambda_diag: torch.Tensor) -> torch.Tensor:
        projection = U.T @ data
        safe_lambda = torch.where(
            lambda_diag.abs() < self.config.eps,
            torch.full_like(lambda_diag, self.config.eps),
            lambda_diag,
        )
        return projection / safe_lambda.unsqueeze(1)

    def _e_step(self, data: torch.Tensor, U: torch.Tensor, lambda_diag: torch.Tensor) -> torch.Tensor:
        latent = self._latent_step(data, U, lambda_diag)
        if self.config.latent_mode == "continuous":
            return latent
        return quantize_nearest(latent, self.codebook.to(data.device))

    def _update_lambda(
        self,
        data: torch.Tensor,
        U: torch.Tensor,
        Z: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        SXZ = weighted_cross(data, weights, Z)
        SZZ = weighted_gram(Z, weights)
        numerator = torch.diag(U.T @ SXZ)
        denominator = torch.diag(SZZ)
        lambda_diag = numerator / (denominator + self.config.eps)
        return lambda_diag, SXZ, SZZ

    def _solve_linear_system(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
        A_regularized = A + self.config.eps * eye
        try:
            return torch.linalg.solve(A_regularized, b)
        except RuntimeError:
            return torch.linalg.lstsq(A_regularized, b.unsqueeze(1)).solution.squeeze(1)

    def _lambda_ip_reg_terms_for_x(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        Z_x: torch.Tensor,
        Z_w: torch.Tensor,
        lambda_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        A = Z_x
        B = lambda_w.unsqueeze(1) * Z_w
        H = (A @ A.T) * (B @ B.T)
        XA = X @ A.T
        WB = W @ B.T
        g = torch.sum(XA * WB, dim=0)
        return H, g

    def _lambda_ip_reg_terms_for_w(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        Z_x: torch.Tensor,
        Z_w: torch.Tensor,
        lambda_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        A = lambda_x.unsqueeze(1) * Z_x
        B = Z_w
        H = (A @ A.T) * (B @ B.T)
        XA = X @ A.T
        WB = W @ B.T
        g = torch.sum(XA * WB, dim=0)
        return H, g

    def _update_lambdas(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        U: torch.Tensor,
        Z_x: torch.Tensor,
        Z_w: torch.Tensor,
        weights_x: torch.Tensor,
        weights_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lambda_x, SXZx, SZZx = self._update_lambda(X, U, Z_x, weights_x)
        lambda_w, SWZw, SZZw = self._update_lambda(W, U, Z_w, weights_w)

        gamma = float(self.config.ip_reg_gamma)
        inner_iters = int(self.config.ip_reg_inner_iters)
        if gamma <= 0.0 or inner_iters <= 0:
            return lambda_x, lambda_w, SXZx, SWZw

        G_x = torch.diag(torch.diag(SZZx))
        h_x = torch.diag(U.T @ SXZx)
        G_w = torch.diag(torch.diag(SZZw))
        h_w = torch.diag(U.T @ SWZw)
        beta_w = float(max(self.config.beta, self.config.eps))
        ip_scale = 1.0 / max(Z_x.shape[1] * Z_w.shape[1], 1)

        for _ in range(inner_iters):
            H_x, g_x = self._lambda_ip_reg_terms_for_x(X, W, Z_x, Z_w, lambda_w)
            lambda_x = self._solve_linear_system(G_x + (gamma * ip_scale) * H_x, h_x + (gamma * ip_scale) * g_x)

            H_w, g_w = self._lambda_ip_reg_terms_for_w(X, W, Z_x, Z_w, lambda_x)
            lambda_w = self._solve_linear_system(
                beta_w * G_w + (gamma * ip_scale) * H_w,
                beta_w * h_w + (gamma * ip_scale) * g_w,
            )

        return lambda_x, lambda_w, SXZx, SWZw

    def _compute_ip_regularizer(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        Z_x: torch.Tensor,
        Z_w: torch.Tensor,
        lambda_x: torch.Tensor,
        lambda_w: torch.Tensor,
    ) -> torch.Tensor:
        gamma = float(self.config.ip_reg_gamma)
        if gamma <= 0.0:
            return torch.zeros((), dtype=X.dtype, device=X.device)

        target_inner = X.T @ W
        approx_inner = Z_x.T @ ((lambda_x * lambda_w).unsqueeze(1) * Z_w)
        diff = target_inner - approx_inner
        return gamma * torch.sum(diff * diff) / max(diff.numel(), 1)

    def _update_U(
        self,
        lambda_x: torch.Tensor,
        lambda_w: torch.Tensor,
        SXZx: torch.Tensor,
        SWZw: torch.Tensor,
    ) -> torch.Tensor:
        M = SXZx @ torch.diag(lambda_x) + self.config.beta * SWZw @ torch.diag(lambda_w)
        P, _, Qt = torch.linalg.svd(M, full_matrices=False)
        return P @ Qt

    def _init_lambda_from_quantiles(
        self,
        proj: torch.Tensor,
        qmax: float,
    ) -> torch.Tensor:
        p = float(getattr(self.config, "lambda_quantile_p", 0.95))
        rho = float(getattr(self.config, "lambda_quantile_rho", 0.8))
        eps = float(self.config.eps)
        lambda_min = float(getattr(self.config, "lambda_min_value", 1e-4))
        lambda_max = float(getattr(self.config, "lambda_max_value", 1e4))

        q = torch.quantile(proj.abs(), p, dim=1)
        target = max(rho * qmax, eps)
        lam = q / target
        lam = torch.clamp(lam, min=lambda_min, max=lambda_max)
        return lam

    def _rebalance_lambda_from_quantiles(
        self,
        proj: torch.Tensor,
        lambda_diag: torch.Tensor,
        qmax: float,
    ) -> torch.Tensor:
        p = float(getattr(self.config, "lambda_quantile_p", 0.95))
        rho = float(getattr(self.config, "lambda_quantile_rho", 0.8))
        alpha = float(getattr(self.config, "lambda_quantile_alpha", 0.3))
        eps = float(self.config.eps)
        ratio_min = float(getattr(self.config, "lambda_rebalance_ratio_min", 0.8))
        ratio_max = float(getattr(self.config, "lambda_rebalance_ratio_max", 1.25))
        lambda_min = float(getattr(self.config, "lambda_min_value", 1e-4))
        lambda_max = float(getattr(self.config, "lambda_max_value", 1e4))

        safe_lambda = torch.clamp(lambda_diag, min=eps)
        z_tilde = proj / safe_lambda.unsqueeze(1)
        q = torch.quantile(z_tilde.abs(), p, dim=1)

        target = max(rho * qmax, eps)
        ratio = (q / target).clamp(min=eps).pow(alpha)
        ratio = ratio.clamp(min=ratio_min, max=ratio_max)

        new_lambda = lambda_diag * ratio
        new_lambda = torch.clamp(new_lambda, min=lambda_min, max=lambda_max)
        return new_lambda

    def _fit_activation_residual(
        self,
        X: torch.Tensor,
        U: torch.Tensor,
        lambda_x: torch.Tensor,
        Z_x: torch.Tensor,
        weights_x: torch.Tensor,
        tag: str,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
        if not self.config.enable_x_residual:
            return None, None, {}

        residual_start = time.perf_counter()
        residual_codebook = self.residual_codebook.to(device=X.device, dtype=X.dtype)
        projected_x = U.T @ X
        main_latent = lambda_x.unsqueeze(1) * Z_x
        residual_target = projected_x - main_latent

        residual_energy = float(torch.sum(residual_target * residual_target).item())
        if residual_energy <= float(self.config.eps):
            delta_x = torch.zeros_like(lambda_x)
            Z_x_res = torch.zeros_like(Z_x)
            return delta_x, Z_x_res, {
                "enabled": True,
                "x_residual_fit_time_sec": time.perf_counter() - residual_start,
                "x_residual_objective_history": [0.0],
                "x_residual_initial_objective": 0.0,
                "x_residual_final_objective": 0.0,
                "x_residual_relative_gain": 0.0,
                "x_residual_target_energy": residual_energy,
            }

        codebook_max = float(torch.max(torch.abs(residual_codebook)).item())
        codebook_max = max(codebook_max, float(self.config.eps))
        init_scale = float(max(self.config.x_residual_init_scale, self.config.eps))
        residual_quantile = torch.quantile(residual_target.abs(), 0.95, dim=1)
        delta_x = torch.clamp((residual_quantile / codebook_max) * init_scale, min=self.config.eps)
        safe_delta = torch.clamp(delta_x, min=self.config.eps)
        Z_x_res = residual_target / safe_delta.unsqueeze(1)
        if self.config.latent_mode != "continuous":
            Z_x_res = quantize_nearest(Z_x_res, residual_codebook)

        objective_history: List[float] = []
        previous_objective = float("inf")
        best_delta = delta_x.detach().clone()
        best_Z = Z_x_res.detach().clone()

        max_iters = max(int(self.config.x_residual_max_iters), 1)
        tol = float(self.config.x_residual_tol)
        stop_iter = max_iters

        for iteration in range(1, max_iters + 1):
            recon_latent = delta_x.unsqueeze(1) * Z_x_res
            objective = reconstruction_objective(
                residual_target,
                recon_latent,
                weights_x,
                self.config.error_mode,
                average=True,
            )
            objective_value = float(objective.item())
            objective_history.append(objective_value)

            if objective_value < previous_objective:
                best_delta = delta_x.detach().clone()
                best_Z = Z_x_res.detach().clone()

            if math.isfinite(previous_objective):
                relative_change = abs(objective_value - previous_objective) / max(1.0, abs(previous_objective))
            else:
                relative_change = float("inf")

            if self.logger is not None:
                self.logger.info(
                    "tag=%s x-residual iter=%03d | J_x_res=%.6f rel_change=%.6e",
                    tag,
                    iteration,
                    objective_value,
                    relative_change,
                )

            if iteration > 1 and relative_change < tol:
                stop_iter = iteration
                break

            safe_delta = torch.where(
                delta_x.abs() < self.config.eps,
                torch.full_like(delta_x, self.config.eps),
                delta_x,
            )
            z_cont = residual_target / safe_delta.unsqueeze(1)
            if self.config.latent_mode == "continuous":
                Z_x_res = z_cont
            else:
                Z_x_res = quantize_nearest(z_cont, residual_codebook)

            numerator = torch.sum(residual_target * Z_x_res * weights_x.unsqueeze(0), dim=1)
            denominator = torch.sum(Z_x_res * Z_x_res * weights_x.unsqueeze(0), dim=1)
            updated_delta = numerator / (denominator + self.config.eps)
            zero_mask = denominator <= self.config.eps
            delta_x = torch.where(zero_mask, torch.zeros_like(updated_delta), updated_delta)
            previous_objective = objective_value

        delta_x = best_delta
        Z_x_res = best_Z
        final_objective = objective_history[-1] if objective_history else 0.0
        initial_objective = objective_history[0] if objective_history else final_objective
        relative_gain = 0.0
        if abs(initial_objective) > 1e-12:
            relative_gain = (initial_objective - final_objective) / abs(initial_objective)

        return delta_x, Z_x_res, {
            "enabled": True,
            "x_residual_stop_iter": stop_iter,
            "x_residual_fit_time_sec": time.perf_counter() - residual_start,
            "x_residual_objective_history": objective_history,
            "x_residual_initial_objective": initial_objective,
            "x_residual_final_objective": final_objective,
            "x_residual_relative_gain": relative_gain,
            "x_residual_target_energy": residual_energy,
        }

    def fit(self, X: torch.Tensor, W: torch.Tensor, tag: str = "") -> QuantizationState:
        fit_start = time.perf_counter()
        X = X.to(dtype=self.dtype)
        W = W.to(dtype=self.dtype)
        self.codebook = self.codebook.to(X.device)
        self.residual_codebook = self.residual_codebook.to(X.device)

        if X.ndim != 2 or W.ndim != 2:
            raise ValueError("X and W must both be 2D matrices.")
        if X.shape[0] != W.shape[0]:
            raise ValueError("X and W must share the same feature dimension.")

        feature_dim, num_tokens = X.shape
        _, out_features = W.shape

        weights_x = 1.0 / (torch.sum(X * X, dim=0) + self.config.eps)
        weights_w = 1.0 / (torch.sum(W * W, dim=0) + self.config.eps)
        if self.config.error_mode == "absolute":
            weights_x = torch.ones_like(weights_x)
            weights_w = torch.ones_like(weights_w)

        if self.logger is not None:
            self.logger.info(
                "Fit quantizer | tag=%s d=%d tokens=%d out_features=%d beta=%.4f ip_reg_gamma=%.4f max_iters=%d tol=%.2e freeze_u_after_init=%s enable_x_residual=%s",
                tag,
                feature_dim,
                num_tokens,
                out_features,
                self.config.beta,
                self.config.ip_reg_gamma,
                self.config.max_iters,
                self.config.tol,
                self.config.freeze_u_after_init,
                self.config.enable_x_residual,
            )
            self.logger.info(
                "Lambda quantile init=%s rebalance=%s p=%.3f rho=%.3f alpha=%.3f",
                self.config.lambda_quantile_init_enable,
                self.config.lambda_quantile_rebalance_enable,
                self.config.lambda_quantile_p,
                self.config.lambda_quantile_rho,
                self.config.lambda_quantile_alpha,
            )

        if self.config.init_mode == "pca":
            U = self._pca_init(X)
        elif self.config.init_mode == "random":
            U = self._random_init(X)
        else:
            raise ValueError(f"Unsupported init_mode: {self.config.init_mode}")

        U_init = U.detach().clone()
        U_prev = U.detach().clone()
        qmax = float((2 ** (self.config.latent_bits - 1)) - 1)
        proj_x = U.T @ X
        proj_w = U.T @ W
        if self.config.lambda_quantile_init_enable:
            lambda_x = self._init_lambda_from_quantiles(proj_x, qmax)
            lambda_w = self._init_lambda_from_quantiles(proj_w, qmax)
        else:
            lambda_x = torch.ones(feature_dim, dtype=X.dtype, device=X.device)
            lambda_w = torch.ones(feature_dim, dtype=W.dtype, device=W.device)
        lambda_x_prev = lambda_x.detach().clone()
        lambda_w_prev = lambda_w.detach().clone()
        Z_x_prev = None
        Z_w_prev = None
        best_iter = self.config.max_iters

        objective_history: List[float] = []
        objective_x_history: List[float] = []
        objective_w_history: List[float] = []
        previous_objective = float("inf")

        for observer in self.observers:
            observer.on_fit_start(tag=tag, U_init=U_init, options=self.tracking_options)

        for iteration in range(1, self.config.max_iters + 1):
            iter_start = time.perf_counter()

            Z_x = self._e_step(X, U, lambda_x)
            Z_w = self._e_step(W, U, lambda_w)

            lambda_x, lambda_w, SXZx, SWZw = self._update_lambdas(
                X=X,
                W=W,
                U=U,
                Z_x=Z_x,
                Z_w=Z_w,
                weights_x=weights_x,
                weights_w=weights_w,
            )

            if self.config.lambda_quantile_rebalance_enable:
                proj_x = U.T @ X
                proj_w = U.T @ W
                lambda_x = self._rebalance_lambda_from_quantiles(proj_x, lambda_x, qmax)
                lambda_w = self._rebalance_lambda_from_quantiles(proj_w, lambda_w, qmax)

                Z_x = self._e_step(X, U, lambda_x)
                Z_w = self._e_step(W, U, lambda_w)
                SXZx = weighted_cross(X, weights_x, Z_x)
                SWZw = weighted_cross(W, weights_w, Z_w)

            if self.config.freeze_u_after_init:
                U_new = U
            else:
                U_new = self._update_U(lambda_x, lambda_w, SXZx, SWZw)

            for observer in self.observers:
                observer.on_iteration_end(
                    tag=tag,
                    iteration=iteration,
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
                    options=self.tracking_options,
                )

            U = U_new
            U_prev = U.detach().clone()
            lambda_x_prev = lambda_x.detach().clone()
            lambda_w_prev = lambda_w.detach().clone()
            Z_x_prev = Z_x.detach().clone()
            Z_w_prev = Z_w.detach().clone()

            X_hat = U @ (lambda_x.unsqueeze(1) * Z_x)
            W_hat = U @ (lambda_w.unsqueeze(1) * Z_w)
            objective_x = reconstruction_objective(X, X_hat, weights_x, self.config.error_mode, average=True)
            objective_w = reconstruction_objective(W, W_hat, weights_w, self.config.error_mode, average=True)
            objective_ip = self._compute_ip_regularizer(X, W, Z_x, Z_w, lambda_x, lambda_w)
            objective = objective_x + self.config.beta * objective_w + objective_ip

            objective_history.append(float(objective.item()))
            objective_x_history.append(float(objective_x.item()))
            objective_w_history.append(float(objective_w.item()))

            if math.isfinite(previous_objective):
                relative_change = abs(float(objective.item()) - previous_objective) / max(1.0, abs(previous_objective))
            else:
                relative_change = float("inf")

            zq_x_p95_med = float("nan")
            zq_w_p95_med = float("nan")
            if self.config.lambda_quantile_rebalance_enable:
                p = float(getattr(self.config, "lambda_quantile_p", 0.95))
                z_tilde_x = self._latent_step(X, U, lambda_x)
                z_tilde_w = self._latent_step(W, U, lambda_w)
                zq_x_p95_med = float(torch.quantile(z_tilde_x.abs(), p, dim=1).median().item())
                zq_w_p95_med = float(torch.quantile(z_tilde_w.abs(), p, dim=1).median().item())

            if self.logger is not None and (iteration == 1 or iteration % self.config.log_every == 0):
                if self.config.lambda_quantile_rebalance_enable:
                    self.logger.info(
                        "tag=%s iter=%03d | J=%.6f J_x=%.6f J_w=%.6f J_ip=%.6f rel_change=%.6e zq_x_p95_med=%.6f zq_w_p95_med=%.6f time=%.3fs",
                        tag,
                        iteration,
                        float(objective.item()),
                        float(objective_x.item()),
                        float(objective_w.item()),
                        float(objective_ip.item()),
                        relative_change,
                        zq_x_p95_med,
                        zq_w_p95_med,
                        time.perf_counter() - iter_start,
                    )
                else:
                    self.logger.info(
                        "tag=%s iter=%03d | J=%.6f J_x=%.6f J_w=%.6f J_ip=%.6f rel_change=%.6e time=%.3fs",
                        tag,
                        iteration,
                        float(objective.item()),
                        float(objective_x.item()),
                        float(objective_w.item()),
                        float(objective_ip.item()),
                        relative_change,
                        time.perf_counter() - iter_start,
                    )

            if iteration % self.config.convergence_check_every == 0:
                if relative_change < self.config.tol:
                    best_iter = iteration
                    break
                previous_objective = float(objective.item())

        delta_x, Z_x_res, residual_tracking = self._fit_activation_residual(
            X=X,
            U=U,
            lambda_x=lambda_x,
            Z_x=Z_x,
            weights_x=weights_x,
            tag=tag,
        )

        fit_time = time.perf_counter() - fit_start
        tracking: Dict[str, Any] = {}
        for observer in self.observers:
            tracking.update(observer.build_state_fields())
        tracking.update(residual_tracking)

        return QuantizationState(
            U=U,
            lambda_x=lambda_x,
            lambda_w=lambda_w,
            Z_x=Z_x,
            Z_w=Z_w,
            codebook=self.codebook.detach().clone(),
            objective_history=objective_history,
            objective_x_history=objective_x_history,
            objective_w_history=objective_w_history,
            convergence_iter=best_iter,
            fit_time_sec=fit_time,
            latent_mode=self.config.latent_mode,
            delta_x=delta_x,
            Z_x_res=Z_x_res,
            residual_codebook=self.residual_codebook.detach().clone() if self.config.enable_x_residual else None,
            x_residual_objective_history=residual_tracking.get("x_residual_objective_history", []),
            tracking=tracking,
        )

    def reconstruct_X(self, X: torch.Tensor, state: QuantizationState, include_residual: bool = True) -> torch.Tensor:
        X = X.to(device=state.U.device, dtype=state.U.dtype)
        projected = state.U.T @ X
        safe_lambda = torch.where(
            state.lambda_x.abs() < self.config.eps,
            torch.full_like(state.lambda_x, self.config.eps),
            state.lambda_x,
        )
        Z_x = projected / safe_lambda.unsqueeze(1)
        if state.latent_mode != "continuous":
            Z_x = quantize_nearest(Z_x, state.codebook.to(X.device))

        latent = state.lambda_x.unsqueeze(1) * Z_x
        if include_residual and state.delta_x is not None:
            safe_delta = torch.where(
                state.delta_x.abs() < self.config.eps,
                torch.full_like(state.delta_x, self.config.eps),
                state.delta_x,
            )
            Z_x_res = (projected - latent) / safe_delta.unsqueeze(1)
            if state.latent_mode != "continuous":
                Z_x_res = quantize_nearest(Z_x_res, state.residual_codebook.to(X.device))
            latent = latent + state.delta_x.unsqueeze(1) * Z_x_res
        return state.U @ latent

    def reconstruct_W(self, state: QuantizationState) -> torch.Tensor:
        return state.U @ (state.lambda_w.unsqueeze(1) * state.Z_w)


# ==============================================================================
# BEGIN model_utils.py
# ==============================================================================

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer



@dataclass
class TargetModuleSpec:
    key: str
    block_index: int
    linear_name: str
    parent_module: nn.Module
    module: nn.Linear


class QuantizedLinear(nn.Module):
    def __init__(self, state: QuantizationState, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("U", state.U.detach().clone())
        self.register_buffer("lambda_x", state.lambda_x.detach().clone())
        self.register_buffer("lambda_w", state.lambda_w.detach().clone())
        self.register_buffer("coeff", state.coeff.detach().clone())
        self.register_buffer("Z_w", state.Z_w.detach().clone())
        self.register_buffer("codebook", state.codebook.detach().clone())
        self.latent_mode = state.latent_mode

        self.has_x_residual = state.delta_x is not None and state.residual_codebook is not None
        if self.has_x_residual:
            assert state.delta_x is not None
            assert state.residual_codebook is not None
            self.register_buffer("delta_x", state.delta_x.detach().clone())
            self.register_buffer("coeff_res", state.coeff_res.detach().clone())
            self.register_buffer("residual_codebook", state.residual_codebook.detach().clone())
        else:
            self.delta_x = None
            self.coeff_res = None
            self.residual_codebook = None

        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().clone())

    def _encode_x(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        projected = x.to(self.U.dtype) @ self.U
        safe_lambda_x = torch.where(
            self.lambda_x.abs() < 1e-8,
            torch.full_like(self.lambda_x, 1e-8),
            self.lambda_x,
        )
        z_continuous = projected / safe_lambda_x.unsqueeze(0)
        if self.latent_mode == "continuous":
            z_x = z_continuous
        else:
            z_x = quantize_nearest(z_continuous, self.codebook)

        if not self.has_x_residual:
            return z_x, None

        assert self.delta_x is not None
        assert self.residual_codebook is not None
        latent_main = z_x * self.lambda_x.unsqueeze(0)
        residual_latent = projected - latent_main
        safe_delta_x = torch.where(
            self.delta_x.abs() < 1e-8,
            torch.full_like(self.delta_x, 1e-8),
            self.delta_x,
        )
        q_continuous = residual_latent / safe_delta_x.unsqueeze(0)
        if self.latent_mode == "continuous":
            q_x_res = q_continuous
        else:
            q_x_res = quantize_nearest(q_continuous, self.residual_codebook)
        return z_x, q_x_res

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape[:-1]
        input_dim = hidden_states.shape[-1]
        x_flat = hidden_states.reshape(-1, input_dim)
        z_x, q_x_res = self._encode_x(x_flat)

        output = (z_x * self.coeff.unsqueeze(0)) @ self.Z_w
        if q_x_res is not None:
            output = output + (q_x_res * self.coeff_res.unsqueeze(0)) @ self.Z_w

        if self.bias is not None:
            output = output + self.bias.to(output.dtype).unsqueeze(0)
        return output.to(hidden_states.dtype).reshape(*original_shape, self.Z_w.shape[1])


class ScalarQuantizedXWLinear(nn.Module):
    def __init__(
        self,
        weight_quantized: torch.Tensor,
        x_scale: torch.Tensor,
        bits: int,
        bias: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.register_buffer("weight_quantized", weight_quantized.detach().clone())
        self.register_buffer("x_scale", x_scale.detach().clone().reshape(()))
        self.bits = int(bits)
        self.eps = float(eps)
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().clone())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape[:-1]
        input_dim = hidden_states.shape[-1]
        x_flat = hidden_states.reshape(-1, input_dim)
        x_quantized, _ = scalar_quantize_maxabs(x_flat, bits=self.bits, scale=self.x_scale, eps=self.eps)
        output = x_quantized.to(self.weight_quantized.dtype) @ self.weight_quantized
        if self.bias is not None:
            output = output + self.bias.to(output.dtype).unsqueeze(0)
        return output.to(hidden_states.dtype).reshape(*original_shape, self.weight_quantized.shape[1])


class MultiLinearInputCollector:
    def __init__(self, max_tokens: int):
        self.max_tokens = int(max_tokens)
        self.collected: Dict[str, List[torch.Tensor]] = {}
        self.num_tokens: Dict[str, int] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, layer_name: str):
        def hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
            current = self.num_tokens.get(layer_name, 0)
            if current >= self.max_tokens:
                return

            hidden_states = inputs[0].detach()
            flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
            remaining = self.max_tokens - current
            if flattened.shape[0] > remaining:
                flattened = flattened[:remaining]

            self.collected.setdefault(layer_name, []).append(flattened.cpu())
            self.num_tokens[layer_name] = current + flattened.shape[0]

        return hook

    def register(self, modules: Dict[str, nn.Module]) -> None:
        for layer_name, module in modules.items():
            self.collected[layer_name] = []
            self.num_tokens[layer_name] = 0
            self.handles.append(module.register_forward_pre_hook(self._make_hook(layer_name)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def get_matrices(self) -> Dict[str, torch.Tensor]:
        matrices: Dict[str, torch.Tensor] = {}
        for layer_name, chunks in self.collected.items():
            if not chunks:
                raise RuntimeError(f"No inputs were collected for layer: {layer_name}")
            matrices[layer_name] = torch.cat(chunks, dim=0).T.contiguous()
        return matrices


def load_model_and_tokenizer(config: ExperimentConfig, logger: logging.Logger):
    logger.info("Load model/tokenizer | model=%s", config.data.model_name)
    start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        config.data.model_name,
        use_fast=config.data.tokenizer_use_fast,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.data.model_name)
    model.eval()
    model.to(config.eval.device)

    elapsed = time.perf_counter() - start
    logger.info(
        "Model loaded | hidden_size=%s vocab_size=%s device=%s time=%.3fs",
        getattr(model.config, "hidden_size", "unknown"),
        getattr(model.config, "vocab_size", "unknown"),
        config.eval.device,
        elapsed,
    )
    return model, tokenizer, elapsed


def load_text_split(config: ExperimentConfig, split: str, logger: logging.Logger) -> str:
    logger.info("Load dataset split | split=%s", split)
    start = time.perf_counter()
    dataset = load_dataset(config.data.dataset_name, config.data.dataset_config, split=split)
    if "text" not in dataset.column_names:
        raise ValueError(f"Dataset split {split} has no 'text' column.")
    texts = [text for text in dataset["text"] if isinstance(text, str) and text.strip()]
    merged = "\n\n".join(texts)
    logger.info(
        "Split loaded | split=%s non_empty_rows=%d chars=%d time=%.3fs",
        split,
        len(texts),
        len(merged),
        time.perf_counter() - start,
    )
    return merged


def tokenize_text(text: str, tokenizer, max_tokens: Optional[int] = None) -> torch.Tensor:
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"][0]
    if max_tokens is not None:
        input_ids = input_ids[:max_tokens]
    return input_ids


def get_attention_target_specs(
    model,
    target_linear_names: Tuple[str, ...],
    block_indices: Optional[Tuple[int, ...]],
) -> Dict[str, TargetModuleSpec]:
    decoder_layers = model.model.decoder.layers
    if block_indices is None:
        resolved_block_indices = tuple(range(len(decoder_layers)))
    else:
        resolved_block_indices = tuple(block_indices)

    specs: Dict[str, TargetModuleSpec] = {}
    for block_index in resolved_block_indices:
        block = decoder_layers[block_index]
        attention = block.self_attn
        for linear_name in target_linear_names:
            if not hasattr(attention, linear_name):
                raise AttributeError(f"Block {block_index} has no linear named {linear_name}")
            module = getattr(attention, linear_name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Target {block_index}.{linear_name} is not nn.Linear")

            key = f"block{block_index}.{linear_name}"
            specs[key] = TargetModuleSpec(
                key=key,
                block_index=int(block_index),
                linear_name=linear_name,
                parent_module=attention,
                module=module,
            )
    return specs


def collect_target_inputs(
    model,
    input_ids: torch.Tensor,
    target_modules: Dict[str, nn.Module],
    max_tokens: int,
    device: str,
    logger: logging.Logger,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    logger.info("Collect calibration activations | max_tokens=%d layers=%d", max_tokens, len(target_modules))
    collector = MultiLinearInputCollector(max_tokens=max_tokens)
    collector.register(target_modules)

    max_length = getattr(model.config, "max_position_embeddings", 2048)
    num_chunks = 0
    start = time.perf_counter()
    try:
        with torch.no_grad():
            for chunk_start in range(0, input_ids.numel(), max_length):
                chunk_end = min(chunk_start + max_length, input_ids.numel())
                chunk = input_ids[chunk_start:chunk_end].unsqueeze(0).to(device)
                _ = model(input_ids=chunk)
                num_chunks += 1
                if all(collector.num_tokens.get(name, 0) >= max_tokens for name in target_modules):
                    break
    finally:
        collector.remove()

    matrices = collector.get_matrices()
    elapsed = time.perf_counter() - start
    for layer_name, matrix in matrices.items():
        logger.info(
            "Collected | layer=%s tokens=%d shape=%s chunks=%d time=%.3fs",
            layer_name,
            collector.num_tokens[layer_name],
            list(matrix.shape),
            num_chunks,
            elapsed,
        )
    return matrices, dict(collector.num_tokens)


def build_runtime_bias(module: nn.Linear, runtime_device: str, dtype_name: str) -> Optional[torch.Tensor]:
    if module.bias is None:
        return None
    return module.bias.detach().to(device=runtime_device, dtype=get_torch_dtype(dtype_name))


def build_weight_matrix(module: nn.Linear, fit_device: str, dtype_name: str) -> torch.Tensor:
    return module.weight.detach().T.to(device=fit_device, dtype=get_torch_dtype(dtype_name))


def replace_target_modules(target_specs: Dict[str, TargetModuleSpec], replacements: Dict[str, nn.Module]) -> Dict[str, nn.Module]:
    original_modules: Dict[str, nn.Module] = {}
    for layer_name, replacement in replacements.items():
        spec = target_specs[layer_name]
        original_modules[layer_name] = getattr(spec.parent_module, spec.linear_name)
        setattr(spec.parent_module, spec.linear_name, replacement)
    return original_modules


def restore_target_modules(target_specs: Dict[str, TargetModuleSpec], original_modules: Dict[str, nn.Module]) -> None:
    for layer_name, original_module in original_modules.items():
        spec = target_specs[layer_name]
        setattr(spec.parent_module, spec.linear_name, original_module)


# ==============================================================================
# BEGIN experiment.py
# ==============================================================================

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn



@dataclass
class ExperimentArtifacts:
    config: Dict
    quantization_completed: bool
    baseline_ppl: float
    quantized_ppl: float
    quant_metrics: Dict[str, Dict[str, float]]
    quant_metrics_avg: Dict[str, float]
    convergence_iters: Dict[str, int]
    objective_histories: Dict[str, List[float]]
    objective_x_histories: Dict[str, List[float]]
    objective_w_histories: Dict[str, List[float]]
    tensor_info: Dict[str, Dict[str, object]]
    timing_info: Dict[str, float]
    ppl_eval_info: Dict[str, Dict[str, float]]
    plot_paths: Dict[str, Dict[str, str]]
    tracking_info: Dict[str, Dict[str, object]]
    target_info: Dict[str, object]


def average_metrics(metrics_by_layer: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if not metrics_by_layer:
        return {}
    key_set = set()
    for layer_metrics in metrics_by_layer.values():
        key_set.update(layer_metrics.keys())
    keys = sorted(key_set)
    averaged: Dict[str, float] = {}
    for key in keys:
        values = [layer_metrics[key] for layer_metrics in metrics_by_layer.values() if key in layer_metrics]
        if values:
            averaged[key] = float(sum(values) / len(values))
    return averaged


def layer_prefix(layer_name: str) -> str:
    return layer_name.replace(".", "_")


def save_loss_plots(
    output_dir: Path,
    layer_name: str,
    objective_history: List[float],
    objective_x_history: List[float],
    objective_w_history: List[float],
) -> Dict[str, str]:
    if not objective_history:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = layer_prefix(layer_name)
    xs = list(range(1, len(objective_history) + 1))
    paths: Dict[str, str] = {}

    plt.figure(figsize=(8, 5))
    plt.plot(xs, objective_history, marker="o", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("J")
    plt.title(f"{layer_name} total objective")
    plt.grid(True, alpha=0.3)
    total_path = output_dir / f"{prefix}_objective_total.png"
    plt.tight_layout()
    plt.savefig(total_path, dpi=160)
    plt.close()
    paths["objective_total"] = str(total_path)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, objective_x_history, marker="o", linewidth=1.5, label="J_x")
    plt.plot(xs, objective_w_history, marker="o", linewidth=1.5, label="J_w")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"{layer_name} objective components")
    plt.legend()
    plt.grid(True, alpha=0.3)
    components_path = output_dir / f"{prefix}_objective_components.png"
    plt.tight_layout()
    plt.savefig(components_path, dpi=160)
    plt.close()
    paths["objective_components"] = str(components_path)
    return paths


def save_distribution_histogram(
    output_dir: Path,
    filename: str,
    values: torch.Tensor,
    title: str,
    bins: int = 120,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    flat = values.detach().float().reshape(-1).cpu().numpy()
    plt.figure(figsize=(8, 5))
    plt.hist(flat, bins=bins, color="#2a6f97", alpha=0.85)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def save_z_tilde_plots(
    output_dir: Path,
    layer_name: str,
    X_calib: torch.Tensor,
    W: torch.Tensor,
    state: QuantizationState,
) -> Dict[str, str]:
    prefix = layer_prefix(layer_name)
    U_x = state.U.to(device=X_calib.device, dtype=X_calib.dtype)
    U_w = state.U.to(device=W.device, dtype=W.dtype)
    lambda_x = state.lambda_x.to(device=X_calib.device, dtype=X_calib.dtype)
    lambda_w = state.lambda_w.to(device=W.device, dtype=W.dtype)

    safe_lambda_x = torch.where(
        lambda_x.abs() < 1e-8,
        torch.full_like(lambda_x, 1e-8),
        lambda_x,
    )
    safe_lambda_w = torch.where(
        lambda_w.abs() < 1e-8,
        torch.full_like(lambda_w, 1e-8),
        lambda_w,
    )

    z_tilde_x = (U_x.T @ X_calib) / safe_lambda_x.unsqueeze(1)
    z_tilde_w = (U_w.T @ W) / safe_lambda_w.unsqueeze(1)

    paths = {
        "z_tilde_x_hist": save_distribution_histogram(
            output_dir=output_dir,
            filename=f"{prefix}_z_tilde_x_hist.png",
            values=z_tilde_x,
            title=f"{layer_name} z_tilde = (U^T X) / lambda_x",
        ),
        "z_tilde_w_hist": save_distribution_histogram(
            output_dir=output_dir,
            filename=f"{prefix}_z_tilde_w_hist.png",
            values=z_tilde_w,
            title=f"{layer_name} z_tilde = (U^T W) / lambda_w",
        ),
    }

    if state.delta_x is not None and state.Z_x_res is not None:
        projected_x = U_x.T @ X_calib
        latent_main = state.lambda_x.to(device=X_calib.device, dtype=X_calib.dtype).unsqueeze(1) * state.Z_x.to(device=X_calib.device, dtype=X_calib.dtype)
        residual_latent = projected_x - latent_main
        safe_delta_x = torch.where(
            state.delta_x.to(device=X_calib.device, dtype=X_calib.dtype).abs() < 1e-8,
            torch.full_like(state.delta_x.to(device=X_calib.device, dtype=X_calib.dtype), 1e-8),
            state.delta_x.to(device=X_calib.device, dtype=X_calib.dtype),
        )
        z_tilde_x_res = residual_latent / safe_delta_x.unsqueeze(1)
        paths["z_tilde_x_res_hist"] = save_distribution_histogram(
            output_dir=output_dir,
            filename=f"{prefix}_z_tilde_x_res_hist.png",
            values=z_tilde_x_res,
            title=f"{layer_name} z_tilde residual = (U^T X - lambda_x Z_x) / delta_x",
        )

    return paths


@torch.no_grad()
def compute_linear_relative_error(
    X: torch.Tensor,
    W: torch.Tensor,
    state: QuantizationState,
    chunk_tokens: int = 128,
    include_residual: bool = True,
) -> float:
    X = X.to(device=state.U.device, dtype=state.U.dtype)
    W = W.to(device=state.U.device, dtype=state.U.dtype)

    coeff = state.coeff
    denominator = 0.0
    numerator = 0.0

    codebook = state.codebook.to(X.device)
    residual_codebook = state.residual_codebook.to(X.device) if state.residual_codebook is not None else None
    coeff_res = state.coeff_res if include_residual else None

    for start in range(0, X.shape[1], chunk_tokens):
        end = min(start + chunk_tokens, X.shape[1])
        X_chunk = X[:, start:end]
        Y_true = X_chunk.T @ W

        projected = state.U.T @ X_chunk
        safe_lambda = torch.where(
            state.lambda_x.abs() < 1e-8,
            torch.full_like(state.lambda_x, 1e-8),
            state.lambda_x,
        )
        Z_x = projected / safe_lambda.unsqueeze(1)
        if state.latent_mode != "continuous":
            diff = torch.abs(Z_x.unsqueeze(-1) - codebook)
            indices = torch.argmin(diff, dim=-1)
            Z_x = codebook[indices]

        Y_hat = (Z_x.T * coeff.unsqueeze(0)) @ state.Z_w
        if coeff_res is not None and state.delta_x is not None and residual_codebook is not None:
            safe_delta_x = torch.where(
                state.delta_x.abs() < 1e-8,
                torch.full_like(state.delta_x, 1e-8),
                state.delta_x,
            )
            residual_latent = projected - state.lambda_x.unsqueeze(1) * Z_x
            Q_x_res = residual_latent / safe_delta_x.unsqueeze(1)
            if state.latent_mode != "continuous":
                diff_res = torch.abs(Q_x_res.unsqueeze(-1) - residual_codebook)
                indices_res = torch.argmin(diff_res, dim=-1)
                Q_x_res = residual_codebook[indices_res]
            Y_hat = Y_hat + (Q_x_res.T * coeff_res.unsqueeze(0)) @ state.Z_w

        delta = Y_true - Y_hat
        numerator += float(torch.sum(delta * delta).item())
        denominator += float(torch.sum(Y_true * Y_true).item())

    return numerator / max(denominator, 1e-12)


def compute_quant_metrics(
    X: torch.Tensor,
    W: torch.Tensor,
    state: QuantizationState,
    quantizer: LatticeLinearQuantizer,
    error_mode: str,
) -> Dict[str, float]:
    X = X.to(device=state.U.device, dtype=state.U.dtype)
    W = W.to(device=state.U.device, dtype=state.U.dtype)

    X_hat_main = quantizer.reconstruct_X(X, state, include_residual=False)
    X_hat = quantizer.reconstruct_X(X, state, include_residual=True)
    W_hat = quantizer.reconstruct_W(state)

    if error_mode == "relative":
        x_error_main = float(torch.sum((X - X_hat_main) ** 2).item() / max(torch.sum(X**2).item(), 1e-12))
        x_error = float(torch.sum((X - X_hat) ** 2).item() / max(torch.sum(X**2).item(), 1e-12))
        w_error = float(torch.sum((W - W_hat) ** 2).item() / max(torch.sum(W**2).item(), 1e-12))
    elif error_mode == "absolute":
        x_error_main = float(torch.mean((X - X_hat_main) ** 2).item())
        x_error = float(torch.mean((X - X_hat) ** 2).item())
        w_error = float(torch.mean((W - W_hat) ** 2).item())
    else:
        raise ValueError(f"Unsupported error_mode: {error_mode}")

    metrics = {
        "x_error_main": x_error_main,
        "x_error": x_error,
        "w_error": w_error,
        "linear_error_main": float(compute_linear_relative_error(X, W, state, include_residual=False)),
        "linear_error": float(compute_linear_relative_error(X, W, state, include_residual=True)),
    }

    if state.delta_x is not None and state.Z_x_res is not None:
        projected_x = state.U.T @ X
        residual_target = projected_x - state.lambda_x.unsqueeze(1) * state.Z_x
        residual_hat = state.delta_x.unsqueeze(1) * state.Z_x_res
        metrics["x_residual_latent_error"] = float(
            torch.sum((residual_target - residual_hat) ** 2).item() / max(torch.sum(residual_target**2).item(), 1e-12)
        )
    return metrics


@torch.no_grad()
def evaluate_perplexity_sliding_window(
    model,
    tokenizer,
    text: str,
    device: str,
    stride: int,
    max_eval_tokens: Optional[int],
    logger: Optional[logging.Logger],
    tag: str,
) -> Tuple[float, Dict[str, float]]:
    if logger is not None:
        logger.info("Evaluate PPL | tag=%s stride=%d max_eval_tokens=%s", tag, stride, str(max_eval_tokens))

    start = time.perf_counter()
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"][0]
    if max_eval_tokens is not None:
        input_ids = input_ids[:max_eval_tokens]

    input_ids = input_ids.to(device)
    max_length = getattr(model.config, "max_position_embeddings", 2048)
    nlls = []
    prev_end = 0
    total_target_tokens = 0
    num_windows = 0

    for begin in range(0, input_ids.size(0), stride):
        end = min(begin + max_length, input_ids.size(0))
        target_length = end - prev_end
        chunk = input_ids[begin:end].unsqueeze(0)
        target_ids = chunk.clone()
        target_ids[:, :-target_length] = -100

        outputs = model(input_ids=chunk, labels=target_ids)
        neg_log_likelihood = outputs.loss * target_length

        nlls.append(neg_log_likelihood)
        total_target_tokens += target_length
        prev_end = end
        num_windows += 1
        if end == input_ids.size(0):
            break

    total_nll = torch.stack(nlls).sum()
    avg_nll = total_nll / total_target_tokens
    ppl = torch.exp(avg_nll)
    elapsed = time.perf_counter() - start
    stats = {
        "elapsed_sec": float(elapsed),
        "num_windows": float(num_windows),
        "num_eval_tokens": float(input_ids.numel()),
        "num_target_tokens": float(total_target_tokens),
        "avg_nll": float(avg_nll.item()),
        "total_nll": float(total_nll.item()),
    }
    if logger is not None:
        logger.info(
            "PPL done | tag=%s ppl=%.6f eval_tokens=%d target_tokens=%d windows=%d time=%.3fs",
            tag,
            float(ppl.item()),
            input_ids.numel(),
            total_target_tokens,
            num_windows,
            elapsed,
        )
    return float(ppl.item()), stats


def build_quantized_modules(
    target_specs: Dict[str, TargetModuleSpec],
    X_calib_by_layer: Dict[str, torch.Tensor],
    config: ExperimentConfig,
    logger: logging.Logger,
    tracking_options: Optional[QuantizerTrackingOptions] = None,
) -> Tuple[
    Dict[str, nn.Module],
    Dict[str, QuantizationState],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, object]],
    Dict[str, Dict[str, str]],
    Dict[str, Dict[str, object]],
]:
    logger.info("Build quantized modules")

    quantized_modules: Dict[str, nn.Module] = {}
    states: Dict[str, QuantizationState] = {}
    metrics_by_layer: Dict[str, Dict[str, float]] = {}
    tensor_info: Dict[str, Dict[str, object]] = {}
    plot_paths: Dict[str, Dict[str, str]] = {}
    tracking_by_layer: Dict[str, Dict[str, object]] = {}
    plots_dir = Path(config.output_dir) / "plots"
    tracking_dir = Path(config.output_dir) / "tracking" / "u_matrices"

    for layer_name, spec in target_specs.items():
        W = build_weight_matrix(spec.module, fit_device=config.quant.fit_device, dtype_name=config.quant.dtype)
        bias = build_runtime_bias(spec.module, runtime_device=config.eval.device, dtype_name=config.quant.dtype)

        observers = []
        if tracking_options is not None and tracking_options.track_u:
            observers.append(UTraceObserver(tracking_options, trace_dir=tracking_dir))

        quantizer = LatticeLinearQuantizer(
            config.quant,
            logger=logger,
            observers=observers,
            tracking_options=tracking_options,
        )
        X_calib = X_calib_by_layer[layer_name].to(device=config.quant.fit_device, dtype=quantizer.dtype)
        state = quantizer.fit(X_calib, W, tag=layer_name)
        metrics_by_layer[layer_name] = compute_quant_metrics(X_calib, W, state, quantizer, config.quant.error_mode)
        quantized_modules[layer_name] = QuantizedLinear(state, bias=bias).to(config.eval.device)
        states[layer_name] = state
        tracking_by_layer[layer_name] = state.tracking

        tensor_info[f"{layer_name}.X_calib"] = tensor_stats(X_calib)
        tensor_info[f"{layer_name}.W"] = tensor_stats(W)
        tensor_info[f"{layer_name}.U"] = tensor_stats(state.U)
        tensor_info[f"{layer_name}.lambda_x"] = tensor_stats(state.lambda_x)
        tensor_info[f"{layer_name}.lambda_w"] = tensor_stats(state.lambda_w)
        tensor_info[f"{layer_name}.Z_w"] = tensor_stats(state.Z_w)
        if state.delta_x is not None:
            tensor_info[f"{layer_name}.delta_x"] = tensor_stats(state.delta_x)
        if state.Z_x_res is not None:
            tensor_info[f"{layer_name}.Z_x_res"] = tensor_stats(state.Z_x_res)
        if bias is not None:
            tensor_info[f"{layer_name}.bias"] = tensor_stats(bias)

        if config.save_plots:
            plot_paths[layer_name] = save_loss_plots(
                output_dir=plots_dir,
                layer_name=layer_name,
                objective_history=state.objective_history,
                objective_x_history=state.objective_x_history,
                objective_w_history=state.objective_w_history,
            )
            plot_paths[layer_name].update(
                save_z_tilde_plots(
                    output_dir=plots_dir,
                    layer_name=layer_name,
                    X_calib=X_calib,
                    W=W,
                    state=state,
                )
            )
        else:
            plot_paths[layer_name] = {}

    return quantized_modules, states, metrics_by_layer, tensor_info, plot_paths, tracking_by_layer


def build_summary(artifacts: ExperimentArtifacts) -> str:
    lines = [
        "QKVO Refactor Summary",
        f"- baseline_ppl: {artifacts.baseline_ppl:.6f}",
        f"- quantized_ppl: {artifacts.quantized_ppl:.6f}",
        f"- quantized_delta: {artifacts.quantized_ppl - artifacts.baseline_ppl:.6f}",
    ]

    lines.append("")
    lines.append("Average Quant Metrics")
    for key, value in artifacts.quant_metrics_avg.items():
        lines.append(f"- {key}: {value:.6f}")

    lines.append("")
    lines.append("Target Info")
    for key, value in artifacts.target_info.items():
        lines.append(f"- {key}: {value}")

    return "\n".join(lines)


def run_experiment(
    config: ExperimentConfig,
    tracking_options: Optional[QuantizerTrackingOptions] = None,
) -> ExperimentArtifacts:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)

    logger.info("=== Experiment start ===")
    logger.info("Config:\n%s", json.dumps(asdict(config), ensure_ascii=False, indent=2))

    timing_info: Dict[str, float] = {}
    ppl_eval_info: Dict[str, Dict[str, float]] = {}
    quantization_completed = False

    model, tokenizer, model_load_time = load_model_and_tokenizer(config, logger)
    timing_info["model_load_sec"] = model_load_time

    target_specs = get_attention_target_specs(
        model=model,
        target_linear_names=config.target.target_linear_names,
        block_indices=config.target.block_indices,
    )
    target_modules = {layer_name: spec.module for layer_name, spec in target_specs.items()}
    resolved_block_indices = sorted({spec.block_index for spec in target_specs.values()})
    logger.info(
        "Targets resolved | num_blocks=%d block_indices=%s target_layers=%d",
        len(resolved_block_indices),
        resolved_block_indices,
        len(target_specs),
    )

    calib_text = load_text_split(config, config.data.calib_split, logger)
    eval_text = load_text_split(config, config.data.eval_split, logger)

    baseline_ppl, baseline_eval_stats = evaluate_perplexity_sliding_window(
        model=model,
        tokenizer=tokenizer,
        text=eval_text,
        device=config.eval.device,
        stride=config.eval.stride,
        max_eval_tokens=config.data.eval_num_tokens,
        logger=logger,
        tag="baseline_fp",
    )
    ppl_eval_info["baseline_fp"] = baseline_eval_stats

    start = time.perf_counter()
    calib_input_ids = tokenize_text(calib_text, tokenizer, max_tokens=config.data.calib_num_tokens)
    timing_info["tokenize_calib_sec"] = time.perf_counter() - start

    start = time.perf_counter()
    X_calib_by_layer, collected_token_counts = collect_target_inputs(
        model=model,
        input_ids=calib_input_ids,
        target_modules=target_modules,
        max_tokens=config.data.calib_num_tokens,
        device=config.eval.device,
        logger=logger,
    )
    timing_info["collect_target_inputs_sec"] = time.perf_counter() - start

    start = time.perf_counter()
    quantized_modules, states, quant_metrics_by_layer, tensor_info, plot_paths, tracking_info = build_quantized_modules(
        target_specs=target_specs,
        X_calib_by_layer=X_calib_by_layer,
        config=config,
        logger=logger,
        tracking_options=tracking_options,
    )
    timing_info["build_quantized_modules_sec"] = time.perf_counter() - start
    timing_info["fit_quantizer_sec_total"] = float(sum(state.fit_time_sec for state in states.values()))

    original_modules = replace_target_modules(target_specs, quantized_modules)
    try:
        quantized_ppl, quantized_eval_stats = evaluate_perplexity_sliding_window(
            model=model,
            tokenizer=tokenizer,
            text=eval_text,
            device=config.eval.device,
            stride=config.eval.stride,
            max_eval_tokens=config.data.eval_num_tokens,
            logger=logger,
            tag="quantized",
        )
        ppl_eval_info["quantized"] = quantized_eval_stats
        quantization_completed = True
    finally:
        restore_target_modules(target_specs, original_modules)

    artifacts = ExperimentArtifacts(
        config=asdict(config),
        quantization_completed=quantization_completed,
        baseline_ppl=baseline_ppl,
        quantized_ppl=quantized_ppl,
        quant_metrics=quant_metrics_by_layer,
        quant_metrics_avg=average_metrics(quant_metrics_by_layer),
        convergence_iters={layer_name: state.convergence_iter for layer_name, state in states.items()},
        objective_histories={layer_name: state.objective_history for layer_name, state in states.items()},
        objective_x_histories={layer_name: state.objective_x_history for layer_name, state in states.items()},
        objective_w_histories={layer_name: state.objective_w_history for layer_name, state in states.items()},
        tensor_info=tensor_info,
        timing_info=timing_info,
        ppl_eval_info=ppl_eval_info,
        plot_paths=plot_paths,
        tracking_info=tracking_info,
        target_info={
            "block_indices": resolved_block_indices,
            "target_linear_names": list(config.target.target_linear_names),
            "num_blocks": len(resolved_block_indices),
            "num_target_layers": len(target_specs),
            "collected_token_counts": collected_token_counts,
            "model_name": config.data.model_name,
            "latent_mode": config.quant.latent_mode,
            "fit_device": config.quant.fit_device,
            "runtime_device": config.eval.device,
            "enable_x_residual": config.quant.enable_x_residual,
            "x_residual_codebook": list(config.quant.x_residual_codebook),
        },
    )

    write_json(output_dir / "results.json", asdict(artifacts))
    write_text(output_dir / "summary.txt", build_summary(artifacts))
    logger.info("Artifacts saved to %s", output_dir)
    logger.info("=== Experiment end ===")
    return artifacts


# ==============================================================================
# BEGIN cli.py
# ==============================================================================

import argparse
import json
import math
from dataclasses import asdict



def add_experiment_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model-name", default="facebook/opt-125m")
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--calib-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--calib-num-tokens", type=int, default=4096)
    parser.add_argument("--eval-num-tokens", type=int, default=None)
    parser.add_argument("--block-indices", default="8,9,10,11", help="Comma-separated block indices, or 'all'.")
    parser.add_argument("--target-linear-names", default="q_proj,k_proj,v_proj,out_proj")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-iters", type=int, default=80)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--codebook", default="-4,-3,-2,-1,0,1,2,3", help="Codebook alias or comma-separated float values.")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--init-mode", choices=("random", "pca"), default="random")
    parser.add_argument("--error-mode", choices=("relative", "absolute"), default="relative")
    parser.add_argument("--latent-mode", choices=("discrete", "continuous"), default="discrete")
    parser.add_argument("--ip-reg-gamma", type=float, default=None)
    parser.add_argument("--ip-reg-inner-iters", type=int, default=1)
    parser.add_argument("--fit-device", default="cuda")
    parser.add_argument("--freeze-u-after-init", action="store_true")

    parser.add_argument("--enable-x-residual", action="store_true")
    parser.add_argument(
        "--x-residual-codebook",
        default="-1,0,1",
        help="Residual codebook alias or comma-separated float values for X latent residual stage.",
    )
    parser.add_argument("--x-residual-max-iters", type=int, default=50)
    parser.add_argument("--x-residual-tol", type=float, default=1e-6)
    parser.add_argument("--x-residual-init-scale", type=float, default=0.25)

    parser.add_argument("--device", default=None, help="Runtime device for model eval. Defaults to torch auto detection.")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./origin")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean QKVO refactor for OPT attention projection quantization.")
    return add_experiment_args(parser)


def args_to_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig()

    config.data.model_name = args.model_name
    config.data.dataset_name = args.dataset_name
    config.data.dataset_config = args.dataset_config
    config.data.calib_split = args.calib_split
    config.data.eval_split = args.eval_split
    config.data.calib_num_tokens = args.calib_num_tokens
    config.data.eval_num_tokens = args.eval_num_tokens

    config.target.block_indices = parse_block_indices(args.block_indices)
    config.target.target_linear_names = parse_target_linear_names(args.target_linear_names)

    config.quant.beta = args.beta
    config.quant.max_iters = args.max_iters
    config.quant.tol = args.tol
    config.quant.codebook = parse_codebook(args.codebook)
    config.quant.latent_bits = int(math.ceil(math.log2(max(len(config.quant.codebook), 2))))
    config.quant.dtype = args.dtype
    config.quant.init_mode = args.init_mode
    config.quant.error_mode = args.error_mode
    config.quant.latent_mode = args.latent_mode
    if args.ip_reg_gamma is not None:
        config.quant.ip_reg_gamma = args.ip_reg_gamma
    config.quant.ip_reg_inner_iters = args.ip_reg_inner_iters
    config.quant.fit_device = args.fit_device
    config.quant.freeze_u_after_init = args.freeze_u_after_init

    config.quant.enable_x_residual = args.enable_x_residual
    config.quant.x_residual_codebook = parse_codebook(args.x_residual_codebook)
    config.quant.x_residual_max_iters = args.x_residual_max_iters
    config.quant.x_residual_tol = args.x_residual_tol
    config.quant.x_residual_init_scale = args.x_residual_init_scale

    if args.device is not None:
        config.eval.device = args.device
    config.eval.stride = args.stride

    config.seed = args.seed
    config.output_dir = args.output_dir
    config.save_plots = not args.no_plots
    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = args_to_config(args)
    artifacts = run_experiment(config)
    print(json.dumps(asdict(artifacts), ensure_ascii=False, indent=2))
    print()
    print(build_summary(artifacts))


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
