from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .common import get_torch_dtype, quantize_nearest, reconstruction_objective, weighted_cross, weighted_gram
from .config import QuantExtConfig, QuantizerConfig
from .orthogonality import (
    compute_orthogonality_error_stats,
    compute_orthogonality_fro_error,
    reorthogonalize_matrix,
)


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
    tracking: Dict[str, Any] = field(default_factory=dict)

    @property
    def coeff(self) -> torch.Tensor:
        return self.lambda_x * self.lambda_w


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


def _format_optional_float(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6e}"


def _summarize_u_orthogonality_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    fro_values = [float(point["orth_err_fro"]) for point in history if point.get("orth_err_fro") is not None]
    summary: Dict[str, Any] = {
        "num_records": len(history),
        "reorth_applied_count": sum(1 for point in history if bool(point.get("reorth_applied"))),
    }
    if fro_values:
        summary.update(
            {
                "orth_err_fro_min": min(fro_values),
                "orth_err_fro_max": max(fro_values),
                "orth_err_fro_final": fro_values[-1],
            }
        )
    return summary


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
        quant_ext_config: Optional[QuantExtConfig] = None,
    ):
        self.config = config
        self.dtype = get_torch_dtype(config.dtype)
        self.codebook = torch.tensor(config.codebook, dtype=self.dtype)
        self.logger = logger
        self.observers = list(observers or [])
        self.tracking_options = tracking_options or QuantizerTrackingOptions()
        self.quant_ext_config = quant_ext_config or QuantExtConfig()

    def _orthogonality_enabled(self) -> bool:
        return bool(self.quant_ext_config.log_orth_error or self.quant_ext_config.reorth_after_u_update)

    def _orthogonality_config_payload(self) -> Dict[str, Any]:
        return {
            "log_orth_error": bool(self.quant_ext_config.log_orth_error),
            "reorth_after_u_update": bool(self.quant_ext_config.reorth_after_u_update),
            "reorth_method": str(self.quant_ext_config.reorth_method),
        }

    def _process_u_update_orthogonality(
        self,
        U_new: torch.Tensor,
        *,
        tag: str,
        iteration: int,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        if not self._orthogonality_enabled():
            return U_new, None

        record: Dict[str, Any] = {
            "iteration": int(iteration),
            "orth_metrics_stage": "post_update",
            "orth_err_fro": None,
            "orth_err_spec": None,
            "orth_diag_max": None,
            "orth_offdiag_max": None,
            "reorth_applied": bool(self.quant_ext_config.reorth_after_u_update),
            "reorth_method": self.quant_ext_config.reorth_method if self.quant_ext_config.reorth_after_u_update else None,
            "reorth_orth_err_fro_before": None,
            "reorth_orth_err_fro_after": None,
        }

        if self.quant_ext_config.reorth_after_u_update:
            record["reorth_orth_err_fro_before"] = compute_orthogonality_fro_error(U_new)
            U_new = reorthogonalize_matrix(U_new, method=self.quant_ext_config.reorth_method)
            record["reorth_orth_err_fro_after"] = compute_orthogonality_fro_error(U_new)
            record["orth_metrics_stage"] = "post_reorth"

        if self.quant_ext_config.log_orth_error:
            record.update(compute_orthogonality_error_stats(U_new))
        elif record["reorth_orth_err_fro_after"] is not None:
            record["orth_err_fro"] = float(record["reorth_orth_err_fro_after"])

        if self.logger is not None:
            self.logger.info(
                "tag=%s iter=%03d | U orth stage=%s fro=%s spec=%s diag_max=%s offdiag_max=%s reorth=%s method=%s fro_before=%s fro_after=%s",
                tag,
                iteration,
                record["orth_metrics_stage"],
                _format_optional_float(record["orth_err_fro"]),
                _format_optional_float(record["orth_err_spec"]),
                _format_optional_float(record["orth_diag_max"]),
                _format_optional_float(record["orth_offdiag_max"]),
                record["reorth_applied"],
                record["reorth_method"] or "off",
                _format_optional_float(record["reorth_orth_err_fro_before"]),
                _format_optional_float(record["reorth_orth_err_fro_after"]),
            )

        return U_new, record

    def _resolve_ip_reg_gamma(self, tag: str) -> float:
        overrides = getattr(self.config, "ip_reg_gamma_overrides", {}) or {}
        if not tag:
            return float(self.config.ip_reg_gamma)

        if tag in overrides:
            return float(overrides[tag])

        block_tag, _, linear_name = tag.partition(".")
        if block_tag and block_tag in overrides:
            return float(overrides[block_tag])
        if linear_name and linear_name in overrides:
            return float(overrides[linear_name])
        return float(self.config.ip_reg_gamma)

    def _scatter_matrix(
        self,
        data: torch.Tensor,
        *,
        centered: bool,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if centered:
            if weights is None:
                mean = data.mean(dim=1, keepdim=True)
            else:
                safe_weight_sum = torch.clamp(weights.sum(), min=self.config.eps)
                mean = (data * weights.unsqueeze(0)).sum(dim=1, keepdim=True) / safe_weight_sum
            basis = data - mean
        else:
            basis = data
        if weights is None:
            scatter = basis @ basis.T
        else:
            scatter = (basis * weights.unsqueeze(0)) @ basis.T
        return 0.5 * (scatter + scatter.T)

    def _eigenvector_init(self, scatter: torch.Tensor) -> torch.Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(scatter)
        order = torch.argsort(eigenvalues, descending=True)
        return eigenvectors[:, order]

    def _pca_init(self, X: torch.Tensor, *, centered: bool) -> torch.Tensor:
        return self._eigenvector_init(self._scatter_matrix(X, centered=centered))

    def _joint_weighted_pca_init(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        weights_x: torch.Tensor,
        weights_w: torch.Tensor,
        *,
        centered: bool,
    ) -> torch.Tensor:
        scatter_x = self._scatter_matrix(X, centered=centered, weights=weights_x)
        scatter_w = self._scatter_matrix(W, centered=centered, weights=weights_w)
        scatter = scatter_x + float(self.config.beta_pca) * scatter_w
        return self._eigenvector_init(scatter)

    def _random_init(self, X: torch.Tensor) -> torch.Tensor:
        random_matrix = torch.randn((X.shape[0], X.shape[0]), dtype=X.dtype, device=X.device)
        Q, _ = torch.linalg.qr(random_matrix)
        return Q

    def _hadamard_block(self, size: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        H = torch.ones((1, 1), dtype=dtype, device=device)
        current_size = 1
        while current_size < size:
            top = torch.cat((H, H), dim=1)
            bottom = torch.cat((H, -H), dim=1)
            H = torch.cat((top, bottom), dim=0)
            current_size *= 2
        return H / math.sqrt(size)

    def _random_signs(self, size: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.where(
            torch.rand(size, device=device) < 0.5,
            -torch.ones(size, dtype=dtype, device=device),
            torch.ones(size, dtype=dtype, device=device),
        )

    def _randomized_hadamard_block(self, size: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        block = self._hadamard_block(size, dtype=dtype, device=device)
        row_perm = torch.randperm(size, device=device)
        col_perm = torch.randperm(size, device=device)
        row_signs = self._random_signs(size, dtype=dtype, device=device)
        col_signs = self._random_signs(size, dtype=dtype, device=device)
        block = block[row_perm][:, col_perm]
        return row_signs.unsqueeze(1) * block * col_signs.unsqueeze(0)

    def _random_hadamard_init(self, X: torch.Tensor) -> torch.Tensor:
        dim = X.shape[0]
        remaining = dim
        block_sizes: List[int] = []
        next_block = 1 << (max(dim, 1).bit_length() - 1)
        while remaining > 0 and next_block > 0:
            if remaining >= next_block:
                block_sizes.append(next_block)
                remaining -= next_block
            next_block >>= 1

        blocks = [self._randomized_hadamard_block(size, dtype=X.dtype, device=X.device) for size in block_sizes]
        U = torch.block_diag(*blocks) if len(blocks) > 1 else blocks[0]

        row_perm = torch.randperm(dim, device=X.device)
        col_perm = torch.randperm(dim, device=X.device)
        row_signs = self._random_signs(dim, dtype=X.dtype, device=X.device)
        col_signs = self._random_signs(dim, dtype=X.dtype, device=X.device)
        U = U[row_perm][:, col_perm]
        return row_signs.unsqueeze(1) * U * col_signs.unsqueeze(0)

    def _init_bases(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        weights_x: torch.Tensor,
        weights_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.init_mode == "random":
            U = self._random_init(X)
            return U, U, U
        if self.config.init_mode == "random_hadamard":
            U = self._random_hadamard_init(X)
            return U, U, U
        if self.config.init_mode == "pca":
            U = self._pca_init(X, centered=True)
            return U, U, U
        if self.config.init_mode == "pca_uncentered":
            U = self._pca_init(X, centered=False)
            return U, U, U
        if self.config.init_mode == "split_pca_z_init":
            U_x = self._pca_init(X, centered=True)
            U_w = self._pca_init(W, centered=True)
            return U_x, U_x, U_w
        if self.config.init_mode == "joint_weighted_pca":
            U = self._joint_weighted_pca_init(X, W, weights_x, weights_w, centered=True)
            return U, U, U
        if self.config.init_mode == "joint_weighted_pca_uncentered":
            U = self._joint_weighted_pca_init(X, W, weights_x, weights_w, centered=False)
            return U, U, U
        raise ValueError(f"Unsupported init_mode: {self.config.init_mode}")

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
        gamma: float,
        weights_x: torch.Tensor,
        weights_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lambda_x, SXZx, SZZx = self._update_lambda(X, U, Z_x, weights_x)
        lambda_w, SWZw, SZZw = self._update_lambda(W, U, Z_w, weights_w)

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
        gamma: float,
    ) -> torch.Tensor:
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

    def _lambda_quantile_target(self) -> float:
        positive_max = float(torch.max(self.codebook).item())
        if positive_max > self.config.eps:
            return positive_max
        abs_max = float(torch.max(torch.abs(self.codebook)).item())
        return max(abs_max, float(self.config.eps))

    def _init_lambda_from_quantiles(
        self,
        proj: torch.Tensor,
        target_value: float,
    ) -> torch.Tensor:
        p = float(self.config.lambda_quantile_p)
        rho = float(self.config.lambda_quantile_rho)
        eps = float(self.config.eps)
        lambda_min = float(self.config.lambda_min_value)
        lambda_max = float(self.config.lambda_max_value)

        q = torch.quantile(proj.abs(), p, dim=1)
        target = max(rho * target_value, eps)
        lam = q / target
        lam = torch.clamp(lam, min=lambda_min, max=lambda_max)
        return lam

    def _rebalance_lambda_from_quantiles(
        self,
        proj: torch.Tensor,
        lambda_diag: torch.Tensor,
        target_value: float,
    ) -> torch.Tensor:
        p = float(self.config.lambda_quantile_p)
        rho = float(self.config.lambda_quantile_rho)
        alpha = float(self.config.lambda_quantile_alpha)
        eps = float(self.config.eps)
        ratio_min = float(self.config.lambda_rebalance_ratio_min)
        ratio_max = float(self.config.lambda_rebalance_ratio_max)
        lambda_min = float(self.config.lambda_min_value)
        lambda_max = float(self.config.lambda_max_value)

        safe_lambda = torch.clamp(lambda_diag, min=eps)
        z_tilde = proj / safe_lambda.unsqueeze(1)
        q = torch.quantile(z_tilde.abs(), p, dim=1)

        target = max(rho * target_value, eps)
        ratio = (q / target).clamp(min=eps).pow(alpha)
        ratio = ratio.clamp(min=ratio_min, max=ratio_max)

        new_lambda = lambda_diag * ratio
        new_lambda = torch.clamp(new_lambda, min=lambda_min, max=lambda_max)
        return new_lambda

    def fit(self, X: torch.Tensor, W: torch.Tensor, tag: str = "") -> QuantizationState:
        fit_start = time.perf_counter()
        X = X.to(dtype=self.dtype)
        W = W.to(dtype=self.dtype)
        self.codebook = self.codebook.to(X.device)
        gamma = self._resolve_ip_reg_gamma(tag)

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
                "Fit quantizer | tag=%s d=%d tokens=%d out_features=%d init_mode=%s beta=%.4f beta_pca=%.4f ip_reg_gamma=%.4f max_iters=%d tol=%.2e",
                tag,
                feature_dim,
                num_tokens,
                out_features,
                self.config.init_mode,
                self.config.beta,
                self.config.beta_pca,
                gamma,
                self.config.max_iters,
                self.config.tol,
            )
            self.logger.info(
                "Lambda quantile init=%s rebalance=%s p=%.3f rho=%.3f alpha=%.3f",
                self.config.lambda_quantile_init_enable,
                self.config.lambda_quantile_rebalance_enable,
                self.config.lambda_quantile_p,
                self.config.lambda_quantile_rho,
                self.config.lambda_quantile_alpha,
            )
            self.logger.info(
                "Orth diagnostics log_orth_error=%s reorth_after_u_update=%s reorth_method=%s",
                self.quant_ext_config.log_orth_error,
                self.quant_ext_config.reorth_after_u_update,
                self.quant_ext_config.reorth_method,
            )

        U, U_zx_init, U_zw_init = self._init_bases(X, W, weights_x, weights_w)

        U_init = U.detach().clone()
        U_prev = U.detach().clone()
        latent_target = self._lambda_quantile_target()
        if self.config.lambda_quantile_init_enable:
            proj_x = U_zx_init.T @ X
            proj_w = U_zw_init.T @ W
            lambda_x = self._init_lambda_from_quantiles(proj_x, latent_target)
            lambda_w = self._init_lambda_from_quantiles(proj_w, latent_target)
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
        orthogonality_history: List[Dict[str, Any]] = []
        previous_objective = float("inf")

        for observer in self.observers:
            observer.on_fit_start(tag=tag, U_init=U_init, options=self.tracking_options)

        for iteration in range(1, self.config.max_iters + 1):
            iter_start = time.perf_counter()

            U_for_zx = U_zx_init if iteration == 1 else U
            U_for_zw = U_zw_init if iteration == 1 else U
            Z_x = self._e_step(X, U_for_zx, lambda_x)
            Z_w = self._e_step(W, U_for_zw, lambda_w)

            lambda_x, lambda_w, SXZx, SWZw = self._update_lambdas(
                X=X,
                W=W,
                U=U,
                Z_x=Z_x,
                Z_w=Z_w,
                gamma=gamma,
                weights_x=weights_x,
                weights_w=weights_w,
            )

            if self.config.lambda_quantile_rebalance_enable:
                proj_x = U_for_zx.T @ X
                proj_w = U_for_zw.T @ W
                lambda_x = self._rebalance_lambda_from_quantiles(proj_x, lambda_x, latent_target)
                lambda_w = self._rebalance_lambda_from_quantiles(proj_w, lambda_w, latent_target)

                Z_x = self._e_step(X, U_for_zx, lambda_x)
                Z_w = self._e_step(W, U_for_zw, lambda_w)
                SXZx = weighted_cross(X, weights_x, Z_x)
                SWZw = weighted_cross(W, weights_w, Z_w)

            U_new = self._update_U(lambda_x, lambda_w, SXZx, SWZw)
            U_new, orthogonality_record = self._process_u_update_orthogonality(
                U_new,
                tag=tag,
                iteration=iteration,
            )
            if orthogonality_record is not None:
                orthogonality_history.append(orthogonality_record)

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
            objective_ip = self._compute_ip_regularizer(X, W, Z_x, Z_w, lambda_x, lambda_w, gamma)
            objective = objective_x + self.config.beta * objective_w + objective_ip

            objective_history.append(float(objective.item()))
            objective_x_history.append(float(objective_x.item()))
            objective_w_history.append(float(objective_w.item()))

            if math.isfinite(previous_objective):
                relative_change = abs(float(objective.item()) - previous_objective) / max(1.0, abs(previous_objective))
            else:
                relative_change = float("inf")

            if self.logger is not None and (iteration == 1 or iteration % self.config.log_every == 0):
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

        fit_time = time.perf_counter() - fit_start
        tracking: Dict[str, Any] = {}
        for observer in self.observers:
            tracking.update(observer.build_state_fields())
        if self._orthogonality_enabled():
            tracking["u_orthogonality"] = {
                "config": self._orthogonality_config_payload(),
                "history": orthogonality_history,
                "summary": _summarize_u_orthogonality_history(orthogonality_history),
                "final": orthogonality_history[-1] if orthogonality_history else None,
            }

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
            tracking=tracking,
        )

    def reconstruct_X(self, X: torch.Tensor, state: QuantizationState) -> torch.Tensor:
        latent = self._e_step(X.to(device=state.U.device, dtype=state.U.dtype), state.U, state.lambda_x)
        return state.U @ (state.lambda_x.unsqueeze(1) * latent)

    def reconstruct_W(self, state: QuantizationState) -> torch.Tensor:
        return state.U @ (state.lambda_w.unsqueeze(1) * state.Z_w)
