from __future__ import annotations

from typing import Dict, Tuple

import torch

from .config import normalize_reorth_method


ORTHOGONALITY_DIAGNOSTIC_COLUMNS: Tuple[str, ...] = (
    "iteration",
    "orth_metrics_stage",
    "orth_err_fro",
    "orth_err_spec",
    "orth_diag_max",
    "orth_offdiag_max",
    "reorth_applied",
    "reorth_method",
    "reorth_orth_err_fro_before",
    "reorth_orth_err_fro_after",
)


def _gram_error(U: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    basis = U.detach().to(dtype=torch.float64)
    gram = basis.mT @ basis
    identity = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return gram, gram - identity


def compute_orthogonality_fro_error(U: torch.Tensor) -> float:
    _, gram_error = _gram_error(U)
    return float(torch.linalg.matrix_norm(gram_error, ord="fro").item())


def compute_orthogonality_error_stats(U: torch.Tensor) -> Dict[str, float]:
    gram, gram_error = _gram_error(U)
    diagonal_error = torch.diagonal(gram_error)
    offdiag_abs = torch.abs(gram).clone()
    if offdiag_abs.numel():
        diag_index = torch.arange(offdiag_abs.shape[0], device=offdiag_abs.device)
        offdiag_abs[diag_index, diag_index] = 0.0

    singular_values = torch.linalg.svdvals(gram_error)
    orth_err_spec = float(singular_values.max().item()) if singular_values.numel() else 0.0
    orth_diag_max = float(torch.max(torch.abs(diagonal_error)).item()) if diagonal_error.numel() else 0.0
    orth_offdiag_max = float(torch.max(offdiag_abs).item()) if offdiag_abs.numel() else 0.0
    return {
        "orth_err_fro": float(torch.linalg.matrix_norm(gram_error, ord="fro").item()),
        "orth_err_spec": orth_err_spec,
        "orth_diag_max": orth_diag_max,
        "orth_offdiag_max": orth_offdiag_max,
    }


def reorthogonalize_matrix(U: torch.Tensor, method: str = "svd") -> torch.Tensor:
    method = normalize_reorth_method(method)
    if method == "svd":
        P, _, Qt = torch.linalg.svd(U, full_matrices=False)
        return P @ Qt
    if method == "qr":
        Q, _ = torch.linalg.qr(U, mode="reduced")
        return Q
    raise ValueError(f"Unsupported reorth_method: {method}")
