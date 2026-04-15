from __future__ import annotations

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
    qmax = float((2**bits) - 1)
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
    qmin = -(2**bits)
    qmax = (2**bits) - 1
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
