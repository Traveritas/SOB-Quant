import math
from typing import Optional, Tuple

import torch


def quantize_nearest(z_continuous: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(z_continuous.unsqueeze(-1) - codebook)
    idx = torch.argmin(diff, dim=-1)
    return codebook[idx]


def uniform_quantize_maxabs(
    x: torch.Tensor,
    bits: int,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    qmin = -(2 ** bits)
    qmax = (2 ** bits) - 1
    maxabs = torch.max(torch.abs(x))
    scale = torch.clamp(maxabs / float(qmax), min=eps)
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    x_q = q * scale
    return x_q, scale


def uniform_quantize_maxabs_codes(
    x: torch.Tensor,
    bits: int,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    qmin = -(2 ** bits)
    qmax = (2 ** bits) - 1
    maxabs = torch.max(torch.abs(x))
    scale = torch.clamp(maxabs / float(qmax), min=eps)
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    return q, scale


def infer_sq_bitwidth_from_codebook(codebook: Tuple[float, ...]) -> int:
    return int(math.ceil(math.log2(max(len(codebook), 2))))


def scalar_quant_scale_maxabs(x: torch.Tensor, bits: int, eps: float = 1e-8) -> torch.Tensor:
    qmax = float((2 ** bits) - 1)
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
    qmin = -(2 ** bits)
    qmax = (2 ** bits) - 1
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    x_q = q * scale
    return x_q, scale
