from __future__ import annotations

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
    max_iters: int = 80
    tol: float = 1e-5
    convergence_check_every: int = 1
    log_every: int = 1
    codebook: Codebook = CODEBOOKS["d5"]
    dtype: str = "float32"
    eps: float = 1e-8
    init_mode: str = "random"
    error_mode: str = "relative"
    latent_mode: str = "discrete"
    ip_reg_gamma: float = 0.0
    ip_reg_inner_iters: int = 1
    fit_device: str = "cpu"


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
    output_dir: str = "./qkvo_refactor_outputs"
    seed: int = 42
    run_sq_baseline: bool = True
    save_plots: bool = True
