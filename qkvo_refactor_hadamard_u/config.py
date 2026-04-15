from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


Codebook = Tuple[float, ...]
GammaOverrideMap = Dict[str, float]
InitMode = str
ReorthMethod = str

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

INIT_MODES: Tuple[InitMode, ...] = (
    "random",
    "random_hadamard",
    "pca",
    "pca_uncentered",
    "split_pca_z_init",
    "joint_weighted_pca",
    "joint_weighted_pca_uncentered",
)

REORTH_METHODS: Tuple[ReorthMethod, ...] = ("svd", "qr")


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


def normalize_ip_reg_gamma_overrides(value: Optional[str | Mapping[str, float]]) -> GammaOverrideMap:
    if value is None:
        return {}

    payload: object
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        if text.startswith("@"):
            text = Path(text[1:]).read_text(encoding="utf-8")
        payload = json.loads(text)
    else:
        payload = dict(value)

    if not isinstance(payload, dict):
        raise ValueError("ip_reg_gamma_overrides must be a JSON object or mapping.")

    overrides: GammaOverrideMap = {}
    for raw_key, raw_gamma in payload.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError("ip_reg_gamma_overrides keys cannot be empty.")
        overrides[key] = float(raw_gamma)
    return overrides


def normalize_reorth_method(spec: str) -> ReorthMethod:
    method = spec.strip().lower()
    if method not in REORTH_METHODS:
        choices = ", ".join(REORTH_METHODS)
        raise ValueError(f"Unsupported reorth_method {spec!r}. Expected one of: {choices}")
    return method


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
    beta_pca: float = 1.0
    max_iters: int = 80
    tol: float = 1e-5
    convergence_check_every: int = 1
    log_every: int = 1
    codebook: Codebook = CODEBOOKS["d5"]
    dtype: str = "float32"
    eps: float = 1e-8
    init_mode: InitMode = "random"
    error_mode: str = "relative"
    latent_mode: str = "discrete"
    ip_reg_gamma: float = 0.0
    ip_reg_gamma_overrides: GammaOverrideMap = field(default_factory=dict)
    ip_reg_inner_iters: int = 1
    lambda_quantile_init_enable: bool = False
    lambda_quantile_rebalance_enable: bool = False
    lambda_quantile_p: float = 0.95
    lambda_quantile_rho: float = 0.8
    lambda_quantile_alpha: float = 0.0
    lambda_rebalance_ratio_min: float = 0.8
    lambda_rebalance_ratio_max: float = 1.25
    lambda_min_value: float = 1e-4
    lambda_max_value: float = 1e4
    fit_device: str = "cpu"


@dataclass
class QuantExtConfig:
    log_orth_error: bool = False
    reorth_after_u_update: bool = False
    reorth_method: ReorthMethod = REORTH_METHODS[0]

    def __post_init__(self) -> None:
        self.reorth_method = normalize_reorth_method(self.reorth_method)


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
    quant_ext: QuantExtConfig = field(default_factory=QuantExtConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    output_dir: str = "./qkvo_refactor_outputs"
    seed: int = 42
    run_sq_baseline: bool = True
    save_plots: bool = True
