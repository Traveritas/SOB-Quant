from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from qkvo_refactor_hadamard_u.cli import add_experiment_args
from qkvo_refactor_hadamard_u.common import (
    get_torch_dtype,
    infer_sq_bitwidth_from_codebook,
    scalar_quant_scale_maxabs,
    scalar_quantize_maxabs,
    set_seed,
    setup_logger,
    tensor_stats,
    write_json,
    write_text,
)
from qkvo_refactor_hadamard_u.config import DataConfig, EvalConfig, TargetConfig, parse_block_indices, parse_codebook, parse_target_linear_names
from qkvo_refactor_hadamard_u.experiment import compute_sq_metrics, evaluate_perplexity_sliding_window, save_loss_plots
from qkvo_refactor_hadamard_u.model_utils import (
    ScalarQuantizedXWLinear,
    TargetModuleSpec,
    build_runtime_bias,
    build_weight_matrix,
    collect_target_inputs,
    get_attention_target_specs,
    load_model_and_tokenizer,
    load_text_split,
    replace_target_modules,
    restore_target_modules,
    tokenize_text,
)
from qkvo_refactor_hadamard_u.quantizer import LatticeLinearQuantizer, QuantizationState


def default_target_config() -> TargetConfig:
    return TargetConfig(
        block_indices=(8, 9, 10, 11),
        target_linear_names=("q_proj", "k_proj", "v_proj", "out_proj"),
    )


@dataclass
class QuantizerConfig:
    beta: float = 1.0
    max_iters: int = 300
    tol: float = 1e-5
    convergence_check_every: int = 1
    log_every: int = 1
    codebook: Tuple[float, ...] = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    dtype: str = "float32"
    eps: float = 1e-8
    init_mode: str = "random"
    error_mode: str = "relative"
    latent_mode: str = "discrete"
    latent_bits: int = 5
    ip_reg_gamma: float = 0.0
    ip_reg_inner_iters: int = 1
    fit_device: str = "cpu"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    quant: QuantizerConfig = field(default_factory=QuantizerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    target: TargetConfig = field(default_factory=default_target_config)
    output_dir: str = "./.new_quant_function"
    seed: int = 42
    run_mode: str = "single"
    continuous_subdir: str = "continuous"
    run_sq_baseline: bool = True
    save_plots: bool = True


@dataclass
class ExperimentArtifacts:
    config: Dict
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
    tensor_info: Dict[str, Dict[str, object]]
    latent_distribution_info: Dict[str, Dict[str, object]]
    timing_info: Dict[str, float]
    ppl_eval_info: Dict[str, Dict[str, float]]
    plot_paths: Dict[str, Dict[str, str]]
    target_info: Dict[str, object]


def uniform_quantize_maxabs(
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


def uniform_quantize_maxabs_codes(
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
    return q, scale


def average_metrics(metrics_by_layer: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if not metrics_by_layer:
        return {}
    keys = sorted(next(iter(metrics_by_layer.values())).keys())
    return {
        key: float(sum(layer_metrics[key] for layer_metrics in metrics_by_layer.values()) / len(metrics_by_layer))
        for key in keys
    }


def distribution_stats(tensor: torch.Tensor) -> Dict[str, object]:
    values = tensor.detach().float().reshape(-1).cpu()
    abs_values = torch.abs(values)
    quantiles = torch.tensor([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999], dtype=torch.float32)
    value_quantiles = torch.quantile(values, quantiles)
    abs_quantiles = torch.quantile(abs_values, quantiles)
    return {
        "numel": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean_abs": float(abs_values.mean().item()),
        "max_abs": float(abs_values.max().item()),
        "quantiles": {f"{float(q):.3f}": float(v.item()) for q, v in zip(quantiles, value_quantiles)},
        "abs_quantiles": {f"{float(q):.3f}": float(v.item()) for q, v in zip(quantiles, abs_quantiles)},
    }


def discrete_value_counts(tensor: torch.Tensor) -> Dict[str, int]:
    values = tensor.detach().float().reshape(-1).cpu()
    unique, counts = torch.unique(values, sorted=True, return_counts=True)
    return {f"{float(value.item()):.6g}": int(count.item()) for value, count in zip(unique, counts)}


def layer_prefix(layer_name: str) -> str:
    return layer_name.replace(".", "_")


def save_distribution_histogram(output_dir: Path, prefix: str, tensor: torch.Tensor, title: str, bins: int = 80) -> str:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    values = tensor.detach().float().reshape(-1).cpu().numpy()
    path = output_dir / f"{prefix}_hist.png"
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, alpha=0.85)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def save_discrete_histogram(output_dir: Path, prefix: str, tensor: torch.Tensor, title: str) -> str:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    values = tensor.detach().float().reshape(-1).cpu()
    unique, counts = torch.unique(values, sorted=True, return_counts=True)
    path = output_dir / f"{prefix}_discrete_hist.png"
    plt.figure(figsize=(10, 5))
    plt.bar([str(float(value.item())) for value in unique], counts.tolist())
    plt.title(title)
    plt.xlabel("discrete value")
    plt.ylabel("count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def save_quantized_overlay_histogram(
    output_dir: Path,
    prefix: str,
    z_continuous: torch.Tensor,
    z_codes: torch.Tensor,
    title: str,
    bins: int = 80,
) -> str:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{prefix}_quantized_overlay.png"
    z_continuous_flat = z_continuous.detach().float().reshape(-1).cpu()
    z_codes_flat = z_codes.detach().float().reshape(-1).cpu()
    code_values = torch.unique(z_codes_flat, sorted=True)

    plt.figure(figsize=(10, 6))
    plt.hist(z_continuous_flat.numpy(), bins=bins, alpha=0.25, color="gray", label="all z_tilde")
    cmap = plt.cm.get_cmap("tab10", max(int(code_values.numel()), 1))
    for index, code_value in enumerate(code_values.tolist()):
        mask = torch.isclose(z_codes_flat, torch.tensor(code_value, dtype=z_codes_flat.dtype))
        if not torch.any(mask):
            continue
        plt.hist(
            z_continuous_flat[mask].numpy(),
            bins=bins,
            alpha=0.55,
            color=cmap(index),
            label=f"code -> {code_value:g}",
        )

    plt.title(title)
    plt.xlabel("continuous z_tilde value")
    plt.ylabel("count")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def collect_latent_distribution_info(
    layer_name: str,
    X_calib: torch.Tensor,
    state: QuantizationState,
    output_dir: Path,
    eps: float,
) -> Dict[str, object]:
    prefix = layer_prefix(layer_name)
    plots_dir = output_dir / "latent_distributions"
    latent_bits = int(getattr(state, "latent_bits", 2))

    U = state.U.to(dtype=X_calib.dtype, device=X_calib.device)
    lambda_x = state.lambda_x.to(dtype=X_calib.dtype, device=X_calib.device)
    u_tx = U.T @ X_calib
    safe_lambda_x = torch.where(lambda_x.abs() < eps, torch.full_like(lambda_x, eps), lambda_x)
    z_tilde_cont = u_tx / safe_lambda_x.unsqueeze(1)
    lambda_u_tx = lambda_x.unsqueeze(1) * u_tx
    z_tilde_quantized, z_tilde_scale = uniform_quantize_maxabs(z_tilde_cont, bits=latent_bits, eps=eps)
    z_tilde_codes, _ = uniform_quantize_maxabs_codes(z_tilde_cont, bits=latent_bits, eps=eps)

    plots = {
        "u_tx_hist": save_distribution_histogram(plots_dir, f"{prefix}_u_tx", u_tx, f"{layer_name} U^T X"),
        "z_tilde_cont_hist": save_distribution_histogram(plots_dir, f"{prefix}_z_tilde_cont", z_tilde_cont, f"{layer_name} z_tilde"),
        "z_tilde_codes_hist": save_discrete_histogram(plots_dir, f"{prefix}_z_tilde_codes", z_tilde_codes, f"{layer_name} quantized codes"),
        "lambda_u_tx_hist": save_distribution_histogram(
            plots_dir,
            f"{prefix}_lambda_u_tx",
            lambda_u_tx,
            f"{layer_name} lambda_x * (U^T X)",
        ),
        "z_tilde_quantized_overlay": save_quantized_overlay_histogram(
            plots_dir,
            f"{prefix}_z_tilde",
            z_tilde_cont,
            z_tilde_codes,
            f"{layer_name} z_tilde with quantized code assignments",
        ),
    }

    return {
        "u_tx": distribution_stats(u_tx),
        "z_tilde_cont": distribution_stats(z_tilde_cont),
        "z_tilde_codes": distribution_stats(z_tilde_codes),
        "z_tilde_code_value_counts": discrete_value_counts(z_tilde_codes),
        "z_tilde_quantized": distribution_stats(z_tilde_quantized),
        "z_tilde_quantized_value_counts": discrete_value_counts(z_tilde_quantized),
        "z_tilde_quant_scale": float(z_tilde_scale.item()),
        "latent_bits": latent_bits,
        "lambda_u_tx": distribution_stats(lambda_u_tx),
        "plots": plots,
    }


class UniformLatentQuantizer(LatticeLinearQuantizer):
    def _e_step(self, data: torch.Tensor, U: torch.Tensor, lambda_diag: torch.Tensor) -> torch.Tensor:
        latent = self._latent_step(data, U, lambda_diag)
        if self.config.latent_mode == "continuous":
            return latent
        quantized, _ = uniform_quantize_maxabs(
            latent,
            bits=int(getattr(self.config, "latent_bits", 2)),
            eps=self.config.eps,
        )
        return quantized

    def fit(self, X: torch.Tensor, W: torch.Tensor, tag: str = "") -> QuantizationState:
        state = super().fit(X, W, tag=tag)
        state.latent_bits = int(getattr(self.config, "latent_bits", 2))
        return state


class UniformLatentQuantizedLinear(nn.Module):
    def __init__(self, state: QuantizationState, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("U", state.U.detach().clone())
        self.register_buffer("lambda_x", state.lambda_x.detach().clone())
        self.register_buffer("lambda_w", state.lambda_w.detach().clone())
        self.register_buffer("coeff", state.coeff.detach().clone())
        self.register_buffer("Z_w", state.Z_w.detach().clone())
        self.latent_mode = state.latent_mode
        self.latent_bits = int(getattr(state, "latent_bits", 2))
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().clone())

    def _encode_x(self, x: torch.Tensor) -> torch.Tensor:
        projected = x.to(self.U.dtype) @ self.U
        safe_lambda_x = torch.where(
            self.lambda_x.abs() < 1e-8,
            torch.full_like(self.lambda_x, 1e-8),
            self.lambda_x,
        )
        z_continuous = projected / safe_lambda_x.unsqueeze(0)
        if self.latent_mode == "continuous":
            return z_continuous
        z_quantized, _ = uniform_quantize_maxabs(z_continuous, bits=self.latent_bits, eps=1e-8)
        return z_quantized

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape[:-1]
        input_dim = hidden_states.shape[-1]
        x_flat = hidden_states.reshape(-1, input_dim)
        z_x = self._encode_x(x_flat)
        output = (z_x * self.coeff.unsqueeze(0)) @ self.Z_w
        if self.bias is not None:
            output = output + self.bias.to(output.dtype).unsqueeze(0)
        return output.to(hidden_states.dtype).reshape(*original_shape, self.Z_w.shape[1])


@torch.no_grad()
def compute_linear_relative_error(
    X: torch.Tensor,
    W: torch.Tensor,
    state: QuantizationState,
    chunk_tokens: int = 128,
) -> float:
    X = X.to(device=state.U.device, dtype=state.U.dtype)
    W = W.to(device=state.U.device, dtype=state.U.dtype)
    coeff = state.coeff
    latent_bits = int(getattr(state, "latent_bits", 2))

    denominator = 0.0
    numerator = 0.0
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
            Z_x, _ = uniform_quantize_maxabs(Z_x, bits=latent_bits, eps=1e-8)

        Y_hat = (Z_x.T * coeff.unsqueeze(0)) @ state.Z_w
        delta = Y_true - Y_hat
        numerator += float(torch.sum(delta * delta).item())
        denominator += float(torch.sum(Y_true * Y_true).item())

    return numerator / max(denominator, 1e-12)


def compute_quant_metrics(
    X: torch.Tensor,
    W: torch.Tensor,
    state: QuantizationState,
    quantizer: UniformLatentQuantizer,
    error_mode: str,
) -> Dict[str, float]:
    X = X.to(device=state.U.device, dtype=state.U.dtype)
    W = W.to(device=state.U.device, dtype=state.U.dtype)

    X_hat = quantizer.reconstruct_X(X, state)
    W_hat = quantizer.reconstruct_W(state)

    if error_mode == "relative":
        x_error = float(torch.sum((X - X_hat) ** 2).item() / max(torch.sum(X**2).item(), 1e-12))
        w_error = float(torch.sum((W - W_hat) ** 2).item() / max(torch.sum(W**2).item(), 1e-12))
    elif error_mode == "absolute":
        x_error = float(torch.mean((X - X_hat) ** 2).item())
        w_error = float(torch.mean((W - W_hat) ** 2).item())
    else:
        raise ValueError(f"Unsupported error_mode: {error_mode}")

    return {
        "x_error": x_error,
        "w_error": w_error,
        "linear_error": float(compute_linear_relative_error(X, W, state)),
    }


def build_quantized_target_modules(
    target_specs: Dict[str, TargetModuleSpec],
    X_calib_by_layer: Dict[str, torch.Tensor],
    config: ExperimentConfig,
    logger,
) -> Tuple[
    Dict[str, nn.Module],
    Dict[str, QuantizationState],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, object]],
    Dict[str, Dict[str, str]],
    Dict[str, Dict[str, object]],
]:
    quantized_modules: Dict[str, nn.Module] = {}
    states: Dict[str, QuantizationState] = {}
    metrics_by_layer: Dict[str, Dict[str, float]] = {}
    tensor_info: Dict[str, Dict[str, object]] = {}
    plot_paths: Dict[str, Dict[str, str]] = {}
    latent_distribution_info: Dict[str, Dict[str, object]] = {}

    plots_dir = Path(config.output_dir) / "plots"
    quantizer = UniformLatentQuantizer(config.quant, logger=logger)

    for layer_name, spec in target_specs.items():
        X_calib = X_calib_by_layer[layer_name].to(device=config.quant.fit_device, dtype=quantizer.dtype)
        W = build_weight_matrix(spec.module, fit_device=config.quant.fit_device, dtype_name=config.quant.dtype)
        bias = build_runtime_bias(spec.module, runtime_device=config.eval.device, dtype_name=config.quant.dtype)

        state = quantizer.fit(X_calib, W, tag=layer_name)
        metrics_by_layer[layer_name] = compute_quant_metrics(X_calib, W, state, quantizer, config.quant.error_mode)
        quantized_modules[layer_name] = UniformLatentQuantizedLinear(state, bias=bias).to(config.eval.device)
        states[layer_name] = state

        tensor_info[f"{layer_name}.X_calib"] = tensor_stats(X_calib)
        tensor_info[f"{layer_name}.W"] = tensor_stats(W)
        tensor_info[f"{layer_name}.U"] = tensor_stats(state.U)
        tensor_info[f"{layer_name}.lambda_x"] = tensor_stats(state.lambda_x)
        tensor_info[f"{layer_name}.lambda_w"] = tensor_stats(state.lambda_w)
        tensor_info[f"{layer_name}.Z_w"] = tensor_stats(state.Z_w)
        if bias is not None:
            tensor_info[f"{layer_name}.bias"] = tensor_stats(bias)

        latent_distribution_info[layer_name] = collect_latent_distribution_info(
            layer_name=layer_name,
            X_calib=X_calib,
            state=state,
            output_dir=Path(config.output_dir),
            eps=config.quant.eps,
        )

        if config.save_plots:
            plot_paths[layer_name] = save_loss_plots(
                output_dir=plots_dir,
                layer_name=layer_name,
                objective_history=state.objective_history,
                objective_x_history=state.objective_x_history,
                objective_w_history=state.objective_w_history,
            )
        else:
            plot_paths[layer_name] = {}

    return quantized_modules, states, metrics_by_layer, tensor_info, plot_paths, latent_distribution_info


def build_sq_target_modules(
    target_specs: Dict[str, TargetModuleSpec],
    X_calib_by_layer: Dict[str, torch.Tensor],
    config: ExperimentConfig,
) -> Tuple[
    Dict[str, nn.Module],
    int,
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, object]],
]:
    sq_modules: Dict[str, nn.Module] = {}
    sq_metrics_by_layer: Dict[str, Dict[str, float]] = {}
    sq_tensor_info: Dict[str, Dict[str, object]] = {}

    bits = infer_sq_bitwidth_from_codebook(config.quant.codebook)
    for layer_name, spec in target_specs.items():
        X_calib = X_calib_by_layer[layer_name].to(device=config.quant.fit_device)
        W = build_weight_matrix(spec.module, fit_device=config.quant.fit_device, dtype_name=config.quant.dtype)
        bias = build_runtime_bias(spec.module, runtime_device=config.eval.device, dtype_name=config.quant.dtype)

        x_scale = scalar_quant_scale_maxabs(X_calib, bits=bits, eps=config.quant.eps)
        W_q, w_scale = scalar_quantize_maxabs(W, bits=bits, eps=config.quant.eps)
        sq_metrics_by_layer[layer_name] = compute_sq_metrics(X_calib, W, x_scale, w_scale, bits)
        sq_modules[layer_name] = ScalarQuantizedXWLinear(
            weight_quantized=W_q,
            x_scale=x_scale,
            bits=bits,
            bias=bias,
            eps=config.quant.eps,
        ).to(config.eval.device)

        sq_tensor_info[f"{layer_name}.sq_x_scale"] = tensor_stats(x_scale)
        sq_tensor_info[f"{layer_name}.sq_w_scale"] = tensor_stats(w_scale)
        sq_tensor_info[f"{layer_name}.sq_weight_quantized"] = tensor_stats(W_q)
        if bias is not None:
            sq_tensor_info[f"{layer_name}.bias"] = tensor_stats(bias)

    return sq_modules, bits, sq_metrics_by_layer, sq_tensor_info


def build_analysis_summary(artifacts: ExperimentArtifacts) -> str:
    lines = [
        "New Quant Function Summary",
        f"- baseline_ppl: {artifacts.baseline_ppl:.6f}",
        f"- quantized_ppl: {artifacts.quantized_ppl:.6f}",
        f"- quantized_delta: {artifacts.quantized_ppl - artifacts.baseline_ppl:.6f}",
        f"- latent_mode: {artifacts.config['quant']['latent_mode']}",
        f"- latent_bits: {artifacts.config['quant']['latent_bits']}",
    ]
    if not math.isnan(artifacts.sq_baseline_ppl):
        lines.append(f"- sq_baseline_ppl: {artifacts.sq_baseline_ppl:.6f}")
        lines.append(f"- sq_delta: {artifacts.sq_baseline_ppl - artifacts.baseline_ppl:.6f}")

    lines.append("")
    lines.append("Average Quant Metrics")
    for key, value in artifacts.quant_metrics_avg.items():
        lines.append(f"- {key}: {value:.6f}")

    if artifacts.sq_metrics_avg:
        lines.append("")
        lines.append("Average SQ Metrics")
        for key, value in artifacts.sq_metrics_avg.items():
            lines.append(f"- {key}: {value:.6f}")

    lines.append("")
    lines.append("Target Info")
    for key, value in artifacts.target_info.items():
        lines.append(f"- {key}: {value}")

    return "\n".join(lines) + "\n"


def run_all_blocks_qkvo_experiment(config: ExperimentConfig) -> ExperimentArtifacts:
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

    sq_baseline_ppl = float("nan")
    sq_metrics_by_layer: Dict[str, Dict[str, float]] = {}
    sq_tensor_info: Dict[str, Dict[str, object]] = {}
    sq_bits: Optional[int] = None

    if config.run_sq_baseline:
        start = time.perf_counter()
        sq_modules, sq_bits, sq_metrics_by_layer, sq_tensor_info = build_sq_target_modules(
            target_specs=target_specs,
            X_calib_by_layer=X_calib_by_layer,
            config=config,
        )
        timing_info["build_sq_baseline_sec"] = time.perf_counter() - start

        original_modules = replace_target_modules(target_specs, sq_modules)
        try:
            sq_baseline_ppl, sq_eval_stats = evaluate_perplexity_sliding_window(
                model=model,
                tokenizer=tokenizer,
                text=eval_text,
                device=config.eval.device,
                stride=config.eval.stride,
                max_eval_tokens=config.data.eval_num_tokens,
                logger=logger,
                tag="sq_baseline",
            )
            ppl_eval_info["sq_baseline"] = sq_eval_stats
        finally:
            restore_target_modules(target_specs, original_modules)

    start = time.perf_counter()
    quantized_modules, states, quant_metrics_by_layer, tensor_info, plot_paths, latent_distribution_info = build_quantized_target_modules(
        target_specs=target_specs,
        X_calib_by_layer=X_calib_by_layer,
        config=config,
        logger=logger,
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

    merged_tensor_info = dict(tensor_info)
    merged_tensor_info.update(sq_tensor_info)
    if sq_bits is not None:
        merged_tensor_info["sq_bitwidth"] = {"value": sq_bits}

    artifacts = ExperimentArtifacts(
        config=asdict(config),
        quantization_completed=quantization_completed,
        baseline_ppl=baseline_ppl,
        sq_baseline_ppl=sq_baseline_ppl,
        quantized_ppl=quantized_ppl,
        sq_metrics=sq_metrics_by_layer,
        sq_metrics_avg=average_metrics(sq_metrics_by_layer),
        quant_metrics=quant_metrics_by_layer,
        quant_metrics_avg=average_metrics(quant_metrics_by_layer),
        convergence_iters={layer_name: state.convergence_iter for layer_name, state in states.items()},
        objective_histories={layer_name: state.objective_history for layer_name, state in states.items()},
        objective_x_histories={layer_name: state.objective_x_history for layer_name, state in states.items()},
        objective_w_histories={layer_name: state.objective_w_history for layer_name, state in states.items()},
        tensor_info=merged_tensor_info,
        latent_distribution_info=latent_distribution_info,
        timing_info=timing_info,
        ppl_eval_info=ppl_eval_info,
        plot_paths=plot_paths,
        target_info={
            "block_indices": resolved_block_indices,
            "target_linear_names": list(config.target.target_linear_names),
            "num_blocks": len(resolved_block_indices),
            "num_target_layers": len(target_specs),
            "collected_token_counts": collected_token_counts,
            "model_name": config.data.model_name,
            "latent_mode": config.quant.latent_mode,
            "latent_bits": config.quant.latent_bits,
            "fit_device": config.quant.fit_device,
            "runtime_device": config.eval.device,
        },
    )

    write_json(output_dir / "results.json", asdict(artifacts))
    write_json(output_dir / "all_blocks_qkvo_results.json", asdict(artifacts))
    summary = build_analysis_summary(artifacts)
    write_text(output_dir / "summary.txt", summary)
    write_text(output_dir / "analysis_summary.txt", summary)
    logger.info("Artifacts saved to %s", output_dir)
    logger.info("=== Experiment end ===")
    return artifacts


def make_block10_qproj_latent_distribution_config(base_config: ExperimentConfig, latent_bits: int) -> ExperimentConfig:
    return replace(
        base_config,
        quant=replace(base_config.quant, latent_mode="discrete", latent_bits=latent_bits),
        target=replace(base_config.target, block_indices=(10,), target_linear_names=("q_proj",)),
        output_dir=str(Path(base_config.output_dir) / f"latent_bits_{latent_bits}"),
        run_mode="single",
    )


def build_brief_payload(artifacts: ExperimentArtifacts, output_dir: str) -> Dict[str, object]:
    payload = {
        "output_dir": output_dir,
        "baseline_ppl": artifacts.baseline_ppl,
        "quantized_ppl": artifacts.quantized_ppl,
        "quantized_delta": artifacts.quantized_ppl - artifacts.baseline_ppl,
        "latent_mode": artifacts.config["quant"]["latent_mode"],
        "latent_bits": artifacts.config["quant"]["latent_bits"],
    }
    if not math.isnan(artifacts.sq_baseline_ppl):
        payload["sq_baseline_ppl"] = artifacts.sq_baseline_ppl
        payload["sq_delta"] = artifacts.sq_baseline_ppl - artifacts.baseline_ppl
    return payload


def parse_int_csv(spec: str) -> Tuple[int, ...]:
    values = tuple(int(part.strip()) for part in spec.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Latent-bits experiment runner built on qkvo_refactor.")
    parser.add_argument(
        "command",
        nargs="?",
        default="experiment",
        choices=("experiment", "latent-dist", "block10-qproj-latent-dist", "latent-sweep", "sweep"),
        help="Run a standard experiment, one block10/q_proj latent-distribution job, or a latent-bits sweep.",
    )
    add_experiment_args(parser)
    parser.set_defaults(
        max_iters=300,
        codebook="s8",
        output_dir="./.new_quant_function",
        block_indices="8,9,10,11",
        target_linear_names="q_proj,k_proj,v_proj,out_proj",
        latent_mode="discrete",
    )
    parser.add_argument("--latent-bits", type=int, default=5, help="Uniform latent quantization bit-width.")
    parser.add_argument(
        "--run-mode",
        choices=("single", "both"),
        default="single",
        help="Run one latent_mode from --latent-mode, or run both discrete and continuous back to back.",
    )
    parser.add_argument("--continuous-subdir", default="continuous")
    parser.add_argument("--latent-bits-grid", default="2,3,4", help="Comma-separated latent bit-widths for sweep mode.")
    parser.add_argument("--print-num-combos", action="store_true", help="Print the number of latent-bits sweep runs and exit.")
    parser.add_argument("--array-task-id", type=int, default=None, help="Run exactly one latent-bits sweep combo by index.")
    parser.add_argument("--rebuild-summary", action="store_true", help="Rebuild sweep summary files from existing outputs.")
    return parser


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
    config.quant.dtype = args.dtype
    config.quant.init_mode = args.init_mode
    config.quant.error_mode = args.error_mode
    config.quant.latent_mode = args.latent_mode
    config.quant.latent_bits = args.latent_bits
    config.quant.ip_reg_gamma = args.ip_reg_gamma
    config.quant.ip_reg_inner_iters = args.ip_reg_inner_iters
    config.quant.fit_device = args.fit_device

    if args.device is not None:
        config.eval.device = args.device
    config.eval.stride = args.stride

    config.output_dir = args.output_dir
    config.seed = args.seed
    config.run_mode = args.run_mode
    config.continuous_subdir = args.continuous_subdir
    config.run_sq_baseline = not args.skip_sq_baseline
    config.save_plots = not args.no_plots
    return config


def run_experiment_mode(config: ExperimentConfig) -> Dict[str, ExperimentArtifacts]:
    results: Dict[str, ExperimentArtifacts] = {}

    if config.run_mode == "single":
        results[config.quant.latent_mode] = run_all_blocks_qkvo_experiment(config)
        return results

    discrete_config = replace(config, quant=replace(config.quant, latent_mode="discrete"))
    continuous_config = replace(
        config,
        quant=replace(config.quant, latent_mode="continuous"),
        output_dir=str(Path(config.output_dir) / config.continuous_subdir),
    )
    results["discrete"] = run_all_blocks_qkvo_experiment(discrete_config)
    results["continuous"] = run_all_blocks_qkvo_experiment(continuous_config)
    return results


def run_latent_distribution_once(base_config: ExperimentConfig, latent_bits: int) -> ExperimentArtifacts:
    config = make_block10_qproj_latent_distribution_config(base_config, latent_bits)
    artifacts = run_all_blocks_qkvo_experiment(config)
    layer_name = "block10.q_proj"
    summary = {
        "output_dir": config.output_dir,
        "latent_bits": latent_bits,
        "layer_name": layer_name,
        "target_info": artifacts.target_info,
        "latent_distribution_info": artifacts.latent_distribution_info.get(layer_name, {}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return artifacts


def load_results_payload(run_dir: Path) -> Optional[Dict[str, object]]:
    for filename in ("results.json", "all_blocks_qkvo_results.json"):
        path = run_dir / filename
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_latent_sweep_summary(output_root: Path, latent_bits_grid: Tuple[int, ...]) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for combo_index, latent_bits in enumerate(latent_bits_grid):
        run_dir = output_root / f"latent_bits_{latent_bits}"
        payload = load_results_payload(run_dir)
        if payload is None:
            continue
        baseline_ppl = float(payload["baseline_ppl"])
        quantized_ppl = float(payload["quantized_ppl"])
        row = {
            "combo_index": combo_index,
            "latent_bits": latent_bits,
            "run_dir": str(run_dir),
            "baseline_ppl": baseline_ppl,
            "sq_baseline_ppl": float(payload["sq_baseline_ppl"]),
            "quantized_ppl": quantized_ppl,
            "quantized_delta": quantized_ppl - baseline_ppl,
        }
        rows.append(row)

    summary = {
        "output_dir": str(output_root),
        "completed_runs": len(rows),
        "latent_bits_grid": list(latent_bits_grid),
        "rows": rows,
    }
    write_json(output_root / "latent_bits_summary.json", summary)

    ranking = sorted(rows, key=lambda row: (row["quantized_ppl"], row["quantized_delta"]))
    ranking_lines = ["Latent Bits Sweep Ranking", ""]
    for rank, row in enumerate(ranking, start=1):
        ranking_lines.append(
            f"{rank:02d}. bits={row['latent_bits']} | quantized={row['quantized_ppl']:.6f} | "
            f"delta={row['quantized_delta']:.6f} | dir={row['run_dir']}"
        )
    write_text(output_root / "latent_bits_ranking.txt", "\n".join(ranking_lines) + "\n")
    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = args_to_config(args)

    if args.command in {"latent-sweep", "sweep"}:
        latent_bits_grid = parse_int_csv(args.latent_bits_grid)
        if args.print_num_combos:
            print(len(latent_bits_grid))
            return
        if args.rebuild_summary:
            summary = save_latent_sweep_summary(Path(config.output_dir), latent_bits_grid)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return
        if args.array_task_id is not None:
            if args.array_task_id < 0 or args.array_task_id >= len(latent_bits_grid):
                raise IndexError(f"array-task-id {args.array_task_id} is out of range for {len(latent_bits_grid)} combos.")
            selected_bits = (latent_bits_grid[args.array_task_id],)
        else:
            selected_bits = latent_bits_grid

        for latent_bits in selected_bits:
            run_latent_distribution_once(config, latent_bits)

        summary = save_latent_sweep_summary(Path(config.output_dir), latent_bits_grid)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command in {"latent-dist", "block10-qproj-latent-dist"}:
        artifacts = run_latent_distribution_once(config, args.latent_bits)
        print(json.dumps(build_brief_payload(artifacts, artifacts.config["output_dir"]), ensure_ascii=False, indent=2))
        return

    results = run_experiment_mode(config)
    payload = {
        name: build_brief_payload(artifacts, artifacts.config["output_dir"])
        for name, artifacts in results.items()
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
