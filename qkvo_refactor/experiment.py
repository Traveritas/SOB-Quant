from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .common import (
    infer_sq_bitwidth_from_codebook,
    scalar_quant_scale_maxabs,
    scalar_quantize_maxabs,
    set_seed,
    setup_logger,
    tensor_stats,
    write_json,
    write_text,
)
from .config import ExperimentConfig
from .model_utils import (
    QuantizedLinear,
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
from .quantizer import LatticeLinearQuantizer, QuantizationState
from .quantizer import QuantizerTrackingOptions, UTraceObserver


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
    timing_info: Dict[str, float]
    ppl_eval_info: Dict[str, Dict[str, float]]
    plot_paths: Dict[str, Dict[str, str]]
    tracking_info: Dict[str, Dict[str, object]]
    target_info: Dict[str, object]


def average_metrics(metrics_by_layer: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if not metrics_by_layer:
        return {}
    keys = sorted(next(iter(metrics_by_layer.values())).keys())
    return {
        key: float(sum(layer_metrics[key] for layer_metrics in metrics_by_layer.values()) / len(metrics_by_layer))
        for key in keys
    }


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
            diff = torch.abs(Z_x.unsqueeze(-1) - state.codebook.to(Z_x.device))
            indices = torch.argmin(diff, dim=-1)
            Z_x = state.codebook.to(Z_x.device)[indices]

        Y_hat = (Z_x.T * coeff.unsqueeze(0)) @ state.Z_w
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


@torch.no_grad()
def compute_sq_metrics(
    X: torch.Tensor,
    W: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bits: int,
) -> Dict[str, float]:
    X = X.to(device=W.device, dtype=W.dtype)
    W = W.to(device=W.device, dtype=W.dtype)

    X_q, _ = scalar_quantize_maxabs(X, bits=bits, scale=x_scale.to(device=W.device, dtype=W.dtype))
    W_q, _ = scalar_quantize_maxabs(W, bits=bits, scale=w_scale.to(device=W.device, dtype=W.dtype))

    x_error = float(torch.sum((X - X_q) ** 2).item() / max(torch.sum(X**2).item(), 1e-12))
    w_error = float(torch.sum((W - W_q) ** 2).item() / max(torch.sum(W**2).item(), 1e-12))

    numerator = 0.0
    denominator = 0.0
    chunk_tokens = 128
    for start in range(0, X.shape[1], chunk_tokens):
        end = min(start + chunk_tokens, X.shape[1])
        X_chunk = X[:, start:end]
        X_chunk_q = X_q[:, start:end]
        Y_true = X_chunk.T @ W
        Y_hat = X_chunk_q.T @ W_q
        delta = Y_true - Y_hat
        numerator += float(torch.sum(delta * delta).item())
        denominator += float(torch.sum(Y_true * Y_true).item())

    return {
        "x_error": x_error,
        "w_error": w_error,
        "linear_error": numerator / max(denominator, 1e-12),
    }


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
        else:
            plot_paths[layer_name] = {}

    return quantized_modules, states, metrics_by_layer, tensor_info, plot_paths, tracking_by_layer


def build_sq_modules(
    target_specs: Dict[str, TargetModuleSpec],
    X_calib_by_layer: Dict[str, torch.Tensor],
    config: ExperimentConfig,
    logger: logging.Logger,
) -> Tuple[
    Dict[str, nn.Module],
    int,
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, object]],
]:
    logger.info("Build SQ baseline modules")

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


def build_summary(artifacts: ExperimentArtifacts) -> str:
    lines = [
        "QKVO Refactor Summary",
        f"- baseline_ppl: {artifacts.baseline_ppl:.6f}",
        f"- quantized_ppl: {artifacts.quantized_ppl:.6f}",
        f"- quantized_delta: {artifacts.quantized_ppl - artifacts.baseline_ppl:.6f}",
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

    sq_baseline_ppl = float("nan")
    sq_metrics_by_layer: Dict[str, Dict[str, float]] = {}
    sq_tensor_info: Dict[str, Dict[str, object]] = {}
    sq_bits: Optional[int] = None

    if config.run_sq_baseline:
        start = time.perf_counter()
        sq_modules, sq_bits, sq_metrics_by_layer, sq_tensor_info = build_sq_modules(
            target_specs=target_specs,
            X_calib_by_layer=X_calib_by_layer,
            config=config,
            logger=logger,
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
        },
    )

    write_json(output_dir / "results.json", asdict(artifacts))
    write_text(output_dir / "summary.txt", build_summary(artifacts))
    logger.info("Artifacts saved to %s", output_dir)
    logger.info("=== Experiment end ===")
    return artifacts
