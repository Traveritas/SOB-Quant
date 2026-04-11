from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Config
# ============================================================


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
class EvalConfig:
    stride: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TargetConfig:
    block_indices: Optional[Tuple[int, ...]] = None
    target_linear_names: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")


@dataclass
class SQConfig:
    bits_list: Tuple[int, ...] = (1, 2, 3, 4)
    eps: float = 1e-8


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    sq: SQConfig = field(default_factory=SQConfig)
    output_dir: str = "./outputs_opt125m_sq_bits_ppl"
    seed: int = 42


@dataclass
class TargetModuleSpec:
    key: str
    block_index: int
    linear_name: str
    parent_module: nn.Module
    module: nn.Linear


@dataclass
class SQRunResult:
    bits: int
    ppl: float
    ppl_delta_vs_fp: float
    elapsed_sec: float
    metrics_by_layer: Dict[str, Dict[str, float]]
    metrics_avg: Dict[str, float]


@dataclass
class ExperimentArtifacts:
    config: Dict
    baseline_ppl: float
    baseline_eval_info: Dict[str, float]
    sq_results: List[SQRunResult]
    target_info: Dict[str, object]
    timing_info: Dict[str, float]


# ============================================================
# Utilities
# ============================================================


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"opt125m_sq_bits_{output_dir.resolve()}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = logging.FileHandler(output_dir / "experiment.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def average_metrics(metrics_by_layer: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if not metrics_by_layer:
        return {}
    keys = sorted(next(iter(metrics_by_layer.values())).keys())
    return {
        key: float(sum(layer_metrics[key] for layer_metrics in metrics_by_layer.values()) / len(metrics_by_layer))
        for key in keys
    }


def tensor_stats(t: torch.Tensor) -> Dict[str, object]:
    t_cpu = t.detach().float().cpu()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "mean": float(t_cpu.mean().item()),
        "std": float(t_cpu.std(unbiased=False).item()),
        "min": float(t_cpu.min().item()),
        "max": float(t_cpu.max().item()),
        "fro_norm": float(torch.linalg.norm(t_cpu).item()),
    }


# ============================================================
# Exact-bits scalar quantization
# ============================================================


class ScalarCodebookQuantizer:
    """
    Online scalar quantizer used for both activations and weights.

    - bits == 1: binary sign quantization with codebook {-scale, +scale}
    - bits >= 2: standard signed integer range [-(2^{b-1}), 2^{b-1}-1]
      with max-abs calibration for scale
    """

    def __init__(self, bits: int, eps: float = 1e-8):
        if bits < 1:
            raise ValueError(f"bits must be >= 1, got {bits}")
        self.bits = int(bits)
        self.eps = float(eps)

    def scale_from_tensor(self, x: torch.Tensor) -> torch.Tensor:
        maxabs = torch.max(torch.abs(x))
        if self.bits == 1:
            scale = maxabs
        else:
            qmax = float((2 ** (self.bits - 1)) - 1)
            if qmax <= 0:
                raise ValueError(f"Invalid qmax for bits={self.bits}")
            scale = maxabs / qmax
        return torch.clamp(scale, min=self.eps)

    def quantize(self, x: torch.Tensor, scale: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if scale is None:
            scale = self.scale_from_tensor(x)
        scale = torch.clamp(scale, min=self.eps)

        if self.bits == 1:
            q = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
            x_q = q * scale
            return x_q, scale

        qmin = -(2 ** (self.bits - 1))
        qmax = (2 ** (self.bits - 1)) - 1
        q = torch.clamp(torch.round(x / scale), qmin, qmax)
        x_q = q * scale
        return x_q, scale


# ============================================================
# SQ module
# ============================================================


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

    def _quantize_x(self, x: torch.Tensor) -> torch.Tensor:
        quantizer = ScalarCodebookQuantizer(bits=self.bits, eps=self.eps)
        x_q, _ = quantizer.quantize(x, scale=self.x_scale.to(device=x.device, dtype=x.dtype))
        return x_q

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape[:-1]
        d = hidden_states.shape[-1]
        x_flat = hidden_states.reshape(-1, d)
        x_q = self._quantize_x(x_flat)
        x_q = x_q.to(self.weight_quantized.dtype)
        output = x_q @ self.weight_quantized
        if self.bias is not None:
            output = output + self.bias.unsqueeze(0)
        output = output.reshape(*orig_shape, self.weight_quantized.shape[1])
        return output.to(hidden_states.dtype)


# ============================================================
# Model / data / hooks
# ============================================================


class MultiLinearInputCollector:
    def __init__(self, max_tokens: int):
        self.max_tokens = int(max_tokens)
        self.collected: Dict[str, List[torch.Tensor]] = {}
        self.num_tokens: Dict[str, int] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, layer_name: str):
        def _hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
            current = self.num_tokens.get(layer_name, 0)
            if current >= self.max_tokens:
                return
            hidden_states = inputs[0].detach()
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            remaining = self.max_tokens - current
            if flat.shape[0] > remaining:
                flat = flat[:remaining]
            self.collected.setdefault(layer_name, []).append(flat.cpu())
            self.num_tokens[layer_name] = current + flat.shape[0]

        return _hook

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
            X = torch.cat(chunks, dim=0)
            matrices[layer_name] = X.T.contiguous()
        return matrices


def load_model_and_tokenizer(config: ExperimentConfig, logger: logging.Logger):
    logger.info("加载模型与 tokenizer：%s", config.data.model_name)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        config.data.model_name,
        use_fast=config.data.tokenizer_use_fast,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.data.model_name)
    model.eval()
    model.to(config.eval.device)
    elapsed = time.perf_counter() - t0
    logger.info(
        "模型加载完成 | hidden_size=%s vocab_size=%s device=%s elapsed=%.3fs",
        getattr(model.config, "hidden_size", "unknown"),
        getattr(model.config, "vocab_size", "unknown"),
        config.eval.device,
        elapsed,
    )
    return model, tokenizer, elapsed


def load_text_split(config: ExperimentConfig, split: str, logger: logging.Logger) -> str:
    logger.info("加载数据集 split=%s", split)
    t0 = time.perf_counter()
    ds = load_dataset(config.data.dataset_name, config.data.dataset_config, split=split)
    if "text" not in ds.column_names:
        raise ValueError(f"Dataset split {split} has no 'text' column.")
    texts = [t for t in ds["text"] if isinstance(t, str) and t.strip()]
    text = "\n\n".join(texts)
    elapsed = time.perf_counter() - t0
    logger.info("split=%s 加载完成 | non-empty text blocks=%d chars=%d elapsed=%.3fs", split, len(texts), len(text), elapsed)
    return text


def tokenize_text(text: str, tokenizer, max_tokens: Optional[int] = None) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"][0]
    if max_tokens is not None:
        input_ids = input_ids[:max_tokens]
    return input_ids


def get_all_block_attention_targets(
    model,
    target_linear_names: Tuple[str, ...],
    block_indices: Optional[Tuple[int, ...]] = None,
) -> Dict[str, TargetModuleSpec]:
    decoder_layers = model.model.decoder.layers
    if block_indices is None:
        resolved_block_indices = tuple(range(len(decoder_layers)))
    else:
        resolved_block_indices = tuple(block_indices)

    specs: Dict[str, TargetModuleSpec] = {}
    for block_index in resolved_block_indices:
        block = decoder_layers[block_index]
        attn = block.self_attn
        for name in target_linear_names:
            if not hasattr(attn, name):
                raise AttributeError(f"Block {block_index} self_attn has no linear named: {name}")
            module = getattr(attn, name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Target module block {block_index} {name} is not nn.Linear, got {type(module)}")
            key = f"block{block_index}.{name}"
            specs[key] = TargetModuleSpec(
                key=key,
                block_index=int(block_index),
                linear_name=name,
                parent_module=attn,
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
    logger.info("开始收集目标线性层输入 hidden states | requested_tokens=%d layers=%s", max_tokens, list(target_modules.keys()))
    t0 = time.perf_counter()
    collector = MultiLinearInputCollector(max_tokens=max_tokens)
    collector.register(target_modules)

    num_forward_chunks = 0
    try:
        max_length = getattr(model.config, "max_position_embeddings", 2048)
        with torch.no_grad():
            for start in range(0, input_ids.numel(), max_length):
                end = min(start + max_length, input_ids.numel())
                chunk = input_ids[start:end].unsqueeze(0).to(device)
                _ = model(input_ids=chunk)
                num_forward_chunks += 1
                if all(collector.num_tokens.get(name, 0) >= max_tokens for name in target_modules):
                    break
    finally:
        collector.remove()

    matrices = collector.get_matrices()
    elapsed = time.perf_counter() - t0
    for layer_name, X in matrices.items():
        logger.info(
            "输入收集完成 | layer=%s collected_tokens=%d forward_chunks=%d elapsed=%.3fs X_shape=%s",
            layer_name,
            collector.num_tokens[layer_name],
            num_forward_chunks,
            elapsed,
            list(X.shape),
        )
    return matrices, dict(collector.num_tokens)


# ============================================================
# Metrics and PPL
# ============================================================


@torch.no_grad()
def compute_sq_metrics(
    X: torch.Tensor,
    W: torch.Tensor,
    bits: int,
    eps: float,
) -> Dict[str, float]:
    quantizer = ScalarCodebookQuantizer(bits=bits, eps=eps)

    device = W.device
    dtype = W.dtype
    X = X.to(device=device, dtype=dtype)
    W = W.to(device=device, dtype=dtype)

    x_scale = quantizer.scale_from_tensor(X)
    w_scale = quantizer.scale_from_tensor(W)
    X_q, _ = quantizer.quantize(X, x_scale)
    W_q, _ = quantizer.quantize(W, w_scale)

    err_x = float(torch.sum((X - X_q) ** 2).item() / max(torch.sum(X ** 2).item(), 1e-12))
    err_w = float(torch.sum((W - W_q) ** 2).item() / max(torch.sum(W ** 2).item(), 1e-12))

    denom = 0.0
    numer = 0.0
    chunk_tokens = 128
    for start in range(0, X.shape[1], chunk_tokens):
        end = min(start + chunk_tokens, X.shape[1])
        X_chunk = X[:, start:end]
        X_chunk_q = X_q[:, start:end]
        Y_true = X_chunk.T @ W
        Y_hat = X_chunk_q.T @ W_q
        diff = Y_true - Y_hat
        numer += float(torch.sum(diff * diff).item())
        denom += float(torch.sum(Y_true * Y_true).item())

    err_linear = numer / max(denom, 1e-12)
    return {
        "rel_recon_error_x": err_x,
        "rel_recon_error_w": err_w,
        "rel_linear_error": err_linear,
        "x_scale": float(x_scale.item()),
        "w_scale": float(w_scale.item()),
    }


@torch.no_grad()
def evaluate_perplexity_sliding_window(
    model,
    tokenizer,
    text: str,
    device: str,
    stride: int = 512,
    max_eval_tokens: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    tag: str = "eval",
) -> Tuple[float, Dict[str, float]]:
    if logger is not None:
        logger.info("开始 PPL 评测 | tag=%s stride=%d max_eval_tokens=%s", tag, stride, str(max_eval_tokens))
    t0 = time.perf_counter()

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"][0]
    if max_eval_tokens is not None:
        input_ids = input_ids[:max_eval_tokens]

    input_ids = input_ids.to(device)
    max_length = getattr(model.config, "max_position_embeddings", 2048)

    nlls: List[torch.Tensor] = []
    prev_end_loc = 0
    total_target_tokens = 0
    num_windows = 0

    for begin_loc in range(0, input_ids.size(0), stride):
        end_loc = min(begin_loc + max_length, input_ids.size(0))
        trg_len = end_loc - prev_end_loc
        input_ids_window = input_ids[begin_loc:end_loc].unsqueeze(0)
        target_ids = input_ids_window.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(input_ids_window, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)

        total_target_tokens += int(trg_len)
        num_windows += 1
        prev_end_loc = end_loc
        if end_loc == input_ids.size(0):
            break

    ppl = torch.exp(torch.stack(nlls).sum() / max(total_target_tokens, 1))
    elapsed = time.perf_counter() - t0
    stats = {
        "elapsed_sec": float(elapsed),
        "num_eval_tokens": float(input_ids.numel()),
        "num_windows": float(num_windows),
        "num_target_tokens": float(total_target_tokens),
    }
    if logger is not None:
        logger.info(
            "PPL 评测完成 | tag=%s ppl=%.6f eval_tokens=%d target_tokens=%d windows=%d elapsed=%.3fs",
            tag,
            float(ppl.item()),
            input_ids.numel(),
            total_target_tokens,
            num_windows,
            elapsed,
        )
    return float(ppl.item()), stats


# ============================================================
# Module replacement
# ============================================================


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


# ============================================================
# SQ builders
# ============================================================


def build_sq_target_modules(
    target_specs: Dict[str, TargetModuleSpec],
    X_calib_by_layer: Dict[str, torch.Tensor],
    bits: int,
    eps: float,
    logger: logging.Logger,
    device: str,
) -> Tuple[
    Dict[str, nn.Module],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, object]],
]:
    logger.info("开始构建 SQ 量化模块 | bits=%d", bits)
    quantizer = ScalarCodebookQuantizer(bits=bits, eps=eps)

    sq_modules: Dict[str, nn.Module] = {}
    metrics_by_layer: Dict[str, Dict[str, float]] = {}
    tensor_info: Dict[str, Dict[str, object]] = {}

    for layer_name, spec in target_specs.items():
        X_calib = X_calib_by_layer[layer_name].to(device=device, dtype=torch.float32)
        W = spec.module.weight.detach().T.to(device=device, dtype=torch.float32)
        bias = None if spec.module.bias is None else spec.module.bias.detach().to(device=device, dtype=torch.float32)

        x_scale = quantizer.scale_from_tensor(X_calib)
        w_q, w_scale = quantizer.quantize(W)
        module = ScalarQuantizedXWLinear(
            weight_quantized=w_q,
            x_scale=x_scale,
            bits=bits,
            bias=bias,
            eps=eps,
        )
        module.to(device)

        sq_modules[layer_name] = module
        metrics_by_layer[layer_name] = compute_sq_metrics(X_calib, W, bits=bits, eps=eps)
        tensor_info[f"{layer_name}.X_calib"] = tensor_stats(X_calib)
        tensor_info[f"{layer_name}.W"] = tensor_stats(W)
        tensor_info[f"{layer_name}.W_q"] = tensor_stats(w_q)
        tensor_info[f"{layer_name}.x_scale"] = {"value": float(x_scale.item())}
        tensor_info[f"{layer_name}.w_scale"] = {"value": float(w_scale.item())}

        logger.info(
            "SQ 层完成 | bits=%d layer=%s x_scale=%.6e w_scale=%.6e rel_linear_error=%.6f",
            bits,
            layer_name,
            float(x_scale.item()),
            float(w_scale.item()),
            metrics_by_layer[layer_name]["rel_linear_error"],
        )

    return sq_modules, metrics_by_layer, tensor_info


# ============================================================
# Runner
# ============================================================


def run_sq_bits_experiment(config: ExperimentConfig) -> ExperimentArtifacts:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    timing_info: Dict[str, float] = {}

    set_seed(config.seed)
    model, tokenizer, model_load_sec = load_model_and_tokenizer(config, logger)
    timing_info["model_load_sec"] = model_load_sec

    target_specs = get_all_block_attention_targets(
        model,
        target_linear_names=config.target.target_linear_names,
        block_indices=config.target.block_indices,
    )
    resolved_block_indices = sorted({spec.block_index for spec in target_specs.values()})

    calib_text = load_text_split(config, config.data.calib_split, logger)
    eval_text = load_text_split(config, config.data.eval_split, logger)

    baseline_ppl, baseline_eval_info = evaluate_perplexity_sliding_window(
        model=model,
        tokenizer=tokenizer,
        text=eval_text,
        device=config.eval.device,
        stride=config.eval.stride,
        max_eval_tokens=config.data.eval_num_tokens,
        logger=logger,
        tag="baseline_fp",
    )

    t0 = time.perf_counter()
    calib_input_ids = tokenize_text(calib_text, tokenizer, max_tokens=config.data.calib_num_tokens)
    timing_info["tokenize_calib_sec"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_calib_by_layer, collected_token_counts = collect_target_inputs(
        model=model,
        input_ids=calib_input_ids,
        target_modules={layer_name: spec.module for layer_name, spec in target_specs.items()},
        max_tokens=config.data.calib_num_tokens,
        device=config.eval.device,
        logger=logger,
    )
    timing_info["collect_target_inputs_sec"] = time.perf_counter() - t0

    sq_results: List[SQRunResult] = []
    summary_rows: List[Dict[str, object]] = []

    for bits in config.sq.bits_list:
        t0 = time.perf_counter()
        logger.info("========== 开始 SQ %d-bit ==========" , bits)

        sq_modules, metrics_by_layer, tensor_info = build_sq_target_modules(
            target_specs=target_specs,
            X_calib_by_layer=X_calib_by_layer,
            bits=bits,
            eps=config.sq.eps,
            logger=logger,
            device=config.eval.device,
        )

        original_modules = replace_target_modules(target_specs, sq_modules)
        try:
            sq_ppl, sq_eval_info = evaluate_perplexity_sliding_window(
                model=model,
                tokenizer=tokenizer,
                text=eval_text,
                device=config.eval.device,
                stride=config.eval.stride,
                max_eval_tokens=config.data.eval_num_tokens,
                logger=logger,
                tag=f"sq_{bits}bit",
            )
        finally:
            restore_target_modules(target_specs, original_modules)

        elapsed = time.perf_counter() - t0
        result = SQRunResult(
            bits=bits,
            ppl=sq_ppl,
            ppl_delta_vs_fp=float(sq_ppl - baseline_ppl),
            elapsed_sec=float(elapsed),
            metrics_by_layer=metrics_by_layer,
            metrics_avg=average_metrics(metrics_by_layer),
        )
        sq_results.append(result)

        bit_dir = output_dir / f"sq_{bits}bit"
        bit_dir.mkdir(parents=True, exist_ok=True)
        with open(bit_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "bits": bits,
                    "baseline_ppl": baseline_ppl,
                    "sq_ppl": sq_ppl,
                    "ppl_delta_vs_fp": sq_ppl - baseline_ppl,
                    "elapsed_sec": elapsed,
                    "eval_info": sq_eval_info,
                    "metrics_by_layer": metrics_by_layer,
                    "metrics_avg": result.metrics_avg,
                    "tensor_info": tensor_info,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        summary_rows.append(
            {
                "bits": bits,
                "baseline_ppl": baseline_ppl,
                "sq_ppl": sq_ppl,
                "ppl_delta_vs_fp": sq_ppl - baseline_ppl,
                "elapsed_sec": elapsed,
            }
        )
        logger.info(
            "SQ %d-bit 完成 | ppl=%.6f delta_vs_fp=%.6f elapsed=%.3fs",
            bits,
            sq_ppl,
            sq_ppl - baseline_ppl,
            elapsed,
        )

    artifacts = ExperimentArtifacts(
        config=asdict(config),
        baseline_ppl=baseline_ppl,
        baseline_eval_info=baseline_eval_info,
        sq_results=sq_results,
        target_info={
            "block_indices": resolved_block_indices,
            "num_blocks": len(resolved_block_indices),
            "target_linear_names": list(config.target.target_linear_names),
            "collected_token_counts": collected_token_counts,
            "model_name": config.data.model_name,
        },
        timing_info=timing_info,
    )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(asdict(artifacts), f, ensure_ascii=False, indent=2)

    lines = [
        "===== OPT-125m all-block QKVO SQ PPL summary =====",
        f"baseline_ppl={baseline_ppl:.6f}",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['bits']} bit | baseline={row['baseline_ppl']:.6f} | sq={row['sq_ppl']:.6f} | "
            f"delta={row['ppl_delta_vs_fp']:.6f} | elapsed={row['elapsed_sec']:.3f}s"
        )
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    logger.info("结果已保存到：%s", output_dir)
    return artifacts


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SQ quantization for all OPT-125m blocks (Q/K/V/O) with PPL eval")
    parser.add_argument("--model-name", type=str, default="facebook/opt-125m")
    parser.add_argument("--dataset-name", type=str, default="wikitext")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--calib-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--calib-num-tokens", type=int, default=4096)
    parser.add_argument("--eval-num-tokens", type=int, default=None)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--bits", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--block-indices", type=int, nargs="*", default=None)
    parser.add_argument("--target-linear-names", type=str, nargs="+", default=["q_proj", "k_proj", "v_proj", "out_proj"])
    parser.add_argument("--output-dir", type=str, default="./outputs_opt125m_sq_bits_ppl")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            calib_split=args.calib_split,
            eval_split=args.eval_split,
            calib_num_tokens=args.calib_num_tokens,
            eval_num_tokens=args.eval_num_tokens,
        ),
        eval=EvalConfig(
            stride=args.stride,
            device=args.device,
        ),
        target=TargetConfig(
            block_indices=None if args.block_indices is None else tuple(args.block_indices),
            target_linear_names=tuple(args.target_linear_names),
        ),
        sq=SQConfig(
            bits_list=tuple(args.bits),
        ),
        output_dir=args.output_dir,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    config = make_config(args)
    artifacts = run_sq_bits_experiment(config)
    print(json.dumps(asdict(artifacts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
