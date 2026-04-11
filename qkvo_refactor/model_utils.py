from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import get_torch_dtype, quantize_nearest, scalar_quantize_maxabs
from .config import ExperimentConfig
from .quantizer import QuantizationState


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
        return quantize_nearest(z_continuous, self.codebook)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape[:-1]
        input_dim = hidden_states.shape[-1]
        x_flat = hidden_states.reshape(-1, input_dim)
        z_x = self._encode_x(x_flat)
        output = (z_x * self.coeff.unsqueeze(0)) @ self.Z_w
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
