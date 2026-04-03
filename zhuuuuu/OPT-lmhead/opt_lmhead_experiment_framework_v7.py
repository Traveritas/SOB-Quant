
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 配置
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
class QuantizerConfig:
    beta: float = 1.0
    max_iters: int = 60
    tol: float = 1e-5
    convergence_check_every: int = 1
    codebook: Tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    dtype: str = "float32"
    eps: float = 1e-8
    log_every: int = 1


@dataclass
class EvalConfig:
    stride: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    quant: QuantizerConfig = field(default_factory=QuantizerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output_dir: str = "./outputs_opt_lmhead_stage1"
    seed: int = 42


# ============================================================
# 工具函数
# ============================================================


def set_seed(seed: int) -> None:
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


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"opt_lmhead_stage1_{output_dir.resolve()}")
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


def quantize_nearest(z_continuous: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(z_continuous.unsqueeze(-1) - codebook)
    idx = torch.argmin(diff, dim=-1)
    return codebook[idx]


def weighted_cross(X: torch.Tensor, weights: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    return (X * weights.unsqueeze(0)) @ Z.T


def weighted_gram(Z: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (Z * weights.unsqueeze(0)) @ Z.T


def relative_weighted_reconstruction_error(
    X: torch.Tensor,
    X_hat: torch.Tensor,
    inv_norms: torch.Tensor,
    average: bool = False,
) -> torch.Tensor:
    residual = X - X_hat
    per_col_err = torch.sum(residual * residual, dim=0)
    loss = torch.sum(inv_norms * per_col_err)
    if average:
        loss = loss / max(X.shape[1], 1)
    return loss


def format_shape(t: torch.Tensor) -> List[int]:
    return list(t.shape)


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


def log_tensor_stats(logger: logging.Logger, name: str, t: torch.Tensor) -> None:
    stats = tensor_stats(t)
    logger.info(
        "%s stats | shape=%s dtype=%s device=%s mean=%.6f std=%.6f min=%.6f max=%.6f fro=%.6f",
        name,
        stats["shape"],
        stats["dtype"],
        stats["device"],
        stats["mean"],
        stats["std"],
        stats["min"],
        stats["max"],
        stats["fro_norm"],
    )


def save_loss_plots(
    output_dir: Path,
    objective_history: List[float],
    objective_x_history: List[float],
    objective_w_history: List[float],
) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    if not objective_history:
        return paths

    xs = list(range(1, len(objective_history) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(xs, objective_history, marker="o", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("J")
    plt.title("Total objective per iteration")
    plt.grid(True, alpha=0.3)
    total_path = output_dir / "objective_total.png"
    plt.tight_layout()
    plt.savefig(total_path, dpi=160)
    plt.close()
    paths["objective_total"] = str(total_path)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, objective_x_history, marker="o", linewidth=1.5, label="J_x")
    plt.plot(xs, objective_w_history, marker="o", linewidth=1.5, label="J_w")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Objective components per iteration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    comp_path = output_dir / "objective_components.png"
    plt.tight_layout()
    plt.savefig(comp_path, dpi=160)
    plt.close()
    paths["objective_components"] = str(comp_path)
    return paths


def build_analysis_summary(artifacts: "ExperimentArtifacts") -> str:
    lines: List[str] = []
    lines.append("实验结果分析")
    lines.append(f"- 量化是否完成: {artifacts.quantization_completed}")
    lines.append(f"- baseline PPL: {artifacts.baseline_ppl:.6f}")
    lines.append(f"- SQ-W PPL: {artifacts.sq_w_baseline_ppl:.6f}")
    lines.append(f"- SQ-XW PPL: {artifacts.sq_xw_baseline_ppl:.6f}")
    lines.append(f"- Ours PPL: {artifacts.quantized_ppl:.6f}")
    lines.append(f"- Ours PPL 增量: {artifacts.quantized_ppl - artifacts.baseline_ppl:.6f}")
    lines.append(f"- SQ-W PPL 增量: {artifacts.sq_w_baseline_ppl - artifacts.baseline_ppl:.6f}")
    lines.append(f"- SQ-XW PPL 增量: {artifacts.sq_xw_baseline_ppl - artifacts.baseline_ppl:.6f}")
    lines.append("")
    lines.append("重建与近似误差（Ours）")
    for k, v in artifacts.quant_metrics.items():
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("重建与近似误差（SQ-W）")
    for k, v in artifacts.sq_w_metrics.items():
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("重建与近似误差（SQ-XW）")
    for k, v in artifacts.sq_xw_metrics.items():
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("关键张量")
    for name, info in artifacts.tensor_info.items():
        if 'shape' in info:
            lines.append(
                f"- {name}: shape={info['shape']}, dtype={info['dtype']}, "
                f"mean={info['mean']:.6f}, std={info['std']:.6f}, "
                f"min={info['min']:.6f}, max={info['max']:.6f}, fro={info['fro_norm']:.6f}"
            )
        elif 'value' in info:
            lines.append(f"- {name}: value={info['value']}")
    lines.append("")
    lines.append("耗时（秒）")
    for k, v in artifacts.timing_info.items():
        lines.append(f"- {k}: {v:.4f}")

    if artifacts.objective_history:
        first_j = artifacts.objective_history[0]
        last_j = artifacts.objective_history[-1]
        rel_drop = (first_j - last_j) / max(abs(first_j), 1e-12)
        lines.append("")
        lines.append("优化过程")
        lines.append(f"- convergence_iter: {artifacts.convergence_iter}")
        lines.append(f"- objective first: {first_j:.6f}")
        lines.append(f"- objective last: {last_j:.6f}")
        lines.append(f"- relative drop: {rel_drop:.6f}")

    return "\n".join(lines)
def run_stage1_lmhead_experiment(config: ExperimentConfig) -> ExperimentArtifacts:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)

    logger.info("========== 实验开始 ==========")
    logger.info("实验配置：%s", json.dumps(asdict(config), ensure_ascii=False, indent=2))

    timing_info: Dict[str, float] = {}
    ppl_eval_info: Dict[str, Dict[str, float]] = {}
    quantization_completed = False

    model, tokenizer, model_load_time = load_model_and_tokenizer(config, logger)
    timing_info["model_load_sec"] = model_load_time

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

    t0 = time.perf_counter()
    calib_input_ids = tokenize_text(calib_text, tokenizer, max_tokens=config.data.calib_num_tokens)
    timing_info["tokenize_calib_sec"] = time.perf_counter() - t0
    logger.info("校准文本 tokenization 完成 | calib_input_ids_len=%d", calib_input_ids.numel())

    t0 = time.perf_counter()
    X_calib = collect_lm_head_inputs(
        model=model,
        input_ids=calib_input_ids,
        max_tokens=config.data.calib_num_tokens,
        device=config.eval.device,
        logger=logger,
    )
    timing_info["collect_lm_head_inputs_sec"] = time.perf_counter() - t0

    # SQ baselines（不改变 ours 路径）
    t0 = time.perf_counter()
    sq_w_lm_head, sq_xw_lm_head, sq_w_metrics, sq_xw_metrics, sq_info = build_sq_baselines_from_model(model, X_calib, config, logger)
    timing_info["build_sq_baselines_sec"] = time.perf_counter() - t0

    # Ours quantizer（保持 v3 原逻辑）
    quantizer = LatticeLMHeadQuantizer(config.quant, logger=logger)
    t0 = time.perf_counter()
    quantized_lm_head, state, metrics, tensor_info = build_quantized_lm_head_from_model(
        model, quantizer, X_calib, logger
    )
    timing_info["fit_quantizer_and_metrics_sec"] = time.perf_counter() - t0
    timing_info["fit_quantizer_sec"] = state.fit_time_sec

    tensor_info.update({
        "sq_w_scale": {"value": sq_info["sq_w_scale"]},
        "sq_x_scale": {"value": sq_info["sq_x_scale"]},
        "sq_weight_q": sq_info["sq_weight_q"],
    })

    original_lm_head = model.lm_head

    try:
        model.lm_head = sq_w_lm_head
        sq_w_baseline_ppl, sq_w_eval_stats = evaluate_perplexity_sliding_window(
            model=model,
            tokenizer=tokenizer,
            text=eval_text,
            device=config.eval.device,
            stride=config.eval.stride,
            max_eval_tokens=config.data.eval_num_tokens,
            logger=logger,
            tag="sq_w_baseline",
        )
        ppl_eval_info["sq_w_baseline"] = sq_w_eval_stats

        model.lm_head = sq_xw_lm_head
        sq_xw_baseline_ppl, sq_xw_eval_stats = evaluate_perplexity_sliding_window(
            model=model,
            tokenizer=tokenizer,
            text=eval_text,
            device=config.eval.device,
            stride=config.eval.stride,
            max_eval_tokens=config.data.eval_num_tokens,
            logger=logger,
            tag="sq_xw_baseline",
        )
        ppl_eval_info["sq_xw_baseline"] = sq_xw_eval_stats

        model.lm_head = quantized_lm_head
        quantized_ppl, quantized_eval_stats = evaluate_perplexity_sliding_window(
            model=model,
            tokenizer=tokenizer,
            text=eval_text,
            device=config.eval.device,
            stride=config.eval.stride,
            max_eval_tokens=config.data.eval_num_tokens,
            logger=logger,
            tag="quantized_lm_head",
        )
        ppl_eval_info["quantized_lm_head"] = quantized_eval_stats
        quantization_completed = True
        logger.info("量化推理完成，SQ 与 Ours 的 PPL 已得到。")
    finally:
        model.lm_head = original_lm_head

    plot_paths = save_loss_plots(
        output_dir,
        state.objective_history,
        state.objective_x_history,
        state.objective_w_history,
    )
    logger.info("损失曲线已保存：%s", plot_paths)

    artifacts = ExperimentArtifacts(
        config=asdict(config),
        quantization_completed=quantization_completed,
        baseline_ppl=baseline_ppl,
        sq_w_baseline_ppl=sq_w_baseline_ppl,
        sq_xw_baseline_ppl=sq_xw_baseline_ppl,
        quantized_ppl=quantized_ppl,
        quant_metrics=metrics,
        sq_w_metrics=sq_w_metrics,
        sq_xw_metrics=sq_xw_metrics,
        convergence_iter=state.convergence_iter,
        objective_history=state.objective_history,
        objective_x_history=state.objective_x_history,
        objective_w_history=state.objective_w_history,
        tensor_info=tensor_info,
        timing_info=timing_info,
        ppl_eval_info=ppl_eval_info,
        plot_paths=plot_paths,
    )

    with open(output_dir / "stage1_results.json", "w", encoding="utf-8") as f:
        json.dump(asdict(artifacts), f, ensure_ascii=False, indent=2)

    summary_text = build_analysis_summary(artifacts)
    (output_dir / "analysis_summary.txt").write_text(summary_text, encoding="utf-8")
    logger.info("结果 JSON 与分析摘要已写入输出目录。")
    logger.info("========== 实验结束 ==========")
    return artifacts

