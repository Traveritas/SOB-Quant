from __future__ import annotations

import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from SQ_function import (
    infer_sq_bitwidth_from_codebook,
    quantize_nearest,
    scalar_quant_scale_maxabs,
    scalar_quantize_maxabs,
    uniform_quantize_maxabs,
    uniform_quantize_maxabs_codes,
)


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
    max_iters: int = 300
    tol: float = 1e-5
    convergence_check_every: int = 1
    codebook: Tuple[float, ...] = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    latent_bits: int = 5
    dtype: str = "float32"
    eps: float = 1e-8
    log_every: int = 1
    init_mode: str = "random"
    error_mode: str = "relative"
    latent_mode: str = "discrete"
    ip_reg_gamma: float = 0.0
    ip_reg_inner_iters: int = 1


@dataclass
class EvalConfig:
    stride: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TargetConfig:
    block_indices: Optional[Tuple[int, ...]] = field(default_factory=lambda: tuple(range(12)))
    target_linear_names: Tuple[str, ...] = ("k_proj", "v_proj")


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    quant: QuantizerConfig = field(default_factory=QuantizerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    output_dir: str = "./.new_quant_function"
    seed: int = 42
    run_mode: str = "both"
    continuous_subdir: str = "continuous"


config = ExperimentConfig(
    target=TargetConfig(
        block_indices=(8, 9, 10, 11),
        target_linear_names=("q_proj", "k_proj", "v_proj", "out_proj"),
    )
)


def make_block10_qproj_latent_distribution_config(latent_bits: int) -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(calib_num_tokens=2048, eval_num_tokens=1024),
        quant=QuantizerConfig(
            latent_mode="discrete",
            latent_bits=latent_bits,
            codebook=(-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
        ),
        eval=EvalConfig(),
        target=TargetConfig(block_indices=(10,), target_linear_names=("q_proj",)),
        output_dir=f"./.new_quant_function/latent_bits_{latent_bits}",
        seed=42,
        run_mode="discrete",
    )

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
    logger = logging.getLogger(f"opt_all_blocks_qkvo_stage1_{output_dir.resolve()}")
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


def distribution_stats(t: torch.Tensor) -> Dict[str, object]:
    t_cpu = t.detach().float().reshape(-1).cpu()
    abs_t = torch.abs(t_cpu)
    quantiles = torch.tensor([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999], dtype=torch.float32)
    q_vals = torch.quantile(t_cpu, quantiles)
    abs_q_vals = torch.quantile(abs_t, quantiles)
    return {
        "numel": int(t_cpu.numel()),
        "mean": float(t_cpu.mean().item()),
        "std": float(t_cpu.std(unbiased=False).item()),
        "min": float(t_cpu.min().item()),
        "max": float(t_cpu.max().item()),
        "mean_abs": float(abs_t.mean().item()),
        "max_abs": float(abs_t.max().item()),
        "quantiles": {f"{float(q):.3f}": float(v.item()) for q, v in zip(quantiles, q_vals)},
        "abs_quantiles": {f"{float(q):.3f}": float(v.item()) for q, v in zip(quantiles, abs_q_vals)},
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
    prefix: str,
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
    plt.title(f"{prefix} total objective per iteration")
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
    plt.title(f"{prefix} objective components per iteration")
    plt.legend()
    plt.grid(True, alpha=0.3)
    comp_path = output_dir / f"{prefix}_objective_components.png"
    plt.tight_layout()
    plt.savefig(comp_path, dpi=160)
    plt.close()
    paths["objective_components"] = str(comp_path)
    return paths


def save_distribution_histogram(
    output_dir: Path,
    prefix: str,
    values: torch.Tensor,
    title: str,
    bins: int = 120,
) -> str:
    path = output_dir / f"{prefix}_hist.png"
    flat = values.detach().float().reshape(-1).cpu().numpy()
    plt.figure(figsize=(8, 5))
    plt.hist(flat, bins=bins, color="#2a6f97", alpha=0.85)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def discrete_value_counts(values: torch.Tensor) -> Dict[str, int]:
    flat = values.detach().float().reshape(-1).cpu()
    uniq, counts = torch.unique(flat, sorted=True, return_counts=True)
    return {f"{float(v):g}": int(c) for v, c in zip(uniq.tolist(), counts.tolist())}


def save_discrete_histogram(
    output_dir: Path,
    prefix: str,
    values: torch.Tensor,
    title: str,
) -> str:
    path = output_dir / f"{prefix}_discrete_hist.png"
    flat = values.detach().float().reshape(-1).cpu()
    uniq, counts = torch.unique(flat, sorted=True, return_counts=True)
    uniq_np = uniq.numpy()
    counts_np = counts.numpy()
    plt.figure(figsize=(8, 5))
    plt.bar(uniq_np, counts_np, width=0.8, color="#c95d63", alpha=0.9)
    plt.title(title)
    plt.xlabel("quantized value")
    plt.ylabel("count")
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
    bins: int = 120,
) -> str:
    path = output_dir / f"{prefix}_quantized_overlay.png"
    z_cont_np = z_continuous.detach().float().reshape(-1).cpu().numpy()
    z_code_flat = z_codes.detach().float().reshape(-1).cpu()
    level_values = torch.unique(z_code_flat, sorted=True)

    plt.figure(figsize=(9, 5.5))
    plt.hist(z_cont_np, bins=bins, color="#d9d9d9", alpha=0.8, label="continuous z_tilde")

    cmap = plt.cm.get_cmap("tab10", max(int(level_values.numel()), 1))
    z_cont_flat = z_continuous.detach().float().reshape(-1).cpu()
    for idx, level in enumerate(level_values.tolist()):
        mask = torch.isclose(z_code_flat, torch.tensor(level, dtype=z_code_flat.dtype))
        if not torch.any(mask):
            continue
        vals = z_cont_flat[mask].numpy()
        plt.hist(
            vals,
            bins=bins,
            alpha=0.55,
            color=cmap(idx),
            label=f"code -> {level:g}",
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
    state: "QuantizationState",
    output_dir: Path,
    logger: logging.Logger,
    eps: float,
) -> Dict[str, object]:
    U = state.U.to(dtype=X_calib.dtype, device=X_calib.device)
    lambda_x = state.lambda_x.to(dtype=X_calib.dtype, device=X_calib.device)
    u_tx = U.T @ X_calib
    safe_lambda_x = torch.where(
        lambda_x.abs() < eps,
        torch.full_like(lambda_x, eps),
        lambda_x,
    )
    z_tilde_cont = u_tx / safe_lambda_x.unsqueeze(1)
    lambda_u_tx = lambda_x.unsqueeze(1) * u_tx
    z_tilde_quantized, z_tilde_scale = uniform_quantize_maxabs(
        z_tilde_cont,
        bits=state.latent_bits,
        eps=eps,
    )
    z_tilde_codes, _ = uniform_quantize_maxabs_codes(
        z_tilde_cont,
        bits=state.latent_bits,
        eps=eps,
    )

    plots = {
        "u_tx_hist": save_distribution_histogram(output_dir, f"{layer_name}_u_tx", u_tx, f"{layer_name} U^T X"),
        "z_tilde_cont_hist": save_distribution_histogram(
            output_dir,
            f"{layer_name}_z_tilde_cont",
            z_tilde_cont,
            f"{layer_name} z_tilde = (U^T X) / lambda_x",
        ),
        "z_tilde_quantized_hist": save_discrete_histogram(
            output_dir,
            f"{layer_name}_z_tilde_codes",
            z_tilde_codes,
            f"{layer_name} quantized codes clip(round(z/s))",
        ),
        "lambda_u_tx_hist": save_distribution_histogram(
            output_dir,
            f"{layer_name}_lambda_u_tx",
            lambda_u_tx,
            f"{layer_name} lambda_x * (U^T X)",
        ),
        "z_tilde_quantized_overlay": save_quantized_overlay_histogram(
            output_dir,
            f"{layer_name}_z_tilde",
            z_tilde_cont,
            z_tilde_codes,
            f"{layer_name} z_tilde with quantized code assignments",
        ),
    }

    info = {
        "u_tx": distribution_stats(u_tx),
        "z_tilde_cont": distribution_stats(z_tilde_cont),
        "z_tilde_codes": distribution_stats(z_tilde_codes),
        "z_tilde_code_value_counts": discrete_value_counts(z_tilde_codes),
        "z_tilde_quantized": distribution_stats(z_tilde_quantized),
        "z_tilde_quantized_value_counts": discrete_value_counts(z_tilde_quantized),
        "z_tilde_quant_scale": float(z_tilde_scale.item()),
        "latent_bits": int(state.latent_bits),
        "lambda_u_tx": distribution_stats(lambda_u_tx),
        "plots": plots,
    }
    logger.info(
        "连续 latent 分布统计完成 | layer=%s z_tilde_mean=%.6f z_tilde_std=%.6f z_tilde_min=%.6f z_tilde_max=%.6f",
        layer_name,
        info["z_tilde_cont"]["mean"],
        info["z_tilde_cont"]["std"],
        info["z_tilde_cont"]["min"],
        info["z_tilde_cont"]["max"],
    )
    return info



def average_metrics(metrics_by_layer: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if not metrics_by_layer:
        return {}
    keys = sorted(next(iter(metrics_by_layer.values())).keys())
    return {
        key: float(sum(layer_metrics[key] for layer_metrics in metrics_by_layer.values()) / len(metrics_by_layer))
        for key in keys
    }



def build_analysis_summary(artifacts: "ExperimentArtifacts") -> str:
    lines: List[str] = []
    lines.append("实验结果分析")
    latent_mode = artifacts.config.get("quant", {}).get("latent_mode", "discrete")
    lines.append(f"- latent_mode: {latent_mode}")
    lines.append(f"- 量化是否完成: {artifacts.quantization_completed}")
    lines.append(f"- baseline PPL: {artifacts.baseline_ppl:.6f}")
    lines.append(f"- SQ-AllBlocks-QKVO baseline PPL: {artifacts.sq_baseline_ppl:.6f}")
    lines.append(f"- Ours-AllBlocks-QKVO quantized PPL: {artifacts.quantized_ppl:.6f}")
    lines.append(f"- Ours 相对 FP 的 PPL 增量: {artifacts.quantized_ppl - artifacts.baseline_ppl:.6f}")
    lines.append(f"- SQ 相对 FP 的 PPL 增量: {artifacts.sq_baseline_ppl - artifacts.baseline_ppl:.6f}")
    lines.append("")

    lines.append("SQ 对照组各层误差")
    for layer_name, metrics in artifacts.sq_metrics.items():
        lines.append(f"- {layer_name}")
        for k, v in metrics.items():
            lines.append(f"  - {k}: {v:.6f}")
    if artifacts.sq_metrics_avg:
        lines.append("- 平均")
        for k, v in artifacts.sq_metrics_avg.items():
            lines.append(f"  - {k}: {v:.6f}")
    lines.append("")

    lines.append("Ours 各层误差")
    for layer_name, metrics in artifacts.quant_metrics.items():
        lines.append(f"- {layer_name}")
        for k, v in metrics.items():
            lines.append(f"  - {k}: {v:.6f}")
    if artifacts.quant_metrics_avg:
        lines.append("- 平均")
        for k, v in artifacts.quant_metrics_avg.items():
            lines.append(f"  - {k}: {v:.6f}")
    lines.append("")

    lines.append("各层收敛轮数")
    for layer_name, conv_iter in artifacts.convergence_iters.items():
        lines.append(f"- {layer_name}: {conv_iter}")
    lines.append("")

    lines.append("耗时（秒）")
    for k, v in artifacts.timing_info.items():
        lines.append(f"- {k}: {v:.4f}")

    lines.append("")
    lines.append("目标层")
    for k, v in artifacts.target_info.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


# ============================================================
# 量化状态
# ============================================================


@dataclass
class QuantizationState:
    U: torch.Tensor
    lambda_x: torch.Tensor
    lambda_w: torch.Tensor
    Z_x: torch.Tensor
    Z_w: torch.Tensor
    codebook: torch.Tensor
    objective_history: List[float]
    objective_x_history: List[float]
    objective_w_history: List[float]
    convergence_iter: int
    fit_time_sec: float
    latent_mode: str
    latent_bits: int

    @property
    def coeff(self) -> torch.Tensor:
        return self.lambda_x * self.lambda_w


# ============================================================
# 目标层描述
# ============================================================


@dataclass
class TargetModuleSpec:
    key: str
    block_index: int
    linear_name: str
    parent_module: nn.Module
    module: nn.Linear


# ============================================================
# 文档算法：E-step / M-step
# ============================================================


class LatticeLinearQuantizer:
    def __init__(self, config: QuantizerConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.dtype = get_torch_dtype(config.dtype)
        self.device = torch.device("cuda")
        self.codebook = torch.tensor(config.codebook, dtype=self.dtype)
        self.logger = logger

    def _pca_init(self, X: torch.Tensor) -> torch.Tensor:
        mu = X.mean(dim=1, keepdim=True)
        Xc = X - mu
        cov = (Xc @ Xc.T) / max(X.shape[1], 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        order = torch.argsort(eigvals, descending=True)
        U = eigvecs[:, order]
        return U

    def _random_init(self, X: torch.Tensor) -> torch.Tensor:
        d = X.shape[0]
        random_mat = torch.randn((d, d), dtype=X.dtype, device=X.device)
        Q, _ = torch.linalg.qr(random_mat)
        return Q

    def _latent_step(self, Data: torch.Tensor, U: torch.Tensor, lambda_diag: torch.Tensor) -> torch.Tensor:
        s = U.T @ Data
        safe_lambda = torch.where(
            lambda_diag.abs() < self.config.eps,
            torch.full_like(lambda_diag, self.config.eps),
            lambda_diag,
        )
        return s / safe_lambda.unsqueeze(1)

    def _e_step(self, Data: torch.Tensor, U: torch.Tensor, lambda_diag: torch.Tensor) -> torch.Tensor:
        z_tilde = self._latent_step(Data, U, lambda_diag)
        if getattr(self.config, "latent_mode", "discrete") == "continuous":
            return z_tilde
        z_q, _ = uniform_quantize_maxabs(
            z_tilde,
            bits=int(getattr(self.config, "latent_bits", 2)),
            eps=self.config.eps,
        )
        return z_q

    def _update_lambda(
        self,
        Data: torch.Tensor,
        U: torch.Tensor,
        Z: torch.Tensor,
        inv_norms: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        SXZ = weighted_cross(Data, inv_norms, Z)
        SZZ = weighted_gram(Z, inv_norms)
        numerator = torch.diag(U.T @ SXZ)
        denominator = torch.diag(SZZ)
        lambda_diag = numerator / (denominator + self.config.eps)
        return lambda_diag, SXZ, SZZ

    def _solve_linear_system(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
        A_reg = A + self.config.eps * eye
        try:
            return torch.linalg.solve(A_reg, b)
        except RuntimeError:
            return torch.linalg.lstsq(A_reg, b.unsqueeze(1)).solution.squeeze(1)

    def _lambda_ip_reg_terms_for_x(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        Z_x: torch.Tensor,
        Z_w: torch.Tensor,
        lambda_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        A = Z_x
        B = lambda_w.unsqueeze(1) * Z_w
        H = (A @ A.T) * (B @ B.T)
        XA = X @ A.T
        WB = W @ B.T
        g = torch.sum(XA * WB, dim=0)
        return H, g

    def _lambda_ip_reg_terms_for_w(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        Z_x: torch.Tensor,
        Z_w: torch.Tensor,
        lambda_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        A = lambda_x.unsqueeze(1) * Z_x
        B = Z_w
        H = (A @ A.T) * (B @ B.T)
        XA = X @ A.T
        WB = W @ B.T
        g = torch.sum(XA * WB, dim=0)
        return H, g

    def _update_lambdas_with_ip_reg(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        U: torch.Tensor,
        Z_x: torch.Tensor,
        Z_w: torch.Tensor,
        inv_norms_x: torch.Tensor,
        inv_norms_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        lambda_x, SXZx, SZZx = self._update_lambda(X, U, Z_x, inv_norms_x)
        lambda_w, SWZw, SZZw = self._update_lambda(W, U, Z_w, inv_norms_w)

        gamma = float(getattr(self.config, "ip_reg_gamma", 0.0))
        inner_iters = int(getattr(self.config, "ip_reg_inner_iters", 1))
        if gamma <= 0.0 or inner_iters <= 0:
            return lambda_x, lambda_w, SXZx, SWZw

        G_x = torch.diag(torch.diag(SZZx))
        h_x = torch.diag(U.T @ SXZx)
        G_w = torch.diag(torch.diag(SZZw))
        h_w = torch.diag(U.T @ SWZw)
        beta_w = float(max(self.config.beta, self.config.eps))
        ip_scale = 1.0 / max(Z_x.shape[1] * Z_w.shape[1], 1)

        for _ in range(inner_iters):
            H_x, g_x = self._lambda_ip_reg_terms_for_x(X, W, Z_x, Z_w, lambda_w)
            lambda_x = self._solve_linear_system(G_x + (gamma * ip_scale) * H_x, h_x + (gamma * ip_scale) * g_x)

            H_w, g_w = self._lambda_ip_reg_terms_for_w(X, W, Z_x, Z_w, lambda_x)
            lambda_w = self._solve_linear_system(
                beta_w * G_w + (gamma * ip_scale) * H_w,
                beta_w * h_w + (gamma * ip_scale) * g_w,
            )

        return lambda_x, lambda_w, SXZx, SWZw

    def _compute_ip_regularizer(
        self,
        X: torch.Tensor,
        W: torch.Tensor,
        Z_x: torch.Tensor,
        Z_w: torch.Tensor,
        lambda_x: torch.Tensor,
        lambda_w: torch.Tensor,
    ) -> torch.Tensor:
        gamma = float(getattr(self.config, "ip_reg_gamma", 0.0))
        if gamma <= 0.0:
            return torch.zeros((), dtype=X.dtype, device=X.device)
        target_inner = X.T @ W
        approx_inner = Z_x.T @ ((lambda_x * lambda_w).unsqueeze(1) * Z_w)
        diff = target_inner - approx_inner
        return gamma * torch.sum(diff * diff) / max(diff.numel(), 1)

    def _update_U(
        self,
        lambda_x: torch.Tensor,
        lambda_w: torch.Tensor,
        SXZx: torch.Tensor,
        SWZw: torch.Tensor,
    ) -> torch.Tensor:
        M = SXZx @ torch.diag(lambda_x) + self.config.beta * SWZw @ torch.diag(lambda_w)
        P, _, Qt = torch.linalg.svd(M, full_matrices=False)
        return P @ Qt

    def fit(self, X: torch.Tensor, W: torch.Tensor, tag: str = "") -> QuantizationState:
        fit_start = time.perf_counter()
        X = X.to(dtype=self.dtype)
        W = W.to(dtype=self.dtype)
        device = X.device
        self.device = device
        self.codebook = self.codebook.to(device)

        d, n = X.shape
        _, m = W.shape
        assert W.shape[0] == d, "X 和 W 的特征维度必须一致"

        inv_norms_x = 1.0 / (torch.sum(X * X, dim=0) + self.config.eps)
        inv_norms_w = 1.0 / (torch.sum(W * W, dim=0) + self.config.eps)

        if getattr(self.config, "error_mode", "relative") == "absolute":
            inv_norms_x = torch.ones_like(inv_norms_x)
            inv_norms_w = torch.ones_like(inv_norms_w)

        if self.logger is not None:
            self.logger.info(
                "开始拟合量化器 | tag=%s d=%d N(tokens)=%d M(out_features)=%d beta=%.4f ip_reg_gamma=%.4f ip_reg_inner_iters=%d max_iters=%d tol=%.2e codebook=%s",
                tag,
                d,
                n,
                m,
                self.config.beta,
                float(getattr(self.config, "ip_reg_gamma", 0.0)),
                int(getattr(self.config, "ip_reg_inner_iters", 1)),
                self.config.max_iters,
                self.config.tol,
                list(self.config.codebook),
            )

        if getattr(self.config, "init_mode", "pca") == "random":
            if self.logger is not None:
                self.logger.info("Using Random Orthogonal Initialization")
            U = self._random_init(X)
        else:
            if self.logger is not None:
                self.logger.info("Using PCA Initialization")
            U = self._pca_init(X)
        lambda_x = torch.ones(d, dtype=X.dtype, device=device)
        lambda_w = torch.ones(d, dtype=W.dtype, device=device)

        J_old = float("inf")
        hist_J: List[float] = []
        hist_Jx: List[float] = []
        hist_Jw: List[float] = []
        convergence_iter = self.config.max_iters

        for t in range(1, self.config.max_iters + 1):
            iter_start = time.perf_counter()

            Z_x = self._e_step(X, U, lambda_x)
            Z_w = self._e_step(W, U, lambda_w)

            lambda_x, lambda_w, SXZx, SWZw = self._update_lambdas_with_ip_reg(
                X=X,
                W=W,
                U=U,
                Z_x=Z_x,
                Z_w=Z_w,
                inv_norms_x=inv_norms_x,
                inv_norms_w=inv_norms_w,
            )

            U = self._update_U(lambda_x, lambda_w, SXZx, SWZw)

            X_hat = U @ (lambda_x.unsqueeze(1) * Z_x)
            W_hat = U @ (lambda_w.unsqueeze(1) * Z_w)
            J_x = relative_weighted_reconstruction_error(X, X_hat, inv_norms_x, average=True)
            J_w = relative_weighted_reconstruction_error(W, W_hat, inv_norms_w, average=True)
            J_ip = self._compute_ip_regularizer(X, W, Z_x, Z_w, lambda_x, lambda_w)
            J = J_x + self.config.beta * J_w + J_ip

            hist_J.append(float(J.item()))
            hist_Jx.append(float(J_x.item()))
            hist_Jw.append(float(J_w.item()))

            if math.isfinite(J_old):
                rel_change = abs(float(J.item()) - J_old) / max(1.0, abs(J_old))
            else:
                rel_change = float("inf")
            iter_time = time.perf_counter() - iter_start

            if self.logger is not None and (t == 1 or t % self.config.log_every == 0):
                self.logger.info(
                    "tag=%s iter=%03d | J=%.6f J_x=%.6f J_w=%.6f J_ip=%.6f rel_change=%.6e | "
                    "lambda_x[min,max]=[%.6f, %.6f] lambda_w[min,max]=[%.6f, %.6f] | time=%.3fs",
                    tag,
                    t,
                    float(J.item()),
                    float(J_x.item()),
                    float(J_w.item()),
                    float(J_ip.item()),
                    rel_change,
                    float(lambda_x.min().item()),
                    float(lambda_x.max().item()),
                    float(lambda_w.min().item()),
                    float(lambda_w.max().item()),
                    iter_time,
                )

            if t % self.config.convergence_check_every == 0:
                if rel_change < self.config.tol:
                    convergence_iter = t
                    if self.logger is not None:
                        self.logger.info("tag=%s 量化器在第 %d 轮收敛。", tag, t)
                    break
                J_old = float(J.item())

        fit_time_sec = time.perf_counter() - fit_start
        if self.logger is not None:
            self.logger.info("量化器拟合完成 | tag=%s convergence_iter=%d total_fit_time=%.3fs", tag, convergence_iter, fit_time_sec)

        return QuantizationState(
            U=U,
            lambda_x=lambda_x,
            lambda_w=lambda_w,
            Z_x=Z_x,
            Z_w=Z_w,
            codebook=self.codebook,
            objective_history=hist_J,
            objective_x_history=hist_Jx,
            objective_w_history=hist_Jw,
            convergence_iter=convergence_iter,
            fit_time_sec=fit_time_sec,
            latent_mode=getattr(self.config, "latent_mode", "discrete"),
            latent_bits=int(getattr(self.config, "latent_bits", 2)),
        )

    def reconstruct_X(self, X: torch.Tensor, state: QuantizationState) -> torch.Tensor:
        Z_x = self._e_step(X.to(state.U.device, dtype=state.U.dtype), state.U, state.lambda_x)
        return state.U @ (state.lambda_x.unsqueeze(1) * Z_x)

    def reconstruct_W(self, state: QuantizationState) -> torch.Tensor:
        return state.U @ (state.lambda_w.unsqueeze(1) * state.Z_w)


# ============================================================
# 通用线性层量化模块
# ============================================================


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
        self.latent_bits = state.latent_bits
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().clone())

    def _encode_x(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.U.dtype)
        s = x @ self.U
        safe_lambda_x = torch.where(
            self.lambda_x.abs() < 1e-8,
            torch.full_like(self.lambda_x, 1e-8),
            self.lambda_x,
        )
        z_cont = s / safe_lambda_x.unsqueeze(0)
        if self.latent_mode == "continuous":
            return z_cont
        z_q, _ = uniform_quantize_maxabs(z_cont, bits=self.latent_bits, eps=1e-8)
        return z_q

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape[:-1]
        d = hidden_states.shape[-1]
        x_flat = hidden_states.reshape(-1, d)
        z_x = self._encode_x(x_flat)
        output = (z_x * self.coeff.unsqueeze(0)) @ self.Z_w
        if self.bias is not None:
            output = output + self.bias.to(output.dtype).unsqueeze(0)
        output = output.to(hidden_states.dtype)
        return output.reshape(*orig_shape, self.Z_w.shape[1])


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
        x_q, _ = scalar_quantize_maxabs(x, bits=self.bits, scale=self.x_scale, eps=self.eps)
        return x_q

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape[:-1]
        d = hidden_states.shape[-1]
        x_flat = hidden_states.reshape(-1, d)
        x_q = self._quantize_x(x_flat)
        matmul_dtype = self.weight_quantized.dtype
        x_q = x_q.to(matmul_dtype)
        output = x_q @ self.weight_quantized
        if self.bias is not None:
            output = output + self.bias.to(output.dtype).unsqueeze(0)
        output = output.to(hidden_states.dtype)
        return output.reshape(*orig_shape, self.weight_quantized.shape[1])


# ============================================================
# 数据与模型
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
# 误差指标
# ============================================================


@torch.no_grad()
def compute_linear_relative_error(
    X: torch.Tensor,
    W: torch.Tensor,
    state: QuantizationState,
    chunk_tokens: int = 128,
) -> float:
    X = X.to(device=state.U.device, dtype=state.U.dtype)
    W = W.to(device=state.U.device, dtype=state.U.dtype)

    denom = 0.0
    numer = 0.0
    coeff = state.coeff

    for start in range(0, X.shape[1], chunk_tokens):
        end = min(start + chunk_tokens, X.shape[1])
        X_chunk = X[:, start:end]
        Y_true = X_chunk.T @ W

        s = state.U.T @ X_chunk
        safe_lambda = torch.where(
            state.lambda_x.abs() < 1e-8,
            torch.full_like(state.lambda_x, 1e-8),
            state.lambda_x,
        )
        z_cont = s / safe_lambda.unsqueeze(1)
        Z_x, _ = uniform_quantize_maxabs(z_cont, bits=state.latent_bits, eps=1e-8)
        Y_hat = (Z_x.T * coeff.unsqueeze(0)) @ state.Z_w

        diff = Y_true - Y_hat
        numer += float(torch.sum(diff * diff).item())
        denom += float(torch.sum(Y_true * Y_true).item())

    return numer / max(denom, 1e-12)



def compute_reconstruction_errors(
    X: torch.Tensor,
    W: torch.Tensor,
    state: QuantizationState,
    quantizer: LatticeLinearQuantizer,
    logger: logging.Logger,
    layer_name: str,
) -> Dict[str, float]:
    logger.info("开始计算重建/近似误差 | layer=%s", layer_name)
    t0 = time.perf_counter()

    device = state.U.device
    dtype = state.U.dtype
    X = X.to(device=device, dtype=dtype)
    W = W.to(device=device, dtype=dtype)

    X_hat = quantizer.reconstruct_X(X, state)
    W_hat = quantizer.reconstruct_W(state)

    err_x = float(torch.sum((X - X_hat) ** 2).item() / max(torch.sum(X ** 2).item(), 1e-12))
    err_w = float(torch.sum((W - W_hat) ** 2).item() / max(torch.sum(W ** 2).item(), 1e-12))
    err_linear = float(compute_linear_relative_error(X, W, state, chunk_tokens=128))

    elapsed = time.perf_counter() - t0
    logger.info(
        "误差计算完成 | layer=%s rel_recon_error_x=%.6f rel_recon_error_w=%.6f rel_linear_error=%.6f elapsed=%.3fs",
        layer_name,
        err_x,
        err_w,
        err_linear,
        elapsed,
    )
    return {
        "rel_recon_error_x": err_x,
        "rel_recon_error_w": err_w,
        "rel_linear_error": err_linear,
    }


@torch.no_grad()
def compute_sq_xw_metrics(
    X: torch.Tensor,
    W: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bits: int,
    logger: logging.Logger,
    layer_name: str,
) -> Dict[str, float]:
    logger.info("开始计算 SQ-XW 对照组误差 | layer=%s", layer_name)
    t0 = time.perf_counter()

    device = W.device
    dtype = W.dtype
    X = X.to(device=device, dtype=dtype)
    W = W.to(device=device, dtype=dtype)

    X_q, _ = scalar_quantize_maxabs(X, bits=bits, scale=x_scale.to(device=device, dtype=dtype))
    W_q, _ = scalar_quantize_maxabs(W, bits=bits, scale=w_scale.to(device=device, dtype=dtype))

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
    elapsed = time.perf_counter() - t0
    logger.info(
        "SQ-XW 误差计算完成 | layer=%s rel_recon_error_x=%.6f rel_recon_error_w=%.6f rel_linear_error=%.6f elapsed=%.3fs",
        layer_name,
        err_x,
        err_w,
        err_linear,
        elapsed,
    )
    return {
        "rel_recon_error_x": err_x,
        "rel_recon_error_w": err_w,
        "rel_linear_error": err_linear,
    }


# ============================================================
# PPL 评测
# ============================================================


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

    nlls = []
    prev_end_loc = 0
    total_target_tokens = 0
    num_windows = 0

    for begin_loc in range(0, input_ids.size(0), stride):
        end_loc = min(begin_loc + max_length, input_ids.size(0))
        trg_len = end_loc - prev_end_loc
        input_ids_chunk = input_ids[begin_loc:end_loc].unsqueeze(0)
        target_ids = input_ids_chunk.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(input_ids=input_ids_chunk, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        total_target_tokens += trg_len
        prev_end_loc = end_loc
        num_windows += 1
        if end_loc == input_ids.size(0):
            break

    total_nll = torch.stack(nlls).sum()
    avg_nll = total_nll / total_target_tokens
    ppl = torch.exp(avg_nll)
    elapsed = time.perf_counter() - t0
    stats = {
        "elapsed_sec": elapsed,
        "num_windows": float(num_windows),
        "num_eval_tokens": float(input_ids.numel()),
        "num_target_tokens": float(total_target_tokens),
        "avg_nll": float(avg_nll.item()),
        "total_nll": float(total_nll.item()),
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
# 模块替换工具
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
# 实验主流程
# ============================================================


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



def build_quantized_target_modules(
    target_specs: Dict[str, TargetModuleSpec],
    X_calib_by_layer: Dict[str, torch.Tensor],
    quantizer: LatticeLinearQuantizer,
    logger: logging.Logger,
    output_dir: Path,
    device: str,
) -> Tuple[
    Dict[str, nn.Module],
    Dict[str, QuantizationState],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, object]],
    Dict[str, Dict[str, object]],
]:
    logger.info("开始为所有目标 transformer blocks 的 Q/K/V/O 线性层构建 Ours 量化器。")
    quantized_modules: Dict[str, nn.Module] = {}
    states: Dict[str, QuantizationState] = {}
    metrics_by_layer: Dict[str, Dict[str, float]] = {}
    tensor_info: Dict[str, Dict[str, object]] = {}
    latent_distribution_info: Dict[str, Dict[str, object]] = {}

    for layer_name, spec in target_specs.items():
        module = spec.module
        dtype = get_torch_dtype(quantizer.config.dtype)
        X_calib = X_calib_by_layer[layer_name].to(device=device, dtype=dtype)
        W = module.weight.detach().T.to(device=device, dtype=dtype)
        bias = None if module.bias is None else module.bias.detach().to(device=device, dtype=get_torch_dtype(quantizer.config.dtype))

        log_tensor_stats(logger, f"{layer_name}.X_calib", X_calib)
        log_tensor_stats(logger, f"{layer_name}.W", W)

        state = quantizer.fit(X_calib, W, tag=layer_name)
        metrics = compute_reconstruction_errors(X_calib, W, state, quantizer, logger, layer_name=layer_name)
        quantized_module = QuantizedLinear(state, bias=bias)
        quantized_module.to(device)

        quantized_modules[layer_name] = quantized_module
        states[layer_name] = state
        metrics_by_layer[layer_name] = metrics

        tensor_info[f"{layer_name}.X_calib"] = tensor_stats(X_calib)
        tensor_info[f"{layer_name}.W"] = tensor_stats(W)
        tensor_info[f"{layer_name}.U"] = tensor_stats(state.U)
        tensor_info[f"{layer_name}.lambda_x"] = tensor_stats(state.lambda_x)
        tensor_info[f"{layer_name}.lambda_w"] = tensor_stats(state.lambda_w)
        tensor_info[f"{layer_name}.Z_w"] = tensor_stats(state.Z_w)
        latent_distribution_info[layer_name] = collect_latent_distribution_info(
            layer_name=layer_name,
            X_calib=X_calib,
            state=state,
            output_dir=output_dir,
            logger=logger,
            eps=quantizer.config.eps,
        )
        if bias is not None:
            tensor_info[f"{layer_name}.bias"] = tensor_stats(bias)

    logger.info("Ours 量化模块构建完成。")
    return quantized_modules, states, metrics_by_layer, tensor_info, latent_distribution_info



def build_sq_target_modules(
    target_specs: Dict[str, TargetModuleSpec],
    X_calib_by_layer: Dict[str, torch.Tensor],
    quant_config: QuantizerConfig,
    logger: logging.Logger,
    device: str,
) -> Tuple[
    Dict[str, nn.Module],
    int,
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, object]],
]:
    logger.info("开始为所有目标 transformer blocks 的 Q/K/V/O 线性层构建 SQ-XW 对照组。")
    sq_modules: Dict[str, nn.Module] = {}
    sq_metrics_by_layer: Dict[str, Dict[str, float]] = {}
    sq_tensor_info: Dict[str, Dict[str, object]] = {}

    bits = infer_sq_bitwidth_from_codebook(quant_config.codebook)
    dtype = get_torch_dtype(quant_config.dtype)

    for layer_name, spec in target_specs.items():
        module = spec.module
        X_calib = X_calib_by_layer[layer_name].to(dtype=dtype)
        W = module.weight.detach().T.cpu().to(dtype=dtype)
        bias = None if module.bias is None else module.bias.detach().to(device=device, dtype=dtype)

        x_scale = scalar_quant_scale_maxabs(X_calib, bits=bits, eps=quant_config.eps)
        W_q, w_scale = scalar_quantize_maxabs(W, bits=bits, eps=quant_config.eps)
        sq_metrics = compute_sq_xw_metrics(X_calib, W, x_scale, w_scale, bits=bits, logger=logger, layer_name=layer_name)

        sq_module = ScalarQuantizedXWLinear(
            weight_quantized=W_q,
            x_scale=x_scale,
            bits=bits,
            bias=bias,
            eps=quant_config.eps,
        )
        sq_module.to(device)
        sq_modules[layer_name] = sq_module
        sq_metrics_by_layer[layer_name] = sq_metrics

        sq_tensor_info[f"{layer_name}.sq_x_scale"] = tensor_stats(x_scale)
        sq_tensor_info[f"{layer_name}.sq_w_scale"] = tensor_stats(w_scale)
        sq_tensor_info[f"{layer_name}.sq_W_quantized"] = tensor_stats(W_q)
        if bias is not None:
            sq_tensor_info[f"{layer_name}.bias"] = tensor_stats(bias)

        logger.info(
            "SQ-XW 对照组构建完成 | layer=%s bits=%d x_scale=%.8f w_scale=%.8f",
            layer_name,
            bits,
            float(x_scale.item()),
            float(w_scale.item()),
        )

    return sq_modules, bits, sq_metrics_by_layer, sq_tensor_info



def run_all_blocks_qkvo_experiment(config: ExperimentConfig) -> ExperimentArtifacts:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)

    logger.info("========== 实验开始 ==========")
    logger.info("实验配置：%s", json.dumps(asdict(config), ensure_ascii=False, indent=2))
    logger.info("说明：本版本沿用 v8 的评测口径；将所有目标 transformer blocks 的 Q/K/V/O 量化，并加入独立 SQ-XW 对照组。")
    logger.info("当前 latent_mode=%s", config.quant.latent_mode)

    timing_info: Dict[str, float] = {}
    ppl_eval_info: Dict[str, Dict[str, float]] = {}
    quantization_completed = False

    model, tokenizer, model_load_time = load_model_and_tokenizer(config, logger)
    timing_info["model_load_sec"] = model_load_time

    target_specs = get_all_block_attention_targets(
        model,
        target_linear_names=config.target.target_linear_names,
        block_indices=config.target.block_indices,
    )
    target_modules = {layer_name: spec.module for layer_name, spec in target_specs.items()}
    resolved_block_indices = sorted({spec.block_index for spec in target_specs.values()})
    logger.info(
        "目标层定位完成 | num_blocks=%d block_indices=%s target_layers=%d",
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

    t0 = time.perf_counter()
    calib_input_ids = tokenize_text(calib_text, tokenizer, max_tokens=config.data.calib_num_tokens)
    timing_info["tokenize_calib_sec"] = time.perf_counter() - t0
    logger.info("校准文本 tokenization 完成 | calib_input_ids_len=%d", calib_input_ids.numel())

    t0 = time.perf_counter()
    X_calib_by_layer, collected_token_counts = collect_target_inputs(
        model=model,
        input_ids=calib_input_ids,
        target_modules=target_modules,
        max_tokens=config.data.calib_num_tokens,
        device=config.eval.device,
        logger=logger,
    )
    timing_info["collect_target_inputs_sec"] = time.perf_counter() - t0

    # ---------- 构建 SQ-XW 对照组（独立，不影响 Ours） ----------
    t0 = time.perf_counter()
    sq_modules, sq_bits, sq_metrics_by_layer, sq_tensor_info = build_sq_target_modules(
        target_specs=target_specs,
        X_calib_by_layer=X_calib_by_layer,
        quant_config=config.quant,
        logger=logger,
        device=config.eval.device,
    )
    timing_info["build_sq_baseline_sec"] = time.perf_counter() - t0

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
            tag="sq_all_blocks_qkvo",
        )
        ppl_eval_info["sq_all_blocks_qkvo"] = sq_eval_stats
        logger.info("SQ-XW 对照组（全目标层）评测完成。")
    finally:
        restore_target_modules(target_specs, original_modules)

    # ---------- 构建 Ours（与 v8 同口径） ----------
    quantizer = LatticeLinearQuantizer(config.quant, logger=logger)
    t0 = time.perf_counter()
    quantized_modules, states, quant_metrics_by_layer, tensor_info, latent_distribution_info = build_quantized_target_modules(
        target_specs=target_specs,
        X_calib_by_layer=X_calib_by_layer,
        quantizer=quantizer,
        logger=logger,
        output_dir=output_dir,
        device=config.eval.device,
    )
    timing_info["fit_quantizer_and_metrics_sec"] = time.perf_counter() - t0
    timing_info["fit_quantizer_sec_total"] = sum(state.fit_time_sec for state in states.values())

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
            tag="ours_all_blocks_qkvo",
        )
        ppl_eval_info["ours_all_blocks_qkvo"] = quantized_eval_stats
        quantization_completed = True
        logger.info("全目标层量化推理完成，quantized PPL 已得到。")
    finally:
        restore_target_modules(target_specs, original_modules)

    plot_paths: Dict[str, Dict[str, str]] = {}
    for layer_name, state in states.items():
        plot_paths[layer_name] = save_loss_plots(
            output_dir=output_dir,
            prefix=layer_name,
            objective_history=state.objective_history,
            objective_x_history=state.objective_x_history,
            objective_w_history=state.objective_w_history,
        )

    logger.info("损失曲线已保存：%s", plot_paths)

    merged_tensor_info = dict(tensor_info)
    merged_tensor_info.update(sq_tensor_info)
    merged_tensor_info["sq_bitwidth"] = {"value": sq_bits}

    convergence_iters = {layer_name: state.convergence_iter for layer_name, state in states.items()}
    objective_histories = {layer_name: state.objective_history for layer_name, state in states.items()}
    objective_x_histories = {layer_name: state.objective_x_history for layer_name, state in states.items()}
    objective_w_histories = {layer_name: state.objective_w_history for layer_name, state in states.items()}

    target_info = {
        "block_indices": resolved_block_indices,
        "num_blocks": len(resolved_block_indices),
        "target_linear_names": list(config.target.target_linear_names),
        "collected_token_counts": collected_token_counts,
        "model_name": config.data.model_name,
        "latent_mode": config.quant.latent_mode,
    }

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
        convergence_iters=convergence_iters,
        objective_histories=objective_histories,
        objective_x_histories=objective_x_histories,
        objective_w_histories=objective_w_histories,
        tensor_info=merged_tensor_info,
        latent_distribution_info=latent_distribution_info,
        timing_info=timing_info,
        ppl_eval_info=ppl_eval_info,
        plot_paths=plot_paths,
        target_info=target_info,
    )

    with open(output_dir / "all_blocks_qkvo_results.json", "w", encoding="utf-8") as f:
        json.dump(asdict(artifacts), f, ensure_ascii=False, indent=2)

    summary_text = build_analysis_summary(artifacts)
    (output_dir / "analysis_summary.txt").write_text(summary_text, encoding="utf-8")
    logger.info("结果 JSON 与分析摘要已写入输出目录。")
    logger.info("========== 实验结束 ==========")
    return artifacts


def run_block10_qproj_latent_distribution_experiment(latent_bits: int) -> ExperimentArtifacts:
    config = make_block10_qproj_latent_distribution_config(latent_bits)
    artifacts = run_all_blocks_qkvo_experiment(config)
    layer_name = "block10.q_proj"
    latent_info = artifacts.latent_distribution_info.get(layer_name, {})
    summary = {
        "output_dir": config.output_dir,
        "latent_bits": latent_bits,
        "layer_name": layer_name,
        "target_info": artifacts.target_info,
        "latent_distribution_info": latent_info,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return artifacts


def run_latent_bits_sweep() -> Dict[int, ExperimentArtifacts]:
    results: Dict[int, ExperimentArtifacts] = {}
    for latent_bits in (2, 3, 4):
        results[latent_bits] = run_block10_qproj_latent_distribution_experiment(latent_bits)
    return results


# ============================================================
# 入口
# ============================================================



def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1].lower() in {"latent-dist", "block10-qproj-latent-dist", "sweep"}:
        run_latent_bits_sweep()
        return
    run_latent_bits_sweep()


if __name__ == "__main__":
    main()
