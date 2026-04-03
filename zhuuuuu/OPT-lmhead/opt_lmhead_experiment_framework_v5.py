
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
    max_iters: int = 200
    tol: float = 1e-5
    convergence_check_every: int = 1
    codebook: Tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    dtype: str = "float32"
    eps: float = 1e-8
    log_every: int = 1
    use_separate_scales: bool = True
    scale_percentile: float = 0.995
    scale_max_samples: int = 2000000


@dataclass
class EvalConfig:
    stride: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    quant: QuantizerConfig = field(default_factory=QuantizerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output_dir: str = "./outputs_opt_lmhead_stage1-v5"
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


def quantize_nearest_scaled(
    z_continuous: torch.Tensor,
    codebook: torch.Tensor,
    scale: torch.Tensor | float,
) -> torch.Tensor:
    if not torch.is_tensor(scale):
        scale = torch.tensor(scale, dtype=z_continuous.dtype, device=z_continuous.device)
    scale = scale.to(dtype=z_continuous.dtype, device=z_continuous.device)
    safe_scale = torch.clamp(scale, min=1e-8)
    q = quantize_nearest(z_continuous / safe_scale, codebook)
    return q * safe_scale


def estimate_scale_from_percentile(
    z_continuous: torch.Tensor,
    codebook: torch.Tensor,
    percentile: float = 0.995,
    max_samples: int = 2000000,
) -> torch.Tensor:
    abs_code_max = float(torch.max(torch.abs(codebook)).item())
    if abs_code_max < 1e-8:
        return torch.tensor(1.0, dtype=z_continuous.dtype, device=z_continuous.device)

    flat = torch.abs(z_continuous.detach()).reshape(-1)
    if flat.numel() == 0:
        return torch.tensor(1.0, dtype=z_continuous.dtype, device=z_continuous.device)

    if flat.numel() > max_samples:
        perm = torch.randperm(flat.numel(), device=flat.device)[:max_samples]
        flat = flat[perm]

    q = torch.quantile(flat, percentile)
    scale = q / abs_code_max
    return torch.clamp(scale, min=1e-8)


def compute_saturation_rate(
    z_continuous: torch.Tensor,
    codebook: torch.Tensor,
    scale: torch.Tensor | float,
) -> Tuple[float, float]:
    if not torch.is_tensor(scale):
        scale = torch.tensor(scale, dtype=z_continuous.dtype, device=z_continuous.device)
    scale = scale.to(dtype=z_continuous.dtype, device=z_continuous.device)
    safe_scale = torch.clamp(scale, min=1e-8)
    normalized = z_continuous / safe_scale
    cmin = float(codebook.min().item())
    cmax = float(codebook.max().item())
    sat_min = float((normalized <= cmin).float().mean().item())
    sat_max = float((normalized >= cmax).float().mean().item())
    return sat_min, sat_max


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


def log_scalar_stats(logger: logging.Logger, name: str, value: float) -> None:
    logger.info("%s = %.8f", name, value)




def infer_bitwidth_from_codebook(codebook: Tuple[float, ...] | List[float]) -> int:
    num_levels = max(len(codebook), 2)
    return int(math.ceil(math.log2(num_levels)))


def scalar_quantize_weight_maxabs(weight: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if bits < 1:
        raise ValueError(f"bits must be >=1, got {bits}")
    max_abs = torch.max(torch.abs(weight))
    qmax = float(2 ** bits - 1)
    qmin = float(-(2 ** bits))
    if float(max_abs.item()) == 0.0:
        scale = torch.tensor(1.0, dtype=weight.dtype, device=weight.device)
    else:
        scale = max_abs / qmax
    safe_scale = torch.clamp(scale, min=1e-8)
    q = torch.round(weight / safe_scale)
    q = torch.clamp(q, qmin, qmax)
    w_q = q * safe_scale
    return w_q, safe_scale


class ScalarQuantizedLMHead(nn.Module):
    def __init__(self, weight_q: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("weight_q", weight_q.detach().clone())
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().clone())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight_q, self.bias)

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
    lines.append(f"- bitwidth: {artifacts.bitwidth}")
    lines.append(f"- baseline PPL: {artifacts.baseline_ppl:.6f}")
    lines.append(f"- SQ baseline PPL: {artifacts.sq_baseline_ppl:.6f}")
    lines.append(f"- our quantized PPL: {artifacts.quantized_ppl:.6f}")
    lines.append(f"- SQ PPL 增量: {artifacts.sq_baseline_ppl - artifacts.baseline_ppl:.6f}")
    lines.append(f"- Ours PPL 增量: {artifacts.quantized_ppl - artifacts.baseline_ppl:.6f}")
    lines.append("")
    lines.append("重建与近似误差（Ours）")
    for k, v in artifacts.quant_metrics.items():
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("重建与近似误差（SQ baseline）")
    for k, v in artifacts.sq_metrics.items():
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("关键张量")
    for name, info in artifacts.tensor_info.items():
        if "value" in info:
            lines.append(f"- {name}: value={info['value']:.6f}")
        else:
            lines.append(
                f"- {name}: shape={info['shape']}, dtype={info['dtype']}, "
                f"mean={info['mean']:.6f}, std={info['std']:.6f}, "
                f"min={info['min']:.6f}, max={info['max']:.6f}, fro={info['fro_norm']:.6f}"
            )
    lines.append("")
    lines.append("耗时（秒）")
    for k, v in artifacts.timing_info.items():
        lines.append(f"- {k}: {v:.4f}")

    if artifacts.objective_history:
        first_j = artifacts.objective_history[0]
        last_j = artifacts.objective_history[-1]
        rel_drop = (first_j - last_j) / max(abs(first_j), 1e-12)
        lines.append("")
        lines.append("损失曲线观察")
        lines.append(f"- 初始 J: {first_j:.6f}")
        lines.append(f"- 最终 J: {last_j:.6f}")
        lines.append(f"- 相对下降比例: {rel_drop:.6%}")
        is_monotone = all(
            artifacts.objective_history[i] <= artifacts.objective_history[i - 1] + 1e-12
            for i in range(1, len(artifacts.objective_history))
        )
        lines.append(f"- J 是否单调不增: {is_monotone}")
    return "\n".join(lines)


# ============================================================
# 量化状态
# ============================================================


@dataclass
class QuantizationState:
    U: torch.Tensor
    lambda_x: torch.Tensor
    lambda_w: torch.Tensor
    scale_x: torch.Tensor
    scale_w: torch.Tensor
    Z_x: torch.Tensor
    Z_w: torch.Tensor
    codebook: torch.Tensor
    objective_history: List[float]
    objective_x_history: List[float]
    objective_w_history: List[float]
    convergence_iter: int
    fit_time_sec: float

    @property
    def coeff(self) -> torch.Tensor:
        return self.lambda_x * self.lambda_w


# ============================================================
# 文档算法：E-step / M-step
# ============================================================


class LatticeLMHeadQuantizer:
    def __init__(self, config: QuantizerConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.dtype = get_torch_dtype(config.dtype)
        self.device = torch.device("cpu")
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

    def _compute_z_cont(self, Data: torch.Tensor, U: torch.Tensor, lambda_diag: torch.Tensor) -> torch.Tensor:
        s = U.T @ Data
        safe_lambda = torch.where(
            lambda_diag.abs() < self.config.eps,
            torch.full_like(lambda_diag, self.config.eps),
            lambda_diag,
        )
        return s / safe_lambda.unsqueeze(1)

    def _update_scale(self, z_cont: torch.Tensor) -> torch.Tensor:
        if not self.config.use_separate_scales:
            return torch.tensor(1.0, dtype=z_cont.dtype, device=z_cont.device)
        return estimate_scale_from_percentile(
            z_continuous=z_cont,
            codebook=self.codebook.to(z_cont.device),
            percentile=self.config.scale_percentile,
            max_samples=self.config.scale_max_samples,
        )

    def _e_step(
        self,
        Data: torch.Tensor,
        U: torch.Tensor,
        lambda_diag: torch.Tensor,
        scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_tilde = self._compute_z_cont(Data, U, lambda_diag)
        z_quant = quantize_nearest_scaled(z_tilde, self.codebook.to(Data.device), scale)
        return z_quant, z_tilde

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

    def fit(self, X: torch.Tensor, W: torch.Tensor) -> QuantizationState:
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

        if self.logger is not None:
            self.logger.info(
                "开始拟合量化器 | d=%d N(tokens)=%d M(vocab)=%d beta=%.4f max_iters=%d tol=%.2e codebook=%s",
                d, n, m, self.config.beta, self.config.max_iters, self.config.tol, list(self.config.codebook)
            )

        U = self._pca_init(X)
        lambda_x = torch.ones(d, dtype=X.dtype, device=device)
        lambda_w = torch.ones(d, dtype=W.dtype, device=device)
        scale_x = torch.tensor(1.0, dtype=X.dtype, device=device)
        scale_w = torch.tensor(1.0, dtype=W.dtype, device=device)

        J_old = float("inf")
        hist_J: List[float] = []
        hist_Jx: List[float] = []
        hist_Jw: List[float] = []
        convergence_iter = self.config.max_iters

        for t in range(1, self.config.max_iters + 1):
            iter_start = time.perf_counter()

            z_cont_x_pre = self._compute_z_cont(X, U, lambda_x)
            z_cont_w_pre = self._compute_z_cont(W, U, lambda_w)
            scale_x = self._update_scale(z_cont_x_pre)
            scale_w = self._update_scale(z_cont_w_pre)

            Z_x, z_cont_x = self._e_step(X, U, lambda_x, scale_x)
            Z_w, z_cont_w = self._e_step(W, U, lambda_w, scale_w)

            lambda_x, SXZx, _ = self._update_lambda(X, U, Z_x, inv_norms_x)
            lambda_w, SWZw, _ = self._update_lambda(W, U, Z_w, inv_norms_w)

            U = self._update_U(lambda_x, lambda_w, SXZx, SWZw)

            X_hat = U @ (lambda_x.unsqueeze(1) * Z_x)
            W_hat = U @ (lambda_w.unsqueeze(1) * Z_w)
            J_x = relative_weighted_reconstruction_error(X, X_hat, inv_norms_x, average=True)
            J_w = relative_weighted_reconstruction_error(W, W_hat, inv_norms_w, average=True)
            J = J_x + self.config.beta * J_w

            hist_J.append(float(J.item()))
            hist_Jx.append(float(J_x.item()))
            hist_Jw.append(float(J_w.item()))

            rel_change = abs(float(J.item()) - J_old) / max(1.0, abs(J_old))
            iter_time = time.perf_counter() - iter_start
            sat_x_min, sat_x_max = compute_saturation_rate(z_cont_x, self.codebook, scale_x)
            sat_w_min, sat_w_max = compute_saturation_rate(z_cont_w, self.codebook, scale_w)

            if self.logger is not None and (t == 1 or t % self.config.log_every == 0):
                self.logger.info(
                    "iter=%03d | J=%.6f J_x=%.6f J_w=%.6f rel_change=%.6e | "
                    "scale_x=%.6f scale_w=%.6f | sat_x[min,max]=[%.4f, %.4f] sat_w[min,max]=[%.4f, %.4f] | "
                    "lambda_x[min,max]=[%.6f, %.6f] lambda_w[min,max]=[%.6f, %.6f] | time=%.3fs",
                    t,
                    float(J.item()),
                    float(J_x.item()),
                    float(J_w.item()),
                    rel_change,
                    float(scale_x.item()),
                    float(scale_w.item()),
                    sat_x_min,
                    sat_x_max,
                    sat_w_min,
                    sat_w_max,
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
                        self.logger.info("量化器在第 %d 轮收敛。", t)
                    break
                J_old = float(J.item())

        fit_time_sec = time.perf_counter() - fit_start
        z_cont_x_final = self._compute_z_cont(X, U, lambda_x)
        z_cont_w_final = self._compute_z_cont(W, U, lambda_w)
        scale_x = self._update_scale(z_cont_x_final)
        scale_w = self._update_scale(z_cont_w_final)
        Z_x = quantize_nearest_scaled(z_cont_x_final, self.codebook, scale_x)
        Z_w = quantize_nearest_scaled(z_cont_w_final, self.codebook, scale_w)
        if self.logger is not None:
            log_tensor_stats(self.logger, "z_cont_x", z_cont_x_final)
            log_tensor_stats(self.logger, "z_cont_w", z_cont_w_final)
            sat_x_min, sat_x_max = compute_saturation_rate(z_cont_x_final, self.codebook, scale_x)
            sat_w_min, sat_w_max = compute_saturation_rate(z_cont_w_final, self.codebook, scale_w)
            self.logger.info(
                "final scales | scale_x=%.6f scale_w=%.6f | sat_x[min,max]=[%.4f, %.4f] sat_w[min,max]=[%.4f, %.4f]",
                float(scale_x.item()),
                float(scale_w.item()),
                sat_x_min, sat_x_max, sat_w_min, sat_w_max,
            )
            self.logger.info("量化器拟合完成 | convergence_iter=%d total_fit_time=%.3fs", convergence_iter, fit_time_sec)

        return QuantizationState(
            U=U,
            lambda_x=lambda_x,
            lambda_w=lambda_w,
            scale_x=scale_x,
            scale_w=scale_w,
            Z_x=Z_x,
            Z_w=Z_w,
            codebook=self.codebook,
            objective_history=hist_J,
            objective_x_history=hist_Jx,
            objective_w_history=hist_Jw,
            convergence_iter=convergence_iter,
            fit_time_sec=fit_time_sec,
        )

    def reconstruct_X(self, X: torch.Tensor, state: QuantizationState) -> torch.Tensor:
        X = X.to(state.U.device, dtype=state.U.dtype)
        z_cont = self._compute_z_cont(X, state.U, state.lambda_x)
        Z_x = quantize_nearest_scaled(z_cont, state.codebook, state.scale_x)
        return state.U @ (state.lambda_x.unsqueeze(1) * Z_x)

    def reconstruct_W(self, state: QuantizationState) -> torch.Tensor:
        return state.U @ (state.lambda_w.unsqueeze(1) * state.Z_w)


# ============================================================
# 量化 lm_head
# ============================================================


class QuantizedLMHead(nn.Module):
    def __init__(self, state: QuantizationState):
        super().__init__()
        self.register_buffer("U", state.U.detach().clone())
        self.register_buffer("lambda_x", state.lambda_x.detach().clone())
        self.register_buffer("lambda_w", state.lambda_w.detach().clone())
        self.register_buffer("scale_x", state.scale_x.detach().clone())
        self.register_buffer("scale_w", state.scale_w.detach().clone())
        self.register_buffer("coeff", state.coeff.detach().clone())
        self.register_buffer("Z_w", state.Z_w.detach().clone())
        self.register_buffer("codebook", state.codebook.detach().clone())

    def _encode_x(self, x: torch.Tensor) -> torch.Tensor:
        s = x @ self.U
        safe_lambda_x = torch.where(
            self.lambda_x.abs() < 1e-8,
            torch.full_like(self.lambda_x, 1e-8),
            self.lambda_x,
        )
        z_cont = s / safe_lambda_x.unsqueeze(0)
        return quantize_nearest_scaled(z_cont, self.codebook, self.scale_x)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape[:-1]
        d = hidden_states.shape[-1]
        x_flat = hidden_states.reshape(-1, d)
        z_x = self._encode_x(x_flat)
        logits = (z_x * self.coeff.unsqueeze(0)) @ self.Z_w
        return logits.reshape(*orig_shape, self.Z_w.shape[1])


# ============================================================
# 数据与模型
# ============================================================


class LMHeadInputCollector:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.collected: List[torch.Tensor] = []
        self.num_tokens = 0
        self.handle = None

    def _hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        if self.num_tokens >= self.max_tokens:
            return
        hidden_states = inputs[0].detach()
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        remaining = self.max_tokens - self.num_tokens
        if flat.shape[0] > remaining:
            flat = flat[:remaining]
        self.collected.append(flat.cpu())
        self.num_tokens += flat.shape[0]

    def register(self, lm_head: nn.Module) -> None:
        self.handle = lm_head.register_forward_pre_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def get_matrix(self) -> torch.Tensor:
        if not self.collected:
            raise RuntimeError("No lm_head inputs were collected.")
        X = torch.cat(self.collected, dim=0)
        return X.T.contiguous()


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


def collect_lm_head_inputs(
    model,
    input_ids: torch.Tensor,
    max_tokens: int,
    device: str,
    logger: logging.Logger,
) -> torch.Tensor:
    logger.info("开始收集 lm_head 输入 hidden states | requested_tokens=%d", max_tokens)
    t0 = time.perf_counter()
    collector = LMHeadInputCollector(max_tokens=max_tokens)
    collector.register(model.lm_head)

    num_forward_chunks = 0
    try:
        max_length = getattr(model.config, "max_position_embeddings", 2048)
        with torch.no_grad():
            for start in range(0, input_ids.numel(), max_length):
                end = min(start + max_length, input_ids.numel())
                chunk = input_ids[start:end].unsqueeze(0).to(device)
                _ = model(input_ids=chunk)
                num_forward_chunks += 1
                if collector.num_tokens >= max_tokens:
                    break
    finally:
        collector.remove()

    X = collector.get_matrix()
    elapsed = time.perf_counter() - t0
    logger.info(
        "lm_head 输入收集完成 | collected_tokens=%d forward_chunks=%d elapsed=%.3fs X_shape=%s",
        collector.num_tokens,
        num_forward_chunks,
        elapsed,
        list(X.shape),
    )
    return X


# ============================================================
# 误差指标
# ============================================================


@torch.no_grad()
def compute_logits_relative_error(
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
        L_true = X_chunk.T @ W

        s = state.U.T @ X_chunk
        safe_lambda = torch.where(
            state.lambda_x.abs() < 1e-8,
            torch.full_like(state.lambda_x, 1e-8),
            state.lambda_x,
        )
        z_cont = s / safe_lambda.unsqueeze(1)
        Z_x = quantize_nearest_scaled(z_cont, state.codebook, state.scale_x)
        L_hat = (Z_x.T * coeff.unsqueeze(0)) @ state.Z_w

        diff = L_true - L_hat
        numer += float(torch.sum(diff * diff).item())
        denom += float(torch.sum(L_true * L_true).item())

    return numer / max(denom, 1e-12)


def compute_reconstruction_errors(
    X: torch.Tensor,
    W: torch.Tensor,
    state: QuantizationState,
    quantizer: LatticeLMHeadQuantizer,
    logger: logging.Logger,
) -> Dict[str, float]:
    logger.info("开始计算重建/近似误差。")
    t0 = time.perf_counter()

    device = state.U.device
    dtype = state.U.dtype
    X = X.to(device=device, dtype=dtype)
    W = W.to(device=device, dtype=dtype)

    X_hat = quantizer.reconstruct_X(X, state)
    W_hat = quantizer.reconstruct_W(state)

    err_x = float(torch.sum((X - X_hat) ** 2).item() / max(torch.sum(X ** 2).item(), 1e-12))
    err_w = float(torch.sum((W - W_hat) ** 2).item() / max(torch.sum(W ** 2).item(), 1e-12))
    err_ip = float(compute_logits_relative_error(X, W, state, chunk_tokens=128))

    elapsed = time.perf_counter() - t0
    logger.info(
        "误差计算完成 | rel_recon_error_x=%.6f rel_recon_error_w=%.6f rel_logits_error=%.6f elapsed=%.3fs",
        err_x, err_w, err_ip, elapsed
    )
    return {
        "rel_recon_error_x": err_x,
        "rel_recon_error_w": err_w,
        "rel_logits_error": err_ip,
    }


@torch.no_grad()
def compute_sq_baseline_metrics(
    X: torch.Tensor,
    W: torch.Tensor,
    weight_q: torch.Tensor,
    logger: logging.Logger,
) -> Dict[str, float]:
    logger.info("开始计算 SQ baseline 误差。")
    t0 = time.perf_counter()
    device = weight_q.device
    dtype = weight_q.dtype
    X = X.to(device=device, dtype=dtype)
    W = W.to(device=device, dtype=dtype)
    Wq_cols = weight_q.T
    err_w = float(torch.sum((W - Wq_cols) ** 2).item() / max(torch.sum(W ** 2).item(), 1e-12))
    numer = 0.0
    denom = 0.0
    chunk_tokens = 128
    for start in range(0, X.shape[1], chunk_tokens):
        end = min(start + chunk_tokens, X.shape[1])
        X_chunk = X[:, start:end]
        L_true = X_chunk.T @ W
        L_sq = X_chunk.T @ Wq_cols
        diff = L_true - L_sq
        numer += float(torch.sum(diff * diff).item())
        denom += float(torch.sum(L_true * L_true).item())
    err_ip = numer / max(denom, 1e-12)
    elapsed = time.perf_counter() - t0
    logger.info(
        "SQ 误差计算完成 | rel_recon_error_w=%.6f rel_logits_error=%.6f elapsed=%.3fs",
        err_w, err_ip, elapsed
    )
    return {
        "rel_recon_error_w": err_w,
        "rel_logits_error": err_ip,
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

    ppl = torch.exp(torch.stack(nlls).sum() / total_target_tokens)
    elapsed = time.perf_counter() - t0
    stats = {
        "elapsed_sec": elapsed,
        "num_windows": float(num_windows),
        "num_eval_tokens": float(input_ids.numel()),
        "num_target_tokens": float(total_target_tokens),
    }
    if logger is not None:
        logger.info(
            "PPL 评测完成 | tag=%s ppl=%.6f eval_tokens=%d target_tokens=%d windows=%d elapsed=%.3fs",
            tag, float(ppl.item()), input_ids.numel(), total_target_tokens, num_windows, elapsed
        )
    return float(ppl.item()), stats


# ============================================================
# 实验主流程
# ============================================================


@dataclass
class ExperimentArtifacts:
    config: Dict
    quantization_completed: bool
    bitwidth: int
    baseline_ppl: float
    sq_baseline_ppl: float
    quantized_ppl: float
    quant_metrics: Dict[str, float]
    sq_metrics: Dict[str, float]
    convergence_iter: int
    objective_history: List[float]
    objective_x_history: List[float]
    objective_w_history: List[float]
    tensor_info: Dict[str, Dict[str, object]]
    timing_info: Dict[str, float]
    ppl_eval_info: Dict[str, Dict[str, float]]
    plot_paths: Dict[str, str]


def build_quantized_lm_head_from_model(
    model,
    quantizer: LatticeLMHeadQuantizer,
    X_calib: torch.Tensor,
    logger: logging.Logger,
) -> Tuple[nn.Module, QuantizationState, Dict[str, float], Dict[str, Dict[str, object]]]:
    logger.info("开始为 lm_head 构建量化器。")
    W = model.lm_head.weight.detach().T.cpu()
    log_tensor_stats(logger, "X_calib", X_calib)
    log_tensor_stats(logger, "W_lm_head", W)

    state = quantizer.fit(X_calib, W)
    metrics = compute_reconstruction_errors(X_calib, W, state, quantizer, logger)
    quantized_head = QuantizedLMHead(state)
    quantized_head.to(next(model.parameters()).device)

    safe_lambda_x = torch.where(state.lambda_x.abs() < quantizer.config.eps, torch.full_like(state.lambda_x, quantizer.config.eps), state.lambda_x)
    safe_lambda_w = torch.where(state.lambda_w.abs() < quantizer.config.eps, torch.full_like(state.lambda_w, quantizer.config.eps), state.lambda_w)
    z_cont_x = (state.U.T @ X_calib.to(dtype=state.U.dtype, device=state.U.device)) / safe_lambda_x.unsqueeze(1)
    z_cont_w = (state.U.T @ W.to(dtype=state.U.dtype, device=state.U.device)) / safe_lambda_w.unsqueeze(1)

    log_scalar_stats(logger, "scale_x", float(state.scale_x.item()))
    log_scalar_stats(logger, "scale_w", float(state.scale_w.item()))
    log_tensor_stats(logger, "z_cont_x", z_cont_x)
    log_tensor_stats(logger, "z_cont_w", z_cont_w)

    tensor_info = {
        "X_calib": tensor_stats(X_calib),
        "W_lm_head": tensor_stats(W),
        "U": tensor_stats(state.U),
        "lambda_x": tensor_stats(state.lambda_x),
        "lambda_w": tensor_stats(state.lambda_w),
        "Z_w": tensor_stats(state.Z_w),
        "z_cont_x": tensor_stats(z_cont_x),
        "z_cont_w": tensor_stats(z_cont_w),
        "scale_x": {"value": float(state.scale_x.item())},
        "scale_w": {"value": float(state.scale_w.item())},
    }
    logger.info("量化 lm_head 构建完成。")
    return quantized_head, state, metrics, tensor_info


def build_sq_baseline_lm_head_from_model(
    model,
    X_calib: torch.Tensor,
    config: ExperimentConfig,
    logger: logging.Logger,
) -> Tuple[nn.Module, Dict[str, float], Dict[str, object]]:
    logger.info("开始构建 SQ baseline lm_head。")
    bits = infer_bitwidth_from_codebook(config.quant.codebook)
    weight = model.lm_head.weight.detach()
    bias = model.lm_head.bias.detach() if getattr(model.lm_head, "bias", None) is not None else None
    q_dtype = get_torch_dtype(config.quant.dtype)
    weight_q, scale = scalar_quantize_weight_maxabs(weight.to(dtype=q_dtype, device=weight.device), bits)
    sq_head = ScalarQuantizedLMHead(weight_q=weight_q, bias=bias)
    sq_head.to(next(model.parameters()).device)
    W_cols = weight.detach().T.cpu()
    sq_metrics = compute_sq_baseline_metrics(X_calib, W_cols, weight_q, logger)
    sq_info = {
        "bitwidth": bits,
        "sq_scale": float(scale.item()),
        "sq_weight_q": tensor_stats(weight_q),
    }
    logger.info("SQ baseline 构建完成 | bits=%d sq_scale=%.8f", bits, float(scale.item()))
    return sq_head, sq_metrics, sq_info


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
    bitwidth = infer_bitwidth_from_codebook(config.quant.codebook)
    logger.info("根据 codebook 大小推断 SQ baseline bitwidth=%d", bitwidth)

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

    quantizer = LatticeLMHeadQuantizer(config.quant, logger=logger)
    t0 = time.perf_counter()
    quantized_lm_head, state, metrics, tensor_info = build_quantized_lm_head_from_model(
        model, quantizer, X_calib, logger
    )
    timing_info["fit_quantizer_and_metrics_sec"] = time.perf_counter() - t0
    timing_info["fit_quantizer_sec"] = state.fit_time_sec

    t0 = time.perf_counter()
    sq_lm_head, sq_metrics, sq_info = build_sq_baseline_lm_head_from_model(model, X_calib, config, logger)
    timing_info["build_sq_baseline_sec"] = time.perf_counter() - t0
    tensor_info.update({
        "sq_scale": {"value": sq_info["sq_scale"]},
        "sq_weight_q": sq_info["sq_weight_q"],
    })

    original_lm_head = model.lm_head

    model.lm_head = sq_lm_head
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
        model.lm_head = original_lm_head

    model.lm_head = quantized_lm_head

    try:
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
        logger.info("量化推理完成，quantized PPL 已得到。")
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
        bitwidth=bitwidth,
        baseline_ppl=baseline_ppl,
        sq_baseline_ppl=sq_baseline_ppl,
        quantized_ppl=quantized_ppl,
        quant_metrics=metrics,
        sq_metrics=sq_metrics,
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


# ============================================================
# 入口
# ============================================================


def main() -> None:
    config = ExperimentConfig()
    artifacts = run_stage1_lmhead_experiment(config)
    print(json.dumps(asdict(artifacts), ensure_ascii=False, indent=2))
    print("\n===== 分析摘要 =====\n")
    print(build_analysis_summary(artifacts))


if __name__ == "__main__":
    main()
