from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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
    max_iters: int = 300
    tol: float = 1e-5
    convergence_check_every: int = 1
    codebook: Tuple[float, ...] = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
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
    output_dir: str = "./Camparison/RPTQ"
    seed: int = 42
    run_mode: str = "both"
    continuous_subdir: str = "continuous"


config = ExperimentConfig(
    target=TargetConfig(
        block_indices=(8, 9, 10, 11),
        target_linear_names=("q_proj", "k_proj", "v_proj", "out_proj"),
    )
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



def quantize_nearest(z_continuous: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    diff = torch.abs(z_continuous.unsqueeze(-1) - codebook)
    idx = torch.argmin(diff, dim=-1)
    return codebook[idx]



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
        return quantize_nearest(z_tilde, self.codebook.to(Data.device))

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
        return quantize_nearest(z_cont, self.codebook)

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
        Z_x = quantize_nearest(z_cont, state.codebook)
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
    timing_info: Dict[str, float]
    ppl_eval_info: Dict[str, Dict[str, float]]
    plot_paths: Dict[str, Dict[str, str]]
    target_info: Dict[str, object]



def build_quantized_target_modules(
    target_specs: Dict[str, TargetModuleSpec],
    X_calib_by_layer: Dict[str, torch.Tensor],
    quantizer: LatticeLinearQuantizer,
    logger: logging.Logger,
    device: str,
) -> Tuple[
    Dict[str, nn.Module],
    Dict[str, QuantizationState],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, object]],
]:
    logger.info("开始为所有目标 transformer blocks 的 Q/K/V/O 线性层构建 Ours 量化器。")
    quantized_modules: Dict[str, nn.Module] = {}
    states: Dict[str, QuantizationState] = {}
    metrics_by_layer: Dict[str, Dict[str, float]] = {}
    tensor_info: Dict[str, Dict[str, object]] = {}

    for layer_name, spec in target_specs.items():
        module = spec.module
        X_calib = X_calib_by_layer[layer_name]
        W = module.weight.detach().T.cpu().to(dtype=get_torch_dtype(quantizer.config.dtype))
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
        if bias is not None:
            tensor_info[f"{layer_name}.bias"] = tensor_stats(bias)

    logger.info("Ours 量化模块构建完成。")
    return quantized_modules, states, metrics_by_layer, tensor_info



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
    quantized_modules, states, quant_metrics_by_layer, tensor_info = build_quantized_target_modules(
        target_specs=target_specs,
        X_calib_by_layer=X_calib_by_layer,
        quantizer=quantizer,
        logger=logger,
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


# ============================================================
# 入口
# ============================================================



def main() -> None:
    config = ExperimentConfig()
    run_mode = config.run_mode.lower()
    if run_mode not in {"discrete", "continuous", "both"}:
        raise ValueError(f"Unsupported run_mode: {config.run_mode}")

    if run_mode in {"discrete", "both"}:
        discrete_artifacts = run_all_blocks_qkvo_experiment(config)
        print(json.dumps(asdict(discrete_artifacts), ensure_ascii=False, indent=2))
        print("\n===== Discrete Summary =====\n")
        print(build_analysis_summary(discrete_artifacts))

    if run_mode in {"continuous", "both"}:
        continuous_config = replace(
            config,
            quant=replace(config.quant, latent_mode="continuous"),
            output_dir=(
                config.output_dir
                if run_mode == "continuous"
                else str(Path(config.output_dir) / config.continuous_subdir)
            ),
            run_mode="continuous",
        )
        continuous_artifacts = run_all_blocks_qkvo_experiment(continuous_config)
        print("\n===== Continuous Summary =====\n")
        print(build_analysis_summary(continuous_artifacts))


if __name__ == "__main__":
    main()
