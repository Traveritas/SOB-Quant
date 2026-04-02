from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    calib_num_tokens: int = 4096  # TODO: 后续做 sensitivity study
    eval_num_tokens: Optional[int] = None  # None 表示用完整评测 split
    tokenizer_use_fast: bool = True


@dataclass
class QuantizerConfig:
    beta: float = 1.0
    max_iters: int = 20
    tol: float = 1e-5
    convergence_check_every: int = 1  # debug 阶段按 1 轮检查；正式版可改回 5
    codebook: Tuple[float, ...] = (-3.0, -1.0, 1.0, 3.0)
    dtype: str = "float32"
    eps: float = 1e-8


@dataclass
class EvalConfig:
    stride: int = 512  # PPL sliding-window stride
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


def quantize_nearest(z_continuous: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """逐元素最近邻量化。"""
    diff = torch.abs(z_continuous.unsqueeze(-1) - codebook)
    idx = torch.argmin(diff, dim=-1)
    return codebook[idx]


def weighted_cross(X: torch.Tensor, weights: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """
    计算 X diag(weights) Z^T，避免显式构造对角矩阵。
    X: [d, n], weights: [n], Z: [d, n]
    返回: [d, d]
    """
    return (X * weights.unsqueeze(0)) @ Z.T


def weighted_gram(Z: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """计算 Z diag(weights) Z^T。"""
    return (Z * weights.unsqueeze(0)) @ Z.T


def relative_weighted_reconstruction_error(
    X: torch.Tensor,
    X_hat: torch.Tensor,
    inv_norms: torch.Tensor,
) -> torch.Tensor:
    residual = X - X_hat
    per_col_err = torch.sum(residual * residual, dim=0)
    return torch.sum(inv_norms * per_col_err)


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

    @property
    def coeff(self) -> torch.Tensor:
        return self.lambda_x * self.lambda_w


# ============================================================
# 文档算法：E-step / M-step
# ============================================================


class LatticeLMHeadQuantizer:
    """
    严格对齐当前实验约定：
    - 仅对 lm_head 使用
    - shared U, separate Lambda_x / Lambda_w
    - 离线学习 U, Lambda_x, Lambda_w, Z_w
    - 在线编码 z_x，并在码域中计算 logits
    """

    def __init__(self, config: QuantizerConfig):
        self.config = config
        self.dtype = get_torch_dtype(config.dtype)
        self.device = torch.device("cpu")
        self.codebook = torch.tensor(config.codebook, dtype=self.dtype)

    def _pca_init(self, X: torch.Tensor) -> torch.Tensor:
        # 文档约定：仅对 X 做中心化后用于 PCA 初始化 U
        mu = X.mean(dim=1, keepdim=True)
        Xc = X - mu
        cov = (Xc @ Xc.T) / max(X.shape[1], 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        order = torch.argsort(eigvals, descending=True)
        U = eigvecs[:, order]
        return U

    def _e_step(self, Data: torch.Tensor, U: torch.Tensor, lambda_diag: torch.Tensor) -> torch.Tensor:
        s = U.T @ Data
        safe_lambda = torch.where(
            lambda_diag.abs() < self.config.eps,
            torch.full_like(lambda_diag, self.config.eps),
            lambda_diag,
        )
        z_tilde = s / safe_lambda.unsqueeze(1)
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
        """
        X: [d, N]  来自 lm_head 输入 hidden states
        W: [d, M]  lm_head 权重按列组织
        """
        X = X.to(dtype=self.dtype)
        W = W.to(dtype=self.dtype)
        device = X.device
        self.device = device
        self.codebook = self.codebook.to(device)

        d, _ = X.shape
        assert W.shape[0] == d, "X 和 W 的特征维度必须一致"

        inv_norms_x = 1.0 / (torch.sum(X * X, dim=0) + self.config.eps)
        inv_norms_w = 1.0 / (torch.sum(W * W, dim=0) + self.config.eps)

        U = self._pca_init(X)
        lambda_x = torch.ones(d, dtype=X.dtype, device=device)
        lambda_w = torch.ones(d, dtype=W.dtype, device=device)

        J_old = float("inf")
        hist_J: List[float] = []
        hist_Jx: List[float] = []
        hist_Jw: List[float] = []
        convergence_iter = self.config.max_iters

        for t in range(1, self.config.max_iters + 1):
            # E-step
            Z_x = self._e_step(X, U, lambda_x)
            Z_w = self._e_step(W, U, lambda_w)

            # M-step: Lambda
            lambda_x, SXZx, _ = self._update_lambda(X, U, Z_x, inv_norms_x)
            lambda_w, SWZw, _ = self._update_lambda(W, U, Z_w, inv_norms_w)

            # M-step: U
            U = self._update_U(lambda_x, lambda_w, SXZx, SWZw)

            # 记录目标函数
            X_hat = U @ (lambda_x.unsqueeze(1) * Z_x)
            W_hat = U @ (lambda_w.unsqueeze(1) * Z_w)
            J_x = relative_weighted_reconstruction_error(X, X_hat, inv_norms_x)
            J_w = relative_weighted_reconstruction_error(W, W_hat, inv_norms_w)
            J = J_x + self.config.beta * J_w

            hist_J.append(float(J.item()))
            hist_Jx.append(float(J_x.item()))
            hist_Jw.append(float(J_w.item()))

            if t % self.config.convergence_check_every == 0:
                rel_change = abs(float(J.item()) - J_old) / max(1.0, abs(J_old))
                if rel_change < self.config.tol:
                    convergence_iter = t
                    break
                J_old = float(J.item())

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
        )

    def reconstruct_X(self, X: torch.Tensor, state: QuantizationState) -> torch.Tensor:
        Z_x = self._e_step(X.to(state.U.device, dtype=state.U.dtype), state.U, state.lambda_x)
        return state.U @ (state.lambda_x.unsqueeze(1) * Z_x)

    def reconstruct_W(self, state: QuantizationState) -> torch.Tensor:
        return state.U @ (state.lambda_w.unsqueeze(1) * state.Z_w)


# ============================================================
# 量化 lm_head
# ============================================================


class QuantizedLMHead(nn.Module):
    """
    用码域内积替代 lm_head 的浮点矩阵乘法。
    输入 hidden_states: [batch, seq, hidden]
    输出 logits: [batch, seq, vocab]
    """

    def __init__(self, state: QuantizationState):
        super().__init__()
        # 使用 register_buffer，使其能随着 module.to(device) 迁移
        self.register_buffer("U", state.U.detach().clone())
        self.register_buffer("lambda_x", state.lambda_x.detach().clone())
        self.register_buffer("lambda_w", state.lambda_w.detach().clone())
        self.register_buffer("coeff", state.coeff.detach().clone())
        self.register_buffer("Z_w", state.Z_w.detach().clone())
        self.register_buffer("codebook", state.codebook.detach().clone())

    def _encode_x(self, x: torch.Tensor) -> torch.Tensor:
        # x: [n, d]
        s = x @ self.U
        safe_lambda_x = torch.where(
            self.lambda_x.abs() < 1e-8,
            torch.full_like(self.lambda_x, 1e-8),
            self.lambda_x,
        )
        z_cont = s / safe_lambda_x.unsqueeze(0)
        return quantize_nearest(z_cont, self.codebook)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape[:-1]
        d = hidden_states.shape[-1]
        x_flat = hidden_states.reshape(-1, d)

        z_x = self._encode_x(x_flat)
        # logits[n, v] = sum_i coeff[i] * z_x[n, i] * Z_w[i, v]
        logits = (z_x * self.coeff.unsqueeze(0)) @ self.Z_w
        return logits.reshape(*orig_shape, self.Z_w.shape[1])


# ============================================================
# 数据与模型
# ============================================================


class LMHeadInputCollector:
    """通过 lm_head forward pre-hook 收集进入 lm_head 的 hidden states。"""

    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.collected: List[torch.Tensor] = []
        self.num_tokens = 0
        self.handle = None

    def _hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        if self.num_tokens >= self.max_tokens:
            return
        hidden_states = inputs[0].detach()  # [batch, seq, hidden]
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
        X = torch.cat(self.collected, dim=0)  # [N, d]
        return X.T.contiguous()  # [d, N]


def load_model_and_tokenizer(config: ExperimentConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.data.model_name,
        use_fast=config.data.tokenizer_use_fast,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.data.model_name)
    model.eval()
    model.to(config.eval.device)
    return model, tokenizer


def load_text_split(config: ExperimentConfig, split: str) -> str:
    ds = load_dataset(config.data.dataset_name, config.data.dataset_config, split=split)
    if "text" not in ds.column_names:
        raise ValueError(f"Dataset split {split} has no 'text' column.")
    texts = [t for t in ds["text"] if isinstance(t, str) and t.strip()]
    return "\n\n".join(texts)


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
) -> torch.Tensor:
    collector = LMHeadInputCollector(max_tokens=max_tokens)
    collector.register(model.lm_head)

    try:
        max_length = getattr(model.config, "max_position_embeddings", 2048)
        with torch.no_grad():
            for start in range(0, input_ids.numel(), max_length):
                end = min(start + max_length, input_ids.numel())
                chunk = input_ids[start:end].unsqueeze(0).to(device)
                _ = model(input_ids=chunk)
                if collector.num_tokens >= max_tokens:
                    break
    finally:
        collector.remove()

    return collector.get_matrix()


# ============================================================
# 误差指标
# ============================================================


def compute_reconstruction_errors(
    X: torch.Tensor,
    W: torch.Tensor,
    state: QuantizationState,
    quantizer: LatticeLMHeadQuantizer,
) -> Dict[str, float]:
    device = state.U.device
    dtype = state.U.dtype

    X = X.to(device=device, dtype=dtype)
    W = W.to(device=device, dtype=dtype)

    X_hat = quantizer.reconstruct_X(X, state)
    W_hat = quantizer.reconstruct_W(state)

    err_x = float(torch.sum((X - X_hat) ** 2).item() / max(torch.sum(X ** 2).item(), 1e-12))
    err_w = float(torch.sum((W - W_hat) ** 2).item() / max(torch.sum(W ** 2).item(), 1e-12))
    err_ip = float(compute_logits_relative_error(X, W, state, chunk_tokens=128))

    return {
        "rel_recon_error_x": err_x,
        "rel_recon_error_w": err_w,
        "rel_logits_error": err_ip,
    }


@torch.no_grad()
def compute_logits_relative_error(
    X: torch.Tensor,
    W: torch.Tensor,
    state: QuantizationState,
    chunk_tokens: int = 128,
) -> float:
    """
    计算 ||X^T W - L_hat||_F^2 / ||X^T W||_F^2
    X: [d, N], W: [d, M]
    """
    device = state.U.device
    dtype = state.U.dtype
    X = X.to(device=device, dtype=dtype)
    W = W.to(device=device, dtype=dtype)

    denom = 0.0
    numer = 0.0
    coeff = state.coeff

    for start in range(0, X.shape[1], chunk_tokens):
        end = min(start + chunk_tokens, X.shape[1])
        X_chunk = X[:, start:end]  # [d, n]

        # 真值 logits
        L_true = X_chunk.T @ W  # [n, M]

        # 码域近似 logits
        s = state.U.T @ X_chunk
        safe_lambda = torch.where(
            state.lambda_x.abs() < 1e-8,
            torch.full_like(state.lambda_x, 1e-8),
            state.lambda_x,
        )
        z_cont = s / safe_lambda.unsqueeze(1)
        Z_x = quantize_nearest(z_cont, state.codebook)
        L_hat = (Z_x.T * coeff.unsqueeze(0)) @ state.Z_w

        diff = L_true - L_hat
        numer += float(torch.sum(diff * diff).item())
        denom += float(torch.sum(L_true * L_true).item())

    return numer / max(denom, 1e-12)


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
) -> float:
    """
    使用 sliding-window 方式评测固定上下文模型的 PPL。
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"][0]
    if max_eval_tokens is not None:
        input_ids = input_ids[:max_eval_tokens]

    input_ids = input_ids.to(device)
    max_length = getattr(model.config, "max_position_embeddings", 2048)

    nlls = []
    prev_end_loc = 0
    total_target_tokens = 0

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
        if end_loc == input_ids.size(0):
            break

    ppl = torch.exp(torch.stack(nlls).sum() / total_target_tokens)
    return float(ppl.item())


# ============================================================
# 实验主流程
# ============================================================


@dataclass
class ExperimentArtifacts:
    config: Dict
    baseline_ppl: float
    quantized_ppl: float
    quant_metrics: Dict[str, float]
    convergence_iter: int
    objective_history: List[float]
    objective_x_history: List[float]
    objective_w_history: List[float]


def build_quantized_lm_head_from_model(
    model,
    quantizer: LatticeLMHeadQuantizer,
    X_calib: torch.Tensor,
) -> Tuple[nn.Module, QuantizationState, Dict[str, float]]:
    W = model.lm_head.weight.detach().T.cpu()  # [hidden, vocab]
    state = quantizer.fit(X_calib, W)
    metrics = compute_reconstruction_errors(X_calib, W, state, quantizer)
    quantized_head = QuantizedLMHead(state)
    quantized_head.to(next(model.parameters()).device)
    return quantized_head, state, metrics


def run_stage1_lmhead_experiment(config: ExperimentConfig) -> ExperimentArtifacts:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(config)

    # 1) 读取文本
    calib_text = load_text_split(config, config.data.calib_split)
    eval_text = load_text_split(config, config.data.eval_split)

    # 2) baseline PPL
    baseline_ppl = evaluate_perplexity_sliding_window(
        model=model,
        tokenizer=tokenizer,
        text=eval_text,
        device=config.eval.device,
        stride=config.eval.stride,
        max_eval_tokens=config.data.eval_num_tokens,
    )

    # 3) 收集进入 lm_head 的 hidden states，形成 X
    calib_input_ids = tokenize_text(calib_text, tokenizer, max_tokens=config.data.calib_num_tokens)
    X_calib = collect_lm_head_inputs(
        model=model,
        input_ids=calib_input_ids,
        max_tokens=config.data.calib_num_tokens,
        device=config.eval.device,
    )

    # 4) 学习 U, Lambda_x, Lambda_w, Z_w
    quantizer = LatticeLMHeadQuantizer(config.quant)
    quantized_lm_head, state, metrics = build_quantized_lm_head_from_model(model, quantizer, X_calib)

    # 5) 替换 lm_head，走真实码域推理路径
    original_lm_head = model.lm_head
    model.lm_head = quantized_lm_head

    try:
        quantized_ppl = evaluate_perplexity_sliding_window(
            model=model,
            tokenizer=tokenizer,
            text=eval_text,
            device=config.eval.device,
            stride=config.eval.stride,
            max_eval_tokens=config.data.eval_num_tokens,
        )
    finally:
        model.lm_head = original_lm_head

    artifacts = ExperimentArtifacts(
        config=asdict(config),
        baseline_ppl=baseline_ppl,
        quantized_ppl=quantized_ppl,
        quant_metrics=metrics,
        convergence_iter=state.convergence_iter,
        objective_history=state.objective_history,
        objective_x_history=state.objective_x_history,
        objective_w_history=state.objective_w_history,
    )

    with open(output_dir / "stage1_results.json", "w", encoding="utf-8") as f:
        json.dump(asdict(artifacts), f, ensure_ascii=False, indent=2)

    return artifacts


# ============================================================
# 入口
# ============================================================


def main() -> None:
    config = ExperimentConfig()
    artifacts = run_stage1_lmhead_experiment(config)
    print(json.dumps(asdict(artifacts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
