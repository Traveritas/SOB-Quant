import copy
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# ==========================================
# 1) 模型定义
# ==========================================
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ==========================================
# 2) 复用你的量化核心方法（针对单个权重矩阵 W）
# ==========================================
def quantize_weight_matrix(W_numpy, max_iters=5):
    """
    对权重矩阵 W 做交替优化量化，返回重建矩阵和误差统计。
    W_numpy 形状为 (d, M)。
    """
    d, m = W_numpy.shape
    codebook = np.array([-3, -1, 1, 3], dtype=np.float64)

    U, _ = np.linalg.qr(np.random.randn(d, d))
    Lambda_w = np.diag(np.random.rand(d) + 0.1)

    inv_norms_W = 1.0 / (np.sum(W_numpy ** 2, axis=0) + 1e-8)
    D_w = np.diag(inv_norms_W)

    Z_w = None
    for _ in range(max_iters):
        # E-step: 更新 Z
        lambda_diag = np.diag(Lambda_w).copy()
        lambda_diag[lambda_diag == 0] = 1e-8
        Z_tilde = np.diag(1.0 / lambda_diag) @ U.T @ W_numpy

        diff = np.abs(Z_tilde[..., np.newaxis] - codebook)
        Z_w = codebook[np.argmin(diff, axis=-1)]

        # M-step: 更新 Lambda
        S_WZ = W_numpy @ D_w @ Z_w.T
        S_ZZ = Z_w @ D_w @ Z_w.T
        lambda_diag = np.diag(U.T @ S_WZ) / (np.diag(S_ZZ) + 1e-8)
        Lambda_w = np.diag(lambda_diag)

        # M-step: 更新 U
        M = W_numpy @ D_w @ (Lambda_w @ Z_w).T
        P, _, Qt = np.linalg.svd(M)
        U = P @ Qt

    W_approx = U @ Lambda_w @ Z_w

    mse = float(np.mean((W_numpy - W_approx) ** 2))
    mae = float(np.mean(np.abs(W_numpy - W_approx)))

    if Z_w is not None:
        unique_vals = np.unique(Z_w)
        codebook_usage = {float(v): int(np.sum(Z_w == v)) for v in unique_vals}
    else:
        codebook_usage = {}

    stats = {
        "mse": mse,
        "mae": mae,
        "codebook_usage": codebook_usage,
        "d": int(d),
        "m": int(m),
        "codebook_size": int(len(codebook)),
    }
    return W_approx, stats


@dataclass
class EvalResult:
    accuracy: float
    latency_mean_batch_ms: float
    latency_std_batch_ms: float
    latency_mean_sample_ms: float
    latency_std_sample_ms: float


def evaluate_accuracy(model, data_loader, device):
    """仅计算准确率。"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return 100.0 * correct / max(total, 1)


def benchmark_inference(
    model,
    data_loader,
    device,
    warmup_batches=5,
    benchmark_batches=20,
    repeat_runs=10,
):
    """重复测速并输出均值/标准差，降低单次测量抖动。"""
    model.eval()

    cached_batches = []
    seen_samples = 0
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            cached_batches.append(data)
            seen_samples += data.size(0)
            if len(cached_batches) >= benchmark_batches:
                break

    if not cached_batches:
        return {
            "mean_batch_ms": 0.0,
            "std_batch_ms": 0.0,
            "mean_sample_ms": 0.0,
            "std_sample_ms": 0.0,
            "total_samples": 0,
            "total_batches": 0,
        }

    warmup_steps = min(warmup_batches, len(cached_batches))
    with torch.no_grad():
        for i in range(warmup_steps):
            _ = model(cached_batches[i])

    run_avg_batch_ms = []
    run_avg_sample_ms = []
    with torch.no_grad():
        for _ in range(repeat_runs):
            start = time.perf_counter()
            for data in cached_batches:
                _ = model(data)
            end = time.perf_counter()

            run_total_ms = (end - start) * 1000.0
            run_avg_batch_ms.append(run_total_ms / len(cached_batches))
            run_avg_sample_ms.append(run_total_ms / max(seen_samples, 1))

    return {
        "mean_batch_ms": float(np.mean(run_avg_batch_ms)),
        "std_batch_ms": float(np.std(run_avg_batch_ms)),
        "mean_sample_ms": float(np.mean(run_avg_sample_ms)),
        "std_sample_ms": float(np.std(run_avg_sample_ms)),
        "total_samples": int(seen_samples),
        "total_batches": int(len(cached_batches)),
    }


def model_size_mb(model):
    """按参数张量实际 dtype 估算模型参数体积（MB）。"""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    return total_bytes / (1024.0 * 1024.0)


def model_size_bits(model):
    """按参数张量实际 dtype 估算模型参数位数。"""
    total_bits = 0
    for p in model.parameters():
        total_bits += p.nelement() * p.element_size() * 8
    return int(total_bits)


def train_baseline(model, train_loader, device, epochs=1, max_train_batches=None):
    """训练基线模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                seen = batch_idx * len(data)
                print(
                    f"Epoch {epoch + 1} | step {batch_idx:4d} | "
                    f"seen {seen:5d} | loss {loss.item():.4f}"
                )


def apply_quantization_on_linear_layers(model, layer_names=("fc1", "fc2"), max_iters=3):
    """
    按给定层名量化 Linear 权重并写回模型。
    返回每层的重建误差统计。
    """
    layer_stats = {}

    with torch.no_grad():
        for name in layer_names:
            layer = getattr(model, name)
            if not isinstance(layer, nn.Linear):
                continue

            # PyTorch Linear 权重是 (out_features, in_features)，转成 (d, M)
            W_original = layer.weight.data.cpu().numpy().T
            W_quantized, stats = quantize_weight_matrix(W_original, max_iters=max_iters)

            # 写回时再转置回来
            layer.weight.data = torch.from_numpy(W_quantized.T).to(layer.weight.data.dtype)
            layer_stats[name] = stats

    return layer_stats


def estimate_quantized_storage_bits(model, layer_stats, layer_names=("fc1", "fc2"), float_bits=32):
    """
    估算量化表示的存储位数：
    - 对量化层权重: U(d*d float) + Lambda(d float) + Z索引(d*m*2bit)
    - 码本: 全局一次 codebook_size * float_bits
    - 非量化参数与偏置: 保持原 dtype 存储
    """
    total_bits = 0
    quantized_weight_params = set()

    codebook_size = 4
    for name in layer_names:
        if name not in layer_stats:
            continue

        st = layer_stats[name]
        d = int(st["d"])
        m = int(st["m"])
        codebook_size = int(st.get("codebook_size", codebook_size))
        index_bits = int(math.ceil(math.log2(max(codebook_size, 2))))

        bits_u = d * d * float_bits
        bits_lambda = d * float_bits
        bits_z = d * m * index_bits
        total_bits += bits_u + bits_lambda + bits_z

        quantized_weight_params.add(f"{name}.weight")

    total_bits += codebook_size * float_bits

    for param_name, p in model.named_parameters():
        if param_name in quantized_weight_params:
            continue
        total_bits += p.nelement() * p.element_size() * 8

    return int(total_bits)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cpu")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("./data", train=False, download=True, transform=transform),
        batch_size=512,
        shuffle=False,
        num_workers=0,
    )

    print("=" * 70)
    print("=== 1) 训练基线 Float32 模型 ===")
    print("=" * 70)
    baseline_model = SimpleNet().to(device)
    train_baseline(
        baseline_model,
        train_loader,
        device,
        epochs=1,
        max_train_batches=250,
    )

    print("\n" + "=" * 70)
    print("=== 2) 评估基线模型（含重复测速）===")
    print("=" * 70)
    base_acc = evaluate_accuracy(baseline_model, test_loader, device)
    base_latency = benchmark_inference(
        baseline_model,
        test_loader,
        device,
        warmup_batches=5,
        benchmark_batches=20,
        repeat_runs=20,
    )
    base_size_mb = model_size_mb(baseline_model)
    base_size_bits = model_size_bits(baseline_model)

    base_eval = EvalResult(
        accuracy=base_acc,
        latency_mean_batch_ms=base_latency["mean_batch_ms"],
        latency_std_batch_ms=base_latency["std_batch_ms"],
        latency_mean_sample_ms=base_latency["mean_sample_ms"],
        latency_std_sample_ms=base_latency["std_sample_ms"],
    )

    print(f"基线准确率: {base_eval.accuracy:.2f}%")
    print(
        f"基线平均 batch 时延: {base_eval.latency_mean_batch_ms:.4f} ± "
        f"{base_eval.latency_std_batch_ms:.4f} ms"
    )
    print(
        f"基线平均 sample 时延: {base_eval.latency_mean_sample_ms:.6f} ± "
        f"{base_eval.latency_std_sample_ms:.6f} ms"
    )
    print(
        f"基线测速配置: warmup=5, batches={base_latency['total_batches']}, "
        f"repeats=20"
    )
    print(f"基线参数体积: {base_size_mb:.4f} MB ({base_size_bits} bits)")

    print("\n" + "=" * 70)
    print("=== 3) 应用你的量化方法（fc1 + fc2） ===")
    print("=" * 70)
    quant_model = copy.deepcopy(baseline_model)
    layer_stats = apply_quantization_on_linear_layers(
        quant_model,
        layer_names=("fc1", "fc2"),
        max_iters=3,
    )

    for layer_name, stats in layer_stats.items():
        print(
            f"层 {layer_name}: MSE={stats['mse']:.6f}, "
            f"MAE={stats['mae']:.6f}"
        )
        print(f"  码本使用统计: {stats['codebook_usage']}")
        print(f"  维度: d={stats['d']}, m={stats['m']}")

    print("\n" + "=" * 70)
    print("=== 4) 评估量化后模型（含重复测速）===")
    print("=" * 70)
    quant_acc = evaluate_accuracy(quant_model, test_loader, device)
    quant_latency = benchmark_inference(
        quant_model,
        test_loader,
        device,
        warmup_batches=5,
        benchmark_batches=20,
        repeat_runs=20,
    )
    quant_eval = EvalResult(
        accuracy=quant_acc,
        latency_mean_batch_ms=quant_latency["mean_batch_ms"],
        latency_std_batch_ms=quant_latency["std_batch_ms"],
        latency_mean_sample_ms=quant_latency["mean_sample_ms"],
        latency_std_sample_ms=quant_latency["std_sample_ms"],
    )
    quant_size_mb = model_size_mb(quant_model)
    estimated_quant_bits = estimate_quantized_storage_bits(
        quant_model,
        layer_stats,
        layer_names=("fc1", "fc2"),
        float_bits=32,
    )
    estimated_quant_mb = estimated_quant_bits / 8.0 / (1024.0 * 1024.0)

    print(f"量化后准确率: {quant_eval.accuracy:.2f}%")
    print(
        f"量化后平均 batch 时延: {quant_eval.latency_mean_batch_ms:.4f} ± "
        f"{quant_eval.latency_std_batch_ms:.4f} ms"
    )
    print(
        f"量化后平均 sample 时延: {quant_eval.latency_mean_sample_ms:.6f} ± "
        f"{quant_eval.latency_std_sample_ms:.6f} ms"
    )
    print(
        f"量化后测速配置: warmup=5, batches={quant_latency['total_batches']}, "
        f"repeats=20"
    )
    print(f"量化后参数体积(float32): {quant_size_mb:.4f} MB")
    print(f"量化表示估算体积(位级): {estimated_quant_mb:.4f} MB ({estimated_quant_bits} bits)")

    print("\n" + "=" * 70)
    print("=== 5) 量化效果详细总结 ===")
    print("=" * 70)
    acc_drop = base_eval.accuracy - quant_eval.accuracy
    latency_change = quant_eval.latency_mean_batch_ms - base_eval.latency_mean_batch_ms
    latency_change_pct = (latency_change / base_eval.latency_mean_batch_ms * 100) if base_eval.latency_mean_batch_ms > 0 else 0
    size_change = quant_size_mb - base_size_mb
    estimated_size_change = estimated_quant_mb - base_size_mb
    estimated_compression_ratio = base_size_bits / max(estimated_quant_bits, 1)

    print(f"【精度指标】")
    print(f"  基线vs量化: {base_eval.accuracy:.2f}% vs {quant_eval.accuracy:.2f}%")
    print(f"  精度变化: {acc_drop:+.2f}% (基线-量化)")

    print(f"\n【速度指标】")
    print(f"  基线 batch 时延: {base_eval.latency_mean_batch_ms:.4f} ms")
    print(f"  量化 batch 时延: {quant_eval.latency_mean_batch_ms:.4f} ms")
    print(f"  时延变化: {latency_change:+.4f} ms ({latency_change_pct:+.2f}%)")

    print(f"\n【存储指标】")
    print(f"  基线参数体积: {base_size_mb:.4f} MB (Float32 native)")
    print(f"  量化后直接写回: {quant_size_mb:.4f} MB (仍为 Float32)")
    print(f"  量化表示估算: {estimated_quant_mb:.4f} MB (离散索引+分解参数)")
    print(f"  存储体积变化(估算): {estimated_size_change:+.4f} MB")
    print(f"  理论压缩率(基线/量化表示): {estimated_compression_ratio:.2f}x")

    print("\n" + "=" * 70)
    print("说明")
    print("=" * 70)
    print("• 基线参数体积不变: 当前实现是把量化后的重建权重写回 float32 张量")
    print("• 位级估算体积: 反映按离散索引+分解参数(U,Lambda,Z)存储的理论体积")
    print("• 理论压缩率: 只针对量化层(fc1, fc2)，未计入输入层和输出层")
    print("• 测速结果: 已包含 warmup 排除缓存效应，多次运行求均值降低波动")


if __name__ == "__main__":
    main()
