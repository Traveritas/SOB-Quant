import copy
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
    d, _ = W_numpy.shape
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
    }
    return W_approx, stats


@dataclass
class EvalResult:
    accuracy: float
    total_infer_ms: float
    avg_batch_ms: float
    avg_sample_ms: float


def evaluate_model(model, data_loader, device):
    """计算准确率与前向推理耗时。"""
    model.eval()
    correct = 0
    total = 0

    total_infer_s = 0.0
    batch_count = 0

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)

            start = time.perf_counter()
            output = model(data)
            end = time.perf_counter()

            total_infer_s += (end - start)
            batch_count += 1

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / max(total, 1)
    total_infer_ms = total_infer_s * 1000.0
    avg_batch_ms = total_infer_ms / max(batch_count, 1)
    avg_sample_ms = total_infer_ms / max(total, 1)

    return EvalResult(
        accuracy=accuracy,
        total_infer_ms=total_infer_ms,
        avg_batch_ms=avg_batch_ms,
        avg_sample_ms=avg_sample_ms,
    )


def model_size_mb(model):
    """按参数张量实际 dtype 估算模型参数体积（MB）。"""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    return total_bytes / (1024.0 * 1024.0)


def train_baseline(model, train_loader, device, epochs=1, max_train_batches=None):
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

    print("=== 1) 训练基线 Float32 模型 ===")
    baseline_model = SimpleNet().to(device)
    train_baseline(
        baseline_model,
        train_loader,
        device,
        epochs=1,
        max_train_batches=250,
    )

    print("\n=== 2) 评估基线模型 ===")
    base_eval = evaluate_model(baseline_model, test_loader, device)
    base_size_mb = model_size_mb(baseline_model)

    print(f"基线准确率: {base_eval.accuracy:.2f}%")
    print(f"基线推理总时延: {base_eval.total_infer_ms:.2f} ms")
    print(f"基线平均 batch 时延: {base_eval.avg_batch_ms:.4f} ms")
    print(f"基线平均 sample 时延: {base_eval.avg_sample_ms:.6f} ms")
    print(f"基线参数体积: {base_size_mb:.4f} MB")

    print("\n=== 3) 应用你的量化方法（fc1 + fc2） ===")
    quant_model = copy.deepcopy(baseline_model)
    layer_stats = apply_quantization_on_linear_layers(
        quant_model,
        layer_names=("fc1", "fc2"),
        max_iters=3,
    )

    for layer_name, stats in layer_stats.items():
        print(
            f"层 {layer_name}: MSE={stats['mse']:.6f}, "
            f"MAE={stats['mae']:.6f}, codebook 使用={stats['codebook_usage']}"
        )

    print("\n=== 4) 评估量化后模型 ===")
    quant_eval = evaluate_model(quant_model, test_loader, device)
    quant_size_mb = model_size_mb(quant_model)

    print(f"量化后准确率: {quant_eval.accuracy:.2f}%")
    print(f"量化后推理总时延: {quant_eval.total_infer_ms:.2f} ms")
    print(f"量化后平均 batch 时延: {quant_eval.avg_batch_ms:.4f} ms")
    print(f"量化后平均 sample 时延: {quant_eval.avg_sample_ms:.6f} ms")
    print(f"量化后参数体积: {quant_size_mb:.4f} MB")

    print("\n=== 5) 量化效果总结 ===")
    acc_drop = base_eval.accuracy - quant_eval.accuracy
    latency_change = quant_eval.avg_batch_ms - base_eval.avg_batch_ms
    size_change = quant_size_mb - base_size_mb

    print(f"精度变化(基线-量化): {acc_drop:+.2f}%")
    print(f"平均 batch 时延变化(量化-基线): {latency_change:+.4f} ms")
    print(f"参数体积变化(量化-基线): {size_change:+.4f} MB")

    print("\n说明: 当前脚本是把量化后的重建权重写回 float32 张量再测试，")
    print("因此参数体积通常不会下降。它主要用于评估你方法对精度和推理速度的影响。")


if __name__ == "__main__":
    main()
