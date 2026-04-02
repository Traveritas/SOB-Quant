import copy
import math
import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

# ==========================================
# 辅助函数：最近邻量化
# ==========================================
def quantize(z_continuous, codebook):
    diff = torch.abs(z_continuous[..., None] - codebook)
    idx = torch.argmin(diff, dim=-1)
    return codebook[idx]

# ==========================================
# E-Step：更新潜在编码 Z 
# ==========================================
def e_step(Data, U, Lambda, codebook):
    lambda_diag = torch.diag(Lambda).clone()
    lambda_diag[lambda_diag == 0] = 1e-8 
    Lambda_inv = torch.diag(1.0 / lambda_diag)
    Z_tilde = Lambda_inv @ U.T @ Data 
    return quantize(Z_tilde, codebook)

# ==========================================
# M-Step：更新缩放矩阵 Lambda 
# ==========================================
def update_lambda(Data, Z, U, D):
    S_XZ = Data @ D @ Z.T
    S_ZZ = Z @ D @ Z.T
    num = torch.diag(U.T @ S_XZ)
    den = torch.diag(S_ZZ)
    lambda_diag = num / (den + 1e-8)
    return torch.diag(lambda_diag)

# ==========================================
# M-Step：更新正交矩阵 U (这里只考虑 X) 
# ==========================================
def update_U_procrustes(X, Z_x, Lambda_x, D_x):
    """
    修改点：去除了 beta 和 W 的传入。
    公式简化为: M = X @ D_x @ Y_x.T
    这代表共享矩阵 U 仅用来最好地逼近激活特征 X 的分布 
    """
    Y_x = Lambda_x @ Z_x
    M = X @ D_x @ Y_x.T 
    P, _, Qt = torch.linalg.svd(M, full_matrices=False)
    return P @ Qt

# ==========================================
# 量化核心逻辑（仅考虑 X 提取特征基底，并将其用于 W 的量化）
# ==========================================
def quantize_weight_and_activation(X_tensor, W_tensor, max_iters=50, tol=1e-5):
    """
    此版本完全由激活矩阵 X 来主导寻找正交基 U。
    权重 W 的角色变成了: 顺应由 X 提取出的 U 平面，通过调整自己的编码 Z_w 和缩放 \Lambda_w 来完成量化。
    """
    d, N = X_tensor.shape
    d_w, M = W_tensor.shape
    assert d == d_w, "Feature dimension mismatch between X and W"
    
    device = X_tensor.device
    dtype = X_tensor.dtype

    codebook = torch.tensor([-3.0, -1.0, 1.0, 3.0], dtype=dtype, device=device)

    U, _ = torch.linalg.qr(torch.randn(d, d, dtype=dtype, device=device))
    Lambda_x = torch.diag(torch.rand(d, dtype=dtype, device=device) + 0.1)
    Lambda_w = torch.diag(torch.rand(d, dtype=dtype, device=device) + 0.1)

    inv_norms_X = 1.0 / (torch.sum(X_tensor ** 2, dim=0) + 1e-8)
    D_x = torch.diag(inv_norms_X)
    
    inv_norms_W = 1.0 / (torch.sum(W_tensor ** 2, dim=0) + 1e-8)
    D_w = torch.diag(inv_norms_W)

    prev_loss_x = float('inf')
    history_x = []
    history_w = []
    
    Z_w = None
    for it in range(max_iters):
        # 1. 给定 U 和 \Lambda_x, \Lambda_w 的情况下，寻找最优量化点
        Z_x = e_step(X_tensor, U, Lambda_x, codebook)
        Z_w = e_step(W_tensor, U, Lambda_w, codebook)

        # 2. 【核心区别】: U 只根据 X 的激活分布来更新
        U = update_U_procrustes(X_tensor, Z_x, Lambda_x, D_x)
        
        # 3. 更新各个尺度的权重
        Lambda_x = update_lambda(X_tensor, Z_x, U, D_x)
        Lambda_w = update_lambda(W_tensor, Z_w, U, D_w)
        
        # 记录误差
        loss_x = torch.sum(inv_norms_X * torch.sum((X_tensor - U @ Lambda_x @ Z_x)**2, dim=0)).item()
        loss_w = torch.sum(inv_norms_W * torch.sum((W_tensor - U @ Lambda_w @ Z_w)**2, dim=0)).item()
        
        history_x.append(loss_x)
        history_w.append(loss_w)
        
        # 由于完全被 X 驱动，我们的提前停止准则也只看 loss_x 的收敛情况
        if abs(prev_loss_x - loss_x) < tol:
            print(f"      [提前终止] 迭代 {it+1}/{max_iters}, Loss_x 变化量小于 {tol}")
            break
        prev_loss_x = loss_x

    W_approx = U @ Lambda_w @ Z_w
    mse = float(torch.mean((W_tensor - W_approx) ** 2).item())
    mae = float(torch.mean(torch.abs(W_tensor - W_approx)).item())

    unique_vals, counts = torch.unique(Z_w, return_counts=True)
    codebook_usage = {float(val.item()): int(count.item()) for val, count in zip(unique_vals, counts)}

    # 图表由于不再追求 TotalLoss，我们这里使用 history_x 充当图表的基准
    stats = {
        "mse": mse,
        "mae": mae,
        "history": history_x, 
        "history_x": history_x,
        "history_w": history_w,
        "codebook_usage": codebook_usage,
        "d": int(d),
        "m": int(M),
        "N": int(N),
        "codebook_size": int(len(codebook)),
    }
    return W_approx, stats


# ==========================================
# 后续网络与评测代码，沿用针对 CIFAR10 的改造模型以及评测方案
# (你同样也可以切换回 MNIST)
# ==========================================
def create_resnet18_cifar10():
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity() 
    
    model.fc = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10),
    )
    return model


@dataclass
class EvalResult:
    accuracy: float
    latency_mean_batch_ms: float
    latency_std_batch_ms: float
    latency_mean_sample_ms: float
    latency_std_sample_ms: float


def evaluate_accuracy(model, data_loader, device):
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


def benchmark_inference(model, data_loader, device, warmup_batches=5, benchmark_batches=20, repeat_runs=10):
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
        return {"mean_batch_ms": 0.0, "std_batch_ms": 0.0, "mean_sample_ms": 0.0, "std_sample_ms": 0.0, "total_samples": 0, "total_batches": 0}

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
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    return total_bytes / (1024.0 * 1024.0)

def model_size_bits(model):
    total_bits = 0
    for p in model.parameters():
        total_bits += p.nelement() * p.element_size() * 8
    return int(total_bits)

def count_model_params(model):
    return sum(p.numel() for p in model.parameters())

def train_baseline(model, train_loader, device, epochs=5, max_train_batches=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0
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
            total_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 100 == 0:
                seen = batch_idx * len(data)
                avg_loss = total_loss / max(batch_count, 1)
                print(f"Epoch {epoch + 1:2d} | step {batch_idx:4d} | seen {seen:5d} | loss {avg_loss:.4f} | lr {scheduler.get_last_lr()[0]:.6f}")

        scheduler.step()


def collect_activation_data(model, data_loader, device, num_batches=1):
    model.eval()
    activation_data = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if name not in activation_data:
                activation_data[name] = []
            x = input[0].detach() 
            x_flat = x.reshape(-1, x.shape[-1])
            activation_data[name].append(x_flat)
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            data = data.to(device)
            _ = model(data)

    for h in hooks:
        h.remove()

    for name in activation_data:
        activation_data[name] = torch.cat(activation_data[name], dim=0).T  

    return activation_data


def apply_quantization_on_linear_layers(model, data_loader, device, max_iters=50, num_batches=1):
    print(f"  收集 {num_batches} 把 Batch 的校准数据 (X)...")
    activation_data = collect_activation_data(model, data_loader, device, num_batches=num_batches)
    
    layer_stats = {}

    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            W_original = module.weight.data.clone().T
            d, m = W_original.shape

            if d < 8 or m < 8:
                print(f"  跳过层 {name}（太小: {d}x{m}）")
                continue
                
            if name not in activation_data:
                print(f"  跳过层 {name}（无激活数据）")
                continue
            
            original_bits = d * m * 32
            quantized_bits = (d * d * 32) + (d * 32) + (d * m * 2) 
            if quantized_bits >= original_bits:
                print(f"  跳过层 {name}（负压缩收益: 原始={original_bits/8e3:.1f}KB, 量化={quantized_bits/8e3:.1f}KB, U矩阵({d}x{d})开销过大）")
                continue

            X_original = activation_data[name]
            print(f"  量化层 {name} (d={d}, W={m}, X_samples={X_original.shape[1]})...")
            
            W_quantized, stats = quantize_weight_and_activation(
                X_original, W_original, max_iters=max_iters
            )

            module.weight.data = W_quantized.T.clone()
            layer_stats[name] = stats
            print(f"  ✓ 完成层 {name} 量化")

    return layer_stats


def estimate_quantized_storage_bits(model, layer_stats, float_bits=32):
    total_bits = 0
    quantized_weight_params = set()

    codebook_size = 4
    for name in layer_stats.keys():
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

def plot_quantization_history_onlyX(layer_stats, save_dir="plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for layer_name, stats in layer_stats.items():
        history_x = stats.get("history_x", [])
        history_w = stats.get("history_w", [])
        
        if not history_x:
            continue

        epochs = range(1, len(history_x) + 1)

        plt.figure(figsize=(10, 6))
        # 移除了 Total Loss，只保留 X 和 W 的部分
        plt.plot(epochs, history_x, label="Loss (Activation X)", linestyle='-', linewidth=2, color='blue', alpha=0.9)
        plt.plot(epochs, history_w, label="Loss (Weight W)", linestyle='--', linewidth=2, color='red', alpha=0.7)

        plt.title(f"Quantization Convergence (X-Driven Only) - Layer {layer_name}\n"
                  f"(d={stats['d']}, m={stats['m']}, N={stats['N']}, Iters={len(history_x)})", fontsize=14)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Relative Reconstruction Error (Loss)", fontsize=12)
        plt.yscale("log")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(fontsize=12)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"convergence_{layer_name}_onlyX.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  -> 生成图表: {save_path}")


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10("./data", train=True, download=True, transform=transform_train),
        batch_size=256, shuffle=True, num_workers=4,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10("./data", train=False, download=True, transform=transform_test),
        batch_size=512, shuffle=False, num_workers=4,
    )

    print("\n" + "=" * 70)
    print("=== 1) 创建并训练 ResNet-18 用于 CIFAR-10 基线模型 ===")
    print("=" * 70)
    baseline_model = create_resnet18_cifar10().to(device)
    total_params = count_model_params(baseline_model)
    print(f"模型参数总数: {total_params:,} ({total_params/1e6:.2f}M)")

    # (由于测试耗时，这里的 epoch 设定为 5。你可以按需要提回 15)
    train_baseline(baseline_model, train_loader, device, epochs=5)

    print("\n" + "=" * 70)
    print("=== 2) 评估基线模型（含重复测速）===")
    print("=" * 70)
    base_acc = evaluate_accuracy(baseline_model, test_loader, device)
    base_latency = benchmark_inference(baseline_model, test_loader, device)
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
    print(f"基线参数体积: {base_size_mb:.4f} MB")

    print("\n" + "=" * 70)
    print("=== 3) 应用基于纯 X 驱动的被动量化到 FC 层 ===")
    print("=" * 70)
    quant_model = copy.deepcopy(baseline_model)
    layer_stats = apply_quantization_on_linear_layers(
        quant_model, test_loader, device, max_iters=50, num_batches=1
    )

    print(f"\n量化了 {len(layer_stats)} 层:")
    for layer_name, stats in layer_stats.items():
        hist_str = f"迭代次数 {len(stats['history_x'])}"
        if stats['history_x']:
            hist_str += f", 最终Loss_X {stats['history_x'][-1]:.6f}, 误差MAE={stats['mae']:.6f}"
        print(f"  {layer_name}: d={stats['d']}, m={stats['m']}, {hist_str}")

    print("\n" + "=" * 70)
    print("=== 4) 评估量化后模型（含重复测速）===")
    print("=" * 70)
    quant_acc = evaluate_accuracy(quant_model, test_loader, device)
    quant_latency = benchmark_inference(quant_model, test_loader, device)
    quant_eval = EvalResult(
        accuracy=quant_acc,
        latency_mean_batch_ms=quant_latency["mean_batch_ms"],
        latency_std_batch_ms=quant_latency["std_batch_ms"],
        latency_mean_sample_ms=quant_latency["mean_sample_ms"],
        latency_std_sample_ms=quant_latency["std_sample_ms"],
    )
    quant_size_mb = model_size_mb(quant_model)
    estimated_quant_bits = estimate_quantized_storage_bits(quant_model, layer_stats, float_bits=32)
    estimated_quant_mb = estimated_quant_bits / 8.0 / (1024.0 * 1024.0)

    print(f"量化后准确率: {quant_eval.accuracy:.2f}%")
    print(f"量化后参数体积(float32): {quant_size_mb:.4f} MB")
    print(f"量化表示估算体积(位级): {estimated_quant_mb:.4f} MB ({estimated_quant_bits/1e6:.2f}M bits)")

    print("\n" + "=" * 70)
    print("=== 5) 量化效果详细总结 ===")
    print("=" * 70)
    
    print(f"\n【开始生成量化全过程的图表分析】...")
    plot_quantization_history_onlyX(layer_stats, save_dir="quantization_plots")
    
    acc_drop = base_eval.accuracy - quant_eval.accuracy
    latency_change = quant_eval.latency_mean_batch_ms - base_eval.latency_mean_batch_ms
    latency_change_pct = (latency_change / base_eval.latency_mean_batch_ms * 100) if base_eval.latency_mean_batch_ms > 0 else 0

    print(f"【精度指标】")
    print(f"  基线: {base_eval.accuracy:.2f}% | 量化: {quant_eval.accuracy:.2f}%")
    print(f"  精度变化: {acc_drop:+.2f}% (正值为基线赢，负值为量化反而升了)")

    print(f"\n【速度指标】")
    print(f"  基线 batch 时延: {base_eval.latency_mean_batch_ms:.4f} ms")
    print(f"  量化 batch 时延: {quant_eval.latency_mean_batch_ms:.4f} ms")
    print(f"  时延变化: {latency_change:+.4f} ms ({latency_change_pct:+.2f}%)")

if __name__ == "__main__":
    main()