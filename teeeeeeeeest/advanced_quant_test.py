import copy
import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os

# ==========================================
# 1) 模型定义（使用更复杂的 CNN 适配远端算力）
# ==========================================
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 8x8
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ==========================================
# 2) 核心量化方法（支持 Linear 和 Conv2d）
# ==========================================
def quantize_weight_matrix(W_numpy, max_iters=10):
    """
    对权重矩阵 W 做交替优化量化
    """
    d, M = W_numpy.shape
    codebook = np.array([-3, -1, 1, 3], dtype=np.float64)

    # 随机初始化正交矩阵 U
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
        M_mat = W_numpy @ D_w @ (Lambda_w @ Z_w).T
        P, _, Qt = np.linalg.svd(M_mat)
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


def apply_quantization_to_all_layers(model, max_iters=10):
    """
    遍历整个模型，自动将所有的 Conv2d 和 Linear 层权重进行量化。
    """
    layer_stats = {}
    
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"正在量化 Linear 层: {name} (shape: {module.weight.shape})")
                W_original = module.weight.data.cpu().numpy().T # Shape: (in_features, out_features)
                W_quantized, stats = quantize_weight_matrix(W_original, max_iters=max_iters)
                module.weight.data = torch.from_numpy(W_quantized.T).to(module.weight.data.device).to(module.weight.data.dtype)
                layer_stats[name] = stats
                
            elif isinstance(module, nn.Conv2d):
                print(f"正在量化 Conv2d 层: {name} (shape: {module.weight.shape})")
                out_c, in_c, kH, kW = module.weight.shape
                # 将卷积核展开成 (out_c, in_c * kH * kW) 形式
                W_original = module.weight.data.cpu().numpy().reshape(out_c, -1)
                W_quantized, stats = quantize_weight_matrix(W_original, max_iters=max_iters)
                # 重新恢复成 4D 张量
                module.weight.data = torch.from_numpy(W_quantized.reshape(out_c, in_c, kH, kW)).to(module.weight.data.device).to(module.weight.data.dtype)
                layer_stats[name] = stats

    return layer_stats

# ==========================================
# 3) 训练与评估工具
# ==========================================
@dataclass
class EvalResult:
    accuracy: float
    total_infer_ms: float
    avg_batch_ms: float
    avg_sample_ms: float

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_infer_s = 0.0
    batch_count = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            output = model(data)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()

            total_infer_s += (end - start)
            batch_count += 1

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / max(total, 1)
    total_infer_ms = total_infer_s * 1000.0
    return EvalResult(
        accuracy=accuracy,
        total_infer_ms=total_infer_ms,
        avg_batch_ms=total_infer_ms / max(batch_count, 1),
        avg_sample_ms=total_infer_ms / max(total, 1),
    )

def train_model(model, train_loader, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 200 == 199:
                print(f"[Epoch {epoch + 1}, Batch {batch_idx + 1:4d}] Loss: {running_loss / 200:.4f}")
                running_loss = 0.0

def model_size_mb(model):
    total_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    return total_bytes / (1024.0 * 1024.0)

# ==========================================
# 4) 主函数入口
# ==========================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # 自动识别加速硬件 (CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的计算设备: {device} | {'GPU: ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    os.makedirs("./data", exist_ok=True)
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

    print("\n加载 CIFAR-10 数据集...")
    train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)

    # 为了加速测试环境，可以用更大型的batch size
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4 if device.type == 'cuda' else 0, pin_memory=True if device.type == 'cuda' else False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4 if device.type == 'cuda' else 0, pin_memory=True if device.type == 'cuda' else False)

    print("\n=== 1) 训练 Float32 基线模型 ===")
    baseline_model = ConvNet().to(device)
    
    # 既然有算力，跑10个Epoch获取基线指标即可
    train_epochs = 10 
    print(f"预计训练 {train_epochs} 个 Epoch...")
    train_start_time = time.time()
    train_model(baseline_model, train_loader, device, epochs=train_epochs)
    print(f"训练完成，耗时: {time.time() - train_start_time:.2f} 秒")

    print("\n=== 2) 评估基线模型 ===")
    base_eval = evaluate_model(baseline_model, test_loader, device)
    base_size_mb = model_size_mb(baseline_model)

    print(f"基线准确率: {base_eval.accuracy:.2f}%")
    print(f"基线平均 batch 时延: {base_eval.avg_batch_ms:.4f} ms")
    print(f"基线参数体积: {base_size_mb:.4f} MB")

    print("\n=== 3) 全面应用量化方法 (包括 Conv2d 与 Linear) ===")
    # 量化非常消耗CPU或GPU矩阵运算，利用算力增加 EM 迭代次数至 15 次以达到更好地收敛
    quant_model = copy.deepcopy(baseline_model).to(device)
    quant_start_time = time.time()
    
    # 执行量化过程（此处耗时可能较长，正好利用算力验证）
    layer_stats = apply_quantization_to_all_layers(quant_model, max_iters=15)
    
    print(f"量化处理完成，总耗时: {time.time() - quant_start_time:.2f} 秒")
    
    for layer_name, stats in layer_stats.items():
        print(f"  - 层 {layer_name}: MSE={stats['mse']:.5f}, MAE={stats['mae']:.5f}")

    print("\n=== 4) 评估量化后模型 ===")
    quant_eval = evaluate_model(quant_model, test_loader, device)
    quant_size_mb = model_size_mb(quant_model)

    print(f"量化后准确率: {quant_eval.accuracy:.2f}%")
    print(f"量化后平均 batch 时延: {quant_eval.avg_batch_ms:.4f} ms")

    print("\n=== 5) 量化效果总结 ===")
    acc_drop = base_eval.accuracy - quant_eval.accuracy
    latency_change = quant_eval.avg_batch_ms - base_eval.avg_batch_ms

    print(f"-> 精度损失: {acc_drop:.2f}% ({base_eval.accuracy:.2f}% -> {quant_eval.accuracy:.2f}%)")
    print(f"-> 平均批次推理时延变化: {latency_change:+.4f} ms")
    print("\n【注意】由于当前量化实现最终依然恢复成了 float32 权重送给 PyTorch 计算，所以参数体积大小和推理速度并没有直接体现出模型压缩的好处。")
    print("如果要获得速度和体积的提升，需要配套编写针对低比特(Codebook)底层加速算子(C++/CUDA)。当前的测试主旨在评估该量化算法对精度的保持能力。")

if __name__ == "__main__":
    main()
