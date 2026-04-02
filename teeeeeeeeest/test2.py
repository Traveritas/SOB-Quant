import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# ==========================================
# 1. 定义一个简单的神经网络模型
# ==========================================
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 将 28x28 的图像展平为 784 维向量，然后映射到隐藏层，再映射到 10 个类别
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784) # 展平图像
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==========================================
# 2. 准备刚才我们写的量化核心逻辑 (简化版用于权重 W)
# ==========================================
def quantize_weight_matrix(W_numpy, d, M_cols, max_iters=5):
    """
    仅对权重矩阵 W 进行交替优化的量化算法
    """
    codebook = np.array([-3, -1, 1, 3]) 
    
    # 随机初始化 U 和 Lambda
    U, _ = np.linalg.qr(np.random.randn(d, d))
    Lambda_w = np.diag(np.random.rand(d) + 0.1)
    
    # 计算权重的范数倒数 D_w
    inv_norms_W = 1.0 / (np.sum(W_numpy**2, axis=0) + 1e-8)
    D_w = np.diag(inv_norms_W)

    print("  开始对权重进行量化分解...")
    for _ in range(max_iters):
        # E-Step: 更新 Z_w
        lambda_diag = np.diag(Lambda_w).copy()
        lambda_diag[lambda_diag == 0] = 1e-8 
        Z_tilde = np.diag(1.0 / lambda_diag) @ U.T @ W_numpy 
        
        diff = np.abs(Z_tilde[..., np.newaxis] - codebook)
        Z_w = codebook[np.argmin(diff, axis=-1)]
        
        # M-Step: 更新 Lambda_w
        S_WZ = W_numpy @ D_w @ Z_w.T
        S_ZZ = Z_w @ D_w @ Z_w.T
        lambda_diag = np.diag(U.T @ S_WZ) / (np.diag(S_ZZ) + 1e-8)
        Lambda_w = np.diag(lambda_diag)
        
        # M-Step: 更新 U (针对单边W的简化版 Procrustes)
        M = W_numpy @ D_w @ (Lambda_w @ Z_w).T
        P, _, Qt = np.linalg.svd(M)
        U = P @ Qt

    # 重建量化后的权重: W_approx = U * Lambda * Z_w
    W_approx = U @ Lambda_w @ Z_w
    return W_approx

# ==========================================
# 3. 训练、测试与量化注入的主流程
# ==========================================
def main():
    # 设定设备 (CPU 即可运行这个小实验)
    device = torch.device("cpu")
    
    # 获取测试用的 MNIST 数据集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=1000, shuffle=False)

    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("--- 步骤 1: 正常训练模型 (Float32) ---")
    model.train()
    # 为了演示，只训练 1 个 Epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print(f"  训练进度: {batch_idx * len(data)}/{len(train_loader.dataset)} Loss: {loss.item():.4f}")

    print("\n--- 步骤 2: 测试原始模型精度 ---")
    def test_model(current_model):
        current_model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = current_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f"  当前模型准确率: {accuracy:.2f}%")
        return accuracy

    original_acc = test_model(model)

    print("\n--- 步骤 3: 提取权重并应用 Lattice K-Means 量化 ---")
    # 我们提取第一层全连接层 (fc1) 的权重
    # 注意：PyTorch 中的 Linear 权重形状是 (out_features, in_features)
    # 我们需要转置它以符合论文中 W 的列表示法 [cite: 109]
    W_tensor = model.fc1.weight.data
    W_numpy = W_tensor.numpy().T 
    d, M_cols = W_numpy.shape
    
    # 运行你的量化算法
    W_quantized_numpy = quantize_weight_matrix(W_numpy, d, M_cols, max_iters=3)
    
    # 将量化并重建后的权重替换回 PyTorch 模型中 (需要再转置回来)
    model.fc1.weight.data = torch.from_numpy(W_quantized_numpy.T).float()

    print("\n--- 步骤 4: 测试量化后的模型精度 ---")
    quantized_acc = test_model(model)
    print(f"\n总结: 原始精度 {original_acc:.2f}%, 量化后精度 {quantized_acc:.2f}%, 精度损失 {original_acc - quantized_acc:.2f}%")

if __name__ == '__main__':
    main()