import numpy as np

# ==========================================
# 辅助函数：最近邻量化 (对应文档公式 17)
# ==========================================
def quantize(z_continuous, codebook):
    """
    将连续的 z 映射到最近的离散码本(codebook)值上 [cite: 146, 147]
    z_continuous: 连续变量数组
    codebook: 离散值的集合，例如 [-3, -1, 1, 3]
    """
    # 计算连续值与码本中所有值的距离，并取最近的索引
    diff = np.abs(z_continuous[..., np.newaxis] - codebook)
    idx = np.argmin(diff, axis=-1)
    return codebook[idx]

# ==========================================
# E-Step：更新潜在编码 Z 
# ==========================================
def e_step(Data, U, Lambda, codebook):
    """
    计算最优的离散编码 Z
    Data: 输入数据 (X 或 W)
    U: 正交矩阵
    Lambda: 对角缩放矩阵
    """
    # 提取对角线元素并避免除以0
    lambda_diag = np.diag(Lambda).copy()
    lambda_diag[lambda_diag == 0] = 1e-8 
    
    # 按照公式计算连续解: z_tilde_i = (u_i^T x) / lambda_i 
    Lambda_inv = np.diag(1.0 / lambda_diag)
    Z_tilde = Lambda_inv @ U.T @ Data 
    
    # 离散化：最近邻量化 
    Z_discrete = quantize(Z_tilde, codebook)
    return Z_discrete

# ==========================================
# M-Step：更新缩放矩阵 Lambda [cite: 260]
# ==========================================
def update_lambda(Data, Z, U, D):
    """
    更新对角缩放矩阵 Lambda_x 或 Lambda_w
    """
    # S_XZ = X D Z^T  [cite: 279, 306]
    S_XZ = Data @ D @ Z.T
    # S_ZZ = Z D Z^T  [cite: 280, 307]
    S_ZZ = Z @ D @ Z.T
    
    # 根据公式求解对角线元素: lambda_i = (U^T S_XZ)_{ii} / (S_ZZ)_{ii} [cite: 283, 310]
    num = np.diag(U.T @ S_XZ)
    den = np.diag(S_ZZ)
    
    # 防止分母为0
    lambda_diag = num / (den + 1e-8)
    return np.diag(lambda_diag)

# ==========================================
# M-Step：更新正交矩阵 U [cite: 166]
# ==========================================
def update_U_procrustes(X, Z_x, Lambda_x, D_x, W, Z_w, Lambda_w, D_w, beta=1.0):
    """
    使用正交普罗克鲁斯提斯问题(Orthogonal Procrustes)的闭式解更新 U 
    """
    # Y = Lambda * Z [cite: 168, 169]
    Y_x = Lambda_x @ Z_x
    Y_w = Lambda_w @ Z_w
    
    # M = X D_x Y_x^T + beta * W D_w Y_w^T [cite: 202]
    M = X @ D_x @ Y_x.T + beta * W @ D_w @ Y_w.T
    
    # 对 M 进行奇异值分解 (SVD): M = P Sigma Q^T [cite: 208, 211]
    P, _, Qt = np.linalg.svd(M)
    
    # 闭式解最优的 U = P Q^T [cite: 220]
    U = P @ Qt
    return U

# ==========================================
# 主训练循环 (Alternating Optimization)
# ==========================================
def train_quantization(X, W, d, N, M_cols, max_iters=10):
    # 1. 初始化
    # 构造码本 (例如一个2-bit的均匀量化器)
    codebook = np.array([-3, -1, 1, 3]) 
    
    # 随机初始化正交矩阵 U (通过 QR 分解生成)
    random_matrix = np.random.randn(d, d)
    U, _ = np.linalg.qr(random_matrix)
    
    # 随机初始化对角矩阵 Lambda_x 和 Lambda_w
    Lambda_x = np.diag(np.random.rand(d) + 0.1)
    Lambda_w = np.diag(np.random.rand(d) + 0.1)
    
    # 计算常数权重矩阵 D_x 和 D_w (即 1/||x||_2^2 ) 
    inv_norms_X = 1.0 / (np.sum(X**2, axis=0) + 1e-8)
    D_x = np.diag(inv_norms_X)
    
    inv_norms_W = 1.0 / (np.sum(W**2, axis=0) + 1e-8)
    D_w = np.diag(inv_norms_W)

    print("开始交替优化...")
    for iteration in range(max_iters):
        # ---------------- E-Step ---------------- 
        Z_x = e_step(X, U, Lambda_x, codebook)
        Z_w = e_step(W, U, Lambda_w, codebook)
        
        # ---------------- M-Step ---------------- [cite: 148]
        # 更新 U (文档提供了闭式解) 
        U = update_U_procrustes(X, Z_x, Lambda_x, D_x, W, Z_w, Lambda_w, D_w, beta=1.0)
        
        # 分别更新 Lambda_x 和 Lambda_w [cite: 260]
        Lambda_x = update_lambda(X, Z_x, U, D_x)
        Lambda_w = update_lambda(W, Z_w, U, D_w)
        
        # 计算相对重建误差 (Loss) 观察收敛情况 [cite: 151]
        loss_x = np.sum(inv_norms_X * np.sum((X - U @ Lambda_x @ Z_x)**2, axis=0))
        loss_w = np.sum(inv_norms_W * np.sum((W - U @ Lambda_w @ Z_w)**2, axis=0))
        total_loss = loss_x + loss_w
        
        print(f"迭代第 {iteration+1} 次, Total Loss: {total_loss:.4f}")

    return U, Lambda_x, Lambda_w, Z_x, Z_w

# ==========================================
# 运行实验
# ==========================================
if __name__ == "__main__":
    # 模拟数据生成
    d_dim = 64        # 向量维度
    N_samples = 1000  # X 的样本数
    M_samples = 500   # W 的列数 (比如网络层的参数量)

    # 随机生成一些特征和权重数据进行测试
    np.random.seed(42)
    X_dummy = np.random.randn(d_dim, N_samples)
    W_dummy = np.random.randn(d_dim, M_samples)

    # 训练我们的量化模型
    U_opt, Lx_opt, Lw_opt, Zx_opt, Zw_opt = train_quantization(
        X_dummy, W_dummy, d=d_dim, N=N_samples, M_cols=M_samples, max_iters=15
    )
    print("训练完成！")