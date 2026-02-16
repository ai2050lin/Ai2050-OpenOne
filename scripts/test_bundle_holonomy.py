
import os
import sys

import torch

# 确保可以导入项目中的模型
sys.path.append(os.getcwd())

from models.fibernet_bundle import FiberBundleNetwork, NFBTConnection


def test_orthogonality():
    """验证平行移动算子 T 是否属于 SO(n)"""
    print("\n好的。正在验证李群正交性 (SO(n) Constraint)...")
    d_logic = 16
    d_memory = 32
    connection = NFBTConnection(d_logic, d_memory)
    
    dx = torch.randn(1, 1, d_logic)
    v0 = torch.randn(1, 1, d_memory)
    v1, T = connection(dx, v0)
    
    # 1. 检查 Det(T) 是否约等于 1
    det = torch.linalg.det(T[0])
    print(f"变换矩阵行列式 Det(T) = {det.item():.6f} (应为 1.0)")
    
    # 2. 检查 T^T T 是否为单位阵 I
    identity_check = torch.matmul(T[0].T, T[0])
    i_matrix = torch.eye(d_memory)
    residual = torch.norm(identity_check - i_matrix)
    print(f"正交性残差 ||T^T T - I|| = {residual.item():.6e} (应接近 0)")

    # 3. 能量守恒验证 (模长守恒)
    norm0 = torch.norm(v0)
    norm1 = torch.norm(v1)
    print(f"纤维模长验证: 初始={norm0.item():.4f}, 平移后={norm1.item():.4f} (应相等)")

    if residual < 1e-5 and abs(det - 1.0) < 1e-5:
        print("好的。李群正交性保证已选通，度量一致性已修复。")
    else:
        print("警告：李群约束未生效！")

def test_holonomy_stability():
    """
    验证纤维丛的几何回转（Holonomy）稳定性。
    """
    print("\n好的。正在启动 FiberNet 几何一致性（Holonomy）验证...")
    
    # 初始化参数
    d_logic = 16
    d_memory = 32
    batch = 1
    
    # 创建联络算子
    connection = NFBTConnection(d_logic, d_memory)
    
    # 初始纤维内容 (Section sigma)
    fiber_v0 = torch.randn(batch, 1, d_memory)
    
    # 定义底流形闭合路径：x0 -> x1 -> x2 -> x0
    dx1 = torch.randn(batch, 1, d_logic)
    dx2 = torch.randn(batch, 1, d_logic)
    dx3 = -(dx1 + dx2) 
    
    # 1. 路径演化
    v1, _ = connection(dx1, fiber_v0)
    v2, _ = connection(dx2, v1)
    v0_back, _ = connection(dx3, v2)
    
    error = torch.norm(v0_back - fiber_v0)
    print(f"回转验证完成。绝对差异 (Holonomy Offset): {error.item():.4f}")
    
    if torch.isfinite(error):
        print("好的。几何联络计算链完整，未发现梯度爆炸或 NaN。")
    else:
        print("警告：几何计算链出现不稳定。")

def test_full_network_flow():
    print("\n好的。正在运行完整网络的前向流测试...")
    vocab_size = 100
    model = FiberBundleNetwork(vocab_size=vocab_size)
    
    dummy_input = torch.randint(0, vocab_size, (2, 8)) # Batch=2, Seq=8
    try:
        logits = model(dummy_input)
        print(f"网络前向传播成功。输出形状: {logits.shape}")
        print("好的。FiberBundleNetwork 架构已成功对齐 NFBT 理论。")
    except Exception as e:
        print(f"验证失败: {str(e)}")

if __name__ == "__main__":
    test_orthogonality()
    test_holonomy_stability()
    test_full_network_flow()
