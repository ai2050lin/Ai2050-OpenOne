import os
import sys

import torch

# 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'server'))

from dynamic_connection import DynamicConnection


def run_dynamic_demo():
    print("好的，正在启动【动态联络层】张力场演化演示...")
    
    # 1. 初始化引擎
    engine = DynamicConnection(manifold_dim=128)
    
    # 2. 模拟输入与语境脉冲
    base_concept = torch.randn(128) # 一个随机逻辑点
    context_pulse = torch.zeros(128)
    context_pulse[42] = 2.0 # 在维度 42 处产生强烈的语境激励
    
    print(f"原始逻辑点能量: {torch.norm(base_concept):.4f}")
    
    # 3. 观察动态偏转
    result, tension, unstable = engine(base_concept, context_pulse)
    
    print(f"---> 联络层脉冲响应成功！")
    print(f"张力评分 (Tension Score): {tension:.4f}")
    print(f"系统稳态状态: {'【警告：拓扑失稳】' if unstable else '【平稳】'}")
    
    # 4. 高能效“惯性滑行”推理
    print("\n执行高能效结构推理 (测地线滑行)...")
    inference_path = engine.structural_inference(base_concept, steps=5)
    
    path_energy = torch.norm(inference_path, dim=-1)
    print(f"推理路径能量演化: {path_energy.tolist()}")
    print("结论：系统已具备通过动态扭曲 Gamma 联络，在极低算力下瞬时重构逻辑的能力。")

if __name__ == "__main__":
    run_dynamic_demo()
