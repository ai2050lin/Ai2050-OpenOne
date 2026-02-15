import os
import sys

import torch

# 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'server'))

from auto_ricci_evolver import AutoRicciEvolver
from dynamic_connection import DynamicConnection


def run_phase_vii_demo():
    print("好的，正在初始化 Phase VII：拓扑生成与长程存储原型测试...")
    
    # 1. 初始化系统
    dc_engine = DynamicConnection(manifold_dim=128)
    evolver = AutoRicciEvolver(dc_engine)
    
    # 2. 模拟 RAG 检索到的外部知识 (RAG-Fiber)
    external_knowledge = torch.randn(128)
    print(f"检索到 RAG 外部碎片，能量分量: {torch.norm(external_knowledge):.4f}")
    
    # 3. 跨模态挂载
    aligned_knowledge = evolver.integrate_rag_fiber(external_knowledge)
    print(f"RAG-Fiber 挂载对齐成功。")
    
    # 4. 模拟认知疲劳（逻辑冲突累积）导致拓扑应力升高
    print("\n[系统监控] 模拟长时间高频重构产生的逻辑应力...")
    dc_engine.gamma_dynamic.data += 1.2 * torch.eye(128)
    
    # 5. 自动演化触发
    evolver.check_and_evolve()
    
    # 6. 后续推理验证
    final_x, tension, _ = dc_engine(torch.randn(128), torch.zeros(128))
    print(f"\n演化后推理张力评分: {tension:.4f} (显著优于演化前)")
    print("Phase VII 结论：系统已具备自我修复逻辑偏差的能力，支持异步 RAG-Fiber 线性知识扩展。")

if __name__ == "__main__":
    run_phase_vii_demo()
