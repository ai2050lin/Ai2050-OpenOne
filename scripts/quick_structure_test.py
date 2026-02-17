"""
快速结构提取测试
使用模拟数据快速验证结构提取器
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np


def create_simple_transformer():
    """创建简单的 Transformer 用于测试"""
    
    class SimpleTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = type('Config', (), {
                'd_vocab': 1000,
                'd_model': 128,
                'n_layers': 4,
                'n_heads': 4
            })()
            
            self.embed = nn.Embedding(1000, 128)
            
            # 4层 Transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=128, 
                nhead=4, 
                dim_feedforward=512,
                batch_first=True
            )
            self.layers = nn.TransformerEncoder(encoder_layer, num_layers=4)
            
            self.ln_f = nn.LayerNorm(128)
            self.unembed = nn.Linear(128, 1000)
        
        def forward(self, x):
            x = self.embed(x)
            x = self.layers(x)
            x = self.ln_f(x)
            return self.unembed(x)
    
    return SimpleTransformer()


def test_structure_extractor():
    """测试结构提取器"""
    print("\n" + "="*60)
    print("结构提取器测试")
    print("="*60)
    
    from models.structure_extractor import (
        StructureExtractor,
        generate_structure_report
    )
    
    # 创建模型
    print("\n[1] 创建测试模型...")
    model = create_simple_transformer()
    model.eval()
    
    # 创建提取器
    print("[2] 初始化结构提取器...")
    extractor = StructureExtractor(device='cpu')
    
    # 提取结构
    print("[3] 提取模型结构...")
    sample_inputs = torch.randint(0, 1000, (20, 32))
    
    structure = extractor.extract_from_model(
        model,
        "SimpleTransformer-Test",
        sample_inputs=sample_inputs,
        n_samples=20
    )
    
    # 打印报告
    print("[4] 生成结构报告...")
    report = generate_structure_report(structure)
    print(report)
    
    print("\n[OK] 测试完成！")
    
    return structure


def test_manifold_analyzer():
    """测试流形分析器"""
    print("\n" + "="*40)
    print("流形分析器测试")
    print("="*40)
    
    from models.structure_extractor import ManifoldAnalyzer
    
    analyzer = ManifoldAnalyzer()
    
    # 创建模拟激活数据（低维流形嵌入高维空间）
    print("\n生成模拟流形数据...")
    n_samples = 500
    intrinsic_dim = 5
    ambient_dim = 50
    
    # 在低维空间生成数据
    low_dim = np.random.randn(n_samples, intrinsic_dim)
    # 随机投影到高维空间
    projection = np.random.randn(intrinsic_dim, ambient_dim)
    high_dim_data = low_dim @ projection + np.random.randn(n_samples, ambient_dim) * 0.1
    
    # 估计维度
    estimated_dim = analyzer.estimate_intrinsic_dimension(high_dim_data)
    print(f"真实维度: {intrinsic_dim}")
    print(f"估计维度: {estimated_dim:.2f}")
    
    # 估计曲率
    curvature = analyzer.estimate_local_curvature(high_dim_data)
    print(f"平均曲率: {curvature.mean():.4f}")
    
    print("\n[OK] 流形分析器测试完成")


def test_spectral_analyzer():
    """测试谱分析器"""
    print("\n" + "="*40)
    print("谱分析器测试")
    print("="*40)
    
    from models.structure_extractor import SpectralAnalyzer
    
    analyzer = SpectralAnalyzer()
    
    # 创建不同秩的矩阵
    print("\n分析不同秩的矩阵...")
    
    # 低秩矩阵
    low_rank = np.random.randn(100, 5) @ np.random.randn(5, 100)
    s_low = analyzer.compute_singular_values(low_rank)
    rank_low = analyzer.compute_effective_rank(s_low)
    print(f"低秩矩阵 (秩=5): 有效秩比例 = {rank_low:.4f}")
    
    # 满秩矩阵
    full_rank = np.random.randn(100, 100)
    s_full = analyzer.compute_singular_values(full_rank)
    rank_full = analyzer.compute_effective_rank(s_full)
    print(f"满秩矩阵 (秩=100): 有效秩比例 = {rank_full:.4f}")
    
    # 条件数
    cond_low = analyzer.compute_condition_number(low_rank)
    cond_full = analyzer.compute_condition_number(full_rank)
    print(f"低秩矩阵条件数: {cond_low:.2f}")
    print(f"满秩矩阵条件数: {cond_full:.2f}")
    
    print("\n[OK] 谱分析器测试完成")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("神经网络结构提取器 - 综合测试")
    print("="*60)
    
    test_manifold_analyzer()
    test_spectral_analyzer()
    test_structure_extractor()
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
