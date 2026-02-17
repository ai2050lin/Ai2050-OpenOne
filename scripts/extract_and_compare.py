"""
GPT-2 和 Qwen3 结构提取与比较

运行命令：
python scripts/extract_and_compare.py

输出：
1. 各模型的详细结构报告
2. 模型间的结构对比
3. 可视化结果
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import json
from datetime import datetime

# 导入结构提取器
from models.structure_extractor import (
    StructureExtractor, 
    compare_structures, 
    generate_structure_report,
    ModelStructure
)


def load_gpt2(device: str = 'cpu'):
    """加载 GPT-2 模型"""
    print("\n加载 GPT-2 模型...")
    
    try:
        import transformer_lens as tl
        from transformer_lens import HookedTransformer
        
        model = HookedTransformer.from_pretrained(
            "gpt2-small",
            device=device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False
        )
        print("✓ GPT-2 加载成功")
        return model, "GPT-2-small"
        
    except Exception as e:
        print(f"GPT-2 加载失败: {e}")
        print("使用模拟模型...")
        return create_mock_model("GPT-2-small"), "GPT-2-small"


def load_qwen(device: str = 'cpu'):
    """加载 Qwen 模型"""
    print("\n加载 Qwen 模型...")
    
    try:
        # 尝试使用 transformers 加载 Qwen
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "Qwen/Qwen2.5-0.5B"  # 使用较小的模型
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device,
            trust_remote_code=True
        )
        
        print(f"✓ {model_name} 加载成功")
        return model, "Qwen2.5-0.5B"
        
    except Exception as e:
        print(f"Qwen 加载失败: {e}")
        print("使用模拟模型...")
        return create_mock_model("Qwen2.5-mock"), "Qwen2.5-mock"


def create_mock_model(name: str):
    """创建模拟模型用于测试"""
    import torch.nn as nn
    
    class MockModel(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.cfg = type('Config', (), {
                'd_vocab': 50000,
                'd_model': 768,
                'n_layers': 12,
                'n_heads': 12
            })()
            
            # 模拟层
            self.embed = nn.Embedding(50000, 768)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072)
                for _ in range(12)
            ])
            self.ln_f = nn.LayerNorm(768)
            self.unembed = nn.Linear(768, 50000)
        
        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers:
                x = layer(x)
            x = self.ln_f(x)
            return self.unembed(x)
        
        def named_modules(self):
            yield "embed", self.embed
            for i, layer in enumerate(self.layers):
                yield f"layer_{i}", layer
                yield f"layer_{i}.mlp", layer.linear1
                yield f"layer_{i}.attn", layer.self_attn
            yield "ln_f", self.ln_f
            yield "unembed", self.unembed
        
        def named_parameters(self):
            for name, param in super().named_parameters():
                yield name, param
        
        def run_with_cache(self, x):
            # 简单模拟
            output = self.forward(x)
            cache = {
                f"layer_{i}_attn_pattern": torch.rand(1, 12, x.shape[1], x.shape[1])
                for i in range(12)
            }
            return output, cache
    
    return MockModel(name)


def extract_and_save(model, model_name: str, extractor: StructureExtractor, 
                     output_dir: str, device: str = 'cpu'):
    """提取并保存模型结构"""
    
    # 生成采样输入
    vocab_size = getattr(model.cfg, 'd_vocab', 50000) if hasattr(model, 'cfg') else 50000
    sample_inputs = torch.randint(0, vocab_size, (50, 64))
    sample_inputs = sample_inputs.to(device)
    
    # 提取结构
    structure = extractor.extract_from_model(
        model, 
        model_name, 
        sample_inputs=sample_inputs,
        n_samples=50
    )
    
    # 生成报告
    report = generate_structure_report(structure)
    print(report)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{model_name}_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    # 保存报告
    with open(os.path.join(output_path, "structure_report.txt"), 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 保存数值数据
    save_structure_data(structure, output_path)
    
    print(f"结果已保存到: {output_path}")
    
    return structure


def save_structure_data(structure: ModelStructure, output_path: str):
    """保存结构数据为 JSON 格式"""
    data = {
        'model_name': structure.model_name,
        'manifold': {
            'intrinsic_dim': float(structure.manifold.intrinsic_dim),
            'dimensions': structure.manifold.dimensions.tolist(),
            'curvature_mean': float(structure.manifold.curvature.mean())
        },
        'topology': {
            'connectivity_ratio': float(structure.topology.connectivity_ratio),
            'avg_betti': np.mean(structure.topology.betti_numbers, axis=0).tolist() if structure.topology.betti_numbers else []
        },
        'info_flow': {
            'flow_efficiency': float(structure.info_flow.flow_efficiency),
            'information_bottleneck': structure.info_flow.information_bottleneck,
            'layer_entropy': structure.info_flow.layer_entropy.tolist()
        },
        'attention': {
            'pattern_diversity': float(structure.attention.pattern_diversity),
            'n_induction_heads': len(structure.attention.induction_heads),
            'head_importance_mean': float(structure.attention.head_importance.mean()) if len(structure.attention.head_importance) > 0 else 0
        },
        'spectral': {
            'avg_spectral_norm': float(structure.spectral.spectral_norms.mean()),
            'avg_rank_ratio': float(structure.spectral.rank_ratio.mean()),
            'avg_condition_number': float(structure.spectral.condition_numbers.mean())
        },
        'layer_statistics': structure.layer_statistics
    }
    
    with open(os.path.join(output_path, "structure_data.json"), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def generate_comparison_report(comp: dict) -> str:
    """生成对比报告"""
    report = []
    report.append(f"\n{'='*60}")
    report.append("模型结构对比报告")
    report.append(f"{'='*60}")
    report.append(f"\n模型A: {comp['model_a']}")
    report.append(f"模型B: {comp['model_b']}")
    report.append(f"\n结构相似度: {comp['similarity_score']:.4f}")
    
    report.append("\n【差异指标】")
    for metric, value in comp['metrics'].items():
        report.append(f"  {metric}: {value:.4f}")
    
    report.append(f"\n{'='*60}\n")
    
    return '\n'.join(report)


def main():
    """主函数"""
    print("\n" + "="*60)
    print("神经网络数学结构提取器")
    print("="*60)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 输出目录
    output_dir = "experiments/structure_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建提取器
    extractor = StructureExtractor(device=device)
    
    # 加载模型
    structures = {}
    
    # 1. 提取 GPT-2 结构
    try:
        gpt2_model, gpt2_name = load_gpt2(device)
        structures['gpt2'] = extract_and_save(
            gpt2_model, gpt2_name, extractor, output_dir, device
        )
    except Exception as e:
        print(f"GPT-2 提取失败: {e}")
    
    # 2. 提取 Qwen 结构
    try:
        qwen_model, qwen_name = load_qwen(device)
        structures['qwen'] = extract_and_save(
            qwen_model, qwen_name, extractor, output_dir, device
        )
    except Exception as e:
        print(f"Qwen 提取失败: {e}")
    
    # 3. 模型对比
    if len(structures) == 2:
        print("\n" + "="*60)
        print("模型对比分析")
        print("="*60)
        
        comparison = compare_structures(structures['gpt2'], structures['qwen'])
        comp_report = generate_comparison_report(comparison)
        print(comp_report)
        
        # 保存对比结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comp_path = os.path.join(output_dir, f"comparison_{timestamp}")
        os.makedirs(comp_path, exist_ok=True)
        
        with open(os.path.join(comp_path, "comparison_report.txt"), 'w', encoding='utf-8') as f:
            f.write(comp_report)
        
        with open(os.path.join(comp_path, "comparison_data.json"), 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        print(f"对比结果已保存到: {comp_path}")
    
    print("\n" + "="*60)
    print("结构提取完成！")
    print("="*60)


if __name__ == "__main__":
    main()
