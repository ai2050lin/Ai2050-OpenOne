"""
简化版几何干预测试
==================

测试基本的几何干预功能
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable

# ============================================================================
# 几何干预模块（简化版）
# ============================================================================

class HeatKernelDiffuser:
    """热核扩散器"""
    
    def __init__(self, t: float = 1.0, alpha: float = 0.1):
        self.t = t
        self.alpha = alpha
        
    def diffuse(self, activations: torch.Tensor, reference: Optional[torch.Tensor] = None) -> torch.Tensor:
        """应用热核扩散"""
        if reference is None:
            return activations
        
        # 计算距离和权重
        flat_act = activations.reshape(-1, activations.size(-1))
        distances = torch.cdist(flat_act, reference)
        weights = torch.exp(-distances**2 / (4 * self.t))
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 加权平均
        diffused = torch.matmul(weights, reference)
        diffused = diffused.reshape_as(activations)
        
        return (1 - self.alpha) * activations + self.alpha * diffused


class GeodesicSteerer:
    """测地线引导器"""
    
    def __init__(self, step_size: float = 0.1):
        self.step_size = step_size
        self.directions = {}
        
    def set_direction(self, name: str, vector: torch.Tensor):
        self.directions[name] = F.normalize(vector, dim=-1)
        
    def steer(self, activations: torch.Tensor, direction_name: str, magnitude: Optional[float] = None) -> torch.Tensor:
        if direction_name not in self.directions:
            return activations
        
        direction = self.directions[direction_name]
        step = magnitude if magnitude is not None else self.step_size
        
        guided = activations + step * direction
        original_norm = torch.norm(activations, dim=-1, keepdim=True)
        return F.normalize(guided, dim=-1) * original_norm


class CurvatureAnalyzer:
    """曲率分析器"""
    
    def compute_curvature(self, activations: torch.Tensor, k: int = 5) -> torch.Tensor:
        """计算局部曲率"""
        flat_act = activations.reshape(-1, activations.size(-1))
        n_points = flat_act.size(0)
        
        if n_points < k + 1:
            return torch.zeros(activations.shape[:2])
        
        distances = torch.cdist(flat_act, flat_act)
        _, indices = torch.topk(distances, k + 1, largest=False)
        indices = indices[:, 1:]
        
        curvatures = []
        for i in range(n_points):
            neighbors = flat_act[indices[i]]
            local_var = torch.var(neighbors, dim=0).mean()
            curvatures.append(local_var)
        
        return torch.stack(curvatures).reshape(*activations.shape[:2])


# ============================================================================
# 激活提取器
# ============================================================================

class ActivationExtractor:
    """激活提取器"""
    
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.handles = []
        
    def get_hook_fn(self, name: str) -> Callable:
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        return hook_fn
    
    def register_hooks(self, layer_indices: List[int]):
        self.clear_hooks()
        
        # GPT-2 架构
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            for idx in layer_indices:
                if idx < len(self.model.transformer.h):
                    hook = self.model.transformer.h[idx].register_forward_hook(
                        self.get_hook_fn(f"layer_{idx}")
                    )
                    self.handles.append(hook)
    
    def clear_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.activations = {}
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self.activations


# ============================================================================
# 测试函数
# ============================================================================

def test_geometric_intervention():
    """测试几何干预"""
    print("=" * 60)
    print("几何干预测试（简化版）")
    print("=" * 60)
    
    # 加载模型
    print("\n[1] 加载 GPT-2 模型...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    device = "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    
    print(f"  [OK] 模型加载成功")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # 创建干预模块
    heat_diffuser = HeatKernelDiffuser(t=1.0, alpha=0.1)
    geodesic_steerer = GeodesicSteerer(step_size=0.1)
    curvature_analyzer = CurvatureAnalyzer()
    extractor = ActivationExtractor(model)
    
    # 提取激活
    print("\n[2] 提取层激活...")
    test_prompts = ["The capital of France is", "2 + 2 equals"]
    layer_indices = [0, 3, 6, 9, 11]
    
    all_activations = {}
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        extractor.register_hooks(layer_indices)
        
        with torch.no_grad():
            _ = model(**inputs)
        
        for name, act in extractor.get_activations().items():
            if name not in all_activations:
                all_activations[name] = []
            all_activations[name].append(act)
        
        extractor.clear_hooks()
    
    print(f"  [OK] 提取了 {len(all_activations)} 层激活")
    
    # 分析激活几何
    print("\n[3] 分析激活几何...")
    
    analysis = {}
    for layer_name, acts in all_activations.items():
        # 不同 prompt 的序列长度可能不同，先展开到 token 维再拼接
        token_blocks = [act.reshape(-1, act.size(-1)) for act in acts]
        combined = torch.cat(token_blocks, dim=0).unsqueeze(0)  # (1, total_tokens, hidden)
        curvature = curvature_analyzer.compute_curvature(combined)
        
        # PCA
        flat = combined.reshape(-1, combined.size(-1))
        U, S, V = torch.linalg.svd(flat - flat.mean(dim=0), full_matrices=False)
        
        analysis[layer_name] = {
            "shape": list(combined.shape),
            "mean_curvature": float(curvature.mean()),
            "max_curvature": float(curvature.max()),
            "pca_top3_variance": float(S[:3].sum() / S.sum())
        }
        
        print(f"\n  {layer_name}:")
        print(f"    形状: {combined.shape}")
        print(f"    平均曲率: {curvature.mean().item():.4f}")
        print(f"    最大曲率: {curvature.max().item():.4f}")
        print(f"    PCA Top-3 方差占比: {(S[:3].sum() / S.sum()).item():.4f}")
    
    # 测试热核扩散
    print("\n[4] 测试热核扩散...")
    
    if "layer_6" in all_activations:
        reference_tokens = torch.cat(
            [act.reshape(-1, act.size(-1)) for act in all_activations["layer_6"]],
            dim=0,
        )
        flat_ref = reference_tokens[:50]
        
        test_act = all_activations["layer_6"][0]
        diffused = heat_diffuser.diffuse(test_act, flat_ref)
        
        diff_diff = torch.norm(diffused - test_act).item()
        print(f"  扩散前后差异: {diff_diff:.4f}")
    
    # 测试测地线引导
    print("\n[5] 测试测地线引导...")
    
    if "layer_6" in all_activations:
        test_act = all_activations["layer_6"][0]
        
        # 设置随机方向
        geodesic_steerer.set_direction("test", torch.randn(768))
        steered = geodesic_steerer.steer(test_act, "test", 0.05)
        
        steer_diff = torch.norm(steered - test_act).item()
        print(f"  引导前后差异: {steer_diff:.4f}")
    
    # 生成对比测试
    print("\n[6] 生成对比测试...")
    
    test_prompt = "The meaning of life is"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    # 原始生成
    with torch.no_grad():
        original_output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    original_text = tokenizer.decode(original_output[0], skip_special_tokens=True)
    print(f"\n  原始: {original_text}")
    
    # 创建干预钩子
    intervention_applied = [False]
    reference_tensor = flat_ref if "layer_6" in all_activations else None
    
    def intervention_hook(module, input, output):
        if reference_tensor is None or intervention_applied[0]:
            return output
        
        intervention_applied[0] = True
        
        if isinstance(output, tuple):
            act = output[0]
            diffused = heat_diffuser.diffuse(act, reference_tensor)
            return (diffused,) + output[1:]
        return output
    
    # 应用干预
    hook_handle = model.transformer.h[6].register_forward_hook(intervention_hook)
    
    with torch.no_grad():
        intervened_output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    intervened_text = tokenizer.decode(intervened_output[0], skip_special_tokens=True)
    print(f"  干预后: {intervened_text}")
    
    hook_handle.remove()
    
    # 保存结果
    results = {
        "model": model_name,
        "test_prompt": test_prompt,
        "original_output": original_text,
        "intervened_output": intervened_text,
        "analysis": analysis
    }
    
    os.makedirs("tempdata", exist_ok=True)
    save_path = "tempdata/geometric_intervention_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] 结果保存到: {save_path}")
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    test_geometric_intervention()
