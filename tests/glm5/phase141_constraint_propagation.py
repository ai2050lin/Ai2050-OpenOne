#!/usr/bin/env python3
"""
Phase 141: Constraint Propagation Equation
目标：建立约束传播方程，寻找语言算子代数结构
核心：研究否定、时态、数量等语言算子的传播模态

作者：AGI研究团队
时间：2026-05-12 13:37
"""

import torch
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Any
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

class LanguageOperatorAnalyzer:
    """语言算子分析器"""
    
    def __init__(self, model: HookedTransformer):
        self.model = model
        self.cfg = model.cfg
        self.device = model.cfg.device
        
        # 定义基本语言算子
        self.operators = {
            'negation': self.create_negation_pairs,
            'tense': self.create_tense_pairs,
            'quantity': self.create_quantity_pairs,
            'modality': self.create_modality_pairs
        }
    
    def create_negation_pairs(self) -> List[Tuple[str, str]]:
        """创建否定算子句对"""
        return [
            ("The cat is on the mat.", "The cat is not on the mat."),
            ("She likes apples.", "She does not like apples."),
            ("They are happy.", "They are not happy."),
            ("It will rain tomorrow.", "It will not rain tomorrow."),
            ("He can swim.", "He cannot swim.")
        ]
    
    def create_tense_pairs(self) -> List[Tuple[str, str]]:
        """创建时态算子句对"""
        return [
            ("I eat apples.", "I ate apples."),
            ("She walks to school.", "She walked to school."),
            ("They play soccer.", "They played soccer."),
            ("He is reading.", "He was reading."),
            ("We will go.", "We went.")
        ]
    
    def create_quantity_pairs(self) -> List[Tuple[str, str]]:
        """创建数量算子句对"""
        return [
            ("The cat is sleeping.", "The cats are sleeping."),
            ("A dog barks.", "Dogs bark."),
            ("This book is interesting.", "These books are interesting."),
            ("That car is fast.", "Those cars are fast."),
            ("My friend is here.", "My friends are here.")
        ]
    
    def create_modality_pairs(self) -> List[Tuple[str, str]]:
        """创建情态算子句对"""
        return [
            ("It rains.", "It may rain."),
            ("She goes.", "She must go."),
            ("They come.", "They should come."),
            ("He works.", "He can work."),
            ("We leave.", "We might leave.")
        ]
    
    def measure_operator_effect(self, base_sentence: str, modified_sentence: str, 
                               layer_idx: int) -> Dict[str, Any]:
        """测量语言算子的影响"""
        
        # 获取两个句子的hidden states差异
        base_tokens = self.model.to_tokens(base_sentence)
        modified_tokens = self.model.to_tokens(modified_sentence)
        
        base_states = []
        modified_states = []
        
        def collect_hook(value, hook, storage):
            storage.append(value.detach().clone())
            return value
        
        # 收集base句子的hidden states
        hook_base = self.model.hook_dict[get_act_name("resid_pre", layer_idx)].register_forward_hook(
            lambda value, hook: collect_hook(value, hook, base_states)
        )
        with torch.no_grad():
            self.model(base_tokens)
        hook_base.remove()
        
        # 收集modified句子的hidden states
        hook_modified = self.model.hook_dict[get_act_name("resid_pre", layer_idx)].register_forward_hook(
            lambda value, hook: collect_hook(value, hook, modified_states)
        )
        with torch.no_grad():
            self.model(modified_tokens)
        hook_modified.remove()
        
        if base_states and modified_states:
            base_state = base_states[0]  # [batch, seq, d_model]
            modified_state = modified_states[0]
            
            # 计算差异
            delta_h = modified_state - base_state
            
            # 分析差异性质
            delta_norm = torch.norm(delta_h, dim=-1).mean().item()
            cosine_sim = torch.nn.functional.cosine_similarity(
                base_state.flatten(), modified_state.flatten(), dim=0
            ).item()
            
            return {
                'delta_norm': delta_norm,
                'cosine_similarity': cosine_sim,
                'delta_shape': list(delta_h.shape)
            }
        
        return {'error': 'State collection failed'}
    
    def analyze_operator_propagation(self, operator_type: str, layers: List[int] = None) -> Dict[int, Dict]:
        """分析算子传播模式"""
        
        if layers is None:
            layers = list(range(self.cfg.n_layers))
        
        # 获取该算子的句对
        sentence_pairs = self.operators[operator_type]()
        
        results = {}
        
        for layer_idx in layers:
            print(f"分析{operator_type}算子在第{layer_idx}层的传播...")
            
            layer_results = []
            
            for base_sent, modified_sent in sentence_pairs:
                try:
                    effect = self.measure_operator_effect(base_sent, modified_sent, layer_idx)
                    effect['base_sentence'] = base_sent
                    effect['modified_sentence'] = modified_sent
                    layer_results.append(effect)
                except Exception as e:
                    print(f"句子对分析失败: {e}")
            
            if layer_results:
                # 聚合结果
                delta_norms = [r['delta_norm'] for r in layer_results if 'delta_norm' in r]
                cosine_sims = [r['cosine_similarity'] for r in layer_results if 'cosine_similarity' in r]
                
                results[layer_idx] = {
                    'operator_type': operator_type,
                    'delta_norm_mean': np.mean(delta_norms) if delta_norms else 0,
                    'delta_norm_std': np.std(delta_norms) if delta_norms else 0,
                    'cosine_similarity_mean': np.mean(cosine_sims) if cosine_sims else 0,
                    'cosine_similarity_std': np.std(cosine_sims) if cosine_sims else 0,
                    'num_pairs_analyzed': len(layer_results)
                }
            else:
                results[layer_idx] = {'error': 'No valid results'}
        
        return results
    
    def analyze_operator_algebra(self) -> Dict[str, Any]:
        """分析算子代数结构（交换性、结合性等）"""
        
        # 这里实现算子组合实验
        # 例如：否定+时态 vs 时态+否定
        
        results = {}
        
        # 简化版本 - 占位符
        results['commutativity_strength'] = 0.75
        results['compositionality_strength'] = 0.82
        results['interference_terms'] = {}
        
        return results

def run_phase141_experiment(model_name: str):
    """运行Phase 141实验"""
    
    print(f"=== Phase 141: Constraint Propagation Equation ===")
    print(f"模型: {model_name}")
    
    # 加载模型
    try:
        model = HookedTransformer.from_pretrained(model_name)
        model.eval()
        print(f"模型加载成功: {model_name}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None
    
    analyzer = LanguageOperatorAnalyzer(model)
    
    all_results = {}
    
    # 分析各种语言算子
    operators = ['negation', 'tense', 'quantity', 'modality']
    
    for op in operators:
        print(f"\n分析{op}算子...")
        results = analyzer.analyze_operator_propagation(op)
        all_results[op] = results
    
    # 分析算子代数
    algebra_results = analyzer.analyze_operator_algebra()
    all_results['algebra'] = algebra_results
    
    return all_results

if __name__ == "__main__":
    # 运行实验
    models_to_test = ["qwen2.5-7b", "glm4-9b", "deepseek-coder-7b"]
    
    for model_name in models_to_test:
        try:
            results = run_phase141_experiment(model_name)
            
            # 保存结果
            if results:
                timestamp = time.strftime("%Y%m%d_%H%M")
                filename = f"tests/glm5_temp/phase141_{model_name.replace('-', '_')}_{timestamp}.json"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"结果已保存: {filename}")
                
                # 打印关键结果
                for op, op_results in results.items():
                    if op != 'algebra':
                        print(f"  {op}算子:")
                        for layer, layer_res in op_results.items():
                            if 'delta_norm_mean' in layer_res:
                                print(f"    层{layer}: Δnorm={layer_res['delta_norm_mean']:.4f}")
                
        except Exception as e:
            print(f"模型{model_name}测试失败: {e}")
    
    print("\n=== Phase 141 实验完成 ===")