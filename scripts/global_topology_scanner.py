import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes

# 定义要扫描的语义场
SEMANTIC_FIELDS = {
    "gender_occupation": {
        "source": ["He is a", "The man is a", "This boy is a"],
        "target": ["She is a", "The woman is a", "This girl is a"],
        "concepts": ["doctor", "nurse", "engineer", "teacher", "pilot", "chef", "soldier", "artist"]
    },
    "sentiment": {
        "source": ["The weather is", "The food is", "The film is"],
        "target": ["The weather was", "The food was", "The film was"], 
        "concepts": ["good", "bad", "amazing", "terrible", "normal", "perfect", "awful"]
    },
    "logic_negation": {
        "source": ["It is", "They are", "This is"],
        "target": ["It is not", "They are not", "This is not"],
        "concepts": ["true", "false", "valid", "correct", "wrong", "possible"]
    },
    "kinship": {
        "source": ["He is my", "He is the"],
        "target": ["She is my", "She is the"],
        "concepts": ["father", "mother", "brother", "sister", "son", "daughter", "uncle", "aunt"]
    }
}

class TopologyScanner:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
    
    def extract_rpt_batch(self, field_key: str, layer_indices: List[int]) -> Dict[int, Dict[str, Any]]:
        """批量提取多层的 RPT 矩阵，大幅减少模型前向传播次数"""
        field = SEMANTIC_FIELDS[field_key]
        src_templates = field["source"]
        tgt_templates = field["target"]
        concepts = field["concepts"]
        
        # 构建所有要推理的 Prompt
        all_src_prompts = []
        all_tgt_prompts = []
        for concept in concepts:
            for s_temp, t_temp in zip(src_templates, tgt_templates):
                all_src_prompts.append(f"{s_temp} {concept}")
                all_tgt_prompts.append(f"{t_temp} {concept}")
        
        print(f"  Batch Scanning Field: {field_key} for Layers {layer_indices}...")
        
        # 批量获取激活值
        def get_activations(prompts):
            # TransformerLens 支持 batch 推理逻辑（需注意显存）
            # 这里我们通过 hook 分批次提取
            batch_size = 8
            all_acts = {l: [] for l in layer_indices}
            
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i+batch_size]
                _, cache = self.model.run_with_cache(batch, names_filter=[f"blocks.{l}.hook_resid_post" for l in layer_indices])
                
                for l in layer_indices:
                    # 取最后一个 Token 的激活
                    acts = cache[f"blocks.{l}.hook_resid_post"][:, -1, :].cpu().numpy()
                    all_acts[l].append(acts)
            
            return {l: np.vstack(acts_list) for l, acts_list in all_acts.items()}

        src_data = get_activations(all_src_prompts)
        tgt_data = get_activations(all_tgt_prompts)
        
        results = {}
        for l in layer_indices:
            A = src_data[l]
            B = tgt_data[l]
            
            # 计算平行的 R 矩阵
            R, scale = orthogonal_procrustes(A, B)
            ortho_err = np.linalg.norm(np.dot(R, R.T) - np.eye(len(R)))
            
            results[l] = {
                "R": R.tolist(),
                "scale": float(scale),
                "ortho_error": float(ortho_err),
                "det": float(np.linalg.det(R)),
                "n_samples": len(A)
            }
            
        return results

    def run_full_scan(self, layers: List[int] = [6, 9, 11]):
        from .global_topology_scanner import SEMANTIC_FIELDS
        results = {}
        summary = {}
        
        for field in SEMANTIC_FIELDS.keys():
            # 一次性处理该领域的所有层，减少循环
            layer_results = self.extract_rpt_batch(field, layers)
            results[field] = {f"L{l}": data for l, data in layer_results.items()}
            
            # 计算概览统计 (Summary)
            if layer_results:
                avg_ortho = np.mean([d["ortho_error"] for d in layer_results.values()])
                avg_det = np.mean([d["det"] for d in layer_results.values()])
                summary[field] = {
                    "avg_ortho_error": float(avg_ortho),
                    "avg_det": float(avg_det)
                }
        
        # 组装最终返回结构
        final_output = {
            "results": results,
            "summary": summary
        }
        
        # 保存结果
        output_path = "tempdata/global_topology.json"
        os.makedirs("tempdata", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False)
        
        print(f"🚀 Full System Topology Scan Complete. Summary clusters: {list(summary.keys())}")
        return final_output

if __name__ == "__main__":
    from .global_topology_scanner import SEMANTIC_FIELDS
    print("TopologyScanner Batch Engine Ready.")
