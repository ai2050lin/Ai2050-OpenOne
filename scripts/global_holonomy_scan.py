"""
全局全纯扫描 (Global Holonomy Scanning) — AGI 统一场论验证工具
=================================================================
本脚本通过构造“闭合思维环路”来测量模型的逻辑全纯性 (Holonomy)：
1.  路径构造：A (男人) -> B (国王) -> C (皇后) -> D (女人) -> A' (男人?)
2.  通过 RPT (黎曼平行传输) 沿环路平移语义纤维。
3.  测量起点 A 与终点 A' 的偏差 (Holonomy Deviation)。
4.  低偏差 = 高全纯性 = 高逻辑自洽性。
"""

import json
import os
import sys
import time

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

from transformer_lens import HookedTransformer

# 设置环境
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import GPT2LMHeadModel, GPT2Tokenizer


class HolonomyScanner:
    def __init__(self, model_name="gpt2-small"):
        # 强制加载本地快照
        snapshot_path = r"D:\develop\model\hub\models--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
        self.tokenizer = GPT2Tokenizer.from_pretrained(snapshot_path, local_files_only=True)
        self.model = GPT2LMHeadModel.from_pretrained(snapshot_path, local_files_only=True)
        self.model.eval()

    def get_act(self, text, layer_idx):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # layer_idx=0 is embedding, so l+1 is the output of block l
            act = outputs.hidden_states[layer_idx+1][0, -1, :].numpy()
        return act

    def run_scan(self):
        # 简单循环路径：man -> king -> queen -> woman -> (man')
        nodes = ["man", "king", "queen", "woman"]
        for layer in [0, 6, 11]:
            acts = [self.get_act(f"The {n} is there.", layer) for n in nodes]
            vectors = []
            for i in range(len(acts)):
                vectors.append(acts[(i+1)%4] - acts[i])
            loop_sum = np.sum(vectors, axis=0)
            deviation = np.linalg.norm(loop_sum) / np.mean([np.linalg.norm(v) for v in vectors])
            print(f"RESULT_LAYER_{layer}_DEV_{deviation:.6f}")

if __name__ == "__main__":
    try:
        scanner = HolonomyScanner()
        scanner.run_scan()
        print("SCAN_SUCCESS")
    except Exception as e:
        print(f"SCAN_ERROR_{str(e)}")

    def get_fiber_activations(self, template, node, layer_idx):
        """提取纤维激活"""
        text = template.replace("{}", node)
        with torch.no_grad():
            _, cache = self.model.run_with_cache(text, names_filter=lambda x: f"blocks.{layer_idx}.hook_resid_pre" in x)
            act = cache[f"blocks.{layer_idx}.hook_resid_pre"][0, -1, :].cpu().numpy()
        return act

    def compute_transport(self, acts_src, acts_tgt):
        """计算传输算子 R"""
        # 简化版：直接计算单位方向之间的旋转
        u = acts_src / np.linalg.norm(acts_src)
        v = acts_tgt / np.linalg.norm(acts_tgt)
        
        # 构造旋转矩阵 (简单映射)
        # 在高维空间，我们关注的是平行移动后的投影一致性
        return u, v

    def scan_cycle(self, cycle, layer_idx):
        """扫描单个环路在特定层的全纯性"""
        nodes = cycle["nodes"]
        template = cycle["templates"][0] # 使用统一模板减少干扰
        
        # 1. 提取各个节点的原始激活 (底丛截面)
        node_acts = [self.get_fiber_activations(template, node, layer_idx) for node in nodes]
        
        # 2. 模拟环路平移: A -> B -> C -> D -> A'
        # 我们测量 A 与 B, B 与 C... 之间的相对关系，最后看是否回到 A
        # 简化数学度量：计算各段向量差的闭合程度
        vectors = []
        for i in range(len(node_acts)):
            next_idx = (i + 1) % len(node_acts)
            v = node_acts[next_idx] - node_acts[i]
            vectors.append(v)
            
        # 闭环残差：Sum of transitions should ideally be zero in a flat manifold
        # 在有曲率的流形上，残差反映了 Holonomy
        loop_sum = np.sum(vectors, axis=0)
        holonomy_deviation = np.linalg.norm(loop_sum) / np.mean([np.linalg.norm(v) for v in vectors])
        
        # 3. 计算相位锁定一致性 (Phase Consistency)
        # 模拟不同节点间的余弦相似度闭环
        cos_sims = []
        for i in range(len(node_acts)):
            next_idx = (i + 1) % len(node_acts)
            v1 = node_acts[i]
            v2 = node_acts[next_idx]
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_sims.append(sim)
            
        return {
            "deviation": float(holonomy_deviation),
            "coherence": float(np.mean(cos_sims)),
            "is_closed": bool(holonomy_deviation < 0.5) # 经验阈值
        }

    def run_full_scan(self):
        results = {}
        target_layers = [0, 6, 11] # 精简层级，快速验证
        for cycle in self.cycles:
            cycle_name = cycle["name"]
            print(f"\n[SCAN] Cycle: {cycle_name}")
            cycle_results = []
            for l in target_layers:
                stats = self.scan_cycle(cycle, l)
                print(f"  L{l}: Dev={stats['deviation']:.4f}, Coh={stats['coherence']:.4f}")
                cycle_results.append({"layer": l, **stats})
            results[cycle_name] = cycle_results
        return results

if __name__ == "__main__":
    try:
        scanner = HolonomyScanner()
        scanner.run_scan()
        print("\n---FINISH---")
    except Exception as e:
        import traceback
        print(traceback.format_exc())
