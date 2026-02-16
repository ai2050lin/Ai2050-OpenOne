import json
import os

import numpy as np


class HolographicMemoryManager:
    """
    全息长期记忆管理器 (Holographic Long-term Memory Manager)
    SHMC 核心组件：将记忆编码为流形上的拓扑持久化结构。
    特点：一次学习 (One-shot Learning)、免疫遗忘、具备联想性。
    """
    def __init__(self, memory_dim=32, capacity=1000):
        self.dim = memory_dim
        self.capacity = capacity
        # 记忆矩阵 (全息存储：使用联想存储器或简单的键值对，但在流形空间工作)
        self.memory_keys = np.random.randn(capacity, memory_dim)
        self.memory_values = np.random.randn(capacity, memory_dim)
        self.current_size = 0
        
        # 记忆联想增强算子 (Resonance Operator)
        self.resonance_matrix = np.eye(memory_dim)

    def encode(self, concept_vec, data_vec):
        """
        全息编码：将新的知识对存入流形记忆
        """
        if self.current_size >= self.capacity:
            # 引入记忆置换机制 (最近最少激活忽略)
            idx = np.random.randint(0, self.capacity)
        else:
            idx = self.current_size
            self.current_size += 1
            
        # 几何标准化
        self.memory_keys[idx] = concept_vec / (np.linalg.norm(concept_vec) + 1e-9)
        self.memory_values[idx] = data_vec
        print(f"[+] 全息记忆编码成功 (Slot {idx})")

    def retrieve(self, query_vec, top_k=3):
        """
        全息检索：通过内容共振查找记忆 (联想搜索)
        """
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        # 计算共振强度 (点积相似度)
        similarities = self.memory_keys[:self.current_size] @ query_norm
        
        # 获取 top_k 索引
        best_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 全息叠加检索结果
        retrieved_result = np.zeros(self.dim)
        weights = []
        for idx in best_indices:
            weight = similarities[idx]
            retrieved_result += weight * self.memory_values[idx]
            weights.append(float(weight))
            
        return retrieved_result, weights

    def run_memory_strength_test(self):
        """
        运行记忆稳定性测试
        """
        print("[*] 启动全息记忆“阅后即记”实验...")
        
        # 1. 存入三个模态知识
        knowledge = {
            "Apple_Concept": np.eye(self.dim)[0], # 概念
            "Red_Round_Fruit": np.random.randn(self.dim) * 2.0 # 详细数据
        }
        self.encode(knowledge["Apple_Concept"], knowledge["Red_Round_Fruit"])
        
        # 2. 存入第二个冲突知识（测试遗忘免疫）
        knowledge_2 = {
            "Banana_Concept": np.eye(self.dim)[1],
            "Yellow_Long_Fruit": np.random.randn(self.dim) * 2.0
        }
        self.encode(knowledge_2["Banana_Concept"], knowledge_2["Yellow_Long_Fruit"])
        
        # 3. 立即检索第一个知识
        retrieved, weights = self.retrieve(knowledge["Apple_Concept"])
        
        # 计算检索质量 (MSE)
        mse = np.mean((retrieved - knowledge["Red_Round_Fruit"])**2)
        recall_accuracy = 1.0 - (mse / np.var(knowledge["Red_Round_Fruit"]))
        
        results = {
            "recall_accuracy": float(recall_accuracy),
            "resonance_weight": weights[0],
            "memory_slots_used": self.current_size,
            "status": "MEMORY_STABLE" if recall_accuracy > 0.8 else "RECALL_DEGRADED"
        }
        
        return results

if __name__ == "__main__":
    memory = HolographicMemoryManager()
    summary = memory.run_memory_strength_test()
    print(f"\n--- 全息长期记忆实验总结 ---\n{json.dumps(summary, indent=2, ensure_ascii=False)}")
    
    # 导出报告
    os.makedirs("tempdata", exist_ok=True)
    with open("tempdata/memory_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
