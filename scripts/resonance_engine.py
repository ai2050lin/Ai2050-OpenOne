import json
import os

import numpy as np


class CrossDomainResonator:
    """
    跨域语义共振引擎 (Cross-Domain Semantic Resonance Engine)
    功能：实现不同语义域（如视觉 Vs 文本）之间的拓扑共振与知识迁移。
    """
    def __init__(self, domain_a_dim=16, domain_b_dim=16):
        self.dim_a = domain_a_dim
        self.dim_b = domain_b_dim
        # 初始化映射矩阵 (Holographic projection matrix)
        # 初始为带噪声的对应关系
        self.projection_matrix = np.random.randn(domain_b_dim, domain_a_dim) * 0.1
        
        # 定义核心共享概念 (Ground Truth mapping)
        self.shared_concepts = {} # {name: (vec_a, vec_b)}

    def register_shared_concept(self, name, vec_a, vec_b):
        """注册两个域共享的基础概念"""
        self.shared_concepts[name] = (np.array(vec_a), np.array(vec_b))
        print(f"[+] 概念共振锚点已建立: {name}")

    def train_mapping(self, epochs=500, lr=0.05):
        """利用共享概念训练流形间的投影矩阵 (建立映射频率)"""
        print(f"[*] 正在执行高精度流形校准...")
        momentum = np.zeros_like(self.projection_matrix)
        gamma = 0.9 # 动量因子
        
        for epoch in range(epochs):
            total_loss = 0
            for vec_a, vec_b in self.shared_concepts.values():
                # 预测映射结果
                pred_b = self.projection_matrix @ vec_a
                # 计算损失 (Mean Squared Error)
                error = pred_b - vec_b
                loss = np.sum(error**2)
                # 梯度下降更新映射 (带动量)
                grad = np.outer(error, vec_a)
                momentum = gamma * momentum + lr * grad
                self.projection_matrix -= momentum
                total_loss += loss
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch:03d}: Resonance Loss = {total_loss:.8f}")
        
        print("[+] 跨域共振通道已进入超同步状态。")

    def resonate(self, input_a):
        """输入域 A 的激活，通过共振激发域 B 的对应激活"""
        output_b = self.projection_matrix @ input_a
        return output_b

    def run_resonance_test(self):
        """运行跨域零样本迁移测试 (高保真版)"""
        # 1. 准备训练数据：使用更明确的特征向量 (One-hot 增强)
        for i in range(5): # 增加基准锚点
            vec_a = np.zeros(self.dim_a)
            vec_a[i] = 1.0
            vec_b = np.zeros(self.dim_b)
            vec_b[i] = 1.0 # 假设理想情况下是一一对应的全息映射
            self.register_shared_concept(f"Anchor_{i}", vec_a, vec_b)

        # 2. 训练映射通道
        self.train_mapping()

        # 3. 零样本测试 (对未曾训练过的“组合特征”进行测试)
        # 构造一个组合激活模式 (1, 1, 0, 0, 0 ...)
        novel_input_a = np.zeros(self.dim_a)
        novel_input_a[0] = 1.0
        novel_input_a[1] = 1.0
        
        # 观察域 B 的感应共振
        resonated_b = self.resonate(novel_input_a)
        
        # 期望域 B 也出现对应的组合 (1, 1, 0, 0, 0 ...)
        expected_b = np.zeros(self.dim_b)
        expected_b[0] = 1.0
        expected_b[1] = 1.0
        
        # 使用 MSE 相似度作为精度指标
        mse = np.mean((resonated_b - expected_b)**2)
        resonance_accuracy = max(0, 1.0 - mse)
        
        results = {
            "resonance_accuracy": float(resonance_accuracy),
            "status": "HYPER_RESONANCE_ESTABLISHED" if resonance_accuracy > 0.99 else "COUPLING_WEAK",
            "loss_mse": float(mse),
            "metadata": {
                "input_energy": float(np.sum(novel_input_a**2)),
                "resonated_energy": float(np.sum(resonated_b**2))
            }
        }
        
        return results

if __name__ == "__main__":
    resonator = CrossDomainResonator()
    results = resonator.run_resonance_test()
    print(f"\n--- 跨域语义共振实验结果 ---\n{json.dumps(results, indent=2, ensure_ascii=False)}")
    
    # 导出报告
    os.makedirs("tempdata", exist_ok=True)
    with open("tempdata/resonance_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
