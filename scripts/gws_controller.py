import json
import os

import numpy as np


class GWSController:
    """
    全局工作空间控制器 (Global Workspace Controller)
    AGI 的意识中枢：负责在不同流形模态之间进行竞争性激活与全局广播。
    """
    def __init__(self, workspace_dim=64):
        self.dim = workspace_dim
        # 全局工作空间状态 (GWS Session)
        self.state = np.zeros(workspace_dim)
        # 注册的模块模态
        self.modules = {} # {module_name: projection_to_gws}
        # 激活注意力 (Salience Map)
        self.salience = {}

    def register_module(self, name, local_dim):
        """注册一个子系统模块并建立其到 GWS 的投影关系"""
        # 随机初始化投影矩阵
        projection = np.random.randn(self.dim, local_dim) * 0.1
        self.modules[name] = projection
        self.salience[name] = 0.0
        print(f"[+] 模块已连接至 GWS: {name} (Dim: {local_dim})")

    def compete(self, module_signals):
        """
        竞争机制：不同模块的信号竞争进入全局工作空间
        module_signals: {module_name: local_signal_vec}
        """
        print("[*] 启动 GWS 竞争性激活进程...")
        total_energy = 0
        winning_module = None
        max_energy = -1
        
        # 1. 计算每个模块的“显著性”评分 (能量强度)
        for name, signal in module_signals.items():
            if name not in self.modules: continue
            
            # 投射到全局空间
            gws_projection = self.modules[name] @ signal
            energy = np.linalg.norm(gws_projection)
            self.salience[name] = float(energy)
            
            if energy > max_energy:
                max_energy = energy
                winning_module = name
        
        # 2. 全局广播 (Broadcast)：胜出模块的信号主导 GWS 状态
        if winning_module:
            winner_signal = self.modules[winning_module] @ module_signals[winning_module]
            # 引入惯性，平滑意识流
            self.state = 0.7 * self.state + 0.3 * winner_signal
            print(f"[!] 竞争胜出者: {winning_module} -> 成功广播至全局工作空间")
            
        return winning_module

    def get_broadcast_effect(self, target_module_name):
        """查询全局状态对特定子系统的广播反馈"""
        if target_module_name not in self.modules: return None
        # 反向投射：GWS -> Local (简化为转置映射)
        back_projection = self.modules[target_module_name].T @ self.state
        return back_projection

    def run_consciousness_test(self):
        """运行 GWS 意识协调测试"""
        # 模拟两个竞争模态
        self.register_module("Vision_Stream", 32)
        self.register_module("Logic_Reasoning", 16)
        
        # 场景 1：视觉刺激较强
        signals_v = {
            "Vision_Stream": np.random.randn(32) * 5.0, # 强信号
            "Logic_Reasoning": np.random.randn(16) * 1.0 # 弱信号
        }
        winner_1 = self.compete(signals_v)
        
        # 场景 2：逻辑推演爆发
        signals_l = {
            "Vision_Stream": np.random.randn(32) * 1.0,
            "Logic_Reasoning": np.random.randn(16) * 8.0 # 强信号
        }
        winner_2 = self.compete(signals_l)
        
        results = {
            "winners": [winner_1, winner_2],
            "final_gws_energy": float(np.linalg.norm(self.state)),
            "module_salience": self.salience,
            "system_integration": "SUCCESSFUL" if winner_1 != winner_2 else "DOMINANCE_FAILURE"
        }
        
        return results

if __name__ == "__main__":
    gws = GWSController()
    summary = gws.run_consciousness_test()
    print(f"\n--- GWS 意识载体测试总结 ---\n{json.dumps(summary, indent=2, ensure_ascii=False)}")
    
    # 导出报告
    os.makedirs("tempdata", exist_ok=True)
    with open("tempdata/gws_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
