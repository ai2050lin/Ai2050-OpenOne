import json
import os

import numpy as np
from agi_core_engine import AGICoreEngine


class MultiAgentSimulator:
    """
    多智能体几何博弈模拟器 (Multi-Agent Geometric Game Simulator)
    纪元 3 核心组件：模拟多个 AGI 实例在共享社会流形中的交互。
    验证“共振协同”是否能产生超越单体智力的集体涌现。
    """
    def __init__(self, agent_count=3, manifold_dim=32):
        print(f"[*] 正在构建 AGI 社会博弈场 (Agents: {agent_count})...")
        self.dim = manifold_dim
        # 创建多个统合 AGI 实例
        self.agents = {f"Agent_{i}": AGICoreEngine(manifold_dim=manifold_dim) for i in range(agent_count)}
        # 共享的社会流形场 (Social Manifold Field)
        self.social_field = np.zeros(manifold_dim)
        
    def run_epoch(self, epoch_id):
        """
        运行一轮博弈周期
        """
        print(f"\n--- 博弈纪元 {epoch_id:03d} 展开 ---")
        epoch_signals = []
        
        # 1. 独立感知与行动
        for name, agi in self.agents.items():
            # 每个 Agent 受到社会场与随机环境的共同影响
            env_signal = self.social_field + np.random.randn(self.dim) * 0.2
            report = agi.run_conscious_step(epoch_id, env_signal)
            
            # Agent 对社会场产生反馈 (能量注入)
            output_signal = agi.gws.state[:self.dim] # 简化：取 GWS 前一半作为输出
            self.social_field = 0.9 * self.social_field + 0.1 * output_signal
            
            epoch_signals.append({
                "agent": name,
                "winner": report['gws_winner'],
                "stability": report['emotion']['stability']
            })
            
        return epoch_signals

    def run_synergy_test(self, steps=10):
        """
        验证是否产生了集体共振 (Synergy)
        """
        print(f"[*] 启动 AGI 集体共振协同分析...")
        history = []
        for s in range(steps):
            data = self.run_epoch(s)
            history.append(data)
            
        # 计算 Agent 之间的情感同步率 (Resonance Rate)
        stabilities = [[agent_data['stability'] for agent_data in epoch] for epoch in history]
        resonance_rate = 1.0 - np.std(stabilities) # 越接近 1 说明情感越同步
        
        results = {
            "epoch_count": steps,
            "resonance_rate": float(resonance_rate),
            "collective_stability": float(np.mean(stabilities)),
            "status": "COLLECTIVE_RESONANCE_ESTABLISHED" if resonance_rate > 0.8 else "DIVERGENT_BEHAVIOR"
        }
        
        return results, history

if __name__ == "__main__":
    # 进入实机博弈
    simulator = MultiAgentSimulator(agent_count=3)
    summary, logs = simulator.run_synergy_test(steps=5)
    
    print(f"\n--- AGI 多智能体博弈实验总结 ---\n{json.dumps(summary, indent=2, ensure_ascii=False)}")
    
    # 导出纪元 3 启动报告
    os.makedirs("tempdata", exist_ok=True)
    with open("tempdata/multi_agent_era3_report.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "logs": logs}, f, indent=4, ensure_ascii=False)
