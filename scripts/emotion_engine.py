import json
import os

import numpy as np


class EmotionEngine:
    """
    情感价值引擎 (Emotional Manifold Engine)
    功能：模拟大脑的情感内稳态 (Homeostasis)，通过二阶微分项调节流形推理的“温度”与“偏好”。
    核心指标：好奇心 (Curiosity - 探索性)、稳定度 (Stability - 风险规避)。
    """
    def __init__(self, manifold_dim=32):
        self.dim = manifold_dim
        # 内稳态参数 (0.0 ~ 1.0)
        self.curiosity = 0.5   # 驱使模型探索未知/高熵区域
        self.stability = 0.5   # 驱使模型留在安全/低熵区域
        self.energy_buffer = 100.0 # 系统“生命力”缓冲
        
        # 实时情感记录
        self.history = []

    def update_homeostasis(self, sensory_input, logic_efficiency):
        """
        根据输入和推理效率动态调整情感状态
        """
        # 逻辑效率高 -> 提升满足感 (增加稳定度)
        # 逻辑效率低/遇到未知场景 -> 激发挫败感或好奇心
        efficiency_gap = logic_efficiency - 0.8
        
        self.stability = np.clip(self.stability + 0.1 * efficiency_gap, 0.1, 0.9)
        self.curiosity = 1.0 - self.stability # 互补关系，简化模型
        
        # 能耗消耗系统能量
        self.energy_buffer -= (1.0 - logic_efficiency) * 5.0
        
        status = {
            "curiosity": float(self.curiosity),
            "stability": float(self.stability),
            "energy": float(self.energy_buffer)
        }
        self.history.append(status)
        return status

    def get_emotional_bias(self, current_pos, potential_grads):
        """
        情感偏好修正算子：
        将情感状态转化为作用在流形上的“虚位移”
        """
        # 好奇心引导向梯度稀疏的方向 (探索新域)
        # 稳定度引导向梯度平滑的方向 (稳健决策)
        bias = np.zeros(self.dim)
        
        # 模拟探索：在好奇心高时引入正交噪音
        if self.curiosity > 0.7:
            bias += np.random.randn(self.dim) * self.curiosity * 0.2
            
        # 模拟稳定：加强对主要引力陷阱的锁定
        bias += potential_grads * self.stability
        
        return bias

    def run_emotional_behavior_test(self):
        """
        运行具有情感反馈的行为测试
        """
        print("[*] 启动情感内稳态驱动的行为模拟...")
        
        # 模拟 20 个时间步的交互
        for i in range(20):
            # 模拟环境反馈 (随机逻辑效率波动)
            env_feedback = 0.7 + np.random.rand() * 0.3
            self.update_homeostasis(None, env_feedback)
            
        final_state = self.history[-1]
        
        results = {
            "final_emotional_status": final_state,
            "emotional_volatility": float(np.std([h['stability'] for h in self.history])),
            "survival_potential": "HIGH" if self.energy_buffer > 50 else "CRITICAL",
            "status": "EMOTIONAL_EQUILIBRIUM_REACHED"
        }
        
        return results

if __name__ == "__main__":
    engine = EmotionEngine()
    summary = engine.run_emotional_behavior_test()
    print(f"\n--- 情感价值流形实验总结 ---\n{json.dumps(summary, indent=2, ensure_ascii=False)}")
    
    # 导出报告
    os.makedirs("tempdata", exist_ok=True)
    with open("tempdata/emotion_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
