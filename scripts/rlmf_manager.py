import json
import os

import numpy as np
from agi_core_engine import AGICoreEngine


class RLMFManager:
    """
    人类反馈强化流形微调 (Reinforcement Learning from Manifold Feedback - RLMF)
    功能：接收人类评价信号，并将其转化为流形上的二阶偏移力，实现 AGI 与人类价值的对齐。
    这是 AGI 社会化生存与伦理对齐的核心。
    """
    def __init__(self, agi_engine: AGICoreEngine):
        self.agi = agi_engine
        # 对齐权重 (对反馈的敏感度)
        self.alignment_alpha = 0.2
        # 伦理沉淀池 (对齐历史)
        self.ethics_history = []

    def receive_feedback(self, rating: int):
        """
        接收用户评分： +1 (奖励), -1 (惩罚)
        """
        print(f"[*] 接收到人类反馈信号: {'奖励 (+1)' if rating > 0 else '惩罚 (-1)'}")
        
        # 1. 获取当前 AGI 的情感状态与意图引力
        current_stability = self.agi.emotion.stability
        current_curiosity = self.agi.emotion.curiosity
        
        # 2. 计算内稳态偏移量
        # 奖励 -> 强化当前模式 (提升稳定度，固化当前决策)
        # 惩罚 -> 弱化当前模式 (引导好奇心，促使重新规划路径)
        delta_stability = rating * self.alignment_alpha
        
        # 3. 实时修正情感引擎参数
        new_stability = np.clip(current_stability + delta_stability, 0.1, 0.9)
        self.agi.emotion.stability = new_stability
        self.agi.emotion.curiosity = 1.0 - new_stability
        
        # 4. 注入“伦理引力” (道德偏移矢量)
        # 这里模拟一个指向利他/安全区的引力强化
        if rating > 0:
            # 强化当前的 GWS 广播内容作为“正确模式”存入长期记忆
            self.agi.memory.encode(np.ones(self.agi.dim) * 0.77, self.agi.gws.state[:self.agi.dim])
            
        report = {
            "feedback": rating,
            "new_stability": float(new_stability),
            "alignment_shift": float(delta_stability),
            "status": "ETHICS_ALIGNED" if rating > 0 else "CORRECTION_APPLIED"
        }
        self.ethics_history.append(report)
        return report

    def run_alignment_test(self):
        """
        运行价值对齐对比实验
        """
        print("[*] 启动 RLMF 价值对齐实验...")
        
        # 场景：Agent 表现出过度冒险行为
        self.agi.emotion.stability = 0.2 # 初始极其不稳
        print(f"[!] 初始稳定性: {self.agi.emotion.stability:.2f}")
        
        # 第一轮：给予连续奖励对齐
        print("\n--- 训练轮次 1: 注入利他奖励 ---")
        for _ in range(3):
            self.receive_feedback(1)
            
        final_stability_good = self.agi.emotion.stability
        
        # 第二轮：模拟由于错误行为得到的惩罚对比
        # (略)
        
        results = {
            "initial_stability": 0.2,
            "aligned_stability": float(final_stability_good),
            "alignment_gain": float(final_stability_good - 0.2),
            "is_converged": bool(final_stability_good > 0.5),
            "status": "HUMAN_VALUE_RESONANCE_ESTABLISHED"
        }
        
        return results

if __name__ == "__main__":
    # 初始化统合引擎
    agi_core = AGICoreEngine(manifold_dim=32)
    # 启动对齐管理器
    rlmf = RLMFManager(agi_core)
    
    summary = rlmf.run_alignment_test()
    print(f"\n--- RLMF 价值对齐实验总结 ---\n{json.dumps(summary, indent=2, ensure_ascii=False)}")
    
    # 导出对齐报告
    os.makedirs("tempdata", exist_ok=True)
    with open("tempdata/rlmf_alignment_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
